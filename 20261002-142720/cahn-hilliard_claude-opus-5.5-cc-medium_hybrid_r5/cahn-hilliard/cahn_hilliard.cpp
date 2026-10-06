// Hybrid MPI + OpenMP + CUDA Cahn-Hilliard benchmark.
//
// Parallelization strategy:
//  - MPI: 1D slab decomposition along z (contiguous planes in memory). Each rank
//    owns L consecutive z-planes and keeps a halo of 2 planes on each side of the
//    concentration field (the scheme needs c at distance 2 to update a cell).
//    Only one halo exchange per time step is needed; it is overlapped with the
//    interior computation on the GPU.
//  - CUDA: each rank drives one GPU (round-robin over the node-local GPUs). Two
//    stencil kernels (chemical potential, update) march along z keeping the
//    z-column in registers.
//  - OpenMP: host-side work (initialization of the local slab, validation
//    reductions).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call)                                                                       \
    do {                                                                                       \
        cudaError_t err_ = (call);                                                             \
        if (err_ != cudaSuccess) {                                                             \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__,    \
                    __LINE__);                                                                 \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                      \
        }                                                                                      \
    } while (0)

// Physical parameters (compile-time constants so divisions by dx*dx etc. fold exactly)
constexpr double dx = 1.0;
constexpr double dy = 1.0;
constexpr double dz = 1.0;
constexpr double dt = 0.01;
constexpr double e_AA = -(2.0 / 9.0);
constexpr double e_BB = -(2.0 / 9.0);
constexpr double e_AB = (2.0 / 9.0);
constexpr double gamma_ = 0.5;
constexpr double D = 1.0;

constexpr int HALO = 2;
constexpr int BX = 32;
constexpr int BY = 8;

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Clamped 7-point Laplacian; z-neighbours are passed in from registers.
__device__ __forceinline__ double laplacian(const double* __restrict__ p, const int off, const int xp,
                                            const int xn, const int yp, const int yn, const double cc,
                                            const double cp, const double cm) {
    const double cxx = (p[off + xp] + p[off + xn] - 2.0 * cc) / (dx * dx);
    const double cyy = (p[off + yp] + p[off + yn] - 2.0 * cc) / (dy * dy);
    const double czz = (cp + cm - 2.0 * cc) / (dz * dz);
    return cxx + cyy + czz;
}

// Kernel arguments: arrays point to local plane 0; z is local, gz0 is the global
// index of local plane 0, nz is the global z size (for clamped boundaries).
template <bool UPDATE>
__global__ void __launch_bounds__(BX * BY)
stencilKernel(const double* __restrict__ in, const double* __restrict__ cold, double* __restrict__ out,
              const int nx, const int ny, const int zbeg, const int zend, const int zchunk,
              const int gz0, const int nz) {
    const int x = blockIdx.x * BX + threadIdx.x;
    const int y = blockIdx.y * BY + threadIdx.y;
    if (x >= nx || y >= ny) return;
    const int zs = zbeg + blockIdx.z * zchunk;
    const int ze = min(zend, zs + zchunk);
    if (zs >= ze) return;

    const ptrdiff_t plane = (ptrdiff_t)nx * ny;
    const int off = y * nx + x;
    const int xp = (x < nx - 1) ? 1 : 0;
    const int xn = (x > 0) ? -1 : 0;
    const int yp = (y < ny - 1) ? nx : 0;
    const int yn = (y > 0) ? -nx : 0;

    const int zm0 = (gz0 + zs > 0) ? zs - 1 : zs;
    const int zp0 = (gz0 + zs < nz - 1) ? zs + 1 : zs;
    double cm = in[zm0 * plane + off];
    double cc = in[zs * plane + off];
    double cp = in[zp0 * plane + off];

    for (int z = zs;;) {
        const double* p = in + z * plane;
        const double lap = laplacian(p, off, xp, xn, yp, yn, cc, cp, cm);
        const ptrdiff_t idx = z * plane + off;
        if constexpr (UPDATE) {
            out[idx] = cold[idx] + dt * D * lap;
        } else {
            const double cv = cc;
            out[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                     + 3.0 * cv + cv * cv * cv
                     - gamma_ * lap;
        }
        if (++z >= ze) break;
        cm = cc;
        cc = cp;
        const int zpn = (gz0 + z < nz - 1) ? z + 1 : z;
        cp = in[zpn * plane + off];
    }
}

struct Launcher {
    int nx, ny, nz, gz0, numSMs;
    cudaStream_t stream;

    template <bool UPDATE>
    void run(const double* in, const double* cold, double* out, int zbeg, int zend) const {
        if (zend <= zbeg) return;
        const int range = zend - zbeg;
        dim3 block(BX, BY, 1);
        const int gx = (nx + BX - 1) / BX;
        const int gy = (ny + BY - 1) / BY;
        const long long xyBlocks = (long long)gx * gy;
        const long long wantBlocks = (long long)numSMs * 16;
        int nzb = (int)std::max<long long>(1, (wantBlocks + xyBlocks - 1) / xyBlocks);
        nzb = std::min(nzb, range);
        int zchunk = (range + nzb - 1) / nzb;
        zchunk = std::max(zchunk, 1);
        nzb = (range + zchunk - 1) / zchunk;
        dim3 grid(gx, gy, nzb);
        stencilKernel<UPDATE><<<grid, block, 0, stream>>>(in, cold, out, nx, ny, zbeg, zend, zchunk, gz0, nz);
    }
};

// Initialize concentration field for global planes [gzBeg, gzEnd) into buffer whose
// plane 0 corresponds to global plane gzBase.
void initializeConcentration(double* c, const size_t nx, const size_t ny, const size_t nz,
                             const long gzBeg, const long gzEnd, const long gzBase) {
    const size_t vol = nx * ny * nz;
    #pragma omp parallel for collapse(2) schedule(static)
    for (long z = gzBeg; z < gzEnd; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            double* row = c + (size_t)(z - gzBase) * nx * ny + y * nx;
            for (size_t x = 0; x < nx; ++x) {
                // Generate pseudo-random value in [-1, 1]
                const size_t linear_id = idx3(x, y, (size_t)z, nx, ny);
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                row[x] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

bool validateResult(const double* c, const size_t n, MPI_Comm comm, const int rank) {
    int bad = 0;
    double minVal = INFINITY, maxVal = -INFINITY;
    #pragma omp parallel for reduction(|| : bad) reduction(min : minVal) reduction(max : maxVal)
    for (size_t i = 0; i < n; ++i) {
        const double val = c[i];
        if (std::isnan(val) || std::isinf(val)) bad = 1;
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }
    MPI_Allreduce(MPI_IN_PLACE, &bad, 1, MPI_INT, MPI_LOR, comm);
    MPI_Allreduce(MPI_IN_PLACE, &minVal, 1, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(MPI_IN_PLACE, &maxVal, 1, MPI_DOUBLE, MPI_MAX, comm);

    // Check for NaN or Inf
    if (bad) {
        if (rank == 0) printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    if (rank == 0) printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);

    // Values should generally stay within reasonable bounds
    if (maxVal > 10.0 || minVal < -10.0) {
        if (rank == 0) printf("Validation failed: values out of expected range\n");
        return false;
    }

    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -x <num>     Grid size in X dimension (default: 64)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of time steps (default: 20)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int worldRank = 0, worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            nx = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            ny = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            nz = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (worldRank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (worldRank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    if (worldRank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const size_t gridSize = nx * ny * nz;
    const size_t plane = nx * ny;

    // Each active rank needs at least HALO planes so that halos come from direct neighbours.
    const int nActive = (int)std::max<size_t>(1, std::min<size_t>((size_t)worldSize, nz / HALO));
    const bool active = worldRank < nActive;
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, active ? 0 : MPI_UNDEFINED, worldRank, &comm);
    if (!active) {
        MPI_Finalize();
        return 0;
    }
    int rank = 0, size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Slab decomposition along z
    std::vector<int> planesOf(size), firstOf(size);
    for (int r = 0, acc = 0; r < size; ++r) {
        planesOf[r] = (int)(nz / size) + (r < (int)(nz % size) ? 1 : 0);
        firstOf[r] = acc;
        acc += planesOf[r];
    }
    const int L = planesOf[rank];
    const int gz0 = firstOf[rank];
    const int lower = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    const int upper = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;

    // GPU selection: round-robin over node-local GPUs
    MPI_Comm nodeComm;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_free(&nodeComm);
    int nDev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&nDev));
    if (nDev == 0) {
        fprintf(stderr, "No CUDA device found\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % nDev));
    int numSMs = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, localRank % nDev));

    // Local buffers including halos
    const size_t localPlanes = (size_t)L + 2 * HALO;
    const size_t localElems = localPlanes * plane;
    const size_t haloElems = (size_t)HALO * plane;

    double* hostC = nullptr;
    CUDA_CHECK(cudaMallocHost(&hostC, localElems * sizeof(double)));

    // Initialize concentration field (owned planes plus halos, directly from the formula)
    if (rank == 0) printf("Initializing concentration field...\n");
    {
        const long gzBase = (long)gz0 - HALO;
        const long b = std::max<long>(0, gzBase);
        const long e = std::min<long>((long)nz, (long)gz0 + L + HALO);
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < localElems; ++i) hostC[i] = 0.0;
        initializeConcentration(hostC, nx, ny, nz, b, e, gzBase);
    }

    double *dA = nullptr, *dB = nullptr, *dMu = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, localElems * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dB, localElems * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dMu, localElems * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(dA, hostC, localElems * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dB, 0, localElems * sizeof(double)));
    CUDA_CHECK(cudaMemset(dMu, 0, localElems * sizeof(double)));

    // Pinned staging buffers for halo exchange
    double *sendLo, *sendHi, *recvLo, *recvHi;
    CUDA_CHECK(cudaMallocHost(&sendLo, haloElems * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&sendHi, haloElems * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&recvLo, haloElems * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&recvHi, haloElems * sizeof(double)));

    cudaStream_t sMain, sComm;
    CUDA_CHECK(cudaStreamCreateWithFlags(&sMain, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&sComm, cudaStreamNonBlocking));
    cudaEvent_t evEdges, evHalo;
    CUDA_CHECK(cudaEventCreateWithFlags(&evEdges, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&evHalo, cudaEventDisableTiming));

    Launcher launch{(int)nx, (int)ny, (int)nz, gz0, numSMs, sMain};

    // Valid local plane ranges: mu is needed on [muLo, muHi), c is updated on [0, L)
    const int muLo = (lower != MPI_PROC_NULL) ? -1 : 0;
    const int muHi = (upper != MPI_PROC_NULL) ? L + 1 : L;
    // Interior parts that do not depend on halo data
    const int muIntLo = 1, muIntHi = std::max(1, L - 1);
    const int cIntLo = std::min(HALO, L), cIntHi = std::max(cIntLo, L - HALO);

    auto off = [&](double* base) { return base + haloElems; };  // pointer to local plane 0
    const bool haveNeighbours = (size > 1);

    // Run simulation
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(comm);
    auto start = std::chrono::high_resolution_clock::now();

    double* cold = dA;
    double* cnew = dB;
    double* mu = off(dMu);

    // Prologue: interior chemical potential of the initial field
    launch.run<false>(off(cold), nullptr, mu, muIntLo, muIntHi);

    for (int t = 0; t < iterations; ++t) {
        double* c = off(cold);
        double* cn = off(cnew);

        // Edge parts (depend on halo planes of c)
        launch.run<false>(c, nullptr, mu, muLo, muIntLo);
        launch.run<false>(c, nullptr, mu, std::max(muIntLo, muIntHi), muHi);
        launch.run<true>(mu, c, cn, 0, cIntLo);
        launch.run<true>(mu, c, cn, cIntHi, L);

        if (haveNeighbours) {
            CUDA_CHECK(cudaEventRecord(evEdges, sMain));
            CUDA_CHECK(cudaStreamWaitEvent(sComm, evEdges, 0));
            if (lower != MPI_PROC_NULL)
                CUDA_CHECK(cudaMemcpyAsync(sendLo, cn, haloElems * sizeof(double), cudaMemcpyDeviceToHost, sComm));
            if (upper != MPI_PROC_NULL)
                CUDA_CHECK(cudaMemcpyAsync(sendHi, cn + (size_t)(L - HALO) * plane, haloElems * sizeof(double),
                                           cudaMemcpyDeviceToHost, sComm));
        }

        // Interior update and next step's interior chemical potential (overlap with exchange)
        launch.run<true>(mu, c, cn, cIntLo, cIntHi);
        if (t + 1 < iterations) launch.run<false>(cn, nullptr, mu, muIntLo, muIntHi);

        if (haveNeighbours) {
            CUDA_CHECK(cudaStreamSynchronize(sComm));
            MPI_Request req[4];
            const int cnt = (int)haloElems;
            MPI_Irecv(recvLo, cnt, MPI_DOUBLE, lower, 0, comm, &req[0]);
            MPI_Irecv(recvHi, cnt, MPI_DOUBLE, upper, 1, comm, &req[1]);
            MPI_Isend(sendLo, cnt, MPI_DOUBLE, lower, 1, comm, &req[2]);
            MPI_Isend(sendHi, cnt, MPI_DOUBLE, upper, 0, comm, &req[3]);
            MPI_Waitall(4, req, MPI_STATUSES_IGNORE);
            if (lower != MPI_PROC_NULL)
                CUDA_CHECK(cudaMemcpyAsync(cnew, recvLo, haloElems * sizeof(double), cudaMemcpyHostToDevice, sComm));
            if (upper != MPI_PROC_NULL)
                CUDA_CHECK(cudaMemcpyAsync(cn + (size_t)L * plane, recvHi, haloElems * sizeof(double),
                                           cudaMemcpyHostToDevice, sComm));
            CUDA_CHECK(cudaEventRecord(evHalo, sComm));
            CUDA_CHECK(cudaStreamWaitEvent(sMain, evHalo, 0));
        }

        // Swap buffers
        std::swap(cold, cnew);
    }

    CUDA_CHECK(cudaStreamSynchronize(sMain));
    CUDA_CHECK(cudaGetLastError());
    MPI_Barrier(comm);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long localElapsedMs = duration.count();
    long maxElapsedMs = 0;
    MPI_Reduce(&localElapsedMs, &maxElapsedMs, 1, MPI_LONG, MPI_MAX, 0, comm);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", maxElapsedMs);

        // Calculate performance
        double cellUpdates = (double)gridSize * iterations;
        double mcups = cellUpdates / (maxElapsedMs / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Bring owned planes back to the host
    double* ownedHost = hostC + haloElems;
    const size_t ownedElems = (size_t)L * plane;
    CUDA_CHECK(cudaMemcpy(ownedHost, off(cold), ownedElems * sizeof(double), cudaMemcpyDeviceToHost));

    // Print results for external validation
    if (printResults) {
        MPI_Datatype planeType;
        MPI_Type_contiguous((int)plane, MPI_DOUBLE, &planeType);
        MPI_Type_commit(&planeType);
        std::vector<double> full;
        if (rank == 0) full.resize(gridSize);
        MPI_Gatherv(ownedHost, L, planeType, rank == 0 ? full.data() : nullptr, planesOf.data(), firstOf.data(),
                    planeType, 0, comm);
        MPI_Type_free(&planeType);
        if (rank == 0) print_results(full, "Concentration");
    }

    int ret = 0;
    // Validation
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        bool valid = validateResult(ownedHost, ownedElems, comm, rank);
        if (rank == 0) printf(valid ? "Validation: PASSED\n" : "Validation: FAILED\n");
        ret = valid ? 0 : 1;
    }

    cudaEventDestroy(evEdges);
    cudaEventDestroy(evHalo);
    cudaStreamDestroy(sMain);
    cudaStreamDestroy(sComm);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dMu);
    cudaFreeHost(hostC);
    cudaFreeHost(sendLo);
    cudaFreeHost(sendHi);
    cudaFreeHost(recvLo);
    cudaFreeHost(recvHi);
    MPI_Comm_free(&comm);
    MPI_Finalize();
    return ret;
}
