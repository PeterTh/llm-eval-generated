// Hybrid MPI + OpenMP + CUDA Cahn-Hilliard benchmark.
//
// Parallelization:
//  * MPI: 1D slab decomposition along z; one rank per GPU. Each step exchanges a
//    two-plane halo of the concentration field (depth 2 because the update is a
//    Laplacian of mu, which itself is a Laplacian of c).
//  * CUDA: a single fused kernel per step computes mu into shared memory (2.5D
//    z-marching tiles) and immediately applies the update, so mu never touches
//    global memory. Boundary slabs are computed first on a separate stream so
//    the halo exchange overlaps with the interior computation.
//  * OpenMP: host-side work (initialization, validation reductions).
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

#define CUDA_CHECK(call)                                                                          \
    do {                                                                                          \
        cudaError_t err_ = (call);                                                                \
        if (err_ != cudaSuccess) {                                                                \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__, __LINE__); \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                         \
        }                                                                                         \
    } while (0)

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

constexpr int HALO = 2;      // halo planes on each side of the local slab
constexpr int BX = 32;       // output tile width (x)
constexpr int BY = 16;       // output tile height (y)
constexpr int TX = BX + 2;   // mu tile width incl. halo
constexpr int TY = BY + 2;   // mu tile height incl. halo
constexpr int NSLOT = 4;     // rolling mu planes in shared memory
constexpr int ZCHUNK = 32;   // output planes per block along z

struct Params {
    int nx, ny, nz;          // global sizes
    int z0;                  // global z of first owned plane
    size_t plane;            // nx * ny
    double dx2, dy2, dz2;    // squared spacings (used for division)
    double idx2, idy2, idz2; // reciprocals (used only when bit-identical to division)
    double gamma, e_AA, e_BB, e_AB;
    double dtD_dt, dtD_D;
    bool exactRecip;         // all of dx2, dy2, dz2 are powers of two    // dt and D (kept separate for identical rounding)
};

// a / d, computed as a * (1/d) when that is exactly equivalent (d a power of two).
// FP64 division is very expensive on GPUs, so this matters for throughput.
template <bool EXACT>
__device__ __forceinline__ double scaleDiv(const double a, const double d, const double inv) {
    if constexpr (EXACT) {
        return a * inv;
    } else {
        return a / d;
    }
}

// Chemical potential at global (x, y, gz), all clamped into the domain.
// c points to the local slab, plane index 0 corresponds to global z0 - HALO.
template <bool EXACT>
__device__ __forceinline__ double chemPot(const double* __restrict__ c, const Params& p, int x, int y, int gz) {
    const int xp = (x < p.nx - 1) ? x + 1 : x;
    const int yp = (y < p.ny - 1) ? y + 1 : y;
    const int zp = (gz < p.nz - 1) ? gz + 1 : gz;
    const int xn = (x > 0) ? x - 1 : 0;
    const int yn = (y > 0) ? y - 1 : 0;
    const int zn = (gz > 0) ? gz - 1 : 0;
    const size_t row = (size_t)y * p.nx;
    const size_t pl = (size_t)(gz - p.z0 + HALO) * p.plane;
    const double* cz = c + pl;
    const double cv = __ldg(cz + row + x);
    const double cxx = scaleDiv<EXACT>(__ldg(cz + row + xp) + __ldg(cz + row + xn) - 2.0 * cv, p.dx2, p.idx2);
    const double cyy = scaleDiv<EXACT>(__ldg(cz + (size_t)yp * p.nx + x) + __ldg(cz + (size_t)yn * p.nx + x) - 2.0 * cv, p.dy2, p.idy2);
    const double czz = scaleDiv<EXACT>(__ldg(c + (size_t)(zp - p.z0 + HALO) * p.plane + row + x) +
                        __ldg(c + (size_t)(zn - p.z0 + HALO) * p.plane + row + x) - 2.0 * cv, p.dz2, p.idz2);
    const double lap = cxx + cyy + czz;
    return 4.5 * ((cv + 1.0) * p.e_AA + (cv - 1.0) * p.e_BB - 2.0 * cv * p.e_AB)
           + 3.0 * cv + cv * cv * cv
           - p.gamma * lap;
}

// Fused mu + update kernel. Each block handles a BX x BY column over ZCHUNK
// planes of the local output range [zbeg, zend) (local owned indices).
template <bool EXACT>
__global__ void __launch_bounds__(BX * BY)
cahnHilliardStep(const double* __restrict__ cold, double* __restrict__ cnew, const Params p, int zbeg, int zend) {
    __shared__ double smu[NSLOT][TY][TX];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * BX + tx;
    const int x0 = blockIdx.x * BX;
    const int y0 = blockIdx.y * BY;
    const int zs = zbeg + blockIdx.z * ZCHUNK;
    if (zs >= zend) return;
    const int ze = min(zs + ZCHUNK, zend);

    const int gx = x0 + tx, gy = y0 + ty;
    const bool active = (gx < p.nx) && (gy < p.ny);

    // Fill a mu plane (global plane gz, clamped) into slot s, incl. 1-cell xy halo.
    auto fill = [&](int gz, int s) {
        gz = min(max(gz, 0), p.nz - 1);
        for (int i = tid; i < TX * TY; i += BX * BY) {
            const int ly = i / TX, lx = i - ly * TX;
            const int x = min(max(x0 + lx - 1, 0), p.nx - 1);
            const int y = min(max(y0 + ly - 1, 0), p.ny - 1);
            smu[s][ly][lx] = chemPot<EXACT>(cold, p, x, y, gz);
        }
    };

    // Local z -> global z: gz = z + z0
    fill(p.z0 + zs - 1, (zs - 1 + NSLOT) % NSLOT);
    fill(p.z0 + zs, zs % NSLOT);

    for (int z = zs; z < ze; ++z) {
        fill(p.z0 + z + 1, (z + 1) % NSLOT);
        __syncthreads();
        if (active) {
            const int sm = (z - 1 + NSLOT) % NSLOT, s0 = z % NSLOT, sp = (z + 1) % NSLOT;
            const int lx = tx + 1, ly = ty + 1;
            const double m = smu[s0][ly][lx];
            // Clamped neighbors at the global boundary coincide with the centre value,
            // which is exactly what the clamped tile halo holds.
            const double mxx = scaleDiv<EXACT>(smu[s0][ly][lx + 1] + smu[s0][ly][lx - 1] - 2.0 * m, p.dx2, p.idx2);
            const double myy = scaleDiv<EXACT>(smu[s0][ly + 1][lx] + smu[s0][ly - 1][lx] - 2.0 * m, p.dy2, p.idy2);
            const double mzz = scaleDiv<EXACT>(smu[sp][ly][lx] + smu[sm][ly][lx] - 2.0 * m, p.dz2, p.idz2);
            const double lap = mxx + myy + mzz;
            const size_t idx = (size_t)(z + HALO) * p.plane + (size_t)gy * p.nx + gx;
            cnew[idx] = cold[idx] + p.dtD_dt * p.dtD_D * lap;
        }
    }
}

static void launchStep(const double* cold, double* cnew, const Params& p, int zbeg, int zend, cudaStream_t s) {
    if (zend <= zbeg) return;
    dim3 block(BX, BY, 1);
    dim3 grid((p.nx + BX - 1) / BX, (p.ny + BY - 1) / BY, (zend - zbeg + ZCHUNK - 1) / ZCHUNK);
    if (p.exactRecip) {
        cahnHilliardStep<true><<<grid, block, 0, s>>>(cold, cnew, p, zbeg, zend);
    } else {
        cahnHilliardStep<false><<<grid, block, 0, s>>>(cold, cnew, p, zbeg, zend);
    }
}

// Initialize the owned part (global planes [z0, z0+lnz)) of the concentration field
void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz,
                             const size_t z0, const size_t lnz) {
    const size_t vol = nx * ny * nz;
#pragma omp parallel for collapse(2) schedule(static)
    for (size_t lz = 0; lz < lnz; ++lz) {
        for (size_t y = 0; y < ny; ++y) {
            const size_t z = z0 + lz;
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, lz, nx, ny);
                // Generate pseudo-random value in [-1, 1]
                const size_t linear_id = z * (nx * ny) + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[idx] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

// Distributed validation: each rank reduces its slab with OpenMP, then MPI reduces.
bool validateResult(const double* c, const size_t n, MPI_Comm comm, int rank) {
    int bad = 0;
    double minVal = INFINITY, maxVal = -INFINITY;
#pragma omp parallel for reduction(| : bad) reduction(min : minVal) reduction(max : maxVal) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        const double val = c[i];
        if (std::isnan(val) || std::isinf(val)) bad = 1;
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }
    int gbad = 0;
    double gmin = 0.0, gmax = 0.0;
    MPI_Allreduce(&bad, &gbad, 1, MPI_INT, MPI_LOR, comm);
    MPI_Allreduce(&minVal, &gmin, 1, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(&maxVal, &gmax, 1, MPI_DOUBLE, MPI_MAX, comm);

    if (gbad) {
        if (rank == 0) printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    if (rank == 0) printf("Concentration range: [%.6f, %.6f]\n", gmin, gmax);

    // Values should generally stay within reasonable bounds
    if (gmax > 10.0 || gmin < -10.0) {
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
    MPI_Init(&argc, &argv);
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

    // Physical parameters
    const double dx = 1.0;
    const double dy = 1.0;
    const double dz = 1.0;
    const double dt = 0.01;
    const double e_AA = -(2.0 / 9.0);
    const double e_BB = -(2.0 / 9.0);
    const double e_AB = (2.0 / 9.0);
    const double gamma = 0.5;
    const double D = 1.0;

    const size_t gridSize = nx * ny * nz;
    const size_t plane = nx * ny;

    // Every rank must own at least HALO planes; surplus ranks stay idle.
    const int nActive = (int)std::max<size_t>(1, std::min<size_t>((size_t)worldSize, nz / HALO));
    const bool isActive = worldRank < nActive;
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, isActive ? 0 : MPI_UNDEFINED, worldRank, &comm);

    if (!isActive) {
        // Idle ranks only take part in the world-level barriers used for timing.
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        int rc = 0;
        MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Finalize();
        return rc;
    }

    int rank = 0, size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Select GPU by node-local rank
    {
        MPI_Comm nodeComm;
        MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
        int localRank = 0;
        MPI_Comm_rank(nodeComm, &localRank);
        MPI_Comm_free(&nodeComm);
        int ndev = 0;
        CUDA_CHECK(cudaGetDeviceCount(&ndev));
        if (ndev < 1) {
            fprintf(stderr, "No CUDA device available\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        CUDA_CHECK(cudaSetDevice(localRank % ndev));
    }

    // Slab decomposition along z
    std::vector<int> zStart(size + 1);
    for (int r = 0; r <= size; ++r) zStart[r] = (int)(((size_t)r * nz) / size);
    const int z0 = zStart[rank];
    const int lnz = zStart[rank + 1] - z0;
    const int up = (rank + 1 < size) ? rank + 1 : MPI_PROC_NULL;   // higher z
    const int down = (rank > 0) ? rank - 1 : MPI_PROC_NULL;        // lower z

    const size_t localCount = (size_t)lnz * plane;
    const size_t storeCount = (size_t)(lnz + 2 * HALO) * plane;
    const size_t haloCount = (size_t)HALO * plane;

    // Host: initialize owned planes
    std::vector<double> hostC(localCount);
    if (rank == 0) printf("Initializing concentration field...\n");
    initializeConcentration(hostC, nx, ny, nz, z0, lnz);

    // Device buffers (local slab with HALO planes on each side)
    double *dOld = nullptr, *dNew = nullptr;
    CUDA_CHECK(cudaMalloc(&dOld, storeCount * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dNew, storeCount * sizeof(double)));
    CUDA_CHECK(cudaMemset(dOld, 0, storeCount * sizeof(double)));
    CUDA_CHECK(cudaMemset(dNew, 0, storeCount * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(dOld + haloCount, hostC.data(), localCount * sizeof(double), cudaMemcpyHostToDevice));

    // Pinned staging buffers for halo exchange: [send low | send high], [recv low | recv high] x 2
    // (receive side is double-buffered: the async upload of the previous exchange may still be
    // in flight when the next receive is posted)
    double *hSend = nullptr, *hRecv = nullptr;
    CUDA_CHECK(cudaMallocHost(&hSend, 2 * haloCount * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&hRecv, 4 * haloCount * sizeof(double)));

    cudaStream_t sEdge, sInner;
    {
        // Edge planes gate the halo exchange: give them scheduling priority
        int prLeast = 0, prGreatest = 0;
        CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&prLeast, &prGreatest));
        CUDA_CHECK(cudaStreamCreateWithPriority(&sEdge, cudaStreamNonBlocking, prGreatest));
        CUDA_CHECK(cudaStreamCreateWithPriority(&sInner, cudaStreamNonBlocking, prLeast));
    }

    Params p;
    p.nx = (int)nx; p.ny = (int)ny; p.nz = (int)nz;
    p.z0 = z0;
    p.plane = plane;
    p.dx2 = dx * dx;
    p.dy2 = dy * dy;
    p.dz2 = dz * dz;
    p.idx2 = 1.0 / p.dx2;
    p.idy2 = 1.0 / p.dy2;
    p.idz2 = 1.0 / p.dz2;
    auto isPow2 = [](double v) { int e; return v > 0.0 && std::isfinite(v) && std::frexp(v, &e) == 0.5; };
    p.exactRecip = isPow2(p.dx2) && isPow2(p.dy2) && isPow2(p.dz2);
    p.gamma = gamma; p.e_AA = e_AA; p.e_BB = e_BB; p.e_AB = e_AB;
    p.dtD_dt = dt; p.dtD_D = D;

    // Planes of the owned range that neighbors need (computed first)
    const bool hasDown = down != MPI_PROC_NULL, hasUp = up != MPI_PROC_NULL;
    const int edgeLo = hasDown ? HALO : 0;            // [0, edgeLo) is a send region
    const int edgeHi = hasUp ? lnz - HALO : lnz;      // [edgeHi, lnz) is a send region
    const bool needExchange = hasDown || hasUp;

    int recvParity = 0;
    cudaEvent_t edgeDone, innerDone;
    CUDA_CHECK(cudaEventCreateWithFlags(&edgeDone, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&innerDone, cudaEventDisableTiming));

    // Exchange halo of a field on device: post receives, stage sends, wait, upload.
    auto exchangeHalo = [&](double* d, bool alreadyStagedOnEdge) {
        if (!needExchange) return;
        double* hR = hRecv + (size_t)recvParity * 2 * haloCount;
        recvParity ^= 1;
        MPI_Request req[4];
        int nreq = 0;
        if (hasDown) MPI_Irecv(hR, (int)haloCount, MPI_DOUBLE, down, 1, comm, &req[nreq++]);
        if (hasUp) MPI_Irecv(hR + haloCount, (int)haloCount, MPI_DOUBLE, up, 0, comm, &req[nreq++]);
        if (!alreadyStagedOnEdge) {
            if (hasDown)
                CUDA_CHECK(cudaMemcpyAsync(hSend, d + haloCount, haloCount * sizeof(double), cudaMemcpyDeviceToHost, sEdge));
            if (hasUp)
                CUDA_CHECK(cudaMemcpyAsync(hSend + haloCount, d + (size_t)lnz * plane, haloCount * sizeof(double),
                                           cudaMemcpyDeviceToHost, sEdge));
        }
        CUDA_CHECK(cudaStreamSynchronize(sEdge));
        if (hasDown) MPI_Isend(hSend, (int)haloCount, MPI_DOUBLE, down, 0, comm, &req[nreq++]);
        if (hasUp) MPI_Isend(hSend + haloCount, (int)haloCount, MPI_DOUBLE, up, 1, comm, &req[nreq++]);
        MPI_Waitall(nreq, req, MPI_STATUSES_IGNORE);
        if (hasDown)
            CUDA_CHECK(cudaMemcpyAsync(d, hR, haloCount * sizeof(double), cudaMemcpyHostToDevice, sEdge));
        if (hasUp)
            CUDA_CHECK(cudaMemcpyAsync(d + (size_t)(lnz + HALO) * plane, hR + haloCount, haloCount * sizeof(double),
                                       cudaMemcpyHostToDevice, sEdge));
    };

    // Initial halo fill
    exchangeHalo(dOld, false);
    CUDA_CHECK(cudaStreamSynchronize(sEdge));
    CUDA_CHECK(cudaEventRecord(edgeDone, sEdge));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Run simulation
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < iterations; ++t) {
        // The interior only reads owned planes of dOld (never its halo), so it just has
        // to wait for the previous step's edge kernels (recorded in edgeDone), not for
        // the halo upload. Edge kernels follow the upload in stream order.
        CUDA_CHECK(cudaStreamWaitEvent(sInner, edgeDone, 0));

        if (needExchange) {
            // Edge planes first, then stage them to host on the same stream
            launchStep(dOld, dNew, p, 0, edgeLo, sEdge);
            launchStep(dOld, dNew, p, std::max(edgeHi, edgeLo), lnz, sEdge);
            CUDA_CHECK(cudaEventRecord(edgeDone, sEdge));
            if (hasDown)
                CUDA_CHECK(cudaMemcpyAsync(hSend, dNew + haloCount, haloCount * sizeof(double),
                                           cudaMemcpyDeviceToHost, sEdge));
            if (hasUp)
                CUDA_CHECK(cudaMemcpyAsync(hSend + haloCount, dNew + (size_t)lnz * plane, haloCount * sizeof(double),
                                           cudaMemcpyDeviceToHost, sEdge));
            // Interior concurrently
            launchStep(dOld, dNew, p, edgeLo, edgeHi, sInner);
            exchangeHalo(dNew, true);
        } else {
            launchStep(dOld, dNew, p, 0, lnz, sInner);
            CUDA_CHECK(cudaEventRecord(edgeDone, sInner));
        }

        // Next step's kernels read dNew; make sure all writes are complete.
        CUDA_CHECK(cudaEventRecord(innerDone, sInner));
        CUDA_CHECK(cudaStreamWaitEvent(sEdge, innerDone, 0));

        // Swap buffers
        std::swap(dOld, dNew);
    }
    CUDA_CHECK(cudaStreamSynchronize(sEdge));
    CUDA_CHECK(cudaStreamSynchronize(sInner));
    CUDA_CHECK(cudaGetLastError());
    MPI_Barrier(MPI_COMM_WORLD);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate performance
        double cellUpdates = (double)gridSize * iterations;
        double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Bring owned planes back to host
    CUDA_CHECK(cudaMemcpy(hostC.data(), dOld + haloCount, localCount * sizeof(double), cudaMemcpyDeviceToHost));

    // Print results for external validation
    if (printResults) {
        MPI_Datatype planeType;
        MPI_Type_contiguous((int)plane, MPI_DOUBLE, &planeType);
        MPI_Type_commit(&planeType);
        if (rank == 0) {
            std::vector<double> full(gridSize);
            std::copy(hostC.begin(), hostC.end(), full.begin());
            for (int r = 1; r < size; ++r) {
                MPI_Recv(full.data() + (size_t)zStart[r] * plane, zStart[r + 1] - zStart[r], planeType, r, 2, comm,
                         MPI_STATUS_IGNORE);
            }
            print_results(full, "Concentration");
        } else {
            MPI_Send(hostC.data(), lnz, planeType, 0, 2, comm);
        }
        MPI_Type_free(&planeType);
    }

    int rc = 0;
    // Validation
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        bool valid = validateResult(hostC.data(), localCount, comm, rank);

        if (rank == 0) printf(valid ? "Validation: PASSED\n" : "Validation: FAILED\n");
        rc = valid ? 0 : 1;
    }

    CUDA_CHECK(cudaEventDestroy(edgeDone));
    CUDA_CHECK(cudaEventDestroy(innerDone));
    CUDA_CHECK(cudaStreamDestroy(sEdge));
    CUDA_CHECK(cudaStreamDestroy(sInner));
    CUDA_CHECK(cudaFreeHost(hSend));
    CUDA_CHECK(cudaFreeHost(hRecv));
    CUDA_CHECK(cudaFree(dOld));
    CUDA_CHECK(cudaFree(dNew));

    MPI_Comm_free(&comm);
    MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return rc;
}
