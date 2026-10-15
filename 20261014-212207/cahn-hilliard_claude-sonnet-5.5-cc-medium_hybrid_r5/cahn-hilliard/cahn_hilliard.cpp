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

#define CUDA_CHECK(call)                                                                   \
    do {                                                                                   \
        cudaError_t err_ = (call);                                                         \
        if (err_ != cudaSuccess) {                                                         \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__, __LINE__); \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                  \
        }                                                                                  \
    } while (0)

// Physical parameters (compile-time constants so the device code can fold them)
static constexpr double dx = 1.0;
static constexpr double dy = 1.0;
static constexpr double dz = 1.0;
static constexpr double dt = 0.01;
static constexpr double e_AA = -(2.0 / 9.0);
static constexpr double e_BB = -(2.0 / 9.0);
static constexpr double e_AB = (2.0 / 9.0);
static constexpr double gamma_ = 0.5;
static constexpr double D = 1.0;

// Domain decomposition: 1D slabs along Z, one MPI rank per GPU.
// Local storage has HALO ghost planes on each side. Local plane index l = (global z) - z0 + HALO.
static constexpr int HALO = 2;

struct Slab {
    size_t nx, ny, nz;  // global dims
    size_t z0, nzl;     // first owned global plane, number of owned planes
};

constexpr int BX = 32;
constexpr int BY = 4;

// Laplacian with clamped boundary conditions. zp/zn are local plane indices (already clamped).
__device__ __forceinline__ double laplacian(const double* __restrict__ f, const size_t nx, const size_t ny,
                                            const size_t x, const size_t y, const size_t l,
                                            const size_t lp, const size_t ln) {
    const size_t nxy = nx * ny;
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;
    const size_t base = l * nxy;
    const double ctr = f[base + y * nx + x];
    const double cxx = (f[base + y * nx + xp] + f[base + y * nx + xn] - 2.0 * ctr) / (dx * dx);
    const double cyy = (f[base + yp * nx + x] + f[base + yn * nx + x] - 2.0 * ctr) / (dy * dy);
    const double czz = (f[lp * nxy + y * nx + x] + f[ln * nxy + y * nx + x] - 2.0 * ctr) / (dz * dz);
    return cxx + cyy + czz;
}

// Chemical potential on local planes [l_lo, l_hi)
__global__ void chemPotKernel(const double* __restrict__ c, double* __restrict__ mu, const size_t nx, const size_t ny,
                              const size_t NZ, const size_t z0, const int l_lo, const int l_hi) {
    const size_t x = blockIdx.x * BX + threadIdx.x;
    const size_t y = blockIdx.y * BY + threadIdx.y;
    if (x >= nx || y >= ny) return;
    for (int l = l_lo + blockIdx.z; l < l_hi; l += gridDim.z) {
        const size_t gz = z0 + l - HALO;  // valid global plane (caller guarantees existence)
        const size_t lp = (gz < NZ - 1) ? l + 1 : l;
        const size_t ln = (gz > 0) ? l - 1 : l;
        const size_t idx = (size_t)l * nx * ny + y * nx + x;
        const double cv = c[idx];
        mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                  + 3.0 * cv + cv * cv * cv
                  - gamma_ * laplacian(c, nx, ny, x, y, l, lp, ln);
    }
}

// Concentration update on local planes [l_lo, l_hi)
__global__ void updateKernel(double* __restrict__ cnew, const double* __restrict__ cold, const double* __restrict__ mu,
                             const size_t nx, const size_t ny, const size_t NZ, const size_t z0,
                             const int l_lo, const int l_hi) {
    const size_t x = blockIdx.x * BX + threadIdx.x;
    const size_t y = blockIdx.y * BY + threadIdx.y;
    if (x >= nx || y >= ny) return;
    for (int l = l_lo + blockIdx.z; l < l_hi; l += gridDim.z) {
        const size_t gz = z0 + l - HALO;
        const size_t lp = (gz < NZ - 1) ? l + 1 : l;
        const size_t ln = (gz > 0) ? l - 1 : l;
        const size_t idx = (size_t)l * nx * ny + y * nx + x;
        cnew[idx] = cold[idx] + dt * D * laplacian(mu, nx, ny, x, y, l, lp, ln);
    }
}

__global__ void initKernel(double* __restrict__ c, const size_t nx, const size_t ny, const size_t NZ, const size_t z0,
                           const size_t nzl) {
    const size_t x = blockIdx.x * BX + threadIdx.x;
    const size_t y = blockIdx.y * BY + threadIdx.y;
    if (x >= nx || y >= ny) return;
    const size_t vol = nx * ny * NZ;
    for (size_t o = blockIdx.z; o < nzl; o += gridDim.z) {
        const size_t linear_id = (z0 + o) * (nx * ny) + y * nx + x;
        const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
        c[(o + HALO) * (nx * ny) + y * nx + x] = -1.0 + 2.0 * pseudo;
    }
}

static dim3 gridFor(const Slab& s, int planes) {
    return dim3((unsigned)((s.nx + BX - 1) / BX), (unsigned)((s.ny + BY - 1) / BY),
                (unsigned)std::max(1, std::min(planes, 65535)));
}

// o_* are owned-relative plane indices (o = l - HALO); ranges are clamped to existing global planes for mu.
static void launchChemPot(const Slab& s, const double* c, double* mu, long o_lo, long o_hi, cudaStream_t st) {
    o_lo = std::max(o_lo, -(long)s.z0);
    o_hi = std::min(o_hi, (long)(s.nz - s.z0));
    if (o_hi <= o_lo) return;
    chemPotKernel<<<gridFor(s, (int)(o_hi - o_lo)), dim3(BX, BY), 0, st>>>(c, mu, s.nx, s.ny, s.nz, s.z0,
                                                                           (int)(o_lo + HALO), (int)(o_hi + HALO));
}

static void launchUpdate(const Slab& s, double* cn, const double* c, const double* mu, long o_lo, long o_hi,
                         cudaStream_t st) {
    if (o_hi <= o_lo) return;
    updateKernel<<<gridFor(s, (int)(o_hi - o_lo)), dim3(BX, BY), 0, st>>>(cn, c, mu, s.nx, s.ny, s.nz, s.z0,
                                                                          (int)(o_lo + HALO), (int)(o_hi + HALO));
}

// Distributed validation: each rank checks its owned slab (OpenMP), results are combined with MPI.
static bool validateResult(const double* c, const size_t n, MPI_Comm comm, const int rank) {
    int bad = 0;
    double minVal = INFINITY, maxVal = -INFINITY;
#pragma omp parallel for reduction(| : bad) reduction(min : minVal) reduction(max : maxVal)
    for (size_t i = 0; i < n; ++i) {
        const double v = c[i];
        if (std::isnan(v) || std::isinf(v)) bad |= 1;
        minVal = std::min(minVal, v);
        maxVal = std::max(maxVal, v);
    }
    int anyBad = 0;
    MPI_Allreduce(&bad, &anyBad, 1, MPI_INT, MPI_MAX, comm);
    if (anyBad) {
        if (rank == 0) printf("Validation failed: found NaN or Inf value\n");
        return false;
    }
    double gmin, gmax;
    MPI_Allreduce(&minVal, &gmin, 1, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(&maxVal, &gmax, 1, MPI_DOUBLE, MPI_MAX, comm);
    if (rank == 0) printf("Concentration range: [%.6f, %.6f]\n", gmin, gmax);
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
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int wrank, wsize;
    MPI_Comm_rank(MPI_COMM_WORLD, &wrank);
    MPI_Comm_size(MPI_COMM_WORLD, &wsize);

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
            if (wrank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (wrank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    // One GPU per rank (round-robin over the GPUs visible on this node)
    {
        MPI_Comm node;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, wrank, MPI_INFO_NULL, &node);
        int lrank;
        MPI_Comm_rank(node, &lrank);
        MPI_Comm_free(&node);
        int ndev = 0;
        CUDA_CHECK(cudaGetDeviceCount(&ndev));
        CUDA_CHECK(cudaSetDevice(lrank % ndev));
    }

    // Use only as many ranks as there are slabs with at least HALO planes each
    const int active = (int)std::max<size_t>(1, std::min<size_t>((size_t)wsize, nz / HALO));
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, wrank < active ? 0 : MPI_UNDEFINED, wrank, &comm);
    if (comm == MPI_COMM_NULL) {
        MPI_Finalize();
        return 0;
    }
    int rank, nranks;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nranks);

    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const size_t gridSize = nx * ny * nz;
    const size_t nxy = nx * ny;

    Slab s;
    s.nx = nx; s.ny = ny; s.nz = nz;
    {
        const size_t base = nz / nranks, rem = nz % nranks;
        s.nzl = base + ((size_t)rank < rem ? 1 : 0);
        s.z0 = (size_t)rank * base + std::min((size_t)rank, rem);
    }
    const long nzl = (long)s.nzl;
    const size_t localElems = (size_t)(s.nzl + 2 * HALO) * nxy;
    const size_t haloElems = (size_t)HALO * nxy;
    const int down = rank > 0 ? rank - 1 : MPI_PROC_NULL;
    const int up = rank < nranks - 1 ? rank + 1 : MPI_PROC_NULL;

    double *d_c, *d_cn, *d_mu;
    CUDA_CHECK(cudaMalloc(&d_c, localElems * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cn, localElems * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_mu, localElems * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_c, 0, localElems * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_cn, 0, localElems * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_mu, 0, localElems * sizeof(double)));

    double *h_sendDown, *h_sendUp, *h_recvDown, *h_recvUp;
    CUDA_CHECK(cudaMallocHost(&h_sendDown, haloElems * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&h_sendUp, haloElems * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&h_recvDown, haloElems * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&h_recvUp, haloElems * sizeof(double)));

    cudaStream_t sc, sh;
    CUDA_CHECK(cudaStreamCreateWithFlags(&sc, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&sh, cudaStreamNonBlocking));
    cudaEvent_t evStep, evHalo;
    CUDA_CHECK(cudaEventCreateWithFlags(&evStep, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&evHalo, cudaEventDisableTiming));

    // Initialize concentration field (on device)
    if (rank == 0) printf("Initializing concentration field...\n");
    initKernel<<<gridFor(s, (int)s.nzl), dim3(BX, BY), 0, sc>>>(d_c, nx, ny, nz, s.z0, s.nzl);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(sc));

    // Exchange the HALO boundary planes of d_c with neighbours (staged through pinned host memory)
    auto exchange = [&](double* c) {
        if (nranks == 1) return;
        if (down != MPI_PROC_NULL)
            CUDA_CHECK(cudaMemcpyAsync(h_sendDown, c + haloElems, haloElems * sizeof(double), cudaMemcpyDeviceToHost, sh));
        if (up != MPI_PROC_NULL)
            CUDA_CHECK(cudaMemcpyAsync(h_sendUp, c + (size_t)s.nzl * nxy, haloElems * sizeof(double),
                                       cudaMemcpyDeviceToHost, sh));
        CUDA_CHECK(cudaStreamSynchronize(sh));
        MPI_Request req[4];
        MPI_Irecv(h_recvDown, (int)haloElems, MPI_DOUBLE, down, 0, comm, &req[0]);
        MPI_Irecv(h_recvUp, (int)haloElems, MPI_DOUBLE, up, 1, comm, &req[1]);
        MPI_Isend(h_sendDown, (int)haloElems, MPI_DOUBLE, down, 1, comm, &req[2]);
        MPI_Isend(h_sendUp, (int)haloElems, MPI_DOUBLE, up, 0, comm, &req[3]);
        MPI_Waitall(4, req, MPI_STATUSES_IGNORE);
        if (down != MPI_PROC_NULL)
            CUDA_CHECK(cudaMemcpyAsync(c, h_recvDown, haloElems * sizeof(double), cudaMemcpyHostToDevice, sh));
        if (up != MPI_PROC_NULL)
            CUDA_CHECK(cudaMemcpyAsync(c + (size_t)(s.nzl + HALO) * nxy, h_recvUp, haloElems * sizeof(double),
                                       cudaMemcpyHostToDevice, sh));
        CUDA_CHECK(cudaEventRecord(evHalo, sh));
    };

    // Planes that do not depend on neighbour data can be computed while the halo exchange is in flight
    const bool overlap = nzl >= 4 && nranks > 1;
    const long muInLo = overlap ? 1 : 0, muInHi = overlap ? nzl - 1 : 0;
    const long upInLo = overlap ? 2 : 0, upInHi = overlap ? nzl - 2 : 0;

    // Run simulation
    MPI_Barrier(comm);
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < iterations; ++t) {
        // halo exchange of the current field runs concurrently with the interior computation
        CUDA_CHECK(cudaEventRecord(evStep, sc));
        CUDA_CHECK(cudaStreamWaitEvent(sh, evStep, 0));

        launchChemPot(s, d_c, d_mu, muInLo, muInHi, sc);
        launchUpdate(s, d_cn, d_c, d_mu, upInLo, upInHi, sc);

        exchange(d_c);
        CUDA_CHECK(cudaStreamWaitEvent(sc, evHalo, 0));

        // remaining planes (all planes if no overlap): mu on [-1, nzl] minus interior, then update
        if (overlap) {
            launchChemPot(s, d_c, d_mu, -1, muInLo, sc);
            launchChemPot(s, d_c, d_mu, muInHi, nzl + 1, sc);
            launchUpdate(s, d_cn, d_c, d_mu, 0, upInLo, sc);
            launchUpdate(s, d_cn, d_c, d_mu, upInHi, nzl, sc);
        } else {
            launchChemPot(s, d_c, d_mu, -1, nzl + 1, sc);
            launchUpdate(s, d_cn, d_c, d_mu, 0, nzl, sc);
        }
        CUDA_CHECK(cudaGetLastError());

        std::swap(d_c, d_cn);
    }
    CUDA_CHECK(cudaStreamSynchronize(sc));
    MPI_Barrier(comm);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long local_ms = duration.count();
    long max_ms = 0;
    MPI_Reduce(&local_ms, &max_ms, 1, MPI_LONG, MPI_MAX, 0, comm);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", max_ms);

        // Calculate performance
        double cellUpdates = (double)gridSize * iterations;
        double mcups = cellUpdates / (max_ms / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    int ret = 0;
    if (printResults || validate) {
        const size_t ownedElems = s.nzl * nxy;
        double* h_c;
        CUDA_CHECK(cudaMallocHost(&h_c, ownedElems * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(h_c, d_c + haloElems, ownedElems * sizeof(double), cudaMemcpyDeviceToHost));

        // Print results for external validation (gather full field on rank 0)
        if (printResults) {
            const size_t chunk = (size_t)1 << 26;
            if (rank == 0) {
                std::vector<double> full(gridSize);
                std::memcpy(full.data(), h_c, ownedElems * sizeof(double));
                size_t off = ownedElems;
                for (int r = 1; r < nranks; ++r) {
                    const size_t base = nz / nranks, rem = nz % nranks;
                    const size_t n = (base + ((size_t)r < rem ? 1 : 0)) * nxy;
                    for (size_t p = 0; p < n; p += chunk) {
                        const int cnt = (int)std::min(chunk, n - p);
                        MPI_Recv(full.data() + off + p, cnt, MPI_DOUBLE, r, 2, comm, MPI_STATUS_IGNORE);
                    }
                    off += n;
                }
                print_results(full, "Concentration");
            } else {
                for (size_t p = 0; p < ownedElems; p += chunk)
                    MPI_Send(h_c + p, (int)std::min(chunk, ownedElems - p), MPI_DOUBLE, 0, 2, comm);
            }
        }

        // Validation
        if (validate) {
            if (rank == 0) printf("Validating result...\n");
            bool valid = validateResult(h_c, ownedElems, comm, rank);
            if (rank == 0) printf(valid ? "Validation: PASSED\n" : "Validation: FAILED\n");
            ret = valid ? 0 : 1;
        }
        CUDA_CHECK(cudaFreeHost(h_c));
    }

    CUDA_CHECK(cudaEventDestroy(evStep));
    CUDA_CHECK(cudaEventDestroy(evHalo));
    CUDA_CHECK(cudaStreamDestroy(sc));
    CUDA_CHECK(cudaStreamDestroy(sh));
    CUDA_CHECK(cudaFreeHost(h_sendDown));
    CUDA_CHECK(cudaFreeHost(h_sendUp));
    CUDA_CHECK(cudaFreeHost(h_recvDown));
    CUDA_CHECK(cudaFreeHost(h_recvUp));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_cn));
    CUDA_CHECK(cudaFree(d_mu));
    MPI_Comm_free(&comm);
    MPI_Finalize();
    return ret;
}
