// Cahn-Hilliard phase separation benchmark: hybrid MPI + OpenMP + CUDA version.
//
// Parallelization strategy
//  * MPI:    the 3D grid is decomposed into z-slabs, one slab per rank. Each
//            rank drives one GPU (selected by node-local rank). Halos are two
//            z-planes wide so that both the chemical potential and the
//            concentration update of a time step can be done with a single
//            halo exchange per step (mu is recomputed redundantly on one extra
//            plane at each slab boundary).
//  * CUDA:   all stencil work (chemical potential and concentration update)
//            runs on the GPU with z-marching kernels. The update of the slab
//            boundary planes is computed first, so that their transfer and
//            the MPI exchange overlap with the update of the slab interior
//            on a second stream.
//  * OpenMP: host-side work (initial condition, validation reductions, result
//            assembly) is multi-threaded.
//
// The numerical scheme and the order of floating-point operations per cell are
// identical to the original serial implementation.

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
        cudaError_t err__ = (call);                                                        \
        if (err__ != cudaSuccess) {                                                        \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__,     \
                    cudaGetErrorString(err__));                                            \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                  \
        }                                                                                  \
    } while (0)

#define MPI_CHECK(call)                                                                    \
    do {                                                                                   \
        int err__ = (call);                                                                \
        if (err__ != MPI_SUCCESS) {                                                        \
            fprintf(stderr, "MPI error %d in %s at %s:%d\n", err__, #call, __FILE__, __LINE__); \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                  \
        }                                                                                  \
    } while (0)

// Number of ghost planes on each side of the local z-slab
constexpr int HALO = 2;

// Thread block shape of the stencil kernels
constexpr int BX = 32;
constexpr int BY = 8;

// 3D index calculation
__host__ __device__ inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z,
                                                 const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Description of the local z-slab owned by this rank
struct Slab {
    size_t nx = 0, ny = 0, nz = 0;  // global grid
    size_t z0 = 0, z1 = 0;          // global z range [z0, z1) owned by this rank
    size_t nloc = 0;                // z1 - z0
    size_t plane = 0;               // nx * ny
    size_t nplanes = 0;             // nloc + 2 * HALO (allocated planes)
    // local plane index of global plane gz is (gz - z0 + HALO)
};

// Compute the clamped 7-point Laplacian of plane-marching values.
// prev/next are the values at the (clamped) z-1 / z+1 neighbours.
__device__ __forceinline__ double laplacian(const double* __restrict__ f, const size_t lidx,
                                            const double cv, const double prev, const double next,
                                            const size_t x, const size_t y,
                                            const size_t nx, const size_t ny,
                                            const double inv_dx2, const double inv_dy2, const double inv_dz2) {
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t yn = (y > 0) ? y - 1 : 0;

    const double cxx = (f[lidx + (xp - x)] + f[lidx - (x - xn)] - 2.0 * cv) * inv_dx2;
    const double cyy = (f[lidx + (yp - y) * nx] + f[lidx - (y - yn) * nx] - 2.0 * cv) * inv_dy2;
    const double czz = (next + prev - 2.0 * cv) * inv_dz2;
    return cxx + cyy + czz;
}

// Chemical potential kernel: computes mu for global planes [gza, gzb).
// Each thread handles one (x, y) column and marches over a chunk of planes.
__global__ void __launch_bounds__(BX * BY)
chemicalPotentialKernel(const double* __restrict__ c, double* __restrict__ mu,
                        const size_t nx, const size_t ny, const size_t nz, const size_t z0,
                        const size_t gza, const size_t gzb, const int zchunk,
                        const double inv_dx2, const double inv_dy2, const double inv_dz2,
                        const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    const size_t x = blockIdx.x * BX + threadIdx.x;
    const size_t y = blockIdx.y * BY + threadIdx.y;
    if (x >= nx || y >= ny) return;

    const size_t plane = nx * ny;
    const size_t zs = gza + static_cast<size_t>(blockIdx.z) * zchunk;
    const size_t ze = min(zs + static_cast<size_t>(zchunk), gzb);
    if (zs >= ze) return;

    const size_t col = y * nx + x;
    // local plane index of global plane gz: gz - z0 + HALO
    size_t lidx = (zs - z0 + HALO) * plane + col;
    double prev = (zs > 0) ? c[lidx - plane] : c[lidx];
    double cv = c[lidx];

    for (size_t gz = zs; gz < ze; ++gz, lidx += plane) {
        const double next = (gz < nz - 1) ? c[lidx + plane] : cv;

        const double lap = laplacian(c, lidx, cv, prev, next, x, y, nx, ny, inv_dx2, inv_dy2, inv_dz2);
        mu[lidx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                   + 3.0 * cv + cv * cv * cv
                   - gamma * lap;

        prev = cv;
        cv = next;
    }
}

// Concentration update kernel: cnew = cold + dt*D*Laplacian(mu) for global planes [gza, gzb).
__global__ void __launch_bounds__(BX * BY)
updateKernel(double* __restrict__ cnew, const double* __restrict__ cold, const double* __restrict__ mu,
             const size_t nx, const size_t ny, const size_t nz, const size_t z0,
             const size_t gza, const size_t gzb, const int zchunk,
             const double inv_dx2, const double inv_dy2, const double inv_dz2,
             const double dtD) {
    const size_t x = blockIdx.x * BX + threadIdx.x;
    const size_t y = blockIdx.y * BY + threadIdx.y;
    if (x >= nx || y >= ny) return;

    const size_t plane = nx * ny;
    const size_t zs = gza + static_cast<size_t>(blockIdx.z) * zchunk;
    const size_t ze = min(zs + static_cast<size_t>(zchunk), gzb);
    if (zs >= ze) return;

    const size_t col = y * nx + x;
    size_t lidx = (zs - z0 + HALO) * plane + col;
    double prev = (zs > 0) ? mu[lidx - plane] : mu[lidx];
    double mv = mu[lidx];

    for (size_t gz = zs; gz < ze; ++gz, lidx += plane) {
        const double next = (gz < nz - 1) ? mu[lidx + plane] : mv;

        const double lap = laplacian(mu, lidx, mv, prev, next, x, y, nx, ny, inv_dx2, inv_dy2, inv_dz2);
        cnew[lidx] = cold[lidx] + dtD * lap;

        prev = mv;
        mv = next;
    }
}

// Choose the z-chunk per thread so that the GPU is well occupied even for small planes.
static int chooseZChunk(const size_t nx, const size_t ny, const size_t nzr, const int numSMs) {
    const size_t bx = (nx + BX - 1) / BX;
    const size_t by = (ny + BY - 1) / BY;
    int zc = 16;
    while (zc > 1 && bx * by * ((nzr + zc - 1) / zc) < static_cast<size_t>(8 * numSMs)) zc /= 2;
    return zc;
}

static dim3 gridFor(const size_t nx, const size_t ny, const size_t nzr, const int zc) {
    return dim3(static_cast<unsigned>((nx + BX - 1) / BX),
                static_cast<unsigned>((ny + BY - 1) / BY),
                static_cast<unsigned>((nzr + zc - 1) / zc));
}

// Physical and numerical parameters (identical to the serial version)
struct Params {
    double dx = 1.0, dy = 1.0, dz = 1.0;
    double dt = 0.01;
    double e_AA = -(2.0 / 9.0);
    double e_BB = -(2.0 / 9.0);
    double e_AB = (2.0 / 9.0);
    double gamma = 0.5;
    double D = 1.0;
};

// Halo exchange state: the two boundary planes on each side are staged through
// pinned host buffers (works with any MPI implementation).
struct HaloExchange {
    int lower = MPI_PROC_NULL, upper = MPI_PROC_NULL;
    size_t bytes = 0;         // bytes of HALO planes
    size_t count = 0;         // doubles of HALO planes
    double* h_send_lo = nullptr;
    double* h_send_hi = nullptr;
    double* h_recv_lo = nullptr;
    double* h_recv_hi = nullptr;
    cudaEvent_t evRecvDone = nullptr;  // signalled once the previous H2D copies out of the recv buffers finished
    cudaEvent_t evD2H_lo = nullptr;    // signalled once the lower / upper send planes have landed on the host
    cudaEvent_t evD2H_hi = nullptr;
};

// Exchange the HALO boundary planes of device array `d` with the neighbours.
// The two sides are pipelined: each side's send starts as soon as its own D2H copy
// has landed, and each side's H2D copy starts as soon as its message has arrived.
// All device work is issued on `stream`.
static void exchangeHalos(double* d, const Slab& s, const HaloExchange& hx, cudaStream_t stream) {
    const size_t plane = s.plane;
    // Local plane layout: [0,HALO) lower ghost, [HALO, HALO+nloc) owned, [HALO+nloc, HALO+nloc+HALO) upper ghost
    double* d_own[2] = { d + HALO * plane, d + s.nloc * plane };            // first / last HALO owned planes
    double* d_ghost[2] = { d, d + (HALO + s.nloc) * plane };                // lower / upper ghost planes
    double* h_send[2] = { hx.h_send_lo, hx.h_send_hi };
    double* h_recv[2] = { hx.h_recv_lo, hx.h_recv_hi };
    const int nbr[2] = { hx.lower, hx.upper };
    const int sendTag[2] = { 0, 1 };  // tag identifies the direction: 0 = towards lower rank, 1 = towards upper
    const int recvTag[2] = { 1, 0 };
    cudaEvent_t evD2H[2] = { hx.evD2H_lo, hx.evD2H_hi };

    // The receive buffers may still be read by the previous exchange's H2D copies
    CUDA_CHECK(cudaEventSynchronize(hx.evRecvDone));

    MPI_Request rreq[2] = { MPI_REQUEST_NULL, MPI_REQUEST_NULL };
    MPI_Request sreq[2] = { MPI_REQUEST_NULL, MPI_REQUEST_NULL };
    for (int i = 0; i < 2; ++i) {
        if (nbr[i] == MPI_PROC_NULL) continue;
        MPI_CHECK(MPI_Irecv(h_recv[i], static_cast<int>(hx.count), MPI_DOUBLE, nbr[i], recvTag[i], MPI_COMM_WORLD, &rreq[i]));
        CUDA_CHECK(cudaMemcpyAsync(h_send[i], d_own[i], hx.bytes, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaEventRecord(evD2H[i], stream));
    }
    for (int i = 0; i < 2; ++i) {
        if (nbr[i] == MPI_PROC_NULL) continue;
        CUDA_CHECK(cudaEventSynchronize(evD2H[i]));
        MPI_CHECK(MPI_Isend(h_send[i], static_cast<int>(hx.count), MPI_DOUBLE, nbr[i], sendTag[i], MPI_COMM_WORLD, &sreq[i]));
    }
    // Upload each side as soon as it has arrived
    for (int done = 0; done < 2; ++done) {
        int i = MPI_UNDEFINED;
        MPI_CHECK(MPI_Waitany(2, rreq, &i, MPI_STATUS_IGNORE));
        if (i == MPI_UNDEFINED) break;
        CUDA_CHECK(cudaMemcpyAsync(d_ghost[i], h_recv[i], hx.bytes, cudaMemcpyHostToDevice, stream));
    }
    MPI_CHECK(MPI_Waitall(2, sreq, MPI_STATUSES_IGNORE));
    CUDA_CHECK(cudaEventRecord(hx.evRecvDone, stream));
}

// Initialize concentration field for the local slab (host, OpenMP).
// Produces exactly the same values as the serial version for the owned planes.
void initializeConcentration(std::vector<double>& c, const Slab& s) {
    const size_t nx = s.nx, ny = s.ny;
    const size_t vol = s.nx * s.ny * s.nz;

#pragma omp parallel for collapse(2) schedule(static)
    for (size_t lz = 0; lz < s.nloc; ++lz) {
        for (size_t y = 0; y < ny; ++y) {
            const size_t z = s.z0 + lz;
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

// Validate the distributed result. `c` holds the owned planes of this rank.
bool validateResult(const std::vector<double>& c, const int rank) {
    const size_t n = c.size();

    // Check for NaN or Inf
    int bad = 0;
#pragma omp parallel for reduction(|| : bad) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        if (std::isnan(c[i]) || std::isinf(c[i])) bad = 1;
    }
    int anyBad = 0;
    MPI_CHECK(MPI_Allreduce(&bad, &anyBad, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD));
    if (anyBad) {
        if (rank == 0) printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    // Check if values are in reasonable range for concentration field
    // After Cahn-Hilliard evolution, values should typically remain bounded
    double minVal = n > 0 ? c[0] : HUGE_VAL;
    double maxVal = n > 0 ? c[0] : -HUGE_VAL;
#pragma omp parallel for reduction(min : minVal) reduction(max : maxVal) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        minVal = std::min(minVal, c[i]);
        maxVal = std::max(maxVal, c[i]);
    }
    double gmin = 0.0, gmax = 0.0;
    MPI_CHECK(MPI_Allreduce(&minVal, &gmin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Allreduce(&maxVal, &gmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));

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
    int provided = 0;
    MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));
    int rank = 0, size = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &size));

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
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    // ---- Select the GPU: one device per rank, round-robin over node-local ranks ----
    int localRank = 0;
    {
        MPI_Comm localComm;
        MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm));
        MPI_CHECK(MPI_Comm_rank(localComm, &localRank));
        MPI_CHECK(MPI_Comm_free(&localComm));
    }
    int numDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));
    if (numDevices <= 0) {
        if (rank == 0) fprintf(stderr, "No CUDA devices found\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int device = localRank % numDevices;
    CUDA_CHECK(cudaSetDevice(device));
    int numSMs = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, device));

    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Parallelization: %d MPI ranks, %d OpenMP threads/rank, CUDA (%d GPU(s)/node)\n",
               size, omp_get_max_threads(), numDevices);
    }

    // ---- Domain decomposition in z ----
    // Every active rank must own at least HALO planes so that a two-plane halo
    // always comes from the direct neighbour; surplus ranks stay idle.
    const int activeRanks = static_cast<int>(std::max<size_t>(1, std::min<size_t>(static_cast<size_t>(size), nz / HALO)));
    Slab s;
    s.nx = nx; s.ny = ny; s.nz = nz;
    s.plane = nx * ny;
    if (rank < activeRanks) {
        const size_t base = nz / activeRanks;
        const size_t rem = nz % activeRanks;
        const size_t r = static_cast<size_t>(rank);
        s.z0 = r * base + std::min(r, rem);
        s.z1 = s.z0 + base + (r < rem ? 1 : 0);
    } else {
        s.z0 = s.z1 = nz;
    }
    s.nloc = s.z1 - s.z0;
    s.nplanes = s.nloc + 2 * HALO;
    const bool active = s.nloc > 0;

    const Params p{};
    const double inv_dx2 = 1.0 / (p.dx * p.dx);
    const double inv_dy2 = 1.0 / (p.dy * p.dy);
    const double inv_dz2 = 1.0 / (p.dz * p.dz);
    const double dtD = p.dt * p.D;

    const size_t gridSize = nx * ny * nz;
    const size_t localOwned = s.nloc * s.plane;
    const size_t localAlloc = s.nplanes * s.plane;

    // ---- Allocate host and device arrays ----
    std::vector<double> h_c(localOwned);

    double* d_cold = nullptr;
    double* d_cnew = nullptr;
    double* d_mu = nullptr;
    if (active) {
        CUDA_CHECK(cudaMalloc(&d_cold, localAlloc * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_cnew, localAlloc * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_mu, localAlloc * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_cold, 0, localAlloc * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_cnew, 0, localAlloc * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_mu, 0, localAlloc * sizeof(double)));
    }

    HaloExchange hx;
    hx.lower = (active && rank > 0) ? rank - 1 : MPI_PROC_NULL;
    hx.upper = (active && rank + 1 < activeRanks) ? rank + 1 : MPI_PROC_NULL;
    hx.count = HALO * s.plane;
    hx.bytes = hx.count * sizeof(double);
    if (active) {
        CUDA_CHECK(cudaMallocHost(&hx.h_send_lo, hx.bytes));
        CUDA_CHECK(cudaMallocHost(&hx.h_send_hi, hx.bytes));
        CUDA_CHECK(cudaMallocHost(&hx.h_recv_lo, hx.bytes));
        CUDA_CHECK(cudaMallocHost(&hx.h_recv_hi, hx.bytes));
        CUDA_CHECK(cudaEventCreateWithFlags(&hx.evRecvDone, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&hx.evD2H_lo, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&hx.evD2H_hi, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventRecord(hx.evRecvDone, 0));
    }

    cudaStream_t sMain, sInner;
    cudaEvent_t evMuBnd, evBnd, evInner;
    // The main stream carries the slab-boundary work that feeds the halo exchange
    // (the critical path), so it gets the highest priority; the interior work on
    // the second stream fills the GPU while the exchange is in flight.
    int prioLeast = 0, prioGreatest = 0;
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&prioLeast, &prioGreatest));
    CUDA_CHECK(cudaStreamCreateWithPriority(&sMain, cudaStreamNonBlocking, prioGreatest));
    CUDA_CHECK(cudaStreamCreateWithPriority(&sInner, cudaStreamNonBlocking, prioLeast));
    CUDA_CHECK(cudaEventCreateWithFlags(&evMuBnd, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&evBnd, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&evInner, cudaEventDisableTiming));

    // ---- Initialize concentration field ----
    if (rank == 0) printf("Initializing concentration field...\n");
    initializeConcentration(h_c, s);
    if (active) {
        CUDA_CHECK(cudaMemcpyAsync(d_cold + HALO * s.plane, h_c.data(), localOwned * sizeof(double),
                                   cudaMemcpyHostToDevice, sMain));
        exchangeHalos(d_cold, s, hx, sMain);
        CUDA_CHECK(cudaStreamSynchronize(sMain));
    }

    // ---- Kernel launch geometry ----
    // Owned planes are split into boundary planes [z0, bLo) and [bHi, z1), which are
    // sent to the neighbours, and the interior [bLo, bHi).
    // The chemical potential is needed on the owned planes plus one plane on each
    // side (clamped to the global domain): [mz0, mz1). The boundary concentration
    // update needs mu on [z0-1, bLo+1) and [bHi-1, z1+1); these "mu boundary" ranges
    // are computed first so that the halo exchange can start as early as possible,
    // while the mu interior [mLo, mHi) and the concentration interior are computed
    // concurrently on a second stream.
    const size_t mz0 = (s.z0 > 0) ? s.z0 - 1 : 0;
    const size_t mz1 = std::min(s.z1 + 1, nz);
    const bool hasNeighbour = (hx.lower != MPI_PROC_NULL || hx.upper != MPI_PROC_NULL);
    size_t bLo, bHi, mLo, mHi;
    if (hasNeighbour && s.nloc >= 2 * HALO + 2) {
        bLo = s.z0 + HALO;
        bHi = s.z1 - HALO;
        mLo = bLo + 1;
        mHi = bHi - 1;
    } else {
        // Single range: everything is "boundary"
        bLo = s.z1;
        bHi = s.z1;
        mLo = mz1;
        mHi = mz1;
    }
    const bool haveSplit = bHi > bLo;  // split boundary/interior path with overlapped exchange

    const dim3 block(BX, BY, 1);
    // boundary ranges (small): mu [mz0, mLo) & [mHi, mz1); c [z0, bLo) & [bHi, z1)
    const int zcB = 4;
    const dim3 gMuLo = gridFor(nx, ny, mLo - mz0, zcB);
    const dim3 gMuHi = gridFor(nx, ny, mz1 - mHi, zcB);
    const dim3 gCLo = gridFor(nx, ny, bLo - s.z0, zcB);
    const dim3 gCHi = gridFor(nx, ny, s.z1 - bHi, zcB);
    // interior ranges
    const int zcMuI = chooseZChunk(nx, ny, std::max<size_t>(mHi - mLo, 1), numSMs);
    const dim3 gMuI = gridFor(nx, ny, mHi - mLo, zcMuI);
    const int zcCI = chooseZChunk(nx, ny, std::max<size_t>(bHi - bLo, 1), numSMs);
    const dim3 gCI = gridFor(nx, ny, bHi - bLo, zcCI);

    auto launchMu = [&](const dim3& g, const size_t a, const size_t b, const int zc, cudaStream_t st) {
        if (b <= a) return;
        chemicalPotentialKernel<<<g, block, 0, st>>>(
            d_cold, d_mu, nx, ny, nz, s.z0, a, b, zc,
            inv_dx2, inv_dy2, inv_dz2, p.gamma, p.e_AA, p.e_BB, p.e_AB);
        CUDA_CHECK(cudaGetLastError());
    };
    auto launchUpdate = [&](const dim3& g, const size_t a, const size_t b, const int zc, cudaStream_t st) {
        if (b <= a) return;
        updateKernel<<<g, block, 0, st>>>(
            d_cnew, d_cold, d_mu, nx, ny, nz, s.z0, a, b, zc,
            inv_dx2, inv_dy2, inv_dz2, dtD);
        CUDA_CHECK(cudaGetLastError());
    };

    // ---- Run simulation ----
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < iterations; ++t) {
        if (active) {
            if (haveSplit) {
                // Main stream: mu near the slab boundaries (needs fresh halos, present on sMain,
                // and the interior update of the previous step, evInner)
                if (t > 0) CUDA_CHECK(cudaStreamWaitEvent(sMain, evInner, 0));
                launchMu(gMuLo, mz0, mLo, zcB, sMain);
                launchMu(gMuHi, mHi, mz1, zcB, sMain);
                CUDA_CHECK(cudaEventRecord(evMuBnd, sMain));

                // Second stream: mu interior (must not overwrite mu before the previous
                // step's boundary update has read it: evBnd)
                if (t > 0) CUDA_CHECK(cudaStreamWaitEvent(sInner, evBnd, 0));
                launchMu(gMuI, mLo, mHi, zcMuI, sInner);

                // Main stream: boundary update, then exchange the new boundary planes
                launchUpdate(gCLo, s.z0, bLo, zcB, sMain);
                launchUpdate(gCHi, bHi, s.z1, zcB, sMain);
                CUDA_CHECK(cudaEventRecord(evBnd, sMain));

                // Second stream: interior update (needs the boundary mu planes as well)
                CUDA_CHECK(cudaStreamWaitEvent(sInner, evMuBnd, 0));
                launchUpdate(gCI, bLo, bHi, zcCI, sInner);
                CUDA_CHECK(cudaEventRecord(evInner, sInner));

                exchangeHalos(d_cnew, s, hx, sMain);
            } else {
                launchMu(gMuLo, mz0, mLo, zcB, sMain);
                launchUpdate(gCLo, s.z0, bLo, zcB, sMain);
                if (hasNeighbour) exchangeHalos(d_cnew, s, hx, sMain);
            }
        }

        // Swap buffers
        std::swap(d_cold, d_cnew);
    }

    if (active) CUDA_CHECK(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(duration.count()));

        // Calculate performance
        double cellUpdates = (double)gridSize * iterations;
        double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // ---- Copy the owned planes back to the host ----
    if (active) {
        CUDA_CHECK(cudaMemcpy(h_c.data(), d_cold + HALO * s.plane, localOwned * sizeof(double),
                              cudaMemcpyDeviceToHost));
    }

    // Print results for external validation (gathered on rank 0, same layout as serial)
    if (printResults) {
        std::vector<int> counts(size), displs(size);
        const int myCount = static_cast<int>(localOwned);
        MPI_CHECK(MPI_Gather(&myCount, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD));
        std::vector<double> full;
        if (rank == 0) {
            long long total = 0;
            for (int i = 0; i < size; ++i) { displs[i] = static_cast<int>(total); total += counts[i]; }
            full.resize(static_cast<size_t>(total));
        }
        MPI_CHECK(MPI_Gatherv(h_c.data(), myCount, MPI_DOUBLE, full.data(), counts.data(), displs.data(),
                              MPI_DOUBLE, 0, MPI_COMM_WORLD));
        if (rank == 0) print_results(full, "Concentration");
    }

    int ret = 0;
    // Validation
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        bool valid = validateResult(h_c, rank);

        if (valid) {
            if (rank == 0) printf("Validation: PASSED\n");
            ret = 0;
        } else {
            if (rank == 0) printf("Validation: FAILED\n");
            ret = 1;
        }
    }

    // ---- Cleanup ----
    CUDA_CHECK(cudaEventDestroy(evMuBnd));
    CUDA_CHECK(cudaEventDestroy(evBnd));
    CUDA_CHECK(cudaEventDestroy(evInner));
    CUDA_CHECK(cudaStreamDestroy(sMain));
    CUDA_CHECK(cudaStreamDestroy(sInner));
    if (active) {
        CUDA_CHECK(cudaFreeHost(hx.h_send_lo));
        CUDA_CHECK(cudaFreeHost(hx.h_send_hi));
        CUDA_CHECK(cudaFreeHost(hx.h_recv_lo));
        CUDA_CHECK(cudaFreeHost(hx.h_recv_hi));
        CUDA_CHECK(cudaEventDestroy(hx.evRecvDone));
        CUDA_CHECK(cudaEventDestroy(hx.evD2H_lo));
        CUDA_CHECK(cudaEventDestroy(hx.evD2H_hi));
        CUDA_CHECK(cudaFree(d_cold));
        CUDA_CHECK(cudaFree(d_cnew));
        CUDA_CHECK(cudaFree(d_mu));
    }

    MPI_Finalize();
    return ret;
}
