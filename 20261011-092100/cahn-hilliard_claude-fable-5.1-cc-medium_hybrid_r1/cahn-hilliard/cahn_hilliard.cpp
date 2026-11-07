// Cahn-Hilliard phase separation benchmark - hybrid MPI + OpenMP + CUDA version.
//
// Parallelization strategy:
//   * MPI:    1D slab decomposition of the global grid along z. Each rank owns a
//             contiguous block of z-planes and drives one GPU. Halos are two
//             planes wide so that a single exchange per time step suffices: the
//             chemical potential is computed redundantly on the first ghost plane
//             and the concentration update then needs no second exchange.
//             Halo communication is overlapped with the interior stencil.
//   * CUDA:   All stencil work runs on the GPU. Each thread owns an (x,y) column
//             and marches along z keeping a 3-plane register window, so every
//             cell is loaded from global memory only once per kernel. Separate
//             compute and communication streams overlap PCIe traffic with
//             computation.
//   * OpenMP: Host-side work (initial field generation, validation reductions)
//             runs multithreaded on each rank.
//
// Boundary conditions (clamped Neumann) are implemented by filling the ghost
// planes at the global z boundaries with copies of the boundary plane, which is
// exactly what the index clamping in the original code does.

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call)                                                                     \
    do {                                                                                     \
        cudaError_t err__ = (call);                                                          \
        if (err__ != cudaSuccess) {                                                          \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err__), __FILE__, \
                    __LINE__);                                                               \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                    \
        }                                                                                    \
    } while (0)

#define MPI_CHECK(call)                                                             \
    do {                                                                            \
        int err__ = (call);                                                         \
        if (err__ != MPI_SUCCESS) {                                                 \
            fprintf(stderr, "MPI error %d at %s:%d\n", err__, __FILE__, __LINE__);  \
            MPI_Abort(MPI_COMM_WORLD, 1);                                           \
        }                                                                           \
    } while (0)

// 3D index calculation
__host__ __device__ inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z,
                                                 const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// ---------------------------------------------------------------------------
// CUDA kernels
// ---------------------------------------------------------------------------

constexpr int BLOCK_X = 32;
constexpr int BLOCK_Y = 8;

// Chemical potential on local planes [zBegin, zEnd) of mu (mu has one ghost
// plane on each side, c has two). mu plane k reads c planes k, k+1, k+2.
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
chemicalPotentialKernel(const double* __restrict__ c, double* __restrict__ mu,
                        const int nx, const int ny, const int zBegin, const int zEnd,
                        const int zChunk, const double inv_dx2, const double inv_dy2,
                        const double inv_dz2, const double centre, const double gamma,
                        const double muLin, const double muConst) {
    const int x = blockIdx.x * BLOCK_X + threadIdx.x;
    const int y = blockIdx.y * BLOCK_Y + threadIdx.y;
    if (x >= nx || y >= ny) return;

    const int z0 = zBegin + blockIdx.z * zChunk;
    const int z1 = min(z0 + zChunk, zEnd);
    if (z0 >= z1) return;

    const size_t plane = static_cast<size_t>(nx) * ny;
    const int xp = (x < nx - 1) ? x + 1 : x;
    const int xn = (x > 0) ? x - 1 : 0;
    const int yp = (y < ny - 1) ? y + 1 : y;
    const int yn = (y > 0) ? y - 1 : 0;

    // mu plane z corresponds to c plane z + 1 (c has two ghost planes, mu one).
    const double* cp = c + plane;
    size_t ic = idx3(x, y, z0, nx, ny);
    double czn = cp[ic - plane];
    double cz = cp[ic];

    for (int z = z0; z < z1; ++z, ic += plane) {
        const double czp = cp[ic + plane];
        // Laplacian with the centre term factored out (mathematically identical to
        // summing the three second differences; keeps the FP64 op count low).
        const double lap = (cp[ic - x + xp] + cp[ic - x + xn]) * inv_dx2
                           + (cp[ic + (yp - y) * nx] + cp[ic + (yn - y) * nx]) * inv_dy2
                           + (czp + czn) * inv_dz2 - cz * centre;

        // 4.5*((c+1)e_AA + (c-1)e_BB - 2c e_AB) + 3c + c^3 - gamma*lap, with the
        // linear and constant parts folded into muLin / muConst.
        mu[ic] = (cz * muLin + muConst) + cz * cz * cz - gamma * lap;

        czn = cz;
        cz = czp;
    }
}

// Concentration update on local real planes [zBegin, zEnd) (mu indexing, i.e.
// real planes are [1, nzLocal + 1)). cnew is written in the same layout as c.
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
updateKernel(double* __restrict__ cnew, const double* __restrict__ cold,
             const double* __restrict__ mu, const int nx, const int ny, const int zBegin,
             const int zEnd, const int zChunk, const double dtD, const double inv_dx2,
             const double inv_dy2, const double inv_dz2, const double centre) {
    const int x = blockIdx.x * BLOCK_X + threadIdx.x;
    const int y = blockIdx.y * BLOCK_Y + threadIdx.y;
    if (x >= nx || y >= ny) return;

    const int z0 = zBegin + blockIdx.z * zChunk;
    const int z1 = min(z0 + zChunk, zEnd);
    if (z0 >= z1) return;

    const size_t plane = static_cast<size_t>(nx) * ny;
    const int xp = (x < nx - 1) ? x + 1 : x;
    const int xn = (x > 0) ? x - 1 : 0;
    const int yp = (y < ny - 1) ? y + 1 : y;
    const int yn = (y > 0) ? y - 1 : 0;

    const double* coldp = cold + plane;  // shift so cold plane index matches mu index
    double* cnewp = cnew + plane;
    size_t im = idx3(x, y, z0, nx, ny);
    double mzn = mu[im - plane];
    double mz = mu[im];

    for (int z = z0; z < z1; ++z, im += plane) {
        const double mzp = mu[im + plane];
        const double lap = (mu[im - x + xp] + mu[im - x + xn]) * inv_dx2
                           + (mu[im + (yp - y) * nx] + mu[im + (yn - y) * nx]) * inv_dy2
                           + (mzp + mzn) * inv_dz2 - mz * centre;

        cnewp[im] = coldp[im] + dtD * lap;

        mzn = mz;
        mz = mzp;
    }
}

// Choose how many z-planes each thread column marches so that enough blocks are
// resident to fill the device even for small grids.
static int chooseChunk(const int nx, const int ny, const int nPlanes) {
    if (nPlanes <= 0) return 1;
    const double cells = static_cast<double>(nx) * ny * nPlanes;
    int chunk = static_cast<int>(cells / (1 << 21));  // target ~2M threads in flight
    chunk = std::max(1, std::min(chunk, 32));
    return std::min(chunk, nPlanes);
}

struct StencilParams {
    double inv_dx2, inv_dy2, inv_dz2;
    double centre;   // 2*(1/dx^2 + 1/dy^2 + 1/dz^2)
    double gamma;
    double muLin;    // 4.5*(e_AA + e_BB - 2*e_AB) + 3
    double muConst;  // 4.5*(e_AA - e_BB)
    double dtD;
};

static void launchMu(const double* c, double* mu, const int nx, const int ny, const int zBegin,
                     const int zEnd, const StencilParams& p, cudaStream_t stream) {
    const int n = zEnd - zBegin;
    if (n <= 0) return;
    const int chunk = chooseChunk(nx, ny, n);
    const dim3 block(BLOCK_X, BLOCK_Y, 1);
    const dim3 grid((nx + BLOCK_X - 1) / BLOCK_X, (ny + BLOCK_Y - 1) / BLOCK_Y,
                    (n + chunk - 1) / chunk);
    chemicalPotentialKernel<<<grid, block, 0, stream>>>(c, mu, nx, ny, zBegin, zEnd, chunk,
                                                        p.inv_dx2, p.inv_dy2, p.inv_dz2,
                                                        p.centre, p.gamma, p.muLin, p.muConst);
    CUDA_CHECK(cudaGetLastError());
}

static void launchUpdate(double* cnew, const double* cold, const double* mu, const int nx,
                         const int ny, const int zBegin, const int zEnd, const StencilParams& p,
                         cudaStream_t stream) {
    const int n = zEnd - zBegin;
    if (n <= 0) return;
    const int chunk = chooseChunk(nx, ny, n);
    const dim3 block(BLOCK_X, BLOCK_Y, 1);
    const dim3 grid((nx + BLOCK_X - 1) / BLOCK_X, (ny + BLOCK_Y - 1) / BLOCK_Y,
                    (n + chunk - 1) / chunk);
    updateKernel<<<grid, block, 0, stream>>>(cnew, cold, mu, nx, ny, zBegin, zEnd, chunk, p.dtD,
                                             p.inv_dx2, p.inv_dy2, p.inv_dz2, p.centre);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Host-side helpers (OpenMP)
// ---------------------------------------------------------------------------

// Initialize the local slab [zStart, zStart + nzLocal) of the concentration field.
void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny,
                             const size_t nz, const size_t zStart, const size_t nzLocal) {
    const size_t vol = nx * ny * nz;

#pragma omp parallel for collapse(2) schedule(static)
    for (size_t zl = 0; zl < nzLocal; ++zl) {
        for (size_t y = 0; y < ny; ++y) {
            const size_t z = zStart + zl;
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, zl, nx, ny);
                // Generate pseudo-random value in [-1, 1]
                const size_t linear_id = z * (nx * ny) + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[idx] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

bool validateResult(const std::vector<double>& c, [[maybe_unused]] const size_t nx,
                    [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    const size_t n = c.size();
    const double* data = c.data();

    // Check for NaN or Inf
    bool bad = false;
#pragma omp parallel for reduction(|| : bad) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        bad = bad || std::isnan(data[i]) || std::isinf(data[i]);
    }
    if (bad) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    // Check if values are in reasonable range for concentration field
    // After Cahn-Hilliard evolution, values should typically remain bounded
    double minVal = c[0];
    double maxVal = c[0];
#pragma omp parallel for reduction(min : minVal) reduction(max : maxVal) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        minVal = std::min(minVal, data[i]);
        maxVal = std::max(maxVal, data[i]);
    }

    printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);

    // Values should generally stay within reasonable bounds
    if (maxVal > 10.0 || minVal < -10.0) {
        printf("Validation failed: values out of expected range\n");
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
    MPI_CHECK(MPI_Init(&argc, &argv));
    int rank = 0, nranks = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &nranks));

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

    // ---- GPU selection: one device per rank, round-robin among node-local ranks ----
    MPI_Comm nodeComm;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
                                  &nodeComm));
    int localRank = 0;
    MPI_CHECK(MPI_Comm_rank(nodeComm, &localRank));
    MPI_CHECK(MPI_Comm_free(&nodeComm));
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount <= 0) {
        if (rank == 0) fprintf(stderr, "No CUDA devices found\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int device = localRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));

    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, OpenMP threads/rank: %d, GPUs/node: %d\n", nranks,
               omp_get_max_threads(), deviceCount);
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

    StencilParams sp;
    sp.inv_dx2 = 1.0 / (dx * dx);
    sp.inv_dy2 = 1.0 / (dy * dy);
    sp.inv_dz2 = 1.0 / (dz * dz);
    sp.centre = 2.0 * (sp.inv_dx2 + sp.inv_dy2 + sp.inv_dz2);
    sp.gamma = gamma;
    sp.muLin = 4.5 * (e_AA + e_BB - 2.0 * e_AB) + 3.0;
    sp.muConst = 4.5 * (e_AA - e_BB);
    sp.dtD = dt * D;

    const size_t gridSize = nx * ny * nz;
    const size_t plane = nx * ny;

    // ---- Domain decomposition along z ----
    // Every active rank needs at least two planes (halo width). Ranks beyond
    // that get no work but still take part in collectives.
    const int activeRanks = static_cast<int>(std::max<size_t>(1, std::min<size_t>(nranks, nz / 2)));
    std::vector<int> zCounts(nranks, 0), zStarts(nranks, 0);
    {
        const size_t base = nz / activeRanks;
        const size_t rem = nz % activeRanks;
        size_t acc = 0;
        for (int r = 0; r < activeRanks; ++r) {
            zStarts[r] = static_cast<int>(acc);
            zCounts[r] = static_cast<int>(base + (static_cast<size_t>(r) < rem ? 1 : 0));
            acc += zCounts[r];
        }
        for (int r = activeRanks; r < nranks; ++r) zStarts[r] = static_cast<int>(nz);
    }
    const int nzLocal = zCounts[rank];
    const size_t zStart = zStarts[rank];
    const bool active = nzLocal > 0;
    const int prevRank = (active && rank > 0) ? rank - 1 : MPI_PROC_NULL;
    const int nextRank = (active && rank + 1 < activeRanks) ? rank + 1 : MPI_PROC_NULL;
    const bool globalLo = active && prevRank == MPI_PROC_NULL;
    const bool globalHi = active && nextRank == MPI_PROC_NULL;

    // ---- Allocate device arrays (c: 2 ghost planes each side, mu: 1 each side) ----
    const size_t cPlanes = static_cast<size_t>(nzLocal) + 4;
    const size_t muPlanes = static_cast<size_t>(nzLocal) + 2;
    double* dCold = nullptr;
    double* dCnew = nullptr;
    double* dMu = nullptr;
    double* hSendLo = nullptr;  // pinned host staging buffers, 2 planes each
    double* hSendHi = nullptr;
    double* hRecvLo = nullptr;
    double* hRecvHi = nullptr;
    cudaStream_t compStream, commStream;
    cudaEvent_t sendReady, haloReady, stepDone;
    CUDA_CHECK(cudaStreamCreateWithFlags(&compStream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&commStream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&sendReady, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&haloReady, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&stepDone, cudaEventDisableTiming));

    if (active) {
        CUDA_CHECK(cudaMalloc(&dCold, cPlanes * plane * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&dCnew, cPlanes * plane * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&dMu, muPlanes * plane * sizeof(double)));
        CUDA_CHECK(cudaMemset(dCold, 0, cPlanes * plane * sizeof(double)));
        CUDA_CHECK(cudaMemset(dCnew, 0, cPlanes * plane * sizeof(double)));
        CUDA_CHECK(cudaMemset(dMu, 0, muPlanes * plane * sizeof(double)));
        CUDA_CHECK(cudaHostAlloc(&hSendLo, 2 * plane * sizeof(double), cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&hSendHi, 2 * plane * sizeof(double), cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&hRecvLo, 2 * plane * sizeof(double), cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&hRecvHi, 2 * plane * sizeof(double), cudaHostAllocDefault));
    }

    // ---- Initialize concentration field (host, OpenMP) and upload ----
    if (rank == 0) printf("Initializing concentration field...\n");
    std::vector<double> hLocal(static_cast<size_t>(nzLocal) * plane);
    if (active) {
        initializeConcentration(hLocal, nx, ny, nz, zStart, nzLocal);
        CUDA_CHECK(cudaMemcpy(dCold + 2 * plane, hLocal.data(), hLocal.size() * sizeof(double),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    const size_t planeBytes = plane * sizeof(double);
    const int nzl = nzLocal;
    const int ix = static_cast<int>(nx);
    const int iy = static_cast<int>(ny);

    // Warm up the device (module load, stream creation) outside the timed region.
    // The warm-up results are fully overwritten by the first real time step.
    if (active) {
        launchMu(dCold, dMu, ix, iy, 1, nzl + 1, sp, compStream);
        launchUpdate(dCnew, dCold, dMu, ix, iy, 1, nzl + 1, sp, compStream);
        CUDA_CHECK(cudaStreamSynchronize(compStream));
    }

    // Run simulation
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < iterations; ++t) {
        if (active) {
            // Plane layout (c): [0,1] lower ghosts, [2, nzl+2) real, [nzl+2, nzl+4) upper ghosts.
            // Plane layout (mu): [0] lower ghost, [1, nzl+1) real, [nzl+1] upper ghost.

            // 1. Stage boundary planes of c to pinned host memory (comm stream).
            //    The comm stream must see the previous step's update first.
            CUDA_CHECK(cudaStreamWaitEvent(commStream, stepDone, 0));
            if (prevRank != MPI_PROC_NULL)
                CUDA_CHECK(cudaMemcpyAsync(hSendLo, dCold + 2 * plane, 2 * planeBytes,
                                           cudaMemcpyDeviceToHost, commStream));
            if (nextRank != MPI_PROC_NULL)
                CUDA_CHECK(cudaMemcpyAsync(hSendHi, dCold + static_cast<size_t>(nzl) * plane,
                                           2 * planeBytes, cudaMemcpyDeviceToHost, commStream));
            CUDA_CHECK(cudaEventRecord(sendReady, commStream));

            // 2. Interior work that needs no halo data, overlapped with communication:
            //    chemical potential on mu planes [2, nzl) (reads only real c planes),
            //    then the concentration update on planes [3, nzl-1) (reads only those).
            launchMu(dCold, dMu, ix, iy, 2, nzl, sp, compStream);
            launchUpdate(dCnew, dCold, dMu, ix, iy, 3, nzl - 1, sp, compStream);

            // 3. Halo exchange (overlapped with the interior kernel).
            CUDA_CHECK(cudaEventSynchronize(sendReady));
            MPI_Request reqs[4];
            int nreq = 0;
            if (prevRank != MPI_PROC_NULL) {
                MPI_CHECK(MPI_Irecv(hRecvLo, static_cast<int>(2 * plane), MPI_DOUBLE, prevRank, 0,
                                    MPI_COMM_WORLD, &reqs[nreq++]));
                MPI_CHECK(MPI_Isend(hSendLo, static_cast<int>(2 * plane), MPI_DOUBLE, prevRank, 1,
                                    MPI_COMM_WORLD, &reqs[nreq++]));
            }
            if (nextRank != MPI_PROC_NULL) {
                MPI_CHECK(MPI_Irecv(hRecvHi, static_cast<int>(2 * plane), MPI_DOUBLE, nextRank, 1,
                                    MPI_COMM_WORLD, &reqs[nreq++]));
                MPI_CHECK(MPI_Isend(hSendHi, static_cast<int>(2 * plane), MPI_DOUBLE, nextRank, 0,
                                    MPI_COMM_WORLD, &reqs[nreq++]));
            }
            MPI_CHECK(MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE));

            // 4. Fill ghost planes: received data, or clamped copies at global boundaries.
            if (prevRank != MPI_PROC_NULL) {
                CUDA_CHECK(cudaMemcpyAsync(dCold, hRecvLo, 2 * planeBytes, cudaMemcpyHostToDevice,
                                           commStream));
            } else if (globalLo) {
                CUDA_CHECK(cudaMemcpyAsync(dCold + plane, dCold + 2 * plane, planeBytes,
                                           cudaMemcpyDeviceToDevice, commStream));
            }
            if (nextRank != MPI_PROC_NULL) {
                CUDA_CHECK(cudaMemcpyAsync(dCold + static_cast<size_t>(nzl + 2) * plane, hRecvHi,
                                           2 * planeBytes, cudaMemcpyHostToDevice, commStream));
            } else if (globalHi) {
                CUDA_CHECK(cudaMemcpyAsync(dCold + static_cast<size_t>(nzl + 2) * plane,
                                           dCold + static_cast<size_t>(nzl + 1) * plane, planeBytes,
                                           cudaMemcpyDeviceToDevice, commStream));
            }
            CUDA_CHECK(cudaEventRecord(haloReady, commStream));
            CUDA_CHECK(cudaStreamWaitEvent(compStream, haloReady, 0));

            // 5. Boundary chemical potential: mu planes [0,2) and [max(2,nzl), nzl+2).
            //    For inter-rank boundaries the ghost mu plane is computed exactly from
            //    the two-wide c halo; at global boundaries it is a clamped copy.
            launchMu(dCold, dMu, ix, iy, 0, 2, sp, compStream);
            launchMu(dCold, dMu, ix, iy, std::max(2, nzl), nzl + 2, sp, compStream);
            if (globalLo)
                CUDA_CHECK(cudaMemcpyAsync(dMu, dMu + plane, planeBytes, cudaMemcpyDeviceToDevice,
                                           compStream));
            if (globalHi)
                CUDA_CHECK(cudaMemcpyAsync(dMu + static_cast<size_t>(nzl + 1) * plane,
                                           dMu + static_cast<size_t>(nzl) * plane, planeBytes,
                                           cudaMemcpyDeviceToDevice, compStream));

            // 6. Concentration update on the remaining boundary planes [1,3) and [nzl-1, nzl+1).
            launchUpdate(dCnew, dCold, dMu, ix, iy, 1, std::min(3, nzl + 1), sp, compStream);
            launchUpdate(dCnew, dCold, dMu, ix, iy, std::max(3, nzl - 1), nzl + 1, sp, compStream);
            CUDA_CHECK(cudaEventRecord(stepDone, compStream));
        }

        // Swap buffers
        std::swap(dCold, dCnew);
    }

    if (active) CUDA_CHECK(cudaStreamSynchronize(compStream));
    auto end = std::chrono::high_resolution_clock::now();
    long elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &elapsedMs, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD));
    auto duration = std::chrono::milliseconds(elapsedMs);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(duration.count()));

        // Calculate performance
        double cellUpdates = (double)gridSize * iterations;
        double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    int exitCode = 0;

    if (printResults || validate) {
        // Download local slab and gather the full field on rank 0.
        if (active) {
            CUDA_CHECK(cudaMemcpy(hLocal.data(), dCold + 2 * plane, hLocal.size() * sizeof(double),
                                  cudaMemcpyDeviceToHost));
        }
        std::vector<int> recvCounts(nranks), displs(nranks);
        for (int r = 0; r < nranks; ++r) {
            recvCounts[r] = static_cast<int>(static_cast<size_t>(zCounts[r]) * plane);
            displs[r] = static_cast<int>(static_cast<size_t>(zStarts[r]) * plane);
        }
        std::vector<double> cold;
        if (rank == 0) cold.resize(gridSize);
        MPI_CHECK(MPI_Gatherv(hLocal.data(), static_cast<int>(hLocal.size()), MPI_DOUBLE,
                              cold.data(), recvCounts.data(), displs.data(), MPI_DOUBLE, 0,
                              MPI_COMM_WORLD));

        if (rank == 0) {
            // Print results for external validation
            if (printResults) {
                print_results(cold, "Concentration");
            }

            // Validation
            if (validate) {
                printf("Validating result...\n");
                bool valid = validateResult(cold, nx, ny, nz);

                if (valid) {
                    printf("Validation: PASSED\n");
                } else {
                    printf("Validation: FAILED\n");
                    exitCode = 1;
                }
            }
        }
        MPI_CHECK(MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD));
    }

    // Cleanup
    if (active) {
        CUDA_CHECK(cudaFree(dCold));
        CUDA_CHECK(cudaFree(dCnew));
        CUDA_CHECK(cudaFree(dMu));
        CUDA_CHECK(cudaFreeHost(hSendLo));
        CUDA_CHECK(cudaFreeHost(hSendHi));
        CUDA_CHECK(cudaFreeHost(hRecvLo));
        CUDA_CHECK(cudaFreeHost(hRecvHi));
    }
    CUDA_CHECK(cudaEventDestroy(sendReady));
    CUDA_CHECK(cudaEventDestroy(haloReady));
    CUDA_CHECK(cudaEventDestroy(stepDone));
    CUDA_CHECK(cudaStreamDestroy(compStream));
    CUDA_CHECK(cudaStreamDestroy(commStream));

    MPI_Finalize();
    return exitCode;
}
