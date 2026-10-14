#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// Hybrid decomposition of the z axis into slabs (one slab per GPU):
//   MPI    - each process owns a contiguous range of slabs; halo planes that cross a
//            process boundary are exchanged with MPI_Isend/MPI_Irecv (one exchange per step).
//   OpenMP - one host thread per GPU in each process. Each thread drives its own GPU, and
//            halo planes that cross a thread boundary go through shared pinned host buffers.
//            OpenMP also runs the host-side validation reductions.
//   CUDA   - chemical potential and update kernels run on each slab's GPU. The interior of
//            the chemical potential overlaps the halo exchange.

namespace {

constexpr size_t kHalo = 2;          // ghost planes per side: two-plane halos allow one exchange per step
constexpr int kTagUp = 10;           // MPI tag: data moving to the higher-ranked process
constexpr int kTagDown = 11;         // MPI tag: data moving to the lower-ranked process

// Physical parameters
constexpr double kDx = 1.0;
constexpr double kDy = 1.0;
constexpr double kDz = 1.0;
constexpr double kDt = 0.01;
constexpr double kEAA = -(2.0 / 9.0);
constexpr double kEBB = -(2.0 / 9.0);
constexpr double kEAB = (2.0 / 9.0);
constexpr double kGamma = 0.5;
constexpr double kD = 1.0;

const dim3 kBlock(32, 8, 1);

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err_));                                  \
            MPI_Abort(MPI_COMM_WORLD, 1);                                       \
        }                                                                       \
    } while (0)

// 7-point Laplacian with clamped x/y neighbours. z neighbours are read from ghost planes,
// which hold replicated edge planes at the global boundary, so no z clamp is needed.
__device__ __forceinline__ double laplacian(const double* __restrict__ f, size_t i, size_t x, size_t y,
                                            size_t nx, size_t ny, size_t nxy) {
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t yn = (y > 0) ? y - 1 : 0;
    const size_t row = i - x;
    const double center = f[i];
    const double cxx = (f[row + xp] + f[row + xn] - 2.0 * center) / (kDx * kDx);
    const double cyy = (f[i + (yp - y) * nx] + f[i - (y - yn) * nx] - 2.0 * center) / (kDy * kDy);
    const double czz = (f[i + nxy] + f[i - nxy] - 2.0 * center) / (kDz * kDz);
    return cxx + cyy + czz;
}

// Initial field, c = -1 + 2 * pseudo(global linear index). Fills the nl owned planes.
__global__ void initKernel(double* __restrict__ c, size_t nx, size_t ny, size_t zStart, size_t nl, size_t vol) {
    const size_t nxy = nx * ny;
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= nx) return;
    for (size_t y = blockIdx.y * blockDim.y + threadIdx.y; y < ny; y += gridDim.y * blockDim.y) {
        for (size_t zl = blockIdx.z; zl < nl; zl += gridDim.z) {
            const size_t linear_id = (zStart + zl) * nxy + y * nx + x;
            const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
            c[(kHalo + zl) * nxy + y * nx + x] = -1.0 + 2.0 * pseudo;
        }
    }
}

// Chemical potential on planes [zBegin, zEnd) of a slab buffer (indices include ghost planes).
__global__ void chemicalPotentialKernel(const double* __restrict__ c, double* __restrict__ mu,
                                        size_t nx, size_t ny, size_t zBegin, size_t zEnd) {
    const size_t nxy = nx * ny;
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= nx) return;
    for (size_t y = blockIdx.y * blockDim.y + threadIdx.y; y < ny; y += gridDim.y * blockDim.y) {
        for (size_t z = zBegin + blockIdx.z; z < zEnd; z += gridDim.z) {
            const size_t i = z * nxy + y * nx + x;
            const double cv = c[i];
            mu[i] = 4.5 * ((cv + 1.0) * kEAA + (cv - 1.0) * kEBB - 2.0 * cv * kEAB)
                  + 3.0 * cv + cv * cv * cv
                  - kGamma * laplacian(c, i, x, y, nx, ny, nxy);
        }
    }
}

// Cahn-Hilliard update on planes [zBegin, zEnd): cnew = cold + dt * D * lap(mu)
__global__ void updateKernel(const double* __restrict__ cold, double* __restrict__ cnew,
                             const double* __restrict__ mu, size_t nx, size_t ny,
                             size_t zBegin, size_t zEnd) {
    const size_t nxy = nx * ny;
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= nx) return;
    for (size_t y = blockIdx.y * blockDim.y + threadIdx.y; y < ny; y += gridDim.y * blockDim.y) {
        for (size_t z = zBegin + blockIdx.z; z < zEnd; z += gridDim.z) {
            const size_t i = z * nxy + y * nx + x;
            cnew[i] = cold[i] + kDt * kD * laplacian(mu, i, x, y, nx, ny, nxy);
        }
    }
}

dim3 planeGrid(size_t nx, size_t ny, size_t planes) {
    const size_t gx = (nx + kBlock.x - 1) / kBlock.x;
    const size_t gy = std::min<size_t>((ny + kBlock.y - 1) / kBlock.y, 65535);
    const size_t gz = std::min<size_t>(planes, 65535);
    return dim3(static_cast<unsigned int>(gx), static_cast<unsigned int>(gy), static_cast<unsigned int>(gz));
}

void launchInit(cudaStream_t s, double* c, size_t nx, size_t ny, size_t zStart, size_t nl, size_t vol) {
    if (nl == 0) return;
    initKernel<<<planeGrid(nx, ny, nl), kBlock, 0, s>>>(c, nx, ny, zStart, nl, vol);
    CUDA_CHECK(cudaGetLastError());
}

void launchChemicalPotential(cudaStream_t s, const double* c, double* mu, size_t nx, size_t ny,
                             size_t zBegin, size_t zEnd) {
    if (zBegin >= zEnd) return;
    chemicalPotentialKernel<<<planeGrid(nx, ny, zEnd - zBegin), kBlock, 0, s>>>(c, mu, nx, ny, zBegin, zEnd);
    CUDA_CHECK(cudaGetLastError());
}

void launchUpdate(cudaStream_t s, const double* cold, double* cnew, const double* mu, size_t nx, size_t ny,
                  size_t zBegin, size_t zEnd) {
    if (zBegin >= zEnd) return;
    updateKernel<<<planeGrid(nx, ny, zEnd - zBegin), kBlock, 0, s>>>(cold, cnew, mu, nx, ny, zBegin, zEnd);
    CUDA_CHECK(cudaGetLastError());
}

// One z-slab owned by one OpenMP thread (and therefore one GPU). Slabs with nl == 0 are idle.
struct Slab {
    int device = 0;
    size_t zStart = 0;          // first global plane owned
    size_t nl = 0;              // owned planes
    size_t hostOffset = 0;      // offset (in elements) of this slab in the process-local host copy
    bool hasLow = false;        // a lower neighbour slab exists
    bool hasHigh = false;       // an upper neighbour slab exists
    int cur = 0;                // index of the current concentration buffer
    double* c[2] = {nullptr, nullptr};
    double* mu = nullptr;
    cudaStream_t comp = nullptr;   // kernels
    cudaStream_t comm = nullptr;   // halo transfers
    cudaEvent_t ready = nullptr;   // recorded after the last update
    double* outTop = nullptr;      // pinned: top owned planes, sent to the upper neighbour
    double* outBot = nullptr;      // pinned: bottom owned planes, sent to the lower neighbour
    double* inLow = nullptr;       // pinned: incoming lower ghost planes
    double* inHigh = nullptr;      // pinned: incoming upper ghost planes
};

struct Config {
    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;
};

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

// Returns false if the program should exit now, with exitCode set.
bool parseArgs(int argc, char** argv, Config& cfg, bool root, int& exitCode) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            cfg.nx = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            cfg.ny = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            cfg.nz = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            cfg.iterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            cfg.validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            cfg.printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (root) printUsage(argv[0]);
            exitCode = 0;
            return false;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            exitCode = 1;
            return false;
        }
    }

    if (cfg.ny == 0) cfg.ny = cfg.nx;
    if (cfg.nz == 0) cfg.nz = cfg.nx;
    return true;
}

int runSimulation(const Config& cfg, int rank, int nprocs, bool root) {
    const size_t nx = cfg.nx;
    const size_t ny = cfg.ny;
    const size_t nz = cfg.nz;
    const size_t nxy = nx * ny;
    const size_t gridSize = nx * ny * nz;
    const int iterations = cfg.iterations;
    const bool needHost = cfg.printResults || cfg.validate;

    if (root) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", cfg.validate ? "enabled" : "disabled");
    }

    // GPUs per process: the ranks on a node share its GPUs, one OpenMP thread per GPU.
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0;
    int localSize = 1;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_size(localComm, &localSize);
    MPI_Comm_free(&localComm);

    int devCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devCount));
    if (devCount < 1) {
        fprintf(stderr, "No CUDA devices available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const bool shareDevices = localSize > devCount;   // more ranks than GPUs: ranks share GPUs
    const int T = std::max(1, devCount / localSize);  // threads (= GPUs) per process
    auto deviceOf = [&](int t) { return shareDevices ? localRank % devCount : localRank * T + t; };

    // Global slab numbering: process p owns slabs [slabBegin[p], slabBegin[p] + threadsOf[p]).
    std::vector<int> threadsOf(nprocs);
    MPI_Allgather(&T, 1, MPI_INT, threadsOf.data(), 1, MPI_INT, MPI_COMM_WORLD);
    std::vector<int> slabBegin(nprocs, 0);
    for (int p = 1; p < nprocs; ++p) slabBegin[p] = slabBegin[p - 1] + threadsOf[p - 1];
    const int totalSlabs = slabBegin[nprocs - 1] + threadsOf[nprocs - 1];

    // Every active slab needs at least two planes so that a neighbour can supply both ghost planes.
    const int activeSlabs = static_cast<int>(std::min<size_t>(totalSlabs, std::max<size_t>(1, nz / 2)));
    const size_t base = nz / activeSlabs;
    const size_t rem = nz % activeSlabs;
    auto slabPlanes = [&](int s) -> size_t { return base + (static_cast<size_t>(s) < rem ? 1 : 0); };
    auto slabStart = [&](int s) -> size_t { return base * s + std::min<size_t>(s, rem); };
    auto ownerOf = [&](int slab) {
        for (int p = 0; p < nprocs; ++p) {
            if (slab >= slabBegin[p] && slab < slabBegin[p] + threadsOf[p]) return p;
        }
        return -1;
    };

    const int firstSlab = slabBegin[rank];
    const int lastSlab = firstSlab + T - 1;
    const bool lowerExists = firstSlab > 0 && firstSlab < activeSlabs;
    const bool upperExists = lastSlab + 1 < activeSlabs;
    const int lowerRank = lowerExists ? ownerOf(firstSlab - 1) : -1;
    const int upperRank = upperExists ? ownerOf(lastSlab + 1) : -1;

    std::vector<Slab> slabs(T);
    size_t localPlanes = 0;
    for (int t = 0; t < T; ++t) {
        Slab& sl = slabs[t];
        const int s = firstSlab + t;
        sl.device = deviceOf(t);
        if (s < activeSlabs) {
            sl.nl = slabPlanes(s);
            sl.zStart = slabStart(s);
            sl.hasLow = s > 0;
            sl.hasHigh = s + 1 < activeSlabs;
        }
        sl.hostOffset = localPlanes * nxy;
        localPlanes += sl.nl;
    }

    // Planes owned by every process, for gathering the final field on rank 0.
    std::vector<int> planeCount(nprocs);
    const int myPlanes = static_cast<int>(localPlanes);
    MPI_Allgather(&myPlanes, 1, MPI_INT, planeCount.data(), 1, MPI_INT, MPI_COMM_WORLD);

    if (root) printf("Initializing concentration field...\n");
    omp_set_dynamic(0);
    std::vector<double> hostLocal;   // final owned concentration of this process (-r / -v only)
    if (needHost) hostLocal.resize(localPlanes * nxy);

    // Allocate per-GPU state and initialise the concentration field on the GPUs.
    #pragma omp parallel num_threads(T)
    {
        const int t = omp_get_thread_num();
        if (omp_get_num_threads() != T) {
            if (t == 0) fprintf(stderr, "OpenMP team size mismatch\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        Slab& sl = slabs[t];
        if (sl.nl > 0) {
            CUDA_CHECK(cudaSetDevice(sl.device));
            const size_t slabBytes = (sl.nl + 2 * kHalo) * nxy * sizeof(double);
            const size_t haloBytes = 2 * nxy * sizeof(double);
            CUDA_CHECK(cudaMalloc(&sl.c[0], slabBytes));
            CUDA_CHECK(cudaMalloc(&sl.c[1], slabBytes));
            CUDA_CHECK(cudaMalloc(&sl.mu, slabBytes));
            CUDA_CHECK(cudaMallocHost(&sl.outTop, haloBytes));
            CUDA_CHECK(cudaMallocHost(&sl.outBot, haloBytes));
            CUDA_CHECK(cudaMallocHost(&sl.inLow, haloBytes));
            CUDA_CHECK(cudaMallocHost(&sl.inHigh, haloBytes));
            CUDA_CHECK(cudaStreamCreateWithFlags(&sl.comp, cudaStreamNonBlocking));
            CUDA_CHECK(cudaStreamCreateWithFlags(&sl.comm, cudaStreamNonBlocking));
            CUDA_CHECK(cudaEventCreateWithFlags(&sl.ready, cudaEventDisableTiming));
            launchInit(sl.comp, sl.c[0], nx, ny, sl.zStart, sl.nl, gridSize);
            CUDA_CHECK(cudaEventRecord(sl.ready, sl.comp));
        }
    }

    if (root) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double tStart = MPI_Wtime();

    // Time loop
    #pragma omp parallel num_threads(T)
    {
        const int t = omp_get_thread_num();
        Slab& sl = slabs[t];
        const size_t nl = sl.nl;
        if (nl > 0) CUDA_CHECK(cudaSetDevice(sl.device));

        for (int it = 0; it < iterations; ++it) {
            if (nl > 0) {
                // Interior chemical potential needs only owned planes, so it overlaps the exchange.
                launchChemicalPotential(sl.comp, sl.c[sl.cur], sl.mu, nx, ny, kHalo + 1, kHalo + nl - 1);

                // Stage the boundary planes for the neighbours.
                CUDA_CHECK(cudaStreamWaitEvent(sl.comm, sl.ready, 0));
                const double* cur = sl.c[sl.cur];
                if (sl.hasLow) {
                    CUDA_CHECK(cudaMemcpyAsync(sl.outBot, cur + kHalo * nxy, 2 * nxy * sizeof(double),
                                               cudaMemcpyDeviceToHost, sl.comm));
                }
                if (sl.hasHigh) {
                    CUDA_CHECK(cudaMemcpyAsync(sl.outTop, cur + (kHalo + nl - 2) * nxy, 2 * nxy * sizeof(double),
                                               cudaMemcpyDeviceToHost, sl.comm));
                }
                CUDA_CHECK(cudaStreamSynchronize(sl.comm));
            }
            #pragma omp barrier

            // Process boundaries: the master thread exchanges halos with the neighbouring processes.
            if (t == 0 && (lowerExists || upperExists)) {
                MPI_Request reqs[4];
                int nreq = 0;
                const int count = static_cast<int>(2 * nxy);
                if (lowerExists) {
                    MPI_Irecv(slabs[0].inLow, count, MPI_DOUBLE, lowerRank, kTagUp, MPI_COMM_WORLD, &reqs[nreq++]);
                    MPI_Isend(slabs[0].outBot, count, MPI_DOUBLE, lowerRank, kTagDown, MPI_COMM_WORLD, &reqs[nreq++]);
                }
                if (upperExists) {
                    MPI_Irecv(slabs[T - 1].inHigh, count, MPI_DOUBLE, upperRank, kTagDown, MPI_COMM_WORLD, &reqs[nreq++]);
                    MPI_Isend(slabs[T - 1].outTop, count, MPI_DOUBLE, upperRank, kTagUp, MPI_COMM_WORLD, &reqs[nreq++]);
                }
                MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);
            }
            #pragma omp barrier

            if (nl > 0) {
                // Fill ghost planes: from the neighbour, or replicate the edge plane at the global boundary.
                double* cur = sl.c[sl.cur];
                const size_t planeBytes = nxy * sizeof(double);
                if (sl.hasLow) {
                    const double* src = (t == 0) ? sl.inLow : slabs[t - 1].outTop;
                    CUDA_CHECK(cudaMemcpyAsync(cur, src, 2 * nxy * sizeof(double), cudaMemcpyDeviceToDevice, sl.comm));
                } else {
                    for (size_t g = 0; g < kHalo; ++g) {
                        CUDA_CHECK(cudaMemcpyAsync(cur + g * nxy, cur + kHalo * nxy, planeBytes,
                                                   cudaMemcpyDeviceToDevice, sl.comm));
                    }
                }
                if (sl.hasHigh) {
                    const double* src = (t == T - 1) ? sl.inHigh : slabs[t + 1].outBot;
                    CUDA_CHECK(cudaMemcpyAsync(cur + (kHalo + nl) * nxy, src, 2 * nxy * sizeof(double),
                                               cudaMemcpyDeviceToDevice, sl.comm));
                } else {
                    for (size_t g = 0; g < kHalo; ++g) {
                        CUDA_CHECK(cudaMemcpyAsync(cur + (kHalo + nl + g) * nxy, cur + (kHalo + nl - 1) * nxy,
                                                   planeBytes, cudaMemcpyDeviceToDevice, sl.comm));
                    }
                }
                CUDA_CHECK(cudaStreamSynchronize(sl.comm));
            }
            #pragma omp barrier

            if (nl > 0) {
                const double* cur = sl.c[sl.cur];
                double* next = sl.c[sl.cur ^ 1];
                // Boundary chemical potential, including the ghost planes the update reads.
                if (sl.hasLow) launchChemicalPotential(sl.comp, cur, sl.mu, nx, ny, kHalo - 1, kHalo);
                launchChemicalPotential(sl.comp, cur, sl.mu, nx, ny, kHalo, kHalo + 1);
                if (nl > 1) launchChemicalPotential(sl.comp, cur, sl.mu, nx, ny, kHalo + nl - 1, kHalo + nl);
                if (sl.hasHigh) launchChemicalPotential(sl.comp, cur, sl.mu, nx, ny, kHalo + nl, kHalo + nl + 1);

                // Global boundary: the clamped neighbour of an edge plane is the edge plane itself.
                const size_t planeBytes = nxy * sizeof(double);
                if (!sl.hasLow) {
                    CUDA_CHECK(cudaMemcpyAsync(sl.mu + (kHalo - 1) * nxy, sl.mu + kHalo * nxy, planeBytes,
                                               cudaMemcpyDeviceToDevice, sl.comp));
                }
                if (!sl.hasHigh) {
                    CUDA_CHECK(cudaMemcpyAsync(sl.mu + (kHalo + nl) * nxy, sl.mu + (kHalo + nl - 1) * nxy, planeBytes,
                                               cudaMemcpyDeviceToDevice, sl.comp));
                }

                launchUpdate(sl.comp, cur, next, sl.mu, nx, ny, kHalo, kHalo + nl);
                CUDA_CHECK(cudaEventRecord(sl.ready, sl.comp));
                sl.cur ^= 1;
            }
        }

        if (nl > 0) CUDA_CHECK(cudaStreamSynchronize(sl.comp));
        #pragma omp barrier
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double elapsed = MPI_Wtime() - tStart;
    double elapsedMax = 0.0;
    MPI_Allreduce(&elapsed, &elapsedMax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    const long durationMs = static_cast<long>(elapsedMax * 1000.0);
    if (root) {
        printf("Computation time: %ld ms\n", durationMs);

        // Calculate performance
        double cellUpdates = (double)gridSize * iterations;
        double mcups = cellUpdates / (durationMs / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Copy the final owned planes back to the host and release GPU resources.
    #pragma omp parallel num_threads(T)
    {
        const int t = omp_get_thread_num();
        Slab& sl = slabs[t];
        if (sl.nl > 0) {
            CUDA_CHECK(cudaSetDevice(sl.device));
            if (needHost) {
                CUDA_CHECK(cudaMemcpy(hostLocal.data() + sl.hostOffset, sl.c[sl.cur] + kHalo * nxy,
                                      sl.nl * nxy * sizeof(double), cudaMemcpyDeviceToHost));
            }
        }
        #pragma omp barrier
        if (sl.nl > 0) {
            CUDA_CHECK(cudaFree(sl.c[0]));
            CUDA_CHECK(cudaFree(sl.c[1]));
            CUDA_CHECK(cudaFree(sl.mu));
            CUDA_CHECK(cudaFreeHost(sl.outTop));
            CUDA_CHECK(cudaFreeHost(sl.outBot));
            CUDA_CHECK(cudaFreeHost(sl.inLow));
            CUDA_CHECK(cudaFreeHost(sl.inHigh));
            CUDA_CHECK(cudaEventDestroy(sl.ready));
            CUDA_CHECK(cudaStreamDestroy(sl.comp));
            CUDA_CHECK(cudaStreamDestroy(sl.comm));
        }
    }

    // Print results for external validation
    if (cfg.printResults) {
        std::vector<double> full;
        std::vector<int> counts(nprocs);
        std::vector<int> displs(nprocs, 0);
        for (int p = 0; p < nprocs; ++p) {
            counts[p] = planeCount[p] * static_cast<int>(nxy);
            if (p > 0) displs[p] = displs[p - 1] + counts[p - 1];
        }
        if (root) full.resize(gridSize);
        MPI_Gatherv(hostLocal.data(), static_cast<int>(localPlanes * nxy), MPI_DOUBLE,
                    full.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (root) print_results(full, "Concentration");
    }

    // Validation
    int status = 0;
    if (cfg.validate) {
        if (root) printf("Validating result...\n");

        // OpenMP scan of this process's owned planes.
        const size_t n = hostLocal.size();
        const double* h = hostLocal.data();
        int localNonFinite = 0;
        double localMin = INFINITY;
        double localMax = -INFINITY;
        #pragma omp parallel for reduction(max : localNonFinite) reduction(min : localMin) reduction(max : localMax)
        for (size_t i = 0; i < n; ++i) {
            const double v = h[i];
            if (!std::isfinite(v)) localNonFinite = 1;
            localMin = std::min(localMin, v);
            localMax = std::max(localMax, v);
        }

        int nonFinite = 0;
        double minVal = 0.0;
        double maxVal = 0.0;
        MPI_Allreduce(&localNonFinite, &nonFinite, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&localMin, &minVal, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&localMax, &maxVal, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        bool valid = true;
        if (nonFinite) {
            if (root) printf("Validation failed: found NaN or Inf value\n");
            valid = false;
        } else {
            if (root) printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);
            if (maxVal > 10.0 || minVal < -10.0) {
                if (root) printf("Validation failed: values out of expected range\n");
                valid = false;
            }
        }

        if (root) printf(valid ? "Validation: PASSED\n" : "Validation: FAILED\n");
        status = valid ? 0 : 1;
    }

    return status;
}

}  // namespace

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "MPI does not provide MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int rank = 0;
    int nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    const bool root = (rank == 0);

    Config cfg;
    int status = 0;
    if (parseArgs(argc, argv, cfg, root, status)) {
        status = runSimulation(cfg, rank, nprocs, root);
    }

    MPI_Finalize();
    return status;
}
