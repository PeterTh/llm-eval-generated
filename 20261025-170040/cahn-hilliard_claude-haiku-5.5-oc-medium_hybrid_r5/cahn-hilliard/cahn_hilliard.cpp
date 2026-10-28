// Cahn-Hilliard phase separation benchmark, hybrid MPI + OpenMP + CUDA.
//
// Decomposition:
//   * MPI:    the z-range is split into contiguous slabs, one group per rank.
//             z-slabs are contiguous in memory (idx = z*nx*ny + y*nx + x), so
//             halo exchanges and the final gather move whole planes.
//   * OpenMP: each rank runs one thread per GPU it may use. Thread t owns a
//             sub-slab of the rank's planes on its own device, and performs
//             the host-side staging for it.
//   * CUDA:   chemical potential and update kernels run on the devices.
//             Halo planes are staged through pinned host buffers. Interior
//             planes (which need no halo data) are computed on a second stream
//             while the MPI halo exchange is in progress.
//
// The per-cell arithmetic and the order of operations are those of the serial
// reference. The fused multiply-adds that GCC forms for the reference build
// (-O3 -march=native, FP contraction enabled) are written as explicit fma()
// calls, so the field, the printed statistics and the hash match bit for bit.

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err_ = (call);                                             \
        if (err_ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err_));                                 \
            MPI_Abort(MPI_COMM_WORLD, 1);                                      \
        }                                                                      \
    } while (0)

// Physical / numerical parameters (identical to the serial reference)
struct Physics {
    double dx, dy, dz, dt, D, gamma, e_AA, e_BB, e_AB;
};

static const Physics kPhysics = {
    1.0, 1.0, 1.0,                // dx, dy, dz
    0.01,                         // dt
    1.0,                          // D
    0.5,                          // gamma
    -(2.0 / 9.0), -(2.0 / 9.0), (2.0 / 9.0)  // e_AA, e_BB, e_AB
};

// Per-thread device state for one sub-slab of the rank's planes.
// Buffers are laid out as (n + 2) planes: plane 0 and plane n+1 are ghosts.
struct DeviceSlab {
    int device = 0;
    size_t offset = 0;  // first local plane of this sub-slab within the rank
    size_t n = 0;       // number of planes owned by this sub-slab
    double* cold = nullptr;
    double* cnew = nullptr;
    double* mu = nullptr;
    cudaStream_t sA = nullptr;  // halo-dependent work
    cudaStream_t sB = nullptr;  // interior work, overlaps with communication
};

// ---------------------------------------------------------------------------
// CUDA kernels
// ---------------------------------------------------------------------------

// Laplacian with clamped boundary conditions in x and y. Along z the ghost
// planes hold clamped copies at the global boundaries, so k-1 / k+1 are always
// valid and equal the clamped reference values.
__device__ __forceinline__ double laplacianAt(const double* __restrict__ f, size_t nx, size_t ny,
                                              size_t x, size_t y, size_t k,
                                              double dx, double dy, double dz) {
    const size_t nxy = nx * ny;
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;

    const size_t row = k * nxy + y * nx;
    const double center = f[row + x];
    const double cxx = (f[row + xp] + f[row + xn] - 2.0 * center) / (dx * dx);
    const double cyy = (f[k * nxy + yp * nx + x] + f[k * nxy + yn * nx + x] - 2.0 * center) / (dy * dy);
    const double czz = (f[(k + 1) * nxy + y * nx + x] + f[(k - 1) * nxy + y * nx + x] - 2.0 * center) / (dz * dz);
    return cxx + cyy + czz;
}

// mu for local planes k0 .. k0+gridDim.z-1
__global__ void chemicalPotentialKernel(const double* __restrict__ c, double* __restrict__ mu,
                                        size_t nx, size_t ny, size_t k0, Physics p) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;

    const size_t k = k0 + blockIdx.z;
    const size_t idx = k * (nx * ny) + y * nx + x;
    const double cv = c[idx];
    const double lap = laplacianAt(c, nx, ny, x, y, k, p.dx, p.dy, p.dz);

    // Same expression as the serial reference, with its fused multiply-adds
    // made explicit (fma is correctly rounded, so this is deterministic):
    //   4.5 * ((cv+1)*e_AA + (cv-1)*e_BB - 2*cv*e_AB) + 3*cv + cv*cv*cv - gamma*lap
    const double s0 = fma(cv + 1.0, p.e_AA, (cv - 1.0) * p.e_BB);
    const double s1 = fma(-(2.0 * cv), p.e_AB, s0);
    const double s2 = fma(s1, 4.5, 3.0 * cv);
    const double s3 = fma(cv, cv * cv, s2);
    mu[idx] = fma(-p.gamma, lap, s3);
}

// Cahn-Hilliard update for local planes k0 .. k0+gridDim.z-1
__global__ void updateKernel(const double* __restrict__ cold, double* __restrict__ cnew,
                             const double* __restrict__ mu,
                             size_t nx, size_t ny, size_t k0, Physics p) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;

    const size_t k = k0 + blockIdx.z;
    const size_t idx = k * (nx * ny) + y * nx + x;
    const double lap = laplacianAt(mu, nx, ny, x, y, k, p.dx, p.dy, p.dz);

    // cold + (dt*D)*lap, fused as in the serial reference
    cnew[idx] = fma(p.dt * p.D, lap, cold[idx]);
}

static void launchChemicalPotential(cudaStream_t st, const double* c, double* mu,
                                    size_t nx, size_t ny, size_t k0, size_t count, const Physics& p) {
    if (count == 0) return;
    const dim3 block(32, 8, 1);
    const dim3 grid(static_cast<unsigned>((nx + block.x - 1) / block.x),
                    static_cast<unsigned>((ny + block.y - 1) / block.y),
                    static_cast<unsigned>(count));
    chemicalPotentialKernel<<<grid, block, 0, st>>>(c, mu, nx, ny, k0, p);
    CUDA_CHECK(cudaGetLastError());
}

static void launchUpdate(cudaStream_t st, const double* cold, double* cnew, const double* mu,
                         size_t nx, size_t ny, size_t k0, size_t count, const Physics& p) {
    if (count == 0) return;
    const dim3 block(32, 8, 1);
    const dim3 grid(static_cast<unsigned>((nx + block.x - 1) / block.x),
                    static_cast<unsigned>((ny + block.y - 1) / block.y),
                    static_cast<unsigned>(count));
    updateKernel<<<grid, block, 0, st>>>(cold, cnew, mu, nx, ny, k0, p);
    CUDA_CHECK(cudaGetLastError());
}

// Asynchronous plane copies (plane = nxy contiguous doubles)
static void copyPlaneAsync(double* dst, const double* src, size_t nxy, cudaMemcpyKind kind, cudaStream_t st) {
    CUDA_CHECK(cudaMemcpyAsync(dst, src, nxy * sizeof(double), kind, st));
}

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------

// Pseudo-random initial value in [-1, 1], identical to the serial reference
static void initializeConcentration(double* c, size_t z0, size_t nzl,
                                    size_t nx, size_t ny, size_t nz) {
    const size_t vol = nx * ny * nz;
    const size_t nxy = nx * ny;

    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t z = 0; z < nzl; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t linear_id = (z0 + z) * nxy + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[z * nxy + y * nx + x] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

bool validateResult(const std::vector<double>& c, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Check for NaN or Inf, and gather min / max in the same pass
    bool nonFinite = false;
    double minVal = c[0];
    double maxVal = c[0];
    #pragma omp parallel for reduction(|| : nonFinite) reduction(min : minVal) reduction(max : maxVal)
    for (size_t i = 0; i < c.size(); ++i) {
        const double val = c[i];
        nonFinite = nonFinite || std::isnan(val) || std::isinf(val);
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }

    if (nonFinite) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
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

// Exchange boundary planes with the neighbouring slabs (master thread only).
// sendLo goes to the lower neighbour, sendHi to the upper neighbour. The ghost
// planes received are recvBelow (from lower) and recvAbove (from upper).
// Missing neighbours are MPI_PROC_NULL, which makes those operations no-ops.
static void exchangeHalos(const double* sendLo, const double* sendHi,
                          double* recvBelow, double* recvAbove,
                          MPI_Datatype plane, int below, int above, int tagBase) {
    MPI_Sendrecv(sendLo, 1, plane, below, tagBase,
                 recvAbove, 1, plane, above, tagBase,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(sendHi, 1, plane, above, tagBase + 1,
                 recvBelow, 1, plane, below, tagBase + 1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

static double* allocPinned(size_t count) {
    double* p = nullptr;
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&p), count * sizeof(double)));
    return p;
}

// ---------------------------------------------------------------------------
// Simulation driver
// ---------------------------------------------------------------------------

struct Options {
    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;
};

static int runBenchmark(const Options& opt, int rank, int numRanks) {
    const size_t nx = opt.nx;
    const size_t ny = opt.ny;
    const size_t nz = opt.nz;
    const size_t nxy = nx * ny;
    const size_t gridSize = nxy * nz;
    const int iterations = opt.iterations;
    const Physics P = kPhysics;
    const bool wantOutput = opt.printResults || opt.validate;

    // ---- Node-local topology and GPU assignment --------------------------
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0;
    int ranksOnNode = 1;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_size(nodeComm, &ranksOnNode);
    MPI_Comm_free(&nodeComm);

    int numDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));
    if (numDevices <= 0) {
        fprintf(stderr, "Rank %d: no CUDA devices found\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    // GPUs per rank on this node (ranks share devices if there are more ranks than GPUs)
    const int devicesPerRank = std::max(1, numDevices / ranksOnNode);

    // ---- Slab decomposition along z ---------------------------------------
    // Ranks beyond nz have no planes and stay idle (they still join collectives).
    const int activeRanks = static_cast<int>(std::min<size_t>(static_cast<size_t>(numRanks), nz));
    const bool active = rank < activeRanks;
    auto planeStart = [&](size_t r) -> size_t {
        return nz * r / static_cast<size_t>(activeRanks);
    };
    const size_t z0 = active ? planeStart(rank) : 0;
    const size_t nzl = active ? planeStart(rank + 1) - z0 : 0;
    const int below = (active && rank > 0) ? rank - 1 : MPI_PROC_NULL;
    const int above = (active && rank + 1 < activeRanks) ? rank + 1 : MPI_PROC_NULL;
    const int T = active ? static_cast<int>(std::min<size_t>(static_cast<size_t>(devicesPerRank), nzl)) : 0;

    MPI_Datatype planeType;
    MPI_Type_contiguous(static_cast<int>(nxy), MPI_DOUBLE, &planeType);
    MPI_Type_commit(&planeType);

    // ---- Allocation and initialization ------------------------------------
    std::vector<DeviceSlab> slabs(static_cast<size_t>(T));
    double* stageCLo = nullptr;   // per-thread lowest / highest plane of c
    double* stageCHi = nullptr;
    double* stageMuLo = nullptr;  // per-thread lowest / highest plane of mu
    double* stageMuHi = nullptr;
    double* recvCBelow = nullptr; // ghost planes received from neighbour ranks
    double* recvCAbove = nullptr;
    double* recvMuBelow = nullptr;
    double* recvMuAbove = nullptr;
    std::vector<double> hostInit;

    if (active) {
        for (int t = 0; t < T; ++t) {
            DeviceSlab& s = slabs[static_cast<size_t>(t)];
            s.device = static_cast<int>((static_cast<size_t>(localRank) * devicesPerRank + t) % numDevices);
            s.offset = nzl * static_cast<size_t>(t) / static_cast<size_t>(T);
            s.n = nzl * static_cast<size_t>(t + 1) / static_cast<size_t>(T) - s.offset;
        }
        // Pinned staging buffers; allocate on this rank's first device
        CUDA_CHECK(cudaSetDevice(slabs[0].device));
        stageCLo = allocPinned(static_cast<size_t>(T) * nxy);
        stageCHi = allocPinned(static_cast<size_t>(T) * nxy);
        stageMuLo = allocPinned(static_cast<size_t>(T) * nxy);
        stageMuHi = allocPinned(static_cast<size_t>(T) * nxy);
        recvCBelow = allocPinned(nxy);
        recvCAbove = allocPinned(nxy);
        recvMuBelow = allocPinned(nxy);
        recvMuAbove = allocPinned(nxy);

        hostInit.resize(nzl * nxy);
        initializeConcentration(hostInit.data(), z0, nzl, nx, ny, nz);

        #pragma omp parallel num_threads(T)
        {
            if (omp_get_num_threads() != T) {
                if (omp_get_thread_num() == 0) {
                    fprintf(stderr, "Rank %d: OpenMP team size mismatch\n", rank);
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
            }
            const int t = omp_get_thread_num();
            DeviceSlab& s = slabs[static_cast<size_t>(t)];
            CUDA_CHECK(cudaSetDevice(s.device));

            const size_t bytes = (s.n + 2) * nxy * sizeof(double);
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&s.cold), bytes));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&s.cnew), bytes));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&s.mu), bytes));
            CUDA_CHECK(cudaStreamCreateWithFlags(&s.sA, cudaStreamNonBlocking));
            CUDA_CHECK(cudaStreamCreateWithFlags(&s.sB, cudaStreamNonBlocking));

            // Upload on the thread's own stream and wait for it. (A legacy-stream
            // cudaMemcpy would not be ordered against the non-blocking streams used below.)
            CUDA_CHECK(cudaMemcpyAsync(s.cold + nxy, hostInit.data() + s.offset * nxy,
                                       s.n * nxy * sizeof(double), cudaMemcpyHostToDevice, s.sA));
            CUDA_CHECK(cudaStreamSynchronize(s.sA));
        }
        hostInit.clear();
        hostInit.shrink_to_fit();
    }

    // ---- Time stepping ----------------------------------------------------
    if (rank == 0) {
        printf("Initializing concentration field...\n");
        printf("Running Cahn-Hilliard simulation...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double tStart = MPI_Wtime();

    if (active) {
        #pragma omp parallel num_threads(T)
        {
            if (omp_get_num_threads() != T) {
                if (omp_get_thread_num() == 0) {
                    fprintf(stderr, "Rank %d: OpenMP team size mismatch\n", rank);
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
            }
            const int t = omp_get_thread_num();
            DeviceSlab& s = slabs[static_cast<size_t>(t)];
            CUDA_CHECK(cudaSetDevice(s.device));

            const size_t n = s.n;
            const bool isFirst = (t == 0);
            const bool isLast = (t == T - 1);

            for (int it = 0; it < iterations; ++it) {
                // Phase 1: export boundary planes of c to host staging.
                // Interior mu (planes 2..n-1) needs no halo, so it is launched
                // now and overlaps with the halo exchange.
                copyPlaneAsync(stageCLo + t * nxy, s.cold + nxy, nxy, cudaMemcpyDeviceToHost, s.sA);
                copyPlaneAsync(stageCHi + t * nxy, s.cold + n * nxy, nxy, cudaMemcpyDeviceToHost, s.sA);
                CUDA_CHECK(cudaStreamSynchronize(s.sA));
                launchChemicalPotential(s.sB, s.cold, s.mu, nx, ny, 2, n > 2 ? n - 2 : 0, P);

                #pragma omp barrier
                #pragma omp master
                exchangeHalos(stageCLo, stageCHi + (T - 1) * nxy,
                              recvCBelow, recvCAbove, planeType, below, above, 100);
                #pragma omp barrier

                // Phase 2: fill concentration ghost planes, then mu on boundary planes.
                if (isFirst) {
                    if (below != MPI_PROC_NULL) copyPlaneAsync(s.cold, recvCBelow, nxy, cudaMemcpyHostToDevice, s.sA);
                    else                        copyPlaneAsync(s.cold, s.cold + nxy, nxy, cudaMemcpyDeviceToDevice, s.sA);
                } else {
                    copyPlaneAsync(s.cold, stageCHi + (t - 1) * nxy, nxy, cudaMemcpyHostToDevice, s.sA);
                }
                if (isLast) {
                    if (above != MPI_PROC_NULL) copyPlaneAsync(s.cold + (n + 1) * nxy, recvCAbove, nxy, cudaMemcpyHostToDevice, s.sA);
                    else                        copyPlaneAsync(s.cold + (n + 1) * nxy, s.cold + n * nxy, nxy, cudaMemcpyDeviceToDevice, s.sA);
                } else {
                    copyPlaneAsync(s.cold + (n + 1) * nxy, stageCLo + (t + 1) * nxy, nxy, cudaMemcpyHostToDevice, s.sA);
                }

                launchChemicalPotential(s.sA, s.cold, s.mu, nx, ny, 1, 1, P);
                if (n > 1) launchChemicalPotential(s.sA, s.cold, s.mu, nx, ny, n, 1, P);
                copyPlaneAsync(stageMuLo + t * nxy, s.mu + nxy, nxy, cudaMemcpyDeviceToHost, s.sA);
                copyPlaneAsync(stageMuHi + t * nxy, s.mu + n * nxy, nxy, cudaMemcpyDeviceToHost, s.sA);
                CUDA_CHECK(cudaStreamSynchronize(s.sA));
                CUDA_CHECK(cudaStreamSynchronize(s.sB));

                #pragma omp barrier

                // Phase 3: update interior cells (needs mu planes 1..n only) while
                // the master exchanges the mu boundary planes.
                launchUpdate(s.sB, s.cold, s.cnew, s.mu, nx, ny, 2, n > 2 ? n - 2 : 0, P);
                #pragma omp master
                exchangeHalos(stageMuLo, stageMuHi + (T - 1) * nxy,
                              recvMuBelow, recvMuAbove, planeType, below, above, 200);
                #pragma omp barrier

                // Phase 4: mu ghost planes, then update boundary planes.
                if (isFirst) {
                    if (below != MPI_PROC_NULL) copyPlaneAsync(s.mu, recvMuBelow, nxy, cudaMemcpyHostToDevice, s.sA);
                    else                        copyPlaneAsync(s.mu, s.mu + nxy, nxy, cudaMemcpyDeviceToDevice, s.sA);
                } else {
                    copyPlaneAsync(s.mu, stageMuHi + (t - 1) * nxy, nxy, cudaMemcpyHostToDevice, s.sA);
                }
                if (isLast) {
                    if (above != MPI_PROC_NULL) copyPlaneAsync(s.mu + (n + 1) * nxy, recvMuAbove, nxy, cudaMemcpyHostToDevice, s.sA);
                    else                        copyPlaneAsync(s.mu + (n + 1) * nxy, s.mu + n * nxy, nxy, cudaMemcpyDeviceToDevice, s.sA);
                } else {
                    copyPlaneAsync(s.mu + (n + 1) * nxy, stageMuLo + (t + 1) * nxy, nxy, cudaMemcpyHostToDevice, s.sA);
                }

                launchUpdate(s.sA, s.cold, s.cnew, s.mu, nx, ny, 1, 1, P);
                if (n > 1) launchUpdate(s.sA, s.cold, s.cnew, s.mu, nx, ny, n, 1, P);
                CUDA_CHECK(cudaStreamSynchronize(s.sA));
                CUDA_CHECK(cudaStreamSynchronize(s.sB));

                std::swap(s.cold, s.cnew);
            }
        }
    }

    const double localElapsed = MPI_Wtime() - tStart;
    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // ---- Results: gather the field on rank 0, then release resources -------
    std::vector<double> outBuf;
    if (wantOutput) {
        outBuf.resize(rank == 0 ? gridSize : nzl * nxy);
    }

    if (active) {
        double* const dst = outBuf.data();
        #pragma omp parallel num_threads(T)
        {
            const int t = omp_get_thread_num();
            DeviceSlab& s = slabs[static_cast<size_t>(t)];
            CUDA_CHECK(cudaSetDevice(s.device));
            if (wantOutput) {
                CUDA_CHECK(cudaMemcpyAsync(dst + s.offset * nxy, s.cold + nxy,
                                           s.n * nxy * sizeof(double), cudaMemcpyDeviceToHost, s.sA));
                CUDA_CHECK(cudaStreamSynchronize(s.sA));
            }
            CUDA_CHECK(cudaFree(s.cold));
            CUDA_CHECK(cudaFree(s.cnew));
            CUDA_CHECK(cudaFree(s.mu));
            CUDA_CHECK(cudaStreamDestroy(s.sA));
            CUDA_CHECK(cudaStreamDestroy(s.sB));
        }
    }

    if (wantOutput) {
        std::vector<int> counts(static_cast<size_t>(numRanks), 0);
        std::vector<int> displs(static_cast<size_t>(numRanks), 0);
        for (int r = 0; r < activeRanks; ++r) {
            counts[static_cast<size_t>(r)] = static_cast<int>(planeStart(r + 1) - planeStart(r));
            displs[static_cast<size_t>(r)] = static_cast<int>(planeStart(r));
        }
        if (rank == 0) {
            MPI_Gatherv(MPI_IN_PLACE, 0, planeType, outBuf.data(), counts.data(), displs.data(),
                        planeType, 0, MPI_COMM_WORLD);
        } else {
            MPI_Gatherv(outBuf.data(), static_cast<int>(nzl), planeType, nullptr, nullptr, nullptr,
                        planeType, 0, MPI_COMM_WORLD);
        }
    }

    if (active) {
        CUDA_CHECK(cudaFreeHost(stageCLo));
        CUDA_CHECK(cudaFreeHost(stageCHi));
        CUDA_CHECK(cudaFreeHost(stageMuLo));
        CUDA_CHECK(cudaFreeHost(stageMuHi));
        CUDA_CHECK(cudaFreeHost(recvCBelow));
        CUDA_CHECK(cudaFreeHost(recvCAbove));
        CUDA_CHECK(cudaFreeHost(recvMuBelow));
        CUDA_CHECK(cudaFreeHost(recvMuAbove));
    }
    MPI_Type_free(&planeType);

    if (rank != 0) {
        return 0;
    }

    // ---- Reporting (rank 0) -----------------------------------------------
    const long durationMs = static_cast<long>(elapsed * 1000.0);
    printf("Computation time: %ld ms\n", durationMs);

    const double cellUpdates = static_cast<double>(gridSize) * iterations;
    const double mcups = cellUpdates / (durationMs / 1000.0) / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

    if (opt.printResults) {
        print_results(outBuf, "Concentration");
    }

    if (opt.validate) {
        printf("Validating result...\n");
        const bool valid = validateResult(outBuf, nx, ny, nz);
        if (valid) {
            printf("Validation: PASSED\n");
            return 0;
        }
        printf("Validation: FAILED\n");
        return 1;
    }

    return 0;
}

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0;
    int numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) fprintf(stderr, "MPI_THREAD_FUNNELED support is required\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    omp_set_dynamic(0);

    Options opt;

    // Parse command line arguments
    int exitCode = 0;
    bool done = false;
    for (int i = 1; i < argc && !done; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            opt.nx = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            opt.ny = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            opt.nz = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            opt.iterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            opt.validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            opt.printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            done = true;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            exitCode = 1;
            done = true;
        }
    }

    if (!done) {
        if (opt.ny == 0) opt.ny = opt.nx;
        if (opt.nz == 0) opt.nz = opt.nx;

        if (opt.nx == 0 || opt.ny == 0 || opt.nz == 0) {
            if (rank == 0) fprintf(stderr, "Grid dimensions must be positive\n");
            exitCode = 1;
        } else if (opt.nx * opt.ny > static_cast<size_t>(INT_MAX) || opt.nz > static_cast<size_t>(INT_MAX)) {
            if (rank == 0) fprintf(stderr, "Grid dimensions exceed supported MPI count limits\n");
            exitCode = 1;
        } else {
            if (rank == 0) {
                printf("Cahn-Hilliard Phase Separation Benchmark\n");
                printf("Grid size: %zu x %zu x %zu\n", opt.nx, opt.ny, opt.nz);
                printf("Time steps: %d\n", opt.iterations);
                printf("Validation: %s\n", opt.validate ? "enabled" : "disabled");
            }
            exitCode = runBenchmark(opt, rank, numRanks);
        }
    }

    MPI_Finalize();
    return exitCode;
}
