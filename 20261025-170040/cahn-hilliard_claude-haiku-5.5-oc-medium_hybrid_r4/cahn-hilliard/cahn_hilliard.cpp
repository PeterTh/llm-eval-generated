// Cahn-Hilliard phase separation benchmark.
//
// Parallelization (hybrid, all three always used):
//   * MPI:    1D slab decomposition along z across ranks. Each slab is contiguous
//             in memory, so gathering the result is a single MPI_Gatherv. Each rank
//             exchanges one ghost plane with each z-neighbour per field update.
//   * CUDA:   each rank drives one GPU (rank mod device count on the node). The
//             chemical potential and the update are stencil kernels. Interior
//             planes are computed while the halo exchange runs on a second stream.
//   * OpenMP: host-side field initialization and the result validation scan.
//
// Numerical semantics match the serial original: same stencil, same clamped
// (Neumann-like) boundaries, same evaluation order per cell. Each cell is computed
// by the same expression regardless of decomposition, so the result does not
// depend on the number of ranks or GPUs.

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

#define CHECK_CUDA(call)                                                                 \
    do {                                                                                 \
        cudaError_t err__ = (call);                                                      \
        if (err__ != cudaSuccess) {                                                      \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,             \
                    cudaGetErrorString(err__));                                          \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                \
        }                                                                                \
    } while (0)

#define CHECK_MPI(call)                                                                  \
    do {                                                                                 \
        int err__ = (call);                                                              \
        if (err__ != MPI_SUCCESS) {                                                      \
            fprintf(stderr, "MPI error at %s:%d\n", __FILE__, __LINE__);                 \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                \
        }                                                                                \
    } while (0)

namespace {

constexpr int TAG_TO_LOWER = 0;  // message sent to the rank below (z - 1)
constexpr int TAG_TO_UPPER = 1;  // message sent to the rank above (z + 1)

// Global problem geometry plus this rank's slab.
struct Grid {
    size_t nx, ny, nz;  // global dimensions
    size_t z0;          // global z index of the first owned plane
    size_t nzl;         // number of owned planes
};

struct Phys {
    double dx2, dy2, dz2;
    double gamma, e_AA, e_BB, e_AB, D, dt;
};

// Device buffers hold nzl + 2 planes: plane 0 and plane nzl+1 are ghost planes,
// owned plane zl is stored at buffer index zl + 1.

// Laplacian with clamped boundaries, evaluated on a device buffer.
// (p) is the buffer index of the plane, (pzp)/(pzn) the buffer indices of the
// z-neighbours (equal to p at a global boundary, which clamps).
__device__ __forceinline__ double laplacianDev(const double* __restrict__ f, const size_t x, const size_t y,
                                               const size_t p, const size_t pzp, const size_t pzn,
                                               const size_t nx, const size_t ny,
                                               const double dx2, const double dy2, const double dz2) {
    const size_t S = nx * ny;
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;

    const size_t rowC = p * S + y * nx;
    const double cc = f[rowC + x];
    const double cxx = (f[rowC + xp] + f[rowC + xn] - 2.0 * cc) / dx2;
    const double cyy = (f[p * S + yp * nx + x] + f[p * S + yn * nx + x] - 2.0 * cc) / dy2;
    const double czz = (f[pzp * S + y * nx + x] + f[pzn * S + y * nx + x] - 2.0 * cc) / dz2;
    return cxx + cyy + czz;
}

// Chemical potential over owned planes [zlBegin, zlBegin + count).
__global__ void chemicalPotentialKernel(const double* __restrict__ c, double* __restrict__ mu, const Grid g,
                                        const Phys P, const size_t zlBegin, const size_t count) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= g.nx || y >= g.ny) return;
    const size_t S = g.nx * g.ny;
    for (size_t k = blockIdx.z; k < count; k += gridDim.z) {
        const size_t zl = zlBegin + k;
        const size_t p = zl + 1;
        const size_t gz = g.z0 + zl;
        const size_t pzp = (gz < g.nz - 1) ? p + 1 : p;
        const size_t pzn = (gz > 0) ? p - 1 : p;
        const size_t idx = p * S + y * g.nx + x;
        const double cv = c[idx];
        mu[idx] = 4.5 * ((cv + 1.0) * P.e_AA + (cv - 1.0) * P.e_BB - 2.0 * cv * P.e_AB)
                  + 3.0 * cv + cv * cv * cv
                  - P.gamma * laplacianDev(c, x, y, p, pzp, pzn, g.nx, g.ny, P.dx2, P.dy2, P.dz2);
    }
}

// Cahn-Hilliard update over owned planes [zlBegin, zlBegin + count).
__global__ void updateKernel(const double* __restrict__ mu, const double* __restrict__ cold,
                             double* __restrict__ cnew, const Grid g, const Phys P,
                             const size_t zlBegin, const size_t count) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= g.nx || y >= g.ny) return;
    const size_t S = g.nx * g.ny;
    for (size_t k = blockIdx.z; k < count; k += gridDim.z) {
        const size_t zl = zlBegin + k;
        const size_t p = zl + 1;
        const size_t gz = g.z0 + zl;
        const size_t pzp = (gz < g.nz - 1) ? p + 1 : p;
        const size_t pzn = (gz > 0) ? p - 1 : p;
        const size_t idx = p * S + y * g.nx + x;
        cnew[idx] = cold[idx] + P.dt * P.D * laplacianDev(mu, x, y, p, pzp, pzn, g.nx, g.ny, P.dx2, P.dy2, P.dz2);
    }
}

void launchChemicalPotential(const double* c, double* mu, const Grid& g, const Phys& P,
                             size_t zlBegin, size_t count, cudaStream_t s) {
    if (count == 0) return;
    const dim3 block(32, 8, 1);
    const dim3 grid(static_cast<unsigned>((g.nx + block.x - 1) / block.x),
                    static_cast<unsigned>((g.ny + block.y - 1) / block.y),
                    static_cast<unsigned>(std::min<size_t>(count, 65535)));
    chemicalPotentialKernel<<<grid, block, 0, s>>>(c, mu, g, P, zlBegin, count);
    CHECK_CUDA(cudaGetLastError());
}

void launchUpdate(const double* mu, const double* cold, double* cnew, const Grid& g, const Phys& P,
                  size_t zlBegin, size_t count, cudaStream_t s) {
    if (count == 0) return;
    const dim3 block(32, 8, 1);
    const dim3 grid(static_cast<unsigned>((g.nx + block.x - 1) / block.x),
                    static_cast<unsigned>((g.ny + block.y - 1) / block.y),
                    static_cast<unsigned>(std::min<size_t>(count, 65535)));
    updateKernel<<<grid, block, 0, s>>>(mu, cold, cnew, g, P, zlBegin, count);
    CHECK_CUDA(cudaGetLastError());
}

// Pinned host staging buffers for ghost-plane exchange.
struct HaloBuffers {
    double* sendLo = nullptr;
    double* sendHi = nullptr;
    double* recvLo = nullptr;
    double* recvHi = nullptr;
};

// Exchanges the boundary planes of a device field with the z-neighbours.
// Global boundaries are not exchanged; the kernels clamp there instead.
// Synchronizes the copy stream before returning; the compute stream is not touched.
void exchangeHalo(double* dev, const Grid& g, size_t S, int rank, int size, MPI_Comm comm,
                  MPI_Datatype planeType, cudaStream_t copyStream, const HaloBuffers& hb) {
    const bool hasLower = rank > 0;
    const bool hasUpper = rank < size - 1;
    const size_t planeBytes = S * sizeof(double);

    if (hasLower) CHECK_CUDA(cudaMemcpyAsync(hb.sendLo, dev + S * 1, planeBytes, cudaMemcpyDeviceToHost, copyStream));
    if (hasUpper) CHECK_CUDA(cudaMemcpyAsync(hb.sendHi, dev + S * g.nzl, planeBytes, cudaMemcpyDeviceToHost, copyStream));
    CHECK_CUDA(cudaStreamSynchronize(copyStream));

    MPI_Request reqs[4];
    int n = 0;
    if (hasLower) {
        CHECK_MPI(MPI_Irecv(hb.recvLo, 1, planeType, rank - 1, TAG_TO_UPPER, comm, &reqs[n++]));
        CHECK_MPI(MPI_Isend(hb.sendLo, 1, planeType, rank - 1, TAG_TO_LOWER, comm, &reqs[n++]));
    }
    if (hasUpper) {
        CHECK_MPI(MPI_Irecv(hb.recvHi, 1, planeType, rank + 1, TAG_TO_LOWER, comm, &reqs[n++]));
        CHECK_MPI(MPI_Isend(hb.sendHi, 1, planeType, rank + 1, TAG_TO_UPPER, comm, &reqs[n++]));
    }
    CHECK_MPI(MPI_Waitall(n, reqs, MPI_STATUSES_IGNORE));

    if (hasLower) CHECK_CUDA(cudaMemcpyAsync(dev + 0, hb.recvLo, planeBytes, cudaMemcpyHostToDevice, copyStream));
    if (hasUpper) CHECK_CUDA(cudaMemcpyAsync(dev + S * (g.nzl + 1), hb.recvHi, planeBytes, cudaMemcpyHostToDevice, copyStream));
    CHECK_CUDA(cudaStreamSynchronize(copyStream));
}

// Initial concentration on owned planes. Uses the global linear index, so the
// field is identical for any decomposition. Host work parallelized with OpenMP.
void initializeConcentration(std::vector<double>& c, const Grid& g) {
    const size_t nx = g.nx, ny = g.ny;
    const size_t S = nx * ny;
    const size_t vol = g.nx * g.ny * g.nz;
    const long long nzl = static_cast<long long>(g.nzl);
    const long long nyl = static_cast<long long>(ny);

#pragma omp parallel for collapse(2) schedule(static)
    for (long long zl = 0; zl < nzl; ++zl) {
        for (long long y = 0; y < nyl; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t local = static_cast<size_t>(zl) * S + static_cast<size_t>(y) * nx + x;
                const size_t linear_id = (g.z0 + static_cast<size_t>(zl)) * S + static_cast<size_t>(y) * nx + x;
                // Pseudo-random value in [-1, 1]
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[local] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

// Validation on the full (gathered) field. Scan parallelized with OpenMP.
bool validateResult(const std::vector<double>& c) {
    const long long n = static_cast<long long>(c.size());

    bool bad = false;
#pragma omp parallel for reduction(|| : bad)
    for (long long i = 0; i < n; ++i) {
        bad = bad || std::isnan(c[i]) || std::isinf(c[i]);
    }
    if (bad) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    double minVal = c[0];
    double maxVal = c[0];
#pragma omp parallel for reduction(min : minVal) reduction(max : maxVal)
    for (long long i = 0; i < n; ++i) {
        minVal = std::min(minVal, c[i]);
        maxVal = std::max(maxVal, c[i]);
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

[[noreturn]] void fatal(const char* msg) {
    fprintf(stderr, "%s\n", msg);
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();  // not reached
}

}  // namespace

int main(int argc, char** argv) {
    int provided = 0;
    CHECK_MPI(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));

    int worldRank = 0, worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    const bool isRoot = (worldRank == 0);

    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (identical on all ranks)
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
            if (isRoot) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (isRoot) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    if (nx == 0 || ny == 0 || nz == 0) {
        if (isRoot) printf("Grid dimensions must be positive\n");
        MPI_Finalize();
        return 1;
    }

    if (isRoot) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Bind this rank to a GPU on its node and split the node's cores among its ranks.
    MPI_Comm nodeComm;
    CHECK_MPI(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, worldRank, MPI_INFO_NULL, &nodeComm));
    int localRank = 0, localSize = 1;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_size(nodeComm, &localSize);
    MPI_Comm_free(&nodeComm);

    int numDevices = 0;
    if (cudaGetDeviceCount(&numDevices) != cudaSuccess || numDevices == 0) {
        fatal("No CUDA devices available");
    }
    CHECK_CUDA(cudaSetDevice(localRank % numDevices));

    if (std::getenv("OMP_NUM_THREADS") == nullptr) {
        omp_set_num_threads(std::max(1, omp_get_num_procs() / localSize));
    }

    // Ranks beyond the number of z-planes have no work; they leave the computation.
    const int nActive = static_cast<int>(std::min<size_t>(static_cast<size_t>(worldSize), nz));
    MPI_Comm comm;
    CHECK_MPI(MPI_Comm_split(MPI_COMM_WORLD, worldRank < nActive ? 0 : MPI_UNDEFINED, worldRank, &comm));
    if (comm == MPI_COMM_NULL) {
        MPI_Finalize();
        return 0;
    }
    int rank = 0, size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Slab decomposition along z.
    Grid g{};
    g.nx = nx;
    g.ny = ny;
    g.nz = nz;
    {
        const size_t base = nz / static_cast<size_t>(size);
        const size_t rem = nz % static_cast<size_t>(size);
        const size_t r = static_cast<size_t>(rank);
        g.nzl = base + (r < rem ? 1 : 0);
        g.z0 = r * base + std::min(r, rem);
    }

    const size_t S = nx * ny;
    const size_t gridSize = S * nz;
    if (S > static_cast<size_t>(INT_MAX)) fatal("Plane size exceeds MPI count limit");

    const Phys P{1.0 * 1.0, 1.0 * 1.0, 1.0 * 1.0,   // dx*dx, dy*dy, dz*dz with dx = dy = dz = 1
                 0.5,                                 // gamma
                 -(2.0 / 9.0), -(2.0 / 9.0), (2.0 / 9.0),  // e_AA, e_BB, e_AB
                 1.0,                                 // D
                 0.01};                               // dt

    MPI_Datatype planeType;
    CHECK_MPI(MPI_Type_contiguous(static_cast<int>(S), MPI_DOUBLE, &planeType));
    CHECK_MPI(MPI_Type_commit(&planeType));

    // Device storage: nzl owned planes + 2 ghost planes per field.
    const size_t bufBytes = (g.nzl + 2) * S * sizeof(double);
    double *dCold = nullptr, *dCnew = nullptr, *dMu = nullptr;
    CHECK_CUDA(cudaMalloc(&dCold, bufBytes));
    CHECK_CUDA(cudaMalloc(&dCnew, bufBytes));
    CHECK_CUDA(cudaMalloc(&dMu, bufBytes));
    // All device work is issued on explicit non-blocking streams. Legacy-stream calls
    // (cudaMemcpy/cudaMemset without Async) are not ordered against these streams, so
    // they are avoided entirely to keep the data dependencies explicit.
    cudaStream_t compute, copy;
    CHECK_CUDA(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
    CHECK_CUDA(cudaStreamCreateWithFlags(&copy, cudaStreamNonBlocking));

    CHECK_CUDA(cudaMemsetAsync(dCold, 0, bufBytes, compute));
    CHECK_CUDA(cudaMemsetAsync(dCnew, 0, bufBytes, compute));
    CHECK_CUDA(cudaMemsetAsync(dMu, 0, bufBytes, compute));
    CHECK_CUDA(cudaStreamSynchronize(compute));

    HaloBuffers hb;
    CHECK_CUDA(cudaMallocHost(&hb.sendLo, S * sizeof(double)));
    CHECK_CUDA(cudaMallocHost(&hb.sendHi, S * sizeof(double)));
    CHECK_CUDA(cudaMallocHost(&hb.recvLo, S * sizeof(double)));
    CHECK_CUDA(cudaMallocHost(&hb.recvHi, S * sizeof(double)));

    if (isRoot) {
        printf("Parallelization: MPI ranks=%d, OpenMP threads/rank=%d, CUDA device/rank=1\n",
               size, omp_get_max_threads());
        printf("Initializing concentration field...\n");
    }

    {
        std::vector<double> hinit(g.nzl * S);
        initializeConcentration(hinit, g);
        CHECK_CUDA(cudaMemcpyAsync(dCold + S, hinit.data(), g.nzl * S * sizeof(double), cudaMemcpyHostToDevice, compute));
        CHECK_CUDA(cudaStreamSynchronize(compute));
    }

    if (isRoot) printf("Running Cahn-Hilliard simulation...\n");

    // Interior owned planes (no ghost reads) vs boundary owned planes (need halos).
    const size_t interiorCount = (g.nzl > 2) ? g.nzl - 2 : 0;
    const size_t lastPlane = g.nzl - 1;

    CHECK_MPI(MPI_Barrier(comm));
    const double tStart = MPI_Wtime();

    for (int t = 0; t < iterations; ++t) {
        // Current cold field is ready (previous update finished).
        CHECK_CUDA(cudaStreamSynchronize(compute));

        // mu interior: needs no halo; overlaps the cold-field halo exchange.
        launchChemicalPotential(dCold, dMu, g, P, 1, interiorCount, compute);
        exchangeHalo(dCold, g, S, rank, size, comm, planeType, copy, hb);

        // mu boundary planes (need cold ghosts).
        launchChemicalPotential(dCold, dMu, g, P, 0, 1, compute);
        if (g.nzl > 1) launchChemicalPotential(dCold, dMu, g, P, lastPlane, 1, compute);
        CHECK_CUDA(cudaStreamSynchronize(compute));

        // Update interior: needs only owned mu planes; overlaps the mu halo exchange.
        launchUpdate(dMu, dCold, dCnew, g, P, 1, interiorCount, compute);
        exchangeHalo(dMu, g, S, rank, size, comm, planeType, copy, hb);

        // Update boundary planes (need mu ghosts).
        launchUpdate(dMu, dCold, dCnew, g, P, 0, 1, compute);
        if (g.nzl > 1) launchUpdate(dMu, dCold, dCnew, g, P, lastPlane, 1, compute);

        std::swap(dCold, dCnew);
    }
    CHECK_CUDA(cudaStreamSynchronize(compute));

    CHECK_MPI(MPI_Barrier(comm));
    const double elapsed = MPI_Wtime() - tStart;
    double maxElapsed = 0.0;
    CHECK_MPI(MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm));

    if (isRoot) {
        const long long durationMs = static_cast<long long>(maxElapsed * 1000.0);
        printf("Computation time: %lld ms\n", durationMs);

        const double cellUpdates = (double)gridSize * iterations;
        const double mcups = cellUpdates / (durationMs / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Gather the final field onto rank 0 only when it is needed for output or validation.
    int valid = 1;
    if (printResults || validate) {
        std::vector<double> hloc(g.nzl * S);
        CHECK_CUDA(cudaMemcpyAsync(hloc.data(), dCold + S, g.nzl * S * sizeof(double), cudaMemcpyDeviceToHost, compute));
        CHECK_CUDA(cudaStreamSynchronize(compute));

        std::vector<double> full;
        std::vector<int> counts, displs;
        if (rank == 0) {
            full.resize(gridSize);
            counts.resize(size);
            displs.resize(size);
            const size_t base = nz / static_cast<size_t>(size);
            const size_t rem = nz % static_cast<size_t>(size);
            for (int r = 0; r < size; ++r) {
                const size_t rr = static_cast<size_t>(r);
                counts[r] = static_cast<int>(base + (rr < rem ? 1 : 0));
                displs[r] = static_cast<int>(rr * base + std::min(rr, rem));
            }
        }
        CHECK_MPI(MPI_Gatherv(hloc.data(), static_cast<int>(g.nzl), planeType,
                              full.data(), counts.data(), displs.data(), planeType, 0, comm));

        if (rank == 0) {
            // Results for external validation
            if (printResults) {
                print_results(full, "Concentration");
            }

            if (validate) {
                printf("Validating result...\n");
                const bool ok = validateResult(full);
                if (ok) {
                    printf("Validation: PASSED\n");
                } else {
                    printf("Validation: FAILED\n");
                    valid = 0;
                }
            }
        }
        if (validate) {
            CHECK_MPI(MPI_Bcast(&valid, 1, MPI_INT, 0, comm));
        }
    }

    CHECK_CUDA(cudaFreeHost(hb.sendLo));
    CHECK_CUDA(cudaFreeHost(hb.sendHi));
    CHECK_CUDA(cudaFreeHost(hb.recvLo));
    CHECK_CUDA(cudaFreeHost(hb.recvHi));
    CHECK_CUDA(cudaStreamDestroy(compute));
    CHECK_CUDA(cudaStreamDestroy(copy));
    CHECK_CUDA(cudaFree(dCold));
    CHECK_CUDA(cudaFree(dCnew));
    CHECK_CUDA(cudaFree(dMu));
    MPI_Type_free(&planeType);
    MPI_Comm_free(&comm);
    MPI_Finalize();

    return valid ? 0 : 1;
}
