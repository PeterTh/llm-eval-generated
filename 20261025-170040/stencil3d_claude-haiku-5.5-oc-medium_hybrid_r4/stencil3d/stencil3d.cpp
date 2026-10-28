// Hybrid MPI + OpenMP + CUDA 3D 7-point stencil benchmark.
//
// Decomposition:
//   * MPI:    1D slab decomposition along Z. Each rank owns a contiguous range
//             of Z-planes and exchanges one halo plane with each neighbour per
//             iteration. Z-planes are contiguous in memory, so halos need no
//             packing.
//   * CUDA:   each rank binds to one GPU (chosen by node-local rank). The grid
//             stays resident on the device for all iterations. Per iteration the
//             boundary planes are computed first on a high-priority stream and
//             copied to pinned host buffers; the halo exchange then overlaps with
//             the interior-plane kernel running on a second stream.
//   * OpenMP: host-side validation scans (NaN/Inf check, min/max) on rank 0.
//
// The per-cell arithmetic and its evaluation order are identical to the serial
// reference, so results (including the printed hash/sum) are bit-identical.

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

using Real = double;

#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        cudaError_t err_ = (call);                                                       \
        if (err_ != cudaSuccess) {                                                       \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,             \
                    cudaGetErrorString(err_));                                           \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                \
        }                                                                                \
    } while (0)

#define ROOT_PRINTF(...)                                                                 \
    do {                                                                                 \
        if (rank == 0) printf(__VA_ARGS__);                                              \
    } while (0)

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Initial grid value for a local element. zl is the local plane index (0 is the
// lower halo plane, zl = nzl+1 is the upper halo plane); z0 is the global Z of the
// first owned plane. Halo planes outside the global domain are zero (never read).
__global__ void initKernel(Real* __restrict__ grid, const size_t nx, const size_t ny,
                           const size_t nz, const size_t z0, const size_t localElems) {
    const size_t nxy = nx * ny;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < localElems; i += stride) {
        const size_t zl = i / nxy;
        const size_t rem = i - zl * nxy;
        const long long gz = static_cast<long long>(z0) + static_cast<long long>(zl) - 1;
        if (gz < 0 || gz >= static_cast<long long>(nz)) {
            grid[i] = 0.0;
        } else {
            const size_t gidx = static_cast<size_t>(gz) * nxy + rem;  // == idx3(x, y, gz, nx, ny)
            grid[i] = static_cast<Real>(gidx % 19) * 1.0;
        }
    }
}

// 7-point stencil over local planes [zlFirst, zlFirst + zCount).
// Global boundary cells are copied; all other cells are averaged.
__global__ void stencilKernel(const Real* __restrict__ in, Real* __restrict__ out,
                              const size_t nx, const size_t ny, const size_t nz, const size_t z0,
                              const size_t zlFirst, const size_t zCount) {
    const size_t nxy = nx * ny;
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (x >= nx) return;
    const size_t yStride = static_cast<size_t>(gridDim.y) * blockDim.y;
    for (size_t zi = blockIdx.z; zi < zCount; zi += gridDim.z) {
        const size_t zl = zlFirst + zi;
        const size_t gz = z0 + zl - 1;
        const bool zBoundary = (gz == 0) || (gz == nz - 1);
        for (size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y; y < ny; y += yStride) {
            const size_t idx = zl * nxy + y * nx + x;
            if (zBoundary || x == 0 || x == nx - 1 || y == 0 || y == ny - 1) {
                out[idx] = in[idx];
                continue;
            }
            const Real center = in[idx];
            const Real left = in[idx - 1];
            const Real right = in[idx + 1];
            const Real front = in[idx - nx];
            const Real back = in[idx + nx];
            const Real bottom = in[idx - nxy];
            const Real top = in[idx + nxy];
            // Simple averaging stencil (same operation order as the reference)
            out[idx] = (center + left + right + front + back + bottom + top) / 7.0;
        }
    }
}

static void launchStencil(cudaStream_t stream, const Real* in, Real* out, const size_t nx, const size_t ny,
                          const size_t nz, const size_t z0, const size_t zlFirst, const size_t zCount) {
    if (zCount == 0 || nx == 0 || ny == 0) return;
    const dim3 block(32, 4, 1);
    const dim3 grid(static_cast<unsigned>((nx + block.x - 1) / block.x),
                    static_cast<unsigned>(std::min<size_t>((ny + block.y - 1) / block.y, 65535)),
                    static_cast<unsigned>(std::min<size_t>(zCount, 65535)));
    stencilKernel<<<grid, block, 0, stream>>>(in, out, nx, ny, nz, z0, zlFirst, zCount);
    CUDA_CHECK(cudaGetLastError());
}

bool validateResult(const std::vector<Real>& grid, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    const long long n = static_cast<long long>(grid.size());

    // 1. No NaN or Inf values
    bool hasNonFinite = false;
#pragma omp parallel for reduction(|| : hasNonFinite)
    for (long long i = 0; i < n; ++i) {
        hasNonFinite = hasNonFinite || std::isnan(grid[i]) || std::isinf(grid[i]);
    }
    if (hasNonFinite) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    // 2. Values should be reasonable (bounded)
    Real minVal = grid[0];
    Real maxVal = grid[0];
#pragma omp parallel for reduction(min : minVal) reduction(max : maxVal)
    for (long long i = 0; i < n; ++i) {
        minVal = std::min(minVal, grid[i]);
        maxVal = std::max(maxVal, grid[i]);
    }

    printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);

    // After averaging, values should be somewhat bounded
    if (maxVal > 1e6 || minVal < -1e6) {
        printf("Validation failed: values out of expected range\n");
        return false;
    }

    // 3. Boundary values should not change significantly
    // (they are copied, so they should be close to initial values)

    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -x <num>     Grid size in X dimension (default: 128)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of iterations (default: 10)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

// Z-slab owned by rank r when `nActive` ranks share `nz` planes.
struct Slab {
    size_t z0;
    size_t nzl;
};

static Slab slabFor(const int r, const int nActive, const size_t nz) {
    if (r >= nActive) return {nz, 0};
    const size_t a = static_cast<size_t>(r) * nz / nActive;
    const size_t b = static_cast<size_t>(r + 1) * nz / nActive;
    return {a, b - a};
}

static int runBenchmark(int argc, char** argv, const int rank, const int size) {
    size_t nx = 128;
    size_t ny = 0;  // Will be set to nx if not specified
    size_t nz = 0;  // Will be set to nx if not specified
    int iterations = 10;
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
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    ROOT_PRINTF("3D Stencil Benchmark\n");
    ROOT_PRINTF("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    ROOT_PRINTF("Iterations: %d\n", iterations);
    ROOT_PRINTF("Validation: %s\n", validate ? "enabled" : "disabled");

    const size_t nxy = nx * ny;
    const size_t gridSize = nxy * nz;
    if (nxy > static_cast<size_t>(INT_MAX)) {
        if (rank == 0) fprintf(stderr, "Plane size nx*ny exceeds supported limit\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const size_t planeBytes = nxy * sizeof(Real);

    // --- Bind this rank to a GPU (round-robin over node-local ranks) ---
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_free(&localComm);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        if (rank == 0) fprintf(stderr, "No CUDA devices available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));

    // --- Domain decomposition along Z ---
    const int nActive = static_cast<int>(std::max<size_t>(1, std::min<size_t>(static_cast<size_t>(size), nz)));
    const Slab mine = slabFor(rank, nActive, nz);
    const size_t z0 = mine.z0;
    const size_t nzl = mine.nzl;
    const int upRank = (rank + 1 < nActive) ? rank + 1 : MPI_PROC_NULL;
    const int downRank = (rank > 0 && rank < nActive) ? rank - 1 : MPI_PROC_NULL;

    MPI_Datatype planeType;
    MPI_Type_contiguous(static_cast<int>(nxy), MPI_DOUBLE, &planeType);
    MPI_Type_commit(&planeType);

    // --- Device / host resources ---
    Real* d1 = nullptr;  // local grid (double buffer), layout: [halo | owned | halo]
    Real* d2 = nullptr;
    Real* hSendLo = nullptr;
    Real* hSendHi = nullptr;
    Real* hRecvLo = nullptr;
    Real* hRecvHi = nullptr;

    int leastPrio = 0, greatestPrio = 0;
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&leastPrio, &greatestPrio));
    cudaStream_t sBnd, sInt;
    CUDA_CHECK(cudaStreamCreateWithPriority(&sBnd, cudaStreamNonBlocking, greatestPrio));
    CUDA_CHECK(cudaStreamCreateWithPriority(&sInt, cudaStreamNonBlocking, leastPrio));

    ROOT_PRINTF("Initializing grid...\n");
    if (nzl > 0) {
        const size_t localElems = (nzl + 2) * nxy;
        CUDA_CHECK(cudaMalloc(&d1, localElems * sizeof(Real)));
        CUDA_CHECK(cudaMalloc(&d2, localElems * sizeof(Real)));
        CUDA_CHECK(cudaMemset(d2, 0, localElems * sizeof(Real)));

        const int blocks = static_cast<int>(std::min<size_t>((localElems + 255) / 256, 65535));
        initKernel<<<blocks, 256>>>(d1, nx, ny, nz, z0, localElems);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMallocHost(&hSendLo, planeBytes));
        CUDA_CHECK(cudaMallocHost(&hSendHi, planeBytes));
        CUDA_CHECK(cudaMallocHost(&hRecvLo, planeBytes));
        CUDA_CHECK(cudaMallocHost(&hRecvHi, planeBytes));
    }

    ROOT_PRINTF("Running stencil computation...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iterations; ++iter) {
        if (nzl == 0) continue;  // Idle rank (more ranks than Z-planes)
        const Real* in = (iter % 2 == 0) ? d1 : d2;
        Real* out = (iter % 2 == 0) ? d2 : d1;

        // 1) Boundary planes first (they feed the halo exchange).
        launchStencil(sBnd, in, out, nx, ny, nz, z0, 1, 1);
        if (nzl > 1) launchStencil(sBnd, in, out, nx, ny, nz, z0, nzl, 1);
        CUDA_CHECK(cudaMemcpyAsync(hSendLo, out + nxy, planeBytes, cudaMemcpyDeviceToHost, sBnd));
        CUDA_CHECK(cudaMemcpyAsync(hSendHi, out + nzl * nxy, planeBytes, cudaMemcpyDeviceToHost, sBnd));

        // 2) Interior planes overlap with the halo exchange.
        if (nzl > 2) launchStencil(sInt, in, out, nx, ny, nz, z0, 2, nzl - 2);

        CUDA_CHECK(cudaStreamSynchronize(sBnd));

        // 3) Halo exchange (tag 0: data moving up, tag 1: data moving down).
        MPI_Request reqs[4];
        MPI_Irecv(hRecvLo, 1, planeType, downRank, 0, MPI_COMM_WORLD, &reqs[0]);
        MPI_Irecv(hRecvHi, 1, planeType, upRank, 1, MPI_COMM_WORLD, &reqs[1]);
        MPI_Isend(hSendHi, 1, planeType, upRank, 0, MPI_COMM_WORLD, &reqs[2]);
        MPI_Isend(hSendLo, 1, planeType, downRank, 1, MPI_COMM_WORLD, &reqs[3]);
        MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

        // 4) Write received neighbour planes into the halo planes of 'out'.
        if (downRank != MPI_PROC_NULL)
            CUDA_CHECK(cudaMemcpyAsync(out, hRecvLo, planeBytes, cudaMemcpyHostToDevice, sBnd));
        if (upRank != MPI_PROC_NULL)
            CUDA_CHECK(cudaMemcpyAsync(out + (nzl + 1) * nxy, hRecvHi, planeBytes, cudaMemcpyHostToDevice, sBnd));

        // 5) Iteration barrier: 'out' must be complete before it becomes 'in'.
        CUDA_CHECK(cudaStreamSynchronize(sBnd));
        CUDA_CHECK(cudaStreamSynchronize(sInt));
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    long local_ms = duration.count();
    long max_ms = local_ms;
    MPI_Reduce(&local_ms, &max_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    ROOT_PRINTF("Computation time: %ld ms\n", max_ms);

    // Calculate performance metrics
    double cellUpdates = (double)((nx - 2) * (ny - 2) * (nz - 2)) * iterations;
    double mcups = cellUpdates / (max_ms / 1000.0) / 1e6;  // Million cell updates per second
    ROOT_PRINTF("Performance: %.3f MCellUpdates/s\n", mcups);

    // Final grid lives in grid1 for an even iteration count, otherwise grid2.
    const Real* finalDev = (iterations % 2 == 0) ? d1 : d2;
    int rc = 0;

    if (printResults || validate) {
        // Gather the full final grid (global Z order) onto rank 0.
        std::vector<Real> finalGrid;
        std::vector<int> counts(size, 0), displs(size, 0);
        for (int r = 0; r < size; ++r) {
            const Slab s = slabFor(r, nActive, nz);
            counts[r] = static_cast<int>(s.nzl);
            displs[r] = static_cast<int>(s.z0);
        }

        if (rank == 0) {
            finalGrid.resize(gridSize);
            if (nzl > 0)
                CUDA_CHECK(cudaMemcpy(finalGrid.data(), finalDev + nxy, nzl * planeBytes, cudaMemcpyDeviceToHost));
            MPI_Gatherv(MPI_IN_PLACE, 0, planeType, finalGrid.data(), counts.data(), displs.data(), planeType,
                        0, MPI_COMM_WORLD);
        } else {
            std::vector<Real> stage(nzl * nxy);
            if (nzl > 0)
                CUDA_CHECK(cudaMemcpy(stage.data(), finalDev + nxy, nzl * planeBytes, cudaMemcpyDeviceToHost));
            MPI_Gatherv(stage.data(), static_cast<int>(nzl), planeType, nullptr, nullptr, nullptr, planeType, 0,
                        MPI_COMM_WORLD);
        }

        if (rank == 0) {
            // Print results for external validation
            if (printResults) print_results(finalGrid, "Grid");

            // Validation
            if (validate) {
                printf("Validating result...\n");
                bool valid = validateResult(finalGrid, nx, ny, nz);
                if (valid) {
                    printf("Validation: PASSED\n");
                } else {
                    printf("Validation: FAILED\n");
                    rc = 1;
                }
            }
        }
    }

    // --- Cleanup ---
    if (d1) CUDA_CHECK(cudaFree(d1));
    if (d2) CUDA_CHECK(cudaFree(d2));
    if (hSendLo) CUDA_CHECK(cudaFreeHost(hSendLo));
    if (hSendHi) CUDA_CHECK(cudaFreeHost(hSendHi));
    if (hRecvLo) CUDA_CHECK(cudaFreeHost(hRecvLo));
    if (hRecvHi) CUDA_CHECK(cudaFreeHost(hRecvHi));
    CUDA_CHECK(cudaStreamDestroy(sBnd));
    CUDA_CHECK(cudaStreamDestroy(sInt));
    MPI_Type_free(&planeType);

    return rc;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int rc = runBenchmark(argc, argv, rank, size);

    MPI_Finalize();
    return rc;
}
