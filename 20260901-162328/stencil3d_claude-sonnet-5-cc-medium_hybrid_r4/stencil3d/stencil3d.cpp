#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

using Real = double;

#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        cudaError_t err__ = (call);                                                      \
        if (err__ != cudaSuccess) {                                                      \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,             \
                    cudaGetErrorString(err__));                                           \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                \
        }                                                                                 \
    } while (0)

// 3D index calculation (host, within a fully contiguous nx*ny*nz buffer)
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Initializes the local (padded with 1 ghost plane on each side) slab owned by this rank.
// zGlobalStart is the global z coordinate corresponding to local plane index 0.
void initializeGridLocal(std::vector<Real>& localGrid, const size_t nx, const size_t ny,
                          const size_t nzLocalPadded, const long zGlobalStart, const size_t nzGlobal) {
    const size_t planeSize = nx * ny;
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t lz = 0; lz < nzLocalPadded; ++lz) {
        for (size_t y = 0; y < ny; ++y) {
            const long z = zGlobalStart + static_cast<long>(lz);
            if (z < 0 || z >= static_cast<long>(nzGlobal)) {
                continue;  // unused padding beyond the global domain, never read by the kernel
            }
            for (size_t x = 0; x < nx; ++x) {
                const size_t localIdx = lz * planeSize + y * nx + x;
                const size_t globalIdx = idx3(x, y, static_cast<size_t>(z), nx, ny);
                localGrid[localIdx] = (globalIdx % 19) * 1.0;
            }
        }
    }
}

// 7-point stencil computation for one owned plane range on the GPU.
// The local buffer is padded with one ghost plane below (lz=0) and one above (lz=nzLocal+1).
__global__ void stencilKernel(const Real* __restrict__ input, Real* __restrict__ output,
                               const size_t nx, const size_t ny, const size_t nzLocal,
                               const long zGlobalStart, const size_t nzGlobal) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t lzOwned = blockIdx.z * blockDim.z + threadIdx.z;  // 0 .. nzLocal-1

    if (x >= nx || y >= ny || lzOwned >= nzLocal) {
        return;
    }

    const size_t planeSize = nx * ny;
    const size_t lz = lzOwned + 1;  // padded local index, 1 .. nzLocal
    const long z = zGlobalStart + static_cast<long>(lz);
    const size_t idx = lz * planeSize + y * nx + x;

    if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1 || z == 0 || z == static_cast<long>(nzGlobal - 1)) {
        output[idx] = input[idx];
        return;
    }

    const Real center = input[idx];
    const Real left = input[idx - 1];
    const Real right = input[idx + 1];
    const Real front = input[idx - nx];
    const Real back = input[idx + nx];
    const Real bottom = input[idx - planeSize];
    const Real top = input[idx + planeSize];

    output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
}

bool validateResult(const std::vector<Real>& grid, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Simple sanity checks

    // 1. No NaN or Inf values
    bool bad = false;
    #pragma omp parallel for reduction(|| : bad) schedule(static)
    for (size_t i = 0; i < grid.size(); ++i) {
        if (std::isnan(grid[i]) || std::isinf(grid[i])) {
            bad = true;
        }
    }
    if (bad) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    // 2. Values should be reasonable (bounded)
    Real minVal = grid[0];
    Real maxVal = grid[0];
    #pragma omp parallel for reduction(min : minVal) reduction(max : maxVal) schedule(static)
    for (size_t i = 0; i < grid.size(); ++i) {
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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    size_t nx = 128;
    size_t ny = 0;  // Will be set to nx if not specified
    size_t nz = 0;  // Will be set to nx if not specified
    int iterations = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (every rank parses identically)
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

    // GPU selection: one GPU per rank (round-robin across the ranks local to a node's visible devices)
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        if (rank == 0) fprintf(stderr, "No CUDA devices found\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int device = rank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));

    // Decompose the Z dimension across MPI ranks
    const size_t base = nz / static_cast<size_t>(numRanks);
    const size_t rem = nz % static_cast<size_t>(numRanks);
    const size_t nzLocal = base + (static_cast<size_t>(rank) < rem ? 1 : 0);
    size_t zStart = static_cast<size_t>(rank) * base + std::min(static_cast<size_t>(rank), rem);

    if (nzLocal == 0) {
        if (rank == 0) fprintf(stderr, "Too many MPI ranks (%d) for grid Z dimension (%zu)\n", numRanks, nz);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    const int leftRank = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    const int rightRank = (rank < numRanks - 1) ? rank + 1 : MPI_PROC_NULL;

    if (rank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, GPUs visible per rank: %d\n", numRanks, deviceCount);
    }

    const size_t planeSize = nx * ny;
    const size_t nzLocalPadded = nzLocal + 2;
    const size_t localBufSize = planeSize * nzLocalPadded;

    // Host staging buffer used for initialization and the final gather
    std::vector<Real> hostLocal(localBufSize);

    if (rank == 0) printf("Initializing grid...\n");
    initializeGridLocal(hostLocal, nx, ny, nzLocalPadded, static_cast<long>(zStart) - 1, nz);

    // Device double-buffers for this rank's slab
    Real* dBufA = nullptr;
    Real* dBufB = nullptr;
    CUDA_CHECK(cudaMalloc(&dBufA, localBufSize * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&dBufB, localBufSize * sizeof(Real)));
    CUDA_CHECK(cudaMemcpy(dBufA, hostLocal.data(), localBufSize * sizeof(Real), cudaMemcpyHostToDevice));

    // Pinned host staging buffers for halo exchange (one xy-plane each)
    Real* hSendLeft = nullptr;
    Real* hSendRight = nullptr;
    Real* hRecvLeft = nullptr;
    Real* hRecvRight = nullptr;
    CUDA_CHECK(cudaMallocHost(&hSendLeft, planeSize * sizeof(Real)));
    CUDA_CHECK(cudaMallocHost(&hSendRight, planeSize * sizeof(Real)));
    CUDA_CHECK(cudaMallocHost(&hRecvLeft, planeSize * sizeof(Real)));
    CUDA_CHECK(cudaMallocHost(&hRecvRight, planeSize * sizeof(Real)));

    Real* dIn = dBufA;
    Real* dOut = dBufB;

    dim3 block(8, 8, 8);
    dim3 grid(static_cast<unsigned int>((nx + block.x - 1) / block.x),
              static_cast<unsigned int>((ny + block.y - 1) / block.y),
              static_cast<unsigned int>((nzLocal + block.z - 1) / block.z));

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) printf("Running stencil computation...\n");
    const double t0 = MPI_Wtime();

    const int planeCount = static_cast<int>(planeSize);

    for (int iter = 0; iter < iterations; ++iter) {
        // Exchange halo planes with neighboring ranks
        CUDA_CHECK(cudaMemcpy(hSendLeft, dIn + 1 * planeSize, planeSize * sizeof(Real), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hSendRight, dIn + nzLocal * planeSize, planeSize * sizeof(Real), cudaMemcpyDeviceToHost));

        MPI_Sendrecv(hSendLeft, planeCount, MPI_DOUBLE, leftRank, 0,
                     hRecvRight, planeCount, MPI_DOUBLE, rightRank, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(hSendRight, planeCount, MPI_DOUBLE, rightRank, 1,
                     hRecvLeft, planeCount, MPI_DOUBLE, leftRank, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if (leftRank != MPI_PROC_NULL) {
            CUDA_CHECK(cudaMemcpy(dIn + 0 * planeSize, hRecvLeft, planeSize * sizeof(Real), cudaMemcpyHostToDevice));
        }
        if (rightRank != MPI_PROC_NULL) {
            CUDA_CHECK(cudaMemcpy(dIn + (nzLocal + 1) * planeSize, hRecvRight, planeSize * sizeof(Real), cudaMemcpyHostToDevice));
        }

        stencilKernel<<<grid, block>>>(dIn, dOut, nx, ny, nzLocal, static_cast<long>(zStart) - 1, nz);
        CUDA_CHECK(cudaGetLastError());

        std::swap(dIn, dOut);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    const double t1 = MPI_Wtime();

    double localElapsed = t1 - t0;
    double maxElapsed = 0.0;
    MPI_Reduce(&localElapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    const long durationMs = static_cast<long>(maxElapsed * 1000.0);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", durationMs);
        const double cellUpdates = static_cast<double>((nx - 2) * (ny - 2) * (nz - 2)) * iterations;
        const double mcups = cellUpdates / (durationMs / 1000.0) / 1e6;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    // Copy the final owned slab (excluding ghost planes) back to host
    CUDA_CHECK(cudaMemcpy(hostLocal.data() + planeSize, dIn + 1 * planeSize,
                           nzLocal * planeSize * sizeof(Real), cudaMemcpyDeviceToHost));

    // Gather the full grid on rank 0 for result printing / validation, matching the
    // single-process semantics of the original implementation.
    std::vector<Real> finalGrid;
    std::vector<int> recvCounts;
    std::vector<int> displs;
    if (rank == 0) {
        finalGrid.resize(nx * ny * nz);
        recvCounts.resize(numRanks);
        displs.resize(numRanks);
        size_t z = 0;
        for (int r = 0; r < numRanks; ++r) {
            const size_t rBase = nz / static_cast<size_t>(numRanks);
            const size_t rRem = nz % static_cast<size_t>(numRanks);
            const size_t rNzLocal = rBase + (static_cast<size_t>(r) < rRem ? 1 : 0);
            recvCounts[r] = static_cast<int>(rNzLocal * planeSize);
            displs[r] = static_cast<int>(z * planeSize);
            z += rNzLocal;
        }
    }

    MPI_Gatherv(hostLocal.data() + planeSize, static_cast<int>(nzLocal * planeSize), MPI_DOUBLE,
                rank == 0 ? finalGrid.data() : nullptr, recvCounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    int exitCode = 0;
    if (rank == 0) {
        if (printResults) {
            print_results(finalGrid, "Grid");
        }

        if (validate) {
            printf("Validating result...\n");
            const bool valid = validateResult(finalGrid, nx, ny, nz);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }

    CUDA_CHECK(cudaFreeHost(hSendLeft));
    CUDA_CHECK(cudaFreeHost(hSendRight));
    CUDA_CHECK(cudaFreeHost(hRecvLeft));
    CUDA_CHECK(cudaFreeHost(hRecvRight));
    CUDA_CHECK(cudaFree(dBufA));
    CUDA_CHECK(cudaFree(dBufB));

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Finalize();
    return exitCode;
}
