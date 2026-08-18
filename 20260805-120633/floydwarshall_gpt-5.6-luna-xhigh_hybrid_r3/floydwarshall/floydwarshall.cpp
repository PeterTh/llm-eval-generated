#include <algorithm>
#include <chrono>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

// Index calculation for the original flattened matrix representation.  The
// first argument is the column and the second is the row, so a source row is
// contiguous in memory.
inline constexpr std::size_t idx2(const std::size_t i, const std::size_t j,
                                  const std::size_t n) noexcept {
    return j * n + i;
}

void initializeDistanceMatrix(std::vector<unsigned int>& dist,
                              const std::size_t numNodes,
                              const unsigned int rangeMin,
                              const unsigned int rangeMax) {
    // Keep this loop serial: rand_r's state progression is part of the
    // benchmark's deterministic input and must match the original program.
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;

    for (std::size_t i = 0; i < numNodes * numNodes; ++i) {
        dist[i] = rangeMin +
                  static_cast<unsigned int>(range * rand_r(&seed) /
                                            static_cast<double>(RAND_MAX));
    }

    // The diagonal writes are independent and are useful host-side work for
    // the OpenMP part of the hybrid implementation.
#pragma omp parallel for schedule(static)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(numNodes); ++i) {
        const auto node = static_cast<std::size_t>(i);
        dist[idx2(node, node, numNodes)] = 0;
    }
}

void initializeLocalPathMatrix(std::vector<unsigned int>& path,
                               const std::size_t numNodes,
                               const std::size_t firstRow,
                               const std::size_t localRows) {
    // In the original initialization every element in source row i receives
    // i.  Initialize rows independently so OpenMP can fill the local block.
#pragma omp parallel for schedule(static)
    for (std::ptrdiff_t row = 0;
         row < static_cast<std::ptrdiff_t>(localRows); ++row) {
        const std::size_t localRow = static_cast<std::size_t>(row);
        const unsigned int source = static_cast<unsigned int>(firstRow + localRow);
        std::fill_n(path.data() + localRow * numNodes, numNodes, source);
    }
}

// One CUDA block covers eight source rows and a 32-column tile.  The pivot
// tile and each row's distance-to-k are shared by the threads in the block,
// leaving the global-memory traffic dominated by the one coalesced matrix
// read and the conditional result/path writes per cell.
__global__ void floydWarshallKernel(unsigned int* __restrict__ dist,
                                    unsigned int* __restrict__ path,
                                    const unsigned int* __restrict__ pivot,
                                    const std::size_t localRows,
                                    const std::size_t numNodes,
                                    const std::size_t k) {
    __shared__ unsigned int pivotTile[32];
    __shared__ unsigned int distanceToK[8];

    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const std::size_t column = static_cast<std::size_t>(blockIdx.x) * 32 + tx;
    const std::size_t row = static_cast<std::size_t>(blockIdx.y) * 8 + ty;

    if (column < numNodes) {
        pivotTile[tx] = pivot[column];
    } else {
        // Every thread must participate in the barrier.  Out-of-range values
        // are never consumed because the corresponding cell is not valid.
        pivotTile[tx] = 0;
    }
    if (tx == 0 && row < localRows) {
        distanceToK[ty] = dist[row * numNodes + k];
    }
    __syncthreads();

    if (row < localRows && column < numNodes) {
        const std::size_t cell = row * numNodes + column;
        const unsigned int candidate = distanceToK[ty] + pivotTile[tx];

        // This is deliberately the same strict comparison and unsigned
        // addition as the original in-place Floyd-Warshall implementation.
        if (candidate < dist[cell]) {
            dist[cell] = candidate;
            path[cell] = static_cast<unsigned int>(k);
        }
    }
}

void checkCuda(const cudaError_t status, const char* operation, const int rank) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "MPI rank %d: CUDA error during %s: %s\n", rank,
                     operation, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
}

bool validateResult(const std::vector<unsigned int>& dist,
                    const std::size_t numNodes) {
    // Basic sanity checks.

    // 1. Diagonal should be zero.
    for (std::size_t i = 0; i < numNodes; ++i) {
        if (dist[idx2(i, i, numNodes)] != 0) {
            std::printf("Validation failed: diagonal element [%zu,%zu] is not zero\n",
                        i, i);
            return false;
        }
    }

    // 2. Triangle inequality: dist[i][k] + dist[k][j] >= dist[i][j].
    // Check a sample of paths to avoid O(n^3) validation time.
    for (std::size_t i = 0; i < std::min(numNodes, static_cast<std::size_t>(10));
         ++i) {
        for (std::size_t j = 0;
             j < std::min(numNodes, static_cast<std::size_t>(10)); ++j) {
            for (std::size_t k = 0; k < numNodes; ++k) {
                const unsigned int distIJ = dist[idx2(j, i, numNodes)];
                const unsigned int distIK = dist[idx2(k, i, numNodes)];
                const unsigned int distKJ = dist[idx2(j, k, numNodes)];

                // Check for overflow before addition, matching the original
                // validation behavior.
                if (distIK < INF && distKJ < INF &&
                    distIK + distKJ < distIJ) {
                    std::printf(
                        "Validation failed: triangle inequality violated at "
                        "[%zu,%zu,%zu]\n",
                        i, j, k);
                    return false;
                }
            }
        }
    }

    return true;
}

void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("Options:\n");
    std::printf("  -n <num>     Number of nodes in the graph (default: 512)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) !=
        MPI_SUCCESS) {
        return EXIT_FAILURE;
    }

    int rank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    std::size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command-line arguments identically on every rank.
    bool parseError = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numNodes = static_cast<std::size_t>(std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            if (rank == 0) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            parseError = true;
        }
    }

    if (parseError || numNodes == 0 || numNodes > static_cast<std::size_t>(INT_MAX) ||
        numNodes > std::numeric_limits<std::size_t>::max() / numNodes) {
        if (rank == 0 && !parseError) {
            std::printf("Number of nodes must be in the range [1, %d].\n",
                        INT_MAX);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    const std::size_t matrixEntries = numNodes * numNodes;
    if (matrixEntries > static_cast<std::size_t>(INT_MAX) ||
        matrixEntries * sizeof(unsigned int) < matrixEntries) {
        if (rank == 0) {
            std::printf("The requested matrix is too large for this MPI build.\n");
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    // Determine a stable local rank so MPI processes on the same host share
    // the available GPUs instead of all selecting device zero.
    MPI_Comm localComm = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                        &localComm);
    int localRank = 0;
    MPI_Comm_rank(localComm, &localRank);

    int deviceCount = 0;
    const cudaError_t deviceQuery = cudaGetDeviceCount(&deviceCount);
    const int hasDevice =
        (deviceQuery == cudaSuccess && deviceCount > 0) ? 1 : 0;
    int allRanksHaveDevice = 0;
    MPI_Allreduce(&hasDevice, &allRanksHaveDevice, 1, MPI_INT, MPI_MIN,
                  MPI_COMM_WORLD);
    if (!allRanksHaveDevice) {
        if (rank == 0) {
            std::fprintf(stderr,
                         "Every MPI rank must have access to at least one CUDA device.\n");
        }
        MPI_Comm_free(&localComm);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    checkCuda(cudaSetDevice(localRank % deviceCount), "cudaSetDevice", rank);
    checkCuda(cudaFree(0), "CUDA context initialization", rank);

    int validationStatus = 0;
    if (rank == 0) {
        std::printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        std::printf("Number of nodes: %zu\n", numNodes);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Hybrid parallelism: MPI + OpenMP + CUDA\n");
    }

    // Divide complete source rows among ranks.  The extra row, when present,
    // is assigned by the quotient/remainder-free proportional partition below
    // so all blocks differ in size by at most one.
    std::vector<std::size_t> rowStarts(static_cast<std::size_t>(worldSize) + 1);
    std::vector<int> matrixCounts(static_cast<std::size_t>(worldSize));
    std::vector<int> matrixDisplacements(static_cast<std::size_t>(worldSize));
    for (int r = 0; r <= worldSize; ++r) {
        rowStarts[static_cast<std::size_t>(r)] =
            numNodes * static_cast<std::size_t>(r) /
            static_cast<std::size_t>(worldSize);
    }
    for (int r = 0; r < worldSize; ++r) {
        const std::size_t rows = rowStarts[static_cast<std::size_t>(r) + 1] -
                                 rowStarts[static_cast<std::size_t>(r)];
        const std::size_t entries = rows * numNodes;
        matrixCounts[static_cast<std::size_t>(r)] = static_cast<int>(entries);
        matrixDisplacements[static_cast<std::size_t>(r)] = static_cast<int>(
            rowStarts[static_cast<std::size_t>(r)] * numNodes);
    }

    const std::size_t firstRow = rowStarts[static_cast<std::size_t>(rank)];
    const std::size_t localRows =
        rowStarts[static_cast<std::size_t>(rank) + 1] - firstRow;
    const std::size_t localEntries = localRows * numNodes;

    std::vector<unsigned int> globalDist;
    if (rank == 0) {
        globalDist.resize(matrixEntries);
        std::printf("Initializing graph...\n");
        initializeDistanceMatrix(globalDist, numNodes, 1, MAX_DISTANCE);
    }

    std::vector<unsigned int> localDist(localEntries);
    std::vector<unsigned int> localPath(localEntries);
    initializeLocalPathMatrix(localPath, numNodes, firstRow, localRows);

    MPI_Scatterv(rank == 0 ? globalDist.data() : nullptr, matrixCounts.data(),
                 matrixDisplacements.data(), MPI_UNSIGNED, localDist.data(),
                 static_cast<int>(localEntries), MPI_UNSIGNED, 0,
                 MPI_COMM_WORLD);

    unsigned int* deviceDist = nullptr;
    unsigned int* devicePath = nullptr;
    const std::size_t deviceEntries = std::max<std::size_t>(localEntries, 1);
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&deviceDist),
                         deviceEntries * sizeof(unsigned int)),
              "cudaMalloc(deviceDist)", rank);
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&devicePath),
                         deviceEntries * sizeof(unsigned int)),
              "cudaMalloc(devicePath)", rank);
    if (localEntries != 0) {
        checkCuda(cudaMemcpy(deviceDist, localDist.data(),
                             localEntries * sizeof(unsigned int),
                             cudaMemcpyHostToDevice),
                  "copying distance matrix to CUDA", rank);
        checkCuda(cudaMemcpy(devicePath, localPath.data(),
                             localEntries * sizeof(unsigned int),
                             cudaMemcpyHostToDevice),
                  "copying path matrix to CUDA", rank);
    }

    unsigned int* pivot = nullptr;
    checkCuda(cudaHostAlloc(reinterpret_cast<void**>(&pivot),
                            numNodes * sizeof(unsigned int),
                            cudaHostAllocPortable),
              "cudaHostAlloc(pivot)", rank);
    unsigned int* devicePivot = nullptr;
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&devicePivot),
                         numNodes * sizeof(unsigned int)),
              "cudaMalloc(devicePivot)", rank);

    // Precompute the owner of each pivot row.  This also handles the valid
    // case where more MPI ranks than graph rows are launched.
    std::vector<int> pivotOwners(numNodes);
    for (std::size_t k = 0; k < numNodes; ++k) {
        for (int r = 0; r < worldSize; ++r) {
            if (rowStarts[static_cast<std::size_t>(r)] <= k &&
                k < rowStarts[static_cast<std::size_t>(r) + 1]) {
                pivotOwners[k] = r;
                break;
            }
        }
    }

    if (rank == 0) {
        std::printf("Computing shortest paths...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    constexpr unsigned int blockX = 32;
    constexpr unsigned int blockY = 8;
    for (std::size_t k = 0; k < numNodes; ++k) {
        const int owner = pivotOwners[k];

        // The owner keeps the authoritative pivot row on its GPU.  A blocking
        // copy here also completes the preceding iteration on that rank;
        // every rank then receives the same row through MPI.
        if (rank == owner) {
            const std::size_t ownerRow = k - firstRow;
            checkCuda(cudaMemcpy(pivot, deviceDist + ownerRow * numNodes,
                                 numNodes * sizeof(unsigned int),
                                 cudaMemcpyDeviceToHost),
                      "copying pivot row from CUDA", rank);
        }
        MPI_Bcast(pivot, static_cast<int>(numNodes), MPI_UNSIGNED, owner,
                  MPI_COMM_WORLD);

        // The host pivot buffer is reused by the next MPI_Bcast.  A blocking
        // copy prevents that broadcast from racing an in-flight DMA read on
        // non-owner ranks; the transfer is only O(n) per iteration and the
        // following O(n^2) kernel remains the dominant operation.
        checkCuda(cudaMemcpy(devicePivot, pivot,
                             numNodes * sizeof(unsigned int),
                             cudaMemcpyHostToDevice),
                  "copying pivot row to CUDA", rank);

        if (localRows != 0) {
            const dim3 block(blockX, blockY, 1);
            const dim3 grid(
                static_cast<unsigned int>((numNodes + blockX - 1) / blockX),
                static_cast<unsigned int>((localRows + blockY - 1) / blockY), 1);
            floydWarshallKernel<<<grid, block>>>(
                deviceDist, devicePath, devicePivot, localRows, numNodes, k);
            checkCuda(cudaGetLastError(), "launching Floyd-Warshall kernel", rank);
        }
    }

    // This copy synchronizes the final kernel before the distributed result is
    // assembled.  Path is intentionally retained on the device: like the
    // original benchmark, only the distance matrix is externally reported.
    if (localEntries != 0) {
        checkCuda(cudaMemcpy(localDist.data(), deviceDist,
                             localEntries * sizeof(unsigned int),
                             cudaMemcpyDeviceToHost),
                  "copying final distance matrix from CUDA", rank);
    }

    const double localSeconds = MPI_Wtime() - start;
    double elapsedSeconds = 0.0;
    MPI_Reduce(&localSeconds, &elapsedSeconds, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);

    MPI_Gatherv(localDist.data(), static_cast<int>(localEntries), MPI_UNSIGNED,
                rank == 0 ? globalDist.data() : nullptr, matrixCounts.data(),
                matrixDisplacements.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    checkCuda(cudaFree(devicePivot), "cudaFree(devicePivot)", rank);
    checkCuda(cudaFreeHost(pivot), "cudaFreeHost(pivot)", rank);
    checkCuda(cudaFree(devicePath), "cudaFree(devicePath)", rank);
    checkCuda(cudaFree(deviceDist), "cudaFree(deviceDist)", rank);
    MPI_Comm_free(&localComm);

    if (rank == 0) {
        const long duration = static_cast<long>(elapsedSeconds * 1000.0);
        std::printf("Computation time: %ld ms\n", duration);

        const double operations = static_cast<double>(numNodes) *
                                  static_cast<double>(numNodes) *
                                  static_cast<double>(numNodes);
        const double gflops = elapsedSeconds > 0.0
                                  ? operations / elapsedSeconds / 1.0e9
                                  : 0.0;
        std::printf("Performance: %.3f GOPS\n", gflops);

        if (printResults) {
            print_results_int(globalDist, "DistanceMatrix");
        }

        if (validate) {
            std::printf("Validating result...\n");
            const bool valid = validateResult(globalDist, numNodes);
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            validationStatus = valid ? 0 : 1;
        }
    }

    // Make validation status visible to every MPI process before finalization,
    // while preserving the original exit semantics.
    if (validate) {
        MPI_Bcast(&validationStatus, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return validationStatus;
}
