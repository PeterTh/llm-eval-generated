#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
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
constexpr int CUDA_TILE = 32;

static_assert(CUDA_TILE * CUDA_TILE <= 1024,
              "The Floyd-Warshall CUDA tile must fit in one CUDA thread block");

// The original idx2(j, i, n) layout is exactly row-major storage for the
// logical (source=i, destination=j) matrix used by Floyd-Warshall.
inline constexpr size_t matrixIndex(const size_t source, const size_t destination,
                                    const size_t n) noexcept {
    return source * n + destination;
}

void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes,
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;

    // Keep this serial: rand_r is stateful, and this preserves the benchmark's
    // exact deterministic input matrix independent of the MPI rank count.
    for (size_t i = 0; i < numNodes * numNodes; ++i) {
        dist[i] = rangeMin +
                  static_cast<unsigned int>(range * rand_r(&seed) / static_cast<double>(RAND_MAX));
    }

    for (size_t i = 0; i < numNodes; ++i) {
        dist[matrixIndex(i, i, numNodes)] = 0;
    }
}

bool validateResult(const std::vector<unsigned int>& dist, const size_t numNodes) {
    for (size_t i = 0; i < numNodes; ++i) {
        if (dist[matrixIndex(i, i, numNodes)] != 0) {
            printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", i, i);
            return false;
        }
    }

    const size_t sampledNodes = std::min(numNodes, static_cast<size_t>(10));
    for (size_t i = 0; i < sampledNodes; ++i) {
        for (size_t j = 0; j < sampledNodes; ++j) {
            for (size_t k = 0; k < numNodes; ++k) {
                const unsigned int distIJ = dist[matrixIndex(i, j, numNodes)];
                const unsigned int distIK = dist[matrixIndex(i, k, numNodes)];
                const unsigned int distKJ = dist[matrixIndex(k, j, numNodes)];

                if (distIK < INF && distKJ < INF && distIK + distKJ < distIJ) {
                    printf("Validation failed: triangle inequality violated at [%zu,%zu,%zu]\n",
                           i, j, k);
                    return false;
                }
            }
        }
    }

    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Number of nodes in the graph (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

[[noreturn]] void abortWithCudaError(const cudaError_t error, const char* expression,
                                     const int rank, const char* file, const int line) {
    fprintf(stderr, "Rank %d: CUDA error at %s:%d while executing %s: %s\n", rank, file, line,
            expression, cudaGetErrorString(error));
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

void checkCuda(const cudaError_t error, const char* expression, const int rank,
               const char* file, const int line) {
    if (error != cudaSuccess) {
        abortWithCudaError(error, expression, rank, file, line);
    }
}

#define CUDA_CHECK(rank, expression) checkCuda((expression), #expression, (rank), __FILE__, __LINE__)

// Phase 1: close the diagonal block.  One CUDA block owns one Floyd-Warshall
// tile, allowing a synchronization after every intermediate vertex.
__global__ void closePivotBlock(unsigned int* __restrict__ dist,
                                unsigned int* __restrict__ path,
                                const int localPivotRow, const int n,
                                const int pivotStart, const int pivotWidth) {
    __shared__ unsigned int tile[CUDA_TILE][CUDA_TILE + 1];

    const int column = threadIdx.x;
    const int row = threadIdx.y;
    const bool active = row < pivotWidth && column < pivotWidth;
    const size_t offset = static_cast<size_t>(localPivotRow + row) * n + pivotStart + column;
    unsigned int currentPath = 0;

    if (active) {
        tile[row][column] = dist[offset];
        currentPath = path[offset];
    }
    __syncthreads();

    for (int k = 0; k < pivotWidth; ++k) {
        if (active) {
            const unsigned int candidate = tile[row][k] + tile[k][column];
            if (candidate < tile[row][column]) {
                tile[row][column] = candidate;
                currentPath = static_cast<unsigned int>(pivotStart + k);
            }
        }
        __syncthreads();
    }

    if (active) {
        dist[offset] = tile[row][column];
        path[offset] = currentPath;
    }
}

// Phase 2a: update every non-diagonal block in the pivot block row.  This is
// executed only by the rank that owns that row.
__global__ void updatePivotRow(unsigned int* __restrict__ dist,
                               unsigned int* __restrict__ path,
                               const int localPivotRow, const int n,
                               const int pivotBlock, const int pivotStart,
                               const int pivotWidth) {
    const int destinationBlock = blockIdx.x;
    if (destinationBlock == pivotBlock) {
        return;
    }

    __shared__ unsigned int pivot[CUDA_TILE][CUDA_TILE + 1];
    __shared__ unsigned int target[CUDA_TILE][CUDA_TILE + 1];

    const int column = threadIdx.x;
    const int row = threadIdx.y;
    const int destination = destinationBlock * CUDA_TILE + column;
    const bool pivotActive = row < pivotWidth && column < pivotWidth;
    const bool active = row < pivotWidth && destination < n;
    const size_t outputOffset = static_cast<size_t>(localPivotRow + row) * n + destination;
    unsigned int currentPath = 0;

    if (pivotActive) {
        pivot[row][column] = dist[static_cast<size_t>(localPivotRow + row) * n + pivotStart + column];
    }
    if (active) {
        target[row][column] = dist[outputOffset];
        currentPath = path[outputOffset];
    }
    __syncthreads();

    for (int k = 0; k < pivotWidth; ++k) {
        if (active) {
            const unsigned int candidate = pivot[row][k] + target[k][column];
            if (candidate < target[row][column]) {
                target[row][column] = candidate;
                currentPath = static_cast<unsigned int>(pivotStart + k);
            }
        }
        __syncthreads();
    }

    if (active) {
        dist[outputOffset] = target[row][column];
        path[outputOffset] = currentPath;
    }
}

// Phase 2b: after the updated pivot rows are broadcast, update the pivot
// column blocks held by this rank.
__global__ void updatePivotColumn(unsigned int* __restrict__ dist,
                                  unsigned int* __restrict__ path,
                                  const unsigned int* __restrict__ pivotRows,
                                  const int localBlockStart, const int n,
                                  const int pivotBlock, const int pivotStart,
                                  const int pivotWidth) {
    const int sourceBlock = localBlockStart + blockIdx.x;
    if (sourceBlock == pivotBlock) {
        return;
    }

    __shared__ unsigned int pivot[CUDA_TILE][CUDA_TILE + 1];
    __shared__ unsigned int target[CUDA_TILE][CUDA_TILE + 1];

    const int column = threadIdx.x;
    const int row = threadIdx.y;
    const int source = sourceBlock * CUDA_TILE + row;
    const int destination = pivotStart + column;
    const bool pivotActive = row < pivotWidth && column < pivotWidth;
    const bool active = source < n && column < pivotWidth;
    const int localRow = (sourceBlock - localBlockStart) * CUDA_TILE + row;
    const size_t outputOffset = static_cast<size_t>(localRow) * n + destination;
    unsigned int currentPath = 0;

    if (pivotActive) {
        pivot[row][column] = pivotRows[static_cast<size_t>(row) * n + pivotStart + column];
    }
    if (active) {
        target[row][column] = dist[outputOffset];
        currentPath = path[outputOffset];
    }
    __syncthreads();

    for (int k = 0; k < pivotWidth; ++k) {
        if (active) {
            const unsigned int candidate = target[row][k] + pivot[k][column];
            if (candidate < target[row][column]) {
                target[row][column] = candidate;
                currentPath = static_cast<unsigned int>(pivotStart + k);
            }
        }
        __syncthreads();
    }

    if (active) {
        dist[outputOffset] = target[row][column];
        path[outputOffset] = currentPath;
    }
}

// Phase 3: update the remaining local blocks using the local pivot column and
// the broadcast pivot row.  This is the dominant O(n^3) GPU work.
__global__ void updateRemainingBlocks(unsigned int* __restrict__ dist,
                                      unsigned int* __restrict__ path,
                                      const unsigned int* __restrict__ pivotRows,
                                      const int localBlockStart, const int n,
                                      const int pivotBlock, const int pivotStart,
                                      const int pivotWidth) {
    const int destinationBlock = blockIdx.x;
    const int sourceBlock = localBlockStart + blockIdx.y;
    if (sourceBlock == pivotBlock || destinationBlock == pivotBlock) {
        return;
    }

    __shared__ unsigned int left[CUDA_TILE][CUDA_TILE + 1];
    __shared__ unsigned int right[CUDA_TILE][CUDA_TILE + 1];

    const int column = threadIdx.x;
    const int row = threadIdx.y;
    const int source = sourceBlock * CUDA_TILE + row;
    const int destination = destinationBlock * CUDA_TILE + column;
    const bool leftActive = source < n && column < pivotWidth;
    const bool rightActive = row < pivotWidth && destination < n;
    const bool active = source < n && destination < n;
    const int localRow = (sourceBlock - localBlockStart) * CUDA_TILE + row;
    const size_t outputOffset = static_cast<size_t>(localRow) * n + destination;
    unsigned int currentDistance = 0;
    unsigned int currentPath = 0;

    if (leftActive) {
        left[row][column] = dist[static_cast<size_t>(localRow) * n + pivotStart + column];
    }
    if (rightActive) {
        right[row][column] = pivotRows[static_cast<size_t>(row) * n + destination];
    }
    if (active) {
        currentDistance = dist[outputOffset];
        currentPath = path[outputOffset];
    }
    __syncthreads();

    if (active) {
        for (int k = 0; k < pivotWidth; ++k) {
            const unsigned int candidate = left[row][k] + right[k][column];
            if (candidate < currentDistance) {
                currentDistance = candidate;
                currentPath = static_cast<unsigned int>(pivotStart + k);
            }
        }
        dist[outputOffset] = currentDistance;
        path[outputOffset] = currentPath;
    }
}

int blocksForRank(const int blockCount, const int rank, const int rankCount) {
    const int baseBlocks = blockCount / rankCount;
    const int remainder = blockCount % rankCount;
    return baseBlocks + (rank < remainder ? 1 : 0);
}

int firstBlockForRank(const int blockCount, const int rank, const int rankCount) {
    const int baseBlocks = blockCount / rankCount;
    const int remainder = blockCount % rankCount;
    return rank * baseBlocks + std::min(rank, remainder);
}

int ownerOfBlock(const int block, const int blockCount, const int rankCount) {
    const int baseBlocks = blockCount / rankCount;
    const int remainder = blockCount % rankCount;
    const int largerRankBlocks = remainder * (baseBlocks + 1);
    if (block < largerRankBlocks) {
        return block / (baseBlocks + 1);
    }
    return remainder + (block - largerRankBlocks) / baseBlocks;
}

bool parseNodeCount(const char* text, size_t& value) {
    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed == 0 ||
        parsed > static_cast<unsigned long long>(std::numeric_limits<int>::max())) {
        return false;
    }
    value = static_cast<size_t>(parsed);
    return true;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int rankCount = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &rankCount);

    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;
    bool showHelp = false;
    bool argumentError = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            if (!parseNodeCount(argv[++i], numNodes)) {
                argumentError = true;
            }
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            showHelp = true;
        } else {
            argumentError = true;
        }
    }

    if (showHelp || argumentError) {
        if (rank == 0) {
            if (argumentError) {
                printf("Invalid command-line arguments\n");
            }
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return argumentError ? EXIT_FAILURE : EXIT_SUCCESS;
    }

    const size_t matrixElements = numNodes * numNodes;
    // MPI's portable vector collectives use int counts/displacements.  This
    // also rejects matrices that cannot fit in the benchmark's practical host
    // and device memory range before any allocation is attempted.
    if (matrixElements > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (rank == 0) {
            fprintf(stderr, "Matrix is too large for MPI collective counts: %zu nodes\n", numNodes);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    int localRank = 0;
    MPI_Comm localComm = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    MPI_Comm_rank(localComm, &localRank);

    int deviceCount = 0;
    CUDA_CHECK(rank, cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        if (rank == 0) {
            fprintf(stderr, "No CUDA device is visible; this benchmark requires CUDA on every MPI node.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return EXIT_FAILURE;
    }
    const int device = localRank % deviceCount;
    CUDA_CHECK(rank, cudaSetDevice(device));

    const int n = static_cast<int>(numNodes);
    const int blockCount = (n + CUDA_TILE - 1) / CUDA_TILE;
    const int localBlockStart = firstBlockForRank(blockCount, rank, rankCount);
    const int localBlockCount = blocksForRank(blockCount, rank, rankCount);
    const int firstLocalRow = localBlockStart * CUDA_TILE;
    const int localRows = firstLocalRow < n
                              ? std::min(n - firstLocalRow, localBlockCount * CUDA_TILE)
                              : 0;

    std::vector<int> receiveCounts(rankCount);
    std::vector<int> displacements(rankCount);
    for (int mpiRank = 0; mpiRank < rankCount; ++mpiRank) {
        const int rankBlockStart = firstBlockForRank(blockCount, mpiRank, rankCount);
        const int rankBlockCount = blocksForRank(blockCount, mpiRank, rankCount);
        const int rankFirstRow = rankBlockStart * CUDA_TILE;
        const int rankRows = rankFirstRow < n
                                 ? std::min(n - rankFirstRow, rankBlockCount * CUDA_TILE)
                                 : 0;
        receiveCounts[mpiRank] = rankRows * n;
        displacements[mpiRank] = rankFirstRow * n;
    }

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", rankCount);
        printf("CUDA tile size: %d\n", CUDA_TILE);
        printf("OpenMP host threads per rank: %d\n", omp_get_max_threads());
        printf("Initializing graph...\n");
    }

    std::vector<unsigned int> globalDist;
    if (rank == 0) {
        globalDist.resize(matrixElements);
        initializeDistanceMatrix(globalDist, numNodes, 1, MAX_DISTANCE);
    }

    std::vector<unsigned int> localDist(static_cast<size_t>(localRows) * n);
    std::vector<unsigned int> localPath(static_cast<size_t>(localRows) * n);
    MPI_Scatterv(rank == 0 ? globalDist.data() : nullptr, receiveCounts.data(), displacements.data(),
                 MPI_UNSIGNED, localRows == 0 ? nullptr : localDist.data(), localRows * n,
                 MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    // Every rank constructs its own path rows.  This is both embarrassingly
    // parallel host work and exactly matches the original path initialization.
#pragma omp parallel for schedule(static)
    for (int row = 0; row < localRows; ++row) {
        unsigned int* const pathRow = localPath.data() + static_cast<size_t>(row) * n;
        for (int column = 0; column < n; ++column) {
            pathRow[column] = static_cast<unsigned int>(column);
        }
    }

    unsigned int* deviceDist = nullptr;
    unsigned int* devicePath = nullptr;
    unsigned int* devicePivotRows = nullptr;
    unsigned int* hostPivotRows = nullptr;
    const size_t localElements = static_cast<size_t>(localRows) * n;
    const size_t allocationElements = std::max(static_cast<size_t>(1), localElements);
    const size_t pivotElements = static_cast<size_t>(CUDA_TILE) * n;

    CUDA_CHECK(rank, cudaMalloc(reinterpret_cast<void**>(&deviceDist),
                                allocationElements * sizeof(unsigned int)));
    CUDA_CHECK(rank, cudaMalloc(reinterpret_cast<void**>(&devicePath),
                                allocationElements * sizeof(unsigned int)));
    CUDA_CHECK(rank, cudaMalloc(reinterpret_cast<void**>(&devicePivotRows),
                                pivotElements * sizeof(unsigned int)));
    CUDA_CHECK(rank, cudaMallocHost(reinterpret_cast<void**>(&hostPivotRows),
                                    pivotElements * sizeof(unsigned int)));

    if (localElements != 0) {
        CUDA_CHECK(rank, cudaMemcpy(deviceDist, localDist.data(), localElements * sizeof(unsigned int),
                                    cudaMemcpyHostToDevice));
        CUDA_CHECK(rank, cudaMemcpy(devicePath, localPath.data(), localElements * sizeof(unsigned int),
                                    cudaMemcpyHostToDevice));
    }

    if (rank == 0) {
        printf("Computing shortest paths...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    const dim3 threads(CUDA_TILE, CUDA_TILE);
    for (int pivotBlock = 0; pivotBlock < blockCount; ++pivotBlock) {
        const int pivotStart = pivotBlock * CUDA_TILE;
        const int pivotWidth = std::min(CUDA_TILE, n - pivotStart);
        const int owner = ownerOfBlock(pivotBlock, blockCount, rankCount);

        if (rank == owner) {
            const int localPivotRow = (pivotBlock - localBlockStart) * CUDA_TILE;
            closePivotBlock<<<1, threads>>>(deviceDist, devicePath, localPivotRow, n, pivotStart,
                                             pivotWidth);
            CUDA_CHECK(rank, cudaGetLastError());

            updatePivotRow<<<blockCount, threads>>>(deviceDist, devicePath, localPivotRow, n,
                                                     pivotBlock, pivotStart, pivotWidth);
            CUDA_CHECK(rank, cudaGetLastError());

            CUDA_CHECK(rank, cudaMemcpy(hostPivotRows,
                                        deviceDist + static_cast<size_t>(localPivotRow) * n,
                                        static_cast<size_t>(pivotWidth) * n * sizeof(unsigned int),
                                        cudaMemcpyDeviceToHost));
        }

        MPI_Bcast(hostPivotRows, pivotWidth * n, MPI_UNSIGNED, owner, MPI_COMM_WORLD);
        CUDA_CHECK(rank, cudaMemcpy(devicePivotRows, hostPivotRows,
                                    static_cast<size_t>(pivotWidth) * n * sizeof(unsigned int),
                                    cudaMemcpyHostToDevice));

        if (localBlockCount != 0) {
            updatePivotColumn<<<localBlockCount, threads>>>(
                deviceDist, devicePath, devicePivotRows, localBlockStart, n, pivotBlock, pivotStart,
                pivotWidth);
            CUDA_CHECK(rank, cudaGetLastError());

            updateRemainingBlocks<<<dim3(blockCount, localBlockCount), threads>>>(
                deviceDist, devicePath, devicePivotRows, localBlockStart, n, pivotBlock, pivotStart,
                pivotWidth);
            CUDA_CHECK(rank, cudaGetLastError());
        }
    }

    CUDA_CHECK(rank, cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    const double localElapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (localElements != 0) {
        CUDA_CHECK(rank, cudaMemcpy(localDist.data(), deviceDist,
                                    localElements * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    }

    MPI_Gatherv(localRows == 0 ? nullptr : localDist.data(), localRows * n, MPI_UNSIGNED,
                rank == 0 ? globalDist.data() : nullptr, receiveCounts.data(), displacements.data(),
                MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    CUDA_CHECK(rank, cudaFreeHost(hostPivotRows));
    CUDA_CHECK(rank, cudaFree(devicePivotRows));
    CUDA_CHECK(rank, cudaFree(devicePath));
    CUDA_CHECK(rank, cudaFree(deviceDist));
    MPI_Comm_free(&localComm);

    int exitCode = EXIT_SUCCESS;
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", elapsed * 1000.0);
        const double operations = static_cast<double>(numNodes) * numNodes * numNodes;
        const double gops = elapsed > 0.0 ? operations / elapsed / 1.0e9 : 0.0;
        printf("Performance: %.3f GOPS\n", gops);

        if (printResults) {
            print_results_int(globalDist, "DistanceMatrix");
        }

        if (validate) {
            printf("Validating result...\n");
            if (validateResult(globalDist, numNodes)) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                exitCode = EXIT_FAILURE;
            }
        }
    }

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
