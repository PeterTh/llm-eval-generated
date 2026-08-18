#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;
constexpr int TILE_SIZE = 32;
constexpr int BLOCK_ROWS = 8;
constexpr int ROWS_PER_THREAD = TILE_SIZE / BLOCK_ROWS;

static_assert(TILE_SIZE % BLOCK_ROWS == 0);

inline void checkCuda(const cudaError_t error, const char* const operation) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "CUDA error during %s: %s\n", operation,
                     cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(operation) checkCuda((operation), #operation)

// Index calculation for flattened 2D array
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes, 
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;
    
    for (size_t i = 0; i < numNodes * numNodes; ++i) {
        dist[i] = rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
    }
    
    // Set diagonal to zero (distance from node to itself is 0)
    for (size_t i = 0; i < numNodes; ++i) {
        dist[idx2(i, i, numNodes)] = 0;
    }
}

// Phase 1 closes the diagonal (pivot) tile for this round.  Each thread owns
// four matrix entries in different rows; a padded shared-memory stride avoids
// bank conflicts for the transposed accesses in the inner loop.
__global__ __launch_bounds__(TILE_SIZE * BLOCK_ROWS)
void updatePivotTile(unsigned int* __restrict__ dist,
                     unsigned int* __restrict__ path,
                     const size_t numNodes, const int round) {
    __shared__ unsigned int pivot[TILE_SIZE][TILE_SIZE + 1];

    const int column = static_cast<int>(threadIdx.x);
    const int firstRow = static_cast<int>(threadIdx.y);
    const size_t tileOffset = static_cast<size_t>(round) * TILE_SIZE;
    int lastIntermediate[ROWS_PER_THREAD];

#pragma unroll
    for (int item = 0; item < ROWS_PER_THREAD; ++item) {
        const int row = firstRow + item * BLOCK_ROWS;
        const size_t globalRow = tileOffset + row;
        const size_t globalColumn = tileOffset + column;
        pivot[row][column] = (globalRow < numNodes && globalColumn < numNodes)
                                 ? dist[globalRow * numNodes + globalColumn]
                                 : INF;
        lastIntermediate[item] = -1;
    }
    __syncthreads();

#pragma unroll
    for (int k = 0; k < TILE_SIZE; ++k) {
#pragma unroll
        for (int item = 0; item < ROWS_PER_THREAD; ++item) {
            const int row = firstRow + item * BLOCK_ROWS;
            const unsigned int candidate = pivot[row][k] + pivot[k][column];
            if (candidate < pivot[row][column]) {
                pivot[row][column] = candidate;
                lastIntermediate[item] = k;
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int item = 0; item < ROWS_PER_THREAD; ++item) {
        const int row = firstRow + item * BLOCK_ROWS;
        const size_t globalRow = tileOffset + row;
        const size_t globalColumn = tileOffset + column;
        if (globalRow < numNodes && globalColumn < numNodes) {
            const size_t index = globalRow * numNodes + globalColumn;
            dist[index] = pivot[row][column];
            if (lastIntermediate[item] >= 0) {
                path[index] = static_cast<unsigned int>(tileOffset +
                                                        lastIntermediate[item]);
            } else if (round == 0) {
                path[index] = static_cast<unsigned int>(globalRow);
            }
        }
    }
}

// Phase 2 updates every tile in the pivot row and pivot column.  The second
// grid dimension selects row (0) or column (1), allowing both sets of
// independent tiles to execute concurrently in one launch.
__global__ __launch_bounds__(TILE_SIZE * BLOCK_ROWS)
void updatePivotRowAndColumn(unsigned int* __restrict__ dist,
                             unsigned int* __restrict__ path,
                             const size_t numNodes, const int round) {
    __shared__ unsigned int pivot[TILE_SIZE][TILE_SIZE + 1];
    __shared__ unsigned int tile[TILE_SIZE][TILE_SIZE + 1];

    const int column = static_cast<int>(threadIdx.x);
    const int firstRow = static_cast<int>(threadIdx.y);
    const int otherTile = static_cast<int>(blockIdx.x) +
                          (static_cast<int>(blockIdx.x) >= round);
    const bool updateColumn = blockIdx.y != 0;
    const size_t pivotOffset = static_cast<size_t>(round) * TILE_SIZE;
    const size_t otherOffset = static_cast<size_t>(otherTile) * TILE_SIZE;
    int lastIntermediate[ROWS_PER_THREAD];

#pragma unroll
    for (int item = 0; item < ROWS_PER_THREAD; ++item) {
        const int row = firstRow + item * BLOCK_ROWS;
        const size_t pivotRow = pivotOffset + row;
        const size_t pivotColumn = pivotOffset + column;
        pivot[row][column] = (pivotRow < numNodes && pivotColumn < numNodes)
                                 ? dist[pivotRow * numNodes + pivotColumn]
                                 : INF;

        const size_t globalRow =
            (updateColumn ? otherOffset : pivotOffset) + row;
        const size_t globalColumn =
            (updateColumn ? pivotOffset : otherOffset) + column;
        tile[row][column] = (globalRow < numNodes && globalColumn < numNodes)
                                ? dist[globalRow * numNodes + globalColumn]
                                : INF;
        lastIntermediate[item] = -1;
    }
    __syncthreads();

#pragma unroll
    for (int k = 0; k < TILE_SIZE; ++k) {
#pragma unroll
        for (int item = 0; item < ROWS_PER_THREAD; ++item) {
            const int row = firstRow + item * BLOCK_ROWS;
            const unsigned int candidate =
                updateColumn ? tile[row][k] + pivot[k][column]
                             : pivot[row][k] + tile[k][column];
            if (candidate < tile[row][column]) {
                tile[row][column] = candidate;
                lastIntermediate[item] = k;
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int item = 0; item < ROWS_PER_THREAD; ++item) {
        const int row = firstRow + item * BLOCK_ROWS;
        const size_t globalRow =
            (updateColumn ? otherOffset : pivotOffset) + row;
        const size_t globalColumn =
            (updateColumn ? pivotOffset : otherOffset) + column;
        if (globalRow < numNodes && globalColumn < numNodes) {
            const size_t index = globalRow * numNodes + globalColumn;
            dist[index] = tile[row][column];
            if (lastIntermediate[item] >= 0) {
                path[index] = static_cast<unsigned int>(pivotOffset +
                                                        lastIntermediate[item]);
            } else if (round == 0) {
                path[index] = static_cast<unsigned int>(globalRow);
            }
        }
    }
}

// Phase 3 is a tiled min-plus product.  All non-pivot output tiles are
// independent, and each of their 256 threads computes four output elements.
__global__ __launch_bounds__(TILE_SIZE * BLOCK_ROWS)
void updateRemainingTiles(unsigned int* __restrict__ dist,
                          unsigned int* __restrict__ path,
                          const size_t numNodes, const int round) {
    __shared__ unsigned int columnTile[TILE_SIZE][TILE_SIZE + 1];
    __shared__ unsigned int rowTile[TILE_SIZE][TILE_SIZE + 1];

    const int column = static_cast<int>(threadIdx.x);
    const int firstRow = static_cast<int>(threadIdx.y);
    const int outputTileColumn = static_cast<int>(blockIdx.x) +
                                 (static_cast<int>(blockIdx.x) >= round);
    const int outputTileRow = static_cast<int>(blockIdx.y) +
                              (static_cast<int>(blockIdx.y) >= round);
    const size_t pivotOffset = static_cast<size_t>(round) * TILE_SIZE;
    const size_t outputColumnOffset =
        static_cast<size_t>(outputTileColumn) * TILE_SIZE;
    const size_t outputRowOffset =
        static_cast<size_t>(outputTileRow) * TILE_SIZE;
    unsigned int values[ROWS_PER_THREAD];
    int lastIntermediate[ROWS_PER_THREAD];

#pragma unroll
    for (int item = 0; item < ROWS_PER_THREAD; ++item) {
        const int row = firstRow + item * BLOCK_ROWS;
        const size_t outputRow = outputRowOffset + row;
        const size_t pivotColumn = pivotOffset + column;
        columnTile[row][column] =
            (outputRow < numNodes && pivotColumn < numNodes)
                ? dist[outputRow * numNodes + pivotColumn]
                : INF;

        const size_t pivotRow = pivotOffset + row;
        const size_t outputColumn = outputColumnOffset + column;
        rowTile[row][column] =
            (pivotRow < numNodes && outputColumn < numNodes)
                ? dist[pivotRow * numNodes + outputColumn]
                : INF;

        values[item] = (outputRow < numNodes && outputColumn < numNodes)
                           ? dist[outputRow * numNodes + outputColumn]
                           : INF;
        lastIntermediate[item] = -1;
    }
    __syncthreads();

#pragma unroll
    for (int k = 0; k < TILE_SIZE; ++k) {
#pragma unroll
        for (int item = 0; item < ROWS_PER_THREAD; ++item) {
            const int row = firstRow + item * BLOCK_ROWS;
            const unsigned int candidate =
                columnTile[row][k] + rowTile[k][column];
            if (candidate < values[item]) {
                values[item] = candidate;
                lastIntermediate[item] = k;
            }
        }
    }

#pragma unroll
    for (int item = 0; item < ROWS_PER_THREAD; ++item) {
        const int row = firstRow + item * BLOCK_ROWS;
        const size_t globalRow = outputRowOffset + row;
        const size_t globalColumn = outputColumnOffset + column;
        if (globalRow < numNodes && globalColumn < numNodes) {
            const size_t index = globalRow * numNodes + globalColumn;
            dist[index] = values[item];
            if (lastIntermediate[item] >= 0) {
                path[index] = static_cast<unsigned int>(pivotOffset +
                                                        lastIntermediate[item]);
            } else if (round == 0) {
                path[index] = static_cast<unsigned int>(globalRow);
            }
        }
    }
}

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes) {
    if (numNodes == 0) {
        return;
    }

    const size_t elements = numNodes * numNodes;
    const size_t matrixBytes = elements * sizeof(unsigned int);
    unsigned int* deviceMatrices = nullptr;
    CUDA_CHECK(cudaMalloc(&deviceMatrices, matrixBytes * 2));
    unsigned int* const deviceDist = deviceMatrices;
    unsigned int* const devicePath = deviceMatrices + elements;

    CUDA_CHECK(cudaMemcpy(deviceDist, dist.data(), matrixBytes,
                          cudaMemcpyHostToDevice));

    const int tileCount = static_cast<int>((numNodes + TILE_SIZE - 1) /
                                           TILE_SIZE);
    const dim3 threads(TILE_SIZE, BLOCK_ROWS);
    for (int round = 0; round < tileCount; ++round) {
        updatePivotTile<<<1, threads>>>(deviceDist, devicePath, numNodes, round);

        if (tileCount > 1) {
            const dim3 rowAndColumnGrid(tileCount - 1, 2);
            updatePivotRowAndColumn<<<rowAndColumnGrid, threads>>>(
                deviceDist, devicePath, numNodes, round);

            const dim3 remainingGrid(tileCount - 1, tileCount - 1);
            updateRemainingTiles<<<remainingGrid, threads>>>(
                deviceDist, devicePath, numNodes, round);
        }
    }
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(dist.data(), deviceDist, matrixBytes,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(path.data(), devicePath, matrixBytes,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(deviceMatrices));
}

bool validateResult(const std::vector<unsigned int>& dist, const size_t numNodes) {
    // Basic sanity checks
    
    // 1. Diagonal should be zero
    for (size_t i = 0; i < numNodes; ++i) {
        if (dist[idx2(i, i, numNodes)] != 0) {
            printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", i, i);
            return false;
        }
    }
    
    // 2. Triangle inequality: dist[i][k] + dist[k][j] >= dist[i][j]
    // Check a sample of paths to avoid O(n^3) validation time
    for (size_t i = 0; i < std::min(numNodes, static_cast<size_t>(10)); ++i) {
        for (size_t j = 0; j < std::min(numNodes, static_cast<size_t>(10)); ++j) {
            for (size_t k = 0; k < numNodes; ++k) {
                const unsigned int distIJ = dist[idx2(j, i, numNodes)];
                const unsigned int distIK = dist[idx2(k, i, numNodes)];
                const unsigned int distKJ = dist[idx2(j, k, numNodes)];
                
                // Check for overflow before addition
                if (distIK < INF && distKJ < INF) {
                    if (distIK + distKJ < distIJ) {
                        printf("Validation failed: triangle inequality violated at [%zu,%zu,%zu]\n", 
                               i, j, k);
                        return false;
                    }
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

int main(int argc, char** argv) {
    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numNodes = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }
    
    printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
    printf("Number of nodes: %zu\n", numNodes);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    // Allocate matrices
    std::vector<unsigned int> dist(numNodes * numNodes);
    std::vector<unsigned int> path(numNodes * numNodes);
    
    // Initialize
    printf("Initializing graph...\n");
    initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);

    // Establish the CUDA context before the timed region.  Context creation is
    // runtime setup rather than part of the shortest-path computation.
    CUDA_CHECK(cudaFree(nullptr));
    
    // Run Floyd-Warshall
    printf("Computing shortest paths...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    floydWarshall(dist, path, numNodes);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate operations per second
    // Floyd-Warshall has O(n³) complexity
    double ops = (double)numNodes * numNodes * numNodes;
    double gflops = ops / (duration.count() / 1000.0) / 1e9;
    printf("Performance: %.3f GOPS\n", gflops);
    
    // Print results for external validation (integer hash-based)
    if (printResults) {
        print_results_int(dist, "DistanceMatrix");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(dist, numNodes);
        
        if (valid) {
            printf("Validation: PASSED\n");
            return 0;
        } else {
            printf("Validation: FAILED\n");
            return 1;
        }
    }
    
    return 0;
}
