#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <vector>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;
constexpr int TILE_SIZE = 32;
constexpr int ROWS_PER_THREAD = 4;
constexpr int THREADS_X = TILE_SIZE;
constexpr int THREADS_Y = TILE_SIZE / ROWS_PER_THREAD;

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

void initializePathMatrix(std::vector<unsigned int>& path, const size_t numNodes) {
    for (size_t j = 0; j < numNodes; ++j) {
        for (size_t i = 0; i < numNodes; ++i) {
            path[idx2(i, j, numNodes)] = j;
            path[idx2(j, i, numNodes)] = i;
        }
        path[idx2(j, j, numNodes)] = j;
    }
}

[[noreturn]] void cudaFailure(const cudaError_t error, const char* expression,
                               const char* file, const int line) {
    std::fprintf(stderr, "CUDA error at %s:%d (%s): %s\n", file, line,
                 expression, cudaGetErrorString(error));
    std::exit(EXIT_FAILURE);
}

#define CUDA_CHECK(expression) \
    do { \
        const cudaError_t cudaStatus = (expression); \
        if (cudaStatus != cudaSuccess) { \
            cudaFailure(cudaStatus, #expression, __FILE__, __LINE__); \
        } \
    } while (false)

// Each thread owns four rows of a 32x32 tile. This keeps the tile-wide
// synchronization cost low while avoiding a 1024-thread block.
__global__ void floydWarshallPivotKernel(unsigned int* const dist,
                                          unsigned int* const path,
                                          const size_t numNodes,
                                          const size_t round) {
    __shared__ unsigned int tile[TILE_SIZE][TILE_SIZE];

    const int column = static_cast<int>(threadIdx.x);
    const int firstRow = static_cast<int>(threadIdx.y);
    unsigned int localPath[ROWS_PER_THREAD] = {};

    #pragma unroll
    for (int rowOffset = 0; rowOffset < ROWS_PER_THREAD; ++rowOffset) {
        const int row = firstRow + rowOffset * THREADS_Y;
        const size_t globalRow = round * TILE_SIZE + static_cast<size_t>(row);
        const size_t globalColumn = round * TILE_SIZE + static_cast<size_t>(column);
        if (globalRow < numNodes && globalColumn < numNodes) {
            const size_t matrixIndex = globalRow * numNodes + globalColumn;
            tile[row][column] = dist[matrixIndex];
            localPath[rowOffset] = path[matrixIndex];
        } else {
            tile[row][column] = INF;
        }
    }
    __syncthreads();

    #pragma unroll
    for (int intermediate = 0; intermediate < TILE_SIZE; ++intermediate) {
        #pragma unroll
        for (int rowOffset = 0; rowOffset < ROWS_PER_THREAD; ++rowOffset) {
            const int row = firstRow + rowOffset * THREADS_Y;
            const unsigned int newDistance =
                tile[row][intermediate] + tile[intermediate][column];
            if (newDistance < tile[row][column]) {
                tile[row][column] = newDistance;
                localPath[rowOffset] =
                    static_cast<unsigned int>(round * TILE_SIZE + intermediate);
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int rowOffset = 0; rowOffset < ROWS_PER_THREAD; ++rowOffset) {
        const int row = firstRow + rowOffset * THREADS_Y;
        const size_t globalRow = round * TILE_SIZE + static_cast<size_t>(row);
        const size_t globalColumn = round * TILE_SIZE + static_cast<size_t>(column);
        if (globalRow < numNodes && globalColumn < numNodes) {
            const size_t matrixIndex = globalRow * numNodes + globalColumn;
            dist[matrixIndex] = tile[row][column];
            path[matrixIndex] = localPath[rowOffset];
        }
    }
}

// Update one tile in the pivot row or pivot column. The pivot tile is already
// closed over all intermediate vertices in the current round.
__global__ void floydWarshallPivotRowColumnKernel(unsigned int* const dist,
                                                   unsigned int* const path,
                                                   const size_t numNodes,
                                                   const size_t round) {
    __shared__ unsigned int pivotTile[TILE_SIZE][TILE_SIZE];
    __shared__ unsigned int target[TILE_SIZE][TILE_SIZE];

    const bool rowPhase = blockIdx.x < (numNodes + TILE_SIZE - 1) / TILE_SIZE;
    const size_t tileCount = (numNodes + TILE_SIZE - 1) / TILE_SIZE;
    const size_t actualTargetTile = rowPhase
        ? static_cast<size_t>(blockIdx.x)
        : static_cast<size_t>(blockIdx.x) - tileCount;

    // The pivot tile was handled by the first phase. Returning the whole block
    // here is safe because this branch is uniform for every thread.
    if (actualTargetTile == round) {
        return;
    }

    const int column = static_cast<int>(threadIdx.x);
    const int firstRow = static_cast<int>(threadIdx.y);
    unsigned int localPath[ROWS_PER_THREAD] = {};

    #pragma unroll
    for (int rowOffset = 0; rowOffset < ROWS_PER_THREAD; ++rowOffset) {
        const int row = firstRow + rowOffset * THREADS_Y;
        const size_t pivotRow = round * TILE_SIZE + static_cast<size_t>(row);
        const size_t pivotColumn = round * TILE_SIZE + static_cast<size_t>(column);
        const size_t targetRow = (rowPhase ? round : actualTargetTile) * TILE_SIZE
                               + static_cast<size_t>(row);
        const size_t targetColumn = (rowPhase ? actualTargetTile : round) * TILE_SIZE
                                  + static_cast<size_t>(column);

        if (pivotRow < numNodes && pivotColumn < numNodes) {
            pivotTile[row][column] = dist[pivotRow * numNodes + pivotColumn];
        } else {
            pivotTile[row][column] = INF;
        }

        if (targetRow < numNodes && targetColumn < numNodes) {
            const size_t matrixIndex = targetRow * numNodes + targetColumn;
            target[row][column] = dist[matrixIndex];
            localPath[rowOffset] = path[matrixIndex];
        } else {
            target[row][column] = INF;
        }
    }
    __syncthreads();

    #pragma unroll
    for (int intermediate = 0; intermediate < TILE_SIZE; ++intermediate) {
        #pragma unroll
        for (int rowOffset = 0; rowOffset < ROWS_PER_THREAD; ++rowOffset) {
            const int row = firstRow + rowOffset * THREADS_Y;
            const unsigned int newDistance = rowPhase
                ? pivotTile[row][intermediate] + target[intermediate][column]
                : target[row][intermediate] + pivotTile[intermediate][column];
            if (newDistance < target[row][column]) {
                target[row][column] = newDistance;
                localPath[rowOffset] =
                    static_cast<unsigned int>(round * TILE_SIZE + intermediate);
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int rowOffset = 0; rowOffset < ROWS_PER_THREAD; ++rowOffset) {
        const int row = firstRow + rowOffset * THREADS_Y;
        const size_t targetRow = (rowPhase ? round : actualTargetTile) * TILE_SIZE
                               + static_cast<size_t>(row);
        const size_t targetColumn = (rowPhase ? actualTargetTile : round) * TILE_SIZE
                                  + static_cast<size_t>(column);
        if (targetRow < numNodes && targetColumn < numNodes) {
            const size_t matrixIndex = targetRow * numNodes + targetColumn;
            dist[matrixIndex] = target[row][column];
            path[matrixIndex] = localPath[rowOffset];
        }
    }
}

// Update all tiles that are neither in the pivot row nor in the pivot column.
// The row and column tiles produced by phase two are read-only inputs here.
__global__ void floydWarshallRemainderKernel(unsigned int* const dist,
                                              unsigned int* const path,
                                              const size_t numNodes,
                                              const size_t round) {
    const size_t rowTile = static_cast<size_t>(blockIdx.y);
    const size_t columnTile = static_cast<size_t>(blockIdx.x);
    if (rowTile == round || columnTile == round) {
        return;
    }

    __shared__ unsigned int rowTileData[TILE_SIZE][TILE_SIZE];
    __shared__ unsigned int columnTileData[TILE_SIZE][TILE_SIZE];
    __shared__ unsigned int target[TILE_SIZE][TILE_SIZE];

    const int column = static_cast<int>(threadIdx.x);
    const int firstRow = static_cast<int>(threadIdx.y);
    unsigned int localPath[ROWS_PER_THREAD] = {};

    #pragma unroll
    for (int rowOffset = 0; rowOffset < ROWS_PER_THREAD; ++rowOffset) {
        const int row = firstRow + rowOffset * THREADS_Y;
        const size_t globalRow = rowTile * TILE_SIZE + static_cast<size_t>(row);
        const size_t globalColumn = columnTile * TILE_SIZE + static_cast<size_t>(column);
        const size_t pivotRow = round * TILE_SIZE + static_cast<size_t>(row);
        const size_t pivotColumn = round * TILE_SIZE + static_cast<size_t>(column);

        if (globalRow < numNodes && pivotColumn < numNodes) {
            rowTileData[row][column] = dist[globalRow * numNodes + pivotColumn];
        } else {
            rowTileData[row][column] = INF;
        }

        if (pivotRow < numNodes && globalColumn < numNodes) {
            columnTileData[row][column] = dist[pivotRow * numNodes + globalColumn];
        } else {
            columnTileData[row][column] = INF;
        }

        if (globalRow < numNodes && globalColumn < numNodes) {
            const size_t matrixIndex = globalRow * numNodes + globalColumn;
            target[row][column] = dist[matrixIndex];
            localPath[rowOffset] = path[matrixIndex];
        } else {
            target[row][column] = INF;
        }
    }
    __syncthreads();

    #pragma unroll
    for (int intermediate = 0; intermediate < TILE_SIZE; ++intermediate) {
        #pragma unroll
        for (int rowOffset = 0; rowOffset < ROWS_PER_THREAD; ++rowOffset) {
            const int row = firstRow + rowOffset * THREADS_Y;
            const unsigned int newDistance =
                rowTileData[row][intermediate] + columnTileData[intermediate][column];
            if (newDistance < target[row][column]) {
                target[row][column] = newDistance;
                localPath[rowOffset] =
                    static_cast<unsigned int>(round * TILE_SIZE + intermediate);
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int rowOffset = 0; rowOffset < ROWS_PER_THREAD; ++rowOffset) {
        const int row = firstRow + rowOffset * THREADS_Y;
        const size_t globalRow = rowTile * TILE_SIZE + static_cast<size_t>(row);
        const size_t globalColumn = columnTile * TILE_SIZE + static_cast<size_t>(column);
        if (globalRow < numNodes && globalColumn < numNodes) {
            const size_t matrixIndex = globalRow * numNodes + globalColumn;
            dist[matrixIndex] = target[row][column];
            path[matrixIndex] = localPath[rowOffset];
        }
    }
}

void floydWarshall(unsigned int* const deviceDist, unsigned int* const devicePath,
                   const size_t numNodes) {
    if (numNodes == 0) {
        return;
    }

    const size_t tileCount = (numNodes + TILE_SIZE - 1) / TILE_SIZE;
    const dim3 block(THREADS_X, THREADS_Y);

    for (size_t round = 0; round < tileCount; ++round) {
        floydWarshallPivotKernel<<<1, block>>>(deviceDist, devicePath, numNodes, round);
        CUDA_CHECK(cudaGetLastError());

        const unsigned int rowColumnBlocks = static_cast<unsigned int>(tileCount * 2);
        floydWarshallPivotRowColumnKernel<<<rowColumnBlocks, block>>>(
            deviceDist, devicePath, numNodes, round);
        CUDA_CHECK(cudaGetLastError());

        const dim3 remainderGrid(static_cast<unsigned int>(tileCount),
                                 static_cast<unsigned int>(tileCount));
        floydWarshallRemainderKernel<<<remainderGrid, block>>>(
            deviceDist, devicePath, numNodes, round);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaDeviceSynchronize());
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
    initializePathMatrix(path, numNodes);
    
    const size_t matrixElements = numNodes * numNodes;
    const size_t matrixBytes = matrixElements * sizeof(unsigned int);
    unsigned int* deviceDist = nullptr;
    unsigned int* devicePath = nullptr;

    if (matrixElements != 0) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceDist), matrixBytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&devicePath), matrixBytes));
        CUDA_CHECK(cudaMemcpy(deviceDist, dist.data(), matrixBytes,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(devicePath, path.data(), matrixBytes,
                              cudaMemcpyHostToDevice));
    }

    // Run the CUDA Floyd-Warshall implementation. Host/device transfers are
    // outside the timed region so the reported time measures GPU computation.
    printf("Computing shortest paths...\n");
    cudaEvent_t startEvent = nullptr;
    cudaEvent_t stopEvent = nullptr;
    CUDA_CHECK(cudaEventCreate(&startEvent));
    CUDA_CHECK(cudaEventCreate(&stopEvent));
    CUDA_CHECK(cudaEventRecord(startEvent));

    floydWarshall(deviceDist, devicePath, numNodes);

    CUDA_CHECK(cudaEventRecord(stopEvent));
    CUDA_CHECK(cudaEventSynchronize(stopEvent));
    float elapsedMilliseconds = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedMilliseconds, startEvent, stopEvent));
    CUDA_CHECK(cudaEventDestroy(startEvent));
    CUDA_CHECK(cudaEventDestroy(stopEvent));

    if (matrixElements != 0) {
        CUDA_CHECK(cudaMemcpy(dist.data(), deviceDist, matrixBytes,
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(path.data(), devicePath, matrixBytes,
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(deviceDist));
        CUDA_CHECK(cudaFree(devicePath));
    }

    printf("Computation time: %.3f ms\n", elapsedMilliseconds);
    
    // Calculate operations per second
    // Floyd-Warshall has O(n³) complexity
    double ops = (double)numNodes * numNodes * numNodes;
    double gflops = elapsedMilliseconds > 0.0F
        ? ops / (static_cast<double>(elapsedMilliseconds) / 1000.0) / 1e9
        : 0.0;
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
