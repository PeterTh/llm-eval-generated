#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;
constexpr unsigned int TILE_SIZE = 32;
constexpr unsigned int TILE_STRIDE = TILE_SIZE + 1;

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

[[noreturn]] void cudaFailure(const cudaError_t status, const char* operation) {
    fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(status));
    std::exit(EXIT_FAILURE);
}

#define CUDA_CHECK(call) \
    do { \
        const cudaError_t status = (call); \
        if (status != cudaSuccess) { \
            cudaFailure(status, #call); \
        } \
    } while (false)

// One block processes the diagonal tile for the current range of intermediate
// vertices.  The tile is kept in shared memory so every intermediate vertex is
// reused by all 1024 threads in the block.
__global__ void floydWarshallPivotKernel(unsigned int* const dist,
                                         unsigned int* const path,
                                         const size_t numNodes,
                                         const size_t pivotStart,
                                         const unsigned int intermediateCount) {
    __shared__ unsigned int tile[TILE_SIZE * TILE_STRIDE];

    const unsigned int localRow = threadIdx.x;
    const unsigned int localCol = threadIdx.y;
    const size_t row = pivotStart + localRow;
    const size_t col = pivotStart + localCol;
    const bool valid = row < numNodes && col < numNodes;
    const size_t matrixIndex = col * numNodes + row;

    tile[localCol * TILE_STRIDE + localRow] = valid ? dist[matrixIndex] : INF;
    unsigned int pathValue = valid ? path[matrixIndex] : 0U;
    __syncthreads();

    for (unsigned int intermediate = 0; intermediate < intermediateCount; ++intermediate) {
        const unsigned int candidate =
            tile[intermediate * TILE_STRIDE + localRow] +
            tile[localCol * TILE_STRIDE + intermediate];

        if (valid && candidate < tile[localCol * TILE_STRIDE + localRow]) {
            tile[localCol * TILE_STRIDE + localRow] = candidate;
            pathValue = static_cast<unsigned int>(pivotStart + intermediate);
        }
        __syncthreads();
    }

    if (valid) {
        dist[matrixIndex] = tile[localCol * TILE_STRIDE + localRow];
        path[matrixIndex] = pathValue;
    }
}

// Process the tiles in the pivot row and pivot column.  The output tile is
// updated in shared memory for each intermediate vertex, which preserves the
// data dependencies within the current tile while keeping the tile-wide work
// parallel.
__global__ void floydWarshallPivotRowColumnKernel(unsigned int* const dist,
                                                  unsigned int* const path,
                                                  const size_t numNodes,
                                                  const size_t pivotStart,
                                                  const size_t pivotTile,
                                                  const unsigned int intermediateCount) {
    const size_t outputTile = blockIdx.x;
    const bool rowPhase = blockIdx.y == 0;

    if (outputTile == pivotTile) {
        return;
    }

    __shared__ unsigned int pivot[TILE_SIZE * TILE_STRIDE];
    __shared__ unsigned int tile[TILE_SIZE * TILE_STRIDE];

    const unsigned int localRow = threadIdx.x;
    const unsigned int localCol = threadIdx.y;
    const size_t outputStart = outputTile * TILE_SIZE;
    const size_t rowStart = rowPhase ? pivotStart : outputStart;
    const size_t colStart = rowPhase ? outputStart : pivotStart;
    const size_t row = rowStart + localRow;
    const size_t col = colStart + localCol;
    const size_t pivotRow = pivotStart + localRow;
    const size_t pivotCol = pivotStart + localCol;
    const bool valid = row < numNodes && col < numNodes;
    const size_t matrixIndex = col * numNodes + row;

    pivot[localCol * TILE_STRIDE + localRow] =
        (pivotRow < numNodes && pivotCol < numNodes)
            ? dist[pivotCol * numNodes + pivotRow]
            : INF;
    tile[localCol * TILE_STRIDE + localRow] = valid ? dist[matrixIndex] : INF;
    unsigned int pathValue = valid ? path[matrixIndex] : 0U;
    __syncthreads();

    for (unsigned int intermediate = 0; intermediate < intermediateCount; ++intermediate) {
        const unsigned int left = rowPhase
            ? pivot[intermediate * TILE_STRIDE + localRow]
            : tile[intermediate * TILE_STRIDE + localRow];
        const unsigned int right = rowPhase
            ? tile[localCol * TILE_STRIDE + intermediate]
            : pivot[localCol * TILE_STRIDE + intermediate];
        const unsigned int candidate = left + right;

        if (valid && candidate < tile[localCol * TILE_STRIDE + localRow]) {
            tile[localCol * TILE_STRIDE + localRow] = candidate;
            pathValue = static_cast<unsigned int>(pivotStart + intermediate);
        }
        __syncthreads();
    }

    if (valid) {
        dist[matrixIndex] = tile[localCol * TILE_STRIDE + localRow];
        path[matrixIndex] = pathValue;
    }
}

// Process all tiles that are outside both the pivot row and pivot column.
// The pivot-row and pivot-column tiles are already closed over the current
// intermediate range by the preceding phase.
__global__ void floydWarshallRemainderKernel(unsigned int* const dist,
                                             unsigned int* const path,
                                             const size_t numNodes,
                                             const size_t pivotStart,
                                             const size_t pivotTile,
                                             const unsigned int intermediateCount) {
    const size_t rowTile = blockIdx.y;
    const size_t colTile = blockIdx.x;

    if (rowTile == pivotTile || colTile == pivotTile) {
        return;
    }

    __shared__ unsigned int pivotColumn[TILE_SIZE * TILE_STRIDE];
    __shared__ unsigned int pivotRow[TILE_SIZE * TILE_STRIDE];

    const unsigned int localRow = threadIdx.x;
    const unsigned int localCol = threadIdx.y;
    const size_t rowStart = rowTile * TILE_SIZE;
    const size_t colStart = colTile * TILE_SIZE;
    const size_t row = rowStart + localRow;
    const size_t col = colStart + localCol;
    const size_t intermediateRow = pivotStart + localRow;
    const size_t intermediateCol = pivotStart + localCol;
    const bool valid = row < numNodes && col < numNodes;
    const size_t matrixIndex = col * numNodes + row;

    // Store each input tile transposed in shared memory.  This makes the
    // global loads contiguous for the column-major matrix layout while the
    // inner loop reads both operands without shared-memory bank conflicts.
    pivotColumn[localRow * TILE_STRIDE + localCol] =
        (row < numNodes && intermediateCol < numNodes)
            ? dist[intermediateCol * numNodes + row]
            : INF;
    pivotRow[localCol * TILE_STRIDE + localRow] =
        (intermediateRow < numNodes && col < numNodes)
            ? dist[col * numNodes + intermediateRow]
            : INF;

    unsigned int value = valid ? dist[matrixIndex] : INF;
    unsigned int pathValue = valid ? path[matrixIndex] : 0U;
    __syncthreads();

    for (unsigned int intermediate = 0; intermediate < intermediateCount; ++intermediate) {
        const unsigned int candidate =
            pivotColumn[localRow * TILE_STRIDE + intermediate] +
            pivotRow[localCol * TILE_STRIDE + intermediate];
        if (valid && candidate < value) {
            value = candidate;
            pathValue = static_cast<unsigned int>(pivotStart + intermediate);
        }
    }

    if (valid) {
        dist[matrixIndex] = value;
        path[matrixIndex] = pathValue;
    }
}

void floydWarshall(std::vector<unsigned int>& dist, 
                   std::vector<unsigned int>& path, 
                   const size_t numNodes) {
    if (numNodes == 0) {
        return;
    }

    const size_t matrixElements = numNodes * numNodes;
    const size_t matrixBytes = matrixElements * sizeof(unsigned int);
    unsigned int* deviceDist = nullptr;
    unsigned int* devicePath = nullptr;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceDist), matrixBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&devicePath), matrixBytes));
    CUDA_CHECK(cudaMemcpy(deviceDist, dist.data(), matrixBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(devicePath, path.data(), matrixBytes, cudaMemcpyHostToDevice));

    const size_t tileCount = (numNodes + TILE_SIZE - 1) / TILE_SIZE;
    if (tileCount > std::numeric_limits<unsigned int>::max()) {
        cudaFree(devicePath);
        cudaFree(deviceDist);
        fprintf(stderr, "CUDA grid is too large for %zu nodes\n", numNodes);
        std::exit(EXIT_FAILURE);
    }

    const dim3 block(TILE_SIZE, TILE_SIZE, 1);
    const dim3 phase2Grid(static_cast<unsigned int>(tileCount), 2, 1);
    const dim3 phase3Grid(static_cast<unsigned int>(tileCount),
                          static_cast<unsigned int>(tileCount), 1);

    for (size_t pivotTile = 0; pivotTile < tileCount; ++pivotTile) {
        const size_t pivotStart = pivotTile * TILE_SIZE;
        const unsigned int intermediateCount = static_cast<unsigned int>(
            std::min(static_cast<size_t>(TILE_SIZE), numNodes - pivotStart));

        floydWarshallPivotKernel<<<1, block>>>(deviceDist, devicePath, numNodes,
                                                pivotStart, intermediateCount);
        CUDA_CHECK(cudaGetLastError());

        floydWarshallPivotRowColumnKernel<<<phase2Grid, block>>>(
            deviceDist, devicePath, numNodes, pivotStart, pivotTile, intermediateCount);
        CUDA_CHECK(cudaGetLastError());

        floydWarshallRemainderKernel<<<phase3Grid, block>>>(
            deviceDist, devicePath, numNodes, pivotStart, pivotTile, intermediateCount);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(dist.data(), deviceDist, matrixBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(path.data(), devicePath, matrixBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(devicePath));
    CUDA_CHECK(cudaFree(deviceDist));
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
    
    // Run Floyd-Warshall
    printf("Computing shortest paths...\n");
    // Warm up the CUDA runtime so one-time context creation is not reported as
    // part of the algorithm's execution time.
    CUDA_CHECK(cudaFree(nullptr));
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
