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
constexpr unsigned int TILE_SIZE = 32;

// Index calculation for flattened 2D array
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

void checkCuda(const cudaError_t status, const char* const operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
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

// The matrix is stored as dist[column * numNodes + row].  Mapping the row to
// threadIdx.x makes a warp access consecutive words in global memory.
__device__ __forceinline__ size_t deviceIdx2(const size_t row, const size_t column,
                                             const size_t numNodes) noexcept {
    return column * numNodes + row;
}

__global__ void floydWarshallPivotKernel(unsigned int* const dist,
                                         unsigned int* const path,
                                         const size_t numNodes,
                                         const size_t kBase) {
    __shared__ unsigned int pivot[TILE_SIZE][TILE_SIZE];

    const unsigned int localRow = threadIdx.x;
    const unsigned int localColumn = threadIdx.y;
    const size_t row = kBase + localRow;
    const size_t column = kBase + localColumn;
    const unsigned int activeSize = static_cast<unsigned int>(
        min(static_cast<size_t>(TILE_SIZE), numNodes - kBase));

    if (row < numNodes && column < numNodes) {
        pivot[localColumn][localRow] = dist[deviceIdx2(row, column, numNodes)];
    } else {
        pivot[localColumn][localRow] = INF;
    }
    __syncthreads();

    unsigned int value = pivot[localColumn][localRow];
    for (unsigned int intermediate = 0; intermediate < activeSize; ++intermediate) {
        if (row < numNodes && column < numNodes) {
            const unsigned int candidate =
                pivot[intermediate][localRow] + pivot[localColumn][intermediate];
            if (candidate < value) {
                value = candidate;
                pivot[localColumn][localRow] = value;
                path[deviceIdx2(row, column, numNodes)] =
                    static_cast<unsigned int>(kBase + intermediate);
            }
        }
        __syncthreads();
    }
    if (row < numNodes && column < numNodes) {
        dist[deviceIdx2(row, column, numNodes)] = value;
    }
}

__global__ void floydWarshallRowKernel(unsigned int* const dist,
                                        unsigned int* const path,
                                        const size_t numNodes,
                                        const size_t kBase,
                                        const size_t pivotBlock) {
    __shared__ unsigned int pivot[TILE_SIZE][TILE_SIZE];
    __shared__ unsigned int target[TILE_SIZE][TILE_SIZE];

    const size_t targetBlock = static_cast<size_t>(blockIdx.x);
    if (targetBlock == pivotBlock) {
        return;
    }

    const unsigned int localRow = threadIdx.x;
    const unsigned int localColumn = threadIdx.y;
    const size_t row = kBase + localRow;
    const size_t column = targetBlock * TILE_SIZE + localColumn;
    const unsigned int activeSize = static_cast<unsigned int>(
        min(static_cast<size_t>(TILE_SIZE), numNodes - kBase));

    if (row < numNodes) {
        if (column < numNodes) {
            target[localColumn][localRow] = dist[deviceIdx2(row, column, numNodes)];
        } else {
            target[localColumn][localRow] = INF;
        }
    } else {
        target[localColumn][localRow] = INF;
    }

    const size_t pivotRow = kBase + localRow;
    const size_t pivotColumn = kBase + localColumn;
    if (pivotRow < numNodes && pivotColumn < numNodes) {
        pivot[localColumn][localRow] = dist[deviceIdx2(pivotRow, pivotColumn, numNodes)];
    } else {
        pivot[localColumn][localRow] = INF;
    }
    __syncthreads();

    unsigned int value = target[localColumn][localRow];
    for (unsigned int intermediate = 0; intermediate < activeSize; ++intermediate) {
        if (row < numNodes && column < numNodes) {
            const unsigned int candidate =
                pivot[intermediate][localRow] + target[localColumn][intermediate];
            if (candidate < value) {
                value = candidate;
                target[localColumn][localRow] = value;
                path[deviceIdx2(row, column, numNodes)] =
                    static_cast<unsigned int>(kBase + intermediate);
            }
        }
        __syncthreads();
    }
    if (row < numNodes && column < numNodes) {
        dist[deviceIdx2(row, column, numNodes)] = value;
    }
}

__global__ void floydWarshallColumnKernel(unsigned int* const dist,
                                           unsigned int* const path,
                                           const size_t numNodes,
                                           const size_t kBase,
                                           const size_t pivotBlock) {
    __shared__ unsigned int pivot[TILE_SIZE][TILE_SIZE];
    __shared__ unsigned int target[TILE_SIZE][TILE_SIZE];

    const size_t targetBlock = static_cast<size_t>(blockIdx.x);
    if (targetBlock == pivotBlock) {
        return;
    }

    const unsigned int localRow = threadIdx.x;
    const unsigned int localColumn = threadIdx.y;
    const size_t row = targetBlock * TILE_SIZE + localRow;
    const size_t column = kBase + localColumn;
    const unsigned int activeSize = static_cast<unsigned int>(
        min(static_cast<size_t>(TILE_SIZE), numNodes - kBase));

    if (row < numNodes) {
        if (column < numNodes) {
            target[localColumn][localRow] = dist[deviceIdx2(row, column, numNodes)];
        } else {
            target[localColumn][localRow] = INF;
        }
    } else {
        target[localColumn][localRow] = INF;
    }

    const size_t pivotRow = kBase + localRow;
    const size_t pivotColumn = kBase + localColumn;
    if (pivotRow < numNodes && pivotColumn < numNodes) {
        pivot[localColumn][localRow] = dist[deviceIdx2(pivotRow, pivotColumn, numNodes)];
    } else {
        pivot[localColumn][localRow] = INF;
    }
    __syncthreads();

    unsigned int value = target[localColumn][localRow];
    for (unsigned int intermediate = 0; intermediate < activeSize; ++intermediate) {
        if (row < numNodes && column < numNodes) {
            const unsigned int candidate =
                target[intermediate][localRow] + pivot[localColumn][intermediate];
            if (candidate < value) {
                value = candidate;
                target[localColumn][localRow] = value;
                path[deviceIdx2(row, column, numNodes)] =
                    static_cast<unsigned int>(kBase + intermediate);
            }
        }
        __syncthreads();
    }
    if (row < numNodes && column < numNodes) {
        dist[deviceIdx2(row, column, numNodes)] = value;
    }
}

__global__ void floydWarshallRemainderKernel(unsigned int* const dist,
                                              unsigned int* const path,
                                              const size_t numNodes,
                                              const size_t kBase,
                                              const size_t pivotBlock) {
    const size_t rowBlock = static_cast<size_t>(blockIdx.x);
    const size_t columnBlock = static_cast<size_t>(blockIdx.y);
    if (rowBlock == pivotBlock || columnBlock == pivotBlock) {
        return;
    }

    __shared__ unsigned int left[TILE_SIZE][TILE_SIZE];
    __shared__ unsigned int top[TILE_SIZE][TILE_SIZE];
    __shared__ unsigned int tile[TILE_SIZE][TILE_SIZE];

    const unsigned int localRow = threadIdx.x;
    const unsigned int localColumn = threadIdx.y;
    const size_t row = rowBlock * TILE_SIZE + localRow;
    const size_t column = columnBlock * TILE_SIZE + localColumn;
    const unsigned int activeSize = static_cast<unsigned int>(
        min(static_cast<size_t>(TILE_SIZE), numNodes - kBase));

    if (row < numNodes) {
        if (column < numNodes) {
            tile[localColumn][localRow] = dist[deviceIdx2(row, column, numNodes)];
        } else {
            tile[localColumn][localRow] = INF;
        }
    } else {
        tile[localColumn][localRow] = INF;
    }

    const size_t leftColumn = kBase + localColumn;
    if (row < numNodes && leftColumn < numNodes) {
        left[localColumn][localRow] = dist[deviceIdx2(row, leftColumn, numNodes)];
    } else {
        left[localColumn][localRow] = INF;
    }

    const size_t topRow = kBase + localRow;
    if (topRow < numNodes && column < numNodes) {
        top[localColumn][localRow] = dist[deviceIdx2(topRow, column, numNodes)];
    } else {
        top[localColumn][localRow] = INF;
    }
    __syncthreads();

    if (row < numNodes && column < numNodes) {
        unsigned int value = tile[localColumn][localRow];
        for (unsigned int intermediate = 0; intermediate < activeSize; ++intermediate) {
            const unsigned int candidate =
                left[intermediate][localRow] + top[localColumn][intermediate];
            if (candidate < value) {
                value = candidate;
                tile[localColumn][localRow] = value;
                path[deviceIdx2(row, column, numNodes)] =
                    static_cast<unsigned int>(kBase + intermediate);
            }
        }
        dist[deviceIdx2(row, column, numNodes)] = value;
    }
}

void floydWarshall(std::vector<unsigned int>& dist, 
                   std::vector<unsigned int>& path, 
                   const size_t numNodes) {
    if (numNodes == 0) {
        return;
    }

    const size_t elementCount = numNodes * numNodes;
    const size_t bytes = elementCount * sizeof(unsigned int);
    unsigned int* deviceDist = nullptr;
    unsigned int* devicePath = nullptr;

    checkCuda(cudaMalloc(reinterpret_cast<void**>(&deviceDist), bytes), "allocating distance matrix");
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&devicePath), bytes), "allocating path matrix");
    checkCuda(cudaMemcpy(deviceDist, dist.data(), bytes, cudaMemcpyHostToDevice),
              "copying distance matrix to device");
    checkCuda(cudaMemcpy(devicePath, path.data(), bytes, cudaMemcpyHostToDevice),
              "copying path matrix to device");

    const size_t tileCount = (numNodes + TILE_SIZE - 1) / TILE_SIZE;
    const dim3 tileThreads(TILE_SIZE, TILE_SIZE, 1);
    const dim3 oneTile(1, 1, 1);
    const dim3 rowOrColumnGrid(static_cast<unsigned int>(tileCount), 1, 1);
    const dim3 remainderGrid(static_cast<unsigned int>(tileCount),
                             static_cast<unsigned int>(tileCount), 1);

    for (size_t pivotBlock = 0; pivotBlock < tileCount; ++pivotBlock) {
        const size_t kBase = pivotBlock * TILE_SIZE;

        floydWarshallPivotKernel<<<oneTile, tileThreads>>>(
            deviceDist, devicePath, numNodes, kBase);
        checkCuda(cudaGetLastError(), "launching pivot kernel");

        floydWarshallRowKernel<<<rowOrColumnGrid, tileThreads>>>(
            deviceDist, devicePath, numNodes, kBase, pivotBlock);
        checkCuda(cudaGetLastError(), "launching pivot-row kernel");

        floydWarshallColumnKernel<<<rowOrColumnGrid, tileThreads>>>(
            deviceDist, devicePath, numNodes, kBase, pivotBlock);
        checkCuda(cudaGetLastError(), "launching pivot-column kernel");

        floydWarshallRemainderKernel<<<remainderGrid, tileThreads>>>(
            deviceDist, devicePath, numNodes, kBase, pivotBlock);
        checkCuda(cudaGetLastError(), "launching remainder kernel");
    }

    checkCuda(cudaMemcpy(dist.data(), deviceDist, bytes, cudaMemcpyDeviceToHost),
              "copying distance matrix to host");
    checkCuda(cudaMemcpy(path.data(), devicePath, bytes, cudaMemcpyDeviceToHost),
              "copying path matrix to host");
    checkCuda(cudaFree(devicePath), "freeing path matrix");
    checkCuda(cudaFree(deviceDist), "freeing distance matrix");
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
