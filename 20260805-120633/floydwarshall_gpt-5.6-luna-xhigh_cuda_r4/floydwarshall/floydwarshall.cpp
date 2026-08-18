#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;
constexpr unsigned int TILE_SIZE = 32;

[[noreturn]] void cudaError(const cudaError_t error, const char* operation,
                            const char* file, const int line) {
    fprintf(stderr, "CUDA error in %s at %s:%d: %s\n", operation, file, line,
            cudaGetErrorString(error));
    std::exit(EXIT_FAILURE);
}

#define CUDA_CHECK(operation)                                                     \
    do {                                                                           \
        const cudaError_t error__ = (operation);                                  \
        if (error__ != cudaSuccess) {                                             \
            cudaError(error__, #operation, __FILE__, __LINE__);                  \
        }                                                                          \
    } while (false)

__device__ __forceinline__ size_t deviceIdx(const unsigned int row,
                                            const unsigned int column,
                                            const unsigned int numNodes) {
    // The host implementation uses column-major storage: idx2(row, column, n).
    return static_cast<size_t>(column) * numNodes + row;
}

__global__ void floydWarshallPivot(unsigned int* __restrict__ dist,
                                   unsigned int* __restrict__ path,
                                   const unsigned int numNodes,
                                   const unsigned int pivotStart,
                                   const unsigned int pivotNodes) {
    __shared__ unsigned int tile[TILE_SIZE][TILE_SIZE];

    const unsigned int rowInTile = threadIdx.x;
    const unsigned int columnInTile = threadIdx.y;
    const unsigned int row = pivotStart + rowInTile;
    const unsigned int column = pivotStart + columnInTile;
    const bool valid = row < numNodes && column < numNodes;

    // Store as [column][row] in shared memory. threadIdx.x maps to row,
    // keeping global loads and shared-memory accesses coalesced.
    tile[columnInTile][rowInTile] = valid
        ? dist[deviceIdx(row, column, numNodes)]
        : INF;
    __syncthreads();

    unsigned int value = tile[columnInTile][rowInTile];
    unsigned int lastPath = 0;
    bool pathChanged = false;

    // Every thread owns one distance in the pivot tile. The synchronization
    // after each k is the dependency barrier required by Floyd-Warshall.
    #pragma unroll
    for (unsigned int k = 0; k < TILE_SIZE; ++k) {
        if (k < pivotNodes && valid) {
            const unsigned int via = tile[k][rowInTile] +
                                     tile[columnInTile][k];
            if (via < value) {
                value = via;
                lastPath = pivotStart + k;
                pathChanged = true;
            }
        }
        tile[columnInTile][rowInTile] = value;
        __syncthreads();
    }

    if (valid) {
        const size_t position = deviceIdx(row, column, numNodes);
        dist[position] = value;
        if (pathChanged) {
            path[position] = lastPath;
        }
    }
}

__global__ void floydWarshallPivotRow(unsigned int* __restrict__ dist,
                                      unsigned int* __restrict__ path,
                                      const unsigned int numNodes,
                                      const unsigned int pivotStart,
                                      const unsigned int pivotNodes) {
    __shared__ unsigned int pivot[TILE_SIZE][TILE_SIZE];
    __shared__ unsigned int tile[TILE_SIZE][TILE_SIZE];

    const unsigned int rowInTile = threadIdx.x;
    const unsigned int columnInTile = threadIdx.y;
    const unsigned int blockColumn = blockIdx.x;
    const unsigned int row = pivotStart + rowInTile;
    const unsigned int column = blockColumn * TILE_SIZE + columnInTile;
    const bool valid = row < numNodes && column < numNodes;

    if (blockColumn == pivotStart / TILE_SIZE) {
        return;
    }

    pivot[columnInTile][rowInTile] =
        (row < numNodes && pivotStart + columnInTile < numNodes)
            ? dist[deviceIdx(row, pivotStart + columnInTile, numNodes)]
            : INF;
    tile[columnInTile][rowInTile] = valid
        ? dist[deviceIdx(row, column, numNodes)]
        : INF;
    __syncthreads();

    unsigned int value = tile[columnInTile][rowInTile];
    unsigned int lastPath = 0;
    bool pathChanged = false;

    #pragma unroll
    for (unsigned int k = 0; k < TILE_SIZE; ++k) {
        if (k < pivotNodes && valid) {
            const unsigned int via = pivot[k][rowInTile] +
                                     tile[columnInTile][k];
            if (via < value) {
                value = via;
                lastPath = pivotStart + k;
                pathChanged = true;
            }
        }
        tile[columnInTile][rowInTile] = value;
        __syncthreads();
    }

    if (valid) {
        const size_t position = deviceIdx(row, column, numNodes);
        dist[position] = value;
        if (pathChanged) {
            path[position] = lastPath;
        }
    }
}

__global__ void floydWarshallPivotColumn(unsigned int* __restrict__ dist,
                                         unsigned int* __restrict__ path,
                                         const unsigned int numNodes,
                                         const unsigned int pivotStart,
                                         const unsigned int pivotNodes) {
    __shared__ unsigned int pivot[TILE_SIZE][TILE_SIZE];
    __shared__ unsigned int tile[TILE_SIZE][TILE_SIZE];

    const unsigned int rowInTile = threadIdx.x;
    const unsigned int columnInTile = threadIdx.y;
    const unsigned int blockRow = blockIdx.x;
    const unsigned int row = blockRow * TILE_SIZE + rowInTile;
    const unsigned int column = pivotStart + columnInTile;
    const bool valid = row < numNodes && column < numNodes;

    if (blockRow == pivotStart / TILE_SIZE) {
        return;
    }

    pivot[columnInTile][rowInTile] =
        (pivotStart + rowInTile < numNodes && column < numNodes)
            ? dist[deviceIdx(pivotStart + rowInTile, column, numNodes)]
            : INF;
    tile[columnInTile][rowInTile] = valid
        ? dist[deviceIdx(row, column, numNodes)]
        : INF;
    __syncthreads();

    unsigned int value = tile[columnInTile][rowInTile];
    unsigned int lastPath = 0;
    bool pathChanged = false;

    #pragma unroll
    for (unsigned int k = 0; k < TILE_SIZE; ++k) {
        if (k < pivotNodes && valid) {
            const unsigned int via = tile[k][rowInTile] +
                                     pivot[columnInTile][k];
            if (via < value) {
                value = via;
                lastPath = pivotStart + k;
                pathChanged = true;
            }
        }
        tile[columnInTile][rowInTile] = value;
        __syncthreads();
    }

    if (valid) {
        const size_t position = deviceIdx(row, column, numNodes);
        dist[position] = value;
        if (pathChanged) {
            path[position] = lastPath;
        }
    }
}

__global__ void floydWarshallRemainder(unsigned int* __restrict__ dist,
                                       unsigned int* __restrict__ path,
                                       const unsigned int numNodes,
                                       const unsigned int pivotStart,
                                       const unsigned int pivotNodes) {
    __shared__ unsigned int left[TILE_SIZE][TILE_SIZE];
    __shared__ unsigned int top[TILE_SIZE][TILE_SIZE];

    const unsigned int blockRow = blockIdx.y;
    const unsigned int blockColumn = blockIdx.x;
    const unsigned int rowInTile = threadIdx.x;
    const unsigned int columnInTile = threadIdx.y;
    const unsigned int row = blockRow * TILE_SIZE + rowInTile;
    const unsigned int column = blockColumn * TILE_SIZE + columnInTile;
    const bool valid = row < numNodes && column < numNodes;

    if (blockRow == pivotStart / TILE_SIZE ||
        blockColumn == pivotStart / TILE_SIZE) {
        return;
    }

    left[columnInTile][rowInTile] =
        (row < numNodes && pivotStart + columnInTile < numNodes)
            ? dist[deviceIdx(row, pivotStart + columnInTile, numNodes)]
            : INF;
    top[columnInTile][rowInTile] =
        (pivotStart + rowInTile < numNodes && column < numNodes)
            ? dist[deviceIdx(pivotStart + rowInTile, column, numNodes)]
            : INF;
    __syncthreads();

    unsigned int value = valid ? dist[deviceIdx(row, column, numNodes)] : INF;
    unsigned int lastPath = 0;
    bool pathChanged = false;

    // The pivot row/column tiles have already been closed over this k-block,
    // so each remainder element can consume them entirely from shared memory.
    #pragma unroll
    for (unsigned int k = 0; k < TILE_SIZE; ++k) {
        if (k < pivotNodes && valid) {
            const unsigned int via = left[k][rowInTile] +
                                     top[columnInTile][k];
            if (via < value) {
                value = via;
                lastPath = pivotStart + k;
                pathChanged = true;
            }
        }
    }

    if (valid) {
        const size_t position = deviceIdx(row, column, numNodes);
        dist[position] = value;
        if (pathChanged) {
            path[position] = lastPath;
        }
    }
}

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

void floydWarshall(std::vector<unsigned int>& dist, 
                   std::vector<unsigned int>& path, 
                   const size_t numNodes) {
    if (numNodes == 0) {
        return;
    }
    if (numNodes > static_cast<size_t>(UINT32_MAX)) {
        fprintf(stderr, "Number of nodes exceeds the CUDA index range\n");
        std::exit(EXIT_FAILURE);
    }

    const size_t matrixBytes = numNodes * numNodes * sizeof(unsigned int);
    unsigned int* deviceDist = nullptr;
    unsigned int* devicePath = nullptr;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceDist), matrixBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&devicePath), matrixBytes));
    CUDA_CHECK(cudaMemcpy(deviceDist, dist.data(), matrixBytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(devicePath, path.data(), matrixBytes,
                          cudaMemcpyHostToDevice));

    const unsigned int deviceNodeCount = static_cast<unsigned int>(numNodes);
    const unsigned int blockCount =
        (deviceNodeCount + TILE_SIZE - 1) / TILE_SIZE;
    const dim3 threads(TILE_SIZE, TILE_SIZE);

    // The three phases are launched in one CUDA stream. CUDA's stream order
    // supplies the global barriers between dependent phases, while each
    // kernel uses shared memory for the intra-tile k dependencies.
    for (unsigned int pivotBlock = 0; pivotBlock < blockCount; ++pivotBlock) {
        const unsigned int pivotStart = pivotBlock * TILE_SIZE;
        const unsigned int pivotNodes =
            std::min(TILE_SIZE, deviceNodeCount - pivotStart);

        floydWarshallPivot<<<1, threads>>>(deviceDist, devicePath,
                                            deviceNodeCount, pivotStart,
                                            pivotNodes);
        CUDA_CHECK(cudaGetLastError());

        floydWarshallPivotRow<<<blockCount, threads>>>(
            deviceDist, devicePath, deviceNodeCount, pivotStart,
            pivotNodes);
        CUDA_CHECK(cudaGetLastError());

        floydWarshallPivotColumn<<<blockCount, threads>>>(
            deviceDist, devicePath, deviceNodeCount, pivotStart,
            pivotNodes);
        CUDA_CHECK(cudaGetLastError());

        floydWarshallRemainder<<<dim3(blockCount, blockCount), threads>>>(
            deviceDist, devicePath, deviceNodeCount, pivotStart, pivotNodes);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(dist.data(), deviceDist, matrixBytes,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(path.data(), devicePath, matrixBytes,
                          cudaMemcpyDeviceToHost));
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
    // Do one-time CUDA context setup outside the measured algorithm interval.
    CUDA_CHECK(cudaFree(0));
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
