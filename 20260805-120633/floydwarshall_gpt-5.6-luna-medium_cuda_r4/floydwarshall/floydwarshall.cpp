#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
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

__device__ __forceinline__ size_t deviceIdx2(const size_t i, const size_t j,
                                             const size_t n) noexcept {
    return j * n + i;
}

// Phase 1: solve the diagonal tile for the current block of intermediate nodes.
__global__ void floydWarshallDiagonal(unsigned int* dist, unsigned int* path,
                                       const size_t n, const size_t kk) {
    __shared__ unsigned int tile[TILE_SIZE][TILE_SIZE];

    const unsigned int x = threadIdx.x;
    const unsigned int y = threadIdx.y;
    const size_t i = kk + y;
    const size_t j = kk + x;
    tile[y][x] = (i < n && j < n) ? dist[deviceIdx2(j, i, n)] : INF;
    __syncthreads();

    const size_t width = min(static_cast<size_t>(TILE_SIZE), n - kk);
    for (size_t k = 0; k < width; ++k) {
        __syncthreads();
        if (i < n && j < n) {
            const unsigned int candidate = tile[y][k] + tile[k][x];
            if (candidate < tile[y][x]) {
                tile[y][x] = candidate;
                path[deviceIdx2(j, i, n)] = static_cast<unsigned int>(kk + k);
            }
        }
    }
    __syncthreads();
    if (i < n && j < n) {
        dist[deviceIdx2(j, i, n)] = tile[y][x];
    }
}

// Phase 2: update one tile in the current block row or block column.
__global__ void floydWarshallRow(unsigned int* dist, unsigned int* path,
                                  const size_t n, const size_t kk) {
    __shared__ unsigned int pivot[TILE_SIZE][TILE_SIZE];
    __shared__ unsigned int tile[TILE_SIZE][TILE_SIZE];

    const unsigned int x = threadIdx.x;
    const unsigned int y = threadIdx.y;
    const size_t i = kk + y;
    const size_t jj = static_cast<size_t>(blockIdx.x) * TILE_SIZE;
    const size_t j = jj + x;
    if (jj == kk) {
        return;
    }
    pivot[y][x] = (i < n && kk + x < n) ? dist[deviceIdx2(kk + x, i, n)] : INF;
    tile[y][x] = (i < n && j < n) ? dist[deviceIdx2(j, i, n)] : INF;
    __syncthreads();

    const size_t width = min(static_cast<size_t>(TILE_SIZE), n - kk);
    for (size_t k = 0; k < width; ++k) {
        const unsigned int candidate = pivot[y][k] + tile[k][x];
        if (i < n && j < n && candidate < tile[y][x]) {
            tile[y][x] = candidate;
            path[deviceIdx2(j, i, n)] = static_cast<unsigned int>(kk + k);
        }
        __syncthreads();
    }
    if (i < n && j < n) {
        dist[deviceIdx2(j, i, n)] = tile[y][x];
    }
}

__global__ void floydWarshallColumn(unsigned int* dist, unsigned int* path,
                                     const size_t n, const size_t kk) {
    __shared__ unsigned int pivot[TILE_SIZE][TILE_SIZE];
    __shared__ unsigned int tile[TILE_SIZE][TILE_SIZE];

    const unsigned int x = threadIdx.x;
    const unsigned int y = threadIdx.y;
    const size_t ii = static_cast<size_t>(blockIdx.x) * TILE_SIZE;
    const size_t i = ii + y;
    const size_t j = kk + x;
    if (ii == kk) {
        return;
    }
    pivot[y][x] = (kk + y < n && j < n) ? dist[deviceIdx2(j, kk + y, n)] : INF;
    tile[y][x] = (i < n && j < n) ? dist[deviceIdx2(j, i, n)] : INF;
    __syncthreads();

    const size_t width = min(static_cast<size_t>(TILE_SIZE), n - kk);
    for (size_t k = 0; k < width; ++k) {
        const unsigned int candidate = tile[y][k] + pivot[k][x];
        if (i < n && j < n && candidate < tile[y][x]) {
            tile[y][x] = candidate;
            path[deviceIdx2(j, i, n)] = static_cast<unsigned int>(kk + k);
        }
        __syncthreads();
    }
    if (i < n && j < n) {
        dist[deviceIdx2(j, i, n)] = tile[y][x];
    }
}

__global__ void floydWarshallRemainder(unsigned int* dist, unsigned int* path,
                                        const size_t n, const size_t kk) {
    const size_t pivotBlock = kk / TILE_SIZE;
    if (blockIdx.x == pivotBlock || blockIdx.y == pivotBlock) {
        return;
    }
    __shared__ unsigned int row[TILE_SIZE][TILE_SIZE];
    __shared__ unsigned int column[TILE_SIZE][TILE_SIZE];

    const unsigned int x = threadIdx.x;
    const unsigned int y = threadIdx.y;
    const size_t ii = static_cast<size_t>(blockIdx.y) * TILE_SIZE;
    const size_t jj = static_cast<size_t>(blockIdx.x) * TILE_SIZE;
    const size_t i = ii + y;
    const size_t j = jj + x;
    row[y][x] = (kk + y < n && j < n) ? dist[deviceIdx2(j, kk + y, n)] : INF;
    column[y][x] = (i < n && kk + x < n) ? dist[deviceIdx2(kk + x, i, n)] : INF;
    __syncthreads();

    const size_t width = min(static_cast<size_t>(TILE_SIZE), n - kk);
    unsigned int value = (i < n && j < n) ? dist[deviceIdx2(j, i, n)] : INF;
    for (size_t k = 0; k < width; ++k) {
        const unsigned int candidate = column[y][k] + row[k][x];
        if (candidate < value) {
            value = candidate;
            if (i < n && j < n) {
                path[deviceIdx2(j, i, n)] = static_cast<unsigned int>(kk + k);
            }
        }
    }
    if (i < n && j < n) {
        dist[deviceIdx2(j, i, n)] = value;
    }
}

void checkCuda(const cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
    }
}

void floydWarshall(std::vector<unsigned int>& dist, 
                   std::vector<unsigned int>& path, 
                   const size_t numNodes) {
    if (numNodes == 0) {
        return;
    }

    unsigned int* deviceDist = nullptr;
    unsigned int* devicePath = nullptr;
    const size_t bytes = numNodes * numNodes * sizeof(unsigned int);
    checkCuda(cudaMalloc(&deviceDist, bytes), "cudaMalloc(dist)");
    checkCuda(cudaMalloc(&devicePath, bytes), "cudaMalloc(path)");
    checkCuda(cudaMemcpy(deviceDist, dist.data(), bytes, cudaMemcpyHostToDevice),
              "cudaMemcpy(dist)");
    checkCuda(cudaMemcpy(devicePath, path.data(), bytes, cudaMemcpyHostToDevice),
              "cudaMemcpy(path)");

    const dim3 threads(TILE_SIZE, TILE_SIZE);
    const size_t tileCount = (numNodes + TILE_SIZE - 1) / TILE_SIZE;
    for (size_t kk = 0; kk < numNodes; kk += TILE_SIZE) {
        floydWarshallDiagonal<<<1, threads>>>(deviceDist, devicePath, numNodes, kk);
        checkCuda(cudaGetLastError(), "diagonal kernel launch");

        floydWarshallRow<<<tileCount, threads>>>(deviceDist, devicePath, numNodes, kk);
        checkCuda(cudaGetLastError(), "row kernel launch");
        floydWarshallColumn<<<tileCount, threads>>>(deviceDist, devicePath, numNodes, kk);
        checkCuda(cudaGetLastError(), "column kernel launch");
        if (tileCount > 1) {
            const dim3 grid(tileCount, tileCount);
            floydWarshallRemainder<<<grid, threads>>>(deviceDist, devicePath,
                                                       numNodes, kk);
            checkCuda(cudaGetLastError(), "remainder kernel launch");
        }
    }

    checkCuda(cudaDeviceSynchronize(), "CUDA computation");
    checkCuda(cudaMemcpy(dist.data(), deviceDist, bytes, cudaMemcpyDeviceToHost),
              "cudaMemcpy(dist back)");
    checkCuda(cudaFree(devicePath), "cudaFree(path)");
    checkCuda(cudaFree(deviceDist), "cudaFree(dist)");
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
