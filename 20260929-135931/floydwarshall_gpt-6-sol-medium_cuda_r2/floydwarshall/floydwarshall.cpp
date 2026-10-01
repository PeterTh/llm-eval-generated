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

constexpr int TILE = 32;
constexpr int ROWS_PER_THREAD = 4;

void checkCuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(result));
        std::exit(EXIT_FAILURE);
    }
}

// Each thread owns four entries in one column of a 32 by 32 tile.
__global__ void pivotTile(unsigned int* dist, unsigned int* path, size_t n, int pivot) {
    __shared__ unsigned int tile[TILE][TILE + 1];
    const int x = threadIdx.x;
    const int baseRow = pivot * TILE + threadIdx.y;
    const int col = pivot * TILE + x;
    unsigned int paths[ROWS_PER_THREAD];

#pragma unroll
    for (int t = 0; t < ROWS_PER_THREAD; ++t) {
        const int y = threadIdx.y + t * blockDim.y;
        const int row = baseRow + t * blockDim.y;
        tile[y][x] = row < n && col < n ? dist[(size_t)row * n + col] : INF;
        paths[t] = row < n && col < n ? path[(size_t)row * n + col] : 0;
    }
    __syncthreads();

    for (int k = 0; k < TILE; ++k) {
#pragma unroll
        for (int t = 0; t < ROWS_PER_THREAD; ++t) {
            const int y = threadIdx.y + t * blockDim.y;
            // The pivot row and column cannot improve: their diagonal is zero.
            // Leaving them untouched also keeps the operands stable this round.
            if (y != k && x != k) {
                const unsigned int candidate = tile[y][k] + tile[k][x];
                if (candidate < tile[y][x]) {
                    tile[y][x] = candidate;
                    paths[t] = pivot * TILE + k;
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int t = 0; t < ROWS_PER_THREAD; ++t) {
        const int y = threadIdx.y + t * blockDim.y;
        const int row = baseRow + t * blockDim.y;
        if (row < n && col < n) {
            dist[(size_t)row * n + col] = tile[y][x];
            path[(size_t)row * n + col] = paths[t];
        }
    }
}

// Update the tiles in the pivot row and column. The pivot tile is read only.
__global__ void pivotEdges(unsigned int* dist, unsigned int* path, size_t n,
                           int pivot) {
    __shared__ unsigned int tile[TILE][TILE + 1];
    __shared__ unsigned int diagonal[TILE][TILE + 1];
    const int other = blockIdx.x < pivot ? blockIdx.x : blockIdx.x + 1;
    const bool rowTile = blockIdx.y == 0;
    const int tileRow = rowTile ? pivot : other;
    const int tileCol = rowTile ? other : pivot;
    const int x = threadIdx.x;
    const int col = tileCol * TILE + x;
    unsigned int paths[ROWS_PER_THREAD];

#pragma unroll
    for (int t = 0; t < ROWS_PER_THREAD; ++t) {
        const int y = threadIdx.y + t * blockDim.y;
        const int row = tileRow * TILE + y;
        const int pivotRow = pivot * TILE + y;
        const int pivotCol = pivot * TILE + x;
        tile[y][x] = row < n && col < n ? dist[(size_t)row * n + col] : INF;
        diagonal[y][x] = pivotRow < n && pivotCol < n
                             ? dist[(size_t)pivotRow * n + pivotCol] : INF;
        paths[t] = row < n && col < n ? path[(size_t)row * n + col] : 0;
    }
    __syncthreads();

    for (int k = 0; k < TILE; ++k) {
#pragma unroll
        for (int t = 0; t < ROWS_PER_THREAD; ++t) {
            const int y = threadIdx.y + t * blockDim.y;
            if ((rowTile && y == k) || (!rowTile && x == k)) continue;
            const unsigned int candidate = rowTile
                ? diagonal[y][k] + tile[k][x]
                : tile[y][k] + diagonal[k][x];
            if (candidate < tile[y][x]) {
                tile[y][x] = candidate;
                paths[t] = pivot * TILE + k;
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int t = 0; t < ROWS_PER_THREAD; ++t) {
        const int y = threadIdx.y + t * blockDim.y;
        const int row = tileRow * TILE + y;
        if (row < n && col < n) {
            dist[(size_t)row * n + col] = tile[y][x];
            path[(size_t)row * n + col] = paths[t];
        }
    }
}

// All remaining tiles are independent after the pivot row and column finish.
__global__ void remainingTiles(unsigned int* dist, unsigned int* path, size_t n,
                               int pivot) {
    __shared__ unsigned int rowTile[TILE][TILE + 1];
    __shared__ unsigned int columnTile[TILE][TILE + 1];
    const int tileRow = blockIdx.y < pivot ? blockIdx.y : blockIdx.y + 1;
    const int tileCol = blockIdx.x < pivot ? blockIdx.x : blockIdx.x + 1;
    const int x = threadIdx.x;
    const int col = tileCol * TILE + x;
    unsigned int values[ROWS_PER_THREAD];
    unsigned int paths[ROWS_PER_THREAD];

#pragma unroll
    for (int t = 0; t < ROWS_PER_THREAD; ++t) {
        const int y = threadIdx.y + t * blockDim.y;
        const int row = tileRow * TILE + y;
        const int pivotRow = pivot * TILE + y;
        const int pivotCol = pivot * TILE + x;
        rowTile[y][x] = pivotRow < n && col < n
                            ? dist[(size_t)pivotRow * n + col] : INF;
        columnTile[y][x] = row < n && pivotCol < n
                               ? dist[(size_t)row * n + pivotCol] : INF;
        values[t] = row < n && col < n ? dist[(size_t)row * n + col] : INF;
        paths[t] = row < n && col < n ? path[(size_t)row * n + col] : 0;
    }
    __syncthreads();

    for (int k = 0; k < TILE; ++k) {
#pragma unroll
        for (int t = 0; t < ROWS_PER_THREAD; ++t) {
            const int y = threadIdx.y + t * blockDim.y;
            const unsigned int candidate = columnTile[y][k] + rowTile[k][x];
            if (candidate < values[t]) {
                values[t] = candidate;
                paths[t] = pivot * TILE + k;
            }
        }
    }

#pragma unroll
    for (int t = 0; t < ROWS_PER_THREAD; ++t) {
        const int row = tileRow * TILE + threadIdx.y + t * blockDim.y;
        if (row < n && col < n) {
            dist[(size_t)row * n + col] = values[t];
            path[(size_t)row * n + col] = paths[t];
        }
    }
}

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes) {
    const size_t bytes = dist.size() * sizeof(unsigned int);
    unsigned int* deviceDist = nullptr;
    unsigned int* devicePath = nullptr;
    checkCuda(cudaMalloc(&deviceDist, bytes), "cudaMalloc distance");
    checkCuda(cudaMalloc(&devicePath, bytes), "cudaMalloc path");
    checkCuda(cudaMemcpy(deviceDist, dist.data(), bytes, cudaMemcpyHostToDevice), "copy distance to GPU");
    checkCuda(cudaMemcpy(devicePath, path.data(), bytes, cudaMemcpyHostToDevice), "copy path to GPU");

    const int tiles = static_cast<int>((numNodes + TILE - 1) / TILE);
    const dim3 threads(TILE, TILE / ROWS_PER_THREAD);
    for (int pivot = 0; pivot < tiles; ++pivot) {
        pivotTile<<<1, threads>>>(deviceDist, devicePath, numNodes, pivot);
        if (tiles > 1) {
            pivotEdges<<<dim3(tiles - 1, 2), threads>>>(deviceDist, devicePath,
                                                        numNodes, pivot);
            remainingTiles<<<dim3(tiles - 1, tiles - 1), threads>>>(
                deviceDist, devicePath, numNodes, pivot);
        }
    }
    checkCuda(cudaGetLastError(), "Floyd-Warshall kernel launch");
    checkCuda(cudaMemcpy(dist.data(), deviceDist, bytes, cudaMemcpyDeviceToHost), "copy distance from GPU");
    checkCuda(cudaMemcpy(path.data(), devicePath, bytes, cudaMemcpyDeviceToHost), "copy path from GPU");
    checkCuda(cudaFree(deviceDist), "cudaFree distance");
    checkCuda(cudaFree(devicePath), "cudaFree path");
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
    checkCuda(cudaFree(nullptr), "initialize CUDA");
    
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
