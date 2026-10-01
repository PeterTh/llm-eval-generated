#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
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

// Every tile has a full 32x32 footprint. Padding is initialized to INF so
// partial tiles use the same kernels without divergent bounds checks.
__global__ void pivotTile(unsigned int* dist, unsigned int* path,
                          int pitch, int pivot, int nodes) {
    __shared__ unsigned int tile[TILE][TILE];
    const int row = threadIdx.y;
    const int col = threadIdx.x;
    const int base = pivot * TILE;
    const int pos = (base + row) * pitch + base + col;
    tile[row][col] = dist[pos];
    unsigned int via = path[pos];
    __syncthreads();
    for (int k = 0; k < TILE && base + k < nodes; ++k) {
        const unsigned int candidate = tile[row][k] + tile[k][col];
        if (candidate < tile[row][col]) {
            tile[row][col] = candidate;
            via = base + k;
        }
        __syncthreads();
    }
    dist[pos] = tile[row][col];
    path[pos] = via;
}

// Grid x indexes all non-pivot tiles in the pivot row and column.
__global__ void adjacentTiles(unsigned int* dist, unsigned int* path,
                              int pitch, int pivot, int nodes) {
    __shared__ unsigned int center[TILE][TILE];
    __shared__ unsigned int tile[TILE][TILE];
    const int row = threadIdx.y;
    const int col = threadIdx.x;
    const int other = blockIdx.x < pivot ? blockIdx.x : blockIdx.x + 1;
    const bool rowTile = blockIdx.y == 0;
    const int tileRow = rowTile ? pivot : other;
    const int tileCol = rowTile ? other : pivot;
    const int pos = (tileRow * TILE + row) * pitch + tileCol * TILE + col;
    center[row][col] = dist[(pivot * TILE + row) * pitch + pivot * TILE + col];
    tile[row][col] = dist[pos];
    unsigned int via = path[pos];
    __syncthreads();
    for (int k = 0; k < TILE && pivot * TILE + k < nodes; ++k) {
        const unsigned int candidate = rowTile
            ? center[row][k] + tile[k][col]
            : tile[row][k] + center[k][col];
        if (candidate < tile[row][col]) {
            tile[row][col] = candidate;
            via = pivot * TILE + k;
        }
        __syncthreads();
    }
    dist[pos] = tile[row][col];
    path[pos] = via;
}

__global__ void remainingTiles(unsigned int* dist, unsigned int* path,
                               int pitch, int pivot, int nodes) {
    __shared__ unsigned int left[TILE][TILE];
    __shared__ unsigned int top[TILE][TILE];
    const int row = threadIdx.y;
    const int col = threadIdx.x;
    const int tileRow = blockIdx.y < pivot ? blockIdx.y : blockIdx.y + 1;
    const int tileCol = blockIdx.x < pivot ? blockIdx.x : blockIdx.x + 1;
    const int baseRow = tileRow * TILE + row;
    const int baseCol = tileCol * TILE + col;
    const int pos = baseRow * pitch + baseCol;
    left[row][col] = dist[baseRow * pitch + pivot * TILE + col];
    top[row][col] = dist[(pivot * TILE + row) * pitch + baseCol];
    unsigned int best = dist[pos];
    unsigned int via = path[pos];
    __syncthreads();
    for (int k = 0; k < TILE && pivot * TILE + k < nodes; ++k) {
        const unsigned int candidate = left[row][k] + top[k][col];
        if (candidate < best) {
            best = candidate;
            via = pivot * TILE + k;
        }
    }
    dist[pos] = best;
    path[pos] = via;
}

void checkCuda(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes) {
    if (numNodes == 0) return;
    if (numNodes > static_cast<size_t>(std::numeric_limits<int>::max() - TILE)) {
        throw std::runtime_error("graph is too large for CUDA indexing");
    }
    const int tiles = static_cast<int>((numNodes + TILE - 1) / TILE);
    const int pitch = tiles * TILE;
    const size_t bytes = static_cast<size_t>(pitch) * pitch * sizeof(unsigned int);
    unsigned int* deviceDist = nullptr;
    unsigned int* devicePath = nullptr;
    try {
        checkCuda(cudaMalloc(&deviceDist, bytes));
        checkCuda(cudaMalloc(&devicePath, bytes));
        // Every padded distance is INF; its largest possible sum fits in uint32.
        std::vector<unsigned int> padded(static_cast<size_t>(pitch) * pitch, INF);
        checkCuda(cudaMemcpy(deviceDist, padded.data(), bytes, cudaMemcpyHostToDevice));
        checkCuda(cudaMemset(devicePath, 0, bytes));
        checkCuda(cudaMemcpy2D(deviceDist, pitch * sizeof(unsigned int),
                               dist.data(), numNodes * sizeof(unsigned int),
                               numNodes * sizeof(unsigned int), numNodes,
                               cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy2D(devicePath, pitch * sizeof(unsigned int),
                               path.data(), numNodes * sizeof(unsigned int),
                               numNodes * sizeof(unsigned int), numNodes,
                               cudaMemcpyHostToDevice));

        const dim3 threads(TILE, TILE);
        for (int pivot = 0; pivot < tiles; ++pivot) {
            pivotTile<<<1, threads>>>(deviceDist, devicePath, pitch, pivot, static_cast<int>(numNodes));
            if (tiles > 1) {
                adjacentTiles<<<dim3(tiles - 1, 2), threads>>>(
                    deviceDist, devicePath, pitch, pivot, static_cast<int>(numNodes));
                remainingTiles<<<dim3(tiles - 1, tiles - 1), threads>>>(
                    deviceDist, devicePath, pitch, pivot, static_cast<int>(numNodes));
            }
        }
        checkCuda(cudaGetLastError());
        checkCuda(cudaDeviceSynchronize());
        checkCuda(cudaMemcpy2D(dist.data(), numNodes * sizeof(unsigned int),
                               deviceDist, pitch * sizeof(unsigned int),
                               numNodes * sizeof(unsigned int), numNodes,
                               cudaMemcpyDeviceToHost));
        checkCuda(cudaMemcpy2D(path.data(), numNodes * sizeof(unsigned int),
                               devicePath, pitch * sizeof(unsigned int),
                               numNodes * sizeof(unsigned int), numNodes,
                               cudaMemcpyDeviceToHost));
    } catch (...) {
        cudaFree(deviceDist);
        cudaFree(devicePath);
        throw;
    }
    checkCuda(cudaFree(deviceDist));
    checkCuda(cudaFree(devicePath));
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
    checkCuda(cudaFree(nullptr)); // Initialize the CUDA context outside the timed computation.
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
