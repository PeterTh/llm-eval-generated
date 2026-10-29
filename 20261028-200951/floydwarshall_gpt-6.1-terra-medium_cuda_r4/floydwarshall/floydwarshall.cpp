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

// A tile contains 32x32 distances. Each thread owns four cells.
constexpr int TILE = 32;

static void cudaCheck(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}

// Phase 0 closes the pivot tile; phase 1 updates the pivot row and column;
// phase 2 updates all remaining tiles. Kernel launches order the phases.
__global__ void updateTiles(unsigned int* dist, unsigned int* path, size_t n,
                            int pivot, int phase, int tiles) {
    int rowTile, colTile;
    if (phase == 0) {
        rowTile = colTile = pivot;
    } else if (phase == 1) {
        int t = blockIdx.x;
        if (t >= pivot) ++t;
        rowTile = blockIdx.y == 0 ? pivot : t;
        colTile = blockIdx.y == 0 ? t : pivot;
    } else {
        rowTile = blockIdx.y;
        colTile = blockIdx.x;
        if (rowTile >= pivot) ++rowTile;
        if (colTile >= pivot) ++colTile;
    }
    (void)tiles;
    __shared__ unsigned int left[TILE][TILE];
    __shared__ unsigned int right[TILE][TILE];
    __shared__ unsigned int tile[TILE][TILE];
    const int x = threadIdx.x;
    const int y0 = threadIdx.y;
    unsigned int values[4], paths[4];
    #pragma unroll
    for (int q = 0; q < 4; ++q) {
        int y = y0 + q * 8;
        size_t row = (size_t)rowTile * TILE + y;
        size_t col = (size_t)colTile * TILE + x;
        size_t pivotRow = (size_t)pivot * TILE + y;
        size_t pivotCol = (size_t)pivot * TILE + x;
        values[q] = row < n && col < n ? dist[row * n + col] : INF;
        paths[q] = row < n && col < n ? path[row * n + col] : 0;
        tile[y][x] = values[q];
        left[y][x] = row < n && pivotCol < n ? dist[row * n + pivotCol] : INF;
        right[y][x] = pivotRow < n && col < n ? dist[pivotRow * n + col] : INF;
    }
    __syncthreads();
    for (int k = 0; k < TILE && (size_t)pivot * TILE + k < n; ++k) {
        #pragma unroll
        for (int q = 0; q < 4; ++q) {
            int y = y0 + q * 8;
            unsigned int a = phase == 0 || (phase == 1 && colTile == pivot) ? tile[y][k] : left[y][k];
            unsigned int b = phase == 0 || (phase == 1 && rowTile == pivot) ? tile[k][x] : right[k][x];
            unsigned int candidate = a + b;
            if (candidate < values[q]) {
                values[q] = candidate;
                paths[q] = pivot * TILE + k;
            }
        }
        if (phase != 2) {
            // All reads of the previous iteration must finish before writes.
            __syncthreads();
            #pragma unroll
            for (int q = 0; q < 4; ++q) tile[y0 + q * 8][x] = values[q];
            __syncthreads();
        }
    }
    #pragma unroll
    for (int q = 0; q < 4; ++q) {
        size_t row = (size_t)rowTile * TILE + y0 + q * 8;
        size_t col = (size_t)colTile * TILE + x;
        if (row < n && col < n) {
            dist[row * n + col] = values[q];
            path[row * n + col] = paths[q];
        }
    }
}

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path, const size_t numNodes) {
    if (numNodes == 0) return;
    const size_t bytes = dist.size() * sizeof(unsigned int);
    unsigned int *deviceDist, *devicePath;
    cudaCheck(cudaMalloc(&deviceDist, bytes), "distance allocation");
    cudaCheck(cudaMalloc(&devicePath, bytes), "path allocation");
    cudaCheck(cudaMemcpy(deviceDist, dist.data(), bytes, cudaMemcpyHostToDevice), "distance upload");
    cudaCheck(cudaMemcpy(devicePath, path.data(), bytes, cudaMemcpyHostToDevice), "path upload");
    const int tiles = (numNodes + TILE - 1) / TILE;
    const dim3 threads(TILE, 8);
    for (int pivot = 0; pivot < tiles; ++pivot) {
        updateTiles<<<1, threads>>>(deviceDist, devicePath, numNodes, pivot, 0, tiles);
        if (tiles > 1) {
            updateTiles<<<dim3(tiles - 1, 2), threads>>>(deviceDist, devicePath, numNodes, pivot, 1, tiles);
            updateTiles<<<dim3(tiles - 1, tiles - 1), threads>>>(deviceDist, devicePath, numNodes, pivot, 2, tiles);
        }
    }
    cudaCheck(cudaGetLastError(), "kernel launch");
    cudaCheck(cudaMemcpy(dist.data(), deviceDist, bytes, cudaMemcpyDeviceToHost), "distance download");
    cudaCheck(cudaMemcpy(path.data(), devicePath, bytes, cudaMemcpyDeviceToHost), "path download");
    cudaCheck(cudaFree(deviceDist), "distance free");
    cudaCheck(cudaFree(devicePath), "path free");
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
