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

__global__ void diagonalTile(unsigned int* d, unsigned int* p, int n, int base, int width) {
    __shared__ unsigned int tile[TILE][TILE];
    __shared__ unsigned int paths[TILE][TILE];
    const int x = threadIdx.x, y = threadIdx.y;
    const int row = base + y, col = base + x;
    tile[y][x] = (row < n && col < n) ? d[(size_t)col * n + row] : INF;
    paths[y][x] = (row < n && col < n) ? p[(size_t)col * n + row] : 0;
    __syncthreads();
    for (int t = 0; t < width; ++t) {
        const unsigned int candidate = tile[y][t] + tile[t][x];
        if (candidate < tile[y][x]) { tile[y][x] = candidate; paths[y][x] = base + t; }
        __syncthreads();
    }
    if (row < n && col < n) { d[(size_t)col * n + row] = tile[y][x]; p[(size_t)col * n + row] = paths[y][x]; }
}

__global__ void updateTile(unsigned int* d, unsigned int* p, int n, int pivot, int tileIndex,
                           int width, bool rowTile) {
    const int x = threadIdx.x, y = threadIdx.y;
    const int tile = tileIndex + static_cast<int>(blockIdx.x) * TILE;
    if (tile == pivot) return;
    const int row = (rowTile ? pivot : tile) + y;
    const int col = (rowTile ? tile : pivot) + x;
    if (row >= n || col >= n) return;
    unsigned int best = d[(size_t)col * n + row], bestPath = p[(size_t)col * n + row];
    for (int t = 0; t < width; ++t) {
        const int k = pivot + t;
        const unsigned int candidate = d[(size_t)k * n + row] + d[(size_t)col * n + k];
        if (candidate < best) { best = candidate; bestPath = k; }
    }
    d[(size_t)col * n + row] = best;
    p[(size_t)col * n + row] = bestPath;
}

__global__ void updateRemainder(unsigned int* d, unsigned int* p, int n, int pivot,
                                int rowBase, int colBase, int width) {
    const int rowTile = rowBase + static_cast<int>(blockIdx.y) * TILE;
    const int colTile = colBase + static_cast<int>(blockIdx.x) * TILE;
    if (rowTile == pivot || colTile == pivot) return;
    const int row = rowTile + threadIdx.y;
    const int col = colTile + threadIdx.x;
    if (row >= n || col >= n) return;
    unsigned int best = d[(size_t)col * n + row], bestPath = p[(size_t)col * n + row];
    for (int t = 0; t < width; ++t) {
        const int k = pivot + t;
        const unsigned int candidate = d[(size_t)k * n + row] + d[(size_t)col * n + k];
        if (candidate < best) { best = candidate; bestPath = k; }
    }
    d[(size_t)col * n + row] = best;
    p[(size_t)col * n + row] = bestPath;
}

static void cudaCheck(cudaError_t error) {
    if (error != cudaSuccess) { fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error)); std::exit(1); }
}

void floydWarshall(std::vector<unsigned int>& dist, std::vector<unsigned int>& path, size_t numNodes) {
    unsigned int *deviceDist = nullptr, *devicePath = nullptr;
    const size_t bytes = numNodes * numNodes * sizeof(unsigned int);
    cudaCheck(cudaMalloc(&deviceDist, bytes));
    cudaCheck(cudaMalloc(&devicePath, bytes));
    cudaCheck(cudaMemcpy(deviceDist, dist.data(), bytes, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(devicePath, path.data(), bytes, cudaMemcpyHostToDevice));
    const dim3 threads(TILE, TILE);
    const int n = static_cast<int>(numNodes);
    for (int pivot = 0; pivot < n; pivot += TILE) {
        const int width = std::min(TILE, n - pivot);
        diagonalTile<<<1, threads>>>(deviceDist, devicePath, n, pivot, width);
        const int tileCount = (n + TILE - 1) / TILE;
        updateTile<<<tileCount, threads>>>(deviceDist, devicePath, n, pivot, 0, width, true);
        updateTile<<<tileCount, threads>>>(deviceDist, devicePath, n, pivot, 0, width, false);
        updateRemainder<<<dim3(tileCount, tileCount), threads>>>(deviceDist, devicePath, n,
                                                                 pivot, 0, 0, width);
    }
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaMemcpy(dist.data(), deviceDist, bytes, cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(path.data(), devicePath, bytes, cudaMemcpyDeviceToHost));
    cudaFree(deviceDist); cudaFree(devicePath);
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
