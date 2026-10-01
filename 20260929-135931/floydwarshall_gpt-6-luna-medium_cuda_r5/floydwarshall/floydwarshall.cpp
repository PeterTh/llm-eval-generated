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

__global__ void diagonalPhase(unsigned int* d, unsigned int* p, int n, int tile) {
    __shared__ unsigned int a[TILE][TILE];
    const int x = threadIdx.x, y = threadIdx.y;
    const int i = tile * TILE + x, j = tile * TILE + y;
    if (i < n && j < n) a[y][x] = d[(size_t)j * n + i];
    else a[y][x] = INF;
    __syncthreads();
    for (int t = 0; t < TILE && tile * TILE + t < n; ++t) {
        const unsigned int cand = a[t][x] + a[y][t];
        if (cand < a[y][x]) {
            a[y][x] = cand;
            if (i < n && j < n) p[(size_t)j * n + i] = tile * TILE + t;
        }
        __syncthreads();
    }
    if (i < n && j < n) d[(size_t)j * n + i] = a[y][x];
}

__global__ void borderPhase(unsigned int* d, unsigned int* p, int n, int pivot, int tiles) {
    const int other = blockIdx.x;
    const bool rowTile = blockIdx.y == 0;
    if (other == pivot) return;
    const int x = threadIdx.x, y = threadIdx.y;
    const int bi = rowTile ? pivot : other;
    const int bj = rowTile ? other : pivot;
    const int i = bi * TILE + x, j = bj * TILE + y;
    if (i >= n || j >= n) return;
    unsigned int best = d[(size_t)j * n + i];
    int bestK = -1;
    for (int t = 0; t < TILE && pivot * TILE + t < n; ++t) {
        const int k = pivot * TILE + t;
        const unsigned int cand = rowTile
            ? d[(size_t)k * n + i] + d[(size_t)j * n + k]
            : d[(size_t)k * n + i] + d[(size_t)j * n + k];
        if (cand < best) { best = cand; bestK = k; }
    }
    d[(size_t)j * n + i] = best;
    if (bestK >= 0) p[(size_t)j * n + i] = bestK;
}

__global__ void outerPhase(unsigned int* d, unsigned int* p, int n, int pivot, int tiles) {
    const int bi = blockIdx.x, bj = blockIdx.y;
    if (bi == pivot || bj == pivot) return;
    const int x = threadIdx.x, y = threadIdx.y;
    const int i = bi * TILE + x, j = bj * TILE + y;
    if (i >= n || j >= n) return;
    unsigned int best = d[(size_t)j * n + i];
    int bestK = -1;
    for (int t = 0; t < TILE && pivot * TILE + t < n; ++t) {
        const int k = pivot * TILE + t;
        const unsigned int cand = d[(size_t)k * n + i] + d[(size_t)j * n + k];
        if (cand < best) { best = cand; bestK = k; }
    }
    d[(size_t)j * n + i] = best;
    if (bestK >= 0) p[(size_t)j * n + i] = bestK;
}

void floydWarshall(std::vector<unsigned int>& dist, std::vector<unsigned int>& path, size_t numNodes) {
    const int n = static_cast<int>(numNodes);
    const int tiles = (n + TILE - 1) / TILE;
    unsigned int *d = nullptr, *p = nullptr;
    const size_t bytes = numNodes * numNodes * sizeof(unsigned int);
    if (cudaMalloc(&d, bytes) != cudaSuccess || cudaMalloc(&p, bytes) != cudaSuccess)
        std::abort();
    cudaMemcpy(d, dist.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(p, path.data(), bytes, cudaMemcpyHostToDevice);
    const dim3 block(TILE, TILE);
    for (int r = 0; r < tiles; ++r) {
        diagonalPhase<<<1, block>>>(d, p, n, r);
        borderPhase<<<dim3(tiles, 2), block>>>(d, p, n, r, tiles);
        outerPhase<<<dim3(tiles, tiles), block>>>(d, p, n, r, tiles);
    }
    cudaDeviceSynchronize();
    cudaMemcpy(dist.data(), d, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(path.data(), p, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d); cudaFree(p);
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
