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

// Each block owns a 32 x 32 tile. The three phases process the pivot,
// its row and column, and then all remaining tiles in stream order.
constexpr int TILE = 32;

#define CUDA_CHECK(call) do { \
    cudaError_t error = (call); \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

__global__ void pivotKernel(unsigned int* dist, unsigned int* path, size_t n, int pivot) {
    __shared__ unsigned int tile[TILE][TILE];
    int x = threadIdx.x, y = threadIdx.y;
    size_t row = size_t(pivot) * TILE + y, col = size_t(pivot) * TILE + x;
    tile[y][x] = row < n && col < n ? dist[row * n + col] : INF;
    __syncthreads();
    unsigned int p = 0; bool changed = false;
    for (int k = 0; k < TILE && size_t(pivot) * TILE + k < n; ++k) {
        unsigned int candidate = tile[y][k] + tile[k][x];
        __syncthreads();
        if (candidate < tile[y][x]) { tile[y][x] = candidate; p = pivot * TILE + k; changed = true; }
        __syncthreads();
    }
    if (row < n && col < n) {
        dist[row * n + col] = tile[y][x];
        if (changed) path[row * n + col] = p;
    }
}

__global__ void edgeKernel(unsigned int* dist, unsigned int* path, size_t n, int pivot) {
    __shared__ unsigned int base[TILE][TILE], tile[TILE][TILE];
    int x = threadIdx.x, y = threadIdx.y;
    int other = blockIdx.x;
    if (other >= pivot) ++other;
    bool pivotRow = blockIdx.y == 0;
    size_t row = size_t(pivotRow ? pivot : other) * TILE + y;
    size_t col = size_t(pivotRow ? other : pivot) * TILE + x;
    size_t pr = size_t(pivot) * TILE + y, pc = size_t(pivot) * TILE + x;
    base[y][x] = pr < n && pc < n ? dist[pr * n + pc] : INF;
    tile[y][x] = row < n && col < n ? dist[row * n + col] : INF;
    __syncthreads();
    unsigned int p = 0; bool changed = false;
    for (int k = 0; k < TILE && size_t(pivot) * TILE + k < n; ++k) {
        unsigned int candidate = pivotRow ? base[y][k] + tile[k][x] : tile[y][k] + base[k][x];
        __syncthreads();
        if (candidate < tile[y][x]) { tile[y][x] = candidate; p = pivot * TILE + k; changed = true; }
        __syncthreads();
    }
    if (row < n && col < n) {
        dist[row * n + col] = tile[y][x];
        if (changed) path[row * n + col] = p;
    }
}

__global__ void remainingKernel(unsigned int* dist, unsigned int* path, size_t n, int pivot) {
    __shared__ unsigned int left[TILE][TILE], top[TILE][TILE];
    int x = threadIdx.x, y = threadIdx.y;
    int tileCol = blockIdx.x, tileRow = blockIdx.y;
    if (tileCol >= pivot) ++tileCol;
    if (tileRow >= pivot) ++tileRow;
    size_t row = size_t(tileRow) * TILE + y, col = size_t(tileCol) * TILE + x;
    size_t pr = size_t(pivot) * TILE + y, pc = size_t(pivot) * TILE + x;
    left[y][x] = row < n && pc < n ? dist[row * n + pc] : INF;
    top[y][x] = pr < n && col < n ? dist[pr * n + col] : INF;
    unsigned int value = row < n && col < n ? dist[row * n + col] : INF;
    __syncthreads();
    unsigned int p = 0; bool changed = false;
    for (int k = 0; k < TILE && size_t(pivot) * TILE + k < n; ++k) {
        unsigned int candidate = left[y][k] + top[k][x];
        if (candidate < value) { value = candidate; p = pivot * TILE + k; changed = true; }
    }
    if (row < n && col < n) {
        dist[row * n + col] = value;
        if (changed) path[row * n + col] = p;
    }
}

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path, const size_t numNodes) {
    if (numNodes == 0) return;
    const size_t bytes = numNodes * numNodes * sizeof(unsigned int);
    unsigned int *deviceDist, *devicePath;
    CUDA_CHECK(cudaMalloc(&deviceDist, bytes));
    CUDA_CHECK(cudaMalloc(&devicePath, bytes));
    CUDA_CHECK(cudaMemcpy(deviceDist, dist.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(devicePath, path.data(), bytes, cudaMemcpyHostToDevice));
    int tiles = static_cast<int>((numNodes + TILE - 1) / TILE);
    dim3 threads(TILE, TILE);
    for (int pivot = 0; pivot < tiles; ++pivot) {
        pivotKernel<<<1, threads>>>(deviceDist, devicePath, numNodes, pivot);
        if (tiles > 1) {
            edgeKernel<<<dim3(tiles - 1, 2), threads>>>(deviceDist, devicePath, numNodes, pivot);
            remainingKernel<<<dim3(tiles - 1, tiles - 1), threads>>>(deviceDist, devicePath, numNodes, pivot);
        }
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(dist.data(), deviceDist, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(path.data(), devicePath, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(deviceDist));
    CUDA_CHECK(cudaFree(devicePath));
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
