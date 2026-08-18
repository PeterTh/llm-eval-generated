#include <algorithm>
#include <chrono>
#include <cmath>
#include <climits>
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

static void cudaCheck(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(status));
        exit(EXIT_FAILURE);
    }
}

__global__ void pivotKernel(unsigned int* __restrict__ dist,
                            unsigned int* __restrict__ path, int n, int pivot) {
    __shared__ unsigned int tile[TILE][TILE + 1];
    const int row = pivot * TILE + threadIdx.y;
    const int col = pivot * TILE + threadIdx.x;
    tile[threadIdx.y][threadIdx.x] = (row < n && col < n) ? dist[row * n + col] : INF;
    __syncthreads();

    #pragma unroll
    for (int k = 0; k < TILE; ++k) {
        const unsigned int candidate = tile[threadIdx.y][k] + tile[k][threadIdx.x];
        __syncthreads();
        if (candidate < tile[threadIdx.y][threadIdx.x]) {
            tile[threadIdx.y][threadIdx.x] = candidate;
            if (row < n && col < n) path[row * n + col] = pivot * TILE + k;
        }
        __syncthreads();
    }
    if (row < n && col < n) dist[row * n + col] = tile[threadIdx.y][threadIdx.x];
}

__global__ void edgeKernel(unsigned int* __restrict__ dist,
                           unsigned int* __restrict__ path, int n, int pivot, int tiles) {
    __shared__ unsigned int fixed[TILE][TILE + 1];
    __shared__ unsigned int active[TILE][TILE + 1];
    const bool column = blockIdx.y != 0;
    int tile = blockIdx.x;
    if (tile >= pivot) ++tile;
    if (tile >= tiles) return;
    const int tileRow = column ? tile : pivot;
    const int tileCol = column ? pivot : tile;
    const int row = tileRow * TILE + threadIdx.y;
    const int col = tileCol * TILE + threadIdx.x;
    const int fixedRow = pivot * TILE + threadIdx.y;
    const int fixedCol = pivot * TILE + threadIdx.x;
    fixed[threadIdx.y][threadIdx.x] =
        (fixedRow < n && fixedCol < n) ? dist[fixedRow * n + fixedCol] : INF;
    active[threadIdx.y][threadIdx.x] = (row < n && col < n) ? dist[row * n + col] : INF;
    __syncthreads();

    #pragma unroll
    for (int k = 0; k < TILE; ++k) {
        const unsigned int candidate = column
            ? active[threadIdx.y][k] + fixed[k][threadIdx.x]
            : fixed[threadIdx.y][k] + active[k][threadIdx.x];
        __syncthreads();
        if (candidate < active[threadIdx.y][threadIdx.x]) {
            active[threadIdx.y][threadIdx.x] = candidate;
            if (row < n && col < n) path[row * n + col] = pivot * TILE + k;
        }
        __syncthreads();
    }
    if (row < n && col < n) dist[row * n + col] = active[threadIdx.y][threadIdx.x];
}

__global__ void remainderKernel(unsigned int* __restrict__ dist,
                                unsigned int* __restrict__ path, int n, int pivot) {
    __shared__ unsigned int left[TILE][TILE + 1];
    __shared__ unsigned int top[TILE][TILE + 1];
    int tileRow = blockIdx.y;
    int tileCol = blockIdx.x;
    if (tileRow >= pivot) ++tileRow;
    if (tileCol >= pivot) ++tileCol;
    const int row = tileRow * TILE + threadIdx.y;
    const int col = tileCol * TILE + threadIdx.x;
    const int pivotRow = pivot * TILE + threadIdx.y;
    const int pivotCol = pivot * TILE + threadIdx.x;
    left[threadIdx.y][threadIdx.x] =
        (row < n && pivotCol < n) ? dist[row * n + pivotCol] : INF;
    top[threadIdx.y][threadIdx.x] =
        (pivotRow < n && col < n) ? dist[pivotRow * n + col] : INF;
    __syncthreads();

    if (row < n && col < n) {
        unsigned int value = dist[row * n + col];
        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            const unsigned int candidate = left[threadIdx.y][k] + top[k][threadIdx.x];
            if (candidate < value) {
                value = candidate;
                path[row * n + col] = pivot * TILE + k;
            }
        }
        dist[row * n + col] = value;
    }
}

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes) {
    if (numNodes == 0) return;
    if (numNodes > static_cast<size_t>(INT_MAX)) {
        fprintf(stderr, "Number of nodes exceeds CUDA kernel index range\n");
        exit(EXIT_FAILURE);
    }
    const size_t bytes = numNodes * numNodes * sizeof(unsigned int);
    unsigned int *deviceDist = nullptr, *devicePath = nullptr;
    cudaCheck(cudaMalloc(&deviceDist, bytes), "distance allocation");
    cudaCheck(cudaMalloc(&devicePath, bytes), "path allocation");
    cudaCheck(cudaMemcpy(deviceDist, dist.data(), bytes, cudaMemcpyHostToDevice), "distance upload");
    cudaCheck(cudaMemcpy(devicePath, path.data(), bytes, cudaMemcpyHostToDevice), "path upload");

    const int n = static_cast<int>(numNodes);
    const int tiles = (n + TILE - 1) / TILE;
    const dim3 threads(TILE, TILE);
    for (int pivot = 0; pivot < tiles; ++pivot) {
        pivotKernel<<<1, threads>>>(deviceDist, devicePath, n, pivot);
        if (tiles > 1) {
            edgeKernel<<<dim3(tiles - 1, 2), threads>>>(deviceDist, devicePath, n, pivot, tiles);
            remainderKernel<<<dim3(tiles - 1, tiles - 1), threads>>>(deviceDist, devicePath, n, pivot);
        }
    }
    cudaCheck(cudaGetLastError(), "kernel launch");
    cudaCheck(cudaMemcpy(dist.data(), deviceDist, bytes, cudaMemcpyDeviceToHost), "distance download");
    cudaCheck(cudaMemcpy(path.data(), devicePath, bytes, cudaMemcpyDeviceToHost), "path download");
    cudaCheck(cudaFree(deviceDist), "distance release");
    cudaCheck(cudaFree(devicePath), "path release");
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
    // Do not charge the benchmark for the CUDA driver's one-time context setup.
    cudaCheck(cudaFree(nullptr), "runtime initialization");
    auto start = std::chrono::high_resolution_clock::now();
    
    floydWarshall(dist, path, numNodes);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    const double elapsedSeconds = std::chrono::duration<double>(end - start).count();
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate operations per second
    // Floyd-Warshall has O(n³) complexity
    double ops = (double)numNodes * numNodes * numNodes;
    double gflops = ops / elapsedSeconds / 1e9;
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
