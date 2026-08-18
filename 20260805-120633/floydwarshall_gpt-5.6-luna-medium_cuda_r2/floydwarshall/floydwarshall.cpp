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
constexpr int TILE_SIZE = 32;

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

__device__ __forceinline__ size_t deviceIndex(const int i, const int j, const int n) {
    return static_cast<size_t>(j) * static_cast<size_t>(n) + static_cast<size_t>(i);
}

__device__ __forceinline__ void relax(unsigned int& value, unsigned int& route,
                                      const unsigned int left, const unsigned int right,
                                      const int k) {
    const unsigned int candidate = left + right;
    if (candidate < value) {
        value = candidate;
        route = static_cast<unsigned int>(k);
    }
}

// All three kernels use column-major tiles, matching idx2() and giving each
// thread a coalesced access pattern when a matrix column is loaded.
__global__ void phase1(unsigned int* dist, unsigned int* path, const int n, const int kb) {
    __shared__ unsigned int tile[TILE_SIZE * TILE_SIZE];
    const int x = threadIdx.x;
    const int y = threadIdx.y;
    const int i = kb * TILE_SIZE + y;
    const int j = kb * TILE_SIZE + x;
    tile[x * TILE_SIZE + y] = (i < n && j < n) ? dist[deviceIndex(i, j, n)] : INF;
    __syncthreads();
    const int limit = min(TILE_SIZE, n - kb * TILE_SIZE);
    for (int kk = 0; kk < limit; ++kk) {
        if (x < limit && y < limit)
            relax(tile[x * TILE_SIZE + y], path[deviceIndex(i, j, n)],
                  tile[kk * TILE_SIZE + y], tile[x * TILE_SIZE + kk], kb * TILE_SIZE + kk);
        __syncthreads();
    }
    if (i < n && j < n) dist[deviceIndex(i, j, n)] = tile[x * TILE_SIZE + y];
}

__global__ void phase2Row(unsigned int* dist, unsigned int* path, const int n, const int kb) {
    __shared__ unsigned int pivot[TILE_SIZE * TILE_SIZE];
    __shared__ unsigned int tile[TILE_SIZE * TILE_SIZE];
    const int x = threadIdx.x;
    const int y = threadIdx.y;
    const int i = kb * TILE_SIZE + y;
    const int j = (blockIdx.x * TILE_SIZE) + x;
    const int pk = kb * TILE_SIZE + x;
    pivot[x * TILE_SIZE + y] = (i < n && pk < n) ? dist[deviceIndex(i, pk, n)] : INF;
    tile[x * TILE_SIZE + y] = (i < n && j < n) ? dist[deviceIndex(i, j, n)] : INF;
    __syncthreads();
    const int limit = min(TILE_SIZE, n - kb * TILE_SIZE);
    const int k = kb * TILE_SIZE;
    if (blockIdx.x != static_cast<unsigned int>(kb)) {
        for (int kk = 0; kk < limit; ++kk) {
            if (x < n - blockIdx.x * TILE_SIZE && y < limit)
                relax(tile[x * TILE_SIZE + y], path[deviceIndex(i, j, n)],
                      pivot[kk * TILE_SIZE + y], tile[x * TILE_SIZE + kk], k + kk);
            __syncthreads();
        }
    }
    if (i < n && j < n) dist[deviceIndex(i, j, n)] = tile[x * TILE_SIZE + y];
}

__global__ void phase2Column(unsigned int* dist, unsigned int* path, const int n, const int kb) {
    __shared__ unsigned int left[TILE_SIZE * TILE_SIZE];
    __shared__ unsigned int pivot[TILE_SIZE * TILE_SIZE];
    __shared__ unsigned int tile[TILE_SIZE * TILE_SIZE];
    const int x = threadIdx.x;
    const int y = threadIdx.y;
    const int i = blockIdx.x * TILE_SIZE + y;
    const int j = kb * TILE_SIZE + x;
    const int pk = kb * TILE_SIZE + x;
    left[x * TILE_SIZE + y] = (i < n && pk < n) ? dist[deviceIndex(i, pk, n)] : INF;
    pivot[x * TILE_SIZE + y] = (kb * TILE_SIZE + y < n && pk < n)
        ? dist[deviceIndex(kb * TILE_SIZE + y, pk, n)] : INF;
    tile[x * TILE_SIZE + y] = (i < n && j < n) ? dist[deviceIndex(i, j, n)] : INF;
    __syncthreads();
    const int limit = min(TILE_SIZE, n - kb * TILE_SIZE);
    if (blockIdx.x != static_cast<unsigned int>(kb)) {
        for (int kk = 0; kk < limit; ++kk) {
            if (x < limit && y < n - blockIdx.x * TILE_SIZE)
                relax(tile[x * TILE_SIZE + y], path[deviceIndex(i, j, n)],
                      left[kk * TILE_SIZE + y], pivot[x * TILE_SIZE + kk], kb * TILE_SIZE + kk);
            __syncthreads();
        }
    }
    if (i < n && j < n) dist[deviceIndex(i, j, n)] = tile[x * TILE_SIZE + y];
}

__global__ void phase3(unsigned int* dist, unsigned int* path, const int n, const int kb) {
    __shared__ unsigned int left[TILE_SIZE * TILE_SIZE];
    __shared__ unsigned int top[TILE_SIZE * TILE_SIZE];
    __shared__ unsigned int tile[TILE_SIZE * TILE_SIZE];
    const int x = threadIdx.x, y = threadIdx.y;
    const int ib = blockIdx.y, jb = blockIdx.x;
    const int i = ib * TILE_SIZE + y, j = jb * TILE_SIZE + x;
    const int ki = kb * TILE_SIZE + y, kj = kb * TILE_SIZE + x;
    left[x * TILE_SIZE + y] = (i < n && kj < n) ? dist[deviceIndex(i, kj, n)] : INF;
    top[x * TILE_SIZE + y] = (ki < n && j < n) ? dist[deviceIndex(ki, j, n)] : INF;
    tile[x * TILE_SIZE + y] = (i < n && j < n) ? dist[deviceIndex(i, j, n)] : INF;
    __syncthreads();
    const int limit = min(TILE_SIZE, n - kb * TILE_SIZE);
    for (int kk = 0; kk < limit; ++kk) {
        if (i < n && j < n)
            relax(tile[x * TILE_SIZE + y], path[deviceIndex(i, j, n)],
                  left[kk * TILE_SIZE + y], top[x * TILE_SIZE + kk], kb * TILE_SIZE + kk);
        __syncthreads();
    }
    if (i < n && j < n) dist[deviceIndex(i, j, n)] = tile[x * TILE_SIZE + y];
}

void checkCuda(const cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

void floydWarshall(std::vector<unsigned int>& dist, std::vector<unsigned int>& path,
                   const size_t numNodes) {
    const int n = static_cast<int>(numNodes);
    const size_t bytes = numNodes * numNodes * sizeof(unsigned int);
    unsigned int* deviceDist = nullptr;
    unsigned int* devicePath = nullptr;
    checkCuda(cudaMalloc(&deviceDist, bytes), "cudaMalloc(dist)");
    checkCuda(cudaMalloc(&devicePath, bytes), "cudaMalloc(path)");
    checkCuda(cudaMemcpy(deviceDist, dist.data(), bytes, cudaMemcpyHostToDevice), "copy dist");
    checkCuda(cudaMemcpy(devicePath, path.data(), bytes, cudaMemcpyHostToDevice), "copy path");

    const int tiles = (n + TILE_SIZE - 1) / TILE_SIZE;
    const dim3 threads(TILE_SIZE, TILE_SIZE);
    for (int kb = 0; kb < tiles; ++kb) {
        phase1<<<1, threads>>>(deviceDist, devicePath, n, kb);
        phase2Row<<<tiles, threads>>>(deviceDist, devicePath, n, kb);
        phase2Column<<<tiles, threads>>>(deviceDist, devicePath, n, kb);
        phase3<<<dim3(tiles, tiles), threads>>>(deviceDist, devicePath, n, kb);
        checkCuda(cudaGetLastError(), "Floyd-Warshall kernel launch");
    }
    checkCuda(cudaDeviceSynchronize(), "Floyd-Warshall execution");
    checkCuda(cudaMemcpy(dist.data(), deviceDist, bytes, cudaMemcpyDeviceToHost), "copy dist back");
    checkCuda(cudaMemcpy(path.data(), devicePath, bytes, cudaMemcpyDeviceToHost), "copy path back");
    cudaFree(devicePath);
    cudaFree(deviceDist);
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
