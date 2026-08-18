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
constexpr unsigned int TILE_SIZE = 32;

inline void cudaCheck(const cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "CUDA error during %s: %s\n", operation,
                     cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
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

__global__ void floydPivot(unsigned int* dist, unsigned int* path,
                           const unsigned int n, const unsigned int kBase) {
    __shared__ unsigned int tile[TILE_SIZE][TILE_SIZE];
    const unsigned int row = kBase + threadIdx.y;
    const unsigned int col = kBase + threadIdx.x;
    const bool valid = row < n && col < n;
    tile[threadIdx.y][threadIdx.x] = valid ? dist[col * n + row] : INF;
    __syncthreads();

    for (unsigned int kk = 0; kk < TILE_SIZE; ++kk) {
        if (valid && kBase + kk < n) {
            const unsigned int candidate = tile[threadIdx.y][kk] + tile[kk][threadIdx.x];
            if (candidate < tile[threadIdx.y][threadIdx.x]) {
                tile[threadIdx.y][threadIdx.x] = candidate;
                path[col * n + row] = kBase + kk;
            }
        }
        __syncthreads();
    }
    if (valid) dist[col * n + row] = tile[threadIdx.y][threadIdx.x];
}

__global__ void floydRow(unsigned int* dist, unsigned int* path,
                         const unsigned int n, const unsigned int kBase) {
    if (blockIdx.x == kBase / TILE_SIZE) return;
    __shared__ unsigned int pivot[TILE_SIZE][TILE_SIZE];
    __shared__ unsigned int tile[TILE_SIZE][TILE_SIZE];
    const unsigned int row = kBase + threadIdx.y;
    const unsigned int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    const bool valid = row < n && col < n;
    pivot[threadIdx.y][threadIdx.x] =
        (row < n && kBase + threadIdx.x < n) ? dist[(kBase + threadIdx.x) * n + row] : INF;
    tile[threadIdx.y][threadIdx.x] = valid ? dist[col * n + row] : INF;
    __syncthreads();

    for (unsigned int kk = 0; kk < TILE_SIZE; ++kk) {
        if (valid && kBase + kk < n) {
            const unsigned int candidate = pivot[threadIdx.y][kk] + tile[kk][threadIdx.x];
            if (candidate < tile[threadIdx.y][threadIdx.x]) {
                tile[threadIdx.y][threadIdx.x] = candidate;
                path[col * n + row] = kBase + kk;
            }
        }
        __syncthreads();
    }
    if (valid) dist[col * n + row] = tile[threadIdx.y][threadIdx.x];
}

__global__ void floydCol(unsigned int* dist, unsigned int* path,
                         const unsigned int n, const unsigned int kBase) {
    if (blockIdx.x == kBase / TILE_SIZE) return;
    __shared__ unsigned int tile[TILE_SIZE][TILE_SIZE];
    __shared__ unsigned int pivot[TILE_SIZE][TILE_SIZE];
    const unsigned int row = blockIdx.x * TILE_SIZE + threadIdx.y;
    const unsigned int col = kBase + threadIdx.x;
    const bool valid = row < n && col < n;
    tile[threadIdx.y][threadIdx.x] = valid ? dist[col * n + row] : INF;
    pivot[threadIdx.y][threadIdx.x] =
        (kBase + threadIdx.y < n && col < n) ? dist[col * n + (kBase + threadIdx.y)] : INF;
    __syncthreads();

    for (unsigned int kk = 0; kk < TILE_SIZE; ++kk) {
        if (valid && kBase + kk < n) {
            const unsigned int candidate = tile[threadIdx.y][kk] + pivot[kk][threadIdx.x];
            if (candidate < tile[threadIdx.y][threadIdx.x]) {
                tile[threadIdx.y][threadIdx.x] = candidate;
                path[col * n + row] = kBase + kk;
            }
        }
        __syncthreads();
    }
    if (valid) dist[col * n + row] = tile[threadIdx.y][threadIdx.x];
}

__global__ void floydIndependent(unsigned int* dist, unsigned int* path,
                                 const unsigned int n, const unsigned int kBase,
                                 const unsigned int tileCount) {
    const unsigned int tileRow = blockIdx.y;
    const unsigned int tileCol = blockIdx.x;
    if (tileRow >= tileCount || tileCol >= tileCount ||
        tileRow == kBase / TILE_SIZE || tileCol == kBase / TILE_SIZE) return;

    __shared__ unsigned int left[TILE_SIZE][TILE_SIZE];
    __shared__ unsigned int right[TILE_SIZE][TILE_SIZE];
    __shared__ unsigned int output[TILE_SIZE][TILE_SIZE];
    const unsigned int row = tileRow * TILE_SIZE + threadIdx.y;
    const unsigned int col = tileCol * TILE_SIZE + threadIdx.x;
    const bool valid = row < n && col < n;
    left[threadIdx.y][threadIdx.x] =
        (row < n && kBase + threadIdx.x < n) ? dist[(kBase + threadIdx.x) * n + row] : INF;
    right[threadIdx.y][threadIdx.x] =
        (kBase + threadIdx.y < n && col < n) ? dist[col * n + (kBase + threadIdx.y)] : INF;
    output[threadIdx.y][threadIdx.x] = valid ? dist[col * n + row] : INF;
    __syncthreads();

    for (unsigned int kk = 0; kk < TILE_SIZE; ++kk) {
        if (valid && kBase + kk < n) {
            const unsigned int candidate = left[threadIdx.y][kk] + right[kk][threadIdx.x];
            if (candidate < output[threadIdx.y][threadIdx.x]) {
                output[threadIdx.y][threadIdx.x] = candidate;
                path[col * n + row] = kBase + kk;
            }
        }
        __syncthreads();
    }
    if (valid) dist[col * n + row] = output[threadIdx.y][threadIdx.x];
}

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes) {
    if (numNodes == 0) return;
    const size_t elements = numNodes * numNodes;
    const size_t bytes = elements * sizeof(unsigned int);
    unsigned int* deviceDist = nullptr;
    unsigned int* devicePath = nullptr;
    cudaCheck(cudaMalloc(&deviceDist, bytes), "allocating distance matrix");
    cudaCheck(cudaMalloc(&devicePath, bytes), "allocating path matrix");
    cudaCheck(cudaMemcpy(deviceDist, dist.data(), bytes, cudaMemcpyHostToDevice), "copying distances");
    cudaCheck(cudaMemcpy(devicePath, path.data(), bytes, cudaMemcpyHostToDevice), "copying paths");

    const dim3 threads(TILE_SIZE, TILE_SIZE);
    const unsigned int tileCount = (static_cast<unsigned int>(numNodes) + TILE_SIZE - 1) / TILE_SIZE;
    for (unsigned int kBase = 0; kBase < numNodes; kBase += TILE_SIZE) {
        floydPivot<<<1, threads>>>(deviceDist, devicePath, static_cast<unsigned int>(numNodes), kBase);
        floydRow<<<tileCount, threads>>>(deviceDist, devicePath, static_cast<unsigned int>(numNodes), kBase);
        floydCol<<<tileCount, threads>>>(deviceDist, devicePath, static_cast<unsigned int>(numNodes), kBase);
        floydIndependent<<<dim3(tileCount, tileCount), threads>>>(
            deviceDist, devicePath, static_cast<unsigned int>(numNodes), kBase, tileCount);
        cudaCheck(cudaGetLastError(), "launching Floyd-Warshall kernels");
    }
    cudaCheck(cudaDeviceSynchronize(), "running Floyd-Warshall kernels");
    cudaCheck(cudaMemcpy(dist.data(), deviceDist, bytes, cudaMemcpyDeviceToHost), "copying distances back");
    cudaCheck(cudaMemcpy(path.data(), devicePath, bytes, cudaMemcpyDeviceToHost), "copying paths back");
    cudaCheck(cudaFree(deviceDist), "freeing distance matrix");
    cudaCheck(cudaFree(devicePath), "freeing path matrix");
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
