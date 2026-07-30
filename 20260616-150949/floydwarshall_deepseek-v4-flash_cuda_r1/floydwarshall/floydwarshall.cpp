#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"
#include <cuda_runtime.h>

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;
constexpr int TILE_SIZE = 16;

// Index calculation for flattened 2D array
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

// CUDA kernel: for a fixed k, processes all (i,j) pairs in parallel
// Uses shared memory to cache the k-th row and column for coalesced access
__global__ void floydWarshallKernel(unsigned int* __restrict__ dist,
                                     unsigned int* __restrict__ path,
                                     const unsigned int n, const unsigned int k) {
    __shared__ unsigned int row_k[TILE_SIZE];
    __shared__ unsigned int col_k[TILE_SIZE];

    const unsigned int i = blockIdx.x * TILE_SIZE + threadIdx.x;
    const unsigned int j = blockIdx.y * TILE_SIZE + threadIdx.y;

    // Cooperative load from global -> shared memory
    if (threadIdx.y == 0 && i < n) {
        col_k[threadIdx.x] = dist[k * n + i];
    }
    if (threadIdx.x == 0 && j < n) {
        row_k[threadIdx.y] = dist[j * n + k];
    }
    __syncthreads();

    if (i < n && j < n) {
        const unsigned int distIJ = dist[j * n + i];
        const unsigned int newDist = col_k[threadIdx.x] + row_k[threadIdx.y];

        if (newDist < distIJ) {
            dist[j * n + i] = newDist;
            path[j * n + i] = k;
        }
    }
}

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess) {
        printf("CUDA error %d: %s at %s:%d\n", code, cudaGetErrorString(code), file, line);
        exit(code);
    }
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
            path[idx2(i, j, numNodes)] = static_cast<unsigned int>(j);
            path[idx2(j, i, numNodes)] = static_cast<unsigned int>(i);
        }
        path[idx2(j, j, numNodes)] = static_cast<unsigned int>(j);
    }
}

void floydWarshall(unsigned int* d_dist, unsigned int* d_path, const size_t numNodes) {
    const dim3 block(TILE_SIZE, TILE_SIZE);
    const dim3 grid((static_cast<unsigned int>(numNodes) + TILE_SIZE - 1) / TILE_SIZE,
                    (static_cast<unsigned int>(numNodes) + TILE_SIZE - 1) / TILE_SIZE);

    for (unsigned int k = 0; k < numNodes; ++k) {
        floydWarshallKernel<<<grid, block>>>(d_dist, d_path, 
                                              static_cast<unsigned int>(numNodes), k);
    }
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
    
    // Allocate host matrices
    std::vector<unsigned int> dist(numNodes * numNodes);
    std::vector<unsigned int> path(numNodes * numNodes);
    
    // Initialize
    printf("Initializing graph...\n");
    initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
    initializePathMatrix(path, numNodes);
    
    // Allocate device memory (also triggers CUDA context init)
    unsigned int *d_dist, *d_path;
    const size_t bytes = numNodes * numNodes * sizeof(unsigned int);
    CUDA_CHECK(cudaMalloc(&d_dist, bytes));
    CUDA_CHECK(cudaMalloc(&d_path, bytes));
    CUDA_CHECK(cudaMemcpy(d_dist, dist.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_path, path.data(), bytes, cudaMemcpyHostToDevice));
    
    // Run Floyd-Warshall - time only the computation (kernel loop)
    printf("Computing shortest paths...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    floydWarshall(d_dist, d_path, numNodes);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_ms = end - start;
    
    // Copy results back
    CUDA_CHECK(cudaMemcpy(dist.data(), d_dist, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(path.data(), d_path, bytes, cudaMemcpyDeviceToHost));
    
    // Cleanup device memory
    CUDA_CHECK(cudaFree(d_dist));
    CUDA_CHECK(cudaFree(d_path));
    
    printf("Computation time: %.2f ms\n", duration_ms.count());
    
    // Calculate operations per second
    // Floyd-Warshall has O(n³) complexity
    double ops = (double)numNodes * numNodes * numNodes;
    double gflops = ops / (duration_ms.count() / 1000.0) / 1e9;
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
