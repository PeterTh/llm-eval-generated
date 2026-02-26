#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;
constexpr int CUDA_BLOCK_SIZE = 16;

// Index calculation for flattened 2D array
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

__device__ __forceinline__ int idx2_device(const int i, const int j, const int n) noexcept {
    return j * n + i;
}

inline void checkCuda(const cudaError_t result, const char* context) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA error %s: %s\n", context, cudaGetErrorString(result));
        std::exit(1);
    }
}

__global__ void floydWarshallKernel(unsigned int* __restrict__ dist,
                                    unsigned int* __restrict__ path,
                                    const int n,
                                    const int k) {
    __shared__ unsigned int row_k[CUDA_BLOCK_SIZE];
    __shared__ unsigned int col_k[CUDA_BLOCK_SIZE];

    const int j = blockIdx.x * CUDA_BLOCK_SIZE + threadIdx.x;
    const int i = blockIdx.y * CUDA_BLOCK_SIZE + threadIdx.y;

    if (threadIdx.y == 0) {
        row_k[threadIdx.x] = (j < n) ? dist[idx2_device(j, k, n)] : INF;
    }
    if (threadIdx.x == 0) {
        col_k[threadIdx.y] = (i < n) ? dist[idx2_device(k, i, n)] : INF;
    }
    __syncthreads();

    if (i < n && j < n) {
        const int idx = idx2_device(j, i, n);
        const unsigned int distIJ = dist[idx];
        const unsigned int newDist = col_k[threadIdx.y] + row_k[threadIdx.x];

        if (newDist < distIJ) {
            dist[idx] = newDist;
            path[idx] = static_cast<unsigned int>(k);
        }
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
            path[idx2(i, j, numNodes)] = j;
            path[idx2(j, i, numNodes)] = i;
        }
        path[idx2(j, j, numNodes)] = j;
    }
}

void floydWarshall(std::vector<unsigned int>& dist, 
                   std::vector<unsigned int>& path, 
                   const size_t numNodes) {
    // Classic Floyd-Warshall algorithm
    // For each intermediate node k
    for (size_t k = 0; k < numNodes; ++k) {
        // For each source node i
        for (size_t i = 0; i < numNodes; ++i) {
            // For each destination node j
            for (size_t j = 0; j < numNodes; ++j) {
                const unsigned int distIJ = dist[idx2(j, i, numNodes)];
                const unsigned int distIK = dist[idx2(k, i, numNodes)];
                const unsigned int distKJ = dist[idx2(j, k, numNodes)];
                
                const unsigned int newDist = distIK + distKJ;
                
                if (newDist < distIJ) {
                    dist[idx2(j, i, numNodes)] = newDist;
                    path[idx2(j, i, numNodes)] = k;
                }
            }
        }
    }
}

float floydWarshallCUDA(std::vector<unsigned int>& dist,
                        std::vector<unsigned int>& path,
                        const size_t numNodes) {
    if (numNodes == 0) {
        return 0.0f;
    }
    if (numNodes > static_cast<size_t>(std::numeric_limits<int>::max())) {
        fprintf(stderr, "Number of nodes too large for CUDA kernel indexing\n");
        std::exit(1);
    }

    const size_t matrixSize = numNodes * numNodes;
    if (matrixSize / numNodes != numNodes) {
        fprintf(stderr, "Matrix size overflow\n");
        std::exit(1);
    }
    const size_t bytes = matrixSize * sizeof(unsigned int);
    if (bytes / sizeof(unsigned int) != matrixSize) {
        fprintf(stderr, "Matrix byte size overflow\n");
        std::exit(1);
    }

    const int n = static_cast<int>(numNodes);
    unsigned int* d_dist = nullptr;
    unsigned int* d_path = nullptr;

    checkCuda(cudaMalloc(&d_dist, bytes), "cudaMalloc(dist)");
    checkCuda(cudaMalloc(&d_path, bytes), "cudaMalloc(path)");
    checkCuda(cudaMemcpy(d_dist, dist.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy(dist H2D)");
    checkCuda(cudaMemcpy(d_path, path.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy(path H2D)");
    checkCuda(cudaFuncSetCacheConfig(floydWarshallKernel, cudaFuncCachePreferShared),
              "cudaFuncSetCacheConfig");

    const dim3 block(CUDA_BLOCK_SIZE, CUDA_BLOCK_SIZE);
    const dim3 grid((n + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE,
                    (n + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE);

    cudaEvent_t startEvent;
    cudaEvent_t stopEvent;
    checkCuda(cudaEventCreate(&startEvent), "cudaEventCreate(start)");
    checkCuda(cudaEventCreate(&stopEvent), "cudaEventCreate(stop)");
    checkCuda(cudaEventRecord(startEvent), "cudaEventRecord(start)");

    for (int k = 0; k < n; ++k) {
        floydWarshallKernel<<<grid, block>>>(d_dist, d_path, n, k);
        checkCuda(cudaGetLastError(), "floydWarshallKernel launch");
    }

    checkCuda(cudaEventRecord(stopEvent), "cudaEventRecord(stop)");
    checkCuda(cudaEventSynchronize(stopEvent), "cudaEventSynchronize(stop)");
    checkCuda(cudaGetLastError(), "floydWarshallKernel execution");

    float elapsedMs = 0.0f;
    checkCuda(cudaEventElapsedTime(&elapsedMs, startEvent, stopEvent), "cudaEventElapsedTime");
    checkCuda(cudaEventDestroy(startEvent), "cudaEventDestroy(start)");
    checkCuda(cudaEventDestroy(stopEvent), "cudaEventDestroy(stop)");

    checkCuda(cudaMemcpy(dist.data(), d_dist, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy(dist D2H)");
    checkCuda(cudaMemcpy(path.data(), d_path, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy(path D2H)");
    checkCuda(cudaFree(d_dist), "cudaFree(dist)");
    checkCuda(cudaFree(d_path), "cudaFree(path)");

    return elapsedMs;
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
    const float elapsedMs = floydWarshallCUDA(dist, path, numNodes);
    printf("Computation time: %.3f ms\n", elapsedMs);
    
    // Calculate operations per second
    // Floyd-Warshall has O(n³) complexity
    double ops = (double)numNodes * numNodes * numNodes;
    double gflops = (elapsedMs > 0.0f) ? (ops / (elapsedMs / 1000.0) / 1e9) : 0.0;
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
