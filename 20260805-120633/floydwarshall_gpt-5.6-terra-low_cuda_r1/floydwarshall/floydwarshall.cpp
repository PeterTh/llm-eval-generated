#include <algorithm>
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

[[noreturn]] void cudaCheckFailed(const cudaError_t error, const char* operation,
                                  const char* file, const int line) {
    fprintf(stderr, "CUDA error during %s at %s:%d: %s\n", operation, file, line,
            cudaGetErrorString(error));
    std::exit(EXIT_FAILURE);
}

#define CUDA_CHECK(operation) do { \
    const cudaError_t cudaStatus = (operation); \
    if (cudaStatus != cudaSuccess) cudaCheckFailed(cudaStatus, #operation, __FILE__, __LINE__); \
} while (false)

// The source matrix is column-major: consecutive x threads access consecutive
// source vertices.  One launch per k is required to retain Floyd-Warshall's
// global intermediate-vertex ordering.
constexpr int TILE_I = 32;
constexpr int TILE_J = 8;

__global__ void floydWarshallStep(unsigned int* __restrict__ dist,
                                  unsigned int* __restrict__ path,
                                  const size_t numNodes, const size_t k) {
    const size_t i = static_cast<size_t>(blockIdx.x) * TILE_I + threadIdx.x;
    const size_t j = static_cast<size_t>(blockIdx.y) * TILE_J + threadIdx.y;

    __shared__ unsigned int distIK[TILE_I];
    __shared__ unsigned int distKJ[TILE_J];

    // Each value is reused by a full block.  Staging avoids redundant global
    // loads while preserving the exact update rule of the original program.
    if (threadIdx.y == 0 && i < numNodes) {
        distIK[threadIdx.x] = dist[k * numNodes + i];
    }
    if (threadIdx.x == 0 && j < numNodes) {
        distKJ[threadIdx.y] = dist[j * numNodes + k];
    }
    __syncthreads();

    if (i < numNodes && j < numNodes) {
        const size_t offset = j * numNodes + i;
        const unsigned int newDist = distIK[threadIdx.x] + distKJ[threadIdx.y];
        if (newDist < dist[offset]) {
            dist[offset] = newDist;
            path[offset] = static_cast<unsigned int>(k);
        }
    }
}

float floydWarshall(std::vector<unsigned int>& dist,
                    std::vector<unsigned int>& path,
                    const size_t numNodes) {
    if (numNodes == 0) return 0.0f;

    const size_t bytes = numNodes * numNodes * sizeof(unsigned int);
    unsigned int* deviceDist = nullptr;
    unsigned int* devicePath = nullptr;
    CUDA_CHECK(cudaMalloc(&deviceDist, bytes));
    CUDA_CHECK(cudaMalloc(&devicePath, bytes));
    CUDA_CHECK(cudaMemcpy(deviceDist, dist.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(devicePath, path.data(), bytes, cudaMemcpyHostToDevice));

    cudaEvent_t start{};
    cudaEvent_t end{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));
    CUDA_CHECK(cudaEventRecord(start));

    const dim3 block(TILE_I, TILE_J);
    const dim3 grid((numNodes + TILE_I - 1) / TILE_I,
                    (numNodes + TILE_J - 1) / TILE_J);
    for (size_t k = 0; k < numNodes; ++k) {
        floydWarshallStep<<<grid, block>>>(deviceDist, devicePath, numNodes, k);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));
    float elapsedMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedMs, start, end));
    CUDA_CHECK(cudaMemcpy(dist.data(), deviceDist, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(path.data(), devicePath, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(end));
    CUDA_CHECK(cudaFree(deviceDist));
    CUDA_CHECK(cudaFree(devicePath));
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
    const float elapsedMs = floydWarshall(dist, path, numNodes);
    
    printf("Computation time: %.3f ms\n", elapsedMs);
    
    // Calculate operations per second
    // Floyd-Warshall has O(n³) complexity
    double ops = (double)numNodes * numNodes * numNodes;
    double gflops = elapsedMs > 0.0f ? ops / (elapsedMs / 1000.0) / 1e9 : 0.0;
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
