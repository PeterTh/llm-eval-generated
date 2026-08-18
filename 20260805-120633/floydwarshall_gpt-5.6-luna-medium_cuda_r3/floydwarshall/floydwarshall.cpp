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

namespace {

constexpr unsigned int TILE_X = 32;
constexpr unsigned int TILE_Y = 8;

[[noreturn]] void cudaFailure(const cudaError_t error, const char* operation) {
    fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(error));
    std::exit(EXIT_FAILURE);
}

void checkCuda(const cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        cudaFailure(error, operation);
    }
}

// x covers source nodes (the contiguous dimension in the column-major layout),
// and y covers destination nodes. A launch for each k is the global barrier.
__global__ void floydWarshallKernel(unsigned int* __restrict__ dist,
                                     unsigned int* __restrict__ path,
                                     const size_t numNodes,
                                     const size_t k) {
    __shared__ unsigned int distIK[TILE_X];
    __shared__ unsigned int distKJ[TILE_Y];

    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const size_t i = static_cast<size_t>(blockIdx.x) * TILE_X + tx;
    const size_t j = static_cast<size_t>(blockIdx.y) * TILE_Y + ty;

    // Every shared slot is initialized, including lanes outside a partial tile.
    distIK[tx] = i < numNodes ? dist[k * numNodes + i] : 0;
    distKJ[ty] = j < numNodes ? dist[j * numNodes + k] : 0;
    __syncthreads();

    if (i < numNodes && j < numNodes) {
        const size_t cell = j * numNodes + i;
        const unsigned int candidate = distIK[tx] + distKJ[ty];
        if (candidate < dist[cell]) {
            dist[cell] = candidate;
            path[cell] = static_cast<unsigned int>(k);
        }
    }
}

} // namespace

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes) {
    if (numNodes == 0) {
        return;
    }

    unsigned int* deviceDist = nullptr;
    unsigned int* devicePath = nullptr;
    const size_t matrixBytes = numNodes * numNodes * sizeof(unsigned int);

    checkCuda(cudaMalloc(reinterpret_cast<void**>(&deviceDist), matrixBytes),
              "allocating distance matrix");
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&devicePath), matrixBytes),
              "allocating path matrix");
    checkCuda(cudaMemcpy(deviceDist, dist.data(), matrixBytes, cudaMemcpyHostToDevice),
              "copying distance matrix to device");
    checkCuda(cudaMemcpy(devicePath, path.data(), matrixBytes, cudaMemcpyHostToDevice),
              "copying path matrix to device");

    const dim3 block(TILE_X, TILE_Y);
    const dim3 grid(static_cast<unsigned int>((numNodes + TILE_X - 1) / TILE_X),
                    static_cast<unsigned int>((numNodes + TILE_Y - 1) / TILE_Y));
    for (size_t k = 0; k < numNodes; ++k) {
        floydWarshallKernel<<<grid, block>>>(deviceDist, devicePath, numNodes, k);
        checkCuda(cudaGetLastError(), "launching Floyd-Warshall kernel");
    }
    checkCuda(cudaDeviceSynchronize(), "executing Floyd-Warshall kernels");

    checkCuda(cudaMemcpy(dist.data(), deviceDist, matrixBytes, cudaMemcpyDeviceToHost),
              "copying distance matrix to host");
    checkCuda(cudaMemcpy(path.data(), devicePath, matrixBytes, cudaMemcpyDeviceToHost),
              "copying path matrix to host");
    checkCuda(cudaFree(devicePath), "freeing path matrix");
    checkCuda(cudaFree(deviceDist), "freeing distance matrix");
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
