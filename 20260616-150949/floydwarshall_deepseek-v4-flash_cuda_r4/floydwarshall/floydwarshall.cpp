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

// CUDA kernel for one Floyd-Warshall iteration over intermediate node k
// Each thread handles one (i,j) pair: dist[j][i] = min(dist[j][i], dist[k][i] + dist[j][k])
// Uses shared memory to cache the k-th row and column segments for the block
__global__ void floydWarshallKernel(unsigned int* __restrict__ dist,
                                     unsigned int* __restrict__ path,
                                     const size_t k, const size_t n) {
    extern __shared__ unsigned int shared[];
    unsigned int* const rowCache = shared;
    unsigned int* const colCache = &shared[blockDim.x];

    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t j = blockIdx.y * blockDim.y + threadIdx.y;

    // Load k-th row segment: dist[k * n + i] for i in [bx*BLOCK_X, (bx+1)*BLOCK_X)
    if (threadIdx.y == 0 && i < n) {
        rowCache[threadIdx.x] = dist[k * n + i];
    }

    // Load k-th column segment: dist[j * n + k] for j in [by*BLOCK_Y, (by+1)*BLOCK_Y)
    if (threadIdx.x == 0 && j < n) {
        colCache[threadIdx.y] = dist[j * n + k];
    }

    __syncthreads();

    if (i < n && j < n) {
        const unsigned int distIJ = dist[j * n + i];
        const unsigned int distIK = rowCache[threadIdx.x];
        const unsigned int distKJ = colCache[threadIdx.y];
        const unsigned int newDist = distIK + distKJ;

        if (newDist < distIJ) {
            dist[j * n + i] = newDist;
            path[j * n + i] = static_cast<unsigned int>(k);
        }
    }
}

void floydWarshall(std::vector<unsigned int>& dist, 
                   std::vector<unsigned int>& path, 
                   const size_t numNodes) {
    unsigned int *d_dist, *d_path;
    const size_t bytes = numNodes * numNodes * sizeof(unsigned int);

    // Allocate device memory
    cudaMalloc(&d_dist, bytes);
    cudaMalloc(&d_path, bytes);

    // Copy data to device
    cudaMemcpy(d_dist, dist.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_path, path.data(), bytes, cudaMemcpyHostToDevice);

    // Kernel launch configuration
    constexpr unsigned int BLOCK_SIZE = 32;
    const dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    const dim3 grid((numNodes + BLOCK_SIZE - 1) / BLOCK_SIZE,
                    (numNodes + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Shared memory: rowCache (BLOCK_SIZE) + colCache (BLOCK_SIZE)
    const size_t sharedMemBytes = (BLOCK_SIZE + BLOCK_SIZE) * sizeof(unsigned int);

    // Run Floyd-Warshall on GPU: one kernel launch per intermediate node k
    for (size_t k = 0; k < numNodes; ++k) {
        floydWarshallKernel<<<grid, block, sharedMemBytes>>>(
            d_dist, d_path, k, numNodes);
    }

    // Wait for all kernels to complete
    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy(dist.data(), d_dist, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(path.data(), d_path, bytes, cudaMemcpyDeviceToHost);

    // Clean up device memory
    cudaFree(d_dist);
    cudaFree(d_path);
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
    
    // Warm up CUDA driver before timing
    cudaFree(0);
    
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
