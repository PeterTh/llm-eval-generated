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

// CUDA kernel: each thread updates one (i,j) entry for fixed k
__global__ void floyd_kernel(unsigned int* dist, unsigned int* path, int n, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= n || j >= n) return;

    const size_t index = static_cast<size_t>(j) * static_cast<size_t>(n) + static_cast<size_t>(i);

    unsigned int distIJ = dist[index];
    unsigned int distIK = dist[static_cast<size_t>(k) * static_cast<size_t>(n) + static_cast<size_t>(i)];
    unsigned int distKJ = dist[static_cast<size_t>(j) * static_cast<size_t>(n) + static_cast<size_t>(k)];

    // Prevent overflow
    if (distIK < INF && distKJ < INF) {
        unsigned int newDist = distIK + distKJ;
        if (newDist < distIJ) {
            dist[index] = newDist;
            path[index] = static_cast<unsigned int>(k);
        }
    }
}

void floydWarshall(std::vector<unsigned int>& dist, 
                   std::vector<unsigned int>& path, 
                   const size_t numNodes) {
    // Allocate device memory and copy matrices once
    const size_t N = numNodes;
    const size_t bytes = N * N * sizeof(unsigned int);

    unsigned int* d_dist = nullptr;
    unsigned int* d_path = nullptr;

    if (cudaMalloc(reinterpret_cast<void**>(&d_dist), bytes) != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_dist failed\n");
        exit(1);
    }
    if (cudaMalloc(reinterpret_cast<void**>(&d_path), bytes) != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_path failed\n");
        cudaFree(d_dist);
        exit(1);
    }

    if (cudaMemcpy(d_dist, dist.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy to device d_dist failed\n");
        cudaFree(d_dist); cudaFree(d_path);
        exit(1);
    }
    if (cudaMemcpy(d_path, path.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy to device d_path failed\n");
        cudaFree(d_dist); cudaFree(d_path);
        exit(1);
    }

    // Configure kernel launch
    const dim3 block(16, 16);
    const dim3 grid(static_cast<unsigned int>((N + block.x - 1) / block.x),
                    static_cast<unsigned int>((N + block.y - 1) / block.y));

    // For each intermediate node k, launch kernel to update all pairs in parallel
    for (int k = 0; k < static_cast<int>(N); ++k) {
        floyd_kernel<<<grid, block>>>(d_dist, d_path, static_cast<int>(N), k);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA kernel launch failed at k=%d: %s\n", k, cudaGetErrorString(err));
            cudaFree(d_dist); cudaFree(d_path);
            exit(1);
        }
        if (cudaDeviceSynchronize() != cudaSuccess) {
            fprintf(stderr, "CUDA device synchronize failed at k=%d\n", k);
            cudaFree(d_dist); cudaFree(d_path);
            exit(1);
        }
    }

    // Copy results back
    if (cudaMemcpy(dist.data(), d_dist, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy to host d_dist failed\n");
        cudaFree(d_dist); cudaFree(d_path);
        exit(1);
    }
    if (cudaMemcpy(path.data(), d_path, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy to host d_path failed\n");
        cudaFree(d_dist); cudaFree(d_path);
        exit(1);
    }

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
