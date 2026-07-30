#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

// Index calculation for flattened 2D array (column-major)
__host__ __device__ inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
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

// CUDA kernel for Floyd-Warshall - parallelizes over (i, j) pairs for a given k
__global__ void floydWarshallKernel(unsigned int* dist, unsigned int* path, size_t n, size_t k) {
    size_t i = blockIdx.y * blockDim.y + threadIdx.y;
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n && j < n) {
        size_t idxIJ = idx2(j, i, n);  // j * n + i (column-major)
        size_t idxIK = idx2(k, i, n);  // k * n + i
        size_t idxKJ = idx2(j, k, n);  // j * n + k

        unsigned int distIJ = dist[idxIJ];
        unsigned int distIK = dist[idxIK];
        unsigned int distKJ = dist[idxKJ];

        unsigned int newDist = distIK + distKJ;

        if (newDist < distIJ) {
            dist[idxIJ] = newDist;
            path[idxIJ] = k;
        }
    }
}

void floydWarshall(unsigned int* d_dist, unsigned int* d_path, const size_t numNodes) {
    // Define block and grid dimensions
    dim3 blockSize(16, 16);  // 256 threads per block
    dim3 gridSize((numNodes + blockSize.x - 1) / blockSize.x,
                  (numNodes + blockSize.y - 1) / blockSize.y);

    // Classic Floyd-Warshall algorithm with CUDA parallelization
    // For each intermediate node k (sequential)
    for (size_t k = 0; k < numNodes; ++k) {
        // Launch kernel to update all (i, j) pairs in parallel
        floydWarshallKernel<<<gridSize, blockSize>>>(d_dist, d_path, numNodes, k);
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
    
    // Allocate matrices
    std::vector<unsigned int> dist(numNodes * numNodes);
    std::vector<unsigned int> path(numNodes * numNodes);
    
    // Initialize
    printf("Initializing graph...\n");
    initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
    initializePathMatrix(path, numNodes);
    
    // Allocate device memory
    unsigned int *d_dist, *d_path;
    size_t matrixSize = numNodes * numNodes * sizeof(unsigned int);

    cudaMalloc(&d_dist, matrixSize);
    cudaMalloc(&d_path, matrixSize);

    // Copy data to device
    cudaMemcpy(d_dist, dist.data(), matrixSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_path, path.data(), matrixSize, cudaMemcpyHostToDevice);

    // Run Floyd-Warshall
    printf("Computing shortest paths...\n");
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);
    floydWarshall(d_dist, d_path, numNodes);
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, end);
    auto duration = std::chrono::milliseconds(static_cast<long>(milliseconds));

    printf("Computation time: %ld ms\n", duration.count());

    // Copy result back to host
    cudaMemcpy(dist.data(), d_dist, matrixSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(path.data(), d_path, matrixSize, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_dist);
    cudaFree(d_path);
    cudaEventDestroy(start);
    cudaEventDestroy(end);

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
