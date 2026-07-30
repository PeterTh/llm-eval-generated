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

// CUDA block size (16x16 = 256 threads per block)
constexpr size_t BLOCK_SIZE = 16;

// Index calculation for flattened 2D array
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

// CUDA kernel for one iteration of Floyd-Warshall (for a fixed k)
// Each thread handles one (i, j) pair with shared memory for the k-th row.
// The k-th column is read from global memory (non-coalesced but only one per row).
__global__ __launch_bounds__(BLOCK_SIZE * BLOCK_SIZE, 6) void floydWarshallKernel(
    unsigned int* __restrict__ dist,
    unsigned int* __restrict__ path,
    const unsigned int n, const unsigned int k)
{
    __shared__ unsigned int s_row[BLOCK_SIZE];

    const unsigned int i = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    const unsigned int j = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    // Load k-th row segment (coalesced row access)
    if (threadIdx.y == 0 && j < n) {
        s_row[threadIdx.x] = dist[k * n + j];
    }
    __syncthreads();

    if (i < n && j < n) {
        const unsigned int dIJ  = dist[i * n + j];
        const unsigned int dIK  = dist[i * n + k];  // single column access per row
        const unsigned int newDist = dIK + s_row[threadIdx.x];

        if (newDist < dIJ) {
            dist[i * n + j] = newDist;
            path[i * n + j] = k;
        }
    }
}

// Host-side helper to run Floyd-Warshall on GPU
// Returns time in milliseconds (only kernel execution, no transfers)
double floydWarshallGPU(unsigned int* h_dist, unsigned int* h_path, size_t numNodes) {
    unsigned int *d_dist, *d_path;
    const unsigned int n = static_cast<unsigned int>(numNodes);
    const size_t bytes = static_cast<size_t>(n) * n * sizeof(unsigned int);

    cudaMalloc(&d_dist, bytes);
    cudaMalloc(&d_path, bytes);
    cudaMemcpy(d_dist, h_dist, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_path, h_path, bytes, cudaMemcpyHostToDevice);

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((n + BLOCK_SIZE - 1) / BLOCK_SIZE,
                 (n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // CUDA events for precise GPU timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for (unsigned int k = 0; k < n; ++k) {
        floydWarshallKernel<<<gridDim, blockDim>>>(d_dist, d_path, n, k);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpuMs = 0;
    cudaEventElapsedTime(&gpuMs, start, stop);

    cudaMemcpy(h_dist, d_dist, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_path, d_path, bytes, cudaMemcpyDeviceToHost);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_dist);
    cudaFree(d_path);

    return static_cast<double>(gpuMs);
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
    
    // Run Floyd-Warshall on GPU
    printf("Computing shortest paths on GPU (CUDA)...\n");
    double gpuMs = floydWarshallGPU(dist.data(), path.data(), numNodes);
    
    printf("Computation time: %.0f ms\n", gpuMs);
    
    // Calculate operations per second
    // Floyd-Warshall has O(n³) complexity
    double ops = (double)numNodes * numNodes * numNodes;
    double gflops = ops / (gpuMs / 1000.0) / 1e9;
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
