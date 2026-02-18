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

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", \
                __FILE__, __LINE__, err, cudaGetErrorString(err), #call); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Index calculation for flattened 2D array
// __host__ __device__ to allow usage in kernel
inline __host__ __device__ size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
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

__global__ void floydWarshallKernel(unsigned int* dist, unsigned int* path, const size_t numNodes, const size_t k) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    
    // We want coalesced access.
    // idx2(col, row, n) = row * n + col.
    // So `col` should vary with `tx`.
    // Let's define:
    // col = bx * blockDim.x + tx;
    // row = by * blockDim.y + ty;
    
    const size_t col = bx * blockDim.x + tx;
    const size_t row = by * blockDim.y + ty;

    // Shared memory for column k and row k segments.
    // We need dist[row][k] (column k segment) and dist[k][col] (row k segment).
    
    __shared__ unsigned int s_col_k[32]; // dist[row][k]
    __shared__ unsigned int s_row_k[32]; // dist[k][col]

    // Load dist[k][col] into shared memory (s_row_k). Row k segment.
    // Coalesced access: idx2(col, k, n) = k*n + col.
    // Loaded by warp (ty=0).
    if (ty == 0) {
        if (col < numNodes)
            s_row_k[tx] = dist[idx2(col, k, numNodes)];
        else
            s_row_k[tx] = INF;
    }

    // Load dist[row][k] into shared memory (s_col_k). Column k segment.
    // idx2(k, row, n) = row*n + k.
    // We want s_col_k[t] to store dist[by*32 + t][k].
    // Loaded by warp (ty=0), using tx as index.
    if (ty == 0) {
        size_t r = by * blockDim.y + tx;
        if (r < numNodes)
            s_col_k[tx] = dist[idx2(k, r, numNodes)];
        else
            s_col_k[tx] = INF;
    }
    
    __syncthreads();

    if (col < numNodes && row < numNodes) {
        const unsigned int distRowK = s_col_k[ty];
        const unsigned int distKCol = s_row_k[tx];
        
        const size_t idx = idx2(col, row, numNodes);
        const unsigned int distRowCol = dist[idx];
        
        const unsigned int newDist = distRowK + distKCol;
        
        if (newDist < distRowCol) {
            dist[idx] = newDist;
            path[idx] = k;
        }
    }
}

void floydWarshall(std::vector<unsigned int>& dist, 
                   std::vector<unsigned int>& path, 
                   const size_t numNodes) {
    unsigned int *d_dist, *d_path;
    size_t size = numNodes * numNodes * sizeof(unsigned int);

    CUDA_CHECK(cudaMalloc((void**)&d_dist, size));
    CUDA_CHECK(cudaMalloc((void**)&d_path, size));

    CUDA_CHECK(cudaMemcpy(d_dist, dist.data(), size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_path, path.data(), size, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((numNodes + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (numNodes + threadsPerBlock.y - 1) / threadsPerBlock.y);

    for (size_t k = 0; k < numNodes; ++k) {
        floydWarshallKernel<<<numBlocks, threadsPerBlock>>>(d_dist, d_path, numNodes, k);
        CUDA_CHECK(cudaGetLastError());
    }
    
    // Ensure all kernels have finished before stopping timing (although memcpy does sync)
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(dist.data(), d_dist, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(path.data(), d_path, size, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_dist));
    CUDA_CHECK(cudaFree(d_path));
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
    
    // Initialize CUDA context to avoid timing it
    cudaFree(0);

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
