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

// Index calculation for flattened 2D array (row-major: idx = i * n + j)
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

// ---------------------------------------------------------------------------
// CUDA Floyd-Warshall kernel
// ---------------------------------------------------------------------------
// Each block caches the k-th row segment and k-th column segment for its tile
// in shared memory to reduce global memory traffic.
constexpr size_t CUDA_BLOCK_SIZE = 16;

template <size_t BLOCK_SIZE>
__global__ void floydWarshallKernel(unsigned int* __restrict__ dist,
                                     unsigned int* __restrict__ path,
                                     const unsigned int k,
                                     const size_t n) {
    unsigned int bx = blockIdx.x;
    unsigned int by = blockIdx.y;
    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;
    unsigned int j = bx * BLOCK_SIZE + tx;
    unsigned int i = by * BLOCK_SIZE + ty;

    __shared__ unsigned int rowK[BLOCK_SIZE];  // dist[k][bx*BLOCK_SIZE .. bx*BLOCK_SIZE+BLOCK_SIZE-1]
    __shared__ unsigned int colK[BLOCK_SIZE];  // dist[by*BLOCK_SIZE .. by*BLOCK_SIZE+BLOCK_SIZE-1][k]

    // First row of threads loads the k-th row segment for this block's j-range
    if (ty == 0 && j < n) {
        rowK[tx] = dist[k * n + j];
    }
    // First column of threads loads the k-th column segment for this block's i-range
    if (tx == 0 && i < n) {
        colK[ty] = dist[i * n + k];
    }

    __syncthreads();

    if (i < n && j < n) {
        unsigned int newDist = colK[ty] + rowK[tx];
        unsigned int idx = i * n + j;
        if (newDist < dist[idx]) {
            dist[idx] = newDist;
            path[idx] = k;
        }
    }
}

// Host-side wrapper that launches the CUDA kernel for each k iteration
void floydWarshallCUDA(unsigned int* d_dist, unsigned int* d_path, const size_t numNodes) {
    dim3 blockDim(CUDA_BLOCK_SIZE, CUDA_BLOCK_SIZE);
    dim3 gridDim((numNodes + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE,
                 (numNodes + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE);

    for (unsigned int k = 0; k < numNodes; ++k) {
        floydWarshallKernel<CUDA_BLOCK_SIZE><<<gridDim, blockDim>>>(d_dist, d_path, k, numNodes);
    }

    cudaDeviceSynchronize();
}

// ---------------------------------------------------------------------------
// Validation helpers (unchanged)
// ---------------------------------------------------------------------------
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

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                 \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

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
    
    // Allocate device memory
    unsigned int *d_dist = nullptr, *d_path = nullptr;
    size_t matrixBytes = numNodes * numNodes * sizeof(unsigned int);
    CUDA_CHECK(cudaMalloc(&d_dist, matrixBytes));
    CUDA_CHECK(cudaMalloc(&d_path, matrixBytes));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_dist, dist.data(), matrixBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_path, path.data(), matrixBytes, cudaMemcpyHostToDevice));
    
    // Run Floyd-Warshall on GPU
    printf("Computing shortest paths...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    floydWarshallCUDA(d_dist, d_path, numNodes);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(dist.data(), d_dist, matrixBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(path.data(), d_path, matrixBytes, cudaMemcpyDeviceToHost));
    
    // Clean up device memory
    CUDA_CHECK(cudaFree(d_dist));
    CUDA_CHECK(cudaFree(d_path));
    
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
