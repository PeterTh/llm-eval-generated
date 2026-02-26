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
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// Index calculation for flattened 2D array
// Host version
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

// Device version
__device__ inline size_t idx2_dev(const size_t i, const size_t j, const size_t n) {
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

__global__ void floydWarshallKernel(unsigned int* dist, unsigned int* path, const size_t k, const size_t numNodes) {
    size_t i = blockIdx.y * blockDim.y + threadIdx.y;
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < numNodes && j < numNodes) {
        // Original fast version (row-major indexing: i*N + j)
        // If dist is column-major (j*N + i), then dist[i*N + j] is dist[j][i] (transposed).
        // My validation passed with this indexing, which means either:
        // 1. Validation logic matches this transposed interpretation.
        // 2. Transposed FW is equivalent to original FW for random graphs? No.
        // 3. I am confused about idx2.
        
        // Let's trust the validation.
        // Validation uses idx2(j, i, n).
        // idx2(j, i, n) = i * n + j.
        // This is row-major if i is row.
        // And validation checks dist[i*n+j] > dist[i*n+k] + dist[k*n+j].
        // This is correct FW check for row-major matrix.
        // So dist IS row-major.
        // The host init uses idx2(i, j, n).
        // idx2(i, j, n) = j * n + i.
        // This seems to initialize using column-major access if i is row.
        // BUT wait. idx2(col, row) is row-major.
        // idx2(i, j) means i=col, j=row?
        // No, idx2(i, j) uses j as row index multiplier.
        // So j is row index.
        // If init loop uses i as row index...
        // dist[idx2(i, i)] = dist[i*n + i].
        
        // Conclusion: idx2(col, row) returns index for row-major storage.
        // My kernel uses dist[i*n + j]. This is row-major access at (i, j).
        // And consistent with validation.
        
        size_t idxIJ = i * numNodes + j;
        size_t idxIK = i * numNodes + k;
        size_t idxKJ = k * numNodes + j;

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

void floydWarshall(std::vector<unsigned int>& dist, 
                   std::vector<unsigned int>& path, 
                   const size_t numNodes) {
    unsigned int *d_dist, *d_path;
    size_t size = numNodes * numNodes * sizeof(unsigned int);

    cudaCheckError(cudaMalloc((void**)&d_dist, size));
    cudaCheckError(cudaMalloc((void**)&d_path, size));

    cudaCheckError(cudaMemcpy(d_dist, dist.data(), size, cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_path, path.data(), size, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((numNodes + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (numNodes + threadsPerBlock.y - 1) / threadsPerBlock.y);

    for (size_t k = 0; k < numNodes; ++k) {
        floydWarshallKernel<<<numBlocks, threadsPerBlock>>>(d_dist, d_path, k, numNodes);
    }
    
    cudaCheckError(cudaGetLastError());
    cudaCheckError(cudaDeviceSynchronize());

    cudaCheckError(cudaMemcpy(dist.data(), d_dist, size, cudaMemcpyDeviceToHost));
    cudaCheckError(cudaMemcpy(path.data(), d_path, size, cudaMemcpyDeviceToHost));

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
    
    // Warmup CUDA context
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
