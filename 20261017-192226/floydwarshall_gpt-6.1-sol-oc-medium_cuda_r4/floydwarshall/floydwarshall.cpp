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

void checkCuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA %s failed: %s\n", operation, cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}

constexpr unsigned int THREADS = 256;
constexpr unsigned int ROWS_PER_BLOCK = 4;

// Each warp updates consecutive columns, reusing its pivot-row value for
// multiple rows. Only one lane loads each source-to-pivot distance.
__global__ void updateDistances(unsigned int* __restrict__ dist,
                                unsigned int* __restrict__ path,
                                size_t n, size_t k) {
    const size_t j = static_cast<size_t>(blockIdx.x) * THREADS + threadIdx.x;
    const size_t firstRow = static_cast<size_t>(blockIdx.y) * ROWS_PER_BLOCK;
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int kj = j < n ? dist[k * n + j] : 0;

    #pragma unroll
    for (unsigned int r = 0; r < ROWS_PER_BLOCK; ++r) {
        const size_t i = firstRow + r;
        if (i < n && i != k) {
            unsigned int ik = lane == 0 ? dist[i * n + k] : 0;
            ik = __shfl_sync(0xffffffffu, ik, 0);
            if (j < n && j != k) {
                const size_t index = i * n + j;
                const unsigned int candidate = ik + kj;
                if (candidate < dist[index]) {
                    dist[index] = candidate;
                    path[index] = static_cast<unsigned int>(k);
                }
            }
        }
    }
}

} // namespace

void floydWarshall(std::vector<unsigned int>& dist, 
                   std::vector<unsigned int>& path, 
                   const size_t numNodes) {
    // The zero diagonal and nonnegative weights mean row/column k never
    // improve at step k. Leaving them unwritten makes all pivot reads race-free.
    // Ordered launches preserve the original strict-improvement path choices,
    // including ties, unlike reordering the intermediates in a blocked solver.
    if (numNodes == 0) {
        return;
    }
    const size_t bytes = dist.size() * sizeof(unsigned int);
    unsigned int* deviceDist = nullptr;
    unsigned int* devicePath = nullptr;
    checkCuda(cudaMalloc(&deviceDist, bytes), "distance allocation");
    checkCuda(cudaMalloc(&devicePath, bytes), "path allocation");
    checkCuda(cudaMemcpy(deviceDist, dist.data(), bytes, cudaMemcpyHostToDevice),
              "distance upload");
    checkCuda(cudaMemcpy(devicePath, path.data(), bytes, cudaMemcpyHostToDevice),
              "path upload");

    const dim3 grid(static_cast<unsigned int>((numNodes + THREADS - 1) / THREADS),
                    static_cast<unsigned int>((numNodes + ROWS_PER_BLOCK - 1) /
                                              ROWS_PER_BLOCK));
    for (size_t k = 0; k < numNodes; ++k) {
        updateDistances<<<grid, THREADS>>>(deviceDist, devicePath, numNodes, k);
    }
    checkCuda(cudaGetLastError(), "kernel launch");
    // These blocking copies also wait for all GPU work, so the benchmark's
    // existing timer includes transfers and the completed computation.
    checkCuda(cudaMemcpy(dist.data(), deviceDist, bytes, cudaMemcpyDeviceToHost),
              "distance download");
    checkCuda(cudaMemcpy(path.data(), devicePath, bytes, cudaMemcpyDeviceToHost),
              "path download");
    checkCuda(cudaFree(devicePath), "path release");
    checkCuda(cudaFree(deviceDist), "distance release");
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

    // Initialize the required CUDA device before timing the algorithm. The
    // timed region still includes allocation, transfers, and synchronization.
    checkCuda(cudaSetDevice(0), "device initialization");
    
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
