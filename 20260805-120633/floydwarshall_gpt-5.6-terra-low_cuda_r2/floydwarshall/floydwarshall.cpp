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

// The matrix is stored as dist[destination][source].  One launch is required
// for each k: the Floyd-Warshall dependency between successive k values is a
// global barrier.  Within a launch all (source, destination) pairs are
// independent.  Caching the needed k row/column per block avoids repeatedly
// fetching them for every element of the tile.
constexpr unsigned int CUDA_TILE_X = 32;
constexpr unsigned int CUDA_TILE_Y = 8;

__global__ void floydWarshallStep(unsigned int* __restrict__ dist,
                                  unsigned int* __restrict__ path,
                                  const size_t n, const size_t k) {
    __shared__ unsigned int distanceToK[CUDA_TILE_X];
    __shared__ unsigned int distanceFromK[CUDA_TILE_Y];

    const size_t source = static_cast<size_t>(blockIdx.x) * CUDA_TILE_X + threadIdx.x;
    const size_t destination = static_cast<size_t>(blockIdx.y) * CUDA_TILE_Y + threadIdx.y;

    if (threadIdx.y == 0 && source < n) {
        distanceToK[threadIdx.x] = dist[k * n + source];
    }
    if (threadIdx.x == 0 && destination < n) {
        distanceFromK[threadIdx.y] = dist[destination * n + k];
    }
    __syncthreads();

    if (source < n && destination < n) {
        const size_t offset = destination * n + source;
        const unsigned int candidate = distanceToK[threadIdx.x] + distanceFromK[threadIdx.y];
        if (candidate < dist[offset]) {
            dist[offset] = candidate;
            path[offset] = static_cast<unsigned int>(k);
        }
    }
}

bool checkCuda(const cudaError_t status, const char* operation) {
    if (status == cudaSuccess) {
        return true;
    }
    fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(status));
    return false;
}

bool floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes, float& elapsedMilliseconds) {
    elapsedMilliseconds = 0.0f;
    if (numNodes == 0) {
        return true;
    }

    const size_t bytes = numNodes * numNodes * sizeof(unsigned int);
    unsigned int* deviceDist = nullptr;
    unsigned int* devicePath = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    bool success = checkCuda(cudaMalloc(&deviceDist, bytes), "allocating distance matrix") &&
                   checkCuda(cudaMalloc(&devicePath, bytes), "allocating path matrix") &&
                   checkCuda(cudaMemcpy(deviceDist, dist.data(), bytes, cudaMemcpyHostToDevice),
                             "copying distance matrix to device") &&
                   checkCuda(cudaMemcpy(devicePath, path.data(), bytes, cudaMemcpyHostToDevice),
                             "copying path matrix to device") &&
                   checkCuda(cudaEventCreate(&start), "creating start event") &&
                   checkCuda(cudaEventCreate(&stop), "creating stop event");

    if (success) {
        const dim3 block(CUDA_TILE_X, CUDA_TILE_Y);
        const dim3 grid(static_cast<unsigned int>((numNodes + CUDA_TILE_X - 1) / CUDA_TILE_X),
                        static_cast<unsigned int>((numNodes + CUDA_TILE_Y - 1) / CUDA_TILE_Y));
        success = checkCuda(cudaEventRecord(start), "recording start event");
        for (size_t k = 0; success && k < numNodes; ++k) {
            floydWarshallStep<<<grid, block>>>(deviceDist, devicePath, numNodes, k);
            success = checkCuda(cudaGetLastError(), "launching Floyd-Warshall kernel");
        }
        success = success && checkCuda(cudaEventRecord(stop), "recording stop event") &&
                  checkCuda(cudaEventSynchronize(stop), "synchronizing GPU computation") &&
                  checkCuda(cudaEventElapsedTime(&elapsedMilliseconds, start, stop), "measuring GPU computation") &&
                  checkCuda(cudaMemcpy(dist.data(), deviceDist, bytes, cudaMemcpyDeviceToHost),
                            "copying distance matrix from device") &&
                  checkCuda(cudaMemcpy(path.data(), devicePath, bytes, cudaMemcpyDeviceToHost),
                            "copying path matrix from device");
    }

    if (start != nullptr) cudaEventDestroy(start);
    if (stop != nullptr) cudaEventDestroy(stop);
    if (deviceDist != nullptr) cudaFree(deviceDist);
    if (devicePath != nullptr) cudaFree(devicePath);
    return success;
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
    float gpuMilliseconds = 0.0f;
    if (!floydWarshall(dist, path, numNodes, gpuMilliseconds)) {
        return 1;
    }

    const auto duration = std::chrono::milliseconds(static_cast<long long>(gpuMilliseconds));
    
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
