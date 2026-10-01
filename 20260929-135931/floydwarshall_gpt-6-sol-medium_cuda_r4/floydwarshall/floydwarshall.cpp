#include <algorithm>
#include <chrono>
#include <climits>
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

constexpr int TILE = 32;

void checkCuda(cudaError_t error) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}

__global__ void pivotKernel(unsigned int* dist, unsigned int* path, int pitch, int pivot) {
    __shared__ unsigned int tile[TILE][TILE];
    const int x = threadIdx.x, y = threadIdx.y;
    const int base = pivot * TILE;
    const size_t index = static_cast<size_t>(base + y) * pitch + base + x;
    tile[y][x] = dist[index];
    __syncthreads();
    unsigned int value = tile[y][x];
    for (int k = 0; k < TILE; ++k) {
        const unsigned int candidate = tile[y][k] + tile[k][x];
        if (candidate < value) {
            value = candidate;
            path[index] = base + k;
        }
        tile[y][x] = value;
        __syncthreads();
    }
    dist[index] = value;
}

__global__ void edgeKernel(unsigned int* dist, unsigned int* path, int pitch, int pivot) {
    __shared__ unsigned int center[TILE][TILE];
    __shared__ unsigned int tile[TILE][TILE];
    const int x = threadIdx.x, y = threadIdx.y;
    const int other = blockIdx.x + (blockIdx.x >= pivot);
    const bool row = blockIdx.y == 0;
    const int rowBase = (row ? pivot : other) * TILE;
    const int colBase = (row ? other : pivot) * TILE;
    const size_t index = static_cast<size_t>(rowBase + y) * pitch + colBase + x;
    center[y][x] = dist[static_cast<size_t>(pivot * TILE + y) * pitch + pivot * TILE + x];
    tile[y][x] = dist[index];
    __syncthreads();
    unsigned int value = tile[y][x];
    for (int k = 0; k < TILE; ++k) {
        const unsigned int candidate = row
            ? center[y][k] + tile[k][x]
            : tile[y][k] + center[k][x];
        if (candidate < value) {
            value = candidate;
            path[index] = pivot * TILE + k;
        }
        tile[y][x] = value;
        __syncthreads();
    }
    dist[index] = value;
}

__global__ void remainderKernel(unsigned int* dist, unsigned int* path, int pitch, int pivot) {
    __shared__ unsigned int row[TILE][TILE];
    __shared__ unsigned int col[TILE][TILE];
    const int x = threadIdx.x, y = threadIdx.y;
    const int tileCol = blockIdx.x + (blockIdx.x >= pivot);
    const int tileRow = blockIdx.y + (blockIdx.y >= pivot);
    const int rowBase = tileRow * TILE, colBase = tileCol * TILE;
    const size_t index = static_cast<size_t>(rowBase + y) * pitch + colBase + x;
    row[y][x] = dist[static_cast<size_t>(pivot * TILE + y) * pitch + colBase + x];
    col[y][x] = dist[static_cast<size_t>(rowBase + y) * pitch + pivot * TILE + x];
    __syncthreads();
    unsigned int value = dist[index];
    for (int k = 0; k < TILE; ++k) {
        const unsigned int candidate = col[y][k] + row[k][x];
        if (candidate < value) {
            value = candidate;
            path[index] = pivot * TILE + k;
        }
    }
    dist[index] = value;
}

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes) {
    if (numNodes == 0) return;
    if (numNodes > static_cast<size_t>(INT_MAX - TILE)) {
        fprintf(stderr, "Graph is too large for CUDA indexing\n");
        std::exit(EXIT_FAILURE);
    }
    const int pitch = static_cast<int>((numNodes + TILE - 1) / TILE * TILE);
    const int count = pitch / TILE;
    const size_t bytes = static_cast<size_t>(pitch) * pitch * sizeof(unsigned int);
    unsigned int *deviceDist, *devicePath;
    checkCuda(cudaMalloc(&deviceDist, bytes));
    checkCuda(cudaMalloc(&devicePath, bytes));
    // All padded edges are larger than any real edge, so extra vertices cannot
    // create a shorter path.  The padding also keeps every tile branch-free.
    checkCuda(cudaMemset(deviceDist, 0x3f, bytes));
    checkCuda(cudaMemcpy2D(deviceDist, pitch * sizeof(unsigned int), dist.data(),
                           numNodes * sizeof(unsigned int), numNodes * sizeof(unsigned int),
                           numNodes, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy2D(devicePath, pitch * sizeof(unsigned int), path.data(),
                           numNodes * sizeof(unsigned int), numNodes * sizeof(unsigned int),
                           numNodes, cudaMemcpyHostToDevice));
    const dim3 threads(TILE, TILE);
    for (int pivot = 0; pivot < count; ++pivot) {
        pivotKernel<<<1, threads>>>(deviceDist, devicePath, pitch, pivot);
        if (count > 1) {
            edgeKernel<<<dim3(count - 1, 2), threads>>>(deviceDist, devicePath, pitch, pivot);
            remainderKernel<<<dim3(count - 1, count - 1), threads>>>(deviceDist, devicePath, pitch, pivot);
        }
    }
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());
    checkCuda(cudaMemcpy2D(dist.data(), numNodes * sizeof(unsigned int), deviceDist,
                           pitch * sizeof(unsigned int), numNodes * sizeof(unsigned int),
                           numNodes, cudaMemcpyDeviceToHost));
    checkCuda(cudaMemcpy2D(path.data(), numNodes * sizeof(unsigned int), devicePath,
                           pitch * sizeof(unsigned int), numNodes * sizeof(unsigned int),
                           numNodes, cudaMemcpyDeviceToHost));
    checkCuda(cudaFree(deviceDist));
    checkCuda(cudaFree(devicePath));
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
    // Create the CUDA context outside the timed computation.
    checkCuda(cudaFree(nullptr));
    
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
