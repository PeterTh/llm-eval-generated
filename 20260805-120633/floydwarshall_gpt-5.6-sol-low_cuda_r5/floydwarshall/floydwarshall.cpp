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

#define CUDA_CHECK(call) do {                                                   \
    const cudaError_t error_ = (call);                                           \
    if (error_ != cudaSuccess) {                                                 \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,       \
                cudaGetErrorString(error_));                                     \
        std::exit(EXIT_FAILURE);                                                 \
    }                                                                            \
} while (0)

constexpr int TILE = 32;

// One thread owns one matrix element.  The first two phases establish the
// pivot row and column before the independent phase-three tiles consume them.
__global__ void pivotKernel(unsigned int* __restrict__ dist,
                            unsigned int* __restrict__ path, int n, int round) {
    __shared__ unsigned int tile[TILE][TILE + 1];
    const int x = threadIdx.x, y = threadIdx.y;
    const int row = round * TILE + y, col = round * TILE + x;
    tile[y][x] = (row < n && col < n) ? dist[row * n + col] : INF;
    __syncthreads();
    const int limit = min(TILE, n - round * TILE);
    #pragma unroll
    for (int k = 0; k < TILE; ++k) {
        if (k < limit) {
            const unsigned int candidate = tile[y][k] + tile[k][x];
            __syncthreads();
            if (candidate < tile[y][x]) {
                tile[y][x] = candidate;
                if (row < n && col < n) path[row * n + col] = round * TILE + k;
            }
            __syncthreads();
        }
    }
    if (row < n && col < n) dist[row * n + col] = tile[y][x];
}

__global__ void edgeKernel(unsigned int* __restrict__ dist,
                           unsigned int* __restrict__ path, int n, int round) {
    __shared__ unsigned int pivot[TILE][TILE + 1];
    __shared__ unsigned int tile[TILE][TILE + 1];
    const int x = threadIdx.x, y = threadIdx.y;
    const bool columnTile = blockIdx.y != 0;
    const int tileIndex = blockIdx.x;
    const int row = (columnTile ? tileIndex : round) * TILE + y;
    const int col = (columnTile ? round : tileIndex) * TILE + x;
    const int prow = round * TILE + y, pcol = round * TILE + x;
    pivot[y][x] = (prow < n && pcol < n) ? dist[prow * n + pcol] : INF;
    tile[y][x] = (row < n && col < n) ? dist[row * n + col] : INF;
    __syncthreads();
    if (tileIndex == round) return;
    const int limit = min(TILE, n - round * TILE);
    #pragma unroll
    for (int k = 0; k < TILE; ++k) {
        if (k < limit) {
            const unsigned int candidate = columnTile
                ? tile[y][k] + pivot[k][x] : pivot[y][k] + tile[k][x];
            __syncthreads();
            if (candidate < tile[y][x]) {
                tile[y][x] = candidate;
                if (row < n && col < n) path[row * n + col] = round * TILE + k;
            }
            __syncthreads();
        }
    }
    if (row < n && col < n) dist[row * n + col] = tile[y][x];
}

__global__ void remainingKernel(unsigned int* __restrict__ dist,
                                unsigned int* __restrict__ path, int n, int round) {
    __shared__ unsigned int rowTile[TILE][TILE + 1];
    __shared__ unsigned int colTile[TILE][TILE + 1];
    const int x = threadIdx.x, y = threadIdx.y;
    const int row = blockIdx.y * TILE + y, col = blockIdx.x * TILE + x;
    const int pivot = round * TILE;
    rowTile[y][x] = (row < n && pivot + x < n) ? dist[row * n + pivot + x] : INF;
    colTile[y][x] = (pivot + y < n && col < n) ? dist[(pivot + y) * n + col] : INF;
    __syncthreads();
    if (blockIdx.x == static_cast<unsigned>(round) ||
        blockIdx.y == static_cast<unsigned>(round) || row >= n || col >= n) return;
    unsigned int value = dist[row * n + col];
    int bestK = -1;
    const int limit = min(TILE, n - pivot);
    #pragma unroll
    for (int k = 0; k < TILE; ++k) {
        if (k < limit) {
            const unsigned int candidate = rowTile[y][k] + colTile[k][x];
            if (candidate < value) { value = candidate; bestK = pivot + k; }
        }
    }
    dist[row * n + col] = value;
    if (bestK >= 0) path[row * n + col] = static_cast<unsigned int>(bestK);
}

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path, const size_t numNodes) {
    if (numNodes == 0) return;
    if (numNodes > static_cast<size_t>(INT_MAX)) {
        fprintf(stderr, "Matrix dimension exceeds CUDA kernel index range\n");
        std::exit(EXIT_FAILURE);
    }
    const size_t bytes = numNodes * numNodes * sizeof(unsigned int);
    unsigned int *deviceDist = nullptr, *devicePath = nullptr;
    CUDA_CHECK(cudaMalloc(&deviceDist, bytes));
    CUDA_CHECK(cudaMalloc(&devicePath, bytes));
    CUDA_CHECK(cudaMemcpy(deviceDist, dist.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(devicePath, path.data(), bytes, cudaMemcpyHostToDevice));
    const int n = static_cast<int>(numNodes);
    const int rounds = (n + TILE - 1) / TILE;
    const dim3 threads(TILE, TILE);
    for (int round = 0; round < rounds; ++round) {
        pivotKernel<<<1, threads>>>(deviceDist, devicePath, n, round);
        edgeKernel<<<dim3(rounds, 2), threads>>>(deviceDist, devicePath, n, round);
        remainingKernel<<<dim3(rounds, rounds), threads>>>(deviceDist, devicePath, n, round);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(dist.data(), deviceDist, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(path.data(), devicePath, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(devicePath));
    CUDA_CHECK(cudaFree(deviceDist));
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

    // Create the CUDA context outside the timed region.  Context startup is a
    // one-time runtime cost, not part of the shortest-path computation.
    CUDA_CHECK(cudaFree(nullptr));
    
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
