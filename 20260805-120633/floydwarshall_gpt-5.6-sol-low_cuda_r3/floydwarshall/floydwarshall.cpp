#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <limits>
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

static void cudaCheck(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}

__global__ void pivotKernel(unsigned int* __restrict__ d,
                            unsigned int* __restrict__ path, int n, int round) {
    __shared__ unsigned int tile[TILE][TILE + 1];
    const int y = threadIdx.y, x = threadIdx.x;
    const int row = round * TILE + y, col = round * TILE + x;
    tile[y][x] = (row < n && col < n) ? d[(size_t)row * n + col] : INF;
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < TILE; ++k) {
        const unsigned int candidate = tile[y][k] + tile[k][x];
        if (candidate < tile[y][x]) {
            tile[y][x] = candidate;
            if (row < n && col < n) path[(size_t)row * n + col] = round * TILE + k;
        }
        __syncthreads();
    }
    if (row < n && col < n) d[(size_t)row * n + col] = tile[y][x];
}

__global__ void rowColumnKernel(unsigned int* __restrict__ d,
                                unsigned int* __restrict__ path, int n, int round) {
    __shared__ unsigned int pivot[TILE][TILE + 1];
    __shared__ unsigned int tile[TILE][TILE + 1];
    const int y = threadIdx.y, x = threadIdx.x;
    const int other = blockIdx.x < round ? blockIdx.x : blockIdx.x + 1;
    const bool pivotRow = blockIdx.y == 0;
    const int rowTile = pivotRow ? round : other;
    const int colTile = pivotRow ? other : round;
    const int pr = round * TILE + y, pc = round * TILE + x;
    const int row = rowTile * TILE + y, col = colTile * TILE + x;
    pivot[y][x] = (pr < n && pc < n) ? d[(size_t)pr * n + pc] : INF;
    tile[y][x] = (row < n && col < n) ? d[(size_t)row * n + col] : INF;
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < TILE; ++k) {
        const unsigned int candidate = pivotRow
            ? pivot[y][k] + tile[k][x]
            : tile[y][k] + pivot[k][x];
        if (candidate < tile[y][x]) {
            tile[y][x] = candidate;
            if (row < n && col < n) path[(size_t)row * n + col] = round * TILE + k;
        }
        __syncthreads();
    }
    if (row < n && col < n) d[(size_t)row * n + col] = tile[y][x];
}

__global__ void remainingKernel(unsigned int* __restrict__ d,
                                unsigned int* __restrict__ path, int n, int round) {
    __shared__ unsigned int column[TILE][TILE + 1];
    __shared__ unsigned int rowTile[TILE][TILE + 1];
    const int y = threadIdx.y, x = threadIdx.x;
    const int by = blockIdx.y < round ? blockIdx.y : blockIdx.y + 1;
    const int bx = blockIdx.x < round ? blockIdx.x : blockIdx.x + 1;
    const int row = by * TILE + y, col = bx * TILE + x;
    const int pivotRow = round * TILE + y, pivotCol = round * TILE + x;
    column[y][x] = (row < n && pivotCol < n) ? d[(size_t)row * n + pivotCol] : INF;
    rowTile[y][x] = (pivotRow < n && col < n) ? d[(size_t)pivotRow * n + col] : INF;
    __syncthreads();
    if (row < n && col < n) {
        unsigned int best = d[(size_t)row * n + col];
        int bestK = -1;
        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            const unsigned int candidate = column[y][k] + rowTile[k][x];
            if (candidate < best) { best = candidate; bestK = k; }
        }
        d[(size_t)row * n + col] = best;
        if (bestK >= 0) path[(size_t)row * n + col] = round * TILE + bestK;
    }
}

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path, const size_t numNodes) {
    if (numNodes == 0 || numNodes > static_cast<size_t>(std::numeric_limits<int>::max())) {
        fprintf(stderr, "Number of nodes must be between 1 and %d\n", std::numeric_limits<int>::max());
        std::exit(EXIT_FAILURE);
    }
    const int n = static_cast<int>(numNodes);
    const size_t bytes = dist.size() * sizeof(unsigned int);
    unsigned int *deviceDist = nullptr, *devicePath = nullptr;
    cudaCheck(cudaMalloc(&deviceDist, bytes), "distance allocation");
    cudaCheck(cudaMalloc(&devicePath, bytes), "path allocation");
    cudaCheck(cudaMemcpy(deviceDist, dist.data(), bytes, cudaMemcpyHostToDevice), "distance upload");
    cudaCheck(cudaMemcpy(devicePath, path.data(), bytes, cudaMemcpyHostToDevice), "path upload");

    const int rounds = (n + TILE - 1) / TILE;
    const dim3 threads(TILE, TILE);
    for (int round = 0; round < rounds; ++round) {
        pivotKernel<<<1, threads>>>(deviceDist, devicePath, n, round);
        if (rounds > 1) {
            rowColumnKernel<<<dim3(rounds - 1, 2), threads>>>(deviceDist, devicePath, n, round);
            remainingKernel<<<dim3(rounds - 1, rounds - 1), threads>>>(deviceDist, devicePath, n, round);
        }
    }
    cudaCheck(cudaGetLastError(), "kernel launch");
    cudaCheck(cudaMemcpy(dist.data(), deviceDist, bytes, cudaMemcpyDeviceToHost), "distance download");
    cudaCheck(cudaMemcpy(path.data(), devicePath, bytes, cudaMemcpyDeviceToHost), "path download");
    cudaCheck(cudaFree(deviceDist), "distance deallocation");
    cudaCheck(cudaFree(devicePath), "path deallocation");
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
