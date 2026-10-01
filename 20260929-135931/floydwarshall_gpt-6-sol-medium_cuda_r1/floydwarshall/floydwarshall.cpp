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

void checkCuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA %s failed: %s\n", operation, cudaGetErrorString(result));
        exit(EXIT_FAILURE);
    }
}

// Phase 1: close the pivot tile. Every k iteration is synchronized before
// the next one reads its row and column.
__global__ void pivotKernel(unsigned int* dist, unsigned int* path, int n, int pivot) {
    __shared__ unsigned int tile[TILE][TILE];
    const int x = threadIdx.x;
    const int y = threadIdx.y;
    const int row = pivot * TILE + y;
    const int col = pivot * TILE + x;
    const bool valid = row < n && col < n;
    tile[y][x] = valid ? dist[static_cast<size_t>(row) * n + col] : INF;
    __syncthreads();

    const int count = min(TILE, n - pivot * TILE);
    int lastImprovement = -1;
    for (int k = 0; k < count; ++k) {
        const unsigned int candidate = tile[y][k] + tile[k][x];
        if (candidate < tile[y][x]) {
            tile[y][x] = candidate;
            lastImprovement = pivot * TILE + k;
        }
        __syncthreads();
    }
    if (valid) {
        const size_t index = static_cast<size_t>(row) * n + col;
        dist[index] = tile[y][x];
        if (lastImprovement >= 0) path[index] = lastImprovement;
    }
}

// Phase 2: close all tiles in the pivot row and column. Each block owns one
// output tile and reads the already completed pivot tile.
__global__ void edgeKernel(unsigned int* dist, unsigned int* path, int n,
                           int pivot) {
    __shared__ unsigned int center[TILE][TILE];
    __shared__ unsigned int edge[TILE][TILE];
    const int x = threadIdx.x;
    const int y = threadIdx.y;
    const int tileIndex = blockIdx.x < pivot ? blockIdx.x : blockIdx.x + 1;
    const bool pivotRow = blockIdx.y == 0;
    const int row = (pivotRow ? pivot : tileIndex) * TILE + y;
    const int col = (pivotRow ? tileIndex : pivot) * TILE + x;
    const int centerRow = pivot * TILE + y;
    const int centerCol = pivot * TILE + x;
    const bool valid = row < n && col < n;
    center[y][x] = centerRow < n && centerCol < n
        ? dist[static_cast<size_t>(centerRow) * n + centerCol] : INF;
    edge[y][x] = valid ? dist[static_cast<size_t>(row) * n + col] : INF;
    __syncthreads();

    const int count = min(TILE, n - pivot * TILE);
    int lastImprovement = -1;
    for (int k = 0; k < count; ++k) {
        const unsigned int candidate = pivotRow
            ? center[y][k] + edge[k][x]
            : edge[y][k] + center[k][x];
        if (candidate < edge[y][x]) {
            edge[y][x] = candidate;
            lastImprovement = pivot * TILE + k;
        }
        __syncthreads();
    }
    if (valid) {
        const size_t index = static_cast<size_t>(row) * n + col;
        dist[index] = edge[y][x];
        if (lastImprovement >= 0) path[index] = lastImprovement;
    }
}

// Phase 3: independent off-pivot tiles. Row and column inputs are cached in
// shared memory while each thread keeps its output distance in a register.
__global__ void remainingKernel(unsigned int* dist, unsigned int* path, int n, int pivot) {
    __shared__ unsigned int rowTile[TILE][TILE];
    __shared__ unsigned int colTile[TILE][TILE];
    const int x = threadIdx.x;
    const int y = threadIdx.y;
    const int tileRow = blockIdx.y < pivot ? blockIdx.y : blockIdx.y + 1;
    const int tileCol = blockIdx.x < pivot ? blockIdx.x : blockIdx.x + 1;
    const int row = tileRow * TILE + y;
    const int col = tileCol * TILE + x;
    const int pivotRow = pivot * TILE + y;
    const int pivotCol = pivot * TILE + x;
    rowTile[y][x] = pivotRow < n && col < n
        ? dist[static_cast<size_t>(pivotRow) * n + col] : INF;
    colTile[y][x] = row < n && pivotCol < n
        ? dist[static_cast<size_t>(row) * n + pivotCol] : INF;
    __syncthreads();

    const bool valid = row < n && col < n;
    const size_t index = static_cast<size_t>(row) * n + col;
    unsigned int best = valid ? dist[index] : INF;
    int lastImprovement = -1;
    const int count = min(TILE, n - pivot * TILE);
    for (int k = 0; k < count; ++k) {
        const unsigned int candidate = colTile[y][k] + rowTile[k][x];
        if (candidate < best) {
            best = candidate;
            lastImprovement = pivot * TILE + k;
        }
    }
    if (valid) {
        dist[index] = best;
        if (lastImprovement >= 0) path[index] = lastImprovement;
    }
}

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes) {
    if (numNodes == 0) return;
    if (numNodes > static_cast<size_t>(INT_MAX)) {
        fprintf(stderr, "Graph has too many nodes for CUDA indexing\n");
        exit(EXIT_FAILURE);
    }
    const int n = static_cast<int>(numNodes);
    const int tileCount = (n - 1) / TILE + 1;
    const size_t bytes = dist.size() * sizeof(unsigned int);
    unsigned int* deviceDist = nullptr;
    unsigned int* devicePath = nullptr;
    checkCuda(cudaMalloc(&deviceDist, bytes), "distance allocation");
    checkCuda(cudaMalloc(&devicePath, bytes), "path allocation");
    checkCuda(cudaMemcpy(deviceDist, dist.data(), bytes, cudaMemcpyHostToDevice), "distance upload");
    checkCuda(cudaMemcpy(devicePath, path.data(), bytes, cudaMemcpyHostToDevice), "path upload");

    const dim3 threads(TILE, TILE);
    for (int pivot = 0; pivot < tileCount; ++pivot) {
        pivotKernel<<<1, threads>>>(deviceDist, devicePath, n, pivot);
        if (tileCount > 1) {
            edgeKernel<<<dim3(tileCount - 1, 2), threads>>>(deviceDist, devicePath, n, pivot);
            remainingKernel<<<dim3(tileCount - 1, tileCount - 1), threads>>>(deviceDist, devicePath, n, pivot);
        }
    }
    checkCuda(cudaGetLastError(), "kernel launch");
    checkCuda(cudaMemcpy(dist.data(), deviceDist, bytes, cudaMemcpyDeviceToHost), "distance download");
    checkCuda(cudaMemcpy(path.data(), devicePath, bytes, cudaMemcpyDeviceToHost), "path download");
    checkCuda(cudaFree(deviceDist), "distance release");
    checkCuda(cudaFree(devicePath), "path release");
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

    // Exclude one-time CUDA context and kernel loading from the computation measurement.
    checkCuda(cudaFree(nullptr), "device initialization");
    cudaFuncAttributes attributes;
    checkCuda(cudaFuncGetAttributes(&attributes, pivotKernel), "pivot kernel loading");
    checkCuda(cudaFuncGetAttributes(&attributes, edgeKernel), "edge kernel loading");
    checkCuda(cudaFuncGetAttributes(&attributes, remainingKernel), "remaining kernel loading");
    
    // Run Floyd-Warshall
    printf("Computing shortest paths...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    floydWarshall(dist, path, numNodes);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    const double seconds = std::chrono::duration<double>(end - start).count();
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate operations per second
    // Floyd-Warshall has O(n³) complexity
    double ops = (double)numNodes * numNodes * numNodes;
    double gflops = ops / seconds / 1e9;
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
