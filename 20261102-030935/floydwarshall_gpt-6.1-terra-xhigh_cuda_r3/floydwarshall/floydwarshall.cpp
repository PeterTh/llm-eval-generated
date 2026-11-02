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

// Each round processes a tile of intermediate vertices.  The two snapshot
// arrays preserve the row and column at each individual k, so that updates to
// both distance and path have exactly the same ordering as the scalar code.
constexpr int TILE = 32;

static void cudaCheck(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(status));
        exit(EXIT_FAILURE);
    }
}

__global__ void pivotTile(unsigned int* dist, unsigned int* path,
                          unsigned int* rows, unsigned int* cols,
                          size_t n, size_t base, int count) {
    __shared__ unsigned int tile[TILE][TILE];
    const int x = threadIdx.x, y = threadIdx.y;
    const size_t i = base + y, j = base + x;
    const bool valid = i < n && j < n;
    unsigned int d = valid ? dist[i * n + j] : INF;
    unsigned int p = valid ? path[i * n + j] : 0;
    tile[y][x] = d;
    __syncthreads();
    for (int t = 0; t < count; ++t) {
        if (y == t && j < n) rows[static_cast<size_t>(t) * n + j] = d;
        if (x == t && i < n) cols[static_cast<size_t>(t) * n + i] = d;
        const unsigned int candidate = tile[y][t] + tile[t][x];
        __syncthreads();
        if (candidate < d) { d = candidate; p = static_cast<unsigned int>(base + t); }
        tile[y][x] = d;
        __syncthreads();
    }
    if (valid) { dist[i * n + j] = d; path[i * n + j] = p; }
}

__global__ void edgeTiles(unsigned int* dist, unsigned int* path,
                          unsigned int* rows, unsigned int* cols,
                          size_t n, size_t base, int count, unsigned int pivot) {
    __shared__ unsigned int tile[TILE][TILE];
    __shared__ unsigned int pivotValues[TILE][TILE + 1];
    const int x = threadIdx.x, y = threadIdx.y;
    const unsigned int other = blockIdx.x + (blockIdx.x >= pivot);
    const bool rowTile = blockIdx.y == 0;
    const size_t i = (rowTile ? base : static_cast<size_t>(other) * TILE) + y;
    const size_t j = (rowTile ? static_cast<size_t>(other) * TILE : base) + x;
    const bool valid = i < n && j < n;
    unsigned int d = valid ? dist[i * n + j] : INF;
    unsigned int p = valid ? path[i * n + j] : 0;
    tile[y][x] = d;
    if (rowTile) {
        pivotValues[x][y] = (base + x < n && y < count)
            ? cols[static_cast<size_t>(y) * n + base + x] : INF;
    } else {
        pivotValues[y][x] = (base + x < n && y < count)
            ? rows[static_cast<size_t>(y) * n + base + x] : INF;
    }
    __syncthreads();
    for (int t = 0; t < count; ++t) {
        if (rowTile && y == t && j < n) rows[static_cast<size_t>(t) * n + j] = d;
        if (!rowTile && x == t && i < n) cols[static_cast<size_t>(t) * n + i] = d;
        const unsigned int left = rowTile ? pivotValues[y][t] : tile[y][t];
        const unsigned int right = rowTile ? tile[t][x] : pivotValues[t][x];
        const unsigned int candidate = left + right;
        __syncthreads();
        if (candidate < d) { d = candidate; p = static_cast<unsigned int>(base + t); }
        tile[y][x] = d;
        __syncthreads();
    }
    if (valid) { dist[i * n + j] = d; path[i * n + j] = p; }
}

__global__ void remainingTiles(unsigned int* dist, unsigned int* path,
                               const unsigned int* rows, const unsigned int* cols,
                               size_t n, size_t base, int count, unsigned int pivot) {
    __shared__ unsigned int left[TILE][TILE + 1];
    __shared__ unsigned int right[TILE][TILE];
    const int x = threadIdx.x, y = threadIdx.y;
    const unsigned int bx = blockIdx.x + (blockIdx.x >= pivot);
    const unsigned int by = blockIdx.y + (blockIdx.y >= pivot);
    const size_t i = static_cast<size_t>(by) * TILE + y;
    const size_t j = static_cast<size_t>(bx) * TILE + x;
    const size_t loadRow = static_cast<size_t>(by) * TILE + x;
    left[x][y] = (loadRow < n && y < count) ? cols[static_cast<size_t>(y) * n + loadRow] : INF;
    right[y][x] = (j < n && y < count) ? rows[static_cast<size_t>(y) * n + j] : INF;
    __syncthreads();
    if (i >= n || j >= n) return;
    unsigned int d = dist[i * n + j];
    unsigned int p = path[i * n + j];
    #pragma unroll 4
    for (int t = 0; t < count; ++t) {
        const unsigned int candidate = left[y][t] + right[t][x];
        if (candidate < d) { d = candidate; p = static_cast<unsigned int>(base + t); }
    }
    dist[i * n + j] = d;
    path[i * n + j] = p;
}

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes) {
    if (numNodes == 0) return;
    const size_t bytes = numNodes * numNodes * sizeof(unsigned int);
    const size_t scratchBytes = TILE * numNodes * sizeof(unsigned int);
    unsigned int *deviceDist, *devicePath, *rows, *cols;
    cudaCheck(cudaMalloc(&deviceDist, bytes), "allocating distances");
    cudaCheck(cudaMalloc(&devicePath, bytes), "allocating paths");
    cudaCheck(cudaMalloc(&rows, scratchBytes), "allocating row snapshots");
    cudaCheck(cudaMalloc(&cols, scratchBytes), "allocating column snapshots");
    cudaCheck(cudaMemcpy(deviceDist, dist.data(), bytes, cudaMemcpyHostToDevice), "copying distances to GPU");
    cudaCheck(cudaMemcpy(devicePath, path.data(), bytes, cudaMemcpyHostToDevice), "copying paths to GPU");
    const unsigned int tiles = static_cast<unsigned int>((numNodes + TILE - 1) / TILE);
    const dim3 threads(TILE, TILE);
    for (unsigned int pivot = 0; pivot < tiles; ++pivot) {
        const size_t base = static_cast<size_t>(pivot) * TILE;
        const int count = static_cast<int>(std::min(static_cast<size_t>(TILE), numNodes - base));
        pivotTile<<<1, threads>>>(deviceDist, devicePath, rows, cols, numNodes, base, count);
        if (tiles > 1) {
            edgeTiles<<<dim3(tiles - 1, 2), threads>>>(deviceDist, devicePath, rows, cols, numNodes, base, count, pivot);
            remainingTiles<<<dim3(tiles - 1, tiles - 1), threads>>>(deviceDist, devicePath, rows, cols, numNodes, base, count, pivot);
        }
    }
    cudaCheck(cudaGetLastError(), "launching Floyd-Warshall kernels");
    cudaCheck(cudaMemcpy(dist.data(), deviceDist, bytes, cudaMemcpyDeviceToHost), "copying distances from GPU");
    cudaCheck(cudaMemcpy(path.data(), devicePath, bytes, cudaMemcpyDeviceToHost), "copying paths from GPU");
    cudaCheck(cudaFree(cols), "freeing column snapshots");
    cudaCheck(cudaFree(rows), "freeing row snapshots");
    cudaCheck(cudaFree(devicePath), "freeing paths");
    cudaCheck(cudaFree(deviceDist), "freeing distances");
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
    // Exclude one-time CUDA context initialization from the computation timer.
    cudaCheck(cudaFree(nullptr), "initializing CUDA");
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
