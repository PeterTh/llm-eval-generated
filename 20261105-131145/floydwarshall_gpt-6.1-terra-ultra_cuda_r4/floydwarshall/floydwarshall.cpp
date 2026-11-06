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

constexpr int TILE = 32;
constexpr int ROWS_PER_THREAD = 4;

static void checkCuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}

// The input graph has nonnegative weights and a zero diagonal. Out-of-range
// entries are infinity, so incomplete tiles need no special cases in the loop.
__device__ __forceinline__ unsigned int loadCell(const unsigned int* dist, size_t n,
                                                  size_t row, size_t col) {
    return row < n && col < n ? dist[row * n + col] : INF;
}

__global__ void pivotTile(unsigned int* dist, size_t n, size_t pivot) {
    __shared__ unsigned int tile[TILE][TILE];
    const int x = threadIdx.x;
    const int y0 = threadIdx.y;
    const size_t base = pivot * TILE;
    #pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; ++r) {
        int y = y0 + r * blockDim.y;
        tile[y][x] = loadCell(dist, n, base + y, base + x);
    }
    __syncthreads();
    for (int k = 0; k < TILE; ++k) {
        #pragma unroll
        for (int r = 0; r < ROWS_PER_THREAD; ++r) {
            int y = y0 + r * blockDim.y;
            // The pivot row and column cannot improve; leaving them untouched
            // also avoids concurrent reads and writes to shared memory.
            if (y != k && x != k) {
                unsigned int candidate = tile[y][k] + tile[k][x];
                if (candidate < tile[y][x]) tile[y][x] = candidate;
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; ++r) {
        size_t y = base + y0 + r * blockDim.y;
        if (y < n && base + x < n) dist[y * n + base + x] = tile[y0 + r * blockDim.y][x];
    }
}

// Update the tiles in the pivot row and column after the pivot tile closes.
__global__ void pivotEdges(unsigned int* dist, size_t n, size_t pivot) {
    const size_t tileIndex = blockIdx.x;
    if (tileIndex == pivot) return;
    const bool isRow = blockIdx.y == 0;
    const size_t rowBase = (isRow ? pivot : tileIndex) * TILE;
    const size_t colBase = (isRow ? tileIndex : pivot) * TILE;
    const size_t pivotBase = pivot * TILE;
    const int x = threadIdx.x;
    const int y0 = threadIdx.y;
    __shared__ unsigned int diag[TILE][TILE];
    __shared__ unsigned int edge[TILE][TILE];
    #pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; ++r) {
        int y = y0 + r * blockDim.y;
        diag[y][x] = loadCell(dist, n, pivotBase + y, pivotBase + x);
        edge[y][x] = loadCell(dist, n, rowBase + y, colBase + x);
    }
    __syncthreads();
    for (int k = 0; k < TILE; ++k) {
        #pragma unroll
        for (int r = 0; r < ROWS_PER_THREAD; ++r) {
            int y = y0 + r * blockDim.y;
            // For row tiles the kth row is unchanged; for column tiles the
            // kth column is unchanged. Those entries are the shared inputs.
            if ((isRow && y != k) || (!isRow && x != k)) {
                unsigned int candidate = isRow ? diag[y][k] + edge[k][x]
                                               : edge[y][k] + diag[k][x];
                if (candidate < edge[y][x]) edge[y][x] = candidate;
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; ++r) {
        size_t y = rowBase + y0 + r * blockDim.y;
        if (y < n && colBase + x < n) dist[y * n + colBase + x] = edge[y0 + r * blockDim.y][x];
    }
}

__global__ void remainingTiles(unsigned int* dist, size_t n, size_t pivot) {
    const size_t tileCol = blockIdx.x, tileRow = blockIdx.y;
    if (tileCol == pivot || tileRow == pivot) return;
    const size_t rowBase = tileRow * TILE, colBase = tileCol * TILE, pivotBase = pivot * TILE;
    const int x = threadIdx.x, y0 = threadIdx.y;
    __shared__ unsigned int left[TILE][TILE];
    __shared__ unsigned int top[TILE][TILE];
    unsigned int value[ROWS_PER_THREAD];
    #pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; ++r) {
        int y = y0 + r * blockDim.y;
        left[y][x] = loadCell(dist, n, rowBase + y, pivotBase + x);
        top[y][x] = loadCell(dist, n, pivotBase + y, colBase + x);
        value[r] = loadCell(dist, n, rowBase + y, colBase + x);
    }
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < TILE; ++k) {
        #pragma unroll
        for (int r = 0; r < ROWS_PER_THREAD; ++r) {
            unsigned int candidate = left[y0 + r * blockDim.y][k] + top[k][x];
            value[r] = min(value[r], candidate);
        }
    }
    #pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; ++r) {
        size_t y = rowBase + y0 + r * blockDim.y;
        if (y < n && colBase + x < n) dist[y * n + colBase + x] = value[r];
    }
}

void floydWarshall(std::vector<unsigned int>& dist, const size_t numNodes) {
    if (numNodes == 0) return;
    unsigned int* deviceDist = nullptr;
    const size_t bytes = dist.size() * sizeof(unsigned int);
    checkCuda(cudaMalloc(&deviceDist, bytes), "allocating distance matrix");
    checkCuda(cudaMemcpy(deviceDist, dist.data(), bytes, cudaMemcpyHostToDevice), "copying input to device");
    const size_t tiles = (numNodes + TILE - 1) / TILE;
    const dim3 block(TILE, TILE / ROWS_PER_THREAD);
    for (size_t pivot = 0; pivot < tiles; ++pivot) {
        pivotTile<<<1, block>>>(deviceDist, numNodes, pivot);
        checkCuda(cudaGetLastError(), "launching pivot tile kernel");
        if (tiles > 1) {
            pivotEdges<<<dim3(tiles, 2), block>>>(deviceDist, numNodes, pivot);
            checkCuda(cudaGetLastError(), "launching pivot edge kernel");
            remainingTiles<<<dim3(tiles, tiles), block>>>(deviceDist, numNodes, pivot);
            checkCuda(cudaGetLastError(), "launching remaining tiles kernel");
        }
    }
    checkCuda(cudaMemcpy(dist.data(), deviceDist, bytes, cudaMemcpyDeviceToHost), "copying result from device");
    checkCuda(cudaFree(deviceDist), "freeing distance matrix");
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
    
    // Initialize
    printf("Initializing graph...\n");
    initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
    checkCuda(cudaFree(nullptr), "initializing CUDA");
    
    // Run Floyd-Warshall
    printf("Computing shortest paths...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    floydWarshall(dist, numNodes);
    
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
