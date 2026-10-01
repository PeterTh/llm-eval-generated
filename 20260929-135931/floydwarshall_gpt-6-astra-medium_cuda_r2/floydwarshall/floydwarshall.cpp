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
constexpr int TILE = 32;
constexpr int ROWS = 8;

void cudaCheck(cudaError_t status) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

// All tiles retain the original ascending-k, strict-improvement semantics,
// including the path matrix. Ordinary blocked Floyd-Warshall can select
// different intermediate vertices because it uses fully relaxed pivot panels.
// Instead, save the pivot row/column just before each individual k update.
__global__ void pivotTile(unsigned int* dist, unsigned int* path,
                          unsigned int* rowHistory, unsigned int* colHistory,
                          size_t n, size_t base, int count) {
    __shared__ unsigned int tile[TILE][TILE];
    const int x = threadIdx.x;
    const int y = threadIdx.y;
    unsigned int p[TILE / ROWS];
    #pragma unroll
    for (int r = 0; r < TILE / ROWS; ++r) {
        const int i = y + r * ROWS;
        const size_t offset = (base + i) * n + base + x;
        const bool valid = i < count && x < count;
        tile[i][x] = valid ? dist[offset] : INF;
        p[r] = valid ? path[offset] : 0;
    }
    __syncthreads();
    for (int k = 0; k < count; ++k) {
        if (y == 0 && x < count) {
            rowHistory[size_t(k) * n + base + x] = tile[k][x];
            colHistory[size_t(k) * n + base + x] = tile[x][k];
        }
        unsigned int candidate[TILE / ROWS];
        #pragma unroll
        for (int r = 0; r < TILE / ROWS; ++r)
            candidate[r] = tile[y + r * ROWS][k] + tile[k][x];
        // Finish all reads of the old pivot row/column before any writes.
        __syncthreads();
        #pragma unroll
        for (int r = 0; r < TILE / ROWS; ++r) {
            const int i = y + r * ROWS;
            if (candidate[r] < tile[i][x]) {
                tile[i][x] = candidate[r];
                p[r] = static_cast<unsigned int>(base + k);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int r = 0; r < TILE / ROWS; ++r) {
        const int i = y + r * ROWS;
        if (i < count && x < count) {
            const size_t offset = (base + i) * n + base + x;
            dist[offset] = tile[i][x];
            path[offset] = p[r];
        }
    }
}

// Each block independently relaxes one tile in the pivot row or column,
// exporting its own per-k history for the remaining tiles.
__global__ void panelTiles(unsigned int* dist, unsigned int* path,
                           unsigned int* rowHistory, unsigned int* colHistory,
                           size_t n, size_t pivot, int count) {
    __shared__ unsigned int tile[TILE][TILE];
    __shared__ unsigned int history[TILE][TILE];
    const bool column = blockIdx.y != 0;
    const size_t other = blockIdx.x + (blockIdx.x >= pivot);
    const size_t rowBase = (column ? other : pivot) * TILE;
    const size_t colBase = (column ? pivot : other) * TILE;
    const size_t base = pivot * TILE;
    const int x = threadIdx.x;
    const int y = threadIdx.y;
    unsigned int p[TILE / ROWS];
    #pragma unroll
    for (int r = 0; r < TILE / ROWS; ++r) {
        const int i = y + r * ROWS;
        const bool valid = rowBase + i < n && colBase + x < n;
        const size_t offset = (rowBase + i) * n + colBase + x;
        tile[i][x] = valid ? dist[offset] : INF;
        p[r] = valid ? path[offset] : 0;
        history[i][x] = i < count && x < count
            ? (column ? rowHistory : colHistory)[size_t(i) * n + base + x]
            : INF;
    }
    __syncthreads();
    for (int k = 0; k < count; ++k) {
        if (y == 0) {
            if (column && rowBase + x < n)
                colHistory[size_t(k) * n + rowBase + x] = tile[x][k];
            if (!column && colBase + x < n)
                rowHistory[size_t(k) * n + colBase + x] = tile[k][x];
        }
        unsigned int candidate[TILE / ROWS];
        #pragma unroll
        for (int r = 0; r < TILE / ROWS; ++r) {
            const int i = y + r * ROWS;
            candidate[r] = column ? tile[i][k] + history[k][x]
                                  : history[k][i] + tile[k][x];
        }
        __syncthreads();
        #pragma unroll
        for (int r = 0; r < TILE / ROWS; ++r) {
            const int i = y + r * ROWS;
            if (candidate[r] < tile[i][x]) {
                tile[i][x] = candidate[r];
                p[r] = static_cast<unsigned int>(base + k);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int r = 0; r < TILE / ROWS; ++r) {
        const int i = y + r * ROWS;
        if (rowBase + i < n && colBase + x < n) {
            const size_t offset = (rowBase + i) * n + colBase + x;
            dist[offset] = tile[i][x];
            path[offset] = p[r];
        }
    }
}

// The O(n^3) work: coalesced panel loads, shared-memory reuse, and four
// independent outputs per thread held in registers across all 32 pivots.
__global__ void remainingTiles(unsigned int* __restrict__ dist,
                               unsigned int* __restrict__ path,
                               const unsigned int* __restrict__ rowHistory,
                               const unsigned int* __restrict__ colHistory,
                               size_t n, size_t pivot, int count) {
    __shared__ unsigned int rows[TILE][TILE];
    __shared__ unsigned int cols[TILE][TILE];
    const size_t rowBase = (blockIdx.y + (blockIdx.y >= pivot)) * size_t(TILE);
    const size_t colBase = (blockIdx.x + (blockIdx.x >= pivot)) * size_t(TILE);
    const int x = threadIdx.x;
    const int y = threadIdx.y;
    unsigned int d[TILE / ROWS], p[TILE / ROWS];
    #pragma unroll
    for (int r = 0; r < TILE / ROWS; ++r) {
        const int i = y + r * ROWS;
        rows[i][x] = i < count && colBase + x < n
            ? rowHistory[size_t(i) * n + colBase + x] : INF;
        cols[i][x] = i < count && rowBase + x < n
            ? colHistory[size_t(i) * n + rowBase + x] : INF;
        const bool valid = rowBase + i < n && colBase + x < n;
        const size_t offset = (rowBase + i) * n + colBase + x;
        d[r] = valid ? dist[offset] : INF;
        p[r] = valid ? path[offset] : 0;
    }
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < TILE; ++k) {
        if (k < count) {
            const unsigned int right = rows[k][x];
            #pragma unroll
            for (int r = 0; r < TILE / ROWS; ++r) {
                const unsigned int candidate = cols[k][y + r * ROWS] + right;
                if (candidate < d[r]) {
                    d[r] = candidate;
                    p[r] = static_cast<unsigned int>(pivot * TILE + k);
                }
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < TILE / ROWS; ++r) {
        const int i = y + r * ROWS;
        if (rowBase + i < n && colBase + x < n) {
            const size_t offset = (rowBase + i) * n + colBase + x;
            dist[offset] = d[r];
            path[offset] = p[r];
        }
    }
}
} // namespace

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes) {
    // CUDA is required even for an empty graph; there is no CPU fallback.
    cudaCheck(cudaFree(nullptr));
    if (numNodes == 0) return;
    const size_t bytes = numNodes * numNodes * sizeof(unsigned int);
    const size_t historyBytes = TILE * numNodes * sizeof(unsigned int);
    unsigned int *deviceDist, *devicePath, *rowHistory, *colHistory;
    cudaCheck(cudaMalloc(&deviceDist, bytes));
    cudaCheck(cudaMalloc(&devicePath, bytes));
    cudaCheck(cudaMalloc(&rowHistory, historyBytes));
    cudaCheck(cudaMalloc(&colHistory, historyBytes));
    cudaCheck(cudaMemcpy(deviceDist, dist.data(), bytes, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(devicePath, path.data(), bytes, cudaMemcpyHostToDevice));
    const size_t tiles = (numNodes + TILE - 1) / TILE;
    const dim3 threads(TILE, ROWS);
    // Default-stream ordering is the global barrier between the three phases.
    for (size_t pivot = 0; pivot < tiles; ++pivot) {
        const int count = static_cast<int>(std::min(size_t(TILE), numNodes - pivot * TILE));
        pivotTile<<<1, threads>>>(deviceDist, devicePath, rowHistory, colHistory,
                                 numNodes, pivot * TILE, count);
        cudaCheck(cudaGetLastError());
        if (tiles > 1) {
            panelTiles<<<dim3(tiles - 1, 2), threads>>>(
                deviceDist, devicePath, rowHistory, colHistory, numNodes, pivot, count);
            cudaCheck(cudaGetLastError());
            remainingTiles<<<dim3(tiles - 1, tiles - 1), threads>>>(
                deviceDist, devicePath, rowHistory, colHistory, numNodes, pivot, count);
            cudaCheck(cudaGetLastError());
        }
    }
    // Blocking copies include completion of all GPU work in benchmark timing.
    cudaCheck(cudaMemcpy(dist.data(), deviceDist, bytes, cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(path.data(), devicePath, bytes, cudaMemcpyDeviceToHost));
    cudaCheck(cudaFree(colHistory));
    cudaCheck(cudaFree(rowHistory));
    cudaCheck(cudaFree(devicePath));
    cudaCheck(cudaFree(deviceDist));
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
    
    // Initialize the CUDA context before timing the algorithm. Device memory
    // allocation, transfers, and synchronization remain inside the timed call.
    cudaCheck(cudaFree(nullptr));

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
