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

// Each path entry on the device is encoded as zero for a direct edge, or
// one plus the largest intermediate vertex on the chosen shortest path.
// Comparing this value on equal distances selects the first k at which the
// sequential algorithm could have found that distance.
constexpr int TILE = 32;
constexpr int ROWS_PER_THREAD = 4;

static void cudaCheck(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

__device__ __forceinline__ void relax(unsigned int& distance, unsigned int& via,
                                       unsigned int left, unsigned int right,
                                       unsigned int leftVia, unsigned int rightVia,
                                       unsigned int k) {
    const unsigned int candidate = left + right;
    const unsigned int candidateVia = max(max(leftVia, rightVia), k + 1);
    if (candidate < distance || (candidate == distance && candidateVia < via)) {
        distance = candidate;
        via = candidateVia;
    }
}

// Close the diagonal tile for this group of intermediate vertices.
__global__ void closePivot(unsigned int* dist, unsigned int* path, size_t n, int round) {
    __shared__ unsigned int d[TILE][TILE], p[TILE][TILE];
    const int x = threadIdx.x, y = threadIdx.y;
    const size_t base = static_cast<size_t>(round) * TILE;
    for (int t = 0; t < ROWS_PER_THREAD; ++t) {
        const int r = y + t * blockDim.y;
        const size_t i = base + r, j = base + x;
        d[r][x] = i < n && j < n ? dist[i * n + j] : INF;
        p[r][x] = i < n && j < n ? path[i * n + j] : 0;
    }
    __syncthreads();
    const int count = static_cast<int>(min(static_cast<size_t>(TILE), n - base));
    for (int k = 0; k < count; ++k) {
        for (int t = 0; t < ROWS_PER_THREAD; ++t) {
            const int r = y + t * blockDim.y;
            unsigned int v = d[r][x], q = p[r][x];
            relax(v, q, d[r][k], d[k][x], p[r][k], p[k][x],
                  static_cast<unsigned int>(base + k));
            // The k-th row and column are invariant, avoiding races with readers.
            if (r != k && x != k) { d[r][x] = v; p[r][x] = q; }
        }
        __syncthreads();
    }
    for (int t = 0; t < ROWS_PER_THREAD; ++t) {
        const int r = y + t * blockDim.y;
        const size_t i = base + r, j = base + x;
        if (i < n && j < n) { dist[i * n + j] = d[r][x]; path[i * n + j] = p[r][x]; }
    }
}

// Update the tiles in the pivot row and column using the closed pivot tile.
__global__ void updateEdges(unsigned int* dist, unsigned int* path, size_t n, int round) {
    __shared__ unsigned int pivot[TILE][TILE], pivotPath[TILE][TILE];
    __shared__ unsigned int edge[TILE][TILE], edgePath[TILE][TILE];
    const int x = threadIdx.x, y = threadIdx.y;
    const int tile = blockIdx.x < round ? blockIdx.x : blockIdx.x + 1;
    const bool row = blockIdx.y == 0;
    const size_t base = static_cast<size_t>(round) * TILE;
    const size_t rowBase = row ? base : static_cast<size_t>(tile) * TILE;
    const size_t colBase = row ? static_cast<size_t>(tile) * TILE : base;
    for (int t = 0; t < ROWS_PER_THREAD; ++t) {
        const int r = y + t * blockDim.y;
        const size_t pi = base + r, pj = base + x;
        const size_t i = rowBase + r, j = colBase + x;
        pivot[r][x] = pi < n && pj < n ? dist[pi * n + pj] : INF;
        pivotPath[r][x] = pi < n && pj < n ? path[pi * n + pj] : 0;
        edge[r][x] = i < n && j < n ? dist[i * n + j] : INF;
        edgePath[r][x] = i < n && j < n ? path[i * n + j] : 0;
    }
    __syncthreads();
    const int count = static_cast<int>(min(static_cast<size_t>(TILE), n - base));
    unsigned int values[ROWS_PER_THREAD], vias[ROWS_PER_THREAD];
    for (int t = 0; t < ROWS_PER_THREAD; ++t) {
        const int r = y + t * blockDim.y;
        values[t] = edge[r][x];
        vias[t] = edgePath[r][x];
    }
    // The closed pivot tile and the original edge tile are immutable here.
    // Their min-plus product computes the complete updated edge tile.
    for (int k = 0; k < count; ++k) {
        for (int t = 0; t < ROWS_PER_THREAD; ++t) {
            const int r = y + t * blockDim.y;
            if (row)
                relax(values[t], vias[t], pivot[r][k], edge[k][x], pivotPath[r][k], edgePath[k][x], base + k);
            else
                relax(values[t], vias[t], edge[r][k], pivot[k][x], edgePath[r][k], pivotPath[k][x], base + k);
        }
    }
    for (int t = 0; t < ROWS_PER_THREAD; ++t) {
        const int r = y + t * blockDim.y;
        const size_t i = rowBase + r, j = colBase + x;
        if (i < n && j < n) { dist[i * n + j] = values[t]; path[i * n + j] = vias[t]; }
    }
}

// The remaining tiles are independent once the pivot row and column are ready.
__global__ void updateRemaining(unsigned int* dist, unsigned int* path, size_t n, int round) {
    __shared__ unsigned int left[TILE][TILE], leftPath[TILE][TILE];
    __shared__ unsigned int right[TILE][TILE], rightPath[TILE][TILE];
    const int x = threadIdx.x, y = threadIdx.y;
    const int tileCol = blockIdx.x < round ? blockIdx.x : blockIdx.x + 1;
    const int tileRow = blockIdx.y < round ? blockIdx.y : blockIdx.y + 1;
    const size_t base = static_cast<size_t>(round) * TILE;
    const size_t rowBase = static_cast<size_t>(tileRow) * TILE;
    const size_t colBase = static_cast<size_t>(tileCol) * TILE;
    unsigned int values[ROWS_PER_THREAD], vias[ROWS_PER_THREAD];
    for (int t = 0; t < ROWS_PER_THREAD; ++t) {
        const int r = y + t * blockDim.y;
        const size_t i = rowBase + r, j = colBase + x;
        const size_t pi = base + r, pj = base + x;
        left[r][x] = i < n && pj < n ? dist[i * n + pj] : INF;
        leftPath[r][x] = i < n && pj < n ? path[i * n + pj] : 0;
        right[r][x] = pi < n && j < n ? dist[pi * n + j] : INF;
        rightPath[r][x] = pi < n && j < n ? path[pi * n + j] : 0;
        values[t] = i < n && j < n ? dist[i * n + j] : INF;
        vias[t] = i < n && j < n ? path[i * n + j] : 0;
    }
    __syncthreads();
    const int count = static_cast<int>(min(static_cast<size_t>(TILE), n - base));
    for (int k = 0; k < count; ++k) {
        const unsigned int rd = right[k][x], rp = rightPath[k][x];
        for (int t = 0; t < ROWS_PER_THREAD; ++t) {
            const int r = y + t * blockDim.y;
            relax(values[t], vias[t], left[r][k], rd, leftPath[r][k], rp, base + k);
        }
    }
    for (int t = 0; t < ROWS_PER_THREAD; ++t) {
        const int r = y + t * blockDim.y;
        const size_t i = rowBase + r, j = colBase + x;
        if (i < n && j < n) { dist[i * n + j] = values[t]; path[i * n + j] = vias[t]; }
    }
}

__global__ void decodePaths(unsigned int* path, size_t n) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n * n) path[i] = path[i] ? path[i] - 1 : i / n;
}

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes) {
    if (numNodes == 0) return;
    const size_t bytes = numNodes * numNodes * sizeof(unsigned int);
    unsigned int *deviceDist, *devicePath;
    cudaCheck(cudaMalloc(&deviceDist, bytes), "allocating distances");
    cudaCheck(cudaMalloc(&devicePath, bytes), "allocating paths");
    cudaCheck(cudaMemcpy(deviceDist, dist.data(), bytes, cudaMemcpyHostToDevice), "copying distances to GPU");
    cudaCheck(cudaMemset(devicePath, 0, bytes), "initializing GPU paths");
    const int tiles = static_cast<int>((numNodes + TILE - 1) / TILE);
    const dim3 threads(TILE, TILE / ROWS_PER_THREAD);
    for (int round = 0; round < tiles; ++round) {
        closePivot<<<1, threads>>>(deviceDist, devicePath, numNodes, round);
        if (tiles > 1) {
            updateEdges<<<dim3(tiles - 1, 2), threads>>>(deviceDist, devicePath, numNodes, round);
            updateRemaining<<<dim3(tiles - 1, tiles - 1), threads>>>(deviceDist, devicePath, numNodes, round);
        }
    }
    decodePaths<<<(numNodes * numNodes + 255) / 256, 256>>>(devicePath, numNodes);
    cudaCheck(cudaGetLastError(), "launching Floyd-Warshall kernels");
    cudaCheck(cudaMemcpy(dist.data(), deviceDist, bytes, cudaMemcpyDeviceToHost), "copying distances from GPU");
    cudaCheck(cudaMemcpy(path.data(), devicePath, bytes, cudaMemcpyDeviceToHost), "copying paths from GPU");
    cudaCheck(cudaFree(deviceDist), "freeing distances");
    cudaCheck(cudaFree(devicePath), "freeing paths");
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
    
    // Create the CUDA context before timing, just as host initialization is excluded.
    cudaCheck(cudaFree(nullptr), "initializing CUDA");

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
