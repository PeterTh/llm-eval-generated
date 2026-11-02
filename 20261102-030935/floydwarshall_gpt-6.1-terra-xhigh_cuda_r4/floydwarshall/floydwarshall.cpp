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

// The second component is the largest intermediate vertex on a shortest
// path, plus one. Zero denotes a direct edge. Minimizing this component on
// distance ties reproduces the sequential algorithm's strict-update path.
constexpr int TILE = 32;
constexpr int THREAD_ROWS = 8;

static void cudaCheck(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}

__device__ __forceinline__ void relax(unsigned int& d, unsigned int& p,
                                       unsigned int a, unsigned int ap,
                                       unsigned int b, unsigned int bp,
                                       unsigned int k) {
    unsigned int candidate = a + b;
    unsigned int candidatePath = max(max(ap, bp), k + 1);
    if (candidate < d || (candidate == d && candidatePath < p)) {
        d = candidate;
        p = candidatePath;
    }
}

// Close the diagonal tile for this group of intermediate vertices.
__global__ void pivotTile(unsigned int* dist, unsigned int* paths, size_t n, int round) {
    __shared__ unsigned int d[TILE][TILE], p[TILE][TILE];
    int x = threadIdx.x, y = threadIdx.y;
    size_t col = size_t(round) * TILE + x;
    for (int r = y; r < TILE; r += THREAD_ROWS) {
        size_t row = size_t(round) * TILE + r;
        d[r][x] = row < n && col < n ? dist[row * n + col] : INF;
        p[r][x] = row < n && col < n ? paths[row * n + col] : 0;
    }
    __syncthreads();
    for (int k = 0; k < TILE && size_t(round) * TILE + k < n; ++k) {
        for (int r = y; r < TILE; r += THREAD_ROWS) {
            unsigned int value = d[r][x], label = p[r][x];
            relax(value, label, d[r][k], p[r][k], d[k][x], p[k][x], round * TILE + k);
            // The pivot row and column do not change in this iteration.
            if (r != k && x != k) { d[r][x] = value; p[r][x] = label; }
        }
        __syncthreads();
    }
    for (int r = y; r < TILE; r += THREAD_ROWS) {
        size_t row = size_t(round) * TILE + r;
        if (row < n && col < n) { dist[row * n + col] = d[r][x]; paths[row * n + col] = p[r][x]; }
    }
}

// Update the tiles in the pivot row and column.
__global__ void edgeTiles(unsigned int* dist, unsigned int* paths, size_t n, int round) {
    __shared__ unsigned int pivot[TILE][TILE], pp[TILE][TILE];
    __shared__ unsigned int tile[TILE][TILE], tp[TILE][TILE];
    int other = blockIdx.x;
    if (other >= round) ++other;
    bool rowTile = blockIdx.y == 0;
    int tileRow = rowTile ? round : other, tileCol = rowTile ? other : round;
    int x = threadIdx.x, y = threadIdx.y;
    for (int r = y; r < TILE; r += THREAD_ROWS) {
        size_t pr = size_t(round) * TILE + r, pc = size_t(round) * TILE + x;
        size_t row = size_t(tileRow) * TILE + r, col = size_t(tileCol) * TILE + x;
        pivot[r][x] = pr < n && pc < n ? dist[pr * n + pc] : INF;
        pp[r][x] = pr < n && pc < n ? paths[pr * n + pc] : 0;
        tile[r][x] = row < n && col < n ? dist[row * n + col] : INF;
        tp[r][x] = row < n && col < n ? paths[row * n + col] : 0;
    }
    __syncthreads();
    for (int k = 0; k < TILE && size_t(round) * TILE + k < n; ++k) {
        for (int r = y; r < TILE; r += THREAD_ROWS) {
            unsigned int value = tile[r][x], label = tp[r][x];
            if (rowTile)
                relax(value, label, pivot[r][k], pp[r][k], tile[k][x], tp[k][x], round * TILE + k);
            else
                relax(value, label, tile[r][k], tp[r][k], pivot[k][x], pp[k][x], round * TILE + k);
            // The row/column read by all threads is unchanged for this k.
            if ((rowTile && r != k) || (!rowTile && x != k)) {
                tile[r][x] = value; tp[r][x] = label;
            }
        }
        __syncthreads();
    }
    for (int r = y; r < TILE; r += THREAD_ROWS) {
        size_t row = size_t(tileRow) * TILE + r, col = size_t(tileCol) * TILE + x;
        if (row < n && col < n) { dist[row * n + col] = tile[r][x]; paths[row * n + col] = tp[r][x]; }
    }
}

// Each block owns one remaining tile, keeping its outputs in registers while
// reusing the corresponding pivot row and column tiles from shared memory.
__global__ void remainingTiles(unsigned int* dist, unsigned int* paths, size_t n, int round) {
    __shared__ unsigned int left[TILE][TILE], lp[TILE][TILE];
    __shared__ unsigned int top[TILE][TILE], topP[TILE][TILE];
    int tileRow = blockIdx.y, tileCol = blockIdx.x;
    if (tileRow >= round) ++tileRow;
    if (tileCol >= round) ++tileCol;
    int x = threadIdx.x, y = threadIdx.y;
    size_t col = size_t(tileCol) * TILE + x, pivotCol = size_t(round) * TILE + x;
    unsigned int value[TILE / THREAD_ROWS], label[TILE / THREAD_ROWS];
    for (int t = 0; t < TILE / THREAD_ROWS; ++t) {
        int r = y + t * THREAD_ROWS;
        size_t row = size_t(tileRow) * TILE + r, pivotRow = size_t(round) * TILE + r;
        left[r][x] = row < n && pivotCol < n ? dist[row * n + pivotCol] : INF;
        lp[r][x] = row < n && pivotCol < n ? paths[row * n + pivotCol] : 0;
        top[r][x] = pivotRow < n && col < n ? dist[pivotRow * n + col] : INF;
        topP[r][x] = pivotRow < n && col < n ? paths[pivotRow * n + col] : 0;
        value[t] = row < n && col < n ? dist[row * n + col] : INF;
        label[t] = row < n && col < n ? paths[row * n + col] : 0;
    }
    __syncthreads();
    for (int k = 0; k < TILE && size_t(round) * TILE + k < n; ++k) {
        #pragma unroll
        for (int t = 0; t < TILE / THREAD_ROWS; ++t) {
            int r = y + t * THREAD_ROWS;
            relax(value[t], label[t], left[r][k], lp[r][k], top[k][x], topP[k][x], round * TILE + k);
        }
    }
    for (int t = 0; t < TILE / THREAD_ROWS; ++t) {
        size_t row = size_t(tileRow) * TILE + y + t * THREAD_ROWS;
        if (row < n && col < n) { dist[row * n + col] = value[t]; paths[row * n + col] = label[t]; }
    }
}

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes) {
    if (numNodes == 0) return;
    const size_t bytes = dist.size() * sizeof(unsigned int);
    unsigned int *deviceDist = nullptr, *devicePath = nullptr;
    cudaCheck(cudaMalloc(&deviceDist, bytes), "allocating distances");
    cudaCheck(cudaMalloc(&devicePath, bytes), "allocating paths");
    cudaCheck(cudaMemcpy(deviceDist, dist.data(), bytes, cudaMemcpyHostToDevice), "copying distances to GPU");
    cudaCheck(cudaMemset(devicePath, 0, bytes), "initializing GPU paths");
    const int rounds = static_cast<int>((numNodes + TILE - 1) / TILE);
    const dim3 threads(TILE, THREAD_ROWS);
    for (int round = 0; round < rounds; ++round) {
        pivotTile<<<1, threads>>>(deviceDist, devicePath, numNodes, round);
        if (rounds > 1) {
            edgeTiles<<<dim3(rounds - 1, 2), threads>>>(deviceDist, devicePath, numNodes, round);
            remainingTiles<<<dim3(rounds - 1, rounds - 1), threads>>>(deviceDist, devicePath, numNodes, round);
        }
    }
    cudaCheck(cudaGetLastError(), "launching Floyd-Warshall kernels");
    cudaCheck(cudaMemcpy(dist.data(), deviceDist, bytes, cudaMemcpyDeviceToHost), "copying distances from GPU");
    cudaCheck(cudaMemcpy(path.data(), devicePath, bytes, cudaMemcpyDeviceToHost), "copying paths from GPU");
    cudaCheck(cudaFree(deviceDist), "freeing distances");
    cudaCheck(cudaFree(devicePath), "freeing paths");
    for (size_t i = 0; i < numNodes; ++i)
        for (size_t j = 0; j < numNodes; ++j) {
            unsigned int& p = path[i * numNodes + j];
            p = p == 0 ? static_cast<unsigned int>(i) : p - 1;
        }
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
    
    // Create the CUDA context before timing the computation.
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
