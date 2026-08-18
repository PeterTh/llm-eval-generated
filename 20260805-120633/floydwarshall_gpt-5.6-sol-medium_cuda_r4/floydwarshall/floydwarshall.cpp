#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;
constexpr int TILE_SIZE = 32;
constexpr int THREAD_ROWS = 8;

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

[[noreturn]] void cudaFail(const cudaError_t error, const char* operation) {
    std::fprintf(stderr, "CUDA error in %s: %s\n", operation, cudaGetErrorString(error));
    std::exit(EXIT_FAILURE);
}

inline void cudaCheck(const cudaError_t error, const char* operation) {
    if (error != cudaSuccess) cudaFail(error, operation);
}

// Each thread owns four vertically separated cells.  This gives a full 32x32
// tile while using only 256 threads, which leaves enough registers for high
// occupancy and makes all global-memory accesses coalesced.
__global__ __launch_bounds__(256) void pivotKernel(unsigned int* __restrict__ dist,
                                                    unsigned int* __restrict__ path,
                                                    const int n, const int round) {
    __shared__ unsigned int tile[TILE_SIZE][TILE_SIZE + 1];
    const int x = threadIdx.x;
    const int firstY = threadIdx.y;
    const int base = round * TILE_SIZE;

#pragma unroll
    for (int q = 0; q < TILE_SIZE / THREAD_ROWS; ++q) {
        const int y = firstY + q * THREAD_ROWS;
        const int row = base + y;
        const int col = base + x;
        tile[y][x] = (row < n && col < n) ? dist[static_cast<size_t>(row) * n + col] : INF;
    }
    __syncthreads();

#pragma unroll 1
    for (int k = 0; k < TILE_SIZE && base + k < n; ++k) {
#pragma unroll
        for (int q = 0; q < TILE_SIZE / THREAD_ROWS; ++q) {
            const int y = firstY + q * THREAD_ROWS;
            const unsigned int candidate = tile[y][k] + tile[k][x];
            if (candidate < tile[y][x]) {
                tile[y][x] = candidate;
                const int row = base + y;
                const int col = base + x;
                if (row < n && col < n)
                    path[static_cast<size_t>(row) * n + col] = base + k;
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int q = 0; q < TILE_SIZE / THREAD_ROWS; ++q) {
        const int y = firstY + q * THREAD_ROWS;
        const int row = base + y;
        const int col = base + x;
        if (row < n && col < n) dist[static_cast<size_t>(row) * n + col] = tile[y][x];
    }
}

__global__ __launch_bounds__(256) void rowColumnKernel(unsigned int* __restrict__ dist,
                                                        unsigned int* __restrict__ path,
                                                        const int n, const int round,
                                                        const int rounds) {
    __shared__ unsigned int pivot[TILE_SIZE][TILE_SIZE + 1];
    __shared__ unsigned int tile[TILE_SIZE][TILE_SIZE + 1];
    const int x = threadIdx.x;
    const int firstY = threadIdx.y;
    const int compact = blockIdx.x;
    const int other = compact >= round ? compact + 1 : compact;
    if (other >= rounds) return;

    const bool columnTile = blockIdx.y != 0;
    const int tileRow = columnTile ? other : round;
    const int tileCol = columnTile ? round : other;
    const int pivotBase = round * TILE_SIZE;

#pragma unroll
    for (int q = 0; q < TILE_SIZE / THREAD_ROWS; ++q) {
        const int y = firstY + q * THREAD_ROWS;
        const int pr = pivotBase + y;
        const int pc = pivotBase + x;
        pivot[y][x] = (pr < n && pc < n) ? dist[static_cast<size_t>(pr) * n + pc] : INF;
        const int row = tileRow * TILE_SIZE + y;
        const int col = tileCol * TILE_SIZE + x;
        tile[y][x] = (row < n && col < n) ? dist[static_cast<size_t>(row) * n + col] : INF;
    }
    __syncthreads();

#pragma unroll 1
    for (int k = 0; k < TILE_SIZE && pivotBase + k < n; ++k) {
#pragma unroll
        for (int q = 0; q < TILE_SIZE / THREAD_ROWS; ++q) {
            const int y = firstY + q * THREAD_ROWS;
            const unsigned int candidate = columnTile
                ? tile[y][k] + pivot[k][x]
                : pivot[y][k] + tile[k][x];
            if (candidate < tile[y][x]) {
                tile[y][x] = candidate;
                const int row = tileRow * TILE_SIZE + y;
                const int col = tileCol * TILE_SIZE + x;
                if (row < n && col < n)
                    path[static_cast<size_t>(row) * n + col] = pivotBase + k;
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int q = 0; q < TILE_SIZE / THREAD_ROWS; ++q) {
        const int y = firstY + q * THREAD_ROWS;
        const int row = tileRow * TILE_SIZE + y;
        const int col = tileCol * TILE_SIZE + x;
        if (row < n && col < n) dist[static_cast<size_t>(row) * n + col] = tile[y][x];
    }
}

__global__ __launch_bounds__(256) void remainderKernel(unsigned int* __restrict__ dist,
                                                        unsigned int* __restrict__ path,
                                                        const int n, const int round,
                                                        const int rounds) {
    __shared__ unsigned int left[TILE_SIZE][TILE_SIZE + 1];
    __shared__ unsigned int top[TILE_SIZE][TILE_SIZE + 1];
    const int compactCol = blockIdx.x;
    const int compactRow = blockIdx.y;
    const int tileCol = compactCol >= round ? compactCol + 1 : compactCol;
    const int tileRow = compactRow >= round ? compactRow + 1 : compactRow;
    if (tileRow >= rounds || tileCol >= rounds) return;

    const int x = threadIdx.x;
    const int firstY = threadIdx.y;
    const int pivotBase = round * TILE_SIZE;
#pragma unroll
    for (int q = 0; q < TILE_SIZE / THREAD_ROWS; ++q) {
        const int y = firstY + q * THREAD_ROWS;
        const int row = tileRow * TILE_SIZE + y;
        const int col = tileCol * TILE_SIZE + x;
        const int pivotRow = pivotBase + y;
        const int pivotCol = pivotBase + x;
        left[y][x] = (row < n && pivotCol < n)
            ? dist[static_cast<size_t>(row) * n + pivotCol] : INF;
        top[y][x] = (pivotRow < n && col < n)
            ? dist[static_cast<size_t>(pivotRow) * n + col] : INF;
    }
    __syncthreads();

#pragma unroll
    for (int q = 0; q < TILE_SIZE / THREAD_ROWS; ++q) {
        const int y = firstY + q * THREAD_ROWS;
        const int row = tileRow * TILE_SIZE + y;
        const int col = tileCol * TILE_SIZE + x;
        if (row >= n || col >= n) continue;
        const size_t index = static_cast<size_t>(row) * n + col;
        unsigned int best = dist[index];
        int bestK = -1;
#pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            const unsigned int candidate = left[y][k] + top[k][x];
            if (candidate < best) {
                best = candidate;
                bestK = pivotBase + k;
            }
        }
        dist[index] = best;
        if (bestK >= 0) path[index] = static_cast<unsigned int>(bestK);
    }
}

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes) {
    if (numNodes == 0) return;
    if (numNodes > static_cast<size_t>(std::numeric_limits<int>::max())) {
        std::fprintf(stderr, "Number of nodes exceeds the CUDA kernel index range\n");
        std::exit(EXIT_FAILURE);
    }
    const size_t bytes = dist.size() * sizeof(unsigned int);
    unsigned int* deviceDist = nullptr;
    unsigned int* devicePath = nullptr;
    cudaCheck(cudaMalloc(&deviceDist, bytes), "cudaMalloc(distance matrix)");
    cudaCheck(cudaMalloc(&devicePath, bytes), "cudaMalloc(path matrix)");
    cudaCheck(cudaMemcpy(deviceDist, dist.data(), bytes, cudaMemcpyHostToDevice),
              "copy distance matrix to GPU");
    cudaCheck(cudaMemcpy(devicePath, path.data(), bytes, cudaMemcpyHostToDevice),
              "copy path matrix to GPU");

    const int n = static_cast<int>(numNodes);
    const int rounds = (n + TILE_SIZE - 1) / TILE_SIZE;
    const dim3 threads(TILE_SIZE, THREAD_ROWS);
    for (int round = 0; round < rounds; ++round) {
        pivotKernel<<<1, threads>>>(deviceDist, devicePath, n, round);
        if (rounds > 1) {
            rowColumnKernel<<<dim3(rounds - 1, 2), threads>>>(deviceDist, devicePath,
                                                              n, round, rounds);
            remainderKernel<<<dim3(rounds - 1, rounds - 1), threads>>>(deviceDist, devicePath,
                                                                       n, round, rounds);
        }
    }
    cudaCheck(cudaGetLastError(), "launch Floyd-Warshall kernels");
    cudaCheck(cudaMemcpy(dist.data(), deviceDist, bytes, cudaMemcpyDeviceToHost),
              "copy distance matrix from GPU");
    cudaCheck(cudaMemcpy(path.data(), devicePath, bytes, cudaMemcpyDeviceToHost),
              "copy path matrix from GPU");
    cudaCheck(cudaFree(deviceDist), "cudaFree(distance matrix)");
    cudaCheck(cudaFree(devicePath), "cudaFree(path matrix)");
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

    // Create the CUDA context before the benchmark interval.  Context startup
    // is a one-time runtime cost and would otherwise dominate smaller graphs.
    cudaCheck(cudaFree(nullptr), "initialize CUDA context");
    
    // Run Floyd-Warshall
    printf("Computing shortest paths...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    floydWarshall(dist, path, numNodes);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    const double elapsedSeconds = std::chrono::duration<double>(end - start).count();
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate operations per second
    // Floyd-Warshall has O(n³) complexity
    double ops = (double)numNodes * numNodes * numNodes;
    double gflops = elapsedSeconds > 0.0 ? ops / elapsedSeconds / 1e9 : 0.0;
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
