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
constexpr int ROWS = 4;

void cudaCheck(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA %s failed: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

// Rank is one plus the greatest intermediate vertex on a path (zero for
// a direct edge). Minimize (distance, rank) lexicographically. This makes
// blocked evaluation reproduce the original increasing-k, strict-improvement
// path matrix, including equal-distance ties, despite reordering relaxations.
__device__ __forceinline__ void relax(unsigned int& distance, unsigned int& rank,
                                     unsigned int candidate, unsigned int candidateRank) {
    if (candidate < distance || (candidate == distance && candidateRank < rank)) {
        distance = candidate;
        rank = candidateRank;
    }
}

// Phase 1 closes the pivot tile; phase 2 closes its row and column tiles.
// Four cells per thread give coalesced memory accesses with 256-thread blocks.
template<bool Pivot>
__global__ void closeTiles(unsigned int* dist, unsigned int* ranks, size_t n,
                           size_t pivot) {
    __shared__ unsigned int d[2][TILE][TILE];
    __shared__ unsigned int r[2][TILE][TILE];
    const int x = threadIdx.x;
    const int y = threadIdx.y;
    size_t other = blockIdx.x;
    if (!Pivot && other >= pivot) ++other;
    const bool row = blockIdx.y == 0;
    const size_t tileRow = Pivot || row ? pivot : other;
    const size_t tileCol = Pivot || !row ? pivot : other;
    #pragma unroll
    for (int t = 0; t < ROWS; ++t) {
        const int yy = y + t * 8;
        const size_t i = tileRow * TILE + yy, j = tileCol * TILE + x;
        d[0][yy][x] = i < n && j < n ? dist[i * n + j] : INF;
        r[0][yy][x] = i < n && j < n ? ranks[i * n + j] : 0;
        if (!Pivot) {
            const size_t pi = pivot * TILE + yy, pj = pivot * TILE + x;
            d[1][yy][x] = pi < n && pj < n ? dist[pi * n + pj] : INF;
            r[1][yy][x] = pi < n && pj < n ? ranks[pi * n + pj] : 0;
        }
    }
    __syncthreads();
    const int count = static_cast<int>(n - pivot * TILE < TILE ? n - pivot * TILE : TILE);
    for (int k = 0; k < count; ++k) {
        unsigned int value[ROWS], rank[ROWS];
        #pragma unroll
        for (int t = 0; t < ROWS; ++t) {
            const int yy = y + t * 8;
            const int left = !Pivot && row ? 1 : 0;
            const int right = !Pivot && !row ? 1 : 0;
            value[t] = d[0][yy][x];
            rank[t] = r[0][yy][x];
            relax(value[t], rank[t], d[left][yy][k] + d[right][k][x],
                  max(static_cast<unsigned int>(pivot * TILE + k + 1),
                      max(r[left][yy][k], r[right][k][x])));
        }
        // All operand reads must finish before any thread overwrites the tile.
        __syncthreads();
        #pragma unroll
        for (int t = 0; t < ROWS; ++t) {
            d[0][y + t * 8][x] = value[t];
            r[0][y + t * 8][x] = rank[t];
        }
        __syncthreads();
    }
    #pragma unroll
    for (int t = 0; t < ROWS; ++t) {
        const int yy = y + t * 8;
        const size_t i = tileRow * TILE + yy, j = tileCol * TILE + x;
        if (i < n && j < n) {
            dist[i * n + j] = d[0][yy][x];
            ranks[i * n + j] = r[0][yy][x];
        }
    }
}

// Phase 3: independent outer tiles reuse the closed pivot row/column in
// shared memory, keeping all four output distances and ranks in registers.
__global__ void updateTiles(unsigned int* dist, unsigned int* ranks, size_t n,
                            size_t pivot, size_t tiles) {
    __shared__ unsigned int left[TILE][TILE], right[TILE][TILE];
    __shared__ unsigned int leftRank[TILE][TILE], rightRank[TILE][TILE];
    const int x = threadIdx.x, y = threadIdx.y;
    size_t tileRow = static_cast<size_t>(blockIdx.x) / (tiles - 1);
    size_t tileCol = static_cast<size_t>(blockIdx.x) % (tiles - 1);
    if (tileRow >= pivot) ++tileRow;
    if (tileCol >= pivot) ++tileCol;
    unsigned int value[ROWS], rank[ROWS];
    #pragma unroll
    for (int t = 0; t < ROWS; ++t) {
        const int yy = y + t * 8;
        const size_t i = tileRow * TILE + yy, j = tileCol * TILE + x;
        const size_t pi = pivot * TILE + yy, pj = pivot * TILE + x;
        left[yy][x] = i < n && pj < n ? dist[i * n + pj] : INF;
        leftRank[yy][x] = i < n && pj < n ? ranks[i * n + pj] : 0;
        right[yy][x] = pi < n && j < n ? dist[pi * n + j] : INF;
        rightRank[yy][x] = pi < n && j < n ? ranks[pi * n + j] : 0;
        value[t] = i < n && j < n ? dist[i * n + j] : INF;
        rank[t] = i < n && j < n ? ranks[i * n + j] : 0;
    }
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < TILE; ++k) {
        #pragma unroll
        for (int t = 0; t < ROWS; ++t) {
            const int yy = y + t * 8;
            relax(value[t], rank[t], left[yy][k] + right[k][x],
                  max(static_cast<unsigned int>(pivot * TILE + k + 1),
                      max(leftRank[yy][k], rightRank[k][x])));
        }
    }
    #pragma unroll
    for (int t = 0; t < ROWS; ++t) {
        const size_t i = tileRow * TILE + y + t * 8, j = tileCol * TILE + x;
        if (i < n && j < n) {
            dist[i * n + j] = value[t];
            ranks[i * n + j] = rank[t];
        }
    }
}
} // namespace

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes) {
    // CUDA is required even for an empty graph; there is no CPU fallback.
    cudaCheck(cudaFree(nullptr), "initialization");
    if (numNodes == 0) return;
    const size_t bytes = dist.size() * sizeof(unsigned int);
    const size_t tiles = (numNodes + TILE - 1) / TILE;
    unsigned int *deviceDist = nullptr, *deviceRank = nullptr;
    cudaCheck(cudaMalloc(&deviceDist, bytes), "distance allocation");
    cudaCheck(cudaMalloc(&deviceRank, bytes), "rank allocation");
    cudaCheck(cudaMemcpy(deviceDist, dist.data(), bytes, cudaMemcpyHostToDevice), "upload");
    cudaCheck(cudaMemset(deviceRank, 0, bytes), "rank initialization");
    const dim3 threads(TILE, 8);
    for (size_t pivot = 0; pivot < tiles; ++pivot) {
        closeTiles<true><<<1, threads>>>(deviceDist, deviceRank, numNodes, pivot);
        cudaCheck(cudaGetLastError(), "pivot launch");
        if (tiles > 1) {
            closeTiles<false><<<dim3(static_cast<unsigned int>(tiles - 1), 2), threads>>>(
                deviceDist, deviceRank, numNodes, pivot);
            cudaCheck(cudaGetLastError(), "pivot row/column launch");
            updateTiles<<<static_cast<unsigned int>((tiles - 1) * (tiles - 1)), threads>>>(
                deviceDist, deviceRank, numNodes, pivot, tiles);
            cudaCheck(cudaGetLastError(), "outer tile launch");
        }
    }
    // Blocking copies also include all GPU execution in the benchmark timer.
    cudaCheck(cudaMemcpy(dist.data(), deviceDist, bytes, cudaMemcpyDeviceToHost), "distance download");
    std::vector<unsigned int> ranks(dist.size());
    cudaCheck(cudaMemcpy(ranks.data(), deviceRank, bytes, cudaMemcpyDeviceToHost), "rank download");
    for (size_t i = 0; i < ranks.size(); ++i) {
        if (ranks[i] != 0) path[i] = ranks[i] - 1;
    }
    cudaCheck(cudaFree(deviceRank), "rank release");
    cudaCheck(cudaFree(deviceDist), "distance release");
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
