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
constexpr int ROWS_PER_THREAD = 4;

void checkCuda(cudaError_t status) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

// A rank is one plus the greatest intermediate vertex on a path; zero means
// a direct edge. Minimizing (distance, rank) lexicographically preserves the
// original increasing-k algorithm's strict-improvement path choices even
// though blocking changes the order in which equal-length paths are found.
__device__ __forceinline__ void relax(unsigned int& distance, unsigned int& rank,
                                     unsigned int left, unsigned int right,
                                     unsigned int leftRank, unsigned int rightRank,
                                     unsigned int intermediate) {
    const unsigned int candidate = left + right;
    const unsigned int candidateRank = max(intermediate, max(leftRank, rightRank));
    if (candidate < distance || (candidate == distance && candidateRank < rank)) {
        distance = candidate;
        rank = candidateRank;
    }
}

// Phase 1 closes the pivot tile. Phase 2 closes every tile in its row/column.
// Each lane owns four cells, and adjacent lanes access adjacent columns.
template <bool Pivot>
__global__ void closeTiles(unsigned int* distances, unsigned int* ranks,
                           size_t n, int pivot) {
    __shared__ unsigned int d[TILE][TILE];
    __shared__ unsigned int p[TILE][TILE];
    __shared__ unsigned int diagonal[TILE][TILE];
    __shared__ unsigned int diagonalRank[TILE][TILE];
    const int x = threadIdx.x;
    const int y = threadIdx.y;
    int tileRow = pivot, tileCol = pivot;
    if (!Pivot) {
        const int other = blockIdx.x + (blockIdx.x >= static_cast<unsigned int>(pivot));
        if (blockIdx.y == 0) tileCol = other;
        else tileRow = other;
    }
    #pragma unroll
    for (int t = 0; t < ROWS_PER_THREAD; ++t) {
        const int row = y + t * (TILE / ROWS_PER_THREAD);
        const size_t i = static_cast<size_t>(tileRow) * TILE + row;
        const size_t j = static_cast<size_t>(tileCol) * TILE + x;
        d[row][x] = (i < n && j < n) ? distances[i * n + j] : INF;
        p[row][x] = (i < n && j < n) ? ranks[i * n + j] : 0;
        if (!Pivot) {
            const size_t pi = static_cast<size_t>(pivot) * TILE + row;
            const size_t pj = static_cast<size_t>(pivot) * TILE + x;
            diagonal[row][x] = (pi < n && pj < n) ? distances[pi * n + pj] : INF;
            diagonalRank[row][x] = (pi < n && pj < n) ? ranks[pi * n + pj] : 0;
        }
    }
    __syncthreads();
    const int count = static_cast<int>(min(static_cast<size_t>(TILE), n - static_cast<size_t>(pivot) * TILE));
    for (int k = 0; k < count; ++k) {
        unsigned int values[ROWS_PER_THREAD], paths[ROWS_PER_THREAD];
        #pragma unroll
        for (int t = 0; t < ROWS_PER_THREAD; ++t) {
            const int row = y + t * (TILE / ROWS_PER_THREAD);
            values[t] = d[row][x];
            paths[t] = p[row][x];
            if (Pivot) {
                relax(values[t], paths[t], d[row][k], d[k][x], p[row][k], p[k][x], pivot * TILE + k + 1);
            } else if (blockIdx.y == 0) {
                relax(values[t], paths[t], diagonal[row][k], d[k][x], diagonalRank[row][k], p[k][x], pivot * TILE + k + 1);
            } else {
                relax(values[t], paths[t], d[row][k], diagonal[k][x], p[row][k], diagonalRank[k][x], pivot * TILE + k + 1);
            }
        }
        // Finish all reads before overwriting the shared pivot row/column.
        __syncthreads();
        #pragma unroll
        for (int t = 0; t < ROWS_PER_THREAD; ++t) {
            const int row = y + t * (TILE / ROWS_PER_THREAD);
            d[row][x] = values[t];
            p[row][x] = paths[t];
        }
        __syncthreads();
    }
    #pragma unroll
    for (int t = 0; t < ROWS_PER_THREAD; ++t) {
        const size_t i = static_cast<size_t>(tileRow) * TILE + y + t * (TILE / ROWS_PER_THREAD);
        const size_t j = static_cast<size_t>(tileCol) * TILE + x;
        if (i < n && j < n) {
            distances[i * n + j] = d[y + t * (TILE / ROWS_PER_THREAD)][x];
            ranks[i * n + j] = p[y + t * (TILE / ROWS_PER_THREAD)][x];
        }
    }
}

// Phase 3 is the bulk of the work: independent min-plus tile products with
// read-only shared input tiles and four register-resident outputs per thread.
__global__ void updateTiles(unsigned int* distances, unsigned int* ranks,
                            size_t n, int pivot) {
    __shared__ unsigned int left[TILE][TILE], right[TILE][TILE];
    __shared__ unsigned int leftRank[TILE][TILE], rightRank[TILE][TILE];
    const int tileRow = blockIdx.y + (blockIdx.y >= static_cast<unsigned int>(pivot));
    const int tileCol = blockIdx.x + (blockIdx.x >= static_cast<unsigned int>(pivot));
    const int x = threadIdx.x, y = threadIdx.y;
    const size_t j = static_cast<size_t>(tileCol) * TILE + x;
    unsigned int values[ROWS_PER_THREAD], paths[ROWS_PER_THREAD];
    #pragma unroll
    for (int t = 0; t < ROWS_PER_THREAD; ++t) {
        const int row = y + t * (TILE / ROWS_PER_THREAD);
        const size_t i = static_cast<size_t>(tileRow) * TILE + row;
        const size_t pi = static_cast<size_t>(pivot) * TILE + row;
        const size_t pj = static_cast<size_t>(pivot) * TILE + x;
        left[row][x] = (i < n && pj < n) ? distances[i * n + pj] : INF;
        leftRank[row][x] = (i < n && pj < n) ? ranks[i * n + pj] : 0;
        right[row][x] = (pi < n && j < n) ? distances[pi * n + j] : INF;
        rightRank[row][x] = (pi < n && j < n) ? ranks[pi * n + j] : 0;
        values[t] = (i < n && j < n) ? distances[i * n + j] : INF;
        paths[t] = (i < n && j < n) ? ranks[i * n + j] : 0;
    }
    __syncthreads();
    const int count = static_cast<int>(min(static_cast<size_t>(TILE), n - static_cast<size_t>(pivot) * TILE));
    #pragma unroll
    for (int k = 0; k < TILE; ++k) {
        if (k < count) {
            const unsigned int r = right[k][x], rp = rightRank[k][x];
            #pragma unroll
            for (int t = 0; t < ROWS_PER_THREAD; ++t) {
                const int row = y + t * (TILE / ROWS_PER_THREAD);
                relax(values[t], paths[t], left[row][k], r, leftRank[row][k], rp, pivot * TILE + k + 1);
            }
        }
    }
    #pragma unroll
    for (int t = 0; t < ROWS_PER_THREAD; ++t) {
        const size_t i = static_cast<size_t>(tileRow) * TILE + y + t * (TILE / ROWS_PER_THREAD);
        if (i < n && j < n) {
            distances[i * n + j] = values[t];
            ranks[i * n + j] = paths[t];
        }
    }
}
} // namespace

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes) {
    // CUDA is required, including for empty inputs; there is no CPU fallback.
    checkCuda(cudaFree(nullptr));
    if (numNodes == 0) return;
    const size_t bytes = dist.size() * sizeof(unsigned int);
    unsigned int *deviceDist = nullptr, *deviceRank = nullptr;
    checkCuda(cudaMalloc(&deviceDist, bytes));
    checkCuda(cudaMalloc(&deviceRank, bytes));
    checkCuda(cudaMemcpy(deviceDist, dist.data(), bytes, cudaMemcpyHostToDevice));
    checkCuda(cudaMemset(deviceRank, 0, bytes));
    const int tiles = static_cast<int>((numNodes + TILE - 1) / TILE);
    const dim3 threads(TILE, TILE / ROWS_PER_THREAD);
    for (int pivot = 0; pivot < tiles; ++pivot) {
        closeTiles<true><<<1, threads>>>(deviceDist, deviceRank, numNodes, pivot);
        if (tiles > 1) {
            closeTiles<false><<<dim3(tiles - 1, 2), threads>>>(deviceDist, deviceRank, numNodes, pivot);
            updateTiles<<<dim3(tiles - 1, tiles - 1), threads>>>(deviceDist, deviceRank, numNodes, pivot);
        }
    }
    checkCuda(cudaGetLastError());
    checkCuda(cudaMemcpy(dist.data(), deviceDist, bytes, cudaMemcpyDeviceToHost));
    checkCuda(cudaMemcpy(path.data(), deviceRank, bytes, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < path.size(); ++i) {
        path[i] = path[i] == 0 ? static_cast<unsigned int>(i / numNodes) : path[i] - 1;
    }
    checkCuda(cudaFree(deviceRank));
    checkCuda(cudaFree(deviceDist));
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
    // Establish the CUDA context before timing the matrix computation.
    checkCuda(cudaFree(nullptr));
    auto start = std::chrono::high_resolution_clock::now();
    
    floydWarshall(dist, path, numNodes);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate operations per second
    // Floyd-Warshall has O(n³) complexity
    double ops = (double)numNodes * numNodes * numNodes;
    double seconds = std::chrono::duration<double>(end - start).count();
    double gflops = seconds > 0.0 ? ops / seconds / 1e9 : 0.0;
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
