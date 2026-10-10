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

namespace {
constexpr int TILE = 32;
constexpr int ROWS_PER_THREAD = 4;

void cudaCheck(cudaError_t error) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}

// A rank is one plus the largest internal vertex on a path; zero denotes
// an original edge. Among equal distances, minimize this rank. This makes
// blocked execution reproduce the original ascending-k, strict-update path
// matrix: its final k is the first k at which the final distance was reached.
struct Tile {
    unsigned int distance[TILE][TILE];
    unsigned int rank[TILE][TILE];
};

__device__ __forceinline__ void loadTile(Tile& tile, const unsigned int* dist,
                                        const unsigned int* rank, size_t n,
                                        size_t blockRow, size_t blockCol) {
    const int col = threadIdx.x;
    #pragma unroll
    for (int t = 0; t < ROWS_PER_THREAD; ++t) {
        const int row = threadIdx.y + t * blockDim.y;
        const size_t i = blockRow * TILE + row;
        const size_t j = blockCol * TILE + col;
        const bool valid = i < n && j < n;
        tile.distance[row][col] = valid ? dist[i * n + j] : INF;
        tile.rank[row][col] = valid ? rank[i * n + j] : 0;
    }
}

__device__ __forceinline__ void storeTile(const Tile& tile, unsigned int* dist,
                                         unsigned int* rank, size_t n,
                                         size_t blockRow, size_t blockCol) {
    const int col = threadIdx.x;
    #pragma unroll
    for (int t = 0; t < ROWS_PER_THREAD; ++t) {
        const int row = threadIdx.y + t * blockDim.y;
        const size_t i = blockRow * TILE + row;
        const size_t j = blockCol * TILE + col;
        if (i < n && j < n) {
            dist[i * n + j] = tile.distance[row][col];
            rank[i * n + j] = tile.rank[row][col];
        }
    }
}

__device__ __forceinline__ void relax(unsigned int candidate, unsigned int candidateRank,
                                     unsigned int& distance, unsigned int& rank) {
    if (candidate < distance || (candidate == distance && candidateRank < rank)) {
        distance = candidate;
        rank = candidateRank;
    }
}

__global__ void closePivot(unsigned int* dist, unsigned int* rank, size_t n, int pivot) {
    __shared__ Tile tile;
    loadTile(tile, dist, rank, n, pivot, pivot);
    __syncthreads();
    const int col = threadIdx.x;
    for (int k = 0; k < TILE; ++k) {
        unsigned int d[ROWS_PER_THREAD], r[ROWS_PER_THREAD];
        #pragma unroll
        for (int t = 0; t < ROWS_PER_THREAD; ++t) {
            const int row = threadIdx.y + t * blockDim.y;
            d[t] = tile.distance[row][col];
            r[t] = tile.rank[row][col];
            relax(tile.distance[row][k] + tile.distance[k][col],
                  max(static_cast<unsigned int>(pivot * TILE + k + 1),
                      max(tile.rank[row][k], tile.rank[k][col])), d[t], r[t]);
        }
        // Snapshot operands before any thread changes shared memory.
        __syncthreads();
        #pragma unroll
        for (int t = 0; t < ROWS_PER_THREAD; ++t) {
            const int row = threadIdx.y + t * blockDim.y;
            tile.distance[row][col] = d[t];
            tile.rank[row][col] = r[t];
        }
        __syncthreads();
    }
    storeTile(tile, dist, rank, n, pivot, pivot);
}

__global__ void closeCross(unsigned int* dist, unsigned int* rank, size_t n, int pivot) {
    __shared__ Tile diagonal;
    __shared__ Tile tile;
    const bool isRow = blockIdx.y == 0;
    const int other = blockIdx.x + (blockIdx.x >= static_cast<unsigned int>(pivot));
    const int blockRow = isRow ? pivot : other;
    const int blockCol = isRow ? other : pivot;
    loadTile(diagonal, dist, rank, n, pivot, pivot);
    loadTile(tile, dist, rank, n, blockRow, blockCol);
    __syncthreads();
    const int col = threadIdx.x;
    for (int k = 0; k < TILE; ++k) {
        unsigned int d[ROWS_PER_THREAD], r[ROWS_PER_THREAD];
        #pragma unroll
        for (int t = 0; t < ROWS_PER_THREAD; ++t) {
            const int row = threadIdx.y + t * blockDim.y;
            d[t] = tile.distance[row][col];
            r[t] = tile.rank[row][col];
            const unsigned int left = isRow ? diagonal.distance[row][k] : tile.distance[row][k];
            const unsigned int right = isRow ? tile.distance[k][col] : diagonal.distance[k][col];
            const unsigned int leftRank = isRow ? diagonal.rank[row][k] : tile.rank[row][k];
            const unsigned int rightRank = isRow ? tile.rank[k][col] : diagonal.rank[k][col];
            relax(left + right, max(static_cast<unsigned int>(pivot * TILE + k + 1),
                                   max(leftRank, rightRank)), d[t], r[t]);
        }
        __syncthreads();
        #pragma unroll
        for (int t = 0; t < ROWS_PER_THREAD; ++t) {
            const int row = threadIdx.y + t * blockDim.y;
            tile.distance[row][col] = d[t];
            tile.rank[row][col] = r[t];
        }
        __syncthreads();
    }
    storeTile(tile, dist, rank, n, blockRow, blockCol);
}

__global__ void closeRemaining(unsigned int* __restrict__ dist,
                               unsigned int* __restrict__ rank, size_t n, int pivot) {
    __shared__ Tile left;
    __shared__ Tile right;
    const int blockRow = blockIdx.y + (blockIdx.y >= static_cast<unsigned int>(pivot));
    const int blockCol = blockIdx.x + (blockIdx.x >= static_cast<unsigned int>(pivot));
    loadTile(left, dist, rank, n, blockRow, pivot);
    loadTile(right, dist, rank, n, pivot, blockCol);
    const int col = threadIdx.x;
    unsigned int d[ROWS_PER_THREAD], r[ROWS_PER_THREAD];
    #pragma unroll
    for (int t = 0; t < ROWS_PER_THREAD; ++t) {
        const size_t i = static_cast<size_t>(blockRow) * TILE + threadIdx.y + t * blockDim.y;
        const size_t j = static_cast<size_t>(blockCol) * TILE + col;
        const bool valid = i < n && j < n;
        d[t] = valid ? dist[i * n + j] : INF;
        r[t] = valid ? rank[i * n + j] : 0;
    }
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < TILE; ++k) {
        const unsigned int rightDistance = right.distance[k][col];
        const unsigned int rightRank = max(right.rank[k][col],
                                          static_cast<unsigned int>(pivot * TILE + k + 1));
        #pragma unroll
        for (int t = 0; t < ROWS_PER_THREAD; ++t) {
            const int row = threadIdx.y + t * blockDim.y;
            relax(left.distance[row][k] + rightDistance,
                  max(left.rank[row][k], rightRank), d[t], r[t]);
        }
    }
    #pragma unroll
    for (int t = 0; t < ROWS_PER_THREAD; ++t) {
        const size_t i = static_cast<size_t>(blockRow) * TILE + threadIdx.y + t * blockDim.y;
        const size_t j = static_cast<size_t>(blockCol) * TILE + col;
        if (i < n && j < n) {
            dist[i * n + j] = d[t];
            rank[i * n + j] = r[t];
        }
    }
}

__global__ void restorePath(unsigned int* rank, size_t n, size_t count) {
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const unsigned int value = rank[index];
        rank[index] = value == 0 ? static_cast<unsigned int>(index / n) : value - 1;
    }
}
} // namespace

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes) {
    if (numNodes == 0) return;
    const size_t bytes = dist.size() * sizeof(unsigned int);
    unsigned int* deviceDist = nullptr;
    unsigned int* deviceRank = nullptr;
    cudaCheck(cudaMalloc(&deviceDist, bytes));
    cudaCheck(cudaMalloc(&deviceRank, bytes));
    cudaCheck(cudaMemcpy(deviceDist, dist.data(), bytes, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemset(deviceRank, 0, bytes));
    const int tiles = static_cast<int>((numNodes + TILE - 1) / TILE);
    const dim3 threads(TILE, TILE / ROWS_PER_THREAD);
    for (int pivot = 0; pivot < tiles; ++pivot) {
        closePivot<<<1, threads>>>(deviceDist, deviceRank, numNodes, pivot);
        if (tiles > 1) {
            closeCross<<<dim3(tiles - 1, 2), threads>>>(deviceDist, deviceRank, numNodes, pivot);
            closeRemaining<<<dim3(tiles - 1, tiles - 1), threads>>>(deviceDist, deviceRank, numNodes, pivot);
        }
    }
    restorePath<<<std::min<size_t>((dist.size() + 255) / 256, 65535), 256>>>(
        deviceRank, numNodes, dist.size());
    cudaCheck(cudaGetLastError());
    // Blocking copies both synchronize computation and surface execution errors.
    cudaCheck(cudaMemcpy(dist.data(), deviceDist, bytes, cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(path.data(), deviceRank, bytes, cudaMemcpyDeviceToHost));
    cudaCheck(cudaFree(deviceRank));
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
            char* end = nullptr;
            const char* value = argv[++i];
            const unsigned long long parsed = std::strtoull(value, &end, 10);
            if (value[0] == '-' || end == value || *end != '\0' ||
                parsed > 65535ULL * TILE ||
                parsed > std::numeric_limits<size_t>::max() / sizeof(unsigned int) /
                             std::max(1ULL, parsed)) {
                fprintf(stderr, "Invalid number of nodes: %s\n", value);
                return 1;
            }
            numNodes = static_cast<size_t>(parsed);
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
    
    // Initialize the CUDA context before timing the algorithm. No CPU fallback.
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
    const double seconds = std::chrono::duration<double>(end - start).count();
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
