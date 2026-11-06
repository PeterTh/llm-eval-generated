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

// Tiles are kept in shared memory while their pivot block is processed.
// The input graph has nonnegative weights and a zero diagonal, so a pivot
// row and column do not change during their own relaxation step.
constexpr int TILE = 32;
constexpr int THREADS = 16;

#define CUDA_CHECK(call) do { \
    const cudaError_t error = (call); \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(error)); \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

__device__ __forceinline__ unsigned int loadDistance(const unsigned int* dist,
                                                       size_t n, int row, int col) {
    return static_cast<size_t>(row) < n && static_cast<size_t>(col) < n
        ? dist[static_cast<size_t>(row) * n + col] : INF;
}

__global__ void pivotTile(unsigned int* dist, size_t n, int pivot) {
    __shared__ unsigned int tile[TILE][TILE];
    const int x = threadIdx.x, y = threadIdx.y;
    const int base = pivot * TILE;
    #pragma unroll
    for (int dy = 0; dy < TILE; dy += THREADS) {
        #pragma unroll
        for (int dx = 0; dx < TILE; dx += THREADS)
            tile[y + dy][x + dx] = loadDistance(dist, n, base + y + dy, base + x + dx);
    }
    __syncthreads();
    for (int k = 0; k < TILE; ++k) {
        #pragma unroll
        for (int dy = 0; dy < TILE; dy += THREADS) {
            #pragma unroll
            for (int dx = 0; dx < TILE; dx += THREADS) {
                const int row = y + dy, col = x + dx;
                const unsigned int candidate = tile[row][k] + tile[k][col];
                if (candidate < tile[row][col]) tile[row][col] = candidate;
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int dy = 0; dy < TILE; dy += THREADS) {
        #pragma unroll
        for (int dx = 0; dx < TILE; dx += THREADS) {
            const int row = base + y + dy, col = base + x + dx;
            if (static_cast<size_t>(row) < n && static_cast<size_t>(col) < n)
                dist[static_cast<size_t>(row) * n + col] = tile[y + dy][x + dx];
        }
    }
}

// blockIdx.y selects the pivot row (0) or pivot column (1).
__global__ void pivotCross(unsigned int* dist, size_t n, int pivot) {
    __shared__ unsigned int diagonal[TILE][TILE];
    __shared__ unsigned int tile[TILE][TILE];
    const int x = threadIdx.x, y = threadIdx.y;
    const int other = blockIdx.x + (static_cast<int>(blockIdx.x) >= pivot);
    const bool isRow = blockIdx.y == 0;
    const int rowBase = (isRow ? pivot : other) * TILE;
    const int colBase = (isRow ? other : pivot) * TILE;
    const int pivotBase = pivot * TILE;
    #pragma unroll
    for (int dy = 0; dy < TILE; dy += THREADS) {
        #pragma unroll
        for (int dx = 0; dx < TILE; dx += THREADS) {
            diagonal[y + dy][x + dx] = loadDistance(dist, n, pivotBase + y + dy, pivotBase + x + dx);
            tile[y + dy][x + dx] = loadDistance(dist, n, rowBase + y + dy, colBase + x + dx);
        }
    }
    __syncthreads();
    for (int k = 0; k < TILE; ++k) {
        #pragma unroll
        for (int dy = 0; dy < TILE; dy += THREADS) {
            #pragma unroll
            for (int dx = 0; dx < TILE; dx += THREADS) {
                const int row = y + dy, col = x + dx;
                const unsigned int candidate = isRow
                    ? diagonal[row][k] + tile[k][col]
                    : tile[row][k] + diagonal[k][col];
                if (candidate < tile[row][col]) tile[row][col] = candidate;
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int dy = 0; dy < TILE; dy += THREADS) {
        #pragma unroll
        for (int dx = 0; dx < TILE; dx += THREADS) {
            const int row = rowBase + y + dy, col = colBase + x + dx;
            if (static_cast<size_t>(row) < n && static_cast<size_t>(col) < n)
                dist[static_cast<size_t>(row) * n + col] = tile[y + dy][x + dx];
        }
    }
}

__global__ void remainingTiles(unsigned int* dist, size_t n, int pivot) {
    __shared__ unsigned int left[TILE][TILE];
    __shared__ unsigned int top[TILE][TILE];
    const int x = threadIdx.x, y = threadIdx.y;
    const int tileCol = blockIdx.x + (static_cast<int>(blockIdx.x) >= pivot);
    const int tileRow = blockIdx.y + (static_cast<int>(blockIdx.y) >= pivot);
    const int rowBase = tileRow * TILE, colBase = tileCol * TILE, pivotBase = pivot * TILE;
    #pragma unroll
    for (int dy = 0; dy < TILE; dy += THREADS) {
        #pragma unroll
        for (int dx = 0; dx < TILE; dx += THREADS) {
            left[y + dy][x + dx] = loadDistance(dist, n, rowBase + y + dy, pivotBase + x + dx);
            top[y + dy][x + dx] = loadDistance(dist, n, pivotBase + y + dy, colBase + x + dx);
        }
    }
    unsigned int values[2][2];
    #pragma unroll
    for (int dy = 0; dy < 2; ++dy)
        #pragma unroll
        for (int dx = 0; dx < 2; ++dx)
            values[dy][dx] = loadDistance(dist, n, rowBase + y + dy * THREADS, colBase + x + dx * THREADS);
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < TILE; ++k) {
        #pragma unroll
        for (int dy = 0; dy < 2; ++dy) {
            const unsigned int a = left[y + dy * THREADS][k];
            #pragma unroll
            for (int dx = 0; dx < 2; ++dx) {
                const unsigned int candidate = a + top[k][x + dx * THREADS];
                values[dy][dx] = min(values[dy][dx], candidate);
            }
        }
    }
    #pragma unroll
    for (int dy = 0; dy < 2; ++dy) {
        #pragma unroll
        for (int dx = 0; dx < 2; ++dx) {
            const int row = rowBase + y + dy * THREADS, col = colBase + x + dx * THREADS;
            if (static_cast<size_t>(row) < n && static_cast<size_t>(col) < n)
                dist[static_cast<size_t>(row) * n + col] = values[dy][dx];
        }
    }
}

void floydWarshall(std::vector<unsigned int>& dist, const size_t numNodes) {
    if (numNodes == 0) return;
    const size_t bytes = dist.size() * sizeof(unsigned int);
    unsigned int* deviceDist = nullptr;
    CUDA_CHECK(cudaMalloc(&deviceDist, bytes));
    CUDA_CHECK(cudaMemcpy(deviceDist, dist.data(), bytes, cudaMemcpyHostToDevice));
    const int tiles = static_cast<int>((numNodes + TILE - 1) / TILE);
    const dim3 threads(THREADS, THREADS);
    for (int pivot = 0; pivot < tiles; ++pivot) {
        pivotTile<<<1, threads>>>(deviceDist, numNodes, pivot);
        if (tiles > 1) {
            pivotCross<<<dim3(tiles - 1, 2), threads>>>(deviceDist, numNodes, pivot);
            remainingTiles<<<dim3(tiles - 1, tiles - 1), threads>>>(deviceDist, numNodes, pivot);
        }
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(dist.data(), deviceDist, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(deviceDist));
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
    
    // Initialize the CUDA runtime before timing the computation.
    CUDA_CHECK(cudaFree(nullptr));

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
