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
constexpr int THREAD_ROWS = 4;
constexpr int ROWS_PER_THREAD = TILE / THREAD_ROWS;
using Entry = unsigned long long;

// The high word is the distance. The low word is the smallest possible
// maximum intermediate vertex (plus one) among paths with that distance.
// Lexicographic minimization preserves the original ascending-k, strict-<
// path updates even though blocked Floyd-Warshall visits vertices differently.
__device__ __forceinline__ Entry extend(Entry a, Entry b, unsigned int k) {
    const unsigned int distance = static_cast<unsigned int>(a >> 32) +
                                  static_cast<unsigned int>(b >> 32);
    const unsigned int rank = max(max(static_cast<unsigned int>(a),
                                     static_cast<unsigned int>(b)), k);
    return (static_cast<Entry>(distance) << 32) | rank;
}

__device__ __forceinline__ Entry better(Entry a, Entry b) {
    return a < b ? a : b;
}

void cudaCheck(cudaError_t status) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

__global__ void packMatrix(const unsigned int* input, Entry* matrix,
                           size_t n, size_t stride, size_t count) {
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<size_t>(blockDim.x) * gridDim.x) {
        const size_t row = index / stride;
        const size_t col = index % stride;
        const unsigned int d = row < n && col < n ? input[row * n + col]
                                                  : (row == col ? 0 : INF);
        matrix[index] = static_cast<Entry>(d) << 32;
    }
}

__global__ void unpackMatrix(const Entry* matrix, unsigned int* dist,
                             unsigned int* path, size_t n, size_t stride) {
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < n * n; index += static_cast<size_t>(blockDim.x) * gridDim.x) {
        const size_t row = index / n;
        const Entry value = matrix[row * stride + index % n];
        dist[index] = static_cast<unsigned int>(value >> 32);
        const unsigned int rank = static_cast<unsigned int>(value);
        path[index] = rank ? rank - 1 : static_cast<unsigned int>(row);
    }
}

// Phase 1: close the diagonal tile. At each k the pivot row and column
// cannot improve (zero diagonal and nonnegative weights), so leaving them
// untouched makes concurrent shared-memory reads and writes race-free.
__global__ void diagonalTile(Entry* matrix, size_t stride, unsigned int pivot) {
    __shared__ Entry tile[TILE][TILE];
    const int col = threadIdx.x;
    const int firstRow = threadIdx.y;
    const size_t base = static_cast<size_t>(pivot) * TILE;
    #pragma unroll
    for (int t = 0; t < ROWS_PER_THREAD; ++t) {
        const int row = firstRow + t * THREAD_ROWS;
        tile[row][col] = matrix[(base + row) * stride + base + col];
    }
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < TILE; ++k) {
        #pragma unroll
        for (int t = 0; t < ROWS_PER_THREAD; ++t) {
            const int row = firstRow + t * THREAD_ROWS;
            if (row != k && col != k) {
                tile[row][col] = better(tile[row][col],
                    extend(tile[row][k], tile[k][col], base + k + 1));
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int t = 0; t < ROWS_PER_THREAD; ++t) {
        const int row = firstRow + t * THREAD_ROWS;
        matrix[(base + row) * stride + base + col] = tile[row][col];
    }
}

// Phase 2: update all other tiles in the pivot row and column. Because the
// diagonal tile is already closed, each is a single min-plus product;
// the source tiles remain read-only and results stay in registers.
__global__ void perimeterTiles(Entry* matrix, size_t stride, unsigned int pivot) {
    __shared__ Entry diagonal[TILE][TILE];
    __shared__ Entry source[TILE][TILE];
    const unsigned int other = blockIdx.x + (blockIdx.x >= pivot);
    const bool inRow = blockIdx.y == 0;
    const size_t rowBase = static_cast<size_t>(inRow ? pivot : other) * TILE;
    const size_t colBase = static_cast<size_t>(inRow ? other : pivot) * TILE;
    const size_t pivotBase = static_cast<size_t>(pivot) * TILE;
    const int col = threadIdx.x;
    Entry result[ROWS_PER_THREAD];
    #pragma unroll
    for (int t = 0; t < ROWS_PER_THREAD; ++t) {
        const int row = threadIdx.y + t * THREAD_ROWS;
        diagonal[row][col] = matrix[(pivotBase + row) * stride + pivotBase + col];
        result[t] = source[row][col] = matrix[(rowBase + row) * stride + colBase + col];
    }
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < TILE; ++k) {
        #pragma unroll
        for (int t = 0; t < ROWS_PER_THREAD; ++t) {
            const int row = threadIdx.y + t * THREAD_ROWS;
            const Entry a = inRow ? diagonal[row][k] : source[row][k];
            const Entry b = inRow ? source[k][col] : diagonal[k][col];
            result[t] = better(result[t], extend(a, b, pivotBase + k + 1));
        }
    }
    #pragma unroll
    for (int t = 0; t < ROWS_PER_THREAD; ++t) {
        const int row = threadIdx.y + t * THREAD_ROWS;
        matrix[(rowBase + row) * stride + colBase + col] = result[t];
    }
}

// Phase 3: independent tiles reuse the two pivot tiles from shared memory.
// Warps load contiguous columns; each thread owns eight register-held cells.
__global__ void remainingTiles(Entry* matrix, size_t stride, unsigned int pivot) {
    __shared__ Entry left[TILE][TILE];
    __shared__ Entry right[TILE][TILE];
    const unsigned int tileRow = blockIdx.y + (blockIdx.y >= pivot);
    const unsigned int tileCol = blockIdx.x + (blockIdx.x >= pivot);
    const size_t rowBase = static_cast<size_t>(tileRow) * TILE;
    const size_t colBase = static_cast<size_t>(tileCol) * TILE;
    const size_t pivotBase = static_cast<size_t>(pivot) * TILE;
    const int col = threadIdx.x;
    Entry result[ROWS_PER_THREAD];
    #pragma unroll
    for (int t = 0; t < ROWS_PER_THREAD; ++t) {
        const int row = threadIdx.y + t * THREAD_ROWS;
        left[row][col] = matrix[(rowBase + row) * stride + pivotBase + col];
        right[row][col] = matrix[(pivotBase + row) * stride + colBase + col];
        result[t] = matrix[(rowBase + row) * stride + colBase + col];
    }
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < TILE; ++k) {
        const Entry b = right[k][col];
        #pragma unroll
        for (int t = 0; t < ROWS_PER_THREAD; ++t) {
            const int row = threadIdx.y + t * THREAD_ROWS;
            result[t] = better(result[t], extend(left[row][k], b, pivotBase + k + 1));
        }
    }
    #pragma unroll
    for (int t = 0; t < ROWS_PER_THREAD; ++t) {
        const int row = threadIdx.y + t * THREAD_ROWS;
        matrix[(rowBase + row) * stride + colBase + col] = result[t];
    }
}
} // namespace

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes) {
    // CUDA is mandatory; never fall back to the serial implementation.
    cudaCheck(cudaFree(nullptr));
    if (numNodes == 0) return;
    const size_t stride = ((numNodes + TILE - 1) / TILE) * TILE;
    const unsigned int tiles = static_cast<unsigned int>(stride / TILE);
    if (tiles > 65535 || stride > std::numeric_limits<size_t>::max() / stride / sizeof(Entry)) {
        fprintf(stderr, "Graph is too large for the CUDA matrix layout\n");
        std::exit(EXIT_FAILURE);
    }
    const size_t bytes = dist.size() * sizeof(unsigned int);
    Entry* matrix = nullptr;
    unsigned int* deviceDist = nullptr;
    unsigned int* devicePath = nullptr;
    cudaCheck(cudaMalloc(&matrix, stride * stride * sizeof(Entry)));
    cudaCheck(cudaMalloc(&deviceDist, bytes));
    cudaCheck(cudaMalloc(&devicePath, bytes));
    cudaCheck(cudaMemcpy(deviceDist, dist.data(), bytes, cudaMemcpyHostToDevice));
    const unsigned int packBlocks = static_cast<unsigned int>(std::min<size_t>(
        (stride * stride + 255) / 256, 65535));
    packMatrix<<<packBlocks, 256>>>(deviceDist, matrix, numNodes, stride, stride * stride);
    const dim3 threads(TILE, THREAD_ROWS);
    // Successive launches in the default stream enforce phase dependencies
    // without host synchronization between pivots.
    for (unsigned int pivot = 0; pivot < tiles; ++pivot) {
        diagonalTile<<<1, threads>>>(matrix, stride, pivot);
        if (tiles > 1) {
            perimeterTiles<<<dim3(tiles - 1, 2), threads>>>(matrix, stride, pivot);
            remainingTiles<<<dim3(tiles - 1, tiles - 1), threads>>>(matrix, stride, pivot);
        }
    }
    unpackMatrix<<<packBlocks, 256>>>(matrix, deviceDist, devicePath, numNodes, stride);
    cudaCheck(cudaGetLastError());
    // Blocking copies also wait for execution and surface asynchronous errors.
    cudaCheck(cudaMemcpy(dist.data(), deviceDist, bytes, cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(path.data(), devicePath, bytes, cudaMemcpyDeviceToHost));
    cudaCheck(cudaFree(devicePath));
    cudaCheck(cudaFree(deviceDist));
    cudaCheck(cudaFree(matrix));
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
    
    if (numNodes > std::numeric_limits<unsigned int>::max() - TILE ||
        (numNodes != 0 && numNodes > std::numeric_limits<size_t>::max() / numNodes / sizeof(unsigned int))) {
        fprintf(stderr, "Graph is too large\n");
        return 1;
    }

    // Allocate matrices
    std::vector<unsigned int> dist(numNodes * numNodes);
    std::vector<unsigned int> path(numNodes * numNodes);
    
    // Initialize
    printf("Initializing graph...\n");
    initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
    initializePathMatrix(path, numNodes);
    
    // Load the CUDA context and kernels before timing, including any PTX JIT.
    cudaCheck(cudaFree(nullptr));
    cudaFuncAttributes attributes;
    cudaCheck(cudaFuncGetAttributes(&attributes, remainingTiles));

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
