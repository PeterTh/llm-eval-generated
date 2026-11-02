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

// A pivot tile is closed first, followed by its row and column, and finally
// all remaining tiles.  All tiles are padded so every thread can participate
// in the synchronization without bounds checks.
constexpr int TILE = 32;
constexpr int THREAD_ROWS = 8;

#define CUDA_CHECK(call) do { \
    cudaError_t error = (call); \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(error)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

__global__ void closePivot(unsigned int* dist, size_t stride, int pivot) {
    __shared__ unsigned int tile[TILE][TILE];
    const int x = threadIdx.x, y = threadIdx.y;
    const size_t base = static_cast<size_t>(pivot) * TILE;
    #pragma unroll
    for (int r = 0; r < TILE; r += THREAD_ROWS)
        tile[y + r][x] = dist[(base + y + r) * stride + base + x];
    __syncthreads();
    for (int k = 0; k < TILE; ++k) {
        #pragma unroll
        for (int r = 0; r < TILE; r += THREAD_ROWS) {
            int row = y + r;
            // The pivot row and column cannot improve when the diagonal is 0.
            if (row != k && x != k) {
                unsigned int candidate = tile[row][k] + tile[k][x];
                tile[row][x] = min(tile[row][x], candidate);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int r = 0; r < TILE; r += THREAD_ROWS)
        dist[(base + y + r) * stride + base + x] = tile[y + r][x];
}

__global__ void updatePivotEdges(unsigned int* dist, size_t stride, int pivot) {
    __shared__ unsigned int center[TILE][TILE];
    __shared__ unsigned int edge[TILE][TILE];
    const int x = threadIdx.x, y = threadIdx.y;
    const int tileIndex = blockIdx.x + (blockIdx.x >= pivot);
    const bool isRow = blockIdx.y == 0;
    const size_t pivotBase = static_cast<size_t>(pivot) * TILE;
    const size_t rowBase = static_cast<size_t>(isRow ? pivot : tileIndex) * TILE;
    const size_t colBase = static_cast<size_t>(isRow ? tileIndex : pivot) * TILE;
    #pragma unroll
    for (int r = 0; r < TILE; r += THREAD_ROWS) {
        center[y + r][x] = dist[(pivotBase + y + r) * stride + pivotBase + x];
        edge[y + r][x] = dist[(rowBase + y + r) * stride + colBase + x];
    }
    __syncthreads();
    for (int k = 0; k < TILE; ++k) {
        #pragma unroll
        for (int r = 0; r < TILE; r += THREAD_ROWS) {
            int row = y + r;
            if ((isRow && row != k) || (!isRow && x != k)) {
                unsigned int candidate = isRow ? center[row][k] + edge[k][x]
                                               : edge[row][k] + center[k][x];
                edge[row][x] = min(edge[row][x], candidate);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int r = 0; r < TILE; r += THREAD_ROWS)
        dist[(rowBase + y + r) * stride + colBase + x] = edge[y + r][x];
}

__global__ void updateRemaining(unsigned int* dist, size_t stride, int pivot) {
    __shared__ unsigned int left[TILE][TILE];
    __shared__ unsigned int top[TILE][TILE];
    const int x = threadIdx.x, y = threadIdx.y;
    const int tileCol = blockIdx.x + (blockIdx.x >= pivot);
    const int tileRow = blockIdx.y + (blockIdx.y >= pivot);
    const size_t rowBase = static_cast<size_t>(tileRow) * TILE;
    const size_t colBase = static_cast<size_t>(tileCol) * TILE;
    const size_t pivotBase = static_cast<size_t>(pivot) * TILE;
    unsigned int value[TILE / THREAD_ROWS];
    #pragma unroll
    for (int r = 0; r < TILE; r += THREAD_ROWS) {
        left[y + r][x] = dist[(rowBase + y + r) * stride + pivotBase + x];
        top[y + r][x] = dist[(pivotBase + y + r) * stride + colBase + x];
        value[r / THREAD_ROWS] = dist[(rowBase + y + r) * stride + colBase + x];
    }
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < TILE; ++k) {
        #pragma unroll
        for (int r = 0; r < TILE; r += THREAD_ROWS)
            value[r / THREAD_ROWS] = min(value[r / THREAD_ROWS],
                                        left[y + r][k] + top[k][x]);
    }
    #pragma unroll
    for (int r = 0; r < TILE; r += THREAD_ROWS)
        dist[(rowBase + y + r) * stride + colBase + x] = value[r / THREAD_ROWS];
}

void floydWarshall(std::vector<unsigned int>& dist, const size_t numNodes) {
    if (numNodes == 0) return;
    const size_t stride = ((numNodes + TILE - 1) / TILE) * TILE;
    const int tiles = static_cast<int>(stride / TILE);
    std::vector<unsigned int> padded(stride * stride, INF);
    for (size_t i = 0; i < numNodes; ++i)
        std::copy_n(dist.data() + i * numNodes, numNodes, padded.data() + i * stride);
    for (size_t i = numNodes; i < stride; ++i) padded[i * stride + i] = 0;

    unsigned int* deviceDist = nullptr;
    CUDA_CHECK(cudaMalloc(&deviceDist, padded.size() * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpy(deviceDist, padded.data(), padded.size() * sizeof(unsigned int),
                          cudaMemcpyHostToDevice));
    const dim3 threads(TILE, THREAD_ROWS);
    for (int pivot = 0; pivot < tiles; ++pivot) {
        closePivot<<<1, threads>>>(deviceDist, stride, pivot);
        if (tiles > 1) {
            updatePivotEdges<<<dim3(tiles - 1, 2), threads>>>(deviceDist, stride, pivot);
            updateRemaining<<<dim3(tiles - 1, tiles - 1), threads>>>(deviceDist, stride, pivot);
        }
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(padded.data(), deviceDist, padded.size() * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(deviceDist));
    for (size_t i = 0; i < numNodes; ++i)
        std::copy_n(padded.data() + i * stride, numNodes, dist.data() + i * numNodes);
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
    
    // Run Floyd-Warshall
    printf("Computing shortest paths...\n");
    if (numNodes != 0) CUDA_CHECK(cudaFree(nullptr)); // Initialize the CUDA context before timing.
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
