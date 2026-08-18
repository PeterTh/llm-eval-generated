#include <algorithm>
#include <chrono>
#include <climits>
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
constexpr int BLOCK_ROWS = 8;

#define CUDA_CHECK(call) do {                                                   \
    const cudaError_t error_ = (call);                                          \
    if (error_ != cudaSuccess) {                                                \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                     cudaGetErrorString(error_));                               \
        std::exit(EXIT_FAILURE);                                                \
    }                                                                           \
} while (false)

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

// Each CUDA block has 256 threads.  Every thread handles four rows, retaining
// coalesced accesses along x while avoiding the occupancy cost of 1024-thread
// blocks.  Padding is represented by INF in shared memory.
__global__ __launch_bounds__(TILE_SIZE * BLOCK_ROWS)
void closePivotKernel(unsigned int* __restrict__ dist,
                      unsigned int* __restrict__ path, int n, int round) {
    __shared__ unsigned int tile[TILE_SIZE][TILE_SIZE];
    const int x = threadIdx.x;
    const int base = round * TILE_SIZE;

#pragma unroll
    for (int y = threadIdx.y; y < TILE_SIZE; y += BLOCK_ROWS) {
        const int row = base + y;
        const int col = base + x;
        tile[y][x] = (row < n && col < n) ? dist[row * n + col] : INF;
    }
    __syncthreads();

    const int kLimit = min(TILE_SIZE, n - base);
    for (int k = 0; k < kLimit; ++k) {
#pragma unroll
        for (int y = threadIdx.y; y < TILE_SIZE; y += BLOCK_ROWS) {
            const unsigned int candidate = tile[y][k] + tile[k][x];
            if (candidate < tile[y][x]) {
                tile[y][x] = candidate;
                const int row = base + y;
                const int col = base + x;
                if (row < n && col < n)
                    path[row * n + col] = static_cast<unsigned int>(base + k);
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int y = threadIdx.y; y < TILE_SIZE; y += BLOCK_ROWS) {
        const int row = base + y;
        const int col = base + x;
        if (row < n && col < n)
            dist[row * n + col] = tile[y][x];
    }
}

__global__ __launch_bounds__(TILE_SIZE * BLOCK_ROWS)
void updatePivotBandsKernel(unsigned int* __restrict__ dist,
                            unsigned int* __restrict__ path, int n, int round) {
    __shared__ unsigned int pivot[TILE_SIZE][TILE_SIZE];
    __shared__ unsigned int target[TILE_SIZE][TILE_SIZE];
    const int x = threadIdx.x;
    const int pivotBase = round * TILE_SIZE;
    int tile = blockIdx.x;
    if (tile >= round) ++tile;
    const int targetBase = tile * TILE_SIZE;
    const bool vertical = blockIdx.y != 0;

#pragma unroll
    for (int y = threadIdx.y; y < TILE_SIZE; y += BLOCK_ROWS) {
        const int pivotRow = pivotBase + y;
        const int pivotCol = pivotBase + x;
        pivot[y][x] = (pivotRow < n && pivotCol < n)
                            ? dist[pivotRow * n + pivotCol] : INF;

        const int row = (vertical ? targetBase : pivotBase) + y;
        const int col = (vertical ? pivotBase : targetBase) + x;
        target[y][x] = (row < n && col < n) ? dist[row * n + col] : INF;
    }
    __syncthreads();

    const int kLimit = min(TILE_SIZE, n - pivotBase);
    for (int k = 0; k < kLimit; ++k) {
#pragma unroll
        for (int y = threadIdx.y; y < TILE_SIZE; y += BLOCK_ROWS) {
            const unsigned int candidate = vertical
                    ? target[y][k] + pivot[k][x]
                    : pivot[y][k] + target[k][x];
            if (candidate < target[y][x]) {
                target[y][x] = candidate;
                const int row = (vertical ? targetBase : pivotBase) + y;
                const int col = (vertical ? pivotBase : targetBase) + x;
                if (row < n && col < n)
                    path[row * n + col] = static_cast<unsigned int>(pivotBase + k);
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int y = threadIdx.y; y < TILE_SIZE; y += BLOCK_ROWS) {
        const int row = (vertical ? targetBase : pivotBase) + y;
        const int col = (vertical ? pivotBase : targetBase) + x;
        if (row < n && col < n)
            dist[row * n + col] = target[y][x];
    }
}

__global__ __launch_bounds__(TILE_SIZE * BLOCK_ROWS)
void updateRemainingKernel(unsigned int* __restrict__ dist,
                           unsigned int* __restrict__ path, int n, int round) {
    __shared__ unsigned int columnTile[TILE_SIZE][TILE_SIZE];
    __shared__ unsigned int rowTile[TILE_SIZE][TILE_SIZE];
    const int x = threadIdx.x;
    int rowTileIndex = blockIdx.y;
    int colTileIndex = blockIdx.x;
    if (rowTileIndex >= round) ++rowTileIndex;
    if (colTileIndex >= round) ++colTileIndex;
    const int rowBase = rowTileIndex * TILE_SIZE;
    const int colBase = colTileIndex * TILE_SIZE;
    const int pivotBase = round * TILE_SIZE;

#pragma unroll
    for (int y = threadIdx.y; y < TILE_SIZE; y += BLOCK_ROWS) {
        const int row = rowBase + y;
        const int col = colBase + x;
        const int pivotRow = pivotBase + y;
        const int pivotCol = pivotBase + x;
        columnTile[y][x] = (row < n && pivotCol < n)
                                  ? dist[row * n + pivotCol] : INF;
        rowTile[y][x] = (pivotRow < n && col < n)
                               ? dist[pivotRow * n + col] : INF;
    }
    __syncthreads();

    const int kLimit = min(TILE_SIZE, n - pivotBase);
#pragma unroll
    for (int y = threadIdx.y; y < TILE_SIZE; y += BLOCK_ROWS) {
        const int row = rowBase + y;
        const int col = colBase + x;
        if (row < n && col < n) {
            unsigned int best = dist[row * n + col];
            unsigned int bestK = 0;
            bool changed = false;
#pragma unroll
            for (int k = 0; k < TILE_SIZE; ++k) {
                if (k < kLimit) {
                    const unsigned int candidate = columnTile[y][k] + rowTile[k][x];
                    if (candidate < best) {
                        best = candidate;
                        bestK = static_cast<unsigned int>(pivotBase + k);
                        changed = true;
                    }
                }
            }
            dist[row * n + col] = best;
            if (changed) path[row * n + col] = bestK;
        }
    }
}

void floydWarshall(unsigned int* deviceDist, unsigned int* devicePath, int numNodes) {
    const int rounds = (numNodes - 1) / TILE_SIZE + 1;
    const dim3 block(TILE_SIZE, BLOCK_ROWS);
    for (int round = 0; round < rounds; ++round) {
        closePivotKernel<<<1, block>>>(deviceDist, devicePath, numNodes, round);
        if (rounds > 1) {
            updatePivotBandsKernel<<<dim3(rounds - 1, 2), block>>>(
                    deviceDist, devicePath, numNodes, round);
            updateRemainingKernel<<<dim3(rounds - 1, rounds - 1), block>>>(
                    deviceDist, devicePath, numNodes, round);
        }
    }
    CUDA_CHECK(cudaGetLastError());
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
            const char* value = argv[++i];
            char* end = nullptr;
            const unsigned long long parsed = std::strtoull(value, &end, 10);
            if (*value == '\0' || *end != '\0' || parsed == 0 || parsed > INT_MAX) {
                std::fprintf(stderr, "Invalid number of nodes: %s\n", value);
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
    
    if (numNodes > std::numeric_limits<size_t>::max() / numNodes /
                           sizeof(unsigned int)) {
        std::fprintf(stderr, "Matrix size is too large\n");
        return 1;
    }
    const size_t matrixElements = numNodes * numNodes;
    const size_t matrixBytes = matrixElements * sizeof(unsigned int);

    // Allocate matrices
    std::vector<unsigned int> dist(matrixElements);
    std::vector<unsigned int> path(matrixElements);
    
    // Initialize
    printf("Initializing graph...\n");
    initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
    initializePathMatrix(path, numNodes);

    unsigned int* deviceDist = nullptr;
    unsigned int* devicePath = nullptr;
    CUDA_CHECK(cudaMalloc(&deviceDist, matrixBytes));
    CUDA_CHECK(cudaMalloc(&devicePath, matrixBytes));
    CUDA_CHECK(cudaMemcpy(deviceDist, dist.data(), matrixBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(devicePath, path.data(), matrixBytes, cudaMemcpyHostToDevice));
    
    // Run Floyd-Warshall
    printf("Computing shortest paths...\n");
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto start = std::chrono::high_resolution_clock::now();
    
    floydWarshall(deviceDist, devicePath, static_cast<int>(numNodes));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    const auto end = std::chrono::high_resolution_clock::now();
    const double durationMs = std::chrono::duration<double, std::milli>(end - start).count();

    CUDA_CHECK(cudaMemcpy(dist.data(), deviceDist, matrixBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(path.data(), devicePath, matrixBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(deviceDist));
    CUDA_CHECK(cudaFree(devicePath));
    
    printf("Computation time: %.3f ms\n", durationMs);
    
    // Calculate operations per second
    // Floyd-Warshall has O(n³) complexity
    double ops = (double)numNodes * numNodes * numNodes;
    double gflops = ops / (durationMs / 1000.0) / 1e9;
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
