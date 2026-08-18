#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
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

constexpr int TILE_SIZE = 32;

inline void cudaCheck(const cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error during %s: %s\n", operation,
                cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}

__device__ __forceinline__ size_t matrixIndex(const int row, const int column,
                                              const int n) {
    return static_cast<size_t>(row) * static_cast<size_t>(n) + column;
}

// Process the diagonal tile for one block of intermediate vertices.
__global__ void floydDiagonal(unsigned int* dist, unsigned int* path, int n,
                              int pivot, int tileCount) {
    (void)tileCount;
    __shared__ unsigned int d[TILE_SIZE][TILE_SIZE];
    __shared__ unsigned int p[TILE_SIZE][TILE_SIZE];
    const int x = threadIdx.x;
    const int y = threadIdx.y;
    const int row = pivot * TILE_SIZE + y;
    const int col = pivot * TILE_SIZE + x;
    const bool valid = row < n && col < n;
    d[y][x] = valid ? dist[matrixIndex(row, col, n)] : INF;
    p[y][x] = valid ? path[matrixIndex(row, col, n)] : 0;
    __syncthreads();
    const int limit = min(TILE_SIZE, n - pivot * TILE_SIZE);
    for (int kk = 0; kk < limit; ++kk) {
        __syncthreads();
        if (valid) {
            const unsigned int candidate = d[kk][x] + d[y][kk];
            if (candidate < d[y][x]) {
                d[y][x] = candidate;
                p[y][x] = pivot * TILE_SIZE + kk;
            }
        }
    }
    __syncthreads();
    if (valid) {
        dist[matrixIndex(row, col, n)] = d[y][x];
        path[matrixIndex(row, col, n)] = p[y][x];
    }
}

// Process a tile in the pivot row or pivot column.
__global__ void floydPivotLine(unsigned int* dist, unsigned int* path, int n,
                               int pivot, bool rowLine) {
    __shared__ unsigned int other[TILE_SIZE][TILE_SIZE];
    __shared__ unsigned int diagonal[TILE_SIZE][TILE_SIZE];
    __shared__ unsigned int outPath[TILE_SIZE][TILE_SIZE];
    const int x = threadIdx.x;
    const int y = threadIdx.y;
    const int tile = blockIdx.x < static_cast<unsigned>(pivot) ? blockIdx.x : blockIdx.x + 1;
    const int outRow = (rowLine ? pivot : tile) * TILE_SIZE + y;
    const int outCol = (rowLine ? tile : pivot) * TILE_SIZE + x;
    const int otherRow = rowLine ? pivot * TILE_SIZE + y : tile * TILE_SIZE + y;
    const int otherCol = rowLine ? tile * TILE_SIZE + x : pivot * TILE_SIZE + x;
    const int pivotRow = pivot * TILE_SIZE + y;
    const int pivotCol = pivot * TILE_SIZE + x;
    const bool valid = outRow < n && outCol < n;
    other[y][x] = (otherRow < n && otherCol < n) ?
        dist[matrixIndex(otherRow, otherCol, n)] : INF;
    diagonal[y][x] = (pivotRow < n && pivotCol < n) ?
        dist[matrixIndex(pivotRow, pivotCol, n)] : INF;
    outPath[y][x] = valid ? path[matrixIndex(outRow, outCol, n)] : 0;
    __syncthreads();
    const int limit = min(TILE_SIZE, n - pivot * TILE_SIZE);
    for (int kk = 0; kk < limit; ++kk) {
        __syncthreads();
        if (valid) {
            const unsigned int candidate = rowLine ?
                other[kk][x] + diagonal[y][kk] :
                diagonal[kk][x] + other[y][kk];
            const unsigned int current = other[y][x];
            if (candidate < current) {
                other[y][x] = candidate;
                outPath[y][x] = pivot * TILE_SIZE + kk;
            }
        }
    }
    __syncthreads();
    if (valid) {
        dist[matrixIndex(outRow, outCol, n)] = other[y][x];
        path[matrixIndex(outRow, outCol, n)] = outPath[y][x];
    }
}

// Process all tiles outside the pivot row and column.
__global__ void floydRemainder(unsigned int* dist, unsigned int* path, int n,
                               int pivot, int tileCount) {
    __shared__ unsigned int rowTile[TILE_SIZE][TILE_SIZE];
    __shared__ unsigned int colTile[TILE_SIZE][TILE_SIZE];
    __shared__ unsigned int outTile[TILE_SIZE][TILE_SIZE];
    __shared__ unsigned int outPath[TILE_SIZE][TILE_SIZE];
    const int x = threadIdx.x;
    const int y = threadIdx.y;
    const int tileX = blockIdx.x < static_cast<unsigned>(pivot) ? blockIdx.x : blockIdx.x + 1;
    const int tileY = blockIdx.y < static_cast<unsigned>(pivot) ? blockIdx.y : blockIdx.y + 1;
    const int row = tileY * TILE_SIZE + y;
    const int col = tileX * TILE_SIZE + x;
    const int pivotRow = pivot * TILE_SIZE + y;
    const int pivotCol = pivot * TILE_SIZE + x;
    rowTile[y][x] = (pivotRow < n && col < n) ? dist[matrixIndex(pivotRow, col, n)] : INF;
    colTile[y][x] = (row < n && pivotCol < n) ? dist[matrixIndex(row, pivotCol, n)] : INF;
    const bool valid = row < n && col < n;
    outTile[y][x] = valid ? dist[matrixIndex(row, col, n)] : INF;
    outPath[y][x] = valid ? path[matrixIndex(row, col, n)] : 0;
    __syncthreads();
    const int limit = min(TILE_SIZE, n - pivot * TILE_SIZE);
    for (int kk = 0; kk < limit; ++kk) {
        __syncthreads();
        if (valid) {
            const unsigned int candidate = rowTile[kk][x] + colTile[y][kk];
            if (candidate < outTile[y][x]) {
                outTile[y][x] = candidate;
                outPath[y][x] = pivot * TILE_SIZE + kk;
            }
        }
    }
    __syncthreads();
    if (valid) {
        dist[matrixIndex(row, col, n)] = outTile[y][x];
        path[matrixIndex(row, col, n)] = outPath[y][x];
    }
}

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path, const size_t numNodes) {
    if (numNodes == 0) return;
    if (numNodes > static_cast<size_t>(INT_MAX)) {
        fprintf(stderr, "Number of nodes is too large for CUDA indexing\n");
        std::exit(EXIT_FAILURE);
    }
    const int n = static_cast<int>(numNodes);
    const int tileCount = (n + TILE_SIZE - 1) / TILE_SIZE;
    const size_t bytes = numNodes * numNodes * sizeof(unsigned int);
    unsigned int* deviceDist = nullptr;
    unsigned int* devicePath = nullptr;
    cudaCheck(cudaMalloc(&deviceDist, bytes), "cudaMalloc(dist)");
    cudaCheck(cudaMalloc(&devicePath, bytes), "cudaMalloc(path)");
    cudaCheck(cudaMemcpy(deviceDist, dist.data(), bytes, cudaMemcpyHostToDevice), "copy dist to device");
    cudaCheck(cudaMemcpy(devicePath, path.data(), bytes, cudaMemcpyHostToDevice), "copy path to device");

    const dim3 threads(TILE_SIZE, TILE_SIZE);
    for (int pivot = 0; pivot < tileCount; ++pivot) {
        floydDiagonal<<<1, threads>>>(deviceDist, devicePath, n, pivot, tileCount);
        cudaCheck(cudaGetLastError(), "diagonal kernel launch");
        if (tileCount > 1) {
            const dim3 lineGrid(tileCount - 1);
            floydPivotLine<<<lineGrid, threads>>>(deviceDist, devicePath, n, pivot, true);
            cudaCheck(cudaGetLastError(), "pivot row kernel launch");
            floydPivotLine<<<lineGrid, threads>>>(deviceDist, devicePath, n, pivot, false);
            cudaCheck(cudaGetLastError(), "pivot column kernel launch");
            const dim3 remainderGrid(tileCount - 1, tileCount - 1);
            floydRemainder<<<remainderGrid, threads>>>(deviceDist, devicePath, n, pivot, tileCount);
            cudaCheck(cudaGetLastError(), "remainder kernel launch");
        }
    }
    cudaCheck(cudaDeviceSynchronize(), "Floyd-Warshall kernels");
    cudaCheck(cudaMemcpy(dist.data(), deviceDist, bytes, cudaMemcpyDeviceToHost), "copy dist to host");
    cudaCheck(cudaMemcpy(path.data(), devicePath, bytes, cudaMemcpyDeviceToHost), "copy path to host");
    cudaCheck(cudaFree(deviceDist), "cudaFree(dist)");
    cudaCheck(cudaFree(devicePath), "cudaFree(path)");
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
    // Pay one-time CUDA context startup cost before the timed region.
    cudaCheck(cudaFree(nullptr), "initialize CUDA context");
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
