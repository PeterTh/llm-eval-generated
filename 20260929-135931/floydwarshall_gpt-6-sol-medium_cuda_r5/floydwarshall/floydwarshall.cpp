#include <algorithm>
#include <chrono>
#include <climits>
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

constexpr int TILE = 32;

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "%s failed: %s\n", operation, cudaGetErrorString(status));
        exit(EXIT_FAILURE);
    }
}

// Each block has 32 columns and eight threads in the row direction. A thread
// owns four rows, keeping the block at 256 threads while covering a 32x32 tile.
__device__ __forceinline__ void loadTile(unsigned int* tile, const unsigned int* matrix,
                                          int tileRow, int tileCol, int n) {
    const int x = threadIdx.x;
    for (int y = threadIdx.y; y < TILE; y += blockDim.y) {
        const int row = tileRow * TILE + y;
        const int col = tileCol * TILE + x;
        tile[y * TILE + x] = row < n && col < n ? matrix[static_cast<size_t>(row) * n + col] : INF;
    }
}

__device__ __forceinline__ void storeTile(const unsigned int* tile, unsigned int* matrix,
                                           int tileRow, int tileCol, int n) {
    const int x = threadIdx.x;
    for (int y = threadIdx.y; y < TILE; y += blockDim.y) {
        const int row = tileRow * TILE + y;
        const int col = tileCol * TILE + x;
        if (row < n && col < n) matrix[static_cast<size_t>(row) * n + col] = tile[y * TILE + x];
    }
}

__global__ void pivotKernel(unsigned int* dist, unsigned int* path, int n, int pivot) {
    __shared__ unsigned int d[TILE * TILE];
    __shared__ unsigned int p[TILE * TILE];
    loadTile(d, dist, pivot, pivot, n);
    loadTile(p, path, pivot, pivot, n);
    __syncthreads();

    for (int k = 0; k < TILE && pivot * TILE + k < n; ++k) {
        for (int i = threadIdx.y; i < TILE; i += blockDim.y) {
            const int offset = i * TILE + threadIdx.x;
            const unsigned int candidate = d[i * TILE + k] + d[k * TILE + threadIdx.x];
            if (candidate < d[offset]) {
                d[offset] = candidate;
                p[offset] = pivot * TILE + k;
            }
        }
        __syncthreads();
    }
    storeTile(d, dist, pivot, pivot, n);
    storeTile(p, path, pivot, pivot, n);
}

__global__ void edgeKernel(unsigned int* dist, unsigned int* path, int n, int pivot) {
    __shared__ unsigned int diagonal[TILE * TILE];
    __shared__ unsigned int d[TILE * TILE];
    __shared__ unsigned int p[TILE * TILE];
    const int other = blockIdx.x < pivot ? blockIdx.x : blockIdx.x + 1;
    const bool rowTile = blockIdx.y == 0;
    const int tileRow = rowTile ? pivot : other;
    const int tileCol = rowTile ? other : pivot;
    loadTile(diagonal, dist, pivot, pivot, n);
    loadTile(d, dist, tileRow, tileCol, n);
    loadTile(p, path, tileRow, tileCol, n);
    __syncthreads();

    for (int k = 0; k < TILE && pivot * TILE + k < n; ++k) {
        for (int i = threadIdx.y; i < TILE; i += blockDim.y) {
            const int offset = i * TILE + threadIdx.x;
            const unsigned int left = rowTile ? diagonal[i * TILE + k] : d[i * TILE + k];
            const unsigned int right = rowTile ? d[k * TILE + threadIdx.x]
                                               : diagonal[k * TILE + threadIdx.x];
            const unsigned int candidate = left + right;
            if (candidate < d[offset]) {
                d[offset] = candidate;
                p[offset] = pivot * TILE + k;
            }
        }
        __syncthreads();
    }
    storeTile(d, dist, tileRow, tileCol, n);
    storeTile(p, path, tileRow, tileCol, n);
}

__global__ void interiorKernel(unsigned int* dist, unsigned int* path, int n, int pivot) {
    __shared__ unsigned int left[TILE * TILE];
    __shared__ unsigned int right[TILE * TILE];
    const int tileRow = blockIdx.y < pivot ? blockIdx.y : blockIdx.y + 1;
    const int tileCol = blockIdx.x < pivot ? blockIdx.x : blockIdx.x + 1;
    loadTile(left, dist, tileRow, pivot, n);
    loadTile(right, dist, pivot, tileCol, n);
    __syncthreads();

    const int col = tileCol * TILE + threadIdx.x;
    for (int i = threadIdx.y; i < TILE; i += blockDim.y) {
        const int row = tileRow * TILE + i;
        if (row >= n || col >= n) continue;
        const size_t offset = static_cast<size_t>(row) * n + col;
        unsigned int best = dist[offset];
        unsigned int via = path[offset];
#pragma unroll
        for (int k = 0; k < TILE; ++k) {
            const unsigned int candidate = left[i * TILE + k] + right[k * TILE + threadIdx.x];
            if (candidate < best) {
                best = candidate;
                via = pivot * TILE + k;
            }
        }
        dist[offset] = best;
        path[offset] = via;
    }
}

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes) {
    if (numNodes == 0) return;
    if (numNodes > static_cast<size_t>(INT_MAX)) {
        fprintf(stderr, "Number of nodes exceeds CUDA indexing limit\n");
        exit(EXIT_FAILURE);
    }
    const int n = static_cast<int>(numNodes);
    const int tiles = (n + TILE - 1) / TILE;
    const size_t bytes = dist.size() * sizeof(unsigned int);
    unsigned int *deviceDist = nullptr, *devicePath = nullptr;
    checkCuda(cudaMalloc(&deviceDist, bytes), "cudaMalloc distance");
    checkCuda(cudaMalloc(&devicePath, bytes), "cudaMalloc path");
    checkCuda(cudaMemcpy(deviceDist, dist.data(), bytes, cudaMemcpyHostToDevice), "copy distance to GPU");
    checkCuda(cudaMemcpy(devicePath, path.data(), bytes, cudaMemcpyHostToDevice), "copy path to GPU");

    const dim3 block(TILE, 8);
    for (int pivot = 0; pivot < tiles; ++pivot) {
        pivotKernel<<<1, block>>>(deviceDist, devicePath, n, pivot);
        if (tiles > 1) {
            edgeKernel<<<dim3(tiles - 1, 2), block>>>(deviceDist, devicePath, n, pivot);
            interiorKernel<<<dim3(tiles - 1, tiles - 1), block>>>(deviceDist, devicePath, n, pivot);
        }
    }
    checkCuda(cudaGetLastError(), "Floyd-Warshall kernels");
    checkCuda(cudaMemcpy(dist.data(), deviceDist, bytes, cudaMemcpyDeviceToHost), "copy distance from GPU");
    checkCuda(cudaMemcpy(path.data(), devicePath, bytes, cudaMemcpyDeviceToHost), "copy path from GPU");
    checkCuda(cudaFree(deviceDist), "cudaFree distance");
    checkCuda(cudaFree(devicePath), "cudaFree path");
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
    // CUDA context creation is one-time process setup, not graph computation.
    if (numNodes != 0) checkCuda(cudaFree(nullptr), "initialize CUDA");
    
    // Run Floyd-Warshall
    printf("Computing shortest paths...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    floydWarshall(dist, path, numNodes);
    
    auto end = std::chrono::high_resolution_clock::now();
    const double seconds = std::chrono::duration<double>(end - start).count();
    
    printf("Computation time: %.3f ms\n", seconds * 1000.0);
    
    // Calculate operations per second
    // Floyd-Warshall has O(n³) complexity
    double ops = (double)numNodes * numNodes * numNodes;
    double gflops = ops / seconds / 1e9;
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
