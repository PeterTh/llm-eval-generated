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

constexpr int TILE = 32;

inline void checkCuda(const cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}

__device__ __forceinline__ size_t deviceIndex(const unsigned int i, const unsigned int j,
                                               const size_t n) {
    return static_cast<size_t>(j) * n + i;
}

// One block closes the diagonal tile.  The k loop remains serial, which is the
// dependency required by Floyd-Warshall; all i,j work within a step is parallel.
__global__ void diagonalTileKernel(unsigned int* dist, unsigned int* path, const size_t n,
                                   const unsigned int pivot, const unsigned int width) {
    __shared__ unsigned int d[TILE][TILE];
    __shared__ unsigned int p[TILE][TILE];
    const unsigned int j = threadIdx.x;
    const unsigned int firstI = threadIdx.y;
    const unsigned int base = pivot * TILE;

    #pragma unroll
    for (unsigned int offset = 0; offset < TILE; offset += blockDim.y) {
        const unsigned int i = firstI + offset;
        if (i < width && j < width) {
            const size_t index = deviceIndex(base + i, base + j, n);
            d[i][j] = dist[index];
            p[i][j] = path[index];
        }
    }
    __syncthreads();

    for (unsigned int k = 0; k < width; ++k) {
        #pragma unroll
        for (unsigned int offset = 0; offset < TILE; offset += blockDim.y) {
            const unsigned int i = firstI + offset;
            if (i < width && j < width) {
                const unsigned int candidate = d[i][k] + d[k][j];
                if (candidate < d[i][j]) {
                    d[i][j] = candidate;
                    p[i][j] = base + k;
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (unsigned int offset = 0; offset < TILE; offset += blockDim.y) {
        const unsigned int i = firstI + offset;
        if (i < width && j < width) {
            const size_t index = deviceIndex(base + i, base + j, n);
            dist[index] = d[i][j];
            path[index] = p[i][j];
        }
    }
}

// Updates one pivot row tile (D[pivot][tile]) after its diagonal tile is closed.
__global__ void pivotRowKernel(unsigned int* dist, unsigned int* path, const size_t n,
                               const unsigned int pivot, const unsigned int pivotWidth) {
    __shared__ unsigned int diagonal[TILE][TILE];
    __shared__ unsigned int d[TILE][TILE];
    __shared__ unsigned int p[TILE][TILE];
    const unsigned int j = threadIdx.x;
    const unsigned int firstI = threadIdx.y;
    const unsigned int pivotBase = pivot * TILE;
    const unsigned int tile = blockIdx.x + (blockIdx.x >= pivot);
    const unsigned int tileBase = tile * TILE;
    const unsigned int tileWidth = static_cast<unsigned int>(min(n - tileBase, static_cast<size_t>(TILE)));

    #pragma unroll
    for (unsigned int offset = 0; offset < TILE; offset += blockDim.y) {
        const unsigned int i = firstI + offset;
        if (i < pivotWidth && j < pivotWidth)
            diagonal[i][j] = dist[deviceIndex(pivotBase + i, pivotBase + j, n)];
        if (i < pivotWidth && j < tileWidth) {
            const size_t index = deviceIndex(pivotBase + i, tileBase + j, n);
            d[i][j] = dist[index];
            p[i][j] = path[index];
        }
    }
    __syncthreads();
    for (unsigned int k = 0; k < pivotWidth; ++k) {
        #pragma unroll
        for (unsigned int offset = 0; offset < TILE; offset += blockDim.y) {
            const unsigned int i = firstI + offset;
            if (i < pivotWidth && j < tileWidth) {
                const unsigned int candidate = diagonal[i][k] + d[k][j];
                if (candidate < d[i][j]) { d[i][j] = candidate; p[i][j] = pivotBase + k; }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (unsigned int offset = 0; offset < TILE; offset += blockDim.y) {
        const unsigned int i = firstI + offset;
        if (i < pivotWidth && j < tileWidth) {
            const size_t index = deviceIndex(pivotBase + i, tileBase + j, n);
            dist[index] = d[i][j]; path[index] = p[i][j];
        }
    }
}

// Updates one pivot column tile (D[tile][pivot]).
__global__ void pivotColumnKernel(unsigned int* dist, unsigned int* path, const size_t n,
                                  const unsigned int pivot, const unsigned int pivotWidth) {
    __shared__ unsigned int d[TILE][TILE];
    __shared__ unsigned int diagonal[TILE][TILE];
    __shared__ unsigned int p[TILE][TILE];
    const unsigned int j = threadIdx.x;
    const unsigned int firstI = threadIdx.y;
    const unsigned int pivotBase = pivot * TILE;
    const unsigned int tile = blockIdx.x + (blockIdx.x >= pivot);
    const unsigned int tileBase = tile * TILE;
    const unsigned int tileWidth = static_cast<unsigned int>(min(n - tileBase, static_cast<size_t>(TILE)));
    #pragma unroll
    for (unsigned int offset = 0; offset < TILE; offset += blockDim.y) {
        const unsigned int i = firstI + offset;
        if (i < tileWidth && j < pivotWidth) {
            const size_t index = deviceIndex(tileBase + i, pivotBase + j, n);
            d[i][j] = dist[index]; p[i][j] = path[index];
        }
        if (i < pivotWidth && j < pivotWidth)
            diagonal[i][j] = dist[deviceIndex(pivotBase + i, pivotBase + j, n)];
    }
    __syncthreads();
    for (unsigned int k = 0; k < pivotWidth; ++k) {
        #pragma unroll
        for (unsigned int offset = 0; offset < TILE; offset += blockDim.y) {
            const unsigned int i = firstI + offset;
            if (i < tileWidth && j < pivotWidth) {
                const unsigned int candidate = d[i][k] + diagonal[k][j];
                if (candidate < d[i][j]) { d[i][j] = candidate; p[i][j] = pivotBase + k; }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (unsigned int offset = 0; offset < TILE; offset += blockDim.y) {
        const unsigned int i = firstI + offset;
        if (i < tileWidth && j < pivotWidth) {
            const size_t index = deviceIndex(tileBase + i, pivotBase + j, n);
            dist[index] = d[i][j]; path[index] = p[i][j];
        }
    }
}

// Updates a non-pivot tile using the completed pivot row and column tiles.
__global__ void remainingTileKernel(unsigned int* dist, unsigned int* path, const size_t n,
                                    const unsigned int pivot, const unsigned int pivotWidth) {
    __shared__ unsigned int column[TILE][TILE];
    __shared__ unsigned int row[TILE][TILE];
    __shared__ unsigned int d[TILE][TILE];
    __shared__ unsigned int p[TILE][TILE];
    const unsigned int j = threadIdx.x;
    const unsigned int firstI = threadIdx.y;
    const unsigned int rowTile = blockIdx.y + (blockIdx.y >= pivot);
    const unsigned int colTile = blockIdx.x + (blockIdx.x >= pivot);
    const unsigned int pivotBase = pivot * TILE, rowBase = rowTile * TILE, colBase = colTile * TILE;
    const unsigned int rowWidth = static_cast<unsigned int>(min(n - rowBase, static_cast<size_t>(TILE)));
    const unsigned int colWidth = static_cast<unsigned int>(min(n - colBase, static_cast<size_t>(TILE)));
    #pragma unroll
    for (unsigned int offset = 0; offset < TILE; offset += blockDim.y) {
        const unsigned int i = firstI + offset;
        if (i < rowWidth && j < pivotWidth)
            column[i][j] = dist[deviceIndex(rowBase + i, pivotBase + j, n)];
        if (i < pivotWidth && j < colWidth)
            row[i][j] = dist[deviceIndex(pivotBase + i, colBase + j, n)];
        if (i < rowWidth && j < colWidth) {
            const size_t index = deviceIndex(rowBase + i, colBase + j, n);
            d[i][j] = dist[index]; p[i][j] = path[index];
        }
    }
    __syncthreads();
    for (unsigned int k = 0; k < pivotWidth; ++k) {
        #pragma unroll
        for (unsigned int offset = 0; offset < TILE; offset += blockDim.y) {
            const unsigned int i = firstI + offset;
            if (i < rowWidth && j < colWidth) {
                const unsigned int candidate = column[i][k] + row[k][j];
                if (candidate < d[i][j]) { d[i][j] = candidate; p[i][j] = pivotBase + k; }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (unsigned int offset = 0; offset < TILE; offset += blockDim.y) {
        const unsigned int i = firstI + offset;
        if (i < rowWidth && j < colWidth) {
            const size_t index = deviceIndex(rowBase + i, colBase + j, n);
            dist[index] = d[i][j]; path[index] = p[i][j];
        }
    }
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

void floydWarshall(std::vector<unsigned int>& dist, 
                   std::vector<unsigned int>& path, 
                   const size_t numNodes) {
    if (numNodes == 0) return;

    unsigned int *deviceDist = nullptr, *devicePath = nullptr;
    const size_t bytes = numNodes * numNodes * sizeof(unsigned int);
    checkCuda(cudaMalloc(&deviceDist, bytes), "allocating distance matrix");
    checkCuda(cudaMalloc(&devicePath, bytes), "allocating path matrix");
    checkCuda(cudaMemcpy(deviceDist, dist.data(), bytes, cudaMemcpyHostToDevice),
              "copying distance matrix to device");
    checkCuda(cudaMemcpy(devicePath, path.data(), bytes, cudaMemcpyHostToDevice),
              "copying path matrix to device");

    const unsigned int tiles = static_cast<unsigned int>((numNodes + TILE - 1) / TILE);
    const dim3 threads(TILE, 8);
    for (unsigned int pivot = 0; pivot < tiles; ++pivot) {
        const unsigned int pivotBase = pivot * TILE;
        const unsigned int pivotWidth = static_cast<unsigned int>(
            std::min(numNodes - pivotBase, static_cast<size_t>(TILE)));
        diagonalTileKernel<<<1, threads>>>(deviceDist, devicePath, numNodes, pivot, pivotWidth);

        if (tiles > 1) {
            const dim3 pivotGrid(tiles - 1);
            pivotRowKernel<<<pivotGrid, threads>>>(deviceDist, devicePath, numNodes, pivot, pivotWidth);
            pivotColumnKernel<<<pivotGrid, threads>>>(deviceDist, devicePath, numNodes, pivot, pivotWidth);
            remainingTileKernel<<<dim3(tiles - 1, tiles - 1), threads>>>(
                deviceDist, devicePath, numNodes, pivot, pivotWidth);
        }
        checkCuda(cudaGetLastError(), "launching Floyd-Warshall kernels");
    }
    checkCuda(cudaDeviceSynchronize(), "executing Floyd-Warshall kernels");
    checkCuda(cudaMemcpy(dist.data(), deviceDist, bytes, cudaMemcpyDeviceToHost),
              "copying distance matrix from device");
    checkCuda(cudaMemcpy(path.data(), devicePath, bytes, cudaMemcpyDeviceToHost),
              "copying path matrix from device");
    checkCuda(cudaFree(deviceDist), "freeing distance matrix");
    checkCuda(cudaFree(devicePath), "freeing path matrix");
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
