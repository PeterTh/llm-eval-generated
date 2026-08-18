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

constexpr int TILE_SIZE = 32;
constexpr int THREAD_ROWS = 8;

inline void checkCuda(const cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

// Matrix elements are addressed as (source, destination), but are stored
// column-major to preserve the original program's layout and result hashes.
__global__ void fwPivotKernel(unsigned int* dist, unsigned int* path, const size_t n,
                              const size_t pivotBase) {
    __shared__ unsigned int tile[TILE_SIZE][TILE_SIZE + 1];
    const int x = threadIdx.x;
    const int y = threadIdx.y;

#pragma unroll
    for (int offset = 0; offset < TILE_SIZE; offset += THREAD_ROWS) {
        const int j = y + offset;
        const size_t iGlobal = pivotBase + x;
        const size_t jGlobal = pivotBase + j;
        tile[x][j] = (iGlobal < n && jGlobal < n) ? dist[jGlobal * n + iGlobal] : INF;
    }
    __syncthreads();

    for (int k = 0; k < TILE_SIZE && pivotBase + k < n; ++k) {
#pragma unroll
        for (int offset = 0; offset < TILE_SIZE; offset += THREAD_ROWS) {
            const int j = y + offset;
            const unsigned int candidate = tile[x][k] + tile[k][j];
            if (candidate < tile[x][j]) {
                tile[x][j] = candidate;
                path[(pivotBase + j) * n + pivotBase + x] = static_cast<unsigned int>(pivotBase + k);
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int offset = 0; offset < TILE_SIZE; offset += THREAD_ROWS) {
        const int j = y + offset;
        const size_t iGlobal = pivotBase + x;
        const size_t jGlobal = pivotBase + j;
        if (iGlobal < n && jGlobal < n) {
            dist[jGlobal * n + iGlobal] = tile[x][j];
        }
    }
}

// Updates the pivot row (row=true) or pivot column (row=false).
__global__ void fwPivotEdgeKernel(unsigned int* dist, unsigned int* path, const size_t n,
                                  const size_t pivotBase, const size_t pivotTile,
                                  const bool row) {
    __shared__ unsigned int pivot[TILE_SIZE][TILE_SIZE + 1];
    __shared__ unsigned int edgeTile[TILE_SIZE][TILE_SIZE + 1];
    const int x = threadIdx.x;
    const int y = threadIdx.y;
    size_t tileIndex = blockIdx.x;
    if (tileIndex >= pivotTile) ++tileIndex;
    const size_t otherBase = tileIndex * TILE_SIZE;

#pragma unroll
    for (int offset = 0; offset < TILE_SIZE; offset += THREAD_ROWS) {
        const int j = y + offset;
        const size_t iGlobal = pivotBase + x;
        const size_t jGlobal = pivotBase + j;
        pivot[x][j] = (iGlobal < n && jGlobal < n) ? dist[jGlobal * n + iGlobal] : INF;
        const size_t edgeRow = row ? pivotBase + x : otherBase + x;
        const size_t edgeColumn = row ? otherBase + j : pivotBase + j;
        edgeTile[x][j] = (edgeRow < n && edgeColumn < n)
                             ? dist[edgeColumn * n + edgeRow]
                             : INF;
    }
    __syncthreads();

#pragma unroll
    for (int offset = 0; offset < TILE_SIZE; offset += THREAD_ROWS) {
        const int j = y + offset;
        const size_t iGlobal = row ? pivotBase + x : otherBase + x;
        const size_t jGlobal = row ? otherBase + j : pivotBase + j;
        if (iGlobal >= n || jGlobal >= n) continue;

        unsigned int value = dist[jGlobal * n + iGlobal];
        for (int k = 0; k < TILE_SIZE && pivotBase + k < n; ++k) {
            const unsigned int candidate = row ? pivot[x][k] + edgeTile[k][j]
                                               : edgeTile[x][k] + pivot[k][j];
            if (candidate < value) {
                value = candidate;
                path[jGlobal * n + iGlobal] = static_cast<unsigned int>(pivotBase + k);
            }
        }
        dist[jGlobal * n + iGlobal] = value;
    }
}

__global__ void fwRemainingKernel(unsigned int* dist, unsigned int* path, const size_t n,
                                  const size_t pivotBase, const size_t pivotTile) {
    __shared__ unsigned int left[TILE_SIZE][TILE_SIZE + 1];
    __shared__ unsigned int right[TILE_SIZE][TILE_SIZE + 1];
    const int x = threadIdx.x;
    const int y = threadIdx.y;
    size_t rowTile = blockIdx.x;
    size_t columnTile = blockIdx.y;
    if (rowTile >= pivotTile) ++rowTile;
    if (columnTile >= pivotTile) ++columnTile;
    const size_t rowBase = rowTile * TILE_SIZE;
    const size_t columnBase = columnTile * TILE_SIZE;

#pragma unroll
    for (int offset = 0; offset < TILE_SIZE; offset += THREAD_ROWS) {
        const int j = y + offset;
        const size_t row = rowBase + x;
        const size_t column = columnBase + j;
        const size_t pivotColumn = pivotBase + j;
        const size_t pivotRow = pivotBase + x;
        left[x][j] = (row < n && pivotColumn < n) ? dist[pivotColumn * n + row] : INF;
        right[x][j] = (pivotRow < n && column < n) ? dist[column * n + pivotRow] : INF;
    }
    __syncthreads();

#pragma unroll
    for (int offset = 0; offset < TILE_SIZE; offset += THREAD_ROWS) {
        const int j = y + offset;
        const size_t row = rowBase + x;
        const size_t column = columnBase + j;
        if (row >= n || column >= n) continue;

        unsigned int value = dist[column * n + row];
        for (int k = 0; k < TILE_SIZE && pivotBase + k < n; ++k) {
            const unsigned int candidate = left[x][k] + right[k][j];
            if (candidate < value) {
                value = candidate;
                path[column * n + row] = static_cast<unsigned int>(pivotBase + k);
            }
        }
        dist[column * n + row] = value;
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

    const size_t bytes = numNodes * numNodes * sizeof(unsigned int);
    unsigned int* deviceDist = nullptr;
    unsigned int* devicePath = nullptr;
    checkCuda(cudaMalloc(&deviceDist, bytes), "allocating distance matrix");
    checkCuda(cudaMalloc(&devicePath, bytes), "allocating path matrix");
    checkCuda(cudaMemcpy(deviceDist, dist.data(), bytes, cudaMemcpyHostToDevice),
              "copying distance matrix to device");
    checkCuda(cudaMemcpy(devicePath, path.data(), bytes, cudaMemcpyHostToDevice),
              "copying path matrix to device");

    const size_t tileCount = (numNodes + TILE_SIZE - 1) / TILE_SIZE;
    const dim3 threads(TILE_SIZE, THREAD_ROWS);
    for (size_t pivotTile = 0; pivotTile < tileCount; ++pivotTile) {
        const size_t pivotBase = pivotTile * TILE_SIZE;
        fwPivotKernel<<<1, threads>>>(deviceDist, devicePath, numNodes, pivotBase);
        checkCuda(cudaPeekAtLastError(), "launching pivot kernel");

        if (tileCount > 1) {
            const dim3 edgeGrid(static_cast<unsigned int>(tileCount - 1));
            fwPivotEdgeKernel<<<edgeGrid, threads>>>(deviceDist, devicePath, numNodes,
                                                      pivotBase, pivotTile, true);
            checkCuda(cudaPeekAtLastError(), "launching pivot-row kernel");
            fwPivotEdgeKernel<<<edgeGrid, threads>>>(deviceDist, devicePath, numNodes,
                                                      pivotBase, pivotTile, false);
            checkCuda(cudaPeekAtLastError(), "launching pivot-column kernel");

            const dim3 remainingGrid(static_cast<unsigned int>(tileCount - 1),
                                     static_cast<unsigned int>(tileCount - 1));
            fwRemainingKernel<<<remainingGrid, threads>>>(deviceDist, devicePath, numNodes,
                                                           pivotBase, pivotTile);
            checkCuda(cudaPeekAtLastError(), "launching remaining-tiles kernel");
        }
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
