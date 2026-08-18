#include <algorithm>
#include <chrono>
#include <cmath>
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

[[noreturn]] void reportCudaError(const char* operation, const cudaError_t error) {
    fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(error));
    std::exit(EXIT_FAILURE);
}

#define CUDA_CHECK(operation) \
    do { \
        const cudaError_t cudaError = (operation); \
        if (cudaError != cudaSuccess) { \
            reportCudaError(#operation, cudaError); \
        } \
    } while (false)

// Each CUDA block owns one TILE_SIZE x TILE_SIZE submatrix.  The matrix keeps
// the original column-major layout: element (row, column) is column * n + row.
__global__ void floydWarshallDiagonalKernel(unsigned int* __restrict__ dist,
                                            unsigned int* __restrict__ path,
                                            const int numNodes,
                                            const int pivotTile) {
    __shared__ unsigned int tile[TILE_SIZE][TILE_SIZE];

    const int localColumn = static_cast<int>(threadIdx.x);
    const int localRow = static_cast<int>(threadIdx.y);
    const int row = pivotTile * TILE_SIZE + localRow;
    const int column = pivotTile * TILE_SIZE + localColumn;
    const bool valid = row < numNodes && column < numNodes;

    tile[localRow][localColumn] = valid
        ? dist[static_cast<size_t>(column) * numNodes + row]
        : INF;
    __syncthreads();

    const int tileEnd = min(TILE_SIZE, numNodes - pivotTile * TILE_SIZE);
    for (int k = 0; k < tileEnd; ++k) {
        const unsigned int candidate = tile[localRow][k] + tile[k][localColumn];
        if (valid && candidate < tile[localRow][localColumn]) {
            tile[localRow][localColumn] = candidate;
            path[static_cast<size_t>(column) * numNodes + row] = pivotTile * TILE_SIZE + k;
        }
        __syncthreads();
    }

    if (valid) {
        dist[static_cast<size_t>(column) * numNodes + row] = tile[localRow][localColumn];
    }
}

// The y dimension selects the row or column phase.  Both phases can use the
// same kernel because their update is A[row][k] + A[k][column].
__global__ void floydWarshallPivotKernel(unsigned int* __restrict__ dist,
                                         unsigned int* __restrict__ path,
                                         const int numNodes,
                                         const int pivotTile) {
    const int otherTile = static_cast<int>(blockIdx.x);
    if (otherTile == pivotTile) {
        return;
    }

    __shared__ unsigned int diagonal[TILE_SIZE][TILE_SIZE];
    __shared__ unsigned int side[TILE_SIZE][TILE_SIZE];

    const int localColumn = static_cast<int>(threadIdx.x);
    const int localRow = static_cast<int>(threadIdx.y);
    const bool rowPhase = blockIdx.y == 0;
    const int targetTileRow = rowPhase ? pivotTile : otherTile;
    const int targetTileColumn = rowPhase ? otherTile : pivotTile;
    const int row = targetTileRow * TILE_SIZE + localRow;
    const int column = targetTileColumn * TILE_SIZE + localColumn;
    const int diagonalRow = pivotTile * TILE_SIZE + localRow;
    const int diagonalColumn = pivotTile * TILE_SIZE + localColumn;

    const bool validDiagonal = diagonalRow < numNodes && diagonalColumn < numNodes;
    const bool validTarget = row < numNodes && column < numNodes;

    diagonal[localRow][localColumn] = validDiagonal
        ? dist[static_cast<size_t>(diagonalColumn) * numNodes + diagonalRow]
        : INF;
    side[localRow][localColumn] = validTarget
        ? dist[static_cast<size_t>(column) * numNodes + row]
        : INF;
    __syncthreads();

    const int tileEnd = min(TILE_SIZE, numNodes - pivotTile * TILE_SIZE);
    for (int k = 0; k < tileEnd; ++k) {
        const unsigned int candidate = rowPhase
            ? diagonal[localRow][k] + side[k][localColumn]
            : side[localRow][k] + diagonal[k][localColumn];
        if (validTarget && candidate < side[localRow][localColumn]) {
            side[localRow][localColumn] = candidate;
            path[static_cast<size_t>(column) * numNodes + row] = pivotTile * TILE_SIZE + k;
        }
        __syncthreads();
    }

    if (validTarget) {
        dist[static_cast<size_t>(column) * numNodes + row] = side[localRow][localColumn];
    }
}

__global__ void floydWarshallRemainingKernel(unsigned int* __restrict__ dist,
                                             unsigned int* __restrict__ path,
                                             const int numNodes,
                                             const int pivotTile) {
    const int tileRow = static_cast<int>(blockIdx.y);
    const int tileColumn = static_cast<int>(blockIdx.x);
    if (tileRow == pivotTile || tileColumn == pivotTile) {
        return;
    }

    __shared__ unsigned int rowTile[TILE_SIZE][TILE_SIZE];
    __shared__ unsigned int columnTile[TILE_SIZE][TILE_SIZE];

    const int localColumn = static_cast<int>(threadIdx.x);
    const int localRow = static_cast<int>(threadIdx.y);
    const int row = tileRow * TILE_SIZE + localRow;
    const int column = tileColumn * TILE_SIZE + localColumn;
    const int pivotRow = pivotTile * TILE_SIZE + localRow;
    const int pivotColumn = pivotTile * TILE_SIZE + localColumn;

    const bool validTarget = row < numNodes && column < numNodes;
    const bool validRowTile = row < numNodes && pivotColumn < numNodes;
    const bool validColumnTile = pivotRow < numNodes && column < numNodes;

    // rowTile contains A[row][k], and columnTile contains A[k][column].
    rowTile[localRow][localColumn] = validRowTile
        ? dist[static_cast<size_t>(pivotColumn) * numNodes + row]
        : INF;
    columnTile[localRow][localColumn] = validColumnTile
        ? dist[static_cast<size_t>(column) * numNodes + pivotRow]
        : INF;
    unsigned int value = validTarget
        ? dist[static_cast<size_t>(column) * numNodes + row]
        : INF;
    __syncthreads();

    const int tileEnd = min(TILE_SIZE, numNodes - pivotTile * TILE_SIZE);
    for (int k = 0; k < tileEnd; ++k) {
        const unsigned int candidate = rowTile[localRow][k] + columnTile[k][localColumn];
        if (validTarget && candidate < value) {
            value = candidate;
            path[static_cast<size_t>(column) * numNodes + row] = pivotTile * TILE_SIZE + k;
        }
        __syncthreads();
    }

    if (validTarget) {
        dist[static_cast<size_t>(column) * numNodes + row] = value;
    }
}

void floydWarshall(std::vector<unsigned int>& dist, 
                   std::vector<unsigned int>& path, 
                   const size_t numNodes) {
    if (numNodes == 0) {
        return;
    }

    if (numNodes > static_cast<size_t>(std::numeric_limits<int>::max())) {
        fprintf(stderr, "Number of nodes is too large for the CUDA implementation\n");
        std::exit(EXIT_FAILURE);
    }

    unsigned int* deviceDist = nullptr;
    unsigned int* devicePath = nullptr;
    const size_t matrixElements = numNodes * numNodes;
    const size_t matrixBytes = matrixElements * sizeof(unsigned int);
    const int nodeCount = static_cast<int>(numNodes);
    const int tileCount = (nodeCount + TILE_SIZE - 1) / TILE_SIZE;
    const dim3 threads(TILE_SIZE, TILE_SIZE);

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceDist), matrixBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&devicePath), matrixBytes));
    CUDA_CHECK(cudaMemcpy(deviceDist, dist.data(), matrixBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(devicePath, path.data(), matrixBytes, cudaMemcpyHostToDevice));

    for (int pivotTile = 0; pivotTile < tileCount; ++pivotTile) {
        floydWarshallDiagonalKernel<<<1, threads>>>(deviceDist, devicePath,
                                                     nodeCount, pivotTile);
        CUDA_CHECK(cudaGetLastError());

        floydWarshallPivotKernel<<<dim3(tileCount, 2), threads>>>(deviceDist, devicePath,
                                                                  nodeCount, pivotTile);
        CUDA_CHECK(cudaGetLastError());

        floydWarshallRemainingKernel<<<dim3(tileCount, tileCount), threads>>>(
            deviceDist, devicePath, nodeCount, pivotTile);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(dist.data(), deviceDist, matrixBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(path.data(), devicePath, matrixBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(devicePath));
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
    std::vector<unsigned int> path(numNodes * numNodes);
    
    // Initialize
    printf("Initializing graph...\n");
    initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
    initializePathMatrix(path, numNodes);
    
    // Run Floyd-Warshall
    printf("Computing shortest paths...\n");
    // Initialize the CUDA context before timing so one-time driver startup
    // does not dominate the algorithm's measured execution time.
    CUDA_CHECK(cudaFree(nullptr));
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
