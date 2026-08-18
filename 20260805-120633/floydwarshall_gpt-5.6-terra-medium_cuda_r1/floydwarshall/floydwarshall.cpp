#include <algorithm>
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
constexpr int TILE_SIZE = 32;

inline void checkCuda(const cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

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

// The storage order is [destination][source].  The three phases below process
// one pivot tile at a time, preserving the original ascending k ordering.
__global__ void floydPhase1(unsigned int* dist, unsigned int* path, const int n,
                            const int pivot) {
    __shared__ unsigned int distances[TILE_SIZE][TILE_SIZE + 1];
    const int col = threadIdx.x;
    const int row = threadIdx.y;
    const int globalRow = pivot * TILE_SIZE + row;
    const int globalCol = pivot * TILE_SIZE + col;
    const bool valid = globalRow < n && globalCol < n;

    distances[row][col] = valid ? dist[globalRow * n + globalCol] : INF;
    unsigned int localPath = valid ? path[globalRow * n + globalCol] : 0;
    __syncthreads();

    const int limit = min(TILE_SIZE, n - pivot * TILE_SIZE);
    for (int k = 0; k < limit; ++k) {
        const unsigned int candidate = distances[row][k] + distances[k][col];
        if (candidate < distances[row][col]) {
            distances[row][col] = candidate;
            localPath = pivot * TILE_SIZE + k;
        }
        __syncthreads();
    }
    if (valid) {
        dist[globalRow * n + globalCol] = distances[row][col];
        path[globalRow * n + globalCol] = localPath;
    }
}

// phase == 0 updates [block, pivot], phase == 1 updates [pivot, block].
__global__ void floydPhase2(unsigned int* dist, unsigned int* path, const int n,
                            const int pivot, const int phase) {
    __shared__ unsigned int tile[TILE_SIZE][TILE_SIZE + 1];
    __shared__ unsigned int pivotTile[TILE_SIZE][TILE_SIZE + 1];
    const int col = threadIdx.x;
    const int row = threadIdx.y;
    const int block = blockIdx.x;
    if (block == pivot) return;

    const int tileRow = (phase == 0 ? block : pivot) * TILE_SIZE + row;
    const int tileCol = (phase == 0 ? pivot : block) * TILE_SIZE + col;
    const int pivotRow = pivot * TILE_SIZE + row;
    const int pivotCol = pivot * TILE_SIZE + col;
    const bool tileValid = tileRow < n && tileCol < n;
    const bool pivotValid = pivotRow < n && pivotCol < n;
    tile[row][col] = tileValid ? dist[tileRow * n + tileCol] : INF;
    pivotTile[row][col] = pivotValid ? dist[pivotRow * n + pivotCol] : INF;
    const unsigned int initialPath = tileValid ? path[tileRow * n + tileCol] : 0;
    unsigned int tilePath = initialPath;
    __syncthreads();

    const int limit = min(TILE_SIZE, n - pivot * TILE_SIZE);
    for (int k = 0; k < limit; ++k) {
        const unsigned int candidate = phase == 0
            ? tile[row][k] + pivotTile[k][col]
            : pivotTile[row][k] + tile[k][col];
        if (candidate < tile[row][col]) {
            tile[row][col] = candidate;
            tilePath = pivot * TILE_SIZE + k;
        }
        __syncthreads();
    }
    if (tileValid) {
        dist[tileRow * n + tileCol] = tile[row][col];
        path[tileRow * n + tileCol] = tilePath;
    }
}

__global__ void floydPhase3(unsigned int* dist, unsigned int* path, const int n,
                            const int pivot) {
    __shared__ unsigned int pivotColumn[TILE_SIZE][TILE_SIZE + 1];
    __shared__ unsigned int pivotRow[TILE_SIZE][TILE_SIZE + 1];
    const int col = threadIdx.x;
    const int row = threadIdx.y;
    const int blockRow = blockIdx.y;
    const int blockCol = blockIdx.x;
    if (blockRow == pivot || blockCol == pivot) return;

    const int globalRow = blockRow * TILE_SIZE + row;
    const int globalCol = blockCol * TILE_SIZE + col;
    const int columnCol = pivot * TILE_SIZE + col;
    const int rowRow = pivot * TILE_SIZE + row;
    const bool targetValid = globalRow < n && globalCol < n;
    unsigned int targetDistance = targetValid ? dist[globalRow * n + globalCol] : INF;
    pivotColumn[row][col] = (globalRow < n && columnCol < n)
        ? dist[globalRow * n + columnCol] : INF;
    pivotRow[row][col] = (rowRow < n && globalCol < n)
        ? dist[rowRow * n + globalCol] : INF;
    unsigned int targetPath = targetValid ? path[globalRow * n + globalCol] : 0;
    __syncthreads();

    const int limit = min(TILE_SIZE, n - pivot * TILE_SIZE);
    for (int k = 0; k < limit; ++k) {
        const unsigned int candidate = pivotColumn[row][k] + pivotRow[k][col];
        if (candidate < targetDistance) {
            targetDistance = candidate;
            targetPath = pivot * TILE_SIZE + k;
        }
    }
    if (targetValid) {
        dist[globalRow * n + globalCol] = targetDistance;
        path[globalRow * n + globalCol] = targetPath;
    }
}

double floydWarshall(std::vector<unsigned int>& dist,
                     std::vector<unsigned int>& path,
                     const size_t numNodes) {
    if (numNodes == 0) return 0.0;
    if (numNodes > static_cast<size_t>(INT_MAX)) {
        fprintf(stderr, "CUDA implementation supports at most INT_MAX nodes\n");
        std::exit(EXIT_FAILURE);
    }
    const int n = static_cast<int>(numNodes);
    const size_t bytes = numNodes * numNodes * sizeof(unsigned int);
    unsigned int* deviceDist = nullptr;
    unsigned int* devicePath = nullptr;
    checkCuda(cudaMalloc(&deviceDist, bytes), "allocating distance matrix");
    checkCuda(cudaMalloc(&devicePath, bytes), "allocating path matrix");
    checkCuda(cudaMemcpy(deviceDist, dist.data(), bytes, cudaMemcpyHostToDevice),
              "copying distance matrix to device");
    checkCuda(cudaMemcpy(devicePath, path.data(), bytes, cudaMemcpyHostToDevice),
              "copying path matrix to device");

    const int blocks = (n + TILE_SIZE - 1) / TILE_SIZE;
    const dim3 threads(TILE_SIZE, TILE_SIZE);
    cudaEvent_t startEvent;
    cudaEvent_t stopEvent;
    checkCuda(cudaEventCreate(&startEvent), "creating start event");
    checkCuda(cudaEventCreate(&stopEvent), "creating stop event");
    checkCuda(cudaEventRecord(startEvent), "recording start event");
    for (int pivot = 0; pivot < blocks; ++pivot) {
        floydPhase1<<<1, threads>>>(deviceDist, devicePath, n, pivot);
        checkCuda(cudaGetLastError(), "launching pivot phase");
        floydPhase2<<<blocks, threads>>>(deviceDist, devicePath, n, pivot, 0);
        checkCuda(cudaGetLastError(), "launching pivot-column phase");
        floydPhase2<<<blocks, threads>>>(deviceDist, devicePath, n, pivot, 1);
        checkCuda(cudaGetLastError(), "launching pivot-row phase");
        floydPhase3<<<dim3(blocks, blocks), threads>>>(deviceDist, devicePath, n, pivot);
        checkCuda(cudaGetLastError(), "launching remaining-tiles phase");
    }
    checkCuda(cudaEventRecord(stopEvent), "recording stop event");
    checkCuda(cudaEventSynchronize(stopEvent), "synchronizing Floyd-Warshall kernels");
    float elapsedMilliseconds = 0.0F;
    checkCuda(cudaEventElapsedTime(&elapsedMilliseconds, startEvent, stopEvent),
              "measuring Floyd-Warshall kernels");
    checkCuda(cudaMemcpy(dist.data(), deviceDist, bytes, cudaMemcpyDeviceToHost),
              "copying distance matrix to host");
    checkCuda(cudaMemcpy(path.data(), devicePath, bytes, cudaMemcpyDeviceToHost),
              "copying path matrix to host");
    checkCuda(cudaFree(deviceDist), "freeing distance matrix");
    checkCuda(cudaFree(devicePath), "freeing path matrix");
    checkCuda(cudaEventDestroy(startEvent), "destroying start event");
    checkCuda(cudaEventDestroy(stopEvent), "destroying stop event");
    return static_cast<double>(elapsedMilliseconds);
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
    const double elapsedMilliseconds = floydWarshall(dist, path, numNodes);
    
    printf("Computation time: %.3f ms\n", elapsedMilliseconds);
    
    // Calculate operations per second
    // Floyd-Warshall has O(n³) complexity
    double ops = (double)numNodes * numNodes * numNodes;
    double gflops = elapsedMilliseconds > 0.0
        ? ops / (elapsedMilliseconds / 1000.0) / 1e9
        : 0.0;
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
