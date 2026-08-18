#include <algorithm>
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

constexpr unsigned int TILE_SIZE = 32;

void checkCuda(const cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error in %s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

// Close the diagonal tile for the current group of intermediate vertices.
__global__ void updatePivotTile(unsigned int* __restrict__ dist,
                                unsigned int* __restrict__ path,
                                const unsigned int n,
                                const unsigned int round) {
    __shared__ unsigned int tile[TILE_SIZE][TILE_SIZE + 1];

    const unsigned int x = threadIdx.x;
    const unsigned int y = threadIdx.y;
    const unsigned int col = round * TILE_SIZE + x;
    const unsigned int row = round * TILE_SIZE + y;

    tile[y][x] = (row < n && col < n) ? dist[row * n + col] : INF;
    __syncthreads();

    unsigned int bestK = 0;
    bool changed = false;
#pragma unroll
    for (unsigned int k = 0; k < TILE_SIZE; ++k) {
        const unsigned int candidate = tile[y][k] + tile[k][x];
        if (candidate < tile[y][x]) {
            tile[y][x] = candidate;
            bestK = round * TILE_SIZE + k;
            changed = true;
        }
        __syncthreads();
    }

    if (row < n && col < n) {
        dist[row * n + col] = tile[y][x];
        if (changed) {
            path[row * n + col] = bestK;
        }
    }
}

// Update every tile in the pivot row and pivot column. Both directions are
// independent after the diagonal tile has been closed, so one launch does both.
__global__ void updatePivotRowAndColumn(unsigned int* __restrict__ dist,
                                       unsigned int* __restrict__ path,
                                       const unsigned int n,
                                       const unsigned int round) {
    __shared__ unsigned int pivot[TILE_SIZE][TILE_SIZE + 1];
    __shared__ unsigned int current[TILE_SIZE][TILE_SIZE + 1];

    const unsigned int tileIndex = blockIdx.x;
    if (tileIndex == round) {
        return;
    }

    const unsigned int x = threadIdx.x;
    const unsigned int y = threadIdx.y;
    const bool updateRow = blockIdx.y == 0;
    const unsigned int pivotCol = round * TILE_SIZE + x;
    const unsigned int pivotRow = round * TILE_SIZE + y;
    const unsigned int col = (updateRow ? tileIndex : round) * TILE_SIZE + x;
    const unsigned int row = (updateRow ? round : tileIndex) * TILE_SIZE + y;

    pivot[y][x] = (pivotRow < n && pivotCol < n)
                      ? dist[pivotRow * n + pivotCol]
                      : INF;
    current[y][x] = (row < n && col < n) ? dist[row * n + col] : INF;
    __syncthreads();

    unsigned int bestK = 0;
    bool changed = false;
#pragma unroll
    for (unsigned int k = 0; k < TILE_SIZE; ++k) {
        const unsigned int candidate = updateRow
                                           ? pivot[y][k] + current[k][x]
                                           : current[y][k] + pivot[k][x];
        if (candidate < current[y][x]) {
            current[y][x] = candidate;
            bestK = round * TILE_SIZE + k;
            changed = true;
        }
        __syncthreads();
    }

    if (row < n && col < n) {
        dist[row * n + col] = current[y][x];
        if (changed) {
            path[row * n + col] = bestK;
        }
    }
}

// Update all remaining tiles. Each block reuses two input strips from shared
// memory for 32 intermediate vertices, giving O(TILE_SIZE) arithmetic per load.
__global__ void updateRemainingTiles(unsigned int* __restrict__ dist,
                                     unsigned int* __restrict__ path,
                                     const unsigned int n,
                                     const unsigned int round) {
    const unsigned int tileCol = blockIdx.x;
    const unsigned int tileRow = blockIdx.y;
    if (tileCol == round || tileRow == round) {
        return;
    }

    __shared__ unsigned int left[TILE_SIZE][TILE_SIZE + 1];
    __shared__ unsigned int top[TILE_SIZE][TILE_SIZE + 1];

    const unsigned int x = threadIdx.x;
    const unsigned int y = threadIdx.y;
    const unsigned int col = tileCol * TILE_SIZE + x;
    const unsigned int row = tileRow * TILE_SIZE + y;
    const unsigned int pivotCol = round * TILE_SIZE + x;
    const unsigned int pivotRow = round * TILE_SIZE + y;

    left[y][x] = (row < n && pivotCol < n) ? dist[row * n + pivotCol] : INF;
    top[y][x] = (pivotRow < n && col < n) ? dist[pivotRow * n + col] : INF;
    __syncthreads();

    unsigned int best = (row < n && col < n) ? dist[row * n + col] : INF;
    unsigned int bestK = 0;
    bool changed = false;
#pragma unroll
    for (unsigned int k = 0; k < TILE_SIZE; ++k) {
        const unsigned int candidate = left[y][k] + top[k][x];
        if (candidate < best) {
            best = candidate;
            bestK = round * TILE_SIZE + k;
            changed = true;
        }
    }

    if (row < n && col < n) {
        dist[row * n + col] = best;
        if (changed) {
            path[row * n + col] = bestK;
        }
    }
}

} // namespace

float floydWarshall(std::vector<unsigned int>& dist,
                    std::vector<unsigned int>& path,
                    const size_t numNodes) {
    const unsigned int n = static_cast<unsigned int>(numNodes);
    const size_t bytes = numNodes * numNodes * sizeof(unsigned int);
    unsigned int* deviceDist = nullptr;
    unsigned int* devicePath = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    checkCuda(cudaMalloc(&deviceDist, bytes), "allocating the distance matrix");
    checkCuda(cudaMalloc(&devicePath, bytes), "allocating the path matrix");
    checkCuda(cudaMemcpy(deviceDist, dist.data(), bytes, cudaMemcpyHostToDevice),
              "copying the distance matrix to the GPU");
    checkCuda(cudaMemcpy(devicePath, path.data(), bytes, cudaMemcpyHostToDevice),
              "copying the path matrix to the GPU");
    checkCuda(cudaEventCreate(&start), "creating the start event");
    checkCuda(cudaEventCreate(&stop), "creating the stop event");

    const unsigned int tileCount = (n + TILE_SIZE - 1) / TILE_SIZE;
    const dim3 threads(TILE_SIZE, TILE_SIZE);
    checkCuda(cudaEventRecord(start), "recording the start event");
    for (unsigned int round = 0; round < tileCount; ++round) {
        updatePivotTile<<<1, threads>>>(deviceDist, devicePath, n, round);
        updatePivotRowAndColumn<<<dim3(tileCount, 2), threads>>>(
            deviceDist, devicePath, n, round);
        updateRemainingTiles<<<dim3(tileCount, tileCount), threads>>>(
            deviceDist, devicePath, n, round);
    }
    checkCuda(cudaGetLastError(), "launching Floyd-Warshall kernels");
    checkCuda(cudaEventRecord(stop), "recording the stop event");
    checkCuda(cudaEventSynchronize(stop), "executing Floyd-Warshall kernels");

    float elapsedMs = 0.0F;
    checkCuda(cudaEventElapsedTime(&elapsedMs, start, stop), "measuring GPU execution");
    checkCuda(cudaMemcpy(dist.data(), deviceDist, bytes, cudaMemcpyDeviceToHost),
              "copying the distance matrix from the GPU");
    checkCuda(cudaMemcpy(path.data(), devicePath, bytes, cudaMemcpyDeviceToHost),
              "copying the path matrix from the GPU");

    checkCuda(cudaEventDestroy(start), "destroying the start event");
    checkCuda(cudaEventDestroy(stop), "destroying the stop event");
    checkCuda(cudaFree(deviceDist), "freeing the distance matrix");
    checkCuda(cudaFree(devicePath), "freeing the path matrix");
    return elapsedMs;
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
    
    if (numNodes == 0 || numNodes > std::numeric_limits<unsigned int>::max() ||
        numNodes > std::numeric_limits<size_t>::max() / numNodes / sizeof(unsigned int)) {
        std::fprintf(stderr, "Number of nodes is out of range\n");
        return 1;
    }

    // Allocate matrices
    std::vector<unsigned int> dist(numNodes * numNodes);
    std::vector<unsigned int> path(numNodes * numNodes);
    
    // Initialize
    printf("Initializing graph...\n");
    initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
    initializePathMatrix(path, numNodes);
    
    // Run Floyd-Warshall
    printf("Computing shortest paths...\n");
    const float durationMs = floydWarshall(dist, path, numNodes);

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
