#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;
constexpr unsigned int TILE_SIZE = 32;

#define CUDA_CHECK(call) do { \
    const cudaError_t error = (call); \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

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

// Matrices are column-major: element (source i, destination j) is j * n + i.
// A 32x32 tile maps one element to each CUDA thread.
__global__ void pivotKernel(unsigned int* dist, unsigned int* path, const size_t n,
                            const unsigned int pivot) {
    __shared__ unsigned int tileDist[TILE_SIZE * TILE_SIZE];
    __shared__ unsigned int tilePath[TILE_SIZE * TILE_SIZE];
    const unsigned int x = threadIdx.x;
    const unsigned int y = threadIdx.y;
    const unsigned int local = y * TILE_SIZE + x;
    const size_t i = static_cast<size_t>(pivot) * TILE_SIZE + x;
    const size_t j = static_cast<size_t>(pivot) * TILE_SIZE + y;
    const size_t global = j * n + i;
    tileDist[local] = dist[global];
    tilePath[local] = path[global];
    __syncthreads();

    for (unsigned int k = 0; k < TILE_SIZE; ++k) {
        const unsigned int candidate = tileDist[k * TILE_SIZE + x] +
                                       tileDist[y * TILE_SIZE + k];
        if (candidate < tileDist[local]) {
            tileDist[local] = candidate;
            tilePath[local] = static_cast<unsigned int>(pivot * TILE_SIZE + k);
        }
        __syncthreads();
    }
    dist[global] = tileDist[local];
    path[global] = tilePath[local];
}

__global__ void pivotRowColumnKernel(unsigned int* dist, unsigned int* path, const size_t n,
                                     const unsigned int pivot) {
    __shared__ unsigned int pivotDist[TILE_SIZE * TILE_SIZE];
    __shared__ unsigned int tileDist[TILE_SIZE * TILE_SIZE];
    __shared__ unsigned int tilePath[TILE_SIZE * TILE_SIZE];
    const unsigned int x = threadIdx.x;
    const unsigned int y = threadIdx.y;
    const unsigned int local = y * TILE_SIZE + x;
    const unsigned int tile = blockIdx.x;
    const bool isRow = blockIdx.y == 0;
    if (tile == pivot) return;
    const size_t pivotBase = static_cast<size_t>(pivot) * TILE_SIZE;
    const size_t otherBase = static_cast<size_t>(tile) * TILE_SIZE;
    const size_t pivotIndex = (pivotBase + y) * n + pivotBase + x;
    pivotDist[local] = dist[pivotIndex];
    const size_t i = isRow ? pivotBase + x : otherBase + x;
    const size_t j = isRow ? otherBase + y : pivotBase + y;
    const size_t global = j * n + i;
    tileDist[local] = dist[global];
    tilePath[local] = path[global];
    __syncthreads();

    for (unsigned int k = 0; k < TILE_SIZE; ++k) {
        const unsigned int candidate = isRow
            ? pivotDist[k * TILE_SIZE + x] + tileDist[y * TILE_SIZE + k]
            : tileDist[k * TILE_SIZE + x] + pivotDist[y * TILE_SIZE + k];
        if (candidate < tileDist[local]) {
            tileDist[local] = candidate;
            tilePath[local] = static_cast<unsigned int>(pivotBase + k);
        }
        __syncthreads();
    }
    dist[global] = tileDist[local];
    path[global] = tilePath[local];
}

__global__ void remainingTilesKernel(unsigned int* dist, unsigned int* path, const size_t n,
                                     const unsigned int pivot) {
    const unsigned int rowTile = blockIdx.x;
    const unsigned int colTile = blockIdx.y;
    if (rowTile == pivot || colTile == pivot) return;

    __shared__ unsigned int left[TILE_SIZE * TILE_SIZE];
    __shared__ unsigned int right[TILE_SIZE * TILE_SIZE];
    __shared__ unsigned int tileDist[TILE_SIZE * TILE_SIZE];
    __shared__ unsigned int tilePath[TILE_SIZE * TILE_SIZE];
    const unsigned int x = threadIdx.x;
    const unsigned int y = threadIdx.y;
    const unsigned int local = y * TILE_SIZE + x;
    const size_t i = static_cast<size_t>(rowTile) * TILE_SIZE + x;
    const size_t j = static_cast<size_t>(colTile) * TILE_SIZE + y;
    const size_t pivotBase = static_cast<size_t>(pivot) * TILE_SIZE;
    left[local] = dist[(pivotBase + y) * n + i];       // D(i, pivot + y)
    right[local] = dist[j * n + pivotBase + x];        // D(pivot + x, j)
    const size_t global = j * n + i;
    tileDist[local] = dist[global];
    tilePath[local] = path[global];
    __syncthreads();

    for (unsigned int k = 0; k < TILE_SIZE; ++k) {
        const unsigned int candidate = left[k * TILE_SIZE + x] +
                                       right[y * TILE_SIZE + k];
        if (candidate < tileDist[local]) {
            tileDist[local] = candidate;
            tilePath[local] = static_cast<unsigned int>(pivotBase + k);
        }
        __syncthreads();
    }
    dist[global] = tileDist[local];
    path[global] = tilePath[local];
}

void floydWarshall(std::vector<unsigned int>& dist, std::vector<unsigned int>& path,
                   const size_t numNodes, float& elapsedMs) {
    if (numNodes == 0) {
        elapsedMs = 0.0F;
        return;
    }
    const size_t paddedNodes = ((numNodes + TILE_SIZE - 1) / TILE_SIZE) * TILE_SIZE;
    const size_t paddedElements = paddedNodes * paddedNodes;
    std::vector<unsigned int> paddedDist(paddedElements, INF);
    std::vector<unsigned int> paddedPath(paddedElements, 0);
    for (size_t j = 0; j < numNodes; ++j) {
        std::memcpy(&paddedDist[j * paddedNodes], &dist[j * numNodes], numNodes * sizeof(unsigned int));
        std::memcpy(&paddedPath[j * paddedNodes], &path[j * numNodes], numNodes * sizeof(unsigned int));
    }

    unsigned int* deviceDist = nullptr;
    unsigned int* devicePath = nullptr;
    CUDA_CHECK(cudaMalloc(&deviceDist, paddedElements * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&devicePath, paddedElements * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpy(deviceDist, paddedDist.data(), paddedElements * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(devicePath, paddedPath.data(), paddedElements * sizeof(unsigned int), cudaMemcpyHostToDevice));

    cudaEvent_t start, end;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));
    CUDA_CHECK(cudaEventRecord(start));
    const unsigned int tileCount = static_cast<unsigned int>(paddedNodes / TILE_SIZE);
    const dim3 threads(TILE_SIZE, TILE_SIZE);
    for (unsigned int pivot = 0; pivot < tileCount; ++pivot) {
        pivotKernel<<<1, threads>>>(deviceDist, devicePath, paddedNodes, pivot);
        pivotRowColumnKernel<<<dim3(tileCount, 2), threads>>>(deviceDist, devicePath, paddedNodes, pivot);
        remainingTilesKernel<<<dim3(tileCount, tileCount), threads>>>(deviceDist, devicePath, paddedNodes, pivot);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));
    CUDA_CHECK(cudaEventElapsedTime(&elapsedMs, start, end));

    CUDA_CHECK(cudaMemcpy(paddedDist.data(), deviceDist, paddedElements * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(paddedPath.data(), devicePath, paddedElements * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    for (size_t j = 0; j < numNodes; ++j) {
        std::memcpy(&dist[j * numNodes], &paddedDist[j * paddedNodes], numNodes * sizeof(unsigned int));
        std::memcpy(&path[j * numNodes], &paddedPath[j * paddedNodes], numNodes * sizeof(unsigned int));
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(end));
    CUDA_CHECK(cudaFree(deviceDist));
    CUDA_CHECK(cudaFree(devicePath));
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
    float elapsedMs = 0.0F;
    floydWarshall(dist, path, numNodes, elapsedMs);
    
    printf("Computation time: %.3f ms\n", elapsedMs);
    
    // Calculate operations per second
    // Floyd-Warshall has O(n³) complexity
    double ops = (double)numNodes * numNodes * numNodes;
    double gflops = elapsedMs > 0.0F ? ops / (elapsedMs / 1000.0) / 1e9 : 0.0;
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
