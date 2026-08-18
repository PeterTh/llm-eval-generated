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
constexpr int TILE_SIZE = 32;

[[noreturn]] void cudaCheckFailed(const cudaError_t error, const char* expression,
                                  const char* file, const int line) {
    std::fprintf(stderr, "CUDA error at %s:%d while executing %s: %s\n", file, line,
                 expression, cudaGetErrorString(error));
    std::exit(EXIT_FAILURE);
}

#define CUDA_CHECK(expression)                                                        \
    do {                                                                              \
        const cudaError_t cudaStatus = (expression);                                  \
        if (cudaStatus != cudaSuccess) {                                              \
            cudaCheckFailed(cudaStatus, #expression, __FILE__, __LINE__);             \
        }                                                                             \
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

// Matrices are stored as matrix[destination * n + source].  Mapping source to x
// makes global loads and stores coalesced while retaining the original layout.
__global__ void phase1Kernel(unsigned int* dist, unsigned int* path, const int n,
                             const int pivotBase) {
    __shared__ unsigned int pivot[TILE_SIZE][TILE_SIZE];
    __shared__ unsigned int pivotPath[TILE_SIZE][TILE_SIZE];
    const int source = threadIdx.x;
    const int destination = threadIdx.y;
    const int globalSource = pivotBase + source;
    const int globalDestination = pivotBase + destination;
    const bool valid = globalSource < n && globalDestination < n;
    pivot[source][destination] = valid ? dist[globalDestination * n + globalSource] : INF;
    pivotPath[source][destination] = valid ? path[globalDestination * n + globalSource] : 0;
    __syncthreads();

    const int limit = min(TILE_SIZE, n - pivotBase);
    for (int k = 0; k < limit; ++k) {
        const unsigned int candidate = pivot[source][k] + pivot[k][destination];
        if (valid && candidate < pivot[source][destination]) {
            pivot[source][destination] = candidate;
            pivotPath[source][destination] = pivotBase + k;
        }
        __syncthreads();
    }
    if (valid) {
        dist[globalDestination * n + globalSource] = pivot[source][destination];
        path[globalDestination * n + globalSource] = pivotPath[source][destination];
    }
}

// Updates either a pivot-row tile (kind 0) or a pivot-column tile (kind 1).
__global__ void phase2Kernel(unsigned int* dist, unsigned int* path, const int n,
                             const int pivotBase, const int pivotTile, const int kind) {
    __shared__ unsigned int pivot[TILE_SIZE][TILE_SIZE];
    __shared__ unsigned int current[TILE_SIZE][TILE_SIZE];
    const int source = threadIdx.x;
    const int destination = threadIdx.y;
    const int compactTile = static_cast<int>(blockIdx.x);
    const int tile = compactTile + (compactTile >= pivotTile);
    const int otherBase = tile * TILE_SIZE;
    const int pivotSource = pivotBase + source;
    const int pivotDestination = pivotBase + destination;
    const bool pivotValid = pivotSource < n && pivotDestination < n;
    pivot[source][destination] = pivotValid ? dist[pivotDestination * n + pivotSource] : INF;

    const int globalSource = kind == 0 ? pivotSource : otherBase + source;
    const int globalDestination = kind == 0 ? otherBase + destination : pivotDestination;
    const bool valid = globalSource < n && globalDestination < n;
    current[source][destination] = valid ? dist[globalDestination * n + globalSource] : INF;
    __syncthreads();

    const int limit = min(TILE_SIZE, n - pivotBase);
    for (int k = 0; k < limit; ++k) {
        const unsigned int candidate = kind == 0
            ? pivot[source][k] + current[k][destination]
            : current[source][k] + pivot[k][destination];
        if (valid && candidate < current[source][destination]) {
            current[source][destination] = candidate;
            path[globalDestination * n + globalSource] = pivotBase + k;
        }
        __syncthreads();
    }
    if (valid) {
        dist[globalDestination * n + globalSource] = current[source][destination];
    }
}

__global__ void phase3Kernel(unsigned int* dist, unsigned int* path, const int n,
                             const int pivotBase) {
    __shared__ unsigned int row[TILE_SIZE][TILE_SIZE];
    __shared__ unsigned int column[TILE_SIZE][TILE_SIZE];
    __shared__ unsigned int current[TILE_SIZE][TILE_SIZE];
    const int source = threadIdx.x;
    const int destination = threadIdx.y;
    const int compactSourceTile = static_cast<int>(blockIdx.x);
    const int compactDestinationTile = static_cast<int>(blockIdx.y);
    const int pivotTile = pivotBase / TILE_SIZE;
    const int sourceTile = compactSourceTile + (compactSourceTile >= pivotTile);
    const int destinationTile = compactDestinationTile +
                                (compactDestinationTile >= pivotTile);
    const int sourceBase = sourceTile * TILE_SIZE;
    const int destinationBase = destinationTile * TILE_SIZE;
    const int globalSource = sourceBase + source;
    const int globalDestination = destinationBase + destination;
    const bool valid = globalSource < n && globalDestination < n;

    row[source][destination] = (globalSource < n && pivotBase + destination < n)
        ? dist[(pivotBase + destination) * n + globalSource] : INF;
    column[source][destination] = (pivotBase + source < n && globalDestination < n)
        ? dist[globalDestination * n + pivotBase + source] : INF;
    current[source][destination] = valid ? dist[globalDestination * n + globalSource] : INF;
    __syncthreads();

    const int limit = min(TILE_SIZE, n - pivotBase);
    for (int k = 0; k < limit; ++k) {
        const unsigned int candidate = row[source][k] + column[k][destination];
        if (valid && candidate < current[source][destination]) {
            current[source][destination] = candidate;
            path[globalDestination * n + globalSource] = pivotBase + k;
        }
        __syncthreads();
    }
    if (valid) {
        dist[globalDestination * n + globalSource] = current[source][destination];
    }
}

float floydWarshall(std::vector<unsigned int>& dist, std::vector<unsigned int>& path,
                    const size_t numNodes) {
    if (numNodes == 0) {
        return 0.0F;
    }
    if (numNodes > static_cast<size_t>(INT_MAX)) {
        std::fprintf(stderr, "Number of nodes exceeds CUDA index range\n");
        std::exit(EXIT_FAILURE);
    }
    const int n = static_cast<int>(numNodes);
    const size_t bytes = numNodes * numNodes * sizeof(unsigned int);
    unsigned int* deviceDist = nullptr;
    unsigned int* devicePath = nullptr;
    CUDA_CHECK(cudaMalloc(&deviceDist, bytes));
    CUDA_CHECK(cudaMalloc(&devicePath, bytes));
    CUDA_CHECK(cudaMemcpy(deviceDist, dist.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(devicePath, path.data(), bytes, cudaMemcpyHostToDevice));

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    const dim3 threads(TILE_SIZE, TILE_SIZE);
    const int tiles = (n + TILE_SIZE - 1) / TILE_SIZE;
    CUDA_CHECK(cudaEventRecord(start));
    for (int pivot = 0; pivot < tiles; ++pivot) {
        const int pivotBase = pivot * TILE_SIZE;
        phase1Kernel<<<1, threads>>>(deviceDist, devicePath, n, pivotBase);
        CUDA_CHECK(cudaGetLastError());
        if (tiles > 1) {
            phase2Kernel<<<tiles - 1, threads>>>(deviceDist, devicePath, n, pivotBase, pivot, 0);
            CUDA_CHECK(cudaGetLastError());
            phase2Kernel<<<tiles - 1, threads>>>(deviceDist, devicePath, n, pivotBase, pivot, 1);
            CUDA_CHECK(cudaGetLastError());
            phase3Kernel<<<dim3(tiles - 1, tiles - 1), threads>>>(deviceDist, devicePath, n, pivotBase);
            CUDA_CHECK(cudaGetLastError());
        }
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float milliseconds = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    CUDA_CHECK(cudaMemcpy(dist.data(), deviceDist, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(path.data(), devicePath, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(deviceDist));
    CUDA_CHECK(cudaFree(devicePath));
    return milliseconds;
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
    const float milliseconds = floydWarshall(dist, path, numNodes);
    printf("Computation time: %.3f ms\n", milliseconds);
    
    // Calculate operations per second
    // Floyd-Warshall has O(n³) complexity
    double ops = (double)numNodes * numNodes * numNodes;
    double gflops = milliseconds > 0.0F ? ops / (milliseconds / 1000.0) / 1e9 : 0.0;
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
