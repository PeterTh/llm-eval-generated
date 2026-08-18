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

inline void checkCuda(const cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error during %s: %s\n", operation,
                     cudaGetErrorString(status));
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

// Each kernel performs all pivots of one tile in order.  This preserves the
// strict comparison and path semantics of the original k-by-k algorithm.
__global__ void pivotKernel(unsigned int* dist, unsigned int* path, int n,
                            int base, int pivotCount) {
    __shared__ unsigned int d[TILE_SIZE][TILE_SIZE]; // [destination][source]
    __shared__ unsigned int p[TILE_SIZE][TILE_SIZE];
    const int source = base + threadIdx.x;
    const int destination = base + threadIdx.y;
    if (threadIdx.x < pivotCount && threadIdx.y < pivotCount) {
        const size_t offset = static_cast<size_t>(source) * n + destination;
        d[threadIdx.y][threadIdx.x] = dist[offset];
        p[threadIdx.y][threadIdx.x] = path[offset];
    }
    __syncthreads();
    for (int localK = 0; localK < pivotCount; ++localK) {
        const unsigned int candidate = d[localK][threadIdx.x] + d[threadIdx.y][localK];
        if (candidate < d[threadIdx.y][threadIdx.x]) {
            d[threadIdx.y][threadIdx.x] = candidate;
            p[threadIdx.y][threadIdx.x] = base + localK;
        }
        __syncthreads();
    }
    if (threadIdx.x < pivotCount && threadIdx.y < pivotCount) {
        const size_t offset = static_cast<size_t>(source) * n + destination;
        dist[offset] = d[threadIdx.y][threadIdx.x];
        path[offset] = p[threadIdx.y][threadIdx.x];
    }
}

// Updates tiles sharing the pivot row or column.  One block owns one tile,
// so its values can be reused from shared memory through every local pivot.
__global__ void edgeKernel(unsigned int* dist, unsigned int* path, int n,
                           int base, int pivotCount, bool pivotRow) {
    __shared__ unsigned int d[TILE_SIZE][TILE_SIZE];
    __shared__ unsigned int pivot[TILE_SIZE][TILE_SIZE];
    const int localSource = threadIdx.x;
    const int localDestination = threadIdx.y;
    const int tile = static_cast<int>(blockIdx.x);
    const int sourceBase = pivotRow ? base : tile * TILE_SIZE;
    const int destinationBase = pivotRow ? tile * TILE_SIZE : base;
    const int source = sourceBase + localSource;
    const int destination = destinationBase + localDestination;
    const bool valid = source < n && destination < n;
    if (valid) d[localDestination][localSource] = dist[static_cast<size_t>(source) * n + destination];
    const int pivotSource = base + localSource;
    const int pivotDestination = base + localDestination;
    if (localSource < pivotCount && localDestination < pivotCount)
        pivot[localDestination][localSource] = dist[static_cast<size_t>(pivotSource) * n + pivotDestination];
    __syncthreads();
    for (int localK = 0; localK < pivotCount; ++localK) {
        unsigned int candidate;
        if (pivotRow)
            candidate = pivot[localK][localSource] + d[localDestination][localK];
        else
            candidate = d[localK][localSource] + pivot[localDestination][localK];
        if (valid && candidate < d[localDestination][localSource]) {
            d[localDestination][localSource] = candidate;
            path[static_cast<size_t>(source) * n + destination] = base + localK;
        }
        __syncthreads();
    }
    if (valid) dist[static_cast<size_t>(source) * n + destination] = d[localDestination][localSource];
}

// Outer tiles are independent once the two edge phases are complete.  The
// pivot column and row are staged coalesced, eliminating repeated global loads.
__global__ void outerKernel(unsigned int* dist, unsigned int* path, int n,
                            int base, int pivotCount, int pivotTile) {
    // The pivot row/column were completed in the edge phases.  Excluding
    // them is essential: outer tiles read those values concurrently.
    if (static_cast<int>(blockIdx.x) == pivotTile ||
        static_cast<int>(blockIdx.y) == pivotTile) return;
    __shared__ unsigned int column[TILE_SIZE][TILE_SIZE]; // [k][source]
    __shared__ unsigned int row[TILE_SIZE][TILE_SIZE];    // [destination][k]
    const int sourceBase = blockIdx.x * TILE_SIZE;
    const int destinationBase = blockIdx.y * TILE_SIZE;
    const int source = sourceBase + threadIdx.x;
    const int destination = destinationBase + threadIdx.y;
    const bool valid = source < n && destination < n;
    if (threadIdx.y < pivotCount && source < n)
        column[threadIdx.y][threadIdx.x] = dist[static_cast<size_t>(source) * n + base + threadIdx.y];
    // Linear threads load a contiguous segment of each pivot-row column.
    const int linear = threadIdx.y * TILE_SIZE + threadIdx.x;
    const int k = linear % TILE_SIZE;
    const int destLocal = linear / TILE_SIZE;
    if (k < pivotCount && destinationBase + destLocal < n)
        row[destLocal][k] = dist[static_cast<size_t>(base + k) * n + destinationBase + destLocal];
    __syncthreads();
    if (!valid) return;
    unsigned int best = dist[static_cast<size_t>(source) * n + destination];
    unsigned int bestPath = path[static_cast<size_t>(source) * n + destination];
    for (int localK = 0; localK < pivotCount; ++localK) {
        const unsigned int candidate = column[localK][threadIdx.x] + row[threadIdx.y][localK];
        if (candidate < best) {
            best = candidate;
            bestPath = base + localK;
        }
    }
    dist[static_cast<size_t>(source) * n + destination] = best;
    path[static_cast<size_t>(source) * n + destination] = bestPath;
}

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes) {
    if (numNodes == 0) return;
    if (numNodes > static_cast<size_t>(INT_MAX)) {
        std::fprintf(stderr, "CUDA implementation requires at most INT_MAX nodes\n");
        std::exit(EXIT_FAILURE);
    }
    const int n = static_cast<int>(numNodes);
    const size_t bytes = numNodes * numNodes * sizeof(unsigned int);
    unsigned int *deviceDist = nullptr, *devicePath = nullptr;
    checkCuda(cudaMalloc(&deviceDist, bytes), "distance allocation");
    checkCuda(cudaMalloc(&devicePath, bytes), "path allocation");
    checkCuda(cudaMemcpy(deviceDist, dist.data(), bytes, cudaMemcpyHostToDevice), "distance upload");
    checkCuda(cudaMemcpy(devicePath, path.data(), bytes, cudaMemcpyHostToDevice), "path upload");

    const dim3 threads(TILE_SIZE, TILE_SIZE);
    const int tiles = (n + TILE_SIZE - 1) / TILE_SIZE;
    for (int pivotTile = 0; pivotTile < tiles; ++pivotTile) {
        const int base = pivotTile * TILE_SIZE;
        const int pivotCount = std::min(TILE_SIZE, n - base);
        pivotKernel<<<1, threads>>>(deviceDist, devicePath, n, base, pivotCount);
        edgeKernel<<<tiles, threads>>>(deviceDist, devicePath, n, base, pivotCount, true);
        edgeKernel<<<tiles, threads>>>(deviceDist, devicePath, n, base, pivotCount, false);
        outerKernel<<<dim3(tiles, tiles), threads>>>(deviceDist, devicePath, n, base, pivotCount, pivotTile);
        checkCuda(cudaGetLastError(), "Floyd-Warshall kernel launch");
    }
    checkCuda(cudaDeviceSynchronize(), "Floyd-Warshall execution");
    checkCuda(cudaMemcpy(dist.data(), deviceDist, bytes, cudaMemcpyDeviceToHost), "distance download");
    checkCuda(cudaMemcpy(path.data(), devicePath, bytes, cudaMemcpyDeviceToHost), "path download");
    checkCuda(cudaFree(deviceDist), "distance release");
    checkCuda(cudaFree(devicePath), "path release");
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
    // Keep one-time CUDA context construction outside the measured algorithm.
    checkCuda(cudaFree(nullptr), "CUDA initialization");
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
