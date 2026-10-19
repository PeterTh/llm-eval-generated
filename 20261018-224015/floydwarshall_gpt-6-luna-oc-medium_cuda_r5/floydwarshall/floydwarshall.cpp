#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
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

constexpr unsigned int TILE = 32;

inline void cudaCheck(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

__global__ void fwDiagonal(unsigned int* d, unsigned int* p, size_t n, size_t base, size_t width) {
    __shared__ unsigned int a[TILE][TILE];
    __shared__ unsigned int q[TILE][TILE];
    const unsigned int x = threadIdx.x, y = threadIdx.y;
    const size_t row = base + y, col = base + x;
    a[y][x] = row < n && col < n ? d[col * n + row] : INF;
    q[y][x] = row < n && col < n ? p[col * n + row] : 0;
    __syncthreads();
    for (unsigned int k = 0; k < width; ++k) {
        unsigned int candidate = a[y][k] + a[k][x];
        if (candidate < a[y][x]) { a[y][x] = candidate; q[y][x] = q[k][x]; }
        __syncthreads();
    }
    if (row < n && col < n) { d[col * n + row] = a[y][x]; p[col * n + row] = q[y][x]; }
}

__global__ void fwRowCol(unsigned int* d, unsigned int* p, size_t n, size_t base,
                         size_t width, size_t tileCount) {
    const size_t tile = blockIdx.x;
    if (tile >= tileCount || tile == base / TILE) return;
    __shared__ unsigned int pivot[TILE][TILE];
    const unsigned int x = threadIdx.x, y = threadIdx.y;
    const size_t pivotBase = base, otherBase = tile * TILE;
    const bool rowPhase = blockIdx.y == 0;
    const size_t row = rowPhase ? pivotBase + y : otherBase + y;
    const size_t col = rowPhase ? otherBase + x : pivotBase + x;
    const size_t pr = pivotBase + y, pc = pivotBase + x;
    pivot[y][x] = pr < n && pc < n ? d[pc * n + pr] : INF;
    __syncthreads();
    if (row >= n || col >= n) return;
    unsigned int value = d[col * n + row], pathValue = p[col * n + row];
    for (unsigned int k = 0; k < width; ++k) {
        unsigned int left, right, candidate;
        if (rowPhase) { left = pivot[y][k]; right = d[col * n + (pivotBase + k)]; }
        else { left = d[(pivotBase + k) * n + row]; right = pivot[k][x]; }
        candidate = left + right;
        if (candidate < value) { value = candidate; pathValue = static_cast<unsigned int>(pivotBase + k); }
    }
    d[col * n + row] = value; p[col * n + row] = pathValue;
}

__global__ void fwRemainder(unsigned int* d, unsigned int* p, size_t n, size_t base,
                            size_t width, size_t tileCount) {
    const size_t tileRow = blockIdx.y, tileCol = blockIdx.x;
    if (tileRow == base / TILE || tileCol == base / TILE) return;
    __shared__ unsigned int rowTile[TILE][TILE], colTile[TILE][TILE];
    const unsigned int x = threadIdx.x, y = threadIdx.y;
    const size_t row = tileRow * TILE + y, col = tileCol * TILE + x;
    const size_t krow = base + y, kcol = base + x;
    rowTile[y][x] = row < n && kcol < n ? d[kcol * n + row] : INF;
    colTile[y][x] = krow < n && col < n ? d[col * n + krow] : INF;
    __syncthreads();
    if (row >= n || col >= n) return;
    unsigned int value = d[col * n + row], pathValue = p[col * n + row];
    for (unsigned int k = 0; k < width; ++k) {
        unsigned int candidate = rowTile[k][y] + colTile[x][k];
        if (candidate < value) { value = candidate; pathValue = static_cast<unsigned int>(base + k); }
    }
    d[col * n + row] = value; p[col * n + row] = pathValue;
}

void floydWarshall(std::vector<unsigned int>& dist, std::vector<unsigned int>& path, size_t numNodes) {
    if (numNodes == 0) return;
    unsigned int *deviceDist = nullptr, *devicePath = nullptr;
    const size_t bytes = numNodes * numNodes * sizeof(unsigned int);
    cudaCheck(cudaMalloc(&deviceDist, bytes));
    try {
        cudaCheck(cudaMalloc(&devicePath, bytes));
        cudaCheck(cudaMemcpy(deviceDist, dist.data(), bytes, cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(devicePath, path.data(), bytes, cudaMemcpyHostToDevice));
        const size_t tiles = (numNodes + TILE - 1) / TILE;
        const dim3 threads(TILE, TILE);
        for (size_t base = 0; base < numNodes; base += TILE) {
            const size_t width = std::min<size_t>(TILE, numNodes - base);
            fwDiagonal<<<1, threads>>>(deviceDist, devicePath, numNodes, base, width);
            fwRowCol<<<dim3(tiles, 2), threads>>>(deviceDist, devicePath, numNodes, base, width, tiles);
            fwRemainder<<<dim3(tiles, tiles), threads>>>(deviceDist, devicePath, numNodes, base, width, tiles);
        }
        cudaCheck(cudaGetLastError());
        cudaCheck(cudaMemcpy(dist.data(), deviceDist, bytes, cudaMemcpyDeviceToHost));
        cudaCheck(cudaMemcpy(path.data(), devicePath, bytes, cudaMemcpyDeviceToHost));
        cudaFree(devicePath); cudaFree(deviceDist);
    } catch (...) { if (devicePath) cudaFree(devicePath); cudaFree(deviceDist); throw; }
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
