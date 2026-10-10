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
constexpr int TILE = 32;
constexpr int ROWS_PER_THREAD = 4;

void checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(result));
        std::exit(EXIT_FAILURE);
    }
}

// Record the pivot row and column BEFORE each successive k update. These
// snapshots preserve the scalar algorithm's strict tie and path semantics.
__global__ void pivotSnapshots(const unsigned int* dist, unsigned int* pivots,
                               size_t n, size_t base, int count) {
    __shared__ unsigned int tile[TILE][TILE];
    const int x = threadIdx.x;
    const int y = threadIdx.y;
    #pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; ++r) {
        const int row = y + r * 8;
        tile[row][x] = base + row < n && base + x < n
            ? dist[(base + row) * n + base + x] : INF;
    }
    __syncthreads();
    for (int k = 0; k < count; ++k) {
        if (y == 0) {
            pivots[k * TILE + x] = tile[k][x];
            pivots[TILE * TILE + k * TILE + x] = tile[x][k];
        }
        unsigned int candidate[ROWS_PER_THREAD];
        #pragma unroll
        for (int r = 0; r < ROWS_PER_THREAD; ++r)
            candidate[r] = tile[y + r * 8][k] + tile[k][x];
        __syncthreads();
        #pragma unroll
        for (int r = 0; r < ROWS_PER_THREAD; ++r)
            tile[y + r * 8][x] = min(tile[y + r * 8][x], candidate[r]);
        __syncthreads();
    }
}

// Each panel evolves independently using the recorded diagonal pivots.
// The two outputs contain d[i,k] and d[k,j] at the original algorithm's k.
__global__ void panelSnapshots(const unsigned int* dist,
                               const unsigned int* pivots,
                               unsigned int* left, unsigned int* right,
                               size_t n, size_t base, int count) {
    __shared__ unsigned int tile[TILE][TILE];
    __shared__ unsigned int pivot[TILE][TILE];
    const int x = threadIdx.x;
    const int y = threadIdx.y;
    const bool columnPanel = blockIdx.y == 0;
    const size_t offset = static_cast<size_t>(blockIdx.x) * TILE;
    #pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; ++r) {
        const int row = y + r * 8;
        const size_t i = (columnPanel ? offset : base) + row;
        const size_t j = (columnPanel ? base : offset) + x;
        tile[row][x] = i < n && j < n ? dist[i * n + j] : INF;
        pivot[row][x] = row < count
            ? pivots[(columnPanel ? 0 : TILE * TILE) + row * TILE + x] : INF;
    }
    __syncthreads();
    for (int k = 0; k < count; ++k) {
        if (y == 0 && offset + x < n) {
            if (columnPanel) left[k * n + offset + x] = tile[x][k];
            else right[k * n + offset + x] = tile[k][x];
        }
        unsigned int candidate[ROWS_PER_THREAD];
        #pragma unroll
        for (int r = 0; r < ROWS_PER_THREAD; ++r) {
            const int row = y + r * 8;
            candidate[r] = columnPanel ? tile[row][k] + pivot[k][x]
                                       : pivot[k][row] + tile[k][x];
        }
        __syncthreads();
        #pragma unroll
        for (int r = 0; r < ROWS_PER_THREAD; ++r)
            tile[y + r * 8][x] = min(tile[y + r * 8][x], candidate[r]);
        __syncthreads();
    }
}

// Coalesced global accesses, shared pivot reuse, and four register-resident
// outputs per thread amortize global memory traffic over 32 intermediate nodes.
__global__ void updateTiles(unsigned int* dist, unsigned int* path,
                            const unsigned int* left, const unsigned int* right,
                            size_t n, size_t base, int count) {
    __shared__ unsigned int l[TILE][TILE + 1];
    __shared__ unsigned int rt[TILE][TILE];
    const int x = threadIdx.x;
    const int y = threadIdx.y;
    const size_t rowBase = static_cast<size_t>(blockIdx.y) * TILE;
    const size_t colBase = static_cast<size_t>(blockIdx.x) * TILE;
    unsigned int value[ROWS_PER_THREAD], via[ROWS_PER_THREAD];
    #pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; ++r) {
        const int row = y + r * 8;
        l[row][x] = row < count && rowBase + x < n ? left[row * n + rowBase + x] : INF;
        rt[row][x] = row < count && colBase + x < n ? right[row * n + colBase + x] : INF;
        const size_t index = (rowBase + row) * n + colBase + x;
        const bool valid = rowBase + row < n && colBase + x < n;
        value[r] = valid ? dist[index] : INF;
        via[r] = valid ? path[index] : 0;
    }
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < TILE; ++k) {
        if (k < count) {
            const unsigned int kj = rt[k][x];
            #pragma unroll
            for (int r = 0; r < ROWS_PER_THREAD; ++r) {
                const unsigned int candidate = l[k][y + r * 8] + kj;
                if (candidate < value[r]) {
                    value[r] = candidate;
                    via[r] = static_cast<unsigned int>(base + k);
                }
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; ++r) {
        const size_t row = rowBase + y + r * 8;
        if (row < n && colBase + x < n) {
            const size_t index = row * n + colBase + x;
            dist[index] = value[r];
            path[index] = via[r];
        }
    }
}
} // namespace

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes) {
    // CUDA is required even for an empty graph; there is no CPU fallback.
    checkCuda(cudaFree(nullptr));
    if (numNodes == 0) return;
    unsigned int *deviceDist, *devicePath, *left, *right, *pivots;
    const size_t bytes = dist.size() * sizeof(unsigned int);
    checkCuda(cudaMalloc(&deviceDist, bytes));
    checkCuda(cudaMalloc(&devicePath, bytes));
    checkCuda(cudaMalloc(&left, TILE * numNodes * sizeof(unsigned int)));
    checkCuda(cudaMalloc(&right, TILE * numNodes * sizeof(unsigned int)));
    checkCuda(cudaMalloc(&pivots, 2 * TILE * TILE * sizeof(unsigned int)));
    checkCuda(cudaMemcpy(deviceDist, dist.data(), bytes, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(devicePath, path.data(), bytes, cudaMemcpyHostToDevice));
    const unsigned int tiles = static_cast<unsigned int>((numNodes + TILE - 1) / TILE);
    const dim3 threads(TILE, 8);
    for (size_t base = 0; base < numNodes; base += TILE) {
        const int count = static_cast<int>(std::min(static_cast<size_t>(TILE), numNodes - base));
        pivotSnapshots<<<1, threads>>>(deviceDist, pivots, numNodes, base, count);
        checkCuda(cudaGetLastError());
        panelSnapshots<<<dim3(tiles, 2), threads>>>(deviceDist, pivots, left, right, numNodes, base, count);
        checkCuda(cudaGetLastError());
        updateTiles<<<dim3(tiles, tiles), threads>>>(deviceDist, devicePath, left, right, numNodes, base, count);
        checkCuda(cudaGetLastError());
    }
    checkCuda(cudaMemcpy(dist.data(), deviceDist, bytes, cudaMemcpyDeviceToHost));
    checkCuda(cudaMemcpy(path.data(), devicePath, bytes, cudaMemcpyDeviceToHost));
    checkCuda(cudaFree(pivots));
    checkCuda(cudaFree(right));
    checkCuda(cudaFree(left));
    checkCuda(cudaFree(devicePath));
    checkCuda(cudaFree(deviceDist));
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
    // Initialize the CUDA context before measuring the computation.
    checkCuda(cudaFree(nullptr));
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
