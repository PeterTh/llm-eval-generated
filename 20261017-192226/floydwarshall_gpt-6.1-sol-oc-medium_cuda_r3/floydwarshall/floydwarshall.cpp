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

void cudaCheck(cudaError_t status) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

// Diagonal and panel tiles retain the pivot row/column as they existed at
// each individual k. Using fully closed panels instead would give the same
// distances, but could change the original algorithm's strict-update paths.
template<bool Diagonal>
__global__ void updatePivot(unsigned int* dist, unsigned int* path,
                            unsigned int* rowHistory, unsigned int* colHistory,
                            size_t n, size_t pivot) {
    __shared__ unsigned int tile[TILE][TILE];
    __shared__ unsigned int history[TILE][TILE];
    const int x = threadIdx.x;
    const int y = threadIdx.y;
    const bool rowPanel = Diagonal || blockIdx.y == 0;
    size_t other = blockIdx.x;
    if (!Diagonal && other >= pivot) ++other;
    const size_t tileRow = Diagonal || rowPanel ? pivot : other;
    const size_t tileCol = Diagonal || !rowPanel ? pivot : other;
    const size_t col = tileCol * TILE + x;
    unsigned int paths[ROWS_PER_THREAD];
    #pragma unroll
    for (int q = 0; q < ROWS_PER_THREAD; ++q) {
        const int r = y + q * 8;
        const size_t row = tileRow * TILE + r;
        const bool valid = row < n && col < n;
        tile[r][x] = valid ? dist[row * n + col] : INF;
        paths[q] = valid ? path[row * n + col] : 0;
        if (!Diagonal) {
            const size_t node = pivot * TILE + x;
            history[r][x] = node < n && pivot * TILE + r < n
                ? (rowPanel ? colHistory[r * n + node]
                            : rowHistory[r * n + node]) : INF;
        }
    }
    __syncthreads();
    const int count = static_cast<int>(n - pivot * TILE < TILE
        ? n - pivot * TILE : TILE);
    for (int k = 0; k < count; ++k) {
        if (y == 0) {
            if ((Diagonal || rowPanel) && col < n)
                rowHistory[k * n + col] = tile[k][x];
            const size_t row = tileRow * TILE + x;
            if ((Diagonal || !rowPanel) && row < n)
                colHistory[k * n + row] = tile[x][k];
        }
        #pragma unroll
        for (int q = 0; q < ROWS_PER_THREAD; ++q) {
            const int r = y + q * 8;
            const unsigned int a = Diagonal || !rowPanel
                ? tile[r][k] : history[k][r];
            const unsigned int b = Diagonal || rowPanel
                ? tile[k][x] : history[k][x];
            // The zero diagonal means these pivot entries never change.
            // Avoid writing them, so simultaneous operand reads are race-free.
            if ((!Diagonal || (r != k && x != k)) &&
                (Diagonal || (rowPanel ? r != k : x != k)) &&
                a + b < tile[r][x]) {
                tile[r][x] = a + b;
                paths[q] = static_cast<unsigned int>(pivot * TILE + k);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int q = 0; q < ROWS_PER_THREAD; ++q) {
        const int r = y + q * 8;
        const size_t row = tileRow * TILE + r;
        if (row < n && col < n) {
            dist[row * n + col] = tile[r][x];
            path[row * n + col] = paths[q];
        }
    }
}

__global__ void updateRemaining(unsigned int* __restrict__ dist,
                                unsigned int* __restrict__ path,
                                const unsigned int* __restrict__ rowHistory,
                                const unsigned int* __restrict__ colHistory,
                                size_t n, size_t pivot) {
    __shared__ unsigned int rows[TILE][TILE];
    __shared__ unsigned int cols[TILE][TILE];
    const int x = threadIdx.x;
    const int y = threadIdx.y;
    const size_t tileRow = blockIdx.y + (blockIdx.y >= pivot);
    const size_t tileCol = blockIdx.x + (blockIdx.x >= pivot);
    const size_t col = tileCol * TILE + x;
    unsigned int values[ROWS_PER_THREAD], paths[ROWS_PER_THREAD];
    #pragma unroll
    for (int q = 0; q < ROWS_PER_THREAD; ++q) {
        const int r = y + q * 8;
        const size_t row = tileRow * TILE + r;
        const size_t sourceRow = tileRow * TILE + x;
        rows[r][x] = col < n && pivot * TILE + r < n
            ? rowHistory[r * n + col] : INF;
        cols[r][x] = sourceRow < n && pivot * TILE + r < n
            ? colHistory[r * n + sourceRow] : INF;
        values[q] = row < n && col < n ? dist[row * n + col] : INF;
        paths[q] = row < n && col < n ? path[row * n + col] : 0;
    }
    __syncthreads();
    const int count = static_cast<int>(n - pivot * TILE < TILE
        ? n - pivot * TILE : TILE);
    #pragma unroll
    for (int k = 0; k < TILE; ++k) {
        if (k < count) {
            const unsigned int b = rows[k][x];
            #pragma unroll
            for (int q = 0; q < ROWS_PER_THREAD; ++q) {
                const unsigned int candidate = cols[k][y + q * 8] + b;
                if (candidate < values[q]) {
                    values[q] = candidate;
                    paths[q] = static_cast<unsigned int>(pivot * TILE + k);
                }
            }
        }
    }
    #pragma unroll
    for (int q = 0; q < ROWS_PER_THREAD; ++q) {
        const size_t row = tileRow * TILE + y + q * 8;
        if (row < n && col < n) {
            dist[row * n + col] = values[q];
            path[row * n + col] = paths[q];
        }
    }
}
} // namespace

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path, const size_t numNodes) {
    // Initialize CUDA even for an empty graph; there is no CPU fallback.
    cudaCheck(cudaFree(nullptr));
    if (numNodes == 0) return;
    const size_t bytes = dist.size() * sizeof(unsigned int);
    unsigned int *deviceDist, *devicePath, *rowHistory, *colHistory;
    cudaCheck(cudaMalloc(&deviceDist, bytes));
    cudaCheck(cudaMalloc(&devicePath, bytes));
    cudaCheck(cudaMalloc(&rowHistory, TILE * numNodes * sizeof(unsigned int)));
    cudaCheck(cudaMalloc(&colHistory, TILE * numNodes * sizeof(unsigned int)));
    cudaCheck(cudaMemcpy(deviceDist, dist.data(), bytes, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(devicePath, path.data(), bytes, cudaMemcpyHostToDevice));
    const size_t tiles = (numNodes + TILE - 1) / TILE;
    const dim3 threads(TILE, 8);
    for (size_t pivot = 0; pivot < tiles; ++pivot) {
        updatePivot<true><<<1, threads>>>(deviceDist, devicePath, rowHistory,
                                        colHistory, numNodes, pivot);
        if (tiles > 1) {
            updatePivot<false><<<dim3(tiles - 1, 2), threads>>>(deviceDist,
                devicePath, rowHistory, colHistory, numNodes, pivot);
            updateRemaining<<<dim3(tiles - 1, tiles - 1), threads>>>(deviceDist,
                devicePath, rowHistory, colHistory, numNodes, pivot);
        }
    }
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaMemcpy(dist.data(), deviceDist, bytes, cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(path.data(), devicePath, bytes, cudaMemcpyDeviceToHost));
    cudaCheck(cudaFree(colHistory));
    cudaCheck(cudaFree(rowHistory));
    cudaCheck(cudaFree(devicePath));
    cudaCheck(cudaFree(deviceDist));
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

    // One-time runtime/context startup is not part of the APSP computation.
    cudaCheck(cudaFree(nullptr));
    
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
    double gflops = ops / std::chrono::duration<double>(end - start).count() / 1e9;
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
