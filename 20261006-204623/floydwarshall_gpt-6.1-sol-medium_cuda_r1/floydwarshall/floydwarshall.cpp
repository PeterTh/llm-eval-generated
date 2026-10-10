#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <limits>
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
constexpr int THREAD_ROWS = 8;

void cudaCheck(cudaError_t status) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

// Save the pivot vectors BEFORE each intermediate vertex is processed.
// Using fully relaxed panels instead would change which intermediate vertex
// is recorded in path when multiple shortest paths have equal lengths.
__global__ void pivotTile(unsigned int* dist, unsigned int* path,
                          unsigned int* rows, unsigned int* cols,
                          size_t n, size_t base, int count) {
    __shared__ unsigned int tile[TILE][TILE];
    __shared__ unsigned int pivotRow[TILE], pivotCol[TILE];
    const int x = threadIdx.x, y = threadIdx.y;
    unsigned int history[4];
    #pragma unroll
    for (int t = 0; t < 4; ++t) {
        int r = y + t * THREAD_ROWS;
        bool valid = base + r < n && base + x < n;
        tile[r][x] = valid ? dist[(base + r) * n + base + x] : INF;
        history[t] = valid ? path[(base + r) * n + base + x] : 0;
    }
    __syncthreads();
    for (int k = 0; k < count; ++k) {
        if (y == 0) {
            pivotRow[x] = tile[k][x];
            pivotCol[x] = tile[x][k];
            if (base + x < n) {
                rows[size_t(k) * n + base + x] = pivotRow[x];
                cols[size_t(k) * n + base + x] = pivotCol[x];
            }
        }
        __syncthreads();
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            int r = y + t * THREAD_ROWS;
            unsigned int candidate = pivotCol[r] + pivotRow[x];
            if (candidate < tile[r][x]) {
                tile[r][x] = candidate;
                history[t] = static_cast<unsigned int>(base + k);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int t = 0; t < 4; ++t) {
        int r = y + t * THREAD_ROWS;
        if (base + r < n && base + x < n) {
            dist[(base + r) * n + base + x] = tile[r][x];
            path[(base + r) * n + base + x] = history[t];
        }
    }
}

__global__ void pivotPanels(unsigned int* dist, unsigned int* path,
                            unsigned int* rows, unsigned int* cols,
                            size_t n, size_t base, int count) {
    __shared__ unsigned int tile[TILE][TILE];
    __shared__ unsigned int diagonal[TILE][TILE];
    __shared__ unsigned int pivot[TILE];
    const int x = threadIdx.x, y = threadIdx.y;
    const bool rowPanel = blockIdx.y == 0;
    size_t other = size_t(blockIdx.x) * TILE;
    if (other >= base) other += TILE;
    const size_t rowBase = rowPanel ? base : other;
    const size_t colBase = rowPanel ? other : base;
    unsigned int history[4];
    #pragma unroll
    for (int t = 0; t < 4; ++t) {
        int r = y + t * THREAD_ROWS;
        bool valid = rowBase + r < n && colBase + x < n;
        tile[r][x] = valid ? dist[(rowBase + r) * n + colBase + x] : INF;
        history[t] = valid ? path[(rowBase + r) * n + colBase + x] : 0;
        diagonal[r][x] = r < count && base + x < n
            ? (rowPanel ? cols[size_t(r) * n + base + x]
                        : rows[size_t(r) * n + base + x]) : INF;
    }
    __syncthreads();
    for (int k = 0; k < count; ++k) {
        if (y == 0) {
            pivot[x] = rowPanel ? tile[k][x] : tile[x][k];
            if (other + x < n) {
                if (rowPanel) rows[size_t(k) * n + other + x] = pivot[x];
                else cols[size_t(k) * n + other + x] = pivot[x];
            }
        }
        __syncthreads();
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            int r = y + t * THREAD_ROWS;
            unsigned int candidate = rowPanel
                ? diagonal[k][r] + pivot[x]
                : pivot[r] + diagonal[k][x];
            if (candidate < tile[r][x]) {
                tile[r][x] = candidate;
                history[t] = static_cast<unsigned int>(base + k);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int t = 0; t < 4; ++t) {
        int r = y + t * THREAD_ROWS;
        if (rowBase + r < n && colBase + x < n) {
            dist[(rowBase + r) * n + colBase + x] = tile[r][x];
            path[(rowBase + r) * n + colBase + x] = history[t];
        }
    }
}

__global__ void remainingTiles(unsigned int* __restrict__ dist,
                               unsigned int* __restrict__ path,
                               const unsigned int* __restrict__ rows,
                               const unsigned int* __restrict__ cols,
                               size_t n, size_t base, int count) {
    __shared__ unsigned int rowPanel[TILE][TILE];
    __shared__ unsigned int colPanel[TILE][TILE + 1];
    const int x = threadIdx.x, y = threadIdx.y;
    size_t rowBase = size_t(blockIdx.y) * TILE;
    size_t colBase = size_t(blockIdx.x) * TILE;
    if (rowBase >= base) rowBase += TILE;
    if (colBase >= base) colBase += TILE;
    unsigned int values[4], history[4];
    #pragma unroll
    for (int t = 0; t < 4; ++t) {
        int r = y + t * THREAD_ROWS;
        rowPanel[r][x] = r < count && colBase + x < n ? rows[size_t(r) * n + colBase + x] : INF;
        colPanel[x][r] = r < count && rowBase + x < n ? cols[size_t(r) * n + rowBase + x] : INF;
        bool valid = rowBase + r < n && colBase + x < n;
        values[t] = valid ? dist[(rowBase + r) * n + colBase + x] : INF;
        history[t] = valid ? path[(rowBase + r) * n + colBase + x] : 0;
    }
    __syncthreads();
    for (int k = 0; k < count; ++k) {
        const unsigned int right = rowPanel[k][x];
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            const unsigned int candidate = colPanel[y + t * THREAD_ROWS][k] + right;
            if (candidate < values[t]) {
                values[t] = candidate;
                history[t] = static_cast<unsigned int>(base + k);
            }
        }
    }
    #pragma unroll
    for (int t = 0; t < 4; ++t) {
        int r = y + t * THREAD_ROWS;
        if (rowBase + r < n && colBase + x < n) {
            dist[(rowBase + r) * n + colBase + x] = values[t];
            path[(rowBase + r) * n + colBase + x] = history[t];
        }
    }
}
} // namespace

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes) {
    if (numNodes == 0) return;
    unsigned int *deviceDist, *devicePath, *rows, *cols;
    const size_t bytes = dist.size() * sizeof(unsigned int);
    const size_t panelBytes = numNodes * TILE * sizeof(unsigned int);
    cudaCheck(cudaMalloc(&deviceDist, bytes));
    cudaCheck(cudaMalloc(&devicePath, bytes));
    cudaCheck(cudaMalloc(&rows, panelBytes));
    cudaCheck(cudaMalloc(&cols, panelBytes));
    cudaCheck(cudaMemcpy(deviceDist, dist.data(), bytes, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(devicePath, path.data(), bytes, cudaMemcpyHostToDevice));
    const size_t tiles = (numNodes + TILE - 1) / TILE;
    const dim3 threads(TILE, THREAD_ROWS);
    for (size_t base = 0; base < numNodes; base += TILE) {
        const int count = static_cast<int>(std::min(size_t(TILE), numNodes - base));
        pivotTile<<<1, threads>>>(deviceDist, devicePath, rows, cols, numNodes, base, count);
        if (tiles > 1) {
            pivotPanels<<<dim3(static_cast<unsigned int>(tiles - 1), 2), threads>>>(
                deviceDist, devicePath, rows, cols, numNodes, base, count);
            remainingTiles<<<dim3(static_cast<unsigned int>(tiles - 1),
                                  static_cast<unsigned int>(tiles - 1)), threads>>>(
                deviceDist, devicePath, rows, cols, numNodes, base, count);
        }
        cudaCheck(cudaGetLastError());
    }
    cudaCheck(cudaMemcpy(dist.data(), deviceDist, bytes, cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(path.data(), devicePath, bytes, cudaMemcpyDeviceToHost));
    cudaCheck(cudaFree(cols));
    cudaCheck(cudaFree(rows));
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
            const char* argument = argv[++i];
            char* end = nullptr;
            const unsigned long long parsed = strtoull(argument, &end, 10);
            if (argument[0] == '-' || end == argument || *end != '\0' ||
                parsed > std::numeric_limits<unsigned int>::max() ||
                (parsed != 0 && parsed > std::numeric_limits<size_t>::max() /
                                                sizeof(unsigned int) / parsed)) {
                fprintf(stderr, "Invalid number of nodes: %s\n", argument);
                return 1;
            }
            numNodes = static_cast<size_t>(parsed);
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
    
    // Initialize the GPU before the timed computation.
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
    const double seconds = std::chrono::duration<double>(end - start).count();
    
    printf("Computation time: %.3f ms\n", seconds * 1000.0);
    
    // Calculate operations per second
    // Floyd-Warshall has O(n³) complexity
    double ops = (double)numNodes * numNodes * numNodes;
    double gflops = seconds > 0.0 ? ops / seconds / 1e9 : 0.0;
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
