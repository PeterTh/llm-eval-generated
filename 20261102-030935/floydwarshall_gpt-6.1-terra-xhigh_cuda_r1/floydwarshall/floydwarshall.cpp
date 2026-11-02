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

constexpr int TILE = 32;
constexpr int ROWS_PER_THREAD = 4;

static void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

// Record the pivot row and column at each k before applying that k. These
// snapshots preserve the sequential algorithm's strict-update path semantics.
__global__ void pivotTile(unsigned int* dist, unsigned int* path,
                          unsigned int* rowHistory, unsigned int* colHistory,
                          size_t n, size_t base) {
    __shared__ unsigned int tile[TILE][TILE];
    const int x = threadIdx.x;
    const int y = threadIdx.y;
    unsigned int values[ROWS_PER_THREAD], paths[ROWS_PER_THREAD];
    #pragma unroll
    for (int t = 0; t < ROWS_PER_THREAD; ++t) {
        int r = y + t * blockDim.y;
        size_t i = base + r, j = base + x;
        values[t] = i < n && j < n ? dist[i * n + j] : INF;
        paths[t] = i < n && j < n ? path[i * n + j] : 0;
        tile[r][x] = values[t];
    }
    __syncthreads();
    for (int k = 0; k < TILE && base + k < n; ++k) {
        if (y == 0 && base + x < n) {
            rowHistory[(size_t)k * n + base + x] = tile[k][x];
            colHistory[(size_t)k * n + base + x] = tile[x][k];
        }
        unsigned int candidates[ROWS_PER_THREAD];
        #pragma unroll
        for (int t = 0; t < ROWS_PER_THREAD; ++t) {
            int r = y + t * blockDim.y;
            candidates[t] = tile[r][k] + tile[k][x];
        }
        __syncthreads();
        #pragma unroll
        for (int t = 0; t < ROWS_PER_THREAD; ++t) {
            int r = y + t * blockDim.y;
            if (candidates[t] < values[t]) {
                values[t] = candidates[t];
                paths[t] = static_cast<unsigned int>(base + k);
            }
            tile[r][x] = values[t];
        }
        __syncthreads();
    }
    #pragma unroll
    for (int t = 0; t < ROWS_PER_THREAD; ++t) {
        size_t i = base + y + t * blockDim.y, j = base + x;
        if (i < n && j < n) {
            dist[i * n + j] = values[t];
            path[i * n + j] = paths[t];
        }
    }
}

// Update the pivot block row and column, recording their state at each k.
__global__ void pivotEdges(unsigned int* dist, unsigned int* path,
                           unsigned int* rowHistory, unsigned int* colHistory,
                           size_t n, size_t base, unsigned int pivotBlock) {
    __shared__ unsigned int tile[TILE][TILE];
    const int x = threadIdx.x, y = threadIdx.y;
    const bool isRow = blockIdx.y == 0;
    unsigned int other = blockIdx.x + (blockIdx.x >= pivotBlock);
    size_t rowBase = isRow ? base : (size_t)other * TILE;
    size_t colBase = isRow ? (size_t)other * TILE : base;
    unsigned int values[ROWS_PER_THREAD], paths[ROWS_PER_THREAD];
    #pragma unroll
    for (int t = 0; t < ROWS_PER_THREAD; ++t) {
        int r = y + t * blockDim.y;
        size_t i = rowBase + r, j = colBase + x;
        values[t] = i < n && j < n ? dist[i * n + j] : INF;
        paths[t] = i < n && j < n ? path[i * n + j] : 0;
        tile[r][x] = values[t];
    }
    __syncthreads();
    for (int k = 0; k < TILE && base + k < n; ++k) {
        if (y == 0) {
            if (isRow && colBase + x < n)
                rowHistory[(size_t)k * n + colBase + x] = tile[k][x];
            if (!isRow && rowBase + x < n)
                colHistory[(size_t)k * n + rowBase + x] = tile[x][k];
        }
        unsigned int candidates[ROWS_PER_THREAD];
        #pragma unroll
        for (int t = 0; t < ROWS_PER_THREAD; ++t) {
            int r = y + t * blockDim.y;
            if (isRow) {
                unsigned int left = rowBase + r < n ? colHistory[(size_t)k * n + rowBase + r] : INF;
                candidates[t] = left + tile[k][x];
            } else {
                unsigned int right = colBase + x < n ? rowHistory[(size_t)k * n + colBase + x] : INF;
                candidates[t] = tile[r][k] + right;
            }
        }
        __syncthreads();
        #pragma unroll
        for (int t = 0; t < ROWS_PER_THREAD; ++t) {
            int r = y + t * blockDim.y;
            if (candidates[t] < values[t]) {
                values[t] = candidates[t];
                paths[t] = static_cast<unsigned int>(base + k);
            }
            tile[r][x] = values[t];
        }
        __syncthreads();
    }
    #pragma unroll
    for (int t = 0; t < ROWS_PER_THREAD; ++t) {
        size_t i = rowBase + y + t * blockDim.y, j = colBase + x;
        if (i < n && j < n) {
            dist[i * n + j] = values[t];
            path[i * n + j] = paths[t];
        }
    }
}

__global__ void remainingTiles(unsigned int* dist, unsigned int* path,
                               const unsigned int* rowHistory, const unsigned int* colHistory,
                               size_t n, size_t base, unsigned int pivotBlock) {
    __shared__ unsigned int rows[TILE][TILE], cols[TILE][TILE];
    int x = threadIdx.x, y = threadIdx.y;
    unsigned int blockCol = blockIdx.x + (blockIdx.x >= pivotBlock);
    unsigned int blockRow = blockIdx.y + (blockIdx.y >= pivotBlock);
    size_t rowBase = (size_t)blockRow * TILE, colBase = (size_t)blockCol * TILE;
    unsigned int values[ROWS_PER_THREAD], paths[ROWS_PER_THREAD];
    #pragma unroll
    for (int t = 0; t < ROWS_PER_THREAD; ++t) {
        int r = y + t * blockDim.y;
        size_t i = rowBase + r, j = colBase + x;
        values[t] = i < n && j < n ? dist[i * n + j] : INF;
        paths[t] = i < n && j < n ? path[i * n + j] : 0;
        rows[r][x] = base + r < n && j < n ? rowHistory[(size_t)r * n + j] : INF;
        cols[r][x] = base + r < n && rowBase + x < n ? colHistory[(size_t)r * n + rowBase + x] : INF;
    }
    __syncthreads();
    for (int k = 0; k < TILE && base + k < n; ++k) {
        unsigned int right = rows[k][x];
        #pragma unroll
        for (int t = 0; t < ROWS_PER_THREAD; ++t) {
            int r = y + t * blockDim.y;
            unsigned int candidate = cols[k][r] + right;
            if (candidate < values[t]) {
                values[t] = candidate;
                paths[t] = static_cast<unsigned int>(base + k);
            }
        }
    }
    #pragma unroll
    for (int t = 0; t < ROWS_PER_THREAD; ++t) {
        size_t i = rowBase + y + t * blockDim.y, j = colBase + x;
        if (i < n && j < n) {
            dist[i * n + j] = values[t];
            path[i * n + j] = paths[t];
        }
    }
}

class CudaFloydWarshall {
public:
    CudaFloydWarshall(const std::vector<unsigned int>& dist,
                      const std::vector<unsigned int>& path, size_t numNodes)
        : n(numNodes), bytes(dist.size() * sizeof(unsigned int)) {
        if (n == 0) return;
        const size_t historyBytes = TILE * n * sizeof(unsigned int);
        checkCuda(cudaMalloc(&deviceDist, bytes), "allocating distances");
        checkCuda(cudaMalloc(&devicePath, bytes), "allocating paths");
        checkCuda(cudaMalloc(&rowHistory, historyBytes), "allocating row history");
        checkCuda(cudaMalloc(&colHistory, historyBytes), "allocating column history");
        checkCuda(cudaMemcpy(deviceDist, dist.data(), bytes, cudaMemcpyHostToDevice), "copying distances to GPU");
        checkCuda(cudaMemcpy(devicePath, path.data(), bytes, cudaMemcpyHostToDevice), "copying paths to GPU");
    }

    void run() {
        if (n == 0) return;
        const unsigned int blocks = static_cast<unsigned int>((n + TILE - 1) / TILE);
        const dim3 threads(TILE, TILE / ROWS_PER_THREAD);
        for (unsigned int b = 0; b < blocks; ++b) {
            size_t base = (size_t)b * TILE;
            pivotTile<<<1, threads>>>(deviceDist, devicePath, rowHistory, colHistory, n, base);
            if (blocks > 1) {
                pivotEdges<<<dim3(blocks - 1, 2), threads>>>(deviceDist, devicePath, rowHistory, colHistory, n, base, b);
                remainingTiles<<<dim3(blocks - 1, blocks - 1), threads>>>(deviceDist, devicePath, rowHistory, colHistory, n, base, b);
            }
        }
        checkCuda(cudaGetLastError(), "launching Floyd-Warshall kernels");
        checkCuda(cudaDeviceSynchronize(), "executing Floyd-Warshall kernels");
    }

    void download(std::vector<unsigned int>& dist, std::vector<unsigned int>& path) {
        if (n == 0) return;
        checkCuda(cudaMemcpy(dist.data(), deviceDist, bytes, cudaMemcpyDeviceToHost), "copying distances from GPU");
        checkCuda(cudaMemcpy(path.data(), devicePath, bytes, cudaMemcpyDeviceToHost), "copying paths from GPU");
    }

    ~CudaFloydWarshall() {
        if (deviceDist) cudaFree(deviceDist);
        if (devicePath) cudaFree(devicePath);
        if (rowHistory) cudaFree(rowHistory);
        if (colHistory) cudaFree(colHistory);
    }
    CudaFloydWarshall(const CudaFloydWarshall&) = delete;
    CudaFloydWarshall& operator=(const CudaFloydWarshall&) = delete;

private:
    size_t n, bytes;
    unsigned int *deviceDist = nullptr, *devicePath = nullptr;
    unsigned int *rowHistory = nullptr, *colHistory = nullptr;
};

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
    
    CudaFloydWarshall gpu(dist, path, numNodes);

    // Run Floyd-Warshall
    printf("Computing shortest paths...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    gpu.run();
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    gpu.download(dist, path);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate operations per second
    // Floyd-Warshall has O(n³) complexity
    double ops = (double)numNodes * numNodes * numNodes;
    const double seconds = std::chrono::duration<double>(end - start).count();
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
