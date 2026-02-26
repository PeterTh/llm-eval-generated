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

static inline void cudaCheck(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d: %s\n", file, line, cudaGetErrorString(err));
        std::exit(1);
    }
}
#define CUDA_CHECK(call) cudaCheck((call), __FILE__, __LINE__)

constexpr int TILE = 16;

__global__ void fw_phase1(unsigned int* __restrict__ dist, unsigned int* __restrict__ path, int n, int r) {
    __shared__ unsigned int s[TILE][TILE + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int base = r * TILE;
    const int i = base + ty;
    const int j = base + tx;

    unsigned int v = INF;
    if (i < n && j < n) v = dist[i * n + j];
    s[ty][tx] = v;
    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < TILE; ++kk) {
        const int k = base + kk;
        if (k >= n) break;
        const unsigned int via = s[ty][kk] + s[kk][tx];
        if (via < s[ty][tx]) {
            s[ty][tx] = via;
            if (i < n && j < n) path[i * n + j] = static_cast<unsigned int>(k);
        }
        __syncthreads();
    }

    if (i < n && j < n) dist[i * n + j] = s[ty][tx];
}

__global__ void fw_phase2_row(unsigned int* __restrict__ dist, unsigned int* __restrict__ path, int n, int r) {
    __shared__ unsigned int pivot[TILE][TILE + 1];
    __shared__ unsigned int tile[TILE][TILE + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int tiles = (n + TILE - 1) / TILE;
    int bj = blockIdx.x;
    if (bj >= r) ++bj;
    if (bj >= tiles) return;

    const int base = r * TILE;
    const int i = base + ty;
    const int j = bj * TILE + tx;

    unsigned int pv = INF;
    if (i < n && (base + tx) < n) pv = dist[i * n + (base + tx)];
    pivot[ty][tx] = pv;

    unsigned int tv = INF;
    if (i < n && j < n) tv = dist[i * n + j];
    tile[ty][tx] = tv;
    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < TILE; ++kk) {
        const int k = base + kk;
        if (k >= n) break;
        const unsigned int via = pivot[ty][kk] + tile[kk][tx];
        if (via < tile[ty][tx]) {
            tile[ty][tx] = via;
            if (i < n && j < n) path[i * n + j] = static_cast<unsigned int>(k);
        }
        __syncthreads();
    }

    if (i < n && j < n) dist[i * n + j] = tile[ty][tx];
}

__global__ void fw_phase2_col(unsigned int* __restrict__ dist, unsigned int* __restrict__ path, int n, int r) {
    __shared__ unsigned int pivot[TILE][TILE + 1];
    __shared__ unsigned int tile[TILE][TILE + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int tiles = (n + TILE - 1) / TILE;
    int bi = blockIdx.x;
    if (bi >= r) ++bi;
    if (bi >= tiles) return;

    const int base = r * TILE;
    const int i = bi * TILE + ty;
    const int j = base + tx;

    unsigned int pv = INF;
    if ((base + ty) < n && j < n) pv = dist[(base + ty) * n + j];
    pivot[ty][tx] = pv;

    unsigned int tv = INF;
    if (i < n && j < n) tv = dist[i * n + j];
    tile[ty][tx] = tv;
    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < TILE; ++kk) {
        const int k = base + kk;
        if (k >= n) break;
        const unsigned int via = tile[ty][kk] + pivot[kk][tx];
        if (via < tile[ty][tx]) {
            tile[ty][tx] = via;
            if (i < n && j < n) path[i * n + j] = static_cast<unsigned int>(k);
        }
        __syncthreads();
    }

    if (i < n && j < n) dist[i * n + j] = tile[ty][tx];
}

__global__ void fw_phase3(unsigned int* __restrict__ dist, unsigned int* __restrict__ path, int n, int r) {
    __shared__ unsigned int colTile[TILE][TILE + 1];
    __shared__ unsigned int rowTile[TILE][TILE + 1];
    __shared__ unsigned int tile[TILE][TILE + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int tiles = (n + TILE - 1) / TILE;
    int bi = blockIdx.y;
    int bj = blockIdx.x;
    if (bj >= r) ++bj;
    if (bi >= r) ++bi;
    if (bi >= tiles || bj >= tiles) return;

    const int base = r * TILE;
    const int i = bi * TILE + ty;
    const int j = bj * TILE + tx;

    unsigned int a = INF;
    if (i < n && (base + tx) < n) a = dist[i * n + (base + tx)];
    colTile[ty][tx] = a;

    unsigned int b = INF;
    if ((base + ty) < n && j < n) b = dist[(base + ty) * n + j];
    rowTile[ty][tx] = b;

    unsigned int c = INF;
    if (i < n && j < n) c = dist[i * n + j];
    tile[ty][tx] = c;
    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < TILE; ++kk) {
        const int k = base + kk;
        if (k >= n) break;
        const unsigned int via = colTile[ty][kk] + rowTile[kk][tx];
        if (via < tile[ty][tx]) {
            tile[ty][tx] = via;
            if (i < n && j < n) path[i * n + j] = static_cast<unsigned int>(k);
        }
        __syncthreads();
    }

    if (i < n && j < n) dist[i * n + j] = tile[ty][tx];
}

static void floydWarshallCUDA(unsigned int* d_dist, unsigned int* d_path, const size_t numNodes) {
    const int n = static_cast<int>(numNodes);
    const int tiles = (n + TILE - 1) / TILE;
    const dim3 threads(TILE, TILE);

    for (int r = 0; r < tiles; ++r) {
        fw_phase1<<<1, threads>>>(d_dist, d_path, n, r);
        CUDA_CHECK(cudaGetLastError());

        if (tiles > 1) {
            fw_phase2_row<<<tiles - 1, threads>>>(d_dist, d_path, n, r);
            CUDA_CHECK(cudaGetLastError());
            fw_phase2_col<<<tiles - 1, threads>>>(d_dist, d_path, n, r);
            CUDA_CHECK(cudaGetLastError());

            const dim3 grid3(tiles - 1, tiles - 1);
            fw_phase3<<<grid3, threads>>>(d_dist, d_path, n, r);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());
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
    
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount <= 0) {
        fprintf(stderr, "No CUDA devices found; this benchmark requires a CUDA-capable GPU.\n");
        return 1;
    }
    CUDA_CHECK(cudaSetDevice(0));

    // Allocate/copy to GPU (excluded from compute timing)
    unsigned int* d_dist = nullptr;
    unsigned int* d_path = nullptr;
    const size_t bytes = numNodes * numNodes * sizeof(unsigned int);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_dist), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_path), bytes));
    CUDA_CHECK(cudaMemcpy(d_dist, dist.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_path, path.data(), bytes, cudaMemcpyHostToDevice));

    // Run Floyd-Warshall on GPU
    printf("Computing shortest paths...\n");
    auto start = std::chrono::high_resolution_clock::now();

    floydWarshallCUDA(d_dist, d_path, numNodes);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Copy results back
    CUDA_CHECK(cudaMemcpy(dist.data(), d_dist, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(path.data(), d_path, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_dist));
    CUDA_CHECK(cudaFree(d_path));
    
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
