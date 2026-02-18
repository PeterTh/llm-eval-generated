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

static inline void cudaCheck(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error (%s): %s\n", what, cudaGetErrorString(err));
        std::exit(1);
    }
}

constexpr int TILE = 32;

__device__ __forceinline__ int idx2d(int i, int j, int n) { return j * n + i; }

// Phase 1: update pivot block (r,r)
__global__ void fw_phase1(unsigned int* __restrict__ dist,
                          unsigned int* __restrict__ path,
                          int n, int r) {
    __shared__ unsigned int s[TILE][TILE];

    const int x = (int)threadIdx.x;
    const int y = (int)threadIdx.y;

    const int row = r * TILE + y; // destination
    const int col = r * TILE + x; // source

    if (row < n && col < n) {
        s[y][x] = dist[idx2d(col, row, n)];
    } else {
        s[y][x] = INF;
    }
    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < TILE; ++kk) {
        const unsigned int a = s[kk][x];
        const unsigned int b = s[y][kk];
        __syncthreads();

        const unsigned int via = a + b;
        if (via < s[y][x]) {
            s[y][x] = via;
            if (row < n && col < n) {
                path[idx2d(col, row, n)] = (unsigned int)(r * TILE + kk);
            }
        }
        __syncthreads();
    }

    if (row < n && col < n) {
        dist[idx2d(col, row, n)] = s[y][x];
    }
}

// Phase 2a: update blocks on pivot row (r, bx), bx != r
__global__ void fw_phase2_row(unsigned int* __restrict__ dist,
                              unsigned int* __restrict__ path,
                              int n, int r) {
    const int bx = (int)blockIdx.x;
    if (bx == r) return;

    __shared__ unsigned int pivot[TILE][TILE];
    __shared__ unsigned int tile[TILE][TILE];

    const int x = (int)threadIdx.x;
    const int y = (int)threadIdx.y;

    const int row = r * TILE + y;
    const int col = bx * TILE + x;

    // pivot element: (row, k) where k is in pivot col range
    const int pivot_row = row;
    const int pivot_col = r * TILE + x;

    if (pivot_row < n && pivot_col < n) {
        pivot[y][x] = dist[idx2d(pivot_col, pivot_row, n)];
    } else {
        pivot[y][x] = INF;
    }

    if (row < n && col < n) {
        tile[y][x] = dist[idx2d(col, row, n)];
    } else {
        tile[y][x] = INF;
    }
    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < TILE; ++kk) {
        const unsigned int a = tile[kk][x];
        const unsigned int b = pivot[y][kk];
        __syncthreads();

        const unsigned int via = a + b;
        if (via < tile[y][x]) {
            tile[y][x] = via;
            if (row < n && col < n) {
                path[idx2d(col, row, n)] = (unsigned int)(r * TILE + kk);
            }
        }
        __syncthreads();
    }

    if (row < n && col < n) {
        dist[idx2d(col, row, n)] = tile[y][x];
    }
}

// Phase 2b: update blocks on pivot column (by, r), by != r
__global__ void fw_phase2_col(unsigned int* __restrict__ dist,
                              unsigned int* __restrict__ path,
                              int n, int r) {
    const int by = (int)blockIdx.x;
    if (by == r) return;

    __shared__ unsigned int pivot[TILE][TILE];
    __shared__ unsigned int tile[TILE][TILE];

    const int x = (int)threadIdx.x;
    const int y = (int)threadIdx.y;

    const int row = by * TILE + y;
    const int col = r * TILE + x;

    // pivot element: (k, col) where k is in pivot row range
    const int pivot_row = r * TILE + y;
    const int pivot_col = col;

    if (pivot_row < n && pivot_col < n) {
        pivot[y][x] = dist[idx2d(pivot_col, pivot_row, n)];
    } else {
        pivot[y][x] = INF;
    }

    if (row < n && col < n) {
        tile[y][x] = dist[idx2d(col, row, n)];
    } else {
        tile[y][x] = INF;
    }
    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < TILE; ++kk) {
        const unsigned int a = pivot[kk][x];
        const unsigned int b = tile[y][kk];
        __syncthreads();

        const unsigned int via = a + b;
        if (via < tile[y][x]) {
            tile[y][x] = via;
            if (row < n && col < n) {
                path[idx2d(col, row, n)] = (unsigned int)(r * TILE + kk);
            }
        }
        __syncthreads();
    }

    if (row < n && col < n) {
        dist[idx2d(col, row, n)] = tile[y][x];
    }
}

// Phase 3: update remaining blocks (by, bx), by != r, bx != r
__global__ void fw_phase3(unsigned int* __restrict__ dist,
                          unsigned int* __restrict__ path,
                          int n, int r) {
    const int bx = (int)blockIdx.x;
    const int by = (int)blockIdx.y;
    if (bx == r || by == r) return;

    __shared__ unsigned int rowTile[TILE][TILE]; // (r, bx)
    __shared__ unsigned int colTile[TILE][TILE]; // (by, r)
    __shared__ unsigned int tile[TILE][TILE];    // (by, bx)

    const int x = (int)threadIdx.x;
    const int y = (int)threadIdx.y;

    const int row = by * TILE + y;
    const int col = bx * TILE + x;

    const int rowTile_row = r * TILE + y;
    const int rowTile_col = col;

    const int colTile_row = row;
    const int colTile_col = r * TILE + x;

    if (rowTile_row < n && rowTile_col < n) {
        rowTile[y][x] = dist[idx2d(rowTile_col, rowTile_row, n)];
    } else {
        rowTile[y][x] = INF;
    }

    if (colTile_row < n && colTile_col < n) {
        colTile[y][x] = dist[idx2d(colTile_col, colTile_row, n)];
    } else {
        colTile[y][x] = INF;
    }

    if (row < n && col < n) {
        tile[y][x] = dist[idx2d(col, row, n)];
    } else {
        tile[y][x] = INF;
    }
    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < TILE; ++kk) {
        const unsigned int a = colTile[y][kk];
        const unsigned int b = rowTile[kk][x];
        __syncthreads();

        const unsigned int via = a + b;
        if (via < tile[y][x]) {
            tile[y][x] = via;
            if (row < n && col < n) {
                path[idx2d(col, row, n)] = (unsigned int)(r * TILE + kk);
            }
        }
        __syncthreads();
    }

    if (row < n && col < n) {
        dist[idx2d(col, row, n)] = tile[y][x];
    }
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

double floydWarshall(std::vector<unsigned int>& dist,
                     std::vector<unsigned int>& path,
                     const size_t numNodes) {
    int deviceCount = 0;
    cudaCheck(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");
    if (deviceCount <= 0) {
        std::fprintf(stderr, "No CUDA devices found\n");
        std::exit(1);
    }
    cudaCheck(cudaSetDevice(0), "cudaSetDevice");

    const int n = (int)numNodes;
    const size_t bytes = (size_t)n * (size_t)n * sizeof(unsigned int);

    unsigned int* d_dist = nullptr;
    unsigned int* d_path = nullptr;
    cudaCheck(cudaMalloc((void**)&d_dist, bytes), "cudaMalloc dist");
    cudaCheck(cudaMalloc((void**)&d_path, bytes), "cudaMalloc path");

    cudaCheck(cudaMemcpy(d_dist, dist.data(), bytes, cudaMemcpyHostToDevice), "H2D dist");
    cudaCheck(cudaMemcpy(d_path, path.data(), bytes, cudaMemcpyHostToDevice), "H2D path");

    const int rounds = (n + TILE - 1) / TILE;
    const dim3 threads(TILE, TILE);

    cudaEvent_t evStart{}, evStop{};
    cudaCheck(cudaEventCreate(&evStart), "cudaEventCreate start");
    cudaCheck(cudaEventCreate(&evStop), "cudaEventCreate stop");
    cudaCheck(cudaEventRecord(evStart), "cudaEventRecord start");

    for (int r = 0; r < rounds; ++r) {
        fw_phase1<<<1, threads>>>(d_dist, d_path, n, r);

        fw_phase2_row<<<rounds, threads>>>(d_dist, d_path, n, r);
        fw_phase2_col<<<rounds, threads>>>(d_dist, d_path, n, r);

        const dim3 grid3(rounds, rounds);
        fw_phase3<<<grid3, threads>>>(d_dist, d_path, n, r);
    }

    cudaCheck(cudaEventRecord(evStop), "cudaEventRecord stop");
    cudaCheck(cudaEventSynchronize(evStop), "cudaEventSynchronize stop");

    float ms = 0.0f;
    cudaCheck(cudaEventElapsedTime(&ms, evStart, evStop), "cudaEventElapsedTime");

    cudaCheck(cudaEventDestroy(evStop), "cudaEventDestroy stop");
    cudaCheck(cudaEventDestroy(evStart), "cudaEventDestroy start");

    cudaCheck(cudaGetLastError(), "kernel launch");

    cudaCheck(cudaMemcpy(dist.data(), d_dist, bytes, cudaMemcpyDeviceToHost), "D2H dist");
    cudaCheck(cudaMemcpy(path.data(), d_path, bytes, cudaMemcpyDeviceToHost), "D2H path");

    cudaCheck(cudaFree(d_path), "cudaFree path");
    cudaCheck(cudaFree(d_dist), "cudaFree dist");

    return (double)ms;
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

    const double ms = floydWarshall(dist, path, numNodes);

    printf("Computation time: %.3f ms\n", ms);

    // Calculate operations per second
    // Floyd-Warshall has O(n³) complexity
    const double ops = (double)numNodes * numNodes * numNodes;
    const double gflops = ops / (ms / 1000.0) / 1e9;
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
