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

constexpr int TILE = 32;

#define CUDA_CHECK(call)                                                                    \
    do {                                                                                    \
        const cudaError_t _err = (call);                                                     \
        if (_err != cudaSuccess) {                                                          \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_err)); \
            std::exit(1);                                                                    \
        }                                                                                   \
    } while (0)

// Index calculation for flattened 2D array (row-major: row=i, col=j)
__host__ __device__ inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

static __device__ __forceinline__ int map_excluding_round(const int b, const int r) {
    return (b < r) ? b : (b + 1);
}

__global__ void fw_phase1(unsigned int* __restrict__ dist,
                          unsigned int* __restrict__ path,
                          const int n,
                          const int r) {
    __shared__ unsigned int s[TILE][TILE];

    const int tx = (int)threadIdx.x;
    const int ty = (int)threadIdx.y;

    const int row = r * TILE + ty;
    const int col = r * TILE + tx;

    if (row < n && col < n) {
        s[ty][tx] = dist[(size_t)row * n + col];
    } else {
        s[ty][tx] = INF;
    }
    __syncthreads();

    for (int k = 0; k < TILE; ++k) {
        const unsigned int through = s[ty][k] + s[k][tx];
        if (through < s[ty][tx]) {
            s[ty][tx] = through;
            if (row < n && col < n) {
                path[(size_t)row * n + col] = (unsigned int)(r * TILE + k);
            }
        }
        __syncthreads();
    }

    if (row < n && col < n) {
        dist[(size_t)row * n + col] = s[ty][tx];
    }
}

__global__ void fw_phase2_row(unsigned int* __restrict__ dist,
                              unsigned int* __restrict__ path,
                              const int n,
                              const int r,
                              const int numBlocks) {
    (void)numBlocks;
    __shared__ unsigned int pivot[TILE][TILE];
    __shared__ unsigned int tile[TILE][TILE];

    const int tx = (int)threadIdx.x;
    const int ty = (int)threadIdx.y;

    const int bj = map_excluding_round((int)blockIdx.x, r);

    const int row = r * TILE + ty;
    const int col = bj * TILE + tx;

    // Load pivot (r,r)
    {
        const int pr = r * TILE + ty;
        const int pc = r * TILE + tx;
        pivot[ty][tx] = (pr < n && pc < n) ? dist[(size_t)pr * n + pc] : INF;
    }

    // Load tile (r,bj)
    tile[ty][tx] = (row < n && col < n) ? dist[(size_t)row * n + col] : INF;
    __syncthreads();

    for (int k = 0; k < TILE; ++k) {
        const unsigned int through = pivot[ty][k] + tile[k][tx];
        if (through < tile[ty][tx]) {
            tile[ty][tx] = through;
            if (row < n && col < n) {
                path[(size_t)row * n + col] = (unsigned int)(r * TILE + k);
            }
        }
        __syncthreads();
    }

    if (row < n && col < n) {
        dist[(size_t)row * n + col] = tile[ty][tx];
    }
}

__global__ void fw_phase2_col(unsigned int* __restrict__ dist,
                              unsigned int* __restrict__ path,
                              const int n,
                              const int r,
                              const int numBlocks) {
    (void)numBlocks;
    __shared__ unsigned int pivot[TILE][TILE];
    __shared__ unsigned int tile[TILE][TILE];

    const int tx = (int)threadIdx.x;
    const int ty = (int)threadIdx.y;

    const int bi = map_excluding_round((int)blockIdx.x, r);

    const int row = bi * TILE + ty;
    const int col = r * TILE + tx;

    // Load pivot (r,r)
    {
        const int pr = r * TILE + ty;
        const int pc = r * TILE + tx;
        pivot[ty][tx] = (pr < n && pc < n) ? dist[(size_t)pr * n + pc] : INF;
    }

    // Load tile (bi,r)
    tile[ty][tx] = (row < n && col < n) ? dist[(size_t)row * n + col] : INF;
    __syncthreads();

    for (int k = 0; k < TILE; ++k) {
        const unsigned int through = tile[ty][k] + pivot[k][tx];
        if (through < tile[ty][tx]) {
            tile[ty][tx] = through;
            if (row < n && col < n) {
                path[(size_t)row * n + col] = (unsigned int)(r * TILE + k);
            }
        }
        __syncthreads();
    }

    if (row < n && col < n) {
        dist[(size_t)row * n + col] = tile[ty][tx];
    }
}

__global__ void fw_phase3(unsigned int* __restrict__ dist,
                          unsigned int* __restrict__ path,
                          const int n,
                          const int r,
                          const int numBlocks) {
    (void)numBlocks;
    __shared__ unsigned int a[TILE][TILE];
    __shared__ unsigned int b[TILE][TILE];

    const int tx = (int)threadIdx.x;
    const int ty = (int)threadIdx.y;

    const int bi = map_excluding_round((int)blockIdx.y, r);
    const int bj = map_excluding_round((int)blockIdx.x, r);

    const int row = bi * TILE + ty;
    const int col = bj * TILE + tx;

    // a = (bi,r), b = (r,bj)
    {
        const int ar = bi * TILE + ty;
        const int ac = r * TILE + tx;
        a[ty][tx] = (ar < n && ac < n) ? dist[(size_t)ar * n + ac] : INF;
    }
    {
        const int br = r * TILE + ty;
        const int bc = bj * TILE + tx;
        b[ty][tx] = (br < n && bc < n) ? dist[(size_t)br * n + bc] : INF;
    }

    unsigned int cur = (row < n && col < n) ? dist[(size_t)row * n + col] : INF;
    __syncthreads();

    #pragma unroll
    for (int k = 0; k < TILE; ++k) {
        const unsigned int through = a[ty][k] + b[k][tx];
        if (through < cur) {
            cur = through;
            if (row < n && col < n) {
                path[(size_t)row * n + col] = (unsigned int)(r * TILE + k);
            }
        }
    }

    if (row < n && col < n) {
        dist[(size_t)row * n + col] = cur;
    }
}

static void floydWarshallCUDA(unsigned int* dDist, unsigned int* dPath, const int n) {
    const int numBlocks = (n + TILE - 1) / TILE;
    const dim3 block(TILE, TILE);

    for (int r = 0; r < numBlocks; ++r) {
        fw_phase1<<<1, block>>>(dDist, dPath, n, r);
        CUDA_CHECK(cudaGetLastError());

        if (numBlocks > 1) {
            fw_phase2_row<<<dim3(numBlocks - 1, 1, 1), block>>>(dDist, dPath, n, r, numBlocks);
            CUDA_CHECK(cudaGetLastError());

            fw_phase2_col<<<dim3(numBlocks - 1, 1, 1), block>>>(dDist, dPath, n, r, numBlocks);
            CUDA_CHECK(cudaGetLastError());

            fw_phase3<<<dim3(numBlocks - 1, numBlocks - 1, 1), block>>>(dDist, dPath, n, r, numBlocks);
            CUDA_CHECK(cudaGetLastError());
        }
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
    
    // Run Floyd-Warshall (CUDA)
    printf("Computing shortest paths...\n");

    const int n = (int)numNodes;
    const size_t bytes = (size_t)n * (size_t)n * sizeof(unsigned int);

    unsigned int* dDist = nullptr;
    unsigned int* dPath = nullptr;
    CUDA_CHECK(cudaMalloc(&dDist, bytes));
    CUDA_CHECK(cudaMalloc(&dPath, bytes));

    CUDA_CHECK(cudaMemcpy(dDist, dist.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dPath, path.data(), bytes, cudaMemcpyHostToDevice));

    cudaEvent_t evStart{}, evStop{};
    CUDA_CHECK(cudaEventCreate(&evStart));
    CUDA_CHECK(cudaEventCreate(&evStop));

    CUDA_CHECK(cudaEventRecord(evStart));
    floydWarshallCUDA(dDist, dPath, n);
    CUDA_CHECK(cudaEventRecord(evStop));
    CUDA_CHECK(cudaEventSynchronize(evStop));

    float elapsedMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedMs, evStart, evStop));

    CUDA_CHECK(cudaEventDestroy(evStart));
    CUDA_CHECK(cudaEventDestroy(evStop));

    CUDA_CHECK(cudaMemcpy(dist.data(), dDist, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(path.data(), dPath, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(dDist));
    CUDA_CHECK(cudaFree(dPath));

    printf("Computation time: %.3f ms\n", (double)elapsedMs);

    // Calculate operations per second
    // Floyd-Warshall has O(n³) complexity
    double ops = (double)numNodes * numNodes * numNodes;
    double gflops = ops / ((double)elapsedMs / 1000.0) / 1e9;
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
