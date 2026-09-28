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

// Tile size used for the blocked GPU Floyd-Warshall algorithm.
constexpr int BLOCK_SIZE = 32;

// Index calculation for flattened 2D array
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

#define CUDA_CHECK(call)                                                             \
    do {                                                                             \
        cudaError_t err__ = (call);                                                  \
        if (err__ != cudaSuccess) {                                                  \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
                    cudaGetErrorString(err__));                                       \
            exit(1);                                                                 \
        }                                                                            \
    } while (0)

// Blocked Floyd-Warshall GPU kernels (Katz & Kider style tiled APSP).
// Matrices are stored row-major: element (row, col) lives at row * n + col,
// which matches dist[idx2(col, row, n)] / path[idx2(col, row, n)] used by the
// original scalar implementation.

// Phase 1: relax the pivot (diagonal) tile against itself.
__global__ void fwPhase1(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
                          int n, int round) {
    __shared__ unsigned int primary_d[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int primary_p[BLOCK_SIZE][BLOCK_SIZE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int base = round * BLOCK_SIZE;
    const int row = base + ty;
    const int col = base + tx;
    const bool valid = row < n && col < n;

    primary_d[ty][tx] = valid ? dist[static_cast<size_t>(row) * n + col] : INF;
    primary_p[ty][tx] = valid ? path[static_cast<size_t>(row) * n + col] : 0u;
    __syncthreads();

#pragma unroll
    for (int k = 0; k < BLOCK_SIZE; ++k) {
        const unsigned int cand = primary_d[ty][k] + primary_d[k][tx];
        if (cand < primary_d[ty][tx]) {
            primary_d[ty][tx] = cand;
            primary_p[ty][tx] = static_cast<unsigned int>(base + k);
        }
        __syncthreads();
    }

    if (valid) {
        dist[static_cast<size_t>(row) * n + col] = primary_d[ty][tx];
        path[static_cast<size_t>(row) * n + col] = primary_p[ty][tx];
    }
}

// Phase 2: relax the pivot row-band and column-band tiles against the pivot tile.
// gridDim.y selects row-band (0) vs column-band (1); gridDim.x selects the band index.
__global__ void fwPhase2(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
                          int n, int round) {
    const int blk = blockIdx.x;
    if (blk == round) return;
    const bool isRow = (blockIdx.y == 0);

    __shared__ unsigned int primary_d[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int other_d[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int other_p[BLOCK_SIZE][BLOCK_SIZE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int base = round * BLOCK_SIZE;

    const int prow = base + ty;
    const int pcol = base + tx;
    const bool pvalid = prow < n && pcol < n;
    primary_d[ty][tx] = pvalid ? dist[static_cast<size_t>(prow) * n + pcol] : INF;

    int row, col;
    if (isRow) {
        row = base + ty;
        col = blk * BLOCK_SIZE + tx;
    } else {
        row = blk * BLOCK_SIZE + ty;
        col = base + tx;
    }
    const bool valid = row < n && col < n;
    other_d[ty][tx] = valid ? dist[static_cast<size_t>(row) * n + col] : INF;
    other_p[ty][tx] = valid ? path[static_cast<size_t>(row) * n + col] : 0u;
    __syncthreads();

#pragma unroll
    for (int k = 0; k < BLOCK_SIZE; ++k) {
        const unsigned int cand =
            isRow ? primary_d[ty][k] + other_d[k][tx] : other_d[ty][k] + primary_d[k][tx];
        if (cand < other_d[ty][tx]) {
            other_d[ty][tx] = cand;
            other_p[ty][tx] = static_cast<unsigned int>(base + k);
        }
        __syncthreads();
    }

    if (valid) {
        dist[static_cast<size_t>(row) * n + col] = other_d[ty][tx];
        path[static_cast<size_t>(row) * n + col] = other_p[ty][tx];
    }
}

// Phase 3: relax all remaining tiles using the updated row-band / column-band tiles.
__global__ void fwPhase3(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
                          int n, int round) {
    const int bi = blockIdx.y;
    const int bj = blockIdx.x;
    if (bi == round || bj == round) return;

    __shared__ unsigned int rowBlock_d[BLOCK_SIZE][BLOCK_SIZE];  // block(bi, round): dist(i,k)
    __shared__ unsigned int colBlock_d[BLOCK_SIZE][BLOCK_SIZE];  // block(round, bj): dist(k,j)

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int base = round * BLOCK_SIZE;

    const int rrow = bi * BLOCK_SIZE + ty;
    const int rcol = base + tx;
    const bool rvalid = rrow < n && rcol < n;
    rowBlock_d[ty][tx] = rvalid ? dist[static_cast<size_t>(rrow) * n + rcol] : INF;

    const int crow = base + ty;
    const int ccol = bj * BLOCK_SIZE + tx;
    const bool cvalid = crow < n && ccol < n;
    colBlock_d[ty][tx] = cvalid ? dist[static_cast<size_t>(crow) * n + ccol] : INF;
    __syncthreads();

    const int row = bi * BLOCK_SIZE + ty;
    const int col = bj * BLOCK_SIZE + tx;
    const bool valid = row < n && col < n;

    unsigned int d = valid ? dist[static_cast<size_t>(row) * n + col] : INF;
    unsigned int p = valid ? path[static_cast<size_t>(row) * n + col] : 0u;

#pragma unroll
    for (int k = 0; k < BLOCK_SIZE; ++k) {
        const unsigned int cand = rowBlock_d[ty][k] + colBlock_d[k][tx];
        if (cand < d) {
            d = cand;
            p = static_cast<unsigned int>(base + k);
        }
    }

    if (valid) {
        dist[static_cast<size_t>(row) * n + col] = d;
        path[static_cast<size_t>(row) * n + col] = p;
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

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes) {
    // Blocked Floyd-Warshall on the GPU: dist[idx2(j,i,n)] == row-major dist(i,j)
    // (see idx2), so the device kernels operate directly on that row-major layout.
    const int n = static_cast<int>(numNodes);
    const size_t bytes = numNodes * numNodes * sizeof(unsigned int);
    const int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    unsigned int *d_dist = nullptr, *d_path = nullptr;
    CUDA_CHECK(cudaMalloc(&d_dist, bytes));
    CUDA_CHECK(cudaMalloc(&d_path, bytes));

    CUDA_CHECK(cudaMemcpy(d_dist, dist.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_path, path.data(), bytes, cudaMemcpyHostToDevice));

    const dim3 tileDim(BLOCK_SIZE, BLOCK_SIZE);
    const dim3 phase2Grid(numBlocks, 2);
    const dim3 phase3Grid(numBlocks, numBlocks);

    for (int round = 0; round < numBlocks; ++round) {
        fwPhase1<<<1, tileDim>>>(d_dist, d_path, n, round);
        fwPhase2<<<phase2Grid, tileDim>>>(d_dist, d_path, n, round);
        fwPhase3<<<phase3Grid, tileDim>>>(d_dist, d_path, n, round);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(dist.data(), d_dist, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(path.data(), d_path, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_dist));
    CUDA_CHECK(cudaFree(d_path));
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
