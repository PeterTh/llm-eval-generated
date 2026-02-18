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

#define BLOCK_SIZE 32

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// Index calculation for flattened 2D array
inline size_t idx2(const size_t i, const size_t j, const size_t n) {
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

// Blocked Floyd-Warshall Phase 1: Process the pivot block (bk, bk)
__global__ void fw_phase1(unsigned int* __restrict__ dist,
                          unsigned int* __restrict__ path,
                          const int n, const int bk) {
    __shared__ unsigned int sd[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int sp[BLOCK_SIZE][BLOCK_SIZE];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int i = bk * BLOCK_SIZE + ty;
    const int j = bk * BLOCK_SIZE + tx;

    sd[ty][tx] = (i < n && j < n) ? dist[i * n + j] : INF;
    sp[ty][tx] = (i < n && j < n) ? path[i * n + j] : 0;
    __syncthreads();

    for (int kk = 0; kk < BLOCK_SIZE; kk++) {
        unsigned int nd = sd[ty][kk] + sd[kk][tx];
        if (nd < sd[ty][tx]) {
            sd[ty][tx] = nd;
            sp[ty][tx] = bk * BLOCK_SIZE + kk;
        }
        __syncthreads();
    }

    if (i < n && j < n) {
        dist[i * n + j] = sd[ty][tx];
        path[i * n + j] = sp[ty][tx];
    }
}

// Phase 2: Process blocks in the same row and column as the pivot
__global__ void fw_phase2(unsigned int* __restrict__ dist,
                          unsigned int* __restrict__ path,
                          const int n, const int bk) {
    int bOther = blockIdx.x;
    if (bOther >= bk) bOther++;
    const int isCol = blockIdx.y; // 0 = row block, 1 = column block

    const int tx = threadIdx.x, ty = threadIdx.y;

    __shared__ unsigned int sdPivot[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int sd[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int sp[BLOCK_SIZE][BLOCK_SIZE];

    {
        const int pi = bk * BLOCK_SIZE + ty;
        const int pj = bk * BLOCK_SIZE + tx;
        sdPivot[ty][tx] = (pi < n && pj < n) ? dist[pi * n + pj] : INF;
    }

    int i, j;
    if (isCol == 0) {
        i = bk * BLOCK_SIZE + ty;
        j = bOther * BLOCK_SIZE + tx;
    } else {
        i = bOther * BLOCK_SIZE + ty;
        j = bk * BLOCK_SIZE + tx;
    }

    sd[ty][tx] = (i < n && j < n) ? dist[i * n + j] : INF;
    sp[ty][tx] = (i < n && j < n) ? path[i * n + j] : 0;
    __syncthreads();

    if (isCol == 0) {
        // Row block: dist[i][k] from pivot, dist[k][j] from self
        for (int kk = 0; kk < BLOCK_SIZE; kk++) {
            unsigned int nd = sdPivot[ty][kk] + sd[kk][tx];
            if (nd < sd[ty][tx]) {
                sd[ty][tx] = nd;
                sp[ty][tx] = bk * BLOCK_SIZE + kk;
            }
            __syncthreads();
        }
    } else {
        // Column block: dist[i][k] from self, dist[k][j] from pivot
        for (int kk = 0; kk < BLOCK_SIZE; kk++) {
            unsigned int nd = sd[ty][kk] + sdPivot[kk][tx];
            if (nd < sd[ty][tx]) {
                sd[ty][tx] = nd;
                sp[ty][tx] = bk * BLOCK_SIZE + kk;
            }
            __syncthreads();
        }
    }

    if (i < n && j < n) {
        dist[i * n + j] = sd[ty][tx];
        path[i * n + j] = sp[ty][tx];
    }
}

// Phase 3: Process all remaining blocks
__global__ void fw_phase3(unsigned int* __restrict__ dist,
                          unsigned int* __restrict__ path,
                          const int n, const int bk) {
    int bj = blockIdx.x;
    if (bj >= bk) bj++;
    int bi = blockIdx.y;
    if (bi >= bk) bi++;

    const int tx = threadIdx.x, ty = threadIdx.y;

    __shared__ unsigned int sdRow[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int sdCol[BLOCK_SIZE][BLOCK_SIZE];

    {
        const int ri = bk * BLOCK_SIZE + ty;
        const int rj = bj * BLOCK_SIZE + tx;
        sdRow[ty][tx] = (ri < n && rj < n) ? dist[ri * n + rj] : INF;
    }
    {
        const int ci = bi * BLOCK_SIZE + ty;
        const int cj = bk * BLOCK_SIZE + tx;
        sdCol[ty][tx] = (ci < n && cj < n) ? dist[ci * n + cj] : INF;
    }

    __syncthreads();

    const int i = bi * BLOCK_SIZE + ty;
    const int j = bj * BLOCK_SIZE + tx;

    if (i < n && j < n) {
        const int idx = i * n + j;
        unsigned int dij = dist[idx];
        unsigned int pij = path[idx];

        #pragma unroll
        for (int kk = 0; kk < BLOCK_SIZE; kk++) {
            unsigned int nd = sdCol[ty][kk] + sdRow[kk][tx];
            if (nd < dij) {
                dij = nd;
                pij = bk * BLOCK_SIZE + kk;
            }
        }

        dist[idx] = dij;
        path[idx] = pij;
    }
}

void floydWarshall(std::vector<unsigned int>& dist, 
                   std::vector<unsigned int>& path, 
                   const size_t numNodes) {
    const size_t matSize = numNodes * numNodes * sizeof(unsigned int);
    const int n = static_cast<int>(numNodes);
    const int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    unsigned int *d_dist, *d_path;
    CUDA_CHECK(cudaMalloc(&d_dist, matSize));
    CUDA_CHECK(cudaMalloc(&d_path, matSize));

    CUDA_CHECK(cudaMemcpy(d_dist, dist.data(), matSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_path, path.data(), matSize, cudaMemcpyHostToDevice));

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

    for (int bk = 0; bk < numBlocks; bk++) {
        fw_phase1<<<1, threads>>>(d_dist, d_path, n, bk);

        if (numBlocks > 1) {
            dim3 grid2(numBlocks - 1, 2);
            fw_phase2<<<grid2, threads>>>(d_dist, d_path, n, bk);

            dim3 grid3(numBlocks - 1, numBlocks - 1);
            fw_phase3<<<grid3, threads>>>(d_dist, d_path, n, bk);
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(dist.data(), d_dist, matSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(path.data(), d_path, matSize, cudaMemcpyDeviceToHost));

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
