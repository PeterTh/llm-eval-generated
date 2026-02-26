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
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// Index calculation for flattened 2D array (column-major)
inline size_t idx2(size_t i, size_t j, size_t n) {
    return j * n + i;
}

void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes, 
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;
    
    for (size_t i = 0; i < numNodes * numNodes; ++i) {
        dist[i] = rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
    }
    
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

// =============== Blocked Floyd-Warshall CUDA Kernels ===============
// Uses 3-phase approach per k-block for shared-memory optimization.
// The original code accesses dist as dist[idx2(j,i,n)] = dist[i*n+j],
// so dist[i*n+j] = distance from source i to destination j (row-major).

// Phase 1: Self-dependent diagonal block
__global__ void fw_phase1(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
                          int n, int k_block) {
    __shared__ unsigned int sDist[BLOCK_SIZE][BLOCK_SIZE + 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int i = k_block * BLOCK_SIZE + ty;
    int j = k_block * BLOCK_SIZE + tx;
    int k_base = k_block * BLOCK_SIZE;

    sDist[ty][tx] = (i < n && j < n) ? dist[i * n + j] : INF;
    unsigned int myPath = (i < n && j < n) ? path[i * n + j] : 0;
    __syncthreads();

    #pragma unroll
    for (int kk = 0; kk < BLOCK_SIZE; kk++) {
        unsigned int newDist = sDist[ty][kk] + sDist[kk][tx];
        if (newDist < sDist[ty][tx]) {
            sDist[ty][tx] = newDist;
            myPath = k_base + kk;
        }
        __syncthreads();
    }

    if (i < n && j < n) {
        dist[i * n + j] = sDist[ty][tx];
        path[i * n + j] = myPath;
    }
}

// Phase 2: Row and column blocks sharing a dimension with the diagonal
// Grid: (numBlocks, 2) — y=0 for row blocks, y=1 for column blocks
__global__ void fw_phase2(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
                          int n, int k_block, int numBlocks) {
    int bx = blockIdx.x;
    if (bx == k_block) return;

    __shared__ unsigned int sDiag[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ unsigned int sCurr[BLOCK_SIZE][BLOCK_SIZE + 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int k_base = k_block * BLOCK_SIZE;

    {
        int di = k_base + ty;
        int dj = k_base + tx;
        sDiag[ty][tx] = (di < n && dj < n) ? dist[di * n + dj] : INF;
    }

    int ci, cj;
    if (blockIdx.y == 0) {
        ci = k_base + ty;
        cj = bx * BLOCK_SIZE + tx;
    } else {
        ci = bx * BLOCK_SIZE + ty;
        cj = k_base + tx;
    }

    sCurr[ty][tx] = (ci < n && cj < n) ? dist[ci * n + cj] : INF;
    unsigned int myPath = (ci < n && cj < n) ? path[ci * n + cj] : 0;
    __syncthreads();

    if (blockIdx.y == 0) {
        #pragma unroll
        for (int kk = 0; kk < BLOCK_SIZE; kk++) {
            unsigned int newDist = sDiag[ty][kk] + sCurr[kk][tx];
            if (newDist < sCurr[ty][tx]) {
                sCurr[ty][tx] = newDist;
                myPath = k_base + kk;
            }
            __syncthreads();
        }
    } else {
        #pragma unroll
        for (int kk = 0; kk < BLOCK_SIZE; kk++) {
            unsigned int newDist = sCurr[ty][kk] + sDiag[kk][tx];
            if (newDist < sCurr[ty][tx]) {
                sCurr[ty][tx] = newDist;
                myPath = k_base + kk;
            }
            __syncthreads();
        }
    }

    if (ci < n && cj < n) {
        dist[ci * n + cj] = sCurr[ty][tx];
        path[ci * n + cj] = myPath;
    }
}

// Phase 3: All remaining blocks (fully independent)
// Grid: (numBlocks, numBlocks)
__global__ void fw_phase3(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
                          int n, int k_block, int numBlocks) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    if (bx == k_block || by == k_block) return;

    __shared__ unsigned int sRow[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ unsigned int sCol[BLOCK_SIZE][BLOCK_SIZE + 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int k_base = k_block * BLOCK_SIZE;

    {
        int ri = k_base + ty;
        int rj = bx * BLOCK_SIZE + tx;
        sRow[ty][tx] = (ri < n && rj < n) ? dist[ri * n + rj] : INF;
    }
    {
        int ci = by * BLOCK_SIZE + ty;
        int cj = k_base + tx;
        sCol[ty][tx] = (ci < n && cj < n) ? dist[ci * n + cj] : INF;
    }

    __syncthreads();

    int curr_i = by * BLOCK_SIZE + ty;
    int curr_j = bx * BLOCK_SIZE + tx;

    unsigned int currDist = (curr_i < n && curr_j < n) ? dist[curr_i * n + curr_j] : INF;
    unsigned int currPath = (curr_i < n && curr_j < n) ? path[curr_i * n + curr_j] : 0;

    #pragma unroll
    for (int kk = 0; kk < BLOCK_SIZE; kk++) {
        unsigned int newDist = sCol[ty][kk] + sRow[kk][tx];
        if (newDist < currDist) {
            currDist = newDist;
            currPath = k_base + kk;
        }
    }

    if (curr_i < n && curr_j < n) {
        dist[curr_i * n + curr_j] = currDist;
        path[curr_i * n + curr_j] = currPath;
    }
}

// =============== Host wrapper ===============

void floydWarshall(std::vector<unsigned int>& dist, 
                   std::vector<unsigned int>& path, 
                   const size_t numNodes) {
    int n = static_cast<int>(numNodes);
    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t matSize = static_cast<size_t>(n) * n * sizeof(unsigned int);

    unsigned int *d_dist, *d_path;
    CUDA_CHECK(cudaMalloc(&d_dist, matSize));
    CUDA_CHECK(cudaMalloc(&d_path, matSize));
    CUDA_CHECK(cudaMemcpy(d_dist, dist.data(), matSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_path, path.data(), matSize, cudaMemcpyHostToDevice));

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

    for (int k = 0; k < numBlocks; k++) {
        fw_phase1<<<1, threads>>>(d_dist, d_path, n, k);

        dim3 grid2(numBlocks, 2);
        fw_phase2<<<grid2, threads>>>(d_dist, d_path, n, k, numBlocks);

        dim3 grid3(numBlocks, numBlocks);
        fw_phase3<<<grid3, threads>>>(d_dist, d_path, n, k, numBlocks);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(dist.data(), d_dist, matSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(path.data(), d_path, matSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_dist));
    CUDA_CHECK(cudaFree(d_path));
}

bool validateResult(const std::vector<unsigned int>& dist, const size_t numNodes) {
    for (size_t i = 0; i < numNodes; ++i) {
        if (dist[idx2(i, i, numNodes)] != 0) {
            printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", i, i);
            return false;
        }
    }
    
    for (size_t i = 0; i < std::min(numNodes, static_cast<size_t>(10)); ++i) {
        for (size_t j = 0; j < std::min(numNodes, static_cast<size_t>(10)); ++j) {
            for (size_t k = 0; k < numNodes; ++k) {
                const unsigned int distIJ = dist[idx2(j, i, numNodes)];
                const unsigned int distIK = dist[idx2(k, i, numNodes)];
                const unsigned int distKJ = dist[idx2(j, k, numNodes)];
                
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
    
    // Warm up CUDA context before timing
    CUDA_CHECK(cudaFree(0));

    printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
    printf("Number of nodes: %zu\n", numNodes);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    std::vector<unsigned int> dist(numNodes * numNodes);
    std::vector<unsigned int> path(numNodes * numNodes);
    
    printf("Initializing graph...\n");
    initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
    initializePathMatrix(path, numNodes);
    
    printf("Computing shortest paths...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    floydWarshall(dist, path, numNodes);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    double ops = (double)numNodes * numNodes * numNodes;
    double gflops = ops / (duration.count() / 1000.0) / 1e9;
    printf("Performance: %.3f GOPS\n", gflops);
    
    if (printResults) {
        print_results_int(dist, "DistanceMatrix");
    }
    
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
