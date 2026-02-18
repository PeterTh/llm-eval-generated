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
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Index calculation for flattened 2D array
inline size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
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

// Phase 1: Update diagonal block (kb, kb) - self-dependent
__global__ void fw_phase1(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
                          const int n, const int kb) {
    // +1 padding avoids shared memory bank conflicts on column access
    __shared__ unsigned int sd[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ unsigned int sp[BLOCK_SIZE][BLOCK_SIZE + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int base = kb * BLOCK_SIZE;
    const int i = base + ty;
    const int j = base + tx;

    if (i < n && j < n) {
        sd[ty][tx] = dist[i * n + j];
        sp[ty][tx] = path[i * n + j];
    } else {
        sd[ty][tx] = INF;
        sp[ty][tx] = 0;
    }
    __syncthreads();

    #pragma unroll
    for (int k = 0; k < BLOCK_SIZE; k++) {
        if (base + k < n) {
            unsigned int nd = sd[ty][k] + sd[k][tx];
            if (nd < sd[ty][tx]) {
                sd[ty][tx] = nd;
                sp[ty][tx] = base + k;
            }
        }
        __syncthreads();
    }

    if (i < n && j < n) {
        dist[i * n + j] = sd[ty][tx];
        path[i * n + j] = sp[ty][tx];
    }
}

// Phase 2: Update row and column blocks - singly-dependent on diagonal
__global__ void fw_phase2(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
                          const int n, const int kb) {
    // blockIdx.y: 0 = row block, 1 = column block
    int b = blockIdx.x;
    if (b >= kb) b++;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int base = kb * BLOCK_SIZE;

    __shared__ unsigned int sd_diag[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ unsigned int sd_cur[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ unsigned int sp_cur[BLOCK_SIZE][BLOCK_SIZE + 1];

    {
        int di = base + ty, dj = base + tx;
        sd_diag[ty][tx] = (di < n && dj < n) ? dist[di * n + dj] : INF;
    }

    int i, j;
    if (blockIdx.y == 0) {
        i = base + ty;
        j = b * BLOCK_SIZE + tx;
    } else {
        i = b * BLOCK_SIZE + ty;
        j = base + tx;
    }

    sd_cur[ty][tx] = (i < n && j < n) ? dist[i * n + j] : INF;
    sp_cur[ty][tx] = (i < n && j < n) ? path[i * n + j] : 0;
    __syncthreads();

    #pragma unroll
    for (int k = 0; k < BLOCK_SIZE; k++) {
        if (base + k < n) {
            unsigned int d_ik, d_kj;
            if (blockIdx.y == 0) {
                d_ik = sd_diag[ty][k];
                d_kj = sd_cur[k][tx];
            } else {
                d_ik = sd_cur[ty][k];
                d_kj = sd_diag[k][tx];
            }
            unsigned int nd = d_ik + d_kj;
            if (nd < sd_cur[ty][tx]) {
                sd_cur[ty][tx] = nd;
                sp_cur[ty][tx] = base + k;
            }
        }
        __syncthreads();
    }

    if (i < n && j < n) {
        dist[i * n + j] = sd_cur[ty][tx];
        path[i * n + j] = sp_cur[ty][tx];
    }
}

// Phase 3: Update all remaining blocks - reads from updated row/column blocks
__global__ void fw_phase3(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
                          const int n, const int kb) {
    int bj = blockIdx.x;
    int bi = blockIdx.y;
    if (bj >= kb) bj++;
    if (bi >= kb) bi++;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int base = kb * BLOCK_SIZE;

    __shared__ unsigned int sd_col[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ unsigned int sd_row[BLOCK_SIZE][BLOCK_SIZE + 1];

    {
        int ci = bi * BLOCK_SIZE + ty, cj = base + tx;
        sd_col[ty][tx] = (ci < n && cj < n) ? dist[ci * n + cj] : INF;
    }
    {
        int ri = base + ty, rj = bj * BLOCK_SIZE + tx;
        sd_row[ty][tx] = (ri < n && rj < n) ? dist[ri * n + rj] : INF;
    }
    __syncthreads();

    const int i = bi * BLOCK_SIZE + ty;
    const int j = bj * BLOCK_SIZE + tx;

    unsigned int cur = (i < n && j < n) ? dist[i * n + j] : INF;
    unsigned int curp = (i < n && j < n) ? path[i * n + j] : 0u;

    #pragma unroll
    for (int k = 0; k < BLOCK_SIZE; k++) {
        if (base + k < n) {
            unsigned int nd = sd_col[ty][k] + sd_row[k][tx];
            if (nd < cur) {
                cur = nd;
                curp = base + k;
            }
        }
    }

    if (i < n && j < n) {
        dist[i * n + j] = cur;
        path[i * n + j] = curp;
    }
}

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes) {
    const int n = static_cast<int>(numNodes);
    const size_t matSize = numNodes * numNodes * sizeof(unsigned int);
    const int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    unsigned int *d_dist, *d_path;
    CUDA_CHECK(cudaMalloc(&d_dist, matSize));
    CUDA_CHECK(cudaMalloc(&d_path, matSize));

    CUDA_CHECK(cudaMemcpy(d_dist, dist.data(), matSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_path, path.data(), matSize, cudaMemcpyHostToDevice));

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

    for (int kb = 0; kb < numBlocks; kb++) {
        fw_phase1<<<1, threads>>>(d_dist, d_path, n, kb);

        if (numBlocks > 1) {
            dim3 grid2(numBlocks - 1, 2);
            fw_phase2<<<grid2, threads>>>(d_dist, d_path, n, kb);

            dim3 grid3(numBlocks - 1, numBlocks - 1);
            fw_phase3<<<grid3, threads>>>(d_dist, d_path, n, kb);
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
