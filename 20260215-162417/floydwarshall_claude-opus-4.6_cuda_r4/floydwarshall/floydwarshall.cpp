#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

#define BLOCK_SIZE 32

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

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

// Blocked Floyd-Warshall Phase 1: self-dependent diagonal block
__global__ void phase1Kernel(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
                             int n, int b) {
    __shared__ unsigned int sd[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ unsigned int sp[BLOCK_SIZE][BLOCK_SIZE + 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int j = b * BLOCK_SIZE + tx;
    int i = b * BLOCK_SIZE + ty;

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
        int gk = b * BLOCK_SIZE + k;
        if (gk < n) {
            unsigned int newDist = sd[ty][k] + sd[k][tx];
            if (newDist < sd[ty][tx]) {
                sd[ty][tx] = newDist;
                sp[ty][tx] = gk;
            }
        }
        __syncthreads();
    }

    if (i < n && j < n) {
        dist[i * n + j] = sd[ty][tx];
        path[i * n + j] = sp[ty][tx];
    }
}

// Phase 2: row and column blocks that share a dimension with the diagonal
__global__ void phase2Kernel(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
                             int n, int b) {
    int bIdx = blockIdx.x;
    if (bIdx == b) return;
    int isCol = blockIdx.y;  // 0 = row block, 1 = column block

    __shared__ unsigned int sd_diag[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ unsigned int sd[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ unsigned int sp[BLOCK_SIZE][BLOCK_SIZE + 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int bi = (isCol == 0) ? b : bIdx;
    int bj = (isCol == 0) ? bIdx : b;

    int i = bi * BLOCK_SIZE + ty;
    int j = bj * BLOCK_SIZE + tx;

    int di = b * BLOCK_SIZE + ty;
    int dj = b * BLOCK_SIZE + tx;
    sd_diag[ty][tx] = (di < n && dj < n) ? dist[di * n + dj] : INF;

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
        int gk = b * BLOCK_SIZE + k;
        if (gk < n) {
            unsigned int dik, dkj;
            if (isCol == 0) {
                dik = sd_diag[ty][k];
                dkj = sd[k][tx];
            } else {
                dik = sd[ty][k];
                dkj = sd_diag[k][tx];
            }
            unsigned int newDist = dik + dkj;
            if (newDist < sd[ty][tx]) {
                sd[ty][tx] = newDist;
                sp[ty][tx] = gk;
            }
        }
        __syncthreads();
    }

    if (i < n && j < n) {
        dist[i * n + j] = sd[ty][tx];
        path[i * n + j] = sp[ty][tx];
    }
}

// Phase 3: all remaining blocks
__global__ void phase3Kernel(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
                             int n, int b) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    if (bx == b || by == b) return;

    __shared__ unsigned int sd_row[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ unsigned int sd_col[BLOCK_SIZE][BLOCK_SIZE + 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Load column-reference block (by, b)
    int ri = by * BLOCK_SIZE + ty;
    int rj = b * BLOCK_SIZE + tx;
    sd_row[ty][tx] = (ri < n && rj < n) ? dist[ri * n + rj] : INF;

    // Load row-reference block (b, bx)
    int ci = b * BLOCK_SIZE + ty;
    int cj = bx * BLOCK_SIZE + tx;
    sd_col[ty][tx] = (ci < n && cj < n) ? dist[ci * n + cj] : INF;

    __syncthreads();

    int i = by * BLOCK_SIZE + ty;
    int j = bx * BLOCK_SIZE + tx;

    unsigned int curDist, curPath;
    if (i < n && j < n) {
        curDist = dist[i * n + j];
        curPath = path[i * n + j];
    } else {
        curDist = INF;
        curPath = 0;
    }

    #pragma unroll
    for (int k = 0; k < BLOCK_SIZE; k++) {
        int gk = b * BLOCK_SIZE + k;
        if (gk < n) {
            unsigned int newDist = sd_row[ty][k] + sd_col[k][tx];
            if (newDist < curDist) {
                curDist = newDist;
                curPath = gk;
            }
        }
    }

    if (i < n && j < n) {
        dist[i * n + j] = curDist;
        path[i * n + j] = curPath;
    }
}

void floydWarshall(unsigned int* d_dist, unsigned int* d_path, int n) {
    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

    for (int b = 0; b < numBlocks; b++) {
        phase1Kernel<<<1, threads>>>(d_dist, d_path, n, b);

        dim3 grid2(numBlocks, 2);
        phase2Kernel<<<grid2, threads>>>(d_dist, d_path, n, b);

        dim3 grid3(numBlocks, numBlocks);
        phase3Kernel<<<grid3, threads>>>(d_dist, d_path, n, b);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
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

    printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
    printf("Number of nodes: %zu\n", numNodes);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");

    std::vector<unsigned int> dist(numNodes * numNodes);
    std::vector<unsigned int> path(numNodes * numNodes);

    printf("Initializing graph...\n");
    initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
    initializePathMatrix(path, numNodes);

    // Allocate device memory and transfer data
    unsigned int *d_dist, *d_path;
    size_t matSize = numNodes * numNodes * sizeof(unsigned int);
    CUDA_CHECK(cudaMalloc(&d_dist, matSize));
    CUDA_CHECK(cudaMalloc(&d_path, matSize));
    CUDA_CHECK(cudaMemcpy(d_dist, dist.data(), matSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_path, path.data(), matSize, cudaMemcpyHostToDevice));

    printf("Computing shortest paths...\n");
    auto start = std::chrono::high_resolution_clock::now();

    floydWarshall(d_dist, d_path, (int)numNodes);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(dist.data(), d_dist, matSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(path.data(), d_path, matSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_dist));
    CUDA_CHECK(cudaFree(d_path));

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
