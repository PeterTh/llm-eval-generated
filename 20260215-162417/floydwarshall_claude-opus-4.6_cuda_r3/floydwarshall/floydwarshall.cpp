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
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Index calculation for flattened 2D array
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

// Phase 1: Self-dependent diagonal block
__global__ void fw_phase1(unsigned int* __restrict__ dist,
                          unsigned int* __restrict__ path,
                          const int n, const int round) {
    __shared__ unsigned int sd[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int sp[BLOCK_SIZE][BLOCK_SIZE];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int base = round * BLOCK_SIZE;
    const int gi = base + ty, gj = base + tx;

    if (gi < n && gj < n) {
        sd[ty][tx] = dist[(size_t)gi * n + gj];
        sp[ty][tx] = path[(size_t)gi * n + gj];
    } else {
        sd[ty][tx] = INF;
        sp[ty][tx] = 0;
    }
    __syncthreads();

    for (int k = 0; k < BLOCK_SIZE; ++k) {
        unsigned int nd = sd[ty][k] + sd[k][tx];
        if (nd < sd[ty][tx]) {
            sd[ty][tx] = nd;
            sp[ty][tx] = (unsigned int)(base + k);
        }
        __syncthreads();
    }

    if (gi < n && gj < n) {
        dist[(size_t)gi * n + gj] = sd[ty][tx];
        path[(size_t)gi * n + gj] = sp[ty][tx];
    }
}

// Phase 2: Row and column blocks adjacent to the diagonal
// blockIdx.x: block index (0..numBlocks-2, maps to actual index skipping round)
// blockIdx.y: 0 = row block, 1 = column block
__global__ void fw_phase2(unsigned int* __restrict__ dist,
                          unsigned int* __restrict__ path,
                          const int n, const int round) {
    int bIdx = blockIdx.x;
    if (bIdx >= round) bIdx++;

    __shared__ unsigned int sDiag[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int sCurr[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int sCurrP[BLOCK_SIZE][BLOCK_SIZE];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int base = round * BLOCK_SIZE;
    const int bBase = bIdx * BLOCK_SIZE;

    // Load diagonal block
    {
        int di = base + ty, dj = base + tx;
        sDiag[ty][tx] = (di < n && dj < n) ? dist[(size_t)di * n + dj] : INF;
    }

    int gi, gj;
    if (blockIdx.y == 0) {
        // Row block (round, bIdx): rows [base..], cols [bBase..]
        gi = base + ty;
        gj = bBase + tx;
    } else {
        // Column block (bIdx, round): rows [bBase..], cols [base..]
        gi = bBase + ty;
        gj = base + tx;
    }

    if (gi < n && gj < n) {
        sCurr[ty][tx] = dist[(size_t)gi * n + gj];
        sCurrP[ty][tx] = path[(size_t)gi * n + gj];
    } else {
        sCurr[ty][tx] = INF;
        sCurrP[ty][tx] = 0;
    }
    __syncthreads();

    for (int k = 0; k < BLOCK_SIZE; ++k) {
        unsigned int nd;
        if (blockIdx.y == 0) {
            // Row: dist[i][k'] from diag, dist[k'][j] from curr
            nd = sDiag[ty][k] + sCurr[k][tx];
        } else {
            // Col: dist[i][k'] from curr, dist[k'][j] from diag
            nd = sCurr[ty][k] + sDiag[k][tx];
        }
        if (nd < sCurr[ty][tx]) {
            sCurr[ty][tx] = nd;
            sCurrP[ty][tx] = (unsigned int)(base + k);
        }
        __syncthreads();
    }

    if (gi < n && gj < n) {
        dist[(size_t)gi * n + gj] = sCurr[ty][tx];
        path[(size_t)gi * n + gj] = sCurrP[ty][tx];
    }
}

// Phase 3: All remaining blocks
// blockIdx.x: column block (0..numBlocks-2, skipping round)
// blockIdx.y: row block (0..numBlocks-2, skipping round)
__global__ void fw_phase3(unsigned int* __restrict__ dist,
                          unsigned int* __restrict__ path,
                          const int n, const int round) {
    int bCol = blockIdx.x;
    int bRow = blockIdx.y;
    if (bCol >= round) bCol++;
    if (bRow >= round) bRow++;

    __shared__ unsigned int sRow[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int sCol[BLOCK_SIZE][BLOCK_SIZE];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int base = round * BLOCK_SIZE;
    const int colBase = bCol * BLOCK_SIZE;
    const int rowBase = bRow * BLOCK_SIZE;

    // Load row block (round, bCol)
    {
        int ri = base + ty, rj = colBase + tx;
        sRow[ty][tx] = (ri < n && rj < n) ? dist[(size_t)ri * n + rj] : INF;
    }
    // Load column block (bRow, round)
    {
        int ci = rowBase + ty, cj = base + tx;
        sCol[ty][tx] = (ci < n && cj < n) ? dist[(size_t)ci * n + cj] : INF;
    }
    __syncthreads();

    const int gi = rowBase + ty;
    const int gj = colBase + tx;

    unsigned int curDist = (gi < n && gj < n) ? dist[(size_t)gi * n + gj] : INF;
    unsigned int curPath = (gi < n && gj < n) ? path[(size_t)gi * n + gj] : 0;

    #pragma unroll
    for (int k = 0; k < BLOCK_SIZE; ++k) {
        unsigned int nd = sCol[ty][k] + sRow[k][tx];
        if (nd < curDist) {
            curDist = nd;
            curPath = (unsigned int)(base + k);
        }
    }

    if (gi < n && gj < n) {
        dist[(size_t)gi * n + gj] = curDist;
        path[(size_t)gi * n + gj] = curPath;
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
    const int n = (int)numNodes;
    const int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const size_t matSize = (size_t)n * n * sizeof(unsigned int);

    unsigned int *d_dist, *d_path;
    CUDA_CHECK(cudaMalloc(&d_dist, matSize));
    CUDA_CHECK(cudaMalloc(&d_path, matSize));
    CUDA_CHECK(cudaMemcpy(d_dist, dist.data(), matSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_path, path.data(), matSize, cudaMemcpyHostToDevice));

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);

    for (int round = 0; round < numBlocks; ++round) {
        // Phase 1: diagonal block
        fw_phase1<<<1, blockDim>>>(d_dist, d_path, n, round);

        if (numBlocks > 1) {
            // Phase 2: row and column blocks
            dim3 grid2(numBlocks - 1, 2);
            fw_phase2<<<grid2, blockDim>>>(d_dist, d_path, n, round);

            // Phase 3: all remaining blocks
            dim3 grid3(numBlocks - 1, numBlocks - 1);
            fw_phase3<<<grid3, blockDim>>>(d_dist, d_path, n, round);
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
