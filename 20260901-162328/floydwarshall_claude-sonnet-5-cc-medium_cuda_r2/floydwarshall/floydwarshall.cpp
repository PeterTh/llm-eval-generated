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
constexpr int BLOCK_SIZE = 32;

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err__ = (call);                                               \
        if (err__ != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err__));                                   \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

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

// Blocked (tiled) Floyd-Warshall for GPU execution (Katz & Kider scheme).
//
// The matrices are stored so that dist[idx2(a, b, n)] == dist[b * n + a],
// i.e. element (row=a, col=b) lives at offset b * n + a. The recurrence
// dist[idx2(j,i)] = min(dist[idx2(j,i)], dist[idx2(k,i)] + dist[idx2(j,k)])
// is exactly the standard FW update D[p][q] = min(D[p][q], D[p][k]+D[k][q])
// with p = j (row) and q = i (col), so the classic 3-phase blocked GPU
// algorithm applies directly to this layout.
//
// Each round processes one diagonal block of intermediate nodes k:
//   Phase 1: update the diagonal block using only itself.
//   Phase 2: update the blocks sharing the diagonal's row/col range.
//   Phase 3: update all remaining blocks using the updated row/col blocks.
// Because rounds run in increasing k order and every phase performs its
// BLOCK_SIZE inner k-steps in increasing order with strict '<' updates,
// the result (including recorded predecessors on ties) is identical to
// the sequential algorithm.

__global__ void fwPhase1(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
                          const int n, const int round) {
    __shared__ unsigned int sDist[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int sPath[BLOCK_SIZE][BLOCK_SIZE];

    const int base = round * BLOCK_SIZE;
    const int tr = threadIdx.y;
    const int tc = threadIdx.x;
    const int gr = base + tr;
    const int gc = base + tc;
    const bool valid = (gr < n) && (gc < n);

    sDist[tr][tc] = valid ? dist[(size_t)gc * n + gr] : INF;
    sPath[tr][tc] = valid ? path[(size_t)gc * n + gr] : 0u;
    __syncthreads();

#pragma unroll
    for (int k = 0; k < BLOCK_SIZE; ++k) {
        const unsigned int newDist = sDist[tr][k] + sDist[k][tc];
        if (newDist < sDist[tr][tc]) {
            sDist[tr][tc] = newDist;
            sPath[tr][tc] = base + k;
        }
        __syncthreads();
    }

    if (valid) {
        dist[(size_t)gc * n + gr] = sDist[tr][tc];
        path[(size_t)gc * n + gr] = sPath[tr][tc];
    }
}

__global__ void fwPhase2(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
                          const int n, const int round) {
    const int blk = blockIdx.x;
    if (blk == round) return;
    const bool isRow = (blockIdx.y == 0);

    const int tr = threadIdx.y;
    const int tc = threadIdx.x;
    const int base = round * BLOCK_SIZE;

    __shared__ unsigned int sDiag[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int sBlk[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int sBlkPath[BLOCK_SIZE][BLOCK_SIZE];

    const int diagR = base + tr;
    const int diagC = base + tc;
    sDiag[tr][tc] = (diagR < n && diagC < n) ? dist[(size_t)diagC * n + diagR] : INF;

    int p, q;
    if (isRow) {
        p = base + tr;
        q = blk * BLOCK_SIZE + tc;
    } else {
        p = blk * BLOCK_SIZE + tr;
        q = base + tc;
    }
    const bool valid = (p < n) && (q < n);
    sBlk[tr][tc] = valid ? dist[(size_t)q * n + p] : INF;
    sBlkPath[tr][tc] = valid ? path[(size_t)q * n + p] : 0u;
    __syncthreads();

#pragma unroll
    for (int k = 0; k < BLOCK_SIZE; ++k) {
        const unsigned int a = isRow ? sDiag[tr][k] : sBlk[tr][k];
        const unsigned int b = isRow ? sBlk[k][tc] : sDiag[k][tc];
        const unsigned int newDist = a + b;
        if (newDist < sBlk[tr][tc]) {
            sBlk[tr][tc] = newDist;
            sBlkPath[tr][tc] = base + k;
        }
        __syncthreads();
    }

    if (valid) {
        dist[(size_t)q * n + p] = sBlk[tr][tc];
        path[(size_t)q * n + p] = sBlkPath[tr][tc];
    }
}

__global__ void fwPhase3(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
                          const int n, const int round) {
    const int bi = blockIdx.y;
    const int bj = blockIdx.x;
    if (bi == round || bj == round) return;

    const int tr = threadIdx.y;
    const int tc = threadIdx.x;
    const int base = round * BLOCK_SIZE;

    __shared__ unsigned int sRow[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int sCol[BLOCK_SIZE][BLOCK_SIZE];

    const int rowR = base + tr;
    const int rowC = bj * BLOCK_SIZE + tc;
    sRow[tr][tc] = (rowR < n && rowC < n) ? dist[(size_t)rowC * n + rowR] : INF;

    const int colR = bi * BLOCK_SIZE + tr;
    const int colC = base + tc;
    sCol[tr][tc] = (colR < n && colC < n) ? dist[(size_t)colC * n + colR] : INF;
    __syncthreads();

    const int p = bi * BLOCK_SIZE + tr;
    const int q = bj * BLOCK_SIZE + tc;
    if (p >= n || q >= n) return;

    unsigned int myDist = dist[(size_t)q * n + p];
    unsigned int myPath = path[(size_t)q * n + p];

#pragma unroll
    for (int k = 0; k < BLOCK_SIZE; ++k) {
        const unsigned int newDist = sCol[tr][k] + sRow[k][tc];
        if (newDist < myDist) {
            myDist = newDist;
            myPath = base + k;
        }
    }

    dist[(size_t)q * n + p] = myDist;
    path[(size_t)q * n + p] = myPath;
}

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes) {
    const int n = static_cast<int>(numNodes);
    const size_t bytes = numNodes * numNodes * sizeof(unsigned int);
    const int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    unsigned int* dDist = nullptr;
    unsigned int* dPath = nullptr;
    CUDA_CHECK(cudaMalloc(&dDist, bytes));
    CUDA_CHECK(cudaMalloc(&dPath, bytes));

    CUDA_CHECK(cudaMemcpy(dDist, dist.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dPath, path.data(), bytes, cudaMemcpyHostToDevice));

    const dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);

    for (int round = 0; round < numBlocks; ++round) {
        fwPhase1<<<dim3(1, 1), blockDim>>>(dDist, dPath, n, round);
        fwPhase2<<<dim3(numBlocks, 2), blockDim>>>(dDist, dPath, n, round);
        if (numBlocks > 1) {
            fwPhase3<<<dim3(numBlocks, numBlocks), blockDim>>>(dDist, dPath, n, round);
        }
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(dist.data(), dDist, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(path.data(), dPath, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(dDist));
    CUDA_CHECK(cudaFree(dPath));
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
