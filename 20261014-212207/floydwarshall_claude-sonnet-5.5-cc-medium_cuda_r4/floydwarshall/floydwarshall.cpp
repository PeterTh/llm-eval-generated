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

#define CUDA_CHECK(call)                                                                  \
    do {                                                                                  \
        cudaError_t err_ = (call);                                                        \
        if (err_ != cudaSuccess) {                                                        \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_),         \
                    __FILE__, __LINE__);                                                  \
            exit(1);                                                                      \
        }                                                                                 \
    } while (0)

constexpr int TILE = 32;
constexpr int BY = 8;              // threads in y; each thread handles TILE/BY rows
constexpr int ROWS = TILE / BY;
constexpr unsigned int PAD = INF;  // out-of-range sentinel; PAD + PAD does not overflow

// Matrices are row-major: element (i, j) lives at i * n + j (== idx2(j, i, n)).

// Phase 1: update the diagonal tile (kb, kb)
__global__ void fwDiagonal(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
                           const int n, const int kb) {
    __shared__ unsigned int t[TILE][TILE + 1];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int base = kb * TILE;
    unsigned int p[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        const int i = base + ty + r * BY, j = base + tx;
        const bool ok = i < n && j < n;
        t[ty + r * BY][tx] = ok ? dist[(size_t)i * n + j] : PAD;
        p[r] = ok ? path[(size_t)i * n + j] : 0;
    }
    __syncthreads();
    for (int k = 0; k < TILE; ++k) {
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const unsigned int nd = t[ty + r * BY][k] + t[k][tx];
            if (nd < t[ty + r * BY][tx]) {
                t[ty + r * BY][tx] = nd;
                p[r] = base + k;
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        const int i = base + ty + r * BY, j = base + tx;
        if (i < n && j < n) {
            dist[(size_t)i * n + j] = t[ty + r * BY][tx];
            path[(size_t)i * n + j] = p[r];
        }
    }
}

// Phase 2: update row tiles (kb, j) [blockIdx.y == 0] and column tiles (i, kb) [blockIdx.y == 1]
__global__ void fwRowCol(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
                         const int n, const int kb) {
    __shared__ unsigned int d[TILE][TILE + 1];  // diagonal tile
    __shared__ unsigned int t[TILE][TILE + 1];  // own tile
    int b = blockIdx.x;
    if (b >= kb) ++b;  // skip the diagonal tile
    const bool isRow = blockIdx.y == 0;
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int base = kb * TILE;
    const int oi = isRow ? base : b * TILE;
    const int oj = isRow ? b * TILE : base;
    unsigned int p[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        const int y = ty + r * BY;
        const int di = base + y, dj = base + tx;
        d[y][tx] = (di < n && dj < n) ? dist[(size_t)di * n + dj] : PAD;
        const int i = oi + y, j = oj + tx;
        const bool ok = i < n && j < n;
        t[y][tx] = ok ? dist[(size_t)i * n + j] : PAD;
        p[r] = ok ? path[(size_t)i * n + j] : 0;
    }
    __syncthreads();
    if (isRow) {
        for (int k = 0; k < TILE; ++k) {
#pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                const int y = ty + r * BY;
                const unsigned int nd = d[y][k] + t[k][tx];
                if (nd < t[y][tx]) {
                    t[y][tx] = nd;
                    p[r] = base + k;
                }
            }
            __syncthreads();
        }
    } else {
        for (int k = 0; k < TILE; ++k) {
#pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                const int y = ty + r * BY;
                const unsigned int nd = t[y][k] + d[k][tx];
                if (nd < t[y][tx]) {
                    t[y][tx] = nd;
                    p[r] = base + k;
                }
            }
            __syncthreads();
        }
    }
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        const int i = oi + ty + r * BY, j = oj + tx;
        if (i < n && j < n) {
            dist[(size_t)i * n + j] = t[ty + r * BY][tx];
            path[(size_t)i * n + j] = p[r];
        }
    }
}

// Phase 3: update all remaining tiles (i, j), i != kb, j != kb
__global__ void fwRest(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
                       const int n, const int kb) {
    __shared__ unsigned int colT[TILE][TILE + 1];  // tile (bi, kb)
    __shared__ unsigned int rowT[TILE][TILE];      // tile (kb, bj)
    int bi = blockIdx.y, bj = blockIdx.x;
    if (bi >= kb) ++bi;
    if (bj >= kb) ++bj;
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int base = kb * TILE;
    const int i0 = bi * TILE, j0 = bj * TILE;
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        const int y = ty + r * BY;
        const int ci = i0 + y, ck = base + tx;
        colT[y][tx] = (ci < n && ck < n) ? dist[(size_t)ci * n + ck] : PAD;
        const int rk = base + y, rj = j0 + tx;
        rowT[y][tx] = (rk < n && rj < n) ? dist[(size_t)rk * n + rj] : PAD;
    }
    unsigned int v[ROWS], p[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        const int i = i0 + ty + r * BY, j = j0 + tx;
        const bool ok = i < n && j < n;
        v[r] = ok ? dist[(size_t)i * n + j] : 0;
        p[r] = ok ? path[(size_t)i * n + j] : 0;
    }
    __syncthreads();
#pragma unroll 8
    for (int k = 0; k < TILE; ++k) {
        const unsigned int rv = rowT[k][tx];
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const unsigned int nd = colT[ty + r * BY][k] + rv;
            if (nd < v[r]) {
                v[r] = nd;
                p[r] = base + k;
            }
        }
    }
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        const int i = i0 + ty + r * BY, j = j0 + tx;
        if (i < n && j < n) {
            dist[(size_t)i * n + j] = v[r];
            path[(size_t)i * n + j] = p[r];
        }
    }
}

// Blocked Floyd-Warshall on the GPU. Every cell still sees the intermediate nodes k in
// ascending order, so distances are identical to the classic algorithm.
void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes) {
    if (numNodes == 0) return;
    const int n = static_cast<int>(numNodes);
    const size_t bytes = numNodes * numNodes * sizeof(unsigned int);
    unsigned int *dDist = nullptr, *dPath = nullptr;
    CUDA_CHECK(cudaMalloc(&dDist, bytes));
    CUDA_CHECK(cudaMalloc(&dPath, bytes));
    CUDA_CHECK(cudaMemcpy(dDist, dist.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dPath, path.data(), bytes, cudaMemcpyHostToDevice));

    const int nb = (n + TILE - 1) / TILE;
    const dim3 block(TILE, BY);
    for (int kb = 0; kb < nb; ++kb) {
        fwDiagonal<<<1, block>>>(dDist, dPath, n, kb);
        if (nb > 1) {
            fwRowCol<<<dim3(nb - 1, 2), block>>>(dDist, dPath, n, kb);
            fwRest<<<dim3(nb - 1, nb - 1), block>>>(dDist, dPath, n, kb);
        }
    }
    CUDA_CHECK(cudaGetLastError());
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
    
    // Create the CUDA context up front so it is not part of the timed region
    CUDA_CHECK(cudaFree(nullptr));

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
