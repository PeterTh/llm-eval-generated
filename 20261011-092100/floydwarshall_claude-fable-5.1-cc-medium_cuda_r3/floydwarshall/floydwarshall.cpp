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

#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        cudaError_t err__ = (call);                                                      \
        if (err__ != cudaSuccess) {                                                      \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__,   \
                    cudaGetErrorString(err__));                                          \
            exit(1);                                                                     \
        }                                                                                \
    } while (0)

// ---------------------------------------------------------------------------
// GPU blocked Floyd-Warshall
//
// Semantics: the sequential algorithm updates dist[i][j] (and sets path[i][j] = k)
// whenever dist[i][k] + dist[k][j] is strictly smaller.  The resulting path entry
// equals the *last* k at which dist[i][j] changed, which is the smallest k such
// that a shortest path exists using only intermediates <= k, i.e. the minimum
// over all shortest i->j paths of the largest intermediate vertex.
//
// To reproduce this exactly in a blocked (tiled) formulation, every entry carries
// the pair (distance, maxIntermediate+1) packed into one 64-bit integer:
//     key = (dist << 32) | (maxIntermediate + 1)        (0 == no intermediate)
// Relaxation through k computes (d_ik + d_kj, max(m_ik, m_kj, k+1)) and keeps the
// lexicographically smaller pair, which is a plain 64-bit unsigned compare.
// This is a monotone path-weight, so the blocked algorithm yields the same
// result as the sequential one for both the distance and the path matrix.
// ---------------------------------------------------------------------------

typedef unsigned long long u64;

constexpr int TILE = 64;               // tile edge (block size of the blocked algorithm)
constexpr int TDIM = 16;               // thread block is TDIM x TDIM
constexpr int EPT = TILE / TDIM;       // elements per thread per dimension
constexpr unsigned int PAD_DIST = 0x3FFFFFFFu;  // "infinite" distance for padded nodes
constexpr size_t TILE_BYTES = (size_t)TILE * TILE * sizeof(u64);

__device__ __forceinline__ void relax(u64& cur, const u64 a, const u64 b, const unsigned kp1) {
    const unsigned d = (unsigned)(a >> 32) + (unsigned)(b >> 32);
    const unsigned m = max(max((unsigned)a, (unsigned)b), kp1);
    const u64 cand = ((u64)d << 32) | m;
    cur = (cand < cur) ? cand : cur;
}

__global__ void packKernel(const unsigned int* __restrict__ dist, u64* __restrict__ A,
                           const int n, const int N) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= N || j >= N) return;
    unsigned d;
    if (i < n && j < n) {
        d = dist[(size_t)i * n + j];
    } else {
        d = (i == j) ? 0u : PAD_DIST;
    }
    A[(size_t)i * N + j] = ((u64)d << 32);
}

__global__ void unpackKernel(const u64* __restrict__ A, unsigned int* __restrict__ dist,
                             unsigned int* __restrict__ path, const int n, const int N) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= n || j >= n) return;
    const u64 v = A[(size_t)i * N + j];
    const unsigned m = (unsigned)v;
    dist[(size_t)i * n + j] = (unsigned)(v >> 32);
    path[(size_t)i * n + j] = (m == 0u) ? (unsigned)i : (m - 1u);
}

// Phase 1: the diagonal tile (round, round), dependent on itself only.
__global__ void __launch_bounds__(TDIM * TDIM)
phase1Kernel(u64* __restrict__ A, const int N, const int round) {
    extern __shared__ u64 smem[];
    u64 (*T)[TILE] = reinterpret_cast<u64 (*)[TILE]>(smem);

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int base = round * TILE;
    u64 C[EPT][EPT];

#pragma unroll
    for (int r = 0; r < EPT; ++r)
#pragma unroll
        for (int c = 0; c < EPT; ++c) {
            const int row = ty + r * TDIM, col = tx + c * TDIM;
            C[r][c] = A[(size_t)(base + row) * N + base + col];
            T[row][col] = C[r][c];
        }
    __syncthreads();

    for (int k = 0; k < TILE; ++k) {
        const unsigned kp1 = (unsigned)(base + k + 1);
        u64 a[EPT], b[EPT];
#pragma unroll
        for (int r = 0; r < EPT; ++r) a[r] = T[ty + r * TDIM][k];
#pragma unroll
        for (int c = 0; c < EPT; ++c) b[c] = T[k][tx + c * TDIM];
#pragma unroll
        for (int r = 0; r < EPT; ++r)
#pragma unroll
            for (int c = 0; c < EPT; ++c) {
                const u64 old = C[r][c];
                relax(C[r][c], a[r], b[c], kp1);
                if (C[r][c] != old) T[ty + r * TDIM][tx + c * TDIM] = C[r][c];
            }
        __syncthreads();
    }

#pragma unroll
    for (int r = 0; r < EPT; ++r)
#pragma unroll
        for (int c = 0; c < EPT; ++c)
            A[(size_t)(base + ty + r * TDIM) * N + base + tx + c * TDIM] = C[r][c];
}

// Phase 2: tiles sharing a row block (ROW == true) or column block with the diagonal tile.
template <bool ROW>
__device__ __forceinline__ void phase2Body(u64* __restrict__ A, const int N, const int round,
                                           const int t, u64* smem) {
    u64 (*D)[TILE] = reinterpret_cast<u64 (*)[TILE]>(smem);                 // diagonal tile
    u64 (*T)[TILE] = reinterpret_cast<u64 (*)[TILE]>(smem + TILE * TILE);   // target tile

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int kbase = round * TILE;
    const int rowBase = ROW ? kbase : t * TILE;
    const int colBase = ROW ? t * TILE : kbase;
    u64 C[EPT][EPT];

#pragma unroll
    for (int r = 0; r < EPT; ++r)
#pragma unroll
        for (int c = 0; c < EPT; ++c) {
            const int row = ty + r * TDIM, col = tx + c * TDIM;
            D[row][col] = A[(size_t)(kbase + row) * N + kbase + col];
            C[r][c] = A[(size_t)(rowBase + row) * N + colBase + col];
            T[row][col] = C[r][c];
        }
    __syncthreads();

    for (int k = 0; k < TILE; ++k) {
        const unsigned kp1 = (unsigned)(kbase + k + 1);
        u64 a[EPT], b[EPT];
        if (ROW) {
#pragma unroll
            for (int r = 0; r < EPT; ++r) a[r] = D[ty + r * TDIM][k];
#pragma unroll
            for (int c = 0; c < EPT; ++c) b[c] = T[k][tx + c * TDIM];
        } else {
#pragma unroll
            for (int r = 0; r < EPT; ++r) a[r] = T[ty + r * TDIM][k];
#pragma unroll
            for (int c = 0; c < EPT; ++c) b[c] = D[k][tx + c * TDIM];
        }
#pragma unroll
        for (int r = 0; r < EPT; ++r)
#pragma unroll
            for (int c = 0; c < EPT; ++c) {
                const u64 old = C[r][c];
                relax(C[r][c], a[r], b[c], kp1);
                if (C[r][c] != old) T[ty + r * TDIM][tx + c * TDIM] = C[r][c];
            }
        __syncthreads();
    }

#pragma unroll
    for (int r = 0; r < EPT; ++r)
#pragma unroll
        for (int c = 0; c < EPT; ++c)
            A[(size_t)(rowBase + ty + r * TDIM) * N + colBase + tx + c * TDIM] = C[r][c];
}

__global__ void __launch_bounds__(TDIM * TDIM)
phase2Kernel(u64* __restrict__ A, const int N, const int round) {
    extern __shared__ u64 smem[];
    const int t = blockIdx.x;
    if (t == round) return;
    if (blockIdx.y == 0)
        phase2Body<true>(A, N, round, t, smem);
    else
        phase2Body<false>(A, N, round, t, smem);
}

// Phase 3: all remaining tiles, using the (already final for this round) row/column tiles.
__global__ void __launch_bounds__(TDIM * TDIM)
phase3Kernel(u64* __restrict__ A, const int N, const int round) {
    extern __shared__ u64 smem[];
    u64 (*L)[TILE] = reinterpret_cast<u64 (*)[TILE]>(smem);                 // tile (bi, round)
    u64 (*R)[TILE] = reinterpret_cast<u64 (*)[TILE]>(smem + TILE * TILE);   // tile (round, bj)

    const int bj = blockIdx.x, bi = blockIdx.y;
    if (bi == round || bj == round) return;

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int kbase = round * TILE;
    const int rowBase = bi * TILE;
    const int colBase = bj * TILE;
    u64 C[EPT][EPT];

#pragma unroll
    for (int r = 0; r < EPT; ++r)
#pragma unroll
        for (int c = 0; c < EPT; ++c) {
            const int row = ty + r * TDIM, col = tx + c * TDIM;
            L[row][col] = A[(size_t)(rowBase + row) * N + kbase + col];
            R[row][col] = A[(size_t)(kbase + row) * N + colBase + col];
            C[r][c] = A[(size_t)(rowBase + row) * N + colBase + col];
        }
    __syncthreads();

#pragma unroll 4
    for (int k = 0; k < TILE; ++k) {
        const unsigned kp1 = (unsigned)(kbase + k + 1);
        u64 a[EPT], b[EPT];
#pragma unroll
        for (int r = 0; r < EPT; ++r) a[r] = L[ty + r * TDIM][k];
#pragma unroll
        for (int c = 0; c < EPT; ++c) b[c] = R[k][tx + c * TDIM];
#pragma unroll
        for (int r = 0; r < EPT; ++r)
#pragma unroll
            for (int c = 0; c < EPT; ++c) relax(C[r][c], a[r], b[c], kp1);
    }

#pragma unroll
    for (int r = 0; r < EPT; ++r)
#pragma unroll
        for (int c = 0; c < EPT; ++c)
            A[(size_t)(rowBase + ty + r * TDIM) * N + colBase + tx + c * TDIM] = C[r][c];
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
    if (numNodes == 0) return;

    const int n = (int)numNodes;
    const int numTiles = (n + TILE - 1) / TILE;
    const int N = numTiles * TILE;   // padded dimension

    unsigned int* dDist = nullptr;
    unsigned int* dPath = nullptr;
    u64* dA = nullptr;
    CUDA_CHECK(cudaMalloc(&dDist, (size_t)n * n * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&dPath, (size_t)n * n * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&dA, (size_t)N * N * sizeof(u64)));

    CUDA_CHECK(cudaMemcpy(dDist, dist.data(), (size_t)n * n * sizeof(unsigned int),
                          cudaMemcpyHostToDevice));

    const dim3 packBlock(32, 8);
    const dim3 packGrid((N + packBlock.x - 1) / packBlock.x, (N + packBlock.y - 1) / packBlock.y);
    packKernel<<<packGrid, packBlock>>>(dDist, dA, n, N);
    CUDA_CHECK(cudaGetLastError());

    const dim3 block(TDIM, TDIM);
    for (int round = 0; round < numTiles; ++round) {
        phase1Kernel<<<1, block, TILE_BYTES>>>(dA, N, round);
        if (numTiles > 1) {
            phase2Kernel<<<dim3(numTiles, 2), block, 2 * TILE_BYTES>>>(dA, N, round);
            phase3Kernel<<<dim3(numTiles, numTiles), block, 2 * TILE_BYTES>>>(dA, N, round);
        }
    }
    CUDA_CHECK(cudaGetLastError());

    const dim3 unpackGrid((n + packBlock.x - 1) / packBlock.x, (n + packBlock.y - 1) / packBlock.y);
    unpackKernel<<<unpackGrid, packBlock>>>(dA, dDist, dPath, n, N);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(dist.data(), dDist, (size_t)n * n * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(path.data(), dPath, (size_t)n * n * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dPath));
    CUDA_CHECK(cudaFree(dDist));
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
    
    // Initialize CUDA context and configure kernels (outside the timed region)
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaFree(0));
    CUDA_CHECK(cudaFuncSetAttribute(phase1Kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    (int)TILE_BYTES));
    CUDA_CHECK(cudaFuncSetAttribute(phase2Kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    (int)(2 * TILE_BYTES)));
    CUDA_CHECK(cudaFuncSetAttribute(phase3Kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    (int)(2 * TILE_BYTES)));

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
