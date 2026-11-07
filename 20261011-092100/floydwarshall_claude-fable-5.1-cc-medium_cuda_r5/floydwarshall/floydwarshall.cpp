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

// ---------------------------------------------------------------------------
// CUDA blocked Floyd-Warshall
//
// The matrix is partitioned into TILE x TILE tiles. For every tile index kb
// along the diagonal, three dependent phases are executed:
//   phase 1: relax the diagonal tile (kb,kb) through all k in the tile
//   phase 2: relax tiles in row kb and column kb using the diagonal tile
//   phase 3: relax all remaining tiles using the row/column tiles of kb
// This is algebraically equivalent to the classic triple loop (same final
// distance matrix), but exposes O(n^2) parallelism per k-step and reuses data
// through shared memory. The matrix is zero-copied into a padded device
// buffer whose extra rows/columns are INF (and 0 on the diagonal), so they
// never participate in a shortest path.
//
// Memory layout matches the original: dist[idx2(j, i, n)] == dist[i*n + j]
// is the distance from node i to node j (row-major).
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                            \
    do {                                                                            \
        cudaError_t err__ = (call);                                                 \
        if (err__ != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__,       \
                    __LINE__, cudaGetErrorString(err__));                           \
            exit(EXIT_FAILURE);                                                     \
        }                                                                           \
    } while (0)

constexpr int TILE = 64;                       // tile edge length
constexpr int TY = 8;                          // thread rows per block
constexpr int ROWS_PER_THREAD = TILE / TY;     // 8 matrix rows per thread
constexpr int SMEM_PAD = 1;                    // avoid bank conflicts on column reads

// Phase 1: diagonal tile. Fully in-tile Floyd-Warshall in shared memory.
__global__ void __launch_bounds__(TILE * TY)
fwPhase1(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
         const int np, const int kb) {
    __shared__ unsigned int s[TILE][TILE + SMEM_PAD];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int base = kb * TILE;

    unsigned int p[ROWS_PER_THREAD];
    bool changed[ROWS_PER_THREAD];

#pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; ++r) {
        const int i = ty + r * TY;
        s[i][tx] = dist[(size_t)(base + i) * np + base + tx];
        changed[r] = false;
    }
    __syncthreads();

    for (int kk = 0; kk < TILE; ++kk) {
        const unsigned int dkj = s[kk][tx];
#pragma unroll
        for (int r = 0; r < ROWS_PER_THREAD; ++r) {
            const int i = ty + r * TY;
            const unsigned int nd = s[i][kk] + dkj;
            if (nd < s[i][tx]) {
                s[i][tx] = nd;
                p[r] = base + kk;
                changed[r] = true;
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; ++r) {
        const int i = ty + r * TY;
        const size_t g = (size_t)(base + i) * np + base + tx;
        dist[g] = s[i][tx];
        if (changed[r]) path[g] = p[r];
    }
}

// Phase 2: tiles sharing a row (blockIdx.y == 0) or a column (blockIdx.y == 1)
// with the diagonal tile kb.
__global__ void __launch_bounds__(TILE * TY)
fwPhase2(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
         const int np, const int kb) {
    __shared__ unsigned int diag[TILE][TILE + SMEM_PAD];
    __shared__ unsigned int cur[TILE][TILE + SMEM_PAD];

    const int bx = blockIdx.x;
    if (bx == kb) return;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const bool rowTile = (blockIdx.y == 0);
    const int kbase = kb * TILE;
    const int rowBase = rowTile ? kbase : bx * TILE;
    const int colBase = rowTile ? bx * TILE : kbase;

    unsigned int p[ROWS_PER_THREAD];
    bool changed[ROWS_PER_THREAD];

#pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; ++r) {
        const int i = ty + r * TY;
        diag[i][tx] = dist[(size_t)(kbase + i) * np + kbase + tx];
        cur[i][tx] = dist[(size_t)(rowBase + i) * np + colBase + tx];
        changed[r] = false;
    }
    __syncthreads();

    if (rowTile) {
        // cur[i][j] = min(cur[i][j], diag[i][kk] + cur[kk][j])
        for (int kk = 0; kk < TILE; ++kk) {
            const unsigned int dkj = cur[kk][tx];
#pragma unroll
            for (int r = 0; r < ROWS_PER_THREAD; ++r) {
                const int i = ty + r * TY;
                const unsigned int nd = diag[i][kk] + dkj;
                if (nd < cur[i][tx]) {
                    cur[i][tx] = nd;
                    p[r] = kbase + kk;
                    changed[r] = true;
                }
            }
            __syncthreads();
        }
    } else {
        // cur[i][j] = min(cur[i][j], cur[i][kk] + diag[kk][j])
        for (int kk = 0; kk < TILE; ++kk) {
            const unsigned int dkj = diag[kk][tx];
#pragma unroll
            for (int r = 0; r < ROWS_PER_THREAD; ++r) {
                const int i = ty + r * TY;
                const unsigned int nd = cur[i][kk] + dkj;
                if (nd < cur[i][tx]) {
                    cur[i][tx] = nd;
                    p[r] = kbase + kk;
                    changed[r] = true;
                }
            }
            __syncthreads();
        }
    }

#pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; ++r) {
        const int i = ty + r * TY;
        const size_t g = (size_t)(rowBase + i) * np + colBase + tx;
        dist[g] = cur[i][tx];
        if (changed[r]) path[g] = p[r];
    }
}

// Phase 3: all remaining tiles. Row tile (kb, bx) and column tile (by, kb)
// are final for this kb, so each output element is independent and kept in
// registers; only the two input tiles go through shared memory.
__global__ void __launch_bounds__(TILE * TY)
fwPhase3(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
         const int np, const int kb) {
    __shared__ unsigned int rowT[TILE][TILE + SMEM_PAD];   // dist[k][j]
    __shared__ unsigned int colT[TILE][TILE + SMEM_PAD];   // dist[i][k]

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    if (bx == kb || by == kb) return;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int kbase = kb * TILE;
    const int rowBase = by * TILE;
    const int colBase = bx * TILE;

    unsigned int d[ROWS_PER_THREAD];
    unsigned int p[ROWS_PER_THREAD];
    bool changed[ROWS_PER_THREAD];

#pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; ++r) {
        const int i = ty + r * TY;
        rowT[i][tx] = dist[(size_t)(kbase + i) * np + colBase + tx];
        colT[i][tx] = dist[(size_t)(rowBase + i) * np + kbase + tx];
        d[r] = dist[(size_t)(rowBase + i) * np + colBase + tx];
        changed[r] = false;
    }
    __syncthreads();

#pragma unroll 16
    for (int kk = 0; kk < TILE; ++kk) {
        const unsigned int dkj = rowT[kk][tx];
#pragma unroll
        for (int r = 0; r < ROWS_PER_THREAD; ++r) {
            const unsigned int nd = colT[ty + r * TY][kk] + dkj;
            if (nd < d[r]) {
                d[r] = nd;
                p[r] = kbase + kk;
                changed[r] = true;
            }
        }
    }

#pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; ++r) {
        const int i = ty + r * TY;
        const size_t g = (size_t)(rowBase + i) * np + colBase + tx;
        dist[g] = d[r];
        if (changed[r]) path[g] = p[r];
    }
}

// Fill the padding region of the device distance matrix: INF everywhere
// outside the real n x n block, except 0 on the diagonal.
__global__ void fwInitPadding(unsigned int* __restrict__ dist, const int n, const int np) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= np || j >= np) return;
    if (i < n && j < n) return;
    dist[(size_t)i * np + j] = (i == j) ? 0u : INF;
}

void floydWarshall(std::vector<unsigned int>& dist, 
                   std::vector<unsigned int>& path, 
                   const size_t numNodes) {
    if (numNodes == 0) return;

    const int n = static_cast<int>(numNodes);
    const int nb = (n + TILE - 1) / TILE;   // number of tiles per dimension
    const int np = nb * TILE;               // padded dimension
    const size_t bytes = (size_t)np * np * sizeof(unsigned int);

    unsigned int* dDist = nullptr;
    unsigned int* dPath = nullptr;
    CUDA_CHECK(cudaMalloc(&dDist, bytes));
    CUDA_CHECK(cudaMalloc(&dPath, bytes));

    // Upload the n x n matrices into the padded np x np device buffers.
    CUDA_CHECK(cudaMemcpy2D(dDist, (size_t)np * sizeof(unsigned int),
                            dist.data(), (size_t)n * sizeof(unsigned int),
                            (size_t)n * sizeof(unsigned int), (size_t)n,
                            cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy2D(dPath, (size_t)np * sizeof(unsigned int),
                            path.data(), (size_t)n * sizeof(unsigned int),
                            (size_t)n * sizeof(unsigned int), (size_t)n,
                            cudaMemcpyHostToDevice));
    if (np != n) {
        dim3 pb(32, 8);
        dim3 pg((np + pb.x - 1) / pb.x, (np + pb.y - 1) / pb.y);
        fwInitPadding<<<pg, pb>>>(dDist, n, np);
        CUDA_CHECK(cudaGetLastError());
    }

    const dim3 block(TILE, TY);
    const dim3 grid2(nb, 2);
    const dim3 grid3(nb, nb);

    for (int kb = 0; kb < nb; ++kb) {
        fwPhase1<<<1, block>>>(dDist, dPath, np, kb);
        if (nb > 1) {
            fwPhase2<<<grid2, block>>>(dDist, dPath, np, kb);
            fwPhase3<<<grid3, block>>>(dDist, dPath, np, kb);
        }
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy2D(dist.data(), (size_t)n * sizeof(unsigned int),
                            dDist, (size_t)np * sizeof(unsigned int),
                            (size_t)n * sizeof(unsigned int), (size_t)n,
                            cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy2D(path.data(), (size_t)n * sizeof(unsigned int),
                            dPath, (size_t)np * sizeof(unsigned int),
                            (size_t)n * sizeof(unsigned int), (size_t)n,
                            cudaMemcpyDeviceToHost));

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
    
    // Initialize the CUDA device/context up front so that one-time driver
    // setup is not attributed to the computation.
    printf("Initializing CUDA device...\n");
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaFree(0));
    {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        printf("CUDA device: %s\n", prop.name);
    }
    
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
