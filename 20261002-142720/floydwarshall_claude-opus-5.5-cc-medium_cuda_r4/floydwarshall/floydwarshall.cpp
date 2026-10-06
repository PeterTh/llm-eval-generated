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
// The matrix is padded to a multiple of TILE. Padding nodes are unreachable
// (INF) so they never produce an improvement for real nodes (INF + x never
// overflows 32 bits and is always larger than any real distance).
// For each block-round K:
//   phase 1: the diagonal tile (K,K) is processed with classic FW
//   phase 2: the tiles in block-row K and block-column K
//   phase 3: all remaining tiles (min-plus product of row/column panels)
// Distances are identical to the sequential algorithm; path[i][j] holds an
// intermediate node k of a shortest i->j path (dist[i][j] == dist[i][k] + dist[k][j]).
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), \
                    __FILE__, __LINE__);                                          \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

constexpr int TILE = 64;          // tile edge (block of k)
constexpr int THR = 16;           // phase 3: 16x16 threads, 4x4 elements each
constexpr int EPT = TILE / THR;   // elements per thread per dimension (phase 3)
constexpr int THR12 = 32;         // phases 1/2: 32x32 threads, 2x2 elements each
constexpr int EPT12 = TILE / THR12;

// Phase 1: diagonal tile
__global__ void __launch_bounds__(THR12 * THR12)
fwPhase1(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
         const int n, const int kb) {
    __shared__ unsigned int sD[TILE][TILE + 1];
    __shared__ unsigned int sP[TILE][TILE + 1];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int base = kb * TILE;

#pragma unroll
    for (int r = 0; r < EPT12; ++r) {
#pragma unroll
        for (int c = 0; c < EPT12; ++c) {
            const int li = ty + r * THR12, lj = tx + c * THR12;
            const size_t g = (size_t)(base + li) * n + base + lj;
            sD[li][lj] = dist[g];
            sP[li][lj] = path[g];
        }
    }
    __syncthreads();

    for (int k = 0; k < TILE; ++k) {
#pragma unroll
        for (int r = 0; r < EPT12; ++r) {
#pragma unroll
            for (int c = 0; c < EPT12; ++c) {
                const int li = ty + r * THR12, lj = tx + c * THR12;
                const unsigned int nd = sD[li][k] + sD[k][lj];
                if (nd < sD[li][lj]) {
                    sD[li][lj] = nd;
                    sP[li][lj] = base + k;
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int r = 0; r < EPT12; ++r) {
#pragma unroll
        for (int c = 0; c < EPT12; ++c) {
            const int li = ty + r * THR12, lj = tx + c * THR12;
            const size_t g = (size_t)(base + li) * n + base + lj;
            dist[g] = sD[li][lj];
            path[g] = sP[li][lj];
        }
    }
}

// Phase 2: tiles in block-row kb (blockIdx.y == 0) and block-column kb (blockIdx.y == 1)
__global__ void __launch_bounds__(THR12 * THR12)
fwPhase2(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
         const int n, const int kb) {
    if ((int)blockIdx.x == kb) return;
    __shared__ unsigned int sK[TILE][TILE + 1];  // diagonal tile (final)
    __shared__ unsigned int sD[TILE][TILE + 1];  // own tile
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int base = kb * TILE;
    const bool isRow = (blockIdx.y == 0);
    const int rowBase = isRow ? base : blockIdx.x * TILE;
    const int colBase = isRow ? blockIdx.x * TILE : base;

    unsigned int p[EPT12][EPT12];
#pragma unroll
    for (int r = 0; r < EPT12; ++r) {
#pragma unroll
        for (int c = 0; c < EPT12; ++c) {
            const int li = ty + r * THR12, lj = tx + c * THR12;
            sK[li][lj] = dist[(size_t)(base + li) * n + base + lj];
            const size_t g = (size_t)(rowBase + li) * n + colBase + lj;
            sD[li][lj] = dist[g];
            p[r][c] = path[g];
        }
    }
    __syncthreads();

    if (isRow) {
        // own[i][j] = min(own[i][j], diag[i][k] + own[k][j])
        for (int k = 0; k < TILE; ++k) {
#pragma unroll
            for (int r = 0; r < EPT12; ++r) {
#pragma unroll
                for (int c = 0; c < EPT12; ++c) {
                    const int li = ty + r * THR12, lj = tx + c * THR12;
                    const unsigned int nd = sK[li][k] + sD[k][lj];
                    if (nd < sD[li][lj]) {
                        sD[li][lj] = nd;
                        p[r][c] = base + k;
                    }
                }
            }
            __syncthreads();
        }
    } else {
        // own[i][j] = min(own[i][j], own[i][k] + diag[k][j])
        for (int k = 0; k < TILE; ++k) {
#pragma unroll
            for (int r = 0; r < EPT12; ++r) {
#pragma unroll
                for (int c = 0; c < EPT12; ++c) {
                    const int li = ty + r * THR12, lj = tx + c * THR12;
                    const unsigned int nd = sD[li][k] + sK[k][lj];
                    if (nd < sD[li][lj]) {
                        sD[li][lj] = nd;
                        p[r][c] = base + k;
                    }
                }
            }
            __syncthreads();
        }
    }

#pragma unroll
    for (int r = 0; r < EPT12; ++r) {
#pragma unroll
        for (int c = 0; c < EPT12; ++c) {
            const int li = ty + r * THR12, lj = tx + c * THR12;
            const size_t g = (size_t)(rowBase + li) * n + colBase + lj;
            dist[g] = sD[li][lj];
            path[g] = p[r][c];
        }
    }
}

// Phase 3: all remaining tiles; independent min-plus updates kept in registers
__global__ void __launch_bounds__(THR * THR)
fwPhase3(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
         const int n, const int kb) {
    const int bi = blockIdx.y, bj = blockIdx.x;
    if (bi == kb || bj == kb) return;
    __shared__ unsigned int sA[TILE][TILE + 1];  // column panel tile (bi, kb)
    __shared__ unsigned int sB[TILE][TILE];      // row panel tile (kb, bj)
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int base = kb * TILE;
    const int rowBase = bi * TILE, colBase = bj * TILE;

    unsigned int d[EPT][EPT], p[EPT][EPT];
#pragma unroll
    for (int r = 0; r < EPT; ++r) {
#pragma unroll
        for (int c = 0; c < EPT; ++c) {
            const int li = ty + r * THR, lj = tx + c * THR;
            sA[li][lj] = dist[(size_t)(rowBase + li) * n + base + lj];
            sB[li][lj] = dist[(size_t)(base + li) * n + colBase + lj];
            const size_t g = (size_t)(rowBase + li) * n + colBase + lj;
            d[r][c] = dist[g];
            p[r][c] = path[g];
        }
    }
    __syncthreads();

#pragma unroll 8
    for (int k = 0; k < TILE; ++k) {
        unsigned int a[EPT], b[EPT];
#pragma unroll
        for (int r = 0; r < EPT; ++r) a[r] = sA[ty + r * THR][k];
#pragma unroll
        for (int c = 0; c < EPT; ++c) b[c] = sB[k][tx + c * THR];
#pragma unroll
        for (int r = 0; r < EPT; ++r) {
#pragma unroll
            for (int c = 0; c < EPT; ++c) {
                const unsigned int nd = a[r] + b[c];
                if (nd < d[r][c]) {
                    d[r][c] = nd;
                    p[r][c] = base + k;
                }
            }
        }
    }

#pragma unroll
    for (int r = 0; r < EPT; ++r) {
#pragma unroll
        for (int c = 0; c < EPT; ++c) {
            const int li = ty + r * THR, lj = tx + c * THR;
            const size_t g = (size_t)(rowBase + li) * n + colBase + lj;
            dist[g] = d[r][c];
            path[g] = p[r][c];
        }
    }
}

void floydWarshall(std::vector<unsigned int>& dist, 
                   std::vector<unsigned int>& path, 
                   const size_t numNodes) {
    if (numNodes == 0) return;
    const size_t nb = (numNodes + TILE - 1) / TILE;
    const size_t np = nb * TILE;  // padded size
    const size_t bytes = np * np * sizeof(unsigned int);

    unsigned int *dDist = nullptr, *dPath = nullptr;
    CUDA_CHECK(cudaMalloc(&dDist, bytes));
    CUDA_CHECK(cudaMalloc(&dPath, bytes));

    if (np != numNodes) {
        // Padding: unreachable nodes (INF), zero self-distance
        std::vector<unsigned int> pad(np * np, INF);
        std::vector<unsigned int> padPath(np * np, 0);
        for (size_t i = 0; i < np; ++i) pad[i * np + i] = 0;
        for (size_t i = 0; i < numNodes; ++i) {
            std::memcpy(&pad[i * np], &dist[i * numNodes], numNodes * sizeof(unsigned int));
            std::memcpy(&padPath[i * np], &path[i * numNodes], numNodes * sizeof(unsigned int));
        }
        CUDA_CHECK(cudaMemcpy(dDist, pad.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dPath, padPath.data(), bytes, cudaMemcpyHostToDevice));
    } else {
        CUDA_CHECK(cudaMemcpy(dDist, dist.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dPath, path.data(), bytes, cudaMemcpyHostToDevice));
    }

    const int n = static_cast<int>(np);
    const dim3 blk12(THR12, THR12);
    const dim3 blk3(THR, THR);
    const dim3 grid2(static_cast<unsigned>(nb), 2);
    const dim3 grid3(static_cast<unsigned>(nb), static_cast<unsigned>(nb));
    for (int kb = 0; kb < static_cast<int>(nb); ++kb) {
        fwPhase1<<<1, blk12>>>(dDist, dPath, n, kb);
        if (nb > 1) {
            fwPhase2<<<grid2, blk12>>>(dDist, dPath, n, kb);
            fwPhase3<<<grid3, blk3>>>(dDist, dPath, n, kb);
        }
    }
    CUDA_CHECK(cudaGetLastError());

    const size_t rowBytes = numNodes * sizeof(unsigned int);
    const size_t pitch = np * sizeof(unsigned int);
    CUDA_CHECK(cudaMemcpy2D(dist.data(), rowBytes, dDist, pitch, rowBytes, numNodes,
                            cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy2D(path.data(), rowBytes, dPath, pitch, rowBytes, numNodes,
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
    
    // Allocate matrices
    std::vector<unsigned int> dist(numNodes * numNodes);
    std::vector<unsigned int> path(numNodes * numNodes);
    
    // Initialize
    printf("Initializing graph...\n");
    initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
    initializePathMatrix(path, numNodes);
    
    // Initialize the CUDA context outside of the timed region
    CUDA_CHECK(cudaFree(0));

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
