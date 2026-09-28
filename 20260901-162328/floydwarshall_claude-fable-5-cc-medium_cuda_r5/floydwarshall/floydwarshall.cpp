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

// Tile size for the blocked Floyd-Warshall kernels (thread block is TILE x TILE)
constexpr int TILE = 32;

#define CUDA_CHECK(call)                                                            \
    do {                                                                            \
        cudaError_t err_ = (call);                                                  \
        if (err_ != cudaSuccess) {                                                  \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_),   \
                    __FILE__, __LINE__);                                            \
            exit(1);                                                                \
        }                                                                           \
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

// The matrix is stored so that element (i, j) lives at dist[i * n + j]
// (see idx2 usage in the original scalar loops). All kernels below use that
// row-major convention on a padded n x n matrix (np is a multiple of TILE).
// Padding cells hold INF off-diagonal and 0 on the diagonal, so they can never
// strictly improve a real shortest path and are simply ignored on copy-back.

__global__ void initPaddedKernel(unsigned int* dist, unsigned int* path, const int np) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < np && j < np) {
        dist[i * np + j] = (i == j) ? 0u : INF;
        path[i * np + j] = j;
    }
}

// Phase 1: relax the dependent tile (kb, kb) against itself.
__global__ void fwPhase1(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
                         const int np, const int kb) {
    __shared__ unsigned int sd[TILE][TILE];
    __shared__ unsigned int sp[TILE][TILE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int base = kb * TILE;
    const size_t g = (size_t)(base + ty) * np + (base + tx);

    sd[ty][tx] = dist[g];
    sp[ty][tx] = path[g];
    __syncthreads();

    for (int k = 0; k < TILE; ++k) {
        // sd[ty][k] and sd[k][tx] are never modified in iteration k because the
        // diagonal element sd[k][k] is 0, so their relaxation is never strict.
        const unsigned int nd = sd[ty][k] + sd[k][tx];
        if (nd < sd[ty][tx]) {
            sd[ty][tx] = nd;
            sp[ty][tx] = base + k;
        }
        __syncthreads();
    }

    dist[g] = sd[ty][tx];
    path[g] = sp[ty][tx];
}

// Phase 2: relax the tiles sharing a block row (y == 0) or block column
// (y == 1) with the dependent tile, using the phase-1 result.
__global__ void fwPhase2(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
                         const int np, const int kb) {
    const int t = blockIdx.x;
    if (t == kb) return;

    __shared__ unsigned int sk[TILE][TILE];  // dependent tile (kb, kb)
    __shared__ unsigned int sd[TILE][TILE];  // this tile's distances

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int base = kb * TILE;

    sk[ty][tx] = dist[(size_t)(base + ty) * np + (base + tx)];

    size_t g;
    if (blockIdx.y == 0) {
        // Row tile (kb, t): dist(i, k) comes from the dependent tile.
        g = (size_t)(base + ty) * np + (t * TILE + tx);
    } else {
        // Column tile (t, kb): dist(k, j) comes from the dependent tile.
        g = (size_t)(t * TILE + ty) * np + (base + tx);
    }
    sd[ty][tx] = dist[g];
    unsigned int pv = path[g];
    __syncthreads();

    if (blockIdx.y == 0) {
        for (int k = 0; k < TILE; ++k) {
            // sd row k is stable in iteration k: its candidate sk[k][k] + sd[k][tx]
            // equals sd[k][tx] (diagonal is 0), so it is never strictly improved.
            const unsigned int nd = sk[ty][k] + sd[k][tx];
            if (nd < sd[ty][tx]) {
                sd[ty][tx] = nd;
                pv = base + k;
            }
            __syncthreads();
        }
    } else {
        for (int k = 0; k < TILE; ++k) {
            const unsigned int nd = sd[ty][k] + sk[k][tx];
            if (nd < sd[ty][tx]) {
                sd[ty][tx] = nd;
                pv = base + k;
            }
            __syncthreads();
        }
    }

    dist[g] = sd[ty][tx];
    path[g] = pv;
}

// Phase 3: relax all remaining tiles using the finished block row and block
// column from phase 2. Each tile is independent, so no synchronization is
// needed inside the k loop and the running value stays in registers.
__global__ void fwPhase3(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
                         const int np, const int kb) {
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    if (bx == kb || by == kb) return;

    __shared__ unsigned int srow[TILE][TILE];  // tile (kb, bx): dist(k, j)
    __shared__ unsigned int scol[TILE][TILE];  // tile (by, kb): dist(i, k)

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int base = kb * TILE;

    srow[ty][tx] = dist[(size_t)(base + ty) * np + (bx * TILE + tx)];
    scol[ty][tx] = dist[(size_t)(by * TILE + ty) * np + (base + tx)];
    __syncthreads();

    const size_t g = (size_t)(by * TILE + ty) * np + (bx * TILE + tx);
    unsigned int dv = dist[g];
    unsigned int pv = path[g];

    #pragma unroll
    for (int k = 0; k < TILE; ++k) {
        const unsigned int nd = scol[ty][k] + srow[k][tx];
        if (nd < dv) {
            dv = nd;
            pv = base + k;
        }
    }

    dist[g] = dv;
    path[g] = pv;
}

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes) {
    const int n = (int)numNodes;
    const int numTiles = (n + TILE - 1) / TILE;
    const int np = numTiles * TILE;  // padded dimension
    const size_t bytes = (size_t)np * np * sizeof(unsigned int);

    unsigned int *dDist, *dPath;
    CUDA_CHECK(cudaMalloc(&dDist, bytes));
    CUDA_CHECK(cudaMalloc(&dPath, bytes));

    // Initialize the padded matrices, then overwrite the real n x n region.
    {
        dim3 block(TILE, TILE);
        dim3 grid(numTiles, numTiles);
        initPaddedKernel<<<grid, block>>>(dDist, dPath, np);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaMemcpy2D(dDist, (size_t)np * sizeof(unsigned int),
                            dist.data(), (size_t)n * sizeof(unsigned int),
                            (size_t)n * sizeof(unsigned int), n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy2D(dPath, (size_t)np * sizeof(unsigned int),
                            path.data(), (size_t)n * sizeof(unsigned int),
                            (size_t)n * sizeof(unsigned int), n, cudaMemcpyHostToDevice));

    const dim3 block(TILE, TILE);
    const dim3 gridP2(numTiles, 2);
    const dim3 gridP3(numTiles, numTiles);

    for (int kb = 0; kb < numTiles; ++kb) {
        fwPhase1<<<1, block>>>(dDist, dPath, np, kb);
        fwPhase2<<<gridP2, block>>>(dDist, dPath, np, kb);
        fwPhase3<<<gridP3, block>>>(dDist, dPath, np, kb);
    }
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy2D(dist.data(), (size_t)n * sizeof(unsigned int),
                            dDist, (size_t)np * sizeof(unsigned int),
                            (size_t)n * sizeof(unsigned int), n, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy2D(path.data(), (size_t)n * sizeof(unsigned int),
                            dPath, (size_t)np * sizeof(unsigned int),
                            (size_t)n * sizeof(unsigned int), n, cudaMemcpyDeviceToHost));

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

    // Initialize the CUDA context so setup cost is not part of the timing.
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
