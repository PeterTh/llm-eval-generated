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

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                         \
                    cudaGetErrorString(err_), __FILE__, __LINE__);              \
            exit(1);                                                            \
        }                                                                       \
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

// Tile size for the blocked Floyd-Warshall kernels (one thread per tile element)
constexpr int TILE = 32;

// The device matrices are padded to a multiple of TILE and the padding is
// filled with PAD_DIST. It is large enough that a padded node never improves
// a real path, and small enough that PAD_DIST + PAD_DIST cannot overflow.
constexpr unsigned int PAD_DIST = 0x3F3F3F3Fu;

// Phase 1: relax the diagonal tile (kb, kb) against all k within the tile.
__global__ void fwPhase1(unsigned int* __restrict__ dist,
                         unsigned int* __restrict__ path,
                         const int n, const int kb) {
    __shared__ unsigned int tile[TILE][TILE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int base = kb * TILE;
    const size_t g = (size_t)(base + ty) * n + (base + tx);

    tile[ty][tx] = dist[g];
    unsigned int p = path[g];
    __syncthreads();

    #pragma unroll
    for (int k = 0; k < TILE; ++k) {
        const unsigned int cand = tile[ty][k] + tile[k][tx];
        __syncthreads();
        if (cand < tile[ty][tx]) {
            tile[ty][tx] = cand;
            p = base + k;
        }
        __syncthreads();
    }

    dist[g] = tile[ty][tx];
    path[g] = p;
}

// Phase 2: relax the tiles sharing a row or column with the diagonal tile.
// blockIdx.y == 0 processes tile (kb, b) in the pivot row,
// blockIdx.y == 1 processes tile (b, kb) in the pivot column.
__global__ void fwPhase2(unsigned int* __restrict__ dist,
                         unsigned int* __restrict__ path,
                         const int n, const int kb) {
    __shared__ unsigned int pivot[TILE][TILE];
    __shared__ unsigned int cur[TILE][TILE];

    int b = blockIdx.x;
    if (b >= kb) ++b;  // skip the diagonal tile

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int base = kb * TILE;

    const size_t gPivot = (size_t)(base + ty) * n + (base + tx);
    size_t g;
    if (blockIdx.y == 0) {
        g = (size_t)(base + ty) * n + (b * TILE + tx);       // pivot row tile
    } else {
        g = (size_t)(b * TILE + ty) * n + (base + tx);       // pivot column tile
    }

    pivot[ty][tx] = dist[gPivot];
    cur[ty][tx] = dist[g];
    unsigned int p = path[g];
    __syncthreads();

    if (blockIdx.y == 0) {
        // dist(i,j) with i in pivot block: dist(i,k) from pivot, dist(k,j) from cur
        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            const unsigned int cand = pivot[ty][k] + cur[k][tx];
            __syncthreads();
            if (cand < cur[ty][tx]) {
                cur[ty][tx] = cand;
                p = base + k;
            }
            __syncthreads();
        }
    } else {
        // dist(i,j) with j in pivot block: dist(i,k) from cur, dist(k,j) from pivot
        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            const unsigned int cand = cur[ty][k] + pivot[k][tx];
            __syncthreads();
            if (cand < cur[ty][tx]) {
                cur[ty][tx] = cand;
                p = base + k;
            }
            __syncthreads();
        }
    }

    dist[g] = cur[ty][tx];
    path[g] = p;
}

// Phase 3: relax all remaining tiles using the already-final pivot row and
// pivot column tiles. No intra-loop synchronization is needed since the
// inputs for every k are read-only here.
__global__ void fwPhase3(unsigned int* __restrict__ dist,
                         unsigned int* __restrict__ path,
                         const int n, const int kb) {
    __shared__ unsigned int rowTile[TILE][TILE];  // tile (kb, bx)
    __shared__ unsigned int colTile[TILE][TILE];  // tile (by, kb)

    int bx = blockIdx.x;
    int by = blockIdx.y;
    if (bx >= kb) ++bx;
    if (by >= kb) ++by;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int base = kb * TILE;

    rowTile[ty][tx] = dist[(size_t)(base + ty) * n + (bx * TILE + tx)];
    colTile[ty][tx] = dist[(size_t)(by * TILE + ty) * n + (base + tx)];
    __syncthreads();

    const size_t g = (size_t)(by * TILE + ty) * n + (bx * TILE + tx);
    unsigned int best = dist[g];
    unsigned int p = path[g];

    #pragma unroll
    for (int k = 0; k < TILE; ++k) {
        const unsigned int cand = colTile[ty][k] + rowTile[k][tx];
        if (cand < best) {
            best = cand;
            p = base + k;
        }
    }

    dist[g] = best;
    path[g] = p;
}

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes) {
    const int n = static_cast<int>(numNodes);
    const int numTiles = (n + TILE - 1) / TILE;
    const int nPad = numTiles * TILE;
    const size_t padBytes = (size_t)nPad * nPad * sizeof(unsigned int);
    const size_t rowBytes = (size_t)n * sizeof(unsigned int);

    unsigned int *dDist, *dPath;
    CUDA_CHECK(cudaMalloc(&dDist, padBytes));
    CUDA_CHECK(cudaMalloc(&dPath, padBytes));

    // Fill padding with a large distance so padded nodes never shorten a path.
    CUDA_CHECK(cudaMemset(dDist, 0x3F, padBytes));

    // Copy the n x n interior into the padded device matrices.
    // Host layout from idx2: element (row i, col j) lives at i * n + j.
    CUDA_CHECK(cudaMemcpy2D(dDist, (size_t)nPad * sizeof(unsigned int),
                            dist.data(), rowBytes, rowBytes, n,
                            cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy2D(dPath, (size_t)nPad * sizeof(unsigned int),
                            path.data(), rowBytes, rowBytes, n,
                            cudaMemcpyHostToDevice));

    const dim3 threads(TILE, TILE);
    const dim3 gridP2(numTiles > 1 ? numTiles - 1 : 1, 2);
    const dim3 gridP3(numTiles > 1 ? numTiles - 1 : 1,
                      numTiles > 1 ? numTiles - 1 : 1);

    for (int kb = 0; kb < numTiles; ++kb) {
        fwPhase1<<<1, threads>>>(dDist, dPath, nPad, kb);
        if (numTiles > 1) {
            fwPhase2<<<gridP2, threads>>>(dDist, dPath, nPad, kb);
            fwPhase3<<<gridP3, threads>>>(dDist, dPath, nPad, kb);
        }
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy2D(dist.data(), rowBytes,
                            dDist, (size_t)nPad * sizeof(unsigned int),
                            rowBytes, n, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy2D(path.data(), rowBytes,
                            dPath, (size_t)nPad * sizeof(unsigned int),
                            rowBytes, n, cudaMemcpyDeviceToHost));

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

    printf("Floyd-Warshall All-Pairs Shortest Path Benchmark (CUDA)\n");
    printf("Number of nodes: %zu\n", numNodes);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");

    // Allocate matrices
    std::vector<unsigned int> dist(numNodes * numNodes);
    std::vector<unsigned int> path(numNodes * numNodes);

    // Initialize
    printf("Initializing graph...\n");
    initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
    initializePathMatrix(path, numNodes);

    // Initialize the CUDA context outside the timed region
    CUDA_CHECK(cudaFree(nullptr));

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
