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

// Tile size for the blocked Floyd-Warshall kernels (32x32 = 1024 threads/block)
constexpr int TILE = 32;

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err__ = (call);                                               \
        if (err__ != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__,      \
                    __LINE__, cudaGetErrorString(err__));                         \
            exit(1);                                                              \
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

// Blocked Floyd-Warshall on the GPU. The matrix is padded to a multiple of
// TILE; D(i,j) is stored at dist[j * n + i] (i is the contiguous dimension),
// so threadIdx.x always indexes i for coalesced accesses.
//
// Phase 1: relax the pivot tile (kb, kb) against its own k values.
__global__ void fwPhase1(unsigned int* __restrict__ dist,
                         unsigned int* __restrict__ path,
                         const int kb, const int n) {
    __shared__ unsigned int s[TILE][TILE + 1];  // s[j_local][i_local] = D(i,j)

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int i = kb * TILE + tx;
    const int j = kb * TILE + ty;
    const size_t gidx = (size_t)j * n + i;

    s[ty][tx] = dist[gidx];
    __syncthreads();

    unsigned int cur = s[ty][tx];
    int bestK = -1;

    for (int k = 0; k < TILE; ++k) {
        // D(i, kk) = s[k][tx], D(kk, j) = s[ty][k]
        const unsigned int cand = s[k][tx] + s[ty][k];
        if (cand < cur) {
            cur = cand;
            bestK = kb * TILE + k;
        }
        __syncthreads();
        s[ty][tx] = cur;
        __syncthreads();
    }

    dist[gidx] = cur;
    if (bestK >= 0) {
        path[gidx] = (unsigned int)bestK;
    }
}

// Phase 2: relax the tiles sharing a row or column with the pivot tile.
// blockIdx.y == 0 -> pivot-row tiles (i in pivot block, j varies)
// blockIdx.y == 1 -> pivot-column tiles (j in pivot block, i varies)
__global__ void fwPhase2(unsigned int* __restrict__ dist,
                         unsigned int* __restrict__ path,
                         const int kb, const int n) {
    const int b = blockIdx.x;
    if (b == kb) return;  // pivot tile handled in phase 1

    __shared__ unsigned int sPivot[TILE][TILE + 1];
    __shared__ unsigned int sCur[TILE][TILE + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // Load pivot tile (kb, kb)
    {
        const int pi = kb * TILE + tx;
        const int pj = kb * TILE + ty;
        sPivot[ty][tx] = dist[(size_t)pj * n + pi];
    }

    int i, j;
    if (blockIdx.y == 0) {
        i = kb * TILE + tx;
        j = b * TILE + ty;
    } else {
        i = b * TILE + tx;
        j = kb * TILE + ty;
    }
    const size_t gidx = (size_t)j * n + i;
    sCur[ty][tx] = dist[gidx];
    __syncthreads();

    unsigned int cur = sCur[ty][tx];
    int bestK = -1;

    if (blockIdx.y == 0) {
        // Row tile: D(i,kk) comes from the pivot tile, D(kk,j) from this tile.
        for (int k = 0; k < TILE; ++k) {
            const unsigned int cand = sPivot[k][tx] + sCur[ty][k];
            if (cand < cur) {
                cur = cand;
                bestK = kb * TILE + k;
            }
            __syncthreads();
            sCur[ty][tx] = cur;
            __syncthreads();
        }
    } else {
        // Column tile: D(i,kk) comes from this tile, D(kk,j) from the pivot.
        for (int k = 0; k < TILE; ++k) {
            const unsigned int cand = sCur[k][tx] + sPivot[ty][k];
            if (cand < cur) {
                cur = cand;
                bestK = kb * TILE + k;
            }
            __syncthreads();
            sCur[ty][tx] = cur;
            __syncthreads();
        }
    }

    dist[gidx] = cur;
    if (bestK >= 0) {
        path[gidx] = (unsigned int)bestK;
    }
}

// Phase 3: relax all remaining tiles using the already-updated pivot-row and
// pivot-column tiles. No intra-tile dependency, so no syncs inside the loop.
__global__ void fwPhase3(unsigned int* __restrict__ dist,
                         unsigned int* __restrict__ path,
                         const int kb, const int n) {
    const int bi = blockIdx.x;
    const int bj = blockIdx.y;
    if (bi == kb || bj == kb) return;  // handled in phases 1 and 2

    __shared__ unsigned int sCol[TILE][TILE + 1];  // tile (bi, kb): D(i, kk)
    __shared__ unsigned int sRow[TILE][TILE + 1];  // tile (kb, bj): D(kk, j)

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int i = bi * TILE + tx;
    const int j = bj * TILE + ty;

    sCol[ty][tx] = dist[(size_t)(kb * TILE + ty) * n + i];
    sRow[ty][tx] = dist[(size_t)j * n + (kb * TILE + tx)];
    __syncthreads();

    const size_t gidx = (size_t)j * n + i;
    unsigned int cur = dist[gidx];
    int bestK = -1;

    #pragma unroll
    for (int k = 0; k < TILE; ++k) {
        // D(i,kk) = sCol[k][tx], D(kk,j) = sRow[ty][k]
        const unsigned int cand = sCol[k][tx] + sRow[ty][k];
        if (cand < cur) {
            cur = cand;
            bestK = kb * TILE + k;
        }
    }

    if (bestK >= 0) {
        dist[gidx] = cur;
        path[gidx] = (unsigned int)bestK;
    }
}

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes) {
    const int n = (int)numNodes;
    const int numTiles = (n + TILE - 1) / TILE;
    const int np = numTiles * TILE;  // padded dimension

    unsigned int* dDist = nullptr;
    unsigned int* dPath = nullptr;
    const size_t paddedBytes = (size_t)np * np * sizeof(unsigned int);
    CUDA_CHECK(cudaMalloc(&dDist, paddedBytes));
    CUDA_CHECK(cudaMalloc(&dPath, paddedBytes));

    if (np == n) {
        CUDA_CHECK(cudaMemcpy(dDist, dist.data(), paddedBytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dPath, path.data(), paddedBytes, cudaMemcpyHostToDevice));
    } else {
        // Pad with INF edges (diagonal 0) so padding nodes never shorten a path.
        // INF + INF fits in unsigned int, so relaxation cannot overflow.
        std::vector<unsigned int> hPad((size_t)np * np, INF);
        for (int j = 0; j < np; ++j) {
            if (j < n) {
                memcpy(&hPad[(size_t)j * np], &dist[(size_t)j * n], n * sizeof(unsigned int));
            }
            hPad[(size_t)j * np + j] = (j < n) ? dist[idx2(j, j, numNodes)] : 0;
        }
        CUDA_CHECK(cudaMemcpy(dDist, hPad.data(), paddedBytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy2D(dPath, (size_t)np * sizeof(unsigned int),
                                path.data(), (size_t)n * sizeof(unsigned int),
                                (size_t)n * sizeof(unsigned int), n,
                                cudaMemcpyHostToDevice));
    }

    const dim3 threads(TILE, TILE);
    const dim3 gridP2(numTiles, 2);
    const dim3 gridP3(numTiles, numTiles);

    for (int kb = 0; kb < numTiles; ++kb) {
        fwPhase1<<<1, threads>>>(dDist, dPath, kb, np);
        fwPhase2<<<gridP2, threads>>>(dDist, dPath, kb, np);
        fwPhase3<<<gridP3, threads>>>(dDist, dPath, kb, np);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    if (np == n) {
        CUDA_CHECK(cudaMemcpy(dist.data(), dDist, paddedBytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(path.data(), dPath, paddedBytes, cudaMemcpyDeviceToHost));
    } else {
        CUDA_CHECK(cudaMemcpy2D(dist.data(), (size_t)n * sizeof(unsigned int),
                                dDist, (size_t)np * sizeof(unsigned int),
                                (size_t)n * sizeof(unsigned int), n,
                                cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy2D(path.data(), (size_t)n * sizeof(unsigned int),
                                dPath, (size_t)np * sizeof(unsigned int),
                                (size_t)n * sizeof(unsigned int), n,
                                cudaMemcpyDeviceToHost));
    }

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
