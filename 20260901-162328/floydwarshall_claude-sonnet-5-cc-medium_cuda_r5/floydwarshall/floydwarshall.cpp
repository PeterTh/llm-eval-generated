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

// Tile size for the blocked GPU Floyd-Warshall (32x32 = 1024 threads/block).
constexpr int TILE = 32;

#define CUDA_CHECK(call)                                                       \
    do {                                                                      \
        cudaError_t err__ = (call);                                           \
        if (err__ != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err__));                               \
            exit(1);                                                          \
        }                                                                     \
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

// -----------------------------------------------------------------------
// Blocked (tiled) GPU Floyd-Warshall, following the classic 3-phase
// formulation (Katz & Kider). The flattened arrays used throughout this
// file store the pair (i, j) at dist[idx2(j, i, n)] == dist[i * n + j],
// which is exactly a plain row-major matrix D(i, j) = dist[i * n + j].
// The kernels below operate directly on that row-major layout.
// -----------------------------------------------------------------------

// Phase 1: update the single diagonal block that contains the pivot round.
__global__ void fwPhase1(unsigned int* dist, unsigned int* path, int n, int round) {
    __shared__ unsigned int sD[TILE][TILE];
    __shared__ unsigned int sP[TILE][TILE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int base = round * TILE;
    const int row = base + ty;
    const int col = base + tx;

    sD[ty][tx] = dist[row * n + col];
    sP[ty][tx] = path[row * n + col];
    __syncthreads();

#pragma unroll
    for (int k = 0; k < TILE; ++k) {
        const unsigned int nd = sD[ty][k] + sD[k][tx];
        if (nd < sD[ty][tx]) {
            sD[ty][tx] = nd;
            sP[ty][tx] = base + k;
        }
        __syncthreads();
    }

    dist[row * n + col] = sD[ty][tx];
    path[row * n + col] = sP[ty][tx];
}

// Phase 2: update the blocks that share a row-block or column-block with the
// pivot block (blockIdx.y == 0 -> row blocks, blockIdx.y == 1 -> column blocks).
__global__ void fwPhase2(unsigned int* dist, unsigned int* path, int n, int round) {
    const int blk = blockIdx.x;
    if (blk == round) return;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int base = round * TILE;

    __shared__ unsigned int sPrim[TILE][TILE];
    __shared__ unsigned int sCur[TILE][TILE];
    __shared__ unsigned int sCurP[TILE][TILE];

    sPrim[ty][tx] = dist[(base + ty) * n + (base + tx)];

    int row, col;
    if (blockIdx.y == 0) {
        // Row blocks: fixed row-block = round, varying column-block = blk.
        row = base + ty;
        col = blk * TILE + tx;
    } else {
        // Column blocks: fixed column-block = round, varying row-block = blk.
        row = blk * TILE + ty;
        col = base + tx;
    }

    sCur[ty][tx] = dist[row * n + col];
    sCurP[ty][tx] = path[row * n + col];
    __syncthreads();

    if (blockIdx.y == 0) {
#pragma unroll
        for (int k = 0; k < TILE; ++k) {
            const unsigned int nd = sPrim[ty][k] + sCur[k][tx];
            if (nd < sCur[ty][tx]) {
                sCur[ty][tx] = nd;
                sCurP[ty][tx] = base + k;
            }
            __syncthreads();
        }
    } else {
#pragma unroll
        for (int k = 0; k < TILE; ++k) {
            const unsigned int nd = sCur[ty][k] + sPrim[k][tx];
            if (nd < sCur[ty][tx]) {
                sCur[ty][tx] = nd;
                sCurP[ty][tx] = base + k;
            }
            __syncthreads();
        }
    }

    dist[row * n + col] = sCur[ty][tx];
    path[row * n + col] = sCurP[ty][tx];
}

// Phase 3: update every remaining block using the already-updated row block
// (rr, round) and column block (round, c) from phases 1 and 2.
__global__ void fwPhase3(unsigned int* dist, unsigned int* path, int n, int round) {
    const int rr = blockIdx.y;
    const int c = blockIdx.x;
    if (rr == round || c == round) return;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int base = round * TILE;

    __shared__ unsigned int sRow[TILE][TILE];
    __shared__ unsigned int sCol[TILE][TILE];

    const int row = rr * TILE + ty;
    const int col = c * TILE + tx;

    sRow[ty][tx] = dist[row * n + (base + tx)];
    sCol[ty][tx] = dist[(base + ty) * n + col];
    __syncthreads();

    unsigned int d = dist[row * n + col];
    unsigned int p = path[row * n + col];

#pragma unroll
    for (int k = 0; k < TILE; ++k) {
        const unsigned int nd = sRow[ty][k] + sCol[k][tx];
        if (nd < d) {
            d = nd;
            p = base + k;
        }
    }

    dist[row * n + col] = d;
    path[row * n + col] = p;
}

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes) {
    if (numNodes == 0) return;

    const int n = static_cast<int>(numNodes);
    const int numBlocks = (n + TILE - 1) / TILE;
    const int nPad = numBlocks * TILE;
    const size_t padElems = static_cast<size_t>(nPad) * static_cast<size_t>(nPad);

    // Pad the matrices to a multiple of TILE. Padding distances are left at
    // INF so padded nodes can never form a shorter path between real nodes.
    std::vector<unsigned int> distPad(padElems, INF);
    std::vector<unsigned int> pathPad(padElems, 0);

    for (int i = 0; i < n; ++i) {
        std::memcpy(&distPad[static_cast<size_t>(i) * nPad], &dist[static_cast<size_t>(i) * n],
                    n * sizeof(unsigned int));
        std::memcpy(&pathPad[static_cast<size_t>(i) * nPad], &path[static_cast<size_t>(i) * n],
                    n * sizeof(unsigned int));
    }

    unsigned int* dDist = nullptr;
    unsigned int* dPath = nullptr;
    CUDA_CHECK(cudaMalloc(&dDist, padElems * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&dPath, padElems * sizeof(unsigned int)));

    CUDA_CHECK(cudaMemcpy(dDist, distPad.data(), padElems * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dPath, pathPad.data(), padElems * sizeof(unsigned int), cudaMemcpyHostToDevice));

    const dim3 blockDim(TILE, TILE);
    const dim3 phase2Grid(numBlocks, 2);
    const dim3 phase3Grid(numBlocks, numBlocks);

    for (int round = 0; round < numBlocks; ++round) {
        fwPhase1<<<1, blockDim>>>(dDist, dPath, nPad, round);
        if (numBlocks > 1) {
            fwPhase2<<<phase2Grid, blockDim>>>(dDist, dPath, nPad, round);
            fwPhase3<<<phase3Grid, blockDim>>>(dDist, dPath, nPad, round);
        }
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(distPad.data(), dDist, padElems * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(pathPad.data(), dPath, padElems * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(dDist));
    CUDA_CHECK(cudaFree(dPath));

    for (int i = 0; i < n; ++i) {
        std::memcpy(&dist[static_cast<size_t>(i) * n], &distPad[static_cast<size_t>(i) * nPad],
                    n * sizeof(unsigned int));
        std::memcpy(&path[static_cast<size_t>(i) * n], &pathPad[static_cast<size_t>(i) * nPad],
                    n * sizeof(unsigned int));
    }
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

    // Warm up the CUDA context/driver so its one-time initialization cost
    // is not attributed to the timed computation below.
    CUDA_CHECK(cudaFree(0));

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
