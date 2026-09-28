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

// Tile size for the blocked GPU Floyd-Warshall kernels.
constexpr int BLOCK_SIZE = 32;

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err__ = (call);                                               \
        if (err__ != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err__));                                   \
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

// ---------------------------------------------------------------------------
// Blocked (tiled) GPU Floyd-Warshall.
//
// The n x n distance/path matrices are conceptually divided into BLOCK_SIZE x
// BLOCK_SIZE tiles. For each "round" (pivot block index r) three kernel
// phases run, mirroring the well known Katz & Kider blocked APSP algorithm:
//   Phase 1: update the pivot tile (r, r) in place.
//   Phase 2: update every tile in the pivot row (r, *) and pivot column
//            (*, r) using the freshly updated pivot tile.
//   Phase 3: update every remaining tile (I, J) using the updated pivot row
//            tile (r, J) and pivot column tile (I, r).
//
// Because the pivot value k is swept in the same 0..n-1 order as the
// sequential algorithm, and each round fully resolves all i,j pairs before
// moving to the next k range, the results are numerically identical to the
// classic triple-nested-loop implementation.
//
// Matrices are stored row-major with index(i, j) = i * n + j, matching the
// element accessed via idx2(j, i, n) in the original sequential code.
// ---------------------------------------------------------------------------

__global__ void fwPhase1(unsigned int* dist, unsigned int* path, const int n, const int round) {
    __shared__ unsigned int distTile[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int pathTile[BLOCK_SIZE][BLOCK_SIZE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int base = round * BLOCK_SIZE;
    const int gi = base + ty;
    const int gj = base + tx;

    distTile[ty][tx] = dist[(size_t)gi * n + gj];
    pathTile[ty][tx] = path[(size_t)gi * n + gj];
    __syncthreads();

    #pragma unroll
    for (int k = 0; k < BLOCK_SIZE; ++k) {
        const unsigned int newDist = distTile[ty][k] + distTile[k][tx];
        if (newDist < distTile[ty][tx]) {
            distTile[ty][tx] = newDist;
            pathTile[ty][tx] = base + k;
        }
        __syncthreads();
    }

    dist[(size_t)gi * n + gj] = distTile[ty][tx];
    path[(size_t)gi * n + gj] = pathTile[ty][tx];
}

__global__ void fwPhase2(unsigned int* dist, unsigned int* path, const int n, const int round) {
    const int blockIdxAlong = blockIdx.x;   // 0 .. numBlocks-1, skipping 'round'
    const bool isRowPass = (blockIdx.y == 0);

    if (blockIdxAlong == round) return;

    __shared__ unsigned int pivotDist[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int ownDist[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int ownPath[BLOCK_SIZE][BLOCK_SIZE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int base = round * BLOCK_SIZE;

    int ownRow, ownCol;
    if (isRowPass) {
        // Tile (round, blockIdxAlong)
        ownRow = base + ty;
        ownCol = blockIdxAlong * BLOCK_SIZE + tx;
    } else {
        // Tile (blockIdxAlong, round)
        ownRow = blockIdxAlong * BLOCK_SIZE + ty;
        ownCol = base + tx;
    }

    pivotDist[ty][tx] = dist[(size_t)(base + ty) * n + (base + tx)];
    ownDist[ty][tx] = dist[(size_t)ownRow * n + ownCol];
    ownPath[ty][tx] = path[(size_t)ownRow * n + ownCol];
    __syncthreads();

    if (isRowPass) {
        // dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
        // i,k in pivot row range -> pivotDist[ty][k]; k,j in own tile -> ownDist[k][tx]
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            const unsigned int newDist = pivotDist[ty][k] + ownDist[k][tx];
            if (newDist < ownDist[ty][tx]) {
                ownDist[ty][tx] = newDist;
                ownPath[ty][tx] = base + k;
            }
            __syncthreads();
        }
    } else {
        // i in own row range, k,j in pivot col range -> ownDist[ty][k], pivotDist[k][tx]
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            const unsigned int newDist = ownDist[ty][k] + pivotDist[k][tx];
            if (newDist < ownDist[ty][tx]) {
                ownDist[ty][tx] = newDist;
                ownPath[ty][tx] = base + k;
            }
            __syncthreads();
        }
    }

    dist[(size_t)ownRow * n + ownCol] = ownDist[ty][tx];
    path[(size_t)ownRow * n + ownCol] = ownPath[ty][tx];
}

__global__ void fwPhase3(unsigned int* dist, unsigned int* path, const int n, const int round) {
    const int blockRow = blockIdx.y;
    const int blockCol = blockIdx.x;

    if (blockRow == round || blockCol == round) return;

    __shared__ unsigned int rowTileDist[BLOCK_SIZE][BLOCK_SIZE]; // tile (round, blockCol)
    __shared__ unsigned int colTileDist[BLOCK_SIZE][BLOCK_SIZE]; // tile (blockRow, round)

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int base = round * BLOCK_SIZE;

    const int gi = blockRow * BLOCK_SIZE + ty;
    const int gj = blockCol * BLOCK_SIZE + tx;

    rowTileDist[ty][tx] = dist[(size_t)(base + ty) * n + gj];
    colTileDist[ty][tx] = dist[(size_t)gi * n + (base + tx)];
    __syncthreads();

    unsigned int ownDist = dist[(size_t)gi * n + gj];
    unsigned int ownPath = path[(size_t)gi * n + gj];

    #pragma unroll
    for (int k = 0; k < BLOCK_SIZE; ++k) {
        const unsigned int newDist = colTileDist[ty][k] + rowTileDist[k][tx];
        if (newDist < ownDist) {
            ownDist = newDist;
            ownPath = base + k;
        }
    }

    dist[(size_t)gi * n + gj] = ownDist;
    path[(size_t)gi * n + gj] = ownPath;
}

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes) {
    const int n = static_cast<int>(numNodes);
    const int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int paddedN = numBlocks * BLOCK_SIZE;

    // Pad the matrices to a multiple of BLOCK_SIZE. Padding "nodes" are
    // fictitious and are given INF distance to every real node so that they
    // can never form part of a real shortest path; the path matrix padding
    // values are never read back.
    std::vector<unsigned int> paddedDist(static_cast<size_t>(paddedN) * paddedN, INF);
    std::vector<unsigned int> paddedPath(static_cast<size_t>(paddedN) * paddedN, 0);

    for (int i = 0; i < n; ++i) {
        std::memcpy(&paddedDist[(size_t)i * paddedN], &dist[idx2(0, i, numNodes)],
                    n * sizeof(unsigned int));
        std::memcpy(&paddedPath[(size_t)i * paddedN], &path[idx2(0, i, numNodes)],
                    n * sizeof(unsigned int));
    }
    for (int i = n; i < paddedN; ++i) {
        paddedDist[(size_t)i * paddedN + i] = 0;
    }

    unsigned int* dDist = nullptr;
    unsigned int* dPath = nullptr;
    const size_t bytes = static_cast<size_t>(paddedN) * paddedN * sizeof(unsigned int);
    CUDA_CHECK(cudaMalloc(&dDist, bytes));
    CUDA_CHECK(cudaMalloc(&dPath, bytes));
    CUDA_CHECK(cudaMemcpy(dDist, paddedDist.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dPath, paddedPath.data(), bytes, cudaMemcpyHostToDevice));

    const dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);

    for (int round = 0; round < numBlocks; ++round) {
        fwPhase1<<<1, blockDim>>>(dDist, dPath, paddedN, round);

        const dim3 phase2Grid(numBlocks, 2);
        fwPhase2<<<phase2Grid, blockDim>>>(dDist, dPath, paddedN, round);

        const dim3 phase3Grid(numBlocks, numBlocks);
        fwPhase3<<<phase3Grid, blockDim>>>(dDist, dPath, paddedN, round);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(paddedDist.data(), dDist, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(paddedPath.data(), dPath, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(dDist));
    CUDA_CHECK(cudaFree(dPath));

    for (int i = 0; i < n; ++i) {
        std::memcpy(&dist[idx2(0, i, numNodes)], &paddedDist[(size_t)i * paddedN],
                    n * sizeof(unsigned int));
        std::memcpy(&path[idx2(0, i, numNodes)], &paddedPath[(size_t)i * paddedN],
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

    // Allocate matrices
    std::vector<unsigned int> dist(numNodes * numNodes);
    std::vector<unsigned int> path(numNodes * numNodes);

    // Warm up the CUDA context/driver so its one-time initialization cost is
    // not attributed to the timed computation below.
    CUDA_CHECK(cudaFree(0));

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
