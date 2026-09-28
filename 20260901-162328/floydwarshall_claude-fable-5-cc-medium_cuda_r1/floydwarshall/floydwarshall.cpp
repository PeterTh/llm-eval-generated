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

// Sentinel used for padding entries on the device. Large enough to never
// improve a real path, small enough that adding two of them cannot overflow.
constexpr unsigned int DEV_INF = 0x3FFFFFFFu;

// Tile edge for the blocked Floyd-Warshall kernels (32x32 = 1024 threads).
constexpr int TILE = 32;

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err_ = (call);                                             \
        if (err_ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__,   \
                    __LINE__, cudaGetErrorString(err_));                       \
            exit(1);                                                           \
        }                                                                      \
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

// Fill the padded device matrices: distances get DEV_INF off-diagonal and 0 on
// the diagonal (so padded phantom nodes are unreachable and never shorten a
// real path), paths get 0. The real sub-matrix is overwritten by the host copy.
__global__ void fwInitPadding(unsigned int* dist, unsigned int* path, const int np) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < np && j < np) {
        dist[(size_t)i * np + j] = (i == j) ? 0u : DEV_INF;
        path[(size_t)i * np + j] = 0u;
    }
}

// Phase 1: relax the pivot tile D[kb][kb] against itself. Updates within the
// tile feed later k iterations, so the loop synchronizes per step. Row/column
// k of the tile never changes during step k (the diagonal stays 0), matching
// the sequential algorithm exactly.
__global__ void fwPhase1(unsigned int* __restrict__ dist,
                         unsigned int* __restrict__ path,
                         const int np, const int kb) {
    __shared__ unsigned int sd[TILE][TILE];
    __shared__ unsigned int sp[TILE][TILE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const size_t g = (size_t)(kb * TILE + ty) * np + (kb * TILE + tx);

    sd[ty][tx] = dist[g];
    sp[ty][tx] = path[g];
    __syncthreads();

    for (int t = 0; t < TILE; ++t) {
        const unsigned int nd = sd[ty][t] + sd[t][tx];
        if (nd < sd[ty][tx]) {
            sd[ty][tx] = nd;
            sp[ty][tx] = kb * TILE + t;
        }
        __syncthreads();
    }

    dist[g] = sd[ty][tx];
    path[g] = sp[ty][tx];
}

// Phase 2: relax the tiles sharing a row (blockIdx.y == 0) or a column
// (blockIdx.y == 1) with the pivot tile. The current tile is updated in place
// as k advances, so the loop synchronizes per step.
__global__ void fwPhase2(unsigned int* __restrict__ dist,
                         unsigned int* __restrict__ path,
                         const int np, const int kb) {
    const int m = blockIdx.x;
    if (m == kb) return;  // pivot tile was handled in phase 1

    __shared__ unsigned int sd[TILE][TILE];   // tile being updated
    __shared__ unsigned int sc[TILE][TILE];   // pivot tile
    __shared__ unsigned int sp[TILE][TILE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const bool rowTile = (blockIdx.y == 0);

    const int bi = rowTile ? kb : m;
    const int bj = rowTile ? m : kb;
    const size_t g = (size_t)(bi * TILE + ty) * np + (bj * TILE + tx);
    const size_t gp = (size_t)(kb * TILE + ty) * np + (kb * TILE + tx);

    sd[ty][tx] = dist[g];
    sp[ty][tx] = path[g];
    sc[ty][tx] = dist[gp];
    __syncthreads();

    for (int t = 0; t < TILE; ++t) {
        const unsigned int nd = rowTile ? (sc[ty][t] + sd[t][tx])
                                        : (sd[ty][t] + sc[t][tx]);
        if (nd < sd[ty][tx]) {
            sd[ty][tx] = nd;
            sp[ty][tx] = kb * TILE + t;
        }
        __syncthreads();
    }

    dist[g] = sd[ty][tx];
    path[g] = sp[ty][tx];
}

// Phase 3: relax all remaining tiles. The needed row tile D[kb][bx] and
// column tile D[by][kb] are already final for this k block, so each thread
// accumulates its minimum in registers with no per-step synchronization.
__global__ void fwPhase3(unsigned int* __restrict__ dist,
                         unsigned int* __restrict__ path,
                         const int np, const int kb) {
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    if (bx == kb || by == kb) return;  // handled in phases 1 and 2

    __shared__ unsigned int scol[TILE][TILE];  // D[by][kb]
    __shared__ unsigned int srow[TILE][TILE];  // D[kb][bx]

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const size_t g = (size_t)(by * TILE + ty) * np + (bx * TILE + tx);

    scol[ty][tx] = dist[(size_t)(by * TILE + ty) * np + (kb * TILE + tx)];
    srow[ty][tx] = dist[(size_t)(kb * TILE + ty) * np + (bx * TILE + tx)];
    __syncthreads();

    unsigned int best = dist[g];
    unsigned int bestPath = 0;
    bool improved = false;

    #pragma unroll
    for (int t = 0; t < TILE; ++t) {
        const unsigned int nd = scol[ty][t] + srow[t][tx];
        if (nd < best) {
            best = nd;
            bestPath = kb * TILE + t;
            improved = true;
        }
    }

    if (improved) {
        dist[g] = best;
        path[g] = bestPath;
    }
}

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes) {
    if (numNodes == 0) return;

    // Pad the matrices to a multiple of the tile size so the kernels need no
    // bounds checks. Phantom nodes are unreachable (DEV_INF edges, 0 diagonal)
    // and therefore never affect real shortest paths.
    const int nBlocks = (int)((numNodes + TILE - 1) / TILE);
    const int np = nBlocks * TILE;
    const size_t rowBytes = numNodes * sizeof(unsigned int);
    const size_t pitchBytes = (size_t)np * sizeof(unsigned int);

    unsigned int* dDist = nullptr;
    unsigned int* dPath = nullptr;
    CUDA_CHECK(cudaMalloc(&dDist, (size_t)np * np * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&dPath, (size_t)np * np * sizeof(unsigned int)));

    {
        const dim3 block(TILE, TILE);
        const dim3 grid(nBlocks, nBlocks);
        fwInitPadding<<<grid, block>>>(dDist, dPath, np);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaMemcpy2D(dDist, pitchBytes, dist.data(), rowBytes,
                            rowBytes, numNodes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy2D(dPath, pitchBytes, path.data(), rowBytes,
                            rowBytes, numNodes, cudaMemcpyHostToDevice));

    const dim3 block(TILE, TILE);
    for (int kb = 0; kb < nBlocks; ++kb) {
        fwPhase1<<<1, block>>>(dDist, dPath, np, kb);
        fwPhase2<<<dim3(nBlocks, 2), block>>>(dDist, dPath, np, kb);
        fwPhase3<<<dim3(nBlocks, nBlocks), block>>>(dDist, dPath, np, kb);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy2D(dist.data(), rowBytes, dDist, pitchBytes,
                            rowBytes, numNodes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy2D(path.data(), rowBytes, dPath, pitchBytes,
                            rowBytes, numNodes, cudaMemcpyDeviceToHost));

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
