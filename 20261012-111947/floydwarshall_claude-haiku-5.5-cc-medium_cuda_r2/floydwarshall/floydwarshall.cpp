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

// Blocked Floyd-Warshall tiling parameters
constexpr int TILE = 32;                      // Nodes per tile edge (tile = TILE x TILE)
constexpr int THREADS_X = TILE;               // Thread block is THREADS_X x THREADS_Y
constexpr int THREADS_Y = 8;
constexpr int ROWS_PER_THREAD = TILE / THREADS_Y;

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err_ = (call);                                             \
        if (err_ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err_));                                 \
            exit(EXIT_FAILURE);                                                \
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

// The sequential algorithm reads D[i][k] and D[k][j] as they stand at step k of the
// pivot block. A later step in the same block can still shorten them, so the tile-wise
// phases below record those values at their own step (the "snapshots") and phase 3
// reads the snapshots. This keeps dist and path identical to the sequential loop.

// Phase 1: relax the pivot tile (b, b) using only its own nodes.
// Records pivot column kk and pivot row kk as they stand at step kk.
__global__ void fwPhase1(unsigned int* dist, unsigned int* path, unsigned int* pivotSnap,
                         const int np, const int b) {
    __shared__ unsigned int tile[TILE][TILE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const size_t origin = static_cast<size_t>(b) * TILE * np + static_cast<size_t>(b) * TILE;

    for (int r = ty; r < TILE; r += THREADS_Y) {
        tile[r][tx] = dist[origin + static_cast<size_t>(r) * np + tx];
    }
    __syncthreads();

    for (int kk = 0; kk < TILE; ++kk) {
        for (int r = ty; r < TILE; r += THREADS_Y) {
            // Column kk (threads with tx == kk) and row kk (threads with r == kk)
            // do not change during step kk, so these reads are the step-kk values
            if (tx == kk) {
                pivotSnap[r * TILE + kk] = tile[r][tx];
            }
            if (r == kk) {
                pivotSnap[TILE * TILE + kk * TILE + tx] = tile[r][tx];
            }

            const unsigned int newDist = tile[r][kk] + tile[kk][tx];
            if (newDist < tile[r][tx]) {
                tile[r][tx] = newDist;
                path[origin + static_cast<size_t>(r) * np + tx] = b * TILE + kk;
            }
        }
        __syncthreads();
    }

    for (int r = ty; r < TILE; r += THREADS_Y) {
        dist[origin + static_cast<size_t>(r) * np + tx] = tile[r][tx];
    }
}

// Phase 2: relax the tiles that share the pivot block's row (b, t) or column (t, b).
// blockIdx.x enumerates t over all blocks except b; blockIdx.y selects row (0) or column (1).
// Records this tile's values at each step for phase 3: rowSnap for row tiles, colSnap for column tiles.
__global__ void fwPhase2(unsigned int* dist, unsigned int* path, const unsigned int* pivotSnap,
                         unsigned int* colSnap, unsigned int* rowSnap,
                         const int np, const int b) {
    __shared__ unsigned int pivotCol[TILE][TILE];   // Pivot column kk at step kk
    __shared__ unsigned int pivotRow[TILE][TILE];   // Pivot row kk at step kk
    __shared__ unsigned int tile[TILE][TILE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int t = blockIdx.x < b ? blockIdx.x : blockIdx.x + 1;
    const bool rowCase = blockIdx.y == 0;

    const size_t tileOrigin = rowCase
        ? static_cast<size_t>(b) * TILE * np + static_cast<size_t>(t) * TILE
        : static_cast<size_t>(t) * TILE * np + static_cast<size_t>(b) * TILE;
    const int tileBase = t * TILE;

    for (int r = ty; r < TILE; r += THREADS_Y) {
        pivotCol[r][tx] = pivotSnap[r * TILE + tx];
        pivotRow[r][tx] = pivotSnap[TILE * TILE + r * TILE + tx];
        tile[r][tx] = dist[tileOrigin + static_cast<size_t>(r) * np + tx];
    }
    __syncthreads();

    for (int kk = 0; kk < TILE; ++kk) {
        for (int r = ty; r < TILE; r += THREADS_Y) {
            unsigned int distIK;
            unsigned int distKJ;
            if (rowCase) {
                distIK = pivotCol[r][kk];
                distKJ = tile[kk][tx];
                if (r == kk) {
                    rowSnap[kk * np + tileBase + tx] = tile[kk][tx];
                }
            } else {
                distIK = tile[r][kk];
                distKJ = pivotRow[kk][tx];
                if (tx == kk) {
                    colSnap[(tileBase + r) * TILE + kk] = tile[r][kk];
                }
            }

            const unsigned int newDist = distIK + distKJ;
            if (newDist < tile[r][tx]) {
                tile[r][tx] = newDist;
                path[tileOrigin + static_cast<size_t>(r) * np + tx] = b * TILE + kk;
            }
        }
        __syncthreads();
    }

    for (int r = ty; r < TILE; r += THREADS_Y) {
        dist[tileOrigin + static_cast<size_t>(r) * np + tx] = tile[r][tx];
    }
}

// Phase 3: relax all remaining tiles (i, j) with i != b and j != b.
// Each tile reads the step-wise pivot snapshots from shared memory, and
// keeps its own values in registers across the k loop.
__global__ void fwPhase3(unsigned int* dist, unsigned int* path, const unsigned int* colSnap,
                         const unsigned int* rowSnap, const int np, const int b) {
    __shared__ unsigned int colTile[TILE][TILE];   // colTile[r][kk]: D[i-block r][b-block kk] at step kk
    __shared__ unsigned int rowTile[TILE][TILE];   // rowTile[kk][c]: D[b-block kk][j-block c] at step kk

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int j = blockIdx.x < b ? blockIdx.x : blockIdx.x + 1;
    const int i = blockIdx.y < b ? blockIdx.y : blockIdx.y + 1;

    const size_t cOrigin = static_cast<size_t>(i) * TILE * np + static_cast<size_t>(j) * TILE;

    unsigned int cur[ROWS_PER_THREAD];
    unsigned int curPath[ROWS_PER_THREAD];
    for (int q = 0; q < ROWS_PER_THREAD; ++q) {
        const int r = ty + q * THREADS_Y;
        cur[q] = dist[cOrigin + static_cast<size_t>(r) * np + tx];
        curPath[q] = path[cOrigin + static_cast<size_t>(r) * np + tx];
    }
    for (int r = ty; r < TILE; r += THREADS_Y) {
        colTile[r][tx] = colSnap[(i * TILE + r) * TILE + tx];
        rowTile[r][tx] = rowSnap[r * np + j * TILE + tx];
    }
    __syncthreads();

    for (int kk = 0; kk < TILE; ++kk) {
        for (int q = 0; q < ROWS_PER_THREAD; ++q) {
            const int r = ty + q * THREADS_Y;
            const unsigned int newDist = colTile[r][kk] + rowTile[kk][tx];
            if (newDist < cur[q]) {
                cur[q] = newDist;
                curPath[q] = b * TILE + kk;
            }
        }
    }

    for (int q = 0; q < ROWS_PER_THREAD; ++q) {
        const int r = ty + q * THREADS_Y;
        dist[cOrigin + static_cast<size_t>(r) * np + tx] = cur[q];
        path[cOrigin + static_cast<size_t>(r) * np + tx] = curPath[q];
    }
}

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes) {
    if (numNodes == 0) {
        return;
    }

    // Pad to a whole number of tiles. Padding nodes have INF distance to every
    // other node and zero distance to themselves, so no real path can go through them.
    const size_t numTiles = (numNodes + TILE - 1) / TILE;
    const size_t paddedN = numTiles * TILE;

    std::vector<unsigned int> hostDist(paddedN * paddedN, INF);
    std::vector<unsigned int> hostPath(paddedN * paddedN, 0);
    for (size_t i = 0; i < paddedN; ++i) {
        hostDist[i * paddedN + i] = 0;
    }
    for (size_t i = 0; i < numNodes; ++i) {
        std::copy_n(&dist[i * numNodes], numNodes, &hostDist[i * paddedN]);
        std::copy_n(&path[i * numNodes], numNodes, &hostPath[i * paddedN]);
    }

    const size_t bytes = paddedN * paddedN * sizeof(unsigned int);
    unsigned int* dDist = nullptr;
    unsigned int* dPath = nullptr;
    unsigned int* dPivotSnap = nullptr;
    unsigned int* dColSnap = nullptr;
    unsigned int* dRowSnap = nullptr;
    CUDA_CHECK(cudaMalloc(&dDist, bytes));
    CUDA_CHECK(cudaMalloc(&dPath, bytes));
    CUDA_CHECK(cudaMalloc(&dPivotSnap, 2 * TILE * TILE * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&dColSnap, paddedN * TILE * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&dRowSnap, TILE * paddedN * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpy(dDist, hostDist.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dPath, hostPath.data(), bytes, cudaMemcpyHostToDevice));

    const int np = static_cast<int>(paddedN);
    const int nb = static_cast<int>(numTiles);
    const dim3 block(THREADS_X, THREADS_Y);

    // Rounds over pivot blocks; each round must finish before the next starts
    for (int b = 0; b < nb; ++b) {
        fwPhase1<<<1, block>>>(dDist, dPath, dPivotSnap, np, b);
        CUDA_CHECK(cudaGetLastError());
        if (nb > 1) {
            fwPhase2<<<dim3(nb - 1, 2), block>>>(dDist, dPath, dPivotSnap, dColSnap, dRowSnap, np, b);
            CUDA_CHECK(cudaGetLastError());
            fwPhase3<<<dim3(nb - 1, nb - 1), block>>>(dDist, dPath, dColSnap, dRowSnap, np, b);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    CUDA_CHECK(cudaMemcpy(hostDist.data(), dDist, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hostPath.data(), dPath, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dDist));
    CUDA_CHECK(cudaFree(dPath));
    CUDA_CHECK(cudaFree(dPivotSnap));
    CUDA_CHECK(cudaFree(dColSnap));
    CUDA_CHECK(cudaFree(dRowSnap));

    for (size_t i = 0; i < numNodes; ++i) {
        std::copy_n(&hostDist[i * paddedN], numNodes, &dist[i * numNodes]);
        std::copy_n(&hostPath[i * paddedN], numNodes, &path[i * numNodes]);
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

// Create the CUDA context up front so driver start-up cost is not part of the timed computation
void initCudaDevice() {
    CUDA_CHECK(cudaFree(nullptr));
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

    initCudaDevice();

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
