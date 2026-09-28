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

// Tiling parameters for the blocked GPU Floyd-Warshall.
// A thread block is TILE x BLOCK_ROWS threads and each thread owns
// ROWS_PER_THREAD cells of a TILE x TILE tile.
constexpr int TILE = 32;
constexpr int BLOCK_ROWS = 8;                       // threadIdx.y extent
constexpr int ROWS_PER_THREAD = TILE / BLOCK_ROWS;  // result cells per thread

// The hot loop of phase 3 keeps the predecessor of a cell in the low bits of
// the distance itself, so a single unsigned minimum updates both. Distances are
// therefore held shifted left by PACK_BITS on the device, which requires
// 2 * (largest distance) to stay below 2^(32 - PACK_BITS). PACK_INF, the
// padding value, is itself a distance that may be doubled, so the bound is one
// bit lower still: every real distance must stay below PACK_LIMIT == PACK_INF.
constexpr int PACK_BITS = 5;  // enough for the TILE possible predecessors
static_assert((1 << PACK_BITS) >= TILE, "PACK_BITS must cover a whole tile");
constexpr unsigned int PACK_LIMIT = 1u << (30 - PACK_BITS);
constexpr unsigned int PACK_INF = PACK_LIMIT;
constexpr unsigned int PATH_UNSET = 0xffffffffu;

// Index calculation for flattened 2D array
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

#define CUDA_CHECK(call)                                                                   \
    do {                                                                                   \
        const cudaError_t err__ = (call);                                                  \
        if (err__ != cudaSuccess) {                                                        \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__,                        \
                   cudaGetErrorString(err__));                                             \
            exit(1);                                                                       \
        }                                                                                  \
    } while (0)

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
// CUDA kernels: blocked Floyd-Warshall (three dependency phases per block step)
//
// The matrices are stored with row index i and column index j at element
// [i * n + j] (this is exactly idx2(j, i, n) of the original code). The device
// matrices are padded up to a multiple of TILE; padding rows/columns hold INF
// (with a zero diagonal) so they can never shorten any real path.
// ---------------------------------------------------------------------------

// Fill the padding of the device distance matrix (INF with a zero diagonal so
// that no shortest path can ever run through a padding node) and, in packed
// mode, shift the real entries into their packed representation.
__global__ void padScaleKernel(unsigned int* __restrict__ dist,
                               const int n, const int paddedN, const bool packed) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= paddedN || j >= paddedN) return;

    const size_t g = (size_t)i * paddedN + j;
    if (i < n && j < n) {
        if (packed) dist[g] <<= PACK_BITS;
    } else {
        const unsigned int inf = packed ? (PACK_INF << PACK_BITS) : INF;
        dist[g] = (i == j) ? 0u : inf;
    }
}

// Reports whether any input distance is too large for the packed representation.
__global__ void rangeCheckKernel(const unsigned int* __restrict__ dist,
                                 const size_t total, int* __restrict__ tooLarge) {
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t g = (size_t)blockIdx.x * blockDim.x + threadIdx.x; g < total; g += stride) {
        if (dist[g] >= PACK_LIMIT) {
            *tooLarge = 1;
            return;
        }
    }
}

// Undo the packing so the host sees the original distance scale again.
__global__ void unscaleKernel(unsigned int* __restrict__ dist, const size_t total) {
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t g = (size_t)blockIdx.x * blockDim.x + threadIdx.x; g < total; g += stride) {
        dist[g] >>= PACK_BITS;
    }
}

// Entries that no pivot ever improved keep their initial predecessor.
__global__ void mergePathKernel(unsigned int* __restrict__ path,
                                const unsigned int* __restrict__ pathInit,
                                const size_t total) {
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t g = (size_t)blockIdx.x * blockDim.x + threadIdx.x; g < total; g += stride) {
        if (path[g] == PATH_UNSET) path[g] = pathInit[g];
    }
}

// Phase 1: the diagonal tile (k, k) depends only on itself.
//
// In all three kernels every thread owns ROWS_PER_THREAD result cells
// exclusively, so the predecessors live in registers and are written back only
// for cells that really changed. Cells that are never improved keep the
// PATH_UNSET marker and are filled in by mergePathKernel() at the end, which is
// what lets the upload of the initial path matrix overlap with the computation.
__global__ void phase1Kernel(unsigned int* __restrict__ dist,
                             unsigned int* __restrict__ path,
                             const int paddedN, const int k) {
    __shared__ unsigned int sd[TILE][TILE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int base = k * TILE;
    const size_t g0 = (size_t)(base + ty) * paddedN + (base + tx);
    const size_t rowStride = (size_t)BLOCK_ROWS * paddedN;

    unsigned int p[ROWS_PER_THREAD];
    bool changed[ROWS_PER_THREAD];

#pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; ++r) {
        sd[ty + r * BLOCK_ROWS][tx] = dist[g0 + (size_t)r * rowStride];
        p[r] = 0;
        changed[r] = false;
    }
    __syncthreads();

    // sd[y][u] and sd[u][tx] cannot change during step u (the pivot diagonal is
    // zero), so one barrier per step is sufficient.
    for (int u = 0; u < TILE; ++u) {
#pragma unroll
        for (int r = 0; r < ROWS_PER_THREAD; ++r) {
            const int y = ty + r * BLOCK_ROWS;
            const unsigned int nd = sd[y][u] + sd[u][tx];
            if (nd < sd[y][tx]) {
                sd[y][tx] = nd;
                p[r] = (unsigned int)(base + u);
                changed[r] = true;
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; ++r) {
        if (changed[r]) {
            const size_t g = g0 + (size_t)r * rowStride;
            dist[g] = sd[ty + r * BLOCK_ROWS][tx];
            path[g] = p[r];
        }
    }
}

// Phase 2: the tiles sharing the pivot block row / block column.
// blockIdx.y == 0 -> row tiles (k, b);  blockIdx.y == 1 -> column tiles (b, k).
__global__ void phase2Kernel(unsigned int* __restrict__ dist,
                             unsigned int* __restrict__ path,
                             const int paddedN, const int k) {
    const int b = blockIdx.x;
    if (b == k) return;  // that is the pivot tile, already done in phase 1

    __shared__ unsigned int pv[TILE][TILE];  // pivot tile (k, k)
    __shared__ unsigned int sd[TILE][TILE];  // tile being updated

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int base = k * TILE;
    const bool rowTile = (blockIdx.y == 0);
    const int tileRow = rowTile ? k : b;
    const int tileCol = rowTile ? b : k;
    const size_t g0 = (size_t)(tileRow * TILE + ty) * paddedN + (tileCol * TILE + tx);
    const size_t pv0 = (size_t)(base + ty) * paddedN + (base + tx);
    const size_t rowStride = (size_t)BLOCK_ROWS * paddedN;

    unsigned int p[ROWS_PER_THREAD];
    bool changed[ROWS_PER_THREAD];

#pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; ++r) {
        const int y = ty + r * BLOCK_ROWS;
        pv[y][tx] = dist[pv0 + (size_t)r * rowStride];
        sd[y][tx] = dist[g0 + (size_t)r * rowStride];
        p[r] = 0;
        changed[r] = false;
    }
    __syncthreads();

    // Row tiles read the pivot on the left, column tiles on the right; in both
    // cases the tile itself is part of the dependency, so a barrier per step.
    for (int u = 0; u < TILE; ++u) {
#pragma unroll
        for (int r = 0; r < ROWS_PER_THREAD; ++r) {
            const int y = ty + r * BLOCK_ROWS;
            const unsigned int nd = rowTile ? (pv[y][u] + sd[u][tx])
                                            : (sd[y][u] + pv[u][tx]);
            if (nd < sd[y][tx]) {
                sd[y][tx] = nd;
                p[r] = (unsigned int)(base + u);
                changed[r] = true;
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; ++r) {
        if (changed[r]) {
            const size_t g = g0 + (size_t)r * rowStride;
            dist[g] = sd[ty + r * BLOCK_ROWS][tx];
            path[g] = p[r];
        }
    }
}

// Phase 3: all remaining tiles. They only depend on the phase-2 tiles of this
// step, so the reduction over the pivot index needs no barriers at all.
//
// Packed mode carries the pivot offset u in the low PACK_BITS of the row tile
// values. Since every distance has zero low bits, the unsigned minimum of the
// candidates yields the smallest distance and, among equal distances, the
// smallest u - exactly the pivot the serial code would have recorded last (the
// first one that reached the final value). That turns the whole "compare, keep
// distance, keep predecessor" body into one add and one minimum.
template <bool Packed>
__global__ void phase3Kernel(unsigned int* __restrict__ dist,
                             unsigned int* __restrict__ path,
                             const int paddedN, const int k) {
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    if (bx == k || by == k) return;

    __shared__ unsigned int rowS[TILE][TILE];  // tile (k, bx)
    __shared__ unsigned int colS[TILE][TILE];  // tile (by, k)

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int base = k * TILE;
    const size_t rowStride = (size_t)BLOCK_ROWS * paddedN;
    const size_t g0 = (size_t)(by * TILE + ty) * paddedN + (bx * TILE + tx);

    unsigned int d[ROWS_PER_THREAD];
    unsigned int orig[ROWS_PER_THREAD];
    unsigned int p[ROWS_PER_THREAD];
    bool changed[ROWS_PER_THREAD];

#pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; ++r) {
        const int y = ty + r * BLOCK_ROWS;
        const unsigned int rowVal = dist[(size_t)(base + y) * paddedN + (bx * TILE + tx)];
        rowS[y][tx] = Packed ? (rowVal + (unsigned int)y) : rowVal;
        colS[y][tx] = dist[(size_t)(by * TILE + y) * paddedN + (base + tx)];
        d[r] = dist[g0 + (size_t)r * rowStride];
        orig[r] = d[r];
        p[r] = 0;
        changed[r] = false;
    }
    __syncthreads();

    // The outputs of this thread live in registers and u is the outer loop, so
    // the row-tile value rowS[u][tx] is read once per u instead of once per
    // (u, row): that halves the shared-memory traffic of the hot loop.
    for (int u = 0; u < TILE; ++u) {
        const unsigned int rw = rowS[u][tx];
#pragma unroll
        for (int r = 0; r < ROWS_PER_THREAD; ++r) {
            const unsigned int nd = colS[ty + r * BLOCK_ROWS][u] + rw;
            if (Packed) {
                d[r] = nd < d[r] ? nd : d[r];
            } else if (nd < d[r]) {
                d[r] = nd;
                p[r] = (unsigned int)(base + u);
                changed[r] = true;
            }
        }
    }

#pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; ++r) {
        unsigned int value = d[r];
        unsigned int pred = p[r];
        bool store = changed[r];
        if (Packed) {
            value = d[r] & ~((1u << PACK_BITS) - 1u);
            pred = (unsigned int)base + (d[r] & ((1u << PACK_BITS) - 1u));
            store = value < orig[r];
        }
        if (store) {
            const size_t g = g0 + (size_t)r * rowStride;
            dist[g] = value;
            path[g] = pred;
        }
    }
}

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes) {
    if (numNodes == 0) return;

    const int n = (int)numNodes;
    const int numTiles = (n + TILE - 1) / TILE;
    const int paddedN = numTiles * TILE;
    const size_t paddedElems = (size_t)paddedN * paddedN;
    const size_t bytes = paddedElems * sizeof(unsigned int);

    unsigned int* dDist = nullptr;
    unsigned int* dPath = nullptr;
    unsigned int* dPathInit = nullptr;
    int* dFlag = nullptr;
    CUDA_CHECK(cudaMalloc(&dDist, bytes));
    CUDA_CHECK(cudaMalloc(&dPath, bytes));
    CUDA_CHECK(cudaMalloc(&dPathInit, bytes));
    CUDA_CHECK(cudaMalloc(&dFlag, sizeof(int)));

    // Every cell starts out "not improved by any pivot"; see mergePathKernel().
    CUDA_CHECK(cudaMemsetAsync(dPath, 0xff, bytes));
    CUDA_CHECK(cudaMemsetAsync(dFlag, 0, sizeof(int)));
    // Touching the other two allocations here is worth it: the first write to a
    // fresh allocation maps it in, and paying that on the (fast) device side
    // keeps it out of the host transfers below.
    CUDA_CHECK(cudaMemsetAsync(dDist, 0, bytes));
    CUDA_CHECK(cudaMemsetAsync(dPathInit, 0, bytes));

    const size_t hostPitch = (size_t)n * sizeof(unsigned int);
    const size_t devPitch = (size_t)paddedN * sizeof(unsigned int);
    CUDA_CHECK(cudaMemcpy2DAsync(dDist, devPitch, dist.data(), hostPitch,
                                 hostPitch, (size_t)n, cudaMemcpyHostToDevice));

    // The packed phase-3 kernel needs headroom in the top bits of a distance;
    // oversized inputs fall back to the plain kernel.
    rangeCheckKernel<<<256, 256>>>(dDist, paddedElems, dFlag);
    CUDA_CHECK(cudaGetLastError());
    int tooLarge = 0;
    CUDA_CHECK(cudaMemcpy(&tooLarge, dFlag, sizeof(int), cudaMemcpyDeviceToHost));
    const bool packed = (tooLarge == 0);

    {
        const dim3 padBlock(32, 8);
        const dim3 padGrid((paddedN + padBlock.x - 1) / padBlock.x,
                           (paddedN + padBlock.y - 1) / padBlock.y);
        padScaleKernel<<<padGrid, padBlock>>>(dDist, n, paddedN, packed);
        CUDA_CHECK(cudaGetLastError());
    }

    const dim3 threads(TILE, BLOCK_ROWS);
    const dim3 grid2(numTiles, 2);
    const dim3 grid3(numTiles, numTiles);

    for (int k = 0; k < numTiles; ++k) {
        phase1Kernel<<<1, threads>>>(dDist, dPath, paddedN, k);
        if (numTiles > 1) {
            phase2Kernel<<<grid2, threads>>>(dDist, dPath, paddedN, k);
            if (packed) {
                phase3Kernel<true><<<grid3, threads>>>(dDist, dPath, paddedN, k);
            } else {
                phase3Kernel<false><<<grid3, threads>>>(dDist, dPath, paddedN, k);
            }
        }
    }
    CUDA_CHECK(cudaGetLastError());

    // While the GPU works through the pivots, upload the initial path matrix -
    // no kernel reads it, it is only needed by the final merge. The upload runs
    // on a non-blocking stream so that it neither waits for the queued kernels
    // nor holds them up.
    {
        cudaStream_t uploadStream;
        CUDA_CHECK(cudaStreamCreateWithFlags(&uploadStream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaMemcpy2DAsync(dPathInit, devPitch, path.data(), hostPitch,
                                     hostPitch, (size_t)n, cudaMemcpyHostToDevice,
                                     uploadStream));
        CUDA_CHECK(cudaStreamSynchronize(uploadStream));
        CUDA_CHECK(cudaStreamDestroy(uploadStream));
    }

    const int mergeBlocks = (int)std::min((size_t)4096, (paddedElems + 255) / 256);
    mergePathKernel<<<mergeBlocks, 256>>>(dPath, dPathInit, paddedElems);
    CUDA_CHECK(cudaGetLastError());
    if (packed) {
        unscaleKernel<<<mergeBlocks, 256>>>(dDist, paddedElems);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy2D(dist.data(), hostPitch, dDist, devPitch,
                            hostPitch, (size_t)n, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy2D(path.data(), hostPitch, dPath, devPitch,
                            hostPitch, (size_t)n, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(dFlag));
    CUDA_CHECK(cudaFree(dPathInit));
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

    // Allocate matrices
    std::vector<unsigned int> dist(numNodes * numNodes);
    std::vector<unsigned int> path(numNodes * numNodes);

    // Initialize
    printf("Initializing graph...\n");
    initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
    initializePathMatrix(path, numNodes);

    // Create the CUDA context up front so its one-time cost is not timed.
    CUDA_CHECK(cudaFree(0));

    // Run Floyd-Warshall
    printf("Computing shortest paths...\n");
    auto start = std::chrono::high_resolution_clock::now();

    floydWarshall(dist, path, numNodes);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    const double seconds = std::chrono::duration<double>(end - start).count();

    printf("Computation time: %ld ms\n", duration.count());

    // Calculate operations per second
    // Floyd-Warshall has O(n³) complexity
    double ops = (double)numNodes * numNodes * numNodes;
    double gflops = ops / seconds / 1e9;
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
