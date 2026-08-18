#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

// A 32x32 tile is processed by 16x16 threads, with each thread owning a 2x2
// group of output elements.  This keeps the three shared tiles used in phase 3
// small while providing substantially more parallelism than one thread per row.
constexpr unsigned int TILE_SIZE = 32;
constexpr unsigned int THREAD_TILE_SIZE = 2;
constexpr unsigned int THREADS_PER_DIM = TILE_SIZE / THREAD_TILE_SIZE;
constexpr unsigned int SHARED_STRIDE = TILE_SIZE + 1;  // Avoid shared-memory bank conflicts.

#define CUDA_CHECK(call)                                                                    \
    do {                                                                                    \
        const cudaError_t cudaStatus = (call);                                             \
        if (cudaStatus != cudaSuccess) {                                                   \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,        \
                         cudaGetErrorString(cudaStatus));                                  \
            std::exit(EXIT_FAILURE);                                                       \
        }                                                                                   \
    } while (false)

// Index calculation for flattened 2D array.  The first argument is the column
// and the second is the row, so the resulting storage is row-major.
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

__device__ __forceinline__ void relax(unsigned int& current,
                                      unsigned int& currentPath,
                                      const unsigned int left,
                                      const unsigned int right,
                                      const unsigned int k) {
    const unsigned int candidate = left + right;
    if (candidate < current) {
        current = candidate;
        currentPath = k;
    }
}

// Phase 1: complete one diagonal (pivot) tile.  A synchronization after every
// intermediate vertex preserves the dependency between successive k values.
__global__ void floydWarshallPhase1(unsigned int* __restrict__ dist,
                                    unsigned int* __restrict__ path,
                                    const unsigned int numNodes,
                                    const unsigned int pivotBase) {
    __shared__ unsigned int pivot[TILE_SIZE][SHARED_STRIDE];

    const unsigned int r0 = threadIdx.y;
    const unsigned int r1 = r0 + THREADS_PER_DIM;
    const unsigned int c0 = threadIdx.x;
    const unsigned int c1 = c0 + THREADS_PER_DIM;
    const unsigned int gR0 = pivotBase + r0;
    const unsigned int gR1 = pivotBase + r1;
    const unsigned int gC0 = pivotBase + c0;
    const unsigned int gC1 = pivotBase + c1;

    const bool valid00 = gR0 < numNodes && gC0 < numNodes;
    const bool valid01 = gR0 < numNodes && gC1 < numNodes;
    const bool valid10 = gR1 < numNodes && gC0 < numNodes;
    const bool valid11 = gR1 < numNodes && gC1 < numNodes;

    unsigned int path00 = valid00 ? path[gR0 * numNodes + gC0] : 0;
    unsigned int path01 = valid01 ? path[gR0 * numNodes + gC1] : 0;
    unsigned int path10 = valid10 ? path[gR1 * numNodes + gC0] : 0;
    unsigned int path11 = valid11 ? path[gR1 * numNodes + gC1] : 0;

    pivot[r0][c0] = valid00 ? dist[gR0 * numNodes + gC0] : INF;
    pivot[r0][c1] = valid01 ? dist[gR0 * numNodes + gC1] : INF;
    pivot[r1][c0] = valid10 ? dist[gR1 * numNodes + gC0] : INF;
    pivot[r1][c1] = valid11 ? dist[gR1 * numNodes + gC1] : INF;
    __syncthreads();

    const unsigned int pivotExtent = min(TILE_SIZE, numNodes - pivotBase);
    for (unsigned int localK = 0; localK < pivotExtent; ++localK) {
        const unsigned int globalK = pivotBase + localK;
        relax(pivot[r0][c0], path00, pivot[r0][localK], pivot[localK][c0], globalK);
        relax(pivot[r0][c1], path01, pivot[r0][localK], pivot[localK][c1], globalK);
        relax(pivot[r1][c0], path10, pivot[r1][localK], pivot[localK][c0], globalK);
        relax(pivot[r1][c1], path11, pivot[r1][localK], pivot[localK][c1], globalK);
        __syncthreads();
    }

    if (valid00) {
        dist[gR0 * numNodes + gC0] = pivot[r0][c0];
        path[gR0 * numNodes + gC0] = path00;
    }
    if (valid01) {
        dist[gR0 * numNodes + gC1] = pivot[r0][c1];
        path[gR0 * numNodes + gC1] = path01;
    }
    if (valid10) {
        dist[gR1 * numNodes + gC0] = pivot[r1][c0];
        path[gR1 * numNodes + gC0] = path10;
    }
    if (valid11) {
        dist[gR1 * numNodes + gC1] = pivot[r1][c1];
        path[gR1 * numNodes + gC1] = path11;
    }
}

// Phase 2: update the pivot tile's block row and block column.  grid.y selects
// row (0) or column (1), and the grid excludes the diagonal tile itself.
__global__ void floydWarshallPhase2(unsigned int* __restrict__ dist,
                                    unsigned int* __restrict__ path,
                                    const unsigned int numNodes,
                                    const unsigned int pivotBase,
                                    const unsigned int pivotTile) {
    __shared__ unsigned int pivot[TILE_SIZE][SHARED_STRIDE];
    __shared__ unsigned int tile[TILE_SIZE][SHARED_STRIDE];

    const unsigned int compactTargetTile = blockIdx.x;
    const unsigned int targetTile = compactTargetTile + (compactTargetTile >= pivotTile);
    const unsigned int targetBase = targetTile * TILE_SIZE;
    const bool updateRow = blockIdx.y == 0;

    const unsigned int r0 = threadIdx.y;
    const unsigned int r1 = r0 + THREADS_PER_DIM;
    const unsigned int c0 = threadIdx.x;
    const unsigned int c1 = c0 + THREADS_PER_DIM;

    const unsigned int pR0 = pivotBase + r0;
    const unsigned int pR1 = pivotBase + r1;
    const unsigned int pC0 = pivotBase + c0;
    const unsigned int pC1 = pivotBase + c1;

    const unsigned int tR0 = updateRow ? pR0 : targetBase + r0;
    const unsigned int tR1 = updateRow ? pR1 : targetBase + r1;
    const unsigned int tC0 = updateRow ? targetBase + c0 : pC0;
    const unsigned int tC1 = updateRow ? targetBase + c1 : pC1;

    const bool valid00 = tR0 < numNodes && tC0 < numNodes;
    const bool valid01 = tR0 < numNodes && tC1 < numNodes;
    const bool valid10 = tR1 < numNodes && tC0 < numNodes;
    const bool valid11 = tR1 < numNodes && tC1 < numNodes;

    unsigned int path00 = valid00 ? path[tR0 * numNodes + tC0] : 0;
    unsigned int path01 = valid01 ? path[tR0 * numNodes + tC1] : 0;
    unsigned int path10 = valid10 ? path[tR1 * numNodes + tC0] : 0;
    unsigned int path11 = valid11 ? path[tR1 * numNodes + tC1] : 0;

    const bool pivot00 = pR0 < numNodes && pC0 < numNodes;
    const bool pivot01 = pR0 < numNodes && pC1 < numNodes;
    const bool pivot10 = pR1 < numNodes && pC0 < numNodes;
    const bool pivot11 = pR1 < numNodes && pC1 < numNodes;
    pivot[r0][c0] = pivot00 ? dist[pR0 * numNodes + pC0] : INF;
    pivot[r0][c1] = pivot01 ? dist[pR0 * numNodes + pC1] : INF;
    pivot[r1][c0] = pivot10 ? dist[pR1 * numNodes + pC0] : INF;
    pivot[r1][c1] = pivot11 ? dist[pR1 * numNodes + pC1] : INF;

    tile[r0][c0] = valid00 ? dist[tR0 * numNodes + tC0] : INF;
    tile[r0][c1] = valid01 ? dist[tR0 * numNodes + tC1] : INF;
    tile[r1][c0] = valid10 ? dist[tR1 * numNodes + tC0] : INF;
    tile[r1][c1] = valid11 ? dist[tR1 * numNodes + tC1] : INF;
    __syncthreads();

    const unsigned int pivotExtent = min(TILE_SIZE, numNodes - pivotBase);
    for (unsigned int localK = 0; localK < pivotExtent; ++localK) {
        const unsigned int globalK = pivotBase + localK;
        if (updateRow) {
            relax(tile[r0][c0], path00, pivot[r0][localK], tile[localK][c0], globalK);
            relax(tile[r0][c1], path01, pivot[r0][localK], tile[localK][c1], globalK);
            relax(tile[r1][c0], path10, pivot[r1][localK], tile[localK][c0], globalK);
            relax(tile[r1][c1], path11, pivot[r1][localK], tile[localK][c1], globalK);
        } else {
            relax(tile[r0][c0], path00, tile[r0][localK], pivot[localK][c0], globalK);
            relax(tile[r0][c1], path01, tile[r0][localK], pivot[localK][c1], globalK);
            relax(tile[r1][c0], path10, tile[r1][localK], pivot[localK][c0], globalK);
            relax(tile[r1][c1], path11, tile[r1][localK], pivot[localK][c1], globalK);
        }
        __syncthreads();
    }

    if (valid00) {
        dist[tR0 * numNodes + tC0] = tile[r0][c0];
        path[tR0 * numNodes + tC0] = path00;
    }
    if (valid01) {
        dist[tR0 * numNodes + tC1] = tile[r0][c1];
        path[tR0 * numNodes + tC1] = path01;
    }
    if (valid10) {
        dist[tR1 * numNodes + tC0] = tile[r1][c0];
        path[tR1 * numNodes + tC0] = path10;
    }
    if (valid11) {
        dist[tR1 * numNodes + tC1] = tile[r1][c1];
        path[tR1 * numNodes + tC1] = path11;
    }
}

// Phase 3: every non-pivot tile is independent after phase 2.  The current
// tile stays in registers; only the two read-only input tiles use shared memory.
__global__ void floydWarshallPhase3(unsigned int* __restrict__ dist,
                                    unsigned int* __restrict__ path,
                                    const unsigned int numNodes,
                                    const unsigned int pivotBase,
                                    const unsigned int pivotTile) {
    __shared__ unsigned int left[TILE_SIZE][SHARED_STRIDE];
    __shared__ unsigned int right[TILE_SIZE][SHARED_STRIDE];

    const unsigned int compactTileCol = blockIdx.x;
    const unsigned int compactTileRow = blockIdx.y;
    const unsigned int tileCol = compactTileCol + (compactTileCol >= pivotTile);
    const unsigned int tileRow = compactTileRow + (compactTileRow >= pivotTile);
    const unsigned int rowBase = tileRow * TILE_SIZE;
    const unsigned int colBase = tileCol * TILE_SIZE;

    const unsigned int r0 = threadIdx.y;
    const unsigned int r1 = r0 + THREADS_PER_DIM;
    const unsigned int c0 = threadIdx.x;
    const unsigned int c1 = c0 + THREADS_PER_DIM;
    const unsigned int gR0 = rowBase + r0;
    const unsigned int gR1 = rowBase + r1;
    const unsigned int gC0 = colBase + c0;
    const unsigned int gC1 = colBase + c1;
    const unsigned int pR0 = pivotBase + r0;
    const unsigned int pR1 = pivotBase + r1;
    const unsigned int pC0 = pivotBase + c0;
    const unsigned int pC1 = pivotBase + c1;

    const bool valid00 = gR0 < numNodes && gC0 < numNodes;
    const bool valid01 = gR0 < numNodes && gC1 < numNodes;
    const bool valid10 = gR1 < numNodes && gC0 < numNodes;
    const bool valid11 = gR1 < numNodes && gC1 < numNodes;

    unsigned int current00 = valid00 ? dist[gR0 * numNodes + gC0] : INF;
    unsigned int current01 = valid01 ? dist[gR0 * numNodes + gC1] : INF;
    unsigned int current10 = valid10 ? dist[gR1 * numNodes + gC0] : INF;
    unsigned int current11 = valid11 ? dist[gR1 * numNodes + gC1] : INF;
    unsigned int path00 = valid00 ? path[gR0 * numNodes + gC0] : 0;
    unsigned int path01 = valid01 ? path[gR0 * numNodes + gC1] : 0;
    unsigned int path10 = valid10 ? path[gR1 * numNodes + gC0] : 0;
    unsigned int path11 = valid11 ? path[gR1 * numNodes + gC1] : 0;

    const bool left00 = gR0 < numNodes && pC0 < numNodes;
    const bool left01 = gR0 < numNodes && pC1 < numNodes;
    const bool left10 = gR1 < numNodes && pC0 < numNodes;
    const bool left11 = gR1 < numNodes && pC1 < numNodes;
    left[r0][c0] = left00 ? dist[gR0 * numNodes + pC0] : INF;
    left[r0][c1] = left01 ? dist[gR0 * numNodes + pC1] : INF;
    left[r1][c0] = left10 ? dist[gR1 * numNodes + pC0] : INF;
    left[r1][c1] = left11 ? dist[gR1 * numNodes + pC1] : INF;

    const bool right00 = pR0 < numNodes && gC0 < numNodes;
    const bool right01 = pR0 < numNodes && gC1 < numNodes;
    const bool right10 = pR1 < numNodes && gC0 < numNodes;
    const bool right11 = pR1 < numNodes && gC1 < numNodes;
    right[r0][c0] = right00 ? dist[pR0 * numNodes + gC0] : INF;
    right[r0][c1] = right01 ? dist[pR0 * numNodes + gC1] : INF;
    right[r1][c0] = right10 ? dist[pR1 * numNodes + gC0] : INF;
    right[r1][c1] = right11 ? dist[pR1 * numNodes + gC1] : INF;
    __syncthreads();

    const unsigned int pivotExtent = min(TILE_SIZE, numNodes - pivotBase);
    for (unsigned int localK = 0; localK < pivotExtent; ++localK) {
        const unsigned int globalK = pivotBase + localK;
        const unsigned int left0 = left[r0][localK];
        const unsigned int left1 = left[r1][localK];
        const unsigned int right0 = right[localK][c0];
        const unsigned int right1 = right[localK][c1];
        relax(current00, path00, left0, right0, globalK);
        relax(current01, path01, left0, right1, globalK);
        relax(current10, path10, left1, right0, globalK);
        relax(current11, path11, left1, right1, globalK);
    }

    if (valid00) {
        dist[gR0 * numNodes + gC0] = current00;
        path[gR0 * numNodes + gC0] = path00;
    }
    if (valid01) {
        dist[gR0 * numNodes + gC1] = current01;
        path[gR0 * numNodes + gC1] = path01;
    }
    if (valid10) {
        dist[gR1 * numNodes + gC0] = current10;
        path[gR1 * numNodes + gC0] = path10;
    }
    if (valid11) {
        dist[gR1 * numNodes + gC1] = current11;
        path[gR1 * numNodes + gC1] = path11;
    }
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

// Input and output transfers are deliberately outside the CUDA event interval:
// the reported time measures the same compute phase timed by the original code.
float floydWarshall(std::vector<unsigned int>& dist,
                    std::vector<unsigned int>& path,
                    const size_t numNodes) {
    if (numNodes == 0) {
        return 0.0F;
    }
    if (numNodes > std::numeric_limits<unsigned int>::max()) {
        std::fprintf(stderr, "Number of nodes is too large for the CUDA implementation\n");
        std::exit(EXIT_FAILURE);
    }

    const unsigned int n = static_cast<unsigned int>(numNodes);
    const size_t bytes = numNodes * numNodes * sizeof(unsigned int);
    unsigned int* deviceDist = nullptr;
    unsigned int* devicePath = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    CUDA_CHECK(cudaMalloc(&deviceDist, bytes));
    CUDA_CHECK(cudaMalloc(&devicePath, bytes));
    CUDA_CHECK(cudaMemcpy(deviceDist, dist.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(devicePath, path.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const unsigned int numTiles = (n + TILE_SIZE - 1) / TILE_SIZE;
    const dim3 threads(THREADS_PER_DIM, THREADS_PER_DIM);
    CUDA_CHECK(cudaEventRecord(start));
    for (unsigned int pivotTile = 0; pivotTile < numTiles; ++pivotTile) {
        const unsigned int pivotBase = pivotTile * TILE_SIZE;
        floydWarshallPhase1<<<1, threads>>>(deviceDist, devicePath, n, pivotBase);

        if (numTiles > 1) {
            const dim3 phase2Blocks(numTiles - 1, 2);
            floydWarshallPhase2<<<phase2Blocks, threads>>>(
                deviceDist, devicePath, n, pivotBase, pivotTile);

            const dim3 phase3Blocks(numTiles - 1, numTiles - 1);
            floydWarshallPhase3<<<phase3Blocks, threads>>>(
                deviceDist, devicePath, n, pivotBase, pivotTile);
        }
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsedMilliseconds = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedMilliseconds, start, stop));
    CUDA_CHECK(cudaMemcpy(dist.data(), deviceDist, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(path.data(), devicePath, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(deviceDist));
    CUDA_CHECK(cudaFree(devicePath));
    return elapsedMilliseconds;
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

    // Run Floyd-Warshall
    printf("Computing shortest paths...\n");
    const long durationMilliseconds = static_cast<long>(floydWarshall(dist, path, numNodes));

    printf("Computation time: %ld ms\n", durationMilliseconds);

    // Calculate operations per second
    // Floyd-Warshall has O(n^3) complexity
    const double ops = (double)numNodes * numNodes * numNodes;
    const double gflops = ops / (durationMilliseconds / 1000.0) / 1e9;
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
