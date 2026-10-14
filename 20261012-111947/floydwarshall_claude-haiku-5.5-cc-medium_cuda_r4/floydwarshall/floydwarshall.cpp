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

// Edge length of a square tile. Each tile is handled by one thread block of
// TILE x TILE threads, one thread per matrix cell.
constexpr size_t TILE = 32;

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err_ = (call);                                           \
        if (err_ != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err_));                               \
            exit(1);                                                         \
        }                                                                    \
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

// Phase 1: the diagonal tile (K,K). Runs the k-steps of round K inside the tile.
// Before each k-step it saves the diagonal's column k and row k so that the
// other tiles of this round can read the values d[.][k] and d[k][.] exactly as
// they were before step k.
__global__ void __launch_bounds__(TILE * TILE)
fwPhase1(unsigned int* dist, unsigned int* path, unsigned int* colSnap, unsigned int* rowSnap,
         const size_t m, const size_t K) {
    __shared__ unsigned int tDist[TILE][TILE];
    __shared__ unsigned int tPath[TILE][TILE];

    const size_t ty = threadIdx.y;
    const size_t tx = threadIdx.x;
    const size_t base = K * TILE;
    const size_t gIdx = (base + ty) * m + base + tx;

    tDist[ty][tx] = dist[gIdx];
    tPath[ty][tx] = path[gIdx];
    __syncthreads();

    for (size_t kk = 0; kk < TILE; ++kk) {
        const unsigned int distIK = tDist[ty][kk];
        const unsigned int distKJ = tDist[kk][tx];
        if (tx == 0) colSnap[(base + ty) * TILE + kk] = distIK;
        if (ty == 0) rowSnap[kk * m + base + tx] = distKJ;
        __syncthreads();

        const unsigned int newDist = distIK + distKJ;
        if (newDist < tDist[ty][tx]) {
            tDist[ty][tx] = newDist;
            tPath[ty][tx] = static_cast<unsigned int>(base + kk);
        }
        __syncthreads();
    }

    dist[gIdx] = tDist[ty][tx];
    path[gIdx] = tPath[ty][tx];
}

// Phase 2: tiles (K,J) for J != K (blockIdx.y == 0) and tiles (I,K) for I != K
// (blockIdx.y == 1), blockIdx.x holds the other block index. Each of these tiles
// reads one operand from the diagonal snapshot (fixed for the round) and the
// other operand from its own, sequentially updated, contents.
__global__ void __launch_bounds__(TILE * TILE)
fwPhase2(unsigned int* dist, unsigned int* path, unsigned int* colSnap, unsigned int* rowSnap,
         const size_t m, const size_t K) {
    const size_t t = blockIdx.x;
    if (t == K) return;
    const bool isRowTile = (blockIdx.y == 0);

    __shared__ unsigned int tDist[TILE][TILE];
    __shared__ unsigned int tPath[TILE][TILE];
    __shared__ unsigned int tFixed[TILE][TILE];

    const size_t ty = threadIdx.y;
    const size_t tx = threadIdx.x;
    const size_t rowBase = isRowTile ? K * TILE : t * TILE;
    const size_t colBase = isRowTile ? t * TILE : K * TILE;
    const size_t gIdx = (rowBase + ty) * m + colBase + tx;

    tDist[ty][tx] = dist[gIdx];
    tPath[ty][tx] = path[gIdx];
    // Row tile: d[i][k] for the rows of block K. Column tile: d[k][j] for the columns of block K.
    tFixed[ty][tx] = isRowTile ? colSnap[(rowBase + ty) * TILE + tx]
                               : rowSnap[ty * m + colBase + tx];
    __syncthreads();

    for (size_t kk = 0; kk < TILE; ++kk) {
        unsigned int distIK;
        unsigned int distKJ;
        if (isRowTile) {
            distIK = tFixed[ty][kk];
            distKJ = tDist[kk][tx];
            if (ty == 0) rowSnap[kk * m + colBase + tx] = distKJ;
        } else {
            distIK = tDist[ty][kk];
            distKJ = tFixed[kk][tx];
            if (tx == 0) colSnap[(rowBase + ty) * TILE + kk] = distIK;
        }
        __syncthreads();

        const unsigned int newDist = distIK + distKJ;
        if (newDist < tDist[ty][tx]) {
            tDist[ty][tx] = newDist;
            tPath[ty][tx] = static_cast<unsigned int>(K * TILE + kk);
        }
        __syncthreads();
    }

    dist[gIdx] = tDist[ty][tx];
    path[gIdx] = tPath[ty][tx];
}

// Phase 3: all remaining tiles (I,J) with I != K and J != K. Both operands come
// from the snapshots written in phases 1 and 2, so the k-steps are independent
// of each other and only the cell's own value changes between steps.
__global__ void __launch_bounds__(TILE * TILE)
fwPhase3(unsigned int* dist, unsigned int* path, const unsigned int* colSnap,
         const unsigned int* rowSnap, const size_t m, const size_t K) {
    const size_t J = blockIdx.x;
    const size_t I = blockIdx.y;
    if (I == K || J == K) return;

    __shared__ unsigned int colOp[TILE][TILE];  // colOp[i][kk] = d[I*TILE+i][k]
    __shared__ unsigned int rowOp[TILE][TILE];  // rowOp[kk][j] = d[k][J*TILE+j]

    const size_t ty = threadIdx.y;
    const size_t tx = threadIdx.x;
    const size_t gRow = I * TILE + ty;
    const size_t gCol = J * TILE + tx;
    const size_t gIdx = gRow * m + gCol;

    unsigned int curDist = dist[gIdx];
    unsigned int curPath = path[gIdx];
    colOp[ty][tx] = colSnap[gRow * TILE + tx];
    rowOp[ty][tx] = rowSnap[ty * m + gCol];
    __syncthreads();

#pragma unroll
    for (size_t kk = 0; kk < TILE; ++kk) {
        const unsigned int newDist = colOp[ty][kk] + rowOp[kk][tx];
        if (newDist < curDist) {
            curDist = newDist;
            curPath = static_cast<unsigned int>(K * TILE + kk);
        }
    }

    dist[gIdx] = curDist;
    path[gIdx] = curPath;
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

// Runs Floyd-Warshall on the GPU as a blocked computation. The matrices are
// padded to a multiple of TILE. Padding nodes have distance 0 to themselves and
// INF to every other node. Their k-steps come after all real k-steps and can
// never shorten a real distance, so the real results are unchanged and the
// padding is dropped when copying back.
void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes) {
    const size_t tiles = (numNodes + TILE - 1) / TILE;
    const size_t m = tiles * TILE;

    std::vector<unsigned int> hostDist(m * m, INF);
    std::vector<unsigned int> hostPath(m * m, 0);
    for (size_t p = 0; p < m; ++p) {
        hostDist[p * m + p] = 0;
    }
    for (size_t i = 0; i < numNodes; ++i) {
        for (size_t j = 0; j < numNodes; ++j) {
            hostDist[i * m + j] = dist[i * numNodes + j];
            hostPath[i * m + j] = path[i * numNodes + j];
        }
    }

    unsigned int *dDist = nullptr, *dPath = nullptr, *dColSnap = nullptr, *dRowSnap = nullptr;
    CUDA_CHECK(cudaMalloc(&dDist, m * m * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&dPath, m * m * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&dColSnap, m * TILE * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&dRowSnap, TILE * m * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpy(dDist, hostDist.data(), m * m * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dPath, hostPath.data(), m * m * sizeof(unsigned int), cudaMemcpyHostToDevice));

    const dim3 block(TILE, TILE);
    const dim3 phase2Grid(tiles, 2);
    const dim3 phase3Grid(tiles, tiles);

    // Each round K performs k-steps K*TILE .. K*TILE+TILE-1 on all tiles, in the
    // same k order as the sequential algorithm.
    for (size_t K = 0; K < tiles; ++K) {
        fwPhase1<<<1, block>>>(dDist, dPath, dColSnap, dRowSnap, m, K);
        fwPhase2<<<phase2Grid, block>>>(dDist, dPath, dColSnap, dRowSnap, m, K);
        fwPhase3<<<phase3Grid, block>>>(dDist, dPath, dColSnap, dRowSnap, m, K);
    }
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(hostDist.data(), dDist, m * m * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hostPath.data(), dPath, m * m * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(dDist));
    CUDA_CHECK(cudaFree(dPath));
    CUDA_CHECK(cudaFree(dColSnap));
    CUDA_CHECK(cudaFree(dRowSnap));

    for (size_t i = 0; i < numNodes; ++i) {
        for (size_t j = 0; j < numNodes; ++j) {
            dist[i * numNodes + j] = hostDist[i * m + j];
            path[i * numNodes + j] = hostPath[i * m + j];
        }
    }
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
    // Create the CUDA context before timing so driver startup is not counted
    CUDA_CHECK(cudaFree(nullptr));
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
