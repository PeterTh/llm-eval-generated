#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

// Tiled GPU Floyd-Warshall configuration.
// Each thread block is TILE x TILE_Y threads; every thread owns ROWS_PER_THREAD
// rows of a TILE x TILE tile.
constexpr int TILE = 32;
constexpr int TILE_Y = 8;
constexpr int ROWS_PER_THREAD = TILE / TILE_Y;

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err_));                                    \
            exit(EXIT_FAILURE);                                                   \
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
// GPU Floyd-Warshall (blocked / tiled formulation).
//
// Row-major storage: cell (i, j) lives at index i * n + j.
//
// The k-loop of the classic algorithm is split into rounds of TILE pivots.
// Each round K runs three kernels:
//   Phase 1: diagonal tile (K,K), sequential over its pivots.
//   Phase 2: tiles (K,t) (row panel) and (t,K) (column panel).
//   Phase 3: all remaining tiles (I,J), fully independent.
//
// Within a round the pivots are applied in increasing k, and every update uses
// exactly the values the sequential algorithm would see at that step. The
// values that the sequential algorithm reads from the pivot row and pivot
// column at step k are snapshotted into colBuf / rowBuf when they are current,
// so phase 3 can consume them without racing with later pivots. As a result the
// distance and path matrices are identical to the reference sequential code.
//
// colBuf[r * TILE + k] = D_{k-1}[r][kb + k]   (pivot column k, seen at step k)
// rowBuf[k * n + c]    = D_{k-1}[kb + k][c]   (pivot row k, seen at step k)
// ---------------------------------------------------------------------------

// Phase 1: diagonal tile (K,K).
__global__ void __launch_bounds__(TILE * TILE_Y)
fwPhase1(unsigned int* dist, unsigned int* path,
         unsigned int* colBuf, unsigned int* rowBuf,
         size_t n, size_t kb, int bs) {
    __shared__ unsigned int s[TILE][TILE];
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    if (tx < bs) {
        #pragma unroll
        for (int r = 0; r < ROWS_PER_THREAD; ++r) {
            const int i = ty + r * TILE_Y;
            if (i < bs) s[i][tx] = dist[(kb + i) * n + kb + tx];
        }
    }
    __syncthreads();

    for (int k = 0; k < bs; ++k) {
        #pragma unroll
        for (int r = 0; r < ROWS_PER_THREAD; ++r) {
            const int i = ty + r * TILE_Y;
            if (i < bs && tx == k) colBuf[(kb + i) * TILE + k] = s[i][k];
            if (i == k && tx < bs) rowBuf[k * n + kb + tx] = s[k][tx];
        }

        #pragma unroll
        for (int r = 0; r < ROWS_PER_THREAD; ++r) {
            const int i = ty + r * TILE_Y;
            if (i < bs && tx < bs) {
                const unsigned int cand = s[i][k] + s[k][tx];
                if (cand < s[i][tx]) {
                    s[i][tx] = cand;
                    path[(kb + i) * n + kb + tx] = kb + k;
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; ++r) {
        const int i = ty + r * TILE_Y;
        if (i < bs && tx < bs) dist[(kb + i) * n + kb + tx] = s[i][tx];
    }
}

// Phase 2: row panel tiles (K,t) and column panel tiles (t,K), t != K.
// blockIdx.y == 0 -> row tile, blockIdx.y == 1 -> column tile.
__global__ void __launch_bounds__(TILE * TILE_Y)
fwPhase2(unsigned int* dist, unsigned int* path,
         unsigned int* colBuf, unsigned int* rowBuf,
         size_t n, int K, size_t kb, int bs) {
    const int t = blockIdx.x;
    if (t == K) return;

    __shared__ unsigned int sC[TILE][TILE];
    __shared__ unsigned int sK[TILE][TILE];
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const size_t t0 = static_cast<size_t>(t) * TILE;

    if (blockIdx.y == 0) {
        // Row tile: rows are pivot-block rows (i < bs), columns are tile t.
        const int cl = static_cast<int>(min(static_cast<size_t>(TILE), n - t0));
        if (tx < cl) {
            #pragma unroll
            for (int r = 0; r < ROWS_PER_THREAD; ++r) {
                const int i = ty + r * TILE_Y;
                if (i < bs) sC[i][tx] = dist[(kb + i) * n + t0 + tx];
            }
        }
        #pragma unroll
        for (int r = 0; r < ROWS_PER_THREAD; ++r) {
            const int i = ty + r * TILE_Y;
            if (i < bs && tx < bs) sK[i][tx] = colBuf[(kb + i) * TILE + tx];
        }
        __syncthreads();

        for (int k = 0; k < bs; ++k) {
            #pragma unroll
            for (int r = 0; r < ROWS_PER_THREAD; ++r) {
                const int i = ty + r * TILE_Y;
                if (i == k && tx < cl) rowBuf[k * n + t0 + tx] = sC[k][tx];
            }

            #pragma unroll
            for (int r = 0; r < ROWS_PER_THREAD; ++r) {
                const int i = ty + r * TILE_Y;
                if (i < bs && tx < cl) {
                    const unsigned int cand = sK[i][k] + sC[k][tx];
                    if (cand < sC[i][tx]) {
                        sC[i][tx] = cand;
                        path[(kb + i) * n + t0 + tx] = kb + k;
                    }
                }
            }
            __syncthreads();
        }

        #pragma unroll
        for (int r = 0; r < ROWS_PER_THREAD; ++r) {
            const int i = ty + r * TILE_Y;
            if (i < bs && tx < cl) dist[(kb + i) * n + t0 + tx] = sC[i][tx];
        }
    } else {
        // Column tile: rows are tile t, columns are pivot-block columns (j < bs).
        const int rl = static_cast<int>(min(static_cast<size_t>(TILE), n - t0));
        #pragma unroll
        for (int r = 0; r < ROWS_PER_THREAD; ++r) {
            const int i = ty + r * TILE_Y;
            if (i < rl && tx < bs) sC[i][tx] = dist[(t0 + i) * n + kb + tx];
        }
        #pragma unroll
        for (int r = 0; r < ROWS_PER_THREAD; ++r) {
            const int k = ty + r * TILE_Y;
            if (k < bs && tx < bs) sK[k][tx] = rowBuf[k * n + kb + tx];
        }
        __syncthreads();

        for (int k = 0; k < bs; ++k) {
            #pragma unroll
            for (int r = 0; r < ROWS_PER_THREAD; ++r) {
                const int i = ty + r * TILE_Y;
                if (i < rl && tx == k) colBuf[(t0 + i) * TILE + k] = sC[i][k];
            }

            #pragma unroll
            for (int r = 0; r < ROWS_PER_THREAD; ++r) {
                const int i = ty + r * TILE_Y;
                if (i < rl && tx < bs) {
                    const unsigned int cand = sC[i][k] + sK[k][tx];
                    if (cand < sC[i][tx]) {
                        sC[i][tx] = cand;
                        path[(t0 + i) * n + kb + tx] = kb + k;
                    }
                }
            }
            __syncthreads();
        }

        #pragma unroll
        for (int r = 0; r < ROWS_PER_THREAD; ++r) {
            const int i = ty + r * TILE_Y;
            if (i < rl && tx < bs) dist[(t0 + i) * n + kb + tx] = sC[i][tx];
        }
    }
}

// Phase 3: all tiles (I,J) with I != K and J != K. Fully parallel.
__global__ void __launch_bounds__(TILE * TILE_Y)
fwPhase3(unsigned int* dist, unsigned int* path,
         const unsigned int* colBuf, const unsigned int* rowBuf,
         size_t n, int K, size_t kb, int bs) {
    const int J = blockIdx.x;
    const int I = blockIdx.y;
    if (I == K || J == K) return;

    __shared__ unsigned int sA[TILE][TILE];  // sA[i][k] = D_{k-1}[r0 + i][kb + k]
    __shared__ unsigned int sB[TILE][TILE];  // sB[k][j] = D_{k-1}[kb + k][c0 + j]
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const size_t r0 = static_cast<size_t>(I) * TILE;
    const size_t c0 = static_cast<size_t>(J) * TILE;
    const int rl = static_cast<int>(min(static_cast<size_t>(TILE), n - r0));
    const int cl = static_cast<int>(min(static_cast<size_t>(TILE), n - c0));

    #pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; ++r) {
        const int i = ty + r * TILE_Y;
        if (i < rl && tx < bs) sA[i][tx] = colBuf[(r0 + i) * TILE + tx];
        if (i < bs && tx < cl) sB[i][tx] = rowBuf[i * n + c0 + tx];
    }
    __syncthreads();

    unsigned int v[ROWS_PER_THREAD];
    unsigned int p[ROWS_PER_THREAD];
    #pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; ++r) {
        const int i = ty + r * TILE_Y;
        if (i < rl && tx < cl) {
            const size_t idx = (r0 + i) * n + c0 + tx;
            v[r] = dist[idx];
            p[r] = path[idx];
        } else {
            v[r] = 0;
            p[r] = 0;
        }
    }

    for (int k = 0; k < bs; ++k) {
        const unsigned int b = sB[k][tx];
        #pragma unroll
        for (int r = 0; r < ROWS_PER_THREAD; ++r) {
            const int i = ty + r * TILE_Y;
            const unsigned int cand = sA[i][k] + b;
            if (cand < v[r]) {
                v[r] = cand;
                p[r] = kb + k;
            }
        }
    }

    #pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; ++r) {
        const int i = ty + r * TILE_Y;
        if (i < rl && tx < cl) {
            const size_t idx = (r0 + i) * n + c0 + tx;
            dist[idx] = v[r];
            path[idx] = p[r];
        }
    }
}

// Runs blocked Floyd-Warshall on the GPU. dist and path are updated in place
// with the same results as the sequential algorithm.
void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes) {
    if (numNodes == 0) return;

    const size_t cells = numNodes * numNodes;
    const size_t bytes = cells * sizeof(unsigned int);
    const int numTiles = static_cast<int>((numNodes + TILE - 1) / TILE);

    unsigned int *dDist = nullptr, *dPath = nullptr, *dCol = nullptr, *dRow = nullptr;
    CUDA_CHECK(cudaMalloc(&dDist, bytes));
    CUDA_CHECK(cudaMalloc(&dPath, bytes));
    CUDA_CHECK(cudaMalloc(&dCol, numNodes * TILE * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&dRow, numNodes * TILE * sizeof(unsigned int)));

    CUDA_CHECK(cudaMemcpy(dDist, dist.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dPath, path.data(), bytes, cudaMemcpyHostToDevice));

    const dim3 block(TILE, TILE_Y);
    for (int K = 0; K < numTiles; ++K) {
        const size_t kb = static_cast<size_t>(K) * TILE;
        const int bs = static_cast<int>(std::min(static_cast<size_t>(TILE), numNodes - kb));

        fwPhase1<<<1, block>>>(dDist, dPath, dCol, dRow, numNodes, kb, bs);
        CUDA_CHECK(cudaGetLastError());

        fwPhase2<<<dim3(numTiles, 2), block>>>(dDist, dPath, dCol, dRow, numNodes, K, kb, bs);
        CUDA_CHECK(cudaGetLastError());

        fwPhase3<<<dim3(numTiles, numTiles), block>>>(dDist, dPath, dCol, dRow, numNodes, K, kb, bs);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaMemcpy(dist.data(), dDist, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(path.data(), dPath, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(dDist));
    CUDA_CHECK(cudaFree(dPath));
    CUDA_CHECK(cudaFree(dCol));
    CUDA_CHECK(cudaFree(dRow));
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

    // Create the CUDA context before timing so one-time driver setup is not
    // counted as computation time.
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
