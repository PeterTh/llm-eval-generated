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

// Tile edge length for the blocked Floyd-Warshall algorithm.
constexpr int TILE = 32;

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        cudaError_t err__ = (call);                                                   \
        if (err__ != cudaSuccess) {                                                   \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__, \
                    cudaGetErrorString(err__));                                       \
            exit(EXIT_FAILURE);                                                       \
        }                                                                             \
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
// Blocked Floyd-Warshall on the GPU.
//
// The (padded) distance matrix is split into TILE x TILE tiles. For every
// k-block kb the algorithm runs three phases:
//   1. the diagonal tile (kb,kb) is relaxed with itself,
//   2. the tiles in row kb and column kb are relaxed using the diagonal tile,
//   3. all remaining tiles are relaxed using their row-kb and column-kb tiles.
// Each phase is one kernel launch; phases within one k-block are dependent,
// so the stream ordering provides the required synchronisation.
//
// Layout: row-major with dist[i * N + j] (equivalent to idx2(j, i, N)).
// ---------------------------------------------------------------------------

// Relax one element against a sequence of TILE intermediate nodes. `rowK`
// holds dist[i][k], `colK` holds dist[k][j] for the local k range, and kBase
// is the global index of the first intermediate node.
__device__ __forceinline__ void relaxTile(unsigned int& d, unsigned int& p,
                                          const unsigned int* __restrict__ rowK,
                                          const unsigned int* __restrict__ colK,
                                          const unsigned int kBase) {
#pragma unroll
    for (int k = 0; k < TILE; ++k) {
        const unsigned int cand = rowK[k] + colK[k];
        if (cand < d) {
            d = cand;
            p = kBase + k;
        }
    }
}

// Phase 1: diagonal tile. In-place updates require a barrier after each k.
__global__ void fwPhase1(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
                         const int N, const int kb) {
    __shared__ unsigned int sD[TILE][TILE + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int base = kb * TILE;
    const size_t g = static_cast<size_t>(base + ty) * N + base + tx;

    unsigned int d = dist[g];
    unsigned int p = path[g];
    sD[ty][tx] = d;
    __syncthreads();

    for (int k = 0; k < TILE; ++k) {
        const unsigned int cand = sD[ty][k] + sD[k][tx];
        if (cand < d) {
            d = cand;
            p = base + k;
        }
        __syncthreads();
        sD[ty][tx] = d;
        __syncthreads();
    }

    dist[g] = d;
    path[g] = p;
}

// Phase 2: tiles sharing a row (blockIdx.y == 0) or column (blockIdx.y == 1)
// with the diagonal tile. blockIdx.x enumerates the tiles; the diagonal tile
// itself is skipped.
__global__ void fwPhase2(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
                         const int N, const int kb) {
    __shared__ unsigned int sDiag[TILE][TILE + 1];
    __shared__ unsigned int sCur[TILE][TILE + 1];

    int tb = blockIdx.x;
    if (tb >= kb) ++tb;  // skip the diagonal tile

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int base = kb * TILE;

    // Load diagonal tile.
    sDiag[ty][tx] = dist[static_cast<size_t>(base + ty) * N + base + tx];

    int rowBase, colBase;
    if (blockIdx.y == 0) {  // tile (kb, tb): same rows as diagonal
        rowBase = base;
        colBase = tb * TILE;
    } else {                // tile (tb, kb): same columns as diagonal
        rowBase = tb * TILE;
        colBase = base;
    }

    const size_t g = static_cast<size_t>(rowBase + ty) * N + colBase + tx;
    unsigned int d = dist[g];
    unsigned int p = path[g];
    sCur[ty][tx] = d;
    __syncthreads();

    if (blockIdx.y == 0) {
        // dist[i][k] comes from the diagonal tile, dist[k][j] from this tile.
        for (int k = 0; k < TILE; ++k) {
            const unsigned int cand = sDiag[ty][k] + sCur[k][tx];
            if (cand < d) {
                d = cand;
                p = base + k;
            }
            __syncthreads();
            sCur[ty][tx] = d;
            __syncthreads();
        }
    } else {
        // dist[i][k] comes from this tile, dist[k][j] from the diagonal tile.
        for (int k = 0; k < TILE; ++k) {
            const unsigned int cand = sCur[ty][k] + sDiag[k][tx];
            if (cand < d) {
                d = cand;
                p = base + k;
            }
            __syncthreads();
            sCur[ty][tx] = d;
            __syncthreads();
        }
    }

    dist[g] = d;
    path[g] = p;
}

// Phase 3: all remaining tiles. Each block handles one tile; its row-kb and
// column-kb tiles are already final for this k-block, so no barriers are
// needed inside the k loop. Each thread processes PH3_ROWS rows of the tile
// to raise ILP and reuse the column tile loads.
constexpr int PH3_ROWS = 4;

__global__ void __launch_bounds__(TILE * TILE / PH3_ROWS)
fwPhase3(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
         const int N, const int kb) {
    __shared__ unsigned int sRow[TILE][TILE + 1];  // tile (ti, kb): dist[i][k]
    __shared__ unsigned int sCol[TILE][TILE + 1];  // tile (kb, tj): dist[k][j]

    int ti = blockIdx.y;
    int tj = blockIdx.x;
    if (ti >= kb) ++ti;
    if (tj >= kb) ++tj;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;  // 0 .. TILE/PH3_ROWS - 1
    const int base = kb * TILE;
    const int rowBase = ti * TILE;
    const int colBase = tj * TILE;

#pragma unroll
    for (int r = 0; r < PH3_ROWS; ++r) {
        const int ly = ty + r * (TILE / PH3_ROWS);
        sRow[ly][tx] = dist[static_cast<size_t>(rowBase + ly) * N + base + tx];
        sCol[ly][tx] = dist[static_cast<size_t>(base + ly) * N + colBase + tx];
    }
    __syncthreads();

    unsigned int d[PH3_ROWS];
    unsigned int p[PH3_ROWS];
    size_t g[PH3_ROWS];
#pragma unroll
    for (int r = 0; r < PH3_ROWS; ++r) {
        const int ly = ty + r * (TILE / PH3_ROWS);
        g[r] = static_cast<size_t>(rowBase + ly) * N + colBase + tx;
        d[r] = dist[g[r]];
        p[r] = path[g[r]];
    }

#pragma unroll
    for (int k = 0; k < TILE; ++k) {
        const unsigned int ckj = sCol[k][tx];
#pragma unroll
        for (int r = 0; r < PH3_ROWS; ++r) {
            const int ly = ty + r * (TILE / PH3_ROWS);
            const unsigned int cand = sRow[ly][k] + ckj;
            if (cand < d[r]) {
                d[r] = cand;
                p[r] = base + k;
            }
        }
    }

#pragma unroll
    for (int r = 0; r < PH3_ROWS; ++r) {
        dist[g[r]] = d[r];
        path[g[r]] = p[r];
    }
}

void floydWarshall(std::vector<unsigned int>& dist, 
                   std::vector<unsigned int>& path, 
                   const size_t numNodes) {
    if (numNodes == 0) return;

    // Pad to a multiple of TILE. Padding rows/columns carry INF (diagonal 0)
    // so they can never shorten a path between real nodes
    // (INF + INF fits in 32 bits without overflow).
    const int numTiles = static_cast<int>((numNodes + TILE - 1) / TILE);
    const int N = numTiles * TILE;
    const size_t bytes = static_cast<size_t>(N) * N * sizeof(unsigned int);

    unsigned int* dDist = nullptr;
    unsigned int* dPath = nullptr;
    CUDA_CHECK(cudaMalloc(&dDist, bytes));
    CUDA_CHECK(cudaMalloc(&dPath, bytes));

    if (static_cast<size_t>(N) == numNodes) {
        CUDA_CHECK(cudaMemcpy(dDist, dist.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dPath, path.data(), bytes, cudaMemcpyHostToDevice));
    } else {
        std::vector<unsigned int> hDist(static_cast<size_t>(N) * N, INF);
        std::vector<unsigned int> hPath(static_cast<size_t>(N) * N, 0);
        for (int i = numNodes; i < N; ++i) hDist[static_cast<size_t>(i) * N + i] = 0;
        for (size_t i = 0; i < numNodes; ++i) {
            std::copy(dist.begin() + i * numNodes, dist.begin() + (i + 1) * numNodes,
                      hDist.begin() + i * N);
            std::copy(path.begin() + i * numNodes, path.begin() + (i + 1) * numNodes,
                      hPath.begin() + i * N);
        }
        CUDA_CHECK(cudaMemcpy(dDist, hDist.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dPath, hPath.data(), bytes, cudaMemcpyHostToDevice));
    }

    const dim3 fullBlock(TILE, TILE);
    const dim3 ph3Block(TILE, TILE / PH3_ROWS);

    for (int kb = 0; kb < numTiles; ++kb) {
        fwPhase1<<<1, fullBlock>>>(dDist, dPath, N, kb);
        if (numTiles > 1) {
            fwPhase2<<<dim3(numTiles - 1, 2), fullBlock>>>(dDist, dPath, N, kb);
            fwPhase3<<<dim3(numTiles - 1, numTiles - 1), ph3Block>>>(dDist, dPath, N, kb);
        }
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    if (static_cast<size_t>(N) == numNodes) {
        CUDA_CHECK(cudaMemcpy(dist.data(), dDist, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(path.data(), dPath, bytes, cudaMemcpyDeviceToHost));
    } else {
        CUDA_CHECK(cudaMemcpy2D(dist.data(), numNodes * sizeof(unsigned int), dDist,
                                static_cast<size_t>(N) * sizeof(unsigned int),
                                numNodes * sizeof(unsigned int), numNodes,
                                cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy2D(path.data(), numNodes * sizeof(unsigned int), dPath,
                                static_cast<size_t>(N) * sizeof(unsigned int),
                                numNodes * sizeof(unsigned int), numNodes,
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

    // Create the CUDA context up front so its one-off cost is not timed.
    CUDA_CHECK(cudaFree(0));
    
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
