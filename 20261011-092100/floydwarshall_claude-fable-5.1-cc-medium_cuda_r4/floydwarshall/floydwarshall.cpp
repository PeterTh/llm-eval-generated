// Floyd-Warshall All-Pairs Shortest Path Benchmark -- CUDA implementation.
//
// This file is compiled as CUDA (see CMakeLists.txt). The algorithm is the
// classic blocked (tiled) Floyd-Warshall: for every block-row/column kb of the
// distance matrix three dependent phases are executed on the GPU:
//   phase 1: the diagonal tile (kb,kb) is relaxed in shared memory,
//   phase 2: the tiles in block-row kb and block-column kb are relaxed using
//            the diagonal tile,
//   phase 3: all remaining tiles are relaxed using their row/column tiles.
// Each phase is a single kernel launch; all blocks within a phase are fully
// independent, which gives the parallel scalability.

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

// Tile edge length and thread-block geometry (TILE x TILE_ROWS threads, every
// thread owns TILE/TILE_ROWS = 4 elements of the tile).
constexpr int TILE = 32;
constexpr int TILE_ROWS = 8;
constexpr int ELEMS_PER_THREAD = TILE / TILE_ROWS;

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err__ = (call);                                               \
        if (err__ != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__,     \
                    __LINE__, cudaGetErrorString(err__));                         \
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
// Device kernels. The device matrices are row-major with a padded pitch `np`
// (a multiple of TILE): D[i * np + j] is the distance from node i to node j,
// which matches the host layout dist[idx2(j, i, n)] == dist[i * n + j].
// ---------------------------------------------------------------------------

// Fill the padding region (rows/cols >= n) with INF, diagonal with 0 and the
// padded path entries with their column index, so padded nodes never shorten
// a real path and the algorithm needs no bounds checks.
__global__ void initPaddingKernel(unsigned int* __restrict__ dist,
                                  unsigned int* __restrict__ path,
                                  const int n, const int np) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= np || j >= np) return;
    if (i >= n || j >= n) {
        dist[(size_t)i * np + j] = (i == j) ? 0u : INF;
        path[(size_t)i * np + j] = j;
    }
}

// Phase 1: relax the diagonal tile (kb, kb) entirely in shared memory.
__global__ void __launch_bounds__(TILE * TILE_ROWS)
fwPhase1Kernel(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
               const int np, const int kb) {
    __shared__ unsigned int sD[TILE][TILE + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int base = kb * TILE;

    unsigned int p[ELEMS_PER_THREAD];
    bool changed[ELEMS_PER_THREAD];

#pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; ++e) {
        const int r = ty + e * TILE_ROWS;
        sD[r][tx] = dist[(size_t)(base + r) * np + base + tx];
        changed[e] = false;
        p[e] = 0;
    }
    __syncthreads();

    for (int k = 0; k < TILE; ++k) {
#pragma unroll
        for (int e = 0; e < ELEMS_PER_THREAD; ++e) {
            const int r = ty + e * TILE_ROWS;
            const unsigned int cand = sD[r][k] + sD[k][tx];
            if (cand < sD[r][tx]) {
                sD[r][tx] = cand;
                p[e] = base + k;
                changed[e] = true;
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; ++e) {
        const int r = ty + e * TILE_ROWS;
        const size_t gi = (size_t)(base + r) * np + base + tx;
        if (changed[e]) {
            dist[gi] = sD[r][tx];
            path[gi] = p[e];
        }
    }
}

// Phase 2: relax the tiles in block-row kb (blockIdx.y == 0) and block-column
// kb (blockIdx.y == 1) using the already finished diagonal tile.
__global__ void __launch_bounds__(TILE * TILE_ROWS)
fwPhase2Kernel(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
               const int np, const int kb) {
    __shared__ unsigned int sDiag[TILE][TILE + 1];
    __shared__ unsigned int sCur[TILE][TILE + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // Skip the diagonal tile itself.
    int tb = blockIdx.x;
    if (tb >= kb) ++tb;

    const bool rowTile = (blockIdx.y == 0);
    const int base = kb * TILE;
    const int tileRow = rowTile ? base : tb * TILE;   // first global row of tile
    const int tileCol = rowTile ? tb * TILE : base;   // first global col of tile

    unsigned int p[ELEMS_PER_THREAD];
    bool changed[ELEMS_PER_THREAD];

#pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; ++e) {
        const int r = ty + e * TILE_ROWS;
        sDiag[r][tx] = dist[(size_t)(base + r) * np + base + tx];
        sCur[r][tx] = dist[(size_t)(tileRow + r) * np + tileCol + tx];
        changed[e] = false;
        p[e] = 0;
    }
    __syncthreads();

    if (rowTile) {
        // Tile (kb, tb): d(i,j) = min(d(i,j), diag(i,k) + cur(k,j))
        for (int k = 0; k < TILE; ++k) {
            const unsigned int dkj = sCur[k][tx];
#pragma unroll
            for (int e = 0; e < ELEMS_PER_THREAD; ++e) {
                const int r = ty + e * TILE_ROWS;
                const unsigned int cand = sDiag[r][k] + dkj;
                if (cand < sCur[r][tx]) {
                    sCur[r][tx] = cand;
                    p[e] = base + k;
                    changed[e] = true;
                }
            }
            __syncthreads();
        }
    } else {
        // Tile (tb, kb): d(i,j) = min(d(i,j), cur(i,k) + diag(k,j))
        for (int k = 0; k < TILE; ++k) {
            const unsigned int dkj = sDiag[k][tx];
#pragma unroll
            for (int e = 0; e < ELEMS_PER_THREAD; ++e) {
                const int r = ty + e * TILE_ROWS;
                const unsigned int cand = sCur[r][k] + dkj;
                if (cand < sCur[r][tx]) {
                    sCur[r][tx] = cand;
                    p[e] = base + k;
                    changed[e] = true;
                }
            }
            __syncthreads();
        }
    }

#pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; ++e) {
        const int r = ty + e * TILE_ROWS;
        const size_t gi = (size_t)(tileRow + r) * np + tileCol + tx;
        if (changed[e]) {
            dist[gi] = sCur[r][tx];
            path[gi] = p[e];
        }
    }
}

// Phase 3: relax every tile (ib, jb) with ib != kb and jb != kb using the
// finished row tile (ib, kb) and column tile (kb, jb). The tile's own values
// live in registers; no synchronisation is needed inside the k loop because
// the row/column tiles are read-only during this phase.
__global__ void __launch_bounds__(TILE * TILE_ROWS)
fwPhase3Kernel(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
               const int np, const int kb) {
    __shared__ unsigned int sRow[TILE][TILE + 1];   // tile (ib, kb)
    __shared__ unsigned int sCol[TILE][TILE + 1];   // tile (kb, jb)

    // Skip block-row kb and block-column kb (handled in phase 2).
    int jb = blockIdx.x;
    int ib = blockIdx.y;
    if (jb >= kb) ++jb;
    if (ib >= kb) ++ib;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int base = kb * TILE;
    const int tileRow = ib * TILE;
    const int tileCol = jb * TILE;

    unsigned int cur[ELEMS_PER_THREAD];
    unsigned int p[ELEMS_PER_THREAD];

#pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; ++e) {
        const int r = ty + e * TILE_ROWS;
        sRow[r][tx] = dist[(size_t)(tileRow + r) * np + base + tx];
        sCol[r][tx] = dist[(size_t)(base + r) * np + tileCol + tx];
        cur[e] = dist[(size_t)(tileRow + r) * np + tileCol + tx];
        p[e] = 0xFFFFFFFFu;
    }
    __syncthreads();

#pragma unroll 8
    for (int k = 0; k < TILE; ++k) {
        const unsigned int dkj = sCol[k][tx];
#pragma unroll
        for (int e = 0; e < ELEMS_PER_THREAD; ++e) {
            const int r = ty + e * TILE_ROWS;
            const unsigned int cand = sRow[r][k] + dkj;
            if (cand < cur[e]) {
                cur[e] = cand;
                p[e] = base + k;
            }
        }
    }

#pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; ++e) {
        const int r = ty + e * TILE_ROWS;
        const size_t gi = (size_t)(tileRow + r) * np + tileCol + tx;
        if (p[e] != 0xFFFFFFFFu) {
            dist[gi] = cur[e];
            path[gi] = p[e];
        }
    }
}

// ---------------------------------------------------------------------------
// Host driver for the GPU Floyd-Warshall.
// ---------------------------------------------------------------------------
struct DeviceBuffers {
    unsigned int* dist = nullptr;
    unsigned int* path = nullptr;
    size_t np = 0;
};

DeviceBuffers allocateDeviceBuffers(const size_t numNodes) {
    DeviceBuffers buf;
    buf.np = ((numNodes + TILE - 1) / TILE) * TILE;
    const size_t bytes = buf.np * buf.np * sizeof(unsigned int);
    CUDA_CHECK(cudaMalloc(&buf.dist, bytes));
    CUDA_CHECK(cudaMalloc(&buf.path, bytes));
    return buf;
}

void freeDeviceBuffers(DeviceBuffers& buf) {
    if (buf.dist) CUDA_CHECK(cudaFree(buf.dist));
    if (buf.path) CUDA_CHECK(cudaFree(buf.path));
    buf.dist = buf.path = nullptr;
}

void floydWarshall(std::vector<unsigned int>& dist, 
                   std::vector<unsigned int>& path, 
                   const size_t numNodes,
                   const DeviceBuffers& buf) {
    if (numNodes == 0) return;

    const int n = static_cast<int>(numNodes);
    const int np = static_cast<int>(buf.np);
    const int nb = np / TILE;
    const size_t hostPitch = numNodes * sizeof(unsigned int);
    const size_t devPitch = buf.np * sizeof(unsigned int);

    // Upload the n x n matrices into the padded device matrices.
    CUDA_CHECK(cudaMemcpy2D(buf.dist, devPitch, dist.data(), hostPitch,
                            hostPitch, numNodes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy2D(buf.path, devPitch, path.data(), hostPitch,
                            hostPitch, numNodes, cudaMemcpyHostToDevice));
    if (np != n) {
        const dim3 block(32, 8);
        const dim3 grid((np + block.x - 1) / block.x, (np + block.y - 1) / block.y);
        initPaddingKernel<<<grid, block>>>(buf.dist, buf.path, n, np);
        CUDA_CHECK(cudaGetLastError());
    }

    // Blocked Floyd-Warshall: for each block-index kb run the three phases.
    const dim3 block(TILE, TILE_ROWS);
    for (int kb = 0; kb < nb; ++kb) {
        fwPhase1Kernel<<<1, block>>>(buf.dist, buf.path, np, kb);
        if (nb > 1) {
            fwPhase2Kernel<<<dim3(nb - 1, 2), block>>>(buf.dist, buf.path, np, kb);
            fwPhase3Kernel<<<dim3(nb - 1, nb - 1), block>>>(buf.dist, buf.path, np, kb);
        }
    }
    CUDA_CHECK(cudaGetLastError());

    // Download results.
    CUDA_CHECK(cudaMemcpy2D(dist.data(), hostPitch, buf.dist, devPitch,
                            hostPitch, numNodes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy2D(path.data(), hostPitch, buf.path, devPitch,
                            hostPitch, numNodes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());
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

    // Select and warm up the GPU (context creation is excluded from timing).
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaFree(0));
    {
        // Force the kernels' modules to be loaded now (CUDA uses lazy loading
        // by default) so the first launch inside the timed region is not
        // penalised.
        cudaFuncAttributes attr;
        CUDA_CHECK(cudaFuncGetAttributes(&attr, initPaddingKernel));
        CUDA_CHECK(cudaFuncGetAttributes(&attr, fwPhase1Kernel));
        CUDA_CHECK(cudaFuncGetAttributes(&attr, fwPhase2Kernel));
        CUDA_CHECK(cudaFuncGetAttributes(&attr, fwPhase3Kernel));
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        printf("CUDA device: %s (%d SMs)\n", prop.name, prop.multiProcessorCount);
    }
    
    // Allocate matrices
    std::vector<unsigned int> dist(numNodes * numNodes);
    std::vector<unsigned int> path(numNodes * numNodes);
    DeviceBuffers devBuf = allocateDeviceBuffers(numNodes);
    
    // Initialize
    printf("Initializing graph...\n");
    initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
    initializePathMatrix(path, numNodes);
    
    // Run Floyd-Warshall
    printf("Computing shortest paths...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    floydWarshall(dist, path, numNodes, devBuf);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate operations per second
    // Floyd-Warshall has O(n³) complexity
    double ops = (double)numNodes * numNodes * numNodes;
    double gflops = ops / (duration.count() / 1000.0) / 1e9;
    printf("Performance: %.3f GOPS\n", gflops);

    freeDeviceBuffers(devBuf);
    
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
