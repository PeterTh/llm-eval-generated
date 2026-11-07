// Floyd-Warshall all-pairs shortest path benchmark, CUDA implementation.
//
// The computation uses the blocked (tiled) Floyd-Warshall formulation:
// for every diagonal tile B the algorithm runs three dependent phases
//   1. update the diagonal tile (B,B) using only itself,
//   2. update all tiles in tile-row B and tile-column B using (B,B),
//   3. update every remaining tile using its row-B and column-B tiles.
// Each phase is a kernel launch; tiles are staged through shared memory and
// each thread keeps a 2x2 register footprint. The resulting distance matrix is
// identical to the classic triple loop.

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

// Tile edge length and thread-block geometry (each thread owns 2x2 elements).
constexpr int TILE = 64;
constexpr int BLOCK_DIM = 32;
static_assert(TILE == 2 * BLOCK_DIM, "each thread handles a 2x2 sub-block");

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__,    \
                    __LINE__, cudaGetErrorString(err__));                       \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
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
// Device kernels. The matrices on the device are row-major with pitch `n`
// (element (row r, col c) lives at r * n + c, matching idx2(c, r, n) on the
// host) and `n` is a multiple of TILE (padded rows/cols carry INF edges).
// ---------------------------------------------------------------------------

// Relax the four elements owned by this thread against tile row/col k.
// rowK[c]  = dist(tileRow + ?, k) for the thread's rows  -> passed as ik0/ik1
// colK[c]  = dist(k, tileCol + c)                        -> kj tile row
__device__ __forceinline__ void relax4(unsigned int& d00, unsigned int& d01,
                                       unsigned int& d10, unsigned int& d11,
                                       int& p00, int& p01, int& p10, int& p11,
                                       const unsigned int ik0, const unsigned int ik1,
                                       const unsigned int kj0, const unsigned int kj1,
                                       const int k) {
    unsigned int c;
    c = ik0 + kj0; if (c < d00) { d00 = c; p00 = k; }
    c = ik0 + kj1; if (c < d01) { d01 = c; p01 = k; }
    c = ik1 + kj0; if (c < d10) { d10 = c; p10 = k; }
    c = ik1 + kj1; if (c < d11) { d11 = c; p11 = k; }
}

__device__ __forceinline__ void storeIfChanged(unsigned int* __restrict__ dist,
                                               unsigned int* __restrict__ path,
                                               const size_t off, const unsigned int d,
                                               const int p) {
    if (p >= 0) {
        dist[off] = d;
        path[off] = static_cast<unsigned int>(p);
    }
}

// Phase 1: diagonal tile (b, b) depends only on itself.
__global__ void __launch_bounds__(BLOCK_DIM * BLOCK_DIM)
fwPhase1(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
         const int n, const int b) {
    __shared__ unsigned int tile[TILE][TILE + 1];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int base = b * TILE;
    const size_t o00 = (size_t)(base + ty) * n + base + tx;
    const size_t o01 = o00 + BLOCK_DIM;
    const size_t o10 = o00 + (size_t)BLOCK_DIM * n;
    const size_t o11 = o10 + BLOCK_DIM;

    unsigned int d00 = dist[o00], d01 = dist[o01], d10 = dist[o10], d11 = dist[o11];
    int p00 = -1, p01 = -1, p10 = -1, p11 = -1;

    tile[ty][tx] = d00;
    tile[ty][tx + BLOCK_DIM] = d01;
    tile[ty + BLOCK_DIM][tx] = d10;
    tile[ty + BLOCK_DIM][tx + BLOCK_DIM] = d11;
    __syncthreads();

    #pragma unroll 8
    for (int k = 0; k < TILE; ++k) {
        const unsigned int ik0 = tile[ty][k];
        const unsigned int ik1 = tile[ty + BLOCK_DIM][k];
        const unsigned int kj0 = tile[k][tx];
        const unsigned int kj1 = tile[k][tx + BLOCK_DIM];
        relax4(d00, d01, d10, d11, p00, p01, p10, p11, ik0, ik1, kj0, kj1, base + k);
        __syncthreads();
        tile[ty][tx] = d00;
        tile[ty][tx + BLOCK_DIM] = d01;
        tile[ty + BLOCK_DIM][tx] = d10;
        tile[ty + BLOCK_DIM][tx + BLOCK_DIM] = d11;
        __syncthreads();
    }

    storeIfChanged(dist, path, o00, d00, p00);
    storeIfChanged(dist, path, o01, d01, p01);
    storeIfChanged(dist, path, o10, d10, p10);
    storeIfChanged(dist, path, o11, d11, p11);
}

// Phase 2: tiles sharing a tile-row or tile-column with the diagonal tile.
// blockIdx.y == 0 -> tile-row b (uses diag as the (i,k) operand)
// blockIdx.y == 1 -> tile-column b (uses diag as the (k,j) operand)
// blockIdx.x enumerates the TILE-index skipping b itself.
__global__ void __launch_bounds__(BLOCK_DIM * BLOCK_DIM)
fwPhase2(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
         const int n, const int b) {
    __shared__ unsigned int diag[TILE][TILE + 1];
    __shared__ unsigned int cur[TILE][TILE + 1];

    const int tx = threadIdx.x, ty = threadIdx.y;
    int t = blockIdx.x;
    if (t >= b) ++t;  // skip the diagonal tile

    const bool rowTile = (blockIdx.y == 0);
    const int tileRow = rowTile ? b : t;
    const int tileCol = rowTile ? t : b;
    const int kbase = b * TILE;

    const size_t dOff = (size_t)(kbase + ty) * n + kbase + tx;
    const size_t o00 = (size_t)(tileRow * TILE + ty) * n + tileCol * TILE + tx;
    const size_t o01 = o00 + BLOCK_DIM;
    const size_t o10 = o00 + (size_t)BLOCK_DIM * n;
    const size_t o11 = o10 + BLOCK_DIM;

    diag[ty][tx] = dist[dOff];
    diag[ty][tx + BLOCK_DIM] = dist[dOff + BLOCK_DIM];
    diag[ty + BLOCK_DIM][tx] = dist[dOff + (size_t)BLOCK_DIM * n];
    diag[ty + BLOCK_DIM][tx + BLOCK_DIM] = dist[dOff + (size_t)BLOCK_DIM * n + BLOCK_DIM];

    unsigned int d00 = dist[o00], d01 = dist[o01], d10 = dist[o10], d11 = dist[o11];
    int p00 = -1, p01 = -1, p10 = -1, p11 = -1;

    cur[ty][tx] = d00;
    cur[ty][tx + BLOCK_DIM] = d01;
    cur[ty + BLOCK_DIM][tx] = d10;
    cur[ty + BLOCK_DIM][tx + BLOCK_DIM] = d11;
    __syncthreads();

    if (rowTile) {
        // dist(i,j) with i in diag block: ik from diag, kj from cur (changes).
        #pragma unroll 8
        for (int k = 0; k < TILE; ++k) {
            const unsigned int ik0 = diag[ty][k];
            const unsigned int ik1 = diag[ty + BLOCK_DIM][k];
            const unsigned int kj0 = cur[k][tx];
            const unsigned int kj1 = cur[k][tx + BLOCK_DIM];
            relax4(d00, d01, d10, d11, p00, p01, p10, p11, ik0, ik1, kj0, kj1, kbase + k);
            __syncthreads();
            cur[ty][tx] = d00;
            cur[ty][tx + BLOCK_DIM] = d01;
            cur[ty + BLOCK_DIM][tx] = d10;
            cur[ty + BLOCK_DIM][tx + BLOCK_DIM] = d11;
            __syncthreads();
        }
    } else {
        // dist(i,j) with j in diag block: ik from cur (changes), kj from diag.
        #pragma unroll 8
        for (int k = 0; k < TILE; ++k) {
            const unsigned int ik0 = cur[ty][k];
            const unsigned int ik1 = cur[ty + BLOCK_DIM][k];
            const unsigned int kj0 = diag[k][tx];
            const unsigned int kj1 = diag[k][tx + BLOCK_DIM];
            relax4(d00, d01, d10, d11, p00, p01, p10, p11, ik0, ik1, kj0, kj1, kbase + k);
            __syncthreads();
            cur[ty][tx] = d00;
            cur[ty][tx + BLOCK_DIM] = d01;
            cur[ty + BLOCK_DIM][tx] = d10;
            cur[ty + BLOCK_DIM][tx + BLOCK_DIM] = d11;
            __syncthreads();
        }
    }

    storeIfChanged(dist, path, o00, d00, p00);
    storeIfChanged(dist, path, o01, d01, p01);
    storeIfChanged(dist, path, o10, d10, p10);
    storeIfChanged(dist, path, o11, d11, p11);
}

// Phase 3: all remaining tiles; operands come from the (already final for this
// round) row-b and column-b tiles, so no intra-loop synchronisation is needed.
__global__ void __launch_bounds__(BLOCK_DIM * BLOCK_DIM)
fwPhase3(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
         const int n, const int b) {
    __shared__ unsigned int rowK[TILE][TILE + 1];  // tile (b, tileCol)   -> (k, j)
    __shared__ unsigned int colK[TILE][TILE + 1];  // tile (tileRow, b)   -> (i, k)

    const int tx = threadIdx.x, ty = threadIdx.y;
    int tileCol = blockIdx.x;
    int tileRow = blockIdx.y;
    if (tileCol >= b) ++tileCol;
    if (tileRow >= b) ++tileRow;

    const int kbase = b * TILE;
    const size_t rOff = (size_t)(kbase + ty) * n + tileCol * TILE + tx;
    const size_t cOff = (size_t)(tileRow * TILE + ty) * n + kbase + tx;
    const size_t o00 = (size_t)(tileRow * TILE + ty) * n + tileCol * TILE + tx;
    const size_t o01 = o00 + BLOCK_DIM;
    const size_t o10 = o00 + (size_t)BLOCK_DIM * n;
    const size_t o11 = o10 + BLOCK_DIM;

    rowK[ty][tx] = dist[rOff];
    rowK[ty][tx + BLOCK_DIM] = dist[rOff + BLOCK_DIM];
    rowK[ty + BLOCK_DIM][tx] = dist[rOff + (size_t)BLOCK_DIM * n];
    rowK[ty + BLOCK_DIM][tx + BLOCK_DIM] = dist[rOff + (size_t)BLOCK_DIM * n + BLOCK_DIM];

    colK[ty][tx] = dist[cOff];
    colK[ty][tx + BLOCK_DIM] = dist[cOff + BLOCK_DIM];
    colK[ty + BLOCK_DIM][tx] = dist[cOff + (size_t)BLOCK_DIM * n];
    colK[ty + BLOCK_DIM][tx + BLOCK_DIM] = dist[cOff + (size_t)BLOCK_DIM * n + BLOCK_DIM];

    unsigned int d00 = dist[o00], d01 = dist[o01], d10 = dist[o10], d11 = dist[o11];
    int p00 = -1, p01 = -1, p10 = -1, p11 = -1;
    __syncthreads();

    #pragma unroll 16
    for (int k = 0; k < TILE; ++k) {
        const unsigned int ik0 = colK[ty][k];
        const unsigned int ik1 = colK[ty + BLOCK_DIM][k];
        const unsigned int kj0 = rowK[k][tx];
        const unsigned int kj1 = rowK[k][tx + BLOCK_DIM];
        relax4(d00, d01, d10, d11, p00, p01, p10, p11, ik0, ik1, kj0, kj1, kbase + k);
    }

    storeIfChanged(dist, path, o00, d00, p00);
    storeIfChanged(dist, path, o01, d01, p01);
    storeIfChanged(dist, path, o10, d10, p10);
    storeIfChanged(dist, path, o11, d11, p11);
}

// Fill padding region: INF everywhere outside the real n x n block, except 0
// on the padded diagonal so padded nodes behave like isolated vertices.
__global__ void fwPadKernel(unsigned int* __restrict__ dist, const int n, const int nPad) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (r >= nPad || c >= nPad) return;
    if (r < n && c < n) return;
    dist[(size_t)r * nPad + c] = (r == c) ? 0u : INF;
}

void floydWarshall(std::vector<unsigned int>& dist, 
                   std::vector<unsigned int>& path, 
                   const size_t numNodes) {
    if (numNodes == 0) return;

    const int n = static_cast<int>(numNodes);
    const int numTiles = (n + TILE - 1) / TILE;
    const int nPad = numTiles * TILE;
    const size_t padBytes = (size_t)nPad * nPad * sizeof(unsigned int);
    const size_t rowBytes = (size_t)n * sizeof(unsigned int);

    unsigned int* dDist = nullptr;
    unsigned int* dPath = nullptr;
    CUDA_CHECK(cudaMalloc(&dDist, padBytes));
    CUDA_CHECK(cudaMalloc(&dPath, padBytes));

    CUDA_CHECK(cudaMemcpy2D(dDist, (size_t)nPad * sizeof(unsigned int),
                            dist.data(), rowBytes, rowBytes, n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy2D(dPath, (size_t)nPad * sizeof(unsigned int),
                            path.data(), rowBytes, rowBytes, n, cudaMemcpyHostToDevice));

    if (nPad != n) {
        const dim3 pb(32, 8);
        const dim3 pg((nPad + pb.x - 1) / pb.x, (nPad + pb.y - 1) / pb.y);
        fwPadKernel<<<pg, pb>>>(dDist, n, nPad);
        CUDA_CHECK(cudaGetLastError());
    }

    const dim3 block(BLOCK_DIM, BLOCK_DIM);
    for (int b = 0; b < numTiles; ++b) {
        fwPhase1<<<1, block>>>(dDist, dPath, nPad, b);
        if (numTiles > 1) {
            fwPhase2<<<dim3(numTiles - 1, 2), block>>>(dDist, dPath, nPad, b);
            fwPhase3<<<dim3(numTiles - 1, numTiles - 1), block>>>(dDist, dPath, nPad, b);
        }
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy2D(dist.data(), rowBytes, dDist, (size_t)nPad * sizeof(unsigned int),
                            rowBytes, n, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy2D(path.data(), rowBytes, dPath, (size_t)nPad * sizeof(unsigned int),
                            rowBytes, n, cudaMemcpyDeviceToHost));

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

    // Create the CUDA context up front so it is not charged to the timed region.
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
