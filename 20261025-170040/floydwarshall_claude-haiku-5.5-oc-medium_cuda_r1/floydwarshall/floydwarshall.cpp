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
// CUDA implementation
//
// Blocked (tiled) Floyd-Warshall. The matrices are row-major: element (r, c)
// lives at index r * n + c, which is the same memory the original code reads
// through idx2(c, r, n). For every pair (i, j) the candidate paths through
// intermediate nodes are considered in the same ascending order of k as the
// original triple loop, using unsigned 32-bit arithmetic, so distances are
// identical to the sequential algorithm.
//
// Each round t processes the diagonal block (t,t) first, then the blocks in
// row t and column t, and finally all remaining blocks, which are independent
// of each other within the round and are executed in parallel.
// ---------------------------------------------------------------------------

constexpr int TILE = 32;  // threads per block: TILE x TILE

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err_));                                  \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// Phase 1: relax the diagonal tile (t,t) using only intermediate nodes in block t.
__global__ void fwPhase1(unsigned int* __restrict__ dist,
                         unsigned int* __restrict__ path,
                         const size_t n, const int t, const int kmax) {
    __shared__ unsigned int sD[TILE][TILE];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const size_t gr = (size_t)t * TILE + ty;
    const size_t gc = (size_t)t * TILE + tx;
    const bool inB = gr < n && gc < n;
    const size_t g = gr * n + gc;

    unsigned int d = inB ? dist[g] : INF;
    unsigned int p = inB ? path[g] : 0u;
    sD[ty][tx] = d;
    __syncthreads();

    for (int kk = 0; kk < kmax; ++kk) {
        const unsigned int cand = sD[ty][kk] + sD[kk][tx];
        __syncthreads();  // all reads of this step complete before any write
        if (cand < d) {
            d = cand;
            p = static_cast<unsigned int>(t * TILE + kk);
            sD[ty][tx] = d;
        }
        __syncthreads();
    }

    if (inB) {
        dist[g] = d;
        path[g] = p;
    }
}

// Phase 2: relax the tiles of row block t (blockIdx.y == 0) and column block t
// (blockIdx.y == 1), using the finalized diagonal tile.
__global__ void fwPhase2(unsigned int* __restrict__ dist,
                         unsigned int* __restrict__ path,
                         const size_t n, const int t, const int kmax) {
    __shared__ unsigned int sD[TILE][TILE];  // diagonal tile (t,t)
    __shared__ unsigned int sC[TILE][TILE];  // own tile
    const int tx = threadIdx.x, ty = threadIdx.y;
    const bool rowTile = (blockIdx.y == 0);
    const int other = blockIdx.x + (blockIdx.x >= t);  // block index skipping t

    const size_t gr = (size_t)(rowTile ? t : other) * TILE + ty;
    const size_t gc = (size_t)(rowTile ? other : t) * TILE + tx;
    const bool inB = gr < n && gc < n;
    const size_t g = gr * n + gc;

    const size_t dr = (size_t)t * TILE + ty;
    const size_t dc = (size_t)t * TILE + tx;
    sD[ty][tx] = (dr < n && dc < n) ? dist[dr * n + dc] : INF;

    unsigned int d = inB ? dist[g] : INF;
    unsigned int p = inB ? path[g] : 0u;
    sC[ty][tx] = d;
    __syncthreads();

    for (int kk = 0; kk < kmax; ++kk) {
        // Row tile: dist(i,k) from diagonal, dist(k,j) from own tile.
        // Column tile: dist(i,k) from own tile, dist(k,j) from diagonal.
        const unsigned int cand = rowTile ? (sD[ty][kk] + sC[kk][tx])
                                          : (sC[ty][kk] + sD[kk][tx]);
        __syncthreads();
        if (cand < d) {
            d = cand;
            p = static_cast<unsigned int>(t * TILE + kk);
            sC[ty][tx] = d;
        }
        __syncthreads();
    }

    if (inB) {
        dist[g] = d;
        path[g] = p;
    }
}

// Phase 3: relax all remaining tiles (ib, jb) with ib != t and jb != t. Each
// one reads only finalized row/column tiles of block t, so it needs no
// synchronization inside the k loop.
__global__ void fwPhase3(unsigned int* __restrict__ dist,
                         unsigned int* __restrict__ path,
                         const size_t n, const int t, const int kmax) {
    __shared__ unsigned int sR[TILE][TILE];   // dist(k, j) from row tile (t, jb)
    __shared__ unsigned int sCo[TILE][TILE];  // dist(i, k) from column tile (ib, t)
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int jb = blockIdx.x + (blockIdx.x >= t);
    const int ib = blockIdx.y + (blockIdx.y >= t);

    const size_t gr = (size_t)ib * TILE + ty;
    const size_t gc = (size_t)jb * TILE + tx;
    const bool inB = gr < n && gc < n;
    const size_t g = gr * n + gc;

    const size_t rr = (size_t)t * TILE + ty;
    const size_t cc = (size_t)t * TILE + tx;
    sR[ty][tx] = (rr < n && gc < n) ? dist[rr * n + gc] : INF;
    sCo[ty][tx] = (gr < n && cc < n) ? dist[gr * n + cc] : INF;

    unsigned int d = inB ? dist[g] : INF;
    unsigned int p = inB ? path[g] : 0u;
    __syncthreads();

    const int base = t * TILE;
    for (int kk = 0; kk < kmax; ++kk) {
        const unsigned int cand = sCo[ty][kk] + sR[kk][tx];
        if (cand < d) {
            d = cand;
            p = static_cast<unsigned int>(base + kk);
        }
    }

    if (inB) {
        dist[g] = d;
        path[g] = p;
    }
}

void floydWarshall(std::vector<unsigned int>& dist, 
                   std::vector<unsigned int>& path, 
                   const size_t numNodes) {
    if (numNodes == 0) {
        return;
    }

    const size_t total = numNodes * numNodes;
    unsigned int* dDist = nullptr;
    unsigned int* dPath = nullptr;
    CUDA_CHECK(cudaMalloc(&dDist, total * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&dPath, total * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpy(dDist, dist.data(), total * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dPath, path.data(), total * sizeof(unsigned int), cudaMemcpyHostToDevice));

    const int nb = static_cast<int>((numNodes + TILE - 1) / TILE);
    const dim3 block(TILE, TILE);

    for (int t = 0; t < nb; ++t) {
        const int kmax = static_cast<int>(std::min(static_cast<size_t>(TILE),
                                                   numNodes - static_cast<size_t>(t) * TILE));
        fwPhase1<<<1, block>>>(dDist, dPath, numNodes, t, kmax);
        if (nb > 1) {
            fwPhase2<<<dim3(nb - 1, 2), block>>>(dDist, dPath, numNodes, t, kmax);
            fwPhase3<<<dim3(nb - 1, nb - 1), block>>>(dDist, dPath, numNodes, t, kmax);
        }
    }
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(dist.data(), dDist, total * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(path.data(), dPath, total * sizeof(unsigned int), cudaMemcpyDeviceToHost));

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
    
    // Create the CUDA context before timing so that one-time driver setup is not measured
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
