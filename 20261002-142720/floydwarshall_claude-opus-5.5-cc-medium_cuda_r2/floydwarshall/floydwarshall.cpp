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
// CUDA blocked Floyd-Warshall
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                         \
                    cudaGetErrorString(err__), __FILE__, __LINE__);             \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

constexpr int TILE = 64;           // tile edge length
constexpr int TPB_DIM = 16;        // threads per block edge (16x16 threads)
constexpr int EPT = TILE / TPB_DIM; // elements per thread per dimension (4)
constexpr int SPAD = TILE + 1;     // padded shared-memory row stride

// Fill padded matrix: INF everywhere, 0 on diagonal
__global__ void fillPaddedKernel(unsigned int* __restrict__ dist,
                                 unsigned int* __restrict__ path, size_t n) {
    const size_t total = n * n;
    for (size_t e = blockIdx.x * (size_t)blockDim.x + threadIdx.x; e < total;
         e += (size_t)gridDim.x * blockDim.x) {
        const size_t r = e / n, c = e % n;
        dist[e] = (r == c) ? 0u : INF;
        path[e] = (unsigned int)c;
    }
}

// Phase 1: the pivot tile (K,K), solved with full in-tile dependency
__global__ void __launch_bounds__(TPB_DIM * TPB_DIM)
fwPhase1(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
         const int n, const int K) {
    __shared__ unsigned int sD[TILE][SPAD];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const size_t base = (size_t)K * TILE * n + (size_t)K * TILE;

    unsigned int p[EPT][EPT];
#pragma unroll
    for (int a = 0; a < EPT; ++a)
#pragma unroll
        for (int b = 0; b < EPT; ++b) {
            const int r = ty + a * TPB_DIM, c = tx + b * TPB_DIM;
            sD[r][c] = dist[base + (size_t)r * n + c];
            p[a][b] = path[base + (size_t)r * n + c];
        }
    __syncthreads();

    const unsigned int kbase = (unsigned int)K * TILE;
    for (int k = 0; k < TILE; ++k) {
#pragma unroll
        for (int a = 0; a < EPT; ++a) {
            const int r = ty + a * TPB_DIM;
            const unsigned int dik = sD[r][k];
#pragma unroll
            for (int b = 0; b < EPT; ++b) {
                const int c = tx + b * TPB_DIM;
                const unsigned int nd = dik + sD[k][c];
                if (nd < sD[r][c]) {
                    sD[r][c] = nd;
                    p[a][b] = kbase + k;
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int a = 0; a < EPT; ++a)
#pragma unroll
        for (int b = 0; b < EPT; ++b) {
            const int r = ty + a * TPB_DIM, c = tx + b * TPB_DIM;
            dist[base + (size_t)r * n + c] = sD[r][c];
            path[base + (size_t)r * n + c] = p[a][b];
        }
}

// Phase 2: tiles in pivot row (K,J) and pivot column (I,K)
__global__ void __launch_bounds__(TPB_DIM * TPB_DIM)
fwPhase2(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
         const int n, const int K) {
    const int T = blockIdx.x;
    if (T == K) return;
    const bool isRow = (blockIdx.y == 0);

    __shared__ unsigned int sP[TILE][SPAD];  // pivot tile
    __shared__ unsigned int sD[TILE][SPAD];  // tile being updated
    const int tx = threadIdx.x, ty = threadIdx.y;
    const size_t pBase = (size_t)K * TILE * n + (size_t)K * TILE;
    const size_t tBase = isRow ? ((size_t)K * TILE * n + (size_t)T * TILE)
                               : ((size_t)T * TILE * n + (size_t)K * TILE);

    unsigned int p[EPT][EPT];
#pragma unroll
    for (int a = 0; a < EPT; ++a)
#pragma unroll
        for (int b = 0; b < EPT; ++b) {
            const int r = ty + a * TPB_DIM, c = tx + b * TPB_DIM;
            sP[r][c] = dist[pBase + (size_t)r * n + c];
            sD[r][c] = dist[tBase + (size_t)r * n + c];
            p[a][b] = path[tBase + (size_t)r * n + c];
        }
    __syncthreads();

    const unsigned int kbase = (unsigned int)K * TILE;
    if (isRow) {
        // D[r][c] = min(D[r][c], P[r][k] + D[k][c])
        for (int k = 0; k < TILE; ++k) {
#pragma unroll
            for (int a = 0; a < EPT; ++a) {
                const int r = ty + a * TPB_DIM;
                const unsigned int dik = sP[r][k];
#pragma unroll
                for (int b = 0; b < EPT; ++b) {
                    const int c = tx + b * TPB_DIM;
                    const unsigned int nd = dik + sD[k][c];
                    if (nd < sD[r][c]) {
                        sD[r][c] = nd;
                        p[a][b] = kbase + k;
                    }
                }
            }
            __syncthreads();
        }
    } else {
        // D[r][c] = min(D[r][c], D[r][k] + P[k][c])
        for (int k = 0; k < TILE; ++k) {
#pragma unroll
            for (int a = 0; a < EPT; ++a) {
                const int r = ty + a * TPB_DIM;
                const unsigned int dik = sD[r][k];
#pragma unroll
                for (int b = 0; b < EPT; ++b) {
                    const int c = tx + b * TPB_DIM;
                    const unsigned int nd = dik + sP[k][c];
                    if (nd < sD[r][c]) {
                        sD[r][c] = nd;
                        p[a][b] = kbase + k;
                    }
                }
            }
            __syncthreads();
        }
    }

#pragma unroll
    for (int a = 0; a < EPT; ++a)
#pragma unroll
        for (int b = 0; b < EPT; ++b) {
            const int r = ty + a * TPB_DIM, c = tx + b * TPB_DIM;
            dist[tBase + (size_t)r * n + c] = sD[r][c];
            path[tBase + (size_t)r * n + c] = p[a][b];
        }
}

// Phase 3: all remaining tiles (I,J), independent given pivot row/column
__global__ void __launch_bounds__(TPB_DIM * TPB_DIM)
fwPhase3(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
         const int n, const int K) {
    const int J = blockIdx.x, I = blockIdx.y;
    if (I == K || J == K) return;

    __shared__ unsigned int sA[TILE][SPAD];  // tile (I,K)
    __shared__ unsigned int sB[TILE][TILE];  // tile (K,J)
    const int tx = threadIdx.x, ty = threadIdx.y;
    const size_t aBase = (size_t)I * TILE * n + (size_t)K * TILE;
    const size_t bBase = (size_t)K * TILE * n + (size_t)J * TILE;
    const size_t tBase = (size_t)I * TILE * n + (size_t)J * TILE;

    unsigned int d[EPT][EPT];
#pragma unroll
    for (int a = 0; a < EPT; ++a)
#pragma unroll
        for (int b = 0; b < EPT; ++b) {
            const int r = ty + a * TPB_DIM, c = tx + b * TPB_DIM;
            sA[r][c] = dist[aBase + (size_t)r * n + c];
            sB[r][c] = dist[bBase + (size_t)r * n + c];
            d[a][b] = dist[tBase + (size_t)r * n + c];
        }
    __syncthreads();

    // Track the last improving k (local index; -1 = no improvement)
    int pk[EPT][EPT];
#pragma unroll
    for (int a = 0; a < EPT; ++a)
#pragma unroll
        for (int b = 0; b < EPT; ++b) pk[a][b] = -1;

#pragma unroll 8
    for (int k = 0; k < TILE; ++k) {
        unsigned int av[EPT], bv[EPT];
#pragma unroll
        for (int a = 0; a < EPT; ++a) av[a] = sA[ty + a * TPB_DIM][k];
#pragma unroll
        for (int b = 0; b < EPT; ++b) bv[b] = sB[k][tx + b * TPB_DIM];
#pragma unroll
        for (int a = 0; a < EPT; ++a)
#pragma unroll
            for (int b = 0; b < EPT; ++b) {
                const unsigned int nd = av[a] + bv[b];
                if (nd < d[a][b]) {
                    d[a][b] = nd;
                    pk[a][b] = k;
                }
            }
    }

    const unsigned int kbase = (unsigned int)K * TILE;
#pragma unroll
    for (int a = 0; a < EPT; ++a)
#pragma unroll
        for (int b = 0; b < EPT; ++b) {
            if (pk[a][b] >= 0) {
                const int r = ty + a * TPB_DIM, c = tx + b * TPB_DIM;
                dist[tBase + (size_t)r * n + c] = d[a][b];
                path[tBase + (size_t)r * n + c] = kbase + (unsigned int)pk[a][b];
            }
        }
}

void floydWarshall(std::vector<unsigned int>& dist, 
                   std::vector<unsigned int>& path, 
                   const size_t numNodes) {
    if (numNodes == 0) return;

    // Pad the matrix to a multiple of TILE; padded nodes are unreachable
    // (INF distances), so they never produce an improvement.
    const size_t nTiles = (numNodes + TILE - 1) / TILE;
    const size_t np = nTiles * TILE;
    const size_t bytes = np * np * sizeof(unsigned int);

    unsigned int *dDist = nullptr, *dPath = nullptr;
    CUDA_CHECK(cudaMalloc(&dDist, bytes));
    CUDA_CHECK(cudaMalloc(&dPath, bytes));

    if (np != numNodes) {
        fillPaddedKernel<<<1024, 256>>>(dDist, dPath, np);
        CUDA_CHECK(cudaGetLastError());
    }
    const size_t hostPitch = numNodes * sizeof(unsigned int);
    const size_t devPitch = np * sizeof(unsigned int);
    CUDA_CHECK(cudaMemcpy2D(dDist, devPitch, dist.data(), hostPitch,
                            hostPitch, numNodes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy2D(dPath, devPitch, path.data(), hostPitch,
                            hostPitch, numNodes, cudaMemcpyHostToDevice));

    const dim3 threads(TPB_DIM, TPB_DIM);
    const dim3 grid2((unsigned int)nTiles, 2);
    const dim3 grid3((unsigned int)nTiles, (unsigned int)nTiles);
    const int n = (int)np;
    for (int K = 0; K < (int)nTiles; ++K) {
        fwPhase1<<<1, threads>>>(dDist, dPath, n, K);
        fwPhase2<<<grid2, threads>>>(dDist, dPath, n, K);
        fwPhase3<<<grid3, threads>>>(dDist, dPath, n, K);
    }
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy2D(dist.data(), hostPitch, dDist, devPitch,
                            hostPitch, numNodes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy2D(path.data(), hostPitch, dPath, devPitch,
                            hostPitch, numNodes, cudaMemcpyDeviceToHost));

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
    CUDA_CHECK(cudaFree(0));
    {
        cudaFuncAttributes attr;
        CUDA_CHECK(cudaFuncGetAttributes(&attr, fillPaddedKernel));
        CUDA_CHECK(cudaFuncGetAttributes(&attr, fwPhase1));
        CUDA_CHECK(cudaFuncGetAttributes(&attr, fwPhase2));
        CUDA_CHECK(cudaFuncGetAttributes(&attr, fwPhase3));
    }

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
