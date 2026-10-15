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

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), \
                    __FILE__, __LINE__);                                          \
            exit(1);                                                              \
        }                                                                         \
    } while (0)

constexpr int TILE = 32;
constexpr int ROWS = 8;  // thread block is TILE x ROWS; each thread handles TILE/ROWS rows

// Phase 1: update the diagonal tile (K,K) entirely in shared memory.
__global__ void fwDiagonal(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
                           const int np, const int K) {
    __shared__ unsigned int d[TILE][TILE + 1];
    __shared__ unsigned int p[TILE][TILE + 1];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const size_t base = (size_t)K * TILE * np + (size_t)K * TILE;
    for (int r = ty; r < TILE; r += ROWS) {
        d[r][tx] = dist[base + (size_t)r * np + tx];
        p[r][tx] = path[base + (size_t)r * np + tx];
    }
    __syncthreads();
    for (int k = 0; k < TILE; ++k) {
        unsigned int dik[TILE / ROWS], dkj = d[k][tx];
        for (int c = 0; c < TILE / ROWS; ++c) dik[c] = d[ty + c * ROWS][k];
        __syncthreads();
        for (int c = 0; c < TILE / ROWS; ++c) {
            const int r = ty + c * ROWS;
            const unsigned int nd = dik[c] + dkj;
            if (nd < d[r][tx]) { d[r][tx] = nd; p[r][tx] = K * TILE + k; }
        }
        __syncthreads();
    }
    for (int r = ty; r < TILE; r += ROWS) {
        dist[base + (size_t)r * np + tx] = d[r][tx];
        path[base + (size_t)r * np + tx] = p[r][tx];
    }
}

// Phase 2: update row tiles (K,J) (blockIdx.y==0) and column tiles (I,K) (blockIdx.y==1).
__global__ void fwRowCol(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
                         const int np, const int K) {
    __shared__ unsigned int dg[TILE][TILE + 1];
    __shared__ unsigned int d[TILE][TILE + 1];
    __shared__ unsigned int p[TILE][TILE + 1];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int nt = np / TILE;
    int t = blockIdx.x;
    if (t >= K) ++t;  // skip the diagonal tile
    if (t >= nt) return;
    const bool isRow = (blockIdx.y == 0);
    const size_t dbase = (size_t)K * TILE * np + (size_t)K * TILE;
    const size_t base = isRow ? (size_t)K * TILE * np + (size_t)t * TILE
                              : (size_t)t * TILE * np + (size_t)K * TILE;
    for (int r = ty; r < TILE; r += ROWS) {
        dg[r][tx] = dist[dbase + (size_t)r * np + tx];
        d[r][tx] = dist[base + (size_t)r * np + tx];
        p[r][tx] = path[base + (size_t)r * np + tx];
    }
    __syncthreads();
    for (int k = 0; k < TILE; ++k) {
        unsigned int a[TILE / ROWS], b[TILE / ROWS];
        for (int c = 0; c < TILE / ROWS; ++c) {
            const int r = ty + c * ROWS;
            if (isRow) { a[c] = dg[r][k]; b[c] = d[k][tx]; }
            else       { a[c] = d[r][k];  b[c] = dg[k][tx]; }
        }
        __syncthreads();
        for (int c = 0; c < TILE / ROWS; ++c) {
            const int r = ty + c * ROWS;
            const unsigned int nd = a[c] + b[c];
            if (nd < d[r][tx]) { d[r][tx] = nd; p[r][tx] = K * TILE + k; }
        }
        __syncthreads();
    }
    for (int r = ty; r < TILE; r += ROWS) {
        dist[base + (size_t)r * np + tx] = d[r][tx];
        path[base + (size_t)r * np + tx] = p[r][tx];
    }
}

// Phase 3: update all remaining tiles (I,J), I!=K, J!=K.
__global__ void fwRest(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
                       const int np, const int K) {
    __shared__ unsigned int colT[TILE][TILE + 1];  // tile (I,K)
    __shared__ unsigned int rowT[TILE][TILE];      // tile (K,J)
    const int tx = threadIdx.x, ty = threadIdx.y;
    int I = blockIdx.y, J = blockIdx.x;
    if (I >= K) ++I;
    if (J >= K) ++J;
    const size_t base = (size_t)I * TILE * np + (size_t)J * TILE;
    const size_t cbase = (size_t)I * TILE * np + (size_t)K * TILE;
    const size_t rbase = (size_t)K * TILE * np + (size_t)J * TILE;
    unsigned int dv[TILE / ROWS], pv[TILE / ROWS];
    for (int c = 0; c < TILE / ROWS; ++c) {
        const int r = ty + c * ROWS;
        colT[r][tx] = dist[cbase + (size_t)r * np + tx];
        rowT[r][tx] = dist[rbase + (size_t)r * np + tx];
        dv[c] = dist[base + (size_t)r * np + tx];
        pv[c] = path[base + (size_t)r * np + tx];
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < TILE; ++k) {
        const unsigned int b = rowT[k][tx];
#pragma unroll
        for (int c = 0; c < TILE / ROWS; ++c) {
            const unsigned int nd = colT[ty + c * ROWS][k] + b;
            if (nd < dv[c]) { dv[c] = nd; pv[c] = K * TILE + k; }
        }
    }
    for (int c = 0; c < TILE / ROWS; ++c) {
        const int r = ty + c * ROWS;
        dist[base + (size_t)r * np + tx] = dv[c];
        path[base + (size_t)r * np + tx] = pv[c];
    }
}

// Blocked Floyd-Warshall on the GPU. Each cell still sees intermediate nodes k in
// increasing order and uses strict '<', so results match the sequential algorithm.
void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes) {
    if (numNodes == 0) return;
    const int n = (int)numNodes;
    const int nt = (n + TILE - 1) / TILE;
    const int np = nt * TILE;
    const size_t pitchBytes = (size_t)np * sizeof(unsigned int);
    const size_t rowBytes = (size_t)n * sizeof(unsigned int);

    unsigned int *dDist, *dPath;
    CUDA_CHECK(cudaMalloc(&dDist, pitchBytes * np));
    CUDA_CHECK(cudaMalloc(&dPath, pitchBytes * np));
    // Padding: INF distances (never improve real cells), zero diagonal
    {
        std::vector<unsigned int> pad((size_t)np * np, INF);
        for (int i = 0; i < np; ++i) pad[(size_t)i * np + i] = 0;
        CUDA_CHECK(cudaMemcpy(dDist, pad.data(), pitchBytes * np, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(dPath, 0, pitchBytes * np));
    }
    CUDA_CHECK(cudaMemcpy2D(dDist, pitchBytes, dist.data(), rowBytes, rowBytes, n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy2D(dPath, pitchBytes, path.data(), rowBytes, rowBytes, n, cudaMemcpyHostToDevice));

    const dim3 block(TILE, ROWS);
    for (int K = 0; K < nt; ++K) {
        fwDiagonal<<<1, block>>>(dDist, dPath, np, K);
        if (nt > 1) {
            fwRowCol<<<dim3(nt - 1, 2), block>>>(dDist, dPath, np, K);
            fwRest<<<dim3(nt - 1, nt - 1), block>>>(dDist, dPath, np, K);
        }
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy2D(dist.data(), rowBytes, dDist, pitchBytes, rowBytes, n, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy2D(path.data(), rowBytes, dPath, pitchBytes, rowBytes, n, cudaMemcpyDeviceToHost));
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
    
    // Run Floyd-Warshall
    printf("Computing shortest paths...\n");
    CUDA_CHECK(cudaFree(0));  // create the CUDA context outside the timed region
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
