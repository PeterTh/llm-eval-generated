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

static inline void cudaCheck(cudaError_t err, const char* call, const char* file, int line) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s:%d: %s failed with %s\n", file, line, call,
                     cudaGetErrorString(err));
        std::exit(1);
    }
}
#define CUDA_CHECK(x) cudaCheck((x), #x, __FILE__, __LINE__)

constexpr int FW_TILE = 32;
constexpr int FW_TX = 32;
constexpr int FW_TY = 8;
constexpr int FW_ROWS_PER_THREAD = FW_TILE / FW_TY;

__global__ void fw_phase1(unsigned int* __restrict__ dist,
                          unsigned int* __restrict__ path,
                          size_t n,
                          int kBlock) {
    __shared__ unsigned int sDist[FW_TILE][FW_TILE + 1];

    const int tx = (int)threadIdx.x; // 0..31
    const int ty = (int)threadIdx.y; // 0..7

    const int base = kBlock * FW_TILE;
    unsigned int p[FW_ROWS_PER_THREAD];

    #pragma unroll
    for (int r = 0; r < FW_ROWS_PER_THREAD; ++r) {
        const int row = ty + r * FW_TY;
        const int i = base + row;
        const int j = base + tx;
        unsigned int v = INF;
        unsigned int pv = 0;
        if ((size_t)i < n && (size_t)j < n) {
            const size_t idx = (size_t)i * n + (size_t)j;
            v = dist[idx];
            pv = path[idx];
        }
        sDist[row][tx] = v;
        p[r] = pv;
    }

    __syncthreads();

    for (int kk = 0; kk < FW_TILE; ++kk) {
        const int k = base + kk;
        if ((size_t)k >= n) break;

        #pragma unroll
        for (int r = 0; r < FW_ROWS_PER_THREAD; ++r) {
            const int row = ty + r * FW_TY;
            const unsigned int dik = sDist[row][kk];
            const unsigned int dkj = sDist[kk][tx];
            const unsigned int nd = dik + dkj;
            if (nd < sDist[row][tx]) {
                sDist[row][tx] = nd;
                p[r] = (unsigned int)k;
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int r = 0; r < FW_ROWS_PER_THREAD; ++r) {
        const int row = ty + r * FW_TY;
        const int i = base + row;
        const int j = base + tx;
        if ((size_t)i < n && (size_t)j < n) {
            const size_t idx = (size_t)i * n + (size_t)j;
            dist[idx] = sDist[row][tx];
            path[idx] = p[r];
        }
    }
}

__global__ void fw_phase2_row(unsigned int* __restrict__ dist,
                             unsigned int* __restrict__ path,
                             size_t n,
                             int kBlock) {
    const int jBlock = (int)blockIdx.x;
    if (jBlock == kBlock) return;

    __shared__ unsigned int sPivot[FW_TILE][FW_TILE + 1];
    __shared__ unsigned int sRow[FW_TILE][FW_TILE + 1];

    const int tx = (int)threadIdx.x;
    const int ty = (int)threadIdx.y;

    const int kBase = kBlock * FW_TILE;
    const int jBase = jBlock * FW_TILE;

    unsigned int p[FW_ROWS_PER_THREAD];

    #pragma unroll
    for (int r = 0; r < FW_ROWS_PER_THREAD; ++r) {
        const int row = ty + r * FW_TY;

        const int pi = kBase + row;
        const int pj = kBase + tx;
        unsigned int pv = INF;
        if ((size_t)pi < n && (size_t)pj < n) {
            pv = dist[(size_t)pi * n + (size_t)pj];
        }
        sPivot[row][tx] = pv;

        const int ri = kBase + row;
        const int rj = jBase + tx;
        unsigned int rv = INF;
        unsigned int rpath = 0;
        if ((size_t)ri < n && (size_t)rj < n) {
            const size_t idx = (size_t)ri * n + (size_t)rj;
            rv = dist[idx];
            rpath = path[idx];
        }
        sRow[row][tx] = rv;
        p[r] = rpath;
    }

    __syncthreads();

    for (int kk = 0; kk < FW_TILE; ++kk) {
        const int k = kBase + kk;
        if ((size_t)k >= n) break;

        #pragma unroll
        for (int r = 0; r < FW_ROWS_PER_THREAD; ++r) {
            const int row = ty + r * FW_TY;
            const unsigned int nd = sPivot[row][kk] + sRow[kk][tx];
            if (nd < sRow[row][tx]) {
                sRow[row][tx] = nd;
                p[r] = (unsigned int)k;
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int r = 0; r < FW_ROWS_PER_THREAD; ++r) {
        const int row = ty + r * FW_TY;
        const int i = kBase + row;
        const int j = jBase + tx;
        if ((size_t)i < n && (size_t)j < n) {
            const size_t idx = (size_t)i * n + (size_t)j;
            dist[idx] = sRow[row][tx];
            path[idx] = p[r];
        }
    }
}

__global__ void fw_phase2_col(unsigned int* __restrict__ dist,
                             unsigned int* __restrict__ path,
                             size_t n,
                             int kBlock) {
    const int iBlock = (int)blockIdx.x;
    if (iBlock == kBlock) return;

    __shared__ unsigned int sPivot[FW_TILE][FW_TILE + 1];
    __shared__ unsigned int sCol[FW_TILE][FW_TILE + 1];

    const int tx = (int)threadIdx.x;
    const int ty = (int)threadIdx.y;

    const int kBase = kBlock * FW_TILE;
    const int iBase = iBlock * FW_TILE;

    unsigned int p[FW_ROWS_PER_THREAD];

    #pragma unroll
    for (int r = 0; r < FW_ROWS_PER_THREAD; ++r) {
        const int row = ty + r * FW_TY;

        const int pi = kBase + row;
        const int pj = kBase + tx;
        unsigned int pv = INF;
        if ((size_t)pi < n && (size_t)pj < n) {
            pv = dist[(size_t)pi * n + (size_t)pj];
        }
        sPivot[row][tx] = pv;

        const int ci = iBase + row;
        const int cj = kBase + tx;
        unsigned int cv = INF;
        unsigned int cpath = 0;
        if ((size_t)ci < n && (size_t)cj < n) {
            const size_t idx = (size_t)ci * n + (size_t)cj;
            cv = dist[idx];
            cpath = path[idx];
        }
        sCol[row][tx] = cv;
        p[r] = cpath;
    }

    __syncthreads();

    for (int kk = 0; kk < FW_TILE; ++kk) {
        const int k = kBase + kk;
        if ((size_t)k >= n) break;

        #pragma unroll
        for (int r = 0; r < FW_ROWS_PER_THREAD; ++r) {
            const int row = ty + r * FW_TY;
            const unsigned int nd = sCol[row][kk] + sPivot[kk][tx];
            if (nd < sCol[row][tx]) {
                sCol[row][tx] = nd;
                p[r] = (unsigned int)k;
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int r = 0; r < FW_ROWS_PER_THREAD; ++r) {
        const int row = ty + r * FW_TY;
        const int i = iBase + row;
        const int j = kBase + tx;
        if ((size_t)i < n && (size_t)j < n) {
            const size_t idx = (size_t)i * n + (size_t)j;
            dist[idx] = sCol[row][tx];
            path[idx] = p[r];
        }
    }
}

__global__ void fw_phase3(unsigned int* __restrict__ dist,
                          unsigned int* __restrict__ path,
                          size_t n,
                          int kBlock) {
    const int jBlock = (int)blockIdx.x;
    const int iBlock = (int)blockIdx.y;
    if (iBlock == kBlock || jBlock == kBlock) return;

    __shared__ unsigned int sIk[FW_TILE][FW_TILE + 1];
    __shared__ unsigned int sKj[FW_TILE][FW_TILE + 1];

    const int tx = (int)threadIdx.x;
    const int ty = (int)threadIdx.y;

    const int kBase = kBlock * FW_TILE;
    const int iBase = iBlock * FW_TILE;
    const int jBase = jBlock * FW_TILE;

    unsigned int dij[FW_ROWS_PER_THREAD];
    unsigned int p[FW_ROWS_PER_THREAD];

    #pragma unroll
    for (int r = 0; r < FW_ROWS_PER_THREAD; ++r) {
        const int row = ty + r * FW_TY;

        const int ik_i = iBase + row;
        const int ik_j = kBase + tx;
        unsigned int ikv = INF;
        if ((size_t)ik_i < n && (size_t)ik_j < n) {
            ikv = dist[(size_t)ik_i * n + (size_t)ik_j];
        }
        sIk[row][tx] = ikv;

        const int kj_i = kBase + row;
        const int kj_j = jBase + tx;
        unsigned int kjv = INF;
        if ((size_t)kj_i < n && (size_t)kj_j < n) {
            kjv = dist[(size_t)kj_i * n + (size_t)kj_j];
        }
        sKj[row][tx] = kjv;

        const int ij_i = iBase + row;
        const int ij_j = jBase + tx;
        unsigned int ijv = INF;
        unsigned int ijp = 0;
        if ((size_t)ij_i < n && (size_t)ij_j < n) {
            const size_t idx = (size_t)ij_i * n + (size_t)ij_j;
            ijv = dist[idx];
            ijp = path[idx];
        }
        dij[r] = ijv;
        p[r] = ijp;
    }

    __syncthreads();

    for (int kk = 0; kk < FW_TILE; ++kk) {
        const int k = kBase + kk;
        if ((size_t)k >= n) break;

        const unsigned int dkj = sKj[kk][tx];
        #pragma unroll
        for (int r = 0; r < FW_ROWS_PER_THREAD; ++r) {
            const int row = ty + r * FW_TY;
            const unsigned int nd = sIk[row][kk] + dkj;
            if (nd < dij[r]) {
                dij[r] = nd;
                p[r] = (unsigned int)k;
            }
        }
    }

    #pragma unroll
    for (int r = 0; r < FW_ROWS_PER_THREAD; ++r) {
        const int row = ty + r * FW_TY;
        const int i = iBase + row;
        const int j = jBase + tx;
        if ((size_t)i < n && (size_t)j < n) {
            const size_t idx = (size_t)i * n + (size_t)j;
            dist[idx] = dij[r];
            path[idx] = p[r];
        }
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

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes) {
    const size_t n = numNodes;
    const size_t bytes = n * n * sizeof(unsigned int);

    unsigned int* dDist = nullptr;
    unsigned int* dPath = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&dDist, bytes));
    CUDA_CHECK(cudaMalloc((void**)&dPath, bytes));

    CUDA_CHECK(cudaMemcpy(dDist, dist.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dPath, path.data(), bytes, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));

    const int numBlocks = (int)((n + (size_t)FW_TILE - 1) / (size_t)FW_TILE);
    const dim3 threads(FW_TX, FW_TY, 1);

    for (int kBlock = 0; kBlock < numBlocks; ++kBlock) {
        fw_phase1<<<1, threads>>>(dDist, dPath, n, kBlock);
        CUDA_CHECK(cudaGetLastError());

        fw_phase2_row<<<numBlocks, threads>>>(dDist, dPath, n, kBlock);
        CUDA_CHECK(cudaGetLastError());

        fw_phase2_col<<<numBlocks, threads>>>(dDist, dPath, n, kBlock);
        CUDA_CHECK(cudaGetLastError());

        const dim3 grid3((unsigned int)numBlocks, (unsigned int)numBlocks, 1);
        fw_phase3<<<grid3, threads>>>(dDist, dPath, n, kBlock);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(dist.data(), dDist, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(path.data(), dPath, bytes, cudaMemcpyDeviceToHost));

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
