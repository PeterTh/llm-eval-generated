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

// Blocked Floyd-Warshall: tile size B, thread block B x RPT-row-groups
constexpr int B = 32;
constexpr int TY = 8;
constexpr int RPT = B / TY;

__global__ void fillKernel(unsigned int* dist, const size_t np) {
    const size_t total = np * np;
    for (size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x; t < total;
         t += (size_t)gridDim.x * blockDim.x) {
        dist[t] = (t / np == t % np) ? 0u : INF;
    }
}

// Phase 1: diagonal tile
__global__ void fwPhase1(unsigned int* dist, unsigned int* path, const int np, const int kb) {
    __shared__ unsigned int sd[B][B];
    __shared__ unsigned int sp[B][B];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const size_t base = (size_t)kb * B * np + (size_t)kb * B;
#pragma unroll
    for (int r = 0; r < RPT; ++r) {
        const int i = ty + r * TY;
        sd[i][tx] = dist[base + (size_t)i * np + tx];
        sp[i][tx] = path[base + (size_t)i * np + tx];
    }
    for (int k = 0; k < B; ++k) {
        __syncthreads();
#pragma unroll
        for (int r = 0; r < RPT; ++r) {
            const int i = ty + r * TY;
            const unsigned int nd = sd[i][k] + sd[k][tx];
            if (nd < sd[i][tx]) {
                sd[i][tx] = nd;
                sp[i][tx] = kb * B + k;
            }
        }
    }
    __syncthreads();
#pragma unroll
    for (int r = 0; r < RPT; ++r) {
        const int i = ty + r * TY;
        dist[base + (size_t)i * np + tx] = sd[i][tx];
        path[base + (size_t)i * np + tx] = sp[i][tx];
    }
}

// Phase 2: tiles in row kb (blockIdx.y==0) and column kb (blockIdx.y==1)
__global__ void fwPhase2(unsigned int* dist, unsigned int* path, const int np, const int kb) {
    const int ob = blockIdx.x;
    if (ob == kb) return;
    __shared__ unsigned int sdiag[B][B];
    __shared__ unsigned int sd[B][B];
    __shared__ unsigned int sp[B][B];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const bool isRow = (blockIdx.y == 0);
    const size_t dbase = (size_t)kb * B * np + (size_t)kb * B;
    const size_t base = isRow ? (size_t)kb * B * np + (size_t)ob * B
                              : (size_t)ob * B * np + (size_t)kb * B;
#pragma unroll
    for (int r = 0; r < RPT; ++r) {
        const int i = ty + r * TY;
        sdiag[i][tx] = dist[dbase + (size_t)i * np + tx];
        sd[i][tx] = dist[base + (size_t)i * np + tx];
        sp[i][tx] = path[base + (size_t)i * np + tx];
    }
    for (int k = 0; k < B; ++k) {
        __syncthreads();
#pragma unroll
        for (int r = 0; r < RPT; ++r) {
            const int i = ty + r * TY;
            unsigned int nd;
            if (isRow) nd = sdiag[i][k] + sd[k][tx];
            else       nd = sd[i][k] + sdiag[k][tx];
            if (nd < sd[i][tx]) {
                sd[i][tx] = nd;
                sp[i][tx] = kb * B + k;
            }
        }
    }
    __syncthreads();
#pragma unroll
    for (int r = 0; r < RPT; ++r) {
        const int i = ty + r * TY;
        dist[base + (size_t)i * np + tx] = sd[i][tx];
        path[base + (size_t)i * np + tx] = sp[i][tx];
    }
}

// Phase 3: all remaining tiles
__global__ void __launch_bounds__(B * TY)
fwPhase3(unsigned int* dist, unsigned int* path, const int np, const int kb) {
    const int jb = blockIdx.x, ib = blockIdx.y;
    if (ib == kb || jb == kb) return;
    __shared__ unsigned int sa[B][B + 1];  // column tile (ib,kb)
    __shared__ unsigned int sb[B][B];      // row tile (kb,jb)
    const int tx = threadIdx.x, ty = threadIdx.y;
    const size_t abase = (size_t)ib * B * np + (size_t)kb * B;
    const size_t bbase = (size_t)kb * B * np + (size_t)jb * B;
    const size_t cbase = (size_t)ib * B * np + (size_t)jb * B;
    unsigned int d[RPT], p[RPT];
#pragma unroll
    for (int r = 0; r < RPT; ++r) {
        const int i = ty + r * TY;
        sa[i][tx] = dist[abase + (size_t)i * np + tx];
        sb[i][tx] = dist[bbase + (size_t)i * np + tx];
        d[r] = dist[cbase + (size_t)i * np + tx];
        p[r] = path[cbase + (size_t)i * np + tx];
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < B; ++k) {
        const unsigned int bv = sb[k][tx];
#pragma unroll
        for (int r = 0; r < RPT; ++r) {
            const unsigned int nd = sa[ty + r * TY][k] + bv;
            if (nd < d[r]) {
                d[r] = nd;
                p[r] = kb * B + k;
            }
        }
    }
#pragma unroll
    for (int r = 0; r < RPT; ++r) {
        const int i = ty + r * TY;
        dist[cbase + (size_t)i * np + tx] = d[r];
        path[cbase + (size_t)i * np + tx] = p[r];
    }
}

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes) {
    if (numNodes == 0) return;
    const int nb = (int)((numNodes + B - 1) / B);
    const int np = nb * B;
    const size_t bytes = (size_t)np * np * sizeof(unsigned int);

    unsigned int *dDist = nullptr, *dPath = nullptr;
    CUDA_CHECK(cudaMalloc(&dDist, bytes));
    CUDA_CHECK(cudaMalloc(&dPath, bytes));

    // Padding nodes: INF distances (never improve real entries), zero diagonal
    fillKernel<<<1024, 256>>>(dDist, np);
    CUDA_CHECK(cudaMemset(dPath, 0, bytes));
    CUDA_CHECK(cudaMemcpy2D(dDist, np * sizeof(unsigned int), dist.data(),
                            numNodes * sizeof(unsigned int), numNodes * sizeof(unsigned int),
                            numNodes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy2D(dPath, np * sizeof(unsigned int), path.data(),
                            numNodes * sizeof(unsigned int), numNodes * sizeof(unsigned int),
                            numNodes, cudaMemcpyHostToDevice));

    const dim3 threads(B, TY);
    const dim3 grid2(nb, 2);
    const dim3 grid3(nb, nb);
    for (int kb = 0; kb < nb; ++kb) {
        fwPhase1<<<1, threads>>>(dDist, dPath, np, kb);
        fwPhase2<<<grid2, threads>>>(dDist, dPath, np, kb);
        fwPhase3<<<grid3, threads>>>(dDist, dPath, np, kb);
    }
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy2D(dist.data(), numNodes * sizeof(unsigned int), dDist,
                            np * sizeof(unsigned int), numNodes * sizeof(unsigned int),
                            numNodes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy2D(path.data(), numNodes * sizeof(unsigned int), dPath,
                            np * sizeof(unsigned int), numNodes * sizeof(unsigned int),
                            numNodes, cudaMemcpyDeviceToHost));
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
    CUDA_CHECK(cudaFree(0));  // context warm-up
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
