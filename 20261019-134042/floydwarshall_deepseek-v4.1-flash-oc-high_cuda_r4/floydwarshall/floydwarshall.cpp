#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;
constexpr int TILE = 32;

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _err = (call);                                              \
        if (_err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error: %s (%s:%d)\n",                         \
                    cudaGetErrorString(_err), __FILE__, __LINE__);              \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

// Index calculation for flattened 2D array
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

__global__ void fwDiagonalKernel(unsigned int* __restrict__ dist,
                                 unsigned int* __restrict__ path,
                                 const int n, const int k0) {
    __shared__ unsigned int tileD[TILE][TILE + 1];
    __shared__ unsigned int tileP[TILE][TILE + 1];

    const int c = threadIdx.x;
    const int gc = k0 + c;

    for (int r = threadIdx.y; r < TILE; r += blockDim.y) {
        const int gr = k0 + r;
        if (gr < n && gc < n) {
            const size_t off = static_cast<size_t>(gr) * n + gc;
            tileD[r][c] = dist[off];
            tileP[r][c] = path[off];
        } else {
            tileD[r][c] = INF;
            tileP[r][c] = 0;
        }
    }
    __syncthreads();

    for (int kk = 0; kk < TILE; ++kk) {
        for (int r = threadIdx.y; r < TILE; r += blockDim.y) {
            const int gr = k0 + r;
            if (gr < n && gc < n) {
                const unsigned int nd = tileD[r][kk] + tileD[kk][c];
                if (nd < tileD[r][c]) {
                    tileD[r][c] = nd;
                    tileP[r][c] = k0 + kk;
                }
            }
        }
        __syncthreads();
    }

    for (int r = threadIdx.y; r < TILE; r += blockDim.y) {
        const int gr = k0 + r;
        if (gr < n && gc < n) {
            const size_t off = static_cast<size_t>(gr) * n + gc;
            dist[off] = tileD[r][c];
            path[off] = tileP[r][c];
        }
    }
}

__global__ void fwCrossKernel(unsigned int* __restrict__ dist,
                              unsigned int* __restrict__ path,
                              const int n, const int k0, const int kt,
                              const int tiles) {
    __shared__ unsigned int diag[TILE][TILE + 1];
    __shared__ unsigned int blk[TILE][TILE + 1];

    const int c = threadIdx.x;
    const int gcDiag = k0 + c;

    for (int r = threadIdx.y; r < TILE; r += blockDim.y) {
        const int gr = k0 + r;
        if (gr < n && gcDiag < n) {
            diag[r][c] = dist[static_cast<size_t>(gr) * n + gcDiag];
        } else {
            diag[r][c] = INF;
        }
    }

    const int crossCount = tiles - 1;
    const int id = blockIdx.x;
    const bool rowBlock = id < crossCount;

    int i0;
    int j0;
    if (rowBlock) {
        const int t = id;
        const int jt = t < kt ? t : t + 1;
        i0 = k0;
        j0 = jt * TILE;
    } else {
        const int t = id - crossCount;
        const int it = t < kt ? t : t + 1;
        i0 = it * TILE;
        j0 = k0;
    }

    for (int r = threadIdx.y; r < TILE; r += blockDim.y) {
        const int gr = i0 + r;
        const int gc = j0 + c;
        if (gr < n && gc < n) {
            const size_t off = static_cast<size_t>(gr) * n + gc;
            blk[r][c] = dist[off];
        } else {
            blk[r][c] = INF;
        }
    }
    __syncthreads();

    for (int r = threadIdx.y; r < TILE; r += blockDim.y) {
        const int gr = i0 + r;
        const int gc = j0 + c;
        if (gr < n && gc < n) {
            const size_t off = static_cast<size_t>(gr) * n + gc;
            unsigned int best = blk[r][c];
            unsigned int bestK = path[off];

            if (rowBlock) {
                #pragma unroll
                for (int kk = 0; kk < TILE; ++kk) {
                    const unsigned int cand = diag[r][kk] + blk[kk][c];
                    if (cand < best) {
                        best = cand;
                        bestK = k0 + kk;
                    }
                }
            } else {
                #pragma unroll
                for (int kk = 0; kk < TILE; ++kk) {
                    const unsigned int cand = blk[r][kk] + diag[kk][c];
                    if (cand < best) {
                        best = cand;
                        bestK = k0 + kk;
                    }
                }
            }

            dist[off] = best;
            path[off] = bestK;
        }
    }
}

__global__ void fwInnerKernel(unsigned int* __restrict__ dist,
                              unsigned int* __restrict__ path,
                              const int n, const int k0, const int kt,
                              const int tiles) {
    __shared__ unsigned int left[TILE][TILE + 1];
    __shared__ unsigned int right[TILE][TILE + 1];
    __shared__ unsigned int blk[TILE][TILE + 1];

    const int span = tiles - 1;
    const int bid = blockIdx.x;
    int it = bid / span;
    const int rem = bid - it * span;
    if (it >= kt) {
        ++it;
    }
    const int jt = rem >= kt ? rem + 1 : rem;

    const int i0 = it * TILE;
    const int j0 = jt * TILE;
    const int c = threadIdx.x;

    for (int r = threadIdx.y; r < TILE; r += blockDim.y) {
        const int grL = i0 + r;
        const int gcL = k0 + c;
        if (grL < n && gcL < n) {
            left[r][c] = dist[static_cast<size_t>(grL) * n + gcL];
        } else {
            left[r][c] = INF;
        }

        const int grR = k0 + r;
        const int gcR = j0 + c;
        if (grR < n && gcR < n) {
            right[r][c] = dist[static_cast<size_t>(grR) * n + gcR];
        } else {
            right[r][c] = INF;
        }

        const int grB = i0 + r;
        const int gcB = j0 + c;
        if (grB < n && gcB < n) {
            const size_t off = static_cast<size_t>(grB) * n + gcB;
            blk[r][c] = dist[off];
        } else {
            blk[r][c] = INF;
        }
    }
    __syncthreads();

    for (int r = threadIdx.y; r < TILE; r += blockDim.y) {
        const int gr = i0 + r;
        const int gc = j0 + c;
        if (gr < n && gc < n) {
            const size_t off = static_cast<size_t>(gr) * n + gc;
            unsigned int best = blk[r][c];
            unsigned int bestK = path[off];

            #pragma unroll
            for (int kk = 0; kk < TILE; ++kk) {
                const unsigned int cand = left[r][kk] + right[kk][c];
                if (cand < best) {
                    best = cand;
                    bestK = k0 + kk;
                }
            }

            dist[off] = best;
            path[off] = bestK;
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

void floydWarshall(unsigned int* d_dist, 
                   unsigned int* d_path, 
                   const size_t numNodes) {
    if (numNodes == 0) {
        return;
    }

    const int n = static_cast<int>(numNodes);
    const int tiles = (n + TILE - 1) / TILE;
    const int blockRows = n <= 1536 ? 16 : 8;
    const dim3 block(TILE, blockRows);

    for (int kt = 0; kt < tiles; ++kt) {
        const int k0 = kt * TILE;
        fwDiagonalKernel<<<1, block>>>(d_dist, d_path, n, k0);

        if (tiles > 1) {
            const int crossBlocks = 2 * (tiles - 1);
            fwCrossKernel<<<crossBlocks, block>>>(d_dist, d_path, n, k0, kt, tiles);

            const int innerBlocks = (tiles - 1) * (tiles - 1);
            fwInnerKernel<<<innerBlocks, block>>>(d_dist, d_path, n, k0, kt, tiles);
        }
    }

    CUDA_CHECK(cudaGetLastError());
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

    if (numNodes > static_cast<size_t>(INT_MAX)) {
        fprintf(stderr, "Number of nodes is too large\n");
        return 1;
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

    const size_t bytes = numNodes * numNodes * sizeof(unsigned int);
    unsigned int* d_dist = nullptr;
    unsigned int* d_path = nullptr;
    if (bytes > 0) {
        CUDA_CHECK(cudaMalloc(&d_dist, bytes));
        CUDA_CHECK(cudaMalloc(&d_path, bytes));
        CUDA_CHECK(cudaMemcpy(d_dist, dist.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_path, path.data(), bytes, cudaMemcpyHostToDevice));
    }
    
    // Run Floyd-Warshall
    printf("Computing shortest paths...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    floydWarshall(d_dist, d_path, numNodes);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::milli>(end - start);
    
    if (bytes > 0) {
        CUDA_CHECK(cudaMemcpy(dist.data(), d_dist, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(path.data(), d_path, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_dist));
        CUDA_CHECK(cudaFree(d_path));
    }
    
    printf("Computation time: %.3f ms\n", duration.count());
    
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
