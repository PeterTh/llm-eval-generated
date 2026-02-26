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

static inline void cuda_check(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error %s:%d: %s\n", file, line, cudaGetErrorString(err));
        std::fflush(stderr);
        std::exit(1);
    }
}
#define CUDA_CHECK(x) cuda_check((x), __FILE__, __LINE__)

// Index calculation for flattened 2D array (dist[row][col] is stored at idx2(col,row,n))
__host__ __device__ __forceinline__ constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

template <int B>
__global__ void fw_diag(unsigned int* __restrict__ dist,
                        unsigned int* __restrict__ path,
                        const int n,
                        const int kk) {
    __shared__ unsigned int s[B][B + 1];

    const int col = static_cast<int>(threadIdx.x);
    const int ty = static_cast<int>(threadIdx.y);

    unsigned int p[4];

#pragma unroll
    for (int m = 0; m < 4; ++m) {
        const int row = ty + m * 8;
        const int gr = kk + row;
        const int gc = kk + col;
        unsigned int v = INF;
        unsigned int pv = 0;
        if (gr < n && gc < n) {
            const size_t id = idx2(static_cast<size_t>(gc), static_cast<size_t>(gr), static_cast<size_t>(n));
            v = dist[id];
            pv = path[id];
        }
        s[row][col] = v;
        p[m] = pv;
    }

    __syncthreads();

#pragma unroll
    for (int t = 0; t < B; ++t) {
        const unsigned int b = s[t][col];
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            const int row = ty + m * 8;
            const unsigned int a = s[row][t];
            const unsigned int nd = a + b;
            if (nd < s[row][col]) {
                s[row][col] = nd;
                p[m] = static_cast<unsigned int>(kk + t);
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int m = 0; m < 4; ++m) {
        const int row = ty + m * 8;
        const int gr = kk + row;
        const int gc = kk + col;
        if (gr < n && gc < n) {
            const size_t id = idx2(static_cast<size_t>(gc), static_cast<size_t>(gr), static_cast<size_t>(n));
            dist[id] = s[row][col];
            path[id] = p[m];
        }
    }
}

template <int B>
__global__ void fw_row(unsigned int* __restrict__ dist,
                       unsigned int* __restrict__ path,
                       const int n,
                       const int kk,
                       const int kkTile,
                       const int numTiles) {
    __shared__ unsigned int diag[B][B + 1];
    __shared__ unsigned int tile[B][B + 1];

    const int col = static_cast<int>(threadIdx.x);
    const int ty = static_cast<int>(threadIdx.y);

    const int tIdx = static_cast<int>(blockIdx.x);
    const int jTile = (tIdx < kkTile) ? tIdx : (tIdx + 1);

    const int baseRow = kk;
    const int baseCol = jTile * B;

    unsigned int p[4];

#pragma unroll
    for (int m = 0; m < 4; ++m) {
        const int row = ty + m * 8;
        const int gr = baseRow + row;
        const int gcDiag = kk + col;
        const int gc = baseCol + col;

        unsigned int vDiag = INF;
        unsigned int v = INF;
        unsigned int pv = 0;

        if (gr < n && gcDiag < n) {
            vDiag = dist[idx2(static_cast<size_t>(gcDiag), static_cast<size_t>(gr), static_cast<size_t>(n))];
        }
        if (gr < n && gc < n) {
            const size_t id = idx2(static_cast<size_t>(gc), static_cast<size_t>(gr), static_cast<size_t>(n));
            v = dist[id];
            pv = path[id];
        }

        diag[row][col] = vDiag;
        tile[row][col] = v;
        p[m] = pv;
    }

    __syncthreads();

#pragma unroll
    for (int t = 0; t < B; ++t) {
        const unsigned int b = tile[t][col];
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            const int row = ty + m * 8;
            const unsigned int a = diag[row][t];
            const unsigned int nd = a + b;
            if (nd < tile[row][col]) {
                tile[row][col] = nd;
                p[m] = static_cast<unsigned int>(kk + t);
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int m = 0; m < 4; ++m) {
        const int row = ty + m * 8;
        const int gr = baseRow + row;
        const int gc = baseCol + col;
        if (gr < n && gc < n) {
            const size_t id = idx2(static_cast<size_t>(gc), static_cast<size_t>(gr), static_cast<size_t>(n));
            dist[id] = tile[row][col];
            path[id] = p[m];
        }
    }

    (void)numTiles;
}

template <int B>
__global__ void fw_col(unsigned int* __restrict__ dist,
                       unsigned int* __restrict__ path,
                       const int n,
                       const int kk,
                       const int kkTile,
                       const int numTiles) {
    __shared__ unsigned int diag[B][B + 1];
    __shared__ unsigned int tile[B][B + 1];

    const int col = static_cast<int>(threadIdx.x);
    const int ty = static_cast<int>(threadIdx.y);

    const int tIdx = static_cast<int>(blockIdx.x);
    const int iTile = (tIdx < kkTile) ? tIdx : (tIdx + 1);

    const int baseRow = iTile * B;
    const int baseCol = kk;

    unsigned int p[4];

#pragma unroll
    for (int m = 0; m < 4; ++m) {
        const int row = ty + m * 8;
        const int grDiag = kk + row;
        const int gcDiag = kk + col;
        const int gr = baseRow + row;
        const int gc = baseCol + col;

        unsigned int vDiag = INF;
        unsigned int v = INF;
        unsigned int pv = 0;

        if (grDiag < n && gcDiag < n) {
            vDiag = dist[idx2(static_cast<size_t>(gcDiag), static_cast<size_t>(grDiag), static_cast<size_t>(n))];
        }
        if (gr < n && gc < n) {
            const size_t id = idx2(static_cast<size_t>(gc), static_cast<size_t>(gr), static_cast<size_t>(n));
            v = dist[id];
            pv = path[id];
        }

        diag[row][col] = vDiag;
        tile[row][col] = v;
        p[m] = pv;
    }

    __syncthreads();

#pragma unroll
    for (int t = 0; t < B; ++t) {
        const unsigned int b = diag[t][col];
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            const int row = ty + m * 8;
            const unsigned int a = tile[row][t];
            const unsigned int nd = a + b;
            if (nd < tile[row][col]) {
                tile[row][col] = nd;
                p[m] = static_cast<unsigned int>(kk + t);
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int m = 0; m < 4; ++m) {
        const int row = ty + m * 8;
        const int gr = baseRow + row;
        const int gc = baseCol + col;
        if (gr < n && gc < n) {
            const size_t id = idx2(static_cast<size_t>(gc), static_cast<size_t>(gr), static_cast<size_t>(n));
            dist[id] = tile[row][col];
            path[id] = p[m];
        }
    }

    (void)numTiles;
}

template <int B>
__global__ void fw_rem(unsigned int* __restrict__ dist,
                       unsigned int* __restrict__ path,
                       const int n,
                       const int kk,
                       const int kkTile,
                       const int numTiles) {
    __shared__ unsigned int rowTile[B][B + 1];
    __shared__ unsigned int colTile[B][B + 1];
    __shared__ unsigned int cur[B][B + 1];

    const int col = static_cast<int>(threadIdx.x);
    const int ty = static_cast<int>(threadIdx.y);

    const int x = static_cast<int>(blockIdx.x);
    const int y = static_cast<int>(blockIdx.y);

    const int jTile = (x < kkTile) ? x : (x + 1);
    const int iTile = (y < kkTile) ? y : (y + 1);

    const int baseRow = iTile * B;
    const int baseCol = jTile * B;

    unsigned int p[4];

#pragma unroll
    for (int m = 0; m < 4; ++m) {
        const int row = ty + m * 8;
        const int gr = baseRow + row;
        const int gc = baseCol + col;

        const int grRow = kk + row;
        const int gcRow = baseCol + col;

        const int grCol = baseRow + row;
        const int gcCol = kk + col;

        unsigned int vRow = INF;
        unsigned int vCol = INF;
        unsigned int vCur = INF;
        unsigned int pv = 0;

        if (grRow < n && gcRow < n) {
            vRow = dist[idx2(static_cast<size_t>(gcRow), static_cast<size_t>(grRow), static_cast<size_t>(n))];
        }
        if (grCol < n && gcCol < n) {
            vCol = dist[idx2(static_cast<size_t>(gcCol), static_cast<size_t>(grCol), static_cast<size_t>(n))];
        }
        if (gr < n && gc < n) {
            const size_t id = idx2(static_cast<size_t>(gc), static_cast<size_t>(gr), static_cast<size_t>(n));
            vCur = dist[id];
            pv = path[id];
        }

        rowTile[row][col] = vRow;
        colTile[row][col] = vCol;
        cur[row][col] = vCur;
        p[m] = pv;
    }

    __syncthreads();

#pragma unroll
    for (int t = 0; t < B; ++t) {
        const unsigned int b = rowTile[t][col];
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            const int row = ty + m * 8;
            const unsigned int a = colTile[row][t];
            const unsigned int nd = a + b;
            if (nd < cur[row][col]) {
                cur[row][col] = nd;
                p[m] = static_cast<unsigned int>(kk + t);
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int m = 0; m < 4; ++m) {
        const int row = ty + m * 8;
        const int gr = baseRow + row;
        const int gc = baseCol + col;
        if (gr < n && gc < n) {
            const size_t id = idx2(static_cast<size_t>(gc), static_cast<size_t>(gr), static_cast<size_t>(n));
            dist[id] = cur[row][col];
            path[id] = p[m];
        }
    }

    (void)numTiles;
}

static double floydWarshall(std::vector<unsigned int>& dist,
                            std::vector<unsigned int>& path,
                            const size_t numNodes) {
    const int n = static_cast<int>(numNodes);
    if (n <= 0) {
        return 0.0;
    }

    unsigned int* d_dist = nullptr;
    unsigned int* d_path = nullptr;
    const size_t bytes = static_cast<size_t>(n) * static_cast<size_t>(n) * sizeof(unsigned int);

    CUDA_CHECK(cudaMalloc(&d_dist, bytes));
    CUDA_CHECK(cudaMalloc(&d_path, bytes));

    CUDA_CHECK(cudaMemcpy(d_dist, dist.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_path, path.data(), bytes, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));

    cudaEvent_t start{}, stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    constexpr int B = 32;
    const int numTiles = (n + B - 1) / B;
    const dim3 threads(B, 8, 1);

    for (int kkTile = 0; kkTile < numTiles; ++kkTile) {
        const int kk = kkTile * B;

        fw_diag<B><<<1, threads>>>(d_dist, d_path, n, kk);
        CUDA_CHECK(cudaGetLastError());

        if (numTiles > 1) {
            fw_row<B><<<numTiles - 1, threads>>>(d_dist, d_path, n, kk, kkTile, numTiles);
            CUDA_CHECK(cudaGetLastError());

            fw_col<B><<<numTiles - 1, threads>>>(d_dist, d_path, n, kk, kkTile, numTiles);
            CUDA_CHECK(cudaGetLastError());

            fw_rem<B><<<dim3(numTiles - 1, numTiles - 1, 1), threads>>>(d_dist, d_path, n, kk, kkTile, numTiles);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float kernel_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaMemcpy(dist.data(), d_dist, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(path.data(), d_path, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_dist));
    CUDA_CHECK(cudaFree(d_path));

    return static_cast<double>(kernel_ms);
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

// (CUDA implementation is defined above)

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

    const double kernel_ms = floydWarshall(dist, path, numNodes);
    const long duration_ms = static_cast<long>(std::ceil(kernel_ms));
    printf("Computation time: %ld ms\n", duration_ms);

    // Calculate operations per second (Floyd-Warshall is O(n^3))
    const double ops = (double)numNodes * numNodes * numNodes;
    const double seconds = std::max(kernel_ms, 0.001) / 1000.0;
    const double gflops = ops / seconds / 1e9;
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
