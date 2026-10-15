#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

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

// Blocked Floyd-Warshall: tiles of TILE x TILE, thread block of TILE x ROWS
constexpr int TILE = 32;
constexpr int ROWS = 8;
constexpr int PER_THREAD = TILE / ROWS;

// Phase 1: update the diagonal tile
__global__ void fwPhase1(unsigned int* dist, unsigned int* path, const int n, const int kk) {
    __shared__ unsigned int sd[TILE][TILE + 1];
    __shared__ unsigned int sp[TILE][TILE + 1];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const size_t base = (size_t)kk * TILE * n + (size_t)kk * TILE;
#pragma unroll
    for (int r = 0; r < PER_THREAD; ++r) {
        const int i = ty + r * ROWS;
        sd[i][tx] = dist[base + (size_t)i * n + tx];
        sp[i][tx] = path[base + (size_t)i * n + tx];
    }
    __syncthreads();
    for (int k = 0; k < TILE; ++k) {
        const unsigned int kj = sd[k][tx];
#pragma unroll
        for (int r = 0; r < PER_THREAD; ++r) {
            const int i = ty + r * ROWS;
            const unsigned int nd = sd[i][k] + kj;
            if (nd < sd[i][tx]) {
                sd[i][tx] = nd;
                sp[i][tx] = kk * TILE + k;
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < PER_THREAD; ++r) {
        const int i = ty + r * ROWS;
        dist[base + (size_t)i * n + tx] = sd[i][tx];
        path[base + (size_t)i * n + tx] = sp[i][tx];
    }
}

// Phase 2: update tiles in pivot row (blockIdx.y==0) and pivot column (blockIdx.y==1)
__global__ void fwPhase2(unsigned int* dist, unsigned int* path, const int n, const int kk) {
    __shared__ unsigned int pd[TILE][TILE + 1];
    __shared__ unsigned int cd[TILE][TILE + 1];
    __shared__ unsigned int cp[TILE][TILE + 1];
    int b = blockIdx.x;
    if (b >= kk) ++b;
    const bool isRow = (blockIdx.y == 0);
    const int tx = threadIdx.x, ty = threadIdx.y;
    const size_t pbase = (size_t)kk * TILE * n + (size_t)kk * TILE;
    const size_t cbase = isRow ? (size_t)kk * TILE * n + (size_t)b * TILE
                               : (size_t)b * TILE * n + (size_t)kk * TILE;
#pragma unroll
    for (int r = 0; r < PER_THREAD; ++r) {
        const int i = ty + r * ROWS;
        pd[i][tx] = dist[pbase + (size_t)i * n + tx];
        cd[i][tx] = dist[cbase + (size_t)i * n + tx];
        cp[i][tx] = path[cbase + (size_t)i * n + tx];
    }
    __syncthreads();
    for (int k = 0; k < TILE; ++k) {
#pragma unroll
        for (int r = 0; r < PER_THREAD; ++r) {
            const int i = ty + r * ROWS;
            unsigned int nd;
            if (isRow) nd = pd[i][k] + cd[k][tx];
            else       nd = cd[i][k] + pd[k][tx];
            if (nd < cd[i][tx]) {
                cd[i][tx] = nd;
                cp[i][tx] = kk * TILE + k;
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < PER_THREAD; ++r) {
        const int i = ty + r * ROWS;
        dist[cbase + (size_t)i * n + tx] = cd[i][tx];
        path[cbase + (size_t)i * n + tx] = cp[i][tx];
    }
}

// Phase 3: update all remaining tiles using the pivot row/column tiles
__global__ void __launch_bounds__(TILE * ROWS)
fwPhase3(unsigned int* dist, unsigned int* path, const int n, const int kk) {
    __shared__ unsigned int colT[TILE][TILE + 1];  // dist[i-tile][kk]
    __shared__ unsigned int rowT[TILE][TILE];      // dist[kk][j-tile]
    int bx = blockIdx.x, by = blockIdx.y;
    if (bx >= kk) ++bx;
    if (by >= kk) ++by;
    const int tx = threadIdx.x, ty = threadIdx.y;
    const size_t cbase = (size_t)by * TILE * n + (size_t)bx * TILE;
    const size_t abase = (size_t)by * TILE * n + (size_t)kk * TILE;
    const size_t bbase = (size_t)kk * TILE * n + (size_t)bx * TILE;

    unsigned int d[PER_THREAD], p[PER_THREAD];
#pragma unroll
    for (int r = 0; r < PER_THREAD; ++r) {
        const int i = ty + r * ROWS;
        colT[i][tx] = dist[abase + (size_t)i * n + tx];
        rowT[i][tx] = dist[bbase + (size_t)i * n + tx];
        d[r] = dist[cbase + (size_t)i * n + tx];
        p[r] = path[cbase + (size_t)i * n + tx];
    }
    __syncthreads();
#pragma unroll 8
    for (int k = 0; k < TILE; ++k) {
        const unsigned int kj = rowT[k][tx];
#pragma unroll
        for (int r = 0; r < PER_THREAD; ++r) {
            const unsigned int nd = colT[ty + r * ROWS][k] + kj;
            if (nd < d[r]) {
                d[r] = nd;
                p[r] = kk * TILE + k;
            }
        }
    }
#pragma unroll
    for (int r = 0; r < PER_THREAD; ++r) {
        const int i = ty + r * ROWS;
        dist[cbase + (size_t)i * n + tx] = d[r];
        path[cbase + (size_t)i * n + tx] = p[r];
    }
}

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes) {
    const int n = (int)numNodes;
    const int nb = (n + TILE - 1) / TILE;
    const int np = nb * TILE;  // padded dimension
    const size_t bytes = (size_t)np * np * sizeof(unsigned int);

    // Padded host staging buffers; padding nodes are unreachable (INF) with zero diagonal
    std::vector<unsigned int> hd((size_t)np * np, INF), hp((size_t)np * np, 0);
    for (int i = 0; i < np; ++i) {
        hd[(size_t)i * np + i] = 0;
        hp[(size_t)i * np + i] = i;
    }
    for (int i = 0; i < n; ++i) {
        memcpy(&hd[(size_t)i * np], &dist[(size_t)i * n], n * sizeof(unsigned int));
        memcpy(&hp[(size_t)i * np], &path[(size_t)i * n], n * sizeof(unsigned int));
    }

    unsigned int *dd, *dp;
    CUDA_CHECK(cudaMalloc(&dd, bytes));
    CUDA_CHECK(cudaMalloc(&dp, bytes));
    CUDA_CHECK(cudaMemcpy(dd, hd.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dp, hp.data(), bytes, cudaMemcpyHostToDevice));

    const dim3 block(TILE, ROWS);
    for (int kk = 0; kk < nb; ++kk) {
        fwPhase1<<<1, block>>>(dd, dp, np, kk);
        if (nb > 1) {
            fwPhase2<<<dim3(nb - 1, 2), block>>>(dd, dp, np, kk);
            fwPhase3<<<dim3(nb - 1, nb - 1), block>>>(dd, dp, np, kk);
        }
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(hd.data(), dd, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hp.data(), dp, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dd));
    CUDA_CHECK(cudaFree(dp));

    for (int i = 0; i < n; ++i) {
        memcpy(&dist[(size_t)i * n], &hd[(size_t)i * np], n * sizeof(unsigned int));
        memcpy(&path[(size_t)i * n], &hp[(size_t)i * np], n * sizeof(unsigned int));
    }
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
    
    // Initialize the CUDA runtime/context up front (outside the timed region)
    CUDA_CHECK(cudaFree(0));

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
