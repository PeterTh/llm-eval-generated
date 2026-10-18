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

namespace {
constexpr int TILE = 32;
constexpr int ROWS = 4;

void cudaCheck(cudaError_t error) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}

// A rank of zero denotes a direct edge; otherwise it is one plus the
// largest intermediate vertex. Minimizing (distance, rank) preserves the
// original ascending-k, strict-improvement path matrix even with blocked k.
__device__ __forceinline__ void relax(unsigned int& d, unsigned int& rank,
                                      unsigned int a, unsigned int b,
                                      unsigned int ar, unsigned int br,
                                      unsigned int k) {
    const unsigned int candidate = a + b;
    const unsigned int candidateRank = max(k + 1, max(ar, br));
    if (candidate < d || (candidate == d && candidateRank < rank)) {
        d = candidate;
        rank = candidateRank;
    }
}

// Phase 1 closes the pivot tile. Phase 2 closes its row and column.
// Each thread owns four rows and one column, with coalesced global accesses.
template<bool Pivot>
__global__ void closeTiles(unsigned int* dist, unsigned int* ranks,
                           size_t stride, int pivot) {
    __shared__ unsigned int pd[TILE][TILE], pr[TILE][TILE];
    __shared__ unsigned int td[TILE][TILE], tr[TILE][TILE];
    const int x = threadIdx.x, y = threadIdx.y;
    int tile = blockIdx.x;
    if (!Pivot && tile >= pivot) ++tile;
    const bool row = Pivot || blockIdx.y == 0;
    const int tileRow = Pivot ? pivot : (row ? pivot : tile);
    const int tileCol = Pivot ? pivot : (row ? tile : pivot);
    #pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        const int yy = y + r * (TILE / ROWS);
        const size_t p = (size_t(pivot) * TILE + yy) * stride + pivot * TILE + x;
        pd[yy][x] = dist[p];
        pr[yy][x] = ranks[p];
        if (!Pivot) {
            const size_t t = (size_t(tileRow) * TILE + yy) * stride + tileCol * TILE + x;
            td[yy][x] = dist[t];
            tr[yy][x] = ranks[t];
        }
    }
    __syncthreads();
    #pragma unroll 1
    for (int k = 0; k < TILE; ++k) {
        unsigned int values[ROWS], paths[ROWS];
        #pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const int yy = y + r * (TILE / ROWS);
            values[r] = Pivot ? pd[yy][x] : td[yy][x];
            paths[r] = Pivot ? pr[yy][x] : tr[yy][x];
            const unsigned int a = (Pivot || row) ? pd[yy][k] : td[yy][k];
            const unsigned int ar = (Pivot || row) ? pr[yy][k] : tr[yy][k];
            const unsigned int b = Pivot ? pd[k][x] : (row ? td[k][x] : pd[k][x]);
            const unsigned int br = Pivot ? pr[k][x] : (row ? tr[k][x] : pr[k][x]);
            relax(values[r], paths[r], a, b, ar, br, pivot * TILE + k);
        }
        // All reads finish before any thread changes a shared pivot operand.
        __syncthreads();
        #pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const int yy = y + r * (TILE / ROWS);
            if (Pivot) { pd[yy][x] = values[r]; pr[yy][x] = paths[r]; }
            else { td[yy][x] = values[r]; tr[yy][x] = paths[r]; }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        const int yy = y + r * (TILE / ROWS);
        const size_t t = (size_t(tileRow) * TILE + yy) * stride + tileCol * TILE + x;
        dist[t] = Pivot ? pd[yy][x] : td[yy][x];
        ranks[t] = Pivot ? pr[yy][x] : tr[yy][x];
    }
}

// Phase 3: independent non-pivot tiles. Operands stay in shared memory;
// four destination elements per thread stay in registers for all 32 steps.
__global__ void updateTiles(unsigned int* dist, unsigned int* ranks,
                            size_t stride, int pivot) {
    __shared__ unsigned int left[TILE][TILE], right[TILE][TILE];
    __shared__ unsigned int lr[TILE][TILE], rr[TILE][TILE];
    const int x = threadIdx.x, y = threadIdx.y;
    const int tileRow = blockIdx.y + (blockIdx.y >= unsigned(pivot));
    const int tileCol = blockIdx.x + (blockIdx.x >= unsigned(pivot));
    unsigned int values[ROWS], paths[ROWS];
    #pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        const int yy = y + r * (TILE / ROWS);
        const size_t a = (size_t(tileRow) * TILE + yy) * stride + pivot * TILE + x;
        const size_t b = (size_t(pivot) * TILE + yy) * stride + tileCol * TILE + x;
        const size_t t = (size_t(tileRow) * TILE + yy) * stride + tileCol * TILE + x;
        left[yy][x] = dist[a]; lr[yy][x] = ranks[a];
        right[yy][x] = dist[b]; rr[yy][x] = ranks[b];
        values[r] = dist[t]; paths[r] = ranks[t];
    }
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < TILE; ++k) {
        const unsigned int b = right[k][x], br = rr[k][x];
        #pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const int yy = y + r * (TILE / ROWS);
            relax(values[r], paths[r], left[yy][k], b, lr[yy][k], br,
                  pivot * TILE + k);
        }
    }
    #pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        const int yy = y + r * (TILE / ROWS);
        const size_t t = (size_t(tileRow) * TILE + yy) * stride + tileCol * TILE + x;
        dist[t] = values[r]; ranks[t] = paths[r];
    }
}

__global__ void decodePaths(unsigned int* ranks, size_t stride, size_t n) {
    const size_t col = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t row = size_t(blockIdx.y) * blockDim.y + threadIdx.y;
    if (row < n && col < n) {
        unsigned int& rank = ranks[row * stride + col];
        rank = rank ? rank - 1 : static_cast<unsigned int>(row);
    }
}
} // namespace

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes) {
    // CUDA is mandatory, including for empty graphs; no CPU fallback.
    cudaCheck(cudaFree(nullptr));
    if (numNodes == 0) return;
    const size_t tiles = (numNodes + TILE - 1) / TILE;
    const size_t stride = tiles * TILE;
    const size_t bytes = stride * stride * sizeof(unsigned int);
    unsigned int *deviceDist = nullptr, *deviceRanks = nullptr;
    cudaCheck(cudaMalloc(&deviceDist, bytes));
    cudaCheck(cudaMalloc(&deviceRanks, bytes));
    // Padded vertices have no finite incident edges. 0x3f3f3f3f can be
    // added twice without unsigned overflow and exceeds every input weight.
    cudaCheck(cudaMemset(deviceDist, 0x3f, bytes));
    cudaCheck(cudaMemset(deviceRanks, 0, bytes));
    cudaCheck(cudaMemcpy2D(deviceDist, stride * sizeof(unsigned int), dist.data(),
                           numNodes * sizeof(unsigned int), numNodes * sizeof(unsigned int),
                           numNodes, cudaMemcpyHostToDevice));
    const dim3 threads(TILE, TILE / ROWS);
    for (size_t p = 0; p < tiles; ++p) {
        closeTiles<true><<<1, threads>>>(deviceDist, deviceRanks, stride, int(p));
        if (tiles > 1) {
            closeTiles<false><<<dim3(tiles - 1, 2), threads>>>(deviceDist, deviceRanks, stride, int(p));
            updateTiles<<<dim3(tiles - 1, tiles - 1), threads>>>(deviceDist, deviceRanks, stride, int(p));
        }
    }
    decodePaths<<<dim3((numNodes + 31) / 32, (numNodes + 7) / 8), dim3(32, 8)>>>(
        deviceRanks, stride, numNodes);
    cudaCheck(cudaGetLastError());
    // Blocking copies include completion of all GPU work in benchmark time.
    cudaCheck(cudaMemcpy2D(dist.data(), numNodes * sizeof(unsigned int), deviceDist,
                           stride * sizeof(unsigned int), numNodes * sizeof(unsigned int),
                           numNodes, cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy2D(path.data(), numNodes * sizeof(unsigned int), deviceRanks,
                           stride * sizeof(unsigned int), numNodes * sizeof(unsigned int),
                           numNodes, cudaMemcpyDeviceToHost));
    cudaCheck(cudaFree(deviceRanks));
    cudaCheck(cudaFree(deviceDist));
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

    // Initialize the CUDA context before timing, just as graph initialization
    // is excluded. Allocations, transfers, and all kernels remain timed.
    cudaCheck(cudaFree(nullptr));
    
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
