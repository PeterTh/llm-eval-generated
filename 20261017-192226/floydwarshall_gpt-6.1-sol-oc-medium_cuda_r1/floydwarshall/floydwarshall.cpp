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
constexpr int THREAD_ROWS = 8;
using Entry = unsigned long long;

void cudaCheck(cudaError_t status) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

// The high word is distance; the low word is one plus the largest
// intermediate vertex (zero for a direct edge). Lexicographic minimization
// retains the first k attaining the shortest distance, exactly as the serial
// strict-less-than update does, even though tiles execute out of k order.
__device__ __forceinline__ Entry join(Entry a, Entry b, unsigned int k) {
    unsigned int distance = static_cast<unsigned int>(a >> 32) +
                            static_cast<unsigned int>(b >> 32);
    unsigned int rank = max(max(static_cast<unsigned int>(a),
                                static_cast<unsigned int>(b)), k);
    return (static_cast<Entry>(distance) << 32) | rank;
}

__global__ void packMatrix(const unsigned int* input, Entry* matrix,
                           size_t n, size_t pitch) {
    const size_t x = static_cast<size_t>(blockIdx.x) * TILE + threadIdx.x;
    for (int r = threadIdx.y; r < TILE; r += THREAD_ROWS) {
        const size_t y = static_cast<size_t>(blockIdx.y) * TILE + r;
        unsigned int d = (x < n && y < n) ? input[y * n + x] :
                         (x == y ? 0 : INF);
        matrix[y * pitch + x] = static_cast<Entry>(d) << 32;
    }
}

__global__ void unpackMatrix(const Entry* matrix, unsigned int* output,
                             size_t n, size_t pitch) {
    const size_t x = static_cast<size_t>(blockIdx.x) * TILE + threadIdx.x;
    for (int r = threadIdx.y; r < TILE; r += THREAD_ROWS) {
        const size_t y = static_cast<size_t>(blockIdx.y) * TILE + r;
        if (x < n && y < n) {
            Entry value = matrix[y * pitch + x];
            unsigned int rank = static_cast<unsigned int>(value);
            output[y * n + x] = static_cast<unsigned int>(value >> 32);
            output[n * n + y * n + x] = rank ? rank - 1 : y;
        }
    }
}

// Phase 1: close the diagonal tile. Pivot row/column entries cannot improve
// for this nonnegative graph, so leave them unwritten to avoid shared-memory
// read/write races within each k iteration.
__global__ void closePivot(Entry* matrix, size_t pitch, int round) {
    __shared__ Entry tile[TILE][TILE];
    const int x = threadIdx.x;
    const size_t base = static_cast<size_t>(round) * TILE;
    #pragma unroll
    for (int r = threadIdx.y; r < TILE; r += THREAD_ROWS)
        tile[r][x] = matrix[(base + r) * pitch + base + x];
    __syncthreads();
    for (int k = 0; k < TILE; ++k) {
        #pragma unroll
        for (int r = threadIdx.y; r < TILE; r += THREAD_ROWS) {
            if (r != k && x != k) {
                Entry candidate = join(tile[r][k], tile[k][x], base + k + 1);
                tile[r][x] = min(tile[r][x], candidate);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int r = threadIdx.y; r < TILE; r += THREAD_ROWS)
        matrix[(base + r) * pitch + base + x] = tile[r][x];
}

// Phase 2: all tiles in the pivot's row and column run independently.
__global__ void closePanels(Entry* matrix, size_t pitch, int round) {
    __shared__ Entry pivot[TILE][TILE];
    __shared__ Entry tile[TILE][TILE];
    const int x = threadIdx.x;
    const int other = blockIdx.x + (blockIdx.x >= static_cast<unsigned int>(round));
    const bool row = blockIdx.y == 0;
    const size_t base = static_cast<size_t>(round) * TILE;
    const size_t ybase = static_cast<size_t>(row ? round : other) * TILE;
    const size_t xbase = static_cast<size_t>(row ? other : round) * TILE;
    #pragma unroll
    for (int r = threadIdx.y; r < TILE; r += THREAD_ROWS) {
        pivot[r][x] = matrix[(base + r) * pitch + base + x];
        tile[r][x] = matrix[(ybase + r) * pitch + xbase + x];
    }
    __syncthreads();
    for (int k = 0; k < TILE; ++k) {
        #pragma unroll
        for (int r = threadIdx.y; r < TILE; r += THREAD_ROWS) {
            if (row ? r != k : x != k) {
                Entry candidate = row ? join(pivot[r][k], tile[k][x], base + k + 1)
                                      : join(tile[r][k], pivot[k][x], base + k + 1);
                tile[r][x] = min(tile[r][x], candidate);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int r = threadIdx.y; r < TILE; r += THREAD_ROWS)
        matrix[(ybase + r) * pitch + xbase + x] = tile[r][x];
}

// Phase 3 dominates the work: reuse both panels in shared memory and keep
// four independent outputs per thread in registers throughout the tile.
__global__ void closeRemaining(Entry* matrix, size_t pitch, int round) {
    __shared__ Entry left[TILE][TILE];
    __shared__ Entry right[TILE][TILE];
    const int x = threadIdx.x;
    const size_t ybase = static_cast<size_t>(blockIdx.y +
                         (blockIdx.y >= static_cast<unsigned int>(round))) * TILE;
    const size_t xbase = static_cast<size_t>(blockIdx.x +
                         (blockIdx.x >= static_cast<unsigned int>(round))) * TILE;
    const size_t base = static_cast<size_t>(round) * TILE;
    Entry values[TILE / THREAD_ROWS];
    #pragma unroll
    for (int t = 0; t < TILE / THREAD_ROWS; ++t) {
        int r = threadIdx.y + t * THREAD_ROWS;
        left[r][x] = matrix[(ybase + r) * pitch + base + x];
        right[r][x] = matrix[(base + r) * pitch + xbase + x];
        values[t] = matrix[(ybase + r) * pitch + xbase + x];
    }
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < TILE; ++k) {
        Entry b = right[k][x];
        #pragma unroll
        for (int t = 0; t < TILE / THREAD_ROWS; ++t) {
            int r = threadIdx.y + t * THREAD_ROWS;
            values[t] = min(values[t], join(left[r][k], b, base + k + 1));
        }
    }
    #pragma unroll
    for (int t = 0; t < TILE / THREAD_ROWS; ++t) {
        int r = threadIdx.y + t * THREAD_ROWS;
        matrix[(ybase + r) * pitch + xbase + x] = values[t];
    }
}
} // namespace

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes) {
    // Even an empty graph requires a working CUDA device; there is no CPU path.
    cudaCheck(cudaFree(nullptr));
    if (numNodes == 0) return;
    const size_t tiles = (numNodes + TILE - 1) / TILE;
    const size_t pitch = tiles * TILE;
    const size_t bytes = numNodes * numNodes * sizeof(unsigned int);
    Entry* matrix = nullptr;
    unsigned int* transfer = nullptr;
    cudaCheck(cudaMalloc(&matrix, pitch * pitch * sizeof(Entry)));
    cudaCheck(cudaMalloc(&transfer, 2 * bytes));
    cudaCheck(cudaMemcpy(transfer, dist.data(), bytes, cudaMemcpyHostToDevice));
    const dim3 threads(TILE, THREAD_ROWS);
    const dim3 grid(tiles, tiles);
    packMatrix<<<grid, threads>>>(transfer, matrix, numNodes, pitch);
    for (size_t round = 0; round < tiles; ++round) {
        closePivot<<<1, threads>>>(matrix, pitch, round);
        if (tiles > 1) {
            closePanels<<<dim3(tiles - 1, 2), threads>>>(matrix, pitch, round);
            closeRemaining<<<dim3(tiles - 1, tiles - 1), threads>>>(matrix, pitch, round);
        }
    }
    unpackMatrix<<<grid, threads>>>(matrix, transfer, numNodes, pitch);
    cudaCheck(cudaGetLastError());
    // Synchronous copies also surface execution errors and include all GPU work
    // in the benchmark's existing wall-clock timer.
    cudaCheck(cudaMemcpy(dist.data(), transfer, bytes, cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(path.data(), transfer + numNodes * numNodes, bytes,
                         cudaMemcpyDeviceToHost));
    cudaCheck(cudaFree(transfer));
    cudaCheck(cudaFree(matrix));
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

    // Initialize the CUDA context before timing, just as host initialization
    // below is excluded. Allocation, transfers, and GPU execution stay timed.
    cudaCheck(cudaSetDevice(0));
    cudaCheck(cudaFree(nullptr));
    
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
    const double milliseconds = std::chrono::duration<double, std::milli>(end - start).count();
    
    printf("Computation time: %.3f ms\n", milliseconds);
    
    // Calculate operations per second
    // Floyd-Warshall has O(n³) complexity
    double ops = (double)numNodes * numNodes * numNodes;
    double gflops = ops / (milliseconds / 1000.0) / 1e9;
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
