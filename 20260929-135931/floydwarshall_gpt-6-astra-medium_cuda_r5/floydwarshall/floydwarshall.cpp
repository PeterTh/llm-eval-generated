#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <limits>
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
using Entry = unsigned long long;

// Lexicographic (distance, largest intermediate vertex + 1). Among equal
// shortest paths the smallest such vertex is exactly the last strict update
// in the original, ascending-k algorithm. Zero denotes an unchanged edge.
__device__ __forceinline__ Entry via(Entry a, Entry b, unsigned int k) {
    const unsigned int distance = static_cast<unsigned int>(a >> 32) +
                                  static_cast<unsigned int>(b >> 32);
    const unsigned int label = max(k, max(static_cast<unsigned int>(a),
                                         static_cast<unsigned int>(b)));
    return (static_cast<Entry>(distance) << 32) | label;
}

void checkCuda(cudaError_t status) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

__global__ void packMatrix(const unsigned int* input, Entry* matrix,
                           size_t n, size_t pitch) {
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < pitch * pitch; index += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const size_t row = index / pitch, col = index % pitch;
        const unsigned int d = row < n && col < n ? input[row * n + col] : INF;
        matrix[index] = static_cast<Entry>(d) << 32;
    }
}

__global__ void unpackMatrix(const Entry* matrix, unsigned int* dist,
                             unsigned int* path, size_t n, size_t pitch) {
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < n * n; index += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const Entry value = matrix[(index / n) * pitch + index % n];
        dist[index] = static_cast<unsigned int>(value >> 32);
        const unsigned int label = static_cast<unsigned int>(value);
        if (label) path[index] = label - 1;
    }
}

// The first two phases have dependencies within a tile. All threads read
// before any thread writes, including pivot row/column entries that tie.
template<bool Pivot>
__global__ void updatePanel(Entry* matrix, size_t pitch, unsigned int pivot) {
    __shared__ Entry tile[TILE][TILE];
    __shared__ Entry diagonal[TILE][TILE];
    const unsigned int other = blockIdx.x + (blockIdx.x >= pivot);
    const bool rowPanel = blockIdx.y == 0;
    const unsigned int br = Pivot || rowPanel ? pivot : other;
    const unsigned int bc = Pivot || !rowPanel ? pivot : other;
    const int x = threadIdx.x, y = threadIdx.y;
    Entry value[ROWS];
    #pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        const int row = y + r * 8;
        tile[row][x] = matrix[(static_cast<size_t>(br) * TILE + row) * pitch + bc * TILE + x];
        if (!Pivot)
            diagonal[row][x] = matrix[(static_cast<size_t>(pivot) * TILE + row) * pitch + pivot * TILE + x];
    }
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < TILE; ++k) {
        #pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const int row = y + r * 8;
            Entry left, right;
            if (Pivot) {
                left = tile[row][k]; right = tile[k][x];
            } else if (rowPanel) {
                left = diagonal[row][k]; right = tile[k][x];
            } else {
                left = tile[row][k]; right = diagonal[k][x];
            }
            value[r] = min(tile[row][x], via(left, right, pivot * TILE + k + 1));
        }
        __syncthreads();
        #pragma unroll
        for (int r = 0; r < ROWS; ++r) tile[y + r * 8][x] = value[r];
        __syncthreads();
    }
    #pragma unroll
    for (int r = 0; r < ROWS; ++r)
        matrix[(static_cast<size_t>(br) * TILE + y + r * 8) * pitch + bc * TILE + x] = value[r];
}

__global__ void updateRemainder(Entry* matrix, size_t pitch, unsigned int pivot) {
    __shared__ Entry left[TILE][TILE];
    __shared__ Entry right[TILE][TILE];
    const unsigned int br = blockIdx.y + (blockIdx.y >= pivot);
    const unsigned int bc = blockIdx.x + (blockIdx.x >= pivot);
    const int x = threadIdx.x, y = threadIdx.y;
    Entry value[ROWS];
    #pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        const size_t row = static_cast<size_t>(br) * TILE + y + r * 8;
        left[y + r * 8][x] = matrix[row * pitch + pivot * TILE + x];
        right[y + r * 8][x] = matrix[(static_cast<size_t>(pivot) * TILE + y + r * 8) * pitch + bc * TILE + x];
        value[r] = matrix[row * pitch + bc * TILE + x];
    }
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < TILE; ++k) {
        const Entry b = right[k][x];
        #pragma unroll
        for (int r = 0; r < ROWS; ++r)
            value[r] = min(value[r], via(left[y + r * 8][k], b, pivot * TILE + k + 1));
    }
    #pragma unroll
    for (int r = 0; r < ROWS; ++r)
        matrix[(static_cast<size_t>(br) * TILE + y + r * 8) * pitch + bc * TILE + x] = value[r];
}
} // namespace

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes) {
    // Require a CUDA device even for an empty graph; there is no CPU fallback.
    checkCuda(cudaFree(nullptr));
    if (numNodes == 0) return;
    const size_t pitch = (numNodes + TILE - 1) / TILE * TILE;
    const unsigned int tiles = static_cast<unsigned int>(pitch / TILE);
    int device;
    cudaDeviceProp properties{};
    checkCuda(cudaGetDevice(&device));
    checkCuda(cudaGetDeviceProperties(&properties, device));
    if (tiles > static_cast<unsigned int>(properties.maxGridSize[1]) ||
        pitch > std::numeric_limits<size_t>::max() / pitch / sizeof(Entry)) {
        fprintf(stderr, "Graph exceeds CUDA matrix/grid limits\n");
        std::exit(EXIT_FAILURE);
    }
    Entry* matrix;
    unsigned int *deviceDist, *devicePath;
    const size_t bytes = numNodes * numNodes * sizeof(unsigned int);
    checkCuda(cudaMalloc(&matrix, pitch * pitch * sizeof(Entry)));
    checkCuda(cudaMalloc(&deviceDist, bytes));
    checkCuda(cudaMalloc(&devicePath, bytes));
    checkCuda(cudaMemcpy(deviceDist, dist.data(), bytes, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(devicePath, path.data(), bytes, cudaMemcpyHostToDevice));
    const unsigned int blocks = static_cast<unsigned int>(std::min<size_t>((pitch * pitch + 255) / 256, 65535));
    packMatrix<<<blocks, 256>>>(deviceDist, matrix, numNodes, pitch);
    checkCuda(cudaGetLastError());
    const dim3 threads(TILE, 8);
    for (unsigned int pivot = 0; pivot < tiles; ++pivot) {
        updatePanel<true><<<1, threads>>>(matrix, pitch, pivot);
        checkCuda(cudaGetLastError());
        if (tiles > 1) {
            updatePanel<false><<<dim3(tiles - 1, 2), threads>>>(matrix, pitch, pivot);
            checkCuda(cudaGetLastError());
            updateRemainder<<<dim3(tiles - 1, tiles - 1), threads>>>(matrix, pitch, pivot);
            checkCuda(cudaGetLastError());
        }
    }
    unpackMatrix<<<blocks, 256>>>(matrix, deviceDist, devicePath, numNodes, pitch);
    checkCuda(cudaGetLastError());
    // Blocking copies include completion of all GPU work in the benchmark time.
    checkCuda(cudaMemcpy(dist.data(), deviceDist, bytes, cudaMemcpyDeviceToHost));
    checkCuda(cudaMemcpy(path.data(), devicePath, bytes, cudaMemcpyDeviceToHost));
    checkCuda(cudaFree(devicePath));
    checkCuda(cudaFree(deviceDist));
    checkCuda(cudaFree(matrix));
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
