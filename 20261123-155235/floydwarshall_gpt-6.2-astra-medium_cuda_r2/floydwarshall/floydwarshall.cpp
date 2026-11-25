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
    for (size_t row = 0; row < numNodes; ++row) {
        std::fill_n(path.data() + row * numNodes, numNodes, row);
    }
}

namespace {
constexpr int tile = 32;
constexpr int blockRows = 8;
constexpr int rowsPerThread = tile / blockRows;

void checkCuda(cudaError_t status) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

enum class Panel { pivot, row, column };

// Capture operands at their serial k-step. Using the fully relaxed panels
// would give the same distances but could select a different intermediate.
template<Panel kind>
__global__ void relaxPanel(unsigned* dist, unsigned* path, unsigned* columns,
                           unsigned* rows, size_t stride, int pivot) {
    const int other = blockIdx.x + (blockIdx.x >= pivot);
    const int rowBase = (kind == Panel::column ? other : pivot) * tile;
    const int colBase = (kind == Panel::row ? other : pivot) * tile;
    const int x = threadIdx.x;
    const int y = threadIdx.y;
    __shared__ unsigned current[tile][tile];
    __shared__ unsigned pivotOperands[tile][tile];
    unsigned via[rowsPerThread];

    #pragma unroll
    for (int t = 0; t < rowsPerThread; ++t) {
        int r = y + t * blockRows;
        current[r][x] = dist[(rowBase + r) * stride + colBase + x];
        via[t] = ~0u;
        if constexpr (kind == Panel::row)
            pivotOperands[r][x] = columns[size_t(rowBase + r) * tile + x];
        if constexpr (kind == Panel::column)
            pivotOperands[r][x] = rows[r * stride + colBase + x];
    }
    __syncthreads();

    for (int k = 0; k < tile; ++k) {
        const unsigned right = kind == Panel::column ? pivotOperands[k][x] : current[k][x];
        if constexpr (kind != Panel::column) {
            if (y == 0) rows[k * stride + colBase + x] = right;
        }
        #pragma unroll
        for (int t = 0; t < rowsPerThread; ++t) {
            int r = y + t * blockRows;
            const unsigned left = kind == Panel::row ? pivotOperands[r][k] : current[r][k];
            if constexpr (kind != Panel::row) {
                if (x == k) columns[size_t(rowBase + r) * tile + k] = left;
            }
            // With a zero diagonal, no strict improvement can write a
            // live operand of this step. The barrier orders successive steps.
            if (left + right < current[r][x]) {
                current[r][x] = left + right;
                via[t] = pivot * tile + k;
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int t = 0; t < rowsPerThread; ++t) {
        int r = y + t * blockRows;
        size_t idx = (rowBase + r) * stride + colBase + x;
        dist[idx] = current[r][x];
        if (via[t] != ~0u) path[idx] = via[t];
    }
}

__global__ void relaxRemaining(unsigned* dist, unsigned* path, const unsigned* columns,
                               const unsigned* rows, size_t stride, int pivot) {
    const int rowBase = (blockIdx.y + (blockIdx.y >= pivot)) * tile;
    const int colBase = (blockIdx.x + (blockIdx.x >= pivot)) * tile;
    const int x = threadIdx.x;
    const int y = threadIdx.y;
    __shared__ unsigned left[tile][tile];
    __shared__ unsigned right[tile][tile];
    // Compare (distance, local intermediate) with one unsigned minimum.
    // Put the intermediate in the low bits only once per shared operand.
    unsigned original[rowsPerThread], best[rowsPerThread];
    #pragma unroll
    for (int t = 0; t < rowsPerThread; ++t) {
        int r = y + t * blockRows;
        left[r][x] = columns[size_t(rowBase + r) * tile + x] * tile;
        right[r][x] = rows[r * stride + colBase + x] * tile + r;
        original[t] = dist[(rowBase + r) * stride + colBase + x] * tile;
        best[t] = original[t];
    }
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < tile; ++k) {
        #pragma unroll
        for (int t = 0; t < rowsPerThread; ++t) {
            unsigned candidate = left[y + t * blockRows][k] + right[k][x];
            best[t] = min(best[t], candidate);
        }
    }
    #pragma unroll
    for (int t = 0; t < rowsPerThread; ++t) {
        if (best[t] < original[t]) {
            size_t idx = (rowBase + y + t * blockRows) * stride + colBase + x;
            dist[idx] = best[t] / tile;
            path[idx] = pivot * tile + best[t] % tile;
        }
    }
}
} // namespace

void floydWarshall(std::vector<unsigned>& dist, std::vector<unsigned>& path, size_t numNodes) {
    if (numNodes == 0) return;
    const int tiles = (numNodes + tile - 1) / tile;
    const size_t stride = size_t(tiles) * tile;
    const size_t pitch = stride * sizeof(unsigned);
    const size_t width = numNodes * sizeof(unsigned);
    unsigned *deviceDist, *devicePath, *columns, *rows;
    checkCuda(cudaMalloc(&deviceDist, stride * pitch));
    checkCuda(cudaMalloc(&devicePath, stride * pitch));
    checkCuda(cudaMalloc(&columns, tile * pitch));
    checkCuda(cudaMalloc(&rows, tile * pitch));
    // 0x01010101 exceeds every real distance, and even two such weights
    // plus the packed intermediate fit in unsigned after scaling by tile.
    if (stride != numNodes) checkCuda(cudaMemset(deviceDist, 0x01, stride * pitch));
    checkCuda(cudaMemcpy2D(deviceDist, pitch, dist.data(), width, width, numNodes, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy2D(devicePath, pitch, path.data(), width, width, numNodes, cudaMemcpyHostToDevice));

    const dim3 threads(tile, blockRows);
    for (int pivot = 0; pivot < tiles; ++pivot) {
        relaxPanel<Panel::pivot><<<1, threads>>>(deviceDist, devicePath, columns, rows, stride, pivot);
        if (tiles > 1) {
            relaxPanel<Panel::row><<<tiles - 1, threads>>>(deviceDist, devicePath, columns, rows, stride, pivot);
            relaxPanel<Panel::column><<<tiles - 1, threads>>>(deviceDist, devicePath, columns, rows, stride, pivot);
            relaxRemaining<<<dim3(tiles - 1, tiles - 1), threads>>>(deviceDist, devicePath, columns, rows, stride, pivot);
        }
        checkCuda(cudaGetLastError());
    }
    checkCuda(cudaMemcpy2D(dist.data(), width, deviceDist, pitch, width, numNodes, cudaMemcpyDeviceToHost));
    checkCuda(cudaMemcpy2D(path.data(), width, devicePath, pitch, width, numNodes, cudaMemcpyDeviceToHost));
    checkCuda(cudaFree(rows));
    checkCuda(cudaFree(columns));
    checkCuda(cudaFree(devicePath));
    checkCuda(cudaFree(deviceDist));
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
    
    // Initialize the CUDA runtime before timing the computation and transfers.
    checkCuda(cudaFree(nullptr));

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
    double gflops = ops / std::chrono::duration<double>(end - start).count() / 1e9;
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
