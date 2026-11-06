#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cuda_runtime.h>
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

// Each matrix element combines a distance with the smallest maximum intermediate
// vertex among shortest paths. A zero rank denotes a direct edge. The rank
// makes strict-update path semantics independent of the blocked execution order.
using Packed = unsigned long long;
constexpr int TILE = 32;
constexpr int THREADS = 256;

static void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

__device__ __forceinline__ Packed combine(Packed left, Packed right, unsigned int k) {
    const unsigned int distance = static_cast<unsigned int>(left >> 32) +
                                  static_cast<unsigned int>(right >> 32);
    unsigned int rank = max(static_cast<unsigned int>(left), static_cast<unsigned int>(right));
    rank = max(rank, k + 1);
    return (static_cast<Packed>(distance) << 32) | rank;
}

// Pack and pad on the GPU so host-side matrix conversion does not become a
// serial bottleneck. Padded entries cannot participate in valid paths.
__global__ void packKernel(const unsigned int* dist, Packed* packed,
                           size_t n, size_t stride) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= stride * stride) return;
    const size_t row = index / stride, col = index % stride;
    const unsigned int value = row < n && col < n ? dist[row * n + col] : INF;
    packed[index] = static_cast<Packed>(value) << 32;
}

__global__ void unpackKernel(const Packed* packed, unsigned int* dist,
                             unsigned int* path, size_t n, size_t stride) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= n * n) return;
    const size_t row = index / n, col = index % n;
    const Packed value = packed[row * stride + col];
    dist[index] = static_cast<unsigned int>(value >> 32);
    const unsigned int rank = static_cast<unsigned int>(value);
    if (rank) path[index] = rank - 1;
}

// Close the diagonal tile for this group of intermediate vertices.
__global__ void pivotKernel(Packed* matrix, size_t stride, int pivot) {
    __shared__ Packed tile[TILE][TILE];
    const int tid = threadIdx.x;
    for (int q = tid; q < TILE * TILE; q += THREADS) {
        const int row = q / TILE, col = q % TILE;
        tile[row][col] = matrix[(static_cast<size_t>(pivot * TILE + row) * stride) + pivot * TILE + col];
    }
    __syncthreads();
    for (int k = 0; k < TILE; ++k) {
        for (int q = tid; q < TILE * TILE; q += THREADS) {
            const int row = q / TILE, col = q % TILE;
            if (row != k && col != k) {
                const Packed candidate = combine(tile[row][k], tile[k][col], pivot * TILE + k);
                if (candidate < tile[row][col]) tile[row][col] = candidate;
            }
        }
        __syncthreads();
    }
    for (int q = tid; q < TILE * TILE; q += THREADS) {
        const int row = q / TILE, col = q % TILE;
        matrix[(static_cast<size_t>(pivot * TILE + row) * stride) + pivot * TILE + col] = tile[row][col];
    }
}

// Close the tiles in the pivot row and column using the closed diagonal tile.
__global__ void edgeKernel(Packed* matrix, size_t stride, int pivot, int tiles) {
    __shared__ Packed diagonal[TILE][TILE];
    __shared__ Packed edge[TILE][TILE];
    const int index = blockIdx.x % (tiles - 1);
    const int other = index + (index >= pivot);
    const bool column = blockIdx.x >= static_cast<unsigned int>(tiles - 1);
    const int tileRow = column ? other : pivot;
    const int tileCol = column ? pivot : other;
    const int tid = threadIdx.x;
    for (int q = tid; q < TILE * TILE; q += THREADS) {
        const int row = q / TILE, col = q % TILE;
        diagonal[row][col] = matrix[static_cast<size_t>(pivot * TILE + row) * stride + pivot * TILE + col];
        edge[row][col] = matrix[static_cast<size_t>(tileRow * TILE + row) * stride + tileCol * TILE + col];
    }
    __syncthreads();
    for (int k = 0; k < TILE; ++k) {
        for (int q = tid; q < TILE * TILE; q += THREADS) {
            const int row = q / TILE, col = q % TILE;
            // The pivot row or column is unchanged; skipping it also avoids
            // concurrent reads and writes to shared memory in this iteration.
            if ((column && col != k) || (!column && row != k)) {
                const Packed left = column ? edge[row][k] : diagonal[row][k];
                const Packed right = column ? diagonal[k][col] : edge[k][col];
                const Packed candidate = combine(left, right, pivot * TILE + k);
                if (candidate < edge[row][col]) edge[row][col] = candidate;
            }
        }
        __syncthreads();
    }
    for (int q = tid; q < TILE * TILE; q += THREADS) {
        const int row = q / TILE, col = q % TILE;
        matrix[static_cast<size_t>(tileRow * TILE + row) * stride + tileCol * TILE + col] = edge[row][col];
    }
}

// All remaining tiles are independent once the pivot row and column are closed.
__global__ void remainingKernel(Packed* matrix, size_t stride, int pivot) {
    __shared__ Packed left[TILE][TILE];
    __shared__ Packed right[TILE][TILE];
    const int tileRow = blockIdx.y + (blockIdx.y >= static_cast<unsigned int>(pivot));
    const int tileCol = blockIdx.x + (blockIdx.x >= static_cast<unsigned int>(pivot));
    const int tid = threadIdx.x;
    Packed values[4];
    for (int a = 0; a < 4; ++a) {
        const int q = tid + a * THREADS;
        const int row = q / TILE, col = q % TILE;
        left[row][col] = matrix[static_cast<size_t>(tileRow * TILE + row) * stride + pivot * TILE + col];
        right[row][col] = matrix[static_cast<size_t>(pivot * TILE + row) * stride + tileCol * TILE + col];
        values[a] = matrix[static_cast<size_t>(tileRow * TILE + row) * stride + tileCol * TILE + col];
    }
    __syncthreads();
    for (int k = 0; k < TILE; ++k) {
        for (int a = 0; a < 4; ++a) {
            const int q = tid + a * THREADS;
            const int row = q / TILE, col = q % TILE;
            const Packed candidate = combine(left[row][k], right[k][col], pivot * TILE + k);
            values[a] = min(values[a], candidate);
        }
    }
    for (int a = 0; a < 4; ++a) {
        const int q = tid + a * THREADS;
        const int row = q / TILE, col = q % TILE;
        matrix[static_cast<size_t>(tileRow * TILE + row) * stride + tileCol * TILE + col] = values[a];
    }
}

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes) {
    if (numNodes == 0) return;
    const size_t stride = ((numNodes + TILE - 1) / TILE) * TILE;
    const int tiles = static_cast<int>(stride / TILE);
    const size_t elements = numNodes * numNodes;
    const size_t inputBytes = elements * sizeof(unsigned int);
    const size_t packedBytes = stride * stride * sizeof(Packed);
    Packed* device = nullptr;
    unsigned int* deviceInputs = nullptr;
    checkCuda(cudaMalloc(&device, packedBytes), "packed matrix allocation");
    checkCuda(cudaMalloc(&deviceInputs, 2 * inputBytes), "input matrix allocation");
    unsigned int* devicePath = deviceInputs + elements;
    checkCuda(cudaMemcpy(deviceInputs, dist.data(), inputBytes, cudaMemcpyHostToDevice), "distance upload");
    checkCuda(cudaMemcpy(devicePath, path.data(), inputBytes, cudaMemcpyHostToDevice), "path upload");
    packKernel<<<(stride * stride + THREADS - 1) / THREADS, THREADS>>>(deviceInputs, device, numNodes, stride);
    for (int p = 0; p < tiles; ++p) {
        pivotKernel<<<1, THREADS>>>(device, stride, p);
        if (tiles > 1) {
            edgeKernel<<<2 * (tiles - 1), THREADS>>>(device, stride, p, tiles);
            const dim3 grid(tiles - 1, tiles - 1);
            remainingKernel<<<grid, THREADS>>>(device, stride, p);
        }
    }
    unpackKernel<<<(elements + THREADS - 1) / THREADS, THREADS>>>(device, deviceInputs, devicePath, numNodes, stride);
    checkCuda(cudaGetLastError(), "kernel launch");
    checkCuda(cudaMemcpy(dist.data(), deviceInputs, inputBytes, cudaMemcpyDeviceToHost), "distance download");
    checkCuda(cudaMemcpy(path.data(), devicePath, inputBytes, cudaMemcpyDeviceToHost), "path download");
    checkCuda(cudaFree(deviceInputs), "input matrix deallocation");
    checkCuda(cudaFree(device), "packed matrix deallocation");
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
    
    // Exclude one-time CUDA runtime initialization from the benchmark timing.
    if (numNodes != 0) checkCuda(cudaFree(nullptr), "runtime initialization");

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
