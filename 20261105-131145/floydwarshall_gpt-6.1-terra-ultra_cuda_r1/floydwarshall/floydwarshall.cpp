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

// Three-stage blocked Floyd-Warshall.  Each tile is 32 x 32, and each
// thread owns four elements in a column.  Padding allows all blocks to use
// the same coalesced loads, including the final partial tile.
constexpr int TILE = 32;

static void checkCuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", operation, cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}

__global__ void closePivot(unsigned int* dist, size_t pitch, int pivot) {
    __shared__ unsigned int tile[TILE][TILE];
    const int x = threadIdx.x;
    const int base = pivot * TILE;
    for (int y = threadIdx.y; y < TILE; y += blockDim.y)
        tile[y][x] = dist[(base + y) * pitch + base + x];
    __syncthreads();

    for (int k = 0; k < TILE; ++k) {
        for (int y = threadIdx.y; y < TILE; y += blockDim.y) {
            // Real pivot diagonals are zero; padded pivots cannot improve any entry.
            if (y != k && x != k) {
                unsigned int candidate = tile[y][k] + tile[k][x];
                tile[y][x] = min(tile[y][x], candidate);
            }
        }
        __syncthreads();
    }
    for (int y = threadIdx.y; y < TILE; y += blockDim.y)
        dist[(base + y) * pitch + base + x] = tile[y][x];
}

// One block processes a tile in the pivot row or pivot column.
__global__ void closeEdges(unsigned int* dist, size_t pitch, int pivot) {
    __shared__ unsigned int diagonal[TILE][TILE];
    __shared__ unsigned int edge[TILE][TILE];
    int tileIndex = blockIdx.x;
    if (tileIndex >= pivot) ++tileIndex;
    const bool row = blockIdx.y == 0;
    const int tileRow = row ? pivot : tileIndex;
    const int tileCol = row ? tileIndex : pivot;
    const int x = threadIdx.x;
    const int base = pivot * TILE;
    for (int y = threadIdx.y; y < TILE; y += blockDim.y) {
        diagonal[y][x] = dist[(base + y) * pitch + base + x];
        edge[y][x] = dist[(tileRow * TILE + y) * pitch + tileCol * TILE + x];
    }
    __syncthreads();

    for (int k = 0; k < TILE; ++k) {
        for (int y = threadIdx.y; y < TILE; y += blockDim.y) {
            if (row ? y != k : x != k) {
                unsigned int candidate = row ? diagonal[y][k] + edge[k][x]
                                             : edge[y][k] + diagonal[k][x];
                edge[y][x] = min(edge[y][x], candidate);
            }
        }
        __syncthreads();
    }
    for (int y = threadIdx.y; y < TILE; y += blockDim.y)
        dist[(tileRow * TILE + y) * pitch + tileCol * TILE + x] = edge[y][x];
}

__global__ void closeRemaining(unsigned int* dist, size_t pitch, int pivot) {
    __shared__ unsigned int left[TILE][TILE];
    __shared__ unsigned int right[TILE][TILE];
    int tileCol = blockIdx.x;
    int tileRow = blockIdx.y;
    if (tileCol >= pivot) ++tileCol;
    if (tileRow >= pivot) ++tileRow;
    const int x = threadIdx.x;
    unsigned int values[TILE / 8];
    #pragma unroll
    for (int r = 0; r < TILE / 8; ++r) {
        int y = threadIdx.y + r * 8;
        left[y][x] = dist[(tileRow * TILE + y) * pitch + pivot * TILE + x];
        right[y][x] = dist[(pivot * TILE + y) * pitch + tileCol * TILE + x];
        values[r] = dist[(tileRow * TILE + y) * pitch + tileCol * TILE + x];
    }
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < TILE; ++k) {
        unsigned int rk = right[k][x];
        #pragma unroll
        for (int r = 0; r < TILE / 8; ++r) {
            int y = threadIdx.y + r * 8;
            values[r] = min(values[r], left[y][k] + rk);
        }
    }
    #pragma unroll
    for (int r = 0; r < TILE / 8; ++r) {
        int y = threadIdx.y + r * 8;
        dist[(tileRow * TILE + y) * pitch + tileCol * TILE + x] = values[r];
    }
}

class GpuDistanceMatrix {
public:
    explicit GpuDistanceMatrix(const std::vector<unsigned int>& dist, size_t numNodes)
        : numNodes_(numNodes), padded_(((numNodes + TILE - 1) / TILE) * TILE),
          tiles_(static_cast<int>(padded_ / TILE)), deviceDist_(nullptr) {
        if (numNodes_ == 0) return;
        checkCuda(cudaMalloc(&deviceDist_, padded_ * padded_ * sizeof(unsigned int)), "cudaMalloc");
        // 0x3b3b3b3b is a large finite distance whose sum does not overflow.
        checkCuda(cudaMemset(deviceDist_, 0x3b, padded_ * padded_ * sizeof(unsigned int)), "cudaMemset");
        checkCuda(cudaMemcpy2D(deviceDist_, padded_ * sizeof(unsigned int), dist.data(),
                               numNodes_ * sizeof(unsigned int), numNodes_ * sizeof(unsigned int),
                               numNodes_, cudaMemcpyHostToDevice), "copy to device");
    }

    ~GpuDistanceMatrix() {
        if (deviceDist_) checkCuda(cudaFree(deviceDist_), "cudaFree");
    }
    GpuDistanceMatrix(const GpuDistanceMatrix&) = delete;
    GpuDistanceMatrix& operator=(const GpuDistanceMatrix&) = delete;

    void compute() {
        if (numNodes_ == 0) return;
        const dim3 threads(TILE, 8);
        for (int pivot = 0; pivot < tiles_; ++pivot) {
            closePivot<<<1, threads>>>(deviceDist_, padded_, pivot);
            if (tiles_ > 1) {
                closeEdges<<<dim3(tiles_ - 1, 2), threads>>>(deviceDist_, padded_, pivot);
                closeRemaining<<<dim3(tiles_ - 1, tiles_ - 1), threads>>>(deviceDist_, padded_, pivot);
            }
        }
        checkCuda(cudaGetLastError(), "kernel launch");
        checkCuda(cudaDeviceSynchronize(), "kernel execution");
    }

    void copyTo(std::vector<unsigned int>& dist) const {
        if (numNodes_ == 0) return;
        checkCuda(cudaMemcpy2D(dist.data(), numNodes_ * sizeof(unsigned int), deviceDist_,
                               padded_ * sizeof(unsigned int), numNodes_ * sizeof(unsigned int),
                               numNodes_, cudaMemcpyDeviceToHost), "copy from device");
    }

private:
    size_t numNodes_;
    size_t padded_;
    int tiles_;
    unsigned int* deviceDist_;
};

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
    
    // Initialize
    printf("Initializing graph...\n");
    initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
    GpuDistanceMatrix gpuDist(dist, numNodes);
    
    // Run Floyd-Warshall
    printf("Computing shortest paths...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    gpuDist.compute();
    
    auto end = std::chrono::high_resolution_clock::now();
    gpuDist.copyTo(dist);
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
