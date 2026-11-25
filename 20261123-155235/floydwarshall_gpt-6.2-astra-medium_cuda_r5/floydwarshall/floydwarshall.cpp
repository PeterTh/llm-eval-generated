#include <cuda_runtime.h>

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
        std::fill_n(path.data() + j * numNodes, numNodes, static_cast<unsigned>(j));
    }
}

namespace {
constexpr int tile = 32;
constexpr int blockRows = 8;
constexpr unsigned noUpdate = ~0u;

void checkCuda(cudaError_t status) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

// For each k, the diagonal and panel kernels save the entering column D[i,k]
// and row D[k,j]. Using these snapshots in the other tiles preserves the
// original order of strict improvements, including ties in the path matrix.
enum class Region { diagonal, column, row };

template<Region region>
__device__ __forceinline__ void solvePanel(unsigned* dist, unsigned* path, size_t stride, int pivot,
                           unsigned* columns, unsigned* rows) {
    __shared__ unsigned work[tile][tile];
    __shared__ unsigned diagonal[tile][tile];
    const int x = threadIdx.x;
    const int y = threadIdx.y;
    const int other = blockIdx.x + (static_cast<int>(blockIdx.x) >= pivot);
    const int tileRow = region == Region::column ? other : pivot;
    const int tileCol = region == Region::row ? other : pivot;
    const size_t firstRow = size_t(tileRow) * tile;
    const size_t firstCol = size_t(tileCol) * tile;
    unsigned via[tile / blockRows];
    unsigned colSnapshot[tile / blockRows];
    unsigned rowSnapshot[tile / blockRows];
    #pragma unroll
    for (int t = 0; t < tile / blockRows; ++t) {
        const int r = y + t * blockRows;
        const size_t pos = (firstRow + r) * stride + firstCol + x;
        work[r][x] = dist[pos];
        via[t] = noUpdate;
        if constexpr (region == Region::column)
            diagonal[r][x] = rows[r * stride + size_t(pivot) * tile + x];
        if constexpr (region == Region::row)
            diagonal[r][x] = columns[(size_t(pivot) * tile + r) * tile + x];
    }

    for (int k = 0; k < tile; ++k) {
        __syncthreads();
        #pragma unroll
        for (int t = 0; t < tile / blockRows; ++t) {
            const int r = y + t * blockRows;
            const unsigned left = region == Region::row ? diagonal[r][k] : work[r][k];
            const unsigned right = region == Region::column ? diagonal[k][x] : work[k][x];
            if constexpr (region != Region::row)
                if (x == k) colSnapshot[t] = left;
            if constexpr (region != Region::column)
                if (r == k) rowSnapshot[t] = right;

            // With a zero pivot diagonal, neither input can strictly improve.
            // Leave these shared locations read-only throughout this iteration.
            if ((region == Region::row || x != k) && (region == Region::column || r != k)) {
                const unsigned candidate = left + right;
                if (candidate < work[r][x]) {
                    work[r][x] = candidate;
                    via[t] = pivot * tile + k;
                }
            }
        }
    }
    #pragma unroll
    for (int t = 0; t < tile / blockRows; ++t) {
        const int r = y + t * blockRows;
        const size_t pos = (firstRow + r) * stride + firstCol + x;
        dist[pos] = work[r][x];
        if (via[t] != noUpdate) path[pos] = via[t];
        if constexpr (region != Region::row)
            columns[(firstRow + r) * tile + x] = colSnapshot[t];
        if constexpr (region != Region::column)
            rows[r * stride + firstCol + x] = rowSnapshot[t];
    }
}

__global__ void solveDiagonal(unsigned* dist, unsigned* path, size_t stride, int pivot,
                              unsigned* columns, unsigned* rows) {
    solvePanel<Region::diagonal>(dist, path, stride, pivot, columns, rows);
}

__global__ void solvePanels(unsigned* dist, unsigned* path, size_t stride, int pivot,
                            unsigned* columns, unsigned* rows) {
    if (blockIdx.y == 0)
        solvePanel<Region::column>(dist, path, stride, pivot, columns, rows);
    else
        solvePanel<Region::row>(dist, path, stride, pivot, columns, rows);
}

__global__ void updateTiles(unsigned* dist, unsigned* path, size_t stride, int pivot,
                            const unsigned* columns, const unsigned* rows) {
    __shared__ unsigned left[tile][tile];
    __shared__ unsigned right[tile][tile];
    const int x = threadIdx.x;
    const int y = threadIdx.y;
    const size_t firstRow = (blockIdx.y + (static_cast<int>(blockIdx.y) >= pivot)) * tile;
    const size_t firstCol = (blockIdx.x + (static_cast<int>(blockIdx.x) >= pivot)) * tile;
    unsigned best[tile / blockRows];
    unsigned via[tile / blockRows];
    #pragma unroll
    for (int t = 0; t < tile / blockRows; ++t) {
        const int r = y + t * blockRows;
        left[r][x] = columns[(firstRow + r) * tile + x];
        right[r][x] = rows[r * stride + firstCol + x];
        const size_t pos = (firstRow + r) * stride + firstCol + x;
        best[t] = dist[pos];
        via[t] = noUpdate;
    }
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < tile; ++k) {
        #pragma unroll
        for (int t = 0; t < tile / blockRows; ++t) {
            const unsigned candidate = left[y + t * blockRows][k] + right[k][x];
            if (candidate < best[t]) via[t] = pivot * tile + k;
            best[t] = min(best[t], candidate);
        }
    }
    #pragma unroll
    for (int t = 0; t < tile / blockRows; ++t) {
        const size_t pos = (firstRow + y + t * blockRows) * stride + firstCol + x;
        dist[pos] = best[t];
        if (via[t] != noUpdate) path[pos] = via[t];
    }
}
}

double floydWarshall(std::vector<unsigned>& dist, std::vector<unsigned>& path, size_t n) {
    if (n == 0) return 0.0;
    const int tiles = (n + tile - 1) / tile;
    const size_t stride = size_t(tiles) * tile;
    const size_t elements = stride * stride;
    const size_t panelSize = stride * tile;
    unsigned* deviceDist;
    checkCuda(cudaMalloc(&deviceDist, (2 * elements + 2 * panelSize) * sizeof(unsigned)));
    unsigned* devicePath = deviceDist + elements;
    unsigned* columns = devicePath + elements;
    unsigned* rows = columns + panelSize;

    // Padded vertices have no edges. The byte pattern is larger than any
    // generated distance and remains representable when added to itself.
    if (stride != n) {
        checkCuda(cudaMemset(deviceDist, 0x3f, elements * sizeof(unsigned)));
    }
    const size_t pitch = stride * sizeof(unsigned);
    const size_t width = n * sizeof(unsigned);
    checkCuda(cudaMemcpy2D(deviceDist, pitch, dist.data(), width, width, n, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy2D(devicePath, pitch, path.data(), width, width, n, cudaMemcpyHostToDevice));
    const dim3 threads(tile, blockRows);
    // As in the CPU benchmark, time the computation on initialized matrices.
    checkCuda(cudaDeviceSynchronize());
    const auto start = std::chrono::steady_clock::now();
    for (int pivot = 0; pivot < tiles; ++pivot) {
        solveDiagonal<<<1, threads>>>(deviceDist, devicePath, stride, pivot, columns, rows);
        if (tiles > 1) {
            solvePanels<<<dim3(tiles - 1, 2), threads>>>(deviceDist, devicePath, stride, pivot, columns, rows);
            updateTiles<<<dim3(tiles - 1, tiles - 1), threads>>>(deviceDist, devicePath, stride, pivot, columns, rows);
        }
    }
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());
    const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start;
    checkCuda(cudaMemcpy2D(dist.data(), width, deviceDist, pitch, width, n, cudaMemcpyDeviceToHost));
    checkCuda(cudaMemcpy2D(path.data(), width, devicePath, pitch, width, n, cudaMemcpyDeviceToHost));
    checkCuda(cudaFree(deviceDist));
    return elapsed.count();
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
    const double seconds = floydWarshall(dist, path, numNodes);
    printf("Computation time: %.3f ms\n", seconds * 1000.0);

    double ops = static_cast<double>(numNodes) * numNodes * numNodes;
    double gflops = seconds > 0.0 ? ops / seconds / 1e9 : 0.0;
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
