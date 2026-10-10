#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
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

constexpr unsigned int TILE = 32;
constexpr unsigned int THREAD_ROWS = 16;
constexpr unsigned int OUTPUT_TILE = 2 * TILE;

void checkCuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error (%s): %s\n", operation, cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}

struct DeviceBuffer {
    unsigned int* data = nullptr;
    explicit DeviceBuffer(size_t bytes) {
        checkCuda(cudaMalloc(&data, bytes), "allocate device matrix");
    }
    ~DeviceBuffer() { cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

// A rank is one plus the largest internal vertex on a path; a direct edge has
// rank zero. Among equal distances retain the smallest rank. This reproduces
// the original ascending-k, strict-improvement path matrix even though tiling
// changes the order in which paths are discovered.
__device__ __forceinline__ void relax(unsigned int& distance, unsigned int& rank,
                                     unsigned int candidate, unsigned int leftRank,
                                     unsigned int rightRank, unsigned int viaRank) {
    if (candidate <= distance) {
        const unsigned int candidateRank = max(max(leftRank, rightRank), viaRank);
        rank = candidate < distance ? candidateRank : min(rank, candidateRank);
        distance = candidate;
    }
}

__global__ void initializeDeviceMatrices(unsigned int* dist, unsigned int* ranks,
                                         size_t stride, size_t numNodes) {
    const size_t count = stride * stride;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const size_t row = index / stride;
        const size_t col = index % stride;
        if (row >= numNodes || col >= numNodes) {
            dist[index] = row == col ? 0 : INF;
        }
        ranks[index] = 0;
    }
}

__global__ void updatePivot(unsigned int* dist, unsigned int* ranks,
                            size_t stride, unsigned int pivot) {
    __shared__ unsigned int tile[TILE][TILE];
    __shared__ unsigned int tileRank[TILE][TILE];
    const unsigned int col = threadIdx.x;
    const size_t base = static_cast<size_t>(pivot) * TILE;
    #pragma unroll
    for (unsigned int r = 0; r < TILE / THREAD_ROWS; ++r) {
        const unsigned int row = threadIdx.y + r * THREAD_ROWS;
        const size_t index = (base + row) * stride + base + col;
        tile[row][col] = dist[index];
        tileRank[row][col] = ranks[index];
    }
    __syncthreads();

    #pragma unroll 1
    for (unsigned int k = 0; k < TILE; ++k) {
        #pragma unroll
        for (unsigned int r = 0; r < TILE / THREAD_ROWS; ++r) {
            const unsigned int row = threadIdx.y + r * THREAD_ROWS;
            // With a zero diagonal these elements cannot improve. Leaving
            // them untouched also keeps the operands read by all threads
            // stable until the barrier, without shared-memory data races.
            if (row != k && col != k) {
                unsigned int value = tile[row][col];
                unsigned int rank = tileRank[row][col];
                relax(value, rank, tile[row][k] + tile[k][col],
                      tileRank[row][k], tileRank[k][col], base + k + 1);
                tile[row][col] = value;
                tileRank[row][col] = rank;
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (unsigned int r = 0; r < TILE / THREAD_ROWS; ++r) {
        const unsigned int row = threadIdx.y + r * THREAD_ROWS;
        const size_t index = (base + row) * stride + base + col;
        dist[index] = tile[row][col];
        ranks[index] = tileRank[row][col];
    }
}

// Row and column tiles use the completed pivot tile. Blocks operate on disjoint
// tiles, and each k only reads an unchanged row/column of its evolving tile.
__global__ void updatePanels(unsigned int* dist, unsigned int* ranks,
                             size_t stride, unsigned int pivot) {
    __shared__ unsigned int diagonal[TILE][TILE];
    __shared__ unsigned int diagonalRank[TILE][TILE];
    __shared__ unsigned int panel[TILE][TILE];
    __shared__ unsigned int panelRank[TILE][TILE];
    const bool columnPanel = blockIdx.y != 0;
    const unsigned int other = blockIdx.x + (blockIdx.x >= pivot);
    const size_t pivotBase = static_cast<size_t>(pivot) * TILE;
    const size_t rowBase = columnPanel ? static_cast<size_t>(other) * TILE : pivotBase;
    const size_t colBase = columnPanel ? pivotBase : static_cast<size_t>(other) * TILE;
    const unsigned int col = threadIdx.x;

    #pragma unroll
    for (unsigned int r = 0; r < TILE / THREAD_ROWS; ++r) {
        const unsigned int row = threadIdx.y + r * THREAD_ROWS;
        const size_t diagonalIndex = (pivotBase + row) * stride + pivotBase + col;
        const size_t index = (rowBase + row) * stride + colBase + col;
        diagonal[row][col] = dist[diagonalIndex];
        diagonalRank[row][col] = ranks[diagonalIndex];
        panel[row][col] = dist[index];
        panelRank[row][col] = ranks[index];
    }
    __syncthreads();

    #pragma unroll 1
    for (unsigned int k = 0; k < TILE; ++k) {
        #pragma unroll
        for (unsigned int r = 0; r < TILE / THREAD_ROWS; ++r) {
            const unsigned int row = threadIdx.y + r * THREAD_ROWS;
            if (columnPanel ? col != k : row != k) {
                unsigned int value = panel[row][col];
                unsigned int rank = panelRank[row][col];
                if (columnPanel) {
                    relax(value, rank, panel[row][k] + diagonal[k][col],
                          panelRank[row][k], diagonalRank[k][col], pivotBase + k + 1);
                } else {
                    relax(value, rank, diagonal[row][k] + panel[k][col],
                          diagonalRank[row][k], panelRank[k][col], pivotBase + k + 1);
                }
                panel[row][col] = value;
                panelRank[row][col] = rank;
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (unsigned int r = 0; r < TILE / THREAD_ROWS; ++r) {
        const unsigned int row = threadIdx.y + r * THREAD_ROWS;
        const size_t index = (rowBase + row) * stride + colBase + col;
        dist[index] = panel[row][col];
        ranks[index] = panelRank[row][col];
    }
}

__device__ __forceinline__ size_t skipPivot(unsigned int coordinate,
                                           unsigned int pivot) {
    return coordinate + static_cast<size_t>(coordinate / TILE >= pivot) * TILE;
}

// Each block updates four 32x32 tiles using two 64x32 panels. All eight results
// owned by a thread stay in registers throughout the reduction over k.
__global__ __launch_bounds__(TILE * THREAD_ROWS, 2)
void updateRemaining(unsigned int* __restrict__ dist,
                     unsigned int* __restrict__ ranks,
                     size_t stride, unsigned int pivot) {
    __shared__ unsigned int left[OUTPUT_TILE][TILE];
    __shared__ unsigned int leftRank[OUTPUT_TILE][TILE];
    __shared__ unsigned int right[TILE][OUTPUT_TILE];
    __shared__ unsigned int rightRank[TILE][OUTPUT_TILE];
    unsigned int values[OUTPUT_TILE / THREAD_ROWS][2];
    unsigned int valueRanks[OUTPUT_TILE / THREAD_ROWS][2];
    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const unsigned int rowBase = blockIdx.y * OUTPUT_TILE;
    const unsigned int colBase = blockIdx.x * OUTPUT_TILE;
    const size_t pivotBase = static_cast<size_t>(pivot) * TILE;

    #pragma unroll
    for (unsigned int r = 0; r < OUTPUT_TILE / THREAD_ROWS; ++r) {
        const unsigned int localRow = ty + r * THREAD_ROWS;
        const size_t row = skipPivot(rowBase + localRow, pivot);
        const size_t index = row * stride + pivotBase + tx;
        left[localRow][tx] = row < stride ? dist[index] : INF;
        leftRank[localRow][tx] = row < stride ? ranks[index] : 0;
        #pragma unroll
        for (unsigned int c = 0; c < 2; ++c) {
            const size_t col = skipPivot(colBase + tx + c * TILE, pivot);
            const size_t resultIndex = row * stride + col;
            values[r][c] = row < stride && col < stride ? dist[resultIndex] : INF;
            valueRanks[r][c] = row < stride && col < stride ? ranks[resultIndex] : 0;
        }
    }
    #pragma unroll
    for (unsigned int r = 0; r < TILE / THREAD_ROWS; ++r) {
        const unsigned int localRow = ty + r * THREAD_ROWS;
        #pragma unroll
        for (unsigned int c = 0; c < 2; ++c) {
            const unsigned int localCol = tx + c * TILE;
            const size_t col = skipPivot(colBase + localCol, pivot);
            const size_t index = (pivotBase + localRow) * stride + col;
            right[localRow][localCol] = col < stride ? dist[index] : INF;
            rightRank[localRow][localCol] = col < stride ? ranks[index] : 0;
        }
    }
    __syncthreads();

    #pragma unroll 1
    for (unsigned int k = 0; k < TILE; ++k) {
        const unsigned int viaRank = pivotBase + k + 1;
        unsigned int b[2];
        unsigned int bRank[2];
        #pragma unroll
        for (unsigned int c = 0; c < 2; ++c) {
            b[c] = right[k][tx + c * TILE];
            bRank[c] = max(rightRank[k][tx + c * TILE], viaRank);
        }
        #pragma unroll
        for (unsigned int r = 0; r < OUTPUT_TILE / THREAD_ROWS; ++r) {
            const unsigned int a = left[ty + r * THREAD_ROWS][k];
            const unsigned int aRank = leftRank[ty + r * THREAD_ROWS][k];
            #pragma unroll
            for (unsigned int c = 0; c < 2; ++c) {
                relax(values[r][c], valueRanks[r][c], a + b[c], aRank, bRank[c], 0);
            }
        }
    }
    #pragma unroll
    for (unsigned int r = 0; r < OUTPUT_TILE / THREAD_ROWS; ++r) {
        const size_t row = skipPivot(rowBase + ty + r * THREAD_ROWS, pivot);
        #pragma unroll
        for (unsigned int c = 0; c < 2; ++c) {
            const size_t col = skipPivot(colBase + tx + c * TILE, pivot);
            if (row < stride && col < stride) {
                const size_t index = row * stride + col;
                dist[index] = values[r][c];
                ranks[index] = valueRanks[r][c];
            }
        }
    }
}

__global__ void restorePathMatrix(unsigned int* ranks, size_t stride) {
    const size_t count = stride * stride;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const unsigned int rank = ranks[index];
        // The original initializer stores the source row for direct edges.
        ranks[index] = rank ? rank - 1 : static_cast<unsigned int>(index / stride);
    }
}

} // namespace

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes) {
    if (numNodes == 0) return;
    const size_t stride = (numNodes + TILE - 1) / TILE * TILE;
    if (stride > std::numeric_limits<size_t>::max() / sizeof(unsigned int) / stride) {
        fprintf(stderr, "Padded graph is too large\n");
        std::exit(EXIT_FAILURE);
    }
    const unsigned int tiles = stride / TILE;
    const size_t pitch = stride * sizeof(unsigned int);
    const size_t hostPitch = numNodes * sizeof(unsigned int);
    DeviceBuffer deviceDist(stride * pitch);
    DeviceBuffer deviceRanks(stride * pitch);
    checkCuda(cudaMemcpy2D(deviceDist.data, pitch, dist.data(), hostPitch,
                           hostPitch, numNodes, cudaMemcpyHostToDevice), "upload distances");

    const unsigned int linearBlocks = static_cast<unsigned int>(
        std::min((stride * stride + 255) / 256, static_cast<size_t>(65535)));
    initializeDeviceMatrices<<<linearBlocks, 256>>>(deviceDist.data, deviceRanks.data,
                                                   stride, numNodes);
    checkCuda(cudaGetLastError(), "initialize device matrices");
    const dim3 threads(TILE, THREAD_ROWS);
    const dim3 remainingGrid(tiles / 2, tiles / 2);
    for (unsigned int pivot = 0; pivot < tiles; ++pivot) {
        // Launches in the default stream provide the global dependencies
        // between phases; no host synchronization is needed inside the loop.
        updatePivot<<<1, threads>>>(deviceDist.data, deviceRanks.data, stride, pivot);
        checkCuda(cudaGetLastError(), "update pivot tile");
        if (tiles > 1) {
            updatePanels<<<dim3(tiles - 1, 2), threads>>>(
                deviceDist.data, deviceRanks.data, stride, pivot);
            checkCuda(cudaGetLastError(), "update pivot row and column");
            updateRemaining<<<remainingGrid, threads>>>(
                deviceDist.data, deviceRanks.data, stride, pivot);
            checkCuda(cudaGetLastError(), "update remaining tiles");
        }
    }
    restorePathMatrix<<<linearBlocks, 256>>>(deviceRanks.data, stride);
    checkCuda(cudaGetLastError(), "restore path matrix");
    // Synchronous downloads include execution in the reported benchmark time.
    checkCuda(cudaMemcpy2D(dist.data(), hostPitch, deviceDist.data, pitch,
                           hostPitch, numNodes, cudaMemcpyDeviceToHost), "download distances");
    checkCuda(cudaMemcpy2D(path.data(), hostPitch, deviceRanks.data, pitch,
                           hostPitch, numNodes, cudaMemcpyDeviceToHost), "download paths");
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
    
    if (numNodes > std::numeric_limits<unsigned int>::max() - TILE ||
        (numNodes && numNodes > std::numeric_limits<size_t>::max() /
                                   sizeof(unsigned int) / numNodes)) {
        fprintf(stderr, "Graph is too large\n");
        return 1;
    }

    // Initialize the CUDA context outside the timed algorithm, just as host
    // matrix allocation and graph initialization are outside it.
    checkCuda(cudaFree(nullptr), "initialize CUDA runtime");

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
    const double seconds = std::chrono::duration<double>(end - start).count();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate operations per second
    // Floyd-Warshall has O(n³) complexity
    double ops = (double)numNodes * numNodes * numNodes;
    double gflops = seconds > 0 ? ops / seconds / 1e9 : 0;
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
