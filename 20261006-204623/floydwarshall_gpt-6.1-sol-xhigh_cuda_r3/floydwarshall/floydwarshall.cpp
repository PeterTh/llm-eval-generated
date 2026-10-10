#include <algorithm>
#include <cerrno>
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

namespace {

constexpr int TILE_SIZE = 32;
constexpr int THREAD_ROWS = 8;
constexpr int ROWS_PER_THREAD = TILE_SIZE / THREAD_ROWS;

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error during %s: %s\n", operation,
                cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

struct Tile {
    unsigned int distance[TILE_SIZE][TILE_SIZE];
    unsigned int witness[TILE_SIZE][TILE_SIZE];
};

// A witness is one plus the largest internal vertex on a shortest path; zero
// denotes an unchanged direct edge. Among equal distances prefer the smallest
// witness. This preserves the original ascending-k, strict-improvement path
// matrix even though the blocked algorithm visits candidates in another order.
__device__ __forceinline__ void relax(unsigned int& distance,
                                      unsigned int& witness,
                                      unsigned int leftDistance,
                                      unsigned int rightDistance,
                                      unsigned int leftWitness,
                                      unsigned int rightWitness,
                                      unsigned int intermediate) {
    const unsigned int candidate = leftDistance + rightDistance;
    if (candidate <= distance) {
        const unsigned int candidateWitness =
            max(max(leftWitness, rightWitness), intermediate);
        if (candidate < distance || candidateWitness < witness) {
            distance = candidate;
            witness = candidateWitness;
        }
    }
}

__device__ __forceinline__ void loadTile(Tile& tile,
                                         const unsigned int* distance,
                                         const unsigned int* witness,
                                         int tileRow, int tileColumn,
                                         size_t stride) {
    const int column = threadIdx.x;
    #pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; ++r) {
        const int row = threadIdx.y + r * THREAD_ROWS;
        const size_t index = (size_t(tileRow) * TILE_SIZE + row) * stride +
                            size_t(tileColumn) * TILE_SIZE + column;
        tile.distance[row][column] = distance[index];
        tile.witness[row][column] = witness[index];
    }
}

__device__ __forceinline__ void storeTile(const Tile& tile,
                                          unsigned int* distance,
                                          unsigned int* witness,
                                          int tileRow, int tileColumn,
                                          size_t stride) {
    const int column = threadIdx.x;
    #pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; ++r) {
        const int row = threadIdx.y + r * THREAD_ROWS;
        const size_t index = (size_t(tileRow) * TILE_SIZE + row) * stride +
                            size_t(tileColumn) * TILE_SIZE + column;
        distance[index] = tile.distance[row][column];
        witness[index] = tile.witness[row][column];
    }
}

__global__ void initializeDeviceMatrix(unsigned int* distance,
                                        unsigned int* witness,
                                        size_t stride, size_t numNodes) {
    const size_t count = stride * stride;
    for (size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += size_t(gridDim.x) * blockDim.x) {
        witness[index] = 0;
        const size_t row = index / stride;
        const size_t column = index % stride;
        if (row >= numNodes || column >= numNodes) {
            distance[index] = row == column ? 0 : INF;
        }
    }
}

// Phase 1: close the diagonal tile. Pivot rows/columns are read-only during
// each iteration (zero diagonal), so one barrier per k is sufficient.
__global__ void closePivot(unsigned int* distance, unsigned int* witness,
                           size_t stride, int pivot) {
    __shared__ Tile tile;
    loadTile(tile, distance, witness, pivot, pivot, stride);
    __syncthreads();
    const int column = threadIdx.x;
    for (int k = 0; k < TILE_SIZE; ++k) {
        #pragma unroll
        for (int r = 0; r < ROWS_PER_THREAD; ++r) {
            const int row = threadIdx.y + r * THREAD_ROWS;
            if (row != k && column != k) {
                relax(tile.distance[row][column], tile.witness[row][column],
                      tile.distance[row][k], tile.distance[k][column],
                      tile.witness[row][k], tile.witness[k][column],
                      pivot * TILE_SIZE + k + 1);
            }
        }
        __syncthreads();
    }
    storeTile(tile, distance, witness, pivot, pivot, stride);
}

// Phase 2: all tiles in the pivot row and column run independently.
template <bool PivotRow>
__device__ __forceinline__ void closePanel(unsigned int* distance,
                                          unsigned int* witness,
                                          size_t stride, int pivot,
                                          Tile& diagonal, Tile& panel) {
    const int other = int(blockIdx.x) + (int(blockIdx.x) >= pivot);
    const int tileRow = PivotRow ? pivot : other;
    const int tileColumn = PivotRow ? other : pivot;
    loadTile(diagonal, distance, witness, pivot, pivot, stride);
    loadTile(panel, distance, witness, tileRow, tileColumn, stride);
    __syncthreads();
    const int column = threadIdx.x;
    for (int k = 0; k < TILE_SIZE; ++k) {
        #pragma unroll
        for (int r = 0; r < ROWS_PER_THREAD; ++r) {
            const int row = threadIdx.y + r * THREAD_ROWS;
            if constexpr (PivotRow) {
                if (row != k) {
                    relax(panel.distance[row][column], panel.witness[row][column],
                          diagonal.distance[row][k], panel.distance[k][column],
                          diagonal.witness[row][k], panel.witness[k][column],
                          pivot * TILE_SIZE + k + 1);
                }
            } else {
                if (column != k) {
                    relax(panel.distance[row][column], panel.witness[row][column],
                          panel.distance[row][k], diagonal.distance[k][column],
                          panel.witness[row][k], diagonal.witness[k][column],
                          pivot * TILE_SIZE + k + 1);
                }
            }
        }
        __syncthreads();
    }
    storeTile(panel, distance, witness, tileRow, tileColumn, stride);
}

__global__ void closePanels(unsigned int* distance, unsigned int* witness,
                            size_t stride, int pivot) {
    __shared__ Tile diagonal;
    __shared__ Tile panel;
    // The branch is uniform within each block. Both independent panel sets
    // share a launch, exposing more parallelism and reducing launch overhead.
    if (blockIdx.y == 0) {
        closePanel<true>(distance, witness, stride, pivot, diagonal, panel);
    } else {
        closePanel<false>(distance, witness, stride, pivot, diagonal, panel);
    }
}

// Phase 3 dominates the work: load each panel once, retain four output cells
// per thread in registers, and reuse the panels without inner-loop barriers.
__global__ void updateRemaining(unsigned int* __restrict__ distance,
                                unsigned int* __restrict__ witness,
                                size_t stride, int pivot) {
    __shared__ Tile left;
    __shared__ Tile right;
    const int tileRow = int(blockIdx.y) + (int(blockIdx.y) >= pivot);
    const int tileColumn = int(blockIdx.x) + (int(blockIdx.x) >= pivot);
    loadTile(left, distance, witness, tileRow, pivot, stride);
    loadTile(right, distance, witness, pivot, tileColumn, stride);
    const int column = threadIdx.x;
    unsigned int outputDistance[ROWS_PER_THREAD];
    unsigned int outputWitness[ROWS_PER_THREAD];
    #pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; ++r) {
        const int row = threadIdx.y + r * THREAD_ROWS;
        const size_t index = (size_t(tileRow) * TILE_SIZE + row) * stride +
                            size_t(tileColumn) * TILE_SIZE + column;
        outputDistance[r] = distance[index];
        outputWitness[r] = witness[index];
    }
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < TILE_SIZE; ++k) {
        const unsigned int rightDistance = right.distance[k][column];
        const unsigned int rightWitness = right.witness[k][column];
        #pragma unroll
        for (int r = 0; r < ROWS_PER_THREAD; ++r) {
            const int row = threadIdx.y + r * THREAD_ROWS;
            relax(outputDistance[r], outputWitness[r],
                  left.distance[row][k], rightDistance,
                  left.witness[row][k], rightWitness,
                  pivot * TILE_SIZE + k + 1);
        }
    }
    #pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; ++r) {
        const int row = threadIdx.y + r * THREAD_ROWS;
        const size_t index = (size_t(tileRow) * TILE_SIZE + row) * stride +
                            size_t(tileColumn) * TILE_SIZE + column;
        distance[index] = outputDistance[r];
        witness[index] = outputWitness[r];
    }
}

__global__ void decodePaths(unsigned int* witness, size_t stride) {
    const size_t count = stride * stride;
    for (size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += size_t(gridDim.x) * blockDim.x) {
        const unsigned int value = witness[index];
        // initializePathMatrix initializes each entry to its source row.
        witness[index] = value ? value - 1 : unsigned(index / stride);
    }
}

} // namespace

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
        std::fill_n(path.data() + row * numNodes, numNodes,
                    static_cast<unsigned int>(row));
    }
}

void floydWarshall(std::vector<unsigned int>& dist, 
                   std::vector<unsigned int>& path, 
                   const size_t numNodes) {
    if (numNodes == 0) {
        return;
    }
    const size_t stride = (numNodes + TILE_SIZE - 1) / TILE_SIZE * TILE_SIZE;
    const size_t matrixBytes = stride * stride * sizeof(unsigned int);
    const int tiles = int(stride / TILE_SIZE);
    unsigned int* deviceDistance = nullptr;
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&deviceDistance), 2 * matrixBytes),
              "allocating GPU matrices");
    unsigned int* deviceWitness = deviceDistance + stride * stride;
    cudaStream_t stream;
    checkCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
              "creating CUDA stream");
    const unsigned int linearBlocks = unsigned(std::min<size_t>(
        (stride * stride + 255) / 256, 65535));
    initializeDeviceMatrix<<<linearBlocks, 256, 0, stream>>>(
        deviceDistance, deviceWitness, stride, numNodes);
    checkCuda(cudaGetLastError(), "initializing GPU padding and witnesses");
    checkCuda(cudaMemcpy2DAsync(deviceDistance, stride * sizeof(unsigned int),
                               dist.data(), numNodes * sizeof(unsigned int),
                               numNodes * sizeof(unsigned int), numNodes,
                               cudaMemcpyHostToDevice, stream),
              "copying distances to GPU");

    const dim3 threads(TILE_SIZE, THREAD_ROWS);
    for (int pivot = 0; pivot < tiles; ++pivot) {
        closePivot<<<1, threads, 0, stream>>>(deviceDistance, deviceWitness,
                                             stride, pivot);
        checkCuda(cudaGetLastError(), "launching diagonal tile");
        if (tiles > 1) {
            closePanels<<<dim3(tiles - 1, 2), threads, 0, stream>>>(
                deviceDistance, deviceWitness, stride, pivot);
            checkCuda(cudaGetLastError(), "launching pivot row and column");
            updateRemaining<<<dim3(tiles - 1, tiles - 1), threads, 0, stream>>>(
                deviceDistance, deviceWitness, stride, pivot);
            checkCuda(cudaGetLastError(), "launching remaining tiles");
        }
    }
    decodePaths<<<linearBlocks, 256, 0, stream>>>(deviceWitness, stride);
    checkCuda(cudaGetLastError(), "decoding path witnesses");
    checkCuda(cudaMemcpy2DAsync(dist.data(), numNodes * sizeof(unsigned int),
                               deviceDistance, stride * sizeof(unsigned int),
                               numNodes * sizeof(unsigned int), numNodes,
                               cudaMemcpyDeviceToHost, stream),
              "copying distances from GPU");
    checkCuda(cudaMemcpy2DAsync(path.data(), numNodes * sizeof(unsigned int),
                               deviceWitness, stride * sizeof(unsigned int),
                               numNodes * sizeof(unsigned int), numNodes,
                               cudaMemcpyDeviceToHost, stream),
              "copying paths from GPU");
    checkCuda(cudaStreamSynchronize(stream), "waiting for Floyd-Warshall");
    checkCuda(cudaStreamDestroy(stream), "destroying CUDA stream");
    checkCuda(cudaFree(deviceDistance), "freeing GPU matrices");
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
            const char* argument = argv[++i];
            char* end = nullptr;
            errno = 0;
            const unsigned long long parsed = strtoull(argument, &end, 10);
            if (argument[0] == '-' || end == argument || *end != '\0' ||
                errno == ERANGE ||
                parsed > static_cast<unsigned long long>(
                    std::numeric_limits<int>::max() - TILE_SIZE)) {
                fprintf(stderr, "Invalid number of nodes: %s\n", argument);
                return 1;
            }
            numNodes = static_cast<size_t>(parsed);
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
    
    const size_t paddedNodes = (numNodes + TILE_SIZE - 1) / TILE_SIZE * TILE_SIZE;
    if (paddedNodes != 0 && paddedNodes >
        std::numeric_limits<size_t>::max() / (2 * sizeof(unsigned int)) / paddedNodes) {
        fprintf(stderr, "The requested matrices are too large\n");
        return 1;
    }

    // Initialize the CUDA runtime outside the computation timer. CUDA is
    // required, including for empty graphs; there is no CPU fallback.
    checkCuda(cudaFree(nullptr), "initializing CUDA");

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
    const double seconds = std::chrono::duration<double>(end - start).count();
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
