#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
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
        std::fill_n(path.data() + j * numNodes, numNodes, static_cast<unsigned int>(j));
    }
}

namespace {

constexpr unsigned int tileSize = 32;
constexpr unsigned int pivotBlockRows = 8;
constexpr unsigned int remainingBlockRows = 4;
constexpr unsigned int paddingThreads = 256;
// The largest possible generated edge is MAX_DISTANCE + 1, so a padded
// node can never offer a shorter route between real nodes.
constexpr unsigned int paddingDistance = MAX_DISTANCE + 2;
constexpr unsigned int pivotBits = 6;
constexpr unsigned int pivotMask = (1u << pivotBits) - 1;
static_assert(tileSize <= pivotMask && paddingDistance <= (~0u >> (pivotBits + 1)));

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

__global__ void initializePadding(unsigned int* dist, size_t numNodes, size_t stride) {
    const size_t padding = stride - numNodes;
    const size_t item = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t bottom = padding * stride;
    if (item < bottom + numNodes * padding) {
        const size_t index = item < bottom
            ? numNodes * stride + item
            : (item - bottom) / padding * stride + numNodes + (item - bottom) % padding;
        dist[index] = paddingDistance;
    }
}

__device__ __forceinline__ void relax(unsigned int candidate, unsigned int pivot,
                                      unsigned int& distance, unsigned int& path) {
    if (candidate < distance) {
        distance = candidate;
        path = pivot;
    }
}

// The zero diagonal keeps row k and column k unchanged during step k. Save
// these operands at each step so the other tiles see the same values as the
// serial recurrence and record the same intermediate node on strict updates.
template <bool PivotTile>
__global__ void updatePivotTiles(unsigned int* __restrict__ dist,
                                 unsigned int* __restrict__ path,
                                 unsigned int* __restrict__ toPivot,
                                 unsigned int* __restrict__ fromPivot,
                                 size_t stride, unsigned int pivot) {
    __shared__ unsigned int tile[tileSize][tileSize + 1];
    __shared__ unsigned int pivotOperands[PivotTile ? 1 : tileSize][tileSize];
    const unsigned int x = threadIdx.x;
    const unsigned int y = threadIdx.y;
    const bool pivotColumn = !PivotTile && blockIdx.y == 0;
    const unsigned int other = blockIdx.x + (blockIdx.x >= pivot);
    const unsigned int base = pivot * tileSize;
    const unsigned int rowBase = PivotTile || !pivotColumn ? base : other * tileSize;
    const unsigned int colBase = PivotTile || pivotColumn ? base : other * tileSize;
    constexpr unsigned int rowsPerThread = tileSize / pivotBlockRows;
    unsigned int updates[rowsPerThread] = {};

    #pragma unroll
    for (unsigned int r = 0; r < rowsPerThread; ++r) {
        const unsigned int row = y + r * pivotBlockRows;
        const size_t index = (static_cast<size_t>(rowBase) + row) * stride + colBase + x;
        tile[row][x] = dist[index];
        if constexpr (!PivotTile) {
            const unsigned int* operands = pivotColumn ? fromPivot : toPivot;
            pivotOperands[row][x] = operands[row * stride + base + x];
        }
    }
    __syncthreads();

    for (unsigned int k = 0; k < tileSize; ++k) {
        const size_t history = k * stride;
        if (y == 0) {
            if (PivotTile || pivotColumn) {
                toPivot[history + rowBase + x] = tile[x][k];
            }
            if (PivotTile || !pivotColumn) {
                fromPivot[history + colBase + x] = tile[k][x];
            }
        }

        const unsigned int right = PivotTile || !pivotColumn
            ? tile[k][x] : pivotOperands[k][x];
        #pragma unroll
        for (unsigned int r = 0; r < rowsPerThread; ++r) {
            const unsigned int row = y + r * pivotBlockRows;
            // These entries also serve as shared operands for this step.
            if (((!PivotTile && !pivotColumn) || x != k) &&
                ((!PivotTile && pivotColumn) || row != k)) {
                const unsigned int left = PivotTile || pivotColumn
                    ? tile[row][k] : pivotOperands[k][row];
                unsigned int distance = tile[row][x];
                relax(left + right, k + 1, distance, updates[r]);
                tile[row][x] = distance;
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (unsigned int r = 0; r < rowsPerThread; ++r) {
        if (updates[r] != 0) {
            const unsigned int row = y + r * pivotBlockRows;
            const size_t index = (static_cast<size_t>(rowBase) + row) * stride + colBase + x;
            dist[index] = tile[row][x];
            path[index] = base + updates[r] - 1;
        }
    }
}

// One block updates two adjacent non-pivot tiles, reusing the left operands.
// A warp accesses contiguous columns and each thread keeps its results in registers.
__global__ void updateRemainingTiles(unsigned int* __restrict__ dist,
                                    unsigned int* __restrict__ path,
                                    const unsigned int* __restrict__ toPivot,
                                    const unsigned int* __restrict__ fromPivot,
                                    size_t stride, unsigned int pivot,
                                    unsigned int numTiles) {
    __shared__ unsigned int left[tileSize][tileSize];
    __shared__ unsigned int right[tileSize][2 * tileSize];
    const unsigned int x = threadIdx.x;
    const unsigned int y = threadIdx.y;
    const unsigned int rowBase = (blockIdx.y + (blockIdx.y >= pivot)) * tileSize;
    const unsigned int colTile = 2 * blockIdx.x;
    const unsigned int colBase = (colTile + (colTile >= pivot)) * tileSize;
    const bool secondTile = colTile + 1 < numTiles - 1;
    const unsigned int nextBase = (colTile + 1 + (colTile + 1 >= pivot)) * tileSize;
    constexpr unsigned int rowsPerThread = tileSize / remainingBlockRows;
    unsigned int best[rowsPerThread][2];

    #pragma unroll
    for (unsigned int r = 0; r < rowsPerThread; ++r) {
        const unsigned int row = y + r * remainingBlockRows;
        const size_t history = row * stride;
        // Encode (distance, pivot order) so min preserves the first strict
        // improvement to the winning distance. The input distance wins ties.
        const unsigned int order = row + 1;
        left[row][x] = toPivot[history + rowBase + x] << pivotBits;
        right[row][x] = (fromPivot[history + colBase + x] << pivotBits) | order;
        const unsigned int second = secondTile ? fromPivot[history + nextBase + x]
                                               : paddingDistance;
        right[row][tileSize + x] = (second << pivotBits) | order;

        const size_t offset = (static_cast<size_t>(rowBase) + row) * stride + x;
        best[r][0] = dist[offset + colBase] << pivotBits;
        best[r][1] = (secondTile ? dist[offset + nextBase] : paddingDistance) << pivotBits;
    }
    __syncthreads();

    #pragma unroll 4
    for (unsigned int k = 0; k < tileSize; ++k) {
        const unsigned int first = right[k][x];
        const unsigned int second = right[k][tileSize + x];
        #pragma unroll
        for (unsigned int r = 0; r < rowsPerThread; ++r) {
            const unsigned int distance = left[k][y + r * remainingBlockRows];
            best[r][0] = min(best[r][0], distance + first);
            best[r][1] = min(best[r][1], distance + second);
        }
    }

    #pragma unroll
    for (unsigned int r = 0; r < rowsPerThread; ++r) {
        const size_t offset = (static_cast<size_t>(rowBase) + y + r * remainingBlockRows) * stride + x;
        const unsigned int first = best[r][0] & pivotMask;
        const unsigned int second = best[r][1] & pivotMask;
        if (first != 0) {
            dist[offset + colBase] = best[r][0] >> pivotBits;
            path[offset + colBase] = pivot * tileSize + first - 1;
        }
        if (secondTile && second != 0) {
            dist[offset + nextBase] = best[r][1] >> pivotBits;
            path[offset + nextBase] = pivot * tileSize + second - 1;
        }
    }
}

} // namespace

std::chrono::duration<double> floydWarshall(std::vector<unsigned int>& dist,
                                          std::vector<unsigned int>& path,
                                          const size_t numNodes) {
    if (numNodes == 0) {
        return {};
    }

    const unsigned int numTiles = (numNodes + tileSize - 1) / tileSize;
    const size_t stride = static_cast<size_t>(numTiles) * tileSize;
    const size_t matrixSize = stride * stride;
    const size_t historySize = stride * tileSize;
    const size_t hostPitch = numNodes * sizeof(unsigned int);
    const size_t devicePitch = stride * sizeof(unsigned int);

    unsigned int* allocation = nullptr;
    checkCuda(cudaMalloc(&allocation, 2 * (matrixSize + historySize) * sizeof(unsigned int)),
              "Allocating CUDA matrices");
    const std::unique_ptr<unsigned int, decltype(&cudaFree)> memory(allocation, cudaFree);
    unsigned int* deviceDist = memory.get();
    unsigned int* devicePath = deviceDist + matrixSize;
    unsigned int* toPivot = devicePath + matrixSize;
    unsigned int* fromPivot = toPivot + historySize;
    checkCuda(cudaMemcpy2D(deviceDist, devicePitch, dist.data(), hostPitch,
                           hostPitch, numNodes, cudaMemcpyHostToDevice), "Uploading distances");
    checkCuda(cudaMemcpy2D(devicePath, devicePitch, path.data(), hostPitch,
                           hostPitch, numNodes, cudaMemcpyHostToDevice), "Uploading paths");
    if (stride != numNodes) {
        const size_t padding = (stride - numNodes) * (stride + numNodes);
        initializePadding<<<(padding + paddingThreads - 1) / paddingThreads, paddingThreads>>>(
            deviceDist, numNodes, stride);
        checkCuda(cudaGetLastError(), "Launching CUDA padding initialization");
    }
    checkCuda(cudaDeviceSynchronize(), "Preparing CUDA matrices");

    const dim3 pivotThreads(tileSize, pivotBlockRows);
    const dim3 remainingThreads(tileSize, remainingBlockRows);
    const auto start = std::chrono::steady_clock::now();
    // Stream order enforces the dependencies from the pivot tile to its
    // bordering tiles and then to the rest of the matrix in every round.
    for (unsigned int pivot = 0; pivot < numTiles; ++pivot) {
        updatePivotTiles<true><<<1, pivotThreads>>>(deviceDist, devicePath, toPivot, fromPivot,
                                                    stride, pivot);
        if (numTiles > 1) {
            updatePivotTiles<false><<<dim3(numTiles - 1, 2), pivotThreads>>>(
                deviceDist, devicePath, toPivot, fromPivot, stride, pivot);
            updateRemainingTiles<<<dim3(numTiles / 2, numTiles - 1), remainingThreads>>>(
                deviceDist, devicePath, toPivot, fromPivot, stride, pivot, numTiles);
        }
    }
    checkCuda(cudaGetLastError(), "Launching Floyd-Warshall CUDA kernels");
    checkCuda(cudaDeviceSynchronize(), "Running Floyd-Warshall CUDA kernels");
    const auto elapsed = std::chrono::steady_clock::now() - start;

    checkCuda(cudaMemcpy2D(dist.data(), hostPitch, deviceDist, devicePitch,
                           hostPitch, numNodes, cudaMemcpyDeviceToHost), "Downloading distances");
    checkCuda(cudaMemcpy2D(path.data(), hostPitch, devicePath, devicePitch,
                           hostPitch, numNodes, cudaMemcpyDeviceToHost), "Downloading paths");
    return elapsed;
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
    std::chrono::duration<double> elapsed;
    try {
        elapsed = floydWarshall(dist, path, numNodes);
    } catch (const std::runtime_error& error) {
        fprintf(stderr, "%s\n", error.what());
        return 1;
    }
    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate operations per second
    // Floyd-Warshall has O(n³) complexity
    double ops = (double)numNodes * numNodes * numNodes;
    double gflops = elapsed.count() > 0.0 ? ops / elapsed.count() / 1e9 : 0.0;
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
