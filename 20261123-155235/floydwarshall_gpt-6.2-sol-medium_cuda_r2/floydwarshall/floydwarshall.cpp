#include <algorithm>
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

namespace {

constexpr int tileSize = 32;
constexpr int blockRows = 8;
constexpr int rowsPerThread = tileSize / blockRows;
constexpr int maxTiles = 65535; // CUDA grid y dimension limit
constexpr int deferredLabelThreshold = 6144;
constexpr unsigned int pivotBits = 23;
constexpr unsigned int pivotMask = (1u << pivotBits) - 1;
constexpr unsigned int distanceMask = ~pivotMask;
constexpr unsigned int padding = 255u << pivotBits;
static_assert(2u * tileSize * maxTiles <= pivotMask);
static_assert(MAX_DISTANCE + 1 < (padding >> pivotBits));
static_assert(2ull * padding <= ~0u);

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

// The low bits store one plus the largest internal vertex on a shortest
// path, minimized across ties. A direct edge has label zero. This ordering
// gives the pivot of the original loop's last strict distance improvement.
__device__ __forceinline__ unsigned int throughPivot(unsigned int left, unsigned int right,
                                                     unsigned int pivotLabel) {
    const unsigned int distance = (left + right) & distanceMask;
    const unsigned int label = max(pivotLabel, max(left & pivotMask, right & pivotMask));
    return distance | label;
}

__device__ __forceinline__ size_t tileOffset(int row, int col, int stride) {
    return static_cast<size_t>(row) * tileSize * stride + col * tileSize;
}

__global__ void packDistances(unsigned int* matrix, int numNodes, int stride) {
    const int col = blockIdx.x * tileSize + threadIdx.x;
    const int firstRow = blockIdx.y * tileSize + threadIdx.y;
    #pragma unroll
    for (int i = 0; i < rowsPerThread; ++i) {
        const int row = firstRow + i * blockRows;
        const size_t offset = static_cast<size_t>(row) * stride + col;
        matrix[offset] = row < numNodes && col < numNodes ? matrix[offset] << pivotBits : padding;
    }
}

__global__ void updatePivotTile(unsigned int* matrix, int stride, int pivotTile) {
    __shared__ unsigned int tile[tileSize][tileSize];
    const int col = threadIdx.x;
    const int firstRow = threadIdx.y;
    unsigned int* data = matrix + tileOffset(pivotTile, pivotTile, stride);

    #pragma unroll
    for (int i = 0; i < rowsPerThread; ++i) {
        const int row = firstRow + i * blockRows;
        tile[row][col] = data[static_cast<size_t>(row) * stride + col];
    }
    __syncthreads();

    for (int k = 0; k < tileSize; ++k) {
        // A zero diagonal leaves the source row and column unchanged.
        // Keeping them read-only within this step avoids shared memory races.
        const unsigned int right = tile[k][col];
        #pragma unroll
        for (int i = 0; i < rowsPerThread; ++i) {
            const int row = firstRow + i * blockRows;
            if (row != k && col != k) {
                const unsigned int candidate = throughPivot(tile[row][k], right,
                                                            pivotTile * tileSize + k + 1);
                tile[row][col] = min(tile[row][col], candidate);
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < rowsPerThread; ++i) {
        const int row = firstRow + i * blockRows;
        data[static_cast<size_t>(row) * stride + col] = tile[row][col];
    }
}

template <bool pivotEdges, bool deferLabels = false>
__global__ void updateTiles(unsigned int* matrix, int stride, int pivotTile) {
    const int col = threadIdx.x;
    const int firstRow = threadIdx.y;
    const int otherCol = blockIdx.x + (blockIdx.x >= pivotTile);
    const int otherRow = blockIdx.y + (blockIdx.y >= pivotTile);
    const bool pivotRow = pivotEdges && blockIdx.y == 0;
    const bool pivotCol = pivotEdges && !pivotRow;
    const int tileRow = pivotRow ? pivotTile : (pivotCol ? otherCol : otherRow);
    const int tileCol = pivotCol ? pivotTile : otherCol;
    const unsigned int* leftData = matrix + tileOffset(tileRow, pivotTile, stride);
    const unsigned int* rightData = matrix + tileOffset(pivotTile, tileCol, stride);
    unsigned int* data = matrix + tileOffset(tileRow, tileCol, stride);
    // Warps broadcast a left value and read consecutive right values.
    __shared__ unsigned int leftTile[tileSize][tileSize];
    __shared__ unsigned int rightTile[tileSize][tileSize];
    unsigned int best[rowsPerThread];
    unsigned int original[rowsPerThread];

    #pragma unroll
    for (int i = 0; i < rowsPerThread; ++i) {
        const int row = firstRow + i * blockRows;
        const size_t index = static_cast<size_t>(row) * stride + col;
        const unsigned int left = leftData[index];
        const unsigned int right = rightData[index];
        leftTile[row][col] = left;
        rightTile[row][col] = right;
        best[i] = pivotRow ? right : (pivotCol ? left : data[index]);
        if constexpr (deferLabels) {
            original[i] = best[i];
        }
    }
    __syncthreads();

    // The pivot tile is already closed when updating its row and column.
    // Those completed tiles then supply both operands for all other tiles.
    #pragma unroll
    for (int k = 0; k < tileSize; ++k) {
        const unsigned int pivotLabel = pivotTile * tileSize + k + 1;
        const unsigned int right = rightTile[k][col];
        #pragma unroll
        for (int i = 0; i < rowsPerThread; ++i) {
            const int row = firstRow + i * blockRows;
            const unsigned int left = leftTile[row][k];
            if constexpr (deferLabels) {
                // Summed labels fit below the distance bits.
                best[i] = min(best[i], left + right);
            } else {
                best[i] = min(best[i], throughPivot(left, right, pivotLabel));
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < rowsPerThread; ++i) {
        const int row = firstRow + i * blockRows;
        if constexpr (deferLabels) {
            const unsigned int distance = best[i] & distanceMask;
            best[i] = original[i];
            // Most entries in large graphs already have their final
            // distance. Revisit the pivots only after a strict improvement.
            if (distance < (best[i] & distanceMask)) {
                best[i] = distance | pivotMask;
                #pragma unroll 4
                for (int k = 0; k < tileSize; ++k) {
                    const unsigned int left = leftTile[row][k];
                    const unsigned int right = rightTile[k][col];
                    if (((left + right) & distanceMask) == distance) {
                        best[i] = min(best[i], throughPivot(left, right, pivotTile * tileSize + k + 1));
                    }
                }
            }
        }
        data[static_cast<size_t>(row) * stride + col] = best[i];
    }
}

} // namespace

float floydWarshall(std::vector<unsigned int>& dist, std::vector<unsigned int>& path,
                    const size_t numNodes) {
    if (numNodes == 0) {
        return 0.0f;
    }

    const int tileCount = (numNodes + tileSize - 1) / tileSize;
    const int stride = tileCount * tileSize;
    const size_t devicePitch = static_cast<size_t>(stride) * sizeof(unsigned int);
    const size_t hostPitch = numNodes * sizeof(unsigned int);
    const dim3 threads(tileSize, blockRows);
    unsigned int* deviceMatrix = nullptr;
    checkCuda(cudaMalloc(&deviceMatrix, stride * devicePitch), "Allocating CUDA distance matrix");
    checkCuda(cudaMemcpy2D(deviceMatrix, devicePitch, dist.data(), hostPitch, hostPitch,
                           numNodes, cudaMemcpyHostToDevice), "Uploading distance matrix");
    packDistances<<<dim3(tileCount, tileCount), threads>>>(deviceMatrix, numNodes, stride);
    checkCuda(cudaGetLastError(), "Launching distance packing kernel");

    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start), "Creating CUDA start event");
    checkCuda(cudaEventCreate(&stop), "Creating CUDA stop event");
    checkCuda(cudaEventRecord(start), "Recording CUDA start event");
    for (int pivot = 0; pivot < tileCount; ++pivot) {
        updatePivotTile<<<1, threads>>>(deviceMatrix, stride, pivot);
        if (tileCount > 1) {
            updateTiles<true><<<dim3(tileCount - 1, 2), threads>>>(deviceMatrix, stride, pivot);
            const dim3 remainingTiles(tileCount - 1, tileCount - 1);
            if (numNodes >= deferredLabelThreshold) {
                updateTiles<false, true><<<remainingTiles, threads>>>(deviceMatrix, stride, pivot);
            } else {
                updateTiles<false><<<remainingTiles, threads>>>(deviceMatrix, stride, pivot);
            }
        }
    }
    checkCuda(cudaGetLastError(), "Launching Floyd-Warshall kernels");
    checkCuda(cudaEventRecord(stop), "Recording CUDA stop event");
    checkCuda(cudaEventSynchronize(stop), "Executing Floyd-Warshall kernels");
    float elapsedMs;
    checkCuda(cudaEventElapsedTime(&elapsedMs, start, stop), "Reading CUDA elapsed time");
    checkCuda(cudaEventDestroy(start), "Destroying CUDA start event");
    checkCuda(cudaEventDestroy(stop), "Destroying CUDA stop event");

    checkCuda(cudaMemcpy2D(dist.data(), hostPitch, deviceMatrix, devicePitch, hostPitch,
                           numNodes, cudaMemcpyDeviceToHost), "Downloading distance matrix");
    checkCuda(cudaFree(deviceMatrix), "Freeing CUDA distance matrix");
    for (size_t row = 0; row < numNodes; ++row) {
        for (size_t col = 0; col < numNodes; ++col) {
            const size_t index = idx2(col, row, numNodes);
            const unsigned int label = dist[index] & pivotMask;
            path[index] = label == 0 ? row : label - 1;
            dist[index] >>= pivotBits;
        }
    }
    return elapsedMs;
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
    
    if (numNodes > static_cast<size_t>(tileSize) * maxTiles) {
        fprintf(stderr, "Number of nodes exceeds the CUDA grid limit\n");
        return 1;
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
    
    // Run Floyd-Warshall
    printf("Computing shortest paths...\n");
    const double elapsedMs = floydWarshall(dist, path, numNodes);
    printf("Computation time: %.3f ms\n", elapsedMs);
    
    // Calculate operations per second
    // Floyd-Warshall has O(n³) complexity
    double ops = (double)numNodes * numNodes * numNodes;
    double gops = elapsedMs > 0.0 ? ops / (elapsedMs * 1e6) : 0.0;
    printf("Performance: %.3f GOPS\n", gops);
    
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
