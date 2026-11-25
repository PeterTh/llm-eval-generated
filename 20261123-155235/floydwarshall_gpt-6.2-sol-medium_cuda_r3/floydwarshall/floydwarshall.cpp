#include <algorithm>
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

namespace {

constexpr int TILE_SIZE = 32;
constexpr int THREAD_ROWS = 4;
constexpr int ROWS_PER_THREAD = TILE_SIZE / THREAD_ROWS;
constexpr int MAX_NODES = TILE_SIZE * 65535;  // CUDA grid dimension limit along y.

// In the serial loop, the final distance is first reached at the smallest
// possible largest intermediate vertex of any shortest path. Pack that vertex
// plus one below the distance. For a direct edge the low bits are zero, which
// keeps the original path entry (the destination) on a tie.
constexpr int PATH_BITS = 23;
constexpr unsigned int PATH_MASK = (1u << PATH_BITS) - 1;
constexpr unsigned int DISTANCE_MASK = ~PATH_MASK;
constexpr unsigned int PADDING_DISTANCE = 255;
static_assert(MAX_DISTANCE + 1 < PADDING_DISTANCE);
static_assert(2 * PADDING_DISTANCE < (1u << (32 - PATH_BITS)));
static_assert(2u * MAX_NODES <= PATH_MASK);

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

__global__ void packMatrix(unsigned int* matrix, int nodes, int stride) {
    const int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    const int firstRow = blockIdx.y * TILE_SIZE + threadIdx.y;
    #pragma unroll
    for (int offset = 0; offset < TILE_SIZE; offset += THREAD_ROWS) {
        const int row = firstRow + offset;
        const size_t index = static_cast<size_t>(row) * stride + col;
        const unsigned int distance = row < nodes && col < nodes
                                      ? matrix[index] : PADDING_DISTANCE;
        matrix[index] = distance << PATH_BITS;
    }
}

__device__ __forceinline__ unsigned int relax(unsigned int current,
                                              unsigned int left,
                                              unsigned int right,
                                              unsigned int pivotPath) {
    // The sum of two path fields fits below the distance and cannot carry into it.
    const unsigned int distance = (left + right) & DISTANCE_MASK;
    const unsigned int path = max(max(left & PATH_MASK, right & PATH_MASK), pivotPath);
    return min(current, distance | path);
}

__global__ void closePivotTile(unsigned int* matrix, int stride, int pivot) {
    __shared__ unsigned int tile[TILE_SIZE][TILE_SIZE];
    const int col = threadIdx.x;
    const int firstRow = threadIdx.y;
    const int start = pivot * TILE_SIZE;
    unsigned int* base = matrix + static_cast<size_t>(start + firstRow) * stride
                         + start + col;

    #pragma unroll
    for (int row = 0; row < TILE_SIZE; row += THREAD_ROWS) {
        tile[firstRow + row][col] = base[static_cast<size_t>(row) * stride];
    }
    __syncthreads();

    for (int k = 0; k < TILE_SIZE; ++k) {
        // A zero diagonal leaves the current pivot row and column unchanged.
        // Skipping their writes lets other threads read them in the same step.
        if (col != k) {
            const unsigned int right = tile[k][col];
            #pragma unroll
            for (int row = 0; row < TILE_SIZE; row += THREAD_ROWS) {
                const int r = firstRow + row;
                if (r != k) {
                    tile[r][col] = relax(tile[r][col], tile[r][k], right, start + k + 1);
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int row = 0; row < TILE_SIZE; row += THREAD_ROWS) {
        base[static_cast<size_t>(row) * stride] = tile[firstRow + row][col];
    }
}

// Each block writes an independent tile. The pivot row and column use the
// closed diagonal tile. The remaining tiles use the completed row and column.
template <bool pivotSides>
__global__ void updateTiles(unsigned int* matrix, int stride, int pivot) {
    const int tileX = blockIdx.x + (blockIdx.x >= static_cast<unsigned int>(pivot));
    int tileRow = pivot;
    int tileCol = tileX;
    if constexpr (pivotSides) {
        if (blockIdx.y == 1) {
            tileRow = tileX;
            tileCol = pivot;
        }
    } else {
        tileRow = blockIdx.y + (blockIdx.y >= static_cast<unsigned int>(pivot));
    }

    __shared__ unsigned int left[TILE_SIZE][TILE_SIZE];
    __shared__ unsigned int right[TILE_SIZE][TILE_SIZE];
    const int col = threadIdx.x;
    const int firstRow = threadIdx.y;
    const int start = pivot * TILE_SIZE;
    const size_t rowOffset = static_cast<size_t>(tileRow * TILE_SIZE + firstRow) * stride;
    const size_t pivotOffset = static_cast<size_t>(start + firstRow) * stride;
    const int outCol = tileCol * TILE_SIZE + col;
    unsigned int best[ROWS_PER_THREAD];

    #pragma unroll
    for (int i = 0; i < ROWS_PER_THREAD; ++i) {
        const int row = i * THREAD_ROWS;
        const size_t offset = static_cast<size_t>(row) * stride;
        left[firstRow + row][col] = matrix[rowOffset + offset + start + col];
        right[firstRow + row][col] = matrix[pivotOffset + offset + outCol];
        best[i] = matrix[rowOffset + offset + outCol];
    }
    __syncthreads();

    #pragma unroll 16
    for (int k = 0; k < TILE_SIZE; ++k) {
        const unsigned int rightValue = right[k][col];
        #pragma unroll
        for (int i = 0; i < ROWS_PER_THREAD; ++i) {
            best[i] = relax(best[i], left[firstRow + i * THREAD_ROWS][k],
                            rightValue, start + k + 1);
        }
    }

    #pragma unroll
    for (int i = 0; i < ROWS_PER_THREAD; ++i) {
        const size_t offset = static_cast<size_t>(i * THREAD_ROWS) * stride;
        matrix[rowOffset + offset + outCol] = best[i];
    }
}

float floydWarshall(std::vector<unsigned int>& dist, int nodes, bool readResult) {
    if (nodes == 0) {
        return 0.0f;
    }

    const int tileCount = (nodes + TILE_SIZE - 1) / TILE_SIZE;
    const int stride = tileCount * TILE_SIZE;
    const size_t hostPitch = static_cast<size_t>(nodes) * sizeof(unsigned int);
    const size_t devicePitch = static_cast<size_t>(stride) * sizeof(unsigned int);
    unsigned int* matrix;
    checkCuda(cudaMalloc(&matrix, devicePitch * stride), "Allocating device matrix");
    checkCuda(cudaMemcpy2D(matrix, devicePitch, dist.data(), hostPitch,
                           hostPitch, nodes, cudaMemcpyHostToDevice), "Uploading distance matrix");

    cudaEvent_t start, end;
    checkCuda(cudaEventCreate(&start), "Creating CUDA start event");
    checkCuda(cudaEventCreate(&end), "Creating CUDA end event");
    const dim3 threads(TILE_SIZE, THREAD_ROWS);
    packMatrix<<<dim3(tileCount, tileCount), threads>>>(matrix, nodes, stride);
    checkCuda(cudaGetLastError(), "Launching matrix packing");
    checkCuda(cudaEventRecord(start), "Recording CUDA start event");

    const int otherTiles = tileCount - 1;
    for (int pivot = 0; pivot < tileCount; ++pivot) {
        closePivotTile<<<1, threads>>>(matrix, stride, pivot);
        if (otherTiles != 0) {
            updateTiles<true><<<dim3(otherTiles, 2), threads>>>(matrix, stride, pivot);
            updateTiles<false><<<dim3(otherTiles, otherTiles), threads>>>(matrix, stride, pivot);
        }
    }
    checkCuda(cudaGetLastError(), "Launching Floyd-Warshall kernels");
    checkCuda(cudaEventRecord(end), "Recording CUDA end event");
    checkCuda(cudaEventSynchronize(end), "Running Floyd-Warshall kernels");
    float milliseconds;
    checkCuda(cudaEventElapsedTime(&milliseconds, start, end), "Measuring CUDA execution");
    checkCuda(cudaEventDestroy(start), "Destroying CUDA start event");
    checkCuda(cudaEventDestroy(end), "Destroying CUDA end event");

    if (readResult) {
        checkCuda(cudaMemcpy2D(dist.data(), hostPitch, matrix, devicePitch,
                               hostPitch, nodes, cudaMemcpyDeviceToHost), "Downloading distance matrix");
        for (unsigned int& value : dist) {
            value >>= PATH_BITS;
        }
    }
    checkCuda(cudaFree(matrix), "Freeing device matrix");
    return milliseconds;
}

}  // namespace

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
            const long parsedNodes = std::strtol(argv[++i], nullptr, 10);
            if (parsedNodes < 0 || parsedNodes > MAX_NODES) {
                fprintf(stderr, "Number of nodes must be between 0 and %d\n", MAX_NODES);
                return 1;
            }
            numNodes = static_cast<size_t>(parsedNodes);
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
    
    // Run Floyd-Warshall
    printf("Computing shortest paths...\n");
    const double milliseconds = floydWarshall(dist, static_cast<int>(numNodes),
                                              validate || printResults);
    printf("Computation time: %.3f ms\n", milliseconds);
    
    // Calculate operations per second
    // Floyd-Warshall has O(n³) complexity
    double ops = (double)numNodes * numNodes * numNodes;
    const double gops = milliseconds > 0.0 ? ops / (milliseconds * 1e6) : 0.0;
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
