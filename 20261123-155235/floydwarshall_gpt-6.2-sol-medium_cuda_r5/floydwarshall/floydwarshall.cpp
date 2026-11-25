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
__host__ __device__ inline constexpr size_t idx2(const size_t i, const size_t j,
                                                const size_t n) noexcept {
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
        std::fill_n(path.data() + row * numNodes, numNodes, static_cast<unsigned int>(row));
    }
}

namespace {

constexpr int TILE = 32;
constexpr int BLOCK_ROWS = 4;
constexpr int DISTANCE_SHIFT = 23;
constexpr unsigned int PATH_MASK = (1u << (DISTANCE_SHIFT - 1)) - 1;
constexpr unsigned int DISTANCE_MASK = ~((1u << DISTANCE_SHIFT) - 1);
constexpr unsigned int PADDING_DISTANCE = 255;
static_assert(MAX_DISTANCE + 1 < PADDING_DISTANCE);
static_assert(2 * PADDING_DISTANCE < (1u << (32 - DISTANCE_SHIFT)));

// The low bits rank equal-distance paths by their largest intermediate vertex,
// plus one. The smallest rank is the first k that attains this distance in the
// scalar algorithm. Direct edges have rank zero. A spare bit absorbs the carry
// when adding the low bits of two packed values.
__device__ __forceinline__ unsigned int extendPath(unsigned int left, unsigned int right,
                                                   unsigned int rank) {
    return ((left + right) & DISTANCE_MASK) | rank;
}

__global__ void packDistances(unsigned int* matrix, unsigned int nodes, unsigned int stride) {
    const unsigned int col = blockIdx.x * TILE + threadIdx.x;
    const unsigned int firstRow = blockIdx.y * TILE + threadIdx.y;
    #pragma unroll
    for (int offset = 0; offset < TILE; offset += BLOCK_ROWS) {
        const unsigned int row = firstRow + offset;
        const size_t index = idx2(col, row, stride);
        const unsigned int distance = row < nodes && col < nodes
                                      ? matrix[index] : (row == col ? 0 : PADDING_DISTANCE);
        matrix[index] = distance << DISTANCE_SHIFT;
    }
}

__global__ void closePivot(unsigned int* matrix, unsigned int stride, unsigned int pivot) {
    __shared__ unsigned int tile[TILE][TILE];
    const unsigned int col = threadIdx.x;
    const unsigned int firstRow = threadIdx.y;
    const unsigned int base = pivot * TILE;

    #pragma unroll
    for (int offset = 0; offset < TILE; offset += BLOCK_ROWS) {
        const unsigned int row = firstRow + offset;
        tile[row][col] = matrix[idx2(base + col, base + row, stride)];
    }
    __syncthreads();

    for (unsigned int k = 0; k < TILE; ++k) {
        // With a zero diagonal these entries cannot improve at iteration k.
        // Keeping them read-only also lets threads update the other entries in place.
        if (col != k) {
            const unsigned int right = tile[k][col];
            #pragma unroll
            for (int offset = 0; offset < TILE; offset += BLOCK_ROWS) {
                const unsigned int row = firstRow + offset;
                if (row != k) {
                    const unsigned int candidate = extendPath(tile[row][k], right, base + k + 1);
                    tile[row][col] = min(tile[row][col], candidate);
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int offset = 0; offset < TILE; offset += BLOCK_ROWS) {
        const unsigned int row = firstRow + offset;
        matrix[idx2(base + col, base + row, stride)] = tile[row][col];
    }
}

// A pivot-row or pivot-column tile only needs one product with the closed
// pivot tile, which already contains all paths through the current block.
// The remaining tiles then use the updated pivot row and column as inputs.
template <bool panel>
__global__ void relaxTiles(unsigned int* matrix, unsigned int stride, unsigned int pivot) {
    __shared__ unsigned int left[TILE][TILE];
    __shared__ unsigned int right[TILE][TILE];
    const unsigned int col = threadIdx.x;
    const unsigned int firstRow = threadIdx.y;
    const unsigned int base = pivot * TILE;
    const unsigned int otherBase = (blockIdx.x + (blockIdx.x >= pivot)) * TILE;
    const unsigned int rowBase = panel ? (blockIdx.y == 0 ? base : otherBase)
                                       : (blockIdx.y + (blockIdx.y >= pivot)) * TILE;
    const unsigned int colBase = panel && blockIdx.y != 0 ? base : otherBase;
    unsigned int best[TILE / BLOCK_ROWS];

    #pragma unroll
    for (int offset = 0; offset < TILE; offset += BLOCK_ROWS) {
        const unsigned int row = firstRow + offset;
        const unsigned int leftValue = matrix[idx2(base + col, rowBase + row, stride)];
        const unsigned int rightValue = matrix[idx2(colBase + col, base + row, stride)];
        const unsigned int rank = max(leftValue & PATH_MASK, base + col + 1);
        left[row][col] = (leftValue & DISTANCE_MASK) | rank;
        right[row][col] = rightValue;
        if constexpr (panel) {
            best[offset / BLOCK_ROWS] = blockIdx.y == 0 ? rightValue : leftValue;
        } else {
            best[offset / BLOCK_ROWS] = matrix[idx2(colBase + col, rowBase + row, stride)];
        }
    }
    __syncthreads();

    #pragma unroll
    for (int k = 0; k < TILE; ++k) {
        const unsigned int rightValue = right[k][col];
        const unsigned int rightRank = rightValue & PATH_MASK;
        #pragma unroll
        for (int offset = 0; offset < TILE; offset += BLOCK_ROWS) {
            const unsigned int leftValue = left[firstRow + offset][k];
            const unsigned int rank = max(leftValue & PATH_MASK, rightRank);
            const unsigned int candidate = extendPath(leftValue, rightValue, rank);
            best[offset / BLOCK_ROWS] = min(best[offset / BLOCK_ROWS], candidate);
        }
    }

    #pragma unroll
    for (int offset = 0; offset < TILE; offset += BLOCK_ROWS) {
        matrix[idx2(colBase + col, rowBase + firstRow + offset, stride)] = best[offset / BLOCK_ROWS];
    }
}

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

}  // namespace

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes) {
    if (numNodes == 0) {
        return;
    }
    if (numNodes > PATH_MASK) {
        fprintf(stderr, "CUDA Floyd-Warshall supports at most %u nodes\n", PATH_MASK);
        std::exit(EXIT_FAILURE);
    }

    const unsigned int tiles = (numNodes + TILE - 1) / TILE;
    const unsigned int stride = tiles * TILE;
    const size_t hostPitch = numNodes * sizeof(unsigned int);
    const size_t devicePitch = stride * sizeof(unsigned int);
    unsigned int* matrix = nullptr;
    checkCuda(cudaMalloc(&matrix, devicePitch * stride), "Allocate CUDA distance matrix");
    checkCuda(cudaMemcpy2D(matrix, devicePitch, dist.data(), hostPitch, hostPitch, numNodes,
                           cudaMemcpyHostToDevice), "Upload distance matrix");

    const dim3 threads(TILE, BLOCK_ROWS);
    packDistances<<<dim3(tiles, tiles), threads>>>(matrix, numNodes, stride);
    checkCuda(cudaGetLastError(), "Launch distance packing");
    for (unsigned int pivot = 0; pivot < tiles; ++pivot) {
        closePivot<<<1, threads>>>(matrix, stride, pivot);
        if (tiles > 1) {
            relaxTiles<true><<<dim3(tiles - 1, 2), threads>>>(matrix, stride, pivot);
            relaxTiles<false><<<dim3(tiles - 1, tiles - 1), threads>>>(matrix, stride, pivot);
        }
    }
    checkCuda(cudaGetLastError(), "Launch Floyd-Warshall kernels");
    checkCuda(cudaMemcpy2D(dist.data(), hostPitch, matrix, devicePitch, hostPitch, numNodes,
                           cudaMemcpyDeviceToHost), "Download shortest paths");
    checkCuda(cudaFree(matrix), "Free CUDA distance matrix");

    for (size_t i = 0; i < dist.size(); ++i) {
        const unsigned int rank = dist[i] & PATH_MASK;
        path[i] = rank == 0 ? path[i] : rank - 1;
        dist[i] >>= DISTANCE_SHIFT;
    }
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
    if (numNodes != 0) {
        checkCuda(cudaFree(nullptr), "Initialize CUDA");
    }
    
    // Run Floyd-Warshall
    printf("Computing shortest paths...\n");
    auto start = std::chrono::steady_clock::now();
    
    floydWarshall(dist, path, numNodes);
    
    const auto elapsed = std::chrono::steady_clock::now() - start;
    const double milliseconds = std::chrono::duration<double, std::milli>(elapsed).count();
    
    printf("Computation time: %ld ms\n",
           std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count());
    
    // Calculate operations per second
    // Floyd-Warshall has O(n³) complexity
    double ops = (double)numNodes * numNodes * numNodes;
    double gflops = ops / (milliseconds * 1e6);
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
