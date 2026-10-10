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
    for (size_t j = 0; j < numNodes; ++j) {
        for (size_t i = 0; i < numNodes; ++i) {
            path[idx2(i, j, numNodes)] = j;
            path[idx2(j, i, numNodes)] = i;
        }
        path[idx2(j, j, numNodes)] = j;
    }
}

namespace {

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

constexpr unsigned int TILE = 32;
constexpr unsigned int ROWS_PER_THREAD = 4;

// A label is one plus the largest intermediate vertex, or zero for a
// direct edge. Among equal-length paths, prefer the smallest label. This
// preserves the original strictly-improving, ascending-pivot path matrix
// even though the blocked algorithm discovers candidate paths out of order.
__device__ __forceinline__ void relax(unsigned int& distance, unsigned int& label,
                                     unsigned int leftDistance, unsigned int rightDistance,
                                     unsigned int leftLabel, unsigned int rightLabel,
                                     unsigned int pivot) {
    const unsigned int candidate = leftDistance + rightDistance;
    const unsigned int candidateLabel = max(pivot + 1, max(leftLabel, rightLabel));
    label = candidate < distance ? candidateLabel
          : candidate == distance ? min(label, candidateLabel) : label;
    distance = min(distance, candidate);
}

__device__ __forceinline__ void loadTile(const unsigned int* dist,
                                        const unsigned int* labels, size_t n,
                                        size_t tileRow, size_t tileColumn,
                                        unsigned int* tileDist, unsigned int* tileLabels) {
    const size_t column = tileColumn * TILE + threadIdx.x;
    #pragma unroll
    for (unsigned int entry = 0; entry < ROWS_PER_THREAD; ++entry) {
        const unsigned int localRow = threadIdx.y + entry * blockDim.y;
        const size_t row = tileRow * TILE + localRow;
        const unsigned int local = localRow * TILE + threadIdx.x;
        // Padded vertices never improve any real path: real distances are
        // at most INF, and 2*INF fits in unsigned int.
        tileDist[local] = row < n && column < n ? dist[row * n + column] : INF;
        tileLabels[local] = row < n && column < n ? labels[row * n + column] : 0;
    }
}

__device__ __forceinline__ void storeTile(unsigned int* dist, unsigned int* labels,
                                         size_t n, size_t tileRow, size_t tileColumn,
                                         const unsigned int* tileDist,
                                         const unsigned int* tileLabels) {
    const size_t column = tileColumn * TILE + threadIdx.x;
    #pragma unroll
    for (unsigned int entry = 0; entry < ROWS_PER_THREAD; ++entry) {
        const unsigned int localRow = threadIdx.y + entry * blockDim.y;
        const size_t row = tileRow * TILE + localRow;
        if (row < n && column < n) {
            const unsigned int local = localRow * TILE + threadIdx.x;
            dist[row * n + column] = tileDist[local];
            labels[row * n + column] = tileLabels[local];
        }
    }
}

// Phase 1: close the diagonal tile using successive pivots.
__global__ void closePivot(unsigned int* dist, unsigned int* labels, size_t n,
                            size_t pivotTile) {
    __shared__ unsigned int tileDist[TILE * TILE];
    __shared__ unsigned int tileLabels[TILE * TILE];
    loadTile(dist, labels, n, pivotTile, pivotTile, tileDist, tileLabels);
    __syncthreads();
    for (unsigned int k = 0; k < TILE; ++k) {
        unsigned int distance[ROWS_PER_THREAD], label[ROWS_PER_THREAD];
        #pragma unroll
        for (unsigned int entry = 0; entry < ROWS_PER_THREAD; ++entry) {
            const unsigned int row = threadIdx.y + entry * blockDim.y;
            const unsigned int index = row * TILE + threadIdx.x;
            distance[entry] = tileDist[index];
            label[entry] = tileLabels[index];
            relax(distance[entry], label[entry], tileDist[row * TILE + k],
                  tileDist[k * TILE + threadIdx.x], tileLabels[row * TILE + k],
                  tileLabels[k * TILE + threadIdx.x], pivotTile * TILE + k);
        }
        // Snapshot all operands before updating the shared tile.
        __syncthreads();
        #pragma unroll
        for (unsigned int entry = 0; entry < ROWS_PER_THREAD; ++entry) {
            const unsigned int index = (threadIdx.y + entry * blockDim.y) * TILE + threadIdx.x;
            tileDist[index] = distance[entry];
            tileLabels[index] = label[entry];
        }
        __syncthreads();
    }
    storeTile(dist, labels, n, pivotTile, pivotTile, tileDist, tileLabels);
}

// Phase 2: independently update each tile in the pivot row and column.
__global__ void closePanels(unsigned int* dist, unsigned int* labels, size_t n,
                             size_t pivotTile) {
    const size_t other = blockIdx.x < pivotTile ? blockIdx.x : blockIdx.x + 1;
    const bool columnPanel = blockIdx.y != 0;
    const size_t tileRow = columnPanel ? other : pivotTile;
    const size_t tileColumn = columnPanel ? pivotTile : other;
    __shared__ unsigned int pivotDist[TILE * TILE];
    __shared__ unsigned int pivotLabels[TILE * TILE];
    __shared__ unsigned int panelDist[TILE * TILE];
    __shared__ unsigned int panelLabels[TILE * TILE];
    loadTile(dist, labels, n, pivotTile, pivotTile, pivotDist, pivotLabels);
    loadTile(dist, labels, n, tileRow, tileColumn, panelDist, panelLabels);
    __syncthreads();
    for (unsigned int k = 0; k < TILE; ++k) {
        unsigned int distance[ROWS_PER_THREAD], label[ROWS_PER_THREAD];
        #pragma unroll
        for (unsigned int entry = 0; entry < ROWS_PER_THREAD; ++entry) {
            const unsigned int row = threadIdx.y + entry * blockDim.y;
            const unsigned int index = row * TILE + threadIdx.x;
            distance[entry] = panelDist[index];
            label[entry] = panelLabels[index];
            const unsigned int* leftDist = columnPanel ? panelDist : pivotDist;
            const unsigned int* rightDist = columnPanel ? pivotDist : panelDist;
            const unsigned int* leftLabels = columnPanel ? panelLabels : pivotLabels;
            const unsigned int* rightLabels = columnPanel ? pivotLabels : panelLabels;
            relax(distance[entry], label[entry], leftDist[row * TILE + k],
                  rightDist[k * TILE + threadIdx.x], leftLabels[row * TILE + k],
                  rightLabels[k * TILE + threadIdx.x], pivotTile * TILE + k);
        }
        __syncthreads();
        #pragma unroll
        for (unsigned int entry = 0; entry < ROWS_PER_THREAD; ++entry) {
            const unsigned int index = (threadIdx.y + entry * blockDim.y) * TILE + threadIdx.x;
            panelDist[index] = distance[entry];
            panelLabels[index] = label[entry];
        }
        __syncthreads();
    }
    storeTile(dist, labels, n, tileRow, tileColumn, panelDist, panelLabels);
}

// Phase 3: all remaining tiles update in parallel, with no synchronization
// inside the pivot loop. The output tile stays in registers throughout.
__global__ void closeRemaining(unsigned int* dist, unsigned int* labels, size_t n,
                                size_t pivotTile) {
    const size_t tileColumn = blockIdx.x < pivotTile ? blockIdx.x : blockIdx.x + 1;
    const size_t tileRow = blockIdx.y < pivotTile ? blockIdx.y : blockIdx.y + 1;
    __shared__ unsigned int leftDist[TILE * TILE];
    __shared__ unsigned int leftLabels[TILE * TILE];
    __shared__ unsigned int rightDist[TILE * TILE];
    __shared__ unsigned int rightLabels[TILE * TILE];
    loadTile(dist, labels, n, tileRow, pivotTile, leftDist, leftLabels);
    loadTile(dist, labels, n, pivotTile, tileColumn, rightDist, rightLabels);
    const size_t column = tileColumn * TILE + threadIdx.x;
    unsigned int distance[ROWS_PER_THREAD], label[ROWS_PER_THREAD];
    #pragma unroll
    for (unsigned int entry = 0; entry < ROWS_PER_THREAD; ++entry) {
        const size_t row = tileRow * TILE + threadIdx.y + entry * blockDim.y;
        distance[entry] = row < n && column < n ? dist[row * n + column] : INF;
        label[entry] = row < n && column < n ? labels[row * n + column] : 0;
    }
    __syncthreads();
    #pragma unroll
    for (unsigned int k = 0; k < TILE; ++k) {
        const unsigned int fromPivot = rightDist[k * TILE + threadIdx.x];
        const unsigned int fromPivotLabel = rightLabels[k * TILE + threadIdx.x];
        #pragma unroll
        for (unsigned int entry = 0; entry < ROWS_PER_THREAD; ++entry) {
            const unsigned int row = threadIdx.y + entry * blockDim.y;
            relax(distance[entry], label[entry], leftDist[row * TILE + k],
                  fromPivot, leftLabels[row * TILE + k], fromPivotLabel,
                  pivotTile * TILE + k);
        }
    }
    #pragma unroll
    for (unsigned int entry = 0; entry < ROWS_PER_THREAD; ++entry) {
        const size_t row = tileRow * TILE + threadIdx.y + entry * blockDim.y;
        if (row < n && column < n) {
            dist[row * n + column] = distance[entry];
            labels[row * n + column] = label[entry];
        }
    }
}

__global__ void decodePaths(unsigned int* labels, size_t n) {
    const size_t column = static_cast<size_t>(blockIdx.x) * TILE + threadIdx.x;
    #pragma unroll
    for (unsigned int entry = 0; entry < ROWS_PER_THREAD; ++entry) {
        const size_t row = static_cast<size_t>(blockIdx.y) * TILE
                         + threadIdx.y + entry * blockDim.y;
        if (row < n && column < n) {
            const size_t index = row * n + column;
            const unsigned int label = labels[index];
            // initializePathMatrix assigns the source row to direct edges.
            labels[index] = label == 0 ? static_cast<unsigned int>(row) : label - 1;
        }
    }
}

} // namespace

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes) {
    // Always use CUDA; there is no CPU execution path.
    checkCuda(cudaFree(nullptr), "initialize CUDA");
    if (numNodes == 0) return;

    const size_t bytes = dist.size() * sizeof(unsigned int);
    unsigned int* deviceDist = nullptr;
    unsigned int* deviceLabels = nullptr;
    checkCuda(cudaMalloc(&deviceDist, bytes), "allocate distance matrix");
    checkCuda(cudaMalloc(&deviceLabels, bytes), "allocate path labels");
    checkCuda(cudaMemcpy(deviceDist, dist.data(), bytes, cudaMemcpyHostToDevice),
              "upload distance matrix");
    checkCuda(cudaMemset(deviceLabels, 0, bytes), "initialize direct-edge labels");

    const size_t tiles = (numNodes + TILE - 1) / TILE;
    const dim3 threads(TILE, TILE / ROWS_PER_THREAD);
    for (size_t pivotTile = 0; pivotTile < tiles; ++pivotTile) {
        // A single stream provides the global dependency barriers between
        // phases, without CPU/device synchronization at each pivot.
        closePivot<<<1, threads>>>(deviceDist, deviceLabels, numNodes, pivotTile);
        if (tiles > 1) {
            closePanels<<<dim3(tiles - 1, 2), threads>>>(
                deviceDist, deviceLabels, numNodes, pivotTile);
            closeRemaining<<<dim3(tiles - 1, tiles - 1), threads>>>(
                deviceDist, deviceLabels, numNodes, pivotTile);
        }
    }
    decodePaths<<<dim3(tiles, tiles), threads>>>(deviceLabels, numNodes);
    checkCuda(cudaGetLastError(), "launch blocked Floyd-Warshall");
    checkCuda(cudaMemcpy(dist.data(), deviceDist, bytes, cudaMemcpyDeviceToHost),
              "download distance matrix");
    checkCuda(cudaMemcpy(path.data(), deviceLabels, bytes, cudaMemcpyDeviceToHost),
              "download path matrix");
    checkCuda(cudaFree(deviceLabels), "free path labels");
    checkCuda(cudaFree(deviceDist), "free distance matrix");
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
    
    // Initialize the CUDA context before measuring the computation.
    checkCuda(cudaFree(nullptr), "initialize CUDA");

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
