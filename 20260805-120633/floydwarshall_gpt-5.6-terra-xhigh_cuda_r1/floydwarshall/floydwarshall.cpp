#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;
constexpr unsigned int TILE_SIZE = 32;

// Index calculation for flattened 2D array
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

inline void checkCuda(const cudaError_t status, const char* const operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error while %s: %s\n", operation,
                     cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(operation) checkCuda((operation), #operation)

// Matrix elements are stored as matrix[destination][source], matching idx2.
// Padding the second dimension prevents shared-memory bank conflicts when a
// warp accesses a fixed source/destination in the inner relaxation loop.
__global__ void updatePivotTile(unsigned int* __restrict__ dist,
                                unsigned int* __restrict__ path,
                                const unsigned int numNodes,
                                const unsigned int pivotTile) {
    __shared__ unsigned int tile[TILE_SIZE][TILE_SIZE + 1];

    const unsigned int sourceInTile = threadIdx.x;
    const unsigned int destinationInTile = threadIdx.y;
    const unsigned int tileStart = pivotTile * TILE_SIZE;
    const unsigned int source = tileStart + sourceInTile;
    const unsigned int destination = tileStart + destinationInTile;
    const bool inBounds = source < numNodes && destination < numNodes;
    const size_t offset = static_cast<size_t>(destination) * numNodes + source;

    unsigned int value = inBounds ? dist[offset] : INF;
    unsigned int pathValue = inBounds ? path[offset] : 0;
    tile[destinationInTile][sourceInTile] = value;
    __syncthreads();

#pragma unroll
    for (unsigned int k = 0; k < TILE_SIZE; ++k) {
        const unsigned int candidate =
            tile[k][sourceInTile] + tile[destinationInTile][k];
        if (candidate < value) {
            value = candidate;
            pathValue = tileStart + k;
        }

        // Every thread must finish reading the old tile before any thread
        // publishes its value for this intermediate vertex.
        __syncthreads();
        tile[destinationInTile][sourceInTile] = value;
        __syncthreads();
    }

    if (inBounds) {
        dist[offset] = value;
        path[offset] = pathValue;
    }
}

// A block handles both non-pivot tiles belonging to one tile index: the
// pivot column (left) and the pivot row (right).  Combining them cuts phase-2
// launches and reuses the pivot tile from shared memory.
__global__ void updatePivotRowAndColumn(unsigned int* __restrict__ dist,
                                        unsigned int* __restrict__ path,
                                        const unsigned int numNodes,
                                        const unsigned int pivotTile) {
    const unsigned int otherTile = blockIdx.x;
    if (otherTile == pivotTile) {
        return;
    }

    __shared__ unsigned int pivot[TILE_SIZE][TILE_SIZE + 1];
    __shared__ unsigned int column[TILE_SIZE][TILE_SIZE + 1];
    __shared__ unsigned int row[TILE_SIZE][TILE_SIZE + 1];

    const unsigned int sourceInTile = threadIdx.x;
    const unsigned int destinationInTile = threadIdx.y;
    const unsigned int pivotStart = pivotTile * TILE_SIZE;
    const unsigned int otherStart = otherTile * TILE_SIZE;

    const unsigned int pivotSource = pivotStart + sourceInTile;
    const unsigned int pivotDestination = pivotStart + destinationInTile;
    const bool pivotInBounds = pivotSource < numNodes && pivotDestination < numNodes;
    const size_t pivotOffset =
        static_cast<size_t>(pivotDestination) * numNodes + pivotSource;
    pivot[destinationInTile][sourceInTile] = pivotInBounds ? dist[pivotOffset] : INF;

    // column = dist[pivot destination][other source]
    const unsigned int columnSource = otherStart + sourceInTile;
    const unsigned int columnDestination = pivotStart + destinationInTile;
    const bool columnInBounds = columnSource < numNodes && columnDestination < numNodes;
    const size_t columnOffset =
        static_cast<size_t>(columnDestination) * numNodes + columnSource;
    unsigned int columnValue = columnInBounds ? dist[columnOffset] : INF;
    unsigned int columnPath = columnInBounds ? path[columnOffset] : 0;
    column[destinationInTile][sourceInTile] = columnValue;

    // row = dist[other destination][pivot source]
    const unsigned int rowSource = pivotStart + sourceInTile;
    const unsigned int rowDestination = otherStart + destinationInTile;
    const bool rowInBounds = rowSource < numNodes && rowDestination < numNodes;
    const size_t rowOffset = static_cast<size_t>(rowDestination) * numNodes + rowSource;
    unsigned int rowValue = rowInBounds ? dist[rowOffset] : INF;
    unsigned int rowPath = rowInBounds ? path[rowOffset] : 0;
    row[destinationInTile][sourceInTile] = rowValue;
    __syncthreads();

#pragma unroll
    for (unsigned int k = 0; k < TILE_SIZE; ++k) {
        const unsigned int columnCandidate =
            column[k][sourceInTile] + pivot[destinationInTile][k];
        const unsigned int rowCandidate =
            pivot[k][sourceInTile] + row[destinationInTile][k];

        if (columnCandidate < columnValue) {
            columnValue = columnCandidate;
            columnPath = pivotStart + k;
        }
        if (rowCandidate < rowValue) {
            rowValue = rowCandidate;
            rowPath = pivotStart + k;
        }

        __syncthreads();
        column[destinationInTile][sourceInTile] = columnValue;
        row[destinationInTile][sourceInTile] = rowValue;
        __syncthreads();
    }

    if (columnInBounds) {
        dist[columnOffset] = columnValue;
        path[columnOffset] = columnPath;
    }
    if (rowInBounds) {
        dist[rowOffset] = rowValue;
        path[rowOffset] = rowPath;
    }
}

// All tiles outside the pivot row and column are independent after phase 2.
// One thread owns one result matrix element and keeps it in a register while
// the two read-only input tiles are cached in shared memory.
__global__ void updateRemainingTiles(unsigned int* __restrict__ dist,
                                     unsigned int* __restrict__ path,
                                     const unsigned int numNodes,
                                     const unsigned int pivotTile) {
    const unsigned int sourceTile = blockIdx.x;
    const unsigned int destinationTile = blockIdx.y;
    if (sourceTile == pivotTile || destinationTile == pivotTile) {
        return;
    }

    __shared__ unsigned int column[TILE_SIZE][TILE_SIZE + 1];
    __shared__ unsigned int row[TILE_SIZE][TILE_SIZE + 1];

    const unsigned int sourceInTile = threadIdx.x;
    const unsigned int destinationInTile = threadIdx.y;
    const unsigned int pivotStart = pivotTile * TILE_SIZE;
    const unsigned int sourceStart = sourceTile * TILE_SIZE;
    const unsigned int destinationStart = destinationTile * TILE_SIZE;

    // dist[pivot destination][source]
    const unsigned int columnSource = sourceStart + sourceInTile;
    const unsigned int columnDestination = pivotStart + destinationInTile;
    const bool columnInBounds = columnSource < numNodes && columnDestination < numNodes;
    const size_t columnOffset =
        static_cast<size_t>(columnDestination) * numNodes + columnSource;
    column[destinationInTile][sourceInTile] =
        columnInBounds ? dist[columnOffset] : INF;

    // dist[destination][pivot source]
    const unsigned int rowSource = pivotStart + sourceInTile;
    const unsigned int rowDestination = destinationStart + destinationInTile;
    const bool rowInBounds = rowSource < numNodes && rowDestination < numNodes;
    const size_t rowOffset = static_cast<size_t>(rowDestination) * numNodes + rowSource;
    row[destinationInTile][sourceInTile] = rowInBounds ? dist[rowOffset] : INF;

    const unsigned int source = sourceStart + sourceInTile;
    const unsigned int destination = destinationStart + destinationInTile;
    const bool resultInBounds = source < numNodes && destination < numNodes;
    const size_t resultOffset = static_cast<size_t>(destination) * numNodes + source;
    unsigned int value = resultInBounds ? dist[resultOffset] : INF;
    unsigned int pathValue = resultInBounds ? path[resultOffset] : 0;
    __syncthreads();

#pragma unroll
    for (unsigned int k = 0; k < TILE_SIZE; ++k) {
        const unsigned int candidate =
            column[k][sourceInTile] + row[destinationInTile][k];
        if (candidate < value) {
            value = candidate;
            pathValue = pivotStart + k;
        }
    }

    if (resultInBounds) {
        dist[resultOffset] = value;
        path[resultOffset] = pathValue;
    }
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

void floydWarshall(unsigned int* const dist, unsigned int* const path,
                   const unsigned int numNodes) {
    if (numNodes == 0) {
        return;
    }

    const unsigned int numTiles =
        numNodes / TILE_SIZE + static_cast<unsigned int>(numNodes % TILE_SIZE != 0);
    const dim3 threads(TILE_SIZE, TILE_SIZE);

    // The three phases for a pivot tile are ordered.  Within phase 2 and
    // phase 3, every tile is independent and therefore runs concurrently.
    for (unsigned int pivotTile = 0; pivotTile < numTiles; ++pivotTile) {
        updatePivotTile<<<1, threads>>>(dist, path, numNodes, pivotTile);
        CUDA_CHECK(cudaGetLastError());

        updatePivotRowAndColumn<<<numTiles, threads>>>(dist, path, numNodes, pivotTile);
        CUDA_CHECK(cudaGetLastError());

        const dim3 tiles(numTiles, numTiles);
        updateRemainingTiles<<<tiles, threads>>>(dist, path, numNodes, pivotTile);
        CUDA_CHECK(cudaGetLastError());
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
    
    if (numNodes > std::numeric_limits<unsigned int>::max()) {
        std::fprintf(stderr, "Number of nodes exceeds the CUDA implementation limit\n");
        return 1;
    }
    if (numNodes != 0 && numNodes >
            std::numeric_limits<size_t>::max() / numNodes) {
        std::fprintf(stderr, "Distance matrix size overflows addressable memory\n");
        return 1;
    }

    // Allocate matrices
    std::vector<unsigned int> dist(numNodes * numNodes);
    std::vector<unsigned int> path(numNodes * numNodes);
    
    // Initialize
    printf("Initializing graph...\n");
    initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
    initializePathMatrix(path, numNodes);
    
    // Transfer the initialized matrices once, run the complete algorithm on
    // the GPU, then transfer the final result back for output/validation.
    const size_t matrixBytes = dist.size() * sizeof(unsigned int);
    unsigned int* deviceDist = nullptr;
    unsigned int* devicePath = nullptr;
    CUDA_CHECK(cudaFree(nullptr)); // create the CUDA context before timing
    CUDA_CHECK(cudaMalloc(&deviceDist, matrixBytes));
    CUDA_CHECK(cudaMalloc(&devicePath, matrixBytes));
    CUDA_CHECK(cudaMemcpy(deviceDist, dist.data(), matrixBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(devicePath, path.data(), matrixBytes, cudaMemcpyHostToDevice));

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Run Floyd-Warshall. CUDA events measure device execution only, so setup
    // and result transfers do not dilute the kernel throughput measurement.
    printf("Computing shortest paths...\n");
    CUDA_CHECK(cudaEventRecord(start));
    floydWarshall(deviceDist, devicePath, static_cast<unsigned int>(numNodes));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsedMilliseconds = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedMilliseconds, start, stop));
    CUDA_CHECK(cudaMemcpy(dist.data(), deviceDist, matrixBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(path.data(), devicePath, matrixBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(deviceDist));
    CUDA_CHECK(cudaFree(devicePath));

    printf("Computation time: %.3f ms\n", static_cast<double>(elapsedMilliseconds));
    
    // Calculate operations per second
    // Floyd-Warshall has O(n³) complexity
    double ops = (double)numNodes * numNodes * numNodes;
    double gflops = elapsedMilliseconds > 0.0F
        ? ops / (static_cast<double>(elapsedMilliseconds) * 1.0e6)
        : 0.0;
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
