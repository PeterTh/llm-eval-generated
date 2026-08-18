#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;
constexpr int TILE_SIZE = 32;
constexpr int BLOCK_SIZE = 16;
constexpr int ELEMENTS_PER_THREAD = TILE_SIZE / BLOCK_SIZE;

static_assert(TILE_SIZE % BLOCK_SIZE == 0);

// Index calculation for flattened 2D array
__host__ __device__ inline constexpr size_t idx2(const size_t i, const size_t j,
                                                  const size_t n) noexcept {
    return j * n + i;
}

#define CUDA_CHECK(call)                                                                          \
    do {                                                                                          \
        const cudaError_t cudaStatus = (call);                                                    \
        if (cudaStatus != cudaSuccess) {                                                          \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,                   \
                    cudaGetErrorString(cudaStatus));                                              \
            std::exit(EXIT_FAILURE);                                                              \
        }                                                                                         \
    } while (false)

// Matrix entries are stored as matrix[column][row].  A 16x16 thread block owns
// a 32x32 tile, so each thread handles a 2x2 sub-tile.  This retains fully
// coalesced accesses while allowing enough blocks to reside on each SM.
__global__ void floydPivotKernel(unsigned int* __restrict__ dist,
                                 unsigned int* __restrict__ path,
                                 const size_t numNodes, const size_t pivotTile) {
    __shared__ unsigned int tileDist[TILE_SIZE * TILE_SIZE];
    __shared__ unsigned int tilePath[TILE_SIZE * TILE_SIZE];

    const size_t pivotStart = pivotTile * TILE_SIZE;
    const size_t pivotRemaining = numNodes - pivotStart;
    const int pivotWidth = pivotRemaining < TILE_SIZE ? static_cast<int>(pivotRemaining) : TILE_SIZE;

    #pragma unroll
    for (int tileJ = 0; tileJ < ELEMENTS_PER_THREAD; ++tileJ) {
        const int localJ = threadIdx.y + tileJ * BLOCK_SIZE;
        const size_t j = pivotStart + static_cast<size_t>(localJ);
        #pragma unroll
        for (int tileI = 0; tileI < ELEMENTS_PER_THREAD; ++tileI) {
            const int localI = threadIdx.x + tileI * BLOCK_SIZE;
            const size_t i = pivotStart + static_cast<size_t>(localI);
            const int localIndex = localJ * TILE_SIZE + localI;
            if (i < numNodes && j < numNodes) {
                tileDist[localIndex] = dist[idx2(i, j, numNodes)];
                tilePath[localIndex] = path[idx2(i, j, numNodes)];
            } else {
                tileDist[localIndex] = INF;
                tilePath[localIndex] = 0;
            }
        }
    }
    __syncthreads();

#pragma unroll
    for (int localK = 0; localK < TILE_SIZE; ++localK) {
        const bool activeK = localK < pivotWidth;
        unsigned int candidate[ELEMENTS_PER_THREAD][ELEMENTS_PER_THREAD];
        unsigned int current[ELEMENTS_PER_THREAD][ELEMENTS_PER_THREAD];

        #pragma unroll
        for (int tileJ = 0; tileJ < ELEMENTS_PER_THREAD; ++tileJ) {
            const int localJ = threadIdx.y + tileJ * BLOCK_SIZE;
            #pragma unroll
            for (int tileI = 0; tileI < ELEMENTS_PER_THREAD; ++tileI) {
                const int localI = threadIdx.x + tileI * BLOCK_SIZE;
                const int localIndex = localJ * TILE_SIZE + localI;
                candidate[tileJ][tileI] = tileDist[localK * TILE_SIZE + localI] +
                                          tileDist[localJ * TILE_SIZE + localK];
                current[tileJ][tileI] = tileDist[localIndex];
            }
        }
        __syncthreads();

        #pragma unroll
        for (int tileJ = 0; tileJ < ELEMENTS_PER_THREAD; ++tileJ) {
            const int localJ = threadIdx.y + tileJ * BLOCK_SIZE;
            const size_t j = pivotStart + static_cast<size_t>(localJ);
            #pragma unroll
            for (int tileI = 0; tileI < ELEMENTS_PER_THREAD; ++tileI) {
                const int localI = threadIdx.x + tileI * BLOCK_SIZE;
                const size_t i = pivotStart + static_cast<size_t>(localI);
                const int localIndex = localJ * TILE_SIZE + localI;
                if (activeK && i < numNodes && j < numNodes &&
                    candidate[tileJ][tileI] < current[tileJ][tileI]) {
                    tileDist[localIndex] = candidate[tileJ][tileI];
                    tilePath[localIndex] = static_cast<unsigned int>(pivotStart + localK);
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int tileJ = 0; tileJ < ELEMENTS_PER_THREAD; ++tileJ) {
        const int localJ = threadIdx.y + tileJ * BLOCK_SIZE;
        const size_t j = pivotStart + static_cast<size_t>(localJ);
        #pragma unroll
        for (int tileI = 0; tileI < ELEMENTS_PER_THREAD; ++tileI) {
            const int localI = threadIdx.x + tileI * BLOCK_SIZE;
            const size_t i = pivotStart + static_cast<size_t>(localI);
            const int localIndex = localJ * TILE_SIZE + localI;
            if (i < numNodes && j < numNodes) {
                dist[idx2(i, j, numNodes)] = tileDist[localIndex];
                path[idx2(i, j, numNodes)] = tilePath[localIndex];
            }
        }
    }
}

// Update every tile in the pivot row.  The completed pivot tile is read only;
// each block owns one destination tile and advances through its local k values
// in the same order as the scalar algorithm.
__global__ void floydRowKernel(unsigned int* __restrict__ dist,
                               unsigned int* __restrict__ path,
                               const size_t numNodes, const size_t pivotTile) {
    const size_t compressedTile = blockIdx.x;
    const size_t destinationTile = compressedTile + (compressedTile >= pivotTile);

    __shared__ unsigned int pivotDist[TILE_SIZE * TILE_SIZE];
    __shared__ unsigned int rowDist[TILE_SIZE * TILE_SIZE];
    __shared__ unsigned int rowPath[TILE_SIZE * TILE_SIZE];

    const size_t pivotStart = pivotTile * TILE_SIZE;
    const size_t destinationStart = destinationTile * TILE_SIZE;
    const size_t pivotRemaining = numNodes - pivotStart;
    const int pivotWidth = pivotRemaining < TILE_SIZE ? static_cast<int>(pivotRemaining) : TILE_SIZE;

    #pragma unroll
    for (int tileJ = 0; tileJ < ELEMENTS_PER_THREAD; ++tileJ) {
        const int localJ = threadIdx.y + tileJ * BLOCK_SIZE;
        const size_t pivotJ = pivotStart + static_cast<size_t>(localJ);
        const size_t destinationJ = destinationStart + static_cast<size_t>(localJ);
        #pragma unroll
        for (int tileI = 0; tileI < ELEMENTS_PER_THREAD; ++tileI) {
            const int localI = threadIdx.x + tileI * BLOCK_SIZE;
            const size_t pivotI = pivotStart + static_cast<size_t>(localI);
            const int localIndex = localJ * TILE_SIZE + localI;
            pivotDist[localIndex] = pivotI < numNodes && pivotJ < numNodes
                                        ? dist[idx2(pivotI, pivotJ, numNodes)]
                                        : INF;
            if (pivotI < numNodes && destinationJ < numNodes) {
                rowDist[localIndex] = dist[idx2(pivotI, destinationJ, numNodes)];
                rowPath[localIndex] = path[idx2(pivotI, destinationJ, numNodes)];
            } else {
                rowDist[localIndex] = INF;
                rowPath[localIndex] = 0;
            }
        }
    }
    __syncthreads();

#pragma unroll
    for (int localK = 0; localK < TILE_SIZE; ++localK) {
        const bool activeK = localK < pivotWidth;
        unsigned int candidate[ELEMENTS_PER_THREAD][ELEMENTS_PER_THREAD];
        unsigned int current[ELEMENTS_PER_THREAD][ELEMENTS_PER_THREAD];
        #pragma unroll
        for (int tileJ = 0; tileJ < ELEMENTS_PER_THREAD; ++tileJ) {
            const int localJ = threadIdx.y + tileJ * BLOCK_SIZE;
            #pragma unroll
            for (int tileI = 0; tileI < ELEMENTS_PER_THREAD; ++tileI) {
                const int localI = threadIdx.x + tileI * BLOCK_SIZE;
                const int localIndex = localJ * TILE_SIZE + localI;
                candidate[tileJ][tileI] = pivotDist[localK * TILE_SIZE + localI] +
                                          rowDist[localJ * TILE_SIZE + localK];
                current[tileJ][tileI] = rowDist[localIndex];
            }
        }
        __syncthreads();
        #pragma unroll
        for (int tileJ = 0; tileJ < ELEMENTS_PER_THREAD; ++tileJ) {
            const int localJ = threadIdx.y + tileJ * BLOCK_SIZE;
            const size_t destinationJ = destinationStart + static_cast<size_t>(localJ);
            #pragma unroll
            for (int tileI = 0; tileI < ELEMENTS_PER_THREAD; ++tileI) {
                const int localI = threadIdx.x + tileI * BLOCK_SIZE;
                const size_t pivotI = pivotStart + static_cast<size_t>(localI);
                const int localIndex = localJ * TILE_SIZE + localI;
                if (activeK && pivotI < numNodes && destinationJ < numNodes &&
                    candidate[tileJ][tileI] < current[tileJ][tileI]) {
                    rowDist[localIndex] = candidate[tileJ][tileI];
                    rowPath[localIndex] = static_cast<unsigned int>(pivotStart + localK);
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int tileJ = 0; tileJ < ELEMENTS_PER_THREAD; ++tileJ) {
        const int localJ = threadIdx.y + tileJ * BLOCK_SIZE;
        const size_t destinationJ = destinationStart + static_cast<size_t>(localJ);
        #pragma unroll
        for (int tileI = 0; tileI < ELEMENTS_PER_THREAD; ++tileI) {
            const int localI = threadIdx.x + tileI * BLOCK_SIZE;
            const size_t pivotI = pivotStart + static_cast<size_t>(localI);
            const int localIndex = localJ * TILE_SIZE + localI;
            if (pivotI < numNodes && destinationJ < numNodes) {
                dist[idx2(pivotI, destinationJ, numNodes)] = rowDist[localIndex];
                path[idx2(pivotI, destinationJ, numNodes)] = rowPath[localIndex];
            }
        }
    }
}

// Update every tile in the pivot column.  This is the transpose of the pivot
// row phase, but uses the original column-major layout directly.
__global__ void floydColumnKernel(unsigned int* __restrict__ dist,
                                  unsigned int* __restrict__ path,
                                  const size_t numNodes, const size_t pivotTile) {
    const size_t compressedTile = blockIdx.x;
    const size_t sourceTile = compressedTile + (compressedTile >= pivotTile);

    __shared__ unsigned int columnDist[TILE_SIZE * TILE_SIZE];
    __shared__ unsigned int columnPath[TILE_SIZE * TILE_SIZE];
    __shared__ unsigned int pivotDist[TILE_SIZE * TILE_SIZE];

    const size_t sourceStart = sourceTile * TILE_SIZE;
    const size_t pivotStart = pivotTile * TILE_SIZE;
    const size_t pivotRemaining = numNodes - pivotStart;
    const int pivotWidth = pivotRemaining < TILE_SIZE ? static_cast<int>(pivotRemaining) : TILE_SIZE;

    #pragma unroll
    for (int tileJ = 0; tileJ < ELEMENTS_PER_THREAD; ++tileJ) {
        const int localJ = threadIdx.y + tileJ * BLOCK_SIZE;
        const size_t pivotJ = pivotStart + static_cast<size_t>(localJ);
        #pragma unroll
        for (int tileI = 0; tileI < ELEMENTS_PER_THREAD; ++tileI) {
            const int localI = threadIdx.x + tileI * BLOCK_SIZE;
            const size_t sourceI = sourceStart + static_cast<size_t>(localI);
            const size_t pivotI = pivotStart + static_cast<size_t>(localI);
            const int localIndex = localJ * TILE_SIZE + localI;
            pivotDist[localIndex] = pivotI < numNodes && pivotJ < numNodes
                                        ? dist[idx2(pivotI, pivotJ, numNodes)]
                                        : INF;
            if (sourceI < numNodes && pivotJ < numNodes) {
                columnDist[localIndex] = dist[idx2(sourceI, pivotJ, numNodes)];
                columnPath[localIndex] = path[idx2(sourceI, pivotJ, numNodes)];
            } else {
                columnDist[localIndex] = INF;
                columnPath[localIndex] = 0;
            }
        }
    }
    __syncthreads();

#pragma unroll
    for (int localK = 0; localK < TILE_SIZE; ++localK) {
        const bool activeK = localK < pivotWidth;
        unsigned int candidate[ELEMENTS_PER_THREAD][ELEMENTS_PER_THREAD];
        unsigned int current[ELEMENTS_PER_THREAD][ELEMENTS_PER_THREAD];
        #pragma unroll
        for (int tileJ = 0; tileJ < ELEMENTS_PER_THREAD; ++tileJ) {
            const int localJ = threadIdx.y + tileJ * BLOCK_SIZE;
            #pragma unroll
            for (int tileI = 0; tileI < ELEMENTS_PER_THREAD; ++tileI) {
                const int localI = threadIdx.x + tileI * BLOCK_SIZE;
                const int localIndex = localJ * TILE_SIZE + localI;
                candidate[tileJ][tileI] = columnDist[localK * TILE_SIZE + localI] +
                                          pivotDist[localJ * TILE_SIZE + localK];
                current[tileJ][tileI] = columnDist[localIndex];
            }
        }
        __syncthreads();
        #pragma unroll
        for (int tileJ = 0; tileJ < ELEMENTS_PER_THREAD; ++tileJ) {
            const int localJ = threadIdx.y + tileJ * BLOCK_SIZE;
            const size_t pivotJ = pivotStart + static_cast<size_t>(localJ);
            #pragma unroll
            for (int tileI = 0; tileI < ELEMENTS_PER_THREAD; ++tileI) {
                const int localI = threadIdx.x + tileI * BLOCK_SIZE;
                const size_t sourceI = sourceStart + static_cast<size_t>(localI);
                const int localIndex = localJ * TILE_SIZE + localI;
                if (activeK && sourceI < numNodes && pivotJ < numNodes &&
                    candidate[tileJ][tileI] < current[tileJ][tileI]) {
                    columnDist[localIndex] = candidate[tileJ][tileI];
                    columnPath[localIndex] = static_cast<unsigned int>(pivotStart + localK);
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int tileJ = 0; tileJ < ELEMENTS_PER_THREAD; ++tileJ) {
        const int localJ = threadIdx.y + tileJ * BLOCK_SIZE;
        const size_t pivotJ = pivotStart + static_cast<size_t>(localJ);
        #pragma unroll
        for (int tileI = 0; tileI < ELEMENTS_PER_THREAD; ++tileI) {
            const int localI = threadIdx.x + tileI * BLOCK_SIZE;
            const size_t sourceI = sourceStart + static_cast<size_t>(localI);
            const int localIndex = localJ * TILE_SIZE + localI;
            if (sourceI < numNodes && pivotJ < numNodes) {
                dist[idx2(sourceI, pivotJ, numNodes)] = columnDist[localIndex];
                path[idx2(sourceI, pivotJ, numNodes)] = columnPath[localIndex];
            }
        }
    }
}

// All non-pivot tiles are independent once their matching pivot row and column
// have been completed.  Each thread updates a 2x2 output sub-tile from the two
// shared-memory input tiles, retaining all four values in registers.
__global__ void floydRemainingKernel(unsigned int* __restrict__ dist,
                                    unsigned int* __restrict__ path,
                                    const size_t numNodes, const size_t pivotTile) {
    const size_t compressedSourceTile = blockIdx.x;
    const size_t compressedDestinationTile = blockIdx.y;
    const size_t sourceTile = compressedSourceTile + (compressedSourceTile >= pivotTile);
    const size_t destinationTile = compressedDestinationTile +
                                   (compressedDestinationTile >= pivotTile);

    __shared__ unsigned int columnDist[TILE_SIZE * TILE_SIZE];
    __shared__ unsigned int rowDist[TILE_SIZE * TILE_SIZE];

    const size_t sourceStart = sourceTile * TILE_SIZE;
    const size_t destinationStart = destinationTile * TILE_SIZE;
    const size_t pivotStart = pivotTile * TILE_SIZE;
    const size_t pivotRemaining = numNodes - pivotStart;
    const int pivotWidth = pivotRemaining < TILE_SIZE ? static_cast<int>(pivotRemaining) : TILE_SIZE;

    #pragma unroll
    for (int tileJ = 0; tileJ < ELEMENTS_PER_THREAD; ++tileJ) {
        const int localJ = threadIdx.y + tileJ * BLOCK_SIZE;
        const size_t pivotJ = pivotStart + static_cast<size_t>(localJ);
        const size_t destinationJ = destinationStart + static_cast<size_t>(localJ);
        #pragma unroll
        for (int tileI = 0; tileI < ELEMENTS_PER_THREAD; ++tileI) {
            const int localI = threadIdx.x + tileI * BLOCK_SIZE;
            const size_t sourceI = sourceStart + static_cast<size_t>(localI);
            const size_t pivotI = pivotStart + static_cast<size_t>(localI);
            const int localIndex = localJ * TILE_SIZE + localI;
            columnDist[localIndex] = sourceI < numNodes && pivotJ < numNodes
                                         ? dist[idx2(sourceI, pivotJ, numNodes)]
                                         : INF;
            rowDist[localIndex] = pivotI < numNodes && destinationJ < numNodes
                                      ? dist[idx2(pivotI, destinationJ, numNodes)]
                                      : INF;
        }
    }
    __syncthreads();

    unsigned int current[ELEMENTS_PER_THREAD][ELEMENTS_PER_THREAD];
    unsigned int currentPath[ELEMENTS_PER_THREAD][ELEMENTS_PER_THREAD];
    bool outputInBounds[ELEMENTS_PER_THREAD][ELEMENTS_PER_THREAD];

    #pragma unroll
    for (int tileJ = 0; tileJ < ELEMENTS_PER_THREAD; ++tileJ) {
        const int localJ = threadIdx.y + tileJ * BLOCK_SIZE;
        const size_t destinationJ = destinationStart + static_cast<size_t>(localJ);
        #pragma unroll
        for (int tileI = 0; tileI < ELEMENTS_PER_THREAD; ++tileI) {
            const int localI = threadIdx.x + tileI * BLOCK_SIZE;
            const size_t sourceI = sourceStart + static_cast<size_t>(localI);
            outputInBounds[tileJ][tileI] = sourceI < numNodes && destinationJ < numNodes;
            if (outputInBounds[tileJ][tileI]) {
                current[tileJ][tileI] = dist[idx2(sourceI, destinationJ, numNodes)];
                currentPath[tileJ][tileI] = path[idx2(sourceI, destinationJ, numNodes)];
            } else {
                current[tileJ][tileI] = INF;
                currentPath[tileJ][tileI] = 0;
            }
        }
    }

#pragma unroll
    for (int localK = 0; localK < TILE_SIZE; ++localK) {
        if (localK < pivotWidth) {
            #pragma unroll
            for (int tileJ = 0; tileJ < ELEMENTS_PER_THREAD; ++tileJ) {
                const int localJ = threadIdx.y + tileJ * BLOCK_SIZE;
                #pragma unroll
                for (int tileI = 0; tileI < ELEMENTS_PER_THREAD; ++tileI) {
                    const int localI = threadIdx.x + tileI * BLOCK_SIZE;
                    const unsigned int candidate = columnDist[localK * TILE_SIZE + localI] +
                                                   rowDist[localJ * TILE_SIZE + localK];
                    if (outputInBounds[tileJ][tileI] && candidate < current[tileJ][tileI]) {
                        current[tileJ][tileI] = candidate;
                        currentPath[tileJ][tileI] = static_cast<unsigned int>(pivotStart + localK);
                    }
                }
            }
        }
    }

    #pragma unroll
    for (int tileJ = 0; tileJ < ELEMENTS_PER_THREAD; ++tileJ) {
        const int localJ = threadIdx.y + tileJ * BLOCK_SIZE;
        const size_t destinationJ = destinationStart + static_cast<size_t>(localJ);
        #pragma unroll
        for (int tileI = 0; tileI < ELEMENTS_PER_THREAD; ++tileI) {
            const int localI = threadIdx.x + tileI * BLOCK_SIZE;
            const size_t sourceI = sourceStart + static_cast<size_t>(localI);
            if (outputInBounds[tileJ][tileI]) {
                dist[idx2(sourceI, destinationJ, numNodes)] = current[tileJ][tileI];
                path[idx2(sourceI, destinationJ, numNodes)] = currentPath[tileJ][tileI];
            }
        }
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

double floydWarshall(std::vector<unsigned int>& dist,
                     std::vector<unsigned int>& path,
                     const size_t numNodes) {
    if (numNodes == 0) {
        return 0.0;
    }

    const size_t matrixElements = numNodes * numNodes;
    const size_t matrixBytes = matrixElements * sizeof(unsigned int);
    unsigned int* deviceDist = nullptr;
    unsigned int* devicePath = nullptr;
    cudaEvent_t startEvent = nullptr;
    cudaEvent_t stopEvent = nullptr;

    CUDA_CHECK(cudaMalloc(&deviceDist, matrixBytes));
    CUDA_CHECK(cudaMalloc(&devicePath, matrixBytes));
    CUDA_CHECK(cudaMemcpy(deviceDist, dist.data(), matrixBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(devicePath, path.data(), matrixBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventCreate(&startEvent));
    CUDA_CHECK(cudaEventCreate(&stopEvent));

    const size_t tileCount = (numNodes + TILE_SIZE - 1) / TILE_SIZE;
    const dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    const dim3 nonPivotTileLine(static_cast<unsigned int>(tileCount - 1));
    const dim3 nonPivotTilePlane(static_cast<unsigned int>(tileCount - 1),
                                 static_cast<unsigned int>(tileCount - 1));

    CUDA_CHECK(cudaEventRecord(startEvent));
    for (size_t pivotTile = 0; pivotTile < tileCount; ++pivotTile) {
        floydPivotKernel<<<1, block>>>(deviceDist, devicePath, numNodes, pivotTile);
        CUDA_CHECK(cudaGetLastError());

        if (tileCount > 1) {
            floydRowKernel<<<nonPivotTileLine, block>>>(deviceDist, devicePath, numNodes, pivotTile);
            CUDA_CHECK(cudaGetLastError());

            floydColumnKernel<<<nonPivotTileLine, block>>>(deviceDist, devicePath, numNodes, pivotTile);
            CUDA_CHECK(cudaGetLastError());

            floydRemainingKernel<<<nonPivotTilePlane, block>>>(deviceDist, devicePath, numNodes,
                                                                pivotTile);
            CUDA_CHECK(cudaGetLastError());
        }
    }
    CUDA_CHECK(cudaEventRecord(stopEvent));
    CUDA_CHECK(cudaEventSynchronize(stopEvent));

    float elapsedMilliseconds = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedMilliseconds, startEvent, stopEvent));
    CUDA_CHECK(cudaMemcpy(dist.data(), deviceDist, matrixBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(path.data(), devicePath, matrixBytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventDestroy(startEvent));
    CUDA_CHECK(cudaEventDestroy(stopEvent));
    CUDA_CHECK(cudaFree(deviceDist));
    CUDA_CHECK(cudaFree(devicePath));
    return static_cast<double>(elapsedMilliseconds);
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
    const double durationMilliseconds = floydWarshall(dist, path, numNodes);

    printf("Computation time: %.3f ms\n", durationMilliseconds);
    
    // Calculate operations per second
    // Floyd-Warshall has O(n³) complexity
    double ops = (double)numNodes * numNodes * numNodes;
    double gflops = durationMilliseconds > 0.0
                        ? ops / (durationMilliseconds / 1000.0) / 1e9
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
