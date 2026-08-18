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

namespace {

constexpr unsigned int TILE_SIZE = 32;
constexpr unsigned int THREAD_ROWS = 8;
constexpr unsigned int VALUES_PER_THREAD = TILE_SIZE / THREAD_ROWS;

[[noreturn]] void cudaFailure(const cudaError_t error, const char* expression,
                              const char* file, const int line) {
    std::fprintf(stderr, "CUDA error at %s:%d while evaluating %s: %s\n", file, line,
                 expression, cudaGetErrorString(error));
    std::exit(EXIT_FAILURE);
}

#define CUDA_CHECK(expression)                                                    \
    do {                                                                          \
        const cudaError_t cuda_check_status = (expression);                       \
        if (cuda_check_status != cudaSuccess) {                                   \
            cudaFailure(cuda_check_status, #expression, __FILE__, __LINE__);      \
        }                                                                         \
    } while (false)

// Close the diagonal tile for one group of 32 intermediate vertices.
__global__ __launch_bounds__(256) void floydWarshallDiagonal(
    unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
    const size_t numNodes, const unsigned int round) {
    __shared__ unsigned int tile[TILE_SIZE][TILE_SIZE + 1];

    const unsigned int x = threadIdx.x;
    const unsigned int threadRow = threadIdx.y;
    const size_t base = static_cast<size_t>(round) * TILE_SIZE;
    const size_t source = base + x;
    unsigned int values[VALUES_PER_THREAD];
    unsigned int lastPivot[VALUES_PER_THREAD];
    unsigned int changedMask = 0;

#pragma unroll
    for (unsigned int item = 0; item < VALUES_PER_THREAD; ++item) {
        const unsigned int y = threadRow + item * THREAD_ROWS;
        const size_t destination = base + y;
        const bool valid = source < numNodes && destination < numNodes;
        values[item] = valid ? dist[destination * numNodes + source] : INF;
        tile[y][x] = values[item];
    }
    __syncthreads();

    const unsigned int remaining = static_cast<unsigned int>(numNodes - base);
    const unsigned int pivotCount = min(TILE_SIZE, remaining);
#pragma unroll
    for (unsigned int k = 0; k < TILE_SIZE; ++k) {
        if (k < pivotCount) {
            const unsigned int sourceToPivot = tile[k][x];
#pragma unroll
            for (unsigned int item = 0; item < VALUES_PER_THREAD; ++item) {
                const unsigned int y = threadRow + item * THREAD_ROWS;
                const unsigned int candidate = sourceToPivot + tile[y][k];
                if (candidate < values[item]) {
                    values[item] = candidate;
                    tile[y][x] = candidate;
                    lastPivot[item] = static_cast<unsigned int>(base) + k;
                    changedMask |= 1U << item;
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (unsigned int item = 0; item < VALUES_PER_THREAD; ++item) {
        const unsigned int y = threadRow + item * THREAD_ROWS;
        const size_t destination = base + y;
        if (source < numNodes && destination < numNodes) {
            const size_t index = destination * numNodes + source;
            dist[index] = values[item];
            if ((changedMask & (1U << item)) != 0) {
                path[index] = lastPivot[item];
            }
        }
    }
}

// Update the tiles sharing either the source or destination dimension with the
// diagonal tile. blockIdx.y selects which of those two sets is processed.
__global__ __launch_bounds__(256) void floydWarshallPivotTiles(
    unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
    const size_t numNodes, const unsigned int round) {
    __shared__ unsigned int diagonal[TILE_SIZE][TILE_SIZE + 1];
    __shared__ unsigned int target[TILE_SIZE][TILE_SIZE + 1];

    const unsigned int x = threadIdx.x;
    const unsigned int threadRow = threadIdx.y;
    const unsigned int compactTile = blockIdx.x;
    const unsigned int otherTile = compactTile >= round ? compactTile + 1 : compactTile;
    const bool sourceIsPivot = blockIdx.y != 0;
    const unsigned int sourceTile = sourceIsPivot ? round : otherTile;
    const unsigned int destinationTile = sourceIsPivot ? otherTile : round;
    const size_t pivotBase = static_cast<size_t>(round) * TILE_SIZE;
    const size_t sourceBase = static_cast<size_t>(sourceTile) * TILE_SIZE;
    const size_t destinationBase = static_cast<size_t>(destinationTile) * TILE_SIZE;
    const size_t source = sourceBase + x;
    unsigned int values[VALUES_PER_THREAD];
    unsigned int lastPivot[VALUES_PER_THREAD];
    unsigned int changedMask = 0;

#pragma unroll
    for (unsigned int item = 0; item < VALUES_PER_THREAD; ++item) {
        const unsigned int y = threadRow + item * THREAD_ROWS;
        const size_t pivotSource = pivotBase + x;
        const size_t pivotDestination = pivotBase + y;
        const size_t destination = destinationBase + y;
        diagonal[y][x] = (pivotSource < numNodes && pivotDestination < numNodes)
                             ? dist[pivotDestination * numNodes + pivotSource]
                             : INF;
        values[item] = (source < numNodes && destination < numNodes)
                           ? dist[destination * numNodes + source]
                           : INF;
        target[y][x] = values[item];
    }
    __syncthreads();

    const unsigned int remaining = static_cast<unsigned int>(numNodes - pivotBase);
    const unsigned int pivotCount = min(TILE_SIZE, remaining);
#pragma unroll
    for (unsigned int k = 0; k < TILE_SIZE; ++k) {
        if (k < pivotCount) {
            const unsigned int firstHalf = sourceIsPivot ? diagonal[k][x] : target[k][x];
#pragma unroll
            for (unsigned int item = 0; item < VALUES_PER_THREAD; ++item) {
                const unsigned int y = threadRow + item * THREAD_ROWS;
                const unsigned int secondHalf =
                    sourceIsPivot ? target[y][k] : diagonal[y][k];
                const unsigned int candidate = firstHalf + secondHalf;
                if (candidate < values[item]) {
                    values[item] = candidate;
                    target[y][x] = candidate;
                    lastPivot[item] = static_cast<unsigned int>(pivotBase) + k;
                    changedMask |= 1U << item;
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (unsigned int item = 0; item < VALUES_PER_THREAD; ++item) {
        const unsigned int y = threadRow + item * THREAD_ROWS;
        const size_t destination = destinationBase + y;
        if (source < numNodes && destination < numNodes) {
            const size_t index = destination * numNodes + source;
            dist[index] = values[item];
            if ((changedMask & (1U << item)) != 0) {
                path[index] = lastPivot[item];
            }
        }
    }
}

// Update every tile that does not share a dimension with the diagonal tile.
// The two required pivot tiles remain resident in shared memory while each
// thread updates four destination rows in registers.
__global__ __launch_bounds__(256) void floydWarshallRemainingTiles(
    unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
    const size_t numNodes, const unsigned int round) {
    __shared__ unsigned int sourceToPivot[TILE_SIZE][TILE_SIZE + 1];
    __shared__ unsigned int pivotToDestination[TILE_SIZE][TILE_SIZE + 1];

    const unsigned int x = threadIdx.x;
    const unsigned int threadRow = threadIdx.y;
    const unsigned int compactSourceTile = blockIdx.x;
    const unsigned int compactDestinationTile = blockIdx.y;
    const unsigned int sourceTile =
        compactSourceTile >= round ? compactSourceTile + 1 : compactSourceTile;
    const unsigned int destinationTile = compactDestinationTile >= round
                                             ? compactDestinationTile + 1
                                             : compactDestinationTile;
    const size_t pivotBase = static_cast<size_t>(round) * TILE_SIZE;
    const size_t sourceBase = static_cast<size_t>(sourceTile) * TILE_SIZE;
    const size_t destinationBase = static_cast<size_t>(destinationTile) * TILE_SIZE;
    const size_t source = sourceBase + x;
    unsigned int values[VALUES_PER_THREAD];
    unsigned int lastPivot[VALUES_PER_THREAD];
    unsigned int changedMask = 0;

#pragma unroll
    for (unsigned int item = 0; item < VALUES_PER_THREAD; ++item) {
        const unsigned int y = threadRow + item * THREAD_ROWS;
        const size_t destination = destinationBase + y;
        const size_t pivotSource = pivotBase + x;
        const size_t pivotDestination = pivotBase + y;
        sourceToPivot[y][x] = (source < numNodes && pivotDestination < numNodes)
                                  ? dist[pivotDestination * numNodes + source]
                                  : INF;
        pivotToDestination[y][x] = (pivotSource < numNodes && destination < numNodes)
                                       ? dist[destination * numNodes + pivotSource]
                                       : INF;
        values[item] = (source < numNodes && destination < numNodes)
                           ? dist[destination * numNodes + source]
                           : INF;
    }
    __syncthreads();

    const unsigned int remaining = static_cast<unsigned int>(numNodes - pivotBase);
    const unsigned int pivotCount = min(TILE_SIZE, remaining);
#pragma unroll
    for (unsigned int k = 0; k < TILE_SIZE; ++k) {
        if (k < pivotCount) {
            const unsigned int firstHalf = sourceToPivot[k][x];
#pragma unroll
            for (unsigned int item = 0; item < VALUES_PER_THREAD; ++item) {
                const unsigned int y = threadRow + item * THREAD_ROWS;
                const unsigned int candidate = firstHalf + pivotToDestination[y][k];
                if (candidate < values[item]) {
                    values[item] = candidate;
                    lastPivot[item] = static_cast<unsigned int>(pivotBase) + k;
                    changedMask |= 1U << item;
                }
            }
        }
    }

#pragma unroll
    for (unsigned int item = 0; item < VALUES_PER_THREAD; ++item) {
        const unsigned int y = threadRow + item * THREAD_ROWS;
        const size_t destination = destinationBase + y;
        if (source < numNodes && destination < numNodes) {
            const size_t index = destination * numNodes + source;
            dist[index] = values[item];
            if ((changedMask & (1U << item)) != 0) {
                path[index] = lastPivot[item];
            }
        }
    }
}

// Force CUDA's lazy module loader to materialize every kernel before benchmark
// timing begins. A fixed two-tile scratch problem keeps this cost independent
// of the requested graph size.
void warmUpCudaKernels() {
    constexpr size_t warmupNodes = 2 * TILE_SIZE;
    constexpr size_t warmupBytes = warmupNodes * warmupNodes * sizeof(unsigned int);
    unsigned int* scratchDist = nullptr;
    unsigned int* scratchPath = nullptr;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&scratchDist), warmupBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&scratchPath), warmupBytes));
    CUDA_CHECK(cudaMemset(scratchDist, 0, warmupBytes));
    CUDA_CHECK(cudaMemset(scratchPath, 0, warmupBytes));

    const dim3 threads(TILE_SIZE, THREAD_ROWS);
    floydWarshallDiagonal<<<1, threads>>>(scratchDist, scratchPath, warmupNodes, 0);
    const dim3 pivotGrid(1, 2);
    floydWarshallPivotTiles<<<pivotGrid, threads>>>(scratchDist, scratchPath, warmupNodes, 0);
    const dim3 remainingGrid(1, 1);
    floydWarshallRemainingTiles<<<remainingGrid, threads>>>(
        scratchDist, scratchPath, warmupNodes, 0);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(scratchDist));
    CUDA_CHECK(cudaFree(scratchPath));
}

}  // namespace

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

float floydWarshall(std::vector<unsigned int>& dist,
                    std::vector<unsigned int>& path,
                    const size_t numNodes) {
    if (numNodes == 0) {
        return 0.0F;
    }

    warmUpCudaKernels();

    const size_t matrixElements = numNodes * numNodes;
    const size_t matrixBytes = matrixElements * sizeof(unsigned int);
    unsigned int* deviceDist = nullptr;
    unsigned int* devicePath = nullptr;
    cudaEvent_t startEvent = nullptr;
    cudaEvent_t stopEvent = nullptr;
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graphInstance = nullptr;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceDist), matrixBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&devicePath), matrixBytes));
    CUDA_CHECK(cudaMemcpy(deviceDist, dist.data(), matrixBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(devicePath, path.data(), matrixBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventCreate(&startEvent));
    CUDA_CHECK(cudaEventCreate(&stopEvent));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    const unsigned int tileCount =
        static_cast<unsigned int>((numNodes + TILE_SIZE - 1) / TILE_SIZE);
    const dim3 threads(TILE_SIZE, THREAD_ROWS);

    // Capture all rounds once so short diagonal and pivot phases do not incur
    // CPU launch gaps between dependent kernels.
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    for (unsigned int round = 0; round < tileCount; ++round) {
        floydWarshallDiagonal<<<1, threads, 0, stream>>>(
            deviceDist, devicePath, numNodes, round);
        CUDA_CHECK(cudaGetLastError());

        if (tileCount > 1) {
            const dim3 pivotGrid(tileCount - 1, 2);
            floydWarshallPivotTiles<<<pivotGrid, threads, 0, stream>>>(
                deviceDist, devicePath, numNodes, round);
            CUDA_CHECK(cudaGetLastError());

            const dim3 remainingGrid(tileCount - 1, tileCount - 1);
            floydWarshallRemainingTiles<<<remainingGrid, threads, 0, stream>>>(
                deviceDist, devicePath, numNodes, round);
            CUDA_CHECK(cudaGetLastError());
        }
    }
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&graphInstance, graph, nullptr, nullptr, 0));
    CUDA_CHECK(cudaGraphUpload(graphInstance, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaEventRecord(startEvent, stream));
    CUDA_CHECK(cudaGraphLaunch(graphInstance, stream));
    CUDA_CHECK(cudaEventRecord(stopEvent, stream));
    CUDA_CHECK(cudaEventSynchronize(stopEvent));

    float milliseconds = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, startEvent, stopEvent));
    CUDA_CHECK(cudaMemcpy(dist.data(), deviceDist, matrixBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(path.data(), devicePath, matrixBytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaGraphExecDestroy(graphInstance));
    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaEventDestroy(startEvent));
    CUDA_CHECK(cudaEventDestroy(stopEvent));
    CUDA_CHECK(cudaFree(deviceDist));
    CUDA_CHECK(cudaFree(devicePath));
    return milliseconds;
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
    const float milliseconds = floydWarshall(dist, path, numNodes);

    printf("Computation time: %.3f ms\n", static_cast<double>(milliseconds));
    
    // Calculate operations per second
    // Floyd-Warshall has O(n³) complexity
    double ops = (double)numNodes * numNodes * numNodes;
    const double seconds = static_cast<double>(milliseconds) / 1000.0;
    const double gflops = seconds > 0.0 ? ops / seconds / 1e9 : 0.0;
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
