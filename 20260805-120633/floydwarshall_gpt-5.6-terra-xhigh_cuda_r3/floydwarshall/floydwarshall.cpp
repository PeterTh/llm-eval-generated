#include <algorithm>
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
constexpr unsigned int TILE_SIZE = 32;
constexpr unsigned int THREAD_ROWS = 8;
constexpr unsigned int OUTPUTS_PER_THREAD = TILE_SIZE / THREAD_ROWS;

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        const cudaError_t error_ = (call);                                     \
        if (error_ != cudaSuccess) {                                           \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,       \
                         __LINE__, cudaGetErrorString(error_));                \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                       \
    } while (false)

namespace {

// Matrices are stored as dist[destination * stride + source].  Threads vary
// source along x so every global-memory matrix load/store is coalesced.
__global__ void floydPivotKernel(unsigned int* __restrict__ dist,
                                 unsigned int* __restrict__ path,
                                 const size_t stride,
                                 const unsigned int pivotTile) {
    __shared__ unsigned int tile[TILE_SIZE][TILE_SIZE + 1];

    const unsigned int source = pivotTile * TILE_SIZE + threadIdx.x;
    const unsigned int destinationBase = pivotTile * TILE_SIZE + threadIdx.y;
    unsigned int pathValue[OUTPUTS_PER_THREAD];

    #pragma unroll
    for (unsigned int output = 0; output < OUTPUTS_PER_THREAD; ++output) {
        const unsigned int destination = destinationBase + output * THREAD_ROWS;
        const size_t element = static_cast<size_t>(destination) * stride + source;
        tile[threadIdx.y + output * THREAD_ROWS][threadIdx.x] = dist[element];
        pathValue[output] = path[element];
    }
    __syncthreads();

    #pragma unroll
    for (unsigned int k = 0; k < TILE_SIZE; ++k) {
        const unsigned int left = tile[k][threadIdx.x];
        #pragma unroll
        for (unsigned int output = 0; output < OUTPUTS_PER_THREAD; ++output) {
            const unsigned int row = threadIdx.y + output * THREAD_ROWS;
            const unsigned int candidate = left + tile[row][k];
            if (candidate < tile[row][threadIdx.x]) {
                tile[row][threadIdx.x] = candidate;
                pathValue[output] = pivotTile * TILE_SIZE + k;
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (unsigned int output = 0; output < OUTPUTS_PER_THREAD; ++output) {
        const unsigned int destination = destinationBase + output * THREAD_ROWS;
        const size_t element = static_cast<size_t>(destination) * stride + source;
        dist[element] = tile[threadIdx.y + output * THREAD_ROWS][threadIdx.x];
        path[element] = pathValue[output];
    }
}

// Updates both the pivot column and pivot row.  The second grid dimension
// selects D(other, pivot) or D(pivot, other), respectively.
__global__ void floydPivotRowColumnKernel(unsigned int* __restrict__ dist,
                                          unsigned int* __restrict__ path,
                                          const size_t stride,
                                          const unsigned int pivotTile) {
    __shared__ unsigned int pivot[TILE_SIZE][TILE_SIZE + 1];
    __shared__ unsigned int current[TILE_SIZE][TILE_SIZE + 1];

    unsigned int otherTile = blockIdx.x;
    if (otherTile >= pivotTile) {
        ++otherTile;
    }
    const bool pivotRow = blockIdx.y != 0;
    const unsigned int sourceTile = pivotRow ? pivotTile : otherTile;
    const unsigned int destinationTile = pivotRow ? otherTile : pivotTile;
    const unsigned int source = sourceTile * TILE_SIZE + threadIdx.x;
    const unsigned int destinationBase = destinationTile * TILE_SIZE + threadIdx.y;
    const unsigned int pivotSource = pivotTile * TILE_SIZE + threadIdx.x;
    const unsigned int pivotDestinationBase = pivotTile * TILE_SIZE + threadIdx.y;
    unsigned int pathValue[OUTPUTS_PER_THREAD];

    #pragma unroll
    for (unsigned int output = 0; output < OUTPUTS_PER_THREAD; ++output) {
        const unsigned int row = threadIdx.y + output * THREAD_ROWS;
        const unsigned int destination = destinationBase + output * THREAD_ROWS;
        const unsigned int pivotDestination = pivotDestinationBase + output * THREAD_ROWS;
        const size_t element = static_cast<size_t>(destination) * stride + source;
        const size_t pivotElement = static_cast<size_t>(pivotDestination) * stride + pivotSource;
        current[row][threadIdx.x] = dist[element];
        pivot[row][threadIdx.x] = dist[pivotElement];
        pathValue[output] = path[element];
    }
    __syncthreads();

    #pragma unroll
    for (unsigned int k = 0; k < TILE_SIZE; ++k) {
        const unsigned int left = pivotRow ? pivot[k][threadIdx.x]
                                           : current[k][threadIdx.x];
        #pragma unroll
        for (unsigned int output = 0; output < OUTPUTS_PER_THREAD; ++output) {
            const unsigned int row = threadIdx.y + output * THREAD_ROWS;
            const unsigned int right = pivotRow ? current[row][k] : pivot[row][k];
            const unsigned int candidate = left + right;
            if (candidate < current[row][threadIdx.x]) {
                current[row][threadIdx.x] = candidate;
                pathValue[output] = pivotTile * TILE_SIZE + k;
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (unsigned int output = 0; output < OUTPUTS_PER_THREAD; ++output) {
        const unsigned int destination = destinationBase + output * THREAD_ROWS;
        const size_t element = static_cast<size_t>(destination) * stride + source;
        dist[element] = current[threadIdx.y + output * THREAD_ROWS][threadIdx.x];
        path[element] = pathValue[output];
    }
}

// The remaining tiles are independent once the pivot row and column have
// been closed.  This phase exposes almost all of the benchmark's parallelism.
__global__ void floydRemainingKernel(unsigned int* __restrict__ dist,
                                     unsigned int* __restrict__ path,
                                     const size_t stride,
                                     const unsigned int pivotTile) {
    __shared__ unsigned int left[TILE_SIZE][TILE_SIZE + 1];
    __shared__ unsigned int right[TILE_SIZE][TILE_SIZE + 1];

    unsigned int sourceTile = blockIdx.x;
    unsigned int destinationTile = blockIdx.y;
    if (sourceTile >= pivotTile) {
        ++sourceTile;
    }
    if (destinationTile >= pivotTile) {
        ++destinationTile;
    }

    const unsigned int source = sourceTile * TILE_SIZE + threadIdx.x;
    const unsigned int destinationBase = destinationTile * TILE_SIZE + threadIdx.y;
    const unsigned int pivotSource = pivotTile * TILE_SIZE + threadIdx.x;
    const unsigned int pivotDestinationBase = pivotTile * TILE_SIZE + threadIdx.y;
    unsigned int distance[OUTPUTS_PER_THREAD];
    unsigned int pathValue[OUTPUTS_PER_THREAD];

    #pragma unroll
    for (unsigned int output = 0; output < OUTPUTS_PER_THREAD; ++output) {
        const unsigned int row = threadIdx.y + output * THREAD_ROWS;
        const unsigned int destination = destinationBase + output * THREAD_ROWS;
        const unsigned int pivotDestination = pivotDestinationBase + output * THREAD_ROWS;
        const size_t element = static_cast<size_t>(destination) * stride + source;
        const size_t leftElement = static_cast<size_t>(pivotDestination) * stride + source;
        const size_t rightElement = static_cast<size_t>(destination) * stride + pivotSource;
        left[row][threadIdx.x] = dist[leftElement];
        right[row][threadIdx.x] = dist[rightElement];
        distance[output] = dist[element];
        pathValue[output] = path[element];
    }
    __syncthreads();

    #pragma unroll
    for (unsigned int k = 0; k < TILE_SIZE; ++k) {
        const unsigned int leftDistance = left[k][threadIdx.x];
        #pragma unroll
        for (unsigned int output = 0; output < OUTPUTS_PER_THREAD; ++output) {
            const unsigned int row = threadIdx.y + output * THREAD_ROWS;
            const unsigned int candidate = leftDistance + right[row][k];
            if (candidate < distance[output]) {
                distance[output] = candidate;
                pathValue[output] = pivotTile * TILE_SIZE + k;
            }
        }
    }

    #pragma unroll
    for (unsigned int output = 0; output < OUTPUTS_PER_THREAD; ++output) {
        const unsigned int destination = destinationBase + output * THREAD_ROWS;
        const size_t element = static_cast<size_t>(destination) * stride + source;
        dist[element] = distance[output];
        path[element] = pathValue[output];
    }
}

float floydWarshallCuda(std::vector<unsigned int>& dist,
                        std::vector<unsigned int>& path,
                        const size_t numNodes) {
    if (numNodes == 0) {
        return 0.0F;
    }

    const size_t paddedNodes = ((numNodes + TILE_SIZE - 1) / TILE_SIZE) * TILE_SIZE;
    if (paddedNodes > std::numeric_limits<unsigned int>::max()) {
        std::fprintf(stderr, "CUDA Floyd-Warshall supports at most %u nodes\n",
                     std::numeric_limits<unsigned int>::max());
        std::exit(EXIT_FAILURE);
    }

    const size_t paddedElements = paddedNodes * paddedNodes;
    std::vector<unsigned int> paddedDist(paddedElements, INF);
    std::vector<unsigned int> paddedPath(paddedElements, 0);
    for (size_t destination = 0; destination < numNodes; ++destination) {
        std::copy_n(dist.data() + destination * numNodes, numNodes,
                    paddedDist.data() + destination * paddedNodes);
        std::copy_n(path.data() + destination * numNodes, numNodes,
                    paddedPath.data() + destination * paddedNodes);
    }

    unsigned int* deviceDist = nullptr;
    unsigned int* devicePath = nullptr;
    const size_t bytes = paddedElements * sizeof(unsigned int);
    CUDA_CHECK(cudaMalloc(&deviceDist, bytes));
    CUDA_CHECK(cudaMalloc(&devicePath, bytes));
    CUDA_CHECK(cudaMemcpy(deviceDist, paddedDist.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(devicePath, paddedPath.data(), bytes, cudaMemcpyHostToDevice));

    cudaEvent_t start{};
    cudaEvent_t stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const unsigned int tileCount = static_cast<unsigned int>(paddedNodes / TILE_SIZE);
    const dim3 threads(TILE_SIZE, THREAD_ROWS);
    CUDA_CHECK(cudaEventRecord(start));
    for (unsigned int pivotTile = 0; pivotTile < tileCount; ++pivotTile) {
        floydPivotKernel<<<1, threads>>>(deviceDist, devicePath, paddedNodes, pivotTile);
        CUDA_CHECK(cudaPeekAtLastError());

        if (tileCount > 1) {
            floydPivotRowColumnKernel<<<dim3(tileCount - 1, 2), threads>>>(
                deviceDist, devicePath, paddedNodes, pivotTile);
            CUDA_CHECK(cudaPeekAtLastError());
            floydRemainingKernel<<<dim3(tileCount - 1, tileCount - 1), threads>>>(
                deviceDist, devicePath, paddedNodes, pivotTile);
            CUDA_CHECK(cudaPeekAtLastError());
        }
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsedMilliseconds = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedMilliseconds, start, stop));

    CUDA_CHECK(cudaMemcpy2D(dist.data(), numNodes * sizeof(unsigned int),
                            deviceDist, paddedNodes * sizeof(unsigned int),
                            numNodes * sizeof(unsigned int), numNodes,
                            cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy2D(path.data(), numNodes * sizeof(unsigned int),
                            devicePath, paddedNodes * sizeof(unsigned int),
                            numNodes * sizeof(unsigned int), numNodes,
                            cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(deviceDist));
    CUDA_CHECK(cudaFree(devicePath));
    return elapsedMilliseconds;
}

}  // namespace

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
    const float durationMilliseconds = floydWarshallCuda(dist, path, numNodes);
    
    printf("Computation time: %.3f ms\n", durationMilliseconds);
    
    // Calculate operations per second
    // Floyd-Warshall has O(n³) complexity
    double ops = (double)numNodes * numNodes * numNodes;
    double gflops = durationMilliseconds > 0.0F
                        ? ops / (static_cast<double>(durationMilliseconds) / 1000.0) / 1e9
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
