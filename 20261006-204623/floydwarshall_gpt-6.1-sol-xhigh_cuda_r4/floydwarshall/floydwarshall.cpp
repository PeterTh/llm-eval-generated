#include <algorithm>
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

namespace {

constexpr int TILE = 32;
constexpr int BLOCK_ROWS = 8;
constexpr int ROWS_PER_THREAD = TILE / BLOCK_ROWS;

void checkCuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}

// rank = 1 + the smallest possible largest intermediate vertex on a shortest
// path; zero denotes a direct edge. Minimizing (distance, rank) preserves the
// original ascending-k, strict-improvement path matrix even with tiled updates.
__device__ __forceinline__ void relax(unsigned int& distance, unsigned int& rank,
                                     unsigned int candidate, unsigned int candidateRank) {
    if (candidate < distance || (candidate == distance && candidateRank < rank)) {
        distance = candidate;
        rank = candidateRank;
    }
}

__global__ void closePivot(unsigned int* dist, unsigned int* ranks,
                           size_t n, size_t pivot) {
    __shared__ unsigned int d[TILE][TILE];
    __shared__ unsigned int p[TILE][TILE];
    const int column = threadIdx.x;
    const size_t globalColumn = pivot * TILE + column;
#pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; ++r) {
        const int row = threadIdx.y + r * BLOCK_ROWS;
        const size_t globalRow = pivot * TILE + row;
        const bool valid = globalRow < n && globalColumn < n;
        d[row][column] = valid ? dist[globalRow * n + globalColumn] : INF;
        p[row][column] = valid ? ranks[globalRow * n + globalColumn] : 0;
    }
    __syncthreads();

    for (int k = 0; k < TILE; ++k) {
        const unsigned int pivotRank = static_cast<unsigned int>(pivot * TILE + k + 1);
#pragma unroll
        for (int r = 0; r < ROWS_PER_THREAD; ++r) {
            const int row = threadIdx.y + r * BLOCK_ROWS;
            // The zero diagonal makes the pivot row and column immutable in
            // this iteration. Skipping their stores also avoids shared races.
            if (row != k && column != k) {
                relax(d[row][column], p[row][column], d[row][k] + d[k][column],
                      max(max(p[row][k], p[k][column]), pivotRank));
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; ++r) {
        const int row = threadIdx.y + r * BLOCK_ROWS;
        const size_t globalRow = pivot * TILE + row;
        if (globalRow < n && globalColumn < n) {
            dist[globalRow * n + globalColumn] = d[row][column];
            ranks[globalRow * n + globalColumn] = p[row][column];
        }
    }
}

__global__ void updatePivotPanels(unsigned int* dist, unsigned int* ranks,
                                  size_t n, size_t pivot) {
    __shared__ unsigned int pivotD[TILE][TILE];
    __shared__ unsigned int pivotP[TILE][TILE];
    __shared__ unsigned int d[TILE][TILE];
    __shared__ unsigned int p[TILE][TILE];
    const bool rowPanel = blockIdx.y == 0;
    const size_t other = blockIdx.x + (blockIdx.x >= pivot);
    const size_t tileRow = rowPanel ? pivot : other;
    const size_t tileColumn = rowPanel ? other : pivot;
    const int column = threadIdx.x;
#pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; ++r) {
        const int row = threadIdx.y + r * BLOCK_ROWS;
        const size_t pivotRow = pivot * TILE + row;
        const size_t pivotColumn = pivot * TILE + column;
        const size_t globalRow = tileRow * TILE + row;
        const size_t globalColumn = tileColumn * TILE + column;
        const bool pivotValid = pivotRow < n && pivotColumn < n;
        pivotD[row][column] = pivotValid ? dist[pivotRow * n + pivotColumn] : INF;
        // Include the connecting vertex once on load rather than in every
        // relaxation. These augmented ranks are used only as operands.
        pivotP[row][column] = max(pivotValid ? ranks[pivotRow * n + pivotColumn] : 0,
            static_cast<unsigned int>(rowPanel ? pivotColumn + 1 : pivotRow + 1));
        const bool valid = globalRow < n && globalColumn < n;
        d[row][column] = valid ? dist[globalRow * n + globalColumn] : INF;
        p[row][column] = valid ? ranks[globalRow * n + globalColumn] : 0;
    }
    __syncthreads();

    for (int k = 0; k < TILE; ++k) {
#pragma unroll
        for (int r = 0; r < ROWS_PER_THREAD; ++r) {
            const int row = threadIdx.y + r * BLOCK_ROWS;
            if (rowPanel) {
                if (row != k) {
                    relax(d[row][column], p[row][column], pivotD[row][k] + d[k][column],
                          max(pivotP[row][k], p[k][column]));
                }
            } else if (column != k) {
                relax(d[row][column], p[row][column], d[row][k] + pivotD[k][column],
                      max(p[row][k], pivotP[k][column]));
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; ++r) {
        const int row = threadIdx.y + r * BLOCK_ROWS;
        const size_t globalRow = tileRow * TILE + row;
        const size_t globalColumn = tileColumn * TILE + column;
        if (globalRow < n && globalColumn < n) {
            dist[globalRow * n + globalColumn] = d[row][column];
            ranks[globalRow * n + globalColumn] = p[row][column];
        }
    }
}

__global__ void updateRemainingTiles(unsigned int* __restrict__ dist,
                                     unsigned int* __restrict__ ranks,
                                     size_t n, size_t pivot) {
    __shared__ unsigned int leftD[TILE][TILE];
    __shared__ unsigned int leftP[TILE][TILE];
    __shared__ unsigned int rightD[TILE][TILE];
    __shared__ unsigned int rightP[TILE][TILE];
    const size_t tileRow = blockIdx.y + (blockIdx.y >= pivot);
    const size_t tileColumn = blockIdx.x + (blockIdx.x >= pivot);
    const int column = threadIdx.x;
    const size_t globalColumn = tileColumn * TILE + column;
    unsigned int distances[ROWS_PER_THREAD];
    unsigned int pathRanks[ROWS_PER_THREAD];
#pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; ++r) {
        const int row = threadIdx.y + r * BLOCK_ROWS;
        const size_t globalRow = tileRow * TILE + row;
        const size_t pivotRow = pivot * TILE + row;
        const size_t pivotColumn = pivot * TILE + column;
        const bool leftValid = globalRow < n && pivotColumn < n;
        const bool rightValid = pivotRow < n && globalColumn < n;
        leftD[row][column] = leftValid ? dist[globalRow * n + pivotColumn] : INF;
        leftP[row][column] = max(leftValid ? ranks[globalRow * n + pivotColumn] : 0,
                                static_cast<unsigned int>(pivotColumn + 1));
        rightD[row][column] = rightValid ? dist[pivotRow * n + globalColumn] : INF;
        rightP[row][column] = rightValid ? ranks[pivotRow * n + globalColumn] : 0;
        const bool valid = globalRow < n && globalColumn < n;
        distances[r] = valid ? dist[globalRow * n + globalColumn] : INF;
        pathRanks[r] = valid ? ranks[globalRow * n + globalColumn] : 0;
    }
    __syncthreads();

    // Each thread keeps four outputs in registers. The right operand is reused
    // across those rows, and a warp broadcasts each shared left operand.
#pragma unroll 4
    for (int k = 0; k < TILE; ++k) {
        const unsigned int rightDistance = rightD[k][column];
        const unsigned int rightRank = rightP[k][column];
#pragma unroll
        for (int r = 0; r < ROWS_PER_THREAD; ++r) {
            const int row = threadIdx.y + r * BLOCK_ROWS;
            relax(distances[r], pathRanks[r], leftD[row][k] + rightDistance,
                  max(leftP[row][k], rightRank));
        }
    }

#pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; ++r) {
        const size_t globalRow = tileRow * TILE + threadIdx.y + r * BLOCK_ROWS;
        if (globalRow < n && globalColumn < n) {
            dist[globalRow * n + globalColumn] = distances[r];
            ranks[globalRow * n + globalColumn] = pathRanks[r];
        }
    }
}

__global__ void decodePaths(unsigned int* ranks, size_t n, size_t elements) {
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < elements; index += static_cast<size_t>(blockDim.x) * gridDim.x) {
        const unsigned int rank = ranks[index];
        ranks[index] = rank == 0 ? static_cast<unsigned int>(index / n) : rank - 1;
    }
}

} // namespace

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes) {
    if (numNodes == 0) {
        return;
    }
    const size_t bytes = dist.size() * sizeof(unsigned int);
    unsigned int* deviceDist = nullptr;
    unsigned int* deviceRanks = nullptr;
    cudaStream_t stream;
    checkCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "creating stream");
    checkCuda(cudaMalloc(&deviceDist, bytes), "allocating distances");
    checkCuda(cudaMalloc(&deviceRanks, bytes), "allocating path ranks");
    checkCuda(cudaMemcpyAsync(deviceDist, dist.data(), bytes, cudaMemcpyHostToDevice, stream),
              "uploading distances");
    checkCuda(cudaMemsetAsync(deviceRanks, 0, bytes, stream), "initializing path ranks");

    const unsigned int tiles = static_cast<unsigned int>((numNodes + TILE - 1) / TILE);
    const dim3 threads(TILE, BLOCK_ROWS);
    for (unsigned int pivot = 0; pivot < tiles; ++pivot) {
        closePivot<<<1, threads, 0, stream>>>(deviceDist, deviceRanks, numNodes, pivot);
        checkCuda(cudaGetLastError(), "launching pivot closure");
        if (tiles > 1) {
            updatePivotPanels<<<dim3(tiles - 1, 2), threads, 0, stream>>>(
                deviceDist, deviceRanks, numNodes, pivot);
            checkCuda(cudaGetLastError(), "launching pivot panels");
            updateRemainingTiles<<<dim3(tiles - 1, tiles - 1), threads, 0, stream>>>(
                deviceDist, deviceRanks, numNodes, pivot);
            checkCuda(cudaGetLastError(), "launching remaining tiles");
        }
    }
    // Launch order in one stream is the global barrier between the three phases.
    const unsigned int decodeBlocks = static_cast<unsigned int>(
        std::min((dist.size() + 255) / 256, static_cast<size_t>(65535)));
    decodePaths<<<decodeBlocks, 256, 0, stream>>>(deviceRanks, numNodes, dist.size());
    checkCuda(cudaGetLastError(), "launching path decoding");
    checkCuda(cudaMemcpyAsync(dist.data(), deviceDist, bytes, cudaMemcpyDeviceToHost, stream),
              "downloading distances");
    checkCuda(cudaMemcpyAsync(path.data(), deviceRanks, bytes, cudaMemcpyDeviceToHost, stream),
              "downloading paths");
    checkCuda(cudaStreamSynchronize(stream), "finishing shortest paths");
    checkCuda(cudaFree(deviceRanks), "freeing path ranks");
    checkCuda(cudaFree(deviceDist), "freeing distances");
    checkCuda(cudaStreamDestroy(stream), "destroying stream");
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
            const char* value = argv[++i];
            char* end = nullptr;
            const unsigned long long parsed = strtoull(value, &end, 10);
            if (*value == '\0' || *value == '-' || *end != '\0' ||
                parsed > std::numeric_limits<unsigned int>::max() ||
                (parsed != 0 && parsed > std::numeric_limits<size_t>::max() /
                                           sizeof(unsigned int) / parsed)) {
                fprintf(stderr, "Invalid number of nodes: %s\n", value);
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
    
    // Allocate matrices
    std::vector<unsigned int> dist(numNodes * numNodes);
    std::vector<unsigned int> path(numNodes * numNodes);
    
    // Initialize
    printf("Initializing graph...\n");
    initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
    initializePathMatrix(path, numNodes);
    
    // Run Floyd-Warshall
    printf("Computing shortest paths...\n");
    // Initialize the CUDA context before measuring the actual computation.
    checkCuda(cudaFree(nullptr), "initializing CUDA");
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
