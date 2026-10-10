#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
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
    // The original two symmetric writes both assign each row its source index.
    for (size_t row = 0; row < numNodes; ++row) {
        std::fill_n(path.data() + row * numNodes, numNodes, static_cast<unsigned int>(row));
    }
}

namespace {

constexpr int TILE = 32;
constexpr int THREAD_ROWS = 8;
constexpr int ROWS_PER_THREAD = TILE / THREAD_ROWS;

void checkCuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
    }
}

struct DeviceMatrices {
    unsigned int* data = nullptr;

    explicit DeviceMatrices(size_t bytes) {
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&data), bytes), "Allocating GPU matrices");
    }
    ~DeviceMatrices() { cudaFree(data); }
    DeviceMatrices(const DeviceMatrices&) = delete;
    DeviceMatrices& operator=(const DeviceMatrices&) = delete;
};

// Alongside each distance, keep the smallest possible largest intermediate
// vertex (encoded as k+1; zero means a direct edge). Comparing these witnesses
// on equal distances preserves the original ascending-k, strict-< path matrix
// even though blocked Floyd-Warshall visits candidates in a different order.
__device__ __forceinline__ void relax(unsigned int& distance, unsigned int& witness,
                                     unsigned int candidate, unsigned int candidateWitness) {
    const bool update = candidate < distance ||
                       (candidate == distance && candidateWitness < witness);
    witness = update ? candidateWitness : witness;
    distance = min(distance, candidate);
}

__device__ __forceinline__ void loadTile(const unsigned int* __restrict__ distances,
                                       const unsigned int* __restrict__ witnesses,
                                       unsigned int (&d)[TILE][TILE],
                                       unsigned int (&w)[TILE][TILE],
                                       size_t stride, int tileRow, int tileColumn) {
    const int column = threadIdx.x;
    #pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; ++r) {
        const int row = threadIdx.y + r * THREAD_ROWS;
        const size_t offset = (static_cast<size_t>(tileRow) * TILE + row) * stride
                            + static_cast<size_t>(tileColumn) * TILE + column;
        d[row][column] = distances[offset];
        w[row][column] = witnesses[offset];
    }
}

__device__ __forceinline__ void storeTile(unsigned int* __restrict__ distances,
                                        unsigned int* __restrict__ witnesses,
                                        const unsigned int (&d)[TILE][TILE],
                                        const unsigned int (&w)[TILE][TILE],
                                        size_t stride, int tileRow, int tileColumn) {
    const int column = threadIdx.x;
    #pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; ++r) {
        const int row = threadIdx.y + r * THREAD_ROWS;
        const size_t offset = (static_cast<size_t>(tileRow) * TILE + row) * stride
                            + static_cast<size_t>(tileColumn) * TILE + column;
        distances[offset] = d[row][column];
        witnesses[offset] = w[row][column];
    }
}

// Phase 1: close the pivot tile in shared memory.
__global__ void pivotTile(unsigned int* distances, unsigned int* witnesses,
                          size_t stride, int pivot, int count) {
    __shared__ unsigned int d[TILE][TILE];
    __shared__ unsigned int w[TILE][TILE];
    loadTile(distances, witnesses, d, w, stride, pivot, pivot);
    __syncthreads();
    const int column = threadIdx.x;
    for (int k = 0; k < count; ++k) {
        const unsigned int vertex = static_cast<unsigned int>(pivot) * TILE + k + 1;
        #pragma unroll
        for (int r = 0; r < ROWS_PER_THREAD; ++r) {
            const int row = threadIdx.y + r * THREAD_ROWS;
            // With nonnegative weights and a zero diagonal the pivot row and
            // column cannot improve. Leave them read-only to avoid shared races.
            if (row != k && column != k) {
                const unsigned int candidate = d[row][k] + d[k][column];
                const unsigned int candidateWitness = max(vertex, max(w[row][k], w[k][column]));
                relax(d[row][column], w[row][column], candidate, candidateWitness);
            }
        }
        __syncthreads();
    }
    storeTile(distances, witnesses, d, w, stride, pivot, pivot);
}

// Phase 2: update every tile in the pivot's row and column concurrently.
__global__ void pivotPanels(unsigned int* distances, unsigned int* witnesses,
                            size_t stride, int pivot, int count) {
    __shared__ unsigned int pd[TILE][TILE], pw[TILE][TILE];
    __shared__ unsigned int d[TILE][TILE], w[TILE][TILE];
    const int other = blockIdx.x + (blockIdx.x >= static_cast<unsigned int>(pivot));
    const bool rowPanel = blockIdx.y == 0;
    const int tileRow = rowPanel ? pivot : other;
    const int tileColumn = rowPanel ? other : pivot;
    loadTile(distances, witnesses, pd, pw, stride, pivot, pivot);
    loadTile(distances, witnesses, d, w, stride, tileRow, tileColumn);
    __syncthreads();
    const int column = threadIdx.x;
    for (int k = 0; k < count; ++k) {
        const unsigned int vertex = static_cast<unsigned int>(pivot) * TILE + k + 1;
        #pragma unroll
        for (int r = 0; r < ROWS_PER_THREAD; ++r) {
            const int row = threadIdx.y + r * THREAD_ROWS;
            if (rowPanel ? row != k : column != k) {
                const unsigned int left = rowPanel ? pd[row][k] : d[row][k];
                const unsigned int right = rowPanel ? d[k][column] : pd[k][column];
                const unsigned int leftWitness = rowPanel ? pw[row][k] : w[row][k];
                const unsigned int rightWitness = rowPanel ? w[k][column] : pw[k][column];
                relax(d[row][column], w[row][column], left + right,
                      max(vertex, max(leftWitness, rightWitness)));
            }
        }
        __syncthreads();
    }
    storeTile(distances, witnesses, d, w, stride, tileRow, tileColumn);
}

// Phase 3: independent output tiles reuse the closed panels, with each thread
// accumulating four outputs in registers and no barriers inside the k loop.
__global__ void remainingTiles(unsigned int* __restrict__ distances,
                               unsigned int* __restrict__ witnesses,
                               size_t stride, int pivot, int count) {
    __shared__ unsigned int ld[TILE][TILE], lw[TILE][TILE];
    __shared__ unsigned int rd[TILE][TILE], rw[TILE][TILE];
    const int tileColumn = blockIdx.x + (blockIdx.x >= static_cast<unsigned int>(pivot));
    const int tileRow = blockIdx.y + (blockIdx.y >= static_cast<unsigned int>(pivot));
    loadTile(distances, witnesses, ld, lw, stride, tileRow, pivot);
    loadTile(distances, witnesses, rd, rw, stride, pivot, tileColumn);
    unsigned int d[ROWS_PER_THREAD], w[ROWS_PER_THREAD];
    const int column = threadIdx.x;
    #pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; ++r) {
        const size_t row = static_cast<size_t>(tileRow) * TILE + threadIdx.y + r * THREAD_ROWS;
        const size_t offset = row * stride + static_cast<size_t>(tileColumn) * TILE + column;
        d[r] = distances[offset];
        w[r] = witnesses[offset];
    }
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < TILE; ++k) {
        if (k < count) {
            const unsigned int right = rd[k][column];
            const unsigned int rightWitness = max(rw[k][column],
                static_cast<unsigned int>(pivot) * TILE + k + 1);
            #pragma unroll
            for (int r = 0; r < ROWS_PER_THREAD; ++r) {
                const int row = threadIdx.y + r * THREAD_ROWS;
                relax(d[r], w[r], ld[row][k] + right, max(lw[row][k], rightWitness));
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; ++r) {
        const size_t row = static_cast<size_t>(tileRow) * TILE + threadIdx.y + r * THREAD_ROWS;
        const size_t offset = row * stride + static_cast<size_t>(tileColumn) * TILE + column;
        distances[offset] = d[r];
        witnesses[offset] = w[r];
    }
}

} // namespace

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes) {
    // Initialize CUDA even for an empty graph; there is no CPU fallback.
    checkCuda(cudaFree(nullptr), "Initializing CUDA");
    if (numNodes == 0) {
        return;
    }
    const size_t tiles = (numNodes + TILE - 1) / TILE;
    const size_t stride = tiles * TILE;
    if (stride > std::numeric_limits<size_t>::max() / stride / (2 * sizeof(unsigned int))) {
        throw std::runtime_error("Graph is too large for GPU matrix storage");
    }
    const size_t bytes = stride * stride * sizeof(unsigned int);
    DeviceMatrices allocation(2 * bytes);
    unsigned int* deviceDistances = allocation.data;
    unsigned int* deviceWitnesses = allocation.data + stride * stride;
    // The padded value exceeds every benchmark edge weight and its sum with
    // itself fits in unsigned int. Padded vertices are never used as pivots.
    checkCuda(cudaMemset(deviceDistances, 0x3f, bytes), "Padding GPU distances");
    checkCuda(cudaMemset(deviceWitnesses, 0, bytes), "Initializing GPU witnesses");
    const size_t rowBytes = numNodes * sizeof(unsigned int);
    checkCuda(cudaMemcpy2D(deviceDistances, stride * sizeof(unsigned int),
                           dist.data(), rowBytes, rowBytes, numNodes,
                           cudaMemcpyHostToDevice), "Uploading distances");

    const dim3 threads(TILE, THREAD_ROWS);
    for (size_t pivot = 0; pivot < tiles; ++pivot) {
        const int count = static_cast<int>(std::min(static_cast<size_t>(TILE), numNodes - pivot * TILE));
        pivotTile<<<1, threads>>>(deviceDistances, deviceWitnesses, stride, static_cast<int>(pivot), count);
        checkCuda(cudaGetLastError(), "Launching pivot tile kernel");
        if (tiles > 1) {
            pivotPanels<<<dim3(static_cast<unsigned int>(tiles - 1), 2), threads>>>(
                deviceDistances, deviceWitnesses, stride, static_cast<int>(pivot), count);
            checkCuda(cudaGetLastError(), "Launching pivot panel kernel");
            remainingTiles<<<dim3(static_cast<unsigned int>(tiles - 1), static_cast<unsigned int>(tiles - 1)), threads>>>(
                deviceDistances, deviceWitnesses, stride, static_cast<int>(pivot), count);
            checkCuda(cudaGetLastError(), "Launching remaining tile kernel");
        }
    }
    checkCuda(cudaMemcpy2D(dist.data(), rowBytes, deviceDistances,
                           stride * sizeof(unsigned int), rowBytes, numNodes,
                           cudaMemcpyDeviceToHost), "Downloading distances");
    // Preserve caller-provided initial path values for edges that never improve.
    std::vector<unsigned int> initialPath = std::move(path);
    path.resize(numNodes * numNodes);
    checkCuda(cudaMemcpy2D(path.data(), rowBytes, deviceWitnesses,
                           stride * sizeof(unsigned int), rowBytes, numNodes,
                           cudaMemcpyDeviceToHost), "Downloading path witnesses");
    for (size_t i = 0; i < path.size(); ++i) {
        path[i] = path[i] == 0 ? initialPath[i] : path[i] - 1;
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

int main(int argc, char** argv) try {
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

    if (numNodes > std::numeric_limits<unsigned int>::max() - 1 ||
        (numNodes != 0 && numNodes > std::numeric_limits<size_t>::max() / numNodes / sizeof(unsigned int))) {
        fprintf(stderr, "Number of nodes is too large\n");
        return 1;
    }

    // CUDA context creation is setup, like the host allocations below. All GPU
    // allocation, matrix transfers, computation and completion remain timed.
    checkCuda(cudaFree(nullptr), "Initializing CUDA");
    
    // Allocate matrices
    std::vector<unsigned int> dist(numNodes * numNodes);
    std::vector<unsigned int> path(numNodes * numNodes);
    
    // Initialize
    printf("Initializing graph...\n");
    initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
    initializePathMatrix(path, numNodes);
    
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
} catch (const std::exception& error) {
    fprintf(stderr, "Floyd-Warshall failed: %s\n", error.what());
    return 1;
}
