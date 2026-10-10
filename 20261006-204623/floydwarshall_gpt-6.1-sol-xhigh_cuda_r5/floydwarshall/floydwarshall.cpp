#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
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
        std::fill_n(path.data() + j * numNodes, numNodes, static_cast<unsigned int>(j));
    }
}

namespace {

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

struct DeviceStorage {
    unsigned int* data = nullptr;
    explicit DeviceStorage(size_t bytes) {
        checkCuda(cudaMalloc(&data, bytes), "Allocate GPU matrices");
    }
    ~DeviceStorage() { cudaFree(data); }
    DeviceStorage(const DeviceStorage&) = delete;
    DeviceStorage& operator=(const DeviceStorage&) = delete;
};

struct Stream {
    cudaStream_t value = nullptr;
    Stream() { checkCuda(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking), "Create CUDA stream"); }
    ~Stream() { cudaStreamDestroy(value); }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
};

// Blocking reorders intermediate vertices. To retain the serial algorithm's
// strict-improvement path semantics, compare (distance, largest intermediate)
// lexicographically. Among equal shortest routes, serial ascending-k traversal
// keeps the route whose largest intermediate is smallest. A tag of zero denotes
// a direct edge; intermediate k has tag k+1. These tags compose with max().
__device__ __forceinline__ bool relax(unsigned int& distance, unsigned int& via,
                                     unsigned int leftDistance, unsigned int rightDistance,
                                     unsigned int leftVia, unsigned int rightVia,
                                     unsigned int intermediate) {
    const unsigned int candidate = leftDistance + rightDistance;
    if (candidate <= distance) {
        const unsigned int candidateVia = max(intermediate, max(leftVia, rightVia));
        if (candidate < distance || candidateVia < via) {
            distance = candidate;
            via = candidateVia;
            return true;
        }
    }
    return false;
}

__global__ void prepareMatrices(unsigned int* distance, unsigned int* via,
                                size_t pitch, size_t nodes) {
    const size_t count = pitch * pitch;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<size_t>(blockDim.x) * gridDim.x) {
        via[index] = 0;
        const size_t row = index / pitch;
        const size_t column = index % pitch;
        if (row >= nodes || column >= nodes) {
            distance[index] = row == column ? 0 : INF;
        }
    }
}

__global__ void decodePaths(unsigned int* via, size_t pitch) {
    const size_t count = pitch * pitch;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<size_t>(blockDim.x) * gridDim.x) {
        const unsigned int tag = via[index];
        // initializePathMatrix assigns the source row to direct edges.
        via[index] = tag == 0 ? static_cast<unsigned int>(index / pitch) : tag - 1;
    }
}

// Each warp accesses consecutive columns, and each thread owns several rows.
// Separate shared arrays avoid the bank conflicts of interleaved distance/tag
// pairs. Padding the device matrices removes bounds checks from the hot loops.
template<int Tile>
__global__ void closePivot(unsigned int* distance, unsigned int* via,
                           size_t pitch, unsigned int pivot) {
    __shared__ unsigned int tileDistance[Tile][Tile];
    __shared__ unsigned int tileVia[Tile][Tile];
    constexpr int Rows = Tile / 8;
    constexpr int Columns = Tile / 32;
    unsigned int value[Rows][Columns];
    unsigned int tag[Rows][Columns];
    const size_t start = static_cast<size_t>(pivot) * Tile;

    #pragma unroll
    for (int r = 0; r < Rows; ++r) {
        const int row = threadIdx.y + r * 8;
        #pragma unroll
        for (int c = 0; c < Columns; ++c) {
            const int column = threadIdx.x + c * 32;
            const size_t index = (start + row) * pitch + start + column;
            tileDistance[row][column] = value[r][c] = distance[index];
            tileVia[row][column] = tag[r][c] = via[index];
        }
    }
    __syncthreads();

    #pragma unroll 1
    for (int k = 0; k < Tile; ++k) {
        #pragma unroll
        for (int r = 0; r < Rows; ++r) {
            const int row = threadIdx.y + r * 8;
            const unsigned int leftDistance = tileDistance[row][k];
            const unsigned int leftVia = tileVia[row][k];
            #pragma unroll
            for (int c = 0; c < Columns; ++c) {
                const int column = threadIdx.x + c * 32;
                if (relax(value[r][c], tag[r][c], leftDistance, tileDistance[k][column],
                          leftVia, tileVia[k][column], static_cast<unsigned int>(start) + k + 1)) {
                    tileDistance[row][column] = value[r][c];
                    tileVia[row][column] = tag[r][c];
                }
            }
        }
        // The pivot row/column cannot improve through their zero diagonal;
        // only other cells are written, so one barrier per k is sufficient.
        __syncthreads();
    }

    #pragma unroll
    for (int r = 0; r < Rows; ++r) {
        #pragma unroll
        for (int c = 0; c < Columns; ++c) {
            const size_t index = (start + threadIdx.y + r * 8) * pitch
                                 + start + threadIdx.x + c * 32;
            distance[index] = value[r][c];
            via[index] = tag[r][c];
        }
    }
}

template<int Tile>
__global__ void closePanels(unsigned int* distance, unsigned int* via,
                            size_t pitch, unsigned int pivot) {
    extern __shared__ unsigned int shared[];
    auto* pivotDistance = reinterpret_cast<unsigned int (*)[Tile]>(shared);
    auto* pivotVia = pivotDistance + Tile;
    auto* panelDistance = pivotVia + Tile;
    auto* panelVia = panelDistance + Tile;
    constexpr int Rows = Tile / 8;
    constexpr int Columns = Tile / 32;
    unsigned int value[Rows][Columns];
    unsigned int tag[Rows][Columns];
    const unsigned int other = blockIdx.x + (blockIdx.x >= pivot);
    const bool columnPanel = blockIdx.y != 0;
    const size_t pivotStart = static_cast<size_t>(pivot) * Tile;
    const size_t rowStart = static_cast<size_t>(columnPanel ? other : pivot) * Tile;
    const size_t columnStart = static_cast<size_t>(columnPanel ? pivot : other) * Tile;

    #pragma unroll
    for (int r = 0; r < Rows; ++r) {
        const int row = threadIdx.y + r * 8;
        #pragma unroll
        for (int c = 0; c < Columns; ++c) {
            const int column = threadIdx.x + c * 32;
            const size_t pivotIndex = (pivotStart + row) * pitch + pivotStart + column;
            const size_t panelIndex = (rowStart + row) * pitch + columnStart + column;
            pivotDistance[row][column] = distance[pivotIndex];
            pivotVia[row][column] = via[pivotIndex];
            panelDistance[row][column] = value[r][c] = distance[panelIndex];
            panelVia[row][column] = tag[r][c] = via[panelIndex];
        }
    }
    __syncthreads();

    #pragma unroll 1
    for (int k = 0; k < Tile; ++k) {
        #pragma unroll
        for (int r = 0; r < Rows; ++r) {
            const int row = threadIdx.y + r * 8;
            const unsigned int leftDistance = columnPanel ? panelDistance[row][k] : pivotDistance[row][k];
            const unsigned int leftVia = columnPanel ? panelVia[row][k] : pivotVia[row][k];
            #pragma unroll
            for (int c = 0; c < Columns; ++c) {
                const int column = threadIdx.x + c * 32;
                const unsigned int rightDistance = columnPanel ? pivotDistance[k][column] : panelDistance[k][column];
                const unsigned int rightVia = columnPanel ? pivotVia[k][column] : panelVia[k][column];
                if (relax(value[r][c], tag[r][c], leftDistance, rightDistance, leftVia, rightVia,
                          static_cast<unsigned int>(pivotStart) + k + 1)) {
                    panelDistance[row][column] = value[r][c];
                    panelVia[row][column] = tag[r][c];
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int r = 0; r < Rows; ++r) {
        #pragma unroll
        for (int c = 0; c < Columns; ++c) {
            const size_t index = (rowStart + threadIdx.y + r * 8) * pitch
                                 + columnStart + threadIdx.x + c * 32;
            distance[index] = value[r][c];
            via[index] = tag[r][c];
        }
    }
}

template<int Tile>
__global__ void closeRemaining(unsigned int* __restrict__ distance, unsigned int* __restrict__ via,
                               size_t pitch, unsigned int pivot) {
    extern __shared__ unsigned int shared[];
    auto* leftDistance = reinterpret_cast<unsigned int (*)[Tile]>(shared);
    auto* leftVia = leftDistance + Tile;
    auto* rightDistance = leftVia + Tile;
    auto* rightVia = rightDistance + Tile;
    constexpr int ThreadRows = Tile == 64 ? 16 : 8;
    constexpr int Rows = Tile / ThreadRows;
    constexpr int Columns = Tile / 32;
    unsigned int value[Rows][Columns];
    unsigned int tag[Rows][Columns];
    const size_t rowStart = static_cast<size_t>(blockIdx.y + (blockIdx.y >= pivot)) * Tile;
    const size_t columnStart = static_cast<size_t>(blockIdx.x + (blockIdx.x >= pivot)) * Tile;
    const size_t pivotStart = static_cast<size_t>(pivot) * Tile;

    #pragma unroll
    for (int r = 0; r < Rows; ++r) {
        const int row = threadIdx.y + r * ThreadRows;
        #pragma unroll
        for (int c = 0; c < Columns; ++c) {
            const int column = threadIdx.x + c * 32;
            const size_t leftIndex = (rowStart + row) * pitch + pivotStart + column;
            const size_t rightIndex = (pivotStart + row) * pitch + columnStart + column;
            leftDistance[row][column] = distance[leftIndex];
            leftVia[row][column] = via[leftIndex];
            rightDistance[row][column] = distance[rightIndex];
            rightVia[row][column] = via[rightIndex];
            const size_t index = (rowStart + row) * pitch + columnStart + column;
            value[r][c] = distance[index];
            tag[r][c] = via[index];
        }
    }
    __syncthreads();

    // The two panels are read-only; output cells remain entirely in registers.
    #pragma unroll 4
    for (int k = 0; k < Tile; ++k) {
        #pragma unroll
        for (int r = 0; r < Rows; ++r) {
            const int row = threadIdx.y + r * ThreadRows;
            const unsigned int left = leftDistance[row][k];
            const unsigned int leftTag = max(static_cast<unsigned int>(pivotStart) + k + 1, leftVia[row][k]);
            #pragma unroll
            for (int c = 0; c < Columns; ++c) {
                const int column = threadIdx.x + c * 32;
                relax(value[r][c], tag[r][c], left, rightDistance[k][column],
                      leftTag, rightVia[k][column], 0);
            }
        }
    }

    #pragma unroll
    for (int r = 0; r < Rows; ++r) {
        #pragma unroll
        for (int c = 0; c < Columns; ++c) {
            const size_t index = (rowStart + threadIdx.y + r * ThreadRows) * pitch
                                 + columnStart + threadIdx.x + c * 32;
            distance[index] = value[r][c];
            via[index] = tag[r][c];
        }
    }
}

template<int Tile>
void computeOnGpu(std::vector<unsigned int>& dist, std::vector<unsigned int>& path, size_t nodes) {
    if (nodes > std::numeric_limits<unsigned int>::max() - Tile + 1) {
        throw std::runtime_error("Graph is too large for GPU matrix indexing");
    }
    const size_t pitch = (nodes + Tile - 1) / Tile * Tile;
    if (pitch > std::numeric_limits<unsigned int>::max() ||
        pitch > std::numeric_limits<size_t>::max() / pitch / (2 * sizeof(unsigned int))) {
        throw std::runtime_error("Graph is too large for GPU matrix indexing");
    }
    const size_t elements = pitch * pitch;
    DeviceStorage storage(2 * elements * sizeof(unsigned int));
    Stream stream;
    unsigned int* deviceDistance = storage.data;
    unsigned int* deviceVia = storage.data + elements;
    const size_t deviceRowBytes = pitch * sizeof(unsigned int);
    const size_t hostRowBytes = nodes * sizeof(unsigned int);
    const unsigned int tiles = static_cast<unsigned int>(pitch / Tile);
    const unsigned int linearBlocks = static_cast<unsigned int>(std::min<size_t>((elements + 255) / 256, 65535));
    constexpr int panelBytes = 4 * Tile * Tile * sizeof(unsigned int);
    if constexpr (panelBytes > 48 * 1024) {
        checkCuda(cudaFuncSetAttribute(closePanels<Tile>, cudaFuncAttributeMaxDynamicSharedMemorySize, panelBytes),
                  "Configure panel shared memory");
        checkCuda(cudaFuncSetAttribute(closeRemaining<Tile>, cudaFuncAttributeMaxDynamicSharedMemorySize, panelBytes),
                  "Configure update shared memory");
    }

    checkCuda(cudaMemcpy2DAsync(deviceDistance, deviceRowBytes, dist.data(), hostRowBytes,
                                hostRowBytes, nodes, cudaMemcpyHostToDevice, stream.value), "Upload distances");
    prepareMatrices<<<linearBlocks, 256, 0, stream.value>>>(deviceDistance, deviceVia, pitch, nodes);
    checkCuda(cudaGetLastError(), "Initialize GPU matrices");

    const dim3 threads(32, 8);
    for (unsigned int pivot = 0; pivot < tiles; ++pivot) {
        closePivot<Tile><<<1, threads, 0, stream.value>>>(deviceDistance, deviceVia, pitch, pivot);
        if (tiles > 1) {
            closePanels<Tile><<<dim3(tiles - 1, 2), threads, panelBytes, stream.value>>>(deviceDistance, deviceVia, pitch, pivot);
            closeRemaining<Tile><<<dim3(tiles - 1, tiles - 1), dim3(32, Tile == 64 ? 16 : 8), panelBytes, stream.value>>>(deviceDistance, deviceVia, pitch, pivot);
        }
    }
    checkCuda(cudaGetLastError(), "Launch blocked Floyd-Warshall");
    decodePaths<<<linearBlocks, 256, 0, stream.value>>>(deviceVia, pitch);
    checkCuda(cudaGetLastError(), "Decode path matrix");
    checkCuda(cudaMemcpy2DAsync(dist.data(), hostRowBytes, deviceDistance, deviceRowBytes,
                                hostRowBytes, nodes, cudaMemcpyDeviceToHost, stream.value), "Download distances");
    checkCuda(cudaMemcpy2DAsync(path.data(), hostRowBytes, deviceVia, deviceRowBytes,
                                hostRowBytes, nodes, cudaMemcpyDeviceToHost, stream.value), "Download paths");
    checkCuda(cudaStreamSynchronize(stream.value), "Complete Floyd-Warshall");
}

} // namespace

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes) {
    if (numNodes == 0) {
        return;
    }
    // Larger tiles improve reuse for large matrices. Keep smaller tiles on
    // small graphs to expose enough blocks, and on GPUs with less shared memory.
    if (numNodes >= 4096) {
        int device = 0;
        int sharedBytes = 0;
        checkCuda(cudaGetDevice(&device), "Get CUDA device");
        checkCuda(cudaDeviceGetAttribute(&sharedBytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, device),
                  "Query GPU shared memory");
        if (sharedBytes >= 4 * 64 * 64 * static_cast<int>(sizeof(unsigned int))) {
            computeOnGpu<64>(dist, path, numNodes);
            return;
        }
    }
    computeOnGpu<32>(dist, path, numNodes);
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
    
    if (numNodes > std::numeric_limits<unsigned int>::max() - 63 ||
        (numNodes != 0 && numNodes > std::numeric_limits<size_t>::max() / numNodes / sizeof(unsigned int))) {
        throw std::runtime_error("Graph is too large for matrix indexing");
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

    // Establish the CUDA context before timing the computation. CUDA is
    // mandatory: initialization failures are reported, with no CPU fallback.
    checkCuda(cudaFree(nullptr), "Initialize CUDA runtime");
    
    // Run Floyd-Warshall
    printf("Computing shortest paths...\n");
    auto start = std::chrono::steady_clock::now();
    
    floydWarshall(dist, path, numNodes);
    
    auto end = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(end - start).count();
    
    printf("Computation time: %.3f ms\n", seconds * 1000.0);
    
    // Calculate operations per second
    // Floyd-Warshall has O(n³) complexity
    double ops = (double)numNodes * numNodes * numNodes;
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
