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

// A 32x32 tile gives the two non-pivot input tiles enough reuse to make the
// main phase compute-bound on recent NVIDIA GPUs.  32x8 threads let every
// warp read and write full contiguous rows while each thread owns four rows.
constexpr unsigned int TILE_SIZE = 32;
constexpr unsigned int BLOCK_X = 32;
constexpr unsigned int BLOCK_Y = 8;

// Index calculation for the destination-major flattened 2D array used by the
// original program.  In matrix notation this is matrix[destination][source].
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

[[noreturn]] void cudaFailure(const cudaError_t error, const char* expression,
                              const char* file, const int line) {
    std::fprintf(stderr, "CUDA error at %s:%d while executing %s: %s\n", file, line,
                 expression, cudaGetErrorString(error));
    std::exit(EXIT_FAILURE);
}

inline void cudaCheck(const cudaError_t error, const char* expression, const char* file,
                      const int line) {
    if (error != cudaSuccess) {
        cudaFailure(error, expression, file, line);
    }
}

#define CUDA_CHECK(expression) cudaCheck((expression), #expression, __FILE__, __LINE__)

__device__ __forceinline__ void relax(unsigned int& distance, unsigned int& path,
                                      const unsigned int left, const unsigned int right,
                                      const unsigned int intermediate) {
    // Unsigned addition deliberately preserves the arithmetic of the original
    // implementation, including its wraparound behavior for extreme inputs.
    const unsigned int candidate = left + right;
    if (candidate < distance) {
        distance = candidate;
        path = intermediate;
    }
}

// Phase 1: close the diagonal (pivot) tile.  The explicit synchronization
// after every vertex maintains the original Floyd-Warshall k ordering.
__global__ void closePivotTile(unsigned int* const dist, unsigned int* const path,
                               unsigned int* const pivotColumnHistory,
                               unsigned int* const pivotRowHistory, const size_t stride,
                               const unsigned int pivotTile) {
    __shared__ unsigned int pivot[TILE_SIZE][TILE_SIZE + 1];

    const unsigned int column = threadIdx.x;
    const unsigned int firstRow = threadIdx.y;
    const unsigned int secondRow = firstRow + BLOCK_Y;
    const unsigned int thirdRow = secondRow + BLOCK_Y;
    const unsigned int fourthRow = thirdRow + BLOCK_Y;
    const size_t base = static_cast<size_t>(pivotTile) * TILE_SIZE;

    const size_t index0 = (base + firstRow) * stride + base + column;
    const size_t index1 = (base + secondRow) * stride + base + column;
    const size_t index2 = (base + thirdRow) * stride + base + column;
    const size_t index3 = (base + fourthRow) * stride + base + column;

    pivot[firstRow][column] = dist[index0];
    pivot[secondRow][column] = dist[index1];
    pivot[thirdRow][column] = dist[index2];
    pivot[fourthRow][column] = dist[index3];

    unsigned int path0 = path[index0];
    unsigned int path1 = path[index1];
    unsigned int path2 = path[index2];
    unsigned int path3 = path[index3];

    __syncthreads();

#pragma unroll
    for (unsigned int localK = 0; localK < TILE_SIZE; ++localK) {
        const unsigned int k = static_cast<unsigned int>(base + localK);
        const size_t historyOffset = static_cast<size_t>(localK) * TILE_SIZE;

        // Capture D^(k-1)[row][k] and D^(k-1)[k][column].  The later phases
        // use these snapshots to retain the scalar algorithm's strict update
        // order within this tile.
        if (column == 0) {
            pivotColumnHistory[historyOffset + firstRow] = pivot[firstRow][localK];
            pivotColumnHistory[historyOffset + secondRow] = pivot[secondRow][localK];
            pivotColumnHistory[historyOffset + thirdRow] = pivot[thirdRow][localK];
            pivotColumnHistory[historyOffset + fourthRow] = pivot[fourthRow][localK];
        }
        if (firstRow == 0) {
            pivotRowHistory[historyOffset + column] = pivot[localK][column];
        }

        relax(pivot[firstRow][column], path0, pivot[firstRow][localK],
              pivot[localK][column], k);
        relax(pivot[secondRow][column], path1, pivot[secondRow][localK],
              pivot[localK][column], k);
        relax(pivot[thirdRow][column], path2, pivot[thirdRow][localK],
              pivot[localK][column], k);
        relax(pivot[fourthRow][column], path3, pivot[fourthRow][localK],
              pivot[localK][column], k);
        __syncthreads();
    }

    dist[index0] = pivot[firstRow][column];
    dist[index1] = pivot[secondRow][column];
    dist[index2] = pivot[thirdRow][column];
    dist[index3] = pivot[fourthRow][column];
    path[index0] = path0;
    path[index1] = path1;
    path[index2] = path2;
    path[index3] = path3;
}

// Phase 2: update one pivot row or one pivot column tile.  The two template
// instantiations avoid a per-thread direction branch in this hot kernel.
template <bool UPDATE_PIVOT_ROW>
__global__ void updatePivotLine(unsigned int* const dist, unsigned int* const path,
                                const unsigned int* const pivotHistory,
                                unsigned int* const targetHistory, const size_t stride,
                                const unsigned int pivotTile) {
    __shared__ unsigned int pivot[TILE_SIZE][TILE_SIZE + 1];
    __shared__ unsigned int target[TILE_SIZE][TILE_SIZE + 1];

    const unsigned int targetTile = blockIdx.x;
    if (targetTile == pivotTile) {
        return;
    }

    const unsigned int column = threadIdx.x;
    const unsigned int firstRow = threadIdx.y;
    const unsigned int secondRow = firstRow + BLOCK_Y;
    const unsigned int thirdRow = secondRow + BLOCK_Y;
    const unsigned int fourthRow = thirdRow + BLOCK_Y;
    const size_t pivotBase = static_cast<size_t>(pivotTile) * TILE_SIZE;
    const size_t targetBase = static_cast<size_t>(targetTile) * TILE_SIZE;

    // Reformat the pivot's pre-k snapshots into the normal shared-memory
    // orientation.  A fully closed pivot tile is enough for distances, but
    // these values are needed to preserve the original path-update semantics.
    if constexpr (UPDATE_PIVOT_ROW) {
        pivot[firstRow][column] = pivotHistory[static_cast<size_t>(column) * TILE_SIZE + firstRow];
        pivot[secondRow][column] = pivotHistory[static_cast<size_t>(column) * TILE_SIZE + secondRow];
        pivot[thirdRow][column] = pivotHistory[static_cast<size_t>(column) * TILE_SIZE + thirdRow];
        pivot[fourthRow][column] = pivotHistory[static_cast<size_t>(column) * TILE_SIZE + fourthRow];
    } else {
        pivot[firstRow][column] = pivotHistory[static_cast<size_t>(firstRow) * TILE_SIZE + column];
        pivot[secondRow][column] = pivotHistory[static_cast<size_t>(secondRow) * TILE_SIZE + column];
        pivot[thirdRow][column] = pivotHistory[static_cast<size_t>(thirdRow) * TILE_SIZE + column];
        pivot[fourthRow][column] = pivotHistory[static_cast<size_t>(fourthRow) * TILE_SIZE + column];
    }

    const size_t targetRowBase = UPDATE_PIVOT_ROW ? pivotBase : targetBase;
    const size_t targetColumnBase = UPDATE_PIVOT_ROW ? targetBase : pivotBase;
    const size_t targetIndex0 = (targetRowBase + firstRow) * stride + targetColumnBase + column;
    const size_t targetIndex1 = (targetRowBase + secondRow) * stride + targetColumnBase + column;
    const size_t targetIndex2 = (targetRowBase + thirdRow) * stride + targetColumnBase + column;
    const size_t targetIndex3 = (targetRowBase + fourthRow) * stride + targetColumnBase + column;
    target[firstRow][column] = dist[targetIndex0];
    target[secondRow][column] = dist[targetIndex1];
    target[thirdRow][column] = dist[targetIndex2];
    target[fourthRow][column] = dist[targetIndex3];

    unsigned int path0 = path[targetIndex0];
    unsigned int path1 = path[targetIndex1];
    unsigned int path2 = path[targetIndex2];
    unsigned int path3 = path[targetIndex3];

    __syncthreads();

#pragma unroll
    for (unsigned int localK = 0; localK < TILE_SIZE; ++localK) {
        const unsigned int k = static_cast<unsigned int>(pivotBase + localK);
        const size_t historyOffset =
            (static_cast<size_t>(targetTile) * TILE_SIZE + localK) * TILE_SIZE;
        if constexpr (UPDATE_PIVOT_ROW) {
            if (firstRow == 0) {
                targetHistory[historyOffset + column] = target[localK][column];
            }
            relax(target[firstRow][column], path0, pivot[firstRow][localK],
                  target[localK][column], k);
            relax(target[secondRow][column], path1, pivot[secondRow][localK],
                  target[localK][column], k);
            relax(target[thirdRow][column], path2, pivot[thirdRow][localK],
                  target[localK][column], k);
            relax(target[fourthRow][column], path3, pivot[fourthRow][localK],
                  target[localK][column], k);
        } else {
            if (column == 0) {
                targetHistory[historyOffset + firstRow] = target[firstRow][localK];
                targetHistory[historyOffset + secondRow] = target[secondRow][localK];
                targetHistory[historyOffset + thirdRow] = target[thirdRow][localK];
                targetHistory[historyOffset + fourthRow] = target[fourthRow][localK];
            }
            relax(target[firstRow][column], path0, target[firstRow][localK],
                  pivot[localK][column], k);
            relax(target[secondRow][column], path1, target[secondRow][localK],
                  pivot[localK][column], k);
            relax(target[thirdRow][column], path2, target[thirdRow][localK],
                  pivot[localK][column], k);
            relax(target[fourthRow][column], path3, target[fourthRow][localK],
                  pivot[localK][column], k);
        }
        __syncthreads();
    }

    dist[targetIndex0] = target[firstRow][column];
    dist[targetIndex1] = target[secondRow][column];
    dist[targetIndex2] = target[thirdRow][column];
    dist[targetIndex3] = target[fourthRow][column];
    path[targetIndex0] = path0;
    path[targetIndex1] = path1;
    path[targetIndex2] = path2;
    path[targetIndex3] = path3;
}

// Phase 3: update every non-pivot tile.  A block holds the two source tiles
// in shared memory and accumulates four output cells per thread in registers.
__global__ void updateRemainingTiles(unsigned int* const dist, unsigned int* const path,
                                     const unsigned int* const columnHistory,
                                     const unsigned int* const rowHistory, const size_t stride,
                                     const unsigned int pivotTile) {
    __shared__ unsigned int left[TILE_SIZE][TILE_SIZE + 1];
    __shared__ unsigned int right[TILE_SIZE][TILE_SIZE + 1];

    const unsigned int columnTile = blockIdx.x;
    const unsigned int rowTile = blockIdx.y;
    if (rowTile == pivotTile || columnTile == pivotTile) {
        return;
    }

    const unsigned int column = threadIdx.x;
    const unsigned int firstRow = threadIdx.y;
    const unsigned int secondRow = firstRow + BLOCK_Y;
    const unsigned int thirdRow = secondRow + BLOCK_Y;
    const unsigned int fourthRow = thirdRow + BLOCK_Y;
    const size_t pivotBase = static_cast<size_t>(pivotTile) * TILE_SIZE;
    const size_t rowBase = static_cast<size_t>(rowTile) * TILE_SIZE;
    const size_t columnBase = static_cast<size_t>(columnTile) * TILE_SIZE;

    const size_t leftHistoryBase = static_cast<size_t>(rowTile) * TILE_SIZE * TILE_SIZE;
    const size_t rightHistoryBase = static_cast<size_t>(columnTile) * TILE_SIZE * TILE_SIZE;
    left[firstRow][column] = columnHistory[leftHistoryBase + static_cast<size_t>(column) * TILE_SIZE + firstRow];
    left[secondRow][column] = columnHistory[leftHistoryBase + static_cast<size_t>(column) * TILE_SIZE + secondRow];
    left[thirdRow][column] = columnHistory[leftHistoryBase + static_cast<size_t>(column) * TILE_SIZE + thirdRow];
    left[fourthRow][column] = columnHistory[leftHistoryBase + static_cast<size_t>(column) * TILE_SIZE + fourthRow];
    right[firstRow][column] = rowHistory[rightHistoryBase + static_cast<size_t>(firstRow) * TILE_SIZE + column];
    right[secondRow][column] = rowHistory[rightHistoryBase + static_cast<size_t>(secondRow) * TILE_SIZE + column];
    right[thirdRow][column] = rowHistory[rightHistoryBase + static_cast<size_t>(thirdRow) * TILE_SIZE + column];
    right[fourthRow][column] = rowHistory[rightHistoryBase + static_cast<size_t>(fourthRow) * TILE_SIZE + column];

    const size_t targetIndex0 = (rowBase + firstRow) * stride + columnBase + column;
    const size_t targetIndex1 = (rowBase + secondRow) * stride + columnBase + column;
    const size_t targetIndex2 = (rowBase + thirdRow) * stride + columnBase + column;
    const size_t targetIndex3 = (rowBase + fourthRow) * stride + columnBase + column;
    unsigned int distance0 = dist[targetIndex0];
    unsigned int distance1 = dist[targetIndex1];
    unsigned int distance2 = dist[targetIndex2];
    unsigned int distance3 = dist[targetIndex3];
    unsigned int path0 = path[targetIndex0];
    unsigned int path1 = path[targetIndex1];
    unsigned int path2 = path[targetIndex2];
    unsigned int path3 = path[targetIndex3];

    __syncthreads();

#pragma unroll
    for (unsigned int localK = 0; localK < TILE_SIZE; ++localK) {
        const unsigned int k = static_cast<unsigned int>(pivotBase + localK);
        relax(distance0, path0, left[firstRow][localK], right[localK][column], k);
        relax(distance1, path1, left[secondRow][localK], right[localK][column], k);
        relax(distance2, path2, left[thirdRow][localK], right[localK][column], k);
        relax(distance3, path3, left[fourthRow][localK], right[localK][column], k);
    }

    dist[targetIndex0] = distance0;
    dist[targetIndex1] = distance1;
    dist[targetIndex2] = distance2;
    dist[targetIndex3] = distance3;
    path[targetIndex0] = path0;
    path[targetIndex1] = path1;
    path[targetIndex2] = path2;
    path[targetIndex3] = path3;
}

// Fill the padding before a 2D host-to-device copy.  Padding with INF keeps
// partial edge tiles from participating in paths between real vertices.
__global__ void initializePaddedMatrices(unsigned int* const dist, unsigned int* const path,
                                         const size_t elementCount) {
    for (size_t element = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         element < elementCount;
         element += static_cast<size_t>(blockDim.x) * gridDim.x) {
        dist[element] = INF;
        path[element] = 0;
    }
}

class CudaFloydWarshall {
public:
    explicit CudaFloydWarshall(const size_t numNodes) : numNodes_(numNodes) {
        if (numNodes_ == 0) {
            return;
        }
        if (numNodes_ > std::numeric_limits<size_t>::max() - (TILE_SIZE - 1)) {
            std::fprintf(stderr, "Matrix dimension is too large\n");
            std::exit(EXIT_FAILURE);
        }

        stride_ = ((numNodes_ + TILE_SIZE - 1) / TILE_SIZE) * TILE_SIZE;
        if (stride_ > std::numeric_limits<size_t>::max() / stride_) {
            std::fprintf(stderr, "Matrix is too large\n");
            std::exit(EXIT_FAILURE);
        }
        elementCount_ = stride_ * stride_;
        tileCount_ = stride_ / TILE_SIZE;
        if (tileCount_ > std::numeric_limits<unsigned int>::max()) {
            std::fprintf(stderr, "Matrix has too many CUDA tiles\n");
            std::exit(EXIT_FAILURE);
        }

        const size_t tileElements = static_cast<size_t>(TILE_SIZE) * TILE_SIZE;
        const size_t lineHistoryElements = tileCount_ * tileElements;
        CUDA_CHECK(cudaMalloc(&deviceDist_, elementCount_ * sizeof(*deviceDist_)));
        CUDA_CHECK(cudaMalloc(&devicePath_, elementCount_ * sizeof(*devicePath_)));
        CUDA_CHECK(cudaMalloc(&pivotColumnHistory_, tileElements * sizeof(*pivotColumnHistory_)));
        CUDA_CHECK(cudaMalloc(&pivotRowHistory_, tileElements * sizeof(*pivotRowHistory_)));
        CUDA_CHECK(cudaMalloc(&columnHistory_, lineHistoryElements * sizeof(*columnHistory_)));
        CUDA_CHECK(cudaMalloc(&rowHistory_, lineHistoryElements * sizeof(*rowHistory_)));
    }

    CudaFloydWarshall(const CudaFloydWarshall&) = delete;
    CudaFloydWarshall& operator=(const CudaFloydWarshall&) = delete;

    ~CudaFloydWarshall() {
        if (deviceDist_ != nullptr) {
            cudaFree(deviceDist_);
        }
        if (devicePath_ != nullptr) {
            cudaFree(devicePath_);
        }
        if (pivotColumnHistory_ != nullptr) {
            cudaFree(pivotColumnHistory_);
        }
        if (pivotRowHistory_ != nullptr) {
            cudaFree(pivotRowHistory_);
        }
        if (columnHistory_ != nullptr) {
            cudaFree(columnHistory_);
        }
        if (rowHistory_ != nullptr) {
            cudaFree(rowHistory_);
        }
    }

    void upload(const std::vector<unsigned int>& dist, const std::vector<unsigned int>& path) {
        if (numNodes_ == 0) {
            return;
        }

        constexpr unsigned int initThreads = 256;
        constexpr unsigned int maxInitBlocks = 65535;
        const size_t requestedBlocks = (elementCount_ + initThreads - 1) / initThreads;
        const unsigned int initBlocks = static_cast<unsigned int>(
            std::min(requestedBlocks, static_cast<size_t>(maxInitBlocks)));
        initializePaddedMatrices<<<initBlocks, initThreads>>>(deviceDist_, devicePath_, elementCount_);
        CUDA_CHECK(cudaGetLastError());

        const size_t hostPitch = numNodes_ * sizeof(unsigned int);
        const size_t devicePitch = stride_ * sizeof(unsigned int);
        CUDA_CHECK(cudaMemcpy2D(deviceDist_, devicePitch, dist.data(), hostPitch, hostPitch,
                                numNodes_, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy2D(devicePath_, devicePitch, path.data(), hostPitch, hostPitch,
                                numNodes_, cudaMemcpyHostToDevice));
    }

    void execute() {
        if (numNodes_ == 0) {
            return;
        }

        const dim3 block(BLOCK_X, BLOCK_Y);
        const unsigned int tileCount = static_cast<unsigned int>(tileCount_);
        for (unsigned int pivot = 0; pivot < tileCount; ++pivot) {
            closePivotTile<<<1, block>>>(deviceDist_, devicePath_, pivotColumnHistory_,
                                          pivotRowHistory_, stride_, pivot);
            if (tileCount > 1) {
                updatePivotLine<true><<<tileCount, block>>>(
                    deviceDist_, devicePath_, pivotColumnHistory_, rowHistory_, stride_, pivot);
                updatePivotLine<false><<<tileCount, block>>>(
                    deviceDist_, devicePath_, pivotRowHistory_, columnHistory_, stride_, pivot);
                updateRemainingTiles<<<dim3(tileCount, tileCount), block>>>(
                    deviceDist_, devicePath_, columnHistory_, rowHistory_, stride_, pivot);
            }
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void download(std::vector<unsigned int>& dist, std::vector<unsigned int>& path) const {
        if (numNodes_ == 0) {
            return;
        }

        const size_t hostPitch = numNodes_ * sizeof(unsigned int);
        const size_t devicePitch = stride_ * sizeof(unsigned int);
        CUDA_CHECK(cudaMemcpy2D(dist.data(), hostPitch, deviceDist_, devicePitch, hostPitch,
                                numNodes_, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy2D(path.data(), hostPitch, devicePath_, devicePitch, hostPitch,
                                numNodes_, cudaMemcpyDeviceToHost));
    }

private:
    size_t numNodes_ = 0;
    size_t stride_ = 0;
    size_t elementCount_ = 0;
    size_t tileCount_ = 0;
    unsigned int* deviceDist_ = nullptr;
    unsigned int* devicePath_ = nullptr;
    unsigned int* pivotColumnHistory_ = nullptr;
    unsigned int* pivotRowHistory_ = nullptr;
    unsigned int* columnHistory_ = nullptr;
    unsigned int* rowHistory_ = nullptr;
};

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

    // Allocate and populate GPU memory outside the measured compute region,
    // matching the original benchmark's convention of timing only the solver.
    CudaFloydWarshall gpu(numNodes);
    gpu.upload(dist, path);

    // Run Floyd-Warshall
    printf("Computing shortest paths...\n");
    auto start = std::chrono::high_resolution_clock::now();

    gpu.execute();

    auto end = std::chrono::high_resolution_clock::now();
    const double elapsedMilliseconds =
        std::chrono::duration<double, std::milli>(end - start).count();

    // Keep result transfer out of the compute measurement just as initialization
    // is excluded, then retain the original host-side output and validation API.
    gpu.download(dist, path);

    printf("Computation time: %.3f ms\n", elapsedMilliseconds);

    // Calculate operations per second
    // Floyd-Warshall has O(n³) complexity
    double ops = (double)numNodes * numNodes * numNodes;
    double gflops = ops / (elapsedMilliseconds / 1000.0) / 1e9;
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
