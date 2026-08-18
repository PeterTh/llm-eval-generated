#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;
constexpr int TILE_SIZE = 32;
constexpr int BLOCK_ROWS = 8;

#define CUDA_CHECK(call)                                                                  \
    do {                                                                                  \
        const cudaError_t error_ = (call);                                                 \
        if (error_ != cudaSuccess) {                                                       \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,        \
                         cudaGetErrorString(error_));                                      \
            std::exit(EXIT_FAILURE);                                                       \
        }                                                                                 \
    } while (false)

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

// Phase 1 closes the diagonal tile. A 32x8 thread block handles four rows per
// thread, giving fully coalesced accesses while keeping the block at 256
// threads. The extra shared-memory column eliminates bank conflicts.
__global__ __launch_bounds__(TILE_SIZE * BLOCK_ROWS, 2)
void floydWarshallPivot(unsigned int* __restrict__ dist,
                        unsigned int* __restrict__ path,
                        const int pitch,
                        const int round) {
    __shared__ unsigned int tile[TILE_SIZE][TILE_SIZE + 1];

    const int x = threadIdx.x;
    const int y = threadIdx.y;
    const int base = round * TILE_SIZE;
    int bestIntermediate[TILE_SIZE / BLOCK_ROWS];

#pragma unroll
    for (int q = 0; q < TILE_SIZE / BLOCK_ROWS; ++q) {
        const int row = y + q * BLOCK_ROWS;
        tile[row][x] = dist[(base + row) * pitch + base + x];
        bestIntermediate[q] = -1;
    }
    __syncthreads();

#pragma unroll
    for (int k = 0; k < TILE_SIZE; ++k) {
#pragma unroll
        for (int q = 0; q < TILE_SIZE / BLOCK_ROWS; ++q) {
            const int row = y + q * BLOCK_ROWS;
            const unsigned int oldDistance = tile[row][x];
            const unsigned int newDistance = tile[row][k] + tile[k][x];
            if (newDistance < oldDistance) {
                tile[row][x] = newDistance;
                bestIntermediate[q] = base + k;
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int q = 0; q < TILE_SIZE / BLOCK_ROWS; ++q) {
        const int row = y + q * BLOCK_ROWS;
        const size_t index = static_cast<size_t>(base + row) * pitch + base + x;
        dist[index] = tile[row][x];
        if (bestIntermediate[q] >= 0) {
            path[index] = static_cast<unsigned int>(bestIntermediate[q]);
        }
    }
}

// Phase 2 closes all tiles in the pivot row and pivot column. blockIdx.y
// selects the row (0) or column (1), so both directions share a single launch.
__global__ __launch_bounds__(TILE_SIZE * BLOCK_ROWS)
void floydWarshallPivotTiles(unsigned int* __restrict__ dist,
                             unsigned int* __restrict__ path,
                             const int pitch,
                             const int round) {
    __shared__ unsigned int pivot[TILE_SIZE][TILE_SIZE + 1];
    __shared__ unsigned int tile[TILE_SIZE][TILE_SIZE + 1];

    const int x = threadIdx.x;
    const int y = threadIdx.y;
    const int pivotBase = round * TILE_SIZE;
    int tileIndex = blockIdx.x;
    tileIndex += tileIndex >= round;

    const bool isPivotRow = blockIdx.y == 0;
    const int rowBase = (isPivotRow ? round : tileIndex) * TILE_SIZE;
    const int columnBase = (isPivotRow ? tileIndex : round) * TILE_SIZE;
    int bestIntermediate[TILE_SIZE / BLOCK_ROWS];

#pragma unroll
    for (int q = 0; q < TILE_SIZE / BLOCK_ROWS; ++q) {
        const int row = y + q * BLOCK_ROWS;
        pivot[row][x] = dist[(pivotBase + row) * pitch + pivotBase + x];
        tile[row][x] = dist[(rowBase + row) * pitch + columnBase + x];
        bestIntermediate[q] = -1;
    }
    __syncthreads();

#pragma unroll
    for (int k = 0; k < TILE_SIZE; ++k) {
#pragma unroll
        for (int q = 0; q < TILE_SIZE / BLOCK_ROWS; ++q) {
            const int row = y + q * BLOCK_ROWS;
            const unsigned int newDistance = isPivotRow
                ? pivot[row][k] + tile[k][x]
                : tile[row][k] + pivot[k][x];
            if (newDistance < tile[row][x]) {
                tile[row][x] = newDistance;
                bestIntermediate[q] = pivotBase + k;
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int q = 0; q < TILE_SIZE / BLOCK_ROWS; ++q) {
        const int row = y + q * BLOCK_ROWS;
        const size_t index = static_cast<size_t>(rowBase + row) * pitch + columnBase + x;
        dist[index] = tile[row][x];
        if (bestIntermediate[q] >= 0) {
            path[index] = static_cast<unsigned int>(bestIntermediate[q]);
        }
    }
}

// Phase 3 is the throughput-critical kernel. Every block updates a complete
// non-pivot tile. Its two input tiles are reused from shared memory and each
// thread updates four independent distances to expose instruction-level
// parallelism as well as tile-level GPU parallelism.
__global__ __launch_bounds__(TILE_SIZE * BLOCK_ROWS)
void floydWarshallRemainingTiles(unsigned int* __restrict__ dist,
                                 unsigned int* __restrict__ path,
                                 const int pitch,
                                 const int round) {
    __shared__ unsigned int fromSource[TILE_SIZE][TILE_SIZE + 1];
    __shared__ unsigned int toDestination[TILE_SIZE][TILE_SIZE + 1];

    const int x = threadIdx.x;
    const int y = threadIdx.y;
    const int pivotBase = round * TILE_SIZE;

    int tileColumn = blockIdx.x;
    tileColumn += tileColumn >= round;
    int tileRow = blockIdx.y;
    tileRow += tileRow >= round;
    const int rowBase = tileRow * TILE_SIZE;
    const int columnBase = tileColumn * TILE_SIZE;

    unsigned int best[TILE_SIZE / BLOCK_ROWS];
    int bestIntermediate[TILE_SIZE / BLOCK_ROWS];

#pragma unroll
    for (int q = 0; q < TILE_SIZE / BLOCK_ROWS; ++q) {
        const int row = y + q * BLOCK_ROWS;
        fromSource[row][x] = dist[(rowBase + row) * pitch + pivotBase + x];
        toDestination[row][x] = dist[(pivotBase + row) * pitch + columnBase + x];
        best[q] = dist[(rowBase + row) * pitch + columnBase + x];
        bestIntermediate[q] = -1;
    }
    __syncthreads();

#pragma unroll
    for (int k = 0; k < TILE_SIZE; ++k) {
        const unsigned int rightDistance = toDestination[k][x];
#pragma unroll
        for (int q = 0; q < TILE_SIZE / BLOCK_ROWS; ++q) {
            const int row = y + q * BLOCK_ROWS;
            const unsigned int newDistance = fromSource[row][k] + rightDistance;
            if (newDistance < best[q]) {
                best[q] = newDistance;
                bestIntermediate[q] = pivotBase + k;
            }
        }
    }

#pragma unroll
    for (int q = 0; q < TILE_SIZE / BLOCK_ROWS; ++q) {
        const int row = y + q * BLOCK_ROWS;
        const size_t index = static_cast<size_t>(rowBase + row) * pitch + columnBase + x;
        dist[index] = best[q];
        if (bestIntermediate[q] >= 0) {
            path[index] = static_cast<unsigned int>(bestIntermediate[q]);
        }
    }
}

class CudaFloydWarshall {
  public:
    CudaFloydWarshall(const std::vector<unsigned int>& dist,
                      const std::vector<unsigned int>& path,
                      const size_t numNodes)
        : numNodes_(numNodes) {
        if (numNodes_ == 0) {
            return;
        }
        if (numNodes_ > static_cast<size_t>(INT_MAX - TILE_SIZE)) {
            std::fprintf(stderr, "Number of nodes is too large for CUDA indexing\n");
            std::exit(EXIT_FAILURE);
        }

        pitch_ = static_cast<int>((numNodes_ + TILE_SIZE - 1) / TILE_SIZE * TILE_SIZE);
        tileCount_ = pitch_ / TILE_SIZE;
        const size_t paddedBytes = static_cast<size_t>(pitch_) * pitch_ * sizeof(unsigned int);
        const size_t rowBytes = numNodes_ * sizeof(unsigned int);

        CUDA_CHECK(cudaMalloc(&deviceDist_, paddedBytes));
        CUDA_CHECK(cudaMalloc(&devicePath_, paddedBytes));

        // Padding makes every kernel branch-free, including the last tile.
        // 0x3f3f3f3f behaves as infinity and remains below UINT_MAX / 2.
        CUDA_CHECK(cudaMemset(deviceDist_, 0x3f, paddedBytes));
        CUDA_CHECK(cudaMemset(devicePath_, 0, paddedBytes));
        CUDA_CHECK(cudaMemcpy2D(deviceDist_, static_cast<size_t>(pitch_) * sizeof(unsigned int),
                                dist.data(), rowBytes, rowBytes, numNodes_,
                                cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy2D(devicePath_, static_cast<size_t>(pitch_) * sizeof(unsigned int),
                                path.data(), rowBytes, rowBytes, numNodes_,
                                cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventCreate(&startEvent_));
        CUDA_CHECK(cudaEventCreate(&stopEvent_));
    }

    CudaFloydWarshall(const CudaFloydWarshall&) = delete;
    CudaFloydWarshall& operator=(const CudaFloydWarshall&) = delete;

    ~CudaFloydWarshall() {
        if (startEvent_ != nullptr) {
            cudaEventDestroy(startEvent_);
            cudaEventDestroy(stopEvent_);
        }
        cudaFree(deviceDist_);
        cudaFree(devicePath_);
    }

    float compute() {
        if (numNodes_ == 0) {
            return 0.0F;
        }

        constexpr dim3 block(TILE_SIZE, BLOCK_ROWS);
        CUDA_CHECK(cudaEventRecord(startEvent_));
        for (int round = 0; round < tileCount_; ++round) {
            floydWarshallPivot<<<1, block>>>(deviceDist_, devicePath_, pitch_, round);
            if (tileCount_ > 1) {
                const dim3 pivotTiles(tileCount_ - 1, 2);
                floydWarshallPivotTiles<<<pivotTiles, block>>>(
                    deviceDist_, devicePath_, pitch_, round);

                const dim3 remainingTiles(tileCount_ - 1, tileCount_ - 1);
                floydWarshallRemainingTiles<<<remainingTiles, block>>>(
                    deviceDist_, devicePath_, pitch_, round);
            }
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(stopEvent_));
        CUDA_CHECK(cudaEventSynchronize(stopEvent_));

        float elapsedMilliseconds = 0.0F;
        CUDA_CHECK(cudaEventElapsedTime(&elapsedMilliseconds, startEvent_, stopEvent_));
        return elapsedMilliseconds;
    }

    void download(std::vector<unsigned int>& dist, std::vector<unsigned int>& path) const {
        if (numNodes_ == 0) {
            return;
        }
        const size_t rowBytes = numNodes_ * sizeof(unsigned int);
        CUDA_CHECK(cudaMemcpy2D(dist.data(), rowBytes,
                                deviceDist_, static_cast<size_t>(pitch_) * sizeof(unsigned int),
                                rowBytes, numNodes_, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy2D(path.data(), rowBytes,
                                devicePath_, static_cast<size_t>(pitch_) * sizeof(unsigned int),
                                rowBytes, numNodes_, cudaMemcpyDeviceToHost));
    }

  private:
    size_t numNodes_ = 0;
    int pitch_ = 0;
    int tileCount_ = 0;
    unsigned int* deviceDist_ = nullptr;
    unsigned int* devicePath_ = nullptr;
    cudaEvent_t startEvent_ = nullptr;
    cudaEvent_t stopEvent_ = nullptr;
};

float floydWarshall(std::vector<unsigned int>& dist,
                    std::vector<unsigned int>& path,
                    const size_t numNodes) {
    CudaFloydWarshall solver(dist, path, numNodes);
    const float elapsedMilliseconds = solver.compute();
    solver.download(dist, path);
    return elapsedMilliseconds;
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
    const float durationMilliseconds = floydWarshall(dist, path, numNodes);

    printf("Computation time: %.3f ms\n", durationMilliseconds);
    
    // Calculate operations per second
    // Floyd-Warshall has O(n³) complexity
    double ops = (double)numNodes * numNodes * numNodes;
    const double seconds = static_cast<double>(durationMilliseconds) / 1000.0;
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
