#include <algorithm>
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
    for (size_t row = 0; row < numNodes; ++row) {
        std::fill_n(path.data() + row * numNodes, numNodes, static_cast<unsigned int>(row));
    }
}

namespace {

constexpr int blockCols = 32;
constexpr int blockRows = 8;
constexpr int smallTile = 32;
constexpr int largeTile = 64;
constexpr size_t largeTileThreshold = 3072;
// Nine high bits hold the sum of two benchmark distances. The remaining
// bits hold the path label, so one unsigned minimum compares both fields.
constexpr int distanceShift = 23;
constexpr unsigned int witnessMask = (1u << distanceShift) - 1;
constexpr unsigned int distanceMask = ~witnessMask;
constexpr unsigned int paddingDistance = 255;
static_assert(MAX_DISTANCE + 1 < paddingDistance);
static_assert(2 * paddingDistance <= (distanceMask >> distanceShift));

void checkCuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        fprintf(stderr, "%s failed: %s\n", operation, cudaGetErrorString(result));
        std::exit(EXIT_FAILURE);
    }
}

template <int TileSize>
__global__ void packDistances(unsigned int* matrix, int numNodes, int stride) {
    const int firstCol = blockIdx.x * TileSize + threadIdx.x;
    const int firstRow = blockIdx.y * TileSize + threadIdx.y;

    #pragma unroll
    for (int r = 0; r < TileSize; r += blockRows) {
        const int row = firstRow + r;
        #pragma unroll
        for (int c = 0; c < TileSize; c += blockCols) {
            const int col = firstCol + c;
            auto& entry = matrix[static_cast<size_t>(row) * stride + col];
            const unsigned int distance =
                row < numNodes && col < numNodes ? entry : paddingDistance;
            entry = distance << distanceShift;
        }
    }
}

// A shortest path is labeled by its largest intermediate index plus one.
// Minimizing that label on distance ties gives the first scalar round that
// reaches the final distance. The label is zero for an original edge.
__device__ __forceinline__ unsigned int through(unsigned int left, unsigned int right,
                                                unsigned int pivot) {
    const unsigned int distance = (left & distanceMask) + (right & distanceMask);
    const unsigned int witness = max(pivot, max(left & witnessMask, right & witnessMask));
    return distance | witness;
}

template <int TileSize>
__global__ void closePivotTile(unsigned int* matrix, int stride, int pivotTile) {
    __shared__ unsigned int tile[TileSize][TileSize];
    const int x = threadIdx.x;
    const int y = threadIdx.y;
    const int first = pivotTile * TileSize;
    auto* const block = matrix + static_cast<size_t>(first) * stride + first;

    #pragma unroll
    for (int r = 0; r < TileSize; r += blockRows) {
        const int row = y + r;
        #pragma unroll
        for (int c = 0; c < TileSize; c += blockCols) {
            const int col = x + c;
            tile[row][col] = block[static_cast<size_t>(row) * stride + col];
        }
    }
    __syncthreads();

    for (int k = 0; k < TileSize; ++k) {
        // The pivot row and column cannot improve through their own vertex.
        // Leaving them untouched also avoids shared-memory read/write races.
        #pragma unroll
        for (int c = 0; c < TileSize; c += blockCols) {
            const int col = x + c;
            if (col != k) {
                const unsigned int right = tile[k][col];
                #pragma unroll
                for (int r = 0; r < TileSize; r += blockRows) {
                    const int row = y + r;
                    if (row != k) {
                        const unsigned int candidate = through(tile[row][k], right, first + k + 1);
                        tile[row][col] = min(tile[row][col], candidate);
                    }
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int r = 0; r < TileSize; r += blockRows) {
        const int row = y + r;
        #pragma unroll
        for (int c = 0; c < TileSize; c += blockCols) {
            const int col = x + c;
            block[static_cast<size_t>(row) * stride + col] = tile[row][col];
        }
    }
}

__device__ __forceinline__ int skipPivot(int tile, int pivotTile) {
    return tile + (tile >= pivotTile);
}

template <int TileSize, bool Border>
__global__ void relaxTiles(unsigned int* matrix, int stride, int pivotTile) {
    __shared__ unsigned int left[TileSize][TileSize];
    __shared__ unsigned int right[TileSize][TileSize];
    const int x = threadIdx.x;
    const int y = threadIdx.y;
    const int pivot = pivotTile * TileSize;
    int col = skipPivot(blockIdx.x, pivotTile) * TileSize;
    int row;
    if constexpr (Border) {
        row = pivot;
        if (blockIdx.y != 0) {
            row = col;
            col = pivot;
        }
    } else {
        row = skipPivot(blockIdx.y, pivotTile) * TileSize;
    }
    auto* const output = matrix + static_cast<size_t>(row) * stride + col;
    const auto* const leftInput = matrix + static_cast<size_t>(row) * stride + pivot;
    const auto* const rightInput = matrix + static_cast<size_t>(pivot) * stride + col;

    // The closed pivot tile lets each border tile be updated by one product
    // with its original values. The remaining tiles use the updated borders.
    constexpr int rowsPerThread = TileSize / blockRows;
    constexpr int colsPerThread = TileSize / blockCols;
    unsigned int best[rowsPerThread][colsPerThread];
    #pragma unroll
    for (int r = 0; r < rowsPerThread; ++r) {
        const int localRow = y + r * blockRows;
        #pragma unroll
        for (int c = 0; c < colsPerThread; ++c) {
            const int localCol = x + c * blockCols;
            const size_t offset = static_cast<size_t>(localRow) * stride + localCol;
            best[r][c] = output[offset];
            left[localRow][localCol] = leftInput[offset];
            right[localRow][localCol] = rightInput[offset];
        }
    }
    __syncthreads();

    #pragma unroll 8
    for (int k = 0; k < TileSize; ++k) {
        unsigned int rightDistance[colsPerThread];
        unsigned int rightWitness[colsPerThread];
        #pragma unroll
        for (int c = 0; c < colsPerThread; ++c) {
            const unsigned int entry = right[k][x + c * blockCols];
            rightDistance[c] = entry & distanceMask;
            rightWitness[c] = max(static_cast<unsigned int>(pivot + k + 1), entry & witnessMask);
        }
        #pragma unroll
        for (int r = 0; r < rowsPerThread; ++r) {
            const unsigned int entry = left[y + r * blockRows][k];
            const unsigned int distance = entry & distanceMask;
            const unsigned int witness = entry & witnessMask;
            #pragma unroll
            for (int c = 0; c < colsPerThread; ++c) {
                const unsigned int candidate = (distance + rightDistance[c])
                                             | max(witness, rightWitness[c]);
                best[r][c] = min(best[r][c], candidate);
            }
        }
    }

    #pragma unroll
    for (int r = 0; r < rowsPerThread; ++r) {
        const size_t rowOffset = static_cast<size_t>(y + r * blockRows) * stride;
        #pragma unroll
        for (int c = 0; c < colsPerThread; ++c) {
            output[rowOffset + x + c * blockCols] = best[r][c];
        }
    }
}

template <int TileSize>
double runCuda(std::vector<unsigned int>& dist, std::vector<unsigned int>& path, size_t numNodes) {
    const int tiles = (numNodes + TileSize - 1) / TileSize;
    const int stride = tiles * TileSize;
    const size_t pitch = static_cast<size_t>(stride) * sizeof(unsigned int);
    const size_t rowBytes = numNodes * sizeof(unsigned int);
    const dim3 threads(blockCols, blockRows);
    unsigned int* deviceMatrix;
    checkCuda(cudaMalloc(&deviceMatrix, stride * pitch), "Allocating CUDA distance matrix");
    checkCuda(cudaMemcpy2D(deviceMatrix, pitch, dist.data(), rowBytes, rowBytes, numNodes,
                           cudaMemcpyHostToDevice), "Uploading distance matrix");
    packDistances<TileSize><<<dim3(tiles, tiles), threads>>>(deviceMatrix, numNodes, stride);
    checkCuda(cudaGetLastError(), "Launching distance packing kernel");

    cudaEvent_t start;
    cudaEvent_t stop;
    checkCuda(cudaEventCreate(&start), "Creating CUDA start event");
    checkCuda(cudaEventCreate(&stop), "Creating CUDA stop event");
    checkCuda(cudaEventRecord(start), "Recording CUDA start event");
    for (int pivot = 0; pivot < tiles; ++pivot) {
        closePivotTile<TileSize><<<1, threads>>>(deviceMatrix, stride, pivot);
        if (tiles > 1) {
            relaxTiles<TileSize, true><<<dim3(tiles - 1, 2), threads>>>(deviceMatrix, stride, pivot);
            relaxTiles<TileSize, false><<<dim3(tiles - 1, tiles - 1), threads>>>(deviceMatrix, stride, pivot);
        }
    }
    checkCuda(cudaGetLastError(), "Launching Floyd-Warshall kernels");
    checkCuda(cudaEventRecord(stop), "Recording CUDA stop event");
    checkCuda(cudaEventSynchronize(stop), "Running Floyd-Warshall kernels");

    float milliseconds;
    checkCuda(cudaEventElapsedTime(&milliseconds, start, stop), "Measuring CUDA computation time");
    checkCuda(cudaEventDestroy(start), "Destroying CUDA start event");
    checkCuda(cudaEventDestroy(stop), "Destroying CUDA stop event");
    checkCuda(cudaMemcpy2D(dist.data(), rowBytes, deviceMatrix, pitch, rowBytes, numNodes,
                           cudaMemcpyDeviceToHost), "Downloading shortest paths");
    checkCuda(cudaFree(deviceMatrix), "Freeing CUDA distance matrix");

    for (size_t i = 0; i < dist.size(); ++i) {
        const unsigned int witness = dist[i] & witnessMask;
        dist[i] >>= distanceShift;
        if (witness != 0) {
            path[i] = witness - 1;
        }
    }
    return milliseconds;
}

} // namespace

double floydWarshall(std::vector<unsigned int>& dist, std::vector<unsigned int>& path,
                     size_t numNodes) {
    if (numNodes == 0) {
        return 0.0;
    }
    // Smaller tiles expose more blocks on small graphs. Larger tiles reuse
    // each shared-memory load across more output entries on large graphs.
    return numNodes < largeTileThreshold ? runCuda<smallTile>(dist, path, numNodes)
                                         : runCuda<largeTile>(dist, path, numNodes);
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
    
    if (numNodes > witnessMask - (largeTile - 1)) {
        fprintf(stderr, "Too many nodes for CUDA shortest-path encoding\n");
        return 1;
    }

    // Allocate matrices
    std::vector<unsigned int> dist(numNodes * numNodes);
    std::vector<unsigned int> path(numNodes * numNodes);
    
    // Initialize
    printf("Initializing graph...\n");
    initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
    initializePathMatrix(path, numNodes);
    
    // Run Floyd-Warshall
    printf("Computing shortest paths...\n");
    const double milliseconds = floydWarshall(dist, path, numNodes);
    printf("Computation time: %.3f ms\n", milliseconds);
    const double ops = static_cast<double>(numNodes) * numNodes * numNodes;
    const double gops = milliseconds > 0.0 ? ops / (milliseconds * 1e6) : 0.0;
    printf("Performance: %.3f GOPS\n", gops);
    
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
