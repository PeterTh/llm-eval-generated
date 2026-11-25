#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

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
constexpr int Tile = 32;
constexpr int BlockRows = 8;
constexpr int Rows = Tile / BlockRows;

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

enum class Panel { Pivot, Column, Row };

// Each panel emits the pivot operands as they stood at step k. Using a
// fully relaxed panel here could select a later intermediate on equal paths.
template<Panel Kind>
__global__ void relaxPanel(unsigned* dist, unsigned* path, size_t stride, int pivot,
                           unsigned* columns, unsigned* rows) {
    const int other = blockIdx.x + (static_cast<int>(blockIdx.x) >= pivot);
    const int tileRow = Kind == Panel::Column ? other : pivot;
    const int tileCol = Kind == Panel::Row ? other : pivot;
    const int x = threadIdx.x;
    const int y = threadIdx.y;
    const size_t base = (size_t(tileRow) * Tile + y) * stride + tileCol * Tile + x;
    const int history = pivot * Tile * Tile;
    __shared__ unsigned work[Tile][Tile];
    __shared__ unsigned diagonal[Tile][Tile];
    __shared__ unsigned colHistory[Tile][Tile];
    __shared__ unsigned rowHistory[Tile][Tile];
    unsigned via[Rows];
    #pragma unroll
    for (int i = 0; i < Rows; ++i) {
        int r = y + i * BlockRows;
        size_t pos = base + i * BlockRows * stride;
        work[r][x] = dist[pos];
        via[i] = path[pos];
        if constexpr (Kind == Panel::Column) diagonal[r][x] = rows[history + r * Tile + x];
        if constexpr (Kind == Panel::Row) diagonal[r][x] = columns[history + r * Tile + x];
    }
    __syncthreads();
    for (int k = 0; k < Tile; ++k) {
        #pragma unroll
        for (int i = 0; i < Rows; ++i) {
            int r = y + i * BlockRows;
            unsigned left, right;
            if constexpr (Kind == Panel::Row) left = diagonal[r][k];
            else left = work[r][k];
            if constexpr (Kind == Panel::Column) right = diagonal[k][x];
            else right = work[k][x];
            if constexpr (Kind != Panel::Row) {
                if (x == k) colHistory[r][x] = left;
            }
            if constexpr (Kind != Panel::Column) {
                if (r == k) rowHistory[r][x] = right;
            }
            // The active row/column cannot improve through its zero diagonal,
            // so these operands remain read-only for the whole block step.
            if (left + right < work[r][x]) {
                work[r][x] = left + right;
                via[i] = pivot * Tile + k;
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < Rows; ++i) {
        int r = y + i * BlockRows;
        size_t pos = base + i * BlockRows * stride;
        dist[pos] = work[r][x];
        path[pos] = via[i];
        if constexpr (Kind != Panel::Row)
            columns[(tileRow * Tile + r) * Tile + x] = colHistory[r][x];
        if constexpr (Kind != Panel::Column)
            rows[(tileCol * Tile + r) * Tile + x] = rowHistory[r][x];
    }
}

__global__ void relaxRemaining(unsigned* dist, unsigned* path, size_t stride, int pivot,
                               const unsigned* columns, const unsigned* rows) {
    const int tileRow = blockIdx.y + (static_cast<int>(blockIdx.y) >= pivot);
    const int tileCol = blockIdx.x + (static_cast<int>(blockIdx.x) >= pivot);
    const int x = threadIdx.x;
    const int y = threadIdx.y;
    const size_t base = (size_t(tileRow) * Tile + y) * stride + tileCol * Tile + x;
    __shared__ unsigned left[Tile][Tile];
    __shared__ unsigned right[Tile][Tile];
    unsigned best[Rows], via[Rows];
    #pragma unroll
    for (int i = 0; i < Rows; ++i) {
        int r = y + i * BlockRows;
        left[r][x] = columns[(tileRow * Tile + r) * Tile + x];
        right[r][x] = rows[(tileCol * Tile + r) * Tile + x];
        best[i] = dist[base + i * BlockRows * stride];
        via[i] = path[base + i * BlockRows * stride];
    }
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < Tile; ++k) {
        unsigned kj = right[k][x];
        #pragma unroll
        for (int i = 0; i < Rows; ++i) {
            unsigned candidate = left[y + i * BlockRows][k] + kj;
            if (candidate < best[i]) {
                best[i] = candidate;
                via[i] = pivot * Tile + k;
            }
        }
    }
    #pragma unroll
    for (int i = 0; i < Rows; ++i) {
        dist[base + i * BlockRows * stride] = best[i];
        path[base + i * BlockRows * stride] = via[i];
    }
}
} // namespace

void floydWarshall(std::vector<unsigned>& dist, std::vector<unsigned>& path, size_t numNodes) {
    if (numNodes == 0) return;
    const int tiles = (numNodes + Tile - 1) / Tile;
    const size_t stride = size_t(tiles) * Tile;
    const size_t pitch = stride * sizeof(unsigned);
    const size_t width = numNodes * sizeof(unsigned);
    unsigned *dDist, *dPath, *history;
    checkCuda(cudaMalloc(&dDist, stride * pitch), "Allocate distances");
    checkCuda(cudaMalloc(&dPath, stride * pitch), "Allocate paths");
    checkCuda(cudaMalloc(&history, 2 * Tile * pitch), "Allocate pivot history");
    unsigned* rows = history + Tile * stride;
    if (stride != numNodes) {
        // Isolate padded vertices. Two padding distances still fit in unsigned.
        checkCuda(cudaMemset(dDist, 0x3f, stride * pitch), "Pad distances");
        checkCuda(cudaMemset(dPath, 0, stride * pitch), "Pad paths");
    }
    checkCuda(cudaMemcpy2D(dDist, pitch, dist.data(), width, width, numNodes,
                           cudaMemcpyHostToDevice), "Upload distances");
    checkCuda(cudaMemcpy2D(dPath, pitch, path.data(), width, width, numNodes,
                           cudaMemcpyHostToDevice), "Upload paths");

    const dim3 block(Tile, BlockRows);
    for (int pivot = 0; pivot < tiles; ++pivot) {
        relaxPanel<Panel::Pivot><<<1, block>>>(dDist, dPath, stride, pivot, history, rows);
        if (tiles > 1) {
            relaxPanel<Panel::Column><<<tiles - 1, block>>>(dDist, dPath, stride, pivot, history, rows);
            relaxPanel<Panel::Row><<<tiles - 1, block>>>(dDist, dPath, stride, pivot, history, rows);
            relaxRemaining<<<dim3(tiles - 1, tiles - 1), block>>>(dDist, dPath, stride, pivot, history, rows);
        }
    }
    checkCuda(cudaGetLastError(), "Launch Floyd-Warshall kernels");
    checkCuda(cudaMemcpy2D(dist.data(), width, dDist, pitch, width, numNodes,
                           cudaMemcpyDeviceToHost), "Download distances");
    checkCuda(cudaMemcpy2D(path.data(), width, dPath, pitch, width, numNodes,
                           cudaMemcpyDeviceToHost), "Download paths");
    checkCuda(cudaFree(history), "Free pivot history");
    checkCuda(cudaFree(dPath), "Free paths");
    checkCuda(cudaFree(dDist), "Free distances");
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
    checkCuda(cudaFree(nullptr), "Initialize CUDA runtime");
    auto start = std::chrono::steady_clock::now();
    
    floydWarshall(dist, path, numNodes);
    
    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate operations per second
    // Floyd-Warshall has O(n³) complexity
    double ops = (double)numNodes * numNodes * numNodes;
    double gflops = ops / std::chrono::duration<double>(end - start).count() / 1e9;
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
