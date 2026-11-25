#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <climits>
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
    for (size_t row = 0; row < numNodes; ++row) {
        std::fill_n(path.data() + row * numNodes, numNodes, row);
    }
}

constexpr int TILE = 64;
constexpr int THREAD_ROWS = 8;

void checkCuda(cudaError_t status) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

// The pivot block closes its own tile first. The panel blocks then close
// the other tiles in the pivot row and column using the saved pivot history.
template<bool Pivot>
__global__ void updatePanel(unsigned* dist, unsigned* path, unsigned* left,
                            unsigned* right, size_t stride, unsigned pivot) {
    __shared__ unsigned tile[TILE][TILE];
    __shared__ unsigned history[TILE][TILE];
    const bool rowPanel = !Pivot && blockIdx.y == 0;
    const unsigned other = blockIdx.x + (blockIdx.x >= pivot);
    const unsigned rowBase = (Pivot || rowPanel ? pivot : other) * TILE;
    const unsigned colBase = (Pivot || !rowPanel ? pivot : other) * TILE;
    const unsigned pivotBase = pivot * TILE;
    const int x = threadIdx.x;
    unsigned via[TILE / THREAD_ROWS];

    #pragma unroll
    for (int t = 0; t < TILE / THREAD_ROWS; ++t) {
        const int y = threadIdx.y + t * THREAD_ROWS;
        tile[y][x] = dist[(rowBase + y) * stride + colBase + x];
        via[t] = UINT_MAX;
        if (!Pivot) {
            history[y][x] = rowPanel ? left[size_t(pivotBase + y) * TILE + x]
                                     : right[y * stride + pivotBase + x];
        }
    }
    __syncthreads();

    for (int k = 0; k < TILE; ++k) {
        const unsigned fromPivot = (Pivot || rowPanel) ? tile[k][x] : history[k][x];
        #pragma unroll
        for (int t = 0; t < TILE / THREAD_ROWS; ++t) {
            const int y = threadIdx.y + t * THREAD_ROWS;
            const unsigned toPivot = (Pivot || !rowPanel) ? tile[y][k] : history[y][k];
            if ((Pivot || rowPanel) && y == k)
                right[k * stride + colBase + x] = fromPivot;
            if ((Pivot || !rowPanel) && x == k)
                left[size_t(rowBase + y) * TILE + k] = toPivot;

            // Active pivot operands stay read-only throughout this iteration.
            if (((Pivot || rowPanel) && y == k) || ((Pivot || !rowPanel) && x == k))
                continue;
            const unsigned candidate = toPivot + fromPivot;
            if (candidate < tile[y][x]) {
                tile[y][x] = candidate;
                via[t] = pivotBase + k;
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int t = 0; t < TILE / THREAD_ROWS; ++t) {
        const int y = threadIdx.y + t * THREAD_ROWS;
        const size_t index = (rowBase + y) * stride + colBase + x;
        dist[index] = tile[y][x];
        if (via[t] != UINT_MAX) path[index] = via[t];
    }
}

constexpr int OUTPUT_COLS = 16;

__global__ void updateRemaining(unsigned* dist, unsigned* path, const unsigned* left,
                                const unsigned* right, size_t stride, unsigned pivot) {
    __shared__ unsigned sources[TILE][TILE];
    __shared__ unsigned targets[TILE][TILE];
    const unsigned rowBase = (blockIdx.y + (blockIdx.y >= pivot)) * TILE;
    const unsigned colBase = (blockIdx.x + (blockIdx.x >= pivot)) * TILE;

    // Rank zero preserves the old path, and ranks 1..64 select the first
    // pivot attaining the minimum. Compare both fields with one integer min.
    constexpr unsigned rankBits = 7;
    constexpr unsigned rankMask = (1u << rankBits) - 1;
    constexpr unsigned unreachable = (1u << (31 - rankBits)) - 1;
    unsigned best[TILE / THREAD_ROWS][TILE / OUTPUT_COLS];
    #pragma unroll
    for (int i = 0; i < TILE / THREAD_ROWS; ++i) {
        const int y = threadIdx.y + i * THREAD_ROWS;
        #pragma unroll
        for (int j = 0; j < TILE / OUTPUT_COLS; ++j) {
            const int x = threadIdx.x + j * OUTPUT_COLS;
            sources[y][x] = min(left[size_t(rowBase + y) * TILE + x], unreachable) << rankBits;
            targets[y][x] = (min(right[y * stride + colBase + x], unreachable) << rankBits) | (y + 1);
            best[i][j] = min(dist[(rowBase + y) * stride + colBase + x], unreachable) << rankBits;
        }
    }
    __syncthreads();
    #pragma unroll 4
    for (int k = 0; k < TILE; ++k) {
        unsigned fromPivot[TILE / OUTPUT_COLS];
        #pragma unroll
        for (int j = 0; j < TILE / OUTPUT_COLS; ++j)
            fromPivot[j] = targets[k][threadIdx.x + j * OUTPUT_COLS];
        #pragma unroll
        for (int i = 0; i < TILE / THREAD_ROWS; ++i) {
            const unsigned toPivot = sources[threadIdx.y + i * THREAD_ROWS][k];
            #pragma unroll
            for (int j = 0; j < TILE / OUTPUT_COLS; ++j)
                best[i][j] = min(best[i][j], toPivot + fromPivot[j]);
        }
    }

    #pragma unroll
    for (int i = 0; i < TILE / THREAD_ROWS; ++i) {
        const size_t row = rowBase + threadIdx.y + i * THREAD_ROWS;
        #pragma unroll
        for (int j = 0; j < TILE / OUTPUT_COLS; ++j) {
            const unsigned rank = best[i][j] & rankMask;
            if (rank) {
                const size_t index = row * stride + colBase + threadIdx.x + j * OUTPUT_COLS;
                dist[index] = best[i][j] >> rankBits;
                path[index] = pivot * TILE + rank - 1;
            }
        }
    }
}

void floydWarshall(std::vector<unsigned int>& dist, std::vector<unsigned int>& path,
                   const size_t numNodes) {
    if (!numNodes) return;
    const unsigned tiles = (numNodes + TILE - 1) / TILE;
    const size_t stride = size_t(tiles) * TILE;
    const size_t pitch = stride * sizeof(unsigned);
    const size_t width = numNodes * sizeof(unsigned);
    unsigned *deviceDist, *devicePath, *left, *right;
    checkCuda(cudaMalloc(&deviceDist, stride * pitch));
    checkCuda(cudaMalloc(&devicePath, stride * pitch));
    checkCuda(cudaMalloc(&left, TILE * pitch));
    checkCuda(cudaMalloc(&right, TILE * pitch));
    // Padding exceeds every generated edge weight.
    if (stride != numNodes) checkCuda(cudaMemset(deviceDist, 0x01, stride * pitch));
    checkCuda(cudaMemcpy2D(deviceDist, pitch, dist.data(), width, width, numNodes, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy2D(devicePath, pitch, path.data(), width, width, numNodes, cudaMemcpyHostToDevice));

    const dim3 threads(TILE, THREAD_ROWS);
    for (unsigned pivot = 0; pivot < tiles; ++pivot) {
        updatePanel<true><<<1, threads>>>(deviceDist, devicePath, left, right, stride, pivot);
        if (tiles > 1) {
            updatePanel<false><<<dim3(tiles - 1, 2), threads>>>(deviceDist, devicePath, left, right, stride, pivot);
            updateRemaining<<<dim3(tiles - 1, tiles - 1), dim3(OUTPUT_COLS, THREAD_ROWS)>>>(deviceDist, devicePath, left, right, stride, pivot);
        }
    }
    checkCuda(cudaGetLastError());
    checkCuda(cudaMemcpy2D(dist.data(), width, deviceDist, pitch, width, numNodes, cudaMemcpyDeviceToHost));
    checkCuda(cudaMemcpy2D(path.data(), width, devicePath, pitch, width, numNodes, cudaMemcpyDeviceToHost));
    checkCuda(cudaFree(right));
    checkCuda(cudaFree(left));
    checkCuda(cudaFree(devicePath));
    checkCuda(cudaFree(deviceDist));
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
    
    // Initialize the CUDA context before timing the solver.
    checkCuda(cudaFree(nullptr));

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
