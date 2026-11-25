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
    for (size_t row = 0; row < numNodes; ++row) {
        std::fill_n(path.data() + row * numNodes, numNodes, row);
    }
}

constexpr int TILE = 32;
constexpr int THREAD_ROWS = 8;

void checkCuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}

enum class Panel { Pivot, Row, Column };

// Close the pivot tile first, then the tiles sharing its row or column.
// Each cell saves its value when it serves as an operand. Using the final
// panel distances instead would change which k wins when paths tie.
template<Panel panel>
__global__ void updatePanel(unsigned* dist, unsigned* path, unsigned* left,
                            unsigned* top, size_t n, unsigned base) {
    __shared__ unsigned tile[TILE][TILE];
    __shared__ unsigned pivot[TILE][TILE];
    unsigned other = (blockIdx.x + (blockIdx.x >= base / TILE)) * TILE;
    unsigned rowBase = panel == Panel::Column ? other : base;
    unsigned colBase = panel == Panel::Row ? other : base;
    int x = threadIdx.x;
    unsigned col = colBase + x;
    int last[TILE / THREAD_ROWS];
    unsigned leftValue[TILE / THREAD_ROWS], topValue[TILE / THREAD_ROWS];

    #pragma unroll
    for (int r = 0; r < TILE / THREAD_ROWS; ++r) {
        int y = threadIdx.y + r * THREAD_ROWS;
        unsigned row = rowBase + y;
        tile[y][x] = row < n && col < n ? dist[row * n + col] : INF;
        last[r] = -1;
        leftValue[r] = topValue[r] = INF;
        if (panel == Panel::Row)
            pivot[y][x] = left[size_t(base + y) * TILE + x];
        if (panel == Panel::Column)
            pivot[y][x] = (base + x < n ? top[y * n + base + x] : INF);
    }
    __syncthreads();
    for (int k = 0; k < TILE; ++k) {
        #pragma unroll
        for (int r = 0; r < TILE / THREAD_ROWS; ++r) {
            int y = threadIdx.y + r * THREAD_ROWS;
            unsigned current = tile[y][x];
            if (panel != Panel::Row && x == k) leftValue[r] = current;
            if (panel != Panel::Column && y == k) topValue[r] = current;
            unsigned a = panel == Panel::Row ? pivot[y][k] : tile[y][k];
            unsigned b = panel == Panel::Column ? pivot[k][x] : tile[k][x];
            // With nonnegative edges and a zero diagonal the active
            // operand row/column never changes during this iteration.
            if (a + b < current) {
                tile[y][x] = a + b;
                last[r] = k;
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int r = 0; r < TILE / THREAD_ROWS; ++r) {
        int y = threadIdx.y + r * THREAD_ROWS;
        unsigned row = rowBase + y;
        if (panel != Panel::Row)
            left[size_t(row) * TILE + x] = leftValue[r];
        if (panel != Panel::Column && col < n)
            top[y * n + col] = topValue[r];
        if (last[r] >= 0 && row < n && col < n) {
            dist[row * n + col] = tile[y][x];
            path[row * n + col] = base + last[r];
        }
    }
}

// Each thread accumulates a rectangle of distances in registers, reusing
// each panel operand across multiple destinations or sources.
constexpr int THREAD_COLS = 8;

__global__ void updateRemaining(unsigned* dist, unsigned* path, const unsigned* left,
                                const unsigned* top, size_t n, unsigned base) {
    __shared__ unsigned a[TILE][TILE + 1];
    __shared__ unsigned b[TILE][TILE];
    unsigned rowBase = (blockIdx.y + (blockIdx.y >= base / TILE)) * TILE;
    unsigned colBase = (blockIdx.x + (blockIdx.x >= base / TILE)) * TILE;
    constexpr int rows = TILE / THREAD_ROWS;
    constexpr int cols = TILE / THREAD_COLS;
    unsigned best[rows][cols];
    int last[rows][cols];
    #pragma unroll
    for (int r = 0; r < rows; ++r) {
        int y = threadIdx.y + r * THREAD_ROWS;
        unsigned row = rowBase + y;
        #pragma unroll
        for (int c = 0; c < cols; ++c) {
            int x = threadIdx.x + c * THREAD_COLS;
            unsigned col = colBase + x;
            a[y][x] = left[size_t(row) * TILE + x];
            b[y][x] = col < n ? top[y * n + col] : INF;
            best[r][c] = row < n && col < n ? dist[row * n + col] : INF;
            last[r][c] = -1;
        }
    }
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < TILE; ++k) {
        #pragma unroll
        for (int r = 0; r < rows; ++r) {
            #pragma unroll
            for (int c = 0; c < cols; ++c) {
                unsigned via = a[threadIdx.y + r * THREAD_ROWS][k]
                             + b[k][threadIdx.x + c * THREAD_COLS];
                if (via < best[r][c]) {
                    best[r][c] = via;
                    last[r][c] = k;
                }
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < rows; ++r) {
        unsigned row = rowBase + threadIdx.y + r * THREAD_ROWS;
        #pragma unroll
        for (int c = 0; c < cols; ++c) {
            unsigned col = colBase + threadIdx.x + c * THREAD_COLS;
            if (last[r][c] >= 0 && row < n && col < n) {
                dist[row * n + col] = best[r][c];
                path[row * n + col] = base + last[r][c];
            }
        }
    }
}

void floydWarshall(std::vector<unsigned>& dist, std::vector<unsigned>& path, size_t n) {
    if (n == 0) return;
    unsigned tiles = (n + TILE - 1) / TILE;
    size_t bytes = dist.size() * sizeof(unsigned);
    size_t panelSize = size_t(tiles) * TILE * TILE;
    unsigned *deviceDist, *devicePath, *panels;
    checkCuda(cudaMalloc(&deviceDist, bytes), "Allocate distances");
    checkCuda(cudaMalloc(&devicePath, bytes), "Allocate paths");
    checkCuda(cudaMalloc(&panels, 2 * panelSize * sizeof(unsigned)), "Allocate panels");
    checkCuda(cudaMemcpy(deviceDist, dist.data(), bytes, cudaMemcpyHostToDevice), "Upload distances");
    checkCuda(cudaMemcpy(devicePath, path.data(), bytes, cudaMemcpyHostToDevice), "Upload paths");
    unsigned* top = panels + panelSize;
    dim3 threads(TILE, THREAD_ROWS);
    for (unsigned t = 0; t < tiles; ++t) {
        unsigned base = t * TILE;
        updatePanel<Panel::Pivot><<<1, threads>>>(deviceDist, devicePath, panels, top, n, base);
        if (tiles > 1) {
            updatePanel<Panel::Row><<<tiles - 1, threads>>>(deviceDist, devicePath, panels, top, n, base);
            updatePanel<Panel::Column><<<tiles - 1, threads>>>(deviceDist, devicePath, panels, top, n, base);
            updateRemaining<<<dim3(tiles - 1, tiles - 1), dim3(THREAD_COLS, THREAD_ROWS)>>>(deviceDist, devicePath, panels, top, n, base);
        }
        checkCuda(cudaGetLastError(), "Launch Floyd-Warshall kernels");
    }
    checkCuda(cudaMemcpy(dist.data(), deviceDist, bytes, cudaMemcpyDeviceToHost), "Download distances");
    checkCuda(cudaMemcpy(path.data(), devicePath, bytes, cudaMemcpyDeviceToHost), "Download paths");
    checkCuda(cudaFree(panels), "Free panels");
    checkCuda(cudaFree(devicePath), "Free paths");
    checkCuda(cudaFree(deviceDist), "Free distances");
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
    checkCuda(cudaFree(nullptr), "Initialize CUDA");
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
