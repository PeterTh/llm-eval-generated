#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

// Index calculation for flattened 2D array
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error: %s (%s:%d)\n",                         \
                    cudaGetErrorString(err_), __FILE__, __LINE__);              \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

constexpr int TILE = 32;
constexpr int BLOCK_ROWS = 8;
constexpr int BLOCK_THREADS = TILE * BLOCK_ROWS;

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

__global__ void floydWarshallKernel(unsigned int* __restrict__ dist,
                                    unsigned int* __restrict__ path,
                                    const int n) {
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();

    __shared__ unsigned int colk[TILE];
    __shared__ unsigned int rowk[TILE];

    const int tilesPerDim = (n + TILE - 1) / TILE;
    const int totalTiles = tilesPerDim * tilesPerDim;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    for (int k = 0; k < n; ++k) {
        for (int tile = blockIdx.x; tile < totalTiles; tile += gridDim.x) {
            const int bi = tile / tilesPerDim;
            const int bj = tile - bi * tilesPerDim;
            const int rowStart = bi * TILE;
            const int colStart = bj * TILE;

            if (ty == 0) {
                const int gi = rowStart + tx;
                const int gj = colStart + tx;
                colk[tx] = (gi < n) ? dist[(size_t)k * n + gi] : INF;
                rowk[tx] = (gj < n) ? dist[(size_t)gj * n + k] : INF;
            }
            __syncthreads();

            const bool fullTile = (rowStart + TILE <= n) && (colStart + TILE <= n);
            if (fullTile) {
                const unsigned int ck = colk[tx];
                const int gi = rowStart + tx;
                #pragma unroll
                for (int jj = ty; jj < TILE; jj += BLOCK_ROWS) {
                    const int gj = colStart + jj;
                    const size_t pos = (size_t)gj * n + gi;
                    const unsigned int newDist = ck + rowk[jj];
                    const unsigned int oldDist = dist[pos];
                    if (newDist < oldDist) {
                        dist[pos] = newDist;
                        path[pos] = static_cast<unsigned int>(k);
                    }
                }
            } else {
                const int gi = rowStart + tx;
                if (gi < n) {
                    const unsigned int ck = colk[tx];
                    for (int jj = ty; jj < TILE; jj += BLOCK_ROWS) {
                        const int gj = colStart + jj;
                        if (gj < n) {
                            const size_t pos = (size_t)gj * n + gi;
                            const unsigned int newDist = ck + rowk[jj];
                            const unsigned int oldDist = dist[pos];
                            if (newDist < oldDist) {
                                dist[pos] = newDist;
                                path[pos] = static_cast<unsigned int>(k);
                            }
                        }
                    }
                }
            }
            __syncthreads();
        }
        grid.sync();
    }
}

void floydWarshall(unsigned int* d_dist, unsigned int* d_path, const size_t numNodes) {
    if (numNodes == 0) {
        return;
    }

    int n = static_cast<int>(numNodes);
    const int tilesPerDim = (n + TILE - 1) / TILE;
    const long long totalTiles = static_cast<long long>(tilesPerDim) * tilesPerDim;

    static int cachedDevice = -1;
    static int maxBlocks = 0;
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    if (cachedDevice != device) {
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
        int blocksPerSM = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocksPerSM, floydWarshallKernel, BLOCK_THREADS, 0));
        if (blocksPerSM < 1) {
            blocksPerSM = 1;
        }
        maxBlocks = prop.multiProcessorCount * blocksPerSM;
        cachedDevice = device;
    }

    long long gridBlocks = maxBlocks;
    if (gridBlocks > totalTiles) {
        gridBlocks = totalTiles;
    }
    if (gridBlocks < 1) {
        gridBlocks = 1;
    }

    void* kernelArgs[] = {
        reinterpret_cast<void*>(&d_dist),
        reinterpret_cast<void*>(&d_path),
        reinterpret_cast<void*>(&n)
    };

    CUDA_CHECK(cudaLaunchCooperativeKernel(
        reinterpret_cast<void*>(floydWarshallKernel),
        dim3(static_cast<unsigned int>(gridBlocks)),
        dim3(TILE, BLOCK_ROWS),
        kernelArgs,
        0,
        0));
    CUDA_CHECK(cudaDeviceSynchronize());
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

    const size_t matrixBytes = numNodes * numNodes * sizeof(unsigned int);
    unsigned int* d_dist = nullptr;
    unsigned int* d_path = nullptr;
    if (numNodes > 0) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_dist), matrixBytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_path), matrixBytes));
        CUDA_CHECK(cudaMemcpy(d_dist, dist.data(), matrixBytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_path, path.data(), matrixBytes, cudaMemcpyHostToDevice));
    }
    
    // Run Floyd-Warshall
    printf("Computing shortest paths...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    floydWarshall(d_dist, d_path, numNodes);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    auto durationUs = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    if (numNodes > 0) {
        CUDA_CHECK(cudaMemcpy(dist.data(), d_dist, matrixBytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(path.data(), d_path, matrixBytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_dist));
        CUDA_CHECK(cudaFree(d_path));
    }
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate operations per second
    // Floyd-Warshall has O(n³) complexity
    double ops = (double)numNodes * numNodes * numNodes;
    const double seconds = static_cast<double>(durationUs.count()) / 1e6;
    double gflops = (seconds > 0.0) ? ops / seconds / 1e9 : 0.0;
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
