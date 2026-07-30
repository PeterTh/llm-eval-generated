#include <algorithm>
#include <chrono>
#include <cmath>
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

// CUDA kernel for Floyd-Warshall
// For a fixed intermediate node k, each thread processes one (i, j) pair.
// Uses shared memory to cache the k-th row (dist[k][*]) and k-th column (dist[*][k])
// for coalesced memory access patterns.
template <unsigned int TILE>
__global__ void floydWarshallKernel(unsigned int* dist, unsigned int* path,
                                    const size_t n, const unsigned int k) {
    // Shared memory tiles for the k-th row and k-th column
    __shared__ unsigned int rowK[TILE];   // dist[k][col] for this tile
    __shared__ unsigned int colK[TILE];   // dist[row][k] for this tile

    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;

    const size_t i = blockIdx.y * TILE + ty;  // row index
    const size_t j = blockIdx.x * TILE + tx;  // col index

    // Load k-th row and k-th column into shared memory (coalesced)
    if (i < n && j < n) {
        rowK[tx] = dist[k * n + j];       // dist[k][j] — coalesced across tx
        colK[ty] = dist[i * n + k];       // dist[i][k] — coalesced across ty
    }
    __syncthreads();

    // Each thread computes the relaxation for its (i, j) pair
    if (i < n && j < n) {
        const unsigned int newDist = colK[ty] + rowK[tx];
        const unsigned int distIJ = dist[i * n + j];

        if (newDist < distIJ) {
            dist[i * n + j] = newDist;
            path[i * n + j] = k;
        }
    }
}

// Kernel launch helper: selects tile size and grid/block dimensions
static void launchFloydWarshallKernel(unsigned int* d_dist, unsigned int* d_path,
                                      const size_t n, const unsigned int k) {
    // Use 32x32 tile for optimal occupancy on modern GPUs
    constexpr unsigned int TILE = 32;
    dim3 block(TILE, TILE);
    dim3 grid((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);

    floydWarshallKernel<TILE><<<grid, block>>>(d_dist, d_path, n, k);
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

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes) {
    unsigned int* d_dist = nullptr;
    unsigned int* d_path = nullptr;
    const size_t nbytes = numNodes * numNodes * sizeof(unsigned int);

    // Allocate device memory
    cudaMalloc(&d_dist, nbytes);
    cudaMalloc(&d_path, nbytes);

    // Copy host data to device
    cudaMemcpy(d_dist, dist.data(), nbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_path, path.data(), nbytes, cudaMemcpyHostToDevice);

    // Sequential k-loop with parallel (i, j) computation per iteration
    for (unsigned int k = 0; k < numNodes; ++k) {
        launchFloydWarshallKernel(d_dist, d_path, numNodes, k);
    }

    // Synchronize to ensure all kernel work is complete
    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy(dist.data(), d_dist, nbytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(path.data(), d_path, nbytes, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_dist);
    cudaFree(d_path);
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
    auto start = std::chrono::high_resolution_clock::now();

    floydWarshall(dist, path, numNodes);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Computation time: %ld ms\n", duration.count());

    // Calculate operations per second
    // Floyd-Warshall has O(n^3) complexity
    double ops = (double)numNodes * numNodes * numNodes;
    double gflops = ops / (duration.count() / 1000.0) / 1e9;
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
