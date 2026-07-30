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

// Index calculation for flattened 2D array: idx2(i, j, n) = j * n + i
// (column-major layout: i is row, j is column)
inline __host__ __device__ constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

// ============================================================================
// CUDA Kernels
// ============================================================================

// Initialize distance matrix on GPU
// Each thread computes one element using a deterministic LCG
__global__ void initDistanceKernel(unsigned int* dist, size_t n, unsigned int rangeMin, unsigned int rangeMax) {
    size_t tid = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + static_cast<size_t>(threadIdx.x);
    size_t total = n * n;

    if (tid < total) {
        // Deterministic LCG matching original rand_r(seed=42) behavior
        unsigned int seed = 42;
        for (size_t step = 0; step <= tid; ++step) {
            seed = seed * 1103515245u + 12345u;
        }
        unsigned int r = (seed >> 16) & 0x7fff;
        unsigned int val = rangeMin + (r % (rangeMax - rangeMin + 1));
        dist[tid] = val;
    }
}

// Set diagonal of distance matrix to zero
__global__ void zeroDiagonalKernel(unsigned int* dist, size_t n) {
    size_t i = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + static_cast<size_t>(threadIdx.x);
    if (i < n) {
        dist[idx2(i, i, n)] = 0;
    }
}

// Initialize path matrix: path[i][j] = j for all (i, j)
__global__ void initPathKernel(unsigned int* path, size_t n) {
    size_t tid = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + static_cast<size_t>(threadIdx.x);
    size_t total = n * n;

    if (tid < total) {
        size_t i = tid / n;
        size_t j = tid % n;
        path[idx2(i, j, n)] = static_cast<unsigned int>(j);
    }
}

// Floyd-Warshall kernel for a single k iteration
// Uses shared memory to cache the k-th row and k-th column,
// dramatically reducing global memory traffic per thread.
__global__ void floydWarshallKernel(unsigned int* dist, unsigned int* path,
                                    size_t n, unsigned int k) {
    // Shared memory: two arrays of size n for k-th row and k-th column
    extern __shared__ unsigned int shared[];
    unsigned int* sRowK = shared;       // sRowK[j] = dist[k][j]
    unsigned int* sColK = &shared[n];   // sColK[i] = dist[i][k]

    // Use full 2D thread index within block for shared memory load
    // Loop to handle cases where n > threads_per_block
    size_t threadsPerBlock = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockDim.y);
    size_t tid = static_cast<size_t>(threadIdx.x) +
                 static_cast<size_t>(threadIdx.y) * static_cast<size_t>(blockDim.x);

    // Cooperative load: each thread loads one or more elements of k-th row and k-th column
    for (size_t t = tid; t < n; t += threadsPerBlock) {
        sRowK[t] = dist[idx2(k, t, n)];   // dist[k][t]
        sColK[t] = dist[idx2(t, k, n)];   // dist[t][k]
    }
    __syncthreads();

    // Each thread computes one (i, j) pair
    size_t i = static_cast<size_t>(blockIdx.y) * static_cast<size_t>(blockDim.y) + static_cast<size_t>(threadIdx.y);
    size_t j = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + static_cast<size_t>(threadIdx.x);

    if (i < n && j < n) {
        unsigned int newDist = sColK[i] + sRowK[j];  // dist[i][k] + dist[k][j]
        size_t idx = idx2(i, j, n);
        if (newDist < dist[idx]) {
            dist[idx] = newDist;
            path[idx] = k;
        }
    }
}

// ============================================================================
// Host-side validation
// ============================================================================

bool validateResult(const std::vector<unsigned int>& dist, const size_t numNodes) {
    // 1. Diagonal should be zero
    for (size_t i = 0; i < numNodes; ++i) {
        if (dist[idx2(i, i, numNodes)] != 0) {
            printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", i, i);
            return false;
        }
    }

    // 2. Triangle inequality: dist[i][k] + dist[k][j] >= dist[i][j]
    // Check a sample to avoid O(n^3) validation time
    for (size_t i = 0; i < std::min(numNodes, static_cast<size_t>(10)); ++i) {
        for (size_t j = 0; j < std::min(numNodes, static_cast<size_t>(10)); ++j) {
            for (size_t k = 0; k < numNodes; ++k) {
                const unsigned int distIJ = dist[idx2(j, i, numNodes)];
                const unsigned int distIK = dist[idx2(k, i, numNodes)];
                const unsigned int distKJ = dist[idx2(j, k, numNodes)];

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

// ============================================================================
// Host utilities
// ============================================================================

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Number of nodes in the graph (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

// ============================================================================
// Main
// ============================================================================

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

    // Host buffers
    std::vector<unsigned int> dist(numNodes * numNodes);
    std::vector<unsigned int> path(numNodes * numNodes);

    // Device buffers
    unsigned int *d_dist, *d_path;
    size_t matBytes = numNodes * numNodes * sizeof(unsigned int);

    cudaMalloc(&d_dist, matBytes);
    cudaMalloc(&d_path, matBytes);

    // GPU initialization
    printf("Initializing graph...\n");
    {
        unsigned int blockSize = 256;
        unsigned int gridSize = static_cast<unsigned int>((numNodes * numNodes + blockSize - 1) / blockSize);

        initDistanceKernel<<<gridSize, blockSize>>>(d_dist, numNodes, 1, MAX_DISTANCE);
        zeroDiagonalKernel<<<(numNodes + 255) / 256, 256>>>(d_dist, numNodes);
        initPathKernel<<<gridSize, blockSize>>>(d_path, numNodes);

        cudaMemcpy(dist.data(), d_dist, matBytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(path.data(), d_path, matBytes, cudaMemcpyDeviceToHost);
    }

    // Floyd-Warshall on GPU
    printf("Computing shortest paths...\n");
    auto start = std::chrono::high_resolution_clock::now();

    {
        // 2D block: blockDim.x threads for j-dimension, blockDim.y for i-dimension
        dim3 blockDim(32, 16);
        dim3 gridDim(
            static_cast<unsigned int>((numNodes + blockDim.x - 1) / blockDim.x),
            static_cast<unsigned int>((numNodes + blockDim.y - 1) / blockDim.y)
        );
        size_t sharedMemBytes = 2 * numNodes * sizeof(unsigned int);

        // Sequential k-loop with parallelized inner loops
        for (size_t k = 0; k < numNodes; ++k) {
            floydWarshallKernel<<<gridDim, blockDim, sharedMemBytes>>>(
                d_dist, d_path, numNodes, static_cast<unsigned int>(k));
            cudaDeviceSynchronize();
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Computation time: %ld ms\n", duration.count());

    // Calculate operations per second
    double ops = static_cast<double>(numNodes) * numNodes * numNodes;
    double gflops = ops / (duration.count() / 1000.0) / 1e9;
    printf("Performance: %.3f GOPS\n", gflops);

    // Copy results back
    cudaMemcpy(dist.data(), d_dist, matBytes, cudaMemcpyDeviceToHost);

    // Print results for external validation
    if (printResults) {
        print_results_int(dist, "DistanceMatrix");
    }

    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(dist, numNodes);

        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
        }

        // Cleanup device memory
        cudaFree(d_dist);
        cudaFree(d_path);

        return valid ? 0 : 1;
    }

    // Cleanup device memory
    cudaFree(d_dist);
    cudaFree(d_path);

    return 0;
}
