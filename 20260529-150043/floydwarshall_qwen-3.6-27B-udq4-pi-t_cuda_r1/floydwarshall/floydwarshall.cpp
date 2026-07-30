#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess) {                                                \
            printf("CUDA error at %s:%d: %s (%s)\n", __FILE__, __LINE__,         \
                   cudaGetErrorString(err), #call);                              \
            exit(1);                                                             \
        }                                                                        \
    } while (0)

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;
constexpr int TILE_SIZE = 32;

// Column-major index: element at row i, column j is at position j*n + i
// Convention: dist[row][col] = shortest path from col to row
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

// CUDA kernel: tiled approach with shared memory.
// For a fixed intermediate k, each block processes a TILE_SIZE x TILE_SIZE tile.
// Uses a 1D thread block with grid-stride loop for scalability.
//
// Memory layout (column-major): dist[row][col] is at position col*n + row
// FW relaxation: dist[j][i] = min(dist[j][i], dist[k][i] + dist[j][k])
//   dist[j][i] at position i*n + j = idx2(j, i, n)
//   dist[k][i] at position i*n + k = idx2(k, i, n)
//   dist[j][k] at position k*n + j = idx2(j, k, n)
//
// Shared memory:
//   col_k[ty] = dist[k][i] = idx2(k, i, n)  (one per row in tile)
//   row_k[tx] = dist[j][k] = idx2(j, k, n)  (one per col in tile)
//
// Bank conflict optimization: col_k is broadcast within a warp (free in CUDA),
// row_k has one access per bank per warp (no conflict).
__global__ void floydWarshallKernel(unsigned int* dist, unsigned int* path,
                                    size_t n, size_t k) {
    extern __shared__ unsigned int shared[];
    unsigned int* col_k = shared;
    unsigned int* row_k = shared + TILE_SIZE;

    unsigned int tid = threadIdx.x;
    size_t tileJ = (size_t)blockIdx.x * TILE_SIZE;
    size_t tileI = (size_t)blockIdx.y * TILE_SIZE;

    // Load dist[k][i] and dist[j][k] into shared memory
    if (tid < TILE_SIZE) {
        size_t j = tileJ + tid;
        if (j < n) {
            row_k[tid] = dist[idx2(j, k, n)];
        }
    }
    if (tid < TILE_SIZE) {
        size_t i = tileI + tid;
        if (i < n) {
            col_k[tid] = dist[idx2(k, i, n)];
        }
    }
    __syncthreads();

    // Each thread processes one (i,j) pair in the tile via grid-stride loop
    unsigned int tileTotal = TILE_SIZE * TILE_SIZE;
    for (unsigned int t = tid; t < tileTotal; t += blockDim.x) {
        unsigned int ty = t / TILE_SIZE;
        unsigned int tx = t % TILE_SIZE;
        size_t i = tileI + ty;
        size_t j = tileJ + tx;

        if (i < n && j < n) {
            const unsigned int distJI = dist[idx2(j, i, n)];
            const unsigned int newDist = col_k[ty] + row_k[tx];

            if (newDist < distJI) {
                dist[idx2(j, i, n)] = newDist;
                path[idx2(j, i, n)] = static_cast<unsigned int>(k);
            }
        }
    }
}

void floydWarshall(unsigned int* d_dist, unsigned int* d_path, size_t numNodes) {
    // Use 256 threads per block for good occupancy on Ampere GPUs
    unsigned int blockSize = 256;
    dim3 grid(
        (numNodes + TILE_SIZE - 1) / TILE_SIZE,
        (numNodes + TILE_SIZE - 1) / TILE_SIZE
    );
    size_t sharedMemBytes = 2 * TILE_SIZE * sizeof(unsigned int);

    for (size_t k = 0; k < numNodes; ++k) {
        floydWarshallKernel<<<grid, blockSize, sharedMemBytes>>>(
            d_dist, d_path, numNodes, k);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

bool validateResult(const std::vector<unsigned int>& dist, const size_t numNodes) {
    for (size_t i = 0; i < numNodes; ++i) {
        if (dist[idx2(i, i, numNodes)] != 0) {
            printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", i, i);
            return false;
        }
    }
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

    printf("Initializing graph...\n");
    std::vector<unsigned int> h_dist(numNodes * numNodes);
    std::vector<unsigned int> h_path(numNodes * numNodes);

    unsigned int seed = 42;
    const double range = static_cast<double>(MAX_DISTANCE - 1) + 1.0;
    for (size_t i = 0; i < numNodes * numNodes; ++i) {
        h_dist[i] = 1 + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
    }
    for (size_t i = 0; i < numNodes; ++i) {
        h_dist[idx2(i, i, numNodes)] = 0;
    }
    for (size_t j = 0; j < numNodes; ++j) {
        for (size_t i = 0; i < numNodes; ++i) {
            h_path[idx2(i, j, numNodes)] = j;
            h_path[idx2(j, i, numNodes)] = i;
        }
        h_path[idx2(j, j, numNodes)] = j;
    }

    unsigned int* d_dist = nullptr;
    unsigned int* d_path = nullptr;
    size_t matBytes = numNodes * numNodes * sizeof(unsigned int);

    CUDA_CHECK(cudaMalloc(&d_dist, matBytes));
    CUDA_CHECK(cudaMalloc(&d_path, matBytes));
    CUDA_CHECK(cudaMemcpy(d_dist, h_dist.data(), matBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_path, h_path.data(), matBytes, cudaMemcpyHostToDevice));

    printf("Computing shortest paths...\n");
    auto start = std::chrono::high_resolution_clock::now();
    floydWarshall(d_dist, d_path, numNodes);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    CUDA_CHECK(cudaMemcpy(h_dist.data(), d_dist, matBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_path.data(), d_path, matBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_dist));
    CUDA_CHECK(cudaFree(d_path));

    printf("Computation time: %ld ms\n", duration.count());
    double ops = (double)numNodes * numNodes * numNodes;
    double gflops = ops / (duration.count() / 1000.0) / 1e9;
    printf("Performance: %.3f GOPS\n", gflops);

    if (printResults) {
        print_results_int(h_dist, "DistanceMatrix");
    }
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(h_dist, numNodes);
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
