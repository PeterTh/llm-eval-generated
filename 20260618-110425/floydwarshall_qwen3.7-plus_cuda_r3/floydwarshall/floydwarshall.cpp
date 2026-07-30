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
constexpr size_t TILE_DIM = 32;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Index calculation for flattened 2D array (row-major for GPU coalesced access)
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return i * n + j;
}

// Phase 1: Update diagonal tile
__global__ void fw_diagonal_tile_kernel(
    unsigned int* __restrict__ dist,
    unsigned int* __restrict__ path,
    const size_t N,
    const size_t kb)
{
    const size_t i = kb + threadIdx.y;
    const size_t j = kb + threadIdx.x;

    __shared__ unsigned int s_ik[TILE_DIM];
    __shared__ unsigned int s_kj[TILE_DIM];

    const size_t k_end = min(kb + TILE_DIM, N);
    for (size_t k = kb; k < k_end; ++k) {
        if (threadIdx.x == 0 && i < N) s_ik[threadIdx.y] = dist[i * N + k];
        if (threadIdx.y == 0 && j < N) s_kj[threadIdx.x] = dist[k * N + j];
        __syncthreads();

        if (i < N && j < N) {
            const unsigned int newDist = s_ik[threadIdx.y] + s_kj[threadIdx.x];
            const size_t idx = i * N + j;
            if (newDist < dist[idx]) {
                dist[idx] = newDist;
                path[idx] = (unsigned int)k;
            }
        }
        __syncthreads();
    }
}

// Phase 2: Update row and column tiles
__global__ void fw_cross_tile_kernel(
    unsigned int* __restrict__ dist,
    unsigned int* __restrict__ path,
    const size_t N,
    const size_t kb,
    const size_t num_tiles)
{
    const size_t kb_tile = kb / TILE_DIM;

    size_t tile_i_start, tile_j_start;

    if (blockIdx.x < num_tiles - 1) {
        size_t jt = blockIdx.x;
        if (jt >= kb_tile) jt++;
        tile_i_start = kb;
        tile_j_start = jt * TILE_DIM;
    } else {
        size_t idx = blockIdx.x - (num_tiles - 1);
        size_t it = idx;
        if (it >= kb_tile) it++;
        tile_i_start = it * TILE_DIM;
        tile_j_start = kb;
    }

    const size_t i = tile_i_start + threadIdx.y;
    const size_t j = tile_j_start + threadIdx.x;

    if (i >= N || j >= N) return;

    __shared__ unsigned int s_ik[TILE_DIM];
    __shared__ unsigned int s_kj[TILE_DIM];

    const size_t k_end = min(kb + TILE_DIM, N);
    for (size_t k = kb; k < k_end; ++k) {
        if (threadIdx.x == 0) s_ik[threadIdx.y] = dist[i * N + k];
        if (threadIdx.y == 0) s_kj[threadIdx.x] = dist[k * N + j];
        __syncthreads();

        const unsigned int newDist = s_ik[threadIdx.y] + s_kj[threadIdx.x];
        const size_t eidx = i * N + j;
        if (newDist < dist[eidx]) {
            dist[eidx] = newDist;
            path[eidx] = (unsigned int)k;
        }
        __syncthreads();
    }
}

// Phase 3: Update remaining tiles
__global__ void fw_remaining_tile_kernel(
    unsigned int* __restrict__ dist,
    unsigned int* __restrict__ path,
    const size_t N,
    const size_t kb,
    const size_t num_tiles)
{
    const size_t kb_tile = kb / TILE_DIM;

    size_t it = blockIdx.y;
    if (it >= kb_tile) it++;
    size_t jt = blockIdx.x;
    if (jt >= kb_tile) jt++;

    const size_t tile_i_start = it * TILE_DIM;
    const size_t tile_j_start = jt * TILE_DIM;

    const size_t i = tile_i_start + threadIdx.y;
    const size_t j = tile_j_start + threadIdx.x;

    if (i >= N || j >= N) return;

    __shared__ unsigned int s_ik[TILE_DIM];
    __shared__ unsigned int s_kj[TILE_DIM];

    const size_t k_end = min(kb + TILE_DIM, N);
    for (size_t k = kb; k < k_end; ++k) {
        if (threadIdx.x == 0) s_ik[threadIdx.y] = dist[i * N + k];
        if (threadIdx.y == 0) s_kj[threadIdx.x] = dist[k * N + j];
        __syncthreads();

        const unsigned int newDist = s_ik[threadIdx.y] + s_kj[threadIdx.x];
        const size_t eidx = i * N + j;
        if (newDist < dist[eidx]) {
            dist[eidx] = newDist;
            path[eidx] = (unsigned int)k;
        }
        __syncthreads();
    }
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
    for (size_t i = 0; i < numNodes; ++i) {
        for (size_t j = 0; j < numNodes; ++j) {
            path[idx2(i, j, numNodes)] = j;
        }
        path[idx2(i, i, numNodes)] = i;
    }
}

void floydWarshall(std::vector<unsigned int>& dist, 
                   std::vector<unsigned int>& path, 
                   const size_t numNodes) {
    // Allocate GPU memory
    unsigned int* d_dist = nullptr;
    unsigned int* d_path = nullptr;
    const size_t matrixBytes = numNodes * numNodes * sizeof(unsigned int);

    CUDA_CHECK(cudaMalloc(&d_dist, matrixBytes));
    CUDA_CHECK(cudaMalloc(&d_path, matrixBytes));

    // Copy to GPU (already in row-major format)
    CUDA_CHECK(cudaMemcpy(d_dist, dist.data(), matrixBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_path, path.data(), matrixBytes, cudaMemcpyHostToDevice));

    // Block-tiled Floyd-Warshall on GPU with 3-phase kernel approach
    const size_t num_tiles = (numNodes + TILE_DIM - 1) / TILE_DIM;
    dim3 tile_block(TILE_DIM, TILE_DIM);

    for (size_t kb = 0; kb < numNodes; kb += TILE_DIM) {
        // Phase 1: Diagonal tile
        fw_diagonal_tile_kernel<<<dim3(1, 1), tile_block>>>(d_dist, d_path, numNodes, kb);

        // Phase 2: Row and column tiles
        if (num_tiles > 1) {
            dim3 cross_grid(2 * (num_tiles - 1), 1);
            fw_cross_tile_kernel<<<cross_grid, tile_block>>>(d_dist, d_path, numNodes, kb, num_tiles);
        }

        // Phase 3: Remaining tiles
        if (num_tiles > 2) {
            dim3 rem_grid(num_tiles - 1, num_tiles - 1);
            fw_remaining_tile_kernel<<<rem_grid, tile_block>>>(d_dist, d_path, numNodes, kb, num_tiles);
        }
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy back to host (still in row-major format)
    CUDA_CHECK(cudaMemcpy(dist.data(), d_dist, matrixBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(path.data(), d_path, matrixBytes, cudaMemcpyDeviceToHost));

    // Free GPU memory
    CUDA_CHECK(cudaFree(d_dist));
    CUDA_CHECK(cudaFree(d_path));
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
                const unsigned int distIJ = dist[idx2(i, j, numNodes)];
                const unsigned int distIK = dist[idx2(i, k, numNodes)];
                const unsigned int distKJ = dist[idx2(k, j, numNodes)];
                
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
    
    // Warm up CUDA context before timing
    cudaFree(0);
    
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
