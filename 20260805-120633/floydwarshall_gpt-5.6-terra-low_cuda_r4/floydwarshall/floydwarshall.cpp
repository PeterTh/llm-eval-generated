#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;
constexpr int TILE = 32;

__host__ __device__ inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

static void checkCuda(const cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

// The extra column eliminates shared-memory bank conflicts for column accesses.
__global__ void fwDiagonal(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
                           const size_t n, const int pivot) {
    __shared__ unsigned int tile[TILE][TILE + 1];
    const int x = threadIdx.x, y = threadIdx.y;
    const size_t i = static_cast<size_t>(pivot) * TILE + x;
    const size_t j = static_cast<size_t>(pivot) * TILE + y;
    const bool valid = i < n && j < n;
    unsigned int route = valid ? path[idx2(i, j, n)] : 0;
    tile[x][y] = valid ? dist[idx2(i, j, n)] : INF;
    __syncthreads();

    const int limit = min(TILE, static_cast<int>(n - static_cast<size_t>(pivot) * TILE));
    for (int k = 0; k < limit; ++k) {
        const unsigned int candidate = tile[x][k] + tile[k][y];
        if (candidate < tile[x][y]) {
            tile[x][y] = candidate;
            route = static_cast<unsigned int>(pivot * TILE + k);
        }
        __syncthreads();
    }
    if (valid) {
        dist[idx2(i, j, n)] = tile[x][y];
        path[idx2(i, j, n)] = route;
    }
}

// Updates the pivot block's row and column after its diagonal block is complete.
__global__ void fwPivotBands(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
                             const size_t n, const int pivot) {
    const int block = blockIdx.x;
    if (block == pivot) return;
    const bool rowBand = blockIdx.y == 0;
    __shared__ unsigned int pivotTile[TILE][TILE + 1];
    __shared__ unsigned int tile[TILE][TILE + 1];
    const int x = threadIdx.x, y = threadIdx.y;
    const size_t base = static_cast<size_t>(pivot) * TILE;
    const size_t other = static_cast<size_t>(block) * TILE;
    const size_t i = rowBand ? base + x : other + x;
    const size_t j = rowBand ? other + y : base + y;
    const bool valid = i < n && j < n;
    const size_t pi = base + x, pj = base + y;
    pivotTile[x][y] = (pi < n && pj < n) ? dist[idx2(pi, pj, n)] : INF;
    tile[x][y] = valid ? dist[idx2(i, j, n)] : INF;
    unsigned int route = valid ? path[idx2(i, j, n)] : 0;
    __syncthreads();

    const int limit = min(TILE, static_cast<int>(n - base));
    for (int k = 0; k < limit; ++k) {
        const unsigned int candidate = rowBand
            ? pivotTile[x][k] + tile[k][y]
            : tile[x][k] + pivotTile[k][y];
        if (candidate < tile[x][y]) {
            tile[x][y] = candidate;
            route = static_cast<unsigned int>(base + k);
        }
        __syncthreads();
    }
    if (valid) {
        dist[idx2(i, j, n)] = tile[x][y];
        path[idx2(i, j, n)] = route;
    }
}

// Updates all blocks not touching the pivot block.  Each block is independent here.
__global__ void fwRemaining(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
                            const size_t n, const int pivot) {
    const int sourceBlock = blockIdx.x, destinationBlock = blockIdx.y;
    if (sourceBlock == pivot || destinationBlock == pivot) return;
    __shared__ unsigned int current[TILE][TILE + 1];
    __shared__ unsigned int left[TILE][TILE + 1];
    __shared__ unsigned int right[TILE][TILE + 1];
    const int x = threadIdx.x, y = threadIdx.y;
    const size_t base = static_cast<size_t>(pivot) * TILE;
    const size_t i = static_cast<size_t>(sourceBlock) * TILE + x;
    const size_t j = static_cast<size_t>(destinationBlock) * TILE + y;
    const bool valid = i < n && j < n;
    const size_t kForColumn = base + y;
    const size_t kForRow = base + x;
    current[x][y] = valid ? dist[idx2(i, j, n)] : INF;
    left[x][y] = (i < n && kForColumn < n) ? dist[idx2(i, kForColumn, n)] : INF;
    right[x][y] = (kForRow < n && j < n) ? dist[idx2(kForRow, j, n)] : INF;
    unsigned int route = valid ? path[idx2(i, j, n)] : 0;
    __syncthreads();

    const int limit = min(TILE, static_cast<int>(n - base));
    for (int k = 0; k < limit; ++k) {
        const unsigned int candidate = left[x][k] + right[k][y];
        if (candidate < current[x][y]) {
            current[x][y] = candidate;
            route = static_cast<unsigned int>(base + k);
        }
    }
    if (valid) {
        dist[idx2(i, j, n)] = current[x][y];
        path[idx2(i, j, n)] = route;
    }
}

void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes,
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;
    for (size_t i = 0; i < numNodes * numNodes; ++i)
        dist[i] = rangeMin + static_cast<unsigned int>(range * rand_r(&seed) / static_cast<double>(RAND_MAX));
    for (size_t i = 0; i < numNodes; ++i) dist[idx2(i, i, numNodes)] = 0;
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

void floydWarshall(std::vector<unsigned int>& dist, std::vector<unsigned int>& path, const size_t numNodes) {
    if (numNodes == 0) return;
    const size_t bytes = numNodes * numNodes * sizeof(unsigned int);
    unsigned int *deviceDist = nullptr, *devicePath = nullptr;
    checkCuda(cudaMalloc(&deviceDist, bytes), "allocating distance matrix");
    checkCuda(cudaMalloc(&devicePath, bytes), "allocating path matrix");
    checkCuda(cudaMemcpy(deviceDist, dist.data(), bytes, cudaMemcpyHostToDevice), "copying distance matrix to device");
    checkCuda(cudaMemcpy(devicePath, path.data(), bytes, cudaMemcpyHostToDevice), "copying path matrix to device");

    const int blocks = static_cast<int>((numNodes + TILE - 1) / TILE);
    const dim3 threads(TILE, TILE);
    for (int pivot = 0; pivot < blocks; ++pivot) {
        fwDiagonal<<<1, threads>>>(deviceDist, devicePath, numNodes, pivot);
        fwPivotBands<<<dim3(blocks, 2), threads>>>(deviceDist, devicePath, numNodes, pivot);
        fwRemaining<<<dim3(blocks, blocks), threads>>>(deviceDist, devicePath, numNodes, pivot);
        checkCuda(cudaGetLastError(), "launching Floyd-Warshall kernels");
    }
    checkCuda(cudaDeviceSynchronize(), "executing Floyd-Warshall kernels");
    checkCuda(cudaMemcpy(dist.data(), deviceDist, bytes, cudaMemcpyDeviceToHost), "copying distance matrix from device");
    checkCuda(cudaMemcpy(path.data(), devicePath, bytes, cudaMemcpyDeviceToHost), "copying path matrix from device");
    checkCuda(cudaFree(deviceDist), "freeing distance matrix");
    checkCuda(cudaFree(devicePath), "freeing path matrix");
}

bool validateResult(const std::vector<unsigned int>& dist, const size_t numNodes) {
    for (size_t i = 0; i < numNodes; ++i) {
        if (dist[idx2(i, i, numNodes)] != 0) {
            std::printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", i, i);
            return false;
        }
    }
    for (size_t i = 0; i < std::min(numNodes, size_t{10}); ++i)
        for (size_t j = 0; j < std::min(numNodes, size_t{10}); ++j)
            for (size_t k = 0; k < numNodes; ++k)
                if (dist[idx2(k, i, numNodes)] < INF && dist[idx2(j, k, numNodes)] < INF &&
                    dist[idx2(k, i, numNodes)] + dist[idx2(j, k, numNodes)] < dist[idx2(j, i, numNodes)]) {
                    std::printf("Validation failed: triangle inequality violated at [%zu,%zu,%zu]\n", i, j, k);
                    return false;
                }
    return true;
}

void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\nOptions:\n  -n <num>     Number of nodes in the graph (default: 512)\n  -v           Enable validation\n  -r           Print results for external validation\n  -h           Show this help message\n", progName);
}

int main(int argc, char** argv) {
    size_t numNodes = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) numNodes = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) printResults = true;
        else if (std::strcmp(argv[i], "-h") == 0) { printUsage(argv[0]); return 0; }
        else { std::printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); return 1; }
    }
    if (numNodes != 0 && numNodes > std::numeric_limits<size_t>::max() / numNodes) {
        std::fprintf(stderr, "Number of nodes is too large\n"); return 1;
    }
    std::printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\nNumber of nodes: %zu\nValidation: %s\n",
                numNodes, validate ? "enabled" : "disabled");
    std::vector<unsigned int> dist(numNodes * numNodes), path(numNodes * numNodes);
    std::printf("Initializing graph...\n");
    initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
    initializePathMatrix(path, numNodes);
    std::printf("Computing shortest paths...\n");
    const auto start = std::chrono::high_resolution_clock::now();
    floydWarshall(dist, path, numNodes);
    const auto end = std::chrono::high_resolution_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::printf("Computation time: %ld ms\n", duration.count());
    const double ops = static_cast<double>(numNodes) * numNodes * numNodes;
    const double gops = duration.count() ? ops / (duration.count() / 1000.0) / 1e9 : 0.0;
    std::printf("Performance: %.3f GOPS\n", gops);
    if (printResults) print_results_int(dist, "DistanceMatrix");
    if (validate) {
        std::printf("Validating result...\n");
        if (validateResult(dist, numNodes)) { std::printf("Validation: PASSED\n"); return 0; }
        std::printf("Validation: FAILED\n"); return 1;
    }
    return 0;
}
