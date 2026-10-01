#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;
constexpr int TILE = 32;

inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

void initializeDistanceMatrix(std::vector<unsigned int>& dist, size_t n,
                              unsigned int rangeMin, unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;
    for (size_t i = 0; i < n * n; ++i)
        dist[i] = rangeMin + static_cast<unsigned int>(range * rand_r(&seed) / (double)RAND_MAX);
    for (size_t i = 0; i < n; ++i) dist[idx2(i, i, n)] = 0;
}

void initializePathMatrix(std::vector<unsigned int>& path, size_t n) {
    for (size_t j = 0; j < n; ++j) {
        for (size_t i = 0; i < n; ++i) {
            path[idx2(i, j, n)] = j;
            path[idx2(j, i, n)] = i;
        }
        path[idx2(j, j, n)] = j;
    }
}

// Each CUDA block operates on one matrix tile. x is the source-node index,
// matching the contiguous dimension of the original column-major-like layout.
__global__ void diagonalTile(unsigned int* d, unsigned int* p, int n, int base) {
    __shared__ unsigned int a[TILE][TILE];
    int x = threadIdx.x, y = threadIdx.y;
    int i = base + x, j = base + y;
    a[y][x] = (i < n && j < n) ? d[(size_t)j * n + i] : INF;
    __syncthreads();
    for (int k = 0; k < TILE && base + k < n; ++k) {
        unsigned int candidate = a[k][x] + a[y][k];
        if (candidate < a[y][x]) { a[y][x] = candidate; if (i < n && j < n) p[(size_t)j * n + i] = base + k; }
        __syncthreads();
    }
    if (i < n && j < n) d[(size_t)j * n + i] = a[y][x];
}

// mode 0 updates a tile in the pivot row, mode 1 the pivot column.
__global__ void pivotLine(unsigned int* d, unsigned int* p, int n, int base, int other, int mode) {
    __shared__ unsigned int pivot[TILE][TILE], target[TILE][TILE];
    int x = threadIdx.x, y = threadIdx.y;
    int ti = mode == 0 ? base : other;
    int tj = mode == 0 ? other : base;
    int i = ti + x, j = tj + y;
    int pi = base + x, pj = base + y;
    int qi = ti + x, qj = tj + y;
    pivot[y][x] = (pi < n && pj < n) ? d[(size_t)pj * n + pi] : INF;
    target[y][x] = (i < n && j < n) ? d[(size_t)j * n + i] : INF;
    __syncthreads();
    for (int k = 0; k < TILE && base + k < n; ++k) {
        unsigned int left = mode == 0 ? pivot[k][x] : target[k][x];
        unsigned int right = mode == 0 ? target[y][k] : pivot[y][k];
        unsigned int candidate = left + right;
        if (candidate < target[y][x]) { target[y][x] = candidate; if (i < n && j < n) p[(size_t)j * n + i] = base + k; }
        __syncthreads();
    }
    if (i < n && j < n) d[(size_t)j * n + i] = target[y][x];
}

__global__ void outerTiles(unsigned int* d, unsigned int* p, int n, int base, int row, int col) {
    __shared__ unsigned int column[TILE][TILE], rowTile[TILE][TILE], target[TILE][TILE];
    int x = threadIdx.x, y = threadIdx.y;
    int i = row + x, j = col + y;
    int ci = row + x, cj = base + y;
    int ri = base + x, rj = col + y;
    column[y][x] = (ci < n && cj < n) ? d[(size_t)cj * n + ci] : INF;
    rowTile[y][x] = (ri < n && rj < n) ? d[(size_t)rj * n + ri] : INF;
    target[y][x] = (i < n && j < n) ? d[(size_t)j * n + i] : INF;
    __syncthreads();
    for (int k = 0; k < TILE && base + k < n; ++k) {
        unsigned int candidate = column[k][x] + rowTile[y][k];
        if (candidate < target[y][x]) { target[y][x] = candidate; if (i < n && j < n) p[(size_t)j * n + i] = base + k; }
        __syncthreads();
    }
    if (i < n && j < n) d[(size_t)j * n + i] = target[y][x];
}

void checkCuda(cudaError_t err) {
    if (err != cudaSuccess) { fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); std::exit(1); }
}

void floydWarshall(std::vector<unsigned int>& dist, std::vector<unsigned int>& path, size_t n) {
    if (!n) return;
    unsigned int *dd = nullptr, *dp = nullptr;
    size_t bytes = n * n * sizeof(unsigned int);
    checkCuda(cudaMalloc(&dd, bytes));
    checkCuda(cudaMalloc(&dp, bytes));
    checkCuda(cudaMemcpy(dd, dist.data(), bytes, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(dp, path.data(), bytes, cudaMemcpyHostToDevice));
    dim3 block(TILE, TILE);
    int tiles = static_cast<int>((n + TILE - 1) / TILE);
    for (int t = 0; t < tiles; ++t) {
        int base = t * TILE;
        diagonalTile<<<1, block>>>(dd, dp, static_cast<int>(n), base);
        for (int q = 0; q < tiles; ++q) if (q != t) {
            pivotLine<<<1, block>>>(dd, dp, static_cast<int>(n), base, q * TILE, 0);
            pivotLine<<<1, block>>>(dd, dp, static_cast<int>(n), base, q * TILE, 1);
        }
        for (int r = 0; r < tiles; ++r) if (r != t)
            for (int c = 0; c < tiles; ++c) if (c != t)
                outerTiles<<<1, block>>>(dd, dp, static_cast<int>(n), base, r * TILE, c * TILE);
    }
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());
    checkCuda(cudaMemcpy(dist.data(), dd, bytes, cudaMemcpyDeviceToHost));
    checkCuda(cudaMemcpy(path.data(), dp, bytes, cudaMemcpyDeviceToHost));
    cudaFree(dp); cudaFree(dd);
}

bool validateResult(const std::vector<unsigned int>& dist, size_t n) {
    for (size_t i = 0; i < n; ++i) if (dist[idx2(i, i, n)] != 0) {
        printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", i, i); return false;
    }
    for (size_t i = 0; i < std::min(n, size_t(10)); ++i)
        for (size_t j = 0; j < std::min(n, size_t(10)); ++j)
            for (size_t k = 0; k < n; ++k) {
                unsigned int ij = dist[idx2(j, i, n)], ik = dist[idx2(k, i, n)], kj = dist[idx2(j, k, n)];
                if (ik < INF && kj < INF && ik + kj < ij) {
                    printf("Validation failed: triangle inequality violated at [%zu,%zu,%zu]\n", i, j, k); return false;
                }
            }
    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\nOptions:\n  -n <num>     Number of nodes in the graph (default: 512)\n  -v           Enable validation\n  -r           Print results for external validation\n  -h           Show this help message\n", progName);
}

int main(int argc, char** argv) {
    size_t numNodes = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) numNodes = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (!strcmp(argv[i], "-h")) { printUsage(argv[0]); return 0; }
        else { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); return 1; }
    }
    printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\nNumber of nodes: %zu\nValidation: %s\n", numNodes, validate ? "enabled" : "disabled");
    std::vector<unsigned int> dist(numNodes * numNodes), path(numNodes * numNodes);
    printf("Initializing graph...\n");
    initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
    initializePathMatrix(path, numNodes);
    printf("Computing shortest paths...\n");
    auto start = std::chrono::high_resolution_clock::now();
    floydWarshall(dist, path, numNodes);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    printf("Computation time: %ld ms\n", duration.count());
    double gops = (double)numNodes * numNodes * numNodes / (duration.count() / 1000.0) / 1e9;
    printf("Performance: %.3f GOPS\n", gops);
    if (printResults) print_results_int(dist, "DistanceMatrix");
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(dist, numNodes);
        printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        return valid ? 0 : 1;
    }
    return 0;
}
