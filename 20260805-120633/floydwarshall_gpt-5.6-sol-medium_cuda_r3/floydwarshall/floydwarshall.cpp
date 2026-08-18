#include <algorithm>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;
constexpr unsigned int TILE = 32;
constexpr unsigned int THREADS = 16;

inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

[[noreturn]] void cudaFail(const cudaError_t error, const char* operation) {
    std::fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(error));
    std::exit(EXIT_FAILURE);
}

inline void cudaCheck(const cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        cudaFail(error, operation);
    }
}

void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes,
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;

    for (size_t i = 0; i < numNodes * numNodes; ++i) {
        dist[i] = rangeMin + static_cast<unsigned int>(range * rand_r(&seed) / RAND_MAX);
    }

    for (size_t i = 0; i < numNodes; ++i) {
        dist[idx2(i, i, numNodes)] = 0;
    }
}

void initializePathMatrix(std::vector<unsigned int>& path, const size_t numNodes) {
    for (size_t j = 0; j < numNodes; ++j) {
        for (size_t i = 0; i < numNodes; ++i) {
            path[idx2(i, j, numNodes)] = static_cast<unsigned int>(j);
            path[idx2(j, i, numNodes)] = static_cast<unsigned int>(i);
        }
        path[idx2(j, j, numNodes)] = static_cast<unsigned int>(j);
    }
}

// Phase 1 closes the pivot tile. Four cells per thread give a full 32x32
// logical tile while retaining a compact 16x16 CUDA block.
__global__ void pivotKernel(unsigned int* __restrict__ dist,
                            unsigned int* __restrict__ path,
                            const unsigned int n, const unsigned int round) {
    __shared__ unsigned int tile[TILE][TILE + 1];

    const unsigned int base = round * TILE;
    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;

#pragma unroll
    for (unsigned int dy = 0; dy < TILE; dy += THREADS) {
#pragma unroll
        for (unsigned int dx = 0; dx < TILE; dx += THREADS) {
            const unsigned int row = base + ty + dy;
            const unsigned int col = base + tx + dx;
            tile[ty + dy][tx + dx] = (row < n && col < n) ? dist[row * n + col] : INF;
        }
    }
    __syncthreads();

#pragma unroll
    for (unsigned int k = 0; k < TILE; ++k) {
#pragma unroll
        for (unsigned int dy = 0; dy < TILE; dy += THREADS) {
#pragma unroll
            for (unsigned int dx = 0; dx < TILE; dx += THREADS) {
                const unsigned int row = ty + dy;
                const unsigned int col = tx + dx;
                const unsigned int candidate = tile[row][k] + tile[k][col];
                if (candidate < tile[row][col]) {
                    tile[row][col] = candidate;
                    const unsigned int globalRow = base + row;
                    const unsigned int globalCol = base + col;
                    if (globalRow < n && globalCol < n) {
                        path[globalRow * n + globalCol] = base + k;
                    }
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (unsigned int dy = 0; dy < TILE; dy += THREADS) {
#pragma unroll
        for (unsigned int dx = 0; dx < TILE; dx += THREADS) {
            const unsigned int row = base + ty + dy;
            const unsigned int col = base + tx + dx;
            if (row < n && col < n) {
                dist[row * n + col] = tile[ty + dy][tx + dx];
            }
        }
    }
}

// Phase 2 updates every tile in the pivot block-row and block-column.
__global__ void rowColumnKernel(unsigned int* __restrict__ dist,
                                unsigned int* __restrict__ path,
                                const unsigned int n, const unsigned int round) {
    __shared__ unsigned int pivot[TILE][TILE + 1];
    __shared__ unsigned int current[TILE][TILE + 1];

    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const unsigned int pivotBase = round * TILE;
    const unsigned int otherTile = blockIdx.x < round ? blockIdx.x : blockIdx.x + 1;
    const bool columnPhase = blockIdx.y != 0;
    const unsigned int rowBase = columnPhase ? otherTile * TILE : pivotBase;
    const unsigned int colBase = columnPhase ? pivotBase : otherTile * TILE;

#pragma unroll
    for (unsigned int dy = 0; dy < TILE; dy += THREADS) {
#pragma unroll
        for (unsigned int dx = 0; dx < TILE; dx += THREADS) {
            const unsigned int localRow = ty + dy;
            const unsigned int localCol = tx + dx;
            const unsigned int pivotRow = pivotBase + localRow;
            const unsigned int pivotCol = pivotBase + localCol;
            const unsigned int row = rowBase + localRow;
            const unsigned int col = colBase + localCol;
            pivot[localRow][localCol] = (pivotRow < n && pivotCol < n)
                                                   ? dist[pivotRow * n + pivotCol] : INF;
            current[localRow][localCol] = (row < n && col < n)
                                                   ? dist[row * n + col] : INF;
        }
    }
    __syncthreads();

#pragma unroll
    for (unsigned int k = 0; k < TILE; ++k) {
#pragma unroll
        for (unsigned int dy = 0; dy < TILE; dy += THREADS) {
#pragma unroll
            for (unsigned int dx = 0; dx < TILE; dx += THREADS) {
                const unsigned int localRow = ty + dy;
                const unsigned int localCol = tx + dx;
                const unsigned int candidate = columnPhase
                    ? current[localRow][k] + pivot[k][localCol]
                    : pivot[localRow][k] + current[k][localCol];
                if (candidate < current[localRow][localCol]) {
                    current[localRow][localCol] = candidate;
                    const unsigned int row = rowBase + localRow;
                    const unsigned int col = colBase + localCol;
                    if (row < n && col < n) {
                        path[row * n + col] = pivotBase + k;
                    }
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (unsigned int dy = 0; dy < TILE; dy += THREADS) {
#pragma unroll
        for (unsigned int dx = 0; dx < TILE; dx += THREADS) {
            const unsigned int row = rowBase + ty + dy;
            const unsigned int col = colBase + tx + dx;
            if (row < n && col < n) {
                dist[row * n + col] = current[ty + dy][tx + dx];
            }
        }
    }
}

// Phase 3 is the throughput-dominant min-plus tile product. The two input
// tiles are loaded once from global memory and each thread keeps four outputs.
__global__ void remainingKernel(unsigned int* __restrict__ dist,
                                unsigned int* __restrict__ path,
                                const unsigned int n, const unsigned int round) {
    __shared__ unsigned int columnTile[TILE][TILE + 1];
    __shared__ unsigned int rowTile[TILE][TILE + 1];

    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const unsigned int rowTileIndex = blockIdx.y < round ? blockIdx.y : blockIdx.y + 1;
    const unsigned int colTileIndex = blockIdx.x < round ? blockIdx.x : blockIdx.x + 1;
    const unsigned int rowBase = rowTileIndex * TILE;
    const unsigned int colBase = colTileIndex * TILE;
    const unsigned int pivotBase = round * TILE;

#pragma unroll
    for (unsigned int dy = 0; dy < TILE; dy += THREADS) {
#pragma unroll
        for (unsigned int dx = 0; dx < TILE; dx += THREADS) {
            const unsigned int localRow = ty + dy;
            const unsigned int localCol = tx + dx;
            const unsigned int row = rowBase + localRow;
            const unsigned int col = colBase + localCol;
            const unsigned int pivotRow = pivotBase + localRow;
            const unsigned int pivotCol = pivotBase + localCol;
            columnTile[localRow][localCol] = (row < n && pivotCol < n)
                ? dist[row * n + pivotCol] : INF;
            rowTile[localRow][localCol] = (pivotRow < n && col < n)
                ? dist[pivotRow * n + col] : INF;
        }
    }
    __syncthreads();

    const unsigned int row0 = rowBase + ty;
    const unsigned int row1 = row0 + THREADS;
    const unsigned int col0 = colBase + tx;
    const unsigned int col1 = col0 + THREADS;
    unsigned int value00 = (row0 < n && col0 < n) ? dist[row0 * n + col0] : INF;
    unsigned int value01 = (row0 < n && col1 < n) ? dist[row0 * n + col1] : INF;
    unsigned int value10 = (row1 < n && col0 < n) ? dist[row1 * n + col0] : INF;
    unsigned int value11 = (row1 < n && col1 < n) ? dist[row1 * n + col1] : INF;
    unsigned int path00 = 0, path01 = 0, path10 = 0, path11 = 0;
    bool changed00 = false, changed01 = false, changed10 = false, changed11 = false;

#pragma unroll
    for (unsigned int k = 0; k < TILE; ++k) {
        unsigned int candidate = columnTile[ty][k] + rowTile[k][tx];
        if (candidate < value00) { value00 = candidate; path00 = pivotBase + k; changed00 = true; }
        candidate = columnTile[ty][k] + rowTile[k][tx + THREADS];
        if (candidate < value01) { value01 = candidate; path01 = pivotBase + k; changed01 = true; }
        candidate = columnTile[ty + THREADS][k] + rowTile[k][tx];
        if (candidate < value10) { value10 = candidate; path10 = pivotBase + k; changed10 = true; }
        candidate = columnTile[ty + THREADS][k] + rowTile[k][tx + THREADS];
        if (candidate < value11) { value11 = candidate; path11 = pivotBase + k; changed11 = true; }
    }

    if (row0 < n && col0 < n) {
        dist[row0 * n + col0] = value00;
        if (changed00) path[row0 * n + col0] = path00;
    }
    if (row0 < n && col1 < n) {
        dist[row0 * n + col1] = value01;
        if (changed01) path[row0 * n + col1] = path01;
    }
    if (row1 < n && col0 < n) {
        dist[row1 * n + col0] = value10;
        if (changed10) path[row1 * n + col0] = path10;
    }
    if (row1 < n && col1 < n) {
        dist[row1 * n + col1] = value11;
        if (changed11) path[row1 * n + col1] = path11;
    }
}

float floydWarshall(std::vector<unsigned int>& dist,
                    std::vector<unsigned int>& path,
                    const size_t numNodes) {
    if (numNodes == 0) {
        return 0.0f;
    }

    const size_t bytes = dist.size() * sizeof(unsigned int);
    unsigned int* deviceDist = nullptr;
    unsigned int* devicePath = nullptr;
    cudaCheck(cudaMalloc(&deviceDist, bytes), "distance matrix allocation");
    cudaCheck(cudaMalloc(&devicePath, bytes), "path matrix allocation");
    cudaCheck(cudaMemcpy(deviceDist, dist.data(), bytes, cudaMemcpyHostToDevice),
              "distance matrix upload");
    cudaCheck(cudaMemcpy(devicePath, path.data(), bytes, cudaMemcpyHostToDevice),
              "path matrix upload");

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cudaCheck(cudaEventCreate(&start), "start event creation");
    cudaCheck(cudaEventCreate(&stop), "stop event creation");
    cudaCheck(cudaEventRecord(start), "start event recording");

    const unsigned int n = static_cast<unsigned int>(numNodes);
    const unsigned int rounds = (n + TILE - 1) / TILE;
    const dim3 threads(THREADS, THREADS);
    for (unsigned int round = 0; round < rounds; ++round) {
        pivotKernel<<<1, threads>>>(deviceDist, devicePath, n, round);
        if (rounds > 1) {
            rowColumnKernel<<<dim3(rounds - 1, 2), threads>>>(deviceDist, devicePath, n, round);
            remainingKernel<<<dim3(rounds - 1, rounds - 1), threads>>>(deviceDist, devicePath, n, round);
        }
    }
    cudaCheck(cudaGetLastError(), "Floyd-Warshall kernel launch");
    cudaCheck(cudaEventRecord(stop), "stop event recording");
    cudaCheck(cudaEventSynchronize(stop), "Floyd-Warshall execution");

    float milliseconds = 0.0f;
    cudaCheck(cudaEventElapsedTime(&milliseconds, start, stop), "elapsed time measurement");
    cudaCheck(cudaMemcpy(dist.data(), deviceDist, bytes, cudaMemcpyDeviceToHost),
              "distance matrix download");
    cudaCheck(cudaMemcpy(path.data(), devicePath, bytes, cudaMemcpyDeviceToHost),
              "path matrix download");

    cudaCheck(cudaEventDestroy(start), "start event destruction");
    cudaCheck(cudaEventDestroy(stop), "stop event destruction");
    cudaCheck(cudaFree(deviceDist), "distance matrix release");
    cudaCheck(cudaFree(devicePath), "path matrix release");
    return milliseconds;
}

bool validateResult(const std::vector<unsigned int>& dist, const size_t numNodes) {
    for (size_t i = 0; i < numNodes; ++i) {
        if (dist[idx2(i, i, numNodes)] != 0) {
            std::printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", i, i);
            return false;
        }
    }

    for (size_t i = 0; i < std::min(numNodes, size_t{10}); ++i) {
        for (size_t j = 0; j < std::min(numNodes, size_t{10}); ++j) {
            for (size_t k = 0; k < numNodes; ++k) {
                const unsigned int distIJ = dist[idx2(j, i, numNodes)];
                const unsigned int distIK = dist[idx2(k, i, numNodes)];
                const unsigned int distKJ = dist[idx2(j, k, numNodes)];
                if (distIK < INF && distKJ < INF && distIK + distKJ < distIJ) {
                    std::printf("Validation failed: triangle inequality violated at [%zu,%zu,%zu]\n",
                                i, j, k);
                    return false;
                }
            }
        }
    }
    return true;
}

void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("Options:\n");
    std::printf("  -n <num>     Number of nodes in the graph (default: 512)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

bool parseNodeCount(const char* text, size_t& result) {
    errno = 0;
    char* end = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value > UINT_MAX) {
        return false;
    }
    result = static_cast<size_t>(value);
    return result == 0 || result <= std::numeric_limits<size_t>::max() / result;
}

int main(int argc, char** argv) {
    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            if (!parseNodeCount(argv[++i], numNodes)) {
                std::fprintf(stderr, "Invalid number of nodes: %s\n", argv[i]);
                return 1;
            }
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            return 0;
        } else {
            std::printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }

    std::printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
    std::printf("Number of nodes: %zu\n", numNodes);
    std::printf("Validation: %s\n", validate ? "enabled" : "disabled");

    std::vector<unsigned int> dist;
    std::vector<unsigned int> path;
    try {
        dist.resize(numNodes * numNodes);
        path.resize(numNodes * numNodes);
    } catch (const std::bad_alloc&) {
        std::fprintf(stderr, "Unable to allocate matrices for %zu nodes\n", numNodes);
        return 1;
    }

    std::printf("Initializing graph...\n");
    initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
    initializePathMatrix(path, numNodes);

    std::printf("Computing shortest paths...\n");
    const float milliseconds = floydWarshall(dist, path, numNodes);
    const long long roundedMilliseconds = static_cast<long long>(milliseconds + 0.5f);
    std::printf("Computation time: %lld ms\n", roundedMilliseconds);

    const double ops = static_cast<double>(numNodes) * numNodes * numNodes;
    const double gops = milliseconds > 0.0f ? ops / (milliseconds * 1.0e6) : 0.0;
    std::printf("Performance: %.3f GOPS\n", gops);

    if (printResults) {
        print_results_int(dist, "DistanceMatrix");
    }

    if (validate) {
        std::printf("Validating result...\n");
        if (validateResult(dist, numNodes)) {
            std::printf("Validation: PASSED\n");
            return 0;
        }
        std::printf("Validation: FAILED\n");
        return 1;
    }
    return 0;
}
