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
constexpr unsigned int TILE_SIZE = 32;
constexpr unsigned int PHASE12_BLOCK_ROWS = 16;
constexpr unsigned int LARGE_PHASE3_BLOCK_ROWS = 8;
constexpr unsigned int SMALL_PHASE3_BLOCK_ROWS = 16;
constexpr unsigned int SMALL_GRAPH_TILE_LIMIT = 32;

// The matrix is column-major: row + column * number_of_rows. Keeping the row
// coordinate in threadIdx.x makes every global-memory transaction coalesced.
inline constexpr size_t idx2(const size_t row, const size_t column,
                             const size_t numNodes) noexcept {
    return column * numNodes + row;
}

[[noreturn]] void cudaFailure(const cudaError_t error, const char* expression,
                              const char* file, const int line) {
    std::fprintf(stderr, "CUDA error at %s:%d while evaluating %s: %s\n", file,
                 line, expression, cudaGetErrorString(error));
    std::exit(EXIT_FAILURE);
}

#define CUDA_CHECK(expression)                                                   \
    do {                                                                         \
        const cudaError_t cuda_check_error = (expression);                       \
        if (cuda_check_error != cudaSuccess) {                                   \
            cudaFailure(cuda_check_error, #expression, __FILE__, __LINE__);      \
        }                                                                        \
    } while (false)

void initializeDistanceMatrix(std::vector<unsigned int>& dist,
                              const size_t numNodes,
                              const unsigned int rangeMin,
                              const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;

    for (size_t i = 0; i < numNodes * numNodes; ++i) {
        dist[i] = rangeMin + static_cast<unsigned int>(
                                 range * rand_r(&seed) /
                                 static_cast<double>(RAND_MAX));
    }

    for (size_t i = 0; i < numNodes; ++i) {
        dist[idx2(i, i, numNodes)] = 0;
    }
}

__device__ __forceinline__ unsigned int matrixIndex(const unsigned int row,
                                                     const unsigned int column,
                                                     const unsigned int n) {
    return column * n + row;
}

// Phase 1: close the diagonal (pivot) tile for this round.
__global__ __launch_bounds__(TILE_SIZE * PHASE12_BLOCK_ROWS, 3)
void updatePivotTile(unsigned int* __restrict__ dist, const unsigned int n,
                     const unsigned int round) {
    __shared__ unsigned int pivot[TILE_SIZE][TILE_SIZE];

    const unsigned int localRow = threadIdx.x;
    const unsigned int row = round * TILE_SIZE + localRow;

#pragma unroll
    for (unsigned int localColumn = threadIdx.y; localColumn < TILE_SIZE;
         localColumn += PHASE12_BLOCK_ROWS) {
        const unsigned int column = round * TILE_SIZE + localColumn;
        pivot[localColumn][localRow] =
            (row < n && column < n) ? dist[matrixIndex(row, column, n)] : INF;
    }
    __syncthreads();

#pragma unroll
    for (unsigned int k = 0; k < TILE_SIZE; ++k) {
#pragma unroll
        for (unsigned int localColumn = threadIdx.y; localColumn < TILE_SIZE;
             localColumn += PHASE12_BLOCK_ROWS) {
            const unsigned int oldDistance = pivot[localColumn][localRow];
            const unsigned int throughK =
                pivot[k][localRow] + pivot[localColumn][k];
            if (throughK < oldDistance) {
                pivot[localColumn][localRow] = throughK;
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (unsigned int localColumn = threadIdx.y; localColumn < TILE_SIZE;
         localColumn += PHASE12_BLOCK_ROWS) {
        const unsigned int column = round * TILE_SIZE + localColumn;
        if (row < n && column < n) {
            dist[matrixIndex(row, column, n)] = pivot[localColumn][localRow];
        }
    }
}

// Phase 2: update every tile that shares rows or columns with the pivot tile.
// One kernel handles both orientations, halving the number of launches here.
__global__ __launch_bounds__(TILE_SIZE * PHASE12_BLOCK_ROWS, 3)
void updatePivotRowAndColumn(unsigned int* __restrict__ dist,
                             const unsigned int n,
                             const unsigned int round,
                             const unsigned int tileCount) {
    __shared__ unsigned int pivot[TILE_SIZE][TILE_SIZE];
    __shared__ unsigned int current[TILE_SIZE][TILE_SIZE];

    const unsigned int tilesWithoutPivot = tileCount - 1;
    const bool pivotRows = blockIdx.x < tilesWithoutPivot;
    const unsigned int compactTile = blockIdx.x % tilesWithoutPivot;
    const unsigned int otherTile = compactTile + (compactTile >= round);
    const unsigned int rowTile = pivotRows ? round : otherTile;
    const unsigned int columnTile = pivotRows ? otherTile : round;
    const unsigned int localRow = threadIdx.x;
    const unsigned int row = rowTile * TILE_SIZE + localRow;
    const unsigned int pivotRow = round * TILE_SIZE + localRow;

#pragma unroll
    for (unsigned int localColumn = threadIdx.y; localColumn < TILE_SIZE;
         localColumn += PHASE12_BLOCK_ROWS) {
        const unsigned int column = columnTile * TILE_SIZE + localColumn;
        const unsigned int pivotColumn = round * TILE_SIZE + localColumn;
        current[localColumn][localRow] =
            (row < n && column < n) ? dist[matrixIndex(row, column, n)] : INF;
        pivot[localColumn][localRow] =
            (pivotRow < n && pivotColumn < n)
                ? dist[matrixIndex(pivotRow, pivotColumn, n)]
                : INF;
    }
    __syncthreads();

    if (pivotRows) {
#pragma unroll
        for (unsigned int k = 0; k < TILE_SIZE; ++k) {
#pragma unroll
            for (unsigned int localColumn = threadIdx.y;
                 localColumn < TILE_SIZE;
                 localColumn += PHASE12_BLOCK_ROWS) {
                const unsigned int oldDistance = current[localColumn][localRow];
                const unsigned int throughK =
                    pivot[k][localRow] + current[localColumn][k];
                if (throughK < oldDistance) {
                    current[localColumn][localRow] = throughK;
                }
            }
            __syncthreads();
        }
    } else {
#pragma unroll
        for (unsigned int k = 0; k < TILE_SIZE; ++k) {
#pragma unroll
            for (unsigned int localColumn = threadIdx.y;
                 localColumn < TILE_SIZE;
                 localColumn += PHASE12_BLOCK_ROWS) {
                const unsigned int oldDistance = current[localColumn][localRow];
                const unsigned int throughK =
                    current[k][localRow] + pivot[localColumn][k];
                if (throughK < oldDistance) {
                    current[localColumn][localRow] = throughK;
                }
            }
            __syncthreads();
        }
    }

#pragma unroll
    for (unsigned int localColumn = threadIdx.y; localColumn < TILE_SIZE;
         localColumn += PHASE12_BLOCK_ROWS) {
        const unsigned int column = columnTile * TILE_SIZE + localColumn;
        if (row < n && column < n) {
            dist[matrixIndex(row, column, n)] = current[localColumn][localRow];
        }
    }
}

// Phase 3: all non-pivot tiles are independent. The two input tiles are staged
// once in shared memory and each thread keeps multiple outputs in registers.
template <unsigned int BLOCK_ROWS, unsigned int MIN_BLOCKS_PER_SM>
__global__ __launch_bounds__(TILE_SIZE * BLOCK_ROWS, MIN_BLOCKS_PER_SM)
void updateRemainingTiles(unsigned int* __restrict__ dist,
                          const unsigned int n, const unsigned int round) {
    __shared__ unsigned int toPivot[TILE_SIZE][TILE_SIZE];
    __shared__ unsigned int fromPivot[TILE_SIZE][TILE_SIZE];

    const unsigned int compactRowTile = blockIdx.x;
    const unsigned int compactColumnTile = blockIdx.y;
    const unsigned int rowTile =
        compactRowTile + (compactRowTile >= round);
    const unsigned int columnTile =
        compactColumnTile + (compactColumnTile >= round);
    const unsigned int localRow = threadIdx.x;
    const unsigned int row = rowTile * TILE_SIZE + localRow;
    const unsigned int pivotRow = round * TILE_SIZE + localRow;

#pragma unroll
    for (unsigned int localColumn = threadIdx.y; localColumn < TILE_SIZE;
         localColumn += BLOCK_ROWS) {
        const unsigned int pivotColumn = round * TILE_SIZE + localColumn;
        const unsigned int column = columnTile * TILE_SIZE + localColumn;
        toPivot[localColumn][localRow] =
            (row < n && pivotColumn < n)
                ? dist[matrixIndex(row, pivotColumn, n)]
                : INF;
        fromPivot[localColumn][localRow] =
            (pivotRow < n && column < n)
                ? dist[matrixIndex(pivotRow, column, n)]
                : INF;
    }
    __syncthreads();

    unsigned int values[TILE_SIZE / BLOCK_ROWS];
#pragma unroll
    for (unsigned int item = 0; item < TILE_SIZE / BLOCK_ROWS; ++item) {
        const unsigned int localColumn = threadIdx.y + item * BLOCK_ROWS;
        const unsigned int column = columnTile * TILE_SIZE + localColumn;
        values[item] = (row < n && column < n)
                           ? dist[matrixIndex(row, column, n)]
                           : INF;
    }

#pragma unroll
    for (unsigned int k = 0; k < TILE_SIZE; ++k) {
        const unsigned int distanceToK = toPivot[k][localRow];
#pragma unroll
        for (unsigned int item = 0; item < TILE_SIZE / BLOCK_ROWS; ++item) {
            const unsigned int localColumn = threadIdx.y + item * BLOCK_ROWS;
            const unsigned int throughK =
                distanceToK + fromPivot[localColumn][k];
            if (throughK < values[item]) {
                values[item] = throughK;
            }
        }
    }

#pragma unroll
    for (unsigned int item = 0; item < TILE_SIZE / BLOCK_ROWS; ++item) {
        const unsigned int localColumn = threadIdx.y + item * BLOCK_ROWS;
        const unsigned int column = columnTile * TILE_SIZE + localColumn;
        if (row < n && column < n) {
            dist[matrixIndex(row, column, n)] = values[item];
        }
    }
}

void floydWarshall(unsigned int* const deviceDist, const unsigned int numNodes) {
    if (numNodes == 0) {
        return;
    }

    const unsigned int tileCount =
        (numNodes + TILE_SIZE - 1) / TILE_SIZE;
    const dim3 phase12Threads(TILE_SIZE, PHASE12_BLOCK_ROWS);

    for (unsigned int round = 0; round < tileCount; ++round) {
        updatePivotTile<<<1, phase12Threads>>>(deviceDist, numNodes, round);

        if (tileCount > 1) {
            updatePivotRowAndColumn<<<2 * (tileCount - 1), phase12Threads>>>(
                deviceDist, numNodes, round, tileCount);
            const dim3 remainingTiles(tileCount - 1, tileCount - 1);
            if (tileCount <= SMALL_GRAPH_TILE_LIMIT) {
                const dim3 phase3Threads(TILE_SIZE, SMALL_PHASE3_BLOCK_ROWS);
                updateRemainingTiles<SMALL_PHASE3_BLOCK_ROWS, 3>
                    <<<remainingTiles, phase3Threads>>>(deviceDist, numNodes,
                                                        round);
            } else {
                const dim3 phase3Threads(TILE_SIZE, LARGE_PHASE3_BLOCK_ROWS);
                updateRemainingTiles<LARGE_PHASE3_BLOCK_ROWS, 4>
                    <<<remainingTiles, phase3Threads>>>(deviceDist, numNodes,
                                                        round);
            }
        }
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

bool validateResult(const std::vector<unsigned int>& dist,
                    const size_t numNodes) {
    for (size_t i = 0; i < numNodes; ++i) {
        if (dist[idx2(i, i, numNodes)] != 0) {
            std::printf("Validation failed: diagonal element [%zu,%zu] is not zero\n",
                        i, i);
            return false;
        }
    }

    for (size_t i = 0; i < std::min(numNodes, static_cast<size_t>(10)); ++i) {
        for (size_t j = 0; j < std::min(numNodes, static_cast<size_t>(10)); ++j) {
            for (size_t k = 0; k < numNodes; ++k) {
                const unsigned int distIJ = dist[idx2(j, i, numNodes)];
                const unsigned int distIK = dist[idx2(k, i, numNodes)];
                const unsigned int distKJ = dist[idx2(j, k, numNodes)];

                if (distIK < INF && distKJ < INF &&
                    distIK + distKJ < distIJ) {
                    std::printf(
                        "Validation failed: triangle inequality violated at "
                        "[%zu,%zu,%zu]\n",
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

int main(int argc, char** argv) {
    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numNodes = static_cast<size_t>(std::strtoull(argv[++i], nullptr, 10));
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

    // Matrix indices are deliberately 32-bit in device code for throughput.
    // 65535^2 is the largest square matrix representable by that indexing.
    if (numNodes > 65535 ||
        numNodes > static_cast<size_t>(-1) / std::max(numNodes, size_t{1})) {
        std::fprintf(stderr, "Number of nodes is too large: %zu\n", numNodes);
        return 1;
    }

    std::printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
    std::printf("Number of nodes: %zu\n", numNodes);
    std::printf("Validation: %s\n", validate ? "enabled" : "disabled");

    std::vector<unsigned int> dist(numNodes * numNodes);

    std::printf("Initializing graph...\n");
    initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);

    unsigned int* deviceDist = nullptr;
    const size_t matrixBytes = dist.size() * sizeof(unsigned int);
    if (matrixBytes != 0) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceDist), matrixBytes));
        CUDA_CHECK(cudaMemcpy(deviceDist, dist.data(), matrixBytes,
                              cudaMemcpyHostToDevice));
    } else {
        // Initialize the CUDA runtime even for an empty graph: execution remains
        // unconditionally tied to the requested GPU implementation.
        CUDA_CHECK(cudaFree(nullptr));
    }

    std::printf("Computing shortest paths...\n");
    const auto start = std::chrono::high_resolution_clock::now();

    floydWarshall(deviceDist, static_cast<unsigned int>(numNodes));

    const auto end = std::chrono::high_resolution_clock::now();
    const double seconds = std::chrono::duration<double>(end - start).count();

    if (matrixBytes != 0) {
        CUDA_CHECK(cudaMemcpy(dist.data(), deviceDist, matrixBytes,
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(deviceDist));
    }

    const auto milliseconds = static_cast<long long>(seconds * 1000.0);
    std::printf("Computation time: %lld ms\n", milliseconds);

    const double n = static_cast<double>(numNodes);
    const double ops = n * n * n;
    const double gops = seconds > 0.0 ? ops / seconds / 1.0e9 : 0.0;
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
