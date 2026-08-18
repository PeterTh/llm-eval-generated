#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;
constexpr int TILE_SIZE = 32;
constexpr int THREAD_TILE = 16;

static_assert(TILE_SIZE == 2 * THREAD_TILE, "the CUDA kernels process a 2x2 register tile");

// The original index helper maps (column, row) to conventional row-major storage.
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

[[noreturn]] void abortMpi(const char* message, const int rank) {
    std::fprintf(stderr, "Rank %d: %s\n", rank, message);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

void checkCuda(const cudaError_t result, const char* operation, const int rank) {
    if (result != cudaSuccess) {
        std::fprintf(stderr, "Rank %d: CUDA failure in %s: %s\n",
                     rank, operation, cudaGetErrorString(result));
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        std::abort();
    }
}

void checkMpi(const int result, const char* operation, const int rank) {
    if (result != MPI_SUCCESS) {
        char error[MPI_MAX_ERROR_STRING] = {};
        int length = 0;
        MPI_Error_string(result, error, &length);
        std::fprintf(stderr, "Rank %d: MPI failure in %s: %.*s\n",
                     rank, operation, length, error);
        MPI_Abort(MPI_COMM_WORLD, result);
        std::abort();
    }
}

void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes,
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    // Keep the original deterministic rand_r stream so external hashes remain identical.
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;

    for (size_t i = 0; i < numNodes * numNodes; ++i) {
        dist[i] = rangeMin + static_cast<unsigned int>(
            range * rand_r(&seed) / static_cast<double>(RAND_MAX));
    }

    // Host setup and validation use OpenMP; MPI calls remain on the main thread.
#pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(numNodes); ++i) {
        dist[idx2(static_cast<size_t>(i), static_cast<size_t>(i), numNodes)] = 0;
    }
}

__global__ void initializePathKernel(unsigned int* __restrict__ path,
                                     const size_t elements,
                                     const size_t numNodes,
                                     const size_t globalRowBegin) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < elements) {
        path[index] = static_cast<unsigned int>(globalRowBegin + index / numNodes);
    }
}

// Close the diagonal tile. Each CUDA thread owns four cells; the padded shared-memory
// stride avoids bank conflicts when a Floyd-Warshall step reads a column.
__global__ void phase1Kernel(unsigned int* __restrict__ dist,
                            unsigned int* __restrict__ path,
                            const size_t numNodes,
                            const size_t pivotLocalRow,
                            const size_t pivotGlobalRow) {
    __shared__ unsigned int tile[TILE_SIZE][TILE_SIZE + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    unsigned int values[2][2];
    unsigned int paths[2][2];

#pragma unroll
    for (int ry = 0; ry < 2; ++ry) {
#pragma unroll
        for (int cx = 0; cx < 2; ++cx) {
            const int y = ty + ry * THREAD_TILE;
            const int x = tx + cx * THREAD_TILE;
            const size_t globalY = pivotGlobalRow + static_cast<size_t>(y);
            const size_t globalX = pivotGlobalRow + static_cast<size_t>(x);
            if (globalY < numNodes && globalX < numNodes) {
                const size_t position = (pivotLocalRow + static_cast<size_t>(y)) * numNodes + globalX;
                values[ry][cx] = dist[position];
                paths[ry][cx] = path[position];
            } else {
                values[ry][cx] = INF;
                paths[ry][cx] = 0;
            }
            tile[y][x] = values[ry][cx];
        }
    }
    __syncthreads();

#pragma unroll
    for (int k = 0; k < TILE_SIZE; ++k) {
#pragma unroll
        for (int ry = 0; ry < 2; ++ry) {
#pragma unroll
            for (int cx = 0; cx < 2; ++cx) {
                const int y = ty + ry * THREAD_TILE;
                const int x = tx + cx * THREAD_TILE;
                const unsigned int candidate = tile[y][k] + tile[k][x];
                if (candidate < values[ry][cx]) {
                    values[ry][cx] = candidate;
                    paths[ry][cx] = static_cast<unsigned int>(pivotGlobalRow + k);
                }
                // Row/column k cannot improve when the diagonal is zero. Leaving
                // those shared cells untouched also removes read/write hazards.
                if (y != k && x != k) {
                    tile[y][x] = values[ry][cx];
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int ry = 0; ry < 2; ++ry) {
#pragma unroll
        for (int cx = 0; cx < 2; ++cx) {
            const int y = ty + ry * THREAD_TILE;
            const int x = tx + cx * THREAD_TILE;
            const size_t globalY = pivotGlobalRow + static_cast<size_t>(y);
            const size_t globalX = pivotGlobalRow + static_cast<size_t>(x);
            if (globalY < numNodes && globalX < numNodes) {
                const size_t position = (pivotLocalRow + static_cast<size_t>(y)) * numNodes + globalX;
                dist[position] = values[ry][cx];
                path[position] = paths[ry][cx];
            }
        }
    }
}

// Close every non-diagonal tile in the pivot tile row.
__global__ void phase2RowKernel(unsigned int* __restrict__ dist,
                               unsigned int* __restrict__ path,
                               const size_t numNodes,
                               const size_t pivotLocalRow,
                               const size_t pivotGlobalRow) {
    __shared__ unsigned int diagonal[TILE_SIZE][TILE_SIZE + 1];
    __shared__ unsigned int target[TILE_SIZE][TILE_SIZE + 1];

    size_t tileColumnIndex = static_cast<size_t>(blockIdx.x);
    if (tileColumnIndex >= pivotGlobalRow / TILE_SIZE) {
        ++tileColumnIndex;
    }
    const size_t tileColumn = tileColumnIndex * TILE_SIZE;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    unsigned int values[2][2];
    unsigned int paths[2][2];

#pragma unroll
    for (int ry = 0; ry < 2; ++ry) {
#pragma unroll
        for (int cx = 0; cx < 2; ++cx) {
            const int y = ty + ry * THREAD_TILE;
            const int x = tx + cx * THREAD_TILE;
            const size_t globalY = pivotGlobalRow + static_cast<size_t>(y);
            const size_t diagonalX = pivotGlobalRow + static_cast<size_t>(x);
            const size_t globalX = tileColumn + static_cast<size_t>(x);

            diagonal[y][x] = (globalY < numNodes && diagonalX < numNodes)
                ? dist[(pivotLocalRow + static_cast<size_t>(y)) * numNodes + diagonalX]
                : INF;
            if (globalY < numNodes && globalX < numNodes) {
                const size_t position = (pivotLocalRow + static_cast<size_t>(y)) * numNodes + globalX;
                values[ry][cx] = dist[position];
                paths[ry][cx] = path[position];
            } else {
                values[ry][cx] = INF;
                paths[ry][cx] = 0;
            }
            target[y][x] = values[ry][cx];
        }
    }
    __syncthreads();

#pragma unroll
    for (int k = 0; k < TILE_SIZE; ++k) {
#pragma unroll
        for (int ry = 0; ry < 2; ++ry) {
#pragma unroll
            for (int cx = 0; cx < 2; ++cx) {
                const int y = ty + ry * THREAD_TILE;
                const int x = tx + cx * THREAD_TILE;
                const unsigned int candidate = diagonal[y][k] + target[k][x];
                if (candidate < values[ry][cx]) {
                    values[ry][cx] = candidate;
                    paths[ry][cx] = static_cast<unsigned int>(pivotGlobalRow + k);
                }
                if (y != k) {
                    target[y][x] = values[ry][cx];
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int ry = 0; ry < 2; ++ry) {
#pragma unroll
        for (int cx = 0; cx < 2; ++cx) {
            const int y = ty + ry * THREAD_TILE;
            const int x = tx + cx * THREAD_TILE;
            const size_t globalY = pivotGlobalRow + static_cast<size_t>(y);
            const size_t globalX = tileColumn + static_cast<size_t>(x);
            if (globalY < numNodes && globalX < numNodes) {
                const size_t position = (pivotLocalRow + static_cast<size_t>(y)) * numNodes + globalX;
                dist[position] = values[ry][cx];
                path[position] = paths[ry][cx];
            }
        }
    }
}

// Close the local tiles in the pivot tile column using the broadcast diagonal tile.
__global__ void phase2ColumnKernel(unsigned int* __restrict__ dist,
                                  unsigned int* __restrict__ path,
                                  const unsigned int* __restrict__ pivotRow,
                                  const size_t numNodes,
                                  const size_t localRowBegin,
                                  const size_t localRows,
                                  const size_t pivotGlobalRow) {
    __shared__ unsigned int diagonal[TILE_SIZE][TILE_SIZE + 1];
    __shared__ unsigned int target[TILE_SIZE][TILE_SIZE + 1];

    size_t localTileIndex = static_cast<size_t>(blockIdx.x);
    if (pivotGlobalRow >= localRowBegin && pivotGlobalRow < localRowBegin + localRows) {
        const size_t pivotLocalTile = (pivotGlobalRow - localRowBegin) / TILE_SIZE;
        if (localTileIndex >= pivotLocalTile) {
            ++localTileIndex;
        }
    }
    const size_t localTileRow = localTileIndex * TILE_SIZE;
    const size_t globalTileRow = localRowBegin + localTileRow;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    unsigned int values[2][2];
    unsigned int paths[2][2];

#pragma unroll
    for (int ry = 0; ry < 2; ++ry) {
#pragma unroll
        for (int cx = 0; cx < 2; ++cx) {
            const int y = ty + ry * THREAD_TILE;
            const int x = tx + cx * THREAD_TILE;
            const size_t localY = localTileRow + static_cast<size_t>(y);
            const size_t globalY = globalTileRow + static_cast<size_t>(y);
            const size_t globalX = pivotGlobalRow + static_cast<size_t>(x);

            diagonal[y][x] = (pivotGlobalRow + static_cast<size_t>(y) < numNodes && globalX < numNodes)
                ? pivotRow[static_cast<size_t>(y) * numNodes + globalX]
                : INF;
            if (localY < localRows && globalY < numNodes && globalX < numNodes) {
                const size_t position = localY * numNodes + globalX;
                values[ry][cx] = dist[position];
                paths[ry][cx] = path[position];
            } else {
                values[ry][cx] = INF;
                paths[ry][cx] = 0;
            }
            target[y][x] = values[ry][cx];
        }
    }
    __syncthreads();

#pragma unroll
    for (int k = 0; k < TILE_SIZE; ++k) {
#pragma unroll
        for (int ry = 0; ry < 2; ++ry) {
#pragma unroll
            for (int cx = 0; cx < 2; ++cx) {
                const int y = ty + ry * THREAD_TILE;
                const int x = tx + cx * THREAD_TILE;
                const unsigned int candidate = target[y][k] + diagonal[k][x];
                if (candidate < values[ry][cx]) {
                    values[ry][cx] = candidate;
                    paths[ry][cx] = static_cast<unsigned int>(pivotGlobalRow + k);
                }
                if (x != k) {
                    target[y][x] = values[ry][cx];
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int ry = 0; ry < 2; ++ry) {
#pragma unroll
        for (int cx = 0; cx < 2; ++cx) {
            const int y = ty + ry * THREAD_TILE;
            const int x = tx + cx * THREAD_TILE;
            const size_t localY = localTileRow + static_cast<size_t>(y);
            const size_t globalY = globalTileRow + static_cast<size_t>(y);
            const size_t globalX = pivotGlobalRow + static_cast<size_t>(x);
            if (localY < localRows && globalY < numNodes && globalX < numNodes) {
                const size_t position = localY * numNodes + globalX;
                dist[position] = values[ry][cx];
                path[position] = paths[ry][cx];
            }
        }
    }
}

// Update all remaining local tiles. The pivot row is replicated, while each rank keeps
// its pivot-column tiles local, so this phase requires no communication.
__global__ void phase3Kernel(unsigned int* __restrict__ dist,
                            unsigned int* __restrict__ path,
                            const unsigned int* __restrict__ pivotRow,
                            const size_t numNodes,
                            const size_t localRowBegin,
                            const size_t localRows,
                            const size_t pivotGlobalRow) {
    __shared__ unsigned int columnTile[TILE_SIZE][TILE_SIZE + 1];
    __shared__ unsigned int rowTile[TILE_SIZE][TILE_SIZE + 1];

    size_t tileColumnIndex = static_cast<size_t>(blockIdx.x);
    if (tileColumnIndex >= pivotGlobalRow / TILE_SIZE) {
        ++tileColumnIndex;
    }
    const size_t tileColumn = tileColumnIndex * TILE_SIZE;

    size_t localTileIndex = static_cast<size_t>(blockIdx.y);
    if (pivotGlobalRow >= localRowBegin && pivotGlobalRow < localRowBegin + localRows) {
        const size_t pivotLocalTile = (pivotGlobalRow - localRowBegin) / TILE_SIZE;
        if (localTileIndex >= pivotLocalTile) {
            ++localTileIndex;
        }
    }
    const size_t localTileRow = localTileIndex * TILE_SIZE;
    const size_t globalTileRow = localRowBegin + localTileRow;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    unsigned int values[2][2];
    unsigned int paths[2][2];

#pragma unroll
    for (int ry = 0; ry < 2; ++ry) {
#pragma unroll
        for (int cx = 0; cx < 2; ++cx) {
            const int y = ty + ry * THREAD_TILE;
            const int x = tx + cx * THREAD_TILE;
            const size_t localY = localTileRow + static_cast<size_t>(y);
            const size_t globalY = globalTileRow + static_cast<size_t>(y);
            const size_t pivotX = pivotGlobalRow + static_cast<size_t>(x);
            const size_t globalX = tileColumn + static_cast<size_t>(x);

            columnTile[y][x] = (localY < localRows && globalY < numNodes && pivotX < numNodes)
                ? dist[localY * numNodes + pivotX]
                : INF;
            rowTile[y][x] = (pivotGlobalRow + static_cast<size_t>(y) < numNodes && globalX < numNodes)
                ? pivotRow[static_cast<size_t>(y) * numNodes + globalX]
                : INF;
            if (localY < localRows && globalY < numNodes && globalX < numNodes) {
                const size_t position = localY * numNodes + globalX;
                values[ry][cx] = dist[position];
                paths[ry][cx] = path[position];
            } else {
                values[ry][cx] = INF;
                paths[ry][cx] = 0;
            }
        }
    }
    __syncthreads();

#pragma unroll
    for (int k = 0; k < TILE_SIZE; ++k) {
#pragma unroll
        for (int ry = 0; ry < 2; ++ry) {
#pragma unroll
            for (int cx = 0; cx < 2; ++cx) {
                const int y = ty + ry * THREAD_TILE;
                const int x = tx + cx * THREAD_TILE;
                const unsigned int candidate = columnTile[y][k] + rowTile[k][x];
                if (candidate < values[ry][cx]) {
                    values[ry][cx] = candidate;
                    paths[ry][cx] = static_cast<unsigned int>(pivotGlobalRow + k);
                }
            }
        }
    }

#pragma unroll
    for (int ry = 0; ry < 2; ++ry) {
#pragma unroll
        for (int cx = 0; cx < 2; ++cx) {
            const int y = ty + ry * THREAD_TILE;
            const int x = tx + cx * THREAD_TILE;
            const size_t localY = localTileRow + static_cast<size_t>(y);
            const size_t globalY = globalTileRow + static_cast<size_t>(y);
            const size_t globalX = tileColumn + static_cast<size_t>(x);
            if (localY < localRows && globalY < numNodes && globalX < numNodes) {
                const size_t position = localY * numNodes + globalX;
                dist[position] = values[ry][cx];
                path[position] = paths[ry][cx];
            }
        }
    }
}

bool validateResult(const std::vector<unsigned int>& dist, const size_t numNodes) {
    int diagonalInvalid = 0;
#pragma omp parallel for schedule(static) reduction(| : diagonalInvalid)
    for (long long i = 0; i < static_cast<long long>(numNodes); ++i) {
        const size_t node = static_cast<size_t>(i);
        diagonalInvalid |= (dist[idx2(node, node, numNodes)] != 0);
    }
    if (diagonalInvalid != 0) {
        std::printf("Validation failed: a diagonal element is not zero\n");
        return false;
    }

    const size_t sample = std::min(numNodes, static_cast<size_t>(10));
    unsigned long long firstViolation = std::numeric_limits<unsigned long long>::max();
#pragma omp parallel for collapse(2) schedule(static) reduction(min : firstViolation)
    for (long long ii = 0; ii < static_cast<long long>(sample); ++ii) {
        for (long long jj = 0; jj < static_cast<long long>(sample); ++jj) {
            const size_t i = static_cast<size_t>(ii);
            const size_t j = static_cast<size_t>(jj);
            for (size_t k = 0; k < numNodes; ++k) {
                const unsigned int distIJ = dist[idx2(j, i, numNodes)];
                const unsigned int distIK = dist[idx2(k, i, numNodes)];
                const unsigned int distKJ = dist[idx2(j, k, numNodes)];
                if (distIK < INF && distKJ < INF && distIK + distKJ < distIJ) {
                    const unsigned long long encoded =
                        (static_cast<unsigned long long>(i) * sample + j) * numNodes + k;
                    firstViolation = std::min(firstViolation, encoded);
                }
            }
        }
    }

    if (firstViolation != std::numeric_limits<unsigned long long>::max()) {
        const size_t k = static_cast<size_t>(firstViolation % numNodes);
        const unsigned long long pair = firstViolation / numNodes;
        const size_t j = static_cast<size_t>(pair % sample);
        const size_t i = static_cast<size_t>(pair / sample);
        std::printf("Validation failed: triangle inequality violated at [%zu,%zu,%zu]\n", i, j, k);
        return false;
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
    int provided = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0;
    int worldSize = 1;
    checkMpi(MPI_Comm_rank(MPI_COMM_WORLD, &rank), "MPI_Comm_rank", rank);
    checkMpi(MPI_Comm_size(MPI_COMM_WORLD, &worldSize), "MPI_Comm_size", rank);
    if (provided < MPI_THREAD_FUNNELED) {
        abortMpi("MPI does not provide the required FUNNELED thread support", rank);
    }

    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;
    bool showHelp = false;
    bool argumentsValid = true;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const unsigned long long parsed = std::strtoull(argv[++i], &end, 10);
            if (end == argv[i] || *end != '\0' || parsed > std::numeric_limits<size_t>::max()) {
                argumentsValid = false;
            } else {
                numNodes = static_cast<size_t>(parsed);
            }
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            showHelp = true;
        } else {
            argumentsValid = false;
            if (rank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
            }
        }
    }

    if (showHelp || !argumentsValid) {
        if (rank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return argumentsValid ? 0 : 1;
    }

    if (numNodes != 0 && numNodes > std::numeric_limits<size_t>::max() / numNodes) {
        abortMpi("matrix dimensions overflow size_t", rank);
    }
    if (numNodes > std::numeric_limits<unsigned int>::max()) {
        abortMpi("path indices require at most UINT_MAX nodes", rank);
    }

    int localRank = 0;
    int nodeSize = 1;
    MPI_Comm nodeCommunicator = MPI_COMM_NULL;
    checkMpi(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                                 MPI_INFO_NULL, &nodeCommunicator),
             "MPI_Comm_split_type", rank);
    checkMpi(MPI_Comm_rank(nodeCommunicator, &localRank), "node-local MPI_Comm_rank", rank);
    checkMpi(MPI_Comm_size(nodeCommunicator, &nodeSize), "node-local MPI_Comm_size", rank);

    int deviceCount = 0;
    checkCuda(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount", rank);
    if (deviceCount == 0) {
        abortMpi("the hybrid benchmark requires at least one CUDA device per node", rank);
    }
    const int device = localRank % deviceCount;
    checkCuda(cudaSetDevice(device), "cudaSetDevice", rank);
    cudaDeviceProp deviceProperties{};
    checkCuda(cudaGetDeviceProperties(&deviceProperties, device), "cudaGetDeviceProperties", rank);

    const size_t tileCount = (numNodes + TILE_SIZE - 1) / TILE_SIZE;
    const size_t rankTileBegin = tileCount * static_cast<size_t>(rank) / worldSize;
    const size_t rankTileEnd = tileCount * static_cast<size_t>(rank + 1) / worldSize;
    const size_t localRowBegin = rankTileBegin * TILE_SIZE;
    const size_t localRowEnd = std::min(numNodes, rankTileEnd * TILE_SIZE);
    const size_t localRows = localRowEnd - localRowBegin;
    const size_t localElements = localRows * numNodes;

    std::vector<int> counts(static_cast<size_t>(worldSize));
    std::vector<int> displacements(static_cast<size_t>(worldSize));
    for (int r = 0; r < worldSize; ++r) {
        const size_t tileBegin = tileCount * static_cast<size_t>(r) / worldSize;
        const size_t tileEnd = tileCount * static_cast<size_t>(r + 1) / worldSize;
        const size_t rowBegin = tileBegin * TILE_SIZE;
        const size_t rowEnd = std::min(numNodes, tileEnd * TILE_SIZE);
        const size_t count = (rowEnd - rowBegin) * numNodes;
        const size_t displacement = rowBegin * numNodes;
        if (count > INT_MAX || displacement > INT_MAX) {
            abortMpi("matrix is too large for MPI_Scatterv/MPI_Gatherv integer counts", rank);
        }
        counts[static_cast<size_t>(r)] = static_cast<int>(count);
        displacements[static_cast<size_t>(r)] = static_cast<int>(displacement);
    }

    if (rank == 0) {
        std::printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        std::printf("Number of nodes: %zu\n", numNodes);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Hybrid configuration: %d MPI rank(s), up to %d OpenMP thread(s) per rank\n",
                    worldSize, omp_get_max_threads());
        std::printf("CUDA device on rank 0: %s (tile size %d)\n",
                    deviceProperties.name, TILE_SIZE);
        if (nodeSize > deviceCount) {
            std::printf("Warning: multiple MPI ranks share each CUDA device on this node\n");
        }
    }

    std::vector<unsigned int> distanceMatrix;
    if (rank == 0) {
        distanceMatrix.resize(numNodes * numNodes);
        std::printf("Initializing graph...\n");
        initializeDistanceMatrix(distanceMatrix, numNodes, 1, MAX_DISTANCE);
    }

    unsigned int* localHostDistance = nullptr;
    unsigned int* pivotHost = nullptr;
    unsigned int* deviceDistance = nullptr;
    unsigned int* devicePath = nullptr;
    unsigned int* devicePivotRow = nullptr;
    const size_t allocatedLocalElements = std::max<size_t>(localElements, 1);
    const size_t pivotCapacity = std::max<size_t>(TILE_SIZE * numNodes, 1);

    checkCuda(cudaMallocHost(reinterpret_cast<void**>(&localHostDistance),
                             allocatedLocalElements * sizeof(unsigned int)),
              "cudaMallocHost(local matrix)", rank);
    checkCuda(cudaMallocHost(reinterpret_cast<void**>(&pivotHost),
                             pivotCapacity * sizeof(unsigned int)),
              "cudaMallocHost(pivot row)", rank);
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&deviceDistance),
                         allocatedLocalElements * sizeof(unsigned int)),
              "cudaMalloc(distance)", rank);
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&devicePath),
                         allocatedLocalElements * sizeof(unsigned int)),
              "cudaMalloc(path)", rank);
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&devicePivotRow),
                         pivotCapacity * sizeof(unsigned int)),
              "cudaMalloc(pivot row)", rank);

    cudaStream_t stream = nullptr;
    checkCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
              "cudaStreamCreateWithFlags", rank);

    checkMpi(MPI_Scatterv(rank == 0 ? distanceMatrix.data() : nullptr,
                          counts.data(), displacements.data(), MPI_UNSIGNED,
                          localHostDistance, static_cast<int>(localElements), MPI_UNSIGNED,
                          0, MPI_COMM_WORLD),
             "MPI_Scatterv", rank);

    if (localElements != 0) {
        checkCuda(cudaMemcpyAsync(deviceDistance, localHostDistance,
                                  localElements * sizeof(unsigned int),
                                  cudaMemcpyHostToDevice, stream),
                  "initial matrix copy", rank);
        constexpr int initThreads = 256;
        const size_t initBlocks = (localElements + initThreads - 1) / initThreads;
        initializePathKernel<<<static_cast<unsigned int>(initBlocks), initThreads, 0, stream>>>(
            devicePath, localElements, numNodes, localRowBegin);
        checkCuda(cudaPeekAtLastError(), "initializePathKernel launch", rank);
    }
    checkCuda(cudaStreamSynchronize(stream), "initial device setup", rank);

    if (rank == 0) {
        std::printf("Computing shortest paths...\n");
    }
    checkMpi(MPI_Barrier(MPI_COMM_WORLD), "pre-compute MPI_Barrier", rank);
    const double start = MPI_Wtime();

    const dim3 threads(THREAD_TILE, THREAD_TILE);
    const unsigned int matrixTileCount = static_cast<unsigned int>(tileCount);
    const unsigned int localTileCount = static_cast<unsigned int>(rankTileEnd - rankTileBegin);

    for (size_t pivotTile = 0; pivotTile < tileCount; ++pivotTile) {
        const size_t pivotGlobalRow = pivotTile * TILE_SIZE;
        const size_t pivotRows = std::min(static_cast<size_t>(TILE_SIZE), numNodes - pivotGlobalRow);
        const int pivotOwner = static_cast<int>(
            ((pivotTile + 1) * static_cast<size_t>(worldSize) - 1) / tileCount);

        if (rank == pivotOwner) {
            const size_t pivotLocalRow = pivotGlobalRow - localRowBegin;
            phase1Kernel<<<1, threads, 0, stream>>>(deviceDistance, devicePath, numNodes,
                                                    pivotLocalRow, pivotGlobalRow);
            checkCuda(cudaPeekAtLastError(), "phase1Kernel launch", rank);

            if (matrixTileCount > 1) {
                phase2RowKernel<<<matrixTileCount - 1, threads, 0, stream>>>(
                    deviceDistance, devicePath, numNodes, pivotLocalRow, pivotGlobalRow);
                checkCuda(cudaPeekAtLastError(), "phase2RowKernel launch", rank);
            }

            checkCuda(cudaMemcpyAsync(pivotHost,
                                      deviceDistance + pivotLocalRow * numNodes,
                                      pivotRows * numNodes * sizeof(unsigned int),
                                      cudaMemcpyDeviceToHost, stream),
                      "pivot row device-to-host copy", rank);
            checkCuda(cudaStreamSynchronize(stream), "pivot row production", rank);
        }

        const size_t pivotElements = pivotRows * numNodes;
        if (pivotElements > static_cast<size_t>(INT_MAX)) {
            abortMpi("pivot row is too large for MPI_Bcast", rank);
        }
        checkMpi(MPI_Bcast(pivotHost, static_cast<int>(pivotElements), MPI_UNSIGNED,
                           pivotOwner, MPI_COMM_WORLD),
                 "pivot-row MPI_Bcast", rank);

        checkCuda(cudaMemcpyAsync(devicePivotRow, pivotHost,
                                  pivotElements * sizeof(unsigned int),
                                  cudaMemcpyHostToDevice, stream),
                  "pivot row host-to-device copy", rank);

        const bool ownsPivotTile = pivotTile >= rankTileBegin && pivotTile < rankTileEnd;
        const unsigned int nonPivotLocalTiles = localTileCount - (ownsPivotTile ? 1U : 0U);
        if (nonPivotLocalTiles != 0) {
            phase2ColumnKernel<<<nonPivotLocalTiles, threads, 0, stream>>>(
                deviceDistance, devicePath, devicePivotRow, numNodes,
                localRowBegin, localRows, pivotGlobalRow);
            checkCuda(cudaPeekAtLastError(), "phase2ColumnKernel launch", rank);

            if (matrixTileCount > 1) {
                const dim3 phase3Grid(matrixTileCount - 1, nonPivotLocalTiles);
                phase3Kernel<<<phase3Grid, threads, 0, stream>>>(
                    deviceDistance, devicePath, devicePivotRow, numNodes,
                    localRowBegin, localRows, pivotGlobalRow);
                checkCuda(cudaPeekAtLastError(), "phase3Kernel launch", rank);
            }
        }
    }

    if (localElements != 0) {
        checkCuda(cudaMemcpyAsync(localHostDistance, deviceDistance,
                                  localElements * sizeof(unsigned int),
                                  cudaMemcpyDeviceToHost, stream),
                  "result device-to-host copy", rank);
    }
    checkCuda(cudaStreamSynchronize(stream), "final CUDA synchronization", rank);

    checkMpi(MPI_Gatherv(localHostDistance, static_cast<int>(localElements), MPI_UNSIGNED,
                         rank == 0 ? distanceMatrix.data() : nullptr,
                         counts.data(), displacements.data(), MPI_UNSIGNED,
                         0, MPI_COMM_WORLD),
             "MPI_Gatherv", rank);

    const double localElapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    checkMpi(MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX,
                        0, MPI_COMM_WORLD),
             "timing MPI_Reduce", rank);

    int valid = 1;
    if (rank == 0) {
        const long durationMs = static_cast<long>(elapsed * 1000.0);
        std::printf("Computation time: %ld ms\n", durationMs);
        const double operations = static_cast<double>(numNodes) * numNodes * numNodes;
        const double gops = elapsed > 0.0 ? operations / elapsed / 1.0e9 : 0.0;
        std::printf("Performance: %.3f GOPS\n", gops);

        if (printResults) {
            print_results_int(distanceMatrix, "DistanceMatrix");
        }
        if (validate) {
            std::printf("Validating result...\n");
            valid = validateResult(distanceMatrix, numNodes) ? 1 : 0;
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
    }

    checkMpi(MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD),
             "validation-status MPI_Bcast", rank);

    checkCuda(cudaStreamDestroy(stream), "cudaStreamDestroy", rank);
    checkCuda(cudaFree(devicePivotRow), "cudaFree(pivot row)", rank);
    checkCuda(cudaFree(devicePath), "cudaFree(path)", rank);
    checkCuda(cudaFree(deviceDistance), "cudaFree(distance)", rank);
    checkCuda(cudaFreeHost(pivotHost), "cudaFreeHost(pivot row)", rank);
    checkCuda(cudaFreeHost(localHostDistance), "cudaFreeHost(local matrix)", rank);
    checkMpi(MPI_Comm_free(&nodeCommunicator), "MPI_Comm_free", rank);
    MPI_Finalize();
    return valid ? 0 : 1;
}
