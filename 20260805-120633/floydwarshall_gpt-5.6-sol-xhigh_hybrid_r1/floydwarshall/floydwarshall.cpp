#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

namespace {

constexpr unsigned int INF = 1000000000U;
constexpr unsigned int MAX_DISTANCE = 200U;
constexpr int TILE_SIZE = 32;

static_assert(sizeof(unsigned int) == sizeof(std::uint32_t),
              "This benchmark requires 32-bit unsigned integers");

inline constexpr std::size_t idx2(const std::size_t i, const std::size_t j,
                                  const std::size_t n) noexcept {
    return j * n + i;
}

[[noreturn]] void abortMpi(const char* what, const int error, const int rank) {
    char message[MPI_MAX_ERROR_STRING] = {};
    int length = 0;
    MPI_Error_string(error, message, &length);
    std::fprintf(stderr, "Rank %d: %s failed: %.*s\n", rank, what, length, message);
    MPI_Abort(MPI_COMM_WORLD, error);
    std::abort();
}

void checkMpi(const int error, const char* what, const int rank) {
    if (error != MPI_SUCCESS) {
        abortMpi(what, error, rank);
    }
}

[[noreturn]] void abortCuda(const char* what, const cudaError_t error, const int rank) {
    std::fprintf(stderr, "Rank %d: %s failed: %s\n", rank, what,
                 cudaGetErrorString(error));
    MPI_Abort(MPI_COMM_WORLD, static_cast<int>(error));
    std::abort();
}

void checkCuda(const cudaError_t error, const char* what, const int rank) {
    if (error != cudaSuccess) {
        abortCuda(what, error, rank);
    }
}

void initializeDistanceMatrix(std::vector<unsigned int>& dist,
                              const std::size_t numNodes,
                              const std::size_t paddedNodes,
                              const unsigned int rangeMin,
                              const unsigned int rangeMax) {
    // Padding makes all GPU tiles full, eliminating boundary branches in the
    // hot kernels.  It is initialized in parallel because it can be large.
#pragma omp parallel for schedule(static)
    for (std::int64_t row = 0; row < static_cast<std::int64_t>(paddedNodes); ++row) {
        std::fill_n(dist.data() + static_cast<std::size_t>(row) * paddedNodes,
                    paddedNodes, INF);
    }

    // Preserve the baseline's rand_r stream exactly so result hashes remain
    // directly comparable with the original benchmark.
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;
    for (std::size_t row = 0; row < numNodes; ++row) {
        for (std::size_t column = 0; column < numNodes; ++column) {
            dist[row * paddedNodes + column] =
                rangeMin + static_cast<unsigned int>(
                               range * rand_r(&seed) / static_cast<double>(RAND_MAX));
        }
    }

#pragma omp parallel for schedule(static)
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(numNodes); ++i) {
        const auto node = static_cast<std::size_t>(i);
        dist[node * paddedNodes + node] = 0;
    }
}

// Close the diagonal tile for one blocked Floyd-Warshall round.
__global__ void phase1Kernel(unsigned int* __restrict__ dist,
                             const std::size_t stride,
                             const int localPivotTile,
                             const int pivotTile) {
    __shared__ unsigned int tile[TILE_SIZE][TILE_SIZE + 1];

    const int x = static_cast<int>(threadIdx.x);
    const int y = static_cast<int>(threadIdx.y);
    const std::size_t row =
        static_cast<std::size_t>(localPivotTile * TILE_SIZE + y);
    const std::size_t column =
        static_cast<std::size_t>(pivotTile * TILE_SIZE + x);

    tile[y][x] = dist[row * stride + column];
    for (int k = 0; k < TILE_SIZE; ++k) {
        __syncthreads();
        const unsigned int candidate = tile[y][k] + tile[k][x];
        __syncthreads();
        if (candidate < tile[y][x]) {
            tile[y][x] = candidate;
        }
    }
    __syncthreads();
    dist[row * stride + column] = tile[y][x];
}

// Update every tile in the pivot tile-row on the rank that owns it.
__global__ void phase2RowKernel(unsigned int* __restrict__ dist,
                                const std::size_t stride,
                                const int localPivotTile,
                                const int pivotTile,
                                const int tileCount) {
    const int columnTile = static_cast<int>(blockIdx.x);
    if (columnTile >= tileCount || columnTile == pivotTile) {
        return;
    }

    __shared__ unsigned int pivot[TILE_SIZE][TILE_SIZE + 1];
    __shared__ unsigned int rowTile[TILE_SIZE][TILE_SIZE + 1];

    const int x = static_cast<int>(threadIdx.x);
    const int y = static_cast<int>(threadIdx.y);
    const std::size_t row =
        static_cast<std::size_t>(localPivotTile * TILE_SIZE + y);
    const std::size_t pivotColumn =
        static_cast<std::size_t>(pivotTile * TILE_SIZE + x);
    const std::size_t column =
        static_cast<std::size_t>(columnTile * TILE_SIZE + x);

    pivot[y][x] = dist[row * stride + pivotColumn];
    rowTile[y][x] = dist[row * stride + column];
    __syncthreads();

    unsigned int best = rowTile[y][x];
#pragma unroll
    for (int k = 0; k < TILE_SIZE; ++k) {
        const unsigned int candidate = pivot[y][k] + rowTile[k][x];
        best = candidate < best ? candidate : best;
    }
    dist[row * stride + column] = best;
}

// Update the pivot-column tile on every local tile-row.
__global__ void phase2ColumnKernel(unsigned int* __restrict__ dist,
                                   const unsigned int* __restrict__ pivotRow,
                                   const std::size_t stride,
                                   const int localTileCount,
                                   const int globalTileStart,
                                   const int pivotTile) {
    const int localRowTile = static_cast<int>(blockIdx.x);
    if (localRowTile >= localTileCount ||
        globalTileStart + localRowTile == pivotTile) {
        return;
    }

    __shared__ unsigned int pivot[TILE_SIZE][TILE_SIZE + 1];
    __shared__ unsigned int columnTile[TILE_SIZE][TILE_SIZE + 1];

    const int x = static_cast<int>(threadIdx.x);
    const int y = static_cast<int>(threadIdx.y);
    const std::size_t localRow =
        static_cast<std::size_t>(localRowTile * TILE_SIZE + y);
    const std::size_t pivotColumn =
        static_cast<std::size_t>(pivotTile * TILE_SIZE + x);

    pivot[y][x] = pivotRow[static_cast<std::size_t>(y) * stride + pivotColumn];
    columnTile[y][x] = dist[localRow * stride + pivotColumn];
    __syncthreads();

    unsigned int best = columnTile[y][x];
#pragma unroll
    for (int k = 0; k < TILE_SIZE; ++k) {
        const unsigned int candidate = columnTile[y][k] + pivot[k][x];
        best = candidate < best ? candidate : best;
    }
    dist[localRow * stride + pivotColumn] = best;
}

// The dominant phase: update all non-pivot tiles owned by this rank.
__global__ void phase3Kernel(unsigned int* __restrict__ dist,
                             const unsigned int* __restrict__ pivotRow,
                             const std::size_t stride,
                             const int localTileCount,
                             const int globalTileStart,
                             const int pivotTile,
                             const int tileCount) {
    const int columnTileIndex = static_cast<int>(blockIdx.x);
    const int localRowTile = static_cast<int>(blockIdx.y);
    if (columnTileIndex >= tileCount || localRowTile >= localTileCount ||
        columnTileIndex == pivotTile ||
        globalTileStart + localRowTile == pivotTile) {
        return;
    }

    __shared__ unsigned int columnTile[TILE_SIZE][TILE_SIZE + 1];
    __shared__ unsigned int rowTile[TILE_SIZE][TILE_SIZE + 1];

    const int x = static_cast<int>(threadIdx.x);
    const int y = static_cast<int>(threadIdx.y);
    const std::size_t localRow =
        static_cast<std::size_t>(localRowTile * TILE_SIZE + y);
    const std::size_t pivotColumn =
        static_cast<std::size_t>(pivotTile * TILE_SIZE + x);
    const std::size_t column =
        static_cast<std::size_t>(columnTileIndex * TILE_SIZE + x);

    columnTile[y][x] = dist[localRow * stride + pivotColumn];
    rowTile[y][x] =
        pivotRow[static_cast<std::size_t>(y) * stride + column];
    __syncthreads();

    unsigned int best = dist[localRow * stride + column];
#pragma unroll
    for (int k = 0; k < TILE_SIZE; ++k) {
        const unsigned int candidate = columnTile[y][k] + rowTile[k][x];
        best = candidate < best ? candidate : best;
    }
    dist[localRow * stride + column] = best;
}

int tileOwner(const int tile, const int rankCount, const int tileCount) {
    const int base = tileCount / rankCount;
    const int extra = tileCount % rankCount;
    const int largerRegion = (base + 1) * extra;
    if (tile < largerRegion) {
        return tile / (base + 1);
    }
    return extra + (tile - largerRegion) / base;
}

void floydWarshall(unsigned int* deviceLocal,
                   unsigned int* devicePivotRow,
                   unsigned int* hostPivotRow,
                   const std::size_t paddedNodes,
                   const int tileCount,
                   const int localTileCount,
                   const int globalTileStart,
                   const int rankCount,
                   const int rank,
                   cudaStream_t stream) {
    const dim3 threads(TILE_SIZE, TILE_SIZE);
    const std::size_t pivotElements =
        static_cast<std::size_t>(TILE_SIZE) * paddedNodes;
    const std::size_t pivotBytes = pivotElements * sizeof(unsigned int);
    const int pivotMpiCount = static_cast<int>(pivotElements);

    for (int pivotTile = 0; pivotTile < tileCount; ++pivotTile) {
        const int owner = tileOwner(pivotTile, rankCount, tileCount);

        if (rank == owner) {
            const int localPivotTile = pivotTile - globalTileStart;
            phase1Kernel<<<1, threads, 0, stream>>>(
                deviceLocal, paddedNodes, localPivotTile, pivotTile);
            phase2RowKernel<<<static_cast<unsigned int>(tileCount), threads, 0,
                              stream>>>(deviceLocal, paddedNodes, localPivotTile,
                                       pivotTile, tileCount);
            checkCuda(cudaGetLastError(), "pivot-row CUDA kernels", rank);

            const unsigned int* source =
                deviceLocal + static_cast<std::size_t>(localPivotTile * TILE_SIZE) *
                                  paddedNodes;
            checkCuda(cudaMemcpyAsync(hostPivotRow, source, pivotBytes,
                                      cudaMemcpyDeviceToHost, stream),
                      "copy pivot row to host", rank);
            checkCuda(cudaStreamSynchronize(stream), "synchronize pivot row", rank);
        }

        // Pinned host staging works with every MPI implementation.  One
        // broadcast per 32 vertices amortizes collective latency and carries
        // the complete row needed by the remaining GPU phases.
        checkMpi(MPI_Bcast(hostPivotRow, pivotMpiCount, MPI_UNSIGNED, owner,
                           MPI_COMM_WORLD),
                 "MPI_Bcast", rank);

        if (rank == owner) {
            const int localPivotTile = pivotTile - globalTileStart;
            const unsigned int* source =
                deviceLocal + static_cast<std::size_t>(localPivotTile * TILE_SIZE) *
                                  paddedNodes;
            checkCuda(cudaMemcpyAsync(devicePivotRow, source, pivotBytes,
                                      cudaMemcpyDeviceToDevice, stream),
                      "copy local pivot row", rank);
        } else {
            checkCuda(cudaMemcpyAsync(devicePivotRow, hostPivotRow, pivotBytes,
                                      cudaMemcpyHostToDevice, stream),
                      "copy received pivot row", rank);
        }

        if (localTileCount > 0) {
            phase2ColumnKernel<<<static_cast<unsigned int>(localTileCount), threads,
                                 0, stream>>>(deviceLocal, devicePivotRow,
                                              paddedNodes, localTileCount,
                                              globalTileStart, pivotTile);
            const dim3 blocks(static_cast<unsigned int>(tileCount),
                              static_cast<unsigned int>(localTileCount));
            phase3Kernel<<<blocks, threads, 0, stream>>>(
                deviceLocal, devicePivotRow, paddedNodes, localTileCount,
                globalTileStart, pivotTile, tileCount);
            checkCuda(cudaGetLastError(), "distributed CUDA update kernels", rank);
        }
    }
}

bool validateResult(const std::vector<unsigned int>& dist,
                    const std::size_t numNodes) {
    for (std::size_t i = 0; i < numNodes; ++i) {
        if (dist[idx2(i, i, numNodes)] != 0) {
            std::printf("Validation failed: diagonal element [%zu,%zu] is not zero\n",
                        i, i);
            return false;
        }
    }

    const std::size_t sampleNodes = std::min(numNodes, std::size_t{10});
    int valid = 1;
    std::size_t badI = 0;
    std::size_t badJ = 0;
    std::size_t badK = 0;
#pragma omp parallel for collapse(2) schedule(static)
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(sampleNodes); ++i) {
        for (std::int64_t j = 0; j < static_cast<std::int64_t>(sampleNodes); ++j) {
            for (std::size_t k = 0; k < numNodes; ++k) {
                const auto source = static_cast<std::size_t>(i);
                const auto destination = static_cast<std::size_t>(j);
                const unsigned int distIJ = dist[idx2(destination, source, numNodes)];
                const unsigned int distIK = dist[idx2(k, source, numNodes)];
                const unsigned int distKJ = dist[idx2(destination, k, numNodes)];
                if (distIK < INF && distKJ < INF && distIK + distKJ < distIJ) {
#pragma omp critical
                    {
                        if (valid != 0) {
                            valid = 0;
                            badI = source;
                            badJ = destination;
                            badK = k;
                        }
                    }
                }
            }
        }
    }

    if (valid == 0) {
        std::printf(
            "Validation failed: triangle inequality violated at [%zu,%zu,%zu]\n",
            badI, badJ, badK);
        return false;
    }
    return true;
}

void printUsage(const char* programName) {
    std::printf("Usage: %s [options]\n", programName);
    std::printf("Options:\n");
    std::printf("  -n <num>     Number of nodes in the graph (default: 512)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

}  // namespace

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    const int initError =
        MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (initError != MPI_SUCCESS) {
        std::fprintf(stderr, "MPI_Init_thread failed\n");
        return 1;
    }

    int rank = 0;
    int rankCount = 1;
    checkMpi(MPI_Comm_rank(MPI_COMM_WORLD, &rank), "MPI_Comm_rank", rank);
    checkMpi(MPI_Comm_size(MPI_COMM_WORLD, &rankCount), "MPI_Comm_size", rank);
    checkMpi(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN),
             "MPI_Comm_set_errhandler", rank);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) {
            std::fprintf(stderr, "MPI does not provide MPI_THREAD_FUNNELED\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    std::size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;
    bool showHelp = false;
    bool argumentsValid = true;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const unsigned long long value = std::strtoull(argv[++i], &end, 10);
            if (end == argv[i] || *end != '\0' ||
                value > std::numeric_limits<std::size_t>::max()) {
                argumentsValid = false;
            } else {
                numNodes = static_cast<std::size_t>(value);
            }
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            showHelp = true;
        } else {
            if (rank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
            }
            argumentsValid = false;
        }
    }

    if (showHelp || !argumentsValid) {
        if (rank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return argumentsValid ? 0 : 1;
    }

    if (numNodes > std::numeric_limits<std::size_t>::max() - (TILE_SIZE - 1)) {
        if (rank == 0) {
            std::fprintf(stderr, "The requested graph is too large\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const std::size_t tileCountSize =
        (numNodes + TILE_SIZE - 1) / TILE_SIZE;
    if (tileCountSize > static_cast<std::size_t>(INT_MAX)) {
        if (rank == 0) {
            std::fprintf(stderr, "The requested graph exceeds CUDA grid limits\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int tileCount = static_cast<int>(tileCountSize);
    const std::size_t paddedNodes = tileCountSize * TILE_SIZE;
    if (paddedNodes != 0 &&
        paddedNodes > std::numeric_limits<std::size_t>::max() / paddedNodes) {
        if (rank == 0) {
            std::fprintf(stderr, "The requested matrix size overflows size_t\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Assign whole, contiguous tile-rows to ranks. This gives balanced GPU
    // work while keeping scatter/gather and pivot-row access contiguous.
    std::vector<int> tilesPerRank(static_cast<std::size_t>(rankCount));
    std::vector<int> tileStarts(static_cast<std::size_t>(rankCount));
    std::vector<int> matrixCounts(static_cast<std::size_t>(rankCount));
    std::vector<int> matrixDisplacements(static_cast<std::size_t>(rankCount));
    const int baseTiles = tileCount / rankCount;
    const int extraTiles = tileCount % rankCount;
    int nextTile = 0;
    for (int process = 0; process < rankCount; ++process) {
        tilesPerRank[process] = baseTiles + (process < extraTiles ? 1 : 0);
        tileStarts[process] = nextTile;
        const std::size_t count =
            static_cast<std::size_t>(tilesPerRank[process]) * TILE_SIZE * paddedNodes;
        const std::size_t displacement =
            static_cast<std::size_t>(nextTile) * TILE_SIZE * paddedNodes;
        if (count > static_cast<std::size_t>(INT_MAX) ||
            displacement > static_cast<std::size_t>(INT_MAX)) {
            if (rank == 0) {
                std::fprintf(stderr,
                             "Matrix exceeds the count range of MPI collectives\n");
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        matrixCounts[process] = static_cast<int>(count);
        matrixDisplacements[process] = static_cast<int>(displacement);
        nextTile += tilesPerRank[process];
    }
    const std::size_t pivotElements =
        static_cast<std::size_t>(TILE_SIZE) * paddedNodes;
    if (pivotElements > static_cast<std::size_t>(INT_MAX)) {
        if (rank == 0) {
            std::fprintf(stderr, "Pivot row exceeds the count range of MPI_Bcast\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // One rank per local GPU in the normal launch configuration. Modulo keeps
    // oversubscribed/debug launches functional without changing the algorithm.
    MPI_Comm localCommunicator = MPI_COMM_NULL;
    checkMpi(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                                 MPI_INFO_NULL, &localCommunicator),
             "MPI_Comm_split_type", rank);
    int localRank = 0;
    checkMpi(MPI_Comm_rank(localCommunicator, &localRank), "local MPI_Comm_rank",
             rank);
    int deviceCount = 0;
    checkCuda(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount", rank);
    if (deviceCount == 0) {
        if (rank == 0) {
            std::fprintf(stderr, "No CUDA devices are available\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    checkCuda(cudaSetDevice(localRank % deviceCount), "cudaSetDevice", rank);
    checkMpi(MPI_Comm_free(&localCommunicator), "MPI_Comm_free", rank);

    if (rank == 0) {
        std::printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        std::printf("Number of nodes: %zu\n", numNodes);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Initializing graph...\n");
    }

    std::vector<unsigned int> globalPadded;
    if (rank == 0) {
        globalPadded.resize(paddedNodes * paddedNodes);
        initializeDistanceMatrix(globalPadded, numNodes, paddedNodes, 1,
                                 MAX_DISTANCE);
    }

    const int localMatrixCount = matrixCounts[rank];
    const std::size_t localAllocationElements =
        std::max<std::size_t>(static_cast<std::size_t>(localMatrixCount), 1);
    const std::size_t pivotAllocationElements =
        std::max<std::size_t>(pivotElements, 1);
    unsigned int* hostLocal = nullptr;
    unsigned int* hostPivotRow = nullptr;
    unsigned int* deviceLocal = nullptr;
    unsigned int* devicePivotRow = nullptr;
    checkCuda(cudaMallocHost(reinterpret_cast<void**>(&hostLocal),
                             localAllocationElements * sizeof(unsigned int)),
              "cudaMallocHost local matrix", rank);
    checkCuda(cudaMallocHost(reinterpret_cast<void**>(&hostPivotRow),
                             pivotAllocationElements * sizeof(unsigned int)),
              "cudaMallocHost pivot row", rank);
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&deviceLocal),
                         localAllocationElements * sizeof(unsigned int)),
              "cudaMalloc local matrix", rank);
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&devicePivotRow),
                         pivotAllocationElements * sizeof(unsigned int)),
              "cudaMalloc pivot row", rank);

    const unsigned int* scatterSource = rank == 0 ? globalPadded.data() : nullptr;
    checkMpi(MPI_Scatterv(scatterSource, matrixCounts.data(),
                          matrixDisplacements.data(), MPI_UNSIGNED, hostLocal,
                          localMatrixCount, MPI_UNSIGNED, 0, MPI_COMM_WORLD),
             "MPI_Scatterv", rank);
    const bool resultNeeded = printResults || validate;
    if (rank == 0 && !resultNeeded) {
        // The default benchmark does not observe the final matrix. Release the
        // root-only initialization copy before the timed computation so large
        // distributed runs retain only their rank-local slabs.
        std::vector<unsigned int>().swap(globalPadded);
    }
    if (localMatrixCount > 0) {
        checkCuda(cudaMemcpy(deviceLocal, hostLocal,
                             static_cast<std::size_t>(localMatrixCount) *
                                 sizeof(unsigned int),
                             cudaMemcpyHostToDevice),
                  "copy local matrix to GPU", rank);
    }

    cudaStream_t stream = nullptr;
    checkCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
              "cudaStreamCreateWithFlags", rank);

    if (rank == 0) {
        std::printf("Computing shortest paths...\n");
    }
    checkMpi(MPI_Barrier(MPI_COMM_WORLD), "pre-compute MPI_Barrier", rank);
    const double start = MPI_Wtime();

    floydWarshall(deviceLocal, devicePivotRow, hostPivotRow, paddedNodes,
                  tileCount, tilesPerRank[rank], tileStarts[rank], rankCount,
                  rank, stream);

    checkCuda(cudaStreamSynchronize(stream), "final CUDA synchronization", rank);
    checkMpi(MPI_Barrier(MPI_COMM_WORLD), "post-compute MPI_Barrier", rank);
    const double localElapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    checkMpi(MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0,
                        MPI_COMM_WORLD),
             "MPI_Reduce", rank);

    if (resultNeeded) {
        if (localMatrixCount > 0) {
            checkCuda(cudaMemcpy(hostLocal, deviceLocal,
                                 static_cast<std::size_t>(localMatrixCount) *
                                     sizeof(unsigned int),
                                 cudaMemcpyDeviceToHost),
                      "copy local result to host", rank);
        }
        unsigned int* gatherDestination =
            rank == 0 ? globalPadded.data() : nullptr;
        checkMpi(MPI_Gatherv(hostLocal, localMatrixCount, MPI_UNSIGNED,
                             gatherDestination, matrixCounts.data(),
                             matrixDisplacements.data(), MPI_UNSIGNED, 0,
                             MPI_COMM_WORLD),
                 "MPI_Gatherv", rank);
    }

    int returnCode = 0;
    if (rank == 0) {
        const long long durationMs =
            static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                       std::chrono::duration<double>(elapsed))
                                       .count());
        std::printf("Computation time: %lld ms\n", durationMs);
        const double operations = static_cast<double>(numNodes) *
                                  static_cast<double>(numNodes) *
                                  static_cast<double>(numNodes);
        const double gops = elapsed > 0.0 ? operations / elapsed / 1.0e9 : 0.0;
        std::printf("Performance: %.3f GOPS\n", gops);

        std::vector<unsigned int> result;
        if (resultNeeded && numNodes == paddedNodes) {
            result.swap(globalPadded);
        } else if (resultNeeded) {
            result.resize(numNodes * numNodes);
#pragma omp parallel for schedule(static)
            for (std::int64_t row = 0; row < static_cast<std::int64_t>(numNodes);
                 ++row) {
                const auto sourceOffset =
                    static_cast<std::size_t>(row) * paddedNodes;
                const auto destinationOffset =
                    static_cast<std::size_t>(row) * numNodes;
                std::copy_n(globalPadded.data() + sourceOffset, numNodes,
                            result.data() + destinationOffset);
            }
        }

        if (printResults) {
            print_results_int(result, "DistanceMatrix");
        }
        if (validate) {
            std::printf("Validating result...\n");
            if (validateResult(result, numNodes)) {
                std::printf("Validation: PASSED\n");
            } else {
                std::printf("Validation: FAILED\n");
                returnCode = 1;
            }
        }
    }

    checkMpi(MPI_Bcast(&returnCode, 1, MPI_INT, 0, MPI_COMM_WORLD),
             "return-code MPI_Bcast", rank);
    checkCuda(cudaStreamDestroy(stream), "cudaStreamDestroy", rank);
    checkCuda(cudaFree(devicePivotRow), "cudaFree pivot row", rank);
    checkCuda(cudaFree(deviceLocal), "cudaFree local matrix", rank);
    checkCuda(cudaFreeHost(hostPivotRow), "cudaFreeHost pivot row", rank);
    checkCuda(cudaFreeHost(hostLocal), "cudaFreeHost local matrix", rank);
    checkMpi(MPI_Finalize(), "MPI_Finalize", rank);
    return returnCode;
}
