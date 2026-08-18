#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#if __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#endif

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

#ifndef _OPENMP
#error "This benchmark requires OpenMP support"
#endif

#ifndef __CUDACC__
#error "This benchmark must be compiled as CUDA"
#endif

constexpr unsigned int INF = 1000000000U;
constexpr unsigned int MAX_DISTANCE = 200U;
constexpr int CUDA_TILE = 32;
static_assert(sizeof(unsigned int) == sizeof(std::uint32_t));

static int worldRank = 0;

[[noreturn]] void abortRun(const char* message) {
    std::fprintf(stderr, "Rank %d: %s\n", worldRank, message);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

void checkCuda(const cudaError_t result, const char* expression,
               const char* file, const int line) {
    if (result != cudaSuccess) {
        char message[1024];
        std::snprintf(message, sizeof(message),
                      "CUDA error at %s:%d for %s: %s", file, line,
                      expression, cudaGetErrorString(result));
        abortRun(message);
    }
}

void checkMpi(const int result, const char* expression,
              const char* file, const int line) {
    if (result != MPI_SUCCESS) {
        char error[MPI_MAX_ERROR_STRING];
        int length = 0;
        MPI_Error_string(result, error, &length);
        char message[1200];
        std::snprintf(message, sizeof(message),
                      "MPI error at %s:%d for %s: %.*s", file, line,
                      expression, length, error);
        abortRun(message);
    }
}

#define CUDA_CHECK(call) checkCuda((call), #call, __FILE__, __LINE__)
#define MPI_CHECK(call) checkMpi((call), #call, __FILE__, __LINE__)

inline constexpr size_t idx2(const size_t i, const size_t j,
                             const size_t n) noexcept {
    return j * n + i;
}

// rand_r in the original benchmark uses this three-step POSIX/glibc LCG. The
// skip-ahead form lets every MPI rank and OpenMP thread initialize its own rows
// independently while producing exactly the original seed-42 matrix.
unsigned int benchmarkRandR(unsigned int& seed) {
    unsigned int next = seed;
    next = next * 1103515245U + 12345U;
    unsigned int result = (next / 65536U) % 2048U;
    next = next * 1103515245U + 12345U;
    result = (result << 10U) ^ ((next / 65536U) % 1024U);
    next = next * 1103515245U + 12345U;
    result = (result << 10U) ^ ((next / 65536U) % 1024U);
    seed = next;
    return result;
}

unsigned int advanceLcg(unsigned int seed, std::uint64_t steps) {
    unsigned int accumulatedMultiplier = 1U;
    unsigned int accumulatedIncrement = 0U;
    unsigned int multiplier = 1103515245U;
    unsigned int increment = 12345U;
    while (steps != 0) {
        if ((steps & 1U) != 0U) {
            accumulatedMultiplier *= multiplier;
            accumulatedIncrement = accumulatedIncrement * multiplier + increment;
        }
        increment *= multiplier + 1U;
        multiplier *= multiplier;
        steps >>= 1U;
    }
    return accumulatedMultiplier * seed + accumulatedIncrement;
}

void initializeLocalMatrices(std::vector<unsigned int>& dist,
                             std::vector<unsigned int>& path,
                             const size_t numNodes, const size_t rowBegin,
                             const size_t rowCount) {
    const double range = static_cast<double>(MAX_DISTANCE) + 0.0;
#pragma omp parallel for schedule(static)
    for (long long local = 0; local < static_cast<long long>(rowCount); ++local) {
        const size_t localRow = static_cast<size_t>(local);
        const size_t globalRow = rowBegin + localRow;
        unsigned int seed = advanceLcg(
            42U, 3U * static_cast<std::uint64_t>(globalRow * numNodes));
        for (size_t column = 0; column < numNodes; ++column) {
            const size_t offset = localRow * numNodes + column;
            dist[offset] = 1U + static_cast<unsigned int>(
                                    range * benchmarkRandR(seed) /
                                    static_cast<double>(RAND_MAX));
            path[offset] = static_cast<unsigned int>(globalRow);
        }
        dist[localRow * numNodes + globalRow] = 0U;
    }
}

// Phase 1: close the diagonal (pivot) tile. One thread owns one matrix cell.
__global__ __launch_bounds__(CUDA_TILE * CUDA_TILE)
void pivotKernel(unsigned int* __restrict__ dist,
                 unsigned int* __restrict__ path,
                 const size_t n, const size_t localPivotRow,
                 const size_t globalPivot, const int pivotRows) {
    __shared__ unsigned int tile[CUDA_TILE][CUDA_TILE + 1];

    const int x = threadIdx.x;
    const int y = threadIdx.y;
    const bool valid = x < pivotRows && y < pivotRows;
    const size_t offset = (localPivotRow + static_cast<size_t>(y)) * n +
                          globalPivot + static_cast<size_t>(x);
    tile[y][x] = valid ? dist[offset] : INF;
    unsigned int predecessor = valid ? path[offset] : 0U;
    __syncthreads();

#pragma unroll
    for (int k = 0; k < CUDA_TILE; ++k) {
        if (k < pivotRows && valid) {
            const unsigned int candidate = tile[y][k] + tile[k][x];
            if (candidate < tile[y][x]) {
                tile[y][x] = candidate;
                predecessor = static_cast<unsigned int>(globalPivot + k);
            }
        }
        __syncthreads();
    }

    if (valid) {
        dist[offset] = tile[y][x];
        path[offset] = predecessor;
    }
}

// Phase 2: update all tiles in the pivot tile-row.
__global__ __launch_bounds__(CUDA_TILE * CUDA_TILE)
void pivotRowKernel(unsigned int* __restrict__ dist,
                    unsigned int* __restrict__ path,
                    const size_t n, const size_t localPivotRow,
                    const size_t globalPivot, const int pivotRows,
                    const int numTiles) {
    const int columnTile = blockIdx.x;
    if (columnTile == static_cast<int>(globalPivot / CUDA_TILE) ||
        columnTile >= numTiles) {
        return;
    }

    __shared__ unsigned int pivot[CUDA_TILE][CUDA_TILE + 1];
    __shared__ unsigned int current[CUDA_TILE][CUDA_TILE + 1];
    const int x = threadIdx.x;
    const int y = threadIdx.y;
    const size_t column = static_cast<size_t>(columnTile) * CUDA_TILE + x;
    const bool valid = y < pivotRows && column < n;
    const size_t pivotOffset = (localPivotRow + static_cast<size_t>(y)) * n +
                               globalPivot + static_cast<size_t>(x);
    const size_t offset = (localPivotRow + static_cast<size_t>(y)) * n + column;

    pivot[y][x] = (y < pivotRows && x < pivotRows) ? dist[pivotOffset] : INF;
    current[y][x] = valid ? dist[offset] : INF;
    unsigned int predecessor = valid ? path[offset] : 0U;
    __syncthreads();

#pragma unroll
    for (int k = 0; k < CUDA_TILE; ++k) {
        if (k < pivotRows && valid) {
            const unsigned int candidate = pivot[y][k] + current[k][x];
            if (candidate < current[y][x]) {
                current[y][x] = candidate;
                predecessor = static_cast<unsigned int>(globalPivot + k);
            }
        }
        __syncthreads();
    }

    if (valid) {
        dist[offset] = current[y][x];
        path[offset] = predecessor;
    }
}

// Phase 3: update every local tile in the pivot tile-column.
__global__ __launch_bounds__(CUDA_TILE * CUDA_TILE)
void pivotColumnKernel(unsigned int* __restrict__ dist,
                       unsigned int* __restrict__ path,
                       const unsigned int* __restrict__ pivotRowsDevice,
                       const size_t n, const size_t localRowBegin,
                       const size_t localRows, const size_t globalPivot,
                       const int pivotRows) {
    const size_t localTileRow = blockIdx.x;
    const size_t globalRow = localRowBegin + localTileRow * CUDA_TILE;
    if (globalRow == globalPivot) {
        return;
    }

    __shared__ unsigned int pivot[CUDA_TILE][CUDA_TILE + 1];
    __shared__ unsigned int current[CUDA_TILE][CUDA_TILE + 1];
    const int x = threadIdx.x;
    const int y = threadIdx.y;
    const size_t localRow = localTileRow * CUDA_TILE + y;
    const bool valid = localRow < localRows && x < pivotRows;
    const size_t offset = localRow * n + globalPivot + static_cast<size_t>(x);

    pivot[y][x] = (y < pivotRows && x < pivotRows)
                      ? pivotRowsDevice[static_cast<size_t>(y) * n +
                                        globalPivot + x]
                      : INF;
    current[y][x] = valid ? dist[offset] : INF;
    unsigned int predecessor = valid ? path[offset] : 0U;
    __syncthreads();

#pragma unroll
    for (int k = 0; k < CUDA_TILE; ++k) {
        if (k < pivotRows && valid) {
            const unsigned int candidate = current[y][k] + pivot[k][x];
            if (candidate < current[y][x]) {
                current[y][x] = candidate;
                predecessor = static_cast<unsigned int>(globalPivot + k);
            }
        }
        __syncthreads();
    }

    if (valid) {
        dist[offset] = current[y][x];
        path[offset] = predecessor;
    }
}

// Phase 4: the remaining local tiles are independent min-plus products.
__global__ __launch_bounds__(CUDA_TILE * CUDA_TILE)
void remainingKernel(unsigned int* __restrict__ dist,
                     unsigned int* __restrict__ path,
                     const unsigned int* __restrict__ pivotRowsDevice,
                     const size_t n, const size_t localRowBegin,
                     const size_t localRows, const size_t globalPivot,
                     const int pivotRows, const int numTiles) {
    const int columnTile = blockIdx.x;
    const size_t localTileRow = blockIdx.y;
    const size_t globalRow = localRowBegin + localTileRow * CUDA_TILE;
    if (globalRow == globalPivot ||
        columnTile == static_cast<int>(globalPivot / CUDA_TILE) ||
        columnTile >= numTiles) {
        return;
    }

    __shared__ unsigned int left[CUDA_TILE][CUDA_TILE + 1];
    __shared__ unsigned int upper[CUDA_TILE][CUDA_TILE + 1];
    const int x = threadIdx.x;
    const int y = threadIdx.y;
    const size_t localRow = localTileRow * CUDA_TILE + y;
    const size_t column = static_cast<size_t>(columnTile) * CUDA_TILE + x;
    const bool valid = localRow < localRows && column < n;

    left[y][x] = (localRow < localRows && x < pivotRows)
                     ? dist[localRow * n + globalPivot + x]
                     : INF;
    upper[y][x] = (y < pivotRows && column < n)
                      ? pivotRowsDevice[static_cast<size_t>(y) * n + column]
                      : INF;
    __syncthreads();

    if (valid) {
        const size_t offset = localRow * n + column;
        unsigned int best = dist[offset];
        unsigned int predecessor = path[offset];
#pragma unroll
        for (int k = 0; k < CUDA_TILE; ++k) {
            if (k < pivotRows) {
                const unsigned int candidate = left[y][k] + upper[k][x];
                if (candidate < best) {
                    best = candidate;
                    predecessor = static_cast<unsigned int>(globalPivot + k);
                }
            }
        }
        dist[offset] = best;
        path[offset] = predecessor;
    }
}

struct RowDistribution {
    size_t tileBegin = 0;
    size_t tileCount = 0;
    size_t rowBegin = 0;
    size_t rowCount = 0;
    std::vector<int> elementCounts;
    std::vector<int> elementDisplacements;
};

RowDistribution makeDistribution(const size_t n, const int rank,
                                 const int ranks) {
    RowDistribution result;
    const size_t numTiles = (n + CUDA_TILE - 1) / CUDA_TILE;
    const size_t quotient = numTiles / static_cast<size_t>(ranks);
    const size_t remainder = numTiles % static_cast<size_t>(ranks);
    result.elementCounts.resize(ranks);
    result.elementDisplacements.resize(ranks);

    for (int r = 0; r < ranks; ++r) {
        const size_t rTileBegin = static_cast<size_t>(r) * quotient +
                                  std::min(static_cast<size_t>(r), remainder);
        const size_t rTileCount = quotient +
                                  (static_cast<size_t>(r) < remainder ? 1U : 0U);
        const size_t rRowBegin = std::min(n, rTileBegin * CUDA_TILE);
        const size_t rRowEnd = std::min(n, (rTileBegin + rTileCount) * CUDA_TILE);
        const size_t count = (rRowEnd - rRowBegin) * n;
        const size_t displacement = rRowBegin * n;
        if (count > static_cast<size_t>(INT_MAX) ||
            displacement > static_cast<size_t>(INT_MAX)) {
            abortRun("matrix is too large for portable MPI Gatherv counts");
        }
        result.elementCounts[r] = static_cast<int>(count);
        result.elementDisplacements[r] = static_cast<int>(displacement);
        if (r == rank) {
            result.tileBegin = rTileBegin;
            result.tileCount = rTileCount;
            result.rowBegin = rRowBegin;
            result.rowCount = rRowEnd - rRowBegin;
        }
    }
    return result;
}

int ownerOfTile(const size_t tile, const size_t numTiles, const int ranks) {
    const size_t quotient = numTiles / static_cast<size_t>(ranks);
    const size_t remainder = numTiles % static_cast<size_t>(ranks);
    const size_t rowsInLargerRanks = (quotient + 1) * remainder;
    if (tile < rowsInLargerRanks) {
        return static_cast<int>(tile / (quotient + 1));
    }
    return static_cast<int>(remainder +
                            (tile - rowsInLargerRanks) / quotient);
}

bool queryCudaAwareMpi() {
    bool supported = false;
#if defined(OMPI_HAVE_MPI_EXT_CUDA)
    supported = MPIX_Query_cuda_support() != 0;
#endif
    // This override supports CUDA-aware MPI implementations without a portable
    // query API. All ranks must receive the same value (enforced below).
    if (const char* setting = std::getenv("FLOYDWARSHALL_CUDA_AWARE_MPI")) {
        supported = std::strcmp(setting, "0") != 0;
    }
    int local = supported ? 1 : 0;
    int global = 0;
    MPI_CHECK(MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MIN,
                            MPI_COMM_WORLD));
    return global != 0;
}

double distributedFloydWarshall(std::vector<unsigned int>& localDist,
                                std::vector<unsigned int>& localPath,
                                const size_t n,
                                const RowDistribution& distribution,
                                const int rank, const int ranks,
                                const bool cudaAwareMpi) {
    const size_t localElements = distribution.rowCount * n;
    const size_t allocationElements = std::max<size_t>(localElements, 1);
    unsigned int* deviceDist = nullptr;
    unsigned int* devicePath = nullptr;
    unsigned int* devicePivot[2] = {nullptr, nullptr};
    unsigned int* hostPivot = nullptr;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceDist),
                          allocationElements * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&devicePath),
                          allocationElements * sizeof(unsigned int)));
    for (auto& buffer : devicePivot) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&buffer),
                              static_cast<size_t>(CUDA_TILE) * n *
                                  sizeof(unsigned int)));
    }
    if (!cudaAwareMpi) {
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&hostPivot),
                                  static_cast<size_t>(CUDA_TILE) * n *
                                      sizeof(unsigned int)));
    }
    if (localElements != 0) {
        CUDA_CHECK(cudaMemcpy(deviceDist, localDist.data(),
                              localElements * sizeof(unsigned int),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(devicePath, localPath.data(),
                              localElements * sizeof(unsigned int),
                              cudaMemcpyHostToDevice));
    }

    const size_t numTiles = (n + CUDA_TILE - 1) / CUDA_TILE;
    const dim3 threads(CUDA_TILE, CUDA_TILE);
    const dim3 remainingGrid(static_cast<unsigned int>(numTiles),
                             static_cast<unsigned int>(distribution.tileCount));
    cudaEvent_t pivotBufferDone[2];
    for (auto& event : pivotBufferDone) {
        CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    }

    // Match the serial benchmark's timing semantics: allocations and initial
    // transfers are setup, while all CUDA work and MPI communication required
    // by Floyd-Warshall are timed.
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    const double start = MPI_Wtime();
    for (size_t pivotTile = 0; pivotTile < numTiles; ++pivotTile) {
        const int slot = static_cast<int>(pivotTile & 1U);
        if (pivotTile >= 2) {
            CUDA_CHECK(cudaEventSynchronize(pivotBufferDone[slot]));
        }

        const size_t globalPivot = pivotTile * CUDA_TILE;
        const int pivotRows = static_cast<int>(std::min<size_t>(CUDA_TILE,
                                                                n - globalPivot));
        const int owner = ownerOfTile(pivotTile, numTiles, ranks);
        unsigned int* pivotDevice = devicePivot[slot];

        if (rank == owner) {
            const size_t localPivotRow = globalPivot - distribution.rowBegin;
            pivotKernel<<<1, threads>>>(deviceDist, devicePath, n,
                                        localPivotRow, globalPivot, pivotRows);
            pivotRowKernel<<<static_cast<unsigned int>(numTiles), threads>>>(
                deviceDist, devicePath, n, localPivotRow, globalPivot,
                pivotRows, static_cast<int>(numTiles));
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            pivotDevice = deviceDist + localPivotRow * n;
        }

        const int broadcastCount = pivotRows * static_cast<int>(n);
        if (cudaAwareMpi) {
            MPI_CHECK(MPI_Bcast(pivotDevice, broadcastCount, MPI_UNSIGNED,
                                owner, MPI_COMM_WORLD));
        } else {
            if (rank == owner) {
                CUDA_CHECK(cudaMemcpy(hostPivot, pivotDevice,
                                      static_cast<size_t>(broadcastCount) *
                                          sizeof(unsigned int),
                                      cudaMemcpyDeviceToHost));
            }
            MPI_CHECK(MPI_Bcast(hostPivot, broadcastCount, MPI_UNSIGNED,
                                owner, MPI_COMM_WORLD));
            if (rank != owner) {
                CUDA_CHECK(cudaMemcpy(devicePivot[slot], hostPivot,
                                      static_cast<size_t>(broadcastCount) *
                                          sizeof(unsigned int),
                                      cudaMemcpyHostToDevice));
                pivotDevice = devicePivot[slot];
            }
        }

        if (distribution.tileCount != 0) {
            pivotColumnKernel<<<static_cast<unsigned int>(distribution.tileCount),
                                threads>>>(
                deviceDist, devicePath, pivotDevice, n, distribution.rowBegin,
                distribution.rowCount, globalPivot, pivotRows);
            remainingKernel<<<remainingGrid, threads>>>(
                deviceDist, devicePath, pivotDevice, n, distribution.rowBegin,
                distribution.rowCount, globalPivot, pivotRows,
                static_cast<int>(numTiles));
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaEventRecord(pivotBufferDone[slot]));
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    const double localDuration = MPI_Wtime() - start;
    if (localElements != 0) {
        CUDA_CHECK(cudaMemcpy(localDist.data(), deviceDist,
                              localElements * sizeof(unsigned int),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(localPath.data(), devicePath,
                              localElements * sizeof(unsigned int),
                              cudaMemcpyDeviceToHost));
    }

    for (auto& event : pivotBufferDone) {
        CUDA_CHECK(cudaEventDestroy(event));
    }
    if (hostPivot != nullptr) {
        CUDA_CHECK(cudaFreeHost(hostPivot));
    }
    for (auto& buffer : devicePivot) {
        CUDA_CHECK(cudaFree(buffer));
    }
    CUDA_CHECK(cudaFree(devicePath));
    CUDA_CHECK(cudaFree(deviceDist));
    return localDuration;
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
                if (distIK < INF && distKJ < INF && distIK + distKJ < distIJ) {
                    std::printf("Validation failed: triangle inequality violated "
                                "at [%zu,%zu,%zu]\n", i, j, k);
                    return false;
                }
            }
        }
    }
    return true;
}

void printUsage(const char* program) {
    std::printf("Usage: %s [options]\n", program);
    std::printf("Options:\n");
    std::printf("  -n <num>     Number of nodes in the graph (default: 512)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int provided = 0;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) {
        std::fprintf(stderr, "Unable to initialize MPI\n");
        return EXIT_FAILURE;
    }
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
    int ranks = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &worldRank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
    if (provided < MPI_THREAD_FUNNELED) {
        abortRun("MPI does not provide the required thread support");
    }

    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;
    bool showHelp = false;
    bool argumentsValid = true;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            errno = 0;
            char* end = nullptr;
            const unsigned long long parsed = std::strtoull(argv[++i], &end, 10);
            if (errno != 0 || end == argv[i] || *end != '\0' || parsed == 0 ||
                parsed > std::numeric_limits<unsigned int>::max()) {
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
            if (worldRank == 0) {
                std::printf("Unknown or incomplete option: %s\n", argv[i]);
            }
        }
    }

    if (showHelp || !argumentsValid) {
        if (worldRank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return argumentsValid ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    if (numNodes > static_cast<size_t>(INT_MAX) ||
        numNodes > static_cast<size_t>(INT_MAX) / numNodes) {
        if (worldRank == 0) {
            std::fprintf(stderr,
                         "The requested matrix exceeds portable MPI count limits\n");
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    MPI_Comm localCommunicator = MPI_COMM_NULL;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                                  MPI_INFO_NULL, &localCommunicator));
    int localRank = 0;
    MPI_CHECK(MPI_Comm_rank(localCommunicator, &localRank));
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        abortRun("no CUDA accelerator is available");
    }
    const int device = localRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    MPI_CHECK(MPI_Comm_free(&localCommunicator));

    const bool cudaAwareMpi = queryCudaAwareMpi();
    const RowDistribution distribution =
        makeDistribution(numNodes, worldRank, ranks);
    const size_t localElements = distribution.rowCount * numNodes;
    std::vector<unsigned int> globalDist;

    if (worldRank == 0) {
        std::printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        std::printf("Number of nodes: %zu\n", numNodes);
        std::printf("MPI ranks: %d\n", ranks);
        std::printf("OpenMP threads per rank: %d\n", omp_get_max_threads());
        std::printf("CUDA tile: %dx%d\n", CUDA_TILE, CUDA_TILE);
        std::printf("Rank 0 GPU: %s\n", properties.name);
        std::printf("CUDA-aware MPI: %s\n", cudaAwareMpi ? "enabled" : "disabled");
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Initializing graph...\n");
    }

    std::vector<unsigned int> localDist(localElements);
    std::vector<unsigned int> localPath(localElements);
    initializeLocalMatrices(localDist, localPath, numNodes,
                            distribution.rowBegin, distribution.rowCount);

    if (worldRank == 0) {
        std::printf("Computing shortest paths...\n");
    }
    const double localDuration = distributedFloydWarshall(
        localDist, localPath, numNodes, distribution, worldRank, ranks,
        cudaAwareMpi);
    double duration = 0.0;
    MPI_CHECK(MPI_Reduce(&localDuration, &duration, 1, MPI_DOUBLE, MPI_MAX, 0,
                         MPI_COMM_WORLD));

    if (worldRank == 0) {
        const double operations = static_cast<double>(numNodes) *
                                  static_cast<double>(numNodes) *
                                  static_cast<double>(numNodes);
        std::printf("Computation time: %.3f ms\n", duration * 1000.0);
        std::printf("Performance: %.3f GOPS\n", operations / duration / 1.0e9);
    }

    if (printResults || validate) {
        if (worldRank == 0) {
            globalDist.resize(numNodes * numNodes);
        }
        MPI_CHECK(MPI_Gatherv(localDist.data(), static_cast<int>(localElements),
                              MPI_UNSIGNED,
                              worldRank == 0 ? globalDist.data() : nullptr,
                              distribution.elementCounts.data(),
                              distribution.elementDisplacements.data(),
                              MPI_UNSIGNED, 0, MPI_COMM_WORLD));
    }

    int valid = 1;
    if (worldRank == 0) {
        if (printResults) {
            print_results_int(globalDist, "DistanceMatrix");
        }
        if (validate) {
            std::printf("Validating result...\n");
            valid = validateResult(globalDist, numNodes) ? 1 : 0;
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
    }
    MPI_CHECK(MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD));
    MPI_Finalize();
    return valid ? EXIT_SUCCESS : EXIT_FAILURE;
}
