#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#if __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#endif
#include <omp.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;
constexpr int TILE_SIZE = 32;
constexpr int ROWS_PER_BLOCK = 8;

// The original program stores a source row contiguously: idx2(destination, source).
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

[[noreturn]] void abortWithMessage(const int rank, const char* message) {
    std::fprintf(stderr, "Rank %d: %s\n", rank, message);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

void checkMpi(const int error, const char* expression, const int rank) {
    if (error == MPI_SUCCESS) {
        return;
    }

    char errorString[MPI_MAX_ERROR_STRING]{};
    int length = 0;
    MPI_Error_string(error, errorString, &length);
    std::fprintf(stderr, "Rank %d: MPI call %s failed: %.*s\n", rank, expression,
                 length, errorString);
    MPI_Abort(MPI_COMM_WORLD, error);
}

void checkCuda(const cudaError_t error, const char* expression, const int line,
               const int rank) {
    if (error == cudaSuccess) {
        return;
    }

    std::fprintf(stderr, "Rank %d: CUDA call %s failed at line %d: %s\n", rank,
                 expression, line, cudaGetErrorString(error));
    MPI_Abort(MPI_COMM_WORLD, static_cast<int>(error));
}

#define MPI_CHECK(call) checkMpi((call), #call, rank)
#define CUDA_CHECK(call) checkCuda((call), #call, __LINE__, rank)

void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes,
                              const unsigned int rangeMin,
                              const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;

    // This loop deliberately remains ordered so that graph generation is bit-for-bit
    // identical to the original rand_r sequence.
    for (size_t i = 0; i < numNodes * numNodes; ++i) {
        dist[i] = rangeMin + static_cast<unsigned int>(
                                 range * rand_r(&seed) / static_cast<double>(RAND_MAX));
    }

#pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(numNodes); ++i) {
        const size_t node = static_cast<size_t>(i);
        dist[idx2(node, node, numNodes)] = 0;
    }
}

void initializeLocalPathMatrix(std::vector<unsigned int>& path,
                               const size_t firstGlobalRow,
                               const size_t numNodes) {
#pragma omp parallel for schedule(static)
    for (long long element = 0; element < static_cast<long long>(path.size()); ++element) {
        const size_t localRow = static_cast<size_t>(element) / numNodes;
        path[static_cast<size_t>(element)] =
            static_cast<unsigned int>(firstGlobalRow + localRow);
    }
}

// Close the diagonal tile. A single CUDA block provides the synchronization that
// Floyd-Warshall requires between successive intermediate vertices.
__global__ void closePivotTile(unsigned int* __restrict__ dist,
                               unsigned int* __restrict__ path,
                               const size_t pitch, const size_t localPivotRow,
                               const size_t globalPivot, const int tileRows) {
    __shared__ unsigned int tile[TILE_SIZE][TILE_SIZE + 1];

    const int column = threadIdx.x;
    const int row = threadIdx.y;
    const bool active = row < tileRows && column < tileRows;
    const size_t matrixIndex =
        (localPivotRow + static_cast<size_t>(row)) * pitch + globalPivot + column;

    tile[row][column] = active ? dist[matrixIndex] : INF;
    unsigned int predecessor = active ? path[matrixIndex] : 0;
    __syncthreads();

    for (int k = 0; k < tileRows; ++k) {
        if (active) {
            const unsigned int candidate = tile[row][k] + tile[k][column];
            if (candidate < tile[row][column]) {
                tile[row][column] = candidate;
                predecessor = static_cast<unsigned int>(globalPivot + k);
            }
        }
        __syncthreads();
    }

    if (active) {
        dist[matrixIndex] = tile[row][column];
        path[matrixIndex] = predecessor;
    }
}

// Update the remainder of the pivot rows. rawPivotRows is an immutable snapshot
// from before this round, avoiding inter-block races while using the closed tile.
__global__ void updatePivotRows(unsigned int* __restrict__ dist,
                                unsigned int* __restrict__ path,
                                const unsigned int* __restrict__ rawPivotRows,
                                const size_t pitch, const size_t localPivotRow,
                                const size_t globalPivot, const int tileRows) {
    const size_t column = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int row = static_cast<int>(blockIdx.y) * blockDim.y + threadIdx.y;
    if (column >= pitch || row >= tileRows ||
        (column >= globalPivot && column < globalPivot + tileRows)) {
        return;
    }

    const size_t rawIndex = static_cast<size_t>(row) * pitch + column;
    const size_t outputIndex = (localPivotRow + static_cast<size_t>(row)) * pitch + column;
    unsigned int best = rawPivotRows[rawIndex];
    unsigned int predecessor = path[outputIndex];

#pragma unroll
    for (int k = 0; k < TILE_SIZE; ++k) {
        if (k < tileRows) {
            const unsigned int toIntermediate =
                dist[(localPivotRow + static_cast<size_t>(row)) * pitch + globalPivot + k];
            const unsigned int candidate =
                toIntermediate + rawPivotRows[static_cast<size_t>(k) * pitch + column];
            if (candidate < best) {
                best = candidate;
                predecessor = static_cast<unsigned int>(globalPivot + k);
            }
        }
    }

    dist[outputIndex] = best;
    path[outputIndex] = predecessor;
}

// Update each local row's columns that intersect the pivot tile. Each group of
// TILE_SIZE threads first snapshots those columns in shared memory.
__global__ void updatePivotColumns(unsigned int* __restrict__ dist,
                                   unsigned int* __restrict__ path,
                                   const unsigned int* __restrict__ pivotRows,
                                   const size_t localRows, const size_t pitch,
                                   const size_t firstGlobalRow,
                                   const size_t globalPivot, const int tileRows) {
    __shared__ unsigned int columns[ROWS_PER_BLOCK][TILE_SIZE];
    __shared__ unsigned int pivotTile[TILE_SIZE][TILE_SIZE + 1];

    const int tileColumn = threadIdx.x;
    const int linearThread = threadIdx.y * TILE_SIZE + threadIdx.x;
    const size_t localRow =
        static_cast<size_t>(blockIdx.x) * ROWS_PER_BLOCK + threadIdx.y;
    const size_t globalRow = firstGlobalRow + localRow;
    const bool validRow = localRow < localRows;
    const bool validColumn = tileColumn < tileRows;

    if (validRow && validColumn) {
        columns[threadIdx.y][tileColumn] =
            dist[localRow * pitch + globalPivot + tileColumn];
    }
    for (int element = linearThread; element < TILE_SIZE * TILE_SIZE;
         element += TILE_SIZE * ROWS_PER_BLOCK) {
        const int pivotRow = element / TILE_SIZE;
        const int pivotColumn = element % TILE_SIZE;
        pivotTile[pivotRow][pivotColumn] =
            (pivotRow < tileRows && pivotColumn < tileRows)
                ? pivotRows[static_cast<size_t>(pivotRow) * pitch + globalPivot +
                            pivotColumn]
                : INF;
    }
    __syncthreads();

    if (!validRow || !validColumn ||
        (globalRow >= globalPivot && globalRow < globalPivot + tileRows)) {
        return;
    }

    const size_t outputIndex = localRow * pitch + globalPivot + tileColumn;
    unsigned int best = columns[threadIdx.y][tileColumn];
    unsigned int predecessor = path[outputIndex];

#pragma unroll
    for (int k = 0; k < TILE_SIZE; ++k) {
        if (k < tileRows) {
            const unsigned int candidate =
                columns[threadIdx.y][k] + pivotTile[k][tileColumn];
            if (candidate < best) {
                best = candidate;
                predecessor = static_cast<unsigned int>(globalPivot + k);
            }
        }
    }

    dist[outputIndex] = best;
    path[outputIndex] = predecessor;
}

// The bulk O(n^3 / ranks) phase. Adjacent threads visit adjacent destinations,
// giving coalesced accesses to both the local matrix and the broadcast pivot rows.
__global__ void updateRemainingMatrix(unsigned int* __restrict__ dist,
                                      unsigned int* __restrict__ path,
                                      const unsigned int* __restrict__ pivotRows,
                                      const size_t localRows, const size_t pitch,
                                      const size_t firstGlobalRow,
                                      const size_t globalPivot,
                                      const int tileRows) {
    __shared__ unsigned int pivotTile[TILE_SIZE][TILE_SIZE + 1];

    const int linearThread = threadIdx.y * TILE_SIZE + threadIdx.x;
    const size_t firstColumn = static_cast<size_t>(blockIdx.x) * TILE_SIZE;
    for (int element = linearThread; element < TILE_SIZE * TILE_SIZE;
         element += TILE_SIZE * ROWS_PER_BLOCK) {
        const int pivotRow = element / TILE_SIZE;
        const int tileColumn = element % TILE_SIZE;
        const size_t globalColumn = firstColumn + static_cast<size_t>(tileColumn);
        pivotTile[pivotRow][tileColumn] =
            (pivotRow < tileRows && globalColumn < pitch)
                ? pivotRows[static_cast<size_t>(pivotRow) * pitch + globalColumn]
                : INF;
    }
    __syncthreads();

    const size_t column = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t localRow =
        static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if (localRow >= localRows || column >= pitch ||
        (column >= globalPivot && column < globalPivot + tileRows)) {
        return;
    }

    const size_t globalRow = firstGlobalRow + localRow;
    if (globalRow >= globalPivot && globalRow < globalPivot + tileRows) {
        return;
    }

    const size_t outputIndex = localRow * pitch + column;
    unsigned int best = dist[outputIndex];
    unsigned int predecessor = path[outputIndex];

#pragma unroll
    for (int k = 0; k < TILE_SIZE; ++k) {
        if (k < tileRows) {
            const unsigned int candidate =
                dist[localRow * pitch + globalPivot + k] +
                pivotTile[k][threadIdx.x];
            if (candidate < best) {
                best = candidate;
                predecessor = static_cast<unsigned int>(globalPivot + k);
            }
        }
    }

    dist[outputIndex] = best;
    path[outputIndex] = predecessor;
}

bool validateResult(const std::vector<unsigned int>& dist, const size_t numNodes) {
    for (size_t i = 0; i < numNodes; ++i) {
        if (dist[idx2(i, i, numNodes)] != 0) {
            std::printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", i, i);
            return false;
        }
    }

    // Match the original sampled O(n^2) validation.
    for (size_t i = 0; i < std::min(numNodes, static_cast<size_t>(10)); ++i) {
        for (size_t j = 0; j < std::min(numNodes, static_cast<size_t>(10)); ++j) {
            for (size_t k = 0; k < numNodes; ++k) {
                const unsigned int distIJ = dist[idx2(j, i, numNodes)];
                const unsigned int distIK = dist[idx2(k, i, numNodes)];
                const unsigned int distKJ = dist[idx2(j, k, numNodes)];
                if (distIK < INF && distKJ < INF && distIK + distKJ < distIJ) {
                    std::printf(
                        "Validation failed: triangle inequality violated at [%zu,%zu,%zu]\n",
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

int ownerOfRow(const size_t row, const std::vector<size_t>& rowOffsets,
               const std::vector<size_t>& rowCounts) {
    const auto upper = std::upper_bound(rowOffsets.begin(), rowOffsets.end(), row);
    int owner = static_cast<int>(upper - rowOffsets.begin()) - 1;
    owner = std::max(owner, 0);
    while (rowCounts[static_cast<size_t>(owner)] == 0) {
        --owner;
    }
    return owner;
}

void runDistributedFloydWarshall(unsigned int* deviceDist,
                                 unsigned int* devicePath,
                                 unsigned int* devicePivotRows,
                                 unsigned int* hostPivotRows,
                                 const size_t numNodes, const size_t localRows,
                                 const size_t firstGlobalRow,
                                 const std::vector<size_t>& rowOffsets,
                                 const std::vector<size_t>& rowCounts,
                                 const bool cudaAwareMpi, const int rank,
                                 cudaStream_t stream) {
    const dim3 matrixBlock(TILE_SIZE, ROWS_PER_BLOCK);
    const size_t pivotCapacity =
        static_cast<size_t>(TILE_SIZE) * std::max(numNodes, static_cast<size_t>(1));
    cudaEvent_t bufferReady[2]{};
    bool bufferInUse[2]{};
    for (cudaEvent_t& event : bufferReady) {
        CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    }

    size_t round = 0;
    for (size_t globalPivot = 0; globalPivot < numNodes; ++round) {
        const int slot = static_cast<int>(round & 1U);
        unsigned int* const devicePivot = devicePivotRows + slot * pivotCapacity;
        unsigned int* const hostPivot = hostPivotRows + slot * pivotCapacity;

        // A two-slot pipeline lets ranks enter the next broadcast while their GPU
        // works on a prior round. Wait only when communication storage is reused.
        if (bufferInUse[slot]) {
            CUDA_CHECK(cudaEventSynchronize(bufferReady[slot]));
            bufferInUse[slot] = false;
        }

        const int owner = ownerOfRow(globalPivot, rowOffsets, rowCounts);
        const size_t ownerEnd = rowOffsets[static_cast<size_t>(owner)] +
                                rowCounts[static_cast<size_t>(owner)];
        const int tileRows = static_cast<int>(
            std::min(static_cast<size_t>(TILE_SIZE), ownerEnd - globalPivot));

        if (rank == owner) {
            const size_t localPivotRow = globalPivot - firstGlobalRow;
            const size_t tileElements = static_cast<size_t>(tileRows) * numNodes;

            CUDA_CHECK(cudaMemcpyAsync(devicePivot,
                                       deviceDist + localPivotRow * numNodes,
                                       tileElements * sizeof(unsigned int),
                                       cudaMemcpyDeviceToDevice, stream));

            closePivotTile<<<1, dim3(TILE_SIZE, TILE_SIZE), 0, stream>>>(
                deviceDist, devicePath, numNodes, localPivotRow, globalPivot, tileRows);
            CUDA_CHECK(cudaGetLastError());

            const dim3 pivotGrid(
                static_cast<unsigned int>((numNodes + matrixBlock.x - 1) / matrixBlock.x),
                static_cast<unsigned int>((tileRows + matrixBlock.y - 1) / matrixBlock.y));
            updatePivotRows<<<pivotGrid, matrixBlock, 0, stream>>>(
                deviceDist, devicePath, devicePivot, numNodes, localPivotRow,
                globalPivot, tileRows);
            CUDA_CHECK(cudaGetLastError());

            CUDA_CHECK(cudaMemcpyAsync(cudaAwareMpi ? devicePivot : hostPivot,
                                       deviceDist + localPivotRow * numNodes,
                                       tileElements * sizeof(unsigned int),
                                       cudaAwareMpi ? cudaMemcpyDeviceToDevice
                                                    : cudaMemcpyDeviceToHost,
                                       stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        const size_t broadcastElements = static_cast<size_t>(tileRows) * numNodes;
        if (broadcastElements > static_cast<size_t>(INT_MAX)) {
            abortWithMessage(rank, "pivot-row broadcast exceeds the MPI count limit");
        }
        MPI_CHECK(MPI_Bcast(cudaAwareMpi ? static_cast<void*>(devicePivot)
                                        : static_cast<void*>(hostPivot),
                           static_cast<int>(broadcastElements),
                           MPI_UNSIGNED, owner, MPI_COMM_WORLD));

        const unsigned int* pivotRows = devicePivot;
        if (!cudaAwareMpi && rank == owner) {
            pivotRows = deviceDist + (globalPivot - firstGlobalRow) * numNodes;
        } else if (!cudaAwareMpi) {
            CUDA_CHECK(cudaMemcpyAsync(devicePivot, hostPivot,
                                       broadcastElements * sizeof(unsigned int),
                                       cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaEventRecord(bufferReady[slot], stream));
            bufferInUse[slot] = true;
        }

        if (localRows != 0) {
            const dim3 columnGrid(
                static_cast<unsigned int>((localRows + ROWS_PER_BLOCK - 1) /
                                          ROWS_PER_BLOCK));
            updatePivotColumns<<<columnGrid, matrixBlock, 0, stream>>>(
                deviceDist, devicePath, pivotRows, localRows, numNodes,
                firstGlobalRow, globalPivot, tileRows);
            CUDA_CHECK(cudaGetLastError());

            const dim3 remainingGrid(
                static_cast<unsigned int>((numNodes + matrixBlock.x - 1) /
                                          matrixBlock.x),
                static_cast<unsigned int>((localRows + matrixBlock.y - 1) /
                                          matrixBlock.y));
            updateRemainingMatrix<<<remainingGrid, matrixBlock, 0, stream>>>(
                deviceDist, devicePath, pivotRows, localRows, numNodes,
                firstGlobalRow, globalPivot, tileRows);
            CUDA_CHECK(cudaGetLastError());
        }

        if (cudaAwareMpi) {
            CUDA_CHECK(cudaEventRecord(bufferReady[slot], stream));
            bufferInUse[slot] = true;
        }

        globalPivot += static_cast<size_t>(tileRows);
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (cudaEvent_t event : bufferReady) {
        CUDA_CHECK(cudaEventDestroy(event));
    }
}

int main(int argc, char** argv) {
    int providedThreadLevel = MPI_THREAD_SINGLE;
    int mpiInitError = MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED,
                                       &providedThreadLevel);
    if (mpiInitError != MPI_SUCCESS) {
        std::fprintf(stderr, "MPI_Init_thread failed\n");
        return EXIT_FAILURE;
    }

    int rank = 0;
    int worldSize = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &worldSize));
    if (providedThreadLevel < MPI_THREAD_FUNNELED) {
        abortWithMessage(rank, "MPI does not provide the required FUNNELED thread level");
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
            if (errno != 0 || end == argv[i] || *end != '\0' ||
                parsed > std::numeric_limits<size_t>::max()) {
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
            if (rank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
            }
            argumentsValid = false;
        }
    }

    if (!argumentsValid || showHelp) {
        if (rank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return argumentsValid ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    if (numNodes != 0 && numNodes > std::numeric_limits<size_t>::max() / numNodes) {
        abortWithMessage(rank, "matrix dimensions overflow addressable memory");
    }
    if (numNodes > std::numeric_limits<unsigned int>::max()) {
        abortWithMessage(rank, "path indices require at most UINT_MAX vertices");
    }

    MPI_Comm localComm = MPI_COMM_NULL;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                                  MPI_INFO_NULL, &localComm));
    int localRank = 0;
    int localSize = 1;
    MPI_CHECK(MPI_Comm_rank(localComm, &localRank));
    MPI_CHECK(MPI_Comm_size(localComm, &localSize));

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        abortWithMessage(rank, "the mandatory CUDA path requires at least one GPU");
    }
    const int device = localRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));
    CUDA_CHECK(cudaFree(nullptr));  // Create the context before benchmark timing.

    bool cudaAwareMpi = false;
#if defined(MPIX_CUDA_AWARE_SUPPORT)
    cudaAwareMpi = MPIX_Query_cuda_support() != 0;
#endif
    int localCudaAware = cudaAwareMpi ? 1 : 0;
    int allCudaAware = 0;
    MPI_CHECK(MPI_Allreduce(&localCudaAware, &allCudaAware, 1, MPI_INT, MPI_MIN,
                            MPI_COMM_WORLD));
    cudaAwareMpi = allCudaAware != 0;

    std::vector<size_t> rowCounts(static_cast<size_t>(worldSize));
    std::vector<size_t> rowOffsets(static_cast<size_t>(worldSize));
    std::vector<int> mpiCounts(static_cast<size_t>(worldSize));
    std::vector<int> mpiOffsets(static_cast<size_t>(worldSize));
    const size_t quotient = numNodes / static_cast<size_t>(worldSize);
    const size_t remainder = numNodes % static_cast<size_t>(worldSize);
    size_t offset = 0;
    for (int process = 0; process < worldSize; ++process) {
        const size_t rows = quotient + (static_cast<size_t>(process) < remainder ? 1 : 0);
        rowCounts[static_cast<size_t>(process)] = rows;
        rowOffsets[static_cast<size_t>(process)] = offset;
        const size_t elements = rows * numNodes;
        const size_t displacement = offset * numNodes;
        if (elements > static_cast<size_t>(INT_MAX) ||
            displacement > static_cast<size_t>(INT_MAX)) {
            abortWithMessage(rank, "matrix distribution exceeds MPI_Scatterv limits");
        }
        mpiCounts[static_cast<size_t>(process)] = static_cast<int>(elements);
        mpiOffsets[static_cast<size_t>(process)] = static_cast<int>(displacement);
        offset += rows;
    }

    const size_t localRows = rowCounts[static_cast<size_t>(rank)];
    const size_t firstGlobalRow = rowOffsets[static_cast<size_t>(rank)];
    const size_t localElements = localRows * numNodes;

    if (rank == 0) {
        std::printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        std::printf("Number of nodes: %zu\n", numNodes);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Hybrid configuration: %d MPI rank(s), %d OpenMP thread(s)/rank, "
                    "%d GPU(s) on rank 0's node\n",
                    worldSize, omp_get_max_threads(), deviceCount);
        std::printf("MPI pivot transport: %s\n",
                    cudaAwareMpi ? "direct GPU buffers" : "pinned host staging");
        if (localSize > deviceCount) {
            std::printf("Warning: %d local MPI ranks share %d GPU(s); one rank per GPU "
                        "is recommended.\n",
                        localSize, deviceCount);
        }
        std::printf("Initializing graph...\n");
    }

    std::vector<unsigned int> globalDist;
    if (rank == 0) {
        globalDist.resize(numNodes * numNodes);
        initializeDistanceMatrix(globalDist, numNodes, 1, MAX_DISTANCE);
    }

    std::vector<unsigned int> localDist(localElements);
    std::vector<unsigned int> localPath(localElements);
    MPI_CHECK(MPI_Scatterv(rank == 0 ? globalDist.data() : nullptr, mpiCounts.data(),
                           mpiOffsets.data(), MPI_UNSIGNED, localDist.data(),
                           mpiCounts[static_cast<size_t>(rank)], MPI_UNSIGNED, 0,
                           MPI_COMM_WORLD));
    globalDist.clear();
    globalDist.shrink_to_fit();
    if (localElements != 0) {
        initializeLocalPathMatrix(localPath, firstGlobalRow, numNodes);
    }

    unsigned int* deviceDist = nullptr;
    unsigned int* devicePath = nullptr;
    unsigned int* devicePivotRows = nullptr;
    unsigned int* hostPivotRows = nullptr;
    const size_t allocatedLocalElements = std::max(localElements, static_cast<size_t>(1));
    const size_t pivotElements =
        static_cast<size_t>(TILE_SIZE) * std::max(numNodes, static_cast<size_t>(1));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceDist),
                          allocatedLocalElements * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&devicePath),
                          allocatedLocalElements * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&devicePivotRows),
                          2 * pivotElements * sizeof(unsigned int)));
    CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&hostPivotRows),
                             2 * pivotElements * sizeof(unsigned int), cudaHostAllocDefault));

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    if (localElements != 0) {
        CUDA_CHECK(cudaMemcpyAsync(deviceDist, localDist.data(),
                                   localElements * sizeof(unsigned int),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(devicePath, localPath.data(),
                                   localElements * sizeof(unsigned int),
                                   cudaMemcpyHostToDevice, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    localPath.clear();
    localPath.shrink_to_fit();
    localDist.clear();
    localDist.shrink_to_fit();

    if (rank == 0) {
        std::printf("Computing shortest paths...\n");
        std::fflush(stdout);
    }
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    const double start = MPI_Wtime();

    runDistributedFloydWarshall(deviceDist, devicePath, devicePivotRows,
                                hostPivotRows, numNodes, localRows, firstGlobalRow,
                                rowOffsets, rowCounts, cudaAwareMpi, rank, stream);

    const double localSeconds = MPI_Wtime() - start;
    double seconds = 0.0;
    MPI_CHECK(MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0,
                         MPI_COMM_WORLD));

    if ((printResults || validate) && localElements != 0) {
        localDist.resize(localElements);
        CUDA_CHECK(cudaMemcpy(localDist.data(), deviceDist,
                              localElements * sizeof(unsigned int),
                              cudaMemcpyDeviceToHost));
    }

    if (rank == 0) {
        const long long durationMs = static_cast<long long>(seconds * 1000.0);
        std::printf("Computation time: %lld ms\n", durationMs);
        const double operations = static_cast<double>(numNodes) * numNodes * numNodes;
        const double gops = seconds > 0.0 ? operations / seconds / 1.0e9 : 0.0;
        std::printf("Performance: %.3f GOPS\n", gops);
    }

    if (printResults || validate) {
        if (rank == 0) {
            globalDist.resize(numNodes * numNodes);
        }
        MPI_CHECK(MPI_Gatherv(localDist.data(), mpiCounts[static_cast<size_t>(rank)],
                              MPI_UNSIGNED, rank == 0 ? globalDist.data() : nullptr,
                              mpiCounts.data(), mpiOffsets.data(), MPI_UNSIGNED, 0,
                              MPI_COMM_WORLD));
    }

    int exitCode = EXIT_SUCCESS;
    if (rank == 0 && printResults) {
        print_results_int(globalDist, "DistanceMatrix");
    }
    if (rank == 0 && validate) {
        std::printf("Validating result...\n");
        if (validateResult(globalDist, numNodes)) {
            std::printf("Validation: PASSED\n");
        } else {
            std::printf("Validation: FAILED\n");
            exitCode = EXIT_FAILURE;
        }
    }
    MPI_CHECK(MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD));

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFreeHost(hostPivotRows));
    CUDA_CHECK(cudaFree(devicePivotRows));
    CUDA_CHECK(cudaFree(devicePath));
    CUDA_CHECK(cudaFree(deviceDist));
    MPI_CHECK(MPI_Comm_free(&localComm));
    MPI_CHECK(MPI_Finalize());
    return exitCode;
}
