#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;
constexpr int TILE_SIZE = 32;
constexpr int BLOCK_ROWS = 8;
constexpr int ROWS_PER_THREAD = TILE_SIZE / BLOCK_ROWS;

static_assert(TILE_SIZE % BLOCK_ROWS == 0, "CUDA tile must divide evenly");

struct Decomposition {
    size_t tileBegin = 0;
    size_t tileCount = 0;
    size_t rowBegin = 0;
    size_t rowCount = 0;
};

inline constexpr size_t idx2(const size_t i, const size_t j,
                             const size_t n) noexcept {
    return j * n + i;
}

[[noreturn]] void mpiFailure(const int error, const char* expression,
                             const char* file, const int line) {
    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    char message[MPI_MAX_ERROR_STRING] = {};
    int length = 0;
    MPI_Error_string(error, message, &length);
    std::fprintf(stderr, "MPI error on rank %d at %s:%d (%s): %.*s\n",
                 rank, file, line, expression, length, message);
    MPI_Abort(MPI_COMM_WORLD, error);
    std::abort();
}

#define MPI_CHECK(call)                                                        \
    do {                                                                       \
        const int mpi_error_ = (call);                                         \
        if (mpi_error_ != MPI_SUCCESS) {                                       \
            mpiFailure(mpi_error_, #call, __FILE__, __LINE__);                 \
        }                                                                      \
    } while (false)

[[noreturn]] void cudaFailure(const cudaError_t error, const char* expression,
                              const char* file, const int line, const int rank) {
    std::fprintf(stderr, "CUDA error on rank %d at %s:%d (%s): %s\n", rank,
                 file, line, expression, cudaGetErrorString(error));
    MPI_Abort(MPI_COMM_WORLD, static_cast<int>(error));
    std::abort();
}

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        const cudaError_t cuda_error_ = (call);                                \
        if (cuda_error_ != cudaSuccess) {                                      \
            cudaFailure(cuda_error_, #call, __FILE__, __LINE__, rank);         \
        }                                                                      \
    } while (false)

Decomposition decomposeRows(const size_t numNodes, const int rank,
                            const int worldSize) {
    const size_t totalTiles =
        (numNodes + static_cast<size_t>(TILE_SIZE) - 1) / TILE_SIZE;
    const size_t ranks = static_cast<size_t>(worldSize);
    const size_t base = totalTiles / ranks;
    const size_t extra = totalTiles % ranks;
    const size_t rankIndex = static_cast<size_t>(rank);

    Decomposition result;
    result.tileCount = base + (rankIndex < extra ? 1 : 0);
    result.tileBegin = rankIndex * base + std::min(rankIndex, extra);
    result.rowBegin =
        std::min(numNodes, result.tileBegin * static_cast<size_t>(TILE_SIZE));
    const size_t rowEnd =
        std::min(numNodes, (result.tileBegin + result.tileCount) * TILE_SIZE);
    result.rowCount = rowEnd - result.rowBegin;
    return result;
}

int ownerOfTile(const size_t tile, const size_t totalTiles,
                const int worldSize) {
    const size_t ranks = static_cast<size_t>(worldSize);
    const size_t base = totalTiles / ranks;
    const size_t extra = totalTiles % ranks;
    const size_t largeRegion = extra * (base + 1);
    if (tile < largeRegion) {
        return static_cast<int>(tile / (base + 1));
    }
    return static_cast<int>(extra + (tile - largeRegion) / base);
}

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

void initializeLocalPathMatrix(std::vector<unsigned int>& path,
                               const size_t numNodes,
                               const Decomposition& decomposition) {
#pragma omp parallel for schedule(static)
    for (long long localRow = 0;
         localRow < static_cast<long long>(decomposition.rowCount);
         ++localRow) {
        const unsigned int source = static_cast<unsigned int>(
            decomposition.rowBegin + static_cast<size_t>(localRow));
        unsigned int* const row =
            path.data() + static_cast<size_t>(localRow) * numNodes;
        for (size_t destination = 0; destination < numNodes; ++destination) {
            row[destination] = source;
        }
    }
}

__global__ void phaseOneKernel(unsigned int* const dist,
                               unsigned int* const path, const size_t n,
                               const size_t localPivotRow,
                               const size_t globalPivot,
                               const int validPivotRows) {
    __shared__ unsigned int tile[TILE_SIZE][TILE_SIZE + 1];

    const int column = threadIdx.x;
#pragma unroll
    for (int item = 0; item < ROWS_PER_THREAD; ++item) {
        const int row = threadIdx.y + item * BLOCK_ROWS;
        if (row < validPivotRows && column < validPivotRows) {
            tile[row][column] =
                dist[(localPivotRow + static_cast<size_t>(row)) * n +
                     globalPivot + static_cast<size_t>(column)];
        } else {
            tile[row][column] = INF;
        }
    }
    __syncthreads();

    for (int k = 0; k < validPivotRows; ++k) {
#pragma unroll
        for (int item = 0; item < ROWS_PER_THREAD; ++item) {
            const int row = threadIdx.y + item * BLOCK_ROWS;
            const unsigned int candidate =
                tile[row][k] + tile[k][column];
            if (row < validPivotRows && column < validPivotRows &&
                candidate < tile[row][column]) {
                tile[row][column] = candidate;
                path[(localPivotRow + static_cast<size_t>(row)) * n +
                     globalPivot + static_cast<size_t>(column)] =
                    static_cast<unsigned int>(globalPivot + k);
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int item = 0; item < ROWS_PER_THREAD; ++item) {
        const int row = threadIdx.y + item * BLOCK_ROWS;
        if (row < validPivotRows && column < validPivotRows) {
            dist[(localPivotRow + static_cast<size_t>(row)) * n +
                 globalPivot + static_cast<size_t>(column)] = tile[row][column];
        }
    }
}

__global__ void phaseTwoRowKernel(unsigned int* const dist,
                                  unsigned int* const path, const size_t n,
                                  const size_t localPivotRow,
                                  const size_t globalPivot,
                                  const int validPivotRows) {
    const size_t destinationBegin =
        static_cast<size_t>(blockIdx.x) * TILE_SIZE;
    if (destinationBegin == globalPivot) {
        return;
    }

    __shared__ unsigned int pivot[TILE_SIZE][TILE_SIZE + 1];
    __shared__ unsigned int target[TILE_SIZE][TILE_SIZE + 1];
    const int column = threadIdx.x;
    const bool validColumn = destinationBegin + column < n;

#pragma unroll
    for (int item = 0; item < ROWS_PER_THREAD; ++item) {
        const int row = threadIdx.y + item * BLOCK_ROWS;
        if (row < validPivotRows) {
            pivot[row][column] =
                column < validPivotRows
                    ? dist[(localPivotRow + static_cast<size_t>(row)) * n +
                           globalPivot + static_cast<size_t>(column)]
                    : INF;
            target[row][column] =
                validColumn
                    ? dist[(localPivotRow + static_cast<size_t>(row)) * n +
                           destinationBegin + static_cast<size_t>(column)]
                    : INF;
        } else {
            pivot[row][column] = INF;
            target[row][column] = INF;
        }
    }
    __syncthreads();

    for (int k = 0; k < validPivotRows; ++k) {
#pragma unroll
        for (int item = 0; item < ROWS_PER_THREAD; ++item) {
            const int row = threadIdx.y + item * BLOCK_ROWS;
            const unsigned int candidate =
                pivot[row][k] + target[k][column];
            if (row < validPivotRows && validColumn &&
                candidate < target[row][column]) {
                target[row][column] = candidate;
                path[(localPivotRow + static_cast<size_t>(row)) * n +
                     destinationBegin + static_cast<size_t>(column)] =
                    static_cast<unsigned int>(globalPivot + k);
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int item = 0; item < ROWS_PER_THREAD; ++item) {
        const int row = threadIdx.y + item * BLOCK_ROWS;
        if (row < validPivotRows && validColumn) {
            dist[(localPivotRow + static_cast<size_t>(row)) * n +
                 destinationBegin + static_cast<size_t>(column)] =
                target[row][column];
        }
    }
}

__global__ void phaseTwoColumnKernel(
    unsigned int* const dist, unsigned int* const path,
    const unsigned int* const pivotRows, const size_t n,
    const size_t localRows, const size_t globalRowBegin,
    const size_t globalPivot, const int validPivotRows) {
    const size_t localRowBegin = static_cast<size_t>(blockIdx.x) * TILE_SIZE;
    if (globalRowBegin + localRowBegin == globalPivot) {
        return;
    }

    __shared__ unsigned int target[TILE_SIZE][TILE_SIZE + 1];
    __shared__ unsigned int pivot[TILE_SIZE][TILE_SIZE + 1];
    const int column = threadIdx.x;

#pragma unroll
    for (int item = 0; item < ROWS_PER_THREAD; ++item) {
        const int row = threadIdx.y + item * BLOCK_ROWS;
        const size_t localRow = localRowBegin + static_cast<size_t>(row);
        target[row][column] =
            localRow < localRows && column < validPivotRows
                ? dist[localRow * n + globalPivot +
                       static_cast<size_t>(column)]
                : INF;
        pivot[row][column] =
            row < validPivotRows && column < validPivotRows
                ? pivotRows[static_cast<size_t>(row) * n + globalPivot +
                            static_cast<size_t>(column)]
                : INF;
    }
    __syncthreads();

    for (int k = 0; k < validPivotRows; ++k) {
#pragma unroll
        for (int item = 0; item < ROWS_PER_THREAD; ++item) {
            const int row = threadIdx.y + item * BLOCK_ROWS;
            const size_t localRow = localRowBegin + static_cast<size_t>(row);
            const unsigned int candidate =
                target[row][k] + pivot[k][column];
            if (localRow < localRows && column < validPivotRows &&
                candidate < target[row][column]) {
                target[row][column] = candidate;
                path[localRow * n + globalPivot +
                     static_cast<size_t>(column)] =
                    static_cast<unsigned int>(globalPivot + k);
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int item = 0; item < ROWS_PER_THREAD; ++item) {
        const int row = threadIdx.y + item * BLOCK_ROWS;
        const size_t localRow = localRowBegin + static_cast<size_t>(row);
        if (localRow < localRows && column < validPivotRows) {
            dist[localRow * n + globalPivot +
                 static_cast<size_t>(column)] = target[row][column];
        }
    }
}

__global__ void phaseThreeKernel(
    unsigned int* const dist, unsigned int* const path,
    const unsigned int* const pivotRows, const size_t n,
    const size_t localRows, const size_t globalRowBegin,
    const size_t globalPivot, const int validPivotRows) {
    const size_t destinationBegin =
        static_cast<size_t>(blockIdx.x) * TILE_SIZE;
    const size_t localRowBegin =
        static_cast<size_t>(blockIdx.y) * TILE_SIZE;
    if (destinationBegin == globalPivot ||
        globalRowBegin + localRowBegin == globalPivot) {
        return;
    }

    __shared__ unsigned int left[TILE_SIZE][TILE_SIZE + 1];
    __shared__ unsigned int right[TILE_SIZE][TILE_SIZE + 1];
    const int column = threadIdx.x;
    const bool validColumn = destinationBegin + column < n;

#pragma unroll
    for (int item = 0; item < ROWS_PER_THREAD; ++item) {
        const int row = threadIdx.y + item * BLOCK_ROWS;
        const size_t localRow = localRowBegin + static_cast<size_t>(row);
        left[row][column] =
            localRow < localRows && column < validPivotRows
                ? dist[localRow * n + globalPivot +
                       static_cast<size_t>(column)]
                : INF;
        right[row][column] =
            row < validPivotRows && validColumn
                ? pivotRows[static_cast<size_t>(row) * n + destinationBegin +
                            static_cast<size_t>(column)]
                : INF;
    }
    __syncthreads();

#pragma unroll
    for (int item = 0; item < ROWS_PER_THREAD; ++item) {
        const int row = threadIdx.y + item * BLOCK_ROWS;
        const size_t localRow = localRowBegin + static_cast<size_t>(row);
        if (localRow < localRows && validColumn) {
            unsigned int best =
                dist[localRow * n + destinationBegin +
                     static_cast<size_t>(column)];
            int bestIntermediate = -1;
#pragma unroll
            for (int k = 0; k < TILE_SIZE; ++k) {
                if (k < validPivotRows) {
                    const unsigned int candidate =
                        left[row][k] + right[k][column];
                    if (candidate < best) {
                        best = candidate;
                        bestIntermediate = k;
                    }
                }
            }
            dist[localRow * n + destinationBegin +
                 static_cast<size_t>(column)] = best;
            if (bestIntermediate >= 0) {
                path[localRow * n + destinationBegin +
                     static_cast<size_t>(column)] =
                    static_cast<unsigned int>(globalPivot +
                                              bestIntermediate);
            }
        }
    }
}

double floydWarshallHybrid(unsigned int* const deviceDist,
                           unsigned int* const devicePath,
                           const size_t numNodes,
                           const Decomposition& decomposition, const int rank,
                           const int worldSize) {
    const size_t totalTiles =
        (numNodes + static_cast<size_t>(TILE_SIZE) - 1) / TILE_SIZE;
    const size_t pivotElements = static_cast<size_t>(TILE_SIZE) * numNodes;
    const size_t pivotBytes = pivotElements * sizeof(unsigned int);

    unsigned int* hostPivot[2] = {nullptr, nullptr};
    unsigned int* devicePivot[2] = {nullptr, nullptr};
    cudaEvent_t hostCopyReady[2] = {};
    bool slotUsed[2] = {false, false};
    cudaStream_t stream = nullptr;

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    for (int slot = 0; slot < 2; ++slot) {
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&hostPivot[slot]),
                                 pivotBytes, cudaHostAllocPortable));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&devicePivot[slot]),
                              pivotBytes));
        CUDA_CHECK(cudaEventCreateWithFlags(&hostCopyReady[slot],
                                            cudaEventDisableTiming));
    }

    const dim3 threads(TILE_SIZE, BLOCK_ROWS);
    const size_t localTiles = decomposition.tileCount;

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    const double start = MPI_Wtime();

    for (size_t pivotTile = 0; pivotTile < totalTiles; ++pivotTile) {
        const int slot = static_cast<int>(pivotTile & 1U);
        const size_t globalPivot = pivotTile * TILE_SIZE;
        const int validPivotRows = static_cast<int>(
            std::min(static_cast<size_t>(TILE_SIZE), numNodes - globalPivot));
        const int owner = ownerOfTile(pivotTile, totalTiles, worldSize);
        const size_t activePivotElements =
            static_cast<size_t>(validPivotRows) * numNodes;
        const size_t activePivotBytes =
            activePivotElements * sizeof(unsigned int);

        if (rank == owner) {
            const size_t localPivotRow = globalPivot - decomposition.rowBegin;
            phaseOneKernel<<<1, threads, 0, stream>>>(
                deviceDist, devicePath, numNodes, localPivotRow, globalPivot,
                validPivotRows);
            CUDA_CHECK(cudaGetLastError());

            phaseTwoRowKernel<<<static_cast<unsigned int>(totalTiles), threads,
                                0, stream>>>(
                deviceDist, devicePath, numNodes, localPivotRow, globalPivot,
                validPivotRows);
            CUDA_CHECK(cudaGetLastError());

            CUDA_CHECK(cudaMemcpyAsync(
                hostPivot[slot], deviceDist + localPivotRow * numNodes,
                activePivotBytes, cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaEventRecord(hostCopyReady[slot], stream));
            CUDA_CHECK(cudaEventSynchronize(hostCopyReady[slot]));
        } else if (slotUsed[slot]) {
            // MPI may overwrite this pinned buffer only after its previous H2D
            // transfer has completed.  Later GPU work remains overlapped.
            CUDA_CHECK(cudaEventSynchronize(hostCopyReady[slot]));
        }

        MPI_CHECK(MPI_Bcast(hostPivot[slot],
                            static_cast<int>(activePivotElements), MPI_UNSIGNED,
                            owner, MPI_COMM_WORLD));

        CUDA_CHECK(cudaMemcpyAsync(devicePivot[slot], hostPivot[slot],
                                   activePivotBytes, cudaMemcpyHostToDevice,
                                   stream));
        CUDA_CHECK(cudaEventRecord(hostCopyReady[slot], stream));
        slotUsed[slot] = true;

        if (localTiles != 0) {
            phaseTwoColumnKernel<<<static_cast<unsigned int>(localTiles),
                                   threads, 0, stream>>>(
                deviceDist, devicePath, devicePivot[slot], numNodes,
                decomposition.rowCount, decomposition.rowBegin, globalPivot,
                validPivotRows);
            CUDA_CHECK(cudaGetLastError());

            const dim3 blocks(static_cast<unsigned int>(totalTiles),
                              static_cast<unsigned int>(localTiles));
            phaseThreeKernel<<<blocks, threads, 0, stream>>>(
                deviceDist, devicePath, devicePivot[slot], numNodes,
                decomposition.rowCount, decomposition.rowBegin, globalPivot,
                validPivotRows);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    const double localElapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_CHECK(MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0,
                         MPI_COMM_WORLD));

    for (int slot = 0; slot < 2; ++slot) {
        CUDA_CHECK(cudaEventDestroy(hostCopyReady[slot]));
        CUDA_CHECK(cudaFree(devicePivot[slot]));
        CUDA_CHECK(cudaFreeHost(hostPivot[slot]));
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
    return elapsed;
}

bool validateResult(const std::vector<unsigned int>& dist,
                    const size_t numNodes) {
    int diagonalValid = 1;
#pragma omp parallel for schedule(static) reduction(&: diagonalValid)
    for (long long i = 0; i < static_cast<long long>(numNodes); ++i) {
        diagonalValid &=
            dist[idx2(static_cast<size_t>(i), static_cast<size_t>(i),
                      numNodes)] == 0;
    }
    if (!diagonalValid) {
        std::printf("Validation failed: a diagonal element is not zero\n");
        return false;
    }

    const size_t sampleSize = std::min(numNodes, static_cast<size_t>(10));
    int triangleValid = 1;
#pragma omp parallel for collapse(2) schedule(static) reduction(&: triangleValid)
    for (long long i = 0; i < static_cast<long long>(sampleSize); ++i) {
        for (long long j = 0; j < static_cast<long long>(sampleSize); ++j) {
            for (size_t k = 0; k < numNodes; ++k) {
                const unsigned int distIJ =
                    dist[idx2(static_cast<size_t>(j), static_cast<size_t>(i),
                              numNodes)];
                const unsigned int distIK =
                    dist[idx2(k, static_cast<size_t>(i), numNodes)];
                const unsigned int distKJ =
                    dist[idx2(static_cast<size_t>(j), k, numNodes)];
                if (distIK < INF && distKJ < INF &&
                    distIK + distKJ < distIJ) {
                    triangleValid = 0;
                }
            }
        }
    }
    if (!triangleValid) {
        std::printf("Validation failed: triangle inequality violated\n");
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
    int threadSupport = MPI_THREAD_SINGLE;
    MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED,
                              &threadSupport));

    int rank = 0;
    int worldSize = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &worldSize));
    if (threadSupport < MPI_THREAD_FUNNELED) {
        if (rank == 0) {
            std::fprintf(stderr,
                         "MPI implementation does not provide FUNNELED thread "
                         "support\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;
    bool argumentsValid = true;
    bool showHelp = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const unsigned long long value = std::strtoull(argv[++i], &end, 10);
            if (end == argv[i] || *end != '\0' || value == 0 ||
                value > std::numeric_limits<size_t>::max()) {
                argumentsValid = false;
            } else {
                numNodes = static_cast<size_t>(value);
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
        MPI_CHECK(MPI_Finalize());
        return argumentsValid ? 0 : 1;
    }

    if (numNodes > std::numeric_limits<size_t>::max() / numNodes ||
        numNodes > std::numeric_limits<unsigned int>::max()) {
        if (rank == 0) {
            std::fprintf(stderr, "Requested graph is too large\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    const Decomposition decomposition =
        decomposeRows(numNodes, rank, worldSize);
    const size_t totalElements = numNodes * numNodes;
    const size_t localElements = decomposition.rowCount * numNodes;
    const size_t pivotElements = static_cast<size_t>(TILE_SIZE) * numNodes;
    if (totalElements > static_cast<size_t>(INT_MAX) * worldSize ||
        localElements > static_cast<size_t>(INT_MAX) ||
        pivotElements > static_cast<size_t>(INT_MAX)) {
        if (rank == 0) {
            std::fprintf(stderr,
                         "Graph exceeds the count range of MPI collectives\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    std::vector<int> counts(static_cast<size_t>(worldSize));
    std::vector<int> displacements(static_cast<size_t>(worldSize));
    for (int process = 0; process < worldSize; ++process) {
        const Decomposition part =
            decomposeRows(numNodes, process, worldSize);
        const size_t count = part.rowCount * numNodes;
        const size_t displacement = part.rowBegin * numNodes;
        if (count > static_cast<size_t>(INT_MAX) ||
            displacement > static_cast<size_t>(INT_MAX)) {
            if (rank == 0) {
                std::fprintf(stderr,
                             "Graph exceeds the count range of MPI collectives\n");
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        counts[static_cast<size_t>(process)] = static_cast<int>(count);
        displacements[static_cast<size_t>(process)] =
            static_cast<int>(displacement);
    }

    MPI_Comm localCommunicator = MPI_COMM_NULL;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                                  MPI_INFO_NULL, &localCommunicator));
    int localRank = 0;
    int localSize = 1;
    MPI_CHECK(MPI_Comm_rank(localCommunicator, &localRank));
    MPI_CHECK(MPI_Comm_size(localCommunicator, &localSize));
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        if (rank == 0) {
            std::fprintf(stderr, "No CUDA-capable device is available\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int device = localRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));
    CUDA_CHECK(cudaFree(nullptr));

    cudaDeviceProp deviceProperties = {};
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProperties, device));

    if (rank == 0) {
        std::printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        std::printf("Number of nodes: %zu\n", numNodes);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Hybrid execution: %d MPI rank(s), up to %d OpenMP "
                    "thread(s)/rank, CUDA tile %dx%d\n",
                    worldSize, omp_get_max_threads(), TILE_SIZE, TILE_SIZE);
        std::printf("Rank 0 CUDA device: %s\n", deviceProperties.name);
        if (localSize > deviceCount) {
            std::printf("Warning: multiple local MPI ranks share each GPU\n");
        }
    }

    std::vector<unsigned int> globalDist;
    if (rank == 0) {
        std::printf("Initializing graph...\n");
        globalDist.resize(totalElements);
        initializeDistanceMatrix(globalDist, numNodes, 1, MAX_DISTANCE);
    }

    std::vector<unsigned int> localDist(localElements);
    std::vector<unsigned int> localPath(localElements);
    initializeLocalPathMatrix(localPath, numNodes, decomposition);

    MPI_CHECK(MPI_Scatterv(
        rank == 0 ? globalDist.data() : nullptr, counts.data(),
        displacements.data(), MPI_UNSIGNED,
        localElements == 0 ? nullptr : localDist.data(),
        static_cast<int>(localElements), MPI_UNSIGNED, 0, MPI_COMM_WORLD));
    if (rank == 0 && !printResults && !validate) {
        globalDist.clear();
        globalDist.shrink_to_fit();
    }

    unsigned int* deviceDist = nullptr;
    unsigned int* devicePath = nullptr;
    const size_t deviceElements = std::max(localElements, static_cast<size_t>(1));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceDist),
                          deviceElements * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&devicePath),
                          deviceElements * sizeof(unsigned int)));
    if (localElements != 0) {
        CUDA_CHECK(cudaMemcpy(deviceDist, localDist.data(),
                              localElements * sizeof(unsigned int),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(devicePath, localPath.data(),
                              localElements * sizeof(unsigned int),
                              cudaMemcpyHostToDevice));
    }
    localPath.clear();
    localPath.shrink_to_fit();

    if (rank == 0) {
        std::printf("Computing shortest paths...\n");
    }
    const double duration =
        floydWarshallHybrid(deviceDist, devicePath, numNodes, decomposition,
                            rank, worldSize);

    const bool needResult = printResults || validate;
    if (needResult) {
        if (localElements != 0) {
            CUDA_CHECK(cudaMemcpy(localDist.data(), deviceDist,
                                  localElements * sizeof(unsigned int),
                                  cudaMemcpyDeviceToHost));
        }
        MPI_CHECK(MPI_Gatherv(
            localElements == 0 ? nullptr : localDist.data(),
            static_cast<int>(localElements), MPI_UNSIGNED,
            rank == 0 ? globalDist.data() : nullptr, counts.data(),
            displacements.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD));
    }

    CUDA_CHECK(cudaFree(devicePath));
    CUDA_CHECK(cudaFree(deviceDist));
    MPI_CHECK(MPI_Comm_free(&localCommunicator));

    int exitCode = 0;
    if (rank == 0) {
        const long long milliseconds =
            static_cast<long long>(std::llround(duration * 1000.0));
        std::printf("Computation time: %lld ms\n", milliseconds);
        const double operations = static_cast<double>(numNodes) *
                                  static_cast<double>(numNodes) *
                                  static_cast<double>(numNodes);
        const double gops = duration > 0.0
                                ? operations / duration / 1.0e9
                                : std::numeric_limits<double>::infinity();
        std::printf("Performance: %.3f GOPS\n", gops);

        if (printResults) {
            print_results_int(globalDist, "DistanceMatrix");
        }
        if (validate) {
            std::printf("Validating result...\n");
            if (validateResult(globalDist, numNodes)) {
                std::printf("Validation: PASSED\n");
            } else {
                std::printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }

    MPI_CHECK(MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Finalize());
    return exitCode;
}
