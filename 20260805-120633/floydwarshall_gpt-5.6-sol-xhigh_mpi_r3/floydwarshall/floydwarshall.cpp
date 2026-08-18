#include <mpi.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;
constexpr size_t TILE_SIZE = 64;

static_assert(sizeof(unsigned int) == sizeof(std::uint32_t),
              "This benchmark requires 32-bit unsigned integers");

#if defined(__GNUC__) || defined(__clang__)
#define FW_IVDEP _Pragma("GCC ivdep")
#else
#define FW_IVDEP
#endif

inline constexpr size_t idx2(const size_t i, const size_t j,
                             const size_t n) noexcept {
    return j * n + i;
}

inline size_t blockExtent(const size_t n, const size_t block) noexcept {
    const size_t begin = block * TILE_SIZE;
    return begin < n ? std::min(TILE_SIZE, n - begin) : 0;
}

size_t localExtent(const size_t n, const int coordinate,
                   const int processCount) noexcept {
    const size_t blocks = (n + TILE_SIZE - 1) / TILE_SIZE;
    size_t extent = 0;
    for (size_t block = static_cast<size_t>(coordinate); block < blocks;
         block += static_cast<size_t>(processCount)) {
        extent += blockExtent(n, block);
    }
    return extent;
}

struct Distribution {
    size_t n;
    size_t blocks;
    int processRows;
    int processCols;
    int row;
    int col;
    size_t localRows;
    size_t localCols;

    size_t localRowOffset(const size_t block) const noexcept {
        return (block / static_cast<size_t>(processRows)) * TILE_SIZE;
    }

    size_t localColOffset(const size_t block) const noexcept {
        return (block / static_cast<size_t>(processCols)) * TILE_SIZE;
    }
};

// rand_r() in glibc advances this 32-bit LCG three times per returned value.
// Reproducing it lets every rank generate only its own block-cyclic portion of
// the matrix while retaining the exact deterministic graph from the serial
// benchmark.
inline std::uint32_t lcgStep(const std::uint32_t state) noexcept {
    return state * UINT32_C(1103515245) + UINT32_C(12345);
}

std::uint32_t advanceLcg(std::uint32_t state, std::uint64_t steps) noexcept {
    std::uint32_t accumulatedMultiplier = 1;
    std::uint32_t accumulatedIncrement = 0;
    std::uint32_t multiplier = UINT32_C(1103515245);
    std::uint32_t increment = UINT32_C(12345);

    while (steps != 0) {
        if ((steps & 1U) != 0) {
            accumulatedMultiplier *= multiplier;
            accumulatedIncrement = accumulatedIncrement * multiplier + increment;
        }
        increment *= multiplier + 1U;
        multiplier *= multiplier;
        steps >>= 1U;
    }
    return accumulatedMultiplier * state + accumulatedIncrement;
}

inline unsigned int distributedRand(std::uint32_t& state) noexcept {
    unsigned int result;
    state = lcgStep(state);
    result = (state / 65536U) % 2048U;
    state = lcgStep(state);
    result = (result << 10U) ^ ((state / 65536U) % 1024U);
    state = lcgStep(state);
    result = (result << 10U) ^ ((state / 65536U) % 1024U);
    return result;
}

void initializeLocalDistanceMatrix(std::vector<unsigned int>& dist,
                                   const Distribution& distribution,
                                   const unsigned int rangeMin,
                                   const unsigned int rangeMax) {
    constexpr std::uint32_t initialSeed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;

    for (size_t blockRow = static_cast<size_t>(distribution.row);
         blockRow < distribution.blocks;
         blockRow += static_cast<size_t>(distribution.processRows)) {
        const size_t globalRowBase = blockRow * TILE_SIZE;
        const size_t localRowBase = distribution.localRowOffset(blockRow);
        const size_t height = blockExtent(distribution.n, blockRow);

        for (size_t ii = 0; ii < height; ++ii) {
            const size_t globalRow = globalRowBase + ii;
            const std::uint64_t rowStart =
                static_cast<std::uint64_t>(globalRow) * distribution.n;
            std::uint32_t state = advanceLcg(initialSeed, 3U * rowStart);
            size_t generatedThrough = 0;

            for (size_t blockCol = static_cast<size_t>(distribution.col);
                 blockCol < distribution.blocks;
                 blockCol += static_cast<size_t>(distribution.processCols)) {
                const size_t globalColBase = blockCol * TILE_SIZE;
                const size_t localColBase = distribution.localColOffset(blockCol);
                const size_t width = blockExtent(distribution.n, blockCol);

                state = advanceLcg(
                    state, 3U * static_cast<std::uint64_t>(globalColBase - generatedThrough));
                unsigned int* const output =
                    dist.data() + (localRowBase + ii) * distribution.localCols +
                    localColBase;
                for (size_t jj = 0; jj < width; ++jj) {
                    output[jj] = rangeMin + static_cast<unsigned int>(
                        range * distributedRand(state) / static_cast<double>(RAND_MAX));
                }
                generatedThrough = globalColBase + width;
            }

            if (blockRow % static_cast<size_t>(distribution.processCols) ==
                static_cast<size_t>(distribution.col)) {
                const size_t diagonalCol =
                    distribution.localColOffset(blockRow) + ii;
                dist[(localRowBase + ii) * distribution.localCols + diagonalCol] = 0;
            }
        }
    }
}

inline void relaxPivotTile(std::vector<unsigned int>& dist,
                           const Distribution& distribution,
                           const size_t pivotBlock) noexcept {
    const size_t width = blockExtent(distribution.n, pivotBlock);
    const size_t rowOffset = distribution.localRowOffset(pivotBlock);
    const size_t colOffset = distribution.localColOffset(pivotBlock);
    const size_t leadingDimension = distribution.localCols;

    for (size_t k = 0; k < width; ++k) {
        const unsigned int* const pivotRow =
            dist.data() + (rowOffset + k) * leadingDimension + colOffset;
        for (size_t i = 0; i < width; ++i) {
            unsigned int* const row =
                dist.data() + (rowOffset + i) * leadingDimension + colOffset;
            const unsigned int dik = row[k];
            FW_IVDEP
            for (size_t j = 0; j < width; ++j) {
                const unsigned int candidate = dik + pivotRow[j];
                if (candidate < row[j]) {
                    row[j] = candidate;
                }
            }
        }
    }
}

void packPivotTile(const std::vector<unsigned int>& dist,
                   const Distribution& distribution, const size_t pivotBlock,
                   std::vector<unsigned int>& pivot) {
    const size_t width = blockExtent(distribution.n, pivotBlock);
    const size_t rowOffset = distribution.localRowOffset(pivotBlock);
    const size_t colOffset = distribution.localColOffset(pivotBlock);
    for (size_t i = 0; i < width; ++i) {
        const unsigned int* const source =
            dist.data() + (rowOffset + i) * distribution.localCols + colOffset;
        std::copy_n(source, width, pivot.data() + i * width);
    }
}

void relaxPivotRow(std::vector<unsigned int>& dist,
                   const Distribution& distribution, const size_t pivotBlock,
                   const std::vector<unsigned int>& pivot) noexcept {
    const size_t pivotWidth = blockExtent(distribution.n, pivotBlock);
    const size_t rowOffset = distribution.localRowOffset(pivotBlock);
    const size_t leadingDimension = distribution.localCols;

    for (size_t blockCol = static_cast<size_t>(distribution.col);
         blockCol < distribution.blocks;
         blockCol += static_cast<size_t>(distribution.processCols)) {
        if (blockCol == pivotBlock) {
            continue;
        }
        const size_t colOffset = distribution.localColOffset(blockCol);
        const size_t width = blockExtent(distribution.n, blockCol);
        for (size_t k = 0; k < pivotWidth; ++k) {
            const unsigned int* const sourceRow =
                dist.data() + (rowOffset + k) * leadingDimension + colOffset;
            for (size_t i = 0; i < pivotWidth; ++i) {
                unsigned int* const row =
                    dist.data() + (rowOffset + i) * leadingDimension + colOffset;
                const unsigned int dik = pivot[i * pivotWidth + k];
                FW_IVDEP
                for (size_t j = 0; j < width; ++j) {
                    const unsigned int candidate = dik + sourceRow[j];
                    if (candidate < row[j]) {
                        row[j] = candidate;
                    }
                }
            }
        }
    }
}

void relaxPivotColumn(std::vector<unsigned int>& dist,
                      const Distribution& distribution, const size_t pivotBlock,
                      const std::vector<unsigned int>& pivot) noexcept {
    const size_t pivotWidth = blockExtent(distribution.n, pivotBlock);
    const size_t colOffset = distribution.localColOffset(pivotBlock);
    const size_t leadingDimension = distribution.localCols;

    for (size_t blockRow = static_cast<size_t>(distribution.row);
         blockRow < distribution.blocks;
         blockRow += static_cast<size_t>(distribution.processRows)) {
        if (blockRow == pivotBlock) {
            continue;
        }
        const size_t rowOffset = distribution.localRowOffset(blockRow);
        const size_t height = blockExtent(distribution.n, blockRow);
        for (size_t k = 0; k < pivotWidth; ++k) {
            const unsigned int* const pivotRow = pivot.data() + k * pivotWidth;
            for (size_t i = 0; i < height; ++i) {
                unsigned int* const row =
                    dist.data() + (rowOffset + i) * leadingDimension + colOffset;
                const unsigned int dik = row[k];
                FW_IVDEP
                for (size_t j = 0; j < pivotWidth; ++j) {
                    const unsigned int candidate = dik + pivotRow[j];
                    if (candidate < row[j]) {
                        row[j] = candidate;
                    }
                }
            }
        }
    }
}

void packColumnPanel(const std::vector<unsigned int>& dist,
                     const Distribution& distribution, const size_t pivotBlock,
                     std::vector<unsigned int>& panel) {
    const size_t pivotWidth = blockExtent(distribution.n, pivotBlock);
    const size_t colOffset = distribution.localColOffset(pivotBlock);
    for (size_t i = 0; i < distribution.localRows; ++i) {
        const unsigned int* const source =
            dist.data() + i * distribution.localCols + colOffset;
        std::copy_n(source, pivotWidth, panel.data() + i * pivotWidth);
    }
}

void relaxRemainingTiles(std::vector<unsigned int>& dist,
                         const Distribution& distribution,
                         const size_t pivotBlock,
                         const std::vector<unsigned int>& rowPanel,
                         const std::vector<unsigned int>& colPanel) noexcept {
    const size_t pivotWidth = blockExtent(distribution.n, pivotBlock);
    const size_t leadingDimension = distribution.localCols;

    for (size_t blockRow = static_cast<size_t>(distribution.row);
         blockRow < distribution.blocks;
         blockRow += static_cast<size_t>(distribution.processRows)) {
        if (blockRow == pivotBlock) {
            continue;
        }
        const size_t rowOffset = distribution.localRowOffset(blockRow);
        const size_t height = blockExtent(distribution.n, blockRow);
        for (size_t blockCol = static_cast<size_t>(distribution.col);
             blockCol < distribution.blocks;
             blockCol += static_cast<size_t>(distribution.processCols)) {
            if (blockCol == pivotBlock) {
                continue;
            }
            const size_t colOffset = distribution.localColOffset(blockCol);
            const size_t width = blockExtent(distribution.n, blockCol);
            for (size_t k = 0; k < pivotWidth; ++k) {
                const unsigned int* const sourceRow =
                    rowPanel.data() + k * leadingDimension + colOffset;
                for (size_t i = 0; i < height; ++i) {
                    unsigned int* const row =
                        dist.data() + (rowOffset + i) * leadingDimension + colOffset;
                    const unsigned int dik =
                        colPanel[(rowOffset + i) * pivotWidth + k];
                    FW_IVDEP
                    for (size_t j = 0; j < width; ++j) {
                        const unsigned int candidate = dik + sourceRow[j];
                        if (candidate < row[j]) {
                            row[j] = candidate;
                        }
                    }
                }
            }
        }
    }
}

int mpiCount(const size_t count, MPI_Comm world) {
    if (count > static_cast<size_t>(INT_MAX)) {
        std::fprintf(stderr, "MPI message exceeds the implementation count limit\n");
        MPI_Abort(world, 2);
    }
    return static_cast<int>(count);
}

void distributedFloydWarshall(std::vector<unsigned int>& dist,
                              const Distribution& distribution,
                              MPI_Comm rowCommunicator,
                              MPI_Comm colCommunicator,
                              MPI_Comm world) {
    std::vector<unsigned int> pivot(TILE_SIZE * TILE_SIZE);
    std::vector<unsigned int> rowPanel(TILE_SIZE * distribution.localCols);
    std::vector<unsigned int> colPanel(distribution.localRows * TILE_SIZE);

    for (size_t pivotBlock = 0; pivotBlock < distribution.blocks; ++pivotBlock) {
        const size_t pivotWidth = blockExtent(distribution.n, pivotBlock);
        const int ownerRow =
            static_cast<int>(pivotBlock % static_cast<size_t>(distribution.processRows));
        const int ownerCol =
            static_cast<int>(pivotBlock % static_cast<size_t>(distribution.processCols));

        if (distribution.row == ownerRow && distribution.col == ownerCol) {
            relaxPivotTile(dist, distribution, pivotBlock);
            packPivotTile(dist, distribution, pivotBlock, pivot);
        }

        const int pivotCount = mpiCount(pivotWidth * pivotWidth, world);
        MPI_Request pivotRequests[2];
        int pivotRequestCount = 0;
        if (distribution.row == ownerRow) {
            MPI_Ibcast(pivot.data(), pivotCount, MPI_UNSIGNED, ownerCol,
                       rowCommunicator, &pivotRequests[pivotRequestCount++]);
        }
        if (distribution.col == ownerCol) {
            MPI_Ibcast(pivot.data(), pivotCount, MPI_UNSIGNED, ownerRow,
                       colCommunicator, &pivotRequests[pivotRequestCount++]);
        }
        if (pivotRequestCount != 0) {
            MPI_Waitall(pivotRequestCount, pivotRequests, MPI_STATUSES_IGNORE);
        }
        if (distribution.row == ownerRow) {
            relaxPivotRow(dist, distribution, pivotBlock, pivot);
        }
        if (distribution.col == ownerCol) {
            relaxPivotColumn(dist, distribution, pivotBlock, pivot);
        }

        const size_t rowPanelElements = pivotWidth * distribution.localCols;
        const size_t colPanelElements = distribution.localRows * pivotWidth;

        if (distribution.row == ownerRow) {
            const size_t rowOffset = distribution.localRowOffset(pivotBlock);
            std::copy_n(dist.data() + rowOffset * distribution.localCols,
                        rowPanelElements, rowPanel.data());
        }
        if (distribution.col == ownerCol) {
            packColumnPanel(dist, distribution, pivotBlock, colPanel);
        }

        MPI_Request requests[2];
        MPI_Ibcast(rowPanel.data(), mpiCount(rowPanelElements, world), MPI_UNSIGNED,
                   ownerRow, colCommunicator, &requests[0]);
        MPI_Ibcast(colPanel.data(), mpiCount(colPanelElements, world), MPI_UNSIGNED,
                   ownerCol, rowCommunicator, &requests[1]);
        MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);

        relaxRemainingTiles(dist, distribution, pivotBlock, rowPanel, colPanel);
    }
}

void transferUnsigned(const unsigned int* sendData, unsigned int* receiveData,
                      const size_t count, const int peer, const bool sending,
                      MPI_Comm communicator) {
    size_t offset = 0;
    while (offset < count) {
        const int chunk = static_cast<int>(
            std::min(count - offset, static_cast<size_t>(INT_MAX)));
        if (sending) {
            MPI_Send(sendData + offset, chunk, MPI_UNSIGNED, peer, 73, communicator);
        } else {
            MPI_Recv(receiveData + offset, chunk, MPI_UNSIGNED, peer, 73,
                     communicator, MPI_STATUS_IGNORE);
        }
        offset += static_cast<size_t>(chunk);
    }
}

void unpackLocalMatrix(const std::vector<unsigned int>& local,
                       std::vector<unsigned int>& global, const size_t n,
                       const int processRows, const int processCols,
                       const int row, const int col) {
    const size_t blocks = (n + TILE_SIZE - 1) / TILE_SIZE;
    const size_t localCols = localExtent(n, col, processCols);
    for (size_t blockRow = static_cast<size_t>(row); blockRow < blocks;
         blockRow += static_cast<size_t>(processRows)) {
        const size_t globalRow = blockRow * TILE_SIZE;
        const size_t localRow =
            (blockRow / static_cast<size_t>(processRows)) * TILE_SIZE;
        const size_t height = blockExtent(n, blockRow);
        for (size_t blockCol = static_cast<size_t>(col); blockCol < blocks;
             blockCol += static_cast<size_t>(processCols)) {
            const size_t globalCol = blockCol * TILE_SIZE;
            const size_t localCol =
                (blockCol / static_cast<size_t>(processCols)) * TILE_SIZE;
            const size_t width = blockExtent(n, blockCol);
            for (size_t i = 0; i < height; ++i) {
                const unsigned int* const source =
                    local.data() + (localRow + i) * localCols + localCol;
                unsigned int* const destination =
                    global.data() + (globalRow + i) * n + globalCol;
                std::copy_n(source, width, destination);
            }
        }
    }
}

std::vector<unsigned int> gatherDistanceMatrix(
    const std::vector<unsigned int>& local, const Distribution& distribution,
    const int worldRank, const int worldSize, MPI_Comm world) {
    if (worldRank != 0) {
        transferUnsigned(local.data(), nullptr, local.size(), 0, true, world);
        return {};
    }

    std::vector<unsigned int> global(distribution.n * distribution.n);
    for (int rank = 0; rank < worldSize; ++rank) {
        const int row = rank / distribution.processCols;
        const int col = rank % distribution.processCols;
        const size_t rows =
            localExtent(distribution.n, row, distribution.processRows);
        const size_t cols =
            localExtent(distribution.n, col, distribution.processCols);

        if (rank == 0) {
            unpackLocalMatrix(local, global, distribution.n,
                              distribution.processRows, distribution.processCols,
                              row, col);
        } else {
            std::vector<unsigned int> received(rows * cols);
            transferUnsigned(nullptr, received.data(), received.size(), rank, false,
                             world);
            unpackLocalMatrix(received, global, distribution.n,
                              distribution.processRows, distribution.processCols,
                              row, col);
        }
    }
    return global;
}

bool validateResult(const std::vector<unsigned int>& dist, const size_t numNodes) {
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
    MPI_Init(&argc, &argv);

    int worldRank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;
    int argumentStatus = 0;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const unsigned long long value = std::strtoull(argv[++i], &end, 10);
            if (end == argv[i] || *end != '\0' ||
                value > std::numeric_limits<size_t>::max()) {
                argumentStatus = 1;
                if (worldRank == 0) {
                    std::printf("Invalid number of nodes: %s\n", argv[i]);
                }
                break;
            }
            numNodes = static_cast<size_t>(value);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            if (worldRank == 0) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            argumentStatus = 1;
            if (worldRank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
            }
            break;
        }
    }

    if (argumentStatus != 0 ||
        (numNodes != 0 && numNodes > std::numeric_limits<size_t>::max() / numNodes)) {
        if (worldRank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    int dimensions[2] = {0, 0};
    MPI_Dims_create(worldSize, 2, dimensions);
    const int processRows = dimensions[0];
    const int processCols = dimensions[1];
    const int processRow = worldRank / processCols;
    const int processCol = worldRank % processCols;

    MPI_Comm rowCommunicator = MPI_COMM_NULL;
    MPI_Comm colCommunicator = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, processRow, processCol, &rowCommunicator);
    MPI_Comm_split(MPI_COMM_WORLD, processCol, processRow, &colCommunicator);

    const Distribution distribution{
        numNodes,
        (numNodes + TILE_SIZE - 1) / TILE_SIZE,
        processRows,
        processCols,
        processRow,
        processCol,
        localExtent(numNodes, processRow, processRows),
        localExtent(numNodes, processCol, processCols)};

    if (worldRank == 0) {
        std::printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        std::printf("Number of nodes: %zu\n", numNodes);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("MPI processes: %d (%d x %d grid)\n", worldSize, processRows,
                    processCols);
        std::printf("Initializing graph...\n");
    }

    std::vector<unsigned int> localDistance(distribution.localRows *
                                            distribution.localCols);
    initializeLocalDistanceMatrix(localDistance, distribution, 1, MAX_DISTANCE);

    if (worldRank == 0) {
        std::printf("Computing shortest paths...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    distributedFloydWarshall(localDistance, distribution, rowCommunicator,
                             colCommunicator, MPI_COMM_WORLD);
    const double localSeconds = MPI_Wtime() - start;
    double elapsedSeconds = 0.0;
    MPI_Reduce(&localSeconds, &elapsedSeconds, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);

    if (worldRank == 0) {
        const long durationMs = static_cast<long>(elapsedSeconds * 1000.0);
        std::printf("Computation time: %ld ms\n", durationMs);
        const double operations = static_cast<double>(numNodes) * numNodes * numNodes;
        const double gops = elapsedSeconds > 0.0
                                ? operations / elapsedSeconds / 1.0e9
                                : std::numeric_limits<double>::infinity();
        std::printf("Performance: %.3f GOPS\n", gops);
    }

    std::vector<unsigned int> globalDistance;
    if (printResults || validate) {
        globalDistance = gatherDistanceMatrix(localDistance, distribution, worldRank,
                                              worldSize, MPI_COMM_WORLD);
    }

    int result = 0;
    if (worldRank == 0) {
        if (printResults) {
            print_results_int(globalDistance, "DistanceMatrix");
        }
        if (validate) {
            std::printf("Validating result...\n");
            if (validateResult(globalDistance, numNodes)) {
                std::printf("Validation: PASSED\n");
            } else {
                std::printf("Validation: FAILED\n");
                result = 1;
            }
        }
    }
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Comm_free(&rowCommunicator);
    MPI_Comm_free(&colCommunicator);
    MPI_Finalize();
    return result;
}
