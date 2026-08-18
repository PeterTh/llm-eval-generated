#include <mpi.h>

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

// Index calculation for a flattened, row-major 2D array.
inline constexpr size_t idx2(const size_t row, const size_t column,
                             const size_t rowLength) noexcept {
    return row * rowLength + column;
}

inline size_t blockSize(const size_t n, const int coordinate,
                        const int parts) noexcept {
    const size_t base = n / static_cast<size_t>(parts);
    const size_t remainder = n % static_cast<size_t>(parts);
    return base + (static_cast<size_t>(coordinate) < remainder ? 1U : 0U);
}

inline size_t blockStart(const size_t n, const int coordinate,
                         const int parts) noexcept {
    const size_t base = n / static_cast<size_t>(parts);
    const size_t remainder = n % static_cast<size_t>(parts);
    return static_cast<size_t>(coordinate) * base +
           std::min(static_cast<size_t>(coordinate), remainder);
}

inline int blockOwner(const size_t index, const size_t n,
                      const int parts) noexcept {
    const size_t base = n / static_cast<size_t>(parts);
    const size_t remainder = n % static_cast<size_t>(parts);
    const size_t largeBlockElements = (base + 1U) * remainder;

    if (index < largeBlockElements) {
        return static_cast<int>(index / (base + 1U));
    }
    return static_cast<int>(remainder +
                            (index - largeBlockElements) / base);
}

// glibc rand_r advances its 32-bit LCG three times per returned value.  Jumping
// that LCG ahead lets every process initialize its own matrix block while
// reproducing exactly the single-process benchmark's seed-42 input matrix.
unsigned int advanceRandomSeed(const unsigned int initialSeed,
                               const uint64_t randomValues) noexcept {
    uint64_t delta = randomValues * 3U;
    uint32_t accumulatedMultiplier = 1U;
    uint32_t accumulatedIncrement = 0U;
    uint32_t currentMultiplier = 1103515245U;
    uint32_t currentIncrement = 12345U;

    while (delta != 0U) {
        if ((delta & 1U) != 0U) {
            accumulatedMultiplier = static_cast<uint32_t>(
                static_cast<uint64_t>(accumulatedMultiplier) * currentMultiplier);
            accumulatedIncrement = static_cast<uint32_t>(
                static_cast<uint64_t>(accumulatedIncrement) * currentMultiplier +
                currentIncrement);
        }
        currentIncrement = static_cast<uint32_t>(
            (static_cast<uint64_t>(currentMultiplier) + 1U) * currentIncrement);
        currentMultiplier = static_cast<uint32_t>(
            static_cast<uint64_t>(currentMultiplier) * currentMultiplier);
        delta >>= 1U;
    }

    return static_cast<unsigned int>(
        static_cast<uint32_t>(
            static_cast<uint64_t>(accumulatedMultiplier) * initialSeed +
            accumulatedIncrement));
}

void initializeLocalMatrices(std::vector<unsigned int>& dist,
                             std::vector<unsigned int>& path,
                             const size_t numNodes, const size_t rowStart,
                             const size_t localRows, const size_t columnStart,
                             const size_t localColumns,
                             const unsigned int rangeMin,
                             const unsigned int rangeMax) {
    if (localRows == 0U || localColumns == 0U) {
        return;
    }

    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;
    const size_t firstElement = rowStart * numNodes + columnStart;
    unsigned int seed = advanceRandomSeed(42U, firstElement);

    for (size_t localRow = 0; localRow < localRows; ++localRow) {
        const size_t globalRow = rowStart + localRow;
        unsigned int* const distRow = dist.data() + localRow * localColumns;
        unsigned int* const pathRow = path.data() + localRow * localColumns;

        for (size_t localColumn = 0; localColumn < localColumns; ++localColumn) {
            distRow[localColumn] = rangeMin + static_cast<unsigned int>(
                range * rand_r(&seed) / static_cast<double>(RAND_MAX));
            pathRow[localColumn] = static_cast<unsigned int>(globalRow);
        }

        if (globalRow >= columnStart &&
            globalRow < columnStart + localColumns) {
            distRow[globalRow - columnStart] = 0U;
        }

        // Move from the end of this local row segment to the same column in
        // the next global row.
        if (localRow + 1U < localRows) {
            seed = advanceRandomSeed(seed, numNodes - localColumns);
        }
    }
}

inline void updateRange(unsigned int* const distRow,
                        unsigned int* const pathRow,
                        const unsigned int distanceToK,
                        const unsigned int* const pivotRow,
                        const size_t begin, const size_t end,
                        const unsigned int k) noexcept {
#if defined(__GNUC__)
#pragma GCC ivdep
#endif
    for (size_t column = begin; column < end; ++column) {
        const unsigned int newDistance = distanceToK + pivotRow[column];
        if (newDistance < distRow[column]) {
            distRow[column] = newDistance;
            pathRow[column] = k;
        }
    }
}

void updateBlock(std::vector<unsigned int>& dist,
                 std::vector<unsigned int>& path,
                 const std::vector<unsigned int>& pivotRow,
                 const std::vector<unsigned int>& pivotColumn,
                 const size_t localRows, const size_t localColumns,
                 const size_t excludedRow, const size_t excludedColumn,
                 const unsigned int k) noexcept {
    if (localColumns == 0U) {
        return;
    }

    for (size_t row = 0; row < localRows; ++row) {
        if (row == excludedRow) {
            continue;
        }

        unsigned int* const distRow = dist.data() + row * localColumns;
        unsigned int* const pathRow = path.data() + row * localColumns;
        const unsigned int distanceToK = pivotColumn[row];

        if (excludedColumn < localColumns) {
            updateRange(distRow, pathRow, distanceToK, pivotRow.data(), 0U,
                        excludedColumn, k);
            updateRange(distRow, pathRow, distanceToK, pivotRow.data(),
                        excludedColumn + 1U, localColumns, k);
        } else {
            updateRange(distRow, pathRow, distanceToK, pivotRow.data(), 0U,
                        localColumns, k);
        }
    }
}

void updateNextPivot(std::vector<unsigned int>& dist,
                     std::vector<unsigned int>& path,
                     const std::vector<unsigned int>& pivotRow,
                     const std::vector<unsigned int>& pivotColumn,
                     const size_t localRows, const size_t localColumns,
                     const size_t nextLocalRow, const size_t nextLocalColumn,
                     const unsigned int k) noexcept {
    if (localColumns == 0U) {
        return;
    }

    if (nextLocalRow < localRows) {
        unsigned int* const distRow =
            dist.data() + nextLocalRow * localColumns;
        unsigned int* const pathRow =
            path.data() + nextLocalRow * localColumns;
        updateRange(distRow, pathRow, pivotColumn[nextLocalRow], pivotRow.data(),
                    0U, localColumns, k);
    }

    if (nextLocalColumn < localColumns) {
        for (size_t row = 0; row < localRows; ++row) {
            if (row == nextLocalRow) {
                continue;
            }
            unsigned int& oldDistance =
                dist[idx2(row, nextLocalColumn, localColumns)];
            const unsigned int newDistance =
                pivotColumn[row] + pivotRow[nextLocalColumn];
            if (newDistance < oldDistance) {
                oldDistance = newDistance;
                path[idx2(row, nextLocalColumn, localColumns)] = k;
            }
        }
    }
}

void preparePivot(const std::vector<unsigned int>& dist,
                  std::vector<unsigned int>& pivotRow,
                  std::vector<unsigned int>& pivotColumn,
                  const size_t k, const size_t rowStart,
                  const size_t localRows, const size_t columnStart,
                  const size_t localColumns) {
    if (localColumns != 0U && k >= rowStart &&
        k < rowStart + localRows) {
        const unsigned int* const source =
            dist.data() + (k - rowStart) * localColumns;
        std::copy_n(source, localColumns, pivotRow.data());
    }
    if (k >= columnStart && k < columnStart + localColumns) {
        const size_t localColumn = k - columnStart;
        for (size_t row = 0; row < localRows; ++row) {
            pivotColumn[row] = dist[idx2(row, localColumn, localColumns)];
        }
    }
}

void startPivotBroadcast(std::vector<unsigned int>& pivotRow,
                         std::vector<unsigned int>& pivotColumn,
                         const size_t k, const size_t numNodes,
                         const int processRows, const int processColumns,
                         MPI_Comm columnComm, MPI_Comm rowComm,
                         MPI_Request requests[2]) {
    const int pivotProcessRow = blockOwner(k, numNodes, processRows);
    const int pivotProcessColumn = blockOwner(k, numNodes, processColumns);
    MPI_Ibcast(pivotRow.data(), static_cast<int>(pivotRow.size()), MPI_UNSIGNED,
               pivotProcessRow, columnComm, &requests[0]);
    MPI_Ibcast(pivotColumn.data(), static_cast<int>(pivotColumn.size()),
               MPI_UNSIGNED, pivotProcessColumn, rowComm, &requests[1]);
}

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes, const size_t rowStart,
                   const size_t localRows, const size_t columnStart,
                   const size_t localColumns, const int processRows,
                   const int processColumns, MPI_Comm columnComm,
                   MPI_Comm rowComm) {
    if (numNodes == 0U) {
        return;
    }

    std::vector<unsigned int> pivotRows[2] = {
        std::vector<unsigned int>(localColumns),
        std::vector<unsigned int>(localColumns)};
    std::vector<unsigned int> pivotColumns[2] = {
        std::vector<unsigned int>(localRows),
        std::vector<unsigned int>(localRows)};
    MPI_Request requests[2][2];

    int slot = 0;
    preparePivot(dist, pivotRows[slot], pivotColumns[slot], 0U, rowStart,
                 localRows, columnStart, localColumns);
    startPivotBroadcast(pivotRows[slot], pivotColumns[slot], 0U, numNodes,
                        processRows, processColumns, columnComm, rowComm,
                        requests[slot]);

    for (size_t k = 0; k < numNodes; ++k) {
        MPI_Waitall(2, requests[slot], MPI_STATUSES_IGNORE);

        if (k + 1U < numNodes) {
            const size_t nextK = k + 1U;
            const size_t nextLocalRow =
                nextK >= rowStart && nextK < rowStart + localRows
                    ? nextK - rowStart
                    : localRows;
            const size_t nextLocalColumn =
                nextK >= columnStart && nextK < columnStart + localColumns
                    ? nextK - columnStart
                    : localColumns;

            // Update the row and column needed by iteration k+1 first.  Their
            // broadcasts can then progress while the rest of this block is
            // computed.
            updateNextPivot(dist, path, pivotRows[slot], pivotColumns[slot],
                            localRows, localColumns, nextLocalRow,
                            nextLocalColumn, static_cast<unsigned int>(k));

            const int nextSlot = slot ^ 1;
            preparePivot(dist, pivotRows[nextSlot], pivotColumns[nextSlot],
                         nextK, rowStart, localRows, columnStart, localColumns);
            startPivotBroadcast(pivotRows[nextSlot], pivotColumns[nextSlot],
                                nextK, numNodes, processRows, processColumns,
                                columnComm, rowComm, requests[nextSlot]);

            updateBlock(dist, path, pivotRows[slot], pivotColumns[slot],
                        localRows, localColumns, nextLocalRow, nextLocalColumn,
                        static_cast<unsigned int>(k));
            slot = nextSlot;
        } else {
            updateBlock(dist, path, pivotRows[slot], pivotColumns[slot],
                        localRows, localColumns, localRows, localColumns,
                        static_cast<unsigned int>(k));
        }
    }
}

std::vector<unsigned int> gatherDistanceMatrix(
    const std::vector<unsigned int>& localDist, const size_t numNodes,
    const int worldRank, const int worldSize, const int processRows,
    const int processColumns, MPI_Comm communicator) {
    constexpr int gatherTag = 417;
    std::vector<unsigned int> result;
    if (worldRank == 0) {
        result.resize(numNodes * numNodes);
    }

    for (int sender = 0; sender < worldSize; ++sender) {
        const int senderRow = sender / processColumns;
        const int senderColumn = sender % processColumns;
        const size_t senderRows = blockSize(numNodes, senderRow, processRows);
        const size_t senderColumns =
            blockSize(numNodes, senderColumn, processColumns);
        const size_t senderRowStart =
            blockStart(numNodes, senderRow, processRows);
        const size_t senderColumnStart =
            blockStart(numNodes, senderColumn, processColumns);

        if (senderRows == 0U || senderColumns == 0U) {
            continue;
        }

        const size_t rowsPerMessage =
            static_cast<size_t>(INT_MAX) / senderColumns;
        for (size_t rowOffset = 0; rowOffset < senderRows;
             rowOffset += rowsPerMessage) {
            const size_t messageRows =
                std::min(rowsPerMessage, senderRows - rowOffset);
            const int messageElements =
                static_cast<int>(messageRows * senderColumns);

            if (worldRank == sender) {
                if (sender == 0) {
                    for (size_t row = 0; row < messageRows; ++row) {
                        const unsigned int* const source =
                            localDist.data() +
                            (rowOffset + row) * senderColumns;
                        unsigned int* const destination =
                            result.data() +
                            (senderRowStart + rowOffset + row) * numNodes +
                            senderColumnStart;
                        std::copy_n(source, senderColumns, destination);
                    }
                } else {
                    MPI_Send(localDist.data() + rowOffset * senderColumns,
                             messageElements, MPI_UNSIGNED, 0, gatherTag,
                             communicator);
                }
            } else if (worldRank == 0) {
                const int globalSizes[2] = {static_cast<int>(numNodes),
                                            static_cast<int>(numNodes)};
                const int subSizes[2] = {static_cast<int>(messageRows),
                                         static_cast<int>(senderColumns)};
                const int starts[2] = {
                    static_cast<int>(senderRowStart + rowOffset),
                    static_cast<int>(senderColumnStart)};
                MPI_Datatype receiveType;
                MPI_Type_create_subarray(2, globalSizes, subSizes, starts,
                                         MPI_ORDER_C, MPI_UNSIGNED,
                                         &receiveType);
                MPI_Type_commit(&receiveType);
                MPI_Recv(result.data(), 1, receiveType, sender, gatherTag,
                         communicator, MPI_STATUS_IGNORE);
                MPI_Type_free(&receiveType);
            }
        }
    }

    return result;
}

bool validateResult(const std::vector<unsigned int>& dist,
                    const size_t numNodes) {
    for (size_t i = 0; i < numNodes; ++i) {
        if (dist[idx2(i, i, numNodes)] != 0U) {
            std::printf(
                "Validation failed: diagonal element [%zu,%zu] is not zero\n",
                i, i);
            return false;
        }
    }

    const size_t sampleSize = std::min(numNodes, static_cast<size_t>(10));
    for (size_t i = 0; i < sampleSize; ++i) {
        for (size_t j = 0; j < sampleSize; ++j) {
            for (size_t k = 0; k < numNodes; ++k) {
                const unsigned int distIJ = dist[idx2(i, j, numNodes)];
                const unsigned int distIK = dist[idx2(i, k, numNodes)];
                const unsigned int distKJ = dist[idx2(k, j, numNodes)];

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

bool parseNodeCount(const char* text, size_t& value) noexcept {
    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || text[0] == '-' ||
        parsed > std::numeric_limits<size_t>::max()) {
        return false;
    }
    value = static_cast<size_t>(parsed);
    return true;
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
    bool showHelp = false;
    bool argumentsValid = true;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const bool nodeCountValid = parseNodeCount(argv[++i], numNodes);
            argumentsValid = argumentsValid && nodeCountValid;
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            showHelp = true;
        } else {
            if (worldRank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
            }
            argumentsValid = false;
        }
    }

    if (showHelp || !argumentsValid) {
        if (worldRank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return argumentsValid ? 0 : 1;
    }

    if (numNodes > static_cast<size_t>(INT_MAX) ||
        (numNodes != 0U &&
         numNodes > std::numeric_limits<size_t>::max() / numNodes)) {
        if (worldRank == 0) {
            std::fprintf(stderr, "The requested matrix dimensions are too large.\n");
        }
        MPI_Finalize();
        return 1;
    }

    int dimensions[2] = {0, 0};
    MPI_Dims_create(worldSize, 2, dimensions);
    const int processRows = dimensions[0];
    const int processColumns = dimensions[1];
    const int processRow = worldRank / processColumns;
    const int processColumn = worldRank % processColumns;

    MPI_Comm rowComm = MPI_COMM_NULL;
    MPI_Comm columnComm = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, processRow, processColumn, &rowComm);
    MPI_Comm_split(MPI_COMM_WORLD, processColumn, processRow, &columnComm);

    const size_t localRows = blockSize(numNodes, processRow, processRows);
    const size_t localColumns =
        blockSize(numNodes, processColumn, processColumns);
    const size_t rowStart = blockStart(numNodes, processRow, processRows);
    const size_t columnStart =
        blockStart(numNodes, processColumn, processColumns);

    int returnCode = 0;
    try {
        if (worldRank == 0) {
            std::printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
            std::printf("Number of nodes: %zu\n", numNodes);
            std::printf("Validation: %s\n",
                        validate ? "enabled" : "disabled");
            std::printf("Initializing graph...\n");
        }

        std::vector<unsigned int> localDist(localRows * localColumns);
        std::vector<unsigned int> localPath(localRows * localColumns);
        initializeLocalMatrices(localDist, localPath, numNodes, rowStart,
                                localRows, columnStart, localColumns, 1U,
                                MAX_DISTANCE);

        if (worldRank == 0) {
            std::printf("Computing shortest paths...\n");
        }
        MPI_Barrier(MPI_COMM_WORLD);
        const double start = MPI_Wtime();

        floydWarshall(localDist, localPath, numNodes, rowStart, localRows,
                      columnStart, localColumns, processRows, processColumns,
                      columnComm, rowComm);

        const double localElapsed = MPI_Wtime() - start;
        double elapsed = 0.0;
        MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0,
                   MPI_COMM_WORLD);

        if (worldRank == 0) {
            const long long durationMs =
                static_cast<long long>(elapsed * 1000.0);
            std::printf("Computation time: %lld ms\n", durationMs);
            const double operations = static_cast<double>(numNodes) *
                                      static_cast<double>(numNodes) *
                                      static_cast<double>(numNodes);
            const double gops = operations / elapsed / 1.0e9;
            std::printf("Performance: %.3f GOPS\n", gops);
        }

        std::vector<unsigned int> distanceMatrix;
        if (printResults || validate) {
            distanceMatrix = gatherDistanceMatrix(
                localDist, numNodes, worldRank, worldSize, processRows,
                processColumns, MPI_COMM_WORLD);
        }

        if (worldRank == 0 && printResults) {
            print_results_int(distanceMatrix, "DistanceMatrix");
        }

        int valid = 1;
        if (validate) {
            if (worldRank == 0) {
                std::printf("Validating result...\n");
                valid = validateResult(distanceMatrix, numNodes) ? 1 : 0;
                std::printf("Validation: %s\n", valid != 0 ? "PASSED" : "FAILED");
            }
            MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
            returnCode = valid != 0 ? 0 : 1;
        }
    } catch (const std::exception& exception) {
        std::fprintf(stderr, "Rank %d failed: %s\n", worldRank,
                     exception.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
        returnCode = 1;
    }

    MPI_Comm_free(&columnComm);
    MPI_Comm_free(&rowComm);
    MPI_Finalize();
    return returnCode;
}
