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

#include "../common/hash.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;
constexpr size_t BLOCK_SIZE = 64;

struct Partition {
    size_t begin;
    size_t count;
};

Partition partitionFor(const size_t n, const int parts, const int coordinate) noexcept {
    const size_t base = n / static_cast<size_t>(parts);
    const size_t remainder = n % static_cast<size_t>(parts);
    const size_t extra = static_cast<size_t>(coordinate) < remainder ? 1 : 0;
    const size_t begin = static_cast<size_t>(coordinate) * base
                       + std::min(static_cast<size_t>(coordinate), remainder);
    return {begin, base + extra};
}

int ownerOf(const size_t index, const size_t n, const int parts) noexcept {
    const size_t base = n / static_cast<size_t>(parts);
    const size_t remainder = n % static_cast<size_t>(parts);
    const size_t largeRegion = (base + 1) * remainder;

    if (index < largeRegion) {
        return static_cast<int>(index / (base + 1));
    }
    return static_cast<int>(remainder + (index - largeRegion) / base);
}

// rand_r advances its 32-bit LCG three times per returned value.  Jumping the
// state lets every rank recreate its rectangular portion of the original
// serial random stream without storing or communicating the complete matrix.
unsigned int advanceRandSeed(const unsigned int initialSeed, size_t calls) noexcept {
    static_assert(std::numeric_limits<unsigned int>::digits == 32);
    constexpr uint32_t multiplier = 1103515245u;
    constexpr uint32_t increment = 12345u;

    uint32_t callMultiplier = 1;
    uint32_t callIncrement = 0;
    for (int i = 0; i < 3; ++i) {
        callIncrement = multiplier * callIncrement + increment;
        callMultiplier = multiplier * callMultiplier;
    }

    uint32_t accumulatedMultiplier = 1;
    uint32_t accumulatedIncrement = 0;
    while (calls != 0) {
        if ((calls & 1U) != 0) {
            accumulatedIncrement = callMultiplier * accumulatedIncrement + callIncrement;
            accumulatedMultiplier = callMultiplier * accumulatedMultiplier;
        }
        callIncrement = (callMultiplier + 1U) * callIncrement;
        callMultiplier *= callMultiplier;
        calls >>= 1U;
    }

    return accumulatedMultiplier * initialSeed + accumulatedIncrement;
}

void initializeDistanceMatrix(std::vector<unsigned int>& dist,
                              const size_t numNodes,
                              const Partition rows,
                              const Partition columns,
                              const unsigned int rangeMin,
                              const unsigned int rangeMax) {
    if (columns.count == 0) {
        return;
    }
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;

    for (size_t localRow = 0; localRow < rows.count; ++localRow) {
        const size_t globalRow = rows.begin + localRow;
        unsigned int seed = advanceRandSeed(42U, globalRow * numNodes + columns.begin);
        unsigned int* const row = dist.data() + localRow * columns.count;

        for (size_t localColumn = 0; localColumn < columns.count; ++localColumn) {
            row[localColumn] = rangeMin + static_cast<unsigned int>(
                range * rand_r(&seed) / static_cast<double>(RAND_MAX));
        }

        if (globalRow >= columns.begin && globalRow < columns.begin + columns.count) {
            row[globalRow - columns.begin] = 0;
        }
    }
}

inline void updateContiguous(unsigned int* __restrict destination,
                             const unsigned int* __restrict source,
                             const unsigned int leftDistance,
                             const size_t begin,
                             const size_t end) noexcept {
    for (size_t j = begin; j < end; ++j) {
        const unsigned int candidate = leftDistance + source[j];
        if (candidate < destination[j]) {
            destination[j] = candidate;
        }
    }
}

void closePivot(std::vector<unsigned int>& pivot, const size_t blockSize) noexcept {
    for (size_t q = 0; q < blockSize; ++q) {
        const unsigned int* const qRow = pivot.data() + q * blockSize;
        for (size_t i = 0; i < blockSize; ++i) {
            if (i == q) {
                continue;
            }
            unsigned int* const row = pivot.data() + i * blockSize;
            updateContiguous(row, qRow, row[q], 0, blockSize);
        }
    }
}

void updatePivotRows(std::vector<unsigned int>& dist,
                     const std::vector<unsigned int>& pivot,
                     const Partition columns,
                     const size_t localPivotRow,
                     const size_t pivotBegin,
                     const size_t blockSize) noexcept {
    if (columns.count == 0) {
        return;
    }
    size_t leftEnd = columns.count;
    size_t rightBegin = columns.count;
    if (pivotBegin >= columns.begin && pivotBegin < columns.begin + columns.count) {
        leftEnd = pivotBegin - columns.begin;
        rightBegin = leftEnd + blockSize;
    }

    for (size_t q = 0; q < blockSize; ++q) {
        const unsigned int* const qRow = dist.data()
                                       + (localPivotRow + q) * columns.count;
        for (size_t i = 0; i < blockSize; ++i) {
            if (i == q) {
                continue;
            }
            unsigned int* const row = dist.data()
                                    + (localPivotRow + i) * columns.count;
            const unsigned int leftDistance = pivot[i * blockSize + q];
            updateContiguous(row, qRow, leftDistance, 0, leftEnd);
            updateContiguous(row, qRow, leftDistance, rightBegin, columns.count);
        }
    }
}

void updatePivotColumns(std::vector<unsigned int>& dist,
                        const std::vector<unsigned int>& pivot,
                        const Partition rows,
                        const Partition columns,
                        const size_t localPivotColumn,
                        const size_t pivotBegin,
                        const size_t blockSize) noexcept {
    for (size_t q = 0; q < blockSize; ++q) {
        const unsigned int* const pivotRow = pivot.data() + q * blockSize;
        for (size_t i = 0; i < rows.count; ++i) {
            const size_t globalRow = rows.begin + i;
            if (globalRow >= pivotBegin && globalRow < pivotBegin + blockSize) {
                continue;
            }

            unsigned int* const row = dist.data() + i * columns.count + localPivotColumn;
            const unsigned int leftDistance = row[q];
            for (size_t j = 0; j < blockSize; ++j) {
                const unsigned int candidate = leftDistance + pivotRow[j];
                if (candidate < row[j]) {
                    row[j] = candidate;
                }
            }
        }
    }
}

void updateRemaining(std::vector<unsigned int>& dist,
                     const std::vector<unsigned int>& pivotRows,
                     const std::vector<unsigned int>& pivotColumns,
                     const Partition rows,
                     const Partition columns,
                     const size_t pivotBegin,
                     const size_t blockSize) noexcept {
    if (rows.count == 0 || columns.count == 0) {
        return;
    }
    constexpr size_t COLUMN_TILE = 256;
    size_t leftEnd = columns.count;
    size_t rightBegin = columns.count;
    if (pivotBegin >= columns.begin && pivotBegin < columns.begin + columns.count) {
        leftEnd = pivotBegin - columns.begin;
        rightBegin = leftEnd + blockSize;
    }

    for (size_t i = 0; i < rows.count; ++i) {
        const size_t globalRow = rows.begin + i;
        if (globalRow >= pivotBegin && globalRow < pivotBegin + blockSize) {
            continue;
        }

        unsigned int* const row = dist.data() + i * columns.count;
        const unsigned int* const column = pivotColumns.data() + i * blockSize;
        const size_t rangeBegins[2] = {0, rightBegin};
        const size_t rangeEnds[2] = {leftEnd, columns.count};

        for (int range = 0; range < 2; ++range) {
            for (size_t tileBegin = rangeBegins[range]; tileBegin < rangeEnds[range];
                 tileBegin += COLUMN_TILE) {
                const size_t tileEnd = std::min(tileBegin + COLUMN_TILE, rangeEnds[range]);
                for (size_t q = 0; q < blockSize; ++q) {
                    const unsigned int* const qRow = pivotRows.data() + q * columns.count;
                    updateContiguous(row, qRow, column[q], tileBegin, tileEnd);
                }
            }
        }
    }
}

void floydWarshall(std::vector<unsigned int>& dist,
                   const size_t numNodes,
                   const Partition rows,
                   const Partition columns,
                   const int processRow,
                   const int processColumn,
                   const int processRows,
                   const int processColumns,
                   MPI_Comm rowCommunicator,
                   MPI_Comm columnCommunicator) {
    std::vector<unsigned int> pivot;
    std::vector<unsigned int> pivotRows;
    std::vector<unsigned int> pivotColumns;
    pivot.reserve(BLOCK_SIZE * BLOCK_SIZE);
    pivotRows.reserve(BLOCK_SIZE * columns.count);
    pivotColumns.reserve(rows.count * BLOCK_SIZE);

    for (size_t pivotBegin = 0; pivotBegin < numNodes;) {
        const int pivotProcessRow = ownerOf(pivotBegin, numNodes, processRows);
        const int pivotProcessColumn = ownerOf(pivotBegin, numNodes, processColumns);
        const Partition pivotRowPartition = partitionFor(numNodes, processRows, pivotProcessRow);
        const Partition pivotColumnPartition = partitionFor(numNodes, processColumns,
                                                             pivotProcessColumn);
        const size_t blockEnd = std::min({pivotBegin + BLOCK_SIZE,
                                          pivotRowPartition.begin + pivotRowPartition.count,
                                          pivotColumnPartition.begin + pivotColumnPartition.count,
                                          numNodes});
        const size_t blockSize = blockEnd - pivotBegin;
        pivot.resize(blockSize * blockSize);

        if (processRow == pivotProcessRow && processColumn == pivotProcessColumn) {
            const size_t localPivotRow = pivotBegin - rows.begin;
            const size_t localPivotColumn = pivotBegin - columns.begin;
            for (size_t i = 0; i < blockSize; ++i) {
                const unsigned int* const source = dist.data()
                                                 + (localPivotRow + i) * columns.count
                                                 + localPivotColumn;
                std::copy_n(source, blockSize, pivot.data() + i * blockSize);
            }
            closePivot(pivot, blockSize);
            for (size_t i = 0; i < blockSize; ++i) {
                unsigned int* const destination = dist.data()
                                                + (localPivotRow + i) * columns.count
                                                + localPivotColumn;
                std::copy_n(pivot.data() + i * blockSize, blockSize, destination);
            }
        }

        const int pivotCount = static_cast<int>(blockSize * blockSize);
        if (processRow == pivotProcessRow) {
            MPI_Bcast(pivot.data(), pivotCount, MPI_UNSIGNED, pivotProcessColumn,
                      rowCommunicator);
            updatePivotRows(dist, pivot, columns, pivotBegin - rows.begin,
                            pivotBegin, blockSize);
        }
        if (processColumn == pivotProcessColumn) {
            MPI_Bcast(pivot.data(), pivotCount, MPI_UNSIGNED, pivotProcessRow,
                      columnCommunicator);
            updatePivotColumns(dist, pivot, rows, columns, pivotBegin - columns.begin,
                               pivotBegin, blockSize);
        }

        pivotRows.resize(blockSize * columns.count);
        if (processRow == pivotProcessRow) {
            const size_t localPivotRow = pivotBegin - rows.begin;
            if (columns.count != 0) {
                std::copy_n(dist.data() + localPivotRow * columns.count,
                            blockSize * columns.count, pivotRows.data());
            }
        }

        pivotColumns.resize(rows.count * blockSize);
        if (processColumn == pivotProcessColumn) {
            const size_t localPivotColumn = pivotBegin - columns.begin;
            for (size_t i = 0; i < rows.count; ++i) {
                std::copy_n(dist.data() + i * columns.count + localPivotColumn,
                            blockSize, pivotColumns.data() + i * blockSize);
            }
        }

        const size_t pivotRowCount = blockSize * columns.count;
        const size_t pivotColumnCount = rows.count * blockSize;
        if (pivotRowCount > static_cast<size_t>(INT_MAX)
            || pivotColumnCount > static_cast<size_t>(INT_MAX)) {
            std::fprintf(stderr, "MPI panel exceeds the implementation's count limit\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        MPI_Request requests[2];
        MPI_Ibcast(pivotRows.data(), static_cast<int>(pivotRowCount), MPI_UNSIGNED,
                   pivotProcessRow, columnCommunicator, &requests[0]);
        MPI_Ibcast(pivotColumns.data(), static_cast<int>(pivotColumnCount), MPI_UNSIGNED,
                   pivotProcessColumn, rowCommunicator, &requests[1]);
        MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);

        updateRemaining(dist, pivotRows, pivotColumns, rows, columns,
                        pivotBegin, blockSize);
        pivotBegin = blockEnd;
    }
}

void reduceUnsignedSum(const std::vector<unsigned int>& local,
                       std::vector<unsigned int>& result,
                       const int root,
                       MPI_Comm communicator) {
    size_t offset = 0;
    while (offset < local.size()) {
        const int count = static_cast<int>(std::min(local.size() - offset,
                                                    static_cast<size_t>(INT_MAX)));
        MPI_Reduce(local.data() + offset, result.data() + offset, count, MPI_UNSIGNED,
                   MPI_SUM, root, communicator);
        offset += static_cast<size_t>(count);
    }
}

bool validateResult(const std::vector<unsigned int>& dist,
                    const size_t numNodes,
                    const Partition rows,
                    const Partition columns,
                    const int rank,
                    MPI_Comm communicator) {
    unsigned long long firstBadDiagonal = static_cast<unsigned long long>(numNodes);
    for (size_t i = 0; i < rows.count; ++i) {
        const size_t globalRow = rows.begin + i;
        if (globalRow >= columns.begin && globalRow < columns.begin + columns.count
            && dist[i * columns.count + globalRow - columns.begin] != 0) {
            firstBadDiagonal = std::min(firstBadDiagonal,
                                        static_cast<unsigned long long>(globalRow));
        }
    }

    unsigned long long globalBadDiagonal = static_cast<unsigned long long>(numNodes);
    MPI_Reduce(&firstBadDiagonal, &globalBadDiagonal, 1, MPI_UNSIGNED_LONG_LONG,
               MPI_MIN, 0, communicator);

    const size_t sampleSize = std::min(numNodes, static_cast<size_t>(10));
    std::vector<unsigned int> localSamples(2 * sampleSize * numNodes, 0);
    for (size_t i = 0; i < sampleSize; ++i) {
        if (columns.count != 0 && i >= rows.begin && i < rows.begin + rows.count) {
            const unsigned int* const source = dist.data() + (i - rows.begin) * columns.count;
            std::copy_n(source, columns.count,
                        localSamples.data() + i * numNodes + columns.begin);
        }
        if (i >= columns.begin && i < columns.begin + columns.count) {
            const size_t localColumn = i - columns.begin;
            unsigned int* const destination = localSamples.data()
                                            + (sampleSize + i) * numNodes + rows.begin;
            for (size_t k = 0; k < rows.count; ++k) {
                destination[k] = dist[k * columns.count + localColumn];
            }
        }
    }

    std::vector<unsigned int> samples(rank == 0 ? localSamples.size() : 0);
    // MPI requires a valid receive pointer even on some older implementations.
    if (rank != 0) {
        samples.resize(localSamples.size());
    }
    reduceUnsignedSum(localSamples, samples, 0, communicator);

    int valid = 1;
    if (rank == 0) {
        if (globalBadDiagonal != numNodes) {
            std::printf("Validation failed: diagonal element [%llu,%llu] is not zero\n",
                        globalBadDiagonal, globalBadDiagonal);
            valid = 0;
        }

        for (size_t i = 0; valid != 0 && i < sampleSize; ++i) {
            const unsigned int* const sourceRow = samples.data() + i * numNodes;
            for (size_t j = 0; valid != 0 && j < sampleSize; ++j) {
                const unsigned int distIJ = sourceRow[j];
                const unsigned int* const destinationColumn = samples.data()
                                                                + (sampleSize + j) * numNodes;
                for (size_t k = 0; k < numNodes; ++k) {
                    const unsigned int distIK = sourceRow[k];
                    const unsigned int distKJ = destinationColumn[k];
                    if (distIK < INF && distKJ < INF && distIK + distKJ < distIJ) {
                        std::printf("Validation failed: triangle inequality violated at "
                                    "[%zu,%zu,%zu]\n", i, j, k);
                        valid = 0;
                        break;
                    }
                }
            }
        }
    }

    MPI_Bcast(&valid, 1, MPI_INT, 0, communicator);
    return valid != 0;
}

void copyLocalToSlab(const std::vector<unsigned int>& local,
                     std::vector<unsigned int>& slab,
                     const size_t rows,
                     const size_t columns,
                     const size_t columnBegin,
                     const size_t numNodes) {
    for (size_t i = 0; i < rows; ++i) {
        std::copy_n(local.data() + i * columns, columns,
                    slab.data() + i * numNodes + columnBegin);
    }
}

void printDistributedResults(const std::vector<unsigned int>& dist,
                             const size_t numNodes,
                             const Partition rows,
                             const Partition columns,
                             const int rank,
                             const int processRows,
                             const int processColumns,
                             MPI_Comm communicator) {
    constexpr int RESULT_TAG = 4107;

    if (rank != 0) {
        if (rows.count != 0 && columns.count != 0) {
            const size_t rowsPerMessage = std::max(
                static_cast<size_t>(1), static_cast<size_t>(INT_MAX) / columns.count);
            for (size_t rowOffset = 0; rowOffset < rows.count; rowOffset += rowsPerMessage) {
                const size_t messageRows = std::min(rowsPerMessage, rows.count - rowOffset);
                const int count = static_cast<int>(messageRows * columns.count);
                MPI_Send(dist.data() + rowOffset * columns.count, count, MPI_UNSIGNED,
                         0, RESULT_TAG, communicator);
            }
        }
        return;
    }

    std::printf("=== RESULTS ===\n");
    std::printf("Name: DistanceMatrix\n");
    const size_t elements = numNodes * numNodes;
    std::printf("Elements: %zu\n", elements);
    if (elements == 0) {
        std::printf("=== END RESULTS ===\n");
        return;
    }

    hash resultHash;
    unsigned int minimum = std::numeric_limits<unsigned int>::max();
    unsigned int maximum = 0;
    const size_t sampleIndices[5] = {0, elements / 4, elements / 2,
                                     3 * elements / 4, elements - 1};
    unsigned int sampleValues[5] = {};

    for (int processRow = 0; processRow < processRows; ++processRow) {
        const Partition slabRows = partitionFor(numNodes, processRows, processRow);
        if (slabRows.count == 0) {
            continue;
        }
        std::vector<unsigned int> slab(slabRows.count * numNodes);

        for (int processColumn = 0; processColumn < processColumns; ++processColumn) {
            const Partition slabColumns = partitionFor(numNodes, processColumns,
                                                        processColumn);
            if (slabColumns.count == 0) {
                continue;
            }
            const int sourceRank = processRow * processColumns + processColumn;
            if (sourceRank == 0) {
                copyLocalToSlab(dist, slab, slabRows.count, slabColumns.count,
                                slabColumns.begin, numNodes);
                continue;
            }

            const size_t rowsPerMessage = std::max(
                static_cast<size_t>(1), static_cast<size_t>(INT_MAX) / slabColumns.count);
            for (size_t rowOffset = 0; rowOffset < slabRows.count;
                 rowOffset += rowsPerMessage) {
                const size_t messageRows = std::min(rowsPerMessage,
                                                    slabRows.count - rowOffset);
                MPI_Datatype slabType;
                MPI_Type_vector(static_cast<int>(messageRows),
                                static_cast<int>(slabColumns.count),
                                static_cast<int>(numNodes), MPI_UNSIGNED, &slabType);
                MPI_Type_commit(&slabType);
                MPI_Recv(slab.data() + rowOffset * numNodes + slabColumns.begin,
                         1, slabType, sourceRank, RESULT_TAG, communicator,
                         MPI_STATUS_IGNORE);
                MPI_Type_free(&slabType);
            }
        }

        const size_t globalOffset = slabRows.begin * numNodes;
        for (size_t i = 0; i < slab.size(); ++i) {
            const unsigned int value = slab[i];
            resultHash.add(value);
            minimum = std::min(minimum, value);
            maximum = std::max(maximum, value);
        }
        for (int sample = 0; sample < 5; ++sample) {
            if (sampleIndices[sample] >= globalOffset
                && sampleIndices[sample] < globalOffset + slab.size()) {
                sampleValues[sample] = slab[sampleIndices[sample] - globalOffset];
            }
        }
    }

    std::printf("Min: %lld\n", static_cast<long long>(minimum));
    std::printf("Max: %lld\n", static_cast<long long>(maximum));
    for (int sample = 0; sample < 5; ++sample) {
        std::printf("Sample[%zu]: %lld\n", sampleIndices[sample],
                    static_cast<long long>(sampleValues[sample]));
    }
    std::printf("Hash: %016llx\n",
                static_cast<unsigned long long>(resultHash.get()));
    std::printf("=== END RESULTS ===\n");
}

void printUsage(const char* programName) {
    std::printf("Usage: %s [options]\n", programName);
    std::printf("Options:\n");
    std::printf("  -n <num>     Number of nodes in the graph (default: 512)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;
    int parseStatus = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numNodes = static_cast<size_t>(std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            parseStatus = 2;
        } else {
            if (rank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
            }
            parseStatus = 1;
        }
    }

    if (parseStatus != 0) {
        if (rank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return parseStatus == 2 ? 0 : 1;
    }

    if (numNodes > static_cast<size_t>(INT_MAX)
        || (numNodes != 0 && numNodes > std::numeric_limits<size_t>::max() / numNodes)) {
        if (rank == 0) {
            std::fprintf(stderr, "Number of nodes is too large for this MPI implementation\n");
        }
        MPI_Finalize();
        return 1;
    }

    int dimensions[2] = {0, 0};
    MPI_Dims_create(worldSize, 2, dimensions);
    const int processRow = rank / dimensions[1];
    const int processColumn = rank % dimensions[1];
    const Partition rows = partitionFor(numNodes, dimensions[0], processRow);
    const Partition columns = partitionFor(numNodes, dimensions[1], processColumn);

    MPI_Comm rowCommunicator;
    MPI_Comm columnCommunicator;
    MPI_Comm_split(MPI_COMM_WORLD, processRow, processColumn, &rowCommunicator);
    MPI_Comm_split(MPI_COMM_WORLD, processColumn, processRow, &columnCommunicator);

    if (rank == 0) {
        std::printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        std::printf("Number of nodes: %zu\n", numNodes);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Initializing graph...\n");
    }

    std::vector<unsigned int> dist(rows.count * columns.count);
    initializeDistanceMatrix(dist, numNodes, rows, columns, 1, MAX_DISTANCE);

    if (rank == 0) {
        std::printf("Computing shortest paths...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    floydWarshall(dist, numNodes, rows, columns, processRow, processColumn,
                  dimensions[0], dimensions[1], rowCommunicator, columnCommunicator);
    const double localElapsed = MPI_Wtime() - start;

    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        const long durationMilliseconds = static_cast<long>(elapsed * 1000.0);
        std::printf("Computation time: %ld ms\n", durationMilliseconds);
        const double operations = static_cast<double>(numNodes)
                                * static_cast<double>(numNodes)
                                * static_cast<double>(numNodes);
        const double gops = operations / elapsed / 1e9;
        std::printf("Performance: %.3f GOPS\n", gops);
    }

    if (printResults) {
        printDistributedResults(dist, numNodes, rows, columns, rank,
                                dimensions[0], dimensions[1], MPI_COMM_WORLD);
    }

    int exitStatus = 0;
    if (validate) {
        if (rank == 0) {
            std::printf("Validating result...\n");
        }
        const bool valid = validateResult(dist, numNodes, rows, columns,
                                          rank, MPI_COMM_WORLD);
        if (rank == 0) {
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        exitStatus = valid ? 0 : 1;
    }

    MPI_Comm_free(&rowCommunicator);
    MPI_Comm_free(&columnCommunicator);
    MPI_Finalize();
    return exitStatus;
}
