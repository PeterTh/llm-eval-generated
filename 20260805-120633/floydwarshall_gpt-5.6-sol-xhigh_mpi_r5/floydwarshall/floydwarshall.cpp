#include <mpi.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;
constexpr int DISTRIBUTE_TAG = 1101;
constexpr int GATHER_TAG = 1102;
constexpr size_t TILE_SIZE = 64;

// Index calculation for a row-major flattened matrix.  This retains the
// original benchmark's indexing convention.
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

struct Range {
    size_t begin;
    size_t count;
};

struct MatrixBlock {
    Range rows;
    Range columns;
};

Range partitionRange(const size_t extent, const int parts, const int coordinate) {
    const size_t partCount = static_cast<size_t>(parts);
    const size_t partCoordinate = static_cast<size_t>(coordinate);
    const size_t base = extent / partCount;
    const size_t remainder = extent % partCount;

    return {
        partCoordinate * base + std::min(partCoordinate, remainder),
        base + (partCoordinate < remainder ? 1U : 0U)
    };
}

int ownerOfIndex(const size_t index, const size_t extent, const int parts) {
    const size_t partCount = static_cast<size_t>(parts);
    const size_t base = extent / partCount;
    const size_t remainder = extent % partCount;
    const size_t largerPartitionElements = (base + 1U) * remainder;

    if (index < largerPartitionElements) {
        return static_cast<int>(index / (base + 1U));
    }
    return static_cast<int>(remainder + (index - largerPartitionElements) / base);
}

MatrixBlock blockForCoordinates(const size_t numNodes, const int gridDimensions[2],
                                const int coordinates[2]) {
    return {
        partitionRange(numNodes, gridDimensions[0], coordinates[0]),
        partitionRange(numNodes, gridDimensions[1], coordinates[1])
    };
}

size_t checkedProduct(const size_t lhs, const size_t rhs) {
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
        throw std::length_error("matrix size overflows size_t");
    }
    return lhs * rhs;
}

MPI_Datatype makeRectangleType(const size_t rows, const size_t columns,
                               const size_t rowStride) {
    MPI_Datatype datatype = MPI_DATATYPE_NULL;
    MPI_Type_vector(static_cast<int>(rows), static_cast<int>(columns),
                    static_cast<int>(rowStride), MPI_UNSIGNED, &datatype);
    MPI_Type_commit(&datatype);
    return datatype;
}

void initializeAndDistributeMatrix(std::vector<unsigned int>& localDist,
                                   const size_t numNodes,
                                   const MatrixBlock& localBlock,
                                   const int gridDimensions[2], const int cartRank,
                                   MPI_Comm cartCommunicator) {
    if (cartRank == 0) {
        unsigned int seed = 42;
        const double range = static_cast<double>(MAX_DISTANCE);

        // Generate one process-row slab at a time.  Keeping the original
        // rand_r call order preserves bit-for-bit input compatibility while
        // limiting rank zero's temporary storage to about N^2/gridRows.
        for (int processRow = 0; processRow < gridDimensions[0]; ++processRow) {
            const Range rows =
                partitionRange(numNodes, gridDimensions[0], processRow);
            if (rows.count == 0) {
                continue;
            }
            std::vector<unsigned int> slab(checkedProduct(rows.count, numNodes));
            for (unsigned int& distance : slab) {
                distance = 1U +
                           static_cast<unsigned int>(range * rand_r(&seed) /
                                                     static_cast<double>(RAND_MAX));
            }
            for (size_t localRow = 0; localRow < rows.count; ++localRow) {
                const size_t globalRow = rows.begin + localRow;
                slab[localRow * numNodes + globalRow] = 0;
            }

            std::vector<MPI_Request> requests;
            std::vector<MPI_Datatype> datatypes;
            requests.reserve(static_cast<size_t>(gridDimensions[1]));
            datatypes.reserve(static_cast<size_t>(gridDimensions[1]));

            for (int processColumn = 0; processColumn < gridDimensions[1];
                 ++processColumn) {
                const Range columns =
                    partitionRange(numNodes, gridDimensions[1], processColumn);
                if (columns.count == 0) {
                    continue;
                }
                const int coordinates[2] = {processRow, processColumn};
                int destination = 0;
                MPI_Cart_rank(cartCommunicator, coordinates, &destination);

                if (destination == 0) {
                    for (size_t row = 0; row < rows.count; ++row) {
                        const unsigned int* source =
                            slab.data() + row * numNodes + columns.begin;
                        unsigned int* localRow =
                            localDist.data() + row * localBlock.columns.count;
                        std::copy_n(source, columns.count, localRow);
                    }
                    continue;
                }

                datatypes.push_back(
                    makeRectangleType(rows.count, columns.count, numNodes));
                requests.push_back(MPI_REQUEST_NULL);
                MPI_Isend(slab.data() + columns.begin, 1, datatypes.back(),
                          destination, DISTRIBUTE_TAG, cartCommunicator,
                          &requests.back());
            }

            if (!requests.empty()) {
                MPI_Waitall(static_cast<int>(requests.size()), requests.data(),
                            MPI_STATUSES_IGNORE);
            }
            for (MPI_Datatype& datatype : datatypes) {
                MPI_Type_free(&datatype);
            }
        }
    } else if (localBlock.rows.count != 0 && localBlock.columns.count != 0) {
        MPI_Datatype localType =
            makeRectangleType(localBlock.rows.count, localBlock.columns.count,
                              localBlock.columns.count);
        MPI_Recv(localDist.data(), 1, localType, 0, DISTRIBUTE_TAG, cartCommunicator,
                 MPI_STATUS_IGNORE);
        MPI_Type_free(&localType);
    }
}

void floydWarshallDistributed(std::vector<unsigned int>& localDist,
                              const MatrixBlock& localBlock, const size_t numNodes,
                              const int gridDimensions[2], const int coordinates[2],
                              MPI_Comm rowCommunicator, MPI_Comm columnCommunicator) {
    const size_t localRows = localBlock.rows.count;
    const size_t localColumns = localBlock.columns.count;
    std::vector<unsigned int> diagonalTile(TILE_SIZE * TILE_SIZE);
    std::vector<unsigned int> pivotRows(
        checkedProduct(TILE_SIZE, localColumns));
    std::vector<unsigned int> pivotColumns(
        checkedProduct(localRows, TILE_SIZE));
    unsigned int zeroCountBuffer = 0;

    for (size_t tileBegin = 0; tileBegin < numNodes;) {
        const int pivotProcessRow =
            ownerOfIndex(tileBegin, numNodes, gridDimensions[0]);
        const int pivotProcessColumn =
            ownerOfIndex(tileBegin, numNodes, gridDimensions[1]);
        const Range pivotRowRange =
            partitionRange(numNodes, gridDimensions[0], pivotProcessRow);
        const Range pivotColumnRange =
            partitionRange(numNodes, gridDimensions[1], pivotProcessColumn);
        const size_t tileEnd =
            std::min({numNodes, tileBegin + TILE_SIZE,
                      pivotRowRange.begin + pivotRowRange.count,
                      pivotColumnRange.begin + pivotColumnRange.count});
        const size_t tileSize = tileEnd - tileBegin;
        const size_t diagonalElements = tileSize * tileSize;

        // Phase 1: the rank at the pivot process-row/process-column
        // intersection closes the diagonal tile.
        if (coordinates[0] == pivotProcessRow &&
            coordinates[1] == pivotProcessColumn) {
            const size_t localTileRow = tileBegin - localBlock.rows.begin;
            const size_t localTileColumn = tileBegin - localBlock.columns.begin;
            for (size_t row = 0; row < tileSize; ++row) {
                const unsigned int* source =
                    localDist.data() + (localTileRow + row) * localColumns +
                    localTileColumn;
                std::copy_n(source, tileSize,
                            diagonalTile.data() + row * tileSize);
            }

            for (size_t k = 0; k < tileSize; ++k) {
                const unsigned int* pivot = diagonalTile.data() + k * tileSize;
                for (size_t row = 0; row < tileSize; ++row) {
                    unsigned int* localRow =
                        diagonalTile.data() + row * tileSize;
                    const unsigned int distIK = localRow[k];
                    for (size_t column = 0; column < tileSize; ++column) {
                        localRow[column] =
                            std::min(localRow[column], distIK + pivot[column]);
                    }
                }
            }

            for (size_t row = 0; row < tileSize; ++row) {
                unsigned int* destination =
                    localDist.data() + (localTileRow + row) * localColumns +
                    localTileColumn;
                std::copy_n(diagonalTile.data() + row * tileSize, tileSize,
                            destination);
            }
        }

        // Only the pivot process row and column need the diagonal tile.  The
        // two broadcasts use independent communicators and progress together.
        MPI_Request diagonalRequests[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};
        int diagonalRequestCount = 0;
        if (coordinates[0] == pivotProcessRow) {
            MPI_Ibcast(diagonalTile.data(), static_cast<int>(diagonalElements),
                       MPI_UNSIGNED, pivotProcessColumn, rowCommunicator,
                       &diagonalRequests[diagonalRequestCount++]);
        }
        if (coordinates[1] == pivotProcessColumn) {
            MPI_Ibcast(diagonalTile.data(), static_cast<int>(diagonalElements),
                       MPI_UNSIGNED, pivotProcessRow, columnCommunicator,
                       &diagonalRequests[diagonalRequestCount++]);
        }
        if (diagonalRequestCount != 0) {
            MPI_Waitall(diagonalRequestCount, diagonalRequests, MPI_STATUSES_IGNORE);
        }

        // Phase 2a: close every local segment of the pivot block row.
        if (coordinates[0] == pivotProcessRow && localColumns != 0) {
            const size_t localTileRow = tileBegin - localBlock.rows.begin;
            size_t excludedColumnBegin = localColumns;
            size_t excludedColumnEnd = localColumns;
            if (coordinates[1] == pivotProcessColumn) {
                excludedColumnBegin = tileBegin - localBlock.columns.begin;
                excludedColumnEnd = excludedColumnBegin + tileSize;
            }
            for (size_t k = 0; k < tileSize; ++k) {
                const unsigned int* pivot =
                    localDist.data() + (localTileRow + k) * localColumns;
                for (size_t row = 0; row < tileSize; ++row) {
                    unsigned int* localRow =
                        localDist.data() + (localTileRow + row) * localColumns;
                    const unsigned int distIK = diagonalTile[row * tileSize + k];
                    for (size_t column = 0; column < excludedColumnBegin; ++column) {
                        localRow[column] =
                            std::min(localRow[column], distIK + pivot[column]);
                    }
                    for (size_t column = excludedColumnEnd; column < localColumns;
                         ++column) {
                        localRow[column] =
                            std::min(localRow[column], distIK + pivot[column]);
                    }
                }
            }
        }

        // Phase 2b: close every local segment of the pivot block column.
        if (coordinates[1] == pivotProcessColumn && localRows != 0) {
            const size_t localTileColumn = tileBegin - localBlock.columns.begin;
            for (size_t k = 0; k < tileSize; ++k) {
                const unsigned int* diagonalPivot =
                    diagonalTile.data() + k * tileSize;
                for (size_t row = 0; row < localRows; ++row) {
                    const size_t globalRow = localBlock.rows.begin + row;
                    if (globalRow >= tileBegin && globalRow < tileEnd) {
                        continue;
                    }
                    unsigned int* localRow = localDist.data() + row * localColumns;
                    const unsigned int distIK = localRow[localTileColumn + k];
                    for (size_t column = 0; column < tileSize; ++column) {
                        localRow[localTileColumn + column] =
                            std::min(localRow[localTileColumn + column],
                                     distIK + diagonalPivot[column]);
                    }
                }
            }
        }

        // Pack the now-closed panels at their roots.
        if (coordinates[0] == pivotProcessRow && localColumns != 0) {
            const size_t localTileRow = tileBegin - localBlock.rows.begin;
            for (size_t row = 0; row < tileSize; ++row) {
                const unsigned int* source =
                    localDist.data() + (localTileRow + row) * localColumns;
                std::copy_n(source, localColumns,
                            pivotRows.data() + row * localColumns);
            }
        }
        if (coordinates[1] == pivotProcessColumn && localRows != 0) {
            const size_t localTileColumn = tileBegin - localBlock.columns.begin;
            for (size_t row = 0; row < localRows; ++row) {
                const unsigned int* source =
                    localDist.data() + row * localColumns + localTileColumn;
                std::copy_n(source, tileSize,
                            pivotColumns.data() + row * tileSize);
            }
        }

        // Phase 3: broadcast the two panels concurrently and update every
        // local block.  Tiling retains each local row in cache across all k in
        // the tile and reduces collective latency by up to TILE_SIZE.
        const size_t pivotRowElements = tileSize * localColumns;
        const size_t pivotColumnElements = tileSize * localRows;
        if (pivotRowElements > static_cast<size_t>(INT_MAX) ||
            pivotColumnElements > static_cast<size_t>(INT_MAX)) {
            throw std::length_error("MPI panel size exceeds the MPI count limit");
        }
        unsigned int* pivotRowBuffer =
            pivotRowElements == 0 ? &zeroCountBuffer : pivotRows.data();
        unsigned int* pivotColumnBuffer =
            pivotColumnElements == 0 ? &zeroCountBuffer : pivotColumns.data();
        MPI_Request panelRequests[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};
        MPI_Ibcast(pivotRowBuffer, static_cast<int>(pivotRowElements), MPI_UNSIGNED,
                   pivotProcessRow, columnCommunicator, &panelRequests[0]);
        MPI_Ibcast(pivotColumnBuffer, static_cast<int>(pivotColumnElements), MPI_UNSIGNED,
                   pivotProcessColumn, rowCommunicator, &panelRequests[1]);
        MPI_Waitall(2, panelRequests, MPI_STATUSES_IGNORE);

        if (localColumns != 0) {
            size_t excludedColumnBegin = localColumns;
            size_t excludedColumnEnd = localColumns;
            if (coordinates[1] == pivotProcessColumn) {
                excludedColumnBegin = tileBegin - localBlock.columns.begin;
                excludedColumnEnd = excludedColumnBegin + tileSize;
            }
            for (size_t row = 0; row < localRows; ++row) {
                const size_t globalRow = localBlock.rows.begin + row;
                if (globalRow >= tileBegin && globalRow < tileEnd) {
                    continue;
                }
                unsigned int* localRow = localDist.data() + row * localColumns;
                const unsigned int* columnPanel =
                    pivotColumns.data() + row * tileSize;
                for (size_t k = 0; k < tileSize; ++k) {
                    const unsigned int distIK = columnPanel[k];
                    const unsigned int* rowPanel =
                        pivotRows.data() + k * localColumns;

                    // The original path matrix was never observed, so omitting
                    // its dead stores halves storage and leaves this hot loop
                    // branch-free and SIMD-vectorizable.
                    for (size_t column = 0; column < excludedColumnBegin; ++column) {
                        localRow[column] =
                            std::min(localRow[column], distIK + rowPanel[column]);
                    }
                    for (size_t column = excludedColumnEnd; column < localColumns;
                         ++column) {
                        localRow[column] =
                            std::min(localRow[column], distIK + rowPanel[column]);
                    }
                }
            }
        }

        tileBegin = tileEnd;
    }
}

void gatherMatrix(const std::vector<unsigned int>& localDist,
                  std::vector<unsigned int>& globalDist, const size_t numNodes,
                  const MatrixBlock& localBlock, const int gridDimensions[2],
                  const int cartRank, const int cartSize, MPI_Comm cartCommunicator) {
    if (cartRank == 0) {
        std::vector<MPI_Request> requests;
        std::vector<MPI_Datatype> datatypes;
        requests.reserve(static_cast<size_t>(cartSize - 1));
        datatypes.reserve(static_cast<size_t>(cartSize - 1));

        for (int source = 0; source < cartSize; ++source) {
            int coordinates[2] = {0, 0};
            MPI_Cart_coords(cartCommunicator, source, 2, coordinates);
            const MatrixBlock block =
                blockForCoordinates(numNodes, gridDimensions, coordinates);

            if (block.rows.count == 0 || block.columns.count == 0) {
                continue;
            }

            if (source == 0) {
                for (size_t row = 0; row < localBlock.rows.count; ++row) {
                    const unsigned int* sourceRow =
                        localDist.data() + row * localBlock.columns.count;
                    unsigned int* destinationRow =
                        globalDist.data() + (localBlock.rows.begin + row) * numNodes +
                        localBlock.columns.begin;
                    std::copy_n(sourceRow, localBlock.columns.count, destinationRow);
                }
                continue;
            }

            datatypes.push_back(makeRectangleType(block.rows.count, block.columns.count,
                                                   numNodes));
            requests.push_back(MPI_REQUEST_NULL);
            const size_t offset = block.rows.begin * numNodes + block.columns.begin;
            MPI_Irecv(globalDist.data() + offset, 1, datatypes.back(), source,
                      GATHER_TAG, cartCommunicator, &requests.back());
        }

        if (!requests.empty()) {
            MPI_Waitall(static_cast<int>(requests.size()), requests.data(),
                        MPI_STATUSES_IGNORE);
        }
        for (MPI_Datatype& datatype : datatypes) {
            MPI_Type_free(&datatype);
        }
    } else if (localBlock.rows.count != 0 && localBlock.columns.count != 0) {
        MPI_Datatype localType =
            makeRectangleType(localBlock.rows.count, localBlock.columns.count,
                              localBlock.columns.count);
        MPI_Send(localDist.data(), 1, localType, 0, GATHER_TAG, cartCommunicator);
        MPI_Type_free(&localType);
    }
}

bool validateResultDistributed(const std::vector<unsigned int>& localDist,
                               const MatrixBlock& localBlock,
                               const size_t numNodes, const int cartRank,
                               MPI_Comm cartCommunicator) {
    unsigned long long firstLocalDiagonalFailure = ULLONG_MAX;
    const size_t diagonalBegin =
        std::max(localBlock.rows.begin, localBlock.columns.begin);
    const size_t diagonalEnd =
        std::min(localBlock.rows.begin + localBlock.rows.count,
                 localBlock.columns.begin + localBlock.columns.count);
    for (size_t i = diagonalBegin; i < diagonalEnd; ++i) {
        const size_t localRow = i - localBlock.rows.begin;
        const size_t localColumn = i - localBlock.columns.begin;
        if (localDist[localRow * localBlock.columns.count + localColumn] != 0) {
            firstLocalDiagonalFailure = static_cast<unsigned long long>(i);
            break;
        }
    }

    unsigned long long firstDiagonalFailure = ULLONG_MAX;
    MPI_Allreduce(&firstLocalDiagonalFailure, &firstDiagonalFailure, 1,
                  MPI_UNSIGNED_LONG_LONG, MPI_MIN, cartCommunicator);
    if (firstDiagonalFailure != ULLONG_MAX) {
        if (cartRank == 0) {
            const size_t i = static_cast<size_t>(firstDiagonalFailure);
            std::printf("Validation failed: diagonal element [%zu,%zu] is not zero\n",
                        i, i);
        }
        return false;
    }

    // The original validator samples the first ten rows and columns.  Reduce
    // just those stripes to rank zero instead of gathering the N-by-N matrix.
    const size_t sampleSize = std::min(numNodes, static_cast<size_t>(10));
    const size_t sampleElements = checkedProduct(sampleSize, numNodes);
    if (sampleElements == 0) {
        return true;
    }
    if (sampleElements > static_cast<size_t>(INT_MAX)) {
        throw std::length_error("validation sample exceeds the MPI count limit");
    }

    std::vector<unsigned int> localSampleRows(sampleElements, INF);
    std::vector<unsigned int> localSampleColumns(sampleElements, INF);
    const size_t sampledLocalRows =
        localBlock.rows.begin < sampleSize
            ? std::min(localBlock.rows.count, sampleSize - localBlock.rows.begin)
            : 0;
    const size_t sampledLocalColumns =
        localBlock.columns.begin < sampleSize
            ? std::min(localBlock.columns.count,
                       sampleSize - localBlock.columns.begin)
            : 0;

    if (localBlock.columns.count != 0) {
        for (size_t localRow = 0; localRow < sampledLocalRows; ++localRow) {
            const size_t globalRow = localBlock.rows.begin + localRow;
            const unsigned int* source =
                localDist.data() + localRow * localBlock.columns.count;
            unsigned int* destination =
                localSampleRows.data() + globalRow * numNodes +
                localBlock.columns.begin;
            std::copy_n(source, localBlock.columns.count, destination);
        }
    }
    if (sampledLocalColumns != 0) {
        for (size_t localRow = 0; localRow < localBlock.rows.count; ++localRow) {
            const size_t globalRow = localBlock.rows.begin + localRow;
            const unsigned int* source =
                localDist.data() + localRow * localBlock.columns.count;
            unsigned int* destination =
                localSampleColumns.data() + globalRow * sampleSize +
                localBlock.columns.begin;
            std::copy_n(source, sampledLocalColumns, destination);
        }
    }

    std::vector<unsigned int> sampleRows;
    std::vector<unsigned int> sampleColumns;
    if (cartRank == 0) {
        sampleRows.resize(sampleElements);
        sampleColumns.resize(sampleElements);
    }
    MPI_Reduce(localSampleRows.data(),
               cartRank == 0 ? sampleRows.data() : nullptr,
               static_cast<int>(sampleElements), MPI_UNSIGNED, MPI_MIN, 0,
               cartCommunicator);
    MPI_Reduce(localSampleColumns.data(),
               cartRank == 0 ? sampleColumns.data() : nullptr,
               static_cast<int>(sampleElements), MPI_UNSIGNED, MPI_MIN, 0,
               cartCommunicator);

    if (cartRank != 0) {
        return true;
    }
    for (size_t i = 0; i < sampleSize; ++i) {
        for (size_t j = 0; j < sampleSize; ++j) {
            for (size_t k = 0; k < numNodes; ++k) {
                const unsigned int distIJ = sampleRows[i * numNodes + j];
                const unsigned int distIK = sampleRows[i * numNodes + k];
                const unsigned int distKJ = sampleColumns[k * sampleSize + j];

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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int worldRank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numNodes = static_cast<size_t>(std::atoi(argv[++i]));
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
            if (worldRank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (numNodes > static_cast<size_t>(INT_MAX)) {
        if (worldRank == 0) {
            std::fprintf(stderr, "Number of nodes exceeds the MPI count limit\n");
        }
        MPI_Finalize();
        return 1;
    }

    int gridDimensions[2] = {0, 0};
    MPI_Dims_create(worldSize, 2, gridDimensions);
    const int periods[2] = {0, 0};
    MPI_Comm cartCommunicator = MPI_COMM_NULL;
    MPI_Cart_create(MPI_COMM_WORLD, 2, gridDimensions, periods, 0, &cartCommunicator);

    int cartRank = 0;
    int cartSize = 1;
    int coordinates[2] = {0, 0};
    MPI_Comm_rank(cartCommunicator, &cartRank);
    MPI_Comm_size(cartCommunicator, &cartSize);
    MPI_Cart_coords(cartCommunicator, cartRank, 2, coordinates);

    MPI_Comm rowCommunicator = MPI_COMM_NULL;
    MPI_Comm columnCommunicator = MPI_COMM_NULL;
    MPI_Comm_split(cartCommunicator, coordinates[0], coordinates[1], &rowCommunicator);
    MPI_Comm_split(cartCommunicator, coordinates[1], coordinates[0], &columnCommunicator);

    int exitCode = 0;
    try {
        const MatrixBlock localBlock =
            blockForCoordinates(numNodes, gridDimensions, coordinates);
        const size_t localElements =
            checkedProduct(localBlock.rows.count, localBlock.columns.count);
        const size_t globalElements = checkedProduct(numNodes, numNodes);
        std::vector<unsigned int> localDist(localElements);
        std::vector<unsigned int> globalDist;

        if (cartRank == 0) {
            std::printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
            std::printf("Number of nodes: %zu\n", numNodes);
            std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
            std::printf("Initializing graph...\n");
        }

        initializeAndDistributeMatrix(localDist, numNodes, localBlock, gridDimensions,
                                      cartRank, cartCommunicator);

        if (cartRank == 0) {
            std::printf("Computing shortest paths...\n");
        }
        MPI_Barrier(cartCommunicator);
        const double start = MPI_Wtime();

        floydWarshallDistributed(localDist, localBlock, numNodes, gridDimensions,
                                 coordinates, rowCommunicator, columnCommunicator);

        const double localDuration = MPI_Wtime() - start;
        double duration = 0.0;
        MPI_Reduce(&localDuration, &duration, 1, MPI_DOUBLE, MPI_MAX, 0,
                   cartCommunicator);

        if (cartRank == 0) {
            const long durationMilliseconds = static_cast<long>(duration * 1000.0);
            std::printf("Computation time: %ld ms\n", durationMilliseconds);

            const double operations = static_cast<double>(numNodes) *
                                      static_cast<double>(numNodes) *
                                      static_cast<double>(numNodes);
            const double gops = duration > 0.0 ? operations / duration / 1.0e9 : 0.0;
            std::printf("Performance: %.3f GOPS\n", gops);
        }

        if (printResults) {
            if (cartRank == 0) {
                globalDist.resize(globalElements);
            }
            gatherMatrix(localDist, globalDist, numNodes, localBlock, gridDimensions,
                         cartRank, cartSize, cartCommunicator);
        }

        if (cartRank == 0 && printResults) {
            print_results_int(globalDist, "DistanceMatrix");
        }

        if (validate) {
            if (cartRank == 0) {
                std::printf("Validating result...\n");
            }
            const bool valid = validateResultDistributed(
                localDist, localBlock, numNodes, cartRank, cartCommunicator);
            if (cartRank == 0) {
                if (valid) {
                    std::printf("Validation: PASSED\n");
                } else {
                    std::printf("Validation: FAILED\n");
                    exitCode = 1;
                }
            }
            MPI_Bcast(&exitCode, 1, MPI_INT, 0, cartCommunicator);
        }
    } catch (const std::exception& error) {
        if (cartRank == 0) {
            std::fprintf(stderr, "Error: %s\n", error.what());
        }
        MPI_Abort(cartCommunicator, 1);
        exitCode = 1;
    }

    MPI_Comm_free(&rowCommunicator);
    MPI_Comm_free(&columnCommunicator);
    MPI_Comm_free(&cartCommunicator);
    MPI_Finalize();
    return exitCode;
}
