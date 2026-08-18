#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

// The original benchmark uses idx2(destination, source), so distances are
// physically stored as contiguous source rows.
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t numNodes) noexcept {
    return j * numNodes + i;
}

void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes,
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;

    for (size_t i = 0; i < numNodes * numNodes; ++i) {
        dist[i] = rangeMin +
                  static_cast<unsigned int>(range * rand_r(&seed) / static_cast<double>(RAND_MAX));
    }

    for (size_t i = 0; i < numNodes; ++i) {
        dist[idx2(i, i, numNodes)] = 0;
    }
}

std::vector<size_t> makeBlockOffsets(const size_t total, const int parts) {
    std::vector<size_t> offsets(static_cast<size_t>(parts) + 1, 0);
    const size_t base = total / static_cast<size_t>(parts);
    const size_t remainder = total % static_cast<size_t>(parts);

    for (int part = 0; part < parts; ++part) {
        offsets[static_cast<size_t>(part) + 1] =
            offsets[static_cast<size_t>(part)] + base +
            (static_cast<size_t>(part) < remainder ? 1 : 0);
    }
    return offsets;
}

int ownerOf(const std::vector<size_t>& offsets, const size_t index) {
    return static_cast<int>(std::upper_bound(offsets.begin(), offsets.end(), index) -
                            offsets.begin()) -
           1;
}

void sendUnsigned(const unsigned int* data, size_t count, const int destination, const int tag,
                  MPI_Comm communicator) {
    constexpr size_t maxCount = static_cast<size_t>(std::numeric_limits<int>::max());
    size_t offset = 0;
    while (offset < count) {
        const size_t chunk = std::min(count - offset, maxCount);
        MPI_Send(data + offset, static_cast<int>(chunk), MPI_UNSIGNED, destination, tag,
                 communicator);
        offset += chunk;
    }
}

void receiveUnsigned(unsigned int* data, size_t count, const int source, const int tag,
                     MPI_Comm communicator) {
    constexpr size_t maxCount = static_cast<size_t>(std::numeric_limits<int>::max());
    size_t offset = 0;
    while (offset < count) {
        const size_t chunk = std::min(count - offset, maxCount);
        MPI_Recv(data + offset, static_cast<int>(chunk), MPI_UNSIGNED, source, tag, communicator,
                 MPI_STATUS_IGNORE);
        offset += chunk;
    }
}

void copyGlobalBlockToLocal(const std::vector<unsigned int>& global,
                            std::vector<unsigned int>& local, const size_t numNodes,
                            const size_t rowOffset, const size_t rowCount,
                            const size_t columnOffset, const size_t columnCount) {
    if (rowCount == 0 || columnCount == 0) {
        return;
    }

    for (size_t localRow = 0; localRow < rowCount; ++localRow) {
        const size_t globalRow = rowOffset + localRow;
        std::copy_n(global.data() + idx2(columnOffset, globalRow, numNodes), columnCount,
                    local.data() + localRow * columnCount);
    }
}

void copyLocalBlockToGlobal(const std::vector<unsigned int>& local,
                            std::vector<unsigned int>& global, const size_t numNodes,
                            const size_t rowOffset, const size_t rowCount,
                            const size_t columnOffset, const size_t columnCount) {
    if (rowCount == 0 || columnCount == 0) {
        return;
    }

    for (size_t localRow = 0; localRow < rowCount; ++localRow) {
        const size_t globalRow = rowOffset + localRow;
        std::copy_n(local.data() + localRow * columnCount, columnCount,
                    global.data() + idx2(columnOffset, globalRow, numNodes));
    }
}

void distributeDistanceMatrix(std::vector<unsigned int>& localDistance, const size_t numNodes,
                              const std::vector<size_t>& rowOffsets,
                              const std::vector<size_t>& columnOffsets, MPI_Comm grid,
                              const int gridRank, const int gridSize) {
    if (gridRank == 0) {
        std::vector<unsigned int> globalDistance(numNodes * numNodes);
        initializeDistanceMatrix(globalDistance, numNodes, 1, MAX_DISTANCE);

        for (int rank = 0; rank < gridSize; ++rank) {
            int coordinates[2];
            MPI_Cart_coords(grid, rank, 2, coordinates);
            const size_t rowOffset = rowOffsets[coordinates[0]];
            const size_t rowCount = rowOffsets[coordinates[0] + 1] - rowOffset;
            const size_t columnOffset = columnOffsets[coordinates[1]];
            const size_t columnCount = columnOffsets[coordinates[1] + 1] - columnOffset;

            if (rank == 0) {
                copyGlobalBlockToLocal(globalDistance, localDistance, numNodes, rowOffset, rowCount,
                                       columnOffset, columnCount);
            } else {
                std::vector<unsigned int> packed(rowCount * columnCount);
                copyGlobalBlockToLocal(globalDistance, packed, numNodes, rowOffset, rowCount,
                                       columnOffset, columnCount);
                sendUnsigned(packed.data(), packed.size(), rank, 0, grid);
            }
        }
    } else {
        receiveUnsigned(localDistance.data(), localDistance.size(), 0, 0, grid);
    }
}

std::vector<unsigned int> gatherDistanceMatrix(const std::vector<unsigned int>& localDistance,
                                                const size_t numNodes,
                                                const std::vector<size_t>& rowOffsets,
                                                const std::vector<size_t>& columnOffsets,
                                                MPI_Comm grid, const int gridRank,
                                                const int gridSize) {
    if (gridRank != 0) {
        sendUnsigned(localDistance.data(), localDistance.size(), 0, 1, grid);
        return {};
    }

    std::vector<unsigned int> globalDistance(numNodes * numNodes);
    for (int rank = 0; rank < gridSize; ++rank) {
        int coordinates[2];
        MPI_Cart_coords(grid, rank, 2, coordinates);
        const size_t rowOffset = rowOffsets[coordinates[0]];
        const size_t rowCount = rowOffsets[coordinates[0] + 1] - rowOffset;
        const size_t columnOffset = columnOffsets[coordinates[1]];
        const size_t columnCount = columnOffsets[coordinates[1] + 1] - columnOffset;

        if (rank == 0) {
            copyLocalBlockToGlobal(localDistance, globalDistance, numNodes, rowOffset, rowCount,
                                   columnOffset, columnCount);
        } else {
            std::vector<unsigned int> packed(rowCount * columnCount);
            receiveUnsigned(packed.data(), packed.size(), rank, 1, grid);
            copyLocalBlockToGlobal(packed, globalDistance, numNodes, rowOffset, rowCount,
                                   columnOffset, columnCount);
        }
    }
    return globalDistance;
}

void initializeLocalPathMatrix(std::vector<unsigned int>& path, const size_t localSourceCount,
                               const size_t localSourceOffset,
                               const size_t localDestinationCount) {
    if (localSourceCount == 0 || localDestinationCount == 0) {
        return;
    }

    for (size_t localSource = 0; localSource < localSourceCount; ++localSource) {
        std::fill_n(path.data() + localSource * localDestinationCount, localDestinationCount,
                    static_cast<unsigned int>(localSourceOffset + localSource));
    }
}

// Matrix blocks are distributed over a 2-D process grid. A process owns a
// contiguous source-row range and destination-column range. For each k, the
// D[i,k] and D[k,j] panels are broadcast only within a process row and column.
void floydWarshallDistributed(std::vector<unsigned int>& distance,
                              std::vector<unsigned int>& path, const size_t numNodes,
                              const size_t localSourceOffset, const size_t localSourceCount,
                              const size_t localDestinationOffset,
                              const size_t localDestinationCount,
                              const std::vector<size_t>& sourceOffsets,
                              const std::vector<size_t>& destinationOffsets,
                              const int processSource, const int processDestination,
                              MPI_Comm columnCommunicator, MPI_Comm rowCommunicator) {
    std::vector<unsigned int> distanceToPivot(localSourceCount);
    std::vector<unsigned int> distanceFromPivot(localDestinationCount);
    unsigned int* const localDistance = distance.data();
    unsigned int* const localPath = path.data();

    for (size_t k = 0; k < numNodes; ++k) {
        const int pivotProcessSource = ownerOf(sourceOffsets, k);
        const int pivotProcessDestination = ownerOf(destinationOffsets, k);

        if (processDestination == pivotProcessDestination) {
            const size_t localK = k - localDestinationOffset;
            for (size_t localSource = 0; localSource < localSourceCount; ++localSource) {
                distanceToPivot[localSource] =
                    localDistance[localSource * localDestinationCount + localK];
            }
        }
        MPI_Bcast(distanceToPivot.data(), static_cast<int>(localSourceCount), MPI_UNSIGNED,
                  pivotProcessDestination, rowCommunicator);

        if (processSource == pivotProcessSource && localDestinationCount != 0) {
            const size_t localK = k - localSourceOffset;
            std::copy_n(localDistance + localK * localDestinationCount, localDestinationCount,
                        distanceFromPivot.data());
        }
        MPI_Bcast(distanceFromPivot.data(), static_cast<int>(localDestinationCount), MPI_UNSIGNED,
                  pivotProcessSource, columnCommunicator);

        for (size_t localSource = 0; localSource < localSourceCount; ++localSource) {
            const unsigned int sourceToPivot = distanceToPivot[localSource];
            unsigned int* const distanceRow =
                localDistance + localSource * localDestinationCount;
            unsigned int* const pathRow = localPath + localSource * localDestinationCount;

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC ivdep
#endif
            for (size_t localDestination = 0; localDestination < localDestinationCount;
                 ++localDestination) {
                const unsigned int newDistance =
                    sourceToPivot + distanceFromPivot[localDestination];
                if (newDistance < distanceRow[localDestination]) {
                    distanceRow[localDestination] = newDistance;
                    pathRow[localDestination] = static_cast<unsigned int>(k);
                }
            }
        }
    }
}

bool validateResult(const std::vector<unsigned int>& dist, const size_t numNodes) {
    for (size_t i = 0; i < numNodes; ++i) {
        if (dist[idx2(i, i, numNodes)] != 0) {
            printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", i, i);
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
                    printf("Validation failed: triangle inequality violated at [%zu,%zu,%zu]\n", i,
                           j, k);
                    return false;
                }
            }
        }
    }

    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Number of nodes in the graph (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int worldSize = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    int dimensions[2] = {0, 0};
    MPI_Dims_create(worldSize, 2, dimensions);
    const int periods[2] = {0, 0};
    MPI_Comm grid = MPI_COMM_NULL;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dimensions, periods, 0, &grid);

    int gridRank = 0;
    int gridSize = 0;
    MPI_Comm_rank(grid, &gridRank);
    MPI_Comm_size(grid, &gridSize);

    int coordinates[2];
    MPI_Cart_coords(grid, gridRank, 2, coordinates);

    MPI_Comm columnCommunicator = MPI_COMM_NULL;
    MPI_Comm rowCommunicator = MPI_COMM_NULL;
    MPI_Comm_split(grid, coordinates[1], coordinates[0], &columnCommunicator);
    MPI_Comm_split(grid, coordinates[0], coordinates[1], &rowCommunicator);

    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numNodes = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (gridRank == 0) {
                printUsage(argv[0]);
            }
            MPI_Comm_free(&rowCommunicator);
            MPI_Comm_free(&columnCommunicator);
            MPI_Comm_free(&grid);
            MPI_Finalize();
            return 0;
        } else {
            if (gridRank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Comm_free(&rowCommunicator);
            MPI_Comm_free(&columnCommunicator);
            MPI_Comm_free(&grid);
            MPI_Finalize();
            return 1;
        }
    }

    if (numNodes > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (gridRank == 0) {
            printf("Number of nodes exceeds the MPI message-size limit\n");
        }
        MPI_Comm_free(&rowCommunicator);
        MPI_Comm_free(&columnCommunicator);
        MPI_Comm_free(&grid);
        MPI_Finalize();
        return 1;
    }

    if (gridRank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing graph...\n");
    }

    const std::vector<size_t> rowOffsets = makeBlockOffsets(numNodes, dimensions[0]);
    const std::vector<size_t> columnOffsets = makeBlockOffsets(numNodes, dimensions[1]);
    const size_t localRowOffset = rowOffsets[coordinates[0]];
    const size_t localRowCount = rowOffsets[coordinates[0] + 1] - localRowOffset;
    const size_t localColumnOffset = columnOffsets[coordinates[1]];
    const size_t localColumnCount = columnOffsets[coordinates[1] + 1] - localColumnOffset;

    std::vector<unsigned int> distance(localRowCount * localColumnCount);
    std::vector<unsigned int> path(localRowCount * localColumnCount);
    distributeDistanceMatrix(distance, numNodes, rowOffsets, columnOffsets, grid, gridRank,
                             gridSize);
    initializeLocalPathMatrix(path, localRowCount, localRowOffset, localColumnCount);

    if (gridRank == 0) {
        printf("Computing shortest paths...\n");
    }

    MPI_Barrier(grid);
    const double start = MPI_Wtime();
    floydWarshallDistributed(distance, path, numNodes, localRowOffset, localRowCount,
                             localColumnOffset, localColumnCount, rowOffsets, columnOffsets,
                             coordinates[0], coordinates[1], columnCommunicator, rowCommunicator);
    const double localElapsedSeconds = MPI_Wtime() - start;

    double elapsedSeconds = 0.0;
    MPI_Reduce(&localElapsedSeconds, &elapsedSeconds, 1, MPI_DOUBLE, MPI_MAX, 0, grid);

    if (gridRank == 0) {
        const long elapsedMilliseconds = static_cast<long>(elapsedSeconds * 1000.0);
        printf("Computation time: %ld ms\n", elapsedMilliseconds);
        const double operations = static_cast<double>(numNodes) * numNodes * numNodes;
        const double gops = elapsedSeconds > 0.0 ? operations / elapsedSeconds / 1e9 : 0.0;
        printf("Performance: %.3f GOPS\n", gops);
    }

    std::vector<unsigned int> globalDistance;
    if (printResults || validate) {
        globalDistance = gatherDistanceMatrix(distance, numNodes, rowOffsets, columnOffsets, grid,
                                              gridRank, gridSize);
    }

    int exitCode = 0;
    if (gridRank == 0 && printResults) {
        print_results_int(globalDistance, "DistanceMatrix");
    }
    if (gridRank == 0 && validate) {
        printf("Validating result...\n");
        if (validateResult(globalDistance, numNodes)) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            exitCode = 1;
        }
    }

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, grid);
    MPI_Comm_free(&rowCommunicator);
    MPI_Comm_free(&columnCommunicator);
    MPI_Comm_free(&grid);
    MPI_Finalize();
    return exitCode;
}
