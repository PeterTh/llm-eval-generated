#include <mpi.h>

#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

// The original benchmark stores element (source, destination) at source*n+destination.
inline constexpr std::size_t idx2(const std::size_t source,
                                  const std::size_t destination,
                                  const std::size_t n) noexcept {
    return source * n + destination;
}

// With a block-cyclic 1D distribution, this is the number of indices owned by
// coordinate coordinate in a dimension of size processCount.
inline std::size_t localExtent(const std::size_t globalExtent,
                               const int processCount,
                               const int coordinate) noexcept {
    if (static_cast<std::size_t>(coordinate) >= globalExtent) {
        return 0;
    }
    return 1 + (globalExtent - 1 - static_cast<std::size_t>(coordinate)) /
                   static_cast<std::size_t>(processCount);
}

inline int mpiCount(const std::size_t count) {
    // The benchmark's matrices are already limited by addressable memory. The
    // explicit check prevents a silent truncation if an unusually large local
    // allocation is requested on an MPI implementation with 32-bit counts.
    if (count > static_cast<std::size_t>(INT_MAX)) {
        std::fprintf(stderr, "MPI message is too large for a 32-bit MPI count\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    return static_cast<int>(count);
}

void rankCoordinates(const int rank, const int processColumns,
                     int& processRow, int& processColumn) noexcept {
    processRow = rank / processColumns;
    processColumn = rank % processColumns;
}

void unpackLocalBlock(const std::vector<unsigned int>& localDist,
                      std::vector<unsigned int>& globalDist,
                      const std::size_t numNodes,
                      const int processRows,
                      const int processColumns,
                      const int processRow,
                      const int processColumn) {
    const std::size_t localColumns =
        localExtent(numNodes, processColumns, processColumn);

    std::size_t localRow = 0;
    for (std::size_t source = static_cast<std::size_t>(processRow);
         source < numNodes;
         source += static_cast<std::size_t>(processRows), ++localRow) {
        std::size_t localColumn = 0;
        for (std::size_t destination = static_cast<std::size_t>(processColumn);
             destination < numNodes;
             destination += static_cast<std::size_t>(processColumns),
             ++localColumn) {
            globalDist[idx2(source, destination, numNodes)] =
                localDist[localRow * localColumns + localColumn];
        }
    }
}

void initializeAndDistributeDistanceMatrix(
    std::vector<unsigned int>& localDist,
    const std::size_t numNodes,
    const unsigned int rangeMin,
    const unsigned int rangeMax,
    const int processRows,
    const int processColumns,
    const int processRow,
    const int processColumn,
    const int worldRank) {
    const std::size_t localColumns =
        localExtent(numNodes, processColumns, processColumn);

    if (worldRank == 0) {
        // Generate the same row-major rand_r stream as the original benchmark,
        // but retain only one row while sending its pieces to their owners.
        // This avoids a root-sized n*n allocation during normal execution.
        unsigned int seed = 42;
        const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;
        std::vector<unsigned int> row(numNodes);
        std::vector<unsigned int> sendBuffer;

        for (std::size_t source = 0; source < numNodes; ++source) {
            for (std::size_t destination = 0; destination < numNodes;
                 ++destination) {
                row[destination] =
                    rangeMin + static_cast<unsigned int>(
                                   range * rand_r(&seed) / RAND_MAX);
            }
            row[source] = 0;

            const int targetProcessRow =
                static_cast<int>(source % static_cast<std::size_t>(processRows));
            const std::size_t localRow =
                source / static_cast<std::size_t>(processRows);

            for (int targetProcessColumn = 0;
                 targetProcessColumn < processColumns; ++targetProcessColumn) {
                const std::size_t targetColumns = localExtent(
                    numNodes, processColumns, targetProcessColumn);
                const int targetRank =
                    targetProcessRow * processColumns + targetProcessColumn;

                if (targetRank == 0) {
                    for (std::size_t localColumn = 0;
                         localColumn < targetColumns; ++localColumn) {
                        const std::size_t destination =
                            static_cast<std::size_t>(targetProcessColumn) +
                            localColumn *
                                static_cast<std::size_t>(processColumns);
                        localDist[localRow * localColumns + localColumn] =
                            row[destination];
                    }
                } else {
                    sendBuffer.resize(targetColumns);
                    for (std::size_t localColumn = 0;
                         localColumn < targetColumns; ++localColumn) {
                        const std::size_t destination =
                            static_cast<std::size_t>(targetProcessColumn) +
                            localColumn *
                                static_cast<std::size_t>(processColumns);
                        sendBuffer[localColumn] = row[destination];
                    }
                    unsigned int emptyBuffer = 0;
                    unsigned int* sendData = sendBuffer.empty()
                                                 ? &emptyBuffer
                                                 : sendBuffer.data();
                    MPI_Send(sendData, mpiCount(sendBuffer.size()),
                             MPI_UNSIGNED, targetRank, 0, MPI_COMM_WORLD);
                }
            }
        }
    } else {
        unsigned int emptyBuffer = 0;
        std::size_t localRow = 0;
        for (std::size_t source = static_cast<std::size_t>(processRow);
             source < numNodes;
             source += static_cast<std::size_t>(processRows), ++localRow) {
            unsigned int* receiveData = localDist.empty()
                                             ? &emptyBuffer
                                             : localDist.data() +
                                                   localRow * localColumns;
            MPI_Recv(receiveData, mpiCount(localColumns), MPI_UNSIGNED, 0, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
}

void initializeLocalPathMatrix(std::vector<unsigned int>& path,
                               const std::size_t numNodes,
                               const int processRows,
                               const int processColumns,
                               const int processRow,
                               const int processColumn) {
    const std::size_t localRows =
        localExtent(numNodes, processRows, processRow);
    const std::size_t localColumns =
        localExtent(numNodes, processColumns, processColumn);

    for (std::size_t localRow = 0; localRow < localRows; ++localRow) {
        for (std::size_t localColumn = 0; localColumn < localColumns;
             ++localColumn) {
            const std::size_t destination =
                static_cast<std::size_t>(processColumn) +
                localColumn * static_cast<std::size_t>(processColumns);
            path[localRow * localColumns + localColumn] =
                static_cast<unsigned int>(destination);
        }
    }
}

void floydWarshall(std::vector<unsigned int>& localDist,
                   std::vector<unsigned int>& localPath,
                   const std::size_t numNodes,
                   const int processRows,
                   const int processColumns,
                   const int processRow,
                   const int processColumn,
                   MPI_Comm rowCommunicator,
                   MPI_Comm columnCommunicator) {
    const std::size_t localRows =
        localExtent(numNodes, processRows, processRow);
    const std::size_t localColumns =
        localExtent(numNodes, processColumns, processColumn);

    std::vector<unsigned int> pivotRow(localColumns);
    std::vector<unsigned int> pivotColumn(localRows);
    unsigned int emptyBuffer = 0;

    for (std::size_t k = 0; k < numNodes; ++k) {
        const int pivotProcessRow =
            static_cast<int>(k % static_cast<std::size_t>(processRows));
        const int pivotProcessColumn =
            static_cast<int>(k % static_cast<std::size_t>(processColumns));

        // The pivot row is distributed across process columns and then
        // broadcast down each process column. The pivot column is distributed
        // across process rows and then broadcast across each process row.
        if (processRow == pivotProcessRow && !pivotRow.empty()) {
            const std::size_t localPivotRow =
                k / static_cast<std::size_t>(processRows);
            const std::size_t offset = localPivotRow * localColumns;
            std::copy_n(localDist.data() + offset, localColumns,
                        pivotRow.data());
        }
        if (processColumn == pivotProcessColumn && !pivotColumn.empty()) {
            const std::size_t localPivotColumn =
                k / static_cast<std::size_t>(processColumns);
            for (std::size_t localRow = 0; localRow < localRows; ++localRow) {
                pivotColumn[localRow] =
                    localDist[localRow * localColumns + localPivotColumn];
            }
        }

        // Starting both collectives before waiting permits the MPI library to
        // overlap the row and column communication when the network supports
        // it. The fallback buffer is only used for zero-count collectives.
        unsigned int* rowBuffer =
            pivotRow.empty() ? &emptyBuffer : pivotRow.data();
        unsigned int* columnBuffer =
            pivotColumn.empty() ? &emptyBuffer : pivotColumn.data();
        MPI_Request requests[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};
        MPI_Ibcast(rowBuffer, mpiCount(localColumns), MPI_UNSIGNED,
                   pivotProcessRow, columnCommunicator, &requests[0]);
        MPI_Ibcast(columnBuffer, mpiCount(localRows), MPI_UNSIGNED,
                   pivotProcessColumn, rowCommunicator, &requests[1]);
        MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);

        if (localRows != 0 && localColumns != 0) {
            const unsigned int intermediate = static_cast<unsigned int>(k);
            for (std::size_t localRow = 0; localRow < localRows; ++localRow) {
                unsigned int* distanceRow =
                    localDist.data() + localRow * localColumns;
                const unsigned int distanceToPivot = pivotColumn[localRow];

                for (std::size_t localColumn = 0; localColumn < localColumns;
                     ++localColumn) {
                    // This intentionally uses unsigned arithmetic, matching
                    // the original benchmark's distance update semantics.
                    const unsigned int candidate =
                        distanceToPivot + pivotRow[localColumn];
                    if (candidate < distanceRow[localColumn]) {
                        distanceRow[localColumn] = candidate;
                        localPath[localRow * localColumns + localColumn] =
                            intermediate;
                    }
                }
            }
        }
    }
}

std::vector<unsigned int> gatherDistanceMatrix(
    const std::vector<unsigned int>& localDist,
    const std::size_t numNodes,
    const int processRows,
    const int processColumns,
    const int worldRank,
    const int worldSize) {
    int processRow = 0;
    int processColumn = 0;
    rankCoordinates(worldRank, processColumns, processRow, processColumn);

    std::vector<unsigned int> globalDist;
    if (worldRank == 0) {
        globalDist.resize(numNodes * numNodes);
        unpackLocalBlock(localDist, globalDist, numNodes, processRows,
                         processColumns, processRow, processColumn);

        std::vector<unsigned int> receiveBuffer;
        for (int sourceRank = 1; sourceRank < worldSize; ++sourceRank) {
            int sourceProcessRow = 0;
            int sourceProcessColumn = 0;
            rankCoordinates(sourceRank, processColumns, sourceProcessRow,
                            sourceProcessColumn);
            const std::size_t sourceRows = localExtent(
                numNodes, processRows, sourceProcessRow);
            const std::size_t sourceColumns = localExtent(
                numNodes, processColumns, sourceProcessColumn);
            receiveBuffer.resize(sourceRows * sourceColumns);
            MPI_Recv(receiveBuffer.data(), mpiCount(receiveBuffer.size()),
                     MPI_UNSIGNED, sourceRank, 0, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
            unpackLocalBlock(receiveBuffer, globalDist, numNodes, processRows,
                             processColumns, sourceProcessRow,
                             sourceProcessColumn);
        }
    } else {
        MPI_Send(localDist.data(), mpiCount(localDist.size()), MPI_UNSIGNED, 0,
                 0, MPI_COMM_WORLD);
    }

    return globalDist;
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

    for (std::size_t i = 0; i < std::min(numNodes, static_cast<std::size_t>(10));
         ++i) {
        for (std::size_t j = 0;
             j < std::min(numNodes, static_cast<std::size_t>(10)); ++j) {
            for (std::size_t k = 0; k < numNodes; ++k) {
                const unsigned int distIJ = dist[idx2(i, j, numNodes)];
                const unsigned int distIK = dist[idx2(i, k, numNodes)];
                const unsigned int distKJ = dist[idx2(k, j, numNodes)];

                if (distIK < INF && distKJ < INF &&
                    distIK + distKJ < distIJ) {
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

    std::size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numNodes = static_cast<std::size_t>(std::atoi(argv[++i]));
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

    int gridDimensions[2] = {0, 0};
    MPI_Dims_create(worldSize, 2, gridDimensions);
    const int processRows = gridDimensions[0];
    const int processColumns = gridDimensions[1];

    int processRow = 0;
    int processColumn = 0;
    rankCoordinates(worldRank, processColumns, processRow, processColumn);

    MPI_Comm rowCommunicator = MPI_COMM_NULL;
    MPI_Comm columnCommunicator = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, processRow, processColumn,
                   &rowCommunicator);
    MPI_Comm_split(MPI_COMM_WORLD, processColumn, processRow,
                   &columnCommunicator);

    const std::size_t localRows =
        localExtent(numNodes, processRows, processRow);
    const std::size_t localColumns =
        localExtent(numNodes, processColumns, processColumn);

    if (worldRank == 0) {
        std::printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        std::printf("Number of nodes: %zu\n", numNodes);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Initializing graph...\n");
    }

    std::vector<unsigned int> localDist(localRows * localColumns);
    std::vector<unsigned int> localPath(localRows * localColumns);

    initializeAndDistributeDistanceMatrix(
        localDist, numNodes, 1, MAX_DISTANCE, processRows, processColumns,
        processRow, processColumn, worldRank);

    initializeLocalPathMatrix(localPath, numNodes, processRows,
                              processColumns, processRow, processColumn);

    // Ensure that data distribution and all local allocations are complete
    // before the timed parallel region begins.
    MPI_Barrier(MPI_COMM_WORLD);
    if (worldRank == 0) {
        std::printf("Computing shortest paths...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    floydWarshall(localDist, localPath, numNodes, processRows, processColumns,
                  processRow, processColumn, rowCommunicator,
                  columnCommunicator);
    MPI_Barrier(MPI_COMM_WORLD);
    const double elapsedSeconds = MPI_Wtime() - start;
    double maxElapsedSeconds = 0.0;
    MPI_Reduce(&elapsedSeconds, &maxElapsedSeconds, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);

    if (worldRank == 0) {
        const long durationMilliseconds =
            static_cast<long>(maxElapsedSeconds * 1000.0);
        std::printf("Computation time: %ld ms\n", durationMilliseconds);
        const double operations = static_cast<double>(numNodes) *
                                  static_cast<double>(numNodes) *
                                  static_cast<double>(numNodes);
        const double gflops = maxElapsedSeconds > 0.0
                                  ? operations / maxElapsedSeconds / 1.0e9
                                  : 0.0;
        std::printf("Performance: %.3f GOPS\n", gflops);
    }

    bool valid = true;
    if (printResults || validate) {
        std::vector<unsigned int> globalDist = gatherDistanceMatrix(
            localDist, numNodes, processRows, processColumns, worldRank,
            worldSize);

        if (worldRank == 0) {
            if (printResults) {
                print_results_int(globalDist, "DistanceMatrix");
            }
            if (validate) {
                std::printf("Validating result...\n");
                valid = validateResult(globalDist, numNodes);
                std::printf("Validation: %s\n",
                            valid ? "PASSED" : "FAILED");
            }
        }
    }

    if (validate) {
        int validOnRoot = valid ? 1 : 0;
        MPI_Bcast(&validOnRoot, 1, MPI_INT, 0, MPI_COMM_WORLD);
        valid = validOnRoot != 0;
    }

    MPI_Comm_free(&rowCommunicator);
    MPI_Comm_free(&columnCommunicator);
    MPI_Finalize();
    return validate && !valid ? 1 : 0;
}
