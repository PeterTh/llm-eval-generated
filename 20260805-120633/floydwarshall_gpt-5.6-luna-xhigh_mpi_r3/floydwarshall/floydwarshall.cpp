#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

// The original benchmark stores row i, column j at idx2(j, i, n).
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

// Return the number of elements in a block of a balanced, contiguous partition.
size_t blockSize(const size_t n, const int block, const int blockCount) noexcept {
    const size_t base = n / static_cast<size_t>(blockCount);
    const size_t remainder = n % static_cast<size_t>(blockCount);
    return base + (static_cast<size_t>(block) < remainder ? 1 : 0);
}

size_t blockStart(const size_t n, const int block, const int blockCount) noexcept {
    const size_t base = n / static_cast<size_t>(blockCount);
    const size_t remainder = n % static_cast<size_t>(blockCount);
    return base * static_cast<size_t>(block) +
           std::min(static_cast<size_t>(block), remainder);
}

// Find the owner of a global index without constructing an n-element owner map.
int blockOwner(const size_t index, const size_t n, const int blockCount) noexcept {
    const size_t base = n / static_cast<size_t>(blockCount);
    const size_t remainder = n % static_cast<size_t>(blockCount);

    // This also handles more MPI processes in a dimension than graph nodes.
    if (base == 0) {
        return static_cast<int>(index);
    }

    const size_t largeBlock = base + 1;
    const size_t largeBlockElements = largeBlock * remainder;
    if (index < largeBlockElements) {
        return static_cast<int>(index / largeBlock);
    }
    return static_cast<int>(remainder + (index - largeBlockElements) / base);
}

int mpiCount(const size_t count) {
    if (count > static_cast<size_t>(std::numeric_limits<int>::max())) {
        return -1;
    }
    return static_cast<int>(count);
}

// Generate the same rand_r stream as the original serial initializer, but
// distribute each generated block directly to its owning MPI rank. This keeps
// initialization memory-distributed and preserves result hashes exactly.
bool initializeDistanceMatrix(std::vector<unsigned int>& dist,
                              const size_t numNodes,
                              const unsigned int rangeMin,
                              const unsigned int rangeMax,
                              const size_t rowStart,
                              const size_t localRows,
                              const size_t localCols,
                              const size_t colStart,
                              const int procRows,
                              const int procCols,
                              const int worldRank,
                              MPI_Comm world) {
    const int segmentCount = mpiCount(localCols);
    if (segmentCount < 0 || mpiCount(localRows) < 0 ||
        mpiCount(localRows * localCols) < 0) {
        return false;
    }

    std::vector<unsigned int> segment(std::max<size_t>(1, localCols));

    if (worldRank == 0) {
        unsigned int seed = 42;
        const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;

        for (size_t globalRow = 0; globalRow < numNodes; ++globalRow) {
            const int ownerRow = blockOwner(globalRow, numNodes, procRows);

            for (int colBlock = 0; colBlock < procCols; ++colBlock) {
                const size_t columns = blockSize(numNodes, colBlock, procCols);
                if (columns == 0) {
                    continue;
                }

                for (size_t localColumn = 0; localColumn < columns; ++localColumn) {
                    segment[localColumn] = rangeMin +
                        static_cast<unsigned int>(
                            range * rand_r(&seed) / static_cast<double>(RAND_MAX));
                }

                const int target = ownerRow * procCols + colBlock;
                if (target == 0) {
                    const size_t localRow = globalRow - rowStart;
                    std::copy_n(segment.data(), columns,
                                dist.data() + localRow * localCols);
                } else {
                    MPI_Send(segment.data(), mpiCount(columns), MPI_UNSIGNED,
                             target, 0, world);
                }
            }
        }
    } else if (localRows != 0 && localCols != 0) {
        const int receiveCount = mpiCount(localCols);
        if (receiveCount < 0) {
            return false;
        }
        for (size_t localRow = 0; localRow < localRows; ++localRow) {
            MPI_Recv(dist.data() + localRow * localCols, receiveCount,
                     MPI_UNSIGNED, 0, 0, world, MPI_STATUS_IGNORE);
        }
    }

    // The diagonal is set after random initialization, exactly as in the
    // original implementation.
    if (localCols != 0) {
        for (size_t localRow = 0; localRow < localRows; ++localRow) {
            const size_t globalRow = rowStart + localRow;
            if (globalRow >= colStart && globalRow < colStart + localCols) {
                dist[localRow * localCols + (globalRow - colStart)] = 0;
            }
        }
    }

    return true;
}

void initializePathMatrix(std::vector<unsigned int>& path,
                          const size_t rowStart,
                          const size_t localRows,
                          const size_t localCols) {
    if (localCols != 0) {
        for (size_t localRow = 0; localRow < localRows; ++localRow) {
            std::fill_n(path.data() + localRow * localCols, localCols,
                        static_cast<unsigned int>(rowStart + localRow));
        }
    }
}

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes,
                   const size_t rowStart,
                   const size_t localRows,
                   const size_t colStart,
                   const size_t localCols,
                   const int procRows,
                   const int procCols,
                   const int rowCoordinate,
                   const int colCoordinate,
                   MPI_Comm rowCommunicator,
                   MPI_Comm columnCommunicator) {
    std::vector<unsigned int> pivotRow(std::max<size_t>(1, localCols));
    std::vector<unsigned int> pivotColumn(std::max<size_t>(1, localRows));

    const int rowCount = mpiCount(localRows);
    const int columnCount = mpiCount(localCols);
    if (rowCount < 0 || columnCount < 0) {
        return;
    }

    for (size_t k = 0; k < numNodes; ++k) {
        const int pivotRowOwner = blockOwner(k, numNodes, procRows);
        const int pivotColumnOwner = blockOwner(k, numNodes, procCols);

        // The pivot row segment is initially on (pivotRowOwner, my column),
        // and is broadcast down a process column.
        if (rowCoordinate == pivotRowOwner && localCols != 0) {
            const size_t localPivotRow = k - rowStart;
            std::copy_n(dist.data() + localPivotRow * localCols, localCols,
                        pivotRow.data());
        }

        // The pivot column segment is initially on (my row, pivotColumnOwner),
        // and is broadcast across a process row.
        if (colCoordinate == pivotColumnOwner && localRows != 0) {
            const size_t localPivotColumn = k - colStart;
            for (size_t localRow = 0; localRow < localRows; ++localRow) {
                pivotColumn[localRow] =
                    dist[localRow * localCols + localPivotColumn];
            }
        }

        // Start both collectives before waiting so the two communication
        // directions can make progress concurrently on capable MPI fabrics.
        MPI_Request requests[2];
        MPI_Ibcast(pivotRow.data(), columnCount, MPI_UNSIGNED,
                   pivotRowOwner, columnCommunicator, &requests[0]);
        MPI_Ibcast(pivotColumn.data(), rowCount, MPI_UNSIGNED,
                   pivotColumnOwner, rowCommunicator, &requests[1]);
        MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);

        for (size_t localRow = 0; localRow < localRows; ++localRow) {
            if (localCols == 0) {
                break;
            }
            unsigned int* distanceRow = dist.data() + localRow * localCols;
            unsigned int* pathRow = path.data() + localRow * localCols;
            const unsigned int distanceToPivot = pivotColumn[localRow];

            for (size_t localColumn = 0; localColumn < localCols; ++localColumn) {
                const unsigned int newDistance =
                    distanceToPivot + pivotRow[localColumn];
                if (newDistance < distanceRow[localColumn]) {
                    distanceRow[localColumn] = newDistance;
                    pathRow[localColumn] = static_cast<unsigned int>(k);
                }
            }
        }
    }
}

// Gather 2D blocks into the original row-major (idx2(j, i, n)) layout.
void gatherDistanceMatrix(const std::vector<unsigned int>& localDist,
                          std::vector<unsigned int>& globalDist,
                          const size_t numNodes,
                          const size_t localRows,
                          const size_t localCols,
                          const int procCols,
                          const int worldSize,
                          const int worldRank,
                          MPI_Comm world) {
    const size_t localElementCount = localRows * localCols;
    const int sendCount = mpiCount(localElementCount);

    std::vector<int> receiveCounts;
    std::vector<int> displacements;
    std::vector<unsigned int> packed;

    if (worldRank == 0) {
        receiveCounts.resize(static_cast<size_t>(worldSize));
        displacements.resize(static_cast<size_t>(worldSize));

        size_t displacement = 0;
        for (int rank = 0; rank < worldSize; ++rank) {
            const int rowBlock = rank / procCols;
            const int columnBlock = rank % procCols;
            const size_t count = blockSize(numNodes, rowBlock, worldSize / procCols) *
                                 blockSize(numNodes, columnBlock, procCols);
            receiveCounts[static_cast<size_t>(rank)] = mpiCount(count);
            displacements[static_cast<size_t>(rank)] = mpiCount(displacement);
            displacement += count;
        }
        packed.resize(displacement);

        if (localElementCount != 0) {
            std::copy_n(localDist.data(), localElementCount,
                        packed.data() + displacements[0]);
        }
        MPI_Gatherv(MPI_IN_PLACE, 0, MPI_UNSIGNED,
                    packed.data(), receiveCounts.data(), displacements.data(),
                    MPI_UNSIGNED, 0, world);
    } else {
        MPI_Gatherv(localDist.data(), sendCount, MPI_UNSIGNED,
                    nullptr, nullptr, nullptr, MPI_UNSIGNED, 0, world);
    }

    if (worldRank == 0) {
        globalDist.assign(numNodes * numNodes, 0);
        const int procRows = worldSize / procCols;

        for (int rank = 0; rank < worldSize; ++rank) {
            const int rowBlock = rank / procCols;
            const int columnBlock = rank % procCols;
            const size_t rows = blockSize(numNodes, rowBlock, procRows);
            const size_t columns = blockSize(numNodes, columnBlock, procCols);
            const size_t sourceOffset =
                static_cast<size_t>(displacements[static_cast<size_t>(rank)]);

            for (size_t localRow = 0; localRow < rows; ++localRow) {
                std::copy_n(packed.data() + sourceOffset + localRow * columns,
                            columns,
                            globalDist.data() +
                                (blockStart(numNodes, rowBlock, procRows) + localRow) *
                                    numNodes +
                                blockStart(numNodes, columnBlock, procCols));
            }
        }
    }
}

bool validateResult(const std::vector<unsigned int>& dist, const size_t numNodes) {
    // 1. Diagonal should be zero.
    for (size_t i = 0; i < numNodes; ++i) {
        if (dist[idx2(i, i, numNodes)] != 0) {
            printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", i, i);
            return false;
        }
    }

    // 2. Triangle inequality: dist[i][k] + dist[k][j] >= dist[i][j].
    // Check a sample of paths to avoid O(n^3) validation time.
    for (size_t i = 0; i < std::min(numNodes, static_cast<size_t>(10)); ++i) {
        for (size_t j = 0; j < std::min(numNodes, static_cast<size_t>(10)); ++j) {
            for (size_t k = 0; k < numNodes; ++k) {
                const unsigned int distIJ = dist[idx2(j, i, numNodes)];
                const unsigned int distIK = dist[idx2(k, i, numNodes)];
                const unsigned int distKJ = dist[idx2(j, k, numNodes)];

                if (distIK < INF && distKJ < INF && distIK + distKJ < distIJ) {
                    printf("Validation failed: triangle inequality violated at [%zu,%zu,%zu]\n",
                           i, j, k);
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
    int worldRank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;
    bool parseError = false;
    bool showHelp = false;

    // Parse command line arguments on every rank so all ranks take the same
    // control path before entering MPI collectives.
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numNodes = static_cast<size_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            showHelp = true;
        } else {
            parseError = true;
        }
    }

    if (showHelp || parseError) {
        if (worldRank == 0) {
            if (parseError) {
                printf("Unknown or incomplete option\n");
            }
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return parseError ? 1 : 0;
    }

    if (numNodes != 0 && numNodes >
        std::numeric_limits<size_t>::max() / numNodes) {
        if (worldRank == 0) {
            printf("Number of nodes is too large\n");
        }
        MPI_Finalize();
        return 1;
    }

    if (worldRank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("MPI processes: %d\n", worldSize);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    int dimensions[2] = {0, 0};
    MPI_Dims_create(worldSize, 2, dimensions);
    const int procRows = dimensions[0];
    const int procCols = dimensions[1];
    const int rowCoordinate = worldRank / procCols;
    const int colCoordinate = worldRank % procCols;

    const size_t rowStart = blockStart(numNodes, rowCoordinate, procRows);
    const size_t localRows = blockSize(numNodes, rowCoordinate, procRows);
    const size_t colStart = blockStart(numNodes, colCoordinate, procCols);
    const size_t localCols = blockSize(numNodes, colCoordinate, procCols);

    std::vector<unsigned int> dist(localRows * localCols);
    std::vector<unsigned int> path(localRows * localCols);

    if (worldRank == 0) {
        printf("Initializing graph...\n");
    }
    const bool initialized = initializeDistanceMatrix(
        dist, numNodes, 1, MAX_DISTANCE, rowStart, localRows, localCols,
        colStart, procRows, procCols, worldRank, MPI_COMM_WORLD);
    if (!initialized) {
        if (worldRank == 0) {
            printf("Matrix dimensions exceed MPI count limits\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    initializePathMatrix(path, rowStart, localRows, localCols);

    MPI_Comm rowCommunicator;
    MPI_Comm columnCommunicator;
    // rowCommunicator contains a process row (fixed row coordinate), while
    // columnCommunicator contains a process column (fixed column coordinate).
    MPI_Comm_split(MPI_COMM_WORLD, rowCoordinate, colCoordinate,
                   &rowCommunicator);
    MPI_Comm_split(MPI_COMM_WORLD, colCoordinate, rowCoordinate,
                   &columnCommunicator);

    if (worldRank == 0) {
        printf("Computing shortest paths...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    floydWarshall(dist, path, numNodes, rowStart, localRows, colStart,
                  localCols, procRows, procCols, rowCoordinate, colCoordinate,
                  rowCommunicator, columnCommunicator);

    const double elapsed = MPI_Wtime() - start;
    double maximumElapsed = 0.0;
    MPI_Reduce(&elapsed, &maximumElapsed, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);

    MPI_Comm_free(&rowCommunicator);
    MPI_Comm_free(&columnCommunicator);

    if (worldRank == 0) {
        const long durationMilliseconds =
            static_cast<long>(maximumElapsed * 1000.0);
        printf("Computation time: %ld ms\n", durationMilliseconds);

        const double ops = static_cast<double>(numNodes) *
                           static_cast<double>(numNodes) *
                           static_cast<double>(numNodes);
        const double gflops = ops / (durationMilliseconds / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);
    }

    // Full-matrix operations are intentionally outside the timed region and
    // happen only when requested, so normal benchmark runs remain distributed.
    std::vector<unsigned int> globalDist;
    if (printResults || validate) {
        gatherDistanceMatrix(dist, globalDist, numNodes, localRows, localCols,
                             procCols, worldSize,
                             worldRank, MPI_COMM_WORLD);
    }

    if (worldRank == 0 && printResults) {
        print_results_int(globalDist, "DistanceMatrix");
    }

    int valid = 1;
    if (validate) {
        if (worldRank == 0) {
            printf("Validating result...\n");
            valid = validateResult(globalDist, numNodes) ? 1 : 0;
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return validate && valid == 0 ? 1 : 0;
}
