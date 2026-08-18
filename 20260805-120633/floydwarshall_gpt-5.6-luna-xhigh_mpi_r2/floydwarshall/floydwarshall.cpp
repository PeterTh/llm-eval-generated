#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

// The original benchmark uses glibc/POSIX rand_r.  Its implementation is a
// three-step 32-bit LCG.  Keeping the generator here lets every rank start at
// the exact position in the original stream belonging to its local tile,
// without constructing the full matrix on rank zero.
constexpr uint32_t RAND_MULTIPLIER = 1103515245u;
constexpr uint32_t RAND_INCREMENT = 12345u;

inline uint32_t advanceLcg(uint32_t seed, uint64_t steps) noexcept {
    uint32_t multiplier = RAND_MULTIPLIER;
    uint32_t increment = RAND_INCREMENT;
    uint32_t resultMultiplier = 1u;
    uint32_t resultIncrement = 0u;

    while (steps != 0u) {
        if ((steps & 1u) != 0u) {
            resultIncrement = multiplier * resultIncrement + increment;
            resultMultiplier *= multiplier;
        }

        increment = multiplier * increment + increment;
        multiplier *= multiplier;
        steps >>= 1u;
    }

    return resultMultiplier * seed + resultIncrement;
}

inline unsigned int benchmarkRandR(unsigned int* seed) noexcept {
    uint32_t next = *seed;
    int result;

    next = next * RAND_MULTIPLIER + RAND_INCREMENT;
    result = static_cast<int>((next / 65536u) % 2048u);
    next = next * RAND_MULTIPLIER + RAND_INCREMENT;
    result = (result << 10) ^ static_cast<int>((next / 65536u) % 1024u);
    next = next * RAND_MULTIPLIER + RAND_INCREMENT;
    result = (result << 10) ^ static_cast<int>((next / 65536u) % 1024u);

    *seed = next;
    return static_cast<unsigned int>(result);
}

inline size_t blockStart(const size_t length, const int blocks, const int block) noexcept {
    const size_t base = length / static_cast<size_t>(blocks);
    const size_t remainder = length % static_cast<size_t>(blocks);
    return static_cast<size_t>(block) * base +
           std::min(static_cast<size_t>(block), remainder);
}

inline size_t blockSize(const size_t length, const int blocks, const int block) noexcept {
    const size_t base = length / static_cast<size_t>(blocks);
    const size_t remainder = length % static_cast<size_t>(blocks);
    return base + (static_cast<size_t>(block) < remainder ? 1u : 0u);
}

inline int blockOwner(const size_t index, const size_t length, const int blocks) noexcept {
    const size_t base = length / static_cast<size_t>(blocks);
    const size_t remainder = length % static_cast<size_t>(blocks);

    // When there are more process-grid blocks than matrix indices, the first
    // 'length' blocks contain one index each and the rest are empty.
    if (base == 0u) {
        return static_cast<int>(index);
    }

    const size_t largeBlockSize = base + 1u;
    const size_t largeBlockElements = remainder * largeBlockSize;
    if (index < largeBlockElements) {
        return static_cast<int>(index / largeBlockSize);
    }

    return static_cast<int>(remainder + (index - largeBlockElements) / base);
}

inline size_t tileIndex(const size_t localRow, const size_t localColumn,
                        const size_t localColumns) noexcept {
    return localRow * localColumns + localColumn;
}

void initializeDistanceTile(std::vector<unsigned int>& dist,
                            const size_t numNodes,
                            const size_t rowStart,
                            const size_t localRows,
                            const size_t columnStart,
                            const size_t localColumns,
                            const unsigned int rangeMin,
                            const unsigned int rangeMax) {
    unsigned int seed;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;

    for (size_t localRow = 0; localRow < localRows; ++localRow) {
        const size_t globalRow = rowStart + localRow;
        const size_t streamOffset = globalRow * numNodes + columnStart;
        seed = advanceLcg(42u, static_cast<uint64_t>(streamOffset) * 3u);

        for (size_t localColumn = 0; localColumn < localColumns; ++localColumn) {
            const unsigned int randomValue = benchmarkRandR(&seed);
            dist[tileIndex(localRow, localColumn, localColumns)] =
                rangeMin + static_cast<unsigned int>(
                               range * randomValue / static_cast<double>(RAND_MAX));
        }
    }

    // Set the diagonal after generating the random stream, exactly as in the
    // original full-matrix initializer.
    for (size_t localRow = 0; localRow < localRows; ++localRow) {
        const size_t globalRow = rowStart + localRow;
        if (globalRow >= columnStart && globalRow < columnStart + localColumns) {
            dist[tileIndex(localRow, globalRow - columnStart, localColumns)] = 0u;
        }
    }
}

void initializePathTile(std::vector<unsigned int>& path,
                        const size_t rowStart,
                        const size_t localRows,
                        const size_t localColumns) {
    for (size_t localRow = 0; localRow < localRows; ++localRow) {
        std::fill(path.begin() + static_cast<ptrdiff_t>(localRow * localColumns),
                  path.begin() + static_cast<ptrdiff_t>((localRow + 1u) * localColumns),
                  static_cast<unsigned int>(rowStart + localRow));
    }
}

void floydWarshallTile(std::vector<unsigned int>& dist,
                       std::vector<unsigned int>& path,
                       const size_t numNodes,
                       const size_t rowStart,
                       const size_t localRows,
                       const size_t columnStart,
                       const size_t localColumns,
                       const int gridRows,
                       const int gridColumns,
                       const int rowCoordinate,
                       const int columnCoordinate,
                       MPI_Comm rowCommunicator,
                       MPI_Comm columnCommunicator) {
    std::vector<unsigned int> pivotRow(localColumns);
    std::vector<unsigned int> pivotColumn(localRows);

    for (size_t k = 0; k < numNodes; ++k) {
        const int ownerRow = blockOwner(k, numNodes, gridRows);
        const int ownerColumn = blockOwner(k, numNodes, gridColumns);

        if (rowCoordinate == ownerRow) {
            const size_t localPivotRow = k - rowStart;
            if (localColumns != 0u) {
                std::memcpy(pivotRow.data(),
                            dist.data() + localPivotRow * localColumns,
                            localColumns * sizeof(unsigned int));
            }
        }

        if (columnCoordinate == ownerColumn) {
            const size_t localPivotColumn = k - columnStart;
            for (size_t localRow = 0; localRow < localRows; ++localRow) {
                pivotColumn[localRow] =
                    dist[tileIndex(localRow, localPivotColumn, localColumns)];
            }
        }

        // The pivot row segment is held by the process row containing k and
        // is shared down each process column.  The pivot column segment is
        // held by the process column containing k and is shared across each
        // process row.
        MPI_Bcast(pivotRow.data(), static_cast<int>(localColumns), MPI_UNSIGNED,
                  ownerRow, columnCommunicator);
        MPI_Bcast(pivotColumn.data(), static_cast<int>(localRows), MPI_UNSIGNED,
                  ownerColumn, rowCommunicator);

        for (size_t localRow = 0; localRow < localRows; ++localRow) {
            unsigned int* const row = dist.data() + localRow * localColumns;
            unsigned int* const pathRow = path.data() + localRow * localColumns;
            const unsigned int distanceToK = pivotColumn[localRow];

            for (size_t localColumn = 0; localColumn < localColumns; ++localColumn) {
                const unsigned int newDistance = distanceToK + pivotRow[localColumn];
                if (newDistance < row[localColumn]) {
                    row[localColumn] = newDistance;
                    pathRow[localColumn] = static_cast<unsigned int>(k);
                }
            }
        }
    }
}

bool validateResult(const std::vector<unsigned int>& dist, const size_t numNodes) {
    for (size_t i = 0; i < numNodes; ++i) {
        if (dist[i * numNodes + i] != 0u) {
            printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", i, i);
            return false;
        }
    }

    for (size_t i = 0; i < std::min(numNodes, static_cast<size_t>(10)); ++i) {
        for (size_t j = 0; j < std::min(numNodes, static_cast<size_t>(10)); ++j) {
            for (size_t k = 0; k < numNodes; ++k) {
                const unsigned int distIJ = dist[i * numNodes + j];
                const unsigned int distIK = dist[i * numNodes + k];
                const unsigned int distKJ = dist[k * numNodes + j];

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

void gatherMatrix(const std::vector<unsigned int>& localDist,
                  std::vector<unsigned int>& fullDist,
                  const size_t numNodes,
                  const int gridRows,
                  const int gridColumns,
                  MPI_Comm communicator) {
    int rank;
    int worldSize;
    MPI_Comm_rank(communicator, &rank);
    MPI_Comm_size(communicator, &worldSize);

    const size_t localElementCount = localDist.size();
    if (localElementCount > static_cast<size_t>(INT_MAX) ||
        numNodes * numNodes > static_cast<size_t>(INT_MAX)) {
        if (rank == 0) {
            fprintf(stderr, "Matrix is too large for the MPI gather used by -r/-v\n");
        }
        MPI_Abort(communicator, 1);
    }

    std::vector<int> counts;
    std::vector<int> displacements;
    std::vector<unsigned int> packed;
    if (rank == 0) {
        counts.resize(static_cast<size_t>(worldSize));
        displacements.resize(static_cast<size_t>(worldSize));
        size_t displacement = 0u;
        for (int source = 0; source < worldSize; ++source) {
            const int sourceRow = source / gridColumns;
            const int sourceColumn = source % gridColumns;
            const size_t sourceRows = blockSize(numNodes, gridRows, sourceRow);
            const size_t sourceColumns = blockSize(numNodes, gridColumns, sourceColumn);
            const size_t sourceElements = sourceRows * sourceColumns;
            counts[static_cast<size_t>(source)] = static_cast<int>(sourceElements);
            displacements[static_cast<size_t>(source)] = static_cast<int>(displacement);
            displacement += sourceElements;
        }
        packed.resize(numNodes * numNodes);
        fullDist.resize(numNodes * numNodes);
    }

    MPI_Gatherv(localDist.data(), static_cast<int>(localElementCount), MPI_UNSIGNED,
                rank == 0 ? packed.data() : nullptr,
                rank == 0 ? counts.data() : nullptr,
                rank == 0 ? displacements.data() : nullptr,
                MPI_UNSIGNED, 0, communicator);

    if (rank == 0) {
        for (int source = 0; source < worldSize; ++source) {
            const int sourceRow = source / gridColumns;
            const int sourceColumn = source % gridColumns;
            const size_t sourceRowStart = blockStart(numNodes, gridRows, sourceRow);
            const size_t sourceColumnStart = blockStart(numNodes, gridColumns, sourceColumn);
            const size_t sourceRows = blockSize(numNodes, gridRows, sourceRow);
            const size_t sourceColumns = blockSize(numNodes, gridColumns, sourceColumn);
            const unsigned int* const sourceData =
                packed.data() + displacements[static_cast<size_t>(source)];

            for (size_t localRow = 0; localRow < sourceRows; ++localRow) {
                std::memcpy(fullDist.data() + (sourceRowStart + localRow) * numNodes + sourceColumnStart,
                            sourceData + localRow * sourceColumns,
                            sourceColumns * sizeof(unsigned int));
            }
        }
    }
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

    int rank;
    int worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    size_t numNodes = 512u;
    bool validate = false;
    bool printResults = false;
    bool parseError = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const unsigned long long parsed = std::strtoull(argv[++i], &end, 10);
            if (argv[i][0] == '\0' || argv[i][0] == '-' || *end != '\0' ||
                parsed > static_cast<unsigned long long>(std::numeric_limits<size_t>::max())) {
                parseError = true;
            } else {
                numNodes = static_cast<size_t>(parsed);
            }
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            if (rank == 0) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            parseError = true;
        }
    }

    if (parseError) {
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        fflush(stdout);
    }

    int dimensions[2] = {0, 0};
    MPI_Dims_create(worldSize, 2, dimensions);
    const int gridRows = dimensions[0];
    const int gridColumns = dimensions[1];
    const int rowCoordinate = rank / gridColumns;
    const int columnCoordinate = rank % gridColumns;

    MPI_Comm rowCommunicator;
    MPI_Comm columnCommunicator;
    MPI_Comm_split(MPI_COMM_WORLD, rowCoordinate, columnCoordinate, &rowCommunicator);
    MPI_Comm_split(MPI_COMM_WORLD, columnCoordinate, rowCoordinate, &columnCommunicator);

    const size_t rowStart = blockStart(numNodes, gridRows, rowCoordinate);
    const size_t localRows = blockSize(numNodes, gridRows, rowCoordinate);
    const size_t columnStart = blockStart(numNodes, gridColumns, columnCoordinate);
    const size_t localColumns = blockSize(numNodes, gridColumns, columnCoordinate);

    std::vector<unsigned int> dist(localRows * localColumns);
    std::vector<unsigned int> path(localRows * localColumns);

    if (rank == 0) {
        printf("Initializing graph...\n");
    }
    initializeDistanceTile(dist, numNodes, rowStart, localRows,
                           columnStart, localColumns, 1u, MAX_DISTANCE);
    initializePathTile(path, rowStart, localRows, localColumns);

    if (rank == 0) {
        printf("Computing shortest paths...\n");
        fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    floydWarshallTile(dist, path, numNodes, rowStart, localRows,
                      columnStart, localColumns, gridRows, gridColumns,
                      rowCoordinate, columnCoordinate,
                      rowCommunicator, columnCommunicator);

    MPI_Barrier(MPI_COMM_WORLD);
    const double localSeconds = MPI_Wtime() - start;
    double computationSeconds = 0.0;
    MPI_Reduce(&localSeconds, &computationSeconds, 1, MPI_DOUBLE, MPI_MAX,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        const long computationMilliseconds = static_cast<long>(computationSeconds * 1000.0);
        printf("Computation time: %ld ms\n", computationMilliseconds);

        const double operations = static_cast<double>(numNodes) *
                                  static_cast<double>(numNodes) *
                                  static_cast<double>(numNodes);
        const double gflops = computationSeconds > 0.0
                                  ? operations / computationSeconds / 1e9
                                  : 0.0;
        printf("Performance: %.3f GOPS\n", gflops);
    }

    std::vector<unsigned int> fullDist;
    if (printResults || validate) {
        gatherMatrix(dist, fullDist, numNodes, gridRows, gridColumns, MPI_COMM_WORLD);
    }

    int valid = 1;
    if (rank == 0) {
        if (printResults) {
            print_results_int(fullDist, "DistanceMatrix");
        }

        if (validate) {
            printf("Validating result...\n");
            valid = validateResult(fullDist, numNodes) ? 1 : 0;
            printf("Validation: %s\n", valid != 0 ? "PASSED" : "FAILED");
        }
    }

    if (validate) {
        MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    MPI_Comm_free(&rowCommunicator);
    MPI_Comm_free(&columnCommunicator);
    MPI_Finalize();
    return validate && valid == 0 ? 1 : 0;
}
