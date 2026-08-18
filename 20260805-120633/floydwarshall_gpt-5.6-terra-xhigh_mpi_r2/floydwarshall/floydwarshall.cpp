#include <algorithm>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;
constexpr int INITIALIZATION_TAG = 0;
constexpr int COLLECTION_TAG = 1;

static_assert(sizeof(unsigned int) == sizeof(std::uint32_t),
              "This benchmark requires 32-bit unsigned int values");

// This is the original column-major index used by result printing and
// validation.  Distributed local blocks use row-major layout for their hot
// update loop, where every destination in a local block is contiguous.
inline constexpr size_t idx2(const size_t source, const size_t destination,
                             const size_t numNodes) noexcept {
    return destination * numNodes + source;
}

struct AxisPartition {
    size_t first;
    size_t count;
};

AxisPartition partitionAxis(const size_t numNodes, const int partition,
                            const int partitionCount) noexcept {
    const size_t partitions = static_cast<size_t>(partitionCount);
    const size_t base = numNodes / partitions;
    const size_t extra = numNodes % partitions;
    const size_t index = static_cast<size_t>(partition);
    return {index * base + std::min(index, extra),
            base + (index < extra ? 1U : 0U)};
}

int ownerOfIndex(const size_t index, const size_t numNodes,
                 const int partitionCount) noexcept {
    const size_t partitions = static_cast<size_t>(partitionCount);
    const size_t base = numNodes / partitions;
    const size_t extra = numNodes % partitions;
    const size_t largerPartitions = (base + 1U) * extra;

    if (index < largerPartitions) {
        return static_cast<int>(index / (base + 1U));
    }
    return static_cast<int>(extra + (index - largerPartitions) / base);
}

int chooseProcessRows(const int processCount) noexcept {
    for (int candidate = static_cast<int>(std::sqrt(processCount));
         candidate >= 1; --candidate) {
        if (processCount % candidate == 0) {
            return candidate;
        }
    }
    return 1;
}

struct ProcessGrid {
    int rows;
    int columns;
    int row;
    int column;
    MPI_Comm rowCommunicator;
    MPI_Comm columnCommunicator;
};

ProcessGrid makeProcessGrid(const int rank, const int processCount) {
    const int gridRows = chooseProcessRows(processCount);
    const int gridColumns = processCount / gridRows;
    const int gridRow = rank / gridColumns;
    const int gridColumn = rank % gridColumns;

    ProcessGrid grid{gridRows, gridColumns, gridRow, gridColumn,
                     MPI_COMM_NULL, MPI_COMM_NULL};
    MPI_Comm_split(MPI_COMM_WORLD, gridRow, gridColumn, &grid.rowCommunicator);
    MPI_Comm_split(MPI_COMM_WORLD, gridColumn, gridRow,
                   &grid.columnCommunicator);
    return grid;
}

// glibc rand_r advances its 32-bit seed through three instances of this LCG
// per generated number.  Affine exponentiation lets every rank begin at its
// exact segment of the original rand_r stream without root-side matrix setup.
struct AffineTransform {
    std::uint32_t multiplier;
    std::uint32_t increment;
};

constexpr AffineTransform LCG_STEP{1103515245U, 12345U};

AffineTransform compose(const AffineTransform outer,
                        const AffineTransform inner) noexcept {
    return {
        static_cast<std::uint32_t>(
            static_cast<std::uint64_t>(outer.multiplier) * inner.multiplier),
        static_cast<std::uint32_t>(
            static_cast<std::uint64_t>(outer.multiplier) * inner.increment +
            outer.increment)};
}

std::uint32_t apply(const AffineTransform transform,
                    const std::uint32_t state) noexcept {
    return static_cast<std::uint32_t>(
        static_cast<std::uint64_t>(transform.multiplier) * state +
        transform.increment);
}

AffineTransform power(AffineTransform transform, size_t exponent) noexcept {
    AffineTransform result{1U, 0U};
    while (exponent != 0U) {
        if ((exponent & 1U) != 0U) {
            result = compose(transform, result);
        }
        transform = compose(transform, transform);
        exponent >>= 1U;
    }
    return result;
}

AffineTransform randRAdvance(const size_t calls) noexcept {
    const AffineTransform oneCall =
        compose(LCG_STEP, compose(LCG_STEP, LCG_STEP));
    return power(oneCall, calls);
}

bool canJumpRandRState() noexcept {
    const AffineTransform oneCall = randRAdvance(1U);
    std::uint32_t predictedState = 0x6d2b79f5U;
    unsigned int actualState = predictedState;

    for (int iteration = 0; iteration < 4; ++iteration) {
        (void)rand_r(&actualState);
        predictedState = apply(oneCall, predictedState);
        if (actualState != predictedState) {
            return false;
        }
    }
    return true;
}

void initializeDistanceMatrixFallback(
    std::vector<unsigned int>& localDist, const size_t numNodes,
    const ProcessGrid& grid, const int rank, const int processCount) {
    if (rank == 0) {
        std::vector<std::vector<unsigned int>> blocks(
            static_cast<size_t>(processCount));
        for (int process = 0; process < processCount; ++process) {
            const int processRow = process / grid.columns;
            const int processColumn = process % grid.columns;
            const AxisPartition processSources =
                partitionAxis(numNodes, processRow, grid.rows);
            const AxisPartition processDestinations =
                partitionAxis(numNodes, processColumn, grid.columns);
            blocks[static_cast<size_t>(process)].resize(
                processSources.count * processDestinations.count);
        }

        unsigned int seed = 42;
        const double range = static_cast<double>(MAX_DISTANCE);
        for (size_t destination = 0; destination < numNodes; ++destination) {
            const int processColumn =
                ownerOfIndex(destination, numNodes, grid.columns);
            const AxisPartition destinationPartition =
                partitionAxis(numNodes, processColumn, grid.columns);
            const size_t localDestination = destination - destinationPartition.first;

            for (size_t source = 0; source < numNodes; ++source) {
                const int processRow = ownerOfIndex(source, numNodes, grid.rows);
                const AxisPartition sourcePartition =
                    partitionAxis(numNodes, processRow, grid.rows);
                const size_t localSource = source - sourcePartition.first;
                const int process = processRow * grid.columns + processColumn;
                const unsigned int value =
                    1U + static_cast<unsigned int>(range * rand_r(&seed) /
                                                    static_cast<double>(RAND_MAX));
                blocks[static_cast<size_t>(process)]
                    [localSource * destinationPartition.count + localDestination] =
                    source == destination ? 0U : value;
            }
        }

        localDist = std::move(blocks[0]);
        for (int process = 1; process < processCount; ++process) {
            const std::vector<unsigned int>& block =
                blocks[static_cast<size_t>(process)];
            MPI_Send(block.empty() ? nullptr : block.data(),
                     static_cast<int>(block.size()), MPI_UNSIGNED, process,
                     INITIALIZATION_TAG, MPI_COMM_WORLD);
        }
    } else {
        MPI_Recv(localDist.empty() ? nullptr : localDist.data(),
                 static_cast<int>(localDist.size()), MPI_UNSIGNED, 0,
                 INITIALIZATION_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

void initializeDistanceMatrix(std::vector<unsigned int>& localDist,
                              const size_t numNodes,
                              const AxisPartition sourceRows,
                              const AxisPartition destinationColumns,
                              const ProcessGrid& grid, const int rank,
                              const int processCount) {
    int canJumpLocally = canJumpRandRState() ? 1 : 0;
    int canJumpEverywhere = 0;
    MPI_Allreduce(&canJumpLocally, &canJumpEverywhere, 1, MPI_INT, MPI_MIN,
                  MPI_COMM_WORLD);

    if (canJumpEverywhere == 0) {
        // This rare compatibility path retains exact rand_r results even on a
        // C library whose state transition cannot be safely jumped ahead.
        initializeDistanceMatrixFallback(localDist, numNodes, grid, rank,
                                         processCount);
        return;
    }

    const double range = static_cast<double>(MAX_DISTANCE);
    const size_t firstStreamElement =
        destinationColumns.first * numNodes + sourceRows.first;
    unsigned int columnSeed = apply(randRAdvance(firstStreamElement), 42U);
    const AffineTransform skipToNextDestination =
        randRAdvance(numNodes - sourceRows.count);

    for (size_t localDestination = 0;
         localDestination < destinationColumns.count; ++localDestination) {
        const size_t destination = destinationColumns.first + localDestination;
        unsigned int seed = columnSeed;
        for (size_t localSource = 0; localSource < sourceRows.count;
             ++localSource) {
            const size_t source = sourceRows.first + localSource;
            const unsigned int value =
                1U + static_cast<unsigned int>(range * rand_r(&seed) /
                                                static_cast<double>(RAND_MAX));
            localDist[localSource * destinationColumns.count + localDestination] =
                source == destination ? 0U : value;
        }
        columnSeed = apply(skipToNextDestination, seed);
    }
}

void initializePathMatrix(std::vector<unsigned int>& localPath,
                          const AxisPartition sourceRows,
                          const AxisPartition destinationColumns) {
    if (destinationColumns.count == 0) {
        return;
    }

    for (size_t localSource = 0; localSource < sourceRows.count;
         ++localSource) {
        unsigned int* const pathRow =
            localPath.data() + localSource * destinationColumns.count;
        for (size_t localDestination = 0;
             localDestination < destinationColumns.count; ++localDestination) {
            pathRow[localDestination] = static_cast<unsigned int>(
                destinationColumns.first + localDestination);
        }
    }
}

void floydWarshall(std::vector<unsigned int>& localDist,
                   std::vector<unsigned int>& localPath,
                   const size_t numNodes,
                   const AxisPartition sourceRows,
                   const AxisPartition destinationColumns,
                   const ProcessGrid& grid) {
    std::vector<unsigned int> pivotColumn(sourceRows.count);
    std::vector<unsigned int> receivedPivotRow(destinationColumns.count);

    for (size_t pivot = 0; pivot < numNodes; ++pivot) {
        const int pivotSourceOwner = ownerOfIndex(pivot, numNodes, grid.rows);
        const int pivotDestinationOwner =
            ownerOfIndex(pivot, numNodes, grid.columns);

        const unsigned int* pivotValues = nullptr;
        if (grid.rows == 1) {
            pivotValues = destinationColumns.count == 0
                ? receivedPivotRow.data()
                : localDist.data() +
                  (pivot - sourceRows.first) * destinationColumns.count;
        } else {
            if (grid.row == pivotSourceOwner &&
                destinationColumns.count != 0) {
                const unsigned int* const localPivot =
                    localDist.data() +
                    (pivot - sourceRows.first) * destinationColumns.count;
                std::copy_n(localPivot, destinationColumns.count,
                            receivedPivotRow.data());
            }
            MPI_Bcast(receivedPivotRow.empty() ? nullptr : receivedPivotRow.data(),
                      static_cast<int>(destinationColumns.count), MPI_UNSIGNED,
                      pivotSourceOwner, grid.columnCommunicator);
            pivotValues = receivedPivotRow.data();
        }

        const bool pivotColumnIsLocal = grid.columns == 1;
        if (!pivotColumnIsLocal) {
            if (grid.column == pivotDestinationOwner) {
                const size_t localPivot = pivot - destinationColumns.first;
                for (size_t localSource = 0; localSource < sourceRows.count;
                     ++localSource) {
                    pivotColumn[localSource] =
                        localDist[localSource * destinationColumns.count + localPivot];
                }
            }
            MPI_Bcast(pivotColumn.empty() ? nullptr : pivotColumn.data(),
                      static_cast<int>(sourceRows.count), MPI_UNSIGNED,
                      pivotDestinationOwner, grid.rowCommunicator);
        }

        if (destinationColumns.count == 0) {
            continue;
        }

        const unsigned int* const __restrict pivotRow = pivotValues;
        for (size_t localSource = 0; localSource < sourceRows.count;
             ++localSource) {
            if (sourceRows.first + localSource == pivot) {
                continue;
            }

            unsigned int* const __restrict distanceRow =
                localDist.data() + localSource * destinationColumns.count;
            unsigned int* const __restrict pathRow =
                localPath.data() + localSource * destinationColumns.count;
            const unsigned int distanceToPivot = pivotColumnIsLocal
                ? distanceRow[pivot - destinationColumns.first]
                : pivotColumn[localSource];

            for (size_t localDestination = 0;
                 localDestination < destinationColumns.count;
                 ++localDestination) {
                const unsigned int newDistance =
                    distanceToPivot + pivotRow[localDestination];
                if (newDistance < distanceRow[localDestination]) {
                    distanceRow[localDestination] = newDistance;
                    pathRow[localDestination] = static_cast<unsigned int>(pivot);
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
                const unsigned int distIJ = dist[idx2(i, j, numNodes)];
                const unsigned int distIK = dist[idx2(i, k, numNodes)];
                const unsigned int distKJ = dist[idx2(k, j, numNodes)];
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

void unpackBlock(std::vector<unsigned int>& globalDist,
                 const unsigned int* const block,
                 const AxisPartition sourceRows,
                 const AxisPartition destinationColumns,
                 const size_t numNodes) {
    for (size_t localSource = 0; localSource < sourceRows.count;
         ++localSource) {
        for (size_t localDestination = 0;
             localDestination < destinationColumns.count; ++localDestination) {
            globalDist[idx2(sourceRows.first + localSource,
                            destinationColumns.first + localDestination,
                            numNodes)] =
                block[localSource * destinationColumns.count + localDestination];
        }
    }
}

void collectDistanceMatrix(const std::vector<unsigned int>& localDist,
                           std::vector<unsigned int>& globalDist,
                           const size_t numNodes,
                           const AxisPartition sourceRows,
                           const AxisPartition destinationColumns,
                           const ProcessGrid& grid, const int rank,
                           const int processCount) {
    if (rank == 0) {
        globalDist.resize(numNodes * numNodes);
        unpackBlock(globalDist, localDist.data(), sourceRows, destinationColumns,
                    numNodes);

        std::vector<unsigned int> receiveBuffer;
        for (int process = 1; process < processCount; ++process) {
            const int processRow = process / grid.columns;
            const int processColumn = process % grid.columns;
            const AxisPartition processSources =
                partitionAxis(numNodes, processRow, grid.rows);
            const AxisPartition processDestinations =
                partitionAxis(numNodes, processColumn, grid.columns);
            receiveBuffer.resize(processSources.count * processDestinations.count);
            MPI_Recv(receiveBuffer.empty() ? nullptr : receiveBuffer.data(),
                     static_cast<int>(receiveBuffer.size()), MPI_UNSIGNED, process,
                     COLLECTION_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            unpackBlock(globalDist, receiveBuffer.data(), processSources,
                        processDestinations, numNodes);
        }
    } else {
        MPI_Send(localDist.empty() ? nullptr : localDist.data(),
                 static_cast<int>(localDist.size()), MPI_UNSIGNED, 0,
                 COLLECTION_TAG, MPI_COMM_WORLD);
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

    int rank = 0;
    int processCount = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processCount);

    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;
    bool showHelp = false;
    bool validArguments = true;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const char* const argument = argv[++i];
            const unsigned long long parsed = strtoull(argument, &end, 10);
            if (argument[0] == '-' || end == argument || *end != '\0' ||
                parsed > std::numeric_limits<size_t>::max()) {
                validArguments = false;
            } else {
                numNodes = static_cast<size_t>(parsed);
            }
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            showHelp = true;
        } else {
            validArguments = false;
        }
    }

    if (showHelp) {
        if (rank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return 0;
    }

    const bool validSize =
        numNodes <= static_cast<size_t>(INT_MAX) &&
        (numNodes == 0 || numNodes <= std::numeric_limits<size_t>::max() / numNodes);
    if (!validArguments || !validSize) {
        if (rank == 0) {
            printf(!validArguments ? "Invalid command line arguments\n"
                                   : "Number of nodes is outside the supported range\n");
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    ProcessGrid grid = makeProcessGrid(rank, processCount);
    const AxisPartition sourceRows =
        partitionAxis(numNodes, grid.row, grid.rows);
    const AxisPartition destinationColumns =
        partitionAxis(numNodes, grid.column, grid.columns);
    const size_t localElements = sourceRows.count * destinationColumns.count;

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("MPI processes: %d (%d x %d grid)\n", processCount, grid.rows,
               grid.columns);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing graph...\n");
    }

    std::vector<unsigned int> localDist(localElements);
    std::vector<unsigned int> localPath(localElements);
    initializeDistanceMatrix(localDist, numNodes, sourceRows, destinationColumns,
                             grid, rank, processCount);
    initializePathMatrix(localPath, sourceRows, destinationColumns);

    if (rank == 0) {
        printf("Computing shortest paths...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    floydWarshall(localDist, localPath, numNodes, sourceRows,
                  destinationColumns, grid);
    const double localSeconds = MPI_Wtime() - start;

    double elapsedSeconds = 0.0;
    MPI_Reduce(&localSeconds, &elapsedSeconds, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);

    if (rank == 0) {
        const long durationMilliseconds =
            static_cast<long>(elapsedSeconds * 1000.0);
        const double operations = static_cast<double>(numNodes) * numNodes * numNodes;
        const double gops = elapsedSeconds > 0.0
                                ? operations / elapsedSeconds / 1.0e9
                                : std::numeric_limits<double>::infinity();
        printf("Computation time: %ld ms\n", durationMilliseconds);
        printf("Performance: %.3f GOPS\n", gops);
    }

    int exitCode = 0;
    if (printResults || validate) {
        const size_t globalElements = numNodes * numNodes;
        if (globalElements > static_cast<size_t>(INT_MAX)) {
            if (rank == 0) {
                printf("Result collection requires at most %d matrix elements\n", INT_MAX);
            }
            exitCode = 1;
        } else {
            std::vector<unsigned int> gatheredDist;
            collectDistanceMatrix(localDist, gatheredDist, numNodes, sourceRows,
                                  destinationColumns, grid, rank, processCount);
            if (rank == 0) {
                if (printResults) {
                    print_results_int(gatheredDist, "DistanceMatrix");
                }
                if (validate) {
                    printf("Validating result...\n");
                    exitCode = validateResult(gatheredDist, numNodes) ? 0 : 1;
                    printf("Validation: %s\n", exitCode == 0 ? "PASSED" : "FAILED");
                }
            }
        }
    }

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Comm_free(&grid.rowCommunicator);
    MPI_Comm_free(&grid.columnCommunicator);
    MPI_Finalize();
    return exitCode;
}
