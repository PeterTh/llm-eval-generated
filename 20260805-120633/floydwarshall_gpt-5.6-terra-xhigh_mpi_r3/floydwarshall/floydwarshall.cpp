#include <algorithm>
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

// The benchmark stores a matrix as consecutive source-node columns.  This
// helper retains the original (destination, source) argument order.
inline constexpr size_t idx2(const size_t destination, const size_t source,
                             const size_t numNodes) noexcept {
    return source * numNodes + destination;
}

size_t blockStart(const size_t total, const int coordinate,
                  const int dimensions) noexcept {
    return static_cast<size_t>(coordinate) * total /
           static_cast<size_t>(dimensions);
}

size_t blockSize(const size_t total, const int coordinate,
                 const int dimensions) noexcept {
    return blockStart(total, coordinate + 1, dimensions) -
           blockStart(total, coordinate, dimensions);
}

int blockOwner(const size_t index, const size_t total,
               const int dimensions) noexcept {
    // This also handles grids with more coordinates than graph nodes, where
    // some blocks are empty.
    return static_cast<int>(((index + 1) * static_cast<size_t>(dimensions) - 1) /
                            total);
}

int mpiCount(const size_t count) {
    if (count > static_cast<size_t>(std::numeric_limits<int>::max())) {
        fprintf(stderr, "MPI message size exceeds the supported count range\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    return static_cast<int>(count);
}

void initializeDistanceMatrix(std::vector<unsigned int>& dist,
                              const size_t numNodes,
                              const unsigned int rangeMin,
                              const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;

    for (size_t i = 0; i < numNodes * numNodes; ++i) {
        dist[i] = rangeMin +
                  static_cast<unsigned int>(range * rand_r(&seed) /
                                            static_cast<double>(RAND_MAX));
    }

    for (size_t i = 0; i < numNodes; ++i) {
        dist[idx2(i, i, numNodes)] = 0;
    }
}

void initializeLocalPathMatrix(std::vector<unsigned int>& path,
                               const size_t localDestinations,
                               const size_t sourceStart,
                               const size_t localSources) {
    for (size_t source = 0; source < localSources; ++source) {
        unsigned int* const column = path.data() + source * localDestinations;
        for (size_t destination = 0; destination < localDestinations; ++destination) {
            column[destination] = static_cast<unsigned int>(sourceStart + source);
        }
    }
}

// Rank zero creates the identical deterministic input produced by the serial
// program, then sends each rank one compact, column-major tile.  This work is
// intentionally outside the timed region.
void distributeDistanceMatrix(std::vector<unsigned int>& localDistance,
                              const size_t numNodes,
                              const int rank, const int worldSize,
                              const int gridRows, const int gridColumns,
                              const size_t destinationStart,
                              const size_t localDestinations,
                              const size_t sourceStart,
                              const size_t localSources) {
    constexpr int initializationTag = 101;

    if (rank == 0) {
        std::vector<unsigned int> globalDistance(numNodes * numNodes);
        initializeDistanceMatrix(globalDistance, numNodes, 1, MAX_DISTANCE);

        std::vector<unsigned int> sendBuffer;
        for (int target = 0; target < worldSize; ++target) {
            const int targetRow = target / gridColumns;
            const int targetColumn = target % gridColumns;
            const size_t targetDestinationStart =
                blockStart(numNodes, targetRow, gridRows);
            const size_t targetDestinations =
                blockSize(numNodes, targetRow, gridRows);
            const size_t targetSourceStart =
                blockStart(numNodes, targetColumn, gridColumns);
            const size_t targetSources =
                blockSize(numNodes, targetColumn, gridColumns);
            const size_t targetElements = targetDestinations * targetSources;

            if (target == 0) {
                for (size_t source = 0; source < localSources; ++source) {
                    const unsigned int* const globalColumn =
                        globalDistance.data() + (sourceStart + source) * numNodes +
                        destinationStart;
                    unsigned int* const localColumn =
                        localDistance.data() + source * localDestinations;
                    std::copy_n(globalColumn, localDestinations, localColumn);
                }
                continue;
            }

            sendBuffer.resize(targetElements);
            for (size_t source = 0; source < targetSources; ++source) {
                const unsigned int* const globalColumn =
                    globalDistance.data() +
                    (targetSourceStart + source) * numNodes + targetDestinationStart;
                unsigned int* const localColumn =
                    sendBuffer.data() + source * targetDestinations;
                std::copy_n(globalColumn, targetDestinations, localColumn);
            }
            MPI_Send(sendBuffer.data(), mpiCount(targetElements), MPI_UNSIGNED, target,
                     initializationTag, MPI_COMM_WORLD);
        }
    } else {
        const size_t localElements = localDestinations * localSources;
        MPI_Recv(localDistance.data(), mpiCount(localElements), MPI_UNSIGNED, 0,
                 initializationTag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

// In a 2D block distribution, a rank owns D[destination][source].  At step k
// it needs D[k][local sources] and D[local destinations][k].  Broadcasting
// those two pivot segments over the column and row communicators respectively
// makes the update fully local while keeping the full matrix distributed.
void floydWarshallMPI(std::vector<unsigned int>& distance,
                      std::vector<unsigned int>& path,
                      const size_t numNodes,
                      const int gridRows, const int gridColumns,
                      const int processRow, const int processColumn,
                      const size_t destinationStart,
                      const size_t localDestinations,
                      const size_t sourceStart, const size_t localSources,
                      MPI_Comm rowComm, MPI_Comm columnComm) {
    std::vector<unsigned int> distanceToPivot(localSources);
    std::vector<unsigned int> distanceFromPivot(localDestinations);

    for (size_t k = 0; k < numNodes; ++k) {
        const int pivotRow = blockOwner(k, numNodes, gridRows);
        const int pivotColumn = blockOwner(k, numNodes, gridColumns);

        if (processRow == pivotRow) {
            const size_t pivotDestination = k - destinationStart;
            for (size_t source = 0; source < localSources; ++source) {
                distanceToPivot[source] =
                    distance[source * localDestinations + pivotDestination];
            }
        }
        MPI_Bcast(distanceToPivot.data(), mpiCount(localSources), MPI_UNSIGNED,
                  pivotRow, columnComm);

        if (processColumn == pivotColumn) {
            const size_t pivotSource = k - sourceStart;
            const unsigned int* const pivotColumnData =
                distance.data() + pivotSource * localDestinations;
            std::copy_n(pivotColumnData, localDestinations,
                        distanceFromPivot.data());
        }
        MPI_Bcast(distanceFromPivot.data(), mpiCount(localDestinations), MPI_UNSIGNED,
                  pivotColumn, rowComm);

        for (size_t source = 0; source < localSources; ++source) {
            const unsigned int distanceSourceToPivot = distanceToPivot[source];
            unsigned int* const distanceColumn =
                distance.data() + source * localDestinations;
            unsigned int* const pathColumn = path.data() + source * localDestinations;

            // Both columns are contiguous.  Compilers can vectorize this
            // min-plus relaxation, while path writes remain conditional to
            // retain the original tie-breaking semantics.
            for (size_t destination = 0; destination < localDestinations;
                 ++destination) {
                const unsigned int newDistance =
                    distanceSourceToPivot + distanceFromPivot[destination];
                if (newDistance < distanceColumn[destination]) {
                    distanceColumn[destination] = newDistance;
                    pathColumn[destination] = static_cast<unsigned int>(k);
                }
            }
        }
    }
}

void gatherDistanceMatrix(const std::vector<unsigned int>& localDistance,
                          std::vector<unsigned int>& globalDistance,
                          const size_t numNodes, const int rank,
                          const int worldSize, const int gridRows,
                          const int gridColumns,
                          const size_t destinationStart,
                          const size_t localDestinations,
                          const size_t sourceStart,
                          const size_t localSources) {
    constexpr int gatheringTag = 102;
    const size_t localElements = localDestinations * localSources;

    if (rank != 0) {
        MPI_Send(localDistance.data(), mpiCount(localElements), MPI_UNSIGNED, 0,
                 gatheringTag, MPI_COMM_WORLD);
        return;
    }

    for (size_t source = 0; source < localSources; ++source) {
        const unsigned int* const localColumn =
            localDistance.data() + source * localDestinations;
        unsigned int* const globalColumn =
            globalDistance.data() + (sourceStart + source) * numNodes + destinationStart;
        std::copy_n(localColumn, localDestinations, globalColumn);
    }

    std::vector<unsigned int> receiveBuffer;
    for (int sender = 1; sender < worldSize; ++sender) {
        const int senderRow = sender / gridColumns;
        const int senderColumn = sender % gridColumns;
        const size_t senderDestinationStart =
            blockStart(numNodes, senderRow, gridRows);
        const size_t senderDestinations =
            blockSize(numNodes, senderRow, gridRows);
        const size_t senderSourceStart =
            blockStart(numNodes, senderColumn, gridColumns);
        const size_t senderSources =
            blockSize(numNodes, senderColumn, gridColumns);
        const size_t senderElements = senderDestinations * senderSources;

        receiveBuffer.resize(senderElements);
        MPI_Recv(receiveBuffer.data(), mpiCount(senderElements), MPI_UNSIGNED, sender,
                 gatheringTag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (size_t source = 0; source < senderSources; ++source) {
            const unsigned int* const receivedColumn =
                receiveBuffer.data() + source * senderDestinations;
            unsigned int* const globalColumn =
                globalDistance.data() +
                (senderSourceStart + source) * numNodes + senderDestinationStart;
            std::copy_n(receivedColumn, senderDestinations, globalColumn);
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

    int rank = 0;
    int worldSize = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;
    int parseStatus = 0;

    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                numNodes = static_cast<size_t>(atoi(argv[++i]));
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = true;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = true;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                parseStatus = 1;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                parseStatus = 2;
                break;
            }
        }
    }

    uint64_t packedNodes = static_cast<uint64_t>(numNodes);
    int options[3] = {parseStatus, static_cast<int>(validate),
                      static_cast<int>(printResults)};
    MPI_Bcast(&packedNodes, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(options, 3, MPI_INT, 0, MPI_COMM_WORLD);
    numNodes = static_cast<size_t>(packedNodes);
    parseStatus = options[0];
    validate = options[1] != 0;
    printResults = options[2] != 0;

    if (parseStatus != 0) {
        MPI_Finalize();
        return parseStatus == 1 ? 0 : 1;
    }

    int gridDimensions[2] = {0, 0};
    MPI_Dims_create(worldSize, 2, gridDimensions);
    const int gridRows = gridDimensions[0];
    const int gridColumns = gridDimensions[1];
    const int processRow = rank / gridColumns;
    const int processColumn = rank % gridColumns;

    MPI_Comm rowComm = MPI_COMM_NULL;
    MPI_Comm columnComm = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, processRow, processColumn, &rowComm);
    MPI_Comm_split(MPI_COMM_WORLD, processColumn, processRow, &columnComm);

    const size_t destinationStart = blockStart(numNodes, processRow, gridRows);
    const size_t localDestinations = blockSize(numNodes, processRow, gridRows);
    const size_t sourceStart = blockStart(numNodes, processColumn, gridColumns);
    const size_t localSources = blockSize(numNodes, processColumn, gridColumns);
    const size_t localElements = localDestinations * localSources;

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI processes: %d (%d x %d process grid)\n", worldSize, gridRows,
               gridColumns);
        printf("Initializing graph...\n");
    }

    std::vector<unsigned int> distance(localElements);
    std::vector<unsigned int> path(localElements);
    distributeDistanceMatrix(distance, numNodes, rank, worldSize, gridRows,
                             gridColumns, destinationStart, localDestinations,
                             sourceStart, localSources);
    initializeLocalPathMatrix(path, localDestinations, sourceStart, localSources);

    if (rank == 0) {
        printf("Computing shortest paths...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    floydWarshallMPI(distance, path, numNodes, gridRows, gridColumns, processRow,
                     processColumn, destinationStart, localDestinations,
                     sourceStart, localSources, rowComm, columnComm);
    const double localDuration = MPI_Wtime() - start;

    double duration = 0.0;
    MPI_Reduce(&localDuration, &duration, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);
    if (rank == 0) {
        const long long milliseconds = static_cast<long long>(duration * 1000.0);
        printf("Computation time: %lld ms\n", milliseconds);
        const double operations = static_cast<double>(numNodes) * numNodes * numNodes;
        const double gops = operations / duration / 1e9;
        printf("Performance: %.3f GOPS\n", gops);
    }

    const bool needGlobalDistance = printResults || validate;
    std::vector<unsigned int> globalDistance;
    if (needGlobalDistance && rank == 0) {
        globalDistance.resize(numNodes * numNodes);
    }
    if (needGlobalDistance) {
        gatherDistanceMatrix(distance, globalDistance, numNodes, rank, worldSize,
                             gridRows, gridColumns, destinationStart,
                             localDestinations, sourceStart, localSources);
    }

    int exitStatus = 0;
    if (rank == 0 && printResults) {
        print_results_int(globalDistance, "DistanceMatrix");
    }
    if (rank == 0 && validate) {
        printf("Validating result...\n");
        if (validateResult(globalDistance, numNodes)) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            exitStatus = 1;
        }
    }

    MPI_Bcast(&exitStatus, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Comm_free(&rowComm);
    MPI_Comm_free(&columnComm);
    MPI_Finalize();
    return exitStatus;
}
