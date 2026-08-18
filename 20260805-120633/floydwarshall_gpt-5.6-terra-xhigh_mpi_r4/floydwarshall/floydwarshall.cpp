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
constexpr int RESULT_TRANSFER_TAG = 701;
constexpr int TRANSFER_CHUNK_ELEMENTS = 1 << 26;

struct Partition {
    size_t firstDestination;
    size_t destinationCount;
};

// The flattened matrix is destination-major: element (source, destination)
// resides at destination * numNodes + source.  Partitioning by destination
// therefore makes each process's portion contiguous and keeps every update
// local once the pivot row has been broadcast.
inline Partition partitionDestinations(const size_t numNodes, const int rank,
                                       const int processCount) noexcept {
    const size_t processes = static_cast<size_t>(processCount);
    const size_t baseCount = numNodes / processes;
    const size_t remainder = numNodes % processes;
    const size_t rankAsSize = static_cast<size_t>(rank);
    return {
        rankAsSize * baseCount + std::min(rankAsSize, remainder),
        baseCount + (rankAsSize < remainder ? 1U : 0U),
    };
}

inline int ownerOfDestination(const size_t destination, const size_t numNodes,
                              const int processCount) noexcept {
    const size_t processes = static_cast<size_t>(processCount);
    const size_t baseCount = numNodes / processes;
    const size_t remainder = numNodes % processes;
    const size_t largerPartitionElements = (baseCount + 1U) * remainder;

    if (destination < largerPartitionElements) {
        return static_cast<int>(destination / (baseCount + 1U));
    }

    return static_cast<int>(remainder +
                            (destination - largerPartitionElements) / baseCount);
}

void initializeDistanceMatrix(std::vector<unsigned int>& localDist,
                              const size_t numNodes,
                              const unsigned int rangeMin,
                              const unsigned int rangeMax,
                              const Partition partition, const int rank,
                              const int processCount, MPI_Comm communicator) {
    // rand_r is stateful.  Rank 0 computes just the starting state for each
    // contiguous segment, allowing every rank to generate its own segment
    // while producing exactly the same matrix as the original serial stream.
    std::vector<unsigned int> startingSeeds;
    if (rank == 0) {
        startingSeeds.resize(static_cast<size_t>(processCount));
        unsigned int seed = 42;
        for (int process = 0; process < processCount; ++process) {
            startingSeeds[static_cast<size_t>(process)] = seed;
            const Partition processPartition =
                partitionDestinations(numNodes, process, processCount);
            const size_t elements = processPartition.destinationCount * numNodes;
            for (size_t element = 0; element < elements; ++element) {
                (void)rand_r(&seed);
            }
        }
    }

    unsigned int localSeed = 0;
    MPI_Scatter(rank == 0 ? startingSeeds.data() : nullptr, 1, MPI_UNSIGNED,
                &localSeed, 1, MPI_UNSIGNED, 0, communicator);

    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;
    for (unsigned int& value : localDist) {
        value = rangeMin + static_cast<unsigned int>(
            range * rand_r(&localSeed) / static_cast<double>(RAND_MAX));
    }

    for (size_t localDestination = 0;
         localDestination < partition.destinationCount; ++localDestination) {
        const size_t destination = partition.firstDestination + localDestination;
        localDist[localDestination * numNodes + destination] = 0;
    }
}

void initializePathMatrix(std::vector<unsigned int>& localPath,
                          const size_t numNodes, const Partition partition) {
    for (size_t localDestination = 0;
         localDestination < partition.destinationCount; ++localDestination) {
        const unsigned int destination = static_cast<unsigned int>(
            partition.firstDestination + localDestination);
        unsigned int* const pathRow = localPath.data() + localDestination * numNodes;
        std::fill_n(pathRow, numNodes, destination);
    }
}

void floydWarshall(std::vector<unsigned int>& localDist,
                   std::vector<unsigned int>& localPath, const size_t numNodes,
                   const Partition partition, const int rank, const int processCount,
                   MPI_Comm communicator) {
    std::vector<unsigned int> receivedPivot(numNodes);

    for (size_t k = 0; k < numNodes; ++k) {
        const int pivotOwner = ownerOfDestination(k, numNodes, processCount);
        unsigned int* broadcastPivot = receivedPivot.data();
        if (pivotOwner == rank) {
            broadcastPivot = localDist.data() +
                (k - partition.firstDestination) * numNodes;
        }

        MPI_Bcast(broadcastPivot, static_cast<int>(numNodes), MPI_UNSIGNED,
                  pivotOwner, communicator);
        const unsigned int* const pivot = broadcastPivot;

        for (size_t localDestination = 0;
             localDestination < partition.destinationCount; ++localDestination) {
            const size_t destination = partition.firstDestination + localDestination;
            // The pivot row is unchanged for this iteration (dist[k][k] is
            // zero), so skipping it also removes the only possible alias with
            // pivot and enables the vectorized inner loop below.
            if (destination == k) {
                continue;
            }

            unsigned int* __restrict distanceRow =
                localDist.data() + localDestination * numNodes;
            unsigned int* __restrict pathRow =
                localPath.data() + localDestination * numNodes;
            const unsigned int distanceFromPivot = distanceRow[k];

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC ivdep
#endif
            for (size_t source = 0; source < numNodes; ++source) {
                const unsigned int newDistance = pivot[source] + distanceFromPivot;
                if (newDistance < distanceRow[source]) {
                    distanceRow[source] = newDistance;
                    pathRow[source] = static_cast<unsigned int>(k);
                }
            }
        }
    }
}

bool validateResult(const std::vector<unsigned int>& localDist,
                    const size_t numNodes, const Partition partition,
                    const int rank, const int processCount, MPI_Comm communicator) {
    const unsigned long long noFailure = std::numeric_limits<unsigned long long>::max();

    unsigned long long localDiagonalFailure = noFailure;
    for (size_t localDestination = 0;
         localDestination < partition.destinationCount; ++localDestination) {
        const size_t destination = partition.firstDestination + localDestination;
        if (localDist[localDestination * numNodes + destination] != 0) {
            localDiagonalFailure = static_cast<unsigned long long>(destination);
            break;
        }
    }

    unsigned long long diagonalFailure = noFailure;
    MPI_Allreduce(&localDiagonalFailure, &diagonalFailure, 1,
                  MPI_UNSIGNED_LONG_LONG, MPI_MIN, communicator);
    if (diagonalFailure != noFailure) {
        if (rank == 0) {
            const size_t node = static_cast<size_t>(diagonalFailure);
            printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", node, node);
        }
        return false;
    }

    const size_t sampleCount = std::min(numNodes, static_cast<size_t>(10));
    std::vector<unsigned int> receivedPivot(numNodes);
    unsigned long long localTriangleFailure = noFailure;

    // Broadcast the required dist[i][k] row once per k, then check the local
    // destinations among the same sampled set as the serial implementation.
    for (size_t k = 0; k < numNodes; ++k) {
        const int pivotOwner = ownerOfDestination(k, numNodes, processCount);
        unsigned int* broadcastPivot = receivedPivot.data();
        if (pivotOwner == rank) {
            broadcastPivot = const_cast<unsigned int*>(
                localDist.data() + (k - partition.firstDestination) * numNodes);
        }
        MPI_Bcast(broadcastPivot, static_cast<int>(numNodes), MPI_UNSIGNED,
                  pivotOwner, communicator);

        const unsigned int* const pivot = broadcastPivot;
        for (size_t localDestination = 0;
             localDestination < partition.destinationCount; ++localDestination) {
            const size_t destination = partition.firstDestination + localDestination;
            if (destination >= sampleCount) {
                continue;
            }

            const unsigned int* const distanceRow =
                localDist.data() + localDestination * numNodes;
            const unsigned int distanceFromPivot = distanceRow[k];
            for (size_t source = 0; source < sampleCount; ++source) {
                const unsigned int distanceToPivot = pivot[source];
                if (distanceToPivot < INF && distanceFromPivot < INF &&
                    distanceToPivot + distanceFromPivot < distanceRow[source]) {
                    const unsigned long long failure =
                        (static_cast<unsigned long long>(source) * sampleCount + destination) *
                            numNodes +
                        k;
                    localTriangleFailure = std::min(localTriangleFailure, failure);
                }
            }
        }
    }

    unsigned long long triangleFailure = noFailure;
    MPI_Allreduce(&localTriangleFailure, &triangleFailure, 1,
                  MPI_UNSIGNED_LONG_LONG, MPI_MIN, communicator);
    if (triangleFailure != noFailure) {
        const unsigned long long source =
            triangleFailure / (static_cast<unsigned long long>(sampleCount) * numNodes);
        const unsigned long long remainder =
            triangleFailure % (static_cast<unsigned long long>(sampleCount) * numNodes);
        const unsigned long long destination = remainder / numNodes;
        const unsigned long long k = remainder % numNodes;
        if (rank == 0) {
            printf("Validation failed: triangle inequality violated at [%llu,%llu,%llu]\n",
                   source, destination, k);
        }
        return false;
    }

    return true;
}

std::vector<unsigned int> gatherDistanceMatrix(
    const std::vector<unsigned int>& localDist, const size_t numNodes,
    const int rank, const int processCount, MPI_Comm communicator) {
    std::vector<unsigned int> globalDist;
    if (rank == 0) {
        globalDist.resize(numNodes * numNodes);
        std::copy(localDist.begin(), localDist.end(), globalDist.begin());

        for (int process = 1; process < processCount; ++process) {
            const Partition processPartition =
                partitionDestinations(numNodes, process, processCount);
            size_t remaining = processPartition.destinationCount * numNodes;
            size_t offset = processPartition.firstDestination * numNodes;
            while (remaining != 0) {
                const int count = static_cast<int>(std::min(
                    remaining, static_cast<size_t>(TRANSFER_CHUNK_ELEMENTS)));
                MPI_Recv(globalDist.data() + offset, count, MPI_UNSIGNED, process,
                         RESULT_TRANSFER_TAG, communicator, MPI_STATUS_IGNORE);
                offset += static_cast<size_t>(count);
                remaining -= static_cast<size_t>(count);
            }
        }
    } else {
        size_t remaining = localDist.size();
        size_t offset = 0;
        while (remaining != 0) {
            const int count = static_cast<int>(std::min(
                remaining, static_cast<size_t>(TRANSFER_CHUNK_ELEMENTS)));
            MPI_Send(localDist.data() + offset, count, MPI_UNSIGNED, 0,
                     RESULT_TRANSFER_TAG, communicator);
            offset += static_cast<size_t>(count);
            remaining -= static_cast<size_t>(count);
        }
    }
    return globalDist;
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

    unsigned long long numNodesValue = 512;
    bool validate = false;
    bool printResults = false;
    int runState = 0;  // 0: run, 1: help, 2: command-line error

    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                numNodesValue = static_cast<unsigned long long>(atoi(argv[++i]));
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = true;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = true;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                runState = 1;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                runState = 2;
                break;
            }
        }
    }

    int options[2] = {validate ? 1 : 0, printResults ? 1 : 0};
    MPI_Bcast(&runState, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&numNodesValue, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(options, 2, MPI_INT, 0, MPI_COMM_WORLD);
    if (runState != 0) {
        MPI_Finalize();
        return runState == 1 ? 0 : 1;
    }

    const size_t numNodes = static_cast<size_t>(numNodesValue);
    validate = options[0] != 0;
    printResults = options[1] != 0;

    const bool validDimension =
        numNodesValue <= static_cast<unsigned long long>(std::numeric_limits<int>::max()) &&
        (numNodes == 0 || numNodes <= std::numeric_limits<size_t>::max() / numNodes);
    if (!validDimension) {
        if (rank == 0) {
            printf("Number of nodes is too large for this MPI implementation\n");
        }
        MPI_Finalize();
        return 1;
    }

    const Partition partition = partitionDestinations(numNodes, rank, processCount);
    const size_t localElements = partition.destinationCount * numNodes;

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing graph...\n");
    }

    std::vector<unsigned int> localDist(localElements);
    std::vector<unsigned int> localPath(localElements);
    initializeDistanceMatrix(localDist, numNodes, 1, MAX_DISTANCE, partition,
                             rank, processCount, MPI_COMM_WORLD);
    initializePathMatrix(localPath, numNodes, partition);

    if (rank == 0) {
        printf("Computing shortest paths...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    floydWarshall(localDist, localPath, numNodes, partition, rank, processCount,
                  MPI_COMM_WORLD);

    const double localElapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const long durationMilliseconds = static_cast<long>(elapsed * 1000.0);
        printf("Computation time: %ld ms\n", durationMilliseconds);

        // Floyd-Warshall has O(n^3) complexity.  Use the slowest rank's time,
        // which is the elapsed time for the distributed computation as a whole.
        const double operations = static_cast<double>(numNodes) * numNodes * numNodes;
        const double gflops = operations / (durationMilliseconds / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);
    }

    if (printResults) {
        const std::vector<unsigned int> globalDist =
            gatherDistanceMatrix(localDist, numNodes, rank, processCount, MPI_COMM_WORLD);
        if (rank == 0) {
            print_results_int(globalDist, "DistanceMatrix");
        }
    }

    int exitCode = 0;
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
        }
        const bool valid = validateResult(localDist, numNodes, partition, rank,
                                          processCount, MPI_COMM_WORLD);
        if (rank == 0) {
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
