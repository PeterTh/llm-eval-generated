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

// Index calculation for the destination-major format used by the original
// benchmark and its result reporting code.
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

// Build source-major rows while preserving the original random-number call
// order: destination is the outer loop and source is the inner loop.
void initializeDistanceRows(std::vector<unsigned int>& dist, const size_t numNodes,
                            const unsigned int rangeMin, const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;

    for (size_t destination = 0; destination < numNodes; ++destination) {
        for (size_t source = 0; source < numNodes; ++source) {
            dist[source * numNodes + destination] =
                rangeMin + static_cast<unsigned int>(range * rand_r(&seed) / static_cast<double>(RAND_MAX));
        }
    }

    // Set diagonal to zero (distance from node to itself is 0)
    for (size_t i = 0; i < numNodes; ++i) {
        dist[i * numNodes + i] = 0;
    }
}

void initializeLocalPathRows(std::vector<unsigned int>& path, const size_t localRows,
                             const size_t numNodes) {
    for (size_t localRow = 0; localRow < localRows; ++localRow) {
        unsigned int* const row = path.data() + localRow * numNodes;
        for (size_t destination = 0; destination < numNodes; ++destination) {
            row[destination] = static_cast<unsigned int>(destination);
        }
    }
}

size_t rowsForRank(const size_t numNodes, const int rank, const int processCount) {
    const size_t baseRows = numNodes / static_cast<size_t>(processCount);
    const size_t remainder = numNodes % static_cast<size_t>(processCount);
    return baseRows + (static_cast<size_t>(rank) < remainder ? 1 : 0);
}

size_t firstRowForRank(const size_t numNodes, const int rank, const int processCount) {
    const size_t baseRows = numNodes / static_cast<size_t>(processCount);
    const size_t remainder = numNodes % static_cast<size_t>(processCount);
    const size_t rankAsSize = static_cast<size_t>(rank);
    return rankAsSize * baseRows + std::min(rankAsSize, remainder);
}

int ownerOfRow(const size_t row, const size_t numNodes, const int processCount) {
    const size_t baseRows = numNodes / static_cast<size_t>(processCount);
    const size_t remainder = numNodes % static_cast<size_t>(processCount);
    const size_t largePartitionRows = (baseRows + 1) * remainder;

    if (row < largePartitionRows) {
        return static_cast<int>(row / (baseRows + 1));
    }
    return static_cast<int>(remainder + (row - largePartitionRows) / baseRows);
}

// Each MPI rank owns a contiguous set of source rows.  For each k, its owner
// broadcasts that completed row; all ranks can then update their local rows
// independently using contiguous inner-loop accesses.
void floydWarshallDistributed(std::vector<unsigned int>& localDist,
                              std::vector<unsigned int>& localPath,
                              const size_t localRows, const size_t firstLocalRow,
                              const size_t numNodes, const int rank,
                              const int processCount) {
    std::vector<unsigned int> receivedKRow(numNodes);

    for (size_t k = 0; k < numNodes; ++k) {
        const int owner = ownerOfRow(k, numNodes, processCount);
        unsigned int* kRow = receivedKRow.data();
        if (rank == owner) {
            kRow = localDist.data() + (k - firstLocalRow) * numNodes;
        }

        MPI_Bcast(kRow, static_cast<int>(numNodes), MPI_UNSIGNED, owner, MPI_COMM_WORLD);

        for (size_t localRow = 0; localRow < localRows; ++localRow) {
            unsigned int* const distRow = localDist.data() + localRow * numNodes;
            unsigned int* const pathRow = localPath.data() + localRow * numNodes;
            const unsigned int distIK = distRow[k];

            for (size_t destination = 0; destination < numNodes; ++destination) {
                const unsigned int oldDistance = distRow[destination];
                const unsigned int newDistance = distIK + kRow[destination];
                if (newDistance < oldDistance) {
                    distRow[destination] = newDistance;
                    pathRow[destination] = static_cast<unsigned int>(k);
                }
            }
        }
    }
}

void transposeSquareInPlace(std::vector<unsigned int>& matrix, const size_t numNodes) {
    for (size_t row = 0; row < numNodes; ++row) {
        for (size_t column = row + 1; column < numNodes; ++column) {
            std::swap(matrix[row * numNodes + column], matrix[column * numNodes + row]);
        }
    }
}

bool validateResult(const std::vector<unsigned int>& dist, const size_t numNodes) {
    // Basic sanity checks
    
    // 1. Diagonal should be zero
    for (size_t i = 0; i < numNodes; ++i) {
        if (dist[idx2(i, i, numNodes)] != 0) {
            printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", i, i);
            return false;
        }
    }
    
    // 2. Triangle inequality: dist[i][k] + dist[k][j] >= dist[i][j]
    // Check a sample of paths to avoid O(n^3) validation time
    for (size_t i = 0; i < std::min(numNodes, static_cast<size_t>(10)); ++i) {
        for (size_t j = 0; j < std::min(numNodes, static_cast<size_t>(10)); ++j) {
            for (size_t k = 0; k < numNodes; ++k) {
                const unsigned int distIJ = dist[idx2(j, i, numNodes)];
                const unsigned int distIK = dist[idx2(k, i, numNodes)];
                const unsigned int distKJ = dist[idx2(j, k, numNodes)];
                
                // Check for overflow before addition
                if (distIK < INF && distKJ < INF) {
                    if (distIK + distKJ < distIJ) {
                        printf("Validation failed: triangle inequality violated at [%zu,%zu,%zu]\n", 
                               i, j, k);
                        return false;
                    }
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
    int processCount = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processCount);

    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
    bool argumentError = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numNodes = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            argumentError = true;
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            break;
        }
    }

    const size_t maxMpiElementCount = static_cast<size_t>(std::numeric_limits<int>::max());
    if (numNodes > maxMpiElementCount ||
        (numNodes != 0 && numNodes > maxMpiElementCount / numNodes)) {
        argumentError = true;
        if (rank == 0) {
            printf("Number of nodes is too large for this MPI implementation: %zu\n", numNodes);
        }
    }

    if (argumentError) {
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing graph...\n");
    }

    const size_t localRows = rowsForRank(numNodes, rank, processCount);
    const size_t firstLocalRow = firstRowForRank(numNodes, rank, processCount);
    const size_t localElementCount = localRows * numNodes;

    std::vector<int> elementCounts(processCount);
    std::vector<int> elementDisplacements(processCount);
    size_t elementOffset = 0;
    for (int process = 0; process < processCount; ++process) {
        const size_t processElementCount = rowsForRank(numNodes, process, processCount) * numNodes;
        elementCounts[process] = static_cast<int>(processElementCount);
        elementDisplacements[process] = static_cast<int>(elementOffset);
        elementOffset += processElementCount;
    }

    // Rank 0 performs the deterministic initialization once, then distributes
    // source rows.  The full matrix is released before timed computation.
    std::vector<unsigned int> distributedRows;
    if (rank == 0) {
        distributedRows.resize(numNodes * numNodes);
        initializeDistanceRows(distributedRows, numNodes, 1, MAX_DISTANCE);
    }

    std::vector<unsigned int> localDist(localElementCount);
    std::vector<unsigned int> localPath(localElementCount);
    initializeLocalPathRows(localPath, localRows, numNodes);

    MPI_Scatterv(rank == 0 ? distributedRows.data() : nullptr,
                 elementCounts.data(), elementDisplacements.data(), MPI_UNSIGNED,
                 localDist.data(), static_cast<int>(localElementCount), MPI_UNSIGNED,
                 0, MPI_COMM_WORLD);
    std::vector<unsigned int>().swap(distributedRows);

    if (rank == 0) {
        printf("Computing shortest paths...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    floydWarshallDistributed(localDist, localPath, localRows, firstLocalRow,
                             numNodes, rank, processCount);
    const double localElapsedSeconds = MPI_Wtime() - start;

    double elapsedSeconds = 0.0;
    MPI_Reduce(&localElapsedSeconds, &elapsedSeconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const long elapsedMilliseconds = static_cast<long>(elapsedSeconds * 1000.0);
        printf("Computation time: %ld ms\n", elapsedMilliseconds);

        // Floyd-Warshall has O(n^3) complexity.  Use the slowest rank's
        // elapsed time, which is the distributed computation completion time.
        const double ops = static_cast<double>(numNodes) * numNodes * numNodes;
        const double gflops = elapsedSeconds > 0.0 ? ops / elapsedSeconds / 1.0e9 : 0.0;
        printf("Performance: %.3f GOPS\n", gflops);
    }

    int exitCode = 0;
    if (printResults || validate) {
        if (rank == 0) {
            distributedRows.resize(numNodes * numNodes);
        }
        MPI_Gatherv(localDist.data(), static_cast<int>(localElementCount), MPI_UNSIGNED,
                    rank == 0 ? distributedRows.data() : nullptr,
                    elementCounts.data(), elementDisplacements.data(), MPI_UNSIGNED,
                    0, MPI_COMM_WORLD);

        if (rank == 0) {
            // Convert source-major distributed rows back to the original
            // destination-major layout before output or validation.
            transposeSquareInPlace(distributedRows, numNodes);

            if (printResults) {
                print_results_int(distributedRows, "DistanceMatrix");
            }

            if (validate) {
                printf("Validating result...\n");
                if (validateResult(distributedRows, numNodes)) {
                    printf("Validation: PASSED\n");
                } else {
                    printf("Validation: FAILED\n");
                    exitCode = 1;
                }
            }
        }
    }

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
