#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

// The original benchmark uses column-major indexing at its call sites.  This
// is equivalent to a row-major matrix where the first logical index is the
// source node, so local rows can be distributed contiguously among ranks.
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

void initializeDistanceRows(std::vector<unsigned int>& dist, const size_t numNodes,
                            const size_t firstRow, unsigned int& seed,
                            const unsigned int rangeMin, const unsigned int rangeMax) {
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;

    for (size_t i = 0; i < dist.size(); ++i) {
        dist[i] = rangeMin + static_cast<unsigned int>(range * rand_r(&seed) / RAND_MAX);
    }

    // Set local diagonal entries to zero after the complete random stream has
    // been generated, matching the original matrix initialization order.
    const size_t localRows = numNodes == 0 ? 0 : dist.size() / numNodes;
    for (size_t localRow = 0; localRow < localRows; ++localRow) {
        const size_t globalRow = firstRow + localRow;
        dist[localRow * numNodes + globalRow] = 0;
    }
}

size_t rowsForRank(const size_t numNodes, const int worldSize, const int rank) noexcept {
    const size_t baseRows = numNodes / static_cast<size_t>(worldSize);
    const size_t extraRows = numNodes % static_cast<size_t>(worldSize);
    return baseRows + (static_cast<size_t>(rank) < extraRows ? 1 : 0);
}

size_t firstRowForRank(const size_t numNodes, const int worldSize, const int rank) noexcept {
    const size_t baseRows = numNodes / static_cast<size_t>(worldSize);
    const size_t extraRows = numNodes % static_cast<size_t>(worldSize);
    return static_cast<size_t>(rank) * baseRows +
           std::min(static_cast<size_t>(rank), extraRows);
}

int ownerOfRow(const size_t row, const size_t numNodes, const int worldSize) noexcept {
    const size_t baseRows = numNodes / static_cast<size_t>(worldSize);
    const size_t extraRows = numNodes % static_cast<size_t>(worldSize);

    if (baseRows == 0) {
        return static_cast<int>(row);
    }
    const size_t firstBlockRows = (baseRows + 1) * extraRows;
    if (row < firstBlockRows) {
        return static_cast<int>(row / (baseRows + 1));
    }
    return static_cast<int>(extraRows + (row - firstBlockRows) / baseRows);
}

void initializePathMatrix(std::vector<unsigned int>& path, const size_t numNodes,
                          const size_t firstRow) {
    const size_t localRows = numNodes == 0 ? 0 : path.size() / numNodes;
    for (size_t localRow = 0; localRow < localRows; ++localRow) {
        const unsigned int source = static_cast<unsigned int>(firstRow + localRow);
        unsigned int* pathRow = path.data() + localRow * numNodes;
        std::fill(pathRow, pathRow + numNodes, source);
    }
}

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes,
                   const size_t firstRow,
                   const int worldSize,
                   const MPI_Comm communicator) {
    const size_t localRows = numNodes == 0 ? 0 : dist.size() / numNodes;
    std::vector<unsigned int> pivotRow(numNodes);

    // The matrix is distributed by source rows.  At iteration k every rank
    // needs row k (dist[k][j]); broadcasting that row is the complete
    // communication needed for the 1-D Floyd-Warshall decomposition.
    for (size_t k = 0; k < numNodes; ++k) {
        const int pivotOwner = ownerOfRow(k, numNodes, worldSize);
        if (firstRow <= k && k < firstRow + localRows) {
            const size_t localPivotRow = k - firstRow;
            std::memcpy(pivotRow.data(), dist.data() + localPivotRow * numNodes,
                        numNodes * sizeof(unsigned int));
        }

        MPI_Bcast(pivotRow.data(), static_cast<int>(numNodes), MPI_UNSIGNED,
                  pivotOwner, communicator);

        for (size_t localRow = 0; localRow < localRows; ++localRow) {
            unsigned int* distRow = dist.data() + localRow * numNodes;
            unsigned int* pathRow = path.data() + localRow * numNodes;
            const unsigned int distToPivot = distRow[k];

            // j is the innermost loop so both the distance and path rows are
            // traversed linearly, enabling compiler vectorization and making
            // the local computation bandwidth/cache friendly.
            for (size_t j = 0; j < numNodes; ++j) {
                const unsigned int newDist = distToPivot + pivotRow[j];
                if (newDist < distRow[j]) {
                    distRow[j] = newDist;
                    pathRow[j] = static_cast<unsigned int>(k);
                }
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

                // Check for overflow before addition.
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
    int providedThreadLevel = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &providedThreadLevel);

    int rank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;
    bool argumentsValid = true;
    bool showHelp = false;

    // Parse command line arguments on every rank so all ranks make the same
    // control-flow decisions before entering collectives.
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
            argumentsValid = false;
        }
    }

    if (showHelp || !argumentsValid) {
        if (rank == 0) {
            if (!argumentsValid) {
                printf("Unknown or incomplete command-line option\n");
            }
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return showHelp && argumentsValid ? 0 : 1;
    }

    // Guard the size calculations used by the local buffers.  The MPI count
    // limit is also checked because the MPI-3 point-to-point and Gatherv
    // interfaces used by this benchmark use int counts.
    const size_t maxSize = std::numeric_limits<size_t>::max();
    if ((numNodes != 0 && numNodes > maxSize / numNodes) ||
        (numNodes != 0 && numNodes * numNodes >
             maxSize / sizeof(unsigned int)) ||
        (numNodes > static_cast<size_t>(INT_MAX))) {
        if (rank == 0) {
            printf("Number of nodes is too large for the available index/count types\n");
        }
        MPI_Finalize();
        return 1;
    }

    std::vector<int> matrixCounts(static_cast<size_t>(worldSize));
    std::vector<int> matrixDisplacements(static_cast<size_t>(worldSize));
    bool mpiCountsValid = true;
    for (int process = 0; process < worldSize; ++process) {
        const size_t processRows = rowsForRank(numNodes, worldSize, process);
        const size_t processFirstRow = firstRowForRank(numNodes, worldSize, process);
        const size_t processElements = processRows * numNodes;
        const size_t processDisplacement = processFirstRow * numNodes;
        if (processElements > static_cast<size_t>(INT_MAX) ||
            processDisplacement > static_cast<size_t>(INT_MAX)) {
            mpiCountsValid = false;
        } else {
            matrixCounts[static_cast<size_t>(process)] = static_cast<int>(processElements);
            matrixDisplacements[static_cast<size_t>(process)] =
                static_cast<int>(processDisplacement);
        }
    }
    if (!mpiCountsValid) {
        if (rank == 0) {
            printf("Number of nodes is too large for MPI Scatterv/Gatherv counts\n");
        }
        MPI_Finalize();
        return 1;
    }

    const size_t localRows = rowsForRank(numNodes, worldSize, rank);
    const size_t firstRow = firstRowForRank(numNodes, worldSize, rank);
    std::vector<unsigned int> localDist(localRows * numNodes);
    std::vector<unsigned int> localPath(localRows * numNodes);
    std::vector<unsigned int> globalDist;

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing graph...\n");
        unsigned int seed = 42;
        std::vector<unsigned int> sendBuffer;
        for (int process = 0; process < worldSize; ++process) {
            const size_t processRows = rowsForRank(numNodes, worldSize, process);
            const size_t processFirstRow = firstRowForRank(numNodes, worldSize, process);
            const size_t processElements = processRows * numNodes;
            if (process == 0) {
                initializeDistanceRows(localDist, numNodes, processFirstRow, seed, 1,
                                       MAX_DISTANCE);
            } else if (processElements != 0) {
                sendBuffer.resize(processElements);
                initializeDistanceRows(sendBuffer, numNodes, processFirstRow, seed, 1,
                                       MAX_DISTANCE);
                MPI_Send(sendBuffer.data(), matrixCounts[static_cast<size_t>(process)],
                         MPI_UNSIGNED, process, 0, MPI_COMM_WORLD);
            }
        }
    } else if (localRows != 0) {
        MPI_Recv(localDist.data(), matrixCounts[static_cast<size_t>(rank)], MPI_UNSIGNED,
                 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    initializePathMatrix(localPath, numNodes, firstRow);

    if (rank == 0) {
        printf("Computing shortest paths...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::high_resolution_clock::now();

    floydWarshall(localDist, localPath, numNodes, firstRow, worldSize, MPI_COMM_WORLD);

    const auto end = std::chrono::high_resolution_clock::now();
    const double localSeconds = std::chrono::duration<double>(end - start).count();
    double maximumSeconds = 0.0;
    MPI_Reduce(&localSeconds, &maximumSeconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Report the slowest rank's time, which is the elapsed time visible
        // to the distributed computation as a whole.
        const long maximumMilliseconds = static_cast<long>(maximumSeconds * 1000.0);
        printf("Computation time: %ld ms\n", maximumMilliseconds);
        const double ops = static_cast<double>(numNodes) * numNodes * numNodes;
        const double gflops = maximumSeconds > 0.0 ? ops / maximumSeconds / 1e9 : 0.0;
        printf("Performance: %.3f GOPS\n", gflops);
    }

    if (printResults || validate) {
        if (rank != 0) {
            globalDist.clear();
        }
        if (rank == 0 && globalDist.empty() && numNodes != 0) {
            globalDist.resize(numNodes * numNodes);
        }

        MPI_Gatherv(localDist.data(), matrixCounts[static_cast<size_t>(rank)], MPI_UNSIGNED,
                    rank == 0 ? globalDist.data() : nullptr,
                    matrixCounts.data(), matrixDisplacements.data(), MPI_UNSIGNED,
                    0, MPI_COMM_WORLD);
    }

    if (rank == 0 && printResults) {
        print_results_int(globalDist, "DistanceMatrix");
    }

    int valid = 1;
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
            valid = validateResult(globalDist, numNodes) ? 1 : 0;
            printf("Validation: %s\n", valid == 1 ? "PASSED" : "FAILED");
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return valid == 1 ? 0 : 1;
}
