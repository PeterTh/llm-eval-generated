#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

// Index calculation for flattened 2D array (global, row-major: element (row, col) -> row*n+col)
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

// Computes the block-decomposition of `numNodes` rows across `numRanks` processes.
// Ranks [0, remainder) receive one extra row so that all rows are covered.
struct RowDistribution {
    size_t baseRows;
    size_t remainder;

    RowDistribution(const size_t numNodes, const int numRanks)
        : baseRows(numNodes / static_cast<size_t>(numRanks)),
          remainder(numNodes % static_cast<size_t>(numRanks)) {}

    size_t rowCount(const int rank) const noexcept {
        return baseRows + (static_cast<size_t>(rank) < remainder ? 1 : 0);
    }

    size_t rowStart(const int rank) const noexcept {
        const size_t r = static_cast<size_t>(rank);
        if (r <= remainder) {
            return r * (baseRows + 1);
        }
        return remainder * (baseRows + 1) + (r - remainder) * baseRows;
    }

    // Returns {ownerRank, localRowIndex} for a given global row.
    std::pair<int, size_t> owner(const size_t globalRow) const noexcept {
        const size_t boundary = remainder * (baseRows + 1);
        if (globalRow < boundary) {
            return {static_cast<int>(globalRow / (baseRows + 1)), globalRow % (baseRows + 1)};
        }
        const size_t rem = globalRow - boundary;
        return {static_cast<int>(remainder + rem / baseRows), rem % baseRows};
    }
};

// Generates the local rows [rowStart, rowStart+rowCount) of the distance matrix while
// preserving the exact random sequence (and therefore exact values) produced by the
// original sequential generator, which fills the flattened n*n array in row-major order.
void initializeDistanceMatrixLocal(std::vector<unsigned int>& localDist, const size_t numNodes,
                                    const size_t rowStart, const size_t rowCount,
                                    const unsigned int rangeMin, const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;

    const size_t skipCount = rowStart * numNodes;
    for (size_t i = 0; i < skipCount; ++i) {
        rand_r(&seed);
    }

    const size_t genCount = rowCount * numNodes;
    for (size_t i = 0; i < genCount; ++i) {
        localDist[i] = rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
    }

    // Set diagonal to zero (distance from node to itself is 0) for locally-owned rows.
    for (size_t lr = 0; lr < rowCount; ++lr) {
        const size_t globalRow = rowStart + lr;
        localDist[lr * numNodes + globalRow] = 0;
    }
}

// Initializes the local rows of the path matrix. The original sequential algorithm reduces
// to path[i][j] = i for every (row, col) pair, which is embarrassingly local per row.
void initializePathMatrixLocal(std::vector<unsigned int>& localPath, const size_t numNodes,
                                const size_t rowStart, const size_t rowCount) {
    for (size_t lr = 0; lr < rowCount; ++lr) {
        const size_t globalRow = rowStart + lr;
        for (size_t j = 0; j < numNodes; ++j) {
            localPath[lr * numNodes + j] = static_cast<unsigned int>(globalRow);
        }
    }
}

// Distributed Floyd-Warshall: each rank owns a contiguous block of rows. At each iteration
// k, the rank owning row k broadcasts that row of the distance matrix to all other ranks;
// every rank then relaxes its own rows against the received pivot row.
void floydWarshallMPI(std::vector<unsigned int>& localDist,
                      std::vector<unsigned int>& localPath,
                      const size_t numNodes,
                      const size_t rowCount,
                      const RowDistribution& dist) {
    std::vector<unsigned int> pivotRow(numNodes);

    for (size_t k = 0; k < numNodes; ++k) {
        const auto [ownerRank, ownerLocalRow] = dist.owner(k);

        if (static_cast<size_t>(ownerLocalRow) < rowCount) {
            std::memcpy(pivotRow.data(), &localDist[ownerLocalRow * numNodes],
                        numNodes * sizeof(unsigned int));
        }
        MPI_Bcast(pivotRow.data(), static_cast<int>(numNodes), MPI_UNSIGNED, ownerRank, MPI_COMM_WORLD);

        for (size_t lr = 0; lr < rowCount; ++lr) {
            const unsigned int distIK = localDist[lr * numNodes + k];
            unsigned int* distRow = &localDist[lr * numNodes];
            unsigned int* pathRow = &localPath[lr * numNodes];
            for (size_t j = 0; j < numNodes; ++j) {
                const unsigned int newDist = distIK + pivotRow[j];
                if (newDist < distRow[j]) {
                    distRow[j] = newDist;
                    pathRow[j] = static_cast<unsigned int>(k);
                }
            }
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

    int worldRank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numNodes = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (worldRank == 0) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            if (worldRank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (worldRank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("MPI ranks: %d\n", worldSize);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const RowDistribution rowDist(numNodes, worldSize);
    const size_t rowStart = rowDist.rowStart(worldRank);
    const size_t rowCount = rowDist.rowCount(worldRank);

    // Allocate only the local portion of each matrix (distributed memory).
    std::vector<unsigned int> localDist(rowCount * numNodes);
    std::vector<unsigned int> localPath(rowCount * numNodes);

    if (worldRank == 0) {
        printf("Initializing graph...\n");
    }
    initializeDistanceMatrixLocal(localDist, numNodes, rowStart, rowCount, 1, MAX_DISTANCE);
    initializePathMatrixLocal(localPath, numNodes, rowStart, rowCount);

    if (worldRank == 0) {
        printf("Computing shortest paths...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    floydWarshallMPI(localDist, localPath, numNodes, rowCount, rowDist);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (worldRank == 0) {
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate operations per second
        // Floyd-Warshall has O(n^3) complexity
        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);
    }

    // Gather the distributed distance matrix on rank 0 for reporting/validation only
    // (the path matrix is never needed off of its owning rank for these purposes).
    if (printResults || validate) {
        std::vector<int> recvCounts;
        std::vector<int> displs;
        std::vector<unsigned int> fullDist;
        if (worldRank == 0) {
            recvCounts.resize(worldSize);
            displs.resize(worldSize);
            for (int r = 0; r < worldSize; ++r) {
                recvCounts[r] = static_cast<int>(rowDist.rowCount(r) * numNodes);
                displs[r] = static_cast<int>(rowDist.rowStart(r) * numNodes);
            }
            fullDist.resize(numNodes * numNodes);
        }
        MPI_Gatherv(localDist.data(), static_cast<int>(localDist.size()), MPI_UNSIGNED,
                    fullDist.data(), recvCounts.data(), displs.data(), MPI_UNSIGNED,
                    0, MPI_COMM_WORLD);

        if (worldRank == 0) {
            if (printResults) {
                print_results_int(fullDist, "DistanceMatrix");
            }

            if (validate) {
                printf("Validating result...\n");
                bool valid = validateResult(fullDist, numNodes);

                if (valid) {
                    printf("Validation: PASSED\n");
                } else {
                    printf("Validation: FAILED\n");
                }

                int exitCode = valid ? 0 : 1;
                MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
                MPI_Finalize();
                return exitCode;
            }
        } else if (validate) {
            int exitCode = 0;
            MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Finalize();
            return exitCode;
        }
    }

    MPI_Finalize();
    return 0;
}
