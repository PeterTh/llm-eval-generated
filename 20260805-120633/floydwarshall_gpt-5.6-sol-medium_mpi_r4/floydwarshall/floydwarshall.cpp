#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

// The original matrix layout is row-major: idx2(column, row, n).
inline constexpr size_t idx2(const size_t column, const size_t row,
                             const size_t n) noexcept {
    return row * n + column;
}

void initializeDistanceMatrix(std::vector<unsigned int>& dist,
                              const size_t numNodes,
                              const unsigned int rangeMin,
                              const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;

    for (size_t i = 0; i < numNodes * numNodes; ++i) {
        dist[i] = rangeMin + static_cast<unsigned int>(
                                 range * rand_r(&seed) /
                                 static_cast<double>(RAND_MAX));
    }

    for (size_t i = 0; i < numNodes; ++i) {
        dist[idx2(i, i, numNodes)] = 0;
    }
}

// Kept separate so the compiler can see that the three rows do not alias.
inline void updateRow(unsigned int* __restrict row,
                      unsigned int* __restrict pathRow,
                      const unsigned int* __restrict pivotRow,
                      const unsigned int distanceToPivot, const int numNodes,
                      const unsigned int pivot) noexcept {
    for (int j = 0; j < numNodes; ++j) {
        const unsigned int candidate = distanceToPivot + pivotRow[j];
        if (candidate < row[j]) {
            row[j] = candidate;
            pathRow[j] = pivot;
        }
    }
}

void floydWarshallDistributed(std::vector<unsigned int>& localDist,
                              std::vector<unsigned int>& localPath,
                              const int numNodes, const int firstRow,
                              const int localRows,
                              const std::vector<int>& rowOwner,
                              const int rank, MPI_Comm communicator) {
    std::vector<unsigned int> pivotRow(static_cast<size_t>(numNodes));

    for (int k = 0; k < numNodes; ++k) {
        const int owner = rowOwner[static_cast<size_t>(k)];
        if (rank == owner) {
            const size_t localK = static_cast<size_t>(k - firstRow);
            std::copy_n(localDist.data() + localK * numNodes, numNodes,
                        pivotRow.data());
        }
        MPI_Bcast(pivotRow.data(), numNodes, MPI_UNSIGNED, owner,
                  communicator);

        for (int localI = 0; localI < localRows; ++localI) {
            const int globalI = firstRow + localI;
            // With a zero diagonal, pivot row k cannot change in iteration k.
            if (globalI == k) {
                continue;
            }
            unsigned int* const row =
                localDist.data() + static_cast<size_t>(localI) * numNodes;
            unsigned int* const pathRow =
                localPath.data() + static_cast<size_t>(localI) * numNodes;
            updateRow(row, pathRow, pivotRow.data(), row[k], numNodes,
                      static_cast<unsigned int>(k));
        }
    }
}

bool validateResult(const std::vector<unsigned int>& dist,
                    const size_t numNodes) {
    for (size_t i = 0; i < numNodes; ++i) {
        if (dist[idx2(i, i, numNodes)] != 0) {
            std::printf(
                "Validation failed: diagonal element [%zu,%zu] is not zero\n",
                i, i);
            return false;
        }
    }

    for (size_t i = 0; i < std::min(numNodes, size_t{10}); ++i) {
        for (size_t j = 0; j < std::min(numNodes, size_t{10}); ++j) {
            for (size_t k = 0; k < numNodes; ++k) {
                const unsigned int distIJ = dist[idx2(j, i, numNodes)];
                const unsigned int distIK = dist[idx2(k, i, numNodes)];
                const unsigned int distKJ = dist[idx2(j, k, numNodes)];
                if (distIK < INF && distKJ < INF &&
                    distIK + distKJ < distIJ) {
                    std::printf(
                        "Validation failed: triangle inequality violated at "
                        "[%zu,%zu,%zu]\n",
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

    int rank = 0;
    int processCount = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processCount);

    size_t numNodesSize = 512;
    bool validate = false;
    bool printResults = false;
    int parseStatus = 0;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* value = argv[++i];
            char* end = nullptr;
            const unsigned long long parsed = std::strtoull(value, &end, 10);
            if (*value == '\0' || *end != '\0' || parsed == 0 ||
                parsed > static_cast<unsigned long long>(
                             std::numeric_limits<int>::max())) {
                if (rank == 0) {
                    std::fprintf(stderr, "Invalid number of nodes: %s\n", value);
                }
                parseStatus = 1;
                break;
            }
            numNodesSize = static_cast<size_t>(parsed);
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
                std::printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            parseStatus = 1;
            break;
        }
    }

    // Scatterv/Gatherv use int counts and displacements in MPI-3.
    if (numNodesSize > static_cast<size_t>(std::sqrt(
                           static_cast<double>(
                               std::numeric_limits<int>::max())))) {
        if (rank == 0) {
            std::fprintf(stderr,
                         "Matrix is too large for this MPI implementation\n");
        }
        parseStatus = 1;
    }
    if (parseStatus != 0) {
        MPI_Finalize();
        return 1;
    }

    const int numNodes = static_cast<int>(numNodesSize);
    const int baseRows = numNodes / processCount;
    const int extraRows = numNodes % processCount;
    std::vector<int> rows(static_cast<size_t>(processCount));
    std::vector<int> firstRows(static_cast<size_t>(processCount));
    std::vector<int> counts(static_cast<size_t>(processCount));
    std::vector<int> displacements(static_cast<size_t>(processCount));
    std::vector<int> rowOwner(numNodesSize);

    int nextRow = 0;
    for (int p = 0; p < processCount; ++p) {
        rows[p] = baseRows + (p < extraRows ? 1 : 0);
        firstRows[p] = nextRow;
        counts[p] = rows[p] * numNodes;
        displacements[p] = nextRow * numNodes;
        for (int r = 0; r < rows[p]; ++r) {
            rowOwner[static_cast<size_t>(nextRow + r)] = p;
        }
        nextRow += rows[p];
    }

    const int localRows = rows[rank];
    const int firstRow = firstRows[rank];
    const size_t localElements = static_cast<size_t>(localRows) * numNodesSize;
    std::vector<unsigned int> localDist(localElements);
    std::vector<unsigned int> localPath(localElements);
    for (int localI = 0; localI < localRows; ++localI) {
        std::fill_n(localPath.data() + static_cast<size_t>(localI) * numNodes,
                    numNodes, static_cast<unsigned int>(firstRow + localI));
    }

    std::vector<unsigned int> globalDist;
    if (rank == 0) {
        std::printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        std::printf("Number of nodes: %zu\n", numNodesSize);
        std::printf("MPI processes: %d\n", processCount);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Initializing graph...\n");
        globalDist.resize(numNodesSize * numNodesSize);
        initializeDistanceMatrix(globalDist, numNodesSize, 1, MAX_DISTANCE);
    }

    MPI_Scatterv(rank == 0 ? globalDist.data() : nullptr, counts.data(),
                 displacements.data(), MPI_UNSIGNED, localDist.data(),
                 counts[rank], MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    std::vector<unsigned int>().swap(globalDist);

    if (rank == 0) {
        std::printf("Computing shortest paths...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    floydWarshallDistributed(localDist, localPath, numNodes, firstRow,
                             localRows, rowOwner, rank, MPI_COMM_WORLD);
    const double localSeconds = MPI_Wtime() - start;
    double elapsedSeconds = 0.0;
    MPI_Reduce(&localSeconds, &elapsedSeconds, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);

    if (rank == 0) {
        const long long milliseconds =
            static_cast<long long>(elapsedSeconds * 1000.0);
        std::printf("Computation time: %lld ms\n", milliseconds);
        const double operations = static_cast<double>(numNodesSize) *
                                  numNodesSize * numNodesSize;
        const double gops = operations / elapsedSeconds / 1.0e9;
        std::printf("Performance: %.3f GOPS\n", gops);
    }

    if (printResults || validate) {
        if (rank == 0) {
            globalDist.resize(numNodesSize * numNodesSize);
        }
        MPI_Gatherv(localDist.data(), counts[rank], MPI_UNSIGNED,
                    rank == 0 ? globalDist.data() : nullptr, counts.data(),
                    displacements.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    }

    int exitStatus = 0;
    if (rank == 0) {
        if (printResults) {
            print_results_int(globalDist, "DistanceMatrix");
        }
        if (validate) {
            std::printf("Validating result...\n");
            if (validateResult(globalDist, numNodesSize)) {
                std::printf("Validation: PASSED\n");
            } else {
                std::printf("Validation: FAILED\n");
                exitStatus = 1;
            }
        }
    }
    MPI_Bcast(&exitStatus, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitStatus;
}
