#include <mpi.h>

#include <algorithm>
#include <array>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

// Index calculation for a row-major flattened 2D array.
inline constexpr size_t idx2(const size_t row, const size_t column,
                             const size_t n) noexcept {
    return row * n + column;
}

void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes,
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

struct RowPartition {
    size_t firstRow;
    size_t rowCount;
};

RowPartition partitionForRank(const size_t numNodes, const int commSize,
                              const int rank) noexcept {
    const size_t ranks = static_cast<size_t>(commSize);
    const size_t base = numNodes / ranks;
    const size_t remainder = numNodes % ranks;
    const size_t rankIndex = static_cast<size_t>(rank);
    return {rankIndex * base + std::min(rankIndex, remainder),
            base + (rankIndex < remainder ? 1U : 0U)};
}

int ownerOfRow(const size_t row, const size_t numNodes, const int commSize) noexcept {
    const size_t ranks = static_cast<size_t>(commSize);
    const size_t base = numNodes / ranks;
    const size_t remainder = numNodes % ranks;
    const size_t longRows = (base + 1) * remainder;
    if (row < longRows) {
        return static_cast<int>(row / (base + 1));
    }
    return static_cast<int>(remainder + (row - longRows) / base);
}

inline void updateRow(unsigned int* distRow, unsigned int* pathRow,
                      const unsigned int* pivotRow, const size_t n,
                      const size_t k) noexcept {
    const unsigned int distIK = distRow[k];

    // The pointers refer to disjoint arrays, and each iteration writes a distinct
    // element. Compilers therefore vectorize this min-plus update at -O3.
    for (size_t j = 0; j < n; ++j) {
        const unsigned int newDist = distIK + pivotRow[j];
        if (newDist < distRow[j]) {
            distRow[j] = newDist;
            pathRow[j] = static_cast<unsigned int>(k);
        }
    }
}

void floydWarshallDistributed(std::vector<unsigned int>& dist,
                              std::vector<unsigned int>& path,
                              const size_t numNodes, const RowPartition local,
                              const int rank, const int commSize,
                              MPI_Comm communicator) {
    if (numNodes == 0) {
        return;
    }

    std::array<std::vector<unsigned int>, 2> pivotRows = {
        std::vector<unsigned int>(numNodes), std::vector<unsigned int>(numNodes)};
    std::array<MPI_Request, 2> requests = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};

    int currentBuffer = 0;
    if (rank == ownerOfRow(0, numNodes, commSize)) {
        const size_t localRow = 0 - local.firstRow;
        std::copy_n(dist.data() + localRow * numNodes, numNodes,
                    pivotRows[currentBuffer].data());
    }
    MPI_Ibcast(pivotRows[currentBuffer].data(), static_cast<int>(numNodes),
               MPI_UNSIGNED, ownerOfRow(0, numNodes, commSize), communicator,
               &requests[currentBuffer]);

    for (size_t k = 0; k < numNodes; ++k) {
        MPI_Wait(&requests[currentBuffer], MPI_STATUS_IGNORE);
        const unsigned int* const pivot = pivotRows[currentBuffer].data();
        const size_t nextPivot = k + 1;

        // The next pivot's owner makes that row ready first. Its nonblocking
        // broadcast can then progress while every rank updates its other rows.
        const int nextBuffer = currentBuffer ^ 1;
        if (nextPivot < numNodes) {
            const int nextOwner = ownerOfRow(nextPivot, numNodes, commSize);
            if (rank == nextOwner) {
                const size_t localRow = nextPivot - local.firstRow;
                updateRow(dist.data() + localRow * numNodes,
                          path.data() + localRow * numNodes, pivot, numNodes, k);
                std::copy_n(dist.data() + localRow * numNodes, numNodes,
                            pivotRows[nextBuffer].data());
            }
            MPI_Ibcast(pivotRows[nextBuffer].data(), static_cast<int>(numNodes),
                       MPI_UNSIGNED, nextOwner, communicator,
                       &requests[nextBuffer]);
        }

        for (size_t localRow = 0; localRow < local.rowCount; ++localRow) {
            const size_t globalRow = local.firstRow + localRow;
            if (globalRow == nextPivot) {
                continue;  // Already updated before starting its broadcast.
            }
            updateRow(dist.data() + localRow * numNodes,
                      path.data() + localRow * numNodes, pivot, numNodes, k);
        }

        currentBuffer = nextBuffer;
    }
}

bool validateResult(const std::vector<unsigned int>& dist, const size_t numNodes) {
    for (size_t i = 0; i < numNodes; ++i) {
        if (dist[idx2(i, i, numNodes)] != 0) {
            std::printf("Validation failed: diagonal element [%zu,%zu] is not zero\n",
                        i, i);
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
    int commSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &commSize);

    size_t numNodes = 512;
    int validate = 0;
    int printResults = 0;
    int parseStatus = 0;  // 0: run, 1: help, 2: invalid arguments

    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                numNodes = static_cast<size_t>(std::strtoull(argv[++i], nullptr, 10));
            } else if (std::strcmp(argv[i], "-v") == 0) {
                validate = 1;
            } else if (std::strcmp(argv[i], "-r") == 0) {
                printResults = 1;
            } else if (std::strcmp(argv[i], "-h") == 0) {
                parseStatus = 1;
            } else {
                std::printf("Unknown option: %s\n", argv[i]);
                parseStatus = 2;
                break;
            }
        }
        if (parseStatus != 0) {
            printUsage(argv[0]);
        }
    }

    unsigned long long nodesWire = static_cast<unsigned long long>(numNodes);
    MPI_Bcast(&parseStatus, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nodesWire, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_INT, 0, MPI_COMM_WORLD);
    numNodes = static_cast<size_t>(nodesWire);

    if (parseStatus != 0) {
        MPI_Finalize();
        return parseStatus == 1 ? 0 : 1;
    }

    bool dimensionsValid = numNodes <= static_cast<size_t>(INT_MAX) &&
                           (numNodes == 0 ||
                            numNodes <= std::numeric_limits<size_t>::max() / numNodes) &&
                           numNodes * numNodes <= static_cast<size_t>(INT_MAX);
    if (!dimensionsValid) {
        if (rank == 0) {
            std::fprintf(stderr,
                         "Matrix is too large for this MPI implementation's "
                         "32-bit collective counts.\n");
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        std::printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        std::printf("Number of nodes: %zu\n", numNodes);
        std::printf("MPI ranks: %d\n", commSize);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Initializing graph...\n");
    }

    const RowPartition local = partitionForRank(numNodes, commSize, rank);
    std::vector<unsigned int> localDist(local.rowCount * numNodes);
    std::vector<unsigned int> localPath(local.rowCount * numNodes);

    for (size_t localRow = 0; localRow < local.rowCount; ++localRow) {
        const unsigned int source =
            static_cast<unsigned int>(local.firstRow + localRow);
        std::fill_n(localPath.data() + localRow * numNodes, numNodes, source);
    }

    std::vector<int> counts(static_cast<size_t>(commSize));
    std::vector<int> displacements(static_cast<size_t>(commSize));
    for (int process = 0; process < commSize; ++process) {
        const RowPartition part = partitionForRank(numNodes, commSize, process);
        counts[static_cast<size_t>(process)] =
            static_cast<int>(part.rowCount * numNodes);
        displacements[static_cast<size_t>(process)] =
            static_cast<int>(part.firstRow * numNodes);
    }

    std::vector<unsigned int> globalDist;
    if (rank == 0) {
        globalDist.resize(numNodes * numNodes);
        initializeDistanceMatrix(globalDist, numNodes, 1, MAX_DISTANCE);
    }
    MPI_Scatterv(rank == 0 ? globalDist.data() : nullptr, counts.data(),
                 displacements.data(), MPI_UNSIGNED, localDist.data(),
                 counts[static_cast<size_t>(rank)], MPI_UNSIGNED, 0,
                 MPI_COMM_WORLD);
    globalDist.clear();
    globalDist.shrink_to_fit();

    if (rank == 0) {
        std::printf("Computing shortest paths...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    floydWarshallDistributed(localDist, localPath, numNodes, local, rank,
                             commSize, MPI_COMM_WORLD);

    const double localDuration = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&localDuration, &duration, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);

    if (rank == 0) {
        const long long durationMs = static_cast<long long>(duration * 1000.0);
        std::printf("Computation time: %lld ms\n", durationMs);
        const double ops = static_cast<double>(numNodes) *
                           static_cast<double>(numNodes) *
                           static_cast<double>(numNodes);
        const double gops = duration > 0.0 ? ops / duration / 1.0e9 : 0.0;
        std::printf("Performance: %.3f GOPS\n", gops);
    }

    if (printResults || validate) {
        if (rank == 0) {
            globalDist.resize(numNodes * numNodes);
        }
        MPI_Gatherv(localDist.data(), counts[static_cast<size_t>(rank)],
                    MPI_UNSIGNED, rank == 0 ? globalDist.data() : nullptr,
                    counts.data(), displacements.data(), MPI_UNSIGNED, 0,
                    MPI_COMM_WORLD);
    }

    int exitStatus = 0;
    if (rank == 0 && printResults) {
        print_results_int(globalDist, "DistanceMatrix");
    }
    if (rank == 0 && validate) {
        std::printf("Validating result...\n");
        if (validateResult(globalDist, numNodes)) {
            std::printf("Validation: PASSED\n");
        } else {
            std::printf("Validation: FAILED\n");
            exitStatus = 1;
        }
    }

    MPI_Bcast(&exitStatus, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitStatus;
}
