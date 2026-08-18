#include <mpi.h>

#include <algorithm>
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

// The original layout is conventional row-major when accessed as idx2(column, row).
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes,
                              const unsigned int rangeMin, const unsigned int rangeMax) {
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

// Every rank owns a contiguous range of source rows. At iteration k only row k is
// needed globally, so communicating that row avoids replicating either matrix.
void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes,
                   const size_t firstRow,
                   const std::vector<int>& rowOwner,
                   const int rank,
                   MPI_Comm comm) {
    const size_t localRows = numNodes == 0 ? 0 : dist.size() / numNodes;
    std::vector<unsigned int> pivotRow(numNodes);

    for (size_t k = 0; k < numNodes; ++k) {
        const int owner = rowOwner[k];
        if (rank == owner) {
            const size_t localK = k - firstRow;
            std::copy_n(dist.data() + localK * numNodes, numNodes, pivotRow.data());
        }
        MPI_Bcast(pivotRow.data(), static_cast<int>(numNodes), MPI_UNSIGNED, owner, comm);

        for (size_t localI = 0; localI < localRows; ++localI) {
            unsigned int* const distRow = dist.data() + localI * numNodes;
            unsigned int* const pathRow = path.data() + localI * numNodes;
            const unsigned int distIK = distRow[k];

            // Rows are contiguous, making this inner loop SIMD-friendly.
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC ivdep
#endif
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

bool validateResult(const std::vector<unsigned int>& dist,
                    const size_t numNodes,
                    const size_t firstRow,
                    const std::vector<int>& rowOwner,
                    const int rank,
                    MPI_Comm comm) {
    const size_t localRows = numNodes == 0 ? 0 : dist.size() / numNodes;
    bool localValid = true;

    for (size_t localI = 0; localI < localRows; ++localI) {
        const size_t globalI = firstRow + localI;
        if (dist[localI * numNodes + globalI] != 0) {
            localValid = false;
        }
    }

    // Reuse the same distributed-row access pattern for the original sampled
    // triangle-inequality validation; no full matrix gather is necessary.
    std::vector<unsigned int> kthRow(numNodes);
    const size_t sampleSize = std::min(numNodes, static_cast<size_t>(10));
    for (size_t k = 0; k < numNodes; ++k) {
        const int owner = rowOwner[k];
        if (rank == owner) {
            std::copy_n(dist.data() + (k - firstRow) * numNodes, numNodes, kthRow.data());
        }
        MPI_Bcast(kthRow.data(), static_cast<int>(numNodes), MPI_UNSIGNED, owner, comm);

        for (size_t localI = 0; localI < localRows; ++localI) {
            const size_t globalI = firstRow + localI;
            if (globalI >= sampleSize) {
                continue;
            }
            const unsigned int* const row = dist.data() + localI * numNodes;
            const unsigned int distIK = row[k];
            if (distIK >= INF) {
                continue;
            }
            for (size_t j = 0; j < sampleSize; ++j) {
                if (kthRow[j] < INF && distIK + kthRow[j] < row[j]) {
                    localValid = false;
                }
            }
        }
    }

    int local = localValid ? 1 : 0;
    int global = 0;
    MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MIN, comm);
    return global != 0;
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
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;
    int parseResult = 0;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numNodes = static_cast<size_t>(strtoull(argv[++i], nullptr, 10));
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
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            parseResult = 1;
        }
    }
    if (parseResult != 0) {
        MPI_Finalize();
        return parseResult;
    }

    if (numNodes > static_cast<size_t>(INT_MAX) ||
        (numNodes != 0 && numNodes > std::numeric_limits<size_t>::max() / numNodes)) {
        if (rank == 0) {
            fprintf(stderr, "Matrix dimensions exceed the supported MPI or address range\n");
        }
        MPI_Finalize();
        return 1;
    }

    const size_t baseRows = numNodes / static_cast<size_t>(worldSize);
    const size_t extraRows = numNodes % static_cast<size_t>(worldSize);
    const size_t localRows = baseRows + (static_cast<size_t>(rank) < extraRows ? 1 : 0);
    const size_t firstRow = static_cast<size_t>(rank) * baseRows +
                            std::min(static_cast<size_t>(rank), extraRows);

    std::vector<int> counts(worldSize);
    std::vector<int> displacements(worldSize);
    std::vector<int> rowOwner(numNodes);
    bool countsFit = true;
    size_t displacement = 0;
    for (int r = 0; r < worldSize; ++r) {
        const size_t rows = baseRows + (static_cast<size_t>(r) < extraRows ? 1 : 0);
        const size_t elements = rows * numNodes;
        if (elements > static_cast<size_t>(INT_MAX) || displacement > static_cast<size_t>(INT_MAX)) {
            countsFit = false;
        } else {
            counts[r] = static_cast<int>(elements);
            displacements[r] = static_cast<int>(displacement);
        }
        const size_t begin = static_cast<size_t>(r) * baseRows +
                             std::min(static_cast<size_t>(r), extraRows);
        std::fill_n(rowOwner.begin() + begin, rows, r);
        displacement += elements;
    }
    if (!countsFit) {
        if (rank == 0) {
            fprintf(stderr, "Per-rank matrix partition exceeds MPI's count range\n");
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing graph...\n");
    }

    std::vector<unsigned int> globalDist;
    if (rank == 0) {
        globalDist.resize(numNodes * numNodes);
        initializeDistanceMatrix(globalDist, numNodes, 1, MAX_DISTANCE);
    }

    std::vector<unsigned int> dist(localRows * numNodes);
    std::vector<unsigned int> path(localRows * numNodes);
    MPI_Scatterv(rank == 0 ? globalDist.data() : nullptr, counts.data(), displacements.data(),
                 MPI_UNSIGNED, dist.data(), counts[rank], MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    // The original initialization makes every entry in a source row equal to
    // that source index before Floyd-Warshall updates it.
    for (size_t localI = 0; localI < localRows; ++localI) {
        std::fill_n(path.data() + localI * numNodes, numNodes,
                    static_cast<unsigned int>(firstRow + localI));
    }
    globalDist.clear();
    globalDist.shrink_to_fit();

    if (rank == 0) {
        printf("Computing shortest paths...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    floydWarshall(dist, path, numNodes, firstRow, rowOwner, rank, MPI_COMM_WORLD);
    const double localSeconds = MPI_Wtime() - start;
    double seconds = 0.0;
    MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const long long milliseconds = static_cast<long long>(seconds * 1000.0);
        printf("Computation time: %lld ms\n", milliseconds);
        const double ops = static_cast<double>(numNodes) * static_cast<double>(numNodes) *
                           static_cast<double>(numNodes);
        const double gops = seconds > 0.0 ? ops / seconds / 1e9 : 0.0;
        printf("Performance: %.3f GOPS\n", gops);
    }

    if (printResults) {
        if (rank == 0) {
            globalDist.resize(numNodes * numNodes);
        }
        MPI_Gatherv(dist.data(), counts[rank], MPI_UNSIGNED,
                    rank == 0 ? globalDist.data() : nullptr, counts.data(), displacements.data(),
                    MPI_UNSIGNED, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            print_results_int(globalDist, "DistanceMatrix");
        }
    }

    bool valid = true;
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
        }
        valid = validateResult(dist, numNodes, firstRow, rowOwner, rank, MPI_COMM_WORLD);
        if (rank == 0) {
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
    }

    MPI_Finalize();
    return valid ? 0 : 1;
}
