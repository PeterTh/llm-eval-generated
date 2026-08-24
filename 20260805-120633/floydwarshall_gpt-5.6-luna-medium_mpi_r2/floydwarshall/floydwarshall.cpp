#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

// The original benchmark uses column-major storage: idx2(i, j) is entry
// (i, j), and columns are contiguous.  Columns are the distributed unit.
inline constexpr size_t idx2(const size_t i, const size_t j,
                             const size_t n) noexcept {
    return j * n + i;
}

void initializeDistanceMatrix(std::vector<unsigned int>& dist,
                              const size_t numNodes,
                              const unsigned int rangeMin,
                              const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;

    for (size_t i = 0; i < numNodes * numNodes; ++i) {
        dist[i] = rangeMin + static_cast<unsigned int>(
            range * rand_r(&seed) / static_cast<double>(RAND_MAX));
    }

    for (size_t i = 0; i < numNodes; ++i) {
        dist[idx2(i, i, numNodes)] = 0;
    }
}

void initializeLocalPathMatrix(std::vector<unsigned int>& path,
                               const size_t firstColumn,
                               const size_t localColumns,
                               const size_t numNodes) {
    // Every initial path entry is the destination column, exactly as in the
    // serial initialization.  Only local columns need to be represented.
    for (size_t column = 0; column < localColumns; ++column) {
        std::fill_n(path.data() + column * numNodes, numNodes,
                    static_cast<unsigned int>(firstColumn + column));
    }
}

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes,
                   const size_t localColumns,
                   const std::vector<size_t>& columnStarts,
                   const MPI_Comm communicator) {
    std::vector<unsigned int> intermediateColumn(numNodes);
    int rank = 0;
    MPI_Comm_rank(communicator, &rank);

    for (size_t k = 0; k < numNodes; ++k) {
        const auto ownerIt = std::upper_bound(columnStarts.begin(),
                                              columnStarts.end(), k);
        const int owner = static_cast<int>(ownerIt - columnStarts.begin() - 1);
        const size_t ownerOffset = k - columnStarts[owner];

        if (rank == owner) {
            const size_t localOffset = ownerOffset * numNodes;
            std::copy_n(dist.data() + localOffset, numNodes,
                        intermediateColumn.data());
        }

        MPI_Bcast(intermediateColumn.data(), static_cast<int>(numNodes),
                  MPI_UNSIGNED, owner, communicator);

        // Work is arranged by column so both the destination row and path
        // accesses are unit-stride in the innermost loop.
        for (size_t column = 0; column < localColumns; ++column) {
            unsigned int* const distanceColumn = dist.data() + column * numNodes;
            unsigned int* const pathColumn = path.data() + column * numNodes;
            const unsigned int distanceKToDestination = distanceColumn[k];

            for (size_t source = 0; source < numNodes; ++source) {
                const unsigned int newDistance =
                    intermediateColumn[source] + distanceKToDestination;
                if (newDistance < distanceColumn[source]) {
                    distanceColumn[source] = newDistance;
                    pathColumn[source] = static_cast<unsigned int>(k);
                }
            }
        }
    }
}

bool validateResult(const std::vector<unsigned int>& dist,
                    const size_t numNodes) {
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
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;
    bool parseOk = true;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numNodes = static_cast<size_t>(std::strtoull(argv[++i], nullptr, 10));
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            parseOk = false;
        }
    }

    if (!parseOk || numNodes == 0 || numNodes > static_cast<size_t>(INT_MAX)) {
        MPI_Finalize();
        return 1;
    }

    std::vector<size_t> columnStarts(static_cast<size_t>(worldSize) + 1, 0);
    for (int r = 0; r < worldSize; ++r) {
        columnStarts[static_cast<size_t>(r) + 1] =
            columnStarts[static_cast<size_t>(r)] +
            numNodes / static_cast<size_t>(worldSize) +
            (static_cast<size_t>(r) < numNodes % static_cast<size_t>(worldSize));
    }
    const size_t firstColumn = columnStarts[static_cast<size_t>(rank)];
    const size_t localColumns = columnStarts[static_cast<size_t>(rank) + 1] - firstColumn;

    std::vector<int> counts(worldSize);
    std::vector<int> displacements(worldSize);
    for (int r = 0; r < worldSize; ++r) {
        const size_t count = (columnStarts[static_cast<size_t>(r) + 1] -
                              columnStarts[static_cast<size_t>(r)]) * numNodes;
        counts[r] = static_cast<int>(count);
        displacements[r] = static_cast<int>(columnStarts[static_cast<size_t>(r)] * numNodes);
    }

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing graph...\n");
        printf("Computing shortest paths...\n");
    }

    std::vector<unsigned int> globalDist;
    if (rank == 0) {
        globalDist.resize(numNodes * numNodes);
        initializeDistanceMatrix(globalDist, numNodes, 1, MAX_DISTANCE);
    }
    std::vector<unsigned int> dist(localColumns * numNodes);
    MPI_Scatterv(rank == 0 ? globalDist.data() : nullptr, counts.data(),
                 displacements.data(), MPI_UNSIGNED, dist.data(), counts[rank],
                 MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    std::vector<unsigned int> path(localColumns * numNodes);
    initializeLocalPathMatrix(path, firstColumn, localColumns, numNodes);

    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::high_resolution_clock::now();
    floydWarshall(dist, path, numNodes, localColumns,
                  columnStarts, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    const auto end = std::chrono::high_resolution_clock::now();

    const double localSeconds = std::chrono::duration<double>(end - start).count();
    double seconds = 0.0;
    MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);

    if (rank == 0) {
        const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::duration<double>(seconds));
        const long long milliseconds = duration.count();
        printf("Computation time: %lld ms\n", milliseconds);
        printf("Performance: %.3f GOPS\n",
               static_cast<double>(numNodes) * numNodes * numNodes / seconds / 1e9);
    }

    const bool needGlobalResult = printResults || validate;
    if (needGlobalResult) {
        if (rank == 0) globalDist.resize(numNodes * numNodes);
        MPI_Gatherv(dist.data(), counts[rank], MPI_UNSIGNED,
                    rank == 0 ? globalDist.data() : nullptr, counts.data(),
                    displacements.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            if (printResults) print_results_int(globalDist, "DistanceMatrix");
            if (validate) {
                printf("Validating result...\n");
                const bool valid = validateResult(globalDist, numNodes);
                printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
                MPI_Finalize();
                return valid ? 0 : 1;
            }
        }
    }

    MPI_Finalize();
    return 0;
}
