#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

// The original layout is column-major: a column is one destination and its
// entries are all sources.  Distributing whole columns makes local updates
// contiguous and only requires one pivot-column broadcast per k.
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes,
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;
    for (size_t i = 0; i < numNodes * numNodes; ++i) {
        dist[i] = rangeMin + static_cast<unsigned int>(range * rand_r(&seed) /
                                                        static_cast<double>(RAND_MAX));
    }
    for (size_t i = 0; i < numNodes; ++i) {
        dist[idx2(i, i, numNodes)] = 0;
    }
}

void initializePathMatrix(std::vector<unsigned int>& path, const size_t numNodes) {
    for (size_t j = 0; j < numNodes; ++j) {
        for (size_t i = 0; i < numNodes; ++i) {
            path[idx2(i, j, numNodes)] = j;
            path[idx2(j, i, numNodes)] = i;
        }
        path[idx2(j, j, numNodes)] = j;
    }
}

int ownerOfColumn(const size_t column, const size_t numNodes, const int ranks) {
    const size_t base = numNodes / static_cast<size_t>(ranks);
    const size_t extra = numNodes % static_cast<size_t>(ranks);
    const size_t largerBlockColumns = extra * (base + 1);
    if (column < largerBlockColumns) {
        return static_cast<int>(column / (base + 1));
    }
    return static_cast<int>(extra + (column - largerBlockColumns) / base);
}

void floydWarshallMPI(std::vector<unsigned int>& localDist,
                      std::vector<unsigned int>& localPath,
                      const size_t numNodes, const size_t firstColumn,
                      const size_t localColumns, const int rank, const int ranks) {
    std::vector<unsigned int> pivot(numNodes);

    for (size_t k = 0; k < numNodes; ++k) {
        const int pivotOwner = ownerOfColumn(k, numNodes, ranks);
        if (rank == pivotOwner) {
            const size_t localPivotColumn = k - firstColumn;
            std::memcpy(pivot.data(), localDist.data() + localPivotColumn * numNodes,
                        numNodes * sizeof(unsigned int));
        }
        MPI_Bcast(pivot.data(), static_cast<int>(numNodes), MPI_UNSIGNED, pivotOwner,
                  MPI_COMM_WORLD);

        for (size_t localJ = 0; localJ < localColumns; ++localJ) {
            unsigned int* const distanceColumn = localDist.data() + localJ * numNodes;
            unsigned int* const pathColumn = localPath.data() + localJ * numNodes;
            const unsigned int distanceKJ = distanceColumn[k];

            // i is contiguous in the established column-major layout. Keeping
            // this loop simple enables compiler vectorization of the hot path.
            for (size_t i = 0; i < numNodes; ++i) {
                const unsigned int newDistance = pivot[i] + distanceKJ;
                if (newDistance < distanceColumn[i]) {
                    distanceColumn[i] = newDistance;
                    pathColumn[i] = static_cast<unsigned int>(k);
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
    int ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;
    int exitCode = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numNodes = static_cast<size_t>(atoi(argv[++i]));
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
            MPI_Finalize();
            return 1;
        }
    }

    if (numNodes > static_cast<size_t>(INT_MAX) ||
        (numNodes != 0 && numNodes > static_cast<size_t>(INT_MAX) / numNodes)) {
        if (rank == 0) printf("Number of nodes is too large for MPI collective counts\n");
        MPI_Finalize();
        return 1;
    }

    const size_t baseColumns = numNodes / static_cast<size_t>(ranks);
    const size_t extraColumns = numNodes % static_cast<size_t>(ranks);
    const size_t localColumns = baseColumns + (static_cast<size_t>(rank) < extraColumns);
    const size_t firstColumn = static_cast<size_t>(rank) * baseColumns +
                               std::min(static_cast<size_t>(rank), extraColumns);
    const size_t localElements = localColumns * numNodes;

    std::vector<int> counts(ranks);
    std::vector<int> displacements(ranks);
    for (int process = 0; process < ranks; ++process) {
        const size_t columns = baseColumns + (static_cast<size_t>(process) < extraColumns);
        counts[process] = static_cast<int>(columns * numNodes);
        displacements[process] = process == 0 ? 0 : displacements[process - 1] + counts[process - 1];
    }

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("MPI ranks: %d\n", ranks);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing graph...\n");
    }

    std::vector<unsigned int> globalDist;
    std::vector<unsigned int> globalPath;
    if (rank == 0) {
        globalDist.resize(numNodes * numNodes);
        globalPath.resize(numNodes * numNodes);
        initializeDistanceMatrix(globalDist, numNodes, 1, MAX_DISTANCE);
        initializePathMatrix(globalPath, numNodes);
    }
    std::vector<unsigned int> localDist(localElements);
    std::vector<unsigned int> localPath(localElements);
    MPI_Scatterv(rank == 0 ? globalDist.data() : nullptr, counts.data(), displacements.data(), MPI_UNSIGNED,
                 localDist.data(), static_cast<int>(localElements), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? globalPath.data() : nullptr, counts.data(), displacements.data(), MPI_UNSIGNED,
                 localPath.data(), static_cast<int>(localElements), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        // Root input copies are no longer needed during the distributed solve.
        std::vector<unsigned int>().swap(globalDist);
        std::vector<unsigned int>().swap(globalPath);
        printf("Computing shortest paths...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    floydWarshallMPI(localDist, localPath, numNodes, firstColumn, localColumns, rank, ranks);
    const double localSeconds = MPI_Wtime() - start;
    double elapsedSeconds = 0.0;
    MPI_Reduce(&localSeconds, &elapsedSeconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const long milliseconds = static_cast<long>(elapsedSeconds * 1000.0);
        printf("Computation time: %ld ms\n", milliseconds);
        const double ops = static_cast<double>(numNodes) * numNodes * numNodes;
        const double gflops = elapsedSeconds > 0.0 ? ops / elapsedSeconds / 1e9 : 0.0;
        printf("Performance: %.3f GOPS\n", gflops);
    }

    if (validate || printResults) {
        if (rank == 0) globalDist.resize(numNodes * numNodes);
        MPI_Gatherv(localDist.data(), static_cast<int>(localElements), MPI_UNSIGNED,
                    rank == 0 ? globalDist.data() : nullptr, counts.data(), displacements.data(), MPI_UNSIGNED,
                    0, MPI_COMM_WORLD);
        if (rank == 0 && printResults) print_results_int(globalDist, "DistanceMatrix");
        if (rank == 0 && validate) {
            printf("Validating result...\n");
            if (validateResult(globalDist, numNodes)) {
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
