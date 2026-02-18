#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

static inline void row_partition(const size_t n, const int rank, const int size,
                                 size_t& startRow, size_t& rowCount) noexcept {
    const size_t base = n / static_cast<size_t>(size);
    const size_t rem = n % static_cast<size_t>(size);
    if (static_cast<size_t>(rank) < rem) {
        rowCount = base + 1;
        startRow = static_cast<size_t>(rank) * rowCount;
    } else {
        rowCount = base;
        startRow = rem * (base + 1) + (static_cast<size_t>(rank) - rem) * base;
    }
}

static inline int owner_of_row(const size_t k, const size_t n, const int size) noexcept {
    const size_t base = n / static_cast<size_t>(size);
    const size_t rem = n % static_cast<size_t>(size);
    const size_t cutoff = rem * (base + 1);
    if (k < cutoff) {
        return static_cast<int>(k / (base + 1));
    }
    return static_cast<int>(rem + (k - cutoff) / base);
}

void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes,
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;

    for (size_t i = 0; i < numNodes * numNodes; ++i) {
        dist[i] = rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
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

static inline void floydWarshallMPI(std::vector<unsigned int>& localDist,
                                   std::vector<unsigned int>& localPath,
                                   const size_t numNodes,
                                   const size_t localStartRow,
                                   const size_t localRowCount,
                                   const int rank,
                                   const int size) {
    std::vector<unsigned int> rowK(numNodes);

    for (size_t k = 0; k < numNodes; ++k) {
        const int owner = owner_of_row(k, numNodes, size);

        if (rank == owner) {
            const size_t lk = k - localStartRow;
            std::memcpy(rowK.data(), localDist.data() + lk * numNodes, numNodes * sizeof(unsigned int));
        }

        MPI_Bcast(rowK.data(), static_cast<int>(numNodes), MPI_UNSIGNED, owner, MPI_COMM_WORLD);

        for (size_t li = 0; li < localRowCount; ++li) {
            unsigned int* __restrict__ di = localDist.data() + li * numNodes;
            unsigned int* __restrict__ pi = localPath.data() + li * numNodes;

            const unsigned int dik = di[k];
            for (size_t j = 0; j < numNodes; ++j) {
                const unsigned int newDist = dik + rowK[j];
                if (newDist < di[j]) {
                    di[j] = newDist;
                    pi[j] = static_cast<unsigned int>(k);
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

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;
    bool showHelp = false;
    bool parseError = false;

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
            parseError = true;
        }
    }

    int exitCode = 0;
    if (numNodes == 0) {
        if (rank == 0) {
            printf("Number of nodes must be > 0\n");
        }
        exitCode = 1;
        MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Finalize();
        return exitCode;
    }

    if (showHelp || parseError) {
        if (rank == 0) {
            if (parseError) {
                printf("Unknown option\n");
                exitCode = 1;
            }
            printUsage(argv[0]);
        }
        MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Finalize();
        return exitCode;
    }

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("MPI ranks: %d\n", size);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    size_t localStartRow = 0, localRowCount = 0;
    row_partition(numNodes, rank, size, localStartRow, localRowCount);
    const size_t localElems = localRowCount * numNodes;

    std::vector<unsigned int> dist;
    std::vector<unsigned int> path;
    if (rank == 0) {
        dist.resize(numNodes * numNodes);
        path.resize(numNodes * numNodes);
        printf("Initializing graph...\n");
        initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
        initializePathMatrix(path, numNodes);
    }

    std::vector<int> counts(size), displs(size);
    for (int r = 0; r < size; ++r) {
        size_t s = 0, c = 0;
        row_partition(numNodes, r, size, s, c);
        const size_t elems = c * numNodes;
        counts[r] = static_cast<int>(elems);
        displs[r] = static_cast<int>(s * numNodes);
    }

    std::vector<unsigned int> localDist(localElems);
    std::vector<unsigned int> localPath(localElems);

    MPI_Scatterv(rank == 0 ? dist.data() : nullptr, counts.data(), displs.data(), MPI_UNSIGNED,
                 localDist.data(), static_cast<int>(localElems), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? path.data() : nullptr, counts.data(), displs.data(), MPI_UNSIGNED,
                 localPath.data(), static_cast<int>(localElems), MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computing shortest paths...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();

    floydWarshallMPI(localDist, localPath, numNodes, localStartRow, localRowCount, rank, size);

    MPI_Barrier(MPI_COMM_WORLD);
    const double t1 = MPI_Wtime();

    const double localTime = t1 - t0;
    double maxTime = 0.0;
    MPI_Reduce(&localTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %.3f ms\n", maxTime * 1000.0);
        const double ops = (double)numNodes * (double)numNodes * (double)numNodes;
        const double gops = ops / maxTime / 1e9;
        printf("Performance: %.3f GOPS\n", gops);
    }

    if (rank == 0) {
        dist.assign(numNodes * numNodes, 0);
    }
    MPI_Gatherv(localDist.data(), static_cast<int>(localElems), MPI_UNSIGNED,
                rank == 0 ? dist.data() : nullptr, counts.data(), displs.data(), MPI_UNSIGNED,
                0, MPI_COMM_WORLD);

    if (rank == 0 && printResults) {
        print_results_int(dist, "DistanceMatrix");
    }

    if (rank == 0 && validate) {
        printf("Validating result...\n");
        const bool valid = validateResult(dist, numNodes);
        if (valid) {
            printf("Validation: PASSED\n");
            exitCode = 0;
        } else {
            printf("Validation: FAILED\n");
            exitCode = 1;
        }
    }

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
