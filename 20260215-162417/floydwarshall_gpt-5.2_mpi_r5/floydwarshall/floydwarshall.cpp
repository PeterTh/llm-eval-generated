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

// Index calculation for flattened 2D array
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

struct RowPartition {
    size_t startRow;
    size_t numRows;
};

static inline RowPartition partitionForRank(const size_t n, const int size, const int rank) noexcept {
    const size_t base = n / static_cast<size_t>(size);
    const size_t rem = n % static_cast<size_t>(size);

    const size_t r = static_cast<size_t>(rank);
    const size_t extra = (r < rem) ? 1u : 0u;
    const size_t numRows = base + extra;
    const size_t startRow = r * base + std::min(r, rem);
    return {startRow, numRows};
}

static inline int ownerOfRow(const size_t k, const size_t base, const size_t rem) noexcept {
    const size_t threshold = (base + 1u) * rem;
    if (k < threshold) {
        return static_cast<int>(k / (base + 1u));
    }
    return static_cast<int>(rem + (k - threshold) / base);
}

void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes,
                             const unsigned int rangeMin, const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;

    for (size_t i = 0; i < numNodes * numNodes; ++i) {
        dist[i] = rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
    }

    // Set diagonal to zero (distance from node to itself is 0)
    for (size_t i = 0; i < numNodes; ++i) {
        dist[idx2(i, i, numNodes)] = 0;
    }
}

static inline void initializeLocalPathMatrix(std::vector<unsigned int>& pathLocal,
                                            const size_t numNodes,
                                            const RowPartition part) {
    for (size_t li = 0; li < part.numRows; ++li) {
        const unsigned int globalI = static_cast<unsigned int>(part.startRow + li);
        unsigned int* row = pathLocal.data() + li * numNodes;
        std::fill(row, row + numNodes, globalI);
    }
}

static void floydWarshallMPI(std::vector<unsigned int>& distLocal,
                            std::vector<unsigned int>& pathLocal,
                            const size_t numNodes,
                            const RowPartition part,
                            const int rank,
                            const int size) {
    const size_t base = numNodes / static_cast<size_t>(size);
    const size_t rem = numNodes % static_cast<size_t>(size);

    std::vector<unsigned int> rowK(numNodes);

    for (size_t k = 0; k < numNodes; ++k) {
        const int owner = ownerOfRow(k, base, rem);

        const unsigned int* rk = nullptr;
        if (rank == owner) {
            const size_t localK = k - part.startRow;
            rk = distLocal.data() + localK * numNodes;
            MPI_Bcast(const_cast<unsigned int*>(rk), static_cast<int>(numNodes), MPI_UNSIGNED, owner, MPI_COMM_WORLD);
        } else {
            MPI_Bcast(rowK.data(), static_cast<int>(numNodes), MPI_UNSIGNED, owner, MPI_COMM_WORLD);
            rk = rowK.data();
        }

        const unsigned int kVal = static_cast<unsigned int>(k);

        for (size_t li = 0; li < part.numRows; ++li) {
            unsigned int* rowI = distLocal.data() + li * numNodes;
            unsigned int* pathI = pathLocal.data() + li * numNodes;

            const unsigned int distIK = rowI[k];
            const unsigned int* __restrict rkp = rk;

            // Update row i using broadcasted row k
            for (size_t j = 0; j < numNodes; ++j) {
                const unsigned int newDist = distIK + rkp[j];
                if (newDist < rowI[j]) {
                    rowI[j] = newDist;
                    pathI[j] = kVal;
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

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t numNodes = 512;
    int validate = 0;
    int printResults = 0;
    int parseOk = 1;
    int parseExitCode = 0;

    if (rank == 0) {
        // Parse command line arguments
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                numNodes = static_cast<size_t>(atoi(argv[++i]));
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                parseOk = 0;
                parseExitCode = 0;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                parseOk = 0;
                parseExitCode = 1;
                break;
            }
        }

        if (parseOk) {
            printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
            printf("Number of nodes: %zu\n", numNodes);
            printf("Validation: %s\n", validate ? "enabled" : "disabled");
            printf("Initializing graph...\n");
        }
    }

    MPI_Bcast(&parseOk, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&parseExitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!parseOk) {
        MPI_Finalize();
        return parseExitCode;
    }

    // Broadcast settings
    unsigned long long numNodesULL = static_cast<unsigned long long>(numNodes);
    MPI_Bcast(&numNodesULL, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    numNodes = static_cast<size_t>(numNodesULL);
    MPI_Bcast(&validate, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_INT, 0, MPI_COMM_WORLD);

    const RowPartition part = partitionForRank(numNodes, size, rank);

    // Prepare sendcounts/displs for scatter/gather
    std::vector<int> sendcounts;
    std::vector<int> displs;
    if (rank == 0) {
        sendcounts.resize(size);
        displs.resize(size);
        for (int r = 0; r < size; ++r) {
            const RowPartition pr = partitionForRank(numNodes, size, r);
            const size_t elems = pr.numRows * numNodes;
            sendcounts[r] = static_cast<int>(elems);
            displs[r] = static_cast<int>(pr.startRow * numNodes);
        }
    }

    std::vector<unsigned int> distGlobal;
    if (rank == 0) {
        distGlobal.resize(numNodes * numNodes);
        initializeDistanceMatrix(distGlobal, numNodes, 1, MAX_DISTANCE);
    }

    std::vector<unsigned int> distLocal(part.numRows * numNodes);
    MPI_Scatterv(rank == 0 ? distGlobal.data() : nullptr,
                 rank == 0 ? sendcounts.data() : nullptr,
                 rank == 0 ? displs.data() : nullptr,
                 MPI_UNSIGNED,
                 distLocal.empty() ? nullptr : distLocal.data(),
                 static_cast<int>(part.numRows * numNodes),
                 MPI_UNSIGNED,
                 0,
                 MPI_COMM_WORLD);

    std::vector<unsigned int> pathLocal(part.numRows * numNodes);
    initializeLocalPathMatrix(pathLocal, numNodes, part);

    if (rank == 0) {
        printf("Computing shortest paths...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();

    floydWarshallMPI(distLocal, pathLocal, numNodes, part, rank, size);

    MPI_Barrier(MPI_COMM_WORLD);
    const double t1 = MPI_Wtime();

    const double localTime = t1 - t0;
    double maxTime = 0.0;
    MPI_Reduce(&localTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather result for output/validation
    if (rank == 0) {
        distGlobal.assign(numNodes * numNodes, 0);
    }

    MPI_Gatherv(distLocal.empty() ? nullptr : distLocal.data(),
                static_cast<int>(part.numRows * numNodes),
                MPI_UNSIGNED,
                rank == 0 ? distGlobal.data() : nullptr,
                rank == 0 ? sendcounts.data() : nullptr,
                rank == 0 ? displs.data() : nullptr,
                MPI_UNSIGNED,
                0,
                MPI_COMM_WORLD);

    int exitCode = 0;
    if (rank == 0) {
        const long ms = static_cast<long>(std::llround(maxTime * 1000.0));
        printf("Computation time: %ld ms\n", ms);

        // Calculate operations per second
        // Floyd-Warshall has O(n³) complexity
        const double ops = (double)numNodes * (double)numNodes * (double)numNodes;
        const double gops = ops / maxTime / 1e9;
        printf("Performance: %.3f GOPS\n", gops);

        if (printResults) {
            print_results_int(distGlobal, "DistanceMatrix");
        }

        if (validate) {
            printf("Validating result...\n");
            const bool valid = validateResult(distGlobal, numNodes);
            if (valid) {
                printf("Validation: PASSED\n");
                exitCode = 0;
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }

    MPI_Finalize();
    return exitCode;
}
