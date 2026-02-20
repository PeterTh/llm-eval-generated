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

// Index calculation for flattened 2D array (column-major: columns are contiguous)
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

inline constexpr size_t idx2_local(const size_t li, const size_t j, const size_t localRows) noexcept {
    return j * localRows + li;
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

static inline int owner_of_row(const size_t k, const size_t base, const size_t rem) noexcept {
    const size_t firstBig = (base + 1) * rem;
    if (k < firstBig) return (int)(k / (base + 1));
    return (int)(rem + (k - firstBig) / base);
}

static inline size_t local_index_of_row(const size_t k, const size_t base, const size_t rem) noexcept {
    const size_t firstBig = (base + 1) * rem;
    if (k < firstBig) return k % (base + 1);
    return (k - firstBig) % base;
}

void floydWarshallMPI(std::vector<unsigned int>& distLocal,
                      std::vector<unsigned int>& pathLocal,
                      const size_t numNodes,
                      const size_t localRows,
                      const size_t base,
                      const size_t rem,
                      MPI_Comm comm) {
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    std::vector<unsigned int> rowK(numNodes);

    // For each intermediate node k
    for (size_t k = 0; k < numNodes; ++k) {
        const int owner = owner_of_row(k, base, rem);
        const size_t lk = local_index_of_row(k, base, rem);

        // Owner packs row-k (dist[j,k]) into a contiguous buffer
        if (rank == owner) {
            for (size_t j = 0; j < numNodes; ++j) {
                rowK[j] = distLocal[idx2_local(lk, j, localRows)];
            }
        }

        MPI_Bcast(rowK.data(), (int)numNodes, MPI_UNSIGNED, owner, comm);

        if (localRows == 0) continue;

        unsigned int* const colK = distLocal.data() + k * localRows;

        // Loop j outermost for contiguous access in i (localRows)
        for (size_t j = 0; j < numNodes; ++j) {
            const unsigned int distKJ = rowK[j];
            unsigned int* const colJ = distLocal.data() + j * localRows;
            unsigned int* const pathJ = pathLocal.data() + j * localRows;

            for (size_t li = 0; li < localRows; ++li) {
                const unsigned int newDist = colK[li] + distKJ;
                if (newDist < colJ[li]) {
                    colJ[li] = newDist;
                    pathJ[li] = (unsigned int)k;
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
    int exitCode = 0;
    int earlyExit = 0;

    if (rank == 0) {
        // Parse command line arguments
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                numNodes = (size_t)atoi(argv[++i]);
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                earlyExit = 1;
                exitCode = 0;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                earlyExit = 1;
                exitCode = 1;
                break;
            }
        }
    }

    // Broadcast config (and early-exit status) to all ranks
    unsigned long long nn = (unsigned long long)numNodes;
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&earlyExit, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nn, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_INT, 0, MPI_COMM_WORLD);
    numNodes = (size_t)nn;

    if (earlyExit) {
        MPI_Finalize();
        return exitCode;
    }

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing graph...\n");
    }

    const size_t base = (size_t)size == 0 ? 0 : (numNodes / (size_t)size);
    const size_t rem = (size_t)size == 0 ? 0 : (numNodes % (size_t)size);
    const size_t localRows = base + ((size_t)rank < rem ? 1 : 0);

    // Global start row for this rank
    const size_t rowStart = (size_t)rank * base + std::min((size_t)rank, rem);

    // Allocate local matrices (column-major, but only a block of rows)
    std::vector<unsigned int> distLocal(numNodes * localRows);
    std::vector<unsigned int> pathLocal(numNodes * localRows);

    // Initialize path locally (matches serial semantics: path[j,i] == i)
    for (size_t j = 0; j < numNodes; ++j) {
        for (size_t li = 0; li < localRows; ++li) {
            pathLocal[idx2_local(li, j, localRows)] = (unsigned int)(rowStart + li);
        }
    }

    // Distribute distance matrix from rank 0 (exactly matches serial initialization)
    std::vector<unsigned int> distGlobal;
    if (rank == 0) {
        distGlobal.resize(numNodes * numNodes);
        initializeDistanceMatrix(distGlobal, numNodes, 1, MAX_DISTANCE);
    }

    std::vector<int> recvcounts(size);
    std::vector<int> displs(size);
    for (int r = 0; r < size; ++r) {
        const size_t rCount = base + ((size_t)r < rem ? 1 : 0);
        const size_t rStart = (size_t)r * base + std::min((size_t)r, rem);
        recvcounts[r] = (int)rCount;
        displs[r] = (int)rStart;
    }

    for (size_t j = 0; j < numNodes; ++j) {
        const unsigned int* sendCol = (rank == 0) ? (distGlobal.data() + j * numNodes) : nullptr;
        unsigned int* recvCol = distLocal.data() + j * localRows;
        MPI_Scatterv((void*)sendCol, recvcounts.data(), displs.data(), MPI_UNSIGNED,
                     (void*)recvCol, (int)localRows, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        printf("Computing shortest paths...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();

    floydWarshallMPI(distLocal, pathLocal, numNodes, localRows, base, rem, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    const double t1 = MPI_Wtime();
    const double localTime = t1 - t0;

    double maxTime = 0.0;
    MPI_Reduce((void*)&localTime, (void*)&maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const long ms = (long)std::llround(maxTime * 1000.0);
        printf("Computation time: %ld ms\n", ms);

        // Calculate operations per second (Floyd-Warshall is O(n^3))
        const double ops = (double)numNodes * (double)numNodes * (double)numNodes;
        const double gops = ops / maxTime / 1e9;
        printf("Performance: %.3f GOPS\n", gops);
    }

    // Gather full results only when requested (validation or external printing)
    if (validate || printResults) {
        if (rank == 0) distGlobal.assign(numNodes * numNodes, 0);

        for (size_t j = 0; j < numNodes; ++j) {
            const unsigned int* sendCol = distLocal.data() + j * localRows;
            unsigned int* recvCol = (rank == 0) ? (distGlobal.data() + j * numNodes) : nullptr;
            MPI_Gatherv((void*)sendCol, (int)localRows, MPI_UNSIGNED,
                        (void*)recvCol, recvcounts.data(), displs.data(), MPI_UNSIGNED,
                        0, MPI_COMM_WORLD);
        }

        if (rank == 0 && printResults) {
            print_results_int(distGlobal, "DistanceMatrix");
        }

        if (rank == 0 && validate) {
            printf("Validating result...\n");
            const bool valid = validateResult(distGlobal, numNodes);
            if (valid) {
                printf("Validation: PASSED\n");
                MPI_Finalize();
                return 0;
            }
            printf("Validation: FAILED\n");
            MPI_Finalize();
            return 1;
        }
    }

    MPI_Finalize();
    return 0;
}
