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

void initializePathMatrix(std::vector<unsigned int>& path, const size_t numNodes) {
    for (size_t j = 0; j < numNodes; ++j) {
        for (size_t i = 0; i < numNodes; ++i) {
            path[idx2(i, j, numNodes)] = j;
            path[idx2(j, i, numNodes)] = i;
        }
        path[idx2(j, j, numNodes)] = j;
    }
}

// Row-block distribution: rank r owns rows [rowStart(r), rowStart(r) + rowCount(r))
inline size_t rowStart(const int rank, const size_t numNodes, const int numRanks) {
    const size_t base = numNodes / numRanks;
    const size_t rem = numNodes % numRanks;
    return static_cast<size_t>(rank) * base + std::min(static_cast<size_t>(rank), rem);
}

inline size_t rowCount(const int rank, const size_t numNodes, const int numRanks) {
    const size_t base = numNodes / numRanks;
    const size_t rem = numNodes % numRanks;
    return base + (static_cast<size_t>(rank) < rem ? 1 : 0);
}

inline int rowOwner(const size_t row, const size_t numNodes, const int numRanks) {
    const size_t base = numNodes / numRanks;
    const size_t rem = numNodes % numRanks;
    if (row < rem * (base + 1)) {
        return static_cast<int>(row / (base + 1));
    }
    return static_cast<int>(rem + (row - rem * (base + 1)) / base);
}

// Relax one row i against pivot row k
inline void updateRow(unsigned int* __restrict rowI, unsigned int* __restrict pathI,
                      const unsigned int* __restrict rowK,
                      const unsigned int distIK, const size_t k, const size_t numNodes) {
    for (size_t j = 0; j < numNodes; ++j) {
        const unsigned int newDist = distIK + rowK[j];
        if (newDist < rowI[j]) {
            rowI[j] = newDist;
            pathI[j] = static_cast<unsigned int>(k);
        }
    }
}

// Distributed Floyd-Warshall: each rank updates only its own block of rows.
// The pivot row for step k+1 is updated early by its owner and broadcast
// asynchronously, overlapping communication with the remaining row updates.
// Since all ranks initialize the full matrix identically, row 0 needs no
// broadcast.
void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes,
                   const int rank, const int numRanks) {
    const size_t myStart = rowStart(rank, numNodes, numRanks);
    const size_t myEnd = myStart + rowCount(rank, numNodes, numRanks);

    for (size_t k = 0; k < numNodes; ++k) {
        const unsigned int* rowK = &dist[idx2(0, k, numNodes)];
        const size_t next = k + 1;
        const bool havePipeline = next < numNodes;
        const bool ownNext = havePipeline && next >= myStart && next < myEnd;

        // Owner relaxes the next pivot row first so it can be sent immediately
        if (ownNext) {
            updateRow(&dist[idx2(0, next, numNodes)], &path[idx2(0, next, numNodes)],
                      rowK, dist[idx2(k, next, numNodes)], k, numNodes);
        }

        MPI_Request req = MPI_REQUEST_NULL;
        if (havePipeline) {
            MPI_Ibcast(&dist[idx2(0, next, numNodes)], static_cast<int>(numNodes),
                       MPI_UNSIGNED, rowOwner(next, numNodes, numRanks),
                       MPI_COMM_WORLD, &req);
        }

        // Relax the remaining owned rows while the broadcast is in flight
        for (size_t i = myStart; i < myEnd; ++i) {
            if (ownNext && i == next) continue;
            updateRow(&dist[idx2(0, i, numNodes)], &path[idx2(0, i, numNodes)],
                      rowK, dist[idx2(k, i, numNodes)], k, numNodes);
        }

        if (havePipeline) {
            MPI_Wait(&req, MPI_STATUS_IGNORE);
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
    int numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

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

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("MPI ranks: %d\n", numRanks);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Allocate matrices (full-size on every rank; each rank updates only its rows)
    std::vector<unsigned int> dist(numNodes * numNodes);
    std::vector<unsigned int> path(numNodes * numNodes);

    // Initialize (deterministic, identical on all ranks)
    if (rank == 0) printf("Initializing graph...\n");
    initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
    initializePathMatrix(path, numNodes);

    // Run Floyd-Warshall
    if (rank == 0) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    floydWarshall(dist, path, numNodes, rank, numRanks);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long localDuration = duration.count();
    long maxDuration = 0;
    MPI_Reduce(&localDuration, &maxDuration, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather the distributed row blocks onto rank 0 for output/validation
    std::vector<int> counts(numRanks);
    std::vector<int> displs(numRanks);
    for (int r = 0; r < numRanks; ++r) {
        counts[r] = static_cast<int>(rowCount(r, numNodes, numRanks) * numNodes);
        displs[r] = static_cast<int>(rowStart(r, numNodes, numRanks) * numNodes);
    }
    if (rank == 0) {
        MPI_Gatherv(MPI_IN_PLACE, 0, MPI_UNSIGNED,
                    dist.data(), counts.data(), displs.data(), MPI_UNSIGNED,
                    0, MPI_COMM_WORLD);
    } else {
        MPI_Gatherv(&dist[static_cast<size_t>(displs[rank])], counts[rank], MPI_UNSIGNED,
                    nullptr, nullptr, nullptr, MPI_UNSIGNED,
                    0, MPI_COMM_WORLD);
    }

    int exitCode = 0;

    if (rank == 0) {
        printf("Computation time: %ld ms\n", maxDuration);

        // Calculate operations per second
        // Floyd-Warshall has O(n³) complexity
        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / (maxDuration / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);

        // Print results for external validation (integer hash-based)
        if (printResults) {
            print_results_int(dist, "DistanceMatrix");
        }

        // Validation
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(dist, numNodes);

            if (valid) {
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
