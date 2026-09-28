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

// Computes the row-block distribution of `numNodes` rows across `numProcs`
// ranks: rows are handed out contiguously, with the first `numNodes %
// numProcs` ranks receiving one extra row each.
void computeRowDistribution(const size_t numNodes, const int numProcs,
                             std::vector<int>& rowCounts, std::vector<int>& rowDispls) {
    rowCounts.assign(numProcs, 0);
    rowDispls.assign(numProcs, 0);

    const size_t base = numNodes / static_cast<size_t>(numProcs);
    const size_t remainder = numNodes % static_cast<size_t>(numProcs);

    int displ = 0;
    for (int p = 0; p < numProcs; ++p) {
        const size_t count = base + (static_cast<size_t>(p) < remainder ? 1 : 0);
        rowCounts[p] = static_cast<int>(count);
        rowDispls[p] = displ;
        displ += static_cast<int>(count);
    }
}

// Distributed Floyd-Warshall: each rank owns a contiguous block of rows of
// `dist`/`path`. For each pivot k, the rank owning row k broadcasts it to
// all other ranks, which then locally update their own rows. This mirrors
// the well-known row-partitioned parallelization of Floyd-Warshall and
// produces bit-identical results to the sequential algorithm because a
// row's update during its own pivot iteration is always a no-op
// (dist[k][k] == 0).
void floydWarshallMPI(std::vector<unsigned int>& distLocal,
                      std::vector<unsigned int>& pathLocal,
                      const size_t numNodes,
                      const std::vector<int>& rowCounts,
                      const std::vector<int>& rowDispls,
                      const int rank) {
    const size_t myRowStart = static_cast<size_t>(rowDispls[rank]);
    const size_t myRowCount = static_cast<size_t>(rowCounts[rank]);

    // Precompute which rank owns each pivot row.
    std::vector<int> rowOwner(numNodes);
    {
        const int numProcs = static_cast<int>(rowCounts.size());
        for (int p = 0; p < numProcs; ++p) {
            const size_t start = static_cast<size_t>(rowDispls[p]);
            const size_t count = static_cast<size_t>(rowCounts[p]);
            for (size_t r = start; r < start + count; ++r) {
                rowOwner[r] = p;
            }
        }
    }

    std::vector<unsigned int> pivotRow(numNodes);

    for (size_t k = 0; k < numNodes; ++k) {
        const int owner = rowOwner[k];

        if (owner == rank) {
            const size_t localK = k - myRowStart;
            std::memcpy(pivotRow.data(), &distLocal[localK * numNodes], numNodes * sizeof(unsigned int));
        }

        MPI_Bcast(pivotRow.data(), static_cast<int>(numNodes), MPI_UNSIGNED, owner, MPI_COMM_WORLD);

        for (size_t li = 0; li < myRowCount; ++li) {
            const size_t rowBase = li * numNodes;
            const unsigned int distIK = distLocal[rowBase + k];

            for (size_t j = 0; j < numNodes; ++j) {
                const unsigned int distIJ = distLocal[rowBase + j];
                const unsigned int distKJ = pivotRow[j];

                const unsigned int newDist = distIK + distKJ;

                if (newDist < distIJ) {
                    distLocal[rowBase + j] = newDist;
                    pathLocal[rowBase + j] = static_cast<unsigned int>(k);
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
    int numProcs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (argv is identical on every rank under MPI)
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numNodes = atoi(argv[++i]);
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
            MPI_Finalize();
            return 1;
        }
    }

    if (numProcs > 1 && numNodes < static_cast<size_t>(numProcs)) {
        if (rank == 0) {
            printf("Error: number of nodes (%zu) must be >= number of MPI ranks (%d)\n",
                   numNodes, numProcs);
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("MPI ranks: %d\n", numProcs);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Row-block distribution of the matrices across ranks.
    std::vector<int> rowCounts, rowDispls;
    computeRowDistribution(numNodes, numProcs, rowCounts, rowDispls);

    // Element counts/displacements (each row is `numNodes` contiguous elements).
    std::vector<int> elemCounts(numProcs), elemDispls(numProcs);
    for (int p = 0; p < numProcs; ++p) {
        elemCounts[p] = rowCounts[p] * static_cast<int>(numNodes);
        elemDispls[p] = rowDispls[p] * static_cast<int>(numNodes);
    }

    // Only rank 0 holds the full matrices, generated identically to the
    // original sequential implementation so results are unchanged.
    std::vector<unsigned int> dist;
    std::vector<unsigned int> path;
    if (rank == 0) {
        printf("Initializing graph...\n");
        dist.resize(numNodes * numNodes);
        path.resize(numNodes * numNodes);
        initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
        initializePathMatrix(path, numNodes);
    }

    const size_t myRowCount = static_cast<size_t>(rowCounts[rank]);
    std::vector<unsigned int> distLocal(myRowCount * numNodes);
    std::vector<unsigned int> pathLocal(myRowCount * numNodes);

    MPI_Scatterv(rank == 0 ? dist.data() : nullptr, elemCounts.data(), elemDispls.data(),
                 MPI_UNSIGNED, distLocal.data(), static_cast<int>(distLocal.size()),
                 MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? path.data() : nullptr, elemCounts.data(), elemDispls.data(),
                 MPI_UNSIGNED, pathLocal.data(), static_cast<int>(pathLocal.size()),
                 MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computing shortest paths...\n");
    }

    auto start = std::chrono::high_resolution_clock::now();
    MPI_Barrier(MPI_COMM_WORLD);

    floydWarshallMPI(distLocal, pathLocal, numNodes, rowCounts, rowDispls, rank);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();

    // Gather the updated row blocks back into the full matrices on rank 0.
    MPI_Gatherv(distLocal.data(), static_cast<int>(distLocal.size()), MPI_UNSIGNED,
                rank == 0 ? dist.data() : nullptr, elemCounts.data(), elemDispls.data(),
                MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Gatherv(pathLocal.data(), static_cast<int>(pathLocal.size()), MPI_UNSIGNED,
                rank == 0 ? path.data() : nullptr, elemCounts.data(), elemDispls.data(),
                MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate operations per second
        // Floyd-Warshall has O(n³) complexity
        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
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
                MPI_Finalize();
                return 0;
            } else {
                printf("Validation: FAILED\n");
                MPI_Finalize();
                return 1;
            }
        }
    }

    MPI_Finalize();
    return 0;
}
