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

// Distributed Floyd-Warshall over contiguous blocks of source rows i.
// The layout idx2(j, i, n) == i*n + j makes each source row contiguous in
// memory, so a rank owning rows [rowStart, rowStart+rowCount) holds the
// contiguous range dist[rowStart*n .. (rowStart+rowCount)*n).
// For every intermediate node k, the rank owning row k broadcasts it; all
// other data accessed (dist[i][k], dist[i][j]) is rank-local.
void floydWarshallMPI(unsigned int* __restrict__ localDist,
                      unsigned int* __restrict__ localPath,
                      const size_t numNodes, const size_t rowStart,
                      const size_t rowCount, const std::vector<int>& rowOwner,
                      unsigned int* __restrict__ rowK) {
    const size_t n = numNodes;
    int myRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    for (size_t k = 0; k < n; ++k) {
        const int owner = rowOwner[k];
        if (owner == myRank) {
            std::memcpy(rowK, localDist + (k - rowStart) * n, n * sizeof(unsigned int));
        }
        MPI_Bcast(rowK, static_cast<int>(n), MPI_UNSIGNED, owner, MPI_COMM_WORLD);

        for (size_t i = 0; i < rowCount; ++i) {
            unsigned int* distRow = localDist + i * n;
            unsigned int* pathRow = localPath + i * n;
            const unsigned int distIK = distRow[k];
            for (size_t j = 0; j < n; ++j) {
                const unsigned int newDist = distIK + rowK[j];
                if (newDist < distRow[j]) {
                    distRow[j] = newDist;
                    pathRow[j] = static_cast<unsigned int>(k);
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
        printf("MPI ranks: %d\n", numProcs);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Row-block decomposition: contiguous blocks of source rows per rank.
    std::vector<int> rowCounts(numProcs);
    std::vector<int> rowStarts(numProcs);
    {
        const size_t base = numNodes / numProcs;
        const size_t rem = numNodes % numProcs;
        size_t offset = 0;
        for (int p = 0; p < numProcs; ++p) {
            rowCounts[p] = static_cast<int>(base + (static_cast<size_t>(p) < rem ? 1 : 0));
            rowStarts[p] = static_cast<int>(offset);
            offset += rowCounts[p];
        }
    }
    const size_t rowStart = rowStarts[rank];
    const size_t rowCount = rowCounts[rank];

    // Owner of each row k (for the per-iteration broadcast)
    std::vector<int> rowOwner(numNodes);
    for (int p = 0; p < numProcs; ++p) {
        for (int r = rowStarts[p]; r < rowStarts[p] + rowCounts[p]; ++r) {
            rowOwner[r] = p;
        }
    }

    // Scatterv element counts/displacements (rows * numNodes elements each)
    std::vector<int> elemCounts(numProcs);
    std::vector<int> elemDispls(numProcs);
    for (int p = 0; p < numProcs; ++p) {
        elemCounts[p] = rowCounts[p] * static_cast<int>(numNodes);
        elemDispls[p] = rowStarts[p] * static_cast<int>(numNodes);
    }

    // Full matrices only on rank 0 (initialization, gather, output)
    std::vector<unsigned int> dist;
    std::vector<unsigned int> path;
    if (rank == 0) {
        dist.resize(numNodes * numNodes);
        path.resize(numNodes * numNodes);
        printf("Initializing graph...\n");
        initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
        initializePathMatrix(path, numNodes);
    }

    // Local row blocks
    std::vector<unsigned int> localDist(rowCount * numNodes);
    std::vector<unsigned int> localPath(rowCount * numNodes);
    std::vector<unsigned int> rowK(numNodes);

    MPI_Scatterv(rank == 0 ? dist.data() : nullptr, elemCounts.data(), elemDispls.data(),
                 MPI_UNSIGNED, localDist.data(), elemCounts[rank], MPI_UNSIGNED,
                 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? path.data() : nullptr, elemCounts.data(), elemDispls.data(),
                 MPI_UNSIGNED, localPath.data(), elemCounts[rank], MPI_UNSIGNED,
                 0, MPI_COMM_WORLD);

    // Run Floyd-Warshall
    if (rank == 0) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    floydWarshallMPI(localDist.data(), localPath.data(), numNodes,
                     rowStart, rowCount, rowOwner, rowK.data());

    MPI_Barrier(MPI_COMM_WORLD);
    const double end = MPI_Wtime();
    const long durationMs = static_cast<long>((end - start) * 1000.0);

    // Gather results back to rank 0
    MPI_Gatherv(localDist.data(), elemCounts[rank], MPI_UNSIGNED,
                rank == 0 ? dist.data() : nullptr, elemCounts.data(), elemDispls.data(),
                MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Gatherv(localPath.data(), elemCounts[rank], MPI_UNSIGNED,
                rank == 0 ? path.data() : nullptr, elemCounts.data(), elemDispls.data(),
                MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    int exitCode = 0;
    if (rank == 0) {
        printf("Computation time: %ld ms\n", durationMs);

        // Calculate operations per second
        // Floyd-Warshall has O(n³) complexity
        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / (durationMs / 1000.0) / 1e9;
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
