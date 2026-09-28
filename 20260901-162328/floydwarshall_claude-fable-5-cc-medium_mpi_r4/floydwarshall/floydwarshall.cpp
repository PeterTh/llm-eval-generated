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

// Distributed Floyd-Warshall: each rank owns a contiguous block of rows
// (source nodes). At iteration k the owner of row k broadcasts it, then all
// ranks update their local rows independently.
void floydWarshallMPI(std::vector<unsigned int>& localDist,
                      std::vector<unsigned int>& localPath,
                      const size_t numNodes,
                      const size_t rowStart,
                      const size_t localRows,
                      const std::vector<int>& rowOwner) {
    int myRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    std::vector<unsigned int> kRow(numNodes);

    for (size_t k = 0; k < numNodes; ++k) {
        const int owner = rowOwner[k];

        if (myRank == owner) {
            std::memcpy(kRow.data(), &localDist[(k - rowStart) * numNodes],
                        numNodes * sizeof(unsigned int));
        }
        MPI_Bcast(kRow.data(), (int)numNodes, MPI_UNSIGNED, owner, MPI_COMM_WORLD);

        const unsigned int pathK = (unsigned int)k;
        for (size_t li = 0; li < localRows; ++li) {
            unsigned int* distRow = &localDist[li * numNodes];
            unsigned int* pathRow = &localPath[li * numNodes];
            const unsigned int distIK = distRow[k];

            for (size_t j = 0; j < numNodes; ++j) {
                const unsigned int newDist = distIK + kRow[j];
                if (newDist < distRow[j]) {
                    distRow[j] = newDist;
                    pathRow[j] = pathK;
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

    int rank, numRanks;
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

    // Block row distribution across ranks
    std::vector<int> rowCounts(numRanks), rowDispls(numRanks);
    std::vector<int> elemCounts(numRanks), elemDispls(numRanks);
    {
        size_t base = numNodes / numRanks;
        size_t rem = numNodes % numRanks;
        size_t offset = 0;
        for (int r = 0; r < numRanks; ++r) {
            size_t rows = base + (static_cast<size_t>(r) < rem ? 1 : 0);
            rowCounts[r] = (int)rows;
            rowDispls[r] = (int)offset;
            elemCounts[r] = (int)(rows * numNodes);
            elemDispls[r] = (int)(offset * numNodes);
            offset += rows;
        }
    }
    const size_t rowStart = (size_t)rowDispls[rank];
    const size_t localRows = (size_t)rowCounts[rank];

    std::vector<int> rowOwner(numNodes);
    for (int r = 0; r < numRanks; ++r) {
        for (int i = 0; i < rowCounts[r]; ++i) {
            rowOwner[(size_t)rowDispls[r] + i] = r;
        }
    }

    // Full matrices only on rank 0; local row blocks everywhere
    std::vector<unsigned int> dist, path;
    std::vector<unsigned int> localDist(localRows * numNodes);
    std::vector<unsigned int> localPath(localRows * numNodes);

    if (rank == 0) {
        dist.resize(numNodes * numNodes);
        path.resize(numNodes * numNodes);

        printf("Initializing graph...\n");
        initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
        initializePathMatrix(path, numNodes);
    }

    MPI_Scatterv(dist.data(), elemCounts.data(), elemDispls.data(), MPI_UNSIGNED,
                 localDist.data(), elemCounts[rank], MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Scatterv(path.data(), elemCounts.data(), elemDispls.data(), MPI_UNSIGNED,
                 localPath.data(), elemCounts[rank], MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    // Run Floyd-Warshall
    if (rank == 0) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    floydWarshallMPI(localDist, localPath, numNodes, rowStart, localRows, rowOwner);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Gather results back to rank 0
    MPI_Gatherv(localDist.data(), elemCounts[rank], MPI_UNSIGNED,
                dist.data(), elemCounts.data(), elemDispls.data(), MPI_UNSIGNED,
                0, MPI_COMM_WORLD);
    MPI_Gatherv(localPath.data(), elemCounts[rank], MPI_UNSIGNED,
                path.data(), elemCounts.data(), elemDispls.data(), MPI_UNSIGNED,
                0, MPI_COMM_WORLD);

    int exitCode = 0;
    if (rank == 0) {
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
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }

    MPI_Finalize();
    return exitCode;
}
