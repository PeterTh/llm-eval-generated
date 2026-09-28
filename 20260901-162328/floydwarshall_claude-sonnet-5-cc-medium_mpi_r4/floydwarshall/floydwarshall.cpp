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

// Initializes the local row-block [rowStart, rowStart+localRows) of the path
// matrix. path[i][j] = i for every i,j (including the diagonal), so this can
// be computed independently by each rank without any communication.
void initializePathMatrixLocal(std::vector<unsigned int>& localPath, const size_t rowStart,
                                const size_t localRows, const size_t numNodes) {
    for (size_t li = 0; li < localRows; ++li) {
        const size_t globalI = rowStart + li;
        for (size_t j = 0; j < numNodes; ++j) {
            localPath[li * numNodes + j] = static_cast<unsigned int>(globalI);
        }
    }
}

// Computes a balanced row-block decomposition of numNodes rows across
// worldSize ranks: rowCounts[r] rows starting at rowStart[r].
void computeRowDistribution(const size_t numNodes, const int worldSize,
                             std::vector<size_t>& rowStart, std::vector<size_t>& rowCounts) {
    rowStart.assign(worldSize, 0);
    rowCounts.assign(worldSize, 0);

    const size_t base = numNodes / static_cast<size_t>(worldSize);
    const size_t rem = numNodes % static_cast<size_t>(worldSize);

    size_t offset = 0;
    for (int r = 0; r < worldSize; ++r) {
        const size_t count = base + (static_cast<size_t>(r) < rem ? 1 : 0);
        rowStart[r] = offset;
        rowCounts[r] = count;
        offset += count;
    }
}

// Determines which rank owns global row k, given the same balanced
// distribution produced by computeRowDistribution.
int ownerOfRow(const size_t k, const size_t numNodes, const int worldSize) {
    const size_t base = numNodes / static_cast<size_t>(worldSize);
    const size_t rem = numNodes % static_cast<size_t>(worldSize);
    const size_t largeBlockRows = rem * (base + 1);

    if (k < largeBlockRows) {
        return static_cast<int>(k / (base + 1));
    }
    return static_cast<int>(rem + (k - largeBlockRows) / base);
}

// Distributed Floyd-Warshall: each rank owns a contiguous block of rows
// [rowStart, rowStart+localRows). At each iteration k, the owner of row k
// broadcasts that row to all ranks, and every rank updates its own rows
// using that pivot row - equivalent to the sequential algorithm's row-major
// update order, just partitioned across ranks by source node.
void floydWarshallMPI(std::vector<unsigned int>& localDist,
                      std::vector<unsigned int>& localPath,
                      const size_t rowStart, const size_t localRows,
                      const size_t numNodes, MPI_Comm comm) {
    int worldSize;
    MPI_Comm_size(comm, &worldSize);

    std::vector<unsigned int> pivotRow(numNodes);

    for (size_t k = 0; k < numNodes; ++k) {
        const int owner = ownerOfRow(k, numNodes, worldSize);

        unsigned int* bcastBuf;
        int rank;
        MPI_Comm_rank(comm, &rank);
        if (rank == owner) {
            bcastBuf = &localDist[(k - rowStart) * numNodes];
        } else {
            bcastBuf = pivotRow.data();
        }

        MPI_Bcast(bcastBuf, static_cast<int>(numNodes), MPI_UNSIGNED, owner, comm);

        for (size_t li = 0; li < localRows; ++li) {
            const unsigned int distIK = localDist[li * numNodes + k];

            for (size_t j = 0; j < numNodes; ++j) {
                const unsigned int newDist = distIK + bcastBuf[j];

                if (newDist < localDist[li * numNodes + j]) {
                    localDist[li * numNodes + j] = newDist;
                    localPath[li * numNodes + j] = static_cast<unsigned int>(k);
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

    int worldRank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

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
            if (worldRank == 0) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            if (worldRank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (worldRank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("MPI ranks: %d\n", worldSize);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Determine the row-block owned by each rank
    std::vector<size_t> rowStart, rowCounts;
    computeRowDistribution(numNodes, worldSize, rowStart, rowCounts);
    const size_t localRows = rowCounts[worldRank];

    std::vector<int> sendCounts(worldSize), sendDispls(worldSize);
    for (int r = 0; r < worldSize; ++r) {
        sendCounts[r] = static_cast<int>(rowCounts[r] * numNodes);
        sendDispls[r] = static_cast<int>(rowStart[r] * numNodes);
    }

    // Initialize graph. The distance matrix uses a sequentially-dependent
    // PRNG (rand_r), so it is generated in full on rank 0 to preserve exact
    // semantics, then scattered as contiguous row-blocks. The path matrix has
    // no such dependency and is initialized directly by each rank.
    std::vector<unsigned int> fullDist;
    if (worldRank == 0) {
        printf("Initializing graph...\n");
        fullDist.resize(numNodes * numNodes);
        initializeDistanceMatrix(fullDist, numNodes, 1, MAX_DISTANCE);
    }

    std::vector<unsigned int> localDist(localRows * numNodes);
    MPI_Scatterv(worldRank == 0 ? fullDist.data() : nullptr, sendCounts.data(), sendDispls.data(),
                 MPI_UNSIGNED, localDist.data(), static_cast<int>(localRows * numNodes),
                 MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    fullDist.clear();
    fullDist.shrink_to_fit();

    std::vector<unsigned int> localPath(localRows * numNodes);
    initializePathMatrixLocal(localPath, rowStart[worldRank], localRows, numNodes);

    // Run Floyd-Warshall
    if (worldRank == 0) {
        printf("Computing shortest paths...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    floydWarshallMPI(localDist, localPath, rowStart[worldRank], localRows, numNodes, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto localDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    const long long localDurationMs = localDuration.count();
    long long durationMs = 0;
    MPI_Reduce(&localDurationMs, &durationMs, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather the full distance matrix on rank 0 for validation/reporting
    std::vector<unsigned int> dist;
    if (worldRank == 0) {
        dist.resize(numNodes * numNodes);
    }
    MPI_Gatherv(localDist.data(), static_cast<int>(localRows * numNodes), MPI_UNSIGNED,
                worldRank == 0 ? dist.data() : nullptr, sendCounts.data(), sendDispls.data(),
                MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    if (worldRank == 0) {
        printf("Computation time: %lld ms\n", durationMs);

        // Calculate operations per second
        // Floyd-Warshall has O(n³) complexity
        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / (durationMs / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);

        // Print results for external validation (integer hash-based)
        if (printResults) {
            print_results_int(dist, "DistanceMatrix");
        }
    }

    // Validation
    int validationResult = 0;
    if (validate) {
        if (worldRank == 0) {
            printf("Validating result...\n");
            bool valid = validateResult(dist, numNodes);

            if (valid) {
                printf("Validation: PASSED\n");
                validationResult = 0;
            } else {
                printf("Validation: FAILED\n");
                validationResult = 1;
            }
        }
        MPI_Bcast(&validationResult, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return validationResult;
}
