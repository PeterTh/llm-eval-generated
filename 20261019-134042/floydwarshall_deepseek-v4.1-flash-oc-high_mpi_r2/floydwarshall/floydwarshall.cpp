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
                              const size_t rowBegin, const size_t rowEnd,
                              const int rank, const int numProcs,
                              const std::vector<size_t>& rowStart,
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    const size_t localRows = rowEnd - rowBegin;

    std::vector<int> counts(numProcs);
    std::vector<int> displs(numProcs);
    for (int r = 0; r < numProcs; ++r) {
        counts[r] = static_cast<int>((rowStart[r + 1] - rowStart[r]) * numNodes);
        displs[r] = (counts[r] > 0) ? static_cast<int>(rowStart[r] * numNodes) : 0;
    }

    std::vector<unsigned int> full;
    if (rank == 0 && numNodes > 0) {
        full.resize(numNodes * numNodes);
        unsigned int seed = 42;
        const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;
        for (size_t j = 0; j < numNodes; ++j) {
            for (size_t i = 0; i < numNodes; ++i) {
                full[i * numNodes + j] =
                    rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
            }
        }
    }

    if (numNodes > 0) {
        MPI_Scatterv((rank == 0) ? full.data() : nullptr,
                     counts.data(), displs.data(), MPI_UNSIGNED,
                     dist.data(), static_cast<int>(localRows * numNodes), MPI_UNSIGNED,
                     0, MPI_COMM_WORLD);
    }

    // Set diagonal to zero (distance from node to itself is 0)
    for (size_t i = rowBegin; i < rowEnd; ++i) {
        dist[(i - rowBegin) * numNodes + i] = 0;
    }
}

void initializePathMatrix(std::vector<unsigned int>& path, const size_t numNodes,
                          const size_t rowBegin, const size_t rowEnd) {
    const size_t localRows = rowEnd - rowBegin;
    for (size_t li = 0; li < localRows; ++li) {
        unsigned int* row = &path[li * numNodes];
        for (size_t j = 0; j < numNodes; ++j) {
            row[j] = static_cast<unsigned int>(j);
        }
    }
}

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes,
                   const size_t rowBegin,
                   const size_t rowEnd,
                   const int rank,
                   const std::vector<int>& owner) {
    if (numNodes == 0) {
        return;
    }

    const int numNodesInt = static_cast<int>(numNodes);
    std::vector<unsigned int> kRow(numNodes);

    for (size_t k = 0; k < numNodes; ++k) {
        const int root = owner[k];

        // The process that owns row k is the only one with the up-to-date row.
        if (rank == root) {
            std::memcpy(kRow.data(), &dist[(k - rowBegin) * numNodes],
                        numNodes * sizeof(unsigned int));
        }
        MPI_Bcast(kRow.data(), numNodesInt, MPI_UNSIGNED, root, MPI_COMM_WORLD);

        const unsigned int* __restrict__ kr = kRow.data();
        const unsigned int kVal = static_cast<unsigned int>(k);

        for (size_t i = rowBegin; i < rowEnd; ++i) {
            unsigned int* __restrict__ row = &dist[(i - rowBegin) * numNodes];
            unsigned int* __restrict__ prow = &path[(i - rowBegin) * numNodes];
            const unsigned int distIK = row[k];

            for (size_t j = 0; j < numNodes; ++j) {
                const unsigned int newDist = distIK + kr[j];
                if (newDist < row[j]) {
                    row[j] = newDist;
                    prow[j] = kVal;
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

void gatherResult(std::vector<unsigned int>& out,
                  const std::vector<unsigned int>& localDist,
                  const size_t numNodes,
                  const int rank, const int numProcs,
                  const std::vector<size_t>& rowStart) {
    std::vector<int> counts(numProcs);
    std::vector<int> displs(numProcs);
    for (int r = 0; r < numProcs; ++r) {
        counts[r] = static_cast<int>((rowStart[r + 1] - rowStart[r]) * numNodes);
        displs[r] = (counts[r] > 0) ? static_cast<int>(rowStart[r] * numNodes) : 0;
    }

    std::vector<unsigned int> gathered;
    if (rank == 0) {
        gathered.resize(numNodes * numNodes);
    }

    MPI_Gatherv(localDist.data(), static_cast<int>(localDist.size()), MPI_UNSIGNED,
                (rank == 0) ? gathered.data() : nullptr,
                counts.data(), displs.data(), MPI_UNSIGNED,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        out.resize(numNodes * numNodes);
        for (size_t i = 0; i < numNodes; ++i) {
            for (size_t j = 0; j < numNodes; ++j) {
                out[idx2(i, j, numNodes)] = gathered[i * numNodes + j];
            }
        }
    }
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
    int exitCode = 0;

    // Parse command line arguments
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

    // Contiguous row-block distribution of the distance/path rows.
    std::vector<size_t> rowStart(numProcs + 1);
    for (int r = 0; r <= numProcs; ++r) {
        rowStart[r] = static_cast<size_t>(r) * numNodes / static_cast<size_t>(numProcs);
    }

    const size_t rowBegin = rowStart[rank];
    const size_t rowEnd = rowStart[rank + 1];
    const size_t localRows = rowEnd - rowBegin;

    std::vector<int> owner(numNodes);
    {
        int r = 0;
        for (size_t k = 0; k < numNodes; ++k) {
            while (r + 1 < numProcs && k >= rowStart[r + 1]) {
                ++r;
            }
            owner[k] = r;
        }
    }

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing graph...\n");
    }

    // Allocate local row blocks
    std::vector<unsigned int> dist(localRows * numNodes);
    std::vector<unsigned int> path(localRows * numNodes);

    // Initialize
    initializeDistanceMatrix(dist, numNodes, rowBegin, rowEnd, rank, numProcs,
                             rowStart, 1, MAX_DISTANCE);
    initializePathMatrix(path, numNodes, rowBegin, rowEnd);

    if (rank == 0) {
        printf("Computing shortest paths...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);

    auto start = std::chrono::high_resolution_clock::now();

    floydWarshall(dist, path, numNodes, rowBegin, rowEnd, rank, owner);

    auto end = std::chrono::high_resolution_clock::now();
    double localSeconds = std::chrono::duration<double>(end - start).count();
    double maxSeconds = 0.0;
    MPI_Reduce(&localSeconds, &maxSeconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        long long durationMs = static_cast<long long>(maxSeconds * 1000.0);
        printf("Computation time: %lld ms\n", durationMs);

        // Calculate operations per second
        // Floyd-Warshall has O(n³) complexity
        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = (maxSeconds > 0.0) ? (ops / maxSeconds / 1e9) : 0.0;
        printf("Performance: %.3f GOPS\n", gflops);
    }

    std::vector<unsigned int> fullDist;
    if (printResults || validate) {
        gatherResult(fullDist, dist, numNodes, rank, numProcs, rowStart);
    }

    // Print results for external validation (integer hash-based)
    if (printResults && rank == 0) {
        print_results_int(fullDist, "DistanceMatrix");
    }

    // Validation
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
            bool valid = validateResult(fullDist, numNodes);
            exitCode = valid ? 0 : 1;
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return exitCode;
}
