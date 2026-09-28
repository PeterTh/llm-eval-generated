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

// Computes a contiguous row-block decomposition of `numNodes` rows across
// `numProcs` MPI ranks: rank r owns rows [rowStart[r], rowStart[r]+rowCount[r]).
void computeRowDistribution(const size_t numNodes, const int numProcs,
                            std::vector<int>& rowCount, std::vector<int>& rowStart) {
    rowCount.assign(numProcs, 0);
    rowStart.assign(numProcs, 0);

    const size_t baseRows = numNodes / static_cast<size_t>(numProcs);
    const size_t remainder = numNodes % static_cast<size_t>(numProcs);

    size_t offset = 0;
    for (int r = 0; r < numProcs; ++r) {
        const size_t rows = baseRows + (static_cast<size_t>(r) < remainder ? 1 : 0);
        rowCount[r] = static_cast<int>(rows);
        rowStart[r] = static_cast<int>(offset);
        offset += rows;
    }
}

// Distributed Floyd-Warshall using MPI row-block decomposition.
// `dist` and `path` (row-major, dist[i*n+j] / path[i*n+j]) must hold valid
// full matrices on rank 0; on return, rank 0 holds the complete result.
void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes) {
    int rank = 0;
    int numProcs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    const int n = static_cast<int>(numNodes);

    std::vector<int> rowCount, rowStart;
    computeRowDistribution(numNodes, numProcs, rowCount, rowStart);

    std::vector<int> elemCount(numProcs), elemDispl(numProcs);
    for (int r = 0; r < numProcs; ++r) {
        elemCount[r] = rowCount[r] * n;
        elemDispl[r] = rowStart[r] * n;
    }

    const int localRows = rowCount[rank];
    const int localStart = rowStart[rank];

    std::vector<unsigned int> localDist(static_cast<size_t>(localRows) * n);
    std::vector<unsigned int> localPath(static_cast<size_t>(localRows) * n);

    MPI_Scatterv(dist.data(), elemCount.data(), elemDispl.data(), MPI_UNSIGNED,
                 localDist.data(), localRows * n, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Scatterv(path.data(), elemCount.data(), elemDispl.data(), MPI_UNSIGNED,
                 localPath.data(), localRows * n, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    // Buffer holding row k's distance and path values, broadcast each iteration.
    std::vector<unsigned int> rowK(static_cast<size_t>(2) * n);
    unsigned int* rowKDist = rowK.data();
    unsigned int* rowKPath = rowK.data() + n;

    for (int k = 0; k < n; ++k) {
        // Determine which rank owns row k (row blocks are contiguous).
        int owner = 0;
        for (int r = numProcs - 1; r >= 0; --r) {
            if (rowCount[r] > 0 && k >= rowStart[r]) {
                owner = r;
                break;
            }
        }

        if (rank == owner) {
            const int localK = k - localStart;
            std::memcpy(rowKDist, &localDist[static_cast<size_t>(localK) * n], n * sizeof(unsigned int));
            std::memcpy(rowKPath, &localPath[static_cast<size_t>(localK) * n], n * sizeof(unsigned int));
        }

        MPI_Bcast(rowK.data(), 2 * n, MPI_UNSIGNED, owner, MPI_COMM_WORLD);

        for (int i = 0; i < localRows; ++i) {
            unsigned int* rowIDist = &localDist[static_cast<size_t>(i) * n];
            unsigned int* rowIPath = &localPath[static_cast<size_t>(i) * n];
            const unsigned int distIK = rowIDist[k];

            for (int j = 0; j < n; ++j) {
                const unsigned int newDist = distIK + rowKDist[j];
                if (newDist < rowIDist[j]) {
                    rowIDist[j] = newDist;
                    rowIPath[j] = k;
                }
            }
        }
    }

    MPI_Gatherv(localDist.data(), localRows * n, MPI_UNSIGNED,
                dist.data(), elemCount.data(), elemDispl.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Gatherv(localPath.data(), localRows * n, MPI_UNSIGNED,
                path.data(), elemCount.data(), elemDispl.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
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
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

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

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Allocate matrices. Only rank 0 needs the full matrices (used to
    // scatter/gather row blocks); other ranks keep them empty to avoid
    // needlessly duplicating O(n^2) memory across the cluster.
    std::vector<unsigned int> dist(rank == 0 ? numNodes * numNodes : 0);
    std::vector<unsigned int> path(rank == 0 ? numNodes * numNodes : 0);

    // Initialize
    if (rank == 0) {
        printf("Initializing graph...\n");
        initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
        initializePathMatrix(path, numNodes);
        printf("Computing shortest paths...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    floydWarshall(dist, path, numNodes);

    auto end = std::chrono::high_resolution_clock::now();
    long localMs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long durationMs = 0;
    MPI_Reduce(&localMs, &durationMs, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

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
    }

    // Validation
    if (validate) {
        bool valid = true;
        if (rank == 0) {
            printf("Validating result...\n");
            valid = validateResult(dist, numNodes);
        }
        int validFlag = valid ? 1 : 0;
        MPI_Bcast(&validFlag, 1, MPI_INT, 0, MPI_COMM_WORLD);
        valid = validFlag != 0;

        if (valid) {
            if (rank == 0) {
                printf("Validation: PASSED\n");
            }
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Validation: FAILED\n");
            }
            MPI_Finalize();
            return 1;
        }
    }

    MPI_Finalize();
    return 0;
}
