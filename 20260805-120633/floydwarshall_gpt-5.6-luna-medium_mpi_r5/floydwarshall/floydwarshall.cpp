#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
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

void floydWarshall(std::vector<unsigned int>& localDist,
                   std::vector<unsigned int>& localPath,
                   const size_t numNodes,
                   const size_t firstRow,
                   const size_t localRows,
                   const int rank,
                   const std::vector<size_t>& rowStarts) {
    // Rows are source vertices.  At each step, broadcast the completed row k;
    // this is the only data dependency between distributed row blocks.
    std::vector<unsigned int> pivot(numNodes);

    for (size_t k = 0; k < numNodes; ++k) {
        const auto ownerIt = std::upper_bound(rowStarts.begin(), rowStarts.end(), k);
        const int owner = static_cast<int>(ownerIt - rowStarts.begin() - 1);

        if (rank == owner) {
            const size_t localK = k - firstRow;
            std::copy_n(localDist.begin() + localK * numNodes, numNodes, pivot.begin());
        }
        MPI_Bcast(pivot.data(), static_cast<int>(numNodes), MPI_UNSIGNED, owner,
                  MPI_COMM_WORLD);

        for (size_t localI = 0; localI < localRows; ++localI) {
            unsigned int* row = localDist.data() + localI * numNodes;
            unsigned int* pathRow = localPath.data() + localI * numNodes;
            const unsigned int distIK = row[k];

            // j is the innermost index because rows are contiguous in memory.
            for (size_t j = 0; j < numNodes; ++j) {
                const unsigned int newDist = distIK + pivot[j];
                if (newDist < row[j]) {
                    row[j] = newDist;
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
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
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

    if (numNodes > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (rank == 0) printf("Number of nodes is too large for MPI counts\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    std::vector<size_t> rowStarts(static_cast<size_t>(worldSize) + 1, 0);
    for (int r = 0; r <= worldSize; ++r) {
        rowStarts[static_cast<size_t>(r)] =
            numNodes * static_cast<size_t>(r) / static_cast<size_t>(worldSize);
    }
    const size_t firstRow = rowStarts[static_cast<size_t>(rank)];
    const size_t localRows = rowStarts[static_cast<size_t>(rank + 1)] - firstRow;

    std::vector<unsigned int> globalDist;
    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI processes: %d\n", worldSize);
        printf("Initializing graph...\n");
        globalDist.resize(numNodes * numNodes);
        initializeDistanceMatrix(globalDist, numNodes, 1, MAX_DISTANCE);
    }

    std::vector<unsigned int> localDist(localRows * numNodes);
    std::vector<unsigned int> localPath(localRows * numNodes);
    for (size_t localI = 0; localI < localRows; ++localI) {
        std::fill_n(localPath.begin() + localI * numNodes, numNodes,
                    static_cast<unsigned int>(firstRow + localI));
    }

    std::vector<int> counts(static_cast<size_t>(worldSize));
    std::vector<int> displacements(static_cast<size_t>(worldSize));
    for (int r = 0; r < worldSize; ++r) {
        const size_t count = (rowStarts[static_cast<size_t>(r + 1)] -
                              rowStarts[static_cast<size_t>(r)]) * numNodes;
        const size_t displacement = rowStarts[static_cast<size_t>(r)] * numNodes;
        if (count > static_cast<size_t>(std::numeric_limits<int>::max()) ||
            displacement > static_cast<size_t>(std::numeric_limits<int>::max())) {
            if (rank == 0) printf("Matrix is too large for MPI counts\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        counts[static_cast<size_t>(r)] = static_cast<int>(count);
        displacements[static_cast<size_t>(r)] = static_cast<int>(displacement);
    }

    MPI_Scatterv(rank == 0 ? globalDist.data() : nullptr, counts.data(),
                 displacements.data(), MPI_UNSIGNED, localDist.data(),
                 counts[static_cast<size_t>(rank)], MPI_UNSIGNED, 0,
                 MPI_COMM_WORLD);

    if (rank == 0) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    floydWarshall(localDist, localPath, numNodes, firstRow, localRows, rank, rowStarts);

    MPI_Barrier(MPI_COMM_WORLD);
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    MPI_Gatherv(localDist.data(), counts[static_cast<size_t>(rank)], MPI_UNSIGNED,
                rank == 0 ? globalDist.data() : nullptr, counts.data(),
                displacements.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const long durationMs = static_cast<long>(maxElapsed * 1000.0);
        printf("Computation time: %ld ms\n", durationMs);

        // Floyd-Warshall has O(n³) complexity.
        const double ops = static_cast<double>(numNodes) * numNodes * numNodes;
        double gflops = maxElapsed > 0.0 ? ops / maxElapsed / 1e9 : 0.0;
        printf("Performance: %.3f GOPS\n", gflops);

        // Print results for external validation (integer hash-based).
        if (printResults) print_results_int(globalDist, "DistanceMatrix");

        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(globalDist, numNodes);

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
