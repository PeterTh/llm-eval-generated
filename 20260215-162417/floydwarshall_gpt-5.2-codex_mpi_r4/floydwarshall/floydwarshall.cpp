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

inline size_t rowsForRank(const size_t numNodes, const int worldSize, const int rank) {
    const size_t base = numNodes / static_cast<size_t>(worldSize);
    const size_t rem = numNodes % static_cast<size_t>(worldSize);
    return base + (static_cast<size_t>(rank) < rem ? 1 : 0);
}

inline size_t startRowForRank(const size_t numNodes, const int worldSize, const int rank) {
    const size_t base = numNodes / static_cast<size_t>(worldSize);
    const size_t rem = numNodes % static_cast<size_t>(worldSize);
    return static_cast<size_t>(rank) * base + std::min(static_cast<size_t>(rank), rem);
}

inline int ownerForRow(const size_t numNodes, const int worldSize, const size_t row) {
    const size_t base = numNodes / static_cast<size_t>(worldSize);
    const size_t rem = numNodes % static_cast<size_t>(worldSize);
    if (base == 0) {
        return static_cast<int>(row);
    }
    const size_t cutoff = (base + 1) * rem;
    if (row < cutoff) {
        return static_cast<int>(row / (base + 1));
    }
    return static_cast<int>((row - cutoff) / base + rem);
}

void floydWarshallMPI(std::vector<unsigned int>& distLocal,
                      std::vector<unsigned int>& pathLocal,
                      const size_t numNodes,
                      const size_t startRow,
                      const size_t localRows,
                      const int rank,
                      const int worldSize) {
    std::vector<unsigned int> rowK(numNodes);
    const int nCount = static_cast<int>(numNodes);
    for (size_t k = 0; k < numNodes; ++k) {
        const int owner = ownerForRow(numNodes, worldSize, k);
        if (owner == rank) {
            const size_t localIndex = k - startRow;
            const unsigned int* src = distLocal.data() + localIndex * numNodes;
            std::memcpy(rowK.data(), src, numNodes * sizeof(unsigned int));
        }
        MPI_Bcast(rowK.data(), nCount, MPI_UNSIGNED, owner, MPI_COMM_WORLD);
        for (size_t local_i = 0; local_i < localRows; ++local_i) {
            unsigned int* distRow = distLocal.data() + local_i * numNodes;
            unsigned int* pathRow = pathLocal.data() + local_i * numNodes;
            const unsigned int distIK = distRow[k];
            for (size_t j = 0; j < numNodes; ++j) {
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
    int worldRank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;
    bool showHelp = false;
    bool parseError = false;
    const char* badOption = nullptr;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numNodes = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            showHelp = true;
        } else {
            parseError = true;
            badOption = argv[i];
        }
    }

    if (showHelp) {
        if (worldRank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return 0;
    }

    if (parseError) {
        if (worldRank == 0) {
            printf("Unknown option: %s\n", badOption ? badOption : "");
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    if (worldRank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const size_t localRows = rowsForRank(numNodes, worldSize, worldRank);
    const size_t startRow = startRowForRank(numNodes, worldSize, worldRank);

    std::vector<int> counts(worldSize);
    std::vector<int> displs(worldSize);
    for (int r = 0; r < worldSize; ++r) {
        const size_t rows = rowsForRank(numNodes, worldSize, r);
        counts[r] = static_cast<int>(rows * numNodes);
        displs[r] = static_cast<int>(startRowForRank(numNodes, worldSize, r) * numNodes);
    }

    std::vector<unsigned int> distFull;
    std::vector<unsigned int> pathFull;
    if (worldRank == 0) {
        distFull.resize(numNodes * numNodes);
        pathFull.resize(numNodes * numNodes);
        printf("Initializing graph...\n");
        initializeDistanceMatrix(distFull, numNodes, 1, MAX_DISTANCE);
        initializePathMatrix(pathFull, numNodes);
    }

    std::vector<unsigned int> distLocal(localRows * numNodes);
    std::vector<unsigned int> pathLocal(localRows * numNodes);

    MPI_Scatterv(worldRank == 0 ? distFull.data() : nullptr,
                 counts.data(),
                 displs.data(),
                 MPI_UNSIGNED,
                 distLocal.data(),
                 counts[worldRank],
                 MPI_UNSIGNED,
                 0,
                 MPI_COMM_WORLD);

    MPI_Scatterv(worldRank == 0 ? pathFull.data() : nullptr,
                 counts.data(),
                 displs.data(),
                 MPI_UNSIGNED,
                 pathLocal.data(),
                 counts[worldRank],
                 MPI_UNSIGNED,
                 0,
                 MPI_COMM_WORLD);

    if (worldRank == 0) {
        printf("Computing shortest paths...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    floydWarshallMPI(distLocal, pathLocal, numNodes, startRow, localRows, worldRank, worldSize);

    MPI_Barrier(MPI_COMM_WORLD);
    const double end = MPI_Wtime();
    const double localElapsed = end - start;
    double maxElapsed = 0.0;
    MPI_Reduce(&localElapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (worldRank == 0) {
        const long long durationMs = static_cast<long long>(maxElapsed * 1000.0);
        printf("Computation time: %lld ms\n", durationMs);
        double ops = static_cast<double>(numNodes) * numNodes * numNodes;
        double gflops = ops / maxElapsed / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);
    }

    const bool needOutput = printResults || validate;
    if (needOutput) {
        if (worldRank == 0) {
            distFull.resize(numNodes * numNodes);
        }
        MPI_Gatherv(distLocal.data(),
                    counts[worldRank],
                    MPI_UNSIGNED,
                    worldRank == 0 ? distFull.data() : nullptr,
                    counts.data(),
                    displs.data(),
                    MPI_UNSIGNED,
                    0,
                    MPI_COMM_WORLD);
    }

    int exitCode = 0;
    if (worldRank == 0) {
        if (printResults) {
            print_results_int(distFull, "DistanceMatrix");
        }
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(distFull, numNodes);
            if (valid) {
                printf("Validation: PASSED\n");
                exitCode = 0;
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
