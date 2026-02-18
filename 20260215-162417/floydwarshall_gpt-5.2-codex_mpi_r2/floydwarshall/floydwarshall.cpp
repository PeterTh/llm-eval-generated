#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
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

struct ColumnDistribution {
    size_t local_cols;
    size_t col_offset;
};

ColumnDistribution getColumnDistribution(const size_t numNodes, const int rank, const int worldSize) {
    const size_t base = numNodes / static_cast<size_t>(worldSize);
    const size_t rem = numNodes % static_cast<size_t>(worldSize);
    const size_t local_cols = base + (static_cast<size_t>(rank) < rem ? 1 : 0);
    const size_t col_offset = base * static_cast<size_t>(rank) + std::min(static_cast<size_t>(rank), rem);
    return {local_cols, col_offset};
}

int columnOwner(const size_t k, const size_t numNodes, const int worldSize) {
    const size_t base = numNodes / static_cast<size_t>(worldSize);
    const size_t rem = numNodes % static_cast<size_t>(worldSize);
    if (k < (base + 1) * rem) {
        return static_cast<int>(k / (base + 1));
    }
    return static_cast<int>(rem + (k - (base + 1) * rem) / base);
}

void floydWarshallMPI(std::vector<unsigned int>& localDist,
                      std::vector<unsigned int>& localPath,
                      const size_t numNodes,
                      const size_t localCols,
                      const size_t colOffset,
                      const int worldSize,
                      const int worldRank) {
    std::vector<unsigned int> k_col(numNodes);

    for (size_t k = 0; k < numNodes; ++k) {
        const int owner = columnOwner(k, numNodes, worldSize);

        if (worldRank == owner) {
            const size_t local_col = k - colOffset;
            const unsigned int* col_ptr = localDist.data() + local_col * numNodes;
            std::memcpy(k_col.data(), col_ptr, numNodes * sizeof(unsigned int));
        }

        MPI_Bcast(k_col.data(), static_cast<int>(numNodes), MPI_UNSIGNED, owner, MPI_COMM_WORLD);

        for (size_t local_col = 0; local_col < localCols; ++local_col) {
            unsigned int* dist_col = localDist.data() + local_col * numNodes;
            unsigned int* path_col = localPath.data() + local_col * numNodes;
            const unsigned int distKJ = dist_col[k];

            for (size_t i = 0; i < numNodes; ++i) {
                const unsigned int newDist = k_col[i] + distKJ;
                if (newDist < dist_col[i]) {
                    dist_col[i] = newDist;
                    path_col[i] = static_cast<unsigned int>(k);
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
    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;

    MPI_Init(&argc, &argv);

    int worldRank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    int action = 0;
    if (worldRank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                numNodes = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = true;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = true;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                action = 1;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                action = 2;
                break;
            }
        }
    }

    MPI_Bcast(&action, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (action != 0) {
        MPI_Finalize();
        return action == 2 ? 1 : 0;
    }

    uint64_t numNodes64 = static_cast<uint64_t>(numNodes);
    int validateInt = validate ? 1 : 0;
    int printResultsInt = printResults ? 1 : 0;

    MPI_Bcast(&numNodes64, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validateInt, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResultsInt, 1, MPI_INT, 0, MPI_COMM_WORLD);

    numNodes = static_cast<size_t>(numNodes64);
    validate = validateInt != 0;
    printResults = printResultsInt != 0;

    if (worldRank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const ColumnDistribution distInfo = getColumnDistribution(numNodes, worldRank, worldSize);
    const size_t localCols = distInfo.local_cols;
    const size_t colOffset = distInfo.col_offset;
    const size_t localCount = localCols * numNodes;

    std::vector<unsigned int> localDist(localCount);
    std::vector<unsigned int> localPath(localCount);
    std::vector<unsigned int> dist;
    std::vector<unsigned int> path;

    std::vector<int> counts;
    std::vector<int> displs;
    if (worldRank == 0) {
        dist.resize(numNodes * numNodes);
        path.resize(numNodes * numNodes);

        printf("Initializing graph...\n");
        initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
        initializePathMatrix(path, numNodes);

        counts.resize(worldSize);
        displs.resize(worldSize);
        for (int rank = 0; rank < worldSize; ++rank) {
            const ColumnDistribution info = getColumnDistribution(numNodes, rank, worldSize);
            counts[rank] = static_cast<int>(info.local_cols * numNodes);
            displs[rank] = static_cast<int>(info.col_offset * numNodes);
        }
    }

    MPI_Scatterv(worldRank == 0 ? dist.data() : nullptr,
                 worldRank == 0 ? counts.data() : nullptr,
                 worldRank == 0 ? displs.data() : nullptr,
                 MPI_UNSIGNED,
                 localDist.data(),
                 static_cast<int>(localCount),
                 MPI_UNSIGNED,
                 0,
                 MPI_COMM_WORLD);

    MPI_Scatterv(worldRank == 0 ? path.data() : nullptr,
                 worldRank == 0 ? counts.data() : nullptr,
                 worldRank == 0 ? displs.data() : nullptr,
                 MPI_UNSIGNED,
                 localPath.data(),
                 static_cast<int>(localCount),
                 MPI_UNSIGNED,
                 0,
                 MPI_COMM_WORLD);

    if (worldRank == 0) {
        printf("Computing shortest paths...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    floydWarshallMPI(localDist, localPath, numNodes, localCols, colOffset, worldSize, worldRank);

    MPI_Barrier(MPI_COMM_WORLD);
    const double end = MPI_Wtime();
    const double elapsed = end - start;
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (worldRank == 0) {
        const long durationMs = static_cast<long>(maxElapsed * 1000.0);
        printf("Computation time: %ld ms\n", durationMs);

        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / maxElapsed / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);
    }

    if (printResults || validate) {
        if (worldRank == 0) {
            if (counts.empty()) {
                counts.resize(worldSize);
                displs.resize(worldSize);
                for (int rank = 0; rank < worldSize; ++rank) {
                    const ColumnDistribution info = getColumnDistribution(numNodes, rank, worldSize);
                    counts[rank] = static_cast<int>(info.local_cols * numNodes);
                    displs[rank] = static_cast<int>(info.col_offset * numNodes);
                }
            }
        }

        MPI_Gatherv(localDist.data(),
                    static_cast<int>(localCount),
                    MPI_UNSIGNED,
                    worldRank == 0 ? dist.data() : nullptr,
                    worldRank == 0 ? counts.data() : nullptr,
                    worldRank == 0 ? displs.data() : nullptr,
                    MPI_UNSIGNED,
                    0,
                    MPI_COMM_WORLD);
    }

    int resultCode = 0;
    if (worldRank == 0) {
        if (printResults) {
            print_results_int(dist, "DistanceMatrix");
        }

        if (validate) {
            printf("Validating result...\n");
            const bool valid = validateResult(dist, numNodes);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            resultCode = valid ? 0 : 1;
        }
    }

    MPI_Bcast(&resultCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return resultCode;
}
