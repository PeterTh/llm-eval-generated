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

struct RowPartition {
    size_t start;
    size_t count;
};

inline size_t rowsForRank(const size_t numNodes, const int size, const int rank) noexcept {
    const size_t base = numNodes / static_cast<size_t>(size);
    const size_t rem = numNodes % static_cast<size_t>(size);
    return base + (static_cast<size_t>(rank) < rem ? 1U : 0U);
}

inline size_t startRowForRank(const size_t numNodes, const int size, const int rank) noexcept {
    const size_t base = numNodes / static_cast<size_t>(size);
    const size_t rem = numNodes % static_cast<size_t>(size);
    return static_cast<size_t>(rank) * base + std::min(static_cast<size_t>(rank), rem);
}

inline int ownerOfRow(const size_t numNodes, const int size, const size_t row) noexcept {
    const size_t base = numNodes / static_cast<size_t>(size);
    const size_t rem = numNodes % static_cast<size_t>(size);
    const size_t split = (base + 1U) * rem;
    if (row < split) {
        return static_cast<int>(row / (base + 1U));
    }
    return static_cast<int>(rem + (row - split) / base);
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

void initializeLocalPathMatrix(std::vector<unsigned int>& path, const size_t numNodes,
                               const size_t localRows) {
    for (size_t localRow = 0; localRow < localRows; ++localRow) {
        unsigned int* row = path.data() + localRow * numNodes;
        for (size_t j = 0; j < numNodes; ++j) {
            row[j] = static_cast<unsigned int>(j);
        }
    }
}

void floydWarshall(std::vector<unsigned int>& dist, 
                   std::vector<unsigned int>& path, 
                   const size_t numNodes) {
    // Classic Floyd-Warshall algorithm
    // For each intermediate node k
    for (size_t k = 0; k < numNodes; ++k) {
        // For each source node i
        for (size_t i = 0; i < numNodes; ++i) {
            // For each destination node j
            for (size_t j = 0; j < numNodes; ++j) {
                const unsigned int distIJ = dist[idx2(j, i, numNodes)];
                const unsigned int distIK = dist[idx2(k, i, numNodes)];
                const unsigned int distKJ = dist[idx2(j, k, numNodes)];
                
                const unsigned int newDist = distIK + distKJ;
                
                if (newDist < distIJ) {
                    dist[idx2(j, i, numNodes)] = newDist;
                    path[idx2(j, i, numNodes)] = k;
                }
            }
        }
    }
}

void floydWarshallMpi(std::vector<unsigned int>& localDist,
                      std::vector<unsigned int>& localPath,
                      const size_t numNodes,
                      const size_t localStart,
                      const size_t localRows,
                      const int rank,
                      const int size) {
    std::vector<unsigned int> rowK(numNodes);

    for (size_t k = 0; k < numNodes; ++k) {
        const int owner = ownerOfRow(numNodes, size, k);
        unsigned int* rowBuffer = rowK.data();
        if (rank == owner && localRows > 0) {
            rowBuffer = localDist.data() + (k - localStart) * numNodes;
        }

        MPI_Bcast(rowBuffer, static_cast<int>(numNodes), MPI_UNSIGNED, owner, MPI_COMM_WORLD);

        const unsigned int* rowKPtr = rowBuffer;
        for (size_t localRow = 0; localRow < localRows; ++localRow) {
            unsigned int* distRow = localDist.data() + localRow * numNodes;
            unsigned int* pathRow = localPath.data() + localRow * numNodes;
            const unsigned int distIK = distRow[k];

            for (size_t j = 0; j < numNodes; ++j) {
                const unsigned int newDist = distIK + rowKPtr[j];
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
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;
    bool parseOk = true;
    bool showHelp = false;

    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                numNodes = static_cast<size_t>(strtoull(argv[++i], nullptr, 10));
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = true;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = true;
            } else if (strcmp(argv[i], "-h") == 0) {
                showHelp = true;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                parseOk = false;
                break;
            }
        }
    }

    uint64_t numNodes64 = static_cast<uint64_t>(numNodes);
    int validateFlag = validate ? 1 : 0;
    int printFlag = printResults ? 1 : 0;
    int parseFlag = parseOk ? 1 : 0;
    int helpFlag = showHelp ? 1 : 0;

    MPI_Bcast(&numNodes64, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validateFlag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printFlag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&parseFlag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&helpFlag, 1, MPI_INT, 0, MPI_COMM_WORLD);

    numNodes = static_cast<size_t>(numNodes64);
    validate = validateFlag != 0;
    printResults = printFlag != 0;
    parseOk = parseFlag != 0;
    showHelp = helpFlag != 0;

    if (!parseOk || showHelp || numNodes == 0) {
        if (rank == 0) {
            if (showHelp) {
                printUsage(argv[0]);
            } else if (numNodes == 0) {
                printf("Number of nodes must be greater than zero\n");
                printUsage(argv[0]);
            }
        }
        MPI_Finalize();
        return (showHelp && parseOk) ? 0 : 1;
    }

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const size_t localRows = rowsForRank(numNodes, size, rank);
    const size_t localStart = startRowForRank(numNodes, size, rank);
    const size_t localCount = localRows * numNodes;

    std::vector<unsigned int> localDist(localCount);
    std::vector<unsigned int> localPath(localCount);

    std::vector<unsigned int> dist;
    if (rank == 0) {
        dist.resize(numNodes * numNodes);
        initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
    }
    std::vector<int> counts(size);
    std::vector<int> displs(size);
    for (int r = 0; r < size; ++r) {
        const size_t rows = rowsForRank(numNodes, size, r);
        counts[r] = static_cast<int>(rows * numNodes);
        displs[r] = static_cast<int>(startRowForRank(numNodes, size, r) * numNodes);
    }

    MPI_Scatterv(rank == 0 ? dist.data() : nullptr,
                 counts.data(),
                 displs.data(),
                 MPI_UNSIGNED,
                 localDist.data(),
                 static_cast<int>(localCount),
                 MPI_UNSIGNED,
                 0,
                 MPI_COMM_WORLD);

    initializeLocalPathMatrix(localPath, numNodes, localRows);

    if (rank == 0) {
        printf("Computing shortest paths...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    floydWarshallMpi(localDist, localPath, numNodes, localStart, localRows, rank, size);

    const double end = MPI_Wtime();
    const double localTime = end - start;
    double maxTime = 0.0;
    MPI_Reduce(&localTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const long durationMs = static_cast<long>(maxTime * 1000.0);
        printf("Computation time: %ld ms\n", durationMs);
        double ops = static_cast<double>(numNodes) * numNodes * numNodes;
        double gflops = ops / maxTime / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);
    }

    std::vector<unsigned int> distFull;
    if (printResults || validate) {
        if (rank == 0) {
            distFull.resize(numNodes * numNodes);
        }

        MPI_Gatherv(localDist.data(),
                    static_cast<int>(localCount),
                    MPI_UNSIGNED,
                    rank == 0 ? distFull.data() : nullptr,
                    counts.data(),
                    displs.data(),
                    MPI_UNSIGNED,
                    0,
                    MPI_COMM_WORLD);
    }

    if (rank == 0 && printResults) {
        print_results_int(distFull, "DistanceMatrix");
    }

    if (rank == 0 && validate) {
        printf("Validating result...\n");
        bool valid = validateResult(distFull, numNodes);
        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            MPI_Finalize();
            return 1;
        }
    }

    MPI_Finalize();
    return 0;
}
