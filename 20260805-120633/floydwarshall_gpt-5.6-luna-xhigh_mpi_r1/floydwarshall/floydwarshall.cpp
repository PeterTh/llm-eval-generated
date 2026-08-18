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

void initializePathMatrix(std::vector<unsigned int>& path, const size_t numNodes,
                          const size_t firstColumn, const size_t numColumns) {
    // The original matrix is column-major and path[i,j] is initialized to j.
    // Initialize only the columns owned by this rank.
    for (size_t localColumn = 0; localColumn < numColumns; ++localColumn) {
        const unsigned int column = static_cast<unsigned int>(firstColumn + localColumn);
        unsigned int* pathColumn = path.data() + localColumn * numNodes;
        std::fill_n(pathColumn, numNodes, column);
    }
}

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes,
                   const size_t firstColumn,
                   const size_t numColumns,
                   const std::vector<size_t>& columnStarts,
                   const int rank) {
    // The matrix is distributed in contiguous blocks of columns.  For each k,
    // the owner broadcasts D[:,k], after which every rank updates its columns
    // independently.  This is the column-major equivalent of the original
    // i/j loop nest and requires only O(n) communication per iteration.
    std::vector<unsigned int> pivotColumn(numNodes);

    for (size_t k = 0; k < numNodes; ++k) {
        const int owner = static_cast<int>(
            std::upper_bound(columnStarts.begin(), columnStarts.end(), k) -
            columnStarts.begin() - 1);

        if (rank == owner) {
            const size_t localPivotColumn = k - firstColumn;
            std::memcpy(pivotColumn.data(),
                        dist.data() + localPivotColumn * numNodes,
                        numNodes * sizeof(unsigned int));
        }

        MPI_Bcast(pivotColumn.data(), static_cast<int>(numNodes), MPI_UNSIGNED,
                  owner, MPI_COMM_WORLD);

        for (size_t localColumn = 0; localColumn < numColumns; ++localColumn) {
            unsigned int* distanceColumn = dist.data() + localColumn * numNodes;
            unsigned int* pathColumn = path.data() + localColumn * numNodes;
            const unsigned int distIK = distanceColumn[k];

            for (size_t j = 0; j < numNodes; ++j) {
                const unsigned int newDist = distIK + pivotColumn[j];
                if (newDist < distanceColumn[j]) {
                    distanceColumn[j] = newDist;
                    pathColumn[j] = static_cast<unsigned int>(k);
                }
            }
        }
    }
}

size_t columnStart(const size_t numNodes, const int rank, const int worldSize) {
    const size_t baseColumns = numNodes / static_cast<size_t>(worldSize);
    const size_t extraColumns = numNodes % static_cast<size_t>(worldSize);
    return static_cast<size_t>(rank) * baseColumns +
           std::min(static_cast<size_t>(rank), extraColumns);
}

size_t columnCount(const size_t numNodes, const int rank, const int worldSize) {
    const size_t baseColumns = numNodes / static_cast<size_t>(worldSize);
    const size_t extraColumns = numNodes % static_cast<size_t>(worldSize);
    return baseColumns + (static_cast<size_t>(rank) < extraColumns ? 1 : 0);
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

    if (numNodes > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (rank == 0) {
            printf("Number of nodes is too large for MPI counts\n");
        }
        MPI_Finalize();
        return 1;
    }

    if (numNodes != 0 &&
        numNodes > std::numeric_limits<size_t>::max() / numNodes) {
        if (rank == 0) {
            printf("The matrix is too large to address\n");
        }
        MPI_Finalize();
        return 1;
    }

    const size_t firstColumn = columnStart(numNodes, rank, worldSize);
    const size_t numColumns = columnCount(numNodes, rank, worldSize);
    const size_t localElements = numColumns * numNodes;

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing graph...\n");
    }

    // Every rank allocates only its owned columns.  Rank 0 uses a temporary
    // complete matrix as the deterministic initialization source and releases
    // it immediately after distribution.
    std::vector<unsigned int> dist(localElements);
    std::vector<unsigned int> initialDist(rank == 0 ? numNodes * numNodes : 0);
    std::vector<unsigned int> path(localElements);

    if (rank == 0) {
        initializeDistanceMatrix(initialDist, numNodes, 1, MAX_DISTANCE);
    }
    initializePathMatrix(path, numNodes, firstColumn, numColumns);

    std::vector<size_t> columnStarts(static_cast<size_t>(worldSize) + 1);
    for (int process = 0; process <= worldSize; ++process) {
        columnStarts[static_cast<size_t>(process)] =
            columnStart(numNodes, process, worldSize);
    }
    columnStarts.back() = numNodes;

    std::vector<int> elementCounts(static_cast<size_t>(worldSize));
    std::vector<int> elementDisplacements(static_cast<size_t>(worldSize));
    for (int process = 0; process < worldSize; ++process) {
        const size_t processCount = columnCount(numNodes, process, worldSize) * numNodes;
        const size_t processOffset = columnStart(numNodes, process, worldSize) * numNodes;
        if (processCount > static_cast<size_t>(std::numeric_limits<int>::max()) ||
            processOffset > static_cast<size_t>(std::numeric_limits<int>::max())) {
            if (rank == 0) {
                printf("The matrix distribution is too large for MPI counts\n");
            }
            MPI_Finalize();
            return 1;
        }
        elementCounts[static_cast<size_t>(process)] = static_cast<int>(processCount);
        elementDisplacements[static_cast<size_t>(process)] = static_cast<int>(processOffset);
    }

    MPI_Scatterv(rank == 0 ? initialDist.data() : nullptr,
                 elementCounts.data(), elementDisplacements.data(), MPI_UNSIGNED,
                 dist.data(), static_cast<int>(localElements), MPI_UNSIGNED,
                 0, MPI_COMM_WORLD);
    std::vector<unsigned int>().swap(initialDist);

    if (rank == 0) {
        printf("Computing shortest paths...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    floydWarshall(dist, path, numNodes, firstColumn, numColumns,
                  columnStarts, rank);

    const double elapsed = MPI_Wtime() - start;
    double computationTime = 0.0;
    MPI_Reduce(&elapsed, &computationTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    const bool needGlobalResult = printResults || validate;
    std::vector<unsigned int> globalDist(
        rank == 0 && needGlobalResult ? numNodes * numNodes : 0);
    if (needGlobalResult) {
        MPI_Gatherv(dist.data(), static_cast<int>(localElements), MPI_UNSIGNED,
                    rank == 0 ? globalDist.data() : nullptr,
                    rank == 0 ? elementCounts.data() : nullptr,
                    rank == 0 ? elementDisplacements.data() : nullptr,
                    MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        const long long milliseconds = static_cast<long long>(computationTime * 1000.0);
        printf("Computation time: %lld ms\n", milliseconds);

        // Floyd-Warshall has O(n^3) complexity.
        const double ops = static_cast<double>(numNodes) * numNodes * numNodes;
        const double gflops = computationTime > 0.0
                                  ? ops / computationTime / 1e9
                                  : 0.0;
        printf("Performance: %.3f GOPS\n", gflops);

        if (printResults) {
            print_results_int(globalDist, "DistanceMatrix");
        }

        if (validate) {
            printf("Validating result...\n");
            const bool valid = validateResult(globalDist, numNodes);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
            MPI_Finalize();
            return valid ? 0 : 1;
        }
    }

    MPI_Finalize();
    return 0;
}
