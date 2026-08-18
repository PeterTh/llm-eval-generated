#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mpi.h>
#include <vector>

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

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes, const size_t firstColumn,
                   const size_t localColumns, const int rank, const int worldSize) {
    // Columns (destinations) are distributed contiguously.  A pivot column is
    // broadcast once per k, after which all local columns can be updated without
    // communication.  This preserves the original column-major layout.
    std::vector<unsigned int> pivot(numNodes);
    for (size_t k = 0; k < numNodes; ++k) {
        const size_t base = numNodes / static_cast<size_t>(worldSize);
        const size_t remainder = numNodes % static_cast<size_t>(worldSize);
        const int owner = k < (base + 1) * remainder
            ? static_cast<int>(k / (base + 1))
            : static_cast<int>(remainder + (k - (base + 1) * remainder) / base);

        if (rank == owner) {
            const size_t localK = k - firstColumn;
            std::copy_n(dist.data() + localK * numNodes, numNodes, pivot.data());
        }
        MPI_Bcast(pivot.data(), static_cast<int>(numNodes), MPI_UNSIGNED, owner, MPI_COMM_WORLD);

        for (size_t localJ = 0; localJ < localColumns; ++localJ) {
            unsigned int* const distColumn = dist.data() + localJ * numNodes;
            unsigned int* const pathColumn = path.data() + localJ * numNodes;
            const unsigned int distKJ = distColumn[k];
            for (size_t i = 0; i < numNodes; ++i) {
                const unsigned int distIJ = distColumn[i];
                const unsigned int distIK = pivot[i];
                const unsigned int newDist = distIK + distKJ;
                if (newDist < distIJ) {
                    distColumn[i] = newDist;
                    pathColumn[i] = k;
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

    unsigned long long nodeCount = numNodes;
    int validationFlag = validate;
    int resultsFlag = printResults;
    MPI_Bcast(&nodeCount, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validationFlag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&resultsFlag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    numNodes = static_cast<size_t>(nodeCount);
    validate = validationFlag != 0;
    printResults = resultsFlag != 0;

    if (numNodes == 0 || numNodes > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        numNodes * numNodes > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (rank == 0) printf("Number of nodes is outside the supported MPI count range\n");
        MPI_Finalize();
        return 1;
    }

    const size_t baseColumns = numNodes / static_cast<size_t>(worldSize);
    const size_t extraColumns = numNodes % static_cast<size_t>(worldSize);
    const size_t localColumns = baseColumns + (static_cast<size_t>(rank) < extraColumns ? 1 : 0);
    const size_t firstColumn = baseColumns * static_cast<size_t>(rank) +
        std::min(static_cast<size_t>(rank), extraColumns);
    std::vector<int> counts(worldSize);
    std::vector<int> displacements(worldSize);
    for (int process = 0; process < worldSize; ++process) {
        const size_t columns = baseColumns + (static_cast<size_t>(process) < extraColumns ? 1 : 0);
        counts[process] = static_cast<int>(columns * numNodes);
        displacements[process] = static_cast<int>((baseColumns * static_cast<size_t>(process) +
            std::min(static_cast<size_t>(process), extraColumns)) * numNodes);
    }

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("MPI processes: %d\n", worldSize);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing graph...\n");
    }

    std::vector<unsigned int> globalDist;
    std::vector<unsigned int> globalPath;
    if (rank == 0) {
        globalDist.resize(numNodes * numNodes);
        globalPath.resize(numNodes * numNodes);
        initializeDistanceMatrix(globalDist, numNodes, 1, MAX_DISTANCE);
        initializePathMatrix(globalPath, numNodes);
    }
    std::vector<unsigned int> dist(localColumns * numNodes);
    std::vector<unsigned int> path(localColumns * numNodes);
    MPI_Scatterv(rank == 0 ? globalDist.data() : nullptr, counts.data(), displacements.data(), MPI_UNSIGNED,
                 dist.data(), counts[rank], MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? globalPath.data() : nullptr, counts.data(), displacements.data(), MPI_UNSIGNED,
                 path.data(), counts[rank], MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    
    // Run Floyd-Warshall
    if (rank == 0) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    
    floydWarshall(dist, path, numNodes, firstColumn, localColumns, rank, worldSize);
    
    const double localDuration = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&localDuration, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    // Calculate operations per second
    // Floyd-Warshall has O(n³) complexity
    if (rank == 0) {
        printf("Computation time: %.0f ms\n", duration * 1000.0);
        const double ops = static_cast<double>(numNodes) * numNodes * numNodes;
        printf("Performance: %.3f GOPS\n", ops / duration / 1e9);
    }

    if (printResults || validate) {
        if (rank == 0) {
            globalDist.resize(numNodes * numNodes);
        }
        MPI_Gatherv(dist.data(), counts[rank], MPI_UNSIGNED,
                    rank == 0 ? globalDist.data() : nullptr, counts.data(), displacements.data(), MPI_UNSIGNED,
                    0, MPI_COMM_WORLD);
    }
    
    // Print results for external validation (integer hash-based)
    if (printResults) {
        if (rank == 0) print_results_int(globalDist, "DistanceMatrix");
    }
    
    // Validation
    if (validate) {
        int valid = 1;
        if (rank == 0) {
            printf("Validating result...\n");
            valid = validateResult(globalDist, numNodes) ? 1 : 0;
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Finalize();
        return valid ? 0 : 1;
    }
    MPI_Finalize();
    return 0;
}
