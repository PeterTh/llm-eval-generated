#include <algorithm>
#include <chrono>
#include <cmath>
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

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes, const size_t firstColumn,
                   const size_t localColumns, const std::vector<int>& columnOwners,
                   const int rank) {
    // Columns (destinations) are distributed.  The column for k is the only
    // non-local input required by an iteration, so broadcast it once per k.
    std::vector<unsigned int> pivot(numNodes);
    for (size_t k = 0; k < numNodes; ++k) {
        const int owner = columnOwners[k];
        if (rank == owner) {
            const size_t localK = k - firstColumn;
            std::memcpy(pivot.data(), dist.data() + localK * numNodes,
                        numNodes * sizeof(unsigned int));
        }
        MPI_Bcast(pivot.data(), static_cast<int>(numNodes), MPI_UNSIGNED, owner,
                  MPI_COMM_WORLD);

        for (size_t localJ = 0; localJ < localColumns; ++localJ) {
            unsigned int* const distColumn = dist.data() + localJ * numNodes;
            unsigned int* const pathColumn = path.data() + localJ * numNodes;
            const unsigned int distKJ = distColumn[k];
            for (size_t i = 0; i < numNodes; ++i) {
                const unsigned int newDist = pivot[i] + distKJ;
                if (newDist < distColumn[i]) {
                    distColumn[i] = newDist;
                    pathColumn[i] = static_cast<unsigned int>(k);
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

    if (numNodes == 0 || numNodes > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        numNodes > static_cast<size_t>(std::numeric_limits<int>::max()) / numNodes) {
        if (rank == 0) printf("Number of nodes is out of supported range\n");
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI processes: %d\n", worldSize);
        printf("Initializing graph...\n");
    }

    const size_t baseColumns = numNodes / static_cast<size_t>(worldSize);
    const size_t remainder = numNodes % static_cast<size_t>(worldSize);
    const size_t localColumns = baseColumns + (static_cast<size_t>(rank) < remainder ? 1 : 0);
    const size_t firstColumn = static_cast<size_t>(rank) * baseColumns +
                               std::min(static_cast<size_t>(rank), remainder);
    std::vector<int> counts(worldSize), displacements(worldSize), columnOwners(numNodes);
    for (int process = 0; process < worldSize; ++process) {
        const size_t columns = baseColumns + (static_cast<size_t>(process) < remainder ? 1 : 0);
        const size_t first = static_cast<size_t>(process) * baseColumns +
                             std::min(static_cast<size_t>(process), remainder);
        counts[process] = static_cast<int>(columns * numNodes);
        displacements[process] = static_cast<int>(first * numNodes);
        for (size_t j = first; j < first + columns; ++j) columnOwners[j] = process;
    }

    std::vector<unsigned int> dist(localColumns * numNodes);
    std::vector<unsigned int> path(localColumns * numNodes);
    std::vector<unsigned int> rootDist;
    std::vector<unsigned int> rootPath;
    if (rank == 0) {
        rootDist.resize(numNodes * numNodes);
        rootPath.resize(numNodes * numNodes);
        initializeDistanceMatrix(rootDist, numNodes, 1, MAX_DISTANCE);
        initializePathMatrix(rootPath, numNodes);
    }
    MPI_Scatterv(rank == 0 ? rootDist.data() : nullptr, counts.data(), displacements.data(), MPI_UNSIGNED,
                 dist.data(), counts[rank], MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? rootPath.data() : nullptr, counts.data(), displacements.data(), MPI_UNSIGNED,
                 path.data(), counts[rank], MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    // The complete matrices are only needed during initialization.  Release
    // them before the timed distributed computation.
    if (rank == 0) {
        std::vector<unsigned int>().swap(rootDist);
        std::vector<unsigned int>().swap(rootPath);
    }
    
    // Run Floyd-Warshall
    if (rank == 0) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    floydWarshall(dist, path, numNodes, firstColumn, localColumns, columnOwners, rank);
    
    auto end = std::chrono::high_resolution_clock::now();
    const double localSeconds = std::chrono::duration<double>(end - start).count();
    double seconds = 0.0;
    MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    // Calculate operations per second
    // Floyd-Warshall has O(n³) complexity
    double ops = (double)numNodes * numNodes * numNodes;
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", seconds * 1000.0);
        double gflops = seconds > 0.0 ? ops / seconds / 1e9 : 0.0;
        printf("Performance: %.3f GOPS\n", gflops);
    }
    
    // Print results for external validation (integer hash-based)
    if (printResults || validate) {
        if (rank == 0) rootDist.resize(numNodes * numNodes);
        MPI_Gatherv(dist.data(), counts[rank], MPI_UNSIGNED,
                    rank == 0 ? rootDist.data() : nullptr, counts.data(), displacements.data(), MPI_UNSIGNED,
                    0, MPI_COMM_WORLD);
    }
    if (printResults && rank == 0) print_results_int(rootDist, "DistanceMatrix");

    int exitCode = 0;
    if (validate && rank == 0) {
        printf("Validating result...\n");
        bool valid = validateResult(rootDist, numNodes);
        
        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            exitCode = 1;
        }
    }
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
