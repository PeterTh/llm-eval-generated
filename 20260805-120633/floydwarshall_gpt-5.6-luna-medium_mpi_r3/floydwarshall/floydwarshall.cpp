#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

void initializePathMatrix(std::vector<unsigned int>& path, const size_t firstRow,
                          const size_t localRows, const size_t numNodes) {
    for (size_t localRow = 0; localRow < localRows; ++localRow) {
        const unsigned int source = static_cast<unsigned int>(firstRow + localRow);
        for (size_t destination = 0; destination < numNodes; ++destination) {
            path[localRow * numNodes + destination] = source;
        }
    }
}

void floydWarshall(std::vector<unsigned int>& dist, std::vector<unsigned int>& path,
                   const size_t firstRow, const size_t localRows, const size_t numNodes,
                   const int rank, const std::vector<int>& pivotOwners) {
    std::vector<unsigned int> pivotRow(numNodes);

    for (size_t k = 0; k < numNodes; ++k) {
        const int pivotOwner = pivotOwners[k];
        if (rank == pivotOwner) {
            const size_t localPivot = k - firstRow;
            std::memcpy(pivotRow.data(), &dist[localPivot * numNodes],
                        numNodes * sizeof(unsigned int));
        }
        MPI_Bcast(pivotRow.data(), static_cast<int>(numNodes), MPI_UNSIGNED,
                  pivotOwner, MPI_COMM_WORLD);

        for (size_t localRow = 0; localRow < localRows; ++localRow) {
            unsigned int* distanceRow = &dist[localRow * numNodes];
            unsigned int* pathRow = &path[localRow * numNodes];
            const unsigned int distanceToPivot = distanceRow[k];

            for (size_t destination = 0; destination < numNodes; ++destination) {
                const unsigned int newDistance = distanceToPivot + pivotRow[destination];
                if (newDistance < distanceRow[destination]) {
                    distanceRow[destination] = newDistance;
                    pathRow[destination] = static_cast<unsigned int>(k);
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

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI processes: %d\n", worldSize);
        printf("Initializing graph...\n");
    }

    // Contiguous source rows are distributed as evenly as possible.  The root
    // creates the same stream of random values as the original implementation.
    const size_t baseRows = numNodes / static_cast<size_t>(worldSize);
    const size_t extraRows = numNodes % static_cast<size_t>(worldSize);
    const size_t localRows = baseRows + (static_cast<size_t>(rank) < extraRows ? 1 : 0);
    const size_t firstRow = static_cast<size_t>(rank) * baseRows +
                            std::min(static_cast<size_t>(rank), extraRows);

    std::vector<int> rowCounts(worldSize);
    std::vector<int> rowDisplacements(worldSize);
    for (int r = 0; r < worldSize; ++r) {
        const size_t rows = baseRows + (static_cast<size_t>(r) < extraRows ? 1 : 0);
        const size_t offset = static_cast<size_t>(r) * baseRows +
                              std::min(static_cast<size_t>(r), extraRows);
        rowCounts[r] = static_cast<int>(rows * numNodes);
        rowDisplacements[r] = static_cast<int>(offset * numNodes);
    }

    std::vector<unsigned int> rootDist;
    if (rank == 0) {
        rootDist.resize(numNodes * numNodes);
        initializeDistanceMatrix(rootDist, numNodes, 1, MAX_DISTANCE);
    }
    std::vector<unsigned int> dist(localRows * numNodes);
    std::vector<unsigned int> path(localRows * numNodes);
    MPI_Scatterv(rank == 0 ? rootDist.data() : nullptr, rowCounts.data(),
                 rowDisplacements.data(), MPI_UNSIGNED, dist.data(),
                 static_cast<int>(localRows * numNodes), MPI_UNSIGNED, 0,
                 MPI_COMM_WORLD);
    initializePathMatrix(path, firstRow, localRows, numNodes);

    // Compute the owner of each pivot from the contiguous row partition.
    // This also handles more MPI processes than vertices.
    std::vector<int> pivotOwners(numNodes);
    for (size_t k = 0; k < numNodes; ++k) {
        pivotOwners[k] = 0;
        while (pivotOwners[k] + 1 < worldSize &&
               static_cast<size_t>(rowDisplacements[pivotOwners[k] + 1]) <= k * numNodes) {
            ++pivotOwners[k];
        }
    }

    if (rank == 0) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    floydWarshall(dist, path, firstRow, localRows, numNodes, rank, pivotOwners);

    double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    const long durationMs = static_cast<long>(maxElapsed * 1000.0);

    if (rank == 0) printf("Computation time: %ld ms\n", durationMs);
    
    // Calculate operations per second
    // Floyd-Warshall has O(n³) complexity
    double ops = (double)numNodes * numNodes * numNodes;
    double gflops = ops / std::max(maxElapsed, 1.0e-9) / 1e9;
    if (rank == 0) printf("Performance: %.3f GOPS\n", gflops);

    if (validate || printResults) {
        if (rank == 0) rootDist.resize(numNodes * numNodes);
        MPI_Gatherv(dist.data(), static_cast<int>(localRows * numNodes), MPI_UNSIGNED,
                    rank == 0 ? rootDist.data() : nullptr, rowCounts.data(),
                    rowDisplacements.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    }
    
    // Print results for external validation (integer hash-based)
    if (printResults && rank == 0) {
        print_results_int(rootDist, "DistanceMatrix");
    }
    
    // Validation
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        bool valid = rank == 0 && validateResult(rootDist, numNodes);
        int validInt = valid ? 1 : 0;
        MPI_Bcast(&validInt, 1, MPI_INT, 0, MPI_COMM_WORLD);
        valid = validInt != 0;

        if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        MPI_Finalize();
        return valid ? 0 : 1;
    }
    MPI_Finalize();
    return 0;
}
