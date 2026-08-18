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

void localBounds(const size_t n, const int rank, const int ranks,
                 size_t& first, size_t& count) {
    first = (n * static_cast<size_t>(rank)) / static_cast<size_t>(ranks);
    const size_t last = (n * static_cast<size_t>(rank + 1)) /
                        static_cast<size_t>(ranks);
    count = last - first;
}

void initializePathMatrix(std::vector<unsigned int>& path, const size_t numNodes,
                          const size_t firstColumn, const size_t localColumns) {
    for (size_t local = 0; local < localColumns; ++local) {
        const size_t column = firstColumn + local;
        for (size_t row = 0; row < numNodes; ++row) {
            path[local * numNodes + row] = static_cast<unsigned int>(column);
        }
    }
}

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes,
                   const size_t localColumns, const int rank, const int ranks) {
    std::vector<unsigned int> pivot(numNodes);
    for (size_t k = 0; k < numNodes; ++k) {
        int owner = static_cast<int>((k * static_cast<size_t>(ranks)) / numNodes);
        size_t ownerFirst, ownerCount;
        localBounds(numNodes, owner, ranks, ownerFirst, ownerCount);
        if (rank == owner) {
            const size_t offset = (k - ownerFirst) * numNodes;
            std::copy_n(dist.begin() + offset, numNodes, pivot.begin());
        }
        MPI_Bcast(pivot.data(), static_cast<int>(numNodes), MPI_UNSIGNED,
                  owner, MPI_COMM_WORLD);

        for (size_t i = 0; i < localColumns; ++i) {
            unsigned int* column = dist.data() + i * numNodes;
            unsigned int* columnPath = path.data() + i * numNodes;
            const unsigned int distIK = column[k];
            for (size_t j = 0; j < numNodes; ++j) {
                const unsigned int candidate = distIK + pivot[j];
                if (candidate < column[j]) {
                    column[j] = candidate;
                    columnPath[j] = static_cast<unsigned int>(k);
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
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

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
            printUsage(argv[0]);
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }
    
    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("MPI ranks: %d\n", ranks);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    size_t firstColumn, localColumns;
    localBounds(numNodes, rank, ranks, firstColumn, localColumns);
    std::vector<int> counts(ranks), displacements(ranks);
    for (int r = 0; r < ranks; ++r) {
        size_t first, count;
        localBounds(numNodes, r, ranks, first, count);
        counts[r] = static_cast<int>(count * numNodes);
        displacements[r] = static_cast<int>(first * numNodes);
    }
    
    // Allocate matrices
    std::vector<unsigned int> dist(localColumns * numNodes);
    std::vector<unsigned int> path(localColumns * numNodes);
    
    // Initialize
    std::vector<unsigned int> globalDist;
    if (rank == 0) {
        printf("Initializing graph...\n");
        globalDist.resize(numNodes * numNodes);
        initializeDistanceMatrix(globalDist, numNodes, 1, MAX_DISTANCE);
    }
    MPI_Scatterv(rank == 0 ? globalDist.data() : nullptr, counts.data(),
                 displacements.data(), MPI_UNSIGNED, dist.data(), counts[rank],
                 MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    initializePathMatrix(path, numNodes, firstColumn, localColumns);
    
    // Run Floyd-Warshall
    if (rank == 0) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    
    floydWarshall(dist, path, numNodes, localColumns, rank, ranks);
    
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Computation time: %ld ms\n", static_cast<long>(maxElapsed * 1000.0));
    
    // Calculate operations per second
    // Floyd-Warshall has O(n³) complexity
    double ops = (double)numNodes * numNodes * numNodes;
    if (rank == 0) printf("Performance: %.3f GOPS\n", ops / maxElapsed / 1e9);

    if (rank == 0) globalDist.resize(numNodes * numNodes);
    MPI_Gatherv(dist.data(), counts[rank], MPI_UNSIGNED,
                rank == 0 ? globalDist.data() : nullptr, counts.data(),
                displacements.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    
    // Print results for external validation (integer hash-based)
    if (rank == 0 && printResults) {
        print_results_int(globalDist, "DistanceMatrix");
    }
    
    // Validation
    int valid = 1;
    if (rank == 0 && validate) {
        printf("Validating result...\n");
        valid = validateResult(globalDist, numNodes) ? 1 : 0;
        
        if (valid) {
            printf("Validation: PASSED\n");
            valid = 1;
        } else {
            printf("Validation: FAILED\n");
            valid = 0;
        }
    }
    MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return valid ? 0 : 1;
}
