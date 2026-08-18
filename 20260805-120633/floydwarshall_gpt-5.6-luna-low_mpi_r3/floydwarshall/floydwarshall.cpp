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

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path, 
                   const size_t numNodes, const size_t firstRow,
                   const size_t localRows, const int rank, const int world) {
    std::vector<unsigned int> pivot(numNodes);
    for (size_t k = 0; k < numNodes; ++k) {
        // Rows are block distributed; locate the rank owning row k.
        size_t ownerRank = 0;
        const size_t base = numNodes / static_cast<size_t>(world);
        const size_t extra = numNodes % static_cast<size_t>(world);
        if (base == 0) ownerRank = k;
        else ownerRank = (k < (base + 1) * extra) ? k / (base + 1)
                                                   : extra + (k - (base + 1) * extra) / base;
        if (rank == static_cast<int>(ownerRank)) {
            const size_t localIndex = k - firstRow;
            std::copy_n(dist.data() + localIndex * numNodes, numNodes, pivot.data());
        }
        MPI_Bcast(pivot.data(), static_cast<int>(numNodes), MPI_UNSIGNED,
                  static_cast<int>(ownerRank), MPI_COMM_WORLD);
        for (size_t local = 0; local < localRows; ++local) {
            unsigned int* row = dist.data() + local * numNodes;
            const unsigned int distIK = row[k];
            for (size_t j = 0; j < numNodes; ++j) {
                const unsigned int newDist = distIK + pivot[j];
                if (newDist < row[j]) {
                    row[j] = newDist;
                    path[local * numNodes + j] = static_cast<unsigned int>(k);
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
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank = 0, world = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &world);
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
            MPI_Finalize(); return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            MPI_Finalize(); return 1;
        }
    }
    
    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\nNumber of nodes: %zu\nValidation: %s\n",
               numNodes, validate ? "enabled" : "disabled");
    }

    const size_t base = numNodes / static_cast<size_t>(world);
    const size_t extra = numNodes % static_cast<size_t>(world);
    const size_t firstRow = static_cast<size_t>(rank) * base + std::min(static_cast<size_t>(rank), extra);
    const size_t localRows = base + (static_cast<size_t>(rank) < extra ? 1 : 0);
    std::vector<int> counts(world), displs(world);
    for (int p = 0; p < world; ++p) {
        const size_t rows = base + (static_cast<size_t>(p) < extra ? 1 : 0);
        counts[p] = static_cast<int>(rows * numNodes);
        displs[p] = static_cast<int>((static_cast<size_t>(p) * base + std::min(static_cast<size_t>(p), extra)) * numNodes);
    }
    
    // Allocate matrices
    std::vector<unsigned int> globalDist(rank == 0 ? numNodes * numNodes : 0);
    std::vector<unsigned int> dist(localRows * numNodes);
    std::vector<unsigned int> path(localRows * numNodes);
    
    // Initialize
    if (rank == 0) initializeDistanceMatrix(globalDist, numNodes, 1, MAX_DISTANCE);
    MPI_Scatterv(rank == 0 ? globalDist.data() : nullptr, counts.data(), displs.data(), MPI_UNSIGNED,
                 dist.data(), counts[rank], MPI_UNSIGNED, 0, comm);
    for (size_t local = 0; local < localRows; ++local)
        for (size_t j = 0; j < numNodes; ++j) path[local * numNodes + j] = static_cast<unsigned int>(firstRow + local);
    
    // Run Floyd-Warshall
    if (rank == 0) printf("Computing shortest paths...\n");
    MPI_Barrier(comm);
    auto start = std::chrono::high_resolution_clock::now();
    
    floydWarshall(dist, path, numNodes, firstRow, localRows, rank, world);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    long long localMs = duration.count(), maxMs = 0;
    MPI_Reduce(&localMs, &maxMs, 1, MPI_LONG_LONG, MPI_MAX, 0, comm);
    if (rank == 0) printf("Computation time: %lld ms\n", maxMs);
    
    // Calculate operations per second
    // Floyd-Warshall has O(n³) complexity
    double ops = (double)numNodes * numNodes * numNodes;
    if (rank == 0) printf("Performance: %.3f GOPS\n", ops / (maxMs / 1000.0) / 1e9);

    std::vector<unsigned int> result(rank == 0 ? numNodes * numNodes : 0);
    MPI_Gatherv(dist.data(), counts[rank], MPI_UNSIGNED,
                rank == 0 ? result.data() : nullptr, counts.data(), displs.data(), MPI_UNSIGNED, 0, comm);
    
    // Print results for external validation (integer hash-based)
    if (printResults) {
        if (rank == 0) print_results_int(result, "DistanceMatrix");
    }
    
    // Validation
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        int valid = rank == 0 && validateResult(result, numNodes) ? 1 : 0;
        MPI_Bcast(&valid, 1, MPI_INT, 0, comm);
        
        if (valid) {
            if (rank == 0) printf("Validation: PASSED\n");
            MPI_Finalize(); return 0;
        } else {
            if (rank == 0) printf("Validation: FAILED\n");
            MPI_Finalize(); return 1;
        }
    }
    
    MPI_Finalize();
    return 0;
}
