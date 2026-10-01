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
                   const size_t numNodes, int rank, int processes) {
    const size_t base = numNodes / static_cast<size_t>(processes);
    const size_t extra = numNodes % static_cast<size_t>(processes);
    const size_t localBegin = static_cast<size_t>(rank) * base + std::min(static_cast<size_t>(rank), extra);
    const size_t localCount = base + (static_cast<size_t>(rank) < extra ? 1 : 0);
    std::vector<unsigned int> pivot(numNodes);
    // Classic Floyd-Warshall algorithm
    // For each intermediate node k
    for (size_t k = 0; k < numNodes; ++k) {
        int pivotOwner = 0;
        while (k >= static_cast<size_t>(pivotOwner) * base + std::min(static_cast<size_t>(pivotOwner), extra) +
                            base + (static_cast<size_t>(pivotOwner) < extra ? 1 : 0)) ++pivotOwner;
        if (rank == pivotOwner)
            for (size_t j = 0; j < numNodes; ++j) pivot[j] = dist[idx2(j, k, numNodes)];
        MPI_Bcast(pivot.data(), static_cast<int>(numNodes), MPI_UNSIGNED, pivotOwner, MPI_COMM_WORLD);
        // For each source node i
        for (size_t i = localBegin; i < localBegin + localCount; ++i) {
            // For each destination node j
            for (size_t j = 0; j < numNodes; ++j) {
                const unsigned int distIJ = dist[idx2(j, i, numNodes)];
                const unsigned int distIK = dist[idx2(k, i, numNodes)];
                const unsigned int distKJ = pivot[j];
                
                const unsigned int newDist = distIK + distKJ;
                
                if (newDist < distIJ) {
                    dist[idx2(j, i, numNodes)] = newDist;
                    path[idx2(j, i, numNodes)] = k;
                }
            }
        }
    }
    std::vector<int> counts(processes), displacements(processes);
    for (int p = 0; p < processes; ++p) {
        const size_t begin = static_cast<size_t>(p) * base + std::min(static_cast<size_t>(p), extra);
        const size_t count = base + (static_cast<size_t>(p) < extra ? 1 : 0);
        counts[p] = static_cast<int>(count * numNodes);
        displacements[p] = static_cast<int>(begin * numNodes);
    }
    std::vector<unsigned int> sendDist(localCount * numNodes), sendPath(localCount * numNodes);
    std::vector<unsigned int> allDist(numNodes * numNodes), allPath(numNodes * numNodes);
    for (size_t l = 0; l < localCount; ++l) for (size_t j = 0; j < numNodes; ++j) {
        sendDist[l * numNodes + j] = dist[idx2(j, localBegin + l, numNodes)];
        sendPath[l * numNodes + j] = path[idx2(j, localBegin + l, numNodes)];
    }
    MPI_Allgatherv(sendDist.data(), counts[rank], MPI_UNSIGNED, allDist.data(), counts.data(), displacements.data(), MPI_UNSIGNED, MPI_COMM_WORLD);
    MPI_Allgatherv(sendPath.data(), counts[rank], MPI_UNSIGNED, allPath.data(), counts.data(), displacements.data(), MPI_UNSIGNED, MPI_COMM_WORLD);
    for (size_t i = 0; i < numNodes; ++i) for (size_t j = 0; j < numNodes; ++j) {
        dist[idx2(j, i, numNodes)] = allDist[i * numNodes + j];
        path[idx2(j, i, numNodes)] = allPath[i * numNodes + j];
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
    int rank, processes;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processes);
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
            if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize();
            return 1;
        }
    }
    
    if (rank == 0) printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
    if (rank == 0) { printf("Number of nodes: %zu\n", numNodes); printf("Validation: %s\n", validate ? "enabled" : "disabled"); }
    
    // Allocate matrices
    std::vector<unsigned int> dist(numNodes * numNodes);
    std::vector<unsigned int> path(numNodes * numNodes);
    
    // Initialize
    if (rank == 0) printf("Initializing graph...\n");
    initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
    initializePathMatrix(path, numNodes);
    
    // Run Floyd-Warshall
    if (rank == 0) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    floydWarshall(dist, path, numNodes, rank, processes);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    long long elapsed = duration.count();
    MPI_Reduce(rank == 0 ? MPI_IN_PLACE : &elapsed, &elapsed, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Computation time: %lld ms\n", elapsed);
    
    // Calculate operations per second
    // Floyd-Warshall has O(n³) complexity
    double ops = (double)numNodes * numNodes * numNodes;
    double gflops = ops / (elapsed / 1000.0) / 1e9;
    if (rank == 0) printf("Performance: %.3f GOPS\n", gflops);
    
    // Print results for external validation (integer hash-based)
    if (rank == 0 && printResults) {
        print_results_int(dist, "DistanceMatrix");
    }
    
    // Validation
    if (rank == 0 && validate) {
        printf("Validating result...\n");
        bool valid = validateResult(dist, numNodes);
        
        if (valid) {
            printf("Validation: PASSED\n");
            MPI_Finalize(); return 0;
        } else {
            printf("Validation: FAILED\n");
            MPI_Finalize(); return 1;
        }
    }
    
    MPI_Finalize();
    return 0;
}
