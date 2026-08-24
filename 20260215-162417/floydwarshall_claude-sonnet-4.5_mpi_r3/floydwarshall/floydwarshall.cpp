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
                   const size_t numNodes,
                   const size_t localRows,
                   const size_t startRow,
                   int rank,
                   int numRanks) {
    // Buffer for broadcasting the k-th row
    std::vector<unsigned int> rowK(numNodes);
    
    // Classic Floyd-Warshall algorithm with MPI parallelization
    // For each intermediate node k
    for (size_t k = 0; k < numNodes; ++k) {
        // Determine which rank owns row k
        int ownerRank = k / (numNodes / numRanks + (numNodes % numRanks != 0 ? 1 : 0));
        if (numRanks > 1) {
            // More precise calculation
            ownerRank = 0;
            size_t rowCount = 0;
            for (int r = 0; r < numRanks; ++r) {
                size_t rowsPerRank = numNodes / numRanks + (r < (int)(numNodes % numRanks) ? 1 : 0);
                if (k < rowCount + rowsPerRank) {
                    ownerRank = r;
                    break;
                }
                rowCount += rowsPerRank;
            }
        }
        
        // Broadcast row k to all ranks
        if (rank == ownerRank) {
            // Copy row k from local storage
            size_t localK = k - startRow;
            for (size_t j = 0; j < numNodes; ++j) {
                rowK[j] = dist[idx2(j, localK, numNodes)];
            }
        }
        MPI_Bcast(rowK.data(), numNodes, MPI_UNSIGNED, ownerRank, MPI_COMM_WORLD);
        
        // For each local source node i
        for (size_t localI = 0; localI < localRows; ++localI) {
            const unsigned int distIK = dist[idx2(k, localI, numNodes)];
            
            // For each destination node j
            for (size_t j = 0; j < numNodes; ++j) {
                const unsigned int distIJ = dist[idx2(j, localI, numNodes)];
                const unsigned int distKJ = rowK[j];
                
                const unsigned int newDist = distIK + distKJ;
                
                if (newDist < distIJ) {
                    dist[idx2(j, localI, numNodes)] = newDist;
                    path[idx2(j, localI, numNodes)] = k;
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
    // Initialize MPI
    MPI_Init(&argc, &argv);
    
    int rank, numRanks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
    
    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments (all ranks parse to ensure consistency)
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
    
    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("MPI ranks: %d\n", numRanks);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Calculate row distribution
    size_t rowsPerRank = numNodes / numRanks;
    size_t remainder = numNodes % numRanks;
    
    // Ranks 0 to remainder-1 get one extra row
    size_t localRows = rowsPerRank + (rank < (int)remainder ? 1 : 0);
    size_t startRow = rank * rowsPerRank + std::min((size_t)rank, remainder);
    
    // Full matrices (rank 0 only for initialization and final result)
    std::vector<unsigned int> fullDist;
    std::vector<unsigned int> fullPath;
    
    if (rank == 0) {
        fullDist.resize(numNodes * numNodes);
        fullPath.resize(numNodes * numNodes);
        
        // Initialize
        printf("Initializing graph...\n");
        initializeDistanceMatrix(fullDist, numNodes, 1, MAX_DISTANCE);
        initializePathMatrix(fullPath, numNodes);
    }
    
    // Local matrices for computation
    std::vector<unsigned int> localDist(localRows * numNodes);
    std::vector<unsigned int> localPath(localRows * numNodes);
    
    // Scatter distance and path matrices to all ranks
    std::vector<int> sendCounts(numRanks);
    std::vector<int> displs(numRanks);
    
    for (int r = 0; r < numRanks; ++r) {
        size_t rRows = numNodes / numRanks + (r < (int)remainder ? 1 : 0);
        sendCounts[r] = rRows * numNodes;
        size_t rStart = r * rowsPerRank + std::min((size_t)r, remainder);
        displs[r] = rStart * numNodes;
    }
    
    MPI_Scatterv(rank == 0 ? fullDist.data() : nullptr, sendCounts.data(), displs.data(), 
                 MPI_UNSIGNED, localDist.data(), localRows * numNodes, MPI_UNSIGNED, 
                 0, MPI_COMM_WORLD);
    
    MPI_Scatterv(rank == 0 ? fullPath.data() : nullptr, sendCounts.data(), displs.data(), 
                 MPI_UNSIGNED, localPath.data(), localRows * numNodes, MPI_UNSIGNED, 
                 0, MPI_COMM_WORLD);
    
    // Synchronize before starting computation
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Run Floyd-Warshall
    if (rank == 0) {
        printf("Computing shortest paths...\n");
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    floydWarshall(localDist, localPath, numNodes, localRows, startRow, rank, numRanks);
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    long long localDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long long globalDuration = 0;
    MPI_Reduce(&localDuration, &globalDuration, 1, MPI_LONG_LONG_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Computation time: %lld ms\n", globalDuration);
        
        // Calculate operations per second
        // Floyd-Warshall has O(n³) complexity
        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / (globalDuration / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);
    }
    
    // Gather results back to rank 0
    MPI_Gatherv(localDist.data(), localRows * numNodes, MPI_UNSIGNED,
                rank == 0 ? fullDist.data() : nullptr, sendCounts.data(), displs.data(), 
                MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    
    MPI_Gatherv(localPath.data(), localRows * numNodes, MPI_UNSIGNED,
                rank == 0 ? fullPath.data() : nullptr, sendCounts.data(), displs.data(), 
                MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    
    // Print results for external validation (integer hash-based) - rank 0 only
    if (rank == 0 && printResults) {
        print_results_int(fullDist, "DistanceMatrix");
    }
    
    // Validation - rank 0 only
    int result = 0;
    if (rank == 0 && validate) {
        printf("Validating result...\n");
        bool valid = validateResult(fullDist, numNodes);
        
        if (valid) {
            printf("Validation: PASSED\n");
            result = 0;
        } else {
            printf("Validation: FAILED\n");
            result = 1;
        }
    }
    
    MPI_Finalize();
    return result;
}
