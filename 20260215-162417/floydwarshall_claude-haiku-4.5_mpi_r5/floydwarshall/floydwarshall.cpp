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
                   const size_t numNodes) {
     int rank, size;
     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
     MPI_Comm_size(MPI_COMM_WORLD, &size);
     
     // Distribute rows across processes
     size_t rowsPerProcess = numNodes / size;
     size_t remainder = numNodes % size;
     size_t startRow = rank * rowsPerProcess + std::min((size_t)rank, remainder);
     size_t numLocalRows = rowsPerProcess + (rank < (int)remainder ? 1 : 0);
     
     // Allocate buffer for broadcasting k-th row
     std::vector<unsigned int> kRowDist(numNodes);
     std::vector<unsigned int> kRowPath(numNodes);
     
     // Floyd-Warshall with row distribution across MPI processes
     // For each intermediate node k
     for (size_t k = 0; k < numNodes; ++k) {
         // Determine which rank owns row k
         size_t kRank = 0;
         size_t tempStartRow = 0;
         for (int r = 0; r < size; ++r) {
             size_t tempRowsPerProc = numNodes / size;
             size_t tempRemainder = numNodes % size;
             size_t tempLocalRows = tempRowsPerProc + (r < (int)tempRemainder ? 1 : 0);
             if (k >= tempStartRow && k < tempStartRow + tempLocalRows) {
                 kRank = r;
                 break;
             }
             tempStartRow += tempLocalRows;
         }
         
         // The process that owns row k prepares it for broadcast
         if ((int)kRank == rank) {
             for (size_t j = 0; j < numNodes; ++j) {
                 kRowDist[j] = dist[idx2(j, k, numNodes)];
                 kRowPath[j] = path[idx2(j, k, numNodes)];
             }
         }
         
         // Broadcast the k-th row to all processes
         MPI_Bcast(kRowDist.data(), numNodes, MPI_UNSIGNED, kRank, MPI_COMM_WORLD);
         MPI_Bcast(kRowPath.data(), numNodes, MPI_UNSIGNED, kRank, MPI_COMM_WORLD);
         
         // Each process updates its assigned rows
         for (size_t i = startRow; i < startRow + numLocalRows; ++i) {
             // For each destination node j
             for (size_t j = 0; j < numNodes; ++j) {
                 const unsigned int distIJ = dist[idx2(j, i, numNodes)];
                 const unsigned int distIK = dist[idx2(k, i, numNodes)];
                 const unsigned int distKJ = kRowDist[j];
                 
                 const unsigned int newDist = distIK + distKJ;
                 
                 if (newDist < distIJ) {
                     dist[idx2(j, i, numNodes)] = newDist;
                     path[idx2(j, i, numNodes)] = kRowPath[j];
                 }
             }
         }
         
         MPI_Barrier(MPI_COMM_WORLD);
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
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
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
    
    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("MPI processes: %d\n", size);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Allocate matrices
    std::vector<unsigned int> dist(numNodes * numNodes);
    std::vector<unsigned int> path(numNodes * numNodes);
    
    // Initialize (only rank 0 initializes, then broadcasts)
    if (rank == 0) {
        printf("Initializing graph...\n");
        initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
        initializePathMatrix(path, numNodes);
    }
    
    // Broadcast initial matrices to all processes
    MPI_Bcast(dist.data(), numNodes * numNodes, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(path.data(), numNodes * numNodes, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    
    // Run Floyd-Warshall
    if (rank == 0) {
        printf("Computing shortest paths...\n");
    }
    auto start = std::chrono::high_resolution_clock::now();
    
    floydWarshall(dist, path, numNodes);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Since all processes have replicated the full matrix, synchronize results
    // Use MPI_Allreduce to ensure all processes have the minimum distances
    std::vector<unsigned int> reducedDist(numNodes * numNodes);
    std::vector<unsigned int> reducedPath(numNodes * numNodes);
    
    MPI_Allreduce(dist.data(), reducedDist.data(), numNodes * numNodes, 
                  MPI_UNSIGNED, MPI_MIN, MPI_COMM_WORLD);
    MPI_Bcast(path.data(), numNodes * numNodes, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        
        // Calculate operations per second
        // Floyd-Warshall has O(n³) complexity
        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);
        
        // Print results for external validation (integer hash-based)
        if (printResults) {
            print_results_int(reducedDist, "DistanceMatrix");
        }
        
        // Validation
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(reducedDist, numNodes);
            
            if (valid) {
                printf("Validation: PASSED\n");
                MPI_Finalize();
                return 0;
            } else {
                printf("Validation: FAILED\n");
                MPI_Finalize();
                return 1;
            }
        }
    }
    
    MPI_Finalize();
    return 0;
}
