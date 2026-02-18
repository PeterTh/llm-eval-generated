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
     
     // Distribute rows: each process gets a contiguous block of rows
     size_t rows_per_process = (numNodes + size - 1) / size;
     size_t start_row = rank * rows_per_process;
     size_t end_row = std::min(start_row + rows_per_process, numNodes);
     size_t local_rows = end_row - start_row;
     
     // Count elements each process will send for gatherings
     std::vector<int> sendcounts(size, 0);
     std::vector<int> displs(size, 0);
     for (int p = 0; p < size; ++p) {
         size_t p_start = p * rows_per_process;
         size_t p_end = std::min(p_start + rows_per_process, numNodes);
         size_t p_rows = p_end - p_start;
         sendcounts[p] = p_rows;
         displs[p] = p_start;
     }
     
     // Allocate space for the k-th row and k-th column (needed by all processes)
     std::vector<unsigned int> row_k(numNodes);
     std::vector<unsigned int> col_k(numNodes);
     
     // Floyd-Warshall with row/column distribution
     for (size_t k = 0; k < numNodes; ++k) {
         // Determine which process holds row k
         int owner_rank = k / rows_per_process;
         if (owner_rank >= size) owner_rank = size - 1;
         
         // Process owner broadcasts row k
         if (rank == owner_rank) {
             // Copy row k to broadcast buffer
             for (size_t j = 0; j < numNodes; ++j) {
                 row_k[j] = dist[idx2(j, k, numNodes)];
             }
         }
         
         MPI_Bcast(row_k.data(), numNodes, MPI_UNSIGNED, owner_rank, MPI_COMM_WORLD);
         
         // Each process contributes its k-th column elements for col_k
         std::vector<unsigned int> local_col_k(local_rows);
         for (size_t i = 0; i < local_rows; ++i) {
             local_col_k[i] = dist[idx2(k, start_row + i, numNodes)];
         }
         
         // Gather all k-th column elements to all processes
         MPI_Allgatherv(local_col_k.data(), local_rows, MPI_UNSIGNED,
                        col_k.data(), sendcounts.data(), displs.data(), MPI_UNSIGNED,
                        MPI_COMM_WORLD);
         
         // Each process updates its local rows
         for (size_t i = start_row; i < end_row; ++i) {
             const unsigned int distIK = col_k[i];
             
             for (size_t j = 0; j < numNodes; ++j) {
                 const unsigned int distIJ = dist[idx2(j, i, numNodes)];
                 const unsigned int distKJ = row_k[j];
                 
                 const unsigned int newDist = distIK + distKJ;
                 
                 if (newDist < distIJ) {
                     dist[idx2(j, i, numNodes)] = newDist;
                     path[idx2(j, i, numNodes)] = k;
                 }
             }
         }
     }
     
     // Update sendcounts for row-based gather (for final result)
     for (int p = 0; p < size; ++p) {
         size_t p_start = p * rows_per_process;
         size_t p_end = std::min(p_start + rows_per_process, numNodes);
         size_t p_rows = p_end - p_start;
         sendcounts[p] = p_rows * numNodes;
         displs[p] = p_start * numNodes;
     }
     
     // Gather all rows back to a complete matrix on all processes
     std::vector<unsigned int> global_dist(numNodes * numNodes);
     std::vector<unsigned int> global_path(numNodes * numNodes);
     
     // Gather distance matrix
     MPI_Allgatherv(dist.data() + start_row * numNodes, local_rows * numNodes, MPI_UNSIGNED,
                    global_dist.data(), sendcounts.data(), displs.data(), MPI_UNSIGNED,
                    MPI_COMM_WORLD);
     
     // Gather path matrix
     MPI_Allgatherv(path.data() + start_row * numNodes, local_rows * numNodes, MPI_UNSIGNED,
                    global_path.data(), sendcounts.data(), displs.data(), MPI_UNSIGNED,
                    MPI_COMM_WORLD);
     
     // Update local copies with gathered data
     dist = global_dist;
     path = global_path;
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
    
    // Parse command line arguments (only on rank 0 for efficiency)
    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                numNodes = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = true;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = true;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                MPI_Finalize();
                return 0;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                MPI_Finalize();
                return 1;
            }
        }
    }
    
    // Broadcast parameters to all processes
    MPI_Bcast(&numNodes, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Number of MPI processes: %d\n", size);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Allocate matrices
    std::vector<unsigned int> dist(numNodes * numNodes);
    std::vector<unsigned int> path(numNodes * numNodes);
    
    // Initialize (all processes initialize their own copy for correctness)
    if (rank == 0) {
        printf("Initializing graph...\n");
    }
    initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
    initializePathMatrix(path, numNodes);
    
    // Broadcast initialized matrices to all processes
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
    
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        
        // Calculate operations per second
        // Floyd-Warshall has O(n³) complexity
        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);
        
        // Print results for external validation (integer hash-based)
        if (printResults) {
            print_results_int(dist, "DistanceMatrix");
        }
        
        // Validation
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(dist, numNodes);
            
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
    }
    
    MPI_Finalize();
    return 0;
}

