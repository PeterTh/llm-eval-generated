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
                   int rank,
                   int num_procs,
                   size_t start_row,
                   size_t local_rows) {
    std::vector<unsigned int> k_row(numNodes);
    std::vector<unsigned int> k_path(numNodes);
    
    size_t rows_per_process = numNodes / num_procs;
    size_t extra_rows = numNodes % num_procs;
    
    // Pre-compute row ranges and sizes for all ranks
    std::vector<size_t> rank_start(num_procs);
    std::vector<size_t> rank_rows(num_procs);
    std::vector<int> sendcounts(num_procs);
    std::vector<int> displs(num_procs);
    
    int displs_val = 0;
    for (int r = 0; r < num_procs; ++r) {
        rank_rows[r] = rows_per_process + (r < extra_rows ? 1 : 0);
        rank_start[r] = r * rows_per_process + std::min((size_t)r, extra_rows);
        sendcounts[r] = rank_rows[r] * numNodes;
        displs[r] = displs_val;
        displs_val += sendcounts[r];
    }
    
    // Pre-allocate buffers for gather/scatter
    std::vector<unsigned int> send_dist(local_rows * numNodes);
    std::vector<unsigned int> send_path(local_rows * numNodes);
    std::vector<unsigned int> recv_dist(displs_val);
    std::vector<unsigned int> recv_path(displs_val);
    
    // For each intermediate node k
    for (size_t k = 0; k < numNodes; ++k) {
        // Determine which rank owns row k
        int k_rank = 0;
        size_t k_row_idx = k;
        
        for (int r = 0; r < num_procs; ++r) {
            if (k_row_idx < rank_rows[r]) {
                k_rank = r;
                break;
            }
            k_row_idx -= rank_rows[r];
        }
        
        // Broadcast the k-th row and path from the owning process
        if (rank == k_rank) {
            for (size_t j = 0; j < numNodes; ++j) {
                k_row[j] = dist[idx2(j, k, numNodes)];
                k_path[j] = path[idx2(j, k, numNodes)];
            }
        }
        MPI_Bcast(k_row.data(), numNodes, MPI_UNSIGNED, k_rank, MPI_COMM_WORLD);
        MPI_Bcast(k_path.data(), numNodes, MPI_UNSIGNED, k_rank, MPI_COMM_WORLD);
        
        // Each process updates its local rows
        for (size_t local_i = 0; local_i < local_rows; ++local_i) {
            size_t i = start_row + local_i;
            
            // For each destination node j
            for (size_t j = 0; j < numNodes; ++j) {
                const unsigned int distIJ = dist[idx2(j, i, numNodes)];
                const unsigned int distIK = dist[idx2(k, i, numNodes)];
                const unsigned int distKJ = k_row[j];
                
                const unsigned int newDist = distIK + distKJ;
                
                if (newDist < distIJ) {
                    dist[idx2(j, i, numNodes)] = newDist;
                    path[idx2(j, i, numNodes)] = k_path[k];
                }
            }
        }
        
        // Pack local rows for all-gather
        for (size_t local_i = 0; local_i < local_rows; ++local_i) {
            for (size_t j = 0; j < numNodes; ++j) {
                send_dist[local_i * numNodes + j] = dist[idx2(j, start_row + local_i, numNodes)];
                send_path[local_i * numNodes + j] = path[idx2(j, start_row + local_i, numNodes)];
            }
        }
        
        // All-gather to synchronize updated rows
        MPI_Allgatherv(send_dist.data(), sendcounts[rank], MPI_UNSIGNED, 
                       recv_dist.data(), sendcounts.data(), displs.data(), MPI_UNSIGNED, MPI_COMM_WORLD);
        MPI_Allgatherv(send_path.data(), sendcounts[rank], MPI_UNSIGNED, 
                       recv_path.data(), sendcounts.data(), displs.data(), MPI_UNSIGNED, MPI_COMM_WORLD);
        
        // Update full matrix with gathered rows
        for (int r = 0; r < num_procs; ++r) {
            for (size_t local_i = 0; local_i < rank_rows[r]; ++local_i) {
                for (size_t j = 0; j < numNodes; ++j) {
                    dist[idx2(j, rank_start[r] + local_i, numNodes)] = recv_dist[displs[r] + local_i * numNodes + j];
                    path[idx2(j, rank_start[r] + local_i, numNodes)] = recv_path[displs[r] + local_i * numNodes + j];
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
    
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    
    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments (only on rank 0)
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
    
    // Broadcast numNodes and flags to all processes
    MPI_Bcast(&numNodes, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    int val_int = validate ? 1 : 0;
    int print_int = printResults ? 1 : 0;
    MPI_Bcast(&val_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&print_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
    validate = (val_int != 0);
    printResults = (print_int != 0);
    
    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark (MPI)\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Number of MPI processes: %d\n", num_procs);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Calculate row distribution
    size_t rows_per_process = numNodes / num_procs;
    size_t extra_rows = numNodes % num_procs;
    size_t start_row = rank * rows_per_process + std::min((size_t)rank, extra_rows);
    size_t local_rows = rows_per_process + (rank < extra_rows ? 1 : 0);
    
    // Allocate full matrices (each process needs full matrix for reading)
    std::vector<unsigned int> dist(numNodes * numNodes);
    std::vector<unsigned int> path(numNodes * numNodes);
    
    // Initialize on rank 0 and broadcast
    if (rank == 0) {
        if (rank == 0) {
            printf("Initializing graph...\n");
        }
        initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
        initializePathMatrix(path, numNodes);
    }
    
    // Broadcast initialized matrices to all processes
    MPI_Bcast(dist.data(), numNodes * numNodes, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(path.data(), numNodes * numNodes, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Computing shortest paths...\n");
    }
    
    // Synchronize before timing
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Run Floyd-Warshall
    auto start = std::chrono::high_resolution_clock::now();
    
    floydWarshall(dist, path, numNodes, rank, num_procs, start_row, local_rows);
    
    // Synchronize after computation
    MPI_Barrier(MPI_COMM_WORLD);
    
    auto end = std::chrono::high_resolution_clock::now();
    
    // Gather results back to rank 0
    if (rank == 0) {
        std::vector<unsigned int> all_dist(numNodes * numNodes);
        std::vector<unsigned int> all_path(numNodes * numNodes);
        
        // Copy rank 0's part
        for (size_t i = 0; i < local_rows; ++i) {
            for (size_t j = 0; j < numNodes; ++j) {
                all_dist[idx2(j, start_row + i, numNodes)] = dist[idx2(j, start_row + i, numNodes)];
                all_path[idx2(j, start_row + i, numNodes)] = path[idx2(j, start_row + i, numNodes)];
            }
        }
        
        // Receive from other ranks
        for (int r = 1; r < num_procs; ++r) {
            size_t r_start = r * rows_per_process + std::min((size_t)r, extra_rows);
            size_t r_local = rows_per_process + (r < extra_rows ? 1 : 0);
            std::vector<unsigned int> recv_dist(r_local * numNodes);
            std::vector<unsigned int> recv_path(r_local * numNodes);
            
            MPI_Recv(recv_dist.data(), r_local * numNodes, MPI_UNSIGNED, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(recv_path.data(), r_local * numNodes, MPI_UNSIGNED, r, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            for (size_t i = 0; i < r_local; ++i) {
                for (size_t j = 0; j < numNodes; ++j) {
                    all_dist[idx2(j, r_start + i, numNodes)] = recv_dist[i * numNodes + j];
                    all_path[idx2(j, r_start + i, numNodes)] = recv_path[i * numNodes + j];
                }
            }
        }
        
        dist = all_dist;
        path = all_path;
    } else {
        // Send local rows to rank 0
        std::vector<unsigned int> send_dist(local_rows * numNodes);
        std::vector<unsigned int> send_path(local_rows * numNodes);
        
        for (size_t i = 0; i < local_rows; ++i) {
            for (size_t j = 0; j < numNodes; ++j) {
                send_dist[i * numNodes + j] = dist[idx2(j, start_row + i, numNodes)];
                send_path[i * numNodes + j] = path[idx2(j, start_row + i, numNodes)];
            }
        }
        
        MPI_Send(send_dist.data(), local_rows * numNodes, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD);
        MPI_Send(send_path.data(), local_rows * numNodes, MPI_UNSIGNED, 0, 1, MPI_COMM_WORLD);
    }
    
    if (rank == 0) {
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
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
    
    if (rank == 0) {
        return validate ? (validateResult(dist, numNodes) ? 0 : 1) : 0;
    }
    
    return 0;
}
