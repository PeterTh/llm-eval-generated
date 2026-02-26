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
                   const int rank, const int size) {
    // MPI-parallel Floyd-Warshall algorithm
    // Each process owns a contiguous block of rows
    
    const size_t rows_per_proc = numNodes / size;
    const size_t remainder = numNodes % size;
    
    // Calculate row range for this process
    size_t start_row, end_row;
    if (rank < (int)remainder) {
        start_row = rank * (rows_per_proc + 1);
        end_row = start_row + rows_per_proc + 1;
    } else {
        start_row = remainder * (rows_per_proc + 1) + (rank - remainder) * rows_per_proc;
        end_row = start_row + rows_per_proc;
    }
    
    const size_t local_rows = end_row - start_row;
    
    // Buffer for broadcasting k-th row
    std::vector<unsigned int> k_row_dist(numNodes);
    std::vector<unsigned int> k_row_path(numNodes);
    
    // For each intermediate node k
    for (size_t k = 0; k < numNodes; ++k) {
        // Determine which process owns row k
        int owner_rank;
        if (k < remainder * (rows_per_proc + 1)) {
            owner_rank = k / (rows_per_proc + 1);
        } else {
            size_t offset = remainder * (rows_per_proc + 1);
            owner_rank = remainder + (k - offset) / rows_per_proc;
        }
        
        // Owner copies k-th row to broadcast buffer
        if (rank == owner_rank) {
            for (size_t j = 0; j < numNodes; ++j) {
                k_row_dist[j] = dist[idx2(j, k, numNodes)];
                k_row_path[j] = path[idx2(j, k, numNodes)];
            }
        }
        
        // Broadcast k-th row to all processes
        MPI_Bcast(k_row_dist.data(), numNodes, MPI_UNSIGNED, owner_rank, MPI_COMM_WORLD);
        MPI_Bcast(k_row_path.data(), numNodes, MPI_UNSIGNED, owner_rank, MPI_COMM_WORLD);
        
        // Update local rows
        for (size_t i = start_row; i < end_row; ++i) {
            const unsigned int distIK = dist[idx2(k, i, numNodes)];
            
            // Early skip if path through k is not beneficial
            if (distIK >= INF) continue;
            
            for (size_t j = 0; j < numNodes; ++j) {
                const unsigned int distIJ = dist[idx2(j, i, numNodes)];
                const unsigned int distKJ = k_row_dist[j];
                
                const unsigned int newDist = distIK + distKJ;
                
                if (newDist < distIJ) {
                    dist[idx2(j, i, numNodes)] = newDist;
                    path[idx2(j, i, numNodes)] = k;
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
    
    // Initialize (all processes initialize the same way for now)
    if (rank == 0) {
        printf("Initializing graph...\n");
    }
    initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
    initializePathMatrix(path, numNodes);
    
    // Ensure all processes are synchronized before starting computation
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Run Floyd-Warshall
    if (rank == 0) {
        printf("Computing shortest paths...\n");
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    floydWarshall(dist, path, numNodes, rank, size);
    
    // Synchronize before timing
    MPI_Barrier(MPI_COMM_WORLD);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        
        // Calculate operations per second
        // Floyd-Warshall has O(n³) complexity
        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);
    }
    
    // Gather all results to rank 0 for validation and output
    if (validate || printResults) {
        // Calculate row distribution
        const size_t rows_per_proc = numNodes / size;
        const size_t remainder = numNodes % size;
        
        std::vector<int> recvcounts(size);
        std::vector<int> displs(size);
        
        for (int r = 0; r < size; ++r) {
            size_t start_row, end_row;
            if (r < (int)remainder) {
                start_row = r * (rows_per_proc + 1);
                end_row = start_row + rows_per_proc + 1;
            } else {
                start_row = remainder * (rows_per_proc + 1) + (r - remainder) * rows_per_proc;
                end_row = start_row + rows_per_proc;
            }
            recvcounts[r] = (end_row - start_row) * numNodes;
            displs[r] = start_row * numNodes;
        }
        
        // Gather distance matrix
        std::vector<unsigned int> dist_gathered;
        if (rank == 0) {
            dist_gathered.resize(numNodes * numNodes);
        }
        
        size_t start_row, end_row;
        if (rank < (int)remainder) {
            start_row = rank * (rows_per_proc + 1);
            end_row = start_row + rows_per_proc + 1;
        } else {
            start_row = remainder * (rows_per_proc + 1) + (rank - remainder) * rows_per_proc;
            end_row = start_row + rows_per_proc;
        }
        
        MPI_Gatherv(dist.data() + start_row * numNodes, 
                    recvcounts[rank], MPI_UNSIGNED,
                    rank == 0 ? dist_gathered.data() : nullptr, 
                    recvcounts.data(), displs.data(), MPI_UNSIGNED,
                    0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            // Print results for external validation
            if (printResults) {
                print_results_int(dist_gathered, "DistanceMatrix");
            }
            
            // Validation
            if (validate) {
                printf("Validating result...\n");
                bool valid = validateResult(dist_gathered, numNodes);
                
                if (valid) {
                    printf("Validation: PASSED\n");
                } else {
                    printf("Validation: FAILED\n");
                    MPI_Finalize();
                    return 1;
                }
            }
        }
    }
    
    MPI_Finalize();
    return 0;
}
