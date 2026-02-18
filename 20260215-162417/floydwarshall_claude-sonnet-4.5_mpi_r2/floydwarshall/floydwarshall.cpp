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
                   int rank, int nprocs) {
    // Determine row distribution for this rank
    const size_t rows_per_proc = numNodes / nprocs;
    const size_t remainder = numNodes % nprocs;
    
    // Each process handles a contiguous block of rows
    size_t row_start = rank * rows_per_proc + std::min(static_cast<size_t>(rank), remainder);
    size_t row_count = rows_per_proc + (rank < static_cast<int>(remainder) ? 1 : 0);
    size_t row_end = row_start + row_count;
    
    // Buffer for broadcasting k-th row
    std::vector<unsigned int> k_row_dist(numNodes);
    std::vector<unsigned int> k_row_path(numNodes);
    
    // Classic Floyd-Warshall algorithm with MPI parallelization
    // For each intermediate node k
    for (size_t k = 0; k < numNodes; ++k) {
        // Determine which rank owns row k
        int k_owner = 0;
        size_t acc = 0;
        for (int r = 0; r < nprocs; ++r) {
            size_t r_count = rows_per_proc + (r < static_cast<int>(remainder) ? 1 : 0);
            if (k < acc + r_count) {
                k_owner = r;
                break;
            }
            acc += r_count;
        }
        
        // The owner of row k broadcasts it to all processes
        if (rank == k_owner) {
            for (size_t j = 0; j < numNodes; ++j) {
                k_row_dist[j] = dist[idx2(j, k, numNodes)];
                k_row_path[j] = path[idx2(j, k, numNodes)];
            }
        }
        
        MPI_Bcast(k_row_dist.data(), numNodes, MPI_UNSIGNED, k_owner, MPI_COMM_WORLD);
        MPI_Bcast(k_row_path.data(), numNodes, MPI_UNSIGNED, k_owner, MPI_COMM_WORLD);
        
        // Each process updates its assigned rows
        for (size_t i = row_start; i < row_end; ++i) {
            const unsigned int distIK = dist[idx2(k, i, numNodes)];
            
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
    
    // Gather all results back to rank 0
    if (nprocs > 1) {
        std::vector<int> recvcounts(nprocs);
        std::vector<int> displs(nprocs);
        
        for (int r = 0; r < nprocs; ++r) {
            size_t r_count = rows_per_proc + (r < static_cast<int>(remainder) ? 1 : 0);
            size_t r_start = r * rows_per_proc + std::min(static_cast<size_t>(r), remainder);
            recvcounts[r] = r_count * numNodes;
            displs[r] = r_start * numNodes;
        }
        
        // Gather distance matrix
        MPI_Gatherv(dist.data() + row_start * numNodes, row_count * numNodes, MPI_UNSIGNED,
                    dist.data(), recvcounts.data(), displs.data(), MPI_UNSIGNED,
                    0, MPI_COMM_WORLD);
        
        // Gather path matrix
        MPI_Gatherv(path.data() + row_start * numNodes, row_count * numNodes, MPI_UNSIGNED,
                    path.data(), recvcounts.data(), displs.data(), MPI_UNSIGNED,
                    0, MPI_COMM_WORLD);
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
    
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    
    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments (only rank 0)
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
            } else if (strcmp(argv[i], "-n") != 0 && strcmp(argv[i], "-v") != 0 && 
                       strcmp(argv[i], "-r") != 0 && strcmp(argv[i], "-h") != 0) {
                // Check if it's not an argument value
                bool isValue = false;
                if (i > 0 && strcmp(argv[i-1], "-n") == 0) {
                    isValue = true;
                }
                if (!isValue) {
                    printf("Unknown option: %s\n", argv[i]);
                    printUsage(argv[0]);
                    MPI_Finalize();
                    return 1;
                }
            }
        }
        
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark (MPI)\n");
        printf("Number of MPI processes: %d\n", nprocs);
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Broadcast parameters to all ranks
    MPI_Bcast(&numNodes, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    
    // Allocate matrices on all ranks
    std::vector<unsigned int> dist(numNodes * numNodes);
    std::vector<unsigned int> path(numNodes * numNodes);
    
    // Initialize on rank 0
    if (rank == 0) {
        printf("Initializing graph...\n");
        initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
        initializePathMatrix(path, numNodes);
    }
    
    // Broadcast initial matrices to all ranks
    MPI_Bcast(dist.data(), numNodes * numNodes, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(path.data(), numNodes * numNodes, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    
    // Run Floyd-Warshall
    if (rank == 0) {
        printf("Computing shortest paths...\n");
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    floydWarshall(dist, path, numNodes, rank, nprocs);
    
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
