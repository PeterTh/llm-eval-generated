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

// Index calculation for flattened 2D array (row-major)
// idx2(col, row, num_cols) -> row * num_cols + col
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

void initializeDistanceMatrix(std::vector<unsigned int>& local_dist, const size_t numNodes, 
                              const unsigned int rangeMin, const unsigned int rangeMax,
                              const size_t start_row, const size_t end_row) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;
    
    // Iterate over all elements to keep rand_r sequence consistent
    for (size_t k = 0; k < numNodes * numNodes; ++k) {
        unsigned int val = rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
        
        size_t row = k / numNodes;
        size_t col = k % numNodes;
        
        if (row >= start_row && row < end_row) {
            local_dist[(row - start_row) * numNodes + col] = val;
        }
    }
    
    // Set diagonal to zero (distance from node to itself is 0)
    for (size_t i = 0; i < numNodes; ++i) {
        if (i >= start_row && i < end_row) {
            local_dist[(i - start_row) * numNodes + i] = 0;
        }
    }
}

void initializePathMatrix(std::vector<unsigned int>& local_path, const size_t numNodes,
                          const size_t start_row, const size_t end_row) {
    // For every row r, fill it with r
    // Since path[i][j] = i for all j
    for (size_t r = start_row; r < end_row; ++r) {
        std::fill(local_path.begin() + (r - start_row) * numNodes,
                  local_path.begin() + (r - start_row + 1) * numNodes,
                  r);
    }
}

void floydWarshall(std::vector<unsigned int>& local_dist, 
                   std::vector<unsigned int>& local_path, 
                   const size_t numNodes,
                   const int rank, const int size,
                   const size_t start_row, const size_t end_row) {
    
    std::vector<unsigned int> k_row(numNodes);

    // Classic Floyd-Warshall algorithm
    // For each intermediate node k
    for (size_t k = 0; k < numNodes; ++k) {
        // Owner of row k broadcasts it
        int owner = -1;
        // Find owner: simple block distribution logic
        // We need to know who owns row k.
        // Assuming block distribution: rows divided as cleanly as possible.
        // Rank r owns [r * (N/P) + min(r, N%P), (r+1)*(N/P) + min(r+1, N%P))
        
        size_t rows_per_proc = numNodes / size;
        size_t remainder = numNodes % size;
        
        if (k < remainder * (rows_per_proc + 1)) {
            owner = k / (rows_per_proc + 1);
        } else {
            owner = remainder + (k - remainder * (rows_per_proc + 1)) / rows_per_proc;
        }

        if (rank == owner) {
            // Copy row k to buffer
            // Row k is at local index k - start_row
            size_t local_k = k - start_row;
            std::copy(local_dist.begin() + local_k * numNodes, 
                      local_dist.begin() + (local_k + 1) * numNodes, 
                      k_row.begin());
        }

        MPI_Bcast(k_row.data(), numNodes, MPI_UNSIGNED, owner, MPI_COMM_WORLD);

        // Update local rows
        for (size_t i = start_row; i < end_row; ++i) {
            size_t local_i = i - start_row;
            // dist[i][k]
            unsigned int distIK = local_dist[local_i * numNodes + k];
            
            // Optimization: if distIK is INF, we can skip
            if (distIK == INF) continue;

            for (size_t j = 0; j < numNodes; ++j) {
                // dist[k][j] comes from broadcasted row
                unsigned int distKJ = k_row[j];
                
                if (distKJ != INF) {
                    unsigned int newDist = distIK + distKJ;
                    unsigned int currentDist = local_dist[local_i * numNodes + j];
                    
                    if (newDist < currentDist) {
                        local_dist[local_i * numNodes + j] = newDist;
                        local_path[local_i * numNodes + j] = k;
                    }
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
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) printf("Unknown option: %s\n", argv[i]);
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 1;
        }
    }
    
    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark (MPI Parallel)\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Number of processes: %d\n", size);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Determine local rows
    size_t rows_per_proc = numNodes / size;
    size_t remainder = numNodes % size;
    size_t start_row = rank * rows_per_proc + std::min((size_t)rank, remainder);
    size_t end_row = start_row + rows_per_proc + (rank < (int)remainder ? 1 : 0);
    size_t num_local_rows = end_row - start_row;
    
    // Allocate local matrices
    std::vector<unsigned int> local_dist(num_local_rows * numNodes);
    std::vector<unsigned int> local_path(num_local_rows * numNodes);
    
    // Initialize
    if (rank == 0) printf("Initializing graph...\n");
    initializeDistanceMatrix(local_dist, numNodes, 1, MAX_DISTANCE, start_row, end_row);
    initializePathMatrix(local_path, numNodes, start_row, end_row);
    
    MPI_Barrier(MPI_COMM_WORLD);

    // Run Floyd-Warshall
    if (rank == 0) printf("Computing shortest paths...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    floydWarshall(local_dist, local_path, numNodes, rank, size, start_row, end_row);
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long long local_duration_ms = duration.count();
    long long max_duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &max_duration_ms, 1, MPI_LONG_LONG_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Computation time: %lld ms\n", max_duration_ms);
        
        // Calculate operations per second
        // Floyd-Warshall has O(n³) complexity
        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / (max_duration_ms / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);
    }
    
    // Collect results for validation or output
    std::vector<unsigned int> global_dist;
    if (validate || printResults) {
        if (rank == 0) {
            global_dist.resize(numNodes * numNodes);
        }
        
        // Gather counts and displacements
        std::vector<int> recvcounts(size);
        std::vector<int> displs(size);
        
        int local_count = num_local_rows * numNodes;
        MPI_Gather(&local_count, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            displs[0] = 0;
            for (int i = 1; i < size; ++i) {
                displs[i] = displs[i-1] + recvcounts[i-1];
            }
        }
        
        MPI_Gatherv(local_dist.data(), local_count, MPI_UNSIGNED, 
                    global_dist.data(), recvcounts.data(), displs.data(), MPI_UNSIGNED, 
                    0, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        // Print results for external validation (integer hash-based)
        if (printResults) {
            print_results_int(global_dist, "DistanceMatrix");
        }
        
        // Validation
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(global_dist, numNodes);
            
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
