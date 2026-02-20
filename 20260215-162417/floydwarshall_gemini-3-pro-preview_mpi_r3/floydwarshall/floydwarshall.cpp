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
// i: row, j: column, n: number of columns (width)
// Returns index for row-major storage
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return i * n + j;
}
// Note: The original code used idx2(j, i, n) = i * n + j, effectively swapping row/col semantics if i was meant to be row.
// But looking at usage:
// dist[idx2(j, i, numNodes)] -> i is row, j is col.
// So dist[row][col].
// Wait, original was: return j * n + i;
// And call was idx2(j, i, n).
// So it was i * n + j.
// Wait, let's re-verify.
// idx2(i, j, n) = j * n + i.
// idx2(col, row, n) = row * n + col.
// In usage: idx2(j, i, n) -> i * n + j. So row=i, col=j.
// This is confusing. I will rewrite idx2 to be standard row-major (row, col, width) -> row*width + col.
// And update call sites to idx2(row, col, n).

// Let's redefine idx2 to be standard (row, col, n) -> row*n + col.
// And adjust the call sites to be idx2(i, j, n) where i is row, j is col.

void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes, 
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;
    
    for (size_t i = 0; i < numNodes * numNodes; ++i) {
        dist[i] = rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
    }
    
    // Set diagonal to zero (distance from node to itself is 0)
    for (size_t i = 0; i < numNodes; ++i) {
        dist[i * numNodes + i] = 0;
    }
}

// Helper to determine row distribution
void get_dist_info(size_t numNodes, int size, int rank, size_t& start_row, size_t& end_row, size_t& num_rows) {
    size_t rows_per_proc = numNodes / size;
    size_t remainder = numNodes % size;
    
    if ((size_t)rank < remainder) {
        start_row = rank * (rows_per_proc + 1);
        num_rows = rows_per_proc + 1;
    } else {
        start_row = remainder * (rows_per_proc + 1) + (rank - remainder) * rows_per_proc;
        num_rows = rows_per_proc;
    }
    end_row = start_row + num_rows;
}

void floydWarshall(std::vector<unsigned int>& local_dist, 
                   const size_t numNodes,
                   int rank, int size) {
    
    size_t start_row, end_row, num_rows;
    get_dist_info(numNodes, size, rank, start_row, end_row, num_rows);

    std::vector<unsigned int> k_row(numNodes);

    // For each intermediate node k
    for (size_t k = 0; k < numNodes; ++k) {
        
        // Find owner of row k
        int owner_rank;
        size_t rows_per_proc = numNodes / size;
        size_t remainder = numNodes % size;
        
        if (k < remainder * (rows_per_proc + 1)) {
            owner_rank = k / (rows_per_proc + 1);
        } else {
            owner_rank = remainder + (k - remainder * (rows_per_proc + 1)) / rows_per_proc;
        }

        // If I own row k, copy it to k_row
        if (rank == owner_rank) {
            // My local index for row k is (k - start_row)
            size_t local_k = k - start_row;
            std::copy(local_dist.begin() + local_k * numNodes, 
                      local_dist.begin() + (local_k + 1) * numNodes, 
                      k_row.begin());
        }

        // Broadcast k_row
        MPI_Bcast(k_row.data(), numNodes, MPI_UNSIGNED, owner_rank, MPI_COMM_WORLD);

        // Update local rows
        for (size_t i = 0; i < num_rows; ++i) {
            // global row index
            // size_t global_i = start_row + i; // Unused but implicit
            
            // Optimization: check if dist[i][k] is INF? 
            // If dist[i][k] is INF, we can't improve anything using k.
            unsigned int distIK = local_dist[i * numNodes + k];
            if (distIK >= INF) continue;

            for (size_t j = 0; j < numNodes; ++j) {
                // Optimization: check if dist[k][j] is INF?
                unsigned int distKJ = k_row[j];
                if (distKJ >= INF) continue;

                unsigned int newDist = distIK + distKJ;
                if (newDist < local_dist[i * numNodes + j]) {
                    local_dist[i * numNodes + j] = newDist;
                }
            }
        }
    }
}

bool validateResult(const std::vector<unsigned int>& dist, const size_t numNodes) {
    // Basic sanity checks
    
    // 1. Diagonal should be zero
    for (size_t i = 0; i < numNodes; ++i) {
        if (dist[i * numNodes + i] != 0) {
            printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", i, i);
            return false;
        }
    }
    
    // 2. Triangle inequality
    for (size_t i = 0; i < std::min(numNodes, static_cast<size_t>(10)); ++i) {
        for (size_t j = 0; j < std::min(numNodes, static_cast<size_t>(10)); ++j) {
            for (size_t k = 0; k < numNodes; ++k) {
                const unsigned int distIJ = dist[i * numNodes + j];
                const unsigned int distIK = dist[i * numNodes + k];
                const unsigned int distKJ = dist[k * numNodes + j];
                
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
        }
    }
    
    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark (MPI)\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("MPI Ranks: %d\n", size);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Determine local size
    size_t start_row, end_row, num_rows;
    get_dist_info(numNodes, size, rank, start_row, end_row, num_rows);
    
    // Full matrix only on rank 0 (for initialization and gather)
    std::vector<unsigned int> dist;
    if (rank == 0) {
        dist.resize(numNodes * numNodes);
        printf("Initializing graph...\n");
        initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
    }
    
    // Local matrix part
    std::vector<unsigned int> local_dist(num_rows * numNodes);
    
    // Scatter the data
    // Prepare counts and displacements for scatterv
    std::vector<int> counts(size);
    std::vector<int> displs(size);
    
    if (rank == 0) {
        for (int r = 0; r < size; ++r) {
            size_t r_start, r_end, r_rows;
            get_dist_info(numNodes, size, r, r_start, r_end, r_rows);
            counts[r] = r_rows * numNodes;
            displs[r] = r_start * numNodes;
        }
    }

    MPI_Scatterv(rank == 0 ? dist.data() : nullptr, 
                 rank == 0 ? counts.data() : nullptr, 
                 rank == 0 ? displs.data() : nullptr, 
                 MPI_UNSIGNED, 
                 local_dist.data(), 
                 num_rows * numNodes, 
                 MPI_UNSIGNED, 
                 0, MPI_COMM_WORLD);
    
    if (rank == 0) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD); // Sync before timing

    auto start = std::chrono::high_resolution_clock::now();
    
    floydWarshall(local_dist, numNodes, rank, size);
    
    MPI_Barrier(MPI_COMM_WORLD); // Sync after timing
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        
        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);
    }

    // Gather results if needed
    if (rank == 0 || validate || printResults) {
        // We reuse the scatter structures for gather
         MPI_Gatherv(local_dist.data(), 
                    num_rows * numNodes, 
                    MPI_UNSIGNED, 
                    rank == 0 ? dist.data() : nullptr, 
                    rank == 0 ? counts.data() : nullptr, 
                    rank == 0 ? displs.data() : nullptr, 
                    MPI_UNSIGNED, 
                    0, MPI_COMM_WORLD);
    }
    
    if (rank == 0) {
        if (printResults) {
            print_results_int(dist, "DistanceMatrix");
        }
        
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
