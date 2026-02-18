#include <mpi.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

// Index calculation for flattened 2D array
// Access: dist[row][col] -> dist[idx2(col, row, n)] -> dist[row * n + col]
inline constexpr size_t idx2(const size_t col, const size_t row, const size_t n) noexcept {
    return row * n + col;
}

void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes, 
                              const unsigned int rangeMin, const unsigned int rangeMax,
                              int rank, int size) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;
    
    // Each rank calculates its own rows
    // To maintain identical random number sequence as original:
    // We must generate all random numbers but only store the ones for our rows.
    // Or simpler: generate full matrix on rank 0 and scatter.
    // Given memory constraints on large N, generating locally is better but requires skipping.
    // rand_r is cheap. We can just generate and discard.
    
    // However, distributing rows:
    // total rows = numNodes.
    // basic rows per rank = numNodes / size
    // remainder = numNodes % size
    // rank r has rows [start_row, end_row)
    
    size_t rows_per_proc = numNodes / size;
    size_t remainder = numNodes % size;
    
    size_t start_row = rank * rows_per_proc + std::min((size_t)rank, remainder);
    size_t end_row = start_row + rows_per_proc + (rank < (int)remainder ? 1 : 0);
    size_t num_local_rows = end_row - start_row;

    // Resize dist to hold only local rows
    // But wait, the original code passes a full vector reference.
    // We need to change the function signature or semantics.
    // Let's change main() to allocate local part only.
    
    // For exact reproducibility of random numbers:
    // It's a linear congruential generator.
    // rand_r(&seed) updates seed.
    // The sequence is determined by the order of calls.
    // The original code fills row by row (i=0..N*N).
    // So row 0 gets first N random numbers, row 1 gets next N, etc.
    
    // Optimization: Skip random numbers for rows before start_row.
    // Since we need to match original behavior exactly:
    
    // Skip rows before start_row
    for (size_t i = 0; i < start_row * numNodes; ++i) {
        rand_r(&seed);
    }
    
    // Generate for local rows
    for (size_t r = 0; r < num_local_rows; ++r) {
        for (size_t c = 0; c < numNodes; ++c) {
             dist[r * numNodes + c] = rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
        }
    }
    
    // If validation/printing is required, we might need full matrix on rank 0 later.
    
    // Set diagonal to zero (distance from node to itself is 0)
    for (size_t r = 0; r < num_local_rows; ++r) {
        size_t global_row = start_row + r;
        dist[r * numNodes + global_row] = 0;
    }
}

void initializePathMatrix(std::vector<unsigned int>& path, const size_t numNodes,
                          int rank, int size) {
    size_t rows_per_proc = numNodes / size;
    size_t remainder = numNodes % size;
    size_t start_row = rank * rows_per_proc + std::min((size_t)rank, remainder);
    size_t end_row = start_row + rows_per_proc + (rank < (int)remainder ? 1 : 0);
    size_t num_local_rows = end_row - start_row;

    for (size_t r = 0; r < num_local_rows; ++r) {
        size_t global_row = start_row + r;
        for (size_t c = 0; c < numNodes; ++c) {
             path[r * numNodes + c] = c; // Original: path[idx2(c, global_row, numNodes)] = c?
             // Original logic:
             // path[idx2(i, j, numNodes)] = j; // path[j][i] = i? No.
             // idx2(col, row, n) = row*n + col.
             // Original loop:
             // for j (row?):
             //   for i (col?):
             //     path[idx2(i, j, n)] = j; // path[row=j][col=i] = j?
             //     path[idx2(j, i, n)] = i; // path[row=i][col=j] = i?
             
             // Wait, the original code initializes path matrix symmetrically-ish?
             /*
             for (size_t j = 0; j < numNodes; ++j) {
                for (size_t i = 0; i < numNodes; ++i) {
                    path[idx2(i, j, numNodes)] = j;
                    path[idx2(j, i, numNodes)] = i;
                }
                path[idx2(j, j, numNodes)] = j;
            }
            */
            // Let's trace indices.
            // path[idx2(i, j, n)] accesses row j, col i. Sets to j.
            // path[idx2(j, i, n)] accesses row i, col j. Sets to i.
            // So path[r][c] is set to r?
            // If row=j, col=i -> path[j][i] = j.
            // If row=i, col=j -> path[i][j] = i.
            // So path[r][c] = r.
            // Wait.
            // If I look at row r, col c.
            // It was set when j=r, i=c as path[r][c] = r.
            // It was ALSO set when j=c, i=r as path[r][c] = c.
            // Which one wins? The later one?
            // The loop order is j outer, i inner.
            // So for a given (r,c):
            // When j=r, i=c -> path[r][c] = r.
            // When j=c, i=r -> path[c][r] = c.
            // Wait.
            // Let's just implement the loop logic directly but carefully.
            // Or observe the pattern.
            // j goes 0..N. i goes 0..N.
            // For a specific cell (r, c):
            // It is touched when j=r, i=c -> path[r][c] = r.
            // It is touched when j=c, i=r -> path[r][c] = r (since path[idx2(c, r, n)] is path[row=r][col=c]).
            // So in both cases it seems to be setting path[r][c] = r (predecessor is row index?).
            // Let's verify idx2 semantics again.
            // idx2(col, row, n) = row * n + col.
            // original: path[idx2(i, j, n)] = j.  -> path[row=j][col=i] = j.
            // original: path[idx2(j, i, n)] = i.  -> path[row=i][col=j] = i.
            // So path[r][c] is set to r when j=r, i=c.
            // And path[r][c] is set to r when j=c, i=r.
            // So yes, path[r][c] should be initialized to r.
            // Predecessor of c on path from r is r? That means direct edge?
            // Usually path[i][j] stores the predecessor of j on shortest path from i.
            // Initially, if edge exists, it is i.
            // So yes, path[r][c] = r seems correct for initialization.
             
             path[r * numNodes + c] = global_row;
        }
        path[r * numNodes + global_row] = global_row;
    }
}

void floydWarshall(std::vector<unsigned int>& dist, 
                   std::vector<unsigned int>& path, 
                   const size_t numNodes,
                   int rank, int size) {
    
    size_t rows_per_proc = numNodes / size;
    size_t remainder = numNodes % size;
    size_t start_row = rank * rows_per_proc + std::min((size_t)rank, remainder);
    size_t end_row = start_row + rows_per_proc + (rank < (int)remainder ? 1 : 0);
    size_t num_local_rows = end_row - start_row;

    // Buffer to receive the k-th row from the owner
    std::vector<unsigned int> k_dist_row(numNodes);
    std::vector<unsigned int> k_path_row(numNodes); // We don't need k-th path row for algo, just dist.

    // For each intermediate node k
    for (size_t k = 0; k < numNodes; ++k) {
        // Find owner of row k
        // We can compute owner based on k
        int owner = -1;
        // Inverse of start_row mapping is tricky with remainder.
        // It's easier to just compute bounds.
        // Actually, simple logic:
        // if k < remainder * (rows_per_proc + 1) -> owner = k / (rows_per_proc + 1)
        // else -> owner = remainder + (k - remainder * (rows_per_proc + 1)) / rows_per_proc
        
        if (k < remainder * (rows_per_proc + 1)) {
            owner = k / (rows_per_proc + 1);
        } else {
            owner = remainder + (k - remainder * (rows_per_proc + 1)) / rows_per_proc;
        }

        // If I am owner, copy row k to buffer
        if (rank == owner) {
            size_t local_k = k - start_row;
            // Copy my local row to buffer
            std::copy(dist.begin() + local_k * numNodes, 
                      dist.begin() + (local_k + 1) * numNodes, 
                      k_dist_row.begin());
        }

        // Broadcast k-th row to all
        MPI_Bcast(k_dist_row.data(), numNodes, MPI_UNSIGNED, owner, MPI_COMM_WORLD);

        // Update local rows
        for (size_t r = 0; r < num_local_rows; ++r) {
            
            // Optimization: If dist[i][k] is INF, we can skip updating this row?
            // Check dist[i][k]
            // In local terms: dist[r][k]
            unsigned int distIK = dist[r * numNodes + k];
            
            if (distIK == INF) continue;

            for (size_t j = 0; j < numNodes; ++j) {
                // dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
                // dist[k][j] is in k_dist_row[j]
                
                unsigned int distKJ = k_dist_row[j];
                // if (distKJ == INF) continue; // Removed for vectorization

                unsigned int newDist = distIK + distKJ;
                if (newDist < dist[r * numNodes + j]) {
                    dist[r * numNodes + j] = newDist;
                    path[r * numNodes + j] = k; 
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
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    
    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark (MPI)\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("MPI processes: %d\n", size);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Determine local rows
    size_t rows_per_proc = numNodes / size;
    size_t remainder = numNodes % size;
    size_t start_row = rank * rows_per_proc + std::min((size_t)rank, remainder);
    size_t end_row = start_row + rows_per_proc + (rank < (int)remainder ? 1 : 0);
    size_t num_local_rows = end_row - start_row;

    // Allocate local matrices
    // dist and path stores only local rows
    std::vector<unsigned int> dist(num_local_rows * numNodes);
    std::vector<unsigned int> path(num_local_rows * numNodes);
    
    // Initialize
    if (rank == 0) printf("Initializing graph...\n");
    initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE, rank, size);
    initializePathMatrix(path, numNodes, rank, size);
    
    // Run Floyd-Warshall
    if (rank == 0) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    floydWarshall(dist, path, numNodes, rank, size);
    
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
    
    // Gather results if needed
    if (printResults || validate) {
        std::vector<unsigned int> global_dist;
        if (rank == 0) {
            global_dist.resize(numNodes * numNodes);
        }

        // We need to gather variable sized chunks.
        std::vector<int> recvcounts(size);
        std::vector<int> displs(size);
        
        // Calculate counts and displacements
        for (int r = 0; r < size; ++r) {
             size_t r_start_row = r * rows_per_proc + std::min((size_t)r, remainder);
             size_t r_end_row = r_start_row + rows_per_proc + (r < (int)remainder ? 1 : 0);
             recvcounts[r] = (r_end_row - r_start_row) * numNodes;
             displs[r] = r_start_row * numNodes;
        }

        MPI_Gatherv(dist.data(), num_local_rows * numNodes, MPI_UNSIGNED,
                    global_dist.data(), recvcounts.data(), displs.data(), MPI_UNSIGNED,
                    0, MPI_COMM_WORLD);

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
    }
    
    MPI_Finalize();
    return 0;
}
