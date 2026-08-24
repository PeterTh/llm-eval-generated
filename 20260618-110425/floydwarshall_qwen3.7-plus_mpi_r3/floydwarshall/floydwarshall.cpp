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

// Index calculation for flattened 2D array (column-major, matching original)
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

bool validateResult(const std::vector<unsigned int>& dist, const size_t numNodes) {
    // 1. Diagonal should be zero
    for (size_t i = 0; i < numNodes; ++i) {
        if (dist[idx2(i, i, numNodes)] != 0) {
            printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", i, i);
            return false;
        }
    }

    // 2. Triangle inequality: dist[i][k] + dist[k][j] >= dist[i][j]
    for (size_t i = 0; i < std::min(numNodes, static_cast<size_t>(10)); ++i) {
        for (size_t j = 0; j < std::min(numNodes, static_cast<size_t>(10)); ++j) {
            for (size_t k = 0; k < numNodes; ++k) {
                const unsigned int distIJ = dist[idx2(j, i, numNodes)];
                const unsigned int distIK = dist[idx2(k, i, numNodes)];
                const unsigned int distKJ = dist[idx2(j, k, numNodes)];

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

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (all processes parse identically)
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
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI processes: %d\n", num_procs);
    }

    const int n = static_cast<int>(numNodes);

    // 2D block distribution: arrange processes in a grid
    int grid_rows = static_cast<int>(std::sqrt(num_procs));
    while (num_procs % grid_rows != 0) grid_rows--;
    int grid_cols = num_procs / grid_rows;

    int my_row = rank / grid_cols;
    int my_col = rank % grid_cols;

    // Compute block dimensions
    int block_rows = (n + grid_rows - 1) / grid_rows;
    int block_cols = (n + grid_cols - 1) / grid_cols;

    // Local block dimensions (handle edge cases)
    int local_row_start = my_row * block_rows;
    int local_col_start = my_col * block_cols;
    int local_rows = std::min(block_rows, n - local_row_start);
    int local_cols = std::min(block_cols, n - local_col_start);

    if (local_rows <= 0 || local_cols <= 0) {
        local_rows = local_cols = 0;
    }

    // Local matrices in row-major order
    std::vector<unsigned int> local_dist(local_rows * local_cols);
    std::vector<unsigned int> local_path(local_rows * local_cols);

    // Initialize path matrix locally
    for (int li = 0; li < local_rows; li++) {
        for (int lj = 0; lj < local_cols; lj++) {
            int global_j = local_col_start + lj;
            local_path[li * local_cols + lj] = static_cast<unsigned int>(global_j);
        }
    }

    // Initialize distance matrix on rank 0 and distribute blocks
    if (rank == 0) {
        printf("Initializing graph...\n");
        fflush(stdout);

        std::vector<unsigned int> dist(n * n);
        initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);

        // Transpose from column-major to row-major
        std::vector<unsigned int> dist_rm(n * n);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                dist_rm[i * n + j] = dist[j * n + i];
            }
        }

        // Distribute blocks to all processes
        for (int p = 0; p < num_procs; p++) {
            int p_row = p / grid_cols;
            int p_col = p % grid_cols;
            int p_row_start = p_row * block_rows;
            int p_col_start = p_col * block_cols;
            int p_rows = std::min(block_rows, n - p_row_start);
            int p_cols = std::min(block_cols, n - p_col_start);

            if (p_rows <= 0 || p_cols <= 0) continue;

            std::vector<unsigned int> block(p_rows * p_cols);
            for (int li = 0; li < p_rows; li++) {
                int global_i = p_row_start + li;
                for (int lj = 0; lj < p_cols; lj++) {
                    int global_j = p_col_start + lj;
                    block[li * p_cols + lj] = dist_rm[global_i * n + global_j];
                }
            }

            if (p == 0) {
                local_dist = block;
            } else {
                MPI_Send(block.data(), p_rows * p_cols, MPI_UNSIGNED, p, 0, MPI_COMM_WORLD);
            }
        }
    } else {
        if (local_rows > 0 && local_cols > 0) {
            MPI_Recv(local_dist.data(), local_rows * local_cols, MPI_UNSIGNED, 0, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    // Create row and column communicators
    MPI_Comm row_comm, col_comm;
    MPI_Comm_split(MPI_COMM_WORLD, my_row, my_col, &row_comm);
    MPI_Comm_split(MPI_COMM_WORLD, my_col, my_row, &col_comm);
    
    // Get ranks within sub-communicators
    int row_rank, col_rank;
    MPI_Comm_rank(row_comm, &row_rank);
    MPI_Comm_rank(col_comm, &col_rank);

    // Buffers for row and column k
    std::vector<unsigned int> row_k(local_rows);
    std::vector<unsigned int> col_k(local_cols);

    if (rank == 0) {
        printf("Computing shortest paths...\n");
        fflush(stdout);
    }

    // Synchronize before timing
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    // === Parallel Floyd-Warshall with 2D block distribution ===
    for (int k = 0; k < n; k++) {
        int k_row_block = k / block_rows;
        int k_col_block = k / block_cols;
        int local_k_row = k - k_row_block * block_rows;
        int local_k_col = k - k_col_block * block_cols;

        // Step 1: Processes in column k_col_block broadcast row k to their row
        // All processes in the row communicator must participate
        if (my_col == k_col_block) {
            // This process has the data to broadcast
            if (local_k_col < local_cols && local_rows > 0) {
                for (int li = 0; li < local_rows; li++) {
                    row_k[li] = local_dist[li * local_cols + local_k_col];
                }
            }
            MPI_Bcast(row_k.data(), local_rows, MPI_UNSIGNED, row_rank, row_comm);
        } else {
            // This process receives the broadcast
            MPI_Bcast(row_k.data(), local_rows, MPI_UNSIGNED, k_col_block, row_comm);
        }

        // Step 2: Processes in row k_row_block broadcast column k to their column
        // All processes in the column communicator must participate
        if (my_row == k_row_block) {
            // This process has the data to broadcast
            if (local_k_row < local_rows && local_cols > 0) {
                for (int lj = 0; lj < local_cols; lj++) {
                    col_k[lj] = local_dist[local_k_row * local_cols + lj];
                }
            }
            MPI_Bcast(col_k.data(), local_cols, MPI_UNSIGNED, col_rank, col_comm);
        } else {
            // This process receives the broadcast
            MPI_Bcast(col_k.data(), local_cols, MPI_UNSIGNED, k_row_block, col_comm);
        }
        // Step 3: Update local block
        if (local_rows > 0 && local_cols > 0) {
            for (int li = 0; li < local_rows; li++) {
                unsigned int dik = row_k[li];
                for (int lj = 0; lj < local_cols; lj++) {
                    unsigned int new_dist = dik + col_k[lj];
                    if (new_dist < local_dist[li * local_cols + lj]) {
                        local_dist[li * local_cols + lj] = new_dist;
                        local_path[li * local_cols + lj] = static_cast<unsigned int>(k);
                    }
                }
            }
        }
    }

    // Synchronize after computation for accurate timing
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long duration_ms = duration.count();
    long max_duration_ms = 0;
    MPI_Reduce(&duration_ms, &max_duration_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather results to rank 0
    if (rank == 0) {
        std::vector<unsigned int> dist(n * n);
        
        // Copy local block first
        for (int li = 0; li < local_rows; li++) {
            int global_i = local_row_start + li;
            for (int lj = 0; lj < local_cols; lj++) {
                int global_j = local_col_start + lj;
                dist[global_i * n + global_j] = local_dist[li * local_cols + lj];
            }
        }

        // Receive blocks from other processes
        for (int p = 1; p < num_procs; p++) {
            int p_row = p / grid_cols;
            int p_col = p % grid_cols;
            int p_row_start = p_row * block_rows;
            int p_col_start = p_col * block_cols;
            int p_rows = std::min(block_rows, n - p_row_start);
            int p_cols = std::min(block_cols, n - p_col_start);

            // Only receive if this process has data
            if (p_rows > 0 && p_cols > 0) {
                std::vector<unsigned int> block(p_rows * p_cols);
                MPI_Recv(block.data(), p_rows * p_cols, MPI_UNSIGNED, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                // Unpack block into full matrix
                for (int li = 0; li < p_rows; li++) {
                    int global_i = p_row_start + li;
                    for (int lj = 0; lj < p_cols; lj++) {
                        int global_j = p_col_start + lj;
                        dist[global_i * n + global_j] = block[li * p_cols + lj];
                    }
                }
            }
        }

        // Transpose back from row-major to column-major for output/validation
        std::vector<unsigned int> dist_cm(n * n);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                dist_cm[j * n + i] = dist[i * n + j];
            }
        }

        printf("Computation time: %ld ms\n", max_duration_ms);

        // Calculate operations per second (Floyd-Warshall has O(n³) complexity)
        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / (max_duration_ms / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);

        // Print results for external validation (integer hash-based)
        if (printResults) {
            print_results_int(dist_cm, "DistanceMatrix");
        }

        // Validation
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(dist_cm, numNodes);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                MPI_Finalize();
                return 1;
            }
        }
    } else {
        if (local_rows > 0 && local_cols > 0) {
            MPI_Send(local_dist.data(), local_rows * local_cols, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD);
        }
    }

    // Ensure all processes complete their MPI operations before cleanup
    MPI_Barrier(MPI_COMM_WORLD);

    // Free sub-communicators only if they were created
    if (local_rows > 0 && local_cols > 0) {
        MPI_Comm_free(&row_comm);
        MPI_Comm_free(&col_comm);
    }

    // Clean up local data structures before MPI_Finalize
    if (!local_dist.empty()) {
        local_dist.clear();
        local_dist.shrink_to_fit();
    }
    if (!local_path.empty()) {
        local_path.clear();
        local_path.shrink_to_fit();
    }
    if (!row_k.empty()) {
        row_k.clear();
        row_k.shrink_to_fit();
    }
    if (!col_k.empty()) {
        col_k.clear();
        col_k.shrink_to_fit();
    }

    MPI_Finalize();
    return 0;
}