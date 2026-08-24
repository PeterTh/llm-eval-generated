#include <algorithm>
#include <chrono>
#include <climits>
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
    
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    
    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments on root
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
    
    // Broadcast parameters from root to all ranks
    {
        int nn = static_cast<int>(numNodes);
        MPI_Bcast(&nn, 1, MPI_INT, 0, MPI_COMM_WORLD);
        numNodes = static_cast<size_t>(nn);
    }
    {
        char v = validate ? 1 : 0;
        MPI_Bcast(&v, 1, MPI_CHAR, 0, MPI_COMM_WORLD);
        validate = (v != 0);
    }
    {
        char p = printResults ? 1 : 0;
        MPI_Bcast(&p, 1, MPI_CHAR, 0, MPI_COMM_WORLD);
        printResults = (p != 0);
    }
    
    // Compute 1D row distribution
    size_t base_rows = numNodes / num_procs;
    size_t rem_rows = numNodes % num_procs;
    
    std::vector<int> send_counts(num_procs, 0);
    std::vector<int> send_displs(num_procs, 0);
    std::vector<size_t> rank_start_row(num_procs, 0);
    
    size_t offset = 0;
    for (int r = 0; r < num_procs; ++r) {
        size_t rows_r = base_rows + (static_cast<size_t>(r) < rem_rows ? 1 : 0);
        send_counts[r] = static_cast<int>(rows_r * numNodes);
        send_displs[r] = static_cast<int>(offset);
        rank_start_row[r] = offset / numNodes;
        offset += rows_r * numNodes;
    }
    
    const size_t local_n = send_counts[rank] / numNodes;
    const size_t start_row = send_displs[rank] / numNodes;
    
    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Number of MPI processes: %d\n", num_procs);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Root allocates and initializes full matrices
    std::vector<unsigned int> full_dist;
    std::vector<unsigned int> full_path;
    if (rank == 0) {
        full_dist.resize(numNodes * numNodes);
        full_path.resize(numNodes * numNodes);
        printf("Initializing graph...\n");
        initializeDistanceMatrix(full_dist, numNodes, 1, MAX_DISTANCE);
        initializePathMatrix(full_path, numNodes);
    }
    
    // Allocate local row blocks
    std::vector<unsigned int> local_dist(local_n * numNodes);
    std::vector<unsigned int> local_path(local_n * numNodes);
    
    // Scatter rows from root to all ranks
    MPI_Scatterv(rank == 0 ? full_dist.data() : nullptr,
                 send_counts.data(), send_displs.data(), MPI_UNSIGNED,
                 local_dist.data(), static_cast<int>(local_n * numNodes), MPI_UNSIGNED,
                 0, MPI_COMM_WORLD);
    
    MPI_Scatterv(rank == 0 ? full_path.data() : nullptr,
                 send_counts.data(), send_displs.data(), MPI_UNSIGNED,
                 local_path.data(), static_cast<int>(local_n * numNodes), MPI_UNSIGNED,
                 0, MPI_COMM_WORLD);
    
    // Free root's full matrices — no longer needed until gather
    if (rank == 0) {
        full_dist.clear();
        full_path.clear();
        full_dist.shrink_to_fit();
        full_path.shrink_to_fit();
    }
    
    // Buffer for receiving the broadcast row k each iteration
    std::vector<unsigned int> row_k_buf(numNodes);
    
    if (rank == 0) {
        printf("Computing shortest paths...\n");
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    // Parallel Floyd-Warshall: 1D row distribution
    // For each intermediate node k:
    //   - The rank owning row k broadcasts it to all ranks
    //   - Each rank updates its local rows using the broadcast row k
    int owning_rank = 0;
    for (size_t k = 0; k < numNodes; ++k) {
        // Find which rank owns row k (monotonic walk, O(1) amortized)
        while (owning_rank + 1 < num_procs && k >= rank_start_row[owning_rank + 1]) {
            owning_rank++;
        }
        
        // The owning rank copies its row k into the broadcast buffer
        if (rank == owning_rank) {
            const size_t local_k = k - start_row;
            std::memcpy(row_k_buf.data(),
                        &local_dist[local_k * numNodes],
                        numNodes * sizeof(unsigned int));
        }
        
        // Broadcast row k to all ranks
        MPI_Bcast(row_k_buf.data(), static_cast<int>(numNodes), MPI_UNSIGNED,
                  owning_rank, MPI_COMM_WORLD);
        
        // Every rank updates its local rows using the received row k
        // Update rule: dist[j][i] = min(dist[j][i], dist[k][i] + dist[j][k])
        for (size_t local_j = 0; local_j < local_n; ++local_j) {
            unsigned int* dist_row = &local_dist[local_j * numNodes];
            unsigned int* path_row = &local_path[local_j * numNodes];
            const unsigned int dist_jk = dist_row[k];   // dist[j][k]  (constant for this row)
            
            for (size_t i = 0; i < numNodes; ++i) {
                const unsigned int new_dist = row_k_buf[i] + dist_jk;
                if (new_dist < dist_row[i]) {
                    dist_row[i] = new_dist;
                    path_row[i] = static_cast<unsigned int>(k);
                }
            }
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long local_duration_ms = static_cast<long>(duration.count());
    long global_duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &global_duration_ms, 1, MPI_LONG, MPI_MAX,
               0, MPI_COMM_WORLD);
    
    // Gather the full distance matrix back to rank 0 for validation/output
    if (rank == 0) {
        full_dist.resize(numNodes * numNodes);
    }
    
    MPI_Gatherv(local_dist.data(), static_cast<int>(local_n * numNodes), MPI_UNSIGNED,
                rank == 0 ? full_dist.data() : nullptr,
                send_counts.data(), send_displs.data(), MPI_UNSIGNED,
                0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Computation time: %ld ms\n", global_duration_ms);
        
        // Calculate operations per second
        double ops = static_cast<double>(numNodes) * numNodes * numNodes;
        double gflops = ops / (global_duration_ms / 1000.0) / 1.0e9;
        printf("Performance: %.3f GOPS\n", gflops);
        
        // Print results for external validation (integer hash-based)
        if (printResults) {
            print_results_int(full_dist, "DistanceMatrix");
        }
        
        // Validation
        if (validate) {
            printf("Validating result...\n");
            if (validateResult(full_dist, numNodes)) {
                printf("Validation: PASSED\n");
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
