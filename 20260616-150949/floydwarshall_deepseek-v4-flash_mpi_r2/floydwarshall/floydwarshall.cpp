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
    
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments on rank 0 only
    if (world_rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                numNodes = static_cast<size_t>(atoi(argv[++i]));
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
    unsigned long numNodes_ul = static_cast<unsigned long>(numNodes);
    MPI_Bcast(&numNodes_ul, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    numNodes = static_cast<size_t>(numNodes_ul);
    
    int validate_int = validate ? 1 : 0;
    MPI_Bcast(&validate_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
    validate = (validate_int != 0);
    
    int print_int = printResults ? 1 : 0;
    MPI_Bcast(&print_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
    printResults = (print_int != 0);
    
    // Rank 0 prints header information
    if (world_rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Number of MPI processes: %d\n", world_size);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Rank 0 allocates and initializes full matrices
    std::vector<unsigned int> full_dist;
    std::vector<unsigned int> full_path;
    if (world_rank == 0) {
        full_dist.resize(numNodes * numNodes);
        full_path.resize(numNodes * numNodes);
        printf("Initializing graph...\n");
        initializeDistanceMatrix(full_dist, numNodes, 1, MAX_DISTANCE);
        initializePathMatrix(full_path, numNodes);
    }
    
    // Determine 1D row-wise block distribution
    // Each process gets a contiguous block of rows
    const size_t rows_per_proc = (numNodes + static_cast<size_t>(world_size) - 1)
                                  / static_cast<size_t>(world_size);
    const size_t start_row = static_cast<size_t>(world_rank) * rows_per_proc;
    const size_t end_row   = std::min(start_row + rows_per_proc, numNodes);
    const size_t local_rows = end_row - start_row;
    const size_t local_elems = local_rows * numNodes;
    
    // Allocate local storage (at least 1 element to avoid null data with MPI)
    const size_t local_alloc = (local_elems > 0) ? local_elems : 1;
    std::vector<unsigned int> local_dist(local_alloc);
    std::vector<unsigned int> local_path(local_alloc);
    
    // Prepare scatter/gather counts and displacements
    std::vector<int> sendcounts(static_cast<size_t>(world_size));
    std::vector<int> displs(static_cast<size_t>(world_size));
    for (int p = 0; p < world_size; ++p) {
        const size_t p_start = static_cast<size_t>(p) * rows_per_proc;
        const size_t p_end   = std::min(p_start + rows_per_proc, numNodes);
        const size_t p_rows  = p_end - p_start;
        sendcounts[p] = static_cast<int>(p_rows * numNodes);
        displs[p]     = static_cast<int>(p_start * numNodes);
    }
    
    const int local_count = static_cast<int>(local_elems);
    
    // Scatter rows of dist and path from rank 0 to all processes
    MPI_Scatterv(world_rank == 0 ? full_dist.data() : nullptr,
                 sendcounts.data(), displs.data(), MPI_UNSIGNED,
                 local_dist.data(), local_count, MPI_UNSIGNED,
                 0, MPI_COMM_WORLD);
    MPI_Scatterv(world_rank == 0 ? full_path.data() : nullptr,
                 sendcounts.data(), displs.data(), MPI_UNSIGNED,
                 local_path.data(), local_count, MPI_UNSIGNED,
                 0, MPI_COMM_WORLD);
    
    // Parallel Floyd-Warshall with 1D row-wise decomposition
    if (world_rank == 0) {
        printf("Computing shortest paths...\n");
    }
    
    const double start_time = MPI_Wtime();
    
    // Reusable broadcast buffer for the currently active row k
    std::vector<unsigned int> row_buf(numNodes);
    
    for (size_t k = 0; k < numNodes; ++k) {
        // The process that owns row k
        const int owner = static_cast<int>(k / rows_per_proc);
        
        // Owner copies its row k into the broadcast buffer
        if (world_rank == owner) {
            const size_t local_k = k - start_row;
            std::copy_n(&local_dist[local_k * numNodes], numNodes, row_buf.data());
        }
        
        // Broadcast row k from its owner to all processes
        MPI_Bcast(row_buf.data(), static_cast<int>(numNodes), MPI_UNSIGNED,
                  owner, MPI_COMM_WORLD);
        
        // Each process updates its local rows using k as the intermediate node
        for (size_t i = start_row; i < end_row; ++i) {
            const size_t local_i = i - start_row;
            const unsigned int distIK = local_dist[local_i * numNodes + k];
            unsigned int* const dist_row = &local_dist[local_i * numNodes];
            unsigned int* const path_row = &local_path[local_i * numNodes];
            
            for (size_t j = 0; j < numNodes; ++j) {
                const unsigned int newDist = distIK + row_buf[j];
                if (newDist < dist_row[j]) {
                    dist_row[j] = newDist;
                    path_row[j] = static_cast<unsigned int>(k);
                }
            }
        }
    }
    
    const double end_time = MPI_Wtime();
    
    // Gather results back to rank 0
    MPI_Gatherv(local_dist.data(), local_count, MPI_UNSIGNED,
                world_rank == 0 ? full_dist.data() : nullptr,
                sendcounts.data(), displs.data(), MPI_UNSIGNED,
                0, MPI_COMM_WORLD);
    MPI_Gatherv(local_path.data(), local_count, MPI_UNSIGNED,
                world_rank == 0 ? full_path.data() : nullptr,
                sendcounts.data(), displs.data(), MPI_UNSIGNED,
                0, MPI_COMM_WORLD);
    
    // Rank 0 handles all output and validation
    if (world_rank == 0) {
        const double duration = end_time - start_time;
        printf("Computation time: %.0f ms\n", duration * 1000.0);
        
        // Floyd-Warshall has O(n³) complexity
        const double ops = static_cast<double>(numNodes)
                         * static_cast<double>(numNodes)
                         * static_cast<double>(numNodes);
        const double gflops = ops / duration / 1.0e9;
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
