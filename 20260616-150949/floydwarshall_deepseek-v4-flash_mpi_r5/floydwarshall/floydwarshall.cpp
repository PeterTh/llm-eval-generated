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

// Index calculation for flattened 2D array (column-major: j*N + i)
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

// --- MPI distribution helpers ---

// Compute local column range for a given rank (block distribution)
inline void get_local_bounds(const size_t n, const int num_procs, const int rank,
                             size_t& start, size_t& count) {
    const size_t base = n / static_cast<size_t>(num_procs);
    const size_t rem = n % static_cast<size_t>(num_procs);
    if (static_cast<size_t>(rank) < rem) {
        start = static_cast<size_t>(rank) * (base + 1);
        count = base + 1;
    } else {
        start = rem * (base + 1) + (static_cast<size_t>(rank) - rem) * base;
        count = base;
    }
}

// Determine which rank owns global column k
inline int find_owner(const size_t k, const size_t n, const int num_procs) {
    const size_t base = n / static_cast<size_t>(num_procs);
    const size_t rem = n % static_cast<size_t>(num_procs);
    const size_t first_uneven = rem * (base + 1);
    if (k < first_uneven) {
        return static_cast<int>(k / (base + 1));
    } else {
        return static_cast<int>(rem + (k - first_uneven) / base);
    }
}

// --- Original sequential functions (used by rank 0 for initialization) ---

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

// --- Parallel Floyd-Warshall (column-distributed MPI) ---

void floydWarshall_mpi(std::vector<unsigned int>& dist_local, 
                       std::vector<unsigned int>& path_local, 
                       const size_t numNodes,
                       const size_t local_start,
                       const size_t local_cols,
                       const int rank,
                       const int num_procs) {
    // Buffer for the column k broadcast
    std::vector<unsigned int> col_k(numNodes);

    for (size_t k = 0; k < numNodes; ++k) {
        const int owner = find_owner(k, numNodes, num_procs);

        if (rank == owner) {
            // Copy our local column k into the broadcast buffer
            const size_t k_local = k - local_start;
            const unsigned int* src = &dist_local[k_local * numNodes];
            std::copy(src, src + numNodes, col_k.begin());
        }

        // Broadcast column k from its owner to all processes
        MPI_Bcast(col_k.data(), static_cast<int>(numNodes), MPI_UNSIGNED, owner, MPI_COMM_WORLD);

        // Update all local columns using the broadcast column k
        for (size_t jl = 0; jl < local_cols; ++jl) {
            // dist[k][j] — row k of local column j — constant across all i in this column
            const unsigned int dist_kj = dist_local[jl * numNodes + k];
            unsigned int* dist_col = &dist_local[jl * numNodes];
            unsigned int* path_col = &path_local[jl * numNodes];

            for (size_t i = 0; i < numNodes; ++i) {
                const unsigned int newDist = col_k[i] + dist_kj;
                if (newDist < dist_col[i]) {
                    dist_col[i] = newDist;
                    path_col[i] = static_cast<unsigned int>(k);
                }
            }
        }
    }
}

// --- Validation (unchanged semantics) ---

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
    int validate_int = 0;
    int printResults_int = 0;
    int exit_code = -1;   // -1 means "continue normally"

    // --- Parse command line on rank 0 ---
    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                numNodes = static_cast<size_t>(atoi(argv[++i]));
            } else if (strcmp(argv[i], "-v") == 0) {
                validate_int = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults_int = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                exit_code = 0;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                exit_code = 1;
            }
        }
    }

    // Broadcast configuration to all ranks
    MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (exit_code >= 0) {
        MPI_Finalize();
        return exit_code;
    }

    MPI_Bcast(&numNodes, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults_int, 1, MPI_INT, 0, MPI_COMM_WORLD);

    const bool validate = (validate_int != 0);
    const bool printResults = (printResults_int != 0);

    // --- Determine local column partitioning ---
    size_t local_start, local_cols;
    get_local_bounds(numNodes, num_procs, rank, local_start, local_cols);

    const size_t local_size = local_cols * numNodes;

    // --- Rank 0 prints header ---
    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Number of processes: %d\n", num_procs);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // --- Build scatter/gather counts and displacements ---
    std::vector<int> scounts(num_procs);
    std::vector<int> sdispls(num_procs);
    for (int p = 0; p < num_procs; ++p) {
        size_t p_start, p_count;
        get_local_bounds(numNodes, num_procs, p, p_start, p_count);
        scounts[p]  = static_cast<int>(p_count * numNodes);
        sdispls[p]  = static_cast<int>(p_start * numNodes);
    }

    // --- Rank 0 initializes full matrices ---
    std::vector<unsigned int> dist_full;
    std::vector<unsigned int> path_full;
    if (rank == 0) {
        dist_full.resize(numNodes * numNodes);
        path_full.resize(numNodes * numNodes);

        printf("Initializing graph...\n");
        initializeDistanceMatrix(dist_full, numNodes, 1, MAX_DISTANCE);
        initializePathMatrix(path_full, numNodes);
    }

    // --- Scatter columns to all ranks ---
    std::vector<unsigned int> dist_local(local_size);
    std::vector<unsigned int> path_local(local_size);

    MPI_Scatterv(rank == 0 ? dist_full.data() : nullptr,
                 scounts.data(), sdispls.data(), MPI_UNSIGNED,
                 dist_local.data(), scounts[rank], MPI_UNSIGNED,
                 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? path_full.data() : nullptr,
                 scounts.data(), sdispls.data(), MPI_UNSIGNED,
                 path_local.data(), scounts[rank], MPI_UNSIGNED,
                 0, MPI_COMM_WORLD);

    // --- Parallel Floyd-Warshall ---
    if (rank == 0) {
        printf("Computing shortest paths...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double t_start = MPI_Wtime();

    floydWarshall_mpi(dist_local, path_local, numNodes,
                      local_start, local_cols, rank, num_procs);

    const double t_end = MPI_Wtime();
    const double elapsed_ms = (t_end - t_start) * 1000.0;

    // --- Gather results back to rank 0 ---
    MPI_Gatherv(dist_local.data(), scounts[rank], MPI_UNSIGNED,
                rank == 0 ? dist_full.data() : nullptr,
                scounts.data(), sdispls.data(), MPI_UNSIGNED,
                0, MPI_COMM_WORLD);
    MPI_Gatherv(path_local.data(), scounts[rank], MPI_UNSIGNED,
                rank == 0 ? path_full.data() : nullptr,
                scounts.data(), sdispls.data(), MPI_UNSIGNED,
                0, MPI_COMM_WORLD);

    // --- Output and validation on rank 0 ---
    if (rank == 0) {
        printf("Computation time: %.0f ms\n", elapsed_ms);

        // Calculate operations per second
        double ops = static_cast<double>(numNodes) * numNodes * numNodes;
        double gflops = ops / (elapsed_ms / 1000.0) / 1.0e9;
        printf("Performance: %.3f GOPS\n", gflops);

        if (printResults) {
            print_results_int(dist_full, "DistanceMatrix");
        }

        if (validate) {
            printf("Validating result...\n");
            if (validateResult(dist_full, numNodes)) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
    }

    MPI_Finalize();
    return 0;
}
