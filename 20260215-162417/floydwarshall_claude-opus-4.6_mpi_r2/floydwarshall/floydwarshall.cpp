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

// MPI-parallel Floyd-Warshall: each rank owns a contiguous block of rows (i-dimension).
// For each k, the owner of row k broadcasts it, then all ranks update their local rows.
void floydWarshall(std::vector<unsigned int>& local_dist,
                   std::vector<unsigned int>& local_path,
                   const size_t numNodes, const size_t local_n,
                   const size_t base_rows, const size_t remainder,
                   int rank, int /*num_procs*/) {
    std::vector<unsigned int> k_row(numNodes);

    for (size_t k = 0; k < numNodes; ++k) {
        // Determine which rank owns row k and its local index
        int owner;
        size_t local_k;
        if (base_rows == 0 || k < remainder * (base_rows + 1)) {
            owner = base_rows == 0 ? static_cast<int>(k) : static_cast<int>(k / (base_rows + 1));
            local_k = base_rows == 0 ? 0 : k % (base_rows + 1);
        } else {
            size_t adjusted = k - remainder * (base_rows + 1);
            owner = static_cast<int>(remainder + adjusted / base_rows);
            local_k = adjusted % base_rows;
        }

        // Owner extracts row k, then broadcast to all ranks
        if (rank == owner) {
            const size_t base = local_k * numNodes;
            for (size_t j = 0; j < numNodes; ++j) {
                k_row[j] = local_dist[base + j];
            }
        }
        MPI_Bcast(k_row.data(), static_cast<int>(numNodes), MPI_UNSIGNED, owner, MPI_COMM_WORLD);

        // Update local rows
        for (size_t li = 0; li < local_n; ++li) {
            const size_t base = li * numNodes;
            const unsigned int dist_ik = local_dist[base + k];
            unsigned int* __restrict__ d_row = &local_dist[base];
            unsigned int* __restrict__ p_row = &local_path[base];
            const unsigned int* __restrict__ kr = k_row.data();

            for (size_t j = 0; j < numNodes; ++j) {
                const unsigned int new_dist = dist_ik + kr[j];
                if (new_dist < d_row[j]) {
                    d_row[j] = new_dist;
                    p_row[j] = static_cast<unsigned int>(k);
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
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

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
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("MPI processes: %d\n", num_procs);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Compute row distribution across ranks
    const size_t base_rows = numNodes / static_cast<size_t>(num_procs);
    const size_t remainder = numNodes % static_cast<size_t>(num_procs);
    const size_t local_n = base_rows + (static_cast<size_t>(rank) < remainder ? 1 : 0);
    const size_t row_start = static_cast<size_t>(rank) * base_rows + std::min(static_cast<size_t>(rank), remainder);

    // Scatter/gather counts and displacements
    std::vector<int> sendcounts(num_procs), displs(num_procs);
    for (int r = 0; r < num_procs; ++r) {
        size_t rn = base_rows + (static_cast<size_t>(r) < remainder ? 1 : 0);
        size_t rs = static_cast<size_t>(r) * base_rows + std::min(static_cast<size_t>(r), remainder);
        sendcounts[r] = static_cast<int>(rn * numNodes);
        displs[r] = static_cast<int>(rs * numNodes);
    }

    // Rank 0 initializes the full distance matrix, then scatter
    std::vector<unsigned int> full_dist;
    if (rank == 0) {
        full_dist.resize(numNodes * numNodes);
        printf("Initializing graph...\n");
        initializeDistanceMatrix(full_dist, numNodes, 1, MAX_DISTANCE);
    }

    // Scatter distance matrix rows to all ranks
    std::vector<unsigned int> local_dist(local_n * numNodes);
    MPI_Scatterv(rank == 0 ? full_dist.data() : nullptr,
                 sendcounts.data(), displs.data(), MPI_UNSIGNED,
                 local_dist.data(), static_cast<int>(local_n * numNodes), MPI_UNSIGNED,
                 0, MPI_COMM_WORLD);

    if (rank == 0) { full_dist.clear(); full_dist.shrink_to_fit(); }

    // Initialize local path matrix directly (path[i*n+j] = i for all i,j)
    std::vector<unsigned int> local_path(local_n * numNodes);
    for (size_t li = 0; li < local_n; ++li) {
        const unsigned int gi = static_cast<unsigned int>(row_start + li);
        for (size_t j = 0; j < numNodes; ++j) {
            local_path[li * numNodes + j] = gi;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) printf("Computing shortest paths...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    floydWarshall(local_dist, local_path, numNodes, local_n, base_rows, remainder, rank, num_procs);
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);
    }
    
    // Gather results to rank 0 for output/validation
    if (printResults || validate) {
        if (rank == 0) full_dist.resize(numNodes * numNodes);
        MPI_Gatherv(local_dist.data(), static_cast<int>(local_n * numNodes), MPI_UNSIGNED,
                     rank == 0 ? full_dist.data() : nullptr,
                     sendcounts.data(), displs.data(), MPI_UNSIGNED,
                     0, MPI_COMM_WORLD);
    }
    
    int ret = 0;
    if (rank == 0) {
        if (printResults) {
            print_results_int(full_dist, "DistanceMatrix");
        }
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(full_dist, numNodes);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                ret = 1;
            }
        }
    }
    
    MPI_Finalize();
    return ret;
}
