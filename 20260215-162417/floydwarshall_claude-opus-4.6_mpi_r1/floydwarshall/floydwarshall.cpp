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

// Index calculation for flattened 2D array (column-major)
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
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

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
        printf("MPI processes: %d\n", nprocs);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Compute column-block distribution across MPI ranks.
    // Matrix is column-major, so columns are contiguous in memory,
    // making scatter/gather and local access cache-friendly.
    const size_t cols_per_proc = numNodes / static_cast<size_t>(nprocs);
    const size_t remainder = numNodes % static_cast<size_t>(nprocs);
    
    std::vector<int> sendcounts(nprocs), displs(nprocs);
    std::vector<size_t> col_start(nprocs), col_count(nprocs);
    for (int r = 0; r < nprocs; ++r) {
        if (static_cast<size_t>(r) < remainder) {
            col_count[r] = cols_per_proc + 1;
            col_start[r] = static_cast<size_t>(r) * (cols_per_proc + 1);
        } else {
            col_count[r] = cols_per_proc;
            col_start[r] = remainder * (cols_per_proc + 1) + (static_cast<size_t>(r) - remainder) * cols_per_proc;
        }
        sendcounts[r] = static_cast<int>(col_count[r] * numNodes);
        displs[r] = static_cast<int>(col_start[r] * numNodes);
    }
    
    const size_t my_col_count = col_count[rank];
    const int recvcount = static_cast<int>(my_col_count * numNodes);
    
    // Allocate local column blocks
    std::vector<unsigned int> local_dist(my_col_count * numNodes);
    std::vector<unsigned int> local_path(my_col_count * numNodes);
    
    // Initialize full matrices on rank 0, then scatter columns
    std::vector<unsigned int> full_dist, full_path;
    if (rank == 0) {
        printf("Initializing graph...\n");
        full_dist.resize(numNodes * numNodes);
        full_path.resize(numNodes * numNodes);
        initializeDistanceMatrix(full_dist, numNodes, 1, MAX_DISTANCE);
        initializePathMatrix(full_path, numNodes);
    }
    
    MPI_Scatterv(rank == 0 ? full_dist.data() : nullptr,
                 sendcounts.data(), displs.data(), MPI_UNSIGNED,
                 local_dist.data(), recvcount, MPI_UNSIGNED,
                 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? full_path.data() : nullptr,
                 sendcounts.data(), displs.data(), MPI_UNSIGNED,
                 local_path.data(), recvcount, MPI_UNSIGNED,
                 0, MPI_COMM_WORLD);
    
    // Free full matrices after scatter
    { std::vector<unsigned int>().swap(full_dist); }
    { std::vector<unsigned int>().swap(full_path); }
    
    if (rank == 0) printf("Computing shortest paths...\n");
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    // Floyd-Warshall: for each intermediate node k, broadcast column k
    // then each rank updates its local columns independently.
    std::vector<unsigned int> k_col(numNodes);
    const size_t threshold = remainder * (cols_per_proc + 1);
    
    for (size_t k = 0; k < numNodes; ++k) {
        // Determine which rank owns column k
        int owner;
        if (k < threshold) {
            owner = static_cast<int>(k / (cols_per_proc + 1));
        } else {
            owner = static_cast<int>(remainder + (k - threshold) / cols_per_proc);
        }
        
        // Owner copies column k into broadcast buffer
        if (rank == owner) {
            const size_t local_k = k - col_start[owner];
            std::memcpy(k_col.data(), &local_dist[local_k * numNodes],
                        numNodes * sizeof(unsigned int));
        }
        
        MPI_Bcast(k_col.data(), static_cast<int>(numNodes), MPI_UNSIGNED,
                  owner, MPI_COMM_WORLD);
        
        // Update local columns
        for (size_t li = 0; li < my_col_count; ++li) {
            unsigned int* __restrict__ col_d = &local_dist[li * numNodes];
            unsigned int* __restrict__ col_p = &local_path[li * numNodes];
            const unsigned int dist_ik = col_d[k];
            
            for (size_t j = 0; j < numNodes; ++j) {
                const unsigned int new_dist = dist_ik + k_col[j];
                if (new_dist < col_d[j]) {
                    col_d[j] = new_dist;
                    col_p[j] = static_cast<unsigned int>(k);
                }
            }
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);
    }
    
    // Gather distance matrix on rank 0 for output/validation
    if (printResults || validate) {
        std::vector<unsigned int> gathered_dist;
        if (rank == 0) {
            gathered_dist.resize(numNodes * numNodes);
        }
        
        MPI_Gatherv(local_dist.data(), recvcount, MPI_UNSIGNED,
                     rank == 0 ? gathered_dist.data() : nullptr,
                     sendcounts.data(), displs.data(), MPI_UNSIGNED,
                     0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            if (printResults) {
                print_results_int(gathered_dist, "DistanceMatrix");
            }
            
            if (validate) {
                printf("Validating result...\n");
                bool valid = validateResult(gathered_dist, numNodes);
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
