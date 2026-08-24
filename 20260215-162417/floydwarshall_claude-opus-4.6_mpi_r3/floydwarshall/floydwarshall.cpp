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
    
    // All ranks parse command line arguments
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

    // Column-block distribution: columns are contiguous in memory (column-major layout)
    const size_t base_cols = numNodes / static_cast<size_t>(nprocs);
    const size_t extra = numNodes % static_cast<size_t>(nprocs);

    std::vector<int> sendcounts(nprocs), displs(nprocs);
    std::vector<size_t> col_offsets(nprocs), col_counts(nprocs);

    for (int r = 0; r < nprocs; ++r) {
        col_counts[r] = base_cols + (static_cast<size_t>(r) < extra ? 1 : 0);
        col_offsets[r] = (r == 0) ? 0 : col_offsets[r - 1] + col_counts[r - 1];
        sendcounts[r] = static_cast<int>(col_counts[r] * numNodes);
        displs[r] = static_cast<int>(col_offsets[r] * numNodes);
    }

    const size_t local_ncols = col_counts[rank];

    // Precompute owner of each column for fast lookup
    std::vector<int> col_owner(numNodes);
    for (int r = 0; r < nprocs; ++r) {
        for (size_t c = col_offsets[r]; c < col_offsets[r] + col_counts[r]; ++c) {
            col_owner[c] = r;
        }
    }

    // Rank 0 initializes full matrices
    std::vector<unsigned int> dist, path;
    if (rank == 0) {
        dist.resize(numNodes * numNodes);
        path.resize(numNodes * numNodes);
        printf("Initializing graph...\n");
        initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
        initializePathMatrix(path, numNodes);
    }

    // Local column blocks
    std::vector<unsigned int> local_dist(local_ncols * numNodes);
    std::vector<unsigned int> local_path(local_ncols * numNodes);

    // Scatter columns to all ranks
    MPI_Scatterv(rank == 0 ? dist.data() : nullptr, sendcounts.data(), displs.data(), MPI_UNSIGNED,
                 local_dist.data(), static_cast<int>(local_ncols * numNodes), MPI_UNSIGNED,
                 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? path.data() : nullptr, sendcounts.data(), displs.data(), MPI_UNSIGNED,
                 local_path.data(), static_cast<int>(local_ncols * numNodes), MPI_UNSIGNED,
                 0, MPI_COMM_WORLD);

    // Free full matrices on rank 0 during computation to save memory
    if (rank == 0) {
        dist.clear();
        dist.shrink_to_fit();
        path.clear();
        path.shrink_to_fit();
    }

    // Buffer for broadcast of column k
    std::vector<unsigned int> kCol(numNodes);

    if (rank == 0) printf("Computing shortest paths...\n");

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    // Floyd-Warshall: for each intermediate node k, broadcast column k,
    // then each rank updates its local columns independently
    for (size_t k = 0; k < numNodes; ++k) {
        const int owner = col_owner[k];

        // Owner copies column k into broadcast buffer
        if (rank == owner) {
            const size_t local_k = k - col_offsets[owner];
            memcpy(kCol.data(), &local_dist[local_k * numNodes], numNodes * sizeof(unsigned int));
        }

        MPI_Bcast(kCol.data(), static_cast<int>(numNodes), MPI_UNSIGNED, owner, MPI_COMM_WORLD);

        // Update local columns
        for (size_t li = 0; li < local_ncols; ++li) {
            unsigned int* __restrict__ col_d = &local_dist[li * numNodes];
            unsigned int* __restrict__ col_p = &local_path[li * numNodes];
            const unsigned int distIK = col_d[k];

            for (size_t j = 0; j < numNodes; ++j) {
                const unsigned int newDist = distIK + kCol[j];
                if (newDist < col_d[j]) {
                    col_d[j] = newDist;
                    col_p[j] = static_cast<unsigned int>(k);
                }
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start);
    long long local_duration_ms = duration.count();
    long long max_duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &max_duration_ms, 1, MPI_LONG_LONG, MPI_MAX,
               0, MPI_COMM_WORLD);

    // Gather results back to rank 0
    if (rank == 0) {
        dist.resize(numNodes * numNodes);
        path.resize(numNodes * numNodes);
    }

    MPI_Gatherv(local_dist.data(), static_cast<int>(local_ncols * numNodes), MPI_UNSIGNED,
                rank == 0 ? dist.data() : nullptr, sendcounts.data(), displs.data(), MPI_UNSIGNED,
                0, MPI_COMM_WORLD);
    MPI_Gatherv(local_path.data(), static_cast<int>(local_ncols * numNodes), MPI_UNSIGNED,
                rank == 0 ? path.data() : nullptr, sendcounts.data(), displs.data(), MPI_UNSIGNED,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %lld ms\n", max_duration_ms);

        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / (max_duration_ms / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);

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
                MPI_Finalize();
                return 1;
            }
        }
    }

    MPI_Finalize();
    return 0;
}
