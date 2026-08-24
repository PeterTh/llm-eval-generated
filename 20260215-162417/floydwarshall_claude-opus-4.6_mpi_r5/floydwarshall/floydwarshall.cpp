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
    
    // Compute row distribution across MPI ranks
    const size_t base_rows = numNodes / static_cast<size_t>(nprocs);
    const size_t rem = numNodes % static_cast<size_t>(nprocs);

    std::vector<int> sendcounts(nprocs), displs(nprocs);
    std::vector<size_t> row_counts(nprocs), row_starts(nprocs);
    for (int p = 0; p < nprocs; ++p) {
        row_counts[p] = base_rows + (static_cast<size_t>(p) < rem ? 1 : 0);
        row_starts[p] = static_cast<size_t>(p) * base_rows + std::min(static_cast<size_t>(p), rem);
        sendcounts[p] = static_cast<int>(row_counts[p] * numNodes);
        displs[p] = static_cast<int>(row_starts[p] * numNodes);
    }

    const size_t local_rows = row_counts[rank];

    // Rank 0 initializes full matrices then scatters
    std::vector<unsigned int> full_dist, full_path;
    if (rank == 0) {
        full_dist.resize(numNodes * numNodes);
        full_path.resize(numNodes * numNodes);
        printf("Initializing graph...\n");
        initializeDistanceMatrix(full_dist, numNodes, 1, MAX_DISTANCE);
        initializePathMatrix(full_path, numNodes);
    }

    // Local row portions
    std::vector<unsigned int> local_dist(local_rows * numNodes);
    std::vector<unsigned int> local_path(local_rows * numNodes);

    MPI_Scatterv(rank == 0 ? full_dist.data() : nullptr,
                 sendcounts.data(), displs.data(), MPI_UNSIGNED,
                 local_dist.data(), sendcounts[rank], MPI_UNSIGNED,
                 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? full_path.data() : nullptr,
                 sendcounts.data(), displs.data(), MPI_UNSIGNED,
                 local_path.data(), sendcounts[rank], MPI_UNSIGNED,
                 0, MPI_COMM_WORLD);

    // Free full matrices to save memory
    if (rank == 0) {
        std::vector<unsigned int>().swap(full_dist);
        std::vector<unsigned int>().swap(full_path);
    }

    if (rank == 0) printf("Computing shortest paths...\n");

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    // Precompute owner and local index for each row k
    std::vector<int> k_owner(numNodes);
    std::vector<size_t> k_local(numNodes);
    for (size_t k = 0; k < numNodes; ++k) {
        if (k < rem * (base_rows + 1)) {
            k_owner[k] = static_cast<int>(k / (base_rows + 1));
        } else {
            k_owner[k] = static_cast<int>(rem + (k - rem * (base_rows + 1)) / base_rows);
        }
        k_local[k] = k - row_starts[k_owner[k]];
    }

    // MPI Floyd-Warshall: broadcast row k, update local rows
    std::vector<unsigned int> k_row(numNodes);

    for (size_t k = 0; k < numNodes; ++k) {
        const int owner = k_owner[k];

        // Owner copies row k into broadcast buffer
        if (rank == owner) {
            const size_t lk = k_local[k];
            std::memcpy(k_row.data(), &local_dist[lk * numNodes],
                        numNodes * sizeof(unsigned int));
        }

        MPI_Bcast(k_row.data(), static_cast<int>(numNodes), MPI_UNSIGNED,
                  owner, MPI_COMM_WORLD);

        // Update local rows
        for (size_t li = 0; li < local_rows; ++li) {
            const unsigned int distIK = local_dist[li * numNodes + k];
            unsigned int* __restrict__ dist_row = &local_dist[li * numNodes];
            unsigned int* __restrict__ path_row = &local_path[li * numNodes];
            for (size_t j = 0; j < numNodes; ++j) {
                const unsigned int newDist = distIK + k_row[j];
                if (newDist < dist_row[j]) {
                    dist_row[j] = newDist;
                    path_row[j] = static_cast<unsigned int>(k);
                }
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    const long long local_duration_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long long duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &duration_ms, 1, MPI_LONG_LONG_INT, MPI_MAX,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %lld ms\n", duration_ms);
        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / (duration_ms / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);
    }

    // Gather dist matrix on rank 0 for validation/output
    if (rank == 0) {
        full_dist.resize(numNodes * numNodes);
    }
    MPI_Gatherv(local_dist.data(), sendcounts[rank], MPI_UNSIGNED,
                rank == 0 ? full_dist.data() : nullptr,
                sendcounts.data(), displs.data(), MPI_UNSIGNED,
                0, MPI_COMM_WORLD);

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
                MPI_Finalize();
                return 1;
            }
        }
    }

    MPI_Finalize();
    return 0;
}
