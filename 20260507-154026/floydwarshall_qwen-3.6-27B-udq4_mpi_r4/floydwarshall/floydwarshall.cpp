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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;

    // Parse arguments on rank 0
    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                numNodes = (size_t)atoi(argv[++i]);
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = true;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = true;
            } else if (strcmp(argv[i], "-h") == 0) {
                printf("Usage: %s [options]\n", argv[0]);
                printf("Options:\n");
                printf("  -n <num>     Number of nodes in the graph (default: 512)\n");
                printf("  -v           Enable validation\n");
                printf("  -r           Print results for external validation\n");
                printf("  -h           Show this help message\n");
                MPI_Finalize();
                return 0;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printf("Usage: %s [options]\n", argv[0]);
                MPI_Finalize();
                return 1;
            }
        }

        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Number of MPI processes: %d\n", num_procs);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Broadcast parameters to all processes
    MPI_Bcast(&numNodes, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

    // Block-row distribution: process p owns rows [local_start, local_start + local_rows)
    const size_t base_rows = numNodes / (size_t)num_procs;
    const size_t remainder = numNodes % (size_t)num_procs;
    const size_t local_rows = base_rows + (rank < (int)remainder ? 1 : 0);
    const size_t local_start = base_rows * (size_t)rank + std::min((size_t)rank, remainder);

    // Local row-major storage: local[ri * numNodes + j] = dist[local_start + ri][j]
    std::vector<unsigned int> local_dist(local_rows * numNodes);
    std::vector<unsigned int> local_path(local_rows * numNodes);

    // Scatter/gather counts and displacements
    std::vector<int> counts(num_procs);
    std::vector<int> displs(num_procs);
    {
        size_t offset = 0;
        for (int p = 0; p < num_procs; ++p) {
            const size_t r = base_rows + (p < (int)remainder ? 1 : 0);
            counts[p] = static_cast<int>(r * numNodes);
            displs[p] = static_cast<int>(offset * numNodes);
            offset += r;
        }
    }

    // Initialize on rank 0, convert to row-major, scatter to all processes
    if (rank == 0) {
        printf("Initializing graph...\n");

        std::vector<unsigned int> full_dist(numNodes * numNodes);
        std::vector<unsigned int> full_path(numNodes * numNodes);
        initializeDistanceMatrix(full_dist, numNodes, 1, MAX_DISTANCE);
        initializePathMatrix(full_path, numNodes);

        // Convert column-major -> row-major for scatter
        std::vector<unsigned int> send_dist(numNodes * numNodes);
        std::vector<unsigned int> send_path(numNodes * numNodes);
        for (size_t i = 0; i < numNodes; ++i) {
            for (size_t j = 0; j < numNodes; ++j) {
                send_dist[i * numNodes + j] = full_dist[idx2(i, j, numNodes)];
                send_path[i * numNodes + j] = full_path[idx2(i, j, numNodes)];
            }
        }

        MPI_Scatterv(send_dist.data(), counts.data(), displs.data(), MPI_UNSIGNED,
                     local_dist.data(), static_cast<int>(local_rows * numNodes), MPI_UNSIGNED,
                     0, MPI_COMM_WORLD);
        MPI_Scatterv(send_path.data(), counts.data(), displs.data(), MPI_UNSIGNED,
                     local_path.data(), static_cast<int>(local_rows * numNodes), MPI_UNSIGNED,
                     0, MPI_COMM_WORLD);
    } else {
        MPI_Scatterv(nullptr, counts.data(), displs.data(), MPI_UNSIGNED,
                     local_dist.data(), static_cast<int>(local_rows * numNodes), MPI_UNSIGNED,
                     0, MPI_COMM_WORLD);
        MPI_Scatterv(nullptr, counts.data(), displs.data(), MPI_UNSIGNED,
                     local_path.data(), static_cast<int>(local_rows * numNodes), MPI_UNSIGNED,
                     0, MPI_COMM_WORLD);
    }

    // Precompute owner of each row for the FW loop
    std::vector<int> row_owner(numNodes);
    for (size_t k = 0; k < numNodes; ++k) {
        if (k < remainder * (base_rows + 1)) {
            row_owner[k] = static_cast<int>(k / (base_rows + 1));
        } else {
            row_owner[k] = static_cast<int>(remainder + (k - remainder * (base_rows + 1)) / base_rows);
        }
    }

    // Parallel Floyd-Warshall: block-row distribution with row-k broadcast
    if (rank == 0) printf("Computing shortest paths...\n");

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<unsigned int> row_k(numNodes);

    for (size_t k = 0; k < numNodes; ++k) {
        // Owner extracts row k; all others receive via broadcast
        if (k >= local_start && k < local_start + local_rows) {
            const unsigned int* src = local_dist.data() + (k - local_start) * numNodes;
            std::copy(src, src + numNodes, row_k.begin());
        }

        MPI_Bcast(row_k.data(), static_cast<int>(numNodes), MPI_UNSIGNED,
                  row_owner[k], MPI_COMM_WORLD);

        // Update local rows: dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
        for (size_t ri = 0; ri < local_rows; ++ri) {
            const unsigned int dist_ik = local_dist[ri * numNodes + k];
            unsigned int* row_i = local_dist.data() + ri * numNodes;
            unsigned int* path_i = local_path.data() + ri * numNodes;

            for (size_t j = 0; j < numNodes; ++j) {
                const unsigned int new_dist = dist_ik + row_k[j];
                if (new_dist < row_i[j]) {
                    row_i[j] = new_dist;
                    path_i[j] = static_cast<unsigned int>(k);
                }
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long long local_duration_ms = duration.count();
    long long global_duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &global_duration_ms, 1, MPI_LONG_LONG_INT, MPI_MAX,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %lld ms\n", global_duration_ms);
        double ops = static_cast<double>(numNodes) * numNodes * numNodes;
        double gflops = ops / (global_duration_ms / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);
    }

    // Gather results back to rank 0, convert back to column-major
    if (rank == 0) {
        std::vector<unsigned int> recv_dist(numNodes * numNodes);
        std::vector<unsigned int> recv_path(numNodes * numNodes);

        MPI_Gatherv(local_dist.data(), static_cast<int>(local_rows * numNodes), MPI_UNSIGNED,
                    recv_dist.data(), counts.data(), displs.data(), MPI_UNSIGNED,
                    0, MPI_COMM_WORLD);
        MPI_Gatherv(local_path.data(), static_cast<int>(local_rows * numNodes), MPI_UNSIGNED,
                    recv_path.data(), counts.data(), displs.data(), MPI_UNSIGNED,
                    0, MPI_COMM_WORLD);

        // Convert row-major -> column-major
        std::vector<unsigned int> full_dist(numNodes * numNodes);
        for (size_t i = 0; i < numNodes; ++i) {
            for (size_t j = 0; j < numNodes; ++j) {
                full_dist[idx2(i, j, numNodes)] = recv_dist[i * numNodes + j];
            }
        }

        if (printResults) {
            print_results_int(full_dist, "DistanceMatrix");
        }

        if (validate) {
            printf("Validating result...\n");
            if (validateResult(full_dist, numNodes)) {
                printf("Validation: PASSED\n");
                MPI_Finalize();
                return 0;
            } else {
                printf("Validation: FAILED\n");
                MPI_Finalize();
                return 1;
            }
        }
    } else {
        MPI_Gatherv(local_dist.data(), static_cast<int>(local_rows * numNodes), MPI_UNSIGNED,
                    nullptr, nullptr, nullptr, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
        MPI_Gatherv(local_path.data(), static_cast<int>(local_rows * numNodes), MPI_UNSIGNED,
                    nullptr, nullptr, nullptr, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
