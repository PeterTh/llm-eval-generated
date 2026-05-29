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

    int rank, numRanks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

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
        printf("Number of MPI ranks: %d\n", numRanks);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // ----------------------------------------------------------------
    // 1D row-block distribution: each rank owns a contiguous block of
    // rows of the N×N distance matrix.  Local storage is row-major:
    //   local[li * numNodes + j]  ==  dist[my_start + li][j]
    // ----------------------------------------------------------------
    size_t base_rows = numNodes / numRanks;
    size_t remainder = numNodes % numRanks;
    size_t my_rows = base_rows + (rank < (int)remainder ? 1 : 0);
    size_t my_start = 0;
    for (int r = 0; r < rank; ++r) {
        my_start += base_rows + (r < (int)remainder ? 1 : 0);
    }

    // Precompute which rank owns each row (needed for Bcast root).
    std::vector<int> row_to_rank(numNodes);
    {
        size_t offset = 0;
        for (int r = 0; r < numRanks; ++r) {
            size_t r_rows = base_rows + (r < (int)remainder ? 1 : 0);
            for (size_t i = 0; i < r_rows; ++i) {
                row_to_rank[offset + i] = r;
            }
            offset += r_rows;
        }
    }

    std::vector<unsigned int> dist_local(my_rows * numNodes);
    std::vector<unsigned int> path_local(my_rows * numNodes);

    // ----------------------------------------------------------------
    // Initialization — rank 0 builds the full column-major matrices,
    // then scatters row blocks (converted to row-major) to all ranks.
    // ----------------------------------------------------------------
    if (rank == 0) {
        printf("Initializing graph...\n");

        std::vector<unsigned int> full_dist(numNodes * numNodes);
        std::vector<unsigned int> full_path(numNodes * numNodes);
        initializeDistanceMatrix(full_dist, numNodes, 1, MAX_DISTANCE);
        initializePathMatrix(full_path, numNodes);

        size_t offset = 0;
        for (int r = 0; r < numRanks; ++r) {
            size_t r_rows = base_rows + (r < (int)remainder ? 1 : 0);

            if (r == 0) {
                for (size_t li = 0; li < r_rows; ++li) {
                    size_t i = offset + li;
                    for (size_t j = 0; j < numNodes; ++j) {
                        dist_local[li * numNodes + j] = full_dist[idx2(j, i, numNodes)];
                        path_local[li * numNodes + j] = full_path[idx2(j, i, numNodes)];
                    }
                }
            } else {
                std::vector<unsigned int> send_dist(r_rows * numNodes);
                std::vector<unsigned int> send_path(r_rows * numNodes);
                for (size_t li = 0; li < r_rows; ++li) {
                    size_t i = offset + li;
                    for (size_t j = 0; j < numNodes; ++j) {
                        send_dist[li * numNodes + j] = full_dist[idx2(j, i, numNodes)];
                        send_path[li * numNodes + j] = full_path[idx2(j, i, numNodes)];
                    }
                }
                MPI_Send(send_dist.data(), static_cast<int>(r_rows * numNodes),
                         MPI_UNSIGNED, r, 10, MPI_COMM_WORLD);
                MPI_Send(send_path.data(), static_cast<int>(r_rows * numNodes),
                         MPI_UNSIGNED, r, 11, MPI_COMM_WORLD);
            }
            offset += r_rows;
        }
    } else {
        MPI_Recv(dist_local.data(), static_cast<int>(my_rows * numNodes),
                 MPI_UNSIGNED, 0, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(path_local.data(), static_cast<int>(my_rows * numNodes),
                 MPI_UNSIGNED, 0, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // ----------------------------------------------------------------
    // Parallel Floyd-Warshall
    //   At each k-iteration the k-th row is broadcast to every rank;
    //   each rank then updates its local rows independently.
    // ----------------------------------------------------------------
    if (rank == 0) printf("Computing shortest paths...\n");

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<unsigned int> dist_pivot(numNodes);

    for (size_t k = 0; k < numNodes; ++k) {
        int rank_k = row_to_rank[k];

        // Rank that owns row k copies it into the pivot buffer.
        if (rank == rank_k) {
            size_t local_k = k - my_start;
            for (size_t j = 0; j < numNodes; ++j) {
                dist_pivot[j] = dist_local[local_k * numNodes + j];
            }
        }

        // Broadcast the k-th row to all ranks.
        MPI_Bcast(dist_pivot.data(), static_cast<int>(numNodes),
                  MPI_UNSIGNED, rank_k, MPI_COMM_WORLD);

        // Each rank updates its local rows.
        for (size_t li = 0; li < my_rows; ++li) {
            const unsigned int dist_ik = dist_local[li * numNodes + k];

            for (size_t j = 0; j < numNodes; ++j) {
                const unsigned int newDist = dist_ik + dist_pivot[j];
                if (newDist < dist_local[li * numNodes + j]) {
                    dist_local[li * numNodes + j] = newDist;
                    path_local[li * numNodes + j] = static_cast<unsigned int>(k);
                }
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // ----------------------------------------------------------------
    // Gather results back to rank 0 (row-major → column-major).
    // ----------------------------------------------------------------
    std::vector<int> recv_counts(numRanks);
    std::vector<int> recv_disps(numRanks);
    {
        size_t offset = 0;
        for (int r = 0; r < numRanks; ++r) {
            size_t r_rows = base_rows + (r < (int)remainder ? 1 : 0);
            recv_counts[r] = static_cast<int>(r_rows * numNodes);
            recv_disps[r] = static_cast<int>(offset);
            offset += r_rows * numNodes;
        }
    }

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());

        double ops = static_cast<double>(numNodes) * numNodes * numNodes;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);

        // Gather all row blocks (row-major) from every rank.
        std::vector<unsigned int> full_dist_rm(numNodes * numNodes);
        MPI_Gatherv(dist_local.data(), static_cast<int>(my_rows * numNodes), MPI_UNSIGNED,
                    full_dist_rm.data(), recv_counts.data(), recv_disps.data(), MPI_UNSIGNED,
                    0, MPI_COMM_WORLD);

        // Convert from row-major back to column-major (original layout).
        std::vector<unsigned int> full_dist(numNodes * numNodes);
        for (size_t i = 0; i < numNodes; ++i) {
            for (size_t j = 0; j < numNodes; ++j) {
                full_dist[idx2(j, i, numNodes)] = full_dist_rm[i * numNodes + j];
            }
        }

        if (printResults) {
            print_results_int(full_dist, "DistanceMatrix");
        }

        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(full_dist, numNodes);
            if (valid) {
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
        // Non-root ranks participate in the collective gather.
        MPI_Gatherv(dist_local.data(), static_cast<int>(my_rows * numNodes), MPI_UNSIGNED,
                    nullptr, recv_counts.data(), recv_disps.data(), MPI_UNSIGNED,
                    0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
