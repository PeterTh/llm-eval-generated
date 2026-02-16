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

// Index calculation for flattened 2D array (column-major layout used by original)
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
    int rank = 0, world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (all ranks to keep argc/argv consistent)
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
    }

    // Prepare distribution: rows are distributed across ranks (each rank holds a set of rows)
    std::vector<int> rows_per_rank(world_size);
    std::vector<int> row_start(world_size);
    int base = static_cast<int>(numNodes) / world_size;
    int rem = static_cast<int>(numNodes) % world_size;
    for (int r = 0; r < world_size; ++r) {
        rows_per_rank[r] = base + (r < rem ? 1 : 0);
        row_start[r] = (r == 0) ? 0 : (row_start[r - 1] + rows_per_rank[r - 1]);
    }
    int local_rows = rows_per_rank[rank];

    // Root initializes full matrices and scatters contiguous rows (row-major per row)
    std::vector<unsigned int> full_dist;
    std::vector<unsigned int> full_path;
    std::vector<unsigned int> sendbuf_dist; // packed rows
    std::vector<unsigned int> sendbuf_path;
    std::vector<int> sendcounts(world_size);
    std::vector<int> displs(world_size);

    if (rank == 0) {
        full_dist.resize(numNodes * numNodes);
        full_path.resize(numNodes * numNodes);
        initializeDistanceMatrix(full_dist, numNodes, 1, MAX_DISTANCE);
        initializePathMatrix(full_path, numNodes);

        // Pack rows into sendbuf (row-major rows of length numNodes)
        sendbuf_dist.resize(numNodes * numNodes);
        sendbuf_path.resize(numNodes * numNodes);

        int pos = 0;
        for (int r = 0; r < (int)numNodes; ++r) {
            for (size_t c = 0; c < numNodes; ++c) {
                // original storage is column-major idx2(row, col)
                sendbuf_dist[pos] = full_dist[idx2(r, c, numNodes)];
                sendbuf_path[pos] = full_path[idx2(r, c, numNodes)];
                ++pos;
            }
        }

        // prepare sendcounts/displs in units of unsigned ints
        int offset = 0;
        for (int r = 0; r < world_size; ++r) {
            sendcounts[r] = rows_per_rank[r] * static_cast<int>(numNodes);
            displs[r] = offset;
            offset += sendcounts[r];
        }
    }

    // Each rank receives its block of rows in row-major contiguous buffer
    std::vector<unsigned int> local_dist(static_cast<size_t>(local_rows) * numNodes);
    std::vector<unsigned int> local_path(static_cast<size_t>(local_rows) * numNodes);

    MPI_Scatterv(rank == 0 ? sendbuf_dist.data() : nullptr, rank == 0 ? sendcounts.data() : nullptr,
                 rank == 0 ? displs.data() : nullptr, MPI_UNSIGNED,
                 local_dist.data(), local_rows * static_cast<int>(numNodes), MPI_UNSIGNED,
                 0, MPI_COMM_WORLD);

    MPI_Scatterv(rank == 0 ? sendbuf_path.data() : nullptr, rank == 0 ? sendcounts.data() : nullptr,
                 rank == 0 ? displs.data() : nullptr, MPI_UNSIGNED,
                 local_path.data(), local_rows * static_cast<int>(numNodes), MPI_UNSIGNED,
                 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    // Buffer to hold row k (row-major across columns)
    std::vector<unsigned int> row_k(static_cast<size_t>(numNodes));

    // Main distributed Floyd-Warshall: each rank updates its owned rows
    for (int k = 0; k < static_cast<int>(numNodes); ++k) {
        // determine owner of row k
        int owner = 0;
        // simple search (cheap) to find owner
        for (int r = 0; r < world_size; ++r) {
            if (k >= row_start[r] && k < row_start[r] + rows_per_rank[r]) {
                owner = r;
                break;
            }
        }

        if (rank == owner) {
            int local_k = k - row_start[rank];
            // copy local row k into row_k
            for (int col = 0; col < static_cast<int>(numNodes); ++col) {
                row_k[col] = local_dist[static_cast<size_t>(local_k) * numNodes + col];
            }
        }

        // broadcast row_k from owner to all ranks
        MPI_Bcast(row_k.data(), static_cast<int>(numNodes), MPI_UNSIGNED, owner, MPI_COMM_WORLD);

        // each rank updates its local rows
        for (int local_r = 0; local_r < local_rows; ++local_r) {
            // global row index j = row_start[rank] + local_r
            for (int i = 0; i < static_cast<int>(numNodes); ++i) {
                unsigned int distIJ = local_dist[static_cast<size_t>(local_r) * numNodes + i];
                unsigned int distIK = row_k[i]; // dist[k][i]
                unsigned int distKJ = local_dist[static_cast<size_t>(local_r) * numNodes + k]; // dist[j][k]

                if (distIK < INF && distKJ < INF) {
                    unsigned int newDist = distIK + distKJ;
                    if (newDist < distIJ) {
                        local_dist[static_cast<size_t>(local_r) * numNodes + i] = newDist;
                        // store intermediate node (global k)
                        local_path[static_cast<size_t>(local_r) * numNodes + i] = static_cast<unsigned int>(k);
                    }
                }
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();
    double local_elapsed = end_time - start_time;
    double max_elapsed = 0.0;
    MPI_Reduce(&local_elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // gather results back to root
    std::vector<unsigned int> recvbuf_dist;
    std::vector<unsigned int> recvbuf_path;
    if (rank == 0) {
        recvbuf_dist.resize(numNodes * numNodes);
        recvbuf_path.resize(numNodes * numNodes);
    }

    MPI_Gatherv(local_dist.data(), local_rows * static_cast<int>(numNodes), MPI_UNSIGNED,
                rank == 0 ? recvbuf_dist.data() : nullptr, rank == 0 ? sendcounts.data() : nullptr,
                rank == 0 ? displs.data() : nullptr, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    MPI_Gatherv(local_path.data(), local_rows * static_cast<int>(numNodes), MPI_UNSIGNED,
                rank == 0 ? recvbuf_path.data() : nullptr, rank == 0 ? sendcounts.data() : nullptr,
                rank == 0 ? displs.data() : nullptr, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // unpack recvbuf (row-major rows) back into original column-major full_dist/full_path
        int pos = 0;
        for (int r = 0; r < static_cast<int>(numNodes); ++r) {
            for (int c = 0; c < static_cast<int>(numNodes); ++c) {
                full_dist[idx2(r, c, numNodes)] = recvbuf_dist[pos];
                full_path[idx2(r, c, numNodes)] = recvbuf_path[pos];
                ++pos;
            }
        }

        // report timing
        long ms = static_cast<long>(max_elapsed * 1000.0);
        printf("Computation time: %ld ms\n", ms);

        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / (max_elapsed) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);

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
    }

    MPI_Finalize();
    return 0;
}
