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

// Row-major index: element at row i, column j
inline constexpr size_t idx(const size_t i, const size_t j, const size_t n) noexcept {
    return i * n + j;
}

void initializeDistanceMatrixRowMajor(std::vector<unsigned int>& dist, const size_t numNodes,
                                      const unsigned int rangeMin, const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;

    for (size_t i = 0; i < numNodes; ++i) {
        for (size_t j = 0; j < numNodes; ++j) {
            dist[idx(i, j, numNodes)] = rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
        }
    }

    // Set diagonal to zero
    for (size_t i = 0; i < numNodes; ++i) {
        dist[idx(i, i, numNodes)] = 0;
    }
}

void initializePathMatrixRowMajor(std::vector<unsigned int>& path, const size_t numNodes) {
    for (size_t i = 0; i < numNodes; ++i) {
        for (size_t j = 0; j < numNodes; ++j) {
            path[idx(i, j, numNodes)] = j;
        }
        path[idx(i, i, numNodes)] = i;
    }
}

bool validateResultRowMajor(const std::vector<unsigned int>& dist, const size_t numNodes) {
    // 1. Diagonal should be zero
    for (size_t i = 0; i < numNodes; ++i) {
        if (dist[idx(i, i, numNodes)] != 0) {
            printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", i, i);
            return false;
        }
    }

    // 2. Triangle inequality check on a sample
    for (size_t i = 0; i < std::min(numNodes, static_cast<size_t>(10)); ++i) {
        for (size_t j = 0; j < std::min(numNodes, static_cast<size_t>(10)); ++j) {
            for (size_t k = 0; k < numNodes; ++k) {
                const unsigned int distIJ = dist[idx(i, j, numNodes)];
                const unsigned int distIK = dist[idx(i, k, numNodes)];
                const unsigned int distKJ = dist[idx(k, j, numNodes)];
                if (distIK < INF && distKJ < INF) {
                    if (distIK + distKJ < distIJ) {
                        printf("Validation failed: triangle inequality violated at [%zu,%zu,%zu]\n", i, j, k);
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
    // Initialize MPI (unconditionally use MPI)
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (only on rank 0 and broadcast)
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
            }
        }
    }

    // Broadcast settings
    MPI_Bcast(&numNodes, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Compute row distribution
    const size_t rows_per = numNodes / size;
    const size_t rem = numNodes % size;
    const size_t local_rows = rows_per + (static_cast<size_t>(rank) < rem ? 1 : 0);
    const size_t row_start = static_cast<size_t>(rank) * rows_per + std::min(static_cast<size_t>(rank), rem);

    // Prepare sendcounts and displacements for scatter/gather (in elements)
    std::vector<int> sendcounts(size);
    std::vector<int> displs(size);
    for (int r = 0; r < size; ++r) {
        size_t r_rows = rows_per + (static_cast<size_t>(r) < rem ? 1 : 0);
        sendcounts[r] = static_cast<int>(r_rows * numNodes);
        // displacement in elements
        size_t disp = static_cast<size_t>(r) * rows_per * numNodes + std::min(static_cast<size_t>(r), rem) * numNodes;
        displs[r] = static_cast<int>(disp);
    }

    // Rank 0 initializes full matrix then scatters
    std::vector<unsigned int> full_dist;
    std::vector<unsigned int> full_path;
    if (rank == 0) {
        full_dist.resize(numNodes * numNodes);
        full_path.resize(numNodes * numNodes);
        initializeDistanceMatrixRowMajor(full_dist, numNodes, 1, MAX_DISTANCE);
        initializePathMatrixRowMajor(full_path, numNodes);
    }

    // Allocate local storage for rows (row-major: local_rows x numNodes)
    std::vector<unsigned int> local_dist(local_rows * numNodes);
    std::vector<unsigned int> local_path(local_rows * numNodes);

    // Scatter full_dist and full_path to local blocks
    MPI_Scatterv(rank == 0 ? full_dist.data() : nullptr, sendcounts.data(), displs.data(), MPI_UNSIGNED,
                 local_dist.data(), static_cast<int>(local_rows * numNodes), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? full_path.data() : nullptr, sendcounts.data(), displs.data(), MPI_UNSIGNED,
                 local_path.data(), static_cast<int>(local_rows * numNodes), MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    // Synchronize and time
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    // Buffer for k-th row
    std::vector<unsigned int> row_k(numNodes);

    // Main distributed Floyd-Warshall: for k=0..n-1, broadcast row k (owner supplies), then update local rows
    for (size_t k = 0; k < numNodes; ++k) {
        // Determine owner of row k
        size_t owner;
        size_t base = rows_per + 1; // rows for ranks < rem
        size_t threshold = base * rem;
        if (k < threshold) {
            owner = k / base;
        } else {
            owner = rem + (k - threshold) / rows_per;
        }

        if (static_cast<size_t>(rank) == owner) {
            size_t local_k = k - row_start;
            // copy owner's k-th row into row_k
            std::memcpy(row_k.data(), &local_dist[local_k * numNodes], numNodes * sizeof(unsigned int));
        }

        // Broadcast k-th row to all processes
        MPI_Bcast(row_k.data(), static_cast<int>(numNodes), MPI_UNSIGNED, static_cast<int>(owner), MPI_COMM_WORLD);

        // Update local rows
        for (size_t i_local = 0; i_local < local_rows; ++i_local) {
            unsigned int dist_ik = local_dist[i_local * numNodes + k];
            if (dist_ik >= INF) continue; // skip if infinite
            unsigned int* row_ptr = &local_dist[i_local * numNodes];
            unsigned int* path_row_ptr = &local_path[i_local * numNodes];

            for (size_t j = 0; j < numNodes; ++j) {
                unsigned int newDist = dist_ik + row_k[j];
                if (newDist < row_ptr[j]) {
                    row_ptr[j] = newDist;
                    path_row_ptr[j] = path_row_ptr[k];
                }
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();
    double local_elapsed = t1 - t0;
    double elapsed_max = 0.0;
    MPI_Reduce(&local_elapsed, &elapsed_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather results back to rank 0
    if (rank == 0) {
        // reuse full_dist and full_path vectors
    }
    MPI_Gatherv(local_dist.data(), static_cast<int>(local_rows * numNodes), MPI_UNSIGNED,
                rank == 0 ? full_dist.data() : nullptr, sendcounts.data(), displs.data(), MPI_UNSIGNED,
                0, MPI_COMM_WORLD);
    MPI_Gatherv(local_path.data(), static_cast<int>(local_rows * numNodes), MPI_UNSIGNED,
                rank == 0 ? full_path.data() : nullptr, sendcounts.data(), displs.data(), MPI_UNSIGNED,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(elapsed_max * 1000.0));
        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / (elapsed_max) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);

        // Print results for external validation (must match original column-major flattening expected)
        if (printResults) {
            // Convert row-major full_dist to column-major ordering used by original implementation
            std::vector<unsigned int> out_col_major(numNodes * numNodes);
            for (size_t i = 0; i < numNodes; ++i) {
                for (size_t j = 0; j < numNodes; ++j) {
                    // original flatten used index = j * n + i
                    out_col_major[j * numNodes + i] = full_dist[i * numNodes + j];
                }
            }
            print_results_int(out_col_major, "DistanceMatrix");
        }

        // Validation
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResultRowMajor(full_dist, numNodes);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
    }

    MPI_Finalize();
    return 0;
}
