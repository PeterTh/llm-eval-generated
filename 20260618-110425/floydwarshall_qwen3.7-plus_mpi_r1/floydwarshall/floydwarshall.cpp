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

// Index calculation for flattened 2D array (column-major, matching original)
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

// Compute starting row and local row count for a given rank
inline void compute_row_range(int rank, int sz, size_t numNodes,
                              size_t& r_start, size_t& local_rows) {
    size_t base_rows = numNodes / sz;
    size_t rem = numNodes % sz;
    if ((size_t)rank < rem) {
        local_rows = base_rows + 1;
        r_start = (size_t)rank * (base_rows + 1);
    } else {
        local_rows = base_rows;
        r_start = rem * (base_rows + 1) + ((size_t)rank - rem) * base_rows;
    }
}

// Determine which rank owns a given global row
inline int compute_owner(size_t row, int sz, size_t numNodes) {
    size_t base_rows = numNodes / sz;
    size_t rem = numNodes % sz;
    if (row < rem * (base_rows + 1)) {
        return (int)(row / (base_rows + 1));
    } else {
        return (int)(rem + (row - rem * (base_rows + 1)) / base_rows);
    }
}

bool validateResult(const std::vector<unsigned int>& dist, const size_t numNodes) {
    for (size_t i = 0; i < numNodes; ++i) {
        if (dist[idx2(i, i, numNodes)] != 0) {
            printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", i, i);
            return false;
        }
    }

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

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;

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
                numNodes = 0;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                MPI_Finalize();
                return 1;
            }
        }
    }

    // Broadcast parameters to all ranks
    unsigned long long nn = numNodes;
    MPI_Bcast(&nn, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    numNodes = (size_t)nn;

    if (numNodes == 0) {
        MPI_Finalize();
        return 0;
    }

    int val_int = validate ? 1 : 0;
    int pr_int = printResults ? 1 : 0;
    MPI_Bcast(&val_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&pr_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
    validate = val_int;
    printResults = pr_int;

    // Compute local row distribution (1D block decomposition)
    size_t r_start, local_rows;
    compute_row_range(rank, nprocs, numNodes, r_start, local_rows);

    // Precompute scatter/gather counts and displacements
    size_t base_rows = numNodes / nprocs;
    size_t rem = numNodes % nprocs;
    std::vector<int> counts(nprocs), displs(nprocs);
    for (int p = 0; p < nprocs; ++p) {
        size_t p_rows, p_start;
        if ((size_t)p < rem) {
            p_rows = base_rows + 1;
            p_start = (size_t)p * (base_rows + 1);
        } else {
            p_rows = base_rows;
            p_start = rem * (base_rows + 1) + ((size_t)p - rem) * base_rows;
        }
        counts[p] = (int)(p_rows * numNodes);
        displs[p] = (int)(p_start * numNodes);
    }

    // Allocate local matrices (row-major: local_dist[li * numNodes + j])
    std::vector<unsigned int> local_dist(local_rows * numNodes);
    std::vector<unsigned int> local_path(local_rows * numNodes);

    // Initialize distance matrix on rank 0 and scatter to all ranks
    if (rank == 0) {
        std::vector<unsigned int> full_dist(numNodes * numNodes);
        unsigned int seed = 42;
        const double range = static_cast<double>(MAX_DISTANCE - 1) + 1.0;

        for (size_t i = 0; i < numNodes * numNodes; ++i) {
            full_dist[i] = 1 + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
        }
        for (size_t i = 0; i < numNodes; ++i) {
            full_dist[i * numNodes + i] = 0;
        }

        MPI_Scatterv(full_dist.data(), counts.data(), displs.data(), MPI_UNSIGNED,
                     local_dist.data(), (int)(local_rows * numNodes), MPI_UNSIGNED,
                     0, MPI_COMM_WORLD);
    } else {
        MPI_Scatterv(nullptr, nullptr, nullptr, MPI_UNSIGNED,
                     local_dist.data(), (int)(local_rows * numNodes), MPI_UNSIGNED,
                     0, MPI_COMM_WORLD);
    }

    // Initialize path matrix locally: path[i][j] = i for all j
    for (size_t li = 0; li < local_rows; ++li) {
        unsigned int row_val = (unsigned int)(r_start + li);
        for (size_t j = 0; j < numNodes; ++j) {
            local_path[li * numNodes + j] = row_val;
        }
    }

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing graph...\n");
        printf("Computing shortest paths...\n");
    }

    // MPI-parallel Floyd-Warshall with 1D block row decomposition
    // For each iteration k: broadcast row k, then update local rows
    std::vector<unsigned int> row_k(numNodes);

    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    for (size_t k = 0; k < numNodes; ++k) {
        int owner = compute_owner(k, nprocs, numNodes);

        // Owner copies row k into broadcast buffer
        if (rank == owner) {
            size_t local_k = k - r_start;
            std::memcpy(row_k.data(), &local_dist[local_k * numNodes],
                        numNodes * sizeof(unsigned int));
        }

        MPI_Bcast(row_k.data(), (int)numNodes, MPI_UNSIGNED, owner, MPI_COMM_WORLD);

        // Update all local rows: D[i][j] = min(D[i][j], D[i][k] + D[k][j])
        for (size_t li = 0; li < local_rows; ++li) {
            const unsigned int dist_ik = local_dist[li * numNodes + k];
            unsigned int* __restrict__ dist_row = &local_dist[li * numNodes];
            unsigned int* __restrict__ path_row = &local_path[li * numNodes];
            const unsigned int* __restrict__ rk = row_k.data();

            for (size_t j = 0; j < numNodes; ++j) {
                const unsigned int newDist = dist_ik + rk[j];
                if (newDist < dist_row[j]) {
                    dist_row[j] = newDist;
                    path_row[j] = (unsigned int)k;
                }
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t_end = MPI_Wtime();

    double local_elapsed = t_end - t_start;
    double max_elapsed;
    MPI_Reduce(&local_elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather full distance matrix on rank 0 for output/validation
    std::vector<unsigned int> full_dist;
    if (rank == 0) {
        full_dist.resize(numNodes * numNodes);
    }

    MPI_Gatherv(local_dist.data(), (int)(local_rows * numNodes), MPI_UNSIGNED,
                full_dist.data(), counts.data(), displs.data(), MPI_UNSIGNED,
                0, MPI_COMM_WORLD);

    // Output and validation on rank 0
    int return_code = 0;
    if (rank == 0) {
        long duration_ms = (long)(max_elapsed * 1000.0);
        printf("Computation time: %ld ms\n", duration_ms);

        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / (max_elapsed > 0.0 ? max_elapsed : 1e-9) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);

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
                return_code = 1;
            }
        }
    }

    MPI_Bcast(&return_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return return_code;
}
