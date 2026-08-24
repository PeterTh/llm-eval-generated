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

// Index calculation for flattened 2D array (column-major, matching original)
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

// Generate the full distance matrix identically to the original sequential code
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

// Compute row ownership for a given global row index
inline int rowOwner(size_t row, size_t base_rows, size_t remainder) {
    size_t big_block = base_rows + 1;
    size_t big_count = remainder * big_block;
    if (row < big_count) {
        return static_cast<int>(row / big_block);
    }
    return static_cast<int>(remainder + (row - big_count) / base_rows);
}

// Compute (my_start, local_n) for a given rank
inline void rowRange(int rank, int size, size_t numNodes, size_t& my_start, size_t& local_n) {
    size_t base_rows = numNodes / size;
    size_t remainder = numNodes % size;
    if (static_cast<size_t>(rank) < remainder) {
        local_n = base_rows + 1;
        my_start = rank * (base_rows + 1);
    } else {
        local_n = base_rows;
        my_start = remainder * (base_rows + 1) + static_cast<size_t>(rank - remainder) * base_rows;
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

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (identical on all ranks)
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

    // Compute this process's row range
    size_t base_rows = numNodes / nprocs;
    size_t remainder = numNodes % nprocs;
    size_t my_start, local_n;
    rowRange(rank, nprocs, numNodes, my_start, local_n);

    // --- Initialization ---
    // Rank 0 generates the full distance matrix, then scatterv distributes rows.
    // The original layout: dist[i*n + j] = distance from i to j
    // (because idx2(j, i, n) = i*n + j).
    std::vector<unsigned int> local_dist(local_n * numNodes);
    std::vector<unsigned int> local_path(local_n * numNodes);

    if (rank == 0) {
        printf("Initializing graph...\n");

        std::vector<unsigned int> full_dist;
        full_dist.resize(numNodes * numNodes);
        initializeDistanceMatrix(full_dist, numNodes, 1, MAX_DISTANCE);

        // Scatterv distance matrix rows to all processes
        std::vector<int> sendcounts(nprocs);
        std::vector<int> displs(nprocs);
        for (int p = 0; p < nprocs; ++p) {
            size_t p_start, p_n;
            rowRange(p, nprocs, numNodes, p_start, p_n);
            sendcounts[p] = static_cast<int>(p_n * numNodes);
            displs[p] = static_cast<int>(p_start * numNodes);
        }

        MPI_Scatterv(full_dist.data(), sendcounts.data(), displs.data(), MPI_UNSIGNED,
                     local_dist.data(), static_cast<int>(local_n * numNodes), MPI_UNSIGNED,
                     0, MPI_COMM_WORLD);
    } else {
        MPI_Scatterv(nullptr, nullptr, nullptr, MPI_UNSIGNED,
                     local_dist.data(), static_cast<int>(local_n * numNodes), MPI_UNSIGNED,
                     0, MPI_COMM_WORLD);
    }

    // Initialize local path matrix: path[i*n + j] = i (derived from original loop)
    for (size_t li = 0; li < local_n; ++li) {
        unsigned int row_val = static_cast<unsigned int>(my_start + li);
        for (size_t j = 0; j < numNodes; ++j) {
            local_path[li * numNodes + j] = row_val;
        }
    }

    // --- MPI Floyd-Warshall (1D block-row decomposition) ---
    if (rank == 0) printf("Computing shortest paths...\n");

    // Buffer for broadcasting row k of the distance matrix
    std::vector<unsigned int> row_k(numNodes);

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (size_t k = 0; k < numNodes; ++k) {
        int owner = rowOwner(k, base_rows, remainder);

        // The owning process copies row k into the broadcast buffer
        if (rank == owner) {
            size_t local_k = k - my_start;
            std::memcpy(row_k.data(), &local_dist[local_k * numNodes],
                        numNodes * sizeof(unsigned int));
        }

        // Broadcast row k from owner to all processes
        MPI_Bcast(row_k.data(), static_cast<int>(numNodes), MPI_UNSIGNED, owner, MPI_COMM_WORLD);

        // Each process updates its local rows
        for (size_t li = 0; li < local_n; ++li) {
            const unsigned int distIK = local_dist[li * numNodes + k];
            unsigned int* __restrict__ dist_row = &local_dist[li * numNodes];
            unsigned int* __restrict__ path_row = &local_path[li * numNodes];
            const unsigned int* __restrict__ rk = row_k.data();
            const unsigned int k_val = static_cast<unsigned int>(k);

            for (size_t j = 0; j < numNodes; ++j) {
                const unsigned int newDist = distIK + rk[j];
                if (newDist < dist_row[j]) {
                    dist_row[j] = newDist;
                    path_row[j] = k_val;
                }
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long local_duration_ms = duration.count();
    long max_duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &max_duration_ms, 1, MPI_LONG, MPI_MAX,
               0, MPI_COMM_WORLD);

    // --- Gather results to rank 0 ---
    std::vector<unsigned int> full_dist;
    if (rank == 0) {
        full_dist.resize(numNodes * numNodes);
    }

    {
        std::vector<int> recvcounts(nprocs);
        std::vector<int> displs(nprocs);
        for (int p = 0; p < nprocs; ++p) {
            size_t p_start, p_n;
            rowRange(p, nprocs, numNodes, p_start, p_n);
            recvcounts[p] = static_cast<int>(p_n * numNodes);
            displs[p] = static_cast<int>(p_start * numNodes);
        }

        MPI_Gatherv(local_dist.data(), static_cast<int>(local_n * numNodes), MPI_UNSIGNED,
                    full_dist.data(), recvcounts.data(), displs.data(), MPI_UNSIGNED,
                    0, MPI_COMM_WORLD);
    }

    // --- Output (rank 0 only) ---
    if (rank == 0) {
        printf("Computation time: %ld ms\n", max_duration_ms);

        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / (max_duration_ms / 1000.0) / 1e9;
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
                MPI_Finalize();
                return 1;
            }
        }
    }

    MPI_Finalize();
    return 0;
}
