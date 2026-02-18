#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// Avoid pulling in deprecated MPI C++ bindings (and their warnings)
#define OMPI_SKIP_MPICXX 1
#include <mpi.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

// Index calculation for flattened 2D array (column-major)
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

static inline size_t block_size(const int rank, const int size, const size_t nrows) noexcept {
    const size_t base = nrows / static_cast<size_t>(size);
    const size_t rem = nrows % static_cast<size_t>(size);
    return base + (static_cast<size_t>(rank) < rem ? 1u : 0u);
}

static inline size_t block_start(const int rank, const int size, const size_t nrows) noexcept {
    const size_t base = nrows / static_cast<size_t>(size);
    const size_t rem = nrows % static_cast<size_t>(size);
    return static_cast<size_t>(rank) * base + std::min(static_cast<size_t>(rank), rem);
}

static inline int owner_of_row(const size_t row, const int size, const size_t nrows) noexcept {
    const size_t base = nrows / static_cast<size_t>(size);
    const size_t rem = nrows % static_cast<size_t>(size);
    const size_t cut = (base + 1u) * rem;
    if (row < cut) {
        return static_cast<int>(row / (base + 1u));
    }
    return static_cast<int>(rem + (row - cut) / base);
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

void floydWarshallMPI(std::vector<unsigned int>& dist_local,
                      std::vector<unsigned int>& path_local,
                      const size_t numNodes,
                      const size_t localRows,
                      const size_t rowStart,
                      const int rank,
                      const int size) {
    // Distributed by rows of the logical matrix; local storage is column-major with leading dimension localRows.
    // For each k, broadcast the full k-th row (dist[k][j]) from the rank owning row k.
    std::vector<unsigned int> row_k(numNodes);

    for (size_t k = 0; k < numNodes; ++k) {
        const int owner = owner_of_row(k, size, numNodes);

        if (rank == owner) {
            const size_t lk = k - rowStart;
            for (size_t j = 0; j < numNodes; ++j) {
                row_k[j] = dist_local[j * localRows + lk];
            }
        }

        MPI_Bcast(row_k.data(), static_cast<int>(numNodes), MPI_UNSIGNED, owner, MPI_COMM_WORLD);

        const unsigned int* __restrict colK = dist_local.data() + k * localRows;
        for (size_t j = 0; j < numNodes; ++j) {
            const unsigned int distKJ = row_k[j];
            unsigned int* __restrict colJ = dist_local.data() + j * localRows;
            unsigned int* __restrict pathJ = path_local.data() + j * localRows;

            for (size_t li = 0; li < localRows; ++li) {
                const unsigned int distIJ = colJ[li];
                const unsigned int distIK = colK[li];
                const unsigned int newDist = distIK + distKJ;
                if (newDist < distIJ) {
                    colJ[li] = newDist;
                    pathJ[li] = static_cast<unsigned int>(k);
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

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;
    bool showHelp = false;
    bool badArgs = false;

    if (rank == 0) {
        // Parse command line arguments (rank 0 only)
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                numNodes = static_cast<size_t>(atoi(argv[++i]));
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = true;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = true;
            } else if (strcmp(argv[i], "-h") == 0) {
                showHelp = true;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                badArgs = true;
                break;
            }
        }

        if (badArgs) {
            printUsage(argv[0]);
        }
        if (showHelp) {
            printUsage(argv[0]);
        }
    }

    int flags[3] = {validate ? 1 : 0, printResults ? 1 : 0, (showHelp || badArgs) ? 1 : 0};
    MPI_Bcast(flags, 3, MPI_INT, 0, MPI_COMM_WORLD);
    validate = flags[0] != 0;
    printResults = flags[1] != 0;
    const bool earlyExit = flags[2] != 0;

    unsigned long long n_ull = static_cast<unsigned long long>(numNodes);
    MPI_Bcast(&n_ull, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    numNodes = static_cast<size_t>(n_ull);

    int early_exit_code = 0;
    if (rank == 0) {
        early_exit_code = badArgs ? 1 : 0;
    }
    MPI_Bcast(&early_exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (earlyExit) {
        MPI_Finalize();
        return early_exit_code;
    }

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark (MPI)\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("MPI ranks: %d\n", size);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const size_t localRows = block_size(rank, size, numNodes);
    const size_t rowStart = block_start(rank, size, numNodes);

    std::vector<unsigned int> dist_local(numNodes * localRows);
    std::vector<unsigned int> path_local(numNodes * localRows);

    // Initialize on rank 0 (to preserve exact serial RNG semantics), then scatter row blocks.
    std::vector<unsigned int> send_dist;
    std::vector<unsigned int> send_path;
    std::vector<int> sendcounts;
    std::vector<int> displs;

    if (rank == 0) {
        printf("Initializing graph...\n");
        std::vector<unsigned int> dist_full(numNodes * numNodes);
        std::vector<unsigned int> path_full(numNodes * numNodes);
        initializeDistanceMatrix(dist_full, numNodes, 1, MAX_DISTANCE);
        initializePathMatrix(path_full, numNodes);

        sendcounts.resize(size);
        displs.resize(size);
        size_t disp = 0;
        for (int r = 0; r < size; ++r) {
            const size_t rows_r = block_size(r, size, numNodes);
            sendcounts[r] = static_cast<int>(rows_r * numNodes);
            displs[r] = static_cast<int>(disp);
            disp += rows_r * numNodes;
        }

        send_dist.resize(numNodes * numNodes);
        send_path.resize(numNodes * numNodes);

        for (int r = 0; r < size; ++r) {
            const size_t rows_r = block_size(r, size, numNodes);
            const size_t start_r = block_start(r, size, numNodes);
            const size_t off_r = static_cast<size_t>(displs[r]);

            for (size_t j = 0; j < numNodes; ++j) {
                std::memcpy(send_dist.data() + off_r + j * rows_r,
                            dist_full.data() + j * numNodes + start_r,
                            rows_r * sizeof(unsigned int));
                std::memcpy(send_path.data() + off_r + j * rows_r,
                            path_full.data() + j * numNodes + start_r,
                            rows_r * sizeof(unsigned int));
            }
        }
    }

    MPI_Scatterv(rank == 0 ? send_dist.data() : nullptr,
                 rank == 0 ? sendcounts.data() : nullptr,
                 rank == 0 ? displs.data() : nullptr,
                 MPI_UNSIGNED,
                 dist_local.data(),
                 static_cast<int>(dist_local.size()),
                 MPI_UNSIGNED,
                 0,
                 MPI_COMM_WORLD);

    MPI_Scatterv(rank == 0 ? send_path.data() : nullptr,
                 rank == 0 ? sendcounts.data() : nullptr,
                 rank == 0 ? displs.data() : nullptr,
                 MPI_UNSIGNED,
                 path_local.data(),
                 static_cast<int>(path_local.size()),
                 MPI_UNSIGNED,
                 0,
                 MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computing shortest paths...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();

    floydWarshallMPI(dist_local, path_local, numNodes, localRows, rowStart, rank, size);

    MPI_Barrier(MPI_COMM_WORLD);
    const double t1 = MPI_Wtime();
    const double local_time = t1 - t0;

    double max_time = 0.0;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %.3f ms\n", max_time * 1000.0);
        const double ops = static_cast<double>(numNodes) * static_cast<double>(numNodes) * static_cast<double>(numNodes);
        const double gops = ops / max_time / 1e9;
        printf("Performance: %.3f GOPS\n", gops);
    }

    std::vector<unsigned int> dist_full_out;
    std::vector<unsigned int> recv_packed;

    if (validate || printResults) {
        if (rank == 0) {
            // Reuse the same distribution arrays from initialization.
            recv_packed.resize(numNodes * numNodes);
        }

        MPI_Gatherv(dist_local.data(),
                    static_cast<int>(dist_local.size()),
                    MPI_UNSIGNED,
                    rank == 0 ? recv_packed.data() : nullptr,
                    rank == 0 ? sendcounts.data() : nullptr,
                    rank == 0 ? displs.data() : nullptr,
                    MPI_UNSIGNED,
                    0,
                    MPI_COMM_WORLD);

        if (rank == 0) {
            dist_full_out.resize(numNodes * numNodes);
            for (int r = 0; r < size; ++r) {
                const size_t rows_r = block_size(r, size, numNodes);
                const size_t start_r = block_start(r, size, numNodes);
                const size_t off_r = static_cast<size_t>(displs[r]);

                for (size_t j = 0; j < numNodes; ++j) {
                    std::memcpy(dist_full_out.data() + j * numNodes + start_r,
                                recv_packed.data() + off_r + j * rows_r,
                                rows_r * sizeof(unsigned int));
                }
            }
        }
    }

    if (printResults && rank == 0) {
        print_results_int(dist_full_out, "DistanceMatrix");
    }

    int exit_code = 0;
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
            const bool valid = validateResult(dist_full_out, numNodes);
            if (valid) {
                printf("Validation: PASSED\n");
                exit_code = 0;
            } else {
                printf("Validation: FAILED\n");
                exit_code = 1;
            }
        }
        MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return exit_code;
}
