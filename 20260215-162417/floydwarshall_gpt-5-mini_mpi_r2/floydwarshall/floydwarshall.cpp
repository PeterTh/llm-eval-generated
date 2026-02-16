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

// Index calculation for flattened 2D array (column-major: element (row=i, col=j) stored at j*n + i)
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

    // Set diagonal to zero
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
    // Initialize MPI (unconditional)
    MPI_Init(&argc, &argv);
    int world_size = 1, world_rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (only done on rank 0 and broadcast)
    if (world_rank == 0) {
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
    // Broadcast options to all ranks
    MPI_Bcast(&numNodes, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Determine column distribution (each process owns a contiguous set of columns)
    std::vector<int> cols_per_proc(world_size);
    int base = static_cast<int>(numNodes) / world_size;
    int rem = static_cast<int>(numNodes) % world_size;
    for (int r = 0; r < world_size; ++r) {
        cols_per_proc[r] = base + (r < rem ? 1 : 0);
    }
    std::vector<int> sendcounts(world_size);
    std::vector<int> displs(world_size);
    int offset = 0;
    for (int r = 0; r < world_size; ++r) {
        sendcounts[r] = cols_per_proc[r] * static_cast<int>(numNodes);
        displs[r] = offset;
        offset += sendcounts[r];
    }

    const int local_cols = cols_per_proc[world_rank];
    std::vector<unsigned int> localDist(local_cols * numNodes);
    std::vector<unsigned int> localPath(local_cols * numNodes);

    std::vector<unsigned int> fullDist;
    std::vector<unsigned int> fullPath;
    if (world_rank == 0) {
        fullDist.resize(numNodes * numNodes);
        fullPath.resize(numNodes * numNodes);
        printf("Initializing graph...\n");
        initializeDistanceMatrix(fullDist, numNodes, 1, MAX_DISTANCE);
        initializePathMatrix(fullPath, numNodes);
    }

    // Scatter columns to all processes
    MPI_Scatterv(fullDist.data(), sendcounts.data(), displs.data(), MPI_UNSIGNED,
                 localDist.data(), sendcounts[world_rank], MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Scatterv(fullPath.data(), sendcounts.data(), displs.data(), MPI_UNSIGNED,
                 localPath.data(), sendcounts[world_rank], MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    // Precompute start column index for each rank
    std::vector<int> start_col(world_size);
    start_col[0] = 0;
    for (int r = 1; r < world_size; ++r) start_col[r] = start_col[r-1] + cols_per_proc[r-1];

    // Allocate buffer for column k that will be broadcast
    std::vector<unsigned int> column_k(numNodes);

    // Synchronize and start timer
    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    // Parallel Floyd-Warshall: iterate k sequentially, broadcast column k, update local columns
    for (size_t k = 0; k < numNodes; ++k) {
        // Determine owner of column k
        int owner = 0;
        // find owner such that start_col[owner] <= k < start_col[owner] + cols_per_proc[owner]
        int low = 0, high = world_size - 1;
        while (low <= high) {
            int mid = (low + high) / 2;
            int s = start_col[mid];
            int e = s + cols_per_proc[mid] - 1;
            if (static_cast<int>(k) < s) { high = mid - 1; }
            else if (static_cast<int>(k) > e) { low = mid + 1; }
            else { owner = mid; break; }
        }

        // Owner copies its column k into column_k
        if (world_rank == owner) {
            int local_index = static_cast<int>(k) - start_col[owner];
            // copy local column into buffer (column stored as local_index * numNodes + row)
            unsigned int* src = localDist.data() + (size_t)local_index * numNodes;
            std::memcpy(column_k.data(), src, numNodes * sizeof(unsigned int));
        }

        // Broadcast column_k from owner to all processes
        MPI_Bcast(column_k.data(), static_cast<int>(numNodes), MPI_UNSIGNED, owner, MPI_COMM_WORLD);

        // Update local columns using received column_k
        for (int local_i = 0; local_i < local_cols; ++local_i) {
            unsigned int* col_ptr = localDist.data() + (size_t)local_i * numNodes;
            unsigned int* path_ptr = localPath.data() + (size_t)local_i * numNodes;
            // distIK is the element at row k in this local column: col_ptr[k]
            unsigned int distIK = col_ptr[k];
            for (size_t j = 0; j < numNodes; ++j) {
                unsigned int distIJ = col_ptr[j];
                unsigned int distKJ = column_k[j];
                unsigned int newDist = distIK + distKJ;
                if (newDist < distIJ) {
                    col_ptr[j] = newDist;
                    path_ptr[j] = static_cast<unsigned int>(k);
                }
            }
        }
    }

    // Synchronize and stop timer
    MPI_Barrier(MPI_COMM_WORLD);
    double t_end = MPI_Wtime();
    double local_elapsed = t_end - t_start;
    double max_elapsed = 0.0;
    MPI_Reduce(&local_elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather results back to root for printing/validation
    MPI_Gatherv(localDist.data(), sendcounts[world_rank], MPI_UNSIGNED,
                fullDist.data(), sendcounts.data(), displs.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    // Root prints performance and optionally prints results or validates
    if (world_rank == 0) {
        long ms = static_cast<long>(max_elapsed * 1000.0);
        printf("Computation time: %ld ms\n", ms);

        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / max_elapsed / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);

        if (printResults) {
            print_results_int(fullDist, "DistanceMatrix");
        }

        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(fullDist, numNodes);
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
