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

// Index calculation for flattened 2D array: M[row=i][col=j] => idx2(i,j) = j*n + i (column-major)
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
    // Initialize MPI first to allow MPI-aware runtimes to manage args
    MPI_Init(&argc, &argv);

    int world_rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (only rank 0 prints usage/errors)
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numNodes = static_cast<size_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (world_rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (world_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (world_rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Compute distribution of columns (i loop) among processes; columns are contiguous blocks of size numNodes
    std::vector<int> cols_per_rank(world_size, 0);
    std::vector<int> cols_disp(world_size, 0);

    int base = static_cast<int>(numNodes) / world_size;
    int rem = static_cast<int>(numNodes) % world_size;
    for (int r = 0; r < world_size; ++r) {
        cols_per_rank[r] = base + (r < rem ? 1 : 0);
    }
    cols_disp[0] = 0;
    for (int r = 1; r < world_size; ++r) cols_disp[r] = cols_disp[r - 1] + cols_per_rank[r - 1];

    int local_cols = cols_per_rank[world_rank];
    int local_col_offset = cols_disp[world_rank];

    // Each process will own local_cols * numNodes unsigned ints
    std::vector<unsigned int> localDist(static_cast<size_t>(local_cols) * numNodes);
    std::vector<unsigned int> localPath(static_cast<size_t>(local_cols) * numNodes);

    std::vector<unsigned int> fullDist; // only on rank 0
    std::vector<unsigned int> fullPath;

    // Root initializes full matrices and scatters columns
    if (world_rank == 0) {
        fullDist.resize(numNodes * numNodes);
        fullPath.resize(numNodes * numNodes);
        initializeDistanceMatrix(fullDist, numNodes, 1, MAX_DISTANCE);
        initializePathMatrix(fullPath, numNodes);
    }

    // Prepare counts and displacements in terms of number of unsigned ints
    std::vector<int> sendcounts(world_size);
    std::vector<int> displs(world_size);
    for (int r = 0; r < world_size; ++r) {
        sendcounts[r] = cols_per_rank[r] * static_cast<int>(numNodes);
        displs[r] = cols_disp[r] * static_cast<int>(numNodes);
    }

    // Scatter columns of dist and path to all processes
    MPI_Scatterv(world_rank == 0 ? fullDist.data() : nullptr, sendcounts.data(), displs.data(), MPI_UNSIGNED,
                 localDist.data(), static_cast<int>(localDist.size()), MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    MPI_Scatterv(world_rank == 0 ? fullPath.data() : nullptr, sendcounts.data(), displs.data(), MPI_UNSIGNED,
                 localPath.data(), static_cast<int>(localPath.size()), MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    // Ensure all processes received data
    MPI_Barrier(MPI_COMM_WORLD);

    if (world_rank == 0) printf("Computing shortest paths...\n");

    // Time only the parallel computation (rank 0 measures)
    MPI_Barrier(MPI_COMM_WORLD);
    auto t0 = std::chrono::high_resolution_clock::now();

    // Buffer to hold column k (dist[idx2(j,k)] for j=0..n-1)
    std::vector<unsigned int> colK(static_cast<size_t>(numNodes));

    for (size_t k = 0; k < numNodes; ++k) {
        // determine owner of column k
        int owner = 0;
        // find owner by cols_disp
        // binary search-like find
        int lo = 0, hi = world_size - 1;
        while (lo <= hi) {
            int mid = (lo + hi) / 2;
            if (k < static_cast<size_t>(cols_disp[mid])) {
                hi = mid - 1;
            } else if (mid + 1 < world_size && k >= static_cast<size_t>(cols_disp[mid] + cols_per_rank[mid])) {
                lo = mid + 1;
            } else {
                owner = mid;
                break;
            }
        }

        // If this rank owns the column k, set pointer to local data; otherwise use colK as receive buffer
        if (owner == world_rank) {
            int local_i = static_cast<int>(k) - local_col_offset; // index into local columns
            // localDist is stored per-column: localDist[local_i * numNodes + j] == dist[idx2(j, k)]
            // Broadcast in-place from owned memory
            MPI_Bcast(localDist.data() + static_cast<size_t>(local_i) * numNodes, static_cast<int>(numNodes), MPI_UNSIGNED, owner, MPI_COMM_WORLD);
            // For convenience, copy into colK for faster indexed access below (avoids pointer arithmetic every time)
            std::memcpy(colK.data(), localDist.data() + static_cast<size_t>(local_i) * numNodes, static_cast<size_t>(numNodes) * sizeof(unsigned int));
        } else {
            MPI_Bcast(colK.data(), static_cast<int>(numNodes), MPI_UNSIGNED, owner, MPI_COMM_WORLD);
        }

        // Each process updates its local columns using received column k
        for (int li = 0; li < local_cols; ++li) {
            // distIK is localDist[li * numNodes + k]
            unsigned int distIK = localDist[static_cast<size_t>(li) * numNodes + k];
            // If distIK is INF, adding will overflow; skip if INF
            if (distIK >= INF) continue;

            unsigned int* colPtr = localDist.data() + static_cast<size_t>(li) * numNodes;
            unsigned int* pathPtr = localPath.data() + static_cast<size_t>(li) * numNodes;

            for (size_t j = 0; j < numNodes; ++j) {
                unsigned int distKJ = colK[j];
                if (distKJ >= INF) continue;
                unsigned int newDist = distIK + distKJ;
                unsigned int cur = colPtr[j];
                if (newDist < cur) {
                    colPtr[j] = newDist;
                    pathPtr[j] = static_cast<unsigned int>(k);
                }
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto t1 = std::chrono::high_resolution_clock::now();

    // Gather results back to rank 0
    MPI_Gatherv(localDist.data(), static_cast<int>(localDist.size()), MPI_UNSIGNED,
                world_rank == 0 ? fullDist.data() : nullptr, sendcounts.data(), displs.data(), MPI_UNSIGNED,
                0, MPI_COMM_WORLD);

    MPI_Gatherv(localPath.data(), static_cast<int>(localPath.size()), MPI_UNSIGNED,
                world_rank == 0 ? fullPath.data() : nullptr, sendcounts.data(), displs.data(), MPI_UNSIGNED,
                0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate operations per second
        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
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
