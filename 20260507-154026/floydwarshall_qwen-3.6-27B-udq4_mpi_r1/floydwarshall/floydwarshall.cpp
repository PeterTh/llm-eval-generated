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

// Index calculation for flattened 2D array (column-major: idx2(i,j,n) = j*n+i)
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

// MPI-parallel Floyd-Warshall using 1D column-block distribution.
//
// The distance matrix D is stored column-major (column j is contiguous at [j*n .. j*n+n)).
// Columns are distributed in blocks across ranks. Each iteration k broadcasts column k
// (the pivot column) to all ranks; each rank then updates its local columns independently.
// Column k itself is never modified (D[i][k] = min(D[i][k], D[i][k]+D[k][k]) = D[i][k]
// because D[k][k] = 0), so no gather-back is needed mid-computation.
void floydWarshallMPI(std::vector<unsigned int>& dist,
                      std::vector<unsigned int>& path,
                      const size_t numNodes,
                      const int numRanks,
                      const int rank) {

    const size_t n = numNodes;

    // --- column-block distribution ---
    const size_t base_cols = n / numRanks;
    const size_t extra     = n % numRanks;

    // Column ranges for every rank
    std::vector<size_t> col_start(numRanks), col_end(numRanks);
    for (int r = 0; r < numRanks; ++r) {
        col_start[r] = r * base_cols + std::min(static_cast<size_t>(r), extra);
        col_end[r]   = (r + 1) * base_cols + std::min(static_cast<size_t>(r + 1), extra);
    }

    const size_t my_cols = col_end[rank] - col_start[rank];

    // Pre-compute column -> owning rank mapping
    std::vector<int> col_owner(n);
    for (int r = 0; r < numRanks; ++r) {
        for (size_t c = col_start[r]; c < col_end[r]; ++c)
            col_owner[c] = r;
    }

    // Scatterv / Gatherv parameters (columns are contiguous, so this is trivial)
    std::vector<int> sendcounts(numRanks), displs(numRanks);
    for (int r = 0; r < numRanks; ++r) {
        sendcounts[r] = static_cast<int>((col_end[r] - col_start[r]) * n);
        displs[r]     = static_cast<int>(col_start[r] * n);
    }

    // --- Scatter from rank 0 ---
    std::vector<unsigned int> local_dist(my_cols * n);
    std::vector<unsigned int> local_path(my_cols * n);

    MPI_Scatterv(dist.data(),  sendcounts.data(), displs.data(), MPI_UNSIGNED,
                 local_dist.data(), static_cast<int>(my_cols * n), MPI_UNSIGNED,
                 0, MPI_COMM_WORLD);
    MPI_Scatterv(path.data(),  sendcounts.data(), displs.data(), MPI_UNSIGNED,
                 local_path.data(), static_cast<int>(my_cols * n), MPI_UNSIGNED,
                 0, MPI_COMM_WORLD);

    // --- Buffers for broadcast pivot column k ---
    std::vector<unsigned int> col_k_dist(n);
    std::vector<unsigned int> col_k_path(n);

    // --- Main FW iterations ---
    for (size_t k = 0; k < n; ++k) {
        const int owner = col_owner[k];

        // Owner rank extracts its copy of column k
        if (rank == owner) {
            const size_t lc = k - col_start[rank];
            for (size_t i = 0; i < n; ++i) {
                col_k_dist[i] = local_dist[lc * n + i];
                col_k_path[i] = local_path[lc * n + i];
            }
        }

        // Broadcast pivot column to all ranks
        MPI_Bcast(col_k_dist.data(), static_cast<int>(n), MPI_UNSIGNED, owner, MPI_COMM_WORLD);
        MPI_Bcast(col_k_path.data(), static_cast<int>(n), MPI_UNSIGNED, owner, MPI_COMM_WORLD);

        // Each rank updates its local columns
        for (size_t c = 0; c < my_cols; ++c) {
            const unsigned int distKJ = local_dist[c * n + k];   // D[k][j]

            for (size_t i = 0; i < n; ++i) {
                const unsigned int distIJ = local_dist[c * n + i];   // D[i][j]
                const unsigned int distIK = col_k_dist[i];            // D[i][k]

                const unsigned int newDist = distIK + distKJ;
                if (newDist < distIJ) {
                    local_dist[c * n + i] = newDist;
                    local_path[c * n + i] = static_cast<unsigned int>(k);
                }
            }
        }
    }

    // --- Gather back to rank 0 ---
    MPI_Gatherv(local_dist.data(), static_cast<int>(my_cols * n), MPI_UNSIGNED,
                dist.data(), sendcounts.data(), displs.data(), MPI_UNSIGNED,
                0, MPI_COMM_WORLD);
    MPI_Gatherv(local_path.data(), static_cast<int>(my_cols * n), MPI_UNSIGNED,
                path.data(), sendcounts.data(), displs.data(), MPI_UNSIGNED,
                0, MPI_COMM_WORLD);
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

    int numRanks, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (on all ranks, but only rank 0 prints)
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

    // Broadcast parameters so every rank agrees
    MPI_Bcast(&numNodes,     1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate,     1, MPI_C_BOOL,        0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_C_BOOL,        0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("MPI ranks: %d\n", numRanks);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Allocate matrices (full on every rank for scatter/gather)
    std::vector<unsigned int> dist(numNodes * numNodes);
    std::vector<unsigned int> path(numNodes * numNodes);

    // Initialize on rank 0
    if (rank == 0) {
        printf("Initializing graph...\n");
        initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
        initializePathMatrix(path, numNodes);
    }

    // Run parallel Floyd-Warshall
    if (rank == 0) printf("Computing shortest paths...\n");

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    floydWarshallMPI(dist, path, numNodes, numRanks, rank);

    auto end = std::chrono::high_resolution_clock::now();
    MPI_Barrier(MPI_COMM_WORLD);

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long long localDurationMs = static_cast<long long>(duration.count());
    long long maxDurationMs = 0;
    MPI_Reduce(&localDurationMs, &maxDurationMs, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    // Rank 0 reports results
    if (rank == 0) {
        printf("Computation time: %lld ms\n", maxDurationMs);

        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / (maxDurationMs / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);

        if (printResults) {
            print_results_int(dist, "DistanceMatrix");
        }

        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(dist, numNodes);

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
