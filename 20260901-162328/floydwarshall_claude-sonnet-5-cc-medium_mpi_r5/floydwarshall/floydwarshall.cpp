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

// Index calculation for flattened 2D array
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

// Computes the [begin, end) row range (source node "i") owned by a given
// MPI rank under a contiguous block decomposition of the rows.
void computeRowRange(const size_t numNodes, const int numRanks, const int rank,
                     size_t& rowBegin, size_t& rowEnd) {
    const size_t base = numNodes / static_cast<size_t>(numRanks);
    const size_t rem = numNodes % static_cast<size_t>(numRanks);
    const size_t r = static_cast<size_t>(rank);
    rowBegin = r * base + std::min(r, rem);
    rowEnd = rowBegin + base + (r < rem ? 1 : 0);
}

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes,
                   const int rank, const int numRanks) {
    // Row-wise block decomposition: each rank owns a contiguous range of
    // source nodes "i". Note that dist[idx2(j, i, n)] == dist[i * n + j],
    // i.e. row i (all destinations j) is contiguous in memory, and
    // dist[idx2(j, k, n)] == dist[k * n + j], so the pivot row k used by
    // every rank each iteration is also contiguous. Each rank keeps its own
    // authoritative copy of the rows it owns and receives the current pivot
    // row from its owning rank via broadcast.
    size_t rowBegin = 0, rowEnd = numNodes;
    std::vector<int> rowOwner(numNodes, 0);
    if (numRanks > 1) {
        computeRowRange(numNodes, numRanks, rank, rowBegin, rowEnd);
        for (int r = 0; r < numRanks; ++r) {
            size_t b = 0, e = 0;
            computeRowRange(numNodes, numRanks, r, b, e);
            for (size_t k = b; k < e; ++k) {
                rowOwner[k] = r;
            }
        }
    }

    // Classic Floyd-Warshall algorithm
    // For each intermediate node k
    for (size_t k = 0; k < numNodes; ++k) {
        // Broadcast the current pivot row (dist[k][*]) from its owning rank
        // so every rank has an up-to-date copy for this iteration.
        if (numRanks > 1) {
            MPI_Bcast(&dist[k * numNodes], static_cast<int>(numNodes),
                      MPI_UNSIGNED, rowOwner[k], MPI_COMM_WORLD);
        }

        // For each source node i owned by this rank
        for (size_t i = rowBegin; i < rowEnd; ++i) {
            // For each destination node j
            for (size_t j = 0; j < numNodes; ++j) {
                const unsigned int distIJ = dist[idx2(j, i, numNodes)];
                const unsigned int distIK = dist[idx2(k, i, numNodes)];
                const unsigned int distKJ = dist[idx2(j, k, numNodes)];

                const unsigned int newDist = distIK + distKJ;

                if (newDist < distIJ) {
                    dist[idx2(j, i, numNodes)] = newDist;
                    path[idx2(j, i, numNodes)] = k;
                }
            }
        }
    }

}

// Gathers each rank's authoritative row block back into rank 0's full
// matrices for validation/printing. Excluded from the timed compute phase.
void gatherResults(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes,
                   const int rank, const int numRanks) {
    if (numRanks <= 1) {
        return;
    }

    size_t rowBegin = 0, rowEnd = 0;
    computeRowRange(numNodes, numRanks, rank, rowBegin, rowEnd);

    std::vector<int> counts(numRanks), displs(numRanks);
    for (int r = 0; r < numRanks; ++r) {
        size_t b = 0, e = 0;
        computeRowRange(numNodes, numRanks, r, b, e);
        counts[r] = static_cast<int>((e - b) * numNodes);
        displs[r] = static_cast<int>(b * numNodes);
    }
    const int myCount = counts[rank];

    MPI_Gatherv(rank == 0 ? MPI_IN_PLACE : &dist[rowBegin * numNodes], myCount, MPI_UNSIGNED,
                dist.data(), counts.data(), displs.data(), MPI_UNSIGNED,
                0, MPI_COMM_WORLD);
    MPI_Gatherv(rank == 0 ? MPI_IN_PLACE : &path[rowBegin * numNodes], myCount, MPI_UNSIGNED,
                path.data(), counts.data(), displs.data(), MPI_UNSIGNED,
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

    int rank = 0, numRanks = 1;
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
            if (rank == 0) {
                printUsage(argv[0]);
            }
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
        printf("MPI ranks: %d\n", numRanks);
    }

    // Allocate matrices
    std::vector<unsigned int> dist(numNodes * numNodes);
    std::vector<unsigned int> path(numNodes * numNodes);

    // Initialize (identical, deterministic computation on every rank so no
    // communication is required to reach a consistent starting state)
    if (rank == 0) {
        printf("Initializing graph...\n");
    }
    initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
    initializePathMatrix(path, numNodes);

    // Run Floyd-Warshall
    if (rank == 0) {
        printf("Computing shortest paths...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    floydWarshall(dist, path, numNodes, rank, numRanks);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    gatherResults(dist, path, numNodes, rank, numRanks);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate operations per second
        // Floyd-Warshall has O(n³) complexity
        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);

        // Print results for external validation (integer hash-based)
        if (printResults) {
            print_results_int(dist, "DistanceMatrix");
        }

        // Validation
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(dist, numNodes);

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
