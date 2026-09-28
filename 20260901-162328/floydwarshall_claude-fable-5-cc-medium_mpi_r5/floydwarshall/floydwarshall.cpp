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

// Block-row distribution: rank r owns rows [rowStart(r), rowStart(r) + rowCount(r)).
// Row i of the matrix occupies the contiguous range [i*n, (i+1)*n) in memory.
inline size_t rowStart(const int rank, const int nprocs, const size_t numNodes) {
    const size_t base = numNodes / nprocs;
    const size_t rem = numNodes % nprocs;
    const size_t r = static_cast<size_t>(rank);
    return r * base + std::min(r, rem);
}

inline size_t rowCount(const int rank, const int nprocs, const size_t numNodes) {
    return rowStart(rank + 1, nprocs, numNodes) - rowStart(rank, nprocs, numNodes);
}

inline int rowOwner(const size_t row, const int nprocs, const size_t numNodes) {
    const size_t base = numNodes / nprocs;
    const size_t rem = numNodes % nprocs;
    const size_t cutoff = rem * (base + 1);
    if (row < cutoff) {
        return static_cast<int>(row / (base + 1));
    }
    return static_cast<int>(rem + (row - cutoff) / base);
}

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes,
                   const int rank, const int nprocs) {
    const size_t myStart = rowStart(rank, nprocs, numNodes);
    const size_t myCount = rowCount(rank, nprocs, numNodes);
    const size_t myEnd = myStart + myCount;

    std::vector<unsigned int> kRowBuf(numNodes);

    // Classic Floyd-Warshall algorithm, distributed by block rows.
    // For each intermediate node k, the owner of row k broadcasts it, then
    // every rank updates its own block of rows.
    for (size_t k = 0; k < numNodes; ++k) {
        const int owner = rowOwner(k, nprocs, numNodes);
        const unsigned int* kRow;
        if (owner == rank) {
            // Broadcast directly from the local matrix to avoid a copy
            MPI_Bcast(&dist[k * numNodes], (int)numNodes, MPI_UNSIGNED, owner, MPI_COMM_WORLD);
            kRow = &dist[k * numNodes];
        } else {
            MPI_Bcast(kRowBuf.data(), (int)numNodes, MPI_UNSIGNED, owner, MPI_COMM_WORLD);
            kRow = kRowBuf.data();
        }

        // For each locally owned source node i
        for (size_t i = myStart; i < myEnd; ++i) {
            unsigned int* __restrict__ distRow = &dist[i * numNodes];
            unsigned int* __restrict__ pathRow = &path[i * numNodes];
            const unsigned int distIK = distRow[k];

            // For each destination node j
            for (size_t j = 0; j < numNodes; ++j) {
                const unsigned int newDist = distIK + kRow[j];

                if (newDist < distRow[j]) {
                    distRow[j] = newDist;
                    pathRow[j] = (unsigned int)k;
                }
            }
        }
    }

    // Gather the distributed rows back to rank 0 for output and validation
    std::vector<int> counts(nprocs), displs(nprocs);
    for (int r = 0; r < nprocs; ++r) {
        counts[r] = (int)(rowCount(r, nprocs, numNodes) * numNodes);
        displs[r] = (int)(rowStart(r, nprocs, numNodes) * numNodes);
    }
    const void* sendbuf = (rank == 0) ? MPI_IN_PLACE : (const void*)&dist[myStart * numNodes];
    MPI_Gatherv(sendbuf, (int)(myCount * numNodes), MPI_UNSIGNED,
                dist.data(), counts.data(), displs.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    sendbuf = (rank == 0) ? MPI_IN_PLACE : (const void*)&path[myStart * numNodes];
    MPI_Gatherv(sendbuf, (int)(myCount * numNodes), MPI_UNSIGNED,
                path.data(), counts.data(), displs.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
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

    int rank = 0, nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

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
        printf("MPI processes: %d\n", nprocs);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Allocate matrices (full-size on every rank; each rank works on its block
    // of rows, and rank 0 receives the full result at the end)
    std::vector<unsigned int> dist(numNodes * numNodes);
    std::vector<unsigned int> path(numNodes * numNodes);

    // Initialize (deterministic, so every rank computes the same matrices)
    if (rank == 0) printf("Initializing graph...\n");
    initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
    initializePathMatrix(path, numNodes);

    // Run Floyd-Warshall
    if (rank == 0) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    floydWarshall(dist, path, numNodes, rank, nprocs);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    int exitCode = 0;
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
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }

    MPI_Finalize();
    return exitCode;
}
