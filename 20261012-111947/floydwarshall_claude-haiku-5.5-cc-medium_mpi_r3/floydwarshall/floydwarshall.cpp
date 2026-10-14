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

// Block-row distribution: rank r owns the contiguous global rows [rowStart[r], rowStart[r] + rowCount[r])
struct RowPartition {
    std::vector<int> rowCount;
    std::vector<int> rowStart;
    std::vector<int> ownerOf;  // rank owning each global row
};

RowPartition makePartition(const size_t numNodes, const int numRanks) {
    const size_t p = static_cast<size_t>(numRanks);
    RowPartition part;
    part.rowCount.resize(p);
    part.rowStart.resize(p);
    part.ownerOf.resize(numNodes);

    size_t start = 0;
    for (size_t r = 0; r < p; ++r) {
        const size_t count = numNodes / p + (r < numNodes % p ? 1 : 0);
        part.rowCount[r] = static_cast<int>(count);
        part.rowStart[r] = static_cast<int>(start);
        for (size_t g = start; g < start + count; ++g) {
            part.ownerOf[g] = static_cast<int>(r);
        }
        start += count;
    }
    return part;
}

// Path matrix holds the row index as its initial value: path[i][j] = i
void initializePathMatrix(std::vector<unsigned int>& path, const size_t firstRow,
                          const size_t numRows, const size_t numNodes) {
    for (size_t r = 0; r < numRows; ++r) {
        std::fill_n(&path[r * numNodes], numNodes, static_cast<unsigned int>(firstRow + r));
    }
}

// Relax the paths of one source row through intermediate node k
inline void relaxRow(unsigned int* __restrict distRow, unsigned int* __restrict pathRow,
                     const unsigned int* __restrict rowK, const unsigned int distIK,
                     const unsigned int k, const size_t numNodes) {
    for (size_t j = 0; j < numNodes; ++j) {
        const unsigned int newDist = distIK + rowK[j];
        const bool improved = newDist < distRow[j];
        distRow[j] = improved ? newDist : distRow[j];
        pathRow[j] = improved ? k : pathRow[j];
    }
}

// Distributed Floyd-Warshall over MPI. Each rank holds rows [firstRow, firstRow + numRows) of dist and path.
// In step k, row k and column k are unchanged, so the owner of row k broadcasts it and every rank relaxes
// its own rows independently. Row k+1 is relaxed and broadcast before the rest of step k runs, so the
// broadcast for the next step overlaps the local computation.
void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const RowPartition& part,
                   const int rank,
                   const size_t numNodes) {
    const size_t n = numNodes;
    if (n == 0) {
        return;
    }
    const size_t firstRow = static_cast<size_t>(part.rowStart[rank]);
    const size_t numRows = static_cast<size_t>(part.rowCount[rank]);

    // Double-buffered broadcast of row k, stored in buffer k % 2
    std::vector<unsigned int> rowBuf[2] = {std::vector<unsigned int>(n), std::vector<unsigned int>(n)};
    MPI_Request req[2];

    auto postBroadcast = [&](const size_t row) {
        unsigned int* buf = rowBuf[row % 2].data();
        const int root = part.ownerOf[row];
        if (root == rank) {
            std::copy_n(&dist[(row - firstRow) * n], n, buf);
        }
        MPI_Ibcast(buf, static_cast<int>(n), MPI_UNSIGNED, root, MPI_COMM_WORLD, &req[row % 2]);
    };

    postBroadcast(0);
    for (size_t k = 0; k < n; ++k) {
        MPI_Wait(&req[k % 2], MPI_STATUS_IGNORE);
        const unsigned int* rowK = rowBuf[k % 2].data();

        if (k + 1 < n) {
            if (part.ownerOf[k + 1] == rank) {
                const size_t lr = k + 1 - firstRow;
                relaxRow(&dist[lr * n], &path[lr * n], rowK, dist[lr * n + k], static_cast<unsigned int>(k), n);
            }
            postBroadcast(k + 1);
        }

        // Relax the remaining local rows. Row k is unchanged by step k and row k+1 is already done.
        for (size_t lr = 0; lr < numRows; ++lr) {
            const size_t g = firstRow + lr;
            if (g == k || g == k + 1) {
                continue;
            }
            relaxRow(&dist[lr * n], &path[lr * n], rowK, dist[lr * n + k], static_cast<unsigned int>(k), n);
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
    int numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (identical on every rank, so all ranks exit the same way)
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
    }

    const RowPartition part = makePartition(numNodes, numRanks);
    const size_t firstRow = static_cast<size_t>(part.rowStart[rank]);
    const size_t numRows = static_cast<size_t>(part.rowCount[rank]);

    // A row datatype lets the scatter and gather counts be expressed in rows instead of elements
    MPI_Datatype rowType;
    MPI_Type_contiguous(static_cast<int>(numNodes), MPI_UNSIGNED, &rowType);
    MPI_Type_commit(&rowType);

    // Allocate local matrices
    std::vector<unsigned int> dist(numRows * numNodes);
    std::vector<unsigned int> path(numRows * numNodes);

    // Initialize
    if (rank == 0) {
        printf("Initializing graph...\n");
    }
    {
        // Rank 0 generates the full matrix so the random sequence matches the serial program, then scatters rows
        std::vector<unsigned int> full;
        if (rank == 0) {
            full.resize(numNodes * numNodes);
            initializeDistanceMatrix(full, numNodes, 1, MAX_DISTANCE);
        }
        MPI_Scatterv(full.data(), part.rowCount.data(), part.rowStart.data(), rowType,
                     dist.data(), static_cast<int>(numRows), rowType, 0, MPI_COMM_WORLD);
    }
    initializePathMatrix(path, firstRow, numRows, numNodes);

    // Run Floyd-Warshall
    if (rank == 0) {
        printf("Computing shortest paths...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    floydWarshall(dist, path, part, rank, numNodes);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    int status = 0;
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate operations per second
        // Floyd-Warshall has O(n³) complexity
        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);
    }

    // Gather the full distance matrix only when it is needed for output or validation
    if (printResults || validate) {
        std::vector<unsigned int> full;
        if (rank == 0) {
            full.resize(numNodes * numNodes);
        }
        MPI_Gatherv(dist.data(), static_cast<int>(numRows), rowType,
                    full.data(), part.rowCount.data(), part.rowStart.data(), rowType, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            // Print results for external validation (integer hash-based)
            if (printResults) {
                print_results_int(full, "DistanceMatrix");
            }

            // Validation
            if (validate) {
                printf("Validating result...\n");
                bool valid = validateResult(full, numNodes);

                if (valid) {
                    printf("Validation: PASSED\n");
                } else {
                    printf("Validation: FAILED\n");
                    status = 1;
                }
            }
        }
    }

    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Type_free(&rowType);
    MPI_Finalize();
    return status;
}
