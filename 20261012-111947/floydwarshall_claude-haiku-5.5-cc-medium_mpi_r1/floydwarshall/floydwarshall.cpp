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

// Block distribution of matrix rows across MPI ranks
struct RowPartition {
    size_t start;
    size_t count;
};

RowPartition partitionRows(const size_t numNodes, const int numRanks, const int rank) {
    const size_t base = numNodes / numRanks;
    const size_t rem = numNodes % numRanks;
    const size_t r = static_cast<size_t>(rank);
    return { r * base + std::min(r, rem), base + (r < rem ? 1 : 0) };
}

// Each rank generates the full random sequence (identical to the serial code)
// but only keeps the rows it owns. Row-major: element (i, j) is at i * n + j.
void initializeLocalDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes,
                                   const RowPartition& part,
                                   const unsigned int rangeMin, const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;

    for (size_t i = 0; i < numNodes; ++i) {
        for (size_t j = 0; j < numNodes; ++j) {
            const unsigned int value = rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
            if (i >= part.start && i < part.start + part.count) {
                dist[(i - part.start) * numNodes + j] = value;
            }
        }
    }

    // Set diagonal to zero (distance from node to itself is 0)
    for (size_t i = part.start; i < part.start + part.count; ++i) {
        dist[(i - part.start) * numNodes + i] = 0;
    }
}

void initializeLocalPathMatrix(std::vector<unsigned int>& path, const size_t numNodes,
                               const RowPartition& part) {
    for (size_t i = 0; i < part.count; ++i) {
        std::fill_n(path.begin() + i * numNodes, numNodes, static_cast<unsigned int>(part.start + i));
    }
}

// Relax one local row against pivot row k: d[i][j] = min(d[i][j], d[i][k] + d[k][j]).
// Written as selects so the j-loop vectorizes.
inline void relaxRow(unsigned int* distRow, unsigned int* pathRow,
                     const unsigned int* kRow, const unsigned int distIK,
                     const unsigned int k, const size_t numNodes) {
    for (size_t j = 0; j < numNodes; ++j) {
        const unsigned int newDist = distIK + kRow[j];
        const unsigned int oldDist = distRow[j];
        const bool better = newDist < oldDist;
        distRow[j] = better ? newDist : oldDist;
        pathRow[j] = better ? k : pathRow[j];
    }
}

// Distributed Floyd-Warshall. Row k is unchanged during iteration k (its diagonal
// is zero), so every rank can relax its rows independently once it has row k.
// Pivot row k+1 is relaxed and broadcast first so its MPI_Ibcast overlaps with the
// remaining local work of iteration k.
void floydWarshallMPI(std::vector<unsigned int>& localDist,
                      std::vector<unsigned int>& localPath,
                      const size_t numNodes,
                      const RowPartition& part,
                      const std::vector<int>& ownerOfRow,
                      const int rank,
                      MPI_Comm comm) {
    if (numNodes == 0) {
        return;
    }
    const int n = static_cast<int>(numNodes);

    std::vector<unsigned int> rowBuf[2] = {
        std::vector<unsigned int>(numNodes),
        std::vector<unsigned int>(numNodes)
    };
    MPI_Request req[2];

    // Pivot row 0
    if (ownerOfRow[0] == rank) {
        std::copy_n(localDist.begin(), numNodes, rowBuf[0].begin());
    }
    MPI_Ibcast(rowBuf[0].data(), n, MPI_UNSIGNED, ownerOfRow[0], comm, &req[0]);

    for (size_t k = 0; k < numNodes; ++k) {
        const int cur = static_cast<int>(k & 1);
        MPI_Wait(&req[cur], MPI_STATUS_IGNORE);
        const unsigned int* kRow = rowBuf[cur].data();
        const unsigned int kk = static_cast<unsigned int>(k);

        // Local row already relaxed this iteration (skipped below)
        size_t done = part.count;

        if (k + 1 < numNodes) {
            const size_t nxt = k + 1;
            const int nb = static_cast<int>(nxt & 1);

            if (ownerOfRow[nxt] == rank) {
                const size_t lr = nxt - part.start;
                relaxRow(&localDist[lr * numNodes], &localPath[lr * numNodes],
                         kRow, localDist[lr * numNodes + k], kk, numNodes);
                std::copy_n(localDist.begin() + lr * numNodes, numNodes, rowBuf[nb].begin());
                done = lr;
            }
            MPI_Ibcast(rowBuf[nb].data(), n, MPI_UNSIGNED, ownerOfRow[nxt], comm, &req[nb]);
        }

        for (size_t lr = 0; lr < part.count; ++lr) {
            if (lr == done) {
                continue;
            }
            relaxRow(&localDist[lr * numNodes], &localPath[lr * numNodes],
                     kRow, localDist[lr * numNodes + k], kk, numNodes);
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

int run(int argc, char** argv, const int rank, const int numRanks) {
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
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    MPI_Comm comm = MPI_COMM_WORLD;

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing graph...\n");
    }

    // Row ownership for every rank (used to find the pivot row owner)
    const RowPartition part = partitionRows(numNodes, numRanks, rank);
    std::vector<int> ownerOfRow(numNodes);
    for (int r = 0; r < numRanks; ++r) {
        const RowPartition p = partitionRows(numNodes, numRanks, r);
        std::fill_n(ownerOfRow.begin() + p.start, p.count, r);
    }

    // Allocate and initialize only the local rows
    std::vector<unsigned int> localDist(part.count * numNodes);
    std::vector<unsigned int> localPath(part.count * numNodes);
    initializeLocalDistanceMatrix(localDist, numNodes, part, 1, MAX_DISTANCE);
    initializeLocalPathMatrix(localPath, numNodes, part);

    // Run Floyd-Warshall
    if (rank == 0) {
        printf("Computing shortest paths...\n");
    }
    MPI_Barrier(comm);
    const double start = MPI_Wtime();

    floydWarshallMPI(localDist, localPath, numNodes, part, ownerOfRow, rank, comm);

    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(maxElapsed * 1000.0));

        // Calculate operations per second
        // Floyd-Warshall has O(n³) complexity
        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / maxElapsed / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);
    }

    // Gather the full distance matrix on rank 0 only when it is needed for output or validation.
    // Rows are contiguous, so one row datatype keeps the MPI counts small.
    std::vector<unsigned int> dist;
    if (printResults || validate) {
        MPI_Datatype rowType;
        MPI_Type_contiguous(static_cast<int>(numNodes), MPI_UNSIGNED, &rowType);
        MPI_Type_commit(&rowType);

        std::vector<int> counts(numRanks), displs(numRanks);
        for (int r = 0; r < numRanks; ++r) {
            const RowPartition p = partitionRows(numNodes, numRanks, r);
            counts[r] = static_cast<int>(p.count);
            displs[r] = static_cast<int>(p.start);
        }
        if (rank == 0) {
            dist.resize(numNodes * numNodes);
        }
        MPI_Gatherv(localDist.data(), static_cast<int>(part.count), rowType,
                    dist.data(), counts.data(), displs.data(), rowType, 0, comm);
        MPI_Type_free(&rowType);
    }

    // Print results for external validation (integer hash-based)
    if (printResults && rank == 0) {
        print_results_int(dist, "DistanceMatrix");
    }

    int exitCode = 0;

    // Validation
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
            bool valid = validateResult(dist, numNodes);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
        // Every rank returns the same status
        MPI_Bcast(&exitCode, 1, MPI_INT, 0, comm);
    }

    return exitCode;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    const int status = run(argc, argv, rank, numRanks);

    MPI_Finalize();
    return status;
}
