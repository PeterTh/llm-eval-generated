#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

// Index calculation for flattened 2D array (row-major: element (i, j) at i * n + j)
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

// Block-row distribution of n rows over p ranks.
// Rank r owns global rows [rowStart(r), rowStart(r+1)).
inline size_t rowStart(const size_t r, const size_t n, const size_t p) noexcept {
    return (r * n) / p;
}

// Per-rank view of the distributed matrices.
struct Partition {
    size_t n = 0;          // total number of nodes
    int rank = 0;
    int size = 1;
    size_t rowBegin = 0;   // first global row owned by this rank
    size_t rowCount = 0;   // number of rows owned by this rank
    std::vector<int> owner;  // owner[k] = rank owning global row k
};

Partition makePartition(const size_t n, const int rank, const int size) {
    Partition part;
    part.n = n;
    part.rank = rank;
    part.size = size;
    const size_t p = static_cast<size_t>(size);
    part.rowBegin = rowStart(static_cast<size_t>(rank), n, p);
    part.rowCount = rowStart(static_cast<size_t>(rank) + 1, n, p) - part.rowBegin;
    part.owner.resize(n);
    for (int r = 0; r < size; ++r) {
        const size_t b = rowStart(static_cast<size_t>(r), n, p);
        const size_t e = rowStart(static_cast<size_t>(r) + 1, n, p);
        for (size_t k = b; k < e; ++k) {
            part.owner[k] = r;
        }
    }
    return part;
}

// Fills the locally owned rows of the distance matrix. The random sequence is
// generated in the same global order as the sequential version, so the graph is
// identical regardless of the number of ranks.
void initializeDistanceMatrix(std::vector<unsigned int>& localDist, const Partition& part,
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    const size_t n = part.n;
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;

    for (size_t r = 0; r < n; ++r) {
        const bool local = (r >= part.rowBegin) && (r < part.rowBegin + part.rowCount);
        for (size_t c = 0; c < n; ++c) {
            const unsigned int value =
                rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
            if (local) {
                localDist[(r - part.rowBegin) * n + c] = value;
            }
        }
    }

    // Set diagonal to zero (distance from node to itself is 0)
    for (size_t r = part.rowBegin; r < part.rowBegin + part.rowCount; ++r) {
        localDist[(r - part.rowBegin) * n + r] = 0;
    }
}

void initializePathMatrix(std::vector<unsigned int>& localPath, const Partition& part) {
    const size_t n = part.n;
    for (size_t lr = 0; lr < part.rowCount; ++lr) {
        const unsigned int gr = static_cast<unsigned int>(part.rowBegin + lr);
        std::fill(localPath.begin() + lr * n, localPath.begin() + (lr + 1) * n, gr);
    }
}

// Relaxes one locally owned row i through intermediate node k, using the
// pivot row (row k of the distance matrix, already final for phase k).
// Row k and column k are unchanged during phase k, so reading D[i][k] once
// is equivalent to re-reading it for every j.
inline void relaxRow(unsigned int* __restrict d, unsigned int* __restrict p,
                     const unsigned int* __restrict pivot, const size_t n,
                     const size_t k) noexcept {
    const unsigned int dik = d[k];
    const unsigned int kk = static_cast<unsigned int>(k);
    for (size_t j = 0; j < n; ++j) {
        const unsigned int newDist = dik + pivot[j];
        const bool better = newDist < d[j];
        d[j] = better ? newDist : d[j];
        p[j] = better ? kk : p[j];
    }
}

// Distributed Floyd-Warshall. Each rank owns a block of rows of dist/path.
// For every intermediate node k, the owner of row k broadcasts that row.
// The owner of row k+1 is updated first and its row for the next phase is
// broadcast non-blockingly, overlapping the broadcast with the remaining updates.
void floydWarshallMPI(std::vector<unsigned int>& localDist,
                      std::vector<unsigned int>& localPath,
                      const Partition& part, MPI_Comm comm) {
    const size_t n = part.n;
    if (n == 0) {
        return;
    }
    const int nInt = static_cast<int>(n);

    std::vector<unsigned int> pivotBuf[2] = {std::vector<unsigned int>(n),
                                             std::vector<unsigned int>(n)};

    // Prime the pipeline with row 0.
    if (part.owner[0] == part.rank) {
        std::memcpy(pivotBuf[0].data(), localDist.data(), n * sizeof(unsigned int));
    }
    MPI_Request req = MPI_REQUEST_NULL;
    MPI_Ibcast(pivotBuf[0].data(), nInt, MPI_UNSIGNED, part.owner[0], comm, &req);

    for (size_t k = 0; k < n; ++k) {
        const size_t cur = k & 1;
        const size_t nxt = cur ^ 1;

        // Pivot row k is now available in pivotBuf[cur].
        MPI_Wait(&req, MPI_STATUS_IGNORE);
        const unsigned int* pivot = pivotBuf[cur].data();

        size_t skipLocal = static_cast<size_t>(-1);
        if (k + 1 < n) {
            const size_t nextRow = k + 1;
            if (part.owner[nextRow] == part.rank) {
                // Update the next pivot row first so it can be broadcast early.
                const size_t lr = nextRow - part.rowBegin;
                relaxRow(localDist.data() + lr * n, localPath.data() + lr * n, pivot, n, k);
                std::memcpy(pivotBuf[nxt].data(), localDist.data() + lr * n,
                            n * sizeof(unsigned int));
                skipLocal = lr;
            }
            MPI_Ibcast(pivotBuf[nxt].data(), nInt, MPI_UNSIGNED, part.owner[nextRow], comm,
                       &req);
        }

        // Relax all remaining local rows while the next pivot is in flight.
        for (size_t lr = 0; lr < part.rowCount; ++lr) {
            if (lr == skipLocal) {
                continue;
            }
            relaxRow(localDist.data() + lr * n, localPath.data() + lr * n, pivot, n, k);
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

// Gathers all distributed rows of `localDist` into `fullDist` on rank 0.
void gatherDistanceMatrix(const std::vector<unsigned int>& localDist, const Partition& part,
                          std::vector<unsigned int>& fullDist, MPI_Comm comm) {
    const size_t n = part.n;
    MPI_Datatype rowType;
    MPI_Type_contiguous(static_cast<int>(n), MPI_UNSIGNED, &rowType);
    MPI_Type_commit(&rowType);

    std::vector<int> counts(part.size), displs(part.size);
    for (int r = 0; r < part.size; ++r) {
        const size_t b = rowStart(static_cast<size_t>(r), n, static_cast<size_t>(part.size));
        const size_t e = rowStart(static_cast<size_t>(r) + 1, n, static_cast<size_t>(part.size));
        counts[r] = static_cast<int>(e - b);
        displs[r] = static_cast<int>(b);
    }

    if (part.rank == 0) {
        fullDist.assign(n * n, 0);
    }
    MPI_Gatherv(localDist.data(), static_cast<int>(part.rowCount), rowType,
                fullDist.data(), counts.data(), displs.data(), rowType, 0, comm);

    MPI_Type_free(&rowType);
}

int runBenchmark(int argc, char** argv, int rank, int size) {
    const MPI_Comm comm = MPI_COMM_WORLD;
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

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", size);
    }

    const Partition part = makePartition(numNodes, rank, size);

    // Allocate local portions of the matrices
    std::vector<unsigned int> localDist(part.rowCount * numNodes);
    std::vector<unsigned int> localPath(part.rowCount * numNodes);

    // Initialize
    if (rank == 0) {
        printf("Initializing graph...\n");
    }
    initializeDistanceMatrix(localDist, part, 1, MAX_DISTANCE);
    initializePathMatrix(localPath, part);

    // Run Floyd-Warshall
    if (rank == 0) {
        printf("Computing shortest paths...\n");
    }
    MPI_Barrier(comm);
    const double start = MPI_Wtime();

    floydWarshallMPI(localDist, localPath, part, comm);

    const double localElapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    // The parallel computation finishes when the slowest rank finishes.
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    if (rank == 0) {
        const long long durationMs = static_cast<long long>(elapsed * 1000.0);
        printf("Computation time: %lld ms\n", durationMs);

        // Calculate operations per second
        // Floyd-Warshall has O(n³) complexity
        const double ops = (double)numNodes * numNodes * numNodes;
        const double gflops = ops / (static_cast<double>(durationMs) / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);
    }

    // Gather the full distance matrix only when it is needed for output/validation
    std::vector<unsigned int> dist;
    if (printResults || validate) {
        gatherDistanceMatrix(localDist, part, dist, comm);
    }

    // Print results for external validation (integer hash-based)
    if (printResults && rank == 0) {
        print_results_int(dist, "DistanceMatrix");
    }

    // Validation
    if (validate) {
        int valid = 0;
        if (rank == 0) {
            printf("Validating result...\n");
            valid = validateResult(dist, numNodes) ? 1 : 0;
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        // Broadcast so every rank exits with the same status
        MPI_Bcast(&valid, 1, MPI_INT, 0, comm);
        return valid ? 0 : 1;
    }

    return 0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int rc = runBenchmark(argc, argv, rank, size);

    MPI_Finalize();
    return rc;
}
