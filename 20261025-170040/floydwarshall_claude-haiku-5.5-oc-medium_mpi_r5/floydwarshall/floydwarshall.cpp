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

// Block row distribution: rank p owns global rows [starts[p], starts[p+1]).
static std::vector<size_t> computeRowStarts(const size_t numNodes, const int numRanks) {
    std::vector<size_t> starts(static_cast<size_t>(numRanks) + 1);
    for (int p = 0; p <= numRanks; ++p) {
        starts[p] = static_cast<size_t>(p) * numNodes / static_cast<size_t>(numRanks);
    }
    return starts;
}

// Rank that owns global row k.
static int ownerOfRow(const std::vector<size_t>& starts, const size_t k) {
    auto it = std::upper_bound(starts.begin(), starts.end(), k);
    return static_cast<int>(it - starts.begin()) - 1;
}

// Fills rows [rowBegin, rowEnd) of the distance matrix. The random stream is
// consumed in exactly the same order as the serial version (row-major over
// the full matrix), so every rank sees identical values.
void initializeDistanceMatrix(std::vector<unsigned int>& local, const size_t numNodes,
                              const size_t rowBegin, const size_t rowEnd,
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;
    local.assign((rowEnd - rowBegin) * numNodes, 0u);

    for (size_t r = 0; r < numNodes; ++r) {
        const bool mine = (r >= rowBegin && r < rowEnd);
        for (size_t c = 0; c < numNodes; ++c) {
            const unsigned int v =
                rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
            if (mine) {
                local[(r - rowBegin) * numNodes + c] = v;
            }
        }
    }

    // Set diagonal to zero (distance from node to itself is 0)
    for (size_t r = rowBegin; r < rowEnd; ++r) {
        local[(r - rowBegin) * numNodes + r] = 0;
    }
}

// Path matrix entry (s,t) is initialized to s (see serial initializePathMatrix).
void initializePathMatrix(std::vector<unsigned int>& local, const size_t numNodes,
                          const size_t rowBegin, const size_t rowEnd) {
    local.assign((rowEnd - rowBegin) * numNodes, 0u);
    for (size_t r = rowBegin; r < rowEnd; ++r) {
        std::fill_n(local.begin() + (r - rowBegin) * numNodes, numNodes,
                    static_cast<unsigned int>(r));
    }
}

// Relaxes one destination row through intermediate node k:
//   dRow[j] = min(dRow[j], dik + kRow[j]), recording k in pRow on improvement.
static inline void relaxRow(unsigned int* __restrict dRow, unsigned int* __restrict pRow,
                            const unsigned int* __restrict kRow, const unsigned int dik,
                            const unsigned int k, const size_t n) {
    for (size_t j = 0; j < n; ++j) {
        const unsigned int nd = dik + kRow[j];
        const unsigned int old = dRow[j];
        const bool lt = nd < old;
        dRow[j] = lt ? nd : old;
        pRow[j] = lt ? k : pRow[j];
    }
}

// Distributed Floyd-Warshall. Each rank owns a contiguous block of rows.
// At step k the owner of row k+1 is relaxed first and its row is broadcast
// non-blockingly, so the broadcast overlaps with the bulk update of step k.
void floydWarshall(std::vector<unsigned int>& localDist,
                   std::vector<unsigned int>& localPath,
                   const size_t numNodes,
                   const std::vector<size_t>& starts,
                   const int rank,
                   MPI_Comm comm) {
    const size_t n = numNodes;
    if (n == 0) {
        return;
    }
    const size_t r0 = starts[rank];
    const size_t r1 = starts[rank + 1];
    const size_t m = r1 - r0;

    // Double-buffered pivot rows.
    std::vector<unsigned int> rowBuf(2 * n);
    unsigned int* bufs[2] = {rowBuf.data(), rowBuf.data() + n};

    // Prime: row 0 broadcast (blocking).
    {
        const int root = ownerOfRow(starts, 0);
        if (root == rank) {
            std::copy_n(localDist.data(), n, bufs[0]);
        }
        MPI_Bcast(bufs[0], static_cast<int>(n), MPI_UNSIGNED, root, comm);
    }

    for (size_t k = 0; k < n; ++k) {
        const unsigned int* cur = bufs[k & 1];
        unsigned int* nxt = bufs[(k + 1) & 1];
        const bool haveNext = (k + 1 < n);
        bool ownNext = false;
        MPI_Request req = MPI_REQUEST_NULL;

        if (haveNext) {
            const int root = ownerOfRow(starts, k + 1);
            if (root == rank) {
                // Update the next pivot row first, then publish it.
                const size_t li = (k + 1) - r0;
                unsigned int* dRow = localDist.data() + li * n;
                relaxRow(dRow, localPath.data() + li * n, cur, dRow[k],
                         static_cast<unsigned int>(k), n);
                std::copy_n(dRow, n, nxt);
                ownNext = true;
            }
            MPI_Ibcast(nxt, static_cast<int>(n), MPI_UNSIGNED, root, comm, &req);
        }

        // Bulk update of the remaining local rows. Row k is unchanged by
        // step k (its diagonal is zero), so it is skipped implicitly.
        for (size_t li = 0; li < m; ++li) {
            const size_t gi = r0 + li;
            if (ownNext && gi == k + 1) {
                continue;
            }
            unsigned int* dRow = localDist.data() + li * n;
            relaxRow(dRow, localPath.data() + li * n, cur, dRow[k],
                     static_cast<unsigned int>(k), n);
        }

        if (haveNext) {
            MPI_Wait(&req, MPI_STATUS_IGNORE);
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
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    const bool isRoot = (rank == 0);

    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (identical on all ranks)
    int exitCode = 0;
    bool exitEarly = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numNodes = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (isRoot) printUsage(argv[0]);
            exitCode = 0;
            exitEarly = true;
            break;
        } else {
            if (isRoot) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            exitCode = 1;
            exitEarly = true;
            break;
        }
    }
    if (exitEarly) {
        MPI_Finalize();
        return exitCode;
    }

    if (isRoot) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing graph...\n");
    }

    const size_t n = numNodes;
    const std::vector<size_t> starts = computeRowStarts(n, size);
    const size_t r0 = starts[rank];
    const size_t r1 = starts[rank + 1];

    // Each rank holds only its block of rows.
    std::vector<unsigned int> localDist;
    std::vector<unsigned int> localPath;
    initializeDistanceMatrix(localDist, n, r0, r1, 1, MAX_DISTANCE);
    initializePathMatrix(localPath, n, r0, r1);

    // Run Floyd-Warshall
    if (isRoot) {
        printf("Computing shortest paths...\n");
    }
    MPI_Barrier(comm);
    auto start = std::chrono::high_resolution_clock::now();

    floydWarshall(localDist, localPath, n, starts, rank, comm);

    MPI_Barrier(comm);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Gather the full distance matrix on the root when it is needed for output.
    std::vector<unsigned int> dist;
    if ((printResults || validate) && n > 0) {
        if (isRoot) {
            dist.resize(n * n);
        }
        MPI_Datatype rowType;
        MPI_Type_contiguous(static_cast<int>(n), MPI_UNSIGNED, &rowType);
        MPI_Type_commit(&rowType);

        std::vector<int> counts(size);
        std::vector<int> displs(size);
        for (int p = 0; p < size; ++p) {
            counts[p] = static_cast<int>(starts[p + 1] - starts[p]);
            displs[p] = static_cast<int>(starts[p]);
        }
        MPI_Gatherv(localDist.data(), static_cast<int>(r1 - r0), rowType,
                    isRoot ? dist.data() : nullptr, counts.data(), displs.data(), rowType,
                    0, comm);
        MPI_Type_free(&rowType);
    }

    if (isRoot) {
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
    }

    // Validation
    int valid = 1;
    if (validate) {
        if (isRoot) {
            printf("Validating result...\n");
            valid = validateResult(dist, n) ? 1 : 0;
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
        // Share the verdict so every rank exits with the same status.
        MPI_Bcast(&valid, 1, MPI_INT, 0, comm);
        exitCode = valid ? 0 : 1;
    }

    MPI_Finalize();
    return exitCode;
}
