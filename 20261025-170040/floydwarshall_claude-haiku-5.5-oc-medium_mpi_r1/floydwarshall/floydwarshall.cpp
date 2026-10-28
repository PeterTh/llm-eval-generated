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

// Block-row distribution: rank r owns rows [rowStartOf(r), rowStartOf(r) + rowCountOf(r)).
// The first (n % P) ranks receive one extra row.
inline size_t rowStartOf(const int rank, const size_t n, const int P) noexcept {
    const size_t r = static_cast<size_t>(rank);
    const size_t base = n / static_cast<size_t>(P);
    const size_t rem = n % static_cast<size_t>(P);
    return r * base + std::min(r, rem);
}

inline size_t rowCountOf(const int rank, const size_t n, const int P) noexcept {
    return n / static_cast<size_t>(P) + (static_cast<size_t>(rank) < n % static_cast<size_t>(P) ? 1 : 0);
}

inline int rowOwner(const size_t row, const size_t n, const int P) noexcept {
    const size_t base = n / static_cast<size_t>(P);
    const size_t rem = n % static_cast<size_t>(P);
    const size_t bigSpan = rem * (base + 1);
    if (row < bigSpan) {
        return static_cast<int>(row / (base + 1));
    }
    return static_cast<int>(rem + (row - bigSpan) / base);
}

// Generates the full random stream in global row-major order (identical on every rank),
// but stores only the locally owned rows [rowStart, rowStart + rowCount).
void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes,
                              const unsigned int rangeMin, const unsigned int rangeMax,
                              const size_t rowStart, const size_t rowCount) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;
    const size_t rowEnd = rowStart + rowCount;

    for (size_t i = 0; i < numNodes; ++i) {
        const bool local = (i >= rowStart && i < rowEnd);
        for (size_t j = 0; j < numNodes; ++j) {
            const unsigned int value = rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
            if (local) {
                dist[(i - rowStart) * numNodes + j] = value;
            }
        }
    }

    // Set diagonal to zero (distance from node to itself is 0)
    for (size_t i = rowStart; i < rowEnd; ++i) {
        dist[(i - rowStart) * numNodes + i] = 0;
    }
}

// Initial path[i][j] = i for every entry (local rows only)
void initializePathMatrix(std::vector<unsigned int>& path, const size_t numNodes,
                          const size_t rowStart, const size_t rowCount) {
    for (size_t li = 0; li < rowCount; ++li) {
        const unsigned int src = static_cast<unsigned int>(rowStart + li);
        unsigned int* row = &path[li * numNodes];
        for (size_t j = 0; j < numNodes; ++j) {
            row[j] = src;
        }
    }
}

// Relaxes one row i against pivot row k: row[j] = min(row[j], dik + pivot[j]).
// Since dist[k][k] == 0, row k and column k are invariant during iteration k,
// so using a snapshot of the pivot row yields exactly the sequential result.
inline void relaxRow(unsigned int* __restrict row, unsigned int* __restrict pathRow,
                     const unsigned int* __restrict pivot, const size_t n,
                     const unsigned int dik, const unsigned int k) noexcept {
    for (size_t j = 0; j < n; ++j) {
        const unsigned int newDist = dik + pivot[j];
        const unsigned int oldDist = row[j];
        const bool improved = newDist < oldDist;
        row[j] = improved ? newDist : oldDist;
        pathRow[j] = improved ? k : pathRow[j];
    }
}

// Distributed Floyd-Warshall over MPI. Each rank owns a contiguous block of rows.
// Iteration k needs pivot row k, which is broadcast from its owner. The broadcast of
// row k+1 is started (non-blocking) before the bulk of iteration k's work, so that it
// overlaps with the local computation.
void floydWarshallMPI(std::vector<unsigned int>& dist,
                      std::vector<unsigned int>& path,
                      const size_t numNodes,
                      const int rank, const int size,
                      const size_t rowStart, const size_t rowCount) {
    if (numNodes == 0) {
        return;
    }

    const int n = static_cast<int>(numNodes);
    const size_t none = static_cast<size_t>(-1);

    // Double-buffered pivot rows
    std::vector<unsigned int> pivot[2] = {std::vector<unsigned int>(numNodes),
                                          std::vector<unsigned int>(numNodes)};

    // Pivot row 0
    {
        const int owner = rowOwner(0, numNodes, size);
        if (rank == owner) {
            std::copy_n(&dist[0], numNodes, pivot[0].data());
        }
        MPI_Bcast(pivot[0].data(), n, MPI_UNSIGNED, owner, MPI_COMM_WORLD);
    }

    for (size_t k = 0; k < numNodes; ++k) {
        const unsigned int* cur = pivot[k & 1].data();
        const unsigned int kk = static_cast<unsigned int>(k);

        MPI_Request req = MPI_REQUEST_NULL;
        size_t doneLocal = none;

        if (k + 1 < numNodes) {
            const size_t nk = k + 1;
            const int nextOwner = rowOwner(nk, numNodes, size);
            unsigned int* nxt = pivot[nk & 1].data();

            if (rank == nextOwner) {
                // Advance the next pivot row first so it can be broadcast early
                const size_t li = nk - rowStart;
                unsigned int* row = &dist[li * numNodes];
                relaxRow(row, &path[li * numNodes], cur, numNodes, row[k], kk);
                std::copy_n(row, numNodes, nxt);
                doneLocal = li;
            }

            MPI_Ibcast(nxt, n, MPI_UNSIGNED, nextOwner, MPI_COMM_WORLD, &req);
        }

        // Relax all remaining local rows against pivot row k
        for (size_t li = 0; li < rowCount; ++li) {
            if (li == doneLocal) {
                continue;
            }
            unsigned int* row = &dist[li * numNodes];
            relaxRow(row, &path[li * numNodes], cur, numNodes, row[k], kk);
        }

        if (req != MPI_REQUEST_NULL) {
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

// Runs the benchmark on all ranks. Returns the exit code (meaningful on rank 0).
int runBenchmark(const size_t numNodes, const bool validate, const bool printResults,
                 const int rank, const int size) {
    const bool isRoot = (rank == 0);
    const size_t rowStart = rowStartOf(rank, numNodes, size);
    const size_t rowCount = rowCountOf(rank, numNodes, size);

    if (isRoot) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Allocate local row blocks
    std::vector<unsigned int> dist(rowCount * numNodes);
    std::vector<unsigned int> path(rowCount * numNodes);

    // Initialize
    if (isRoot) {
        printf("Initializing graph...\n");
    }
    initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE, rowStart, rowCount);
    initializePathMatrix(path, numNodes, rowStart, rowCount);

    // Run Floyd-Warshall
    if (isRoot) {
        printf("Computing shortest paths...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();

    floydWarshallMPI(dist, path, numNodes, rank, size, rowStart, rowCount);

    const double localSeconds = MPI_Wtime() - t0;
    double seconds = 0.0;
    MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather full matrix on root only when it is needed for output or validation
    std::vector<unsigned int> full;
    if ((printResults || validate) && numNodes > 0) {
        if (isRoot) {
            full.resize(numNodes * numNodes);
        }
        std::vector<int> counts(size), displs(size);
        for (int r = 0; r < size; ++r) {
            counts[r] = static_cast<int>(rowCountOf(r, numNodes, size));
            displs[r] = static_cast<int>(rowStartOf(r, numNodes, size));
        }
        MPI_Datatype rowType;
        MPI_Type_contiguous(static_cast<int>(numNodes), MPI_UNSIGNED, &rowType);
        MPI_Type_commit(&rowType);
        MPI_Gatherv(dist.data(), static_cast<int>(rowCount), rowType,
                    full.data(), counts.data(), displs.data(), rowType,
                    0, MPI_COMM_WORLD);
        MPI_Type_free(&rowType);
    }

    int rc = 0;
    if (isRoot) {
        const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::duration<double>(seconds));

        printf("Computation time: %ld ms\n", duration.count());

        // Calculate operations per second
        // Floyd-Warshall has O(n³) complexity
        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / seconds / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);

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
                rc = 1;
            }
        }
    }
    return rc;
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

    // Parse command line arguments (identical on every rank)
    int rc = -1;  // -1: continue to benchmark
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
            rc = 0;
            break;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            rc = 1;
            break;
        }
    }

    if (rc < 0) {
        rc = runBenchmark(numNodes, validate, printResults, rank, size);
    }

    // Propagate the root's exit status so all ranks agree
    MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return rc;
}
