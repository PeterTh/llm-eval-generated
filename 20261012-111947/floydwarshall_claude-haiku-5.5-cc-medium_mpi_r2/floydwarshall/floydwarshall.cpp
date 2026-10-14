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

// Index calculation for flattened 2D array
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

// Contiguous block of matrix rows owned by one MPI rank
struct RowBlock {
    size_t start;  // first global row
    size_t count;  // number of rows
};

// Block row distribution: the first (numNodes % numProcs) ranks each get one extra row
RowBlock rowBlockOf(const int rank, const int numProcs, const size_t numNodes) {
    const size_t r = static_cast<size_t>(rank);
    const size_t p = static_cast<size_t>(numProcs);
    const size_t base = numNodes / p;
    const size_t rem = numNodes % p;
    return { r * base + std::min(r, rem), base + (r < rem ? 1 : 0) };
}

// Rank that owns global row `row` under rowBlockOf
int ownerOfRow(const size_t row, const int numProcs, const size_t numNodes) {
    const size_t p = static_cast<size_t>(numProcs);
    const size_t base = numNodes / p;
    const size_t rem = numNodes % p;
    const size_t bigRows = rem * (base + 1);
    if (row < bigRows) {
        return static_cast<int>(row / (base + 1));
    }
    return static_cast<int>(rem + (row - bigRows) / base);
}

// Fills this rank's rows of the distance matrix. Every rank replays the same rand_r
// sequence as the serial version so the values match exactly, but keeps only its own rows.
void initializeDistanceRows(std::vector<unsigned int>& dist, const size_t numNodes, const RowBlock& rows,
                            const unsigned int rangeMin, const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;

    const size_t firstFlat = rows.start * numNodes;
    const size_t endFlat = (rows.start + rows.count) * numNodes;
    dist.assign(rows.count * numNodes, 0);

    for (size_t i = 0; i < endFlat; ++i) {
        const unsigned int value = rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
        if (i >= firstFlat) {
            dist[i - firstFlat] = value;
        }
    }

    // Set diagonal to zero (distance from node to itself is 0)
    for (size_t i = 0; i < rows.count; ++i) {
        dist[i * numNodes + rows.start + i] = 0;
    }
}

// Path entry (i, j) starts as i, matching the serial initialization
void initializePathRows(std::vector<unsigned int>& path, const size_t numNodes, const RowBlock& rows) {
    path.resize(rows.count * numNodes);
    for (size_t i = 0; i < rows.count; ++i) {
        std::fill_n(path.begin() + i * numNodes, numNodes, static_cast<unsigned int>(rows.start + i));
    }
}

// Relaxes one row through intermediate node k: d[j] = min(d[j], d[k] + pivotRow[j]).
// Row k and column k do not change during iteration k (the diagonal is zero and weights are
// nonnegative), so reading d[k] once before the loop is equivalent to reading it per element.
inline void relaxRow(unsigned int* __restrict d, unsigned int* __restrict p,
                     const unsigned int* __restrict pivotRow, const size_t numNodes, const unsigned int k) {
    const unsigned int distIK = d[k];
    for (size_t j = 0; j < numNodes; ++j) {
        const unsigned int newDist = distIK + pivotRow[j];
        const bool improved = newDist < d[j];
        d[j] = improved ? newDist : d[j];
        p[j] = improved ? k : p[j];
    }
}

// Distributed Floyd-Warshall over row blocks. At step k the owner of row k has the final pivot
// row, which is broadcast to all ranks. The owner of row k+1 relaxes that row first and starts
// broadcasting it early, so the broadcast overlaps with the remaining local work of step k.
void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes,
                   const RowBlock& rows,
                   const int rank,
                   const int numProcs) {
    if (numNodes == 0) {
        return;
    }

    const int count = static_cast<int>(numNodes);
    std::vector<unsigned int> pivot[2] = {std::vector<unsigned int>(numNodes),
                                          std::vector<unsigned int>(numNodes)};
    MPI_Request request;

    // Row 0 is the first pivot row
    const int firstOwner = ownerOfRow(0, numProcs, numNodes);
    if (firstOwner == rank) {
        std::copy_n(dist.begin(), numNodes, pivot[0].begin());
    }
    MPI_Bcast(pivot[0].data(), count, MPI_UNSIGNED, firstOwner, MPI_COMM_WORLD);

    for (size_t k = 0; k < numNodes; ++k) {
        const int cur = static_cast<int>(k & 1);
        const int nxt = 1 - cur;
        const unsigned int* pivotK = pivot[cur].data();
        const unsigned int kk = static_cast<unsigned int>(k);
        const bool hasNext = k + 1 < numNodes;

        if (hasNext) {
            const size_t next = k + 1;
            const int nextOwner = ownerOfRow(next, numProcs, numNodes);
            if (nextOwner == rank) {
                const size_t li = next - rows.start;
                relaxRow(dist.data() + li * numNodes, path.data() + li * numNodes, pivotK, numNodes, kk);
                std::copy_n(dist.begin() + li * numNodes, numNodes, pivot[nxt].begin());
            }
            MPI_Ibcast(pivot[nxt].data(), count, MPI_UNSIGNED, nextOwner, MPI_COMM_WORLD, &request);
        }

        // Relax the remaining local rows (row k is unchanged by step k, and row k+1 is already done)
        for (size_t li = 0; li < rows.count; ++li) {
            const size_t global = rows.start + li;
            if (global == k || (hasNext && global == k + 1)) {
                continue;
            }
            relaxRow(dist.data() + li * numNodes, path.data() + li * numNodes, pivotK, numNodes, kk);
        }

        if (hasNext) {
            MPI_Wait(&request, MPI_STATUS_IGNORE);
        }
    }
}

// Gathers all row blocks onto rank 0, in global row order
void gatherDistanceMatrix(const std::vector<unsigned int>& localDist, std::vector<unsigned int>& fullDist,
                          const RowBlock& rows, const size_t numNodes, const int numProcs, const int rank) {
    MPI_Datatype rowType;
    MPI_Type_contiguous(static_cast<int>(numNodes), MPI_UNSIGNED, &rowType);
    MPI_Type_commit(&rowType);

    std::vector<int> counts(numProcs);
    std::vector<int> displs(numProcs);
    for (int r = 0; r < numProcs; ++r) {
        const RowBlock block = rowBlockOf(r, numProcs, numNodes);
        counts[r] = static_cast<int>(block.count);
        displs[r] = static_cast<int>(block.start);
    }

    if (rank == 0) {
        fullDist.resize(numNodes * numNodes);
    }
    MPI_Gatherv(localDist.data(), static_cast<int>(rows.count), rowType,
                fullDist.data(), counts.data(), displs.data(), rowType, 0, MPI_COMM_WORLD);

    MPI_Type_free(&rowType);
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
    int numProcs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
    const bool root = rank == 0;

    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (every rank parses identically, so all ranks take the same path)
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numNodes = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (root) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (root) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", numProcs);
    }

    // Allocate and initialize this rank's rows
    const RowBlock rows = rowBlockOf(rank, numProcs, numNodes);
    std::vector<unsigned int> dist;
    std::vector<unsigned int> path;
    if (root) {
        printf("Initializing graph...\n");
    }
    initializeDistanceRows(dist, numNodes, rows, 1, MAX_DISTANCE);
    initializePathRows(path, numNodes, rows);

    // Run Floyd-Warshall
    if (root) {
        printf("Computing shortest paths...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    floydWarshall(dist, path, numNodes, rows, rank, numProcs);

    MPI_Barrier(MPI_COMM_WORLD);
    const double seconds = MPI_Wtime() - start;

    if (root) {
        printf("Computation time: %ld ms\n", static_cast<long>(seconds * 1000.0));

        // Calculate operations per second
        // Floyd-Warshall has O(n³) complexity
        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / seconds / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);
    }

    if (!printResults && !validate) {
        MPI_Finalize();
        return 0;
    }

    // Results and validation need the full matrix on rank 0
    std::vector<unsigned int> fullDist;
    gatherDistanceMatrix(dist, fullDist, rows, numNodes, numProcs, rank);

    // Print results for external validation (integer hash-based)
    if (printResults && root) {
        print_results_int(fullDist, "DistanceMatrix");
    }

    // Validation
    int exitCode = 0;
    if (validate) {
        if (root) {
            printf("Validating result...\n");
        }
        int valid = 0;
        if (root) {
            valid = validateResult(fullDist, numNodes) ? 1 : 0;
            printf(valid ? "Validation: PASSED\n" : "Validation: FAILED\n");
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
        exitCode = valid ? 0 : 1;
    }

    MPI_Finalize();
    return exitCode;
}
