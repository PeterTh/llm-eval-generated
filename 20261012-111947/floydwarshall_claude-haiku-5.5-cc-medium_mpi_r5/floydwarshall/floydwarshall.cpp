#include <algorithm>
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

// Block row distribution: rank r owns the contiguous rows [begin(r), begin(r) + count(r)).
// The first `rem` ranks own one extra row so the load differs by at most one row.
struct RowPartition {
    size_t base;
    size_t rem;

    RowPartition(const size_t numNodes, const int numRanks)
        : base(numNodes / numRanks), rem(numNodes % numRanks) {}

    size_t begin(const int rank) const {
        return rank * base + std::min(static_cast<size_t>(rank), rem);
    }

    size_t count(const int rank) const {
        return base + (static_cast<size_t>(rank) < rem ? 1 : 0);
    }

    int owner(const size_t row) const {
        const size_t bigRows = rem * (base + 1);
        if (row < bigRows) return static_cast<int>(row / (base + 1));
        return static_cast<int>(rem + (row - bigRows) / base);
    }
};

// Fills only the rows [firstRow, firstRow + numLocal) of the global matrix.
// Every rank walks the full global random sequence so the values match the
// sequential generator exactly.
void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes,
                              const size_t firstRow, const size_t numLocal,
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;

    const size_t localBegin = firstRow * numNodes;
    const size_t localEnd = localBegin + numLocal * numNodes;
    for (size_t i = 0; i < numNodes * numNodes; ++i) {
        const unsigned int value = rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
        if (i >= localBegin && i < localEnd) {
            dist[i - localBegin] = value;
        }
    }

    // Set diagonal to zero (distance from node to itself is 0)
    for (size_t li = 0; li < numLocal; ++li) {
        dist[li * numNodes + firstRow + li] = 0;
    }
}

// Entry (r, c) of the global path matrix is initialized to r
void initializePathMatrix(std::vector<unsigned int>& path, const size_t firstRow,
                          const size_t numLocal, const size_t numNodes) {
    for (size_t li = 0; li < numLocal; ++li) {
        std::fill_n(path.begin() + li * numNodes, numNodes, static_cast<unsigned int>(firstRow + li));
    }
}

// Relaxes one row i against pivot row k: dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
inline void relaxRow(unsigned int* distRow, unsigned int* pathRow, const unsigned int* pivotRow,
                     const unsigned int distIK, const unsigned int k, const size_t numNodes) {
    for (size_t j = 0; j < numNodes; ++j) {
        const unsigned int oldDist = distRow[j];
        const unsigned int newDist = distIK + pivotRow[j];
        const bool improved = newDist < oldDist;
        distRow[j] = improved ? newDist : oldDist;
        pathRow[j] = improved ? k : pathRow[j];
    }
}

// Distributed Floyd-Warshall. Each phase k broadcasts pivot row k from its owner.
// Row k is unchanged during phase k (dist[k][k] == 0), so every rank relaxes against
// the same values as the sequential algorithm and the results are identical.
// Row k+1 is relaxed first and its broadcast is posted immediately, so that the
// transfer overlaps with the remaining local work of phase k.
void floydWarshall(std::vector<unsigned int>& localDist,
                   std::vector<unsigned int>& localPath,
                   const size_t numNodes, const RowPartition& partition, const int rank) {
    if (numNodes == 0) return;

    const size_t firstRow = partition.begin(rank);
    const size_t numLocal = partition.count(rank);

    // Double-buffered pivot rows: phase k reads buffer k % 2 while k + 1 is being received
    std::vector<unsigned int> pivots(2 * numNodes);
    auto pivotBuf = [&](const size_t k) { return pivots.data() + (k % 2) * numNodes; };

    MPI_Request request = MPI_REQUEST_NULL;
    const int firstOwner = partition.owner(0);
    if (firstOwner == rank) {
        std::copy_n(localDist.begin(), numNodes, pivotBuf(0));
    }
    MPI_Ibcast(pivotBuf(0), static_cast<int>(numNodes), MPI_UNSIGNED, firstOwner, MPI_COMM_WORLD, &request);

    for (size_t k = 0; k < numNodes; ++k) {
        MPI_Wait(&request, MPI_STATUS_IGNORE);
        const unsigned int* pivotRow = pivotBuf(k);

        // Publish pivot row k + 1 as early as possible
        if (k + 1 < numNodes) {
            const int nextOwner = partition.owner(k + 1);
            unsigned int* nextBuf = pivotBuf(k + 1);
            if (nextOwner == rank) {
                const size_t li = k + 1 - firstRow;
                unsigned int* distRow = localDist.data() + li * numNodes;
                relaxRow(distRow, localPath.data() + li * numNodes, pivotRow, distRow[k],
                         static_cast<unsigned int>(k), numNodes);
                std::copy_n(distRow, numNodes, nextBuf);
            }
            MPI_Ibcast(nextBuf, static_cast<int>(numNodes), MPI_UNSIGNED, nextOwner, MPI_COMM_WORLD, &request);
        }

        // Relax all remaining local rows. Rows k and k + 1 are skipped:
        // row k does not change in phase k and row k + 1 was already relaxed above.
        for (size_t li = 0; li < numLocal; ++li) {
            const size_t i = firstRow + li;
            if (i == k || i == k + 1) continue;

            unsigned int* distRow = localDist.data() + li * numNodes;
            relaxRow(distRow, localPath.data() + li * numNodes, pivotRow, distRow[k],
                     static_cast<unsigned int>(k), numNodes);
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

int runBenchmark(int argc, char** argv, const int rank, const int numRanks) {
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
        printf("Initializing graph...\n");
    }

    // Allocate and initialize only the rows owned by this rank
    const RowPartition partition(numNodes, numRanks);
    const size_t firstRow = partition.begin(rank);
    const size_t numLocal = partition.count(rank);

    std::vector<unsigned int> dist(numLocal * numNodes);
    std::vector<unsigned int> path(numLocal * numNodes);
    initializeDistanceMatrix(dist, numNodes, firstRow, numLocal, 1, MAX_DISTANCE);
    initializePathMatrix(path, firstRow, numLocal, numNodes);

    // Run Floyd-Warshall
    if (rank == 0) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    floydWarshall(dist, path, numNodes, partition, rank);

    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather the full distance matrix on rank 0 only when it is needed for output
    const bool needFullMatrix = printResults || validate;
    std::vector<unsigned int> fullDist;
    if (needFullMatrix) {
        std::vector<int> counts(numRanks), displs(numRanks);
        for (int r = 0; r < numRanks; ++r) {
            counts[r] = static_cast<int>(partition.count(r) * numNodes);
            displs[r] = static_cast<int>(partition.begin(r) * numNodes);
        }
        if (rank == 0) fullDist.resize(numNodes * numNodes);
        MPI_Gatherv(dist.data(), static_cast<int>(dist.size()), MPI_UNSIGNED,
                    fullDist.data(), counts.data(), displs.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    }

    if (rank != 0) return 0;

    printf("Computation time: %lld ms\n", static_cast<long long>(maxElapsed * 1000.0));

    // Calculate operations per second
    // Floyd-Warshall has O(n³) complexity
    double ops = (double)numNodes * numNodes * numNodes;
    double gflops = ops / maxElapsed / 1e9;
    printf("Performance: %.3f GOPS\n", gflops);

    // Print results for external validation (integer hash-based)
    if (printResults) {
        print_results_int(fullDist, "DistanceMatrix");
    }

    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(fullDist, numNodes);

        if (valid) {
            printf("Validation: PASSED\n");
            return 0;
        } else {
            printf("Validation: FAILED\n");
            return 1;
        }
    }

    return 0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    const int status = runBenchmark(argc, argv, rank, numRanks);

    MPI_Finalize();
    return status;
}
