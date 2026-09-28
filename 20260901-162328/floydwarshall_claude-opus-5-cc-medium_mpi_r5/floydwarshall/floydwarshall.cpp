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

// ---------------------------------------------------------------------------
// Row-block decomposition helpers
//
// The matrix is stored row-major with respect to the source node: element
// (i, j) lives at flat index i * n + j (= idx2(j, i, n)).  Each MPI rank owns a
// contiguous block of source rows, so the innermost j-loop always runs over a
// contiguous, locally owned stretch of memory.
// ---------------------------------------------------------------------------
struct RowBlock {
    size_t start; // first row owned by this rank
    size_t count; // number of rows owned by this rank
};

inline RowBlock rowBlockFor(const int rank, const int numRanks, const size_t numNodes) noexcept {
    const size_t base = numNodes / static_cast<size_t>(numRanks);
    const size_t rem = numNodes % static_cast<size_t>(numRanks);
    const size_t r = static_cast<size_t>(rank);
    const size_t count = base + (r < rem ? 1 : 0);
    const size_t start = r * base + std::min(r, rem);
    return {start, count};
}

inline int ownerOfRow(const size_t row, const int numRanks, const size_t numNodes) noexcept {
    const size_t base = numNodes / static_cast<size_t>(numRanks);
    const size_t rem = numNodes % static_cast<size_t>(numRanks);
    const size_t split = rem * (base + 1);
    if (row < split) {
        return static_cast<int>(row / (base + 1));
    }
    return static_cast<int>(rem + (row - split) / base);
}

// ---------------------------------------------------------------------------
// Initialization
//
// The random sequence is strictly sequential (rand_r on a shared seed), so every
// rank replays the whole O(n^2) stream but only stores the rows it owns.  This
// reproduces the serial values bit-for-bit without any communication.
// ---------------------------------------------------------------------------
void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes,
                              const RowBlock& block,
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;

    const size_t first = block.start * numNodes;
    const size_t last = first + block.count * numNodes;

    for (size_t i = 0; i < numNodes * numNodes; ++i) {
        const unsigned int value = rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
        if (i >= first && i < last) {
            dist[i - first] = value;
        }
    }

    // Set diagonal to zero (distance from node to itself is 0)
    for (size_t r = 0; r < block.count; ++r) {
        const size_t row = block.start + r;
        dist[r * numNodes + row] = 0;
    }
}

void initializePathMatrix(std::vector<unsigned int>& path, const size_t numNodes,
                          const RowBlock& block) {
    // The serial initialization ends up assigning path[i][j] = i for all i, j.
    for (size_t r = 0; r < block.count; ++r) {
        unsigned int* row = &path[r * numNodes];
        const unsigned int value = static_cast<unsigned int>(block.start + r);
        for (size_t j = 0; j < numNodes; ++j) {
            row[j] = value;
        }
    }
}

// Branchless relaxation of one source row against the broadcast pivot row.
// Written so the compiler can emit a vectorized compare/blend sequence.
static inline void relaxRow(unsigned int* __restrict distRow, unsigned int* __restrict pathRow,
                            const unsigned int* __restrict pivotRow, const unsigned int distIK,
                            const unsigned int k, const size_t numNodes) noexcept {
    for (size_t j = 0; j < numNodes; ++j) {
        const unsigned int newDist = distIK + pivotRow[j];
        const bool better = newDist < distRow[j];
        distRow[j] = better ? newDist : distRow[j];
        pathRow[j] = better ? k : pathRow[j];
    }
}

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes,
                   const RowBlock& block,
                   const int numRanks) {
    if (numNodes == 0) {
        return;
    }

    // Double-buffered receive space for the pivot rows so that the broadcast of
    // row k+1 can be posted (and progress) while row k is still being applied.
    std::vector<unsigned int> pivotBuf(2 * numNodes);
    MPI_Request requests[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};
    const unsigned int* pivotPtr[2] = {nullptr, nullptr};

    auto localRow = [&](const size_t row) { return &dist[(row - block.start) * numNodes]; };
    auto ownsRow = [&](const size_t row) { return row >= block.start && row < block.start + block.count; };

    // Prime the pipeline with row 0, which is already final.
    {
        const int root = ownerOfRow(0, numRanks, numNodes);
        unsigned int* buf = ownsRow(0) ? localRow(0) : &pivotBuf[0];
        pivotPtr[0] = buf;
        MPI_Ibcast(buf, static_cast<int>(numNodes), MPI_UNSIGNED, root, MPI_COMM_WORLD, &requests[0]);
    }

    for (size_t k = 0; k < numNodes; ++k) {
        const size_t slot = k % 2;
        MPI_Wait(&requests[slot], MPI_STATUS_IGNORE);
        const unsigned int* __restrict pivotRow = pivotPtr[slot];

        const size_t next = k + 1;

        if (next < numNodes) {
            const int nextRoot = ownerOfRow(next, numRanks, numNodes);
            unsigned int* buf;
            if (ownsRow(next)) {
                // Finalize row k+1 first so it can be handed to the network
                // while the remaining local rows are still being relaxed.
                unsigned int* distRow = localRow(next);
                relaxRow(distRow, &path[(next - block.start) * numNodes], pivotRow,
                         distRow[k], static_cast<unsigned int>(k), numNodes);
                buf = distRow;
            } else {
                buf = &pivotBuf[(next % 2) * numNodes];
            }
            pivotPtr[next % 2] = buf;
            MPI_Ibcast(buf, static_cast<int>(numNodes), MPI_UNSIGNED, nextRoot, MPI_COMM_WORLD,
                       &requests[next % 2]);
        }

        for (size_t r = 0; r < block.count; ++r) {
            const size_t i = block.start + r;
            if (i == next || i == k) {
                // row k+1 was already relaxed above; relaxing the pivot row
                // against itself is a no-op (dist[k][k] == 0)
                continue;
            }
            unsigned int* __restrict distRow = &dist[r * numNodes];
            const unsigned int distIK = distRow[k];
            relaxRow(distRow, &path[r * numNodes], pivotRow, distIK, static_cast<unsigned int>(k),
                     numNodes);
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
        printf("MPI ranks: %d\n", numRanks);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const RowBlock block = rowBlockFor(rank, numRanks, numNodes);

    // Allocate the locally owned row block of each matrix
    std::vector<unsigned int> dist(block.count * numNodes);
    std::vector<unsigned int> path(block.count * numNodes);

    // Initialize
    if (rank == 0) {
        printf("Initializing graph...\n");
    }
    initializeDistanceMatrix(dist, numNodes, block, 1, MAX_DISTANCE);
    initializePathMatrix(path, numNodes, block);

    // Run Floyd-Warshall
    if (rank == 0) {
        printf("Computing shortest paths...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    floydWarshall(dist, path, numNodes, block, numRanks);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate operations per second
        // Floyd-Warshall has O(n³) complexity
        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);
    }

    // Gather the full distance matrix on rank 0 only if it is actually needed.
    std::vector<unsigned int> fullDist;
    if (printResults || validate) {
        std::vector<int> counts;
        std::vector<int> displs;
        if (rank == 0) {
            fullDist.resize(numNodes * numNodes);
            counts.resize(numRanks);
            displs.resize(numRanks);
            for (int r = 0; r < numRanks; ++r) {
                const RowBlock rb = rowBlockFor(r, numRanks, numNodes);
                counts[r] = static_cast<int>(rb.count * numNodes);
                displs[r] = static_cast<int>(rb.start * numNodes);
            }
        }
        MPI_Gatherv(dist.data(), static_cast<int>(block.count * numNodes), MPI_UNSIGNED,
                    rank == 0 ? fullDist.data() : nullptr, counts.data(), displs.data(),
                    MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    }

    // Print results for external validation (integer hash-based)
    if (printResults && rank == 0) {
        print_results_int(fullDist, "DistanceMatrix");
    }

    // Validation
    int exitCode = 0;
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
            bool valid = validateResult(fullDist, numNodes);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
        MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return exitCode;
}
