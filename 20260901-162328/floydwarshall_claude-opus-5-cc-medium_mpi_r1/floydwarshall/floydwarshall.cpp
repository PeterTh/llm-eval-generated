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

// The flattened layout is row-major with respect to the algorithm's access pattern:
// dist[idx2(j, i, n)] == dist[i * n + j], i.e. row i, column j. The distance matrix is
// therefore distributed by contiguous blocks of rows over the MPI ranks; every rank owns
// rows [rowBegin, rowEnd) of both the distance and the path matrix.

// Row distribution: block distribution with the remainder spread over the first ranks.
inline size_t rowBeginOf(const size_t numNodes, const int rank, const int numRanks) noexcept {
    const size_t base = numNodes / static_cast<size_t>(numRanks);
    const size_t rem = numNodes % static_cast<size_t>(numRanks);
    const size_t r = static_cast<size_t>(rank);
    return r * base + std::min(r, rem);
}

void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes,
                              const size_t rowBegin, const size_t numLocalRows,
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;

    // Reproduce the sequential random stream: consume the values belonging to the rows
    // owned by lower ranks, then generate the local ones.
    const size_t skip = rowBegin * numNodes;
    for (size_t i = 0; i < skip; ++i) {
        (void)rand_r(&seed);
    }

    const size_t localElements = numLocalRows * numNodes;
    for (size_t i = 0; i < localElements; ++i) {
        dist[i] = rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
    }

    // Set diagonal to zero (distance from node to itself is 0)
    for (size_t i = 0; i < numLocalRows; ++i) {
        dist[i * numNodes + (rowBegin + i)] = 0;
    }
}

void initializePathMatrix(std::vector<unsigned int>& path, const size_t numNodes,
                          const size_t rowBegin, const size_t numLocalRows) {
    // The sequential initialization assigns path[row][col] = row for every entry.
    for (size_t i = 0; i < numLocalRows; ++i) {
        std::fill_n(path.data() + i * numNodes, numNodes, static_cast<unsigned int>(rowBegin + i));
    }
}

// Relax one row against the pivot row. Branchless so that the compiler can vectorize the
// conditional distance/path update into blended stores.
static inline void relaxRow(unsigned int* __restrict distRow, unsigned int* __restrict pathRow,
                            const unsigned int* __restrict pivotRow, const unsigned int distIK,
                            const unsigned int k, const size_t numNodes) noexcept {
    for (size_t j = 0; j < numNodes; ++j) {
        const unsigned int newDist = distIK + pivotRow[j];
        const bool update = newDist < distRow[j];
        distRow[j] = update ? newDist : distRow[j];
        pathRow[j] = update ? k : pathRow[j];
    }
}

// Upper bound for the number of pivot rows that are communicated and applied as one panel.
constexpr size_t MAX_PANEL_ROWS = 64;

// Pivot rows are handled in panels: blocking the k-loop amortizes the broadcast latency and
// keeps the local rows in cache while several pivot rows are applied to them. The panel owner
// however has to resolve the panel by itself while the other ranks only apply it, so the panel
// must stay small relative to the number of rows per rank to keep that extra work off the
// critical path.
inline size_t choosePanelRows(const size_t numNodes, const int numRanks) noexcept {
    const size_t rowsPerRank = numNodes / static_cast<size_t>(numRanks);
    const size_t panel = rowsPerRank / 16;
    return std::clamp(panel, static_cast<size_t>(1), MAX_PANEL_ROWS);
}

// A panel is a run of consecutive pivot rows that is owned by a single rank.
struct Panel {
    size_t k0;    // first pivot row
    size_t kb;    // number of pivot rows
    int owner;    // rank owning these rows
};

inline Panel panelAt(const size_t k0, const size_t numNodes, const int numRanks,
                     const size_t panelRows, const std::vector<int>& rowOwner) noexcept {
    const int owner = rowOwner[k0];
    const size_t ownerEnd = rowBeginOf(numNodes, owner + 1, numRanks);
    return Panel{k0, std::min(std::min(k0 + panelRows, ownerEnd), numNodes) - k0, owner};
}

// Number of local rows processed together against one panel. Chosen so that the touched
// distance and path rows stay resident in the private caches.
inline size_t rowBlockSize(const size_t numNodes) noexcept {
    const size_t rows = 49152 / (numNodes * sizeof(unsigned int) * 2);
    return std::clamp(rows, static_cast<size_t>(1), static_cast<size_t>(64));
}

// Apply all pivot rows of a panel to the local rows [rowsBegin, rowsEnd). Rows are relaxed in
// blocks, but every single row still sees the pivot rows in ascending k order, which makes the
// result bit-identical to the sequential triple loop (rows are independent within one k step).
static void applyPanel(unsigned int* __restrict dist, unsigned int* __restrict path,
                       const unsigned int* __restrict panel, const size_t numNodes,
                       const size_t rowBegin, const size_t k0, const size_t kb,
                       const size_t rowsBegin, const size_t rowsEnd) noexcept {
    if (rowsBegin >= rowsEnd) {
        return;
    }
    const size_t rowBlock = rowBlockSize(numNodes);
    for (size_t i0 = rowsBegin; i0 < rowsEnd; i0 += rowBlock) {
        const size_t i1 = std::min(i0 + rowBlock, rowsEnd);
        for (size_t k = k0; k < k0 + kb; ++k) {
            const unsigned int* pivotRow = panel + (k - k0) * numNodes;
            for (size_t i = i0; i < i1; ++i) {
                unsigned int* distRow = dist + (i - rowBegin) * numNodes;
                unsigned int* pathRow = path + (i - rowBegin) * numNodes;
                relaxRow(distRow, pathRow, pivotRow, distRow[k], static_cast<unsigned int>(k), numNodes);
            }
        }
    }
}

// Resolve the panel rows against each other (they are all owned by this rank) and snapshot every
// pivot row at the point in time at which the sequential algorithm would use it.
static void preparePanel(unsigned int* __restrict dist, unsigned int* __restrict path,
                         unsigned int* __restrict panel, const size_t numNodes,
                         const size_t rowBegin, const size_t k0, const size_t kb) noexcept {
    for (size_t k = k0; k < k0 + kb; ++k) {
        unsigned int* snapshot = panel + (k - k0) * numNodes;
        std::memcpy(snapshot, dist + (k - rowBegin) * numNodes, numNodes * sizeof(unsigned int));
        for (size_t i = k0; i < k0 + kb; ++i) {
            // Row k is a fixed point of its own iteration (dist[k][k] == 0).
            if (i == k) {
                continue;
            }
            unsigned int* distRow = dist + (i - rowBegin) * numNodes;
            unsigned int* pathRow = path + (i - rowBegin) * numNodes;
            relaxRow(distRow, pathRow, snapshot, distRow[k], static_cast<unsigned int>(k), numNodes);
        }
    }
}

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes,
                   const size_t rowBegin,
                   const size_t numLocalRows,
                   const std::vector<int>& rowOwner,
                   const int numRanks,
                   MPI_Comm comm) {
    // Distributed Floyd-Warshall over a 1D block-row distribution. The pivot rows of the
    // k-loop are handled in panels: the owner of a panel resolves it locally and broadcasts
    // the snapshots of its pivot rows, then every rank applies the whole panel to its own
    // rows. The broadcast of the next panel is overlapped with the application of the current
    // one -- the owner of the next panel brings those rows up to date and prepares them first,
    // so its non-blocking broadcast can proceed while all ranks are still computing.
    if (numNodes == 0) {
        return;
    }

    const size_t rowEnd = rowBegin + numLocalRows;

    const size_t panelRows = choosePanelRows(numNodes, numRanks);
    std::vector<unsigned int> panelBuf(2 * panelRows * numNodes);
    unsigned int* panel = panelBuf.data();
    unsigned int* panelNext = panelBuf.data() + panelRows * numNodes;

    unsigned int* const distData = dist.data();
    unsigned int* const pathData = path.data();

    // Prime the pipeline with the first panel.
    Panel cur = panelAt(0, numNodes, numRanks, panelRows, rowOwner);
    if (cur.k0 >= rowBegin && cur.k0 < rowEnd) {
        preparePanel(distData, pathData, panel, numNodes, rowBegin, cur.k0, cur.kb);
    }
    MPI_Bcast(panel, static_cast<int>(cur.kb * numNodes), MPI_UNSIGNED, cur.owner, comm);

    while (cur.kb > 0) {
        const size_t nextK0 = cur.k0 + cur.kb;
        const bool hasNext = nextK0 < numNodes;
        Panel next{numNodes, 0, 0};
        MPI_Request req = MPI_REQUEST_NULL;

        // Rows already handled in this iteration: the current panel plus, on the owner of the
        // next panel, the next panel's rows.
        size_t doneEnd = nextK0;

        if (hasNext) {
            next = panelAt(nextK0, numNodes, numRanks, panelRows, rowOwner);
            if (next.k0 >= rowBegin && next.k0 < rowEnd) {
                // Bring the next panel's rows up to date with the current panel, then resolve
                // them so that the broadcast can start before the bulk of the work is done.
                applyPanel(distData, pathData, panel, numNodes, rowBegin, cur.k0, cur.kb,
                           next.k0, next.k0 + next.kb);
                preparePanel(distData, pathData, panelNext, numNodes, rowBegin, next.k0, next.kb);
                doneEnd = next.k0 + next.kb;
            }
            MPI_Ibcast(panelNext, static_cast<int>(next.kb * numNodes), MPI_UNSIGNED, next.owner,
                       comm, &req);
        }

        // Apply the current panel to the remaining local rows, i.e. everything outside
        // [cur.k0, doneEnd).
        const size_t skipBegin = std::clamp(cur.k0, rowBegin, rowEnd);
        const size_t skipEnd = std::clamp(doneEnd, rowBegin, rowEnd);
        applyPanel(distData, pathData, panel, numNodes, rowBegin, cur.k0, cur.kb, rowBegin, skipBegin);
        applyPanel(distData, pathData, panel, numNodes, rowBegin, cur.k0, cur.kb, skipEnd, rowEnd);

        if (hasNext) {
            MPI_Wait(&req, MPI_STATUS_IGNORE);
            std::swap(panel, panelNext);
        }
        cur = next;
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
    const bool isRoot = (rank == 0);

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
            if (isRoot) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            if (isRoot) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (isRoot) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", numRanks);
    }

    // Row distribution over the ranks
    const size_t rowBegin = rowBeginOf(numNodes, rank, numRanks);
    const size_t rowEnd = rowBeginOf(numNodes, rank + 1, numRanks);
    const size_t numLocalRows = rowEnd - rowBegin;

    std::vector<int> rowOwner(numNodes);
    std::vector<int> sendCounts(numRanks);
    std::vector<int> displs(numRanks);
    for (int r = 0; r < numRanks; ++r) {
        const size_t begin = rowBeginOf(numNodes, r, numRanks);
        const size_t end = rowBeginOf(numNodes, r + 1, numRanks);
        sendCounts[r] = static_cast<int>((end - begin) * numNodes);
        displs[r] = static_cast<int>(begin * numNodes);
        for (size_t i = begin; i < end; ++i) {
            rowOwner[i] = r;
        }
    }

    // Allocate local matrix blocks
    std::vector<unsigned int> dist(numLocalRows * numNodes);
    std::vector<unsigned int> path(numLocalRows * numNodes);

    // Initialize
    if (isRoot) {
        printf("Initializing graph...\n");
    }
    initializeDistanceMatrix(dist, numNodes, rowBegin, numLocalRows, 1, MAX_DISTANCE);
    initializePathMatrix(path, numNodes, rowBegin, numLocalRows);

    // Run Floyd-Warshall
    if (isRoot) {
        printf("Computing shortest paths...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    floydWarshall(dist, path, numNodes, rowBegin, numLocalRows, rowOwner, numRanks, MPI_COMM_WORLD);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    long long localMs = static_cast<long long>(duration.count());
    long long elapsedMs = localMs;
    MPI_Reduce(&localMs, &elapsedMs, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (isRoot) {
        printf("Computation time: %lld ms\n", elapsedMs);

        // Calculate operations per second
        // Floyd-Warshall has O(n³) complexity
        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / (elapsedMs / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);
    }

    int exitCode = 0;
    if (printResults || validate) {
        // Collect the full distance matrix on the root for output and validation.
        std::vector<unsigned int> fullDist(isRoot ? numNodes * numNodes : 0);
        MPI_Gatherv(dist.data(), static_cast<int>(numLocalRows * numNodes), MPI_UNSIGNED,
                    isRoot ? fullDist.data() : nullptr, sendCounts.data(), displs.data(),
                    MPI_UNSIGNED, 0, MPI_COMM_WORLD);

        if (isRoot) {
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
                } else {
                    printf("Validation: FAILED\n");
                    exitCode = 1;
                }
            }
        }
        MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return exitCode;
}
