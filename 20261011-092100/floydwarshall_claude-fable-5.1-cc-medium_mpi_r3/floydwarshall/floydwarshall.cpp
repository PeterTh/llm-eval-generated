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

// ---------------------------------------------------------------------------
// MPI decomposition: the n x n matrices are stored row-major in memory
// (element (i,j) at i*n+j, i.e. idx2(j,i,n)).  Rows are distributed in
// contiguous blocks over the ranks.
//
// Iteration k of Floyd-Warshall only needs row k (the "pivot" row) plus the
// row being updated, so consecutive pivots k..k+B-1 are grouped into a block
// that is owned by a single rank, computed there, and broadcast once.  Every
// rank then applies all B pivots to each of its rows in one pass, which cuts
// the memory traffic of the kernel by a factor of B.  The update order of
// every individual element is exactly that of the sequential algorithm, so
// both the distance and the path matrix are bit-identical to the original.
// The broadcast of the next pivot block is overlapped with the computation
// of the current one.
// ---------------------------------------------------------------------------

constexpr size_t PIVOT_BLOCK = 16;    // pivots per block (B)
constexpr size_t COL_CHUNK = 1024;    // columns processed per pass (cache tile)

struct RowPartition {
    int rank = 0;
    int size = 1;
    size_t numNodes = 0;
    std::vector<size_t> rowStart;  // size+1 entries; rank r owns [rowStart[r], rowStart[r+1])

    RowPartition(int rank_, int size_, size_t n) : rank(rank_), size(size_), numNodes(n), rowStart(size_ + 1) {
        for (int r = 0; r <= size_; ++r) {
            rowStart[r] = (n * static_cast<size_t>(r)) / static_cast<size_t>(size_);
        }
    }

    size_t localStart() const { return rowStart[rank]; }
    size_t localEnd() const { return rowStart[rank + 1]; }
    size_t localRows() const { return localEnd() - localStart(); }
};

struct PivotBlock {
    size_t kStart;
    size_t kEnd;
    int owner;
};

// Generates exactly the same values as the original sequential generator for
// the rows owned by this rank (the sequential stream is replayed and the
// entries preceding the local block are discarded).
void initializeDistanceMatrix(std::vector<unsigned int>& dist, const RowPartition& part,
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;
    const size_t n = part.numNodes;

    const size_t firstIdx = part.localStart() * n;
    const size_t lastIdx = part.localEnd() * n;

    for (size_t i = 0; i < firstIdx; ++i) {
        (void)rand_r(&seed);
    }
    for (size_t i = firstIdx; i < lastIdx; ++i) {
        dist[i - firstIdx] = rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
    }

    // Set diagonal to zero (distance from node to itself is 0)
    for (size_t i = part.localStart(); i < part.localEnd(); ++i) {
        dist[(i - part.localStart()) * n + i] = 0;
    }
}

// The original initialisation results in path[i][j] = i for all i, j
// (memory index i*n+j), which is what we reproduce for the local rows.
void initializePathMatrix(std::vector<unsigned int>& path, const RowPartition& part) {
    const size_t n = part.numNodes;
    for (size_t i = part.localStart(); i < part.localEnd(); ++i) {
        unsigned int* row = &path[(i - part.localStart()) * n];
        std::fill(row, row + n, static_cast<unsigned int>(i));
    }
}

// Relax len entries of row i against the pivot row k:
//   row[j] = min(row[j], dik + rowk[j]), path updated on strict improvement.
// Written with unconditional stores so the compiler can vectorise it.
static inline void relaxRow(unsigned int* __restrict row, unsigned int* __restrict prow,
                            const unsigned int* __restrict rowk, const unsigned int dik,
                            const unsigned int k, const size_t len) {
    for (size_t j = 0; j < len; ++j) {
        const unsigned int distIJ = row[j];
        const unsigned int newDist = dik + rowk[j];
        const bool better = newDist < distIJ;
        row[j] = better ? newDist : distIJ;
        prow[j] = better ? k : prow[j];
    }
}

// Applies the pivots kStart..kStart+nb-1 (in order) to one local row.
// piv holds pivot t (row kStart+t as it is after iterations < kStart+t) at
// piv + t*n.  Re-applying a pivot to an already relaxed element is a no-op
// (the comparison is strict), which keeps the chunked pass exact.
static void applyPivotBlock(unsigned int* __restrict row, unsigned int* __restrict prow,
                            const unsigned int* __restrict piv, const size_t kStart,
                            const size_t nb, const size_t n, unsigned int* __restrict dik) {
    // 1. Columns kStart..kStart+nb-1 first, to obtain dist[i][k+t] exactly as
    //    it is after iterations < k+t (the "dik" value of each iteration).
    for (size_t t = 0; t < nb; ++t) {
        dik[t] = row[kStart + t];
        relaxRow(row + kStart, prow + kStart, piv + t * n + kStart, dik[t],
                 static_cast<unsigned int>(kStart + t), nb);
    }
    // 2. Full row, in cache-sized column chunks, all pivots per chunk.
    for (size_t j0 = 0; j0 < n; j0 += COL_CHUNK) {
        const size_t len = std::min(COL_CHUNK, n - j0);
        for (size_t t = 0; t < nb; ++t) {
            relaxRow(row + j0, prow + j0, piv + t * n + j0, dik[t],
                     static_cast<unsigned int>(kStart + t), len);
        }
    }
}

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const RowPartition& part) {
    const size_t n = part.numNodes;
    const size_t r0 = part.localStart();
    const size_t r1 = part.localEnd();
    if (n == 0) return;

    // Pivot blocks: consecutive pivots, never crossing a rank boundary.
    std::vector<PivotBlock> blocks;
    for (int r = 0; r < part.size; ++r) {
        for (size_t k = part.rowStart[r]; k < part.rowStart[r + 1]; k += PIVOT_BLOCK) {
            blocks.push_back({k, std::min(k + PIVOT_BLOCK, part.rowStart[r + 1]), r});
        }
    }

    std::vector<unsigned int> pivBuf[2] = {std::vector<unsigned int>(PIVOT_BLOCK * n),
                                           std::vector<unsigned int>(PIVOT_BLOCK * n)};
    std::vector<unsigned int> dik(PIVOT_BLOCK);

    auto localRowPtr = [&](size_t i) { return &dist[(i - r0) * n]; };
    auto localPathPtr = [&](size_t i) { return &path[(i - r0) * n]; };

    // Owner only: bring rows kStart..kEnd-1 (which already have every earlier
    // pivot applied) up to date within the block and store them as pivots.
    auto computePivotBlock = [&](const PivotBlock& b, unsigned int* piv) {
        const size_t nb = b.kEnd - b.kStart;
        for (size_t t = 0; t < nb; ++t) {
            unsigned int* row = localRowPtr(b.kStart + t);
            unsigned int* prow = localPathPtr(b.kStart + t);
            for (size_t s = 0; s < t; ++s) {
                relaxRow(row, prow, piv + s * n, row[b.kStart + s],
                         static_cast<unsigned int>(b.kStart + s), n);
            }
            std::copy(row, row + n, piv + t * n);
        }
    };

    // Bootstrap: pivot block 0 via blocking broadcast.
    if (blocks[0].owner == part.rank) {
        computePivotBlock(blocks[0], pivBuf[0].data());
    }
    MPI_Bcast(pivBuf[0].data(), static_cast<int>((blocks[0].kEnd - blocks[0].kStart) * n),
              MPI_UNSIGNED, blocks[0].owner, MPI_COMM_WORLD);

    for (size_t b = 0; b < blocks.size(); ++b) {
        const PivotBlock& cur = blocks[b];
        const size_t nb = cur.kEnd - cur.kStart;
        const unsigned int* piv = pivBuf[b & 1].data();
        unsigned int* nextPiv = pivBuf[(b + 1) & 1].data();

        MPI_Request req = MPI_REQUEST_NULL;
        const bool hasNext = (b + 1 < blocks.size());
        size_t skipStart = 0, skipEnd = 0;  // rows handled early (next pivot block)

        if (hasNext) {
            const PivotBlock& nxt = blocks[b + 1];
            if (nxt.owner == part.rank) {
                // Finish the next pivot rows first so they can be sent early.
                for (size_t i = nxt.kStart; i < nxt.kEnd; ++i) {
                    applyPivotBlock(localRowPtr(i), localPathPtr(i), piv, cur.kStart, nb, n, dik.data());
                }
                computePivotBlock(nxt, nextPiv);
                skipStart = nxt.kStart;
                skipEnd = nxt.kEnd;
            }
            MPI_Ibcast(nextPiv, static_cast<int>((nxt.kEnd - nxt.kStart) * n), MPI_UNSIGNED,
                       nxt.owner, MPI_COMM_WORLD, &req);
        }

        // Relax all remaining local rows with the current pivot block.
        size_t sinceProgress = 0;
        for (size_t i = r0; i < r1; ++i) {
            if (i >= skipStart && i < skipEnd) continue;
            applyPivotBlock(localRowPtr(i), localPathPtr(i), piv, cur.kStart, nb, n, dik.data());
            if (hasNext && ++sinceProgress == 8) {
                // Give the MPI library a chance to progress the broadcast.
                int done = 0;
                MPI_Test(&req, &done, MPI_STATUS_IGNORE);
                sinceProgress = 0;
            }
        }

        if (hasNext) {
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

// Collects the distributed rows into a full matrix on rank 0.
void gatherMatrix(const std::vector<unsigned int>& local, std::vector<unsigned int>& full,
                  const RowPartition& part) {
    const size_t n = part.numNodes;
    std::vector<int> counts(part.size), displs(part.size);
    for (int r = 0; r < part.size; ++r) {
        counts[r] = static_cast<int>((part.rowStart[r + 1] - part.rowStart[r]) * n);
        displs[r] = static_cast<int>(part.rowStart[r] * n);
    }
    if (part.rank == 0) full.resize(n * n);
    MPI_Gatherv(local.data(), static_cast<int>(part.localRows() * n), MPI_UNSIGNED,
                part.rank == 0 ? full.data() : nullptr, counts.data(), displs.data(), MPI_UNSIGNED,
                0, MPI_COMM_WORLD);
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
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
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
            if (isRoot) printUsage(argv[0]);
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
        printf("MPI ranks: %d\n", size);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const RowPartition part(rank, size, numNodes);

    // Allocate local row blocks of the matrices
    std::vector<unsigned int> dist(part.localRows() * numNodes);
    std::vector<unsigned int> path(part.localRows() * numNodes);

    // Initialize
    if (isRoot) printf("Initializing graph...\n");
    initializeDistanceMatrix(dist, part, 1, MAX_DISTANCE);
    initializePathMatrix(path, part);

    // Run Floyd-Warshall
    if (isRoot) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    floydWarshall(dist, path, part);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (isRoot) {
        printf("Computation time: %ld ms\n", static_cast<long>(duration.count()));

        // Calculate operations per second
        // Floyd-Warshall has O(n³) complexity
        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);
    }

    int exitCode = 0;
    if (printResults || validate) {
        std::vector<unsigned int> fullDist;
        gatherMatrix(dist, fullDist, part);

        // Print results for external validation (integer hash-based)
        if (printResults && isRoot) {
            print_results_int(fullDist, "DistanceMatrix");
        }

        // Validation
        if (validate) {
            if (isRoot) {
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
    }

    MPI_Finalize();
    return exitCode;
}
