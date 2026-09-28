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

// The global matrix is stored row-major: entry (row, col) lives at row * n + col,
// i.e. the sequential code's dist[idx2(j, i, n)] is source i / destination j.
//
// Parallelization: the matrix is distributed over a 2D process grid of
// pr x pc ranks; rank (r, c) owns the contiguous row block r and column block c.
// For every pivot k, the process row owning k broadcasts its slice of row k down
// the process columns and the process column owning k broadcasts its slice of
// column k along the process rows.  Both messages are only n/pc resp. n/pr
// elements long and involve only pr resp. pc ranks, which keeps the
// communication cost far below a 1D row decomposition at high rank counts.
// The pivot data of iteration k+1 is produced first and its broadcasts are
// posted before the bulk of the local update, so communication overlaps compute.

// Contiguous block decomposition of `numNodes` items over `parts` parts.
struct Blocks {
    std::vector<int> counts;
    std::vector<int> offsets;

    Blocks(const size_t numNodes, const int parts) : counts(parts), offsets(parts) {
        const size_t base = numNodes / static_cast<size_t>(parts);
        const size_t rem = numNodes % static_cast<size_t>(parts);
        size_t off = 0;
        for (int p = 0; p < parts; ++p) {
            const size_t c = base + (static_cast<size_t>(p) < rem ? 1u : 0u);
            counts[p] = static_cast<int>(c);
            offsets[p] = static_cast<int>(off);
            off += c;
        }
    }

    // Part that owns a given global index.
    int owner(const size_t index) const {
        // Blocks are monotonically ordered, so a direct formula works.
        const size_t parts = counts.size();
        const size_t base = static_cast<size_t>(offsets.back() + counts.back()) / parts;
        const size_t rem = static_cast<size_t>(offsets.back() + counts.back()) % parts;
        const size_t fat = (base + 1) * rem;  // covered by the parts with one extra item
        if (index < fat) {
            return static_cast<int>(index / (base + 1));
        }
        return static_cast<int>(rem + (index - fat) / base);
    }
};

// Generates exactly the same values as the sequential initialization, but only
// stores the sub-block owned by this rank.
void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes,
                              const size_t rowStart, const size_t rowCount,
                              const size_t colStart, const size_t colCount,
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;

    for (size_t row = 0; row < numNodes; ++row) {
        const bool mine = (row >= rowStart && row < rowStart + rowCount);
        if (!mine) {
            // Still has to advance the RNG stream to stay bit-identical.
            for (size_t col = 0; col < numNodes; ++col) {
                (void)rand_r(&seed);
            }
            continue;
        }
        unsigned int* __restrict d = dist.data() + (row - rowStart) * colCount;
        for (size_t col = 0; col < numNodes; ++col) {
            const unsigned int v =
                rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
            if (col >= colStart && col < colStart + colCount) {
                d[col - colStart] = v;
            }
        }
    }

    // Set diagonal to zero (distance from node to itself is 0)
    const size_t dStart = std::max(rowStart, colStart);
    const size_t dEnd = std::min(rowStart + rowCount, colStart + colCount);
    for (size_t g = dStart; g < dEnd; ++g) {
        dist[(g - rowStart) * colCount + (g - colStart)] = 0;
    }
}

// The sequential initialization assigns path[row][col] = row for every entry.
void initializePathMatrix(std::vector<unsigned int>& path, const size_t rowStart,
                          const size_t rowCount, const size_t colCount) {
    for (size_t r = 0; r < rowCount; ++r) {
        unsigned int* __restrict p = path.data() + r * colCount;
        const unsigned int v = static_cast<unsigned int>(rowStart + r);
        for (size_t j = 0; j < colCount; ++j) {
            p[j] = v;
        }
    }
}

// Relaxes one local row against the pivot row slice.
static inline void relaxRow(unsigned int* __restrict d, unsigned int* __restrict p,
                            const unsigned int* __restrict rowK, const size_t colCount,
                            const unsigned int distIK, const unsigned int k) {
    // Improvements become rare after the first few pivots, so the conditional
    // store is both well predicted and much cheaper than an unconditional
    // (vectorized) read-modify-write of the distance and path rows: it keeps the
    // cache lines clean and roughly halves the memory traffic of the kernel.
    for (size_t j = 0; j < colCount; ++j) {
        const unsigned int newDist = distIK + rowK[j];
        if (__builtin_expect(newDist < d[j], 0)) {
            d[j] = newDist;
            p[j] = k;
        }
    }
}

struct Grid {
    int pr = 1, pc = 1;      // process grid dimensions
    int myRow = 0, myCol = 0;
    MPI_Comm rowComm = MPI_COMM_NULL;  // ranks of the same process row (varying column)
    MPI_Comm colComm = MPI_COMM_NULL;  // ranks of the same process column (varying row)
};

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes,
                   const Blocks& rowBlocks, const Blocks& colBlocks,
                   const Grid& grid) {
    if (numNodes == 0) {
        return;
    }

    const size_t rowStart = static_cast<size_t>(rowBlocks.offsets[grid.myRow]);
    const size_t rowCount = static_cast<size_t>(rowBlocks.counts[grid.myRow]);
    const size_t colStart = static_cast<size_t>(colBlocks.offsets[grid.myCol]);
    const size_t colCount = static_cast<size_t>(colBlocks.counts[grid.myCol]);

    // Double buffers for the pivot row/column slices (current and prefetched).
    std::vector<unsigned int> rowBuf(2 * colCount), colBuf(2 * rowCount);
    unsigned int* curRow = rowBuf.data();
    unsigned int* nextRow = rowBuf.data() + colCount;
    unsigned int* curCol = colBuf.data();
    unsigned int* nextCol = colBuf.data() + rowCount;

    const int rowLen = static_cast<int>(colCount);
    const int colLen = static_cast<int>(rowCount);

    MPI_Request reqs[2];

    // Fills the pivot slices for iteration `k` from local data (only the owning
    // ranks contribute) and starts the two broadcasts.
    auto postPivot = [&](const size_t k, unsigned int* rowDst, unsigned int* colDst) {
        const int kr = rowBlocks.owner(k);
        const int kc = colBlocks.owner(k);
        if (grid.myRow == kr) {
            std::memcpy(rowDst, dist.data() + (k - rowStart) * colCount,
                        colCount * sizeof(unsigned int));
        }
        if (grid.myCol == kc) {
            const size_t lc = k - colStart;
            for (size_t i = 0; i < rowCount; ++i) {
                colDst[i] = dist[i * colCount + lc];
            }
        }
        MPI_Ibcast(rowDst, rowLen, MPI_UNSIGNED, kr, grid.colComm, &reqs[0]);
        MPI_Ibcast(colDst, colLen, MPI_UNSIGNED, kc, grid.rowComm, &reqs[1]);
    };

    postPivot(0, curRow, curCol);

    for (size_t k = 0; k < numNodes; ++k) {
        MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
        const unsigned int ku = static_cast<unsigned int>(k);

        const size_t kNext = k + 1;
        const bool haveNext = kNext < numNodes;
        const int nr = haveNext ? rowBlocks.owner(kNext) : 0;
        const int nc = haveNext ? colBlocks.owner(kNext) : 0;
        const bool rowNextMine = haveNext && grid.myRow == nr;
        const bool colNextMine = haveNext && grid.myCol == nc;
        const size_t pivotLocalRow = rowNextMine ? kNext - rowStart : 0;

        // Produce the pivot data of the next iteration first, so its broadcasts
        // can be posted before the (much larger) bulk update.  Relaxation with a
        // fixed pivot is idempotent, so the overlap between the row and the
        // column update below is harmless.
        if (rowNextMine) {
            relaxRow(dist.data() + pivotLocalRow * colCount, path.data() + pivotLocalRow * colCount,
                     curRow, colCount, curCol[pivotLocalRow], ku);
            std::memcpy(nextRow, dist.data() + pivotLocalRow * colCount,
                        colCount * sizeof(unsigned int));
        }
        if (colNextMine) {
            const size_t lc = kNext - colStart;
            const unsigned int distKJ = curRow[lc];
            for (size_t i = 0; i < rowCount; ++i) {
                const size_t off = i * colCount + lc;
                const unsigned int newDist = curCol[i] + distKJ;
                if (newDist < dist[off]) {
                    dist[off] = newDist;
                    path[off] = ku;
                }
                nextCol[i] = dist[off];
            }
        }
        if (haveNext) {
            MPI_Ibcast(nextRow, rowLen, MPI_UNSIGNED, nr, grid.colComm, &reqs[0]);
            MPI_Ibcast(nextCol, colLen, MPI_UNSIGNED, nc, grid.rowComm, &reqs[1]);
        }

        for (size_t i = 0; i < rowCount; ++i) {
            if (rowNextMine && i == pivotLocalRow) {
                continue;  // already relaxed above
            }
            relaxRow(dist.data() + i * colCount, path.data() + i * colCount, curRow, colCount,
                     curCol[i], ku);
        }

        std::swap(curRow, nextRow);
        std::swap(curCol, nextCol);
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

    int rank = 0, numRanks = 1;
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

    // Build the process grid: as square as possible, but never with more parts
    // than there are nodes in either dimension.
    Grid grid;
    {
        int pr = 1;
        for (int cand = static_cast<int>(std::sqrt(static_cast<double>(numRanks))); cand >= 1;
             --cand) {
            if (numRanks % cand == 0) {
                pr = cand;
                break;
            }
        }
        grid.pr = pr;
        grid.pc = numRanks / pr;
        // Keep both dimensions non-degenerate for very small problems.
        while (grid.pr > 1 && static_cast<size_t>(grid.pr) > numNodes) {
            grid.pr /= 2;
            grid.pc = numRanks / grid.pr;
        }
        grid.myRow = rank / grid.pc;
        grid.myCol = rank % grid.pc;
        MPI_Comm_split(MPI_COMM_WORLD, grid.myRow, grid.myCol, &grid.rowComm);
        MPI_Comm_split(MPI_COMM_WORLD, grid.myCol, grid.myRow, &grid.colComm);
    }

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("MPI ranks: %d (process grid %d x %d)\n", numRanks, grid.pr, grid.pc);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const Blocks rowBlocks(numNodes, grid.pr);
    const Blocks colBlocks(numNodes, grid.pc);
    const size_t rowStart = static_cast<size_t>(rowBlocks.offsets[grid.myRow]);
    const size_t rowCount = static_cast<size_t>(rowBlocks.counts[grid.myRow]);
    const size_t colStart = static_cast<size_t>(colBlocks.offsets[grid.myCol]);
    const size_t colCount = static_cast<size_t>(colBlocks.counts[grid.myCol]);

    // Allocate the local sub-block of both matrices
    std::vector<unsigned int> dist(rowCount * colCount);
    std::vector<unsigned int> path(rowCount * colCount);

    // Initialize
    if (rank == 0) {
        printf("Initializing graph...\n");
    }
    initializeDistanceMatrix(dist, numNodes, rowStart, rowCount, colStart, colCount, 1,
                             MAX_DISTANCE);
    initializePathMatrix(path, rowStart, rowCount, colCount);

    // Run Floyd-Warshall
    if (rank == 0) {
        printf("Computing shortest paths...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    floydWarshall(dist, path, numNodes, rowBlocks, colBlocks, grid);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long long elapsed = static_cast<long long>(duration.count());
    MPI_Allreduce(MPI_IN_PLACE, &elapsed, 1, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %lld ms\n", elapsed);

        // Calculate operations per second
        // Floyd-Warshall has O(n³) complexity
        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / (elapsed / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);
    }

    // Collect the full distance matrix on rank 0 for output / validation
    std::vector<unsigned int> fullDist;
    if (printResults || validate) {
        std::vector<int> counts(numRanks), displs(numRanks);
        int off = 0;
        for (int r = 0; r < numRanks; ++r) {
            counts[r] = rowBlocks.counts[r / grid.pc] * colBlocks.counts[r % grid.pc];
            displs[r] = off;
            off += counts[r];
        }
        std::vector<unsigned int> gathered;
        if (rank == 0) {
            gathered.resize(numNodes * numNodes);
        }
        MPI_Gatherv(dist.data(), static_cast<int>(dist.size()), MPI_UNSIGNED,
                    rank == 0 ? gathered.data() : nullptr, counts.data(), displs.data(),
                    MPI_UNSIGNED, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            fullDist.resize(numNodes * numNodes);
            for (int r = 0; r < numRanks; ++r) {
                const int br = r / grid.pc, bc = r % grid.pc;
                const size_t rs = static_cast<size_t>(rowBlocks.offsets[br]);
                const size_t rc = static_cast<size_t>(rowBlocks.counts[br]);
                const size_t cs = static_cast<size_t>(colBlocks.offsets[bc]);
                const size_t cc = static_cast<size_t>(colBlocks.counts[bc]);
                const unsigned int* src = gathered.data() + displs[r];
                for (size_t i = 0; i < rc; ++i) {
                    std::memcpy(fullDist.data() + (rs + i) * numNodes + cs, src + i * cc,
                                cc * sizeof(unsigned int));
                }
            }
        }
    }

    // Print results for external validation (integer hash-based)
    if (printResults && rank == 0) {
        print_results_int(fullDist, "DistanceMatrix");
    }

    // Validation
    int status = 0;
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
            bool valid = validateResult(fullDist, numNodes);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                status = 1;
            }
        }
        MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    MPI_Comm_free(&grid.rowComm);
    MPI_Comm_free(&grid.colComm);
    MPI_Finalize();
    return status;
}
