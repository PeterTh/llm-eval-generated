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
// Row distribution: the kernel addresses the matrix as dist[idx2(j, i, n)] ==
// dist[i * n + j], i.e. row i (source node) is contiguous in memory. Rows are
// distributed in contiguous blocks over the MPI ranks; rank r owns rows
// [rowStart(r), rowStart(r) + rowCount(r)).
// ---------------------------------------------------------------------------
struct RowDist {
    size_t n = 0;
    int size = 1;
    int rank = 0;

    size_t rowStart(const int r) const noexcept { return (n * static_cast<size_t>(r)) / size; }
    size_t rowCount(const int r) const noexcept { return rowStart(r + 1) - rowStart(r); }
    size_t localStart() const noexcept { return rowStart(rank); }
    size_t localCount() const noexcept { return rowCount(rank); }

    // Owner rank of global row i.
    int owner(const size_t i) const noexcept {
        // Initial guess, then fix up (block sizes differ by at most one).
        int r = static_cast<int>((i * static_cast<size_t>(size)) / n);
        if (r >= size) r = size - 1;
        while (r > 0 && rowStart(r) > i) --r;
        while (r + 1 < size && rowStart(r + 1) <= i) ++r;
        return r;
    }
};

// Every rank replays the identical rand_r stream of the original program
// (deterministic, seed 42) and keeps only the rows it owns. This costs O(n^2)
// per rank, is outside the timed region, and needs no communication.
void initializeDistanceMatrix(std::vector<unsigned int>& dist, const RowDist& rd,
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    const size_t numNodes = rd.n;
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;

    const size_t first = rd.localStart() * numNodes;
    const size_t last = first + rd.localCount() * numNodes;

    // Skip the random values belonging to rows owned by lower ranks.
    for (size_t i = 0; i < first; ++i) {
        (void)rand_r(&seed);
    }
    for (size_t i = first; i < last; ++i) {
        dist[i - first] = rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
    }

    // Set diagonal to zero (distance from node to itself is 0)
    const size_t r0 = rd.localStart();
    for (size_t i = 0; i < rd.localCount(); ++i) {
        dist[i * numNodes + (r0 + i)] = 0;
    }
}

// The original initialization results in path[i*n + j] == i for all i, j
// (both writes in the original loop store the row index).
void initializePathMatrix(std::vector<unsigned int>& path, const RowDist& rd) {
    const size_t numNodes = rd.n;
    const size_t r0 = rd.localStart();
    for (size_t li = 0; li < rd.localCount(); ++li) {
        unsigned int* row = path.data() + li * numNodes;
        const unsigned int v = static_cast<unsigned int>(r0 + li);
        for (size_t j = 0; j < numNodes; ++j) {
            row[j] = v;
        }
    }
}

// Apply the pivots k in [k0, k1) to one local row (dist/path row of node i).
// The pivot rows are stored contiguously in tile (row k at tile + (k-k0)*n)
// and already carry all updates for pivots < k. Processing the pivots in
// increasing order on each element reproduces exactly the sequence of
// min-updates the classic algorithm performs, so results are bitwise equal.
//
// Several pivots are fused into one sweep over the row: the element d[j]
// is loaded once, relaxed with pivots k, k+1, ... in order and stored once.
// The required values dist[i][k+u] (after pivots < k+u) are computed up
// front with the same chain of updates the sweep would apply to them.
template <size_t U>
static inline void relaxRowFused(unsigned int* __restrict d, unsigned int* __restrict p,
                                 const unsigned int* __restrict tile, const size_t numNodes,
                                 const size_t k) {
    unsigned int dik[U];
    unsigned int kk[U];
    for (size_t u = 0; u < U; ++u) {
        unsigned int v = d[k + u];
        for (size_t w = 0; w < u; ++w) {
            const unsigned int cand = dik[w] + tile[w * numNodes + (k + u)];
            v = cand < v ? cand : v;
        }
        dik[u] = v;
        kk[u] = static_cast<unsigned int>(k + u);
    }
    for (size_t j = 0; j < numNodes; ++j) {
        unsigned int distIJ = d[j];
        unsigned int pathIJ = p[j];
        for (size_t u = 0; u < U; ++u) {
            const unsigned int newDist = dik[u] + tile[u * numNodes + j];
            const bool better = newDist < distIJ;
            distIJ = better ? newDist : distIJ;
            pathIJ = better ? kk[u] : pathIJ;
        }
        d[j] = distIJ;
        p[j] = pathIJ;
    }
}

static inline void relaxRow(unsigned int* __restrict d, unsigned int* __restrict p,
                            const unsigned int* __restrict tile, const size_t numNodes,
                            const size_t k0, const size_t k1) {
    size_t k = k0;
    for (; k + 4 <= k1; k += 4) {
        relaxRowFused<4>(d, p, tile + (k - k0) * numNodes, numNodes, k);
    }
    for (; k + 2 <= k1; k += 2) {
        relaxRowFused<2>(d, p, tile + (k - k0) * numNodes, numNodes, k);
    }
    for (; k < k1; ++k) {
        relaxRowFused<1>(d, p, tile + (k - k0) * numNodes, numNodes, k);
    }
}

// Apply pivots [k0, k1) to local rows [lBegin, lEnd). If req is non-null,
// the pending broadcast is polled between rows so that MPI makes progress on
// it while we compute (Open MPI only progresses in MPI calls).
static inline void relaxRows(unsigned int* __restrict dist, unsigned int* __restrict path,
                             const unsigned int* __restrict tile, const size_t numNodes,
                             const size_t k0, const size_t k1, const size_t lBegin,
                             const size_t lEnd, MPI_Request* req) {
    for (size_t li = lBegin; li < lEnd; ++li) {
        relaxRow(dist + li * numNodes, path + li * numNodes, tile, numNodes, k0, k1);
        if (req != nullptr && *req != MPI_REQUEST_NULL) {
            int flag = 0;
            MPI_Test(req, &flag, MPI_STATUS_IGNORE);
        }
    }
}

// Pivot rows are broadcast in tiles of consecutive rows that belong to a
// single rank. Each local row is then updated with all pivots of the tile
// while it stays in cache, which cuts memory traffic by the tile size and
// amortizes the broadcast latency. Larger tiles increase the serial work of
// the tile owner (quadratic in the tile size), so the size is chosen relative
// to the rows per rank.
constexpr size_t MIN_TILE_ROWS = 8;
constexpr size_t MAX_TILE_ROWS = 32;

static size_t chooseTileRows(const RowDist& rd) {
    const size_t rowsPerRank = rd.n / static_cast<size_t>(rd.size);
    size_t t = rowsPerRank / 4;
    t = std::max(t, MIN_TILE_ROWS);
    t = std::min(t, MAX_TILE_ROWS);
    return std::max<size_t>(1, std::min(t, rd.n));
}

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const RowDist& rd) {
    const size_t numNodes = rd.n;
    if (numNodes == 0) return;

    const size_t lStart = rd.localStart();
    const size_t lCount = rd.localCount();
    const size_t tileRows = chooseTileRows(rd);

    // Tile covers pivots [k0, k1) with k1 = min(k0 + tileRows, end of owner's rows).
    auto tileEnd = [&](const size_t k0) -> size_t {
        const int own = rd.owner(k0);
        return std::min({k0 + tileRows, rd.rowStart(own) + rd.rowCount(own), numNodes});
    };

    // Double-buffered tile storage. Tile row (k - k0) holds row k in the state
    // "all pivots < k applied" (the state the classic algorithm reads it in).
    std::vector<unsigned int> tileBuf[2];
    tileBuf[0].resize(tileRows * numNodes);
    tileBuf[1].resize(tileRows * numNodes);

    unsigned int* distP = dist.data();
    unsigned int* pathP = path.data();

    // Owner-side preparation of a tile [k0, k1) into buf: for k = k0, k0+1,
    // ... relax row k with the tile pivots k0..k-1 (fused, using the
    // snapshots already in buf) and snapshot it. The remaining pivots
    // k+1..k1-1 for these rows are applied later in the regular pass.
    auto prepareTile = [&](const size_t k0, const size_t k1, unsigned int* buf) {
        for (size_t k = k0; k < k1; ++k) {
            const size_t lk = k - lStart;
            unsigned int* row = distP + lk * numNodes;
            relaxRow(row, pathP + lk * numNodes, buf, numNodes, k0, k);
            std::memcpy(buf + (k - k0) * numNodes, row, numNodes * sizeof(unsigned int));
        }
    };

    // Finish the owner's own tile rows: row k still needs pivots k+1..k1-1
    // (pivot k itself never changes row k because dist[k][k] == 0).
    auto finishTileRows = [&](const size_t k0, const size_t k1, const unsigned int* buf,
                              MPI_Request* req) {
        for (size_t k = k0; k < k1; ++k) {
            const size_t lk = k - lStart;
            relaxRow(distP + lk * numNodes, pathP + lk * numNodes,
                     buf + (k + 1 - k0) * numNodes, numNodes, k + 1, k1);
            if (req != nullptr && *req != MPI_REQUEST_NULL) {
                int flag = 0;
                MPI_Test(req, &flag, MPI_STATUS_IGNORE);
            }
        }
    };

    // First tile.
    size_t k0 = 0;
    size_t k1 = tileEnd(0);
    int owner = rd.owner(0);
    int buf = 0;
    if (owner == rd.rank) prepareTile(k0, k1, tileBuf[buf].data());
    MPI_Bcast(tileBuf[buf].data(), static_cast<int>((k1 - k0) * numNodes), MPI_UNSIGNED, owner,
              MPI_COMM_WORLD);

    MPI_Request nextReq = MPI_REQUEST_NULL;

    while (k0 < numNodes) {
        const unsigned int* tile = tileBuf[buf].data();

        const bool hasNext = (k1 < numNodes);
        const size_t n0 = k1;
        const size_t n1 = hasNext ? tileEnd(n0) : n0;
        const int nextOwner = hasNext ? rd.owner(n0) : -1;
        const int nextBuf = buf ^ 1;

        // Local row ranges that get special treatment:
        //   [skipA0, skipA1): rows of the current tile (owner only)
        //   [skipB0, skipB1): rows of the next tile (next owner only)
        size_t skipA0 = 0, skipA1 = 0, skipB0 = 0, skipB1 = 0;
        if (owner == rd.rank) {
            skipA0 = k0 - lStart;
            skipA1 = k1 - lStart;
        }

        if (hasNext) {
            if (nextOwner == rd.rank) {
                // Owner of the next tile: bring its rows up to date with the
                // current tile, prepare the tile, then start the broadcast so
                // it overlaps with the remaining local work.
                skipB0 = n0 - lStart;
                skipB1 = n1 - lStart;
                relaxRows(distP, pathP, tile, numNodes, k0, k1, skipB0, skipB1, nullptr);
                prepareTile(n0, n1, tileBuf[nextBuf].data());
            }
            MPI_Ibcast(tileBuf[nextBuf].data(), static_cast<int>((n1 - n0) * numNodes),
                       MPI_UNSIGNED, nextOwner, MPI_COMM_WORLD, &nextReq);
        }

        // Regular pass over all other local rows (ranges A and B excluded).
        if (skipA0 > skipB0) {
            std::swap(skipA0, skipB0);
            std::swap(skipA1, skipB1);
        }
        relaxRows(distP, pathP, tile, numNodes, k0, k1, 0, skipA0, &nextReq);
        relaxRows(distP, pathP, tile, numNodes, k0, k1, skipA1, skipB0, &nextReq);
        relaxRows(distP, pathP, tile, numNodes, k0, k1, skipB1, lCount, &nextReq);

        // Owner of the current tile: complete its own tile rows.
        if (owner == rd.rank) {
            finishTileRows(k0, k1, tile, &nextReq);
        }

        if (hasNext) {
            MPI_Wait(&nextReq, MPI_STATUS_IGNORE);
        }

        k0 = n0;
        k1 = n1;
        owner = nextOwner;
        buf = nextBuf;
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

// Gather the row-distributed matrix into a full matrix on rank 0.
static void gatherMatrix(const std::vector<unsigned int>& local, std::vector<unsigned int>& full,
                         const RowDist& rd) {
    std::vector<int> counts(rd.size), displs(rd.size);
    for (int r = 0; r < rd.size; ++r) {
        counts[r] = static_cast<int>(rd.rowCount(r) * rd.n);
        displs[r] = static_cast<int>(rd.rowStart(r) * rd.n);
    }
    if (rd.rank == 0) full.resize(rd.n * rd.n);
    MPI_Gatherv(local.data(), static_cast<int>(rd.localCount() * rd.n), MPI_UNSIGNED,
                rd.rank == 0 ? full.data() : nullptr, counts.data(), displs.data(), MPI_UNSIGNED,
                0, MPI_COMM_WORLD);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    RowDist rd;
    MPI_Comm_size(MPI_COMM_WORLD, &rd.size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rd.rank);
    const bool isRoot = (rd.rank == 0);

    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments (identical on all ranks)
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
    rd.n = numNodes;
    
    if (isRoot) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("MPI ranks: %d\n", rd.size);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Allocate local row blocks
    const size_t localElems = rd.localCount() * numNodes;
    std::vector<unsigned int> dist(localElems);
    std::vector<unsigned int> path(localElems);
    
    // Initialize
    if (isRoot) printf("Initializing graph...\n");
    initializeDistanceMatrix(dist, rd, 1, MAX_DISTANCE);
    initializePathMatrix(path, rd);
    
    // Run Floyd-Warshall
    if (isRoot) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double tStart = MPI_Wtime();
    
    floydWarshall(dist, path, rd);
    
    MPI_Barrier(MPI_COMM_WORLD);
    const double tEnd = MPI_Wtime();
    const long durationMs = static_cast<long>((tEnd - tStart) * 1000.0);
    
    if (isRoot) {
        printf("Computation time: %ld ms\n", durationMs);
        
        // Calculate operations per second
        // Floyd-Warshall has O(n³) complexity
        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / (durationMs / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);
    }
    
    int exitCode = 0;
    if (printResults || validate) {
        std::vector<unsigned int> fullDist;
        gatherMatrix(dist, fullDist, rd);

        // Print results for external validation (integer hash-based)
        if (printResults && isRoot) {
            print_results_int(fullDist, "DistanceMatrix");
        }
        
        // Validation
        if (validate) {
            int valid = 1;
            if (isRoot) {
                printf("Validating result...\n");
                valid = validateResult(fullDist, numNodes) ? 1 : 0;
                if (valid) {
                    printf("Validation: PASSED\n");
                } else {
                    printf("Validation: FAILED\n");
                }
            }
            MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
            exitCode = valid ? 0 : 1;
        }
    }
    
    MPI_Finalize();
    return exitCode;
}
