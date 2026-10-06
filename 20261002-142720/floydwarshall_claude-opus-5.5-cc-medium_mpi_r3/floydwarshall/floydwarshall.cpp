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

// Pivot block size (number of k steps processed per communication round)
constexpr size_t PIVOT_BLOCK = 64;
// Column tile width for the bulk update (keeps pivot rows cache resident)
constexpr size_t COL_TILE = 1024;

// Index calculation for flattened 2D array
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

// 1D block partition of [0, n) over p parts
inline size_t blockStart(const size_t part, const size_t n, const size_t p) noexcept {
    return part * n / p;
}

// Distributed state: 2D process grid, each rank owns a contiguous block of
// rows [r0, r1) and columns [c0, c1) of the row-major distance/path matrices.
struct Grid {
    int rank = 0, size = 1;
    int pr = 1, pc = 1;        // process grid dimensions
    int myRow = 0, myCol = 0;  // coordinates in grid
    MPI_Comm rowComm{}, colComm{};  // rowComm: same grid row (rank == myCol)
                                    // colComm: same grid column (rank == myRow)
    std::vector<size_t> rowStarts, colStarts;  // size pr+1 / pc+1
    size_t r0 = 0, r1 = 0, c0 = 0, c1 = 0;
    size_t nr = 0, nc = 0;
};

static void setupGrid(Grid& g, const size_t n) {
    MPI_Comm_rank(MPI_COMM_WORLD, &g.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g.size);
    int dims[2] = {0, 0};
    MPI_Dims_create(g.size, 2, dims);
    g.pr = dims[0];
    g.pc = dims[1];
    g.myRow = g.rank / g.pc;
    g.myCol = g.rank % g.pc;
    MPI_Comm_split(MPI_COMM_WORLD, g.myRow, g.myCol, &g.rowComm);
    MPI_Comm_split(MPI_COMM_WORLD, g.myCol, g.myRow, &g.colComm);
    g.rowStarts.resize(g.pr + 1);
    g.colStarts.resize(g.pc + 1);
    for (int p = 0; p <= g.pr; ++p) g.rowStarts[p] = blockStart(p, n, g.pr);
    for (int p = 0; p <= g.pc; ++p) g.colStarts[p] = blockStart(p, n, g.pc);
    g.r0 = g.rowStarts[g.myRow];
    g.r1 = g.rowStarts[g.myRow + 1];
    g.c0 = g.colStarts[g.myCol];
    g.c1 = g.colStarts[g.myCol + 1];
    g.nr = g.r1 - g.r0;
    g.nc = g.c1 - g.c0;
}

static int ownerOf(const std::vector<size_t>& starts, const size_t k) {
    // starts is sorted; find p with starts[p] <= k < starts[p+1]
    const auto it = std::upper_bound(starts.begin(), starts.end(), k);
    return static_cast<int>(it - starts.begin()) - 1;
}

// Generates the same random sequence as the sequential code, keeping only the
// local block. Matrix element (row r, col c) lives at linear index r*n + c.
void initializeDistanceMatrix(std::vector<unsigned int>& dist, const Grid& g, const size_t numNodes,
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;

    const size_t skipBefore = g.r0 * numNodes;
    for (size_t i = 0; i < skipBefore; ++i) rand_r(&seed);
    for (size_t r = g.r0; r < g.r1; ++r) {
        unsigned int* row = dist.data() + (r - g.r0) * g.nc;
        for (size_t c = 0; c < numNodes; ++c) {
            const unsigned int v = rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
            if (c >= g.c0 && c < g.c1) row[c - g.c0] = v;
        }
    }

    // Set diagonal to zero (distance from node to itself is 0)
    for (size_t r = g.r0; r < g.r1; ++r) {
        if (r >= g.c0 && r < g.c1) dist[(r - g.r0) * g.nc + (r - g.c0)] = 0;
    }
}

void initializePathMatrix(std::vector<unsigned int>& path, const Grid& g) {
    // Sequential initialization results in path[row r][col c] = r
    for (size_t r = g.r0; r < g.r1; ++r) {
        unsigned int* row = path.data() + (r - g.r0) * g.nc;
        for (size_t c = 0; c < g.nc; ++c) row[c] = static_cast<unsigned int>(r);
    }
}

// Relax one row segment with pivot k: D[j] = min(D[j], dIK + R[j]), path = k
static inline void relaxRow(unsigned int* __restrict D, unsigned int* __restrict P,
                            const unsigned int* __restrict R, const unsigned int dIK,
                            const unsigned int k, const size_t len) {
    for (size_t j = 0; j < len; ++j) {
        const unsigned int nd = dIK + R[j];
        const bool better = nd < D[j];
        D[j] = better ? nd : D[j];
        P[j] = better ? k : P[j];
    }
}

// Blocked-pivot distributed Floyd-Warshall. For each pivot block K = [k0,k1)
// the panels (rows K and columns K) are advanced step by step, recording the
// exact per-step pivot row/column values; the remaining elements are then
// updated with those values in increasing k order. This reproduces the
// update sequence of the classic algorithm element by element (dist and path).
void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const Grid& g, const size_t numNodes) {
    const size_t nr = g.nr, nc = g.nc;
    std::vector<unsigned int> diagBlk(PIVOT_BLOCK * PIVOT_BLOCK);
    std::vector<unsigned int> diagCol(PIVOT_BLOCK * PIVOT_BLOCK);  // [k'][r]: d_{k'-1}[r][k']
    std::vector<unsigned int> diagRow(PIVOT_BLOCK * PIVOT_BLOCK);  // [k'][c]: d_{k'-1}[k'][c]
    std::vector<unsigned int> rowSnap(PIVOT_BLOCK * std::max<size_t>(nc, 1));  // [k'][j]
    std::vector<unsigned int> colSnap(std::max<size_t>(nr, 1) * PIVOT_BLOCK);  // [i][k']

    size_t k0 = 0;
    while (k0 < numNodes) {
        const int ownR = ownerOf(g.rowStarts, k0);
        const int ownC = ownerOf(g.colStarts, k0);
        const size_t k1 = std::min({k0 + PIVOT_BLOCK, g.rowStarts[ownR + 1], g.colStarts[ownC + 1]});
        const size_t b = k1 - k0;
        const bool inRowPanel = (g.myRow == ownR);
        const bool inColPanel = (g.myCol == ownC);

        // Phase 1: diagonal block owner evolves a copy of the K x K block
        if (inRowPanel && inColPanel) {
            const size_t lr = k0 - g.r0, lc = k0 - g.c0;
            for (size_t r = 0; r < b; ++r)
                std::memcpy(&diagBlk[r * b], &dist[(lr + r) * nc + lc], b * sizeof(unsigned int));
            for (size_t kk = 0; kk < b; ++kk) {
                for (size_t r = 0; r < b; ++r) diagCol[kk * b + r] = diagBlk[r * b + kk];
                std::memcpy(&diagRow[kk * b], &diagBlk[kk * b], b * sizeof(unsigned int));
                const unsigned int* R = &diagBlk[kk * b];
                for (size_t r = 0; r < b; ++r) {
                    const unsigned int dIK = diagBlk[r * b + kk];
                    unsigned int* D = &diagBlk[r * b];
                    for (size_t c = 0; c < b; ++c) {
                        const unsigned int nd = dIK + R[c];
                        D[c] = nd < D[c] ? nd : D[c];
                    }
                }
            }
        }
        MPI_Request reqs[2];
        int nreq = 0;
        if (inRowPanel) MPI_Ibcast(diagCol.data(), (int)(b * b), MPI_UNSIGNED, ownC, g.rowComm, &reqs[nreq++]);
        if (inColPanel) MPI_Ibcast(diagRow.data(), (int)(b * b), MPI_UNSIGNED, ownR, g.colComm, &reqs[nreq++]);
        MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);

        // Phase 2a: row panel (rows K, local columns)
        if (inRowPanel) {
            const size_t lr = k0 - g.r0;
            for (size_t kk = 0; kk < b; ++kk) {
                const unsigned int* Rk = &dist[(lr + kk) * nc];
                std::memcpy(&rowSnap[kk * nc], Rk, nc * sizeof(unsigned int));
                const unsigned int* R = &rowSnap[kk * nc];
                for (size_t r = 0; r < b; ++r) {
                    if (r == kk) continue;  // pivot row is unchanged in its own step
                    relaxRow(&dist[(lr + r) * nc], &path[(lr + r) * nc], R,
                             diagCol[kk * b + r], (unsigned int)(k0 + kk), nc);
                }
            }
        }
        // Phase 2b: column panel (local rows outside K, columns K)
        if (inColPanel) {
            const size_t lc = k0 - g.c0;
            for (size_t i = 0; i < nr; ++i) {
                const size_t gi = g.r0 + i;
                if (gi >= k0 && gi < k1) continue;  // handled by row panel
                unsigned int* D = &dist[i * nc + lc];
                unsigned int* P = &path[i * nc + lc];
                unsigned int* S = &colSnap[i * b];
                for (size_t kk = 0; kk < b; ++kk) {
                    S[kk] = D[kk];
                    relaxRow(D, P, &diagRow[kk * b], S[kk], (unsigned int)(k0 + kk), b);
                }
            }
        }
        nreq = 0;
        MPI_Ibcast(rowSnap.data(), (int)(b * nc), MPI_UNSIGNED, ownR, g.colComm, &reqs[nreq++]);
        MPI_Ibcast(colSnap.data(), (int)(nr * b), MPI_UNSIGNED, ownC, g.rowComm, &reqs[nreq++]);
        MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);

        // Phase 3: remaining local elements (rows outside K, columns outside K)
        size_t skipLo = nc, skipHi = nc;
        if (inColPanel) {
            skipLo = k0 - g.c0;
            skipHi = k1 - g.c0;
        }
        for (size_t jt = 0; jt < nc; jt += COL_TILE) {
            const size_t jtEnd = std::min(jt + COL_TILE, nc);
            // Split the tile around the skipped (column panel) range
            const size_t aLo = jt, aHi = std::min(jtEnd, std::max(jt, skipLo));
            const size_t bLo = std::max(jt, std::min(jtEnd, skipHi)), bHi = jtEnd;
            for (size_t i = 0; i < nr; ++i) {
                const size_t gi = g.r0 + i;
                if (gi >= k0 && gi < k1) continue;
                unsigned int* D = &dist[i * nc];
                unsigned int* P = &path[i * nc];
                const unsigned int* S = &colSnap[i * b];
                for (size_t kk = 0; kk < b; ++kk) {
                    const unsigned int* R = &rowSnap[kk * nc];
                    const unsigned int k = (unsigned int)(k0 + kk);
                    if (aHi > aLo) relaxRow(D + aLo, P + aLo, R + aLo, S[kk], k, aHi - aLo);
                    if (bHi > bLo) relaxRow(D + bLo, P + bLo, R + bLo, S[kk], k, bHi - bLo);
                }
            }
        }
        k0 = k1;
    }
}

// Assemble the full distance matrix on rank 0
static void gatherMatrix(const std::vector<unsigned int>& local, std::vector<unsigned int>& full,
                         const Grid& g, const size_t n) {
    if (g.rank == 0) {
        full.assign(n * n, 0);
        std::vector<unsigned int> buf;
        for (int p = 0; p < g.size; ++p) {
            const int prow = p / g.pc, pcol = p % g.pc;
            const size_t r0 = g.rowStarts[prow], r1 = g.rowStarts[prow + 1];
            const size_t c0 = g.colStarts[pcol], c1 = g.colStarts[pcol + 1];
            const size_t cnt = (r1 - r0) * (c1 - c0);
            if (cnt == 0) continue;
            const unsigned int* src;
            if (p == 0) {
                src = local.data();
            } else {
                buf.resize(cnt);
                MPI_Recv(buf.data(), (int)cnt, MPI_UNSIGNED, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                src = buf.data();
            }
            for (size_t r = r0; r < r1; ++r)
                std::memcpy(&full[r * n + c0], src + (r - r0) * (c1 - c0), (c1 - c0) * sizeof(unsigned int));
        }
    } else if (g.nr * g.nc > 0) {
        MPI_Send(local.data(), (int)(g.nr * g.nc), MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD);
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

static int finish(const int code) {
    MPI_Finalize();
    return code;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    const bool root = (rank == 0);

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
            if (root) printUsage(argv[0]);
            return finish(0);
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return finish(1);
        }
    }
    
    Grid g;
    setupGrid(g, numNodes);

    if (root) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Allocate local blocks of the matrices
    std::vector<unsigned int> dist(g.nr * g.nc);
    std::vector<unsigned int> path(g.nr * g.nc);
    
    // Initialize
    if (root) printf("Initializing graph...\n");
    initializeDistanceMatrix(dist, g, numNodes, 1, MAX_DISTANCE);
    initializePathMatrix(path, g);
    
    // Run Floyd-Warshall
    if (root) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    floydWarshall(dist, path, g, numNodes);
    MPI_Barrier(MPI_COMM_WORLD);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (root) {
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate operations per second
        // Floyd-Warshall has O(n³) complexity
        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);
    }

    std::vector<unsigned int> fullDist;
    if (printResults || validate) gatherMatrix(dist, fullDist, g, numNodes);
    
    // Print results for external validation (integer hash-based)
    if (printResults && root) {
        print_results_int(fullDist, "DistanceMatrix");
    }
    
    // Validation
    if (validate) {
        int valid = 0;
        if (root) {
            printf("Validating result...\n");
            valid = validateResult(fullDist, numNodes) ? 1 : 0;
            printf(valid ? "Validation: PASSED\n" : "Validation: FAILED\n");
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
        return finish(valid ? 0 : 1);
    }
    
    return finish(0);
}
