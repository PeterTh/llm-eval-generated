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

// Maximum number of pivot nodes (k) processed per block.
constexpr size_t MAX_BLOCK = 32;
// Column tile width for the main update kernel (keeps the row tile in L1).
constexpr size_t COL_TILE = 512;

// Index calculation for flattened 2D array
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes,
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;

    for (size_t i = 0; i < numNodes * numNodes; ++i) {
        dist[i] = rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
    }

    // Set diagonal to zero (distance from node to itself is 0)
    for (size_t i = 0; i < numNodes; ++i) {
        dist[idx2(i, i, numNodes)] = 0;
    }
}

// ---------------------------------------------------------------------------
// Distributed Floyd-Warshall
//
// Memory layout: dist[idx2(j, i, n)] == dist[i * n + j] is the distance from
// source i to destination j, i.e. memory row i holds all destinations of
// source i.
//
// The matrix is distributed over a Pr x Pc process grid in a block-cyclic
// fashion with block size B: node block c (rows/columns [c*B, (c+1)*B)) is
// owned by process row c % Pr and process column c % Pc. The k loop is
// processed in the same blocks. For block c:
//   phase 1: the diagonal rank evolves the B x B diagonal tile through the
//            B pivots, recording per-pivot snapshots of its rows/columns;
//   phase 2: the ranks in process row c % Pr evolve the pivot row strip and
//            the ranks in process column c % Pc evolve the pivot column strip,
//            recording per-pivot snapshots; the snapshots are broadcast along
//            process columns / rows;
//   phase 3: every rank applies the B pivots, in order, to the rest of its tile.
// Every element receives exactly the same sequence of min-updates as in the
// serial algorithm, so distances and path entries are bit-identical.
// The phases 1/2 of block c+1 are computed and broadcast while phase 3 of
// block c runs.
// ---------------------------------------------------------------------------

struct Range {
    size_t a, b;  // [a, b)
};

struct Grid {
    int P = 1, me = 0;
    int Pr = 1, Pc = 1;  // process grid
    int pr = 0, pc = 0;  // my coordinates
    size_t n = 0;
    size_t B = 1;        // block size
    size_t nb = 0;       // number of blocks
    size_t rowsLocal = 0, colsLocal = 0;
    std::vector<size_t> rowBlkStart;  // local offset of local row block lb (size nRowBlk+1)
    std::vector<size_t> colBlkStart;  // local offset of local col block lb (size nColBlk+1)

    size_t blockStart(size_t c) const { return c * B; }
    size_t blockLen(size_t c) const { return std::min(n, (c + 1) * B) - c * B; }
    int rowOwner(size_t c) const { return static_cast<int>(c % static_cast<size_t>(Pr)); }
    int colOwner(size_t c) const { return static_cast<int>(c % static_cast<size_t>(Pc)); }
    bool inRow(size_t c) const { return rowOwner(c) == pr; }
    bool inCol(size_t c) const { return colOwner(c) == pc; }
    bool isDiag(size_t c) const { return inRow(c) && inCol(c); }
    // local offsets of block c (only valid if inRow(c) / inCol(c))
    size_t localRowOfBlock(size_t c) const { return rowBlkStart[c / static_cast<size_t>(Pr)]; }
    size_t localColOfBlock(size_t c) const { return colBlkStart[c / static_cast<size_t>(Pc)]; }
    size_t rowBlocks() const { return rowBlkStart.size() - 1; }
    size_t colBlocks() const { return colBlkStart.size() - 1; }
    size_t globalRowBlock(size_t lb) const { return static_cast<size_t>(pr) + lb * Pr; }
    size_t globalColBlock(size_t lb) const { return static_cast<size_t>(pc) + lb * Pc; }
};

static Grid makeGrid(const size_t n, const int P, const int me) {
    Grid g;
    g.P = P;
    g.me = me;
    g.n = n;
    // Process grid: Pr = largest divisor of P not exceeding sqrt(P)
    g.Pr = 1;
    for (int d = 1; static_cast<long long>(d) * d <= P; ++d) {
        if (P % d == 0) g.Pr = d;
    }
    g.Pc = P / g.Pr;
    g.pr = me / g.Pc;
    g.pc = me % g.Pc;

    const size_t maxDim = static_cast<size_t>(std::max(g.Pr, g.Pc));
    g.B = std::max<size_t>(1, std::min<size_t>(MAX_BLOCK, (n + maxDim - 1) / maxDim));
    g.nb = (n + g.B - 1) / g.B;

    g.rowBlkStart.push_back(0);
    for (size_t c = static_cast<size_t>(g.pr); c < g.nb; c += g.Pr) {
        g.rowBlkStart.push_back(g.rowBlkStart.back() + g.blockLen(c));
    }
    g.colBlkStart.push_back(0);
    for (size_t c = static_cast<size_t>(g.pc); c < g.nb; c += g.Pc) {
        g.colBlkStart.push_back(g.colBlkStart.back() + g.blockLen(c));
    }
    g.rowsLocal = g.rowBlkStart.back();
    g.colsLocal = g.colBlkStart.back();
    return g;
}

// [0, len) minus the given (possibly overlapping/unsorted) ranges.
static std::vector<Range> subtractRanges(const size_t len, std::vector<Range> excl) {
    std::sort(excl.begin(), excl.end(), [](const Range& x, const Range& y) { return x.a < y.a; });
    std::vector<Range> out;
    size_t pos = 0;
    for (const Range& e : excl) {
        if (e.a > pos) out.push_back({pos, e.a});
        pos = std::max(pos, e.b);
    }
    if (pos < len) out.push_back({pos, len});
    return out;
}

// Local part of the path matrix: the serial initialization sets every entry
// of memory row r (all columns) to r.
static void initializeLocalPathMatrix(std::vector<unsigned int>& path, const Grid& g) {
    for (size_t lb = 0; lb < g.rowBlocks(); ++lb) {
        const size_t c = g.globalRowBlock(lb);
        for (size_t i = 0; i < g.blockLen(c); ++i) {
            std::fill_n(path.data() + (g.rowBlkStart[lb] + i) * g.colsLocal, g.colsLocal,
                        static_cast<unsigned int>(g.blockStart(c) + i));
        }
    }
}

// Copy between the full matrix (rank 0) and the tile of a rank.
template <bool ToTile>
static void copyTile(unsigned int* full, unsigned int* tile, const Grid& g) {
    for (size_t lb = 0; lb < g.rowBlocks(); ++lb) {
        const size_t c = g.globalRowBlock(lb);
        for (size_t i = 0; i < g.blockLen(c); ++i) {
            const size_t gi = g.blockStart(c) + i;
            const size_t li = g.rowBlkStart[lb] + i;
            for (size_t cb = 0; cb < g.colBlocks(); ++cb) {
                const size_t cc = g.globalColBlock(cb);
                unsigned int* f = full + gi * g.n + g.blockStart(cc);
                unsigned int* t = tile + li * g.colsLocal + g.colBlkStart[cb];
                if (ToTile) std::copy_n(f, g.blockLen(cc), t);
                else std::copy_n(t, g.blockLen(cc), f);
            }
        }
    }
}

struct FWState {
    const Grid& g;
    std::vector<unsigned int>& dist;  // local tile, rowsLocal x colsLocal
    std::vector<unsigned int>& path;
    // double-buffered (by block parity) snapshot buffers
    std::vector<unsigned int> rowSnap[2];   // [k][jj]  diagonal tile rows
    std::vector<unsigned int> colSnap[2];   // [k][ii]  diagonal tile columns
    std::vector<unsigned int> rowStrip[2];  // [k][lj]  pivot row k, my columns
    std::vector<unsigned int> colStrip[2];  // [li][k]  pivot column k, my rows
    MPI_Request reqRowSnap[2], reqColSnap[2], reqRowStrip[2], reqColStrip[2];
    MPI_Comm rowComm, colComm;
};

// Phase 1: evolve the diagonal tile of block c through its own pivots.
static void phase1(FWState& s, const size_t c) {
    const Grid& g = s.g;
    const int par = c & 1;
    const size_t Bc = g.blockLen(c);
    const size_t kc0 = g.blockStart(c);
    const size_t cl = g.colsLocal;
    unsigned int* D = s.dist.data() + g.localRowOfBlock(c) * cl + g.localColOfBlock(c);
    unsigned int* Pm = s.path.data() + g.localRowOfBlock(c) * cl + g.localColOfBlock(c);
    unsigned int* rs = s.rowSnap[par].data();
    unsigned int* cs = s.colSnap[par].data();

    for (size_t k = 0; k < Bc; ++k) {
        const unsigned int kg = static_cast<unsigned int>(kc0 + k);
        unsigned int* rsk = rs + k * Bc;
        unsigned int* csk = cs + k * Bc;
        std::copy_n(D + k * cl, Bc, rsk);
        for (size_t ii = 0; ii < Bc; ++ii) csk[ii] = D[ii * cl + k];
        for (size_t ii = 0; ii < Bc; ++ii) {
            if (ii == k) continue;
            const unsigned int a = csk[ii];
            unsigned int* __restrict d = D + ii * cl;
            unsigned int* __restrict p = Pm + ii * cl;
            for (size_t jj = 0; jj < Bc; ++jj) {
                const unsigned int nd = a + rsk[jj];
                const bool lt = nd < d[jj];
                d[jj] = lt ? nd : d[jj];
                p[jj] = lt ? kg : p[jj];
            }
        }
    }
}

// Phase 2a: evolve the pivot row strip (rows of block c, my columns in cols).
static void phase2a(FWState& s, const size_t c, const std::vector<Range>& cols) {
    const Grid& g = s.g;
    const int par = c & 1;
    const size_t Bc = g.blockLen(c);
    const size_t kc0 = g.blockStart(c);
    const size_t cl = g.colsLocal;
    const size_t lr0 = g.localRowOfBlock(c);
    unsigned int* S = s.dist.data() + lr0 * cl;
    unsigned int* Pm = s.path.data() + lr0 * cl;
    unsigned int* strip = s.rowStrip[par].data();
    const unsigned int* cs = s.colSnap[par].data();

    for (size_t k = 0; k < Bc; ++k) {
        const unsigned int kg = static_cast<unsigned int>(kc0 + k);
        unsigned int* snapk = strip + k * cl;
        const unsigned int* Sk = S + k * cl;
        for (const Range& r : cols) std::copy_n(Sk + r.a, r.b - r.a, snapk + r.a);
        for (size_t kk = 0; kk < Bc; ++kk) {
            if (kk == k) continue;
            const unsigned int a = cs[k * Bc + kk];
            unsigned int* Skk = S + kk * cl;
            unsigned int* Pkk = Pm + kk * cl;
            for (const Range& r : cols) {
                unsigned int* __restrict d = Skk + r.a;
                unsigned int* __restrict p = Pkk + r.a;
                const unsigned int* __restrict pk = snapk + r.a;
                const size_t len = r.b - r.a;
                for (size_t j = 0; j < len; ++j) {
                    const unsigned int nd = a + pk[j];
                    const bool lt = nd < d[j];
                    d[j] = lt ? nd : d[j];
                    p[j] = lt ? kg : p[j];
                }
            }
        }
    }
    if (g.isDiag(c)) {
        const size_t lc0 = g.localColOfBlock(c);
        for (size_t k = 0; k < Bc; ++k) {
            std::copy_n(s.rowSnap[par].data() + k * Bc, Bc, strip + k * cl + lc0);
        }
    }
}

// Phase 2b: evolve the pivot column strip (columns of block c, my rows in rows).
static void phase2b(FWState& s, const size_t c, const std::vector<Range>& rows) {
    const Grid& g = s.g;
    const int par = c & 1;
    const size_t Bc = g.blockLen(c);
    const size_t kc0 = g.blockStart(c);
    const size_t cl = g.colsLocal;
    const size_t lc0 = g.localColOfBlock(c);
    unsigned int* strip = s.colStrip[par].data();
    const unsigned int* rs = s.rowSnap[par].data();

    for (size_t k = 0; k < Bc; ++k) {
        const unsigned int kg = static_cast<unsigned int>(kc0 + k);
        const unsigned int* __restrict rsk = rs + k * Bc;  // rsk[k] == 0
        for (const Range& r : rows) {
            for (size_t li = r.a; li < r.b; ++li) {
                unsigned int* __restrict d = s.dist.data() + li * cl + lc0;
                unsigned int* __restrict p = s.path.data() + li * cl + lc0;
                const unsigned int a = d[k];
                strip[li * Bc + k] = a;
                for (size_t kk = 0; kk < Bc; ++kk) {
                    const unsigned int nd = a + rsk[kk];
                    const bool lt = nd < d[kk];
                    d[kk] = lt ? nd : d[kk];
                    p[kk] = lt ? kg : p[kk];
                }
            }
        }
    }
    if (g.isDiag(c)) {
        const size_t lr0 = g.localRowOfBlock(c);
        const unsigned int* cs = s.colSnap[par].data();
        for (size_t ii = 0; ii < Bc; ++ii) {
            for (size_t k = 0; k < Bc; ++k) strip[(lr0 + ii) * Bc + k] = cs[k * Bc + ii];
        }
    }
}

static inline void progressRequests(FWState& s, const int par) {
    MPI_Request reqs[4] = {s.reqRowSnap[par], s.reqColSnap[par], s.reqRowStrip[par],
                           s.reqColStrip[par]};
    int flag;
    MPI_Testall(4, reqs, &flag, MPI_STATUSES_IGNORE);
    s.reqRowSnap[par] = reqs[0];
    s.reqColSnap[par] = reqs[1];
    s.reqRowStrip[par] = reqs[2];
    s.reqColStrip[par] = reqs[3];
}

// Innermost kernel: apply Bc pivots (in order) to one row tile dd/pp of
// length tl. a[k] = dist[i][k] before pivot k, pk[k*cl + j] = dist[k][j]
// before pivot k.
__attribute__((noinline)) static void applyPivotsTile(unsigned int* __restrict dd,
                                                      unsigned int* __restrict pp,
                                                      const unsigned int* __restrict a,
                                                      const unsigned int* __restrict strip,
                                                      const size_t cl, const size_t Bc,
                                                      const unsigned int kc0, const size_t tl) {
    for (size_t k = 0; k < Bc; ++k) {
        const unsigned int ak = a[k];
        const unsigned int kg = kc0 + static_cast<unsigned int>(k);
        const unsigned int* __restrict pk = strip + k * cl;
        for (size_t j = 0; j < tl; ++j) {
            const unsigned int nd = ak + pk[j];
            const bool lt = nd < dd[j];
            dd[j] = lt ? nd : dd[j];
            pp[j] = lt ? kg : pp[j];
        }
    }
}

// Phase 3: apply the pivots of block c to rows x cols of the local tile.
// If pollPar >= 0, outstanding requests of that parity are progressed periodically.
static void phase3(FWState& s, const size_t c, const std::vector<Range>& rows,
                   const std::vector<Range>& cols, const int pollPar) {
    const Grid& g = s.g;
    const int par = c & 1;
    const size_t Bc = g.blockLen(c);
    const size_t kc0 = g.blockStart(c);
    const size_t cl = g.colsLocal;
    const unsigned int* rowStrip = s.rowStrip[par].data();
    const unsigned int* colStrip = s.colStrip[par].data();
    size_t sincePoll = 0;

    for (const Range& rr : rows) {
        for (size_t li = rr.a; li < rr.b; ++li) {
            const unsigned int* a = colStrip + li * Bc;
            unsigned int* d = s.dist.data() + li * cl;
            unsigned int* p = s.path.data() + li * cl;
            for (const Range& cr : cols) {
                for (size_t jt = cr.a; jt < cr.b; jt += COL_TILE) {
                    const size_t tl = std::min(COL_TILE, cr.b - jt);
                    applyPivotsTile(d + jt, p + jt, a, rowStrip + jt, cl, Bc,
                                    static_cast<unsigned int>(kc0), tl);
                }
            }
            if (pollPar >= 0 && ++sincePoll >= 4) {
                sincePoll = 0;
                progressRequests(s, pollPar);
            }
        }
    }
}

// Prepare block c (phases 1 and 2, posting the broadcasts). If pre is true,
// the regions touched here first receive the pending update of block c-1.
static void prepareBlock(FWState& s, const size_t c, const bool pre) {
    const Grid& g = s.g;
    const int par = c & 1;
    const size_t Bc = g.blockLen(c);
    const int cnt = static_cast<int>(Bc * Bc);
    const bool inRow = g.inRow(c), inCol = g.inCol(c), diag = inRow && inCol;

    // regions already finished for block c-1 (by its phases 1/2)
    std::vector<Range> xr, xc;
    if (pre) {
        if (g.inRow(c - 1)) {
            const size_t o = g.localRowOfBlock(c - 1);
            xr.push_back({o, o + g.blockLen(c - 1)});
        }
        if (g.inCol(c - 1)) {
            const size_t o = g.localColOfBlock(c - 1);
            xc.push_back({o, o + g.blockLen(c - 1)});
        }
    }
    Range rowsC{0, 0}, colsC{0, 0};
    if (inRow) { rowsC.a = g.localRowOfBlock(c); rowsC.b = rowsC.a + Bc; }
    if (inCol) { colsC.a = g.localColOfBlock(c); colsC.b = colsC.a + Bc; }

    if (diag) {
        if (pre) phase3(s, c - 1, std::vector<Range>(1, rowsC), std::vector<Range>(1, colsC), -1);
        phase1(s, c);
        MPI_Ibcast(s.rowSnap[par].data(), cnt, MPI_UNSIGNED, g.pr, s.colComm, &s.reqRowSnap[par]);
        MPI_Ibcast(s.colSnap[par].data(), cnt, MPI_UNSIGNED, g.pc, s.rowComm, &s.reqColSnap[par]);
    }
    if (inRow) {
        // columns handled by phase 2a: all except the diagonal tile (phase 1);
        // the pending update of block c-1 additionally skips its column strip.
        std::vector<Range> ex;
        if (diag) ex.push_back(colsC);
        const std::vector<Range> cols = subtractRanges(g.colsLocal, ex);
        for (const Range& r : xc) ex.push_back(r);
        if (pre) phase3(s, c - 1, std::vector<Range>(1, rowsC), subtractRanges(g.colsLocal, ex), -1);
        if (!diag) {
            MPI_Ibcast(s.colSnap[par].data(), cnt, MPI_UNSIGNED, g.colOwner(c), s.rowComm,
                       &s.reqColSnap[par]);
        }
        MPI_Wait(&s.reqColSnap[par], MPI_STATUS_IGNORE);
        phase2a(s, c, cols);
        MPI_Ibcast(s.rowStrip[par].data(), static_cast<int>(Bc * g.colsLocal), MPI_UNSIGNED, g.pr,
                   s.colComm, &s.reqRowStrip[par]);
    }
    if (inCol) {
        std::vector<Range> ex;
        if (diag) ex.push_back(rowsC);
        const std::vector<Range> rows = subtractRanges(g.rowsLocal, ex);
        for (const Range& r : xr) ex.push_back(r);
        if (pre) phase3(s, c - 1, subtractRanges(g.rowsLocal, ex), std::vector<Range>(1, colsC), -1);
        if (!diag) {
            MPI_Ibcast(s.rowSnap[par].data(), cnt, MPI_UNSIGNED, g.rowOwner(c), s.colComm,
                       &s.reqRowSnap[par]);
        }
        MPI_Wait(&s.reqRowSnap[par], MPI_STATUS_IGNORE);
        phase2b(s, c, rows);
        MPI_Ibcast(s.colStrip[par].data(), static_cast<int>(g.rowsLocal * Bc), MPI_UNSIGNED, g.pc,
                   s.rowComm, &s.reqColStrip[par]);
    }
    if (!inRow) {
        MPI_Ibcast(s.rowStrip[par].data(), static_cast<int>(Bc * g.colsLocal), MPI_UNSIGNED,
                   g.rowOwner(c), s.colComm, &s.reqRowStrip[par]);
    }
    if (!inCol) {
        MPI_Ibcast(s.colStrip[par].data(), static_cast<int>(g.rowsLocal * Bc), MPI_UNSIGNED,
                   g.colOwner(c), s.rowComm, &s.reqColStrip[par]);
    }
}

static void waitBlock(FWState& s, const size_t c) {
    const int par = c & 1;
    MPI_Wait(&s.reqRowSnap[par], MPI_STATUS_IGNORE);
    MPI_Wait(&s.reqColSnap[par], MPI_STATUS_IGNORE);
    MPI_Wait(&s.reqRowStrip[par], MPI_STATUS_IGNORE);
    MPI_Wait(&s.reqColStrip[par], MPI_STATUS_IGNORE);
}

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const Grid& g) {
    const size_t nb = g.nb;
    if (nb == 0) return;
    const size_t B = g.B;

    FWState s{g, dist, path, {}, {}, {}, {}, {}, {}, {}, {}, MPI_COMM_NULL, MPI_COMM_NULL};
    for (int par = 0; par < 2; ++par) {
        s.rowSnap[par].resize(B * B);
        s.colSnap[par].resize(B * B);
        s.rowStrip[par].resize(B * g.colsLocal);
        s.colStrip[par].resize(g.rowsLocal * B);
        s.reqRowSnap[par] = s.reqColSnap[par] = s.reqRowStrip[par] = s.reqColStrip[par] =
            MPI_REQUEST_NULL;
    }
    MPI_Comm_split(MPI_COMM_WORLD, g.pr, g.pc, &s.rowComm);
    MPI_Comm_split(MPI_COMM_WORLD, g.pc, g.pr, &s.colComm);

    prepareBlock(s, 0, false);
    waitBlock(s, 0);

    for (size_t c = 0; c < nb; ++c) {
        const bool hasNext = (c + 1 < nb);
        if (hasNext) prepareBlock(s, c + 1, true);

        // Remaining part of the local tile for block c
        std::vector<Range> xr, xc;
        if (g.inRow(c)) {
            const size_t o = g.localRowOfBlock(c);
            xr.push_back({o, o + g.blockLen(c)});
        }
        if (g.inCol(c)) {
            const size_t o = g.localColOfBlock(c);
            xc.push_back({o, o + g.blockLen(c)});
        }
        if (hasNext && g.inRow(c + 1)) {
            const size_t o = g.localRowOfBlock(c + 1);
            xr.push_back({o, o + g.blockLen(c + 1)});
        }
        if (hasNext && g.inCol(c + 1)) {
            const size_t o = g.localColOfBlock(c + 1);
            xc.push_back({o, o + g.blockLen(c + 1)});
        }
        phase3(s, c, subtractRanges(g.rowsLocal, xr), subtractRanges(g.colsLocal, xc),
               hasNext ? static_cast<int>((c + 1) & 1) : -1);

        if (hasNext) waitBlock(s, c + 1);
    }

    MPI_Comm_free(&s.rowComm);
    MPI_Comm_free(&s.colComm);
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

// Distribute the full matrix from rank 0 into the tiles (toTiles) or collect
// the tiles into the full matrix on rank 0.
static void distributeMatrix(std::vector<unsigned int>& full, std::vector<unsigned int>& tile,
                             const Grid& g, const bool toTiles) {
    if (g.me == 0) {
        std::vector<unsigned int> buf;
        for (int r = 0; r < g.P; ++r) {
            const Grid gr = (r == 0) ? g : makeGrid(g.n, g.P, r);
            const size_t cnt = gr.rowsLocal * gr.colsLocal;
            if (cnt == 0) continue;
            if (toTiles) {
                if (r == 0) {
                    copyTile<true>(full.data(), tile.data(), gr);
                } else {
                    buf.resize(cnt);
                    copyTile<true>(full.data(), buf.data(), gr);
                    MPI_Datatype rowT;
                    MPI_Type_contiguous(static_cast<int>(gr.colsLocal), MPI_UNSIGNED, &rowT);
                    MPI_Type_commit(&rowT);
                    MPI_Send(buf.data(), static_cast<int>(gr.rowsLocal), rowT, r, 0, MPI_COMM_WORLD);
                    MPI_Type_free(&rowT);
                }
            } else {
                if (r == 0) {
                    copyTile<false>(full.data(), tile.data(), gr);
                } else {
                    buf.resize(cnt);
                    MPI_Datatype rowT;
                    MPI_Type_contiguous(static_cast<int>(gr.colsLocal), MPI_UNSIGNED, &rowT);
                    MPI_Type_commit(&rowT);
                    MPI_Recv(buf.data(), static_cast<int>(gr.rowsLocal), rowT, r, 1, MPI_COMM_WORLD,
                             MPI_STATUS_IGNORE);
                    MPI_Type_free(&rowT);
                    copyTile<false>(full.data(), buf.data(), gr);
                }
            }
        }
    } else if (g.rowsLocal * g.colsLocal > 0) {
        MPI_Datatype rowT;
        MPI_Type_contiguous(static_cast<int>(g.colsLocal), MPI_UNSIGNED, &rowT);
        MPI_Type_commit(&rowT);
        if (toTiles) {
            MPI_Recv(tile.data(), static_cast<int>(g.rowsLocal), rowT, 0, 0, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
        } else {
            MPI_Send(tile.data(), static_cast<int>(g.rowsLocal), rowT, 0, 1, MPI_COMM_WORLD);
        }
        MPI_Type_free(&rowT);
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int P = 1, me = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &P);
    MPI_Comm_rank(MPI_COMM_WORLD, &me);
    const bool root = (me == 0);

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

    const Grid g = makeGrid(numNodes, P, me);

    if (root) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d (%d x %d grid, block size %zu)\n", P, g.Pr, g.Pc, g.B);
    }

    // Allocate matrices: full distance matrix on rank 0 (for initialization and
    // result collection), distributed tiles everywhere.
    std::vector<unsigned int> dist;
    if (root) dist.resize(numNodes * numNodes);
    std::vector<unsigned int> distLocal(g.rowsLocal * g.colsLocal);
    std::vector<unsigned int> pathLocal(g.rowsLocal * g.colsLocal);

    // Initialize
    if (root) {
        printf("Initializing graph...\n");
        initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
    }
    initializeLocalPathMatrix(pathLocal, g);
    distributeMatrix(dist, distLocal, g, true);

    // Run Floyd-Warshall
    if (root) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    floydWarshall(distLocal, pathLocal, g);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Collect the full distance matrix on rank 0
    distributeMatrix(dist, distLocal, g, false);

    int exitCode = 0;
    if (root) {
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

        // Validation
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(dist, numNodes);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
