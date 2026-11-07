#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// MPI-parallel Cholesky decomposition (distributed memory, 2D process grid).
//
// Decomposes a positive definite matrix A into L * L^T where L is lower
// triangular.
//
// Data distribution: the P processes form a Pr x Pc grid.  Rows are
// distributed over the Pr process rows block-cyclically in small blocks of DB
// rows; columns are distributed over the Pc process columns in blocks of NB
// columns (the panel width).  In both dimensions the blocks of consecutive
// "rounds" are assigned in alternating (zig-zag) order, which balances the
// triangular work.  Every process stores its (local rows x local columns)
// sub-matrix row-major.
//
// Algorithm: blocked right-looking Cholesky with panel width NB:
//   for every panel k (columns kb .. kb+kw, owned by process column kc):
//     1. the processes of column kc gather the kw x kw diagonal block and
//        factorize it redundantly (POTRF);
//     2. they solve L_ik = A_ik * L_kk^{-T} for their rows below the diagonal
//        block (TRSM), producing the panel in a packed tile layout;
//     3. the panel is sent along the process rows (as the "A" operand of the
//        update) and, row-block-wise, to the process columns that own the
//        corresponding column blocks (as the "B" operand);
//     4. every process applies A_ij -= L_ik * L_jk^T to the lower-triangular
//        part of its sub-matrix with a register-tiled SIMD kernel.
//
// Matrix generation needs no communication (every process regenerates the
// random rows it needs from the same deterministic stream); validation is
// distributed as well.  Only printing and the optional gathering of L (-r)
// involve rank 0 alone.

namespace {

size_t NB = 128;            // panel width (= column distribution block); set in Grid::init
constexpr size_t MR = 4;    // micro-kernel rows   (one SIMD register of doubles)
constexpr size_t NR = 8;    // micro-kernel columns (two SIMD registers)

int g_rank = 0;
int g_size = 1;

// ---------------------------------------------------------------------------
// Portable SIMD (compiler vector extension; no library dependency)
// ---------------------------------------------------------------------------

typedef double v4d __attribute__((vector_size(32)));

inline v4d loadU(const double* p) {
    v4d v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}
inline void storeU(double* p, v4d v) { std::memcpy(p, &v, sizeof(v)); }
inline v4d bcast(double x) { return v4d{x, x, x, x}; }

// ---------------------------------------------------------------------------
// Micro kernel:  C[i][j] (-)= sum_t Ap[t][i] * Bp[t][j]   (i < MR, j < NR)
//   Ap : packed A tile, K x MR contiguous
//   Bp : packed B tile, K x NR contiguous
//   C  : row stride ldc; only the first `rows` x `cols` entries are written.
// All accumulation happens in 8 vector registers.
// ---------------------------------------------------------------------------

// The MR x NR accumulation loop.  Instantiated twice: with fused multiply-add
// (fast path for the factorization) and without contraction (used for matrix
// generation, so that the generated matrix is bit-identical to the original
// sequential code, which rounds every product before adding it).
#define CHOLESKY_KERNEL_BODY                                                    \
    v4d c00 = {}, c01 = {}, c10 = {}, c11 = {}, c20 = {}, c21 = {}, c30 = {}, c31 = {}; \
    for (size_t t = 0; t < K; ++t) {                                            \
        v4d b0 = loadU(Bp + t * NR);                                            \
        v4d b1 = loadU(Bp + t * NR + 4);                                        \
        /* keep b0/b1 in registers (otherwise GCC re-loads them per FMA) */     \
        __asm__("" : "+x"(b0), "+x"(b1));                                       \
        const double* a = Ap + t * MR;                                          \
        v4d av;                                                                 \
        av = bcast(a[0]); c00 += av * b0; c01 += av * b1;                       \
        av = bcast(a[1]); c10 += av * b0; c11 += av * b1;                       \
        av = bcast(a[2]); c20 += av * b0; c21 += av * b1;                       \
        av = bcast(a[3]); c30 += av * b0; c31 += av * b1;                       \
    }

struct Acc {
    v4d c[MR][2];
};

inline Acc accumulateFMA(size_t K, const double* __restrict Ap, const double* __restrict Bp) {
    CHOLESKY_KERNEL_BODY
    return Acc{{{c00, c01}, {c10, c11}, {c20, c21}, {c30, c31}}};
}

__attribute__((optimize("-ffp-contract=off")))
inline Acc accumulateNoFMA(size_t K, const double* __restrict Ap, const double* __restrict Bp) {
    CHOLESKY_KERNEL_BODY
    return Acc{{{c00, c01}, {c10, c11}, {c20, c21}, {c30, c31}}};
}
#undef CHOLESKY_KERNEL_BODY

template <bool SUB, bool FMA = true>
inline void microKernel(size_t K, const double* __restrict Ap, const double* __restrict Bp,
                        double* __restrict C, size_t ldc, size_t rows, size_t cols) {
    const Acc acc = FMA ? accumulateFMA(K, Ap, Bp) : accumulateNoFMA(K, Ap, Bp);
    const v4d c00 = acc.c[0][0], c01 = acc.c[0][1], c10 = acc.c[1][0], c11 = acc.c[1][1];
    const v4d c20 = acc.c[2][0], c21 = acc.c[2][1], c30 = acc.c[3][0], c31 = acc.c[3][1];
    if (rows == MR && cols == NR) {
        auto upd = [](double* c, v4d v0, v4d v1) {
            if constexpr (SUB) {
                storeU(c, loadU(c) - v0);
                storeU(c + 4, loadU(c + 4) - v1);
            } else {
                storeU(c, loadU(c) + v0);
                storeU(c + 4, loadU(c + 4) + v1);
            }
        };
        upd(C, c00, c01);
        upd(C + ldc, c10, c11);
        upd(C + 2 * ldc, c20, c21);
        upd(C + 3 * ldc, c30, c31);
    } else {
        double tmp[MR][NR];
        storeU(tmp[0], c00); storeU(tmp[0] + 4, c01);
        storeU(tmp[1], c10); storeU(tmp[1] + 4, c11);
        storeU(tmp[2], c20); storeU(tmp[2] + 4, c21);
        storeU(tmp[3], c30); storeU(tmp[3] + 4, c31);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                if constexpr (SUB) C[i * ldc + j] -= tmp[i][j];
                else C[i * ldc + j] += tmp[i][j];
            }
        }
    }
}

// Pack `rows` rows (row stride lda, K columns) into the tiled layout
// [tile][t][MR]; rows beyond `rows` in the last tile are zero-filled.
void packA(const double* A, size_t lda, size_t rows, size_t K, double* Ap) {
    const size_t tiles = (rows + MR - 1) / MR;
    for (size_t mt = 0; mt < tiles; ++mt) {
        double* dst = Ap + mt * K * MR;
        const size_t r0 = mt * MR;
        const size_t rv = std::min(MR, rows - r0);
        for (size_t t = 0; t < K; ++t) {
            for (size_t i = 0; i < rv; ++i) dst[t * MR + i] = A[(r0 + i) * lda + t];
            for (size_t i = rv; i < MR; ++i) dst[t * MR + i] = 0.0;
        }
    }
}

// ---------------------------------------------------------------------------
// Block-cyclic (zig-zag) distribution of one dimension
// ---------------------------------------------------------------------------

struct Dist {
    size_t n = 0;        // extent
    size_t P = 1;        // number of processes along this dimension
    size_t me = 0;       // my coordinate
    size_t B = 4;        // block size
    size_t nblocks = 0;  // number of blocks

    void init(size_t n_, size_t P_, size_t me_, size_t B_) {
        n = n_;
        P = P_;
        me = me_;
        B = B_;
        nblocks = (n + B - 1) / B;
    }
    // Owner of block b: zig-zag block cyclic.
    size_t owner(size_t b) const {
        const size_t round = b / P;
        const size_t pos = b % P;
        return (round % 2 == 0) ? pos : (P - 1 - pos);
    }
    // Block owned by process p in a given round (strictly increasing in round).
    size_t blockOfRound(size_t p, size_t round) const {
        return round * P + ((round % 2 == 0) ? p : (P - 1 - p));
    }
    size_t blockSize(size_t b) const { return std::min(B, n - b * B); }
    size_t localOffset(size_t b) const { return (b / P) * B; }  // local index of block start

    size_t numLocalBlocks(size_t p) const {
        size_t cnt = 0;
        for (size_t r = 0; blockOfRound(p, r) < nblocks; ++r) ++cnt;
        return cnt;
    }
    size_t localSize(size_t p) const {
        size_t s = 0;
        for (size_t r = 0; blockOfRound(p, r) < nblocks; ++r) s += blockSize(blockOfRound(p, r));
        return s;
    }
    size_t globalIndex(size_t p, size_t l) const {
        return blockOfRound(p, l / B) * B + (l % B);
    }
    // First local index of process p whose global index is >= g
    // (g must be a multiple of B, or n).
    size_t firstLocalAtOrAfter(size_t p, size_t g) const {
        const size_t bfirst = (g + B - 1) / B;
        if (bfirst >= nblocks) return localSize(p);
        size_t r = bfirst / P;
        if (blockOfRound(p, r) < bfirst) ++r;
        if (blockOfRound(p, r) >= nblocks) return localSize(p);
        return r * B;
    }
};

// Process grid and distribution state.
struct Grid {
    size_t n = 0;
    size_t Pr = 1, Pc = 1;  // grid dimensions
    size_t r = 0, c = 0;    // my coordinates (rank = r * Pc + c)
    MPI_Comm rowComm = MPI_COMM_NULL;  // processes with the same r (varying c)
    MPI_Comm colComm = MPI_COMM_NULL;  // processes with the same c (varying r)
    Dist rows, cols;
    size_t nlocR = 0, nlocC = 0;  // local sub-matrix dimensions
    std::vector<size_t> rowG;     // local row -> global row

    void init(size_t n_, int size, int rank) {
        n = n_;
        const size_t P = static_cast<size_t>(size);
        // Pc = largest divisor of P with Pc <= sqrt(P); Pr >= Pc (a "tall"
        // grid keeps the per-step work of the process columns balanced and
        // makes the row panel, which is broadcast along rows, small).
        Pc = 1;
        for (size_t d = 1; d * d <= P; ++d)
            if (P % d == 0) Pc = d;
        Pr = P / Pc;
        // Panel width: 128 gives the best kernel efficiency; 64 gives a finer
        // (better balanced) column distribution when there are few column
        // blocks per process column.
        NB = (n >= 2048 * Pc) ? 128 : 64;
        r = static_cast<size_t>(rank) / Pc;
        c = static_cast<size_t>(rank) % Pc;
        MPI_Comm_split(MPI_COMM_WORLD, static_cast<int>(r), static_cast<int>(c), &rowComm);
        MPI_Comm_split(MPI_COMM_WORLD, static_cast<int>(c), static_cast<int>(r), &colComm);
        // Row block size: as large as possible (<= 32) while giving every
        // process row several blocks for load balance.
        size_t DB = 32;
        while (DB > MR && (n + DB - 1) / DB < 4 * Pr) DB /= 2;
        rows.init(n, Pr, r, DB);
        cols.init(n, Pc, c, NB);
        nlocR = rows.localSize(r);
        nlocC = cols.localSize(c);
        rowG.resize(nlocR);
        for (size_t l = 0; l < nlocR; ++l) rowG[l] = rows.globalIndex(r, l);
    }
    void finalize() {
        MPI_Comm_free(&rowComm);
        MPI_Comm_free(&colComm);
    }
    int rankOf(size_t rr, size_t cc) const { return static_cast<int>(rr * Pc + cc); }
};

Grid g;

// All-gather global rows [g0, g1) and (local) columns [c0, c0+cols) of the
// row-distributed matrix `local` (row stride ld) inside a process column
// (communicator colComm); result `out` is (g1-g0) x cols ordered by global row.
void gatherRowRange(const double* local, size_t ld, size_t c0, size_t cols, size_t g0, size_t g1,
                    std::vector<double>& out, std::vector<double>& sendBuf,
                    std::vector<double>& recvBuf) {
    const Dist& d = g.rows;
    const size_t P = d.P;
    const size_t rows = g1 - g0;
    out.resize(rows * cols);
    if (rows == 0) return;
    const size_t b0 = g0 / d.B;
    const size_t b1 = (g1 - 1) / d.B;
    std::vector<int> counts(P, 0), displs(P, 0);
    for (size_t b = b0; b <= b1; ++b) {
        const size_t lo = std::max(g0, b * d.B);
        const size_t hi = std::min(g1, b * d.B + d.blockSize(b));
        counts[d.owner(b)] += static_cast<int>((hi - lo) * cols);
    }
    {
        int off = 0;
        for (size_t p = 0; p < P; ++p) {
            displs[p] = off;
            off += counts[p];
        }
    }
    const size_t myCount = static_cast<size_t>(counts[d.me]);
    sendBuf.resize(std::max<size_t>(myCount, 1));
    {
        size_t pos = 0;
        for (size_t b = b0; b <= b1; ++b) {
            if (d.owner(b) != d.me) continue;
            const size_t lo = std::max(g0, b * d.B);
            const size_t hi = std::min(g1, b * d.B + d.blockSize(b));
            for (size_t gi = lo; gi < hi; ++gi) {
                const size_t l = d.localOffset(b) + (gi - b * d.B);
                std::memcpy(sendBuf.data() + pos, local + l * ld + c0, cols * sizeof(double));
                pos += cols;
            }
        }
    }
    recvBuf.resize(rows * cols);
    MPI_Allgatherv(sendBuf.data(), static_cast<int>(myCount), MPI_DOUBLE, recvBuf.data(),
                   counts.data(), displs.data(), MPI_DOUBLE, g.colComm);
    std::vector<size_t> cursor(P);
    for (size_t p = 0; p < P; ++p) cursor[p] = static_cast<size_t>(displs[p]);
    for (size_t b = b0; b <= b1; ++b) {
        const size_t p = d.owner(b);
        const size_t lo = std::max(g0, b * d.B);
        const size_t hi = std::min(g1, b * d.B + d.blockSize(b));
        const size_t cnt = (hi - lo) * cols;
        std::memcpy(out.data() + (lo - g0) * cols, recvBuf.data() + cursor[p], cnt * sizeof(double));
        cursor[p] += cnt;
    }
}

// ---------------------------------------------------------------------------
// Distributed Cholesky
// ---------------------------------------------------------------------------

// Aloc: local sub-matrix (nlocR x nlocC, row-major).  On return it holds the
// corresponding part of L (upper triangular part zeroed).  Returns false (on
// all ranks) if the matrix is not positive definite.
//
// The loop is software-pipelined with a lookahead of one panel: while the
// trailing update of panel k is running, panel k+1 is already factorized
// (by the processes owning it, which update its column block first) and its
// communication is in flight.
bool choleskyDecomposition(std::vector<double>& Aloc, const size_t n) {
    const Dist& rd = g.rows;
    const Dist& cd = g.cols;
    const size_t Pr = g.Pr, Pc = g.Pc;
    const size_t nlocR = g.nlocR, nlocC = g.nlocC;
    const size_t npanels = cd.nblocks;
    const size_t nLocColBlocks = cd.numLocalBlocks(g.c);
    const size_t tilesPerColBlock = NB / NR;

    std::vector<double> D, DT(NB * NB), sendBuf, recvBuf;
    // Row panel (A operand): packed tiles of my local rows below the diagonal
    // block, plus one status word.  Two buffers (current panel, next panel).
    const size_t maxTilesR = (nlocR + MR - 1) / MR;
    std::vector<double> ApkBuf[2] = {std::vector<double>(maxTilesR * MR * NB + 1),
                                     std::vector<double>(maxTilesR * MR * NB + 1)};
    // Column panel (B operand): per local column block NB/NR tiles of NB x NR.
    std::vector<double> Bpk(nLocColBlocks * tilesPerColBlock * NB * NR, 0.0);
    std::vector<std::vector<double>> sendC(Pc);
    std::vector<double> piece, recvB;
    std::vector<int> counts(Pr), displs(Pr);
    std::vector<MPI_Request> reqs;
    MPI_Request bcastReq = MPI_REQUEST_NULL;
    MPI_Request allgReq = MPI_REQUEST_NULL;
    bool stage1Done = true;

    // Per-local-row-tile info: first/last global row of the tile.
    std::vector<size_t> tileFirst(maxTilesR), tileLast(maxTilesR);
    for (size_t mt = 0; mt < maxTilesR; ++mt) {
        tileFirst[mt] = g.rowG[mt * MR];
        tileLast[mt] = g.rowG[std::min(nlocR, mt * MR + MR) - 1];
    }
    // The same for every process row (needed to interpret received panels).
    std::vector<std::vector<size_t>> tileFirstAll(Pr);
    for (size_t rr = 0; rr < Pr; ++rr) {
        const size_t nl = rd.localSize(rr);
        const size_t nt = (nl + MR - 1) / MR;
        tileFirstAll[rr].resize(nt);
        for (size_t mt = 0; mt < nt; ++mt) tileFirstAll[rr][mt] = rd.globalIndex(rr, mt * MR);
    }

    // Geometry of panel k.
    struct PanelInfo {
        size_t kb, kw, pr0, kc, lcol, lp0, myRows, myTiles, mt0;
        bool inPanelCol;
    };
    auto panelInfo = [&](size_t k) {
        PanelInfo p;
        p.kb = k * NB;
        p.kw = std::min(NB, n - p.kb);
        p.pr0 = p.kb + p.kw;
        p.kc = cd.owner(k);
        p.inPanelCol = (g.c == p.kc);
        p.lcol = cd.localOffset(k);
        p.lp0 = rd.firstLocalAtOrAfter(g.r, p.pr0);
        p.myRows = nlocR - p.lp0;
        p.myTiles = (p.myRows + MR - 1) / MR;
        p.mt0 = (p.lp0 + MR - 1) / MR;
        return p;
    };

    // --- Panel factorization (processes of the panel's column only) ---------
    // Gathers and factorizes the diagonal block, solves the panel and leaves
    // it in packed form in Apk (status word appended).
    auto factorPanel = [&](const PanelInfo& p, double* Apk) {
        double status = -1.0;
        gatherRowRange(Aloc.data(), nlocC, p.lcol, p.kw, p.kb, p.pr0, D, sendBuf, recvBuf);
        // Work on the transpose so that the inner loops are contiguous:
        // DT[c][i] = D[i][c].  Right-looking Cholesky on the lower triangle.
        const size_t kw = p.kw;
        for (size_t i = 0; i < kw; ++i)
            for (size_t cc = 0; cc < kw; ++cc) DT[cc * kw + i] = D[i * kw + cc];
        for (size_t cc = 0; cc < kw; ++cc) {
            double* col = DT.data() + cc * kw;
            const double val = col[cc];
            if (val <= 0.0) {
                status = static_cast<double>(p.kb + cc);
                break;
            }
            const double ljj = sqrt(val);
            col[cc] = ljj;
            for (size_t i = cc + 1; i < kw; ++i) col[i] /= ljj;
            for (size_t t = cc + 1; t < kw; ++t) {
                double* colt = DT.data() + t * kw;
                const double l = col[t];
                for (size_t i = t; i < kw; ++i) colt[i] -= l * col[i];
            }
        }
        if (status < 0.0) {
            // write the factorized diagonal block back into the owned rows
            const size_t l0 = rd.firstLocalAtOrAfter(g.r, p.kb);
            for (size_t l = l0; l < p.lp0; ++l) {
                const size_t i = g.rowG[l] - p.kb;
                double* row = Aloc.data() + l * nlocC + p.lcol;
                for (size_t cc = 0; cc <= i; ++cc) row[cc] = DT[cc * kw + i];
            }
            // Panel TRSM: L_ik = A_ik * L_kk^{-T}, per tile of MR rows directly
            // in the packed layout Apk[tile][t][MR].
            packA(Aloc.data() + p.lp0 * nlocC + p.lcol, nlocC, p.myRows, kw, Apk);
            for (size_t mt = 0; mt < p.myTiles; ++mt) {
                double* X = Apk + mt * kw * MR;
                for (size_t cc = 0; cc < kw; ++cc) {
                    const double* Lc = DT.data() + cc * kw;  // Lc[t] = L_kk[t][cc]
                    const v4d x = loadU(X + cc * MR) / bcast(Lc[cc]);
                    storeU(X + cc * MR, x);
                    for (size_t t = cc + 1; t < kw; ++t) {
                        storeU(X + t * MR, loadU(X + t * MR) - bcast(Lc[t]) * x);
                    }
                }
                const size_t rv = std::min(MR, p.myRows - mt * MR);
                for (size_t i = 0; i < rv; ++i) {
                    double* row = Aloc.data() + (p.lp0 + mt * MR + i) * nlocC + p.lcol;
                    for (size_t cc = 0; cc < kw; ++cc) row[cc] = X[cc * MR + i];
                }
            }
        }
        Apk[p.myTiles * p.kw * MR] = status;
    };

    // --- Panel distribution, part 1 (non-blocking) ----------------------------
    // A operand: broadcast along the process row.  B operand, stage 1: the
    // panel-column processes send the tiles whose rows fall into column
    // blocks owned by process column cc to (r, cc).
    auto startPanelComm = [&](const PanelInfo& p, double* Apk) {
        const size_t rowMsg = p.myTiles * p.kw * MR + 1;
        MPI_Ibcast(Apk, static_cast<int>(rowMsg), MPI_DOUBLE, static_cast<int>(p.kc), g.rowComm,
                   &bcastReq);
        reqs.clear();
        for (size_t rr = 0; rr < Pr; ++rr) {
            const size_t lp0r = rd.firstLocalAtOrAfter(rr, p.pr0);
            const std::vector<size_t>& tf = tileFirstAll[rr];
            size_t cnt = 0;
            for (size_t mt = (lp0r + MR - 1) / MR; mt < tf.size(); ++mt)
                if (cd.owner(tf[mt] / NB) == g.c) ++cnt;
            counts[rr] = static_cast<int>(cnt * p.kw * MR);
            displs[rr] = (rr == 0) ? 0 : displs[rr - 1] + counts[rr - 1];
        }
        const size_t myPiece = static_cast<size_t>(counts[g.r]);
        recvB.resize(static_cast<size_t>(displs[Pr - 1] + counts[Pr - 1]));
        if (p.inPanelCol) {
            for (size_t cc = 0; cc < Pc; ++cc) sendC[cc].clear();
            for (size_t mt = 0; mt < p.myTiles; ++mt) {
                const size_t J = tileFirst[p.mt0 + mt] / NB;
                std::vector<double>& buf = sendC[cd.owner(J)];
                const double* src = Apk + mt * p.kw * MR;
                buf.insert(buf.end(), src, src + p.kw * MR);
            }
            for (size_t cc = 0; cc < Pc; ++cc) {
                if (cc == g.c || sendC[cc].empty()) continue;
                reqs.emplace_back();
                MPI_Isend(sendC[cc].data(), static_cast<int>(sendC[cc].size()), MPI_DOUBLE,
                          g.rankOf(g.r, cc), 2, MPI_COMM_WORLD, &reqs.back());
            }
            piece.swap(sendC[g.c]);
        } else {
            piece.resize(myPiece);
            if (myPiece > 0) {
                reqs.emplace_back();
                MPI_Irecv(piece.data(), static_cast<int>(myPiece), MPI_DOUBLE,
                          g.rankOf(g.r, p.kc), 2, MPI_COMM_WORLD, &reqs.back());
            }
        }
        stage1Done = false;
    };
    // B operand, stage 2: all-gather the pieces inside the process column
    // (non-blocking; started as soon as stage 1 has completed).
    auto startStage2 = [&]() {
        reqs.clear();
        stage1Done = true;
        MPI_Iallgatherv(piece.data(), counts[g.r], MPI_DOUBLE, recvB.data(), counts.data(),
                        displs.data(), MPI_DOUBLE, g.colComm, &allgReq);
    };
    // Drive progress of the outstanding communication (called between blocks
    // of the trailing update).
    auto progressComm = [&]() {
        int flag = 1;
        if (!stage1Done) {
            if (!reqs.empty())
                MPI_Testall(static_cast<int>(reqs.size()), reqs.data(), &flag, MPI_STATUSES_IGNORE);
            if (flag) startStage2();
        } else if (allgReq != MPI_REQUEST_NULL) {
            MPI_Test(&allgReq, &flag, MPI_STATUS_IGNORE);
        }
        if (bcastReq != MPI_REQUEST_NULL) MPI_Test(&bcastReq, &flag, MPI_STATUS_IGNORE);
    };
    // --- Panel distribution, part 2 (completes the exchange) -----------------
    // B operand, stage 2: all-gather the pieces inside the process column and
    // re-pack them into Bpk[localColBlock][tile][t][NR].  Returns the status.
    auto finishPanelComm = [&](const PanelInfo& p, double* Apk) {
        if (!stage1Done) {
            MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);
            startStage2();
        }
        MPI_Wait(&allgReq, MPI_STATUS_IGNORE);
        MPI_Wait(&bcastReq, MPI_STATUS_IGNORE);
        const double status = Apk[p.myTiles * p.kw * MR];
        if (status >= 0.0 || p.pr0 >= n) return status;
        for (size_t rr = 0; rr < Pr; ++rr) {
            const size_t lp0r = rd.firstLocalAtOrAfter(rr, p.pr0);
            const std::vector<size_t>& tf = tileFirstAll[rr];
            const double* src = recvB.data() + static_cast<size_t>(displs[rr]);
            for (size_t mt = (lp0r + MR - 1) / MR; mt < tf.size(); ++mt) {
                const size_t J = tf[mt] / NB;
                if (cd.owner(J) != g.c) continue;
                const size_t q = tf[mt] - J * NB;  // row offset inside the column block
                double* dst = Bpk.data() + (cd.localOffset(J) / NB) * tilesPerColBlock * NB * NR +
                              (q / NR) * p.kw * NR + (q % NR);
                for (size_t t = 0; t < p.kw; ++t) storeU(dst + t * NR, loadU(src + t * MR));
                src += p.kw * MR;
            }
        }
        return status;
    };

    // --- Trailing update of one local column block ----------------------------
    // A_ij -= L_ik * L_jk^T, lower triangle only (tiles rounded up to NR).
    auto updateBlock = [&](const PanelInfo& p, const double* Apk, size_t jl) {
        const size_t J = cd.blockOfRound(g.c, jl);
        const size_t jb = J * NB;
        const size_t hj = cd.blockSize(J);
        const size_t nTiles = (hj + NR - 1) / NR;
        const double* Bp = Bpk.data() + jl * tilesPerColBlock * NB * NR;
        double* Ccol = Aloc.data() + jl * NB;
        for (size_t mt = p.mt0; mt < p.mt0 + p.myTiles; ++mt) {
            if (tileLast[mt] < jb) continue;  // tile entirely above this column block
            const size_t l = mt * MR;
            const size_t rv = std::min(MR, nlocR - l);
            const size_t ntEnd = std::min(nTiles, (tileLast[mt] - jb) / NR + 1);
            const double* Ap = Apk + (mt - p.mt0) * p.kw * MR;
            double* Crow = Ccol + l * nlocC;
            for (size_t nt = 0; nt < ntEnd; ++nt) {
                const size_t cv = std::min(NR, hj - nt * NR);
                microKernel<true>(p.kw, Ap, Bp + nt * p.kw * NR, Crow + nt * NR, nlocC, rv, cv);
            }
        }
    };

    // --- Main loop with one-panel lookahead -------------------------------------
    PanelInfo cur = panelInfo(0);
    double* ApkCur = ApkBuf[0].data();
    double* ApkNext = ApkBuf[1].data();
    if (cur.inPanelCol) factorPanel(cur, ApkCur);
    startPanelComm(cur, ApkCur);
    double status = finishPanelComm(cur, ApkCur);

    for (size_t k = 0; k < npanels; ++k) {
        if (status >= 0.0) {
            if (g_rank == 0) {
                printf("Error: Matrix is not positive definite at diagonal element %zu\n",
                       static_cast<size_t>(status));
            }
            return false;
        }
        if (cur.pr0 >= n) break;  // last panel: nothing below the diagonal block

        const bool haveNext = (k + 1 < npanels);
        PanelInfo next;
        size_t jlNext = static_cast<size_t>(-1);  // local index of column block k+1 (if mine)
        if (haveNext) {
            next = panelInfo(k + 1);
            if (next.inPanelCol) {
                // Lookahead: bring column block k+1 up to date, factorize the
                // next panel and start sending it.
                jlNext = cd.localOffset(k + 1) / NB;
                updateBlock(cur, ApkCur, jlNext);
                factorPanel(next, ApkNext);
            }
            startPanelComm(next, ApkNext);
        }
        // Remaining trailing update of panel k.
        for (size_t jl = 0; jl < nLocColBlocks; ++jl) {
            if (cd.blockOfRound(g.c, jl) <= k || jl == jlNext) continue;
            updateBlock(cur, ApkCur, jl);
            if (haveNext) progressComm();
        }
        if (haveNext) {
            status = finishPanelComm(next, ApkNext);
            cur = next;
            std::swap(ApkCur, ApkNext);
        }
    }

    // Zero out the upper triangular part of the local sub-matrix.
    for (size_t l = 0; l < nlocR; ++l) {
        const size_t gi = g.rowG[l];
        double* row = Aloc.data() + l * nlocC;
        for (size_t jl = 0; jl < nLocColBlocks; ++jl) {
            const size_t J = cd.blockOfRound(g.c, jl);
            const size_t hj = cd.blockSize(J);
            for (size_t cc = 0; cc < hj; ++cc)
                if (J * NB + cc > gi) row[jl * NB + cc] = 0.0;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Matrix generation (distributed, communication-free): A = B * B^T + n * I
// with B random.  The random stream is identical to the sequential version;
// every process walks the full stream and keeps the rows of B it needs (its
// local rows, and the rows matching its local column blocks).  The products
// are accumulated in the same (sequential k) order as the original code.
// ---------------------------------------------------------------------------

void generatePositiveDefiniteMatrix(std::vector<double>& Aloc, const size_t n) {
    const Dist& cd = g.cols;
    const size_t nlocR = g.nlocR, nlocC = g.nlocC;
    const size_t myTiles = (nlocR + MR - 1) / MR;
    const size_t nLocColBlocks = cd.numLocalBlocks(g.c);

    // Packed operands written directly while walking the random stream.
    std::vector<double> Apk(myTiles * MR * n + MR, 0.0);               // [tile][t][MR]
    std::vector<double> Bpk(nLocColBlocks * (NB / NR) * NR * n, 0.0);  // [colblk][tile][t][NR]
    {
        std::vector<size_t> localRowOf(n, static_cast<size_t>(-1));
        for (size_t l = 0; l < nlocR; ++l) localRowOf[g.rowG[l]] = l;
        unsigned int seed = 42;
        for (size_t i = 0; i < n; ++i) {
            const size_t l = localRowOf[i];
            const size_t J = i / NB;
            double* dstA = nullptr;
            double* dstB = nullptr;
            if (l != static_cast<size_t>(-1)) dstA = Apk.data() + (l / MR) * MR * n + (l % MR);
            if (cd.owner(J) == g.c) {
                const size_t q = i - J * NB;
                dstB = Bpk.data() + (cd.localOffset(J) / NB) * (NB / NR) * NR * n +
                       (q / NR) * NR * n + (q % NR);
            }
            if (dstA == nullptr && dstB == nullptr) {
                for (size_t j = 0; j < n; ++j) (void)rand_r(&seed);
            } else {
                for (size_t j = 0; j < n; ++j) {
                    const double v = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
                    if (dstA) dstA[j * MR] = v;
                    if (dstB) dstB[j * NR] = v;
                }
            }
        }
    }

    std::fill(Aloc.begin(), Aloc.end(), 0.0);
    for (size_t jl = 0; jl < nLocColBlocks; ++jl) {
        const size_t J = cd.blockOfRound(g.c, jl);
        const size_t hj = cd.blockSize(J);
        const size_t nTiles = (hj + NR - 1) / NR;
        const double* Bp = Bpk.data() + jl * (NB / NR) * NR * n;
        for (size_t mt = 0; mt < myTiles; ++mt) {
            const size_t rv = std::min(MR, nlocR - mt * MR);
            double* Crow = Aloc.data() + mt * MR * nlocC + jl * NB;
            for (size_t nt = 0; nt < nTiles; ++nt) {
                const size_t cv = std::min(NR, hj - nt * NR);
                microKernel<false, false>(n, Apk.data() + mt * n * MR, Bp + nt * n * NR,
                                          Crow + nt * NR, nlocC, rv, cv);
            }
        }
    }

    // Add diagonal dominance to ensure positive definiteness
    for (size_t l = 0; l < nlocR; ++l) {
        const size_t gi = g.rowG[l];
        const size_t J = gi / NB;
        if (cd.owner(J) == g.c) Aloc[l * nlocC + cd.localOffset(J) + (gi - J * NB)] += n;
    }
}

// ---------------------------------------------------------------------------
// Validation (distributed): reconstruct L * L^T for the local sub-matrix and
// compare with the local part of the original matrix.
// ---------------------------------------------------------------------------

bool validateCholesky(const std::vector<double>& Lloc, const std::vector<double>& AorigLoc,
                      const size_t n) {
    const Dist& rd = g.rows;
    const Dist& cd = g.cols;
    const size_t Pr = g.Pr, Pc = g.Pc;
    const size_t nlocR = g.nlocR, nlocC = g.nlocC;
    const size_t myTiles = (nlocR + MR - 1) / MR;
    const size_t nLocColBlocks = cd.numLocalBlocks(g.c);

    // 1. full rows of L for my local rows: all-gather along the process row.
    std::vector<double> Lrows(nlocR * n);
    {
        std::vector<int> counts(Pc), displs(Pc);
        for (size_t cc = 0; cc < Pc; ++cc) {
            counts[cc] = static_cast<int>(nlocR * cd.localSize(cc));
            displs[cc] = (cc == 0) ? 0 : displs[cc - 1] + counts[cc - 1];
        }
        std::vector<double> recv(nlocR * n);
        MPI_Allgatherv(Lloc.data(), static_cast<int>(nlocR * nlocC), MPI_DOUBLE, recv.data(),
                       counts.data(), displs.data(), MPI_DOUBLE, g.rowComm);
        for (size_t cc = 0; cc < Pc; ++cc) {
            const size_t ncc = cd.localSize(cc);
            const size_t nb = cd.numLocalBlocks(cc);
            for (size_t jl = 0; jl < nb; ++jl) {
                const size_t J = cd.blockOfRound(cc, jl);
                const size_t hj = cd.blockSize(J);
                for (size_t l = 0; l < nlocR; ++l) {
                    std::memcpy(Lrows.data() + l * n + J * NB,
                                recv.data() + static_cast<size_t>(displs[cc]) + l * ncc + jl * NB,
                                hj * sizeof(double));
                }
            }
        }
    }
    std::vector<double> Apk(myTiles * MR * n + MR);
    packA(Lrows.data(), n, nlocR, n, Apk.data());

    // 2. full rows of L for the rows of my column blocks: all-gather along the
    //    process column (every process contributes its matching local rows).
    std::vector<double> Bpk(nLocColBlocks * (NB / NR) * NR * n, 0.0);
    {
        std::vector<int> counts(Pr), displs(Pr);
        std::vector<size_t> rowsOf(Pr, 0);
        for (size_t rr = 0; rr < Pr; ++rr) {
            const size_t nl = rd.localSize(rr);
            for (size_t l = 0; l < nl; ++l)
                if (cd.owner(rd.globalIndex(rr, l) / NB) == g.c) ++rowsOf[rr];
            counts[rr] = static_cast<int>(rowsOf[rr] * n);
            displs[rr] = (rr == 0) ? 0 : displs[rr - 1] + counts[rr - 1];
        }
        std::vector<double> send(std::max<size_t>(rowsOf[g.r] * n, 1));
        {
            size_t pos = 0;
            for (size_t l = 0; l < nlocR; ++l) {
                if (cd.owner(g.rowG[l] / NB) != g.c) continue;
                std::memcpy(send.data() + pos * n, Lrows.data() + l * n, n * sizeof(double));
                ++pos;
            }
        }
        std::vector<double> recv(nlocC * n);
        MPI_Allgatherv(send.data(), static_cast<int>(rowsOf[g.r] * n), MPI_DOUBLE, recv.data(),
                       counts.data(), displs.data(), MPI_DOUBLE, g.colComm);
        for (size_t rr = 0; rr < Pr; ++rr) {
            const size_t nl = rd.localSize(rr);
            const double* src = recv.data() + static_cast<size_t>(displs[rr]);
            for (size_t l = 0; l < nl; ++l) {
                const size_t gi = rd.globalIndex(rr, l);
                const size_t J = gi / NB;
                if (cd.owner(J) != g.c) continue;
                const size_t q = gi - J * NB;
                double* dst = Bpk.data() + (cd.localOffset(J) / NB) * (NB / NR) * NR * n +
                              (q / NR) * NR * n + (q % NR);
                for (size_t t = 0; t < n; ++t) dst[t * NR] = src[t];
                src += n;
            }
        }
    }
    Lrows.clear();
    Lrows.shrink_to_fit();

    // 3. recon = L * L^T on my sub-matrix (rows of L are zero beyond the diagonal)
    std::vector<double> recon(nlocR * nlocC, 0.0);
    for (size_t jl = 0; jl < nLocColBlocks; ++jl) {
        const size_t J = cd.blockOfRound(g.c, jl);
        const size_t hj = cd.blockSize(J);
        const size_t K = J * NB + hj;  // L rows of this block are zero beyond column K
        const size_t nTiles = (hj + NR - 1) / NR;
        const double* Bp = Bpk.data() + jl * (NB / NR) * NR * n;
        for (size_t mt = 0; mt < myTiles; ++mt) {
            const size_t rv = std::min(MR, nlocR - mt * MR);
            double* Crow = recon.data() + mt * MR * nlocC + jl * NB;
            for (size_t nt = 0; nt < nTiles; ++nt) {
                const size_t cv = std::min(NR, hj - nt * NR);
                microKernel<false>(K, Apk.data() + mt * n * MR, Bp + nt * n * NR, Crow + nt * NR,
                                   nlocC, rv, cv);
            }
        }
    }

    double maxError = 0.0;
    double relError = 0.0;
    for (size_t i = 0; i < nlocR * nlocC; ++i) {
        const double error = fabs(recon[i] - AorigLoc[i]);
        maxError = std::max(maxError, error);
        const double rel = error / (fabs(AorigLoc[i]) + 1e-10);
        relError = std::max(relError, rel);
    }
    double red[2] = {maxError, relError};
    MPI_Allreduce(MPI_IN_PLACE, red, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    maxError = red[0];
    relError = red[1];

    if (g_rank == 0) {
        printf("Max absolute error: %.10e\n", maxError);
        printf("Max relative error: %.10e\n", relError);
    }

    if (relError > 1e-6) {
        if (g_rank == 0) printf("Validation failed: relative error too large\n");
        return false;
    }
    return true;
}

// Gather the distributed matrix to rank 0 (row-major n x n).
void gatherMatrix(const std::vector<double>& Aloc, std::vector<double>& A, const size_t n) {
    const Dist& rd = g.rows;
    const Dist& cd = g.cols;
    const size_t P = g.Pr * g.Pc;
    std::vector<int> counts(P), displs(P);
    std::vector<double> tmp;
    if (g_rank == 0) {
        size_t off = 0;
        for (size_t p = 0; p < P; ++p) {
            counts[p] = static_cast<int>(rd.localSize(p / g.Pc) * cd.localSize(p % g.Pc));
            displs[p] = static_cast<int>(off);
            off += static_cast<size_t>(counts[p]);
        }
        tmp.resize(n * n);
    }
    MPI_Gatherv(Aloc.data(), static_cast<int>(g.nlocR * g.nlocC), MPI_DOUBLE, tmp.data(),
                counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (g_rank == 0) {
        A.resize(n * n);
        for (size_t p = 0; p < P; ++p) {
            const size_t rr = p / g.Pc, cc = p % g.Pc;
            const size_t nlr = rd.localSize(rr), nlc = cd.localSize(cc);
            const size_t nb = cd.numLocalBlocks(cc);
            const double* src = tmp.data() + static_cast<size_t>(displs[p]);
            for (size_t l = 0; l < nlr; ++l) {
                const size_t gi = rd.globalIndex(rr, l);
                for (size_t jl = 0; jl < nb; ++jl) {
                    const size_t J = cd.blockOfRound(cc, jl);
                    std::memcpy(A.data() + gi * n + J * NB, src + l * nlc + jl * NB,
                                cd.blockSize(J) * sizeof(double));
                }
            }
        }
    }
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_size);

    size_t n = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (g_rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (g_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    g.init(n, g_size, g_rank);

    if (g_rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d (%zu x %zu grid, panel width %zu)\n", g_size, g.Pr, g.Pc, NB);
    }

    // Allocate local part of the matrix
    std::vector<double> Aloc(g.nlocR * g.nlocC);
    std::vector<double> AorigLoc;

    // Generate positive definite matrix
    if (g_rank == 0) printf("Generating positive definite matrix...\n");
    generatePositiveDefiniteMatrix(Aloc, n);

    if (validate) {
        AorigLoc = Aloc;  // Save original for validation
    }

    // Perform Cholesky decomposition
    if (g_rank == 0) printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    bool success = choleskyDecomposition(Aloc, n);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long maxDuration = 0;
    MPI_Reduce(&duration, &maxDuration, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (!success) {
        if (g_rank == 0) printf("Cholesky decomposition failed\n");
        g.finalize();
        MPI_Finalize();
        return 1;
    }

    if (g_rank == 0) {
        printf("Computation time: %ld ms\n", maxDuration);

        // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (maxDuration / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Print results for external validation
    if (printResults) {
        std::vector<double> A;
        gatherMatrix(Aloc, A, n);
        if (g_rank == 0) print_results(A, "CholeskyL");
    }

    // Validation
    int rc = 0;
    if (validate) {
        if (g_rank == 0) printf("Validating result...\n");
        bool valid = validateCholesky(Aloc, AorigLoc, n);

        if (g_rank == 0) {
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
        rc = valid ? 0 : 1;
    }

    g.finalize();
    MPI_Finalize();
    return rc;
}
