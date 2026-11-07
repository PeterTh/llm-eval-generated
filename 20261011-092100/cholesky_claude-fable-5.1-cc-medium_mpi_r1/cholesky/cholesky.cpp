#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// MPI-parallel Cholesky decomposition.
// Decomposes positive definite matrix A into L * L^T where L is lower triangular.
//
// Data distribution: 2D block-cyclic over a Pr x Pc process grid.
//   * rows are dealt in blocks of DB rows round-robin over the Pr process rows,
//   * columns are dealt in blocks of NB columns round-robin over the Pc process
//     columns (NB is also the panel width of the blocked algorithm).
// Every rank stores its rows x columns sub-matrix contiguously (row-major, row
// stride = number of local columns), rows and columns in increasing global order.
//
// Algorithm: right-looking blocked Cholesky with one panel of lookahead.
// For panel (block column) k, owned by process column k % Pc:
//   1. the NB x NB diagonal block is allgathered within that process column and
//      factored redundantly by its Pr ranks (cheap; replaces a broadcast),
//   2. those ranks solve their rows below the block: L_ik = A_ik * L_kk^-T,
//   3. the panel rows L_ik are broadcast along each process row (nonblocking),
//   4. every rank collects, within its process column, the panel rows L_jk for
//      the columns j it owns (nonblocking allgather),
//   5. every rank applies A_ij -= L_ik * L_jk^T to its local block.
// Lookahead: once panel k is available the update is first applied to block
// column k+1 only, panel k+1 is factored and its communication started, and the
// remaining update of panel k overlaps with that communication.
//
// The floating-point operation order of every element is independent of the
// process grid, so the result is bitwise identical for any process count.

namespace {

constexpr size_t NB = 48;  // panel width and column distribution block
constexpr size_t DB = 8;   // row distribution block
constexpr size_t JW = 8;   // columns per packed panel tile
constexpr size_t JC = 48;  // tiles per L2 column chunk of the update
static_assert(NB % DB == 0, "panel width must be a multiple of the row block");
static_assert(DB % 4 == 0, "row block must be a multiple of the row group size");
static_assert(DB == JW, "row block must match the tile width");

// --- block-cyclic index helpers ---------------------------------------------
// Indices are dealt in blocks of bs round-robin over np owners.

// Global index of the li-th local index of owner r.
inline size_t cyclicGlobal(size_t li, size_t bs, int np, int r) {
    return (li / bs * (size_t)np + (size_t)r) * bs + li % bs;
}

// Number of indices < g owned by r.
inline size_t cyclicBefore(size_t g, size_t bs, int np, int r) {
    const size_t b = g / bs;  // block containing g
    if (b < (size_t)r) return 0;
    size_t cnt = (b > (size_t)r) ? ((b - 1 - (size_t)r) / (size_t)np + 1) * bs : 0;
    if ((b - (size_t)r) % (size_t)np == 0) {
        cnt += g - b * bs;  // leading part of block b, owned by r
    }
    return cnt;
}

struct Layout {
    size_t n = 0;
    int rank = 0, P = 1;
    int Pr = 1, Pc = 1;  // process grid
    int pr = 0, pc = 0;  // my grid coordinates
    size_t nloc = 0;     // local rows
    size_t ncloc = 0;    // local columns
    MPI_Comm rowComm = MPI_COMM_NULL;  // ranks sharing my process row (index pc)
    MPI_Comm colComm = MPI_COMM_NULL;  // ranks sharing my process column (index pr)

    void init(size_t n_, int rank_, int P_) {
        n = n_;
        rank = rank_;
        P = P_;
        // closest-to-square grid with Pr >= Pc
        Pc = 1;
        for (int c = 1; (long)c * c <= P; ++c) {
            if (P % c == 0) Pc = c;
        }
        Pr = P / Pc;
        pr = rank / Pc;
        pc = rank % Pc;
        MPI_Comm_split(MPI_COMM_WORLD, pr, pc, &rowComm);
        MPI_Comm_split(MPI_COMM_WORLD, pc, pr, &colComm);
        nloc = localRows(pr);
        ncloc = localCols(pc);
    }

    void finalize() {
        MPI_Comm_free(&rowComm);
        MPI_Comm_free(&colComm);
    }

    static int rowOf(int r, int Pc_) { return r / Pc_; }
    static int colOf(int r, int Pc_) { return r % Pc_; }

    size_t globalRow(size_t li, int r) const { return cyclicGlobal(li, DB, Pr, r); }
    size_t globalRow(size_t li) const { return globalRow(li, pr); }
    size_t rowsBefore(size_t g, int r) const { return cyclicBefore(g, DB, Pr, r); }
    size_t rowsBefore(size_t g) const { return rowsBefore(g, pr); }
    size_t localRows(int r) const { return rowsBefore(n, r); }

    size_t globalCol(size_t lj, int c) const { return cyclicGlobal(lj, NB, Pc, c); }
    size_t globalCol(size_t lj) const { return globalCol(lj, pc); }
    size_t colsBefore(size_t g, int c) const { return cyclicBefore(g, NB, Pc, c); }
    size_t colsBefore(size_t g) const { return colsBefore(g, pc); }
    size_t localCols(int c) const { return colsBefore(n, c); }
    int colOwner(size_t g) const { return (int)((g / NB) % (size_t)Pc); }
};

// Unblocked Cholesky of a dense nb x nb block (row stride nb) in place.
// Returns the failing diagonal index or -1 on success.
long factorDiagonalBlock(double* blk, const size_t nb) {
    for (size_t i = 0; i < nb; ++i) {
        double* ri = blk + i * nb;
        for (size_t j = 0; j < i; ++j) {
            const double* rj = blk + j * nb;
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k) {
                sum += ri[k] * rj[k];
            }
            ri[j] = (ri[j] - sum) / rj[j];
        }
        double sum = 0.0;
        for (size_t k = 0; k < i; ++k) {
            sum += ri[k] * ri[k];
        }
        const double val = ri[i] - sum;
        if (val <= 0.0) {
            return (long)i;
        }
        ri[i] = sqrt(val);
    }
    return -1;
}

// Triangular solve of R rows at once: row[r][c] for c < nb becomes L_i,(j0+c).
// LkkT is the transposed diagonal block (LkkT[d * nb + c] = L_kk[c][d]).
template <int R>
inline void trsmRows(double* const* rows, const double* __restrict LkkT, const size_t nb) {
    for (size_t d = 0; d < nb; ++d) {
        const double* ld = LkkT + d * nb;
        const double ldd = ld[d];
        for (int r = 0; r < R; ++r) {
            double* __restrict x = rows[r];
            const double xd = x[d] / ldd;
            x[d] = xd;
            for (size_t c = d + 1; c < nb; ++c) {
                x[c] -= xd * ld[c];
            }
        }
    }
}

// --- Trailing update micro-kernel -------------------------------------------
// The collected panel rows L_jk are stored tile-packed by local column: columns
// are grouped into tiles of JW consecutive local columns, and within a tile the
// nb panel entries of the JW columns are interleaved:
//   packed[(tile * nb + c) * JW + jj]  ==  L_(j, j0 + c)  with j = tile*JW + jj.

typedef double v4d __attribute__((vector_size(32)));

inline v4d loadv(const double* p) {
    v4d v;
    memcpy(&v, p, sizeof(v));
    return v;
}
inline void storev(double* p, v4d v) { memcpy(p, &v, sizeof(v)); }
inline v4d bcastv(double x) { return v4d{x, x, x, x}; }
// Pin a loaded vector to a register so the compiler does not fold the load into
// every FMA that uses it (which would make the kernel load-port bound).
inline void pin(v4d& v) { asm volatile("" : "+x"(v)); }

// ar[0..JW) -= acc, but only for local columns j + jj < end.
inline void storeTile(double* ar, v4d acc0, v4d acc1, const size_t j, const size_t end) {
    if (j + JW <= end) {
        storev(ar, loadv(ar) - acc0);
        storev(ar + 4, loadv(ar + 4) - acc1);
    } else {
        double tmp[JW];
        storev(tmp, acc0);
        storev(tmp + 4, acc1);
        const size_t len = (end > j) ? end - j : 0;
        for (size_t jj = 0; jj < len; ++jj) {
            ar[jj] -= tmp[jj];
        }
    }
}

// A[r][j] -= sum_c L[r][c] * P[c][j] for 4 rows and tiles [tb, te).
// Row r only stores local columns j < end[r]. jbase is the local column of tile 0.
inline void updateTiles4(double* const* a, const double* const* l, const size_t* end,
                         const double* __restrict packed, const size_t nb, const size_t jbase,
                         const size_t tb, const size_t te) {
    const double* __restrict l0 = l[0];
    const double* __restrict l1 = l[1];
    const double* __restrict l2 = l[2];
    const double* __restrict l3 = l[3];
    for (size_t t = tb; t < te; ++t) {
        const double* __restrict pk = packed + t * nb * JW;
        v4d a00 = bcastv(0.0), a01 = a00, a10 = a00, a11 = a00;
        v4d a20 = a00, a21 = a00, a30 = a00, a31 = a00;
        for (size_t c = 0; c < nb; ++c) {
            v4d p0 = loadv(pk + c * JW);
            v4d p1 = loadv(pk + c * JW + 4);
            pin(p0);
            pin(p1);
            v4d b;
            b = bcastv(l0[c]); a00 += b * p0; a01 += b * p1;
            b = bcastv(l1[c]); a10 += b * p0; a11 += b * p1;
            b = bcastv(l2[c]); a20 += b * p0; a21 += b * p1;
            b = bcastv(l3[c]); a30 += b * p0; a31 += b * p1;
        }
        const size_t j = jbase + t * JW;
        storeTile(a[0] + j, a00, a01, j, end[0]);
        storeTile(a[1] + j, a10, a11, j, end[1]);
        storeTile(a[2] + j, a20, a21, j, end[2]);
        storeTile(a[3] + j, a30, a31, j, end[3]);
    }
}

inline void updateTiles1(double* a, const double* __restrict l0, const size_t end,
                         const double* __restrict packed, const size_t nb, const size_t jbase,
                         const size_t tb, const size_t te) {
    for (size_t t = tb; t < te; ++t) {
        const double* __restrict pk = packed + t * nb * JW;
        v4d a00 = bcastv(0.0), a01 = a00;
        for (size_t c = 0; c < nb; ++c) {
            v4d b = bcastv(l0[c]);
            pin(b);
            a00 += b * loadv(pk + c * JW);
            a01 += b * loadv(pk + c * JW + 4);
        }
        const size_t j = jbase + t * JW;
        storeTile(a + j, a00, a01, j, end);
    }
}

// Pipelined panel factorization with one panel of lookahead.
class PanelFactorizer {
  public:
    PanelFactorizer(std::vector<double>& Aloc, const Layout& L) : A_(Aloc), L_(L), n_(L.n) {
        blockBuf_.resize(NB * NB);
        blockRecv_.resize(NB * NB);
        LkkT_.resize(NB * NB);
        for (auto& b : Lrows_) b.resize(L.nloc * NB + 1);
        colSend_.resize(L.nloc * NB);
        colRecv_.resize(L.ncloc * NB);
        packed_.resize((L.ncloc + JW) * NB);
        blkCounts_.resize(L.Pr);
        blkDispls_.resize(L.Pr);
        counts_.resize(L.Pr);
        displs_.resize(L.Pr);
    }

    // Factor the diagonal block of panel k, solve the rows below it (on the
    // owning process column) and start the panel communication.
    // Returns false if the matrix was found not to be positive definite; ranks
    // outside the owning process column learn that later from finishPanel().
    bool startPanel(const size_t k) {
        Panel& p = cur_;
        p.k = k;
        p.j0 = k * NB;
        p.nb = std::min(NB, n_ - p.j0);
        p.j1 = p.j0 + p.nb;
        p.liStart = L_.rowsBefore(p.j1);
        p.buf = (int)(k % 2);
        const size_t nBelow = L_.nloc - p.liStart;
        const int root = L_.colOwner(p.j0);
        std::vector<double>& lrows = Lrows_[p.buf];
        lrows[nBelow * p.nb] = -1.0;  // status element: < 0 means success
        bool ok = true;

        if (L_.pc == root) {
            const size_t lc0 = L_.colsBefore(p.j0);
            const long status = factorDiagonal(p.j0, p.nb, lc0);
            if (status >= 0) {
                if (L_.pr == 0) {  // printed once, by the first rank of the owning column
                    printf("Error: Matrix is not positive definite at diagonal element %zu\n",
                           (size_t)(p.j0 + status));
                }
                lrows[nBelow * p.nb] = (double)(p.j0 + status);
                ok = false;
            } else if (p.j1 < n_) {
                // TRSM of local rows below the diagonal block, then pack them
                size_t li = p.liStart;
                for (; li + 4 <= L_.nloc; li += 4) {
                    double* rows[4];
                    for (int r = 0; r < 4; ++r) rows[r] = A_.data() + (li + r) * L_.ncloc + lc0;
                    trsmRows<4>(rows, LkkT_.data(), p.nb);
                }
                for (; li < L_.nloc; ++li) {
                    double* rows[1] = {A_.data() + li * L_.ncloc + lc0};
                    trsmRows<1>(rows, LkkT_.data(), p.nb);
                }
                for (size_t r = 0; r < nBelow; ++r) {
                    memcpy(lrows.data() + r * p.nb, A_.data() + (p.liStart + r) * L_.ncloc + lc0,
                           p.nb * sizeof(double));
                }
            }
        }

        // broadcast the panel rows (+ status) along the process row
        MPI_Ibcast(lrows.data(), (int)(nBelow * p.nb + 1), MPI_DOUBLE, root, L_.rowComm,
                   &bcastReq_);
        p.state = Panel::BCAST;
        if (!ok) {
            MPI_Wait(&bcastReq_, MPI_STATUS_IGNORE);
            p.state = Panel::FAILED;
        }
        return ok;
    }

    // Let the pending communication of the started panel make progress.
    void poll() {
        Panel& p = cur_;
        if (p.state == Panel::BCAST) {
            int flag = 0;
            MPI_Test(&bcastReq_, &flag, MPI_STATUS_IGNORE);
            if (flag) onBcastDone();
        } else if (p.state == Panel::GATHER) {
            int flag = 0;
            MPI_Test(&gatherReq_, &flag, MPI_STATUS_IGNORE);
            if (flag) p.state = Panel::DONE;
        }
    }

    // Wait for the started panel, unpack it and make it the panel to apply.
    // Returns false if the factorization failed.
    bool finishPanel() {
        Panel& p = cur_;
        if (p.state == Panel::BCAST) {
            MPI_Wait(&bcastReq_, MPI_STATUS_IGNORE);
            onBcastDone();
        }
        if (p.state == Panel::GATHER) {
            MPI_Wait(&gatherReq_, MPI_STATUS_IGNORE);
            p.state = Panel::DONE;
        }
        if (p.state == Panel::FAILED) return false;
        if (p.j1 < n_) unpackPanel();
        upd_ = p;
        return true;
    }

    // Apply the update of the last finished panel to local columns [cb, ce)
    // of local rows >= rowFrom (global), polling the pending communication.
    void update(const size_t rowFrom, const size_t cb, const size_t ce) {
        const Panel& p = upd_;
        const size_t liFrom = L_.rowsBefore(rowFrom);
        if (cb >= ce || liFrom >= L_.nloc) return;
        const size_t ncloc = L_.ncloc;
        const size_t jbase = L_.colsBefore(p.j1);  // local column of tile 0
        const double* lrows = Lrows_[p.buf].data();
        const double* packed = packed_.data();
        const size_t tBegin = (cb - jbase) / JW;
        const size_t tEnd = (ce - jbase + JW - 1) / JW;
        for (size_t tc = tBegin; tc < tEnd; tc += JC) {
            const size_t tcEnd = std::min(tEnd, tc + JC);
            size_t li = liFrom;
            // Row groups of 4 never straddle a row block (DB % 4 == 0 and liFrom
            // is a block boundary); only the last partial block has a remainder.
            for (; li + 4 <= L_.nloc; li += 4) {
                double* a[4];
                const double* l[4];
                size_t end[4];
                for (int r = 0; r < 4; ++r) {
                    a[r] = A_.data() + (li + r) * ncloc;
                    l[r] = lrows + (li + r - p.liStart) * p.nb;
                    end[r] = L_.colsBefore(L_.globalRow(li + r) + 1);
                }
                const size_t te =
                    end[3] > jbase ? std::min(tcEnd, (end[3] - jbase + JW - 1) / JW) : 0;
                if (tc < te) {
                    updateTiles4(a, l, end, packed, p.nb, jbase, tc, te);
                    poll();
                }
            }
            for (; li < L_.nloc; ++li) {
                double* a = A_.data() + li * ncloc;
                const double* l = lrows + (li - p.liStart) * p.nb;
                const size_t end = L_.colsBefore(L_.globalRow(li) + 1);
                const size_t te = end > jbase ? std::min(tcEnd, (end - jbase + JW - 1) / JW) : 0;
                if (tc < te) updateTiles1(a, l, end, packed, p.nb, jbase, tc, te);
            }
        }
    }

  private:
    struct Panel {
        enum State { NONE, BCAST, GATHER, DONE, FAILED };
        size_t k = 0, j0 = 0, nb = 0, j1 = 0, liStart = 0;
        int buf = 0;
        State state = NONE;
    };

    // Allgather the diagonal block within the process column, factor it and
    // write the owned rows back. Returns the failing index or -1.
    long factorDiagonal(const size_t j0, const size_t nb, const size_t lc0) {
        const size_t j1 = j0 + nb;
        const size_t myFirst = L_.rowsBefore(j0);
        const size_t myCnt = L_.rowsBefore(j1) - myFirst;
        for (size_t q = 0; q < myCnt; ++q) {
            memcpy(blockBuf_.data() + q * nb, A_.data() + (myFirst + q) * L_.ncloc + lc0,
                   nb * sizeof(double));
        }
        int disp = 0;
        for (int r = 0; r < L_.Pr; ++r) {
            const size_t cnt = L_.rowsBefore(j1, r) - L_.rowsBefore(j0, r);
            blkCounts_[r] = (int)(cnt * nb);
            blkDispls_[r] = disp;
            disp += blkCounts_[r];
        }
        MPI_Allgatherv(blockBuf_.data(), (int)(myCnt * nb), MPI_DOUBLE, blockRecv_.data(),
                       blkCounts_.data(), blkDispls_.data(), MPI_DOUBLE, L_.colComm);
        for (int r = 0; r < L_.Pr; ++r) {
            const size_t first = L_.rowsBefore(j0, r);
            const size_t cnt = (size_t)blkCounts_[r] / nb;
            for (size_t q = 0; q < cnt; ++q) {
                const size_t i = L_.globalRow(first + q, r) - j0;
                memcpy(blockBuf_.data() + i * nb, blockRecv_.data() + blkDispls_[r] + q * nb,
                       nb * sizeof(double));
            }
        }

        const long status = factorDiagonalBlock(blockBuf_.data(), nb);
        if (status >= 0) return status;

        for (size_t q = 0; q < myCnt; ++q) {
            const size_t i = L_.globalRow(myFirst + q) - j0;
            memcpy(A_.data() + (myFirst + q) * L_.ncloc + lc0, blockBuf_.data() + i * nb,
                   nb * sizeof(double));
        }
        for (size_t i = 0; i < nb; ++i) {
            for (size_t c = 0; c < nb; ++c) {
                LkkT_[c * nb + i] = (c <= i) ? blockBuf_[i * nb + c] : 0.0;
            }
        }
        return -1;
    }

    // Number of rows >= j1 of process row r that lie in my column blocks.
    size_t contribRows(const size_t j1, const int r) const {
        size_t cnt = 0;
        const size_t nrows = L_.localRows(r);
        for (size_t li = L_.rowsBefore(j1, r); li < nrows; li += DB) {
            const size_t g = L_.globalRow(li, r);
            if (L_.colOwner(g) == L_.pc) cnt += std::min(DB, nrows - li);
        }
        return cnt;
    }

    // The row broadcast completed: check the status and start the column
    // allgather of the panel rows belonging to my columns.
    void onBcastDone() {
        Panel& p = cur_;
        const size_t nBelow = L_.nloc - p.liStart;
        if (Lrows_[p.buf][nBelow * p.nb] >= 0.0) {
            p.state = Panel::FAILED;
            return;
        }
        if (p.j1 >= n_) {
            p.state = Panel::DONE;
            return;
        }
        // pack my rows that are columns of mine
        const double* lrows = Lrows_[p.buf].data();
        size_t cnt = 0;
        for (size_t li = p.liStart; li < L_.nloc; li += DB) {
            const size_t g = L_.globalRow(li);
            if (L_.colOwner(g) != L_.pc) continue;
            const size_t rows = std::min(DB, L_.nloc - li);
            memcpy(colSend_.data() + cnt * p.nb, lrows + (li - p.liStart) * p.nb,
                   rows * p.nb * sizeof(double));
            cnt += rows;
        }
        int disp = 0;
        for (int r = 0; r < L_.Pr; ++r) {
            counts_[r] = (int)(contribRows(p.j1, r) * p.nb);
            displs_[r] = disp;
            disp += counts_[r];
        }
        MPI_Iallgatherv(colSend_.data(), (int)(cnt * p.nb), MPI_DOUBLE, colRecv_.data(),
                        counts_.data(), displs_.data(), MPI_DOUBLE, L_.colComm, &gatherReq_);
        p.state = Panel::GATHER;
    }

    // Unpack the collected panel rows into the tile-packed layout (by local column).
    void unpackPanel() {
        const Panel& p = cur_;
        const size_t nb = p.nb;
        const size_t jbase = L_.colsBefore(p.j1);
        for (int r = 0; r < L_.Pr; ++r) {
            const double* src = colRecv_.data() + displs_[r];
            const size_t nrows = L_.localRows(r);
            size_t q = 0;
            for (size_t li = L_.rowsBefore(p.j1, r); li < nrows; li += DB) {
                const size_t g = L_.globalRow(li, r);
                if (L_.colOwner(g) != L_.pc) continue;
                const size_t rows = std::min(DB, nrows - li);
                const size_t lj = L_.colsBefore(g) - jbase;  // local column of row g
                const double* s = src + q * nb;
                double* dst = packed_.data() + (lj / JW) * nb * JW + (lj % JW);
                if (rows == JW) {
                    // DB == JW aligned run: transpose JW rows into one tile
                    for (size_t c = 0; c < nb; ++c) {
                        for (size_t jj = 0; jj < JW; ++jj) {
                            dst[c * JW + jj] = s[jj * nb + c];
                        }
                    }
                } else {
                    for (size_t x = 0; x < rows; ++x) {
                        for (size_t c = 0; c < nb; ++c) {
                            dst[c * JW + x] = s[x * nb + c];
                        }
                    }
                }
                q += rows;
            }
        }
    }

    std::vector<double>& A_;
    const Layout& L_;
    const size_t n_;
    std::vector<double> blockBuf_, blockRecv_, LkkT_;
    std::vector<double> Lrows_[2];  // broadcast panel rows (double buffered)
    std::vector<double> colSend_, colRecv_, packed_;
    std::vector<int> blkCounts_, blkDispls_, counts_, displs_;
    MPI_Request bcastReq_ = MPI_REQUEST_NULL, gatherReq_ = MPI_REQUEST_NULL;
    Panel cur_;  // panel whose communication is in flight
    Panel upd_;  // panel being applied
};

// Distributed Cholesky on the local block. Returns true on success (on all ranks).
bool choleskyDecomposition(std::vector<double>& Aloc, const Layout& L) {
    const size_t n = L.n;
    const size_t nblocks = (n + NB - 1) / NB;
    PanelFactorizer pf(Aloc, L);

    bool ok = true;
    if (nblocks > 0 && !pf.startPanel(0)) ok = false;

    for (size_t k = 0; ok && k < nblocks; ++k) {
        if (!pf.finishPanel()) {
            ok = false;
            break;
        }
        const size_t j1 = std::min(n, (k + 1) * NB);  // first column after panel k
        if (j1 >= n) break;
        const size_t j2 = std::min(n, j1 + NB);  // end of block column k+1
        const size_t lc1 = L.colsBefore(j1), lc2 = L.colsBefore(j2);

        // lookahead: update block column k+1 first, then factor it and start
        // its communication, then apply the rest of panel k's update
        pf.update(j1, lc1, lc2);
        if (!pf.startPanel(k + 1)) {
            ok = false;
            break;
        }
        pf.update(j2, lc2, L.ncloc);
    }
    if (!ok) return false;

    // Zero out upper triangular part of the local block
    for (size_t li = 0; li < L.nloc; ++li) {
        const size_t g = L.globalRow(li);
        double* row = Aloc.data() + li * L.ncloc;
        for (size_t lj = L.colsBefore(g + 1); lj < L.ncloc; ++lj) {
            row[lj] = 0.0;
        }
    }
    return true;
}

// Generate the local block of a symmetric positive definite matrix.
// Identical values to the sequential generator: A = B * B^T + n * I with B from
// rand_r(seed = 42).
void generatePositiveDefiniteMatrix(std::vector<double>& Aloc, const Layout& L) {
    const size_t n = L.n;
    std::vector<double> B(n * n);
    unsigned int seed = 42;

    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

    for (size_t li = 0; li < L.nloc; ++li) {
        const size_t i = L.globalRow(li);
        const double* bi = B.data() + i * n;
        double* arow = Aloc.data() + li * L.ncloc;
        for (size_t lj = 0; lj < L.ncloc; ++lj) {
            const size_t j = L.globalCol(lj);
            const double* bj = B.data() + j * n;
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += bi[k] * bj[k];
            }
            arow[lj] = sum;
            if (i == j) arow[lj] += n;
        }
    }
}

// Gather the distributed blocks into a full matrix in global order.
// If toAll is true every rank receives the result, otherwise only rank 0.
void gatherMatrix(const std::vector<double>& Aloc, const Layout& L, std::vector<double>& A,
                  bool toAll) {
    const size_t n = L.n;
    std::vector<int> counts(L.P), displs(L.P);
    int disp = 0;
    for (int r = 0; r < L.P; ++r) {
        counts[r] =
            (int)(L.localRows(Layout::rowOf(r, L.Pc)) * L.localCols(Layout::colOf(r, L.Pc)));
        displs[r] = disp;
        disp += counts[r];
    }
    std::vector<double> tmp;
    if (toAll || L.rank == 0) {
        tmp.resize(n * n);
    }
    if (toAll) {
        MPI_Allgatherv(Aloc.data(), (int)Aloc.size(), MPI_DOUBLE, tmp.data(), counts.data(),
                       displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    } else {
        MPI_Gatherv(Aloc.data(), (int)Aloc.size(), MPI_DOUBLE, tmp.data(), counts.data(),
                    displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    if (toAll || L.rank == 0) {
        A.resize(n * n);
        for (int r = 0; r < L.P; ++r) {
            const int pr = Layout::rowOf(r, L.Pc), pc = Layout::colOf(r, L.Pc);
            const size_t nr = L.localRows(pr), nc = L.localCols(pc);
            const double* src = tmp.data() + displs[r];
            for (size_t li = 0; li < nr; ++li) {
                const size_t gi = L.globalRow(li, pr);
                for (size_t lj = 0; lj < nc; lj += NB) {
                    const size_t gj = L.globalCol(lj, pc);
                    const size_t w = std::min(NB, nc - lj);
                    memcpy(A.data() + gi * n + gj, src + li * nc + lj, w * sizeof(double));
                }
            }
        }
    }
}

// Validate by computing L * L^T (distributed) and comparing with the original.
bool validateCholesky(const std::vector<double>& Lfull, const std::vector<double>& AorigLoc,
                      const Layout& L) {
    const size_t n = L.n;
    double maxError = 0.0;
    double relError = 0.0;

    for (size_t li = 0; li < L.nloc; ++li) {
        const size_t i = L.globalRow(li);
        const double* Li = Lfull.data() + i * n;
        const double* orig = AorigLoc.data() + li * L.ncloc;
        for (size_t lj = 0; lj < L.ncloc; ++lj) {
            const size_t j = L.globalCol(lj);
            const double* Lj = Lfull.data() + j * n;
            // entries beyond min(i, j) are exactly zero in L and contribute nothing
            const size_t kend = std::min(i, j) + 1;
            double sum = 0.0;
            for (size_t k = 0; k < kend; ++k) {
                sum += Li[k] * Lj[k];
            }
            const double error = fabs(sum - orig[lj]);
            maxError = std::max(maxError, error);
            const double rel = error / (fabs(orig[lj]) + 1e-10);
            relError = std::max(relError, rel);
        }
    }

    double gmax[2] = {0.0, 0.0};
    double lmax[2] = {maxError, relError};
    MPI_Allreduce(lmax, gmax, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    maxError = gmax[0];
    relError = gmax[1];

    if (L.rank == 0) {
        printf("Max absolute error: %.10e\n", maxError);
        printf("Max relative error: %.10e\n", relError);
    }

    if (relError > 1e-6) {
        if (L.rank == 0) printf("Validation failed: relative error too large\n");
        return false;
    }
    return true;
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
    int rank = 0, nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

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
            if (rank == 0) printUsage(argv[0]);
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

    Layout L;
    L.init(n, rank, nprocs);

    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI processes: %d (%d x %d grid)\n", nprocs, L.Pr, L.Pc);
    }

    // Allocate local block
    std::vector<double> Aloc(L.nloc * L.ncloc);
    std::vector<double> AorigLoc;

    // Generate positive definite matrix (distributed)
    if (rank == 0) printf("Generating positive definite matrix...\n");
    generatePositiveDefiniteMatrix(Aloc, L);

    if (validate) {
        AorigLoc = Aloc;  // Save original block for validation
    }

    // Perform Cholesky decomposition
    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    bool success = choleskyDecomposition(Aloc, L);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (!success) {
        if (rank == 0) printf("Cholesky decomposition failed\n");
        L.finalize();
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("Computation time: %ld ms\n", (long)duration.count());

        // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Assemble the full L where needed
    std::vector<double> A;
    if (validate || printResults) {
        gatherMatrix(Aloc, L, A, validate);
    }

    // Print results for external validation
    if (printResults && rank == 0) {
        print_results(A, "CholeskyL");
    }

    // Validation
    int ret = 0;
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        bool valid = validateCholesky(A, AorigLoc, L);

        if (rank == 0) {
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
        ret = valid ? 0 : 1;
    }

    L.finalize();
    MPI_Finalize();
    return ret;
}
