#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// Distributed (MPI) Cholesky decomposition.
// Decomposes positive definite matrix A into L * L^T where L is lower triangular.
//
// Parallelization: right-looking blocked Cholesky over a two dimensional
// block-cyclic distribution of the matrix (ScaLAPACK style).  The ranks form a
// Pr x Pc process grid, block (I,J) of the matrix lives on process
// (I % Pr, J % Pc).  For every block column k
//   * the owner of the diagonal block factorizes it and broadcasts it down its
//     process column,
//   * the ranks of that process column solve their rows against it (TRSM),
//   * the resulting panel is broadcast along the process rows (giving every
//     rank the panel rows of the matrix rows it owns) and gathered along the
//     process columns (giving every rank the panel rows of the matrix columns
//     it owns),
//   * every rank updates its part of the trailing sub-matrix with a cache and
//     register blocked rank-nb update.
// This keeps the O(n^3) work perfectly distributed and the communication volume
// per rank at O(n^2 / sqrt(P)) instead of O(n^2) for a one dimensional layout.
//
// Matrix generation, the upper triangle zeroing and the validation use the same
// distribution, so no rank ever stores the whole matrix (only printing the
// result for external validation collects it on rank 0).

static int g_rank = 0;
static int g_nranks = 1;

// ---------------------------------------------------------------------------
// Data distribution
// ---------------------------------------------------------------------------

struct Dist {
    size_t n = 0;   // matrix dimension
    size_t nb = 1;  // block size
    size_t nt = 0;  // number of blocks per dimension
    int Pr = 1;     // process grid rows
    int Pc = 1;     // process grid columns
    int pi = 0;     // this rank's process grid row
    int pj = 0;     // this rank's process grid column
    MPI_Comm rowComm = MPI_COMM_NULL;  // ranks with the same pi
    MPI_Comm colComm = MPI_COMM_NULL;  // ranks with the same pj
    size_t rblocks = 0;  // number of block rows owned
    size_t cblocks = 0;  // number of block columns owned
    size_t mloc = 0;     // local rows (padded to full blocks)
    size_t nloc = 0;     // local columns (padded to full blocks)
    size_t ld = 0;       // leading dimension of the local matrix
};

// Leading dimension: an odd multiple of 8 avoids cache set aliasing between the
// rows of a register tile.
static inline size_t paddedLd(size_t nloc) { return (nloc == 0) ? 0 : (((nloc + 15) / 16) * 16 + 8); }

// Number of valid rows/columns of block I
static inline size_t blockLen(const Dist& d, size_t I) {
    const size_t s = I * d.nb;
    return (s < d.n) ? std::min(d.nb, d.n - s) : 0;
}

static inline size_t blocksOwnedLE(size_t k, int p, int P) {
    return (k >= (size_t)p) ? ((k - (size_t)p) / (size_t)P + 1) : 0;
}

static inline size_t ownedRowBlocksLE(const Dist& d, size_t k) { return blocksOwnedLE(k, d.pi, d.Pr); }
static inline size_t ownedColBlocksLE(const Dist& d, size_t k) { return blocksOwnedLE(k, d.pj, d.Pc); }

// Global row of the r-th row of the lI-th locally owned block row
static inline size_t globalRow(const Dist& d, size_t lI, size_t r) {
    return ((size_t)d.pi + lI * (size_t)d.Pr) * d.nb + r;
}

// Exclusive local column index bound covering all global columns <= g
static inline size_t localColBound(const Dist& d, size_t g) {
    const size_t J = g / d.nb;
    const size_t before = (J == 0) ? 0 : ownedColBlocksLE(d, J - 1);
    if ((J % (size_t)d.Pc) == (size_t)d.pj) {
        return before * d.nb + (g % d.nb) + 1;
    }
    return before * d.nb;
}

static size_t chooseBlockSize(size_t n) {
    if (n == 0) {
        return 1;
    }
    size_t nb = 48;
    if (nb > n) {
        nb = n;
    }
    return nb;
}

static Dist makeDistribution(size_t n) {
    Dist d;
    d.n = n;
    d.nb = chooseBlockSize(n);
    d.nt = (n + d.nb - 1) / d.nb;

    // as square a process grid as possible
    int pr = 1;
    for (int c = 1; c * c <= g_nranks; ++c) {
        if (g_nranks % c == 0) {
            pr = c;
        }
    }
    d.Pr = pr;
    d.Pc = g_nranks / pr;
    d.pi = g_rank / d.Pc;
    d.pj = g_rank % d.Pc;

    MPI_Comm_split(MPI_COMM_WORLD, d.pi, d.pj, &d.rowComm);
    MPI_Comm_split(MPI_COMM_WORLD, d.pj, d.pi, &d.colComm);

    d.rblocks = (d.nt > 0) ? ownedRowBlocksLE(d, d.nt - 1) : 0;
    d.cblocks = (d.nt > 0) ? ownedColBlocksLE(d, d.nt - 1) : 0;
    d.mloc = d.rblocks * d.nb;
    d.nloc = d.cblocks * d.nb;
    d.ld = paddedLd(d.nloc);
    return d;
}

// ---------------------------------------------------------------------------
// Compute kernels
// ---------------------------------------------------------------------------

// SIMD vector of 4 doubles (compiler vector extension, no external dependency).
// Declared unaligned so that plain pointer casts are safe.
typedef double vec4 __attribute__((vector_size(32), aligned(8)));

static inline vec4 vload(const double* p) { return *reinterpret_cast<const vec4*>(p); }
static inline void vstore(double* p, vec4 v) { *reinterpret_cast<vec4*>(p) = v; }
static inline vec4 vsplat(double x) { return vec4{x, x, x, x}; }

// Dot product with independent accumulator chains so that it vectorizes.
static inline double dotProduct(const double* __restrict a, const double* __restrict b, size_t len) {
    vec4 v0 = vsplat(0.0), v1 = vsplat(0.0);
    size_t k = 0;
    for (; k + 8 <= len; k += 8) {
        v0 += vload(a + k) * vload(b + k);
        v1 += vload(a + k + 4) * vload(b + k + 4);
    }
    v0 += v1;
    double s = (v0[0] + v0[1]) + (v0[2] + v0[3]);
    for (; k < len; ++k) {
        s += a[k] * b[k];
    }
    return s;
}

// Rows per register tile of the trailing update kernel (6 x 8 doubles of
// accumulators fit into the AVX register file).
static constexpr size_t kMR = 6;

// Trailing update micro kernel: C[ii][j] -= sum_c A[ii][c] * PT[c][j]
// for ii < MR and j < ncols.  The MR x 8 accumulator tile stays in registers
// over the whole c loop, so the panel is streamed only once per row tile.
template <size_t MR>
static void updateTile(const double* __restrict A, size_t lda, double* __restrict C, size_t ldc, size_t cur,
                       const double* __restrict PT, size_t ldpt, size_t ncols) {
    size_t j = 0;
    for (; j + 8 <= ncols; j += 8) {
        vec4 acc0[MR];
        vec4 acc1[MR];
        for (size_t ii = 0; ii < MR; ++ii) {
            acc0[ii] = vsplat(0.0);
            acc1[ii] = vsplat(0.0);
        }
        const double* __restrict p = PT + j;
        for (size_t c = 0; c < cur; ++c, p += ldpt) {
            const vec4 p0 = vload(p);
            const vec4 p1 = vload(p + 4);
            for (size_t ii = 0; ii < MR; ++ii) {
                const vec4 av = vsplat(A[ii * lda + c]);
                acc0[ii] += av * p0;
                acc1[ii] += av * p1;
            }
        }
        for (size_t ii = 0; ii < MR; ++ii) {
            double* __restrict c2 = C + ii * ldc + j;
            vstore(c2, vload(c2) - acc0[ii]);
            vstore(c2 + 4, vload(c2 + 4) - acc1[ii]);
        }
    }
    for (; j + 4 <= ncols; j += 4) {
        vec4 acc0[MR];
        for (size_t ii = 0; ii < MR; ++ii) {
            acc0[ii] = vsplat(0.0);
        }
        const double* __restrict p = PT + j;
        for (size_t c = 0; c < cur; ++c, p += ldpt) {
            const vec4 p0 = vload(p);
            for (size_t ii = 0; ii < MR; ++ii) {
                acc0[ii] += vsplat(A[ii * lda + c]) * p0;
            }
        }
        for (size_t ii = 0; ii < MR; ++ii) {
            double* __restrict c2 = C + ii * ldc + j;
            vstore(c2, vload(c2) - acc0[ii]);
        }
    }
    for (; j < ncols; ++j) {
        for (size_t ii = 0; ii < MR; ++ii) {
            double s = 0.0;
            for (size_t c = 0; c < cur; ++c) {
                s += A[ii * lda + c] * PT[c * ldpt + j];
            }
            C[ii * ldc + j] -= s;
        }
    }
}

static inline void updateTileDyn(const double* A, size_t lda, double* C, size_t ldc, size_t m, size_t cur,
                                 const double* PT, size_t ldpt, size_t ncols) {
    switch (m) {
        case 6: updateTile<6>(A, lda, C, ldc, cur, PT, ldpt, ncols); break;
        case 5: updateTile<5>(A, lda, C, ldc, cur, PT, ldpt, ncols); break;
        case 4: updateTile<4>(A, lda, C, ldc, cur, PT, ldpt, ncols); break;
        case 3: updateTile<3>(A, lda, C, ldc, cur, PT, ldpt, ncols); break;
        case 2: updateTile<2>(A, lda, C, ldc, cur, PT, ldpt, ncols); break;
        default: updateTile<1>(A, lda, C, ldc, cur, PT, ldpt, ncols); break;
    }
}

// ---------------------------------------------------------------------------
// Distributed Cholesky decomposition
// ---------------------------------------------------------------------------

// Per block column geometry
struct PanelInfo {
    size_t kb = 0;       // global index of the first row/column of the block
    size_t cur = 0;      // valid rows/columns of the block
    size_t lr_lo = 0;    // first local row below the block
    size_t lc_lo = 0;    // first local column right of the block
    size_t mbelow = 0;   // local rows below the block (padded)
    size_t ncbelow = 0;  // local columns right of the block (padded)
    size_t ldpt = 0;     // leading dimension of the transposed panel
    int pik = 0;         // process grid row owning the block row
    int pjk = 0;         // process grid column owning the block column
};

static PanelInfo panelInfo(const Dist& d, size_t k) {
    PanelInfo p;
    p.kb = k * d.nb;
    p.cur = blockLen(d, k);
    p.lr_lo = ownedRowBlocksLE(d, k) * d.nb;
    p.lc_lo = ownedColBlocksLE(d, k) * d.nb;
    p.mbelow = d.mloc - p.lr_lo;
    p.ncbelow = d.nloc - p.lc_lo;
    // odd multiple of 8: avoids cache set aliasing between the panel rows
    p.ldpt = ((p.ncbelow + 15) / 16) * 16 + 8;
    p.pik = (int)(k % (size_t)d.Pr);
    p.pjk = (int)(k % (size_t)d.Pc);
    return p;
}

// The factorization is software pipelined: while the panel of block column k+1
// is in flight, the trailing update of block column k is executed.  Both
// collectives (the panel broadcast along the process rows and the transposed
// panel gather along the process columns) are non-blocking and are covered by
// one half of the trailing update each.
static bool choleskyDecomposition(const Dist& d, std::vector<double>& A) {
    const size_t nb = d.nb;
    const size_t nt = d.nt;
    const size_t ld = d.ld;
    if (nt == 0) {
        return true;
    }

    std::vector<double> Lkk(nb * nb + 2);
    std::vector<double> LkkT(nb * nb);
    std::vector<double> panel[2];
    std::vector<double> sendbuf[2];
    std::vector<double> recvbuf[2];
    std::vector<double> PT;
    std::vector<int> counts(d.Pr);
    std::vector<int> displs(d.Pr);
    MPI_Request bcastReq = MPI_REQUEST_NULL;
    MPI_Request gathReq = MPI_REQUEST_NULL;
    int fail[2] = {0, 0};

    // Factorize the diagonal block of block column k, solve the rows below it
    // and post the panel broadcast along the process rows.
    auto startPanel = [&](size_t k) {
        const PanelInfo p = panelInfo(d, k);
        std::vector<double>& pan = panel[k % 2];
        pan.resize(p.mbelow * p.cur + 2);
        int lfail[2] = {0, 0};

        if (d.pj == p.pjk) {
            const size_t lck = (k / (size_t)d.Pc) * nb;

            if (d.pi == p.pik) {
                const size_t lrk = (k / (size_t)d.Pr) * nb;
                double* base = &A[lrk * ld + lck];
                for (size_t c = 0; c < p.cur; ++c) {
                    memcpy(&Lkk[c * p.cur], base + c * ld, p.cur * sizeof(double));
                }
                for (size_t i = 0; i < p.cur; ++i) {
                    double* ri = &Lkk[i * p.cur];
                    for (size_t j = 0; j < i; ++j) {
                        const double s = dotProduct(ri, &Lkk[j * p.cur], j);
                        ri[j] = (ri[j] - s) / Lkk[j * p.cur + j];
                    }
                    const double s = dotProduct(ri, ri, i);
                    const double val = ri[i] - s;
                    if (val <= 0.0) {
                        lfail[0] = 1;
                        lfail[1] = (int)(p.kb + i);
                        break;
                    }
                    ri[i] = sqrt(val);
                }
                if (!lfail[0]) {
                    for (size_t c = 0; c < p.cur; ++c) {
                        memcpy(base + c * ld, &Lkk[c * p.cur], (c + 1) * sizeof(double));
                    }
                }
                Lkk[p.cur * p.cur] = lfail[0];
                Lkk[p.cur * p.cur + 1] = lfail[1];
            }

            MPI_Bcast(Lkk.data(), (int)(p.cur * p.cur + 2), MPI_DOUBLE, p.pik, d.colComm);
            lfail[0] = (int)Lkk[p.cur * p.cur];
            lfail[1] = (int)Lkk[p.cur * p.cur + 1];

            if (!lfail[0] && p.mbelow > 0) {
                for (size_t c = 0; c < p.cur; ++c) {
                    for (size_t r = 0; r < p.cur; ++r) {
                        LkkT[c * p.cur + r] = Lkk[r * p.cur + c];
                    }
                }
                for (size_t lI = ownedRowBlocksLE(d, k); lI < d.rblocks; ++lI) {
                    const size_t rows = blockLen(d, (size_t)d.pi + lI * (size_t)d.Pr);
                    for (size_t r = 0; r < rows; ++r) {
                        double* __restrict x = &A[(lI * nb + r) * ld + lck];
                        for (size_t c = 0; c < p.cur; ++c) {
                            const double t = x[c] / Lkk[c * p.cur + c];
                            x[c] = t;
                            const double* __restrict col = &LkkT[c * p.cur];
                            for (size_t dd = c + 1; dd < p.cur; ++dd) {
                                x[dd] -= t * col[dd];
                            }
                        }
                    }
                }
            }

            for (size_t lr = p.lr_lo; lr < d.mloc; ++lr) {
                memcpy(&pan[(lr - p.lr_lo) * p.cur], &A[lr * ld + lck], p.cur * sizeof(double));
            }
            pan[p.mbelow * p.cur] = lfail[0];
            pan[p.mbelow * p.cur + 1] = lfail[1];
        }

        MPI_Ibcast(pan.data(), (int)(p.mbelow * p.cur + 2), MPI_DOUBLE, p.pjk, d.rowComm, &bcastReq);
    };

    // Wait for the panel broadcast and post the gather of the panel rows that
    // belong to the locally owned columns (block J lives on process row J % Pr).
    auto startGather = [&](size_t k) {
        MPI_Wait(&bcastReq, MPI_STATUS_IGNORE);
        const PanelInfo p = panelInfo(d, k);
        const std::vector<double>& pan = panel[k % 2];
        fail[0] = (int)pan[p.mbelow * p.cur];
        fail[1] = (int)pan[p.mbelow * p.cur + 1];
        if (fail[0]) {
            return;
        }
        for (int r = 0; r < d.Pr; ++r) {
            counts[r] = 0;
        }
        for (size_t J = k + 1; J < nt; ++J) {
            if ((J % (size_t)d.Pc) == (size_t)d.pj) {
                counts[J % (size_t)d.Pr] += (int)(nb * p.cur);
            }
        }
        displs[0] = 0;
        for (int r = 1; r < d.Pr; ++r) {
            displs[r] = displs[r - 1] + counts[r - 1];
        }
        const size_t total = (size_t)(displs[d.Pr - 1] + counts[d.Pr - 1]);
        sendbuf[k % 2].resize((size_t)counts[d.pi] + 1);
        recvbuf[k % 2].resize(total + 1);
        size_t q = 0;
        for (size_t J = k + 1; J < nt; ++J) {
            if ((J % (size_t)d.Pc) != (size_t)d.pj || (J % (size_t)d.Pr) != (size_t)d.pi) {
                continue;
            }
            const size_t lr = (J / (size_t)d.Pr) * nb;
            memcpy(&sendbuf[k % 2][q], &pan[(lr - p.lr_lo) * p.cur], nb * p.cur * sizeof(double));
            q += nb * p.cur;
        }
        MPI_Iallgatherv(sendbuf[k % 2].data(), counts[d.pi], MPI_DOUBLE, recvbuf[k % 2].data(), counts.data(),
                        displs.data(), MPI_DOUBLE, d.colComm, &gathReq);
    };

    // Wait for the gather and transpose it into PT[c][lc - lc_lo]
    auto finishGather = [&](size_t k) {
        MPI_Wait(&gathReq, MPI_STATUS_IGNORE);
        const PanelInfo p = panelInfo(d, k);
        const std::vector<double>& rbuf = recvbuf[k % 2];
        PT.resize(p.cur * p.ldpt);
        for (int r = 0; r < d.Pr; ++r) {
            size_t q = (size_t)displs[r];
            for (size_t J = k + 1; J < nt; ++J) {
                if ((J % (size_t)d.Pc) != (size_t)d.pj || (J % (size_t)d.Pr) != (size_t)r) {
                    continue;
                }
                const size_t lc0 = (J / (size_t)d.Pc) * nb - p.lc_lo;
                const size_t cols = blockLen(d, J);
                for (size_t rr = 0; rr < cols; ++rr) {
                    const double* src = &rbuf[q + rr * p.cur];
                    for (size_t c = 0; c < p.cur; ++c) {
                        PT[c * p.ldpt + lc0 + rr] = src[c];
                    }
                }
                q += nb * p.cur;
            }
        }
    };

    // Trailing update of the local rows below block k for the local columns
    // [jfrom, jto) (relative to the first local column right of block k).
    auto doUpdate = [&](size_t k, size_t jfrom, size_t jto) {
        if (jfrom >= jto) {
            return;
        }
        const PanelInfo p = panelInfo(d, k);
        const double* pan = panel[k % 2].data();
        const size_t ldpt = p.ldpt;
        const size_t firstBlock = ownedRowBlocksLE(d, k);
        size_t jc_size = 65536 / p.cur;
        jc_size = std::max<size_t>(64, (jc_size / 8) * 8);

        for (size_t jc = jfrom; jc < jto; jc += jc_size) {
            const size_t jc_end = std::min(jto, jc + jc_size);
            for (size_t lI = firstBlock; lI < d.rblocks; ++lI) {
                const size_t rows = blockLen(d, (size_t)d.pi + lI * (size_t)d.Pr);
                for (size_t r0 = 0; r0 < rows; r0 += kMR) {
                    const size_t m = std::min<size_t>(kMR, rows - r0);
                    const size_t bound = localColBound(d, globalRow(d, lI, r0));
                    if (bound <= p.lc_lo + jc) {
                        continue;
                    }
                    const size_t jend = std::min(bound - p.lc_lo, jc_end);
                    if (jc >= jend) {
                        continue;
                    }
                    const size_t lr = lI * nb + r0;
                    updateTileDyn(pan + (lr - p.lr_lo) * p.cur, p.cur, &A[lr * ld + p.lc_lo + jc], ld, m, p.cur,
                                  PT.data() + jc, ldpt, jend - jc);
                }
            }
        }
        // triangular remainder of the row tiles (a few columns per row)
        for (size_t lI = firstBlock; lI < d.rblocks; ++lI) {
            const size_t rows = blockLen(d, (size_t)d.pi + lI * (size_t)d.Pr);
            for (size_t r0 = 0; r0 < rows; r0 += kMR) {
                const size_t m = std::min<size_t>(kMR, rows - r0);
                const size_t g0 = globalRow(d, lI, r0);
                const size_t b0 = std::max(localColBound(d, g0), p.lc_lo + jfrom);
                for (size_t ii = 1; ii < m; ++ii) {
                    const size_t bi = std::min(localColBound(d, g0 + ii), p.lc_lo + jto);
                    if (bi <= b0) {
                        continue;
                    }
                    const size_t lr = lI * nb + r0 + ii;
                    const double* __restrict a = pan + (lr - p.lr_lo) * p.cur;
                    double* __restrict c2 = &A[lr * ld];
                    for (size_t lc = b0; lc < bi; ++lc) {
                        double s = 0.0;
                        for (size_t c = 0; c < p.cur; ++c) {
                            s += a[c] * PT[c * ldpt + (lc - p.lc_lo)];
                        }
                        c2[lc] -= s;
                    }
                }
            }
        }
    };

    startPanel(0);
    startGather(0);

    for (size_t k = 0; k < nt && !fail[0]; ++k) {
        const PanelInfo p = panelInfo(d, k);
        finishGather(k);

        // columns of the next block column are updated first so that the next
        // panel can be produced while the rest of the update is running
        const size_t prio =
            (p.ncbelow > 0 && k + 1 < nt && (int)((k + 1) % (size_t)d.Pc) == d.pj) ? std::min(nb, p.ncbelow) : 0;
        doUpdate(k, 0, prio);

        if (k + 1 < nt) {
            startPanel(k + 1);
        }

        const size_t half = prio + (p.ncbelow - prio) / 2;
        doUpdate(k, prio, half);

        if (k + 1 < nt) {
            startGather(k + 1);
        }

        doUpdate(k, half, p.ncbelow);
    }

    if (fail[0]) {
        if (g_rank == 0) {
            printf("Error: Matrix is not positive definite at diagonal element %d\n", fail[1]);
        }
        return false;
    }

    // Zero out the upper triangular part
    for (size_t lI = 0; lI < d.rblocks; ++lI) {
        const size_t rows = blockLen(d, (size_t)d.pi + lI * (size_t)d.Pr);
        for (size_t r = 0; r < rows; ++r) {
            const size_t g = globalRow(d, lI, r);
            const size_t from = localColBound(d, g);
            if (from < d.nloc) {
                memset(&A[(lI * nb + r) * ld + from], 0, (d.nloc - from) * sizeof(double));
            }
        }
    }

    return true;
}

// ---------------------------------------------------------------------------
// Matrix generation (distributed)
// ---------------------------------------------------------------------------

// Generate a symmetric positive definite matrix
// Method: A = B * B^T with B random, plus diagonal dominance.  Every rank runs
// the same random sequence (which is cheap) but only keeps the rows of B that
// correspond to its own matrix rows and columns, so no rank stores all of B and
// no communication is needed.  The product itself is a distributed matrix
// multiplication using the same register blocked kernel as the factorization.
static void generatePositiveDefiniteMatrix(const Dist& d, std::vector<double>& A, bool needFullRows) {
    const size_t n = d.n;
    if (n == 0) {
        return;
    }
    const size_t nb = d.nb;
    const size_t ld = d.ld;

    size_t cb = ((1u << 20) / n / nb) * nb;
    cb = std::max(cb, nb);

    std::vector<double> chunk(cb * n);
    std::vector<double> Brow(d.mloc * n);   // B rows of the locally owned matrix rows
    std::vector<double> BcolT(n * ld);      // transposed B rows of the owned columns

    unsigned int seed = 42;
    for (size_t s = 0; s < n; s += cb) {
        const size_t cnt = std::min(cb, n - s);
        const size_t elems = cnt * n;
        for (size_t i = 0; i < elems; ++i) {
            chunk[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
        }
        for (size_t g = s; g < s + cnt; ++g) {
            const size_t I = g / nb;
            if ((I % (size_t)d.Pr) != (size_t)d.pi) {
                continue;
            }
            const size_t lr = (I / (size_t)d.Pr) * nb + (g % nb);
            memcpy(&Brow[lr * n], &chunk[(g - s) * n], n * sizeof(double));
        }
        for (size_t J = s / nb; J < d.nt && J * nb < s + cnt; ++J) {
            if ((J % (size_t)d.Pc) != (size_t)d.pj) {
                continue;
            }
            const size_t cols = blockLen(d, J);
            const size_t lc0 = (J / (size_t)d.Pc) * nb;
            for (size_t k0 = 0; k0 < n; k0 += 64) {
                const size_t kn = std::min<size_t>(64, n - k0);
                for (size_t c = 0; c < cols; ++c) {
                    const double* src = &chunk[(J * nb - s + c) * n + k0];
                    for (size_t k = 0; k < kn; ++k) {
                        BcolT[(k0 + k) * ld + lc0 + c] = src[k];
                    }
                }
            }
        }
    }

    // A = B * B^T; the kernel subtracts, so the negated product is accumulated
    // and flipped afterwards.  Row tiles compute up to the last row's diagonal,
    // the few resulting entries above the diagonal are never read.
    constexpr size_t KC = 512;
    constexpr size_t JC = 256;
    for (size_t kc = 0; kc < n; kc += KC) {
        const size_t kn = std::min(KC, n - kc);
        for (size_t jc = 0; jc < d.nloc; jc += JC) {
            const size_t jn = std::min(JC, d.nloc - jc);
            for (size_t lI = 0; lI < d.rblocks; ++lI) {
                const size_t rows = blockLen(d, (size_t)d.pi + lI * (size_t)d.Pr);
                for (size_t r0 = 0; r0 < rows; r0 += kMR) {
                    const size_t m = std::min<size_t>(kMR, rows - r0);
                    const size_t bound =
                        needFullRows ? d.nloc : localColBound(d, globalRow(d, lI, r0 + m - 1));
                    if (bound <= jc) {
                        continue;
                    }
                    const size_t jend = std::min(bound, jc + jn);
                    updateTileDyn(&Brow[(lI * nb + r0) * n + kc], n, &A[(lI * nb + r0) * ld + jc], ld, m, kn,
                                  &BcolT[kc * ld + jc], ld, jend - jc);
                }
            }
        }
    }
    for (size_t i = 0; i < A.size(); ++i) {
        A[i] = -A[i];
    }

    // Add diagonal dominance to ensure positive definiteness
    for (size_t lI = 0; lI < d.rblocks; ++lI) {
        const size_t rows = blockLen(d, (size_t)d.pi + lI * (size_t)d.Pr);
        for (size_t r = 0; r < rows; ++r) {
            const size_t g = globalRow(d, lI, r);
            if (((g / nb) % (size_t)d.Pc) != (size_t)d.pj) {
                continue;
            }
            const size_t lc = ((g / nb) / (size_t)d.Pc) * nb + (g % nb);
            A[(lI * nb + r) * ld + lc] += (double)n;
        }
    }
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

// Collect the full matrix rows that this rank needs: all n columns of its own
// rows (gathered along the process row) and all n columns of the rows that
// correspond to its own columns (gathered along the process column).
static void gatherRowsAndCols(const Dist& d, const std::vector<double>& A, std::vector<double>& rowsFull,
                              std::vector<double>& colsFull) {
    const size_t n = d.n;
    const size_t nb = d.nb;

    rowsFull.assign(d.mloc * n, 0.0);
    {
        std::vector<int> counts(d.Pc), displs(d.Pc);
        for (int q = 0; q < d.Pc; ++q) {
            const size_t cblk = (d.nt > 0) ? blocksOwnedLE(d.nt - 1, q, d.Pc) : 0;
            counts[q] = (int)(d.mloc * paddedLd(cblk * nb));
        }
        displs[0] = 0;
        for (int q = 1; q < d.Pc; ++q) {
            displs[q] = displs[q - 1] + counts[q - 1];
        }
        std::vector<double> buf((size_t)(displs[d.Pc - 1] + counts[d.Pc - 1]) + 1);
        MPI_Allgatherv(A.data(), counts[d.pj], MPI_DOUBLE, buf.data(), counts.data(), displs.data(), MPI_DOUBLE,
                       d.rowComm);
        for (int q = 0; q < d.Pc; ++q) {
            const size_t qnloc = (size_t)counts[q] / std::max<size_t>(1, d.mloc);
            const double* src = &buf[displs[q]];
            for (size_t J = (size_t)q; J < d.nt; J += (size_t)d.Pc) {
                const size_t lc = (J / (size_t)d.Pc) * nb;
                const size_t cols = blockLen(d, J);
                for (size_t lr = 0; lr < d.mloc; ++lr) {
                    memcpy(&rowsFull[lr * n + J * nb], src + lr * qnloc + lc, cols * sizeof(double));
                }
            }
        }
    }

    colsFull.assign(d.nloc * n, 0.0);
    {
        std::vector<int> counts(d.Pr), displs(d.Pr);
        for (int r = 0; r < d.Pr; ++r) {
            counts[r] = 0;
        }
        for (size_t J = 0; J < d.nt; ++J) {
            if ((J % (size_t)d.Pc) == (size_t)d.pj) {
                counts[J % (size_t)d.Pr] += (int)(nb * n);
            }
        }
        displs[0] = 0;
        for (int r = 1; r < d.Pr; ++r) {
            displs[r] = displs[r - 1] + counts[r - 1];
        }
        std::vector<double> sendbuf((size_t)counts[d.pi] + 1);
        std::vector<double> buf((size_t)(displs[d.Pr - 1] + counts[d.Pr - 1]) + 1);
        size_t p = 0;
        for (size_t J = 0; J < d.nt; ++J) {
            if ((J % (size_t)d.Pc) != (size_t)d.pj || (J % (size_t)d.Pr) != (size_t)d.pi) {
                continue;
            }
            memcpy(&sendbuf[p], &rowsFull[(J / (size_t)d.Pr) * nb * n], nb * n * sizeof(double));
            p += nb * n;
        }
        MPI_Allgatherv(sendbuf.data(), counts[d.pi], MPI_DOUBLE, buf.data(), counts.data(), displs.data(),
                       MPI_DOUBLE, d.colComm);
        for (int r = 0; r < d.Pr; ++r) {
            size_t q = (size_t)displs[r];
            for (size_t J = 0; J < d.nt; ++J) {
                if ((J % (size_t)d.Pc) != (size_t)d.pj || (J % (size_t)d.Pr) != (size_t)r) {
                    continue;
                }
                memcpy(&colsFull[(J / (size_t)d.Pc) * nb * n], &buf[q], nb * n * sizeof(double));
                q += nb * n;
            }
        }
    }
}

static bool validateCholesky(const Dist& d, const std::vector<double>& L, const std::vector<double>& A_orig) {
    // Validate by computing L * L^T and comparing with original matrix
    const size_t n = d.n;
    const size_t nb = d.nb;
    const size_t ld = d.ld;

    std::vector<double> rowsFull, colsFull;
    gatherRowsAndCols(d, L, rowsFull, colsFull);

    double local[2] = {0.0, 0.0};
    for (size_t lI = 0; lI < d.rblocks; ++lI) {
        const size_t rows = blockLen(d, (size_t)d.pi + lI * (size_t)d.Pr);
        for (size_t r = 0; r < rows; ++r) {
            const size_t lr = lI * nb + r;
            const double* li = &rowsFull[lr * n];
            const double* ai = &A_orig[lr * ld];
            for (size_t lJ = 0; lJ < d.cblocks; ++lJ) {
                const size_t cols = blockLen(d, (size_t)d.pj + lJ * (size_t)d.Pc);
                for (size_t c = 0; c < cols; ++c) {
                    const size_t lc = lJ * nb + c;
                    const double sum = dotProduct(li, &colsFull[lc * n], n);
                    const double error = fabs(sum - ai[lc]);
                    local[0] = std::max(local[0], error);
                    local[1] = std::max(local[1], error / (fabs(ai[lc]) + 1e-10));
                }
            }
        }
    }

    double global[2] = {0.0, 0.0};
    MPI_Allreduce(local, global, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    if (g_rank == 0) {
        printf("Max absolute error: %.10e\n", global[0]);
        printf("Max relative error: %.10e\n", global[1]);
    }

    // Check if error is within tolerance
    if (global[1] > 1e-6) {
        if (g_rank == 0) {
            printf("Validation failed: relative error too large\n");
        }
        return false;
    }

    return true;
}

// Collect the distributed matrix into a full row-major matrix on rank 0.
static void gatherFullMatrix(const Dist& d, const std::vector<double>& A, std::vector<double>& full) {
    const size_t n = d.n;
    const size_t nb = d.nb;

    if (g_rank != 0) {
        MPI_Send(A.data(), (int)(d.mloc * d.ld), MPI_DOUBLE, 0, 7, MPI_COMM_WORLD);
        return;
    }

    full.assign(n * n, 0.0);
    std::vector<double> buf;
    for (int r = 0; r < g_nranks; ++r) {
        const int pi = r / d.Pc;
        const int pj = r % d.Pc;
        const size_t mloc = ((d.nt > 0) ? blocksOwnedLE(d.nt - 1, pi, d.Pr) : 0) * nb;
        const size_t nloc = ((d.nt > 0) ? blocksOwnedLE(d.nt - 1, pj, d.Pc) : 0) * nb;
        const size_t ldr = paddedLd(nloc);
        const double* src;
        if (r == 0) {
            src = A.data();
        } else {
            buf.resize(mloc * ldr + 1);
            MPI_Recv(buf.data(), (int)(mloc * ldr), MPI_DOUBLE, r, 7, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            src = buf.data();
        }
        if (mloc == 0 || nloc == 0) {
            continue;
        }
        for (size_t I = (size_t)pi; I < d.nt; I += (size_t)d.Pr) {
            const size_t rows = blockLen(d, I);
            for (size_t rr = 0; rr < rows; ++rr) {
                const size_t lr = (I / (size_t)d.Pr) * nb + rr;
                for (size_t J = (size_t)pj; J < d.nt; J += (size_t)d.Pc) {
                    const size_t cols = blockLen(d, J);
                    const size_t lc = (J / (size_t)d.Pc) * nb;
                    memcpy(&full[(I * nb + rr) * n + J * nb], src + lr * ldr + lc, cols * sizeof(double));
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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_nranks);

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
            if (g_rank == 0) {
                printUsage(argv[0]);
            }
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

    if (g_rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const Dist d = makeDistribution(n);

    // Allocate the local part of the matrix
    std::vector<double> A(d.mloc * d.ld);
    std::vector<double> A_orig;

    // Generate positive definite matrix
    if (g_rank == 0) {
        printf("Generating positive definite matrix...\n");
    }
    generatePositiveDefiniteMatrix(d, A, validate);

    if (validate) {
        A_orig = A; // Save original for validation
    }

    // Perform Cholesky decomposition
    if (g_rank == 0) {
        printf("Computing Cholesky decomposition...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    bool success = choleskyDecomposition(d, A);

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();
    double maxElapsed = elapsed;
    MPI_Allreduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    const long durationMs = (long)(maxElapsed * 1000.0);

    if (!success) {
        if (g_rank == 0) {
            printf("Cholesky decomposition failed\n");
        }
        MPI_Finalize();
        return 1;
    }

    if (g_rank == 0) {
        printf("Computation time: %ld ms\n", durationMs);

        // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (durationMs / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Print results for external validation
    if (printResults) {
        std::vector<double> full;
        gatherFullMatrix(d, A, full);
        if (g_rank == 0) {
            print_results(full, "CholeskyL");
        }
    }

    // Validation
    if (validate) {
        if (g_rank == 0) {
            printf("Validating result...\n");
        }
        const bool valid = validateCholesky(d, A, A_orig);

        if (g_rank == 0) {
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        MPI_Finalize();
        return valid ? 0 : 1;
    }

    MPI_Finalize();
    return 0;
}
