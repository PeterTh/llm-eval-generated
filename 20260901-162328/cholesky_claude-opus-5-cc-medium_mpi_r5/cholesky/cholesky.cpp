#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// Cholesky decomposition, distributed-memory parallel version (MPI).
//
// Decomposes a positive definite matrix A into L * L^T where L is lower triangular.
//
// Parallelization strategy:
//   * The matrix is padded to a multiple of the block size and distributed over a
//     2D process grid (pr x pc) in a block-cyclic fashion (the standard ScaLAPACK
//     style layout), which gives good load balance and O(n^2/sqrt(P)) communication
//     volume per rank.
//   * A right-looking blocked Cholesky is used: for every block column K the owner
//     factorizes the diagonal block, the block column below it is updated with a
//     triangular solve and the resulting panel is broadcast along the process rows
//     and (transposed) along the process columns.  Every rank then updates its part
//     of the trailing sub-matrix with a local GEMM.
//   * Matrix generation (A = B*B^T) and the validation step (L*L^T) reuse the very
//     same panel-exchange machinery, so they are fully distributed as well.
//
// The matrix is never replicated; only the final result printing (-r) assembles the
// full matrix on rank 0.

namespace {

int g_rank = 0;
int g_nranks = 1;

#define ROOT_PRINT(...)                                                                            \
    do {                                                                                           \
        if (g_rank == 0) printf(__VA_ARGS__);                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// Reproduction of glibc's rand_r() sequence with jump-ahead, so that every rank
// can generate exactly the sub-set of the random matrix B that it needs while
// producing bit-identical values to the sequential reference implementation.
// ---------------------------------------------------------------------------

struct LcgMap {
    uint32_t a, c;
};

inline LcgMap lcg_compose(const LcgMap f, const LcgMap g) { // f(g(x))
    return LcgMap{f.a * g.a, f.a * g.c + f.c};
}

// State of the underlying LCG after `steps` advances, starting from `x`.
inline uint32_t lcg_jump(uint32_t x, uint64_t steps) {
    LcgMap r{1u, 0u};
    LcgMap base{1103515245u, 12345u};
    while (steps) {
        if (steps & 1ull) r = lcg_compose(base, r);
        base = lcg_compose(base, base);
        steps >>= 1;
    }
    return r.a * x + r.c;
}

// One rand_r() call (advances the LCG three times).
inline int lcg_rand(uint32_t& next) {
    int result;
    next = next * 1103515245u + 12345u;
    result = static_cast<int>((next / 65536u) % 2048u);
    next = next * 1103515245u + 12345u;
    result <<= 10;
    result ^= static_cast<int>((next / 65536u) % 1024u);
    next = next * 1103515245u + 12345u;
    result <<= 10;
    result ^= static_cast<int>((next / 65536u) % 1024u);
    return result;
}

// ---------------------------------------------------------------------------
// Distributed context
// ---------------------------------------------------------------------------

struct DistCtx {
    int pr = 1, pc = 1, myrow = 0, mycol = 0;
    MPI_Comm rowcomm = MPI_COMM_NULL; // all ranks of my process row  (ordered by mycol)
    MPI_Comm colcomm = MPI_COMM_NULL; // all ranks of my process col  (ordered by myrow)

    size_t n = 0;    // logical matrix dimension
    size_t b = 64;   // block size
    size_t nb = 0;   // number of blocks per dimension
    size_t npad = 0; // padded matrix dimension (nb * b)
    size_t mrb = 0;  // local number of block rows
    size_t mcb = 0;  // local number of block columns

};

DistCtx C;

inline size_t local_block_count(size_t nb, int p, int me) {
    if (static_cast<size_t>(me) >= nb) return 0;
    return (nb - static_cast<size_t>(me) - 1) / static_cast<size_t>(p) + 1;
}

// First local block index whose global block index is >= gmin.
inline size_t first_local(size_t gmin, int p, int me, size_t nloc) {
    if (gmin <= static_cast<size_t>(me)) return 0;
    const size_t v = (gmin - static_cast<size_t>(me) + static_cast<size_t>(p) - 1) / static_cast<size_t>(p);
    return std::min(v, nloc);
}

// ---------------------------------------------------------------------------
// Local kernels (all blocks are b x b, row major)
// ---------------------------------------------------------------------------

// C += SIGN * A * B^T, where B is supplied already transposed: BT[k*b+j] = B[j*b+k].
// Accumulation over k is strictly ascending for every output element, matching the
// summation order of the sequential reference code.
//
// The register tile is 6x8 (12 AVX2 accumulators), which reaches ~94% of the
// machine's FMA peak; the block size is therefore kept a multiple of 24.
template <int SIGN>
void gemm_nt_block(double* __restrict Cb, const double* __restrict A, const double* __restrict BT,
                   const int b) {
#if defined(__AVX2__) && defined(__FMA__)
    constexpr int MR = 6;
    constexpr int NV = 2; // AVX2 vectors per row tile -> NR = 8
    for (int i0 = 0; i0 < b; i0 += MR) {
        for (int j0 = 0; j0 < b; j0 += 4 * NV) {
            __m256d acc[MR][NV];
            for (int i = 0; i < MR; ++i) {
                for (int v = 0; v < NV; ++v)
                    acc[i][v] = _mm256_loadu_pd(Cb + (i0 + i) * b + j0 + 4 * v);
            }
            for (int k = 0; k < b; ++k) {
                const double* __restrict bt = BT + static_cast<size_t>(k) * b + j0;
                __m256d bv[NV];
                for (int v = 0; v < NV; ++v) bv[v] = _mm256_loadu_pd(bt + 4 * v);
                for (int i = 0; i < MR; ++i) {
                    const __m256d a = _mm256_set1_pd(A[static_cast<size_t>(i0 + i) * b + k]);
                    for (int v = 0; v < NV; ++v) {
                        acc[i][v] = (SIGN < 0) ? _mm256_fnmadd_pd(a, bv[v], acc[i][v])
                                               : _mm256_fmadd_pd(a, bv[v], acc[i][v]);
                    }
                }
            }
            for (int i = 0; i < MR; ++i) {
                for (int v = 0; v < NV; ++v)
                    _mm256_storeu_pd(Cb + (i0 + i) * b + j0 + 4 * v, acc[i][v]);
            }
        }
    }
#else
    constexpr int MR = 2;
    constexpr int NR = 8;
    for (int i0 = 0; i0 < b; i0 += MR) {
        for (int j0 = 0; j0 < b; j0 += NR) {
            double acc[MR][NR];
            for (int i = 0; i < MR; ++i) {
                for (int j = 0; j < NR; ++j) acc[i][j] = Cb[(i0 + i) * b + j0 + j];
            }
            for (int k = 0; k < b; ++k) {
                const double* __restrict bt = BT + static_cast<size_t>(k) * b + j0;
                for (int i = 0; i < MR; ++i) {
                    const double a = SIGN * A[static_cast<size_t>(i0 + i) * b + k];
                    for (int j = 0; j < NR; ++j) acc[i][j] += a * bt[j];
                }
            }
            for (int i = 0; i < MR; ++i) {
                for (int j = 0; j < NR; ++j) Cb[(i0 + i) * b + j0 + j] = acc[i][j];
            }
        }
    }
#endif
}

// Unblocked Cholesky of a single diagonal block (lower triangle, row major).
bool potrf_block(double* __restrict D, const int b, const size_t goff) {
    for (int j = 0; j < b; ++j) {
        double sum = 0.0;
        for (int k = 0; k < j; ++k) sum += D[j * b + k] * D[j * b + k];
        const double val = D[j * b + j] - sum;
        if (val <= 0.0) {
            printf("Error: Matrix is not positive definite at diagonal element %zu\n",
                   goff + static_cast<size_t>(j));
            return false;
        }
        const double djj = sqrt(val);
        D[j * b + j] = djj;
        for (int i = j + 1; i < b; ++i) {
            double s = 0.0;
            for (int k = 0; k < j; ++k) s += D[i * b + k] * D[j * b + k];
            D[i * b + j] = (D[i * b + j] - s) / djj;
        }
    }
    return true;
}

// Solve X * D^T = X in place (X is m x b row major, D is b x b lower triangular).
// Dt holds the transpose of D so that the inner loop is contiguous.
void trsm_block(double* __restrict X, const double* __restrict D, const double* __restrict Dt,
                const int b, const int m) {
    constexpr int MR = 4; // four independent rows keep the vector units busy
    for (int i0 = 0; i0 < m; i0 += MR) {
        double* __restrict x0 = X + static_cast<size_t>(i0 + 0) * b;
        double* __restrict x1 = X + static_cast<size_t>(i0 + 1) * b;
        double* __restrict x2 = X + static_cast<size_t>(i0 + 2) * b;
        double* __restrict x3 = X + static_cast<size_t>(i0 + 3) * b;
        for (int j = 0; j < b; ++j) {
            const double djj = D[j * b + j];
            const double v0 = x0[j] / djj;
            const double v1 = x1[j] / djj;
            const double v2 = x2[j] / djj;
            const double v3 = x3[j] / djj;
            x0[j] = v0;
            x1[j] = v1;
            x2[j] = v2;
            x3[j] = v3;
            const double* __restrict dt = Dt + static_cast<size_t>(j) * b;
            for (int k = j + 1; k < b; ++k) {
                x0[k] -= v0 * dt[k];
                x1[k] -= v1 * dt[k];
                x2[k] -= v2 * dt[k];
                x3[k] -= v3 * dt[k];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Panel exchange: make block column K of a matrix available everywhere it is
// needed for the rank-b update of the trailing sub-matrix:
//   rowpanel[ib - ib0]     = M[I][K]     for the local block rows I >= Imin
//   colpanel[colpos[jb]]   = M[J][K]^T   for the local block cols J >= Imin
// This takes two steps: a broadcast along the process row (from the process
// column owning block column K) followed by an all-gather along the process
// column.  Both are issued as non-blocking collectives so that they can be
// overlapped with the trailing update of the previous step.
//
// `status` is piggy-backed on the broadcast so error conditions reach every rank.
// ---------------------------------------------------------------------------
struct Panel {
    std::vector<double> rowpanel;
    std::vector<double> colpanel;
    std::vector<double> packbuf;
    std::vector<int> cnt, dispb, counts, displs;
    std::vector<size_t> colpos; // local block col -> block index inside colpanel
    std::vector<std::pair<uint32_t, uint32_t>> items; // (ib, jb) update work list
    size_t nprio = 0;                                 // leading items of the next panel column
    size_t K = 0, Imin = 0, ib0 = 0, jb0 = 0, nrow_act = 0;
    MPI_Request req = MPI_REQUEST_NULL;
    double status = 0.0;
};

void panel_start_row(Panel& P, const std::vector<double>& M, size_t K, size_t Imin, double status) {
    const size_t bb = C.b * C.b;
    P.K = K;
    P.Imin = Imin;
    P.ib0 = first_local(Imin, C.pr, C.myrow, C.mrb);
    P.nrow_act = C.mrb - P.ib0;
    P.rowpanel.resize(P.nrow_act * bb + 1);
    const int root_col = static_cast<int>(K % static_cast<size_t>(C.pc));
    if (C.mycol == root_col) {
        const size_t jbK = K / static_cast<size_t>(C.pc);
        for (size_t ib = P.ib0; ib < C.mrb; ++ib) {
            memcpy(&P.rowpanel[(ib - P.ib0) * bb], &M[(ib * C.mcb + jbK) * bb], bb * sizeof(double));
        }
        P.rowpanel[P.nrow_act * bb] = status;
    }
    MPI_Ibcast(P.rowpanel.data(), static_cast<int>(P.nrow_act * bb + 1), MPI_DOUBLE, root_col,
               C.rowcomm, &P.req);
}

// Completes the row broadcast and builds the local work list for this panel.
void panel_finish_row(Panel& P) {
    MPI_Wait(&P.req, MPI_STATUS_IGNORE);
    P.status = P.rowpanel[P.nrow_act * C.b * C.b];
    if (P.status != 0.0) return;

    P.jb0 = first_local(P.Imin, C.pc, C.mycol, C.mcb);

    // Work list: blocks (I,J) with I >= Imin, Imin <= J <= I.  The blocks of the
    // next panel column are put first so that they can be updated (and the next
    // panel produced) before the bulk of the trailing update.
    const size_t nextJ = P.K + 1;
    const bool prio = P.jb0 < C.mcb &&
                      static_cast<size_t>(C.mycol) + P.jb0 * static_cast<size_t>(C.pc) == nextJ;
    P.items.clear();
    if (prio) {
        for (size_t ib = P.ib0; ib < C.mrb; ++ib) {
            const size_t I = static_cast<size_t>(C.myrow) + ib * static_cast<size_t>(C.pr);
            if (nextJ <= I) P.items.emplace_back(static_cast<uint32_t>(ib), static_cast<uint32_t>(P.jb0));
        }
    }
    P.nprio = P.items.size();
    const size_t jbs = prio ? P.jb0 + 1 : P.jb0;
    for (size_t ib = P.ib0; ib < C.mrb; ++ib) {
        const size_t I = static_cast<size_t>(C.myrow) + ib * static_cast<size_t>(C.pr);
        for (size_t jb = jbs; jb < C.mcb; ++jb) {
            const size_t J = static_cast<size_t>(C.mycol) + jb * static_cast<size_t>(C.pc);
            if (J > I) break;
            P.items.emplace_back(static_cast<uint32_t>(ib), static_cast<uint32_t>(jb));
        }
    }
}

void panel_start_col(Panel& P) {
    const size_t b = C.b;
    const size_t bb = b * b;

    P.cnt.assign(C.pr, 0);
    for (size_t jb = P.jb0; jb < C.mcb; ++jb) {
        const size_t J = static_cast<size_t>(C.mycol) + jb * static_cast<size_t>(C.pc);
        P.cnt[J % static_cast<size_t>(C.pr)]++;
    }
    P.dispb.resize(C.pr);
    P.counts.resize(C.pr);
    P.displs.resize(C.pr);
    int acc = 0;
    for (int r = 0; r < C.pr; ++r) {
        P.dispb[r] = acc;
        P.counts[r] = static_cast<int>(P.cnt[r] * static_cast<int>(bb));
        P.displs[r] = static_cast<int>(acc * static_cast<int>(bb));
        acc += P.cnt[r];
    }
    P.colpos.resize(C.mcb);
    {
        std::vector<int> idx(P.dispb);
        for (size_t jb = P.jb0; jb < C.mcb; ++jb) {
            const size_t J = static_cast<size_t>(C.mycol) + jb * static_cast<size_t>(C.pc);
            P.colpos[jb] = static_cast<size_t>(idx[J % static_cast<size_t>(C.pr)]++);
        }
    }

    // pack (transposed) the blocks that I own and that are needed as column operands
    P.packbuf.resize(static_cast<size_t>(P.cnt[C.myrow]) * bb);
    size_t p = 0;
    for (size_t ib = P.ib0; ib < C.mrb; ++ib) {
        const size_t I = static_cast<size_t>(C.myrow) + ib * static_cast<size_t>(C.pr);
        if (I % static_cast<size_t>(C.pc) != static_cast<size_t>(C.mycol)) continue;
        const double* src = &P.rowpanel[(ib - P.ib0) * bb];
        double* dst = &P.packbuf[p * bb];
        for (size_t r = 0; r < b; ++r) {
            for (size_t c = 0; c < b; ++c) dst[c * b + r] = src[r * b + c];
        }
        ++p;
    }

    P.colpanel.resize(static_cast<size_t>(acc) * bb);
    MPI_Iallgatherv(P.packbuf.data(), P.counts[C.myrow], MPI_DOUBLE, P.colpanel.data(),
                    P.counts.data(), P.displs.data(), MPI_DOUBLE, C.colcomm, &P.req);
}

inline void panel_finish_col(Panel& P) { MPI_Wait(&P.req, MPI_STATUS_IGNORE); }

void panel_exchange(Panel& P, const std::vector<double>& M, size_t K, size_t Imin) {
    panel_start_row(P, M, K, Imin, 0.0);
    panel_finish_row(P);
    panel_start_col(P);
    panel_finish_col(P);
}

// Apply the panel update to the work list entries [from, to).  `progress` (if given)
// is polled regularly, which is what actually advances the overlapped collectives.
template <int SIGN>
void update_items(const Panel& P, std::vector<double>& dst, size_t from, size_t to,
                  MPI_Request* progress = nullptr) {
    const size_t bb = C.b * C.b;
    const int bi = static_cast<int>(C.b);
    for (size_t it = from; it < to; ++it) {
        const size_t ib = P.items[it].first;
        const size_t jb = P.items[it].second;
        gemm_nt_block<SIGN>(&dst[(ib * C.mcb + jb) * bb], &P.rowpanel[(ib - P.ib0) * bb],
                            &P.colpanel[P.colpos[jb] * bb], bi);
        if (progress && *progress != MPI_REQUEST_NULL) {
            int flag = 0;
            MPI_Test(progress, &flag, MPI_STATUS_IGNORE);
        }
    }
}

// ---------------------------------------------------------------------------
// Distributed right-looking blocked Cholesky with one step of look-ahead:
// the block column needed by the next step is updated first, the next panel is
// then produced and its communication is overlapped with the remaining update.
// ---------------------------------------------------------------------------

// Factorize the diagonal block of block column K, apply the triangular solve to
// the block column below it and start the panel broadcast.
void produce_panel(Panel& P, std::vector<double>& A, size_t K, std::vector<double>& D,
                   std::vector<double>& Dt) {
    const size_t b = C.b;
    const size_t bb = b * b;
    double status = 0.0;
    const int kr = static_cast<int>(K % static_cast<size_t>(C.pr));
    const int kc = static_cast<int>(K % static_cast<size_t>(C.pc));

    if (C.mycol == kc) {
        if (C.myrow == kr) {
            double* blk =
                &A[((K / static_cast<size_t>(C.pr)) * C.mcb + K / static_cast<size_t>(C.pc)) * bb];
            if (!potrf_block(blk, static_cast<int>(b), K * b)) status = 1.0;
            memcpy(D.data(), blk, bb * sizeof(double));
        }
        D[bb] = status;
        MPI_Bcast(D.data(), static_cast<int>(bb + 1), MPI_DOUBLE, kr, C.colcomm);
        status = D[bb];

        if (status == 0.0) {
            for (size_t r = 0; r < b; ++r) {
                for (size_t c = 0; c < b; ++c) Dt[c * b + r] = D[r * b + c];
            }
            const size_t ibs = first_local(K + 1, C.pr, C.myrow, C.mrb);
            const size_t jbK = K / static_cast<size_t>(C.pc);
            for (size_t ib = ibs; ib < C.mrb; ++ib) {
                trsm_block(&A[(ib * C.mcb + jbK) * bb], D.data(), Dt.data(), static_cast<int>(b),
                           static_cast<int>(b));
            }
        }
    }
    panel_start_row(P, A, K, K + 1, status);
}

bool choleskyDecomposition(std::vector<double>& A) {
    const size_t b = C.b;
    const size_t bb = b * b;
    std::vector<double> D(bb + 1), Dt(bb);
    Panel panels[2];
    bool failed = false;
    if (C.nb == 0) return true;

    // prologue: first panel
    produce_panel(panels[0], A, 0, D, Dt);
    panel_finish_row(panels[0]);
    if (panels[0].status == 0.0) {
        panel_start_col(panels[0]);
        panel_finish_col(panels[0]);
    } else {
        failed = true;
    }

    for (size_t K = 0; !failed && K < C.nb; ++K) {
        Panel& cur = panels[K & 1];
        Panel& nxt = panels[(K + 1) & 1];
        const bool more = (K + 1 < C.nb);

        // 1) update the block column needed by the next step, then produce that panel
        update_items<-1>(cur, A, 0, cur.nprio);
        if (more) produce_panel(nxt, A, K + 1, D, Dt);

        // 2) bulk of the trailing update, overlapped with the panel communication
        const size_t rest = cur.items.size();
        const size_t mid = cur.nprio + (rest - cur.nprio) / 2;
        update_items<-1>(cur, A, cur.nprio, mid, more ? &nxt.req : nullptr);
        if (more) {
            panel_finish_row(nxt);
            if (nxt.status != 0.0) failed = true;
            else panel_start_col(nxt);
        }
        update_items<-1>(cur, A, mid, rest, (more && !failed) ? &nxt.req : nullptr);
        if (more && !failed) panel_finish_col(nxt);
    }

    if (failed) return false;

    // Zero out the upper triangular part (the lower triangle holds L).
    for (size_t ib = 0; ib < C.mrb; ++ib) {
        const size_t I = static_cast<size_t>(C.myrow) + ib * static_cast<size_t>(C.pr);
        for (size_t jb = 0; jb < C.mcb; ++jb) {
            const size_t J = static_cast<size_t>(C.mycol) + jb * static_cast<size_t>(C.pc);
            double* blk = &A[(ib * C.mcb + jb) * bb];
            if (J > I) {
                memset(blk, 0, bb * sizeof(double));
            } else if (J == I) {
                for (size_t r = 0; r < b; ++r) {
                    for (size_t c = r + 1; c < b; ++c) blk[r * b + c] = 0.0;
                }
            }
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Generate a symmetric positive definite matrix: A = B*B^T + n*I with B random.
// Only the lower block triangle is computed (A is symmetric); the padding region
// is filled with the identity so that the padded matrix stays positive definite.
// ---------------------------------------------------------------------------
void generatePositiveDefiniteMatrix(std::vector<double>& A) {
    const size_t b = C.b;
    const size_t bb = b * b;
    const size_t n = C.n;

    // Generate the local part of B directly from the jump-ahead random sequence.
    std::vector<double> B(C.mrb * C.mcb * bb, 0.0);
    for (size_t ib = 0; ib < C.mrb; ++ib) {
        const size_t I = static_cast<size_t>(C.myrow) + ib * static_cast<size_t>(C.pr);
        for (size_t jb = 0; jb < C.mcb; ++jb) {
            const size_t J = static_cast<size_t>(C.mycol) + jb * static_cast<size_t>(C.pc);
            double* blk = &B[(ib * C.mcb + jb) * bb];
            for (size_t r = 0; r < b; ++r) {
                const size_t i = I * b + r;
                if (i >= n) break;
                const size_t j0 = J * b;
                if (j0 >= n) break;
                const size_t len = std::min(b, n - j0);
                uint32_t state = lcg_jump(42u, 3ull * (static_cast<uint64_t>(i) * n + j0));
                double* row = blk + r * b;
                for (size_t c = 0; c < len; ++c) {
                    row[c] = (lcg_rand(state) / static_cast<double>(RAND_MAX)) - 0.5;
                }
            }
        }
    }

    std::fill(A.begin(), A.end(), 0.0);
    Panel P;
    for (size_t K = 0; K < C.nb; ++K) {
        panel_exchange(P, B, K, 0);
        update_items<1>(P, A, 0, P.items.size());
    }

    // Diagonal: add n for the real part, identity for the padding.
    for (size_t ib = 0; ib < C.mrb; ++ib) {
        const size_t I = static_cast<size_t>(C.myrow) + ib * static_cast<size_t>(C.pr);
        if (I % static_cast<size_t>(C.pc) != static_cast<size_t>(C.mycol)) continue;
        const size_t jb = I / static_cast<size_t>(C.pc);
        double* blk = &A[(ib * C.mcb + jb) * bb];
        for (size_t r = 0; r < b; ++r) {
            const size_t i = I * b + r;
            if (i < n) {
                blk[r * b + r] += static_cast<double>(n);
            } else {
                blk[r * b + r] = 1.0;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Validation: compute L*L^T (distributed) and compare against the original matrix.
// ---------------------------------------------------------------------------
bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig) {
    const size_t b = C.b;
    const size_t bb = b * b;

    std::vector<double> R(L.size(), 0.0);
    Panel P;
    for (size_t K = 0; K < C.nb; ++K) {
        panel_exchange(P, L, K, K);
        update_items<1>(P, R, 0, P.items.size());
    }

    double local[2] = {0.0, 0.0};
    for (size_t ib = 0; ib < C.mrb; ++ib) {
        const size_t I = static_cast<size_t>(C.myrow) + ib * static_cast<size_t>(C.pr);
        for (size_t jb = 0; jb < C.mcb; ++jb) {
            const size_t J = static_cast<size_t>(C.mycol) + jb * static_cast<size_t>(C.pc);
            if (J > I) break;
            const double* r = &R[(ib * C.mcb + jb) * bb];
            const double* a = &A_orig[(ib * C.mcb + jb) * bb];
            for (size_t e = 0; e < bb; ++e) {
                const double error = fabs(r[e] - a[e]);
                local[0] = std::max(local[0], error);
                local[1] = std::max(local[1], error / (fabs(a[e]) + 1e-10));
            }
        }
    }

    double global[2];
    MPI_Allreduce(local, global, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    ROOT_PRINT("Max absolute error: %.10e\n", global[0]);
    ROOT_PRINT("Max relative error: %.10e\n", global[1]);

    if (global[1] > 1e-6) {
        ROOT_PRINT("Validation failed: relative error too large\n");
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Collect the distributed result into a full n x n matrix on rank 0.
// ---------------------------------------------------------------------------
void gatherFull(const std::vector<double>& A, std::vector<double>& full) {
    const size_t b = C.b;
    const size_t bb = b * b;
    const size_t n = C.n;

    auto unpack = [&](const double* buf, int src) {
        const int rrow = src / C.pc;
        const int rcol = src % C.pc;
        const size_t mrb = local_block_count(C.nb, C.pr, rrow);
        const size_t mcb = local_block_count(C.nb, C.pc, rcol);
        for (size_t ib = 0; ib < mrb; ++ib) {
            const size_t I = static_cast<size_t>(rrow) + ib * static_cast<size_t>(C.pr);
            if (I * b >= n) continue;
            const size_t nr = std::min(b, n - I * b);
            for (size_t jb = 0; jb < mcb; ++jb) {
                const size_t J = static_cast<size_t>(rcol) + jb * static_cast<size_t>(C.pc);
                if (J * b >= n) continue;
                const size_t nc = std::min(b, n - J * b);
                const double* blk = buf + (ib * mcb + jb) * bb;
                for (size_t r = 0; r < nr; ++r) {
                    memcpy(&full[(I * b + r) * n + J * b], blk + r * b, nc * sizeof(double));
                }
            }
        }
    };

    if (g_rank == 0) {
        full.assign(n * n, 0.0);
        unpack(A.data(), 0);
        std::vector<double> tmp;
        for (int src = 1; src < g_nranks; ++src) {
            const size_t cnt = local_block_count(C.nb, C.pr, src / C.pc) *
                               local_block_count(C.nb, C.pc, src % C.pc) * bb;
            tmp.resize(std::max<size_t>(cnt, 1));
            if (cnt > 0) {
                MPI_Recv(tmp.data(), static_cast<int>(cnt), MPI_DOUBLE, src, 7, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
                unpack(tmp.data(), src);
            }
        }
    } else if (!A.empty()) {
        MPI_Send(A.data(), static_cast<int>(A.size()), MPI_DOUBLE, 0, 7, MPI_COMM_WORLD);
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

} // namespace

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

    // --- build the 2D process grid (as square as possible, pr >= pc)
    int pc = 1;
    for (int c = 1; c * c <= g_nranks; ++c) {
        if (g_nranks % c == 0) pc = c;
    }
    C.pr = g_nranks / pc;
    C.pc = pc;
    C.myrow = g_rank / pc;
    C.mycol = g_rank % pc;
    MPI_Comm_split(MPI_COMM_WORLD, C.myrow, C.mycol, &C.rowcomm);
    MPI_Comm_split(MPI_COMM_WORLD, C.mycol, C.myrow, &C.colcomm);

    // --- block size (multiple of 24, see gemm_nt_block).  48 is the empirical sweet
    //     spot between the cost of the serial critical path (diagonal factorization,
    //     triangular solve and panel latency, all growing with b) and the arithmetic
    //     intensity of the trailing update (growing with b).  It is reduced when the
    //     matrix is too small to give every process grid dimension enough blocks.
    size_t b = 48;
    const size_t want = 2 * static_cast<size_t>(std::max(C.pr, C.pc));
    while (b > 24 && (n + b - 1) / b < want) b -= 24;

    C.n = n;
    C.b = b;
    C.nb = (n + b - 1) / b;
    C.npad = C.nb * b;
    C.mrb = local_block_count(C.nb, C.pr, C.myrow);
    C.mcb = local_block_count(C.nb, C.pc, C.mycol);

    ROOT_PRINT("Cholesky Decomposition Benchmark\n");
    ROOT_PRINT("Matrix size: %zu x %zu\n", n, n);
    ROOT_PRINT("Validation: %s\n", validate ? "enabled" : "disabled");
    ROOT_PRINT("MPI ranks: %d (process grid %d x %d, block size %zu)\n", g_nranks, C.pr, C.pc, b);

    // Allocate the local part of the matrix
    std::vector<double> A(C.mrb * C.mcb * b * b);
    std::vector<double> A_orig;

    // Generate positive definite matrix
    ROOT_PRINT("Generating positive definite matrix...\n");
    generatePositiveDefiniteMatrix(A);

    if (validate) {
        A_orig = A; // Save original for validation
    }

    // Perform Cholesky decomposition
    ROOT_PRINT("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    bool success = choleskyDecomposition(A);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (!success) {
        ROOT_PRINT("Cholesky decomposition failed\n");
        MPI_Finalize();
        return 1;
    }

    ROOT_PRINT("Computation time: %ld ms\n", duration.count());

    // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
    double ops = (double)n * n * n / 3.0;
    double gflops = ops / (duration.count() / 1000.0) / 1e9;
    ROOT_PRINT("Performance: %.3f GFLOPS\n", gflops);

    // Print results for external validation
    if (printResults) {
        std::vector<double> full;
        gatherFull(A, full);
        if (g_rank == 0) print_results(full, "CholeskyL");
    }

    // Validation
    int rc = 0;
    if (validate) {
        ROOT_PRINT("Validating result...\n");
        bool valid = validateCholesky(A, A_orig);

        if (valid) {
            ROOT_PRINT("Validation: PASSED\n");
        } else {
            ROOT_PRINT("Validation: FAILED\n");
            rc = 1;
        }
    }

    MPI_Finalize();
    return rc;
}
