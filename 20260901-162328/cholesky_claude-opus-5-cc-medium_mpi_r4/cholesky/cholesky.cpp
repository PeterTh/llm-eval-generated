#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <immintrin.h>
#include <mpi.h>

#include "../common/results_output.hpp"

// Cholesky decomposition, distributed with MPI (2D block-cyclic, right-looking
// blocked algorithm).  Decomposes a positive definite matrix A into L * L^T
// where L is lower triangular.
//
// The matrix is distributed over a pr x pc process grid in a 2D block-cyclic
// fashion (ScaLAPACK style).  Each step of the outer loop factors one block
// column (panel) and then updates the trailing submatrix with a rank-nb update
// (C -= L_panel * L_panel^T), which is where essentially all of the flops are.
// The update runs on a packed AVX2/FMA micro-kernel; the panel of the next step
// is factored and broadcast ahead of the bulk of the current update (one-step
// lookahead) so that the panel critical path and all panel communication
// overlap with computation.
//
// The random matrix B used to build A = B*B^T + n*I is replicated once per
// compute node in an MPI-3 shared memory window, so that every rank can
// generate (and validate) its own part of the matrix without holding a private
// copy of an n x n array.

// ---------------------------------------------------------------------------
// Process grid / distribution helpers
// ---------------------------------------------------------------------------

struct Grid {
    MPI_Comm world = MPI_COMM_NULL;
    MPI_Comm row = MPI_COMM_NULL; // ranks with the same process row (varying pj)
    MPI_Comm col = MPI_COMM_NULL;  // ranks with the same process column (varying pi)
    MPI_Comm colT = MPI_COMM_NULL; // duplicate of 'col', used for the panel transpose
    int rank = 0;
    int size = 1;
    int pr = 1, pc = 1;
    int pi = 0, pj = 0;
};

// Number of local elements among the first 'nglob' global indices for process
// 'iproc' of 'nprocs' with block size 'nb' (ScaLAPACK numroc).  Doubles as the
// local index of global index 'nglob'.
static inline size_t numroc(size_t nglob, size_t nb, int iproc, int nprocs) {
    const size_t nblocks = nglob / nb;
    const size_t rem = nglob % nb;
    const size_t full = nblocks / (size_t)nprocs;
    const size_t extra = nblocks % (size_t)nprocs;
    size_t loc = full * nb;
    if((size_t)iproc < extra) {
        loc += nb;
    } else if((size_t)iproc == extra) {
        loc += rem;
    }
    return loc;
}

// ---------------------------------------------------------------------------
// Micro-kernel based DGEMM: C(m x nn) -= A(m x k) * Bt(nn x k)^T
// All matrices are column-major.
// ---------------------------------------------------------------------------

static constexpr size_t GEMM_MR = 8;
static constexpr size_t GEMM_NR = 6;
static constexpr size_t GEMM_MC = 192; // rows per packed A block (multiple of MR)

static inline void kernel_8x6(size_t k, const double* __restrict Ap, const double* __restrict Bp,
    double* __restrict C, size_t ldc, size_t mr, size_t nr) {
    __m256d c00 = _mm256_setzero_pd(), c01 = _mm256_setzero_pd();
    __m256d c10 = _mm256_setzero_pd(), c11 = _mm256_setzero_pd();
    __m256d c20 = _mm256_setzero_pd(), c21 = _mm256_setzero_pd();
    __m256d c30 = _mm256_setzero_pd(), c31 = _mm256_setzero_pd();
    __m256d c40 = _mm256_setzero_pd(), c41 = _mm256_setzero_pd();
    __m256d c50 = _mm256_setzero_pd(), c51 = _mm256_setzero_pd();

    for(size_t p = 0; p < k; ++p) {
        const __m256d a0 = _mm256_loadu_pd(Ap);
        const __m256d a1 = _mm256_loadu_pd(Ap + 4);
        __m256d b;
        b = _mm256_broadcast_sd(Bp + 0);
        c00 = _mm256_fmadd_pd(a0, b, c00);
        c01 = _mm256_fmadd_pd(a1, b, c01);
        b = _mm256_broadcast_sd(Bp + 1);
        c10 = _mm256_fmadd_pd(a0, b, c10);
        c11 = _mm256_fmadd_pd(a1, b, c11);
        b = _mm256_broadcast_sd(Bp + 2);
        c20 = _mm256_fmadd_pd(a0, b, c20);
        c21 = _mm256_fmadd_pd(a1, b, c21);
        b = _mm256_broadcast_sd(Bp + 3);
        c30 = _mm256_fmadd_pd(a0, b, c30);
        c31 = _mm256_fmadd_pd(a1, b, c31);
        b = _mm256_broadcast_sd(Bp + 4);
        c40 = _mm256_fmadd_pd(a0, b, c40);
        c41 = _mm256_fmadd_pd(a1, b, c41);
        b = _mm256_broadcast_sd(Bp + 5);
        c50 = _mm256_fmadd_pd(a0, b, c50);
        c51 = _mm256_fmadd_pd(a1, b, c51);
        Ap += GEMM_MR;
        Bp += GEMM_NR;
    }

    if(mr == GEMM_MR && nr == GEMM_NR) {
#define CHOL_STORE_COL(j, lo, hi)                                                                                      \
    {                                                                                                                  \
        double* cp = C + (j) * ldc;                                                                                    \
        _mm256_storeu_pd(cp, _mm256_sub_pd(_mm256_loadu_pd(cp), lo));                                                   \
        _mm256_storeu_pd(cp + 4, _mm256_sub_pd(_mm256_loadu_pd(cp + 4), hi));                                           \
    }
        CHOL_STORE_COL(0, c00, c01)
        CHOL_STORE_COL(1, c10, c11)
        CHOL_STORE_COL(2, c20, c21)
        CHOL_STORE_COL(3, c30, c31)
        CHOL_STORE_COL(4, c40, c41)
        CHOL_STORE_COL(5, c50, c51)
#undef CHOL_STORE_COL
    } else {
        alignas(32) double tmp[GEMM_NR * GEMM_MR];
        _mm256_store_pd(tmp + 0, c00);
        _mm256_store_pd(tmp + 4, c01);
        _mm256_store_pd(tmp + 8, c10);
        _mm256_store_pd(tmp + 12, c11);
        _mm256_store_pd(tmp + 16, c20);
        _mm256_store_pd(tmp + 20, c21);
        _mm256_store_pd(tmp + 24, c30);
        _mm256_store_pd(tmp + 28, c31);
        _mm256_store_pd(tmp + 32, c40);
        _mm256_store_pd(tmp + 36, c41);
        _mm256_store_pd(tmp + 40, c50);
        _mm256_store_pd(tmp + 44, c51);
        for(size_t j = 0; j < nr; ++j) {
            for(size_t i = 0; i < mr; ++i) {
                C[i + j * ldc] -= tmp[j * GEMM_MR + i];
            }
        }
    }
}

static void gemm_nt_sub(size_t m, size_t nn, size_t k, const double* A, size_t lda, const double* Bt, size_t ldb,
    double* C, size_t ldc, std::vector<double>& packA, std::vector<double>& packB) {
    if(m == 0 || nn == 0 || k == 0) { return; }

    const size_t njp = (nn + GEMM_NR - 1) / GEMM_NR;
    packB.resize(njp * GEMM_NR * k);
    for(size_t jp = 0; jp < njp; ++jp) {
        double* dst = packB.data() + jp * GEMM_NR * k;
        const size_t nr = std::min(GEMM_NR, nn - jp * GEMM_NR);
        if(nr == GEMM_NR) {
            for(size_t p = 0; p < k; ++p) {
                const double* src = Bt + jp * GEMM_NR + p * ldb;
                for(size_t jj = 0; jj < GEMM_NR; ++jj) {
                    dst[p * GEMM_NR + jj] = src[jj];
                }
            }
        } else {
            for(size_t p = 0; p < k; ++p) {
                const double* src = Bt + jp * GEMM_NR + p * ldb;
                for(size_t jj = 0; jj < GEMM_NR; ++jj) {
                    dst[p * GEMM_NR + jj] = (jj < nr) ? src[jj] : 0.0;
                }
            }
        }
    }

    packA.resize(GEMM_MC * k);
    for(size_t i0 = 0; i0 < m; i0 += GEMM_MC) {
        const size_t mb = std::min(GEMM_MC, m - i0);
        const size_t nip = (mb + GEMM_MR - 1) / GEMM_MR;
        for(size_t ip = 0; ip < nip; ++ip) {
            double* dst = packA.data() + ip * GEMM_MR * k;
            const size_t mr = std::min(GEMM_MR, mb - ip * GEMM_MR);
            if(mr == GEMM_MR) {
                for(size_t p = 0; p < k; ++p) {
                    const double* src = A + i0 + ip * GEMM_MR + p * lda;
                    for(size_t ii = 0; ii < GEMM_MR; ++ii) {
                        dst[p * GEMM_MR + ii] = src[ii];
                    }
                }
            } else {
                for(size_t p = 0; p < k; ++p) {
                    const double* src = A + i0 + ip * GEMM_MR + p * lda;
                    for(size_t ii = 0; ii < GEMM_MR; ++ii) {
                        dst[p * GEMM_MR + ii] = (ii < mr) ? src[ii] : 0.0;
                    }
                }
            }
        }
        for(size_t jp = 0; jp < njp; ++jp) {
            const size_t nr = std::min(GEMM_NR, nn - jp * GEMM_NR);
            for(size_t ip = 0; ip < nip; ++ip) {
                const size_t mr = std::min(GEMM_MR, mb - ip * GEMM_MR);
                kernel_8x6(k, packA.data() + ip * GEMM_MR * k, packB.data() + jp * GEMM_NR * k,
                    C + (i0 + ip * GEMM_MR) + jp * GEMM_NR * ldc, ldc, mr, nr);
            }
        }
    }
}

static inline double hsum_pair(__m256d s0, __m256d s1) {
    const __m256d s = _mm256_add_pd(s0, s1);
    __m128d lo = _mm256_castpd256_pd128(s);
    const __m128d hi = _mm256_extractf128_pd(s, 1);
    lo = _mm_add_pd(lo, hi);
    return _mm_cvtsd_f64(_mm_add_sd(lo, _mm_unpackhi_pd(lo, lo)));
}

// Vectorized dot product of two contiguous vectors.
static inline double dot_vec(const double* __restrict a, const double* __restrict b, size_t len) {
    __m256d s0 = _mm256_setzero_pd(), s1 = _mm256_setzero_pd();
    size_t i = 0;
    for(; i + 8 <= len; i += 8) {
        s0 = _mm256_fmadd_pd(_mm256_loadu_pd(a + i), _mm256_loadu_pd(b + i), s0);
        s1 = _mm256_fmadd_pd(_mm256_loadu_pd(a + i + 4), _mm256_loadu_pd(b + i + 4), s1);
    }
    double sum = hsum_pair(s0, s1);
    for(; i < len; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

// Four dot products sharing the vector 'a'.  Each result is accumulated exactly
// like dot_vec() does, so the two routines can be mixed freely, but 'a' is read
// only once, which matters because these products are memory bound.
static inline void dot_vec4(const double* __restrict a, const double* __restrict b0, const double* __restrict b1,
    const double* __restrict b2, const double* __restrict b3, size_t len, double* out) {
    __m256d s00 = _mm256_setzero_pd(), s01 = _mm256_setzero_pd();
    __m256d s10 = _mm256_setzero_pd(), s11 = _mm256_setzero_pd();
    __m256d s20 = _mm256_setzero_pd(), s21 = _mm256_setzero_pd();
    __m256d s30 = _mm256_setzero_pd(), s31 = _mm256_setzero_pd();
    size_t i = 0;
    for(; i + 8 <= len; i += 8) {
        const __m256d a0 = _mm256_loadu_pd(a + i);
        const __m256d a1 = _mm256_loadu_pd(a + i + 4);
        s00 = _mm256_fmadd_pd(a0, _mm256_loadu_pd(b0 + i), s00);
        s01 = _mm256_fmadd_pd(a1, _mm256_loadu_pd(b0 + i + 4), s01);
        s10 = _mm256_fmadd_pd(a0, _mm256_loadu_pd(b1 + i), s10);
        s11 = _mm256_fmadd_pd(a1, _mm256_loadu_pd(b1 + i + 4), s11);
        s20 = _mm256_fmadd_pd(a0, _mm256_loadu_pd(b2 + i), s20);
        s21 = _mm256_fmadd_pd(a1, _mm256_loadu_pd(b2 + i + 4), s21);
        s30 = _mm256_fmadd_pd(a0, _mm256_loadu_pd(b3 + i), s30);
        s31 = _mm256_fmadd_pd(a1, _mm256_loadu_pd(b3 + i + 4), s31);
    }
    out[0] = hsum_pair(s00, s01);
    out[1] = hsum_pair(s10, s11);
    out[2] = hsum_pair(s20, s21);
    out[3] = hsum_pair(s30, s31);
    for(; i < len; ++i) {
        out[0] += a[i] * b0[i];
        out[1] += a[i] * b1[i];
        out[2] += a[i] * b2[i];
        out[3] += a[i] * b3[i];
    }
}

// ---------------------------------------------------------------------------
// Panel kernels (sequential, operate on local data)
// ---------------------------------------------------------------------------

// In-place unblocked Cholesky of the w x w lower triangular block D (col-major,
// leading dimension ld).  Returns -1 on success, or the index of the first
// non-positive-definite diagonal element.
static long factor_diag_block(double* D, size_t ld, size_t w) {
    for(size_t j = 0; j < w; ++j) {
        for(size_t k = 0; k < j; ++k) {
            const double djk = D[j + k * ld];
            double* __restrict col = D + j * ld;
            const double* __restrict prev = D + k * ld;
            for(size_t i = j; i < w; ++i) {
                col[i] -= prev[i] * djk;
            }
        }
        const double val = D[j + j * ld];
        if(val <= 0.0) { return (long)j; }
        const double s = sqrt(val);
        D[j + j * ld] = s;
        double* __restrict col = D + j * ld;
        for(size_t i = j + 1; i < w; ++i) {
            col[i] /= s;
        }
    }
    return -1;
}

// X (m x w, leading dimension ld) := X * D^{-T}, D lower triangular w x w with
// leading dimension w.
static void trsm_right_lower_trans(double* X, size_t ld, size_t m, const double* D, size_t w) {
    if(m == 0) { return; }
    for(size_t j = 0; j < w; ++j) {
        double* __restrict col = X + j * ld;
        for(size_t k = 0; k < j; ++k) {
            const double djk = D[j + k * w];
            const double* __restrict prev = X + k * ld;
            for(size_t i = 0; i < m; ++i) {
                col[i] -= prev[i] * djk;
            }
        }
        const double d = D[j + j * w];
        for(size_t i = 0; i < m; ++i) {
            col[i] /= d;
        }
    }
}

// ---------------------------------------------------------------------------
// Distributed Cholesky decomposition
// ---------------------------------------------------------------------------

// Right-looking blocked Cholesky on a 2D block-cyclic distribution with
// one-step lookahead: the panel for step K+1 is factored (and its broadcast
// posted) before the bulk of the trailing update of step K is performed, so
// that the panel critical path and the panel communication overlap with the
// rank-nb updates of the other ranks.
static bool choleskyDecompositionDistributed(
    double* Aloc, size_t mloc, size_t nloc, const size_t n, const size_t nb, const Grid& g) {
    const size_t nblk = (n + nb - 1) / nb;
    if(nblk == 0) { return true; }

    std::vector<double> Dbuf(nb * nb + 1);   // diagonal block of the panel
    std::vector<double> Dn(nb * nb);         // panel rows of the look-ahead block
    std::vector<double> PA[2] = {std::vector<double>(mloc * nb + 1), std::vector<double>(mloc * nb + 1)};
    std::vector<double> PB(nloc * nb + 1);
    std::vector<double> Tsend(std::min(mloc, nloc) * nb + 1);
    std::vector<double> Trecv(nloc * nb + 1);
    std::vector<int> counts(g.pr), displs(g.pr);
    std::vector<double> packA, packB;

    // Factor the block column K (which must already be fully updated) and pack
    // the local part of the resulting panel into PA[idx].  Only executed by the
    // ranks of the owning process column.
    auto factorPanel = [&](size_t K, int idx) {
        const size_t kc = K * nb;
        const size_t w = std::min(nb, n - kc);
        const int prow = (int)(K % (size_t)g.pr);
        const size_t rstart0 = numroc(kc, nb, g.pi, g.pr);
        const size_t cstart0 = numroc(kc, nb, g.pj, g.pc);
        const size_t mA = mloc - rstart0;

        double status = -1.0; // >= 0: global index of the failing diagonal element
        if(g.pi == prow) {
            const double* src = Aloc + rstart0 + cstart0 * mloc;
            for(size_t j = 0; j < w; ++j) {
                for(size_t i = 0; i < w; ++i) {
                    Dbuf[i + j * w] = src[i + j * mloc];
                }
            }
            const long fail = factor_diag_block(Dbuf.data(), w, w);
            status = (fail < 0) ? -1.0 : (double)(kc + (size_t)fail);
            Dbuf[w * w] = status;
            double* dst = Aloc + rstart0 + cstart0 * mloc;
            for(size_t j = 0; j < w; ++j) {
                for(size_t i = 0; i < w; ++i) {
                    dst[i + j * mloc] = (i >= j) ? Dbuf[i + j * w] : 0.0;
                }
            }
        }
        MPI_Bcast(Dbuf.data(), (int)(w * w + 1), MPI_DOUBLE, prow, g.col);
        status = Dbuf[w * w];

        if(status < 0.0) {
            const size_t rtrsm = rstart0 + ((g.pi == prow) ? w : 0);
            trsm_right_lower_trans(Aloc + rtrsm + cstart0 * mloc, mloc, mloc - rtrsm, Dbuf.data(), w);
        }

        double* dst = PA[idx].data();
        const double* src = Aloc + rstart0 + cstart0 * mloc;
        for(size_t j = 0; j < w; ++j) {
            std::memcpy(dst + j * mA, src + j * mloc, mA * sizeof(double));
        }
        dst[mA * w] = status;
    };

    // Post the broadcast of panel K along the process rows.
    auto postPanelBcast = [&](size_t K, int idx, MPI_Request* req) {
        const size_t kc = K * nb;
        const size_t w = std::min(nb, n - kc);
        const size_t mA = mloc - numroc(kc, nb, g.pi, g.pr);
        MPI_Ibcast(PA[idx].data(), (int)(mA * w + 1), MPI_DOUBLE, (int)(K % (size_t)g.pc), g.row, req);
    };

    int cur = 0;
    MPI_Request reqPA[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};

    if(g.pj == 0) { factorPanel(0, cur); }
    postPanelBcast(0, cur, &reqPA[cur]);

    for(size_t K = 0; K < nblk; ++K) {
        const size_t kc = K * nb;
        const size_t w = std::min(nb, n - kc);
        const size_t rstart0 = numroc(kc, nb, g.pi, g.pr);
        const size_t cstart0 = numroc(kc, nb, g.pj, g.pc);
        const size_t mA = mloc - rstart0;
        const size_t mB = nloc - cstart0;
        const double* panel = PA[cur].data();

        MPI_Wait(&reqPA[cur], MPI_STATUS_IGNORE);
        const double status = panel[mA * w];
        if(status >= 0.0) {
            if(g.rank == 0) {
                printf("Error: Matrix is not positive definite at diagonal element %zu\n", (size_t)status);
            }
            MPI_Wait(&reqPA[1 - cur], MPI_STATUS_IGNORE);
            return false;
        }

        // --- start the panel transpose: every rank needs the panel rows that
        //     correspond to its local columns ---
        size_t off = 0;
        for(size_t I = K; I < nblk; ++I) {
            if((int)(I % (size_t)g.pr) != g.pi || (int)(I % (size_t)g.pc) != g.pj) { continue; }
            const size_t bw = std::min(nb, n - I * nb);
            const size_t rloc = (I / (size_t)g.pr) * nb - rstart0;
            for(size_t j = 0; j < w; ++j) {
                std::memcpy(Tsend.data() + off + j * bw, panel + rloc + j * mA, bw * sizeof(double));
            }
            off += bw * w;
        }
        int total = 0;
        for(int src = 0; src < g.pr; ++src) {
            size_t rows = 0;
            for(size_t I = K; I < nblk; ++I) {
                if((int)(I % (size_t)g.pr) != src || (int)(I % (size_t)g.pc) != g.pj) { continue; }
                rows += std::min(nb, n - I * nb);
            }
            counts[src] = (int)(rows * w);
            displs[src] = total;
            total += counts[src];
        }
        MPI_Request reqT;
        MPI_Iallgatherv(Tsend.data(), counts[g.pi], MPI_DOUBLE, Trecv.data(), counts.data(), displs.data(),
            MPI_DOUBLE, g.colT, &reqT);

        // --- lookahead: update and factor the next panel ---
        const int nxt = 1 - cur;
        if(K + 1 < nblk) {
            const size_t KN = K + 1;
            const int pcolN = (int)(KN % (size_t)g.pc);
            const int prowN = (int)(KN % (size_t)g.pr);
            if(g.pj == pcolN) {
                const size_t bwN = std::min(nb, n - KN * nb);
                if(g.pi == prowN) {
                    const size_t rloc = (KN / (size_t)g.pr) * nb - rstart0;
                    for(size_t j = 0; j < w; ++j) {
                        std::memcpy(Dn.data() + j * bwN, panel + rloc + j * mA, bwN * sizeof(double));
                    }
                }
                MPI_Bcast(Dn.data(), (int)(bwN * w), MPI_DOUBLE, prowN, g.col);

                const size_t cloc = (KN / (size_t)g.pc) * nb;
                const size_t rs = numroc(KN * nb, nb, g.pi, g.pr);
                if(rs < mloc) {
                    gemm_nt_sub(mloc - rs, bwN, w, panel + (rs - rstart0), mA, Dn.data(), bwN,
                        Aloc + rs + cloc * mloc, mloc, packA, packB);
                }
                factorPanel(KN, nxt);
            }
            postPanelBcast(KN, nxt, &reqPA[nxt]);
        }

        // --- unpack the transposed panel ---
        MPI_Wait(&reqT, MPI_STATUS_IGNORE);
        for(int src = 0; src < g.pr; ++src) {
            size_t off2 = (size_t)displs[src];
            for(size_t I = K; I < nblk; ++I) {
                if((int)(I % (size_t)g.pr) != src || (int)(I % (size_t)g.pc) != g.pj) { continue; }
                const size_t bw = std::min(nb, n - I * nb);
                const size_t cloc = (I / (size_t)g.pc) * nb - cstart0;
                for(size_t j = 0; j < w; ++j) {
                    std::memcpy(PB.data() + cloc + j * mB, Trecv.data() + off2 + j * bw, bw * sizeof(double));
                }
                off2 += bw * w;
            }
        }

        // --- trailing submatrix update, one local block column at a time ---
        const size_t nlblk = (nloc + nb - 1) / nb;
        for(size_t lb = 0; lb < nlblk; ++lb) {
            const size_t J = lb * (size_t)g.pc + (size_t)g.pj;
            if(J <= K + 1 || J >= nblk) { continue; } // block K + 1 was done by the lookahead
            const size_t cloc = lb * nb;
            const size_t bw = std::min(nb, nloc - cloc);
            const size_t rs = numroc(J * nb, nb, g.pi, g.pr);
            if(rs >= mloc) { continue; }
            gemm_nt_sub(mloc - rs, bw, w, panel + (rs - rstart0), mA, PB.data() + (cloc - cstart0), mB,
                Aloc + rs + cloc * mloc, mloc, packA, packB);
        }

        cur = nxt;
    }

    return true;
}

// ---------------------------------------------------------------------------
// Matrix generation / validation
// ---------------------------------------------------------------------------

// Fill the shared random matrix B (row-major, n x n) -- done once per node.
static void generateRandomMatrix(double* B, const size_t n) {
    unsigned int seed = 42;
    for(size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }
}

// A = B * B^T with n added to the diagonal; each rank fills its local part.
static void generateLocalMatrix(
    double* Aloc, size_t mloc, size_t nloc, const double* B, const size_t n, const size_t nb, const Grid& g) {
    // Four columns at a time: the row of B belonging to the current row index is
    // then read once for four output elements.
    for(size_t jl0 = 0; jl0 < nloc; jl0 += 4) {
        const size_t nt = std::min<size_t>(4, nloc - jl0);
        size_t jg[4];
        const double* Bj[4];
        for(size_t t = 0; t < 4; ++t) {
            const size_t jl = jl0 + std::min(t, nt - 1);
            const size_t J = (jl / nb) * (size_t)g.pc + (size_t)g.pj;
            jg[t] = J * nb + jl % nb;
            Bj[t] = B + jg[t] * n;
        }
        for(size_t il = 0; il < mloc; ++il) {
            const size_t I = (il / nb) * (size_t)g.pr + (size_t)g.pi;
            const size_t i = I * nb + il % nb;
            const double* Bi = B + i * n;
            double vals[4];
            if(nt == 4 && i >= jg[3]) {
                dot_vec4(Bi, Bj[0], Bj[1], Bj[2], Bj[3], n, vals);
            } else {
                for(size_t t = 0; t < nt; ++t) {
                    vals[t] = (i < jg[t]) ? 0.0 : dot_vec(Bi, Bj[t], n);
                }
            }
            for(size_t t = 0; t < nt; ++t) {
                double val = (i < jg[t]) ? 0.0 : vals[t];
                if(i == jg[t]) { val += (double)n; }
                Aloc[il + (jl0 + t) * mloc] = val;
            }
        }
    }
}

// Validate L * L^T against the original matrix (recomputed from B).
static bool validateCholesky(const double* L, const double* B, const size_t n, const Grid& g) {
    double maxError = 0.0;
    double relError = 0.0;

    // Both L*L^T and the original matrix are symmetric (and the corresponding
    // dot products are computed in the same order for (i, j) and (j, i)), so
    // scanning the lower triangle yields exactly the same maxima as scanning
    // the full matrix.  Rows are handed out cyclically for load balance.
    for(size_t i = (size_t)g.rank; i < n; i += (size_t)g.size) {
        const double* Li = L + i * n;
        const double* Bi = B + i * n;
        for(size_t j0 = 0; j0 <= i; j0 += 4) {
            const size_t nt = std::min<size_t>(4, i + 1 - j0);
            double orig[4];
            if(nt == 4) {
                dot_vec4(Bi, B + j0 * n, B + (j0 + 1) * n, B + (j0 + 2) * n, B + (j0 + 3) * n, n, orig);
            } else {
                for(size_t t = 0; t < nt; ++t) {
                    orig[t] = dot_vec(Bi, B + (j0 + t) * n, n);
                }
            }
            for(size_t t = 0; t < nt; ++t) {
                const size_t j = j0 + t;
                // L is lower triangular, so the dot product ends at min(i, j) = j.
                const double sum = dot_vec(Li, L + j * n, j + 1);
                if(i == j) { orig[t] += (double)n; }

                const double error = fabs(sum - orig[t]);
                maxError = std::max(maxError, error);
                const double rel = error / (fabs(orig[t]) + 1e-10);
                relError = std::max(relError, rel);
            }
        }
    }

    double errors[2] = {maxError, relError};
    MPI_Allreduce(MPI_IN_PLACE, errors, 2, MPI_DOUBLE, MPI_MAX, g.world);
    maxError = errors[0];
    relError = errors[1];

    if(g.rank == 0) {
        printf("Max absolute error: %.10e\n", maxError);
        printf("Max relative error: %.10e\n", relError);
    }

    // Check if error is within tolerance
    if(relError > 1e-6) {
        if(g.rank == 0) { printf("Validation failed: relative error too large\n"); }
        return false;
    }

    return true;
}

// Collect the distributed factor into a full row-major n x n matrix on rank 0.
static void gatherFullMatrix(
    const double* Aloc, size_t mloc, size_t nloc, double* Lfull, const size_t n, const size_t nb, const Grid& g) {
    if(g.rank != 0) {
        MPI_Send(Aloc, (int)(mloc * nloc), MPI_DOUBLE, 0, 0, g.world);
        return;
    }

    std::fill(Lfull, Lfull + n * n, 0.0);
    std::vector<double> buf;
    for(int src = 0; src < g.size; ++src) {
        const int spi = src / g.pc;
        const int spj = src % g.pc;
        const size_t sm = numroc(n, nb, spi, g.pr);
        const size_t sn = numroc(n, nb, spj, g.pc);
        const double* data = Aloc;
        if(src != 0) {
            buf.resize(std::max<size_t>(1, sm * sn));
            MPI_Recv(buf.data(), (int)(sm * sn), MPI_DOUBLE, src, 0, g.world, MPI_STATUS_IGNORE);
            data = buf.data();
        }
        for(size_t jl = 0; jl < sn; ++jl) {
            const size_t j = ((jl / nb) * (size_t)g.pc + (size_t)spj) * nb + jl % nb;
            for(size_t il = 0; il < sm; ++il) {
                const size_t i = ((il / nb) * (size_t)g.pr + (size_t)spi) * nb + il % nb;
                if(i >= j) { Lfull[i * n + j] = data[il + jl * sm]; }
            }
        }
    }
}

// Broadcast a large buffer in chunks (counts may exceed INT_MAX).
static void bcastLarge(double* data, size_t count, int root, MPI_Comm comm) {
    const size_t maxChunk = 1u << 27;
    for(size_t off = 0; off < count; off += maxChunk) {
        const size_t c = std::min(maxChunk, count - off);
        MPI_Bcast(data + off, (int)c, MPI_DOUBLE, root, comm);
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

    Grid g;
    g.world = MPI_COMM_WORLD;
    MPI_Comm_rank(g.world, &g.rank);
    MPI_Comm_size(g.world, &g.size);

    size_t n = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
    for(int i = 1; i < argc; ++i) {
        if(strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = atoi(argv[++i]);
        } else if(strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if(strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if(strcmp(argv[i], "-h") == 0) {
            if(g.rank == 0) { printUsage(argv[0]); }
            MPI_Finalize();
            return 0;
        } else {
            if(g.rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if(g.rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    if(n == 0) {
        if(g.rank == 0) {
            printf("Generating positive definite matrix...\n");
            printf("Computing Cholesky decomposition...\n");
            printf("Computation time: 0 ms\n");
            printf("Performance: %.3f GFLOPS\n", 0.0 / 0.0);
            if(printResults) {
                std::vector<double> empty;
                print_results(empty, "CholeskyL");
            }
            if(validate) {
                printf("Validating result...\n");
                printf("Max absolute error: %.10e\n", 0.0);
                printf("Max relative error: %.10e\n", 0.0);
                printf("Validation: PASSED\n");
            }
        }
        MPI_Finalize();
        return 0;
    }

    // --- build the process grid (as square as possible) ---
    g.pr = 1;
    for(int p = (int)sqrt((double)g.size); p >= 1; --p) {
        if(g.size % p == 0) {
            g.pr = p;
            break;
        }
    }
    g.pc = g.size / g.pr;
    g.pi = g.rank / g.pc;
    g.pj = g.rank % g.pc;
    MPI_Comm_split(g.world, g.pi, g.pj, &g.row);
    MPI_Comm_split(g.world, g.pj, g.pi, &g.col);
    MPI_Comm_dup(g.col, &g.colT);

    // Block size: small enough to keep the 2D distribution well balanced (and
    // the panel critical path short), large enough for an efficient rank-nb
    // update.  48 is a good compromise over a wide range of sizes and rank
    // counts; for small matrices the distribution granularity dominates.
    size_t nb = 48;
    const size_t gridMax = (size_t)std::max(g.pr, g.pc);
    const size_t cap = n / (2 * gridMax);
    if(cap < nb) { nb = std::max<size_t>(8, (cap / 8) * 8); }
    if(nb > n) { nb = n; }

    const size_t mloc = numroc(n, nb, g.pi, g.pr);
    const size_t nloc = numroc(n, nb, g.pj, g.pc);
    std::vector<double> Aloc(std::max<size_t>(1, mloc * nloc));

    // --- shared (per node) copy of the random matrix B ---
    MPI_Comm shmComm;
    MPI_Comm_split_type(g.world, MPI_COMM_TYPE_SHARED, g.rank, MPI_INFO_NULL, &shmComm);
    int shmRank = 0;
    MPI_Comm_rank(shmComm, &shmRank);
    MPI_Comm leaderComm;
    MPI_Comm_split(g.world, shmRank == 0 ? 0 : MPI_UNDEFINED, g.rank, &leaderComm);

    MPI_Win winB;
    double* Bwin = nullptr;
    {
        double* local = nullptr;
        const MPI_Aint bytes = (shmRank == 0) ? (MPI_Aint)(n * n * sizeof(double)) : 0;
        MPI_Win_allocate_shared(bytes, sizeof(double), MPI_INFO_NULL, shmComm, &local, &winB);
        MPI_Aint sz = 0;
        int disp = 0;
        MPI_Win_shared_query(winB, 0, &sz, &disp, &Bwin);
        MPI_Win_lock_all(MPI_MODE_NOCHECK, winB);
    }

    // Generate positive definite matrix
    if(g.rank == 0) { printf("Generating positive definite matrix...\n"); }
    if(shmRank == 0) { generateRandomMatrix(Bwin, n); }
    MPI_Win_sync(winB);
    MPI_Barrier(shmComm);
    MPI_Win_sync(winB);

    generateLocalMatrix(Aloc.data(), mloc, nloc, Bwin, n, nb, g);

    // Perform Cholesky decomposition
    if(g.rank == 0) { printf("Computing Cholesky decomposition...\n"); }
    MPI_Barrier(g.world);
    auto start = std::chrono::high_resolution_clock::now();

    bool success = choleskyDecompositionDistributed(Aloc.data(), mloc, nloc, n, nb, g);

    MPI_Barrier(g.world);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if(!success) {
        if(g.rank == 0) { printf("Cholesky decomposition failed\n"); }
        MPI_Win_unlock_all(winB);
        MPI_Win_free(&winB);
        MPI_Finalize();
        return 1;
    }

    if(g.rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    int exitCode = 0;
    if(printResults || validate) {
        // Assemble the full factor, replicated once per node in shared memory.
        MPI_Win winL;
        double* Lwin = nullptr;
        double* local = nullptr;
        const MPI_Aint bytes = (shmRank == 0) ? (MPI_Aint)(n * n * sizeof(double)) : 0;
        MPI_Win_allocate_shared(bytes, sizeof(double), MPI_INFO_NULL, shmComm, &local, &winL);
        MPI_Aint sz = 0;
        int disp = 0;
        MPI_Win_shared_query(winL, 0, &sz, &disp, &Lwin);
        MPI_Win_lock_all(MPI_MODE_NOCHECK, winL);

        gatherFullMatrix(Aloc.data(), mloc, nloc, Lwin, n, nb, g);

        if(printResults && g.rank == 0) {
            std::vector<double> Lvec(Lwin, Lwin + n * n);
            print_results(Lvec, "CholeskyL");
        }

        if(validate) {
            if(leaderComm != MPI_COMM_NULL) { bcastLarge(Lwin, n * n, 0, leaderComm); }
            MPI_Win_sync(winL);
            MPI_Barrier(shmComm);
            MPI_Win_sync(winL);

            if(g.rank == 0) { printf("Validating result...\n"); }
            const bool valid = validateCholesky(Lwin, Bwin, n, g);
            if(g.rank == 0) { printf("Validation: %s\n", valid ? "PASSED" : "FAILED"); }
            exitCode = valid ? 0 : 1;
        }

        MPI_Win_unlock_all(winL);
        MPI_Win_free(&winL);
    }

    MPI_Win_unlock_all(winB);
    MPI_Win_free(&winB);
    if(leaderComm != MPI_COMM_NULL) { MPI_Comm_free(&leaderComm); }
    MPI_Comm_free(&shmComm);
    MPI_Comm_free(&g.row);
    MPI_Comm_free(&g.col);
    MPI_Comm_free(&g.colT);
    MPI_Finalize();
    return exitCode;
}
