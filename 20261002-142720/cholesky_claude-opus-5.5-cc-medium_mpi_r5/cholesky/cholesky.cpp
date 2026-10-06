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

// Distributed-memory (MPI) blocked Cholesky decomposition.
// Decomposes positive definite matrix A into L * L^T where L is lower triangular.
//
// The matrix is distributed in a 2D block-cyclic layout over a Pr x Pc process
// grid. The factorization is right-looking with one step of lookahead: the panel
// of step K+1 is factored and broadcast while the trailing update of step K runs.

namespace {

constexpr int MR = 6;   // micro-kernel rows
constexpr int NR = 8;   // micro-kernel columns (2 AVX2 vectors)

inline size_t cntUpTo(long K, int p, int Q) {
    // number of blocks b in [0, K] with b % Q == p
    return K >= p ? (size_t)((K - p) / Q + 1) : 0;
}

struct Grid {
    int P = 1, rank = 0, Pr = 1, Pc = 1, pr = 0, pc = 0;
    size_t n = 0, nb = 1, nblk = 0;
    size_t mloc = 0, ncol = 0;  // local rows / cols
    size_t nlbr = 0, nlbc = 0;  // local row / col blocks
    MPI_Comm rowComm = MPI_COMM_NULL;  // same pr, size Pc, rank = pc
    MPI_Comm colComm = MPI_COMM_NULL;  // same pc, size Pr, rank = pr

    size_t bs(size_t I) const { return std::min(nb, n - I * nb); }
    size_t rowStart(long K) const { return std::min(cntUpTo(K, pr, Pr) * nb, mloc); }
    size_t colStart(long K) const { return std::min(cntUpTo(K, pc, Pc) * nb, ncol); }
    size_t colEnd(size_t I) const { return colStart((long)I); }

    static size_t localCount(size_t n, size_t nb, size_t nblk, int p, int Q) {
        size_t cnt = cntUpTo((long)nblk - 1, p, Q) * nb;
        if (nblk > 0 && (int)((nblk - 1) % Q) == p) cnt -= nb - (n - (nblk - 1) * nb);
        return cnt;
    }
    // local sizes of an arbitrary grid coordinate
    size_t mlocOf(int p) const { return localCount(n, nb, nblk, p, Pr); }
    size_t ncolOf(int q) const { return localCount(n, nb, nblk, q, Pc); }
    size_t globalRow(size_t lr, int p) const { return ((lr / nb) * Pr + p) * nb + lr % nb; }
    size_t globalCol(size_t lc, int q) const { return ((lc / nb) * Pc + q) * nb + lc % nb; }

    void init(size_t n_, int P_, int rank_) {
        n = n_;
        P = P_;
        rank = rank_;
        Pr = 1;
        for (int d = 1; (long)d * d <= P; ++d)
            if (P % d == 0) Pr = d;
        Pc = P / Pr;
        pr = rank / Pc;
        pc = rank % Pc;

        // Block size: multiple of NR, small enough for load balance, large enough for efficiency
        const size_t maxQ = (size_t)std::max(Pr, Pc);
        size_t target = n / (10 * maxQ);
        target = (target / NR) * NR;
        nb = std::clamp<size_t>(target, 3 * NR, 120);
        nblk = n == 0 ? 0 : (n + nb - 1) / nb;

        mloc = mlocOf(pr);
        ncol = ncolOf(pc);
        nlbr = cntUpTo((long)nblk - 1, pr, Pr);
        nlbc = cntUpTo((long)nblk - 1, pc, Pc);

        MPI_Comm_split(MPI_COMM_WORLD, pr, pc, &rowComm);
        MPI_Comm_split(MPI_COMM_WORLD, pc, pr, &colComm);
    }

    ~Grid() {
        if (rowComm != MPI_COMM_NULL) MPI_Comm_free(&rowComm);
        if (colComm != MPI_COMM_NULL) MPI_Comm_free(&colComm);
    }
};

// C[M x NR] -= A[M x kc] * B[kc x NR]; B packed with NR contiguous per k
template <int M>
inline void kernel(size_t kc, const double* a, size_t lda, const double* b, double* c, size_t ldc) {
    __m256d x[M][2];
#pragma GCC unroll 6
    for (int r = 0; r < M; ++r) {
        x[r][0] = _mm256_loadu_pd(c + r * ldc);
        x[r][1] = _mm256_loadu_pd(c + r * ldc + 4);
    }
    for (size_t k = 0; k < kc; ++k) {
        const __m256d b0 = _mm256_loadu_pd(b + k * NR);
        const __m256d b1 = _mm256_loadu_pd(b + k * NR + 4);
#pragma GCC unroll 6
        for (int r = 0; r < M; ++r) {
            const __m256d ar = _mm256_broadcast_sd(a + r * lda + k);
            x[r][0] = _mm256_fnmadd_pd(ar, b0, x[r][0]);
            x[r][1] = _mm256_fnmadd_pd(ar, b1, x[r][1]);
        }
    }
#pragma GCC unroll 6
    for (int r = 0; r < M; ++r) {
        _mm256_storeu_pd(c + r * ldc, x[r][0]);
        _mm256_storeu_pd(c + r * ldc + 4, x[r][1]);
    }
}

inline void kernelAny(int mr, int nc, size_t kc, const double* a, size_t lda, const double* b, double* c,
                      size_t ldc) {
    double tmp[MR * NR];
    double* cc = c;
    size_t ldcc = ldc;
    if (nc < NR) {
        for (int r = 0; r < mr; ++r)
            for (int x = 0; x < NR; ++x) tmp[r * NR + x] = x < nc ? c[r * ldc + x] : 0.0;
        cc = tmp;
        ldcc = NR;
    }
    switch (mr) {
        case 6: kernel<6>(kc, a, lda, b, cc, ldcc); break;
        case 5: kernel<5>(kc, a, lda, b, cc, ldcc); break;
        case 4: kernel<4>(kc, a, lda, b, cc, ldcc); break;
        case 3: kernel<3>(kc, a, lda, b, cc, ldcc); break;
        case 2: kernel<2>(kc, a, lda, b, cc, ldcc); break;
        default: kernel<1>(kc, a, lda, b, cc, ldcc); break;
    }
    if (nc < NR) {
        for (int r = 0; r < mr; ++r)
            for (int x = 0; x < nc; ++x) c[r * ldc + x] = tmp[r * NR + x];
    }
}

struct Progress {
    virtual void poll() = 0;
};

// Lower-triangular (block-wise) update of the local matrix C (ld = g.ncol):
// for local row blocks lb >= lb0 (global block I), local cols [colA, min(colB, colEnd(I))):
//   C[r][c] -= sum_k A[r - aRow0][k] * Pt(c)[k]
// Pt holds NR-wide tiles of packed columns starting at local column ptBase.
void gemmLower(const Grid& g, double* C, size_t lb0, size_t colA, size_t colB, const double* A, size_t aRow0,
               size_t lda, const double* Pt, size_t ptBase, size_t kc, Progress* prog) {
    if (colA >= colB || kc == 0) return;
    const size_t ldc = g.ncol;
    constexpr size_t NC = 24 * NR;
    for (size_t cc = colA; cc < colB; cc += NC) {
        const size_t ccEnd = std::min(cc + NC, colB);
        for (size_t lb = lb0; lb < g.nlbr; ++lb) {
            const size_t I = lb * g.Pr + g.pr;
            const size_t r0 = lb * g.nb, r1 = r0 + g.bs(I);
            const size_t cEnd = std::min(ccEnd, g.colEnd(I));
            if (cEnd <= cc) continue;
            for (size_t r = r0; r < r1; r += MR) {
                const int mr = (int)std::min<size_t>(MR, r1 - r);
                const double* a = A + (r - aRow0) * lda;
                for (size_t c = cc; c < cEnd; c += NR) {
                    if (c + NR < cEnd)
                        for (int q = 0; q < mr; ++q) {
                            _mm_prefetch((const char*)(C + (r + q) * ldc + c + NR), _MM_HINT_T0);
                            _mm_prefetch((const char*)(C + (r + q) * ldc + c + 2 * NR - 1), _MM_HINT_T0);
                        }
                    const int nc = (int)std::min<size_t>(NR, cEnd - c);
                    kernelAny(mr, nc, kc, a, lda, Pt + ((c - ptBase) / NR) * kc * NR, C + r * ldc + c, ldc);
                }
            }
            if (prog) prog->poll();
        }
    }
}

// Pack column source rows into NR-wide tiles: Pt[tile][k][NR]
inline void packTile(double* Pt, size_t kc, size_t col, const double* src) {
    double* t = Pt + (col / NR) * kc * NR + col % NR;
    for (size_t k = 0; k < kc; ++k) t[k * NR] = src[k];
}

// Unblocked Cholesky of a w x w diagonal block (in place). Returns failing index or -1.
long factorDiag(double* a, size_t ld, size_t w, std::vector<double>& col) {
    col.resize(w);
    for (size_t j = 0; j < w; ++j) {
        const double val = a[j * ld + j];
        if (val <= 0.0) return (long)j;
        const double d = sqrt(val);
        a[j * ld + j] = d;
        for (size_t i = j + 1; i < w; ++i) {
            a[i * ld + j] /= d;
            col[i] = a[i * ld + j];
        }
        for (size_t i = j + 1; i < w; ++i) {
            const double lij = col[i];
            double* ai = a + i * ld;
            for (size_t k = j + 1; k <= i; ++k) ai[k] -= lij * col[k];
        }
    }
    return -1;
}

struct Factorizer : Progress {
    Grid& g;
    std::vector<double>& A;  // local mloc x ncol
    std::vector<double> rowP[2], Pt[2], Dbuf, Dt, colTmp, sendBuf, recvBuf;
    std::vector<int> counts, displs;

    Factorizer(Grid& g_, std::vector<double>& A_) : g(g_), A(A_) {}

    size_t rowPSize(size_t K) const { return (g.mloc - g.rowStart((long)K)) * g.bs(K) + 1; }

    // Factor diagonal block K, broadcast it down the process column, and solve the panel.
    // Only called on ranks with pc == K % Pc. Fills buf with local panel rows of blocks > K.
    void panelCompute(size_t K, std::vector<double>& buf) {
        const size_t w = g.bs(K), ld = g.ncol;
        const size_t lc0 = (K / g.Pc) * g.nb;
        Dbuf.assign(w * w + 1, 0.0);
        if (g.pr == (int)(K % g.Pr)) {
            double* d = A.data() + (K / g.Pr) * g.nb * ld + lc0;
            const long fail = factorDiag(d, ld, w, colTmp);
            for (size_t i = 0; i < w; ++i)
                for (size_t j = 0; j <= i; ++j) Dbuf[i * w + j] = d[i * ld + j];
            Dbuf[w * w] = fail >= 0 ? (double)(K * g.nb + fail + 1) : 0.0;
        }
        MPI_Bcast(Dbuf.data(), (int)(w * w + 1), MPI_DOUBLE, (int)(K % g.Pr), g.colComm);
        const double status = Dbuf[w * w];

        const size_t rs = g.rowStart((long)K);
        buf.resize(rowPSize(K));
        if (status == 0.0) {
            // Triangular solve X * D^T = B for local rows of blocks > K, vectorized across rows
            constexpr size_t RB = 32;
            Dt.assign(w * RB, 0.0);
            for (size_t r0 = rs; r0 < g.mloc; r0 += RB) {
                const size_t nr = std::min(RB, g.mloc - r0);
                double* xb = A.data() + r0 * ld + lc0;
                for (size_t r = 0; r < nr; ++r)
                    for (size_t j = 0; j < w; ++j) Dt[j * RB + r] = xb[r * ld + j];
                for (size_t jj = 0; jj < w; ++jj) {
                    double* xjj = Dt.data() + jj * RB;
                    __m256d acc[RB / 4];
#pragma GCC unroll 8
                    for (size_t v = 0; v < RB / 4; ++v) acc[v] = _mm256_loadu_pd(xjj + 4 * v);
                    const double* drow = Dbuf.data() + jj * w;
                    for (size_t j = 0; j < jj; ++j) {
                        const __m256d d = _mm256_broadcast_sd(drow + j);
                        const double* xj = Dt.data() + j * RB;
#pragma GCC unroll 8
                        for (size_t v = 0; v < RB / 4; ++v)
                            acc[v] = _mm256_fnmadd_pd(d, _mm256_loadu_pd(xj + 4 * v), acc[v]);
                    }
                    const __m256d djj = _mm256_set1_pd(drow[jj]);
#pragma GCC unroll 8
                    for (size_t v = 0; v < RB / 4; ++v) _mm256_storeu_pd(xjj + 4 * v, _mm256_div_pd(acc[v], djj));
                }
                for (size_t r = 0; r < nr; ++r) {
                    double* dst = buf.data() + (r0 + r - rs) * w;
                    for (size_t j = 0; j < w; ++j) xb[r * ld + j] = dst[j] = Dt[j * RB + r];
                }
            }
        }
        buf.back() = status;
    }

    void startPanel(size_t K, std::vector<double>& buf, MPI_Request* req) {
        if (g.pc == (int)(K % g.Pc))
            panelCompute(K, buf);
        else
            buf.resize(rowPSize(K));
        MPI_Ibcast(buf.data(), (int)buf.size(), MPI_DOUBLE, (int)(K % g.Pc), g.rowComm, req);
    }

    // Gather the column-panel rows (blocks J > K, J % Pc == pc) within the process column.
    void colGatherStart(size_t K, const std::vector<double>& rp, MPI_Request* req) {
        const size_t w = g.bs(K);
        const size_t rs = g.rowStart((long)K);
        counts.assign(g.Pr, 0);
        displs.assign(g.Pr, 0);
        sendBuf.clear();
        for (size_t J = firstColBlock(K); J < g.nblk; J += g.Pc) {
            const int q = (int)(J % g.Pr);
            counts[q] += (int)(g.bs(J) * w);
            if (q == g.pr) {
                const double* src = rp.data() + ((J / g.Pr) * g.nb - rs) * w;
                sendBuf.insert(sendBuf.end(), src, src + g.bs(J) * w);
            }
        }
        int total = 0;
        for (int q = 0; q < g.Pr; ++q) {
            displs[q] = total;
            total += counts[q];
        }
        recvBuf.resize(std::max(total, 1));
        MPI_Iallgatherv(sendBuf.data(), counts[g.pr], MPI_DOUBLE, recvBuf.data(), counts.data(), displs.data(),
                        MPI_DOUBLE, g.colComm, req);
    }

    // Pack the gathered column panel into NR tiles for the update kernel.
    void colGatherFinish(size_t K, std::vector<double>& pt) {
        const size_t w = g.bs(K);
        const size_t base = g.colStart((long)K);
        const size_t nc = g.ncol - base;
        const size_t ntiles = (nc + NR - 1) / NR;
        pt.assign(std::max<size_t>(ntiles * w * NR, 1), 0.0);
        std::vector<int> off(displs);
        for (size_t J = firstColBlock(K); J < g.nblk; J += g.Pc) {
            const int q = (int)(J % g.Pr);
            const double* src = recvBuf.data() + off[q];
            const size_t b = g.bs(J);
            off[q] += (int)(b * w);
            const size_t lcb = (J / g.Pc) * g.nb - base;
            for (size_t r = 0; r < b; ++r) packTile(pt.data(), w, lcb + r, src + r * w);
        }
    }

    size_t firstColBlock(size_t K) const {
        size_t J = K + 1;
        while (J < g.nblk && (int)(J % g.Pc) != g.pc) ++J;
        return J;
    }

    // Communication state of the next panel: row broadcast -> column allgather -> done
    enum class Comm { RowBcast, ColGather, Done };
    Comm state = Comm::Done;
    MPI_Request req = MPI_REQUEST_NULL;
    size_t commK = 0;
    std::vector<double>* commRowP = nullptr;
    long commFail = -1;

    void advance() {
        if (state == Comm::RowBcast) {
            if (commRowP->back() != 0.0) {
                commFail = (long)commRowP->back() - 1;
                state = Comm::Done;
            } else {
                colGatherStart(commK, *commRowP, &req);
                state = Comm::ColGather;
            }
        } else if (state == Comm::ColGather) {
            state = Comm::Done;
        }
    }

    void poll() override {
        while (state != Comm::Done) {
            int flag = 0;
            MPI_Test(&req, &flag, MPI_STATUS_IGNORE);
            if (!flag) return;
            advance();
        }
    }

    void finishComm() {
        while (state != Comm::Done) {
            MPI_Wait(&req, MPI_STATUS_IGNORE);
            advance();
        }
    }

    void beginComm(size_t K, std::vector<double>& rp) {
        commK = K;
        commRowP = &rp;
        commFail = -1;
        startPanel(K, rp, &req);
        state = Comm::RowBcast;
    }

    // Returns -1 on success, or the global index of the failing diagonal element.
    long run() {
        long result = -1;
        if (g.nblk == 0) return result;
        beginComm(0, rowP[0]);
        finishComm();
        if (commFail >= 0) return commFail;
        colGatherFinish(0, Pt[0]);

        for (size_t K = 0; K < g.nblk; ++K) {
            const int cur = K & 1, nxt = cur ^ 1;
            const bool more = K + 1 < g.nblk;
            const size_t w = g.bs(K);
            const size_t lb0 = cntUpTo((long)K, g.pr, g.Pr);
            const size_t rs = g.rowStart((long)K);
            const size_t cs = g.colStart((long)K);
            const size_t cs1 = more ? g.colStart((long)K + 1) : g.ncol;

            // Lookahead: update column block K+1 first, then start panel K+1
            gemmLower(g, A.data(), lb0, cs, cs1, rowP[cur].data(), rs, w, Pt[cur].data(), cs, w, nullptr);
            if (more) beginComm(K + 1, rowP[nxt]);

            // Remaining trailing update overlapped with the panel communication
            gemmLower(g, A.data(), lb0, cs1, g.ncol, rowP[cur].data(), rs, w, Pt[cur].data(), cs, w, this);

            if (more) {
                finishComm();
                if (commFail >= 0) {
                    result = commFail;
                    break;
                }
                colGatherFinish(K + 1, Pt[nxt]);
            }
        }

        // Zero out upper triangular part
        for (size_t r = 0; r < g.mloc; ++r) {
            const size_t gi = g.globalRow(r, g.pr);
            double* row = A.data() + r * g.ncol;
            for (size_t lb = 0; lb < g.nlbc; ++lb) {
                const size_t J = lb * g.Pc + g.pc;
                if (J * g.nb + g.bs(J) <= gi + 1) continue;
                for (size_t c = 0; c < g.bs(J); ++c)
                    if (J * g.nb + c > gi) row[lb * g.nb + c] = 0.0;
            }
        }
        return result;
    }
};

// Generate the local part of a symmetric positive definite matrix A = B * B^T + n*I
void generatePositiveDefiniteMatrix(const Grid& g, std::vector<double>& A) {
    const size_t n = g.n;
    A.assign(g.mloc * g.ncol, 0.0);
    if (n == 0) return;

    // Generate random matrix B (same sequence as the sequential version); keep needed rows
    std::vector<double> Brow(g.mloc * n), Bcol(g.ncol * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < n; ++i) {
        const size_t I = i / g.nb;
        double* dr = (int)(I % g.Pr) == g.pr ? Brow.data() + ((I / g.Pr) * g.nb + i % g.nb) * n : nullptr;
        double* dc = (int)(I % g.Pc) == g.pc ? Bcol.data() + ((I / g.Pc) * g.nb + i % g.nb) * n : nullptr;
        for (size_t k = 0; k < n; ++k) {
            const double v = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
            if (dr) dr[k] = v;
            if (dc) dc[k] = v;
        }
    }

    // A = -( -B * B^T ) on lower blocks
    constexpr size_t KC = 256;
    const size_t ntiles = (g.ncol + NR - 1) / NR;
    std::vector<double> Pt;
    for (size_t kk = 0; kk < n; kk += KC) {
        const size_t kc = std::min(KC, n - kk);
        Pt.assign(std::max<size_t>(ntiles * kc * NR, 1), 0.0);
        for (size_t c = 0; c < g.ncol; ++c) packTile(Pt.data(), kc, c, Bcol.data() + c * n + kk);
        gemmLower(g, A.data(), 0, 0, g.ncol, Brow.data() + kk, 0, n, Pt.data(), 0, kc, nullptr);
    }
    for (double& v : A) v = -v;

    // Add diagonal dominance to ensure positive definiteness
    for (size_t r = 0; r < g.mloc; ++r) {
        const size_t gi = g.globalRow(r, g.pr);
        const size_t J = gi / g.nb;
        if ((int)(J % g.Pc) == g.pc) A[r * g.ncol + (J / g.Pc) * g.nb + gi % g.nb] += n;
    }
}

// Gather the distributed matrix into a full n x n row-major matrix on rank 0
void gatherMatrix(const Grid& g, const std::vector<double>& Aloc, std::vector<double>& full) {
    const size_t n = g.n;
    if (g.rank == 0) {
        full.assign(n * n, 0.0);
        std::vector<double> buf;
        for (int p = 0; p < g.P; ++p) {
            const int ppr = p / g.Pc, ppc = p % g.Pc;
            const size_t pm = g.mlocOf(ppr), pn = g.ncolOf(ppc);
            for (size_t r0 = 0; r0 < pm; r0 += g.nb) {
                const size_t rows = std::min(g.nb, pm - r0);
                const double* src;
                if (p == 0) {
                    src = Aloc.data() + r0 * pn;
                } else {
                    buf.resize(rows * pn);
                    MPI_Recv(buf.data(), (int)(rows * pn), MPI_DOUBLE, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    src = buf.data();
                }
                for (size_t r = 0; r < rows; ++r) {
                    double* dst = full.data() + g.globalRow(r0 + r, ppr) * n;
                    for (size_t c = 0; c < pn; ++c) dst[g.globalCol(c, ppc)] = src[r * pn + c];
                }
            }
        }
    } else {
        for (size_t r0 = 0; r0 < g.mloc; r0 += g.nb) {
            const size_t rows = std::min(g.nb, g.mloc - r0);
            MPI_Send(Aloc.data() + r0 * g.ncol, (int)(rows * g.ncol), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        }
    }
}

inline double dot(const double* x, const double* y, size_t len) {
    __m256d s0 = _mm256_setzero_pd(), s1 = s0, s2 = s0, s3 = s0;
    size_t k = 0;
    for (; k + 16 <= len; k += 16) {
        s0 = _mm256_fmadd_pd(_mm256_loadu_pd(x + k), _mm256_loadu_pd(y + k), s0);
        s1 = _mm256_fmadd_pd(_mm256_loadu_pd(x + k + 4), _mm256_loadu_pd(y + k + 4), s1);
        s2 = _mm256_fmadd_pd(_mm256_loadu_pd(x + k + 8), _mm256_loadu_pd(y + k + 8), s2);
        s3 = _mm256_fmadd_pd(_mm256_loadu_pd(x + k + 12), _mm256_loadu_pd(y + k + 12), s3);
    }
    s0 = _mm256_add_pd(_mm256_add_pd(s0, s1), _mm256_add_pd(s2, s3));
    double t[4];
    _mm256_storeu_pd(t, s0);
    double sum = (t[0] + t[1]) + (t[2] + t[3]);
    for (; k < len; ++k) sum += x[k] * y[k];
    return sum;
}

// Send / receive a contiguous range of rows in bounded-size messages
void sendRows(const double* p, size_t rows, size_t n, int dest) {
    const size_t chunk = std::max<size_t>(1, (1u << 26) / std::max<size_t>(n, 1));
    for (size_t r = 0; r < rows; r += chunk)
        MPI_Send(p + r * n, (int)(std::min(chunk, rows - r) * n), MPI_DOUBLE, dest, 1, MPI_COMM_WORLD);
}
void recvRows(double* p, size_t rows, size_t n, int src) {
    const size_t chunk = std::max<size_t>(1, (1u << 26) / std::max<size_t>(n, 1));
    for (size_t r = 0; r < rows; r += chunk)
        MPI_Recv(p + r * n, (int)(std::min(chunk, rows - r) * n), MPI_DOUBLE, src, 1, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
}

// Validate by computing L * L^T and comparing with the original matrix.
// L and A_orig are full matrices on rank 0; work is distributed by rows.
bool validateCholesky(const Grid& g, const std::vector<double>& L, const std::vector<double>& A_orig) {
    const size_t n = g.n;
    auto rowBegin = [&](int p) { return n * (size_t)p / g.P; };
    const size_t i0 = rowBegin(g.rank), i1 = rowBegin(g.rank + 1), mine = i1 - i0;

    std::vector<double> Lloc, Aloc;
    const double *Lp, *Ap;
    if (g.rank == 0) {
        for (int p = 1; p < g.P; ++p) {
            const size_t b = rowBegin(p), e = rowBegin(p + 1);
            sendRows(L.data() + b * n, e - b, n, p);
            sendRows(A_orig.data() + b * n, e - b, n, p);
        }
        Lp = L.data();
        Ap = A_orig.data();
    } else {
        Lloc.resize(mine * n);
        Aloc.resize(mine * n);
        recvRows(Lloc.data(), mine, n, 0);
        recvRows(Aloc.data(), mine, n, 0);
        Lp = Lloc.data();
        Ap = Aloc.data();
    }

    double maxError = 0.0;
    double relError = 0.0;
    const size_t CH = std::max<size_t>(1, std::min<size_t>(n, (1u << 22) / std::max<size_t>(n, 1)));
    std::vector<double> chunk;
    for (size_t j0 = 0; j0 < n; j0 += CH) {
        const size_t rows = std::min(CH, n - j0);
        const double* Lj;
        if (g.rank == 0) {
            Lj = L.data() + j0 * n;
            MPI_Bcast(const_cast<double*>(Lj), (int)(rows * n), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        } else {
            chunk.resize(rows * n);
            MPI_Bcast(chunk.data(), (int)(rows * n), MPI_DOUBLE, 0, MPI_COMM_WORLD);
            Lj = chunk.data();
        }
        for (size_t i = 0; i < mine; ++i) {
            const size_t gi = i0 + i;
            for (size_t jj = 0; jj < rows; ++jj) {
                const size_t gj = j0 + jj;
                const double rec = dot(Lp + i * n, Lj + jj * n, std::min(gi, gj) + 1);
                const double a = Ap[i * n + gj];
                const double error = fabs(rec - a);
                maxError = std::max(maxError, error);
                const double rel = error / (fabs(a) + 1e-10);
                relError = std::max(relError, rel);
            }
        }
    }

    double loc[2] = {maxError, relError}, glob[2];
    MPI_Reduce(loc, glob, 2, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    int ok = 1;
    if (g.rank == 0) {
        maxError = glob[0];
        relError = glob[1];
        printf("Max absolute error: %.10e\n", maxError);
        printf("Max relative error: %.10e\n", relError);

        // Check if error is within tolerance
        if (relError > 1e-6) {
            printf("Validation failed: relative error too large\n");
            ok = 0;
        }
    }
    MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return ok != 0;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int run(int argc, char** argv, int rank, int nprocs) {
    size_t n = 512;
    bool validate = false;
    bool printResults = false;
    const bool root = rank == 0;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (root) printUsage(argv[0]);
            return 0;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    if (root) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    Grid g;
    g.init(n, nprocs, rank);

    // Allocate and generate the local part of the positive definite matrix
    std::vector<double> A;
    std::vector<double> A_orig;
    if (root) printf("Generating positive definite matrix...\n");
    generatePositiveDefiniteMatrix(g, A);

    if (validate) {
        gatherMatrix(g, A, A_orig);  // Save original for validation
        // Only the lower triangle is generated; mirror it (A is exactly symmetric)
        for (size_t i = 0; i < A_orig.size() / std::max<size_t>(n, 1); ++i)
            for (size_t j = i + 1; j < n; ++j) A_orig[i * n + j] = A_orig[j * n + i];
    }

    // Perform Cholesky decomposition
    if (root) printf("Computing Cholesky decomposition...\n");
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    Factorizer f(g, A);
    const long failIdx = f.run();

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (failIdx >= 0) {
        if (root) {
            printf("Error: Matrix is not positive definite at diagonal element %zu\n", (size_t)failIdx);
            printf("Cholesky decomposition failed\n");
        }
        return 1;
    }

    if (root) {
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    std::vector<double> L;
    if (printResults || validate) gatherMatrix(g, A, L);
    A.clear();
    A.shrink_to_fit();

    // Print results for external validation
    if (printResults && root) {
        print_results(L, "CholeskyL");
    }

    // Validation
    if (validate) {
        if (root) printf("Validating result...\n");
        bool valid = validateCholesky(g, L, A_orig);

        if (valid) {
            if (root) printf("Validation: PASSED\n");
            return 0;
        } else {
            if (root) printf("Validation: FAILED\n");
            return 1;
        }
    }

    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    const int rc = run(argc, argv, rank, nprocs);
    fflush(stdout);
    MPI_Finalize();
    return rc;
}
