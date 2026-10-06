#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// Distributed Cholesky decomposition (MPI, 2D block-cyclic, right-looking with lookahead)
// Decomposes positive definite matrix A into L * L^T where L is lower triangular
//
// Bitwise equivalence with the sequential unblocked algorithm is preserved: for every
// element (i,j) the dot product sum_{k<j} L[i][k]*L[j][k] is accumulated separately (array S)
// strictly in increasing k order, and only at the end combined as (A[i][j] - sum) / L[j][j].
// The reference build (gcc -O3 -march=native) evaluates each dot product of length m as an
// in-order sum of separately rounded products for k < (m & ~3) and with fused multiply-adds
// for the remaining tail; this rounding is reproduced explicitly (compiled with
// -ffp-contract=off, tails use fma()).

typedef double v4 __attribute__((vector_size(32), aligned(8)));

static int g_rank = 0, g_size = 1;

// ---------------------------------------------------------------------------
// Distribution context
// ---------------------------------------------------------------------------
struct Dist {
    size_t n = 0;
    size_t nb = 96;
    size_t nblk = 0;
    int Pr = 1, Pc = 1, pr = 0, pc = 0;
    size_t lrows = 0, lcols = 0, ld = 0;

    size_t bw(size_t J) const { return std::min(nb, n - J * nb); }
    int rankOf(int r, int c) const { return r * Pc + c; }
    // number of blocks J < X with J % P == p
    static size_t countBelow(size_t X, int p, int P) {
        return X > (size_t)p ? (X - p + P - 1) / P : 0;
    }
    // local row/col offset of an owned block
    size_t lro(size_t I) const { return (I / Pr) * nb; }
    size_t lco(size_t J) const { return (J / Pc) * nb; }
    // first local row/col of owned blocks >= X
    size_t lrStart(size_t X) const { return std::min(countBelow(X, pr, Pr) * nb, lrows); }
    size_t lcStart(size_t X) const { return std::min(countBelow(X, pc, Pc) * nb, lcols); }
    // number of local cols whose global index < g
    size_t lcount(size_t g) const {
        const size_t Jg = g / nb;
        size_t r = countBelow(Jg, pc, Pc) * nb;
        if (Jg < nblk && (int)(Jg % Pc) == pc) r += g - Jg * nb;
        return std::min(r, lcols);
    }
    // first block >= s with block % P == p
    static size_t firstGE(size_t s, int p, int P) {
        return s + (size_t)(((long)p - (long)(s % P) + P) % P);
    }
    bool needRow(size_t J, int c, size_t K) const { return firstGE(K + 1, c, Pc) <= J; }
    bool needCol(size_t J, int r) const { return firstGE(J, r, Pr) < nblk; }
    // does rank (r,c) need panel block (J,K) (excluding its owner)
    bool expects(int r, int c, size_t J, size_t K) const {
        if (r == (int)(J % Pr) && c == (int)(K % Pc)) return false;
        return (r == (int)(J % Pr) && needRow(J, c, K)) || (c == (int)(J % Pc) && needCol(J, r));
    }
};

// ---------------------------------------------------------------------------
// Register-blocked kernel: C[MR x 8] += X[MR x K] * P[K x 8] with k-ordered accumulation
// X is packed k-major (element (r,k) at X[k*6 + r]), P is packed [k][8]
// ---------------------------------------------------------------------------
constexpr size_t MR = 6;

template <int M>
static inline void ukernel(int K, const double* __restrict X, const double* __restrict P,
                           double* __restrict C, size_t ldc) {
    v4 c0[M], c1[M];
    for (int r = 0; r < M; ++r) {
        c0[r] = *(const v4*)(C + r * ldc);
        c1[r] = *(const v4*)(C + r * ldc + 4);
    }
    for (int k = 0; k < K; ++k) {
        const v4 b0 = *(const v4*)(P + 8 * k);
        const v4 b1 = *(const v4*)(P + 8 * k + 4);
        for (int r = 0; r < M; ++r) {
            const double a = X[MR * k + r];
            const v4 av = {a, a, a, a};
            c0[r] += av * b0;
            c1[r] += av * b1;
        }
    }
    for (int r = 0; r < M; ++r) {
        *(v4*)(C + r * ldc) = c0[r];
        *(v4*)(C + r * ldc + 4) = c1[r];
    }
}

static inline void ukernelDispatch(int mr, int K, const double* X, const double* P, double* C, size_t ldc) {
    switch (mr) {
        case 6: ukernel<6>(K, X, P, C, ldc); break;
        case 5: ukernel<5>(K, X, P, C, ldc); break;
        case 4: ukernel<4>(K, X, P, C, ldc); break;
        case 3: ukernel<3>(K, X, P, C, ldc); break;
        case 2: ukernel<2>(K, X, P, C, ldc); break;
        default: ukernel<1>(K, X, P, C, ldc); break;
    }
}

// Update one local row block: C[rows x (lcb..lce)] += X[rows x K] * Pp^T
// X is row-major with leading dimension ldx; it is packed into Xp (size >= roundup(rows,6)*K).
// Pp is packed in tiles of 8 local columns (tile t at Pp + t*tstride, layout [k][8]).
// If tri, only columns with global index <= global row are needed (rounded up to 8).
static void gemmRows(const Dist& d, double* C, const double* X, size_t ldx, size_t rows, size_t gi0,
                     const double* Pp, size_t tstride, int K, size_t lcb, size_t lce, bool tri, double* Xp) {
    if (lcb >= lce || rows == 0) return;
    const size_t ntiles = (rows + MR - 1) / MR;
    std::vector<size_t> lim(ntiles);
    size_t maxLim = 0;
    for (size_t t = 0; t < ntiles; ++t) {
        size_t l = lce;
        if (tri) {
            const size_t glast = gi0 + std::min(rows, (t + 1) * MR) - 1;
            l = std::min(lce, (d.lcount(glast + 1) + 7) & ~(size_t)7);
        }
        lim[t] = l;
        maxLim = std::max(maxLim, l);
    }
    if (maxLim <= lcb) return;
    // pack X
    for (size_t t = 0; t < ntiles; ++t) {
        const size_t r0 = t * MR, mr = std::min(MR, rows - r0);
        double* dst = Xp + t * MR * K;
        for (size_t r = 0; r < mr; ++r) {
            const double* src = X + (r0 + r) * ldx;
            for (int k = 0; k < K; ++k) dst[k * MR + r] = src[k];
        }
    }
    for (size_t c = lcb; c < maxLim; c += 8) {
        const double* P = Pp + (c / 8) * tstride;
        for (size_t t = 0; t < ntiles; ++t) {
            if (c >= lim[t]) continue;
            const size_t r0 = t * MR;
            const int mr = (int)std::min(MR, rows - r0);
            // prefetch the accumulator tile of the next call
            {
                size_t tn = t + 1, cn = c;
                if (tn == ntiles) tn = 0, cn = c + 8;
                if (cn < lim[tn]) {
                    const double* nc = C + tn * MR * d.ld + cn;
                    const size_t mrn = std::min(MR, rows - tn * MR);
                    for (size_t r = 0; r < mrn; ++r) __builtin_prefetch(nc + r * d.ld, 1, 3);
                }
            }
            ukernelDispatch(mr, K, Xp + t * MR * K, P, C + r0 * d.ld + c, d.ld);
        }
    }
}

// ---------------------------------------------------------------------------
// Distributed factorization
// ---------------------------------------------------------------------------
struct Factorizer {
    const Dist& d;
    std::vector<double>& A;  // local A, overwritten with L
    std::vector<double> S;   // accumulated dot products
    std::vector<double> Pp;  // packed column operand
    std::vector<double> Pbuf[2];
    std::vector<double> D;
    std::vector<double> Xp;  // packed row operand
    std::vector<double> Tbuf;  // transposed TRSM tile
    std::vector<MPI_Request> sendReq[2], recvReq[2], dReq;

    Factorizer(const Dist& dist, std::vector<double>& Aloc) : d(dist), A(Aloc) {
        S.assign(d.lrows * d.ld, 0.0);
        Pp.assign(d.ld * d.nb, 0.0);
        Pbuf[0].assign(d.nblk * d.nb * d.nb, 0.0);
        Pbuf[1].assign(d.nblk * d.nb * d.nb, 0.0);
        D.assign(d.nb * d.nb, 0.0);
        Xp.assign((d.nb + MR) * d.nb, 0.0);
        Tbuf.assign(d.nb * 32 + 8, 0.0);
    }

    void release() {
        std::vector<double>().swap(S);
        std::vector<double>().swap(Pp);
        std::vector<double>().swap(Pbuf[0]);
        std::vector<double>().swap(Pbuf[1]);
    }

    double* blk(int b, size_t J) { return Pbuf[b].data() + J * d.nb * d.nb; }
    int tag(size_t J, size_t K) const { return (int)(J + d.nblk * (K & 1)); }
    int dtag() const { return (int)(2 * d.nblk); }

    void fail(size_t idx) {
        printf("Error: Matrix is not positive definite at diagonal element %zu\n", idx);
        printf("Cholesky decomposition failed\n");
        fflush(stdout);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Factor diagonal block K (owned by this rank) in place and copy into D
    void factorDiag(size_t K) {
        const size_t bw = d.bw(K), lr0 = d.lro(K), lc0 = d.lco(K), ld = d.ld;
        double* L = A.data() + lr0 * ld + lc0;
        const double* Sb = S.data() + lr0 * ld + lc0;
        for (size_t jj = 0; jj < bw; ++jj) {
            double* Lj = L + jj * ld;
            const size_t jj4 = jj & ~(size_t)3;
            double sum = Sb[jj * ld + jj];
            for (size_t k = 0; k < jj4; ++k) sum += Lj[k] * Lj[k];
            for (size_t k = jj4; k < jj; ++k) sum = fma(Lj[k], Lj[k], sum);
            const double val = Lj[jj] - sum;
            if (val <= 0.0) fail(K * d.nb + jj);
            Lj[jj] = sqrt(val);
            const double piv = Lj[jj];
            size_t ii = jj + 1;
            for (; ii + 4 <= bw; ii += 4) {
                double s[4];
                for (int r = 0; r < 4; ++r) s[r] = Sb[(ii + r) * ld + jj];
                for (size_t k = 0; k < jj4; ++k) {
                    const double lk = Lj[k];
                    for (int r = 0; r < 4; ++r) s[r] += L[(ii + r) * ld + k] * lk;
                }
                for (size_t k = jj4; k < jj; ++k)
                    for (int r = 0; r < 4; ++r) s[r] = fma(L[(ii + r) * ld + k], Lj[k], s[r]);
                for (int r = 0; r < 4; ++r) L[(ii + r) * ld + jj] = (L[(ii + r) * ld + jj] - s[r]) / piv;
            }
            for (; ii < bw; ++ii) {
                double s = Sb[ii * ld + jj];
                for (size_t k = 0; k < jj4; ++k) s += L[ii * ld + k] * Lj[k];
                for (size_t k = jj4; k < jj; ++k) s = fma(L[ii * ld + k], Lj[k], s);
                L[ii * ld + jj] = (L[ii * ld + jj] - s) / piv;
            }
        }
        for (size_t r = 0; r < bw; ++r) memcpy(D.data() + r * d.nb, L + r * ld, (r + 1) * sizeof(double));
    }

    // Triangular solve for local rows [lrb, lre) in column block K using D.
    // Rows are processed in tiles of 32, transposed so that each k step is vectorized across rows.
    void trsm(size_t K, size_t lrb, size_t lre) {
        const size_t bw = d.bw(K), lc0 = d.lco(K), ld = d.ld, nb = d.nb;
        constexpr int RV = 8;  // vectors per tile
        constexpr size_t R = 4 * RV;
        double* T = Tbuf.data();
        for (size_t r0 = lrb; r0 < lre; r0 += R) {
            const size_t mr = std::min(R, lre - r0);
            for (size_t k = 0; k < bw; ++k)
                for (size_t r = 0; r < R; ++r) T[k * R + r] = r < mr ? A[(r0 + r) * ld + lc0 + k] : 0.0;
            for (size_t jj = 0; jj < bw; ++jj) {
                const double* Dj = D.data() + jj * nb;
                const size_t jj4 = jj & ~(size_t)3;
                v4 sv[RV];
                double* sp = (double*)sv;
                for (size_t r = 0; r < R; ++r) sp[r] = r < mr ? S[(r0 + r) * ld + lc0 + jj] : 0.0;
                for (size_t k = 0; k < jj4; ++k) {
                    const double dk = Dj[k];
                    const v4 dv = {dk, dk, dk, dk};
                    const v4* tk = (const v4*)(T + k * R);
                    for (int v = 0; v < RV; ++v) sv[v] += tk[v] * dv;
                }
                for (size_t k = jj4; k < jj; ++k)
                    for (size_t r = 0; r < R; ++r) sp[r] = fma(T[k * R + r], Dj[k], sp[r]);
                const double piv = Dj[jj];
                const v4 pv = {piv, piv, piv, piv};
                v4* tj = (v4*)(T + jj * R);
                for (int v = 0; v < RV; ++v) tj[v] = (tj[v] - sv[v]) / pv;
            }
            for (size_t r = 0; r < mr; ++r)
                for (size_t k = 0; k < bw; ++k) A[(r0 + r) * ld + lc0 + k] = T[k * R + r];
        }
    }

    void postRecvs(size_t K) {
        const int b = K & 1;
        recvReq[b].clear();
        const size_t bwK = d.bw(K);
        for (size_t J = K + 1; J < d.nblk; ++J) {
            if (!d.expects(d.pr, d.pc, J, K)) continue;
            MPI_Request rq;
            const int src = d.rankOf((int)(J % d.Pr), (int)(K % d.Pc));
            MPI_Irecv(blk(b, J), (int)(d.bw(J) * bwK), MPI_DOUBLE, src, tag(J, K), MPI_COMM_WORLD, &rq);
            recvReq[b].push_back(rq);
        }
    }

    // Factor column block K: diag + D distribution + TRSM + panel sends (only process column K%Pc)
    void doPanel(size_t K) {
        if ((int)(K % d.Pc) != d.pc) return;
        const int b = K & 1;
        const int ownerRow = (int)(K % d.Pr);
        const size_t bwK = d.bw(K);
        const size_t lrb = d.lrStart(K + 1);
        if (ownerRow == d.pr) {
            factorDiag(K);
            for (int r = 0; r < d.Pr; ++r) {
                if (r == d.pr || Dist::firstGE(K + 1, r, d.Pr) >= d.nblk) continue;
                MPI_Request rq;
                MPI_Isend(D.data(), (int)(d.nb * d.nb), MPI_DOUBLE, d.rankOf(r, d.pc), dtag(), MPI_COMM_WORLD, &rq);
                dReq.push_back(rq);
            }
        } else if (lrb < d.lrows) {
            MPI_Recv(D.data(), (int)(d.nb * d.nb), MPI_DOUBLE, d.rankOf(ownerRow, d.pc), dtag(), MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
        }
        trsm(K, lrb, d.lrows);
        if (!dReq.empty()) {
            MPI_Waitall((int)dReq.size(), dReq.data(), MPI_STATUSES_IGNORE);
            dReq.clear();
        }
        if (!sendReq[b].empty()) {
            MPI_Waitall((int)sendReq[b].size(), sendReq[b].data(), MPI_STATUSES_IGNORE);
            sendReq[b].clear();
        }
        const size_t lc0 = d.lco(K);
        for (size_t I = Dist::firstGE(K + 1, d.pr, d.Pr); I < d.nblk; I += d.Pr) {
            const size_t bwI = d.bw(I), lr0 = d.lro(I);
            double* dst = blk(b, I);
            for (size_t t = 0; t < bwI; ++t) memcpy(dst + t * bwK, A.data() + (lr0 + t) * d.ld + lc0, bwK * sizeof(double));
            for (int r = 0; r < d.Pr; ++r) {
                for (int c = 0; c < d.Pc; ++c) {
                    if (!d.expects(r, c, I, K)) continue;
                    MPI_Request rq;
                    MPI_Isend(dst, (int)(bwI * bwK), MPI_DOUBLE, d.rankOf(r, c), tag(I, K), MPI_COMM_WORLD, &rq);
                    sendReq[b].push_back(rq);
                }
            }
        }
    }

    // Pack column operand (local col blocks J > K) from panel K
    void buildPp(size_t K) {
        const int b = K & 1;
        const size_t nb = d.nb;
        for (size_t J = Dist::firstGE(K + 1, d.pc, d.Pc); J < d.nblk; J += d.Pc) {
            const size_t bwJ = d.bw(J), lc0 = d.lco(J);
            const double* src = blk(b, J);
            for (size_t t = 0; t < bwJ; ++t) {
                const size_t lc = lc0 + t;
                double* dst = Pp.data() + (lc / 8) * nb * 8 + (lc % 8);
                for (size_t k = 0; k < nb; ++k) dst[k * 8] = src[t * nb + k];
            }
        }
    }

    // Trailing update with panel K for local columns [lcb, lce), row blocks >= Ifirst
    void update(size_t K, size_t Ifirst, size_t lcb, size_t lce, bool progress) {
        const int b = K & 1;
        const size_t nb = d.nb;
        for (size_t I = Dist::firstGE(Ifirst, d.pr, d.Pr); I < d.nblk; I += d.Pr) {
            gemmRows(d, S.data() + d.lro(I) * d.ld, blk(b, I), nb, d.bw(I), I * nb, Pp.data(), nb * 8, (int)nb,
                     lcb, lce, true, Xp.data());
            if (progress) {
                const int nb2 = b ^ 1;
                int flag;
                if (!recvReq[nb2].empty())
                    MPI_Testall((int)recvReq[nb2].size(), recvReq[nb2].data(), &flag, MPI_STATUSES_IGNORE);
            }
        }
    }

    void run() {
        if (d.nblk == 0) return;
        postRecvs(0);
        doPanel(0);
        for (size_t K = 0; K + 1 < d.nblk; ++K) {
            const int b = K & 1, nb2 = b ^ 1;
            if (!recvReq[b].empty()) {
                MPI_Waitall((int)recvReq[b].size(), recvReq[b].data(), MPI_STATUSES_IGNORE);
                recvReq[b].clear();
            }
            buildPp(K);
            if (!sendReq[nb2].empty()) {
                MPI_Waitall((int)sendReq[nb2].size(), sendReq[nb2].data(), MPI_STATUSES_IGNORE);
                sendReq[nb2].clear();
            }
            postRecvs(K + 1);
            // Lookahead: update column block K+1 first, then factor it while the rest is updated
            if ((int)((K + 1) % d.Pc) == d.pc) {
                const size_t lc0 = d.lco(K + 1);
                update(K, K + 1, lc0, std::min(lc0 + d.nb, d.lcols), false);
            }
            doPanel(K + 1);
            update(K, K + 2, d.lcStart(K + 2), d.lcols, true);
        }
        for (int b = 0; b < 2; ++b)
            if (!sendReq[b].empty()) MPI_Waitall((int)sendReq[b].size(), sendReq[b].data(), MPI_STATUSES_IGNORE);
    }
};

// ---------------------------------------------------------------------------
// C_local += Src * Src^T over all k (k ascending), Src is a full n x n row-major matrix
// (used for matrix generation and validation)
// ---------------------------------------------------------------------------
static void localGramFull(const Dist& d, const double* Src, std::vector<double>& C, bool tri) {
    const size_t n = d.n, KC = 256, n4 = n & ~(size_t)3;
    std::vector<double> Pk(d.ld * KC, 0.0);
    std::vector<double> Xp((d.nb + MR) * KC, 0.0);
    for (size_t k0 = 0; k0 < n; k0 += KC) {
        const size_t kc = std::min(KC, n - k0);
        for (size_t J = d.pc; J < d.nblk; J += d.Pc) {
            const size_t lc0 = d.lco(J);
            for (size_t t = 0; t < d.bw(J); ++t) {
                const size_t lc = lc0 + t;
                const double* src = Src + (J * d.nb + t) * n + k0;
                double* dst = Pk.data() + (lc / 8) * KC * 8 + (lc % 8);
                for (size_t k = 0; k < kc; ++k) dst[k * 8] = src[k];
            }
        }
        // separately rounded products for k < n4, fused tail beyond
        const size_t kv = k0 + kc <= n4 ? kc : (k0 < n4 ? n4 - k0 : 0);
        for (size_t I = d.pr; I < d.nblk; I += d.Pr) {
            double* Cb = C.data() + d.lro(I) * d.ld;
            const double* Xb = Src + I * d.nb * n + k0;
            if (kv > 0)
                gemmRows(d, Cb, Xb, n, d.bw(I), I * d.nb, Pk.data(), KC * 8, (int)kv, 0, d.lcols, tri,
                         Xp.data());
            if (kv == kc) continue;
            for (size_t t = 0; t < d.bw(I); ++t) {
                const size_t lcEnd = tri ? d.lcount(I * d.nb + t + 1) : d.lcols;
                for (size_t lc = 0; lc < lcEnd; ++lc) {
                    const double* P = Pk.data() + (lc / 8) * KC * 8 + (lc % 8);
                    double c = Cb[t * d.ld + lc];
                    for (size_t k = kv; k < kc; ++k) c = fma(Xb[t * n + k], P[k * 8], c);
                    Cb[t * d.ld + lc] = c;
                }
            }
        }
    }
}

// Generate a symmetric positive definite matrix (local blocks of the 2D distribution)
// B is shared per node and generated by the node leader exactly as in the sequential code.
void generatePositiveDefiniteMatrix(const Dist& d, double* B, MPI_Comm nodeComm, MPI_Win win,
                                    std::vector<double>& Aloc, bool full) {
    // Method: Create A = B * B^T where B is random
    // This guarantees A is positive semi-definite
    // Then add identity to make it strictly positive definite
    const size_t n = d.n;
    int nodeRank;
    MPI_Comm_rank(nodeComm, &nodeRank);
    MPI_Win_fence(0, win);
    if (nodeRank == 0) {
        unsigned int seed = 42;
        // Generate random matrix B
        for (size_t i = 0; i < n * n; ++i) {
            B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
        }
    }
    MPI_Win_fence(0, win);

    // Compute A = B * B^T (local part)
    localGramFull(d, B, Aloc, !full);

    // Add diagonal dominance to ensure positive definiteness
    for (size_t I = d.pr; I < d.nblk; I += d.Pr) {
        if ((int)(I % d.Pc) != d.pc) continue;
        for (size_t t = 0; t < d.bw(I); ++t) Aloc[(d.lro(I) + t) * d.ld + d.lco(I) + t] += n;
    }
    MPI_Win_fence(0, win);
}

// Gather distributed L to rank 0 (full n x n, upper part zero)
static void gatherL(const Dist& d, const std::vector<double>& Aloc, std::vector<double>& Afull) {
    const size_t n = d.n, nb = d.nb;
    if (g_rank == 0) {
        Afull.assign(n * n, 0.0);
        std::vector<double> tmp;
        for (int src = 0; src < g_size; ++src) {
            const int r = src / d.Pc, c = src % d.Pc;
            Dist ds = d;
            ds.pr = r;
            ds.pc = c;
            size_t lcols = 0;
            for (size_t J = c; J < d.nblk; J += d.Pc) lcols += d.bw(J);
            for (size_t I = r; I < d.nblk; I += d.Pr) {
                const size_t bwI = d.bw(I);
                const double* rows;
                if (src == 0) {
                    tmp.resize(bwI * lcols);
                    for (size_t t = 0; t < bwI; ++t)
                        memcpy(tmp.data() + t * lcols, Aloc.data() + (d.lro(I) + t) * d.ld, lcols * sizeof(double));
                } else {
                    tmp.resize(bwI * lcols);
                    MPI_Recv(tmp.data(), (int)(bwI * lcols), MPI_DOUBLE, src, 7, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
                rows = tmp.data();
                for (size_t J = c; J <= I && J < d.nblk; J += d.Pc) {
                    const size_t lc0 = ds.lco(J);
                    for (size_t t = 0; t < bwI; ++t) {
                        const size_t i = I * nb + t;
                        const size_t jmax = std::min(d.bw(J), i - J * nb + 1);
                        memcpy(Afull.data() + i * n + J * nb, rows + t * lcols + lc0, jmax * sizeof(double));
                    }
                }
            }
        }
    } else {
        std::vector<double> tmp;
        for (size_t I = d.pr; I < d.nblk; I += d.Pr) {
            const size_t bwI = d.bw(I);
            tmp.resize(bwI * d.lcols);
            for (size_t t = 0; t < bwI; ++t)
                memcpy(tmp.data() + t * d.lcols, Aloc.data() + (d.lro(I) + t) * d.ld, d.lcols * sizeof(double));
            MPI_Send(tmp.data(), (int)(bwI * d.lcols), MPI_DOUBLE, 0, 7, MPI_COMM_WORLD);
        }
    }
}

bool validateCholesky(const Dist& d, const double* Lfull, const std::vector<double>& Aorig) {
    // Validate by computing L * L^T and comparing with original matrix

    std::vector<double> reconstructed(d.lrows * d.ld, 0.0);

    // Compute L * L^T (local blocks)
    localGramFull(d, Lfull, reconstructed, false);

    // Compare with original
    double err[2] = {0.0, 0.0};  // maxError, relError

    for (size_t I = d.pr; I < d.nblk; I += d.Pr) {
        for (size_t t = 0; t < d.bw(I); ++t) {
            const size_t lr = d.lro(I) + t;
            for (size_t lc = 0; lc < d.lcols; ++lc) {
                const double error = fabs(reconstructed[lr * d.ld + lc] - Aorig[lr * d.ld + lc]);
                err[0] = std::max(err[0], error);

                const double rel = error / (fabs(Aorig[lr * d.ld + lc]) + 1e-10);
                err[1] = std::max(err[1], rel);
            }
        }
    }
    double gerr[2];
    MPI_Allreduce(err, gerr, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    const double maxError = gerr[0], relError = gerr[1];

    if (g_rank == 0) {
        printf("Max absolute error: %.10e\n", maxError);
        printf("Max relative error: %.10e\n", relError);
    }

    // Check if error is within tolerance
    if (relError > 1e-6) {
        if (g_rank == 0) printf("Validation failed: relative error too large\n");
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

static Dist makeDist(size_t n) {
    Dist d;
    d.n = n;
    // Process grid: Pr x Pc as square as possible, Pr >= Pc (more rows share the panel solve)
    int Pc = (int)std::sqrt((double)g_size);
    while (g_size % Pc) --Pc;
    d.Pc = Pc;
    d.Pr = g_size / Pc;
    d.pr = g_rank / d.Pc;
    d.pc = g_rank % d.Pc;
    // Block size (multiple of 24 = lcm of kernel tile sizes): as large as possible (less memory
    // traffic) while keeping enough blocks per process dimension for load balance and pipelining
    static const size_t cand[] = {240, 192, 144, 120, 96, 72, 48, 24};
    for (size_t c : cand) {
        d.nb = c;
        if ((n + c - 1) / c >= (size_t)10 * std::max(d.Pr, d.Pc)) break;
    }
    d.nblk = (n + d.nb - 1) / d.nb;
    for (size_t I = d.pr; I < d.nblk; I += d.Pr) d.lrows += d.bw(I);
    for (size_t J = d.pc; J < d.nblk; J += d.Pc) d.lcols += d.bw(J);
    d.ld = (d.lcols + 7) & ~(size_t)7;
    if (d.ld % 512 == 0) d.ld += 8;
    if (d.ld == 0) d.ld = 8;
    return d;
}

static void bcastLarge(double* buf, size_t count, MPI_Comm comm) {
    const size_t chunk = (size_t)1 << 28;
    for (size_t off = 0; off < count; off += chunk)
        MPI_Bcast(buf + off, (int)std::min(chunk, count - off), MPI_DOUBLE, 0, comm);
}

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

    if (g_rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const Dist d = makeDist(n);

    // Node-shared full matrix buffer (random B for generation, full L for validation)
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, g_rank, MPI_INFO_NULL, &nodeComm);
    int nodeRank;
    MPI_Comm_rank(nodeComm, &nodeRank);
    MPI_Comm leaderComm;
    MPI_Comm_split(MPI_COMM_WORLD, nodeRank == 0 ? 0 : MPI_UNDEFINED, g_rank, &leaderComm);
    MPI_Win win;
    double* shared = nullptr;
    {
        const MPI_Aint bytes = nodeRank == 0 ? (MPI_Aint)(std::max<size_t>(n * n, 1) * sizeof(double)) : 0;
        double* base;
        MPI_Win_allocate_shared(bytes, sizeof(double), MPI_INFO_NULL, nodeComm, &base, &win);
        MPI_Aint sz;
        int du;
        MPI_Win_shared_query(win, 0, &sz, &du, &shared);
    }

    // Allocate matrix (local blocks)
    std::vector<double> A(d.lrows * d.ld, 0.0);
    std::vector<double> A_orig;

    // Generate positive definite matrix
    if (g_rank == 0) printf("Generating positive definite matrix...\n");
    generatePositiveDefiniteMatrix(d, shared, nodeComm, win, A, validate);

    if (validate) {
        A_orig = A; // Save original for validation
    }

    // Perform Cholesky decomposition (work buffers are allocated up front)
    if (g_rank == 0) printf("Computing Cholesky decomposition...\n");
    Factorizer f(d, A);
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    f.run();
    MPI_Barrier(MPI_COMM_WORLD);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    f.release();

    if (g_rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    int rc = 0;
    if (printResults || validate) {
        std::vector<double> Afull;
        gatherL(d, A, Afull);

        // Print results for external validation
        if (printResults && g_rank == 0) {
            print_results(Afull, "CholeskyL");
        }

        // Validation
        if (validate) {
            if (g_rank == 0) printf("Validating result...\n");
            MPI_Win_fence(0, win);
            if (g_rank == 0 && n > 0) memcpy(shared, Afull.data(), n * n * sizeof(double));
            if (leaderComm != MPI_COMM_NULL) bcastLarge(shared, n * n, leaderComm);
            MPI_Win_fence(0, win);
            bool valid = validateCholesky(d, shared, A_orig);

            if (g_rank == 0) printf(valid ? "Validation: PASSED\n" : "Validation: FAILED\n");
            rc = valid ? 0 : 1;
        }
    }

    MPI_Win_free(&win);
    if (leaderComm != MPI_COMM_NULL) MPI_Comm_free(&leaderComm);
    MPI_Comm_free(&nodeComm);
    MPI_Finalize();
    return rc;
}
