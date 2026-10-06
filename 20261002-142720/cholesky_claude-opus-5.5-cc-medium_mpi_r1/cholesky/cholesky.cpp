#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// Distributed Cholesky decomposition (MPI, 2D block-cyclic, right-looking blocked algorithm)
// Decomposes positive definite matrix A into L * L^T where L is lower triangular
//
// The matrix is split into nb x nb tiles distributed block-cyclically over a Pr x Pc
// process grid. Every element of L is computed with exactly the same sequence of
// floating point operations as the sequential unblocked algorithm: the dot product
// sum_{k<j} L_ik * L_jk is accumulated in k order into a separate array S (trailing
// updates add each panel's contribution in order), and the final (A_ij - sum) / L_jj
// (or sqrt(A_jj - sum)) is formed only when the element is factored.
//
// Dot products follow the arithmetic of the reference build (GCC -O3 -march=native):
// the reduction loop is vectorized 4-wide with separately rounded products added in
// order, and the remaining (length % 4) terms use fused multiply-add. This file is
// compiled with -ffp-contract=off and FMAs are explicit, so results are bitwise
// identical to the sequential program for any process count.

namespace {

// Dot-product terms k >= fmaStart(len) are accumulated with fused multiply-add
inline size_t fmaStart(size_t len) { return len & ~size_t(3); }

constexpr int MR = 6;  // micro-kernel rows
constexpr int NR = 8;  // micro-kernel columns
constexpr int TILE_ALIGN = 24;  // lcm(MR, NR)

typedef double v4d __attribute__((vector_size(32), aligned(8)));

// C[MR x NR] += Ap * Bp^T, accumulating over k in increasing order for every element
// (rounded products, no contraction).
// Ap: packed [kc][MR], Bp: packed [kc][NR]
inline void microKernel(int kc, const double* __restrict Ap, const double* __restrict Bp,
                        double* __restrict C, size_t ldc) {
    v4d c[MR][2];
    for (int r = 0; r < MR; ++r) {
        c[r][0] = *reinterpret_cast<const v4d*>(C + r * ldc);
        c[r][1] = *reinterpret_cast<const v4d*>(C + r * ldc + 4);
    }
    for (int k = 0; k < kc; ++k) {
        const v4d b0 = *reinterpret_cast<const v4d*>(Bp + k * NR);
        const v4d b1 = *reinterpret_cast<const v4d*>(Bp + k * NR + 4);
        for (int r = 0; r < MR; ++r) {
            const double a = Ap[k * MR + r];
            const v4d av = {a, a, a, a};
            c[r][0] += av * b0;
            c[r][1] += av * b1;
        }
    }
    for (int r = 0; r < MR; ++r) {
        *reinterpret_cast<v4d*>(C + r * ldc) = c[r][0];
        *reinterpret_cast<v4d*>(C + r * ldc + 4) = c[r][1];
    }
}

// Pack rows [0, rows) x cols [k0, k0 + kc) of row-major src into strips of W rows: [strip][k][W]
template <int W>
void packStrips(double* dst, const double* src, size_t lds, size_t rows, size_t k0, size_t kc) {
    for (size_t s = 0; s < rows; s += W) {
        double* d = dst + s * kc;
        for (int r = 0; r < W; ++r) {
            const double* row = src + (s + r) * lds + k0;
            for (size_t k = 0; k < kc; ++k) d[k * W + r] = row[k];
        }
    }
}

// C[M x N] += A[M x K] * B[N x K]^T (row-major). M % MR == 0, N % NR == 0.
void gemmNT(double* C, size_t ldc, const double* A, size_t lda, const double* B, size_t ldb,
            size_t M, size_t N, size_t K) {
    constexpr size_t KC = 256, NC = 480;
    std::vector<double> Ap(MR * KC), Bp(NC * KC);
    for (size_t k0 = 0; k0 < K; k0 += KC) {
        const size_t kc = std::min(KC, K - k0);
        for (size_t n0 = 0; n0 < N; n0 += NC) {
            const size_t nc = std::min(NC, N - n0);
            packStrips<NR>(Bp.data(), B + n0 * ldb, ldb, nc, k0, kc);
            for (size_t m0 = 0; m0 < M; m0 += MR) {
                packStrips<MR>(Ap.data(), A + m0 * lda, lda, MR, k0, kc);
                for (size_t s = 0; s < nc; s += NR) {
                    microKernel((int)kc, Ap.data(), Bp.data() + s * kc, C + m0 * ldc + n0 + s, ldc);
                }
            }
        }
    }
}

// C[M x N] = fma over k in [k0, k1) of A[M x K] * B[N x K]^T
void fmaTail(double* C, size_t ldc, const double* A, size_t lda, const double* B, size_t ldb,
             size_t M, size_t N, size_t k0, size_t k1) {
    if (k0 == k1) return;
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            double c = C[i * ldc + j];
            for (size_t k = k0; k < k1; ++k) c = std::fma(A[i * lda + k], B[j * ldb + k], c);
            C[i * ldc + j] = c;
        }
    }
}

struct Grid {
    int rank = 0, size = 1;
    int Pr = 1, Pc = 1, pr = 0, pc = 0;
    MPI_Comm rowComm = MPI_COMM_NULL;  // ranks with same pr, ordered by pc
    MPI_Comm colComm = MPI_COMM_NULL;  // ranks with same pc, ordered by pr
};

struct Layout {
    size_t n = 0, nb = 0, nt = 0;  // matrix size, tile size, number of tiles per dimension
    size_t mt = 0, ntc = 0;        // number of local row / column tiles
    size_t rows = 0, ld = 0;       // local storage: rows x ld (row-major)

    size_t tileSize(size_t I) const { return std::min(nb, n - I * nb); }
};

size_t countOwned(size_t nt, int p, int P) { return (size_t)p < nt ? (nt - p + P - 1) / P : 0; }
// number of owned tiles (index % P == p) with index <= K
size_t countOwnedUpTo(size_t K, int p, int P) { return K >= (size_t)p ? (K - p) / P + 1 : 0; }

Grid makeGrid() {
    Grid g;
    MPI_Comm_rank(MPI_COMM_WORLD, &g.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g.size);
    // Pr >= Pc, as close to square as possible (CHOLESKY_PR overrides the number of process rows)
    int Pc = (int)std::sqrt((double)g.size);
    while (Pc > 1 && g.size % Pc != 0) --Pc;
    int Pr = g.size / Pc;
    if (const char* e = getenv("CHOLESKY_PR")) {
        const int r = atoi(e);
        if (r > 0 && g.size % r == 0) Pr = r;
    }
    g.Pr = Pr;
    g.Pc = g.size / Pr;
    g.pr = g.rank / g.Pc;
    g.pc = g.rank % g.Pc;
    MPI_Comm_split(MPI_COMM_WORLD, g.pr, g.pc, &g.rowComm);
    MPI_Comm_split(MPI_COMM_WORLD, g.pc, g.pr, &g.colComm);
    return g;
}

Layout makeLayout(size_t n, size_t nb, const Grid& g) {
    Layout L;
    L.n = n;
    L.nb = nb;
    L.nt = (n + nb - 1) / nb;
    L.mt = countOwned(L.nt, g.pr, g.Pr);
    L.ntc = countOwned(L.nt, g.pc, g.Pc);
    L.rows = L.mt * nb;
    L.ld = std::max<size_t>(L.ntc * nb, NR);
    return L;
}

inline size_t globalRow(size_t lr, const Layout& L, const Grid& g) {
    return ((lr / L.nb) * g.Pr + g.pr) * L.nb + lr % L.nb;
}
inline size_t globalCol(size_t lc, const Layout& L, const Grid& g) {
    return ((lc / L.nb) * g.Pc + g.pc) * L.nb + lc % L.nb;
}

// Generate the local part of the symmetric positive definite matrix A = B * B^T + n * I
void generatePositiveDefiniteMatrix(std::vector<double>& A, const Layout& L, const Grid& g) {
    const size_t n = L.n, nb = L.nb;
    std::vector<double> Brow(L.rows * n, 0.0), Bcol(L.ntc * nb * n, 0.0);
    std::vector<double> tmp(n);
    unsigned int seed = 42;

    // Generate random matrix B (same sequential stream), keeping only the needed rows
    for (size_t i = 0; i < n; ++i) {
        for (size_t k = 0; k < n; ++k) {
            tmp[k] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
        }
        const size_t I = i / nb;
        if ((int)(I % g.Pr) == g.pr) {
            std::memcpy(&Brow[((I / g.Pr) * nb + i % nb) * n], tmp.data(), n * sizeof(double));
        }
        if ((int)(I % g.Pc) == g.pc) {
            std::memcpy(&Bcol[((I / g.Pc) * nb + i % nb) * n], tmp.data(), n * sizeof(double));
        }
    }

    // Compute A = B * B^T (per element: sequential sum over k)
    A.assign(L.rows * L.ld, 0.0);
    const size_t kf = fmaStart(n);
    gemmNT(A.data(), L.ld, Brow.data(), n, Bcol.data(), n, L.rows, L.ntc * nb, kf);
    fmaTail(A.data(), L.ld, Brow.data(), n, Bcol.data(), n, L.rows, L.ntc * nb, kf, n);

    // Add diagonal dominance to ensure positive definiteness
    for (size_t lr = 0; lr < L.rows; ++lr) {
        const size_t gi = globalRow(lr, L, g);
        if (gi >= n) continue;
        const size_t J = gi / nb;
        if ((int)(J % g.Pc) != g.pc) continue;
        A[lr * L.ld + (J / g.Pc) * nb + gi % nb] += n;
    }
}

// Returns 0 on success, otherwise (index of failing diagonal element + 1)
size_t choleskyDecomposition(std::vector<double>& Avec, const Layout& L, const Grid& g) {
    const size_t nb = L.nb, ld = L.ld;
    double* A = Avec.data();
    std::vector<double> S(L.rows * ld, 0.0);  // accumulated dot products

    std::vector<double> Lkk(nb * nb + 1), rowPanel(L.rows * nb + 1), colPanel(L.ntc * nb * nb + 1);
    std::vector<double> sendBuf(L.rows * nb + 1);
    std::vector<double> Ap(L.rows * nb), Bp(L.ntc * nb * nb);
    std::vector<double> Pt(nb * nb), St(nb * nb);
    constexpr size_t RCH = 32;  // panel-solve row chunk
    std::vector<double> acc(std::max(nb, RCH));
    std::vector<double> Pc(nb * RCH);
    std::vector<int> counts(g.Pr), displs(g.Pr);
    std::vector<size_t> qIndex(g.Pr);
    std::vector<const double*> colTilePtr(L.ntc);

    MPI_Request rowReq = MPI_REQUEST_NULL, colReq = MPI_REQUEST_NULL;
    int stage = 0;  // 0: row broadcast in flight, 1: column allgather in flight, 2: panel received
    size_t pendingK = 0;
    size_t pendingFlag = 0;

    // Factor panel K on its process column (diagonal tile + rows below) and start
    // broadcasting the panel rows along process rows
    auto factorPanel = [&](size_t K) {
        const size_t k0 = K * nb;
        const size_t kb = L.tileSize(K);
        const int prK = (int)(K % g.Pr), pcK = (int)(K % g.Pc);
        const size_t li0 = countOwnedUpTo(K, g.pr, g.Pr);  // first local row tile below K
        const size_t nrp = (L.mt - li0) * nb;              // rows in the row panel
        double flag = 0.0;

        if (g.pc == pcK) {
            const size_t c0 = (K / g.Pc) * nb;  // local column of tile K
            if (g.pr == prK) {
                // Factor diagonal tile (column by column, transposed for vectorization)
                const size_t r0 = (K / g.Pr) * nb;
                for (size_t i = 0; i < kb; ++i) {
                    for (size_t t = 0; t < kb; ++t) {
                        Pt[t * kb + i] = A[(r0 + i) * ld + c0 + t];
                        St[t * kb + i] = S[(r0 + i) * ld + c0 + t];
                    }
                }
                for (size_t j = 0; j < kb && flag == 0.0; ++j) {
                    // terms t < tf: product then add; t >= tf: fused multiply-add
                    const size_t tf = fmaStart(k0 + j) - k0;
                    double sum = St[j * kb + j];
                    for (size_t t = 0; t < tf; ++t) sum += Pt[t * kb + j] * Pt[t * kb + j];
                    for (size_t t = tf; t < j; ++t) sum = std::fma(Pt[t * kb + j], Pt[t * kb + j], sum);
                    const double val = Pt[j * kb + j] - sum;
                    if (val <= 0.0) {
                        flag = (double)(k0 + j + 1);
                        break;
                    }
                    const double ljj = sqrt(val);
                    Pt[j * kb + j] = ljj;
                    for (size_t i = j + 1; i < kb; ++i) acc[i] = St[j * kb + i];
                    for (size_t t = 0; t < tf; ++t) {
                        const double l = Pt[t * kb + j];
                        const double* p = &Pt[t * kb];
                        for (size_t i = j + 1; i < kb; ++i) acc[i] += p[i] * l;
                    }
                    for (size_t t = tf; t < j; ++t) {
                        const double l = Pt[t * kb + j];
                        const double* p = &Pt[t * kb];
                        for (size_t i = j + 1; i < kb; ++i) acc[i] = std::fma(p[i], l, acc[i]);
                    }
                    for (size_t i = j + 1; i < kb; ++i) Pt[j * kb + i] = (Pt[j * kb + i] - acc[i]) / ljj;
                }
                for (size_t i = 0; i < kb; ++i) {
                    for (size_t t = 0; t < kb; ++t) {
                        const double v = t <= i ? Pt[t * kb + i] : 0.0;
                        Lkk[i * kb + t] = v;
                        A[(r0 + i) * ld + c0 + t] = v;
                    }
                }
                Lkk[kb * kb] = flag;
            }
            MPI_Bcast(Lkk.data(), (int)(kb * kb + 1), MPI_DOUBLE, prK, g.colComm);
            flag = Lkk[kb * kb];

            if (flag == 0.0) {
                // Solve panel rows below the diagonal: L_ij = (A_ij - sum) / L_jj
                for (size_t rs = 0; rs < nrp; rs += RCH) {
                    const size_t R = std::min(RCH, nrp - rs);
                    const size_t lr0 = li0 * nb + rs;
                    for (size_t r = 0; r < R; ++r)
                        for (size_t t = 0; t < kb; ++t) Pc[t * RCH + r] = A[(lr0 + r) * ld + c0 + t];
                    for (size_t j = 0; j < kb; ++j) {
                        double* a = acc.data();
                        for (size_t r = 0; r < R; ++r) a[r] = S[(lr0 + r) * ld + c0 + j];
                        const size_t tf = fmaStart(k0 + j) - k0;
                        for (size_t t = 0; t < tf; ++t) {
                            const double l = Lkk[j * kb + t];
                            const double* p = &Pc[t * RCH];
                            for (size_t r = 0; r < R; ++r) a[r] += p[r] * l;
                        }
                        for (size_t t = tf; t < j; ++t) {
                            const double l = Lkk[j * kb + t];
                            const double* p = &Pc[t * RCH];
                            for (size_t r = 0; r < R; ++r) a[r] = std::fma(p[r], l, a[r]);
                        }
                        const double ljj = Lkk[j * kb + j];
                        double* p = &Pc[j * RCH];
                        for (size_t r = 0; r < R; ++r) p[r] = (p[r] - a[r]) / ljj;
                    }
                    for (size_t r = 0; r < R; ++r) {
                        for (size_t t = 0; t < kb; ++t) {
                            const double v = Pc[t * RCH + r];
                            A[(lr0 + r) * ld + c0 + t] = v;
                            rowPanel[(rs + r) * kb + t] = v;
                        }
                    }
                }
            }
            rowPanel[nrp * kb] = flag;
        }
        MPI_Ibcast(rowPanel.data(), (int)(nrp * kb + 1), MPI_DOUBLE, pcK, g.rowComm, &rowReq);
        pendingK = K;
        stage = 0;
    };

    // Row broadcast of the pending panel completed: start distributing its rows along
    // process columns
    auto startColumnGather = [&]() {
        const size_t K = pendingK;
        const size_t kb = L.tileSize(K);
        const size_t li0 = countOwnedUpTo(K, g.pr, g.Pr);
        const size_t nrp = (L.mt - li0) * nb;
        stage = 1;
        const double flag = rowPanel[nrp * kb];
        if (flag != 0.0 || K + 1 == L.nt) {
            pendingFlag = (size_t)flag;
            return;
        }
        pendingFlag = 0;

        std::fill(counts.begin(), counts.end(), 0);
        for (size_t J = K + 1; J < L.nt; ++J) {
            if ((int)(J % g.Pc) == g.pc) counts[J % g.Pr] += (int)(nb * kb);
        }
        displs[0] = 0;
        for (int q = 1; q < g.Pr; ++q) displs[q] = displs[q - 1] + counts[q - 1];
        size_t sendCount = 0;
        for (size_t J = K + 1; J < L.nt; ++J) {
            if ((int)(J % g.Pc) == g.pc && (int)(J % g.Pr) == g.pr) {
                std::memcpy(&sendBuf[sendCount], &rowPanel[(J / g.Pr - li0) * nb * kb],
                            nb * kb * sizeof(double));
                sendCount += nb * kb;
            }
        }
        MPI_Iallgatherv(sendBuf.data(), (int)sendCount, MPI_DOUBLE, colPanel.data(), counts.data(),
                        displs.data(), MPI_DOUBLE, g.colComm, &colReq);
    };

    // Advance the pending panel communication without blocking
    auto poll = [&]() {
        int done = 0;
        if (stage == 0) {
            MPI_Test(&rowReq, &done, MPI_STATUS_IGNORE);
            if (done) startColumnGather();
        }
        if (stage == 1) MPI_Test(&colReq, &done, MPI_STATUS_IGNORE);
    };

    // Complete the communication of panel K and pack it for the trailing update.
    // Returns the failure flag.
    auto receivePanel = [&](size_t K) -> size_t {
        const size_t kb = L.tileSize(K);
        const size_t li0 = countOwnedUpTo(K, g.pr, g.Pr);
        const size_t lj0 = countOwnedUpTo(K, g.pc, g.Pc);
        const size_t nrp = (L.mt - li0) * nb;
        if (stage == 0) {
            MPI_Wait(&rowReq, MPI_STATUS_IGNORE);
            startColumnGather();
        }
        MPI_Wait(&colReq, MPI_STATUS_IGNORE);
        stage = 2;
        if (pendingFlag != 0 || K + 1 == L.nt) return pendingFlag;

        std::fill(qIndex.begin(), qIndex.end(), 0);
        for (size_t lj = lj0; lj < L.ntc; ++lj) {
            const size_t J = lj * g.Pc + g.pc;
            const int q = (int)(J % g.Pr);
            colTilePtr[lj] = &colPanel[displs[q] + qIndex[q]++ * nb * kb];
        }

        // Pack panels for the trailing update
        packStrips<MR>(Ap.data(), rowPanel.data(), kb, nrp, 0, kb);
        for (size_t lj = lj0; lj < L.ntc; ++lj) {
            packStrips<NR>(&Bp[(lj - lj0) * nb * kb], colTilePtr[lj], kb, nb, 0, kb);
        }
        return 0;
    };

    // Trailing update with panel K for local column tiles [ljBegin, ljEnd):
    // S_IJ += L_IK * L_JK^T for K < J <= I
    auto update = [&](size_t K, size_t ljBegin, size_t ljEnd, bool polling) {
        const size_t kb = L.tileSize(K);
        const size_t li0 = countOwnedUpTo(K, g.pr, g.Pr);
        const size_t lj0 = countOwnedUpTo(K, g.pc, g.Pc);
        for (size_t lj = ljBegin; lj < ljEnd; ++lj) {
            const size_t J = lj * g.Pc + g.pc;
            const double* Bt = &Bp[(lj - lj0) * nb * kb];
            for (size_t li = std::max(li0, countOwnedUpTo(J - 1, g.pr, g.Pr)); li < L.mt; ++li) {
                const size_t I = li * g.Pr + g.pr;
                for (size_t is = 0; is < nb; is += MR) {
                    const double* At = &Ap[((li - li0) * nb + is) * kb];
                    const size_t jsEnd = (I == J) ? std::min(nb, (is + MR + NR - 1) / NR * NR) : nb;
                    double* C = &S[(li * nb + is) * ld + lj * nb];
                    for (size_t js = 0; js < jsEnd; js += NR) {
                        microKernel((int)kb, At, Bt + js * kb, C + js, ld);
                    }
                }
                if (polling) poll();
            }
        }
    };

    if (L.nt == 0) return 0;
    factorPanel(0);
    size_t failed = receivePanel(0);
    for (size_t K = 0; K + 1 < L.nt && !failed; ++K) {
        const size_t lj0 = countOwnedUpTo(K, g.pc, g.Pc);
        // Lookahead: update the next panel's column first, factor it, and overlap its
        // broadcast with the rest of the trailing update
        const bool ownsNext = (int)((K + 1) % g.Pc) == g.pc;
        if (ownsNext) update(K, lj0, lj0 + 1, false);
        factorPanel(K + 1);
        update(K, ownsNext ? lj0 + 1 : lj0, L.ntc, true);
        failed = receivePanel(K + 1);
    }
    return failed;
}

// Zero out upper triangular part (and anything never factored there)
void zeroUpper(std::vector<double>& A, const Layout& L, const Grid& g) {
    for (size_t lr = 0; lr < L.rows; ++lr) {
        const size_t gi = globalRow(lr, L, g);
        for (size_t lc = 0; lc < L.ntc * L.nb; ++lc) {
            if (globalCol(lc, L, g) > gi) A[lr * L.ld + lc] = 0.0;
        }
    }
}

// Gather full L (n x n row-major) on rank 0
std::vector<double> gatherToRoot(const std::vector<double>& A, const Layout& L, const Grid& g) {
    const int localCount = (int)(L.rows * L.ld);
    std::vector<int> counts(g.size), displs(g.size);
    MPI_Gather(&localCount, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    std::vector<double> recv, full;
    if (g.rank == 0) {
        size_t total = 0;
        for (int q = 0; q < g.size; ++q) {
            displs[q] = (int)total;
            total += counts[q];
        }
        recv.resize(std::max<size_t>(total, 1));
        full.assign(L.n * L.n, 0.0);
    }
    MPI_Gatherv(A.data(), localCount, MPI_DOUBLE, recv.data(), counts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    if (g.rank == 0) {
        for (int q = 0; q < g.size; ++q) {
            Grid gq = g;
            gq.pr = q / g.Pc;
            gq.pc = q % g.Pc;
            const Layout Lq = makeLayout(L.n, L.nb, gq);
            const double* src = recv.data() + displs[q];
            for (size_t lr = 0; lr < Lq.rows; ++lr) {
                const size_t gi = globalRow(lr, Lq, gq);
                if (gi >= L.n) continue;
                for (size_t lc = 0; lc < Lq.ntc * Lq.nb; ++lc) {
                    const size_t gj = globalCol(lc, Lq, gq);
                    if (gj < L.n) full[gi * L.n + gj] = src[lr * Lq.ld + lc];
                }
            }
        }
    }
    return full;
}

bool validateCholesky(const std::vector<double>& Lloc, const std::vector<double>& Aorig, const Layout& L,
                      const Grid& g) {
    // Validate by computing L * L^T and comparing with original matrix
    const size_t n = L.n, nb = L.nb, np = L.nt * nb;

    // Full rows of L for local row tiles (gather along process row)
    std::vector<int> counts(g.Pc), displs(g.Pc);
    for (int q = 0; q < g.Pc; ++q) {
        counts[q] = (int)(L.rows * std::max<size_t>(countOwned(L.nt, q, g.Pc) * nb, NR));
        displs[q] = q ? displs[q - 1] + counts[q - 1] : 0;
    }
    std::vector<double> rowRecv((size_t)displs[g.Pc - 1] + counts[g.Pc - 1] + 1);
    MPI_Allgatherv(Lloc.data(), (int)(L.rows * L.ld), MPI_DOUBLE, rowRecv.data(), counts.data(),
                   displs.data(), MPI_DOUBLE, g.rowComm);
    std::vector<double> Lrows(L.rows * np + 1, 0.0);
    for (int q = 0; q < g.Pc; ++q) {
        const size_t ldq = counts[q] / std::max<size_t>(L.rows, 1);
        const size_t ntq = countOwned(L.nt, q, g.Pc);
        for (size_t lr = 0; lr < L.rows; ++lr) {
            for (size_t lt = 0; lt < ntq; ++lt) {
                std::memcpy(&Lrows[lr * np + (lt * g.Pc + q) * nb], &rowRecv[displs[q] + lr * ldq + lt * nb],
                            nb * sizeof(double));
            }
        }
    }
    rowRecv.clear();
    rowRecv.shrink_to_fit();

    // Full rows of L for local column tiles (gather along process column)
    std::vector<int> ccounts(g.Pr), cdispls(g.Pr);
    std::vector<double> sendBuf(L.rows * np + 1);
    size_t sendCount = 0;
    for (size_t I = 0; I < L.nt; ++I) {
        if ((int)(I % g.Pc) != g.pc) continue;
        const int q = (int)(I % g.Pr);
        if (q == g.pr) {
            std::memcpy(&sendBuf[sendCount], &Lrows[(I / g.Pr) * nb * np], nb * np * sizeof(double));
            sendCount += nb * np;
        }
    }
    for (int q = 0; q < g.Pr; ++q) {
        ccounts[q] = 0;
        for (size_t I = 0; I < L.nt; ++I)
            if ((int)(I % g.Pc) == g.pc && (int)(I % g.Pr) == q) ccounts[q] += (int)(nb * np);
        cdispls[q] = q ? cdispls[q - 1] + ccounts[q - 1] : 0;
    }
    std::vector<double> colRecv((size_t)cdispls[g.Pr - 1] + ccounts[g.Pr - 1] + 1);
    MPI_Allgatherv(sendBuf.data(), (int)sendCount, MPI_DOUBLE, colRecv.data(), ccounts.data(), cdispls.data(),
                   MPI_DOUBLE, g.colComm);
    std::vector<double> Lcols(L.ntc * nb * np + 1, 0.0);
    {
        std::vector<size_t> qi(g.Pr, 0);
        for (size_t lj = 0; lj < L.ntc; ++lj) {
            const size_t J = lj * g.Pc + g.pc;
            const int q = (int)(J % g.Pr);
            std::memcpy(&Lcols[lj * nb * np], &colRecv[cdispls[q] + qi[q]++ * nb * np], nb * np * sizeof(double));
        }
    }

    // Compute L * L^T for local tiles
    std::vector<double> reconstructed(L.rows * L.ld, 0.0);
    const size_t kf = fmaStart(n);
    gemmNT(reconstructed.data(), L.ld, Lrows.data(), np, Lcols.data(), np, L.rows, L.ntc * nb, kf);
    fmaTail(reconstructed.data(), L.ld, Lrows.data(), np, Lcols.data(), np, L.rows, L.ntc * nb, kf, n);

    // Compare with original
    double err[2] = {0.0, 0.0};
    for (size_t lr = 0; lr < L.rows; ++lr) {
        if (globalRow(lr, L, g) >= n) continue;
        for (size_t lc = 0; lc < L.ntc * nb; ++lc) {
            if (globalCol(lc, L, g) >= n) continue;
            const double a = Aorig[lr * L.ld + lc];
            const double error = fabs(reconstructed[lr * L.ld + lc] - a);
            err[0] = std::max(err[0], error);
            const double rel = error / (fabs(a) + 1e-10);
            err[1] = std::max(err[1], rel);
        }
    }
    double gerr[2];
    MPI_Allreduce(err, gerr, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    const double maxError = gerr[0], relError = gerr[1];

    if (g.rank == 0) {
        printf("Max absolute error: %.10e\n", maxError);
        printf("Max relative error: %.10e\n", relError);
    }

    // Check if error is within tolerance
    if (relError > 1e-6) {
        if (g.rank == 0) printf("Validation failed: relative error too large\n");
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

int run(int argc, char** argv, const Grid& g) {
    size_t n = 512;
    bool validate = false;
    bool printResults = false;
    const bool root = g.rank == 0;

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

    // Tile size: multiple of the micro-kernel dimensions, shrunk when there are few tiles
    // per process (CHOLESKY_NB overrides the default)
    size_t nb = 120;
    if (const char* env = getenv("CHOLESKY_NB")) nb = std::max<size_t>(1, atoi(env));
    const size_t perDim = std::max(g.Pr, g.Pc);
    while (nb > TILE_ALIGN && n < nb * perDim * 12) nb -= TILE_ALIGN;
    nb = std::max<size_t>(TILE_ALIGN, nb / TILE_ALIGN * TILE_ALIGN);

    if (root) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const Layout L = makeLayout(n, nb, g);

    // Allocate and generate positive definite matrix (local tiles)
    std::vector<double> A;
    std::vector<double> A_orig;
    if (root) printf("Generating positive definite matrix...\n");
    generatePositiveDefiniteMatrix(A, L, g);

    if (validate) {
        A_orig = A;  // Save original for validation
    }

    // Perform Cholesky decomposition
    if (root) printf("Computing Cholesky decomposition...\n");
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    const size_t failed = choleskyDecomposition(A, L, g);
    MPI_Barrier(MPI_COMM_WORLD);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (failed) {
        if (root) {
            printf("Error: Matrix is not positive definite at diagonal element %zu\n", failed - 1);
            printf("Cholesky decomposition failed\n");
        }
        return 1;
    }
    zeroUpper(A, L, g);

    if (root) {
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Print results for external validation
    if (printResults) {
        std::vector<double> full = gatherToRoot(A, L, g);
        if (root) print_results(full, "CholeskyL");
    }

    // Validation
    if (validate) {
        if (root) printf("Validating result...\n");
        bool valid = validateCholesky(A, A_orig, L, g);

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
    Grid g = makeGrid();
    const int ret = run(argc, argv, g);
    fflush(stdout);
    MPI_Comm_free(&g.rowComm);
    MPI_Comm_free(&g.colComm);
    MPI_Finalize();
    return ret;
}
