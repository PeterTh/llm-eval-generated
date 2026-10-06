#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#ifdef __FMA__
#include <immintrin.h>
#endif

#include "../common/results_output.hpp"

// Distributed Cholesky decomposition (MPI, 2D block-cyclic, right-looking with lookahead)
// Decomposes positive definite matrix A into L * L^T where L is lower triangular.
//
// Every element L[i][j] is computed with exactly the same floating point operation
// sequence as the sequential unblocked algorithm: the dot product over k is accumulated
// in ascending k order starting from 0.0 (partial sums are carried in the tile storage
// between steps), then (A - sum) / L[j][j] resp. sqrt(A - sum).
// The reference build (GCC, -O3 -march=native) vectorizes these dot products as an
// in-order reduction: products of the first VW*floor(K/VW) terms are rounded before being
// added, while the remaining scalar tail terms are contracted to FMA (when the target has
// FMA). This file is compiled with -ffp-contract=off and reproduces that split explicitly,
// so the result is bitwise identical to the sequential code.

typedef double v4d __attribute__((vector_size(32)));
typedef double v4du __attribute__((vector_size(32), aligned(8)));

constexpr int VW = 4;  // vector width of the reference reduction
static inline size_t fusedStart(size_t K) { return K / VW * VW; }

static inline double fmaS(double a, double b, double c) {
#ifdef __FMA__
    return std::fma(a, b, c);
#else
    return a * b + c;
#endif
}

static inline v4d fmaV(v4d a, v4d b, v4d c) {
#ifdef __FMA__
    return (v4d)_mm256_fmadd_pd((__m256d)a, (__m256d)b, (__m256d)c);
#else
    return a * b + c;
#endif
}

static inline v4d ld4(const double* p) { return *reinterpret_cast<const v4du*>(p); }
static inline void st4(double* p, v4d v) { *reinterpret_cast<v4du*>(p) = v; }

// ---------------------------------------------------------------------------------------
// Compute kernels
// ---------------------------------------------------------------------------------------

constexpr int MR = 6;
constexpr int NR = 8;

// Packed A layout ("micro panels"): row r, column k of an M x K block is stored at
// Ap[(r / MR) * aps + k * MR + r % MR], where aps (panel stride) >= K * MR.
static void packRows(double* Ap, int aps, const double* A, int lda, int M, int K) {
    for (int ir = 0; ir < M; ir += MR) {
        double* dst = Ap + (size_t)(ir / MR) * aps;
        for (int r = 0; r < MR; ++r) {
            const double* src = A + (size_t)(ir + r) * lda;
            for (int k = 0; k < K; ++k) dst[(size_t)k * MR + r] = src[k];
        }
    }
}

// C[0..6)[0..8) += sum_k A[r][k] * BT[k][c], k ascending (per element sequential chain of
// rounded products, as in the vectorized reference reduction). A is a packed micro panel.
static inline void micro6x8(int K, const double* __restrict A, const double* __restrict BT, int ldb,
                            double* __restrict C, int ldc) {
    v4d c00 = ld4(C + 0 * ldc), c01 = ld4(C + 0 * ldc + 4);
    v4d c10 = ld4(C + 1 * ldc), c11 = ld4(C + 1 * ldc + 4);
    v4d c20 = ld4(C + 2 * ldc), c21 = ld4(C + 2 * ldc + 4);
    v4d c30 = ld4(C + 3 * ldc), c31 = ld4(C + 3 * ldc + 4);
    v4d c40 = ld4(C + 4 * ldc), c41 = ld4(C + 4 * ldc + 4);
    v4d c50 = ld4(C + 5 * ldc), c51 = ld4(C + 5 * ldc + 4);
    for (int k = 0; k < K; ++k) {
        const v4d b0 = ld4(BT + (size_t)k * ldb);
        const v4d b1 = ld4(BT + (size_t)k * ldb + 4);
        const double* ak = A + (size_t)k * MR;
        double a;
        a = ak[0]; c00 += a * b0; c01 += a * b1;
        a = ak[1]; c10 += a * b0; c11 += a * b1;
        a = ak[2]; c20 += a * b0; c21 += a * b1;
        a = ak[3]; c30 += a * b0; c31 += a * b1;
        a = ak[4]; c40 += a * b0; c41 += a * b1;
        a = ak[5]; c50 += a * b0; c51 += a * b1;
    }
    st4(C + 0 * ldc, c00); st4(C + 0 * ldc + 4, c01);
    st4(C + 1 * ldc, c10); st4(C + 1 * ldc + 4, c11);
    st4(C + 2 * ldc, c20); st4(C + 2 * ldc + 4, c21);
    st4(C + 3 * ldc, c30); st4(C + 3 * ldc + 4, c31);
    st4(C + 4 * ldc, c40); st4(C + 4 * ldc + 4, c41);
    st4(C + 5 * ldc, c50); st4(C + 5 * ldc + 4, c51);
}

// C (M x N, ldc) += A (M x K, packed, panel stride aps) * BT (K x N, ldb); M % 6 == 0, N % 8 == 0.
// If lowerOnly, micro tiles entirely above the diagonal are skipped.
static void gemmTile(int M, int N, int K, const double* Ap, int aps, const double* BT, int ldb, double* C, int ldc,
                     bool lowerOnly) {
    if (K <= 0) return;
    for (int ir = 0; ir < M; ir += MR) {
        const int jend = lowerOnly ? std::min(N, ir + MR) : N;
        for (int jc = 0; jc < jend; jc += NR)
            micro6x8(K, Ap + (size_t)(ir / MR) * aps, BT + jc, ldb, C + (size_t)ir * ldc + jc, ldc);
    }
}

// Fused tail: C[r][c] = fma(A[r][k], BT[k][c], C[r][c]) for k in [k0, k1), ascending (A packed)
static void fusedTail(int M, int N, int k0, int k1, const double* Ap, int aps, const double* BT, int ldb, double* C,
                      int ldc) {
    for (int k = k0; k < k1; ++k)
        for (int r = 0; r < M; ++r) {
            const double a = Ap[(size_t)(r / MR) * aps + (size_t)k * MR + r % MR];
            for (int c = 0; c < N; ++c)
                C[(size_t)r * ldc + c] = fmaS(a, BT[(size_t)k * ldb + c], C[(size_t)r * ldc + c]);
        }
}

static inline int roundUp(int x, int m) { return (x + m - 1) / m * m; }

// Finalize one tile of column block k.
//   L     : nb x nb row-major tile, on input holds the accumulated partial sums S, on output L
//   A     : nb x nb row-major original matrix tile
//   D     : finalized diagonal tile L(k,k) (row-major), ignored if isDiag
//   rows  : valid rows of the tile, cols: valid columns
//   W, AT : nb x nb scratch
// Returns the local column of a non positive definite diagonal element, or -1.
static int factorTile(double* L, const double* A, const double* D, int rows, int cols, bool isDiag, int nb,
                      double* W, double* AT) {
    constexpr int FR = 48, FV = FR / 4;  // rows per register block
    const int Mr = roundUp(rows, FR);
    double lrow[512];
    // Column-major copies: W[c*nb + r] = S[r][c]
    for (int r = 0; r < Mr; ++r) {
        for (int c = 0; c < cols; ++c) {
            W[(size_t)c * nb + r] = L[(size_t)r * nb + c];
            AT[(size_t)c * nb + r] = A[(size_t)r * nb + c];
        }
    }
    for (int c = 0; c < cols; ++c) {
        double* Wc = W + (size_t)c * nb;
        // Tile columns start at multiples of nb (nb % VW == 0), so the local split equals the global one
        const int kf = (int)fusedStart(c);
        double d;
        int r0 = 0;
        if (isDiag) {
            double s = Wc[c];
            for (int kk = 0; kk < kf; ++kk) {
                const double v = W[(size_t)kk * nb + c];
                s += v * v;
            }
            for (int kk = kf; kk < c; ++kk) {
                const double v = W[(size_t)kk * nb + c];
                s = fmaS(v, v, s);
            }
            const double val = AT[(size_t)c * nb + c] - s;
            if (val <= 0.0) return c;
            d = sqrt(val);
            r0 = (c + 1) / FR * FR;
        } else {
            d = D[(size_t)c * nb + c];
        }
        // Row c of L (columns < c), contiguous
        for (int kk = 0; kk < c; ++kk) lrow[kk] = isDiag ? W[(size_t)kk * nb + c] : D[(size_t)c * nb + kk];
        const v4d dv = {d, d, d, d};
        for (int rb = r0; rb < Mr; rb += FR) {
            v4d acc[FV];
            for (int v = 0; v < FV; ++v) acc[v] = ld4(Wc + rb + 4 * v);
            for (int kk = 0; kk < kf; ++kk) {
                const double lk = lrow[kk];
                const double* Wk = W + (size_t)kk * nb + rb;
                for (int v = 0; v < FV; ++v) acc[v] += ld4(Wk + 4 * v) * lk;
            }
            for (int kk = kf; kk < c; ++kk) {
                const double lk = lrow[kk];
                const v4d lv = {lk, lk, lk, lk};
                const double* Wk = W + (size_t)kk * nb + rb;
                for (int v = 0; v < FV; ++v) acc[v] = fmaV(ld4(Wk + 4 * v), lv, acc[v]);
            }
            const double* Ac = AT + (size_t)c * nb + rb;
            for (int v = 0; v < FV; ++v) st4(Wc + rb + 4 * v, (ld4(Ac + 4 * v) - acc[v]) / dv);
        }
        if (isDiag) {
            for (int r = 0; r < c; ++r) Wc[r] = 0.0;
            Wc[c] = d;
        }
    }
    for (int r = 0; r < Mr; ++r) {
        double* Lr = L + (size_t)r * nb;
        for (int c = 0; c < cols; ++c) Lr[c] = W[(size_t)c * nb + r];
        for (int c = cols; c < nb; ++c) Lr[c] = 0.0;
    }
    return -1;
}

// ---------------------------------------------------------------------------------------
// Distributed matrix
// ---------------------------------------------------------------------------------------

struct Dist {
    size_t n = 0;
    int nb = 0, nt = 0;
    size_t nb2 = 0;
    int rank = 0, size = 1;
    int P = 1, Q = 1, pr = 0, pc = 0;
    MPI_Comm rowComm, colComm;  // rowComm: same pr (rank = pc); colComm: same pc (rank = pr)

    struct Tile {
        int i, j;
    };
    std::vector<Tile> tiles;   // local lower tiles, sorted by j then i
    std::vector<int> tileIdx;  // [(i/P) * nlc + j/Q] -> local tile index or -1
    int nlr = 0, nlc = 0;
    std::vector<double> Lst;  // partial sums / L
    std::vector<double> Ast;  // original A

    int dim(int t) const { return (int)std::min<size_t>(nb, n - (size_t)t * nb); }
    int idx(int i, int j) const { return tileIdx[(size_t)(i / P) * nlc + j / Q]; }
    double* Lt(int t) { return Lst.data() + (size_t)t * nb2; }
    double* At(int t) { return Ast.data() + (size_t)t * nb2; }
    int owner(int i, int j) const { return (i % P) * Q + (j % Q); }

    void init(size_t n_, int rank_, int size_) {
        n = n_;
        rank = rank_;
        size = size_;
        P = 1;
        for (int p = 1; (long)p * p <= size; ++p)
            if (size % p == 0) P = p;
        Q = size / P;
        pr = rank / Q;
        pc = rank % Q;
        MPI_Comm_split(MPI_COMM_WORLD, pr, pc, &rowComm);
        MPI_Comm_split(MPI_COMM_WORLD, pc, pr, &colComm);

        // Largest tile size that still leaves enough tile columns per process column for
        // load balance and pipelining (nb must be a multiple of 48)
        const int cand[] = {144, 96, 48};
        nb = 48;
        for (int c : cand) {
            if ((size_t)(n + c - 1) / c >= (size_t)12 * Q) {
                nb = c;
                break;
            }
        }
        if (size == 1) nb = 144;
        nb2 = (size_t)nb * nb;
        nt = (int)((n + nb - 1) / nb);

        nlr = (nt - pr + P - 1) / P;
        nlc = (nt - pc + Q - 1) / Q;
        if (nlr < 0) nlr = 0;
        if (nlc < 0) nlc = 0;
        tileIdx.assign((size_t)nlr * nlc, -1);
        for (int j = pc; j < nt; j += Q)
            for (int i = pr; i < nt; i += P)
                if (i >= j) {
                    tileIdx[(size_t)(i / P) * nlc + j / Q] = (int)tiles.size();
                    tiles.push_back({i, j});
                }
        Lst.assign(tiles.size() * nb2, 0.0);
        Ast.assign(tiles.size() * nb2, 0.0);
    }
};

// Distribution of one tile column k: every rank receives the tiles (i,k) for its tile rows
// (row broadcast) and the tiles (j,k) for its tile columns (column allgather, stored transposed).
struct Panel {
    std::vector<double> rowbuf, rowPk, sendbuf, colrecv, colT;
    std::vector<int> rowSlot, colSlot;
    std::vector<int> counts, displs;
    MPI_Request req = MPI_REQUEST_NULL;
    int k = -1, first = 0, nr = 0;

    void alloc(const Dist& D) {
        const int maxR = D.nlr, maxC = D.nlc;
        rowbuf.resize((size_t)maxR * D.nb2 + 1);
        rowPk.resize((size_t)std::max(1, maxR) * D.nb2);
        sendbuf.resize((size_t)std::max(1, std::min(maxR, maxC)) * D.nb2);
        colrecv.resize((size_t)std::max(1, maxC) * D.nb2);
        colT.resize((size_t)std::max(1, maxC) * D.nb2);
        rowSlot.assign(D.nt, -1);
        colSlot.assign(D.nt, -1);
        counts.resize(D.P);
        displs.resize(D.P);
    }
    // Row tile i in packed micro panel layout (panel stride nb * MR)
    const double* row(int i) const { return rowPk.data() + (size_t)rowSlot[i] * nbsq; }
    const double* rowRaw(int i) const { return rowbuf.data() + (size_t)rowSlot[i] * nbsq; }
    const double* col(int j) const { return colT.data() + (size_t)colSlot[j] * nbsq; }
    size_t nbsq = 0;

    // Post the row broadcast of column k (root column k%Q packs its tiles plus a status word).
    void start(Dist& D, int k_, bool includeDiag, double status) {
        k = k_;
        nbsq = D.nb2;
        first = includeDiag ? k : k + 1;
        nr = 0;
        std::fill(rowSlot.begin(), rowSlot.end(), -1);
        int i0 = D.pr;
        if (i0 < first) i0 += (first - i0 + D.P - 1) / D.P * D.P;
        for (int i = i0; i < D.nt; i += D.P) rowSlot[i] = nr++;
        const bool root = (D.pc == k % D.Q);
        if (root) {
            for (int i = i0; i < D.nt; i += D.P)
                memcpy(rowbuf.data() + (size_t)rowSlot[i] * D.nb2, D.Lt(D.idx(i, k)), D.nb2 * sizeof(double));
            rowbuf[(size_t)nr * D.nb2] = status;
        }
        stage = 0;
        MPI_Ibcast(rowbuf.data(), (int)((size_t)nr * D.nb2 + 1), MPI_DOUBLE, k % D.Q, D.rowComm, &req);
    }

    // Progress: 0 = row broadcast pending, 1 = column allgather pending, 2 = done
    int stage = 2;
    double status = -1.0;

    // Row broadcast complete: pack row tiles and post the column exchange.
    void rowArrived(Dist& D) {
        status = rowbuf[(size_t)nr * D.nb2];
        if (status >= 0.0) {
            stage = 2;
            return;
        }
        // Contribution: tiles j >= first with j%P == pr and j%Q == pc
        int ns = 0;
        for (int j = D.pc; j < D.nt; j += D.Q)
            if (j >= first && j % D.P == D.pr)
                memcpy(sendbuf.data() + (size_t)(ns++) * D.nb2, rowRaw(j), D.nb2 * sizeof(double));
        std::fill(colSlot.begin(), colSlot.end(), -1);
        int slot = 0;
        for (int p = 0; p < D.P; ++p) {
            int cnt = 0;
            for (int j = D.pc; j < D.nt; j += D.Q)
                if (j >= first && j % D.P == p) colSlot[j] = slot + cnt++;
            displs[p] = (int)((size_t)slot * D.nb2);
            counts[p] = (int)((size_t)cnt * D.nb2);
            slot += cnt;
        }
        ncol = slot;
        if (D.P > 1) {
            MPI_Iallgatherv(sendbuf.data(), (int)((size_t)ns * D.nb2), MPI_DOUBLE, colrecv.data(), counts.data(),
                            displs.data(), MPI_DOUBLE, D.colComm, &req);
            stage = 1;
        } else {
            memcpy(colrecv.data(), sendbuf.data(), (size_t)ns * D.nb2 * sizeof(double));
            stage = 2;
        }
        const int nb = D.nb;
        for (int s = 0; s < nr; ++s)
            packRows(rowPk.data() + (size_t)s * D.nb2, nb * MR, rowbuf.data() + (size_t)s * D.nb2, nb, nb, nb);
    }
    int ncol = 0;

    // Drive the communication without blocking
    void test(Dist& D) {
        int flag = 0;
        if (stage == 0) {
            MPI_Test(&req, &flag, MPI_STATUS_IGNORE);
            if (flag) rowArrived(D);
        } else if (stage == 1) {
            MPI_Test(&req, &flag, MPI_STATUS_IGNORE);
            if (flag) stage = 2;
        }
    }

    // Complete the row broadcast and the column exchange. Returns the status word.
    double finish(Dist& D) {
        if (stage == 0) {
            MPI_Wait(&req, MPI_STATUS_IGNORE);
            rowArrived(D);
        }
        if (stage == 1) {
            MPI_Wait(&req, MPI_STATUS_IGNORE);
            stage = 2;
        }
        if (status >= 0.0) return status;
        const int nb = D.nb;
        for (int s = 0; s < ncol; ++s) {
            const double* src = colrecv.data() + (size_t)s * D.nb2;
            double* dst = colT.data() + (size_t)s * D.nb2;
            for (int r = 0; r < nb; ++r)
                for (int c = 0; c < nb; ++c) dst[(size_t)c * nb + r] = src[(size_t)r * nb + c];
        }
        return status;
    }
};

// Generate a symmetric positive definite matrix (local tiles only)
// Method: A = B * B^T + n * I where B is random (rand_r stream with seed 42, row-major).

// Replica of the glibc rand_r generator, used to jump ahead in the stream. Only used if it
// is verified to reproduce rand_r exactly on this system.
static inline int randr_replica(unsigned int* seed) {
    unsigned int next = *seed;
    int result;
    next *= 1103515245;
    next += 12345;
    result = (unsigned int)(next / 65536) % 2048;
    next *= 1103515245;
    next += 12345;
    result <<= 10;
    result ^= (unsigned int)(next / 65536) % 1024;
    next *= 1103515245;
    next += 12345;
    result <<= 10;
    result ^= (unsigned int)(next / 65536) % 1024;
    *seed = next;
    return result;
}

static bool replicaMatches() {
    const unsigned int seeds[] = {42u, 1u, 0u, 123456789u, 0xffffffffu};
    for (unsigned int s0 : seeds) {
        unsigned int a = s0, b = s0;
        for (int i = 0; i < 4096; ++i) {
            if (rand_r(&a) != randr_replica(&b) || a != b) return false;
        }
    }
    return true;
}

// State after 'calls' replica calls starting from 'seed' (3 LCG steps per call).
static unsigned int jumpSeed(unsigned int seed, uint64_t calls) {
    uint32_t am = 1, cm = 0;            // accumulated affine map
    uint32_t a = 1103515245u, c = 12345u;  // map to apply (power of two steps)
    uint64_t steps = calls * 3;
    while (steps) {
        if (steps & 1) {
            cm = a * cm + c;
            am = a * am;
        }
        c = a * c + c;
        a = a * a;
        steps >>= 1;
    }
    return am * seed + cm;
}

void generatePositiveDefiniteMatrix(Dist& D) {
    const size_t n = D.n;
    const int nb = D.nb, nt = D.nt;
    // Needed tile rows of B: my tile rows (i%P==pr, stored packed) and my tile columns
    // (j%Q==pc, stored transposed: BT[k][c])
    std::vector<int> pslot(nt, -1), tslot(nt, -1);
    int np = 0, ntr = 0;
    for (int t = 0; t < nt; ++t) {
        if (t % D.P == D.pr) pslot[t] = np++;
        if (t % D.Q == D.pc) tslot[t] = ntr++;
    }
    const int aps = (int)n * MR;
    std::vector<double> Bp((size_t)np * nb * n, 0.0), BT((size_t)ntr * nb * n, 0.0);
    std::vector<double> row(n);
    auto store = [&](size_t gi) {
        const int t = (int)(gi / nb), r = (int)(gi % nb);
        if (pslot[t] >= 0) {
            double* dst = Bp.data() + (size_t)pslot[t] * nb * n + (size_t)(r / MR) * aps + r % MR;
            for (size_t k = 0; k < n; ++k) dst[k * MR] = row[k];
        }
        if (tslot[t] >= 0) {
            double* dst = BT.data() + (size_t)tslot[t] * nb * n + r;
            for (size_t k = 0; k < n; ++k) dst[k * nb] = row[k];
        }
    };
    if (replicaMatches()) {
        for (int t = 0; t < nt; ++t) {
            if (pslot[t] < 0 && tslot[t] < 0) continue;
            for (size_t gi = (size_t)t * nb; gi < std::min(n, (size_t)(t + 1) * nb); ++gi) {
                unsigned int seed = jumpSeed(42u, (uint64_t)gi * n);
                for (size_t k = 0; k < n; ++k) row[k] = (randr_replica(&seed) / (double)RAND_MAX) - 0.5;
                store(gi);
            }
        }
    } else {
        unsigned int seed = 42;
        for (size_t gi = 0; gi < n; ++gi) {
            const int t = (int)(gi / nb);
            if (pslot[t] >= 0 || tslot[t] >= 0) {
                for (size_t k = 0; k < n; ++k) row[k] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
                store(gi);
            } else {
                for (size_t k = 0; k < n; ++k) rand_r(&seed);
            }
        }
    }

    // A tile (i,j) = sum_k B[i][k] * B[j][k], k ascending
    const size_t kc = 256;
    const size_t kfG = fusedStart(n);
    for (size_t t = 0; t < D.tiles.size(); ++t) {
        const int ti = D.tiles[t].i, tj = D.tiles[t].j;
        double* C = D.At(t);
        const int M = roundUp(D.dim(ti), MR), N = roundUp(D.dim(tj), NR);
        const double* Bi = Bp.data() + (size_t)pslot[ti] * nb * n;
        const double* Bj = BT.data() + (size_t)tslot[tj] * nb * n;
        for (size_t k0 = 0; k0 < n; k0 += kc) {
            const int K = (int)std::min(kc, n - k0);
            const int Kp = (int)std::min<size_t>(K, kfG > k0 ? kfG - k0 : 0);  // rounded-product part
            gemmTile(M, N, Kp, Bi + k0 * MR, aps, Bj + k0 * nb, nb, C, nb, ti == tj);
            fusedTail(M, N, Kp, K, Bi + k0 * MR, aps, Bj + k0 * nb, nb, C, nb);
        }
        if (ti == tj) {
            for (int r = 0; r < D.dim(ti); ++r) C[(size_t)r * nb + r] += n;
        }
    }
}

// Distributed decomposition. Returns -1 on success or the global index of the first
// non positive definite diagonal element.
long choleskyDecomposition(Dist& D) {
    const int nt = D.nt, nb = D.nb;
    if (nt == 0) return -1;
    Panel pan[2];
    pan[0].alloc(D);
    pan[1].alloc(D);
    std::vector<double> W(D.nb2), AT(D.nb2), diagbuf(D.nb2 + 1);

    // Factor tile column k (only ranks of process column k%Q work). Returns status.
    auto panelFactor = [&](int k) -> double {
        if (D.pc != k % D.Q) return -1.0;
        const bool diagOwner = (D.pr == k % D.P);
        if (diagOwner) {
            const int t = D.idx(k, k);
            const int f = factorTile(D.Lt(t), D.At(t), nullptr, D.dim(k), D.dim(k), true, nb, W.data(), AT.data());
            diagbuf[D.nb2] = (f < 0) ? -1.0 : (double)((size_t)k * nb + f);
            if (D.P > 1) memcpy(diagbuf.data(), D.Lt(t), D.nb2 * sizeof(double));
        }
        if (D.P > 1) MPI_Bcast(diagbuf.data(), (int)D.nb2 + 1, MPI_DOUBLE, k % D.P, D.colComm);
        const double status = diagbuf[D.nb2];
        if (status >= 0.0) return status;
        const double* Ld = diagOwner ? D.Lt(D.idx(k, k)) : diagbuf.data();
        int i0 = D.pr;
        if (i0 <= k) i0 += (k + 1 - i0 + D.P - 1) / D.P * D.P;
        for (int i = i0; i < nt; i += D.P) {
            const int t = D.idx(i, k);
            factorTile(D.Lt(t), D.At(t), Ld, D.dim(i), D.dim(k), false, nb, W.data(), AT.data());
        }
        return -1.0;
    };

    auto updateTile = [&](size_t t, const Panel& p) {
        const int i = D.tiles[t].i, j = D.tiles[t].j;
        gemmTile(roundUp(D.dim(i), MR), roundUp(D.dim(j), NR), D.dim(p.k), p.row(i), nb * MR, p.col(j), nb, D.Lt(t),
                 nb, i == j);
    };

    // Tiles are sorted by column; colStart[j] = first local tile index with column >= j
    std::vector<size_t> colStart(nt + 1, D.tiles.size());
    for (size_t t = D.tiles.size(); t-- > 0;) colStart[D.tiles[t].j] = t;
    for (int j = nt - 1; j >= 0; --j) colStart[j] = std::min(colStart[j], colStart[j + 1]);

    double status = panelFactor(0);
    pan[0].start(D, 0, false, status);
    status = pan[0].finish(D);
    for (int k = 0; k < nt && status < 0.0; ++k) {
        Panel& cur = pan[k & 1];
        Panel& nxt = pan[(k + 1) & 1];
        size_t restBegin = colStart[k + 1];
        if (k + 1 < nt) {
            // Lookahead: update and factor the next tile column first
            const size_t e = colStart[k + 2 <= nt ? k + 2 : nt];
            for (size_t t = colStart[k + 1]; t < e; ++t) updateTile(t, cur);
            restBegin = e;
            const double st = panelFactor(k + 1);
            nxt.start(D, k + 1, false, st);
        }
        for (size_t t = restBegin; t < D.tiles.size(); ++t) {
            updateTile(t, cur);
            nxt.test(D);
        }
        if (k + 1 < nt) status = nxt.finish(D);
    }
    return status < 0.0 ? -1 : (long)status;
}

bool validateCholesky(Dist& D) {
    // Validate by computing L * L^T (local lower tiles) and comparing with original matrix.
    // Both matrices are exactly symmetric, so the lower triangle covers all errors.
    const int nt = D.nt, nb = D.nb;
    std::vector<double> R(D.tiles.size() * D.nb2, 0.0);
    const size_t kfG = fusedStart(D.n);
    Panel p;
    p.alloc(D);
    for (int kb = 0; kb < nt; ++kb) {
        p.start(D, kb, true, -1.0);
        p.finish(D);
        for (size_t t = 0; t < D.tiles.size(); ++t) {
            const int i = D.tiles[t].i, j = D.tiles[t].j;
            if (j < kb) continue;
            const int M = roundUp(D.dim(i), MR), N = roundUp(D.dim(j), NR), K = D.dim(kb);
            const size_t kbeg = (size_t)kb * nb;
            const int Kp = (int)std::min<size_t>(K, kfG > kbeg ? kfG - kbeg : 0);
            gemmTile(M, N, Kp, p.row(i), nb * MR, p.col(j), nb, R.data() + t * D.nb2, nb, i == j);
            fusedTail(M, N, Kp, K, p.row(i), nb * MR, p.col(j), nb, R.data() + t * D.nb2, nb);
        }
    }

    double err[2] = {0.0, 0.0};
    for (size_t t = 0; t < D.tiles.size(); ++t) {
        const int i = D.tiles[t].i, j = D.tiles[t].j;
        const double* Rt = R.data() + t * D.nb2;
        const double* At = D.At(t);
        for (int r = 0; r < D.dim(i); ++r) {
            const int cmax = (i == j) ? r + 1 : D.dim(j);
            for (int c = 0; c < cmax; ++c) {
                const double a = At[(size_t)r * nb + c];
                const double error = fabs(Rt[(size_t)r * nb + c] - a);
                err[0] = std::max(err[0], error);
                const double rel = error / (fabs(a) + 1e-10);
                err[1] = std::max(err[1], rel);
            }
        }
    }
    double gerr[2];
    MPI_Reduce(err, gerr, 2, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    int ok = 1;
    if (D.rank == 0) {
        const double maxError = gerr[0];
        const double relError = gerr[1];
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

// Collect the full L (upper part zero) on rank 0
std::vector<double> gatherL(Dist& D) {
    const size_t n = D.n;
    const int nb = D.nb;
    std::vector<double> full;
    if (D.rank == 0) full.assign(n * n, 0.0);
    auto tilesOf = [&](int r) {
        std::vector<std::pair<int, int>> v;
        const int rp = r / D.Q, rc = r % D.Q;
        for (int j = rc; j < D.nt; j += D.Q)
            for (int i = rp; i < D.nt; i += D.P)
                if (i >= j) v.push_back({i, j});
        return v;
    };
    auto tileSize = [&](int i, int j) { return (size_t)D.dim(i) * D.dim(j); };
    if (D.rank != 0) {
        size_t cnt = 0;
        for (auto& t : D.tiles) cnt += tileSize(t.i, t.j);
        std::vector<double> buf(cnt);
        size_t o = 0;
        for (size_t t = 0; t < D.tiles.size(); ++t) {
            const int i = D.tiles[t].i, j = D.tiles[t].j;
            const double* L = D.Lt(t);
            for (int r = 0; r < D.dim(i); ++r)
                for (int c = 0; c < D.dim(j); ++c) buf[o++] = L[(size_t)r * nb + c];
        }
        MPI_Send(buf.data(), (int)cnt, MPI_DOUBLE, 0, 7, MPI_COMM_WORLD);
    } else {
        std::vector<double> buf;
        for (int src = 0; src < D.size; ++src) {
            auto tl = tilesOf(src);
            size_t cnt = 0;
            for (auto& t : tl) cnt += tileSize(t.first, t.second);
            const double* data;
            if (src == 0) {
                buf.resize(cnt);
                size_t o = 0;
                for (size_t t = 0; t < D.tiles.size(); ++t) {
                    const double* L = D.Lt(t);
                    for (int r = 0; r < D.dim(D.tiles[t].i); ++r)
                        for (int c = 0; c < D.dim(D.tiles[t].j); ++c) buf[o++] = L[(size_t)r * nb + c];
                }
            } else {
                buf.resize(cnt);
                MPI_Recv(buf.data(), (int)cnt, MPI_DOUBLE, src, 7, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            data = buf.data();
            size_t o = 0;
            for (auto& t : tl) {
                for (int r = 0; r < D.dim(t.first); ++r)
                    for (int c = 0; c < D.dim(t.second); ++c)
                        full[((size_t)t.first * nb + r) * n + (size_t)t.second * nb + c] = data[o++];
            }
        }
    }
    return full;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

static int run(int argc, char** argv, int rank, int size) {
    size_t n = 512;
    bool validate = false;
    bool printResults = false;
    const bool root = (rank == 0);

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

    // Allocate distributed matrix
    Dist D;
    D.init(n, rank, size);

    // Generate positive definite matrix (original A is kept in the local tiles for validation)
    if (root) printf("Generating positive definite matrix...\n");
    generatePositiveDefiniteMatrix(D);
    // Partial sums start at zero
    std::fill(D.Lst.begin(), D.Lst.end(), 0.0);

    // Perform Cholesky decomposition
    if (root) {
        printf("Computing Cholesky decomposition...\n");
        fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    const long failIdx = choleskyDecomposition(D);
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

    // Print results for external validation
    if (printResults) {
        std::vector<double> A = gatherL(D);
        if (root) print_results(A, "CholeskyL");
    }

    // Validation
    if (validate) {
        if (root) printf("Validating result...\n");
        bool valid = validateCholesky(D);

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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const int ret = run(argc, argv, rank, size);
    fflush(stdout);
    MPI_Finalize();
    return ret;
}
