#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// MPI-parallel Cholesky decomposition (right-looking, blocked/tiled algorithm)
// Decomposes positive definite matrix A into L * L^T where L is lower triangular
//
// Data distribution: the matrix is split into b x b tiles which are distributed
// over a Pr x Pc process grid in 2D block-cyclic fashion: tile (I, J) lives on
// process (I % Pr, J % Pc). Each process stores its tiles contiguously
// (row-major tile grid, row-major inside a tile). The matrix is padded with an
// identity block up to a multiple of b, which leaves the factor unchanged.
//
// Per tile column k:
//   1. the owner of tile (k,k) factors it and broadcasts L_kk down its process column
//   2. the processes of that column compute their panel tiles L_ik = A_ik * L_kk^-T
//   3. panel tiles are broadcast along process rows (each process gets the L_ik
//      for its tile rows) and all-gathered along process columns (each process
//      gets the L_jk for its tile columns)
//   4. every process updates its trailing tiles: A_ij -= L_ik * L_jk^T
// Step 4 dominates the flop count and is perfectly parallel; the 2D
// distribution keeps the per-process communication volume at O(n^2 / sqrt(p)).

#if defined(__AVX512F__)
#define VL 8
#elif defined(__AVX__)
#define VL 4
#else
#define VL 2
#endif

typedef double vdouble __attribute__((vector_size(VL * sizeof(double))));

static inline vdouble vload(const double* p) {
    vdouble v;
    memcpy(&v, p, sizeof(v));
    return v;
}

static inline void vstore(double* p, const vdouble& v) {
    memcpy(p, &v, sizeof(v));
}

// Process grid and tile layout
struct Layout {
    size_t n;         // matrix size
    size_t b;         // tile size
    size_t nblocks;   // number of tile rows/columns
    size_t npad;      // padded matrix size (nblocks * b)
    int p, rank;
    int Pr, Pc;       // process grid
    int ir, jc;       // my grid coordinates
    size_t nLocRows;  // my number of tile rows
    size_t nLocCols;  // my number of tile columns
    MPI_Comm rowComm; // processes with the same ir (rank in rowComm == jc)
    MPI_Comm colComm; // processes with the same jc (rank in colComm == ir)

    Layout(size_t n_, size_t b_, int p_, int rank_) : n(n_), b(b_), p(p_), rank(rank_) {
        nblocks = (n + b - 1) / b;
        npad = nblocks * b;
        // Most square grid with Pr >= Pc
        Pc = 1;
        for (int c = (int)std::sqrt((double)p); c >= 1; --c) {
            if (p % c == 0) { Pc = c; break; }
        }
        Pr = p / Pc;
        ir = rank / Pc;
        jc = rank % Pc;
        nLocRows = locRows(ir);
        nLocCols = locCols(jc);
        MPI_Comm_split(MPI_COMM_WORLD, ir, jc, &rowComm);
        MPI_Comm_split(MPI_COMM_WORLD, jc, ir, &colComm);
    }
    ~Layout() {
        MPI_Comm_free(&rowComm);
        MPI_Comm_free(&colComm);
    }
    Layout(const Layout&) = delete;
    Layout& operator=(const Layout&) = delete;

    size_t locRows(int r) const { return ((size_t)r < nblocks) ? (nblocks - r + Pr - 1) / Pr : 0; }
    size_t locCols(int c) const { return ((size_t)c < nblocks) ? (nblocks - c + Pc - 1) / Pc : 0; }
    int worldRank(int r, int c) const { return r * Pc + c; }

    size_t globalRow(size_t li) const { return li * Pr + ir; }  // tile row index
    size_t globalCol(size_t lj) const { return lj * Pc + jc; }  // tile column index

    // first local tile row with global tile row > k
    size_t firstLocRowAfter(size_t k) const {
        return ((size_t)ir <= k) ? (k - ir) / Pr + 1 : 0;
    }
    // first local tile row with global tile row >= J
    size_t firstLocRowAtOrAfter(size_t J) const {
        return ((size_t)ir < J) ? (J - ir + Pr - 1) / Pr : 0;
    }
    size_t firstLocColAfter(size_t k) const {
        return ((size_t)jc <= k) ? (k - jc) / Pc + 1 : 0;
    }

    size_t tileOffset(size_t li, size_t lj) const { return (li * nLocCols + lj) * b * b; }
};

// Tile update micro-kernel: R rows x (W*VL) columns of a tile:
//   C[r][j] -= sum_k A[r][k] * BT[k][j]   for r in [0,R), j in [0, W*VL)
// BT is the transposed L_jk tile: BT[k][j] = L_jk[j][k]. C and A point at the
// first row of the row group, BT at the first column of the strip (ld = b).
template <int R, int W>
static inline void updateBlock(double* C, const double* A, const double* BT, const size_t b) {
    vdouble acc[R][W];
    for (int r = 0; r < R; ++r)
        for (int w = 0; w < W; ++w) acc[r][w] = vload(C + r * b + w * VL);
    for (size_t k = 0; k < b; ++k, BT += b) {
        vdouble p[W];
        for (int w = 0; w < W; ++w) p[w] = vload(BT + w * VL);
        for (int r = 0; r < R; ++r) {
            const double l = A[r * b + k];
            for (int w = 0; w < W; ++w) acc[r][w] -= l * p[w];
        }
    }
    for (int r = 0; r < R; ++r)
        for (int w = 0; w < W; ++w) vstore(C + r * b + w * VL, acc[r][w]);
}

// Scalar fallback for a partial strip of columns [j0, j1) and R rows
static inline void updateScalar(double* C, const double* A, const double* BT, const size_t b,
                                const size_t rows, const size_t j0, const size_t j1) {
    for (size_t r = 0; r < rows; ++r) {
        for (size_t j = j0; j < j1; ++j) {
            double s = C[r * b + j];
            for (size_t k = 0; k < b; ++k) s -= A[r * b + k] * BT[k * b + j];
            C[r * b + j] = s;
        }
    }
}

// Register-block shape of the micro-kernel (MR rows x NR vectors), chosen so
// that the accumulators plus the NR loaded vectors fit the register file.
#if defined(__AVX512F__)
#define MR 8
#define NR 3
#else
#define MR 6
#define NR 2
#endif

template <int W>
static inline void updateStrip(double* C, const double* A, const double* BT, const size_t b) {
    size_t r0 = 0;
    for (; r0 + MR <= b; r0 += MR) updateBlock<MR, W>(C + r0 * b, A + r0 * b, BT, b);
    for (; r0 + 2 <= b; r0 += 2) updateBlock<2, W>(C + r0 * b, A + r0 * b, BT, b);
    for (; r0 < b; ++r0) updateBlock<1, W>(C + r0 * b, A + r0 * b, BT, b);
}

// C -= A * B^T for b x b tiles, where BT already holds B transposed.
// Loop order: column strips outermost so that a strip of BT (b x NR*VL) stays
// in L1 while all row groups of the tile are processed.
static void tileUpdate(double* C, const double* A, const double* BT, const size_t b) {
    size_t j = 0;
    for (; j + NR * VL <= b; j += NR * VL) updateStrip<NR>(C + j, A, BT + j, b);
    for (; j + VL <= b; j += VL) updateStrip<1>(C + j, A, BT + j, b);
    if (j < b) updateScalar(C, A, BT, b, b, j, b);
}

static void transposeTile(const double* T, double* TT, const size_t b) {
    for (size_t i = 0; i < b; ++i)
        for (size_t j = 0; j < b; ++j) TT[j * b + i] = T[i * b + j];
}

// Factor the diagonal tile in place (lower triangle). Returns the index of the
// first non-positive pivot within the tile, or b on success.
static size_t factorDiagonalTile(double* T, const size_t b) {
    for (size_t i = 0; i < b; ++i) {
        double* ai = T + i * b;
        for (size_t j = 0; j <= i; ++j) {
            double* aj = T + j * b;
            double sum = 0.0;
            if (i == j) {
                for (size_t k = 0; k < j; ++k) sum += aj[k] * aj[k];
                const double val = aj[j] - sum;
                if (val <= 0.0) return j;
                aj[j] = sqrt(val);
            } else {
                for (size_t k = 0; k < j; ++k) sum += ai[k] * aj[k];
                ai[j] = (ai[j] - sum) / aj[j];
            }
        }
    }
    return b;
}

// Solve T := T * Lkk^-T in place for a b x b tile (Lkk lower triangular).
// Works on the transposed tile so that the inner loops are contiguous axpys.
static void trsmTile(double* T, const double* Lkk, const size_t b, double* tmp) {
    transposeTile(T, tmp, b);  // tmp[c][r] = T[r][c]
    for (size_t c = 0; c < b; ++c) {
        double* tc = tmp + c * b;
        const double d = Lkk[c * b + c];
        for (size_t r = 0; r < b; ++r) tc[r] /= d;
        for (size_t c2 = c + 1; c2 < b; ++c2) {
            const double l = Lkk[c2 * b + c];
            double* t2 = tmp + c2 * b;
            for (size_t r = 0; r < b; ++r) t2[r] -= l * tc[r];
        }
    }
    transposeTile(tmp, T, b);
}

// A holds this process's tiles (2D block-cyclic distribution described by `lay`).
// On return A holds the local tiles of L (upper triangle zeroed).
//
// The loop uses a depth-1 lookahead: once the tiles of column k+1 have been
// updated, the diagonal factorization, the panel solve and the panel broadcast
// of step k+1 are started with non-blocking collectives while the remaining
// trailing update of step k proceeds (polling MPI_Test to drive progress).
bool choleskyDecomposition(std::vector<double>& A, const Layout& lay) {
    const size_t b = lay.b;
    const size_t bb = b * b;
    const size_t nb = lay.nblocks;
    const size_t nLocRows = lay.nLocRows;
    const size_t nLocCols = lay.nLocCols;

    // Double-buffered per-step data (step parity selects the buffer)
    std::vector<double> Lkk[2] = {std::vector<double>(bb + 1), std::vector<double>(bb + 1)};  // + failure flag
    std::vector<double> rowBuf[2] = {std::vector<double>(nLocRows * bb + 1),
                                     std::vector<double>(nLocRows * bb + 1)};                   // L_ik tiles + flag
    MPI_Request reqRow[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};

    std::vector<double> tmp(bb);                   // scratch for trsm
    std::vector<double> BT(bb);                    // transposed L_jk
    std::vector<double> sendBuf(nLocRows * bb);    // my contribution to the column exchange
    std::vector<double> colBuf(nLocCols * bb);     // L_jk for my tile columns (j > k)
    std::vector<int> counts(lay.Pr), displs(lay.Pr);
    std::vector<size_t> posOfJ(nb);

    if (nb == 0) return true;

    auto trailRowsFirst = [&](size_t k) { return std::min(lay.firstLocRowAfter(k), nLocRows); };

    // Factor diagonal tile k, solve the panel below it on process column k % Pc
    // and start broadcasting the panel tiles along the process rows. The
    // failure flag travels with the panel so that every process sees it.
    auto startPanel = [&](size_t k) {
        const int cur = (int)(k & 1);
        const int kr = (int)(k % lay.Pr);
        const int kc = (int)(k % lay.Pc);
        const size_t liFirst = trailRowsFirst(k);
        const size_t nTrail = nLocRows - liFirst;
        double* rb = rowBuf[cur].data();

        if (lay.jc == kc) {
            double* lkk = Lkk[cur].data();
            if (lay.ir == kr) {
                double* T = A.data() + lay.tileOffset(k / lay.Pr, k / lay.Pc);
                const size_t bad = factorDiagonalTile(T, b);
                if (bad < b) {
                    printf("Error: Matrix is not positive definite at diagonal element %zu\n", k * b + bad);
                    fflush(stdout);
                    lkk[bb] = 1.0;
                } else {
                    memcpy(lkk, T, bb * sizeof(double));
                    lkk[bb] = 0.0;
                }
            }
            MPI_Bcast(lkk, (int)(bb + 1), MPI_DOUBLE, kr, lay.colComm);
            if (lkk[bb] == 0.0) {
                const size_t lk = k / lay.Pc;
                for (size_t li = liFirst; li < nLocRows; ++li) {
                    double* T = A.data() + lay.tileOffset(li, lk);
                    trsmTile(T, lkk, b, tmp.data());
                    memcpy(rb + (li - liFirst) * bb, T, bb * sizeof(double));
                }
            }
            rb[nTrail * bb] = lkk[bb];
        }
        MPI_Ibcast(rb, (int)(nTrail * bb + 1), MPI_DOUBLE, kc, lay.rowComm, &reqRow[cur]);
    };

    startPanel(0);

    for (size_t k = 0; k < nb; ++k) {
        const int cur = (int)(k & 1);
        const size_t liFirst = trailRowsFirst(k);
        const size_t nTrail = nLocRows - liFirst;
        const double* rb = rowBuf[cur].data();

        // Panel tiles L_ik of step k (and the failure flag)
        MPI_Wait(&reqRow[cur], MPI_STATUS_IGNORE);
        if (rb[nTrail * bb] != 0.0) return false;
        if (k + 1 == nb) break;

        // All-gather along process columns the L_jk for tile columns j owned by
        // this process column. Process row r contributes tiles j > k with
        // j % Pr == r and j % Pc == jc, in increasing order of j.
        {
            size_t pos = 0;
            for (int r = 0; r < lay.Pr; ++r) {
                displs[r] = (int)(pos * bb);
                size_t cnt = 0;
                size_t J = ((size_t)r <= k) ? ((k - r) / lay.Pr + 1) * lay.Pr + r : (size_t)r;
                for (; J < nb; J += lay.Pr) {
                    if ((int)(J % lay.Pc) == lay.jc) {
                        posOfJ[J] = pos + cnt;
                        ++cnt;
                    }
                }
                counts[r] = (int)(cnt * bb);
                pos += cnt;
            }
            size_t mine = 0;
            for (size_t li = liFirst; li < nLocRows; ++li) {
                if ((int)(lay.globalRow(li) % lay.Pc) == lay.jc) {
                    memcpy(sendBuf.data() + mine * bb, rb + (li - liFirst) * bb, bb * sizeof(double));
                    ++mine;
                }
            }
            MPI_Allgatherv(sendBuf.data(), (int)(mine * bb), MPI_DOUBLE,
                           colBuf.data(), counts.data(), displs.data(), MPI_DOUBLE, lay.colComm);
        }

        // Trailing update of my tiles: A_ij -= L_ik * L_jk^T for i >= j > k.
        // Diagonal tiles are updated in full; their strictly upper part is
        // never read and is zeroed at the end.
        auto updateColumn = [&](size_t lj) {
            const size_t J = lay.globalCol(lj);
            transposeTile(colBuf.data() + posOfJ[J] * bb, BT.data(), b);
            for (size_t li = lay.firstLocRowAtOrAfter(J); li < nLocRows; ++li) {
                tileUpdate(A.data() + lay.tileOffset(li, lj), rb + (li - liFirst) * bb, BT.data(), b);
            }
        };

        // Lookahead: update tile column k+1 first, then start the next panel
        const size_t ljNext = lay.firstLocColAfter(k);
        if (ljNext < nLocCols && lay.globalCol(ljNext) == k + 1) updateColumn(ljNext);
        startPanel(k + 1);

        // Remaining trailing update, polling the pending panel broadcast
        const int nxt = 1 - cur;
        for (size_t lj = lay.firstLocColAfter(k + 1); lj < nLocCols; ++lj) {
            const size_t J = lay.globalCol(lj);
            transposeTile(colBuf.data() + posOfJ[J] * bb, BT.data(), b);
            for (size_t li = lay.firstLocRowAtOrAfter(J); li < nLocRows; ++li) {
                tileUpdate(A.data() + lay.tileOffset(li, lj), rb + (li - liFirst) * bb, BT.data(), b);
                if (reqRow[nxt] != MPI_REQUEST_NULL) {
                    int done;
                    MPI_Test(&reqRow[nxt], &done, MPI_STATUS_IGNORE);
                }
            }
        }
    }

    // Zero out upper triangular part
    for (size_t li = 0; li < nLocRows; ++li) {
        const size_t I = lay.globalRow(li);
        for (size_t lj = 0; lj < nLocCols; ++lj) {
            const size_t J = lay.globalCol(lj);
            double* T = A.data() + lay.tileOffset(li, lj);
            if (I < J) {
                std::fill(T, T + bb, 0.0);
            } else if (I == J) {
                for (size_t i = 0; i < b; ++i) std::fill(T + i * b + i + 1, T + (i + 1) * b, 0.0);
            }
        }
    }

    return true;
}

// Generate a symmetric positive definite matrix (local tiles only)
void generatePositiveDefiniteMatrix(std::vector<double>& A, const Layout& lay) {
    // Method: Create A = B * B^T where B is random
    // This guarantees A is positive semi-definite
    // Then add identity to make it strictly positive definite
    //
    // Every process replays the same deterministic random stream and keeps the
    // rows of B it needs: those of its tile rows and those of its tile columns.
    // Only the lower triangle (plus diagonal tiles) is generated; the strictly
    // upper tiles are never read by the factorization. Padding rows/columns
    // beyond n get an identity block.

    const size_t n = lay.n;
    const size_t b = lay.b;
    const size_t nLocRows = lay.nLocRows;
    const size_t nLocCols = lay.nLocCols;

    std::vector<double> Brow(nLocRows * b * n);
    std::vector<double> Bcol(nLocCols * b * n);
    std::vector<double> line(n);

    unsigned int seed = 42;

    // Generate random matrix B, keep the rows I need
    for (size_t i = 0; i < n; ++i) {
        for (size_t k = 0; k < n; ++k) line[k] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
        const size_t I = i / b;
        if ((int)(I % lay.Pr) == lay.ir) {
            memcpy(Brow.data() + ((I / lay.Pr) * b + i % b) * n, line.data(), n * sizeof(double));
        }
        if ((int)(I % lay.Pc) == lay.jc) {
            memcpy(Bcol.data() + ((I / lay.Pc) * b + i % b) * n, line.data(), n * sizeof(double));
        }
    }

    // Compute my tiles of A = B * B^T (lower triangle incl. diagonal tiles)
    std::fill(A.begin(), A.end(), 0.0);
    for (size_t li = 0; li < nLocRows; ++li) {
        const size_t I = lay.globalRow(li);
        for (size_t lj = 0; lj < nLocCols; ++lj) {
            const size_t J = lay.globalCol(lj);
            if (J > I) continue;
            double* T = A.data() + lay.tileOffset(li, lj);
            for (size_t r = 0; r < b; ++r) {
                const size_t gi = I * b + r;
                const double* bi = Brow.data() + (li * b + r) * n;
                for (size_t c = 0; c < b; ++c) {
                    const size_t gj = J * b + c;
                    if (gi >= n || gj >= n) {
                        T[r * b + c] = (gi == gj) ? 1.0 : 0.0;
                        continue;
                    }
                    const double* bj = Bcol.data() + (lj * b + c) * n;
                    double sum = 0.0;
                    for (size_t k = 0; k < n; ++k) sum += bi[k] * bj[k];
                    T[r * b + c] = sum;
                }
            }
            // Add diagonal dominance to ensure positive definiteness
            if (I == J) {
                for (size_t r = 0; r < b && I * b + r < n; ++r) T[r * b + r] += n;
            }
        }
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const Layout& lay) {
    // Validate by computing L * L^T and comparing with original matrix.
    // Each process checks its lower-triangle tiles. For that it needs the full
    // tile rows of L for its tile rows (gathered along the process row) and
    // for its tile columns (gathered along the process column). Since both
    // L * L^T and A are exactly symmetric, checking the lower triangle yields
    // the same maximum errors as checking the full matrix. L is lower
    // triangular, so each dot product can stop at min(i, j) without changing
    // the result.

    const size_t n = lay.n;
    const size_t b = lay.b;
    const size_t bb = b * b;
    const size_t nb = lay.nblocks;
    const size_t npad = lay.npad;
    const size_t nLocRows = lay.nLocRows;
    const size_t nLocCols = lay.nLocCols;

    // Phase 1: full tile rows of L for my tile rows (row-major, ld = npad)
    std::vector<double> Lrows(nLocRows * b * npad);
    {
        std::vector<int> counts(lay.Pc), displs(lay.Pc);
        for (int c = 0; c < lay.Pc; ++c) {
            counts[c] = (int)(nLocRows * lay.locCols(c) * bb);
            displs[c] = (c == 0) ? 0 : displs[c - 1] + counts[c - 1];
        }
        std::vector<double> recv(nLocRows * nb * bb);
        MPI_Allgatherv(L.data(), (int)(nLocRows * nLocCols * bb), MPI_DOUBLE,
                       recv.data(), counts.data(), displs.data(), MPI_DOUBLE, lay.rowComm);
        for (int c = 0; c < lay.Pc; ++c) {
            const size_t ncols = lay.locCols(c);
            const double* src = recv.data() + displs[c];
            for (size_t li = 0; li < nLocRows; ++li) {
                for (size_t lj = 0; lj < ncols; ++lj) {
                    const size_t J = lj * lay.Pc + c;
                    const double* T = src + (li * ncols + lj) * bb;
                    for (size_t r = 0; r < b; ++r) {
                        memcpy(Lrows.data() + (li * b + r) * npad + J * b, T + r * b, b * sizeof(double));
                    }
                }
            }
        }
    }

    // Phase 2: full tile rows of L for my tile columns, gathered along the
    // process column. Process row r contributes tile rows J with J % Pr == r
    // and J % Pc == jc.
    std::vector<double> Lcols(nLocCols * b * npad);
    std::vector<size_t> posOfJ(nb);
    {
        std::vector<int> counts(lay.Pr), displs(lay.Pr);
        size_t pos = 0;
        for (int r = 0; r < lay.Pr; ++r) {
            displs[r] = (int)(pos * b * npad);
            size_t cnt = 0;
            for (size_t J = r; J < nb; J += lay.Pr) {
                if ((int)(J % lay.Pc) == lay.jc) posOfJ[J] = pos + cnt++;
            }
            counts[r] = (int)(cnt * b * npad);
            pos += cnt;
        }
        std::vector<double> send;
        for (size_t li = 0; li < nLocRows; ++li) {
            if ((int)(lay.globalRow(li) % lay.Pc) == lay.jc) {
                send.insert(send.end(), Lrows.begin() + li * b * npad, Lrows.begin() + (li + 1) * b * npad);
            }
        }
        MPI_Allgatherv(send.data(), (int)send.size(), MPI_DOUBLE,
                       Lcols.data(), counts.data(), displs.data(), MPI_DOUBLE, lay.colComm);
    }

    double maxError = 0.0;
    double relError = 0.0;

    for (size_t li = 0; li < nLocRows; ++li) {
        const size_t I = lay.globalRow(li);
        for (size_t lj = 0; lj < nLocCols; ++lj) {
            const size_t J = lay.globalCol(lj);
            if (J > I) continue;
            const double* T = A_orig.data() + lay.tileOffset(li, lj);
            const double* Lj0 = Lcols.data() + posOfJ[J] * b * npad;
            for (size_t r = 0; r < b; ++r) {
                const size_t gi = I * b + r;
                if (gi >= n) break;
                const double* Li = Lrows.data() + (li * b + r) * npad;
                const size_t cmax = (I == J) ? r + 1 : b;
                for (size_t c = 0; c < cmax; ++c) {
                    const size_t gj = J * b + c;
                    if (gj >= n) break;
                    const double* Lj = Lj0 + c * npad;
                    const size_t kmax = std::min(gi, gj) + 1;
                    double sum = 0.0;
                    for (size_t k = 0; k < kmax; ++k) sum += Li[k] * Lj[k];

                    const double orig = T[r * b + c];
                    const double error = fabs(sum - orig);
                    maxError = std::max(maxError, error);

                    const double rel = error / (fabs(orig) + 1e-10);
                    relError = std::max(relError, rel);
                }
            }
        }
    }

    double local[2] = {maxError, relError};
    double global[2];
    MPI_Allreduce(local, global, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    maxError = global[0];
    relError = global[1];

    if (lay.rank == 0) {
        printf("Max absolute error: %.10e\n", maxError);
        printf("Max relative error: %.10e\n", relError);
    }

    // Check if error is within tolerance
    if (relError > 1e-6) {
        if (lay.rank == 0) printf("Validation failed: relative error too large\n");
        return false;
    }

    return true;
}

// Gather the distributed matrix into a full row-major n x n matrix on rank 0
static void gatherMatrix(const std::vector<double>& A, const Layout& lay, std::vector<double>& full) {
    const size_t n = lay.n;
    const size_t b = lay.b;
    const size_t bb = b * b;

    MPI_Datatype tileType;
    MPI_Type_contiguous((int)bb, MPI_DOUBLE, &tileType);
    MPI_Type_commit(&tileType);

    if (lay.rank == 0) {
        full.assign(n * n, 0.0);
        std::vector<double> tmp;
        for (int r = 0; r < lay.Pr; ++r) {
            const size_t nrows = lay.locRows(r);
            for (int c = 0; c < lay.Pc; ++c) {
                const size_t ncols = lay.locCols(c);
                const size_t ntiles = nrows * ncols;
                if (ntiles == 0) continue;
                const double* src;
                if (lay.worldRank(r, c) == 0) {
                    src = A.data();
                } else {
                    tmp.resize(ntiles * bb);
                    MPI_Recv(tmp.data(), (int)ntiles, tileType, lay.worldRank(r, c), 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    src = tmp.data();
                }
                for (size_t li = 0; li < nrows; ++li) {
                    const size_t I = li * lay.Pr + r;
                    for (size_t lj = 0; lj < ncols; ++lj) {
                        const size_t J = lj * lay.Pc + c;
                        const double* T = src + (li * ncols + lj) * bb;
                        const size_t ncopy = std::min(b, n - J * b);
                        for (size_t rr = 0; rr < b && I * b + rr < n; ++rr) {
                            memcpy(full.data() + (I * b + rr) * n + J * b, T + rr * b, ncopy * sizeof(double));
                        }
                    }
                }
            }
        }
    } else if (lay.nLocRows * lay.nLocCols > 0) {
        MPI_Send(A.data(), (int)(lay.nLocRows * lay.nLocCols), tileType, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Type_free(&tileType);
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
    int rank = 0, nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    size_t n = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (all ranks see the same argv)
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

    int ret = 0;
    {
        // Choose the tile size: large enough for an efficient update kernel,
        // small enough that the tiles spread over the process grid.
        size_t b = 64;
        {
            int Pc = 1;
            for (int c = (int)std::sqrt((double)nprocs); c >= 1; --c) {
                if (nprocs % c == 0) { Pc = c; break; }
            }
            const size_t Pr = nprocs / Pc;
            while (b > 16 && (n + b - 1) / b < 2 * Pr) b /= 2;
        }

        const Layout lay(n, b, nprocs, rank);

        if (rank == 0) {
            printf("Cholesky Decomposition Benchmark\n");
            printf("Matrix size: %zu x %zu\n", n, n);
            printf("MPI ranks: %d (%d x %d grid, tile size %zu)\n", nprocs, lay.Pr, lay.Pc, b);
            printf("Validation: %s\n", validate ? "enabled" : "disabled");
        }

        // Allocate local tiles of the matrix
        std::vector<double> A(lay.nLocRows * lay.nLocCols * b * b);
        std::vector<double> A_orig;

        // Generate positive definite matrix
        if (rank == 0) printf("Generating positive definite matrix...\n");
        generatePositiveDefiniteMatrix(A, lay);

        if (validate) {
            A_orig = A; // Save original for validation
        }

        // Perform Cholesky decomposition
        if (rank == 0) {
            printf("Computing Cholesky decomposition...\n");
            fflush(stdout);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        auto start = std::chrono::high_resolution_clock::now();

        bool success = choleskyDecomposition(A, lay);

        MPI_Barrier(MPI_COMM_WORLD);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        if (!success) {
            if (rank == 0) printf("Cholesky decomposition failed\n");
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

        // Print results for external validation
        if (printResults) {
            std::vector<double> full;
            gatherMatrix(A, lay, full);
            if (rank == 0) print_results(full, "CholeskyL");
        }

        // Validation
        if (validate) {
            if (rank == 0) printf("Validating result...\n");
            bool valid = validateCholesky(A, A_orig, lay);

            if (rank == 0) {
                if (valid) {
                    printf("Validation: PASSED\n");
                } else {
                    printf("Validation: FAILED\n");
                }
            }
            ret = valid ? 0 : 1;
        }
    }

    MPI_Finalize();
    return ret;
}
