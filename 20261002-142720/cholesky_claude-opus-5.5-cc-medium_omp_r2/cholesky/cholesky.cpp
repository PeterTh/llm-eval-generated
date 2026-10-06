#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <vector>

#include <omp.h>

#include "../common/results_output.hpp"

// Parallel blocked Cholesky decomposition (OpenMP)
// Decomposes positive definite matrix A into L * L^T where L is lower triangular

// Allocator that zero-fills new memory in parallel, so that pages are first
// touched (and thus placed) across all NUMA nodes instead of a single one.
template <typename T>
struct FirstTouchAllocator {
    using value_type = T;
    FirstTouchAllocator() = default;
    template <typename U>
    FirstTouchAllocator(const FirstTouchAllocator<U>&) {}

    T* allocate(const size_t count) {
        const size_t bytes = (count * sizeof(T) + 63) / 64 * 64;
        T* p = static_cast<T*>(std::aligned_alloc(64, bytes));
        if (!p) throw std::bad_alloc();
        char* c = reinterpret_cast<char*>(p);
        constexpr size_t CHUNK = size_t(1) << 16;
        const size_t nChunks = (bytes + CHUNK - 1) / CHUNK;
        #pragma omp parallel for schedule(static) if (nChunks > 16)
        for (size_t i = 0; i < nChunks; ++i) {
            memset(c + i * CHUNK, 0, std::min(CHUNK, bytes - i * CHUNK));
        }
        return p;
    }
    void deallocate(T* p, size_t) { std::free(p); }

    template <typename U>
    bool operator==(const FirstTouchAllocator<U>&) const { return true; }
    template <typename U>
    bool operator!=(const FirstTouchAllocator<U>&) const { return false; }
};

using Matrix = std::vector<double, FirstTouchAllocator<double>>;

typedef double v4d __attribute__((vector_size(32)));
typedef double v4du __attribute__((vector_size(32), aligned(8)));   // unaligned access

// Register-blocked micro-kernel on column-major operands:
//   C[c*ldc + r] += sum_{p<kc} A[p*lda + r] * B[p*ldb + c],  r < MR, c < NR
// Each element accumulates over p in increasing order. With FUSED == false the
// product is rounded before it is added (no FMA contraction).
constexpr size_t MR = 8;
constexpr size_t NR = 6;

template <bool FUSED>
static inline v4d mulAdd(const v4d c, const v4d a, const v4d b) {
    if constexpr (FUSED) {
        return c + a * b;
    } else {
        v4d prod = a * b;
        __asm__("" : "+x"(prod));   // keep the product rounded separately
        return c + prod;
    }
}

template <bool FUSED>
static inline void microKernel(const size_t kc, const double* __restrict A, const size_t lda,
                               const double* __restrict B, const size_t ldb,
                               double* __restrict C, const size_t ldc) {
    constexpr size_t MV = MR / 4;
    v4d c[NR][MV];
    #pragma GCC unroll 16
    for (size_t j = 0; j < NR; ++j) {
        #pragma GCC unroll 16
        for (size_t m = 0; m < MV; ++m) c[j][m] = *(const v4du*)(C + j * ldc + 4 * m);
    }
    for (size_t p = 0; p < kc; ++p) {
        v4d a[MV];
        #pragma GCC unroll 16
        for (size_t m = 0; m < MV; ++m) a[m] = *(const v4du*)(A + 4 * m);
        #pragma GCC unroll 16
        for (size_t j = 0; j < NR; ++j) {
            const v4d bj = (v4d){B[j], B[j], B[j], B[j]};
            #pragma GCC unroll 16
            for (size_t m = 0; m < MV; ++m) c[j][m] = mulAdd<FUSED>(c[j][m], a[m], bj);
        }
        A += lda;
        B += ldb;
    }
    #pragma GCC unroll 16
    for (size_t j = 0; j < NR; ++j) {
        #pragma GCC unroll 16
        for (size_t m = 0; m < MV; ++m) *(v4du*)(C + j * ldc + 4 * m) = c[j][m];
    }
}

// ---------------------------------------------------------------------------
// Blocked parallel Cholesky (right-looking with panel groups), in place.
//
// Like the unblocked algorithm, every element L(i,j) is obtained as
//   (A(i,j) - sum_{k<j} L(i,k) * L(j,k)) / L(j,j)
// with the sum accumulated separately in increasing k order. The original values
// stay in the lower triangle of A until the element is finalized; the partial
// sums S(i,j), i > j, are kept transposed in the (otherwise unused) upper
// triangle at A[j*n + i], and the diagonal sums in a separate vector.
// Products are rounded before being added, except for the last (j % 4) terms of
// each sum, matching the arithmetic of the reference build.
// ---------------------------------------------------------------------------

constexpr size_t PG = 24;          // row group of the packed panel (multiple of MR, NR)
constexpr size_t MAX_TILE = 96;    // upper bound for the tile size
constexpr size_t UPD_BLOCK = 96;   // block size of update work items (multiple of PG)
constexpr size_t KC = 256;         // depth chunk of the update kernels

// acc[r] += x[r] * l for r in [r0, r1), each product rounded before the addition
static inline void axpyPlain(double* __restrict acc, const double* __restrict x, const double l,
                             const size_t r0, const size_t r1) {
    const v4d lv = {l, l, l, l};
    size_t r = r0;
    for (; r + 4 <= r1; r += 4) {
        v4d prod = *(const v4du*)(x + r) * lv;
        __asm__("" : "+x"(prod));
        *(v4du*)(acc + r) = *(const v4du*)(acc + r) + prod;
    }
    for (; r < r1; ++r) {
        double prod = x[r] * l;
        __asm__("" : "+x"(prod));
        acc[r] += prod;
    }
}

// acc[r] = fma(x[r], l, acc[r]) for r in [r0, r1) (the reference's scalar epilogue)
static inline void axpyFused(double* __restrict acc, const double* __restrict x, const double l,
                             const size_t r0, const size_t r1) {
    for (size_t r = r0; r < r1; ++r) acc[r] = fma(x[r], l, acc[r]);
}

// Diagonal tile (m x m, column-major, ld = m). On entry T holds the partial sums
// and Av the original values (lower parts); on exit T holds L.
// Returns the local index of the failing pivot or -1.
static long localPotrf(double* __restrict T, const double* __restrict Av, const size_t m) {
    double acc[MAX_TILE];
    const bool blocked = (m % PG == 0);
    for (size_t c = 0; c < m; ++c) {
        double* tc = T + c * m;
        // Column block start: the sums over p < j0 are added with the micro-kernel
        const size_t j0 = blocked ? c / PG * PG : 0;
        if (blocked && c == j0 && j0 > 0) {
            for (size_t cs = j0; cs < j0 + PG; cs += NR) {
                for (size_t rs = j0; rs < m; rs += MR) {
                    microKernel<false>(j0, T + rs, m, T + cs, m, T + cs * m + rs, m);
                }
            }
        }
        for (size_t r = c; r < m; ++r) acc[r] = tc[r];
        const size_t cMain = c / 4 * 4;
        for (size_t p = j0; p < cMain; ++p) axpyPlain(acc, T + p * m, T[p * m + c], c, m);
        for (size_t p = cMain; p < c; ++p) axpyFused(acc, T + p * m, T[p * m + c], c, m);
        const double val = Av[c * m + c] - acc[c];
        if (val <= 0.0) {
            return (long)c;
        }
        const double d = sqrt(val);
        tc[c] = d;
        for (size_t r = c + 1; r < m; ++r) {
            tc[r] = (Av[c * m + r] - acc[r]) / d;
        }
    }
    return -1;
}

// Row chunk below a diagonal tile: X (PG rows x m columns, column-major, ld = PG)
// holds the partial sums on entry and L on exit; Av holds the original values in
// the same layout. Lkk is the factored diagonal tile (m x m, column-major).
static void localTrsm(double* __restrict X, const double* __restrict Av, const size_t m,
                      const double* __restrict Lkk, const size_t rows) {
    const bool blocked = (m % PG == 0);
    for (size_t c = 0; c < m; ++c) {
        double* xc = X + c * PG;
        // Column block start: the sums over p < j0 are added with the micro-kernel
        const size_t j0 = blocked ? c / PG * PG : 0;
        if (blocked && c == j0 && j0 > 0) {
            for (size_t cs = j0; cs < j0 + PG; cs += NR) {
                for (size_t rs = 0; rs < PG; rs += MR) {
                    microKernel<false>(j0, X + rs, PG, Lkk + cs, m, X + cs * PG + rs, PG);
                }
            }
        }
        const size_t cMain = c / 4 * 4;
        for (size_t p = j0; p < cMain; ++p) axpyPlain(xc, X + p * PG, Lkk[p * m + c], 0, rows);
        for (size_t p = cMain; p < c; ++p) axpyFused(xc, X + p * PG, Lkk[p * m + c], 0, rows);
        const double d = Lkk[c * m + c];
        const double* ac = Av + c * PG;
        for (size_t r = 0; r < rows; ++r) {
            xc[r] = (ac[r] - xc[r]) / d;
        }
    }
}

// Packed panel: rows grouped by PG, P[(q / PG) * W * PG + p * PG + q % PG] = L(rowBase + q, colBase + p)
struct Panel {
    double* P;
    size_t W;        // depth capacity (group width)
    size_t rowBase;  // first row (= first column of the group)
    double* row(size_t globalRow, size_t p) const {
        const size_t q = globalRow - rowBase;
        return P + (q / PG) * W * PG + p * PG + q % PG;
    }
};

// S(rows [r0, r1), cols [c0, c1), lower part only) += L(rows, depth) * L(cols, depth)^T
// for the panel depth range [p0, p0 + kc). With first == true the previous sums
// are zero (not read). The block is staged in a contiguous local buffer
// (column-major, rows contiguous), which avoids cache-set conflicts from the
// large row stride of A.
static void updateBlock(double* __restrict A, double* __restrict Sd, const size_t n,
                        const Panel& pan, const size_t r0, const size_t r1,
                        const size_t c0, const size_t c1, const size_t p0, const size_t kc,
                        const bool first) {
    constexpr size_t LDC = UPD_BLOCK;
    alignas(64) double Cl[UPD_BLOCK * UPD_BLOCK];
    const size_t nr = r1 - r0, nc = c1 - c0;
    const size_t nrPad = (nr + MR - 1) / MR * MR;
    const size_t ncPad = (nc + NR - 1) / NR * NR;

    for (size_t cc = 0; cc < ncPad; ++cc) {
        double* dst = Cl + cc * LDC;
        const size_t j = c0 + cc;
        // valid rows i in [max(r0, j), r1)
        const size_t lo = (cc < nc) ? std::min(nr, j > r0 ? j - r0 : 0) : nr;
        for (size_t rr = 0; rr < lo; ++rr) dst[rr] = 0.0;
        if (first) {
            for (size_t rr = lo; rr < nrPad; ++rr) dst[rr] = 0.0;
        } else {
            size_t rr = lo;
            if (rr < nr && r0 + rr == j) {
                dst[rr] = Sd[j];
                ++rr;
            }
            const double* src = A + j * n + r0;
            for (; rr < nr; ++rr) dst[rr] = src[rr];
            for (rr = nr; rr < nrPad; ++rr) dst[rr] = 0.0;
        }
    }

    for (size_t k0 = 0; k0 < kc; k0 += KC) {
        const size_t kk = std::min(KC, kc - k0);
        const size_t p = p0 + k0;
        for (size_t cs = 0; cs < ncPad; cs += NR) {
            const double* Bp = pan.row(c0 + cs, p);
            for (size_t rs = 0; rs < nrPad; rs += MR) {
                if (r0 + rs + MR - 1 < c0 + cs) continue;   // entirely above the diagonal
                microKernel<false>(kk, pan.row(r0 + rs, p), PG, Bp, PG, Cl + cs * LDC + rs, LDC);
            }
        }
    }

    for (size_t cc = 0; cc < nc; ++cc) {
        const double* srcl = Cl + cc * LDC;
        const size_t j = c0 + cc;
        size_t rr = std::min(nr, j > r0 ? j - r0 : 0);
        if (rr < nr && r0 + rr == j) {
            Sd[j] = srcl[rr];
            ++rr;
        }
        double* dst = A + j * n + r0;
        for (; rr < nr; ++rr) dst[rr] = srcl[rr];
    }
}

static size_t chooseTileSize(const size_t n) {
    return n <= 1536 ? 48 : 96;
}

// Panel group (outer level) and sub-group (inner level) widths, in tiles
static size_t chooseGroupTiles(const size_t n) {
    if (n <= 1536) return 2;
    return 6;
}

static size_t chooseSubGroupTiles(const size_t n) {
    if (n <= 1536) return 2;
    return 3;
}

// Map a linear index t to (i, j) with j <= i, t = i*(i+1)/2 + j
static inline void triIndex(const size_t t, size_t& i, size_t& j) {
    size_t r = (size_t)((sqrt(8.0 * (double)t + 1.0) - 1.0) / 2.0);
    while (r * (r + 1) / 2 > t) --r;
    while ((r + 1) * (r + 2) / 2 <= t) ++r;
    i = r;
    j = t - r * (r + 1) / 2;
}

// Number of threads to use: the OpenMP setting if given explicitly, otherwise one
// thread per physical core (the dense kernels saturate a core's FP units, so SMT
// siblings only add synchronization cost).
static int defaultThreadCount() {
    const int maxThreads = omp_get_max_threads();
    if (getenv("OMP_NUM_THREADS") != nullptr) return maxThreads;
    int smtActive = 0;
    if (FILE* f = fopen("/sys/devices/system/cpu/smt/active", "r")) {
        if (fscanf(f, "%d", &smtActive) != 1) smtActive = 0;
        fclose(f);
    }
    int threadsPerCore = 1;
    if (smtActive == 1) {
        if (FILE* f = fopen("/sys/devices/system/cpu/cpu0/topology/thread_siblings_list", "r")) {
            // Format: "0,128" or "0-1": count the listed siblings
            char buf[256] = {0};
            if (fgets(buf, sizeof(buf), f)) {
                int a = 0, bnum = 0;
                if (sscanf(buf, "%d-%d", &a, &bnum) == 2 && bnum > a) {
                    threadsPerCore = bnum - a + 1;
                } else {
                    threadsPerCore = 1;
                    for (const char* p = buf; *p; ++p) if (*p == ',') ++threadsPerCore;
                }
            }
            fclose(f);
        }
    }
    return std::max(1, std::min(maxThreads, omp_get_num_procs() / std::max(1, threadsPerCore)));
}

// Work phase of the factorization schedule. All items of a phase are independent;
// a phase starts only when the previous phase of its sequence has completed.
struct Phase {
    enum Kind { POTRF, TRSM, UPDATE } kind;
    size_t kc0;      // POTRF/TRSM: first column of the tile
    size_t R0;       // UPDATE: first row/column of the region (rows [R0, n))
    size_t C1;       // UPDATE: end of the column range [R0, C1)
    size_t RB, CB;   // UPDATE: row/column block sizes of the items
    size_t p0, kc;   // UPDATE: panel depth range
    bool first;      // UPDATE: no previous partial sums
    int buf;         // packed panel buffer
    size_t gs;       // first column of the panel group
    size_t nbc;      // UPDATE: number of column blocks
    size_t nItems;
};

static Phase makeUpdate(const size_t n, const size_t R0, const size_t C1, const size_t RB,
                        const size_t CB, const size_t p0, const size_t kc, const bool first,
                        const int buf, const size_t gs) {
    Phase ph{};
    ph.kind = Phase::UPDATE;
    ph.R0 = R0; ph.C1 = C1; ph.RB = RB; ph.CB = CB;
    ph.p0 = p0; ph.kc = kc; ph.first = first; ph.buf = buf; ph.gs = gs;
    const size_t nbr = (n - R0 + RB - 1) / RB;
    ph.nbc = (C1 - R0 + CB - 1) / CB;
    ph.nItems = (R0 < C1) ? nbr * ph.nbc : 0;
    return ph;
}

static inline void cpuRelax() {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#endif
}

bool choleskyDecomposition(Matrix& A, const size_t n) {
    // A is stored in row-major order
    if (n == 0) return true;

    const size_t b = chooseTileSize(n);
    const size_t W = b * chooseGroupTiles(n);
    const size_t W2 = b * chooseSubGroupTiles(n);
    const size_t packRows = (n + PG - 1) / PG * PG;
    const size_t numGroups = (n + W - 1) / W;

    // Two packed panel buffers: the next group's panel is factored while the
    // trailing update of the current group still reads the current one.
    double* P[2];
    P[0] = static_cast<double*>(std::aligned_alloc(64, packRows * W * sizeof(double)));
    P[1] = (numGroups > 1) ? static_cast<double*>(std::aligned_alloc(64, packRows * W * sizeof(double)))
                           : nullptr;
    if (!P[0] || (numGroups > 1 && !P[1])) {
        printf("Error: allocation failed\n");
        std::free(P[0]);
        std::free(P[1]);
        return false;
    }
    std::vector<double> diagSum(n, 0.0);
    std::vector<double> LkkBuf(b * b), AkkBuf(b * b);

    double* Ap = A.data();
    double* Sd = diagSum.data();
    double* Lk = LkkBuf.data();
    double* Ak = AkkBuf.data();
    long failIndex = -1;
    std::atomic<bool> failed{false};

    // Limit team size for small problems where synchronization dominates
    const int maxThreads = omp_get_max_threads();
    const size_t nb = (n + UPD_BLOCK - 1) / UPD_BLOCK;
    const size_t usefulThreads = std::max<size_t>(1, nb * (nb + 1) / 2);
    const int nthreads = (int)std::min<size_t>((size_t)maxThreads, usefulThreads);

    // Schedule state of the current step (shared)
    std::vector<Phase> phases;
    Phase background{};
    std::vector<std::atomic<size_t>> claimed(64), done(64);
    std::atomic<size_t> curPhase{0}, bgClaimed{0};

    auto groupStart = [&](size_t g) { return g * W; };
    auto groupEnd = [&](size_t g) { return std::min(n, (g + 1) * W); };

    // Append the phases that factor the panel of group h (columns [gs, ge))
    auto addPanelPhases = [&](size_t h) {
        const size_t gs = groupStart(h), ge = groupEnd(h);
        const int buf = (int)(h % 2);
        for (size_t ss = gs; ss < ge; ss += W2) {
            const size_t se = std::min(ge, ss + W2);
            for (size_t kc0 = ss; kc0 < se; kc0 += b) {
                const size_t ke = std::min(n, kc0 + b);
                Phase pf{};
                pf.kind = Phase::POTRF; pf.kc0 = kc0; pf.buf = buf; pf.gs = gs; pf.nItems = 1;
                phases.push_back(pf);
                if (ke < n) {
                    Phase pt{};
                    pt.kind = Phase::TRSM; pt.kc0 = kc0; pt.buf = buf; pt.gs = gs;
                    pt.nItems = (n - ke + PG - 1) / PG;
                    phases.push_back(pt);
                }
                if (ke < se) {
                    // Remaining columns of the sub-group
                    phases.push_back(makeUpdate(n, ke, se, 2 * PG, UPD_BLOCK, kc0 - gs, ke - kc0,
                                                kc0 == 0, buf, gs));
                }
            }
            if (se < ge) {
                // Remaining columns of the panel group, with the whole sub-group
                phases.push_back(makeUpdate(n, se, ge, UPD_BLOCK, UPD_BLOCK, ss - gs, se - ss,
                                            ss == 0, buf, gs));
            }
        }
    };

    auto runPotrf = [&](const Phase& ph) {
        const size_t kc0 = ph.kc0;
        const size_t wk = std::min(b, n - kc0);
        for (size_t c = 0; c < wk; ++c) {
            for (size_t r = c; r < wk; ++r) {
                Ak[c * wk + r] = Ap[(kc0 + r) * n + kc0 + c];
                Lk[c * wk + r] = (kc0 == 0) ? 0.0
                               : (r == c) ? Sd[kc0 + c] : Ap[(kc0 + c) * n + kc0 + r];
            }
        }
        const long loc = localPotrf(Lk, Ak, wk);
        if (loc >= 0) {
            failIndex = (long)(kc0 + loc);
            failed.store(true);
        } else {
            for (size_t r = 0; r < wk; ++r) {
                for (size_t c = 0; c <= r; ++c) Ap[(kc0 + r) * n + kc0 + c] = Lk[c * wk + r];
            }
        }
    };

    // Triangular solve of one row chunk below the diagonal tile, packing the result
    auto runTrsm = [&](const Phase& ph, size_t t) {
        const size_t kc0 = ph.kc0;
        const size_t wk = b;   // tiles with rows below are always full
        const size_t r0 = kc0 + wk + t * PG;
        const size_t rows = std::min(PG, n - r0);
        alignas(64) double X[PG * MAX_TILE];
        alignas(64) double Av[PG * MAX_TILE];
        for (size_t c = 0; c < wk; ++c) {
            const double* src = Ap + (kc0 + c) * n + r0;
            for (size_t r = 0; r < rows; ++r) X[c * PG + r] = (kc0 == 0) ? 0.0 : src[r];
            for (size_t r = rows; r < PG; ++r) X[c * PG + r] = 0.0;
        }
        for (size_t r = 0; r < rows; ++r) {
            const double* src = Ap + (r0 + r) * n + kc0;
            for (size_t c = 0; c < wk; ++c) Av[c * PG + r] = src[c];
        }
        localTrsm(X, Av, wk, Lk, rows);
        for (size_t r = 0; r < rows; ++r) {
            double* dst = Ap + (r0 + r) * n + kc0;
            for (size_t c = 0; c < wk; ++c) dst[c] = X[c * PG + r];
        }
        const Panel pan{P[ph.buf], W, ph.gs};
        double* pk = pan.row(r0, kc0 - ph.gs);
        for (size_t c = 0; c < wk; ++c) {
            for (size_t r = 0; r < PG; ++r) pk[c * PG + r] = r < rows ? X[c * PG + r] : 0.0;
        }
    };

    auto runUpdate = [&](const Phase& ph, size_t t) {
        const size_t br = t / ph.nbc, bc = t % ph.nbc;
        const size_t r0 = ph.R0 + br * ph.RB, c0 = ph.R0 + bc * ph.CB;
        if (r0 + ph.RB <= c0) return;   // entirely above the diagonal
        const Panel pan{P[ph.buf], W, ph.gs};
        updateBlock(Ap, Sd, n, pan, r0, std::min(n, r0 + ph.RB), c0, std::min(ph.C1, c0 + ph.CB),
                    ph.p0, ph.kc, ph.first);
    };

    auto runItem = [&](const Phase& ph, size_t t) {
        switch (ph.kind) {
            case Phase::POTRF: runPotrf(ph); break;
            case Phase::TRSM: runTrsm(ph, t); break;
            case Phase::UPDATE: runUpdate(ph, t); break;
        }
    };

    // Threads reserved for the (latency-critical) panel phases
    const int panelThreads = std::max(1, nthreads / 8);

    #pragma omp parallel num_threads(nthreads)
    {
        const int tid = omp_get_thread_num();

        // Step g = -1 factors the first panel group. Step g >= 0 completes the
        // update of group g + 1's columns with group g, then factors the panel of
        // group g + 1 while the rest of group g's trailing update runs alongside.
        for (long g = -1; g + 1 < (long)numGroups; ++g) {
            #pragma omp single
            {
                phases.clear();
                background = Phase{};
                if (g >= 0) {
                    const size_t gs = groupStart(g), ge = groupEnd(g), ge1 = groupEnd(g + 1);
                    phases.push_back(makeUpdate(n, ge, ge1, UPD_BLOCK, UPD_BLOCK, 0, ge - gs,
                                                g == 0, (int)(g % 2), gs));
                    if (ge1 < n) {
                        background = makeUpdate(n, ge1, n, UPD_BLOCK, UPD_BLOCK, 0, ge - gs,
                                                g == 0, (int)(g % 2), gs);
                    }
                }
                addPanelPhases(g + 1);
                if (claimed.size() < phases.size()) {
                    claimed = std::vector<std::atomic<size_t>>(phases.size());
                    done = std::vector<std::atomic<size_t>>(phases.size());
                }
                for (size_t p = 0; p < phases.size(); ++p) {
                    claimed[p].store(0);
                    done[p].store(0);
                }
                curPhase.store(0);
                bgClaimed.store(0);
            }

            const size_t nPhases = phases.size();
            const size_t nBg = background.nItems;
            while (!failed.load(std::memory_order_relaxed)) {
                const size_t p = curPhase.load(std::memory_order_acquire);
                if (p < nPhases) {
                    const size_t i = claimed[p].fetch_add(1);
                    if (i < phases[p].nItems) {
                        runItem(phases[p], i);
                        if (failed.load(std::memory_order_relaxed)) break;
                        if (done[p].fetch_add(1, std::memory_order_acq_rel) + 1 == phases[p].nItems) {
                            curPhase.store(p + 1, std::memory_order_release);
                        }
                        continue;
                    }
                }
                // No panel work available right now: take trailing-update work
                if ((p >= nPhases || tid >= panelThreads) && bgClaimed.load(std::memory_order_relaxed) < nBg) {
                    const size_t j = bgClaimed.fetch_add(1);
                    if (j < nBg) {
                        runUpdate(background, j);
                        continue;
                    }
                }
                if (p >= nPhases && bgClaimed.load(std::memory_order_relaxed) >= nBg) break;
                cpuRelax();
            }
            #pragma omp barrier
            if (failed.load()) break;
        }

        if (!failed.load()) {
            // Zero out upper triangular part
            #pragma omp for schedule(static)
            for (size_t i = 0; i < n; ++i) {
                memset(Ap + i * n + i + 1, 0, (n - i - 1) * sizeof(double));
            }
        }
    }

    std::free(P[0]);
    std::free(P[1]);

    if (failed.load()) {
        // Matrix is not positive definite
        printf("Error: Matrix is not positive definite at diagonal element %ld\n", failIndex);
        return false;
    }
    return true;
}

// out = B * B^T for a row-major n x n matrix B (out row-major, n x n). Every
// element is accumulated over k in increasing order starting from 0.0, exactly
// as the naive triple loop. The arithmetic also mirrors the reference build
// (products rounded before being added, with a scalar FMA epilogue for the last
// n % 4 terms), so the results are bitwise identical.
static void productWithTranspose(const double* B, const size_t n, double* out) {
    if (n == 0) return;
    constexpr size_t BS = 96;   // block size, multiple of MR and NR
    const size_t np = (n + BS - 1) / BS * BS;
    const size_t nb = np / BS;
    const size_t kMain = n / 4 * 4;

    // BT: column-major copy of B (BT[k*np + i] = B[i][k]), zero-padded rows
    double* BT = static_cast<double*>(std::aligned_alloc(64, np * n * sizeof(double)));
    double* C = static_cast<double*>(std::aligned_alloc(64, np * np * sizeof(double)));
    if (!BT || !C) {
        std::free(BT);
        std::free(C);
        throw std::bad_alloc();
    }

    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for (size_t k = 0; k < n; ++k) {
            double* dst = BT + k * np;
            for (size_t i = 0; i < n; ++i) dst[i] = B[i * n + k];
            for (size_t i = n; i < np; ++i) dst[i] = 0.0;
        }

        // Lower block triangle of C (column-major): blocks (bi, bj) with bi >= bj
        const size_t nBlocks = nb * (nb + 1) / 2;
        #pragma omp for schedule(dynamic, 1)
        for (size_t t = 0; t < nBlocks; ++t) {
            size_t bi, bj;
            triIndex(t, bi, bj);
            const size_t r0 = bi * BS, c0 = bj * BS;
            for (size_t c = c0; c < c0 + BS; ++c) {
                for (size_t r = r0; r < r0 + BS; ++r) C[c * np + r] = 0.0;
            }
            for (size_t k0 = 0; k0 < n; k0 += KC) {
                const size_t kc = std::min(KC, n - k0);
                // Terms k < kMain: rounded product then add; remaining tail terms: fused
                const size_t kcPlain = std::min(kc, kMain > k0 ? kMain - k0 : 0);
                for (size_t c = c0; c < c0 + BS; c += NR) {
                    for (size_t r = r0; r < r0 + BS; r += MR) {
                        microKernel<false>(kcPlain, BT + k0 * np + r, np, BT + k0 * np + c, np,
                                           C + c * np + r, np);
                        microKernel<true>(kc - kcPlain, BT + (k0 + kcPlain) * np + r, np,
                                          BT + (k0 + kcPlain) * np + c, np, C + c * np + r, np);
                    }
                }
            }
        }

        // out[i][j] = C(i, j) with C(i, j) at C[j*np + i]; use symmetry for upper blocks
        #pragma omp for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                const bool lower = (i / BS) >= (j / BS);
                out[i * n + j] = lower ? C[j * np + i] : C[i * np + j];
            }
        }
    }

    std::free(BT);
    std::free(C);
}

// Generate a symmetric positive definite matrix
void generatePositiveDefiniteMatrix(Matrix& A, const size_t n) {
    // Method: Create A = B * B^T where B is random
    // This guarantees A is positive semi-definite
    // Then add identity to make it strictly positive definite
    
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    
    // Generate random matrix B (sequential to preserve the random sequence)
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }
    
    // Compute A = B * B^T
    productWithTranspose(B.data(), n, A.data());
    
    // Add diagonal dominance to ensure positive definiteness
    for (size_t i = 0; i < n; ++i) {
        A[i * n + i] += n;
    }
}

bool validateCholesky(const Matrix& L, const Matrix& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix
    
    std::vector<double> reconstructed(n * n);
    
    // Compute L * L^T
    productWithTranspose(L.data(), n, reconstructed.data());
    
    // Compare with original
    double maxError = 0.0;
    double relError = 0.0;
    
    #pragma omp parallel for schedule(static) reduction(max:maxError, relError)
    for (size_t i = 0; i < n * n; ++i) {
        const double error = fabs(reconstructed[i] - A_orig[i]);
        maxError = std::max(maxError, error);
        
        const double rel = error / (fabs(A_orig[i]) + 1e-10);
        relError = std::max(relError, rel);
    }
    
    printf("Max absolute error: %.10e\n", maxError);
    printf("Max relative error: %.10e\n", relError);
    
    // Check if error is within tolerance
    if (relError > 1e-6) {
        printf("Validation failed: relative error too large\n");
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

int main(int argc, char** argv) {
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
            printUsage(argv[0]);
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }
    
    // Use one thread per physical core unless OMP_NUM_THREADS is set explicitly
    omp_set_num_threads(defaultThreadCount());

    printf("Cholesky Decomposition Benchmark\n");
    printf("Matrix size: %zu x %zu\n", n, n);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    // Allocate matrix
    Matrix A(n * n);
    Matrix A_orig;
    
    // Generate positive definite matrix
    printf("Generating positive definite matrix...\n");
    generatePositiveDefiniteMatrix(A, n);
    
    if (validate) {
        A_orig = A; // Save original for validation
    }
    
    // Perform Cholesky decomposition
    printf("Computing Cholesky decomposition...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    bool success = choleskyDecomposition(A, n);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (!success) {
        printf("Cholesky decomposition failed\n");
        return 1;
    }
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
    double ops = (double)n * n * n / 3.0;
    double gflops = ops / (duration.count() / 1000.0) / 1e9;
    printf("Performance: %.3f GFLOPS\n", gflops);
    
    // Print results for external validation
    if (printResults) {
        print_results(std::vector<double>(A.begin(), A.end()), "CholeskyL");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateCholesky(A, A_orig, n);
        
        if (valid) {
            printf("Validation: PASSED\n");
            return 0;
        } else {
            printf("Validation: FAILED\n");
            return 1;
        }
    }
    
    return 0;
}
