#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

#include <omp.h>

// Parallel tiled Cholesky decomposition (OpenMP, shared memory).
// Decomposes positive definite matrix A into L * L^T where L is lower triangular.
//
// The matrix is processed in nb x nb tiles with a left-looking algorithm: every tile
// L(I,J) is owned by one thread, which accumulates the updates of all tiles K < J and
// then solves the tile (TRSM) or, on the diagonal, factors it (POTRF). Threads claim
// tiles in dependency order and synchronize through per-tile completion flags.
//
// To stay bit-identical to the sequential unblocked algorithm, every element keeps
// its dot product sum_{k<j} L[i][k]*L[j][k] accumulated from 0.0 in ascending k
// order in a separate partial sum tile S, and only at the end computes
// (A - sum) / L[j][j]. Tiles are column-major so that the kernels vectorize across
// independent elements while each element's accumulation stays sequential.

namespace {

constexpr size_t MAX_NB = 96;  // largest tile size
constexpr int ACTIVE = 2;      // tiles in flight per thread

typedef double v4d __attribute__((vector_size(32), aligned(8), may_alias));

// The reference sequential build (gcc -O3 -march=native) evaluates each dot product
// sum_{k<j} x[k]*y[k] by adding rounded products in order for k < (j & ~3) and with
// fused multiply-adds (when FMA is available) for the remaining j % 4 terms.
// The kernels below reproduce exactly that per-element rounding sequence.

constexpr size_t MR = 8;  // rows per micro-tile (2 x v4d)
constexpr size_t NR = 6;  // cols per GEMM micro-tile
constexpr size_t TR = 4;  // cols per TRSM column group (must be 4 for fused-term alignment)

inline v4d ld4(const double* p) { return *reinterpret_cast<const v4d*>(p); }
inline void st4(double* p, v4d v) { *reinterpret_cast<v4d*>(p) = v; }
inline v4d bc4(double x) { return v4d{x, x, x, x}; }

// Rounded product, never contracted into an FMA
inline v4d mulR(v4d a, v4d b) {
    v4d p = a * b;
    __asm__("" : "+x"(p));
    return p;
}
inline double mulR(double a, double b) {
    double p = a * b;
    __asm__("" : "+x"(p));
    return p;
}

// Fused multiply-add as emitted by the reference build
inline v4d fmaF(v4d a, v4d b, v4d c) {
#ifdef __FMA__
    v4d r;
    for (int l = 0; l < 4; ++l) r[l] = __builtin_fma(a[l], b[l], c[l]);
    return r;
#else
    return c + mulR(a, b);
#endif
}
inline double fmaF(double a, double b, double c) {
#ifdef __FMA__
    return __builtin_fma(a, b, c);
#else
    return c + mulR(a, b);
#endif
}

// Contiguous memory ranges to prefetch (into L2) while a GEMM tile update runs
struct PrefetchList {
    static constexpr int MAXR = 16;
    const double* p[MAXR];
    size_t lines[MAXR];
    int count = 0;
    size_t total = 0;
    void add(const double* ptr, size_t len) {
        if (count == MAXR || len == 0) return;
        p[count] = ptr;
        lines[count] = (len + 7) / 8;
        total += lines[count];
        ++count;
    }
};

// C[:, j] += sum_{k<kc} A[:, k] * B[j, k]   (column-major tiles, leading dimension ld)
// i.e. S(I,J) += L(I,K) * L(J,K)^T with K < J: all terms are unfused, k ascending.
void gemmTile(const double* __restrict A, const double* __restrict B, double* __restrict C,
              const size_t nb, const size_t ld, const size_t kc,
              const PrefetchList* pf = nullptr) {
    // Spread prefetches of the next operands evenly over the micro-tiles
    const size_t nMicro = (nb / NR) * (nb / MR);
    const size_t perMicro = pf ? (pf->total + nMicro - 1) / nMicro : 0;
    int pr = 0;
    size_t pl = 0;
    for (size_t j0 = 0; j0 < nb; j0 += NR) {
        for (size_t i0 = 0; i0 < nb; i0 += MR) {
            for (size_t e = 0; e < perMicro && pr < pf->count; ++e) {
                __builtin_prefetch(pf->p[pr] + pl * 8, 0, 2);
                if (++pl == pf->lines[pr]) {
                    pl = 0;
                    ++pr;
                }
            }
            v4d c[NR][2];
            for (size_t c_ = 0; c_ < NR; ++c_) {
                c[c_][0] = ld4(C + (j0 + c_) * ld + i0);
                c[c_][1] = ld4(C + (j0 + c_) * ld + i0 + 4);
            }
            const double* a = A + i0;
            const double* b = B + j0;
            for (size_t k = 0; k < kc; ++k) {
                const v4d a0 = ld4(a + k * ld);
                const v4d a1 = ld4(a + k * ld + 4);
                for (size_t c_ = 0; c_ < NR; ++c_) {
                    const v4d bv = bc4(b[k * ld + c_]);
                    c[c_][0] += mulR(a0, bv);
                    c[c_][1] += mulR(a1, bv);
                }
            }
            for (size_t c_ = 0; c_ < NR; ++c_) {
                st4(C + (j0 + c_) * ld + i0, c[c_][0]);
                st4(C + (j0 + c_) * ld + i0 + 4, c[c_][1]);
            }
        }
    }
}

// Solve off-diagonal tile T(I,J) using factored diagonal tile D = L(J,J).
// T holds A values on entry and L values on exit; S holds partial sums over k < J*nb.
void trsmTile(const double* __restrict D, double* __restrict T, const double* __restrict S,
              const size_t nb, const size_t ld) {
    for (size_t i0 = 0; i0 < nb; i0 += MR) {
        for (size_t j0 = 0; j0 < nb; j0 += TR) {
            v4d c[TR][2];
            for (size_t c_ = 0; c_ < TR; ++c_) {
                c[c_][0] = ld4(S + (j0 + c_) * ld + i0);
                c[c_][1] = ld4(S + (j0 + c_) * ld + i0 + 4);
            }
            // Contributions of already finished columns k < j0 (unfused)
            for (size_t k = 0; k < j0; ++k) {
                const v4d a0 = ld4(T + k * ld + i0);
                const v4d a1 = ld4(T + k * ld + i0 + 4);
                for (size_t c_ = 0; c_ < TR; ++c_) {
                    const v4d dv = bc4(D[k * ld + j0 + c_]);
                    c[c_][0] += mulR(a0, dv);
                    c[c_][1] += mulR(a1, dv);
                }
            }
            // Triangular part inside the column group (fused), finishing columns in order
            for (size_t c_ = 0; c_ < TR; ++c_) {
                const size_t j = j0 + c_;
                for (size_t k = j0; k < j; ++k) {
                    const v4d dv = bc4(D[k * ld + j]);
                    c[c_][0] = fmaF(ld4(T + k * ld + i0), dv, c[c_][0]);
                    c[c_][1] = fmaF(ld4(T + k * ld + i0 + 4), dv, c[c_][1]);
                }
                const v4d dd = bc4(D[j * ld + j]);
                double* t = T + j * ld + i0;
                st4(t, (ld4(t) - c[c_][0]) / dd);
                st4(t + 4, (ld4(t + 4) - c[c_][1]) / dd);
            }
        }
    }
}

// Factor diagonal tile (m = actual size). Returns local index of failing diagonal or -1.
long potrfTile(double* __restrict T, const double* __restrict S, const size_t ld, const size_t m) {
    alignas(64) double s[MAX_NB];
    for (size_t j = 0; j < m; ++j) {
        for (size_t i = j; i < m; ++i) s[i] = S[j * ld + i];
        const size_t kf = j & ~size_t(3);
        for (size_t k = 0; k < kf; ++k) {
            const double ljk = T[k * ld + j];
            const double* tk = T + k * ld;
            for (size_t i = j; i < m; ++i) s[i] += mulR(tk[i], ljk);
        }
        for (size_t k = kf; k < j; ++k) {
            const double ljk = T[k * ld + j];
            const double* tk = T + k * ld;
            for (size_t i = j; i < m; ++i) s[i] = fmaF(tk[i], ljk, s[i]);
        }
        const double val = T[j * ld + j] - s[j];
        if (val <= 0.0) return (long)j;
        const double d = sqrt(val);
        T[j * ld + j] = d;
        double* tj = T + j * ld;
        for (size_t i = j + 1; i < m; ++i) tj[i] = (tj[i] - s[i]) / d;
    }
    return -1;
}

size_t chooseBlockSize(const size_t n) {
    if (n <= 1536) return 48;
    return 96;
}

inline void cpuRelax() {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#endif
}

// Copy an (rows x cols) column-major block with stride `ld` into a zero padded
// nb x nb column-major buffer with stride nb.
inline void packTile(double* __restrict dst, const double* __restrict src, const size_t ld,
                     const size_t rows, const size_t cols, const size_t nb) {
    for (size_t c = 0; c < cols; ++c) {
        memcpy(dst + c * nb, src + c * ld, rows * sizeof(double));
        if (rows < nb) memset(dst + c * nb + rows, 0, (nb - rows) * sizeof(double));
    }
    if (cols < nb) memset(dst + cols * nb, 0, (nb - cols) * nb * sizeof(double));
}

// Plain cached copy (large memcpy calls may use non-temporal stores, which would
// evict the packed operands we are about to use from the cache)
inline void copyDoubles(double* __restrict dst, const double* __restrict src, size_t len) {
    for (size_t i = 0; i < len; ++i) dst[i] = src[i];
}

// Off-diagonal L tiles of column J are kept, packed and contiguous, in the strict upper
// part of row block J of A: nb rows of (n - (J+1)*nb) doubles each. Its capacity equals
// the size of all (unpadded) tiles L(I,J), I > J. The rows form one virtual buffer.
struct PanelStore {
    double* U;
    size_t n, nb;
    double* addr(size_t J, size_t v, size_t& avail) const {
        const size_t seg = n - (J + 1) * nb;
        const size_t r = v / seg;
        const size_t c = v % seg;
        avail = seg - c;
        return U + (J * nb + r) * n + (J + 1) * nb + c;
    }
    void load(double* dst, size_t J, size_t v, size_t len) const {
        while (len > 0) {
            size_t avail;
            const double* p = addr(J, v, avail);
            const size_t c = std::min(avail, len);
            copyDoubles(dst, p, c);
            dst += c; v += c; len -= c;
        }
    }
    void store(const double* src, size_t J, size_t v, size_t len) const {
        while (len > 0) {
            size_t avail;
            double* p = addr(J, v, avail);
            const size_t c = std::min(avail, len);
            copyDoubles(p, src, c);
            src += c; v += c; len -= c;
        }
    }
    void zero(size_t J, size_t v, size_t len) const {
        while (len > 0) {
            size_t avail;
            double* p = addr(J, v, avail);
            const size_t c = std::min(avail, len);
            memset(p, 0, c * sizeof(double));
            v += c; len -= c;
        }
    }
    void addPrefetch(PrefetchList& pf, size_t J, size_t v, size_t len) const {
        while (len > 0) {
            size_t avail;
            const double* p = addr(J, v, avail);
            const size_t c = std::min(avail, len);
            pf.add(p, c);
            v += c; len -= c;
        }
    }
};

} // namespace

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    // A is stored in row-major order.
    //
    // Everything happens in place. The original values of a tile are read (only by
    // its owner) from the lower triangle. Finished off-diagonal tiles are published in
    // packed form in the upper triangle (see PanelStore), finished diagonal tiles as L^T
    // in the upper half of their diagonal block. Finished tiles are also written in
    // row-major form to the lower triangle, and the upper triangle is zeroed as soon as
    // the data stored there is no longer needed.
    if (n == 0) return true;

    double* const U = A.data();
    const size_t nb = chooseBlockSize(n);
    const size_t nt = (n + nb - 1) / nb;
    const size_t tileElems = nb * nb;
    const size_t numTiles = nt * (nt + 1) / 2;
    auto tileRows = [&](size_t I) { return std::min(nb, n - I * nb); };
    const PanelStore store{U, n, nb};
    auto tileOffset = [&](size_t I, size_t J) { return (I - J - 1) * tileElems; };

    // Tiles are claimed in column order (a topological order of the dependencies);
    // upper triangle slots are zeroed in row order (the order in which they become free).
    std::vector<std::pair<unsigned, unsigned>> byColumn, byRow;  // (I, J)
    byColumn.reserve(numTiles);
    byRow.reserve(numTiles);
    for (size_t J = 0; J < nt; ++J) {
        for (size_t I = J; I < nt; ++I) byColumn.emplace_back((unsigned)I, (unsigned)J);
    }
    for (size_t I = 0; I < nt; ++I) {
        for (size_t J = 0; J <= I; ++J) byRow.emplace_back((unsigned)I, (unsigned)J);
    }

    std::vector<std::atomic<int>> done(nt * nt);
    for (auto& d : done) d.store(0, std::memory_order_relaxed);
    std::vector<std::atomic<size_t>> columnRemaining(nt);
    for (size_t J = 0; J < nt; ++J) columnRemaining[J].store(nt - J, std::memory_order_relaxed);
    std::atomic<size_t> completeColumns{0};  // columns 0 .. completeColumns-1 are finished
    std::atomic<size_t> nextTile{0};
    std::atomic<size_t> nextSlot{0};
    std::atomic<long> failIndex{-1};

    // Team size: about one thread per 8 tiles keeps all threads busy; more threads only
    // add fork/wake-up overhead (which is substantial with hundreds of threads).
    const size_t maxThreads = (size_t)omp_get_max_threads();
    int numThreads = (int)std::min(maxThreads, std::max<size_t>(std::min<size_t>(4, numTiles), numTiles / 8));

    #pragma omp parallel num_threads(numThreads)
    {
        // Per-thread buffers (on the thread's stack, ~370 KB): partial sums of the tiles
        // in flight and packed A, B and diagonal tiles
        alignas(64) double work[(ACTIVE + 3) * MAX_NB * MAX_NB];
        double* const S = work;
        double* const PA = S + ACTIVE * tileElems;
        double* const PB = PA + tileElems;
        double* const PD = PB + tileElems;

        auto failed = [&]() { return failIndex.load(std::memory_order_relaxed) >= 0; };
        auto isDone = [&](size_t I, size_t J) {
            return done[I * nt + J].load(std::memory_order_acquire) != 0;
        };
        auto finishTile = [&](size_t I, size_t J) {
            done[I * nt + J].store(1, std::memory_order_release);
            if (columnRemaining[J].fetch_sub(1) == 1) {
                // Advance the prefix of complete columns as far as possible
                size_t c = completeColumns.load();
                while (c < nt && columnRemaining[c].load() == 0) {
                    if (completeColumns.compare_exchange_weak(c, c + 1)) ++c;
                }
            }
        };
        // Load finished off-diagonal tile L(I,K) as zero padded nb x nb column-major
        auto loadTile = [&](double* dst, size_t I, size_t K) {
            const size_t mi = tileRows(I);
            if (mi == nb) {
                store.load(dst, K, tileOffset(I, K), tileElems);
            } else {
                for (size_t c = 0; c < nb; ++c) {
                    store.load(dst + c * nb, K, tileOffset(I, K) + c * mi, mi);
                    memset(dst + c * nb + mi, 0, (nb - mi) * sizeof(double));
                }
            }
        };
        // Pack the original A values of tile (I,J) from the lower triangle (row-major)
        auto loadOriginal = [&](double* dst, size_t I, size_t J) {
            const size_t mi = tileRows(I);
            const size_t mj = tileRows(J);
            if (mi < nb || mj < nb) memset(dst, 0, tileElems * sizeof(double));
            const double* src = U + (I * nb) * n + J * nb;
            for (size_t rr = 0; rr < mi; ++rr) {
                const size_t cend = (I == J) ? rr + 1 : mj;
                for (size_t cc = 0; cc < cend; ++cc) dst[cc * nb + rr] = src[rr * n + cc];
            }
        };

        // Left-looking tiled factorization: each thread claims whole tiles in dependency
        // order and applies their updates in ascending K order. A thread keeps up to
        // ACTIVE tiles in flight and works on whichever one has its next dependency
        // satisfied. (The earliest claimed unfinished tile is always ready, and its
        // owner polls all of its tiles, so this cannot deadlock.)
        struct Active {
            size_t I, J, K, claim;
            double* S;
            bool used;
        };
        Active act[ACTIVE];
        for (int a = 0; a < ACTIVE; ++a) act[a] = {0, 0, 0, 0, S + a * tileElems, false};
        int numActive = 0;
        bool exhausted = false;

        // Advance tile `a` as far as its dependencies allow; returns true on any progress
        auto advance = [&](Active& a) {
            const size_t I = a.I, J = a.J;
            const size_t mi = tileRows(I);
            const size_t mj = tileRows(J);
            bool progress = false;
            // With several tiles in flight do one update at a time, so that the older
            // (more urgent) tile is resumed as soon as it becomes ready
            while (a.K < J && isDone(J, a.K) && isDone(I, a.K) && !(progress && numActive > 1)) {
                const size_t K = a.K;
                loadTile(PB, J, K);
                loadTile(PA, I, K);
                // Prefetch the operands of the next update while computing this one
                PrefetchList pf;
                if (K + 1 < J) {
                    store.addPrefetch(pf, K + 1, tileOffset(J, K + 1), mj * nb);
                    store.addPrefetch(pf, K + 1, tileOffset(I, K + 1), mi * nb);
                }
                gemmTile(PA, PB, a.S, nb, nb, nb, &pf);
                ++a.K;
                progress = true;
            }
            if (a.K < J) return progress;

            if (I == J) {
                loadOriginal(PD, J, J);
                const long r = potrfTile(PD, a.S, nb, mj);
                if (r >= 0) {
                    failIndex.store((long)(J * nb) + r);
                    return true;
                }
                // L^T into the upper half of the diagonal block, publish, then L into the lower half
                double* blk = U + (J * nb) * n + J * nb;
                for (size_t cc = 0; cc < mj; ++cc) {
                    memcpy(blk + cc * n + cc, PD + cc * nb + cc, (mj - cc) * sizeof(double));
                }
                finishTile(I, J);
                for (size_t rr = 1; rr < mj; ++rr) {
                    for (size_t cc = 0; cc < rr; ++cc) blk[rr * n + cc] = PD[cc * nb + rr];
                }
            } else {
                if (!isDone(J, J)) return progress;
                packTile(PD, U + (J * nb) * n + J * nb, n, nb, nb, nb);
                loadOriginal(PA, I, J);
                trsmTile(PD, PA, a.S, nb, nb);
                // Publish packed tile in the upper triangle, then L into the lower triangle
                if (mi == nb) {
                    store.store(PA, J, tileOffset(I, J), tileElems);
                } else {
                    for (size_t cc = 0; cc < nb; ++cc) store.store(PA + cc * nb, J, tileOffset(I, J) + cc * mi, mi);
                }
                finishTile(I, J);
                double* tl = U + (I * nb) * n + J * nb;
                for (size_t rr = 0; rr < mi; ++rr) {
                    for (size_t cc = 0; cc < nb; ++cc) tl[rr * n + cc] = PA[cc * nb + rr];
                }
            }
            a.used = false;
            --numActive;
            return true;
        };

        while (!failed()) {
            // Claim a new tile when idle (nothing in flight, or everything blocked)
            if (numActive == 0 && exhausted) break;
            // Try the oldest tile first
            int order[ACTIVE];
            for (int a = 0; a < ACTIVE; ++a) order[a] = a;
            std::sort(order, order + ACTIVE, [&](int x, int y) { return act[x].claim < act[y].claim; });
            bool progress = false;
            for (int o = 0; o < ACTIVE && !progress && !failed(); ++o) {
                if (act[order[o]].used) progress = advance(act[order[o]]);
            }
            if (progress) continue;
            if (!exhausted && numActive < ACTIVE) {
                const size_t t = nextTile.fetch_add(1, std::memory_order_relaxed);
                if (t >= numTiles) {
                    exhausted = true;
                    continue;
                }
                for (int a = 0; a < ACTIVE; ++a) {
                    if (!act[a].used) {
                        act[a].I = byColumn[t].first;
                        act[a].J = byColumn[t].second;
                        act[a].K = 0;
                        act[a].claim = t;
                        act[a].used = true;
                        memset(act[a].S, 0, tileElems * sizeof(double));
                        ++numActive;
                        break;
                    }
                }
                continue;
            }
            cpuRelax();
        }

        // Zero out upper triangular part. The slot of tile (I,J) is dead once all
        // columns up to I are finished, so this overlaps with the end of the factorization.
        while (!failed()) {
            const size_t sl = nextSlot.fetch_add(1, std::memory_order_relaxed);
            if (sl >= numTiles) break;
            const size_t I = byRow[sl].first;
            const size_t J = byRow[sl].second;
            while (completeColumns.load(std::memory_order_acquire) <= I && !failed()) cpuRelax();
            if (failed()) break;
            if (I == J) {
                double* blk = U + (J * nb) * n + J * nb;
                const size_t mj = tileRows(J);
                for (size_t rr = 0; rr + 1 < mj; ++rr) memset(blk + rr * n + rr + 1, 0, (mj - rr - 1) * sizeof(double));
            } else {
                store.zero(J, tileOffset(I, J), tileRows(I) * nb);
            }
        }
    }

    const long f = failIndex.load();
    if (f >= 0) {
        // Matrix is not positive definite
        printf("Error: Matrix is not positive definite at diagonal element %zu\n", (size_t)f);
        return false;
    }

    return true;
}

// Size a vector to `count` zero-initialized elements such that its pages are first
// touched by all OpenMP threads, which spreads the memory over the NUMA nodes instead
// of placing all of it on the node of the allocating thread.
void allocateDistributed(std::vector<double>& v, const size_t count) {
    // Touch the freshly reserved (not yet used) storage page by page in parallel;
    // resize() then zero-fills it in place, keeping the page placement.
    v.clear();
    v.reserve(count);
    double* const p = v.data();
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < count; i += 512) p[i] = 0.0;  // one write per 4 KB page
    v.resize(count);
}

// Generate a symmetric positive definite matrix
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    // Method: Create A = B * B^T where B is random
    // This guarantees A is positive semi-definite
    // Then add identity to make it strictly positive definite
    
    std::vector<double> B;
    allocateDistributed(B, n * n);
    unsigned int seed = 42;
    
    // Generate random matrix B
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }
    
    // Compute A = B * B^T
    #pragma omp parallel for schedule(dynamic, 4)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += B[i * n + k] * B[j * n + k];
            }
            A[i * n + j] = sum;
        }
    }
    
    // Add diagonal dominance to ensure positive definiteness
    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        A[i * n + i] += n;
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix
    
    std::vector<double> reconstructed;
    allocateDistributed(reconstructed, n * n);
    
    // Compute L * L^T
    #pragma omp parallel for schedule(dynamic, 4)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += L[i * n + k] * L[j * n + k];
            }
            reconstructed[i * n + j] = sum;
        }
    }
    
    // Compare with original
    double maxError = 0.0;
    double relError = 0.0;
    
    #pragma omp parallel
    {
        double localMax = 0.0;
        double localRel = 0.0;
        #pragma omp for nowait
        for (size_t i = 0; i < n * n; ++i) {
            const double error = fabs(reconstructed[i] - A_orig[i]);
            localMax = std::max(localMax, error);

            const double rel = error / (fabs(A_orig[i]) + 1e-10);
            localRel = std::max(localRel, rel);
        }
        #pragma omp critical
        {
            maxError = std::max(maxError, localMax);
            relError = std::max(relError, localRel);
        }
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
    
    printf("Cholesky Decomposition Benchmark\n");
    printf("Matrix size: %zu x %zu\n", n, n);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    // Allocate matrix
    std::vector<double> A;
    allocateDistributed(A, n * n);
    std::vector<double> A_orig;
    
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
        print_results(A, "CholeskyL");
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
