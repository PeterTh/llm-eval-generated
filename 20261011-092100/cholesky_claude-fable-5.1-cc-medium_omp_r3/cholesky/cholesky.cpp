#include <algorithm>
#include <atomic>
#include <chrono>
#include <thread>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <omp.h>
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif
#if defined(__linux__)
#include <sched.h>
#include <sys/mman.h>
#endif

#include "../common/results_output.hpp"

// Blocked (tiled) right-looking Cholesky decomposition parallelized with OpenMP.
// Decomposes positive definite matrix A into L * L^T where L is lower triangular.
// A is stored in row-major order; on return the lower triangle holds L and the
// strict upper triangle is zero (same contract as the original unblocked code).
//
// The matrix is copied into a tile-major layout (each T x T tile contiguous) so
// that every kernel works on cache-friendly, conflict-free blocks regardless of
// the (often power-of-two) leading dimension of the user's matrix.
//
// Tile operations (block column k, tile size T):
//   POTRF(k,k)                  factor the diagonal tile
//   TRSM(i,k)  for i > k        A(i,k) <- A(i,k) * L(k,k)^-T, then pack A(i,k)
//   GEMM(i,j)  for k < j <= i   A(i,j) -= A(i,k) * A(j,k)^T
// Step k of the pipeline (column k already solved and packed):
//   phase A: GEMM(i,k+1) for all i (row chunks, dynamic); POTRF(k+1) starts on
//            one thread as soon as the diagonal tile's chunks are done
//   phase B: GEMM(i,j) for j >= k+2 on owner threads with work stealing, and
//            TRSM(i,k+1) + packing interleaved as soon as POTRF(k+1) is done
// Two spin barriers per step; all other synchronization is through atomics.

namespace {

// Tile size in rows/columns. 128 doubles -> 128 KiB per tile; A panel, B panel
// and C tile of one GEMM together fit in a 512 KiB L2.
constexpr size_t kTile = 128;
// Register-block shape of the GEMM micro-kernel: kMR rows x kNR columns of C
// (12 vector accumulators + 2 B vectors + 1 broadcast = 15 of 16 AVX registers).
constexpr size_t kMR = 6;
constexpr size_t kNR = 8;
// Row chunk used to split TRSM/GEMM tiles into finer work items (lcm(kMR, kNR)).
constexpr size_t kChunk = 24;

typedef double v4d __attribute__((vector_size(32)));

inline v4d load4(const double* p) {
    v4d v;
    __builtin_memcpy(&v, p, sizeof(v));
    return v;
}
inline void store4(double* p, v4d v) { __builtin_memcpy(p, &v, sizeof(v)); }
inline v4d bcast4(double x) { return v4d{x, x, x, x}; }

// Streaming (non-temporal) stores for data that is produced by one thread and
// then read by many others. Cross-core reads of lines that are dirty in a
// remote L2 are several times slower than streaming them from memory, so the
// producer writes them around its cache. stream_fence() must be issued before
// the data is published to other threads.
#if defined(__AVX__)
inline void stream8(double* p, double v) {
    long long bits;
    __builtin_memcpy(&bits, &v, sizeof(bits));
    _mm_stream_si64(reinterpret_cast<long long*>(p), bits);
}
inline void stream_fence() { _mm_sfence(); }
#else
inline void stream8(double* p, double v) { *p = v; }
inline void stream_fence() {}
#endif

// Prefetch `rows` rows of `cols` doubles at X (stride ld) for writing.
inline void prefetch_rows(const double* X, const size_t ld, const size_t rows, const size_t cols) {
    for (size_t r = 0; r < rows; ++r)
        for (size_t c = 0; c < cols; c += 8) __builtin_prefetch(X + r * ld + c, 1, 3);
}

// Busy-wait helper: spin briefly, then yield the CPU so that an oversubscribed
// or unpinned worker thread sharing this CPU can make progress.
struct Spinner {
    unsigned iter = 0;
    void relax() {
        if (++iter < 4096) {
#if defined(__x86_64__) || defined(__i386__)
            __builtin_ia32_pause();
#endif
        } else {
            std::this_thread::yield();
        }
    }
};

// Sense-reversing spin barrier on dedicated cache lines. The OpenMP runtime's
// barrier lets waiting threads fall asleep, and waking 100+ threads costs
// hundreds of microseconds per step, which dominates the small later steps.
struct SpinBarrier {
    alignas(64) std::atomic<size_t> count{0};
    alignas(64) std::atomic<size_t> gen{0};
    alignas(64) size_t n = 1;
    void wait() {
        const size_t g = gen.load(std::memory_order_acquire);
        if (count.fetch_add(1, std::memory_order_acq_rel) + 1 == n) {
            count.store(0, std::memory_order_relaxed);
            gen.store(g + 1, std::memory_order_release);
        } else {
            Spinner sp;
            while (gen.load(std::memory_order_acquire) == g) sp.relax();
        }
    }
};

// Pin the calling thread to the tid-th CPU of the process's affinity mask when
// the user has not requested an OpenMP binding policy. Unpinned spinning
// threads otherwise make the OS scheduler stack workers on busy CPUs.
inline void pin_thread_if_unbound(const size_t tid, const size_t nthreads) {
#if defined(__linux__)
    if (omp_get_proc_bind() != omp_proc_bind_false) return;
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) return;
    const size_t ncpu = (size_t)CPU_COUNT(&allowed);
    if (ncpu < nthreads) return;                  // oversubscribed: leave it to the OS
    size_t seen = 0;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (!CPU_ISSET(cpu, &allowed)) continue;
        if (seen++ == tid) {
            cpu_set_t one;
            CPU_ZERO(&one);
            CPU_SET(cpu, &one);
            sched_setaffinity(0, sizeof(one), &one);
            return;
        }
    }
#else
    (void)tid; (void)nthreads;
#endif
}

// Aligned, uninitialized buffer (pages are first-touched by the parallel
// copy-in, which spreads them over NUMA nodes). Large buffers are 2 MiB
// aligned and, where available, marked for transparent huge pages to cut
// page-fault and TLB overhead.
struct AlignedBuffer {
    double* p = nullptr;
    explicit AlignedBuffer(size_t count) {
        constexpr size_t kHuge = 2u << 20;
        const size_t align = (count * sizeof(double) >= kHuge) ? kHuge : 64;
        const size_t bytes = (count * sizeof(double) + align - 1) / align * align;
        p = static_cast<double*>(std::aligned_alloc(align, bytes));
        if (!p) {
            fprintf(stderr, "Error: out of memory\n");
            std::abort();
        }
#if defined(__linux__) && defined(MADV_HUGEPAGE)
        if (align == kHuge) madvise(p, bytes, MADV_HUGEPAGE);
#endif
    }
    ~AlignedBuffer() { std::free(p); }
    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;
};

// Pack rows [r0, r1) of the (rows x tb) tile at B (stride ld) into W-row panels,
// k-major:  P[(r / W) * (tb * W) + k * W + (r % W)] = B[r][k].
// Whoever packs the last rows also zero pads up to a multiple of W.
// With W = kNR this is the packed B^T operand of the GEMM; with W = kMR the
// packed A operand.
// Panels consumed by other threads are written with sequential streaming
// stores (see stream8); kStream = false for thread-local scratch panels.
template <size_t W, bool kStream = true>
inline void pack_panel(const double* __restrict B, const size_t ld, size_t r0, size_t r1,
                       const size_t rows, const size_t tb, double* __restrict P) {
    if (r1 == rows) r1 = (rows + W - 1) / W * W;
    for (size_t rb = r0; rb < r1; rb += W) {
        double* __restrict dst = P + (rb / W) * (tb * W);
        const double* src[W];
        for (size_t r = 0; r < W; ++r) src[r] = (rb + r < rows) ? B + (rb + r) * ld : nullptr;
        for (size_t k = 0; k < tb; ++k) {
            for (size_t r = 0; r < W; ++r) {
                const double v = src[r] ? src[r][k] : 0.0;
                if (kStream) stream8(dst + k * W + r, v);
                else dst[k * W + r] = v;
            }
        }
    }
    if (kStream) stream_fence();
}

// C(kMR rows, stride ld) -= Ap(packed kMR x depth) * Bp(packed depth x kNR).
// Writes only the first `nrows` rows and `ncols` columns of the block.
inline void gemm_micro(double* __restrict C, const double* __restrict Ap,
                       const double* __restrict Bp, const size_t ld, const size_t depth,
                       const size_t nrows, const size_t ncols) {
    // The C block is often in another core's cache; start fetching it now so
    // the latency overlaps with the k loop.
    for (size_t r = 0; r < nrows; ++r) __builtin_prefetch(C + r * ld, 1, 3);
    v4d acc0[kMR], acc1[kMR];
    for (size_t r = 0; r < kMR; ++r) {
        acc0[r] = bcast4(0.0);
        acc1[r] = bcast4(0.0);
    }
    for (size_t k = 0; k < depth; ++k) {
        const v4d b0 = load4(Bp + k * kNR);
        const v4d b1 = load4(Bp + k * kNR + 4);
        const double* __restrict ak = Ap + k * kMR;
        for (size_t r = 0; r < kMR; ++r) {
            const v4d x = bcast4(ak[r]);
            acc0[r] += x * b0;
            acc1[r] += x * b1;
        }
    }
    if (nrows == kMR && ncols == kNR) {
        for (size_t r = 0; r < kMR; ++r) {
            double* __restrict c = C + r * ld;
            store4(c, load4(c) - acc0[r]);
            store4(c + 4, load4(c + 4) - acc1[r]);
        }
    } else {
        for (size_t r = 0; r < nrows; ++r) {
            double tmp[kNR];
            store4(tmp, acc0[r]);
            store4(tmp + 4, acc1[r]);
            double* __restrict c = C + r * ld;
            for (size_t j = 0; j < ncols; ++j) c[j] -= tmp[j];
        }
    }
}

// Rows [r0, r1) of C(mrows x ncols, stride ld) -= A * B^T with A packed in Ap
// (kMR panels) and B packed in Bp (kNR panels), both of depth tb. r0 % kMR == 0.
// If `lower` is set (diagonal tile), only column blocks that intersect the lower
// triangle of each row block are updated; stray entries above the diagonal are
// never read and are dropped when the result is copied out.
inline void gemm_tile(double* __restrict C, const double* __restrict Ap,
                      const double* __restrict Bp, const size_t ld, const size_t tb,
                      const size_t r0, const size_t r1, const size_t ncols, const bool lower) {
    const size_t nblk = (ncols + kNR - 1) / kNR;
    for (size_t r = r0; r < r1; r += kMR) {
        const size_t nrows = std::min(kMR, r1 - r);
        const size_t bend = lower ? std::min(nblk, (r + nrows - 1) / kNR + 1) : nblk;
        const double* __restrict ap = Ap + (r / kMR) * (tb * kMR);
        for (size_t b = 0; b < bend; ++b) {
            gemm_micro(C + r * ld + b * kNR, ap, Bp + b * (tb * kNR), ld, tb, nrows,
                       std::min(kNR, ncols - b * kNR));
        }
    }
}

// Solve X * L^T = B in place for `nrows` (<= kMR) rows at X (stride ld), where L
// is the tb x tb lower-triangular diagonal tile at Lkk (stride ldl) and LtP is
// the same tile packed with pack_panel<kNR> (LtP panel b, row j, lane c = L[b*kNR+c][j]).
// Blocked: solve one kNR-wide column block with the small triangular system,
// then eliminate it from all later column blocks with the GEMM micro-kernel.
inline void trsm_rows(double* __restrict X, const size_t ld, const double* __restrict Lkk,
                      const size_t ldl, const double* __restrict LtP, const size_t tb,
                      const size_t nrows) {
    prefetch_rows(X, ld, nrows, tb);
    for (size_t j0 = 0; j0 < tb; j0 += kNR) {
        const size_t jn = std::min(kNR, tb - j0);
        for (size_t r = 0; r < nrows; ++r) {
            double* __restrict x = X + r * ld;
            for (size_t j = j0; j < j0 + jn; ++j) {
                const double* __restrict lj = Lkk + j * ldl;
                double s = 0.0;
                for (size_t c = j0; c < j; ++c) s += x[c] * lj[c];
                x[j] = (x[j] - s) / lj[j];
            }
        }
        if (j0 + jn >= tb) break;
        // Pack the solved kMR x jn block k-major as the A operand.
        double ap[kNR * kMR];
        for (size_t k = 0; k < jn; ++k)
            for (size_t r = 0; r < kMR; ++r)
                ap[k * kMR + r] = (r < nrows) ? X[r * ld + j0 + k] : 0.0;
        for (size_t b = (j0 + jn) / kNR; b * kNR < tb; ++b) {
            gemm_micro(X + b * kNR, ap, LtP + b * (tb * kNR) + j0 * kNR, ld, jn, nrows,
                       std::min(kNR, tb - b * kNR));
        }
    }
}

inline void trsm_tile(double* __restrict X, const size_t ld, const double* __restrict Lkk,
                      const size_t ldl, const double* __restrict LtP, const size_t tb,
                      const size_t r0, const size_t r1) {
    for (size_t r = r0; r < r1; r += kMR)
        trsm_rows(X + r * ld, ld, Lkk, ldl, LtP, tb, std::min(kMR, r1 - r));
}

// Cholesky of the diagonal tile (tb x tb block at Ad, row stride ld), right
// looking in kNR-wide column blocks: factor the small diagonal block, solve the
// rows below it, then apply the rank-kNR update to the trailing lower triangle
// with the GEMM micro-kernel.
// Returns index (relative to tile) of failing diagonal element, or -1 on success.
inline long potrf_tile(double* __restrict Ad, const size_t ld, const size_t tb) {
    constexpr size_t kPadA = (kTile + kMR - 1) / kMR * kMR;
    constexpr size_t kPadB = (kTile + kNR - 1) / kNR * kNR;
    alignas(64) double Apk[kPadA * kNR];
    alignas(64) double Bpk[kPadB * kNR];
    prefetch_rows(Ad, ld, tb, tb);
    for (size_t j0 = 0; j0 < tb; j0 += kNR) {
        const size_t jn = std::min(kNR, tb - j0);
        // Unblocked factorization of the jn x jn diagonal block (its entries
        // already include all updates from previous column blocks).
        for (size_t i = j0; i < j0 + jn; ++i) {
            double* __restrict rowi = Ad + i * ld;
            for (size_t j = j0; j < i; ++j) {
                const double* __restrict rowj = Ad + j * ld;
                double sum = 0.0;
                for (size_t k = j0; k < j; ++k) sum += rowi[k] * rowj[k];
                rowi[j] = (rowi[j] - sum) / rowj[j];
            }
            double sum = 0.0;
            for (size_t k = j0; k < i; ++k) sum += rowi[k] * rowi[k];
            const double val = rowi[i] - sum;
            if (val <= 0.0) return (long)i;
            rowi[i] = std::sqrt(val);
        }
        const size_t m0 = j0 + jn;        // first row below the diagonal block
        if (m0 >= tb) break;
        const size_t m = tb - m0;
        // Triangular solve of the rows below against the diagonal block.
        for (size_t r = m0; r < tb; ++r) {
            double* __restrict x = Ad + r * ld;
            for (size_t j = j0; j < m0; ++j) {
                const double* __restrict lj = Ad + j * ld;
                double sum = 0.0;
                for (size_t k = j0; k < j; ++k) sum += x[k] * lj[k];
                x[j] = (x[j] - sum) / lj[j];
            }
        }
        // Rank-jn update of the trailing lower triangle.
        const double* panel = Ad + m0 * ld + j0;
        pack_panel<kMR, false>(panel, ld, 0, m, m, jn, Apk);
        pack_panel<kNR, false>(panel, ld, 0, m, m, jn, Bpk);
        gemm_tile(Ad + m0 * ld + m0, Apk, Bpk, ld, jn, 0, m, m, true);
    }
    return -1;
}

} // namespace

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    if (n == 0) return true;
    double* __restrict a = A.data();

    const size_t T = std::min(kTile, n);
    const size_t nt = (n + T - 1) / T;           // number of tile rows/cols
    const size_t TT = T * T;
    auto tsz = [&](size_t t) { return std::min(T, n - t * T); };

    // Tile-major working copy of the lower triangle (row stride T inside a tile).
    AlignedBuffer W(nt * nt * TT);
    auto tile = [&](size_t i, size_t j) { return W.p + (i * nt + j) * TT; };

    // Packed copies of a block column's TRSM results: A-operand panels (kMR
    // rows) and B-operand panels (kNR rows) for the GEMM kernel. Double
    // buffered by column parity so column k+1 can be packed while column k's
    // panels are still being consumed.
    const size_t padA = (T + kMR - 1) / kMR * kMR;
    const size_t padB = (T + kNR - 1) / kNR * kNR;
    AlignedBuffer packedA(2 * nt * padA * T);
    AlignedBuffer packedB(2 * nt * padB * T);
    auto packA = [&](size_t k, size_t i) { return packedA.p + ((k & 1) * nt + i) * padA * T; };
    auto packB = [&](size_t k, size_t i) { return packedB.p + ((k & 1) * nt + i) * padB * T; };
    // Current diagonal tile L(k,k) packed as kNR panels for the TRSM kernel.
    AlignedBuffer LtP(padB * T);

    const size_t nchunks = (T + kChunk - 1) / kChunk;
    // Row chunk [r0, r1) number c of a tile with `rows` valid rows.
    auto chunk = [&](size_t c, size_t rows, size_t& r0, size_t& r1) {
        r0 = std::min(c * kChunk, rows);
        r1 = std::min(r0 + kChunk, rows);
    };

    // Trailing-update scheduler state. Every tile (i,j) has a fixed owner
    // thread (2D cyclic over a P x Q thread grid) so that its C tile and most
    // of the panels it needs stay in that core's caches from step to step;
    // idle threads steal from other owners' lists for load balance.
    // Per-thread item lists live in one flat buffer sized and touched up front
    // by a single thread: 100+ threads doing their first malloc concurrently
    // serialize in the allocator for tens of milliseconds.
    struct WorkItem { unsigned i, j, r0, r1; };
    std::vector<WorkItem> ownBuf;
    std::vector<size_t> ownSize;                  // per owner, padded to 128 B
    std::vector<size_t> ownNext;                  // per owner, padded to 128 B
    constexpr size_t kPad = 16;
    size_t ownCap = 0;                            // items per thread
    // Per-step counters, double buffered by step parity: the counters of step
    // k+1 are reset by thread 0 during step k, and the barrier that ends step k
    // publishes them before anyone starts step k+1.
    struct alignas(64) StepCounters {
        alignas(64) std::atomic<size_t> aNext{0};       // phase A item counter
        alignas(64) std::atomic<size_t> remaining{0};   // unclaimed phase B GEMM items
        alignas(64) std::atomic<size_t> trsmNext{0};    // phase B TRSM item counter
        alignas(64) std::atomic<int> potrfState{0};     // 0 pending, 1 done, 2 failed
        alignas(64) std::atomic<size_t> diagDone{0};    // finished phase A chunks of the diagonal tile
    } counters[2];
    SpinBarrier barrier;

    // Number of trailing-update items (tiles or row chunks) in step k, and
    // whether tiles are split into row chunks (few tiles: better balance).
    auto splitStep = [&](size_t k) {
        const size_t m = nt - k - 2;
        return m * (m + 1) / 2 < 2 * (size_t)omp_get_num_threads();
    };
    auto countItems = [&](size_t k) {
        size_t total = 0;
        const bool split = splitStep(k);
        for (size_t i = k + 2; i < nt; ++i) {
            size_t perTile = 1;
            if (split) {
                perTile = 0;
                for (size_t c = 0; c < nchunks; ++c) {
                    size_t r0, r1;
                    chunk(c, tsz(i), r0, r1);
                    if (r0 < r1) ++perTile;
                }
            }
            total += perTile * (i - k - 1);          // tiles j = k+2..i
        }
        return total;
    };
    auto resetCounters = [&](size_t k) {
        StepCounters& c = counters[k & 1];
        c.aNext.store(0, std::memory_order_relaxed);
        c.remaining.store(countItems(k), std::memory_order_relaxed);
        c.trsmNext.store(0, std::memory_order_relaxed);
        c.potrfState.store(0, std::memory_order_relaxed);
        c.diagDone.store(0, std::memory_order_relaxed);
    };

    long failTile = -1;      // first failing diagonal tile
    long failIdx = -1;       // index within that tile

#pragma omp parallel
    {
        const size_t nthreads = (size_t)omp_get_num_threads();
        const size_t tid = (size_t)omp_get_thread_num();
        pin_thread_if_unbound(tid, nthreads);
        size_t Q = (size_t)std::sqrt((double)nthreads);
        while (nthreads % Q != 0) --Q;
        const size_t P = nthreads / Q;
        const size_t myP = tid / Q, myQ = tid % Q;
#pragma omp single
        {
            // Upper bound on the items one thread can own in a step.
            ownCap = ((nt + P - 1) / P) * ((nt + Q - 1) / Q) * nchunks;
            ownBuf.assign(nthreads * ownCap, WorkItem{0, 0, 0, 0});
            ownSize.assign(nthreads * kPad, 0);
            ownNext.assign(nthreads * kPad, 0);
            barrier.n = nthreads;
            resetCounters(0);
        }
        WorkItem* const myList = ownBuf.data() + tid * ownCap;

        // First-touch the packed-panel buffers now, in parallel, so that page
        // faults do not land on the critical path of the first steps.
        {
            const size_t nA = 2 * nt * padA * T, nB = 2 * nt * padB * T;
#pragma omp for schedule(static) nowait
            for (size_t x = 0; x < nA; x += 512) std::memset(packedA.p + x, 0, std::min<size_t>(512, nA - x) * sizeof(double));
#pragma omp for schedule(static) nowait
            for (size_t x = 0; x < nB; x += 512) std::memset(packedB.p + x, 0, std::min<size_t>(512, nB - x) * sizeof(double));
        }

        // Copy the lower triangle (incl. diagonal tiles) into tile-major layout.
#pragma omp for schedule(dynamic, 1)
        for (size_t t = 0; t < nt * nt; ++t) {
            const size_t i = t / nt, j = t % nt;
            if (j > i) continue;
            double* __restrict dst = tile(i, j);
            const double* __restrict src = a + (i * T) * n + j * T;
            const size_t rows = tsz(i), cols = tsz(j);
            for (size_t r = 0; r < rows; ++r)
                std::memcpy(dst + r * T, src + r * n, cols * sizeof(double));
        }

        // Factor the first diagonal tile.
#pragma omp single
        {
            const long r = potrf_tile(tile(0, 0), T, tsz(0));
            if (r >= 0) { failTile = 0; failIdx = r; }
            else pack_panel<kNR>(tile(0, 0), T, 0, tsz(0), tsz(0), tsz(0), LtP.p);
        }
        // (implicit barrier)

        if (failTile < 0) {
            // Triangular solves of block column 0 + packing.
            const size_t t0 = tsz(0);
#pragma omp for schedule(dynamic, 1)
            for (size_t w = 0; w < (nt - 1) * nchunks; ++w) {
                const size_t i = 1 + w / nchunks;
                size_t r0, r1;
                chunk(w % nchunks, tsz(i), r0, r1);
                if (r0 >= r1) continue;
                trsm_tile(tile(i, 0), T, tile(0, 0), T, LtP.p, t0, r0, r1);
                pack_panel<kMR>(tile(i, 0), T, r0, r1, tsz(i), t0, packA(0, i));
                pack_panel<kNR>(tile(i, 0), T, r0, r1, tsz(i), t0, packB(0, i));
            }
        }

        // Step k: block column k has been solved and packed. Apply its update
        // to column k+1, factor the new diagonal tile, and while the trailing
        // update of the remaining columns runs, solve and pack column k+1.
        for (size_t k = 0; k + 1 < nt && failTile < 0; ++k) {
            const size_t tk = tsz(k);
            const size_t j1 = k + 1;
            const size_t tj1 = tsz(j1);
            const size_t below = nt - j1;

            StepCounters& sh = counters[k & 1];
            if (tid == 0 && k + 2 < nt) resetCounters(k + 1);

            // Build this thread's list of owned trailing-update items for
            // j >= k+2 (published to stealers by the barrier after phase A).
            {
                size_t cnt = 0;
                const bool split = splitStep(k);
                for (size_t i = k + 2; i < nt; ++i) {
                    const size_t rows = tsz(i);
                    if (i % P != myP) continue;
                    for (size_t j = k + 2; j <= i; ++j) {
                        if (j % Q != myQ) continue;
                        if (!split) {
                            myList[cnt++] = {(unsigned)i, (unsigned)j, 0u, (unsigned)rows};
                        } else {
                            for (size_t c = 0; c < nchunks; ++c) {
                                size_t r0, r1;
                                chunk(c, rows, r0, r1);
                                if (r0 < r1) myList[cnt++] = {(unsigned)i, (unsigned)j, (unsigned)r0, (unsigned)r1};
                            }
                        }
                    }
                }
                ownSize[tid * kPad] = cnt;
                ownNext[tid * kPad] = 0;
            }
            // POTRF(k+1): run by one thread as soon as the diagonal tile's
            // phase A chunks are complete (they are scheduled first; every
            // chunk index of the tile is counted, empty ones included).
            const size_t potrfThread = j1 % nthreads;
            const size_t ndiag = nchunks;
            auto doPotrf = [&]() {
                const long r = potrf_tile(tile(j1, j1), T, tj1);
                if (r >= 0) {
                    failTile = (long)j1;
                    failIdx = r;
                    sh.potrfState.store(2, std::memory_order_release);
                } else {
                    pack_panel<kNR>(tile(j1, j1), T, 0, tj1, tj1, tj1, LtP.p);
                    sh.potrfState.store(1, std::memory_order_release);
                }
            };

            // Phase A: A(i,k+1) -= A(i,k) * A(k+1,k)^T for i >= k+1 (row chunks),
            // dynamically scheduled through a shared counter.
            {
                bool potrfDone = false;
                const size_t nitems = below * nchunks;
                while (true) {
                    if (tid == potrfThread && !potrfDone &&
                        sh.diagDone.load(std::memory_order_acquire) == ndiag) {
                        doPotrf();
                        potrfDone = true;
                    }
                    const size_t w = sh.aNext.fetch_add(1, std::memory_order_relaxed);
                    if (w >= nitems) break;
                    const size_t i = j1 + w / nchunks;
                    size_t r0, r1;
                    chunk(w % nchunks, tsz(i), r0, r1);
                    if (r0 < r1)
                        gemm_tile(tile(i, j1), packA(k, i), packB(k, j1), T, tk, r0, r1, tj1, i == j1);
                    if (i == j1) sh.diagDone.fetch_add(1, std::memory_order_acq_rel);
                }
                if (tid == potrfThread && !potrfDone) {
                    Spinner sp;
                    while (sh.diagDone.load(std::memory_order_acquire) != ndiag) sp.relax();
                    doPotrf();
                }
            }
            barrier.wait();

            // TRSM items of column k+1: row chunks of tiles i > k+1. They are
            // on the critical path of the next step, so take them as soon as
            // the diagonal tile is ready.
            const size_t ntrsm = (nt - j1 - 1) * nchunks;
            auto tryTrsm = [&]() -> bool {
                if (sh.potrfState.load(std::memory_order_acquire) != 1) return false;
                const size_t w = sh.trsmNext.fetch_add(1, std::memory_order_relaxed);
                if (w >= ntrsm) return false;
                const size_t i = j1 + 1 + w / nchunks;
                const size_t rows = tsz(i);
                size_t r0, r1;
                chunk(w % nchunks, rows, r0, r1);
                if (r0 < r1) {
                    double* Ai = tile(i, j1);
                    trsm_tile(Ai, T, tile(j1, j1), T, LtP.p, tj1, r0, r1);
                    pack_panel<kMR>(Ai, T, r0, r1, rows, tj1, packA(j1, i));
                    pack_panel<kNR>(Ai, T, r0, r1, rows, tj1, packB(j1, i));
                }
                return true;
            };

            // Phase B.2: trailing update A(i,j) -= A(i,k) A(j,k)^T, j >= k+2.
            // Own items first, then steal while unclaimed items remain;
            // interleave TRSM items as they become available.
            for (size_t v = 0; v < nthreads; ++v) {
                const size_t victim = (tid + v) % nthreads;
                const WorkItem* const lst = ownBuf.data() + victim * ownCap;
                const size_t lstSize = ownSize[victim * kPad];
                if (v > 0) {
                    if (sh.remaining.load(std::memory_order_acquire) == 0) break;
                    // Cheap read-only probe before touching the victim's counter.
                    if (__atomic_load_n(&ownNext[victim * kPad], __ATOMIC_RELAXED) >= lstSize) continue;
                }
                while (true) {
                    while (tryTrsm()) {}
                    const size_t idx = __atomic_fetch_add(&ownNext[victim * kPad], 1, __ATOMIC_RELAXED);
                    if (idx >= lstSize) break;
                    sh.remaining.fetch_sub(1, std::memory_order_acq_rel);
                    const WorkItem it = lst[idx];
                    gemm_tile(tile(it.i, it.j), packA(k, it.i), packB(k, it.j), T, tk,
                              it.r0, it.r1, tsz(it.j), it.i == it.j);
                }
            }
            // Drain the TRSM items (waiting for the diagonal tile if needed).
            {
                Spinner sp;
                while (sh.potrfState.load(std::memory_order_acquire) == 0) sp.relax();
            }
            while (tryTrsm()) {}
            barrier.wait();
            // (publishes column k+1, the packed panels and the failure flag)
        }

        if (failTile < 0) {
            // Copy L back row by row (contiguous writes): lower tiles verbatim,
            // the diagonal element's row prefix, and zeros above the diagonal.
#pragma omp for schedule(static)
            for (size_t row = 0; row < n; ++row) {
                const size_t i = row / T, r = row % T;
                double* __restrict dst = a + row * n;
                for (size_t j = 0; j < i; ++j)
                    std::memcpy(dst + j * T, tile(i, j) + r * T, tsz(j) * sizeof(double));
                std::memcpy(dst + i * T, tile(i, i) + r * T, (r + 1) * sizeof(double));
                std::memset(dst + row + 1, 0, (n - row - 1) * sizeof(double));
            }
        }
    }

    if (failTile >= 0) {
        const size_t jglob = (size_t)failTile * T + (size_t)failIdx;
        // Matrix is not positive definite
        printf("Error: Matrix is not positive definite at diagonal element %zu\n", jglob);
        return false;
    }
    return true;
}

// Generate a symmetric positive definite matrix
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    // Method: Create A = B * B^T where B is random
    // This guarantees A is positive semi-definite
    // Then add identity to make it strictly positive definite
    
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    
    // Generate random matrix B (sequential: rand_r stream must be reproducible)
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }
    
    // Compute A = B * B^T (each element independent; same summation order as before)
    const double* __restrict b = B.data();
    double* __restrict a = A.data();
#pragma omp parallel for schedule(dynamic, 8)
    for (size_t i = 0; i < n; ++i) {
        const double* __restrict bi = b + i * n;
        for (size_t j = 0; j < n; ++j) {
            const double* __restrict bj = b + j * n;
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += bi[k] * bj[k];
            }
            a[i * n + j] = sum;
        }
    }
    
    // Add diagonal dominance to ensure positive definiteness
    for (size_t i = 0; i < n; ++i) {
        A[i * n + i] += n;
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix
    
    std::vector<double> reconstructed(n * n);
    const double* __restrict l = L.data();
    double* __restrict rec = reconstructed.data();
    
    // Compute L * L^T
#pragma omp parallel for schedule(dynamic, 8)
    for (size_t i = 0; i < n; ++i) {
        const double* __restrict li = l + i * n;
        for (size_t j = 0; j < n; ++j) {
            const double* __restrict lj = l + j * n;
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += li[k] * lj[k];
            }
            rec[i * n + j] = sum;
        }
    }
    
    // Compare with original
    double maxError = 0.0;
    double relError = 0.0;
    
#pragma omp parallel for reduction(max : maxError, relError)
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
    
    printf("Cholesky Decomposition Benchmark\n");
    printf("Matrix size: %zu x %zu\n", n, n);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    printf("OpenMP threads: %d\n", omp_get_max_threads());
    
    // Allocate matrix
    std::vector<double> A(n * n);
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
