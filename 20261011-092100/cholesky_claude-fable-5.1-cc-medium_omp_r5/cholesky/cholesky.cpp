#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#include <omp.h>

#if defined(__linux__)
#include <sched.h>
#include <sys/mman.h>
#endif

#include "../common/results_output.hpp"

// Blocked, right-looking Cholesky decomposition parallelized with OpenMP.
// Decomposes positive definite matrix A (row-major, n x n) in place into L * L^T,
// where L is lower triangular; the strictly upper triangle is set to zero.
//
// The lower triangle is copied into a tile-contiguous working buffer (each
// nb x nb tile is stored contiguously, backed by transparent huge pages where
// available) so that every tile operation streams through a compact memory
// region instead of touching nb rows of a huge row-major matrix. For every tile
// column k the operations are:
//   POTRF  : factor the diagonal tile (k,k)
//   TRSM   : solve the tiles below it, (i,k) for i > k
//   GEMM   : update the trailing tiles (i,j), k < j <= i
// They are scheduled dynamically by a dependency-driven scheduler (see
// TileScheduler) running inside one OpenMP parallel region: tile columns
// overlap, the critical path (POTRF/TRSM) is prioritized, GEMM updates of a
// tile prefer the thread that owns the tile (cache locality), and idle threads
// steal work, so all cores stay busy without global barriers.
// Each tile is copied in by its first operation and copied back to A by its
// last one; the strictly upper triangle of A is zeroed by independent tasks.
// Updates are applied in a fixed order, so the result does not depend on the
// number of threads or on the scheduling.

namespace {

// Register-tile sizes for the GEMM micro-kernel (6 rows x 8 columns of C):
// 12 accumulator vectors + 2 B vectors + 1 broadcast fit in 16 AVX registers.
constexpr size_t MR = 6;
constexpr size_t NR = 8;

// Portable SIMD vector of 4 doubles (GCC/Clang vector extension; maps to one
// AVX register on x86-64 with AVX/AVX2, or is split by the compiler otherwise).
typedef double v4d __attribute__((vector_size(32)));

inline v4d loadv(const double* p) {
    v4d v;
    __builtin_memcpy(&v, p, sizeof(v));
    return v;
}
inline void storev(double* p, const v4d v) { __builtin_memcpy(p, &v, sizeof(v)); }
inline v4d splat(const double x) { return v4d{x, x, x, x}; }

// Select a tile size based on the problem size: large tiles have a higher
// arithmetic intensity (less memory traffic per flop), small tiles expose more
// parallelism and a shorter critical path for small matrices.
// All sizes are multiples of lcm(MR, NR) = 24 so full tiles have no edges.
size_t chooseTileSize(const size_t n) {
    if (n >= 5120) return 144;
    if (n >= 2560) return 120;
    if (n >= 1536) return 96;
    return 72;
}

// Thread-local scratch buffer used for transposed tiles.
double* scratch(const size_t count) {
    static thread_local std::vector<double> buf;
    if (buf.size() < count) buf.resize(count);
    return buf.data();
}

inline void cpuRelax() {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#else
    std::this_thread::yield();
#endif
}

// Number of threads used for the factorization. The GEMM kernel saturates the
// floating-point units of a core, so SMT siblings only compete with each other;
// unless the user pinned the count with OMP_NUM_THREADS, use one thread per
// physical core (falls back to the OpenMP default if the topology is unknown).
int choleskyThreads() {
    static int cached = 0;
    if (cached > 0) return cached;
    int threads = omp_get_max_threads();
    if (std::getenv("OMP_NUM_THREADS") == nullptr) {
#if defined(__linux__)
        cpu_set_t mask;
        CPU_ZERO(&mask);
        if (sched_getaffinity(0, sizeof(mask), &mask) == 0) {
            int physical = 0;
            for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
                if (!CPU_ISSET(cpu, &mask)) continue;
                char path[128];
                snprintf(path, sizeof(path),
                         "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", cpu);
                FILE* f = fopen(path, "r");
                if (f == nullptr) {
                    physical = 0;  // topology unknown: keep the OpenMP default
                    break;
                }
                int first = -1;
                const int got = fscanf(f, "%d", &first);
                fclose(f);
                if (got != 1) {
                    physical = 0;
                    break;
                }
                // Count a core once: when its lowest-numbered sibling is in the mask,
                // it is this cpu; otherwise count the cpu itself.
                if (first == cpu || !CPU_ISSET(first, &mask)) ++physical;
            }
            if (physical > 0) threads = std::min(threads, physical);
        }
#endif
    }
    cached = std::max(1, threads);
    return cached;
}

// Large aligned allocation that asks the kernel for transparent huge pages
// (with 4 KiB pages the tile kernels suffer from TLB misses). The buffer is
// kept as a process-wide workspace and grown on demand: releasing hundreds of
// megabytes back to the kernel costs tens of milliseconds, which should not be
// paid inside the factorization.
class Workspace {
public:
    static double* get(const size_t count) {
        static Workspace ws;
        if (ws.count_ < count) ws.reallocate(count);
        return ws.data_;
    }

private:
    Workspace() = default;
    ~Workspace() { std::free(data_); }
    Workspace(const Workspace&) = delete;
    Workspace& operator=(const Workspace&) = delete;

    void reallocate(const size_t count) {
        std::free(data_);
        const size_t align = size_t(1) << 21;  // 2 MiB
        size_t bytes = count * sizeof(double);
        bytes = (bytes + align - 1) / align * align;
        if (bytes == 0) bytes = align;
        data_ = static_cast<double*>(std::aligned_alloc(align, bytes));
        if (data_ == nullptr) {
            fprintf(stderr, "Error: out of memory\n");
            std::exit(1);
        }
        count_ = bytes / sizeof(double);
#if defined(__linux__) && defined(MADV_HUGEPAGE)
        madvise(data_, bytes, MADV_HUGEPAGE);  // best effort
#endif
    }

    double* data_ = nullptr;
    size_t count_ = 0;
};

// ---------------------------------------------------------------------------
// Dense kernels on tiles. All tiles are stored row-major with leading
// dimension `ld` (== nb for the working buffer).
// ---------------------------------------------------------------------------

// Micro-kernel: C[0..MR)[0..NR) -= sum_{t<kk} a[m*lda + t] * bt[t*ldb + q].
// `a` points at MR consecutive rows (stride lda) of the left operand,
// `bt` at kk consecutive rows (stride ldb) of the transposed right operand.
inline void microKernel(const double* __restrict a, const size_t lda, const double* __restrict bt,
                        const size_t ldb, const size_t kk, double* __restrict c, const size_t ldc) {
    v4d acc00 = {0, 0, 0, 0}, acc01 = {0, 0, 0, 0};
    v4d acc10 = {0, 0, 0, 0}, acc11 = {0, 0, 0, 0};
    v4d acc20 = {0, 0, 0, 0}, acc21 = {0, 0, 0, 0};
    v4d acc30 = {0, 0, 0, 0}, acc31 = {0, 0, 0, 0};
    v4d acc40 = {0, 0, 0, 0}, acc41 = {0, 0, 0, 0};
    v4d acc50 = {0, 0, 0, 0}, acc51 = {0, 0, 0, 0};
    const double* __restrict a0 = a;
    const double* __restrict a1 = a + lda;
    const double* __restrict a2 = a + 2 * lda;
    const double* __restrict a3 = a + 3 * lda;
    const double* __restrict a4 = a + 4 * lda;
    const double* __restrict a5 = a + 5 * lda;
    for (size_t t = 0; t < kk; ++t) {
        const v4d b0 = loadv(bt + t * ldb);
        const v4d b1 = loadv(bt + t * ldb + 4);
        v4d v;
        v = splat(a0[t]); acc00 += v * b0; acc01 += v * b1;
        v = splat(a1[t]); acc10 += v * b0; acc11 += v * b1;
        v = splat(a2[t]); acc20 += v * b0; acc21 += v * b1;
        v = splat(a3[t]); acc30 += v * b0; acc31 += v * b1;
        v = splat(a4[t]); acc40 += v * b0; acc41 += v * b1;
        v = splat(a5[t]); acc50 += v * b0; acc51 += v * b1;
    }
    double* __restrict c0 = c;
    double* __restrict c1 = c + ldc;
    double* __restrict c2 = c + 2 * ldc;
    double* __restrict c3 = c + 3 * ldc;
    double* __restrict c4 = c + 4 * ldc;
    double* __restrict c5 = c + 5 * ldc;
    storev(c0, loadv(c0) - acc00); storev(c0 + 4, loadv(c0 + 4) - acc01);
    storev(c1, loadv(c1) - acc10); storev(c1 + 4, loadv(c1 + 4) - acc11);
    storev(c2, loadv(c2) - acc20); storev(c2 + 4, loadv(c2 + 4) - acc21);
    storev(c3, loadv(c3) - acc30); storev(c3 + 4, loadv(c3 + 4) - acc31);
    storev(c4, loadv(c4) - acc40); storev(c4 + 4, loadv(c4 + 4) - acc41);
    storev(c5, loadv(c5) - acc50); storev(c5 + 4, loadv(c5 + 4) - acc51);
}

// Generic (edge) kernel for the same operation on an arbitrary rows x cols block.
inline void edgeKernel(const double* __restrict a, const size_t lda, const double* __restrict bt,
                       const size_t ldb, const size_t kk, double* __restrict c, const size_t ldc,
                       const size_t rows, const size_t cols) {
    for (size_t m = 0; m < rows; ++m) {
        const double* __restrict am = a + m * lda;
        double* __restrict cm = c + m * ldc;
        for (size_t q = 0; q < cols; ++q) {
            double s = 0.0;
            for (size_t t = 0; t < kk; ++t) {
                s += am[t] * bt[t * ldb + q];
            }
            cm[q] -= s;
        }
    }
}

// Block update: C[rows x cols] -= A[rows x kk] * Bt[kk x cols], tiled MR x NR.
// Columns are processed in panels narrow enough that the Bt panel (kk x NC
// doubles) stays resident in the core's L2 cache while all rows sweep over it;
// otherwise Bt would be streamed from the shared L3 once per row group, which
// saturates the L3 when all cores of a complex are busy.
void blockUpdate(const double* __restrict a, const size_t lda, const double* __restrict bt,
                 const size_t ldb, const size_t kk, double* __restrict c, const size_t ldc,
                 const size_t rows, const size_t cols) {
    constexpr size_t PANEL_BYTES = 64 * 1024;
    size_t NC = PANEL_BYTES / (std::max(kk, (size_t)1) * sizeof(double)) / NR * NR;
    NC = std::clamp(NC, NR, (size_t)192);
    const size_t rFull = rows - rows % MR;
    const size_t cFull = cols - cols % NR;
    for (size_t q0 = 0; q0 < cFull; q0 += NC) {
        const size_t q1 = std::min(cFull, q0 + NC);
        for (size_t r = 0; r < rFull; r += MR) {
            for (size_t q = q0; q < q1; q += NR) {
                microKernel(a + r * lda, lda, bt + q, ldb, kk, c + r * ldc + q, ldc);
            }
        }
    }
    if (cFull < cols) edgeKernel(a, lda, bt + cFull, ldb, kk, c + cFull, ldc, rFull, cols - cFull);
    if (rFull < rows) edgeKernel(a + rFull * lda, lda, bt, ldb, kk, c + rFull * ldc, ldc, rows - rFull, cols);
}

// Unblocked Cholesky factorization of a diagonal block T (m x m, leading
// dimension ld). Returns the local index of the first non-positive pivot, or
// SIZE_MAX on success.
size_t potrfUnblocked(double* __restrict T, const size_t ld, const size_t m) {
    for (size_t j = 0; j < m; ++j) {
        double* __restrict rowj = T + j * ld;
        double sum = 0.0;
#pragma omp simd reduction(+ : sum)
        for (size_t t = 0; t < j; ++t) {
            sum += rowj[t] * rowj[t];
        }
        const double val = rowj[j] - sum;
        if (val <= 0.0) {
            return j;
        }
        const double ljj = sqrt(val);
        rowj[j] = ljj;
        const double inv = 1.0 / ljj;
        for (size_t i = j + 1; i < m; ++i) {
            double* __restrict rowi = T + i * ld;
            double s = 0.0;
#pragma omp simd reduction(+ : s)
            for (size_t t = 0; t < j; ++t) {
                s += rowi[t] * rowj[t];
            }
            rowi[j] = (rowi[j] - s) * inv;
        }
    }
    return SIZE_MAX;
}

void trsmTile(double* X, size_t ldx, size_t rows, const double* L, size_t ldl, size_t m);
void blockUpdate(const double* a, size_t lda, const double* bt, size_t ldb, size_t kk, double* c,
                 size_t ldc, size_t rows, size_t cols);

// Blocked Cholesky factorization of a diagonal tile T (m x m, leading dimension
// ld): the tile is processed in IB-wide panels with the fast TRSM/GEMM kernels,
// only the IB x IB diagonal blocks use the unblocked algorithm. Returns the
// local index of the first non-positive pivot, or SIZE_MAX on success.
size_t potrfTile(double* __restrict T, const size_t ld, const size_t m) {
    constexpr size_t IB = 48;  // multiple of lcm(MR, NR)
    for (size_t cb = 0; cb < m; cb += IB) {
        const size_t ib = std::min(IB, m - cb);
        double* __restrict diag = T + cb * ld + cb;
        const size_t bad = potrfUnblocked(diag, ld, ib);
        if (bad != SIZE_MAX) return cb + bad;
        const size_t rest = m - cb - ib;
        if (rest == 0) break;
        // Panel below the diagonal block: rows [cb+ib, m), columns [cb, cb+ib).
        double* __restrict panel = T + (cb + ib) * ld + cb;
        trsmTile(panel, ld, rest, diag, ld, ib);
        // Transpose the panel (Pt[t][r] = panel[r][t]) for the trailing update.
        double* __restrict Pt = scratch(ib * rest);
        for (size_t r = 0; r < rest; ++r) {
            for (size_t t = 0; t < ib; ++t) {
                Pt[t * rest + r] = panel[r * ld + t];
            }
        }
        // Trailing update of the lower part, one IB-row block at a time:
        // C(rows rb.., cols cb+ib .. rb+rows) -= panel(rows) * panel(cols)^T.
        for (size_t rb = cb + ib; rb < m; rb += IB) {
            const size_t rows = std::min(IB, m - rb);
            const size_t cols = rb + rows - (cb + ib);
            blockUpdate(T + rb * ld + cb, ld, Pt, rest, ib, T + rb * ld + cb + ib, ld, rows, cols);
        }
    }
    return SIZE_MAX;
}

// Triangular solve: X (rows x m, leading dimension ldx) := X * L^-T, where L
// is the already factored lower-triangular m x m tile (leading dimension ldl).
// Columns are processed in panels of NR: a small forward substitution within
// the panel, then a rank-NR update of the remaining columns via the GEMM kernel.
void trsmTile(double* __restrict X, const size_t ldx, const size_t rows, const double* __restrict L,
              const size_t ldl, const size_t m) {
    // Transpose L so the update loops are contiguous: Lt[c][c2] = L[c2][c].
    double* __restrict Lt = scratch(m * m);
    for (size_t c2 = 0; c2 < m; ++c2) {
        const double* __restrict Lrow = L + c2 * ldl;
        for (size_t c = 0; c <= c2; ++c) {
            Lt[c * m + c2] = Lrow[c];
        }
    }
    for (size_t cb = 0; cb < m; cb += NR) {
        const size_t pw = std::min(NR, m - cb);  // panel width
        // Forward substitution inside the panel for every row.
        for (size_t r = 0; r < rows; ++r) {
            double* __restrict x = X + r * ldx + cb;
            for (size_t c = 0; c < pw; ++c) {
                const double* __restrict lt = Lt + (cb + c) * m + cb;
                const double xc = x[c] / lt[c];
                x[c] = xc;
                for (size_t c2 = c + 1; c2 < pw; ++c2) {
                    x[c2] -= xc * lt[c2];
                }
            }
        }
        // Rank-pw update of the columns to the right of the panel.
        const size_t rest = m - cb - pw;
        if (rest > 0) {
            blockUpdate(X + cb, ldx, Lt + cb * m + cb + pw, m, pw, X + cb + pw, ldx, rows, rest);
        }
    }
}

// Trailing update: C (rows x cols) -= A (rows x kk) * B (cols x kk)^T.
void gemmTile(double* __restrict C, const size_t ldc, const double* __restrict A, const size_t lda,
              const double* __restrict B, const size_t ldb, const size_t rows, const size_t cols,
              const size_t kk) {
    // Transpose B so that Bt[t][c] = B[c][t] is contiguous in c.
    double* __restrict Bt = scratch(kk * cols);
    for (size_t c = 0; c < cols; ++c) {
        const double* __restrict brow = B + c * ldb;
        for (size_t t = 0; t < kk; ++t) {
            Bt[t * cols + c] = brow[t];
        }
    }
    blockUpdate(A, lda, Bt, cols, kk, C, ldc, rows, cols);
}

// ---------------------------------------------------------------------------
// Dependency-driven tile scheduler.
//
// Every lower tile (i,j), i >= j, goes through j+1 ordered steps:
//   steps 0..j-1 : GEMM update with tile column k = step
//   step  j      : POTRF (i == j) or TRSM (i > j)
// Step s of tile (i,j) is ready when the previous step of the tile is done and
//   s <  j : TRSM(i,s) and TRSM(j,s) are done
//   s == j : POTRF(j) is done (nothing extra for the diagonal tile itself)
// Updates of a tile are applied in increasing k, so the result is bitwise
// reproducible regardless of thread count or scheduling.
//
// The first step of a tile first copies the tile in from the row-major matrix;
// the last step writes the finished L tile back. Zeroing of the strictly upper
// triangle outside the diagonal tiles is done by independent "zero" tasks.
//
// Operations on the critical path (TRSM, and the GEMMs feeding the next tile
// column) are split into several row parts that run on different threads, which
// shortens the critical path; the last part to finish publishes completion.
//
// Ready work lives in lock-free FIFO queues: one global queue for the
// critical path and one queue per thread for the GEMM updates of the tiles
// that thread owns (2D block-cyclic ownership, so consecutive tasks of a thread
// share operands and a tile tends to stay in that core's caches). Threads take
// critical work first, then their own queue, then steal. Completion of an
// operation publishes its flag and tries to enqueue every dependent tile; the
// per-tile `claimed` counter guarantees each step is enqueued exactly once.
// ---------------------------------------------------------------------------

// Lock-free multi-producer / multi-consumer FIFO with a fixed capacity that is
// never exceeded (every work item of the DAG is pushed at most once).
class ReadyQueue {
public:
    ReadyQueue() = default;
    void init(const size_t capacity) {
        slots_ = std::vector<std::atomic<int>>(capacity);
        for (auto& s : slots_) s.store(0, std::memory_order_relaxed);
    }
    void push(const int item) {
        const size_t t = tail_.fetch_add(1, std::memory_order_acq_rel);
        slots_[t].store(item + 1, std::memory_order_release);
    }
    bool empty() const {
        return head_.load(std::memory_order_acquire) >= tail_.load(std::memory_order_acquire);
    }
    // Returns -1 if no item is currently available.
    int tryPop() {
        size_t h = head_.load(std::memory_order_acquire);
        for (;;) {
            if (h >= tail_.load(std::memory_order_acquire)) return -1;
            if (head_.compare_exchange_weak(h, h + 1, std::memory_order_acq_rel)) break;
        }
        int v;
        while ((v = slots_[h].load(std::memory_order_acquire)) == 0) cpuRelax();
        return v - 1;
    }

private:
    std::vector<std::atomic<int>> slots_;
    alignas(64) std::atomic<size_t> head_{0};
    alignas(64) std::atomic<size_t> tail_{0};
};

class TileScheduler {
public:
    static constexpr int MAX_PARTS = 8;
    static constexpr size_t MIN_PART_ROWS = 48;

    // `a` is the row-major n x n input/output matrix; `tiles` is the
    // tile-contiguous lower triangle: tile (i,j) at tiles + tileIndex(i,j)*nb*nb,
    // leading dimension nb.
    TileScheduler(double* a, double* tiles, const size_t n, const size_t nb, const int numThreads)
        : a_(a), tiles_(tiles), n_(n), nb_(nb), nt_((n + nb - 1) / nb), numThreads_(numThreads),
          step_(nt_ * nt_), claimed_(nt_ * nt_), done_(nt_ * nt_), pending_(nt_ * nt_),
          local_(numThreads), total_(totalOps(nt_) + nt_) {
        for (size_t t = 0; t < nt_ * nt_; ++t) {
            step_[t].store(0, std::memory_order_relaxed);
            claimed_[t].store(0, std::memory_order_relaxed);
            done_[t].store(0, std::memory_order_relaxed);
            pending_[t].store(0, std::memory_order_relaxed);
        }
        // 2D block-cyclic ownership grid.
        gridP_ = std::max(1, (int)std::sqrt((double)numThreads));
        gridQ_ = std::max(1, numThreads / gridP_);
        // Size every queue exactly: each work item is pushed to one known
        // queue, so count the items per queue up front (cheap, O(nt^2)).
        size_t criticalItems = 0;
        std::vector<size_t> localItems(numThreads, 0);
        for (size_t i = 0; i < nt_; ++i) {
            for (size_t j = 0; j <= i; ++j) {
                if (i == j) {
                    criticalItems += 1 + (j >= 1 ? 1 : 0);  // POTRF + GEMM of step j-1
                } else {
                    const size_t parts = (size_t)numParts(tileSize(i), true);
                    criticalItems += parts * (1 + (j >= 1 ? 1 : 0));  // TRSM + GEMM of step j-1
                }
                if (j >= 2) localItems[owner(i, j)] += j - 1;  // GEMM steps 0..j-2
            }
        }
        for (size_t i = 0; i < nt_; ++i) localItems[i % (size_t)numThreads] += 1;  // zero tasks
        critical_.init(std::max<size_t>(1, criticalItems));
        for (int t = 0; t < numThreads; ++t) local_[t].init(std::max<size_t>(1, localItems[t]));
    }

    static size_t tileIndex(const size_t i, const size_t j) { return i * (i + 1) / 2 + j; }

    // Executed by every thread of the parallel region.
    void run(const int me) {
        // Seed the DAG: POTRF of the first diagonal tile, and the independent
        // zero tasks (one per tile row), dealt round-robin to the local queues.
        // Thread 0 seeds everything so that the DAG completes even if the
        // runtime provides fewer threads than requested (idle queues of
        // missing threads are drained by stealing).
        if (me == 0) {
            tryPush(0, 0);
            for (size_t i = 0; i < nt_; ++i) {
                local_[i % (size_t)numThreads_].push(encode((int)(nt_ * nt_ + i), 0));
            }
        }

        unsigned idle = 0;
        int victim = me;
        for (;;) {
            if (failed_.load(std::memory_order_relaxed) != SIZE_MAX) break;
            if (completed_.load(std::memory_order_acquire) == total_) break;
            int item = critical_.tryPop();
            if (item < 0) item = local_[me].tryPop();
            if (item < 0) {
                // Steal: scan the other threads' queues starting after the last victim.
                for (int v = 0; v < numThreads_ && item < 0; ++v) {
                    victim = (victim + 1 == numThreads_) ? 0 : victim + 1;
                    if (victim != me && !local_[victim].empty()) item = local_[victim].tryPop();
                }
            }
            if (item < 0) {
                // No ready work right now: back off (yield after a while so
                // that an oversubscribed machine does not waste CPU time).
                if (++idle < 64) {
                    cpuRelax();
                } else {
                    idle = 0;
                    std::this_thread::yield();
                }
                continue;
            }
            idle = 0;
            execute(item);
        }
    }

    size_t failedAt() const { return failed_.load(); }

private:
    static size_t totalOps(const size_t nt) {
        size_t ops = 0;
        for (size_t j = 0; j < nt; ++j) ops += (nt - j) * (j + 1);
        return ops;
    }
    static int encode(const int id, const int part) { return id * MAX_PARTS + part; }

    size_t tileSize(const size_t t) const { return std::min(nb_, n_ - t * nb_); }
    int tileId(const size_t i, const size_t j) const { return (int)(i * nt_ + j); }
    double* tile(const size_t i, const size_t j) const { return tiles_ + tileIndex(i, j) * nb_ * nb_; }
    int owner(const size_t i, const size_t j) const {
        return (int)(((i % (size_t)gridP_) * (size_t)gridQ_ + (j % (size_t)gridQ_)) % (size_t)numThreads_);
    }
    // Rows [r0, r1) of a tile with `rows` rows handled by part p of `parts`.
    static void partRange(const size_t rows, const int parts, const int p, size_t& r0, size_t& r1) {
        const size_t per = ((rows + parts - 1) / parts + MR - 1) / MR * MR;
        r0 = std::min(rows, (size_t)p * per);
        r1 = std::min(rows, r0 + per);
    }
    static int numParts(const size_t rows, const bool critical) {
        if (!critical) return 1;
        return (int)std::clamp(rows / MIN_PART_ROWS, (size_t)1, (size_t)MAX_PARTS);
    }

    // Copy rows [r0, r1) of tile (i,j) in from the row-major matrix.
    void copyIn(const size_t i, const size_t j, const size_t r0, const size_t r1) const {
        const size_t nbj = tileSize(j);
        double* __restrict dst = tile(i, j);
        const double* __restrict src = a_ + (i * nb_) * n_ + j * nb_;
        for (size_t r = r0; r < r1; ++r) {
            std::memcpy(dst + r * nb_, src + r * n_, nbj * sizeof(double));
        }
    }
    // Copy rows [r0, r1) of the finished tile (i,j) back (diagonal tile: lower
    // part only, the rest of the tile row is zeroed).
    void copyOut(const size_t i, const size_t j, const size_t r0, const size_t r1) const {
        const size_t nbj = tileSize(j);
        const double* __restrict srcTile = tile(i, j);
        double* __restrict dst = a_ + (i * nb_) * n_ + j * nb_;
        for (size_t r = r0; r < r1; ++r) {
            const size_t cnt = (i == j) ? r + 1 : nbj;
            std::memcpy(dst + r * n_, srcTile + r * nb_, cnt * sizeof(double));
            if (i == j) std::fill(dst + r * n_ + cnt, dst + r * n_ + nbj, 0.0);
        }
    }
    // Zero the strictly upper part of tile row i outside the diagonal tile.
    void zeroUpper(const size_t i) const {
        const size_t nbi = tileSize(i);
        const size_t c0 = (i + 1) * nb_;
        if (c0 >= n_) return;
        for (size_t r = 0; r < nbi; ++r) {
            double* __restrict row = a_ + (i * nb_ + r) * n_;
            std::fill(row + c0, row + n_, 0.0);
        }
    }

    void execute(const int item) {
        const int id = item / MAX_PARTS;
        const int part = item % MAX_PARTS;
        if ((size_t)id >= nt_ * nt_) {  // zero task
            zeroUpper((size_t)id - nt_ * nt_);
            completed_.fetch_add(1, std::memory_order_acq_rel);
            return;
        }
        const size_t i = (size_t)id / nt_;
        const size_t j = (size_t)id % nt_;
        const size_t s = (size_t)step_[id].load(std::memory_order_acquire);
        const size_t nbi = tileSize(i), nbj = tileSize(j);
        const int parts = pending_[id].load(std::memory_order_acquire) > 0
                              ? numParts(nbi, isCritical(j, s))
                              : 1;
        size_t r0, r1;
        partRange(nbi, parts, part, r0, r1);
        const size_t rows = r1 - r0;

        if (s == 0) copyIn(i, j, r0, r1);

        if (s < j) {
            if (rows > 0) {
                gemmTile(tile(i, j) + r0 * nb_, nb_, tile(i, s) + r0 * nb_, nb_, tile(j, s), nb_,
                         rows, nbj, tileSize(s));
            }
        } else if (i == j) {
            const size_t bad = potrfTile(tile(j, j), nb_, nbj);
            if (bad != SIZE_MAX) {
                size_t expected = SIZE_MAX;
                failed_.compare_exchange_strong(expected, j * nb_ + bad);
                return;  // successors are never released; all threads stop
            }
            copyOut(i, j, r0, r1);
        } else {
            if (rows > 0) trsmTile(tile(i, j) + r0 * nb_, nb_, rows, tile(j, j), nb_, nbj);
            copyOut(i, j, r0, r1);
        }

        // The last part to finish publishes completion and releases dependents.
        if (parts > 1 && pending_[id].fetch_sub(1, std::memory_order_acq_rel) != 1) return;

        step_[id].store((int)(s + 1), std::memory_order_seq_cst);
        if (s < j) {
            tryPush(i, j);
        } else {
            done_[id].store(1, std::memory_order_seq_cst);
            if (i == j) {
                for (size_t r = j + 1; r < nt_; ++r) tryPush(r, j);  // TRSMs below
            } else {
                for (size_t c = j + 1; c <= i; ++c) tryPush(i, c);   // row i
                for (size_t r = i + 1; r < nt_; ++r) tryPush(r, i);  // column i
            }
        }
        completed_.fetch_add(1, std::memory_order_acq_rel);
    }

    // TRSM and the GEMMs feeding the next tile column are on the critical path.
    static bool isCritical(const size_t j, const size_t s) { return s == j || j == s + 1; }

    // Enqueue the next step of tile (i,j) if all its inputs are available.
    void tryPush(const size_t i, const size_t j) {
        const int id = tileId(i, j);
        const int s = step_[id].load(std::memory_order_seq_cst);
        if ((size_t)s > j) return;  // tile is finished
        bool ready;
        if ((size_t)s < j) {
            ready = done_[tileId(i, s)].load(std::memory_order_seq_cst) &&
                    done_[tileId(j, s)].load(std::memory_order_seq_cst);
        } else {
            ready = (i == j) || done_[tileId(j, j)].load(std::memory_order_seq_cst);
        }
        if (!ready) return;
        int expected = s;
        if (!claimed_[id].compare_exchange_strong(expected, s + 1, std::memory_order_acq_rel)) return;

        const bool critical = isCritical(j, (size_t)s);
        if (critical && i != j) {
            const int parts = numParts(tileSize(i), true);
            pending_[id].store(parts, std::memory_order_seq_cst);
            for (int p = 0; p < parts; ++p) critical_.push(encode(id, p));
        } else if (critical) {
            pending_[id].store(0, std::memory_order_seq_cst);
            critical_.push(encode(id, 0));  // POTRF
        } else {
            pending_[id].store(0, std::memory_order_seq_cst);
            local_[owner(i, j)].push(encode(id, 0));
        }
    }

    double* a_;
    double* tiles_;
    const size_t n_, nb_, nt_;
    const int numThreads_;
    int gridP_ = 1, gridQ_ = 1;
    std::vector<std::atomic<int>> step_;     // completed steps per tile
    std::vector<std::atomic<int>> claimed_;  // enqueued steps per tile
    std::vector<std::atomic<char>> done_;    // POTRF/TRSM finished per tile
    std::vector<std::atomic<int>> pending_;  // unfinished parts of the current step
    ReadyQueue critical_;
    std::vector<ReadyQueue> local_;
    const size_t total_;
    alignas(64) std::atomic<size_t> completed_{0};
    alignas(64) std::atomic<size_t> failed_{SIZE_MAX};
};

}  // namespace

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    if (n == 0) return true;

    const size_t nb = chooseTileSize(n);
    const size_t nt = (n + nb - 1) / nb;

    // Tile-contiguous storage of the lower triangle of tiles.
    double* tiles = Workspace::get(TileScheduler::tileIndex(nt, 0) * nb * nb);

    const int numThreads = choleskyThreads();
    TileScheduler scheduler(A.data(), tiles, n, nb, numThreads);

#pragma omp parallel num_threads(numThreads) default(none) shared(scheduler)
    {
        scheduler.run(omp_get_thread_num());
        // All threads leave run() only once every operation has completed
        // (or a failure was detected), so A holds the final factor here.
    }

    const size_t bad = scheduler.failedAt();
    if (bad != SIZE_MAX) {
        // Matrix is not positive definite
        printf("Error: Matrix is not positive definite at diagonal element %zu\n", bad);
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
    
    // Generate random matrix B (sequential: the PRNG stream must be preserved)
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }
    
    // Compute A = B * B^T (symmetric: compute the lower triangle and mirror it).
    // Rows are distributed across threads, which also distributes first-touch
    // of A across NUMA nodes. Eight independent dot products are evaluated in
    // lockstep so that the FMA latency is hidden while every element keeps the
    // exact sequential summation order of the reference implementation.
    const double* __restrict b = B.data();
    double* __restrict a = A.data();
#pragma omp parallel for schedule(dynamic, 4) default(none) shared(a, b, n)
    for (size_t i = 0; i < n; ++i) {
        const double* __restrict bi = b + i * n;
        size_t j = 0;
        for (; j + 8 <= i + 1; j += 8) {
            const double* __restrict b0 = b + (j + 0) * n;
            const double* __restrict b1 = b + (j + 1) * n;
            const double* __restrict b2 = b + (j + 2) * n;
            const double* __restrict b3 = b + (j + 3) * n;
            const double* __restrict b4 = b + (j + 4) * n;
            const double* __restrict b5 = b + (j + 5) * n;
            const double* __restrict b6 = b + (j + 6) * n;
            const double* __restrict b7 = b + (j + 7) * n;
            double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0, s4 = 0.0, s5 = 0.0, s6 = 0.0, s7 = 0.0;
            for (size_t k = 0; k < n; ++k) {
                const double v = bi[k];
                s0 += v * b0[k]; s1 += v * b1[k]; s2 += v * b2[k]; s3 += v * b3[k];
                s4 += v * b4[k]; s5 += v * b5[k]; s6 += v * b6[k]; s7 += v * b7[k];
            }
            const double s[8] = {s0, s1, s2, s3, s4, s5, s6, s7};
            for (size_t q = 0; q < 8; ++q) {
                a[i * n + j + q] = s[q];
                a[(j + q) * n + i] = s[q];
            }
        }
        for (; j <= i; ++j) {
            const double* __restrict bj = b + j * n;
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += bi[k] * bj[k];
            }
            a[i * n + j] = sum;
            a[j * n + i] = sum;
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
    
    // Compute L * L^T (same lockstep scheme as the generator: identical
    // per-element summation order to the reference implementation).
    const double* __restrict l = L.data();
    double* __restrict rec = reconstructed.data();
#pragma omp parallel for schedule(dynamic, 4) default(none) shared(l, rec, n)
    for (size_t i = 0; i < n; ++i) {
        const double* __restrict li = l + i * n;
        size_t j = 0;
        for (; j + 8 <= n; j += 8) {
            const double* __restrict l0 = l + (j + 0) * n;
            const double* __restrict l1 = l + (j + 1) * n;
            const double* __restrict l2 = l + (j + 2) * n;
            const double* __restrict l3 = l + (j + 3) * n;
            const double* __restrict l4 = l + (j + 4) * n;
            const double* __restrict l5 = l + (j + 5) * n;
            const double* __restrict l6 = l + (j + 6) * n;
            const double* __restrict l7 = l + (j + 7) * n;
            double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0, s4 = 0.0, s5 = 0.0, s6 = 0.0, s7 = 0.0;
            for (size_t k = 0; k < n; ++k) {
                const double v = li[k];
                s0 += v * l0[k]; s1 += v * l1[k]; s2 += v * l2[k]; s3 += v * l3[k];
                s4 += v * l4[k]; s5 += v * l5[k]; s6 += v * l6[k]; s7 += v * l7[k];
            }
            rec[i * n + j + 0] = s0; rec[i * n + j + 1] = s1;
            rec[i * n + j + 2] = s2; rec[i * n + j + 3] = s3;
            rec[i * n + j + 4] = s4; rec[i * n + j + 5] = s5;
            rec[i * n + j + 6] = s6; rec[i * n + j + 7] = s7;
        }
        for (; j < n; ++j) {
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
    
    const double* __restrict orig = A_orig.data();
#pragma omp parallel for schedule(static) default(none) shared(rec, orig, n) \
    reduction(max : maxError, relError)
    for (size_t i = 0; i < n * n; ++i) {
        const double error = fabs(rec[i] - orig[i]);
        maxError = std::max(maxError, error);
        
        const double rel = error / (fabs(orig[i]) + 1e-10);
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
    
    // Use one thread per physical core for the whole program (unless
    // OMP_NUM_THREADS is set explicitly), so that the OS spreads the threads
    // over the cores instead of pairing SMT siblings.
    omp_set_num_threads(choleskyThreads());

    printf("Cholesky Decomposition Benchmark\n");
    printf("Matrix size: %zu x %zu\n", n, n);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    printf("OpenMP threads: %d\n", choleskyThreads());
    
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
