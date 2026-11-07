#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <omp.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <unistd.h>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

#include "../common/results_output.hpp"

// OpenMP-parallel Cholesky decomposition (blocked left-looking algorithm)
// Decomposes positive definite matrix A into L * L^T where L is lower triangular.
//
// Every element L(i,j) is computed with exactly the same arithmetic as the
// original unblocked code:  (A(i,j) - sum_{k<j} L(i,k)*L(j,k)) / L(j,j)
// with the sum accumulated sequentially in increasing k. The compiler turns the
// original reduction into: separately rounded multiply + add for the first
// (j & ~3) terms, then fused multiply-add for the remaining 0..3 tail terms.
// The kernels below reproduce exactly that, so results match the sequential
// build to the last bit (verified by hash comparison).

#if defined(__AVX2__) && defined(__FMA__)
#define CHOL_HAVE_AVX2_FMA 1
#else
#define CHOL_HAVE_AVX2_FMA 0
#endif

namespace {

inline void cpuRelax() {
#if defined(__x86_64__) || defined(__i386__)
    _mm_pause();
#endif
}

// Separately rounded multiply followed by add. The empty asm statement stops the compiler
// from contracting the pair into a fused multiply-add (GCC does so by default even for
// intrinsics), which would change the rounding relative to the reference build.
#if CHOL_HAVE_AVX2_FMA
inline __m256d mulAdd(const __m256d acc, const __m256d a, const __m256d b) {
    __m256d prod = _mm256_mul_pd(a, b);
    __asm__("" : "+x"(prod));
    return _mm256_add_pd(acc, prod);
}
#endif
inline double mulAddScalar(const double acc, const double a, const double b) {
    double prod = a * b;
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
    __asm__("" : "+x"(prod));
#endif
    return acc + prod;
}

constexpr size_t NB = 64;   // panel width (columns per panel)
constexpr size_t RB = 6;    // rows processed together by the kernel (RB*CB/4 + CB/4 + 1 <= 16 ymm registers)
constexpr size_t CB = 8;    // columns processed together by the kernel (NB % CB == 0)
constexpr size_t KB = 32;   // k-block: KB x NB panel slice (16 KB) stays in L1 across a thread's rows

// Accumulate S[r][jj] += sum_{k in [k0,k1)} A(i0+r, k) * P[k][jj]  for RB rows and nb columns.
// P is the transposed panel (row-major, row stride nb, nb a multiple of CB); S has row stride nb.
// If `first` is set the accumulators start from zero instead of the stored partial sums.
// Within a (row, column) element the k order is strictly increasing, and the partial sums are
// only spilled to memory between k blocks, so the arithmetic is identical to a plain
// sequential (multiply, then add) dot product over k = 0 .. jb-1.
inline void partialSums(const double* __restrict__ A, const size_t n, const size_t i0,
                        const size_t k0, const size_t k1, const size_t nb,
                        const double* __restrict__ P, double* __restrict__ S, const bool first) {
    // Rows past the end of the matrix are clamped to the last row; their (duplicate)
    // results land in the padding area of S and are never read.
    const double* rows[RB];
    for (size_t r = 0; r < RB; ++r) rows[r] = A + std::min(i0 + r, n - 1) * n;

    for (size_t jc = 0; jc < nb; jc += CB) {
#if CHOL_HAVE_AVX2_FMA
        __m256d acc[RB][2];
        if (first) {
            for (size_t r = 0; r < RB; ++r) {
                acc[r][0] = _mm256_setzero_pd();
                acc[r][1] = _mm256_setzero_pd();
            }
        } else {
            for (size_t r = 0; r < RB; ++r) {
                acc[r][0] = _mm256_loadu_pd(S + r * nb + jc);
                acc[r][1] = _mm256_loadu_pd(S + r * nb + jc + 4);
            }
        }
        const double* p = P + k0 * nb + jc;
        for (size_t k = k0; k < k1; ++k, p += nb) {
            const __m256d p0 = _mm256_loadu_pd(p);
            const __m256d p1 = _mm256_loadu_pd(p + 4);
            for (size_t r = 0; r < RB; ++r) {
                const __m256d a = _mm256_broadcast_sd(rows[r] + k);
                acc[r][0] = mulAdd(acc[r][0], a, p0);
                acc[r][1] = mulAdd(acc[r][1], a, p1);
            }
        }
        for (size_t r = 0; r < RB; ++r) {
            _mm256_storeu_pd(S + r * nb + jc, acc[r][0]);
            _mm256_storeu_pd(S + r * nb + jc + 4, acc[r][1]);
        }
#else
        double s[RB][CB];
        for (size_t r = 0; r < RB; ++r)
            for (size_t c = 0; c < CB; ++c) s[r][c] = first ? 0.0 : S[r * nb + jc + c];
        const double* p = P + k0 * nb + jc;
        for (size_t k = k0; k < k1; ++k, p += nb) {
            for (size_t r = 0; r < RB; ++r) {
                const double a = rows[r][k];
                for (size_t c = 0; c < CB; ++c) s[r][c] = mulAddScalar(s[r][c], a, p[c]);
            }
        }
        for (size_t r = 0; r < RB; ++r)
            for (size_t c = 0; c < CB; ++c) S[r * nb + jc + c] = s[r][c];
#endif
    }
}

// s[jj] += a * d[jj] for jj in [0, nbp), nbp a multiple of CB, where `a` is term number kk of
// the dot product that ends at column jj. Columns with (jj & ~3) > kk take the term as a
// separately rounded multiply + add, the last (jj & 3) terms as a fused multiply-add, which
// is exactly what the compiler emits for the original reduction loop. Lanes jj <= kk are
// don't-care (their sums were already consumed).
inline void axpyRow(double* __restrict__ s, const double a, const double* __restrict__ d,
                    const size_t nbp, const size_t kk) {
#if CHOL_HAVE_AVX2_FMA
    const __m256d av = _mm256_set1_pd(a);
    const size_t fmaEnd = (kk & ~static_cast<size_t>(3)) + 4;  // vectors with 4q <= kk use FMA
    size_t jj = 0;
    for (; jj < fmaEnd; jj += 4)
        _mm256_storeu_pd(s + jj, _mm256_fmadd_pd(av, _mm256_loadu_pd(d + jj), _mm256_loadu_pd(s + jj)));
    for (; jj < nbp; jj += 4)
        _mm256_storeu_pd(s + jj, mulAdd(_mm256_loadu_pd(s + jj), av, _mm256_loadu_pd(d + jj)));
#else
    const size_t fmaEnd = (kk & ~static_cast<size_t>(3)) + 4;
    for (size_t jj = 0; jj < fmaEnd; ++jj) s[jj] = std::fma(a, d[jj], s[jj]);
    for (size_t jj = fmaEnd; jj < nbp; ++jj) s[jj] = mulAddScalar(s[jj], a, d[jj]);
#endif
}

// Same as axpyRow but only for lanes jj <= jmax: whole 4-lane vectors below (jmax & ~3),
// then scalars up to jmax. Used inside the diagonal block, where lanes beyond the current
// row belong to rows that other threads may still be writing.
inline void axpyRowUpTo(double* __restrict__ s, const double a, const double* __restrict__ d,
                        const size_t jmax, const size_t kk) {
    const size_t vecEnd = jmax & ~static_cast<size_t>(3);
    const size_t fmaEnd = std::min(vecEnd, (kk & ~static_cast<size_t>(3)) + 4);
#if CHOL_HAVE_AVX2_FMA
    const __m256d av = _mm256_set1_pd(a);
    size_t jj = 0;
    for (; jj < fmaEnd; jj += 4)
        _mm256_storeu_pd(s + jj, _mm256_fmadd_pd(av, _mm256_loadu_pd(d + jj), _mm256_loadu_pd(s + jj)));
    for (; jj < vecEnd; jj += 4)
        _mm256_storeu_pd(s + jj, mulAdd(_mm256_loadu_pd(s + jj), av, _mm256_loadu_pd(d + jj)));
#else
    size_t jj = 0;
    for (; jj < fmaEnd; ++jj) s[jj] = std::fma(a, d[jj], s[jj]);
    for (; jj < vecEnd; ++jj) s[jj] = mulAddScalar(s[jj], a, d[jj]);
#endif
    for (size_t j = vecEnd; j <= jmax; ++j) {
        if ((j & ~static_cast<size_t>(3)) <= kk) s[j] = std::fma(a, d[j], s[j]);
        else s[j] = mulAddScalar(s[j], a, d[j]);
    }
}

// Finish row i (below the diagonal block) of the panel [jb, jb+nb): for each panel column j
// the full sum is   S_i[j-jb] + sum_{k=jb}^{j-1} L(i,k) L(j,k)   (k increasing), which is
// accumulated for all columns at once: as soon as L(i,k) is final it is folded into the
// running sums of every later column via the transposed diagonal block D,
// D[kk][jj] = L(jb+jj, jb+kk) for jj > kk (lanes jj <= kk are zero / already consumed).
inline void finishRow(double* __restrict__ Ai, const double* __restrict__ Si,
                      const double* __restrict__ D, const double* __restrict__ diag,
                      const size_t jb, const size_t nb, const size_t nbp) {
    alignas(32) double s[NB + CB];
    for (size_t jj = 0; jj < nbp; ++jj) s[jj] = Si[jj];
    for (size_t kk = 0; kk < nb; ++kk) {
        const double l = (Ai[jb + kk] - s[kk]) / diag[kk];
        Ai[jb + kk] = l;
        axpyRow(s, l, D + kk * nbp, nbp, kk);
    }
}

// Row li of the diagonal block, executed concurrently with the other rows of the block
// as a wavefront: step kk of row li may run once row li-1 has completed step kk
// (prog[li-1] > kk), which by induction means every row < li has, so diag[kk] and
// D[kk][jj] for jj < li are final. Progress is published after each step.
struct alignas(64) RowProgress {
    std::atomic<int> v{-1};
};

inline bool diagRow(double* __restrict__ Ai, const double* __restrict__ Si, double* __restrict__ D,
                    double* __restrict__ diag, RowProgress* prog, std::atomic<bool>& ok,
                    const size_t jb, const size_t li, const size_t nbp) {
    alignas(32) double s[NB + CB];
    for (size_t jj = 0; jj < nbp; ++jj) s[jj] = Si[jj];
    for (size_t kk = 0; kk < li; ++kk) {
        while (prog[li - 1].v.load(std::memory_order_acquire) <= static_cast<int>(kk)) cpuRelax();
        const double l = (Ai[jb + kk] - s[kk]) / diag[kk];
        Ai[jb + kk] = l;
        D[kk * nbp + li] = l;
        axpyRowUpTo(s, l, D + kk * nbp, li, kk);
        prog[li].v.store(static_cast<int>(kk) + 1, std::memory_order_release);
    }
    const double val = Ai[jb + li] - s[li];
    bool good = true;
    if (val <= 0.0) {
        if (ok.exchange(false)) {
            printf("Error: Matrix is not positive definite at diagonal element %zu\n", jb + li);
        }
        good = false;
    } else {
        Ai[jb + li] = sqrt(val);
        diag[li] = Ai[jb + li];
    }
    prog[li].v.store(static_cast<int>(NB) + 1, std::memory_order_release);  // row done
    return good;
}

// Lightweight sense-reversing spin barrier. The factorization performs a few barriers per
// panel; the runtime's default barrier may park threads in the kernel, and waking a large
// team costs far more than the work between barriers. Spinning keeps latency minimal.
class SpinBarrier {
public:
    explicit SpinBarrier(int nthreads) : nthreads_(nthreads) {}

    void wait() {
        const unsigned gen = generation_.load(std::memory_order_acquire);
        if (arrived_.fetch_add(1, std::memory_order_acq_rel) + 1 == nthreads_) {
            arrived_.store(0, std::memory_order_relaxed);
            generation_.store(gen + 1, std::memory_order_release);
        } else {
            while (generation_.load(std::memory_order_acquire) == gen) cpuRelax();
        }
    }

private:
    const int nthreads_;
    alignas(64) std::atomic<int> arrived_{0};
    alignas(64) std::atomic<unsigned> generation_{0};
};

// Call fn(x) for every integer x of a cpuset-style list such as "0-3,8,10-11".
template <typename F>
void forEachInList(const char* str, F fn) {
    const char* p = str;
    while (*p) {
        char* end;
        const long a = strtol(p, &end, 10);
        if (end == p) break;
        long b = a;
        p = end;
        if (*p == '-') { b = strtol(p + 1, &end, 10); p = end; }
        for (long x = a; x <= b; ++x) fn(x);
        if (*p == ',') ++p; else break;
    }
}

// Sibling list of the physical core hosting `cpu` (from sysfs); just {cpu} if unavailable.
std::vector<int> siblingsOf(const int cpu) {
    std::vector<int> sib;
    char path[128];
    snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", cpu);
    if (FILE* f = fopen(path, "r")) {
        char line[1024];
        if (fgets(line, sizeof(line), f)) forEachInList(line, [&](long x) { sib.push_back(static_cast<int>(x)); });
        fclose(f);
    }
    if (sib.empty()) sib.push_back(cpu);
    return sib;
}

// Pin OpenMP threads to physical cores of the process' affinity mask ("close" order, one
// place per core so a thread may use either SMT sibling, like OMP_PLACES=cores).
// Only done when the user did not request a binding policy via OMP_PROC_BIND/OMP_PLACES.
// Unpinned teams spanning every hardware thread of a large NUMA box migrate between
// cores and stall at barriers, which costs an order of magnitude on small problems.
// When the thread count was not given either, one thread per physical core is used:
// SMT siblings share the FMA units and only add synchronization cost for this kernel.
void configureThreads() {
    if (omp_get_proc_bind() != omp_proc_bind_false) return;

    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) return;

    // Build the list of places: one per physical core, restricted to allowed CPUs.
    std::vector<cpu_set_t> places;
    cpu_set_t seen;
    CPU_ZERO(&seen);
    for (int c = 0; c < CPU_SETSIZE; ++c) {
        if (!CPU_ISSET(c, &allowed) || CPU_ISSET(c, &seen)) continue;
        cpu_set_t place;
        CPU_ZERO(&place);
        for (int s : siblingsOf(c)) {
            if (s >= 0 && s < CPU_SETSIZE && CPU_ISSET(s, &allowed)) {
                CPU_SET(s, &place);
                CPU_SET(s, &seen);
            }
        }
        CPU_SET(c, &place);
        CPU_SET(c, &seen);
        places.push_back(place);
    }
    if (places.empty()) return;

    if (getenv("OMP_NUM_THREADS") == nullptr) {
        omp_set_num_threads(static_cast<int>(places.size()));
    }

#pragma omp parallel
    {
        const cpu_set_t& mine = places[static_cast<size_t>(omp_get_thread_num()) % places.size()];
        sched_setaffinity(0, sizeof(mine), &mine);  // affects the calling thread only
    }
}

// NUMA nodes this process may allocate on (from /proc/self/status), as a bitmask.
unsigned long allowedNumaNodes() {
    static const unsigned long mask = [] {
        unsigned long m = 0;
        if (FILE* f = fopen("/proc/self/status", "r")) {
            char line[512];
            while (fgets(line, sizeof(line), f)) {
                if (strncmp(line, "Mems_allowed_list:", 18) == 0) {
                    forEachInList(line + 18, [&](long x) { if (x >= 0 && x < 64) m |= 1UL << x; });
                    break;
                }
            }
            fclose(f);
        }
        return m;
    }();
    return mask;
}

// Allocate `count` doubles in `v` with the storage advised for transparent huge pages
// (fewer TLB misses and far fewer NUMA-balancing hinting faults on big NUMA machines) and,
// on multi-node machines, interleaved across the allowed NUMA nodes so that every thread
// sees the same memory bandwidth (the matrix is touched by all threads in every panel).
// reserve() obtains untouched memory, so the policies apply before the first touch.
void allocateHugeVector(std::vector<double>& v, const size_t count) {
    // Re-enable THP for this process in case a parent disabled it (inherited flag).
    static const bool thpEnabled = (prctl(PR_SET_THP_DISABLE, 0, 0, 0, 0), true);
    (void)thpEnabled;
    v.clear();
    v.shrink_to_fit();
    v.reserve(count);
    const long page = sysconf(_SC_PAGESIZE);
    if (page > 0 && count > 0) {
        const uintptr_t begin = reinterpret_cast<uintptr_t>(v.data());
        const uintptr_t end = begin + count * sizeof(double);
        const uintptr_t pb = (begin + page - 1) & ~static_cast<uintptr_t>(page - 1);
        const uintptr_t pe = end & ~static_cast<uintptr_t>(page - 1);
        if (pe > pb) {
            madvise(reinterpret_cast<void*>(pb), pe - pb, MADV_HUGEPAGE);
            const unsigned long nodes = allowedNumaNodes();
            if (nodes != 0 && (nodes & (nodes - 1)) != 0) {  // more than one node
                constexpr int MPOL_INTERLEAVE_ = 3;
                syscall(SYS_mbind, reinterpret_cast<void*>(pb), pe - pb, MPOL_INTERLEAVE_, &nodes,
                        static_cast<unsigned long>(64), 0UL);
            }
        }
    }
    v.resize(count);
}

// Pointer into v aligned to a 64-byte cache line (v has >= 8 doubles of slack).
inline double* alignedData(std::vector<double>& v) {
    const uintptr_t p = reinterpret_cast<uintptr_t>(v.data());
    return reinterpret_cast<double*>((p + 63) & ~static_cast<uintptr_t>(63));
}

// Anonymous mmap-backed scratch buffer: zero-filled, cache-line aligned, huge-page advised,
// interleaved across NUMA nodes and pre-faulted with a single syscall (page faults taken one
// by one from user space cost several microseconds each on large NUMA machines).
class ScratchBuffer {
public:
    explicit ScratchBuffer(const size_t count) {
        bytes_ = (count * sizeof(double) + 4095) & ~static_cast<size_t>(4095);
        void* p = mmap(nullptr, bytes_, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED) {
            fallback_.assign(count + 8, 0.0);
            data_ = alignedData(fallback_);
            return;
        }
        mem_ = p;
        madvise(p, bytes_, MADV_HUGEPAGE);
        const unsigned long nodes = allowedNumaNodes();
        if (nodes != 0 && (nodes & (nodes - 1)) != 0) {
            constexpr int MPOL_INTERLEAVE_ = 3;
            syscall(SYS_mbind, p, bytes_, MPOL_INTERLEAVE_, &nodes, static_cast<unsigned long>(64), 0UL);
        }
#ifdef MADV_POPULATE_WRITE
        if (madvise(p, bytes_, MADV_POPULATE_WRITE) != 0)
#endif
        {
            // Fallback: touch one byte per page.
            volatile char* c = static_cast<volatile char*>(p);
            for (size_t off = 0; off < bytes_; off += 4096) c[off] = 0;
        }
        data_ = static_cast<double*>(p);
    }
    ~ScratchBuffer() {
        if (mem_) munmap(mem_, bytes_);
    }
    ScratchBuffer(const ScratchBuffer&) = delete;
    ScratchBuffer& operator=(const ScratchBuffer&) = delete;
    double* data() const { return data_; }

private:
    void* mem_ = nullptr;
    size_t bytes_ = 0;
    double* data_ = nullptr;
    std::vector<double> fallback_;
};

// Prefetch the k-slice [k0,k1) of the RB rows starting at row i0 (used for the next row group).
inline void prefetchRows(const double* A, const size_t n, const size_t i0, const size_t k0,
                         const size_t k1) {
    for (size_t r = 0; r < RB; ++r) {
        const double* row = A + std::min(i0 + r, n - 1) * n;
        for (size_t k = k0; k < k1; k += 8) __builtin_prefetch(row + k, 0, 3);
    }
}

// Threads to use for a problem of size n (avoid oversubscribing tiny matrices).
int threadsFor(const size_t n) {
    const size_t maxT = static_cast<size_t>(omp_get_max_threads());
    const size_t want = std::max<size_t>(1, n / RB);
    return static_cast<int>(std::min(maxT, want));
}

}  // namespace

bool choleskyDecomposition(std::vector<double>& Avec, const size_t n) {
    double* __restrict__ A = Avec.data();
    if (n == 0) return true;
    std::atomic<bool> ok{true};

    // Work buffers
    ScratchBuffer Pbuf0(n * NB), Pbuf1(n * NB);   // transposed panel: P[k][jj] = L(jb+jj, k), k < jb
                                                  // (double buffered: next panel is built during this one)
    ScratchBuffer Sbuf((n + 2 * RB) * NB);        // partial sums S[i][jj] = sum_{k<jb} L(i,k) L(jb+jj,k) (+ padding)
    std::vector<double> Dbuf(NB * NB + 8, 0.0);   // transposed diagonal block D[kk][jj] = L(jb+jj, jb+kk), jj > kk
    std::vector<double> diagBuf(NB, 0.0);         // diag[kk] = L(jb+kk, jb+kk)
    double* __restrict__ Pbufs[2] = {Pbuf0.data(), Pbuf1.data()};
    double* __restrict__ S = Sbuf.data();
    double* __restrict__ D = alignedData(Dbuf);   // lanes jj <= kk are never written: stay zero
    double* __restrict__ diag = diagBuf.data();
    std::vector<RowProgress> prog(NB);            // wavefront progress of the diagonal block rows

    const int nthreads = threadsFor(n);
    SpinBarrier barrier(nthreads);
    const size_t diagThreads = std::min<size_t>(static_cast<size_t>(nthreads), 16);

#pragma omp parallel num_threads(nthreads)
    {
        const size_t T = static_cast<size_t>(nthreads);
        const size_t t = static_cast<size_t>(omp_get_thread_num());

        // Rows are owned by threads in a block-cyclic fashion: row group G = i / RB belongs to
        // thread G % T. The mapping is fixed for the whole factorization, so a thread keeps
        // touching the same rows (cache locality) and the remaining work stays balanced.
        auto firstGroup = [&](const size_t G0) { return G0 + ((t + T - G0 % T) % T); };

        for (size_t jb = 0; jb < n; jb += NB) {
            const size_t nb = std::min(NB, n - jb);  // columns in this panel
            const size_t je = jb + nb;
            const size_t nbp = ((nb + CB - 1) / CB) * CB;  // padded width (stride of P, S and D)
            const size_t G0 = jb / RB;                     // first row group touching this panel
            const size_t sBase = G0 * RB;                  // S row index of matrix row i is i - sBase
            const size_t nextNb = (je < n) ? std::min(NB, n - je) : 0;
            const size_t nextNbp = ((nextNb + CB - 1) / CB) * CB;
            const double* __restrict__ P = Pbufs[(jb / NB) & 1];
            double* __restrict__ Pnext = Pbufs[((jb / NB) + 1) & 1];

            // 1) Partial sums for every row i >= jb over the already-finished columns k < jb
            //    (P[k][jj] = L(jb+jj, k) was transposed during the previous panel).
            //    k is blocked so that the panel slice P[k0..k1) stays in L1 while a thread
            //    sweeps all of its row groups. Rows of the next panel are transposed into
            //    Pnext on the fly while their k-slice is in L1.
            if (jb == 0) {
                for (size_t G = firstGroup(G0); G * RB < n; G += T)
                    for (size_t jj = 0; jj < RB * nbp; ++jj) S[(G * RB - sBase) * nbp + jj] = 0.0;
            } else {
                for (size_t k0 = 0; k0 < jb; k0 += KB) {
                    const size_t k1 = std::min(jb, k0 + KB);
                    for (size_t G = firstGroup(G0); G * RB < n; G += T) {
                        const size_t i0 = G * RB;
                        if ((G + T) * RB < n) prefetchRows(A, n, (G + T) * RB, k0, k1);
                        partialSums(A, n, i0, k0, k1, nbp, P, S + (i0 - sBase) * nbp, k0 == 0);
                        const size_t pi0 = std::max(i0, je), pi1 = std::min(i0 + RB, je + nextNb);
                        if (pi0 < pi1) {
                            for (size_t k = k0; k < k1; ++k) {
                                double* dst = Pnext + k * nextNbp;
                                for (size_t i = pi0; i < pi1; ++i) dst[i - je] = A[i * n + k];
                            }
                        }
                    }
                }
            }
            barrier.wait();

            // 2) Factor the diagonal block (rows/cols jb..je-1) as a wavefront over a few threads.
            if (t < diagThreads) {
                for (size_t li = t; li < nb; li += diagThreads)
                    diagRow(A + (jb + li) * n, S + (jb + li - sBase) * nbp, D, diag, prog.data(), ok, jb, li, nbp);
            }
            barrier.wait();

            if (!ok.load(std::memory_order_relaxed)) break;

            // 3) Finish the panel columns for all rows below the diagonal block.
            //    Each row only depends on the (finished) diagonal block and its own entries.
            //    Rows of the next panel also store their finished columns into Pnext.
            for (size_t G = firstGroup(je / RB); G * RB < n; G += T) {  // first group may straddle je
                const size_t iEnd = std::min(n, (G + 1) * RB);
                for (size_t i = std::max(je, G * RB); i < iEnd; ++i) {
                    double* __restrict__ Ai = A + i * n;
                    finishRow(Ai, S + (i - sBase) * nbp, D, diag, jb, nb, nbp);
                    if (i < je + nextNb)
                        for (size_t k = jb; k < je; ++k) Pnext[k * nextNbp + (i - je)] = Ai[k];
                }
            }
            if (t == 0)
                for (size_t li = 0; li < NB; ++li) prog[li].v.store(-1, std::memory_order_relaxed);
            barrier.wait();
        }

        // Zero out upper triangular part
        if (ok.load(std::memory_order_relaxed)) {
#pragma omp for schedule(static)
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = i + 1; j < n; ++j) A[i * n + j] = 0.0;
            }
        }
    }

    return ok.load();
}

// Generate a symmetric positive definite matrix
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    // Method: Create A = B * B^T where B is random
    // This guarantees A is positive semi-definite
    // Then add identity to make it strictly positive definite
    
    std::vector<double> B;
    allocateHugeVector(B, n * n);
    unsigned int seed = 42;
    
    // Generate random matrix B (sequential: the RNG stream must stay identical)
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }
    
    // Compute A = B * B^T (rows are independent; per-element summation order preserved)
    const double* __restrict__ Bp = B.data();
    double* __restrict__ Ap = A.data();
#pragma omp parallel for schedule(static) num_threads(threadsFor(n))
    for (size_t i = 0; i < n; ++i) {
        const double* __restrict__ Bi = Bp + i * n;
        for (size_t j = 0; j < n; ++j) {
            const double* __restrict__ Bj = Bp + j * n;
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += Bi[k] * Bj[k];
            }
            Ap[i * n + j] = sum;
        }
    }
    
    // Add diagonal dominance to ensure positive definiteness
    for (size_t i = 0; i < n; ++i) {
        A[i * n + i] += n;
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix
    
    std::vector<double> reconstructed;
    allocateHugeVector(reconstructed, n * n);
    const double* __restrict__ Lp = L.data();
    double* __restrict__ Rp = reconstructed.data();
    
    // Compute L * L^T
#pragma omp parallel for schedule(static) num_threads(threadsFor(n))
    for (size_t i = 0; i < n; ++i) {
        const double* __restrict__ Li = Lp + i * n;
        for (size_t j = 0; j < n; ++j) {
            const double* __restrict__ Lj = Lp + j * n;
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += Li[k] * Lj[k];
            }
            Rp[i * n + j] = sum;
        }
    }
    
    // Compare with original
    double maxError = 0.0;
    double relError = 0.0;
    
#pragma omp parallel for schedule(static) reduction(max : maxError, relError)
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
    configureThreads();
    printf("OpenMP threads: %d\n", threadsFor(n));
    
    // Allocate matrix
    std::vector<double> A;
    allocateHugeVector(A, n * n);
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
