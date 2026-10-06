#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <memory>
#include <new>
#include <vector>

#include <omp.h>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>

#include "../common/results_output.hpp"

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

void initMatrix(std::vector<double>& mat, const size_t N) {
    double* const m = mat.data();
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            m[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

// ---------------------------------------------------------------------------
// Blocked, packed, register-tiled matrix multiplication parallelized with OpenMP.
// Every C[i][j] is accumulated in increasing k order starting from 0.0, exactly
// like the reference triple loop, so the result is numerically equivalent.
// ---------------------------------------------------------------------------
namespace {

constexpr size_t MR = 6;   // micro-tile rows
constexpr size_t NR = 8;   // micro-tile cols (2 x 4 doubles)
constexpr size_t KC = 256; // k-block depth
constexpr size_t MC = 96;  // rows of a packed A block (multiple of MR)
constexpr size_t NC = 512; // cols of a packed B block (multiple of NR)

typedef double v4d __attribute__((vector_size(32), aligned(8)));

inline size_t roundUp(size_t x, size_t m) { return (x + m - 1) / m * m; }

// Pack an mc x kc block of A (row-major, leading dim N) into MR-row panels:
// for each panel, for each k, MR consecutive values (zero padded).
inline void packA(const double* __restrict A, double* __restrict Ap, size_t N,
                  size_t i0, size_t mc, size_t k0, size_t kc) {
    for (size_t ip = 0; ip < mc; ip += MR) {
        const size_t mr = std::min(MR, mc - ip);
        const double* src = A + (i0 + ip) * N + k0;
        for (size_t k = 0; k < kc; ++k) {
            size_t r = 0;
            for (; r < mr; ++r) Ap[r] = src[r * N + k];
            for (; r < MR; ++r) Ap[r] = 0.0;
            Ap += MR;
        }
    }
}

// Pack a kc x nc block of B into NR-column panels: for each panel, for each k,
// NR consecutive values (zero padded).
inline void packB(const double* __restrict B, double* __restrict Bp, size_t N,
                  size_t k0, size_t kc, size_t j0, size_t nc) {
    for (size_t jp = 0; jp < nc; jp += NR) {
        const size_t nr = std::min(NR, nc - jp);
        const double* src = B + k0 * N + j0 + jp;
        for (size_t k = 0; k < kc; ++k) {
            size_t q = 0;
            for (; q < nr; ++q) Bp[q] = src[k * N + q];
            for (; q < NR; ++q) Bp[q] = 0.0;
            Bp += NR;
        }
    }
}

// C[MR x NR] (leading dim ldc) = (first ? 0 : C) + Ap * Bp, accumulated in k order.
inline void microKernel(size_t kc, const double* __restrict Ap, const double* __restrict Bp,
                        double* __restrict C, size_t ldc, bool first) {
    v4d acc[MR][2];
    if (first) {
        for (size_t r = 0; r < MR; ++r) {
            acc[r][0] = v4d{0.0, 0.0, 0.0, 0.0};
            acc[r][1] = v4d{0.0, 0.0, 0.0, 0.0};
        }
    } else {
        for (size_t r = 0; r < MR; ++r) {
            std::memcpy(&acc[r][0], C + r * ldc, sizeof(v4d));
            std::memcpy(&acc[r][1], C + r * ldc + 4, sizeof(v4d));
        }
    }
    for (size_t k = 0; k < kc; ++k) {
        v4d b0, b1;
        std::memcpy(&b0, Bp, sizeof(v4d));
        std::memcpy(&b1, Bp + 4, sizeof(v4d));
        for (size_t r = 0; r < MR; ++r) {
            const double a = Ap[r];
            acc[r][0] += a * b0;
            acc[r][1] += a * b1;
        }
        Ap += MR;
        Bp += NR;
    }
    for (size_t r = 0; r < MR; ++r) {
        std::memcpy(C + r * ldc, &acc[r][0], sizeof(v4d));
        std::memcpy(C + r * ldc + 4, &acc[r][1], sizeof(v4d));
    }
}

// Compute rows [i0, i0+m) x cols [j0, j0+n) of C using the packing buffers Ap, Bp.
void multiplyBlock(const double* __restrict A, const double* __restrict B, double* __restrict C,
                   size_t N, size_t i0, size_t m, size_t j0, size_t n,
                   double* __restrict Ap, double* __restrict Bp) {
    double edge[MR * NR];
    for (size_t jc = j0; jc < j0 + n; jc += NC) {
        const size_t nc = std::min(NC, j0 + n - jc);
        for (size_t k0 = 0; k0 < N; k0 += KC) {
            const size_t kc = std::min(KC, N - k0);
            const bool first = (k0 == 0);
            packB(B, Bp, N, k0, kc, jc, nc);
            for (size_t ic = i0; ic < i0 + m; ic += MC) {
                const size_t mc = std::min(MC, i0 + m - ic);
                packA(A, Ap, N, ic, mc, k0, kc);
                for (size_t jp = 0; jp < nc; jp += NR) {
                    const size_t nr = std::min(NR, nc - jp);
                    const double* bpan = Bp + jp * kc;
                    for (size_t ip = 0; ip < mc; ip += MR) {
                        const size_t mr = std::min(MR, mc - ip);
                        const double* apan = Ap + ip * kc;
                        double* ct = C + (ic + ip) * N + jc + jp;
                        if (mr == MR && nr == NR) {
                            microKernel(kc, apan, bpan, ct, N, first);
                        } else {
                            if (!first) {
                                for (size_t r = 0; r < mr; ++r)
                                    for (size_t q = 0; q < nr; ++q)
                                        edge[r * NR + q] = ct[r * N + q];
                            }
                            microKernel(kc, apan, bpan, edge, NR, first);
                            for (size_t r = 0; r < mr; ++r)
                                for (size_t q = 0; q < nr; ++q)
                                    ct[r * N + q] = edge[r * NR + q];
                        }
                    }
                }
            }
        }
    }
}

// Split `units` work units into `parts` near-equal contiguous ranges; return range p.
inline void splitRange(size_t units, size_t parts, size_t p, size_t& begin, size_t& end) {
    const size_t base = units / parts, rem = units % parts;
    begin = p * base + std::min(p, rem);
    end = begin + base + (p < rem ? 1 : 0);
}

// CPUs of the process affinity mask ordered "one hardware thread per physical core
// first", followed by the remaining SMT siblings. Falls back to mask order when the
// Linux sysfs topology is unavailable. Computed once.
struct CpuTopology {
    std::vector<int> order;
    size_t physicalCores = 0;
};

const CpuTopology& cpuTopology() {
    static const CpuTopology topo = [] {
        CpuTopology t;
        cpu_set_t mask;
        CPU_ZERO(&mask);
        if (sched_getaffinity(0, sizeof(mask), &mask) != 0) return t;
        // Walk the allowed CPUs; the first CPU seen of each physical core is taken as
        // its primary, its SMT siblings (from sysfs) are appended at the end.
        std::vector<char> isSibling(CPU_SETSIZE, 0);
        std::vector<int> siblings;
        for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
            if (!CPU_ISSET(cpu, &mask)) continue;
            if (isSibling[cpu]) { siblings.push_back(cpu); continue; }
            t.order.push_back(cpu);
            char path[128];
            std::snprintf(path, sizeof(path),
                          "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", cpu);
            FILE* f = std::fopen(path, "r");
            if (!f) continue;
            char line[1024];
            if (std::fgets(line, sizeof(line), f)) {
                // Format: comma separated list of CPUs or ranges, e.g. "0,128" or "0-1".
                const char* p = line;
                while (*p) {
                    char* end;
                    const long lo = std::strtol(p, &end, 10);
                    if (end == p) break;
                    long hi = lo;
                    p = end;
                    if (*p == '-') {
                        hi = std::strtol(p + 1, &end, 10);
                        p = end;
                    }
                    for (long x = lo; x <= hi && x < CPU_SETSIZE; ++x)
                        if (x > cpu) isSibling[x] = 1;
                    if (*p != ',') break;
                    ++p;
                }
            }
            std::fclose(f);
        }
        t.physicalCores = t.order.size();
        t.order.insert(t.order.end(), siblings.begin(), siblings.end());
        return t;
    }();
    return topo;
}

// Persistent packing workspace shared by all threads (each uses its own slice).
// Kept across calls so that neither allocation nor unmapping happens per call,
// and backed by transparent huge pages where available to cut page-fault cost.
double* workspace(size_t doubles) {
    struct FreeDeleter { void operator()(void* p) const { std::free(p); } };
    static std::unique_ptr<double, FreeDeleter> buf;
    static size_t capacity = 0;
    if (capacity < doubles) {
        constexpr size_t kHuge = size_t(2) << 20;
        const size_t bytes = roundUp(doubles * sizeof(double), kHuge);
        buf.reset(static_cast<double*>(std::aligned_alloc(kHuge, bytes)));
        if (!buf) throw std::bad_alloc();
#ifdef MADV_HUGEPAGE
        madvise(buf.get(), bytes, MADV_HUGEPAGE);
#endif
        capacity = bytes / sizeof(double);
    }
    return buf.get();
}

} // namespace

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B, 
                    std::vector<double>& C, const size_t N) {
    if (N == 0) return;
    const double* const a = A.data();
    const double* const b = B.data();
    double* const c = C.data();

    // Number of micro-tile rows / columns of C.
    const size_t rowUnits = (N + MR - 1) / MR;
    const size_t colUnits = (N + NR - 1) / NR;

    // Use only as many threads as there is meaningful work for (>= ~16 MFLOP each).
    const double flops = 2.0 * static_cast<double>(N) * static_cast<double>(N) * static_cast<double>(N);
    size_t nthreads = static_cast<size_t>(omp_get_max_threads());
    nthreads = std::min(nthreads, std::max<size_t>(1, static_cast<size_t>(flops / 16e6)));

    // Unless the user configured thread placement, pin threads (physical cores first):
    // the static decomposition relies on cache locality and suffers from migrations.
    // If the thread count was not chosen explicitly either, use one thread per
    // physical core, since SMT siblings only compete for the FMA units and caches.
    // (Only for large problems: detecting the topology costs a few milliseconds.)
    const bool pinThreads = nthreads > 1 && flops >= 1e9 && omp_get_proc_bind() == omp_proc_bind_false;
    const CpuTopology* topo = pinThreads ? &cpuTopology() : nullptr;
    if (topo && topo->physicalCores > 0 && std::getenv("OMP_NUM_THREADS") == nullptr)
        nthreads = std::min(nthreads, topo->physicalCores);
    nthreads = std::min(nthreads, rowUnits * colUnits);

    // Arrange the work in a pr x pc grid of rectangles of C, one per thread.
    // Pick the factorization that minimizes redundant reads of A and B, i.e.
    // keeps the rectangles as square as possible.
    auto gridCost = [&](size_t r, size_t q) {
        return static_cast<double>(r) / static_cast<double>(colUnits * NR) +
               static_cast<double>(q) / static_cast<double>(rowUnits * MR);
    };
    size_t pr = 1, pc = nthreads;
    for (size_t r = 1; r <= nthreads; ++r) {
        if (nthreads % r != 0) continue;
        const size_t q = nthreads / r;
        if (r > rowUnits || q > colUnits) continue;
        if (pc > colUnits || gridCost(r, q) < gridCost(pr, pc)) { pr = r; pc = q; }
    }
    const size_t ntasks = pr * pc;

    // Packing buffer sizes for the largest rectangle.
    const size_t kcMax = std::min(KC, N);
    const size_t maxRows = (rowUnits + pr - 1) / pr * MR;
    const size_t maxCols = (colUnits + pc - 1) / pc * NR;
    const size_t apSize = roundUp(std::min(MC, maxRows) * kcMax, 8);
    const size_t bpSize = roundUp(std::min(NC, maxCols) * kcMax, 8);
    const size_t perThread = apSize + bpSize;
    double* const work = workspace(perThread * ntasks);

    #pragma omp parallel num_threads(static_cast<int>(ntasks))
    {
        const size_t tid = static_cast<size_t>(omp_get_thread_num());
        cpu_set_t oldMask;
        const bool pinned = topo && !topo->order.empty() &&
                            pthread_getaffinity_np(pthread_self(), sizeof(oldMask), &oldMask) == 0;
        if (pinned) {
            cpu_set_t one;
            CPU_ZERO(&one);
            CPU_SET(topo->order[tid % topo->order.size()], &one);
            pthread_setaffinity_np(pthread_self(), sizeof(one), &one);
        }
        double* const ap = work + tid * perThread;
        double* const bp = ap + apSize;

        // With a full team, schedule(static, 1) gives thread t exactly rectangle t.
        #pragma omp for schedule(static, 1)
        for (size_t t = 0; t < ntasks; ++t) {
            size_t rb, re, cb, ce;
            splitRange(rowUnits, pr, t / pc, rb, re);
            splitRange(colUnits, pc, t % pc, cb, ce);
            const size_t i0 = rb * MR, i1 = std::min(N, re * MR);
            const size_t j0 = cb * NR, j1 = std::min(N, ce * NR);
            if (i0 < i1 && j0 < j1)
                multiplyBlock(a, b, c, N, i0, i1 - i0, j0, j1 - j0, ap, bp);
        }

        if (pinned) pthread_setaffinity_np(pthread_self(), sizeof(oldMask), &oldMask);
    }
}

// Simple validation: compute a single element and compare
bool validateResult(const std::vector<double>& A, const std::vector<double>& B,
                   const std::vector<double>& C, const size_t N) {
    // Check a few random positions
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};
    
    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = checkPoints[pi] % N;
            const size_t j = checkPoints[pj] % N;
            
            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += A[i * N + k] * B[k * N + j];
            }
            
            const double actual = C[i * N + j];
            const double relError = std::abs((actual - expected) / (expected + 1e-10));
            
            if (relError > 1e-6) {
                printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                       i, j, expected, actual, relError);
                return false;
            }
        }
    }
    
    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    size_t N = 512;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            N = atoi(argv[++i]);
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
    
    printf("Matrix Multiplication Benchmark\n");
    printf("Matrix size: %zu x %zu\n", N, N);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    // Allocate matrices
    std::vector<double> A(N * N);
    std::vector<double> B(N * N);
    std::vector<double> C(N * N);
    
    // Initialize matrices
    printf("Initializing matrices...\n");
    initMatrix(A, N);
    initMatrix(B, N);
    
    // Perform matrix multiplication
    printf("Computing matrix multiplication...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    matrixMultiply(A, B, C, N);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate GFLOPS
    double gflops = (2.0 * N * N * N) / (duration.count() / 1000.0) / 1e9;
    printf("Performance: %.3f GFLOPS\n", gflops);
    
    // Print results for external validation
    if (printResults) {
        print_results(C, "MatrixC");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(A, B, C, N);
        
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
