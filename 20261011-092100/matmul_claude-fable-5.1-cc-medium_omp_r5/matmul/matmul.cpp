#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include <omp.h>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

#include "../common/results_output.hpp"

// Pin every OpenMP thread to its own CPU (thread t -> t-th CPU of the set the
// process is allowed to run on). This prevents the scheduler from migrating
// threads across cores/NUMA nodes mid-computation, which otherwise costs a
// large factor on multi-socket machines. The user's own binding policy
// (OMP_PROC_BIND / OMP_PLACES) always takes precedence: if any binding was
// requested through the OpenMP runtime, nothing is done here.
static void bindThreadsToCpus() {
#if defined(__linux__)
    if (omp_get_proc_bind() != omp_proc_bind_false) return;

    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) return;

    std::vector<int> cpus;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (CPU_ISSET(cpu, &allowed)) cpus.push_back(cpu);
    }
    if (cpus.empty()) return;

#pragma omp parallel
    {
        const size_t tid = static_cast<size_t>(omp_get_thread_num());
        cpu_set_t one;
        CPU_ZERO(&one);
        CPU_SET(cpus[tid % cpus.size()], &one);
        pthread_setaffinity_np(pthread_self(), sizeof(one), &one);
    }
#endif
}

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

// Parallel initialization: also performs NUMA "first touch" so that each row of
// the matrix is backed by memory local to the thread that will later read it.
void initMatrix(std::vector<double>& mat, const size_t N) {
    double* __restrict__ m = mat.data();
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            m[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

// ---------------------------------------------------------------------------
// Parallel, cache-blocked GEMM (C = A * B, row-major, square N x N)
//
// Thread-level decomposition (NUMA / bandwidth friendly):
//   The T threads are arranged in a Pi x Pj grid and C is cut into Pi x Pj
//   blocks; each thread owns exactly one block and therefore only ever reads
//   its own row strip of A and column strip of B. Every element of C is written
//   by a single thread, so there are no races, atomics or reductions.
//
// Cache-level blocking inside a thread's block (GotoBLAS structure):
//   for each k-chunk of BK:
//     pack the BK x (my columns) block of B into a contiguous panel  (L3)
//     for each BI-row slab of my rows:
//       pack the BI x BK slab of A into a contiguous panel             (L2)
//       for each NR-wide micro-panel of B:                              (L1)
//         for each MR-row micro-panel of A:
//           MR x NR register kernel over the whole k-chunk
//
// Numerical semantics: every C element is accumulated over k in strictly
// ascending order starting from 0.0, exactly like the original triple loop.
// ---------------------------------------------------------------------------
constexpr size_t MR = 6;    // rows per register block
constexpr size_t NR = 8;    // columns per register block (two 4-wide vectors)
constexpr size_t BI = 96;   // rows of A packed per slab    (multiple of MR); 96*256*8 = 192 KiB
constexpr size_t BK = 256;  // k-chunk length
static_assert(BI % MR == 0, "BI must be a multiple of MR");

// 4 doubles per vector register (AVX2 width). Uses GCC/Clang vector
// extensions only -- no external dependency, works on any target.
constexpr size_t VL = 4;
constexpr size_t NV = NR / VL;
typedef double v4d __attribute__((vector_size(VL * sizeof(double))));

// Packed A slab:  Ap[(p * BK + k) * MR + r] = A[i0 + p*MR + r][k0 + k]
// Packed B block: Bp[(q * BK + k) * NR + c] = B[k0 + k][j0 + q*NR + c]
// Rows/columns beyond the range are zero padded; padded results are never
// written back to C.
static inline void packA(const double* __restrict__ A, double* __restrict__ Ap, const size_t N,
                         const size_t i0, const size_t i1, const size_t k0, const size_t k1) {
    const size_t klen = k1 - k0;
    const size_t numPanels = (i1 - i0 + MR - 1) / MR;
    for (size_t p = 0; p < numPanels; ++p) {
        double* __restrict__ dst = Ap + p * BK * MR;
        const size_t ibase = i0 + p * MR;
        for (size_t r = 0; r < MR; ++r) {
            const size_t i = ibase + r;
            if (i < i1) {
                const double* __restrict__ srcRow = A + i * N + k0;
                for (size_t k = 0; k < klen; ++k) dst[k * MR + r] = srcRow[k];
            } else {
                for (size_t k = 0; k < klen; ++k) dst[k * MR + r] = 0.0;
            }
        }
    }
}

static inline void packB(const double* __restrict__ B, double* __restrict__ Bp, const size_t N,
                         const size_t j0, const size_t j1, const size_t k0, const size_t k1) {
    const size_t klen = k1 - k0;
    const size_t numPanels = (j1 - j0 + NR - 1) / NR;
    for (size_t q = 0; q < numPanels; ++q) {
        double* __restrict__ dst = Bp + q * BK * NR;
        const size_t jbase = j0 + q * NR;
        const size_t cols = std::min(NR, j1 - jbase);
        if (cols == NR) {
            for (size_t k = 0; k < klen; ++k) {
                const double* __restrict__ srcRow = B + (k0 + k) * N + jbase;
                double* __restrict__ d = dst + k * NR;
#pragma omp simd
                for (size_t c = 0; c < NR; ++c) d[c] = srcRow[c];
            }
        } else {
            for (size_t k = 0; k < klen; ++k) {
                const double* __restrict__ srcRow = B + (k0 + k) * N + jbase;
                double* __restrict__ d = dst + k * NR;
                size_t c = 0;
                for (; c < cols; ++c) d[c] = srcRow[c];
                for (; c < NR; ++c) d[c] = 0.0;
            }
        }
    }
}

static inline v4d loadv(const double* p) {
    v4d v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}
static inline void storev(double* p, const v4d v) {
    std::memcpy(p, &v, sizeof(v));
}
static inline v4d broadcastv(const double x) {
    return v4d{x, x, x, x};
}

// Register micro-kernel:
//   acc[MR][NR] (held in MR*NV vector registers) += Ap-panel(k) * Bp-panel(k)
//   for k in [0, klen), in ascending k order.
static inline void microKernel(const double* __restrict__ Ap, const double* __restrict__ Bp,
                               const size_t klen, v4d (&acc)[MR][NV]) {
    for (size_t k = 0; k < klen; ++k) {
        const double* __restrict__ a = Ap + k * MR;
        const double* __restrict__ b = Bp + k * NR;
        v4d bv[NV];
#pragma GCC unroll 8
        for (size_t v = 0; v < NV; ++v) bv[v] = loadv(b + v * VL);
#pragma GCC unroll 8
        for (size_t r = 0; r < MR; ++r) {
            const v4d av = broadcastv(a[r]);
#pragma GCC unroll 8
            for (size_t v = 0; v < NV; ++v) acc[r][v] += av * bv[v];
        }
    }
}

// C[i0..i1) x [j0..j1) += Ap-slab * Bp-block over the packed k-chunk.
// Ap holds rows [i0, i1), Bp holds columns [j0, j1).
static inline void slabKernel(const double* __restrict__ Ap, const double* __restrict__ Bp,
                              double* __restrict__ C, const size_t N,
                              const size_t i0, const size_t i1,
                              const size_t j0, const size_t j1,
                              const size_t klen) {
    const size_t numPanelsI = (i1 - i0 + MR - 1) / MR;
    const size_t numPanelsJ = (j1 - j0 + NR - 1) / NR;

    for (size_t q = 0; q < numPanelsJ; ++q) {
        const size_t j = j0 + q * NR;
        const size_t cols = std::min(NR, j1 - j);
        const double* __restrict__ bp = Bp + q * BK * NR;
        for (size_t p = 0; p < numPanelsI; ++p) {
            const size_t i = i0 + p * MR;
            const size_t rows = std::min(MR, i1 - i);
            const double* __restrict__ ap = Ap + p * BK * MR;

            v4d acc[MR][NV];
            if (rows == MR && cols == NR) {
#pragma GCC unroll 8
                for (size_t r = 0; r < MR; ++r)
#pragma GCC unroll 8
                    for (size_t v = 0; v < NV; ++v) acc[r][v] = loadv(C + (i + r) * N + j + v * VL);
                microKernel(ap, bp, klen, acc);
#pragma GCC unroll 8
                for (size_t r = 0; r < MR; ++r)
#pragma GCC unroll 8
                    for (size_t v = 0; v < NV; ++v) storev(C + (i + r) * N + j + v * VL, acc[r][v]);
            } else {
                // Edge block: compute on a zero-padded register block, write
                // back only the entries that exist in C.
                alignas(32) double tmp[MR][NR];
                for (size_t r = 0; r < MR; ++r)
                    for (size_t c = 0; c < NR; ++c)
                        tmp[r][c] = (r < rows && c < cols) ? C[(i + r) * N + j + c] : 0.0;
                for (size_t r = 0; r < MR; ++r)
                    for (size_t v = 0; v < NV; ++v) acc[r][v] = loadv(&tmp[r][v * VL]);
                microKernel(ap, bp, klen, acc);
                for (size_t r = 0; r < MR; ++r)
                    for (size_t v = 0; v < NV; ++v) storev(&tmp[r][v * VL], acc[r][v]);
                for (size_t r = 0; r < rows; ++r)
                    for (size_t c = 0; c < cols; ++c)
                        C[(i + r) * N + j + c] = tmp[r][c];
            }
        }
    }
}

// Split [0, N) into P nearly equal ranges whose boundaries are aligned to
// `align` (so interior micro-panels are always full). Returns the start of
// range p (p == P yields N).
static inline size_t partitionStart(const size_t N, const size_t P, const size_t p, const size_t align) {
    if (p >= P) return N;
    const size_t raw = (N * p) / P;
    return std::min(N, (raw / align) * align);
}

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, const size_t N) {
    const double* __restrict__ a = A.data();
    const double* __restrict__ b = B.data();
    double* __restrict__ c = C.data();

    // One contiguous packing workspace for the whole team, allocated by a
    // single thread once the team size is known and sliced per thread. It is
    // kept alive across calls: a single persistent allocation (instead of one
    // heap arena per thread, or a fresh mapping per call) avoids serialising
    // all threads on the process memory-map lock and the TLB-shootdown cost of
    // unmapping memory that every core has touched. The memory is deliberately
    // left uninitialised so each thread first-touches (and thus NUMA-places)
    // its own slice in parallel.
    static std::unique_ptr<double[]> workspace;
    static size_t workspaceSize = 0;
    size_t perThread = 0;

#pragma omp parallel
    {
        const size_t T = static_cast<size_t>(omp_get_num_threads());
        const size_t tid = static_cast<size_t>(omp_get_thread_num());

        // Thread grid Pi x Pj = T, as close to square as possible.
        size_t Pi = static_cast<size_t>(std::sqrt(static_cast<double>(T)));
        while (Pi > 1 && T % Pi != 0) --Pi;
        const size_t Pj = T / Pi;
        const size_t pi = tid / Pj;
        const size_t pj = tid % Pj;

        // This thread's block of C.
        const size_t r0 = partitionStart(N, Pi, pi, MR);
        const size_t r1 = partitionStart(N, Pi, pi + 1, MR);
        const size_t c0 = partitionStart(N, Pj, pj, NR);
        const size_t c1 = partitionStart(N, Pj, pj + 1, NR);

#pragma omp single
        {
            size_t maxCols = 0;
            for (size_t q = 0; q < Pj; ++q) {
                maxCols = std::max(maxCols, partitionStart(N, Pj, q + 1, NR) - partitionStart(N, Pj, q, NR));
            }
            const size_t paddedCols = ((maxCols + NR - 1) / NR) * NR;
            perThread = BI * BK + BK * paddedCols;
            perThread = ((perThread + 7) / 8) * 8;  // keep every slice 64-byte aligned
            if (workspaceSize < perThread * T) {
                workspace.reset(new double[perThread * T]);
                workspaceSize = perThread * T;
            }
        }  // implicit barrier: workspace / perThread visible to all threads

        if (r1 > r0 && c1 > c0) {
            double* __restrict__ ap = workspace.get() + tid * perThread;
            double* __restrict__ bp = ap + BI * BK;

            // C block starts from zero (same as the original "sum = 0.0").
            for (size_t i = r0; i < r1; ++i) {
                std::fill(c + i * N + c0, c + i * N + c1, 0.0);
            }

            for (size_t k0 = 0; k0 < N; k0 += BK) {
                const size_t k1 = std::min(k0 + BK, N);
                packB(b, bp, N, c0, c1, k0, k1);
                for (size_t i0 = r0; i0 < r1; i0 += BI) {
                    const size_t i1 = std::min(i0 + BI, r1);
                    packA(a, ap, N, i0, i1, k0, k1);
                    slabKernel(ap, bp, c, N, i0, i1, c0, c1, k1 - k0);
                }
            }
        }
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
    printf("OpenMP threads: %d\n", omp_get_max_threads());
    bindThreadsToCpus();
    
    // Allocate matrices
    std::vector<double> A(N * N);
    std::vector<double> B(N * N);
    std::vector<double> C(N * N);
    
    // Initialize matrices
    printf("Initializing matrices...\n");
    initMatrix(A, N);
    initMatrix(B, N);
    initMatrix(C, N);  // first-touch placement of C; overwritten by the multiply
    
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
