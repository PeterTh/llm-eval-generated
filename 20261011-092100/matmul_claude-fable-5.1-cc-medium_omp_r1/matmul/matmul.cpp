#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <omp.h>

#include "../common/results_output.hpp"

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

#ifdef __linux__
#include <sched.h>
#endif

// ---------------------------------------------------------------------------
// Thread placement
//
// Static work distribution below relies on each thread repeatedly touching the
// same rows (first-touch NUMA placement in initMatrix, then the same rows in
// matrixMultiply). That only pays off if threads do not migrate between cores
// or sockets. If the OpenMP runtime was not asked to bind threads
// (OMP_PROC_BIND / OMP_PLACES unset), pin thread t to the t-th CPU of the
// process' inherited affinity mask. If the user already bound threads, or the
// mask has fewer CPUs than threads (oversubscription), leave everything alone.
// ---------------------------------------------------------------------------
static void pinThreadsIfUnbound() {
#ifdef __linux__
    static bool done = false;
    if (done) return;
    done = true;
    if (omp_get_proc_bind() != omp_proc_bind_false) return;

    cpu_set_t mask;
    CPU_ZERO(&mask);
    if (sched_getaffinity(0, sizeof(mask), &mask) != 0) return;
    std::vector<int> cpus;
    for (int c = 0; c < CPU_SETSIZE; ++c) {
        if (CPU_ISSET(c, &mask)) cpus.push_back(c);
    }
    if (static_cast<size_t>(omp_get_max_threads()) > cpus.size()) return;

#pragma omp parallel
    {
        const int t = omp_get_thread_num();
        if (static_cast<size_t>(t) < cpus.size()) {
            cpu_set_t one;
            CPU_ZERO(&one);
            CPU_SET(cpus[t], &one);
            sched_setaffinity(0, sizeof(one), &one);
        }
    }
#endif
}

void initMatrix(std::vector<double>& mat, const size_t N) {
    pinThreadsIfUnbound();
    double* __restrict__ m = mat.data();
    // Static row distribution: the thread that initializes a row is the one
    // that later computes with it, so first-touch places the pages on the
    // right NUMA node.
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            m[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

// ---------------------------------------------------------------------------
// Blocked, register-tiled matrix multiplication
//
// C is split into (IB x JB) tiles, each owned by exactly one thread, so no
// synchronization on C is needed. Within a tile the k dimension is processed
// in blocks of KB. For each k block, the B sub-panel B[k0:k1, j0:j1] is packed
// into contiguous NR-wide micro-panels in a per-thread buffer: this keeps the
// panel in L2 and avoids the cache-set aliasing that strided row access
// suffers from when N is a power of two. A micro-kernel then computes an
// MR x NR block of C held entirely in vector registers.
//
// Numerics: every C[i][j] starts at 0 and receives A[i][k]*B[k][j] for
// k = 0,1,...,N-1 in increasing order, exactly like the original i-j-k loop.
// ---------------------------------------------------------------------------
constexpr size_t MR = 4;         // rows of C in the register tile
constexpr size_t NR = 12;        // columns of C in the register tile (3 x 4 doubles)
constexpr size_t JB = 192;       // column block (multiple of NR)
constexpr size_t KB = 256;       // k block: packed B panel = KB*JB doubles = 384 KiB
constexpr size_t IB_MAX = 64;    // largest row block (multiple of MR)
static_assert(JB % NR == 0 && IB_MAX % MR == 0);

typedef double v4d __attribute__((vector_size(4 * sizeof(double)), aligned(alignof(double))));

static inline v4d loadv(const double* p) { return *reinterpret_cast<const v4d*>(p); }
static inline void storev(double* p, v4d v) { *reinterpret_cast<v4d*>(p) = v; }

// MR x NR = 4 x 12 register tile: 12 accumulators + 3 B vectors + 1 broadcast
// fit the 16 AVX2 ymm registers. a0..a3 point at A[i+r][k0] (unit stride in k),
// bp at the packed B micro-panel (bp[k*NR + c]), c at C[i][j].
static inline void microKernel(const double* __restrict__ a0, const double* __restrict__ a1,
                               const double* __restrict__ a2, const double* __restrict__ a3,
                               const double* __restrict__ bp, double* __restrict__ c,
                               const size_t N, const size_t kc) {
    v4d c00 = loadv(c + 0 * N + 0), c01 = loadv(c + 0 * N + 4), c02 = loadv(c + 0 * N + 8);
    v4d c10 = loadv(c + 1 * N + 0), c11 = loadv(c + 1 * N + 4), c12 = loadv(c + 1 * N + 8);
    v4d c20 = loadv(c + 2 * N + 0), c21 = loadv(c + 2 * N + 4), c22 = loadv(c + 2 * N + 8);
    v4d c30 = loadv(c + 3 * N + 0), c31 = loadv(c + 3 * N + 4), c32 = loadv(c + 3 * N + 8);
    for (size_t k = 0; k < kc; ++k) {
        const v4d b0 = loadv(bp + 0), b1 = loadv(bp + 4), b2 = loadv(bp + 8);
        v4d av;
        av = v4d{} + a0[k]; c00 += av * b0; c01 += av * b1; c02 += av * b2;
        av = v4d{} + a1[k]; c10 += av * b0; c11 += av * b1; c12 += av * b2;
        av = v4d{} + a2[k]; c20 += av * b0; c21 += av * b1; c22 += av * b2;
        av = v4d{} + a3[k]; c30 += av * b0; c31 += av * b1; c32 += av * b2;
        bp += NR;
    }
    storev(c + 0 * N + 0, c00); storev(c + 0 * N + 4, c01); storev(c + 0 * N + 8, c02);
    storev(c + 1 * N + 0, c10); storev(c + 1 * N + 4, c11); storev(c + 1 * N + 8, c12);
    storev(c + 2 * N + 0, c20); storev(c + 2 * N + 4, c21); storev(c + 2 * N + 8, c22);
    storev(c + 3 * N + 0, c30); storev(c + 3 * N + 4, c31); storev(c + 3 * N + 8, c32);
}

// Edge rows / columns that do not fill a whole MR x NR tile: same i-k-j
// accumulation order, reading A and B in place.
static inline void edgeKernel(const double* __restrict__ a, const double* __restrict__ b,
                              double* __restrict__ c, const size_t N,
                              const size_t i0, const size_t i1,
                              const size_t j0, const size_t j1,
                              const size_t k0, const size_t k1) {
    for (size_t i = i0; i < i1; ++i) {
        const double* __restrict__ arow = a + i * N;
        double* __restrict__ crow = c + i * N;
        for (size_t k = k0; k < k1; ++k) {
            const double aik = arow[k];
            const double* __restrict__ brow = b + k * N;
#pragma omp simd
            for (size_t j = j0; j < j1; ++j) {
                crow[j] += aik * brow[j];
            }
        }
    }
}

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, const size_t N) {
    const double* __restrict__ a = A.data();
    const double* __restrict__ b = B.data();
    double* __restrict__ c = C.data();

    const size_t numJB = (N + JB - 1) / JB;

    // Row block: as large as possible (amortizes packing of the B panel over
    // more rows) while still giving every thread at least ~8 tiles under the
    // static schedule, which keeps the load imbalance from tile granularity
    // small.
    size_t IB = IB_MAX;
    {
        const size_t nth = static_cast<size_t>(omp_get_max_threads());
        while (IB > MR && ((N + IB - 1) / IB) * numJB < 8 * nth) IB /= 2;
    }
    const size_t numIB = (N + IB - 1) / IB;

#pragma omp parallel
    {
        // Per-thread packing buffer for one KB x JB panel of B.
        alignas(64) double bpack[KB * JB];

        // schedule(static) is essential: with collapse(2) the tiles of one
        // thread are consecutive, i.e. the same row block across all column
        // blocks, and thread t gets the rows it first-touched in initMatrix.
#pragma omp for collapse(2) schedule(static)
        for (size_t ib = 0; ib < numIB; ++ib) {
            for (size_t jb = 0; jb < numJB; ++jb) {
                const size_t i0 = ib * IB;
                const size_t i1 = (i0 + IB < N) ? i0 + IB : N;
                const size_t j0 = jb * JB;
                const size_t j1 = (j0 + JB < N) ? j0 + JB : N;
                const size_t iFull = i0 + (i1 - i0) / MR * MR;   // rows covered by micro-kernel
                const size_t jFull = j0 + (j1 - j0) / NR * NR;   // cols covered by micro-kernel

                for (size_t i = i0; i < i1; ++i) {
                    double* __restrict__ crow = c + i * N;
                    for (size_t j = j0; j < j1; ++j) {
                        crow[j] = 0.0;
                    }
                }

                for (size_t k0 = 0; k0 < N; k0 += KB) {
                    const size_t k1 = (k0 + KB < N) ? k0 + KB : N;
                    const size_t kc = k1 - k0;

                    // Pack B[k0:k1, j0:jFull] as NR-wide micro-panels [panel][k][c].
                    for (size_t jp = j0; jp < jFull; jp += NR) {
                        double* __restrict__ dst = bpack + (jp - j0) * kc;
                        for (size_t k = k0; k < k1; ++k) {
                            const double* __restrict__ srcp = b + k * N + jp;
                            for (size_t cc = 0; cc < NR; ++cc) {
                                dst[cc] = srcp[cc];
                            }
                            dst += NR;
                        }
                    }

                    for (size_t ip = i0; ip < iFull; ip += MR) {
                        const double* __restrict__ ap = a + ip * N + k0;
                        for (size_t jp = j0; jp < jFull; jp += NR) {
                            microKernel(ap, ap + N, ap + 2 * N, ap + 3 * N,
                                        bpack + (jp - j0) * kc, c + ip * N + jp, N, kc);
                        }
                    }
                    if (jFull < j1) edgeKernel(a, b, c, N, i0, iFull, jFull, j1, k0, k1);
                    if (iFull < i1) edgeKernel(a, b, c, N, iFull, i1, j0, j1, k0, k1);
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
