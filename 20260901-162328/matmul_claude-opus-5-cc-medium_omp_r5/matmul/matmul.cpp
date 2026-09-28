#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <omp.h>

#if defined(__linux__)
#include <sys/syscall.h>
#include <unistd.h>
#endif

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define MATMUL_USE_AVX2 1
#endif

#include "../common/results_output.hpp"

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

void initMatrix(std::vector<double>& mat, const size_t N) {
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

// ---------------------------------------------------------------------------
// Blocked, packed, register-tiled matrix multiplication parallelized with OpenMP.
//
// The output matrix C is split into (MC x NC) tiles which are distributed over
// the threads; every thread packs the A/B slivers it needs into thread-private
// buffers (so the innermost kernel only sees contiguous, aligned data) and
// accumulates the tile with a MR x NR register-blocked micro-kernel.
// ---------------------------------------------------------------------------

// Spread (or stop spreading) subsequent page allocations of this thread over all
// NUMA nodes, equivalent to running under `numactl --interleave=all`.  Best
// effort: if the platform does not support it, nothing changes.
void setInterleavedMemoryPolicy(const bool interleave) {
#if defined(__linux__)
    constexpr int MPOL_DEFAULT = 0;
    constexpr int MPOL_INTERLEAVE = 3;
    constexpr int MPOL_F_MEMS_ALLOWED = 4;
    constexpr unsigned long MAX_NODES = 1024;

    unsigned long nodemask[MAX_NODES / (8 * sizeof(unsigned long))] = {};

    if (!interleave) {
        syscall(SYS_set_mempolicy, MPOL_DEFAULT, nodemask, MAX_NODES);
        return;
    }

    if (syscall(SYS_get_mempolicy, nullptr, nodemask, MAX_NODES, nullptr, MPOL_F_MEMS_ALLOWED) == 0) {
        syscall(SYS_set_mempolicy, MPOL_INTERLEAVE, nodemask, MAX_NODES);
    }
#else
    (void)interleave;
#endif
}

namespace {

constexpr size_t MR = 6; // micro-tile rows
constexpr size_t NR = 8; // micro-tile columns (2 AVX2 vectors)

// C_tile[MR][NR] += Ap^T * Bp  (Ap: kc x MR, Bp: kc x NR, both packed)
inline void microKernel(const double* __restrict Ap, const double* __restrict Bp, const size_t kc,
                        double* __restrict C, const size_t ldc) {
#ifdef MATMUL_USE_AVX2
    __m256d c00 = _mm256_setzero_pd(), c01 = _mm256_setzero_pd();
    __m256d c10 = _mm256_setzero_pd(), c11 = _mm256_setzero_pd();
    __m256d c20 = _mm256_setzero_pd(), c21 = _mm256_setzero_pd();
    __m256d c30 = _mm256_setzero_pd(), c31 = _mm256_setzero_pd();
    __m256d c40 = _mm256_setzero_pd(), c41 = _mm256_setzero_pd();
    __m256d c50 = _mm256_setzero_pd(), c51 = _mm256_setzero_pd();

    for (size_t k = 0; k < kc; ++k) {
        const __m256d b0 = _mm256_load_pd(Bp);
        const __m256d b1 = _mm256_load_pd(Bp + 4);
        Bp += NR;

        __m256d a = _mm256_broadcast_sd(Ap + 0);
        c00 = _mm256_fmadd_pd(a, b0, c00);
        c01 = _mm256_fmadd_pd(a, b1, c01);
        a = _mm256_broadcast_sd(Ap + 1);
        c10 = _mm256_fmadd_pd(a, b0, c10);
        c11 = _mm256_fmadd_pd(a, b1, c11);
        a = _mm256_broadcast_sd(Ap + 2);
        c20 = _mm256_fmadd_pd(a, b0, c20);
        c21 = _mm256_fmadd_pd(a, b1, c21);
        a = _mm256_broadcast_sd(Ap + 3);
        c30 = _mm256_fmadd_pd(a, b0, c30);
        c31 = _mm256_fmadd_pd(a, b1, c31);
        a = _mm256_broadcast_sd(Ap + 4);
        c40 = _mm256_fmadd_pd(a, b0, c40);
        c41 = _mm256_fmadd_pd(a, b1, c41);
        a = _mm256_broadcast_sd(Ap + 5);
        c50 = _mm256_fmadd_pd(a, b0, c50);
        c51 = _mm256_fmadd_pd(a, b1, c51);
        Ap += MR;
    }

#define MATMUL_STORE_ROW(r, lo, hi)                                                                \
    _mm256_storeu_pd(C + (r)*ldc, _mm256_add_pd(_mm256_loadu_pd(C + (r)*ldc), lo));                 \
    _mm256_storeu_pd(C + (r)*ldc + 4, _mm256_add_pd(_mm256_loadu_pd(C + (r)*ldc + 4), hi));
    MATMUL_STORE_ROW(0, c00, c01)
    MATMUL_STORE_ROW(1, c10, c11)
    MATMUL_STORE_ROW(2, c20, c21)
    MATMUL_STORE_ROW(3, c30, c31)
    MATMUL_STORE_ROW(4, c40, c41)
    MATMUL_STORE_ROW(5, c50, c51)
#undef MATMUL_STORE_ROW
#else
    double acc[MR][NR] = {};
    for (size_t k = 0; k < kc; ++k) {
        for (size_t i = 0; i < MR; ++i) {
            const double a = Ap[k * MR + i];
            for (size_t j = 0; j < NR; ++j) {
                acc[i][j] += a * Bp[k * NR + j];
            }
        }
    }
    for (size_t i = 0; i < MR; ++i) {
        for (size_t j = 0; j < NR; ++j) {
            C[i * ldc + j] += acc[i][j];
        }
    }
#endif
}

// Pack A[i0 .. i0+mc, p0 .. p0+kc] into MR-row panels (k-major within a panel).
void packA(const double* __restrict A, const size_t N, const size_t i0, const size_t mc,
           const size_t p0, const size_t kc, double* __restrict Ap) {
    for (size_t ip = 0; ip < mc; ip += MR) {
        const size_t rows = std::min(MR, mc - ip);
        double* __restrict dst = Ap + ip * kc;
        for (size_t ii = 0; ii < rows; ++ii) {
            const double* __restrict src = A + (i0 + ip + ii) * N + p0;
            for (size_t k = 0; k < kc; ++k) {
                dst[k * MR + ii] = src[k];
            }
        }
        for (size_t ii = rows; ii < MR; ++ii) {
            for (size_t k = 0; k < kc; ++k) {
                dst[k * MR + ii] = 0.0;
            }
        }
    }
}

// Pack B[p0 .. p0+kc, j0 .. j0+nc] into NR-column panels (k-major within a panel).
void packB(const double* __restrict B, const size_t N, const size_t p0, const size_t kc,
           const size_t j0, const size_t nc, double* __restrict Bp) {
    for (size_t jp = 0; jp < nc; jp += NR) {
        const size_t cols = std::min(NR, nc - jp);
        double* __restrict dst = Bp + jp * kc;
        for (size_t k = 0; k < kc; ++k) {
            const double* __restrict src = B + (p0 + k) * N + j0 + jp;
            for (size_t jj = 0; jj < cols; ++jj) {
                dst[k * NR + jj] = src[jj];
            }
            for (size_t jj = cols; jj < NR; ++jj) {
                dst[k * NR + jj] = 0.0;
            }
        }
    }
}

inline double* allocAligned(const size_t count) {
    const size_t bytes = ((count * sizeof(double)) + 63) & ~size_t(63);
    return static_cast<double*>(std::aligned_alloc(64, bytes));
}

} // namespace

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, const size_t N) {
    if (N == 0) {
        return;
    }

    const double* __restrict Ad = A.data();
    const double* __restrict Bd = B.data();
    double* __restrict Cd = C.data();

    const int numThreads = omp_get_max_threads();

    // Pick the tile sizes: the packed panels and the C tile have to stay cache
    // resident, the tile has to be big enough to amortize the packing, and there
    // have to be enough tiles to keep every thread busy.  The starting points
    // were tuned on a dual socket AVX2 machine.
    constexpr size_t MIN_MC = 48;
    const size_t KC = std::min(N, size_t(256));
    // NC/MC are rounded up to whole micro-panels: the packed buffers always hold
    // full MR x NR panels (zero padded at the matrix borders).
    const size_t NC = (std::min(N, size_t(64)) + NR - 1) / NR * NR;
    size_t MC = std::min(size_t(768), std::max(MIN_MC, N / 6));

    const size_t target = static_cast<size_t>(numThreads) + static_cast<size_t>(numThreads) / 4;
    const auto numTiles = [&](const size_t mc) { return ((N + mc - 1) / mc) * ((N + NC - 1) / NC); };
    while (MC > MIN_MC && numTiles(MC) < target) {
        MC = std::max(MIN_MC, MC / 2);
    }
    MC = (std::min(MC, N) + MR - 1) / MR * MR;

    const size_t apSize = MC * KC;
    const size_t bpSize = NC * KC;

#pragma omp parallel proc_bind(spread)
    {
#pragma omp for schedule(static)
        for (size_t i = 0; i < N; ++i) {
            std::memset(Cd + i * N, 0, N * sizeof(double));
        }

        double* const Ap = allocAligned(apSize);
        double* const Bp = allocAligned(bpSize);
        alignas(64) double ctmp[MR * NR];

        if (Ap == nullptr || Bp == nullptr) {
            printf("Failed to allocate packing buffers\n");
            std::exit(1);
        }

#pragma omp for collapse(2) schedule(dynamic)
        for (size_t ic = 0; ic < N; ic += MC) {
            for (size_t jc = 0; jc < N; jc += NC) {
                const size_t mc = std::min(MC, N - ic);
                const size_t nc = std::min(NC, N - jc);

                for (size_t pc = 0; pc < N; pc += KC) {
                    const size_t kc = std::min(KC, N - pc);
                    packB(Bd, N, pc, kc, jc, nc, Bp);
                    packA(Ad, N, ic, mc, pc, kc, Ap);

                    for (size_t jp = 0; jp < nc; jp += NR) {
                        const size_t cols = std::min(NR, nc - jp);
                        for (size_t ip = 0; ip < mc; ip += MR) {
                            const size_t rows = std::min(MR, mc - ip);
                            double* const Ctile = Cd + (ic + ip) * N + jc + jp;

                            if (rows == MR && cols == NR) {
                                microKernel(Ap + ip * kc, Bp + jp * kc, kc, Ctile, N);
                            } else {
                                std::memset(ctmp, 0, sizeof(ctmp));
                                microKernel(Ap + ip * kc, Bp + jp * kc, kc, ctmp, NR);
                                for (size_t ii = 0; ii < rows; ++ii) {
                                    for (size_t jj = 0; jj < cols; ++jj) {
                                        Ctile[ii * N + jj] += ctmp[ii * NR + jj];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        std::free(Ap);
        std::free(Bp);
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

// A tuned GEMM saturates the FP pipelines of a core, so SMT siblings only add
// cache pressure.  If the user did not request a specific thread count, default
// to one thread per physical core.
void selectDefaultThreadCount() {
    if (getenv("OMP_NUM_THREADS") != nullptr) {
        return;
    }

    int smtFactor = 1;
#if defined(__linux__)
    if (FILE* f = fopen("/sys/devices/system/cpu/cpu0/topology/thread_siblings_list", "r")) {
        char line[256] = {};
        if (fgets(line, sizeof(line), f) != nullptr) {
            int siblings = 1;
            for (const char* p = line; *p != '\0'; ++p) {
                if (*p == ',') {
                    ++siblings;
                } else if (*p == '-') { // range "0-1" covers 2 siblings
                    siblings += 1;
                }
            }
            smtFactor = siblings;
        }
        fclose(f);
    }
#endif

    const int threads = omp_get_max_threads() / std::max(1, smtFactor);
    if (threads >= 1) {
        omp_set_num_threads(threads);
    }
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
    
    selectDefaultThreadCount();

    printf("Matrix Multiplication Benchmark\n");
    printf("Matrix size: %zu x %zu\n", N, N);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    // Allocate matrices.  The pages are spread over all NUMA nodes so that the
    // aggregate memory bandwidth of all sockets is available to the threads
    // (the matrices are first-touched by this thread when they are zeroed).
    setInterleavedMemoryPolicy(true);
    std::vector<double> A(N * N);
    std::vector<double> B(N * N);
    std::vector<double> C(N * N);
    setInterleavedMemoryPolicy(false);

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
