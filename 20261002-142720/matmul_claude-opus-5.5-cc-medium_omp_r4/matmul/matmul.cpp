#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

#include <omp.h>
#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
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

// Blocked, packed, OpenMP-parallel GEMM.
// Every C[i][j] is accumulated over k in strictly increasing order with the same
// multiply/add rounding as the reference loop, so results are bitwise identical.
namespace {

constexpr size_t MR = 6;    // micro-tile rows
constexpr size_t NR = 8;    // micro-tile cols (2 x 4 doubles)
constexpr size_t KC_MAX = 512;  // max k-block depth
constexpr size_t MCI = MR * 4;   // rows per packed A sub-block

inline size_t roundUp(size_t x, size_t m) { return (x + m - 1) / m * m; }

// Kernel: C_tile(MR x NR) (+)= Ap(kc x MR) * Bp(kc x NR); acc starts at 0 if first.
// Rounding mirrors the reference build (GCC -O3 -march=native): k terms are pairwise
// SLP-vectorized as separate mul + add, and only the leftover term for odd N is fused.
inline void microKernel(const size_t kc, const double* __restrict Ap, const double* __restrict Bp,
                        double* __restrict Ct, const size_t ldc, const bool first,
                        const bool fmaLast) {
#if defined(__AVX2__) && defined(__FMA__)
    __m256d c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, c50, c51;
    if (first) {
        c00 = c01 = c10 = c11 = c20 = c21 = c30 = c31 = c40 = c41 = c50 = c51 = _mm256_setzero_pd();
    } else {
        c00 = _mm256_loadu_pd(Ct + 0 * ldc); c01 = _mm256_loadu_pd(Ct + 0 * ldc + 4);
        c10 = _mm256_loadu_pd(Ct + 1 * ldc); c11 = _mm256_loadu_pd(Ct + 1 * ldc + 4);
        c20 = _mm256_loadu_pd(Ct + 2 * ldc); c21 = _mm256_loadu_pd(Ct + 2 * ldc + 4);
        c30 = _mm256_loadu_pd(Ct + 3 * ldc); c31 = _mm256_loadu_pd(Ct + 3 * ldc + 4);
        c40 = _mm256_loadu_pd(Ct + 4 * ldc); c41 = _mm256_loadu_pd(Ct + 4 * ldc + 4);
        c50 = _mm256_loadu_pd(Ct + 5 * ldc); c51 = _mm256_loadu_pd(Ct + 5 * ldc + 4);
    }
    // Reference rounding: product rounded, then added (no fusion) ...
    const size_t kMul = fmaLast ? kc - 1 : kc;
    for (size_t k = 0; k < kMul; ++k) {
        const __m256d b0 = _mm256_load_pd(Bp);
        const __m256d b1 = _mm256_load_pd(Bp + 4);
        __m256d a;
        a = _mm256_broadcast_sd(Ap + 0); c00 = _mm256_add_pd(c00, _mm256_mul_pd(a, b0)); c01 = _mm256_add_pd(c01, _mm256_mul_pd(a, b1));
        a = _mm256_broadcast_sd(Ap + 1); c10 = _mm256_add_pd(c10, _mm256_mul_pd(a, b0)); c11 = _mm256_add_pd(c11, _mm256_mul_pd(a, b1));
        a = _mm256_broadcast_sd(Ap + 2); c20 = _mm256_add_pd(c20, _mm256_mul_pd(a, b0)); c21 = _mm256_add_pd(c21, _mm256_mul_pd(a, b1));
        a = _mm256_broadcast_sd(Ap + 3); c30 = _mm256_add_pd(c30, _mm256_mul_pd(a, b0)); c31 = _mm256_add_pd(c31, _mm256_mul_pd(a, b1));
        a = _mm256_broadcast_sd(Ap + 4); c40 = _mm256_add_pd(c40, _mm256_mul_pd(a, b0)); c41 = _mm256_add_pd(c41, _mm256_mul_pd(a, b1));
        a = _mm256_broadcast_sd(Ap + 5); c50 = _mm256_add_pd(c50, _mm256_mul_pd(a, b0)); c51 = _mm256_add_pd(c51, _mm256_mul_pd(a, b1));
        Ap += MR;
        Bp += NR;
    }
    // ... except a trailing odd k term, which the reference build fuses.
    if (fmaLast) {
        const __m256d b0 = _mm256_load_pd(Bp);
        const __m256d b1 = _mm256_load_pd(Bp + 4);
        __m256d a;
        a = _mm256_broadcast_sd(Ap + 0); c00 = _mm256_fmadd_pd(a, b0, c00); c01 = _mm256_fmadd_pd(a, b1, c01);
        a = _mm256_broadcast_sd(Ap + 1); c10 = _mm256_fmadd_pd(a, b0, c10); c11 = _mm256_fmadd_pd(a, b1, c11);
        a = _mm256_broadcast_sd(Ap + 2); c20 = _mm256_fmadd_pd(a, b0, c20); c21 = _mm256_fmadd_pd(a, b1, c21);
        a = _mm256_broadcast_sd(Ap + 3); c30 = _mm256_fmadd_pd(a, b0, c30); c31 = _mm256_fmadd_pd(a, b1, c31);
        a = _mm256_broadcast_sd(Ap + 4); c40 = _mm256_fmadd_pd(a, b0, c40); c41 = _mm256_fmadd_pd(a, b1, c41);
        a = _mm256_broadcast_sd(Ap + 5); c50 = _mm256_fmadd_pd(a, b0, c50); c51 = _mm256_fmadd_pd(a, b1, c51);
    }
    _mm256_storeu_pd(Ct + 0 * ldc, c00); _mm256_storeu_pd(Ct + 0 * ldc + 4, c01);
    _mm256_storeu_pd(Ct + 1 * ldc, c10); _mm256_storeu_pd(Ct + 1 * ldc + 4, c11);
    _mm256_storeu_pd(Ct + 2 * ldc, c20); _mm256_storeu_pd(Ct + 2 * ldc + 4, c21);
    _mm256_storeu_pd(Ct + 3 * ldc, c30); _mm256_storeu_pd(Ct + 3 * ldc + 4, c31);
    _mm256_storeu_pd(Ct + 4 * ldc, c40); _mm256_storeu_pd(Ct + 4 * ldc + 4, c41);
    _mm256_storeu_pd(Ct + 5 * ldc, c50); _mm256_storeu_pd(Ct + 5 * ldc + 4, c51);
#else
    double acc[MR][NR];
    for (size_t r = 0; r < MR; ++r)
        for (size_t c = 0; c < NR; ++c)
            acc[r][c] = first ? 0.0 : Ct[r * ldc + c];
    const size_t kMul = fmaLast ? kc - 1 : kc;
    for (size_t k = 0; k < kMul; ++k) {
        for (size_t r = 0; r < MR; ++r)
            for (size_t c = 0; c < NR; ++c)
                acc[r][c] += Ap[r] * Bp[c];
        Ap += MR;
        Bp += NR;
    }
    if (fmaLast) {
        for (size_t r = 0; r < MR; ++r)
            for (size_t c = 0; c < NR; ++c)
                acc[r][c] = std::fma(Ap[r], Bp[c], acc[r][c]);
    }
    for (size_t r = 0; r < MR; ++r)
        for (size_t c = 0; c < NR; ++c)
            Ct[r * ldc + c] = acc[r][c];
#endif
}

struct AlignedBuf {
    double* p = nullptr;
    explicit AlignedBuf(size_t n) {
        p = static_cast<double*>(std::aligned_alloc(64, roundUp(std::max<size_t>(n, 1) * sizeof(double), 64)));
        if (!p) { fprintf(stderr, "Allocation failed\n"); std::exit(1); }
    }
    ~AlignedBuf() { std::free(p); }
    AlignedBuf(const AlignedBuf&) = delete;
    AlignedBuf& operator=(const AlignedBuf&) = delete;
};

} // namespace

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B, 
                    std::vector<double>& C, const size_t N) {
    if (N == 0) return;
    const size_t Np = roundUp(N, NR);
    const size_t nPanels = Np / NR;
    const size_t nKBlocks = (N + KC_MAX - 1) / KC_MAX;
    const size_t KC = (N + nKBlocks - 1) / nKBlocks;  // balanced k-blocks

    // Task tiles (TM x TN of C): choose the grid minimizing a simple makespan model,
    // rounds(tasks / threads) * tileWork * (1 + reload overhead ~ 1/TM + 1/TN).
    const size_t nThreads = static_cast<size_t>(omp_get_max_threads());
    size_t TM = MR, TN = NR;
    {
        double best = 1e300;
        const size_t tmMax = std::min<size_t>(roundUp(N, MR), 1024);
        const size_t tnMax = std::min<size_t>(roundUp(N, NR), 512);
        for (size_t tm = MR; tm <= tmMax; tm += MR) {
            for (size_t tn = NR; tn <= tnMax; tn += NR) {
                const size_t tasks = ((N + tm - 1) / tm) * ((N + tn - 1) / tn);
                const size_t rounds = (tasks + nThreads - 1) / nThreads;
                const double cost = rounds * double(tm) * double(tn) * (1.0 + 32.0 / tm + 32.0 / tn);
                if (cost < best) { best = cost; TM = tm; TN = tn; }
            }
        }
    }
    const size_t nIB = (N + TM - 1) / TM;
    const size_t nJB = (N + TN - 1) / TN;

    AlignedBuf Bpack(N * Np);

    #pragma omp parallel if (N >= 96)
    {
        // Pack B: for each k-block and NR-wide column panel, a contiguous (kc x NR) slab.
        // Layout: Bp[kb][panel][k][NR], zero-padded in columns.
        #pragma omp for collapse(2) schedule(static)
        for (size_t kb = 0; kb < nKBlocks; ++kb) {
            for (size_t p = 0; p < nPanels; ++p) {
                const size_t k0 = kb * KC;
                const size_t kc = std::min(KC, N - k0);
                double* dst = Bpack.p + k0 * Np + p * kc * NR;
                const size_t j0 = p * NR;
                const size_t nc = std::min(NR, N - j0);
                for (size_t k = 0; k < kc; ++k) {
                    const double* src = &B[(k0 + k) * N + j0];
                    size_t c = 0;
                    for (; c < nc; ++c) dst[k * NR + c] = src[c];
                    for (; c < NR; ++c) dst[k * NR + c] = 0.0;
                }
            }
        }
        // implicit barrier: Bpack complete

        AlignedBuf Apack(std::min(MCI, TM) * KC);
        alignas(64) double edge[MR * NR];

        #pragma omp for collapse(2) schedule(dynamic, 1)
        for (size_t ib = 0; ib < nIB; ++ib) {
            for (size_t jb = 0; jb < nJB; ++jb) {
                const size_t iBeg = ib * TM;
                const size_t iEnd = std::min(iBeg + TM, N);
                const size_t j0 = jb * TN;
                const size_t jEnd = std::min(j0 + TN, N);
                const size_t p0 = j0 / NR;
                const size_t p1 = (jEnd + NR - 1) / NR;

                for (size_t kb = 0; kb < nKBlocks; ++kb) {
                    const size_t k0 = kb * KC;
                    const size_t kc = std::min(KC, N - k0);
                    const bool first = (kb == 0);
                    const bool fmaLast = (N % 2 == 1) && (k0 + kc == N);
                    const double* Bblk = Bpack.p + k0 * Np;

                    for (size_t i0 = iBeg; i0 < iEnd; i0 += MCI) {
                        const size_t mc = std::min(MCI, iEnd - i0);

                        // Pack A block (mc x kc) into MR-row panels: Ap[panel][k][MR], zero-padded rows.
                        for (size_t ir = 0; ir < mc; ir += MR) {
                            const size_t mr = std::min(MR, mc - ir);
                            double* dst = Apack.p + ir * kc;
                            for (size_t k = 0; k < kc; ++k) {
                                size_t r = 0;
                                for (; r < mr; ++r) dst[k * MR + r] = A[(i0 + ir + r) * N + k0 + k];
                                for (; r < MR; ++r) dst[k * MR + r] = 0.0;
                            }
                        }

                        for (size_t p = p0; p < p1; ++p) {
                            const double* Bp = Bblk + p * kc * NR;
                            const size_t j = p * NR;
                            const size_t nc = std::min(NR, N - j);
                            for (size_t ir = 0; ir < mc; ir += MR) {
                                const size_t mr = std::min(MR, mc - ir);
                                const double* Ap = Apack.p + ir * kc;
                                double* Ct = &C[(i0 + ir) * N + j];
                                if (mr == MR && nc == NR) {
                                    microKernel(kc, Ap, Bp, Ct, N, first, fmaLast);
                                } else {
                                    if (!first) {
                                        for (size_t r = 0; r < mr; ++r)
                                            for (size_t c = 0; c < nc; ++c)
                                                edge[r * NR + c] = Ct[r * N + c];
                                    }
                                    microKernel(kc, Ap, Bp, edge, NR, first, fmaLast);
                                    for (size_t r = 0; r < mr; ++r)
                                        for (size_t c = 0; c < nc; ++c)
                                            Ct[r * N + c] = edge[r * NR + c];
                                }
                            }
                        }
                    }
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

// Hardware threads per physical core (SMT width), from Linux sysfs; 1 if unknown.
static int smtWidth() {
    FILE* f = fopen("/sys/devices/system/cpu/cpu0/topology/thread_siblings_list", "r");
    if (!f) return 1;
    char buf[256] = {0};
    const bool ok = fgets(buf, sizeof(buf), f) != nullptr;
    fclose(f);
    if (!ok) return 1;
    // Format: comma-separated items, each "a" or "a-b".
    int count = 0;
    for (char* tok = strtok(buf, ",\n"); tok; tok = strtok(nullptr, ",\n")) {
        int a = 0, b = 0;
        if (sscanf(tok, "%d-%d", &a, &b) == 2) count += b - a + 1;
        else count += 1;
    }
    return count > 0 ? count : 1;
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
    
    // FP-bound kernel: SMT siblings compete for the same FMA/ADD pipes and caches,
    // so default to one thread per physical core unless the user chose explicitly.
    if (!getenv("OMP_NUM_THREADS")) {
        const int smt = smtWidth();
        omp_set_num_threads(std::max(1, omp_get_num_procs() / smt));
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
