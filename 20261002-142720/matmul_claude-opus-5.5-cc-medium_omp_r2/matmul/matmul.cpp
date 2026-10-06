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

// ---------------------------------------------------------------------------
// Blocked, packed, OpenMP-parallel GEMM.
//
// Every C(i,j) is accumulated exactly like the reference loop compiled with
// GCC -O3: starting from 0.0, products are added in increasing k order as
// separately rounded mul + add; when N is odd the final term is fused (FMA),
// mirroring the reference's scalar remainder iteration.  K-blocking only
// spills/reloads the running sum through C, so the results are bit-identical.
// (Built with -ffp-contract=off so mul+add is not silently fused.)
// ---------------------------------------------------------------------------
namespace {

constexpr size_t MR = 6;    // micro-tile rows
constexpr size_t NR = 8;    // micro-tile cols (2 x 4 doubles)
constexpr size_t KC = 256;  // k-block depth (B micro-panel ~16 KB in L1)

#if defined(__AVX2__) && defined(__FMA__)

// c[MR x NR] (row stride ldc) op= sum_k a[k*MR + r] * b[k*NR + c]
static inline void microKernel(const size_t kc, const double* __restrict a,
                               const double* __restrict b, double* __restrict c,
                               const size_t ldc, const bool first, const bool fuseLast) {
    __m256d c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, c50, c51;
    if (first) {
        c00 = c01 = c10 = c11 = c20 = c21 = c30 = c31 = c40 = c41 = c50 = c51 = _mm256_setzero_pd();
    } else {
        c00 = _mm256_loadu_pd(c + 0 * ldc); c01 = _mm256_loadu_pd(c + 0 * ldc + 4);
        c10 = _mm256_loadu_pd(c + 1 * ldc); c11 = _mm256_loadu_pd(c + 1 * ldc + 4);
        c20 = _mm256_loadu_pd(c + 2 * ldc); c21 = _mm256_loadu_pd(c + 2 * ldc + 4);
        c30 = _mm256_loadu_pd(c + 3 * ldc); c31 = _mm256_loadu_pd(c + 3 * ldc + 4);
        c40 = _mm256_loadu_pd(c + 4 * ldc); c41 = _mm256_loadu_pd(c + 4 * ldc + 4);
        c50 = _mm256_loadu_pd(c + 5 * ldc); c51 = _mm256_loadu_pd(c + 5 * ldc + 4);
    }

    const size_t kend = fuseLast ? kc - 1 : kc;
    #pragma GCC unroll 4
    for (size_t k = 0; k < kend; ++k) {
        const __m256d b0 = _mm256_load_pd(b);
        const __m256d b1 = _mm256_load_pd(b + 4);
        __m256d av;
        av = _mm256_broadcast_sd(a + 0);
        c00 = _mm256_add_pd(c00, _mm256_mul_pd(av, b0)); c01 = _mm256_add_pd(c01, _mm256_mul_pd(av, b1));
        av = _mm256_broadcast_sd(a + 1);
        c10 = _mm256_add_pd(c10, _mm256_mul_pd(av, b0)); c11 = _mm256_add_pd(c11, _mm256_mul_pd(av, b1));
        av = _mm256_broadcast_sd(a + 2);
        c20 = _mm256_add_pd(c20, _mm256_mul_pd(av, b0)); c21 = _mm256_add_pd(c21, _mm256_mul_pd(av, b1));
        av = _mm256_broadcast_sd(a + 3);
        c30 = _mm256_add_pd(c30, _mm256_mul_pd(av, b0)); c31 = _mm256_add_pd(c31, _mm256_mul_pd(av, b1));
        av = _mm256_broadcast_sd(a + 4);
        c40 = _mm256_add_pd(c40, _mm256_mul_pd(av, b0)); c41 = _mm256_add_pd(c41, _mm256_mul_pd(av, b1));
        av = _mm256_broadcast_sd(a + 5);
        c50 = _mm256_add_pd(c50, _mm256_mul_pd(av, b0)); c51 = _mm256_add_pd(c51, _mm256_mul_pd(av, b1));
        a += MR;
        b += NR;
    }
    if (fuseLast) {
        const __m256d b0 = _mm256_load_pd(b);
        const __m256d b1 = _mm256_load_pd(b + 4);
        __m256d av;
        av = _mm256_broadcast_sd(a + 0); c00 = _mm256_fmadd_pd(av, b0, c00); c01 = _mm256_fmadd_pd(av, b1, c01);
        av = _mm256_broadcast_sd(a + 1); c10 = _mm256_fmadd_pd(av, b0, c10); c11 = _mm256_fmadd_pd(av, b1, c11);
        av = _mm256_broadcast_sd(a + 2); c20 = _mm256_fmadd_pd(av, b0, c20); c21 = _mm256_fmadd_pd(av, b1, c21);
        av = _mm256_broadcast_sd(a + 3); c30 = _mm256_fmadd_pd(av, b0, c30); c31 = _mm256_fmadd_pd(av, b1, c31);
        av = _mm256_broadcast_sd(a + 4); c40 = _mm256_fmadd_pd(av, b0, c40); c41 = _mm256_fmadd_pd(av, b1, c41);
        av = _mm256_broadcast_sd(a + 5); c50 = _mm256_fmadd_pd(av, b0, c50); c51 = _mm256_fmadd_pd(av, b1, c51);
    }

    _mm256_storeu_pd(c + 0 * ldc, c00); _mm256_storeu_pd(c + 0 * ldc + 4, c01);
    _mm256_storeu_pd(c + 1 * ldc, c10); _mm256_storeu_pd(c + 1 * ldc + 4, c11);
    _mm256_storeu_pd(c + 2 * ldc, c20); _mm256_storeu_pd(c + 2 * ldc + 4, c21);
    _mm256_storeu_pd(c + 3 * ldc, c30); _mm256_storeu_pd(c + 3 * ldc + 4, c31);
    _mm256_storeu_pd(c + 4 * ldc, c40); _mm256_storeu_pd(c + 4 * ldc + 4, c41);
    _mm256_storeu_pd(c + 5 * ldc, c50); _mm256_storeu_pd(c + 5 * ldc + 4, c51);
}
#else
// Portable fallback: same accumulation order as the reference loop.
static inline void microKernel(const size_t kc, const double* __restrict a,
                               const double* __restrict b, double* __restrict c,
                               const size_t ldc, const bool first, const bool fuseLast) {
    double acc[MR][NR];
    for (size_t r = 0; r < MR; ++r)
        for (size_t j = 0; j < NR; ++j) acc[r][j] = first ? 0.0 : c[r * ldc + j];
    const size_t kend = fuseLast ? kc - 1 : kc;
    for (size_t k = 0; k < kend; ++k, a += MR, b += NR)
        for (size_t r = 0; r < MR; ++r)
            for (size_t j = 0; j < NR; ++j) acc[r][j] += a[r] * b[j];
    if (fuseLast)
        for (size_t r = 0; r < MR; ++r)
            for (size_t j = 0; j < NR; ++j) acc[r][j] = std::fma(a[r], b[j], acc[r][j]);
    for (size_t r = 0; r < MR; ++r)
        for (size_t j = 0; j < NR; ++j) c[r * ldc + j] = acc[r][j];
}
#endif

struct AlignedBuffer {
    double* p = nullptr;
    explicit AlignedBuffer(size_t n) {
        p = static_cast<double*>(std::aligned_alloc(64, ((n * sizeof(double) + 63) / 64) * 64));
        if (!p) { fprintf(stderr, "Allocation failed\n"); std::exit(1); }
    }
    ~AlignedBuffer() { std::free(p); }
    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;
};

} // namespace

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B, 
                    std::vector<double>& C, const size_t N) {
    if (N == 0) return;

    const double* __restrict Ad = A.data();
    const double* __restrict Bd = B.data();
    double* __restrict Cd = C.data();

    const size_t nPanels = (N + NR - 1) / NR;
    const bool oddN = (N & 1) != 0;

    // Pack all of B into NR-wide column panels: Bp[p][k][0..NR) (zero padded).
    AlignedBuffer Bp(nPanels * N * NR);
    #pragma omp parallel for schedule(static)
    for (size_t p = 0; p < nPanels; ++p) {
        const size_t j0 = p * NR;
        const size_t nj = std::min(NR, N - j0);
        double* dst = Bp.p + p * N * NR;
        for (size_t k = 0; k < N; ++k) {
            const double* src = Bd + k * N + j0;
            size_t j = 0;
            for (; j < nj; ++j) dst[k * NR + j] = src[j];
            for (; j < NR; ++j) dst[k * NR + j] = 0.0;
        }
    }

    // Choose C tile sizes (MC rows x NC cols) to give enough parallel work.
    const size_t nThreads = static_cast<size_t>(omp_get_max_threads());
    size_t MC = 96, NC = 512;
    auto tiles = [&](size_t mc, size_t nc) { return ((N + mc - 1) / mc) * ((N + nc - 1) / nc); };
    while (tiles(MC, NC) < 2 * nThreads && NC > 64) NC /= 2;
    while (tiles(MC, NC) < 2 * nThreads && MC > 24) MC /= 2;

    const size_t mTiles = (N + MC - 1) / MC;
    const size_t nTiles = (N + NC - 1) / NC;

    #pragma omp parallel
    {
        AlignedBuffer Ap(MC * KC);          // packed A block: [microRow][k][0..MR)
        alignas(64) double edge[MR * NR];   // scratch for partial micro-tiles

        #pragma omp for collapse(2) schedule(dynamic, 1) nowait
        for (size_t it = 0; it < mTiles; ++it) {
            for (size_t jt = 0; jt < nTiles; ++jt) {
                const size_t i0 = it * MC;
                const size_t mc = std::min(MC, N - i0);
                const size_t p0 = (jt * NC) / NR;
                const size_t p1 = std::min(nPanels, ((jt + 1) * NC) / NR);
                const size_t mBlocks = (mc + MR - 1) / MR;

                for (size_t k0 = 0; k0 < N; k0 += KC) {
                    const size_t kc = std::min(KC, N - k0);
                    const bool first = (k0 == 0);
                    const bool fuseLast = oddN && (k0 + kc == N);

                    // Pack A(i0:i0+mc, k0:k0+kc) into MR-row micro-panels.
                    for (size_t mb = 0; mb < mBlocks; ++mb) {
                        double* dst = Ap.p + mb * MR * kc;
                        const size_t r0 = i0 + mb * MR;
                        const size_t nr = std::min(MR, i0 + mc - r0);
                        for (size_t r = 0; r < nr; ++r) {
                            const double* src = Ad + (r0 + r) * N + k0;
                            for (size_t k = 0; k < kc; ++k) dst[k * MR + r] = src[k];
                        }
                        for (size_t r = nr; r < MR; ++r)
                            for (size_t k = 0; k < kc; ++k) dst[k * MR + r] = 0.0;
                    }

                    for (size_t p = p0; p < p1; ++p) {
                        const double* bp = Bp.p + p * N * NR + k0 * NR;
                        const size_t j0 = p * NR;
                        const size_t nj = std::min(NR, N - j0);
                        for (size_t mb = 0; mb < mBlocks; ++mb) {
                            const double* ap = Ap.p + mb * MR * kc;
                            const size_t r0 = i0 + mb * MR;
                            const size_t nr = std::min(MR, i0 + mc - r0);
                            double* cp = Cd + r0 * N + j0;
                            if (nr == MR && nj == NR) {
                                microKernel(kc, ap, bp, cp, N, first, fuseLast);
                            } else {
                                if (!first)
                                    for (size_t r = 0; r < nr; ++r)
                                        for (size_t j = 0; j < nj; ++j) edge[r * NR + j] = cp[r * N + j];
                                microKernel(kc, ap, bp, edge, NR, first, fuseLast);
                                for (size_t r = 0; r < nr; ++r)
                                    for (size_t j = 0; j < nj; ++j) cp[r * N + j] = edge[r * NR + j];
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
