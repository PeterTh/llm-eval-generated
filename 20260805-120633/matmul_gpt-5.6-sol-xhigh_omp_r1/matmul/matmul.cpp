#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
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
    const std::int64_t rows = static_cast<std::int64_t>(N);

#pragma omp parallel for schedule(static)
    for (std::int64_t row = 0; row < rows; ++row) {
        const size_t i = static_cast<size_t>(row);
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

// This fixed-width micro-kernel keeps all partial sums in SIMD registers and
// reuses every B vector across six output rows.
inline void multiply6x8(const double* __restrict a, const double* __restrict b,
                        double* __restrict c, const size_t N,
                        const size_t bStride, const size_t i, const size_t j) {
#if defined(__AVX2__) && defined(__FMA__)
    __m256d sum00 = _mm256_setzero_pd();
    __m256d sum01 = _mm256_setzero_pd();
    __m256d sum10 = _mm256_setzero_pd();
    __m256d sum11 = _mm256_setzero_pd();
    __m256d sum20 = _mm256_setzero_pd();
    __m256d sum21 = _mm256_setzero_pd();
    __m256d sum30 = _mm256_setzero_pd();
    __m256d sum31 = _mm256_setzero_pd();
    __m256d sum40 = _mm256_setzero_pd();
    __m256d sum41 = _mm256_setzero_pd();
    __m256d sum50 = _mm256_setzero_pd();
    __m256d sum51 = _mm256_setzero_pd();

    for (size_t k = 0; k < N; ++k) {
        const double* const bValues = b + k * bStride;
        const __m256d b0 = _mm256_loadu_pd(bValues);
        const __m256d b1 = _mm256_loadu_pd(bValues + 4);

        const __m256d a0 = _mm256_broadcast_sd(a + (i + 0) * N + k);
        sum00 = _mm256_fmadd_pd(a0, b0, sum00);
        sum01 = _mm256_fmadd_pd(a0, b1, sum01);
        const __m256d a1 = _mm256_broadcast_sd(a + (i + 1) * N + k);
        sum10 = _mm256_fmadd_pd(a1, b0, sum10);
        sum11 = _mm256_fmadd_pd(a1, b1, sum11);
        const __m256d a2 = _mm256_broadcast_sd(a + (i + 2) * N + k);
        sum20 = _mm256_fmadd_pd(a2, b0, sum20);
        sum21 = _mm256_fmadd_pd(a2, b1, sum21);
        const __m256d a3 = _mm256_broadcast_sd(a + (i + 3) * N + k);
        sum30 = _mm256_fmadd_pd(a3, b0, sum30);
        sum31 = _mm256_fmadd_pd(a3, b1, sum31);
        const __m256d a4 = _mm256_broadcast_sd(a + (i + 4) * N + k);
        sum40 = _mm256_fmadd_pd(a4, b0, sum40);
        sum41 = _mm256_fmadd_pd(a4, b1, sum41);
        const __m256d a5 = _mm256_broadcast_sd(a + (i + 5) * N + k);
        sum50 = _mm256_fmadd_pd(a5, b0, sum50);
        sum51 = _mm256_fmadd_pd(a5, b1, sum51);
    }

    _mm256_storeu_pd(c + (i + 0) * N + j, sum00);
    _mm256_storeu_pd(c + (i + 0) * N + j + 4, sum01);
    _mm256_storeu_pd(c + (i + 1) * N + j, sum10);
    _mm256_storeu_pd(c + (i + 1) * N + j + 4, sum11);
    _mm256_storeu_pd(c + (i + 2) * N + j, sum20);
    _mm256_storeu_pd(c + (i + 2) * N + j + 4, sum21);
    _mm256_storeu_pd(c + (i + 3) * N + j, sum30);
    _mm256_storeu_pd(c + (i + 3) * N + j + 4, sum31);
    _mm256_storeu_pd(c + (i + 4) * N + j, sum40);
    _mm256_storeu_pd(c + (i + 4) * N + j + 4, sum41);
    _mm256_storeu_pd(c + (i + 5) * N + j, sum50);
    _mm256_storeu_pd(c + (i + 5) * N + j + 4, sum51);
#else
    constexpr size_t width = 8;
    alignas(64) double sum0[width] = {};
    alignas(64) double sum1[width] = {};
    alignas(64) double sum2[width] = {};
    alignas(64) double sum3[width] = {};
    alignas(64) double sum4[width] = {};
    alignas(64) double sum5[width] = {};

    for (size_t k = 0; k < N; ++k) {
        const double* const bValues = b + k * bStride;
        const double a0 = a[(i + 0) * N + k];
        const double a1 = a[(i + 1) * N + k];
        const double a2 = a[(i + 2) * N + k];
        const double a3 = a[(i + 3) * N + k];
        const double a4 = a[(i + 4) * N + k];
        const double a5 = a[(i + 5) * N + k];

#pragma omp simd
        for (size_t x = 0; x < width; ++x) {
            const double bValue = bValues[x];
            sum0[x] += a0 * bValue;
            sum1[x] += a1 * bValue;
            sum2[x] += a2 * bValue;
            sum3[x] += a3 * bValue;
            sum4[x] += a4 * bValue;
            sum5[x] += a5 * bValue;
        }
    }

#pragma omp simd
    for (size_t x = 0; x < width; ++x) {
        c[(i + 0) * N + j + x] = sum0[x];
        c[(i + 1) * N + j + x] = sum1[x];
        c[(i + 2) * N + j + x] = sum2[x];
        c[(i + 3) * N + j + x] = sum3[x];
        c[(i + 4) * N + j + x] = sum4[x];
        c[(i + 5) * N + j + x] = sum5[x];
    }
#endif
}

inline void multiply1x8(const double* __restrict a, const double* __restrict b,
                        double* __restrict c, const size_t N,
                        const size_t bStride, const size_t i, const size_t j) {
    constexpr size_t width = 8;
    alignas(64) double sums[width] = {};

    for (size_t k = 0; k < N; ++k) {
        const double aValue = a[i * N + k];
        const double* const bValues = b + k * bStride;

#pragma omp simd
        for (size_t x = 0; x < width; ++x) {
            sums[x] += aValue * bValues[x];
        }
    }

#pragma omp simd
    for (size_t x = 0; x < width; ++x) {
        c[i * N + j + x] = sums[x];
    }
}

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, const size_t N) {
    // Each thread owns complete output tiles, eliminating synchronization in
    // the hot path and exposing enough independent work to scale across cores.
    constexpr size_t blockRows = 18;
    constexpr size_t blockCols = 64;
    constexpr size_t microCols = 8;

    const double* __restrict a = A.data();
    const double* __restrict b = B.data();
    double* __restrict c = C.data();
    const size_t panelCount = (N + microCols - 1) / microCols;
    const bool usePackedB = omp_get_max_threads() <= 40;
    std::vector<double> packedB(usePackedB ? panelCount * N * microCols : 0);
    double* __restrict packed = packedB.data();

#pragma omp parallel
    {
        // Panel-major storage turns strided reads into sequential streams at
        // modest thread counts. Large teams use B directly: their aggregate
        // hardware prefetching is faster than a separate packing pass.
        if (usePackedB) {
#pragma omp for collapse(2) schedule(static)
            for (size_t panel = 0; panel < panelCount; ++panel) {
                for (size_t k = 0; k < N; ++k) {
                    const size_t firstCol = panel * microCols;
                    const size_t width = std::min(microCols, N - firstCol);
                    const double* const source = b + k * N + firstCol;
                    double* const destination = packed + (panel * N + k) * microCols;
#pragma omp simd
                    for (size_t x = 0; x < width; ++x) {
                        destination[x] = source[x];
                    }
                }
            }
        }

#pragma omp for collapse(2) schedule(static)
        for (size_t rowBlock = 0; rowBlock < N; rowBlock += blockRows) {
            for (size_t colBlock = 0; colBlock < N; colBlock += blockCols) {
                const size_t rowEnd = std::min(rowBlock + blockRows, N);
                const size_t colEnd = std::min(colBlock + blockCols, N);
                size_t j = colBlock;

                // Reuse one packed B panel across all rows before advancing
                // to the next panel. A row tile and one B panel fit in L2.
                for (; j + microCols <= colEnd; j += microCols) {
                    const double* const bPanel =
                        usePackedB ? packed + (j / microCols) * N * microCols
                                   : b + j;
                    const size_t bStride = usePackedB ? microCols : N;
                    size_t i = rowBlock;
                    for (; i + 5 < rowEnd; i += 6) {
                        multiply6x8(a, bPanel, c, N, bStride, i, j);
                    }
                    for (; i < rowEnd; ++i) {
                        multiply1x8(a, bPanel, c, N, bStride, i, j);
                    }
                }

                // Only non-multiple-of-eight matrix widths use this edge path.
                for (; j < colEnd; ++j) {
                    for (size_t i = rowBlock; i < rowEnd; ++i) {
                        double sum = 0.0;
                        for (size_t k = 0; k < N; ++k) {
                            sum += a[i * N + k] * b[k * N + j];
                        }
                        c[i * N + j] = sum;
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
