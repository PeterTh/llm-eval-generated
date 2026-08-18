#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

#include "../common/results_output.hpp"

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

void initMatrix(double* const mat, const size_t N) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

// Compute one complete 6x8 output micro-tile.  AVX2 builds use twelve independent
// vector accumulators, enough to hide FMA latency without spilling registers.
// The CMake configuration enables this path on capable hosts via -march=native.
inline void multiply6x8(const double* __restrict a, const double* __restrict b,
                        double* __restrict c, const size_t N, const size_t ii,
                        const size_t jj, const size_t kBegin, const size_t kEnd) {
#if defined(__AVX2__) && defined(__FMA__)
    double* const c0 = c + (ii + 0) * N + jj;
    double* const c1 = c + (ii + 1) * N + jj;
    double* const c2 = c + (ii + 2) * N + jj;
    double* const c3 = c + (ii + 3) * N + jj;
    double* const c4 = c + (ii + 4) * N + jj;
    double* const c5 = c + (ii + 5) * N + jj;

    __m256d c00;
    __m256d c01;
    __m256d c10;
    __m256d c11;
    __m256d c20;
    __m256d c21;
    __m256d c30;
    __m256d c31;
    __m256d c40;
    __m256d c41;
    __m256d c50;
    __m256d c51;

    if (kBegin == 0) {
        c00 = c01 = c10 = c11 = _mm256_setzero_pd();
        c20 = c21 = c30 = c31 = _mm256_setzero_pd();
        c40 = c41 = c50 = c51 = _mm256_setzero_pd();
    } else {
        c00 = _mm256_loadu_pd(c0 + 0);
        c01 = _mm256_loadu_pd(c0 + 4);
        c10 = _mm256_loadu_pd(c1 + 0);
        c11 = _mm256_loadu_pd(c1 + 4);
        c20 = _mm256_loadu_pd(c2 + 0);
        c21 = _mm256_loadu_pd(c2 + 4);
        c30 = _mm256_loadu_pd(c3 + 0);
        c31 = _mm256_loadu_pd(c3 + 4);
        c40 = _mm256_loadu_pd(c4 + 0);
        c41 = _mm256_loadu_pd(c4 + 4);
        c50 = _mm256_loadu_pd(c5 + 0);
        c51 = _mm256_loadu_pd(c5 + 4);
    }

    for (size_t k = kBegin; k < kEnd; ++k) {
        const double* const bRow = b + k * N + jj;
        const __m256d b0 = _mm256_loadu_pd(bRow + 0);
        const __m256d b1 = _mm256_loadu_pd(bRow + 4);

        const __m256d a0 = _mm256_broadcast_sd(a + (ii + 0) * N + k);
        c00 = _mm256_fmadd_pd(a0, b0, c00);
        c01 = _mm256_fmadd_pd(a0, b1, c01);

        const __m256d a1 = _mm256_broadcast_sd(a + (ii + 1) * N + k);
        c10 = _mm256_fmadd_pd(a1, b0, c10);
        c11 = _mm256_fmadd_pd(a1, b1, c11);

        const __m256d a2 = _mm256_broadcast_sd(a + (ii + 2) * N + k);
        c20 = _mm256_fmadd_pd(a2, b0, c20);
        c21 = _mm256_fmadd_pd(a2, b1, c21);

        const __m256d a3 = _mm256_broadcast_sd(a + (ii + 3) * N + k);
        c30 = _mm256_fmadd_pd(a3, b0, c30);
        c31 = _mm256_fmadd_pd(a3, b1, c31);

        const __m256d a4 = _mm256_broadcast_sd(a + (ii + 4) * N + k);
        c40 = _mm256_fmadd_pd(a4, b0, c40);
        c41 = _mm256_fmadd_pd(a4, b1, c41);

        const __m256d a5 = _mm256_broadcast_sd(a + (ii + 5) * N + k);
        c50 = _mm256_fmadd_pd(a5, b0, c50);
        c51 = _mm256_fmadd_pd(a5, b1, c51);
    }

    _mm256_storeu_pd(c0 + 0, c00);
    _mm256_storeu_pd(c0 + 4, c01);
    _mm256_storeu_pd(c1 + 0, c10);
    _mm256_storeu_pd(c1 + 4, c11);
    _mm256_storeu_pd(c2 + 0, c20);
    _mm256_storeu_pd(c2 + 4, c21);
    _mm256_storeu_pd(c3 + 0, c30);
    _mm256_storeu_pd(c3 + 4, c31);
    _mm256_storeu_pd(c4 + 0, c40);
    _mm256_storeu_pd(c4 + 4, c41);
    _mm256_storeu_pd(c5 + 0, c50);
    _mm256_storeu_pd(c5 + 4, c51);
#else
    alignas(64) double sums[6][8];
    for (size_t i = 0; i < 6; ++i) {
        const double* const cRow = c + (ii + i) * N + jj;
        #pragma omp simd
        for (size_t j = 0; j < 8; ++j) {
            sums[i][j] = (kBegin == 0) ? 0.0 : cRow[j];
        }
    }

    for (size_t k = kBegin; k < kEnd; ++k) {
        const double* const bRow = b + k * N + jj;
        for (size_t i = 0; i < 6; ++i) {
            const double aValue = a[(ii + i) * N + k];
            #pragma omp simd
            for (size_t j = 0; j < 8; ++j) {
                sums[i][j] += aValue * bRow[j];
            }
        }
    }

    for (size_t i = 0; i < 6; ++i) {
        double* const cRow = c + (ii + i) * N + jj;
        #pragma omp simd
        for (size_t j = 0; j < 8; ++j) {
            cRow[j] = sums[i][j];
        }
    }
#endif
}

void matrixMultiply(const double* __restrict A, const double* __restrict B,
                    double* __restrict C, const size_t N) {
    // Six rows share every loaded segment of B.  Keeping eight columns of
    // each row in local accumulators also avoids repeatedly loading and storing
    // partial sums in the innermost loop.
    constexpr size_t rowBlock = 6;
    constexpr size_t columnBlock = 8;
    constexpr size_t tileRows = 18;
    constexpr size_t tileColumns = 64;
    constexpr size_t tileDepth = 128;

    const double* __restrict a = A;
    const double* __restrict b = B;
    double* __restrict c = C;

    // Each collapsed-loop iteration owns a disjoint tile of C.  Depth blocking
    // keeps the active portions of A and B cache resident, while a static
    // schedule keeps adjacent tiles together and gives deterministic results.
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t tileI = 0; tileI < N; tileI += tileRows) {
        for (size_t tileJ = 0; tileJ < N; tileJ += tileColumns) {
            const size_t tileIEnd = (N - tileI < tileRows) ? N : tileI + tileRows;
            const size_t tileJEnd = (N - tileJ < tileColumns) ? N : tileJ + tileColumns;

            for (size_t tileK = 0; tileK < N; tileK += tileDepth) {
                const size_t tileKEnd = (N - tileK < tileDepth) ? N : tileK + tileDepth;

                for (size_t ii = tileI; ii < tileIEnd; ii += rowBlock) {
                    const size_t rows = (tileIEnd - ii < rowBlock) ? tileIEnd - ii : rowBlock;

                    for (size_t jj = tileJ; jj < tileJEnd; jj += columnBlock) {
                        const size_t columns = (tileJEnd - jj < columnBlock)
                                                   ? tileJEnd - jj
                                                   : columnBlock;
                        if (rows == rowBlock && columns == columnBlock) {
                            multiply6x8(a, b, c, N, ii, jj, tileK, tileKEnd);
                        } else {
                            // Handle dimensions which are not multiples of the
                            // register block sizes.
                            alignas(64) double sums[rowBlock][columnBlock];
                            for (size_t i = 0; i < rows; ++i) {
                                const double* const cRow = c + (ii + i) * N + jj;
                                #pragma omp simd
                                for (size_t j = 0; j < columns; ++j) {
                                    sums[i][j] = (tileK == 0) ? 0.0 : cRow[j];
                                }
                            }

                            for (size_t k = tileK; k < tileKEnd; ++k) {
                                const double* const bRow = b + k * N + jj;
                                for (size_t i = 0; i < rows; ++i) {
                                    const double aValue = a[(ii + i) * N + k];
                                    #pragma omp simd
                                    for (size_t j = 0; j < columns; ++j) {
                                        sums[i][j] += aValue * bRow[j];
                                    }
                                }
                            }

                            for (size_t i = 0; i < rows; ++i) {
                                double* const cRow = c + (ii + i) * N + jj;
                                #pragma omp simd
                                for (size_t j = 0; j < columns; ++j) {
                                    cRow[j] = sums[i][j];
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
bool validateResult(const double* const A, const double* const B,
                    const double* const C, const size_t N) {
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
    
    // Avoid serial zero filling here: parallel initialization and output tiling
    // provide first-touch NUMA placement on multisocket systems.
    const size_t elementCount = N * N;
    auto A = std::make_unique_for_overwrite<double[]>(elementCount);
    auto B = std::make_unique_for_overwrite<double[]>(elementCount);
    auto C = std::make_unique_for_overwrite<double[]>(elementCount);
    
    // Initialize matrices
    printf("Initializing matrices...\n");
    initMatrix(A.get(), N);
    initMatrix(B.get(), N);
    
    // Perform matrix multiplication
    printf("Computing matrix multiplication...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    matrixMultiply(A.get(), B.get(), C.get(), N);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate GFLOPS
    double gflops = (2.0 * N * N * N) / (duration.count() / 1000.0) / 1e9;
    printf("Performance: %.3f GFLOPS\n", gflops);
    
    // Print results for external validation
    if (printResults) {
        const std::vector<double> results(C.get(), C.get() + elementCount);
        print_results(results, "MatrixC");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(A.get(), B.get(), C.get(), N);
        
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
