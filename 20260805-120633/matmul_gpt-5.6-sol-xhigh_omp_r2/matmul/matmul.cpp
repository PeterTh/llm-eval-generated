#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <omp.h>
#include <vector>

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
#pragma omp simd
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B, 
                    std::vector<double>& C, const size_t N) {
#if defined(__AVX2__) && defined(__FMA__)
    // A 4x12 register tile exposes enough independent work to keep the vector
    // pipelines busy without repeatedly loading and storing partial sums.
    constexpr size_t rowBlock = 4;
    constexpr size_t columnBlock = 12;
    const size_t vectorizedRows = N - N % rowBlock;
    const size_t vectorizedColumns = N - N % columnBlock;

    const double* __restrict__ a = A.data();
    const double* __restrict__ b = B.data();
    double* __restrict__ c = C.data();

#pragma omp parallel
    {
#pragma omp for schedule(static) nowait
        for (size_t row = 0; row < vectorizedRows; row += rowBlock) {
            for (size_t column = 0; column < vectorizedColumns; column += columnBlock) {
                __m256d c00 = _mm256_setzero_pd();
                __m256d c01 = _mm256_setzero_pd();
                __m256d c02 = _mm256_setzero_pd();
                __m256d c10 = _mm256_setzero_pd();
                __m256d c11 = _mm256_setzero_pd();
                __m256d c12 = _mm256_setzero_pd();
                __m256d c20 = _mm256_setzero_pd();
                __m256d c21 = _mm256_setzero_pd();
                __m256d c22 = _mm256_setzero_pd();
                __m256d c30 = _mm256_setzero_pd();
                __m256d c31 = _mm256_setzero_pd();
                __m256d c32 = _mm256_setzero_pd();

                for (size_t k = 0; k < N; ++k) {
                    const double* const bValues = b + k * N + column;
                    const __m256d b0 = _mm256_loadu_pd(bValues);
                    const __m256d b1 = _mm256_loadu_pd(bValues + 4);
                    const __m256d b2 = _mm256_loadu_pd(bValues + 8);

                    const __m256d a0 = _mm256_broadcast_sd(a + row * N + k);
                    c00 = _mm256_fmadd_pd(a0, b0, c00);
                    c01 = _mm256_fmadd_pd(a0, b1, c01);
                    c02 = _mm256_fmadd_pd(a0, b2, c02);

                    const __m256d a1 = _mm256_broadcast_sd(a + (row + 1) * N + k);
                    c10 = _mm256_fmadd_pd(a1, b0, c10);
                    c11 = _mm256_fmadd_pd(a1, b1, c11);
                    c12 = _mm256_fmadd_pd(a1, b2, c12);

                    const __m256d a2 = _mm256_broadcast_sd(a + (row + 2) * N + k);
                    c20 = _mm256_fmadd_pd(a2, b0, c20);
                    c21 = _mm256_fmadd_pd(a2, b1, c21);
                    c22 = _mm256_fmadd_pd(a2, b2, c22);

                    const __m256d a3 = _mm256_broadcast_sd(a + (row + 3) * N + k);
                    c30 = _mm256_fmadd_pd(a3, b0, c30);
                    c31 = _mm256_fmadd_pd(a3, b1, c31);
                    c32 = _mm256_fmadd_pd(a3, b2, c32);
                }

                _mm256_storeu_pd(c + row * N + column, c00);
                _mm256_storeu_pd(c + row * N + column + 4, c01);
                _mm256_storeu_pd(c + row * N + column + 8, c02);
                _mm256_storeu_pd(c + (row + 1) * N + column, c10);
                _mm256_storeu_pd(c + (row + 1) * N + column + 4, c11);
                _mm256_storeu_pd(c + (row + 1) * N + column + 8, c12);
                _mm256_storeu_pd(c + (row + 2) * N + column, c20);
                _mm256_storeu_pd(c + (row + 2) * N + column + 4, c21);
                _mm256_storeu_pd(c + (row + 2) * N + column + 8, c22);
                _mm256_storeu_pd(c + (row + 3) * N + column, c30);
                _mm256_storeu_pd(c + (row + 3) * N + column + 4, c31);
                _mm256_storeu_pd(c + (row + 3) * N + column + 8, c32);
            }

            // At most eleven columns remain after the register tiles.  Keep
            // these columns contiguous as well so the compiler can vectorize.
            for (size_t i = row; i < row + rowBlock; ++i) {
                double* const cRow = c + i * N;
#pragma omp simd
                for (size_t j = vectorizedColumns; j < N; ++j) {
                    cRow[j] = 0.0;
                }
            }
            for (size_t k = 0; k < N; ++k) {
                const double* const bRow = b + k * N;
                for (size_t i = row; i < row + rowBlock; ++i) {
                    const double aValue = a[i * N + k];
                    double* const cRow = c + i * N;
#pragma omp simd
                    for (size_t j = vectorizedColumns; j < N; ++j) {
                        cRow[j] += aValue * bRow[j];
                    }
                }
            }
        }

        // Fewer than four rows remain.  They still use contiguous, SIMD-sized
        // column tiles instead of falling back to strided dot products.
#pragma omp for schedule(static)
        for (size_t i = vectorizedRows; i < N; ++i) {
            double* const cRow = c + i * N;
            constexpr size_t remainderColumnBlock = 128;
            for (size_t column = 0; column < N; column += remainderColumnBlock) {
                const size_t columnEnd = std::min(column + remainderColumnBlock, N);
#pragma omp simd
                for (size_t j = column; j < columnEnd; ++j) {
                    cRow[j] = 0.0;
                }
                for (size_t k = 0; k < N; ++k) {
                    const double aValue = a[i * N + k];
                    const double* const bRow = b + k * N;
#pragma omp simd
                    for (size_t j = column; j < columnEnd; ++j) {
                        cRow[j] += aValue * bRow[j];
                    }
                }
            }
        }
    }
#else
    // Keep one output tile private to a thread.  The K dimension remains in
    // increasing order, while the innermost, contiguous loop can use SIMD.
    // The three tiles fit comfortably in a typical private L2 cache.
    constexpr size_t maxRowBlock = 16;
    constexpr size_t columnBlock = 128;
    constexpr size_t depthBlock = 128;

    const size_t threadCount = static_cast<size_t>(omp_get_max_threads());
    const size_t rowsPerThread = (N + threadCount - 1) / threadCount;
    const size_t rowBlock = std::max<size_t>(1, std::min(maxRowBlock, rowsPerThread));

    const double* __restrict__ a = A.data();
    const double* __restrict__ b = B.data();
    double* __restrict__ c = C.data();

#pragma omp parallel for schedule(static)
    for (size_t rowBegin = 0; rowBegin < N; rowBegin += rowBlock) {
        const size_t rowEnd = std::min(rowBegin + rowBlock, N);

        for (size_t columnBegin = 0; columnBegin < N; columnBegin += columnBlock) {
            const size_t columnEnd = std::min(columnBegin + columnBlock, N);

            for (size_t i = rowBegin; i < rowEnd; ++i) {
                double* const cRow = c + i * N;
#pragma omp simd
                for (size_t j = columnBegin; j < columnEnd; ++j) {
                    cRow[j] = 0.0;
                }
            }

            for (size_t depthBegin = 0; depthBegin < N; depthBegin += depthBlock) {
                const size_t depthEnd = std::min(depthBegin + depthBlock, N);

                // Reuse each contiguous row of B across every row in the
                // thread-private output tile before moving to the next row.
                for (size_t k = depthBegin; k < depthEnd; ++k) {
                    const double* const bRow = b + k * N;

                    for (size_t i = rowBegin; i < rowEnd; ++i) {
                        const double aValue = a[i * N + k];
                        double* const cRow = c + i * N;

#pragma omp simd
                        for (size_t j = columnBegin; j < columnEnd; ++j) {
                            cRow[j] += aValue * bRow[j];
                        }
                    }
                }
            }
        }
    }
#endif
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
