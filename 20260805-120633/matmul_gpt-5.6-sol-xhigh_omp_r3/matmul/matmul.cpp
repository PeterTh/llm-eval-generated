#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif
#include <omp.h>

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

namespace {

// Four rows reuse every cache line fetched from B four times.  The column
// width gives the compiler eight independent vector accumulators, which is
// enough to hide FMA latency without spilling registers.
constexpr size_t rowBlock = 4;
constexpr size_t maxRowCacheBlock = 16;
#if defined(__AVX512F__)
constexpr size_t columnBlock = 16;
#else
constexpr size_t columnBlock = 8;
#endif

inline void multiplyFullTile(const double* const A, const double* const B,
                             double* const C, const size_t N,
                             const size_t row, const size_t column) noexcept {
    const double* const bPanel = B + (column / columnBlock) * N * columnBlock;
#if defined(__AVX512F__)
    __m512d sum00 = _mm512_setzero_pd();
    __m512d sum01 = _mm512_setzero_pd();
    __m512d sum10 = _mm512_setzero_pd();
    __m512d sum11 = _mm512_setzero_pd();
    __m512d sum20 = _mm512_setzero_pd();
    __m512d sum21 = _mm512_setzero_pd();
    __m512d sum30 = _mm512_setzero_pd();
    __m512d sum31 = _mm512_setzero_pd();

    for (size_t k = 0; k < N; ++k) {
        __m512d b0 = _mm512_loadu_pd(bPanel + k * columnBlock);
        __m512d b1 = _mm512_loadu_pd(bPanel + k * columnBlock + 8);
        // Keep these shared operands in registers.  Without this constraint,
        // some compilers fold the same load into all four row FMAs.
        __asm__ volatile("" : "+v"(b0), "+v"(b1));
        const __m512d a0 = _mm512_set1_pd(A[(row + 0) * N + k]);
        const __m512d a1 = _mm512_set1_pd(A[(row + 1) * N + k]);
        const __m512d a2 = _mm512_set1_pd(A[(row + 2) * N + k]);
        const __m512d a3 = _mm512_set1_pd(A[(row + 3) * N + k]);

        sum00 = _mm512_fmadd_pd(a0, b0, sum00);
        sum01 = _mm512_fmadd_pd(a0, b1, sum01);
        sum10 = _mm512_fmadd_pd(a1, b0, sum10);
        sum11 = _mm512_fmadd_pd(a1, b1, sum11);
        sum20 = _mm512_fmadd_pd(a2, b0, sum20);
        sum21 = _mm512_fmadd_pd(a2, b1, sum21);
        sum30 = _mm512_fmadd_pd(a3, b0, sum30);
        sum31 = _mm512_fmadd_pd(a3, b1, sum31);
    }

    _mm512_storeu_pd(C + (row + 0) * N + column, sum00);
    _mm512_storeu_pd(C + (row + 0) * N + column + 8, sum01);
    _mm512_storeu_pd(C + (row + 1) * N + column, sum10);
    _mm512_storeu_pd(C + (row + 1) * N + column + 8, sum11);
    _mm512_storeu_pd(C + (row + 2) * N + column, sum20);
    _mm512_storeu_pd(C + (row + 2) * N + column + 8, sum21);
    _mm512_storeu_pd(C + (row + 3) * N + column, sum30);
    _mm512_storeu_pd(C + (row + 3) * N + column + 8, sum31);
#elif defined(__AVX2__) && defined(__FMA__)
    __m256d sum00 = _mm256_setzero_pd();
    __m256d sum01 = _mm256_setzero_pd();
    __m256d sum10 = _mm256_setzero_pd();
    __m256d sum11 = _mm256_setzero_pd();
    __m256d sum20 = _mm256_setzero_pd();
    __m256d sum21 = _mm256_setzero_pd();
    __m256d sum30 = _mm256_setzero_pd();
    __m256d sum31 = _mm256_setzero_pd();

    for (size_t k = 0; k < N; ++k) {
        __m256d b0 = _mm256_loadu_pd(bPanel + k * columnBlock);
        __m256d b1 = _mm256_loadu_pd(bPanel + k * columnBlock + 4);
        __asm__ volatile("" : "+x"(b0), "+x"(b1));
        const __m256d a0 = _mm256_broadcast_sd(A + (row + 0) * N + k);
        const __m256d a1 = _mm256_broadcast_sd(A + (row + 1) * N + k);
        const __m256d a2 = _mm256_broadcast_sd(A + (row + 2) * N + k);
        const __m256d a3 = _mm256_broadcast_sd(A + (row + 3) * N + k);

        sum00 = _mm256_fmadd_pd(a0, b0, sum00);
        sum01 = _mm256_fmadd_pd(a0, b1, sum01);
        sum10 = _mm256_fmadd_pd(a1, b0, sum10);
        sum11 = _mm256_fmadd_pd(a1, b1, sum11);
        sum20 = _mm256_fmadd_pd(a2, b0, sum20);
        sum21 = _mm256_fmadd_pd(a2, b1, sum21);
        sum30 = _mm256_fmadd_pd(a3, b0, sum30);
        sum31 = _mm256_fmadd_pd(a3, b1, sum31);
    }

    _mm256_storeu_pd(C + (row + 0) * N + column, sum00);
    _mm256_storeu_pd(C + (row + 0) * N + column + 4, sum01);
    _mm256_storeu_pd(C + (row + 1) * N + column, sum10);
    _mm256_storeu_pd(C + (row + 1) * N + column + 4, sum11);
    _mm256_storeu_pd(C + (row + 2) * N + column, sum20);
    _mm256_storeu_pd(C + (row + 2) * N + column + 4, sum21);
    _mm256_storeu_pd(C + (row + 3) * N + column, sum30);
    _mm256_storeu_pd(C + (row + 3) * N + column + 4, sum31);
#else
    alignas(64) double sums[rowBlock][columnBlock] = {};

    // k remains strictly increasing, as in the original implementation.
    // SIMD is across independent columns, so it does not reassociate any dot
    // product and results remain deterministic for every OpenMP thread count.
    for (size_t k = 0; k < N; ++k) {
        const double* const b = bPanel + k * columnBlock;
        const double a0 = A[(row + 0) * N + k];
        const double a1 = A[(row + 1) * N + k];
        const double a2 = A[(row + 2) * N + k];
        const double a3 = A[(row + 3) * N + k];

#pragma omp simd
        for (size_t j = 0; j < columnBlock; ++j) {
            const double bValue = b[j];
            sums[0][j] += a0 * bValue;
            sums[1][j] += a1 * bValue;
            sums[2][j] += a2 * bValue;
            sums[3][j] += a3 * bValue;
        }
    }

    for (size_t i = 0; i < rowBlock; ++i) {
#pragma omp simd
        for (size_t j = 0; j < columnBlock; ++j) {
            C[(row + i) * N + column + j] = sums[i][j];
        }
    }
#endif
}

inline void multiplyPartialTile(const double* const A, const double* const B,
                                double* const C, const size_t N,
                                const size_t row, const size_t rows,
                                const size_t column,
                                const size_t columns) noexcept {
    alignas(64) double sums[rowBlock][columnBlock] = {};

    for (size_t k = 0; k < N; ++k) {
        const double* const b = B + k * N + column;
        for (size_t i = 0; i < rows; ++i) {
            const double aValue = A[(row + i) * N + k];
#pragma omp simd
            for (size_t j = 0; j < columns; ++j) {
                sums[i][j] += aValue * b[j];
            }
        }
    }

    for (size_t i = 0; i < rows; ++i) {
#pragma omp simd
        for (size_t j = 0; j < columns; ++j) {
            C[(row + i) * N + column + j] = sums[i][j];
        }
    }
}

}  // namespace

void matrixMultiply(const double* const a, const double* const b,
                    double* const c, const size_t N) {
    const size_t fullRows = N - N % rowBlock;
    const size_t fullColumns = N - N % columnBlock;
    const size_t threadCount = static_cast<size_t>(omp_get_max_threads());
    const size_t rowCacheBlock =
        fullRows / maxRowCacheBlock >= threadCount
            ? maxRowCacheBlock
            : (fullRows / (2 * rowBlock) >= threadCount ? 2 * rowBlock
                                                         : rowBlock);
    auto packedB = std::make_unique_for_overwrite<double[]>(N * fullColumns);

    // One persistent team packs B and computes C without repeatedly entering
    // parallel regions.  Panel-major packing makes the hot kernel's B reads
    // contiguous while preserving the original values and arithmetic order.
#pragma omp parallel
    {
#pragma omp for schedule(static)
        for (size_t j = 0; j < fullColumns; j += columnBlock) {
            double* const destination =
                packedB.get() + (j / columnBlock) * N * columnBlock;
            for (size_t k = 0; k < N; ++k) {
#pragma omp simd
                for (size_t offset = 0; offset < columnBlock; ++offset) {
                    destination[k * columnBlock + offset] =
                        b[k * N + j + offset];
                }
            }
        }

        // A cache-sized row slab remains hot while it visits all column
        // panels.  Within a panel, several row micro-tiles reuse the same B
        // data from private cache.  Threads own disjoint slabs, eliminating
        // races and false sharing in the main multiplication.
#pragma omp for schedule(static) nowait
        for (size_t rowStart = 0; rowStart < fullRows;
             rowStart += rowCacheBlock) {
            const size_t rowEnd =
                std::min(rowStart + rowCacheBlock, fullRows);
            for (size_t j = 0; j < fullColumns; j += columnBlock) {
                for (size_t i = rowStart; i < rowEnd; i += rowBlock) {
                    multiplyFullTile(a, packedB.get(), c, N, i, j);
                }
            }
            if (fullColumns < N) {
                for (size_t i = rowStart; i < rowEnd; i += rowBlock) {
                    multiplyPartialTile(a, b, c, N, i, rowBlock,
                                        fullColumns, N - fullColumns);
                }
            }
        }

        // At most three rows take this path.  Keeping them in the same team
        // handles arbitrary matrix sizes without a serial cleanup phase.
#pragma omp for schedule(static)
        for (size_t i = fullRows; i < N; ++i) {
            for (size_t j = 0; j < fullColumns; j += columnBlock) {
                multiplyPartialTile(a, b, c, N, i, 1, j, columnBlock);
            }
            if (fullColumns < N) {
                multiplyPartialTile(a, b, c, N, i, 1, fullColumns,
                                    N - fullColumns);
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
    
    // Leave pages untouched until the OpenMP workers initialize or compute
    // them.  This first-touch placement avoids pinning all large matrices to
    // the NUMA node on which the main thread happened to allocate them.
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
        std::vector<double> result;
        if (elementCount != 0) {
            result.assign(C.get(), C.get() + elementCount);
        }
        print_results(result, "MatrixC");
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
