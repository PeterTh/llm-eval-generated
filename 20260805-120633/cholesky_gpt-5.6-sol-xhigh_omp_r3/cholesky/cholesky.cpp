#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

#include <omp.h>

#include "../common/results_output.hpp"

namespace {

// A tile is large enough to amortize OpenMP task overhead while keeping the
// three tiles used by an update in private cache on current multicore CPUs.
constexpr size_t kBlockSize = 96;

inline double dotProduct(const double* const left, const double* const right,
                         const size_t count) {
    size_t p = 0;
    double sum = 0.0;

#if defined(__AVX2__) && defined(__FMA__)
    // Four independent accumulators hide FMA latency.  Unaligned loads are
    // intentional because tile rows need not begin on a vector boundary.
    __m256d sum0 = _mm256_setzero_pd();
    __m256d sum1 = _mm256_setzero_pd();
    __m256d sum2 = _mm256_setzero_pd();
    __m256d sum3 = _mm256_setzero_pd();
    for (; p + 16 <= count; p += 16) {
        sum0 = _mm256_fmadd_pd(_mm256_loadu_pd(left + p),
                               _mm256_loadu_pd(right + p), sum0);
        sum1 = _mm256_fmadd_pd(_mm256_loadu_pd(left + p + 4),
                               _mm256_loadu_pd(right + p + 4), sum1);
        sum2 = _mm256_fmadd_pd(_mm256_loadu_pd(left + p + 8),
                               _mm256_loadu_pd(right + p + 8), sum2);
        sum3 = _mm256_fmadd_pd(_mm256_loadu_pd(left + p + 12),
                               _mm256_loadu_pd(right + p + 12), sum3);
    }

    const __m256d vectorSum =
        _mm256_add_pd(_mm256_add_pd(sum0, sum1), _mm256_add_pd(sum2, sum3));
    const __m128d halves =
        _mm_add_pd(_mm256_castpd256_pd128(vectorSum),
                   _mm256_extractf128_pd(vectorSum, 1));
    sum = _mm_cvtsd_f64(_mm_hadd_pd(halves, halves));
#endif

    const size_t tailBegin = p;
#pragma omp simd reduction(+ : sum)
    for (size_t tail = tailBegin; tail < count; ++tail) {
        sum += left[tail] * right[tail];
    }
    return sum;
}

// Factor the diagonal tile in place.  Earlier block products have already
// been subtracted from it by updateTile().
void factorDiagonalTile(double* const A, const size_t n, const size_t begin,
                        const size_t end,
                        std::atomic<size_t>& failureIndex) {
    if (failureIndex.load(std::memory_order_relaxed) != n) {
        return;
    }

    for (size_t j = begin; j < end; ++j) {
        double* const rowJ = A + j * n;
        const double sum = dotProduct(rowJ + begin, rowJ + begin, j - begin);

        const double value = rowJ[j] - sum;
        if (value <= 0.0 || !std::isfinite(value)) {
            size_t noFailure = n;
            failureIndex.compare_exchange_strong(
                noFailure, j, std::memory_order_relaxed);
            return;
        }

        rowJ[j] = std::sqrt(value);
        const double inverseDiagonal = 1.0 / rowJ[j];

        for (size_t i = j + 1; i < end; ++i) {
            double* const rowI = A + i * n;
            const double dot =
                dotProduct(rowI + begin, rowJ + begin, j - begin);
            rowI[j] = (rowI[j] - dot) * inverseDiagonal;
        }
    }
}

// Solve L(rowBlock, diagonalBlock) * L(diagonalBlock,
// diagonalBlock)^T = A(rowBlock, diagonalBlock).
void solvePanelTile(double* const A, const size_t n, const size_t diagonalBegin,
                    const size_t diagonalEnd, const size_t rowBegin,
                    const size_t rowEnd,
                    const std::atomic<size_t>& failureIndex) {
    if (failureIndex.load(std::memory_order_relaxed) != n) {
        return;
    }

    for (size_t i = rowBegin; i < rowEnd; ++i) {
        double* const rowI = A + i * n;
        for (size_t j = diagonalBegin; j < diagonalEnd; ++j) {
            const double* const rowJ = A + j * n;
            const double dot = dotProduct(rowI + diagonalBegin,
                                          rowJ + diagonalBegin,
                                          j - diagonalBegin);
            rowI[j] = (rowI[j] - dot) / rowJ[j];
        }
    }
}

// A(rowBlock, columnBlock) -= L(rowBlock, diagonalBlock) *
// L(columnBlock, diagonalBlock)^T.  Only the lower half of diagonal tiles is
// touched, so the upper triangle never participates in the factorization.
void updateTile(double* const A, const size_t n, const size_t diagonalBegin,
                const size_t diagonalEnd, const size_t rowBegin,
                const size_t rowEnd, const size_t columnBegin,
                const size_t columnEnd,
                const std::atomic<size_t>& failureIndex) {
    if (failureIndex.load(std::memory_order_relaxed) != n) {
        return;
    }

    const bool diagonalTile = rowBegin == columnBegin;
    for (size_t i = rowBegin; i < rowEnd; ++i) {
        double* const rowI = A + i * n;
        const size_t lastColumn = diagonalTile ? std::min(i + 1, columnEnd)
                                               : columnEnd;
        for (size_t j = columnBegin; j < lastColumn; ++j) {
            const double* const rowJ = A + j * n;
            const double dot = dotProduct(rowI + diagonalBegin,
                                          rowJ + diagonalBegin,
                                          diagonalEnd - diagonalBegin);
            rowI[j] -= dot;
        }
    }
}

}  // namespace

// Blocked, right-looking Cholesky decomposition.  Explicit dependency tokens
// describe the tile dataflow to OpenMP, allowing updates from different block
// columns to execute concurrently while preserving update order for each tile.
bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    if (n == 0) {
        return true;
    }

    double* const matrix = A.data();
    const size_t blockCount = (n + kBlockSize - 1) / kBlockSize;
    std::vector<unsigned char> tokenStorage(blockCount * blockCount);
    [[maybe_unused]] unsigned char* const dependencyTokens =
        tokenStorage.data();
    std::atomic<size_t> failureIndex(n);

#pragma omp parallel
    {
#pragma omp single
        {
            for (size_t diagonalBlock = 0; diagonalBlock < blockCount;
                 ++diagonalBlock) {
                const size_t diagonalBegin = diagonalBlock * kBlockSize;
                const size_t diagonalEnd =
                    std::min(diagonalBegin + kBlockSize, n);
                const size_t diagonalToken =
                    diagonalBlock * blockCount + diagonalBlock;

#pragma omp task firstprivate(diagonalBegin, diagonalEnd, diagonalToken)      \
    shared(matrix, dependencyTokens, failureIndex)                           \
    depend(inout : dependencyTokens[diagonalToken]) priority(2)
                factorDiagonalTile(matrix, n, diagonalBegin, diagonalEnd,
                                   failureIndex);

                for (size_t rowBlock = diagonalBlock + 1;
                     rowBlock < blockCount; ++rowBlock) {
                    const size_t rowBegin = rowBlock * kBlockSize;
                    const size_t rowEnd = std::min(rowBegin + kBlockSize, n);
                    const size_t panelToken =
                        rowBlock * blockCount + diagonalBlock;

#pragma omp task firstprivate(diagonalBegin, diagonalEnd, rowBegin, rowEnd,  \
                              diagonalToken, panelToken)                     \
    shared(matrix, dependencyTokens, failureIndex)                           \
    depend(in : dependencyTokens[diagonalToken])                             \
    depend(inout : dependencyTokens[panelToken]) priority(1)
                    solvePanelTile(matrix, n, diagonalBegin, diagonalEnd,
                                   rowBegin, rowEnd, failureIndex);
                }

                for (size_t rowBlock = diagonalBlock + 1;
                     rowBlock < blockCount; ++rowBlock) {
                    const size_t rowBegin = rowBlock * kBlockSize;
                    const size_t rowEnd = std::min(rowBegin + kBlockSize, n);
                    const size_t rowPanelToken =
                        rowBlock * blockCount + diagonalBlock;

                    for (size_t columnBlock = diagonalBlock + 1;
                         columnBlock <= rowBlock; ++columnBlock) {
                        const size_t columnBegin = columnBlock * kBlockSize;
                        const size_t columnEnd =
                            std::min(columnBegin + kBlockSize, n);
                        const size_t columnPanelToken =
                            columnBlock * blockCount + diagonalBlock;
                        const size_t targetToken =
                            rowBlock * blockCount + columnBlock;

                        if (rowBlock == columnBlock) {
#pragma omp task firstprivate(diagonalBegin, diagonalEnd, rowBegin, rowEnd,  \
                              columnBegin, columnEnd, rowPanelToken,         \
                              targetToken)                                  \
    shared(matrix, dependencyTokens, failureIndex)                           \
    depend(in : dependencyTokens[rowPanelToken])                             \
    depend(inout : dependencyTokens[targetToken])
                            updateTile(matrix, n, diagonalBegin, diagonalEnd,
                                       rowBegin, rowEnd, columnBegin,
                                       columnEnd, failureIndex);
                        } else {
#pragma omp task firstprivate(diagonalBegin, diagonalEnd, rowBegin, rowEnd,  \
                              columnBegin, columnEnd, rowPanelToken,         \
                              columnPanelToken, targetToken)                 \
    shared(matrix, dependencyTokens, failureIndex)                           \
    depend(in : dependencyTokens[rowPanelToken],                             \
           dependencyTokens[columnPanelToken])                              \
    depend(inout : dependencyTokens[targetToken])
                            updateTile(matrix, n, diagonalBegin, diagonalEnd,
                                       rowBegin, rowEnd, columnBegin,
                                       columnEnd, failureIndex);
                        }
                    }
                }
            }
        }
    }

    const size_t badDiagonal = failureIndex.load(std::memory_order_relaxed);
    if (badDiagonal != n) {
        printf("Error: Matrix is not positive definite at diagonal element "
               "%zu\n",
               badDiagonal);
        return false;
    }

    // Zeroing is independent of the numerical factorization and scales well
    // enough to include all threads without adding synchronization to tasks.
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        std::fill(matrix + i * n + i + 1, matrix + (i + 1) * n, 0.0);
    }

    return true;
}

// Generate a symmetric positive definite matrix
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    // Method: Create A = B * B^T where B is random
    // This guarantees A is positive semi-definite
    // Then add identity to make it strictly positive definite
    
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    
    // Generate random matrix B
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }
    
    // Compute A = B * B^T.  Parallelizing by output row distributes the cubic
    // setup work and the writes across the participating threads.
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;

#pragma omp simd reduction(+ : sum)
            for (size_t k = 0; k < n; ++k) {
                sum += B[i * n + k] * B[j * n + k];
            }
            A[i * n + j] = sum;
        }
    }
    
    // Add diagonal dominance to ensure positive definiteness
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        A[i * n + i] += n;
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix
    
    // Reconstruct and compare in one pass.  L is triangular, so entries past
    // min(i,j) are known zeros and need not be read or multiplied.
    double maxError = 0.0;
    double relError = 0.0;

#pragma omp parallel for schedule(static) reduction(max : maxError, relError)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double reconstructed = 0.0;
            const size_t last = std::min(i, j) + 1;

#pragma omp simd reduction(+ : reconstructed)
            for (size_t k = 0; k < last; ++k) {
                reconstructed += L[i * n + k] * L[j * n + k];
            }

            const size_t index = i * n + j;
            const double error = std::fabs(reconstructed - A_orig[index]);
            maxError = std::max(maxError, error);

            const double rel = error / (std::fabs(A_orig[index]) + 1e-10);
            relError = std::max(relError, rel);
        }
    }
    
    printf("Max absolute error: %.10e\n", maxError);
    printf("Max relative error: %.10e\n", relError);
    
    // Check if error is within tolerance
    if (relError > 1e-6) {
        printf("Validation failed: relative error too large\n");
        return false;
    }
    
    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    size_t n = 512;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = atoi(argv[++i]);
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
    
    printf("Cholesky Decomposition Benchmark\n");
    printf("Matrix size: %zu x %zu\n", n, n);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    // Allocate matrix
    std::vector<double> A(n * n);
    std::vector<double> A_orig;
    
    // Generate positive definite matrix
    printf("Generating positive definite matrix...\n");
    generatePositiveDefiniteMatrix(A, n);
    
    if (validate) {
        A_orig = A; // Save original for validation
    }
    
    // Perform Cholesky decomposition
    printf("Computing Cholesky decomposition...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    bool success = choleskyDecomposition(A, n);
    
    auto end = std::chrono::high_resolution_clock::now();
    const auto durationMs =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    const double elapsedSeconds =
        std::chrono::duration<double>(end - start).count();
    
    if (!success) {
        printf("Cholesky decomposition failed\n");
        return 1;
    }
    
    printf("Computation time: %ld ms\n", durationMs.count());
    
    // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
    double ops = (double)n * n * n / 3.0;
    double gflops = ops / elapsedSeconds / 1e9;
    printf("Performance: %.3f GFLOPS\n", gflops);
    
    // Print results for external validation
    if (printResults) {
        print_results(A, "CholeskyL");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateCholesky(A, A_orig, n);
        
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
