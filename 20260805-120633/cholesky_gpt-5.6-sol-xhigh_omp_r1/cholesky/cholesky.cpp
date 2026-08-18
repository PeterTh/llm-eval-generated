#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <omp.h>

#include "../common/results_output.hpp"

// The task kernels operate on the lower-triangular tiles of a row-major matrix.
// The two tuned tile sizes keep update operands in private caches while making
// each task large enough to amortize OpenMP scheduling overhead.
namespace {
constexpr size_t kSmallBlockSize = 64;
constexpr size_t kLargeBlockSize = 96;

bool factorDiagonalBlock(double* const a, const size_t n,
                         const size_t begin, const size_t end,
                         size_t& badDiagonal) {
    for (size_t j = begin; j < end; ++j) {
        const double* const rowJ = a + j * n;
        double sum = 0.0;
#pragma omp simd reduction(+ : sum)
        for (size_t p = begin; p < j; ++p) {
            sum += rowJ[p] * rowJ[p];
        }

        const double value = rowJ[j] - sum;
        if (!(value > 0.0)) {
            badDiagonal = j;
            return false;
        }
        a[j * n + j] = std::sqrt(value);

        const double inverseDiagonal = 1.0 / a[j * n + j];
        for (size_t i = j + 1; i < end; ++i) {
            double* const rowI = a + i * n;
            double product = 0.0;
#pragma omp simd reduction(+ : product)
            for (size_t p = begin; p < j; ++p) {
                product += rowI[p] * rowJ[p];
            }
            rowI[j] = (rowI[j] - product) * inverseDiagonal;
        }
    }
    return true;
}

void solvePanelBlock(double* const a, const size_t n,
                     const size_t rowBegin, const size_t rowEnd,
                     const size_t panelBegin, const size_t panelEnd) {
    for (size_t i = rowBegin; i < rowEnd; ++i) {
        double* const rowI = a + i * n;
        for (size_t j = panelBegin; j < panelEnd; ++j) {
            const double* const rowJ = a + j * n;
            double product = 0.0;
#pragma omp simd reduction(+ : product)
            for (size_t p = panelBegin; p < j; ++p) {
                product += rowI[p] * rowJ[p];
            }
            rowI[j] = (rowI[j] - product) / rowJ[j];
        }
    }
}

void updateTrailingBlock(double* const a, const size_t n,
                         const size_t rowBegin, const size_t rowEnd,
                         const size_t columnBegin, const size_t columnEnd,
                         const size_t panelBegin, const size_t panelEnd) {
    const bool diagonalBlock = rowBegin == columnBegin;
    for (size_t i = rowBegin; i < rowEnd; ++i) {
        double* const rowI = a + i * n;
        const size_t lastColumn = diagonalBlock ? std::min(columnEnd, i + 1)
                                                : columnEnd;
        size_t j = columnBegin;

        // Four simultaneous dot products reuse rowI and give the compiler
        // enough independent accumulators to hide FMA latency.
        for (; j + 3 < lastColumn; j += 4) {
            const double* const rowJ0 = a + j * n;
            const double* const rowJ1 = rowJ0 + n;
            const double* const rowJ2 = rowJ1 + n;
            const double* const rowJ3 = rowJ2 + n;
            double product0 = 0.0;
            double product1 = 0.0;
            double product2 = 0.0;
            double product3 = 0.0;
#pragma omp simd reduction(+ : product0, product1, product2, product3)
            for (size_t p = panelBegin; p < panelEnd; ++p) {
                const double left = rowI[p];
                product0 += left * rowJ0[p];
                product1 += left * rowJ1[p];
                product2 += left * rowJ2[p];
                product3 += left * rowJ3[p];
            }
            rowI[j] -= product0;
            rowI[j + 1] -= product1;
            rowI[j + 2] -= product2;
            rowI[j + 3] -= product3;
        }
        for (; j < lastColumn; ++j) {
            const double* const rowJ = a + j * n;
            double product = 0.0;
#pragma omp simd reduction(+ : product)
            for (size_t p = panelBegin; p < panelEnd; ++p) {
                product += rowI[p] * rowJ[p];
            }
            rowI[j] -= product;
        }
    }
}
}  // namespace

// In-place blocked Cholesky decomposition. OpenMP dependencies express the
// factor/solve/update DAG, so updates to separate tiles execute concurrently
// and completed tiles can immediately feed the next panel (look-ahead).
bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    if (n == 0) {
        return true;
    }

    const size_t blockSize = n <= 1024 ? kSmallBlockSize : kLargeBlockSize;
    const size_t blockCount = (n + blockSize - 1) / blockSize;
    std::vector<unsigned char> dependencyTokens(blockCount * blockCount);
    unsigned char* const token = dependencyTokens.data();
    static_cast<void>(token);  // Referenced by OpenMP depend clauses.
    double* const a = A.data();
    std::atomic<size_t> badDiagonal{std::numeric_limits<size_t>::max()};

    // Avoid excessive startup and task-scheduling overhead on small matrices.
    // Three workers per tile row retains the useful task-DAG concurrency seen
    // in the update wavefront without starting an unproductive large team.
    const size_t firstUpdateTasks = blockCount > 1
                                        ? blockCount * (blockCount - 1) / 2
                                        : 1;
    const size_t usefulWorkers = blockCount > 1
                                     ? std::min(firstUpdateTasks,
                                                3 * (blockCount - 1))
                                     : 1;
    const int teamSize = std::max(1, std::min<int>(
        omp_get_max_threads(), static_cast<int>(std::min<size_t>(
                                   usefulWorkers,
                                   static_cast<size_t>(std::numeric_limits<int>::max())))));

#pragma omp parallel num_threads(teamSize) proc_bind(close) shared(a, token, badDiagonal)
    {
#pragma omp single
        {
            for (size_t kb = 0; kb < blockCount; ++kb) {
                const size_t panelBegin = kb * blockSize;
                const size_t panelEnd = std::min(n, panelBegin + blockSize);
                const size_t diagonalToken = kb * blockCount + kb;

#pragma omp task firstprivate(panelBegin, panelEnd, diagonalToken) depend(inout : token[diagonalToken])
                {
                    if (badDiagonal.load(std::memory_order_relaxed) ==
                        std::numeric_limits<size_t>::max()) {
                        size_t bad = std::numeric_limits<size_t>::max();
                        if (!factorDiagonalBlock(a, n, panelBegin, panelEnd, bad)) {
                            size_t expected = std::numeric_limits<size_t>::max();
                            badDiagonal.compare_exchange_strong(
                                expected, bad, std::memory_order_relaxed);
                        }
                    }
                }

                // Triangular solves produce the tiles directly below the panel.
                for (size_t ib = kb + 1; ib < blockCount; ++ib) {
                    const size_t rowBegin = ib * blockSize;
                    const size_t rowEnd = std::min(n, rowBegin + blockSize);
                    const size_t panelToken = ib * blockCount + kb;
#pragma omp task firstprivate(rowBegin, rowEnd, panelBegin, panelEnd, diagonalToken, panelToken) depend(in : token[diagonalToken]) depend(inout : token[panelToken])
                    {
                        if (badDiagonal.load(std::memory_order_relaxed) ==
                            std::numeric_limits<size_t>::max()) {
                            solvePanelBlock(a, n, rowBegin, rowEnd,
                                            panelBegin, panelEnd);
                        }
                    }
                }

                // Rank-k updates to different lower-triangular tiles are fully
                // independent. Inout dependencies serialize successive panels
                // that update the same destination tile.
                for (size_t jb = kb + 1; jb < blockCount; ++jb) {
                    const size_t columnBegin = jb * blockSize;
                    const size_t columnEnd = std::min(n, columnBegin + blockSize);
                    const size_t rightPanelToken = jb * blockCount + kb;
                    for (size_t ib = jb; ib < blockCount; ++ib) {
                        const size_t rowBegin = ib * blockSize;
                        const size_t rowEnd = std::min(n, rowBegin + blockSize);
                        const size_t leftPanelToken = ib * blockCount + kb;
                        const size_t destinationToken = ib * blockCount + jb;

                        if (ib == jb) {
#pragma omp task firstprivate(rowBegin, rowEnd, columnBegin, columnEnd, panelBegin, panelEnd, leftPanelToken, destinationToken) depend(in : token[leftPanelToken]) depend(inout : token[destinationToken])
                            {
                                if (badDiagonal.load(std::memory_order_relaxed) ==
                                    std::numeric_limits<size_t>::max()) {
                                    updateTrailingBlock(a, n, rowBegin, rowEnd,
                                                        columnBegin, columnEnd,
                                                        panelBegin, panelEnd);
                                }
                            }
                        } else {
#pragma omp task firstprivate(rowBegin, rowEnd, columnBegin, columnEnd, panelBegin, panelEnd, leftPanelToken, rightPanelToken, destinationToken) depend(in : token[leftPanelToken], token[rightPanelToken]) depend(inout : token[destinationToken])
                            {
                                if (badDiagonal.load(std::memory_order_relaxed) ==
                                    std::numeric_limits<size_t>::max()) {
                                    updateTrailingBlock(a, n, rowBegin, rowEnd,
                                                        columnBegin, columnEnd,
                                                        panelBegin, panelEnd);
                                }
                            }
                        }
                    }
                }
            }
#pragma omp taskwait
        }

        // Only the lower triangle is factorized; preserve the original output
        // contract by clearing the upper triangle in parallel.
#pragma omp for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            std::fill(a + i * n + i + 1, a + (i + 1) * n, 0.0);
        }
    }

    const size_t bad = badDiagonal.load(std::memory_order_relaxed);
    if (bad != std::numeric_limits<size_t>::max()) {
        printf("Error: Matrix is not positive definite at diagonal element %zu\n", bad);
        return false;
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
    
    // Compute one triangle of A = B * B^T and mirror it. The seeded input and
    // reduction are deterministic while the expensive O(n^3) work is parallel.
#pragma omp parallel for schedule(dynamic, 1) proc_bind(close)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double sum = 0.0;
#pragma omp simd reduction(+ : sum)
            for (size_t k = 0; k < n; ++k) {
                sum += B[i * n + k] * B[j * n + k];
            }
            A[i * n + j] = sum;
            A[j * n + i] = sum;
        }
    }
    
    // Add diagonal dominance to ensure positive definiteness
#pragma omp parallel for schedule(static) proc_bind(close)
    for (size_t i = 0; i < n; ++i) {
        A[i * n + i] += n;
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix
    double maxError = 0.0;
    double relError = 0.0;

    // L*L^T and A are symmetric, so validating one triangle is sufficient.
    // The reduction avoids a second n-by-n temporary matrix.
#pragma omp parallel for schedule(dynamic, 1) proc_bind(close) reduction(max : maxError, relError)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double reconstructed = 0.0;
#pragma omp simd reduction(+ : reconstructed)
            for (size_t k = 0; k <= j; ++k) {
                reconstructed += L[i * n + k] * L[j * n + k];
            }

            const double error = std::fabs(reconstructed - A_orig[i * n + j]);
            maxError = std::max(maxError, error);
            const double rel = error / (std::fabs(A_orig[i * n + j]) + 1e-10);
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

    // The task graph needs a fixed team; do not let runtime dynamic adjustment
    // silently reduce the requested OpenMP parallelism.
    omp_set_dynamic(0);
    
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
    const std::chrono::duration<double, std::milli> duration = end - start;
    
    if (!success) {
        printf("Cholesky decomposition failed\n");
        return 1;
    }
    
    printf("Computation time: %.3f ms\n", duration.count());
    
    // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
    double ops = (double)n * n * n / 3.0;
    double gflops = ops / (duration.count() / 1000.0) / 1e9;
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
