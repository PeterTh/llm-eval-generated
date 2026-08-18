#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <omp.h>

#include "../common/results_output.hpp"

// Cache-blocked Cholesky decomposition.  Work on the diagonal is inherently
// ordered, but the panel solves and the (dominant) trailing-matrix updates are
// independent and are distributed over one persistent OpenMP team.

namespace {

constexpr size_t kBlockSize = 64;

bool factorDiagonalBlock(double* A, const size_t n, const size_t begin,
                         const size_t end, size_t& failedDiagonal) {
    for (size_t j = begin; j < end; ++j) {
        double* const rowJ = A + j * n;
        double sum = 0.0;
#pragma omp simd reduction(+ : sum)
        for (size_t p = begin; p < j; ++p) {
            sum += rowJ[p] * rowJ[p];
        }
        const double diagonal = rowJ[j] - sum;
        if (diagonal <= 0.0 || !std::isfinite(diagonal)) {
            failedDiagonal = j;
            return false;
        }

        rowJ[j] = std::sqrt(diagonal);
        const double inverseDiagonal = 1.0 / rowJ[j];
        for (size_t i = j + 1; i < end; ++i) {
            double* const rowI = A + i * n;
            sum = 0.0;
#pragma omp simd reduction(+ : sum)
            for (size_t p = begin; p < j; ++p) {
                sum += rowI[p] * rowJ[p];
            }
            rowI[j] = (rowI[j] - sum) * inverseDiagonal;
        }
    }
    return true;
}

void solvePanelBlock(double* A, const size_t n, const size_t rowBegin,
                     const size_t rowEnd, const size_t columnBegin,
                     const size_t columnEnd) {
    for (size_t i = rowBegin; i < rowEnd; ++i) {
        double* const rowI = A + i * n;
        for (size_t j = columnBegin; j < columnEnd; ++j) {
            const double* const rowJ = A + j * n;
            double sum = 0.0;
#pragma omp simd reduction(+ : sum)
            for (size_t p = columnBegin; p < j; ++p) {
                sum += rowI[p] * rowJ[p];
            }
            rowI[j] = (rowI[j] - sum) / rowJ[j];
        }
    }
}

void updateTrailingBlock(double* A, const size_t n, const size_t rowBegin,
                         const size_t rowEnd, const size_t columnBegin,
                         const size_t columnEnd, const size_t panelBegin,
                         const size_t panelEnd) {
    for (size_t i = rowBegin; i < rowEnd; ++i) {
        double* const rowI = A + i * n;
        const size_t lastColumn = std::min(columnEnd, i + 1);
        size_t j = columnBegin;
        for (; j + 3 < lastColumn; j += 4) {
            const double* const rowJ0 = A + j * n;
            const double* const rowJ1 = rowJ0 + n;
            const double* const rowJ2 = rowJ1 + n;
            const double* const rowJ3 = rowJ2 + n;
            double sum0 = 0.0;
            double sum1 = 0.0;
            double sum2 = 0.0;
            double sum3 = 0.0;
#pragma omp simd reduction(+ : sum0, sum1, sum2, sum3)
            for (size_t p = panelBegin; p < panelEnd; ++p) {
                const double left = rowI[p];
                sum0 += left * rowJ0[p];
                sum1 += left * rowJ1[p];
                sum2 += left * rowJ2[p];
                sum3 += left * rowJ3[p];
            }
            rowI[j] -= sum0;
            rowI[j + 1] -= sum1;
            rowI[j + 2] -= sum2;
            rowI[j + 3] -= sum3;
        }
        for (; j < lastColumn; ++j) {
            const double* const rowJ = A + j * n;
            double sum = 0.0;
#pragma omp simd reduction(+ : sum)
            for (size_t p = panelBegin; p < panelEnd; ++p) {
                sum += rowI[p] * rowJ[p];
            }
            rowI[j] -= sum;
        }
    }
}

}  // namespace

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    if (n == 0) {
        return true;
    }

    double* const data = A.data();
    bool positiveDefinite = true;
    size_t failedDiagonal = std::numeric_limits<size_t>::max();
    const size_t blockCount = (n + kBlockSize - 1) / kBlockSize;
    const size_t availableTileParallelism =
        std::max<size_t>(1, std::min(2 * blockCount,
                                    blockCount * (blockCount - 1) / 2));
    const int teamSize = static_cast<int>(std::min<size_t>(
        static_cast<size_t>(omp_get_max_threads()), availableTileParallelism));

#pragma omp parallel default(none) num_threads(teamSize) \
    shared(data, n, positiveDefinite, failedDiagonal)
    {
        for (size_t blockBegin = 0; blockBegin < n;
             blockBegin += kBlockSize) {
            const size_t blockEnd = std::min(n, blockBegin + kBlockSize);

#pragma omp single
            {
                if (positiveDefinite &&
                    !factorDiagonalBlock(data, n, blockBegin, blockEnd,
                                         failedDiagonal)) {
                    positiveDefinite = false;
                }
            }

            if (positiveDefinite) {
                const size_t panelBlocks =
                    (n - blockEnd + kBlockSize - 1) / kBlockSize;
#pragma omp for schedule(static)
                for (size_t block = 0; block < panelBlocks; ++block) {
                    const size_t rowBegin = blockEnd + block * kBlockSize;
                    solvePanelBlock(data, n, rowBegin,
                                    std::min(n, rowBegin + kBlockSize),
                                    blockBegin, blockEnd);
                }

                const size_t trailingBlocks = panelBlocks * (panelBlocks + 1) / 2;
#pragma omp for schedule(dynamic, 1)
                for (size_t tile = 0; tile < trailingBlocks; ++tile) {
                    // Invert tile = rowBlock * (rowBlock + 1) / 2 + columnBlock.
                    size_t rowBlock = static_cast<size_t>(
                        (std::sqrt(8.0 * static_cast<double>(tile) + 1.0) - 1.0) *
                        0.5);
                    while ((rowBlock + 1) * (rowBlock + 2) / 2 <= tile) {
                        ++rowBlock;
                    }
                    while (rowBlock * (rowBlock + 1) / 2 > tile) {
                        --rowBlock;
                    }
                    const size_t columnBlock =
                        tile - rowBlock * (rowBlock + 1) / 2;
                    const size_t rowBegin = blockEnd + rowBlock * kBlockSize;
                    const size_t columnBegin = blockEnd + columnBlock * kBlockSize;
                    updateTrailingBlock(
                        data, n, rowBegin, std::min(n, rowBegin + kBlockSize),
                        columnBegin, std::min(n, columnBegin + kBlockSize),
                        blockBegin, blockEnd);
                }
            }
        }

#pragma omp for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            std::fill(data + i * n + i + 1, data + (i + 1) * n, 0.0);
        }
    }

    if (!positiveDefinite) {
        printf("Error: Matrix is not positive definite at diagonal element %zu\n",
               failedDiagonal);
    }
    return positiveDefinite;
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
    
    // Compute A = B * B^T
#pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
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
    
    std::vector<double> reconstructed(n * n);
    
    // Compute L * L^T
#pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
#pragma omp simd reduction(+ : sum)
            for (size_t k = 0; k <= std::min(i, j); ++k) {
                sum += L[i * n + k] * L[j * n + k];
            }
            reconstructed[i * n + j] = sum;
        }
    }
    
    // Compare with original
    double maxError = 0.0;
    double relError = 0.0;
    
#pragma omp parallel for reduction(max : maxError, relError) schedule(static)
    for (size_t i = 0; i < n * n; ++i) {
        const double error = fabs(reconstructed[i] - A_orig[i]);
        maxError = std::max(maxError, error);
        
        const double rel = error / (fabs(A_orig[i]) + 1e-10);
        relError = std::max(relError, rel);
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
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (!success) {
        printf("Cholesky decomposition failed\n");
        return 1;
    }
    
    printf("Computation time: %ld ms\n", duration.count());
    
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
