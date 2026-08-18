#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <omp.h>
#include <vector>

#include "../common/results_output.hpp"

// Decomposes positive definite matrix A into L * L^T where L is lower triangular

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    // Blocking exposes a large parallel rank-k update and substantially improves
    // cache reuse.  Each thread owns complete destination rows, so no atomics or
    // reductions are needed in the numerical kernel.
    constexpr size_t blockSize = 64;
    int failed = 0;
    size_t failedAt = 0;

#pragma omp parallel shared(failed, failedAt)
    {
        for (size_t begin = 0; begin < n; begin += blockSize) {
            const size_t end = std::min(begin + blockSize, n);

#pragma omp single
            {
                if (!failed) {
                    // Factor the (already updated) diagonal block.
                    for (size_t j = begin; j < end; ++j) {
                        double diagonal = A[j * n + j];
#pragma omp simd reduction(- : diagonal)
                        for (size_t p = begin; p < j; ++p)
                            diagonal -= A[j * n + p] * A[j * n + p];
                        if (!(diagonal > 0.0)) {
                            failed = 1;
                            failedAt = j;
                            break;
                        }
                        A[j * n + j] = std::sqrt(diagonal);
                        const double inverseDiagonal = 1.0 / A[j * n + j];
                        for (size_t i = j + 1; i < end; ++i) {
                            double value = A[i * n + j];
#pragma omp simd reduction(- : value)
                            for (size_t p = begin; p < j; ++p)
                                value -= A[i * n + p] * A[j * n + p];
                            A[i * n + j] = value * inverseDiagonal;
                        }
                    }
                }
            }

            // Block triangular solve: L21 = A21 * inv(L11^T).
#pragma omp for schedule(static)
            for (size_t i = end; i < n; ++i) {
                if (!failed) {
                    double* const row = A.data() + i * n;
                    for (size_t j = begin; j < end; ++j) {
                        double value = row[j];
#pragma omp simd reduction(- : value)
                        for (size_t p = begin; p < j; ++p)
                            value -= row[p] * A[j * n + p];
                        row[j] = value / A[j * n + j];
                    }
                }
            }

            // Symmetric trailing update; only the lower triangle is required.
#pragma omp for schedule(static)
            for (size_t i = end; i < n; ++i) {
                if (!failed) {
                    double* const rowI = A.data() + i * n;
                    for (size_t j = end; j <= i; ++j) {
                        const double* const rowJ = A.data() + j * n;
                        double value = rowI[j];
#pragma omp simd reduction(- : value)
                        for (size_t p = begin; p < end; ++p)
                            value -= rowI[p] * rowJ[p];
                        rowI[j] = value;
                    }
                }
            }
        }

#pragma omp for schedule(static)
        for (size_t i = 0; i < n; ++i)
            std::fill(A.begin() + i * n + i + 1, A.begin() + (i + 1) * n, 0.0);
    }

    if (failed)
        printf("Error: Matrix is not positive definite at diagonal element %zu\n", failedAt);
    return !failed;
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
    for (size_t i = 0; i < n; ++i) {
        A[i * n + i] += n;
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix
    
    double maxError = 0.0;
    double relError = 0.0;

#pragma omp parallel for schedule(static) reduction(max : maxError, relError)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            const size_t limit = std::min(i, j) + 1;
#pragma omp simd reduction(+ : sum)
            for (size_t k = 0; k < limit; ++k)
                sum += L[i * n + k] * L[j * n + k];
            const double error = std::fabs(sum - A_orig[i * n + j]);
            maxError = std::max(maxError, error);
            relError = std::max(relError,
                                error / (std::fabs(A_orig[i * n + j]) + 1e-10));
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
