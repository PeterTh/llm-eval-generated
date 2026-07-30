#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <omp.h>

#include "../common/results_output.hpp"

// Parallel Cholesky decomposition (right-looking, column-oriented algorithm)
// Decomposes positive definite matrix A into L * L^T where L is lower triangular
//
// Parallelization strategy: the outer column loop runs sequentially (each column
// depends on all previous columns). Within each column, off-diagonal elements
// L[i][j] for i > j are independent and computed in parallel by all threads. A
// single #pragma omp parallel region is created outside the column loop to
// eliminate thread creation overhead across column iterations.

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    const ptrdiff_t n_signed = static_cast<ptrdiff_t>(n);
    bool success = true;

    #pragma omp parallel
    {
        for (ptrdiff_t j = 0; j < n_signed && success; ++j) {
            // Diagonal element for column j (computed by one thread only)
            #pragma omp single
            {
                double sum_diag = 0.0;
                for (ptrdiff_t k = 0; k < j; ++k) {
                    sum_diag += A[static_cast<size_t>(j) * n + static_cast<size_t>(k)]
                              * A[static_cast<size_t>(j) * n + static_cast<size_t>(k)];
                }
                const double val = A[static_cast<size_t>(j) * n + static_cast<size_t>(j)] - sum_diag;
                if (val <= 0.0) {
                    #pragma omp atomic write
                    success = false;
                } else {
                    A[static_cast<size_t>(j) * n + static_cast<size_t>(j)] = sqrt(val);
                }

                // Zero out upper triangular part (row j, columns > j)
                for (ptrdiff_t k = j + 1; k < n_signed; ++k) {
                    A[static_cast<size_t>(j) * n + static_cast<size_t>(k)] = 0.0;
                }
            }

            // Compute off-diagonal elements for column j (all threads in parallel)
            if (success) {
                #pragma omp for
                for (ptrdiff_t i = j + 1; i < n_signed; ++i) {
                    double sum = 0.0;
                    for (ptrdiff_t k = 0; k < j; ++k) {
                        sum += A[static_cast<size_t>(i) * n + static_cast<size_t>(k)]
                             * A[static_cast<size_t>(j) * n + static_cast<size_t>(k)];
                    }
                    A[static_cast<size_t>(i) * n + static_cast<size_t>(j)] =
                        (A[static_cast<size_t>(i) * n + static_cast<size_t>(j)] - sum)
                        / A[static_cast<size_t>(j) * n + static_cast<size_t>(j)];
                }
            }
        }
    }

    if (!success) {
        printf("Error: Matrix is not positive definite\n");
    }
    return success;
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
    
    // Compute A = B * B^T in parallel
    const ptrdiff_t n_signed = static_cast<ptrdiff_t>(n);
    #pragma omp parallel for
    for (ptrdiff_t i = 0; i < n_signed; ++i) {
        for (ptrdiff_t j = 0; j < n_signed; ++j) {
            double sum = 0.0;
            for (ptrdiff_t k = 0; k < n_signed; ++k) {
                sum += B[static_cast<size_t>(i) * n + static_cast<size_t>(k)]
                     * B[static_cast<size_t>(j) * n + static_cast<size_t>(k)];
            }
            A[static_cast<size_t>(i) * n + static_cast<size_t>(j)] = sum;
        }
    }
    
    // Add diagonal dominance to ensure positive definiteness (in parallel)
    #pragma omp parallel for
    for (ptrdiff_t i = 0; i < n_signed; ++i) {
        A[static_cast<size_t>(i) * n + static_cast<size_t>(i)] += static_cast<double>(n);
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix
    
    std::vector<double> reconstructed(n * n);
    
    // Compute L * L^T in parallel
    const ptrdiff_t n_signed = static_cast<ptrdiff_t>(n);
    #pragma omp parallel for
    for (ptrdiff_t i = 0; i < n_signed; ++i) {
        for (ptrdiff_t j = 0; j < n_signed; ++j) {
            double sum = 0.0;
            for (ptrdiff_t k = 0; k < n_signed; ++k) {
                sum += L[static_cast<size_t>(i) * n + static_cast<size_t>(k)]
                     * L[static_cast<size_t>(j) * n + static_cast<size_t>(k)];
            }
            reconstructed[static_cast<size_t>(i) * n + static_cast<size_t>(j)] = sum;
        }
    }
    
    // Compare with original (in parallel)
    double maxError = 0.0;
    double relError = 0.0;
    
    const ptrdiff_t nn_signed = static_cast<ptrdiff_t>(n * n);
    #pragma omp parallel for reduction(max: maxError, relError)
    for (ptrdiff_t idx = 0; idx < nn_signed; ++idx) {
        const double error = fabs(reconstructed[static_cast<size_t>(idx)] - A_orig[static_cast<size_t>(idx)]);
        if (error > maxError) maxError = error;
        const double rel = error / (fabs(A_orig[static_cast<size_t>(idx)]) + 1e-10);
        if (rel > relError) relError = rel;
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
