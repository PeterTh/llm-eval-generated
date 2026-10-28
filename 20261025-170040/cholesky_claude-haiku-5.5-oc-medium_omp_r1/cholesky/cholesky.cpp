#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// Cholesky decomposition, OpenMP-parallel (column-oriented / left-looking).
// Decomposes positive definite matrix A into L * L^T where L is lower triangular.
//
// For each column j:
//   1. L[j][j] = sqrt(A[j][j] - sum_{k<j} L[j][k]^2)            (serial, O(j))
//   2. L[i][j] = (A[i][j] - sum_{k<j} L[i][k] * L[j][k]) / L[j][j]
//      for all i > j                                              (parallel)
// Every entry is accumulated over k in the same increasing order as the
// original row-oriented algorithm (no reassociation). Rows are processed in
// blocks of 4 to give the CPU independent accumulation chains (ILP); this may
// change the compiler's FMA contraction, so results can differ from the
// sequential original by floating-point rounding only.

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    // A is stored in row-major order
    double* const a = A.data();
    bool ok = true;

#pragma omp parallel shared(ok)
    {
        for (size_t j = 0; j < n; ++j) {
            const double* const Lj = a + j * n;

            // Diagonal element (serial; barrier at end of single)
#pragma omp single
            {
                double sum = 0.0;
                for (size_t k = 0; k < j; ++k) {
                    sum += Lj[k] * Lj[k];
                }
                const double val = a[j * n + j] - sum;
                if (val <= 0.0) {
                    // Matrix is not positive definite
                    printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
                    ok = false;
                } else {
                    a[j * n + j] = sqrt(val);
                }
            }
            // All threads observe ok consistently after the implicit barrier
            if (!ok) {
                break;
            }

            const double Ljj = Lj[j];
            const size_t first = j + 1;
            const size_t count = n - first;
            const size_t full = count / 4;          // number of 4-row blocks
            const size_t items = full + (count % 4); // blocks + leftover single rows

            // Off-diagonal elements of column j, rows first..n-1
#pragma omp for schedule(static)
            for (size_t t = 0; t < items; ++t) {
                if (t < full) {
                    const size_t i0 = first + 4 * t;
                    const double* const r0 = a + (i0 + 0) * n;
                    const double* const r1 = a + (i0 + 1) * n;
                    const double* const r2 = a + (i0 + 2) * n;
                    const double* const r3 = a + (i0 + 3) * n;
                    double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
                    for (size_t k = 0; k < j; ++k) {
                        const double w = Lj[k];
                        s0 += r0[k] * w;
                        s1 += r1[k] * w;
                        s2 += r2[k] * w;
                        s3 += r3[k] * w;
                    }
                    a[(i0 + 0) * n + j] = (a[(i0 + 0) * n + j] - s0) / Ljj;
                    a[(i0 + 1) * n + j] = (a[(i0 + 1) * n + j] - s1) / Ljj;
                    a[(i0 + 2) * n + j] = (a[(i0 + 2) * n + j] - s2) / Ljj;
                    a[(i0 + 3) * n + j] = (a[(i0 + 3) * n + j] - s3) / Ljj;
                } else {
                    const size_t i = first + 4 * full + (t - full);
                    const double* const r = a + i * n;
                    double sum = 0.0;
                    for (size_t k = 0; k < j; ++k) {
                        sum += r[k] * Lj[k];
                    }
                    a[i * n + j] = (a[i * n + j] - sum) / Ljj;
                }
            }
        }

        if (ok) {
            // Zero out upper triangular part (never read by the algorithm)
#pragma omp for schedule(static)
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = i + 1; j < n; ++j) {
                    a[i * n + j] = 0.0;
                }
            }
        }
    }

    return ok;
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
    
    // Compute A = B * B^T (rows are independent)
#pragma omp parallel for schedule(static)
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
    for (size_t i = 0; i < n; ++i) {
        A[i * n + i] += n;
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix
    
    std::vector<double> reconstructed(n * n);
    
    // Compute L * L^T (rows are independent)
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += L[i * n + k] * L[j * n + k];
            }
            reconstructed[i * n + j] = sum;
        }
    }
    
    // Compare with original (max is order-independent)
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
