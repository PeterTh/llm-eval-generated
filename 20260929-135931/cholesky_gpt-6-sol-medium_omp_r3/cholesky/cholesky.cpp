#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <omp.h>

#include "../common/results_output.hpp"

// Blocked, right-looking Cholesky decomposition.
// Decomposes positive definite matrix A into L * L^T where L is lower triangular

int workerCount(const size_t n) {
    return static_cast<int>(std::min<size_t>(
        omp_get_max_threads(), std::max<size_t>(1, (n + 31) / 32)));
}

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    constexpr size_t blockSize = 64;
    bool success = true;
    double* const a = A.data();

    // Keep small matrices from launching more workers than useful row chunks.
#pragma omp parallel num_threads(workerCount(n)) shared(success)
    {
        for (size_t k = 0; k < n; k += blockSize) {
            const size_t end = std::min(k + blockSize, n);

            // Previous panels have already updated the diagonal block.
#pragma omp single
            {
                for (size_t j = k; j < end && success; ++j) {
                    double sum = 0.0;
                    for (size_t q = k; q < j; ++q) {
                        sum += a[j * n + q] * a[j * n + q];
                    }
                    const double val = a[j * n + j] - sum;
                    if (val <= 0.0) {
                        printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
                        success = false;
                        break;
                    }
                    a[j * n + j] = std::sqrt(val);
                    for (size_t i = j + 1; i < end; ++i) {
                        double dot = 0.0;
                        for (size_t q = k; q < j; ++q) {
                            dot += a[i * n + q] * a[j * n + q];
                        }
                        a[i * n + j] = (a[i * n + j] - dot) / a[j * n + j];
                    }
                }
            }
            if (!success) break;

            // Each row solve uses only the completed diagonal block.
#pragma omp for schedule(static)
            for (size_t i = end; i < n; ++i) {
                double* const row = a + i * n;
                for (size_t j = k; j < end; ++j) {
                    double dot = 0.0;
                    for (size_t q = k; q < j; ++q) {
                        dot += row[q] * a[j * n + q];
                    }
                    row[j] = (row[j] - dot) / a[j * n + j];
                }
            }

            // Independent rows of the Schur complement can be updated in parallel.
#pragma omp for schedule(dynamic, 8)
            for (size_t i = end; i < n; ++i) {
                double* const row = a + i * n;
                for (size_t j = end; j <= i; ++j) {
                    const double* const other = a + j * n;
                    double dot = 0.0;
#pragma omp simd reduction(+:dot)
                    for (size_t q = k; q < end; ++q) {
                        dot += row[q] * other[q];
                    }
                    row[j] -= dot;
                }
            }
        }

        // Upper triangle is not needed during factorization.
        if (success) {
#pragma omp for schedule(static)
            for (size_t i = 0; i < n; ++i) {
                std::fill(a + i * n + i + 1, a + (i + 1) * n, 0.0);
            }
        }
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
    
    // Compute A = B * B^T
#pragma omp parallel for num_threads(workerCount(n)) schedule(static)
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
    
    // Compute L * L^T
#pragma omp parallel for num_threads(workerCount(n)) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += L[i * n + k] * L[j * n + k];
            }
            reconstructed[i * n + j] = sum;
        }
    }
    
    // Compare with original
    double maxError = 0.0;
    double relError = 0.0;
    
#pragma omp parallel for num_threads(workerCount(n)) reduction(max:maxError,relError) schedule(static)
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
