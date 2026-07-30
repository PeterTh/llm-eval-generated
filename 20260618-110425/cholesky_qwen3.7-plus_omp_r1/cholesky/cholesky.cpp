#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <omp.h>

#include "../common/results_output.hpp"

// Block size for blocked Cholesky decomposition
static const size_t BLOCK_SIZE = 64;

// Unblocked right-looking Cholesky for small diagonal blocks
static bool cholesky_unblocked(double* A, size_t lda, size_t n) {
    for (size_t j = 0; j < n; ++j) {
        for (size_t k = 0; k < j; ++k) {
            double Ljk = A[j * lda + k];
            for (size_t i = j; i < n; ++i) {
                A[i * lda + j] -= A[i * lda + k] * Ljk;
            }
        }
        double diag = A[j * lda + j];
        if (diag <= 0.0) {
            return false;
        }
        double inv_sqrt = 1.0 / sqrt(diag);
        for (size_t i = j; i < n; ++i) {
            A[i * lda + j] *= inv_sqrt;
        }
    }
    return true;
}

// Blocked right-looking Cholesky decomposition with OpenMP parallelism
// Decomposes positive definite matrix A into L * L^T where L is lower triangular
bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    double* data = A.data();

    for (size_t K = 0; K < n; K += BLOCK_SIZE) {
        size_t nb = std::min(BLOCK_SIZE, n - K);

        // 1. Factor diagonal block (sequential, small)
        if (!cholesky_unblocked(data + K * n + K, n, nb)) {
            printf("Error: Matrix is not positive definite at diagonal element near %zu\n", K);
            return false;
        }

        size_t m2 = n - K - nb;
        if (m2 == 0) continue;

        // 2. Triangular solve: L21 = A21 * inv(L11^T), parallelized over rows
        #pragma omp parallel for schedule(static)
        for (size_t i = K + nb; i < n; ++i) {
            for (size_t j = 0; j < nb; ++j) {
                double sum = 0.0;
                for (size_t k = 0; k < j; ++k) {
                    sum += data[i * n + K + k] * data[(K + j) * n + K + k];
                }
                data[i * n + K + j] = (data[i * n + K + j] - sum) / data[(K + j) * n + K + j];
            }
        }

        // 3. Symmetric rank-k update: A22 -= L21 * L21^T, parallelized over rows
        #pragma omp parallel for schedule(static)
        for (size_t i = K + nb; i < n; ++i) {
            for (size_t j = K + nb; j <= i; ++j) {
                double sum = 0.0;
                for (size_t k = 0; k < nb; ++k) {
                    sum += data[i * n + K + k] * data[j * n + K + k];
                }
                data[i * n + j] -= sum;
            }
        }
    }

    // Zero out upper triangular part
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            data[i * n + j] = 0.0;
        }
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

    // Generate random matrix B (sequential for reproducibility)
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

    // Compute A = B * B^T (parallelized over rows)
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

    // Compute L * L^T (parallelized over rows)
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

    // Compare with original (parallel reduction for max error)
    double maxError = 0.0;
    double relError = 0.0;

    #pragma omp parallel for reduction(max:maxError,relError) schedule(static)
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
