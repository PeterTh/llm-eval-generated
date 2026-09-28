#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <omp.h>

#include "../common/results_output.hpp"

// Blocked right-looking Cholesky decomposition, parallelized with OpenMP.
// Decomposes positive definite matrix A into L * L^T where L is lower triangular.

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    // A is stored in row-major order
    constexpr size_t B = 128; // block (panel) width

    for (size_t k0 = 0; k0 < n; k0 += B) {
        const size_t kend = std::min(k0 + B, n);

        // 1) Factor the diagonal block A[k0:kend, k0:kend] (small, sequential)
        for (size_t j = k0; j < kend; ++j) {
            double sum = 0.0;
            for (size_t k = k0; k < j; ++k) {
                sum += A[j * n + k] * A[j * n + k];
            }
            const double val = A[j * n + j] - sum;
            if (val <= 0.0) {
                // Matrix is not positive definite
                printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
                return false;
            }
            const double diag = sqrt(val);
            A[j * n + j] = diag;

            for (size_t i = j + 1; i < kend; ++i) {
                double s = 0.0;
                for (size_t k = k0; k < j; ++k) {
                    s += A[i * n + k] * A[j * n + k];
                }
                A[i * n + j] = (A[i * n + j] - s) / diag;
            }
        }

        // 2) Triangular solve for the panel below the diagonal block:
        //    L[i, k0:kend] = A[i, k0:kend] * inv(L_diag^T), parallel over rows
        #pragma omp parallel for schedule(static)
        for (size_t i = kend; i < n; ++i) {
            for (size_t j = k0; j < kend; ++j) {
                double sum = 0.0;
                for (size_t k = k0; k < j; ++k) {
                    sum += A[i * n + k] * A[j * n + k];
                }
                A[i * n + j] = (A[i * n + j] - sum) / A[j * n + j];
            }
        }

        // 3) Rank-B update of the trailing submatrix (lower triangle only):
        //    A[i, j] -= dot(L[i, k0:kend], L[j, k0:kend])
        //    Parallelized over tile pairs for load balance and cache locality.
        const size_t m = (n - kend + B - 1) / B;      // number of tile rows
        const size_t numPairs = m * (m + 1) / 2;       // lower-triangular tile pairs

        #pragma omp parallel for schedule(dynamic)
        for (size_t p = 0; p < numPairs; ++p) {
            // Map linear index p to tile coordinates (ti, tj) with tj <= ti
            size_t ti = (size_t)((std::sqrt(8.0 * (double)p + 1.0) - 1.0) / 2.0);
            while (ti * (ti + 1) / 2 > p) --ti;
            while ((ti + 1) * (ti + 2) / 2 <= p) ++ti;
            const size_t tj = p - ti * (ti + 1) / 2;

            const size_t i0 = kend + ti * B;
            const size_t j0 = kend + tj * B;
            const size_t iend = std::min(i0 + B, n);
            const size_t jend = std::min(j0 + B, n);

            for (size_t i = i0; i < iend; ++i) {
                const size_t jmax = std::min(jend, i + 1);
                for (size_t j = j0; j < jmax; ++j) {
                    double sum = 0.0;
                    for (size_t k = k0; k < kend; ++k) {
                        sum += A[i * n + k] * A[j * n + k];
                    }
                    A[i * n + j] -= sum;
                }
            }
        }
    }

    // Zero out upper triangular part
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            A[i * n + j] = 0.0;
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
    
    // Generate random matrix B
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }
    
    // Compute A = B * B^T
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
    
    // Compute L * L^T
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
    
    // Compare with original
    double maxError = 0.0;
    double relError = 0.0;

    #pragma omp parallel for schedule(static) reduction(max : maxError, relError)
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
