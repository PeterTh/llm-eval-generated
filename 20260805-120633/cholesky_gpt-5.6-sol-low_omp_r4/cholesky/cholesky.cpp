#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// Cache-blocked right-looking Cholesky decomposition.  The diagonal block is
// small and serial; the panel solve and the O(n^3) trailing update are shared
// across the OpenMP team.

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    constexpr size_t blockSize = 64;
    bool positiveDefinite = true;
    size_t failedAt = 0;

#pragma omp parallel shared(positiveDefinite, failedAt)
    {
        for (size_t k = 0; k < n; k += blockSize) {
            const size_t end = std::min(k + blockSize, n);

#pragma omp single
            {
                if (positiveDefinite) {
                    for (size_t j = k; j < end; ++j) {
                        double diagonal = A[j * n + j];
#pragma omp simd reduction(- : diagonal)
                        for (size_t p = k; p < j; ++p)
                            diagonal -= A[j * n + p] * A[j * n + p];
                        if (!(diagonal > 0.0)) {
                            positiveDefinite = false;
                            failedAt = j;
                            break;
                        }
                        A[j * n + j] = std::sqrt(diagonal);
                        const double inverseDiagonal = 1.0 / A[j * n + j];
                        for (size_t i = j + 1; i < end; ++i) {
                            double value = A[i * n + j];
#pragma omp simd reduction(- : value)
                            for (size_t p = k; p < j; ++p)
                                value -= A[i * n + p] * A[j * n + p];
                            A[i * n + j] = value * inverseDiagonal;
                        }
                    }
                }
            }

            if (positiveDefinite) {
                // A_ik <- A_ik * inv(L_kk^T); rows are independent.
#pragma omp for schedule(static)
                for (size_t i = end; i < n; ++i) {
                    for (size_t j = k; j < end; ++j) {
                        double value = A[i * n + j];
#pragma omp simd reduction(- : value)
                        for (size_t p = k; p < j; ++p)
                            value -= A[i * n + p] * A[j * n + p];
                        A[i * n + j] = value / A[j * n + j];
                    }
                }

                // A_ij <- A_ij - L_ik*L_jk^T.  Assign independent lower
                // triangular tiles dynamically to balance edge tiles.
                const size_t blocks = (n - end + blockSize - 1) / blockSize;
#pragma omp for schedule(dynamic, 1)
                for (size_t tile = 0; tile < blocks * blocks; ++tile) {
                    const size_t bi = tile / blocks;
                    const size_t bj = tile % blocks;
                    if (bj > bi) continue;
                    const size_t i0 = end + bi * blockSize;
                    const size_t j0 = end + bj * blockSize;
                    const size_t i1 = std::min(i0 + blockSize, n);
                    const size_t j1 = std::min(j0 + blockSize, n);
                    for (size_t i = i0; i < i1; ++i) {
                        const size_t lastJ = (bi == bj) ? std::min(j1, i + 1) : j1;
                        for (size_t j = j0; j < lastJ; ++j) {
                            double value = A[i * n + j];
#pragma omp simd reduction(- : value)
                            for (size_t p = k; p < end; ++p)
                                value -= A[i * n + p] * A[j * n + p];
                            A[i * n + j] = value;
                        }
                    }
                }
            }
        }

#pragma omp for schedule(static)
        for (size_t i = 0; i < n; ++i)
            std::fill(A.begin() + i * n + i + 1, A.begin() + (i + 1) * n, 0.0);
    }

    if (!positiveDefinite)
        printf("Error: Matrix is not positive definite at diagonal element %zu\n", failedAt);
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
    
    // Compute one triangle of A = B * B^T and mirror it.
#pragma omp parallel for schedule(static)
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
