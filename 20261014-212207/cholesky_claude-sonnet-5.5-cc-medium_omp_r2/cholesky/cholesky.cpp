#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <omp.h>

#include "../common/results_output.hpp"

// Simple Cholesky decomposition (sequential, unblocked algorithm)
// Decomposes positive definite matrix A into L * L^T where L is lower triangular

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    // Blocked right-looking algorithm, A is stored in row-major order.
    // Only the lower triangle is referenced; the upper triangle is zeroed.
    constexpr size_t NB = 64;
    double* __restrict a = A.data();
    const long nl = (long)n;

    // Zero out upper triangular part
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < nl; ++i) {
        for (size_t j = i + 1; j < n; ++j) a[i * n + j] = 0.0;
    }

    for (size_t k0 = 0; k0 < n; k0 += NB) {
        const size_t kb = std::min(NB, n - k0);
        const size_t k1 = k0 + kb;

        // Factor diagonal block (small, sequential)
        for (size_t j = k0; j < k1; ++j) {
            double sum = 0.0;
            for (size_t k = k0; k < j; ++k) sum += a[j * n + k] * a[j * n + k];
            const double val = a[j * n + j] - sum;
            if (val <= 0.0) {
                printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
                return false;
            }
            const double d = sqrt(val);
            a[j * n + j] = d;
            for (size_t i = j + 1; i < k1; ++i) {
                double s = 0.0;
                for (size_t k = k0; k < j; ++k) s += a[i * n + k] * a[j * n + k];
                a[i * n + j] = (a[i * n + j] - s) / d;
            }
        }
        if (k1 >= n) break;

        // Panel solve: rows below the diagonal block
        #pragma omp parallel for schedule(static)
        for (long i = (long)k1; i < nl; ++i) {
            for (size_t j = k0; j < k1; ++j) {
                double s = 0.0;
                #pragma omp simd reduction(+:s)
                for (size_t k = k0; k < j; ++k) s += a[i * n + k] * a[j * n + k];
                a[i * n + j] = (a[i * n + j] - s) / a[j * n + j];
            }
        }

        // Trailing update over lower-triangular tiles: A[i][j] -= L[i][k0:k1] . L[j][k0:k1]
        const size_t T = 64;
        const size_t nt = (n - k1 + T - 1) / T;
        const size_t ntiles = nt * (nt + 1) / 2;
        #pragma omp parallel for schedule(dynamic, 1)
        for (long t = (long)ntiles - 1; t >= 0; --t) {
            // map linear index to (ti, tj), tj <= ti
            size_t ti = (size_t)((sqrt(8.0 * (double)t + 1.0) - 1.0) / 2.0);
            while (ti * (ti + 1) / 2 > (size_t)t) --ti;
            while ((ti + 1) * (ti + 2) / 2 <= (size_t)t) ++ti;
            const size_t tj = (size_t)t - ti * (ti + 1) / 2;
            const size_t i0 = k1 + ti * T, i1 = std::min(i0 + T, n);
            const size_t j0 = k1 + tj * T, j1 = std::min(j0 + T, n);
            for (size_t i = i0; i < i1; ++i) {
                const double* __restrict li = a + i * n + k0;
                const size_t jend = std::min(j1, i + 1);
                size_t j = j0;
                for (; j + 4 <= jend; j += 4) {
                    const double* __restrict l0 = a + (j + 0) * n + k0;
                    const double* __restrict l1 = a + (j + 1) * n + k0;
                    const double* __restrict l2 = a + (j + 2) * n + k0;
                    const double* __restrict l3 = a + (j + 3) * n + k0;
                    double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
                    #pragma omp simd reduction(+:s0,s1,s2,s3)
                    for (size_t k = 0; k < kb; ++k) {
                        const double v = li[k];
                        s0 += v * l0[k];
                        s1 += v * l1[k];
                        s2 += v * l2[k];
                        s3 += v * l3[k];
                    }
                    a[i * n + j + 0] -= s0;
                    a[i * n + j + 1] -= s1;
                    a[i * n + j + 2] -= s2;
                    a[i * n + j + 3] -= s3;
                }
                for (; j < jend; ++j) {
                    const double* __restrict l0 = a + j * n + k0;
                    double s0 = 0;
                    #pragma omp simd reduction(+:s0)
                    for (size_t k = 0; k < kb; ++k) s0 += li[k] * l0[k];
                    a[i * n + j] -= s0;
                }
            }
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
    // (sequential: rand_r stream order defines the matrix)
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }
    
    // Compute A = B * B^T
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < (long)n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
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
    for (long i = 0; i < (long)n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (size_t k = 0; k < n; ++k) {
                sum += L[i * n + k] * L[j * n + k];
            }
            reconstructed[i * n + j] = sum;
        }
    }
    
    // Compare with original
    double maxError = 0.0;
    double relError = 0.0;
    
    #pragma omp parallel for schedule(static) reduction(max:maxError,relError)
    for (long i = 0; i < (long)(n * n); ++i) {
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
