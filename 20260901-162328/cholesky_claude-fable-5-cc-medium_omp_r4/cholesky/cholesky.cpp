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
// The small diagonal block is factored sequentially; the panel solve and the
// trailing submatrix update (the O(n^3) bulk of the work) run in parallel.

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    // A is stored in row-major order
    constexpr size_t NB = 128; // block size
    double* const a = A.data();

    for (size_t k = 0; k < n; k += NB) {
        const size_t kb = std::min(NB, n - k);
        const size_t kend = k + kb;

        // Factor the diagonal block A[k:kend, k:kend] (unblocked, sequential).
        // Contributions from columns < k were already subtracted by earlier
        // trailing updates.
        for (size_t j = k; j < kend; ++j) {
            double sum = 0.0;
            for (size_t p = k; p < j; ++p) {
                sum += a[j * n + p] * a[j * n + p];
            }
            const double val = a[j * n + j] - sum;
            if (val <= 0.0) {
                // Matrix is not positive definite
                printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
                return false;
            }
            const double djj = sqrt(val);
            a[j * n + j] = djj;

            for (size_t i = j + 1; i < kend; ++i) {
                double s = 0.0;
                for (size_t p = k; p < j; ++p) {
                    s += a[i * n + p] * a[j * n + p];
                }
                a[i * n + j] = (a[i * n + j] - s) / djj;
            }
        }

        // Panel solve: L21 = A21 * L11^-T for rows below the diagonal block.
        #pragma omp parallel for schedule(static)
        for (size_t i = kend; i < n; ++i) {
            for (size_t j = k; j < kend; ++j) {
                double s = 0.0;
                for (size_t p = k; p < j; ++p) {
                    s += a[i * n + p] * a[j * n + p];
                }
                a[i * n + j] = (a[i * n + j] - s) / a[j * n + j];
            }
        }

        // Trailing update: A22 -= L21 * L21^T (lower triangle only).
        // The triangular set of tile pairs is flattened into a single loop so
        // all pairs are distributed across threads for load balance.
        const size_t m = (n - kend + NB - 1) / NB;
        const size_t npairs = m * (m + 1) / 2;
        #pragma omp parallel for schedule(dynamic, 1)
        for (size_t t = 0; t < npairs; ++t) {
            // Map flat index t to tile pair (bi, bj) with bj <= bi
            size_t bi = (size_t)((sqrt(8.0 * (double)t + 1.0) - 1.0) / 2.0);
            while ((bi + 1) * (bi + 2) / 2 <= t) ++bi;
            while (bi * (bi + 1) / 2 > t) --bi;
            const size_t bj = t - bi * (bi + 1) / 2;

            const size_t ib = kend + bi * NB;
            const size_t jb = kend + bj * NB;
            const size_t iend = std::min(ib + NB, n);
            const size_t jend = std::min(jb + NB, n);
            for (size_t i = ib; i < iend; ++i) {
                const double* const li = a + i * n + k;
                const size_t jmax = std::min(jend, i + 1);
                for (size_t j = jb; j < jmax; ++j) {
                    const double* const lj = a + j * n + k;
                    double s = 0.0;
                    for (size_t p = 0; p < kb; ++p) {
                        s += li[p] * lj[p];
                    }
                    a[i * n + j] -= s;
                }
            }
        }
    }

    // Zero out upper triangular part
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            a[i * n + j] = 0.0;
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
