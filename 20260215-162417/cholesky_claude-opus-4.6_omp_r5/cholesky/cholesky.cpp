#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <omp.h>

#include "../common/results_output.hpp"

// Blocked Cholesky decomposition with OpenMP parallelism
// Decomposes positive definite matrix A into L * L^T where L is lower triangular

static const size_t BLOCK_SIZE = 64;

// Sequential Cholesky factorization of a diagonal block (in-place)
static bool cholesky_diag_block(double* __restrict__ A, size_t ld, size_t sz) {
    for (size_t j = 0; j < sz; ++j) {
        double s = A[j * ld + j];
        for (size_t k = 0; k < j; ++k)
            s -= A[j * ld + k] * A[j * ld + k];
        if (s <= 0.0) {
            printf("Error: Matrix is not positive definite\n");
            return false;
        }
        A[j * ld + j] = sqrt(s);
        double inv = 1.0 / A[j * ld + j];
        for (size_t i = j + 1; i < sz; ++i) {
            double t = A[i * ld + j];
            for (size_t k = 0; k < j; ++k)
                t -= A[i * ld + k] * A[j * ld + k];
            A[i * ld + j] = t * inv;
        }
    }
    return true;
}

// TRSM: solve panel * L_diag^{-T} for a panel below the diagonal block
static void trsm_lower(double* __restrict__ panel, const double* __restrict__ diag,
                        size_t ld, size_t rows, size_t cols) {
    for (size_t j = 0; j < cols; ++j) {
        double inv = 1.0 / diag[j * ld + j];
        for (size_t i = 0; i < rows; ++i) {
            double s = panel[i * ld + j];
            for (size_t k = 0; k < j; ++k)
                s -= panel[i * ld + k] * diag[j * ld + k];
            panel[i * ld + j] = s * inv;
        }
    }
}

// SYRK: A_ii -= A_ik * A_ik^T (lower triangle only)
static void syrk_update(double* __restrict__ A_ii, const double* __restrict__ A_ik,
                         size_t ld, size_t sz, size_t ksz) {
    for (size_t i = 0; i < sz; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double s = 0.0;
            for (size_t k = 0; k < ksz; ++k)
                s += A_ik[i * ld + k] * A_ik[j * ld + k];
            A_ii[i * ld + j] -= s;
        }
    }
}

// GEMM: A_ij -= A_ik * A_jk^T
static void gemm_update(double* __restrict__ A_ij, const double* __restrict__ A_ik,
                         const double* __restrict__ A_jk,
                         size_t ld, size_t m, size_t ndim, size_t ksz) {
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < ndim; ++j) {
            double s = 0.0;
            for (size_t k = 0; k < ksz; ++k)
                s += A_ik[i * ld + k] * A_jk[j * ld + k];
            A_ij[i * ld + j] -= s;
        }
    }
}

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    const size_t bs = std::min(BLOCK_SIZE, n);
    const size_t nb = (n + bs - 1) / bs;
    bool success = true;

    #pragma omp parallel
    {
        for (size_t kb = 0; kb < nb && success; ++kb) {
            const size_t ks = kb * bs;
            const size_t ksz = std::min(bs, n - ks);

            // Factor diagonal block (single thread)
            #pragma omp single
            {
                if (!cholesky_diag_block(&A[ks * n + ks], n, ksz))
                    success = false;
            }

            if (!success) break;

            // TRSM: update column panels below the diagonal block
            #pragma omp for schedule(dynamic)
            for (size_t ib = kb + 1; ib < nb; ++ib) {
                const size_t is_ = ib * bs;
                const size_t isz = std::min(bs, n - is_);
                trsm_lower(&A[is_ * n + ks], &A[ks * n + ks], n, isz, ksz);
            }

            // Update trailing submatrix (SYRK on diagonal blocks, GEMM off-diagonal)
            #pragma omp for schedule(dynamic) collapse(2)
            for (size_t ib = kb + 1; ib < nb; ++ib) {
                for (size_t jb = kb + 1; jb < nb; ++jb) {
                    if (jb <= ib) {
                        const size_t is_ = ib * bs;
                        const size_t js = jb * bs;
                        const size_t isz = std::min(bs, n - is_);
                        const size_t jsz = std::min(bs, n - js);
                        if (ib == jb) {
                            syrk_update(&A[is_ * n + is_], &A[is_ * n + ks], n, isz, ksz);
                        } else {
                            gemm_update(&A[is_ * n + js], &A[is_ * n + ks],
                                        &A[js * n + ks], n, isz, jsz, ksz);
                        }
                    }
                }
            }
        }

        // Zero out upper triangular part
        #pragma omp for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i + 1; j < n; ++j)
                A[i * n + j] = 0.0;
        }
    }

    return success;
}

// Generate a symmetric positive definite matrix
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;

    // Sequential random generation to preserve determinism
    for (size_t i = 0; i < n * n; ++i)
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;

    // Compute A = B * B^T (parallel)
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k)
                sum += B[i * n + k] * B[j * n + k];
            A[i * n + j] = sum;
            A[j * n + i] = sum;
        }
    }

    // Add diagonal dominance to ensure positive definiteness
    for (size_t i = 0; i < n; ++i)
        A[i * n + i] += n;
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    std::vector<double> reconstructed(n * n);

    // Compute L * L^T (parallel)
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k)
                sum += L[i * n + k] * L[j * n + k];
            reconstructed[i * n + j] = sum;
        }
    }

    // Compare with original (parallel reduction)
    double maxError = 0.0;
    double relError = 0.0;

    #pragma omp parallel for reduction(max:maxError,relError)
    for (size_t i = 0; i < n * n; ++i) {
        const double error = fabs(reconstructed[i] - A_orig[i]);
        maxError = std::max(maxError, error);
        const double rel = error / (fabs(A_orig[i]) + 1e-10);
        relError = std::max(relError, rel);
    }

    printf("Max absolute error: %.10e\n", maxError);
    printf("Max relative error: %.10e\n", relError);

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
    
    printf("Cholesky Decomposition Benchmark (OpenMP)\n");
    printf("Matrix size: %zu x %zu\n", n, n);
    printf("OpenMP threads: %d\n", omp_get_max_threads());
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
