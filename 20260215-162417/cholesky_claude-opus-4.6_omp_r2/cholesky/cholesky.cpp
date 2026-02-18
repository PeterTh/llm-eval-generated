#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <omp.h>

#include "../common/results_output.hpp"

// Blocked Cholesky decomposition parallelized with OpenMP
// Decomposes positive definite matrix A into L * L^T where L is lower triangular

static constexpr size_t BLOCK_SIZE = 128;

// Unblocked Cholesky on a submatrix block
// a points to top-left of block, lda is the full matrix row stride
static bool cholesky_unblocked(double* a, size_t lda, size_t nb) {
    for (size_t j = 0; j < nb; j++) {
        double s = a[j * lda + j];
        for (size_t k = 0; k < j; k++)
            s -= a[j * lda + k] * a[j * lda + k];
        if (s <= 0.0) return false;
        a[j * lda + j] = std::sqrt(s);
        double inv_diag = 1.0 / a[j * lda + j];
        for (size_t i = j + 1; i < nb; i++) {
            double t = a[i * lda + j];
            for (size_t k = 0; k < j; k++)
                t -= a[i * lda + k] * a[j * lda + k];
            a[i * lda + j] = t * inv_diag;
        }
    }
    return true;
}

// TRSM: solve X * L^T = B in-place (result overwrites B)
// L is nb×nb lower triangular, B is mb×nb
static void trsm_right_lower_trans(double* b, const double* l, size_t lda, size_t mb, size_t nb) {
    for (size_t j = 0; j < nb; j++) {
        double inv_diag = 1.0 / l[j * lda + j];
        for (size_t i = 0; i < mb; i++) {
            double s = b[i * lda + j];
            for (size_t k = 0; k < j; k++)
                s -= b[i * lda + k] * l[j * lda + k];
            b[i * lda + j] = s * inv_diag;
        }
    }
}

// SYRK: A -= B * B^T (lower triangle only), A is mb×mb, B is mb×kb
static void syrk_lower(double* a, const double* b, size_t lda, size_t mb, size_t kb) {
    for (size_t i = 0; i < mb; i++) {
        for (size_t j = 0; j <= i; j++) {
            double s = 0.0;
            for (size_t k = 0; k < kb; k++)
                s += b[i * lda + k] * b[j * lda + k];
            a[i * lda + j] -= s;
        }
    }
}

// GEMM: C -= A * B^T, C is mb×nb, A is mb×kb, B is nb×kb
static void gemm_nt(double* c, const double* a, const double* b, size_t lda, size_t mb, size_t nb, size_t kb) {
    for (size_t i = 0; i < mb; i++) {
        for (size_t j = 0; j < nb; j++) {
            double s = 0.0;
            for (size_t k = 0; k < kb; k++)
                s += a[i * lda + k] * b[j * lda + k];
            c[i * lda + j] -= s;
        }
    }
}

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    double* a = A.data();
    const size_t bs = std::min(BLOCK_SIZE, n);

    for (size_t jj = 0; jj < n; jj += bs) {
        const size_t jb = std::min(bs, n - jj);

        // 1. Factor diagonal block
        if (!cholesky_unblocked(&a[jj * n + jj], n, jb)) {
            printf("Error: Matrix is not positive definite at diagonal element %zu\n", jj);
            return false;
        }

        if (jj + jb >= n) break;

        // 2. TRSM: compute block column below diagonal
        #pragma omp parallel for schedule(dynamic)
        for (size_t ii = jj + jb; ii < n; ii += bs) {
            const size_t ib = std::min(bs, n - ii);
            trsm_right_lower_trans(&a[ii * n + jj], &a[jj * n + jj], n, ib, jb);
        }

        // 3. Update trailing submatrix with linearized block indices
        const size_t trail = jj + jb;
        const size_t nblocks = (n - trail + bs - 1) / bs;
        const size_t ntasks = nblocks * (nblocks + 1) / 2;

        #pragma omp parallel for schedule(dynamic)
        for (size_t idx = 0; idx < ntasks; idx++) {
            // Map linear index to lower-triangular block indices (bi >= bj)
            size_t bi = (size_t)((std::sqrt(8.0 * (double)idx + 1.0) - 1.0) * 0.5);
            if ((bi + 1) * (bi + 2) / 2 <= idx) bi++;
            const size_t bj = idx - bi * (bi + 1) / 2;

            const size_t ii = trail + bi * bs;
            const size_t kk = trail + bj * bs;
            const size_t ib = std::min(bs, n - ii);
            const size_t kb = std::min(bs, n - kk);

            if (bi == bj) {
                syrk_lower(&a[ii * n + ii], &a[ii * n + jj], n, ib, jb);
            } else {
                gemm_nt(&a[ii * n + kk], &a[ii * n + jj], &a[kk * n + jj], n, ib, kb, jb);
            }
        }
    }

    // Zero upper triangle
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i++) {
        for (size_t j = i + 1; j < n; j++) {
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
    
    // Compute A = B * B^T (parallel)
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
    
    // Compute L * L^T (parallel)
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
