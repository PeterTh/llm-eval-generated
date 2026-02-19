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

static constexpr size_t BLOCK_SIZE = 128;

// Unblocked Cholesky on a small diagonal block (in-place, within full matrix)
static bool choleskySmall(double* __restrict__ A, size_t n, size_t r0, size_t bs) {
    for (size_t j = 0; j < bs; ++j) {
        double sum = 0.0;
        for (size_t k = 0; k < j; ++k) {
            double v = A[(r0 + j) * n + (r0 + k)];
            sum += v * v;
        }
        double val = A[(r0 + j) * n + (r0 + j)] - sum;
        if (val <= 0.0) {
            printf("Error: Matrix is not positive definite at diagonal element %zu\n", r0 + j);
            return false;
        }
        A[(r0 + j) * n + (r0 + j)] = sqrt(val);
        double diag_inv = 1.0 / A[(r0 + j) * n + (r0 + j)];

        for (size_t i = j + 1; i < bs; ++i) {
            double s = 0.0;
            for (size_t k = 0; k < j; ++k) {
                s += A[(r0 + i) * n + (r0 + k)] * A[(r0 + j) * n + (r0 + k)];
            }
            A[(r0 + i) * n + (r0 + j)] = (A[(r0 + i) * n + (r0 + j)] - s) * diag_inv;
        }
    }
    return true;
}

// TRSM: solve X * L_diag^T = A_panel  =>  row i of panel: for each col j
// L_diag is bs x bs lower triangular at (r0,r0), panel row i is at row ri
static void trsmRow(double* __restrict__ A, size_t n, size_t r0, size_t bs, size_t ri) {
    for (size_t j = 0; j < bs; ++j) {
        double s = 0.0;
        for (size_t k = 0; k < j; ++k) {
            s += A[ri * n + (r0 + k)] * A[(r0 + j) * n + (r0 + k)];
        }
        A[ri * n + (r0 + j)] = (A[ri * n + (r0 + j)] - s) / A[(r0 + j) * n + (r0 + j)];
    }
}

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    double* Ad = A.data();
    size_t nb = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (size_t kb = 0; kb < nb; ++kb) {
        size_t r0 = kb * BLOCK_SIZE;
        size_t bs = std::min(BLOCK_SIZE, n - r0);

        // Factor diagonal block
        if (!choleskySmall(Ad, n, r0, bs))
            return false;

        // TRSM: solve panel blocks below diagonal
        size_t panel_rows = n - r0 - bs;
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < panel_rows; ++i) {
            trsmRow(Ad, n, r0, bs, r0 + bs + i);
        }

        // SYRK/GEMM: update trailing submatrix
        size_t trail = nb - kb - 1;
        // Parallelize over block-row pairs in the lower triangle
        #pragma omp parallel for schedule(dynamic) collapse(2)
        for (size_t ib = 0; ib < trail; ++ib) {
            for (size_t jb = 0; jb < trail; ++jb) {
                if (jb > ib) continue; // lower triangle only
                size_t bi = r0 + bs + ib * BLOCK_SIZE;
                size_t bj = r0 + bs + jb * BLOCK_SIZE;
                size_t bsi = std::min(BLOCK_SIZE, n - bi);
                size_t bsj = std::min(BLOCK_SIZE, n - bj);

                if (ib == jb) {
                    // Symmetric rank-k update on diagonal block
                    for (size_t ii = 0; ii < bsi; ++ii) {
                        size_t ri = bi + ii;
                        for (size_t k = 0; k < bs; ++k) {
                            double l_ik = Ad[ri * n + (r0 + k)];
                            for (size_t jj = 0; jj <= ii; ++jj) {
                                Ad[ri * n + (bi + jj)] -= l_ik * Ad[(bi + jj) * n + (r0 + k)];
                            }
                        }
                    }
                } else {
                    // General update: A[bi:, bj:] -= L[bi:, r0:r0+bs] * L[bj:, r0:r0+bs]^T
                    for (size_t ii = 0; ii < bsi; ++ii) {
                        size_t ri = bi + ii;
                        for (size_t k = 0; k < bs; ++k) {
                            double l_ik = Ad[ri * n + (r0 + k)];
                            for (size_t jj = 0; jj < bsj; ++jj) {
                                Ad[ri * n + (bj + jj)] -= l_ik * Ad[(bj + jj) * n + (r0 + k)];
                            }
                        }
                    }
                }
            }
        }
    }

    // Zero out upper triangular part
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            Ad[i * n + j] = 0.0;
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

    // Generate random matrix B (sequential to preserve determinism with rand_r)
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

    // Compute A = B * B^T (parallelized)
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

    // Compute L * L^T (parallelized)
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
    
    printf("Cholesky Decomposition Benchmark (OpenMP)\n");
    printf("Matrix size: %zu x %zu\n", n, n);
    printf("Threads: %d\n", omp_get_max_threads());
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
