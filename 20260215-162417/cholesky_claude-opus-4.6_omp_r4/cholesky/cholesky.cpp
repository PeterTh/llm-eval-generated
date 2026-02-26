#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <omp.h>

#include "../common/results_output.hpp"

// Blocked Cholesky decomposition with OpenMP task-based parallelism
// Decomposes positive definite matrix A into L * L^T where L is lower triangular

static constexpr size_t BLOCK_SIZE = 64;

// Cholesky factorization of a single diagonal block (sequential)
static bool potrf(double* A, size_t n, size_t b0, size_t bs) {
    for (size_t j = 0; j < bs; ++j) {
        const size_t gj = b0 + j;
        double s = 0.0;
        for (size_t k = 0; k < j; ++k) {
            double v = A[gj * n + b0 + k];
            s += v * v;
        }
        double val = A[gj * n + gj] - s;
        if (val <= 0.0) {
            printf("Error: Matrix is not positive definite at diagonal element %zu\n", gj);
            return false;
        }
        A[gj * n + gj] = sqrt(val);
        double inv_d = 1.0 / A[gj * n + gj];
        for (size_t i = j + 1; i < bs; ++i) {
            const size_t gi = b0 + i;
            s = 0.0;
            for (size_t k = 0; k < j; ++k) {
                s += A[gi * n + b0 + k] * A[gj * n + b0 + k];
            }
            A[gi * n + gj] = (A[gi * n + gj] - s) * inv_d;
        }
    }
    return true;
}

// Triangular solve: L_ik = A_ik * inv(L_kk^T)
static void trsm(double* A, size_t n, size_t bi, size_t bk, size_t bsi, size_t bsk) {
    for (size_t j = 0; j < bsk; ++j) {
        double inv_d = 1.0 / A[(bk + j) * n + (bk + j)];
        for (size_t i = 0; i < bsi; ++i) {
            double s = 0.0;
            for (size_t m = 0; m < j; ++m) {
                s += A[(bi + i) * n + (bk + m)] * A[(bk + j) * n + (bk + m)];
            }
            A[(bi + i) * n + (bk + j)] = (A[(bi + i) * n + (bk + j)] - s) * inv_d;
        }
    }
}

// Symmetric rank-k update: A_ii -= L_ik * L_ik^T (lower triangle only)
static void syrk(double* A, size_t n, size_t bi, size_t bk, size_t bsi, size_t bsk) {
    for (size_t i = 0; i < bsi; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double s = 0.0;
            for (size_t k = 0; k < bsk; ++k) {
                s += A[(bi + i) * n + (bk + k)] * A[(bi + j) * n + (bk + k)];
            }
            A[(bi + i) * n + (bi + j)] -= s;
        }
    }
}

// General matrix multiply: A_ij -= L_ik * L_jk^T
static void gemm(double* A, size_t n, size_t bi, size_t bj, size_t bk,
                  size_t bsi, size_t bsj, size_t bsk) {
    for (size_t i = 0; i < bsi; ++i) {
        for (size_t j = 0; j < bsj; ++j) {
            double s = 0.0;
            for (size_t k = 0; k < bsk; ++k) {
                s += A[(bi + i) * n + (bk + k)] * A[(bj + j) * n + (bk + k)];
            }
            A[(bi + i) * n + (bj + j)] -= s;
        }
    }
}

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    double* data = A.data();
    const size_t nb = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // Sentinel array for OpenMP task dependencies
    std::vector<char> dep(nb * nb, 0);
    char* __attribute__((unused)) d = dep.data();
    bool success = true;

    #pragma omp parallel
    #pragma omp single
    {
        for (size_t k = 0; k < nb; ++k) {
            const size_t k0 = k * BLOCK_SIZE;
            const size_t ks = std::min(BLOCK_SIZE, n - k0);

            // Factor diagonal block
            #pragma omp task depend(inout: d[k*nb+k]) shared(success)
            {
                if (success && !potrf(data, n, k0, ks))
                    success = false;
            }

            // Solve column panel below diagonal
            for (size_t i = k + 1; i < nb; ++i) {
                const size_t i0 = i * BLOCK_SIZE;
                const size_t is_ = std::min(BLOCK_SIZE, n - i0);

                #pragma omp task depend(in: d[k*nb+k]) depend(inout: d[i*nb+k])
                {
                    trsm(data, n, i0, k0, is_, ks);
                }
            }

            // Update trailing submatrix
            for (size_t i = k + 1; i < nb; ++i) {
                const size_t i0 = i * BLOCK_SIZE;
                const size_t is_ = std::min(BLOCK_SIZE, n - i0);

                // Diagonal: symmetric rank-k update
                #pragma omp task depend(in: d[i*nb+k]) depend(inout: d[i*nb+i])
                {
                    syrk(data, n, i0, k0, is_, ks);
                }

                // Off-diagonal: general matrix multiply
                for (size_t j = k + 1; j < i; ++j) {
                    const size_t j0 = j * BLOCK_SIZE;
                    const size_t js = std::min(BLOCK_SIZE, n - j0);

                    #pragma omp task depend(in: d[i*nb+k]) depend(in: d[j*nb+k]) depend(inout: d[i*nb+j])
                    {
                        gemm(data, n, i0, j0, k0, is_, js, ks);
                    }
                }
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

    return success;
}

// Generate a symmetric positive definite matrix
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;

    // Sequential RNG for deterministic results
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

    // Compute A = B * B^T in parallel
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
    std::vector<double> reconstructed(n * n);

    // Compute L * L^T in parallel
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
    
    printf("Cholesky Decomposition Benchmark\n");
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
