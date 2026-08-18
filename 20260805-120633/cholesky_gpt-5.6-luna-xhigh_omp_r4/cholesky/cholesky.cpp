#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <omp.h>

#include "../common/results_output.hpp"

// Blocked Cholesky decomposition.  The diagonal block is factored using the
// usual left-looking algorithm, while the panel solve and trailing update are
// distributed across the OpenMP team.  A blocked implementation keeps the
// operands of the update in cache long enough to make the parallel work
// substantially more useful than parallelizing the short inner loops of the
// unblocked algorithm.
// Decomposes positive definite matrix A into L * L^T where L is lower triangular

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    // A is stored in row-major order.  This size makes each update tile large
    // enough to amortize OpenMP scheduling overhead while fitting the active
    // panel and a useful part of the trailing matrix in cache.
    constexpr size_t block_size = 128;
    bool success = true;

    // Keep one team alive for the complete factorization.  The barriers at
    // each worksharing construct are the synchronization points required by
    // the Cholesky panel dependency chain.
    omp_set_dynamic(0);
#pragma omp parallel default(none) shared(A, n, success, block_size)
    {
        for (size_t kk = 0; kk < n; kk += block_size) {
            const size_t kb = std::min(block_size, n - kk);

            // The diagonal block is the only sequential part of this panel.
            // Every thread must observe it before solving the panel below it.
#pragma omp single
            {
                if (success) {
                    for (size_t j = kk; j < kk + kb; ++j) {
                        double sum = 0.0;
                        // Earlier panels have already been applied to the
                        // diagonal block by the trailing updates below.
                        // Only contributions within this residual block are
                        // still to be removed.
                        for (size_t k = kk; k < j; ++k) {
                            sum += A[j * n + k] * A[j * n + k];
                        }

                        const double val = A[j * n + j] - sum;
                        if (val <= 0.0) {
                            printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
                            success = false;
                            break;
                        }
                        A[j * n + j] = sqrt(val);

                        for (size_t i = j + 1; i < kk + kb; ++i) {
                            double block_sum = 0.0;
                            for (size_t k = kk; k < j; ++k) {
                                block_sum += A[i * n + k] * A[j * n + k];
                            }
                            A[i * n + j] = (A[i * n + j] - block_sum) / A[j * n + j];
                        }
                    }
                }
            }

            // Solve L21 * L11^T = A21.  Rows are independent once the
            // diagonal block has been factored.
#pragma omp for schedule(static)
            for (size_t i = kk + kb; i < n; ++i) {
                if (!success) {
                    continue;
                }
                for (size_t j = kk; j < kk + kb; ++j) {
                    double sum = 0.0;
                    for (size_t k = kk; k < j; ++k) {
                        sum += A[i * n + k] * A[j * n + k];
                    }
                    A[i * n + j] = (A[i * n + j] - sum) / A[j * n + j];
                }
            }

            // Update only the lower triangle of the trailing matrix.  The
            // rectangular tile iteration gives the runtime enough independent
            // units for scalable scheduling, while the diagonal tiles handle
            // their own triangular half.
            const size_t first = kk + kb;
#pragma omp for collapse(2) schedule(dynamic, 1)
            for (size_t ii = first; ii < n; ii += block_size) {
                for (size_t jj = first; jj < n; jj += block_size) {
                    if (!success || jj > ii) {
                        continue;
                    }

                    const size_t i_end = std::min(ii + block_size, n);
                    const size_t j_end = std::min(jj + block_size, n);
                    for (size_t i = ii; i < i_end; ++i) {
                        const size_t j_stop = std::min(j_end, i + 1);
                        for (size_t j = jj; j < j_stop; ++j) {
                            double sum = 0.0;
#pragma omp simd reduction(+:sum)
                            for (size_t k = kk; k < kk + kb; ++k) {
                                sum += A[i * n + k] * A[j * n + k];
                            }
                            A[i * n + j] -= sum;
                        }
                    }
                }
            }
        }

        // Preserve the original routine's output semantics.  The factor is
        // lower triangular and its upper half is explicitly zeroed.
#pragma omp for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i + 1; j < n; ++j) {
                A[i * n + j] = 0.0;
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
