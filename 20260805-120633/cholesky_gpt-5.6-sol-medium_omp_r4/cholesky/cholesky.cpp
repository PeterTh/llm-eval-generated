#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <omp.h>
#include <vector>

#include "../common/results_output.hpp"

// Cache-blocked, right-looking Cholesky decomposition.  The diagonal block is
// small and sequential; the panel solve and (dominant) trailing update are
// shared among all threads in one persistent OpenMP team.

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    constexpr size_t blockSize = 32;
    double* const a = A.data();
    bool success = true;

#pragma omp parallel shared(success)
    {
        for (size_t k = 0; k < n; k += blockSize) {
            const size_t blockEnd = std::min(k + blockSize, n);

            // Factor the diagonal block.  Contributions from earlier block
            // columns have already been removed by the trailing updates.
#pragma omp single
            {
                if (success) {
                    for (size_t j = k; j < blockEnd; ++j) {
                        double sum = 0.0;
#pragma omp simd reduction(+ : sum)
                        for (size_t p = k; p < j; ++p) {
                            sum += a[j * n + p] * a[j * n + p];
                        }
                        const double diagonal = a[j * n + j] - sum;

                        if (diagonal <= 0.0) {
                            printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
                            success = false;
                            break;
                        }

                        const double root = std::sqrt(diagonal);
                        const double inverseDiagonal = 1.0 / root;
                        a[j * n + j] = root;
                        for (size_t i = j + 1; i < blockEnd; ++i) {
                            double innerProduct = 0.0;
#pragma omp simd reduction(+ : innerProduct)
                            for (size_t p = k; p < j; ++p) {
                                innerProduct += a[i * n + p] * a[j * n + p];
                            }
                            a[i * n + j] = (a[i * n + j] - innerProduct) * inverseDiagonal;
                        }
                    }
                }
            }

            // Solve L_ik * L_kk^T = A_ik.  Rows are independent.
#pragma omp for schedule(static)
            for (size_t i = blockEnd; i < n; ++i) {
                if (success) {
                    for (size_t j = k; j < blockEnd; ++j) {
                        double innerProduct = 0.0;
#pragma omp simd reduction(+ : innerProduct)
                        for (size_t p = k; p < j; ++p) {
                            innerProduct += a[i * n + p] * a[j * n + p];
                        }
                        a[i * n + j] = (a[i * n + j] - innerProduct) / a[j * n + j];
                    }
                }
            }

            // A_ij -= L_ik * L_jk^T.  Flatten the lower-triangular set of
            // tile pairs so that even late, narrow updates balance well.
            const size_t tileCount = (n - blockEnd + blockSize - 1) / blockSize;
            const size_t pairCount = tileCount * (tileCount + 1) / 2;
#pragma omp for schedule(dynamic, 1)
            for (size_t pair = 0; pair < pairCount; ++pair) {
                if (success) {
                    size_t tileRow = static_cast<size_t>((std::sqrt(8.0 * pair + 1.0) - 1.0) * 0.5);
                    while ((tileRow + 1) * (tileRow + 2) / 2 <= pair) {
                        ++tileRow;
                    }
                    while (tileRow * (tileRow + 1) / 2 > pair) {
                        --tileRow;
                    }
                    const size_t tileColumn = pair - tileRow * (tileRow + 1) / 2;
                    const size_t rowBegin = blockEnd + tileRow * blockSize;
                    const size_t rowEnd = std::min(rowBegin + blockSize, n);
                    const size_t columnBegin = blockEnd + tileColumn * blockSize;
                    const size_t columnEnd = std::min(columnBegin + blockSize, n);

                    for (size_t i = rowBegin; i < rowEnd; ++i) {
                        const size_t lastColumn = (tileRow == tileColumn)
                                                      ? std::min(columnEnd, i + 1)
                                                      : columnEnd;
                        for (size_t j = columnBegin; j < lastColumn; ++j) {
                            double update = 0.0;
#pragma omp simd reduction(+ : update)
                            for (size_t p = k; p < blockEnd; ++p) {
                                update += a[i * n + p] * a[j * n + p];
                            }
                            a[i * n + j] -= update;
                        }
                    }
                }
            }
        }

        // Only the lower triangle represents L.
#pragma omp for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i + 1; j < n; ++j) {
                a[i * n + j] = 0.0;
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
    
    // Compute one triangle of A = B * B^T, then mirror it.  Each dot product
    // retains the original operation order, while matrix rows are parallel.
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
    for (size_t i = 0; i < n; ++i) {
        A[i * n + i] += n;
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix
    
    double maxError = 0.0;
    double relError = 0.0;

    // Reconstruct and compare without materializing another n-by-n matrix.
#pragma omp parallel for schedule(static) reduction(max : maxError, relError)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double reconstructed = 0.0;
            const size_t terms = std::min(i, j) + 1;
#pragma omp simd reduction(+ : reconstructed)
            for (size_t k = 0; k < terms; ++k) {
                reconstructed += L[i * n + k] * L[j * n + k];
            }
            const double error = std::fabs(reconstructed - A_orig[i * n + j]);
            maxError = std::max(maxError, error);
            const double rel = error / (std::fabs(A_orig[i * n + j]) + 1e-10);
            relError = std::max(relError, rel);
        }
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
