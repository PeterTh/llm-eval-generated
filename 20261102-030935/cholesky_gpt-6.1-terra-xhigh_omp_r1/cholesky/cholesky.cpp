#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <omp.h>

#include "../common/results_output.hpp"

// Avoid launching more workers than the matrix has useful blocks of rows.
// Honor a smaller limit selected through the OpenMP runtime.
static int threadCount(size_t n) {
    return static_cast<int>(std::min(n / 64 + 1,
                                     static_cast<size_t>(omp_get_max_threads())));
}

// Blocked, right-looking Cholesky decomposition. Only the lower triangle is
// used; each trailing row can be updated independently after a panel is done.

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    constexpr size_t blockSize = 64;
    std::vector<double> panel(blockSize * n);
    bool success = true;
    double* const a = A.data();
    double* const packed = panel.data();

    #pragma omp parallel num_threads(threadCount(n)) shared(success)
    {
        for (size_t base = 0; base < n; base += blockSize) {
            const size_t end = std::min(base + blockSize, n);
            #pragma omp single
            {
                // Factor the diagonal block, which depends on earlier blocks.
                for (size_t i = base; i < end && success; ++i) {
                    for (size_t j = base; j <= i; ++j) {
                        double sum = 0.0;
                        #pragma omp simd reduction(+:sum)
                        for (size_t k = base; k < j; ++k)
                            sum += a[i * n + k] * a[j * n + k];
                        const double val = a[i * n + j] - sum;
                        if (i == j) {
                            if (val <= 0.0) {
                                printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
                                success = false;
                                break;
                            }
                            a[i * n + i] = sqrt(val);
                        } else {
                            a[i * n + j] = val / a[j * n + j];
                        }
                    }
                }
            }
            if (!success) break;

            // Solve the remaining rows of this panel independently.
            #pragma omp for schedule(static)
            for (size_t i = end; i < n; ++i) {
                for (size_t j = base; j < end; ++j) {
                    double sum = 0.0;
                    #pragma omp simd reduction(+:sum)
                    for (size_t k = base; k < j; ++k)
                        sum += a[i * n + k] * a[j * n + k];
                    a[i * n + j] = (a[i * n + j] - sum) / a[j * n + j];
                }
            }

            // Transpose the panel so the trailing update reads contiguous data.
            #pragma omp for schedule(static)
            for (size_t k = base; k < end; ++k)
                for (size_t j = end; j < n; ++j)
                    packed[(k - base) * n + j] = a[j * n + k];

            #pragma omp for schedule(dynamic, 8)
            for (size_t i = end; i < n; ++i) {
                size_t k = base;
                for (; k + 3 < end; k += 4) {
                    const double x0 = a[i * n + k];
                    const double x1 = a[i * n + k + 1];
                    const double x2 = a[i * n + k + 2];
                    const double x3 = a[i * n + k + 3];
                    const double* const p0 = packed + (k - base) * n;
                    const double* const p1 = p0 + n;
                    const double* const p2 = p1 + n;
                    const double* const p3 = p2 + n;
                    #pragma omp simd
                    for (size_t j = end; j <= i; ++j)
                        a[i * n + j] -= x0 * p0[j] + x1 * p1[j] + x2 * p2[j] + x3 * p3[j];
                }
                for (; k < end; ++k) {
                    const double x = a[i * n + k];
                    const double* const p = packed + (k - base) * n;
                    #pragma omp simd
                    for (size_t j = end; j <= i; ++j)
                        a[i * n + j] -= x * p[j];
                }
            }
        }
        if (success) {
            #pragma omp for schedule(static)
            for (size_t i = 0; i < n; ++i)
                for (size_t j = i + 1; j < n; ++j)
                    a[i * n + j] = 0.0;
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
    #pragma omp parallel for num_threads(threadCount(n)) schedule(dynamic, 8)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (size_t k = 0; k < n; ++k) {
                sum += B[i * n + k] * B[j * n + k];
            }
            A[i * n + j] = sum;
            A[j * n + i] = sum;
        }
    }
    
    // Add diagonal dominance to ensure positive definiteness
    #pragma omp parallel for num_threads(threadCount(n)) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        A[i * n + i] += n;
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix
    
    std::vector<double> reconstructed(n * n);
    
    // Compute L * L^T
    #pragma omp parallel for num_threads(threadCount(n)) schedule(static)
    for (size_t i = 0; i < n; ++i) {
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
    
    #pragma omp parallel for num_threads(threadCount(n)) reduction(max:maxError,relError) schedule(static)
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
