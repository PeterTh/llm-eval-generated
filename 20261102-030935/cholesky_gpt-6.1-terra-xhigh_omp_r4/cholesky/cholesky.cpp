#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <omp.h>

#include "../common/results_output.hpp"

// Keep synchronization costs proportional to the amount of work while
// respecting the thread limit chosen through the OpenMP runtime.
static int threadCount(const size_t n) {
    const size_t usefulThreads = std::max(size_t{1}, (n / 128) + (n % 128 != 0));
    return static_cast<int>(std::min(usefulThreads, static_cast<size_t>(omp_get_max_threads())));
}

// Blocked, right-looking Cholesky decomposition of a row-major matrix.
// The diagonal block must be factored in order; rows of the panel and
// individual rows of the trailing matrix are independent within each phase.
bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    constexpr size_t blockSize = 64;
    double* const a = A.data();
    bool success = true;

    #pragma omp parallel shared(success) num_threads(threadCount(n))
    {
        for (size_t base = 0; base < n; base += blockSize) {
            const size_t end = std::min(base + blockSize, n);

            #pragma omp single
            {
                for (size_t i = base; i < end; ++i) {
                    for (size_t j = base; j <= i; ++j) {
                        double sum = 0.0;
                        #pragma omp simd reduction(+:sum)
                        for (size_t k = base; k < j; ++k) {
                            sum += a[i * n + k] * a[j * n + k];
                        }
                        if (i == j) {
                            const double val = a[i * n + i] - sum;
                            if (val <= 0.0) {
                                printf("Error: Matrix is not positive definite at diagonal element %zu\n", i);
                                success = false;
                                break;
                            }
                            a[i * n + i] = sqrt(val);
                        } else {
                            a[i * n + j] = (a[i * n + j] - sum) / a[j * n + j];
                        }
                    }
                    if (!success) break;
                    std::fill(a + i * n + i + 1, a + (i + 1) * n, 0.0);
                }
            }
            // The single construct's barrier makes the status and diagonal
            // block visible to every thread before the panel solve.
            if (!success) break;

            #pragma omp for schedule(static)
            for (size_t i = end; i < n; ++i) {
                for (size_t j = base; j < end; ++j) {
                    double sum = 0.0;
                    #pragma omp simd reduction(+:sum)
                    for (size_t k = base; k < j; ++k) {
                        sum += a[i * n + k] * a[j * n + k];
                    }
                    a[i * n + j] = (a[i * n + j] - sum) / a[j * n + j];
                }
            }

            #pragma omp for schedule(dynamic, 4)
            for (size_t i = end; i < n; ++i) {
                const double* const row = a + i * n;
                size_t j = end;
                // Accumulate four dot products together to reuse the panel row.
                for (; j + 3 <= i; j += 4) {
                    const double* const r0 = a + j * n;
                    const double* const r1 = r0 + n;
                    const double* const r2 = r1 + n;
                    const double* const r3 = r2 + n;
                    double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
                    #pragma omp simd reduction(+:s0,s1,s2,s3)
                    for (size_t k = base; k < end; ++k) {
                        const double x = row[k];
                        s0 += x * r0[k];
                        s1 += x * r1[k];
                        s2 += x * r2[k];
                        s3 += x * r3[k];
                    }
                    a[i * n + j] -= s0;
                    a[i * n + j + 1] -= s1;
                    a[i * n + j + 2] -= s2;
                    a[i * n + j + 3] -= s3;
                }
                for (; j <= i; ++j) {
                    double sum = 0.0;
                    #pragma omp simd reduction(+:sum)
                    for (size_t k = base; k < end; ++k) {
                        sum += row[k] * a[j * n + k];
                    }
                    a[i * n + j] -= sum;
                }
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
    
    // Compute one triangle of A = B * B^T and mirror it.
    #pragma omp parallel for schedule(dynamic, 4) num_threads(threadCount(n))
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (size_t k = 0; k < n; ++k) {
                sum += B[i * n + k] * B[j * n + k];
            }
            A[i * n + j] = sum;
            if (i != j) A[j * n + i] = sum;
        }
        A[i * n + i] += n;
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Reconstruct and compare each entry independently.
    double maxError = 0.0;
    double relError = 0.0;
    #pragma omp parallel for reduction(max:maxError,relError) schedule(static) num_threads(threadCount(n))
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (size_t k = 0; k < n; ++k) {
                sum += L[i * n + k] * L[j * n + k];
            }
            const double error = fabs(sum - A_orig[i * n + j]);
            maxError = std::max(maxError, error);
            const double rel = error / (fabs(A_orig[i * n + j]) + 1e-10);
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
