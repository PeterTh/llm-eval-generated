#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <omp.h>

#include "../common/results_output.hpp"

// Keep small matrices from creating a machine-sized team by default. An
// explicit OpenMP thread count remains under the user's control.
int threadCount(const size_t n) {
    const int maxThreads = omp_get_max_threads();
    if (std::getenv("OMP_NUM_THREADS")) return maxThreads;
    const size_t usefulThreads = std::max<size_t>(1, (n + 31) / 32);
    return static_cast<int>(std::min<size_t>(maxThreads, usefulThreads));
}

// Blocked, right-looking Cholesky decomposition of a row-major matrix.
bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    constexpr size_t blockSize = 64;
    bool success = true;
    double* const a = A.data();

    #pragma omp parallel shared(success, a) num_threads(threadCount(n))
    {
        for (size_t begin = 0; begin < n; begin += blockSize) {
            const size_t end = std::min(begin + blockSize, n);

            // Previous blocks have already updated this diagonal block.
            #pragma omp single
            {
                for (size_t i = begin; i < end && success; ++i) {
                    for (size_t j = begin; j <= i; ++j) {
                        double sum = 0.0;
                        #pragma omp simd reduction(+:sum)
                        for (size_t k = begin; k < j; ++k) {
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
                }
            }
            // The implicit barrier publishes both the block and success.
            if (!success) break;

            // Solve each row of the panel against the diagonal block.
            #pragma omp for schedule(static)
            for (size_t i = end; i < n; ++i) {
                for (size_t j = begin; j < end; ++j) {
                    double sum = 0.0;
                    #pragma omp simd reduction(+:sum)
                    for (size_t k = begin; k < j; ++k) {
                        sum += a[i * n + k] * a[j * n + k];
                    }
                    a[i * n + j] = (a[i * n + j] - sum) / a[j * n + j];
                }
            }

            // Each iteration owns one row of the lower trailing triangle.
            #pragma omp for schedule(dynamic, 4)
            for (size_t i = end; i < n; ++i) {
                for (size_t j = end; j <= i; ++j) {
                    double sum = 0.0;
                    #pragma omp simd reduction(+:sum)
                    for (size_t k = begin; k < end; ++k) {
                        sum += a[i * n + k] * a[j * n + k];
                    }
                    a[i * n + j] -= sum;
                }
            }
        }

        if (success) {
            #pragma omp for schedule(static)
            for (size_t i = 0; i < n; ++i) {
                std::fill(a + i * n + i + 1, a + (i + 1) * n, 0.0);
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
    
    // Each pair of rows has a single owner; mirror the symmetric result.
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
    // Reconstruct and compare each element independently.
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
            relError = std::max(relError, error / (fabs(A_orig[i * n + j]) + 1e-10));
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
