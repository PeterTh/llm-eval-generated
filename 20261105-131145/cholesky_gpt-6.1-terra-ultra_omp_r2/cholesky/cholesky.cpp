#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <omp.h>

#include "../common/results_output.hpp"

// Factor a block at a time, updating the lower triangular trailing matrix.
// Each panel and trailing update is independent across rows.
bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    constexpr size_t blockSize = 64;
    const int threads = static_cast<int>(std::min<size_t>(
        omp_get_max_threads(), std::max<size_t>(1, (n + 31) / 32)));
    bool success = true;

    #pragma omp parallel num_threads(threads) shared(success)
    {
        for (size_t begin = 0; begin < n; begin += blockSize) {
            const size_t end = std::min(begin + blockSize, n);

            #pragma omp single
            {
                for (size_t i = begin; i < end && success; ++i) {
                    for (size_t j = begin; j <= i; ++j) {
                        double sum = 0.0;
                        #pragma omp simd reduction(+:sum)
                        for (size_t k = begin; k < j; ++k)
                            sum += A[i * n + k] * A[j * n + k];
                        if (i == j) {
                            const double val = A[i * n + i] - sum;
                            if (val <= 0.0) {
                                printf("Error: Matrix is not positive definite at diagonal element %zu\n", i);
                                success = false;
                                break;
                            }
                            A[i * n + i] = sqrt(val);
                        } else {
                            A[i * n + j] = (A[i * n + j] - sum) / A[j * n + j];
                        }
                    }
                }
            }
            // The single region's barrier makes the failure and panel visible.
            if (!success) break;

            #pragma omp for schedule(static)
            for (size_t i = end; i < n; ++i) {
                for (size_t j = begin; j < end; ++j) {
                    double sum = 0.0;
                    #pragma omp simd reduction(+:sum)
                    for (size_t k = begin; k < j; ++k)
                        sum += A[i * n + k] * A[j * n + k];
                    A[i * n + j] = (A[i * n + j] - sum) / A[j * n + j];
                }
            }

            #pragma omp for schedule(static, 1)
            for (size_t i = end; i < n; ++i) {
                size_t j = end;
                // Reuse the current row while updating four columns at once.
                for (; j + 3 <= i; j += 4) {
                    double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
                    #pragma omp simd reduction(+:s0,s1,s2,s3)
                    for (size_t k = begin; k < end; ++k) {
                        const double x = A[i * n + k];
                        s0 += x * A[j * n + k];
                        s1 += x * A[(j + 1) * n + k];
                        s2 += x * A[(j + 2) * n + k];
                        s3 += x * A[(j + 3) * n + k];
                    }
                    A[i * n + j] -= s0;
                    A[i * n + j + 1] -= s1;
                    A[i * n + j + 2] -= s2;
                    A[i * n + j + 3] -= s3;
                }
                for (; j <= i; ++j) {
                    double sum = 0.0;
                    #pragma omp simd reduction(+:sum)
                    for (size_t k = begin; k < end; ++k)
                        sum += A[i * n + k] * A[j * n + k];
                    A[i * n + j] -= sum;
                }
            }
        }

        if (success) {
            #pragma omp for schedule(static)
            for (size_t i = 0; i < n; ++i)
                std::fill(A.begin() + i * n + i + 1, A.begin() + (i + 1) * n, 0.0);
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
    
    // Each dot product is symmetric, so compute one triangle and mirror it.
    const int threads = static_cast<int>(std::min<size_t>(
        omp_get_max_threads(), std::max<size_t>(1, (n + 31) / 32)));
    #pragma omp parallel for schedule(dynamic, 1) num_threads(threads)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (size_t k = 0; k < n; ++k)
                sum += B[i * n + k] * B[j * n + k];
            if (i == j) sum += n;
            A[i * n + j] = sum;
            A[j * n + i] = sum;
        }
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Compare each reconstructed element with the corresponding input.
    double maxError = 0.0;
    double relError = 0.0;
    const int threads = static_cast<int>(std::min<size_t>(
        omp_get_max_threads(), std::max<size_t>(1, (n + 31) / 32)));
    #pragma omp parallel for schedule(static) num_threads(threads) reduction(max:maxError,relError)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (size_t k = 0; k < n; ++k)
                sum += L[i * n + k] * L[j * n + k];
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
