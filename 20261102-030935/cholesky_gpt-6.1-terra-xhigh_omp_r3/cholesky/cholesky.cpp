#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <omp.h>

#include "../common/results_output.hpp"

// Avoid creating more workers than there are useful chunks of rows. Respect
// the OpenMP runtime's configured maximum when it is lower.
static int threadCount(size_t n) {
    return static_cast<int>(std::min<size_t>(omp_get_max_threads(), std::max<size_t>(1, n / 32)));
}

// Compute one diagonal block at a time. Once that block is complete, each
// remaining row of the block's column panel can be solved independently.
bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    constexpr size_t blockSize = 32;
    bool success = true;
    double* const data = A.data();

    #pragma omp parallel num_threads(threadCount(n)) shared(success)
    {
        for (size_t block = 0; block < n; block += blockSize) {
            const size_t end = std::min(block + blockSize, n);
            #pragma omp single
            {
                for (size_t i = block; i < end && success; ++i) {
                    double* const row = data + i * n;
                    for (size_t j = block; j <= i; ++j) {
                        const double* const pivot = data + j * n;
                        double sum = 0.0;
                        #pragma omp simd reduction(+:sum)
                        for (size_t k = 0; k < j; ++k)
                            sum += row[k] * pivot[k];
                        if (i == j) {
                            const double val = row[j] - sum;
                            if (val <= 0.0) {
                                printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
                                success = false;
                                break;
                            }
                            row[j] = sqrt(val);
                        } else {
                            row[j] = (row[j] - sum) / pivot[j];
                        }
                    }
                }
            }
            // The single region's barrier publishes the diagonal block and
            // ensures every thread takes the same path on failure.
            if (!success) break;
            #pragma omp for schedule(static)
            for (size_t i = end; i < n; ++i) {
                double* const row = data + i * n;
                for (size_t j = block; j < end; ++j) {
                    const double* const pivot = data + j * n;
                    double sum = 0.0;
                    #pragma omp simd reduction(+:sum)
                    for (size_t k = 0; k < j; ++k)
                        sum += row[k] * pivot[k];
                    row[j] = (row[j] - sum) / pivot[j];
                }
            }
        }
        if (success) {
            #pragma omp for schedule(static)
            for (size_t i = 0; i < n; ++i)
                std::fill(data + i * n + i + 1, data + (i + 1) * n, 0.0);
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
    
    // The two halves are identical; compute each dot product only once.
    #pragma omp parallel for num_threads(threadCount(n)) schedule(dynamic, 4)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (size_t k = 0; k < n; ++k)
                sum += B[i * n + k] * B[j * n + k];
            A[i * n + j] = sum;
            if (i != j) A[j * n + i] = sum;
        }
        A[i * n + i] += n;
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Compare each reconstructed element directly, without an n-by-n temporary.
    double maxError = 0.0;
    double relError = 0.0;
    #pragma omp parallel for num_threads(threadCount(n)) schedule(static) reduction(max:maxError,relError)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (size_t k = 0; k < n; ++k)
                sum += L[i * n + k] * L[j * n + k];
            const double original = A_orig[i * n + j];
            const double error = fabs(sum - original);
            maxError = std::max(maxError, error);
            relError = std::max(relError, error / (fabs(original) + 1e-10));
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
