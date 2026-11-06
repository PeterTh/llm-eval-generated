#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <omp.h>

#include "../common/results_output.hpp"

// Limit small matrices to a useful number of workers; the OpenMP runtime's
// maximum still controls the number used for larger problems.
static int workerCount(const size_t n) {
    return static_cast<int>(std::min<size_t>(omp_get_max_threads(),
                                             std::max<size_t>(1, (n + 63) / 64)));
}

// Right-looking blocked Cholesky. The lower triangle of each trailing block
// is updated before it becomes part of the diagonal block or panel.
bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    constexpr size_t blockSize = 64;
    bool success = true;
    size_t failedDiagonal = 0;
    double* const matrix = A.data();

    #pragma omp parallel num_threads(workerCount(n)) shared(success, failedDiagonal)
    {
        for (size_t block = 0; block < n; block += blockSize) {
            const size_t end = std::min(n, block + blockSize);

            #pragma omp single
            {
                for (size_t j = block; j < end; ++j) {
                    double* const row = matrix + j * n;
                    double sum = 0.0;
                    #pragma omp simd reduction(+:sum)
                    for (size_t k = block; k < j; ++k)
                        sum += row[k] * row[k];
                    const double val = row[j] - sum;
                    if (val <= 0.0) {
                        success = false;
                        failedDiagonal = j;
                        break;
                    }
                    row[j] = std::sqrt(val);
                    for (size_t i = j + 1; i < end; ++i) {
                        double* const next = matrix + i * n;
                        double dot = 0.0;
                        #pragma omp simd reduction(+:dot)
                        for (size_t k = block; k < j; ++k)
                            dot += next[k] * row[k];
                        next[j] = (next[j] - dot) / row[j];
                    }
                }
            }
            // The single region's barrier publishes failure to every worker.
            if (!success) break;

            #pragma omp for schedule(static)
            for (size_t i = end; i < n; ++i) {
                double* const row = matrix + i * n;
                for (size_t j = block; j < end; ++j) {
                    const double* const diagonalRow = matrix + j * n;
                    double dot = 0.0;
                    #pragma omp simd reduction(+:dot)
                    for (size_t k = block; k < j; ++k)
                        dot += row[k] * diagonalRow[k];
                    row[j] = (row[j] - dot) / diagonalRow[j];
                }
            }

            #pragma omp for schedule(dynamic, 4)
            for (size_t i = end; i < n; ++i) {
                double* const row = matrix + i * n;
                size_t j = end;
                // Four independent products reuse the panel row and expose
                // enough independent accumulators for vector execution.
                for (; j + 3 <= i; j += 4) {
                    const double* const r0 = matrix + j * n;
                    const double* const r1 = r0 + n;
                    const double* const r2 = r1 + n;
                    const double* const r3 = r2 + n;
                    double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
                    #pragma omp simd reduction(+:s0,s1,s2,s3)
                    for (size_t k = block; k < end; ++k) {
                        const double x = row[k];
                        s0 += x * r0[k];
                        s1 += x * r1[k];
                        s2 += x * r2[k];
                        s3 += x * r3[k];
                    }
                    row[j] -= s0;
                    row[j + 1] -= s1;
                    row[j + 2] -= s2;
                    row[j + 3] -= s3;
                }
                for (; j <= i; ++j) {
                    const double* const other = matrix + j * n;
                    double dot = 0.0;
                    #pragma omp simd reduction(+:dot)
                    for (size_t k = block; k < end; ++k)
                        dot += row[k] * other[k];
                    row[j] -= dot;
                }
            }
        }

        if (success) {
            #pragma omp for schedule(static)
            for (size_t i = 0; i < n; ++i)
                std::fill(matrix + i * n + i + 1, matrix + (i + 1) * n, 0.0);
        }
    }
    if (!success)
        printf("Error: Matrix is not positive definite at diagonal element %zu\n", failedDiagonal);
    return success;
}

// Generate the same seeded random matrix as the original implementation.
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < n * n; ++i)
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;

    #pragma omp parallel for num_threads(workerCount(n)) schedule(dynamic, 4)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (size_t k = 0; k < n; ++k)
                sum += B[i * n + k] * B[j * n + k];
            A[i * n + j] = sum;
            A[j * n + i] = sum;
        }
        A[i * n + i] += n;
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    double maxError = 0.0;
    double relError = 0.0;
    #pragma omp parallel for num_threads(workerCount(n)) schedule(dynamic, 4) reduction(max:maxError,relError)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (size_t k = 0; k < n; ++k)
                sum += L[i * n + k] * L[j * n + k];
            const double error = std::fabs(sum - A_orig[i * n + j]);
            maxError = std::max(maxError, error);
            relError = std::max(relError, error / (std::fabs(A_orig[i * n + j]) + 1e-10));
            // Compare both halves, including the original upper triangle.
            const double upperError = std::fabs(sum - A_orig[j * n + i]);
            maxError = std::max(maxError, upperError);
            relError = std::max(relError, upperError / (std::fabs(A_orig[j * n + i]) + 1e-10));
        }
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
