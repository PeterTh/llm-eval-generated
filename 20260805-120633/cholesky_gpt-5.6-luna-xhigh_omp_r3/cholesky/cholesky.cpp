#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <omp.h>
#include <vector>

#include "../common/results_output.hpp"

// Cholesky decomposition using a cache-blocked right-looking algorithm.
// Decomposes positive definite matrix A into L * L^T where L is lower triangular

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    // A is stored in row-major order. The block size keeps the diagonal
    // factorization and the panel in cache while leaving enough independent
    // rows for the OpenMP team to process.
    constexpr size_t blockSize = 128;
    bool success = true;

    // Keep one OpenMP team alive for the complete factorization. The barriers
    // between the phases are the dependency boundaries of blocked Cholesky.
    #pragma omp parallel default(none) shared(A, n, success)
    {
        for (size_t blockStart = 0; blockStart < n; blockStart += blockSize) {
            const size_t blockEnd = std::min(blockStart + blockSize, n);

            // Factor the diagonal block. Columns are dependent, but the
            // entries below each diagonal element are independent.
            for (size_t j = blockStart; j < blockEnd; ++j) {
                #pragma omp single
                {
                    double sum = 0.0;
                    const double* rowJ = A.data() + j * n;
                    #pragma omp simd reduction(+:sum)
                    for (size_t k = blockStart; k < j; ++k) {
                        sum += rowJ[k] * rowJ[k];
                    }

                    const double val = rowJ[j] - sum;
                    if (val <= 0.0) {
                        printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
                        success = false;
                    } else {
                        A[j * n + j] = sqrt(val);
                    }
                }

                #pragma omp for schedule(static)
                for (size_t i = j + 1; i < blockEnd; ++i) {
                    if (success) {
                        double sum = 0.0;
                        double* rowI = A.data() + i * n;
                        const double* rowJ = A.data() + j * n;
                        #pragma omp simd reduction(+:sum)
                        for (size_t k = blockStart; k < j; ++k) {
                            sum += rowI[k] * rowJ[k];
                        }
                        rowI[j] = (rowI[j] - sum) / rowJ[j];
                    }
                }
            }

            // Solve the panel below the diagonal block. Each row is
            // independent and its columns must be handled from left to right.
            #pragma omp for schedule(static)
            for (size_t i = blockEnd; i < n; ++i) {
                if (success) {
                    double* rowI = A.data() + i * n;
                    for (size_t j = blockStart; j < blockEnd; ++j) {
                        double sum = 0.0;
                        const double* rowJ = A.data() + j * n;
                        #pragma omp simd reduction(+:sum)
                        for (size_t k = blockStart; k < j; ++k) {
                            sum += rowI[k] * rowJ[k];
                        }
                        rowI[j] = (rowI[j] - sum) / rowJ[j];
                    }
                }
            }

            // Update the trailing lower triangle with the newly computed
            // panel. Every (i,j) element is independent in this phase.
            #pragma omp for schedule(guided)
            for (size_t i = blockEnd; i < n; ++i) {
                if (success) {
                    double* rowI = A.data() + i * n;
                    for (size_t j = blockEnd; j <= i; ++j) {
                        double sum = 0.0;
                        const double* rowJ = A.data() + j * n;
                        #pragma omp simd reduction(+:sum)
                        for (size_t k = blockStart; k < blockEnd; ++k) {
                            sum += rowI[k] * rowJ[k];
                        }
                        rowI[j] -= sum;
                    }
                }
            }
        }

        // Zero the upper triangle after all numerical work has completed.
        #pragma omp for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            double* rowI = A.data() + i * n;
            for (size_t j = i + 1; j < n; ++j) {
                rowI[j] = 0.0;
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
    
    // Compute A = B * B^T. Keep the reduction order unchanged while
    // distributing independent output elements across the OpenMP team.
    #pragma omp parallel for collapse(2) schedule(static)
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
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        A[i * n + i] += n;
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix
    
    std::vector<double> reconstructed(n * n);
    
    // Compute L * L^T
    #pragma omp parallel for collapse(2) schedule(static)
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
