#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// Blocked, right-looking Cholesky decomposition.
// Decomposes positive definite matrix A into L * L^T where L is lower triangular

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    // A block fits comfortably in private caches while leaving enough rows for
    // useful parallel work.  A single parallel region avoids a fork/join at
    // every panel.
    constexpr size_t blockSize = 64;
    bool positiveDefinite = true;

    #pragma omp parallel default(none) shared(A, n, positiveDefinite)
    {
        for (size_t block = 0; block < n; block += blockSize) {
            const size_t blockEnd = std::min(block + blockSize, n);

            // Factor the small diagonal block. Its columns are inherently
            // dependent, so one thread performs this cache-resident work.
            #pragma omp single
            {
                if (positiveDefinite) {
                    for (size_t j = block; j < blockEnd; ++j) {
                        double sum = 0.0;
                        #pragma omp simd reduction(+:sum)
                        for (size_t k = block; k < j; ++k) {
                            sum += A[j * n + k] * A[j * n + k];
                        }

                        const double value = A[j * n + j] - sum;
                        if (value <= 0.0) {
                            printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
                            positiveDefinite = false;
                            break;
                        }
                        A[j * n + j] = std::sqrt(value);

                        for (size_t i = j + 1; i < blockEnd; ++i) {
                            double product = 0.0;
                            #pragma omp simd reduction(+:product)
                            for (size_t k = block; k < j; ++k) {
                                product += A[i * n + k] * A[j * n + k];
                            }
                            A[i * n + j] = (A[i * n + j] - product) / A[j * n + j];
                        }
                    }
                }
            }

            if (positiveDefinite) {
                // Triangular solve for the panel below the diagonal block.
                // Rows are independent and contiguous in row-major storage.
                #pragma omp for schedule(static)
                for (size_t i = blockEnd; i < n; ++i) {
                    for (size_t j = block; j < blockEnd; ++j) {
                        double product = 0.0;
                        #pragma omp simd reduction(+:product)
                        for (size_t k = block; k < j; ++k) {
                            product += A[i * n + k] * A[j * n + k];
                        }
                        A[i * n + j] = (A[i * n + j] - product) / A[j * n + j];
                    }
                }

                // Symmetric rank-k update of the remaining lower triangle.
                // Dynamic row chunks balance the triangular iteration space.
                #pragma omp for schedule(dynamic, 4)
                for (size_t i = blockEnd; i < n; ++i) {
                    for (size_t j = blockEnd; j <= i; ++j) {
                        double product = 0.0;
                        #pragma omp simd reduction(+:product)
                        for (size_t k = block; k < blockEnd; ++k) {
                            product += A[i * n + k] * A[j * n + k];
                        }
                        A[i * n + j] -= product;
                    }
                }
            }
        }

        // The factor is returned in the same lower-triangular representation
        // as the original implementation.
        #pragma omp for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            std::fill(A.begin() + i * n + i + 1, A.begin() + (i + 1) * n, 0.0);
        }
    }

    return positiveDefinite;
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
    #pragma omp parallel for schedule(static)
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
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        A[i * n + i] += n;
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix
    
    double maxError = 0.0;
    double relError = 0.0;

    // Reconstruct and compare without allocating another n-by-n matrix.
    #pragma omp parallel for schedule(static) reduction(max:maxError, relError)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double reconstructed = 0.0;
            const size_t terms = std::min(i, j) + 1;
            #pragma omp simd reduction(+:reconstructed)
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
