#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <omp.h>

#include "../common/results_output.hpp"

// Decomposes positive definite matrix A into L * L^T where L is lower triangular

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    if (n == 0) {
        return true;
    }

    // A tile is large enough to amortize OpenMP scheduling and small enough
    // to keep the panel and update operands in cache.
    constexpr size_t blockSize = 128;
    bool success = true;

    #pragma omp parallel shared(A, success)
    {
        for (size_t k = 0; k < n; k += blockSize) {
            const size_t kb = std::min(blockSize, n - k);

            // The diagonal tile has an intrinsic dependency chain.  Keeping
            // this small factorization in one thread also makes the failure
            // check deterministic.
            #pragma omp single
            {
                for (size_t jj = 0; jj < kb && success; ++jj) {
                    const size_t j = k + jj;
                    double diagonal = A[j * n + j];
                    #pragma omp simd reduction(-: diagonal)
                    for (size_t p = 0; p < jj; ++p) {
                        const double value = A[j * n + k + p];
                        diagonal -= value * value;
                    }
                    if (diagonal <= 0.0) {
                        printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
                        success = false;
                        break;
                    }
                    A[j * n + j] = sqrt(diagonal);

                    for (size_t ii = jj + 1; ii < kb; ++ii) {
                        const size_t i = k + ii;
                        double value = A[i * n + j];
                        for (size_t p = 0; p < jj; ++p) {
                            value -= A[i * n + k + p] * A[j * n + k + p];
                        }
                        A[i * n + j] = value / A[j * n + j];
                    }
                }
            }
            #pragma omp barrier

            if (success) {
                // Solve each tile of the panel independently: X L(k,k)^T = A.
                #pragma omp for schedule(static)
                for (size_t i = k + kb; i < n; i += blockSize) {
                    const size_t ib = std::min(blockSize, n - i);
                    for (size_t jj = 0; jj < kb; ++jj) {
                        const size_t j = k + jj;
                        const double diagonal = A[j * n + j];
                        for (size_t row = 0; row < ib; ++row) {
                            double value = A[(i + row) * n + j];
                            for (size_t p = 0; p < jj; ++p) {
                                value -= A[(i + row) * n + k + p] *
                                         A[j * n + k + p];
                            }
                            A[(i + row) * n + j] = value / diagonal;
                        }
                    }
                }
                #pragma omp barrier

                // Update only the lower triangle.  Every pair of output
                // tiles is disjoint, so no locks or atomics are needed.
                #pragma omp for schedule(dynamic, 1)
                for (size_t i = k + kb; i < n; i += blockSize) {
                    for (size_t j = k + kb; j <= i; j += blockSize) {
                        const size_t ib = std::min(blockSize, n - i);
                        const size_t jb = std::min(blockSize, n - j);
                        for (size_t row = 0; row < ib; ++row) {
                            const size_t maxCol = (i == j) ? row + 1 : jb;
                            for (size_t col = 0; col < maxCol; ++col) {
                                double value = A[(i + row) * n + (j + col)];
                                #pragma omp simd reduction(-: value)
                                for (size_t p = 0; p < kb; ++p) {
                                    value -= A[(i + row) * n + k + p] *
                                             A[(j + col) * n + k + p];
                                }
                                A[(i + row) * n + (j + col)] = value;
                            }
                        }
                    }
                }
                #pragma omp barrier
            } else {
                // Keep all threads at the same loop barrier if a bad matrix
                // is encountered, then leave the parallel region together.
                #pragma omp barrier
            }
        }
    }

    // The upper triangle is not used by the blocked algorithm, but clearing
    // it retains the original function's exact output contract.
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            A[i * n + j] = 0.0;
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
