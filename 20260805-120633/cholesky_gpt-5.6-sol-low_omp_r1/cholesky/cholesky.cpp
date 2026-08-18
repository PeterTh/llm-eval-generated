#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <omp.h>

#include "../common/results_output.hpp"

// Cache-blocked Cholesky decomposition.  Work on the block diagonal is
// necessarily ordered, but the block solves and trailing updates are
// independent and are distributed among the OpenMP team.
// Decomposes positive definite matrix A into L * L^T where L is lower triangular

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    constexpr size_t blockSize = 64;
    bool positiveDefinite = true;
    size_t badDiagonal = 0;

#pragma omp parallel shared(positiveDefinite, badDiagonal)
    {
        for (size_t jb = 0; jb < n; jb += blockSize) {
            const size_t je = std::min(jb + blockSize, n);

            // Factor the diagonal block after all earlier block updates.
#pragma omp single
            {
                if (positiveDefinite) {
                    for (size_t j = jb; j < je; ++j) {
                        double sum = 0.0;
#pragma omp simd reduction(+:sum)
                        for (size_t p = jb; p < j; ++p)
                            sum += A[j * n + p] * A[j * n + p];
                        const double value = A[j * n + j] - sum;
                        if (value <= 0.0 || !std::isfinite(value)) {
                            positiveDefinite = false;
                            badDiagonal = j;
                            break;
                        }
                        A[j * n + j] = std::sqrt(value);
                        for (size_t i = j + 1; i < je; ++i) {
                            double dot = 0.0;
#pragma omp simd reduction(+:dot)
                            for (size_t p = jb; p < j; ++p)
                                dot += A[i * n + p] * A[j * n + p];
                            A[i * n + j] = (A[i * n + j] - dot) / A[j * n + j];
                        }
                    }
                }
            }

            // Solve the rectangular panel below the diagonal block.
#pragma omp for schedule(static)
            for (size_t i = je; i < n; ++i) {
                if (positiveDefinite) {
                    for (size_t j = jb; j < je; ++j) {
                        double dot = 0.0;
#pragma omp simd reduction(+:dot)
                        for (size_t p = jb; p < j; ++p)
                            dot += A[i * n + p] * A[j * n + p];
                        A[i * n + j] = (A[i * n + j] - dot) / A[j * n + j];
                    }
                }
            }

            // Update every lower-triangular tile exactly once.  Each thread
            // owns complete output tiles, avoiding synchronization and races.
            const size_t first = (je + blockSize - 1) / blockSize;
            const size_t blocks = (n + blockSize - 1) / blockSize;
#pragma omp for schedule(dynamic, 1)
            for (size_t pair = 0; pair < blocks * blocks; ++pair) {
                const size_t bi = pair / blocks;
                const size_t bk = pair % blocks;
                if (!positiveDefinite || bi < first || bk < first || bk > bi)
                    continue;
                const size_t ib = bi * blockSize;
                const size_t ie = std::min(ib + blockSize, n);
                const size_t kb = bk * blockSize;
                const size_t ke = std::min(kb + blockSize, n);
                for (size_t i = ib; i < ie; ++i) {
                    const size_t kend = (bi == bk) ? std::min(ke, i + 1) : ke;
                    for (size_t k = kb; k < kend; ++k) {
                        double dot = 0.0;
#pragma omp simd reduction(+:dot)
                        for (size_t p = jb; p < je; ++p)
                            dot += A[i * n + p] * A[k * n + p];
                        A[i * n + k] -= dot;
                    }
                }
            }
        }

#pragma omp for schedule(static)
        for (size_t i = 0; i < n; ++i)
            std::fill(A.begin() + i * n + i + 1, A.begin() + (i + 1) * n, 0.0);
    }

    if (!positiveDefinite)
        printf("Error: Matrix is not positive definite at diagonal element %zu\n", badDiagonal);
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
#pragma omp parallel for schedule(static)
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
