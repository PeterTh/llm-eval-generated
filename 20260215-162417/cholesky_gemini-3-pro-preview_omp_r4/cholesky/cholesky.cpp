#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <omp.h>
#include "../common/results_output.hpp"

// Simple Cholesky decomposition (sequential, unblocked algorithm)
// Decomposes positive definite matrix A into L * L^T where L is lower triangular

// Blocked Cholesky decomposition
bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    const size_t BLOCK_SIZE = 64; // Adjust based on cache size

    // Iterate over blocks
    for (size_t k = 0; k < n; k += BLOCK_SIZE) {
        const size_t kb = std::min(n - k, BLOCK_SIZE); // Current block size
        
        // 1. Factorize diagonal block A[k:k+kb][k:k+kb]
        // This must be done sequentially because of dependencies within the block
        for (size_t j = 0; j < kb; ++j) {
            size_t global_j = k + j;
            
            // Diagonal element
            double sum = 0.0;
            for (size_t l = 0; l < j; ++l) {
                sum += A[global_j * n + (k + l)] * A[global_j * n + (k + l)];
            }
            
            double val = A[global_j * n + global_j] - sum;
            if (val <= 0.0) {
                 // Error handling
                 A[global_j * n + global_j] = 1.0; // avoid nan
            } else {
                A[global_j * n + global_j] = sqrt(val);
            }
            
            double diag = A[global_j * n + global_j];
            double invDiag = 1.0 / diag;
            
            // Off-diagonal in the block (column j, rows i > j)
            for (size_t i = j + 1; i < kb; ++i) {
                size_t global_i = k + i;
                double inner_sum = 0.0;
                for (size_t l = 0; l < j; ++l) {
                    inner_sum += A[global_i * n + (k + l)] * A[global_j * n + (k + l)];
                }
                A[global_i * n + global_j] = (A[global_i * n + global_j] - inner_sum) * invDiag;
            }
        }
        
        // 2. Update panel (off-diagonal blocks in column k)
        // For rows i > k+kb, update columns j in k..k+kb
        // A[i][j] = (A[i][j] - sum(A[i][k..k+j-1] * A[j][k..k+j-1])) / A[j][j]
        // Note: A[i][j] already has contributions from previous blocks subtracted.
        // We need to subtract contributions from current block columns 0..j-1.
        
        // We can parallelize over i (rows below the diagonal block)
        #pragma omp parallel for schedule(static)
        for (size_t i = k + kb; i < n; ++i) {
            for (size_t j = 0; j < kb; ++j) {
                size_t global_j = k + j;
                double sum = 0.0;
                // Dot product with previously computed columns in this block
                for (size_t l = 0; l < j; ++l) {
                     sum += A[i * n + (k + l)] * A[global_j * n + (k + l)];
                }
                A[i * n + global_j] = (A[i * n + global_j] - sum) / A[global_j * n + global_j];
            }
        }

        // 3. Update trailing submatrix
        // Update A[i][j] for i, j >= k+kb
        // A[i][j] -= A[i][k..k+kb] * A[j][k..k+kb]^T
        // Only lower triangle: i >= j
        
        #pragma omp parallel for schedule(dynamic)
        for (size_t i = k + kb; i < n; ++i) {
             // For each row i, update columns j from k+kb to i
             for (size_t j = k + kb; j <= i; ++j) {
                 double sum = 0.0;
                 // Vectorize dot product of length kb
                 #pragma omp simd reduction(+:sum)
                 for (size_t l = 0; l < kb; ++l) {
                     sum += A[i * n + (k + l)] * A[j * n + (k + l)];
                 }
                 A[i * n + j] -= sum;
             }
        }
    }
    
    // Zero out upper triangular part
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            A[i * n + j] = 0.0;
        }
    }
    
    return true;
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
