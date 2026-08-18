#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <omp.h>
#include <vector>

#include "../common/results_output.hpp"

// Blocked, right-looking Cholesky decomposition.  The panel factorization has
// loop-carried dependencies; the triangular solves and trailing updates expose
// the bulk of the work to all OpenMP threads.

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    constexpr size_t blockSize = 64;
    bool positiveDefinite = true;
    size_t failedDiagonal = 0;

#pragma omp parallel shared(A, positiveDefinite, failedDiagonal)
    {
        for (size_t block = 0; block < n; block += blockSize) {
            const size_t blockEnd = std::min(block + blockSize, n);

#pragma omp single
            {
                // Factor the current diagonal block in place.  Earlier trailing
                // updates have already been accumulated in this block.
                if (positiveDefinite) {
                    for (size_t row = block; row < blockEnd; ++row) {
                        double* const rowData = A.data() + row * n;
                        for (size_t col = block; col < row; ++col) {
                            double sum = rowData[col];
                            for (size_t k = block; k < col; ++k) {
                                sum -= rowData[k] * A[col * n + k];
                            }
                            rowData[col] = sum / A[col * n + col];
                        }

                        double diagonal = rowData[row];
                        for (size_t k = block; k < row; ++k) {
                            diagonal -= rowData[k] * rowData[k];
                        }
                        if (diagonal <= 0.0) {
                            positiveDefinite = false;
                            failedDiagonal = row;
                            break;
                        }
                        rowData[row] = sqrt(diagonal);
                    }
                }
            }

            if (!positiveDefinite) {
                continue;
            }

            // Solve the block column below the diagonal panel: L_ik = A_ik L_kk^-T.
#pragma omp for schedule(static)
            for (size_t row = blockEnd; row < n; ++row) {
                double* const rowData = A.data() + row * n;
                for (size_t col = block; col < blockEnd; ++col) {
                    double sum = rowData[col];
                    for (size_t k = block; k < col; ++k) {
                        sum -= rowData[k] * A[col * n + k];
                    }
                    rowData[col] = sum / A[col * n + col];
                }
            }

            // Update only the stored lower triangle of the trailing matrix.
#pragma omp for schedule(dynamic)
            for (size_t tileRow = blockEnd; tileRow < n; tileRow += blockSize) {
                const size_t rowEnd = std::min(tileRow + blockSize, n);
                for (size_t tileCol = blockEnd; tileCol <= tileRow; tileCol += blockSize) {
                    const size_t colEnd = std::min(tileCol + blockSize, n);
                    for (size_t row = tileRow; row < rowEnd; ++row) {
                        double* const rowData = A.data() + row * n;
                        const size_t lastCol = tileRow == tileCol ? std::min(row + 1, colEnd) : colEnd;
                        for (size_t col = tileCol; col < lastCol; ++col) {
                            double sum = 0.0;
#pragma omp simd reduction(+ : sum)
                            for (size_t k = block; k < blockEnd; ++k) {
                                sum += rowData[k] * A[col * n + k];
                            }
                            rowData[col] -= sum;
                        }
                    }
                }
            }
        }
    }

    if (!positiveDefinite) {
        printf("Error: Matrix is not positive definite at diagonal element %zu\n", failedDiagonal);
    }

    // Keep the externally visible representation identical to the original.
#pragma omp parallel for schedule(static)
    for (size_t row = 0; row < n; ++row) {
        for (size_t col = row + 1; col < n; ++col) {
            A[row * n + col] = 0.0;
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
    
    #pragma omp parallel for reduction(max : maxError, relError) schedule(static)
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
