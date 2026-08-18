#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// Blocked, right-looking Cholesky decomposition.  Factoring a diagonal block is
// necessarily serial, but the panel solve and the trailing rank-k update expose
// the bulk of the work to all OpenMP threads.
// Decomposes positive definite matrix A into L * L^T where L is lower triangular.

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    constexpr size_t blockSize = 64;

    // A is stored in row-major order.  Only the lower triangle is updated until
    // the end, so the original upper triangle remains available but is ignored.
    for (size_t block = 0; block < n; block += blockSize) {
        const size_t width = std::min(blockSize, n - block);
        const size_t trailing = block + width;

        // Factor the current diagonal block.  Its prior contributions have
        // already been removed by preceding trailing updates.
        for (size_t col = 0; col < width; ++col) {
            const size_t j = block + col;
            double diagonal = A[j * n + j];
            for (size_t k = 0; k < col; ++k) {
                const double value = A[j * n + block + k];
                diagonal -= value * value;
            }
            if (diagonal <= 0.0) {
                printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
                return false;
            }
            A[j * n + j] = sqrt(diagonal);

            for (size_t row = col + 1; row < width; ++row) {
                const size_t i = block + row;
                double value = A[i * n + j];
                for (size_t k = 0; k < col; ++k) {
                    value -= A[i * n + block + k] * A[j * n + block + k];
                }
                A[i * n + j] = value / A[j * n + j];
            }
        }

        // Solve L_ik * L_kk^T = A_ik.  Each trailing row is independent.
        #pragma omp parallel for schedule(static)
        for (std::ptrdiff_t row = static_cast<std::ptrdiff_t>(trailing);
             row < static_cast<std::ptrdiff_t>(n); ++row) {
            double* const rowData = A.data() + static_cast<size_t>(row) * n;
            for (size_t col = 0; col < width; ++col) {
                double value = rowData[block + col];
                const double* const diagonalRow = A.data() + (block + col) * n + block;
                #pragma omp simd reduction(-:value)
                for (size_t k = 0; k < col; ++k) {
                    value -= rowData[block + k] * diagonalRow[k];
                }
                rowData[block + col] = value / diagonalRow[col];
            }
        }

        // A_trailing <- A_trailing - L_trailing,k * L_trailing,k^T.
        // Tiling gives the scheduler enough independent, cache-friendly work
        // even near the end of the factorization.
        const std::ptrdiff_t firstTile = static_cast<std::ptrdiff_t>(trailing);
        const std::ptrdiff_t lastTile = static_cast<std::ptrdiff_t>(n);
        const std::ptrdiff_t tile = static_cast<std::ptrdiff_t>(blockSize);
        #pragma omp parallel for collapse(2) schedule(dynamic)
        for (std::ptrdiff_t ii = firstTile; ii < lastTile; ii += tile) {
            for (std::ptrdiff_t jj = firstTile; jj < lastTile; jj += tile) {
                if (jj > ii) {
                    continue;
                }
                const size_t iEnd = std::min(static_cast<size_t>(ii + tile), n);
                const size_t jEnd = std::min(static_cast<size_t>(jj + tile), n);
                for (size_t i = static_cast<size_t>(ii); i < iEnd; ++i) {
                    double* const rowI = A.data() + i * n;
                    const size_t endJ = std::min(jEnd, i + 1);
                    for (size_t j = static_cast<size_t>(jj); j < endJ; ++j) {
                        const double* const rowJ = A.data() + j * n + block;
                        double update = 0.0;
                        #pragma omp simd reduction(+:update)
                        for (size_t k = 0; k < width; ++k) {
                            update += rowI[block + k] * rowJ[k];
                        }
                        rowI[j] -= update;
                    }
                }
            }
        }
    }

    // Preserve the original result representation: L in the lower triangle and
    // explicit zeros above it.
    #pragma omp parallel for schedule(static)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
        double* const row = A.data() + static_cast<size_t>(i) * n;
        for (size_t j = static_cast<size_t>(i) + 1; j < n; ++j) {
            row[j] = 0.0;
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
