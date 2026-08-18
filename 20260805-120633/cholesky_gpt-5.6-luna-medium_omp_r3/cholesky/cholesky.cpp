#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <omp.h>

#include "../common/results_output.hpp"

// Blocked right-looking Cholesky factorization.  A is row-major and is updated
// in place to contain L in its lower triangle.  The barriers between phases
// are the tile-level dependencies of the factorization; all work within a
// phase is independent and is distributed across the OpenMP team.
namespace {

constexpr size_t kBlockSize = 128;

bool factorDiagonalTile(std::vector<double>& A, const size_t n,
                        const size_t row, const size_t width) {
    for (size_t j = 0; j < width; ++j) {
        double sum = 0.0;
        const size_t diagonal = row + j;
        for (size_t k = 0; k < j; ++k) {
            const double value = A[diagonal * n + row + k];
            sum += value * value;
        }

        const double value = A[diagonal * n + diagonal] - sum;
        if (value <= 0.0) {
            return false;
        }
        A[diagonal * n + diagonal] = sqrt(value);

        for (size_t i = j + 1; i < width; ++i) {
            const size_t target = row + i;
            double product = 0.0;
            for (size_t k = 0; k < j; ++k) {
                product += A[target * n + row + k] *
                           A[diagonal * n + row + k];
            }
            A[target * n + diagonal] =
                (A[target * n + diagonal] - product) /
                A[diagonal * n + diagonal];
        }
    }
    return true;
}

void solveTileAgainstDiagonal(std::vector<double>& A, const size_t n,
                              const size_t row, const size_t diagonal,
                              const size_t height, const size_t width) {
    for (size_t j = 0; j < width; ++j) {
        const size_t column = diagonal + j;
        for (size_t i = 0; i < height; ++i) {
            const size_t target = row + i;
            double product = 0.0;
            for (size_t k = 0; k < j; ++k) {
                product += A[target * n + diagonal + k] *
                           A[(diagonal + j) * n + diagonal + k];
            }
            A[target * n + column] =
                (A[target * n + column] - product) /
                A[column * n + column];
        }
    }
}

void updateTile(std::vector<double>& A, const size_t n, const size_t row,
                const size_t column, const size_t shared, const size_t rows,
                const size_t columns, const size_t depth) {
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < columns; ++j) {
            double product = 0.0;
            for (size_t k = 0; k < depth; ++k) {
                product += A[(row + i) * n + shared + k] *
                           A[(column + j) * n + shared + k];
            }
            A[(row + i) * n + column + j] -= product;
        }
    }
}

} // namespace

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    if (n == 0) return true;

    const size_t tiles = (n + kBlockSize - 1) / kBlockSize;
    bool positiveDefinite = true;

    #pragma omp parallel shared(A, positiveDefinite)
    {
        for (size_t k = 0; k < tiles; ++k) {
            #pragma omp single
            {
                const size_t diagonal = k * kBlockSize;
                const size_t width = std::min(kBlockSize, n - diagonal);
                positiveDefinite = factorDiagonalTile(A, n, diagonal, width);
            }
            #pragma omp barrier
            if (!positiveDefinite) break;

            const size_t diagonal = k * kBlockSize;
            const size_t width = std::min(kBlockSize, n - diagonal);
            #pragma omp for schedule(static)
            for (size_t tile = k + 1; tile < tiles; ++tile) {
                const size_t row = tile * kBlockSize;
                solveTileAgainstDiagonal(A, n, row, diagonal,
                                         std::min(kBlockSize, n - row), width);
            }
            #pragma omp barrier

            #pragma omp for collapse(2) schedule(static)
            for (size_t i = k + 1; i < tiles; ++i) {
                for (size_t j = k + 1; j < tiles; ++j) {
                    if (j > i) continue;
                    const size_t row = i * kBlockSize;
                    const size_t column = j * kBlockSize;
                    updateTile(A, n, row, column, diagonal,
                               std::min(kBlockSize, n - row),
                               std::min(kBlockSize, n - column), width);
                }
            }
            #pragma omp barrier
        }
    }

    // The original interface promises a lower-triangular matrix.
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i)
        for (size_t j = i + 1; j < n; ++j)
            A[i * n + j] = 0.0;

    if (!positiveDefinite) {
        printf("Error: Matrix is not positive definite\n");
        return false;
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
