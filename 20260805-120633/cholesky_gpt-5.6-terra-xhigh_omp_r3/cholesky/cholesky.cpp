#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// Blocked right-looking Cholesky decomposition.  Each panel is factored
// serially, but the triangular solve beneath it and all trailing tiles are
// independent.  Keeping the panel small limits the serial fraction while the
// tiled update provides enough cache-friendly work for all OpenMP threads.
namespace {

constexpr size_t kPanelSize = 64;

bool factorDiagonalPanel(double* const matrix, const size_t n,
                         const size_t begin, const size_t end) {
    for (size_t row = begin; row < end; ++row) {
        double* const currentRow = matrix + row * n;

        for (size_t column = begin; column <= row; ++column) {
            const double* const panelRow = matrix + column * n;
            double product = 0.0;

            #pragma omp simd reduction(+:product)
            for (size_t k = begin; k < column; ++k) {
                product += currentRow[k] * panelRow[k];
            }

            const double value = currentRow[column] - product;
            if (row == column) {
                if (value <= 0.0) {
                    printf("Error: Matrix is not positive definite at diagonal element %zu\n", column);
                    return false;
                }
                currentRow[column] = sqrt(value);
            } else {
                currentRow[column] = value / panelRow[column];
            }
        }
    }

    return true;
}

void solvePanelRow(double* const matrix, const size_t n, const size_t row,
                   const size_t panelBegin, const size_t panelEnd) {
    double* const currentRow = matrix + row * n;

    // Solve L(row, panel) * L(panel, panel)^T = A(row, panel).
    for (size_t column = panelBegin; column < panelEnd; ++column) {
        double product = 0.0;

        #pragma omp simd reduction(+:product)
        for (size_t k = panelBegin; k < column; ++k) {
            product += currentRow[k] * matrix[column * n + k];
        }

        currentRow[column] = (currentRow[column] - product) /
                             matrix[column * n + column];
    }
}

void updateTrailingTile(double* const matrix, const size_t n,
                        const size_t panelBegin, const size_t panelEnd,
                        const size_t rowBegin, const size_t rowEnd,
                        const size_t columnBegin, const size_t columnEnd) {
    for (size_t row = rowBegin; row < rowEnd; ++row) {
        double* const currentRow = matrix + row * n;
        const size_t lastColumn = std::min(columnEnd, row + 1);

        for (size_t column = columnBegin; column < lastColumn; ++column) {
            const double* const columnRow = matrix + column * n;
            double product = 0.0;

            #pragma omp simd reduction(+:product)
            for (size_t k = panelBegin; k < panelEnd; ++k) {
                product += currentRow[k] * columnRow[k];
            }

            currentRow[column] -= product;
        }
    }
}

}  // namespace

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    double* const matrix = A.data();
    const size_t updateTileSize = n < 2048 ? 32 : 64;
    bool success = true;

    #pragma omp parallel shared(success, matrix)
    {
        for (size_t panelBegin = 0; panelBegin < n; panelBegin += kPanelSize) {
            const size_t panelEnd = std::min(panelBegin + kPanelSize, n);

            #pragma omp single
            {
                if (success) {
                    success = factorDiagonalPanel(matrix, n, panelBegin, panelEnd);
                }
            }

            if (success) {
                // The rows in this solve have no dependencies on one another.
                #pragma omp for schedule(static)
                for (std::ptrdiff_t row = static_cast<std::ptrdiff_t>(panelEnd);
                     row < static_cast<std::ptrdiff_t>(n); ++row) {
                    solvePanelRow(matrix, n, static_cast<size_t>(row), panelBegin, panelEnd);
                }

                // A tile is written by exactly one task.  Tasks use distinct
                // output tiles and only read the fully-computed panel.
                #pragma omp single
                {
                    #pragma omp taskgroup
                    {
                        for (size_t rowBegin = panelEnd; rowBegin < n;
                             rowBegin += updateTileSize) {
                            const size_t rowEnd = std::min(rowBegin + updateTileSize, n);

                            for (size_t columnBegin = panelEnd; columnBegin <= rowBegin;
                                 columnBegin += updateTileSize) {
                                const size_t columnEnd = std::min(columnBegin + updateTileSize, n);

                                #pragma omp task firstprivate(panelBegin, panelEnd, rowBegin, rowEnd, columnBegin, columnEnd)
                                updateTrailingTile(matrix, n, panelBegin, panelEnd,
                                                   rowBegin, rowEnd, columnBegin, columnEnd);
                            }
                        }
                    }
                }
            }
        }
    }

    if (!success) {
        return false;
    }

    // The trailing update only needs the lower triangle, so clear the upper
    // triangle once after all panels have been processed.
    #pragma omp parallel for schedule(static)
    for (std::ptrdiff_t row = 0; row < static_cast<std::ptrdiff_t>(n); ++row) {
        double* const currentRow = matrix + static_cast<size_t>(row) * n;
        std::fill(currentRow + row + 1, currentRow + n, 0.0);
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
    
    // Compute A = B * B^T.  Only one triangle is evaluated; the matching
    // entry is identical and can be written by the same iteration.
    #pragma omp parallel for schedule(dynamic, 1)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
        for (size_t j = 0; j <= static_cast<size_t>(i); ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += B[static_cast<size_t>(i) * n + k] * B[j * n + k];
            }
            A[static_cast<size_t>(i) * n + j] = sum;
            A[j * n + static_cast<size_t>(i)] = sum;
        }
    }
    
    // Add diagonal dominance to ensure positive definiteness
    #pragma omp parallel for schedule(static)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
        A[i * n + i] += n;
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix
    
    // Compute L * L^T and compare immediately, avoiding an extra n^2 buffer.
    double maxError = 0.0;
    double relError = 0.0;

    #pragma omp parallel for collapse(2) reduction(max:maxError, relError) schedule(static)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
        for (std::ptrdiff_t j = 0; j < static_cast<std::ptrdiff_t>(n); ++j) {
            double sum = 0.0;
            const size_t lastK = std::min(static_cast<size_t>(i), static_cast<size_t>(j)) + 1;
            for (size_t k = 0; k < lastK; ++k) {
                sum += L[static_cast<size_t>(i) * n + k] * L[static_cast<size_t>(j) * n + k];
            }
            const size_t index = static_cast<size_t>(i) * n + static_cast<size_t>(j);
            const double error = fabs(sum - A_orig[index]);
            maxError = std::max(maxError, error);
            relError = std::max(relError, error / (fabs(A_orig[index]) + 1e-10));
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
    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    const double elapsedSeconds = std::chrono::duration<double>(end - start).count();
    
    if (!success) {
        printf("Cholesky decomposition failed\n");
        return 1;
    }
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
    double ops = (double)n * n * n / 3.0;
    double gflops = elapsedSeconds > 0.0 ? ops / elapsedSeconds / 1e9 : 0.0;
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
