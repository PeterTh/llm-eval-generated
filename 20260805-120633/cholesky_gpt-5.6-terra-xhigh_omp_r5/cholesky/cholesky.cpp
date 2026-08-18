#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <omp.h>
#include <vector>

#include "../common/results_output.hpp"

// Cache-blocked Cholesky decomposition.  A panel has unavoidable dependencies,
// but the triangular solve below it and the trailing rank-k update are parallel.
namespace {

constexpr size_t kCholeskyBlockSize = 64;

bool factorDiagonalBlock(double* const matrix, const size_t n,
                         const size_t first, const size_t last) {
    for (size_t i = first; i < last; ++i) {
        double* const rowI = matrix + i * n;
        for (size_t j = first; j <= i; ++j) {
            double* const rowJ = matrix + j * n;
            double value = rowI[j];

            #pragma omp simd reduction(-:value)
            for (size_t k = first; k < j; ++k) {
                value -= rowI[k] * rowJ[k];
            }

            if (i == j) {
                if (value <= 0.0) {
                    printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
                    return false;
                }
                rowI[j] = sqrt(value);
            } else {
                rowI[j] = value / rowJ[j];
            }
        }
    }

    return true;
}

} // namespace

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    double* const matrix = A.data();
    // This is L_21 stored by panel column.  It turns the rank-k update into
    // contiguous SIMD streams for both its source and destination matrices.
    std::vector<double> transposedPanel(kCholeskyBlockSize * n);
    double* const transposed = transposedPanel.data();
    bool success = true;

    #pragma omp parallel default(none) shared(matrix, n, transposed, success)
    {
        for (size_t panel = 0; panel < n; panel += kCholeskyBlockSize) {
            const size_t panelEnd = std::min(panel + kCholeskyBlockSize, n);

            // Factor L_11.  This small, cache-resident panel is the dependency
            // frontier for the following parallel operations.
            #pragma omp single
            {
                if (success) {
                    success = factorDiagonalBlock(matrix, n, panel, panelEnd);
                }
            }

            // Compute L_21 = A_21 * inv(L_11^T).  Rows are independent.
            if (success) {
                #pragma omp for schedule(static)
                for (size_t i = panelEnd; i < n; ++i) {
                    double* const rowI = matrix + i * n;
                    for (size_t j = panel; j < panelEnd; ++j) {
                        const double* const rowJ = matrix + j * n;
                        double value = rowI[j];

                        #pragma omp simd reduction(-:value)
                        for (size_t k = panel; k < j; ++k) {
                            value -= rowI[k] * rowJ[k];
                        }
                        rowI[j] = value / rowJ[j];
                    }
                }
            }

            // Store L_21 by panel column so that every rank-k update is a
            // pair of contiguous streams instead of a strided gather.
            if (success) {
                #pragma omp for schedule(static)
                for (size_t k = panel; k < panelEnd; ++k) {
                    double* const transposedRow =
                        transposed + (k - panel) * n;
                    for (size_t i = panelEnd; i < n; ++i) {
                        transposedRow[i] = matrix[i * n + k];
                    }
                }
            }

            // Update A_22 -= L_21 * L_21^T.  Independent tiles avoid false
            // sharing, and the transposed panel makes the hot inner loop SIMD
            // friendly and cache efficient.
            if (success) {
                const size_t trailingBlocks =
                    (n - panelEnd + kCholeskyBlockSize - 1) / kCholeskyBlockSize;
                const size_t tileCount = trailingBlocks * (trailingBlocks + 1) / 2;

                #pragma omp for schedule(dynamic, 1)
                for (size_t tile = 0; tile < tileCount; ++tile) {
                    size_t tileRow = static_cast<size_t>(
                        (sqrt(static_cast<double>(8 * tile + 1)) - 1.0) * 0.5);
                    while ((tileRow + 1) * (tileRow + 2) / 2 <= tile) {
                        ++tileRow;
                    }
                    while (tileRow * (tileRow + 1) / 2 > tile) {
                        --tileRow;
                    }
                    const size_t tileColumn = tile - tileRow * (tileRow + 1) / 2;
                    const size_t rowBegin = panelEnd + tileRow * kCholeskyBlockSize;
                    const size_t rowEnd = std::min(rowBegin + kCholeskyBlockSize, n);
                    const size_t columnBegin = panelEnd + tileColumn * kCholeskyBlockSize;
                    const size_t columnEnd = std::min(columnBegin + kCholeskyBlockSize, n);

                    for (size_t i = rowBegin; i < rowEnd; ++i) {
                        double* const rowI = matrix + i * n;
                        const size_t updateEnd =
                            tileRow == tileColumn ? std::min(columnEnd, i + 1) : columnEnd;
                        for (size_t k = panel; k < panelEnd; ++k) {
                            const double value = rowI[k];
                            const double* const transposedRow =
                                transposed + (k - panel) * n;

                            #pragma omp simd
                            for (size_t j = columnBegin; j < updateEnd; ++j) {
                                rowI[j] -= value * transposedRow[j];
                            }
                        }
                    }
                }
            }
        }

        // Keep the lower-triangular output contract of the original routine.
        if (success) {
            #pragma omp for schedule(static)
            for (size_t i = 0; i < n; ++i) {
                double* const row = matrix + i * n;
                for (size_t j = i + 1; j < n; ++j) {
                    row[j] = 0.0;
                }
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
    
    // Compute A = B * B^T.  Each result element is independent.
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            const double* const rowI = B.data() + i * n;
            const double* const rowJ = B.data() + j * n;

            #pragma omp simd reduction(+:sum)
            for (size_t k = 0; k < n; ++k) {
                sum += rowI[k] * rowJ[k];
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
    
    // Compute L * L^T.  The output elements are independent.
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            const double* const rowI = L.data() + i * n;
            const double* const rowJ = L.data() + j * n;

            #pragma omp simd reduction(+:sum)
            for (size_t k = 0; k < n; ++k) {
                sum += rowI[k] * rowJ[k];
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
