#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <omp.h>

#include "../common/results_output.hpp"

// Tiled, task-based Cholesky decomposition.  Each lower-triangular matrix tile
// has a dependency token.  This lets the OpenMP runtime execute trailing-matrix
// updates in parallel and start a later panel as soon as its own updates finish,
// without a global barrier between panels.

template <bool diagonalTile>
static void updateTrailingTile(double* const matrix, const size_t n,
                               const size_t kBegin, const size_t kEnd,
                               const size_t rowBegin, const size_t rowEnd,
                               const size_t columnBegin, const size_t columnEnd) {
    const size_t panelWidth = kEnd - kBegin;

    for (size_t row = rowBegin; row < rowEnd; ++row) {
        const double* const rowPanel = matrix + row * n + kBegin;
        const size_t limit = diagonalTile ? std::min(row + 1, columnEnd) : columnEnd;
        size_t column = columnBegin;

        // Evaluate eight independent dot products together.  The independent
        // accumulators hide FMA latency and reuse rowPanel from L1, while the
        // SIMD reduction vectorizes each short tile dot product.
        for (; column + 7 < limit; column += 8) {
            const double* const panel0 = matrix + (column + 0) * n + kBegin;
            const double* const panel1 = matrix + (column + 1) * n + kBegin;
            const double* const panel2 = matrix + (column + 2) * n + kBegin;
            const double* const panel3 = matrix + (column + 3) * n + kBegin;
            const double* const panel4 = matrix + (column + 4) * n + kBegin;
            const double* const panel5 = matrix + (column + 5) * n + kBegin;
            const double* const panel6 = matrix + (column + 6) * n + kBegin;
            const double* const panel7 = matrix + (column + 7) * n + kBegin;
            double sum0 = 0.0;
            double sum1 = 0.0;
            double sum2 = 0.0;
            double sum3 = 0.0;
            double sum4 = 0.0;
            double sum5 = 0.0;
            double sum6 = 0.0;
            double sum7 = 0.0;

            #pragma omp simd reduction(+: sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7)
            for (size_t p = 0; p < panelWidth; ++p) {
                const double value = rowPanel[p];
                sum0 += value * panel0[p];
                sum1 += value * panel1[p];
                sum2 += value * panel2[p];
                sum3 += value * panel3[p];
                sum4 += value * panel4[p];
                sum5 += value * panel5[p];
                sum6 += value * panel6[p];
                sum7 += value * panel7[p];
            }

            double* const output = matrix + row * n + column;
            output[0] -= sum0;
            output[1] -= sum1;
            output[2] -= sum2;
            output[3] -= sum3;
            output[4] -= sum4;
            output[5] -= sum5;
            output[6] -= sum6;
            output[7] -= sum7;
        }

        for (; column + 3 < limit; column += 4) {
            const double* const panel0 = matrix + (column + 0) * n + kBegin;
            const double* const panel1 = matrix + (column + 1) * n + kBegin;
            const double* const panel2 = matrix + (column + 2) * n + kBegin;
            const double* const panel3 = matrix + (column + 3) * n + kBegin;
            double sum0 = 0.0;
            double sum1 = 0.0;
            double sum2 = 0.0;
            double sum3 = 0.0;

            #pragma omp simd reduction(+: sum0, sum1, sum2, sum3)
            for (size_t p = 0; p < panelWidth; ++p) {
                const double value = rowPanel[p];
                sum0 += value * panel0[p];
                sum1 += value * panel1[p];
                sum2 += value * panel2[p];
                sum3 += value * panel3[p];
            }

            double* const output = matrix + row * n + column;
            output[0] -= sum0;
            output[1] -= sum1;
            output[2] -= sum2;
            output[3] -= sum3;
        }

        for (; column < limit; ++column) {
            const double* const columnPanel = matrix + column * n + kBegin;
            double update = 0.0;
            #pragma omp simd reduction(+: update)
            for (size_t p = 0; p < panelWidth; ++p) {
                update += rowPanel[p] * columnPanel[p];
            }
            matrix[row * n + column] -= update;
        }
    }
}

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    if (n == 0) {
        return true;
    }

    // Smaller matrices need more tasks on the ready queue; larger matrices
    // benefit from a wider tile that amortizes runtime and dependency costs.
    const size_t tileSize = n < 1536 ? 64 : 128;
    const size_t numTiles = (n + tileSize - 1) / tileSize;
    std::vector<unsigned char> tokens(numTiles * numTiles);
    [[maybe_unused]] unsigned char* const token = tokens.data();
    std::atomic<size_t> failedDiagonal{n};

    #pragma omp parallel default(none) shared(A, failedDiagonal, token) firstprivate(n, numTiles, tileSize)
    {
        #pragma omp single
        {
            for (size_t kTile = 0; kTile < numTiles; ++kTile) {
                const size_t kBegin = kTile * tileSize;
                const size_t kEnd = std::min(kBegin + tileSize, n);
                const size_t diagonalToken = kTile * numTiles + kTile;

                // Factor the diagonal tile (POTRF).
                #pragma omp task default(none) shared(A, failedDiagonal, token) firstprivate(n, kBegin, kEnd, diagonalToken) depend(inout: token[diagonalToken])
                {
                    if (failedDiagonal.load(std::memory_order_relaxed) == n) {
                        for (size_t column = kBegin; column < kEnd; ++column) {
                            double diagonal = A[column * n + column];
                            #pragma omp simd reduction(-: diagonal)
                            for (size_t p = kBegin; p < column; ++p) {
                                diagonal -= A[column * n + p] * A[column * n + p];
                            }

                            if (diagonal <= 0.0) {
                                size_t expected = n;
                                failedDiagonal.compare_exchange_strong(
                                    expected, column, std::memory_order_relaxed);
                                break;
                            }

                            const double diagonalRoot = std::sqrt(diagonal);
                            const double inverseDiagonal = 1.0 / diagonalRoot;
                            A[column * n + column] = diagonalRoot;

                            for (size_t row = column + 1; row < kEnd; ++row) {
                                double value = A[row * n + column];
                                #pragma omp simd reduction(-: value)
                                for (size_t p = kBegin; p < column; ++p) {
                                    value -= A[row * n + p] * A[column * n + p];
                                }
                                A[row * n + column] = value * inverseDiagonal;
                            }
                        }
                    }
                }

                // Solve all tiles below the diagonal tile (TRSM).
                for (size_t rowTile = kTile + 1; rowTile < numTiles; ++rowTile) {
                    const size_t rowBegin = rowTile * tileSize;
                    const size_t rowEnd = std::min(rowBegin + tileSize, n);
                    const size_t panelToken = rowTile * numTiles + kTile;

                    #pragma omp task default(none) shared(A, failedDiagonal, token) firstprivate(n, kBegin, kEnd, rowBegin, rowEnd, diagonalToken, panelToken) depend(in: token[diagonalToken]) depend(inout: token[panelToken])
                    {
                        if (failedDiagonal.load(std::memory_order_relaxed) == n) {
                            // Adaptive tiling is capped at 128 columns.
                            double inverseDiagonal[128];
                            for (size_t column = kBegin; column < kEnd; ++column) {
                                inverseDiagonal[column - kBegin] =
                                    1.0 / A[column * n + column];
                            }
                            for (size_t row = rowBegin; row < rowEnd; ++row) {
                                for (size_t column = kBegin; column < kEnd; ++column) {
                                    double value = A[row * n + column];
                                    #pragma omp simd reduction(-: value)
                                    for (size_t p = kBegin; p < column; ++p) {
                                        value -= A[row * n + p] * A[column * n + p];
                                    }
                                    A[row * n + column] =
                                        value * inverseDiagonal[column - kBegin];
                                }
                            }
                        }
                    }
                }

                // Update the lower-triangular trailing matrix.  Diagonal
                // updates (SYRK) have one panel input; off-diagonal updates
                // (GEMM) have two independent panel inputs.
                for (size_t rowTile = kTile + 1; rowTile < numTiles; ++rowTile) {
                    const size_t rowBegin = rowTile * tileSize;
                    const size_t rowEnd = std::min(rowBegin + tileSize, n);
                    const size_t rowPanelToken = rowTile * numTiles + kTile;
                    const size_t targetToken = rowTile * numTiles + rowTile;

                    #pragma omp task default(none) shared(A, failedDiagonal, token) firstprivate(n, kBegin, kEnd, rowBegin, rowEnd, rowPanelToken, targetToken) depend(in: token[rowPanelToken]) depend(inout: token[targetToken])
                    {
                        if (failedDiagonal.load(std::memory_order_relaxed) == n) {
                            updateTrailingTile<true>(A.data(), n, kBegin, kEnd,
                                                     rowBegin, rowEnd, rowBegin, rowEnd);
                        }
                    }

                    for (size_t columnTile = kTile + 1; columnTile < rowTile; ++columnTile) {
                        const size_t columnBegin = columnTile * tileSize;
                        const size_t columnEnd = std::min(columnBegin + tileSize, n);
                        const size_t columnPanelToken = columnTile * numTiles + kTile;
                        const size_t offDiagonalToken = rowTile * numTiles + columnTile;

                        #pragma omp task default(none) shared(A, failedDiagonal, token) firstprivate(n, kBegin, kEnd, rowBegin, rowEnd, columnBegin, columnEnd, rowPanelToken, columnPanelToken, offDiagonalToken) depend(in: token[rowPanelToken], token[columnPanelToken]) depend(inout: token[offDiagonalToken])
                        {
                            if (failedDiagonal.load(std::memory_order_relaxed) == n) {
                                updateTrailingTile<false>(A.data(), n, kBegin, kEnd,
                                                          rowBegin, rowEnd,
                                                          columnBegin, columnEnd);
                            }
                        }
                    }
                }
            }
        }

        // The implicit single barrier also waits for all factorization tasks.
        // Reuse the same team to clear the upper triangle and avoid
        // starting another parallel region.
        const bool succeeded = failedDiagonal.load(std::memory_order_relaxed) == n;
        #pragma omp for schedule(static) nowait
        for (size_t row = 0; row < n; ++row) {
            if (succeeded) {
                std::fill(A.begin() + row * n + row + 1,
                          A.begin() + (row + 1) * n, 0.0);
            }
        }
    }

    const size_t failure = failedDiagonal.load(std::memory_order_relaxed);
    if (failure != n) {
        printf("Error: Matrix is not positive definite at diagonal element %zu\n", failure);
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
    
    // Compute A = B * B^T.  Rows are independent and are deliberately
    // distributed statically to retain locality.
    #pragma omp parallel for schedule(static) default(none) shared(A, B) firstprivate(n)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            #pragma omp simd reduction(+: sum)
            for (size_t k = 0; k < n; ++k) {
                sum += B[i * n + k] * B[j * n + k];
            }
            A[i * n + j] = sum;
        }
        A[i * n + i] += n;
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix
    
    std::vector<double> reconstructed(n * n);
    
    // Compute L * L^T
    #pragma omp parallel for collapse(2) schedule(static) default(none) shared(reconstructed, L) firstprivate(n)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            #pragma omp simd reduction(+: sum)
            for (size_t k = 0; k < n; ++k) {
                sum += L[i * n + k] * L[j * n + k];
            }
            reconstructed[i * n + j] = sum;
        }
    }
    
    // Compare with original
    double maxError = 0.0;
    double relError = 0.0;
    
    #pragma omp parallel for schedule(static) reduction(max: maxError, relError) default(none) shared(reconstructed, A_orig) firstprivate(n)
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
