#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <omp.h>
#include <vector>

#include "../common/results_output.hpp"

namespace {

constexpr size_t kUpdateTileSize = 30;

inline int usefulThreadCount(const size_t n) {
    // Very small matrices cannot amortize hundreds of workers.  Sixteen rows
    // per worker gives each thread enough update work, and 128 is also the
    // physical-core count of the largest target used by this benchmark.
    const size_t workLimited =
        std::max(size_t{1}, std::min(size_t{128}, (n + 15) / 16));
    return static_cast<int>(
        std::min(workLimited, static_cast<size_t>(omp_get_max_threads())));
}

inline double dotProduct(const double* const lhs, const double* const rhs,
                         const size_t length) {
    double sum = 0.0;
#pragma omp simd reduction(+ : sum)
    for (size_t k = 0; k < length; ++k) {
        sum += lhs[k] * rhs[k];
    }
    return sum;
}

// Update a rectangular tile that is strictly below the diagonal.  Computing
// three rows by three columns lets each loaded panel value feed several FMAs and
// avoids the horizontal reduction overhead of one independent dot product per
// matrix entry.
inline void updateRectangle(double* const matrix, const size_t n,
                            const size_t panel, const size_t panelWidth,
                            const size_t rowBegin, const size_t rowEnd,
                            const size_t columnBegin,
                            const size_t columnEnd) {
    size_t i = rowBegin;
    for (; i + 2 < rowEnd; i += 3) {
        const double* const lhs0 = matrix + i * n + panel;
        const double* const lhs1 = matrix + (i + 1) * n + panel;
        const double* const lhs2 = matrix + (i + 2) * n + panel;
        double* const output0 = matrix + i * n;
        double* const output1 = matrix + (i + 1) * n;
        double* const output2 = matrix + (i + 2) * n;

        size_t j = columnBegin;
        for (; j + 2 < columnEnd; j += 3) {
            const double* const rhs0 = matrix + j * n + panel;
            const double* const rhs1 = matrix + (j + 1) * n + panel;
            const double* const rhs2 = matrix + (j + 2) * n + panel;
            double sum00 = 0.0;
            double sum01 = 0.0;
            double sum02 = 0.0;
            double sum10 = 0.0;
            double sum11 = 0.0;
            double sum12 = 0.0;
            double sum20 = 0.0;
            double sum21 = 0.0;
            double sum22 = 0.0;

#pragma omp simd reduction(+ : sum00, sum01, sum02, sum10, sum11, sum12, sum20, sum21, sum22)
            for (size_t k = 0; k < panelWidth; ++k) {
                const double left0 = lhs0[k];
                const double left1 = lhs1[k];
                const double left2 = lhs2[k];
                sum00 += left0 * rhs0[k];
                sum01 += left0 * rhs1[k];
                sum02 += left0 * rhs2[k];
                sum10 += left1 * rhs0[k];
                sum11 += left1 * rhs1[k];
                sum12 += left1 * rhs2[k];
                sum20 += left2 * rhs0[k];
                sum21 += left2 * rhs1[k];
                sum22 += left2 * rhs2[k];
            }

            output0[j] -= sum00;
            output0[j + 1] -= sum01;
            output0[j + 2] -= sum02;
            output1[j] -= sum10;
            output1[j + 1] -= sum11;
            output1[j + 2] -= sum12;
            output2[j] -= sum20;
            output2[j + 1] -= sum21;
            output2[j + 2] -= sum22;
        }

        for (; j < columnEnd; ++j) {
            const double* const rhs = matrix + j * n + panel;
            output0[j] -= dotProduct(lhs0, rhs, panelWidth);
            output1[j] -= dotProduct(lhs1, rhs, panelWidth);
            output2[j] -= dotProduct(lhs2, rhs, panelWidth);
        }
    }

    for (; i < rowEnd; ++i) {
        const double* const lhs = matrix + i * n + panel;
        double* const output = matrix + i * n;
        for (size_t j = columnBegin; j < columnEnd; ++j) {
            output[j] -=
                dotProduct(lhs, matrix + j * n + panel, panelWidth);
        }
    }
}

} // namespace

// Blocked, right-looking Cholesky decomposition.
// Decomposes positive definite matrix A into L * L^T where L is lower triangular

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    bool positiveDefinite = true;
    size_t failedDiagonal = 0;
    // Longer panels amortize reduction and synchronization overhead on large
    // matrices, while shorter panels expose more parallelism on small ones.
    const size_t panelSize =
        n >= 3072 ? size_t{256} : (n >= 1536 ? size_t{128} : size_t{64});
    const int threadCount = usefulThreadCount(n);

    // Keep one team alive for the whole factorization.  Only the small
    // diagonal panel is serial; the triangular solve and Schur-complement
    // update operate on independent rows.
#pragma omp parallel num_threads(threadCount) shared(positiveDefinite, failedDiagonal)
    {
        for (size_t panel = 0; panel < n; panel += panelSize) {
            const size_t panelEnd = std::min(panel + panelSize, n);

#pragma omp single
            {
                // Factor the diagonal block.
                for (size_t j = panel; j < panelEnd; ++j) {
                    double* const rowJ = A.data() + j * n;
                    const double diagonalUpdate =
                        dotProduct(rowJ + panel, rowJ + panel, j - panel);
                    const double val = rowJ[j] - diagonalUpdate;

                    if (val <= 0.0) {
                        positiveDefinite = false;
                        failedDiagonal = j;
                        break;
                    }
                    rowJ[j] = std::sqrt(val);

                    const double inverseDiagonal = 1.0 / rowJ[j];
                    for (size_t i = j + 1; i < panelEnd; ++i) {
                        double* const rowI = A.data() + i * n;
                        const double update =
                            dotProduct(rowI + panel, rowJ + panel, j - panel);
                        rowI[j] = (rowI[j] - update) * inverseDiagonal;
                    }
                }
            }

            if (!positiveDefinite) {
                break;
            }

            // Solve L21 * L11^T = A21.  Rows of L21 are independent.
#pragma omp for schedule(static)
            for (size_t i = panelEnd; i < n; ++i) {
                double* const rowI = A.data() + i * n;
                for (size_t j = panel; j < panelEnd; ++j) {
                    const double* const rowJ = A.data() + j * n;
                    const double update =
                        dotProduct(rowI + panel, rowJ + panel, j - panel);
                    rowI[j] = (rowI[j] - update) / rowJ[j];
                }
            }

            // A22 -= L21 * L21^T.  The lower triangle is flattened into
            // equal-size tiles for balanced dynamic scheduling.  Off-diagonal
            // tiles use a register-blocked matrix multiplication kernel.
            const size_t trailingSize = n - panelEnd;
            const size_t tileRows =
                (trailingSize + kUpdateTileSize - 1) / kUpdateTileSize;
            const size_t tileCount = tileRows * (tileRows + 1) / 2;
#pragma omp for schedule(dynamic, 1)
            for (size_t tile = 0; tile < tileCount; ++tile) {
                size_t tileRow = static_cast<size_t>(
                    (std::sqrt(8.0 * static_cast<double>(tile) + 1.0) - 1.0) /
                    2.0);
                while ((tileRow + 1) * (tileRow + 2) / 2 <= tile) {
                    ++tileRow;
                }
                while (tileRow * (tileRow + 1) / 2 > tile) {
                    --tileRow;
                }
                const size_t tileColumn =
                    tile - tileRow * (tileRow + 1) / 2;
                const size_t rowBegin =
                    panelEnd + tileRow * kUpdateTileSize;
                const size_t rowEnd =
                    std::min(rowBegin + kUpdateTileSize, n);
                const size_t columnBegin =
                    panelEnd + tileColumn * kUpdateTileSize;
                const size_t columnEnd =
                    std::min(columnBegin + kUpdateTileSize, n);

                if (tileColumn != tileRow) {
                    updateRectangle(A.data(), n, panel,
                                    panelEnd - panel, rowBegin, rowEnd,
                                    columnBegin, columnEnd);
                } else {
                    // Within a diagonal tile, all 3x3 blocks below its tiny
                    // block diagonal can still use the register-blocked
                    // rectangular kernel.
                    for (size_t blockRow = rowBegin; blockRow < rowEnd;
                         blockRow += 3) {
                        const size_t blockRowEnd =
                            std::min(blockRow + size_t{3}, rowEnd);
                        for (size_t blockColumn = columnBegin;
                             blockColumn < blockRow; blockColumn += 3) {
                            updateRectangle(
                                A.data(), n, panel, panelEnd - panel,
                                blockRow, blockRowEnd, blockColumn,
                                std::min(blockColumn + size_t{3}, rowEnd));
                        }
                        for (size_t i = blockRow; i < blockRowEnd; ++i) {
                            double* const rowI = A.data() + i * n;
                            for (size_t j = blockRow; j <= i; ++j) {
                                rowI[j] -= dotProduct(
                                    rowI + panel, A.data() + j * n + panel,
                                    panelEnd - panel);
                            }
                        }
                    }
                }
            }
        }

        if (positiveDefinite) {
            // The factor is returned in the same explicit lower-triangular
            // representation as the original implementation.
#pragma omp for schedule(static)
            for (size_t i = 0; i < n; ++i) {
                std::fill(A.data() + i * n + i + 1, A.data() + (i + 1) * n,
                          0.0);
            }
        }
    }

    if (!positiveDefinite) {
        printf("Error: Matrix is not positive definite at diagonal element %zu\n",
               failedDiagonal);
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
    
    // Compute one triangle of A = B * B^T and mirror it.  Each pair has a
    // single owner, so this needs no synchronization.
#pragma omp parallel for num_threads(usefulThreadCount(n)) schedule(dynamic, 1)
    for (size_t i = 0; i < n; ++i) {
        const double* const rowI = B.data() + i * n;
        for (size_t j = 0; j <= i; ++j) {
            const double sum = dotProduct(rowI, B.data() + j * n, n);
            A[i * n + j] = sum;
            A[j * n + i] = sum;
        }
        A[i * n + i] += n;
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix
    
    double maxError = 0.0;
    double relError = 0.0;

    // L * L^T is symmetric, so compute each dot product only once while
    // still checking both entries of the supplied original matrix.
#pragma omp parallel for num_threads(usefulThreadCount(n)) reduction(max : maxError, relError) schedule(dynamic, 1)
    for (size_t i = 0; i < n; ++i) {
        const double* const rowI = L.data() + i * n;
        for (size_t j = 0; j <= i; ++j) {
            const double reconstructed =
                dotProduct(rowI, L.data() + j * n, j + 1);

            const size_t lower = i * n + j;
            const double lowerError = std::fabs(reconstructed - A_orig[lower]);
            maxError = std::max(maxError, lowerError);
            relError = std::max(
                relError, lowerError / (std::fabs(A_orig[lower]) + 1e-10));

            if (i != j) {
                const size_t upper = j * n + i;
                const double upperError =
                    std::fabs(reconstructed - A_orig[upper]);
                maxError = std::max(maxError, upperError);
                relError = std::max(
                    relError,
                    upperError / (std::fabs(A_orig[upper]) + 1e-10));
            }
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

    // Establish the work-sized team before the first parallel region so the
    // runtime does not create an oversized default worker pool.
    omp_set_dynamic(0);
    omp_set_num_threads(usefulThreadCount(n));
    
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
