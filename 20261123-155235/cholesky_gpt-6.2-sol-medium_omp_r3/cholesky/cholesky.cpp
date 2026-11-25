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

namespace {
constexpr size_t tileSize = 64;
constexpr size_t gramTileSize = 16;
constexpr size_t updateCols = 8;
constexpr size_t packedDepth = tileSize / updateCols;

int teamSize(size_t n) {
    // Keep small matrices from starting more workers than their tiles can use.
    return static_cast<int>(std::min<size_t>(omp_get_max_threads(), 1 + n / 128));
}

double dotProduct(const double* left, const double* right, size_t length) {
    double sum = 0.0;
    #pragma omp simd reduction(+:sum)
    for (size_t k = 0; k < length; ++k) {
        sum += left[k] * right[k];
    }
    return sum;
}

double gramDotProduct(const double* left, const double* right, size_t length) {
    // Independent accumulators hide the SIMD addition latency of long dot products.
    const size_t span = length / 4;
    double sum0 = 0.0;
    double sum1 = 0.0;
    double sum2 = 0.0;
    double sum3 = 0.0;
    #pragma omp simd reduction(+:sum0, sum1, sum2, sum3)
    for (size_t k = 0; k < span; ++k) {
        sum0 += left[k] * right[k];
        sum1 += left[k + span] * right[k + span];
        sum2 += left[k + 2 * span] * right[k + 2 * span];
        sum3 += left[k + 3 * span] * right[k + 3 * span];
    }
    return (sum0 + sum1 + sum2 + sum3) + dotProduct(left + 4 * span, right + 4 * span, length % 4);
}

size_t factorDiagonal(double* matrix, size_t n, size_t begin, size_t end) {
    for (size_t i = begin; i < end; ++i) {
        double* row = matrix + i * n;
        for (size_t j = begin; j < i; ++j) {
            const double* factorRow = matrix + j * n;
            row[j] = (row[j] - dotProduct(row + begin, factorRow + begin, j - begin))
                     / factorRow[j];
        }
        const double pivot = row[i] - dotProduct(row + begin, row + begin, i - begin);
        if (pivot <= 0.0) {
            return i;
        }
        row[i] = std::sqrt(pivot);
    }
    return n;
}

void solvePanel(double* matrix, size_t n, size_t rowBegin, size_t rowEnd,
                size_t colBegin, size_t colEnd) {
    for (size_t i = rowBegin; i < rowEnd; ++i) {
        double* row = matrix + i * n;
        for (size_t j = colBegin; j < colEnd; ++j) {
            const double* factorRow = matrix + j * n;
            row[j] = (row[j] - dotProduct(row + colBegin, factorRow + colBegin, j - colBegin))
                     / factorRow[j];
        }
    }

    // Pack full transposed panels into strips so each microkernel reads contiguous data.
    if (rowEnd - rowBegin == tileSize) {
        for (size_t j = colBegin; j < colEnd; j += packedDepth) {
            for (size_t i = rowBegin; i < rowEnd; i += updateCols) {
                double* packed = matrix + (j + (i - rowBegin) / updateCols) * n + rowBegin;
                for (size_t k = 0; k < packedDepth; ++k) {
                    for (size_t col = 0; col < updateCols; ++col) {
                        packed[k * updateCols + col] = matrix[(i + col) * n + j + k];
                    }
                }
            }
        }
        return;
    }

    // A partial panel uses the same upper-triangle storage with a plain transpose.
    for (size_t j = colBegin; j < colEnd; ++j) {
        double* transposed = matrix + j * n;
        for (size_t i = rowBegin; i < rowEnd; ++i) {
            transposed[i] = matrix[i * n + j];
        }
    }
}

template<size_t Rows, size_t Cols, bool Packed>
void updateBlock(double* matrix, size_t n, size_t i, size_t j,
                 size_t colBegin, size_t panelBegin, size_t panelEnd) {
    double values[Rows][Cols];
    for (size_t row = 0; row < Rows; ++row) {
        std::copy_n(matrix + (i + row) * n + j, Cols, values[row]);
    }

    for (size_t k = panelBegin; k < panelEnd;) {
        const double* right = Packed ? matrix + (k + (j - colBegin) / updateCols) * n + colBegin
                                     : matrix + k * n + j;
        const size_t kEnd = Packed ? k + packedDepth : panelEnd;
        for (; k < kEnd; ++k, right += Packed ? updateCols : n) {
            for (size_t row = 0; row < Rows; ++row) {
                const double left = matrix[(i + row) * n + k];
                #pragma omp simd
                for (size_t col = 0; col < Cols; ++col) {
                    values[row][col] -= left * right[col];
                }
            }
        }
    }

    for (size_t row = 0; row < Rows; ++row) {
        std::copy_n(values[row], Cols, matrix + (i + row) * n + j);
    }
}

template<size_t Rows>
void updateRows(double* matrix, size_t n, size_t i, size_t colBegin, size_t colEnd,
                size_t liveColEnd, size_t panelBegin, size_t panelEnd) {
    if (colEnd - colBegin == tileSize) {
        for (size_t j = colBegin; j < liveColEnd; j += updateCols) {
            updateBlock<Rows, updateCols, true>(matrix, n, i, j, colBegin, panelBegin, panelEnd);
        }
        return;
    }

    size_t j = colBegin;
    for (; j + updateCols <= liveColEnd; j += updateCols) {
        updateBlock<Rows, updateCols, false>(matrix, n, i, j, colBegin, panelBegin, panelEnd);
    }
    for (; j < liveColEnd; ++j) {
        updateBlock<Rows, 1, false>(matrix, n, i, j, colBegin, panelBegin, panelEnd);
    }
}

void updateTile(double* matrix, size_t n, size_t rowBegin, size_t rowEnd,
                size_t colBegin, size_t colEnd, size_t panelBegin, size_t panelEnd) {
    constexpr size_t rowsPerBlock = 4;
    const bool diagonal = rowBegin == colBegin;
    size_t i = rowBegin;
    for (; i + rowsPerBlock <= rowEnd; i += rowsPerBlock) {
        updateRows<rowsPerBlock>(matrix, n, i, colBegin, colEnd,
                                 diagonal ? i + rowsPerBlock : colEnd, panelBegin, panelEnd);
    }
    for (; i < rowEnd; ++i) {
        updateRows<1>(matrix, n, i, colBegin, colEnd, diagonal ? i + 1 : colEnd, panelBegin, panelEnd);
    }
}
} // namespace

// Factor the lower triangle in place as A = L * L^T.
bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    double* matrix = A.data();
    std::atomic<size_t> failedPivot{n};

    #pragma omp parallel num_threads(teamSize(n))
    {
        #pragma omp single
        {
            // The first element identifies each tile, including a panel tile's transpose.
            for (size_t k = 0; k < n; k += tileSize) {
                const size_t kEnd = std::min(k + tileSize, n);
                const size_t diagonal = k * n + k;
                #pragma omp task depend(inout: matrix[diagonal])
                {
                    if (failedPivot.load(std::memory_order_relaxed) == n) {
                        const size_t pivot = factorDiagonal(matrix, n, k, kEnd);
                        if (pivot != n) {
                            failedPivot.store(pivot, std::memory_order_relaxed);
                        }
                    }
                }

                for (size_t i = kEnd; i < n; i += tileSize) {
                    const size_t iEnd = std::min(i + tileSize, n);
                    const size_t panel = i * n + k;
                    #pragma omp task depend(in: matrix[diagonal]) depend(inout: matrix[panel])
                    {
                        if (failedPivot.load(std::memory_order_relaxed) == n) {
                            solvePanel(matrix, n, i, iEnd, k, kEnd);
                        }
                    }
                }

                for (size_t i = kEnd; i < n; i += tileSize) {
                    const size_t iEnd = std::min(i + tileSize, n);
                    const size_t left = i * n + k;
                    #pragma omp task depend(in: matrix[left]) depend(inout: matrix[i * n + i])
                    {
                        if (failedPivot.load(std::memory_order_relaxed) == n) {
                            updateTile(matrix, n, i, iEnd, i, iEnd, k, kEnd);
                        }
                    }
                    for (size_t j = kEnd; j < i; j += tileSize) {
                        const size_t jEnd = std::min(j + tileSize, n);
                        const size_t right = j * n + k;
                        const size_t target = i * n + j;
                        #pragma omp task depend(in: matrix[left], matrix[right]) depend(inout: matrix[target])
                        {
                            if (failedPivot.load(std::memory_order_relaxed) == n) {
                                updateTile(matrix, n, i, iEnd, j, jEnd, k, kEnd);
                            }
                        }
                    }
                }
            }
        } // The barrier also waits for all tile tasks.

        #pragma omp for schedule(static, 1)
        for (size_t i = 0; i < n; ++i) {
            std::fill(matrix + i * n + i + 1, matrix + (i + 1) * n, 0.0);
        }
    }

    const size_t pivot = failedPivot.load(std::memory_order_relaxed);
    if (pivot != n) {
        printf("Error: Matrix is not positive definite at diagonal element %zu\n", pivot);
        return false;
    }
    return true;
}

void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    for (double& value : B) {
        value = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

    // Compute one triangle of B * B^T, then mirror it.
    #pragma omp parallel for num_threads(teamSize(n)) schedule(static, 1)
    for (size_t ii = 0; ii < n; ii += gramTileSize) {
        const size_t iEnd = std::min(ii + gramTileSize, n);
        for (size_t jj = 0; jj <= ii; jj += gramTileSize) {
            const size_t jEnd = std::min(jj + gramTileSize, n);
            for (size_t i = ii; i < iEnd; ++i) {
                for (size_t j = jj; j < std::min(jEnd, i); ++j) {
                    const double value = gramDotProduct(B.data() + i * n, B.data() + j * n, n);
                    A[i * n + j] = value;
                    A[j * n + i] = value;
                }
            }
        }
        for (size_t i = ii; i < iEnd; ++i) {
            const double* row = B.data() + i * n;
            A[i * n + i] = gramDotProduct(row, row, n) + n;
        }
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    double maxError = 0.0;
    double relError = 0.0;

    #pragma omp parallel for num_threads(teamSize(n)) schedule(static, 1) reduction(max:maxError, relError)
    for (size_t ii = 0; ii < n; ii += gramTileSize) {
        const size_t iEnd = std::min(ii + gramTileSize, n);
        for (size_t jj = 0; jj <= ii; jj += gramTileSize) {
            const size_t jEnd = std::min(jj + gramTileSize, n);
            for (size_t i = ii; i < iEnd; ++i) {
                for (size_t j = jj; j < std::min(jEnd, i + 1); ++j) {
                    // Row j is zero beyond column j.
                    const double value = gramDotProduct(L.data() + i * n, L.data() + j * n, j + 1);
                    const double lower = A_orig[i * n + j];
                    const double upper = A_orig[j * n + i];
                    const double lowerError = std::abs(value - lower);
                    const double upperError = std::abs(value - upper);
                    maxError = std::max(maxError, std::max(lowerError, upperError));
                    relError = std::max(relError, std::max(lowerError / (std::abs(lower) + 1e-10),
                                                          upperError / (std::abs(upper) + 1e-10)));
                }
            }
        }
    }

    printf("Max absolute error: %.10e\n", maxError);
    printf("Max relative error: %.10e\n", relError);

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
    auto start = std::chrono::steady_clock::now();
    
    bool success = choleskyDecomposition(A, n);
    
    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (!success) {
        printf("Cholesky decomposition failed\n");
        return 1;
    }
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
    double ops = (double)n * n * n / 3.0;
    double gflops = ops / std::chrono::duration<double>(end - start).count() / 1e9;
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
