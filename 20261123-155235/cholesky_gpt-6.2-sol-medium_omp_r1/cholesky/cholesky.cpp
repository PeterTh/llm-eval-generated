#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include <omp.h>

#include "../common/results_output.hpp"

namespace {

constexpr size_t maxTileSize = 128;
constexpr size_t updateWidth = 8;

size_t matrixTileSize(size_t n) {
    return n <= 1024 ? 64 : maxTileSize;
}

int teamSize(size_t n) {
    // Honor the OpenMP thread limit and keep teams small for small matrices.
    return std::min<size_t>(omp_get_max_threads(), std::max<size_t>(1, n / 48));
}

size_t factorDiagonal(double* matrix, size_t n, size_t begin, size_t end,
                      double* transposed, size_t stride) {
    for (size_t i = begin; i < end; ++i) {
        double* row = matrix + i * n;
        for (size_t j = begin; j <= i; ++j) {
            const double* other = matrix + j * n;
            double dot = 0.0;
            for (size_t k = begin; k < j; ++k) {
                dot += row[k] * other[k];
            }
            const double value = row[j] - dot;
            if (i == j) {
                if (value <= 0.0) {
                    return j;
                }
                row[j] = std::sqrt(value);
            } else {
                row[j] = value / other[j];
            }
            transposed[(j - begin) * stride + (i - begin)] = row[j];
        }
    }
    return n;
}

void solvePanel(double* matrix, size_t n, size_t rowBegin, size_t rowEnd,
                size_t begin, size_t end, const double* transposed, size_t stride) {
    const size_t width = end - begin;
    for (size_t i = rowBegin; i < rowEnd; ++i) {
        double* row = matrix + i * n + begin;
        for (size_t j = 0; j < width; ++j) {
            const double* column = transposed + j * stride;
            const double value = row[j] / column[j];
            row[j] = value;
            #pragma omp simd
            for (size_t k = j + 1; k < width; ++k) {
                row[k] -= value * column[k];
            }
        }
    }
}

template <size_t rows, bool subtract>
void updateRows(double* tile, size_t n, const double* panel, size_t depth,
                const double* packed, size_t width) {
    // Keep a small output block in registers while accumulating across the panel.
    double values[rows][updateWidth]{};
    for (size_t i = 0; i < rows; ++i) {
        std::copy_n(tile + i * n, width, values[i]);
    }
    for (size_t k = 0; k < depth; ++k) {
        const double* weights = packed + k * updateWidth;
        for (size_t i = 0; i < rows; ++i) {
            const double value = panel[i * n + k];
            #pragma omp simd
            for (size_t j = 0; j < updateWidth; ++j) {
                if constexpr (subtract) {
                    values[i][j] -= value * weights[j];
                } else {
                    values[i][j] += value * weights[j];
                }
            }
        }
    }
    for (size_t i = 0; i < rows; ++i) {
        std::copy_n(values[i], width, tile + i * n);
    }
}

template <bool subtract>
void updateTile(double* output, const double* input, size_t n, size_t tileSize,
                size_t rowBegin, size_t colBegin, size_t panelBegin) {
    const size_t rowEnd = std::min(n, rowBegin + tileSize);
    const size_t colEnd = std::min(n, colBegin + tileSize);
    const size_t depth = std::min(n - panelBegin, tileSize);
    alignas(64) double packed[maxTileSize * updateWidth];

    for (size_t j = colBegin; j < colEnd; j += updateWidth) {
        const size_t width = std::min(updateWidth, colEnd - j);
        // Transpose a strip of the right panel for contiguous SIMD loads.
        for (size_t k = 0; k < depth; ++k) {
            for (size_t col = 0; col < width; ++col) {
                packed[k * updateWidth + col] = input[(j + col) * n + panelBegin + k];
            }
            std::fill(packed + k * updateWidth + width, packed + (k + 1) * updateWidth, 0.0);
        }

        // In diagonal tiles only strips intersecting the lower triangle are needed.
        size_t i = std::max(j, rowBegin);
        for (; i + 4 <= rowEnd; i += 4) {
            updateRows<4, subtract>(output + i * n + j, n, input + i * n + panelBegin,
                                   depth, packed, width);
        }
        for (; i < rowEnd; ++i) {
            updateRows<1, subtract>(output + i * n + j, n, input + i * n + panelBegin,
                                   depth, packed, width);
        }
    }
}

} // namespace

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    const size_t tileSize = matrixTileSize(n);
    const size_t tiles = (n + tileSize - 1) / tileSize;
    double* const matrix = A.data();
    // A separate token per tile tracks dependencies between sibling OpenMP tasks.
    std::vector<unsigned char> tileTokens(tiles * tiles);
    [[maybe_unused]] unsigned char* const token = tileTokens.data();
    const auto diagonalPanels = std::make_unique_for_overwrite<double[]>(tiles * tileSize * tileSize);
    std::atomic<size_t> badPivot{n};

    #pragma omp parallel num_threads(teamSize(n))
    {
        // Factor the diagonal, solve its column, then update the trailing tiles.
        // Per-tile dependencies let the next column start as soon as it is ready.
        #pragma omp single
        for (size_t k = 0; k < tiles; ++k) {
            const size_t begin = k * tileSize;
            const size_t end = std::min(n, begin + tileSize);
            const size_t diagonal = k * tiles + k;
            double* const transposed = diagonalPanels.get() + k * tileSize * tileSize;
            #pragma omp task depend(inout: token[diagonal])
            {
                if (badPivot.load(std::memory_order_relaxed) == n) {
                    const size_t failed = factorDiagonal(matrix, n, begin, end, transposed, tileSize);
                    if (failed != n) {
                        badPivot.store(failed, std::memory_order_relaxed);
                    }
                }
            }

            for (size_t i = k + 1; i < tiles; ++i) {
                const size_t row = i * tileSize;
                const size_t panel = i * tiles + k;
                #pragma omp task depend(in: token[diagonal]) depend(inout: token[panel])
                {
                    if (badPivot.load(std::memory_order_relaxed) == n) {
                        solvePanel(matrix, n, row, std::min(n, row + tileSize),
                                   begin, end, transposed, tileSize);
                    }
                }
            }

            for (size_t j = k + 1; j < tiles; ++j) {
                const size_t col = j * tileSize;
                const size_t right = j * tiles + k;
                #pragma omp task depend(in: token[right]) depend(inout: token[j * tiles + j])
                {
                    if (badPivot.load(std::memory_order_relaxed) == n) {
                        updateTile<true>(matrix, matrix, n, tileSize, col, col, begin);
                    }
                }
                for (size_t i = j + 1; i < tiles; ++i) {
                    const size_t row = i * tileSize;
                    const size_t left = i * tiles + k;
                    const size_t output = i * tiles + j;
                    #pragma omp task depend(in: token[left], token[right]) depend(inout: token[output])
                    {
                        if (badPivot.load(std::memory_order_relaxed) == n) {
                            updateTile<true>(matrix, matrix, n, tileSize, row, col, begin);
                        }
                    }
                }
            }
        }

        // The single region's implicit barrier also completes its tasks.
        const bool success = badPivot.load(std::memory_order_relaxed) == n;
        #pragma omp for schedule(static) nowait
        for (size_t i = 0; i < n; ++i) {
            if (success) {
                std::fill(matrix + i * n + i + 1, matrix + (i + 1) * n, 0.0);
            }
        }
    }

    const size_t failed = badPivot.load(std::memory_order_relaxed);
    if (failed != n) {
        printf("Error: Matrix is not positive definite at diagonal element %zu\n", failed);
        return false;
    }
    return true;
}

// Generate a symmetric positive definite matrix
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    // Form B * B^T + nI using the original seeded random stream.
    const auto B = std::make_unique_for_overwrite<double[]>(n * n);
    unsigned int seed = 42;
    
    // Generate random matrix B
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }
    
    const size_t tileSize = matrixTileSize(n);
    #pragma omp parallel for collapse(2) schedule(dynamic, 1) num_threads(teamSize(n))
    for (size_t row = 0; row < n; row += tileSize) {
        for (size_t col = 0; col < n; col += tileSize) {
            if (col > row) {
                continue;
            }
            const size_t rowEnd = std::min(n, row + tileSize);
            const size_t colEnd = std::min(n, col + tileSize);
            for (size_t i = row; i < rowEnd; ++i) {
                std::fill(A.data() + i * n + col, A.data() + i * n + colEnd, 0.0);
            }
            for (size_t k = 0; k < n; k += tileSize) {
                updateTile<false>(A.data(), B.get(), n, tileSize, row, col, k);
            }
            for (size_t i = row; i < rowEnd; ++i) {
                for (size_t j = col; j < std::min(i, colEnd); ++j) {
                    A[j * n + i] = A[i * n + j];
                }
                if (row == col) {
                    A[i * n + i] += n;
                }
            }
        }
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    double maxError = 0.0;
    double relError = 0.0;

    // Compare each entry of L * L^T as it is computed.
    #pragma omp parallel for schedule(static, 8) reduction(max: maxError, relError) num_threads(teamSize(n))
    for (size_t i = 0; i < n; ++i) {
        const double* row = L.data() + i * n;
        for (size_t j = 0; j <= i; ++j) {
            const double* other = L.data() + j * n;
            double sum = 0.0;
            #pragma omp simd reduction(+: sum)
            for (size_t k = 0; k < n; ++k) {
                sum += row[k] * other[k];
            }
            const double original = A_orig[i * n + j];
            const double error = std::abs(sum - original);
            maxError = std::max(maxError, error);
            relError = std::max(relError, error / (std::abs(original) + 1e-10));
            if (j != i) {
                const double mirror = A_orig[j * n + i];
                const double mirrorError = std::abs(sum - mirror);
                maxError = std::max(maxError, mirrorError);
                relError = std::max(relError, mirrorError / (std::abs(mirror) + 1e-10));
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
    const double elapsed = std::chrono::duration<double>(end - start).count();
    double gflops = ops / elapsed / 1e9;
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
