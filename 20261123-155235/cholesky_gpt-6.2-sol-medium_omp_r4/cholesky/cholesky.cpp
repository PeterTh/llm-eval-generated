#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <omp.h>

#include "../common/results_output.hpp"

namespace {

constexpr size_t panelWidth = 64;
constexpr size_t rowChunk = 16;
constexpr size_t columnGroup = 8;

int teamSize(size_t n, size_t rowsPerThread) {
    return static_cast<int>(std::min<size_t>(omp_get_max_threads(),
                                           1 + n / rowsPerThread));
}

// Factor the diagonal block, keeping a transposed copy for the panel solve.
size_t factorDiagonal(double* block, size_t stride, size_t width, double* transpose) {
    for (size_t k = 0; k < width; ++k) {
        double& pivot = block[k * stride + k];
        if (pivot <= 0.0) {
            return k;
        }
        pivot = std::sqrt(pivot);
        double* column = transpose + k * panelWidth;
        column[k] = pivot;
        for (size_t i = k + 1; i < width; ++i) {
            double* row = block + i * stride;
            row[k] /= pivot;
            column[i] = row[k];
        }
        for (size_t i = k + 1; i < width; ++i) {
            double* row = block + i * stride;
            const double value = column[i];
            #pragma omp simd
            for (size_t j = k + 1; j <= i; ++j) {
                row[j] -= value * column[j];
            }
        }
    }
    return width;
}

void solvePanelRow(double* row, const double* transpose) {
    for (size_t k = 0; k < panelWidth; ++k) {
        const double* column = transpose + k * panelWidth;
        const double value = row[k] / column[k];
        row[k] = value;
        #pragma omp simd
        for (size_t j = k + 1; j < panelWidth; ++j) {
            row[j] -= value * column[j];
        }
    }
}

// Accumulate a small output block in registers. Each SIMD lane updates a
// separate column using contiguous values from the packed panel.
template <size_t rows>
void updateBlock(double* output, size_t n, const double* left, const double* right,
                 size_t columns) {
    double sums[rows][columnGroup] = {};
    for (size_t k = 0; k < panelWidth; ++k) {
        const double* values = right + k * columnGroup;
        for (size_t i = 0; i < rows; ++i) {
            const double value = left[i * n + k];
            #pragma omp simd
            for (size_t j = 0; j < columnGroup; ++j) {
                sums[i][j] += value * values[j];
            }
        }
    }
    for (size_t i = 0; i < rows; ++i) {
        #pragma omp simd
        for (size_t j = 0; j < columns; ++j) {
            output[i * n + j] -= sums[i][j];
        }
    }
}

} // namespace

// Blocked right-looking Cholesky on the lower triangle of a row-major matrix.
bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    double* matrix = A.data();
    const size_t groups = (n + columnGroup - 1) / columnGroup;
    std::vector<double> packed(groups * columnGroup * panelWidth);
    alignas(64) double diagonal[panelWidth * panelWidth];
    size_t badDiagonal = n;
    const int threads = teamSize(n, 48);

    #pragma omp parallel num_threads(threads)
    {
        for (size_t first = 0; first < n; first += panelWidth) {
            const size_t end = std::min(first + panelWidth, n);
            const size_t width = end - first;

            #pragma omp single
            {
                const size_t pivot = factorDiagonal(matrix + first * n + first, n, width, diagonal);
                if (pivot != width) {
                    badDiagonal = first + pivot;
                }
            }
            if (badDiagonal != n || end == n) {
                break;
            }

            // Solve and pack each group of panel rows before updating the
            // trailing matrix. Pad the last group to a full SIMD width.
            #pragma omp for schedule(static)
            for (size_t j = end; j < n; j += columnGroup) {
                const size_t rows = std::min(columnGroup, n - j);
                for (size_t i = j; i < j + rows; ++i) {
                    solvePanelRow(matrix + i * n + first, diagonal);
                }
                double* dest = packed.data() + j * panelWidth;
                for (size_t k = 0; k < panelWidth; ++k) {
                    for (size_t i = 0; i < rows; ++i) {
                        dest[k * columnGroup + i] = matrix[(j + i) * n + first + k];
                    }
                }
            }

            // Whole row chunks have one owner. Cyclic assignment balances
            // the growing amount of work toward the bottom of the triangle.
            #pragma omp for schedule(static, 1)
            for (size_t row = end; row < n; row += rowChunk) {
                const size_t rowEnd = std::min(row + rowChunk, n);
                for (size_t j = end; j < rowEnd; j += columnGroup) {
                    const double* right = packed.data() + j * panelWidth;
                    size_t i = std::max(row, j + columnGroup);
                    for (; i + 4 <= rowEnd; i += 4) {
                        updateBlock<4>(matrix + i * n + j, n,
                                       matrix + i * n + first, right, columnGroup);
                    }
                    for (; i < rowEnd; ++i) {
                        updateBlock<1>(matrix + i * n + j, n,
                                       matrix + i * n + first, right, columnGroup);
                    }
                    for (i = std::max(row, j); i < std::min(j + columnGroup, rowEnd); ++i) {
                        updateBlock<1>(matrix + i * n + j, n,
                                       matrix + i * n + first, right, i - j + 1);
                    }
                }
            }
        }

        if (badDiagonal == n) {
            #pragma omp for schedule(static) nowait
            for (size_t i = 0; i < n; ++i) {
                std::fill(matrix + i * n + i + 1, matrix + (i + 1) * n, 0.0);
            }
        }
    }

    if (badDiagonal != n) {
        printf("Error: Matrix is not positive definite at diagonal element %zu\n", badDiagonal);
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

    // A = B * B^T + nI. Compute each symmetric pair once.
    const int threads = teamSize(n, 24);
    #pragma omp parallel for num_threads(threads) schedule(static, 1)
    for (size_t first = 0; first < n; first += columnGroup) {
        const size_t end = std::min(first + columnGroup, n);
        // Reuse each row of B across a group of output rows.
        for (size_t j = 0; j < end; ++j) {
            const double* other = B.data() + j * n;
            for (size_t i = std::max(first, j); i < end; ++i) {
                const double* row = B.data() + i * n;
                double sum = 0.0;
                #pragma omp simd reduction(+:sum)
                for (size_t k = 0; k < n; ++k) {
                    sum += row[k] * other[k];
                }
                A[i * n + j] = sum;
                if (i != j) {
                    A[j * n + i] = sum;
                }
            }
        }
        for (size_t i = first; i < end; ++i) {
            A[i * n + i] += n;
        }
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    double maxError = 0.0;
    double relError = 0.0;
    const int threads = teamSize(n, 24);

    // L is lower triangular, so the product of rows i and j ends at j.
    #pragma omp parallel for num_threads(threads) schedule(static, 1) reduction(max:maxError, relError)
    for (size_t i = 0; i < n; ++i) {
        const double* row = L.data() + i * n;
        for (size_t j = 0; j <= i; ++j) {
            const double* other = L.data() + j * n;
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (size_t k = 0; k <= j; ++k) {
                sum += row[k] * other[k];
            }
            const double original = A_orig[i * n + j];
            const double error = std::fabs(sum - original);
            maxError = std::max(maxError, error);
            relError = std::max(relError, error / (std::fabs(original) + 1e-10));
            if (i != j) {
                const double reflected = A_orig[j * n + i];
                const double reflectedError = std::fabs(sum - reflected);
                maxError = std::max(maxError, reflectedError);
                relError = std::max(relError, reflectedError / (std::fabs(reflected) + 1e-10));
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
    const double seconds = std::chrono::duration<double>(end - start).count();
    double gflops = ops / seconds / 1e9;
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
