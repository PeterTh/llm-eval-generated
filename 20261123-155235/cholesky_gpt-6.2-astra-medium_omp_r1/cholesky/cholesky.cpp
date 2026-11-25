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
constexpr size_t panelWidth = 32;
constexpr size_t rowGroup = 4;
constexpr size_t lanes = 8;

// Keep several independent dot products in registers. Packing eight right-hand
// rows side by side makes the vector loads contiguous throughout the k loop.
void packRows(const double* matrix, size_t n, size_t col, size_t width,
              size_t depth, double* packed) {
    for (size_t j = 0; j < width; j += lanes) {
        const size_t count = std::min(lanes, width - j);
        for (size_t k = 0; k < depth; ++k) {
            double* dst = packed + j * depth + k * lanes;
            for (size_t c = 0; c < count; ++c)
                dst[c] = matrix[(col + j + c) * n + k];
            std::fill(dst + count, dst + lanes, 0.0);
        }
    }
}

void panelProducts(const double* rows, size_t n, size_t count, size_t width,
                   size_t depth, const double* packed, double* out, size_t stride) {
    for (size_t j = 0; j < width; j += lanes) {
        double sum[rowGroup][lanes] = {};
        const double* right = packed + j * depth;
        for (size_t k = 0; k < depth; ++k) {
            for (size_t r = 0; r < rowGroup; ++r) {
                // Repeat the first row for unused lanes in the final row group.
                const double value = rows[(r < count ? r : 0) * n + k];
                #pragma omp simd
                for (size_t c = 0; c < lanes; ++c)
                    sum[r][c] += value * right[k * lanes + c];
            }
        }
        for (size_t r = 0; r < count; ++r)
            std::copy_n(sum[r], std::min(lanes, width - j), out + r * stride + j);
    }
}

// Finish one row using the completed rows of the diagonal block. Each sum
// already includes all columns preceding this panel.
bool solveRow(double* row, const double* diagonal, size_t n, const double* sums,
              size_t width, bool hasPivot) {
    for (size_t j = 0; j < width; ++j) {
        const double* previous = diagonal + j * n;
        double sum = sums[j];
        for (size_t k = 0; k < j; ++k)
            sum += row[k] * previous[k];
        const double residual = row[j] - sum;
        if (hasPivot && j + 1 == width) {
            if (residual <= 0.0) return false;
            row[j] = std::sqrt(residual);
        } else {
            row[j] = residual / previous[j];
        }
    }
    return true;
}

int workers(size_t n) {
    // Bound synchronization overhead when only a few rows are available.
    return static_cast<int>(std::min<size_t>(omp_get_max_threads(), 1 + n / 64));
}
} // namespace

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    std::vector<double> packed(n * panelWidth), sums(n * panelWidth);
    size_t failed = n;
    #pragma omp parallel num_threads(workers(n))
    {
        for (size_t col = 0; col < n; col += panelWidth) {
            const size_t width = std::min(panelWidth, n - col);
            #pragma omp single
            packRows(A.data(), n, col, width, col, packed.data());

            #pragma omp for schedule(static)
            for (size_t i = col; i < n; i += rowGroup)
                panelProducts(&A[i * n], n, std::min(rowGroup, n - i), width, col,
                              packed.data(), &sums[i * panelWidth], panelWidth);

            double* diagonal = &A[col * n + col];
            #pragma omp single
            {
                for (size_t r = 0; r < width; ++r) {
                    if (!solveRow(diagonal + r * n, diagonal, n,
                                  &sums[(col + r) * panelWidth], r + 1, true)) {
                        failed = col + r;
                        break;
                    }
                }
            }
            if (failed != n) break;

            #pragma omp for schedule(static)
            for (size_t i = col + width; i < n; ++i)
                solveRow(&A[i * n + col], diagonal, n, &sums[i * panelWidth], width, false);
        }
        #pragma omp for schedule(static)
        for (size_t i = 0; i < failed; ++i)
            std::fill(A.begin() + i * n + i + 1, A.begin() + (i + 1) * n, 0.0);
    }
    if (failed != n)
        printf("Error: Matrix is not positive definite at diagonal element %zu\n", failed);
    return failed == n;
}

void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    for (double& value : B)
        value = (rand_r(&seed) / (double)RAND_MAX) - 0.5;

    #pragma omp parallel num_threads(workers(n))
    {
        std::vector<double> packed(n * panelWidth);
        #pragma omp for schedule(dynamic, 1)
        for (size_t col = 0; col < n; col += panelWidth) {
            const size_t width = std::min(panelWidth, n - col);
            packRows(B.data(), n, col, width, n, packed.data());
            for (size_t i = col; i < n; i += rowGroup)
                panelProducts(&B[i * n], n, std::min(rowGroup, n - i), width, n,
                              packed.data(), &A[i * n + col], n);
        }
        #pragma omp for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            A[i * n + i] += n;
            for (size_t j = i + 1; j < n; ++j)
                A[i * n + j] = A[j * n + i];
        }
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    double maxError = 0.0;
    double relError = 0.0;
    #pragma omp parallel num_threads(workers(n)) reduction(max:maxError,relError)
    {
        std::vector<double> packed(n * panelWidth);
        double product[rowGroup * panelWidth];
        #pragma omp for schedule(dynamic, 1)
        for (size_t col = 0; col < n; col += panelWidth) {
            const size_t width = std::min(panelWidth, n - col);
            packRows(L.data(), n, col, width, n, packed.data());
            for (size_t i = col; i < n; i += rowGroup) {
                const size_t count = std::min(rowGroup, n - i);
                panelProducts(&L[i * n], n, count, width, n,
                              packed.data(), product, panelWidth);
                for (size_t r = 0; r < count; ++r) {
                    for (size_t c = 0; c < std::min(width, i + r - col + 1); ++c) {
                        const double sum = product[r * panelWidth + c];
                        for (double original : {A_orig[(i + r) * n + col + c],
                                                A_orig[(col + c) * n + i + r]}) {
                            const double error = std::fabs(sum - original);
                            maxError = std::max(maxError, error);
                            relError = std::max(relError, error / (std::fabs(original) + 1e-10));
                        }
                    }
                }
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
    std::chrono::duration<double> duration = end - start;
    
    if (!success) {
        printf("Cholesky decomposition failed\n");
        return 1;
    }
    
    printf("Computation time: %.3f ms\n", duration.count() * 1000.0);
    
    // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
    double ops = (double)n * n * n / 3.0;
    double gflops = ops / duration.count() / 1e9;
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
