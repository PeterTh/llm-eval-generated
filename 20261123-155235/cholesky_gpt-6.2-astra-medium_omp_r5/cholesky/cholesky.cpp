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
constexpr size_t panelWidth = 16;
constexpr size_t rowBlock = 4;

// Pack a panel of rows by column so adjacent SIMD lanes compute independent
// dot products. Only the already factored prefix is needed for Cholesky.
void packPanel(const double* rows, size_t stride, size_t width, size_t depth,
               double* packed) {
    for (size_t j = 0; j < width; ++j)
        for (size_t k = 0; k < depth; ++k)
            packed[k * panelWidth + j] = rows[j * stride + k];
}

// Keep the separate multiply and add used by an ordered dot product's
// vectorized main loop. Vectorizing across outputs preserves the k order.
#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("fp-contract=off")))
#endif
void panelProducts(const double* rows, size_t stride, const double* packed,
                   size_t depth, double (&sums)[rowBlock][panelWidth]) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    for (size_t k = 0; k < depth; ++k) {
        for (size_t r = 0; r < rowBlock; ++r) {
            const double value = rows[r * stride + k];
            #pragma omp simd
            for (size_t j = 0; j < panelWidth; ++j)
                sums[r][j] += value * packed[k * panelWidth + j];
        }
    }
}

double dot(const double* x, const double* y, size_t count, double sum = 0.0) {
    for (size_t k = 0; k < count; ++k)
        sum += x[k] * y[k];
    return sum;
}

int panelThreads(size_t n) {
    const size_t workUnits = (n + panelWidth - 1) / panelWidth;
    return static_cast<int>(std::min<size_t>(omp_get_max_threads(),
                                             std::max<size_t>(1, workUnits)));
}

// Complete a block of rows using the packed prefix and a short triangular
// solve inside the current panel. Diagonal blocks also produce the pivots.
template<bool diagonal>
size_t solveRows(double* matrix, size_t n, size_t first, size_t col,
                 size_t width, const double* packed) {
    double* rows = matrix + first * n;
    const size_t count = std::min(rowBlock, n - first);
    alignas(64) double sums[rowBlock][panelWidth]{};
    if (count == rowBlock)
        panelProducts(rows, n, packed, col, sums);

    for (size_t r = 0; r < count; ++r) {
        const size_t i = first + r;
        double* row = rows + r * n;
        const size_t end = diagonal ? i - col + 1 : width;
        for (size_t j = 0; j < end; ++j) {
            const double* pivot = matrix + (col + j) * n;
            const double sum = count == rowBlock
                ? dot(row + col, pivot + col, j, sums[r][j])
                : dot(row, pivot, col + j);
            const double value = row[col + j] - sum;
            if (diagonal && col + j == i) {
                if (value <= 0.0) return i;
                row[i] = std::sqrt(value);
            } else {
                row[col + j] = value / pivot[col + j];
            }
        }
        if (diagonal)
            std::fill(row + i + 1, row + n, 0.0);
    }
    return n;
}

// Visit one lower-triangular panel of matrix * matrix^T.
template<class Visitor>
void gramPanel(const double* matrix, size_t n, size_t col, double* packed, Visitor visit) {
    const size_t width = std::min(panelWidth, n - col);
    const double* panel = matrix + col * n;
    const size_t depth = n - n % panelWidth;
    packPanel(panel, n, width, depth, packed);
    for (size_t i = col; i < n; i += rowBlock) {
        const size_t count = std::min(rowBlock, n - i);
        const double* rows = matrix + i * n;
        alignas(64) double sums[rowBlock][panelWidth]{};
        if (count == rowBlock)
            panelProducts(rows, n, packed, depth, sums);
        for (size_t r = 0; r < count; ++r) {
            const double* row = rows + r * n;
            for (size_t j = 0; j < width && col + j <= i + r; ++j) {
                const double* other = panel + j * n;
                const double sum = count == rowBlock
                    ? dot(row + depth, other + depth, n - depth, sums[r][j])
                    : dot(row, other, n);
                visit(i + r, col + j, sum);
            }
        }
    }
}
} // namespace

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    std::vector<double> packed(n * panelWidth);
    size_t failed = n;
    #pragma omp parallel num_threads(panelThreads(n))
    {
        for (size_t col = 0; col < n; col += panelWidth) {
            const size_t width = std::min(panelWidth, n - col);
            double* panel = A.data() + col * n;
            #pragma omp single
            {
                packPanel(panel, n, width, col, packed.data());
                for (size_t i = col; i < col + width; i += rowBlock) {
                    failed = solveRows<true>(A.data(), n, i, col, width, packed.data());
                    if (failed != n) break;
                }
            }
            if (failed != n) break;

            #pragma omp for schedule(static)
            for (size_t i = col + width; i < n; i += rowBlock) {
                solveRows<false>(A.data(), n, i, col, width, packed.data());
            }
        }
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

    #pragma omp parallel num_threads(panelThreads(n))
    {
        std::vector<double> packed(n * panelWidth);
        #pragma omp for schedule(dynamic, 1)
        for (size_t col = 0; col < n; col += panelWidth) {
            gramPanel(B.data(), n, col, packed.data(), [&](size_t i, size_t j, double sum) {
                A[i * n + j] = A[j * n + i] = sum;
            });
            for (size_t j = col; j < std::min(n, col + panelWidth); ++j)
                A[j * n + j] += n;
        }
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    double maxError = 0.0;
    double relError = 0.0;
    #pragma omp parallel num_threads(panelThreads(n)) reduction(max:maxError,relError)
    {
        std::vector<double> packed(n * panelWidth);
        #pragma omp for schedule(dynamic, 1)
        for (size_t col = 0; col < n; col += panelWidth) {
            gramPanel(L.data(), n, col, packed.data(), [&](size_t i, size_t j, double sum) {
                for (const size_t index : {i * n + j, j * n + i}) {
                    const double error = std::fabs(sum - A_orig[index]);
                    maxError = std::max(maxError, error);
                    relError = std::max(relError, error / (std::fabs(A_orig[index]) + 1e-10));
                }
            });
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
