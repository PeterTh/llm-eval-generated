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
constexpr size_t width = 16;
constexpr size_t rowGroup = 4;

int teamSize(size_t n) {
    return static_cast<int>(std::min<size_t>(omp_get_max_threads(),
                                             std::max<size_t>(1, (n + 31) / 32)));
}

// Transpose a strip so that each SIMD lane accumulates a separate dot
// product. All lanes visit the shared dimension in its original order.
void packStrip(const double* matrix, size_t n, size_t col, size_t depth,
               double* packed) {
    const size_t count = std::min(width, n - col);
    for (size_t k = 0; k < depth; ++k) {
        for (size_t j = 0; j < count; ++j)
            packed[k * width + j] = matrix[(col + j) * n + k];
        std::fill(packed + k * width + count, packed + (k + 1) * width, 0.0);
    }
}

template<size_t rows>
void products(const double* input, size_t stride, const double* packed,
              size_t depth, double (&sums)[rows][width]) {
    for (size_t k = 0; k < depth; ++k) {
        for (size_t r = 0; r < rows; ++r) {
            const double x = input[r * stride + k];
            #pragma omp simd
            for (size_t j = 0; j < width; ++j)
                sums[r][j] += x * packed[k * width + j];
        }
    }
}

template<size_t rows>
bool solveRows(double* matrix, size_t n, size_t first, size_t col,
               const double* packed) {
    double sums[rows][width] = {};
    products(matrix + first * n, n, packed, col, sums);
    for (size_t r = 0; r < rows; ++r) {
        const size_t i = first + r;
        double* row = matrix + i * n;
        const size_t end = std::min({col + width, i + 1, n});
        for (size_t j = col; j < end; ++j) {
            double sum = sums[r][j - col];
            const double* pivot = matrix + j * n;
            for (size_t k = col; k < j; ++k)
                sum += row[k] * pivot[k];
            const double value = row[j] - sum;
            if (i == j) {
                if (value <= 0.0) {
                    printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
                    return false;
                }
                row[j] = std::sqrt(value);
            } else {
                row[j] = value / pivot[j];
            }
        }
    }
    return true;
}
} // namespace

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    std::vector<double> packed(width * n);
    bool success = true;
    #pragma omp parallel num_threads(teamSize(n))
    {
        for (size_t col = 0; col < n; col += width) {
            const size_t end = std::min(col + width, n);
            // Publish the packed prefix and diagonal block before solving
            // independent groups of rows below it.
            #pragma omp single
            {
                packStrip(A.data(), n, col, col, packed.data());
                size_t i = col;
                for (; success && i + rowGroup <= end; i += rowGroup)
                    success = solveRows<rowGroup>(A.data(), n, i, col, packed.data());
                for (; success && i < end; ++i)
                    success = solveRows<1>(A.data(), n, i, col, packed.data());
            }
            if (!success) break;
            #pragma omp for schedule(static)
            for (size_t i = end; i < n; i += rowGroup) {
                if (i + rowGroup <= n)
                    solveRows<rowGroup>(A.data(), n, i, col, packed.data());
                else
                    for (size_t r = i; r < n; ++r)
                        solveRows<1>(A.data(), n, r, col, packed.data());
            }
        }
        #pragma omp for schedule(static)
        for (size_t i = 0; i < n; ++i)
            std::fill(A.begin() + i * n + i + 1, A.begin() + (i + 1) * n, 0.0);
    }
    return success;
}

void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    for (double& x : B)
        x = (rand_r(&seed) / (double)RAND_MAX) - 0.5;

    // Each strip owns one part of the lower triangle and its reflection.
    #pragma omp parallel num_threads(teamSize(n))
    {
        std::vector<double> packed(width * n);
        #pragma omp for schedule(dynamic, 1)
        for (size_t col = 0; col < n; col += width) {
            packStrip(B.data(), n, col, n, packed.data());
            for (size_t i = col; i < n; i += rowGroup) {
                const size_t count = std::min(rowGroup, n - i);
                double sums[rowGroup][width] = {};
                if (count == rowGroup) {
                    products(B.data() + i * n, n, packed.data(), n, sums);
                } else {
                    for (size_t r = 0; r < count; ++r) {
                        double tail[1][width] = {};
                        products(B.data() + (i + r) * n, n, packed.data(), n, tail);
                        std::copy_n(tail[0], width, sums[r]);
                    }
                }
                for (size_t r = 0; r < count; ++r) {
                    const size_t row = i + r;
                    for (size_t j = col; j < std::min(col + width, row + 1); ++j) {
                        const double value = sums[r][j - col];
                        A[row * n + j] = A[j * n + row] = value;
                    }
                    if (row < col + width) A[row * n + row] += n;
                }
            }
        }
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    double maxError = 0.0;
    double relError = 0.0;
    #pragma omp parallel for num_threads(teamSize(n)) schedule(static) reduction(max:maxError,relError)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (size_t k = 0; k < n; ++k)
                sum += L[i * n + k] * L[j * n + k];
            const double original = A_orig[i * n + j];
            const double error = std::fabs(sum - original);
            maxError = std::max(maxError, error);
            relError = std::max(relError, error / (std::fabs(original) + 1e-10));
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
    double gflops = ops / (std::chrono::duration<double>(end - start).count()) / 1e9;
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
