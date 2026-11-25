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
constexpr size_t tile = 32;
constexpr size_t lanes = 8;

int teamSize(size_t n) {
    const size_t blocks = std::max<size_t>(1, (n + tile - 1) / tile);
    return static_cast<int>(std::min<size_t>(omp_get_max_threads(), blocks));
}

// Multiply four rows at a time by strips of a transposed panel. Each SIMD
// lane accumulates a separate output column, with no horizontal reduction.
template<bool subtract, size_t rows>
void multiplyStrip(const double* left, const double* packed, double* out,
                   size_t stride, size_t depth, size_t width) {
    for (size_t col = 0; col < width; col += lanes) {
        double sums[rows][lanes] = {};
        for (size_t k = 0; k < depth; ++k) {
            for (size_t r = 0; r < rows; ++r) {
                const double x = left[r * stride + k];
                #pragma omp simd
                for (size_t c = 0; c < lanes; ++c)
                    sums[r][c] += x * packed[k * tile + col + c];
            }
        }
        for (size_t r = 0; r < rows; ++r) {
            for (size_t c = 0; c < std::min(lanes, width - col); ++c) {
                double& dest = out[r * stride + col + c];
                if constexpr (subtract) dest -= sums[r][c];
                else dest = sums[r][c];
            }
        }
    }
}

template<bool subtract>
void multiplyTile(const double* left, const double* packed, double* out,
                  size_t stride, size_t depth, size_t height, size_t width) {
    size_t r = 0;
    for (; r + 4 <= height; r += 4)
        multiplyStrip<subtract, 4>(left + r * stride, packed, out + r * stride, stride, depth, width);
    for (; r < height; ++r)
        multiplyStrip<subtract, 1>(left + r * stride, packed, out + r * stride, stride, depth, width);
}

// Solve a row of the current panel against its completed diagonal block.
void solveRow(double* A, size_t n, size_t row, size_t begin, size_t end) {
    for (size_t j = begin; j < end; ++j) {
        double sum = 0;
        for (size_t k = begin; k < j; ++k)
            sum += A[row * n + k] * A[j * n + k];
        A[row * n + j] = (A[row * n + j] - sum) / A[j * n + j];
    }
}
}

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    const size_t blocks = (n + tile - 1) / tile;
    std::vector<double> panel(blocks * tile * tile);
    bool success = true;
    #pragma omp parallel num_threads(teamSize(n))
    {
        for (size_t p = 0; p < blocks; ++p) {
            const size_t begin = p * tile;
            const size_t end = std::min(n, begin + tile);
            #pragma omp single
            {
                for (size_t j = begin; j < end; ++j) {
                    solveRow(A.data(), n, j, begin, j);
                    double sum = 0;
                    for (size_t k = begin; k < j; ++k)
                        sum += A[j * n + k] * A[j * n + k];
                    const double val = A[j * n + j] - sum;
                    if (val <= 0.0) {
                        printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
                        success = false;
                        break;
                    }
                    A[j * n + j] = std::sqrt(val);
                }
            }
            if (!success) break;

            #pragma omp for schedule(static)
            for (size_t row = end; row < n; ++row) {
                solveRow(A.data(), n, row, begin, end);
                double* packed = panel.data() + (row / tile) * tile * tile + row % tile;
                for (size_t k = begin; k < end; ++k)
                    packed[(k - begin) * tile] = A[row * n + k];
            }

            #pragma omp for collapse(2) schedule(dynamic, 1)
            for (size_t i = p + 1; i < blocks; ++i) {
                for (size_t j = p + 1; j < blocks; ++j) {
                    if (j > i) continue;
                    const size_t row = i * tile, col = j * tile;
                    multiplyTile<true>(&A[row * n + begin], &panel[j * tile * tile],
                                       &A[row * n + col], n, end - begin,
                                       std::min(tile, n - row), std::min(tile, n - col));
                }
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
    for (double& value : B)
        value = (rand_r(&seed) / (double)RAND_MAX) - 0.5;

    const size_t blocks = (n + tile - 1) / tile;
    #pragma omp parallel num_threads(teamSize(n))
    {
        // Start with the longest columns. Reuse each packed panel for all
        // lower-triangle tiles in its column before moving to the next one.
        std::vector<double> packed(n * tile);
        #pragma omp for schedule(dynamic, 1)
        for (size_t j = 0; j < blocks; ++j) {
            const size_t col = j * tile;
            const size_t width = std::min(tile, n - col);
            for (size_t k = 0; k < n; ++k)
                for (size_t c = 0; c < width; ++c)
                    packed[k * tile + c] = B[(col + c) * n + k];

            for (size_t row = col; row < n; row += tile)
                multiplyTile<false>(&B[row * n], packed.data(), &A[row * n + col],
                                    n, n, std::min(tile, n - row), width);
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
    #pragma omp parallel for num_threads(teamSize(n)) schedule(static, 1) reduction(max:maxError,relError)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (size_t k = 0; k <= j; ++k)
                sum += L[i * n + k] * L[j * n + k];
            const double error = std::fabs(sum - A_orig[i * n + j]);
            maxError = std::max(maxError, error);
            relError = std::max(relError, error / (std::fabs(A_orig[i * n + j]) + 1e-10));
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
