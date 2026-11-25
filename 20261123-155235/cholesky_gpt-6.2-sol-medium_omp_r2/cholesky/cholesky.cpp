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

constexpr size_t blockSize = 32;

int threadCount(size_t n) {
    // Keep enough work per thread to amortize scheduling and panel barriers.
    return static_cast<int>(std::min<size_t>(omp_get_max_threads(), 1 + n / 64));
}

template <typename Operation>
void forEachDotProduct(const double* matrix, size_t stride, const double* row,
                       size_t first, size_t last, size_t length, Operation useSum) {
    // Four independent dot products reuse each load from the input row.
    size_t j = first;
    for (; j + 4 <= last; j += 4) {
        const double* b0 = matrix + j * stride;
        const double* b1 = b0 + stride;
        const double* b2 = b1 + stride;
        const double* b3 = b2 + stride;
        double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
        #pragma omp simd reduction(+:s0,s1,s2,s3)
        for (size_t k = 0; k < length; ++k) {
            const double a = row[k];
            s0 += a * b0[k];
            s1 += a * b1[k];
            s2 += a * b2[k];
            s3 += a * b3[k];
        }
        useSum(j, s0);
        useSum(j + 1, s1);
        useSum(j + 2, s2);
        useSum(j + 3, s3);
    }
    for (; j < last; ++j) {
        const double* other = matrix + j * stride;
        double sum = 0.0;
        #pragma omp simd reduction(+:sum)
        for (size_t k = 0; k < length; ++k) {
            sum += row[k] * other[k];
        }
        useSum(j, sum);
    }
}

// Forward substitution across a row of the current panel.
void solvePanelRow(const double diagonal[][blockSize], double* row, size_t width) {
    for (size_t j = 0; j < width; ++j) {
        const double* pivot = diagonal[j];
        const double value = row[j] / pivot[j];
        row[j] = value;
        #pragma omp simd
        for (size_t k = j + 1; k < width; ++k) {
            row[k] -= value * pivot[k];
        }
    }
}

size_t lowerTileCount(size_t n) {
    const size_t blocks = n / blockSize + (n % blockSize != 0);
    return blocks * (blocks + 1) / 2;
}

struct Tile {
    size_t row;
    size_t col;
};

Tile lowerTile(size_t index) {
    // Invert the triangular number giving the first tile in each row.
    const size_t row = static_cast<size_t>(std::sqrt(2.0 * index + 0.25) - 0.5);
    return {row * blockSize, (index - row * (row + 1) / 2) * blockSize};
}

} // namespace

// Blocked, left-looking Cholesky factorization of the row-major lower triangle.
bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    double* matrix = A.data();
    // Store the transposed diagonal block contiguously for the panel solves.
    alignas(64) double diagonal[blockSize][blockSize];
    size_t failedPivot = n;

    #pragma omp parallel num_threads(threadCount(n))
    {
        for (size_t first = 0; first < n; first += blockSize) {
            const size_t width = std::min(blockSize, n - first);
            const size_t last = first + width;

            // Apply previously factored columns only to the current panel.
            if (first != 0) {
                #pragma omp for schedule(static)
                for (size_t i = first; i < n; ++i) {
                    double* row = matrix + i * n;
                    forEachDotProduct(matrix, n, row, first, std::min(i + 1, last), first,
                                      [row](size_t j, double sum) { row[j] -= sum; });
                }
            }

            #pragma omp single
            {
                for (size_t i = 0; i < width; ++i) {
                    double* row = matrix + (first + i) * n + first;
                    solvePanelRow(diagonal, row, i);
                    double sum = 0.0;
                    #pragma omp simd reduction(+:sum)
                    for (size_t j = 0; j < i; ++j) {
                        sum += row[j] * row[j];
                    }
                    const double pivot = row[i] - sum;
                    if (pivot <= 0.0) {
                        failedPivot = first + i;
                        break;
                    }
                    row[i] = std::sqrt(pivot);
                    for (size_t j = 0; j <= i; ++j) {
                        diagonal[j][i] = row[j];
                    }
                }
            }

            if (failedPivot != n) {
                break;
            }

            #pragma omp for schedule(static)
            for (size_t i = last; i < n; ++i) {
                solvePanelRow(diagonal, matrix + i * n + first, width);
            }
        }

        if (failedPivot == n) {
            #pragma omp for schedule(static) nowait
            for (size_t i = 0; i < n; ++i) {
                std::fill(matrix + i * n + i + 1, matrix + (i + 1) * n, 0.0);
            }
        }
    }

    if (failedPivot != n) {
        printf("Error: Matrix is not positive definite at diagonal element %zu\n", failedPivot);
        return false;
    }
    return true;
}

// Generate A = B * B^T + n * I with the original seeded random sequence.
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    for (double& value : B) {
        value = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

    const size_t tiles = lowerTileCount(n);
    #pragma omp parallel for num_threads(threadCount(n)) schedule(static, 1)
    for (size_t tile = 0; tile < tiles; ++tile) {
        const auto [rowStart, colStart] = lowerTile(tile);
        const size_t rowEnd = std::min(rowStart + blockSize, n);
        const size_t colEnd = std::min(colStart + blockSize, n);
        for (size_t i = rowStart; i < rowEnd; ++i) {
            forEachDotProduct(B.data(), n, B.data() + i * n, colStart, std::min(i + 1, colEnd), n,
                              [&](size_t j, double sum) {
                                  if (i == j) {
                                      A[i * n + i] = sum + n;
                                  } else {
                                      A[i * n + j] = sum;
                                      A[j * n + i] = sum;
                                  }
                              });
        }
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    double maxError = 0.0;
    double relError = 0.0;

    const size_t tiles = lowerTileCount(n);
    #pragma omp parallel for num_threads(threadCount(n)) schedule(static, 1) reduction(max:maxError,relError)
    for (size_t tile = 0; tile < tiles; ++tile) {
        const auto [rowStart, colStart] = lowerTile(tile);
        const size_t rowEnd = std::min(rowStart + blockSize, n);
        const size_t colEnd = std::min(colStart + blockSize, n);
        for (size_t i = rowStart; i < rowEnd; ++i) {
            const size_t last = std::min(i + 1, colEnd);
            forEachDotProduct(L.data(), n, L.data() + i * n, colStart, last, last,
                              [&](size_t j, double sum) {
                                  const double lower = A_orig[i * n + j];
                                  const double upper = A_orig[j * n + i];
                                  const double lowerError = std::abs(sum - lower);
                                  const double upperError = std::abs(sum - upper);
                                  maxError = std::max(maxError, std::max(lowerError, upperError));
                                  relError = std::max(relError, lowerError / (std::abs(lower) + 1e-10));
                                  relError = std::max(relError, upperError / (std::abs(upper) + 1e-10));
                              });
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
    const std::chrono::duration<double> elapsed = end - start;
    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed);
    
    if (!success) {
        printf("Cholesky decomposition failed\n");
        return 1;
    }
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
    double ops = (double)n * n * n / 3.0;
    double gflops = ops / elapsed.count() / 1e9;
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
