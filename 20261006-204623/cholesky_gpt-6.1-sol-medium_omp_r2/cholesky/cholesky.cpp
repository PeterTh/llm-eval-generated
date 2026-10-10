#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// Right-looking blocked Cholesky. Only the lower triangle is used during
// factorization; the upper triangle is cleared once all blocks are complete.
bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    constexpr size_t blockSize = 64;
    // Transpose each solved panel so the trailing update has contiguous SIMD
    // loads instead of strided accesses to columns of the row-major matrix.
    std::vector<double> panel(blockSize * n);
    bool success = true;

    #pragma omp parallel shared(success, A, panel)
    {
        for (size_t base = 0; base < n; base += blockSize) {
            const size_t end = std::min(base + blockSize, n);
            const size_t width = end - base;
            #pragma omp single
            {
                for (size_t j = base; j < end; ++j) {
                    double sum = 0.0;
                    for (size_t k = base; k < j; ++k)
                        sum += A[j * n + k] * A[j * n + k];
                    const double val = A[j * n + j] - sum;
                    if (val <= 0.0) {
                        printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
                        success = false;
                        break;
                    }
                    A[j * n + j] = sqrt(val);
                    for (size_t i = j + 1; i < end; ++i) {
                        double dot = 0.0;
                        for (size_t k = base; k < j; ++k)
                            dot += A[i * n + k] * A[j * n + k];
                        A[i * n + j] = (A[i * n + j] - dot) / A[j * n + j];
                    }
                }
            }
            // The single-region barrier publishes both the diagonal and the
            // failure flag. Every thread takes the same path on failure.
            if (!success)
                break;

            #pragma omp for schedule(static)
            for (size_t i = end; i < n; ++i) {
                for (size_t j = base; j < end; ++j) {
                    double sum = 0.0;
                    for (size_t k = base; k < j; ++k)
                        sum += A[i * n + k] * A[j * n + k];
                    const double value = (A[i * n + j] - sum) / A[j * n + j];
                    A[i * n + j] = value;
                    panel[(j - base) * n + i] = value;
                }
            }

            const size_t blocks = (n - end + blockSize - 1) / blockSize;
            const size_t tiles = blocks * (blocks + 1) / 2;
            #pragma omp for schedule(dynamic, 1)
            for (size_t tile = 0; tile < tiles; ++tile) {
                // Enumerate only lower-triangular tiles, balancing the work
                // across threads even as the trailing matrix gets smaller.
                size_t bi = static_cast<size_t>((sqrt(8.0 * tile + 1.0) - 1.0) / 2.0);
                while ((bi + 1) * (bi + 2) / 2 <= tile) ++bi;
                while (bi * (bi + 1) / 2 > tile) --bi;
                const size_t bj = tile - bi * (bi + 1) / 2;
                const size_t rowBegin = end + bi * blockSize;
                const size_t rowEnd = std::min(rowBegin + blockSize, n);
                const size_t colBegin = end + bj * blockSize;
                const size_t columns = std::min(blockSize, n - colBegin);
                for (size_t i = rowBegin; i < rowEnd; i += 4) {
                    const size_t rows = std::min(size_t(4), rowEnd - i);
                    alignas(64) double sums[4][blockSize] = {};
                    for (size_t k = 0; k < width; ++k) {
                        const double* right = panel.data() + k * n + colBegin;
                        for (size_t r = 0; r < rows; ++r) {
                            const double left = A[(i + r) * n + base + k];
                            #pragma omp simd
                            for (size_t j = 0; j < columns; ++j)
                                sums[r][j] += left * right[j];
                        }
                    }
                    for (size_t r = 0; r < rows; ++r) {
                        double* out = A.data() + (i + r) * n + colBegin;
                        #pragma omp simd
                        for (size_t j = 0; j < columns; ++j)
                            out[j] -= sums[r][j];
                    }
                }
            }
            // The worksharing barrier completes all updates before the next
            // diagonal block reads its entries.
        }
        if (success) {
            #pragma omp for schedule(static)
            for (size_t i = 0; i < n; ++i)
                std::fill(A.begin() + i * n + i + 1, A.begin() + (i + 1) * n, 0.0);
        }
    }
    return success;
}

// Generate the same seeded random matrix, then form A = B * B^T + n * I.
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    // rand_r has a sequential state: keep this loop ordered for reproducibility.
    for (size_t i = 0; i < n * n; ++i)
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;

    #pragma omp parallel for schedule(dynamic, 1)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (size_t k = 0; k < n; ++k)
                sum += B[i * n + k] * B[j * n + k];
            A[i * n + j] = sum;
            A[j * n + i] = sum;
        }
        A[i * n + i] += n;
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    double maxError = 0.0;
    double relError = 0.0;
    // L is triangular, so entries beyond min(i, j) contribute only zeros.
    // Compare directly, avoiding a second full-sized matrix allocation.
    #pragma omp parallel for schedule(dynamic, 1) reduction(max:maxError,relError)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (size_t k = 0; k <= j; ++k)
                sum += L[i * n + k] * L[j * n + k];
            const double error = fabs(sum - A_orig[i * n + j]);
            maxError = std::max(maxError, error);
            relError = std::max(relError, error / (fabs(A_orig[i * n + j]) + 1e-10));
            // Also check the upper half if the caller supplies a nonsymmetric A.
            const double upperError = fabs(sum - A_orig[j * n + i]);
            maxError = std::max(maxError, upperError);
            relError = std::max(relError, upperError / (fabs(A_orig[j * n + i]) + 1e-10));
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
