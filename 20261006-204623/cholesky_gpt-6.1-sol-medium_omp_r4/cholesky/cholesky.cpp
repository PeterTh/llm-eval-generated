#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

namespace {
constexpr size_t tileSize = 64;

// Factor a diagonal tile after all preceding updates have completed.
bool factorDiagonal(double* A, size_t n, size_t begin, size_t end,
                    size_t& failedDiagonal) {
    for (size_t i = begin; i < end; ++i) {
        for (size_t j = begin; j <= i; ++j) {
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (size_t k = begin; k < j; ++k) {
                sum += A[i * n + k] * A[j * n + k];
            }
            const double value = A[i * n + j] - sum;
            if (i == j) {
                if (value <= 0.0) {
                    failedDiagonal = j;
                    return false;
                }
                A[i * n + j] = std::sqrt(value);
            } else {
                A[i * n + j] = value / A[j * n + j];
            }
        }
    }
    return true;
}

// Solve L_ik * L_kk^T = A_ik, independently for each tile below the diagonal.
void solvePanel(double* A, size_t n, size_t row, size_t rowEnd,
                size_t column, size_t columnEnd) {
    for (size_t i = row; i < rowEnd; ++i) {
        for (size_t j = column; j < columnEnd; ++j) {
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (size_t k = column; k < j; ++k) {
                sum += A[i * n + k] * A[j * n + k];
            }
            A[i * n + j] = (A[i * n + j] - sum) / A[j * n + j];
        }
    }
}

// A_ij -= L_ik * L_jk^T. Pack the right panel so the SIMD loop has
// contiguous loads instead of strided accesses through the row-major matrix.
void updateTile(double* A, size_t n, size_t row, size_t rowEnd,
                size_t column, size_t columnEnd, size_t panel, size_t panelEnd) {
    alignas(64) double right[tileSize * tileSize];
    const size_t width = columnEnd - column;
    for (size_t k = panel; k < panelEnd; ++k) {
        #pragma omp simd
        for (size_t j = 0; j < width; ++j) {
            right[(k - panel) * tileSize + j] = A[(column + j) * n + k];
        }
    }
    // Accumulate four output rows in registers across the panel reduction.
    // SIMD runs across columns, sharing each packed-panel load between rows.
    size_t i = row;
    for (; i + 3 < rowEnd; i += 4) {
        size_t columnOffset = 0;
        for (; columnOffset + 3 < width; columnOffset += 4) {
            double c0[4], c1[4], c2[4], c3[4];
            #pragma omp simd
            for (size_t lane = 0; lane < 4; ++lane) {
                const size_t j = column + columnOffset + lane;
                c0[lane] = A[i * n + j];
                c1[lane] = A[(i + 1) * n + j];
                c2[lane] = A[(i + 2) * n + j];
                c3[lane] = A[(i + 3) * n + j];
            }
            for (size_t k = panel; k < panelEnd; ++k) {
                const double a0 = A[i * n + k];
                const double a1 = A[(i + 1) * n + k];
                const double a2 = A[(i + 2) * n + k];
                const double a3 = A[(i + 3) * n + k];
                const double* packed = right + (k - panel) * tileSize + columnOffset;
                #pragma omp simd
                for (size_t lane = 0; lane < 4; ++lane) {
                    c0[lane] -= a0 * packed[lane];
                    c1[lane] -= a1 * packed[lane];
                    c2[lane] -= a2 * packed[lane];
                    c3[lane] -= a3 * packed[lane];
                }
            }
            #pragma omp simd
            for (size_t lane = 0; lane < 4; ++lane) {
                const size_t j = column + columnOffset + lane;
                A[i * n + j] = c0[lane];
                A[(i + 1) * n + j] = c1[lane];
                A[(i + 2) * n + j] = c2[lane];
                A[(i + 3) * n + j] = c3[lane];
            }
        }
        for (size_t j = columnOffset; j < width; ++j) {
            double c0 = A[i * n + column + j];
            double c1 = A[(i + 1) * n + column + j];
            double c2 = A[(i + 2) * n + column + j];
            double c3 = A[(i + 3) * n + column + j];
            for (size_t k = panel; k < panelEnd; ++k) {
                const double value = right[(k - panel) * tileSize + j];
                c0 -= A[i * n + k] * value;
                c1 -= A[(i + 1) * n + k] * value;
                c2 -= A[(i + 2) * n + k] * value;
                c3 -= A[(i + 3) * n + k] * value;
            }
            A[i * n + column + j] = c0;
            A[(i + 1) * n + column + j] = c1;
            A[(i + 2) * n + column + j] = c2;
            A[(i + 3) * n + column + j] = c3;
        }
    }
    for (; i < rowEnd; ++i) {
        #pragma omp simd
        for (size_t j = 0; j < width; ++j) {
            double value = A[i * n + column + j];
            for (size_t k = panel; k < panelEnd; ++k) {
                value -= A[i * n + k] * right[(k - panel) * tileSize + j];
            }
            A[i * n + column + j] = value;
        }
    }
}
} // namespace

// In-place, tiled Cholesky decomposition of a row-major matrix. Tile corner
// addresses serve as dependency tokens: each output tile has one writer at a
// time, and panel reads wait for the corresponding factorization/solve.
bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    double* data = A.data();
    std::atomic<bool> failed{false};
    size_t failedDiagonal = 0;
    #pragma omp parallel
    {
        #pragma omp single
        {
            for (size_t k = 0; k < n; k += tileSize) {
                const size_t kEnd = std::min(k + tileSize, n);
                const size_t diagonal = k * n + k;
                #pragma omp task depend(inout: data[diagonal]) firstprivate(k, kEnd) shared(failed, failedDiagonal)
                {
                    if (!failed.load(std::memory_order_relaxed) &&
                        !factorDiagonal(data, n, k, kEnd, failedDiagonal)) {
                        failed.store(true, std::memory_order_relaxed);
                    }
                }
                for (size_t i = kEnd; i < n; i += tileSize) {
                    const size_t iEnd = std::min(i + tileSize, n);
                    const size_t panel = i * n + k;
                    #pragma omp task depend(in: data[diagonal]) depend(inout: data[panel]) firstprivate(i, iEnd, k, kEnd) shared(failed)
                    {
                        if (!failed.load(std::memory_order_relaxed)) {
                            solvePanel(data, n, i, iEnd, k, kEnd);
                        }
                    }
                }
                for (size_t i = kEnd; i < n; i += tileSize) {
                    const size_t iEnd = std::min(i + tileSize, n);
                    for (size_t j = kEnd; j <= i; j += tileSize) {
                        const size_t jEnd = std::min(j + tileSize, n);
                        const size_t left = i * n + k;
                        const size_t right = j * n + k;
                        const size_t output = i * n + j;
                        #pragma omp task depend(in: data[left], data[right]) depend(inout: data[output]) firstprivate(i, iEnd, j, jEnd, k, kEnd) shared(failed)
                        {
                            if (!failed.load(std::memory_order_relaxed)) {
                                updateTile(data, n, i, iEnd, j, jEnd, k, kEnd);
                            }
                        }
                    }
                }
            }
            #pragma omp taskwait
        }
        #pragma omp for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            std::fill(data + i * n + i + 1, data + (i + 1) * n, 0.0);
        }
    }
    if (failed.load(std::memory_order_relaxed)) {
        printf("Error: Matrix is not positive definite at diagonal element %zu\n", failedDiagonal);
        return false;
    }
    return true;
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
    
    // Compute each symmetric pair once. Keep the original summation order
    // and random stream so the generated input is independent of thread count.
    #pragma omp parallel for schedule(dynamic, 1)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += B[i * n + k] * B[j * n + k];
            }
            A[i * n + j] = sum;
            A[j * n + i] = sum;
        }
    }

    // Add diagonal dominance to ensure positive definiteness
    for (size_t i = 0; i < n; ++i) {
        A[i * n + i] += n;
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Only the lower triangle is needed: both products are symmetric, and
    // entries beyond min(i,j) in the zeroed upper triangle contribute zero.
    double maxError = 0.0;
    double relError = 0.0;
    #pragma omp parallel for schedule(dynamic, 1) reduction(max:maxError,relError)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (size_t k = 0; k <= j; ++k) {
                sum += L[i * n + k] * L[j * n + k];
            }
            // Check both input triangles, including nonsymmetric inputs.
            const double lowerError = std::fabs(sum - A_orig[i * n + j]);
            const double upperError = std::fabs(sum - A_orig[j * n + i]);
            maxError = std::max(maxError, std::max(lowerError, upperError));
            relError = std::max(relError, lowerError / (std::fabs(A_orig[i * n + j]) + 1e-10));
            relError = std::max(relError, upperError / (std::fabs(A_orig[j * n + i]) + 1e-10));
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
    const double seconds = std::chrono::duration<double>(end - start).count();
    double gflops = seconds > 0.0 ? ops / seconds / 1e9 : 0.0;
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
