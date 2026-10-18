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
constexpr size_t blockSize = 64;

// Factor a diagonal tile whose contributions from earlier tiles are complete.
bool factorDiagonal(double* A, size_t n, size_t p) {
    const size_t end = std::min(p + blockSize, n);
    for (size_t j = p; j < end; ++j) {
        double sum = 0.0;
        for (size_t k = p; k < j; ++k)
            sum += A[j * n + k] * A[j * n + k];
        const double val = A[j * n + j] - sum;
        if (val <= 0.0) {
            printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
            return false;
        }
        A[j * n + j] = sqrt(val);
        for (size_t i = j + 1; i < end; ++i) {
            sum = 0.0;
            for (size_t k = p; k < j; ++k)
                sum += A[i * n + k] * A[j * n + k];
            A[i * n + j] = (A[i * n + j] - sum) / A[j * n + j];
        }
    }
    return true;
}

void solvePanel(double* A, size_t n, size_t row, size_t p) {
    const size_t rowEnd = std::min(row + blockSize, n);
    const size_t end = std::min(p + blockSize, n);
    for (size_t i = row; i < rowEnd; ++i) {
        for (size_t j = p; j < end; ++j) {
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (size_t k = p; k < j; ++k)
                sum += A[i * n + k] * A[j * n + k];
            A[i * n + j] = (A[i * n + j] - sum) / A[j * n + j];
        }
    }
}

// Pack the right-hand tile transposed so the update has contiguous SIMD
// accesses. Each output tile has one owner and stays in cache across k.
void updateTile(double* A, size_t n, size_t row, size_t col, size_t p) {
    alignas(64) double right[blockSize * blockSize];
    const size_t rows = std::min(blockSize, n - row);
    const size_t cols = std::min(blockSize, n - col);
    const size_t width = std::min(blockSize, n - p);
    for (size_t k = 0; k < width; ++k)
        for (size_t j = 0; j < cols; ++j)
            right[k * blockSize + j] = A[(col + j) * n + p + k];

    for (size_t i = 0; i < rows; ++i) {
        double* out = A + (row + i) * n + col;
        for (size_t k = 0; k < width; ++k) {
            const double left = A[(row + i) * n + p + k];
            const double* rhs = right + k * blockSize;
            #pragma omp simd
            for (size_t j = 0; j < cols; ++j)
                out[j] -= left * rhs[j];
        }
    }
}
} // namespace

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    double* data = A.data();
    std::atomic<bool> success{true};
    #pragma omp parallel
    {
        #pragma omp single
        {
            // The first element of each tile is its dependency token. All tasks
            // are siblings; inout orders successive updates to the same tile.
            // Dependencies allow the next panel to start before distant updates.
            for (size_t p = 0; p < n; p += blockSize) {
                #pragma omp task firstprivate(p) depend(inout: data[p * n + p]) shared(success)
                {
                    if (success.load(std::memory_order_relaxed) &&
                        !factorDiagonal(data, n, p))
                        success.store(false, std::memory_order_relaxed);
                }
                for (size_t row = p + blockSize; row < n; row += blockSize) {
                    #pragma omp task firstprivate(row, p) depend(in: data[p * n + p]) depend(inout: data[row * n + p]) shared(success)
                    {
                        if (success.load(std::memory_order_relaxed))
                            solvePanel(data, n, row, p);
                    }
                }
                for (size_t row = p + blockSize; row < n; row += blockSize) {
                    for (size_t col = p + blockSize; col <= row; col += blockSize) {
                        #pragma omp task firstprivate(row, col, p) depend(in: data[row * n + p], data[col * n + p]) depend(inout: data[row * n + col]) shared(success)
                        {
                            if (success.load(std::memory_order_relaxed))
                                updateTile(data, n, row, col, p);
                        }
                    }
                }
            }
        }

        // The single's barrier waits for all factorization tasks first.
        #pragma omp for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            if (success.load(std::memory_order_relaxed))
                std::fill(data + i * n + i + 1, data + (i + 1) * n, 0.0);
        }
    }
    return success.load(std::memory_order_relaxed);
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
    
    // Compute one triangle of B * B^T and mirror it. Each pair has one
    // owner; keeping rand_r above serial preserves the original input sequence.
    #pragma omp parallel for schedule(dynamic, 8)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
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
    // Validate by computing L * L^T and comparing with original matrix
    
    double maxError = 0.0;
    double relError = 0.0;

    // Reconstruct and compare in place, exploiting triangularity and symmetry.
    #pragma omp parallel for schedule(dynamic, 8) reduction(max:maxError,relError)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (size_t k = 0; k <= j; ++k) {
                sum += L[i * n + k] * L[j * n + k];
            }
            const double error = fabs(sum - A_orig[i * n + j]);
            const double mirrorError = fabs(sum - A_orig[j * n + i]);
            maxError = std::max(maxError, std::max(error, mirrorError));
            relError = std::max(relError, error / (fabs(A_orig[i * n + j]) + 1e-10));
            relError = std::max(relError, mirrorError / (fabs(A_orig[j * n + i]) + 1e-10));
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
