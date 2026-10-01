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

// C -= X * Y^T. Packing Y makes the innermost update contiguous and
// vectorizable, with reuse of the packed panel and output rows in cache.
void updateTile(double* a, size_t n, size_t i, size_t j, size_t k) {
    const size_t rows = std::min(tileSize, n - i);
    const size_t cols = std::min(tileSize, n - j);
    const size_t depth = std::min(tileSize, n - k);
    alignas(64) double packed[tileSize][tileSize];
    for (size_t p = 0; p < depth; ++p)
        for (size_t c = 0; c < cols; ++c)
            packed[p][c] = a[(j + c) * n + k + p];
    for (size_t r = 0; r < rows; ++r) {
        double* out = a + (i + r) * n + j;
        // Only the lower triangle of diagonal tiles is used.
        const size_t width = i == j ? r + 1 : cols;
        for (size_t p = 0; p < depth; ++p) {
            const double x = a[(i + r) * n + k + p];
            #pragma omp simd
            for (size_t c = 0; c < width; ++c)
                out[c] -= x * packed[p][c];
        }
    }
}
} // namespace

// Right-looking tiled Cholesky. Each tile's first element serves as its
// dependency token. All tasks are siblings, and updates to a tile are ordered.
bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    double* a = A.data();
    std::atomic<bool> failed{false};
    #pragma omp parallel
    {
        #pragma omp single
        {
            for (size_t k = 0; k < n; k += tileSize) {
                const size_t end = std::min(k + tileSize, n);
                #pragma omp task firstprivate(k, end) depend(inout: a[k * n + k]) shared(failed)
                {
                    if (!failed.load(std::memory_order_relaxed)) {
                        for (size_t j = k; j < end; ++j) {
                            double sum = 0.0;
                            #pragma omp simd reduction(+:sum)
                            for (size_t p = k; p < j; ++p)
                                sum += a[j * n + p] * a[j * n + p];
                            const double val = a[j * n + j] - sum;
                            if (val <= 0.0) {
                                printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
                                failed.store(true, std::memory_order_relaxed);
                                break;
                            }
                            a[j * n + j] = sqrt(val);
                            for (size_t i = j + 1; i < end; ++i) {
                                double dot = 0.0;
                                #pragma omp simd reduction(+:dot)
                                for (size_t p = k; p < j; ++p)
                                    dot += a[i * n + p] * a[j * n + p];
                                a[i * n + j] = (a[i * n + j] - dot) / a[j * n + j];
                            }
                        }
                    }
                }
                for (size_t i = end; i < n; i += tileSize) {
                    #pragma omp task firstprivate(i, k, end) depend(in: a[k * n + k]) depend(inout: a[i * n + k]) shared(failed)
                    {
                        if (!failed.load(std::memory_order_relaxed)) {
                            const size_t rowEnd = std::min(i + tileSize, n);
                            for (size_t r = i; r < rowEnd; ++r) {
                                for (size_t j = k; j < end; ++j) {
                                    double sum = 0.0;
                                    #pragma omp simd reduction(+:sum)
                                    for (size_t p = k; p < j; ++p)
                                        sum += a[r * n + p] * a[j * n + p];
                                    a[r * n + j] = (a[r * n + j] - sum) / a[j * n + j];
                                }
                            }
                        }
                    }
                }
                // Submit near-diagonal updates first to expose the next panel.
                for (size_t j = end; j < n; j += tileSize) {
                    for (size_t i = j; i < n; i += tileSize) {
                        #pragma omp task firstprivate(i, j, k) depend(in: a[i * n + k], a[j * n + k]) depend(inout: a[i * n + j]) shared(failed)
                        {
                            if (!failed.load(std::memory_order_relaxed))
                                updateTile(a, n, i, j, k);
                        }
                    }
                }
            }
        }
        // The single region's barrier waits for all factorization tasks.
        #pragma omp for schedule(static)
        for (size_t i = 0; i < n; ++i)
            for (size_t j = i + 1; j < n; ++j)
                a[i * n + j] = 0.0;
    }
    return !failed.load(std::memory_order_relaxed);
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
    
    // Compute one triangle and mirror it; retain the seeded random stream.
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
    #pragma omp parallel for schedule(dynamic, 8) reduction(max:maxError, relError)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (size_t k = 0; k <= j; ++k)
                sum += L[i * n + k] * L[j * n + k];
            // Check both original entries, including for asymmetric input.
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
