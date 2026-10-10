#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// Right-looking blocked Cholesky, with one persistent OpenMP team.
// Only the lower triangle is used; each trailing tile has a single writer.
bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    constexpr size_t block = 64;
    const size_t stride = (n + block - 1) / block * block;
    // Pack the transposed panel in block-by-block tiles. This makes SIMD
    // loads contiguous and avoids cache conflicts and long strides between k.
    std::vector<double> panel(block * stride);
    bool success = true;

    #pragma omp parallel shared(success, A, panel)
    {
        for (size_t first = 0; first < n; first += block) {
            const size_t end = std::min(first + block, n);
            #pragma omp single
            {
                for (size_t i = first; i < end && success; ++i) {
                    for (size_t j = first; j <= i; ++j) {
                        double sum = 0.0;
                        for (size_t k = first; k < j; ++k)
                            sum += A[i * n + k] * A[j * n + k];
                        const double val = A[i * n + j] - sum;
                        if (i == j) {
                            if (val <= 0.0) {
                                printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
                                success = false;
                                break;
                            }
                            A[i * n + j] = sqrt(val);
                        } else {
                            A[i * n + j] = val / A[j * n + j];
                        }
                    }
                }
            } // Barrier publishes the diagonal block and failure status.
            if (!success) break;

            #pragma omp for schedule(static)
            for (size_t i = end; i < n; ++i) {
                for (size_t j = first; j < end; ++j) {
                    double sum = 0.0;
                    for (size_t k = first; k < j; ++k)
                        sum += A[i * n + k] * A[j * n + k];
                    A[i * n + j] = (A[i * n + j] - sum) / A[j * n + j];
                    panel[(i / block) * block * block + (j - first) * block + i % block] = A[i * n + j];
                }
            } // All panel rows must be ready before updating trailing tiles.

            const size_t tiles = (n - end + block - 1) / block;
            const size_t tileCount = tiles * (tiles + 1) / 2;
            #pragma omp for schedule(dynamic, 1)
            for (size_t tile = 0; tile < tileCount; ++tile) {
                // Flatten the lower triangular tile grid for load balancing.
                size_t ti = static_cast<size_t>((sqrt(8.0 * tile + 1.0) - 1.0) / 2.0);
                while ((ti + 1) * (ti + 2) / 2 <= tile) ++ti;
                while (ti * (ti + 1) / 2 > tile) --ti;
                const size_t tj = tile - ti * (ti + 1) / 2;
                const size_t rowBegin = end + ti * block;
                const size_t rowEnd = std::min(rowBegin + block, n);
                const size_t colBegin = end + tj * block;
                const size_t colEnd = std::min(colBegin + block, n);
                size_t i = rowBegin;
                // Four rows share the packed panel loads. Each tile stays in
                // cache throughout the rank-block update.
                for (; i + 3 < rowEnd; i += 4) {
                    double* r0 = A.data() + i * n;
                    double* r1 = r0 + n;
                    double* r2 = r1 + n;
                    double* r3 = r2 + n;
                    size_t j = colBegin;
                    // Keep a 4-by-8 microtile in SIMD registers over the entire
                    // panel, loading and storing the matrix only once.
                    for (; j + 7 < colEnd; j += 8) {
                        double s0[8], s1[8], s2[8], s3[8];
                        #pragma omp simd
                        for (size_t lane = 0; lane < 8; ++lane) {
                            s0[lane] = r0[j + lane];
                            s1[lane] = r1[j + lane];
                            s2[lane] = r2[j + lane];
                            s3[lane] = r3[j + lane];
                        }
                        for (size_t k = first; k < end; ++k) {
                            const double l0 = r0[k], l1 = r1[k];
                            const double l2 = r2[k], l3 = r3[k];
                            const double* packed = panel.data() + (colBegin / block) * block * block
                                                 + (k - first) * block + j - colBegin;
                            #pragma omp simd
                            for (size_t lane = 0; lane < 8; ++lane) {
                                const double v = packed[lane];
                                s0[lane] -= l0 * v;
                                s1[lane] -= l1 * v;
                                s2[lane] -= l2 * v;
                                s3[lane] -= l3 * v;
                            }
                        }
                        #pragma omp simd
                        for (size_t lane = 0; lane < 8; ++lane) {
                            r0[j + lane] = s0[lane];
                            r1[j + lane] = s1[lane];
                            r2[j + lane] = s2[lane];
                            r3[j + lane] = s3[lane];
                        }
                    }
                    for (size_t k = first; k < end; ++k) {
                        const double l0 = r0[k], l1 = r1[k];
                        const double l2 = r2[k], l3 = r3[k];
                        const double* packed = panel.data() + (colBegin / block) * block * block
                                             + (k - first) * block;
                        #pragma omp simd
                        for (size_t tail = j; tail < colEnd; ++tail) {
                            const double v = packed[tail - colBegin];
                            r0[tail] -= l0 * v;
                            r1[tail] -= l1 * v;
                            r2[tail] -= l2 * v;
                            r3[tail] -= l3 * v;
                        }
                    }
                }
                for (; i < rowEnd; ++i) {
                    double* row = A.data() + i * n;
                    for (size_t k = first; k < end; ++k) {
                        const double l = row[k];
                        const double* packed = panel.data() + (colBegin / block) * block * block
                                             + (k - first) * block;
                        #pragma omp simd
                        for (size_t j = colBegin; j < colEnd; ++j)
                            row[j] -= l * packed[j - colBegin];
                    }
                }
            } // Finish all updates before factoring the next diagonal block.
        }
        if (success) {
            #pragma omp for schedule(static)
            for (size_t i = 0; i < n; ++i)
                std::fill(A.begin() + i * n + i + 1, A.begin() + (i + 1) * n, 0.0);
        }
    }
    return success;
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
    
    // Compute each symmetric pair once, retaining the original summation order.
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
    // Validate by computing L * L^T and comparing with original matrix
    
    double maxError = 0.0;
    double relError = 0.0;
    // The upper triangle of L is zero, so only the common row prefix
    // contributes. Compare both halves of the original matrix.
    #pragma omp parallel for schedule(dynamic, 1) reduction(max:maxError,relError)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k <= j; ++k)
                sum += L[i * n + k] * L[j * n + k];
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
