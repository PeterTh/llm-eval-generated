#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// Right-looking blocked Cholesky. Each trailing tile has one writer, and
// implicit worksharing barriers separate the dependent stages of each panel.

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    constexpr size_t blockSize = 64;
    bool success = true;
    double* const a = A.data();

    #pragma omp parallel shared(success)
    {
        for (size_t panel = 0; panel < n; panel += blockSize) {
            const size_t end = std::min(panel + blockSize, n);
            #pragma omp single
            {
                for (size_t j = panel; j < end; ++j) {
                    double sum = 0.0;
                    #pragma omp simd reduction(+:sum)
                    for (size_t k = panel; k < j; ++k)
                        sum += a[j * n + k] * a[j * n + k];
                    const double val = a[j * n + j] - sum;
                    if (val <= 0.0) {
                        printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
                        success = false;
                        break;
                    }
                    a[j * n + j] = sqrt(val);
                    for (size_t i = j + 1; i < end; ++i) {
                        double dot = 0.0;
                        #pragma omp simd reduction(+:dot)
                        for (size_t k = panel; k < j; ++k)
                            dot += a[i * n + k] * a[j * n + k];
                        a[i * n + j] = (a[i * n + j] - dot) / a[j * n + j];
                    }
                }
            }
            // All threads observe the same status after the single barrier.
            if (!success) break;

            #pragma omp for schedule(static)
            for (size_t i = end; i < n; ++i) {
                for (size_t j = panel; j < end; ++j) {
                    double sum = 0.0;
                    #pragma omp simd reduction(+:sum)
                    for (size_t k = panel; k < j; ++k)
                        sum += a[i * n + k] * a[j * n + k];
                    a[i * n + j] = (a[i * n + j] - sum) / a[j * n + j];
                    // Use the unused upper triangle as a transposed panel
                    // workspace for contiguous SIMD loads in the update.
                    a[j * n + i] = a[i * n + j];
                }
            }

            // Rectangular iteration space exposes individual tiles to OpenMP.
            // Four rows reuse each vector of the transposed panel operand.
            #pragma omp for collapse(2) schedule(dynamic, 1)
            for (size_t ib = end; ib < n; ib += blockSize) {
                for (size_t jb = end; jb < n; jb += blockSize) {
                    if (jb > ib) continue;
                    const size_t ie = std::min(ib + blockSize, n);
                    const size_t je = std::min(jb + blockSize, n);
                    size_t i = ib;
                    for (; i + 3 < ie; i += 4) {
                        double* const c0 = a + i * n;
                        double* const c1 = c0 + n;
                        double* const c2 = c1 + n;
                        double* const c3 = c2 + n;
                        size_t j = jb;
                        for (; j + 7 < je; j += 8) {
                            // Keep a 4 x 8 microtile in registers across the
                            // panel, rather than loading/storing it every k.
                            double s0[8], s1[8], s2[8], s3[8];
                            #pragma omp simd
                            for (size_t lane = 0; lane < 8; ++lane) {
                                s0[lane] = c0[j + lane];
                                s1[lane] = c1[j + lane];
                                s2[lane] = c2[j + lane];
                                s3[lane] = c3[j + lane];
                            }
                            for (size_t k = panel; k < end; ++k) {
                                const double x0 = c0[k], x1 = c1[k];
                                const double x2 = c2[k], x3 = c3[k];
                                const double* const b = a + k * n + j;
                                #pragma omp simd
                                for (size_t lane = 0; lane < 8; ++lane) {
                                    s0[lane] -= x0 * b[lane];
                                    s1[lane] -= x1 * b[lane];
                                    s2[lane] -= x2 * b[lane];
                                    s3[lane] -= x3 * b[lane];
                                }
                            }
                            #pragma omp simd
                            for (size_t lane = 0; lane < 8; ++lane) {
                                c0[j + lane] = s0[lane];
                                c1[j + lane] = s1[lane];
                                c2[j + lane] = s2[lane];
                                c3[j + lane] = s3[lane];
                            }
                        }
                        for (size_t k = panel; k < end; ++k) {
                            const double x0 = c0[k], x1 = c1[k];
                            const double x2 = c2[k], x3 = c3[k];
                            #pragma omp simd
                            for (size_t tail = j; tail < je; ++tail) {
                                const double b = a[k * n + tail];
                                c0[tail] -= x0 * b;
                                c1[tail] -= x1 * b;
                                c2[tail] -= x2 * b;
                                c3[tail] -= x3 * b;
                            }
                        }
                    }
                    for (; i < ie; ++i) {
                        for (size_t k = panel; k < end; ++k) {
                            const double x = a[i * n + k];
                            #pragma omp simd
                            for (size_t j = jb; j < je; ++j)
                                a[i * n + j] -= x * a[k * n + j];
                        }
                    }
                }
            }
        }

        #pragma omp for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i + 1; j < n; ++j)
                a[i * n + j] = 0.0;
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
    
    // Compute A = B * B^T
    #pragma omp parallel for schedule(dynamic, 1)
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

    // Only the common nonzero prefix of each pair of rows contributes.
    #pragma omp parallel for schedule(dynamic, 1) reduction(max:maxError,relError)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (size_t k = 0; k <= std::min(i, j); ++k) {
                sum += L[i * n + k] * L[j * n + k];
            }
            const double error = fabs(sum - A_orig[i * n + j]);
            maxError = std::max(maxError, error);
            const double rel = error / (fabs(A_orig[i * n + j]) + 1e-10);
            relError = std::max(relError, rel);
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
