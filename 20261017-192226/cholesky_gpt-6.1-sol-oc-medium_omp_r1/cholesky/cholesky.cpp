#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// Blocked, right-looking Cholesky. Each trailing tile has a single writer;
// barriers separate diagonal factorization, panel solves, and tile updates.

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    constexpr size_t block = 64;
    const size_t tiles = (n + block - 1) / block;
    // Transposing just the current panel makes the update's inner loop unit
    // stride for both operands, without a second full-sized matrix.
    std::vector<double> panel(block * n);
    bool success = true;
    #pragma omp parallel shared(success, A, panel)
    {
        for (size_t p = 0; p < n; p += block) {
            const size_t end = std::min(p + block, n);
            #pragma omp single
            {
                for (size_t i = p; i < end && success; ++i) {
                    for (size_t j = p; j <= i; ++j) {
                        double sum = 0.0;
                        for (size_t k = p; k < j; ++k)
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
            }
            // The single's barrier publishes both the diagonal and failure flag.
            if (!success) break;

            #pragma omp for schedule(static)
            for (size_t i = end; i < n; ++i) {
                for (size_t j = p; j < end; ++j) {
                    double sum = 0.0;
                    for (size_t k = p; k < j; ++k)
                        sum += A[i * n + k] * A[j * n + k];
                    const double value = (A[i * n + j] - sum) / A[j * n + j];
                    A[i * n + j] = value;
                    panel[(j - p) * n + i] = value;
                }
            }

            // Rectangular iteration space permits collapse; only lower tiles
            // are updated. Dynamic scheduling balances the triangular boundary.
            #pragma omp for collapse(2) schedule(dynamic, 1)
            for (size_t ti = p / block + 1; ti < tiles; ++ti) {
                for (size_t tj = p / block + 1; tj < tiles; ++tj) {
                    if (tj > ti) continue;
                    const size_t ib = ti * block;
                    const size_t jb = tj * block;
                    const size_t ie = std::min(ib + block, n);
                    const size_t je = std::min(jb + block, n);
                    size_t firstRow = ib;
                    if (tj < ti) {
                        // A 4 x 8 register microtile reuses each packed panel
                        // vector across four rows and stores only after the
                        // complete rank-block update.
                        for (; firstRow + 4 <= ie; firstRow += 4) {
                            double* r0 = A.data() + firstRow * n;
                            double* r1 = r0 + n;
                            double* r2 = r1 + n;
                            double* r3 = r2 + n;
                            size_t j = jb;
                            for (; j + 8 <= je; j += 8) {
                                double c0[8], c1[8], c2[8], c3[8];
                                #pragma omp simd
                                for (size_t x = 0; x < 8; ++x) {
                                    c0[x] = r0[j + x];
                                    c1[x] = r1[j + x];
                                    c2[x] = r2[j + x];
                                    c3[x] = r3[j + x];
                                }
                                for (size_t k = p; k < end; ++k) {
                                    const double* packed = panel.data() + (k - p) * n + j;
                                    const double a0 = r0[k], a1 = r1[k];
                                    const double a2 = r2[k], a3 = r3[k];
                                    #pragma omp simd
                                    for (size_t x = 0; x < 8; ++x) {
                                        c0[x] -= a0 * packed[x];
                                        c1[x] -= a1 * packed[x];
                                        c2[x] -= a2 * packed[x];
                                        c3[x] -= a3 * packed[x];
                                    }
                                }
                                #pragma omp simd
                                for (size_t x = 0; x < 8; ++x) {
                                    r0[j + x] = c0[x];
                                    r1[j + x] = c1[x];
                                    r2[j + x] = c2[x];
                                    r3[j + x] = c3[x];
                                }
                            }
                            for (size_t k = p; k < end; ++k) {
                                const double* packed = panel.data() + (k - p) * n;
                                #pragma omp simd
                                for (size_t x = j; x < je; ++x) {
                                    r0[x] -= r0[k] * packed[x];
                                    r1[x] -= r1[k] * packed[x];
                                    r2[x] -= r2[k] * packed[x];
                                    r3[x] -= r3[k] * packed[x];
                                }
                            }
                        }
                    }
                    // Diagonal tiles and incomplete microtile rows.
                    for (size_t i = firstRow; i < ie; ++i) {
                        const size_t stop = std::min(je, i + 1);
                        double* row = A.data() + i * n;
                        for (size_t k = p; k < end; ++k) {
                            const double lik = row[k];
                            const double* packed = panel.data() + (k - p) * n;
                            #pragma omp simd
                            for (size_t j = jb; j < stop; ++j)
                                row[j] -= lik * packed[j];
                        }
                    }
                }
            }
        }
        #pragma omp for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i + 1; j < n; ++j)
                A[i * n + j] = 0.0;
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
    
    // Compute one triangle of B * B^T and mirror it. Keep the random-number
    // stream above sequential so inputs are independent of thread count.
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
    // Compute and compare one triangle directly, avoiding a temporary matrix.
    #pragma omp parallel for schedule(dynamic, 1) reduction(max:maxError,relError)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (size_t k = 0; k <= j; ++k) {
                sum += L[i * n + k] * L[j * n + k];
            }
            const double error = fabs(sum - A_orig[i * n + j]);
            maxError = std::max(maxError, error);
            relError = std::max(relError, error / (fabs(A_orig[i * n + j]) + 1e-10));
            const double mirrorError = fabs(sum - A_orig[j * n + i]);
            maxError = std::max(maxError, mirrorError);
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
