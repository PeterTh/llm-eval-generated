#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// Blocked right-looking Cholesky, with one persistent OpenMP team.
// Only the lower triangle is used; the upper triangle is cleared on success.
bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    constexpr size_t block = 64;
    // Transpose the current panel so the trailing update has contiguous SIMD
    // accesses to both its destination and its shared panel operand.
    std::vector<double> panel(((n + block - 1) / block) * block * block);
    bool success = true;

    #pragma omp parallel shared(success, A, panel)
    {
        for (size_t base = 0; base < n; base += block) {
            const size_t end = std::min(base + block, n);
            #pragma omp single
            {
                for (size_t i = base; i < end && success; ++i) {
                    for (size_t j = base; j <= i; ++j) {
                        double sum = 0.0;
                        for (size_t k = base; k < j; ++k)
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
            // The single-region barrier publishes the factor and failure flag.
            if (!success) break;

            #pragma omp for schedule(static)
            for (size_t i = end; i < n; ++i) {
                for (size_t j = base; j < end; ++j) {
                    double sum = 0.0;
                    #pragma omp simd reduction(+:sum)
                    for (size_t k = base; k < j; ++k)
                        sum += A[i * n + k] * A[j * n + k];
                    const double value = (A[i * n + j] - sum) / A[j * n + j];
                    A[i * n + j] = value;
                    panel[((i - end) / block) * block * block
                          + (j - base) * block + (i - end) % block] = value;
                }
            }

            // Each iteration owns one tile. Include the unused upper half of
            // diagonal tiles to keep the innermost loop contiguous and regular.
            const size_t tiles = (n - end + block - 1) / block;
            #pragma omp for collapse(2) schedule(dynamic, 1)
            for (size_t ti = 0; ti < tiles; ++ti) {
                for (size_t tj = 0; tj < tiles; ++tj) {
                    if (tj > ti) continue;
                    const size_t ib = end + ti * block;
                    const size_t jb = end + tj * block;
                    const size_t ie = std::min(ib + block, n);
                    const size_t je = std::min(jb + block, n);
                    size_t i = ib;
                    // A 4-by-8 register tile reuses each packed vector across
                    // four rows and keeps partial sums out of matrix memory.
                    for (; i + 4 <= ie; i += 4) {
                        size_t j = jb;
                        for (; j + 8 <= je; j += 8) {
                            double c0[8], c1[8], c2[8], c3[8];
                            #pragma omp simd
                            for (size_t x = 0; x < 8; ++x) {
                                c0[x] = A[i * n + j + x];
                                c1[x] = A[(i + 1) * n + j + x];
                                c2[x] = A[(i + 2) * n + j + x];
                                c3[x] = A[(i + 3) * n + j + x];
                            }
                            for (size_t k = base; k < end; ++k) {
                                const double* packed = panel.data() + tj * block * block
                                                     + (k - base) * block + j - jb;
                                const double v0 = A[i * n + k];
                                const double v1 = A[(i + 1) * n + k];
                                const double v2 = A[(i + 2) * n + k];
                                const double v3 = A[(i + 3) * n + k];
                                #pragma omp simd
                                for (size_t x = 0; x < 8; ++x) {
                                    c0[x] -= v0 * packed[x];
                                    c1[x] -= v1 * packed[x];
                                    c2[x] -= v2 * packed[x];
                                    c3[x] -= v3 * packed[x];
                                }
                            }
                            #pragma omp simd
                            for (size_t x = 0; x < 8; ++x) {
                                A[i * n + j + x] = c0[x];
                                A[(i + 1) * n + j + x] = c1[x];
                                A[(i + 2) * n + j + x] = c2[x];
                                A[(i + 3) * n + j + x] = c3[x];
                            }
                        }
                        // Partial column tile at the matrix boundary.
                        for (size_t r = i; r < i + 4; ++r)
                            for (size_t k = base; k < end; ++k) {
                                const double value = A[r * n + k];
                                const double* packed = panel.data() + tj * block * block
                                                     + (k - base) * block;
                                #pragma omp simd
                                for (size_t x = j; x < je; ++x)
                                    A[r * n + x] -= value * packed[x - jb];
                            }
                    }
                    // Partial row tile at the matrix boundary.
                    for (; i < ie; ++i) {
                        double* row = A.data() + i * n;
                        for (size_t k = base; k < end; ++k) {
                            const double value = row[k];
                            const double* packed = panel.data() + tj * block * block
                                                 + (k - base) * block;
                            #pragma omp simd
                            for (size_t j = jb; j < je; ++j)
                                row[j] -= value * packed[j - jb];
                        }
                    }
                }
            }
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
    
    // Compute each symmetric pair once. Keep the random stream above serial.
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
    // Reconstruct lower-triangular dot products directly, without an O(n^2)
    // temporary. Check both original entries of each symmetric pair.
    double maxError = 0.0;
    double relError = 0.0;
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
