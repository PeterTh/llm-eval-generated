#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// Blocked, right-looking Cholesky of a row-major matrix. Only the lower
// triangle is read; each trailing tile has a single OpenMP owner.
bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    constexpr size_t block = 64;
    const size_t blocks = (n + block - 1) / block;
    // Transpose the current panel so trailing updates vectorize across columns.
    // Padding prevents power-of-two matrix sizes from mapping every panel
    // column to the same cache sets.
    const size_t panelStride = n + 8;
    std::vector<double> panel(block * panelStride);
    double* const a = A.data();
    double* const packed = panel.data();
    bool success = true;

    // Keep one team alive throughout the factorization. Worksharing barriers
    // publish the diagonal block, solved panel, and updated trailing matrix.
    #pragma omp parallel shared(success)
    {
        for (size_t b = 0; b < blocks; ++b) {
            const size_t first = b * block;
            const size_t last = std::min(first + block, n);
            #pragma omp single
            {
                for (size_t j = first; j < last; ++j) {
                    double sum = 0.0;
                    #pragma omp simd reduction(+:sum)
                    for (size_t k = first; k < j; ++k)
                        sum += a[j * n + k] * a[j * n + k];
                    const double val = a[j * n + j] - sum;
                    if (val <= 0.0) {
                        printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
                        success = false;
                        break;
                    }
                    a[j * n + j] = sqrt(val);
                    for (size_t i = j + 1; i < last; ++i) {
                        double dot = 0.0;
                        #pragma omp simd reduction(+:dot)
                        for (size_t k = first; k < j; ++k)
                            dot += a[i * n + k] * a[j * n + k];
                        a[i * n + j] = (a[i * n + j] - dot) / a[j * n + j];
                    }
                }
            }
            if (!success) break;

            #pragma omp for schedule(static)
            for (size_t i = last; i < n; ++i) {
                for (size_t j = first; j < last; ++j) {
                    double sum = 0.0;
                    #pragma omp simd reduction(+:sum)
                    for (size_t k = first; k < j; ++k)
                        sum += a[i * n + k] * a[j * n + k];
                    a[i * n + j] = (a[i * n + j] - sum) / a[j * n + j];
                    packed[(j - first) * panelStride + i] = a[i * n + j];
                }
            }

            const size_t remaining = blocks - b - 1;
            const size_t tiles = remaining * (remaining + 1) / 2;
            #pragma omp for schedule(dynamic, 1)
            for (size_t tile = 0; tile < tiles; ++tile) {
                // Enumerate the triangular tile grid in a flat worksharing loop.
                size_t row = static_cast<size_t>((sqrt(8.0 * tile + 1.0) - 1.0) / 2.0);
                while ((row + 1) * (row + 2) / 2 <= tile) ++row;
                while (row * (row + 1) / 2 > tile) --row;
                const size_t col = tile - row * (row + 1) / 2;
                const size_t ib = last + row * block;
                const size_t jb = last + col * block;
                const size_t ie = std::min(ib + block, n);
                const size_t je = std::min(jb + block, n);
                size_t i = ib;
                // Keep a 4-by-8 microtile in registers across the entire panel
                // update, reusing each packed vector for four destination rows.
                if (row != col) {
                    for (; i + 4 <= ie; i += 4) {
                        size_t j = jb;
                        for (; j + 8 <= je; j += 8) {
                            double c0[8], c1[8], c2[8], c3[8];
                            #pragma omp simd
                            for (size_t x = 0; x < 8; ++x) {
                                c0[x] = a[i * n + j + x];
                                c1[x] = a[(i + 1) * n + j + x];
                                c2[x] = a[(i + 2) * n + j + x];
                                c3[x] = a[(i + 3) * n + j + x];
                            }
                            for (size_t k = first; k < last; ++k) {
                                const double f0 = a[i * n + k];
                                const double f1 = a[(i + 1) * n + k];
                                const double f2 = a[(i + 2) * n + k];
                                const double f3 = a[(i + 3) * n + k];
                                const double* const source = packed + (k - first) * panelStride + j;
                                #pragma omp simd
                                for (size_t x = 0; x < 8; ++x) {
                                    c0[x] -= f0 * source[x];
                                    c1[x] -= f1 * source[x];
                                    c2[x] -= f2 * source[x];
                                    c3[x] -= f3 * source[x];
                                }
                            }
                            #pragma omp simd
                            for (size_t x = 0; x < 8; ++x) {
                                a[i * n + j + x] = c0[x];
                                a[(i + 1) * n + j + x] = c1[x];
                                a[(i + 2) * n + j + x] = c2[x];
                                a[(i + 3) * n + j + x] = c3[x];
                            }
                        }
                        // Only the last matrix tile can have a column tail.
                        for (size_t r = i; r < i + 4; ++r)
                            for (size_t k = first; k < last; ++k) {
                                const double factor = a[r * n + k];
                                const double* const source = packed + (k - first) * panelStride;
                                #pragma omp simd
                                for (size_t x = j; x < je; ++x)
                                    a[r * n + x] -= factor * source[x];
                            }
                    }
                }
                for (; i < ie; ++i) {
                    const size_t limit = std::min(je, i + 1);
                    double* const dest = a + i * n;
                    for (size_t k = first; k < last; ++k) {
                        const double factor = dest[k];
                        const double* const source = packed + (k - first) * panelStride;
                        #pragma omp simd
                        for (size_t j = jb; j < limit; ++j)
                            dest[j] -= factor * source[j];
                    }
                }
            }
        }
        if (success) {
            #pragma omp for schedule(static)
            for (size_t i = 0; i < n; ++i)
                std::fill(a + i * n + i + 1, a + (i + 1) * n, 0.0);
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
    
    // Compute one triangle and mirror it; each pair has a unique owner.
    #pragma omp parallel for schedule(dynamic, 1)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (size_t k = 0; k < n; ++k)
                sum += B[i * n + k] * B[j * n + k];
            if (i == j) sum += n;
            A[i * n + j] = sum;
            A[j * n + i] = sum;
        }
    }

}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Upper entries of L are zero, so only the shared lower prefix matters.
    // Symmetry lets us check each pair once without a reconstructed matrix.
    double maxError = 0.0;
    double relError = 0.0;
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
    double gflops = ops / std::chrono::duration<double>(end - start).count() / 1e9;
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
