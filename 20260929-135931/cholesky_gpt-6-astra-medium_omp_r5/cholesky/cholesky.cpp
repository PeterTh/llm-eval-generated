#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// Blocked, right-looking Cholesky decomposition of a row-major matrix.
// Only the lower triangle contributes; on success A contains L with a zero upper
// triangle. The barriers between stages publish each panel before it is used.
bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    constexpr size_t block = 64;
    bool success = true;
    // Pack transposed panel tiles for contiguous SIMD loads. Padding avoids
    // cache-set conflicts, especially for power-of-two matrix dimensions.
    constexpr size_t packedStride = block + 8;
    std::vector<double> panel(((n + block - 1) / block) * block * packedStride);

    #pragma omp parallel shared(A, success)
    {
        for (size_t first = 0; first < n; first += block) {
            const size_t end = std::min(first + block, n);

            #pragma omp single
            {
                // Previous panels have already updated this diagonal tile.
                for (size_t i = first; i < end && success; ++i) {
                    for (size_t j = first; j <= i; ++j) {
                        double sum = 0.0;
                        #pragma omp simd reduction(+:sum)
                        for (size_t k = first; k < j; ++k) {
                            sum += A[i * n + k] * A[j * n + k];
                        }
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
            // All threads observe failure after the single-region barrier.
            if (!success) break;

            // Solve the panel. Each row is independent of other panel rows.
            #pragma omp for schedule(static)
            for (size_t i = end; i < n; ++i) {
                for (size_t j = first; j < end; ++j) {
                    double sum = 0.0;
                    #pragma omp simd reduction(+:sum)
                    for (size_t k = first; k < j; ++k) {
                        sum += A[i * n + k] * A[j * n + k];
                    }
                    A[i * n + j] = (A[i * n + j] - sum) / A[j * n + j];
                    panel[(i / block) * block * packedStride +
                          (j - first) * packedStride + i % block] = A[i * n + j];
                }
            }

            // Distribute tiles rather than whole rows to keep the triangular
            // update balanced and reuse the panel data in cache.
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
                    // A 4-by-8 microkernel keeps independent sums in SIMD
                    // registers and reuses each packed panel vector four times.
                    // Diagonal tiles also update their unused upper half to
                    // keep the kernel regular; it is cleared before returning.
                    for (; i + 4 <= ie; i += 4) {
                        size_t j = jb;
                        for (; j + 8 <= je; j += 8) {
                            double s0[8] = {}, s1[8] = {}, s2[8] = {}, s3[8] = {};
                            const double* packed = panel.data() +
                                (jb / block) * block * packedStride + (j - jb);
                            for (size_t k = first; k < end; ++k) {
                                const double a0 = A[i * n + k];
                                const double a1 = A[(i + 1) * n + k];
                                const double a2 = A[(i + 2) * n + k];
                                const double a3 = A[(i + 3) * n + k];
                                const double* p = packed + (k - first) * packedStride;
                                #pragma omp simd
                                for (size_t c = 0; c < 8; ++c) {
                                    s0[c] += a0 * p[c];
                                    s1[c] += a1 * p[c];
                                    s2[c] += a2 * p[c];
                                    s3[c] += a3 * p[c];
                                }
                            }
                            #pragma omp simd
                            for (size_t c = 0; c < 8; ++c) {
                                A[i * n + j + c] -= s0[c];
                                A[(i + 1) * n + j + c] -= s1[c];
                                A[(i + 2) * n + j + c] -= s2[c];
                                A[(i + 3) * n + j + c] -= s3[c];
                            }
                        }
                        for (size_t r = i; r < i + 4; ++r) {
                            for (size_t c = j; c < je; ++c) {
                                double sum = 0.0;
                                #pragma omp simd reduction(+:sum)
                                for (size_t k = first; k < end; ++k) {
                                    sum += A[r * n + k] * A[c * n + k];
                                }
                                A[r * n + c] -= sum;
                            }
                        }
                    }
                    for (; i < ie; ++i) {
                        for (size_t j = jb; j < je; ++j) {
                            double sum = 0.0;
                            #pragma omp simd reduction(+:sum)
                            for (size_t k = first; k < end; ++k) {
                                sum += A[i * n + k] * A[j * n + k];
                            }
                            A[i * n + j] -= sum;
                        }
                    }
                }
            }
        }

        if (success) {
            #pragma omp for schedule(static)
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = i + 1; j < n; ++j) {
                    A[i * n + j] = 0.0;
                }
            }
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
    
    // Compute each symmetric pair once. Keep the seeded RNG above serial so
    // that the generated input is independent of the OpenMP thread count.
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
        A[i * n + i] += n;
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Fuse reconstruction and comparison, avoiding a second dense matrix.
    double maxError = 0.0;
    double relError = 0.0;
    #pragma omp parallel for schedule(dynamic, 8) reduction(max:maxError,relError)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            const size_t end = std::min(i, j) + 1;
            #pragma omp simd reduction(+:sum)
            for (size_t k = 0; k < end; ++k) {
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
