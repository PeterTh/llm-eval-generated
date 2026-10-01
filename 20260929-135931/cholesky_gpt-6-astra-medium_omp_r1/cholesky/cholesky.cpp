#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// Blocked right-looking Cholesky, operating on the lower triangle only.
// A single OpenMP team persists across panels; worksharing barriers enforce
// diagonal -> panel solve -> trailing update dependencies.
bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    constexpr size_t panelSize = 32;
    constexpr size_t tileSize = 64;
    // Pack panel rows with padding to avoid cache-set conflicts for power-of-two n.
    constexpr size_t panelStride = panelSize + 8;
    std::vector<double> panel(n * panelStride);
    bool success = true;

    #pragma omp parallel shared(A, success)
    {
        for (size_t first = 0; first < n; first += panelSize) {
            const size_t last = std::min(first + panelSize, n);
            #pragma omp single
            {
                for (size_t i = first; i < last && success; ++i) {
                    for (size_t j = first; j <= i; ++j) {
                        double sum = 0.0;
                        #pragma omp simd reduction(+:sum)
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
            }
            // The single-region barrier publishes failure to the entire team.
            if (!success) break;

            #pragma omp for schedule(static)
            for (size_t i = last; i < n; ++i) {
                for (size_t j = first; j < last; ++j) {
                    double sum = 0.0;
                    #pragma omp simd reduction(+:sum)
                    for (size_t k = first; k < j; ++k)
                        sum += A[i * n + k] * A[j * n + k];
                    A[i * n + j] = (A[i * n + j] - sum) / A[j * n + j];
                }
                std::copy_n(A.data() + i * n + first, last - first,
                            panel.data() + i * panelStride);
            }

            const size_t tiles = (n - last + tileSize - 1) / tileSize;
            #pragma omp for collapse(2) schedule(dynamic, 1)
            for (size_t ti = 0; ti < tiles; ++ti) {
                for (size_t tj = 0; tj < tiles; ++tj) {
                    if (tj > ti) continue;
                    const size_t ib = last + ti * tileSize;
                    const size_t jb = last + tj * tileSize;
                    const size_t ie = std::min(ib + tileSize, n);
                    const size_t je = std::min(jb + tileSize, n);
                    for (size_t i = ib; i < ie; ++i) {
                        const size_t stop = std::min(je, i + 1);
                        size_t j = jb;
                        // Reuse one panel row across four vectorized dot products.
                        for (; j + 4 <= stop; j += 4) {
                            double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
                            #pragma omp simd reduction(+:s0,s1,s2,s3)
                            for (size_t k = 0; k < last - first; ++k) {
                                const double x = panel[i * panelStride + k];
                                s0 += x * panel[j * panelStride + k];
                                s1 += x * panel[(j + 1) * panelStride + k];
                                s2 += x * panel[(j + 2) * panelStride + k];
                                s3 += x * panel[(j + 3) * panelStride + k];
                            }
                            A[i * n + j] -= s0;
                            A[i * n + j + 1] -= s1;
                            A[i * n + j + 2] -= s2;
                            A[i * n + j + 3] -= s3;
                        }
                        for (; j < stop; ++j) {
                            double sum = 0.0;
                            #pragma omp simd reduction(+:sum)
                            for (size_t k = 0; k < last - first; ++k)
                                sum += panel[i * panelStride + k] * panel[j * panelStride + k];
                            A[i * n + j] -= sum;
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
    
    // Keep the original random stream independent of the thread count.
    // Generate random matrix B
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }
    
    // Compute each symmetric pair once. Rows have independent ownership.
    #pragma omp parallel for schedule(dynamic, 4)
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
    // Fuse reconstruction and comparison, avoiding another dense matrix.
    double maxError = 0.0;
    double relError = 0.0;
    #pragma omp parallel for schedule(dynamic, 4) reduction(max:maxError,relError)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (size_t k = 0; k <= j; ++k)
                sum += L[i * n + k] * L[j * n + k];
            // Check both entries even if the supplied original is asymmetric.
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
