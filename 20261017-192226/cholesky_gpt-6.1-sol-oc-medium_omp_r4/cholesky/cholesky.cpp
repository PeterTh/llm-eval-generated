#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// Blocked, right-looking Cholesky decomposition using a persistent OpenMP team.
// Decomposes positive definite matrix A into L * L^T where L is lower triangular

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    constexpr size_t blockSize = 64;
    double* const a = A.data();
    bool success = true;

    #pragma omp parallel shared(success)
    {
        for (size_t first = 0; first < n; first += blockSize) {
            const size_t end = std::min(first + blockSize, n);

            // Previous trailing updates have already subtracted earlier panels.
            #pragma omp single
            {
                for (size_t j = first; j < end; ++j) {
                    double sum = 0.0;
                    for (size_t k = first; k < j; ++k) {
                        sum += a[j * n + k] * a[j * n + k];
                    }
                    const double val = a[j * n + j] - sum;
                    if (val <= 0.0) {
                        printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
                        success = false;
                        break;
                    }
                    a[j * n + j] = sqrt(val);
                    for (size_t i = j + 1; i < end; ++i) {
                        double dot = 0.0;
                        for (size_t k = first; k < j; ++k) {
                            dot += a[i * n + k] * a[j * n + k];
                        }
                        a[i * n + j] = (a[i * n + j] - dot) / a[j * n + j];
                    }
                }
            } // The single barrier publishes the panel and failure status.
            if (!success) break;

            // Solve L21 * L11^T = A21; each row is independent.
            #pragma omp for schedule(static)
            for (size_t i = end; i < n; ++i) {
                for (size_t j = first; j < end; ++j) {
                    double sum = 0.0;
                    for (size_t k = first; k < j; ++k) {
                        sum += a[i * n + k] * a[j * n + k];
                    }
                    a[i * n + j] = (a[i * n + j] - sum) / a[j * n + j];
                }
            }

            // Each task owns a disjoint lower-triangular tile. Pack the right
            // panel transposed on the stack so updates vectorize across columns.
            const size_t tiles = (n - end + blockSize - 1) / blockSize;
            #pragma omp for schedule(dynamic, 1)
            for (size_t tile = 0; tile < tiles * tiles; ++tile) {
                const size_t ti = tile / tiles;
                const size_t tj = tile % tiles;
                if (tj > ti) continue;
                const size_t ib = end + ti * blockSize;
                const size_t jb = end + tj * blockSize;
                const size_t ie = std::min(ib + blockSize, n);
                const size_t je = std::min(jb + blockSize, n);
                alignas(64) double panel[blockSize][blockSize];
                for (size_t k = first; k < end; ++k) {
                    for (size_t j = jb; j < je; ++j) {
                        panel[k - first][j - jb] = a[j * n + k];
                    }
                }
                size_t i = ib;
                // A 4-by-8 register microtile reuses each packed value for
                // four rows and keeps partial sums out of the matrix memory.
                // Off-diagonal tiles always have a full block of columns.
                for (; ti != tj && i + 3 < ie; i += 4) {
                    for (size_t j = jb; j < je; j += 8) {
                        constexpr size_t width = 8;
                        double c0[8], c1[8], c2[8], c3[8];
                        #pragma omp simd
                        for (size_t x = 0; x < width; ++x) {
                            c0[x] = a[i * n + j + x];
                            c1[x] = a[(i + 1) * n + j + x];
                            c2[x] = a[(i + 2) * n + j + x];
                            c3[x] = a[(i + 3) * n + j + x];
                        }
                        for (size_t k = first; k < end; ++k) {
                            const double f0 = a[i * n + k];
                            const double f1 = a[(i + 1) * n + k];
                            const double f2 = a[(i + 2) * n + k];
                            const double f3 = a[(i + 3) * n + k];
                            #pragma omp simd
                            for (size_t x = 0; x < width; ++x) {
                                const double p = panel[k - first][j - jb + x];
                                c0[x] -= f0 * p;
                                c1[x] -= f1 * p;
                                c2[x] -= f2 * p;
                                c3[x] -= f3 * p;
                            }
                        }
                        #pragma omp simd
                        for (size_t x = 0; x < width; ++x) {
                            a[i * n + j + x] = c0[x];
                            a[(i + 1) * n + j + x] = c1[x];
                            a[(i + 2) * n + j + x] = c2[x];
                            a[(i + 3) * n + j + x] = c3[x];
                        }
                    }
                }
                // Diagonal tiles and remaining rows need triangular bounds.
                for (; i < ie; ++i) {
                    const size_t limit = std::min(je, i + 1);
                    for (size_t k = first; k < end; ++k) {
                        const double factor = a[i * n + k];
                        #pragma omp simd
                        for (size_t j = jb; j < limit; ++j) {
                            a[i * n + j] -= factor * panel[k - first][j - jb];
                        }
                    }
                }
            } // All trailing updates finish before the next diagonal panel.
        }

        #pragma omp for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i + 1; j < n; ++j) {
                a[i * n + j] = 0.0;
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
    
    // Compute only one triangle, preserving the original summation order and
    // random stream. Distinct iterations own distinct symmetric element pairs.
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
        // Add diagonal dominance to ensure positive definiteness.
        A[i * n + i] += n;
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix
    
    double maxError = 0.0;
    double relError = 0.0;

    // L is triangular, so terms beyond min(i,j) are zero. Fuse the
    // reconstruction and comparison to avoid an additional n-by-n matrix.
    #pragma omp parallel for schedule(dynamic, 1) reduction(max:maxError,relError)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k <= j; ++k) {
                sum += L[i * n + k] * L[j * n + k];
            }
            const double error = fabs(sum - A_orig[i * n + j]);
            maxError = std::max(maxError, error);
            relError = std::max(relError, error / (fabs(A_orig[i * n + j]) + 1e-10));
            // Also check the upper half if the supplied reference is asymmetric.
            const double upperError = fabs(sum - A_orig[j * n + i]);
            maxError = std::max(maxError, upperError);
            relError = std::max(relError, upperError / (fabs(A_orig[j * n + i]) + 1e-10));
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
