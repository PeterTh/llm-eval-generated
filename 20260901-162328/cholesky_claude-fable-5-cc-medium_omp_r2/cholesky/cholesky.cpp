#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <omp.h>

#include "../common/results_output.hpp"

// Blocked right-looking Cholesky decomposition, parallelized with OpenMP.
// Decomposes positive definite matrix A into L * L^T where L is lower triangular

// Rank-kb update of one tile of the trailing matrix:
// A[i][j] -= dot(A[i][k0:k0+kb], A[j][k0:k0+kb]) for i in [i0,i1), j in [j0, min(j1,i+1))
static inline void trailingUpdateTile(double* const a, const size_t n,
                                      const size_t k0, const size_t kb,
                                      const size_t i0, const size_t i1,
                                      const size_t j0, const size_t j1) {
    if (j1 <= i0) {
        // Rectangular tile strictly below the diagonal: register-blocked 4x2 kernel
        size_t i = i0;
        for (; i + 4 <= i1; i += 4) {
            const double* const ai0 = a + (i + 0) * n + k0;
            const double* const ai1 = a + (i + 1) * n + k0;
            const double* const ai2 = a + (i + 2) * n + k0;
            const double* const ai3 = a + (i + 3) * n + k0;
            size_t j = j0;
            for (; j + 2 <= j1; j += 2) {
                const double* const aj0 = a + (j + 0) * n + k0;
                const double* const aj1 = a + (j + 1) * n + k0;
                double s00 = 0.0, s01 = 0.0, s10 = 0.0, s11 = 0.0;
                double s20 = 0.0, s21 = 0.0, s30 = 0.0, s31 = 0.0;
                for (size_t k = 0; k < kb; ++k) {
                    const double b0 = aj0[k];
                    const double b1 = aj1[k];
                    s00 += ai0[k] * b0; s01 += ai0[k] * b1;
                    s10 += ai1[k] * b0; s11 += ai1[k] * b1;
                    s20 += ai2[k] * b0; s21 += ai2[k] * b1;
                    s30 += ai3[k] * b0; s31 += ai3[k] * b1;
                }
                a[(i + 0) * n + j] -= s00; a[(i + 0) * n + j + 1] -= s01;
                a[(i + 1) * n + j] -= s10; a[(i + 1) * n + j + 1] -= s11;
                a[(i + 2) * n + j] -= s20; a[(i + 2) * n + j + 1] -= s21;
                a[(i + 3) * n + j] -= s30; a[(i + 3) * n + j + 1] -= s31;
            }
            for (; j < j1; ++j) {
                const double* const aj = a + j * n + k0;
                double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
                for (size_t k = 0; k < kb; ++k) {
                    const double b = aj[k];
                    s0 += ai0[k] * b;
                    s1 += ai1[k] * b;
                    s2 += ai2[k] * b;
                    s3 += ai3[k] * b;
                }
                a[(i + 0) * n + j] -= s0;
                a[(i + 1) * n + j] -= s1;
                a[(i + 2) * n + j] -= s2;
                a[(i + 3) * n + j] -= s3;
            }
        }
        for (; i < i1; ++i) {
            const double* const ai = a + i * n + k0;
            for (size_t j = j0; j < j1; ++j) {
                const double* const aj = a + j * n + k0;
                double s = 0.0;
                for (size_t k = 0; k < kb; ++k) {
                    s += ai[k] * aj[k];
                }
                a[i * n + j] -= s;
            }
        }
    } else {
        // Tile straddling the diagonal: honor the triangular bound
        for (size_t i = i0; i < i1; ++i) {
            const double* const ai = a + i * n + k0;
            const size_t jend = std::min(j1, i + 1);
            for (size_t j = j0; j < jend; ++j) {
                const double* const aj = a + j * n + k0;
                double s = 0.0;
                for (size_t k = 0; k < kb; ++k) {
                    s += ai[k] * aj[k];
                }
                a[i * n + j] -= s;
            }
        }
    }
}

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    // A is stored in row-major order
    constexpr size_t NB = 128; // panel width
    constexpr size_t TB = 64;  // trailing-update tile size

    double* const a = A.data();
    bool ok = true;

    // Cap the team size by the available tile-level parallelism so small
    // problems are not dominated by synchronization overhead
    const size_t nt0 = (n + TB - 1) / TB;
    const size_t npairs0 = nt0 * (nt0 + 1) / 2;
    const int nthreads = (int)std::min<size_t>(
        (size_t)omp_get_max_threads(), std::max<size_t>(1, npairs0 / 4));

    #pragma omp parallel shared(ok) num_threads(nthreads)
    {
        for (size_t k0 = 0; k0 < n; k0 += NB) {
            const size_t kb = std::min(NB, n - k0);
            const size_t kend = k0 + kb;

            // Factor the diagonal block A[k0:kend, k0:kend] (small, sequential)
            #pragma omp single
            {
                for (size_t j = k0; j < kend && ok; ++j) {
                    double sum = 0.0;
                    for (size_t k = k0; k < j; ++k) {
                        sum += a[j * n + k] * a[j * n + k];
                    }
                    const double val = a[j * n + j] - sum;
                    if (val <= 0.0) {
                        // Matrix is not positive definite
                        printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
                        ok = false;
                    } else {
                        const double d = sqrt(val);
                        a[j * n + j] = d;
                        const double inv = 1.0 / d;
                        for (size_t i = j + 1; i < kend; ++i) {
                            double s = 0.0;
                            for (size_t k = k0; k < j; ++k) {
                                s += a[i * n + k] * a[j * n + k];
                            }
                            a[i * n + j] = (a[i * n + j] - s) * inv;
                        }
                    }
                }
            } // implicit barrier

            if (!ok) break;

            // Panel solve: rows below the diagonal block against its transpose
            #pragma omp for schedule(static)
            for (size_t i = kend; i < n; ++i) {
                for (size_t j = k0; j < kend; ++j) {
                    double s = 0.0;
                    for (size_t k = k0; k < j; ++k) {
                        s += a[i * n + k] * a[j * n + k];
                    }
                    a[i * n + j] = (a[i * n + j] - s) / a[j * n + j];
                }
            }

            // Trailing update (symmetric rank-kb update of the lower triangle)
            // Flatten the triangular set of (ib, jb) tile pairs into one index
            // so every thread gets work even late in the factorization
            const size_t nt = (n - kend + TB - 1) / TB;
            const size_t npairs = nt * (nt + 1) / 2;
            #pragma omp for schedule(dynamic)
            for (size_t t = 0; t < npairs; ++t) {
                size_t ti = (size_t)((std::sqrt(8.0 * (double)t + 1.0) - 1.0) / 2.0);
                while (ti * (ti + 1) / 2 > t) --ti;
                while ((ti + 1) * (ti + 2) / 2 <= t) ++ti;
                const size_t tj = t - ti * (ti + 1) / 2;
                const size_t ib = kend + ti * TB;
                const size_t jb = kend + tj * TB;
                trailingUpdateTile(a, n, k0, kb, ib, std::min(ib + TB, n),
                                   jb, std::min(jb + TB, n));
            }
        }
    }

    if (!ok) {
        return false;
    }

    // Zero out upper triangular part
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            a[i * n + j] = 0.0;
        }
    }

    return true;
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
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += B[i * n + k] * B[j * n + k];
            }
            A[i * n + j] = sum;
        }
    }
    
    // Add diagonal dominance to ensure positive definiteness
    for (size_t i = 0; i < n; ++i) {
        A[i * n + i] += n;
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix
    
    std::vector<double> reconstructed(n * n);
    
    // Compute L * L^T
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += L[i * n + k] * L[j * n + k];
            }
            reconstructed[i * n + j] = sum;
        }
    }
    
    // Compare with original
    double maxError = 0.0;
    double relError = 0.0;
    
    #pragma omp parallel for schedule(static) reduction(max : maxError, relError)
    for (size_t i = 0; i < n * n; ++i) {
        const double error = fabs(reconstructed[i] - A_orig[i]);
        maxError = std::max(maxError, error);

        const double rel = error / (fabs(A_orig[i]) + 1e-10);
        relError = std::max(relError, rel);
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
