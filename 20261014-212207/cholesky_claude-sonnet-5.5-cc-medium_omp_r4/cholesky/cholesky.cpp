#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <omp.h>

#include "../common/results_output.hpp"

// Blocked right-looking Cholesky decomposition parallelized with OpenMP.
// Decomposes positive definite matrix A into L * L^T where L is lower triangular

static const size_t NB = 64;   // panel width
static const size_t TS = 64;   // trailing update tile size

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    // A is stored in row-major order
    double* const a = A.data();
    bool ok = true;

    for (size_t kb = 0; kb < n; kb += NB) {
        const size_t ke = std::min(kb + NB, n);

        // 1. Factor the diagonal block (small, sequential)
        for (size_t i = kb; i < ke; ++i) {
            for (size_t j = kb; j <= i; ++j) {
                double sum = 0.0;
                for (size_t k = kb; k < j; ++k) {
                    sum += a[i * n + k] * a[j * n + k];
                }
                if (i == j) {
                    const double val = a[j * n + j] - sum;
                    if (val <= 0.0) {
                        printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
                        ok = false;
                        break;
                    }
                    a[j * n + j] = sqrt(val);
                } else {
                    a[i * n + j] = (a[i * n + j] - sum) / a[j * n + j];
                }
            }
            if (!ok) return false;
        }

        // 2. Panel solve: rows below the diagonal block
        #pragma omp parallel for schedule(static)
        for (size_t i = ke; i < n; ++i) {
            double* ri = a + i * n;
            for (size_t j = kb; j < ke; ++j) {
                const double* rj = a + j * n;
                double sum = 0.0;
                #pragma omp simd reduction(+:sum)
                for (size_t k = kb; k < j; ++k) {
                    sum += ri[k] * rj[k];
                }
                ri[j] = (ri[j] - sum) / rj[j];
            }
        }

        // 3. Trailing update (lower triangle tiles): A[i][j] -= L[i,kb:ke] . L[j,kb:ke]
        if (ke < n) {
            const size_t nt = (n - ke + TS - 1) / TS;
            const size_t ntiles = nt * (nt + 1) / 2;
            #pragma omp parallel for schedule(dynamic, 1)
            for (size_t t = 0; t < ntiles; ++t) {
                // map linear index to (ti >= tj), largest-row tiles first not needed
                size_t ti = (size_t)((sqrt(8.0 * (double)t + 1.0) - 1.0) / 2.0);
                while (ti * (ti + 1) / 2 > t) --ti;
                while ((ti + 1) * (ti + 2) / 2 <= t) ++ti;
                const size_t tj = t - ti * (ti + 1) / 2;
                const size_t i0 = ke + ti * TS, i1 = std::min(i0 + TS, n);
                const size_t j0 = ke + tj * TS, j1 = std::min(j0 + TS, n);
                for (size_t i = i0; i < i1; ++i) {
                    double* ri = a + i * n;
                    const size_t jend = (tj == ti) ? i + 1 : j1;
                    size_t j = j0;
                    for (; j + 4 <= jend; j += 4) {
                        const double* r0 = a + j * n;
                        const double* r1 = r0 + n;
                        const double* r2 = r1 + n;
                        const double* r3 = r2 + n;
                        double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
                        #pragma omp simd reduction(+:s0,s1,s2,s3)
                        for (size_t k = kb; k < ke; ++k) {
                            const double x = ri[k];
                            s0 += x * r0[k];
                            s1 += x * r1[k];
                            s2 += x * r2[k];
                            s3 += x * r3[k];
                        }
                        ri[j] -= s0; ri[j + 1] -= s1; ri[j + 2] -= s2; ri[j + 3] -= s3;
                    }
                    for (; j < jend; ++j) {
                        const double* r0 = a + j * n;
                        double s0 = 0;
                        #pragma omp simd reduction(+:s0)
                        for (size_t k = kb; k < ke; ++k) s0 += ri[k] * r0[k];
                        ri[j] -= s0;
                    }
                }
            }
        }
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
            #pragma omp simd reduction(+:sum)
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
    
    #pragma omp parallel for schedule(static) reduction(max:maxError,relError)
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
