#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// Simple Cholesky decomposition (sequential, unblocked algorithm)
// Decomposes positive definite matrix A into L * L^T where L is lower triangular

bool choleskyDecomposition(std::vector<double>& Av, const size_t n) {
    // Blocked right-looking Cholesky; A is stored in row-major order
    constexpr size_t NB = 64;   // panel width
    constexpr size_t TI = 64;   // trailing update tile size
    double* __restrict A = Av.data();
    std::vector<double> Ptv(NB * n);  // transposed panel: Pt[k*n + j] = A[j, kb+k]
    double* Pt = Ptv.data();

    for (size_t kb = 0; kb < n; kb += NB) {
        const size_t b = std::min(NB, n - kb);
        const size_t s = kb + b;

        // 1. Factor diagonal block (small, sequential)
        for (size_t i = kb; i < s; ++i) {
            for (size_t j = kb; j <= i; ++j) {
                double sum = 0.0;
                const double* ri = A + i * n;
                const double* rj = A + j * n;
                #pragma omp simd reduction(+:sum)
                for (size_t k = kb; k < j; ++k) sum += ri[k] * rj[k];
                if (i == j) {
                    const double val = A[j * n + j] - sum;
                    if (val <= 0.0) {
                        printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
                        return false;
                    }
                    A[j * n + j] = sqrt(val);
                } else {
                    A[i * n + j] = (A[i * n + j] - sum) / A[j * n + j];
                }
            }
        }

        if (s >= n) break;

        // 2. Panel solve: rows below the diagonal block (independent rows)
        #pragma omp parallel for schedule(static) if(n - s >= 128)
        for (size_t i = s; i < n; ++i) {
            double* ri = A + i * n;
            for (size_t j = kb; j < s; ++j) {
                const double* rj = A + j * n;
                double sum = 0.0;
                #pragma omp simd reduction(+:sum)
                for (size_t k = kb; k < j; ++k) sum += ri[k] * rj[k];
                ri[j] = (ri[j] - sum) / rj[j];
            }
            for (size_t k = 0; k < b; ++k) Pt[k * n + i] = ri[kb + k];
        }

        // 3. Trailing update of lower triangle: A[i,j] -= sum_k A[i,k]*A[j,k]
        const size_t nt = n - s;
        const size_t ntI = (nt + TI - 1) / TI;
        #pragma omp parallel for schedule(dynamic, 1) if(nt >= 128)
        for (size_t t = 0; t < ntI * ntI; ++t) {
            const size_t ib = s + (t / ntI) * TI;
            const size_t jb = s + (t % ntI) * TI;
            if (jb > ib) continue;
            const size_t iend = std::min(ib + TI, n);
            const size_t jend0 = std::min(jb + TI, n);
            for (size_t i = ib; i < iend; ++i) {
                double* __restrict ci = A + i * n;
                const size_t jend = std::min(jend0, i + 1);
                const size_t len = jend - jb;
                double acc[TI];
                for (size_t j = 0; j < len; ++j) acc[j] = 0.0;
                for (size_t k = 0; k < b; ++k) {
                    const double aik = ci[kb + k];
                    const double* p = Pt + k * n + jb;
                    #pragma omp simd
                    for (size_t j = 0; j < len; ++j) acc[j] += aik * p[j];
                }
                for (size_t j = 0; j < len; ++j) ci[jb + j] -= acc[j];
            }
        }
    }

    // Zero out upper triangular part
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i)
        for (size_t j = i + 1; j < n; ++j) A[i * n + j] = 0.0;

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
    #pragma omp parallel for schedule(dynamic, 4)
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
    #pragma omp parallel for schedule(dynamic, 4)
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
    
    #pragma omp parallel for reduction(max:maxError,relError)
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
