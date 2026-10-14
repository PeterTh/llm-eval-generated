#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <omp.h>

#include "../common/results_output.hpp"

// Simple Cholesky decomposition (column-oriented, unblocked algorithm, OpenMP parallel)
// Decomposes positive definite matrix A into L * L^T where L is lower triangular

// Team size for an O(work) kernel, where work counts multiply-adds. Each thread needs enough
// work to amortize fork and barrier costs, and very large teams synchronize slowly.
static int teamSize(const size_t work) {
    constexpr size_t kWorkPerThread = size_t(1) << 23;
    constexpr int kMaxThreads = 64;
    const size_t maxThreads = std::min<size_t>(omp_get_max_threads(), kMaxThreads);
    return static_cast<int>(std::clamp<size_t>(work / kWorkPerThread, 1, maxThreads));
}

// Computes L[i][j] for `count` consecutive rows starting at i0 (all > j), given that
// columns 0..j-1 are final. Each sum still accumulates k in ascending order, so results
// are bitwise identical to the sequential algorithm. For full blocks of four rows the
// four independent dependency chains are interleaved to hide floating-point add latency.
static void updateColumnRows(double* A, const size_t n, const size_t j, const double ljj,
                             const size_t i0, const size_t count) {
    const double* Lj = A + j * n;
    if (count == 4) {
        double* r0 = A + i0 * n;
        double* r1 = r0 + n;
        double* r2 = r1 + n;
        double* r3 = r2 + n;
        double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
        for (size_t k = 0; k < j; ++k) {
            const double b = Lj[k];
            s0 += r0[k] * b;
            s1 += r1[k] * b;
            s2 += r2[k] * b;
            s3 += r3[k] * b;
        }
        r0[j] = (r0[j] - s0) / ljj;
        r1[j] = (r1[j] - s1) / ljj;
        r2[j] = (r2[j] - s2) / ljj;
        r3[j] = (r3[j] - s3) / ljj;
    } else {
        for (size_t r = 0; r < count; ++r) {
            double* row = A + (i0 + r) * n;
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k) {
                sum += row[k] * Lj[k];
            }
            row[j] = (row[j] - sum) / ljj;
        }
    }
}

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    // A is stored in row-major order
    // Once columns 0..j-1 of L are final, every entry L[i][j] with i > j depends only on
    // those columns, so the entries of column j below the diagonal are computed in parallel.
    const int team = teamSize(n * n * n / 6);
    size_t failedAt = n;
    #pragma omp parallel num_threads(team)
    for (size_t j = 0; j < n; ++j) {
        // Diagonal element: every thread computes it redundantly (O(j) work), which avoids
        // a barrier. All threads read the same inputs, so they reach the same decision.
        double sum = 0.0;
        for (size_t k = 0; k < j; ++k) {
            sum += A[j * n + k] * A[j * n + k];
        }
        const double val = A[j * n + j] - sum;
        if (val <= 0.0) {
            // Matrix is not positive definite
            #pragma omp master
            {
                printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
                failedAt = j;
            }
            break;
        }
        const double ljj = sqrt(val);

        // Off-diagonal elements of column j
        #pragma omp for schedule(static)
        for (size_t i = j + 1; i < n; i += 4) {
            updateColumnRows(A.data(), n, j, ljj, i, std::min<size_t>(4, n - i));
        }

        // Store the diagonal after the barrier above, when no thread still reads it
        #pragma omp master
        A[j * n + j] = ljj;
    }

    if (failedAt != n) {
        return false;
    }

    // Zero out upper triangular part
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            A[i * n + j] = 0.0;
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
    #pragma omp parallel for schedule(static) num_threads(teamSize(n * n * n))
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
    #pragma omp parallel for schedule(static) num_threads(teamSize(n * n * n))
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
