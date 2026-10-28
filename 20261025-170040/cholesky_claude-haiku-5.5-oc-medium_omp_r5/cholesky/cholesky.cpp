#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <omp.h>

#include "../common/results_output.hpp"

// Team size for parallel regions over 'rows' rows. Each thread should own a
// meaningful number of rows; very large teams are dominated by synchronization
// overhead. Using one size everywhere also avoids creating an oversized thread
// pool that would slow down the (timed) decomposition.
static int teamSize(const size_t rows) {
    const size_t maxThreads = static_cast<size_t>(omp_get_max_threads());
    return static_cast<int>(std::clamp<size_t>(rows / 16, 1, maxThreads));
}

// Simple Cholesky decomposition (unblocked, column-oriented, OpenMP parallel)
// Decomposes positive definite matrix A into L * L^T where L is lower triangular
//
// The factorization proceeds column by column. For column j, every thread
// redundantly computes the diagonal element L[j][j] (identical deterministic
// arithmetic, O(j) work), then the off-diagonal elements L[i][j] (i > j) are
// computed in parallel with a single barrier per column. Each element's dot
// product is accumulated in the same ascending-k order as the sequential
// algorithm, so results are bit-for-bit identical to the sequential version.

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    // A is stored in row-major order
    double* const a = A.data();
    bool ok = true;

    #pragma omp parallel num_threads(teamSize(n)) shared(ok)
    {
        for (size_t j = 0; j < n; ++j) {
            // Diagonal element L[j][j]. Entry a[j][j] is only read in column j,
            // and is published by the master thread after this column's barrier.
            const double* const Lj = a + j * n;
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k) {
                sum += Lj[k] * Lj[k];
            }
            const double val = a[j * n + j] - sum;
            if (val <= 0.0) {
                // Matrix is not positive definite (all threads take this path together)
                #pragma omp master
                {
                    printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
                    ok = false;
                }
                break;
            }
            const double Ljj = sqrt(val);

            // Off-diagonal elements of column j: independent across rows i
            #pragma omp for schedule(static)
            for (size_t i = j + 1; i < n; ++i) {
                const double* const Li = a + i * n;
                double s = 0.0;
                for (size_t k = 0; k < j; ++k) {
                    s += Li[k] * Lj[k];
                }
                a[i * n + j] = (a[i * n + j] - s) / Ljj;
            }

            // All reads of a[j][j] in this column are complete (implicit barrier above)
            #pragma omp master
            {
                a[j * n + j] = Ljj;
            }
        }
    }

    if (!ok) {
        return false;
    }

    // Zero out upper triangular part
    #pragma omp parallel for num_threads(teamSize(n)) schedule(static)
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

    // Generate random matrix B (sequential to keep the sequence deterministic)
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

    // Compute A = B * B^T (rows are independent)
    #pragma omp parallel for num_threads(teamSize(n)) schedule(static)
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

    // Compute L * L^T (rows are independent)
    #pragma omp parallel for num_threads(teamSize(n)) schedule(static)
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

    #pragma omp parallel for num_threads(teamSize(n)) schedule(static) reduction(max : maxError, relError)
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
