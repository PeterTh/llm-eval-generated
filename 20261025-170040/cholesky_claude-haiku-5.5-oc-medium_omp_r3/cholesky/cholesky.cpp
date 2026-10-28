#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <omp.h>

#include "../common/results_output.hpp"

// Parallel Cholesky decomposition (OpenMP, shared memory)
// Decomposes positive definite matrix A into L * L^T where L is lower triangular
//
// The algorithm is a left-looking, column-oriented variant of the classic
// row-oriented algorithm. For every column j:
//   - L[j][j] is computed sequentially (a short reduction over k < j),
//   - all L[i][j] for i > j are independent of each other and are computed
//     in parallel. Each one uses the same reduction order over k as the
//     sequential formulation, so the results are bit-identical to it.
// Rows are processed four at a time so that four independent dot-product
// chains are in flight per thread, improving instruction-level parallelism
// without changing the summation order of any individual element.

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    // A is stored in row-major order
    double* const a = A.data();

    for (size_t j = 0; j < n; ++j) {
        double* const rowj = a + j * n;

        // Diagonal element
        double dsum = 0.0;
        for (size_t k = 0; k < j; ++k) {
            dsum += rowj[k] * rowj[k];
        }
        const double val = rowj[j] - dsum;
        if (val <= 0.0) {
            // Matrix is not positive definite
            printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
            return false;
        }
        const double d = sqrt(val);
        rowj[j] = d;

        // Off-diagonal elements of column j: rows i = j+1 .. n-1
        const size_t first = j + 1;
        const size_t rows = n - first;
        const size_t blocks = rows / 4;
        const size_t work = rows * j;

        #pragma omp parallel for schedule(static) if (work >= 4096)
        for (size_t b = 0; b < blocks; ++b) {
            double* const r0 = a + (first + 4 * b + 0) * n;
            double* const r1 = a + (first + 4 * b + 1) * n;
            double* const r2 = a + (first + 4 * b + 2) * n;
            double* const r3 = a + (first + 4 * b + 3) * n;

            double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
            for (size_t k = 0; k < j; ++k) {
                const double lk = rowj[k];
                s0 += r0[k] * lk;
                s1 += r1[k] * lk;
                s2 += r2[k] * lk;
                s3 += r3[k] * lk;
            }
            r0[j] = (r0[j] - s0) / d;
            r1[j] = (r1[j] - s1) / d;
            r2[j] = (r2[j] - s2) / d;
            r3[j] = (r3[j] - s3) / d;
        }

        // Remainder rows (at most 3)
        for (size_t i = first + 4 * blocks; i < n; ++i) {
            double* const rowi = a + i * n;
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k) {
                sum += rowi[k] * rowj[k];
            }
            rowi[j] = (rowi[j] - sum) / d;
        }
    }

    // Zero out upper triangular part (never read by the algorithm above)
    #pragma omp parallel for schedule(guided)
    for (size_t i = 0; i < n; ++i) {
        double* const rowi = a + i * n;
        for (size_t j = i + 1; j < n; ++j) {
            rowi[j] = 0.0;
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

    // Generate random matrix B (inherently sequential RNG stream)
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

    // Compute A = B * B^T (each element is independent)
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

    // Compute L * L^T (each element is independent)
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
