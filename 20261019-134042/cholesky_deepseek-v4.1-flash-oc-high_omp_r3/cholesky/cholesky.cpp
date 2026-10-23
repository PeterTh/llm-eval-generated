#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <omp.h>

#include "../common/results_output.hpp"

// Shared-memory parallel Cholesky decomposition (OpenMP).
// Decomposes positive definite matrix A into L * L^T where L is lower triangular.
//
// The left-looking formulation of the sequential code is preserved: every
// entry L[i][j] is still obtained from the exact dot product
//   sum_{k<j} L[i][k] * L[j][k]
// evaluated in increasing k order.  Consequently the numerical results are
// identical to the original implementation.
//
// Columns are processed in panels.  For each panel:
//   1. the small diagonal block is factorized (sequential, since its rows
//      depend on each other),
//   2. all rows below the panel are solved independently and distributed over
//      the cores.  This step contains the O(n^3) bulk of the work.
// Keeping one parallel region alive for the entire factorization avoids
// repeated thread creation, and using panels rather than single columns keeps
// the number of barriers low, which is essential for scalability.

static constexpr size_t BLOCK_SIZE = 8;

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    double* const base = A.data();
    bool ok = true;

#pragma omp parallel
    {
        for (size_t blockStart = 0; blockStart < n; blockStart += BLOCK_SIZE) {
            const size_t blockEnd = std::min(blockStart + BLOCK_SIZE, n);

            // ---------------------------------------------------------------
            // 1. Factorize the diagonal block sequentially.
            // ---------------------------------------------------------------
#pragma omp single
            {
                for (size_t i = blockStart; i < blockEnd && ok; ++i) {
                    double* const Ai = base + i * n;

                    // Off-diagonal entries of the panel rows.
                    for (size_t j = blockStart; j < i; ++j) {
                        const double* const Aj = base + j * n;
                        double sum = 0.0;
                        for (size_t k = 0; k < j; ++k) {
                            sum += Ai[k] * Aj[k];
                        }
                        Ai[j] = (Ai[j] - sum) / Aj[j];
                    }

                    // Diagonal entry.
                    double sum = 0.0;
                    for (size_t k = 0; k < i; ++k) {
                        sum += Ai[k] * Ai[k];
                    }
                    const double val = Ai[i] - sum;
                    if (val <= 0.0) {
                        printf("Error: Matrix is not positive definite at diagonal element %zu\n", i);
                        ok = false;
                        break;
                    }
                    Ai[i] = std::sqrt(val);
                }
            }
            // implicit barrier: all threads observe `ok`
            if (!ok) {
                break;
            }

            // ---------------------------------------------------------------
            // 2. Solve the panel below the diagonal block.  Rows are fully
            //    independent; within a row the columns of the panel are
            //    solved in order, each with the exact reference dot product.
            // ---------------------------------------------------------------
#pragma omp for schedule(static)
            for (size_t i = blockEnd; i < n; ++i) {
                double* const Ai = base + i * n;
                for (size_t j = blockStart; j < blockEnd; ++j) {
                    const double* const Aj = base + j * n;
                    double sum = 0.0;
                    for (size_t k = 0; k < j; ++k) {
                        sum += Ai[k] * Aj[k];
                    }
                    Ai[j] = (Ai[j] - sum) / Aj[j];
                }
            }
            // implicit barrier
        }
    }

    if (!ok) {
        return false;
    }

    // Zero out the upper triangular part (the original code guarantees this).
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        double* const Ai = base + i * n;
        for (size_t j = i + 1; j < n; ++j) {
            Ai[j] = 0.0;
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

    // Compute A = B * B^T (each entry is computed independently)
#pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            const double* const Bi = B.data() + i * n;
            const double* const Bj = B.data() + j * n;
            for (size_t k = 0; k < n; ++k) {
                sum += Bi[k] * Bj[k];
            }
            A[i * n + j] = sum;
        }
    }

    // Add diagonal dominance to ensure positive definiteness
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        A[i * n + i] += n;
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix

    std::vector<double> reconstructed(n * n);

    // Compute L * L^T
#pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            const double* const Li = L.data() + i * n;
            const double* const Lj = L.data() + j * n;
            for (size_t k = 0; k < n; ++k) {
                sum += Li[k] * Lj[k];
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
