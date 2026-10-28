#include <omp.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>

#include "../common/results_output.hpp"

// Parallel Cholesky decomposition (OpenMP, unblocked row-oriented algorithm)
// Decomposes positive definite matrix A into L * L^T where L is lower triangular
//
// Parallelization strategy:
//   - Row i of L depends only on the finished rows j < i (their lower parts).
//     Each row is one task; rows are handed out in increasing order via
//     dynamic scheduling, so any row a task waits for has already been started.
//   - Row j is published as finished through an atomic flag (release/acquire),
//     which makes its values visible to the rows that consume them.
//   - Every dot product is accumulated in exactly the same order as the
//     sequential algorithm, so the results are bit-identical to it.
//   - If a diagonal element is not positive, the row that failed is the lowest
//     failing row (all earlier rows were completed before it can be reached),
//     which is reported exactly as in the sequential version.

// Wait until row is finished. Returns false if a failure was signalled meanwhile.
static inline bool waitForRow(const std::atomic<bool>& rowDone, const std::atomic<bool>& failed) {
    unsigned int spins = 0;
    while (!rowDone.load(std::memory_order_acquire)) {
        if (failed.load(std::memory_order_acquire)) {
            return false;
        }
        if (++spins > 64) {
            std::this_thread::yield();
        }
    }
    return true;
}

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    // A is stored in row-major order
    double* const a = A.data();

    std::unique_ptr<std::atomic<bool>[]> rowDone(new std::atomic<bool>[n]);
    for (size_t r = 0; r < n; ++r) {
        rowDone[r].store(false, std::memory_order_relaxed);
    }

    std::atomic<bool> failed{false};
    std::atomic<size_t> failedRow{n};

    #pragma omp parallel for schedule(dynamic, 1)
    for (size_t i = 0; i < n; ++i) {
        if (failed.load(std::memory_order_acquire)) {
            continue;
        }

        double* const Ai = a + i * n;
        bool ok = true;

        // Off-diagonal elements of row i
        for (size_t j = 0; j < i; ++j) {
            if (!waitForRow(rowDone[j], failed)) {
                ok = false;
                break;
            }
            const double* const Aj = a + j * n;
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k) {
                sum += Ai[k] * Aj[k];
            }
            Ai[j] = (Ai[j] - sum) / Aj[j];
        }

        if (!ok) {
            continue;
        }

        // Diagonal element of row i
        {
            double sum = 0.0;
            for (size_t k = 0; k < i; ++k) {
                sum += Ai[k] * Ai[k];
            }
            const double val = Ai[i] - sum;
            if (val <= 0.0) {
                // Matrix is not positive definite
                size_t expected = n;
                failedRow.compare_exchange_strong(expected, i, std::memory_order_relaxed);
                failed.store(true, std::memory_order_release);
                continue;
            }
            Ai[i] = sqrt(val);
        }

        // Zero out upper triangular part
        for (size_t j = i + 1; j < n; ++j) {
            Ai[j] = 0.0;
        }

        rowDone[i].store(true, std::memory_order_release);
    }

    if (failed.load(std::memory_order_acquire)) {
        printf("Error: Matrix is not positive definite at diagonal element %zu\n", failedRow.load());
        return false;
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

    // Generate random matrix B (sequential: the random sequence must be preserved)
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

    // Compute A = B * B^T (rows are independent)
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

    // Compute L * L^T (rows are independent)
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
