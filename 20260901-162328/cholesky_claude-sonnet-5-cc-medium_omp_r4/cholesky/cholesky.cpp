#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

#include <omp.h>

#include "../common/results_output.hpp"

// Blocked, right-looking Cholesky decomposition parallelized with OpenMP.
// Decomposes positive definite matrix A into L * L^T where L is lower triangular.
// Mathematically equivalent to the naive unblocked algorithm; only the order
// in which floating point sums are accumulated changes (as with any blocked
// factorization), which is within the validator's relative-error tolerance.

namespace {

constexpr size_t kBlockSize = 64;

// Factorizes the kb x kb diagonal block starting at (k, k) in place.
// This block has already received all trailing updates from previous
// block columns, so only within-block sums are required.
bool factorizeDiagonalBlock(std::vector<double>& A, const size_t n, const size_t k, const size_t kb) {
    for (size_t ii = 0; ii < kb; ++ii) {
        const size_t i = k + ii;
        for (size_t jj = 0; jj <= ii; ++jj) {
            const size_t j = k + jj;
            double sum = 0.0;
            for (size_t m = k; m < j; ++m) {
                sum += A[i * n + m] * A[j * n + m];
            }
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
    return true;
}

// Applies the triangular solve for the (i, k) panel block: turns
// A[i..i+ib, k..k+kb] into L[i..i+ib, k..k+kb] given the already
// factorized diagonal block L[k..k+kb, k..k+kb].
void solvePanelBlock(std::vector<double>& A, const size_t n, const size_t i, const size_t ib,
                      const size_t k, const size_t kb) {
    for (size_t jj = 0; jj < kb; ++jj) {
        const size_t j = k + jj;
        const double diag = A[j * n + j];
        for (size_t rr = 0; rr < ib; ++rr) {
            const size_t r = i + rr;
            double sum = 0.0;
            for (size_t m = k; m < j; ++m) {
                sum += A[r * n + m] * A[j * n + m];
            }
            A[r * n + j] = (A[r * n + j] - sum) / diag;
        }
    }
}

// Applies the trailing (Schur complement) update to the block at
// (bi, bj): A[bi,bj] -= L[bi,k] * L[bj,k]^T. Only the lower triangle is
// updated when bi == bj since the upper triangle is discarded at the end.
void updateTrailingBlock(std::vector<double>& A, const size_t n, const size_t bi, const size_t ib,
                          const size_t bj, const size_t jb, const size_t k, const size_t kb) {
    const bool diagBlock = (bi == bj);
    for (size_t rr = 0; rr < ib; ++rr) {
        const size_t r = bi + rr;
        const size_t cmax = diagBlock ? rr + 1 : jb;
        for (size_t cc = 0; cc < cmax; ++cc) {
            const size_t c = bj + cc;
            double sum = 0.0;
            for (size_t m = k; m < k + kb; ++m) {
                sum += A[r * n + m] * A[c * n + m];
            }
            A[r * n + c] -= sum;
        }
    }
}

}  // namespace

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    // A is stored in row-major order.
    bool success = true;

    std::vector<size_t> rowBlocks;
    std::vector<std::pair<size_t, size_t>> blockPairs;

    for (size_t k = 0; k < n && success; k += kBlockSize) {
        const size_t kb = std::min(kBlockSize, n - k);

        if (!factorizeDiagonalBlock(A, n, k, kb)) {
            success = false;
            break;
        }

        rowBlocks.clear();
        for (size_t i = k + kb; i < n; i += kBlockSize) {
            rowBlocks.push_back(i);
        }

        // Triangular solve for every panel block below the diagonal block;
        // each block is independent since they write disjoint rows.
        #pragma omp parallel for schedule(dynamic) default(none) shared(A, n, k, kb, rowBlocks, kBlockSize)
        for (size_t idx = 0; idx < rowBlocks.size(); ++idx) {
            const size_t i = rowBlocks[idx];
            const size_t ib = std::min(kBlockSize, n - i);
            solvePanelBlock(A, n, i, ib, k, kb);
        }

        blockPairs.clear();
        for (size_t idx = 0; idx < rowBlocks.size(); ++idx) {
            for (size_t jdx = 0; jdx <= idx; ++jdx) {
                blockPairs.emplace_back(rowBlocks[idx], rowBlocks[jdx]);
            }
        }

        // Trailing submatrix update: every (bi, bj) block pair writes
        // disjoint entries, so all pairs can be processed in parallel.
        #pragma omp parallel for schedule(dynamic) default(none) shared(A, n, k, kb, rowBlocks, blockPairs, kBlockSize)
        for (size_t p = 0; p < blockPairs.size(); ++p) {
            const size_t bi = blockPairs[p].first;
            const size_t bj = blockPairs[p].second;
            const size_t ib = std::min(kBlockSize, n - bi);
            const size_t jb = std::min(kBlockSize, n - bj);
            updateTrailingBlock(A, n, bi, ib, bj, jb, k, kb);
        }
    }

    if (!success) {
        return false;
    }

    // Zero out upper triangular part.
    #pragma omp parallel for schedule(static) default(none) shared(A, n)
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
    #pragma omp parallel for schedule(static) default(none) shared(A, B, n)
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
    #pragma omp parallel for schedule(static) default(none) shared(A, n)
    for (size_t i = 0; i < n; ++i) {
        A[i * n + i] += n;
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix
    
    std::vector<double> reconstructed(n * n);
    
    // Compute L * L^T
    #pragma omp parallel for schedule(static) default(none) shared(L, reconstructed, n)
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

    #pragma omp parallel for schedule(static) default(none) shared(reconstructed, A_orig, n) \
        reduction(max : maxError, relError)
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
