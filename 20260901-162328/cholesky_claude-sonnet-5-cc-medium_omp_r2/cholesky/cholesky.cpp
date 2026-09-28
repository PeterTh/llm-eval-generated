#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <omp.h>

#include "../common/results_output.hpp"

// Tiled (blocked) Cholesky decomposition, parallelized with OpenMP tasks.
// Decomposes positive definite matrix A into L * L^T where L is lower triangular.
//
// The matrix is split into blockSize x blockSize tiles. For each panel k:
//   1. The diagonal tile (k,k) is factorized in place (small, sequential).
//   2. The tiles below it in the same column are updated via a triangular
//      solve ("panel solve"), each tile independent of the others.
//   3. The trailing submatrix tiles are updated with a symmetric rank-kb
//      update, each tile independent of the others.
// Task dependencies are expressed through a grid of dummy proxy variables
// (one per tile) so the OpenMP runtime can build the task graph and let
// independent tiles/panels run concurrently and even overlap across
// different values of k, which is what gives this approach good scalability
// on many-core machines.

namespace {

// Factorizes the diagonal tile [k0, k0+kb) x [k0, k0+kb) in place using the
// plain unblocked algorithm, restricted to columns local to this tile (any
// contributions from earlier panels have already been subtracted by the
// trailing updates from previous iterations).
void factorizeDiagonalBlock(std::vector<double>& A, const size_t n, const size_t k0, const size_t kb,
                             std::atomic<bool>& ok) {
    for (size_t i = k0; i < k0 + kb; ++i) {
        for (size_t j = k0; j <= i; ++j) {
            double sum = 0.0;

            if (i == j) {
                for (size_t k = k0; k < j; ++k) {
                    sum += A[j * n + k] * A[j * n + k];
                }
                const double val = A[j * n + j] - sum;
                if (val <= 0.0) {
                    ok.store(false, std::memory_order_relaxed);
                    return;
                }
                A[j * n + j] = sqrt(val);
            } else {
                for (size_t k = k0; k < j; ++k) {
                    sum += A[i * n + k] * A[j * n + k];
                }
                A[i * n + j] = (A[i * n + j] - sum) / A[j * n + j];
            }
        }
    }
}

// Panel (triangular) solve: given the freshly factorized diagonal tile
// (k0,k0)-(k0+kb,k0+kb), solve for the tile rows [i0, i0+ib) of column-block k.
void solvePanelBlock(std::vector<double>& A, const size_t n, const size_t k0, const size_t kb, const size_t i0,
                      const size_t ib) {
    for (size_t r = i0; r < i0 + ib; ++r) {
        for (size_t j = k0; j < k0 + kb; ++j) {
            double sum = 0.0;
            for (size_t c = k0; c < j; ++c) {
                sum += A[r * n + c] * A[j * n + c];
            }
            A[r * n + j] = (A[r * n + j] - sum) / A[j * n + j];
        }
    }
}

// Trailing submatrix update: subtracts the contribution of column-block k
// from tile (i,j). When diag is true, tile (i,j) is a diagonal tile and only
// the lower triangle is updated.
void updateBlock(std::vector<double>& A, const size_t n, const size_t k0, const size_t kb, const size_t i0,
                  const size_t ib, const size_t j0, const size_t jb, const bool diag) {
    for (size_t r = i0; r < i0 + ib; ++r) {
        const size_t jmax = diag ? (r - j0 + 1) : jb;
        for (size_t jj = 0; jj < jmax; ++jj) {
            const size_t c = j0 + jj;
            double sum = 0.0;
            for (size_t kk = k0; kk < k0 + kb; ++kk) {
                sum += A[r * n + kk] * A[c * n + kk];
            }
            A[r * n + c] -= sum;
        }
    }
}

}  // namespace

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    if (n == 0) {
        return true;
    }

    // A is stored in row-major order.
    const size_t blockSize = std::min<size_t>(128, n);
    const size_t numBlocks = (n + blockSize - 1) / blockSize;

    // Dummy proxy variables used purely to express tile-level dependencies
    // to the OpenMP task scheduler. depend() clauses need a plain array
    // section, not a std::vector, hence the raw pointer.
    std::vector<char> tagsStorage(numBlocks * numBlocks);
    char* const tags = tagsStorage.data();
    std::atomic<bool> ok(true);

    #pragma omp parallel
    {
        #pragma omp single
        {
            for (size_t k = 0; k < numBlocks; ++k) {
                const size_t k0 = k * blockSize;
                const size_t kb = std::min(blockSize, n - k0);

                #pragma omp task depend(inout: tags[k * numBlocks + k]) firstprivate(k0, kb) shared(A, ok)
                factorizeDiagonalBlock(A, n, k0, kb, ok);

                for (size_t i = k + 1; i < numBlocks; ++i) {
                    const size_t i0 = i * blockSize;
                    const size_t ib = std::min(blockSize, n - i0);

                    #pragma omp task depend(in: tags[k * numBlocks + k]) \
                                      depend(inout: tags[i * numBlocks + k]) \
                                      firstprivate(k0, kb, i0, ib) shared(A)
                    solvePanelBlock(A, n, k0, kb, i0, ib);
                }

                for (size_t j = k + 1; j < numBlocks; ++j) {
                    const size_t j0 = j * blockSize;
                    const size_t jb = std::min(blockSize, n - j0);

                    for (size_t i = j; i < numBlocks; ++i) {
                        const size_t i0 = i * blockSize;
                        const size_t ib = std::min(blockSize, n - i0);
                        const bool diag = (i == j);

                        #pragma omp task depend(in: tags[i * numBlocks + k], tags[j * numBlocks + k]) \
                                          depend(inout: tags[i * numBlocks + j]) \
                                          firstprivate(k0, kb, i0, ib, j0, jb, diag) shared(A)
                        updateBlock(A, n, k0, kb, i0, ib, j0, jb, diag);
                    }
                }
            }
        }
    }

    if (!ok.load(std::memory_order_relaxed)) {
        printf("Error: Matrix is not positive definite\n");
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

    // Generate random matrix B. This must stay sequential: rand_r() carries
    // a running seed state across calls, so the sequence (and thus the
    // generated matrix) depends on the calls being made in order.
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

    // Compute A = B * B^T
    #pragma omp parallel for schedule(static) collapse(2)
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
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        A[i * n + i] += n;
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix
    
    std::vector<double> reconstructed(n * n);

    // Compute L * L^T
    #pragma omp parallel for schedule(static) collapse(2)
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

    #pragma omp parallel for schedule(static) reduction(max: maxError, relError)
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
