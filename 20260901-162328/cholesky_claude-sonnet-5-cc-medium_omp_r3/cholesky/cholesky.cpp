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

// Blocked, task-parallel Cholesky decomposition (OpenMP)
// Decomposes positive definite matrix A into L * L^T where L is lower triangular.
// The matrix is processed in BLOCK_SIZE x BLOCK_SIZE tiles using a right-looking
// algorithm: diagonal blocks are factorized, then the panel below is solved via
// triangular solves, then the trailing submatrix is updated. OpenMP tasks with
// depend clauses express the resulting dependency DAG so that independent tile
// operations (in particular the O(n^3) trailing-submatrix updates) run concurrently.

constexpr size_t BLOCK_SIZE = 64;

// Unblocked Cholesky factorization restricted to the diagonal tile [off, off+bs).
static bool potrfBlock(std::vector<double>& A, const size_t n, const size_t off, const size_t bs) {
    for (size_t i = off; i < off + bs; ++i) {
        for (size_t j = off; j <= i; ++j) {
            double sum = 0.0;
            if (i == j) {
                for (size_t k = off; k < j; ++k) {
                    sum += A[j * n + k] * A[j * n + k];
                }
                const double val = A[j * n + j] - sum;
                if (val <= 0.0) {
                    printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
                    return false;
                }
                A[j * n + j] = sqrt(val);
            } else {
                for (size_t k = off; k < j; ++k) {
                    sum += A[i * n + k] * A[j * n + k];
                }
                A[i * n + j] = (A[i * n + j] - sum) / A[j * n + j];
            }
        }
    }
    return true;
}

// Triangular solve: computes tile L[i][k] = A[i][k] * inv(L[k][k])^T given the
// already-factorized diagonal tile L[k][k].
static void trsmBlock(std::vector<double>& A, const size_t n,
                       const size_t iOff, const size_t iSize,
                       const size_t kOff, const size_t kSize) {
    for (size_t ig = iOff; ig < iOff + iSize; ++ig) {
        for (size_t jg = kOff; jg < kOff + kSize; ++jg) {
            double sum = 0.0;
            for (size_t p = kOff; p < jg; ++p) {
                sum += A[ig * n + p] * A[jg * n + p];
            }
            A[ig * n + jg] = (A[ig * n + jg] - sum) / A[jg * n + jg];
        }
    }
}

// Trailing submatrix update for a diagonal tile: A[i][i] -= L[i][k] * L[i][k]^T
// (only the lower triangle is needed/updated).
static void syrkBlock(std::vector<double>& A, const size_t n,
                       const size_t iOff, const size_t iSize,
                       const size_t kOff, const size_t kSize) {
    for (size_t r = iOff; r < iOff + iSize; ++r) {
        for (size_t c = iOff; c <= r; ++c) {
            double sum = 0.0;
            for (size_t p = kOff; p < kOff + kSize; ++p) {
                sum += A[r * n + p] * A[c * n + p];
            }
            A[r * n + c] -= sum;
        }
    }
}

// Trailing submatrix update for an off-diagonal tile: A[i][j] -= L[i][k] * L[j][k]^T
static void gemmBlock(std::vector<double>& A, const size_t n,
                       const size_t iOff, const size_t iSize,
                       const size_t jOff, const size_t jSize,
                       const size_t kOff, const size_t kSize) {
    for (size_t r = iOff; r < iOff + iSize; ++r) {
        for (size_t c = jOff; c < jOff + jSize; ++c) {
            double sum = 0.0;
            for (size_t p = kOff; p < kOff + kSize; ++p) {
                sum += A[r * n + p] * A[c * n + p];
            }
            A[r * n + c] -= sum;
        }
    }
}

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    if (n == 0) {
        return true;
    }

    const size_t nb = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // Dummy per-tile variables used purely as addresses for OpenMP task
    // dependency tracking (depend clauses key on address identity, not value).
    std::vector<char> tags(nb * nb);
    [[maybe_unused]] char* tagsPtr = tags.data();
    std::atomic<bool> posDef{true};

    #pragma omp parallel
    {
        #pragma omp single
        {
            for (size_t k = 0; k < nb; ++k) {
                const size_t kOff = k * BLOCK_SIZE;
                const size_t kSize = std::min(BLOCK_SIZE, n - kOff);
                #pragma omp task depend(inout: tagsPtr[k * nb + k]) shared(A, posDef)
                {
                    if (posDef.load(std::memory_order_relaxed)) {
                        if (!potrfBlock(A, n, kOff, kSize)) {
                            posDef.store(false, std::memory_order_relaxed);
                        }
                    }
                }

                for (size_t i = k + 1; i < nb; ++i) {
                    const size_t iOff = i * BLOCK_SIZE;
                    const size_t iSize = std::min(BLOCK_SIZE, n - iOff);
                    #pragma omp task depend(in: tagsPtr[k * nb + k]) depend(inout: tagsPtr[i * nb + k]) shared(A, posDef)
                    {
                        if (posDef.load(std::memory_order_relaxed)) {
                            trsmBlock(A, n, iOff, iSize, kOff, kSize);
                        }
                    }
                }

                for (size_t i = k + 1; i < nb; ++i) {
                    const size_t iOff = i * BLOCK_SIZE;
                    const size_t iSize = std::min(BLOCK_SIZE, n - iOff);

                    for (size_t j = k + 1; j <= i; ++j) {
                        const size_t jOff = j * BLOCK_SIZE;
                        const size_t jSize = std::min(BLOCK_SIZE, n - jOff);

                        if (i == j) {
                            #pragma omp task depend(in: tagsPtr[i * nb + k]) depend(inout: tagsPtr[i * nb + j]) shared(A, posDef)
                            {
                                if (posDef.load(std::memory_order_relaxed)) {
                                    syrkBlock(A, n, iOff, iSize, kOff, kSize);
                                }
                            }
                        } else {
                            #pragma omp task depend(in: tagsPtr[i * nb + k], tagsPtr[j * nb + k]) depend(inout: tagsPtr[i * nb + j]) shared(A, posDef)
                            {
                                if (posDef.load(std::memory_order_relaxed)) {
                                    gemmBlock(A, n, iOff, iSize, jOff, jSize, kOff, kSize);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if (!posDef.load(std::memory_order_relaxed)) {
        return false;
    }

    // Zero out the upper triangular part
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
    #pragma omp parallel for schedule(static)
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
