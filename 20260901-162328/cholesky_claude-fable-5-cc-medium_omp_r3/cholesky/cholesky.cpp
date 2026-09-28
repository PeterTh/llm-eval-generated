#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <unistd.h>

#include <omp.h>

#include "../common/results_output.hpp"

// Cholesky decomposition parallelized with OpenMP.
// Uses a tiled right-looking algorithm with task dependencies:
// POTRF on diagonal tiles, TRSM on panel tiles, SYRK/GEMM trailing updates.
// Decomposes positive definite matrix A into L * L^T where L is lower triangular.

// Unblocked Cholesky on an m x m diagonal tile (leading dimension lda).
// Returns m on success, otherwise the local index of the first
// non-positive-definite diagonal element.
static size_t potrfTile(double* A, const size_t lda, const size_t m) {
    for (size_t j = 0; j < m; ++j) {
        double* rowj = A + j * lda;
        double sum = 0.0;
        for (size_t k = 0; k < j; ++k) {
            sum += rowj[k] * rowj[k];
        }
        const double val = rowj[j] - sum;
        if (val <= 0.0) {
            return j;
        }
        rowj[j] = sqrt(val);
        const double inv = 1.0 / rowj[j];
        for (size_t i = j + 1; i < m; ++i) {
            double* rowi = A + i * lda;
            double s = 0.0;
            for (size_t k = 0; k < j; ++k) {
                s += rowi[k] * rowj[k];
            }
            rowi[j] = (rowi[j] - s) * inv;
        }
    }
    return m;
}

// Solve X * L^T = X in place, where L is the kb x kb lower-triangular
// diagonal tile and X is an m x kb panel tile.
static void trsmTile(const double* L, double* X, const size_t lda,
                     const size_t m, const size_t kb) {
    for (size_t r = 0; r < m; ++r) {
        double* x = X + r * lda;
        for (size_t c = 0; c < kb; ++c) {
            const double* lrow = L + c * lda;
            double s = 0.0;
            for (size_t t = 0; t < c; ++t) {
                s += x[t] * lrow[t];
            }
            x[c] = (x[c] - s) / lrow[c];
        }
    }
}

// C -= Aik * Ajk^T for an mi x mj tile (kb inner dimension).
// If diagonal is true, only the lower part (c <= r) is updated.
static void gemmTile(double* C, const double* Aik, const double* Ajk,
                     const size_t lda, const size_t mi, const size_t mj,
                     const size_t kb, const bool diagonal) {
    for (size_t r = 0; r < mi; ++r) {
        const double* ar = Aik + r * lda;
        double* cr = C + r * lda;
        const size_t cend = diagonal ? r + 1 : mj;
        for (size_t c = 0; c < cend; ++c) {
            const double* bc = Ajk + c * lda;
            double s = 0.0;
            for (size_t t = 0; t < kb; ++t) {
                s += ar[t] * bc[t];
            }
            cr[c] -= s;
        }
    }
}

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    // A is stored in row-major order
    double* a = A.data();

    // Pick a tile size that keeps tiles cache-friendly while exposing
    // enough tasks for all threads.
    size_t bs = 128;
    while (bs > 32 && (n + bs - 1) / bs < 8) {
        bs >>= 1;
    }
    const size_t nt = (n + bs - 1) / bs;

    // Global index of the first non-positive-definite diagonal element
    // (n means no failure).
    size_t failIndex = n;

    #pragma omp parallel
    #pragma omp single
    {
        for (size_t kk = 0; kk < nt; ++kk) {
            const size_t k0 = kk * bs;
            const size_t kb = std::min(bs, n - k0);

            #pragma omp task depend(inout: a[k0 * n + k0]) \
                firstprivate(k0, kb) shared(failIndex)
            {
                size_t seenFail;
                #pragma omp atomic read
                seenFail = failIndex;
                if (seenFail == n) {
                    const size_t r = potrfTile(a + k0 * n + k0, n, kb);
                    if (r < kb) {
                        #pragma omp critical(cholesky_fail)
                        failIndex = std::min(failIndex, k0 + r);
                    }
                }
            }

            for (size_t ii = kk + 1; ii < nt; ++ii) {
                const size_t i0 = ii * bs;
                const size_t ib = std::min(bs, n - i0);

                #pragma omp task depend(in: a[k0 * n + k0]) \
                    depend(inout: a[i0 * n + k0]) \
                    firstprivate(k0, kb, i0, ib) shared(failIndex)
                {
                    size_t seenFail;
                    #pragma omp atomic read
                    seenFail = failIndex;
                    if (seenFail == n) {
                        trsmTile(a + k0 * n + k0, a + i0 * n + k0, n, ib, kb);
                    }
                }
            }

            for (size_t ii = kk + 1; ii < nt; ++ii) {
                const size_t i0 = ii * bs;
                const size_t ib = std::min(bs, n - i0);

                for (size_t jj = kk + 1; jj <= ii; ++jj) {
                    const size_t j0 = jj * bs;
                    const size_t jb = std::min(bs, n - j0);

                    #pragma omp task depend(in: a[i0 * n + k0], a[j0 * n + k0]) \
                        depend(inout: a[i0 * n + j0]) \
                        firstprivate(k0, kb, i0, ib, j0, jb, ii, jj) \
                        shared(failIndex)
                    {
                        size_t seenFail;
                        #pragma omp atomic read
                        seenFail = failIndex;
                        if (seenFail == n) {
                            gemmTile(a + i0 * n + j0, a + i0 * n + k0,
                                     a + j0 * n + k0, n, ib, jb, kb, ii == jj);
                        }
                    }
                }
            }
        }
    }

    if (failIndex < n) {
        // Matrix is not positive definite
        printf("Error: Matrix is not positive definite at diagonal element %zu\n",
               failIndex);
        return false;
    }

    // Zero out upper triangular part
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        std::fill(a + i * n + i + 1, a + i * n + n, 0.0);
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

    // Generate random matrix B (sequential to keep the deterministic stream)
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

    #pragma omp parallel for reduction(max:maxError, relError)
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
    // Passive waiting avoids severe spin-wait contention in the task
    // scheduler on high-core-count machines. The OpenMP runtime reads the
    // environment before main runs, so re-exec once with the variable set;
    // a user-provided setting is left untouched.
    if (getenv("OMP_WAIT_POLICY") == nullptr) {
        setenv("OMP_WAIT_POLICY", "passive", 1);
        execv("/proc/self/exe", argv);
        // If re-exec fails, continue with the default wait policy.
    }

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
    printf("OpenMP threads: %d\n", omp_get_max_threads());

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
