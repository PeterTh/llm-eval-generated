#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <omp.h>
#ifdef __linux__
#include <sched.h>
#endif

#include "../common/results_output.hpp"

// Pin each OpenMP worker thread to its own CPU (round-robin over the CPUs the
// process is allowed to run on). Unpinned threads migrate between cores and
// sockets, which badly hurts the barrier-heavy blocked algorithm on large
// machines. Skipped when the user already requested binding via OMP_PROC_BIND /
// OMP_PLACES, or on non-Linux platforms.
static void pinOpenMPThreads() {
#ifdef __linux__
    if (getenv("OMP_PROC_BIND") != nullptr || getenv("OMP_PLACES") != nullptr) return;
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) return;
    std::vector<int> cpus;
    for (int c = 0; c < CPU_SETSIZE; ++c) {
        if (CPU_ISSET(c, &allowed)) cpus.push_back(c);
    }
    if (cpus.empty()) return;
#pragma omp parallel
    {
        const int cpu = cpus[(size_t)omp_get_thread_num() % cpus.size()];
        cpu_set_t one;
        CPU_ZERO(&one);
        CPU_SET(cpu, &one);
        sched_setaffinity(0, sizeof(one), &one);
    }
#endif
}

// Blocked right-looking Cholesky decomposition parallelized with OpenMP.
// Decomposes positive definite matrix A into L * L^T where L is lower triangular.
//
// For each column panel [jb, jb+nb):
//   1. Factor the diagonal block (small, sequential).
//   2. Solve the panel below the diagonal block (parallel over rows).
//   3. Update the trailing submatrix A[i][j] -= L[i][k] * L[j][k] (parallel over tiles).
// The trailing update works on a transposed copy of the panel so that the
// innermost loop is a contiguous, vectorizable axpy along a row of A.

static constexpr size_t CHOL_NB = 64;  // panel width
static constexpr size_t CHOL_TJ = 64;  // tile width (columns) for the trailing update

// Factor the nb x nb diagonal block starting at (jb, jb). Returns false if not PD.
static bool choleskyDiagonalBlock(double* __restrict A, const size_t n, const size_t jb,
                                  const size_t nb) {
    for (size_t jj = 0; jj < nb; ++jj) {
        const size_t j = jb + jj;
        double* __restrict rowj = A + j * n;
        double sum = 0.0;
        for (size_t k = jb; k < j; ++k) {
            sum += rowj[k] * rowj[k];
        }
        const double val = rowj[j] - sum;
        if (val <= 0.0) {
            // Matrix is not positive definite
            printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
            return false;
        }
        const double ljj = sqrt(val);
        rowj[j] = ljj;
        const double inv = 1.0 / ljj;
        for (size_t ii = jj + 1; ii < nb; ++ii) {
            const size_t i = jb + ii;
            double* __restrict rowi = A + i * n;
            double s = 0.0;
            for (size_t k = jb; k < j; ++k) {
                s += rowi[k] * rowj[k];
            }
            rowi[j] = (rowi[j] - s) * inv;
        }
    }
    return true;
}

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    // A is stored in row-major order
    double* __restrict a = A.data();
    const size_t nb = std::min(CHOL_NB, n);
    bool ok = true;

    // Avoid spawning far more threads than there are rows to share per panel.
    const int maxThreads = omp_get_max_threads();
    const int nthreads = (int)std::clamp<size_t>(n / 8, 1, (size_t)maxThreads);

    // Transposed panel buffer: T[kk][j] = L[j][jb + kk] for j in trailing rows.
    std::vector<double> Tbuf(nb * n);
    double* __restrict T = Tbuf.data();

#pragma omp parallel default(shared) num_threads(nthreads)
    {
        // Note: 'ok' is only inspected right after the barrier that follows the
        // single, so every thread sees the same value and executes the same
        // sequence of barriers.
        for (size_t jb = 0; jb < n; jb += nb) {
            const size_t cnb = std::min(nb, n - jb);
            const size_t rest = jb + cnb;  // first row/col of trailing submatrix

            // 1. Factor diagonal block
#pragma omp single
            {
                if (!choleskyDiagonalBlock(a, n, jb, cnb)) ok = false;
            }
            // implicit barrier
            if (!ok) break;

            // 2. Panel solve: rows i >= rest, columns [jb, rest).
            //    Forward substitution against the (lower-triangular) diagonal block.
            //    Also stores the transposed panel into T for the trailing update.
#pragma omp for schedule(dynamic, 4)
            for (size_t i = rest; i < n; ++i) {
                double* __restrict rowi = a + i * n;
                for (size_t j = jb; j < rest; ++j) {
                    const double* __restrict rowj = a + j * n;
                    double s = 0.0;
                    for (size_t k = jb; k < j; ++k) {
                        s += rowi[k] * rowj[k];
                    }
                    const double lij = (rowi[j] - s) / rowj[j];
                    rowi[j] = lij;
                    T[(j - jb) * n + i] = lij;
                }
            }
            // implicit barrier

            // 3. Trailing update: A[i][j] -= sum_k L[i][k] * L[j][k], rest <= j <= i < n.
            //    Work is distributed over (row, column-tile) pairs for load balance.
            const size_t ntiles = (n - rest + CHOL_TJ - 1) / CHOL_TJ;
            const size_t nrows = n - rest;
#pragma omp for schedule(dynamic, 8) nowait
            for (size_t w = 0; w < nrows * ntiles; ++w) {
                const size_t ri = w / ntiles;          // row offset
                const size_t ct = w % ntiles;          // column tile
                const size_t i = n - 1 - ri;            // longest rows first
                const size_t j0 = rest + ct * CHOL_TJ;
                if (j0 > i) continue;
                const size_t j1 = std::min(j0 + CHOL_TJ, i + 1);
                double* __restrict rowi = a + i * n;
                const double* __restrict lrow = rowi + jb;
                size_t kk = 0;
                // Four panel columns per sweep: each A element is loaded/stored
                // once per four multiply-adds instead of once per one.
                for (; kk + 4 <= cnb; kk += 4) {
                    const double l0 = lrow[kk], l1 = lrow[kk + 1];
                    const double l2 = lrow[kk + 2], l3 = lrow[kk + 3];
                    const double* __restrict t0 = T + (kk + 0) * n;
                    const double* __restrict t1 = T + (kk + 1) * n;
                    const double* __restrict t2 = T + (kk + 2) * n;
                    const double* __restrict t3 = T + (kk + 3) * n;
#pragma omp simd
                    for (size_t j = j0; j < j1; ++j) {
                        rowi[j] -= l0 * t0[j] + l1 * t1[j] + l2 * t2[j] + l3 * t3[j];
                    }
                }
                for (; kk < cnb; ++kk) {
                    const double lik = lrow[kk];
                    const double* __restrict trow = T + kk * n;
#pragma omp simd
                    for (size_t j = j0; j < j1; ++j) {
                        rowi[j] -= lik * trow[j];
                    }
                }
            }
#pragma omp barrier
        }

        // Zero out upper triangular part
        if (ok) {
#pragma omp for schedule(static)
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = i + 1; j < n; ++j) {
                    a[i * n + j] = 0.0;
                }
            }
        }
    }

    return ok;
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
#pragma omp parallel for schedule(dynamic, 8)
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
    
    // Compute L * L^T (L is lower triangular, so k only runs to min(i, j))
#pragma omp parallel for schedule(dynamic, 8)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            const size_t kmax = std::min(i, j) + 1;
            for (size_t k = 0; k < kmax; ++k) {
                sum += L[i * n + k] * L[j * n + k];
            }
            reconstructed[i * n + j] = sum;
        }
    }
    
    // Compare with original
    double maxError = 0.0;
    double relError = 0.0;
    
#pragma omp parallel for reduction(max : maxError, relError)
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
    
    pinOpenMPThreads();

    printf("Cholesky Decomposition Benchmark\n");
    printf("OpenMP threads: %d\n", omp_get_max_threads());
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
