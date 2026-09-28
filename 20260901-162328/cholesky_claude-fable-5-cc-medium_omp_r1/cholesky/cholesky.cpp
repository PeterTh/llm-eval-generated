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

// Blocked Cholesky decomposition parallelized with OpenMP tasks.
// Decomposes positive definite matrix A into L * L^T where L is lower triangular.
// Right-looking algorithm on square blocks: factorize the diagonal block (POTRF),
// solve the panel below it (TRSM), then rank-k update the trailing submatrix
// (SYRK/GEMM). Task dependencies on block anchors expose the full DAG parallelism.

static constexpr size_t BS_MAX = 128;

// Unblocked Cholesky of the jb x jb diagonal block starting at (j0, j0).
// Returns the global index of the first non-positive-definite diagonal element,
// or SIZE_MAX on success.
static size_t potrfBlock(double* A, const size_t n, const size_t j0, const size_t jb) {
    for (size_t j = j0; j < j0 + jb; ++j) {
        double sum = 0.0;
        for (size_t k = j0; k < j; ++k) {
            sum += A[j * n + k] * A[j * n + k];
        }
        const double val = A[j * n + j] - sum;
        if (val <= 0.0) {
            return j;
        }
        const double dj = sqrt(val);
        A[j * n + j] = dj;
        const double inv = 1.0 / dj;
        for (size_t i = j + 1; i < j0 + jb; ++i) {
            double s = 0.0;
            for (size_t k = j0; k < j; ++k) {
                s += A[i * n + k] * A[j * n + k];
            }
            A[i * n + j] = (A[i * n + j] - s) * inv;
        }
    }
    return SIZE_MAX;
}

// Solve X * L11^T = A21 in place for the ib x jb panel block at (i0, j0),
// where L11 is the already-factorized diagonal block at (j0, j0).
static void trsmBlock(double* A, const size_t n,
                      const size_t i0, const size_t ib,
                      const size_t j0, const size_t jb) {
    for (size_t i = i0; i < i0 + ib; ++i) {
        double* Ai = &A[i * n];
        for (size_t j = j0; j < j0 + jb; ++j) {
            const double* Aj = &A[j * n];
            double sum = 0.0;
            for (size_t k = j0; k < j; ++k) {
                sum += Ai[k] * Aj[k];
            }
            Ai[j] = (Ai[j] - sum) / Aj[j];
        }
    }
}

// Trailing update: C(i0,j0) -= A(i0,k0) * A(j0,k0)^T for an ib x jb block.
// When i0 == j0 (diagonal block) only the lower triangle is updated,
// matching what the unblocked algorithm ever reads or writes there.
static void gemmBlock(double* A, const size_t n,
                      const size_t i0, const size_t ib,
                      const size_t j0, const size_t jb,
                      const size_t k0, const size_t kb) {
    // Transpose the B panel block into a contiguous buffer so the innermost
    // loop is a stride-1 SAXPY over j (vectorizes without FP reassociation).
    double Bt[BS_MAX * BS_MAX];
    for (size_t j = 0; j < jb; ++j) {
        const double* Bj = &A[(j0 + j) * n + k0];
        for (size_t k = 0; k < kb; ++k) {
            Bt[k * jb + j] = Bj[k];
        }
    }
    const bool diag = (i0 == j0);

    // Register-blocked micro-kernel: accumulate an MR x NR tile of C in
    // registers across the whole k range, touching memory for C only once.
    constexpr size_t MR = 4;
    constexpr size_t NR = 8;
    const size_t iMain = diag ? 0 : (ib / MR) * MR; // diagonal blocks use the generic path
    const size_t jMain = (jb / NR) * NR;

    for (size_t i = 0; i < iMain; i += MR) {
        const double* Arows[MR];
        for (size_t ii = 0; ii < MR; ++ii) {
            Arows[ii] = &A[(i0 + i + ii) * n + k0];
        }
        for (size_t j = 0; j < jMain; j += NR) {
            double acc[MR][NR];
            for (size_t ii = 0; ii < MR; ++ii) {
                for (size_t jj = 0; jj < NR; ++jj) {
                    acc[ii][jj] = A[(i0 + i + ii) * n + j0 + j + jj];
                }
            }
            for (size_t k = 0; k < kb; ++k) {
                const double* btk = &Bt[k * jb + j];
                for (size_t ii = 0; ii < MR; ++ii) {
                    const double a = Arows[ii][k];
                    for (size_t jj = 0; jj < NR; ++jj) {
                        acc[ii][jj] -= a * btk[jj];
                    }
                }
            }
            for (size_t ii = 0; ii < MR; ++ii) {
                for (size_t jj = 0; jj < NR; ++jj) {
                    A[(i0 + i + ii) * n + j0 + j + jj] = acc[ii][jj];
                }
            }
        }
        // Column remainder of the MR-row band.
        for (size_t ii = 0; ii < MR; ++ii) {
            double* Ci = &A[(i0 + i + ii) * n + j0];
            const double* Ai = Arows[ii];
            for (size_t k = 0; k < kb; ++k) {
                const double a = Ai[k];
                const double* btk = &Bt[k * jb];
                for (size_t j = jMain; j < jb; ++j) {
                    Ci[j] -= a * btk[j];
                }
            }
        }
    }

    // Generic path: remaining rows, and entire diagonal blocks (lower triangle only).
    for (size_t i = iMain; i < ib; ++i) {
        double* Ci = &A[(i0 + i) * n + j0];
        const double* Ai = &A[(i0 + i) * n + k0];
        const size_t jend = diag ? (i + 1) : jb;
        for (size_t k = 0; k < kb; ++k) {
            const double a = Ai[k];
            const double* btk = &Bt[k * jb];
            for (size_t j = 0; j < jend; ++j) {
                Ci[j] -= a * btk[j];
            }
        }
    }
}

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    double* a = A.data();

    // Block size: large enough for cache-efficient kernels, small enough to
    // expose plenty of tasks to the available threads.
    const size_t bs = std::clamp<size_t>(((n / 32 + 7) / 8) * 8, 32, BS_MAX);
    const size_t nb = (n + bs - 1) / bs; // number of block rows/columns

    // Global index of the first non-positive-definite pivot (SIZE_MAX if none).
    size_t badIndex = SIZE_MAX;

    auto bdim = [&](size_t b) { return std::min(bs, n - b * bs); };

    #pragma omp parallel shared(badIndex)
    {
        for (size_t k = 0; k < nb; ++k) {
            const size_t k0 = k * bs;
            const size_t kb = bdim(k);

            // Factorize the diagonal block on one thread.
            #pragma omp single
            badIndex = potrfBlock(a, n, k0, kb);
            // (implicit barrier: all threads see badIndex and the factored block)

            if (badIndex != SIZE_MAX) {
                break; // uniform across threads
            }

            // Solve the panel below the diagonal block.
            #pragma omp for schedule(dynamic)
            for (size_t i = k + 1; i < nb; ++i) {
                trsmBlock(a, n, i * bs, bdim(i), k0, kb);
            }
            // (implicit barrier: panel done before trailing update reads it)

            // Rank-kb update of the trailing submatrix. Flatten the triangular
            // set of blocks {(i, j) : k < j <= i < nb} into a 1-D iteration
            // space so a single worksharing loop covers all of them.
            const size_t rows = nb - (k + 1);
            const size_t nBlocks = rows * (rows + 1) / 2;
            #pragma omp for schedule(dynamic)
            for (size_t t = 0; t < nBlocks; ++t) {
                // Invert t = ri*(ri+1)/2 + cj with 0 <= cj <= ri.
                size_t ri = (size_t)((sqrt(8.0 * (double)t + 1.0) - 1.0) / 2.0);
                while (ri * (ri + 1) / 2 > t) --ri;
                while ((ri + 1) * (ri + 2) / 2 <= t) ++ri;
                const size_t cj = t - ri * (ri + 1) / 2;
                const size_t i = k + 1 + ri;
                const size_t j = k + 1 + cj;
                gemmBlock(a, n, i * bs, bdim(i), j * bs, bdim(j), k0, kb);
            }
            // (implicit barrier: block (k+1, k+1) fully updated before potrf(k+1))
        }
    }

    const size_t bad = badIndex;
    if (bad != SIZE_MAX) {
        // Matrix is not positive definite
        printf("Error: Matrix is not positive definite at diagonal element %zu\n", bad);
        return false;
    }

    // Zero out upper triangular part
    #pragma omp parallel for schedule(static)
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

// SMT/hyperthread siblings only contend for the FP units in this workload.
// Unless the user pinned a thread count via OMP_NUM_THREADS, default to one
// OpenMP thread per physical core.
static void configureThreads() {
    if (getenv("OMP_NUM_THREADS") != nullptr) {
        return;
    }
    int threadsPerCore = 1;
    FILE* f = fopen("/sys/devices/system/cpu/cpu0/topology/thread_siblings_list", "r");
    if (f) {
        char buf[256];
        if (fgets(buf, sizeof(buf), f)) {
            // Formats like "0,128" or "0-1": count of sibling CPUs listed.
            int count = 0;
            for (char* tok = strtok(buf, ","); tok; tok = strtok(nullptr, ",")) {
                int lo, hi;
                if (sscanf(tok, "%d-%d", &lo, &hi) == 2) {
                    count += hi - lo + 1;
                } else {
                    count += 1;
                }
            }
            if (count > 1) {
                threadsPerCore = count;
            }
        }
        fclose(f);
    }
    const int physCores = std::max(1, omp_get_num_procs() / threadsPerCore);
    omp_set_num_threads(physCores);
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
    
    configureThreads();

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
