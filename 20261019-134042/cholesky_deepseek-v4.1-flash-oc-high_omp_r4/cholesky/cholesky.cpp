#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <omp.h>

#include "../common/results_output.hpp"

// Blocked right-looking Cholesky decomposition with OpenMP.
// Decomposes positive definite matrix A into L * L^T where L is lower triangular.
// The input is symmetric, so its row-major storage is also a valid
// column-major representation of the same matrix. The factor is converted
// back to row-major storage (with a zeroed upper triangle) before returning.

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    if (n == 0) {
        return true;
    }

    // Block size: 64 for small matrices, 128 for larger ones.
    const size_t nb = (n < 512) ? 64 : 128;
    const size_t max_panel = std::min(n, nb);

    // Reused panel workspace. panel[jj * m + i] holds L21(i, jj).
    std::vector<double> panel(static_cast<size_t>(n) * max_panel);

    bool positive = true;

    // A single persistent parallel region avoids repeatedly starting worker
    // threads for the many short phases of the factorization.
    #pragma omp parallel shared(A, panel, n, nb, positive)
    {
        const size_t team = static_cast<size_t>(omp_get_num_threads());

        for (size_t kb = 0; kb < n; kb += nb) {
            const size_t ib = std::min(nb, n - kb);

            // Factor the diagonal block A[kb:kb+ib, kb:kb+ib] in place.
            // The block has already been updated by all previous panels.
            #pragma omp single
            {
                for (size_t jj = 0; jj < ib; ++jj) {
                    const size_t j = kb + jj;
                    const double val = A[j + j * n];
                    if (val <= 0.0) {
                        // Matrix is not positive definite
                        printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
                        positive = false;
                        break;
                    }

                    const double d = std::sqrt(val);
                    const double inv_d = 1.0 / d;
                    A[j + j * n] = d;

                    // Scale the column below the diagonal.
                    for (size_t ii = jj + 1; ii < ib; ++ii) {
                        A[(kb + ii) + j * n] *= inv_d;
                    }

                    // Rank-1 update of the remaining lower part of the block.
                    for (size_t ll = jj + 1; ll < ib; ++ll) {
                        const size_t l = kb + ll;
                        const double fl = A[l + j * n];
                        const double* col_j = A.data() + j * n;
                        double* col_l = A.data() + l * n;
                        #pragma omp simd
                        for (size_t ii = ll; ii < ib; ++ii) {
                            const size_t i = kb + ii;
                            col_l[i] -= col_j[i] * fl;
                        }
                    }
                }
            }

            if (!positive) {
                break;
            }

            const size_t r0 = kb + ib;
            if (r0 < n) {
                const size_t m = n - r0;

                // Copy the panel below the diagonal block into a contiguous
                // column-major working buffer.
                #pragma omp for schedule(static)
                for (size_t jj = 0; jj < ib; ++jj) {
                    const double* src = A.data() + r0 + (kb + jj) * n;
                    double* dst = panel.data() + jj * m;
                    #pragma omp simd
                    for (size_t i = 0; i < m; ++i) {
                        dst[i] = src[i];
                    }
                }

                // Solve L11^T * X = A21 for the whole panel. Row blocks are
                // independent; the inner loops are vectorized over rows.
                constexpr size_t row_block = 8;
                const size_t nblocks = (m + row_block - 1) / row_block;
                #pragma omp for schedule(static)
                for (size_t b = 0; b < nblocks; ++b) {
                    const size_t i0 = b * row_block;
                    const size_t i1 = std::min(i0 + row_block, m);
                    for (size_t jj = 0; jj < ib; ++jj) {
                        double* xj = panel.data() + jj * m;
                        for (size_t ll = 0; ll < jj; ++ll) {
                            const double f = A[(kb + jj) + (kb + ll) * n];
                            const double* xl = panel.data() + ll * m;
                            #pragma omp simd
                            for (size_t i = i0; i < i1; ++i) {
                                xj[i] -= xl[i] * f;
                            }
                        }
                        const double inv_d = 1.0 / A[(kb + jj) + (kb + jj) * n];
                        #pragma omp simd
                        for (size_t i = i0; i < i1; ++i) {
                            xj[i] *= inv_d;
                        }
                    }
                }

                // Store the solved panel back into A (contiguous columns).
                // It is disjoint from the trailing update, so no barrier is
                // needed before starting the update.
                #pragma omp for schedule(static) nowait
                for (size_t jj = 0; jj < ib; ++jj) {
                    const double* src = panel.data() + jj * m;
                    double* dst = A.data() + r0 + (kb + jj) * n;
                    #pragma omp simd
                    for (size_t i = 0; i < m; ++i) {
                        dst[i] = src[i];
                    }
                }

                // Trailing update: A22 -= L21 * L21^T (lower triangle only).
                // Pick a tile size that still provides enough work for all
                // threads on small matrices, while keeping tiles cache resident.
                size_t ts = 64;
                size_t nt = (m + ts - 1) / ts;
                while (ts > 8 && (nt * (nt + 1)) / 2 < team) {
                    ts >>= 1;
                    nt = (m + ts - 1) / ts;
                }

                const size_t ntasks = (nt * (nt + 1)) / 2;
                #pragma omp for schedule(dynamic, 1)
                for (size_t task = 0; task < ntasks; ++task) {
                    // Decode the triangular task index into (ti, tj), tj <= ti.
                    size_t ti = static_cast<size_t>(
                        (std::sqrt(8.0 * static_cast<double>(task) + 1.0) - 1.0) * 0.5);
                    while ((ti + 1) * (ti + 2) / 2 <= task) {
                        ++ti;
                    }
                    while (ti * (ti + 1) / 2 > task) {
                        --ti;
                    }
                    const size_t tj = task - ti * (ti + 1) / 2;

                    const size_t i0 = ti * ts;
                    const size_t i1 = std::min(i0 + ts, m);
                    const size_t j0 = tj * ts;
                    const size_t j1 = std::min(j0 + ts, m);

                    constexpr size_t MR = 8;
                    constexpr size_t NR = 8;
                    for (size_t im = i0; im < i1; im += MR) {
                        const size_t imax = std::min(im + MR, i1);
                        const size_t jend = std::min(j1, imax);
                        for (size_t jm = j0; jm < jend; jm += NR) {
                            const size_t jmax = std::min(jm + NR, jend);
                            double acc[MR][NR] = {};
                            for (size_t kk = 0; kk < ib; ++kk) {
                                const double* bk = panel.data() + kk * m;
                                for (size_t ii = 0; ii < imax - im; ++ii) {
                                    const double a = bk[im + ii];
                                    for (size_t jj = 0; jj < jmax - jm; ++jj) {
                                        acc[ii][jj] += a * bk[jm + jj];
                                    }
                                }
                            }
                            for (size_t ii = 0; ii < imax - im; ++ii) {
                                for (size_t jj = 0; jj < jmax - jm; ++jj) {
                                    if (im + ii >= jm + jj) {
                                        A[(r0 + im + ii) + (r0 + jm + jj) * n] -= acc[ii][jj];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        if (positive) {
            // Convert the column-major lower-triangular factor to the
            // row-major layout expected by the caller, zeroing the upper
            // triangle.
            #pragma omp for schedule(static)
            for (size_t i = 1; i < n; ++i) {
                for (size_t j = 0; j < i; ++j) {
                    const double v = A[i + j * n];
                    A[i + j * n] = 0.0;
                    A[i * n + j] = v;
                }
            }
        }
    }

    return positive;
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
    
    // Compute A = B * B^T. Rows are independent, so this also distributes
    // the matrix pages across the NUMA nodes used by the worker threads.
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
    
    // Avoid oversubscribing SMT siblings, which hurts this memory-bound
    // workload on large machines. Respect a smaller user-specified limit.
    const int maxThreads = omp_get_max_threads();
    const int threads = std::min(maxThreads, 128);
    omp_set_num_threads(threads);

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
