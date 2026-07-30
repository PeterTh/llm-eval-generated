#include <mpi.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// MPI-parallel Cholesky decomposition (right-looking algorithm)
// Decomposes positive definite matrix A into L * L^T where L is lower triangular
// Uses 1D column-block distribution for parallelism

// Generate a symmetric positive definite matrix (full matrix, rank-local)
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
        A[i * n + i] += static_cast<double>(n);
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix

    std::vector<double> reconstructed(n * n);

    // Compute L * L^T
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
    MPI_Init(&argc, &argv);

    int nprocs, myrank;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    // Parse command line arguments (all ranks parse identically)
    size_t n = 512;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (myrank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (myrank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (myrank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("MPI processes: %d\n", nprocs);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const int n_int = static_cast<int>(n);

    // 1D column-block distribution: divide columns among ranks
    const int blk = (n_int + nprocs - 1) / nprocs;
    const int start_col = myrank * blk;
    const int end_col = std::min(start_col + blk, n_int);
    const int local_ncols = end_col - start_col;

    // Allocate local storage: row-major, n rows x local_ncols columns
    std::vector<double> A_local;
    if (local_ncols > 0) {
        A_local.resize(n_int * local_ncols, 0.0);
    }

    // === Generate positive definite matrix (distributed) ===
    // Each rank generates the same B matrix with the same seed, then
    // computes A = B * B^T only for its local columns
    std::vector<double> B(n_int * n_int);
    unsigned int seed = 42;
    for (int i = 0; i < n_int * n_int; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

    // Compute A[:, local_cols] = B * B^T for local columns only
    for (int lc = 0; lc < local_ncols; ++lc) {
        const int j_global = start_col + lc;
        for (int i = 0; i < n_int; ++i) {
            double sum = 0.0;
            for (int k = 0; k < n_int; ++k) {
                sum += B[i * n_int + k] * B[j_global * n_int + k];
            }
            A_local[i * local_ncols + lc] = sum;
        }
    }

    // Add diagonal dominance
    for (int lc = 0; lc < local_ncols; ++lc) {
        const int j_global = start_col + lc;
        A_local[j_global * local_ncols + lc] += static_cast<double>(n_int);
    }

    // Save original columns for validation
    std::vector<double> A_orig_local;
    if (validate) {
        A_orig_local = A_local;
    }

    // Free B to reduce peak memory
    B.clear();
    B.shrink_to_fit();

    // === Parallel Cholesky factorization (right-looking) ===
    // Right-looking algorithm with 1D column-block distribution:
    // for j = 0..n-1:
    //   1. Owner of column j computes L[j][j] = sqrt(A[j][j]),
    //      then L[i][j] = A[i][j] / L[j][j] for i > j
    //   2. Broadcast column j (elements j..n-1) from owner to all ranks
    //   3. Each rank updates its local columns k > j using the rank-1 update:
    //      A[i][k] -= L[i][j] * L[k][j] for i = k..n-1

    if (myrank == 0) {
        printf("Computing Cholesky decomposition...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double start_time = MPI_Wtime();

    // Broadcast buffer for the current factor column
    std::vector<double> col_buf(n_int, 0.0);

    for (int j = 0; j < n_int; ++j) {
        const int owner = j / blk;  // rank that owns column j

        if (myrank == owner) {
            const int lc = j - start_col;

            // Diagonal factor: L[j][j] = sqrt(A[j][j])
            A_local[j * local_ncols + lc] = sqrt(A_local[j * local_ncols + lc]);
            const double diag_val = A_local[j * local_ncols + lc];

            // Scale below diagonal: L[i][j] = A[i][j] / L[j][j]
            for (int i = j + 1; i < n_int; ++i) {
                A_local[i * local_ncols + lc] /= diag_val;
            }

            // Copy factor column j (elements j..n-1) to broadcast buffer
            for (int i = j; i < n_int; ++i) {
                col_buf[i] = A_local[i * local_ncols + lc];
            }
        }

        // Broadcast column j (elements j..n-1) to all ranks
        MPI_Bcast(&col_buf[j], n_int - j, MPI_DOUBLE, owner, MPI_COMM_WORLD);

        // Update trailing submatrix: each rank updates its local columns k > j
        // A[i][k] -= L[i][j] * L[k][j]  for i >= k > j
        for (int lc = 0; lc < local_ncols; ++lc) {
            const int k = start_col + lc;
            if (k > j) {
                const double L_kj = col_buf[k];
                for (int i = k; i < n_int; ++i) {
                    A_local[i * local_ncols + lc] -= col_buf[i] * L_kj;
                }
            }
        }
    }

    const double end_time = MPI_Wtime();
    const double local_time = end_time - start_time;

    // Report max time across all ranks
    double max_time = 0.0;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Zero out upper triangular part in local columns
    for (int lc = 0; lc < local_ncols; ++lc) {
        const int k = start_col + lc;
        for (int i = 0; i < k; ++i) {
            A_local[i * local_ncols + lc] = 0.0;
        }
    }

    if (myrank == 0) {
        const double duration_ms = max_time * 1000.0;
        printf("Computation time: %.0f ms\n", duration_ms);

        // Calculate GFLOPS (approximately n^3/3 operations for Cholesky)
        const double ops = (double)n_int * n_int * n_int / 3.0;
        const double gflops = ops / max_time / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // === Validation ===
    if (validate) {
        // Gather the L matrix from all ranks onto rank 0 for validation
        std::vector<int> recvcounts(nprocs);
        std::vector<int> displs(nprocs);

        int total_size = 0;
        for (int p = 0; p < nprocs; ++p) {
            int p_start = p * blk;
            int p_end = std::min(p_start + blk, n_int);
            int p_ncols = std::max(0, p_end - p_start);
            recvcounts[p] = p_ncols * n_int;
            displs[p] = total_size;
            total_size += recvcounts[p];
        }

        std::vector<double> gathered;
        if (myrank == 0) {
            gathered.resize(total_size);
        }

        MPI_Gatherv(A_local.data(), local_ncols * n_int, MPI_DOUBLE,
                    myrank == 0 ? gathered.data() : nullptr,
                    recvcounts.data(), displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);

        if (myrank == 0) {
            // Reconstruct full L matrix from gathered column blocks
            std::vector<double> L_full(n_int * n_int, 0.0);
            for (int p = 0; p < nprocs; ++p) {
                int p_start = p * blk;
                int p_end = std::min(p_start + blk, n_int);
                int p_ncols = p_end - p_start;
                int disp = displs[p];
                for (int lc = 0; lc < p_ncols; ++lc) {
                    int j_global = p_start + lc;
                    for (int i = 0; i < n_int; ++i) {
                        L_full[i * n_int + j_global] = gathered[disp + i * p_ncols + lc];
                    }
                }
            }

            // Generate original A matrix for validation
            std::vector<double> A_orig(n_int * n_int);
            generatePositiveDefiniteMatrix(A_orig, n_int);

            printf("Validating result...\n");
            bool valid = validateCholesky(L_full, A_orig, n_int);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
    }

    // === Print results for external validation ===
    if (printResults) {
        // Gather L matrix to rank 0 for printing
        std::vector<int> recvcounts_r(nprocs);
        std::vector<int> displs_r(nprocs);

        int total_size_r = 0;
        for (int p = 0; p < nprocs; ++p) {
            int p_start = p * blk;
            int p_end = std::min(p_start + blk, n_int);
            int p_ncols = std::max(0, p_end - p_start);
            recvcounts_r[p] = p_ncols * n_int;
            displs_r[p] = total_size_r;
            total_size_r += recvcounts_r[p];
        }

        std::vector<double> gathered_r;
        if (myrank == 0) {
            gathered_r.resize(total_size_r);
        }

        MPI_Gatherv(A_local.data(), local_ncols * n_int, MPI_DOUBLE,
                    myrank == 0 ? gathered_r.data() : nullptr,
                    recvcounts_r.data(), displs_r.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);

        if (myrank == 0) {
            std::vector<double> L_full(n_int * n_int, 0.0);
            for (int p = 0; p < nprocs; ++p) {
                int p_start = p * blk;
                int p_end = std::min(p_start + blk, n_int);
                int p_ncols = p_end - p_start;
                int disp = displs_r[p];
                for (int lc = 0; lc < p_ncols; ++lc) {
                    int j_global = p_start + lc;
                    for (int i = 0; i < n_int; ++i) {
                        L_full[i * n_int + j_global] = gathered_r[disp + i * p_ncols + lc];
                    }
                }
            }
            print_results(L_full, "CholeskyL");
        }
    }

    MPI_Finalize();
    return 0;
}
