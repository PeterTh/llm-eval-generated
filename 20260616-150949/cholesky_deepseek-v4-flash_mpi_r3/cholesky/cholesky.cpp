#include <mpi.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// ---------------------------------------------------------------------------
// 1D column-block distribution helpers
// ---------------------------------------------------------------------------

// Compute the column range [col_start, col_end) owned by a given rank
// using a balanced block distribution (remainder spread across first ranks)
static void getColumnRange(const int rank, const int num_procs,
                           const size_t n,
                           size_t& col_start, size_t& col_end) {
    const size_t cols_per_proc = n / static_cast<size_t>(num_procs);
    const size_t rem = n % static_cast<size_t>(num_procs);
    if (static_cast<size_t>(rank) < rem) {
        col_start = static_cast<size_t>(rank) * (cols_per_proc + 1);
        col_end   = col_start + cols_per_proc + 1;
    } else {
        col_start = rem * (cols_per_proc + 1) +
                    (static_cast<size_t>(rank) - rem) * cols_per_proc;
        col_end   = col_start + cols_per_proc;
    }
}

// Return the rank that owns column c
static int getColumnOwner(const size_t c, const size_t n, int num_procs) {
    const size_t cols_per_proc = n / static_cast<size_t>(num_procs);
    const size_t rem = n % static_cast<size_t>(num_procs);
    const size_t first_uneven = rem * (cols_per_proc + 1);
    if (c < first_uneven) {
        return static_cast<int>(c / (cols_per_proc + 1));
    }
    const size_t adjusted = c - first_uneven;
    return static_cast<int>(rem + adjusted / cols_per_proc);
}

// ---------------------------------------------------------------------------
// 1D column-block right-looking Cholesky decomposition (MPI)
// ---------------------------------------------------------------------------
// Input / Output : local_A  – column-major storage of locally owned columns
//                             local_A[local_col * n + row] = A[row][col]
//                             On entry A holds the original matrix (lower part).
//                             On exit  L overwrites the lower triangle.
// n              : global matrix dimension
// col_start/end  : range of global columns owned by this rank
// ---------------------------------------------------------------------------
static bool choleskyDecomposition_MPI(std::vector<double>& local_A,
                                       const size_t n,
                                       const size_t col_start,
                                       const size_t col_end,
                                       const int    rank,
                                       const int    num_procs) {
    const size_t local_cols = col_end - col_start;

    // Precompute column → owner mapping for fast lookup
    std::vector<int> col_owner(static_cast<size_t>(n));
    for (size_t c = 0; c < n; ++c)
        col_owner[c] = getColumnOwner(c, n, num_procs);

    // Buffer reused for every MPI_Bcast of a column
    std::vector<double> buf(static_cast<size_t>(n));

    for (size_t j = 0; j < n; ++j) {
        const int owner = col_owner[j];

        // ---- step 1 : factor column j (owner only) ------------------------
        if (rank == owner) {
            const size_t local_j = j - col_start;
            double* __restrict__ col = &local_A[local_j * n];

            // Diagonal element : A[j][j] = sqrt(A[j][j])
            if (col[j] <= 0.0) {
                if (rank == 0)
                    printf("Error: Matrix not positive definite at column %zu\n", j);
                return false;
            }
            col[j] = std::sqrt(col[j]);

            // Sub-diagonal elements : A[i][j] /= A[j][j]   (i > j)
            const double inv_diag = 1.0 / col[j];
            for (size_t i = j + 1; i < n; ++i)
                col[i] *= inv_diag;

            // Copy column j into the broadcast buffer
            std::memcpy(buf.data(), col + j, (n - j) * sizeof(double));
        }

        // ---- step 2 : broadcast column j -----------------------------------
        MPI_Bcast(buf.data(), static_cast<int>(n - j), MPI_DOUBLE,
                  owner, MPI_COMM_WORLD);

        // ---- step 3 : rank-1 trailing update on local columns --------------
        //   For every local column k > j :
        //     A[i][k] -= L[i][j] * L[k][j]   for i = k … n-1
        for (size_t lc = 0; lc < local_cols; ++lc) {
            const size_t k = col_start + lc;
            if (k <= j) continue;

            const double L_kj = buf[k - j];   // L[k][j]
            double* __restrict__ col_k = &local_A[lc * n];

            for (size_t i = k; i < n; ++i)
                col_k[i] -= buf[i - j] * L_kj;
        }
    }

    return true;
}

// ---------------------------------------------------------------------------
// Generate local columns of a symmetric positive definite matrix  A = B*B^T
// (each rank reproduces the same random B by using the same seed).
// ---------------------------------------------------------------------------
static void generateLocalColumns(std::vector<double>& local_A,
                                  const size_t n,
                                  const size_t col_start,
                                  const size_t col_end) {
    const size_t local_cols = col_end - col_start;

    // Generate random B (same on every rank – same seed)
    std::vector<double> B(static_cast<size_t>(n * n));
    unsigned int seed = 42;
    for (size_t i = 0; i < static_cast<size_t>(n * n); ++i)
        B[i] = (static_cast<double>(rand_r(&seed)) / static_cast<double>(RAND_MAX)) - 0.5;

    // Compute local columns of A = B * B^T
    for (size_t lc = 0; lc < local_cols; ++lc) {
        const size_t col = col_start + lc;
        double* col_ptr = &local_A[lc * n];
        for (size_t i = 0; i < n; ++i) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k)
                sum += B[i * n + k] * B[col * n + k];
            col_ptr[i] = sum;
        }
        // Add diagonal dominance
        col_ptr[col] += static_cast<double>(n);
    }
}

// ---------------------------------------------------------------------------
// Validation (sequential – called only on rank 0)
// ---------------------------------------------------------------------------
static bool validateCholesky(const std::vector<double>& L,
                              const std::vector<double>& A_orig,
                              const size_t n) {
    std::vector<double> recon(static_cast<size_t>(n * n), 0.0);

    // L * L^T
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            for (size_t k = 0; k < n; ++k)
                recon[i * n + j] += L[i * n + k] * L[j * n + k];

    double maxError = 0.0, relError = 0.0;
    for (size_t i = 0; i < static_cast<size_t>(n * n); ++i) {
        double err = std::fabs(recon[i] - A_orig[i]);
        maxError = std::max(maxError, err);
        relError = std::max(relError, err / (std::fabs(A_orig[i]) + 1e-10));
    }

    printf("Max absolute error: %.10e\n", maxError);
    printf("Max relative error: %.10e\n", relError);

    if (relError > 1e-6) {
        printf("Validation failed: relative error too large\n");
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Gather local column blocks into a full row-major matrix on rank 0
// ---------------------------------------------------------------------------
static void gatherL(std::vector<double>& full,
                     const std::vector<double>& local_A,
                     const size_t n,
                     int rank, int num_procs) {
    // Prepare displacements / counts for MPI_Gatherv
    std::vector<int> recv_counts(static_cast<size_t>(num_procs));
    std::vector<int> displs(static_cast<size_t>(num_procs));

    size_t total = 0;
    for (int r = 0; r < num_procs; ++r) {
        size_t cs, ce;
        getColumnRange(r, num_procs, n, cs, ce);
        recv_counts[r] = static_cast<int>((ce - cs) * n);
        displs[r]      = static_cast<int>(total);
        total += static_cast<size_t>(recv_counts[r]);
    }

    std::vector<double> recvbuf;
    if (rank == 0) recvbuf.resize(total);

    MPI_Gatherv(local_A.data(), static_cast<int>(local_A.size()), MPI_DOUBLE,
                rank == 0 ? recvbuf.data() : nullptr,
                recv_counts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // Unpack column-major blocks into row-major full matrix on rank 0
    if (rank == 0) {
        full.assign(static_cast<size_t>(n * n), 0.0);
        for (int r = 0; r < num_procs; ++r) {
            size_t cs, ce;
            getColumnRange(r, num_procs, n, cs, ce);
            size_t offset = static_cast<size_t>(displs[r]);
            for (size_t c = cs; c < ce; ++c) {
                for (size_t i = 0; i < n; ++i)
                    full[i * n + c] = recvbuf[offset + (c - cs) * n + i];
            }
        }
        // Zero out upper triangle for compatibility with the sequential output
        for (size_t i = 0; i < n; ++i)
            for (size_t j = i + 1; j < n; ++j)
                full[i * n + j] = 0.0;
    }
}

// ---------------------------------------------------------------------------
// Usage
// ---------------------------------------------------------------------------
static void printUsage(const char* progName) {
    printf("Usage: mpirun -np <procs> %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, num_procs = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // ---- parse command line arguments (all ranks) -------------------------
    size_t n = 512;
    bool validate     = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = static_cast<size_t>(std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else if (rank == 0) {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            MPI_Finalize();
            return 1;
        }
    }

    size_t col_start = 0, col_end = 0;
    getColumnRange(rank, num_procs, n, col_start, col_end);
    const size_t local_cols = col_end - col_start;
    const size_t local_size = (local_cols > 0) ? local_cols * n : 1;

    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark (MPI)\n");
        printf("Matrix size:  %zu x %zu\n", n, n);
        printf("MPI processes: %d\n", num_procs);
        printf("Validation:   %s\n", validate ? "enabled" : "disabled");
        printf("Generating positive definite matrix...\n");
    }

    // ---- generate matrix (distributed generation, no communication) -------
    std::vector<double> local_A(local_size);
    generateLocalColumns(local_A, n, col_start, col_end);

    // Save a copy on rank 0 for validation
    std::vector<double> A_orig;
    if (rank == 0 && validate) {
        std::vector<double> full_A;
        gatherL(full_A, local_A, n, rank, num_procs);
        A_orig = full_A;
    }

    if (rank == 0)
        printf("Computing Cholesky decomposition...\n");

    // ---- parallel Cholesky decomposition ----------------------------------
    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    bool success = choleskyDecomposition_MPI(
        local_A, n, col_start, col_end, rank, num_procs);

    double t_end = MPI_Wtime();
    double elapsed = t_end - t_start;

    if (!success) {
        if (rank == 0) printf("Cholesky decomposition failed\n");
        MPI_Finalize();
        return 1;
    }

    // ---- gather results on rank 0 -----------------------------------------
    std::vector<double> L_full;
    if (rank == 0 || validate || printResults)
        gatherL(L_full, local_A, n, rank, num_procs);

    if (rank == 0) {
        printf("Computation time: %.0f ms\n", elapsed * 1e3);

        // GFLOPS ≈ n³ / 3  (one flop per multiply-add ≈ 2 flops, times n³/6 …)
        // More precisely, Cholesky requires ≈ n³/3 floating-point operations.
        double ops = static_cast<double>(n) * static_cast<double>(n) *
                     static_cast<double>(n) / 3.0;
        double gflops = ops / elapsed / 1.0e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults)
            print_results(L_full, "CholeskyL");
    }

    // ---- validation on rank 0 ---------------------------------------------
    if (validate && rank == 0) {
        printf("Validating result...\n");
        bool valid = validateCholesky(L_full, A_orig, n);
        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            MPI_Finalize();
            return 1;
        }
    }

    MPI_Finalize();
    return 0;
}
