#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// ============================================================================
// MPI Parallel Cholesky Decomposition (Left-looking, column-by-column)
//
// Data distribution: 1D row-block distribution
//   - Matrix A (n x n, row-major) is replicated on all processes
//   - Process p owns rows [local_row_start, local_row_end)
//   - Work is distributed: each process computes off-diagonal elements for
//     its owned rows, then column j is synchronized via Allgatherv
//
// Algorithm: left-looking column-by-column Cholesky
//   For each column j = 0..n-1:
//     1. Process owning row j computes diagonal L(j,j), broadcasts to all
//     2. Each process computes L(i,j) for its owned rows i > j
//     3. Synchronize column j via MPI_Allgatherv so all processes have it
// ============================================================================

// Generate a symmetric positive definite matrix
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    // Method: Create A = B * B^T where B is random, then add diagonal dominance
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
        A[i * n + i] += n;
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

// MPI parallel Cholesky decomposition
// A is stored in row-major order, replicated on all processes
// Each process computes off-diagonal elements for its owned rows
bool choleskyDecompositionMPI(std::vector<double>& A, const size_t n,
                               const int rank, const int num_procs) {
    // Compute row distribution
    const size_t rows_per_proc = n / num_procs;
    const size_t remainder = n % num_procs;
    const size_t local_row_start = rank * rows_per_proc + std::min((size_t)rank, remainder);
    const size_t local_row_end = (rank + 1) * rows_per_proc + std::min((size_t)(rank + 1), remainder);

    // Precompute receive counts and displacements for Allgatherv
    std::vector<int> recv_counts(num_procs);
    std::vector<int> recv_displs(num_procs);
    for (int p = 0; p < num_procs; ++p) {
        size_t s = (size_t)p * rows_per_proc + std::min((size_t)p, remainder);
        size_t e = (size_t)(p + 1) * rows_per_proc + std::min((size_t)(p + 1), remainder);
        recv_counts[p] = static_cast<int>(e - s);
        recv_displs[p] = static_cast<int>(s);
    }

    const int send_count = static_cast<int>(local_row_end - local_row_start);
    std::vector<double> local_col(send_count);
    std::vector<double> full_col(n);

    // Helper: find which process owns a given row
    auto get_row_rank = [rows_per_proc, remainder, num_procs](size_t row) -> int {
        for (int p = 0; p < num_procs; ++p) {
            size_t s = (size_t)p * rows_per_proc + std::min((size_t)p, remainder);
            size_t e = (size_t)(p + 1) * rows_per_proc + std::min((size_t)(p + 1), remainder);
            if (row >= s && row < e) return p;
        }
        return 0;
    };

    // Left-looking column-by-column Cholesky
    for (size_t j = 0; j < n; ++j) {
        // Determine which process owns row j
        const int diag_rank = get_row_rank(j);

        // ---- Step 1: Compute diagonal element L(j,j) ----
        // Only the process that owns row j computes the diagonal
        double lj = 0.0;
        if (rank == diag_rank) {
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k) {
                sum += A[j * n + k] * A[j * n + k];
            }
            const double diag_val = A[j * n + j] - sum;
            if (diag_val <= 0.0) {
                printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
                return false;
            }
            lj = sqrt(diag_val);
        }
        // Broadcast L(j,j) to all processes
        MPI_Bcast(&lj, 1, MPI_DOUBLE, diag_rank, MPI_COMM_WORLD);
        A[j * n + j] = lj;

        // ---- Step 2: Compute off-diagonal elements L(i,j) for owned rows i > j ----
        for (size_t i = local_row_start; i < local_row_end; ++i) {
            if (i <= j) {
                // Row at or above diagonal: copy current value
                local_col[i - local_row_start] = A[i * n + j];
                continue;
            }
            // Compute L(i,j) = (A(i,j) - sum_{k<j} L(i,k)*L(j,k)) / L(j,j)
            double s = 0.0;
            for (size_t k = 0; k < j; ++k) {
                s += A[i * n + k] * A[j * n + k];
            }
            const double val = (A[i * n + j] - s) / lj;
            local_col[i - local_row_start] = val;
            A[i * n + j] = val;
        }

        // ---- Step 3: Synchronize column j across all processes ----
        MPI_Allgatherv(local_col.data(), send_count, MPI_DOUBLE,
                       full_col.data(), recv_counts.data(), recv_displs.data(),
                       MPI_DOUBLE, MPI_COMM_WORLD);

        // Update column j from gathered values
        for (size_t i = 0; i < n; ++i) {
            A[i * n + j] = full_col[i];
        }
    }

    // Zero out upper triangular part (all processes since they have full matrix)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            A[i * n + j] = 0.0;
        }
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

    // Initialize MPI
    int num_procs, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("MPI processes: %d\n", num_procs);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Allocate matrix (each process holds the full matrix)
    std::vector<double> A(n * n);

    // Generate positive definite matrix on rank 0
    if (rank == 0) {
        printf("Generating positive definite matrix...\n");
        generatePositiveDefiniteMatrix(A, n);
    }

    // Broadcast the matrix to all processes
    MPI_Bcast(A.data(), n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Save original for validation on rank 0
    std::vector<double> A_orig;
    if (validate && rank == 0) {
        A_orig = A;
    }

    // Perform Cholesky decomposition
    if (rank == 0) {
        printf("Computing Cholesky decomposition...\n");
    }

    // Synchronize before timing
    MPI_Barrier(MPI_COMM_WORLD);

    auto start = std::chrono::high_resolution_clock::now();

    bool success = choleskyDecompositionMPI(A, n, rank, num_procs);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Synchronize after timing
    MPI_Barrier(MPI_COMM_WORLD);

    // Check success across all processes
    int success_all = success ? 1 : 0;
    int success_all_gathered = 0;
    MPI_Allreduce(&success_all, &success_all_gathered, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    success = (success_all_gathered == 1);

    if (!success) {
        if (rank == 0) {
            printf("Cholesky decomposition failed\n");
        }
        MPI_Finalize();
        return 1;
    }

    // Calculate performance metrics (on rank 0)
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate GFLOPS (approximately n^3/3 operations for Cholesky)
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Print results for external validation (rank 0 only)
    if (printResults && rank == 0) {
        print_results(A, "CholeskyL");
    }

    // Validation (rank 0 only)
    if (validate && rank == 0) {
        printf("Validating result...\n");
        bool valid = validateCholesky(A, A_orig, n);

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
