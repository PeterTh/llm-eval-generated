#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// Parallel Cholesky decomposition using MPI
// 1D row-block distribution with left-looking algorithm
// Decomposes positive definite matrix A into L * L^T where L is lower triangular
//
// Distribution: matrix rows are distributed in blocks across MPI processes.
//   Process p owns rows [rowStart_p, rowEnd_p).
// Algorithm: left-looking panel Cholesky.
//   For each column j:
//     1. Compute column j of L (diagonal + off-diagonal elements)
//        - Each process computes its locally-owned portion
//     2. Allgather column j so every process has the complete column
//     3. Rank-1 update: A[i,k] -= L[i,j]*L[k,j] for i,k > j
//        - Each process updates its locally-owned rows

bool choleskyDecomposition(std::vector<double>& A, const size_t n, int rank, int num_procs) {
    // Compute row-block distribution
    size_t baseRows = n / num_procs;
    size_t extraRows = n % num_procs;
    size_t rowStart = baseRows * rank + std::min(static_cast<size_t>(rank), extraRows);
    size_t rowEnd = baseRows * (rank + 1) + std::min(static_cast<size_t>(rank + 1), extraRows);
    size_t localRows = rowEnd - rowStart;

    // Temporary buffer for the full column j (all processes need complete column)
    std::vector<double> colJ(n);

    for (size_t j = 0; j < n; ++j) {
        // --- Step 1: Compute column j locally ---
        // Each process computes elements A[i*n+j] for its owned rows i
        for (size_t i = rowStart; i < rowEnd; ++i) {
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k) {
                sum += A[i * n + k] * A[j * n + k];
            }

            if (i == j) {
                // Diagonal element
                double val = A[j * n + j] - sum;
                if (val <= 0.0) {
                    if (rank == 0) {
                        printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
                    }
                    return false;
                }
                A[j * n + j] = sqrt(val);
            } else if (i > j) {
                // Off-diagonal element
                A[i * n + j] = (A[i * n + j] - sum) / A[j * n + j];
            }
        }

        // --- Step 2: Allgather column j ---
        // Each process contributes its owned rows' values for column j
        // We need to gather values for all rows [0, n), but only rows [j, n) are meaningful
        // for this column. However, we gather all for simplicity.

        // Compute recvcounts and displacements for Allgatherv
        std::vector<int> recvcounts(num_procs);
        std::vector<int> displs(num_procs);
        size_t offset = 0;
        for (int p = 0; p < num_procs; ++p) {
            size_t pBaseRows = n / num_procs;
            size_t pExtraRows = n % num_procs;
            size_t pRowStart = pBaseRows * p + std::min(static_cast<size_t>(p), pExtraRows);
            size_t pRowEnd = pBaseRows * (p + 1) + std::min(static_cast<size_t>(p + 1), pExtraRows);
            size_t pLocalRows = pRowEnd - pRowStart;
            recvcounts[p] = static_cast<int>(pLocalRows);
            displs[p] = static_cast<int>(offset);
            offset += pLocalRows;
        }

        // Determine send buffer: the portion of column j owned by this process
        const double* sendbuf = nullptr;
        int sendcount = 0;
        if (localRows > 0) {
            // We need to send values A[i*n+j] for i in [rowStart, rowEnd)
            // But these are not contiguous in memory (stride = n)
            // So we copy to a temporary contiguous buffer
            static std::vector<double> sendbuf_local;
            sendbuf_local.assign(localRows, 0.0);
            for (size_t i = 0; i < localRows; ++i) {
                sendbuf_local[i] = A[(rowStart + i) * n + j];
            }
            sendbuf = sendbuf_local.data();
            sendcount = static_cast<int>(localRows);
        }

        MPI_Allgatherv(
            sendbuf, sendcount, MPI_DOUBLE,
            colJ.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
            MPI_COMM_WORLD
        );

        // Scatter received column values back to the correct positions in A
        for (size_t i = rowStart; i < rowEnd; ++i) {
            A[i * n + j] = colJ[i];
        }

        // --- Step 3: Rank-1 update of trailing submatrix ---
        // A[i,k] -= L[i,j] * L[k,j]  for i in [rowStart, rowEnd), k in [j+1, n)
        for (size_t i = rowStart; i < rowEnd; ++i) {
            double li_j = A[i * n + j];
            for (size_t k = j + 1; k < n; ++k) {
                A[i * n + k] -= li_j * colJ[k];
            }
        }
    }

    // Zero out upper triangular part for locally-owned rows
    for (size_t i = rowStart; i < rowEnd; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            A[i * n + j] = 0.0;
        }
    }

    return true;
}

int main(int argc, char** argv) {
    // Initialize MPI
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    size_t n = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (all processes parse, but only rank 0 prints)
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = static_cast<size_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) {
                printf("Usage: %s [options]\n", argv[0]);
                printf("Options:\n");
                printf("  -n <num>     Matrix size (default: 512)\n");
                printf("  -v           Enable validation\n");
                printf("  -r           Print results for external validation\n");
                printf("  -h           Show this help message\n");
            }
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    // Synchronize and validate parameters
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark (MPI)\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("MPI processes: %d\n", num_procs);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");

        if (n < static_cast<size_t>(num_procs)) {
            printf("Error: Matrix size (%zu) must be >= number of processes (%d)\n", n, num_procs);
            MPI_Finalize();
            return 1;
        }
    }

    // Compute row-block distribution
    size_t baseRows = n / num_procs;
    size_t extraRows = n % num_procs;
    size_t rowStart = baseRows * rank + std::min(static_cast<size_t>(rank), extraRows);
    size_t rowEnd = baseRows * (rank + 1) + std::min(static_cast<size_t>(rank + 1), extraRows);
    size_t localRows = rowEnd - rowStart;

    // Allocate local portion of matrix (owned rows)
    std::vector<double> A(localRows * n);
    std::vector<double> A_orig;

    // Generate positive definite matrix: A = B * B^T + n * I
    printf("Generating positive definite matrix...\n");
    fflush(stdout);

    // Each process generates its portion of random matrix B
    std::vector<double> B_local(localRows * n);
    unsigned int seed = 42 + rank * 1000;
    for (size_t i = 0; i < localRows * n; ++i) {
        B_local[i] = (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 0.5;
    }

    // Gather all B portions so every process has the full B matrix
    std::vector<double> B_full(n * n);
    std::vector<int> recvcounts(num_procs);
    std::vector<int> displs(num_procs);
    size_t offset = 0;
    for (int p = 0; p < num_procs; ++p) {
        size_t pBaseRows = n / num_procs;
        size_t pExtraRows = n % num_procs;
        size_t pLocalRows = pBaseRows + (p < static_cast<int>(pExtraRows) ? 1 : 0);
        recvcounts[p] = static_cast<int>(pLocalRows * n);
        displs[p] = static_cast<int>(offset);
        offset += pLocalRows * n;
    }

    MPI_Allgatherv(
        B_local.data(), static_cast<int>(localRows * n), MPI_DOUBLE,
        B_full.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
        MPI_COMM_WORLD
    );

    // Compute A = B * B^T locally on every process
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += B_full[i * n + k] * B_full[j * n + k];
            }
            // Only store rows we own
            if (i >= rowStart && i < rowEnd) {
                A[(i - rowStart) * n + j] = sum;
            }
        }
    }

    // Add diagonal dominance to ensure positive definiteness
    for (size_t i = rowStart; i < rowEnd; ++i) {
        A[(i - rowStart) * n + i] += static_cast<double>(n);
    }

    if (validate) {
        A_orig = A;
    }

    // Perform Cholesky decomposition
    printf("Computing Cholesky decomposition...\n");
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);

    auto start = std::chrono::high_resolution_clock::now();
    bool success = choleskyDecomposition(A, n, rank, num_procs);
    auto end = std::chrono::high_resolution_clock::now();

    MPI_Barrier(MPI_COMM_WORLD);
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (!success) {
        if (rank == 0) {
            printf("Cholesky decomposition failed\n");
        }
        MPI_Finalize();
        return 1;
    }

    // Calculate GFLOPS (approximately n^3/3 operations for Cholesky)
    double ops = static_cast<double>(n) * n * n / 3.0;
    double gflops = ops / (duration.count() / 1000.0) / 1e9;

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Print results for external validation
    if (printResults) {
        print_results(A, "CholeskyL");
    }

    // Validation: gather full L and A_orig, compute L*L^T, compare
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
        }
        fflush(stdout);

        // Gather full L matrix on every process
        std::vector<double> L_full(n * n);
        MPI_Allgatherv(
            A.data(), static_cast<int>(localRows * n), MPI_DOUBLE,
            L_full.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
            MPI_COMM_WORLD
        );

        // Gather full A_orig on every process
        std::vector<double> A_orig_full(n * n);
        MPI_Allgatherv(
            A_orig.data(), static_cast<int>(localRows * n), MPI_DOUBLE,
            A_orig_full.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
            MPI_COMM_WORLD
        );

        // Compute L * L^T locally
        std::vector<double> reconstructed(n * n);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                double sum = 0.0;
                for (size_t k = 0; k < n; ++k) {
                    sum += L_full[i * n + k] * L_full[j * n + k];
                }
                reconstructed[i * n + j] = sum;
            }
        }

        // Compare with original A
        double maxError = 0.0;
        double relError = 0.0;
        for (size_t idx = 0; idx < n * n; ++idx) {
            double error = fabs(reconstructed[idx] - A_orig_full[idx]);
            maxError = std::max(maxError, error);
            double rel = error / (fabs(A_orig_full[idx]) + 1e-10);
            relError = std::max(relError, rel);
        }

        if (rank == 0) {
            printf("Max absolute error: %.10e\n", maxError);
            printf("Max relative error: %.10e\n", relError);
        }

        bool valid = relError <= 1e-6;
        if (rank == 0) {
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }

        MPI_Finalize();
        return valid ? 0 : 1;
    }

    MPI_Finalize();
    return 0;
}
