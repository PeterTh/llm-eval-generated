#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// MPI-parallel Cholesky decomposition using right-looking blocked algorithm.
// The matrix is fully replicated on all ranks. Work is distributed by rows
// for the trailing matrix update. Communication is minimized by batching.
//
// For each column block [jb, je):
//   1. Owner rank factorizes the panel
//   2. Panel is broadcast to all ranks
//   3. Trailing submatrix update is distributed across ranks by rows
//   4. Updated rows are shared via a single Alltoallv per block

static const size_t BLOCK_SIZE = 64;

// Determine which rank owns column j (for panel factorization)
static int ownerOf(int nprocs, size_t n, size_t j) {
    size_t block = n / nprocs;
    size_t rem = n % nprocs;
    for (int r = 0; r < nprocs; ++r) {
        size_t start = r * block + std::min((size_t)r, rem);
        size_t end = start + block + (r < (int)rem ? 1 : 0);
        if (j >= start && j < end) return r;
    }
    return nprocs - 1;
}

// Factorize a panel of columns [col_start, col_end)
static bool factorPanel(double* A, size_t n, size_t col_start, size_t col_end) {
    for (size_t j = col_start; j < col_end; ++j) {
        double sum = 0.0;
        for (size_t k = col_start; k < j; ++k) {
            sum += A[j * n + k] * A[j * n + k];
        }
        const double val = A[j * n + j] - sum;
        if (val <= 0.0) {
            return false;
        }
        A[j * n + j] = sqrt(val);
        const double invLjj = 1.0 / A[j * n + j];

        for (size_t i = j + 1; i < n; ++i) {
            double s = 0.0;
            for (size_t k = col_start; k < j; ++k) {
                s += A[i * n + k] * A[j * n + k];
            }
            A[i * n + j] = (A[i * n + j] - s) * invLjj;
        }
    }
    return true;
}

// Distribute rows [start, end) across np procs; return range for rank r
static void getRowRange(size_t start, size_t end, int r, int np, size_t& rs, size_t& re) {
    size_t total = end - start;
    size_t per_rank = total / np;
    size_t rem = total % np;
    rs = start + r * per_rank + std::min((size_t)r, rem);
    re = rs + per_rank + (r < (int)rem ? 1 : 0);
}

bool choleskyDecomposition(std::vector<double>& A, const size_t n, MPI_Comm comm) {
    int rank, nprocs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    for (size_t jb = 0; jb < n; jb += BLOCK_SIZE) {
        size_t je = std::min(jb + BLOCK_SIZE, n);
        int panel_owner = ownerOf(nprocs, n, jb);

        // Step 1: Owner factorizes the panel
        if (rank == panel_owner) {
            if (!factorPanel(A.data(), n, jb, je)) {
                printf("Error: Matrix is not positive definite at column %zu\n", jb);
                int fail = 1;
                MPI_Bcast(&fail, 1, MPI_INT, panel_owner, comm);
                return false;
            }
        }

        int fail = 0;
        MPI_Bcast(&fail, 1, MPI_INT, panel_owner, comm);
        if (fail) return false;

        // Step 2: Broadcast the factored panel to all ranks
        // Pack: columns [jb, je), all rows [0, n)
        size_t panel_cols = je - jb;
        size_t panel_size = panel_cols * n;
        std::vector<double> panel_buf(panel_size);

        if (rank == panel_owner) {
            for (size_t j = jb; j < je; ++j) {
                for (size_t i = 0; i < n; ++i) {
                    panel_buf[(j - jb) * n + i] = A[i * n + j];
                }
            }
        }

        MPI_Bcast(panel_buf.data(), panel_size, MPI_DOUBLE, panel_owner, comm);

        if (rank != panel_owner) {
            for (size_t j = jb; j < je; ++j) {
                for (size_t i = 0; i < n; ++i) {
                    A[i * n + j] = panel_buf[(j - jb) * n + i];
                }
            }
        }

        if (je >= n) continue;

        // Step 3: Trailing matrix update distributed by rows
        // Each rank updates its assigned rows of the trailing submatrix
        size_t row_start, row_end;
        getRowRange(je, n, rank, nprocs, row_start, row_end);

        // Update: for each column j in [je, n), for each row i in [max(j, row_start), row_end)
        //   A[i,j] -= sum_{k=jb}^{je-1} A[i,k] * A[j,k]
        //
        // Store the updated rows in a local buffer for efficient communication
        // We need to share updated values. Each rank computed updates for rows [row_start, row_end)
        // across all trailing columns [je, n). But for column j, only rows [j, n) are valid.

        size_t num_rows = row_end - row_start;
        size_t num_cols = n - je;
        if (num_rows == 0 || num_cols == 0) continue;

        // Compute updates and store in a buffer (row_start..row_end) x (je..n)
        // buf[local_row * num_cols + local_col] = A[(row_start + local_row) * n + (je + local_col)]
        std::vector<double> update_buf(num_rows * num_cols);

        for (size_t jc = 0; jc < num_cols; ++jc) {
            size_t j = je + jc;
            for (size_t ir = 0; ir < num_rows; ++ir) {
                size_t i = row_start + ir;
                if (i >= j) {
                    double s = 0.0;
                    for (size_t k = jb; k < je; ++k) {
                        s += A[i * n + k] * A[j * n + k];
                    }
                    update_buf[ir * num_cols + jc] = A[i * n + j] - s;
                } else {
                    update_buf[ir * num_cols + jc] = 0.0;
                }
            }
        }

        // Step 4: Share updated rows across all ranks using Allgather
        // Each rank has computed full rows (for columns je..n-1) for its row range.
        // We need all ranks to have the complete trailing submatrix.
        //
        // Pack: each rank sends its rows of the trailing submatrix
        // For rank r with rows [rs, re): send (re-rs) * num_cols doubles
        // Receive buffer: full trailing submatrix (n-je) * num_cols = num_cols * num_cols

        // Actually, we want each rank to have the complete trailing submatrix.
        // Each rank sends its computed rows. Use Allgatherv.

        std::vector<int> send_counts(nprocs);
        std::vector<int> recv_counts(nprocs);
        std::vector<int> displs(nprocs);

        for (int r = 0; r < nprocs; ++r) {
            size_t rs, re;
            getRowRange(je, n, r, nprocs, rs, re);
            recv_counts[r] = (int)((re - rs) * num_cols);
        }
        send_counts[rank] = recv_counts[rank];

        // Allgather to share counts (in case they differ due to rounding)
        MPI_Allgather(MPI_IN_PLACE, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, comm);

        displs[0] = 0;
        for (int r = 1; r < nprocs; ++r) {
            displs[r] = displs[r - 1] + recv_counts[r - 1];
        }

        int total_size = displs[nprocs - 1] + recv_counts[nprocs - 1];
        std::vector<double> full_trailing(total_size);

        MPI_Allgatherv(update_buf.data(), (int)update_buf.size(), MPI_DOUBLE,
                       full_trailing.data(), recv_counts.data(), displs.data(), MPI_DOUBLE, comm);

        // Unpack: copy the complete trailing submatrix back into A
        size_t offset = 0;
        for (int r = 0; r < nprocs; ++r) {
            size_t rs, re;
            getRowRange(je, n, r, nprocs, rs, re);
            size_t r_rows = re - rs;
            for (size_t ir = 0; ir < r_rows; ++ir) {
                size_t i = rs + ir;
                for (size_t jc = 0; jc < num_cols; ++jc) {
                    size_t j = je + jc;
                    if (i >= j) {
                        A[i * n + j] = full_trailing[offset * num_cols + jc];
                    }
                }
                offset++;
            }
        }
    }

    // Zero out upper triangular part
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            A[i * n + j] = 0.0;
        }
    }

    return true;
}

// Generate a symmetric positive definite matrix
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;

    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += B[i * n + k] * B[j * n + k];
            }
            A[i * n + j] = sum;
        }
    }

    for (size_t i = 0; i < n; ++i) {
        A[i * n + i] += n;
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    std::vector<double> reconstructed(n * n);

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += L[i * n + k] * L[j * n + k];
            }
            reconstructed[i * n + j] = sum;
        }
    }

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

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

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
            printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("MPI processes: %d\n", nprocs);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    std::vector<double> A(n * n);
    std::vector<double> A_orig;

    if (rank == 0) {
        printf("Generating positive definite matrix...\n");
    }
    generatePositiveDefiniteMatrix(A, n);

    if (validate) {
        A_orig = A;
    }

    if (rank == 0) {
        printf("Computing Cholesky decomposition...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    bool success = choleskyDecomposition(A, n, MPI_COMM_WORLD);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (!success) {
        if (rank == 0) printf("Cholesky decomposition failed\n");
        MPI_Finalize();
        return 1;
    }

    long long dur_ms = duration.count();
    long long max_dur_ms;
    MPI_Reduce(&dur_ms, &max_dur_ms, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %lld ms\n", max_dur_ms);
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (max_dur_ms / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    if (printResults) {
        print_results(A, "CholeskyL");
    }

    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        bool valid = validateCholesky(A, A_orig, n);

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
