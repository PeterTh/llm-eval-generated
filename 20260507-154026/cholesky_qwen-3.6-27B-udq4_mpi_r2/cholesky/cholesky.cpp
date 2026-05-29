#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// Compute local row range for a process with 1D block distribution
static void compute_row_range(int rank, int num_procs, size_t n,
                               size_t& start, size_t& count) {
    size_t base = n / num_procs;
    size_t remainder = n % num_procs;
    start = rank * base + std::min(static_cast<size_t>(rank), remainder);
    count = base + (rank < static_cast<int>(remainder) ? 1 : 0);
}

// Column-synchronous MPI Cholesky decomposition with 1D block row distribution.
// Each process owns a contiguous block of rows of the matrix.
// For each column j (processed sequentially):
//   1. The process owning row j computes L(j, j).
//   2. Column j of L (values L(j,0)..L(j,j)) is broadcast to all processes.
//   3. Each process computes L(i, j) for its locally-owned rows i > j.
static bool choleskyDecomposition(std::vector<double>& local_A, const size_t n,
                                   const size_t local_start, const size_t local_count,
                                   int rank, int num_procs, MPI_Comm comm) {
    std::vector<double> col_buf(n);

    for (size_t j = 0; j < n; ++j) {
        const int owner = static_cast<int>((j * num_procs) / n);

        if (rank == owner) {
            const size_t lr = j - local_start;

            // Diagonal: L(j,j) = sqrt(A(j,j) - sum_k L(j,k)^2)
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k) {
                sum += local_A[lr * n + k] * local_A[lr * n + k];
            }
            const double val = local_A[lr * n + j] - sum;
            if (val <= 0.0) {
                if (rank == 0) {
                    printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
                }
                return false;
            }
            local_A[lr * n + j] = sqrt(val);

            // Prepare broadcast buffer with L(j, 0..j)
            for (size_t k = 0; k <= j; ++k) {
                col_buf[k] = local_A[lr * n + k];
            }
        }

        // Broadcast column j of L to all processes
        MPI_Bcast(col_buf.data(), static_cast<int>(j + 1), MPI_DOUBLE, owner, comm);

        // Each process computes L(i, j) for its owned rows i > j
        for (size_t li = 0; li < local_count; ++li) {
            const size_t i = local_start + li;
            if (i <= j) continue;

            double sum = 0.0;
            for (size_t k = 0; k < j; ++k) {
                sum += local_A[li * n + k] * col_buf[k];
            }
            local_A[li * n + j] = (local_A[li * n + j] - sum) / col_buf[j];
        }
    }

    // Zero out upper triangular part for locally-owned rows
    for (size_t li = 0; li < local_count; ++li) {
        const size_t i = local_start + li;
        for (size_t j = i + 1; j < n; ++j) {
            local_A[li * n + j] = 0.0;
        }
    }

    return true;
}

// Generate a symmetric positive definite matrix A = B * B^T + n*I
static void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;

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

    for (size_t i = 0; i < n; ++i) {
        A[i * n + i] += n;
    }
}

// Validate by computing L * L^T and comparing with original matrix
static bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig,
                              const size_t n) {
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

static void printUsage(const char* progName) {
    printf("Usage: mpirun -np <procs> %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int rank, num_procs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    size_t n = 512;
    int validate = 0;
    int printResults = 0;

    // Parse command line arguments on rank 0
    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                n = static_cast<size_t>(atoi(argv[++i]));
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                MPI_Finalize();
                return 0;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                MPI_Finalize();
                return 1;
            }
        }
    }

    // Broadcast parameters to all processes
    MPI_Bcast(&n, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark (MPI)\n");
        printf("MPI processes: %d\n", num_procs);
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Compute local row range for this process
    size_t local_start, local_count;
    compute_row_range(rank, num_procs, n, local_start, local_count);

    // Generate positive definite matrix on rank 0
    std::vector<double> A_full;
    std::vector<double> A_orig;
    if (rank == 0) {
        printf("Generating positive definite matrix...\n");
        A_full.resize(n * n);
        generatePositiveDefiniteMatrix(A_full, n);
        if (validate) {
            A_orig = A_full;
        }
    }

    // Scatter rows to all processes (1D block distribution)
    std::vector<double> local_A(local_count * n);

    std::vector<int> sendcounts(num_procs);
    std::vector<int> displs(num_procs);
    if (rank == 0) {
        for (int p = 0; p < num_procs; ++p) {
            size_t ps, pc;
            compute_row_range(p, num_procs, n, ps, pc);
            sendcounts[p] = static_cast<int>(pc * n);
            displs[p] = static_cast<int>(ps * n);
        }
    }

    MPI_Scatterv(A_full.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
                 local_A.data(), static_cast<int>(local_count * n), MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    // Perform Cholesky decomposition
    if (rank == 0) {
        printf("Computing Cholesky decomposition...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);

    auto start = std::chrono::high_resolution_clock::now();

    bool success = choleskyDecomposition(local_A, n, local_start, local_count,
                                          rank, num_procs, MPI_COMM_WORLD);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (!success) {
        if (rank == 0) {
            printf("Cholesky decomposition failed\n");
        }
        MPI_Finalize();
        return 1;
    }

    // Gather results to rank 0
    std::vector<double> L_full;
    if (rank == 0) {
        L_full.resize(n * n);
    }

    MPI_Gatherv(local_A.data(), static_cast<int>(local_count * n), MPI_DOUBLE,
                L_full.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());

        double ops = static_cast<double>(n) * n * n / 3.0;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) {
            print_results(L_full, "CholeskyL");
        }

        if (validate) {
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
    }

    MPI_Finalize();
    return 0;
}
