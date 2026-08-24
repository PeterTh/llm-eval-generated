#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// Parallel right-looking Cholesky decomposition using MPI.
//
// Columns are distributed cyclically: process p owns columns p, p+P, p+2P, ...
// Store is replicated (each process has the full n×n matrix).
//
// Algorithm (right-looking):
//   For column k in 0..n-1:
//     1. Owner factors column k: L[k][k]=sqrt(A[k][k]), L[i][k]=A[i][k]/L[k][k]
//     2. Owner broadcasts factored column k to all processes.
//     3. Each process stores broadcast column k into its local A.
//     4. Each process updates its OWNED columns j>k using row-major traversal:
//          A[i][j] -= L[i][k]*L[j][k]  for i >= j > k
//
// Row-major trailing update: for each row i, iterate owned columns j<=i
// for stride-1 cache-efficient access.

bool choleskyDecomposition(std::vector<double>& A, const size_t n, MPI_Comm comm) {
    int rank, nprocs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    const size_t p = static_cast<size_t>(nprocs);
    const size_t r = static_cast<size_t>(rank);

    // Buffer for broadcasting a column slice
    std::vector<double> col_buf(n);

    for (size_t k = 0; k < n; ++k) {
        const int owner = static_cast<int>(k % p);

        // ----- Step 1: Owner factors column k --------------------------------
        if (rank == owner) {
            const double diag_val = A[k * n + k];
            if (diag_val <= 0.0) {
                printf("Error: Matrix not positive definite at diagonal %zu\n", k);
                return false;
            }
            A[k * n + k] = sqrt(diag_val);
            const double Lkk = A[k * n + k];

            for (size_t i = k + 1; i < n; ++i) {
                A[i * n + k] /= Lkk;
            }
            for (size_t j = k + 1; j < n; ++j) {
                A[k * n + j] = 0.0;
            }
            for (size_t i = k; i < n; ++i) {
                col_buf[i - k] = A[i * n + k];
            }
        }

        // ----- Step 2: Broadcast column k -----------------------------------
        MPI_Bcast(col_buf.data(), static_cast<int>(n - k), MPI_DOUBLE, owner, comm);

        // ----- Step 3: Store broadcast column locally -----------------------
        if (rank != owner) {
            for (size_t i = k; i < n; ++i) {
                A[i * n + k] = col_buf[i - k];
            }
            for (size_t j = k + 1; j < n; ++j) {
                A[k * n + j] = 0.0;
            }
        }

        // ----- Step 4: Row-major trailing update on owned columns -----------
        //   A[i][j] -= L[i][k] * L[j][k]   for i >= j > k
        // Row-major order: for each row i, iterate owned columns j<=i
        const size_t j_lo = k + 1;
        // First owned column >= j_lo
        size_t j0 = j_lo;
        {
            const size_t rem = j0 % p;
            j0 += (r + p - rem) % p;
        }
        if (j0 < n) {
            for (size_t i = j0; i < n; ++i) {
                const double Lik = col_buf[i - k];
                const size_t row_i = i * n;
                for (size_t j = j0; j <= i; j += p) {
                    A[row_i + j] -= Lik * col_buf[j - k];
                }
            }
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
    printf("Usage: mpirun -np <P> %s [options]\n", progName);
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
            n = static_cast<size_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
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
        generatePositiveDefiniteMatrix(A, n);
    }
    MPI_Bcast(A.data(), static_cast<int>(n * n), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (validate && rank == 0) {
        A_orig = A;
    }

    if (rank == 0) {
        printf("Computing Cholesky decomposition...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    bool success = choleskyDecomposition(A, n, MPI_COMM_WORLD);

    double end_time = MPI_Wtime();
    double duration_ms = (end_time - start_time) * 1000.0;
    double max_duration_ms = 0.0;
    MPI_Reduce(&duration_ms, &max_duration_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (!success) {
        if (rank == 0) printf("Cholesky decomposition failed\n");
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("Computation time: %.2f ms\n", max_duration_ms);
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (max_duration_ms / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) {
            print_results(A, "CholeskyL");
        }
    }

    if (validate) {
        int valid_int = 1;
        if (rank == 0) {
            printf("Validating result...\n");
            valid_int = validateCholesky(A, A_orig, n) ? 1 : 0;
            if (valid_int) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
        MPI_Bcast(&valid_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Finalize();
        return valid_int ? 0 : 1;
    }

    MPI_Finalize();
    return 0;
}
