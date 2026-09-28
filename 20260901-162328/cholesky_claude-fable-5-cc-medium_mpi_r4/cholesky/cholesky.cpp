#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// MPI-parallel Cholesky decomposition (distributed memory)
// Decomposes positive definite matrix A into L * L^T where L is lower triangular
//
// Parallelization: rows are distributed cyclically across ranks (row i is owned
// by rank i % P). Every rank allocates the full n x n matrix, but only its own
// rows hold data (the rest stay zero). At step k the owner finalizes the
// diagonal element of row k and broadcasts the row; all ranks then compute
// column k of their local rows below k (left-looking update). At the end the
// full factor is assembled by summing the disjoint row sets across ranks.

static int g_rank = 0;
static int g_nprocs = 1;

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    // A is stored in row-major order; this rank only touches rows it owns
    std::vector<double> rowk(n);

    for (size_t k = 0; k < n; ++k) {
        const int owner = (int)(k % (size_t)g_nprocs);

        if (g_rank == owner) {
            // Finalize diagonal element of row k (columns < k are already done)
            double sum = 0.0;
            const double* Ak = &A[k * n];
            for (size_t m = 0; m < k; ++m) {
                sum += Ak[m] * Ak[m];
            }
            const double val = Ak[k] - sum;
            if (val <= 0.0) {
                // Matrix is not positive definite; signal via non-positive pivot
                A[k * n + k] = -1.0;
            } else {
                A[k * n + k] = sqrt(val);
            }
            memcpy(rowk.data(), Ak, (k + 1) * sizeof(double));
        }

        MPI_Bcast(rowk.data(), (int)(k + 1), MPI_DOUBLE, owner, MPI_COMM_WORLD);

        if (rowk[k] <= 0.0) {
            if (g_rank == 0) {
                printf("Error: Matrix is not positive definite at diagonal element %zu\n", k);
            }
            return false;
        }

        // Compute column k of all local rows below k
        const double invDiag = 1.0 / rowk[k];
        for (size_t i = k + 1 + (size_t)((g_rank - (int)((k + 1) % (size_t)g_nprocs) + g_nprocs) % g_nprocs);
             i < n; i += (size_t)g_nprocs) {
            double* Ai = &A[i * n];
            double sum = 0.0;
            for (size_t m = 0; m < k; ++m) {
                sum += Ai[m] * rowk[m];
            }
            Ai[k] = (Ai[k] - sum) * invDiag;
        }
    }

    // Zero out upper triangular part of local rows
    for (size_t i = (size_t)g_rank; i < n; i += (size_t)g_nprocs) {
        for (size_t j = i + 1; j < n; ++j) {
            A[i * n + j] = 0.0;
        }
    }

    return true;
}

// Generate a symmetric positive definite matrix (local rows only)
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    // Method: Create A = B * B^T where B is random
    // This guarantees A is positive semi-definite
    // Then add identity to make it strictly positive definite

    std::vector<double> B(n * n);
    unsigned int seed = 42;

    // Generate random matrix B (identical on every rank)
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

    // Compute A = B * B^T for the rows this rank owns
    for (size_t i = (size_t)g_rank; i < n; i += (size_t)g_nprocs) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += B[i * n + k] * B[j * n + k];
            }
            A[i * n + j] = sum;
        }
    }

    // Add diagonal dominance to ensure positive definiteness
    for (size_t i = (size_t)g_rank; i < n; i += (size_t)g_nprocs) {
        A[i * n + i] += n;
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix.
    // L and A_orig hold the full matrices on every rank; each rank checks
    // the rows it owns and the maximum errors are reduced across ranks.

    double maxError = 0.0;
    double relError = 0.0;

    std::vector<double> row(n);
    for (size_t i = (size_t)g_rank; i < n; i += (size_t)g_nprocs) {
        // Compute row i of L * L^T
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += L[i * n + k] * L[j * n + k];
            }
            row[j] = sum;
        }

        // Compare with original
        for (size_t j = 0; j < n; ++j) {
            const double error = fabs(row[j] - A_orig[i * n + j]);
            maxError = std::max(maxError, error);

            const double rel = error / (fabs(A_orig[i * n + j]) + 1e-10);
            relError = std::max(relError, rel);
        }
    }

    double localErr[2] = {maxError, relError};
    double globalErr[2];
    MPI_Allreduce(localErr, globalErr, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    if (g_rank == 0) {
        printf("Max absolute error: %.10e\n", globalErr[0]);
        printf("Max relative error: %.10e\n", globalErr[1]);
    }

    // Check if error is within tolerance
    if (globalErr[1] > 1e-6) {
        if (g_rank == 0) {
            printf("Validation failed: relative error too large\n");
        }
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
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_nprocs);

    size_t n = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (identical on every rank)
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (g_rank == 0) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            if (g_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (g_rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI processes: %d\n", g_nprocs);
    }

    // Allocate matrix (full size on every rank; only owned rows are non-zero)
    std::vector<double> A(n * n, 0.0);
    std::vector<double> A_orig;

    // Generate positive definite matrix
    if (g_rank == 0) {
        printf("Generating positive definite matrix...\n");
    }
    generatePositiveDefiniteMatrix(A, n);

    if (validate) {
        // Assemble the full original matrix on every rank for validation
        A_orig.assign(n * n, 0.0);
        MPI_Allreduce(A.data(), A_orig.data(), (int)(n * n), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    }

    // Perform Cholesky decomposition
    if (g_rank == 0) {
        printf("Computing Cholesky decomposition...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double tStart = MPI_Wtime();

    bool success = choleskyDecomposition(A, n);

    MPI_Barrier(MPI_COMM_WORLD);
    const double tEnd = MPI_Wtime();
    const long durationMs = (long)((tEnd - tStart) * 1000.0);

    if (!success) {
        if (g_rank == 0) {
            printf("Cholesky decomposition failed\n");
        }
        MPI_Finalize();
        return 1;
    }

    if (g_rank == 0) {
        printf("Computation time: %ld ms\n", durationMs);

        // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (durationMs / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Assemble the full factor on every rank if needed for output/validation
    std::vector<double> L;
    if (printResults || validate) {
        L.assign(n * n, 0.0);
        MPI_Allreduce(A.data(), L.data(), (int)(n * n), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    }

    // Print results for external validation
    if (printResults && g_rank == 0) {
        print_results(L, "CholeskyL");
    }

    // Validation
    if (validate) {
        if (g_rank == 0) {
            printf("Validating result...\n");
        }
        bool valid = validateCholesky(L, A_orig, n);

        if (g_rank == 0) {
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        MPI_Finalize();
        return valid ? 0 : 1;
    }

    MPI_Finalize();
    return 0;
}
