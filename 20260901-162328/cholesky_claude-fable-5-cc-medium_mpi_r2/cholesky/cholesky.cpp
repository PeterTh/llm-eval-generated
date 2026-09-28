#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// MPI-parallel Cholesky decomposition (distributed memory, pipelined fan-out)
// Decomposes positive definite matrix A into L * L^T where L is lower triangular
//
// Parallelization scheme:
//  - Rows are distributed cyclically: rank r owns rows i with i % nprocs == r.
//  - Every rank allocates the full n x n matrix, but only generates/updates its
//    owned rows; the prefixes of the other rows arrive via broadcast as the
//    factorization proceeds, so at the end every rank holds the complete L.
//  - Step j: after row j of L is broadcast, each rank computes element (i, j)
//    for all of its owned rows i > j. The owner of row j+1 finishes that row
//    (element and diagonal) first and starts a non-blocking broadcast of it,
//    which overlaps with the remaining updates of step j (lookahead).
//  - Each element is computed with exactly the same expression and summation
//    order as the sequential algorithm, so the result is bit-identical.

static int g_rank = 0;
static int g_nprocs = 1;

// Smallest row index >= start owned by this rank (cyclic distribution)
static inline size_t firstOwnedRow(const size_t start) {
    const size_t p = (size_t)g_nprocs;
    const size_t r = (size_t)g_rank;
    const size_t m = start % p;
    return start + (r >= m ? r - m : r + p - m);
}

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    if (n == 0) {
        return true;
    }

    const size_t p = (size_t)g_nprocs;
    MPI_Request req;

    // Finish row 0 (diagonal only) on its owner and start broadcasting it
    if (g_rank == 0) {
        const double val = A[0];
        if (val <= 0.0) {
            printf("Error: Matrix is not positive definite at diagonal element 0\n");
            A[0] = val; // non-positive value signals failure to all ranks
        } else {
            A[0] = sqrt(val);
        }
    }
    MPI_Ibcast(&A[0], 1, MPI_DOUBLE, 0, MPI_COMM_WORLD, &req);

    for (size_t j = 0; j < n; ++j) {
        MPI_Wait(&req, MPI_STATUS_IGNORE);

        const double* rowj = &A[j * n];
        const double dj = rowj[j];
        if (dj <= 0.0) {
            // Matrix is not positive definite (owner already printed the error)
            return false;
        }

        // Lookahead: the owner of row j+1 completes it and starts its broadcast
        // so the transfer overlaps with the trailing updates below
        if (j + 1 < n) {
            const size_t next = j + 1;
            if (next % p == (size_t)g_rank) {
                double* rowi = &A[next * n];

                // Off-diagonal element (next, j)
                double sum = 0.0;
                for (size_t k = 0; k < j; ++k) {
                    sum += rowi[k] * rowj[k];
                }
                rowi[j] = (rowi[j] - sum) / dj;

                // Diagonal element (next, next)
                sum = 0.0;
                for (size_t k = 0; k < next; ++k) {
                    sum += rowi[k] * rowi[k];
                }
                const double val = rowi[next] - sum;
                if (val <= 0.0) {
                    printf("Error: Matrix is not positive definite at diagonal element %zu\n", next);
                    rowi[next] = val; // non-positive value signals failure to all ranks
                } else {
                    rowi[next] = sqrt(val);
                }
            }
            MPI_Ibcast(&A[next * n], next + 1, MPI_DOUBLE, (int)(next % p), MPI_COMM_WORLD, &req);
        }

        // Compute element (i, j) for the remaining owned rows i >= j + 2
        // (row j+1 was already handled above by its owner)
        for (size_t i = firstOwnedRow(j + 2); i < n; i += p) {
            double* rowi = &A[i * n];
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k) {
                sum += rowi[k] * rowj[k];
            }
            rowi[j] = (rowi[j] - sum) / dj;
        }
    }

    // Zero out upper triangular part (every rank now holds the complete L)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            A[i * n + j] = 0.0;
        }
    }

    return true;
}

// Generate a symmetric positive definite matrix (owned rows only)
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

    // Compute A = B * B^T for the rows owned by this rank
    for (size_t i = firstOwnedRow(0); i < n; i += (size_t)g_nprocs) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += B[i * n + k] * B[j * n + k];
            }
            A[i * n + j] = sum;
        }
    }

    // Add diagonal dominance to ensure positive definiteness
    for (size_t i = firstOwnedRow(0); i < n; i += (size_t)g_nprocs) {
        A[i * n + i] += n;
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix.
    // Every rank holds the complete L; each rank reconstructs and checks its
    // owned rows (the only rows for which it holds A_orig) and the elementwise
    // maxima are combined with a max-reduction.

    double localErr[2] = {0.0, 0.0}; // {maxError, relError}

    for (size_t i = firstOwnedRow(0); i < n; i += (size_t)g_nprocs) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += L[i * n + k] * L[j * n + k];
            }

            const double error = fabs(sum - A_orig[i * n + j]);
            localErr[0] = std::max(localErr[0], error);

            const double rel = error / (fabs(A_orig[i * n + j]) + 1e-10);
            localErr[1] = std::max(localErr[1], rel);
        }
    }

    double globalErr[2] = {0.0, 0.0};
    MPI_Reduce(localErr, globalErr, 2, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    int valid = 1;
    if (g_rank == 0) {
        printf("Max absolute error: %.10e\n", globalErr[0]);
        printf("Max relative error: %.10e\n", globalErr[1]);

        // Check if error is within tolerance
        if (globalErr[1] > 1e-6) {
            printf("Validation failed: relative error too large\n");
            valid = 0;
        }
    }
    MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);

    return valid != 0;
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

    // Parse command line arguments
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

    // Allocate matrix
    std::vector<double> A(n * n);
    std::vector<double> A_orig;

    // Generate positive definite matrix
    if (g_rank == 0) {
        printf("Generating positive definite matrix...\n");
    }
    generatePositiveDefiniteMatrix(A, n);

    if (validate) {
        A_orig = A; // Save original for validation (owned rows are meaningful)
    }

    // Perform Cholesky decomposition
    if (g_rank == 0) {
        printf("Computing Cholesky decomposition...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    bool success = choleskyDecomposition(A, n);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (!success) {
        if (g_rank == 0) {
            printf("Cholesky decomposition failed\n");
        }
        MPI_Finalize();
        return 1;
    }

    if (g_rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        // Print results for external validation (every rank holds the full L)
        if (printResults) {
            print_results(A, "CholeskyL");
        }
    }

    // Validation
    if (validate) {
        if (g_rank == 0) {
            printf("Validating result...\n");
        }
        bool valid = validateCholesky(A, A_orig, n);

        if (g_rank == 0) {
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        MPI_Finalize();
        return valid ? 0 : 1;
    }

    MPI_Finalize();
    return 0;
}
