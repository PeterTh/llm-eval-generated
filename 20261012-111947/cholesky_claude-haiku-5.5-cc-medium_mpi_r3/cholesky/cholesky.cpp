#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// Distributed Cholesky decomposition (MPI, row-cyclic distribution)
// Decomposes positive definite matrix A into L * L^T where L is lower triangular
//
// Row i is owned by rank i % P. The recurrence is the same left-looking one as
// the sequential algorithm, and every element is accumulated in the same order,
// so the result does not depend on the number of ranks.
//
// Step j:
//   1. The owner of row j finishes the diagonal element L(j,j).
//   2. Row j (entries 0..j) is broadcast to all ranks.
//   3. Each rank updates its own rows i > j: L(i,j) = (A(i,j) - sum_k L(i,k)*L(j,k)) / L(j,j).
// Every rank therefore ends up holding the full lower-triangular factor.

// Every accumulation is an explicit fused multiply-add in index order. Pinning
// the rounding this way keeps results independent of how the compiler
// vectorizes or contracts each loop, and therefore of the number of ranks.
// Computes out[t] = sum_k a[k] * b_t[k] for t = 0..3. The four independent
// chains only hide FP latency; each individual sum is unchanged.
static inline void dot4(const double* a, const double* b0, const double* b1, const double* b2, const double* b3,
                        size_t len, double* out) {
    double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
    for (size_t k = 0; k < len; ++k) {
        const double x = a[k];
        s0 = fma(x, b0[k], s0);
        s1 = fma(x, b1[k], s1);
        s2 = fma(x, b2[k], s2);
        s3 = fma(x, b3[k], s3);
    }
    out[0] = s0;
    out[1] = s1;
    out[2] = s2;
    out[3] = s3;
}

// Computes sum_k a[k] * b[k] in index order
static inline double dot1(const double* a, const double* b, size_t len) {
    double sum = 0.0;
    for (size_t k = 0; k < len; ++k) {
        sum = fma(a[k], b[k], sum);
    }
    return sum;
}

// out[j] = row a times row j of M (M is n x n, row-major), for all j in [0, n)
static void rowTimesMatrixT(const double* a, const double* M, const size_t n, double* out) {
    size_t j = 0;
    for (; j + 4 <= n; j += 4) {
        dot4(a, M + j * n, M + (j + 1) * n, M + (j + 2) * n, M + (j + 3) * n, n, out + j);
    }
    for (; j < n; ++j) {
        out[j] = dot1(a, M + j * n, n);
    }
}

bool choleskyDecomposition(std::vector<double>& A, const size_t n, const size_t rank, const size_t nprocs) {
    // A is stored in row-major order; only rows owned by this rank (and rows
    // received through the broadcasts) are meaningful during the computation
    for (size_t j = 0; j < n; ++j) {
        const size_t owner = j % nprocs;
        double* rowJ = &A[j * n];

        if (rank == owner) {
            // Diagonal element
            const double sum = dot1(rowJ, rowJ, j);
            const double val = rowJ[j] - sum;
            // Keep a non-positive value unchanged so that every rank can detect the failure
            rowJ[j] = (val <= 0.0) ? val : sqrt(val);
        }

        MPI_Bcast(rowJ, static_cast<int>(j + 1), MPI_DOUBLE, static_cast<int>(owner), MPI_COMM_WORLD);

        if (rowJ[j] <= 0.0) {
            // Matrix is not positive definite (all ranks take this branch together)
            if (rank == 0) {
                printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
            }
            return false;
        }

        // Zero out upper triangular part of row j
        std::fill(rowJ + j + 1, rowJ + n, 0.0);

        // Off-diagonal elements of the owned rows below the diagonal
        const double diag = rowJ[j];
        const size_t first = (j + 1) + (rank + nprocs - (j + 1) % nprocs) % nprocs;
        size_t i = first;
        while (i < n) {
            if (i + 3 * nprocs < n) {
                double sums[4];
                dot4(rowJ, &A[i * n], &A[(i + nprocs) * n], &A[(i + 2 * nprocs) * n], &A[(i + 3 * nprocs) * n], j,
                     sums);
                for (size_t t = 0; t < 4; ++t) {
                    double& elem = A[(i + t * nprocs) * n + j];
                    elem = (elem - sums[t]) / diag;
                }
                i += 4 * nprocs;
            } else {
                double& elem = A[i * n + j];
                elem = (elem - dot1(rowJ, &A[i * n], j)) / diag;
                i += nprocs;
            }
        }
    }

    return true;
}

// Generate a symmetric positive definite matrix
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n, const size_t rank, const size_t nprocs) {
    // Method: Create A = B * B^T where B is random
    // This guarantees A is positive semi-definite
    // Then add identity to make it strictly positive definite

    std::vector<double> B(n * n);
    unsigned int seed = 42;

    // Generate random matrix B (identical on every rank)
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

    // Compute the locally owned rows of A = B * B^T
    for (size_t i = rank; i < n; i += nprocs) {
        rowTimesMatrixT(&B[i * n], B.data(), n, &A[i * n]);
        A[i * n + i] += n;
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n,
                      const size_t rank, const size_t nprocs) {
    // Validate by computing L * L^T and comparing with original matrix
    // Each rank checks its owned rows; L is complete on every rank

    double maxError = 0.0;
    double relError = 0.0;
    std::vector<double> reconstructed(n);

    for (size_t i = rank; i < n; i += nprocs) {
        rowTimesMatrixT(&L[i * n], L.data(), n, reconstructed.data());

        // Compare with original
        for (size_t j = 0; j < n; ++j) {
            const double error = fabs(reconstructed[j] - A_orig[i * n + j]);
            maxError = std::max(maxError, error);

            const double rel = error / (fabs(A_orig[i * n + j]) + 1e-10);
            relError = std::max(relError, rel);
        }
    }

    double errors[2] = {maxError, relError};
    MPI_Allreduce(MPI_IN_PLACE, errors, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    maxError = errors[0];
    relError = errors[1];

    if (rank == 0) {
        printf("Max absolute error: %.10e\n", maxError);
        printf("Max relative error: %.10e\n", relError);
    }

    // Check if error is within tolerance
    if (relError > 1e-6) {
        if (rank == 0) {
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

int runBenchmark(int argc, char** argv, const size_t rank, const size_t nprocs) {
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
            if (rank == 0) {
                printUsage(argv[0]);
            }
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Allocate matrix
    std::vector<double> A(n * n);
    std::vector<double> A_orig;

    // Generate positive definite matrix
    if (rank == 0) {
        printf("Generating positive definite matrix...\n");
    }
    generatePositiveDefiniteMatrix(A, n, rank, nprocs);

    if (validate) {
        A_orig = A; // Save original for validation
    }

    // Perform Cholesky decomposition
    if (rank == 0) {
        printf("Computing Cholesky decomposition...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    bool success = choleskyDecomposition(A, n, rank, nprocs);

    double elapsed = MPI_Wtime() - start;
    MPI_Allreduce(MPI_IN_PLACE, &elapsed, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    if (!success) {
        if (rank == 0) {
            printf("Cholesky decomposition failed\n");
        }
        return 1;
    }

    const long durationMs = static_cast<long>(elapsed * 1000.0);
    if (rank == 0) {
        printf("Computation time: %ld ms\n", durationMs);

        // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (durationMs / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        // Print results for external validation
        if (printResults) {
            print_results(A, "CholeskyL");
        }
    }

    // Validation
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
        }
        bool valid = validateCholesky(A, A_orig, n, rank, nprocs);

        if (rank == 0) {
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
        return valid ? 0 : 1;
    }

    return 0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    const int status = runBenchmark(argc, argv, static_cast<size_t>(rank), static_cast<size_t>(nprocs));

    MPI_Finalize();
    return status;
}
