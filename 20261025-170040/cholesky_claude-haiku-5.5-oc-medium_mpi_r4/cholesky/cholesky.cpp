#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// Distributed-memory Cholesky decomposition (MPI).
//
// Decomposes positive definite matrix A into L * L^T where L is lower triangular.
// The matrix is stored in row-major order.
//
// Parallelization:
//  - Rows are distributed cyclically: row i is owned by rank (i % size).
//  - Every rank keeps a full copy of the matrix buffer, so the final factor L is
//    available everywhere (needed for output and validation).
//  - Row i is final once its owner has computed it. The owner then broadcasts the
//    lower part (i+1 entries) with MPI_Ibcast. Other ranks wait for a row only when
//    they first need it, so rows are factored in a pipelined wavefront.
//  - Each element is computed with exactly the same sequence of floating-point
//    operations as the sequential algorithm, so results are bit-identical for any
//    number of ranks.

namespace {

inline int ownerOf(const size_t row, const int size) {
    return static_cast<int>(row % static_cast<size_t>(size));
}

// Number of rows owned by `rank` in an n-row matrix.
inline size_t numOwnedRows(const size_t n, const int rank, const int size) {
    const size_t r = static_cast<size_t>(rank);
    if (r >= n) {
        return 0;
    }
    return (n - r - 1) / static_cast<size_t>(size) + 1;
}

// Generate a symmetric positive definite matrix. Every rank regenerates the same
// random matrix B (seed 42) and computes only the rows it owns.
// If fullRows is false, only the lower triangle (j <= i) of owned rows is filled,
// which is all the factorization reads.
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n, const int rank, const int size,
                                    const bool fullRows) {
    // Method: Create A = B * B^T where B is random
    // This guarantees A is positive semi-definite
    // Then add identity to make it strictly positive definite

    std::vector<double> B(n * n);
    unsigned int seed = 42;

    // Generate random matrix B
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

    // Compute owned rows of A = B * B^T
    for (size_t i = static_cast<size_t>(rank); i < n; i += static_cast<size_t>(size)) {
        const size_t jEnd = fullRows ? n : i + 1;
        for (size_t j = 0; j < jEnd; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += B[i * n + k] * B[j * n + k];
            }
            A[i * n + j] = sum;
        }
    }

    // Add diagonal dominance to ensure positive definiteness
    for (size_t i = static_cast<size_t>(rank); i < n; i += static_cast<size_t>(size)) {
        A[i * n + i] += n;
    }
}

// Collective. Factorizes A in place (row-major, lower triangular result, upper part zeroed).
// Every rank must hold the full buffer A on entry (owned rows filled, others unused).
// Returns the smallest row index whose pivot was not positive, or n on success.
size_t choleskyDistributed(std::vector<double>& A, const size_t n, const int rank, const int size) {
    std::vector<MPI_Request> reqs(n, MPI_REQUEST_NULL);
    std::vector<char> ready(n, 0);  // row is available locally (owned or already received)
    size_t failedRow = n;
    bool failed = false;

    for (size_t i = 0; i < n; ++i) {
        const int owner = ownerOf(i, size);

        if (owner == rank) {
            ready[i] = 1;
            if (!failed) {
                double* Ai = &A[i * n];
                for (size_t j = 0; j <= i; ++j) {
                    if (j < i && !ready[j]) {
                        // Row j is needed for this column: wait for its broadcast.
                        MPI_Wait(&reqs[j], MPI_STATUS_IGNORE);
                        ready[j] = 1;
                    }
                    if ((j & 31) == 0) {
                        // Let MPI advance pending broadcasts while computing
                        int flag = 0;
                        MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
                    }

                    const double* Aj = &A[j * n];
                    double sum = 0.0;

                    if (i == j) {
                        // Diagonal element
                        for (size_t k = 0; k < j; ++k) {
                            sum += Ai[k] * Ai[k];
                        }
                        const double val = Ai[j] - sum;
                        if (val <= 0.0) {
                            // Matrix is not positive definite
                            failedRow = i;
                            failed = true;
                            break;
                        }
                        Ai[j] = sqrt(val);
                    } else {
                        // Off-diagonal element
                        for (size_t k = 0; k < j; ++k) {
                            sum += Ai[k] * Aj[k];
                        }
                        Ai[j] = (Ai[j] - sum) / Aj[j];
                    }
                }

                // Zero out upper triangular part
                for (size_t j = i + 1; j < n; ++j) {
                    Ai[j] = 0.0;
                }
            }

            // Publish the finished row (lower part, i+1 entries) to all ranks.
            MPI_Ibcast(&A[i * n], static_cast<int>(i + 1), MPI_DOUBLE, owner, MPI_COMM_WORLD, &reqs[i]);
        } else {
            // Post the receive for a row owned by another rank. The data is
            // written directly into this rank's copy of the matrix.
            MPI_Ibcast(&A[i * n], static_cast<int>(i + 1), MPI_DOUBLE, owner, MPI_COMM_WORLD, &reqs[i]);
        }
    }

    // All broadcasts must complete before the factor is used or the buffers are released.
    MPI_Waitall(static_cast<int>(n), reqs.data(), MPI_STATUSES_IGNORE);

    // Upper part of rows received from other ranks was never written and stays zero.

    // Agree on the first failing row (identical on all ranks).
    long long localFail = failed ? static_cast<long long>(failedRow) : static_cast<long long>(n);
    long long globalFail = localFail;
    MPI_Allreduce(&localFail, &globalFail, 1, MPI_LONG_LONG, MPI_MIN, MPI_COMM_WORLD);
    return static_cast<size_t>(globalFail);
}

// Collective. Validate by computing L * L^T and comparing with the original matrix.
// A_origOwned holds the original rows owned by this rank (in increasing row order).
bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_origOwned, const size_t n,
                      const int rank, const int size) {
    double maxError = 0.0;
    double relError = 0.0;

    // Each rank checks its own rows of L * L^T
    size_t local = 0;
    for (size_t i = static_cast<size_t>(rank); i < n; i += static_cast<size_t>(size), ++local) {
        const double* Li = &L[i * n];
        const double* Aorig = &A_origOwned[local * n];
        for (size_t j = 0; j < n; ++j) {
            const double* Lj = &L[j * n];
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += Li[k] * Lj[k];
            }

            const double error = fabs(sum - Aorig[j]);
            maxError = std::max(maxError, error);

            const double rel = error / (fabs(Aorig[j]) + 1e-10);
            relError = std::max(relError, rel);
        }
    }

    double errs[2] = {maxError, relError};
    MPI_Allreduce(MPI_IN_PLACE, errs, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    maxError = errs[0];
    relError = errs[1];

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

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const bool root = (rank == 0);

    size_t n = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (identical on every rank)
    int status = -1;  // -1: continue running; otherwise exit with this code
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (root) {
                printUsage(argv[0]);
            }
            status = 0;
            break;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            status = 1;
            break;
        }
    }

    if (status >= 0) {
        MPI_Finalize();
        return status;
    }

    if (root) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Allocate matrix (full copy on every rank)
    std::vector<double> A(n * n);
    std::vector<double> A_origOwned;

    // Generate positive definite matrix (owned rows only)
    if (root) {
        printf("Generating positive definite matrix...\n");
    }
    generatePositiveDefiniteMatrix(A, n, rank, size, validate);

    if (validate) {
        // Save this rank's original rows for validation
        const size_t numOwned = numOwnedRows(n, rank, size);
        A_origOwned.resize(numOwned * n);
        size_t local = 0;
        for (size_t i = static_cast<size_t>(rank); i < n; i += static_cast<size_t>(size), ++local) {
            std::copy_n(&A[i * n], n, &A_origOwned[local * n]);
        }
    }

    // Perform Cholesky decomposition
    if (root) {
        printf("Computing Cholesky decomposition...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    const size_t failedRow = choleskyDistributed(A, n, rank, size);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (failedRow < n) {
        if (root) {
            printf("Error: Matrix is not positive definite at diagonal element %zu\n", failedRow);
            printf("Cholesky decomposition failed\n");
        }
        MPI_Finalize();
        return 1;
    }

    if (root) {
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        // Print results for external validation
        if (printResults) {
            print_results(A, "CholeskyL");
        }
    }

    int exitCode = 0;

    // Validation
    if (validate) {
        if (root) {
            printf("Validating result...\n");
        }
        bool valid = validateCholesky(A, A_origOwned, n, rank, size);

        if (valid) {
            if (root) {
                printf("Validation: PASSED\n");
            }
        } else {
            if (root) {
                printf("Validation: FAILED\n");
            }
            exitCode = 1;
        }
    }

    MPI_Finalize();
    return exitCode;
}
