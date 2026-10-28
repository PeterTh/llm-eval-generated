#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// Distributed-memory Cholesky decomposition (MPI, row-cyclic distribution)
// Decomposes positive definite matrix A into L * L^T where L is lower triangular
//
// Data layout: global row i is owned by rank (i % P) and stored at local row
// index (i / P) in a row-major local buffer of (localRows x n) doubles.
//
// Algorithm (right-looking over pivot rows, bit-identical to the sequential
// row-oriented algorithm because every dot product is accumulated in the same
// k = 0..j-1 order):
//   for j = 0 .. n-1:
//     owner(j) finishes row j (diagonal L[j][j]) and broadcasts L[j][0..j]
//     every rank updates its owned rows i > j: L[i][j] = (A[i][j] - dot) / L[j][j]

namespace {

int g_rank = 0;  // rank of this process in MPI_COMM_WORLD (set in main)

// Print only from rank 0 so output matches the sequential program.
template <typename... Args>
void rootPrintf(const char* fmt, Args... args) {
    if (g_rank == 0) {
        printf(fmt, args...);
    }
}

// Literal message overload (avoids passing a non-literal format string)
void rootPrintf(const char* msg) {
    if (g_rank == 0) {
        fputs(msg, stdout);
    }
}

// Number of rows owned by `rank` in a cyclic distribution of n rows over `size` ranks.
size_t localRowCount(const size_t n, const int rank, const int size) {
    const size_t r = static_cast<size_t>(rank);
    const size_t p = static_cast<size_t>(size);
    return r < n ? (n - r + p - 1) / p : 0;
}

}  // namespace

// Distributed Cholesky decomposition.
// `A` holds this rank's rows (localRowCount x n, row-major). On success every
// rank's owned rows contain the corresponding rows of L (upper part zeroed).
// Returns the same value on all ranks.
bool choleskyDecomposition(std::vector<double>& A, const size_t n, const int rank, const int size) {
    const size_t P = static_cast<size_t>(size);
    const size_t R = static_cast<size_t>(rank);
    const size_t nloc = localRowCount(n, rank, size);

    // Pivot buffer: L[j][0..j] followed by a failure flag at index j+1.
    std::vector<double> pivot(n + 1);

    for (size_t j = 0; j < n; ++j) {
        const size_t owner = j % P;
        const size_t lj = j / P;
        double failed = 0.0;

        // Owner finalizes pivot row j: diagonal element and zeroed upper part.
        if (owner == R) {
            double* rowj = &A[lj * n];
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k) {
                sum += rowj[k] * rowj[k];
            }
            const double val = rowj[j] - sum;
            if (val <= 0.0) {
                // Matrix is not positive definite
                printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
                failed = 1.0;
            } else {
                rowj[j] = sqrt(val);
                for (size_t k = j + 1; k < n; ++k) {
                    rowj[k] = 0.0;
                }
            }
            std::copy(rowj, rowj + j + 1, pivot.begin());
            pivot[j + 1] = failed;
        }

        MPI_Bcast(pivot.data(), static_cast<int>(j + 2), MPI_DOUBLE, static_cast<int>(owner), MPI_COMM_WORLD);

        if (pivot[j + 1] != 0.0) {
            return false;  // identical decision on all ranks
        }

        const double* pj = pivot.data();
        const double djj = pj[j];

        // Update owned rows i > j. Local index li corresponds to global row R + li*P.
        size_t li = (j >= R) ? (j - R) / P + 1 : 0;
        for (; li < nloc; ++li) {
            double* row = &A[li * n];
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k) {
                sum += row[k] * pj[k];
            }
            row[j] = (row[j] - sum) / djj;
        }
    }

    return true;
}

// Generate a symmetric positive definite matrix (this rank's rows only).
// Every rank regenerates the same B from the fixed seed, so the result is
// identical to the sequential generator.
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n, const int rank, const int size) {
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
    const size_t nloc = localRowCount(n, rank, size);
    A.assign(nloc * n, 0.0);
    for (size_t li = 0; li < nloc; ++li) {
        const size_t i = static_cast<size_t>(rank) + li * static_cast<size_t>(size);
        double* row = &A[li * n];
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += B[i * n + k] * B[j * n + k];
            }
            row[j] = sum;
        }
        // Add diagonal dominance to ensure positive definiteness
        row[i] += n;
    }
}

// Gather the row-distributed matrix into a full row-major n x n matrix on rank 0.
void gatherMatrix(const std::vector<double>& local, const size_t n, const int rank, const int size,
                  std::vector<double>& full) {
    const size_t P = static_cast<size_t>(size);
    if (rank == 0) {
        full.assign(n * n, 0.0);
    }
    for (size_t i = 0; i < n; ++i) {
        const int owner = static_cast<int>(i % P);
        const size_t li = i / P;
        if (owner == 0) {
            if (rank == 0) {
                std::copy(&local[li * n], &local[li * n] + n, &full[i * n]);
            }
        } else if (rank == owner) {
            MPI_Send(&local[li * n], static_cast<int>(n), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        } else if (rank == 0) {
            MPI_Recv(&full[i * n], static_cast<int>(n), MPI_DOUBLE, owner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
}

// Validation runs on rank 0 with the gathered matrices.
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

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

static int runBenchmark(int argc, char** argv, const int rank, const int size) {
    size_t n = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (identical on all ranks; only rank 0 prints)
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

    rootPrintf("Cholesky Decomposition Benchmark\n");
    rootPrintf("Matrix size: %zu x %zu\n", n, n);
    rootPrintf("Validation: %s\n", validate ? "enabled" : "disabled");

    // Generate positive definite matrix (distributed by rows)
    rootPrintf("Generating positive definite matrix...\n");
    std::vector<double> A;
    generatePositiveDefiniteMatrix(A, n, rank, size);

    std::vector<double> A_orig;
    if (validate) {
        gatherMatrix(A, n, rank, size, A_orig);  // Save original for validation
    }

    // Perform Cholesky decomposition
    rootPrintf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    const bool success = choleskyDecomposition(A, n, rank, size);

    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = elapsed;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (!success) {
        rootPrintf("Cholesky decomposition failed\n");
        return 1;
    }

    const long durationMs = static_cast<long>(maxElapsed * 1000.0);
    rootPrintf("Computation time: %ld ms\n", durationMs);

    // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
    const double ops = (double)n * n * n / 3.0;
    const double gflops = ops / (durationMs / 1000.0) / 1e9;
    rootPrintf("Performance: %.3f GFLOPS\n", gflops);

    // Gather the factor on rank 0 for printing / validation
    std::vector<double> L;
    if (printResults || validate) {
        gatherMatrix(A, n, rank, size, L);
    }

    // Print results for external validation
    if (printResults) {
        if (rank == 0) {
            print_results(L, "CholeskyL");
        }
    }

    // Validation
    if (validate) {
        int valid = 0;
        if (rank == 0) {
            rootPrintf("Validating result...\n");
            valid = validateCholesky(L, A_orig, n) ? 1 : 0;
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
        return valid ? 0 : 1;
    }

    return 0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    g_rank = rank;

    const int code = runBenchmark(argc, argv, rank, size);

    MPI_Finalize();
    return code;
}
