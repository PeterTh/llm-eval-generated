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
//
// Data distribution: rows are distributed cyclically over the ranks (global row i
// lives on rank i % P as local row i / P). Each rank stores only its rows, row-major.
//
// Algorithm: column-oriented (left-looking) sweep. At step j:
//   1. The owner of row j computes the pivot L[j][j] and the prefix L[j][0..j].
//   2. The prefix is broadcast to all ranks.
//   3. Every rank updates column j of its local rows i > j.
// Every element L[i][j] is computed as a sum over k = 0..j-1 in increasing order, as in
// the original row-oriented algorithm, so the arithmetic is the same. The result does not
// depend on the number of ranks. Compiler FMA contraction of the blocked kernel can
// differ from the original single-row loop in the last bit.

namespace {

// Number of rows owned by `rank` when n rows are distributed cyclically over `size` ranks.
size_t localRowCount(size_t n, int rank, int size) {
    const size_t r = static_cast<size_t>(rank);
    const size_t P = static_cast<size_t>(size);
    return n > r ? (n - r - 1) / P + 1 : 0;
}

// Update column j of local rows [li0, nLoc) using the broadcast prefix rowj[0..j-1]
// and pivot Ljj. Rows are processed in blocks of four to expose instruction-level
// parallelism; each row's sum is still accumulated in increasing k order. Every row
// (including the padded last block) goes through the same kernel, so each element's
// result does not depend on how rows are distributed over ranks.
void updateColumn(double* Aloc, size_t n, size_t li0, size_t nLoc, size_t j,
                  const double* rowj, double Ljj) {
    for (size_t li = li0; li < nLoc; li += 4) {
        const size_t cnt = std::min<size_t>(4, nLoc - li);
        // Missing rows of a partial block alias row li; their results are discarded.
        const double* r0 = Aloc + li * n;
        const double* r1 = cnt > 1 ? r0 + n : r0;
        const double* r2 = cnt > 2 ? r0 + 2 * n : r0;
        const double* r3 = cnt > 3 ? r0 + 3 * n : r0;
        double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
        for (size_t k = 0; k < j; ++k) {
            const double b = rowj[k];
            s0 += r0[k] * b;
            s1 += r1[k] * b;
            s2 += r2[k] * b;
            s3 += r3[k] * b;
        }
        double* out = Aloc + li * n + j;
        out[0] = (out[0] - s0) / Ljj;
        if (cnt > 1) out[n] = (out[n] - s1) / Ljj;
        if (cnt > 2) out[2 * n] = (out[2 * n] - s2) / Ljj;
        if (cnt > 3) out[3 * n] = (out[3 * n] - s3) / Ljj;
    }
}

// Decompose the distributed matrix in place. On success, Aloc holds the local rows of L
// (zeros above the diagonal). Returns the same value on all ranks.
bool choleskyDecomposition(std::vector<double>& Aloc, const size_t n, int rank, int size) {
    const size_t P = static_cast<size_t>(size);
    const size_t nLoc = localRowCount(n, rank, size);

    // rowBuf[0]: failure flag; rowBuf[1 + k]: L[j][k] for k = 0..j
    std::vector<double> rowBuf(n + 1);

    for (size_t j = 0; j < n; ++j) {
        const int owner = static_cast<int>(j % P);
        double* buf = rowBuf.data() + 1;
        rowBuf[0] = 0.0;

        if (owner == rank) {
            const size_t lj = j / P;
            double* rowj = Aloc.data() + lj * n;
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k) {
                sum += rowj[k] * rowj[k];
            }
            const double val = rowj[j] - sum;
            if (val <= 0.0) {
                rowBuf[0] = 1.0;  // Matrix is not positive definite
            } else {
                const double Ljj = sqrt(val);
                rowj[j] = Ljj;
                std::copy(rowj, rowj + j, buf);
                buf[j] = Ljj;
            }
        }

        MPI_Bcast(rowBuf.data(), static_cast<int>(j + 2), MPI_DOUBLE, owner, MPI_COMM_WORLD);

        if (rowBuf[0] != 0.0) {
            if (rank == 0) {
                printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
            }
            return false;
        }

        // First local row with global index i > j (global i = rank + li * P).
        const size_t li0 = (static_cast<size_t>(rank) > j)
                               ? 0
                               : (j - static_cast<size_t>(rank)) / P + 1;
        updateColumn(Aloc.data(), n, li0, nLoc, j, buf, buf[j]);
    }

    // Zero out upper triangular part of local rows
    for (size_t li = 0; li < nLoc; ++li) {
        const size_t i = static_cast<size_t>(rank) + li * P;
        double* r = Aloc.data() + li * n;
        for (size_t j = i + 1; j < n; ++j) {
            r[j] = 0.0;
        }
    }

    return true;
}

// Generate a symmetric positive definite matrix (distributed: each rank fills its own rows).
void generatePositiveDefiniteMatrix(std::vector<double>& Aloc, const size_t n, int rank, int size) {
    // Method: Create A = B * B^T where B is random
    // This guarantees A is positive semi-definite
    // Then add identity to make it strictly positive definite

    // Every rank generates the same full B from the same seed.
    std::vector<double> B(n * n);
    unsigned int seed = 42;

    // Generate random matrix B
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

    // Compute local rows of A = B * B^T
    const size_t nLoc = localRowCount(n, rank, size);
    for (size_t li = 0; li < nLoc; ++li) {
        const size_t i = static_cast<size_t>(rank) + li * static_cast<size_t>(size);
        double* row = Aloc.data() + li * n;
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

// Gather the row-distributed matrix into a full n x n row-major matrix on every rank.
std::vector<double> gatherMatrix(const std::vector<double>& Aloc, const size_t n, int rank, int size) {
    MPI_Datatype rowType;
    MPI_Type_contiguous(static_cast<int>(n), MPI_DOUBLE, &rowType);
    MPI_Type_commit(&rowType);

    std::vector<int> counts(size), displs(size);
    int offset = 0;
    for (int r = 0; r < size; ++r) {
        counts[r] = static_cast<int>(localRowCount(n, r, size));
        displs[r] = offset;
        offset += counts[r];
    }

    std::vector<double> ordered(n * n);  // rows grouped by owning rank
    MPI_Allgatherv(Aloc.data(), counts[rank], rowType, ordered.data(), counts.data(),
                   displs.data(), rowType, MPI_COMM_WORLD);
    MPI_Type_free(&rowType);

    std::vector<double> full(n * n);
    for (int r = 0; r < size; ++r) {
        for (int li = 0; li < counts[r]; ++li) {
            const size_t i = static_cast<size_t>(r) + static_cast<size_t>(li) * size;
            std::memcpy(&full[i * n], &ordered[(static_cast<size_t>(displs[r]) + li) * n],
                        n * sizeof(double));
        }
    }
    return full;
}

// Validate by computing L * L^T on the local rows and comparing with the original matrix.
bool validateCholesky(const std::vector<double>& L, const std::vector<double>& AorigLoc,
                      const size_t n, int rank, int size) {
    const size_t nLoc = localRowCount(n, rank, size);
    double maxError = 0.0;
    double relError = 0.0;

    for (size_t li = 0; li < nLoc; ++li) {
        const size_t i = static_cast<size_t>(rank) + li * static_cast<size_t>(size);
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += L[i * n + k] * L[j * n + k];
            }
            const double orig = AorigLoc[li * n + j];
            const double error = fabs(sum - orig);
            maxError = std::max(maxError, error);

            const double rel = error / (fabs(orig) + 1e-10);
            relError = std::max(relError, rel);
        }
    }

    MPI_Allreduce(MPI_IN_PLACE, &maxError, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &relError, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

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

int runBenchmark(int argc, char** argv, int rank, int size) {
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
            if (rank == 0) printUsage(argv[0]);
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

    // Allocate local rows of the matrix
    const size_t nLoc = localRowCount(n, rank, size);
    std::vector<double> A(nLoc * n);
    std::vector<double> A_orig;

    // Generate positive definite matrix
    if (rank == 0) printf("Generating positive definite matrix...\n");
    generatePositiveDefiniteMatrix(A, n, rank, size);

    if (validate) {
        A_orig = A;  // Save original for validation
    }

    // Perform Cholesky decomposition
    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();

    bool success = choleskyDecomposition(A, n, rank, size);

    const double localElapsed = MPI_Wtime() - t0;
    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (!success) {
        if (rank == 0) printf("Cholesky decomposition failed\n");
        return 1;
    }

    const long durationMs = static_cast<long>(elapsed * 1000.0);
    if (rank == 0) {
        printf("Computation time: %ld ms\n", durationMs);

        // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (durationMs / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Assemble the full L matrix when it is needed for output or validation
    std::vector<double> L;
    if (printResults || validate) {
        L = gatherMatrix(A, n, rank, size);
    }

    // Print results for external validation
    if (printResults && rank == 0) {
        print_results(L, "CholeskyL");
    }

    // Validation
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        bool valid = validateCholesky(L, A_orig, n, rank, size);

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

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int ret = runBenchmark(argc, argv, rank, size);

    MPI_Finalize();
    return ret;
}
