#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// Distributed Cholesky decomposition (MPI, row-cyclic distribution)
// Decomposes positive definite matrix A into L * L^T where L is lower triangular
//
// Global row i is owned by rank i % P and stored locally as row i / P, so each rank
// holds roughly n/P rows. The factorization runs column by column: the owner of row j
// finishes the diagonal L[j][j] and broadcasts the prefix L[j][0..j], then every rank
// computes L[i][j] for its own rows i > j. Each entry's dot product still accumulates
// in ascending k, so with -ffp-contract=off (set in CMakeLists.txt) results are
// bit-identical to the sequential row-oriented algorithm for any number of ranks.

// Number of rows owned by `rank` when n rows are dealt cyclically over `size` ranks
size_t localRowCount(const size_t n, const size_t rank, const size_t size) {
    return rank < n ? (n - rank + size - 1) / size : 0;
}

// A holds this rank's rows in row-major order (localRowCount(n, rank, size) rows of length n)
bool choleskyDecomposition(std::vector<double>& A, const size_t n, const size_t rank, const size_t size) {
    const size_t nloc = localRowCount(n, rank, size);

    // Broadcast buffer: L[j][0..j] followed by a failure flag
    std::vector<double> pivot(n + 1);

    for (size_t j = 0; j < n; ++j) {
        const size_t owner = j % size;

        if (rank == owner) {
            double* row = &A[(j / size) * n];
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k) {
                sum += row[k] * row[k];
            }
            const double val = row[j] - sum;
            if (val <= 0.0) {
                // Matrix is not positive definite
                pivot[j + 1] = 1.0;
            } else {
                row[j] = sqrt(val);
                pivot[j + 1] = 0.0;
            }
            std::copy(row, row + j + 1, pivot.begin());
        }

        MPI_Bcast(pivot.data(), static_cast<int>(j + 2), MPI_DOUBLE, static_cast<int>(owner), MPI_COMM_WORLD);

        if (pivot[j + 1] != 0.0) {
            if (rank == 0) {
                printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
            }
            return false;
        }

        // Off-diagonal entries of column j for the local rows below the diagonal
        const double diag = pivot[j];
        size_t li = (j < rank) ? 0 : (j - rank) / size + 1;

        // Four rows at a time: independent accumulators hide FP add latency,
        // and each row still sums its terms in ascending k
        for (; li + 4 <= nloc; li += 4) {
            double* r0 = &A[li * n];
            double* r1 = r0 + n;
            double* r2 = r1 + n;
            double* r3 = r2 + n;
            double s0 = 0.0;
            double s1 = 0.0;
            double s2 = 0.0;
            double s3 = 0.0;
            for (size_t k = 0; k < j; ++k) {
                const double p = pivot[k];
                s0 += r0[k] * p;
                s1 += r1[k] * p;
                s2 += r2[k] * p;
                s3 += r3[k] * p;
            }
            r0[j] = (r0[j] - s0) / diag;
            r1[j] = (r1[j] - s1) / diag;
            r2[j] = (r2[j] - s2) / diag;
            r3[j] = (r3[j] - s3) / diag;
        }
        for (; li < nloc; ++li) {
            double* row = &A[li * n];
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k) {
                sum += row[k] * pivot[k];
            }
            row[j] = (row[j] - sum) / diag;
        }
    }

    // Zero out upper triangular part
    for (size_t li = 0; li < nloc; ++li) {
        const size_t i = li * size + rank;
        for (size_t j = i + 1; j < n; ++j) {
            A[li * n + j] = 0.0;
        }
    }

    return true;
}

// Generate a symmetric positive definite matrix (each rank computes only its own rows)
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n, const size_t rank, const size_t size) {
    // Method: Create A = B * B^T where B is random
    // This guarantees A is positive semi-definite
    // Then add identity to make it strictly positive definite

    std::vector<double> B(n * n);
    unsigned int seed = 42;

    // Generate random matrix B (identical on every rank)
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

    // Compute local rows of A = B * B^T
    const size_t nloc = localRowCount(n, rank, size);
    A.resize(nloc * n);
    for (size_t li = 0; li < nloc; ++li) {
        const size_t i = li * size + rank;
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += B[i * n + k] * B[j * n + k];
            }
            A[li * n + j] = sum;
        }
    }

    // Add diagonal dominance to ensure positive definiteness
    for (size_t li = 0; li < nloc; ++li) {
        const size_t i = li * size + rank;
        A[li * n + i] += n;
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n,
                      const size_t rank, const size_t size) {
    // Validate by computing L * L^T and comparing with original matrix
    // Each rank checks its own rows; row j of L is broadcast from its owner

    const size_t nloc = localRowCount(n, rank, size);
    std::vector<double> Lj(n);
    double maxError = 0.0;
    double relError = 0.0;

    for (size_t j = 0; j < n; ++j) {
        const size_t owner = j % size;
        if (rank == owner) {
            const double* row = &L[(j / size) * n];
            std::copy(row, row + n, Lj.begin());
        }
        MPI_Bcast(Lj.data(), static_cast<int>(n), MPI_DOUBLE, static_cast<int>(owner), MPI_COMM_WORLD);

        for (size_t li = 0; li < nloc; ++li) {
            const double* Li = &L[li * n];
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += Li[k] * Lj[k];
            }

            const double error = fabs(sum - A_orig[li * n + j]);
            maxError = std::max(maxError, error);

            const double rel = error / (fabs(A_orig[li * n + j]) + 1e-10);
            relError = std::max(relError, rel);
        }
    }

    double errors[2] = {maxError, relError};
    MPI_Allreduce(MPI_IN_PLACE, errors, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Max absolute error: %.10e\n", errors[0]);
        printf("Max relative error: %.10e\n", errors[1]);
    }

    // Check if error is within tolerance
    if (errors[1] > 1e-6) {
        if (rank == 0) {
            printf("Validation failed: relative error too large\n");
        }
        return false;
    }

    return true;
}

// Gather the full row-major matrix on rank 0 (returns an empty vector on other ranks)
std::vector<double> gatherMatrix(const std::vector<double>& A, const size_t n, const size_t rank, const size_t size) {
    std::vector<int> counts(size);
    std::vector<int> displs(size);
    int offset = 0;
    for (size_t r = 0; r < size; ++r) {
        counts[r] = static_cast<int>(localRowCount(n, r, size) * n);
        displs[r] = offset;
        offset += counts[r];
    }

    // Blocks arrive ordered by rank: rank r holds global rows r, r + size, ...
    std::vector<double> packed(rank == 0 ? n * n : 0);
    MPI_Gatherv(A.data(), static_cast<int>(A.size()), MPI_DOUBLE,
                packed.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        return {};
    }

    std::vector<double> full(n * n);
    for (size_t r = 0; r < size; ++r) {
        const size_t rows = localRowCount(n, r, size);
        for (size_t li = 0; li < rows; ++li) {
            const double* src = &packed[displs[r] + li * n];
            std::copy(src, src + n, full.begin() + (li * size + r) * n);
        }
    }
    return full;
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

    int worldRank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    const size_t rank = static_cast<size_t>(worldRank);
    const size_t size = static_cast<size_t>(worldSize);
    const bool root = rank == 0;

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
            if (root) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (root) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Ranks: %zu\n", size);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Generating positive definite matrix...\n");
    }

    // Generate positive definite matrix (local rows)
    std::vector<double> A;
    generatePositiveDefiniteMatrix(A, n, rank, size);

    std::vector<double> A_orig;
    if (validate) {
        A_orig = A; // Save original for validation
    }

    // Perform Cholesky decomposition
    if (root) {
        printf("Computing Cholesky decomposition...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::steady_clock::now();

    bool success = choleskyDecomposition(A, n, rank, size);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::steady_clock::now();
    double seconds = std::chrono::duration<double>(end - start).count();

    if (!success) {
        if (root) {
            printf("Cholesky decomposition failed\n");
        }
        MPI_Finalize();
        return 1;
    }

    if (root) {
        printf("Computation time: %ld ms\n", static_cast<long>(seconds * 1000.0));

        // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / seconds / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Print results for external validation
    if (printResults) {
        std::vector<double> L = gatherMatrix(A, n, rank, size);
        if (root) {
            print_results(L, "CholeskyL");
        }
    }

    // Validation
    if (validate) {
        if (root) {
            printf("Validating result...\n");
        }
        bool valid = validateCholesky(A, A_orig, n, rank, size);

        if (valid) {
            if (root) {
                printf("Validation: PASSED\n");
            }
            MPI_Finalize();
            return 0;
        } else {
            if (root) {
                printf("Validation: FAILED\n");
            }
            MPI_Finalize();
            return 1;
        }
    }

    MPI_Finalize();
    return 0;
}
