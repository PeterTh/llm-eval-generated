#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// Distributed-memory parallel Cholesky decomposition (MPI).
//
// The matrix is distributed by rows in a cyclic fashion: rank r owns global
// rows r, r + P, r + 2P, ...  Cyclic ownership balances the triangular
// workload of Cholesky well (row i requires O(i^2) operations).
//
// The algorithm is the same column-oriented (left-looking) formulation as the
// original serial code.  Every element is accumulated over k in increasing
// order, so the floating-point results are bit-for-bit identical to the
// sequential implementation.  At step j the owner of row j computes the
// diagonal element and broadcasts the pivot row L[j][0..j]; every rank then
// updates its locally owned rows i > j.

bool choleskyDecomposition(std::vector<double>& localA,
                           const size_t n,
                           const int rank,
                           const int size) {
    const size_t numLocal = (n == 0) ? 0 : localA.size() / n;
    std::vector<double> pivot(n);
    bool ok = true;

    for (size_t j = 0; j < n; ++j) {
        const int owner = static_cast<int>(j % static_cast<size_t>(size));
        const size_t pivotLocal =
            (j - static_cast<size_t>(owner)) / static_cast<size_t>(size);

        double* pivotRow = nullptr;
        if (rank == owner) {
            double* row = &localA[pivotLocal * n];

            // Diagonal element: A[j][j] - sum_{k<j} L[j][k]^2
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k) {
                sum += row[k] * row[k];
            }
            const double val = row[j] - sum;
            if (val <= 0.0) {
                printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
                ok = false;
                row[j] = 0.0;
            } else {
                row[j] = sqrt(val);
            }

            // Row j is now complete; zero its upper triangular part.
            for (size_t t = j + 1; t < n; ++t) {
                row[t] = 0.0;
            }
            pivotRow = row;
        } else {
            pivotRow = pivot.data();
        }

        // Share L[j][0..j] with all ranks.
        MPI_Bcast(pivotRow, static_cast<int>(j + 1), MPI_DOUBLE, owner,
                  MPI_COMM_WORLD);

        // Compute column j for all locally owned rows i > j.
        size_t startIdx;
        if (j >= static_cast<size_t>(rank)) {
            startIdx = (j - static_cast<size_t>(rank)) /
                           static_cast<size_t>(size) +
                       1;
        } else {
            startIdx = 0;
        }

        for (size_t li = startIdx; li < numLocal; ++li) {
            double* row = &localA[li * n];

            // Off-diagonal element: (A[i][j] - sum_{k<j} L[i][k]*L[j][k])
            //                      / L[j][j]
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k) {
                sum += row[k] * pivotRow[k];
            }
            row[j] = (row[j] - sum) / pivotRow[j];
        }
    }

    int localOk = ok ? 1 : 0;
    int globalOk = 1;
    MPI_Allreduce(&localOk, &globalOk, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    return globalOk != 0;
}

// Generate the same symmetric positive definite matrix as the serial code,
// but only compute the rows owned by this rank.  Every rank reproduces the
// identical random matrix B (same seed and sequence), so no communication is
// required and the resulting A is bit-for-bit identical.
void generateLocalPositiveDefiniteMatrix(std::vector<double>& localA,
                                         const size_t n,
                                         const int rank,
                                         const int size) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;

    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

    const size_t numLocal = (n == 0) ? 0 : localA.size() / n;
    for (size_t li = 0; li < numLocal; ++li) {
        const size_t i = static_cast<size_t>(rank) +
                         li * static_cast<size_t>(size);

        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += B[i * n + k] * B[j * n + k];
            }
            localA[li * n + j] = sum;
        }

        // Add diagonal dominance to ensure positive definiteness.
        localA[li * n + i] += static_cast<double>(n);
    }
}

// Collect the cyclically distributed rows into a conventional row-major
// matrix on rank 0.
void gatherMatrix(const std::vector<double>& localA,
                  const size_t n,
                  const int rank,
                  const int size,
                  std::vector<double>& full) {
    std::vector<int> counts(size);
    std::vector<int> displs(size);

    int offset = 0;
    for (int r = 0; r < size; ++r) {
        const size_t rows =
            (n > static_cast<size_t>(r))
                ? (n - static_cast<size_t>(r) + static_cast<size_t>(size) - 1) /
                      static_cast<size_t>(size)
                : 0;
        counts[r] = static_cast<int>(rows * n);
        displs[r] = offset;
        offset += counts[r];
    }

    std::vector<double> gathered;
    if (rank == 0) {
        gathered.resize(static_cast<size_t>(offset));
    }

    const int localCount = static_cast<int>(localA.size());
    MPI_Gatherv(localA.data(), localCount, MPI_DOUBLE,
                (rank == 0) ? gathered.data() : nullptr, counts.data(),
                displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        full.assign(n * n, 0.0);
        for (int r = 0; r < size; ++r) {
            const size_t rows =
                (n > static_cast<size_t>(r))
                    ? (n - static_cast<size_t>(r) +
                       static_cast<size_t>(size) - 1) /
                          static_cast<size_t>(size)
                    : 0;
            for (size_t li = 0; li < rows; ++li) {
                const size_t g = static_cast<size_t>(r) +
                                 li * static_cast<size_t>(size);
                std::memcpy(&full[g * n], &gathered[displs[r] + li * n],
                            n * sizeof(double));
            }
        }
    }
}

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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t n = 512;
    bool validate = false;
    bool printResults = false;
    bool run = true;
    int exitCode = 0;

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
            run = false;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            run = false;
            exitCode = 1;
        }
    }

    if (!run) {
        MPI_Finalize();
        return exitCode;
    }

    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Distributed row-cyclic storage: each rank owns only its local rows.
    const size_t numLocal =
        (n > static_cast<size_t>(rank))
            ? (n - static_cast<size_t>(rank) + static_cast<size_t>(size) - 1) /
                  static_cast<size_t>(size)
            : 0;
    std::vector<double> localA(numLocal * n);
    std::vector<double> A_orig;
    std::vector<double> result;

    // Generate positive definite matrix
    if (rank == 0) {
        printf("Generating positive definite matrix...\n");
    }
    generateLocalPositiveDefiniteMatrix(localA, n, rank, size);

    if (validate) {
        // Keep the original matrix on rank 0 for validation.
        gatherMatrix(localA, n, rank, size, A_orig);
    }

    // Perform Cholesky decomposition
    if (rank == 0) {
        printf("Computing Cholesky decomposition...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    bool success = choleskyDecomposition(localA, n, rank, size);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (!success) {
        if (rank == 0) {
            printf("Cholesky decomposition failed\n");
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    if (printResults || validate) {
        gatherMatrix(localA, n, rank, size, result);
    }

    // Print results for external validation
    if (printResults && rank == 0) {
        print_results(result, "CholeskyL");
    }

    // Validation
    int valid = 1;
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
            valid = validateCholesky(result, A_orig, n) ? 1 : 0;
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return valid ? 0 : 1;
}
