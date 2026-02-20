#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// MPI-parallel Cholesky decomposition (distributed-memory, 1D row-block layout)
// Computes the same unblocked Cholesky factor L (lower-triangular) as the original code,
// but parallelizes each column's off-diagonal updates across MPI ranks.

static void computeRowDecomposition(size_t n, int size, std::vector<int>& rowsPerRank, std::vector<int>& rowDispls) {
    rowsPerRank.assign(size, 0);
    rowDispls.assign(size, 0);

    const size_t base = (size > 0) ? (n / static_cast<size_t>(size)) : 0;
    const size_t rem = (size > 0) ? (n % static_cast<size_t>(size)) : 0;

    int disp = 0;
    for (int r = 0; r < size; ++r) {
        const size_t rows = base + (static_cast<size_t>(r) < rem ? 1 : 0);
        rowsPerRank[r] = static_cast<int>(rows);
        rowDispls[r] = disp;
        disp += rowsPerRank[r];
    }
}

static bool choleskyDecompositionMPI(std::vector<double>& localA,
                                    size_t n,
                                    int rank,
                                    int size,
                                    const std::vector<int>& rowsPerRank,
                                    const std::vector<int>& rowDispls) {
    const int myRows = rowsPerRank[rank];
    const int myRow0 = rowDispls[rank];

    std::vector<int> rowOwner(n);
    for (int r = 0; r < size; ++r) {
        const int start = rowDispls[r];
        const int end = start + rowsPerRank[r];
        for (int i = start; i < end; ++i) {
            rowOwner[static_cast<size_t>(i)] = r;
        }
    }

    std::vector<double> pivotRow;
    pivotRow.reserve(n);

    for (size_t k = 0; k < n; ++k) {
        const int owner = rowOwner[k];
        double Lkk = 0.0;
        int ok = 1;
        int badk = -1;

        if (rank == owner) {
            const int lk = static_cast<int>(k) - myRow0;
            double* rowk = localA.data() + static_cast<size_t>(lk) * n;

            double sum = 0.0;
            for (size_t j = 0; j < k; ++j) {
                const double v = rowk[j];
                sum += v * v;
            }

            const double val = rowk[k] - sum;
            if (val <= 0.0) {
                ok = 0;
                badk = static_cast<int>(k);
            } else {
                Lkk = std::sqrt(val);
                rowk[k] = Lkk;
            }

            pivotRow.resize(k);
            if (k > 0) {
                std::memcpy(pivotRow.data(), rowk, k * sizeof(double));
            }
        } else {
            pivotRow.resize(k);
        }

        MPI_Bcast(&ok, 1, MPI_INT, owner, MPI_COMM_WORLD);
        if (!ok) {
            MPI_Bcast(&badk, 1, MPI_INT, owner, MPI_COMM_WORLD);
            if (rank == 0) {
                printf("Error: Matrix is not positive definite at diagonal element %d\n", badk);
            }
            return false;
        }

        MPI_Bcast(&Lkk, 1, MPI_DOUBLE, owner, MPI_COMM_WORLD);
        if (k > 0) {
            MPI_Bcast(pivotRow.data(), static_cast<int>(k), MPI_DOUBLE, owner, MPI_COMM_WORLD);
        }

        // Compute column k for local rows i > k.
        for (int li = 0; li < myRows; ++li) {
            const size_t i = static_cast<size_t>(myRow0 + li);
            if (i <= k) continue;

            double* rowi = localA.data() + static_cast<size_t>(li) * n;
            double sum = 0.0;
            const double* piv = pivotRow.data();
            for (size_t j = 0; j < k; ++j) {
                sum += rowi[j] * piv[j];
            }
            rowi[k] = (rowi[k] - sum) / Lkk;
        }
    }

    // Match original semantics: zero upper triangular part.
    for (int li = 0; li < myRows; ++li) {
        const size_t i = static_cast<size_t>(myRow0 + li);
        double* rowi = localA.data() + static_cast<size_t>(li) * n;
        for (size_t j = i + 1; j < n; ++j) {
            rowi[j] = 0.0;
        }
    }

    return true;
}

// Generate a symmetric positive definite matrix
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    // Method: Create A = B * B^T where B is random
    // This guarantees A is positive semi-definite
    // Then add identity to make it strictly positive definite

    std::vector<double> B(n * n);
    unsigned int seed = 42;

    // Generate random matrix B
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

    // Add diagonal dominance to ensure positive definiteness
    for (size_t i = 0; i < n; ++i) {
        A[i * n + i] += n;
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
    bool showHelp = false;
    bool parseOk = true;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = static_cast<size_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            showHelp = true;
        } else {
            parseOk = false;
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
            }
        }
    }

    if (showHelp || !parseOk) {
        if (rank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return showHelp ? 0 : 1;
    }

    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    std::vector<int> rowsPerRank, rowDispls;
    computeRowDecomposition(n, size, rowsPerRank, rowDispls);

    const int myRows = rowsPerRank[rank];
    std::vector<double> localA(static_cast<size_t>(myRows) * n);

    std::vector<double> A_full;
    std::vector<double> A_orig;

    std::vector<int> sendcounts(size), displs_el(size);
    for (int r = 0; r < size; ++r) {
        sendcounts[r] = rowsPerRank[r] * static_cast<int>(n);
        displs_el[r] = rowDispls[r] * static_cast<int>(n);
    }

    if (rank == 0) {
        A_full.resize(n * n);
        if (rank == 0) {
            printf("Generating positive definite matrix...\n");
        }
        generatePositiveDefiniteMatrix(A_full, n);
        if (validate) {
            A_orig = A_full;
        }
    }

    MPI_Scatterv(rank == 0 ? A_full.data() : nullptr,
                 sendcounts.data(),
                 displs_el.data(),
                 MPI_DOUBLE,
                 localA.data(),
                 sendcounts[rank],
                 MPI_DOUBLE,
                 0,
                 MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computing Cholesky decomposition...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();

    const bool success = choleskyDecompositionMPI(localA, n, rank, size, rowsPerRank, rowDispls);

    MPI_Barrier(MPI_COMM_WORLD);
    const double t1 = MPI_Wtime();
    const double localSec = t1 - t0;

    double maxSec = 0.0;
    MPI_Reduce(&localSec, &maxSec, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (!success) {
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        const long ms = static_cast<long>(maxSec * 1000.0);
        printf("Computation time: %ld ms\n", ms);

        const double ops = (double)n * (double)n * (double)n / 3.0;
        const double gflops = ops / maxSec / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    if (validate || printResults) {
        if (rank == 0) {
            A_full.assign(n * n, 0.0);
        }

        MPI_Gatherv(localA.data(),
                    sendcounts[rank],
                    MPI_DOUBLE,
                    rank == 0 ? A_full.data() : nullptr,
                    sendcounts.data(),
                    displs_el.data(),
                    MPI_DOUBLE,
                    0,
                    MPI_COMM_WORLD);

        if (rank == 0 && printResults) {
            print_results(A_full, "CholeskyL");
        }

        if (rank == 0 && validate) {
            printf("Validating result...\n");
            const bool valid = validateCholesky(A_full, A_orig, n);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            MPI_Finalize();
            return valid ? 0 : 1;
        }
    }

    MPI_Finalize();
    return 0;
}
