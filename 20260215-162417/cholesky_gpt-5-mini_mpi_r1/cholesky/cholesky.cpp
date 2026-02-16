#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mpi.h>

#include "../common/results_output.hpp"

// Parallel Cholesky decomposition using MPI with row-cyclic distribution.
// Each process stores a subset of matrix rows (row i is owned by rank = i % size).
// For pivot j, owner computes pivot row (L[j][0..j]) and broadcasts it to all ranks
// which then update their local rows i > j.

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix
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
    int rank = 0, world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    size_t n = 512;
    bool validate = false;
    bool printResults = false;

    if (rank == 0) {
        // Parse command line arguments on root
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                n = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = true;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = true;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                MPI_Finalize();
                return 0;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                MPI_Finalize();
                return 1;
            }
        }
    }

    // Broadcast n, validate, printResults to all ranks
    MPI_Bcast(&n, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    int val_int = validate ? 1 : 0;
    int pr_int = printResults ? 1 : 0;
    MPI_Bcast(&val_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&pr_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
    validate = (val_int != 0);
    printResults = (pr_int != 0);

    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Generating positive definite matrix...\n");
    }

    std::vector<double> A;            // only root keeps full A (for generation and optional validation)
    std::vector<double> A_orig;       // only root keeps original for validation

    if (rank == 0) {
        A.resize(n * n);
        generatePositiveDefiniteMatrix(A, n);
        if (validate) A_orig = A;
    }

    // Determine local row counts (row-cyclic distribution)
    auto local_rows_for_rank = [&](int r)->size_t {
        if ((size_t)r >= n) return 0;
        return (n + world_size - 1 - r) / world_size;
    };

    std::vector<int> sendcounts(world_size);
    std::vector<int> displs(world_size);
    for (int r = 0; r < world_size; ++r) {
        sendcounts[r] = (int)(local_rows_for_rank(r) * n);
        displs[r] = (r==0) ? 0 : (displs[r-1] + sendcounts[r-1]);
    }

    size_t local_rows = local_rows_for_rank(rank);
    std::vector<double> localA(local_rows * n);

    // Scatter rows from root to localA
    MPI_Scatterv(rank == 0 ? A.data() : nullptr, sendcounts.data(), displs.data(), MPI_DOUBLE,
                 localA.data(), (int)(local_rows * n), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    // Buffer for pivot row; allocate max size n
    std::vector<double> pivot_row(n);

    // Main Cholesky loop over pivots j
    for (size_t j = 0; j < n; ++j) {
        int owner = (int)(j % world_size);
        if (rank == owner) {
            // local index of row j
            size_t local_j = j / world_size;
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k) {
                double v = localA[local_j * n + k];
                sum += v * v;
            }
            double val = localA[local_j * n + j] - sum;
            if (val <= 0.0) {
                if (rank == 0) fprintf(stderr, "Error: Matrix is not positive definite at diagonal element %zu\n", j);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            double Ljj = sqrt(val);
            localA[local_j * n + j] = Ljj;
            // prepare pivot_row[0..j]
            for (size_t k = 0; k <= j; ++k) pivot_row[k] = localA[local_j * n + k];
        }
        // Broadcast pivot_row[0..j] from owner to all ranks
        MPI_Bcast(pivot_row.data(), (int)(j + 1), MPI_DOUBLE, owner, MPI_COMM_WORLD);

        // Each rank updates its local rows with global index > j
        for (size_t local_i = 0; local_i < local_rows; ++local_i) {
            size_t global_i = rank + local_i * world_size;
            if (global_i <= j) continue;
            // compute dot product of localA[local_i][0..j-1] and pivot_row[0..j-1]
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k) {
                sum += localA[local_i * n + k] * pivot_row[k];
            }
            localA[local_i * n + j] = (localA[local_i * n + j] - sum) / pivot_row[j];
        }

        // Owner zeros out upper triangular part of its row j (columns j+1..n-1)
        if (rank == owner) {
            size_t local_j = j / world_size;
            for (size_t col = j + 1; col < n; ++col) localA[local_j * n + col] = 0.0;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();
    double elapsed_ms = (t1 - t0) * 1000.0;

    // Gather results back to root for printing/validation
    std::vector<double> gatheredA;
    if (rank == 0) gatheredA.resize(n * n);
    MPI_Gatherv(localA.data(), (int)(local_rows * n), MPI_DOUBLE,
                rank == 0 ? gatheredA.data() : nullptr, sendcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %.0f ms\n", elapsed_ms);
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (elapsed_ms / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) {
            print_results(gatheredA, "CholeskyL");
        }

        if (validate) {
            printf("Validating result...\n");
            bool valid = validateCholesky(gatheredA, A_orig, n);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
    }

    MPI_Finalize();
    return 0;
}
