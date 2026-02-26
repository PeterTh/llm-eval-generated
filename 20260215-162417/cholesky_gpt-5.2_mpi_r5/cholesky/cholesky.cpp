#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// MPI-parallel Cholesky decomposition (row-cyclic distribution).
// Decomposes positive definite matrix A into L * L^T where L is lower triangular.
//
// Data distribution: global row i is owned by rank (i % nranks) and stored at local index (i / nranks).

static size_t local_row_count(const size_t n, const int rank, const int nranks) {
    if (static_cast<size_t>(rank) >= n) return 0;
    return 1 + (n - 1 - static_cast<size_t>(rank)) / static_cast<size_t>(nranks);
}

static void compute_counts_displs(const size_t n, const int nranks, std::vector<int>& counts, std::vector<int>& displs) {
    counts.assign(nranks, 0);
    displs.assign(nranks, 0);

    long long disp = 0;
    for (int r = 0; r < nranks; ++r) {
        const size_t rows = local_row_count(n, r, nranks);
        const long long elems = static_cast<long long>(rows) * static_cast<long long>(n);
        counts[r] = static_cast<int>(elems);
        displs[r] = static_cast<int>(disp);
        disp += elems;
    }
}

static bool choleskyDecompositionMPI(std::vector<double>& localA, const size_t n, MPI_Comm comm) {
    int rank = 0, nranks = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nranks);

    const size_t local_rows = local_row_count(n, rank, nranks);
    if (localA.size() != local_rows * n) {
        return false;
    }

    std::vector<double> rowbuf(n);

    for (size_t j = 0; j < n; ++j) {
        const int owner = static_cast<int>(j % static_cast<size_t>(nranks));

        int ok = 1;
        if (rank == owner) {
            const size_t j_local = j / static_cast<size_t>(nranks);
            double* rowj = localA.data() + j_local * n;

            double sum = 0.0;
            for (size_t k = 0; k < j; ++k) {
                const double v = rowj[k];
                sum += v * v;
            }

            const double val = rowj[j] - sum;
            if (!(val > 0.0) || !std::isfinite(val)) {
                ok = 0;
            } else {
                rowj[j] = std::sqrt(val);
                // Copy prefix (including diagonal) for broadcast.
                std::memcpy(rowbuf.data(), rowj, (j + 1) * sizeof(double));
            }
        }

        int all_ok = 0;
        MPI_Allreduce(&ok, &all_ok, 1, MPI_INT, MPI_MIN, comm);
        if (!all_ok) {
            int bad = ok ? -1 : static_cast<int>(j);
            int bad_global = -1;
            MPI_Allreduce(&bad, &bad_global, 1, MPI_INT, MPI_MAX, comm);
            if (rank == 0) {
                printf("Error: Matrix is not positive definite at diagonal element %d\n", bad_global);
            }
            return false;
        }

        MPI_Bcast(rowbuf.data(), static_cast<int>(j + 1), MPI_DOUBLE, owner, comm);
        const double diag = rowbuf[j];

        // Update local rows i > j: compute L(i,j).
        size_t i_global = static_cast<size_t>(rank);
        for (size_t i_local = 0; i_local < local_rows; ++i_local, i_global += static_cast<size_t>(nranks)) {
            if (i_global <= j) continue;

            double* rowi = localA.data() + i_local * n;
            double sum = 0.0;

            for (size_t k = 0; k < j; ++k) {
                sum += rowi[k] * rowbuf[k];
            }

            rowi[j] = (rowi[j] - sum) / diag;
        }
    }

    // Zero out upper triangular part for owned rows.
    size_t i_global = static_cast<size_t>(rank);
    for (size_t i_local = 0; i_local < local_rows; ++i_local, i_global += static_cast<size_t>(nranks)) {
        double* rowi = localA.data() + i_local * n;
        for (size_t j = i_global + 1; j < n; ++j) {
            rowi[j] = 0.0;
        }
    }

    return true;
}

// Generate a symmetric positive definite matrix
static void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
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

static bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
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

static void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    size_t n = 512;
    int validate = 0;
    int printResults = 0;

    int early_exit = 0;
    int exit_code = 0;

    if (rank == 0) {
        // Parse command line arguments
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                n = static_cast<size_t>(atoi(argv[++i]));
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                early_exit = 1;
                exit_code = 0;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                early_exit = 1;
                exit_code = 1;
                break;
            }
        }

        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Broadcast run configuration.
    unsigned long long n_u64 = static_cast<unsigned long long>(n);
    MPI_Bcast(&n_u64, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&early_exit, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
    n = static_cast<size_t>(n_u64);

    if (early_exit) {
        MPI_Finalize();
        return exit_code;
    }

    // Root generates the full matrix using the original deterministic generator, then scatters rows cyclically.
    std::vector<double> A_full;
    std::vector<double> A_orig;

    std::vector<int> counts, displs;
    compute_counts_displs(n, nranks, counts, displs);

    const size_t local_rows = local_row_count(n, rank, nranks);
    std::vector<double> localA(local_rows * n);

    std::vector<double> packed;
    if (rank == 0) {
        A_full.resize(n * n);
        if (validate) {
            A_orig.resize(n * n);
        }

        printf("Generating positive definite matrix...\n");
        generatePositiveDefiniteMatrix(A_full, n);
        if (validate) {
            A_orig = A_full;
        }

        packed.reserve(n * n);
        for (int r = 0; r < nranks; ++r) {
            for (size_t i = static_cast<size_t>(r); i < n; i += static_cast<size_t>(nranks)) {
                const double* row = A_full.data() + i * n;
                packed.insert(packed.end(), row, row + n);
            }
        }
    }

    MPI_Scatterv(rank == 0 ? packed.data() : nullptr,
                 counts.data(),
                 displs.data(),
                 MPI_DOUBLE,
                 localA.data(),
                 counts[rank],
                 MPI_DOUBLE,
                 0,
                 MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computing Cholesky decomposition...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();

    const bool success = choleskyDecompositionMPI(localA, n, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    const double t1 = MPI_Wtime();
    const double elapsed = t1 - t0;

    double max_elapsed = 0.0;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (!success) {
        if (rank == 0) {
            printf("Cholesky decomposition failed\n");
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(max_elapsed * 1000.0));

        // Calculate GFLOPS (approximately n^3/3 operations for Cholesky)
        const double ops = (double)n * (double)n * (double)n / 3.0;
        const double gflops = ops / max_elapsed / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Gather results to rank 0 only when requested.
    if (printResults || validate) {
        std::vector<double> packedL;
        if (rank == 0) {
            packedL.resize(n * n);
        }

        MPI_Gatherv(localA.data(),
                    counts[rank],
                    MPI_DOUBLE,
                    rank == 0 ? packedL.data() : nullptr,
                    counts.data(),
                    displs.data(),
                    MPI_DOUBLE,
                    0,
                    MPI_COMM_WORLD);

        if (rank == 0) {
            std::vector<double> L(n * n);
            size_t off = 0;
            for (int r = 0; r < nranks; ++r) {
                const size_t rows = static_cast<size_t>(counts[r]) / n;
                for (size_t t = 0; t < rows; ++t) {
                    const size_t i = static_cast<size_t>(r) + static_cast<size_t>(nranks) * t;
                    std::memcpy(L.data() + i * n, packedL.data() + off, n * sizeof(double));
                    off += n;
                }
            }

            if (printResults) {
                print_results(L, "CholeskyL");
            }

            if (validate) {
                printf("Validating result...\n");
                const bool valid = validateCholesky(L, A_orig, n);
                if (valid) {
                    printf("Validation: PASSED\n");
                    MPI_Finalize();
                    return 0;
                }
                printf("Validation: FAILED\n");
                MPI_Finalize();
                return 1;
            }
        }
    }

    MPI_Finalize();
    return 0;
}
