#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// Determine which MPI rank owns a given row (balanced block distribution)
inline int owner_of_row(size_t row, size_t n, int nprocs) {
    size_t base = n / nprocs;
    size_t extra = n % nprocs;
    size_t threshold = (base + 1) * extra;
    if (row < threshold) {
        return (int)(row / (base + 1));
    } else {
        if (base == 0) return (int)extra;
        return (int)(extra + (row - threshold) / base);
    }
}

// Get the range of rows [start, end) owned by process p
inline void get_row_range(int p, size_t n, int nprocs, size_t& start, size_t& end) {
    size_t base = n / nprocs;
    size_t extra = n % nprocs;
    start = (size_t)p * base + std::min((size_t)p, extra);
    end = start + base + ((size_t)p < extra ? 1 : 0);
}

// Generate local rows of a symmetric positive definite matrix
// All processes generate the same B matrix (deterministic seed), then compute only their local rows of A = B*B^T + n*I
void generateLocalPositiveDefiniteMatrix(std::vector<double>& A_local, size_t n,
                                          size_t row_start, size_t row_end) {
    size_t n_local = row_end - row_start;

    // Generate full B (identical on all processes for reproducibility)
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

    // Compute local rows of A = B * B^T
    A_local.resize(n_local * n);
    for (size_t li = 0; li < n_local; ++li) {
        size_t i = row_start + li;
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += B[i * n + k] * B[j * n + k];
            }
            A_local[li * n + j] = sum;
        }
    }

    // Add diagonal dominance to ensure positive definiteness
    for (size_t li = 0; li < n_local; ++li) {
        size_t i = row_start + li;
        A_local[li * n + i] += (double)n;
    }
}

// MPI-parallel Cholesky decomposition (right-looking, row-distributed)
// Matrix rows are distributed across processes. For each column j:
//   1. The owner of row j computes L[j][j] and L[i][j] for its local rows i > j
//   2. Row j (elements 0..j) is broadcast to all processes
//   3. Non-owners compute L[i][j] for their local rows i > j using the broadcast data
bool choleskyDecompositionMPI(std::vector<double>& A_local, size_t n,
                               size_t row_start, size_t row_end,
                               int rank, int nprocs) {
    size_t n_local = row_end - row_start;
    std::vector<double> row_buf(n);

    for (size_t j = 0; j < n; ++j) {
        int owner = owner_of_row(j, n, nprocs);

        if (rank == owner) {
            size_t local_j = j - row_start;

            // Compute diagonal element L[j][j]
            double diag_sum = 0.0;
            for (size_t k = 0; k < j; ++k) {
                const double v = A_local[local_j * n + k];
                diag_sum += v * v;
            }
            const double diag_val = A_local[local_j * n + j] - diag_sum;
            if (diag_val <= 0.0) {
                if (rank == 0) {
                    printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
                }
                return false;
            }
            const double Ljj = sqrt(diag_val);
            A_local[local_j * n + j] = Ljj;

            // Compute off-diagonal elements L[i][j] for local rows i > j
            for (size_t li = 0; li < n_local; ++li) {
                size_t i = row_start + li;
                if (i <= j) continue;
                double sum = 0.0;
                for (size_t k = 0; k < j; ++k) {
                    sum += A_local[li * n + k] * A_local[local_j * n + k];
                }
                A_local[li * n + j] = (A_local[li * n + j] - sum) / Ljj;
            }

            // Pack row j for broadcast: L[j][0..j]
            for (size_t k = 0; k <= j; ++k) {
                row_buf[k] = A_local[local_j * n + k];
            }
        }

        // Broadcast computed row j to all processes
        MPI_Bcast(row_buf.data(), (int)(j + 1), MPI_DOUBLE, owner, MPI_COMM_WORLD);

        // Non-owner processes compute their local rows using broadcast data
        if (rank != owner) {
            const double Ljj = row_buf[j];
            for (size_t li = 0; li < n_local; ++li) {
                size_t i = row_start + li;
                if (i <= j) continue;
                double sum = 0.0;
                for (size_t k = 0; k < j; ++k) {
                    sum += A_local[li * n + k] * row_buf[k];
                }
                A_local[li * n + j] = (A_local[li * n + j] - sum) / Ljj;
            }
        }
    }

    // Zero out upper triangular part for local rows
    for (size_t li = 0; li < n_local; ++li) {
        size_t i = row_start + li;
        for (size_t j = i + 1; j < n; ++j) {
            A_local[li * n + j] = 0.0;
        }
    }

    return true;
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
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

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    size_t n = 512;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("MPI processes: %d\n", nprocs);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Determine local row range
    size_t row_start, row_end;
    get_row_range(rank, n, nprocs, row_start, row_end);
    size_t n_local = row_end - row_start;

    // Generate local rows of positive definite matrix
    if (rank == 0) printf("Generating positive definite matrix...\n");
    std::vector<double> A_local(n_local * n);
    generateLocalPositiveDefiniteMatrix(A_local, n, row_start, row_end);

    std::vector<double> A_orig_local;
    if (validate) {
        A_orig_local = A_local;
    }

    // Perform Cholesky decomposition
    if (rank == 0) printf("Computing Cholesky decomposition...\n");

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    bool success = choleskyDecompositionMPI(A_local, n, row_start, row_end, rank, nprocs);

    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();

    // Check success across all processes
    int local_success = success ? 1 : 0;
    int global_success = 0;
    MPI_Allreduce(&local_success, &global_success, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    if (!global_success) {
        if (rank == 0) printf("Cholesky decomposition failed\n");
        MPI_Finalize();
        return 1;
    }

    // Report timing (max across all processes)
    double local_time = end_time - start_time;
    double max_time;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        long duration_ms = (long)(max_time * 1000.0);
        printf("Computation time: %ld ms\n", duration_ms);
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / max_time / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Print results for external validation
    if (printResults) {
        // Gather row counts from all processes
        int local_count = (int)(n_local * n);
        std::vector<int> recvcounts(nprocs), displs(nprocs);
        MPI_Allgather(&local_count, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, MPI_COMM_WORLD);

        displs[0] = 0;
        for (int p = 1; p < nprocs; ++p) {
            displs[p] = displs[p - 1] + recvcounts[p - 1];
        }

        std::vector<double> L_full;
        if (rank == 0) L_full.resize(n * n);

        MPI_Gatherv(A_local.data(), local_count, MPI_DOUBLE,
                     rank == 0 ? L_full.data() : nullptr,
                     recvcounts.data(), displs.data(), MPI_DOUBLE,
                     0, MPI_COMM_WORLD);

        if (rank == 0) {
            print_results(L_full, "CholeskyL");
        }
    }

    // Validation
    if (validate) {
        int local_count = (int)(n_local * n);
        std::vector<int> recvcounts(nprocs), displs(nprocs);
        MPI_Allgather(&local_count, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, MPI_COMM_WORLD);

        displs[0] = 0;
        for (int p = 1; p < nprocs; ++p) {
            displs[p] = displs[p - 1] + recvcounts[p - 1];
        }

        std::vector<double> L_full, A_orig_full;
        if (rank == 0) {
            L_full.resize(n * n);
            A_orig_full.resize(n * n);
        }

        MPI_Gatherv(A_local.data(), local_count, MPI_DOUBLE,
                     rank == 0 ? L_full.data() : nullptr,
                     recvcounts.data(), displs.data(), MPI_DOUBLE,
                     0, MPI_COMM_WORLD);

        MPI_Gatherv(A_orig_local.data(), local_count, MPI_DOUBLE,
                     rank == 0 ? A_orig_full.data() : nullptr,
                     recvcounts.data(), displs.data(), MPI_DOUBLE,
                     0, MPI_COMM_WORLD);

        if (rank == 0) {
            printf("Validating result...\n");
            bool valid = validateCholesky(L_full, A_orig_full, n);

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
