#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// Cache block size for tiling optimization
constexpr size_t BLOCK_SIZE = 64;

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

// Initialize a single row of a matrix
void initRow(double* row, const size_t N, const size_t global_i) {
    for (size_t j = 0; j < N; ++j) {
        row[j] = getPseudoRndValue(N, global_i, j);
    }
}

// Blocked (tiled) matrix multiplication for a subset of rows
// C_local (local_rows x N) = A_local (local_rows x N) * B (N x N)
void matmulLocal(const double* A_local, const double* B, double* C_local,
                 const size_t N, const size_t local_rows) {
    // Zero out C_local
    std::fill(C_local, C_local + local_rows * N, 0.0);

    // Tiled i-k-j loop order for cache efficiency
    for (size_t ii = 0; ii < local_rows; ii += BLOCK_SIZE) {
        const size_t i_end = std::min(ii + BLOCK_SIZE, local_rows);
        for (size_t kk = 0; kk < N; kk += BLOCK_SIZE) {
            const size_t k_end = std::min(kk + BLOCK_SIZE, N);
            for (size_t jj = 0; jj < N; jj += BLOCK_SIZE) {
                const size_t j_end = std::min(jj + BLOCK_SIZE, N);
                for (size_t i = ii; i < i_end; ++i) {
                    const double* A_row = &A_local[i * N];
                    double* C_row = &C_local[i * N];
                    for (size_t k = kk; k < k_end; ++k) {
                        const double aik = A_row[k];
                        const double* B_row = &B[k * N];
                        for (size_t j = jj; j < j_end; ++j) {
                            C_row[j] += aik * B_row[j];
                        }
                    }
                }
            }
        }
    }
}

// Validation: verify selected elements against the canonical serial computation
bool validateResult(const std::vector<double>& A, const std::vector<double>& B,
                   const std::vector<double>& C, const size_t N) {
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};

    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = checkPoints[pi] % N;
            const size_t j = checkPoints[pj] % N;

            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += A[i * N + k] * B[k * N + j];
            }

            const double actual = C[i * N + j];
            const double relError = std::abs((actual - expected) / (expected + 1e-10));

            if (relError > 1e-6) {
                printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                       i, j, expected, actual, relError);
                return false;
            }
        }
    }

    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    size_t N = 512;
    int ivalidate = 0;
    int iprintResults = 0;

    // Parse command-line on rank 0; broadcast results
    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                N = static_cast<size_t>(atol(argv[++i]));
            } else if (strcmp(argv[i], "-v") == 0) {
                ivalidate = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                iprintResults = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
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
    }

    unsigned long long n_ull = static_cast<unsigned long long>(N);
    MPI_Bcast(&n_ull, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    N = static_cast<size_t>(n_ull);
    MPI_Bcast(&ivalidate, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&iprintResults, 1, MPI_INT, 0, MPI_COMM_WORLD);

    const bool validate = (ivalidate != 0);
    const bool printResults = (iprintResults != 0);

    // Row distribution: contiguous blocks
    const size_t base_rows = N / static_cast<size_t>(num_procs);
    const size_t rem = N % static_cast<size_t>(num_procs);
    const size_t local_rows = base_rows + (static_cast<size_t>(rank) < rem ? 1 : 0);

    // Build send-counts / displacements for MPI_Gatherv
    std::vector<int> sendcounts(num_procs);
    std::vector<int> displs(num_procs);
    {
        size_t offset = 0;
        for (int r = 0; r < num_procs; ++r) {
            const size_t rows_r = base_rows + (static_cast<size_t>(r) < rem ? 1 : 0);
            sendcounts[r] = static_cast<int>(rows_r * N);
            displs[r] = static_cast<int>(offset);
            offset += rows_r * N;
        }
    }

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark (MPI)\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Processes: %d\n", num_procs);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing matrices...\n");
    }

    // Compute this rank's first global row index for A
    size_t global_row_start = 0;
    for (int r = 0; r < rank; ++r) {
        global_row_start += base_rows + (static_cast<size_t>(r) < rem ? 1 : 0);
    }

    // Each rank independently initialises its rows of A (deterministic)
    std::vector<double> A_local(local_rows * N);
    for (size_t i = 0; i < local_rows; ++i) {
        initRow(&A_local[i * N], N, global_row_start + i);
    }

    // Each rank independently initialises the full B (identical on every rank)
    std::vector<double> B(N * N);
    for (size_t i = 0; i < N; ++i) {
        initRow(&B[i * N], N, i);
    }

    std::vector<double> C_local(local_rows * N, 0.0);

    if (rank == 0) {
        printf("Computing matrix multiplication...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double t_start = MPI_Wtime();

    matmulLocal(A_local.data(), B.data(), C_local.data(), N, local_rows);

    const double t_end = MPI_Wtime();
    const double local_time = t_end - t_start;

    double max_time = 0.0;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather results on rank 0 when output or validation is requested
    std::vector<double> C_full;
    std::vector<double> A_full;
    if (rank == 0) {
        C_full.resize(N * N);
        if (validate) {
            A_full.resize(N * N);
        }
    }

    MPI_Gatherv(C_local.data(), static_cast<int>(local_rows * N), MPI_DOUBLE,
                rank == 0 ? C_full.data() : nullptr,
                sendcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (validate) {
        MPI_Gatherv(A_local.data(), static_cast<int>(local_rows * N), MPI_DOUBLE,
                    A_full.data(),
                    sendcounts.data(), displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
    }

    // Output and validation (rank 0 only)
    if (rank == 0) {
        const long duration_ms = static_cast<long>(max_time * 1000.0);
        printf("Computation time: %ld ms\n", duration_ms);

        const double gflops = (2.0 * N * N * N) / max_time / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) {
            print_results(C_full, "MatrixC");
        }

        int exit_code = 0;
        if (validate) {
            printf("Validating result...\n");
            const bool valid = validateResult(A_full, B, C_full, N);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                exit_code = 1;
            }
        }

        MPI_Finalize();
        return exit_code;
    }

    MPI_Finalize();
    return 0;
}
