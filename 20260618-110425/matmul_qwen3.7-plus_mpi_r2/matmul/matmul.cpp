#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

// Initialize rows [row_start, row_start + nrows) of a matrix
void initMatrixRows(std::vector<double>& mat, const size_t N, const size_t row_start, const size_t nrows) {
    for (size_t i = row_start; i < row_start + nrows; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[(i - row_start) * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

// Initialize full matrix
void initMatrix(std::vector<double>& mat, const size_t N) {
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

// Cache-optimized local matrix multiply: C_local = A_local * B
// Uses i-k-j loop order for row-major cache efficiency
void matrixMultiplyLocal(const std::vector<double>& A_local, const std::vector<double>& B,
                         std::vector<double>& C_local, const size_t N, const size_t nrows) {
    std::memset(C_local.data(), 0, nrows * N * sizeof(double));

    for (size_t i = 0; i < nrows; ++i) {
        for (size_t k = 0; k < N; ++k) {
            const double a_ik = A_local[i * N + k];
            const double* __restrict__ b_row = &B[k * N];
            double* __restrict__ c_row = &C_local[i * N];
            for (size_t j = 0; j < N; ++j) {
                c_row[j] += a_ik * b_row[j];
            }
        }
    }
}

// Simple validation: compute elements and compare
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

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    size_t N = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments on all processes
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            N = atoi(argv[++i]);
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

    // Compute 1D block-row decomposition
    // Each process gets a contiguous block of rows
    const size_t base_rows = N / nprocs;
    const size_t remainder = N % nprocs;
    // Processes [0, remainder) get (base_rows+1) rows, rest get base_rows
    const size_t local_nrows = base_rows + (static_cast<size_t>(rank) < remainder ? 1 : 0);
    size_t row_start = 0;
    for (int p = 0; p < rank; ++p) {
        row_start += base_rows + (static_cast<size_t>(p) < remainder ? 1 : 0);
    }

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark (MPI)\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Processes: %d\n", nprocs);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Each process generates its local rows of A and the full B matrix
    // (deterministic initialization, no communication needed for inputs)
    std::vector<double> A_local(local_nrows * N);
    std::vector<double> B(N * N);

    if (rank == 0) {
        printf("Initializing matrices...\n");
    }

    initMatrixRows(A_local, N, row_start, local_nrows);
    initMatrix(B, N);

    // Local result buffer
    std::vector<double> C_local(local_nrows * N, 0.0);

    // Barrier to ensure all processes are ready before timing
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computing matrix multiplication...\n");
    }

    // Time the local computation
    double local_start = MPI_Wtime();

    matrixMultiplyLocal(A_local, B, C_local, N, local_nrows);

    double local_end = MPI_Wtime();
    double local_elapsed = local_end - local_start;

    // Get the maximum elapsed time across all processes
    double max_elapsed;
    MPI_Reduce(&local_elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather results to rank 0 using MPI_Gatherv
    // Compute counts and displacements for all processes
    std::vector<int> recv_counts(nprocs);
    std::vector<int> displacements(nprocs);
    {
        int offset = 0;
        for (int p = 0; p < nprocs; ++p) {
            size_t p_rows = base_rows + (static_cast<size_t>(p) < remainder ? 1 : 0);
            recv_counts[p] = static_cast<int>(p_rows * N);
            displacements[p] = offset;
            offset += recv_counts[p];
        }
    }

    std::vector<double> C_full;
    if (rank == 0) {
        C_full.resize(N * N);
    }

    MPI_Gatherv(C_local.data(), static_cast<int>(local_nrows * N), MPI_DOUBLE,
                C_full.data(), recv_counts.data(), displacements.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // Rank 0 reports results
    if (rank == 0) {
        long duration_ms = static_cast<long>(max_elapsed * 1000.0);
        printf("Computation time: %ld ms\n", duration_ms);

        double gflops = (2.0 * N * N * N) / (max_elapsed) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        // Print results for external validation
        if (printResults) {
            print_results(C_full, "MatrixC");
        }

        // Validation
        if (validate) {
            printf("Validating result...\n");
            // Regenerate full A for validation (deterministic)
            std::vector<double> A_full(N * N);
            initMatrix(A_full, N);
            bool valid = validateResult(A_full, B, C_full, N);

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
