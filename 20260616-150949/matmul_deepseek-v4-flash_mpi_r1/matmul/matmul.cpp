#include <algorithm>
#include <chrono>
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

void initMatrix(std::vector<double>& mat, const size_t N) {
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

// Cache-friendly matrix multiplication using i-k-j loop order.
// Each process computes its assigned row block of C = A * B.
void matrixMultiplyLocal(const std::vector<double>& A_local, const std::vector<double>& B,
                         std::vector<double>& C_local, const size_t local_rows, const size_t N) {
    for (size_t i = 0; i < local_rows; ++i) {
        for (size_t k = 0; k < N; ++k) {
            const double aik = A_local[i * N + k];
            for (size_t j = 0; j < N; ++j) {
                C_local[i * N + j] += aik * B[k * N + j];
            }
        }
    }
}

// Simple validation: compute a single element and compare
bool validateResult(const std::vector<double>& A, const std::vector<double>& B,
                   const std::vector<double>& C, const size_t N) {
    // Check a few random positions
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
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments on all ranks
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

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI processes: %d\n", num_procs);
    }

    // Distribute rows among processes (handles non-divisible N)
    const size_t rows_per_proc = N / num_procs;
    const size_t rem = N % num_procs;
    const size_t start_row = rank * rows_per_proc + std::min(static_cast<size_t>(rank), rem);
    const size_t local_rows = rows_per_proc + (static_cast<size_t>(rank) < rem ? 1 : 0);

    // Each process allocates its portion of A rows, the full B, and its C rows
    std::vector<double> A_local(local_rows * N);
    std::vector<double> B(N * N);
    std::vector<double> C_local(local_rows * N, 0.0);

    // Initialize local A rows deterministically (no communication needed)
    for (size_t i = 0; i < local_rows; ++i) {
        for (size_t j = 0; j < N; ++j) {
            A_local[i * N + j] = getPseudoRndValue(N, start_row + i, j);
        }
    }

    if (rank == 0) {
        printf("Initializing matrices...\n");
        initMatrix(B, N);
    }
    MPI_Bcast(B.data(), static_cast<int>(N * N), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computing matrix multiplication...\n");
    }

    // Synchronize all processes before the timed computation
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    matrixMultiplyLocal(A_local, B, C_local, local_rows, N);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    const long long local_time = duration.count();

    // Determine the wall-clock time of the slowest process
    long long max_time = 0;
    MPI_Reduce(&local_time, &max_time, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather the full C matrix on rank 0 using Gatherv (handles variable row counts)
    std::vector<double> C_full;
    std::vector<int> recv_counts(num_procs);
    std::vector<int> displs(num_procs);

    if (rank == 0) {
        C_full.resize(N * N);
        for (int p = 0; p < num_procs; ++p) {
            const size_t p_rows = rows_per_proc + (static_cast<size_t>(p) < rem ? 1 : 0);
            const size_t p_start = p * rows_per_proc + std::min(static_cast<size_t>(p), rem);
            recv_counts[p] = static_cast<int>(p_rows * N);
            displs[p] = static_cast<int>(p_start * N);
        }
    }

    MPI_Gatherv(C_local.data(), static_cast<int>(local_rows * N), MPI_DOUBLE,
                C_full.data(), recv_counts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %lld ms\n", max_time);

        // Calculate GFLOPS: 2*N^3 operations divided by wall time
        const double gflops = (2.0 * static_cast<double>(N) * N * N) / (static_cast<double>(max_time) / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) {
            print_results(C_full, "MatrixC");
        }

        if (validate) {
            // Reconstruct full A on rank 0 for validation
            std::vector<double> A_full(N * N);
            initMatrix(A_full, N);

            printf("Validating result...\n");
            const bool valid = validateResult(A_full, B, C_full, N);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                MPI_Finalize();
                return 1;
            }
        }
    }

    MPI_Finalize();
    return 0;
}
