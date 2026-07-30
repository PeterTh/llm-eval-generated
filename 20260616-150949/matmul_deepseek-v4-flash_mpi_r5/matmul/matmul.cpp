#include <mpi.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// Generate pseudo-random values for matrix initialization (deterministic, position-based)
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

// Initialize a contiguous block of rows of matrix A (no communication needed - deterministic)
void initLocalA(std::vector<double>& A_local, const size_t N,
                const size_t local_rows, const size_t start_row) {
    for (size_t i = 0; i < local_rows; ++i) {
        for (size_t j = 0; j < N; ++j) {
            A_local[i * N + j] = getPseudoRndValue(N, start_row + i, j);
        }
    }
}

// Initialize full matrix B on every rank (deterministic - no communication needed)
void initFullB(std::vector<double>& B, const size_t N) {
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            B[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

// Blocked matrix multiplication for optimal cache utilization
// Computes C_local = A_local * B, where A_local has local_rows rows
void matrixMultiplyBlocked(const std::vector<double>& A_local,
                           const std::vector<double>& B,
                           std::vector<double>& C_local,
                           const size_t N, const size_t local_rows,
                           const size_t block_size = 64) {
    std::fill(C_local.begin(), C_local.end(), 0.0);

    for (size_t ii = 0; ii < local_rows; ii += block_size) {
        const size_t i_end = std::min(ii + block_size, local_rows);
        for (size_t kk = 0; kk < N; kk += block_size) {
            const size_t k_end = std::min(kk + block_size, N);
            for (size_t jj = 0; jj < N; jj += block_size) {
                const size_t j_end = std::min(jj + block_size, N);
                for (size_t i = ii; i < i_end; ++i) {
                    for (size_t k = kk; k < k_end; ++k) {
                        const double aik = A_local[i * N + k];
                        const double* B_row = &B[k * N];
                        double* C_row = &C_local[i * N];
                        for (size_t j = jj; j < j_end; ++j) {
                            C_row[j] += aik * B_row[j];
                        }
                    }
                }
            }
        }
    }
}

// Validate result using deterministic element computation (no A matrix needed)
bool validateResult(const std::vector<double>& B, const std::vector<double>& C,
                    const size_t N) {
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};

    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = checkPoints[pi] % N;
            const size_t j = checkPoints[pj] % N;

            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += getPseudoRndValue(N, i, k) * B[k * N + j];
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

    // Parse command line arguments (each rank parses independently - same argv from mpirun)
    bool parseError = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            N = static_cast<size_t>(atol(argv[++i]));
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else if (rank == 0) {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            parseError = true;
        } else {
            parseError = true;
        }
    }

    if (parseError) {
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark (MPI)\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("MPI Processes: %d\n", num_procs);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Compute row distribution
    const size_t base_rows = N / static_cast<size_t>(num_procs);
    const size_t extra = N % static_cast<size_t>(num_procs);
    const size_t local_rows = base_rows + (static_cast<size_t>(rank) < extra ? 1 : 0);

    // Compute start row for this rank
    size_t start_row = static_cast<size_t>(rank) * base_rows
                     + std::min(static_cast<size_t>(rank), extra);

    // Allocate local matrices
    std::vector<double> A_local(local_rows * N);
    std::vector<double> B(N * N);
    std::vector<double> C_local(local_rows * N, 0.0);

    // Initialize matrices deterministically in parallel (no communication needed)
    initLocalA(A_local, N, local_rows, start_row);
    initFullB(B, N);

    // Prepare gather parameters for rank 0
    std::vector<int> recv_counts(num_procs);
    std::vector<int> displacements(num_procs);
    if (rank == 0) {
        size_t offset = 0;
        for (int i = 0; i < num_procs; ++i) {
            size_t rows = base_rows + (static_cast<size_t>(i) < extra ? 1 : 0);
            recv_counts[i] = static_cast<int>(rows * N);
            displacements[i] = static_cast<int>(offset);
            offset += rows * N;
        }
    }

    // Perform matrix multiplication
    if (rank == 0) {
        printf("Computing matrix multiplication...\n");
    }

    auto start = std::chrono::high_resolution_clock::now();

    matrixMultiplyBlocked(A_local, B, C_local, N, local_rows);

    auto end = std::chrono::high_resolution_clock::now();
    auto local_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Synchronize and get the maximum (wall-clock) time across all processes
    long long local_ms = local_duration.count();
    long long max_duration_ms;
    MPI_Reduce(&local_ms, &max_duration_ms, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather local C blocks into full C on rank 0
    std::vector<double> C;
    if (rank == 0) {
        C.resize(N * N);
    }

    MPI_Gatherv(C_local.data(), static_cast<int>(local_rows * N), MPI_DOUBLE,
                rank == 0 ? C.data() : nullptr, recv_counts.data(), displacements.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Report results on rank 0
    if (rank == 0) {
        printf("Computation time: %lld ms\n", max_duration_ms);

        double gflops = (2.0 * N * N * N) / (max_duration_ms / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) {
            print_results(C, "MatrixC");
        }

        if (validate) {
            printf("Validating result...\n");
            // B is fully available on rank 0 (initialized identically on all ranks)
            bool valid = validateResult(B, C, N);

            if (valid) {
                printf("Validation: PASSED\n");
                MPI_Finalize();
                return 0;
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
