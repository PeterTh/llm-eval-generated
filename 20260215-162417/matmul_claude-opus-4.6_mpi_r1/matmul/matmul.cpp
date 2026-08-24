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

// Initialize a subset of rows [row_start, row_start+num_rows) of an NxN matrix
void initMatrixRows(double* mat, const size_t N, const size_t row_start, const size_t num_rows) {
    for (size_t i = 0; i < num_rows; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, row_start + i, j);
        }
    }
}

// Local matmul: C_local[num_rows x N] = A_local[num_rows x N] * B[N x N]
// Uses ikj loop order for better cache performance on B
void matrixMultiplyLocal(const double* A_local, const double* B,
                         double* C_local, const size_t num_rows, const size_t N) {
    std::memset(C_local, 0, num_rows * N * sizeof(double));
    constexpr size_t BLOCK = 64;
    for (size_t ii = 0; ii < num_rows; ii += BLOCK) {
        const size_t i_end = std::min(ii + BLOCK, num_rows);
        for (size_t kk = 0; kk < N; kk += BLOCK) {
            const size_t k_end = std::min(kk + BLOCK, N);
            for (size_t jj = 0; jj < N; jj += BLOCK) {
                const size_t j_end = std::min(jj + BLOCK, N);
                for (size_t i = ii; i < i_end; ++i) {
                    for (size_t k = kk; k < k_end; ++k) {
                        const double a_ik = A_local[i * N + k];
                        for (size_t j = jj; j < j_end; ++j) {
                            C_local[i * N + j] += a_ik * B[k * N + j];
                        }
                    }
                }
            }
        }
    }
}

// Simple validation: compute a single element and compare
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

    // Parse command line arguments
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
        printf("MPI processes: %d\n", nprocs);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Compute row distribution: each rank gets a contiguous block of rows
    const size_t base_rows = N / static_cast<size_t>(nprocs);
    const size_t remainder = N % static_cast<size_t>(nprocs);
    const size_t my_rows = base_rows + (static_cast<size_t>(rank) < remainder ? 1 : 0);
    size_t my_row_start = base_rows * static_cast<size_t>(rank) + std::min(static_cast<size_t>(rank), remainder);

    // Precompute sendcounts/displacements for Gatherv
    std::vector<int> recvcounts(nprocs);
    std::vector<int> displs(nprocs);
    for (int r = 0; r < nprocs; ++r) {
        size_t r_rows = base_rows + (static_cast<size_t>(r) < remainder ? 1 : 0);
        recvcounts[r] = static_cast<int>(r_rows * N);
        size_t r_start = base_rows * static_cast<size_t>(r) + std::min(static_cast<size_t>(r), remainder);
        displs[r] = static_cast<int>(r_start * N);
    }

    // Each rank initializes its own rows of A locally (deterministic, no communication)
    std::vector<double> A_local(my_rows * N);
    initMatrixRows(A_local.data(), N, my_row_start, my_rows);

    // Each rank initializes full B locally (deterministic, no communication)
    std::vector<double> B(N * N);
    initMatrixRows(B.data(), N, 0, N);

    if (rank == 0) printf("Initializing matrices...\n");

    // Allocate local C
    std::vector<double> C_local(my_rows * N);

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) printf("Computing matrix multiplication...\n");
    auto start = std::chrono::high_resolution_clock::now();

    matrixMultiplyLocal(A_local.data(), B.data(), C_local.data(), my_rows, N);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long local_duration_ms = static_cast<long>(duration.count());
    long max_duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &max_duration_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", max_duration_ms);
        double gflops = (2.0 * N * N * N) / (max_duration_ms / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Gather full C on rank 0 for validation/output
    std::vector<double> C;
    if (rank == 0) C.resize(N * N);

    MPI_Gatherv(C_local.data(), static_cast<int>(my_rows * N), MPI_DOUBLE,
                C.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        if (printResults) {
            print_results(C, "MatrixC");
        }

        if (validate) {
            // Reconstruct full A on rank 0 for validation
            std::vector<double> A(N * N);
            initMatrixRows(A.data(), N, 0, N);

            printf("Validating result...\n");
            bool valid = validateResult(A, B, C, N);

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
