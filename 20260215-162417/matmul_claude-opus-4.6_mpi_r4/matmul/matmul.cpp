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

// Initialize rows [row_start, row_start + num_rows) of the matrix
void initMatrixRows(double* mat, const size_t N, const size_t row_start, const size_t num_rows) {
    for (size_t i = 0; i < num_rows; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, row_start + i, j);
        }
    }
}

void initMatrix(double* mat, const size_t N) {
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

// Cache-optimized tiled multiply: C_local = A_local * B_T (B transposed for contiguous access)
void matrixMultiplyLocal(const double* A_local, const double* B_T,
                         double* C_local, const size_t num_rows, const size_t N) {
    std::memset(C_local, 0, num_rows * N * sizeof(double));

    constexpr size_t TILE = 64;

    for (size_t ii = 0; ii < num_rows; ii += TILE) {
        const size_t i_end = std::min(ii + TILE, num_rows);
        for (size_t jj = 0; jj < N; jj += TILE) {
            const size_t j_end = std::min(jj + TILE, N);
            for (size_t kk = 0; kk < N; kk += TILE) {
                const size_t k_end = std::min(kk + TILE, N);
                for (size_t i = ii; i < i_end; ++i) {
                    for (size_t j = jj; j < j_end; ++j) {
                        double sum = C_local[i * N + j];
                        const double* a_row = &A_local[i * N + kk];
                        const double* bt_row = &B_T[j * N + kk];
                        for (size_t k = 0; k < k_end - kk; ++k) {
                            sum += a_row[k] * bt_row[k];
                        }
                        C_local[i * N + j] = sum;
                    }
                }
            }
        }
    }
}

// Simple validation: compute a single element and compare
bool validateResult(const double* A, const double* B,
                    const double* C, const size_t N) {
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

    // Parse command line arguments (all ranks)
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

    // Compute row distribution across ranks
    const size_t base_rows = N / static_cast<size_t>(nprocs);
    const size_t extra = N % static_cast<size_t>(nprocs);
    const size_t my_rows = base_rows + (static_cast<size_t>(rank) < extra ? 1 : 0);
    size_t my_row_start = 0;
    for (int r = 0; r < rank; ++r) {
        my_row_start += base_rows + (static_cast<size_t>(r) < extra ? 1 : 0);
    }

    // Each rank initializes its own rows of A and full B (deterministic, no communication)
    if (rank == 0) printf("Initializing matrices...\n");

    std::vector<double> A_local(my_rows * N);
    std::vector<double> B(N * N);
    std::vector<double> B_T(N * N);
    std::vector<double> C_local(my_rows * N);

    initMatrixRows(A_local.data(), N, my_row_start, my_rows);
    initMatrix(B.data(), N);

    // Transpose B for cache-friendly inner-loop access
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            B_T[j * N + i] = B[i * N + j];
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) printf("Computing matrix multiplication...\n");

    auto start = std::chrono::high_resolution_clock::now();

    matrixMultiplyLocal(A_local.data(), B_T.data(), C_local.data(), my_rows, N);

    // Gather all C rows to rank 0
    std::vector<int> recvcounts(nprocs);
    std::vector<int> displs(nprocs);
    for (int r = 0; r < nprocs; ++r) {
        size_t rows_r = base_rows + (static_cast<size_t>(r) < extra ? 1 : 0);
        recvcounts[r] = static_cast<int>(rows_r * N);
    }
    displs[0] = 0;
    for (int r = 1; r < nprocs; ++r) {
        displs[r] = displs[r - 1] + recvcounts[r - 1];
    }

    std::vector<double> C;
    if (rank == 0) {
        C.resize(N * N);
    }

    MPI_Gatherv(C_local.data(), static_cast<int>(my_rows * N), MPI_DOUBLE,
                C.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    auto end = std::chrono::high_resolution_clock::now();
    long long local_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long long duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &duration_ms, 1, MPI_LONG_LONG_INT, MPI_MAX,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %lld ms\n", duration_ms);

        // Calculate GFLOPS
        double gflops = (2.0 * N * N * N) / (duration_ms / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        // Print results for external validation
        if (printResults) {
            print_results(C, "MatrixC");
        }

        // Validation
        if (validate) {
            printf("Validating result...\n");
            // Reconstruct full A for validation
            std::vector<double> A_full(N * N);
            initMatrix(A_full.data(), N);
            bool valid = validateResult(A_full.data(), B.data(), C.data(), N);

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
