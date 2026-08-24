#include <mpi.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

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

// Cache-optimized tiled matrix multiply for local rows
void matrixMultiplyLocal(const double* A_local, const double* B,
                         double* C_local, const size_t local_rows, const size_t N) {
    std::memset(C_local, 0, local_rows * N * sizeof(double));

    constexpr size_t TILE = 64;
    for (size_t ii = 0; ii < local_rows; ii += TILE) {
        const size_t i_end = std::min(ii + TILE, local_rows);
        for (size_t kk = 0; kk < N; kk += TILE) {
            const size_t k_end = std::min(kk + TILE, N);
            for (size_t jj = 0; jj < N; jj += TILE) {
                const size_t j_end = std::min(jj + TILE, N);
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

// Validation using getPseudoRndValue to recompute A elements on the fly
bool validateResult(const double* B, const double* C, const size_t N) {
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

    // Compute row distribution across ranks
    const size_t base_rows = N / static_cast<size_t>(nprocs);
    const size_t remainder = N % static_cast<size_t>(nprocs);
    const size_t local_rows = base_rows + (static_cast<size_t>(rank) < remainder ? 1 : 0);
    const size_t row_start = base_rows * static_cast<size_t>(rank) +
                             std::min(static_cast<size_t>(rank), remainder);

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("MPI processes: %d\n", nprocs);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing matrices...\n");
    }

    // Each rank initializes its own rows of A (no communication needed)
    std::vector<double> A_local(local_rows * N);
    initMatrixRows(A_local.data(), N, row_start, local_rows);

    // Each rank initializes full B locally (deterministic formula, no communication)
    std::vector<double> B(N * N);
    initMatrix(B.data(), N);

    // Allocate local C
    std::vector<double> C_local(local_rows * N);

    if (rank == 0) {
        printf("Computing matrix multiplication...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    matrixMultiplyLocal(A_local.data(), B.data(), C_local.data(), local_rows, N);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long local_duration_ms = static_cast<long>(duration.count());
    long max_duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &max_duration_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather C on rank 0 for output and validation
    std::vector<double> C;
    std::vector<int> recvcounts(nprocs);
    std::vector<int> displs(nprocs);

    for (int p = 0; p < nprocs; ++p) {
        size_t p_rows = base_rows + (static_cast<size_t>(p) < remainder ? 1 : 0);
        recvcounts[p] = static_cast<int>(p_rows * N);
        displs[p] = (p == 0) ? 0 : displs[p - 1] + recvcounts[p - 1];
    }

    if (rank == 0) {
        C.resize(N * N);
    }

    MPI_Gatherv(C_local.data(), static_cast<int>(local_rows * N), MPI_DOUBLE,
                C.data(), recvcounts.data(), displs.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", max_duration_ms);

        double gflops = (2.0 * N * N * N) / (max_duration_ms / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        // Print results for external validation
        if (printResults) {
            print_results(C, "MatrixC");
        }

        // Validation
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(B.data(), C.data(), N);

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
