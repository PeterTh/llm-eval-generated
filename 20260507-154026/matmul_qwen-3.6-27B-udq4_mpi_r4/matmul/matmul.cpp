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

// Initialize a local block of rows of a matrix (row_start .. row_start+local_M)
void initMatrixLocal(std::vector<double>& mat, const size_t N,
                     const size_t row_start, const size_t local_M) {
    for (size_t i = row_start; i < row_start + local_M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[(i - row_start) * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

// Initialize a full N x N matrix
void initMatrixFull(std::vector<double>& mat, const size_t N) {
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

// Tiled matrix multiply: C[local_M x N] = A[local_M x N] * B[N x N]
// Uses i-j-k loop order with blocking for cache efficiency.
void matrixMultiplyTiled(const std::vector<double>& A, const std::vector<double>& B,
                         std::vector<double>& C, const size_t local_M, const size_t N) {
    constexpr size_t BLOCK = 64;

    std::fill(C.begin(), C.end(), 0.0);

    for (size_t i = 0; i < local_M; i += BLOCK) {
        const size_t i_end = std::min(i + BLOCK, local_M);
        for (size_t j = 0; j < N; j += BLOCK) {
            const size_t j_end = std::min(j + BLOCK, N);
            for (size_t k = 0; k < N; k += BLOCK) {
                const size_t k_end = std::min(k + BLOCK, N);
                for (size_t ii = i; ii < i_end; ++ii) {
                    for (size_t kk = k; kk < k_end; ++kk) {
                        const double a_ik = A[ii * N + kk];
                        for (size_t jj = j; jj < j_end; ++jj) {
                            C[ii * N + jj] += a_ik * B[kk * N + jj];
                        }
                    }
                }
            }
        }
    }
}

// Simple validation: compute a few elements and compare
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

    int rank, num_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

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
            if (rank == 0) {
                printUsage(argv[0]);
            }
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
        printf("Matrix Multiplication Benchmark (MPI)\n");
        printf("MPI ranks: %d\n", num_ranks);
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Compute row-block distribution
    const size_t rows_per_rank = N / static_cast<size_t>(num_ranks);
    const size_t remainder = N % static_cast<size_t>(num_ranks);
    const size_t local_M = rows_per_rank + (rank < static_cast<int>(remainder) ? 1 : 0);
    const size_t row_start =
        static_cast<size_t>(rank) * rows_per_rank + std::min(static_cast<size_t>(rank), remainder);

    // Allocate: A and C are row-distributed (local_M x N), B is replicated (N x N)
    std::vector<double> A(local_M * N);
    std::vector<double> B(N * N);
    std::vector<double> C(local_M * N);

    // Initialize matrices
    if (rank == 0) printf("Initializing matrices...\n");

    // Each rank initializes its local rows of A independently
    initMatrixLocal(A, N, row_start, local_M);

    // Each rank initializes full B independently (deterministic, identical on all ranks)
    initMatrixFull(B, N);

    // Synchronize before timing
    MPI_Barrier(MPI_COMM_WORLD);

    // Perform tiled matrix multiplication on local rows
    if (rank == 0) printf("Computing matrix multiplication...\n");
    auto start = std::chrono::high_resolution_clock::now();

    matrixMultiplyTiled(A, B, C, local_M, N);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Precompute Gatherv counts/displacements for C (and A if needed)
    std::vector<int> counts(num_ranks);
    std::vector<int> displs(num_ranks);
    int offset = 0;
    for (int r = 0; r < num_ranks; ++r) {
        const size_t r_local_M = rows_per_rank + (r < static_cast<int>(remainder) ? 1 : 0);
        counts[r] = static_cast<int>(r_local_M * N);
        displs[r] = offset;
        offset += counts[r];
    }

    // Gather C to rank 0
    std::vector<double> C_full(N * N);
    MPI_Gatherv(C.data(), static_cast<int>(local_M * N), MPI_DOUBLE,
                C_full.data(), counts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    std::vector<double> A_full;
    if (validate && rank == 0) A_full.resize(N * N);

    if (validate) {
        // Collective gather of A (all ranks must participate)
        MPI_Gatherv(A.data(), static_cast<int>(local_M * N), MPI_DOUBLE,
                    rank == 0 ? A_full.data() : nullptr,
                    counts.data(), displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate GFLOPS (total operations / wall-clock time)
        double gflops = (2.0 * N * N * N) / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        // Print results for external validation
        if (printResults) {
            print_results(C_full, "MatrixC");
        }

        // Validation
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(A_full, B, C_full, N);

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
