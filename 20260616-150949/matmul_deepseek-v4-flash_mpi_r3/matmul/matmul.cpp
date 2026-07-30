#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

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

// Cache-friendly blocked matrix multiplication with ikj ordering.
// local_A: local_rows × N, B: N × N, local_C: local_rows × N (output)
void matrixMultiply(const std::vector<double>& local_A, const std::vector<double>& B,
                    std::vector<double>& local_C, const size_t local_rows, const size_t N) {
    constexpr size_t BLOCK = 32;

    std::fill(local_C.begin(), local_C.end(), 0.0);

    for (size_t ii = 0; ii < local_rows; ii += BLOCK) {
        const size_t i_end = std::min(ii + BLOCK, local_rows);
        for (size_t kk = 0; kk < N; kk += BLOCK) {
            const size_t k_end = std::min(kk + BLOCK, N);
            for (size_t jj = 0; jj < N; jj += BLOCK) {
                const size_t j_end = std::min(jj + BLOCK, N);
                for (size_t i = ii; i < i_end; ++i) {
                    for (size_t k = kk; k < k_end; ++k) {
                        const double aik = local_A[i * N + k];
                        for (size_t j = jj; j < j_end; ++j) {
                            local_C[i * N + j] += aik * B[k * N + j];
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

    // Parse options on root, then broadcast
    size_t N = 512;
    bool validate = false;
    bool printResults = false;

    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                N = static_cast<size_t>(atol(argv[++i]));
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = true;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = true;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                MPI_Finalize();
                return 0;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                MPI_Finalize();
                return 1;
            }
        }
    }

    // Broadcast parsed parameters
    int64_t N_i64 = static_cast<int64_t>(N);
    int val_int = validate ? 1 : 0;
    int pr_int = printResults ? 1 : 0;

    MPI_Bcast(&N_i64, 1, MPI_INT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&val_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&pr_int, 1, MPI_INT, 0, MPI_COMM_WORLD);

    N = static_cast<size_t>(N_i64);
    validate = (val_int != 0);
    printResults = (pr_int != 0);

    // Root prints info
    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("MPI processes: %d\n", num_procs);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // ---- Distribute work among processes ----
    // Row-wise decomposition of A and C.
    // Each process gets a contiguous block of rows.

    std::vector<int> sendcounts(num_procs);
    std::vector<int> displs(num_procs);
    std::vector<int> local_rows_arr(num_procs);
    int offset = 0;
    for (int p = 0; p < num_procs; ++p) {
        int rows = static_cast<int>(N) / num_procs;
        if (p < static_cast<int>(N) % num_procs) ++rows;
        local_rows_arr[p] = rows;
        sendcounts[p] = rows * static_cast<int>(N);
        displs[p] = offset;
        offset += sendcounts[p];
    }
    const int local_rows = local_rows_arr[rank];
    const size_t local_rows_s = static_cast<size_t>(local_rows);

    // Allocate matrices
    std::vector<double> A;           // full A on root only
    std::vector<double> B(N * N);    // B on all processes
    std::vector<double> local_A(local_rows_s * N);
    std::vector<double> local_C(local_rows_s * N);

    // Root initializes matrices
    if (rank == 0) {
        if (rank == 0) printf("Initializing matrices...\n");
        A.resize(N * N);
        initMatrix(A, N);
        initMatrix(B, N);
    }

    // Broadcast B to all processes
    MPI_Bcast(B.data(), static_cast<int>(N * N), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Scatter rows of A to all processes
    MPI_Scatterv(
        rank == 0 ? A.data() : nullptr, sendcounts.data(), displs.data(),
        MPI_DOUBLE,
        local_A.data(), sendcounts[rank], MPI_DOUBLE,
        0, MPI_COMM_WORLD);

    // ---- Compute local matrix multiplication ----
    MPI_Barrier(MPI_COMM_WORLD);
    const double start_time = MPI_Wtime();

    matrixMultiply(local_A, B, local_C, local_rows_s, N);

    const double end_time = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);

    // ---- Gather results to root ----
    std::vector<double> C;
    if (rank == 0) {
        C.resize(N * N);
    }
    MPI_Gatherv(
        local_C.data(), sendcounts[rank], MPI_DOUBLE,
        rank == 0 ? C.data() : nullptr, sendcounts.data(), displs.data(), MPI_DOUBLE,
        0, MPI_COMM_WORLD);

    // ---- Root: output, validation ----
    if (rank == 0) {
        const double elapsed = end_time - start_time;
        const long duration_ms = static_cast<long>(elapsed * 1000.0);
        printf("Computation time: %ld ms\n", duration_ms);

        const double gflops = (2.0 * static_cast<double>(N) * N * N) / elapsed / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) {
            print_results(C, "MatrixC");
        }

        if (validate) {
            printf("Validating result...\n");
            const bool valid = validateResult(A, B, C, N);
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
