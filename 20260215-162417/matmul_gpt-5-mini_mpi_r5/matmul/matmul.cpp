#include <mpi.h>
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

void initMatrix(std::vector<double>& mat, const size_t N) {
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

// Local multiplication: A (local_rows x N), B (N x N), C (local_rows x N)
void matrixMultiplyLocal(const std::vector<double>& A_local, const std::vector<double>& B,
                         std::vector<double>& C_local, const size_t local_rows, const size_t N) {
    // Initialize C_local to zero
    std::fill(C_local.begin(), C_local.end(), 0.0);

    for (size_t i = 0; i < local_rows; ++i) {
        const size_t a_row_offset = i * N;
        const size_t c_row_offset = i * N;
        for (size_t k = 0; k < N; ++k) {
            const double a_val = A_local[a_row_offset + k];
            const size_t b_row_offset = k * N;
            // iterate j inner for contiguous access to B's row
            for (size_t j = 0; j < N; ++j) {
                C_local[c_row_offset + j] += a_val * B[b_row_offset + j];
            }
        }
    }
}

// Simple validation: compute a few elements and compare
bool validateResult(const std::vector<double>& A, const std::vector<double>& B,
                   const std::vector<double>& C, const size_t N) {
    // Check a few deterministic positions
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
    if (progName) printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    // Initialize MPI unconditionally
    MPI_Init(&argc, &argv);
    int rank = 0, world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    size_t N = 512;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments (only on rank 0 then broadcast)
    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                N = atoi(argv[++i]);
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

    // Broadcast N, validate, printResults to all ranks
    MPI_Bcast(&N, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    int vflag = validate ? 1 : 0;
    int rflag = printResults ? 1 : 0;
    MPI_Bcast(&vflag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&rflag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    validate = (vflag != 0);
    printResults = (rflag != 0);

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark (MPI)\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("MPI ranks: %d\n", world_size);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Determine row distribution
    size_t rows_per_rank = N / world_size;
    size_t remainder = N % world_size;
    size_t local_rows = rows_per_rank + (static_cast<size_t>(rank) < remainder ? 1 : 0);
    size_t row_offset = rows_per_rank * static_cast<size_t>(rank) + std::min(static_cast<size_t>(rank), remainder);

    // Allocate local A and full B and local C
    std::vector<double> A_local(local_rows * N);
    std::vector<double> B(N * N);
    std::vector<double> C_local(local_rows * N);

    // Initialize B fully on all ranks (deterministic RNG)
    initMatrix(B, N);

    // Initialize local A using global indices so values match a full initialization
    for (size_t i = 0; i < local_rows; ++i) {
        const size_t global_i = row_offset + i;
        for (size_t j = 0; j < N; ++j) {
            A_local[i * N + j] = getPseudoRndValue(N, global_i, j);
        }
    }

    if (rank == 0) printf("Computing matrix multiplication (distributed)...\n");

    // Synchronize and time the parallel multiplication
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    matrixMultiplyLocal(A_local, B, C_local, local_rows, N);

    double local_elapsed = MPI_Wtime() - t0;
    double max_elapsed = 0.0;
    MPI_Reduce(&local_elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather results to rank 0 if needed for printing/validation
    std::vector<double> C;
    std::vector<int> recvcounts(world_size);
    std::vector<int> displs(world_size);
    for (int r = 0; r < world_size; ++r) {
        size_t r_rows = rows_per_rank + (static_cast<size_t>(r) < remainder ? 1 : 0);
        recvcounts[r] = static_cast<int>(r_rows * N);
        displs[r] = static_cast<int>((rows_per_rank * static_cast<size_t>(r) + std::min(static_cast<size_t>(r), remainder)) * N);
    }

    if (rank == 0) {
        C.resize(N * N);
    }

    MPI_Gatherv(C_local.data(), static_cast<int>(local_rows * N), MPI_DOUBLE,
                rank == 0 ? C.data() : nullptr, recvcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Print timing and performance
        printf("Computation time (max across ranks): %.3f ms\n", max_elapsed * 1000.0);
        double gflops = (2.0 * static_cast<double>(N) * static_cast<double>(N) * static_cast<double>(N)) / (max_elapsed) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        // Print results for external validation
        if (printResults) {
            print_results(C, "MatrixC");
        }

        // Validation
        if (validate) {
            printf("Validating result...\n");
            // Recreate full A and B locally and validate
            std::vector<double> A_full(N * N);
            std::vector<double> B_full(N * N);
            initMatrix(A_full, N);
            initMatrix(B_full, N);
            bool valid = validateResult(A_full, B_full, C, N);
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
