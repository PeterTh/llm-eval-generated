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

// Local multiply: A is (local_rows x N), B is (N x N), C is (local_rows x N)
void matrixMultiplyLocal(const std::vector<double>& A_local, const std::vector<double>& B, 
                          std::vector<double>& C_local, const size_t local_rows, const size_t N) {
    for (size_t i = 0; i < local_rows; ++i) {
        for (size_t j = 0; j < N; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < N; ++k) {
                sum += A_local[i * N + k] * B[k * N + j];
            }
            C_local[i * N + j] = sum;
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
    MPI_Init(&argc, &argv);
    int world_size = 1, world_rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    size_t N = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments on all ranks (deterministic)
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            N = static_cast<size_t>(atoi(argv[++i]));
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (world_rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (world_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (world_rank == 0) {
        printf("Matrix Multiplication Benchmark (MPI)\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Processes: %d\n", world_size);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Compute row distribution (block rows)
    std::vector<int> rows_per_rank(world_size);
    size_t base = N / world_size;
    int rem = static_cast<int>(N % world_size);
    for (int r = 0; r < world_size; ++r) {
        rows_per_rank[r] = static_cast<int>(base + (r < rem ? 1 : 0));
    }
    std::vector<int> displs_rows(world_size);
    displs_rows[0] = 0;
    for (int r = 1; r < world_size; ++r) displs_rows[r] = displs_rows[r-1] + rows_per_rank[r-1];

    const int local_rows = rows_per_rank[world_rank];
    const int local_size = local_rows * static_cast<int>(N);

    // Allocate local A and full B and local C
    std::vector<double> A_local(static_cast<size_t>(local_size));
    std::vector<double> B(static_cast<size_t>(N) * N);
    std::vector<double> C_local(static_cast<size_t>(local_size));

    // Initialize local A (only rows assigned to this rank) and full B (locally)
    for (int i = 0; i < local_rows; ++i) {
        const size_t global_i = static_cast<size_t>(displs_rows[world_rank] + i);
        for (size_t j = 0; j < N; ++j) {
            A_local[static_cast<size_t>(i) * N + j] = getPseudoRndValue(N, global_i, j);
        }
    }
    // Each rank initializes full B (deterministic init avoids communication)
    initMatrix(B, N);

    // Synchronize and measure time using MPI_Wtime
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    matrixMultiplyLocal(A_local, B, C_local, static_cast<size_t>(local_rows), N);

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();
    double local_elapsed = t1 - t0;

    double max_elapsed = 0.0;
    MPI_Reduce(&local_elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather results to root
    std::vector<int> recv_counts(world_size);
    std::vector<int> recv_displs(world_size);
    for (int r = 0; r < world_size; ++r) {
        recv_counts[r] = rows_per_rank[r] * static_cast<int>(N);
        recv_displs[r] = displs_rows[r] * static_cast<int>(N);
    }

    std::vector<double> C;
    if (world_rank == 0) {
        C.assign(static_cast<size_t>(N) * N, 0.0);
    }

    MPI_Gatherv(C_local.data(), local_size, MPI_DOUBLE,
                world_rank == 0 ? C.data() : nullptr, recv_counts.data(), recv_displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        // Print timing and performance
        long ms = static_cast<long>(max_elapsed * 1000.0);
        printf("Computation time: %ld ms\n", ms);

        double gflops = (2.0 * N * N * N) / (max_elapsed) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        // Print results if requested
        if (printResults) {
            print_results(C, "MatrixC");
        }

        // Validation if requested
        if (validate) {
            printf("Validating result...\n");
            std::vector<double> A_full(static_cast<size_t>(N) * N);
            std::vector<double> B_full(static_cast<size_t>(N) * N);
            initMatrix(A_full, N);
            initMatrix(B_full, N);
            bool valid = validateResult(A_full, B_full, C, N);
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
