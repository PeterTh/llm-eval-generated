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

void matrixMultiply(const double* A, const double* B, double* C, size_t row_start, size_t row_end, size_t N) {
    for (size_t i = row_start; i < row_end; ++i) {
        for (size_t j = 0; j < N; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
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
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

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
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Calculate row distribution
    size_t rows_per_proc = N / world_size;
    size_t remainder = N % world_size;
    size_t row_start = world_rank * rows_per_proc + (world_rank < remainder ? world_rank : remainder);
    size_t local_rows = rows_per_proc + (world_rank < remainder ? 1 : 0);
    size_t row_end = row_start + local_rows;

    // Allocate local matrices
    std::vector<double> A_local(local_rows * N);
    std::vector<double> C_local(local_rows * N, 0.0);
    std::vector<double> B(N * N);
    std::vector<double> A, C;
    if (world_rank == 0) {
        A.resize(N * N);
        B.resize(N * N);
        C.resize(N * N);
        printf("Initializing matrices...\n");
        initMatrix(A, N);
        initMatrix(B, N);
    }

    // Scatter A
    std::vector<int> sendcounts(world_size), displs(world_size);
    for (int i = 0, disp = 0; i < world_size; ++i) {
        size_t rows = rows_per_proc + (i < remainder ? 1 : 0);
        sendcounts[i] = rows * N;
        displs[i] = disp;
        disp += rows * N;
    }
    MPI_Scatterv(A.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
                 A_local.data(), local_rows * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Broadcast B
    if (world_rank == 0) {
        // B already filled
    }
    MPI_Bcast(B.data(), N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Timing
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    // Local computation
    matrixMultiply(A_local.data(), B.data(), C_local.data(), 0, local_rows, N);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long local_duration_ms = duration.count();
    long max_duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &max_duration_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather C
    MPI_Gatherv(C_local.data(), local_rows * N, MPI_DOUBLE,
                C.data(), sendcounts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        printf("Computation time: %ld ms\n", max_duration_ms);
        double gflops = (2.0 * N * N * N) / (max_duration_ms / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
        if (printResults) {
            print_results(C, "MatrixC");
        }
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(A, B, C, N);
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
