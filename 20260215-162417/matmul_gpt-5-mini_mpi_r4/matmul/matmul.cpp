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

// Initialize a contiguous block of rows [rowStart, rowStart+rows)
void initMatrixBlock(std::vector<double>& matBlock, const size_t N, const size_t rowStart, const size_t rows) {
    for (size_t ii = 0; ii < rows; ++ii) {
        const size_t i = rowStart + ii;
        for (size_t j = 0; j < N; ++j) {
            matBlock[ii * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

void matrixMultiplyBlock(const std::vector<double>& Ablock, const std::vector<double>& B, 
                          std::vector<double>& Cblock, const size_t rows, const size_t N) {
    for (size_t ii = 0; ii < rows; ++ii) {
        for (size_t j = 0; j < N; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < N; ++k) {
                sum += Ablock[ii * N + k] * B[k * N + j];
            }
            Cblock[ii * N + j] = sum;
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
    // Initialize MPI
    MPI_Init(&argc, &argv);
    int world_size = 1;
    int world_rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    size_t N = 512;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments (all ranks parse the same argv)
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
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Determine local row distribution
    std::vector<int> rows_per_rank(world_size, 0);
    std::vector<int> displs(world_size, 0);
    int base = static_cast<int>(N) / world_size;
    int rem = static_cast<int>(N) % world_size;
    for (int r = 0; r < world_size; ++r) {
        rows_per_rank[r] = base + (r < rem ? 1 : 0);
    }
    displs[0] = 0;
    for (int r = 1; r < world_size; ++r) displs[r] = displs[r-1] + rows_per_rank[r-1];

    int local_rows = rows_per_rank[world_rank];
    int local_row_start = displs[world_rank];

    // Allocate local blocks
    std::vector<double> Ablock(static_cast<size_t>(local_rows) * N);
    std::vector<double> B(N * N); // replicate B on all ranks for simplicity
    std::vector<double> Cblock(static_cast<size_t>(local_rows) * N);

    if (local_rows > 0) initMatrixBlock(Ablock, N, static_cast<size_t>(local_row_start), static_cast<size_t>(local_rows));
    // All ranks initialize B deterministically to avoid communication
    initMatrix(B, N);

    if (world_rank == 0) printf("Computing matrix multiplication on %d ranks...\n", world_size);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    // Local multiplication
    if (local_rows > 0) matrixMultiplyBlock(Ablock, B, Cblock, static_cast<size_t>(local_rows), N);

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();
    double local_time = t1 - t0;
    double max_time = 0.0;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather results to root
    std::vector<int> sendcounts(world_size);
    std::vector<int> displs_elems(world_size);
    for (int r = 0; r < world_size; ++r) {
        sendcounts[r] = rows_per_rank[r] * static_cast<int>(N);
        displs_elems[r] = displs[r] * static_cast<int>(N);
    }

    std::vector<double> C;
    if (world_rank == 0) C.resize(static_cast<size_t>(N) * N);

    MPI_Gatherv(Cblock.data(), sendcounts[world_rank], MPI_DOUBLE,
                C.data(), sendcounts.data(), displs_elems.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        long duration_ms = static_cast<long>(max_time * 1000.0);
        printf("Computation time: %ld ms\n", duration_ms);
        double gflops = (2.0 * N * N * N) / (max_time) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) print_results(C, "MatrixC");

        if (validate) {
            // Recreate full A and B for validation
            std::vector<double> Afull(N * N);
            std::vector<double> Bfull(N * N);
            initMatrix(Afull, N);
            initMatrix(Bfull, N);
            printf("Validating result...\n");
            bool valid = validateResult(Afull, Bfull, C, N);
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
