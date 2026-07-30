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

// Initialize a block of rows [start_row, end_row) of matrix
void initMatrixBlock(std::vector<double>& mat, const size_t N, const size_t start_row, const size_t end_row) {
    for (size_t i = start_row; i < end_row; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[(i - start_row) * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

// Cache-optimized matrix multiplication with i-k-j loop order
void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B, 
                    std::vector<double>& C, const size_t N) {
    // Initialize C to zero
    std::fill(C.begin(), C.end(), 0.0);
    
    // i-k-j loop order for better cache locality
    for (size_t i = 0; i < N; ++i) {
        for (size_t k = 0; k < N; ++k) {
            const double a_ik = A[i * N + k];
            for (size_t j = 0; j < N; ++j) {
                C[i * N + j] += a_ik * B[k * N + j];
            }
        }
    }
}

// MPI-parallelized matrix multiplication using 1D block-row decomposition
void matrixMultiplyMPI(const std::vector<double>& A_local, const std::vector<double>& B, 
                       std::vector<double>& C_local, const size_t local_rows, const size_t N) {
    // Initialize C_local to zero
    std::fill(C_local.begin(), C_local.end(), 0.0);
    
    // i-k-j loop order for better cache locality
    for (size_t i = 0; i < local_rows; ++i) {
        for (size_t k = 0; k < N; ++k) {
            const double a_ik = A_local[i * N + k];
            for (size_t j = 0; j < N; ++j) {
                C_local[i * N + j] += a_ik * B[k * N + j];
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

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t N = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (only rank 0 needs to parse, then broadcast)
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

    // Broadcast parameters from rank 0
    int validate_int = validate ? 1 : 0;
    int printRes_int = printResults ? 1 : 0;
    size_t params[3] = {N, static_cast<size_t>(validate_int), static_cast<size_t>(printRes_int)};
    MPI_Bcast(params, 3, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    N = params[0];
    validate = params[1] != 0;
    printResults = params[2] != 0;

    // Compute block-row decomposition: distribute rows of A and C across processes
    // Each process gets a contiguous block of rows
    const size_t total_rows = N;
    const size_t base_rows = total_rows / size;
    const size_t remainder = total_rows % size;

    // Processes 0..remainder-1 get (base_rows+1) rows, the rest get base_rows
    const size_t local_rows = static_cast<size_t>(rank) < remainder ? base_rows + 1 : base_rows;

    // Compute starting global row index for this rank
    size_t start_row = 0;
    if (static_cast<size_t>(rank) < remainder) {
        start_row = static_cast<size_t>(rank) * (base_rows + 1);
    } else {
        start_row = remainder * (base_rows + 1) + (static_cast<size_t>(rank) - remainder) * base_rows;
    }

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark (MPI)\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("MPI processes: %d\n", size);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Each process independently initializes its local rows of A (deterministic formula)
    // and the full B matrix. This avoids communication for initialization.
    std::vector<double> A_local(local_rows * N);
    std::vector<double> B(N * N);
    std::vector<double> C_local(local_rows * N, 0.0);

    // Initialize local block of A
    for (size_t i = 0; i < local_rows; ++i) {
        const size_t global_i = start_row + i;
        for (size_t j = 0; j < N; ++j) {
            A_local[i * N + j] = getPseudoRndValue(N, global_i, j);
        }
    }

    // Initialize full B matrix (same on all processes)
    initMatrix(B, N);

    // Perform distributed matrix multiplication
    if (rank == 0) {
        printf("Computing matrix multiplication...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    matrixMultiplyMPI(A_local, B, C_local, local_rows, N);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();

    // Compute max time across all processes for accurate wall-clock measurement
    double local_elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    double max_elapsed = 0.0;
    MPI_Reduce(&local_elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        long duration_ms = static_cast<long>(max_elapsed);
        printf("Computation time: %ld ms\n", duration_ms);

        // Calculate GFLOPS
        double gflops = (2.0 * N * N * N) / (max_elapsed / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Gather results to rank 0 for print_results and validation
    // Use MPI_Gatherv since rows may be unevenly distributed
    std::vector<int> recvcounts(size);
    std::vector<int> displs(size);
    for (int r = 0; r < size; ++r) {
        size_t r_rows = static_cast<size_t>(r) < remainder ? base_rows + 1 : base_rows;
        recvcounts[r] = static_cast<int>(r_rows * N);
        size_t r_start = 0;
        if (static_cast<size_t>(r) < remainder) {
            r_start = static_cast<size_t>(r) * (base_rows + 1);
        } else {
            r_start = remainder * (base_rows + 1) + (static_cast<size_t>(r) - remainder) * base_rows;
        }
        displs[r] = static_cast<int>(r_start * N);
    }

    std::vector<double> C_full;
    if (printResults || validate) {
        if (rank == 0) {
            C_full.resize(N * N);
        }
        MPI_Gatherv(C_local.data(), static_cast<int>(local_rows * N), MPI_DOUBLE,
                    C_full.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
    }

    // Print results for external validation (rank 0 only)
    if (printResults && rank == 0) {
        print_results(C_full, "MatrixC");
    }

    // Validation (rank 0 only)
    if (validate && rank == 0) {
        printf("Validating result...\n");
        // For validation, we need the full A matrix too
        std::vector<double> A_full(N * N);
        initMatrix(A_full, N);
        bool valid = validateResult(A_full, B, C_full, N);

        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            MPI_Finalize();
            return 1;
        }
    }

    MPI_Finalize();
    return 0;
}
