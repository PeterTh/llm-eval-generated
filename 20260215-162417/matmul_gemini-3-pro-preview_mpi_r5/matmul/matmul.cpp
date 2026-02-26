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

// Initialize local part of A
void initMatrixA(std::vector<double>& mat, const size_t N, const size_t row_start, const size_t row_count) {
    for (size_t i = 0; i < row_count; ++i) {
        size_t global_i = row_start + i;
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, global_i, j);
        }
    }
}

// Initialize full B
void initMatrixB(std::vector<double>& mat, const size_t N) {
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B, 
                    std::vector<double>& C, const size_t N, const size_t rows_count) {
    std::fill(C.begin(), C.end(), 0.0);
    
    // Tiling optimization
    const size_t BLOCK_SIZE = 64;

    for (size_t i = 0; i < rows_count; ++i) {
        double* C_row = &C[i * N];
        const double* A_row = &A[i * N];
        
        for (size_t jj = 0; jj < N; jj += BLOCK_SIZE) {
            size_t j_max = std::min(jj + BLOCK_SIZE, N);
            for (size_t k = 0; k < N; ++k) {
                double a_val = A_row[k];
                const double* B_row = &B[k * N];
                
                // Vectorizable inner loop
                for (size_t j = jj; j < j_max; ++j) {
                    C_row[j] += a_val * B_row[j];
                }
            }
        }
    }
}

// Local validation
bool validateResultLocal(const std::vector<double>& local_A, const std::vector<double>& B,
                        const std::vector<double>& local_C, const size_t N, 
                        const size_t row_start, const size_t row_count) {
    // Check a few random positions within the local range
    // We check 5 points per process to be thorough
    
    bool local_valid = true;
    
    // Deterministic checks based on row index
    for (size_t r = 0; r < 5; ++r) {
        // Pick a row in local range
        size_t local_i = (r * (row_count / 5)) % row_count;
        size_t global_i = row_start + local_i;
        
        // Pick a column
        size_t j = (global_i + r * 17) % N;
        
        double expected = 0.0;
        for (size_t k = 0; k < N; ++k) {
            expected += local_A[local_i * N + k] * B[k * N + j];
        }
        
        const double actual = local_C[local_i * N + j];
        double denom = std::abs(expected);
        if (denom < 1e-10) denom = 1.0;

        const double relError = std::abs(actual - expected) / denom;
        
        if (relError > 1e-6) {
            printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                   global_i, j, expected, actual, relError);
            local_valid = false;
            break; // Report first error
        }
    }
    
    return local_valid;
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
        printf("Matrix Multiplication Benchmark (MPI Parallel)\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Processes: %d\n", size);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Workload distribution
    std::vector<int> sendcounts(size);
    std::vector<int> displs(size);
    
    size_t rows_per_proc = N / size;
    size_t remainder = N % size;
    
    size_t current_disp = 0;
    for (int i = 0; i < size; ++i) {
        sendcounts[i] = (i < (int)remainder) ? (rows_per_proc + 1) * N : rows_per_proc * N;
        displs[i] = current_disp;
        current_disp += sendcounts[i];
    }
    
    int local_rows_count = sendcounts[rank] / N;
    size_t local_N_size = sendcounts[rank];
    size_t row_start = displs[rank] / N;

    // Allocate matrices
    // All ranks allocate B (replicated)
    std::vector<double> B(N * N);
    
    // Local buffers
    std::vector<double> local_A(local_N_size);
    std::vector<double> local_C(local_N_size);

    // Initialization
    if (rank == 0) printf("Initializing matrices...\n");
    
    // Parallel Initialization
    // Everyone inits B (redundant computation to save Bcast bandwidth/latency)
    initMatrixB(B, N);
    
    // Everyone inits their part of A
    initMatrixA(local_A, N, row_start, local_rows_count);
    
    // Perform matrix multiplication
    if (rank == 0) printf("Computing matrix multiplication...\n");
    
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();
    
    matrixMultiply(local_A, B, local_C, N, local_rows_count);
    
    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();
    double duration_sec = end_time - start_time;
    
    if (rank == 0) {
        printf("Computation time: %.0f ms\n", duration_sec * 1000.0);
        double gflops = (2.0 * N * N * N) / duration_sec / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Validation
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        bool local_valid = validateResultLocal(local_A, B, local_C, N, row_start, local_rows_count);
        
        int local_valid_int = local_valid ? 1 : 0;
        int global_valid_int = 0;
        
        MPI_Reduce(&local_valid_int, &global_valid_int, 1, MPI_INT, MPI_LAND, 0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            if (global_valid_int) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
    }

    // Print results for external validation (gather only if needed)
    if (printResults) {
        std::vector<double> C;
        if (rank == 0) C.resize(N * N);
        
        MPI_Gatherv(local_C.data(), local_N_size, MPI_DOUBLE,
                    C.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
                    
        if (rank == 0) {
            print_results(C, "MatrixC");
        }
    }
    
    MPI_Finalize();
    return 0;
}

