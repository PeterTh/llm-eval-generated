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

void initMatrixBlock(std::vector<double>& mat, const size_t N, const size_t start_row, const size_t num_rows) {
    for (size_t i = 0; i < num_rows; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, start_row + i, j);
        }
    }
}

void initMatrix(std::vector<double>& mat, const size_t N) {
    initMatrixBlock(mat, N, 0, N);
}

void matrixMultiplyBlock(const std::vector<double>& local_A, const std::vector<double>& B, 
                         std::vector<double>& local_C, const size_t N, const size_t num_rows) {
    for (size_t i = 0; i < num_rows; ++i) {
        for (size_t j = 0; j < N; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < N; ++k) {
                sum += local_A[i * N + k] * B[k * N + j];
            }
            local_C[i * N + j] = sum;
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
    
    // Parse command line arguments
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
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
    }

    // Broadcast parameters
    MPI_Bcast(&N, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    bool flags[2] = {validate, printResults};
    MPI_Bcast(flags, 2, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
    validate = flags[0];
    printResults = flags[1];
    
    if (rank == 0) {
        printf("Matrix Multiplication Benchmark (MPI)\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Processes: %d\n", size);
    }
    
    // Determine local work
    size_t rows_per_proc = N / size;
    size_t remainder = N % size;
    size_t local_rows = rows_per_proc + (static_cast<size_t>(rank) < remainder ? 1 : 0);
    size_t start_row = rank * rows_per_proc + (static_cast<size_t>(rank) < remainder ? rank : remainder);

    // Prepare counts and displacements for gather/scatter
    std::vector<int> sendcounts(size);
    std::vector<int> displs(size);
    
    if (rank == 0 || validate) { // Validation requires full matrices on rank 0
         size_t current_disp = 0;
         for (int i = 0; i < size; ++i) {
             size_t r_rows = rows_per_proc + (static_cast<size_t>(i) < remainder ? 1 : 0);
             sendcounts[i] = r_rows * N;
             displs[i] = current_disp;
             current_disp += sendcounts[i];
         }
    }

    // Allocate matrices
    // Rank 0 needs full A, B, C for validation/print. 
    // Everyone needs full B.
    // Everyone needs local part of A and C.

    std::vector<double> A_full, C_full;
    if (rank == 0) {
        A_full.resize(N * N);
        C_full.resize(N * N);
    }

    std::vector<double> B(N * N);
    std::vector<double> local_A(local_rows * N);
    std::vector<double> local_C(local_rows * N);
    
    // Initialize matrices
    if (rank == 0) printf("Initializing matrices...\n");
    
    // Parallel initialization of B (all ranks do it to save comms)
    initMatrix(B, N);

    // Parallel initialization of A (each rank inits its own part)
    initMatrixBlock(local_A, N, start_row, local_rows);

    // Also init full A on rank 0 if needed for validation
    if (rank == 0 && validate) {
        initMatrix(A_full, N);
    }
    
    // Perform matrix multiplication
    if (rank == 0) printf("Computing matrix multiplication...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    matrixMultiplyBlock(local_A, B, local_C, N, local_rows);
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    long local_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long max_duration = 0;
    MPI_Reduce(&local_duration, &max_duration, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) printf("Computation time: %ld ms\n", max_duration);
    
    // Gather results
    // Use MPI_Gatherv because row counts might differ
    MPI_Gatherv(local_C.data(), local_rows * N, MPI_DOUBLE,
                C_full.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Calculate GFLOPS
        double gflops = (2.0 * N * N * N) / (max_duration / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
        
        // Print results for external validation
        if (printResults) {
            print_results(C_full, "MatrixC");
        }
        
        // Validation
        if (validate) {
            printf("Validating result...\n");
            // Note: A_full is initialized separately if validate is true
            bool valid = validateResult(A_full, B, C_full, N);
            
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
