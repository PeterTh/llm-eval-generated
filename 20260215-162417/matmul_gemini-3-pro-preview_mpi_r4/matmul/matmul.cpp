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

// Local initialization of matrix chunk
void initLocalMatrixA(std::vector<double>& local_A, const size_t N, const size_t row_start, const size_t num_rows) {
    for (size_t i = 0; i < num_rows; ++i) {
        size_t global_i = row_start + i;
        for (size_t j = 0; j < N; ++j) {
            local_A[i * N + j] = getPseudoRndValue(N, global_i, j);
        }
    }
}

// Initialize full matrix B (replicated on all nodes)
void initMatrixB(std::vector<double>& B, const size_t N) {
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            B[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

// Compute C = A * B locally
void matrixMultiplyLocal(const std::vector<double>& local_A, const std::vector<double>& B, 
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
        printf("Matrix Multiplication Benchmark (MPI)\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Processes: %d\n", size);
    }

    // Determine local rows for each process
    size_t base_rows = N / size;
    size_t extra_rows = N % size;
    size_t local_rows = base_rows + (static_cast<size_t>(rank) < extra_rows ? 1 : 0);
    
    // Calculate displacements and counts for Gatherv (needed by rank 0 for gathering)
    std::vector<int> counts(size);
    std::vector<int> displs(size);
    size_t current_disp = 0;
    size_t my_row_start = 0;
    
    for (int i = 0; i < size; ++i) {
        size_t rows = base_rows + (static_cast<size_t>(i) < extra_rows ? 1 : 0);
        counts[i] = rows * N;
        displs[i] = current_disp;
        if (i == rank) my_row_start = current_disp / N;
        current_disp += rows * N;
    }

    // Allocate matrices
    // All processes need full B and local chunks of A and C
    std::vector<double> B(N * N);
    std::vector<double> local_A(local_rows * N);
    std::vector<double> local_C(local_rows * N);
    
    // Initialize matrices
    if (rank == 0) printf("Initializing matrices...\n");
    
    // Initialize B on all nodes (replicated) - avoids broadcast
    initMatrixB(B, N);
    
    // Initialize local part of A on all nodes - avoids scatter
    initLocalMatrixA(local_A, N, my_row_start, local_rows);
    
    MPI_Barrier(MPI_COMM_WORLD);

    // Perform matrix multiplication
    if (rank == 0) printf("Computing matrix multiplication...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    matrixMultiplyLocal(local_A, B, local_C, N, local_rows);
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Gather results to Rank 0
    std::vector<double> C;
    if (rank == 0) {
        C.resize(N * N);
    }
    
    // Use MPI_Gatherv to collect variable sized chunks
    MPI_Gatherv(local_C.data(), local_rows * N, MPI_DOUBLE,
                C.data(), counts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        
        // Calculate GFLOPS
        double gflops = (2.0 * N * N * N) / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
        
        // Print results for external validation
        if (printResults) {
            print_results(C, "MatrixC");
        }
        
        // Validation
        if (validate) {
            printf("Validating result...\n");
            // Need full A to validate. Re-generate it on rank 0.
            std::vector<double> A(N * N);
            initLocalMatrixA(A, N, 0, N);
            
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
