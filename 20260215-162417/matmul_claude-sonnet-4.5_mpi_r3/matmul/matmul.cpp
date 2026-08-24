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

// Parallel matrix multiply using MPI row-block decomposition
void matrixMultiplyMPI(const std::vector<double>& A_local, const std::vector<double>& B, 
                       std::vector<double>& C_local, const size_t local_rows, const size_t N) {
    // Each process computes its assigned rows of C = A * B
    // A_local contains local_rows x N elements
    // B is the full N x N matrix (broadcasted to all processes)
    // C_local will contain local_rows x N elements
    
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
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    size_t N = 512;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments (only rank 0)
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
    
    // Broadcast parameters to all processes
    MPI_Bcast(&N, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Matrix Multiplication Benchmark (MPI)\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("MPI processes: %d\n", size);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Determine row distribution
    size_t rows_per_proc = N / static_cast<size_t>(size);
    size_t remainder = N % static_cast<size_t>(size);
    
    // Calculate local rows for this process
    size_t local_rows = rows_per_proc + (static_cast<size_t>(rank) < remainder ? 1 : 0);
    
    // Allocate local portions of A and C
    std::vector<double> A_local(local_rows * N);
    std::vector<double> C_local(local_rows * N);
    
    // Allocate full matrix B (needed by all processes)
    std::vector<double> B(N * N);
    
    // Full matrices for rank 0 (for initialization and gathering results)
    std::vector<double> A, C;
    if (rank == 0) {
        A.resize(N * N);
        C.resize(N * N);
    }
    
    // Initialize matrices on rank 0
    if (rank == 0) {
        if (rank == 0) printf("Initializing matrices...\n");
        initMatrix(A, N);
        initMatrix(B, N);
    }
    
    // Broadcast matrix B to all processes
    MPI_Bcast(B.data(), N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Scatter rows of A to all processes
    std::vector<int> sendcounts(size);
    std::vector<int> displs(size);
    
    for (int i = 0; i < size; ++i) {
        size_t proc_rows = rows_per_proc + (static_cast<size_t>(i) < remainder ? 1 : 0);
        sendcounts[i] = static_cast<int>(proc_rows * N);
        size_t proc_offset = static_cast<size_t>(i) * rows_per_proc + (static_cast<size_t>(i) < remainder ? static_cast<size_t>(i) : remainder);
        displs[i] = static_cast<int>(proc_offset * N);
    }
    
    MPI_Scatterv(A.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
                 A_local.data(), static_cast<int>(local_rows * N), MPI_DOUBLE,
                 0, MPI_COMM_WORLD);
    
    // Synchronize before timing
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Perform matrix multiplication
    if (rank == 0) {
        printf("Computing matrix multiplication...\n");
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    matrixMultiplyMPI(A_local, B, C_local, local_rows, N);
    
    // Synchronize after computation
    MPI_Barrier(MPI_COMM_WORLD);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long long local_duration_ms = static_cast<long long>(duration.count());
    long long max_duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &max_duration_ms, 1, MPI_LONG_LONG_INT,
               MPI_MAX, 0, MPI_COMM_WORLD);
    
    // Gather results back to rank 0
    MPI_Gatherv(C_local.data(), static_cast<int>(local_rows * N), MPI_DOUBLE,
                C.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Computation time: %lld ms\n", max_duration_ms);
        
        // Calculate GFLOPS
        double gflops = (2.0 * N * N * N) / (max_duration_ms / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
        
        // Print results for external validation
        if (printResults) {
            print_results(C, "MatrixC");
        }
        
        // Validation
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(A, B, C, N);
            
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
