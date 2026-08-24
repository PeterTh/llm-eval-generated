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

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B, 
                    std::vector<double>& C, const size_t N) {
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

void matrixMultiplyLocal(const std::vector<double>& A_local, const std::vector<double>& B, 
                         std::vector<double>& C_local, const size_t N, const size_t local_rows) {
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
    
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    
    size_t N = 512;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments (only on rank 0)
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
    
    // Broadcast matrix size to all processes
    MPI_Bcast(&N, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Number of MPI processes: %d\n", num_procs);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Allocate full matrices on all processes for initialization
    std::vector<double> A(N * N);
    std::vector<double> B(N * N);
    
    // Calculate local rows per process
    size_t rows_per_proc = N / num_procs;
    size_t extra_rows = N % num_procs;
    size_t local_rows = rows_per_proc + (rank < (int)extra_rows ? 1 : 0);
    
    // Initialize matrices on all processes (same deterministic values)
    if (rank == 0) {
        printf("Initializing matrices...\n");
    }
    initMatrix(A, N);
    initMatrix(B, N);
    
    // Broadcast B to all processes
    MPI_Bcast(B.data(), N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Allocate local matrices
    std::vector<double> A_local(local_rows * N);
    std::vector<double> C_local(local_rows * N);
    
    // Send A rows to each process using scatterv
    std::vector<int> sendcounts(num_procs);
    std::vector<int> displs(num_procs);
    
    if (rank == 0) {
        size_t cur_row = 0;
        for (int i = 0; i < num_procs; ++i) {
            size_t proc_rows = rows_per_proc + (i < (int)extra_rows ? 1 : 0);
            sendcounts[i] = proc_rows * N;
            displs[i] = cur_row * N;
            cur_row += proc_rows;
        }
    }
    
    // Scatterv A
    MPI_Scatterv(rank == 0 ? A.data() : nullptr, 
                 sendcounts.data(), 
                 displs.data(), 
                 MPI_DOUBLE,
                 A_local.data(), 
                 local_rows * N, 
                 MPI_DOUBLE,
                 0, 
                 MPI_COMM_WORLD);
    
    // Perform local matrix multiplication
    if (rank == 0) {
        printf("Computing matrix multiplication...\n");
    }
    auto start = std::chrono::high_resolution_clock::now();
    
    matrixMultiplyLocal(A_local, B, C_local, N, local_rows);
    
    auto end = std::chrono::high_resolution_clock::now();
    long local_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long max_duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &max_duration_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    
    // Allocate full C matrix on rank 0 for gathering
    std::vector<double> C(N * N);
    
    // Gatherv C from all processes
    MPI_Gatherv(C_local.data(), 
                local_rows * N, 
                MPI_DOUBLE,
                rank == 0 ? C.data() : nullptr,
                sendcounts.data(),
                displs.data(),
                MPI_DOUBLE,
                0,
                MPI_COMM_WORLD);
    
    // Only rank 0 reports timing and performs validation
    if (rank == 0) {
        printf("Computation time: %ld ms\n", max_duration_ms);
        
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
            } else {
                printf("Validation: FAILED\n");
                MPI_Finalize();
                return 1;
            }
        }
    }
    
    // Finalize MPI
    MPI_Finalize();
    return 0;
}
