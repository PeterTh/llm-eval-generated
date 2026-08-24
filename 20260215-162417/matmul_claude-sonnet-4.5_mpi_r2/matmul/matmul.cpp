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
                    std::vector<double>& C, const size_t N, 
                    const size_t startRow, const size_t numRows) {
    for (size_t i = 0; i < numRows; ++i) {
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
            if (rank == 0) {
                printUsage(argv[0]);
            }
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
        printf("MPI processes: %d\n", size);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Distribute rows across processes
    size_t rowsPerProc = N / size;
    size_t remainder = N % size;
    size_t localRows = rowsPerProc + (rank < remainder ? 1 : 0);
    size_t startRow = rank * rowsPerProc + std::min(static_cast<size_t>(rank), remainder);
    
    // Allocate matrices
    std::vector<double> A_full, B_full, C_full;
    if (rank == 0) {
        A_full.resize(N * N);
        B_full.resize(N * N);
        C_full.resize(N * N);
        
        // Initialize matrices on rank 0
        if (rank == 0) {
            printf("Initializing matrices...\n");
        }
        initMatrix(A_full, N);
        initMatrix(B_full, N);
    } else {
        B_full.resize(N * N);
    }
    
    // Local matrices
    std::vector<double> A_local(localRows * N);
    std::vector<double> C_local(localRows * N);
    
    // Compute send counts and displacements for Scatterv/Gatherv
    std::vector<int> sendcounts(size);
    std::vector<int> displs(size);
    for (int i = 0; i < size; ++i) {
        size_t rows = rowsPerProc + (i < remainder ? 1 : 0);
        sendcounts[i] = rows * N;
        size_t start = i * rowsPerProc + std::min(static_cast<size_t>(i), remainder);
        displs[i] = start * N;
    }
    
    // Scatter rows of A
    MPI_Scatterv(A_full.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
                 A_local.data(), localRows * N, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);
    
    // Broadcast B to all processes
    if (rank == 0) {
        B_full.assign(B_full.begin(), B_full.end());
    }
    MPI_Bcast(B_full.data(), N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Synchronize before computation
    MPI_Barrier(MPI_COMM_WORLD);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Perform local matrix multiplication
    if (rank == 0) {
        printf("Computing matrix multiplication...\n");
    }
    matrixMultiply(A_local, B_full, C_local, N, startRow, localRows);
    
    // Gather results
    MPI_Gatherv(C_local.data(), localRows * N, MPI_DOUBLE,
                C_full.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    
    auto end = std::chrono::high_resolution_clock::now();
    long long local_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long long duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &duration_ms, 1, MPI_LONG_LONG_INT, MPI_MAX,
               0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Computation time: %lld ms\n", duration_ms);
        
        // Calculate GFLOPS
        double gflops = (2.0 * N * N * N) / (duration_ms / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
        
        // Print results for external validation
        if (printResults) {
            print_results(C_full, "MatrixC");
        }
        
        // Validation
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(A_full, B_full, C_full, N);
            
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
