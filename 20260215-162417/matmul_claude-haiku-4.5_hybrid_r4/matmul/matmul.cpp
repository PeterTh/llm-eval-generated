#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

// Forward declare CUDA function if available
#ifdef USE_CUDA
extern "C" void matmulGPU(const double* h_A, const double* h_B, double* h_C,
                         size_t N, size_t rowStart, size_t rowEnd);
#endif

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

void matrixMultiplyOpenMP(const double* A, const double* B, double* C, 
                         size_t N, size_t rowStart, size_t rowEnd) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = rowStart; i < rowEnd; ++i) {
        for (size_t j = 0; j < N; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B, 
                    std::vector<double>& C, const size_t N, 
                    int rank, int size) {
    size_t rows_per_rank = N / size;
    size_t extra_rows = N % size;
    
    size_t local_rows = rows_per_rank + (rank < static_cast<int>(extra_rows) ? 1 : 0);
    size_t row_start = rank * rows_per_rank + std::min(static_cast<size_t>(rank), extra_rows);
    size_t row_end = row_start + local_rows;
    
    // Broadcast B to all ranks
    MPI_Bcast(const_cast<double*>(B.data()), N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Try CUDA if available
    #ifdef USE_CUDA
    matmulGPU(A.data(), B.data(), C.data(), N, row_start, row_end);
    #else
    // Fallback to OpenMP
    matrixMultiplyOpenMP(A.data(), B.data(), C.data(), N, row_start, row_end);
    #endif
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
        printf("Matrix Multiplication Benchmark (Hybrid MPI+OpenMP+CUDA)\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Number of MPI processes: %d\n", size);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Allocate matrices
    std::vector<double> A(N * N);
    std::vector<double> B(N * N);
    std::vector<double> C(N * N);
    
    // Initialize matrices (only rank 0 initializes, then broadcast)
    if (rank == 0) {
        printf("Initializing matrices...\n");
        initMatrix(A, N);
        initMatrix(B, N);
    }
    
    // Broadcast A and B to all ranks
    MPI_Bcast(A.data(), N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(B.data(), N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Perform matrix multiplication
    if (rank == 0) printf("Computing matrix multiplication...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    matrixMultiply(A, B, C, N, rank, size);
    
    // Gather results from all ranks
    std::vector<double> global_C;
    if (rank == 0) {
        global_C.resize(N * N);
    }
    
    std::vector<int> recvCounts(size);
    std::vector<int> displs(size);
    for (int i = 0; i < size; ++i) {
        size_t rows_per_rank = N / size;
        size_t extra_rows = N % size;
        size_t local_rows = rows_per_rank + (i < static_cast<int>(extra_rows) ? 1 : 0);
        recvCounts[i] = local_rows * N;
        displs[i] = (i * rows_per_rank + std::min(static_cast<size_t>(i), extra_rows)) * N;
    }
    
    // Calculate local chunk size
    size_t rows_per_rank = N / size;
    size_t extra_rows = N % size;
    size_t local_rows = rows_per_rank + (rank < static_cast<int>(extra_rows) ? 1 : 0);
    size_t row_start = rank * rows_per_rank + std::min(static_cast<size_t>(rank), extra_rows);
    
    MPI_Gatherv(C.data() + row_start * N, local_rows * N, MPI_DOUBLE,
                 global_C.data(), recvCounts.data(), displs.data(), MPI_DOUBLE,
                 0, MPI_COMM_WORLD);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        
        // Calculate GFLOPS
        double gflops = (2.0 * N * N * N) / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
        
        // Print results for external validation
        if (printResults) {
            print_results(global_C, "MatrixC");
        }
        
        // Validation
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(A, B, global_C, N);
            
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
