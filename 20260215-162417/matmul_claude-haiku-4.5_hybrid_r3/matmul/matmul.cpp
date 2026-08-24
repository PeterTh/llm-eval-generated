#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

// CUDA kernel for matrix multiplication
extern "C" {
    void cuda_matmul(double* A, double* B, double* C, size_t N, size_t block_start, size_t block_end);
    bool cuda_available();
    void cuda_init();
    void cuda_finalize();
}

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

void initMatrix(std::vector<double>& mat, const size_t N, int rank, int world_size) {
    size_t rows_per_proc = (N + world_size - 1) / world_size;
    size_t start_row = rank * rows_per_proc;
    size_t end_row = std::min(start_row + rows_per_proc, N);
    
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = start_row; i < end_row; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[(i - start_row) * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B, 
                    std::vector<double>& C, const size_t N, int rank, int world_size, 
                    bool use_cuda) {
    size_t rows_per_proc = (N + world_size - 1) / world_size;
    size_t local_rows = std::min(rows_per_proc, N - rank * rows_per_proc);
    
    if (use_cuda && cuda_available()) {
        // Use CUDA for local computation
        cuda_matmul((double*)A.data(), (double*)B.data(), (double*)C.data(), 
                    N, 0, local_rows);
    } else {
        // CPU computation with OpenMP
        #pragma omp parallel for collapse(2) schedule(dynamic, 32)
        for (size_t i = 0; i < local_rows; ++i) {
            for (size_t j = 0; j < N; ++j) {
                double sum = 0.0;
                #pragma omp simd reduction(+:sum)
                for (size_t k = 0; k < N; ++k) {
                    sum += A[i * N + k] * B[k * N + j];
                }
                C[i * N + j] = sum;
            }
        }
    }
}

// Simple validation: compute a single element and compare
bool validateResult(const std::vector<double>& A, const std::vector<double>& B,
                   const std::vector<double>& C, const size_t N, int rank, int world_size) {
    size_t rows_per_proc = (N + world_size - 1) / world_size;
    size_t local_rows = std::min(rows_per_proc, N - rank * rows_per_proc);
    
    // Check a few local positions
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};
    
    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = checkPoints[pi] % local_rows;
            const size_t j = checkPoints[pj] % N;
            
            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += A[i * N + k] * B[k * N + j];
            }
            
            const double actual = C[i * N + j];
            const double relError = std::abs((actual - expected) / (expected + 1e-10));
            
            if (relError > 1e-6) {
                printf("[Rank %d] Validation failed at local (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                       rank, i, j, expected, actual, relError);
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
    
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    // Initialize CUDA if available
    bool use_cuda = false;
    cuda_init();
    if (cuda_available()) {
        use_cuda = true;
        if (rank == 0) printf("CUDA is available, will use GPU acceleration\n");
    }
    
    size_t N = 512;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments (only on rank 0, then broadcast)
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
                cuda_finalize();
                return 0;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                MPI_Finalize();
                cuda_finalize();
                return 1;
            }
        }
    }
    
    // Broadcast parameters
    MPI_Bcast(&N, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Matrix Multiplication Benchmark (Hybrid MPI + OpenMP + CUDA)\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("MPI processes: %d\n", world_size);
        printf("OpenMP threads: %d\n", omp_get_max_threads());
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Calculate local matrix dimensions
    size_t rows_per_proc = (N + world_size - 1) / world_size;
    size_t local_rows = std::min(rows_per_proc, N - rank * rows_per_proc);
    
    // Allocate matrices
    std::vector<double> A(local_rows * N);
    std::vector<double> B(N * N);  // Full B matrix needed for computation
    std::vector<double> C(local_rows * N);
    
    // Initialize matrices
    if (rank == 0) printf("Initializing matrices...\n");
    initMatrix(A, N, rank, world_size);
    initMatrix(B, N, rank, world_size);  // Each process initializes its rows
    
    // Gather full B matrix using MPI_Allgather
    std::vector<double> B_full(N * N);
    MPI_Allgather(B.data(), local_rows * N, MPI_DOUBLE, B_full.data(), local_rows * N, MPI_DOUBLE, MPI_COMM_WORLD);
    
    // Perform matrix multiplication
    if (rank == 0) printf("Computing matrix multiplication...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    matrixMultiply(A, B_full, C, N, rank, world_size, use_cuda);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long long local_duration_ms = duration.count();
    long long global_duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &global_duration_ms, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Computation time: %lld ms\n", global_duration_ms);
        
        // Calculate GFLOPS
        double gflops = (2.0 * N * N * N) / (global_duration_ms / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }
    
    // Gather results to rank 0
    std::vector<double> C_full;
    if (rank == 0) {
        C_full.resize(N * N);
    }
    
    std::vector<int> recvcounts(world_size);
    std::vector<int> displs(world_size);
    for (int i = 0; i < world_size; ++i) {
        size_t rows = std::min(rows_per_proc, N - i * rows_per_proc);
        recvcounts[i] = rows * N;
        displs[i] = i * rows_per_proc * N;
    }
    
    MPI_Gatherv(C.data(), local_rows * N, MPI_DOUBLE, C_full.data(), 
                recvcounts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Print results for external validation (rank 0 only)
    if (rank == 0 && printResults) {
        print_results(C_full, "MatrixC");
    }
    
    // Validation
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        bool valid = validateResult(A, B_full, C, N, rank, world_size);
        
        int all_valid = valid ? 1 : 0;
        MPI_Allreduce(MPI_IN_PLACE, &all_valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        
        if (rank == 0) {
            if (all_valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
    }
    
    // Cleanup
    cuda_finalize();
    MPI_Finalize();
    
    return 0;
}
