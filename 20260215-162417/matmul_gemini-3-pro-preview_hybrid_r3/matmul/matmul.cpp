#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "../common/results_output.hpp"

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s)\n", \
                    __FILE__, __LINE__, err, cudaGetErrorString(err)); \
            MPI_Abort(MPI_COMM_WORLD, 1); \
        } \
    } while (0)

// cuBLAS error checking macro
#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS error at %s:%d\n", __FILE__, __LINE__); \
            MPI_Abort(MPI_COMM_WORLD, 1); \
        } \
    } while (0)

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

void initMatrix(std::vector<double>& mat, const size_t N, const size_t rows, const size_t start_row) {
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, start_row + i, j);
        }
    }
}

// Simple validation: compute a single element and compare
bool validateResult(const std::vector<double>& local_C, const size_t N, const size_t rank, const size_t size, const size_t local_rows) {
    // Check 5 random points per rank
    constexpr size_t num_checks = 5;
    bool all_passed = true;
    
    // We need B on host for validation. Each rank should have a copy of B.
    // However, A is distributed. We only validate C rows that we own.
    
    size_t start_row = rank * local_rows;

    for (size_t k = 0; k < num_checks; ++k) {
        // Pick a random local row and a random global col
        size_t local_i = (k * 17 + rank * 13) % local_rows;
        size_t j = (k * 23 + rank * 7) % N;
        size_t global_i = start_row + local_i;
        
        // Compute expected value
        double expected = 0.0;
        // Parallel reduction on CPU
        #pragma omp parallel for reduction(+:expected)
        for (size_t l = 0; l < N; ++l) {
            double a_val = getPseudoRndValue(N, global_i, l);
            double b_val = getPseudoRndValue(N, l, j); 
            expected += a_val * b_val;
        }
        
        double actual = local_C[local_i * N + j];
        double relError = std::abs((actual - expected) / (expected + 1e-10));
        
        if (relError > 1e-6) {
            printf("Rank %zu: Validation failed at local (%zu, %zu) global (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                   rank, local_i, j, global_i, j, expected, actual, relError);
            all_passed = false;
        }
    }
    
    int local_result = all_passed ? 1 : 0;
    int global_result = 0;
    MPI_Allreduce(&local_result, &global_result, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    
    return global_result == 1;
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
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) printf("Warning: MPI thread support insufficient\n");
    }

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
        printf("MPI Ranks: %d\n", size);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Assign GPU to rank
    int num_devices;
    cudaGetDeviceCount(&num_devices);
    if (num_devices > 0) {
        cudaSetDevice(rank % num_devices);
    } else {
        if (rank == 0) fprintf(stderr, "Error: No CUDA devices found.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    // 1D Row Decomposition
    if (N % size != 0) {
        if (rank == 0) printf("Error: N must be divisible by MPI size for this benchmark.\n");
        MPI_Finalize();
        return 1;
    }
    
    size_t rows_per_rank = N / size;
    size_t start_row = rank * rows_per_rank;
    
    // Allocate host memory
    std::vector<double> local_A(rows_per_rank * N);
    std::vector<double> B(N * N);
    std::vector<double> local_C(rows_per_rank * N);
    
    // Initialize matrices
    if (rank == 0) printf("Initializing matrices...\n");
    
    // Initialize local part of A
    initMatrix(local_A, N, rows_per_rank, start_row);
    
    // Initialize full B on all ranks (replicated)
    initMatrix(B, N, N, 0);
    
    // Device memory
    double *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, rows_per_rank * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_B, N * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_C, rows_per_rank * N * sizeof(double)));
    
    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_A, local_A.data(), rows_per_rank * N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), N * N * sizeof(double), cudaMemcpyHostToDevice));
    
    // Create cuBLAS handle
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) printf("Computing matrix multiplication...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    double alpha = 1.0;
    double beta = 0.0;
    
    // Perform multiplication C = A * B
    // Using trick C^T = B^T * A^T for column-major cuBLAS
    CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             N, rows_per_rank, N,
                             &alpha,
                             d_B, N,          // "A" in gemm call (actually B^T)
                             d_A, N,          // "B" in gemm call (actually A^T)
                             &beta,
                             d_C, N));        // "C" in gemm call (actually C^T)
                             
    CUDA_CHECK(cudaDeviceSynchronize());
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    
    // Copy result back
    CUDA_CHECK(cudaMemcpy(local_C.data(), d_C, rows_per_rank * N * sizeof(double), cudaMemcpyDeviceToHost));
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long long local_duration_ms = static_cast<long long>(duration.count());
    long long global_duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &global_duration_ms, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Computation time: %lld ms\n", global_duration_ms);
        double gflops = (2.0 * N * N * N) / (global_duration_ms / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }
    
    // Gather results if needed for printing
    if (printResults) {
        std::vector<double> global_C;
        if (rank == 0) global_C.resize(N * N);
        
        MPI_Gather(local_C.data(), rows_per_rank * N, MPI_DOUBLE,
                   global_C.data(), rows_per_rank * N, MPI_DOUBLE,
                   0, MPI_COMM_WORLD);
                   
        if (rank == 0) {
            print_results(global_C, "MatrixC");
        }
    }
    
    // Validation
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        bool valid = validateResult(local_C, N, rank, size, rows_per_rank);
        
        if (rank == 0) {
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
        
        if (rank == 0 && !valid) {
            cublasDestroy(handle);
            cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_C);
            MPI_Finalize();
            return 1;
        }
    }
    
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    MPI_Finalize();
    return 0;
}
