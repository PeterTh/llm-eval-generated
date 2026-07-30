#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <omp.h>

#include "../common/results_output.hpp"

// Hybrid MPI+OpenMP+CUDA Cholesky decomposition
// Uses blocked right-looking algorithm with GPU acceleration

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            MPI_Abort(MPI_COMM_WORLD, 1); \
        } \
    } while(0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, status); \
            MPI_Abort(MPI_COMM_WORLD, 1); \
        } \
    } while(0)

#define CUSOLVER_CHECK(call) \
    do { \
        cusolverStatus_t status = call; \
        if (status != CUSOLVER_STATUS_SUCCESS) { \
            fprintf(stderr, "cuSOLVER error at %s:%d: %d\n", __FILE__, __LINE__, status); \
            MPI_Abort(MPI_COMM_WORLD, 1); \
        } \
    } while(0)

// Hybrid Cholesky decomposition using MPI+OpenMP+CUDA
// Uses cuSOLVER for GPU-accelerated Cholesky factorization
bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    
    // Initialize CUDA - assign one GPU per MPI rank
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    int device_id = mpi_rank % device_count;
    CUDA_CHECK(cudaSetDevice(device_id));
    
    if (mpi_rank == 0) {
        printf("Using %d MPI ranks, %d GPUs available\n", mpi_size, device_count);
    }
    
    // Initialize cuSOLVER
    cusolverDnHandle_t cusolver_handle;
    CUSOLVER_CHECK(cusolverDnCreate(&cusolver_handle));
    
    // Allocate GPU memory
    double* d_A;
    CUDA_CHECK(cudaMalloc(&d_A, n * n * sizeof(double)));
    
    // Allocate column-major buffer for cuSOLVER (Fortran-style)
    std::vector<double> A_col(n * n);
    
    // Convert row-major to column-major using OpenMP
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            A_col[j * n + i] = A[i * n + j];
        }
    }
    
    // Copy to GPU
    CUDA_CHECK(cudaMemcpy(d_A, A_col.data(), n * n * sizeof(double), cudaMemcpyHostToDevice));
    
    // Query workspace size for Cholesky factorization
    int lwork = 0;
    CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(
        cusolver_handle,
        CUBLAS_FILL_MODE_LOWER,
        static_cast<int>(n),
        d_A,
        static_cast<int>(n),
        &lwork));
    
    // Allocate workspace
    double* d_workspace;
    CUDA_CHECK(cudaMalloc(&d_workspace, lwork * sizeof(double)));
    
    // Allocate device info
    int* d_info;
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));
    
    // Perform Cholesky factorization using cuSOLVER
    CUSOLVER_CHECK(cusolverDnDpotrf(
        cusolver_handle,
        CUBLAS_FILL_MODE_LOWER,
        static_cast<int>(n),
        d_A,
        static_cast<int>(n),
        d_workspace,
        lwork,
        d_info));
    
    // Check result
    int h_info;
    CUDA_CHECK(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    
    if (h_info != 0) {
        if (mpi_rank == 0) {
            if (h_info < 0) {
                fprintf(stderr, "Error: Matrix is not positive definite (leading minor of order %d)\n", h_info);
            } else {
                fprintf(stderr, "Error: cuSOLVER dpotrf failed with info = %d\n", h_info);
            }
        }
        cudaFree(d_workspace);
        cudaFree(d_info);
        cudaFree(d_A);
        cusolverDnDestroy(cusolver_handle);
        return false;
    }
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(A_col.data(), d_A, n * n * sizeof(double), cudaMemcpyDeviceToHost));
    
    // Convert column-major back to row-major and zero out upper triangular part
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            if (j <= i) {
                A[i * n + j] = A_col[j * n + i];
            } else {
                A[i * n + j] = 0.0;
            }
        }
    }
    
    // Cleanup
    cudaFree(d_workspace);
    cudaFree(d_info);
    cudaFree(d_A);
    cusolverDnDestroy(cusolver_handle);
    
    return true;
}

// Generate a symmetric positive definite matrix
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    // Method: Create A = B * B^T where B is random
    // This guarantees A is positive semi-definite
    // Then add identity to make it strictly positive definite
    
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    
    // Generate random matrix B
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }
    
    // Compute A = B * B^T
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += B[i * n + k] * B[j * n + k];
            }
            A[i * n + j] = sum;
        }
    }
    
    // Add diagonal dominance to ensure positive definiteness
    for (size_t i = 0; i < n; ++i) {
        A[i * n + i] += n;
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix
    
    std::vector<double> reconstructed(n * n);
    
    // Compute L * L^T
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += L[i * n + k] * L[j * n + k];
            }
            reconstructed[i * n + j] = sum;
        }
    }
    
    // Compare with original
    double maxError = 0.0;
    double relError = 0.0;
    
    for (size_t i = 0; i < n * n; ++i) {
        const double error = fabs(reconstructed[i] - A_orig[i]);
        maxError = std::max(maxError, error);
        
        const double rel = error / (fabs(A_orig[i]) + 1e-10);
        relError = std::max(relError, rel);
    }
    
    printf("Max absolute error: %.10e\n", maxError);
    printf("Max relative error: %.10e\n", relError);
    
    // Check if error is within tolerance
    if (relError > 1e-6) {
        printf("Validation failed: relative error too large\n");
        return false;
    }
    
    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);
    
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    
    size_t n = 512;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments (only rank 0 needs to parse)
    if (mpi_rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                n = atoi(argv[++i]);
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
    
    // Broadcast parameters to all ranks
    MPI_Bcast(&n, sizeof(size_t), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, sizeof(bool), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, sizeof(bool), MPI_BYTE, 0, MPI_COMM_WORLD);
    
    if (mpi_rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Allocate matrix
    std::vector<double> A(n * n);
    std::vector<double> A_orig;
    
    // Generate positive definite matrix (all ranks generate the same matrix)
    if (mpi_rank == 0) {
        printf("Generating positive definite matrix...\n");
    }
    generatePositiveDefiniteMatrix(A, n);
    
    if (validate) {
        A_orig = A; // Save original for validation
    }
    
    // Perform Cholesky decomposition
    if (mpi_rank == 0) {
        printf("Computing Cholesky decomposition...\n");
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    bool success = choleskyDecomposition(A, n);
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (!success) {
        if (mpi_rank == 0) {
            printf("Cholesky decomposition failed\n");
        }
        MPI_Finalize();
        return 1;
    }
    
    if (mpi_rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        
        // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }
    
    // Print results for external validation (only rank 0)
    if (printResults && mpi_rank == 0) {
        print_results(A, "CholeskyL");
    }
    
    // Validation (only rank 0)
    if (validate && mpi_rank == 0) {
        printf("Validating result...\n");
        bool valid = validateCholesky(A, A_orig, n);
        
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
