#include <algorithm>
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
#include <cusolverDn.h>

#include "../common/results_output.hpp"

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

// Hybrid MPI+OpenMP+CUDA Cholesky decomposition
// Uses blocked right-looking algorithm with:
// - MPI for inter-process communication
// - OpenMP for intra-node parallelism
// - CUDA (cuSOLVER + cuBLAS) for GPU acceleration

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    
    // Initialize CUDA
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    int device_id = mpi_rank % device_count;
    CUDA_CHECK(cudaSetDevice(device_id));
    
    // Initialize cuBLAS and cuSOLVER
    cublasHandle_t cublas_handle;
    cusolverDnHandle_t cusolver_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    CUSOLVER_CHECK(cusolverDnCreate(&cusolver_handle));
    
    // Set cuBLAS to use the correct stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUBLAS_CHECK(cublasSetStream(cublas_handle, stream));
    CUSOLVER_CHECK(cusolverDnSetStream(cusolver_handle, stream));
    
    // Block size for blocked algorithm
    const size_t NB = 256;
    
    // Allocate GPU memory for the full matrix
    double* d_A;
    CUDA_CHECK(cudaMalloc(&d_A, n * n * sizeof(double)));
    
    // Copy matrix to GPU (column-major for cuBLAS/cuSOLVER)
    std::vector<double> A_col_major(n * n);
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            A_col_major[j * n + i] = A[i * n + j];
        }
    }
    CUDA_CHECK(cudaMemcpy(d_A, A_col_major.data(), n * n * sizeof(double), cudaMemcpyHostToDevice));
    
    // Allocate workspace for cuSOLVER
    int* d_info;
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));
    
    // Blocked right-looking Cholesky decomposition
    for (size_t k = 0; k < n; k += NB) {
        size_t kb = std::min(NB, n - k);
        
        // Step 1: Factor diagonal block A(k:k+kb, k:k+kb) using cuSOLVER
        if (mpi_rank == 0) {
            // Extract diagonal block (column-major: element (i,j) at offset i + j*n)
            std::vector<double> diag_block(kb * kb);
            CUDA_CHECK(cudaMemcpy2D(
                diag_block.data(), kb * sizeof(double),
                d_A + k + k * n, n * sizeof(double),
                kb * sizeof(double), kb,
                cudaMemcpyDeviceToHost
            ));
            
            // Factor using cuSOLVER (potrf)
            double* d_diag;
            CUDA_CHECK(cudaMalloc(&d_diag, kb * kb * sizeof(double)));
            CUDA_CHECK(cudaMemcpy(d_diag, diag_block.data(), kb * kb * sizeof(double), cudaMemcpyHostToDevice));
            
            int lwork = 0;
            CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(cusolver_handle, CUBLAS_FILL_MODE_LOWER, kb, d_diag, kb, &lwork));
            double* d_work;
            CUDA_CHECK(cudaMalloc(&d_work, lwork * sizeof(double)));
            
            CUSOLVER_CHECK(cusolverDnDpotrf(cusolver_handle, CUBLAS_FILL_MODE_LOWER, kb, d_diag, kb, d_work, lwork, d_info));
            
            int h_info;
            CUDA_CHECK(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
            
            if (h_info != 0) {
                if (mpi_rank == 0) {
                    printf("Error: Matrix is not positive definite at diagonal element %zu\n", k + h_info);
                }
                CUDA_CHECK(cudaFree(d_diag));
                CUDA_CHECK(cudaFree(d_work));
                CUDA_CHECK(cudaFree(d_A));
                CUDA_CHECK(cudaFree(d_info));
                CUDA_CHECK(cudaStreamDestroy(stream));
                CUBLAS_CHECK(cublasDestroy(cublas_handle));
                CUSOLVER_CHECK(cusolverDnDestroy(cusolver_handle));
                return false;
            }
            
            // Copy factored diagonal block back
            CUDA_CHECK(cudaMemcpy(diag_block.data(), d_diag, kb * kb * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy2D(
                d_A + k + k * n, n * sizeof(double),
                diag_block.data(), kb * sizeof(double),
                kb * sizeof(double), kb,
                cudaMemcpyHostToDevice
            ));
            
            CUDA_CHECK(cudaFree(d_diag));
            CUDA_CHECK(cudaFree(d_work));
        }
        
        // Broadcast diagonal block to all ranks
        std::vector<double> diag_block(kb * kb);
        if (mpi_rank == 0) {
            CUDA_CHECK(cudaMemcpy2D(
                diag_block.data(), kb * sizeof(double),
                d_A + k + k * n, n * sizeof(double),
                kb * sizeof(double), kb,
                cudaMemcpyDeviceToHost
            ));
        }
        MPI_Bcast(diag_block.data(), kb * kb, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        
        // Copy diagonal block back to GPU
        double* d_diag;
        CUDA_CHECK(cudaMalloc(&d_diag, kb * kb * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_diag, diag_block.data(), kb * kb * sizeof(double), cudaMemcpyHostToDevice));
        
        // Step 2: Solve for off-diagonal blocks: A(k+kb:n, k:k+kb) = A(k+kb:n, k:k+kb) * inv(L(k:k+kb, k:k+kb))^T
        if (k + kb < n) {
            size_t m = n - k - kb;
            const double alpha = 1.0;
            
            // A(k+kb:n, k:k+kb) starts at d_A + (k+kb) + k*n in column-major
            // B is m x kb, ldb = n
            CUBLAS_CHECK(cublasDtrsm(
                cublas_handle,
                CUBLAS_SIDE_RIGHT,
                CUBLAS_FILL_MODE_LOWER,
                CUBLAS_OP_T,
                CUBLAS_DIAG_NON_UNIT,
                m, kb,
                &alpha,
                d_diag, kb,
                d_A + (k + kb) + k * n, n
            ));
        }
        
        // Step 3: Update trailing matrix: A(k+kb:n, k+kb:n) -= A(k+kb:n, k:k+kb) * A(k+kb:n, k:k+kb)^T
        if (k + kb < n) {
            size_t m = n - k - kb;
            const double alpha = -1.0;
            const double beta = 1.0;
            
            // A(k+kb:n, k+kb:n) starts at d_A + (k+kb) + (k+kb)*n
            CUBLAS_CHECK(cublasDsyrk(
                cublas_handle,
                CUBLAS_FILL_MODE_LOWER,
                CUBLAS_OP_N,
                m, kb,
                &alpha,
                d_A + (k + kb) + k * n, n,
                &beta,
                d_A + (k + kb) + (k + kb) * n, n
            ));
        }
        
        CUDA_CHECK(cudaFree(d_diag));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(A_col_major.data(), d_A, n * n * sizeof(double), cudaMemcpyDeviceToHost));
    
    // Convert back to row-major
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            A[i * n + j] = A_col_major[j * n + i];
        }
    }
    
    // Zero out upper triangular part using OpenMP
    #pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        for (size_t j = i + 1; j < n; j++) {
            A[i * n + j] = 0.0;
        }
    }
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_info));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUBLAS_CHECK(cublasDestroy(cublas_handle));
    CUSOLVER_CHECK(cusolverDnDestroy(cusolver_handle));
    
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
    
    // Compute A = B * B^T using OpenMP
    #pragma omp parallel for schedule(dynamic)
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
    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        A[i * n + i] += n;
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix
    
    std::vector<double> reconstructed(n * n);
    
    // Compute L * L^T using OpenMP
    #pragma omp parallel for schedule(dynamic)
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
    
    #pragma omp parallel
    {
        double local_max_err = 0.0;
        double local_rel_err = 0.0;
        
        #pragma omp for
        for (size_t i = 0; i < n * n; ++i) {
            const double error = fabs(reconstructed[i] - A_orig[i]);
            local_max_err = std::max(local_max_err, error);
            
            const double rel = error / (fabs(A_orig[i]) + 1e-10);
            local_rel_err = std::max(local_rel_err, rel);
        }
        
        #pragma omp critical
        {
            maxError = std::max(maxError, local_max_err);
            relError = std::max(relError, local_rel_err);
        }
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
    MPI_Bcast(&n, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    
    if (mpi_rank == 0) {
        printf("Cholesky Decomposition Benchmark (Hybrid MPI+OpenMP+CUDA)\n");
        printf("MPI ranks: %d\n", mpi_size);
        printf("OpenMP threads: %d\n", omp_get_max_threads());
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
    
    if (!success) {
        if (mpi_rank == 0) {
            printf("Cholesky decomposition failed\n");
        }
        MPI_Finalize();
        return 1;
    }
    
    long long duration_ms = 0;
    if (mpi_rank == 0) {
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        duration_ms = duration.count();
        printf("Computation time: %lld ms\n", duration_ms);
        
        // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (duration_ms / 1000.0) / 1e9;
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
