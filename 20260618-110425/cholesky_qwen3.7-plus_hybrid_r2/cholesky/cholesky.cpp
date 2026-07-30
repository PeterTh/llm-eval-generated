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

// Convert row-major to column-major
void rowToColMajor(const double* row, double* col, size_t n) {
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            col[j * n + i] = row[i * n + j];
        }
    }
}

// Convert column-major to row-major
void colToRowMajor(const double* col, double* row, size_t n) {
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            row[i * n + j] = col[j * n + i];
        }
    }
}

// Hybrid MPI+OpenMP+CUDA Cholesky decomposition
// Uses right-looking blocked algorithm with GPU acceleration
bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Select GPU based on rank
    int num_gpus;
    CUDA_CHECK(cudaGetDeviceCount(&num_gpus));
    int gpu_id = rank % num_gpus;
    CUDA_CHECK(cudaSetDevice(gpu_id));
    
    // Initialize cuBLAS and cuSOLVER
    cublasHandle_t cublas_handle;
    cusolverDnHandle_t cusolver_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    CUSOLVER_CHECK(cusolverDnCreate(&cusolver_handle));
    
    // Convert row-major to column-major for cuBLAS/cuSOLVER
    std::vector<double> A_col(n * n);
    rowToColMajor(A.data(), A_col.data(), n);
    
    // Allocate GPU memory
    double *d_A;
    CUDA_CHECK(cudaMalloc(&d_A, n * n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_A, A_col.data(), n * n * sizeof(double), cudaMemcpyHostToDevice));
    
    // Allocate workspace for cuSOLVER
    int lwork = 0;
    CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(cusolver_handle, CUBLAS_FILL_MODE_LOWER, n, d_A, n, &lwork));
    double *d_workspace;
    CUDA_CHECK(cudaMalloc(&d_workspace, lwork * sizeof(double)));
    
    // Allocate devInfo
    int *d_devInfo;
    CUDA_CHECK(cudaMalloc(&d_devInfo, sizeof(int)));
    
    // Block size for blocked algorithm
    const size_t NB = 256;
    const double one = 1.0;
    const double minus_one = -1.0;
    
    // Right-looking blocked Cholesky
    for (size_t k = 0; k < n; k += NB) {
        size_t kb = std::min(NB, n - k);
        size_t m = n - k - kb;
        
        // Factor diagonal block using cuSOLVER POTRF
        // A(k:k+kb, k:k+kb) = chol(A(k:k+kb, k:k+kb))
        CUSOLVER_CHECK(cusolverDnDpotrf(cusolver_handle, CUBLAS_FILL_MODE_LOWER, kb, 
                                       d_A + k * n + k, n, d_workspace, lwork, d_devInfo));
        
        // Check if factorization succeeded
        int devInfo_h;
        CUDA_CHECK(cudaMemcpy(&devInfo_h, d_devInfo, sizeof(int), cudaMemcpyDeviceToHost));
        if (devInfo_h != 0) {
            fprintf(stderr, "cuSOLVER POTRF failed at block %zu, info=%d\n", k, devInfo_h);
            CUDA_CHECK(cudaFree(d_A));
            CUDA_CHECK(cudaFree(d_workspace));
            CUDA_CHECK(cudaFree(d_devInfo));
            CUBLAS_CHECK(cublasDestroy(cublas_handle));
            CUSOLVER_CHECK(cusolverDnDestroy(cusolver_handle));
            return false;
        }
        
        // Update trailing matrix if exists
        if (m > 0) {
            // Solve A(k+kb:n, k:k+kb) = A(k+kb:n, k:k+kb) * inv(L(k:k+kb, k:k+kb)^T)
            CUBLAS_CHECK(cublasDtrsm(cublas_handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                                    CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
                                    m, kb, &one, d_A + k * n + k, n, 
                                    d_A + (k + kb) * n + k, n));
            
            // Update A(k+kb:n, k+kb:n) = A(k+kb:n, k+kb:n) - A(k+kb:n, k:k+kb) * A(k+kb:n, k:k+kb)^T
            CUBLAS_CHECK(cublasDsyrk(cublas_handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                                    m, kb, &minus_one, d_A + (k + kb) * n + k, n,
                                    &one, d_A + (k + kb) * n + (k + kb), n));
        }
    }
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(A_col.data(), d_A, n * n * sizeof(double), cudaMemcpyDeviceToHost));
    
    // Convert column-major back to row-major
    colToRowMajor(A_col.data(), A.data(), n);
    
    // Zero out upper triangular part using OpenMP
    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        #pragma omp simd
        for (size_t j = i + 1; j < n; ++j) {
            A[i * n + j] = 0.0;
        }
    }
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_workspace));
    CUDA_CHECK(cudaFree(d_devInfo));
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
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    size_t n = 512;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments (only rank 0 prints help)
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = atoi(argv[++i]);
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
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("MPI ranks: %d\n", size);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Allocate matrix
    std::vector<double> A(n * n);
    std::vector<double> A_orig;
    
    // Generate positive definite matrix (all ranks generate same matrix)
    if (rank == 0) {
        printf("Generating positive definite matrix...\n");
    }
    generatePositiveDefiniteMatrix(A, n);
    
    if (validate) {
        A_orig = A; // Save original for validation
    }
    
    // Synchronize before timing
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Perform Cholesky decomposition
    if (rank == 0) {
        printf("Computing Cholesky decomposition...\n");
    }
    auto start = std::chrono::high_resolution_clock::now();
    
    bool success = choleskyDecomposition(A, n);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (!success) {
        if (rank == 0) printf("Cholesky decomposition failed\n");
        MPI_Finalize();
        return 1;
    }
    
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        
        // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }
    
    // Print results for external validation
    if (printResults) {
        print_results(A, "CholeskyL");
    }
    
    // Validation
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        bool valid = validateCholesky(A, A_orig, n);
        
        if (rank == 0) {
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
        MPI_Finalize();
        return valid ? 0 : 1;
    }
    
    MPI_Finalize();
    return 0;
}
