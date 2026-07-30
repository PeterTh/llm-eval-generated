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
// Distributed blocked algorithm for true parallel scalability:
// - Matrix is distributed across MPI processes by block columns
// - Each process owns specific blocks and computes them
// - Processes communicate to share results via MPI
// - OpenMP accelerates CPU operations
// - CUDA/cuSOLVER/cuBLAS for GPU acceleration

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    
    // Initialize CUDA - each MPI process uses a different GPU
    int num_gpus;
    CUDA_CHECK(cudaGetDeviceCount(&num_gpus));
    int gpu_id = mpi_rank % num_gpus;
    CUDA_CHECK(cudaSetDevice(gpu_id));
    
    // Create cuSOLVER and cuBLAS handles
    cusolverDnHandle_t cusolver_handle;
    cublasHandle_t cublas_handle;
    CUSOLVER_CHECK(cusolverDnCreate(&cusolver_handle));
    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    
    // Block size for blocked algorithm
    const size_t block_size = 256;
    const size_t num_blocks = (n + block_size - 1) / block_size;
    
    // Transpose matrix from row-major to column-major for cuSOLVER/cuBLAS
    // In column-major: element (row, col) is at offset col * n + row
    std::vector<double> A_col_major(n * n);
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            A_col_major[j * n + i] = A[i * n + j];
        }
    }
    
    // Allocate GPU memory for the full matrix (each process has full copy)
    double* d_A;
    CUDA_CHECK(cudaMalloc(&d_A, n * n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_A, A_col_major.data(), n * n * sizeof(double), cudaMemcpyHostToDevice));
    
    // Allocate workspace and info
    double* d_workspace = nullptr;
    int* d_info;
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));
    
    // Helper: pointer to start of block (block_row, block_col) in column-major
    // Block (br, bc) starts at column bc*block_size, row br*block_size
    // In column-major: offset = col * n + row
    #define BLOCK_PTR(br, bc) (d_A + (bc) * block_size * n + (br) * block_size)
    #define BLOCK_COL_PTR(br, bc, br_size) (d_A + (bc) * block_size * n + (br) * block_size)
    
    // Blocked Cholesky decomposition with distributed work
    for (size_t k = 0; k < num_blocks; ++k) {
        size_t k_start = k * block_size;
        size_t k_size = std::min(block_size, n - k_start);
        
        // Step 1: Factor diagonal block A[k,k]
        // Process k % mpi_size owns this block
        int diag_owner = k % mpi_size;
        
        if (mpi_rank == diag_owner) {
            // Get workspace size for this block
            int lwork = 0;
            CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(cusolver_handle, CUBLAS_FILL_MODE_LOWER,
                                                        (int)k_size, d_A + k_start * n + k_start, (int)n, &lwork));
            
            if (lwork > 0) {
                if (d_workspace) CUDA_CHECK(cudaFree(d_workspace));
                CUDA_CHECK(cudaMalloc(&d_workspace, lwork * sizeof(double)));
            }
            
            // Factor diagonal block using cuSOLVER
            CUSOLVER_CHECK(cusolverDnDpotrf(cusolver_handle, CUBLAS_FILL_MODE_LOWER,
                                              (int)k_size, d_A + k_start * n + k_start, (int)n,
                                              d_workspace, lwork, d_info));
            
            int h_info;
            CUDA_CHECK(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
            if (h_info != 0) {
                if (mpi_rank == 0) {
                    printf("Error: Matrix is not positive definite at block %zu (info = %d)\n", k, h_info);
                }
                if (d_workspace) CUDA_CHECK(cudaFree(d_workspace));
                CUDA_CHECK(cudaFree(d_info));
                CUDA_CHECK(cudaFree(d_A));
                CUSOLVER_CHECK(cusolverDnDestroy(cusolver_handle));
                CUBLAS_CHECK(cublasDestroy(cublas_handle));
                return false;
            }
        }
        
        // Broadcast the factored diagonal block to all processes
        std::vector<double> L_kk(k_size * k_size);
        if (mpi_rank == diag_owner) {
            // Extract block column by column from column-major matrix
            for (size_t c = 0; c < k_size; ++c) {
                CUDA_CHECK(cudaMemcpy(L_kk.data() + c * k_size,
                                     d_A + (k_start + c) * n + k_start,
                                     k_size * sizeof(double), cudaMemcpyDeviceToHost));
            }
        }
        MPI_Bcast(L_kk.data(), k_size * k_size, MPI_DOUBLE, diag_owner, MPI_COMM_WORLD);
        
        // Update diagonal block on all processes
        if (mpi_rank != diag_owner) {
            for (size_t c = 0; c < k_size; ++c) {
                CUDA_CHECK(cudaMemcpy(d_A + (k_start + c) * n + k_start,
                                     L_kk.data() + c * k_size,
                                     k_size * sizeof(double), cudaMemcpyHostToDevice));
            }
        }
        
        // Step 2: Solve for off-diagonal blocks L[i,k] = A[i,k] * inv(L[k,k]^T)
        // Distribute blocks across processes
        for (size_t i = k + 1; i < num_blocks; ++i) {
            int block_owner = i % mpi_size;
            
            if (mpi_rank == block_owner) {
                size_t i_start = i * block_size;
                size_t i_size = std::min(block_size, n - i_start);
                
                // Solve: L[i,k] = A[i,k] * inv(L[k,k]^T)
                // In column-major, block (i,k) starts at: k_start * n + i_start
                // L[k,k] starts at: k_start * n + k_start
                // trsm: B = alpha * B * inv(A^T) where A is L[k,k], B is A[i,k]
                const double alpha = 1.0;
                CUBLAS_CHECK(cublasDtrsm(cublas_handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                                         CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
                                         (int)i_size, (int)k_size, &alpha,
                                         d_A + k_start * n + k_start, (int)n,
                                         d_A + k_start * n + i_start, (int)n));
            }
        }
        
        // Gather all off-diagonal blocks to all processes
        for (size_t i = k + 1; i < num_blocks; ++i) {
            int block_owner = i % mpi_size;
            size_t i_start = i * block_size;
            size_t i_size = std::min(block_size, n - i_start);
            
            std::vector<double> L_ik(i_size * k_size);
            if (mpi_rank == block_owner) {
                for (size_t c = 0; c < k_size; ++c) {
                    CUDA_CHECK(cudaMemcpy(L_ik.data() + c * i_size,
                                         d_A + (k_start + c) * n + i_start,
                                         i_size * sizeof(double), cudaMemcpyDeviceToHost));
                }
            }
            MPI_Bcast(L_ik.data(), i_size * k_size, MPI_DOUBLE, block_owner, MPI_COMM_WORLD);
            
            if (mpi_rank != block_owner) {
                for (size_t c = 0; c < k_size; ++c) {
                    CUDA_CHECK(cudaMemcpy(d_A + (k_start + c) * n + i_start,
                                         L_ik.data() + c * i_size,
                                         i_size * sizeof(double), cudaMemcpyHostToDevice));
                }
            }
        }
        
        // Step 3: Update trailing matrix A[i,j] = A[i,j] - L[i,k] * L[j,k]^T
        // Distribute updates across processes
        for (size_t i = k + 1; i < num_blocks; ++i) {
            size_t i_start = i * block_size;
            size_t i_size = std::min(block_size, n - i_start);
            
            for (size_t j = k + 1; j <= i; ++j) {
                int update_owner = (i + j) % mpi_size;
                
                if (mpi_rank == update_owner) {
                    size_t j_start = j * block_size;
                    size_t j_size = std::min(block_size, n - j_start);
                    
                    // Update: A[i,j] = A[i,j] - L[i,k] * L[j,k]^T
                    // Block (i,k) at: k_start * n + i_start, dims i_size x k_size
                    // Block (j,k) at: k_start * n + j_start, dims j_size x k_size
                    // Block (i,j) at: j_start * n + i_start, dims i_size x j_size
                    // gemm: C = alpha * A * B^T + beta * C
                    const double alpha = -1.0;
                    const double beta = 1.0;
                    CUBLAS_CHECK(cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                                             (int)i_size, (int)j_size, (int)k_size, &alpha,
                                             d_A + k_start * n + i_start, (int)n,
                                             d_A + k_start * n + j_start, (int)n, &beta,
                                             d_A + j_start * n + i_start, (int)n));
                }
            }
        }
        
        // Gather all updated blocks to all processes
        for (size_t i = k + 1; i < num_blocks; ++i) {
            size_t i_start = i * block_size;
            size_t i_size = std::min(block_size, n - i_start);
            
            for (size_t j = k + 1; j <= i; ++j) {
                int update_owner = (i + j) % mpi_size;
                size_t j_start = j * block_size;
                size_t j_size = std::min(block_size, n - j_start);
                
                std::vector<double> A_ij(i_size * j_size);
                if (mpi_rank == update_owner) {
                    for (size_t c = 0; c < j_size; ++c) {
                        CUDA_CHECK(cudaMemcpy(A_ij.data() + c * i_size,
                                             d_A + (j_start + c) * n + i_start,
                                             i_size * sizeof(double), cudaMemcpyDeviceToHost));
                    }
                }
                MPI_Bcast(A_ij.data(), i_size * j_size, MPI_DOUBLE, update_owner, MPI_COMM_WORLD);
                
                if (mpi_rank != update_owner) {
                    for (size_t c = 0; c < j_size; ++c) {
                        CUDA_CHECK(cudaMemcpy(d_A + (j_start + c) * n + i_start,
                                             A_ij.data() + c * i_size,
                                             i_size * sizeof(double), cudaMemcpyHostToDevice));
                    }
                }
            }
        }
    }
    
    #undef BLOCK_PTR
    #undef BLOCK_COL_PTR
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(A_col_major.data(), d_A, n * n * sizeof(double), cudaMemcpyDeviceToHost));
    
    // Transpose back from column-major to row-major
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            A[i * n + j] = A_col_major[j * n + i];
        }
    }
    
    // Zero out upper triangular part using OpenMP
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            A[i * n + j] = 0.0;
        }
    }
    
    // Cleanup
    if (d_workspace) CUDA_CHECK(cudaFree(d_workspace));
    CUDA_CHECK(cudaFree(d_info));
    CUDA_CHECK(cudaFree(d_A));
    CUSOLVER_CHECK(cusolverDnDestroy(cusolver_handle));
    CUBLAS_CHECK(cublasDestroy(cublas_handle));
    
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
    
    int mpi_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    
    size_t n = 512;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (mpi_rank == 0) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            if (mpi_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    
    // Only rank 0 prints header
    if (mpi_rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Allocate matrix
    std::vector<double> A(n * n);
    std::vector<double> A_orig;
    
    // Generate positive definite matrix
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
    auto start = std::chrono::high_resolution_clock::now();
    
    bool success = choleskyDecomposition(A, n);
    
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
    
    // Print results for external validation
    if (printResults && mpi_rank == 0) {
        print_results(A, "CholeskyL");
    }
    
    // Validation
    if (validate) {
        if (mpi_rank == 0) {
            printf("Validating result...\n");
        }
        bool valid = validateCholesky(A, A_orig, n);
        
        if (mpi_rank == 0) {
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
