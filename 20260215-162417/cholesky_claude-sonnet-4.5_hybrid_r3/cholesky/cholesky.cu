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

#include "../common/results_output.hpp"

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t stat = call; \
    if (stat != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, stat); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

// CUDA kernel for computing sum of squares for diagonal element
__global__ void compute_diag_sum_kernel(const double* L, double* partial_sums, size_t j, size_t n) {
    int tid = threadIdx.x;
    int k = blockIdx.x * blockDim.x + tid;
    
    __shared__ double sdata[256];
    
    double sum = 0.0;
    if (k < j) {
        sum = L[j * n + k] * L[j * n + k];
    }
    sdata[tid] = sum;
    __syncthreads();
    
    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && k + s < j) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

// CUDA kernel for computing dot products for off-diagonal elements
__global__ void compute_offdiag_dots_kernel(const double* L, double* dots, size_t i_start, size_t num_rows, size_t j, size_t n) {
    int row_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (row_idx >= num_rows) return;
    
    size_t i = i_start + row_idx;
    if (i <= j) return;
    
    __shared__ double sdata[256];
    
    double sum = 0.0;
    for (size_t k = tid; k < j; k += blockDim.x) {
        sum += L[i * n + k] * L[j * n + k];
    }
    sdata[tid] = sum;
    __syncthreads();
    
    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        dots[row_idx] = sdata[0];
    }
}

// CUDA kernel for finalizing off-diagonal elements
__global__ void finalize_offdiag_simple_kernel(double* L, const double* A, const double* dots, 
                                                 size_t i_start, size_t num_rows, size_t j, 
                                                 size_t n, double diag_val) {
    int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row_idx >= num_rows) return;
    
    size_t i = i_start + row_idx;
    if (i <= j) return;
    
    L[i * n + j] = (A[i * n + j] - dots[row_idx]) / diag_val;
}

// CUDA kernel for zeroing upper triangular part
__global__ void zero_upper_kernel(double* L, size_t i_start, size_t i_end, size_t n) {
    size_t i = i_start + blockIdx.y;
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < i_end && j < n && j > i) {
        L[i * n + j] = 0.0;
    }
}

// Simple Cholesky decomposition (sequential, unblocked algorithm)
// Decomposes positive definite matrix A into L * L^T where L is lower triangular

bool choleskyDecomposition(std::vector<double>& A, const size_t n, int rank, int size) {
    // A is stored in row-major order
    // Use column-wise parallelization with MPI
    
    // Allocate GPU memory for entire matrix on each rank
    double *d_L, *d_A;
    CUDA_CHECK(cudaMalloc(&d_L, n * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_A, n * n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_L, A.data(), n * n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), n * n * sizeof(double), cudaMemcpyHostToDevice));
    
    // Create cuBLAS handle
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    
    // Column-wise Cholesky decomposition
    for (size_t j = 0; j < n; ++j) {
        // Determine which rank computes this column
        int owner_rank = j % size;
        
        // Owner rank computes diagonal element and the entire column j
        if (rank == owner_rank) {
            // Compute sum of squares: sum(L[j,k]^2 for k < j)
            double sum = 0.0;
            
            if (j > 0) {
                // Use GPU kernel for sum of squares
                int block_size = 256;
                int num_blocks = (j + block_size - 1) / block_size;
                double *d_partial_sums;
                CUDA_CHECK(cudaMalloc(&d_partial_sums, num_blocks * sizeof(double)));
                
                compute_diag_sum_kernel<<<num_blocks, block_size>>>(d_L, d_partial_sums, j, n);
                CUDA_CHECK(cudaDeviceSynchronize());
                
                std::vector<double> h_partial_sums(num_blocks);
                CUDA_CHECK(cudaMemcpy(h_partial_sums.data(), d_partial_sums, num_blocks * sizeof(double), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaFree(d_partial_sums));
                
                #pragma omp parallel for reduction(+:sum)
                for (int b = 0; b < num_blocks; ++b) {
                    sum += h_partial_sums[b];
                }
            }
            
            // Get diagonal element
            double a_jj;
            CUDA_CHECK(cudaMemcpy(&a_jj, d_A + j * n + j, sizeof(double), cudaMemcpyDeviceToHost));
            
            const double val = a_jj - sum;
            if (val <= 0.0) {
                printf("Rank %d: Error: Matrix is not positive definite at diagonal element %zu (val=%f)\n", rank, j, val);
                CUDA_CHECK(cudaFree(d_L));
                CUDA_CHECK(cudaFree(d_A));
                CUBLAS_CHECK(cublasDestroy(handle));
                return false;
            }
            double diag_val = sqrt(val);
            
            // Write back to GPU
            CUDA_CHECK(cudaMemcpy(d_L + j * n + j, &diag_val, sizeof(double), cudaMemcpyHostToDevice));
            
            // Compute all off-diagonal elements in column j using GPU
            size_t num_rows = n - j - 1;
            if (num_rows > 0) {
                if (j > 0) {
                    // Use GPU kernel for dot products
                    double *d_dots;
                    CUDA_CHECK(cudaMalloc(&d_dots, num_rows * sizeof(double)));
                    
                    int block_size = 256;
                    int num_blocks = num_rows;
                    compute_offdiag_dots_kernel<<<num_blocks, block_size>>>(d_L, d_dots, j + 1, num_rows, j, n);
                    CUDA_CHECK(cudaDeviceSynchronize());
                    
                    // Finalize off-diagonal elements
                    int final_block_size = 256;
                    int final_num_blocks = (num_rows + final_block_size - 1) / final_block_size;
                    finalize_offdiag_simple_kernel<<<final_num_blocks, final_block_size>>>(
                        d_L, d_A, d_dots, j + 1, num_rows, j, n, diag_val);
                    CUDA_CHECK(cudaDeviceSynchronize());
                    
                    CUDA_CHECK(cudaFree(d_dots));
                } else {
                    // j == 0, simple division on GPU
                    #pragma omp parallel for
                    for (size_t i = j + 1; i < n; ++i) {
                        double a_ij;
                        CUDA_CHECK(cudaMemcpy(&a_ij, d_A + i * n + j, sizeof(double), cudaMemcpyDeviceToHost));
                        double result = a_ij / diag_val;
                        CUDA_CHECK(cudaMemcpy(d_L + i * n + j, &result, sizeof(double), cudaMemcpyHostToDevice));
                    }
                }
            }
        }
        
        // Broadcast entire column j to all ranks
        std::vector<double> col_j(n);
        if (rank == owner_rank) {
            // Extract column j from GPU
            for (size_t i = 0; i < n; ++i) {
                CUDA_CHECK(cudaMemcpy(&col_j[i], d_L + i * n + j, sizeof(double), cudaMemcpyDeviceToHost));
            }
        }
        
        // Broadcast to all ranks
        MPI_Bcast(col_j.data(), n, MPI_DOUBLE, owner_rank, MPI_COMM_WORLD);
        
        // Non-owner ranks update their GPU copy with column j
        if (rank != owner_rank) {
            for (size_t i = 0; i < n; ++i) {
                CUDA_CHECK(cudaMemcpy(d_L + i * n + j, &col_j[i], sizeof(double), cudaMemcpyHostToDevice));
            }
        }
        
        // Synchronize
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    // Zero out upper triangular part
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            double zero = 0.0;
            CUDA_CHECK(cudaMemcpy(d_L + i * n + j, &zero, sizeof(double), cudaMemcpyHostToDevice));
        }
    }
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(A.data(), d_L, n * n * sizeof(double), cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_L));
    CUDA_CHECK(cudaFree(d_A));
    CUBLAS_CHECK(cublasDestroy(handle));
    
    return true;
}

// Generate a symmetric positive definite matrix
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n, int rank, int size) {
    // Method: Create A = B * B^T where B is random
    // This guarantees A is positive semi-definite
    // Then add identity to make it strictly positive definite
    
    // All ranks generate the same matrix with same seed
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    
    // Generate random matrix B (all ranks generate the same values)
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }
    
    // Compute A = B * B^T (distribute work across ranks, then gather)
    size_t rows_per_rank = n / size;
    size_t remainder = n % size;
    size_t my_row_start = rank * rows_per_rank + std::min((size_t)rank, remainder);
    size_t my_row_count = rows_per_rank + (rank < (int)remainder ? 1 : 0);
    size_t my_row_end = my_row_start + my_row_count;
    
    #pragma omp parallel for collapse(2)
    for (size_t i = my_row_start; i < my_row_end; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += B[i * n + k] * B[j * n + k];
            }
            A[i * n + j] = sum;
        }
    }
    
    // Gather all rows to all ranks
    std::vector<int> recvcounts(size);
    std::vector<int> displs(size);
    for (int r = 0; r < size; ++r) {
        size_t r_rows_per_rank = n / size;
        size_t r_remainder = n % size;
        size_t r_row_start = r * r_rows_per_rank + std::min((size_t)r, r_remainder);
        size_t r_row_count = r_rows_per_rank + (r < (int)r_remainder ? 1 : 0);
        recvcounts[r] = r_row_count * n;
        displs[r] = r_row_start * n;
    }
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, A.data(), 
                   recvcounts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    
    // Add diagonal dominance to ensure positive definiteness
    for (size_t i = 0; i < n; ++i) {
        A[i * n + i] += n;
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n, int rank, int size) {
    // Validate by computing L * L^T and comparing with original matrix
    
    std::vector<double> reconstructed(n * n);
    
    // Distribute rows across ranks for computation
    size_t rows_per_rank = n / size;
    size_t remainder = n % size;
    size_t my_row_start = rank * rows_per_rank + std::min((size_t)rank, remainder);
    size_t my_row_count = rows_per_rank + (rank < (int)remainder ? 1 : 0);
    size_t my_row_end = my_row_start + my_row_count;
    
    // Compute L * L^T for owned rows
    #pragma omp parallel for collapse(2)
    for (size_t i = my_row_start; i < my_row_end; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += L[i * n + k] * L[j * n + k];
            }
            reconstructed[i * n + j] = sum;
        }
    }
    
    // Gather all rows to rank 0
    if (rank == 0) {
        for (int r = 1; r < size; ++r) {
            size_t r_rows_per_rank = n / size;
            size_t r_remainder = n % size;
            size_t r_row_start = r * r_rows_per_rank + std::min((size_t)r, r_remainder);
            size_t r_row_count = r_rows_per_rank + (r < (int)r_remainder ? 1 : 0);
            MPI_Status status;
            MPI_Recv(reconstructed.data() + r_row_start * n, r_row_count * n, 
                     MPI_DOUBLE, r, 0, MPI_COMM_WORLD, &status);
        }
    } else {
        MPI_Send(reconstructed.data() + my_row_start * n, my_row_count * n, 
                 MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
    
    // Compare with original (only on rank 0)
    bool valid = true;
    if (rank == 0) {
        double maxError = 0.0;
        double relError = 0.0;
        
        #pragma omp parallel for reduction(max:maxError,relError)
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
            valid = false;
        }
    }
    
    // Broadcast validation result
    MPI_Bcast(&valid, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    
    return valid;
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
    
    // Initialize CUDA for this rank
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    int device_id = rank % device_count;
    CUDA_CHECK(cudaSetDevice(device_id));
    
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
        printf("Cholesky Decomposition Benchmark (MPI+OpenMP+CUDA)\n");
        printf("MPI ranks: %d\n", size);
        printf("OpenMP threads per rank: %d\n", omp_get_max_threads());
        printf("CUDA devices: %d\n", device_count);
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Allocate matrix
    std::vector<double> A(n * n);
    std::vector<double> A_orig;
    
    // Generate positive definite matrix
    if (rank == 0) {
        printf("Generating positive definite matrix...\n");
    }
    generatePositiveDefiniteMatrix(A, n, rank, size);
    
    if (validate) {
        A_orig = A; // Save original for validation
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Perform Cholesky decomposition
    if (rank == 0) {
        printf("Computing Cholesky decomposition...\n");
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    bool success = choleskyDecomposition(A, n, rank, size);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (!success) {
        if (rank == 0) {
            printf("Cholesky decomposition failed\n");
        }
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
    if (printResults && rank == 0) {
        print_results(A, "CholeskyL");
    }
    
    // Validation
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
        }
        bool valid = validateCholesky(A, A_orig, n, rank, size);
        
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
