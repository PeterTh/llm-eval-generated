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

#include "../common/results_output.hpp"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            MPI_Abort(MPI_COMM_WORLD, 1); \
        } \
    } while (0)

// CUDA kernel for dot product
__global__ void dotProductKernel(const double* a, const double* b, double* partial, size_t len) {
    __shared__ double sdata[256];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < len) ? a[i] * b[i] : 0.0;
    __syncthreads();
    
    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) partial[blockIdx.x] = sdata[0];
}

// CUDA kernel for column update
__global__ void columnUpdateKernel(double* A, const double* Lcol, size_t n, size_t j, 
                                   size_t start_row, double diag) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x + start_row;
    
    if (i > j && i < n) {
        double sum = 0.0;
        for (size_t k = 0; k < j; ++k) {
            sum += A[i * n + k] * Lcol[k];
        }
        A[i * n + j] = (A[i * n + j] - sum) / diag;
    }
}

// CUDA kernel to zero upper triangle
__global__ void zeroUpperKernel(double* A, size_t n, size_t row) {
    size_t j = blockIdx.x * blockDim.x + threadIdx.x + row + 1;
    if (j < n) {
        A[row * n + j] = 0.0;
    }
}

// Hybrid MPI+OpenMP+CUDA Cholesky decomposition
bool choleskyDecomposition(std::vector<double>& A, const size_t n, int rank, int size) {
    // Determine row distribution
    size_t rows_per_rank = n / size;
    size_t remainder = n % size;
    size_t start_row = rank * rows_per_rank + std::min((size_t)rank, remainder);
    size_t end_row = start_row + rows_per_rank + (rank < remainder ? 1 : 0);
    
    // Allocate device memory
    double *d_A, *d_Lcol, *d_partial;
    CUDA_CHECK(cudaMalloc(&d_A, n * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Lcol, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_partial, 256 * sizeof(double)));
    
    // Copy matrix to device
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), n * n * sizeof(double), cudaMemcpyHostToDevice));
    
    std::vector<double> h_partial(256);
    std::vector<double> col_buffer(n);
    
    for (size_t j = 0; j < n; ++j) {
        int owner = j / rows_per_rank;
        if (owner >= size) owner = size - 1;
        
        double diag_val;
        
        // Compute diagonal element
        if (rank == owner) {
            if (j > 0) {
                // Compute dot product using CUDA
                dim3 block(256);
                dim3 grid((j + block.x - 1) / block.x);
                dotProductKernel<<<grid, block>>>(d_A + j * n, d_A + j * n, d_partial, j);
                CUDA_CHECK(cudaDeviceSynchronize());
                
                CUDA_CHECK(cudaMemcpy(h_partial.data(), d_partial, grid.x * sizeof(double), 
                                     cudaMemcpyDeviceToHost));
                
                double sum = 0.0;
                #pragma omp parallel for reduction(+:sum)
                for (size_t b = 0; b < grid.x; ++b) {
                    sum += h_partial[b];
                }
                
                CUDA_CHECK(cudaMemcpy(&diag_val, d_A + j * n + j, sizeof(double), 
                                     cudaMemcpyDeviceToHost));
                diag_val = diag_val - sum;
            } else {
                CUDA_CHECK(cudaMemcpy(&diag_val, d_A + j * n + j, sizeof(double), 
                                     cudaMemcpyDeviceToHost));
            }
            
            if (diag_val <= 0.0) {
                CUDA_CHECK(cudaFree(d_A));
                CUDA_CHECK(cudaFree(d_Lcol));
                CUDA_CHECK(cudaFree(d_partial));
                if (rank == 0) {
                    printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
                }
                return false;
            }
            
            diag_val = sqrt(diag_val);
            CUDA_CHECK(cudaMemcpy(d_A + j * n + j, &diag_val, sizeof(double), 
                                 cudaMemcpyHostToDevice));
        }
        
        // Broadcast diagonal value
        MPI_Bcast(&diag_val, 1, MPI_DOUBLE, owner, MPI_COMM_WORLD);
        
        // Get column j up to position j
        if (rank == owner) {
            CUDA_CHECK(cudaMemcpy(col_buffer.data(), d_A + j * n, (j + 1) * sizeof(double), 
                                 cudaMemcpyDeviceToHost));
        }
        MPI_Bcast(col_buffer.data(), j + 1, MPI_DOUBLE, owner, MPI_COMM_WORLD);
        
        // Copy column to device for all ranks
        CUDA_CHECK(cudaMemcpy(d_Lcol, col_buffer.data(), (j + 1) * sizeof(double), 
                             cudaMemcpyHostToDevice));
        
        // Update rows owned by this rank
        if (start_row <= j && j < end_row) {
            // Zero upper triangle for row j
            if (j + 1 < n) {
                dim3 block(256);
                dim3 grid((n - j - 1 + block.x - 1) / block.x);
                zeroUpperKernel<<<grid, block>>>(d_A, n, j);
            }
        }
        
        // Update column j for rows below diagonal
        size_t update_start = std::max(start_row, j + 1);
        if (update_start < end_row) {
            dim3 block(256);
            dim3 grid((end_row - update_start + block.x - 1) / block.x);
            columnUpdateKernel<<<grid, block>>>(d_A, d_Lcol, n, j, update_start, diag_val);
        }
        
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    // Copy result back
    CUDA_CHECK(cudaMemcpy(A.data(), d_A, n * n * sizeof(double), cudaMemcpyDeviceToHost));
    
    // Gather results to rank 0
    if (rank == 0) {
        for (int r = 1; r < size; ++r) {
            size_t r_start = r * rows_per_rank + std::min((size_t)r, remainder);
            size_t r_end = r_start + rows_per_rank + (r < remainder ? 1 : 0);
            size_t r_count = r_end - r_start;
            
            MPI_Recv(A.data() + r_start * n, r_count * n, MPI_DOUBLE, r, 0, 
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    } else {
        MPI_Send(A.data() + start_row * n, (end_row - start_row) * n, MPI_DOUBLE, 0, 0, 
                MPI_COMM_WORLD);
    }
    
    // Broadcast final result to all ranks
    MPI_Bcast(A.data(), n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_Lcol));
    CUDA_CHECK(cudaFree(d_partial));
    
    return true;
}

// Generate a symmetric positive definite matrix
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    // Method: Create A = B * B^T where B is random
    // This guarantees A is positive semi-definite
    // Then add identity to make it strictly positive definite
    
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    
    // Generate random matrix B (parallel)
    #pragma omp parallel
    {
        unsigned int local_seed = seed + omp_get_thread_num();
        #pragma omp for
        for (size_t i = 0; i < n * n; ++i) {
            B[i] = (rand_r(&local_seed) / (double)RAND_MAX) - 0.5;
        }
    }
    
    // Compute A = B * B^T (parallel)
    #pragma omp parallel for collapse(2)
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
    
    // Compute L * L^T (parallel)
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += L[i * n + k] * L[j * n + k];
            }
            reconstructed[i * n + j] = sum;
        }
    }
    
    // Compare with original (parallel reduction)
    double maxError = 0.0;
    double relError = 0.0;
    
    #pragma omp parallel
    {
        double local_max = 0.0;
        double local_rel = 0.0;
        
        #pragma omp for
        for (size_t i = 0; i < n * n; ++i) {
            const double error = fabs(reconstructed[i] - A_orig[i]);
            local_max = std::max(local_max, error);
            
            const double rel = error / (fabs(A_orig[i]) + 1e-10);
            local_rel = std::max(local_rel, rel);
        }
        
        #pragma omp critical
        {
            maxError = std::max(maxError, local_max);
            relError = std::max(relError, local_rel);
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
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Initialize CUDA device (one device per rank)
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        if (rank == 0) {
            printf("Error: No CUDA devices found\n");
        }
        MPI_Finalize();
        return 1;
    }
    CUDA_CHECK(cudaSetDevice(rank % deviceCount));
    
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
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("MPI ranks: %d\n", size);
        printf("OpenMP threads per rank: %d\n", omp_get_max_threads());
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Allocate matrix
    std::vector<double> A(n * n);
    std::vector<double> A_orig;
    
    // Generate positive definite matrix (only rank 0)
    if (rank == 0) {
        printf("Generating positive definite matrix...\n");
        generatePositiveDefiniteMatrix(A, n);
        
        if (validate) {
            A_orig = A; // Save original for validation
        }
    }
    
    // Broadcast matrix to all ranks
    MPI_Bcast(A.data(), n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (validate) {
        if (rank == 0) {
            A_orig = A;
        }
    }
    
    // Perform Cholesky decomposition
    if (rank == 0) {
        printf("Computing Cholesky decomposition...\n");
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    bool success = choleskyDecomposition(A, n, rank, size);
    
    MPI_Barrier(MPI_COMM_WORLD);
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
        
        // Print results for external validation
        if (printResults) {
            print_results(A, "CholeskyL");
        }
        
        // Validation
        if (validate) {
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
    }
    
    MPI_Finalize();
    return 0;
}
