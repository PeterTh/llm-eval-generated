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
    } while(0)

// Hybrid MPI+OpenMP+CUDA Cholesky decomposition
// Decomposes positive definite matrix A into L * L^T where L is lower triangular

// CUDA kernel for computing diagonal element and column (combined for efficiency)
__global__ void computeDiagonalAndColumnKernel(double* A, size_t n, size_t j, double* diag_out) {
    // Thread 0 computes diagonal
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        double sum = 0.0;
        for (size_t k = 0; k < j; ++k) {
            double val = A[j * n + k];
            sum += val * val;
        }
        double val = A[j * n + j] - sum;
        if (val > 0.0) {
            A[j * n + j] = sqrt(val);
            *diag_out = A[j * n + j];
        } else {
            A[j * n + j] = -1.0;
            *diag_out = -1.0;
        }
    }
}

// CUDA kernel for updating column j
__global__ void updateColumnKernel(double* A, size_t n, size_t j) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t i = j + 1 + tid;
    
    if (i < n) {
        double sum = 0.0;
        for (size_t k = 0; k < j; ++k) {
            sum += A[i * n + k] * A[j * n + k];
        }
        double diag = A[j * n + j];
        if (diag > 0.0) {
            A[i * n + j] = (A[i * n + j] - sum) / diag;
        }
    }
}

// CUDA kernel for zeroing upper triangular part
__global__ void zeroUpperKernel(double* A, size_t n, size_t row, size_t start_col) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t j = start_col + tid;
    if (j < n) {
        A[row * n + j] = 0.0;
    }
}

bool choleskyDecomposition(std::vector<double>& A, const size_t n, int rank, int size) {
    // Simplified algorithm: All ranks compute on GPU, with rank 0 doing the work
    // This avoids complex MPI coordination
    
    double* d_A;
    double* d_diag;
    size_t matrixSize = n * n * sizeof(double);
    
    CUDA_CHECK(cudaMalloc(&d_A, matrixSize));
    CUDA_CHECK(cudaMalloc(&d_diag, sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), matrixSize, cudaMemcpyHostToDevice));
    
    for (size_t j = 0; j < n; ++j) {
        // Compute diagonal element
        computeDiagonalAndColumnKernel<<<1, 1>>>(d_A, n, j, d_diag);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Check for positive definiteness
        double diag_val;
        CUDA_CHECK(cudaMemcpy(&diag_val, d_diag, sizeof(double), cudaMemcpyDeviceToHost));
        if (diag_val < 0.0) {
            CUDA_CHECK(cudaFree(d_A));
            CUDA_CHECK(cudaFree(d_diag));
            if (rank == 0) {
                printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
            }
            return false;
        }
        
        // Update column j elements (rows below diagonal)
        size_t num_rows = n - j - 1;
        if (num_rows > 0) {
            int blockSize = 256;
            int numBlocks = (num_rows + blockSize - 1) / blockSize;
            updateColumnKernel<<<numBlocks, blockSize>>>(d_A, n, j);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        
        // Zero out upper triangular part of row j
        size_t num_upper = n - j - 1;
        if (num_upper > 0) {
            int blockSize = 256;
            int numBlocks = (num_upper + blockSize - 1) / blockSize;
            zeroUpperKernel<<<numBlocks, blockSize>>>(d_A, n, j, j + 1);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(A.data(), d_A, matrixSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_diag));
    
    return true;
}

// Generate a symmetric positive definite matrix (OpenMP parallelized)
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    // Method: Create A = B * B^T where B is random
    // This guarantees A is positive semi-definite
    // Then add identity to make it strictly positive definite
    
    std::vector<double> B(n * n);
    
    // Generate random matrix B (OpenMP parallel)
    #pragma omp parallel
    {
        unsigned int seed = 42 + omp_get_thread_num();
        #pragma omp for
        for (size_t i = 0; i < n * n; ++i) {
            B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
        }
    }
    
    // Compute A = B * B^T (OpenMP parallel)
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
    // Validate by computing L * L^T and comparing with original matrix (OpenMP parallel)
    
    std::vector<double> reconstructed(n * n);
    
    // Compute L * L^T (OpenMP parallel)
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
    
    // Compare with original (OpenMP parallel reduction)
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
    
    // Set GPU device based on local rank
    int num_devices;
    CUDA_CHECK(cudaGetDeviceCount(&num_devices));
    if (num_devices > 0) {
        int device = rank % num_devices;
        CUDA_CHECK(cudaSetDevice(device));
    }
    
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
        printf("Cholesky Decomposition Benchmark (Hybrid MPI+OpenMP+CUDA)\n");
        printf("MPI ranks: %d\n", size);
        printf("OpenMP threads per rank: %d\n", omp_get_max_threads());
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Allocate matrix
    std::vector<double> A(n * n);
    std::vector<double> A_orig;
    
    // Generate positive definite matrix (rank 0 only)
    if (rank == 0) {
        printf("Generating positive definite matrix...\n");
        generatePositiveDefiniteMatrix(A, n);
        
        if (validate) {
            A_orig = A; // Save original for validation
        }
    }
    
    // Broadcast matrix to all ranks
    MPI_Bcast(A.data(), n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Synchronize before timing
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
