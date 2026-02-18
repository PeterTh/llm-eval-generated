#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); \
        exit(1); \
    } \
}

// CUDA kernel: Matrix multiplication C = A * B^T for positive definite matrix generation
__global__ void matmul_transpose_kernel(const double* A, const double* B, double* C, size_t n) {
    size_t i = blockIdx.y * blockDim.y + threadIdx.y;
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n && j < n) {
        double sum = 0.0;
        for (size_t k = 0; k < n; ++k) {
            sum += A[i * n + k] * B[j * n + k];
        }
        C[i * n + j] = sum;
    }
}

// CUDA kernel: Add diagonal dominance (add n to diagonal)
__global__ void add_diagonal_kernel(double* A, size_t n, double val) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        A[i * n + i] += val;
    }
}

// CUDA kernel: Process column j - diagonal element
__global__ void cholesky_diag_kernel(double* A, size_t n, size_t j) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    __shared__ double local_sum;
    
    if (idx == 0) {
        double sum = 0.0;
        for (size_t k = 0; k < j; ++k) {
            sum += A[j * n + k] * A[j * n + k];
        }
        local_sum = sum;
    }
    __syncthreads();
    
    if (idx == 0) {
        double val = A[j * n + j] - local_sum;
        A[j * n + j] = sqrt(val);
    }
}

// CUDA kernel: Process row i, column j (off-diagonal)
__global__ void cholesky_offdiag_kernel(double* A, size_t n, size_t j) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x + j + 1;
    
    if (i >= n) return;
    
    double sum = 0.0;
    for (size_t k = 0; k < j; ++k) {
        sum += A[i * n + k] * A[j * n + k];
    }
    A[i * n + j] = (A[i * n + j] - sum) / A[j * n + j];
}

// CUDA kernel: Zero out upper triangular part
__global__ void zero_upper_triangle_kernel(double* A, size_t n, size_t i) {
    size_t j = blockIdx.x * blockDim.x + threadIdx.x + i + 1;
    if (j < n) {
        A[i * n + j] = 0.0;
    }
}

// CUDA kernel: Validate by computing L * L^T
__global__ void matmul_validate_kernel(const double* L, double* C, size_t n) {
    size_t i = blockIdx.y * blockDim.y + threadIdx.y;
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n && j < n) {
        double sum = 0.0;
        for (size_t k = 0; k < n; ++k) {
            sum += L[i * n + k] * L[j * n + k];
        }
        C[i * n + j] = sum;
    }
}

// Generate a symmetric positive definite matrix on GPU
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    // Allocate GPU memory
    double* d_B = nullptr;
    double* d_A = nullptr;
    
    CUDA_CHECK(cudaMalloc(&d_B, n * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_A, n * n * sizeof(double)));
    
    // Generate random matrix B on CPU
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }
    
    // Copy B to GPU
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), n * n * sizeof(double), cudaMemcpyHostToDevice));
    
    // Compute A = B * B^T on GPU
    dim3 block(16, 16);
    dim3 grid((n + 15) / 16, (n + 15) / 16);
    matmul_transpose_kernel<<<grid, block>>>(d_B, d_B, d_A, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Add diagonal dominance
    dim3 block_diag(256);
    dim3 grid_diag((n + 255) / 256);
    add_diagonal_kernel<<<grid_diag, block_diag>>>(d_A, n, (double)n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy result back to CPU
    CUDA_CHECK(cudaMemcpy(A.data(), d_A, n * n * sizeof(double), cudaMemcpyDeviceToHost));
    
    // Free GPU memory
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_A));
}

// Cholesky decomposition on GPU
bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    double* d_A = nullptr;
    
    CUDA_CHECK(cudaMalloc(&d_A, n * n * sizeof(double)));
    
    // Copy matrix to GPU
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), n * n * sizeof(double), cudaMemcpyHostToDevice));
    
    bool h_error = false;
    
    // Process each column
    for (size_t j = 0; j < n; ++j) {
        // Compute diagonal element
        dim3 block_diag(256);
        dim3 grid_diag(1);
        cholesky_diag_kernel<<<grid_diag, block_diag>>>(d_A, n, j);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Copy back diagonal element to check validity
        double diag_val;
        CUDA_CHECK(cudaMemcpy(&diag_val, &d_A[j * n + j], sizeof(double), cudaMemcpyDeviceToHost));
        
        if (diag_val <= 0.0) {
            printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
            h_error = true;
            break;
        }
        
        // Compute off-diagonal elements (parallelize over rows)
        if (j < n - 1) {
            size_t num_rows = n - j - 1;
            dim3 block_offdiag(256);
            dim3 grid_offdiag((num_rows + 255) / 256);
            cholesky_offdiag_kernel<<<grid_offdiag, block_offdiag>>>(d_A, n, j);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        
        // Zero out upper triangle row
        size_t num_cols = n - j - 1;
        if (num_cols > 0) {
            dim3 block_zero(256);
            dim3 grid_zero((num_cols + 255) / 256);
            zero_upper_triangle_kernel<<<grid_zero, block_zero>>>(d_A, n, j);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }
    
    // Copy result back to CPU
    CUDA_CHECK(cudaMemcpy(A.data(), d_A, n * n * sizeof(double), cudaMemcpyDeviceToHost));
    
    // Free GPU memory
    CUDA_CHECK(cudaFree(d_A));
    
    return !h_error;
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T on GPU and comparing with original matrix
    
    double* d_L = nullptr;
    double* d_reconstructed = nullptr;
    
    CUDA_CHECK(cudaMalloc(&d_L, n * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_reconstructed, n * n * sizeof(double)));
    
    // Copy L to GPU
    CUDA_CHECK(cudaMemcpy(d_L, L.data(), n * n * sizeof(double), cudaMemcpyHostToDevice));
    
    // Compute L * L^T on GPU
    dim3 block(16, 16);
    dim3 grid((n + 15) / 16, (n + 15) / 16);
    matmul_validate_kernel<<<grid, block>>>(d_L, d_reconstructed, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy result back to CPU
    std::vector<double> reconstructed(n * n);
    CUDA_CHECK(cudaMemcpy(reconstructed.data(), d_reconstructed, n * n * sizeof(double), cudaMemcpyDeviceToHost));
    
    // Free GPU memory
    CUDA_CHECK(cudaFree(d_L));
    CUDA_CHECK(cudaFree(d_reconstructed));
    
    // Compare with original on CPU
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
            printUsage(argv[0]);
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }
    
    printf("Cholesky Decomposition Benchmark\n");
    printf("Matrix size: %zu x %zu\n", n, n);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    // Allocate matrix
    std::vector<double> A(n * n);
    std::vector<double> A_orig;
    
    // Generate positive definite matrix
    printf("Generating positive definite matrix...\n");
    generatePositiveDefiniteMatrix(A, n);
    
    if (validate) {
        A_orig = A; // Save original for validation
    }
    
    // Perform Cholesky decomposition
    printf("Computing Cholesky decomposition...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    bool success = choleskyDecomposition(A, n);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (!success) {
        printf("Cholesky decomposition failed\n");
        return 1;
    }
    
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
            return 0;
        } else {
            printf("Validation: FAILED\n");
            return 1;
        }
    }
    
    return 0;
}
