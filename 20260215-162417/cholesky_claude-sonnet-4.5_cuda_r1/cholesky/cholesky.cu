#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// CUDA kernel for computing diagonal sum
__global__ void computeDiagonalSumKernel(double* A, double* result, size_t n, size_t j) {
    __shared__ double sdata[256];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    double sum = 0.0;
    if (idx < j) {
        double val = A[j * n + idx];
        sum = val * val;
    }
    
    sdata[tid] = sum;
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(result, sdata[0]);
    }
}

// CUDA kernel for computing off-diagonal elements in parallel
__global__ void choleskyOffDiagKernel(double* A, size_t n, size_t j) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x + j + 1;
    
    if (i < n) {
        double sum = 0.0;
        for (size_t k = 0; k < j; ++k) {
            sum += A[i * n + k] * A[j * n + k];
        }
        A[i * n + j] = (A[i * n + j] - sum) / A[j * n + j];
    }
}

// CUDA kernel for zeroing upper triangular part
__global__ void zeroUpperKernel(double* A, size_t n, size_t i) {
    size_t j = blockIdx.x * blockDim.x + threadIdx.x + i + 1;
    
    if (j < n) {
        A[i * n + j] = 0.0;
    }
}

// CUDA-accelerated Cholesky decomposition
bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    // Allocate device memory
    double* d_A;
    double* d_sum;
    CUDA_CHECK(cudaMalloc(&d_A, n * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_sum, sizeof(double)));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), n * n * sizeof(double), cudaMemcpyHostToDevice));
    
    const int blockSize = 256;
    
    // A is stored in row-major order
    for (size_t j = 0; j < n; ++j) {
        // Compute diagonal element
        double sum = 0.0;
        double diagonal_val = 0.0;
        
        if (j > 0) {
            // Reset sum on device
            CUDA_CHECK(cudaMemset(d_sum, 0, sizeof(double)));
            
            // Compute sum on GPU
            int numBlocks = (j + blockSize - 1) / blockSize;
            computeDiagonalSumKernel<<<numBlocks, blockSize>>>(d_A, d_sum, n, j);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            
            // Copy sum back
            CUDA_CHECK(cudaMemcpy(&sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost));
        }
        
        // Get current diagonal value
        CUDA_CHECK(cudaMemcpy(&diagonal_val, d_A + j * n + j, sizeof(double), cudaMemcpyDeviceToHost));
        
        const double val = diagonal_val - sum;
        if (val <= 0.0) {
            printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
            CUDA_CHECK(cudaFree(d_A));
            CUDA_CHECK(cudaFree(d_sum));
            return false;
        }
        
        diagonal_val = sqrt(val);
        
        // Copy diagonal element back to device
        CUDA_CHECK(cudaMemcpy(d_A + j * n + j, &diagonal_val, sizeof(double), cudaMemcpyHostToDevice));
        
        // Compute off-diagonal elements in parallel on GPU
        if (j + 1 < n) {
            size_t numElements = n - j - 1;
            int numBlocks = (numElements + blockSize - 1) / blockSize;
            choleskyOffDiagKernel<<<numBlocks, blockSize>>>(d_A, n, j);
            CUDA_CHECK(cudaGetLastError());
        }
        
        // Zero out upper triangular part in parallel
        if (j + 1 < n) {
            size_t numElements = n - j - 1;
            int numBlocks = (numElements + blockSize - 1) / blockSize;
            zeroUpperKernel<<<numBlocks, blockSize>>>(d_A, n, j);
            CUDA_CHECK(cudaGetLastError());
        }
        
        // Synchronize only every few iterations to reduce overhead
        if (j % 16 == 15 || j == n - 1) {
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }
    
    // Final synchronization
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(A.data(), d_A, n * n * sizeof(double), cudaMemcpyDeviceToHost));
    
    // Free device memory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_sum));
    
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
