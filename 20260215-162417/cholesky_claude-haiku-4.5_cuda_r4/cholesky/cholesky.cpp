#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "../common/results_output.hpp"

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error: %s\n", cudaGetErrorString(err)); \
            return false; \
        } \
    } while(0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            printf("cuBLAS error: %d\n", status); \
            return false; \
        } \
    } while(0)

// CUDA kernels for Cholesky decomposition

// Kernel to compute diagonal element L[j,j] = sqrt(A[j,j] - sum)
__global__ void computeDiagonalKernel(double* A, int j, int n, bool* isPositiveDefinite) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        double sum = 0.0;
        for (int k = 0; k < j; ++k) {
            sum += A[j * n + k] * A[j * n + k];
        }
        double val = A[j * n + j] - sum;
        if (val <= 0.0) {
            *isPositiveDefinite = false;
        } else {
            A[j * n + j] = sqrt(val);
        }
    }
}

// Kernel to perform rank-1 update on a row: A[i, j:] -= A[i, :j] * A[j, :j]
__global__ void rank1UpdateKernel(double* A, int j, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + j + 1;
    if (i < n) {
        double sum = 0.0;
        for (int k = 0; k < j; ++k) {
            sum += A[i * n + k] * A[j * n + k];
        }
        A[i * n + j] = (A[i * n + j] - sum) / A[j * n + j];
    }
}

// Kernel to zero out upper triangular part
__global__ void zeroUpperTriangularKernel(double* A, int i, int n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x + i + 1;
    if (j < n) {
        A[i * n + j] = 0.0;
    }
}

// CUDA Cholesky decomposition - Optimized GPU implementation
bool choleskyDecompositionCUDA(std::vector<double>& A, const size_t n) {
    double* d_A = nullptr;
    bool* d_isPositDef = nullptr;
    bool h_isPositDef = true;
    
    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_A, n * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_isPositDef, sizeof(bool)));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), n * n * sizeof(double), cudaMemcpyHostToDevice));
    
    // Initialize cublasHandle
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    
    const int blockSize = 512;
    
    // Main Cholesky loop - process columns with GPU parallelism
    for (size_t j = 0; j < n; ++j) {
        // Compute diagonal element L[j, j]
        CUDA_CHECK(cudaMemcpy(d_isPositDef, &h_isPositDef, sizeof(bool), cudaMemcpyHostToDevice));
        computeDiagonalKernel<<<1, 1>>>(d_A, j, n, d_isPositDef);
        CUDA_CHECK(cudaMemcpy(&h_isPositDef, d_isPositDef, sizeof(bool), cudaMemcpyDeviceToHost));
        
        if (!h_isPositDef) {
            printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
            cudaFree(d_A);
            cudaFree(d_isPositDef);
            cublasDestroy(handle);
            return false;
        }
        
        // Compute L[j+1:, j] - parallelize across rows
        if (j + 1 < n) {
            int numRows = n - j - 1;
            int numBlocks = (numRows + blockSize - 1) / blockSize;
            rank1UpdateKernel<<<numBlocks, blockSize>>>(d_A, j, n);
        }
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Zero out upper triangular part
    for (size_t i = 0; i < n; ++i) {
        int numCols = n - i - 1;
        if (numCols > 0) {
            int numBlocks = (numCols + blockSize - 1) / blockSize;
            zeroUpperTriangularKernel<<<numBlocks, blockSize>>>(d_A, i, n);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(A.data(), d_A, n * n * sizeof(double), cudaMemcpyDeviceToHost));
    
    // Clean up
    cudaFree(d_A);
    cudaFree(d_isPositDef);
    cublasDestroy(handle);
    
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
    
    bool success = choleskyDecompositionCUDA(A, n);
    
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
