#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// CUDA kernels for parallel Cholesky decomposition (GPU)
// Decomposes positive definite matrix A into L * L^T where L is lower triangular
// ---------------------------------------------------------------------------

// Kernel: compute diagonal element L[j][j] = sqrt(A[j][j] - sum_{k<j} L[j][k]^2)
// Uses a single block with parallel reduction for the sum of squares.
__global__ void choleskyDiagKernel(double* d_L, int n, int j, int* d_error) {
    __shared__ double sdata[256];

    double sum = 0.0;
    for (int k = threadIdx.x; k < j; k += blockDim.x) {
        double val = d_L[j * n + k];
        sum += val * val;
    }
    sdata[threadIdx.x] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        double val = d_L[j * n + j] - sdata[0];
        if (val <= 0.0) {
            *d_error = j;  // signal non-positive-definite
            d_L[j * n + j] = 0.0;
        } else {
            d_L[j * n + j] = sqrt(val);
        }
    }
}

// Kernel: compute off-diagonal elements L[i][j] for i > j
// One thread per row — each thread does a sequential dot product.
__global__ void choleskyOffDiagKernel(double* d_L, int n, int j) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + j + 1;
    if (i >= n) return;

    double sum = 0.0;
    for (int k = 0; k < j; ++k) {
        sum += d_L[i * n + k] * d_L[j * n + k];
    }
    d_L[i * n + j] = (d_L[i * n + j] - sum) / d_L[j * n + j];
}

// Kernel: zero out the strictly upper triangular part
__global__ void choleskyZeroUpperKernel(double* d_L, int n) {
    int i = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j > i && j < n) {
        d_L[i * n + j] = 0.0;
    }
}

// Host-side Cholesky decomposition using CUDA
bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    const int n_int = static_cast<int>(n);
    double* d_A = nullptr;
    int* d_error = nullptr;

    cudaError_t err;

    // Allocate device memory
    err = cudaMalloc(&d_A, n * n * sizeof(double));
    if (err != cudaSuccess) {
        printf("CUDA malloc failed: %s\n", cudaGetErrorString(err));
        return false;
    }
    err = cudaMalloc(&d_error, sizeof(int));
    if (err != cudaSuccess) {
        printf("CUDA malloc failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_A);
        return false;
    }

    // Copy input matrix to device
    err = cudaMemcpy(d_A, A.data(), n * n * sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("CUDA memcpy H2D failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_A);
        cudaFree(d_error);
        return false;
    }
    cudaMemset(d_error, -1, sizeof(int));

    const int blockSize = 256;

    // Main factorization loop — all kernels launch on the default stream,
    // which guarantees in-order execution on the GPU.
    for (int j = 0; j < n_int; ++j) {
        choleskyDiagKernel<<<1, blockSize>>>(d_A, n_int, j, d_error);

        int remainingRows = n_int - j - 1;
        if (remainingRows > 0) {
            int gridSize = (remainingRows + blockSize - 1) / blockSize;
            choleskyOffDiagKernel<<<gridSize, blockSize>>>(d_A, n_int, j);
        }
    }

    // Zero out the strictly upper triangle
    {
        dim3 blockDim2(blockSize, 1);
        dim3 gridDim2((n_int + blockSize - 1) / blockSize, n_int);
        choleskyZeroUpperKernel<<<gridDim2, blockDim2>>>(d_A, n_int);
    }

    // Wait for all GPU work to finish
    cudaDeviceSynchronize();

    // Copy result back to host
    err = cudaMemcpy(A.data(), d_A, n * n * sizeof(double), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("CUDA memcpy D2H failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_A);
        cudaFree(d_error);
        return false;
    }

    // Check error flag from device (non-positive-definite)
    int host_err = -1;
    cudaMemcpy(&host_err, d_error, sizeof(int), cudaMemcpyDeviceToHost);
    if (host_err >= 0) {
        printf("Error: Matrix is not positive definite at diagonal element %d\n", host_err);
        cudaFree(d_A);
        cudaFree(d_error);
        return false;
    }

    cudaFree(d_A);
    cudaFree(d_error);
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

    // Warm up CUDA context so driver initialisation doesn't pollute timing
    cudaFree(0);          // forces full context creation
    cudaDeviceSynchronize();

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
