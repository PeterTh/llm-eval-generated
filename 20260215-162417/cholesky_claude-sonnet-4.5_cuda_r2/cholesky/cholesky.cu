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

// Kernel to compute diagonal element using reduction
__global__ void computeDiagonalKernel(const double* __restrict__ A, size_t n, size_t j, double* result) {
    __shared__ double sdata[256];
    
    int tid = threadIdx.x;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread computes one element of the sum
    double sum = 0.0;
    if (k < j) {
        double val = A[j * n + k];
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
    
    // Write result for this block
    if (tid == 0) {
        atomicAdd(result, sdata[0]);
    }
}

// Kernel to update diagonal element
__global__ void updateDiagonalKernel(double* A, size_t n, size_t j, double sum) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        double val = A[j * n + j] - sum;
        if (val > 0.0) {
            A[j * n + j] = sqrt(val);
        } else {
            A[j * n + j] = -1.0; // Error flag
        }
    }
}

// Optimized kernel to compute off-diagonal elements using shared memory
__global__ void computeColumnKernelOptimized(double* __restrict__ A, size_t n, size_t j) {
    extern __shared__ double shared_row[];
    
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    
    // Cooperatively load row j into shared memory
    for (int k = tx; k < j; k += blockDim.x) {
        shared_row[k] = A[j * n + k];
    }
    __syncthreads();
    
    // Each thread processes one row
    int i = j + 1 + bx * blockDim.x + tx;
    
    if (i < n) {
        double sum = 0.0;
        // Use shared memory for row j
        #pragma unroll 4
        for (size_t k = 0; k < j; ++k) {
            sum += A[i * n + k] * shared_row[k];
        }
        double diag = A[j * n + j];
        A[i * n + j] = (A[i * n + j] - sum) / diag;
        
        // Zero out upper triangular element
        A[j * n + i] = 0.0;
    }
}

// Fast kernel for small columns
__global__ void computeColumnKernelFast(double* __restrict__ A, size_t n, size_t j) {
    int i = j + 1 + blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n) {
        double sum = 0.0;
        double diag_j = A[j * n + j];
        
        #pragma unroll 8
        for (size_t k = 0; k < j; ++k) {
            sum += A[i * n + k] * A[j * n + k];
        }
        A[i * n + j] = (A[i * n + j] - sum) / diag_j;
        
        // Zero out upper triangular element
        if (i < n) {
            A[j * n + i] = 0.0;
        }
    }
}

// Vectorized kernel for medium-sized columns
__global__ void computeColumnKernelVectorized(double* __restrict__ A, size_t n, size_t j) {
    int i = j + 1 + blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n) {
        double sum1 = 0.0, sum2 = 0.0, sum3 = 0.0, sum4 = 0.0;
        double diag_j = A[j * n + j];
        
        size_t k;
        // Process 4 elements at a time
        for (k = 0; k + 3 < j; k += 4) {
            sum1 += A[i * n + k] * A[j * n + k];
            sum2 += A[i * n + k + 1] * A[j * n + k + 1];
            sum3 += A[i * n + k + 2] * A[j * n + k + 2];
            sum4 += A[i * n + k + 3] * A[j * n + k + 3];
        }
        
        // Handle remaining elements
        for (; k < j; ++k) {
            sum1 += A[i * n + k] * A[j * n + k];
        }
        
        double sum = sum1 + sum2 + sum3 + sum4;
        A[i * n + j] = (A[i * n + j] - sum) / diag_j;
        
        // Zero out upper triangular element
        A[j * n + i] = 0.0;
    }
}

// Kernel to zero upper triangular efficiently
__global__ void zeroUpperTriangular(double* A, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = (n * (n - 1)) / 2;
    
    if (idx < total) {
        // Map linear index to (i, j) where j > i
        int i = 0;
        int remaining = idx;
        while (remaining >= (n - i - 1)) {
            remaining -= (n - i - 1);
            i++;
        }
        int j = i + 1 + remaining;
        A[i * n + j] = 0.0;
    }
}

// CUDA-accelerated Cholesky decomposition with optimized GPU kernels
bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    // Allocate device memory
    double* d_A;
    double* d_sum;
    CUDA_CHECK(cudaMalloc(&d_A, n * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_sum, sizeof(double)));
    
    // Copy matrix to device
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), n * n * sizeof(double), cudaMemcpyHostToDevice));
    
    const int THREADS = 256;
    const int SMALL_THRESHOLD = 32;
    const int MEDIUM_THRESHOLD = 128;
    
    // Perform Cholesky decomposition
    for (size_t j = 0; j < n; ++j) {
        // Compute diagonal element
        double h_sum = 0.0;
        CUDA_CHECK(cudaMemcpy(d_sum, &h_sum, sizeof(double), cudaMemcpyHostToDevice));
        
        if (j > 0) {
            int blocks = (j + THREADS - 1) / THREADS;
            computeDiagonalKernel<<<blocks, THREADS>>>(d_A, n, j, d_sum);
            CUDA_CHECK(cudaGetLastError());
            
            CUDA_CHECK(cudaMemcpy(&h_sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost));
        }
        
        updateDiagonalKernel<<<1, 1>>>(d_A, n, j, h_sum);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Check for error (negative value indicates non-positive-definite)
        double diag_val;
        CUDA_CHECK(cudaMemcpy(&diag_val, d_A + j * n + j, sizeof(double), cudaMemcpyDeviceToHost));
        if (diag_val < 0.0) {
            printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
            cudaFree(d_A);
            cudaFree(d_sum);
            return false;
        }
        
        // Compute off-diagonal elements in parallel
        size_t remaining_rows = n - j - 1;
        if (remaining_rows > 0) {
            int blocks = (remaining_rows + THREADS - 1) / THREADS;
            
            if (j < SMALL_THRESHOLD) {
                // For small j, use simpler kernel with loop unrolling
                computeColumnKernelFast<<<blocks, THREADS>>>(d_A, n, j);
            } else if (j < MEDIUM_THRESHOLD) {
                // For medium j, use vectorized kernel
                computeColumnKernelVectorized<<<blocks, THREADS>>>(d_A, n, j);
            } else {
                // For larger j, use shared memory optimization
                size_t shared_mem = j * sizeof(double);
                if (shared_mem <= 48 * 1024) { // Check shared memory limit
                    computeColumnKernelOptimized<<<blocks, THREADS, shared_mem>>>(d_A, n, j);
                } else {
                    // Fall back to vectorized if shared memory too large
                    computeColumnKernelVectorized<<<blocks, THREADS>>>(d_A, n, j);
                }
            }
            CUDA_CHECK(cudaGetLastError());
        }
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Zero upper triangular in one efficient kernel
    int total = (n * (n - 1)) / 2;
    if (total > 0) {
        int blocks = (total + THREADS - 1) / THREADS;
        zeroUpperTriangular<<<blocks, THREADS>>>(d_A, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(A.data(), d_A, n * n * sizeof(double), cudaMemcpyDeviceToHost));
    
    // Free device memory
    cudaFree(d_A);
    cudaFree(d_sum);
    
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
