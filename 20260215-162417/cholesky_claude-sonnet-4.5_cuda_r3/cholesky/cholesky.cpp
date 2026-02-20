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
    } while(0)

// CUDA kernel for computing diagonal element with parallel reduction
__global__ void cholesky_diag_kernel(double* A, size_t n, size_t j) {
    __shared__ double partial_sums[256];
    
    int tid = threadIdx.x;
    int k = tid;
    
    // Each thread computes partial sum
    double sum = 0.0;
    while (k < j) {
        double val = A[j * n + k];
        sum += val * val;
        k += blockDim.x;
    }
    partial_sums[tid] = sum;
    __syncthreads();
    
    // Parallel reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            partial_sums[tid] += partial_sums[tid + s];
        }
        __syncthreads();
    }
    
    // Thread 0 computes final result
    if (tid == 0) {
        double val = A[j * n + j] - partial_sums[0];
        A[j * n + j] = sqrt(val);
    }
}

__global__ void cholesky_col_kernel(double* A, size_t n, size_t j) {
    // Each thread computes one element of column j below diagonal
    int i = blockIdx.x * blockDim.x + threadIdx.x + j + 1;
    
    if (i < n) {
        double sum = 0.0;
        // Unroll loop for better performance
        size_t k = 0;
        for (; k + 3 < j; k += 4) {
            sum += A[i * n + k] * A[j * n + k];
            sum += A[i * n + k + 1] * A[j * n + k + 1];
            sum += A[i * n + k + 2] * A[j * n + k + 2];
            sum += A[i * n + k + 3] * A[j * n + k + 3];
        }
        for (; k < j; ++k) {
            sum += A[i * n + k] * A[j * n + k];
        }
        A[i * n + j] = (A[i * n + j] - sum) / A[j * n + j];
    }
}

__global__ void zero_upper_kernel(double* A, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * n;
    
    // Each thread processes multiple elements
    for (int id = idx; id < total; id += blockDim.x * gridDim.x) {
        int i = id / n;
        int j = id % n;
        if (j > i) {
            A[id] = 0.0;
        }
    }
}

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    // Allocate device memory
    double* d_A;
    CUDA_CHECK(cudaMalloc(&d_A, n * n * sizeof(double)));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), n * n * sizeof(double), cudaMemcpyHostToDevice));
    
    // Use streams for better overlapping if possible
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    // Process column by column (inherently sequential due to data dependencies)
    for (size_t j = 0; j < n; ++j) {
        // Compute diagonal element with reduction
        int diagBlockSize = 256;
        cholesky_diag_kernel<<<1, diagBlockSize, 0, stream>>>(d_A, n, j);
        
        // Compute column below diagonal in parallel
        if (j + 1 < n) {
            int remaining = n - j - 1;
            int blockSize = 256;
            int numBlocks = (remaining + blockSize - 1) / blockSize;
            cholesky_col_kernel<<<numBlocks, blockSize, 0, stream>>>(d_A, n, j);
        }
    }
    
    // Wait for all column computations to complete
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    // Zero out upper triangular part
    int blockSize = 256;
    int numBlocks = (n * n + blockSize - 1) / blockSize;
    if (numBlocks > 1024) numBlocks = 1024; // Limit grid size
    zero_upper_kernel<<<numBlocks, blockSize, 0, stream>>>(d_A, n);
    
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(A.data(), d_A, n * n * sizeof(double), cudaMemcpyDeviceToHost));
    
    // Check for negative definite
    for (size_t i = 0; i < n; ++i) {
        if (std::isnan(A[i * n + i]) || A[i * n + i] <= 0.0) {
            printf("Error: Matrix is not positive definite at diagonal element %zu\n", i);
            CUDA_CHECK(cudaFree(d_A));
            return false;
        }
    }
    
    CUDA_CHECK(cudaFree(d_A));
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
