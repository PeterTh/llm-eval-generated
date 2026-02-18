#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#define CHECK_CUDA(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        printf("Error: %s:%d, ", __FILE__, __LINE__); \
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}

// Kernel to compute square root of diagonal element
__global__ void kernel_diagonal(double* A, int n, int k, int* info) {
    if (threadIdx.x == 0) {
        double val = A[k * n + k];
        if (val <= 0.0) {
            *info = k; // Store the index of failure
            return;
        }
        A[k * n + k] = sqrt(val);
    }
}

// Kernel to scale the column below the diagonal
__global__ void kernel_column(double* A, int n, int k) {
    int i = k + 1 + blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        A[i * n + k] /= A[k * n + k];
    }
}

#define BLOCK_SIZE 16

// Kernel to update the trailing submatrix
__global__ void kernel_update(double* A, int n, int k) {
    int j = k + 1 + blockIdx.x * blockDim.x + threadIdx.x; // Column
    int i = k + 1 + blockIdx.y * blockDim.y + threadIdx.y; // Row

    __shared__ double col_k_cache_j[BLOCK_SIZE]; // Cache A[j][k]
    __shared__ double col_k_cache_i[BLOCK_SIZE]; // Cache A[i][k]

    // Load A[j][k] into shared memory
    // Each thread (tx, ty) computes for (j, i).
    // We need A[j][k] for all threads in a column of the block (same j).
    // We need A[i][k] for all threads in a row of the block (same i).

    // Let threads with threadIdx.y == 0 load col_k_cache_j
    if (threadIdx.y == 0) {
        if (j < n) {
            col_k_cache_j[threadIdx.x] = A[j * n + k];
        } else {
            col_k_cache_j[threadIdx.x] = 0.0;
        }
    }
    
    // Let threads with threadIdx.x == 0 load col_k_cache_i
    if (threadIdx.x == 0) {
        if (i < n) {
            col_k_cache_i[threadIdx.y] = A[i * n + k];
        } else {
            col_k_cache_i[threadIdx.y] = 0.0;
        }
    }
    
    __syncthreads();

    // Only process lower triangular part including diagonal
    if (i < n && j < n && j <= i) {
        double val = A[i * n + j];
        // Use cached values
        val -= col_k_cache_i[threadIdx.y] * col_k_cache_j[threadIdx.x];
        A[i * n + j] = val;
    }
}

// Simple Cholesky decomposition (CUDA parallel implementation)
// Decomposes positive definite matrix A into L * L^T where L is lower triangular

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    double* d_A = nullptr;
    int* d_info = nullptr;
    int h_info = -1;

    size_t size = n * n * sizeof(double);

    CHECK_CUDA(cudaMalloc((void**)&d_A, size));
    CHECK_CUDA(cudaMalloc((void**)&d_info, sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_info, &h_info, sizeof(int), cudaMemcpyHostToDevice)); // Initialize to -1
    CHECK_CUDA(cudaMemcpy(d_A, A.data(), size, cudaMemcpyHostToDevice));

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    
    for (int k = 0; k < (int)n; ++k) {
        // 1. Diagonal update
        kernel_diagonal<<<1, 1>>>(d_A, n, k, d_info);
        
        // 2. Column update
        int threadsPerBlock = 256;
        int num_rows = n - (k + 1);
        if (num_rows > 0) {
            int blocks = (num_rows + threadsPerBlock - 1) / threadsPerBlock;
            kernel_column<<<blocks, threadsPerBlock>>>(d_A, n, k);
        }

        // 3. Trailing submatrix update
        // Update submatrix starting at (k+1, k+1) of size num_rows x num_rows
        if (num_rows > 0) {
            // We need to cover j from k+1 to n-1 (cols)
            // and i from k+1 to n-1 (rows)
            // with constraint j <= i
            dim3 grid((num_rows + block.x - 1) / block.x, (num_rows + block.y - 1) / block.y);
            kernel_update<<<grid, block>>>(d_A, n, k);
        }
    }
    
    // Copy back result
    CHECK_CUDA(cudaMemcpy(A.data(), d_A, size, cudaMemcpyDeviceToHost));
    
    // Check for failure
    CHECK_CUDA(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_info));

    if (h_info != -1) {
        printf("Error: Matrix is not positive definite at diagonal element %d\n", h_info);
        return false;
    }
    
    // Zero out upper triangular part
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            A[i * n + j] = 0.0;
        }
    }

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
