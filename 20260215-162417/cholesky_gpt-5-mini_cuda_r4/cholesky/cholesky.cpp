#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// CUDA error check
#define CUDA_CHECK(call) do { 
    cudaError_t err = call; 
    if (err != cudaSuccess) { 
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); 
        return false; 
    } 
} while(0)

// Device kernels
__global__ void diag_sum_kernel(double* A, int n, int k, double* d_sum) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    double local = 0.0;
    for (int t = idx; t < k; t += stride) {
        double v = A[k * (size_t)n + t];
        local += v * v;
    }
    if (local != 0.0) atomicAdd(d_sum, local);
}

__global__ void col_update_kernel(double* A, int n, int k) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int i = k + 1 + tid;
    if (i >= n) return;
    double sum = 0.0;
    for (int t = 0; t < k; ++t) {
        sum += A[i * (size_t)n + t] * A[k * (size_t)n + t];
    }
    A[i * (size_t)n + k] = (A[i * (size_t)n + k] - sum) / A[k * (size_t)n + k];
}

__global__ void trailing_update_kernel(double* A, int n, int k) {
    int i = k + 1 + blockIdx.x;
    int j = k + 1 + threadIdx.x;
    if (i >= n || j > i) return;
    A[i * (size_t)n + j] -= A[i * (size_t)n + k] * A[j * (size_t)n + k];
}

// GPU-accelerated Cholesky decomposition using simple hybrid approach
bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    // Allocate device memory and copy matrix
    const size_t bytes = n * n * sizeof(double);
    double* dA = nullptr;
    double* d_sum = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&dA, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_sum, sizeof(double)));
    CUDA_CHECK(cudaMemcpy(dA, A.data(), bytes, cudaMemcpyHostToDevice));

    const int THREADS = 256;

    for (int k = 0; k < (int)n; ++k) {
        // compute sum of squares for diagonal element (row k, columns 0..k-1)
        CUDA_CHECK(cudaMemset(d_sum, 0, sizeof(double)));
        int blocks = (k + THREADS - 1) / THREADS;
        if (blocks < 1) blocks = 1;
        diag_sum_kernel<<<blocks, THREADS>>>(dA, (int)n, k, d_sum);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // copy sum and diagonal element to host, perform sqrt and check
        double sum_host = 0.0;
        CUDA_CHECK(cudaMemcpy(&sum_host, d_sum, sizeof(double), cudaMemcpyDeviceToHost));
        double Akk = 0.0;
        CUDA_CHECK(cudaMemcpy(&Akk, dA + k * (size_t)n + k, sizeof(double), cudaMemcpyDeviceToHost));

        const double val = Akk - sum_host;
        if (val <= 0.0) {
            fprintf(stderr, "Error: Matrix is not positive definite at diagonal element %d\n", k);
            cudaFree(dA);
            cudaFree(d_sum);
            return false;
        }
        double diag = sqrt(val);
        CUDA_CHECK(cudaMemcpy(dA + k * (size_t)n + k, &diag, sizeof(double), cudaMemcpyHostToDevice));

        // update column k for rows i > k
        int rows = (int)n - (k + 1);
        if (rows > 0) {
            int col_blocks = (rows + THREADS - 1) / THREADS;
            col_update_kernel<<<col_blocks, THREADS>>>(dA, (int)n, k);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // rank-1 update of trailing submatrix: A[i,j] -= A[i,k] * A[j,k] for i>k, j>k, j<=i
        int tri_rows = (int)n - (k + 1);
        if (tri_rows > 0) {
            dim3 grid(tri_rows);
            dim3 block(THREADS);
            trailing_update_kernel<<<grid, block>>>(dA, (int)n, k);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // Zeroing upper triangle for row k to match sequential semantics (optional on device)
        // We'll zero on host after copy-back to keep kernels simpler
    }

    // copy back
    CUDA_CHECK(cudaMemcpy(A.data(), dA, bytes, cudaMemcpyDeviceToHost));

    // Zero out upper triangular part to keep same semantics as original implementation
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            A[i * n + j] = 0.0;
        }
    }

    cudaFree(dA);
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
