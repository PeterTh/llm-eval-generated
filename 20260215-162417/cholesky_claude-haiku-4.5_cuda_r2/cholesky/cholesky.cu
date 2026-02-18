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

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); \
            return false; \
        } \
    } while (0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS error: %d\n", status); \
            return false; \
        } \
    } while (0)

// Kernel: compute partial products for diagonal element
__global__ void compute_diag_products(const double* A, size_t n, size_t j, double* products) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < j) {
        products[idx] = A[j * n + idx] * A[j * n + idx];
    }
}

// Kernel: compute partial products for off-diagonal elements
__global__ void compute_offdiag_products(const double* A, size_t n, size_t i, size_t j, double* products) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < j) {
        products[idx] = A[i * n + idx] * A[j * n + idx];
    }
}

// GPU-optimized Cholesky using CUDA for parallelism
bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    double* d_A = nullptr;
    double* d_products = nullptr;
    
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    
    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_A, n * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_products, n * sizeof(double)));
    
    // Copy matrix to device
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), n * n * sizeof(double), cudaMemcpyHostToDevice));
    
    int threads_per_block = 256;
    
    // Main Cholesky loop
    for (size_t j = 0; j < n; ++j) {
        // Step 1: Compute diagonal element
        if (j > 0) {
            // Compute sum of squares: sum(L[j][k]^2) for k < j using GPU
            int blocks = (j + threads_per_block - 1) / threads_per_block;
            compute_diag_products<<<blocks, threads_per_block>>>(d_A, n, j, d_products);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            
            // Copy and sum on CPU (small vector)
            std::vector<double> h_products(j);
            CUDA_CHECK(cudaMemcpy(h_products.data(), d_products, j * sizeof(double), cudaMemcpyDeviceToHost));
            
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k) {
                sum += h_products[k];
            }
            
            double val = A[j * n + j] - sum;
            if (val <= 0.0) {
                printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
                CUDA_CHECK(cudaFree(d_A));
                CUDA_CHECK(cudaFree(d_products));
                cublasDestroy(handle);
                return false;
            }
            A[j * n + j] = sqrt(val);
        } else {
            if (A[0] <= 0.0) {
                printf("Error: Matrix is not positive definite at diagonal element 0\n");
                CUDA_CHECK(cudaFree(d_A));
                CUDA_CHECK(cudaFree(d_products));
                cublasDestroy(handle);
                return false;
            }
            A[0] = sqrt(A[0]);
        }
        
        // Update device with diagonal value
        CUDA_CHECK(cudaMemcpy(d_A + j * n + j, &A[j * n + j], sizeof(double), cudaMemcpyHostToDevice));
        
        // Step 2: Compute off-diagonal elements in column j with GPU parallelism
        if (j < n - 1) {
            // Parallelize computation of all off-diagonal elements in column j
            for (size_t i = j + 1; i < n; ++i) {
                if (j > 0) {
                    // Compute sum of products: sum(L[i][k] * L[j][k]) for k < j
                    int blocks = (j + threads_per_block - 1) / threads_per_block;
                    compute_offdiag_products<<<blocks, threads_per_block>>>(d_A, n, i, j, d_products);
                    CUDA_CHECK(cudaGetLastError());
                    CUDA_CHECK(cudaDeviceSynchronize());
                    
                    // Copy and sum on CPU
                    std::vector<double> h_products(j);
                    CUDA_CHECK(cudaMemcpy(h_products.data(), d_products, j * sizeof(double), cudaMemcpyDeviceToHost));
                    
                    double sum = 0.0;
                    for (size_t k = 0; k < j; ++k) {
                        sum += h_products[k];
                    }
                    A[i * n + j] = (A[i * n + j] - sum) / A[j * n + j];
                } else {
                    A[i * n + j] = A[i * n + j] / A[0];
                }
                
                // Update device
                CUDA_CHECK(cudaMemcpy(d_A + i * n + j, &A[i * n + j], sizeof(double), cudaMemcpyHostToDevice));
            }
        }
    }
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(A.data(), d_A, n * n * sizeof(double), cudaMemcpyDeviceToHost));
    
    // Zero out upper triangular part
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            A[i * n + j] = 0.0;
        }
    }
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_products));
    cublasDestroy(handle);
    
    return true;
}

// Generate a symmetric positive definite matrix
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
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
    
    // Add diagonal dominance
    for (size_t i = 0; i < n; ++i) {
        A[i * n + i] += n;
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
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
        A_orig = A;
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
    
    // Calculate GFLOPS
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
