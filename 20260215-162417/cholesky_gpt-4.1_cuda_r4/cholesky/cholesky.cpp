#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

__global__ void zero_upper(double* A, size_t n, size_t i) {
    size_t j = blockIdx.x * blockDim.x + threadIdx.x + i + 1;
    if (j < n) {
        A[i * n + j] = 0.0;
    }
}
__global__ void cholesky_offdiag_kernel(double* A, size_t n, size_t i) {
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < i) {
        double sum = 0.0;
        for (size_t k = 0; k < j; ++k) {
            sum += A[i * n + k] * A[j * n + k];
        }
        A[i * n + j] = (A[i * n + j] - sum) / A[j * n + j];
    }
}
__global__ void matmul_lower(double* L, double* out, size_t n) {
    size_t i = blockIdx.y * blockDim.y + threadIdx.y;
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && j < n) {
        double sum = 0.0;
        for (size_t k = 0; k < n; ++k) {
            sum += L[i * n + k] * L[j * n + k];
        }
        out[i * n + j] = sum;
    }
}
__global__ void generate_pd_kernel(const double* B, double* A, size_t n) {
    size_t i = blockIdx.y * blockDim.y + threadIdx.y;
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && j < n) {
        double sum = 0.0;
        for (size_t k = 0; k < n; ++k) {
            sum += B[i * n + k] * B[j * n + k];
        }
        A[i * n + j] = sum;
    }
}

// Host Cholesky wrapper using CUDA for parallelism
bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    // Allocate device memory
    double* d_A;
    CUDA_CHECK(cudaMalloc(&d_A, n * n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), n * n * sizeof(double), cudaMemcpyHostToDevice));

    for (size_t i = 0; i < n; ++i) {
        // Diagonal element (sequential, as dependency chain)
        double sum = 0.0;
        if (i > 0) {
            for (size_t k = 0; k < i; ++k) {
                double val;
                CUDA_CHECK(cudaMemcpy(&val, d_A + i * n + k, sizeof(double), cudaMemcpyDeviceToHost));
                sum += val * val;
            }
        }
        double diag;
        CUDA_CHECK(cudaMemcpy(&diag, d_A + i * n + i, sizeof(double), cudaMemcpyDeviceToHost));
        double val = diag - sum;
        if (val <= 0.0) {
            printf("Error: Matrix is not positive definite at diagonal element %zu\n", i);
            cudaFree(d_A);
            return false;
        }
        val = sqrt(val);
        CUDA_CHECK(cudaMemcpy(d_A + i * n + i, &val, sizeof(double), cudaMemcpyHostToDevice));

        // Off-diagonal elements (parallelize over i > j)
        size_t threads = 256;
        size_t blocks = (i + threads) / threads;
        cholesky_offdiag_kernel<<<blocks, threads>>>(d_A, n, i);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Zero out upper triangle (parallel)
        size_t threads2 = 256;
        size_t blocks2 = ((n - (i + 1)) + threads2 - 1) / threads2;
        zero_upper<<<blocks2, threads2>>>(d_A, n, i);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    CUDA_CHECK(cudaMemcpy(A.data(), d_A, n * n * sizeof(double), cudaMemcpyDeviceToHost));
    cudaFree(d_A);
    return true;
}

// CUDA kernel for off-diagonal update (i > j)
__global__ void cholesky_offdiag_kernel(double* A, size_t n, size_t i) {
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < i) {
        double sum = 0.0;
        for (size_t k = 0; k < j; ++k) {
            sum += A[i * n + k] * A[j * n + k];
        }
        A[i * n + j] = (A[i * n + j] - sum) / A[j * n + j];
    }
}

// Generate a symmetric positive definite matrix
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }
    // Parallelize A = B * B^T on GPU
    double *d_B, *d_A;
    CUDA_CHECK(cudaMalloc(&d_B, n * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_A, n * n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), n * n * sizeof(double), cudaMemcpyHostToDevice));
    dim3 threads(16, 16);
    dim3 blocks((n + 15) / 16, (n + 15) / 16);
    generate_pd_kernel<<<blocks, threads>>>(d_B, d_A, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(A.data(), d_A, n * n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_A));
    for (size_t i = 0; i < n; ++i) {
        A[i * n + i] += n;
    }
}

extern "C" __global__ void generate_pd_kernel(const double* B, double* A, size_t n) {
    size_t i = blockIdx.y * blockDim.y + threadIdx.y;
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && j < n) {
        double sum = 0.0;
        for (size_t k = 0; k < n; ++k) {
            sum += B[i * n + k] * B[j * n + k];
        }
        A[i * n + j] = sum;
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    std::vector<double> reconstructed(n * n);
    double *d_L, *d_reconstructed;
    CUDA_CHECK(cudaMalloc(&d_L, n * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_reconstructed, n * n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_L, L.data(), n * n * sizeof(double), cudaMemcpyHostToDevice));
    dim3 threads(16, 16);
    dim3 blocks((n + 15) / 16, (n + 15) / 16);
    matmul_lower<<<blocks, threads>>>(d_L, d_reconstructed, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(reconstructed.data(), d_reconstructed, n * n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_L));
    CUDA_CHECK(cudaFree(d_reconstructed));
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
