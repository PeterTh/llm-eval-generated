#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// Simple Cholesky decomposition (sequential, unblocked algorithm)
// Decomposes positive definite matrix A into L * L^T where L is lower triangular

__global__ void pivotKernel(double* a, size_t n, size_t p, int* bad) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const double v = a[p * n + p];
        if (v <= 0.0) { *bad = 1; return; }
        a[p * n + p] = sqrt(v);
    }
}

__global__ void panelUpdateKernel(double* a, size_t n, size_t p) {
    const size_t r = p + 1 + blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n) return;
    const double lr = a[r * n + p] / a[p * n + p];
    a[r * n + p] = lr;
}

__global__ void trailingUpdateKernel(double* a, size_t n, size_t p) {
    const size_t width = n - p - 1;
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= width * width) return;
    const size_t r = p + 1 + index / width;
    const size_t c = p + 1 + index % width;
    const double lr = a[r * n + p];
    const double lc = a[c * n + p];
    a[r * n + c] -= lr * lc;
}

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    double* deviceA = nullptr;
    int* deviceBad = nullptr;
    int bad = 0;
    if (cudaMalloc(&deviceA, n * n * sizeof(double)) != cudaSuccess || cudaMalloc(&deviceBad, sizeof(int)) != cudaSuccess) {
        fprintf(stderr, "CUDA allocation failed\n");
        if (deviceA) cudaFree(deviceA);
        if (deviceBad) cudaFree(deviceBad);
        return false;
    }
    cudaMemcpy(deviceA, A.data(), n * n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceBad, &bad, sizeof(int), cudaMemcpyHostToDevice);
    for (size_t p = 0; p < n; ++p) {
        pivotKernel<<<1, 1>>>(deviceA, n, p, deviceBad);
        const size_t width = n - p - 1;
        if (width) {
            const unsigned rows = static_cast<unsigned>((width + 255) / 256);
            panelUpdateKernel<<<rows, 256>>>(deviceA, n, p);
            trailingUpdateKernel<<<static_cast<unsigned>((width * width + 255) / 256), 256>>>(deviceA, n, p);
        }
    }
    cudaError_t err = cudaGetLastError();
    if (err == cudaSuccess) err = cudaDeviceSynchronize();
    cudaMemcpy(&bad, deviceBad, sizeof(int), cudaMemcpyDeviceToHost);
    if (err == cudaSuccess && !bad) cudaMemcpy(A.data(), deviceA, n * n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(deviceA);
    cudaFree(deviceBad);
    if (err != cudaSuccess) { fprintf(stderr, "CUDA execution failed: %s\n", cudaGetErrorString(err)); return false; }
    if (bad) { printf("Error: Matrix is not positive definite\n"); return false; }
    for (size_t i = 0; i < n; ++i) for (size_t j = i + 1; j < n; ++j) A[i * n + j] = 0.0;
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
