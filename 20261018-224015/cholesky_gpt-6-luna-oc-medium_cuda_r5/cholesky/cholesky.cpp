#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// The pivot order is sequential, while rows and trailing updates run in parallel.
__global__ void factorPivot(double* A, size_t n, size_t k, int* failed) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        double sum = 0.0;
        for (size_t j = 0; j < k; ++j) { const double x = A[k * n + j]; sum += x * x; }
        const double value = A[k * n + k] - sum;
        if (value <= 0.0) *failed = static_cast<int>(k + 1);
        else A[k * n + k] = sqrt(value);
    }
}

__global__ void factorColumn(double* A, size_t n, size_t k) {
    const size_t i = k + 1 + blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double sum = 0.0;
    for (size_t j = 0; j < k; ++j) sum += A[i * n + j] * A[k * n + j];
    A[i * n + k] = (A[i * n + k] - sum) / A[k * n + k];
}

__global__ void updateTrailing(double* A, size_t n, size_t k) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t i = k + 1 + index / (n - k - 1);
    const size_t j = k + 1 + index % (n - k - 1);
    if (i < n && j <= i) A[i * n + j] -= A[i * n + k] * A[j * n + k];
}

__global__ void zeroUpper(double* A, size_t n) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n * n && index / n < index % n) A[index] = 0.0;
}

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    double* deviceA = nullptr;
    int* deviceFailed = nullptr;
    int failed = 0;
    if (cudaMalloc(&deviceA, A.size() * sizeof(double)) != cudaSuccess ||
        cudaMalloc(&deviceFailed, sizeof(int)) != cudaSuccess) {
        if (deviceA) cudaFree(deviceA);
        if (deviceFailed) cudaFree(deviceFailed);
        fprintf(stderr, "CUDA allocation failed\n");
        return false;
    }
    cudaMemcpy(deviceA, A.data(), A.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceFailed, &failed, sizeof(int), cudaMemcpyHostToDevice);
    for (size_t k = 0; k < n; ++k) {
        factorPivot<<<1, 1>>>(deviceA, n, k, deviceFailed);
        if (k + 1 < n) {
            const size_t rows = n - k - 1;
            factorColumn<<<static_cast<unsigned>((rows+255)/256), 256>>>(deviceA, n, k);
            updateTrailing<<<static_cast<unsigned>((rows*rows+255)/256), 256>>>(deviceA, n, k);
        }
    }
    zeroUpper<<<static_cast<unsigned>((A.size()+255)/256), 256>>>(deviceA, n);
    const cudaError_t status = cudaDeviceSynchronize();
    cudaMemcpy(&failed, deviceFailed, sizeof(int), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) fprintf(stderr, "CUDA Cholesky failed: %s\n", cudaGetErrorString(status));
    if (status == cudaSuccess && failed == 0)
        cudaMemcpy(A.data(), deviceA, A.size() * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(deviceA);
    cudaFree(deviceFailed);
    if (status != cudaSuccess) return false;
    if (failed) printf("Error: Matrix is not positive definite at diagonal element %d\n", failed - 1);
    return failed == 0;
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
