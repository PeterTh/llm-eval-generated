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

__global__ void diagonalKernel(double* a, size_t n, size_t j, double* blockSums) {
    extern __shared__ double s[];
    double sum = 0.0;
    for (size_t k = threadIdx.x; k < j; k += blockDim.x) {
        const double x = a[j * n + k];
        sum += x * x;
    }
    s[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride; stride >>= 1) {
        if (threadIdx.x < stride) s[threadIdx.x] += s[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) blockSums[blockIdx.x] = s[0];
}

__global__ void finishDiagonalKernel(double* a, size_t n, size_t j, const double* sums, unsigned blocks) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        double sum = 0.0;
        for (unsigned b = 0; b < blocks; ++b) sum += sums[b];
        const double value = a[j * n + j] - sum;
        a[j * n + j] = value > 0.0 ? sqrt(value) : nan("");
    }
}

__global__ void columnKernel(double* a, size_t n, size_t j) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x + j + 1;
    if (i >= n) return;
    double sum = 0.0;
    for (size_t k = 0; k < j; ++k) sum += a[i * n + k] * a[j * n + k];
    a[i * n + j] = (a[i * n + j] - sum) / a[j * n + j];
}

__global__ void clearUpperKernel(double* a, size_t n) {
    const size_t p = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t i = p / n, j = p % n;
    if (i < n && j > i) a[p] = 0.0;
}

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    double* dA = nullptr;
    if (cudaMalloc(&dA, A.size() * sizeof(double)) != cudaSuccess ||
        cudaMemcpy(dA, A.data(), A.size() * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess) return false;
    constexpr unsigned threads = 256;
    const unsigned maxBlocks = 64;
    double* sums = nullptr;
    if (cudaMalloc(&sums, maxBlocks * sizeof(double)) != cudaSuccess) { cudaFree(dA); return false; }
    for (size_t j = 0; j < n; ++j) {
        const unsigned blocks = static_cast<unsigned>(std::max<size_t>(1, std::min<size_t>(maxBlocks, (j + threads - 1) / threads)));
        diagonalKernel<<<blocks, threads, threads * sizeof(double)>>>(dA, n, j, sums);
        // A single block consumes the per-block reduction deterministically.
        finishDiagonalKernel<<<1, 1>>>(dA, n, j, sums, blocks);
        if (j + 1 < n)
            columnKernel<<<static_cast<unsigned>((n - j - 1 + threads - 1) / threads), threads>>>(dA, n, j);
    }
    clearUpperKernel<<<static_cast<unsigned>((A.size() + threads - 1) / threads), threads>>>(dA, n);
    const bool ok = cudaDeviceSynchronize() == cudaSuccess &&
                    cudaMemcpy(A.data(), dA, A.size() * sizeof(double), cudaMemcpyDeviceToHost) == cudaSuccess;
    cudaFree(sums); cudaFree(dA);
    return ok;
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
