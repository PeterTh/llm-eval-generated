#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// Right-looking Cholesky. Each pivot is ordered by the CUDA stream, while the
// column solve and O(n^2) trailing update are parallel GPU kernels.
__global__ void factorPivot(double* __restrict__ A, size_t n, size_t k,
                            unsigned long long* failedAt) {
    if (*failedAt != ~0ULL) return;
    const double value = A[k * n + k];
    if (!(value > 0.0)) {
        atomicMin(failedAt, static_cast<unsigned long long>(k));
        return;
    }
    A[k * n + k] = sqrt(value);
}

__global__ void solveColumn(double* __restrict__ A, size_t n, size_t k) {
    const size_t row = k + 1 + blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n) A[row * n + k] /= A[k * n + k];
}

__global__ void updateTrailing(double* __restrict__ A, size_t n, size_t k) {
    const size_t col = k + 1 + blockIdx.x * blockDim.x + threadIdx.x;
    const size_t row = k + 1 + blockIdx.y * blockDim.y + threadIdx.y;
    if (row < n && col <= row)
        A[row * n + col] -= A[row * n + k] * A[col * n + k];
}

__global__ void clearUpper(double* A, size_t n) {
    const size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < n && col < n && col > row) A[row * n + col] = 0.0;
}

static bool cudaOK(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return true;
    fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(status));
    return false;
}

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    if (n == 0) return true;
    double* deviceA = nullptr;
    unsigned long long* deviceFailure = nullptr;
    const size_t bytes = n * n * sizeof(double);
    const unsigned long long noFailure = ~0ULL;

    if (!cudaOK(cudaMalloc(&deviceA, bytes), "matrix allocation") ||
        !cudaOK(cudaMalloc(&deviceFailure, sizeof(*deviceFailure)), "status allocation")) {
        cudaFree(deviceA);
        cudaFree(deviceFailure);
        return false;
    }
    bool ok = cudaOK(cudaMemcpy(deviceA, A.data(), bytes, cudaMemcpyHostToDevice), "matrix upload") &&
              cudaOK(cudaMemcpy(deviceFailure, &noFailure, sizeof(noFailure),
                                cudaMemcpyHostToDevice), "status initialization");

    constexpr unsigned columnThreads = 256;
    const dim3 tile(32, 8);
    for (size_t k = 0; ok && k < n; ++k) {
        factorPivot<<<1, 1>>>(deviceA, n, k, deviceFailure);
        const size_t remaining = n - k - 1;
        if (remaining) {
            solveColumn<<<(remaining + columnThreads - 1) / columnThreads, columnThreads>>>(
                deviceA, n, k);
            const dim3 grid((remaining + tile.x - 1) / tile.x,
                            (remaining + tile.y - 1) / tile.y);
            updateTrailing<<<grid, tile>>>(deviceA, n, k);
        }
        ok = cudaOK(cudaPeekAtLastError(), "factorization kernel launch");
    }

    if (ok) {
        const dim3 grid((n + tile.x - 1) / tile.x, (n + tile.y - 1) / tile.y);
        clearUpper<<<grid, tile>>>(deviceA, n);
        ok = cudaOK(cudaMemcpy(A.data(), deviceA, bytes, cudaMemcpyDeviceToHost),
                    "result download");
    }
    unsigned long long failedAt = noFailure;
    if (ok) ok = cudaOK(cudaMemcpy(&failedAt, deviceFailure, sizeof(failedAt),
                                   cudaMemcpyDeviceToHost), "status download");
    cudaFree(deviceFailure);
    cudaFree(deviceA);
    if (ok && failedAt != noFailure) {
        printf("Error: Matrix is not positive definite at diagonal element %llu\n", failedAt);
        return false;
    }
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
    // Create the CUDA context before timing so one-time driver startup is not
    // reported as factorization work.
    if (!cudaOK(cudaFree(nullptr), "CUDA initialization")) return 1;
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
