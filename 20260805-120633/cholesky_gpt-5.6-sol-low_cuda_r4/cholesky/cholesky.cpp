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

__global__ void factorDiagonal(double* A, size_t n, size_t k, int* failed) {
    const double value = A[k * n + k];
    if (!(value > 0.0)) {
        atomicCAS(failed, -1, static_cast<int>(k));
    } else {
        A[k * n + k] = sqrt(value);
    }
}

__global__ void scaleColumn(double* A, size_t n, size_t k) {
    const size_t row = k + 1 + blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n)
        A[row * n + k] /= A[k * n + k];
}

// A lower-triangular rank-1 update. Consecutive x threads access consecutive
// columns, producing coalesced loads/stores in the row-major matrix.
__global__ void updateTrailing(double* A, size_t n, size_t k) {
    const size_t col = k + 1 + blockIdx.x * blockDim.x + threadIdx.x;
    const size_t row = k + 1 + blockIdx.y * blockDim.y + threadIdx.y;
    if (row < n && col <= row)
        A[row * n + col] -= A[row * n + k] * A[col * n + k];
}

__global__ void clearUpper(double* A, size_t n) {
    const size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < n && col < n && col > row)
        A[row * n + col] = 0.0;
}

static bool cudaOk(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return true;
    fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(status));
    return false;
}

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    if (n == 0) return true;

    double* deviceA = nullptr;
    int* deviceFailed = nullptr;
    const size_t bytes = n * n * sizeof(double);
    if (!cudaOk(cudaMalloc(&deviceA, bytes), "matrix allocation") ||
        !cudaOk(cudaMalloc(&deviceFailed, sizeof(int)), "status allocation")) {
        cudaFree(deviceA);
        cudaFree(deviceFailed);
        return false;
    }

    int failed = -1;
    bool ok = cudaOk(cudaMemcpy(deviceA, A.data(), bytes, cudaMemcpyHostToDevice), "matrix upload") &&
              cudaOk(cudaMemcpy(deviceFailed, &failed, sizeof(int), cudaMemcpyHostToDevice), "status initialization");
    constexpr unsigned columnThreads = 256;
    const dim3 tile(32, 8);

    for (size_t k = 0; ok && k < n; ++k) {
        factorDiagonal<<<1, 1>>>(deviceA, n, k, deviceFailed);
        const size_t remaining = n - k - 1;
        if (remaining != 0) {
            scaleColumn<<<static_cast<unsigned>((remaining + columnThreads - 1) / columnThreads), columnThreads>>>(deviceA, n, k);
            const dim3 grid(static_cast<unsigned>((remaining + tile.x - 1) / tile.x),
                            static_cast<unsigned>((remaining + tile.y - 1) / tile.y));
            updateTrailing<<<grid, tile>>>(deviceA, n, k);
        }
        ok = cudaOk(cudaPeekAtLastError(), "Cholesky kernel launch");
    }

    if (ok) {
        const dim3 grid(static_cast<unsigned>((n + tile.x - 1) / tile.x),
                        static_cast<unsigned>((n + tile.y - 1) / tile.y));
        clearUpper<<<grid, tile>>>(deviceA, n);
        ok = cudaOk(cudaMemcpy(&failed, deviceFailed, sizeof(int), cudaMemcpyDeviceToHost), "status download") &&
             cudaOk(cudaMemcpy(A.data(), deviceA, bytes, cudaMemcpyDeviceToHost), "matrix download");
    }

    cudaFree(deviceFailed);
    cudaFree(deviceA);
    if (!ok) return false;
    if (failed >= 0) {
        printf("Error: Matrix is not positive definite at diagonal element %d\n", failed);
        return false;
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

    // Create the CUDA context before matrix generation so one-time driver
    // initialization is not charged to the factorization benchmark.
    if (!cudaOk(cudaFree(nullptr), "CUDA initialization")) return 1;
    
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
