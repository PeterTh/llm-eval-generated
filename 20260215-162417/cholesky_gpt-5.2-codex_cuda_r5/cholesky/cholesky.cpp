#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// Simple Cholesky decomposition (CUDA parallel, right-looking algorithm)
// Decomposes positive definite matrix A into L * L^T where L is lower triangular

static bool checkCuda(cudaError_t result, const char* context) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", context, cudaGetErrorString(result));
        return false;
    }
    return true;
}

__global__ void cholesky_diagonal_kernel(double* __restrict__ A, size_t n, size_t j, int* info) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const double val = A[j * n + j];
        if (val <= 0.0) {
            atomicCAS(info, 0, static_cast<int>(j + 1));
            A[j * n + j] = 0.0;
            return;
        }
        A[j * n + j] = sqrt(val);
    }
}

__global__ void cholesky_scale_column(double* __restrict__ A, size_t n, size_t j) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x + j + 1;
    if (i < n) {
        A[i * n + j] /= A[j * n + j];
    }
}

__global__ void cholesky_update_trailing(double* __restrict__ A, size_t n, size_t j) {
    const size_t row = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y + j + 1;
    const size_t col = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x + j + 1;
    if (row < n && col <= row) {
        A[row * n + col] -= A[row * n + j] * A[col * n + j];
    }
}

__global__ void cholesky_zero_upper(double* __restrict__ A, size_t n) {
    const size_t row = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const size_t col = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row < n && col < n && col > row) {
        A[row * n + col] = 0.0;
    }
}

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    if (n == 0) {
        return true;
    }

    const size_t bytes = n * n * sizeof(double);
    double* d_A = nullptr;
    int* d_info = nullptr;
    bool ok = true;

    if (!checkCuda(cudaMalloc(&d_A, bytes), "cudaMalloc d_A")) {
        ok = false;
    }
    if (ok && !checkCuda(cudaMemcpy(d_A, A.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy H2D A")) {
        ok = false;
    }
    if (ok && !checkCuda(cudaMalloc(&d_info, sizeof(int)), "cudaMalloc d_info")) {
        ok = false;
    }
    if (ok && !checkCuda(cudaMemset(d_info, 0, sizeof(int)), "cudaMemset d_info")) {
        ok = false;
    }

    if (ok) {
        const dim3 updateBlock(16, 16);
        const int scaleBlock = 256;
        for (size_t j = 0; j < n; ++j) {
            cholesky_diagonal_kernel<<<1, 1>>>(d_A, n, j, d_info);
            if (!checkCuda(cudaGetLastError(), "cholesky_diagonal_kernel")) {
                ok = false;
                break;
            }

            const size_t remaining = n - j - 1;
            if (remaining > 0) {
                const dim3 scaleGrid(static_cast<unsigned int>((remaining + scaleBlock - 1) / scaleBlock), 1, 1);
                cholesky_scale_column<<<scaleGrid, scaleBlock>>>(d_A, n, j);
                if (!checkCuda(cudaGetLastError(), "cholesky_scale_column")) {
                    ok = false;
                    break;
                }

                const dim3 updateGrid(
                    static_cast<unsigned int>((remaining + updateBlock.x - 1) / updateBlock.x),
                    static_cast<unsigned int>((remaining + updateBlock.y - 1) / updateBlock.y),
                    1);
                cholesky_update_trailing<<<updateGrid, updateBlock>>>(d_A, n, j);
                if (!checkCuda(cudaGetLastError(), "cholesky_update_trailing")) {
                    ok = false;
                    break;
                }
            }
        }

        if (ok) {
            const dim3 zeroBlock(16, 16);
            const dim3 zeroGrid(
                static_cast<unsigned int>((n + zeroBlock.x - 1) / zeroBlock.x),
                static_cast<unsigned int>((n + zeroBlock.y - 1) / zeroBlock.y),
                1);
            cholesky_zero_upper<<<zeroGrid, zeroBlock>>>(d_A, n);
            ok = checkCuda(cudaGetLastError(), "cholesky_zero_upper");
        }
    }

    if (ok && !checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize")) {
        ok = false;
    }

    int info = 0;
    if (ok && !checkCuda(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy D2H info")) {
        ok = false;
    }

    if (ok && !checkCuda(cudaMemcpy(A.data(), d_A, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy D2H A")) {
        ok = false;
    }

    if (d_A != nullptr) {
        cudaFree(d_A);
    }
    if (d_info != nullptr) {
        cudaFree(d_info);
    }

    if (ok && info != 0) {
        printf("Error: Matrix is not positive definite at diagonal element %d\n", info - 1);
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
