#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// CUDA-accelerated Cholesky decomposition (unblocked, lower triangular)
// Decomposes positive definite matrix A into L * L^T where L is lower triangular

namespace {
constexpr int kBlockSize = 256;
constexpr int kTileSize = 16;

__global__ void diagSumKernel(const double* __restrict__ A, size_t n, size_t k, double* sum) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    double local = 0.0;
    for (size_t j = idx; j < k; j += blockDim.x * gridDim.x) {
        double val = A[k * n + j];
        local += val * val;
    }

    extern __shared__ double shared[];
    shared[threadIdx.x] = local;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(sum, shared[0]);
    }
}

__global__ void diagUpdateKernel(double* A, size_t n, size_t k, const double* sum, int* info) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        double val = A[k * n + k] - *sum;
        if (val <= 0.0) {
            *info = 1;
            return;
        }
        A[k * n + k] = sqrt(val);
    }
}

__global__ void columnUpdateKernel(double* __restrict__ A, size_t n, size_t k) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x + k + 1;
    if (i >= n) {
        return;
    }

    double sum = 0.0;
    for (size_t j = 0; j < k; ++j) {
        sum += A[i * n + j] * A[k * n + j];
    }
    A[i * n + k] = (A[i * n + k] - sum) / A[k * n + k];
}

__global__ void trailingUpdateKernel(double* __restrict__ A, size_t n, size_t k) {
    size_t row = blockIdx.y * blockDim.y + threadIdx.y + k + 1;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x + k + 1;
    if (row >= n || col >= n || row < col) {
        return;
    }

    A[row * n + col] -= A[row * n + k] * A[col * n + k];
}

__global__ void zeroUpperKernel(double* __restrict__ A, size_t n) {
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n && col > row) {
        A[row * n + col] = 0.0;
    }
}
} // namespace

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
        printf("Error: No CUDA devices available\n");
        return false;
    }

    double* d_A = nullptr;
    double* d_sum = nullptr;
    int* d_info = nullptr;

    if (cudaSetDevice(0) != cudaSuccess) {
        printf("Error: Failed to set CUDA device\n");
        return false;
    }
    if (cudaMalloc(&d_A, n * n * sizeof(double)) != cudaSuccess ||
        cudaMalloc(&d_sum, sizeof(double)) != cudaSuccess ||
        cudaMalloc(&d_info, sizeof(int)) != cudaSuccess) {
        printf("Error: Failed to allocate CUDA buffers\n");
        if (d_A) {
            cudaFree(d_A);
        }
        if (d_sum) {
            cudaFree(d_sum);
        }
        if (d_info) {
            cudaFree(d_info);
        }
        return false;
    }

    if (cudaMemcpy(d_A, A.data(), n * n * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemset(d_info, 0, sizeof(int)) != cudaSuccess) {
        printf("Error: Failed to initialize CUDA buffers\n");
        cudaFree(d_A);
        cudaFree(d_sum);
        cudaFree(d_info);
        return false;
    }

    for (size_t k = 0; k < n; ++k) {
        if (cudaMemset(d_sum, 0, sizeof(double)) != cudaSuccess) {
            printf("Error: Failed to reset CUDA reduction buffer\n");
            cudaFree(d_A);
            cudaFree(d_sum);
            cudaFree(d_info);
            return false;
        }

        int sumBlocks = k > 0 ? static_cast<int>((k + kBlockSize - 1) / kBlockSize) : 0;
        if (sumBlocks > 0) {
            diagSumKernel<<<sumBlocks, kBlockSize, kBlockSize * sizeof(double)>>>(d_A, n, k, d_sum);
            if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
                printf("Error: Failed during CUDA diagonal reduction\n");
                cudaFree(d_A);
                cudaFree(d_sum);
                cudaFree(d_info);
                return false;
            }
        }

        diagUpdateKernel<<<1, 1>>>(d_A, n, k, d_sum, d_info);
        if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
            printf("Error: Failed during CUDA diagonal update\n");
            cudaFree(d_A);
            cudaFree(d_sum);
            cudaFree(d_info);
            return false;
        }

        int infoHost = 0;
        if (cudaMemcpy(&infoHost, d_info, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) {
            printf("Error: Failed to read CUDA status\n");
            cudaFree(d_A);
            cudaFree(d_sum);
            cudaFree(d_info);
            return false;
        }
        if (infoHost != 0) {
            printf("Error: Matrix is not positive definite at diagonal element %zu\n", k);
            cudaFree(d_A);
            cudaFree(d_sum);
            cudaFree(d_info);
            return false;
        }

        if (k + 1 < n) {
            int colBlocks = static_cast<int>((n - k - 1 + kBlockSize - 1) / kBlockSize);
            columnUpdateKernel<<<colBlocks, kBlockSize>>>(d_A, n, k);
            if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
                printf("Error: Failed during CUDA column update\n");
                cudaFree(d_A);
                cudaFree(d_sum);
                cudaFree(d_info);
                return false;
            }

            size_t remaining = n - k - 1;
            dim3 block(kTileSize, kTileSize);
            dim3 grid((remaining + block.x - 1) / block.x, (remaining + block.y - 1) / block.y);
            trailingUpdateKernel<<<grid, block>>>(d_A, n, k);
            if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
                printf("Error: Failed during CUDA trailing update\n");
                cudaFree(d_A);
                cudaFree(d_sum);
                cudaFree(d_info);
                return false;
            }
        }
    }

    dim3 block(kTileSize, kTileSize);
    dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);
    zeroUpperKernel<<<grid, block>>>(d_A, n);
    if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
        printf("Error: Failed during CUDA upper-triangular zeroing\n");
        cudaFree(d_A);
        cudaFree(d_sum);
        cudaFree(d_info);
        return false;
    }

    if (cudaMemcpy(A.data(), d_A, n * n * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) {
        printf("Error: Failed to copy results from CUDA device\n");
        cudaFree(d_A);
        cudaFree(d_sum);
        cudaFree(d_info);
        return false;
    }

    cudaFree(d_A);
    cudaFree(d_sum);
    cudaFree(d_info);
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
