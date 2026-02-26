#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

constexpr int kUpdateBlockX = 16;
constexpr int kUpdateBlockY = 16;
constexpr int kScaleBlockSize = 256;

static bool checkCuda(cudaError_t status, const char* context) {
    if (status != cudaSuccess) {
        printf("CUDA error %s: %s\n", context, cudaGetErrorString(status));
        return false;
    }
    return true;
}

__global__ void cholesky_diag_sqrt(double* A, size_t n, size_t k, int* error_flag) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const double val = A[k * n + k];
        if (val <= 0.0) {
            atomicCAS(error_flag, 0, static_cast<int>(k + 1));
            return;
        }
        A[k * n + k] = sqrt(val);
    }
}

__global__ void cholesky_scale_column(double* A, size_t n, size_t k) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x + k + 1;
    if (i < n) {
        A[i * n + k] /= A[k * n + k];
    }
}

__global__ void cholesky_rank1_update(double* A, size_t n, size_t k) {
    __shared__ double s_row[kUpdateBlockY];
    __shared__ double s_col[kUpdateBlockX];

    const size_t row = blockIdx.y * blockDim.y + threadIdx.y + k + 1;
    const size_t col = blockIdx.x * blockDim.x + threadIdx.x + k + 1;

    if (threadIdx.x == 0 && row < n) {
        s_row[threadIdx.y] = A[row * n + k];
    }
    if (threadIdx.y == 0 && col < n) {
        s_col[threadIdx.x] = A[col * n + k];
    }
    __syncthreads();

    if (row < n && col < n && row >= col) {
        A[row * n + col] -= s_row[threadIdx.y] * s_col[threadIdx.x];
    }
}

__global__ void zero_upper_triangle(double* A, size_t n) {
    const size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n && col > row) {
        A[row * n + col] = 0.0;
    }
}

// CUDA-accelerated Cholesky decomposition (right-looking algorithm)
// Decomposes positive definite matrix A into L * L^T where L is lower triangular
bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    const size_t bytes = n * n * sizeof(double);
    double* d_A = nullptr;
    int* d_error = nullptr;

    if (!checkCuda(cudaMalloc(&d_A, bytes), "cudaMalloc A")) {
        return false;
    }
    if (!checkCuda(cudaMalloc(&d_error, sizeof(int)), "cudaMalloc error")) {
        cudaFree(d_A);
        return false;
    }
    if (!checkCuda(cudaMemcpy(d_A, A.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy A H2D")) {
        cudaFree(d_A);
        cudaFree(d_error);
        return false;
    }
    if (!checkCuda(cudaMemset(d_error, 0, sizeof(int)), "cudaMemset error")) {
        cudaFree(d_A);
        cudaFree(d_error);
        return false;
    }

    const dim3 updateBlock(kUpdateBlockX, kUpdateBlockY);

    for (size_t k = 0; k < n; ++k) {
        cholesky_diag_sqrt<<<1, 1>>>(d_A, n, k, d_error);
        if (!checkCuda(cudaGetLastError(), "diag kernel")) {
            cudaFree(d_A);
            cudaFree(d_error);
            return false;
        }

        const size_t remaining = n - k - 1;
        if (remaining > 0) {
            const int scaleBlocks = static_cast<int>((remaining + kScaleBlockSize - 1) / kScaleBlockSize);
            cholesky_scale_column<<<scaleBlocks, kScaleBlockSize>>>(d_A, n, k);
            if (!checkCuda(cudaGetLastError(), "scale kernel")) {
                cudaFree(d_A);
                cudaFree(d_error);
                return false;
            }

            const dim3 updateGrid(
                static_cast<unsigned int>((remaining + kUpdateBlockX - 1) / kUpdateBlockX),
                static_cast<unsigned int>((remaining + kUpdateBlockY - 1) / kUpdateBlockY));
            cholesky_rank1_update<<<updateGrid, updateBlock>>>(d_A, n, k);
            if (!checkCuda(cudaGetLastError(), "rank1 update kernel")) {
                cudaFree(d_A);
                cudaFree(d_error);
                return false;
            }
        }
    }

    int h_error = 0;
    if (!checkCuda(cudaMemcpy(&h_error, d_error, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy error flag")) {
        cudaFree(d_A);
        cudaFree(d_error);
        return false;
    }
    if (h_error != 0) {
        printf("Error: Matrix is not positive definite at diagonal element %d\n", h_error - 1);
        cudaFree(d_A);
        cudaFree(d_error);
        return false;
    }

    const dim3 zeroBlock(kUpdateBlockX, kUpdateBlockY);
    const dim3 zeroGrid(
        static_cast<unsigned int>((n + kUpdateBlockX - 1) / kUpdateBlockX),
        static_cast<unsigned int>((n + kUpdateBlockY - 1) / kUpdateBlockY));
    zero_upper_triangle<<<zeroGrid, zeroBlock>>>(d_A, n);
    if (!checkCuda(cudaGetLastError(), "zero upper kernel")) {
        cudaFree(d_A);
        cudaFree(d_error);
        return false;
    }
    if (!checkCuda(cudaMemcpy(A.data(), d_A, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy A D2H")) {
        cudaFree(d_A);
        cudaFree(d_error);
        return false;
    }

    cudaFree(d_A);
    cudaFree(d_error);
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
