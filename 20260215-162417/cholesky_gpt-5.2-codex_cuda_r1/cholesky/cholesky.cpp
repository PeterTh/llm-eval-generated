#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// CUDA Cholesky decomposition (right-looking, unblocked algorithm)
// Decomposes positive definite matrix A into L * L^T where L is lower triangular

namespace {
constexpr int kColumnThreads = 256;
constexpr int kUpdateBlock = 16;

bool checkCuda(cudaError_t err, const char* context) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error during %s: %s\n", context, cudaGetErrorString(err));
        return false;
    }
    return true;
}

__global__ void cholesky_diag_kernel(double* __restrict__ A, int n, int k, int* status, int* bad_index) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const int idx = k * n + k;
        const double val = A[idx];
        if (val <= 0.0) {
            if (atomicCAS(status, 0, 1) == 0) {
                *bad_index = k;
            }
            A[idx] = 0.0;
        } else {
            A[idx] = sqrt(val);
        }
    }
}

__global__ void cholesky_column_kernel(double* __restrict__ A, int n, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + k + 1;
    if (i < n) {
        const double diag = A[k * n + k];
        if (diag > 0.0) {
            const double inv_diag = 1.0 / diag;
            A[i * n + k] *= inv_diag;
        }
    }
}

__global__ void cholesky_update_kernel(double* __restrict__ A, int n, int k) {
    int i = blockIdx.y * blockDim.y + threadIdx.y + k + 1;
    int j = blockIdx.x * blockDim.x + threadIdx.x + k + 1;
    if (i < n && j <= i) {
        const double aik = A[i * n + k];
        const double ajk = A[j * n + k];
        A[i * n + j] -= aik * ajk;
    }
}

__global__ void zero_upper_kernel(double* __restrict__ A, int n) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && j < n && j > i) {
        A[i * n + j] = 0.0;
    }
}
} // namespace

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    if (n == 0) {
        return true;
    }
    if (n > static_cast<size_t>(std::numeric_limits<int>::max())) {
        printf("Error: Matrix size too large for CUDA indexing\n");
        return false;
    }
    const size_t elements = n * n;
    if (n != 0 && elements / n != n) {
        printf("Error: Matrix size overflow\n");
        return false;
    }
    const size_t bytes = elements * sizeof(double);
    int device_count = 0;
    if (!checkCuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount")) {
        return false;
    }
    if (device_count == 0) {
        printf("Error: No CUDA devices found\n");
        return false;
    }
    if (!checkCuda(cudaSetDevice(0), "cudaSetDevice")) {
        return false;
    }

    double* d_A = nullptr;
    int* d_status = nullptr;
    int* d_bad_index = nullptr;
    if (!checkCuda(cudaMalloc(&d_A, bytes), "cudaMalloc A")) {
        return false;
    }
    if (!checkCuda(cudaMalloc(&d_status, sizeof(int)), "cudaMalloc status")) {
        cudaFree(d_A);
        return false;
    }
    if (!checkCuda(cudaMalloc(&d_bad_index, sizeof(int)), "cudaMalloc bad_index")) {
        cudaFree(d_status);
        cudaFree(d_A);
        return false;
    }

    int status_init = 0;
    int bad_index_init = -1;
    if (!checkCuda(cudaMemcpy(d_A, A.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy H2D A") ||
        !checkCuda(cudaMemcpy(d_status, &status_init, sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy H2D status") ||
        !checkCuda(cudaMemcpy(d_bad_index, &bad_index_init, sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy H2D bad_index")) {
        cudaFree(d_bad_index);
        cudaFree(d_status);
        cudaFree(d_A);
        return false;
    }

    const int n_int = static_cast<int>(n);
    for (int k = 0; k < n_int; ++k) {
        cholesky_diag_kernel<<<1, 1>>>(d_A, n_int, k, d_status, d_bad_index);
        if (!checkCuda(cudaGetLastError(), "cholesky_diag_kernel")) {
            cudaFree(d_bad_index);
            cudaFree(d_status);
            cudaFree(d_A);
            return false;
        }

        const int tail = n_int - k - 1;
        if (tail > 0) {
            const int grid_col = (tail + kColumnThreads - 1) / kColumnThreads;
            cholesky_column_kernel<<<grid_col, kColumnThreads>>>(d_A, n_int, k);
            if (!checkCuda(cudaGetLastError(), "cholesky_column_kernel")) {
                cudaFree(d_bad_index);
                cudaFree(d_status);
                cudaFree(d_A);
                return false;
            }

            dim3 block(kUpdateBlock, kUpdateBlock);
            dim3 grid((tail + block.x - 1) / block.x, (tail + block.y - 1) / block.y);
            cholesky_update_kernel<<<grid, block>>>(d_A, n_int, k);
            if (!checkCuda(cudaGetLastError(), "cholesky_update_kernel")) {
                cudaFree(d_bad_index);
                cudaFree(d_status);
                cudaFree(d_A);
                return false;
            }
        }
    }

    dim3 block_zero(kUpdateBlock, kUpdateBlock);
    dim3 grid_zero((n_int + block_zero.x - 1) / block_zero.x, (n_int + block_zero.y - 1) / block_zero.y);
    zero_upper_kernel<<<grid_zero, block_zero>>>(d_A, n_int);
    if (!checkCuda(cudaGetLastError(), "zero_upper_kernel")) {
        cudaFree(d_bad_index);
        cudaFree(d_status);
        cudaFree(d_A);
        return false;
    }

    if (!checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize")) {
        cudaFree(d_bad_index);
        cudaFree(d_status);
        cudaFree(d_A);
        return false;
    }

    int status = 0;
    int bad_index = -1;
    if (!checkCuda(cudaMemcpy(&status, d_status, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy D2H status") ||
        !checkCuda(cudaMemcpy(&bad_index, d_bad_index, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy D2H bad_index")) {
        cudaFree(d_bad_index);
        cudaFree(d_status);
        cudaFree(d_A);
        return false;
    }

    if (status != 0) {
        printf("Error: Matrix is not positive definite at diagonal element %d\n", bad_index);
        cudaFree(d_bad_index);
        cudaFree(d_status);
        cudaFree(d_A);
        return false;
    }

    if (!checkCuda(cudaMemcpy(A.data(), d_A, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy D2H A")) {
        cudaFree(d_bad_index);
        cudaFree(d_status);
        cudaFree(d_A);
        return false;
    }

    cudaFree(d_bad_index);
    cudaFree(d_status);
    cudaFree(d_A);
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
