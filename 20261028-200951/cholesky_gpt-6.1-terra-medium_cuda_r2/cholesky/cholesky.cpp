#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// A blocked right-looking Cholesky factorization.  The diagonal block is
// small enough to factor in one CUDA block; the panel and Schur complement
// expose parallelism across the remaining rows and matrix tiles.
#include <cuda_runtime.h>

constexpr int BLOCK = 32;

static void cudaCheck(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", operation, cudaGetErrorString(status));
        exit(1);
    }
}

__global__ void factorDiagonal(double* A, size_t n, size_t start, int width, int* failure) {
    __shared__ double tile[BLOCK][BLOCK];
    int t = threadIdx.x;
    for (int idx = t; idx < width * width; idx += blockDim.x) {
        int row = idx / width, col = idx % width;
        tile[row][col] = A[(start + row) * n + start + col];
    }
    __syncthreads();
    for (int k = 0; k < width; ++k) {
        if (t == 0) {
            double value = tile[k][k];
            if (value <= 0.0 && *failure == 0) *failure = int(start + k + 1);
            tile[k][k] = sqrt(value);
        }
        __syncthreads();
        if (t > k && t < width) tile[t][k] /= tile[k][k];
        __syncthreads();
        for (int idx = t; idx < width * width; idx += blockDim.x) {
            int row = idx / width, col = idx % width;
            if (row > k && col > k && row >= col)
                tile[row][col] -= tile[row][k] * tile[col][k];
        }
        __syncthreads();
    }
    for (int idx = t; idx < width * width; idx += blockDim.x) {
        int row = idx / width, col = idx % width;
        if (row >= col) A[(start + row) * n + start + col] = tile[row][col];
    }
}

__global__ void solvePanel(double* A, size_t n, size_t start, int width) {
    size_t row = start + width + blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;
    for (int j = 0; j < width; ++j) {
        double value = A[row * n + start + j];
        for (int k = 0; k < j; ++k)
            value -= A[row * n + start + k] * A[(start + j) * n + start + k];
        A[row * n + start + j] = value / A[(start + j) * n + start + j];
    }
}

// Each 16x16 tile reuses panel values from shared memory.
__global__ void updateTrailing(double* A, size_t n, size_t start, int width) {
    constexpr int T = 16;
    __shared__ double left[T][BLOCK], right[T][BLOCK];
    int x = threadIdx.x, y = threadIdx.y;
    size_t base = start + width;
    size_t row = base + blockIdx.y * T + y;
    size_t col = base + blockIdx.x * T + x;
    if (blockIdx.x > blockIdx.y) return;
    for (int k = x; k < width; k += T) {
        left[y][k] = row < n ? A[row * n + start + k] : 0.0;
        size_t rightRow = base + blockIdx.x * T + y;
        right[y][k] = rightRow < n ? A[rightRow * n + start + k] : 0.0;
    }
    __syncthreads();
    if (row < n && col < n && row >= col) {
        double sum = 0.0;
        for (int k = 0; k < width; ++k) sum += left[y][k] * right[x][k];
        A[row * n + col] -= sum;
    }
}

__global__ void clearUpper(double* A, size_t n) {
    size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n * n && idx % n > idx / n) A[idx] = 0.0;
}

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    if (n == 0) return true;
    double* deviceA = nullptr;
    int* failure = nullptr;
    cudaCheck(cudaMalloc(&deviceA, n * n * sizeof(double)), "allocate matrix");
    cudaCheck(cudaMalloc(&failure, sizeof(int)), "allocate status");
    cudaCheck(cudaMemset(failure, 0, sizeof(int)), "initialize status");
    cudaCheck(cudaMemcpy(deviceA, A.data(), n * n * sizeof(double), cudaMemcpyHostToDevice), "copy matrix to GPU");
    for (size_t start = 0; start < n; start += BLOCK) {
        int width = int(std::min<size_t>(BLOCK, n - start));
        factorDiagonal<<<1, 256>>>(deviceA, n, start, width, failure);
        size_t remaining = n - start - width;
        if (remaining) {
            solvePanel<<<(remaining + 255) / 256, 256>>>(deviceA, n, start, width);
            unsigned tiles = unsigned((remaining + 15) / 16);
            updateTrailing<<<dim3(tiles, tiles), dim3(16, 16)>>>(deviceA, n, start, width);
        }
    }
    clearUpper<<<(n * n + 255) / 256, 256>>>(deviceA, n);
    cudaCheck(cudaGetLastError(), "launch factorization kernels");
    int error = 0;
    cudaCheck(cudaMemcpy(&error, failure, sizeof(int), cudaMemcpyDeviceToHost), "read factorization status");
    cudaCheck(cudaMemcpy(A.data(), deviceA, n * n * sizeof(double), cudaMemcpyDeviceToHost), "copy result to host");
    cudaCheck(cudaFree(failure), "free status");
    cudaCheck(cudaFree(deviceA), "free matrix");
    if (error) {
        printf("Error: Matrix is not positive definite at diagonal element %d\n", error - 1);
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
    
    // Allocate matrix
    std::vector<double> A(n * n);
    std::vector<double> A_orig;
    
    // Generate positive definite matrix
    printf("Generating positive definite matrix...\n");
    generatePositiveDefiniteMatrix(A, n);
    
    if (validate) {
        A_orig = A; // Save original for validation
    }
    
    // Initialize the CUDA context outside the timed factorization.
    cudaCheck(cudaFree(nullptr), "initialize CUDA");

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
