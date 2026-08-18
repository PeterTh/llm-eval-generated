#include <algorithm>
#include <chrono>
#include <cmath>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// Blocked right-looking Cholesky.  The diagonal panel is small and inherently
// dependent; the panel solve and O(n^3) trailing update expose GPU parallelism.
constexpr int kBlock = 32;
constexpr int kTile = 16;

#define CUDA_CHECK(call) do { \
    const cudaError_t error_ = (call); \
    if (error_ != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\\n", __FILE__, __LINE__, cudaGetErrorString(error_)); \
        return false; \
    } \
} while (0)

__global__ void factorDiagonalKernel(double* a, int n, int offset, int width, int* failure) {
    __shared__ double panel[kBlock * kBlock];
    const int tid = threadIdx.x;
    for (int index = tid; index < width * width; index += blockDim.x) {
        const int row = index / width;
        const int col = index - row * width;
        panel[row * kBlock + col] = a[(offset + row) * n + offset + col];
    }
    __syncthreads();
    if (tid == 0) {
        for (int col = 0; col < width; ++col) {
            double diagonal = panel[col * kBlock + col];
            for (int k = 0; k < col; ++k) diagonal -= panel[col * kBlock + k] * panel[col * kBlock + k];
            if (diagonal <= 0.0) { *failure = 1; return; }
            const double pivot = sqrt(diagonal);
            panel[col * kBlock + col] = pivot;
            for (int row = col + 1; row < width; ++row) {
                double value = panel[row * kBlock + col];
                for (int k = 0; k < col; ++k) value -= panel[row * kBlock + k] * panel[col * kBlock + k];
                panel[row * kBlock + col] = value / pivot;
            }
            for (int upper = col + 1; upper < width; ++upper) panel[col * kBlock + upper] = 0.0;
        }
    }
    __syncthreads();
    for (int index = tid; index < width * width; index += blockDim.x) {
        const int row = index / width;
        const int col = index - row * width;
        a[(offset + row) * n + offset + col] = panel[row * kBlock + col];
    }
}

__global__ void solvePanelKernel(double* a, int n, int offset, int width) {
    const int row = offset + width + blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;
    for (int col = 0; col < width; ++col) {
        double value = a[row * n + offset + col];
        for (int k = 0; k < col; ++k) value -= a[row * n + offset + k] * a[(offset + col) * n + offset + k];
        a[row * n + offset + col] = value / a[(offset + col) * n + offset + col];
    }
}

__global__ void updateTrailingKernel(double* a, int n, int offset, int width) {
    const int blockRow = offset + width + blockIdx.y * kTile;
    const int blockCol = offset + width + blockIdx.x * kTile;
    if (blockCol > blockRow) return;
    const int row = blockRow + threadIdx.y;
    const int col = blockCol + threadIdx.x;
    if (row >= n || col >= n) return;
    double value = a[row * n + col];
    #pragma unroll
    for (int k = 0; k < kBlock; ++k) {
        if (k < width) value -= a[row * n + offset + k] * a[col * n + offset + k];
    }
    a[row * n + col] = value;
}

__global__ void zeroUpperKernel(double* a, int n) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < n && col > row) a[row * n + col] = 0.0;
}

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    if (n > static_cast<size_t>(INT_MAX)) {
        fprintf(stderr, "Matrix dimension is too large for CUDA indexing\n");
        return false;
    }
    // Pad to a complete panel.  This avoids a poorly utilized tail panel and
    // keeps every CUDA kernel on its fast, fixed-size path; the added rows are
    // an independent identity block and therefore do not affect A's factor.
    const int dimension = static_cast<int>((n + kBlock - 1) / kBlock * kBlock);
    std::vector<double> paddedMatrix(static_cast<size_t>(dimension) * dimension, 0.0);
    for (size_t row = 0; row < n; ++row) {
        std::copy_n(A.data() + row * n, n, paddedMatrix.data() + row * dimension);
    }
    for (int diagonal = static_cast<int>(n); diagonal < dimension; ++diagonal) {
        paddedMatrix[static_cast<size_t>(diagonal) * dimension + diagonal] = 1.0;
    }
    double* deviceA = nullptr;
    int* failure = nullptr;
    if (cudaMalloc(&deviceA, paddedMatrix.size() * sizeof(double)) != cudaSuccess ||
        cudaMalloc(&failure, sizeof(int)) != cudaSuccess) {
        fprintf(stderr, "CUDA allocation failed\n");
        cudaFree(deviceA); cudaFree(failure);
        return false;
    }
    if (cudaMemcpy(deviceA, paddedMatrix.data(), paddedMatrix.size() * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemset(failure, 0, sizeof(int)) != cudaSuccess) {
        fprintf(stderr, "CUDA transfer failed\n");
        cudaFree(deviceA); cudaFree(failure);
        return false;
    }

    for (int offset = 0; offset < dimension; offset += kBlock) {
        const int width = std::min(kBlock, dimension - offset);
        factorDiagonalKernel<<<1, 128>>>(deviceA, dimension, offset, width, failure);
        CUDA_CHECK(cudaGetLastError());
        if (offset + width < dimension) {
            solvePanelKernel<<<(dimension - offset - width + 255) / 256, 256>>>(deviceA, dimension, offset, width);
            CUDA_CHECK(cudaGetLastError());
            const int trailing = dimension - offset - width;
            const dim3 threads(kTile, kTile);
            const dim3 blocks((trailing + kTile - 1) / kTile, (trailing + kTile - 1) / kTile);
            updateTrailingKernel<<<blocks, threads>>>(deviceA, dimension, offset, width);
            CUDA_CHECK(cudaGetLastError());
        }
    }
    const dim3 clearThreads(16, 16);
    const dim3 clearBlocks((dimension + 15) / 16, (dimension + 15) / 16);
    zeroUpperKernel<<<clearBlocks, clearThreads>>>(deviceA, dimension);
    CUDA_CHECK(cudaGetLastError());

    int failed = 0;
    CUDA_CHECK(cudaMemcpy(&failed, failure, sizeof(int), cudaMemcpyDeviceToHost));
    if (!failed) CUDA_CHECK(cudaMemcpy(paddedMatrix.data(), deviceA, paddedMatrix.size() * sizeof(double), cudaMemcpyDeviceToHost));
    cudaFree(deviceA);
    cudaFree(failure);
    if (failed) {
        printf("Error: Matrix is not positive definite\n");
        return false;
    }
    for (size_t row = 0; row < n; ++row) {
        std::copy_n(paddedMatrix.data() + row * dimension, n, A.data() + row * n);
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
    
    // Perform Cholesky decomposition
    printf("Computing Cholesky decomposition...\n");
    // Create the CUDA context before timing; it is a one-time runtime cost, not
    // part of the decomposition itself.
    if (cudaFree(0) != cudaSuccess) {
        fprintf(stderr, "CUDA initialization failed: %s\n", cudaGetErrorString(cudaGetLastError()));
        return 1;
    }
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
