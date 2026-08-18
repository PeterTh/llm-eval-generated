#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// A blocked, right-looking Cholesky factorization.  The panel and Schur-complement
// updates are independent across matrix tiles and run on the GPU; only the small
// diagonal block has the inherent Cholesky dependency chain.
namespace {
constexpr int kBlockSize = 32;
constexpr int kTileSize = 16;

bool cudaOk(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(status));
        return false;
    }
    return true;
}

__global__ void factorDiagonalBlock(double* a, size_t n, size_t begin, int width,
                                    int* notPositiveDefinite) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    for (int col = 0; col < width; ++col) {
        const size_t j = begin + static_cast<size_t>(col);
        double diagonal = a[j * n + j];
        for (int k = 0; k < col; ++k) {
            const double x = a[j * n + begin + k];
            diagonal -= x * x;
        }
        if (diagonal <= 0.0) {
            *notPositiveDefinite = 1;
            return;
        }
        const double pivot = sqrt(diagonal);
        a[j * n + j] = pivot;
        for (int row = col + 1; row < width; ++row) {
            const size_t i = begin + static_cast<size_t>(row);
            double value = a[i * n + j];
            for (int k = 0; k < col; ++k)
                value -= a[i * n + begin + k] * a[j * n + begin + k];
            a[i * n + j] = value / pivot;
        }
    }
}

__global__ void solvePanelColumn(double* a, size_t n, size_t begin, size_t rows, int width, int col) {
    const size_t row = begin + width + static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const size_t j = begin + static_cast<size_t>(col);
    double value = a[row * n + j];
    for (int k = 0; k < col; ++k)
        value -= a[row * n + begin + k] * a[j * n + begin + k];
    a[row * n + j] = value / a[j * n + j];
}

__global__ void updateTrailing(double* a, size_t n, size_t begin, size_t rows, int width) {
    const size_t row = begin + width + static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const size_t col = begin + width + static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    __shared__ double rowPanel[kTileSize][kBlockSize];
    __shared__ double colPanel[kTileSize][kBlockSize];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    if (tx < width)
        rowPanel[ty][tx] = row < rows ? a[row * n + begin + tx] : 0.0;
    if (ty < width)
        colPanel[tx][ty] = col < rows ? a[col * n + begin + ty] : 0.0;
    if (tx + kTileSize < width)
        rowPanel[ty][tx + kTileSize] = row < rows ? a[row * n + begin + tx + kTileSize] : 0.0;
    if (ty + kTileSize < width)
        colPanel[tx][ty + kTileSize] = col < rows ? a[col * n + begin + ty + kTileSize] : 0.0;
    __syncthreads();
    if (row >= rows || col >= rows) return;
    double value = a[row * n + col];
    #pragma unroll
    for (int k = 0; k < kBlockSize; ++k) {
        if (k < width)
            value -= rowPanel[ty][k] * colPanel[tx][k];
    }
    a[row * n + col] = value;
}

}  // namespace

bool choleskyDecomposition(std::vector<double>& A, const size_t n, float& elapsedMs) {
    elapsedMs = 0.0f;
    if (n == 0) return true;
    double* deviceA = nullptr;
    int* deviceFailure = nullptr;
    cudaEvent_t start = nullptr, stop = nullptr;
    const size_t bytes = n * n * sizeof(double);
    int failed = 0;
    bool ok = cudaOk(cudaMalloc(&deviceA, bytes), "matrix allocation") &&
              cudaOk(cudaMalloc(&deviceFailure, sizeof(int)), "status allocation") &&
              cudaOk(cudaMemset(deviceFailure, 0, sizeof(int)), "status initialization") &&
              cudaOk(cudaMemcpy(deviceA, A.data(), bytes, cudaMemcpyHostToDevice), "matrix upload") &&
              cudaOk(cudaEventCreate(&start), "start event creation") &&
              cudaOk(cudaEventCreate(&stop), "stop event creation");
    if (!ok) goto cleanup;

    cudaEventRecord(start);
    for (size_t begin = 0; begin < n; begin += kBlockSize) {
        const int width = static_cast<int>(std::min(static_cast<size_t>(kBlockSize), n - begin));
        factorDiagonalBlock<<<1, 1>>>(deviceA, n, begin, width, deviceFailure);
        if (!cudaOk(cudaGetLastError(), "diagonal block factorization")) { ok = false; goto cleanup; }
        const size_t firstTrailing = begin + width;
        if (firstTrailing < n) {
            for (int col = 0; col < width; ++col) {
                solvePanelColumn<<<(n - firstTrailing + 255) / 256, 256>>>(
                    deviceA, n, begin, n, width, col);
                if (!cudaOk(cudaGetLastError(), "panel solve")) { ok = false; goto cleanup; }
            }
            const dim3 threads(kTileSize, kTileSize);
            const dim3 updateGrid((n - firstTrailing + kTileSize - 1) / kTileSize,
                                  (n - firstTrailing + kTileSize - 1) / kTileSize);
            updateTrailing<<<updateGrid, threads>>>(deviceA, n, begin, n, width);
            if (!cudaOk(cudaGetLastError(), "trailing update")) { ok = false; goto cleanup; }
        }
    }
    ok = cudaOk(cudaEventRecord(stop), "stop event recording") &&
         cudaOk(cudaEventSynchronize(stop), "factorization synchronization") &&
         cudaOk(cudaEventElapsedTime(&elapsedMs, start, stop), "elapsed-time measurement");
    if (ok) ok = cudaOk(cudaMemcpy(&failed, deviceFailure, sizeof(int), cudaMemcpyDeviceToHost), "status download");
    if (ok && failed) {
        printf("Error: Matrix is not positive definite\n");
        ok = false;
    }
    if (ok) {
        ok = cudaOk(cudaMemcpy(A.data(), deviceA, bytes, cudaMemcpyDeviceToHost), "factor download");
        for (size_t row = 0; row < n; ++row)
            for (size_t col = row + 1; col < n; ++col)
                A[row * n + col] = 0.0;
    }

cleanup:
    if (start) cudaEventDestroy(start);
    if (stop) cudaEventDestroy(stop);
    if (deviceFailure) cudaFree(deviceFailure);
    if (deviceA) cudaFree(deviceA);
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
    float elapsedMs = 0.0f;
    bool success = choleskyDecomposition(A, n, elapsedMs);
    
    if (!success) {
        printf("Cholesky decomposition failed\n");
        return 1;
    }
    
    printf("Computation time: %.3f ms\n", elapsedMs);
    
    // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
    double ops = (double)n * n * n / 3.0;
    double gflops = elapsedMs > 0.0f ? ops / (elapsedMs / 1000.0) / 1e9 : 0.0;
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
