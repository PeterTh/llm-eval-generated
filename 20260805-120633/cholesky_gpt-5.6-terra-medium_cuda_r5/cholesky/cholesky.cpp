#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// A blocked right-looking Cholesky factorization.  Keeping the matrix on the
// device throughout is important: only the small panel operations synchronize;
// the O(n^3) work is performed by the tiled trailing-update kernel.
namespace {
constexpr int PANEL = 32;
constexpr int TILE = 16;

inline bool cudaCheck(cudaError_t status, const char* where) {
    if (status != cudaSuccess) {
        printf("CUDA error at %s: %s\n", where, cudaGetErrorString(status));
        return false;
    }
    return true;
}

// Each panel is deliberately factored by one thread.  PANEL is small, while
// this avoids global-memory traffic and leaves the dominant update fully
// parallel.  The input matrix is guaranteed SPD by the benchmark generator.
__global__ void factorPanel(double* a, size_t n, size_t first, int width,
                            int* success, size_t* failedAt) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    for (int j = 0; j < width; ++j) {
        const size_t col = first + static_cast<size_t>(j);
        double diagonal = a[col * n + col];
        for (int k = 0; k < j; ++k) {
            const double value = a[col * n + first + k];
            diagonal -= value * value;
        }
        if (diagonal <= 0.0) {
            *success = 0;
            *failedAt = col;
            return;
        }
        const double pivot = sqrt(diagonal);
        a[col * n + col] = pivot;
        for (int row = j + 1; row < width; ++row) {
            const size_t target = first + static_cast<size_t>(row);
            double value = a[target * n + col];
            for (int k = 0; k < j; ++k)
                value -= a[target * n + first + k] * a[col * n + first + k];
            a[target * n + col] = value / pivot;
        }
    }
}

// Solve L21 * L11^T = A21, one independent matrix row per CUDA thread.
__global__ void solvePanel(double* a, size_t n, size_t first, int width) {
    const size_t row = first + static_cast<size_t>(width) +
                       static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= n) return;
    for (int j = 0; j < width; ++j) {
        const size_t col = first + static_cast<size_t>(j);
        double value = a[row * n + col];
        for (int k = 0; k < j; ++k)
            value -= a[row * n + first + k] * a[col * n + first + k];
        a[row * n + col] = value / a[col * n + col];
    }
}

// C(lower) -= L21 * L21^T.  Block tiles above the diagonal return immediately,
// so only the lower triangle is written and no unnecessary race-prone symmetry
// handling is needed.
__global__ void trailingUpdate(double* a, size_t n, size_t first, int width,
                               size_t trailing) {
    const size_t rowBase = first + static_cast<size_t>(width) + blockIdx.y * TILE;
    const size_t colBase = first + static_cast<size_t>(width) + blockIdx.x * TILE;
    if (colBase > rowBase) return;

    __shared__ double left[TILE][TILE];
    __shared__ double right[TILE][TILE];
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const size_t row = rowBase + ty;
    const size_t col = colBase + tx;
    double sum = 0.0;

    for (int k0 = 0; k0 < width; k0 += TILE) {
        const int kLeft = k0 + tx;
        const int kRight = k0 + ty;
        left[ty][tx] = (row < first + static_cast<size_t>(width) + trailing && kLeft < width)
            ? a[row * n + first + kLeft] : 0.0;
        right[ty][tx] = (col < first + static_cast<size_t>(width) + trailing && kRight < width)
            ? a[col * n + first + kRight] : 0.0;
        __syncthreads();
#pragma unroll
        for (int k = 0; k < TILE; ++k) sum += left[ty][k] * right[k][tx];
        __syncthreads();
    }
    if (row < n && col < n && col <= row) a[row * n + col] -= sum;
}

__global__ void clearUpper(double* a, size_t n) {
    const size_t col = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t row = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if (row < n && col > row && col < n) a[row * n + col] = 0.0;
}
} // namespace

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    if (n == 0) return true;
    double* deviceA = nullptr;
    int* deviceSuccess = nullptr;
    size_t* deviceFailure = nullptr;
    const size_t bytes = n * n * sizeof(double);
    bool ok = cudaCheck(cudaMalloc(&deviceA, bytes), "cudaMalloc(matrix)") &&
              cudaCheck(cudaMalloc(&deviceSuccess, sizeof(int)), "cudaMalloc(status)") &&
              cudaCheck(cudaMalloc(&deviceFailure, sizeof(size_t)), "cudaMalloc(failure)");
    if (ok) ok = cudaCheck(cudaMemcpy(deviceA, A.data(), bytes, cudaMemcpyHostToDevice), "matrix upload");
    int success = 1;
    if (ok) ok = cudaCheck(cudaMemcpy(deviceSuccess, &success, sizeof(success), cudaMemcpyHostToDevice), "status upload");

    for (size_t first = 0; ok && first < n; first += PANEL) {
        const int width = static_cast<int>(std::min(static_cast<size_t>(PANEL), n - first));
        factorPanel<<<1, 1>>>(deviceA, n, first, width, deviceSuccess, deviceFailure);
        ok = cudaCheck(cudaGetLastError(), "panel factorization launch");
        const size_t next = first + width;
        if (ok && next < n) {
            const size_t rows = n - next;
            solvePanel<<<static_cast<unsigned int>((rows + 255) / 256), 256>>>(deviceA, n, first, width);
            ok = cudaCheck(cudaGetLastError(), "panel solve launch");
            const size_t tiles = (rows + TILE - 1) / TILE;
            if (ok) {
                trailingUpdate<<<dim3(static_cast<unsigned int>(tiles), static_cast<unsigned int>(tiles)), dim3(TILE, TILE)>>>(deviceA, n, first, width, rows);
                ok = cudaCheck(cudaGetLastError(), "trailing update launch");
            }
        }
    }
    if (ok) {
        clearUpper<<<dim3(static_cast<unsigned int>((n + TILE - 1) / TILE), static_cast<unsigned int>((n + TILE - 1) / TILE)), dim3(TILE, TILE)>>>(deviceA, n);
        ok = cudaCheck(cudaGetLastError(), "upper-triangle clear launch") &&
             cudaCheck(cudaDeviceSynchronize(), "factorization");
    }
    size_t failedAt = 0;
    if (ok) ok = cudaCheck(cudaMemcpy(&success, deviceSuccess, sizeof(success), cudaMemcpyDeviceToHost), "status download");
    if (ok && !success) {
        cudaMemcpy(&failedAt, deviceFailure, sizeof(failedAt), cudaMemcpyDeviceToHost);
        printf("Error: Matrix is not positive definite at diagonal element %zu\n", failedAt);
        ok = false;
    }
    if (ok) ok = cudaCheck(cudaMemcpy(A.data(), deviceA, bytes, cudaMemcpyDeviceToHost), "matrix download");
    cudaFree(deviceFailure);
    cudaFree(deviceSuccess);
    cudaFree(deviceA);
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
