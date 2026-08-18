#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

namespace {

constexpr int TILE = 32;

__global__ void factorDiagonal(double* a, size_t n, size_t tile, size_t rows) {
    __shared__ double s[TILE][TILE];
    const int t = threadIdx.x;
    const size_t base = tile * TILE;
    s[t][0] = 0.0;
    for (int col = 0; col < TILE; ++col)
        s[t][col] = (base + t < n && base + col < n) ? a[(base + t) * n + base + col] : 0.0;
    __syncthreads();

    for (int j = 0; j < TILE; ++j) {
        if (j < static_cast<int>(rows)) {
            if (t == j) {
                double sum = 0.0;
                for (int p = 0; p < j; ++p) sum += s[j][p] * s[j][p];
                s[j][j] = sqrt(s[j][j] - sum);
            }
            __syncthreads();
            if (t > j && t < static_cast<int>(rows)) {
                double sum = 0.0;
                for (int p = 0; p < j; ++p) sum += s[t][p] * s[j][p];
                s[t][j] = (s[t][j] - sum) / s[j][j];
            }
            __syncthreads();
        }
    }
    if (base + t < n)
        for (int col = 0; col <= t && col < static_cast<int>(rows); ++col)
            if (base + col < n) a[(base + t) * n + base + col] = s[t][col];
}

__global__ void solveBelow(double* a, size_t n, size_t diagonal, size_t rows) {
    __shared__ double lkk[TILE][TILE];
    __shared__ double aik[TILE][TILE];
    const int r = threadIdx.x;
    const size_t kbase = diagonal * TILE;
    const size_t ibase = (blockIdx.x + diagonal + 1) * TILE;
    for (int c = 0; c < TILE; ++c) {
        lkk[r][c] = (kbase + r < n && kbase + c < n) ? a[(kbase + r) * n + kbase + c] : 0.0;
        aik[r][c] = (ibase + r < n && kbase + c < n) ? a[(ibase + r) * n + kbase + c] : 0.0;
    }
    __syncthreads();
    for (int c = 0; c < TILE; ++c) {
        if (ibase + r < n && kbase + c < n) {
                double sum = 0.0;
                // A(i,k) * inv(L(k,k)^T): solve from the left to the right.
                for (int p = 0; p < c; ++p) sum += aik[r][p] * lkk[c][p];
                aik[r][c] = (aik[r][c] - sum) / lkk[c][c];
        }
        __syncthreads();
    }
    if (ibase + r < n)
        for (int c = 0; c < TILE && kbase + c < n; ++c)
            a[(ibase + r) * n + kbase + c] = aik[r][c];
}

__global__ void updateTrailing(double* a, size_t n, size_t k) {
    __shared__ double left[TILE][TILE], right[TILE][TILE];
    const size_t rowTile = blockIdx.y + k + 1;
    const size_t colTile = blockIdx.x + k + 1;
    if (rowTile < colTile) return;
    const int row = threadIdx.y, col = threadIdx.x;
    const size_t rowIndex = rowTile * TILE + row;
    const size_t colIndex = colTile * TILE + col;
    const size_t rightRowIndex = colTile * TILE + row;
    double value = 0.0;
    const size_t kt = k * TILE;
    left[row][col] = (rowIndex < n && kt + col < n) ? a[rowIndex * n + kt + col] : 0.0;
    right[row][col] = (rightRowIndex < n && kt + col < n) ? a[rightRowIndex * n + kt + col] : 0.0;
    __syncthreads();
    for (int p = 0; p < TILE; ++p) value += left[row][p] * right[col][p];
    if (rowIndex < n && colIndex < n) a[rowIndex * n + colIndex] -= value;
}

__global__ void clearUpper(double* a, size_t n) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < n * n && index % n > index / n) a[index] = 0.0;
}

bool cudaOk(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return true;
    fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(status));
    return false;
}

} // namespace

// Tiled, right-looking Cholesky decomposition. All arithmetic in the factorization
// is performed on the GPU; the host vector is updated only after completion.

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    if (n == 0) return true;
    double* deviceA = nullptr;
    const size_t bytes = n * n * sizeof(double);
    if (!cudaOk(cudaMalloc(&deviceA, bytes), "allocation") ||
        !cudaOk(cudaMemcpy(deviceA, A.data(), bytes, cudaMemcpyHostToDevice), "host-to-device copy")) {
        if (deviceA) cudaFree(deviceA);
        return false;
    }
    const size_t tiles = (n + TILE - 1) / TILE;
    for (size_t k = 0; k < tiles; ++k) {
        const size_t extent = std::min<size_t>(TILE, n - k * TILE);
        factorDiagonal<<<1, TILE>>>(deviceA, n, k, extent);
        if (!cudaOk(cudaGetLastError(), "diagonal factorization")) { cudaFree(deviceA); return false; }
        if (k + 1 < tiles) {
            solveBelow<<<tiles - k - 1, TILE>>>(deviceA, n, k, extent);
            if (!cudaOk(cudaGetLastError(), "triangular solve")) { cudaFree(deviceA); return false; }
            dim3 grid(tiles - k - 1, tiles - k - 1);
            updateTrailing<<<grid, dim3(TILE, TILE)>>>(deviceA, n, k);
            if (!cudaOk(cudaGetLastError(), "trailing update")) { cudaFree(deviceA); return false; }
        }
    }
    clearUpper<<<(n * n + 255) / 256, 256>>>(deviceA, n);
    if (!cudaOk(cudaGetLastError(), "upper triangle cleanup") ||
        !cudaOk(cudaDeviceSynchronize(), "decomposition")) { cudaFree(deviceA); return false; }
    const bool copied = cudaOk(cudaMemcpy(A.data(), deviceA, bytes, cudaMemcpyDeviceToHost), "device-to-host copy");
    cudaFree(deviceA);
    return copied;
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
