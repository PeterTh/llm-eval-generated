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

__global__ void potrfTile(double* a, size_t n, size_t k, int* failure) {
    __shared__ double t[TILE][TILE];
    const int x = threadIdx.x, y = threadIdx.y;
    const size_t r = k * TILE + y, c = k * TILE + x;
    t[y][x] = (r < n && c < n) ? a[r * n + c] : 0.0;
    __syncthreads();
    if (x == 0 && y == 0) {
        const int width = (int)min((size_t)TILE, n - k * TILE);
        for (int i = 0; i < width; ++i) {
            double s = t[i][i];
            for (int q = 0; q < i; ++q) s -= t[i][q] * t[i][q];
            if (!(s > 0.0)) { *failure = 1; return; }
            t[i][i] = sqrt(s);
            for (int j = i + 1; j < width; ++j) {
                double v = t[j][i];
                for (int q = 0; q < i; ++q) v -= t[j][q] * t[i][q];
                t[j][i] = v / t[i][i];
            }
        }
    }
    __syncthreads();
    if (r < n && c < n) a[r * n + c] = (y >= x) ? t[y][x] : 0.0;
}

__global__ void panelSolve(double* a, size_t n, size_t k) {
    const size_t row = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t base = k * TILE;
    if (row >= n - base) return;
    const size_t r = base + row;
    const int width = (int)min((size_t)TILE, n - base);
    for (int j = 0; j < width; ++j) {
        double v = a[r * n + base + j];
        for (int q = 0; q < j; ++q) v -= a[r * n + base + q] * a[(base + j) * n + base + q];
        a[r * n + base + j] = v / a[(base + j) * n + base + j];
    }
}

__global__ void trailingUpdate(double* a, size_t n, size_t base, size_t width) {
    const size_t i = base + (size_t)blockIdx.y * TILE + threadIdx.y;
    const size_t j = base + (size_t)blockIdx.x * TILE + threadIdx.x;
    if (i >= n || j >= n || i < j) return;
    double v = a[i * n + j];
    for (size_t q = 0; q < width; ++q) v -= a[i * n + base - width + q] * a[j * n + base - width + q];
    a[i * n + j] = v;
}
}

// GPU tiled Cholesky decomposition, A is row-major and overwritten with L.

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    if (n == 0) return true;
    double* d = nullptr; int* err = nullptr, hErr = 0;
    if (cudaMalloc(&d, A.size() * sizeof(double)) != cudaSuccess ||
        cudaMalloc(&err, sizeof(int)) != cudaSuccess) return false;
    cudaMemcpy(d, A.data(), A.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(err, 0, sizeof(int));
    const size_t tiles = (n + TILE - 1) / TILE;
    for (size_t k = 0; k < tiles; ++k) {
        potrfTile<<<1, dim3(TILE, TILE)>>>(d, n, k, err);
        const size_t base = (k + 1) * TILE;
        if (base < n) {
            panelSolve<<<(n - base + 255) / 256, 256>>>(d, n, k);
            cudaDeviceSynchronize();
            const size_t w = min((size_t)TILE, n - k * TILE);
            dim3 block(TILE, TILE), grid((n - base + TILE - 1) / TILE, (n - base + TILE - 1) / TILE);
            trailingUpdate<<<grid, block>>>(d, n, base, w);
        }
    }
    bool ok = cudaGetLastError() == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess;
    cudaMemcpy(&hErr, err, sizeof(int), cudaMemcpyDeviceToHost);
    if (ok && hErr == 0) cudaMemcpy(A.data(), d, A.size() * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(err); cudaFree(d);
    if (!ok || hErr) { printf("Error: Matrix is not positive definite\n"); return false; }
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
