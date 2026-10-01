#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// Blocked right-looking Cholesky. Each panel update is parallelized over the GPU.
constexpr int CHOL_BLOCK = 32;

__global__ void factorDiagonal(double* a, size_t n, size_t base, int width, int* failure) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    for (int j = 0; j < width; ++j) {
        double d = a[(base + j) * n + base + j];
        for (int k = 0; k < j; ++k) {
            const double x = a[(base + j) * n + base + k];
            d -= x * x;
        }
        if (d <= 0.0) { *failure = static_cast<int>(base + j); return; }
        a[(base + j) * n + base + j] = sqrt(d);
        for (int i = j + 1; i < width; ++i) {
            double x = a[(base + i) * n + base + j];
            for (int k = 0; k < j; ++k)
                x -= a[(base + i) * n + base + k] * a[(base + j) * n + base + k];
            a[(base + i) * n + base + j] = x / a[(base + j) * n + base + j];
        }
    }
}

__global__ void solvePanel(double* a, size_t n, size_t base, size_t end) {
    size_t i = end + blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    for (size_t c = 0; base + c < end; ++c) {
        double x = a[i * n + base + c];
        for (size_t k = 0; k < c; ++k)
            x -= a[i * n + base + k] * a[(base + c) * n + base + k];
        a[i * n + base + c] = x / a[(base + c) * n + base + c];
    }
}

__global__ void updateTrailing(double* a, size_t n, size_t end, size_t width) {
    size_t i = end + blockIdx.y * blockDim.y + threadIdx.y;
    size_t j = end + blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || j >= n || i < j) return;
    double x = a[i * n + j];
    for (size_t k = 0; k < width; ++k) x -= a[i * n + end - width + k] * a[j * n + end - width + k];
    a[i * n + j] = x;
}

__global__ void clearUpper(double* a, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n * n && idx / n < idx % n) a[idx] = 0.0;
}

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    if (n == 0) return true;
    double* deviceA = nullptr;
    int* deviceFailure = nullptr;
    int failure = -1;
    auto check = [](cudaError_t e) { return e == cudaSuccess; };
    if (!check(cudaMalloc(&deviceA, n * n * sizeof(double))) ||
        !check(cudaMalloc(&deviceFailure, sizeof(int))) ||
        !check(cudaMemcpy(deviceA, A.data(), n * n * sizeof(double), cudaMemcpyHostToDevice)) ||
        !check(cudaMemcpy(deviceFailure, &failure, sizeof(int), cudaMemcpyHostToDevice))) {
        fprintf(stderr, "CUDA allocation or transfer failed: %s\n", cudaGetErrorString(cudaGetLastError()));
        if (deviceA) cudaFree(deviceA);
        if (deviceFailure) cudaFree(deviceFailure);
        return false;
    }
    bool ok = true;
    for (size_t base = 0; base < n && ok; base += CHOL_BLOCK) {
        const int width = static_cast<int>(std::min<size_t>(CHOL_BLOCK, n - base));
        factorDiagonal<<<1, 1>>>(deviceA, n, base, width, deviceFailure);
        size_t end = base + width;
        if (end < n) {
            dim3 block(256);
            solvePanel<<<static_cast<unsigned>((n - end + 255) / 256), block>>>(deviceA, n, base, end);
            dim3 updateBlock(16, 16);
            dim3 updateGrid(static_cast<unsigned>((n - end + 15) / 16), static_cast<unsigned>((n - end + 15) / 16));
            updateTrailing<<<updateGrid, updateBlock>>>(deviceA, n, end, width);
        }
        if (!check(cudaGetLastError()) || !check(cudaMemcpy(&failure, deviceFailure, sizeof(int), cudaMemcpyDeviceToHost))) ok = false;
        else if (failure >= 0) {
            printf("Error: Matrix is not positive definite at diagonal element %d\n", failure);
            ok = false;
        }
    }
    if (ok) {
        clearUpper<<<static_cast<unsigned>((n * n + 255) / 256), 256>>>(deviceA, n);
        ok = check(cudaGetLastError()) && check(cudaMemcpy(A.data(), deviceA, n * n * sizeof(double), cudaMemcpyDeviceToHost));
    }
    cudaFree(deviceA);
    cudaFree(deviceFailure);
    if (!ok && failure < 0) fprintf(stderr, "CUDA Cholesky operation failed\n");
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
