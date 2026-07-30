#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include "../common/results_output.hpp"

// ---------------------------------------------------------------------------
// CUDA kernels
// ---------------------------------------------------------------------------

// Simple LCG-based random number generator kernel (column-major output)
__global__ void generateRandomKernel(double* B, size_t n, unsigned int seed) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = n * n;

    if (idx < total) {
        unsigned int state = seed + static_cast<unsigned int>(idx);
        // LCG:  state = a * state + c  (Numerical Recipes constants)
        state = state * 1103515245u + 12345u;
        double val = ((state >> 16) & 0x7FFF) / static_cast<double>(0x7FFF) - 0.5;
        B[idx] = val;
    }
}

// Add a value to every diagonal element (column-major layout)
__global__ void addDiagonalKernel(double* A, size_t n, double val) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        A[idx * n + idx] += val;
    }
}

// Transpose a square matrix: src is column-major, dst is row-major
__global__ void transposeKernel(const double* src, double* dst, size_t n) {
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        // element (row,col) in column-major is at src[col*n + row]
        // element (row,col) in row-major  is at dst[row*n + col]
        dst[row * n + col] = src[col * n + row];
    }
}

// Zero out the upper triangle in row-major storage
__global__ void zeroUpperTriangleKernel(double* A, size_t n) {
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n && col > row) {
        A[row * n + col] = 0.0;
    }
}

// ---------------------------------------------------------------------------
// Helper: error checking
// ---------------------------------------------------------------------------
static void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s - %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

static void checkCublas(cublasStatus_t status, const char* msg) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS error: %s (status=%d)\n", msg, static_cast<int>(status));
        exit(1);
    }
}

static void checkCusolver(cusolverStatus_t status, const char* msg) {
    if (status != CUSOLVER_STATUS_SUCCESS) {
        fprintf(stderr, "cuSOLVER error: %s (status=%d)\n", msg, static_cast<int>(status));
        exit(1);
    }
}

// ---------------------------------------------------------------------------
// Generate a symmetric positive definite matrix on GPU, copy to host
// ---------------------------------------------------------------------------
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    size_t bytes = n * n * sizeof(double);

    cublasHandle_t handle;
    checkCublas(cublasCreate(&handle), "cublasCreate");

    double *d_B = nullptr, *d_A = nullptr, *d_AT = nullptr;
    checkCuda(cudaMalloc(&d_B, bytes), "cudaMalloc d_B");
    checkCuda(cudaMalloc(&d_A, bytes), "cudaMalloc d_A");
    checkCuda(cudaMalloc(&d_AT, bytes), "cudaMalloc d_AT");

    // 1) Generate random matrix B on GPU (column-major)
    int blockSize = 256;
    int gridSize = static_cast<int>((n * n + blockSize - 1) / blockSize);
    generateRandomKernel<<<gridSize, blockSize>>>(d_B, n, 42);
    checkCuda(cudaGetLastError(), "generateRandomKernel launch");

    // 2) Compute A = B * B^T  (column-major, via cuBLAS dgemm)
    double alpha = 1.0, beta = 0.0;
    checkCublas(cublasDgemm(handle,
                            CUBLAS_OP_N, CUBLAS_OP_T,
                            static_cast<int>(n), static_cast<int>(n), static_cast<int>(n),
                            &alpha,
                            d_B, static_cast<int>(n),
                            d_B, static_cast<int>(n),
                            &beta,
                            d_A, static_cast<int>(n)),
                "cublasDgemm");

    // 3) Add diagonal dominance to ensure positive definiteness
    int diagBlockSize = 256;
    int diagGridSize = static_cast<int>((n + diagBlockSize - 1) / diagBlockSize);
    addDiagonalKernel<<<diagGridSize, diagBlockSize>>>(d_A, n, static_cast<double>(n));
    checkCuda(cudaGetLastError(), "addDiagonalKernel launch");

    // 4) Transpose from column-major (GPU) to row-major (host)
    dim3 tBlock(16, 16);
    dim3 tGrid((n + 15) / 16, (n + 15) / 16);
    transposeKernel<<<tGrid, tBlock>>>(d_A, d_AT, n);
    checkCuda(cudaGetLastError(), "transposeKernel launch");

    // 5) Copy to host
    checkCuda(cudaMemcpy(A.data(), d_AT, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy d_A->host");

    // Cleanup
    cudaFree(d_B);
    cudaFree(d_A);
    cudaFree(d_AT);
    cublasDestroy(handle);
}

// ---------------------------------------------------------------------------
// Cholesky decomposition on GPU: A = L * L^T
// A is row-major on host; result L is row-major on host.
// ---------------------------------------------------------------------------
bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    size_t bytes = n * n * sizeof(double);

    // Create cuSOLVER handle
    cusolverDnHandle_t solverHandle;
    checkCusolver(cusolverDnCreate(&solverHandle), "cusolverDnCreate");

    double *d_A = nullptr, *d_L = nullptr, *d_workspace = nullptr;
    checkCuda(cudaMalloc(&d_A, bytes), "cudaMalloc d_A");
    checkCuda(cudaMalloc(&d_L, bytes), "cudaMalloc d_L");

    // 1) Copy host A (row-major) to GPU
    checkCuda(cudaMemcpy(d_A, A.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy host->d_A");

    // 2) Transpose row-major -> column-major for cuSOLVER
    dim3 tBlock(16, 16);
    dim3 tGrid((n + 15) / 16, (n + 15) / 16);
    transposeKernel<<<tGrid, tBlock>>>(d_A, d_L, n);
    checkCuda(cudaGetLastError(), "transposeKernel launch");

    // 3) Query workspace size for Cholesky
    int lwork = 0;
    checkCusolver(cusolverDnDpotrf_bufferSize(solverHandle,
                                               CUBLAS_FILL_MODE_LOWER,
                                               static_cast<int>(n),
                                               d_L, static_cast<int>(n),
                                               &lwork),
                  "cusolverDnDpotrf_bufferSize");

    // 4) Allocate workspace
    checkCuda(cudaMalloc(&d_workspace, lwork * sizeof(double)), "cudaMalloc workspace");

    // 5) Allocate device info
    int *d_devInfo = nullptr;
    checkCuda(cudaMalloc(&d_devInfo, sizeof(int)), "cudaMalloc d_devInfo");

    // 6) Cholesky factorization in-place on d_L (column-major, lower triangle)
    checkCusolver(cusolverDnDpotrf(solverHandle,
                                    CUBLAS_FILL_MODE_LOWER,
                                    static_cast<int>(n),
                                    d_L, static_cast<int>(n),
                                    d_workspace, lwork,
                                    d_devInfo),
                  "cusolverDnDpotrf");

    // 7) Check result
    int info = 0;
    checkCuda(cudaMemcpy(&info, d_devInfo, sizeof(int), cudaMemcpyDeviceToHost),
              "cudaMemcpy devInfo->host");

    if (info != 0) {
        printf("Error: Matrix is not positive definite (potrf info=%d)\n", info);
        cudaFree(d_A);
        cudaFree(d_L);
        cudaFree(d_workspace);
        cudaFree(d_devInfo);
        cusolverDnDestroy(solverHandle);
        return false;
    }

    // 8) Transpose result back: column-major L -> row-major L (into d_A)
    transposeKernel<<<tGrid, tBlock>>>(d_L, d_A, n);
    checkCuda(cudaGetLastError(), "transposeKernel (L back) launch");

    // 9) Zero out upper triangle in row-major result
    zeroUpperTriangleKernel<<<tGrid, tBlock>>>(d_A, n);
    checkCuda(cudaGetLastError(), "zeroUpperTriangleKernel launch");

    // 10) Copy result back to host
    checkCuda(cudaMemcpy(A.data(), d_A, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy d_A->host");

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_L);
    cudaFree(d_workspace);
    cudaFree(d_devInfo);
    cusolverDnDestroy(solverHandle);

    return true;
}

// ---------------------------------------------------------------------------
// Validation (CPU): compute L * L^T and compare with original matrix
// ---------------------------------------------------------------------------
bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
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

    if (relError > 1e-6) {
        printf("Validation failed: relative error too large\n");
        return false;
    }

    return true;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
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
            n = static_cast<size_t>(atoi(argv[++i]));
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

    // Calculate GFLOPS (approximately n^3/3 operations for Cholesky)
    double ops = static_cast<double>(n) * n * n / 3.0;
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
