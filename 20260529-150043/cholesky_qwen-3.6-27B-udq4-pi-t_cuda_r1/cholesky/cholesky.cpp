#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <cusolverDn.h>

#include "../common/results_output.hpp"

// ---------------------------------------------------------------------------
// CUDA kernel: matrix transpose (row-major <-> column-major)
// Each thread handles one element: dst[col * n + row] = src[row * n + col]
// ---------------------------------------------------------------------------
__global__ void transposeKernel(const double* __restrict__ src,
                                double* __restrict__ dst, const size_t n) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = n * n;
    if (idx < total) {
        const size_t row = idx / n;
        const size_t col = idx % n;
        dst[col * n + row] = src[idx];
    }
}

// Launch transpose kernel for an n×n matrix
static void gpuTranspose(const double* src, double* dst, size_t n) {
    const int blockSize = 256;
    const size_t total = n * n;
    const int numBlocks = static_cast<int>((total + blockSize - 1) / blockSize);
    transposeKernel<<<numBlocks, blockSize>>>(src, dst, n);
}

// ---------------------------------------------------------------------------
// CUDA kernel: zero out upper triangular part of an n×n matrix
// ---------------------------------------------------------------------------
__global__ void zeroUpperKernel(double* A, const size_t n) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = n * n;
    if (idx < total) {
        const size_t row = idx / n;
        const size_t col = idx % n;
        if (col > row) {
            A[idx] = 0.0;
        }
    }
}

// Launch zero-upper kernel for an n×n matrix
static void gpuZeroUpper(double* A, size_t n) {
    const int blockSize = 256;
    const size_t total = n * n;
    const int numBlocks = static_cast<int>((total + blockSize - 1) / blockSize);
    zeroUpperKernel<<<numBlocks, blockSize>>>(A, n);
}

// ---------------------------------------------------------------------------
// CUDA Cholesky decomposition using cuSOLVER cusolverDnDpotrf
// Decomposes positive definite matrix A (row-major) into L * L^T.
// L is stored in the lower triangular part of A; upper is zeroed out.
//
// Layout notes:
// - Input: symmetric matrix in row-major. Since A is symmetric, the raw
//   byte layout is identical for row-major and column-major.
// - cuSOLVER reads as column-major and writes L in column-major format.
// - Output L is NOT symmetric, so we must transpose back to row-major.
// ---------------------------------------------------------------------------
bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    const size_t nnz = n * n;
    const size_t bytes = nnz * sizeof(double);

    // Allocate device memory
    double *d_A = nullptr, *d_tmp = nullptr;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_tmp, bytes);

    // Copy host matrix to device
    // For symmetric matrices, row-major == column-major byte layout.
    cudaMemcpy(d_A, A.data(), bytes, cudaMemcpyHostToDevice);

    // cuSOLVER Cholesky factorization (column-major, lower triangular)
    cusolverDnHandle_t handle;
    cusolverDnCreate(&handle);

    // Query workspace size
    int lwork = 0;
    cusolverDnDpotrf_bufferSize(handle, CUBLAS_FILL_MODE_LOWER,
                                static_cast<int>(n), d_A,
                                static_cast<int>(n), &lwork);

    // Allocate workspace
    double *d_workspace = nullptr;
    if (lwork > 0) {
        cudaMalloc(&d_workspace, lwork * sizeof(double));
    }

    // Device info pointer
    int *d_devInfoPtr = nullptr;
    cudaMalloc(&d_devInfoPtr, sizeof(int));
    int h_devInfo = 0;
    cudaMemcpy(d_devInfoPtr, &h_devInfo, sizeof(int), cudaMemcpyHostToDevice);

    // Perform factorization
    cusolverDnDpotrf(handle, CUBLAS_FILL_MODE_LOWER,
                     static_cast<int>(n), d_A,
                     static_cast<int>(n), d_workspace, lwork, d_devInfoPtr);

    // Synchronize and retrieve device info
    cudaDeviceSynchronize();
    cudaMemcpy(&h_devInfo, d_devInfoPtr, sizeof(int), cudaMemcpyDeviceToHost);

    // Cleanup cuSOLVER
    cusolverDnDestroy(handle);

    // Check factorization status
    if (h_devInfo > 0) {
        printf("Error: Matrix is not positive definite at diagonal element %d\n",
               h_devInfo - 1);
        cudaFree(d_A);
        cudaFree(d_tmp);
        cudaFree(d_workspace);
        cudaFree(d_devInfoPtr);
        return false;
    } else if (h_devInfo < 0) {
        printf("Error: cuSOLVER potrf returned illegal parameter %d\n", h_devInfo);
        cudaFree(d_A);
        cudaFree(d_tmp);
        cudaFree(d_workspace);
        cudaFree(d_devInfoPtr);
        return false;
    }

    // Transpose from column-major to row-major
    gpuTranspose(d_A, d_tmp, n);

    // Zero out upper triangular part on GPU
    gpuZeroUpper(d_tmp, n);

    // Synchronize and copy result back to host
    cudaDeviceSynchronize();
    cudaMemcpy(A.data(), d_tmp, bytes, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_tmp);
    cudaFree(d_workspace);
    cudaFree(d_devInfoPtr);

    return true;
}

// ---------------------------------------------------------------------------
// Generate a symmetric positive definite matrix
// ---------------------------------------------------------------------------
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    // Method: Create A = B * B^T where B is random
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

// ---------------------------------------------------------------------------
// Validate by computing L * L^T and comparing with original matrix
// ---------------------------------------------------------------------------
bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig,
                      const size_t n) {
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

    // Select GPU device
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        fprintf(stderr, "Error: No CUDA devices found\n");
        return 1;
    }

    // Use the first GPU
    cudaSetDevice(0);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Cholesky Decomposition Benchmark (CUDA)\n");
    printf("GPU: %s (Compute %d.%d)\n", prop.name, prop.major, prop.minor);
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

    // Perform Cholesky decomposition on GPU
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
