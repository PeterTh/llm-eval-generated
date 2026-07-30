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
// CUDA error handling
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t _err = (call);                                               \
        if (_err != cudaSuccess) {                                               \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__,              \
                   cudaGetErrorString(_err));                                     \
            return false;                                                        \
        }                                                                        \
    } while (0)

// ---------------------------------------------------------------------------
// Device-side RNG  (Xorshift32 – fast, good quality for this use)
// ---------------------------------------------------------------------------
__device__ __forceinline__ unsigned int xorshift32(unsigned int &state) {
    unsigned int x = state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return state = x;
}

// ---------------------------------------------------------------------------
// Kernel: generate random matrix B  (column-major layout)
// ---------------------------------------------------------------------------
__global__ void generateRandomMatrix(double *B, size_t n, unsigned int seed) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t total = n * n;
    if (idx < total) {
        unsigned int state = seed ^ (static_cast<unsigned int>(idx) * 2654435761u);
        xorshift32(state); // discard initial state
        B[idx] = ((static_cast<double>(xorshift32(state)) /
                   static_cast<double>(0xFFFFFFFFu)) -
                  0.5);
    }
}

// ---------------------------------------------------------------------------
// Kernel: add diagonal dominance  A[i,i] += n
// ---------------------------------------------------------------------------
__global__ void addDiagonal(double *A, size_t n) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        A[idx * n + idx] += static_cast<double>(n);
    }
}

// ---------------------------------------------------------------------------
// Generate symmetric positive-definite matrix on GPU (column-major)
//   A = B * B^T  +  n * I
// Uses cuBLAS GEMM for the matrix multiply.
// ---------------------------------------------------------------------------
bool generateSPDMatrixOnGPU(double *d_A, size_t n) {
    double *d_B = nullptr;
    CUDA_CHECK(cudaMalloc(&d_B, n * n * sizeof(double)));

    cublasHandle_t handle;
    cublasCreate(&handle);

    unsigned int seed = 42;
    const int blockSize = 256;
    const int gridSize = static_cast<int>((n * n + blockSize - 1) / blockSize);

    // 1. Fill B with random values
    generateRandomMatrix<<<gridSize, blockSize>>>(d_B, n, seed);

    // 2. A = B * B^T  (column-major)
    double alpha = 1.0, beta = 0.0;
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                static_cast<int>(n), static_cast<int>(n),
                static_cast<int>(n),
                &alpha, d_B, static_cast<int>(n),
                d_B, static_cast<int>(n),
                &beta, d_A, static_cast<int>(n));

    // 3. Add diagonal dominance
    const int diagBlock = 256;
    const int diagGrid = static_cast<int>((n + diagBlock - 1) / diagBlock);
    addDiagonal<<<diagGrid, diagBlock>>>(d_A, n);

    CUDA_CHECK(cudaDeviceSynchronize());

    cublasDestroy(handle);
    cudaFree(d_B);
    return true;
}

// ---------------------------------------------------------------------------
// Cholesky decomposition on GPU  (column-major, lower-triangle)
//   A = L * L^T   using cuSOLVER potrf
// ---------------------------------------------------------------------------
bool choleskyDecompositionGPU(double *d_A, size_t n) {
    cusolverDnHandle_t handle;
    cusolverDnCreate(&handle);

    // 1. Query workspace size
    int lwork = 0;
    cusolverDnDpotrf_bufferSize(handle,
                                CUBLAS_FILL_MODE_LOWER,
                                static_cast<int>(n),
                                d_A,
                                static_cast<int>(n),
                                &lwork);

    // 2. Allocate workspace and devInfo
    double *d_workspace = nullptr;
    CUDA_CHECK(cudaMalloc(&d_workspace, lwork * sizeof(double)));

    int *d_devInfo = nullptr;
    CUDA_CHECK(cudaMalloc(&d_devInfo, sizeof(int)));

    // 3. Perform Cholesky decomposition
    cusolverStatus_t status = cusolverDnDpotrf(handle,
                                               CUBLAS_FILL_MODE_LOWER,
                                               static_cast<int>(n),
                                               d_A,
                                               static_cast<int>(n),
                                               d_workspace,
                                               lwork,
                                               d_devInfo);

    int info = 0;
    cudaMemcpy(&info, d_devInfo, sizeof(int), cudaMemcpyDeviceToHost);

    bool success = (status == CUSOLVER_STATUS_SUCCESS && info == 0);
    if (!success) {
        printf("Error: Cholesky decomposition failed (info=%d, status=%d)\n",
               info, status);
    }

    cusolverDnDestroy(handle);
    cudaFree(d_workspace);
    cudaFree(d_devInfo);
    CUDA_CHECK(cudaDeviceSynchronize());
    return success;
}

// ---------------------------------------------------------------------------
// Usage
// ---------------------------------------------------------------------------
void printUsage(const char *progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char **argv) {
    size_t n = 512;
    bool validate = false;
    bool printResults = false;

    // ---- parse arguments --------------------------------------------------
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

    // ---- allocate GPU memory (column-major) --------------------------------
    double *d_A = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, n * n * sizeof(double)));

    // ---- generate SPD matrix on GPU ----------------------------------------
    printf("Generating positive definite matrix...\n");
    bool ok = generateSPDMatrixOnGPU(d_A, n);
    if (!ok) {
        printf("Matrix generation failed\n");
        cudaFree(d_A);
        return 1;
    }

    // ---- save original for validation (column-major -> row-major) ----------
    std::vector<double> A_orig(n * n);
    if (validate) {
        std::vector<double> tmp(n * n);
        cudaMemcpy(tmp.data(), d_A, n * n * sizeof(double),
                   cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j)
                A_orig[i * n + j] = tmp[j * n + i];
    }

    // ---- Cholesky decomposition (timed) ------------------------------------
    printf("Computing Cholesky decomposition...\n");
    auto start = std::chrono::high_resolution_clock::now();

    ok = choleskyDecompositionGPU(d_A, n);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (!ok) {
        printf("Cholesky decomposition failed\n");
        cudaFree(d_A);
        return 1;
    }

    printf("Computation time: %ld ms\n", duration.count());

    // ---- performance -------------------------------------------------------
    double ops = static_cast<double>(n) * n * n / 3.0;
    double gflops =
        ops / (static_cast<double>(duration.count()) / 1000.0) / 1e9;
    printf("Performance: %.3f GFLOPS\n", gflops);

    // ---- copy result back (column-major -> row-major) ----------------------
    std::vector<double> A(n * n);
    cudaMemcpy(A.data(), d_A, n * n * sizeof(double), cudaMemcpyDeviceToHost);
    // transpose in-place (column-major -> row-major)
    for (size_t i = 0; i < n; ++i)
        for (size_t j = i + 1; j < n; ++j)
            std::swap(A[i * n + j], A[j * n + i]);
    // potrf only writes the lower triangle; zero out the upper triangle
    for (size_t i = 0; i < n; ++i)
        for (size_t j = i + 1; j < n; ++j)
            A[i * n + j] = 0.0;

    cudaFree(d_A);

    // ---- print results for external validation -----------------------------
    if (printResults) {
        print_results(A, "CholeskyL");
    }

    // ---- validation --------------------------------------------------------
    if (validate) {
        printf("Validating result...\n");

        // Reconstruct A from L:  L * L^T
        std::vector<double> reconstructed(n * n, 0.0);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                double sum = 0.0;
                for (size_t k = 0; k < n; ++k) {
                    sum += A[i * n + k] * A[j * n + k];
                }
                reconstructed[i * n + j] = sum;
            }
        }

        double maxError = 0.0;
        double relError = 0.0;
        for (size_t i = 0; i < n * n; ++i) {
            double error = fabs(reconstructed[i] - A_orig[i]);
            maxError = std::max(maxError, error);
            double rel = error / (fabs(A_orig[i]) + 1e-10);
            relError = std::max(relError, rel);
        }

        printf("Max absolute error: %.10e\n", maxError);
        printf("Max relative error: %.10e\n", relError);

        if (relError > 1e-6) {
            printf("Validation failed: relative error too large\n");
            printf("Validation: FAILED\n");
            return 1;
        }

        printf("Validation: PASSED\n");
    }

    return 0;
}
