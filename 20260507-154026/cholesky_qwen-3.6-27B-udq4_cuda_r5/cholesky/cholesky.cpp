#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "../common/results_output.hpp"

#define BLOCK_SIZE 64
#define TPB 256

#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess) {                                                \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__,              \
                   cudaGetErrorString(err));                                      \
            exit(1);                                                             \
        }                                                                        \
    } while (0)

#define CUBLAS_CHECK(call)                                                       \
    do {                                                                         \
        cublasStatus_t status = call;                                            \
        if (status != CUBLAS_STATUS_SUCCESS) {                                   \
            printf("cuBLAS error at %s:%d\n", __FILE__, __LINE__);               \
            exit(1);                                                             \
        }                                                                        \
    } while (0)

// Compute diagonal element of column j (column-major storage)
__global__ void cholesky_diagonal_kernel(double* A, int n, int j, int k_start) {
    if (threadIdx.x == 0) {
        double sum = 0.0;
        for (int p = k_start; p < j; ++p) {
            double ljp = A[p * n + j];
            sum += ljp * ljp;
        }
        double val = A[j * n + j] - sum;
        if (val > 0.0) {
            A[j * n + j] = sqrt(val);
        }
    }
}

// Compute off-diagonal elements of column j (column-major storage)
__global__ void cholesky_offdiagonal_kernel(double* A, int n, int j, int k_start) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > j && i < n) {
        double sum = 0.0;
        for (int p = k_start; p < j; ++p) {
            sum += A[p * n + i] * A[p * n + j];
        }
        A[j * n + i] = (A[j * n + i] - sum) / A[j * n + j];
    }
}

// Zero upper triangle (column-major)
__global__ void zero_upper_triangular(double* A, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * n;
    if (idx < total) {
        int col = idx / n;
        int row = idx % n;
        if (col > row) {
            A[idx] = 0.0;
        }
    }
}

// Blocked Cholesky decomposition on GPU (column-major storage)
bool choleskyDecomposition(double* d_A, const size_t n, cublasHandle_t handle) {
    int block_size = BLOCK_SIZE;
    int N = static_cast<int>(n);

    for (int k = 0; k < N; k += block_size) {
        int kb = std::min(k + block_size, N) - k;

        // 1. Panel factorization: factor columns k to k+kb-1
        for (int j = k; j < k + kb; ++j) {
            // Compute diagonal element first
            cholesky_diagonal_kernel<<<1, 1>>>(d_A, N, j, k);

            // Compute off-diagonal elements in parallel across rows
            int blocks = (N + TPB - 1) / TPB;
            cholesky_offdiagonal_kernel<<<blocks, TPB>>>(d_A, N, j, k);
        }

        // 2. Update trailing submatrix: W = W - V * V^T via cuBLAS dsyrk
        if (k + kb < N) {
            int m = N - (k + kb);
            int kdim = kb;

            double alpha = -1.0;
            double beta = 1.0;

            CUBLAS_CHECK(cublasDsyrk(handle, CUBLAS_FILL_MODE_LOWER,
                                     CUBLAS_OP_N,
                                     m, kdim,
                                     &alpha,
                                     d_A + k * N + (k + kb), N,
                                     &beta,
                                     d_A + (k + kb) * N + (k + kb), N));
        }
    }

    // Zero upper triangle
    int blocks = (N * N + TPB - 1) / TPB;
    zero_upper_triangular<<<blocks, TPB>>>(d_A, N);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return true;
}

// Generate a symmetric positive definite matrix
// Uses the same rand_r-based PRNG as the original for reproducibility
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;

    // Generate random matrix B (row-major)
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

// Validate Cholesky decomposition on GPU
bool validateCholesky(double* d_L, double* d_A_orig, const size_t n,
                      cublasHandle_t handle) {
    double* d_reconstructed = nullptr;
    CUDA_CHECK(cudaMalloc(&d_reconstructed, n * n * sizeof(double)));

    // Compute L * L^T on GPU
    double alpha = 1.0;
    double beta = 0.0;
    CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                              n, n, n,
                              &alpha, d_L, n, d_L, n,
                              &beta, d_reconstructed, n));

    // Copy results to host (column-major) then transpose to row-major
    std::vector<double> reconstructed_col(n * n);
    std::vector<double> a_orig_col(n * n);

    CUDA_CHECK(cudaMemcpy(reconstructed_col.data(), d_reconstructed, n * n * sizeof(double),
                           cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(a_orig_col.data(), d_A_orig, n * n * sizeof(double),
                           cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_reconstructed));

    std::vector<double> reconstructed(n * n);
    std::vector<double> a_orig(n * n);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            reconstructed[i * n + j] = reconstructed_col[j * n + i];
            a_orig[i * n + j] = a_orig_col[j * n + i];
        }
    }

    // Compare on host
    double maxError = 0.0;
    double relError = 0.0;

    for (size_t i = 0; i < n * n; ++i) {
        const double error = fabs(reconstructed[i] - a_orig[i]);
        maxError = std::max(maxError, error);

        const double rel = error / (fabs(a_orig[i]) + 1e-10);
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

    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Allocate matrix on host
    std::vector<double> A(n * n);
    std::vector<double> A_orig;

    // Generate positive definite matrix on host (for exact reproducibility)
    printf("Generating positive definite matrix...\n");
    generatePositiveDefiniteMatrix(A, n);

    if (validate) {
        A_orig = A; // Save original for validation
    }

    // Allocate GPU memory
    double* d_A = nullptr;
    double* d_A_orig = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, n * n * sizeof(double)));

    // Transpose A from row-major to column-major on host
    std::vector<double> A_col(n * n);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            A_col[j * n + i] = A[i * n + j];
        }
    }
    CUDA_CHECK(cudaMemcpy(d_A, A_col.data(), n * n * sizeof(double),
                           cudaMemcpyHostToDevice));

    // Save original for validation
    if (validate) {
        CUDA_CHECK(cudaMalloc(&d_A_orig, n * n * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_A_orig, A_col.data(), n * n * sizeof(double),
                               cudaMemcpyHostToDevice));
    }

    // Perform Cholesky decomposition
    printf("Computing Cholesky decomposition...\n");
    auto start = std::chrono::high_resolution_clock::now();

    bool success = choleskyDecomposition(d_A, n, handle);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (!success) {
        printf("Cholesky decomposition failed\n");
        CUDA_CHECK(cudaFree(d_A));
        if (validate) CUDA_CHECK(cudaFree(d_A_orig));
        cublasDestroy(handle);
        return 1;
    }

    printf("Computation time: %ld ms\n", duration.count());

    // Calculate GFLOPS (approximately n^3/3 operations for Cholesky)
    double ops = static_cast<double>(n) * n * n / 3.0;
    double gflops = ops / (duration.count() / 1000.0) / 1e9;
    printf("Performance: %.3f GFLOPS\n", gflops);

    // Copy result to host (column-major -> row-major)
    std::vector<double> L_col(n * n);
    CUDA_CHECK(cudaMemcpy(L_col.data(), d_A, n * n * sizeof(double),
                           cudaMemcpyDeviceToHost));
    std::vector<double> L(n * n);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            L[i * n + j] = L_col[j * n + i];
        }
    }

    // Print results for external validation
    if (printResults) {
        print_results(L, "CholeskyL");
    }

    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateCholesky(d_A, d_A_orig, n, handle);

        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            CUDA_CHECK(cudaFree(d_A));
            CUDA_CHECK(cudaFree(d_A_orig));
            cublasDestroy(handle);
            return 1;
        }
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_A));
    if (validate) CUDA_CHECK(cudaFree(d_A_orig));
    cublasDestroy(handle);

    return 0;
}
