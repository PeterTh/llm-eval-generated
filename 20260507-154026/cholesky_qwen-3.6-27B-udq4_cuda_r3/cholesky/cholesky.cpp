#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <math.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess) {                                                \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__,             \
                   cudaGetErrorString(err));                                     \
            return false;                                                        \
        }                                                                        \
    } while (0)

#define CUDA_CHECK_VOID(call)                                                    \
    do {                                                                         \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess) {                                                \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__,             \
                   cudaGetErrorString(err));                                     \
            exit(1);                                                             \
        }                                                                        \
    } while (0)

// ---------------------------------------------------------------------------
// CUDA kernels
// ---------------------------------------------------------------------------

// Kernel: compute A = B * B^T  (each thread handles one element)
__global__ void matmulKernel(const double* B, double* A, const size_t n) {
    size_t i = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    size_t j = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n && j < n) {
        double sum = 0.0;
        for (size_t k = 0; k < n; ++k) {
            sum += B[i * n + k] * B[j * n + k];
        }
        A[i * n + j] = sum;
    }
}

// Kernel: add val to diagonal of A
__global__ void addDiagonalKernel(double* A, const size_t n, double val) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        A[i * n + i] += val;
    }
}

// Kernel phase 1: compute dot products for column j and store in sums buffer.
// For the diagonal (tid == j), store sqrt(A[j*n+j] - sum) directly in sums[j].
__global__ void choleskyDotKernel(const double* __restrict__ A, double* __restrict__ sums,
                                   const size_t n, const size_t j) {
    size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= n || tid < j) return;

    double sum = 0.0;
    for (size_t k = 0; k < j; ++k) {
        sum += A[tid * n + k] * A[j * n + k];
    }

    if (tid == j) {
        double val = A[j * n + j] - sum;
        sums[tid] = (val > 0.0) ? sqrt(val) : -1.0;
    } else {
        sums[tid] = sum;
    }
}

// Kernel phase 2: update column j using precomputed sums and diagonal
__global__ void choleskyUpdateKernel(double* __restrict__ A, const double* __restrict__ sums,
                                      const size_t n, const size_t j, double diag) {
    size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= n || tid < j) return;

    if (tid == j) {
        A[j * n + j] = diag;
    } else {
        A[tid * n + j] = (A[tid * n + j] - sums[tid]) / diag;
    }
}

// Kernel: zero out upper triangle row j
__global__ void zeroUpperKernel(double* A, const size_t n, const size_t j) {
    size_t col = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (col > j && col < n) {
        A[j * n + col] = 0.0;
    }
}

// Kernel: compute L * L^T for validation
__global__ void validateKernel(const double* __restrict__ L, double* __restrict__ R,
                               const size_t n) {
    size_t i = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    size_t j = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n && j < n) {
        double sum = 0.0;
        for (size_t k = 0; k < n; ++k) {
            sum += L[i * n + k] * L[j * n + k];
        }
        R[i * n + j] = sum;
    }
}

// ---------------------------------------------------------------------------
// Host functions
// ---------------------------------------------------------------------------

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    double* d_A = nullptr;
    double* d_sums = nullptr;
    size_t nnz = n * n;

    CUDA_CHECK(cudaMalloc(&d_A, nnz * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), nnz * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_sums, n * sizeof(double)));

    const int blockSize = 256;
    const int gridSize  = (n + blockSize - 1) / blockSize;

    // Cholesky: column-by-column
    for (size_t j = 0; j < n; ++j) {
        // Phase 1: compute dot products and diagonal sqrt on GPU
        choleskyDotKernel<<<gridSize, blockSize>>>(d_A, d_sums, n, j);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Read back the diagonal element (just one double)
        double diag;
        CUDA_CHECK(cudaMemcpy(&diag, d_sums + j, sizeof(double), cudaMemcpyDeviceToHost));
        if (diag < 0.0) {
            printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
            cudaFree(d_A);
            cudaFree(d_sums);
            return false;
        }

        // Phase 2: update column j on GPU
        choleskyUpdateKernel<<<gridSize, blockSize>>>(d_A, d_sums, n, j, diag);
        CUDA_CHECK(cudaGetLastError());

        // Zero out upper triangle row j
        if (j + 1 < n) {
            zeroUpperKernel<<<gridSize, blockSize>>>(d_A, n, j);
        }
    }

    // Synchronize and copy result back
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(A.data(), d_A, nnz * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_sums));

    return true;
}

// Generate a symmetric positive definite matrix on GPU
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    size_t nnz = n * n;

    // Generate random matrix B on host (preserves original rand_r semantics)
    std::vector<double> B(nnz);
    unsigned int seed = 42;
    for (size_t i = 0; i < nnz; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

    double* d_A = nullptr;
    double* d_B = nullptr;

    CUDA_CHECK_VOID(cudaMalloc(&d_A, nnz * sizeof(double)));
    CUDA_CHECK_VOID(cudaMalloc(&d_B, nnz * sizeof(double)));
    CUDA_CHECK_VOID(cudaMemcpy(d_B, B.data(), nnz * sizeof(double), cudaMemcpyHostToDevice));

    // Compute A = B * B^T on GPU
    const dim3 mBlock(16, 16);
    const dim3 mGrid((n + 15) / 16, (n + 15) / 16);
    matmulKernel<<<mGrid, mBlock>>>(d_B, d_A, n);
    CUDA_CHECK_VOID(cudaDeviceSynchronize());

    // Add n to diagonal on GPU
    const int diagGrid = (n + 255) / 256;
    addDiagonalKernel<<<diagGrid, 256>>>(d_A, n, static_cast<double>(n));
    CUDA_CHECK_VOID(cudaDeviceSynchronize());

    CUDA_CHECK_VOID(cudaMemcpy(A.data(), d_A, nnz * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK_VOID(cudaFree(d_A));
    CUDA_CHECK_VOID(cudaFree(d_B));
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    std::vector<double> reconstructed(n * n);
    double* d_L = nullptr;
    double* d_R = nullptr;
    size_t nnz = n * n;

    CUDA_CHECK(cudaMalloc(&d_L, nnz * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_R, nnz * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_L, L.data(), nnz * sizeof(double), cudaMemcpyHostToDevice));

    const dim3 vBlock(16, 16);
    const dim3 vGrid((n + 15) / 16, (n + 15) / 16);
    validateKernel<<<vGrid, vBlock>>>(d_L, d_R, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(reconstructed.data(), d_R, nnz * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_L));
    CUDA_CHECK(cudaFree(d_R));

    // Compare with original on host
    double maxError = 0.0;
    double relError = 0.0;

    for (size_t i = 0; i < nnz; ++i) {
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
