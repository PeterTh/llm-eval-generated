#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#define NB 64

// CUDA kernel: in-place Cholesky of a small diagonal block in shared memory.
// The jb×jb block at A[start:start+jb, start:start+jb] is loaded into shared
// memory (row-major), factored, and written back.
__global__ void blockCholeskyKernel(double* __restrict__ A, int n, int start, int jb) {
    extern __shared__ double sblock[];
    const int tid = threadIdx.x;

    // Load block into shared memory (row-major)
    for (int idx = tid; idx < jb * jb; idx += blockDim.x) {
        int li = idx / jb;
        int lj = idx % jb;
        sblock[idx] = A[(start + li) * n + (start + lj)];
    }
    __syncthreads();

    // Column-by-column Cholesky on shared block (row-major layout)
    for (int j = 0; j < jb; ++j) {
        __syncthreads();
        // Diagonal element (single thread)
        if (tid == 0) {
            double sum = 0.0;
            for (int k = 0; k < j; ++k) {
                double v = sblock[j * jb + k];
                sum += v * v;
            }
            sblock[j * jb + j] = sqrt(sblock[j * jb + j] - sum);
        }
        __syncthreads();
        // Off-diagonal elements in column j (parallel across threads)
        for (int i = j + 1 + tid; i < jb; i += blockDim.x) {
            double s = 0.0;
            for (int k = 0; k < j; ++k)
                s += sblock[i * jb + k] * sblock[j * jb + k];
            sblock[i * jb + j] = (sblock[i * jb + j] - s) / sblock[j * jb + j];
        }
    }
    __syncthreads();

    // Write back to global memory (row-major)
    for (int idx = tid; idx < jb * jb; idx += blockDim.x) {
        int li = idx / jb;
        int lj = idx % jb;
        A[(start + li) * n + (start + lj)] = sblock[idx];
    }
}

// CUDA kernel: triangular solve for off-diagonal block
// Solve: L21 * L11^T = A21 (row-major)
// Each thread computes one row of L21
__global__ void triangularSolveKernel(double* __restrict__ A, int n, int j, int jb, int m) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m) return;
    
    int i = j + jb + row;
    
    // Solve for row i of L21
    for (int col = 0; col < jb; ++col) {
        double sum = 0.0;
        for (int k = 0; k < col; ++k) {
            sum += A[i * n + (j + k)] * A[(j + col) * n + (j + k)];
        }
        A[i * n + (j + col)] = (A[i * n + (j + col)] - sum) / A[(j + col) * n + (j + col)];
    }
}

// CUDA kernel: symmetric rank-k update
// A22 = A22 - L21 * L21^T (row-major)
__global__ void syrkUpdateKernel(double* __restrict__ A, int n, int j, int jb, int m) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= m || col > row) return;
    
    int i = j + jb + row;
    int k_idx = j + jb + col;
    
    double sum = 0.0;
    for (int p = 0; p < jb; ++p) {
        sum += A[i * n + (j + p)] * A[k_idx * n + (j + p)];
    }
    A[i * n + k_idx] -= sum;
}

// CUDA kernel: zero out the strictly upper-triangular part (row-major)
__global__ void zeroUpperTriangleKernel(double* __restrict__ A, int n) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)n * n;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = idx; i < total; i += stride) {
        int row = (int)(i / n);
        int col = (int)(i % n);
        if (col > row)
            A[i] = 0.0;
    }
}

// Blocked right-looking Cholesky using cuBLAS for BLAS-3 updates.
// Matrix A is n×n symmetric positive-definite, stored row-major.
// On exit the lower triangle contains L such that A = L*L^T; upper triangle is zeroed.
bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    double* d_A = nullptr;
    cudaError_t err;

    err = cudaMalloc(&d_A, n * n * sizeof(double));
    if (err != cudaSuccess) {
        printf("CUDA malloc failed: %s\n", cudaGetErrorString(err));
        return false;
    }

    err = cudaMemcpy(d_A, A.data(), n * n * sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("CUDA memcpy H2D failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_A);
        return false;
    }

    cublasHandle_t handle;
    cublasStatus_t st = cublasCreate(&handle);
    if (st != CUBLAS_STATUS_SUCCESS) {
        printf("cuBLAS create failed: %d\n", (int)st);
        cudaFree(d_A);
        return false;
    }

    // Set up the diagonal-block kernel shared memory
    const int sharedMemSize = NB * NB * (int)sizeof(double);
    cudaFuncSetAttribute(blockCholeskyKernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         sharedMemSize);

    // Use 256 threads for the diagonal block kernel (covers up to 256-wide blocks)
    const int diagThreads = 256;

    for (size_t j = 0; j < n; j += NB) {
        const int jb = (int)std::min((size_t)NB, n - j);

        // Step 1: factor the diagonal block A[j:j+jb, j:j+jb] = L11 * L11^T
        blockCholeskyKernel<<<1, diagThreads, sharedMemSize>>>(d_A, (int)n, (int)j, jb);
        cudaDeviceSynchronize();  // Ensure block factorization completes before cuBLAS

        if ((int)(j + NB) < (int)n) {
            const int m = (int)(n - j - jb);

            // Step 2: triangular solve for off-diagonal block
            // Solve: L21 * L11^T = A21 (row-major)
            dim3 block(256);
            dim3 grid((m + block.x - 1) / block.x);
            triangularSolveKernel<<<grid, block>>>(d_A, (int)n, (int)j, jb, m);

            // Step 3: symmetric rank-k update of trailing sub-matrix
            // A22 = A22 - L21 * L21^T (row-major)
            dim3 block2d(16, 16);
            dim3 grid2d((m + block2d.x - 1) / block2d.x, (m + block2d.y - 1) / block2d.y);
            syrkUpdateKernel<<<grid2d, block2d>>>(d_A, (int)n, (int)j, jb, m);
        }
    }

    // Zero out the strictly upper-triangular part (row-major)
    {
        const long long total = (long long)n * n;
        const int threads = 256;
        const int blocks = (int)std::min((long long)65535, (total + threads - 1) / threads);
        zeroUpperTriangleKernel<<<blocks, threads>>>(d_A, (int)n);
    }

    // Copy result back to host
    err = cudaMemcpy(A.data(), d_A, n * n * sizeof(double), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("CUDA memcpy D2H failed: %s\n", cudaGetErrorString(err));
        cublasDestroy(handle);
        cudaFree(d_A);
        return false;
    }

    cudaDeviceSynchronize();

    // Post-check positive-definiteness (equivalent to per-element check in original)
    for (size_t i = 0; i < n; ++i) {
        const double diag = A[i * n + i];
        if (std::isnan(diag) || diag <= 0.0) {
            printf("Error: Matrix is not positive definite at diagonal element %zu\n", i);
            cublasDestroy(handle);
            cudaFree(d_A);
            return false;
        }
    }

    cublasDestroy(handle);
    cudaFree(d_A);
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
