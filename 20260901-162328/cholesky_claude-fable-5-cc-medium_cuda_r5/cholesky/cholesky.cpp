#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// Blocked Cholesky decomposition parallelized with CUDA (right-looking algorithm)
// Decomposes positive definite matrix A into L * L^T where L is lower triangular

#define CUDA_CHECK(call)                                                            \
    do {                                                                            \
        cudaError_t err_ = (call);                                                  \
        if (err_ != cudaSuccess) {                                                  \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,        \
                    cudaGetErrorString(err_));                                      \
            exit(1);                                                                \
        }                                                                           \
    } while (0)

constexpr int NB = 32; // Panel width / tile size

// Factor the m x m diagonal block starting at (k, k) using a single thread block.
// On a non-positive-definite pivot, records the failing global index in *info
// (first failure wins) and leaves the block untouched.
__global__ void cholDiagKernel(double* A, size_t n, size_t k, int m, int* info) {
    __shared__ double s[NB][NB + 1];
    __shared__ int bad;
    const int tx = threadIdx.x;

    if (tx == 0) bad = 0;
    if (tx < m) {
        for (int j = 0; j <= tx; ++j) {
            s[tx][j] = A[(k + tx) * n + (k + j)];
        }
    }
    __syncthreads();

    for (int j = 0; j < m; ++j) {
        if (tx == j) {
            double sum = 0.0;
            for (int p = 0; p < j; ++p) {
                sum += s[j][p] * s[j][p];
            }
            const double val = s[j][j] - sum;
            if (val <= 0.0) {
                bad = 1;
                atomicCAS(info, -1, (int)(k + j));
            } else {
                s[j][j] = sqrt(val);
            }
        }
        __syncthreads();
        if (bad) return;
        if (tx > j && tx < m) {
            double sum = 0.0;
            for (int p = 0; p < j; ++p) {
                sum += s[tx][p] * s[j][p];
            }
            s[tx][j] = (s[tx][j] - sum) / s[j][j];
        }
        __syncthreads();
    }

    if (tx < m) {
        for (int j = 0; j <= tx; ++j) {
            A[(k + tx) * n + (k + j)] = s[tx][j];
        }
    }
}

// Triangular solve for the panel below the diagonal block:
// L21 = A21 * L11^{-T}. One thread per row of the panel; the diagonal block
// is staged in shared memory and each row's partial results stay in registers.
__global__ void cholTrsmKernel(double* A, size_t n, size_t k, int m) {
    __shared__ double L[NB][NB + 1];

    for (int idx = threadIdx.x; idx < NB * NB; idx += blockDim.x) {
        const int r = idx / NB;
        const int c = idx % NB;
        if (r < m && c <= r) {
            L[r][c] = A[(k + r) * n + (k + c)];
        }
    }
    __syncthreads();

    const size_t row = k + m + (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;

    double x[NB];
#pragma unroll
    for (int j = 0; j < NB; ++j) {
        x[j] = (j < m) ? A[row * n + k + j] : 0.0;
    }
#pragma unroll
    for (int j = 0; j < NB; ++j) {
        if (j < m) {
            double sum = x[j];
#pragma unroll
            for (int p = 0; p < j; ++p) {
                sum -= x[p] * L[j][p];
            }
            x[j] = sum / L[j][j];
        }
    }
    for (int j = 0; j < m; ++j) {
        A[row * n + k + j] = x[j];
    }
}

// Symmetric rank-m update of the trailing submatrix (lower triangle only):
// A22 -= L21 * L21^T. Tiled with shared memory; each thread computes four
// rows of a column for register-level parallelism. Blocks strictly above the
// diagonal exit immediately.
__global__ void cholSyrkKernel(double* A, size_t n, size_t k, int m) {
    if (blockIdx.y > blockIdx.x) return;

    __shared__ double Ls[NB][NB + 1];
    __shared__ double Rs[NB][NB + 1];

    const size_t base = k + m;
    const int tx = threadIdx.x; // column within tile, 0..NB-1
    const int ty = threadIdx.y; // row group, 0..7 (each handles 4 rows)

    for (int r = ty; r < NB; r += 8) {
        const size_t lr = base + (size_t)blockIdx.x * NB + r;
        Ls[r][tx] = (lr < n && tx < m) ? A[lr * n + k + tx] : 0.0;
        const size_t rr = base + (size_t)blockIdx.y * NB + r;
        Rs[r][tx] = (rr < n && tx < m) ? A[rr * n + k + tx] : 0.0;
    }
    __syncthreads();

    double acc[4] = {0.0, 0.0, 0.0, 0.0};
#pragma unroll
    for (int p = 0; p < NB; ++p) {
        const double rv = Rs[tx][p];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            acc[q] += Ls[ty * 4 + q][p] * rv;
        }
    }

    const size_t col = base + (size_t)blockIdx.y * NB + tx;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const size_t row = base + (size_t)blockIdx.x * NB + ty * 4 + q;
        if (row < n && col <= row) {
            A[row * n + col] -= acc[q];
        }
    }
}

// Zero out the strictly upper triangular part of the matrix
__global__ void zeroUpperKernel(double* A, size_t n) {
    const size_t j = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t i = (size_t)blockIdx.y * blockDim.y + threadIdx.y;
    if (i < n && j < n && j > i) {
        A[i * n + j] = 0.0;
    }
}

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    if (n == 0) return true;

    double* dA = nullptr;
    int* dInfo = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, n * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dInfo, sizeof(int)));

    const int noFailure = -1;
    CUDA_CHECK(cudaMemcpy(dInfo, &noFailure, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dA, A.data(), n * n * sizeof(double), cudaMemcpyHostToDevice));

    for (size_t k = 0; k < n; k += NB) {
        const int m = (int)std::min((size_t)NB, n - k);
        cholDiagKernel<<<1, NB>>>(dA, n, k, m, dInfo);

        const size_t t = n - k - m; // rows below the current panel
        if (t > 0) {
            const int trsmBlock = 256;
            const int trsmGrid = (int)((t + trsmBlock - 1) / trsmBlock);
            cholTrsmKernel<<<trsmGrid, trsmBlock>>>(dA, n, k, m);

            const int g = (int)((t + NB - 1) / NB);
            cholSyrkKernel<<<dim3(g, g), dim3(NB, 8)>>>(dA, n, k, m);
        }
    }

    {
        const dim3 block(32, 8);
        const dim3 grid((unsigned)((n + block.x - 1) / block.x),
                        (unsigned)((n + block.y - 1) / block.y));
        zeroUpperKernel<<<grid, block>>>(dA, n);
    }
    CUDA_CHECK(cudaGetLastError());

    int info = -1;
    CUDA_CHECK(cudaMemcpy(&info, dInfo, sizeof(int), cudaMemcpyDeviceToHost));
    if (info >= 0) {
        // Matrix is not positive definite
        printf("Error: Matrix is not positive definite at diagonal element %zu\n",
               (size_t)info);
        cudaFree(dA);
        cudaFree(dInfo);
        return false;
    }

    CUDA_CHECK(cudaMemcpy(A.data(), dA, n * n * sizeof(double), cudaMemcpyDeviceToHost));
    cudaFree(dA);
    cudaFree(dInfo);
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

    // Initialize the CUDA context so device setup is not part of the timing
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaFree(0));

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
