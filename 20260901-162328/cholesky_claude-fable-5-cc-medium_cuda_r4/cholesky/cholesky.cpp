#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// CUDA-parallel Cholesky decomposition (blocked right-looking algorithm)
// Decomposes positive definite matrix A into L * L^T where L is lower triangular

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), \
                    __FILE__, __LINE__);                                          \
            exit(1);                                                              \
        }                                                                         \
    } while (0)

constexpr int NB = 64;    // panel width
constexpr int TILE = 32;  // GEMM tile size

// Factorize the kb x kb diagonal block at (k, k) in place (single thread block).
// On failure (non-positive-definite pivot), records the first failing global
// index in *flag.
__global__ void potf2Kernel(double* __restrict__ A, const size_t n, const size_t k,
                            const int kb, int* __restrict__ flag) {
    __shared__ double s[NB][NB + 1];
    const int tid = threadIdx.x;

    if (tid < kb) {
        for (int j = 0; j < kb; ++j) {
            s[tid][j] = A[(k + tid) * n + k + j];
        }
    }
    __syncthreads();

    for (int j = 0; j < kb; ++j) {
        if (tid == j) {
            const double val = s[j][j];
            if (val <= 0.0) {
                atomicCAS(flag, -1, (int)(k + j));
            }
            s[j][j] = sqrt(val);
        }
        __syncthreads();
        if (tid > j && tid < kb) {
            s[tid][j] /= s[j][j];
        }
        __syncthreads();
        if (tid > j && tid < kb) {
            const double lij = s[tid][j];
            for (int t = j + 1; t <= tid; ++t) {
                s[tid][t] -= lij * s[t][j];
            }
        }
        __syncthreads();
    }

    if (tid < kb) {
        for (int j = 0; j <= tid; ++j) {
            A[(k + tid) * n + k + j] = s[tid][j];
        }
    }
}

// Compute the inverse of the factored lower-triangular diagonal block into
// invD (NB-stride, row-major). One thread per column (single thread block).
__global__ void trinvKernel(const double* __restrict__ A, const size_t n, const size_t k,
                            const int kb, double* __restrict__ invD) {
    __shared__ double L[NB][NB + 1];
    const int tid = threadIdx.x;

    if (tid < kb) {
        for (int t = 0; t <= tid; ++t) {
            L[tid][t] = A[(k + tid) * n + k + t];
        }
    }
    __syncthreads();

    const int j = tid;
    if (j < kb) {
        double x[NB];
        for (int i = 0; i < j; ++i) {
            invD[i * NB + j] = 0.0;
        }
        for (int i = j; i < kb; ++i) {
            double sum = (i == j) ? 1.0 : 0.0;
            for (int t = j; t < i; ++t) {
                sum -= L[i][t] * x[t];
            }
            x[i] = sum / L[i][i];
            invD[i * NB + j] = x[i];
        }
    }
}

// Panel update: L21 = A21 * inv(L11)^T for the m x kb panel below the
// diagonal block. Each 32x32 thread block handles 32 full rows of the panel
// (both column halves) so the in-place update has no cross-block hazards.
__global__ void trsmPanelKernel(double* __restrict__ A, const size_t n, const size_t k,
                                const int kb, const size_t m,
                                const double* __restrict__ invD) {
    __shared__ double sA[TILE][NB + 1];
    __shared__ double sI[NB][TILE + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const size_t row = (size_t)blockIdx.x * TILE + ty;
    double* panel = A + (k + kb) * n + k;

    // Load this block's 32 x kb panel rows into shared memory.
    for (int j = tx; j < NB; j += TILE) {
        sA[ty][j] = (row < m && j < kb) ? panel[row * n + j] : 0.0;
    }
    __syncthreads();

    double acc0 = 0.0;
    double acc1 = 0.0;
    for (int t0 = 0; t0 < kb; t0 += TILE) {
        sI[ty][tx] = (t0 + tx < kb) ? invD[ty * NB + t0 + tx] : 0.0;
        sI[ty + TILE][tx] = (ty + TILE < NB && t0 + tx < kb) ? invD[(ty + TILE) * NB + t0 + tx] : 0.0;
        __syncthreads();
#pragma unroll
        for (int tt = 0; tt < TILE; ++tt) {
            const double a = sA[ty][t0 + tt];
            acc0 += a * sI[tx][tt];
            acc1 += a * sI[tx + TILE][tt];
        }
        __syncthreads();
    }

    if (row < m) {
        if (tx < kb) panel[row * n + tx] = acc0;
        if (tx + TILE < kb) panel[row * n + tx + TILE] = acc1;
    }
}

// Trailing update (SYRK): A22 -= L21 * L21^T on the m x m trailing matrix at
// (k+kb, k+kb). Only tiles on or below the diagonal are computed.
__global__ void syrkKernel(double* __restrict__ A, const size_t n, const size_t k,
                           const int kb, const size_t m) {
    if (blockIdx.x > blockIdx.y) return;

    __shared__ double sA[TILE][TILE + 1];
    __shared__ double sB[TILE][TILE + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const size_t i = (size_t)blockIdx.y * TILE + ty;
    const size_t j = (size_t)blockIdx.x * TILE + tx;
    const double* panel = A + (k + kb) * n + k;  // L21, m x kb

    double acc = 0.0;
    for (int t0 = 0; t0 < kb; t0 += TILE) {
        const size_t ra = (size_t)blockIdx.y * TILE + ty;
        const size_t rb = (size_t)blockIdx.x * TILE + ty;
        sA[ty][tx] = (ra < m && t0 + tx < kb) ? panel[ra * n + t0 + tx] : 0.0;
        sB[ty][tx] = (rb < m && t0 + tx < kb) ? panel[rb * n + t0 + tx] : 0.0;
        __syncthreads();
#pragma unroll
        for (int tt = 0; tt < TILE; ++tt) {
            acc += sA[ty][tt] * sB[tx][tt];
        }
        __syncthreads();
    }

    if (i < m && j < m) {
        A[(k + kb + i) * n + k + kb + j] -= acc;
    }
}

// Zero out the strict upper triangular part.
__global__ void zeroUpperKernel(double* __restrict__ A, const size_t n) {
    const size_t i = (size_t)blockIdx.y * blockDim.y + threadIdx.y;
    const size_t j = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && j < n && j > i) {
        A[i * n + j] = 0.0;
    }
}

// Tiled C = B * B^T (used for matrix generation and validation).
// Each element accumulates over k in ascending order, matching the
// sequential reference summation order.
__global__ void gemmAATKernel(const double* __restrict__ B, double* __restrict__ C,
                              const size_t n) {
    __shared__ double sA[TILE][TILE + 1];
    __shared__ double sB[TILE][TILE + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const size_t i = (size_t)blockIdx.y * TILE + ty;
    const size_t j = (size_t)blockIdx.x * TILE + tx;

    double acc = 0.0;
    for (size_t t0 = 0; t0 < n; t0 += TILE) {
        const size_t ra = (size_t)blockIdx.y * TILE + ty;
        const size_t rb = (size_t)blockIdx.x * TILE + ty;
        sA[ty][tx] = (ra < n && t0 + tx < n) ? B[ra * n + t0 + tx] : 0.0;
        sB[ty][tx] = (rb < n && t0 + tx < n) ? B[rb * n + t0 + tx] : 0.0;
        __syncthreads();
#pragma unroll
        for (int tt = 0; tt < TILE; ++tt) {
            acc += sA[ty][tt] * sB[tx][tt];
        }
        __syncthreads();
    }

    if (i < n && j < n) {
        C[i * n + j] = acc;
    }
}

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    double* d_A = nullptr;
    double* d_invD = nullptr;
    int* d_flag = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, n * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_invD, NB * NB * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_flag, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), n * n * sizeof(double), cudaMemcpyHostToDevice));

    const int initFlag = -1;
    CUDA_CHECK(cudaMemcpy(d_flag, &initFlag, sizeof(int), cudaMemcpyHostToDevice));

    const dim3 tileBlock(TILE, TILE);

    for (size_t k = 0; k < n; k += NB) {
        const int kb = (int)std::min((size_t)NB, n - k);
        const size_t m = n - k - kb;  // rows below the diagonal block

        potf2Kernel<<<1, NB>>>(d_A, n, k, kb, d_flag);

        if (m > 0) {
            trinvKernel<<<1, NB>>>(d_A, n, k, kb, d_invD);

            const unsigned int mt = (unsigned int)((m + TILE - 1) / TILE);
            trsmPanelKernel<<<mt, tileBlock>>>(d_A, n, k, kb, m, d_invD);
            syrkKernel<<<dim3(mt, mt), tileBlock>>>(d_A, n, k, kb, m);
        }
    }

    // Zero out upper triangular part
    {
        const unsigned int nt = (unsigned int)((n + TILE - 1) / TILE);
        zeroUpperKernel<<<dim3(nt, nt), tileBlock>>>(d_A, n);
    }

    int flag = -1;
    CUDA_CHECK(cudaMemcpy(&flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaGetLastError());

    if (flag >= 0) {
        // Matrix is not positive definite
        printf("Error: Matrix is not positive definite at diagonal element %zu\n",
               (size_t)flag);
        cudaFree(d_A);
        cudaFree(d_invD);
        cudaFree(d_flag);
        return false;
    }

    CUDA_CHECK(cudaMemcpy(A.data(), d_A, n * n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_invD));
    CUDA_CHECK(cudaFree(d_flag));

    return true;
}

// Compute C = B * B^T on the GPU.
static void gemmAAT(const std::vector<double>& B, std::vector<double>& C, const size_t n) {
    double* d_B = nullptr;
    double* d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_B, n * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_C, n * n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), n * n * sizeof(double), cudaMemcpyHostToDevice));

    const unsigned int nt = (unsigned int)((n + TILE - 1) / TILE);
    gemmAATKernel<<<dim3(nt, nt), dim3(TILE, TILE)>>>(d_B, d_C, n);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(C.data(), d_C, n * n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
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
    gemmAAT(B, A, n);

    // Add diagonal dominance to ensure positive definiteness
    for (size_t i = 0; i < n; ++i) {
        A[i * n + i] += n;
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix

    std::vector<double> reconstructed(n * n);

    // Compute L * L^T
    gemmAAT(L, reconstructed, n);

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

    // Initialize CUDA context up front so it is not part of the timed region
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
