#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// CUDA Cholesky decomposition (blocked right-looking algorithm)
// Decomposes positive definite matrix A into L * L^T where L is lower triangular

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        cudaError_t err_ = (call);                                                    \
        if (err_ != cudaSuccess) {                                                    \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_),     \
                    __FILE__, __LINE__);                                              \
            exit(1);                                                                  \
        }                                                                             \
    } while (0)

// Panel width for the blocked factorization
constexpr int NB = 64;
// Tile size for the trailing-matrix update and GEMM kernels
constexpr int TILE = 32;

// Factor the NB x NB diagonal block starting at (j0, j0) with a single thread
// block (unblocked Cholesky in shared memory). On a non-positive-definite
// diagonal, records the 1-based global index of the first failure in *info.
__global__ void potf2_kernel(double* A, size_t n, size_t j0, int nb, int* info) {
    __shared__ double s[NB][NB + 1];
    const int tid = threadIdx.x;

    // Load the lower triangle (and diagonal) of the block
    for (int idx = tid; idx < nb * nb; idx += blockDim.x) {
        const int r = idx / nb;
        const int c = idx % nb;
        if (c <= r) {
            s[r][c] = A[(j0 + r) * n + (j0 + c)];
        }
    }
    __syncthreads();

    for (int k = 0; k < nb; ++k) {
        if (tid == 0) {
            const double d = s[k][k];
            if (d <= 0.0 && *info == 0) {
                *info = (int)(j0 + k) + 1;
            }
            s[k][k] = sqrt(d);
        }
        __syncthreads();
        if (tid > k && tid < nb) {
            s[tid][k] /= s[k][k];
        }
        __syncthreads();
        if (tid > k && tid < nb) {
            const double lik = s[tid][k];
            for (int j = k + 1; j <= tid; ++j) {
                s[tid][j] -= lik * s[j][k];
            }
        }
        __syncthreads();
    }

    // Store the factored lower triangle back
    for (int idx = tid; idx < nb * nb; idx += blockDim.x) {
        const int r = idx / nb;
        const int c = idx % nb;
        if (c <= r) {
            A[(j0 + r) * n + (j0 + c)] = s[r][c];
        }
    }
}

// Panel solve: L21 = A21 * L11^{-T}. Each thread performs the forward
// substitution for one row of the panel, with L11 cached in shared memory.
__global__ void trsm_kernel(double* A, size_t n, size_t j0, int nb) {
    __shared__ double L[NB][NB + 1];
    const int tid = threadIdx.x;

    for (int idx = tid; idx < nb * nb; idx += blockDim.x) {
        const int r = idx / nb;
        const int c = idx % nb;
        if (c <= r) {
            L[r][c] = A[(j0 + r) * n + (j0 + c)];
        }
    }
    __syncthreads();

    const size_t row = j0 + nb + (size_t)blockIdx.x * blockDim.x + tid;
    if (row >= n) return;

    double x[NB];
    double* a = &A[row * n + j0];
    for (int k = 0; k < nb; ++k) {
        x[k] = a[k];
    }
    for (int k = 0; k < nb; ++k) {
        double sum = x[k];
        for (int m = 0; m < k; ++m) {
            sum -= x[m] * L[k][m];
        }
        x[k] = sum / L[k][k];
    }
    for (int k = 0; k < nb; ++k) {
        a[k] = x[k];
    }
}

// Cooperative load of a TILE x TILE tile of rows [r0, r0+TILE) x columns
// [c0, c0+TILE) into shared memory, zero-padded outside the matrix.
__device__ __forceinline__ void load_tile(double dst[TILE][TILE + 1], const double* A,
                                          size_t n, size_t r0, size_t c0, int tid,
                                          int nthreads) {
    for (int idx = tid; idx < TILE * TILE; idx += nthreads) {
        const int r = idx / TILE;
        const int c = idx % TILE;
        const size_t row = r0 + r;
        const size_t col = c0 + c;
        dst[r][c] = (row < n && col < n) ? A[row * n + col] : 0.0;
    }
}

// Trailing-matrix update: A22 -= L21 * L21^T (lower triangle only). Each
// block of 16x16 threads computes a TILE x TILE output tile with 2x2 register
// blocking; blocks strictly above the diagonal exit early.
__global__ void syrk_kernel(double* A, size_t n, size_t j0, int nb) {
    __shared__ double Pi[TILE][TILE + 1];
    __shared__ double Pj[TILE][TILE + 1];

    if (blockIdx.x > blockIdx.y) return;

    const size_t t0 = j0 + nb;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * 16 + tx;
    const size_t ri0 = t0 + (size_t)blockIdx.y * TILE;
    const size_t rj0 = t0 + (size_t)blockIdx.x * TILE;

    double acc00 = 0.0, acc01 = 0.0, acc10 = 0.0, acc11 = 0.0;
    for (int kk = 0; kk < nb; kk += TILE) {
        load_tile(Pi, A, n, ri0, j0 + kk, tid, 256);
        load_tile(Pj, A, n, rj0, j0 + kk, tid, 256);
        __syncthreads();
#pragma unroll
        for (int k = 0; k < TILE; ++k) {
            const double a0 = Pi[ty][k];
            const double a1 = Pi[ty + 16][k];
            const double b0 = Pj[tx][k];
            const double b1 = Pj[tx + 16][k];
            acc00 += a0 * b0;
            acc01 += a0 * b1;
            acc10 += a1 * b0;
            acc11 += a1 * b1;
        }
        __syncthreads();
    }

    const size_t row0 = ri0 + ty;
    const size_t row1 = ri0 + ty + 16;
    const size_t col0 = rj0 + tx;
    const size_t col1 = rj0 + tx + 16;
    if (row0 < n && col0 <= row0) A[row0 * n + col0] -= acc00;
    if (row0 < n && col1 <= row0) A[row0 * n + col1] -= acc01;
    if (row1 < n && col0 <= row1) A[row1 * n + col0] -= acc10;
    if (row1 < n && col1 <= row1) A[row1 * n + col1] -= acc11;
}

// C = M * M^T for an n x n matrix (used for generation and validation)
__global__ void mmt_kernel(double* C, const double* M, size_t n) {
    __shared__ double Mi[TILE][TILE + 1];
    __shared__ double Mj[TILE][TILE + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * 16 + tx;
    const size_t ri0 = (size_t)blockIdx.y * TILE;
    const size_t rj0 = (size_t)blockIdx.x * TILE;

    double acc00 = 0.0, acc01 = 0.0, acc10 = 0.0, acc11 = 0.0;
    for (size_t kk = 0; kk < n; kk += TILE) {
        load_tile(Mi, M, n, ri0, kk, tid, 256);
        load_tile(Mj, M, n, rj0, kk, tid, 256);
        __syncthreads();
#pragma unroll
        for (int k = 0; k < TILE; ++k) {
            const double a0 = Mi[ty][k];
            const double a1 = Mi[ty + 16][k];
            const double b0 = Mj[tx][k];
            const double b1 = Mj[tx + 16][k];
            acc00 += a0 * b0;
            acc01 += a0 * b1;
            acc10 += a1 * b0;
            acc11 += a1 * b1;
        }
        __syncthreads();
    }

    const size_t row0 = ri0 + ty;
    const size_t row1 = ri0 + ty + 16;
    const size_t col0 = rj0 + tx;
    const size_t col1 = rj0 + tx + 16;
    if (row0 < n && col0 < n) C[row0 * n + col0] = acc00;
    if (row0 < n && col1 < n) C[row0 * n + col1] = acc01;
    if (row1 < n && col0 < n) C[row1 * n + col0] = acc10;
    if (row1 < n && col1 < n) C[row1 * n + col1] = acc11;
}

__global__ void add_diagonal_kernel(double* A, size_t n, double val) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        A[i * n + i] += val;
    }
}

// Zero out the upper triangular part
__global__ void zero_upper_kernel(double* A, size_t n) {
    const size_t col = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t row = (size_t)blockIdx.y * blockDim.y + threadIdx.y;
    if (row < n && col < n && col > row) {
        A[row * n + col] = 0.0;
    }
}

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    // A is stored in row-major order
    const size_t bytes = n * n * sizeof(double);

    double* d_A = nullptr;
    int* d_info = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_info, 0, sizeof(int)));

    int info = 0;
    for (size_t j0 = 0; j0 < n; j0 += NB) {
        const int nb = (int)std::min((size_t)NB, n - j0);

        potf2_kernel<<<1, NB>>>(d_A, n, j0, nb, d_info);
        CUDA_CHECK(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
        if (info != 0) {
            // Matrix is not positive definite
            printf("Error: Matrix is not positive definite at diagonal element %zu\n",
                   (size_t)(info - 1));
            cudaFree(d_A);
            cudaFree(d_info);
            return false;
        }

        const size_t m = n - j0 - nb; // rows below the panel
        if (m > 0) {
            const int threads = 128;
            const unsigned blocks = (unsigned)((m + threads - 1) / threads);
            trsm_kernel<<<blocks, threads>>>(d_A, n, j0, nb);

            const unsigned tiles = (unsigned)((m + TILE - 1) / TILE);
            dim3 grid(tiles, tiles);
            dim3 block(16, 16);
            syrk_kernel<<<grid, block>>>(d_A, n, j0, nb);
        }
    }

    // Zero out upper triangular part
    {
        dim3 block(32, 8);
        dim3 grid((unsigned)((n + 31) / 32), (unsigned)((n + 7) / 8));
        zero_upper_kernel<<<grid, block>>>(d_A, n);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(A.data(), d_A, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_info));

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

    // Compute A = B * B^T on the GPU
    const size_t bytes = n * n * sizeof(double);
    double* d_B = nullptr;
    double* d_A = nullptr;
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), bytes, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((unsigned)((n + TILE - 1) / TILE), (unsigned)((n + TILE - 1) / TILE));
    mmt_kernel<<<grid, block>>>(d_A, d_B, n);

    // Add diagonal dominance to ensure positive definiteness
    add_diagonal_kernel<<<(unsigned)((n + 255) / 256), 256>>>(d_A, n, (double)n);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(A.data(), d_A, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_A));
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix

    std::vector<double> reconstructed(n * n);

    // Compute L * L^T on the GPU
    const size_t bytes = n * n * sizeof(double);
    double* d_L = nullptr;
    double* d_R = nullptr;
    CUDA_CHECK(cudaMalloc(&d_L, bytes));
    CUDA_CHECK(cudaMalloc(&d_R, bytes));
    CUDA_CHECK(cudaMemcpy(d_L, L.data(), bytes, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((unsigned)((n + TILE - 1) / TILE), (unsigned)((n + TILE - 1) / TILE));
    mmt_kernel<<<grid, block>>>(d_R, d_L, n);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(reconstructed.data(), d_R, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_L));
    CUDA_CHECK(cudaFree(d_R));

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

    // Initialize the CUDA context up front so it is not part of the timing
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
