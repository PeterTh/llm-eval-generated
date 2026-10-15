#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// GPU Cholesky decomposition (blocked, right-looking) using CUDA.
// Decomposes positive definite matrix A into L * L^T where L is lower triangular.
// A is stored in row-major order.

#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err_ = (call);                                               \
        if (err_ != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), \
                    __FILE__, __LINE__);                                         \
            exit(1);                                                             \
        }                                                                        \
    } while (0)

constexpr int NB = 64;          // panel width
constexpr int TILE = 64;        // GEMM output tile
constexpr int TK = 16;          // GEMM k step
constexpr int TPB = 256;        // GEMM threads per block
constexpr int NOT_PD = 0x7fffffff;

// C (+)= alpha * A * B^T  with A (M x K), B (N x K), C (M x N), all row-major.
// If LOWER, only tiles/elements with col <= row are computed.
template <bool LOWER, bool ACC>
__global__ void __launch_bounds__(TPB)
gemmNT(const double* __restrict__ A, const double* __restrict__ B, double* C,
       size_t lda, size_t ldb, size_t ldc, int M, int N, int K, double alpha) {
    if (LOWER && blockIdx.x > blockIdx.y) return;

    __shared__ double As[TK][TILE + 1];
    __shared__ double Bs[TK][TILE + 1];

    const int tid = threadIdx.x;
    const int tx = tid % 16, ty = tid / 16;
    const int i0 = blockIdx.y * TILE, j0 = blockIdx.x * TILE;

    double acc[4][4];
#pragma unroll
    for (int r = 0; r < 4; ++r)
#pragma unroll
        for (int c = 0; c < 4; ++c) acc[r][c] = 0.0;

    const int lc = tid % TK, lr = tid / TK;

    for (int kk = 0; kk < K; kk += TK) {
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const int r = lr + 16 * q;
            const int gk = kk + lc;
            const int gi = i0 + r, gj = j0 + r;
            As[lc][r] = (gi < M && gk < K) ? A[(size_t)gi * lda + gk] : 0.0;
            Bs[lc][r] = (gj < N && gk < K) ? B[(size_t)gj * ldb + gk] : 0.0;
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < TK; ++k) {
            double a[4], b[4];
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                a[r] = As[k][ty + 16 * r];
                b[r] = Bs[k][tx + 16 * r];
            }
#pragma unroll
            for (int r = 0; r < 4; ++r)
#pragma unroll
                for (int c = 0; c < 4; ++c) acc[r][c] = fma(a[r], b[c], acc[r][c]);
        }
        __syncthreads();
    }

#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int gi = i0 + ty + 16 * r;
        if (gi >= M) continue;
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const int gj = j0 + tx + 16 * c;
            if (gj >= N || (LOWER && gj > gi)) continue;
            double* p = C + (size_t)gi * ldc + gj;
            *p = (ACC ? *p : 0.0) + alpha * acc[r][c];
        }
    }
}

__global__ void addDiagonal(double* A, size_t n, double v) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) A[i * n + i] += v;
}

// Factor the ib x ib diagonal block at (k,k) with a single thread block.
__global__ void factorDiag(double* A, size_t n, int k, int ib, int* status) {
    __shared__ double S[NB][NB + 1];
    const int tid = threadIdx.x;
    for (int idx = tid; idx < ib * ib; idx += blockDim.x) {
        const int i = idx / ib, j = idx % ib;
        S[i][j] = A[(size_t)(k + i) * n + k + j];
    }
    __syncthreads();
    for (int j = 0; j < ib; ++j) {
        if (tid == 0) {
            const double val = S[j][j];
            if (val <= 0.0) atomicMin(status, k + j);
            S[j][j] = sqrt(val);
        }
        __syncthreads();
        const double d = S[j][j];
        for (int i = j + 1 + tid; i < ib; i += blockDim.x) S[i][j] /= d;
        __syncthreads();
        const int rem = ib - j - 1;
        for (int idx = tid; idx < rem * rem; idx += blockDim.x) {
            const int i = j + 1 + idx / rem, c = j + 1 + idx % rem;
            if (c <= i) S[i][c] -= S[i][j] * S[c][j];
        }
        __syncthreads();
    }
    for (int idx = tid; idx < ib * ib; idx += blockDim.x) {
        const int i = idx / ib, j = idx % ib;
        if (j <= i) A[(size_t)(k + i) * n + k + j] = S[i][j];
    }
}

// Panel solve: rows below the diagonal block: X * L11^T = A21, one thread per row.
__global__ void trsmPanel(double* A, size_t n, int k, int ib) {
    __shared__ double L[NB][NB + 1];
    for (int idx = threadIdx.x; idx < ib * ib; idx += blockDim.x) {
        const int i = idx / ib, j = idx % ib;
        L[i][j] = A[(size_t)(k + i) * n + k + j];
    }
    __syncthreads();
    const size_t row = (size_t)k + ib + blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (row >= n) return;
    double* p = A + row * n + k;
    double x[NB];
#pragma unroll
    for (int j = 0; j < NB; ++j) {
        if (j < ib) {
            double s = p[j];
#pragma unroll
            for (int c = 0; c < j; ++c) s -= x[c] * L[j][c];
            x[j] = s / L[j][j];
            p[j] = x[j];
        }
    }
}

__global__ void zeroUpper(double* A, size_t n) {
    const size_t j = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    const size_t i = blockIdx.y;
    if (j < n && j > i) A[i * n + j] = 0.0;
}

static void launchGemm(bool lower, bool acc, const double* A, const double* B, double* C,
                       size_t lda, size_t ldb, size_t ldc, int M, int N, int K, double alpha) {
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    if (lower)
        gemmNT<true, true><<<grid, TPB>>>(A, B, C, lda, ldb, ldc, M, N, K, alpha);
    else if (acc)
        gemmNT<false, true><<<grid, TPB>>>(A, B, C, lda, ldb, ldc, M, N, K, alpha);
    else
        gemmNT<false, false><<<grid, TPB>>>(A, B, C, lda, ldb, ldc, M, N, K, alpha);
}

// Blocked Cholesky on a device-resident matrix. Returns false if not positive definite.
bool choleskyDecomposition(double* dA, const size_t n, int* dStatus) {
    const int init = NOT_PD;
    CUDA_CHECK(cudaMemcpy(dStatus, &init, sizeof(int), cudaMemcpyHostToDevice));

    for (size_t k = 0; k < n; k += NB) {
        const int ib = (int)std::min<size_t>(NB, n - k);
        factorDiag<<<1, 256>>>(dA, n, (int)k, ib, dStatus);
        const size_t rest = n - k - ib;
        if (rest == 0) break;
        trsmPanel<<<(unsigned)((rest + 63) / 64), 64>>>(dA, n, (int)k, ib);
        const double* L21 = dA + (k + ib) * n + k;
        double* A22 = dA + (k + ib) * n + (k + ib);
        launchGemm(true, true, L21, L21, A22, n, n, n, (int)rest, (int)rest, ib, -1.0);
    }
    zeroUpper<<<dim3((unsigned)((n + 255) / 256), (unsigned)n), 256>>>(dA, n);
    CUDA_CHECK(cudaGetLastError());

    int status;
    CUDA_CHECK(cudaMemcpy(&status, dStatus, sizeof(int), cudaMemcpyDeviceToHost));
    if (status != NOT_PD) {
        printf("Error: Matrix is not positive definite at diagonal element %d\n", status);
        return false;
    }
    return true;
}

// Generate a symmetric positive definite matrix directly into device memory
void generatePositiveDefiniteMatrix(double* dA, const size_t n) {
    // Method: A = B * B^T where B is random, plus n on the diagonal
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }
    double* dB;
    CUDA_CHECK(cudaMalloc(&dB, n * n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(dB, B.data(), n * n * sizeof(double), cudaMemcpyHostToDevice));
    launchGemm(false, false, dB, dB, dA, n, n, n, (int)n, (int)n, (int)n, 1.0);
    addDiagonal<<<(unsigned)((n + 255) / 256), 256>>>(dA, n, (double)n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(dB));
}

// Validate by computing L * L^T on the GPU and comparing with the original on the host
bool validateCholesky(const double* dL, const std::vector<double>& A_orig, const size_t n) {
    double* dR;
    CUDA_CHECK(cudaMalloc(&dR, n * n * sizeof(double)));
    launchGemm(false, false, dL, dL, dR, n, n, n, (int)n, (int)n, (int)n, 1.0);
    CUDA_CHECK(cudaGetLastError());
    std::vector<double> reconstructed(n * n);
    CUDA_CHECK(cudaMemcpy(reconstructed.data(), dR, n * n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dR));

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
    double* dA;
    int* dStatus;
    CUDA_CHECK(cudaMalloc(&dA, n * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dStatus, sizeof(int)));
    
    // Generate positive definite matrix
    printf("Generating positive definite matrix...\n");
    generatePositiveDefiniteMatrix(dA, n);
    
    if (validate) {
        A_orig.resize(n * n); // Save original for validation
        CUDA_CHECK(cudaMemcpy(A_orig.data(), dA, n * n * sizeof(double), cudaMemcpyDeviceToHost));
    }
    
    // Perform Cholesky decomposition
    printf("Computing Cholesky decomposition...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    bool success = choleskyDecomposition(dA, n, dStatus);
    if (success) {
        CUDA_CHECK(cudaMemcpy(A.data(), dA, n * n * sizeof(double), cudaMemcpyDeviceToHost));
    }
    
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
        bool valid = validateCholesky(dA, A_orig, n);
        
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
