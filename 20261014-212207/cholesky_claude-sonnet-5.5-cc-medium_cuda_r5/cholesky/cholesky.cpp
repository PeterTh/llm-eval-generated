#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <climits>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), \
                    __FILE__, __LINE__);                                          \
            exit(1);                                                              \
        }                                                                         \
    } while (0)

// Blocked right-looking Cholesky decomposition on the GPU.
// Decomposes positive definite matrix A into L * L^T where L is lower triangular.
// A is stored in row-major order on the device.

constexpr int NB = 64;    // panel width
constexpr int TILE = 64;  // GEMM tile (rows and columns)
constexpr int BK = 16;    // GEMM k-step

// C[i,j] = beta*C[i,j] + alpha * sum_k X[i,k] * Y[j,k]   (C = beta*C + alpha*X*Y^T)
// If lowerOnly, tiles strictly above the diagonal are skipped.
__global__ void __launch_bounds__(256)
gemmNT(double* __restrict__ C, size_t ldc, const double* __restrict__ X, size_t ldx,
       const double* __restrict__ Y, size_t ldy, int M, int N, int K,
       double alpha, double beta, bool lowerOnly) {
    if (lowerOnly && blockIdx.x > blockIdx.y) return;
    __shared__ double Xs[BK][TILE + 1];
    __shared__ double Ys[BK][TILE + 1];

    const int tx = threadIdx.x & 15, ty = threadIdx.x >> 4;
    const int row0 = blockIdx.y * TILE, col0 = blockIdx.x * TILE;
    double acc[4][4] = {};

    for (int k0 = 0; k0 < K; k0 += BK) {
        const int c = threadIdx.x & 15;
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            const int r = (threadIdx.x >> 4) + 16 * s;
            const int gk = k0 + c;
            const int gi = row0 + r, gj = col0 + r;
            Xs[c][r] = (gi < M && gk < K) ? X[gi * ldx + gk] : 0.0;
            Ys[c][r] = (gj < N && gk < K) ? Y[gj * ldy + gk] : 0.0;
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            double xr[4], yr[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                xr[i] = Xs[kk][ty + 16 * i];
                yr[i] = Ys[kk][tx + 16 * i];
            }
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] = fma(xr[i], yr[j], acc[i][j]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int gi = row0 + ty + 16 * i;
        if (gi >= M) continue;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int gj = col0 + tx + 16 * j;
            if (gj >= N) continue;
            double* p = &C[gi * ldc + gj];
            *p = (beta == 0.0 ? 0.0 : beta * *p) + alpha * acc[i][j];
        }
    }
}

__global__ void addDiagonal(double* A, int n, double v) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) A[(size_t)i * n + i] += v;
}

// Factor the b x b diagonal block at (k0,k0) with one thread block (one thread per row).
__global__ void factorDiagonal(double* A, int n, int k0, int b, int* failIdx) {
    __shared__ double s[NB][NB + 1];
    const int t = threadIdx.x;
    if (t < b)
        for (int j = 0; j < b; ++j) s[t][j] = A[(size_t)(k0 + t) * n + k0 + j];
    __syncthreads();
    for (int k = 0; k < b; ++k) {
        if (t == 0) {
            const double val = s[k][k];
            if (!(val > 0.0)) atomicMin(failIdx, k0 + k);
            s[k][k] = sqrt(val);
        }
        __syncthreads();
        if (t > k && t < b) s[t][k] /= s[k][k];
        __syncthreads();
        if (t > k && t < b) {
            const double l = s[t][k];
            for (int j = k + 1; j <= t; ++j) s[t][j] -= l * s[j][k];
        }
        __syncthreads();
    }
    if (t < b)
        for (int j = 0; j <= t; ++j) A[(size_t)(k0 + t) * n + k0 + j] = s[t][j];
}

// Panel solve: L21 = A21 * L11^{-T}; one thread per row below the diagonal block.
__global__ void solvePanel(double* A, int n, int k0, int b) {
    __shared__ double L[NB][NB + 1];
    for (int e = threadIdx.x; e < b * b; e += blockDim.x) {
        const int r = e / b, c = e % b;
        L[r][c] = (c <= r) ? A[(size_t)(k0 + r) * n + k0 + c] : 0.0;
    }
    __syncthreads();
    const int row = k0 + b + blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;
    double* p = &A[(size_t)row * n + k0];
    double x[NB];
#pragma unroll
    for (int j = 0; j < NB; ++j) x[j] = (j < b) ? p[j] : 0.0;
#pragma unroll
    for (int j = 0; j < NB; ++j) {
        if (j < b) {
            double sum = 0.0;
#pragma unroll
            for (int k = 0; k < j; ++k) sum = fma(x[k], L[j][k], sum);
            x[j] = (x[j] - sum) / L[j][j];
            p[j] = x[j];
        }
    }
}

__global__ void zeroUpper(double* A, int n) {
    size_t e = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= (size_t)n * n) return;
    if ((e % n) > e / n) A[e] = 0.0;
}

// Factor the device matrix in place. Returns false if not positive definite.
bool choleskyDecomposition(double* dA, const size_t n) {
    int* dFail;
    CUDA_CHECK(cudaMalloc(&dFail, sizeof(int)));
    const int none = INT_MAX;
    CUDA_CHECK(cudaMemcpy(dFail, &none, sizeof(int), cudaMemcpyHostToDevice));

    const int N = (int)n;
    for (int k0 = 0; k0 < N; k0 += NB) {
        const int b = std::min(NB, N - k0);
        factorDiagonal<<<1, NB>>>(dA, N, k0, b, dFail);
        const int m = N - k0 - b;
        if (m > 0) {
            solvePanel<<<(m + 31) / 32, 32>>>(dA, N, k0, b);
            const int tiles = (m + TILE - 1) / TILE;
            double* trail = dA + (size_t)(k0 + b) * n + (k0 + b);
            const double* panel = dA + (size_t)(k0 + b) * n + k0;
            gemmNT<<<dim3(tiles, tiles), 256>>>(trail, n, panel, n, panel, n, m, m, b,
                                                -1.0, 1.0, true);
        }
    }
    zeroUpper<<<(unsigned)((n * n + 255) / 256), 256>>>(dA, N);
    CUDA_CHECK(cudaGetLastError());

    int fail;
    CUDA_CHECK(cudaMemcpy(&fail, dFail, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dFail));
    if (fail != none) {
        printf("Error: Matrix is not positive definite at diagonal element %d\n", fail);
        return false;
    }
    return true;
}

// Generate a symmetric positive definite matrix directly on the device
void generatePositiveDefiniteMatrix(double* dA, const size_t n) {
    // Method: Create A = B * B^T where B is random
    // This guarantees A is positive semi-definite
    // Then add identity to make it strictly positive definite

    std::vector<double> B(n * n);
    unsigned int seed = 42;

    // Generate random matrix B
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

    double* dB;
    CUDA_CHECK(cudaMalloc(&dB, n * n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(dB, B.data(), n * n * sizeof(double), cudaMemcpyHostToDevice));

    // Compute A = B * B^T
    const int N = (int)n, tiles = (N + TILE - 1) / TILE;
    gemmNT<<<dim3(tiles, tiles), 256>>>(dA, n, dB, n, dB, n, N, N, N, 1.0, 0.0, false);

    // Add diagonal dominance to ensure positive definiteness
    addDiagonal<<<(N + 255) / 256, 256>>>(dA, N, (double)n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(dB));
}

__device__ void atomicMaxNonNeg(double* addr, double v) {
    // non-negative doubles order the same as their bit patterns
    atomicMax((unsigned long long*)addr, (unsigned long long)__double_as_longlong(v));
}

__global__ void compareMax(const double* R, const double* A, size_t cnt, double* maxAbs,
                           double* maxRel) {
    double ma = 0.0, mr = 0.0;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < cnt;
         i += (size_t)gridDim.x * blockDim.x) {
        const double err = fabs(R[i] - A[i]);
        ma = fmax(ma, err);
        mr = fmax(mr, err / (fabs(A[i]) + 1e-10));
    }
    for (int o = 16; o > 0; o >>= 1) {
        ma = fmax(ma, __shfl_down_sync(0xffffffff, ma, o));
        mr = fmax(mr, __shfl_down_sync(0xffffffff, mr, o));
    }
    if ((threadIdx.x & 31) == 0) {
        atomicMaxNonNeg(maxAbs, ma);
        atomicMaxNonNeg(maxRel, mr);
    }
}

bool validateCholesky(const double* dL, const double* dA_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original
    double *dR, *dErr;
    CUDA_CHECK(cudaMalloc(&dR, n * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dErr, 2 * sizeof(double)));
    CUDA_CHECK(cudaMemset(dErr, 0, 2 * sizeof(double)));

    const int N = (int)n, tiles = (N + TILE - 1) / TILE;
    gemmNT<<<dim3(tiles, tiles), 256>>>(dR, n, dL, n, dL, n, N, N, N, 1.0, 0.0, false);
    compareMax<<<1024, 256>>>(dR, dA_orig, n * n, dErr, dErr + 1);
    CUDA_CHECK(cudaGetLastError());

    double err[2];
    CUDA_CHECK(cudaMemcpy(err, dErr, sizeof(err), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dR));
    CUDA_CHECK(cudaFree(dErr));
    const double maxError = err[0], relError = err[1];

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
    
    // Allocate matrix (device resident; host copy only for output)
    std::vector<double> A(n * n);
    double *dA = nullptr, *dA_orig = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, n * n * sizeof(double)));

    // Generate positive definite matrix
    printf("Generating positive definite matrix...\n");
    generatePositiveDefiniteMatrix(dA, n);

    if (validate) {
        CUDA_CHECK(cudaMalloc(&dA_orig, n * n * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(dA_orig, dA, n * n * sizeof(double), cudaMemcpyDeviceToDevice));
    }

    // Perform Cholesky decomposition
    printf("Computing Cholesky decomposition...\n");
    CUDA_CHECK(cudaDeviceSynchronize());
    auto start = std::chrono::high_resolution_clock::now();

    bool success = choleskyDecomposition(dA, n);
    CUDA_CHECK(cudaDeviceSynchronize());

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
        CUDA_CHECK(cudaMemcpy(A.data(), dA, n * n * sizeof(double), cudaMemcpyDeviceToHost));
        print_results(A, "CholeskyL");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateCholesky(dA, dA_orig, n);
        
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
