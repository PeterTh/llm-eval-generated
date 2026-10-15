#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <cuda_runtime.h>
#include <vector>

#include "../common/results_output.hpp"

// GPU Cholesky decomposition (blocked, right-looking) using CUDA.
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

constexpr int NB = 64;      // panel width
constexpr int TILE = 64;    // output tile of the update kernel
constexpr int KC = 32;      // k-chunk staged in shared memory
constexpr int TPB = 256;    // threads per update block (16x16, 4x4 outputs each)

// Factor the NB x NB diagonal block at (k0,k0) in place (one block, one thread per row).
// On failure records the smallest failing global diagonal index in *errIdx.
__global__ void factorDiagKernel(double* __restrict__ A, const int n, const int k0,
                                 int* __restrict__ errIdx) {
    __shared__ double s[NB][NB + 1];
    const int i = threadIdx.x;
    const int bs = min(NB, n - k0);
    if (i < bs)
        for (int j = 0; j <= i; ++j) s[i][j] = A[(size_t)(k0 + i) * n + k0 + j];
    __syncthreads();
    for (int j = 0; j < bs; ++j) {
        if (i == j) {
            double sum = 0.0;
            for (int k = 0; k < j; ++k) sum += s[j][k] * s[j][k];
            const double val = s[j][j] - sum;
            if (val <= 0.0) atomicMin(errIdx, k0 + j);
            s[j][j] = sqrt(val);
        }
        __syncthreads();
        if (i > j && i < bs) {
            double sum = 0.0;
            for (int k = 0; k < j; ++k) sum += s[i][k] * s[j][k];
            s[i][j] = (s[i][j] - sum) / s[j][j];
        }
        __syncthreads();
    }
    if (i < bs)
        for (int j = 0; j <= i; ++j) A[(size_t)(k0 + i) * n + k0 + j] = s[i][j];
}

// L21 = A21 * L11^{-T}: one thread per row below the diagonal block.
__global__ void trsmKernel(double* __restrict__ A, const int n, const int k0) {
    __shared__ double l[NB][NB + 1];
    const int bs = min(NB, n - k0);
    for (int idx = threadIdx.x; idx < NB * NB; idx += blockDim.x) {
        const int r = idx / NB, c = idx % NB;
        if (r < bs && c <= r) l[r][c] = A[(size_t)(k0 + r) * n + k0 + c];
    }
    __syncthreads();
    const int row = k0 + bs + blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;
    double* x = A + (size_t)row * n + k0;
    double v[NB];
    for (int j = 0; j < bs; ++j) v[j] = x[j];
    #pragma unroll
    for (int j = 0; j < NB; ++j) {
        if (j < bs) {
            double sum = 0.0;
            #pragma unroll
            for (int k = 0; k < j; ++k) sum += v[k] * l[j][k];
            v[j] = (v[j] - sum) / l[j][j];
        }
    }
    for (int j = 0; j < bs; ++j) x[j] = v[j];
}

// C[i][j] = c0*C[i][j] + sign * sum_k X[i][k]*X[j][k]  for rows/cols in [row0, n).
// X points at element (0, kofs) with leading dimension ldx; C is n x n row-major (ld n).
// lowerOnly: skip tiles above the diagonal. addDiag: add value to diagonal (matrix generation).
__global__ void __launch_bounds__(TPB)
gemmNTKernel(const double* __restrict__ X, const size_t ldx, const int K, double* __restrict__ C,
             const int n, const int row0, const double c0, const double sign,
             const bool lowerOnly, const double addDiag) {
    const int bi = blockIdx.y, bj = blockIdx.x;
    if (lowerOnly && bj > bi) return;
    __shared__ double As[KC][TILE + 1];
    __shared__ double Bs[KC][TILE + 1];
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    const int rBase = row0 + bi * TILE, cBase = row0 + bj * TILE;
    double acc[4][4] = {};
    for (int kb = 0; kb < K; kb += KC) {
        __syncthreads();
        for (int idx = tid; idx < KC * TILE; idx += TPB) {
            const int kk = idx % KC, i = idx / KC;
            const int gk = kb + kk;
            const int r = rBase + i, c = cBase + i;
            As[kk][i] = (gk < K && r < n) ? X[r * ldx + gk] : 0.0;
            Bs[kk][i] = (gk < K && c < n) ? X[c * ldx + gk] : 0.0;
        }
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < KC; ++kk) {
            double a[4], b[4];
            #pragma unroll
            for (int q = 0; q < 4; ++q) { a[q] = As[kk][ty + 16 * q]; b[q] = Bs[kk][tx + 16 * q]; }
            #pragma unroll
            for (int p = 0; p < 4; ++p)
                #pragma unroll
                for (int q = 0; q < 4; ++q) acc[p][q] += a[p] * b[q];
        }
    }
    #pragma unroll
    for (int p = 0; p < 4; ++p) {
        const int r = rBase + ty + 16 * p;
        if (r >= n) continue;
        #pragma unroll
        for (int q = 0; q < 4; ++q) {
            const int c = cBase + tx + 16 * q;
            if (c >= n || (lowerOnly && c > r)) continue;
            double v = sign * acc[p][q];
            if (c0 != 0.0) v += c0 * C[(size_t)r * n + c];
            if (r == c) v += addDiag;
            C[(size_t)r * n + c] = v;
        }
    }
}

__global__ void zeroUpperKernel(double* __restrict__ A, const int n) {
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)n * n) return;
    if (idx % n > idx / n) A[idx] = 0.0;
}

__global__ void errorKernel(const double* __restrict__ R, const double* __restrict__ A,
                            const size_t total, unsigned long long* __restrict__ maxAbs,
                            unsigned long long* __restrict__ maxRel) {
    double ea = 0.0, er = 0.0;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += (size_t)gridDim.x * blockDim.x) {
        const double e = fabs(R[i] - A[i]);
        const double rel = e / (fabs(A[i]) + 1e-10);
        if (!(e <= ea)) ea = e;     // NaN propagates (and fails validation)
        if (!(rel <= er)) er = rel;
    }
    // non-negative doubles order like their bit patterns
    atomicMax(maxAbs, (unsigned long long)__double_as_longlong(ea));
    atomicMax(maxRel, (unsigned long long)__double_as_longlong(er));
}

static void launchGemmNT(const double* X, size_t ldx, int K, double* C, int n, int row0,
                         double c0, double sign, bool lowerOnly, double addDiag) {
    const int tiles = (n - row0 + TILE - 1) / TILE;
    if (tiles <= 0) return;
    dim3 grid(tiles, tiles);
    gemmNTKernel<<<grid, TPB>>>(X, ldx, K, C, n, row0, c0, sign, lowerOnly, addDiag);
}

// dA: device matrix (row-major n x n). Returns false if not positive definite.
bool choleskyDecomposition(double* dA, const size_t n_, int* dErr) {
    const int n = (int)n_;
    const int big = INT_MAX;
    CUDA_CHECK(cudaMemcpy(dErr, &big, sizeof(int), cudaMemcpyHostToDevice));
    for (int k0 = 0; k0 < n; k0 += NB) {
        const int bs = std::min(NB, n - k0);
        factorDiagKernel<<<1, NB>>>(dA, n, k0, dErr);
        const int rest = n - k0 - bs;
        if (rest > 0) {
            trsmKernel<<<(rest + 63) / 64, 64>>>(dA, n, k0);
            launchGemmNT(dA + k0, n, bs, dA, n, k0 + bs, 1.0, -1.0, true, 0.0);
        }
    }
    zeroUpperKernel<<<(unsigned)(((size_t)n * n + 255) / 256), 256>>>(dA, n);
    CUDA_CHECK(cudaGetLastError());
    int err = big;
    CUDA_CHECK(cudaMemcpy(&err, dErr, sizeof(int), cudaMemcpyDeviceToHost));
    if (err != big) {
        printf("Error: Matrix is not positive definite at diagonal element %d\n", err);
        return false;
    }
    return true;
}

// Generate a symmetric positive definite matrix directly on the device
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
    launchGemmNT(dB, n, (int)n, dA, (int)n, 0, 0.0, 1.0, false, (double)n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(dB));
}

bool validateCholesky(const double* dL, const double* dA_orig, const size_t n) {
    // Validate by computing L * L^T on the device and comparing with original
    double* dR;
    CUDA_CHECK(cudaMalloc(&dR, n * n * sizeof(double)));
    launchGemmNT(dL, n, (int)n, dR, (int)n, 0, 0.0, 1.0, false, 0.0);
    unsigned long long* dM;
    CUDA_CHECK(cudaMalloc(&dM, 2 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(dM, 0, 2 * sizeof(unsigned long long)));
    errorKernel<<<1024, 256>>>(dR, dA_orig, n * n, dM, dM + 1);
    CUDA_CHECK(cudaGetLastError());
    unsigned long long h[2];
    CUDA_CHECK(cudaMemcpy(h, dM, sizeof(h), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dR));
    CUDA_CHECK(cudaFree(dM));
    double maxError, relError;
    memcpy(&maxError, &h[0], sizeof(double));
    memcpy(&relError, &h[1], sizeof(double));

    printf("Max absolute error: %.10e\n", maxError);
    printf("Max relative error: %.10e\n", relError);

    // Check if error is within tolerance
    if (!(relError <= 1e-6)) {
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
    
    // Allocate matrix on the device
    double* dA = nullptr;
    double* dA_orig = nullptr;
    int* dErr = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, n * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dErr, sizeof(int)));

    // Generate positive definite matrix
    printf("Generating positive definite matrix...\n");
    generatePositiveDefiniteMatrix(dA, n);

    if (validate) {
        // Save original for validation
        CUDA_CHECK(cudaMalloc(&dA_orig, n * n * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(dA_orig, dA, n * n * sizeof(double), cudaMemcpyDeviceToDevice));
    }

    // Perform Cholesky decomposition
    printf("Computing Cholesky decomposition...\n");
    CUDA_CHECK(cudaDeviceSynchronize());
    auto start = std::chrono::high_resolution_clock::now();

    bool success = choleskyDecomposition(dA, n, dErr);
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
        std::vector<double> A(n * n);
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
