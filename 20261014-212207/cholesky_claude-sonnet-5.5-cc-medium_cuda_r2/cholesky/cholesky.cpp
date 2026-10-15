#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <climits>
#include <cstring>
#include <cuda_runtime.h>
#include <vector>

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

// Blocked right-looking Cholesky decomposition on the GPU (row-major storage).
// Decomposes positive definite matrix A into L * L^T where L is lower triangular.
// Per block step: factor diagonal tile, triangular solve of the panel below it,
// then a symmetric rank-NB update of the trailing matrix (lower tiles only).

constexpr int NB = 64;        // algorithmic block size
constexpr int TILE = 64;      // GEMM output tile
constexpr int KC = 16;        // GEMM k-chunk
constexpr int TRSM_ROWS = 32; // rows per TRSM block
constexpr int TRSM_LD = NB + 1;

// C[i][j] = beta*C[i][j] + alpha * sum_k X[i][k] * Y[j][k]; 64x64 tile / 16x16 threads.
// If LOWER, tiles strictly above the block diagonal are skipped.
template <bool LOWER>
__global__ void __launch_bounds__(256)
gemmNT(int M, int N, int K, const double* __restrict__ X, size_t ldx,
       const double* __restrict__ Y, size_t ldy, double* C, size_t ldc,
       double alpha, double beta) {
    const int bi = blockIdx.y, bj = blockIdx.x;
    if (LOWER && bj > bi) return;
    __shared__ double Xs[KC][TILE + 1];
    __shared__ double Ys[KC][TILE + 1];
    const int tx = threadIdx.x & 15, ty = threadIdx.x >> 4;
    const int row0 = bi * TILE, col0 = bj * TILE;
    double acc[4][4] = {};

    for (int k0 = 0; k0 < K; k0 += KC) {
        // 64 rows x 16 k: thread loads 4 elements, 16 consecutive threads cover one row
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int idx = threadIdx.x + r * 256;
            const int kk = idx & (KC - 1), rr = idx >> 4;
            const int gk = k0 + kk;
            const int gi = row0 + rr, gj = col0 + rr;
            Xs[kk][rr] = (gi < M && gk < K) ? X[(size_t)gi * ldx + gk] : 0.0;
            Ys[kk][rr] = (gj < N && gk < K) ? Y[(size_t)gj * ldy + gk] : 0.0;
        }
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < KC; ++kk) {
            double a[4], b[4];
            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                a[r] = Xs[kk][ty + 16 * r];
                b[r] = Ys[kk][tx + 16 * r];
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
        const int gi = row0 + ty + 16 * r;
        if (gi >= M) continue;
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            const int gj = col0 + tx + 16 * c;
            if (gj >= N) continue;
            double* p = C + (size_t)gi * ldc + gj;
            *p = (beta == 0.0 ? 0.0 : beta * *p) + alpha * acc[r][c];
        }
    }
}

// Factor the b x b diagonal tile at (k,k) in shared memory.
__global__ void potrfDiag(double* A, int n, int k, int b, int* failIdx) {
    __shared__ double S[NB][NB + 1];
    const int tid = threadIdx.x, nt = blockDim.x;
    for (int e = tid; e < b * b; e += nt) {
        const int i = e / b, j = e % b;
        S[i][j] = A[(size_t)(k + i) * n + k + j];
    }
    __syncthreads();
    for (int j = 0; j < b; ++j) {
        if (tid == 0) {
            double v = S[j][j];
            if (!(v > 0.0)) {
                atomicMin(failIdx, k + j);
                v = 1.0;
            }
            S[j][j] = sqrt(v);
        }
        __syncthreads();
        const double d = S[j][j];
        for (int i = j + 1 + tid; i < b; i += nt) S[i][j] /= d;
        __syncthreads();
        const int r = b - j - 1;
        for (int e = tid; e < r * r; e += nt) {
            const int ii = j + 1 + e / r, jj = j + 1 + e % r;
            if (jj <= ii) S[ii][jj] -= S[ii][j] * S[jj][j];
        }
        __syncthreads();
    }
    for (int e = tid; e < b * b; e += nt) {
        const int i = e / b, j = e % b;
        if (j <= i) A[(size_t)(k + i) * n + k + j] = S[i][j];
    }
}

// Panel rows i: solve x * L11^T = a  (one thread per row, data staged in shared memory).
__global__ void __launch_bounds__(TRSM_ROWS)
trsmPanel(double* A, int n, int k, int b) {
    extern __shared__ double sm[];
    double* Ls = sm;                 // NB x NB
    double* Xs = sm + NB * NB;       // TRSM_ROWS x TRSM_LD
    const int t = threadIdx.x;
    const int i0 = k + b + blockIdx.x * TRSM_ROWS;
    const int rows = min(TRSM_ROWS, n - i0);
    for (int e = t; e < b * b; e += TRSM_ROWS) {
        const int i = e / b, j = e % b;
        Ls[i * NB + j] = (j <= i) ? A[(size_t)(k + i) * n + k + j] : 0.0;
    }
    for (int r = 0; r < rows; ++r)
        for (int c = t; c < b; c += TRSM_ROWS)
            Xs[r * TRSM_LD + c] = A[(size_t)(i0 + r) * n + k + c];
    __syncthreads();
    if (t < rows) {
        double* x = Xs + t * TRSM_LD;
        for (int j = 0; j < b; ++j) {
            double s = x[j];
            for (int kk = 0; kk < j; ++kk) s -= x[kk] * Ls[j * NB + kk];
            x[j] = s / Ls[j * NB + j];
        }
    }
    __syncthreads();
    for (int r = 0; r < rows; ++r)
        for (int c = t; c < b; c += TRSM_ROWS)
            A[(size_t)(i0 + r) * n + k + c] = Xs[r * TRSM_LD + c];
}

__global__ void zeroUpper(double* A, size_t n) {
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * n) return;
    if (idx % n > idx / n) A[idx] = 0.0;
}

__global__ void addDiag(double* A, size_t n, double v) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) A[i * n + i] += v;
}

// Max abs / relative error between two matrices (non-negative doubles: bit order == value order).
__global__ void compareMax(const double* R, const double* O, size_t total,
                           unsigned long long* maxAbs, unsigned long long* maxRel) {
    double ma = 0.0, mr = 0.0;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += (size_t)gridDim.x * blockDim.x) {
        const double err = fabs(R[i] - O[i]);
        ma = fmax(ma, err);
        mr = fmax(mr, err / (fabs(O[i]) + 1e-10));
    }
    for (int off = 16; off > 0; off >>= 1) {
        ma = fmax(ma, __shfl_down_sync(0xffffffffu, ma, off));
        mr = fmax(mr, __shfl_down_sync(0xffffffffu, mr, off));
    }
    if ((threadIdx.x & 31) == 0) {
        atomicMax(maxAbs, (unsigned long long)__double_as_longlong(ma));
        atomicMax(maxRel, (unsigned long long)__double_as_longlong(mr));
    }
}

static void launchGemm(bool lower, int M, int N, int K, const double* X, size_t ldx,
                       const double* Y, size_t ldy, double* C, size_t ldc, double alpha,
                       double beta) {
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    if (lower)
        gemmNT<true><<<grid, 256>>>(M, N, K, X, ldx, Y, ldy, C, ldc, alpha, beta);
    else
        gemmNT<false><<<grid, 256>>>(M, N, K, X, ldx, Y, ldy, C, ldc, alpha, beta);
}

// dA: device matrix (row-major, n x n). Returns false if not positive definite.
bool choleskyDecomposition(double* dA, const size_t n) {
    int* dFail;
    CUDA_CHECK(cudaMalloc(&dFail, sizeof(int)));
    const int none = INT_MAX;
    CUDA_CHECK(cudaMemcpy(dFail, &none, sizeof(int), cudaMemcpyHostToDevice));

    const size_t trsmSmem = (NB * NB + TRSM_ROWS * TRSM_LD) * sizeof(double);
    CUDA_CHECK(cudaFuncSetAttribute(trsmPanel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    (int)trsmSmem));
    const int N = (int)n;
    for (int k = 0; k < N; k += NB) {
        const int b = std::min(NB, N - k);
        potrfDiag<<<1, 256>>>(dA, N, k, b, dFail);
        const int m = N - k - b;
        if (m <= 0) break;
        trsmPanel<<<(m + TRSM_ROWS - 1) / TRSM_ROWS, TRSM_ROWS, trsmSmem>>>(dA, N, k, b);
        const double* P = dA + (size_t)(k + b) * N + k;
        launchGemm(true, m, m, b, P, N, P, N, dA + (size_t)(k + b) * N + k + b, N, -1.0, 1.0);
    }
    zeroUpper<<<(unsigned)((n * n + 255) / 256), 256>>>(dA, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    int fail;
    CUDA_CHECK(cudaMemcpy(&fail, dFail, sizeof(int), cudaMemcpyDeviceToHost));
    cudaFree(dFail);
    if (fail != none) {
        printf("Error: Matrix is not positive definite at diagonal element %d\n", fail);
        return false;
    }
    return true;
}

// Generate a symmetric positive definite matrix directly on the device
void generatePositiveDefiniteMatrix(double* dA, const size_t n) {
    // Method: Create A = B * B^T where B is random
    // Then add n to the diagonal to make it strictly positive definite
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }
    double* dB;
    CUDA_CHECK(cudaMalloc(&dB, n * n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(dB, B.data(), n * n * sizeof(double), cudaMemcpyHostToDevice));
    launchGemm(false, (int)n, (int)n, (int)n, dB, n, dB, n, dA, n, 1.0, 0.0);
    addDiag<<<(unsigned)((n + 255) / 256), 256>>>(dA, n, (double)n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaFree(dB);
}

bool validateCholesky(const double* dL, const double* dA_orig, const size_t n) {
    // Validate by computing L * L^T on the device and comparing with original
    double* dR;
    unsigned long long* dErr;
    CUDA_CHECK(cudaMalloc(&dR, n * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dErr, 2 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(dErr, 0, 2 * sizeof(unsigned long long)));
    launchGemm(false, (int)n, (int)n, (int)n, dL, n, dL, n, dR, n, 1.0, 0.0);
    compareMax<<<1024, 256>>>(dR, dA_orig, n * n, dErr, dErr + 1);
    CUDA_CHECK(cudaGetLastError());
    unsigned long long e[2];
    CUDA_CHECK(cudaMemcpy(e, dErr, sizeof(e), cudaMemcpyDeviceToHost));
    cudaFree(dR);
    cudaFree(dErr);
    double maxError, relError;
    memcpy(&maxError, &e[0], sizeof(double));
    memcpy(&relError, &e[1], sizeof(double));

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
    
    // Allocate matrix
    std::vector<double> A;
    double *dA = nullptr, *dA_orig = nullptr;
    CUDA_CHECK(cudaFree(0));
    CUDA_CHECK(cudaMalloc(&dA, n * n * sizeof(double)));
    
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
    auto start = std::chrono::high_resolution_clock::now();
    
    bool success = choleskyDecomposition(dA, n);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (success && (printResults)) {
        A.resize(n * n);
        CUDA_CHECK(cudaMemcpy(A.data(), dA, n * n * sizeof(double), cudaMemcpyDeviceToHost));
    }

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
