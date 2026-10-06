#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// Cholesky decomposition (GPU, blocked right-looking algorithm, CUDA)
// Decomposes positive definite matrix A into L * L^T where L is lower triangular

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), \
                    __FILE__, __LINE__);                                          \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

constexpr int NB = 64;        // Cholesky block size
constexpr int GT = 64;        // GEMM tile size (GT x GT per thread block)
constexpr int GK = 16;        // GEMM k-slice
constexpr int TRSM_ROWS = 32; // rows per TRSM thread block (one row per thread)

// C (=|-=) X * Y^T on an M x N tile region, K inner dimension, all row-major.
// If lowerOnly, tiles strictly above the tile diagonal are skipped.
template <bool SUBTRACT>
__global__ void __launch_bounds__(256)
gemmNT(const double* __restrict__ X, const double* __restrict__ Y, size_t ldxy,
       double* __restrict__ C, size_t ldc, int M, int N, int K, bool lowerOnly,
       const int* __restrict__ fail) {
    if (fail && *fail) return;
    const int bm = blockIdx.y, bn = blockIdx.x;
    if (lowerOnly && bn > bm) return;

    __shared__ double Xs[GK][GT + 1];
    __shared__ double Ys[GK][GT + 1];

    const int t = threadIdx.x;
    const int tx = t % 16, ty = t / 16;
    const int lc = t % GK, lr = t / GK;  // load mapping
    const int rowBase = bm * GT, colBase = bn * GT;

    double acc[4][4] = {};

    for (int k0 = 0; k0 < K; k0 += GK) {
        const int gk = k0 + lc;
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const int r = lr + 16 * q;
            const int gx = rowBase + r, gy = colBase + r;
            Xs[lc][r] = (gx < M && gk < K) ? X[(size_t)gx * ldxy + gk] : 0.0;
            Ys[lc][r] = (gy < N && gk < K) ? Y[(size_t)gy * ldxy + gk] : 0.0;
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < GK; ++kk) {
            double a[4], b[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) a[i] = Xs[kk][ty + 16 * i];
#pragma unroll
            for (int j = 0; j < 4; ++j) b[j] = Ys[kk][tx + 16 * j];
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] = fma(a[i], b[j], acc[i][j]);
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int r = rowBase + ty + 16 * i;
        if (r >= M) continue;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int c = colBase + tx + 16 * j;
            if (c >= N) continue;
            double* p = &C[(size_t)r * ldc + c];
            if (SUBTRACT) *p -= acc[i][j];
            else *p = acc[i][j];
        }
    }
}

// Factor the nb x nb diagonal block starting at (k, k) in place (single thread block).
__global__ void __launch_bounds__(256)
potrfDiag(double* __restrict__ A, size_t n, int k, int nb, int* __restrict__ fail) {
    if (*fail) return;
    __shared__ double s[NB][NB + 1];
    __shared__ int bad;
    const int t = threadIdx.x;
    if (t == 0) bad = 0;
    for (int p = t; p < nb * nb; p += blockDim.x) {
        const int i = p / nb, c = p % nb;
        if (c <= i) s[i][c] = A[(size_t)(k + i) * n + k + c];
    }
    __syncthreads();

    for (int j = 0; j < nb; ++j) {
        if (t == 0) {
            const double val = s[j][j];
            if (val <= 0.0) {
                bad = 1;
                *fail = k + j + 1;
            } else {
                s[j][j] = sqrt(val);
            }
        }
        __syncthreads();
        if (bad) return;
        const double d = s[j][j];
        for (int i = j + 1 + t; i < nb; i += blockDim.x) s[i][j] /= d;
        __syncthreads();
        const int m = nb - j - 1;
        for (int p = t; p < m * m; p += blockDim.x) {
            const int i = j + 1 + p / m, c = j + 1 + p % m;
            if (c <= i) s[i][c] -= s[i][j] * s[c][j];
        }
        __syncthreads();
    }

    for (int p = t; p < nb * nb; p += blockDim.x) {
        const int i = p / nb, c = p % nb;
        A[(size_t)(k + i) * n + k + c] = (c <= i) ? s[i][c] : 0.0;
    }
}

// Solve the panel below the diagonal block: rows [k+nb, n), columns [k, k+nb):
// X <- X * L_kk^{-T}. One thread per row, row kept in registers.
__global__ void __launch_bounds__(TRSM_ROWS)
trsmPanel(double* __restrict__ A, size_t n, int k, int nb, int* __restrict__ fail) {
    if (*fail) return;
    __shared__ double L[NB][NB + 1];
    const int t = threadIdx.x;
    for (int p = t; p < NB * NB; p += blockDim.x) {
        const int i = p / NB, c = p % NB;
        double v;
        if (i < nb && c < nb) v = (c <= i) ? A[(size_t)(k + i) * n + k + c] : 0.0;
        else v = (i == c) ? 1.0 : 0.0;  // identity padding for partial blocks
        L[i][c] = v;
    }
    __syncthreads();

    const size_t row = (size_t)k + nb + (size_t)blockIdx.x * TRSM_ROWS + t;
    if (row >= n) return;
    double* a = &A[row * n + k];
    double x[NB];
#pragma unroll
    for (int c = 0; c < NB; ++c) x[c] = (c < nb) ? a[c] : 0.0;
#pragma unroll
    for (int j = 0; j < NB; ++j) {
        double sum = 0.0;
#pragma unroll
        for (int p = 0; p < j; ++p) sum = fma(x[p], L[j][p], sum);
        x[j] = (x[j] - sum) / L[j][j];
    }
#pragma unroll
    for (int c = 0; c < NB; ++c)
        if (c < nb) a[c] = x[c];
}

// Zero the strictly upper part of rows [r0, r1) from column c0 onwards (c > row).
__global__ void zeroUpperRows(double* __restrict__ A, size_t n, int r0, int r1) {
    const int row = r0 + blockIdx.y;
    if (row >= r1) return;
    for (size_t c = (size_t)row + 1 + blockIdx.x * blockDim.x + threadIdx.x; c < n;
         c += (size_t)gridDim.x * blockDim.x)
        A[(size_t)row * n + c] = 0.0;
}

static inline unsigned ceilDiv(size_t a, size_t b) { return (unsigned)((a + b - 1) / b); }

static void launchGemm(bool subtract, const double* X, const double* Y, size_t ld, double* C,
                       size_t ldc, int M, int N, int K, bool lowerOnly, const int* fail,
                       cudaStream_t s) {
    dim3 grid(ceilDiv(N, GT), ceilDiv(M, GT));
    if (subtract) gemmNT<true><<<grid, 256, 0, s>>>(X, Y, ld, C, ldc, M, N, K, lowerOnly, fail);
    else gemmNT<false><<<grid, 256, 0, s>>>(X, Y, ld, C, ldc, M, N, K, lowerOnly, fail);
    CUDA_CHECK(cudaGetLastError());
}

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    if (n == 0) return true;
    const size_t bytes = n * n * sizeof(double);

    double* dA;
    int* dFail;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dFail, sizeof(int)));
    CUDA_CHECK(cudaMemset(dFail, 0, sizeof(int)));

    cudaStream_t sMain, sLook, sCopy;
    CUDA_CHECK(cudaStreamCreateWithFlags(&sMain, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&sLook, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&sCopy, cudaStreamNonBlocking));
    const size_t nBlocks = (n + NB - 1) / NB;
    std::vector<cudaEvent_t> rowDone(nBlocks);
    for (auto& e : rowDone) CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
    cudaEvent_t panelReady, mainDone;
    CUDA_CHECK(cudaEventCreateWithFlags(&panelReady, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&mainDone, cudaEventDisableTiming));

    CUDA_CHECK(cudaMemcpyAsync(dA, A.data(), bytes, cudaMemcpyHostToDevice, sMain));

    // Factor first panel
    {
        const int nb = (int)std::min<size_t>(NB, n);
        potrfDiag<<<1, 256, 0, sMain>>>(dA, n, 0, nb, dFail);
        if (n > (size_t)nb)
            trsmPanel<<<ceilDiv(n - nb, TRSM_ROWS), TRSM_ROWS, 0, sMain>>>(dA, n, 0, nb, dFail);
        CUDA_CHECK(cudaGetLastError());
    }

    // Right-looking with one-panel lookahead: at step b, panel b is already factored.
    // The next panel column is updated and factored on sLook while the remaining
    // trailing matrix is updated on sMain. Finished row blocks are streamed back on sCopy.
    for (size_t b = 0; b < nBlocks; ++b) {
        const size_t k = b * NB;
        const int nb = (int)std::min<size_t>(NB, n - k);
        const size_t t0 = k + nb;  // trailing start
        const size_t m = n - t0;   // trailing size

        // Row block b is final now: zero its upper part and send it back.
        {
            dim3 g(ceilDiv(n, 256 * 4), nb);
            zeroUpperRows<<<g, 256, 0, sMain>>>(dA, n, (int)k, (int)t0);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(rowDone[b], sMain));
            CUDA_CHECK(cudaStreamWaitEvent(sCopy, rowDone[b], 0));
            CUDA_CHECK(cudaMemcpyAsync(A.data() + k * n, dA + k * n, (size_t)nb * n * sizeof(double),
                                       cudaMemcpyDeviceToHost, sCopy));
        }
        if (m == 0) break;

        const double* P = dA + t0 * n + k;  // factored panel rows [t0, n), cols [k, k+nb)
        const int nb2 = (int)std::min<size_t>(NB, m);
        const size_t t1 = t0 + nb2;

        // Lookahead: update block column [t0, t0+nb2) for all rows >= t0, then factor it.
        CUDA_CHECK(cudaEventRecord(mainDone, sMain));
        CUDA_CHECK(cudaStreamWaitEvent(sLook, mainDone, 0));
        launchGemm(true, P, P, n, dA + t0 * n + t0, n, (int)m, nb2, nb, true, dFail, sLook);
        potrfDiag<<<1, 256, 0, sLook>>>(dA, n, (int)t0, nb2, dFail);
        if (n > t1)
            trsmPanel<<<ceilDiv(n - t1, TRSM_ROWS), TRSM_ROWS, 0, sLook>>>(dA, n, (int)t0, nb2, dFail);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(panelReady, sLook));

        // Remaining trailing update: rows/cols [t1, n).
        if (n > t1) {
            const int mr = (int)(n - t1);
            const double* P1 = dA + t1 * n + k;
            launchGemm(true, P1, P1, n, dA + t1 * n + t1, n, mr, mr, nb, true, dFail, sMain);
        }
        CUDA_CHECK(cudaStreamWaitEvent(sMain, panelReady, 0));
    }

    CUDA_CHECK(cudaStreamSynchronize(sMain));
    CUDA_CHECK(cudaStreamSynchronize(sCopy));
    int fail = 0;
    CUDA_CHECK(cudaMemcpy(&fail, dFail, sizeof(int), cudaMemcpyDeviceToHost));

    for (auto& e : rowDone) CUDA_CHECK(cudaEventDestroy(e));
    CUDA_CHECK(cudaEventDestroy(panelReady));
    CUDA_CHECK(cudaEventDestroy(mainDone));
    CUDA_CHECK(cudaStreamDestroy(sMain));
    CUDA_CHECK(cudaStreamDestroy(sLook));
    CUDA_CHECK(cudaStreamDestroy(sCopy));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dFail));

    if (fail) {
        printf("Error: Matrix is not positive definite at diagonal element %zu\n", (size_t)(fail - 1));
        return false;
    }
    return true;
}

__global__ void addToDiagonal(double* __restrict__ A, size_t n, double v) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) A[i * n + i] += v;
}

// Generate a symmetric positive definite matrix
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    // Method: Create A = B * B^T where B is random
    // This guarantees A is positive semi-definite
    // Then add identity to make it strictly positive definite
    if (n == 0) return;

    std::vector<double> B(n * n);
    unsigned int seed = 42;

    // Generate random matrix B (sequential RNG stream)
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

    // Compute A = B * B^T on the GPU, then add diagonal dominance
    const size_t bytes = n * n * sizeof(double);
    double *dB, *dA;
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMemcpy(dB, B.data(), bytes, cudaMemcpyHostToDevice));
    launchGemm(false, dB, dB, n, dA, n, (int)n, (int)n, (int)n, false, nullptr, 0);
    addToDiagonal<<<ceilDiv(n, 256), 256>>>(dA, n, (double)n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(A.data(), dA, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dA));
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T (on the GPU) and comparing with original matrix

    std::vector<double> reconstructed(n * n);

    if (n > 0) {
        const size_t bytes = n * n * sizeof(double);
        double *dL, *dR;
        CUDA_CHECK(cudaMalloc(&dL, bytes));
        CUDA_CHECK(cudaMalloc(&dR, bytes));
        CUDA_CHECK(cudaMemcpy(dL, L.data(), bytes, cudaMemcpyHostToDevice));
        launchGemm(false, dL, dL, n, dR, n, (int)n, (int)n, (int)n, false, nullptr, 0);
        CUDA_CHECK(cudaMemcpy(reconstructed.data(), dR, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(dL));
        CUDA_CHECK(cudaFree(dR));
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
    
    // Pin host buffer so transfers can overlap with GPU computation
    if (n > 0) CUDA_CHECK(cudaHostRegister(A.data(), n * n * sizeof(double), cudaHostRegisterDefault));

    // Perform Cholesky decomposition
    printf("Computing Cholesky decomposition...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    bool success = choleskyDecomposition(A, n);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (n > 0) CUDA_CHECK(cudaHostUnregister(A.data()));

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
