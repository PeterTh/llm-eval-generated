#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// CUDA blocked (right-looking) Cholesky decomposition.
// Decomposes positive definite matrix A into L * L^T where L is lower triangular.
// All O(n^3) work (diagonal block factorization, triangular solve of the panel,
// symmetric rank-NB update of the trailing matrix) runs on the GPU with
// hand-written kernels; no external BLAS/LAPACK libraries are used.

#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        cudaError_t err__ = (call);                                                      \
        if (err__ != cudaSuccess) {                                                      \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__,   \
                    cudaGetErrorString(err__));                                          \
            exit(1);                                                                     \
        }                                                                                \
    } while (0)

// Algorithmic block size (panel width). Also the output tile size of the GEMM kernel.
constexpr int NB = 64;
// Shared-memory padding to avoid bank conflicts on column accesses.
constexpr int NBP = NB + 1;
// GEMM tiling parameters: 64x64 output tile, 16-wide K slices, 16x16 threads, 4x4 per thread.
constexpr int GEMM_TILE = 64;
constexpr int GEMM_K = 16;
constexpr int GEMM_THREADS_X = 16;
constexpr int GEMM_THREADS_Y = 16;
constexpr int GEMM_THREADS = GEMM_THREADS_X * GEMM_THREADS_Y;
constexpr int GEMM_TM = GEMM_TILE / GEMM_THREADS_Y;  // 4 rows per thread
constexpr int GEMM_TN = GEMM_TILE / GEMM_THREADS_X;  // 4 cols per thread
constexpr int POTRF_THREADS = 256;
constexpr int TRSM_THREADS = 256;
constexpr size_t TRSM_SMEM_BYTES = 2 * NB * NBP * sizeof(double);

// ---------------------------------------------------------------------------
// Kernel 1: unblocked Cholesky of the nb x nb diagonal block A[k:k+nb, k:k+nb]
// (nb <= NB). One thread block, computation done entirely in shared memory.
// On a non-positive-definite pivot, records the (global) failing diagonal index
// in *fail (first failure wins because blocks are processed in order).
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(POTRF_THREADS)
potrf_diag_kernel(double* __restrict__ A, const size_t n, const int k, const int nb,
                  int* __restrict__ fail) {
    __shared__ double s[NB][NBP];
    __shared__ int s_fail;

    const int tid = threadIdx.x;
    if (tid == 0) s_fail = *fail;
    __syncthreads();
    if (s_fail >= 0) return;  // an earlier block already failed; nothing meaningful to do

    double* blk = A + (size_t)k * n + k;

    // Load lower triangle (plus diagonal) of the block.
    for (int idx = tid; idx < nb * nb; idx += POTRF_THREADS) {
        const int i = idx / nb;
        const int j = idx - i * nb;
        if (j <= i) s[i][j] = blk[(size_t)i * n + j];
    }
    __syncthreads();

    for (int j = 0; j < nb; ++j) {
        // Pivot
        if (tid == 0) {
            const double val = s[j][j];
            if (val <= 0.0) {
                s_fail = k + j;
            } else {
                s[j][j] = sqrt(val);
            }
        }
        __syncthreads();
        if (s_fail >= 0) break;

        // Scale column j below the diagonal.
        const double inv = 1.0 / s[j][j];
        for (int i = j + 1 + tid; i < nb; i += POTRF_THREADS) {
            s[i][j] *= inv;
        }
        __syncthreads();

        // Rank-1 update of the trailing lower triangle: s[i][c] -= s[i][j]*s[c][j], i >= c > j
        const int rem = nb - j - 1;
        const int tri = rem * (rem + 1) / 2;
        for (int t = tid; t < tri; t += POTRF_THREADS) {
            // Map linear triangular index t to (r, c) with 0 <= c <= r < rem
            int r = (int)((sqrtf(8.0f * (float)t + 1.0f) - 1.0f) * 0.5f);
            while (r * (r + 1) / 2 > t) --r;
            while ((r + 1) * (r + 2) / 2 <= t) ++r;
            const int c = t - r * (r + 1) / 2;
            const int gi = j + 1 + r;
            const int gc = j + 1 + c;
            s[gi][gc] -= s[gi][j] * s[gc][j];
        }
        __syncthreads();
    }

    if (tid == 0 && s_fail >= 0) *fail = s_fail;
    if (s_fail >= 0) return;

    // Store the factorized lower triangle back.
    for (int idx = tid; idx < nb * nb; idx += POTRF_THREADS) {
        const int i = idx / nb;
        const int j = idx - i * nb;
        if (j <= i) blk[(size_t)i * n + j] = s[i][j];
    }
}

// ---------------------------------------------------------------------------
// Kernel 2: triangular solve of the panel below the diagonal block.
//   L21 = A21 * L11^{-T}   (A21 is m x nb, L11 is nb x nb lower triangular)
// Each thread block handles up to NB rows of the panel: L11 and the row tile
// are staged in shared memory; thread t solves row t forward-substitution style.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(TRSM_THREADS)
trsm_panel_kernel(double* __restrict__ A, const size_t n, const int k, const int nb,
                  const int m, const int* __restrict__ fail) {
    // Dynamic shared memory (2 * NB * NBP doubles, > 48 KB so it is opted in by the host).
    extern __shared__ double trsm_smem[];
    double (*L)[NBP] = reinterpret_cast<double (*)[NBP]>(trsm_smem);
    double (*X)[NBP] = reinterpret_cast<double (*)[NBP]>(trsm_smem + NB * NBP);

    if (*fail >= 0) return;

    const int tid = threadIdx.x;
    const int row0 = k + nb + blockIdx.x * NB;
    const int rows = min(NB, k + nb + m - row0);

    const double* L11 = A + (size_t)k * n + k;
    double* A21 = A + (size_t)row0 * n + k;

    // Stage L11 (lower triangle incl. diagonal) and the A21 tile.
    for (int idx = tid; idx < nb * nb; idx += TRSM_THREADS) {
        const int i = idx / nb;
        const int j = idx - i * nb;
        if (j <= i) L[i][j] = L11[(size_t)i * n + j];
    }
    for (int idx = tid; idx < rows * nb; idx += TRSM_THREADS) {
        const int i = idx / nb;
        const int j = idx - i * nb;
        X[i][j] = A21[(size_t)i * n + j];
    }
    __syncthreads();

    // Solve: for each row r, x_j = (a_j - sum_{c<j} L[j][c] * x_c) / L[j][j]
    // Rows are independent; L accesses are warp-uniform broadcasts.
    if (tid < rows) {
        for (int j = 0; j < nb; ++j) {
            double sum = X[tid][j];
            for (int c = 0; c < j; ++c) {
                sum -= L[j][c] * X[tid][c];
            }
            X[tid][j] = sum / L[j][j];
        }
    }
    __syncthreads();

    for (int idx = tid; idx < rows * nb; idx += TRSM_THREADS) {
        const int i = idx / nb;
        const int j = idx - i * nb;
        A21[(size_t)i * n + j] = X[i][j];
    }
}

// ---------------------------------------------------------------------------
// Kernel 3: tiled double-precision "NT" matrix multiply
//   C[i][j] = beta * C[i][j] + alpha * sum_k Arow[i][k] * Brow[j][k]
// with A (M x K, lda), B (N x K, ldb), C (M x N, ldc), all row-major.
// If LOWER_ONLY, only tiles/elements with i >= j are computed, and the grid is a
// 1-D enumeration of the lower-triangular tile set (requires M == N). With
// LOWER_ONLY and col_tiles == 1, only the first tile column (bj == 0) is
// enumerated (grid = number of row tiles); this is the lookahead panel update.
// Used for the trailing update (SYRK, alpha=-1, beta=1, K = NB) and for the
// full products in matrix generation / validation (alpha=1, beta=0, K = n).
// ---------------------------------------------------------------------------
template <bool LOWER_ONLY>
__global__ void __launch_bounds__(GEMM_THREADS)
gemm_nt_kernel(const double* __restrict__ Am, const size_t lda,
               const double* __restrict__ Bm, const size_t ldb,
               double* __restrict__ C, const size_t ldc,
               const int M, const int N, const int K,
               const double alpha, const double beta,
               const int col_tiles, const int* __restrict__ fail) {
    if (fail != nullptr && *fail >= 0) return;

    __shared__ double As[GEMM_K][GEMM_TILE + 1];  // As[kk][i]
    __shared__ double Bs[GEMM_K][GEMM_TILE + 1];  // Bs[kk][j]

    int bi, bj;
    if (LOWER_ONLY && col_tiles == 1) {
        bi = blockIdx.x;
        bj = 0;
    } else if (LOWER_ONLY) {
        // Map linear block index to lower-triangular tile (bi >= bj).
        const int t = blockIdx.x;
        int r = (int)((sqrt(8.0 * (double)t + 1.0) - 1.0) * 0.5);
        while (r * (r + 1) / 2 > t) --r;
        while ((r + 1) * (r + 2) / 2 <= t) ++r;
        bi = r;
        bj = t - r * (r + 1) / 2;
    } else {
        bi = blockIdx.y;
        bj = blockIdx.x;
    }

    const int tx = threadIdx.x;  // 0..15
    const int ty = threadIdx.y;  // 0..15
    const int tid = ty * GEMM_THREADS_X + tx;

    const int i0 = bi * GEMM_TILE;
    const int j0 = bj * GEMM_TILE;

    double acc[GEMM_TM][GEMM_TN];
#pragma unroll
    for (int a = 0; a < GEMM_TM; ++a)
#pragma unroll
        for (int b = 0; b < GEMM_TN; ++b) acc[a][b] = 0.0;

    // Each thread loads 4 elements of As and 4 of Bs per K-slice:
    // 64 rows x 16 k = 1024 elements / 256 threads. Consecutive threads read
    // consecutive k (contiguous in memory for row-major A/B).
    const int lk = tid & (GEMM_K - 1);   // 0..15
    const int lr = tid >> 4;             // 0..15

    for (int k0 = 0; k0 < K; k0 += GEMM_K) {
#pragma unroll
        for (int q = 0; q < GEMM_TILE / 16; ++q) {
            const int r = lr + q * 16;
            const int kk = k0 + lk;
            const int gi = i0 + r;
            const int gj = j0 + r;
            As[lk][r] = (gi < M && kk < K) ? Am[(size_t)gi * lda + kk] : 0.0;
            Bs[lk][r] = (gj < N && kk < K) ? Bm[(size_t)gj * ldb + kk] : 0.0;
        }
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < GEMM_K; ++kk) {
            double ra[GEMM_TM], rb[GEMM_TN];
#pragma unroll
            for (int a = 0; a < GEMM_TM; ++a) ra[a] = As[kk][ty + a * GEMM_THREADS_Y];
#pragma unroll
            for (int b = 0; b < GEMM_TN; ++b) rb[b] = Bs[kk][tx + b * GEMM_THREADS_X];
#pragma unroll
            for (int a = 0; a < GEMM_TM; ++a)
#pragma unroll
                for (int b = 0; b < GEMM_TN; ++b) acc[a][b] = fma(ra[a], rb[b], acc[a][b]);
        }
        __syncthreads();
    }

#pragma unroll
    for (int a = 0; a < GEMM_TM; ++a) {
        const int gi = i0 + ty + a * GEMM_THREADS_Y;
        if (gi >= M) continue;
#pragma unroll
        for (int b = 0; b < GEMM_TN; ++b) {
            const int gj = j0 + tx + b * GEMM_THREADS_X;
            if (gj >= N) continue;
            if (LOWER_ONLY && gj > gi) continue;
            double* c = C + (size_t)gi * ldc + gj;
            *c = (beta == 0.0) ? alpha * acc[a][b] : beta * (*c) + alpha * acc[a][b];
        }
    }
}

// Zero the strictly upper triangular part of an n x n row-major matrix.
__global__ void zero_upper_kernel(double* __restrict__ A, const size_t n) {
    const size_t j = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t i = blockIdx.y;
    if (i < n && j < n && j > i) A[i * n + j] = 0.0;
}

// Add a constant to the diagonal.
__global__ void add_diag_kernel(double* __restrict__ A, const size_t n, const double v) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) A[i * n + i] += v;
}

// Elementwise error reduction: max |R - A| and max |R - A| / (|A| + 1e-10).
__global__ void error_reduce_kernel(const double* __restrict__ R, const double* __restrict__ A,
                                    const size_t count, double* __restrict__ out_abs,
                                    double* __restrict__ out_rel) {
    __shared__ double s_abs[256];
    __shared__ double s_rel[256];
    double mabs = 0.0, mrel = 0.0;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += (size_t)gridDim.x * blockDim.x) {
        const double e = fabs(R[i] - A[i]);
        mabs = fmax(mabs, e);
        mrel = fmax(mrel, e / (fabs(A[i]) + 1e-10));
    }
    s_abs[threadIdx.x] = mabs;
    s_rel[threadIdx.x] = mrel;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (threadIdx.x < (unsigned)st) {
            s_abs[threadIdx.x] = fmax(s_abs[threadIdx.x], s_abs[threadIdx.x + st]);
            s_rel[threadIdx.x] = fmax(s_rel[threadIdx.x], s_rel[threadIdx.x + st]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out_abs[blockIdx.x] = s_abs[0];
        out_rel[blockIdx.x] = s_rel[0];
    }
}

// ---------------------------------------------------------------------------
// Host-side helpers
// ---------------------------------------------------------------------------

// Full product C = A * B^T (A: M x K, B: N x K, row-major, contiguous), on device.
static void device_gemm_nt_full(const double* dA, const double* dB, double* dC,
                                int M, int N, int K, cudaStream_t stream) {
    if (M <= 0 || N <= 0) return;
    dim3 block(GEMM_THREADS_X, GEMM_THREADS_Y);
    dim3 grid((N + GEMM_TILE - 1) / GEMM_TILE, (M + GEMM_TILE - 1) / GEMM_TILE);
    gemm_nt_kernel<false><<<grid, block, 0, stream>>>(dA, (size_t)K, dB, (size_t)K, dC, (size_t)N,
                                                      M, N, K, 1.0, 0.0, 0, nullptr);
    CUDA_CHECK(cudaGetLastError());
}

// Symmetric rank-nb update of the lower triangle of an m x m trailing block:
//   A22 -= L21 * L21^T,  L21 is m x nb with leading dimension n.
// If panel_only, only the first NB-wide column block of A22 is updated.
static void launch_syrk_lower(const double* L21, double* A22, const size_t n, const int m,
                              const int nb, const bool panel_only, const int* dFail,
                              cudaStream_t stream) {
    if (m <= 0) return;
    const int tiles = (m + GEMM_TILE - 1) / GEMM_TILE;
    const int col_tiles = panel_only ? 1 : tiles;
    const int nblocks = panel_only ? tiles : tiles * (tiles + 1) / 2;
    dim3 block(GEMM_THREADS_X, GEMM_THREADS_Y);
    gemm_nt_kernel<true><<<nblocks, block, 0, stream>>>(L21, n, L21, n, A22, n, m, m, nb, -1.0,
                                                        1.0, col_tiles, dFail);
    CUDA_CHECK(cudaGetLastError());
}

// Blocked right-looking Cholesky on the device with one-step lookahead.
// A is n x n row-major on the GPU. Returns the index of the first failing
// diagonal element, or -1 on success.
//
// Stream s1 runs the critical path: potrf(k) -> trsm(k) -> update of the next
// column block only. Stream s2 runs the (large) update of the remaining
// trailing matrix, which overlaps with potrf(k+1)/trsm(k+1) on s1.
static int device_cholesky(double* dA, const size_t n, int* dFail) {
    if (n == 0) return -1;
    const int N = (int)n;
    const int nsteps = (N + NB - 1) / NB;

    CUDA_CHECK(cudaFuncSetAttribute(trsm_panel_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    (int)TRSM_SMEM_BYTES));

    cudaStream_t s1, s2;
    CUDA_CHECK(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking));
    std::vector<cudaEvent_t> evA(nsteps), evB(nsteps);
    for (int i = 0; i < nsteps; ++i) {
        CUDA_CHECK(cudaEventCreateWithFlags(&evA[i], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&evB[i], cudaEventDisableTiming));
    }

    CUDA_CHECK(cudaMemsetAsync(dFail, 0xFF, sizeof(int), s1));  // -1

    for (int step = 0; step < nsteps; ++step) {
        const int k = step * NB;
        const int nb = std::min(NB, N - k);
        const int m = N - k - nb;  // rows below the diagonal block

        // Column block k was last touched by the remainder update of step-2
        // (step-1's remainder starts one block further right).
        if (step >= 2) CUDA_CHECK(cudaStreamWaitEvent(s1, evB[step - 2], 0));

        // 1. Factor diagonal block
        potrf_diag_kernel<<<1, POTRF_THREADS, 0, s1>>>(dA, n, k, nb, dFail);
        CUDA_CHECK(cudaGetLastError());

        if (m <= 0) break;

        // 2. Triangular solve of the panel below it
        {
            const int blocks = (m + NB - 1) / NB;
            trsm_panel_kernel<<<blocks, TRSM_THREADS, TRSM_SMEM_BYTES, s1>>>(dA, n, k, nb, m,
                                                                           dFail);
            CUDA_CHECK(cudaGetLastError());
        }

        const double* L21 = dA + (size_t)(k + nb) * n + k;
        double* A22 = dA + (size_t)(k + nb) * n + (k + nb);

        // 3a. Lookahead: update only the next column block (needs step-1's remainder).
        if (step >= 1) CUDA_CHECK(cudaStreamWaitEvent(s1, evB[step - 1], 0));
        launch_syrk_lower(L21, A22, n, m, nb, /*panel_only=*/true, dFail, s1);
        CUDA_CHECK(cudaEventRecord(evA[step], s1));

        // 3b. Remainder of the trailing update on the second stream.
        CUDA_CHECK(cudaStreamWaitEvent(s2, evA[step], 0));
        launch_syrk_lower(L21 + (size_t)NB * n, A22 + (size_t)NB * n + NB, n, m - NB, nb,
                          /*panel_only=*/false, dFail, s2);
        CUDA_CHECK(cudaEventRecord(evB[step], s2));
    }

    CUDA_CHECK(cudaStreamSynchronize(s1));
    CUDA_CHECK(cudaStreamSynchronize(s2));

    int fail = -1;
    CUDA_CHECK(cudaMemcpy(&fail, dFail, sizeof(int), cudaMemcpyDeviceToHost));

    if (fail < 0) {
        // Zero out the upper triangular part.
        dim3 block(256);
        dim3 grid((unsigned)((n + 255) / 256), (unsigned)n);
        zero_upper_kernel<<<grid, block>>>(dA, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    for (int i = 0; i < nsteps; ++i) {
        CUDA_CHECK(cudaEventDestroy(evA[i]));
        CUDA_CHECK(cudaEventDestroy(evB[i]));
    }
    CUDA_CHECK(cudaStreamDestroy(s1));
    CUDA_CHECK(cudaStreamDestroy(s2));
    return fail;
}

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    // A is stored in row-major order
    const size_t bytes = n * n * sizeof(double);
    double* dA = nullptr;
    int* dFail = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dFail, sizeof(int)));

    CUDA_CHECK(cudaMemcpy(dA, A.data(), bytes, cudaMemcpyHostToDevice));

    const int fail = device_cholesky(dA, n, dFail);

    if (fail < 0) {
        CUDA_CHECK(cudaMemcpy(A.data(), dA, bytes, cudaMemcpyDeviceToHost));
    }

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dFail));

    if (fail >= 0) {
        // Matrix is not positive definite
        printf("Error: Matrix is not positive definite at diagonal element %d\n", fail);
        return false;
    }
    return true;
}

// Generate a symmetric positive definite matrix
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    // Method: Create A = B * B^T where B is random
    // This guarantees A is positive semi-definite
    // Then add identity to make it strictly positive definite

    std::vector<double> B(n * n);
    unsigned int seed = 42;

    // Generate random matrix B (sequential PRNG stream, kept on the host for reproducibility)
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

    // Compute A = B * B^T on the GPU
    const size_t bytes = n * n * sizeof(double);
    double *dB = nullptr, *dA = nullptr;
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMemcpy(dB, B.data(), bytes, cudaMemcpyHostToDevice));
    device_gemm_nt_full(dB, dB, dA, (int)n, (int)n, (int)n, 0);

    // Add diagonal dominance to ensure positive definiteness
    if (n > 0) {
        add_diag_kernel<<<(unsigned)((n + 255) / 256), 256>>>(dA, n, (double)n);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaMemcpy(A.data(), dA, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dA));
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix (on the GPU)
    const size_t bytes = n * n * sizeof(double);
    double *dL = nullptr, *dR = nullptr, *dA = nullptr;
    CUDA_CHECK(cudaMalloc(&dL, bytes));
    CUDA_CHECK(cudaMalloc(&dR, bytes));
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMemcpy(dL, L.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dA, A_orig.data(), bytes, cudaMemcpyHostToDevice));

    // Compute L * L^T
    device_gemm_nt_full(dL, dL, dR, (int)n, (int)n, (int)n, 0);

    // Compare with original
    const int blocks = 1024;
    double *dAbs = nullptr, *dRel = nullptr;
    CUDA_CHECK(cudaMalloc(&dAbs, blocks * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dRel, blocks * sizeof(double)));
    error_reduce_kernel<<<blocks, 256>>>(dR, dA, n * n, dAbs, dRel);
    CUDA_CHECK(cudaGetLastError());

    std::vector<double> hAbs(blocks), hRel(blocks);
    CUDA_CHECK(cudaMemcpy(hAbs.data(), dAbs, blocks * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hRel.data(), dRel, blocks * sizeof(double), cudaMemcpyDeviceToHost));

    double maxError = 0.0;
    double relError = 0.0;
    for (int i = 0; i < blocks; ++i) {
        maxError = std::max(maxError, hAbs[i]);
        relError = std::max(relError, hRel[i]);
    }

    CUDA_CHECK(cudaFree(dL));
    CUDA_CHECK(cudaFree(dR));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dAbs));
    CUDA_CHECK(cudaFree(dRel));

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

    // Initialize the CUDA context up front so it is not attributed to the timed region.
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
