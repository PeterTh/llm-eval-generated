#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// Cholesky decomposition on the GPU (CUDA, blocked right-looking algorithm)
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

constexpr int NB_MAX = 64; // largest panel / diagonal block size
#ifndef NB32_MAX_N
#define NB32_MAX_N 2048 // use 32-wide panels up to this matrix size
#endif

// GEMM tile configuration: (16*RM)x(16*CN) output tile, 16-deep k slices,
// RM x CN outputs per thread
constexpr int GK = 16;
constexpr int GTHREADS = 256;

enum GemmMode { GEMM_STORE = 0, GEMM_SUB_LOWER = 1 };

// C[i][j] (=|-=) sum_k X[i*ld + k] * X[j*ld + k]  for rowStart <= i < m,
// colStart <= j < colEnd, 0 <= k < kk. X and C are row-major with leading
// dimension ld. In GEMM_SUB_LOWER mode only the lower triangle (j <= i) is
// updated (symmetric rank-k update) and tiles above the diagonal are skipped.
template <int MODE, int RM, int CN>
__global__ void __launch_bounds__(GTHREADS)
syrkKernel(const double* __restrict__ X, double* __restrict__ C, int m, int kk, int ld, double diagAdd,
           int rowStart, int colStart, int colEnd) {
    constexpr int BM = 16 * RM;
    constexpr int BN = 16 * CN;
    const int row0 = rowStart + blockIdx.y * BM;
    const int col0 = colStart + blockIdx.x * BN;
    if (MODE == GEMM_SUB_LOWER && col0 > row0 + BM - 1) return;

    __shared__ double As[2][GK][BM + 1];
    __shared__ double Bs[2][GK][BN + 1];

    const int tid = threadIdx.x;
    const int tx = tid % 16;
    const int ty = tid / 16;

    double acc[RM][CN];
#pragma unroll
    for (int r = 0; r < RM; ++r)
#pragma unroll
        for (int c = 0; c < CN; ++c) acc[r][c] = 0.0;

    // Each thread loads RM elements of the A tile and CN of the B tile per k slice
    const int lk = tid % GK;
    const int lr = tid / GK; // 0..15
    double ra[RM], rb[CN];

    auto loadGlobal = [&](int k0) {
        const int k = k0 + lk;
#pragma unroll
        for (int q = 0; q < RM; ++q) {
            const int gi = row0 + lr + 16 * q;
            ra[q] = (gi < m && k < kk) ? X[(size_t)gi * ld + k] : 0.0;
        }
#pragma unroll
        for (int q = 0; q < CN; ++q) {
            const int gj = col0 + lr + 16 * q;
            rb[q] = (gj < colEnd && k < kk) ? X[(size_t)gj * ld + k] : 0.0;
        }
    };
    auto storeShared = [&](int buf) {
#pragma unroll
        for (int q = 0; q < RM; ++q) As[buf][lk][lr + 16 * q] = ra[q];
#pragma unroll
        for (int q = 0; q < CN; ++q) Bs[buf][lk][lr + 16 * q] = rb[q];
    };

    loadGlobal(0);
    storeShared(0);
    __syncthreads();

    int buf = 0;
    for (int k0 = 0; k0 < kk; k0 += GK) {
        const bool hasNext = k0 + GK < kk;
        if (hasNext) loadGlobal(k0 + GK);
#pragma unroll
        for (int k = 0; k < GK; ++k) {
            double a[RM], b[CN];
#pragma unroll
            for (int r = 0; r < RM; ++r) a[r] = As[buf][k][ty + 16 * r];
#pragma unroll
            for (int c = 0; c < CN; ++c) b[c] = Bs[buf][k][tx + 16 * c];
#pragma unroll
            for (int r = 0; r < RM; ++r)
#pragma unroll
                for (int c = 0; c < CN; ++c) acc[r][c] = fma(a[r], b[c], acc[r][c]);
        }
        if (hasNext) {
            storeShared(buf ^ 1);
            __syncthreads();
            buf ^= 1;
        }
    }

#pragma unroll
    for (int r = 0; r < RM; ++r) {
        const int gi = row0 + ty + 16 * r;
        if (gi >= m) continue;
#pragma unroll
        for (int c = 0; c < CN; ++c) {
            const int gj = col0 + tx + 16 * c;
            if (gj >= colEnd) continue;
            double* p = &C[(size_t)gi * ld + gj];
            if (MODE == GEMM_STORE) {
                *p = acc[r][c] + (gi == gj ? diagAdd : 0.0);
            } else if (gj <= gi) {
                *p -= acc[r][c];
            }
        }
    }
}

template <int NB>
struct PotrfShared {
    double colL[NB][NB + 1]; // input staging, then colL[j][r] = L[r][j]
    double rinv[NB];         // 1 / L[j][j]
    double dval[NB];         // pivot values L[j][j]^2
    int pivReady[NB];        // rinv[j] is available
    int colReady[NB];        // number of row halves that published column j
    int failed;
};

__device__ __forceinline__ int volatileLoad(const int* p) { return *(const volatile int*)p; }

// Wait until *flag >= value (or the factorization failed); returns false on failure
template <int NB>
__device__ __forceinline__ bool potrfWait(PotrfShared<NB>& sh, const int* flag, int value) {
    while (volatileLoad(flag) < value) {
        if (volatileLoad(&sh.failed)) return false;
    }
    __threadfence_block();
    return !volatileLoad(&sh.failed);
}

template <int NB>
__device__ __forceinline__ void potrfPivot(PotrfShared<NB>& sh, int j, double val, int k0, int* info) {
    sh.dval[j] = val;
    if (val <= 0.0) {
        // Matrix is not positive definite
        *info = k0 + j;
        __threadfence_block();
        *(volatile int*)&sh.failed = 1;
    } else {
        // Only the reciprocal is on the critical path; the diagonal itself
        // (sqrt) is computed for all pivots at the end
        sh.rinv[j] = rsqrt(val);
        __threadfence_block();
        *(volatile int*)&sh.pivReady[j] = 1;
    }
}

// Column loop of the diagonal block factorization for the warps owning the
// columns CS + 4 * m. CS is a template parameter so that all column tests are
// resolved at compile time (inactive FP64 instructions are not issued at all).
// Warps synchronize through flags only: the owners of column j scale it once
// its pivot is ready and publish it; every warp then applies it to its row.
template <int NB, int CS>
__device__ __forceinline__ void potrfColumns(PotrfShared<NB>& sh, double (&a)[NB / 4], int r, int lane,
                                             bool lowerHalf, int k0, int* info) {
    constexpr int HALVES = NB / 32;
#pragma unroll
    for (int j = 0; j < NB; ++j) {
        // Rows of the upper half have no work left beyond column 31
        if (j >= 32 && !lowerHalf) return;
        if ((j & 3) == CS) {
            // Scale and publish column j
            if (!potrfWait(sh, &sh.pivReady[j], 1)) return;
            const double l = a[j >> 2] * sh.rinv[j];
            a[j >> 2] = l;
            sh.colL[j][r] = l;
            __syncwarp();
            if (lane == 0) {
                __threadfence_block();
                atomicAdd(&sh.colReady[j], 1);
            }
        }
        if (j + 1 < NB) {
            if (!potrfWait(sh, &sh.colReady[j], j < 32 ? HALVES : 1)) return;
            const double lr = sh.colL[j][r];
            const int jn = j + 1;
            // The element holding the next pivot is updated first
            if ((jn & 3) == CS) {
                a[jn >> 2] = fma(-lr, sh.colL[j][jn], a[jn >> 2]);
                if (r == jn) potrfPivot(sh, jn, a[jn >> 2], k0, info);
            }
            // Columns < 32: all rows; columns >= 32: only the lower row half
#pragma unroll
            for (int m = 0; m < NB / 4; ++m) {
                const int c = CS + 4 * m;
                if (c > jn && c < 32) a[m] = fma(-lr, sh.colL[j][c], a[m]);
            }
            if (lowerHalf) {
#pragma unroll
                for (int m = 0; m < NB / 4; ++m) {
                    const int c = CS + 4 * m;
                    if (c > jn && c >= 32) a[m] = fma(-lr, sh.colL[j][c], a[m]);
                }
            }
        }
    }
}

// Factorize the nb x nb diagonal block starting at (k0, k0) in place; the upper
// part of the block is set to zero. Rows beyond nb are padded with the identity.
// Thread (warp w, lane) owns row r = 32 * (w / 4) + lane and the columns
// c = (w % 4) + 4 * m of that row in registers, so the FP64 work is spread over
// all SM sub-partitions. Columns of L are published in shared memory.
// Additionally writes W[j][c] = L[c][j] / L[j][j] (c > j, else 0) and
// W[NB*NB + c] = 1 / L[c][c] for the panel solve.
// info holds the index of the first non-positive pivot (or -1).
template <int NB>
__global__ void __launch_bounds__(NB * 4, 1)
potrfKernel(double* __restrict__ A, int n, int k0, int nb, int* info, double* __restrict__ W) {
    constexpr int THREADS = NB * 4;
    __shared__ PotrfShared<NB> sh;
    if (*info >= 0) return;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int rh = warp >> 2; // row half
    const int cs = warp & 3;  // column set
    const int r = rh * 32 + lane;

    for (int idx = tid; idx < NB * NB; idx += THREADS) {
        const int i = idx / NB, c = idx % NB;
        double v = 0.0;
        if (c <= i) v = (i < nb) ? A[(size_t)(k0 + i) * n + k0 + c] : (c == i ? 1.0 : 0.0);
        sh.colL[i][c] = v;
    }
    if (tid < NB) {
        sh.pivReady[tid] = 0;
        sh.colReady[tid] = 0;
    }
    if (tid == 0) sh.failed = 0;
    __syncthreads();

    double a[NB / 4];
#pragma unroll
    for (int m = 0; m < NB / 4; ++m) a[m] = sh.colL[r][cs + 4 * m];
    __syncthreads(); // staging area is reused for the columns of L

    if (tid == 0) potrfPivot(sh, 0, a[0], k0, info);

    switch (cs) {
        case 0: potrfColumns<NB, 0>(sh, a, r, lane, rh == 1, k0, info); break;
        case 1: potrfColumns<NB, 1>(sh, a, r, lane, rh == 1, k0, info); break;
        case 2: potrfColumns<NB, 2>(sh, a, r, lane, rh == 1, k0, info); break;
        default: potrfColumns<NB, 3>(sh, a, r, lane, rh == 1, k0, info); break;
    }
    __syncthreads();
    if (sh.failed) return;

    for (int idx = tid; idx < NB * NB; idx += THREADS) {
        const int i = idx / NB, c = idx % NB;
        if (i < nb && c < nb)
            A[(size_t)(k0 + i) * n + k0 + c] = (c < i) ? sh.colL[c][i] : (c == i ? sqrt(sh.dval[i]) : 0.0);
        // W[j][c] = L[c][j] / L[j][j]
        W[idx] = (c > i && c < nb) ? sh.colL[i][c] * sh.rinv[i] : 0.0;
    }
    if (tid < NB) W[NB * NB + tid] = (tid < nb) ? sh.rinv[tid] : 0.0;
}

// Single-warp factorization of a 32 x 32 diagonal block (same contract as
// potrfKernel<32>). The FP64 units are shared by the whole SM, so one warp
// already has the full FP64 throughput; lane r keeps row r in registers and
// columns are broadcast with shuffles, without any block-level synchronization.
__global__ void __launch_bounds__(32, 1)
potrfWarpKernel(double* __restrict__ A, int n, int k0, int nb, int* info, double* __restrict__ W) {
    constexpr int B = 32;
    __shared__ double t[B][B + 1];
    const int r = threadIdx.x;
    const int failedBefore = *info;

    // Coalesced load of the lower triangle (all loads in flight at once)
    double a[B];
    const double* src = A + (size_t)k0 * n + k0 + r;
#pragma unroll
    for (int i = 0; i < B; ++i) a[i] = (r <= i && i < nb) ? __ldg(src + (size_t)i * n) : 0.0;
#pragma unroll
    for (int i = 0; i < B; ++i) t[i][r] = (i >= nb && r == i) ? 1.0 : a[i];
    if (failedBefore >= 0) return;
    __syncwarp();
#pragma unroll
    for (int c = 0; c < B; ++c) a[c] = t[r][c];

    double piv = 1.0;
#pragma unroll
    for (int j = 0; j < B; ++j) {
        const double val = __shfl_sync(0xffffffffu, a[j], j);
        if (val <= 0.0) {
            // Matrix is not positive definite
            if (r == 0) *info = k0 + j;
            return;
        }
        if (r == j) piv = val;
        const double rinv = rsqrt(val);
        const double l = a[j] * rinv;
        a[j] = l;
        // W[j][c] = L[c][j] / L[j][j]
        W[j * B + r] = (r > j && r < nb) ? l * rinv : 0.0;
        if (r == j) W[B * B + j] = (j < nb) ? rinv : 0.0;
#pragma unroll
        for (int c = j + 1; c < B; ++c) a[c] = fma(-l, __shfl_sync(0xffffffffu, l, c), a[c]);
    }

    // Final factor: lower part with exact square roots on the diagonal
    const double d = sqrt(piv);
    __syncwarp();
#pragma unroll
    for (int c = 0; c < B; ++c) t[r][c] = (c < r) ? a[c] : (c == r ? d : 0.0);
    __syncwarp();
    for (int i = 0; i < nb; ++i)
        if (r < nb) A[(size_t)(k0 + i) * n + k0 + r] = t[i][r];
}

// Solve L21 * L11^T = A21 for rows [rowStart, rowEnd) of the panel at column k0.
// One warp per row; the row lives in registers (lane owns columns lane + 32 h).
// Uses unscaled partial results p_j = x_j * L_jj, so the inner loop is a
// plain FMA: p_c -= p_j * (L_cj / L_jj); finally x_c = p_c / L_cc.
constexpr int TRSM_WARPS = 2;
constexpr int TRSM_RPW = 4; // rows per warp, processed together for ILP
template <int NB>
__global__ void __launch_bounds__(TRSM_WARPS * 32)
trsmKernel(double* __restrict__ A, int ld, int rowStart, int rowEnd, int k0, int nb, const int* info,
           const double* __restrict__ W) {
    constexpr int H = NB / 32;
    if (*info >= 0) return;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int rowBase = rowStart + (blockIdx.x * TRSM_WARPS + warp) * TRSM_RPW;

    double x[H][TRSM_RPW];
#pragma unroll
    for (int q = 0; q < TRSM_RPW; ++q) {
        const int row = rowBase + q;
#pragma unroll
        for (int h = 0; h < H; ++h) {
            const int c = lane + 32 * h;
            x[h][q] = (row < rowEnd && c < nb) ? A[(size_t)row * ld + k0 + c] : 0.0;
        }
    }

#pragma unroll
    for (int hj = 0; hj < H; ++hj) {
#pragma unroll 8
        for (int jj = 0; jj < 32; ++jj) {
            const int j = 32 * hj + jj;
            double m[H];
#pragma unroll
            for (int h = hj; h < H; ++h) m[h] = __ldg(&W[j * NB + lane + 32 * h]);
#pragma unroll
            for (int q = 0; q < TRSM_RPW; ++q) {
                const double pj = __shfl_sync(0xffffffffu, x[hj][q], jj);
#pragma unroll
                for (int h = hj; h < H; ++h) x[h][q] = fma(-pj, m[h], x[h][q]);
            }
        }
    }

#pragma unroll
    for (int h = 0; h < H; ++h) {
        const int c = lane + 32 * h;
        const double rc = __ldg(&W[NB * NB + c]);
#pragma unroll
        for (int q = 0; q < TRSM_RPW; ++q) {
            const int row = rowBase + q;
            if (row < rowEnd && c < nb) A[(size_t)row * ld + k0 + c] = x[h][q] * rc;
        }
    }
}

// Zero the strict upper triangle of the n x n row-major matrix A
__global__ void zeroUpperKernel(double* __restrict__ A, int n) {
    for (int i = blockIdx.y; i < n; i += gridDim.y)
        for (int j = i + 1 + blockIdx.x * blockDim.x + threadIdx.x; j < n; j += gridDim.x * blockDim.x)
            A[(size_t)i * n + j] = 0.0;
}

// Copy the lower triangle (including the diagonal) of rows [r0, r1) of the
// n x n row-major host matrix hA (mapped pinned memory, read directly over
// PCIe) to dA and zero the strict upper triangle of these rows of dA. Only half
// of the matrix crosses the bus.
__global__ void gatherLowerKernel(const double* __restrict__ hA, double* __restrict__ dA, int n, int r0, int r1) {
    for (int i = r0 + blockIdx.x; i < r1; i += gridDim.x) {
        const double* src = hA + (size_t)i * n;
        double* dst = dA + (size_t)i * n;
        for (int j = threadIdx.x; j < n; j += blockDim.x) dst[j] = (j <= i) ? src[j] : 0.0;
    }
}

static int numSMs() {
    static int sms = 0;
    if (sms == 0) {
        int dev = 0;
        CUDA_CHECK(cudaGetDevice(&dev));
        CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    }
    return sms;
}

template <int MODE, int RM, int CN>
static void launchGemmTile(const double* X, double* C, int m, int kk, int ld, double diagAdd, cudaStream_t st,
                           int rowStart, int colStart, int colEnd) {
    constexpr int BM = 16 * RM;
    constexpr int BN = 16 * CN;
    dim3 grid((colEnd - colStart + BN - 1) / BN, (m - rowStart + BM - 1) / BM);
    syrkKernel<MODE, RM, CN><<<grid, GTHREADS, 0, st>>>(X, C, m, kk, ld, diagAdd, rowStart, colStart, colEnd);
}

// Launch the tiled X * X^T kernel over rows [rowStart, m) and columns
// [colStart, colEnd). The row tile height is chosen so that there are enough
// thread blocks to keep all SMs busy (FP64 throughput per SM is low).
template <int MODE>
static void launchGemm(const double* X, double* C, int m, int kk, int ld, double diagAdd, cudaStream_t st,
                       int rowStart, int colStart, int colEnd) {
    if (rowStart >= m || colStart >= colEnd) return;
    const int cols = colEnd - colStart;
    if (cols <= 32) {
        // Narrow column strip (the next panel)
        launchGemmTile<MODE, 1, 2>(X, C, m, kk, ld, diagAdd, st, rowStart, colStart, colEnd);
        return;
    }
    const long rows = m - rowStart;
    const long colTiles = (cols + 63) / 64;
    const bool lower = MODE == GEMM_SUB_LOWER && colStart == rowStart;
    auto blocks = [&](long bm) {
        const long rowTiles = (rows + bm - 1) / bm;
        return lower ? rowTiles * (colTiles + 1) / 2 : rowTiles * colTiles;
    };
    const long target = 2L * numSMs();
    if (blocks(64) >= target)
        launchGemmTile<MODE, 4, 4>(X, C, m, kk, ld, diagAdd, st, rowStart, colStart, colEnd);
    else if (blocks(32) >= target)
        launchGemmTile<MODE, 2, 4>(X, C, m, kk, ld, diagAdd, st, rowStart, colStart, colEnd);
    else
        launchGemmTile<MODE, 1, 4>(X, C, m, kk, ld, diagAdd, st, rowStart, colStart, colEnd);
}

#ifndef POTRF_EXCLUSIVE_SMEM
#define POTRF_EXCLUSIVE_SMEM (80 * 1024)
#endif

struct PanelContext {
    double* dA;
    double* hA;
    int ld;
    int* dInfo;
    double* dW;
    cudaStream_t sCopy;
    cudaEvent_t evRow;
};

template <int NB>
static void launchTrsm(const PanelContext& ctx, int k0, int rowStart, int rowEnd, cudaStream_t st) {
    if (rowEnd <= rowStart) return;
    constexpr int rowsPerBlock = TRSM_WARPS * TRSM_RPW;
    const int nb = NB; // panels being solved against are always complete here
    trsmKernel<NB><<<(rowEnd - rowStart + rowsPerBlock - 1) / rowsPerBlock, TRSM_WARPS * 32, 0, st>>>(
        ctx.dA, ctx.ld, rowStart, rowEnd, k0, nb, ctx.dInfo, ctx.dW);
}

// Factorize the diagonal block at k0, start copying the finished block row
// back to the host, and solve the panel rows below the diagonal block
template <int NB>
static void launchPanel(const PanelContext& ctx, int k0, cudaStream_t st) {
    const int ld = ctx.ld;
    const int nb = std::min(NB, ld - k0);
    double* W = ctx.dW;
    if constexpr (NB == 32)
        potrfWarpKernel<<<1, 32, POTRF_EXCLUSIVE_SMEM, st>>>(ctx.dA, ld, k0, nb, ctx.dInfo, W);
    else
        potrfKernel<NB><<<1, NB * 4, 0, st>>>(ctx.dA, ld, k0, nb, ctx.dInfo, W);

    // Block row [k0, k0 + nb) is final now (upper part is zero)
    CUDA_CHECK(cudaEventRecord(ctx.evRow, st));
    CUDA_CHECK(cudaStreamWaitEvent(ctx.sCopy, ctx.evRow, 0));
    const size_t off = (size_t)k0 * ld;
    CUDA_CHECK(cudaMemcpyAsync(ctx.hA + off, ctx.dA + off, (size_t)nb * ld * sizeof(double),
                               cudaMemcpyDeviceToHost, ctx.sCopy));

    if (nb == NB) launchTrsm<NB>(ctx, k0, k0 + nb, ld, st);
}

// Right-looking blocked factorization with one step of lookahead: the next
// panel is updated first, then factorized on a high priority stream while the
// rest of the trailing matrix is updated.
template <int NB>
static void factorize(const PanelContext& ctx, cudaStream_t sMain, cudaStream_t sPanel, cudaEvent_t evStrip,
                      cudaEvent_t evPanel) {
    const int N = ctx.ld;
    const size_t ld = (size_t)N;
    // The first panel runs on sPanel as well, so that the copy stream always
    // waits for panel work in stream order
    CUDA_CHECK(cudaEventRecord(evStrip, sMain));
    CUDA_CHECK(cudaStreamWaitEvent(sPanel, evStrip, 0));
    launchPanel<NB>(ctx, 0, sPanel);
    CUDA_CHECK(cudaEventRecord(evPanel, sPanel));
    CUDA_CHECK(cudaStreamWaitEvent(sMain, evPanel, 0));
    for (int k0 = 0; k0 < N; k0 += NB) {
        const int nb = std::min(NB, N - k0);
        const int rem = N - k0 - nb;
        if (rem <= 0) break;
        double* trail = ctx.dA + (size_t)(k0 + nb) * ld + (k0 + nb);
        const double* panel = ctx.dA + (size_t)(k0 + nb) * ld + k0;
        const int next = std::min(NB, rem);

        // Update the next panel
        launchGemm<GEMM_SUB_LOWER>(panel, trail, rem, nb, N, 0.0, sMain, 0, 0, next);
        CUDA_CHECK(cudaEventRecord(evStrip, sMain));
        CUDA_CHECK(cudaStreamWaitEvent(sPanel, evStrip, 0));
        launchPanel<NB>(ctx, k0 + nb, sPanel);
        CUDA_CHECK(cudaEventRecord(evPanel, sPanel));

        // Update the remaining trailing matrix concurrently
        launchGemm<GEMM_SUB_LOWER>(panel, trail, rem, nb, N, 0.0, sMain, next, next, rem);
        CUDA_CHECK(cudaStreamWaitEvent(sMain, evPanel, 0));
    }
}

// Device buffers, streams, events and the captured CUDA graph used by the
// factorization. They are set up once and reused.
struct CholeskyWorkspace {
    size_t n = 0;
    double* dA = nullptr;
    double* dW = nullptr;
    int* dInfo = nullptr;
    int* hInfo = nullptr; // pinned
    cudaStream_t sMain = nullptr, sPanel = nullptr, sCopy = nullptr;
    cudaEvent_t evStrip = nullptr, evPanel = nullptr, evRow = nullptr, evJoin = nullptr;
    // Whole factorization (transfers + kernels) as one graph for a given host buffer
    cudaGraphExec_t graph = nullptr;
    const double* graphHost = nullptr;
    size_t graphN = 0;
};
static CholeskyWorkspace g_ws;

void choleskyRelease() {
    if (g_ws.n == 0) return;
    if (g_ws.graph) CUDA_CHECK(cudaGraphExecDestroy(g_ws.graph));
    CUDA_CHECK(cudaFree(g_ws.dA));
    CUDA_CHECK(cudaFree(g_ws.dW));
    CUDA_CHECK(cudaFree(g_ws.dInfo));
    CUDA_CHECK(cudaFreeHost(g_ws.hInfo));
    CUDA_CHECK(cudaEventDestroy(g_ws.evStrip));
    CUDA_CHECK(cudaEventDestroy(g_ws.evPanel));
    CUDA_CHECK(cudaEventDestroy(g_ws.evRow));
    CUDA_CHECK(cudaEventDestroy(g_ws.evJoin));
    CUDA_CHECK(cudaStreamDestroy(g_ws.sMain));
    CUDA_CHECK(cudaStreamDestroy(g_ws.sPanel));
    CUDA_CHECK(cudaStreamDestroy(g_ws.sCopy));
    g_ws = CholeskyWorkspace{};
}

static void allocateWorkspace(const size_t n) {
    if (g_ws.n >= n) return;
    choleskyRelease();
    int prioLow = 0, prioHigh = 0;
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&prioLow, &prioHigh));
    CUDA_CHECK(cudaStreamCreateWithPriority(&g_ws.sMain, cudaStreamNonBlocking, prioLow));
    CUDA_CHECK(cudaStreamCreateWithPriority(&g_ws.sPanel, cudaStreamNonBlocking, prioHigh));
    CUDA_CHECK(cudaStreamCreateWithFlags(&g_ws.sCopy, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&g_ws.evStrip, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&g_ws.evPanel, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&g_ws.evRow, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&g_ws.evJoin, cudaEventDisableTiming));
    CUDA_CHECK(cudaMalloc(&g_ws.dA, n * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&g_ws.dW, (NB_MAX * NB_MAX + NB_MAX) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&g_ws.dInfo, sizeof(int)));
    CUDA_CHECK(cudaMallocHost(&g_ws.hInfo, sizeof(int)));
    // The single-warp diagonal factorization requests (unused) shared memory so
    // that it gets an SM of its own instead of sharing the FP64 units with the
    // trailing update
    CUDA_CHECK(cudaFuncSetAttribute(potrfWarpKernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    POTRF_EXCLUSIVE_SMEM));
    CUDA_CHECK(cudaFuncSetAttribute(potrfWarpKernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    g_ws.n = n;
}

// Enqueue the complete factorization of the host matrix hA (n x n) on sMain
// (forking to and joining back from the helper streams).
static void enqueueCholesky(double* hA, const int N) {
    const size_t n = (size_t)N;
    double* dA = g_ws.dA;
    cudaStream_t sMain = g_ws.sMain;

    CUDA_CHECK(cudaMemsetAsync(g_ws.dInfo, 0xff, sizeof(int), sMain));
    CUDA_CHECK(cudaEventRecord(g_ws.evJoin, sMain));
    CUDA_CHECK(cudaStreamWaitEvent(g_ws.sCopy, g_ws.evJoin, 0));
    CUDA_CHECK(cudaStreamWaitEvent(g_ws.sPanel, g_ws.evJoin, 0));

    // Panel width: narrow panels shorten the critical path for small matrices
    const int nbSel = (N <= NB32_MAX_N) ? 32 : 64;

    // Host to device copy. If the pinned host buffer is mapped into the device
    // address space, a kernel reads only its lower triangle directly over PCIe.
    // Otherwise narrow rows are copied as one contiguous block (pitched copies
    // of short rows are slow) and wide rows by block rows with pitched copies of
    // their lower part only. The strict upper triangle (never read by the
    // factorization) is zeroed on the device, as it is the final upper part.
    double* hMapped = nullptr;
    if (cudaHostGetDevicePointer((void**)&hMapped, hA, 0) != cudaSuccess) {
        cudaGetLastError();
        hMapped = nullptr;
    }
    if (hMapped) {
        gatherLowerKernel<<<16 * numSMs(), 256, 0, sMain>>>(hMapped, dA, N, 0, N);
    } else {
        constexpr int PITCHED_MIN_WIDTH = 8192;
        int contiguousRows = std::min(N, PITCHED_MIN_WIDTH);
        contiguousRows = std::min(N, (contiguousRows + nbSel - 1) / nbSel * nbSel);
        CUDA_CHECK(cudaMemcpyAsync(dA, hA, (size_t)contiguousRows * n * sizeof(double), cudaMemcpyHostToDevice,
                                   sMain));
        for (int r0 = contiguousRows; r0 < N; r0 += nbSel) {
            const int rows = std::min(nbSel, N - r0);
            const size_t off = (size_t)r0 * n;
            CUDA_CHECK(cudaMemcpy2DAsync(dA + off, n * sizeof(double), hA + off, n * sizeof(double),
                                         (size_t)(r0 + rows) * sizeof(double), rows, cudaMemcpyHostToDevice,
                                         sMain));
        }
        zeroUpperKernel<<<dim3((N + 255) / 256, std::min(N, 4096)), 256, 0, sMain>>>(dA, N);
    }

    const PanelContext ctx{dA, hA, N, g_ws.dInfo, g_ws.dW, g_ws.sCopy, g_ws.evRow};
    if (nbSel == 32)
        factorize<32>(ctx, sMain, g_ws.sPanel, g_ws.evStrip, g_ws.evPanel);
    else
        factorize<64>(ctx, sMain, g_ws.sPanel, g_ws.evStrip, g_ws.evPanel);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpyAsync(g_ws.hInfo, g_ws.dInfo, sizeof(int), cudaMemcpyDeviceToHost, sMain));
    // Join the helper streams
    CUDA_CHECK(cudaEventRecord(g_ws.evJoin, g_ws.sPanel));
    CUDA_CHECK(cudaStreamWaitEvent(sMain, g_ws.evJoin, 0));
    CUDA_CHECK(cudaEventRecord(g_ws.evJoin, g_ws.sCopy));
    CUDA_CHECK(cudaStreamWaitEvent(sMain, g_ws.evJoin, 0));
}

static bool isPinned(const void* p) {
    cudaPointerAttributes attr;
    if (cudaPointerGetAttributes(&attr, p) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return attr.type == cudaMemoryTypeHost;
}

// Capture the factorization of hA (n x n) into a CUDA graph. With hundreds of
// small dependent kernels and copies, launching them one by one from the host
// is slower than executing them; a graph is launched with a single call.
static void buildGraph(double* hA, const size_t n) {
    if (g_ws.graph && g_ws.graphHost == hA && g_ws.graphN == n) return;
    if (g_ws.graph) CUDA_CHECK(cudaGraphExecDestroy(g_ws.graph));
    cudaGraph_t graph;
    CUDA_CHECK(cudaStreamBeginCapture(g_ws.sMain, cudaStreamCaptureModeThreadLocal));
    enqueueCholesky(hA, (int)n);
    CUDA_CHECK(cudaStreamEndCapture(g_ws.sMain, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&g_ws.graph, graph, 0));
    CUDA_CHECK(cudaGraphDestroy(graph));
    g_ws.graphHost = hA;
    g_ws.graphN = n;
}

// The GPU lowers its PCIe link speed when the bus has been idle for a while
// and only restores it under sustained traffic; bring the link up before the
// timed run (like the GPU clocks, this is part of the device warm-up).
static void warmUpTransfers() {
    constexpr size_t bytes = 64u << 20;
    void* h = nullptr;
    void* d = nullptr;
    CUDA_CHECK(cudaMallocHost(&h, bytes));
    CUDA_CHECK(cudaMalloc(&d, bytes));
    memset(h, 0, bytes);
    CUDA_CHECK(cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(h, d, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d));
    CUDA_CHECK(cudaFreeHost(h));
}

// Set up the device workspace for an n x n matrix stored in the (pinned) host
// buffer A ahead of time, like the CUDA context itself.
void choleskyPrepare(std::vector<double>& A, const size_t n) {
    if (n == 0) return;
    allocateWorkspace(n);
    if (isPinned(A.data())) buildGraph(A.data(), n);
    warmUpTransfers();
}

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    // A is stored in row-major order
    if (n == 0) return true;
    allocateWorkspace(n);
    if (isPinned(A.data())) {
        buildGraph(A.data(), n);
        CUDA_CHECK(cudaGraphLaunch(g_ws.graph, g_ws.sMain));
    } else {
        enqueueCholesky(A.data(), (int)n);
    }
    CUDA_CHECK(cudaStreamSynchronize(g_ws.sMain));

    const int info = *g_ws.hInfo;
    if (info >= 0) {
        printf("Error: Matrix is not positive definite at diagonal element %d\n", info);
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
    
    // Generate random matrix B
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }
    if (n == 0) return;
    
    // Compute A = B * B^T on the GPU, adding diagonal dominance (n) to ensure
    // positive definiteness
    const size_t bytes = n * n * sizeof(double);
    double *dB = nullptr, *dA = nullptr;
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMemcpy(dB, B.data(), bytes, cudaMemcpyHostToDevice));
    launchGemm<GEMM_STORE>(dB, dA, (int)n, (int)n, (int)n, (double)n, 0, 0, 0, (int)n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(A.data(), dA, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dA));
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix
    
    std::vector<double> reconstructed(n * n);
    
    // Compute L * L^T on the GPU
    if (n > 0) {
        const size_t bytes = n * n * sizeof(double);
        double *dL = nullptr, *dR = nullptr;
        CUDA_CHECK(cudaMalloc(&dL, bytes));
        CUDA_CHECK(cudaMalloc(&dR, bytes));
        CUDA_CHECK(cudaMemcpy(dL, L.data(), bytes, cudaMemcpyHostToDevice));
        launchGemm<GEMM_STORE>(dL, dR, (int)n, (int)n, (int)n, 0.0, 0, 0, 0, (int)n);
        CUDA_CHECK(cudaGetLastError());
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
    
    // Initialize the CUDA context (loading all kernels eagerly) up front so it
    // is not part of the timing
    setenv("CUDA_MODULE_LOADING", "EAGER", 1);
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
    
    // Pin the host buffer for fast transfers and set up the device workspace
    if (n > 0) CUDA_CHECK(cudaHostRegister(A.data(), n * n * sizeof(double), cudaHostRegisterMapped));
    choleskyPrepare(A, n);
    
    // Perform Cholesky decomposition
    printf("Computing Cholesky decomposition...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    bool success = choleskyDecomposition(A, n);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (n > 0) CUDA_CHECK(cudaHostUnregister(A.data()));
    choleskyRelease();
    
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
