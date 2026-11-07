// Hybrid MPI + OpenMP + CUDA Cholesky decomposition benchmark.
//
// Parallelization strategy
// ------------------------
//  * MPI    : block rows of the (row-major) matrix are distributed block-cyclically
//             over the ranks; one rank drives one GPU.  Per step the factored panel
//             (one block column) is all-gathered, the small diagonal block is
//             broadcast.  Works across nodes (host-staged communication).
//  * CUDA   : all O(n^3) work (trailing-matrix update, panel TRSM, matrix
//             generation, validation GEMM) runs on the GPU with custom kernels.
//             A look-ahead pipeline on two streams overlaps the panel
//             factorization/communication of step k+1 with the large trailing
//             update of step k.
//  * OpenMP : host side work - diagonal block factorization, verification of the
//             random number generator, result comparison, gathering of results.
//
// Semantics match the original sequential program: same SPD matrix (identical
// glibc rand_r sequence), same L (up to floating point summation order), same
// output / validation / exit codes.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>
#include <sched.h>

#include "../common/results_output.hpp"

// ---------------------------------------------------------------------------
// Error handling helpers
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                                      \
    do {                                                                                      \
        cudaError_t err__ = (call);                                                           \
        if (err__ != cudaSuccess) {                                                           \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err__), __FILE__, \
                    __LINE__);                                                                \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                     \
        }                                                                                     \
    } while (0)

#define MPI_CHECK(call)                                                   \
    do {                                                                  \
        int err__ = (call);                                               \
        if (err__ != MPI_SUCCESS) {                                       \
            fprintf(stderr, "MPI error %d at %s:%d\n", err__, __FILE__, __LINE__); \
            MPI_Abort(MPI_COMM_WORLD, 1);                                 \
        }                                                                 \
    } while (0)

// ---------------------------------------------------------------------------
// Distributed layout description
// ---------------------------------------------------------------------------
struct Layout {
    long n = 0;        // matrix dimension
    int nb = 256;      // block size (rows per block row, columns per block column)
    long nblocks = 0;  // number of block rows
    int rank = 0;
    int nprocs = 1;
    long nloc = 0;     // number of block rows owned by this rank

    int owner(long i) const { return (int)(i % nprocs); }
    long localIndex(long i) const { return i / nprocs; }  // valid if owner(i)==rank
    long blockRows(long i) const { return std::min<long>(nb, n - i * nb); }
    // first local block index with global block index >= i0
    long firstLocalAtOrAfter(long i0) const { return (i0 + nprocs - 1 - rank) / nprocs; }
};

// ---------------------------------------------------------------------------
// glibc rand_r reimplementation with jump-ahead (for parallel generation).
// The three-step LCG state update is verified at runtime against the real
// rand_r; if it does not match (non-glibc libc) we fall back to sequential rand_r.
// ---------------------------------------------------------------------------
__host__ __device__ inline unsigned int lcg_jump(unsigned int seed, unsigned long long steps) {
    // state_{k+1} = a*state_k + c (mod 2^32). Compose f(x)=A x + C by squaring.
    unsigned int A = 1103515245u, C = 12345u;
    unsigned int accA = 1u, accC = 0u;
    while (steps) {
        if (steps & 1ull) {
            accC = A * accC + C;
            accA = A * accA;
        }
        C = A * C + C;
        A = A * A;
        steps >>= 1;
    }
    return accA * seed + accC;
}

__host__ __device__ inline int glibc_rand_r(unsigned int* seed) {
    unsigned int next = *seed;
    int result;
    next *= 1103515245u;
    next += 12345u;
    result = (int)((next / 65536u) % 2048u);
    next *= 1103515245u;
    next += 12345u;
    result <<= 10;
    result ^= (int)((next / 65536u) % 1024u);
    next *= 1103515245u;
    next += 12345u;
    result <<= 10;
    result ^= (int)((next / 65536u) % 1024u);
    *seed = next;
    return result;
}

// Verify that our reimplementation reproduces the platform rand_r exactly.
static bool verifyRandImplementation() {
    const int checks = 1 << 16;
    unsigned int s1 = 42, s2 = 42;
    for (int i = 0; i < checks; ++i) {
        if (rand_r(&s1) != glibc_rand_r(&s2)) return false;
        if (s1 != s2) return false;
        if (lcg_jump(42u, 3ull * (unsigned long long)(i + 1)) != s1) return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// CUDA kernels
// ---------------------------------------------------------------------------

// Generate rows [row0, row0+nrows) of the random matrix B (n columns) exactly as
// the sequential loop  B[e] = rand_r(&seed)/RAND_MAX - 0.5  would.
__global__ void genBRowsKernel(double* __restrict__ out, long row0, long nrows, long n, double randMax) {
    long total = nrows * n;
    for (long idx = blockIdx.x * (long)blockDim.x + threadIdx.x; idx < total;
         idx += (long)gridDim.x * blockDim.x) {
        long r = idx / n;
        long c = idx - r * n;
        unsigned long long e = (unsigned long long)(row0 + r) * (unsigned long long)n + (unsigned long long)c;
        unsigned int seed = lcg_jump(42u, 3ull * e);
        int v = glibc_rand_r(&seed);
        out[idx] = (v / randMax) - 0.5;
    }
}

// ---------------------------------------------------------------------------
// GEMM "C += alpha * X * Y^T" with row-major X (m x k, ld ldx), Y (ncols x k, ld ldy).
// Shared-memory tiled, TM x TN tile per 256-thread block, 16-wide k steps.
// ---------------------------------------------------------------------------
#define GEMM_TK 16

template <int TM, int TN>
__device__ __forceinline__ void gemmTile(const double* __restrict__ Xp, long ldx, const double* __restrict__ Yp,
                                         long ldy, double* __restrict__ Cp, long ldc, long m, long ncols, long kdim,
                                         long tileRow, long tileCol, double alpha) {
    constexpr int MR = TM / 16;
    constexpr int NR = TN / 16;
    __shared__ double Xs[GEMM_TK][TM + 1];
    __shared__ double Ys[GEMM_TK][TN + 1];

    const int tid = threadIdx.x;
    const int tx = tid & 15;
    const int ty = tid >> 4;
    const int lt = tid & 15;  // k index this thread loads
    const int lr = tid >> 4;  // row index base this thread loads (lr, lr+16, ...)

    double acc[MR][NR];
#pragma unroll
    for (int p = 0; p < MR; ++p)
#pragma unroll
        for (int q = 0; q < NR; ++q) acc[p][q] = 0.0;

    for (long t0 = 0; t0 < kdim; t0 += GEMM_TK) {
        const bool tOk = (t0 + lt) < kdim;
#pragma unroll
        for (int q = 0; q < MR; ++q) {
            const int r = lr + 16 * q;
            const long gr = tileRow + r;
            double v = 0.0;
            if (tOk && gr < m) v = Xp[gr * ldx + t0 + lt];
            Xs[lt][r] = v;
        }
#pragma unroll
        for (int q = 0; q < NR; ++q) {
            const int c = lr + 16 * q;
            const long gc = tileCol + c;
            double v = 0.0;
            if (tOk && gc < ncols) v = Yp[gc * ldy + t0 + lt];
            Ys[lt][c] = v;
        }
        __syncthreads();
#pragma unroll
        for (int t = 0; t < GEMM_TK; ++t) {
            double a[MR], b[NR];
#pragma unroll
            for (int p = 0; p < MR; ++p) a[p] = Xs[t][ty * MR + p];
#pragma unroll
            for (int q = 0; q < NR; ++q) b[q] = Ys[t][tx * NR + q];
#pragma unroll
            for (int p = 0; p < MR; ++p)
#pragma unroll
                for (int q = 0; q < NR; ++q) acc[p][q] += a[p] * b[q];
        }
        __syncthreads();
    }
#pragma unroll
    for (int p = 0; p < MR; ++p) {
        const long gr = tileRow + ty * MR + p;
        if (gr >= m) continue;
#pragma unroll
        for (int q = 0; q < NR; ++q) {
            const long gc = tileCol + tx * NR + q;
            if (gc < ncols) Cp[gr * ldc + gc] += alpha * acc[p][q];
        }
    }
}

// Batched over the block rows owned by this rank: blockIdx.z selects local block
// row l = lstart + z (global block i = l*P + rank).
//   X rows for block i : xLocal ? X + l*nb*ldx : X + (i*nb - xrow0)*ldx
//   Y rows for column c: Y + (c - yrow0)*ldy
//   C rows for block i : C + l*nb*ldc
//   columns: mode 0: [c0, c1)   mode 1: [c0, i*nb) (strictly left of the diagonal
//            block)   mode 2: the diagonal block [i*nb, min((i+1)*nb, n))
template <int TM, int TN>
__global__ void __launch_bounds__(256) gemmBatchedKernel(const double* __restrict__ X, long ldx, long xrow0,
                                                         int xLocal, const double* __restrict__ Y, long ldy,
                                                         long yrow0, double* __restrict__ C, long ldc, long kdim,
                                                         long n, int nb, int rank, int P, long lstart, int mode,
                                                         long c0, long c1, double alpha) {
    const long l = lstart + blockIdx.z;
    const long i = l * P + rank;
    const long r0 = i * (long)nb;
    const long m = min((long)nb, n - r0);
    const long cstart = (mode == 2) ? r0 : c0;
    const long cend = (mode == 0) ? c1 : ((mode == 1) ? r0 : min(r0 + nb, n));
    const long ncols = cend - cstart;
    const long tileRow = (long)blockIdx.y * TM;
    const long tileCol = (long)blockIdx.x * TN;
    if (tileRow >= m || tileCol >= ncols) return;
    const double* Xp = xLocal ? (X + l * (long)nb * ldx) : (X + (r0 - xrow0) * ldx);
    const double* Yp = Y + (cstart - yrow0) * ldy;
    double* Cp = C + l * (long)nb * ldc + cstart;
    gemmTile<TM, TN>(Xp, ldx, Yp, ldy, Cp, ldc, m, ncols, kdim, tileRow, tileCol, alpha);
}

// Plain GEMM on a contiguous row range.
template <int TM, int TN>
__global__ void __launch_bounds__(256) gemmPlainKernel(const double* __restrict__ X, long ldx,
                                                       const double* __restrict__ Y, long ldy, double* __restrict__ C,
                                                       long ldc, long m, long ncols, long kdim, double alpha) {
    const long tileRow = (long)blockIdx.y * TM;
    const long tileCol = (long)blockIdx.x * TN;
    if (tileRow >= m || tileCol >= ncols) return;
    gemmTile<TM, TN>(X, ldx, Y, ldy, C, ldc, m, ncols, kdim, tileRow, tileCol, alpha);
}

// Small triangular solve  X * L^T = B  (in place, row-wise), L lower triangular
// w x w with w <= 64 held in shared memory; `rows` contiguous rows of X.
#define TS_ROWS 16
#define TS_LANES 16
#define TS_W 64

__global__ void __launch_bounds__(256) trsmSmallKernel(double* __restrict__ X, long ldx, long rows,
                                                       const double* __restrict__ L, long ldl, int w) {
    __shared__ double Ls[TS_W][TS_W + 1];
    __shared__ double Xs[TS_ROWS][TS_W + 1];
    const int tid = threadIdx.x;
    for (int e = tid; e < w * w; e += 256) {
        const int r = e / w, c = e - r * w;
        Ls[r][c] = L[(long)r * ldl + c];
    }
    const long row0 = (long)blockIdx.x * TS_ROWS;
    const int row = tid / TS_LANES;
    const int lane = tid % TS_LANES;
    const bool rowOk = (row0 + row) < rows;
    double* Xrow = X + (row0 + row) * ldx;
    if (rowOk)
        for (int c = lane; c < w; c += TS_LANES) Xs[row][c] = Xrow[c];
    __syncthreads();

    for (int c = 0; c < w; ++c) {
        double s = 0.0;
        for (int t = lane; t < c; t += TS_LANES) s += Xs[row][t] * Ls[c][t];
#pragma unroll
        for (int off = TS_LANES / 2; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffffu, s, off, TS_LANES);
        if (lane == 0) Xs[row][c] = (Xs[row][c] - s) / Ls[c][c];
        __syncwarp();
    }
    if (rowOk)
        for (int c = lane; c < w; c += TS_LANES) Xrow[c] = Xs[row][c];
}

// In-place Cholesky factorization of a w x w (w <= 64) block in shared memory by
// one thread block, left-looking column order as in the original algorithm.
// On a non-positive pivot j, *fail is set to failBase + j and the block is left
// unchanged.
__global__ void __launch_bounds__(256) potrf64Kernel(double* __restrict__ A, long ld, int w,
                                                     double* __restrict__ fail, int failBase) {
    __shared__ double As[TS_W][TS_W + 1];
    __shared__ double red[8];
    __shared__ double pivot;
    __shared__ int failed;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    for (int e = tid; e < w * w; e += 256) {
        const int r = e / w, c = e - r * w;
        As[r][c] = A[(long)r * ld + c];
    }
    if (tid == 0) failed = 0;
    __syncthreads();

    for (int j = 0; j < w; ++j) {
        double s = 0.0;
        for (int t = tid; t < j; t += 256) s += As[j][t] * As[j][t];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffffu, s, off);
        if (lane == 0) red[warp] = s;
        __syncthreads();
        if (tid == 0) {
            double tot = 0.0;
            for (int q = 0; q < 8; ++q) tot += red[q];
            const double val = As[j][j] - tot;
            if (val <= 0.0) {
                failed = 1;
                if (*fail < 0.0) *fail = (double)(failBase + j);  // first failure wins
            } else {
                pivot = sqrt(val);
                As[j][j] = pivot;
            }
        }
        __syncthreads();
        if (failed) return;
        const double d = pivot;
        for (int i = j + 1 + tid; i < w; i += 256) {
            double acc = 0.0;
            for (int t = 0; t < j; ++t) acc += As[i][t] * As[j][t];
            As[i][j] = (As[i][j] - acc) / d;
        }
        __syncthreads();
    }
    for (int e = tid; e < w * w; e += 256) {
        const int r = e / w, c = e - r * w;
        A[(long)r * ld + c] = As[r][c];
    }
}

// Pack column block [c0, c0+w) of the local block rows l >= lstart into a compact
// buffer (rows in local order) and also into the panel buffer in global row order
// (panel row index = global row - panelRow0).
__global__ void packPanelKernel(const double* __restrict__ A, long ldA, double* __restrict__ pack,
                                double* __restrict__ panel, long panelRow0, int w, long n, int nb, int rank,
                                int P, long lstart, long nbatch, long c0) {
    long total = nbatch * (long)nb * w;
    for (long idx = blockIdx.x * (long)blockDim.x + threadIdx.x; idx < total; idx += (long)gridDim.x * blockDim.x) {
        long lr = idx / w;  // local row relative to lstart*nb
        int c = (int)(idx - lr * w);
        long l = lstart + lr / nb;
        int t = (int)(lr % nb);
        long i = l * P + rank;
        long r0 = i * (long)nb;
        if (t >= n - r0) continue;
        double v = A[(l * (long)nb + t) * ldA + c0 + c];
        pack[idx] = v;
        panel[(r0 + t - panelRow0) * w + c] = v;
    }
}

// Zero the strict upper triangle in the local block rows.
__global__ void zeroUpperKernel(double* __restrict__ A, long ldA, long n, int nb, int rank, int P, long nloc) {
    long total = nloc * (long)nb * n;
    for (long idx = blockIdx.x * (long)blockDim.x + threadIdx.x; idx < total; idx += (long)gridDim.x * blockDim.x) {
        long lrow = idx / n;
        long c = idx - lrow * n;
        long l = lrow / nb;
        long t = lrow - l * nb;
        long grow = (l * P + rank) * (long)nb + t;
        if (grow >= n) continue;
        if (c > grow) A[idx] = 0.0;
    }
}

// Add n to the diagonal entries in the local block rows.
__global__ void addDiagonalKernel(double* __restrict__ A, long ldA, long n, int nb, int rank, int P, long nloc,
                                  double val) {
    long total = nloc * (long)nb;
    for (long lrow = blockIdx.x * (long)blockDim.x + threadIdx.x; lrow < total; lrow += (long)gridDim.x * blockDim.x) {
        long l = lrow / nb;
        long t = lrow - l * nb;
        long grow = (l * P + rank) * (long)nb + t;
        if (grow >= n) continue;
        A[lrow * ldA + grow] += val;
    }
}

// ---------------------------------------------------------------------------
// Host-side helpers
// ---------------------------------------------------------------------------
static inline unsigned int gridFor(long total, int block) {
    long g = (total + block - 1) / block;
    return (unsigned int)std::min<long>(g, 65535 * 16L);
}

// Everything the distributed solver needs.
//
// Pipeline (step k handles panel k = column block k, rows >= k+1):
//   sPanel : TRSM(k) -> pack (first the block row k+1, the "head") -> D2H
//            -> update of the owned diagonal blocks with panel k (needs only
//            local data) -> factorization of block (k+1,k+1) by its owner
//            -> [head of panel k received] update of column k+1 with panel k
//   host   : nonblocking sends of the packed groups, nonblocking broadcast of
//            L(k+1), receives of panel k-1 drained into dPanel -> trailing
//            update of panel k-1 launched on sUpdate (column k+1 first)
//   sUpdate: trailing updates (off-diagonal blocks only)
// Consecutive panel exchanges therefore overlap each other and the GPU work.
struct DistCholesky {
    Layout lay;
    MPI_Comm comm = MPI_COMM_WORLD;

    static constexpr int NPANEL = 3;
    double* dA = nullptr;                      // local block rows: nloc*nb x n
    double* dPanel[NPANEL] = {nullptr};        // n x nb (global row order)
    double* dPack = nullptr;                   // nloc*nb x nb compact (send staging)
    double* dLkk = nullptr;                    // nb x nb
    double* dFail = nullptr;                   // pivot failure flag (double)
    double* hMsg[2] = {nullptr, nullptr};      // pinned nb*nb+1: factored diagonal block + fail flag
    double* hPack[2] = {nullptr, nullptr};     // pinned nloc*nb x nb
    double* hRecv[2] = {nullptr, nullptr};     // pinned n x nb
    std::vector<cudaEvent_t> evGroup;          // per local block row: "group is in hPack"
    MPI_Request bcastReq = MPI_REQUEST_NULL;

    struct PanelComm {
        bool active = false;
        long k = -1;
        int buf = 0;
        std::vector<MPI_Request> recv;
        std::vector<long> recvBlock;
        size_t ndone = 0;
        std::vector<MPI_Request> send;
        std::vector<int> idx;  // scratch for Testsome/Waitsome
        MPI_Request headReq = MPI_REQUEST_NULL;
    } pc[2];

    cudaStream_t sPanel = nullptr, sUpdate = nullptr, sCopy = nullptr;
    cudaEvent_t evPanel[NPANEL] = {nullptr}, evUpdate[NPANEL] = {nullptr}, evColNext = nullptr, evDiag = nullptr,
                evHead = nullptr, evLkk = nullptr;

    long localRows() const { return lay.nloc * lay.nb; }
    long rowsFrom(long lstart) const {
        long r = 0;
        for (long l = lstart; l < lay.nloc; ++l) r += lay.blockRows(l * lay.nprocs + lay.rank);
        return r;
    }

    void init(long n, int nb, MPI_Comm c) {
        comm = c;
        lay.n = n;
        lay.nb = nb;
        lay.nblocks = (n + nb - 1) / nb;
        MPI_CHECK(MPI_Comm_rank(comm, &lay.rank));
        MPI_CHECK(MPI_Comm_size(comm, &lay.nprocs));
        lay.nloc = (lay.nblocks > lay.rank) ? (lay.nblocks - lay.rank + lay.nprocs - 1) / lay.nprocs : 0;

        const size_t localElems = (size_t)std::max<long>(1, localRows() * n);
        const size_t panelElems = (size_t)std::max<long>(1, n * (long)nb);
        const size_t packElems = (size_t)std::max<long>(1, localRows() * nb);
        CUDA_CHECK(cudaMalloc(&dA, localElems * sizeof(double)));
        CUDA_CHECK(cudaMemset(dA, 0, localElems * sizeof(double)));
        for (int b = 0; b < NPANEL; ++b) CUDA_CHECK(cudaMalloc(&dPanel[b], panelElems * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&dPack, packElems * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&dLkk, (size_t)nb * nb * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&dFail, sizeof(double)));
        for (int b = 0; b < 2; ++b) {
            CUDA_CHECK(cudaMallocHost(&hMsg[b], ((size_t)nb * nb + 1) * sizeof(double)));
            CUDA_CHECK(cudaMallocHost(&hPack[b], packElems * sizeof(double)));
            CUDA_CHECK(cudaMallocHost(&hRecv[b], panelElems * sizeof(double)));
        }

        int prioLo, prioHi;
        CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&prioLo, &prioHi));
        CUDA_CHECK(cudaStreamCreateWithPriority(&sPanel, cudaStreamNonBlocking, prioHi));
        CUDA_CHECK(cudaStreamCreateWithPriority(&sCopy, cudaStreamNonBlocking, prioHi));
        CUDA_CHECK(cudaStreamCreateWithPriority(&sUpdate, cudaStreamNonBlocking, prioLo));
        for (int b = 0; b < NPANEL; ++b) {
            CUDA_CHECK(cudaEventCreateWithFlags(&evPanel[b], cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&evUpdate[b], cudaEventDisableTiming));
        }
        CUDA_CHECK(cudaEventCreateWithFlags(&evColNext, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&evDiag, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&evHead, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&evLkk, cudaEventDisableTiming));
        evGroup.resize(lay.nloc);
        for (auto& e : evGroup) CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
        for (auto& p : pc) {
            p.recv.reserve(lay.nblocks);
            p.recvBlock.reserve(lay.nblocks);
            p.idx.resize(std::max<long>(1, lay.nblocks));
            p.send.reserve((size_t)lay.nloc * lay.nprocs);
        }
    }

    void destroy() {
        cudaFree(dA);
        for (int b = 0; b < NPANEL; ++b) cudaFree(dPanel[b]);
        cudaFree(dPack);
        cudaFree(dLkk);
        cudaFree(dFail);
        for (int b = 0; b < 2; ++b) {
            cudaFreeHost(hMsg[b]);
            cudaFreeHost(hPack[b]);
            cudaFreeHost(hRecv[b]);
        }
        cudaStreamDestroy(sPanel);
        cudaStreamDestroy(sUpdate);
        cudaStreamDestroy(sCopy);
        for (int b = 0; b < NPANEL; ++b) {
            cudaEventDestroy(evPanel[b]);
            cudaEventDestroy(evUpdate[b]);
        }
        cudaEventDestroy(evColNext);
        cudaEventDestroy(evDiag);
        cudaEventDestroy(evHead);
        cudaEventDestroy(evLkk);
        for (auto& e : evGroup) cudaEventDestroy(e);
    }

    // ---- kernel launch helpers -------------------------------------------------
    template <int T>
    void gemmBatched(cudaStream_t s, const double* X, long ldx, long xrow0, bool xLocal, const double* Y, long ldy,
                     long yrow0, double* C, long ldc, long kdim, long lstart, long count, int mode, long c0, long c1,
                     double alpha) {
        if (count <= 0 || kdim <= 0) return;
        long maxCols = (mode == 0) ? (c1 - c0) : ((mode == 1) ? (lay.n - c0) : lay.nb);
        if (maxCols <= 0) return;
        dim3 grid((unsigned)((maxCols + T - 1) / T), (unsigned)((lay.nb + T - 1) / T), (unsigned)count);
        gemmBatchedKernel<T, T><<<grid, 256, 0, s>>>(X, ldx, xrow0, xLocal ? 1 : 0, Y, ldy, yrow0, C, ldc, kdim,
                                                     lay.n, lay.nb, lay.rank, lay.nprocs, lstart, mode, c0, c1, alpha);
        CUDA_CHECK(cudaGetLastError());
    }

    template <int T>
    void gemmPlain(cudaStream_t s, const double* X, long ldx, const double* Y, long ldy, double* C, long ldc, long m,
                   long ncols, long kdim, double alpha) {
        if (m <= 0 || ncols <= 0 || kdim <= 0) return;
        dim3 grid((unsigned)((ncols + T - 1) / T), (unsigned)((m + T - 1) / T));
        gemmPlainKernel<T, T><<<grid, 256, 0, s>>>(X, ldx, Y, ldy, C, ldc, m, ncols, kdim, alpha);
        CUDA_CHECK(cudaGetLastError());
    }

    // X (rows x w, ld ldx) := X * L^{-T} with L lower triangular w x w (ld ldl),
    // blocked in 64-wide column panels.
    void trsmRows(cudaStream_t s, double* X, long ldx, long rows, const double* L, long ldl, int w) {
        if (rows <= 0) return;
        for (int c = 0; c < w; c += TS_W) {
            const int wc = std::min(TS_W, w - c);
            if (c > 0) gemmPlain<32>(s, X, ldx, L + (long)c * ldl, ldl, X + c, ldx, rows, wc, c, -1.0);
            trsmSmallKernel<<<(unsigned)((rows + TS_ROWS - 1) / TS_ROWS), 256, 0, s>>>(X + c, ldx, rows,
                                                                                       L + (long)c * ldl + c, ldl, wc);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    // In-place Cholesky of the w x w block at A (ld ld), blocked with 64-wide panels.
    void potrfBlock(cudaStream_t s, double* A, long ld, int w) {
        for (int c = 0; c < w; c += TS_W) {
            const int wc = std::min(TS_W, w - c);
            double* diag = A + (long)c * ld + c;
            potrf64Kernel<<<1, 256, 0, s>>>(diag, ld, wc, dFail, c);
            CUDA_CHECK(cudaGetLastError());
            const long below = w - (c + wc);
            if (below > 0) {
                double* X = A + (long)(c + wc) * ld + c;
                trsmSmallKernel<<<(unsigned)((below + TS_ROWS - 1) / TS_ROWS), 256, 0, s>>>(X, ld, below, diag, ld, wc);
                CUDA_CHECK(cudaGetLastError());
                gemmPlain<32>(s, X, ld, X, ld, A + (long)(c + wc) * ld + (c + wc), ld, below, below, wc, -1.0);
            }
        }
    }

    // Factor diagonal block k in place (owner only, on sPanel) and download the
    // result plus the failure flag into hMsg[k%2]; records evDiag.
    void factorDiagAsync(long k) {
        const int w = (int)lay.blockRows(k);
        const long l = lay.localIndex(k);
        double* blk = dA + l * lay.nb * lay.n + k * lay.nb;
        double* msg = hMsg[k % 2];
        CUDA_CHECK(cudaMemcpyAsync(dFail, msg + (size_t)lay.nb * lay.nb, sizeof(double), cudaMemcpyHostToDevice, sPanel));
        potrfBlock(sPanel, blk, lay.n, w);
        CUDA_CHECK(cudaMemcpy2DAsync(msg, (size_t)w * sizeof(double), blk, (size_t)lay.n * sizeof(double),
                                     (size_t)w * sizeof(double), (size_t)w, cudaMemcpyDeviceToHost, sPanel));
        CUDA_CHECK(cudaMemcpyAsync(msg + (size_t)w * w, dFail, sizeof(double), cudaMemcpyDeviceToHost, sPanel));
        CUDA_CHECK(cudaEventRecord(evDiag, sPanel));
    }

    // ---- communication helpers --------------------------------------------------
    int tagOf(long k, long i) const { return (int)((k % 2) * lay.nblocks + i); }

    void uploadReceived(PanelComm& p, int outcount) {
        const long kb0 = p.k + 1;
        const int w = (int)lay.blockRows(p.k);
        for (int q = 0; q < outcount; ++q) {
            const long i = p.recvBlock[p.idx[q]];
            CUDA_CHECK(cudaMemcpyAsync(dPanel[p.buf] + (i - kb0) * lay.nb * w, hRecv[p.k % 2] + (i - kb0) * lay.nb * w,
                                       (size_t)lay.blockRows(i) * w * sizeof(double), cudaMemcpyHostToDevice, sCopy));
        }
        p.ndone += outcount;
    }

    // Test outstanding receives of panel comm p and upload completed blocks.
    void progress(PanelComm& p) {
        if (!p.active || p.ndone == p.recv.size()) return;
        int outcount = 0;
        MPI_CHECK(MPI_Testsome((int)p.recv.size(), p.recv.data(), &outcount, p.idx.data(), MPI_STATUSES_IGNORE));
        if (outcount == MPI_UNDEFINED || outcount <= 0) return;
        uploadReceived(p, outcount);
    }

    // Wait for a CUDA event while keeping the panel exchange moving.
    void waitEvent(cudaEvent_t ev) {
        for (;;) {
            cudaError_t st = cudaEventQuery(ev);
            if (st == cudaSuccess) return;
            if (st != cudaErrorNotReady) CUDA_CHECK(st);
            progress(pc[0]);
            progress(pc[1]);
        }
    }

    // Post the receives for panel k (rows >= k+1 of column block k) into hRecv[k%2].
    void postRecvs(long k) {
        PanelComm& p = pc[k % 2];
        const long kb0 = k + 1;
        const int w = (int)lay.blockRows(k);
        p.active = true;
        p.k = k;
        p.buf = (int)(k % NPANEL);
        p.recv.clear();
        p.recvBlock.clear();
        p.ndone = 0;
        p.send.clear();
        p.headReq = MPI_REQUEST_NULL;
        if (lay.nprocs == 1) return;
        for (long i = kb0; i < lay.nblocks; ++i) {
            const int r = lay.owner(i);
            if (r == lay.rank) continue;
            MPI_Request q;
            MPI_CHECK(MPI_Irecv(hRecv[k % 2] + (i - kb0) * lay.nb * w, (int)(lay.blockRows(i) * w), MPI_DOUBLE, r,
                                tagOf(k, i), comm, &q));
            if (i == kb0) {
                p.headReq = q;  // the head block is waited for separately
            } else {
                p.recv.push_back(q);
                p.recvBlock.push_back(i);
            }
        }
    }

    // Finish the exchange of panel k, then launch its trailing update:
    // column k+2 first (needed by the next look-ahead step), then the rest.
    void drainAndUpdate(long k) {
        PanelComm& p = pc[k % 2];
        if (!p.active) return;
        const long kb0 = k + 1;
        const int w = (int)lay.blockRows(k);
        const int buf = p.buf;
        while (p.ndone < p.recv.size()) {
            int outcount = 0;
            MPI_CHECK(MPI_Waitsome((int)p.recv.size(), p.recv.data(), &outcount, p.idx.data(), MPI_STATUSES_IGNORE));
            if (outcount == MPI_UNDEFINED) break;
            uploadReceived(p, outcount);
        }
        CUDA_CHECK(cudaEventRecord(evPanel[buf], sCopy));
        if (!p.send.empty()) MPI_CHECK(MPI_Waitall((int)p.send.size(), p.send.data(), MPI_STATUSES_IGNORE));
        p.active = false;

        // trailing update with panel k on the off-diagonal blocks of rows >= k+2
        CUDA_CHECK(cudaStreamWaitEvent(sUpdate, evPanel[buf], 0));
        const double* P = dPanel[buf];
        const long panelRow0 = kb0 * lay.nb;
        if (k + 2 < lay.nblocks) {
            const long c0 = (k + 2) * lay.nb;
            const long c1 = std::min(c0 + lay.nb, lay.n);
            // column block k+2 (rows k+3..; row k+2 holds only its diagonal block there)
            const long lstart2 = lay.firstLocalAtOrAfter(k + 3);
            gemmBatched<32>(sUpdate, P, w, panelRow0, false, P, w, panelRow0, dA, lay.n, w, lstart2, lay.nloc - lstart2,
                            0, c0, c1, -1.0);
            CUDA_CHECK(cudaEventRecord(evColNext, sUpdate));
            // everything strictly left of the diagonal blocks, columns >= (k+3)*nb
            gemmBatched<64>(sUpdate, P, w, panelRow0, false, P, w, panelRow0, dA, lay.n, w, lstart2, lay.nloc - lstart2,
                            1, c1, 0, -1.0);
        }
        CUDA_CHECK(cudaEventRecord(evUpdate[buf], sUpdate));
    }

    // Wait for the broadcast of diagonal block k, check the pivot flag and
    // upload the block to dLkk.
    bool receiveDiag(long k) {
        const int w = (int)lay.blockRows(k);
        double* msg = hMsg[k % 2];
        MPI_CHECK(MPI_Wait(&bcastReq, MPI_STATUS_IGNORE));
        const double failFlag = msg[(size_t)w * w];
        if (failFlag >= 0.0) {
            if (lay.owner(k) == lay.rank) {
                printf("Error: Matrix is not positive definite at diagonal element %zu\n",
                       (size_t)(k * lay.nb + (long)failFlag));
                fflush(stdout);
            }
            // make sure nothing is still in flight before returning
            for (auto& p : pc)
                if (p.active) {
                    if (!p.recv.empty()) MPI_CHECK(MPI_Waitall((int)p.recv.size(), p.recv.data(), MPI_STATUSES_IGNORE));
                    if (p.headReq != MPI_REQUEST_NULL) MPI_CHECK(MPI_Wait(&p.headReq, MPI_STATUS_IGNORE));
                    if (!p.send.empty()) MPI_CHECK(MPI_Waitall((int)p.send.size(), p.send.data(), MPI_STATUSES_IGNORE));
                    p.active = false;
                }
            CUDA_CHECK(cudaDeviceSynchronize());
            return false;
        }
        CUDA_CHECK(cudaMemcpyAsync(dLkk, msg, (size_t)w * w * sizeof(double), cudaMemcpyHostToDevice, sPanel));
        CUDA_CHECK(cudaEventRecord(evLkk, sPanel));
        return true;
    }

    // ---- the factorization ----------------------------------------------------------
    // Returns false (on all ranks) if the matrix is not positive definite.
    bool factorize() {
        const long nbk = lay.nblocks;
        if (nbk == 0) return true;
        for (int b = 0; b < NPANEL; ++b) {
            CUDA_CHECK(cudaEventRecord(evPanel[b], sCopy));
            CUDA_CHECK(cudaEventRecord(evUpdate[b], sUpdate));
        }
        CUDA_CHECK(cudaEventRecord(evColNext, sUpdate));
        CUDA_CHECK(cudaEventRecord(evLkk, sPanel));

        // block (0,0)
        hMsg[0][(size_t)lay.nb * lay.nb] = -1.0;
        if (lay.owner(0) == lay.rank) {
            factorDiagAsync(0);
            waitEvent(evDiag);
        }
        MPI_CHECK(MPI_Ibcast(hMsg[0], (int)(lay.blockRows(0) * lay.blockRows(0) + 1), MPI_DOUBLE, lay.owner(0), comm,
                             &bcastReq));
        if (!receiveDiag(0)) return false;

        for (long k = 0; k + 1 < nbk; ++k) {
            const int w = (int)lay.blockRows(k);
            const long kb0 = k + 1;
            const int buf = (int)(k % NPANEL);
            const long c0 = k * lay.nb;
            const long panelRow0 = kb0 * lay.nb;
            const long lstart = lay.firstLocalAtOrAfter(kb0);
            const long rows = rowsFrom(lstart);
            double* P = dPanel[buf];

            // host buffers of this parity are free again (panel k-2 was drained);
            // make sure its device uploads have finished before refilling hRecv.
            if (k >= 2) CUDA_CHECK(cudaEventSynchronize(evPanel[(k - 2) % NPANEL]));
            postRecvs(k);

            // dPanel[buf] was last read by the trailing update of panel k-3
            CUDA_CHECK(cudaStreamWaitEvent(sPanel, evUpdate[buf], 0));

            // TRSM of column k on the owned rows below the diagonal block
            if (rows > 0) trsmRows(sPanel, dA + lstart * lay.nb * lay.n + c0, lay.n, rows, dLkk, w, w);

            // pack groups: head (block row k+1, if owned) first, then up to 4 groups
            std::vector<std::pair<long, long>> groups;
            {
                long g0 = lstart;
                if (lay.owner(kb0) == lay.rank && lstart < lay.nloc) {
                    groups.push_back({g0, 1});
                    ++g0;
                }
                const long rest = lay.nloc - g0;
                const long gs = std::max<long>(1, (rest + 3) / 4);
                for (; g0 < lay.nloc; g0 += gs) groups.push_back({g0, std::min(gs, lay.nloc - g0)});
            }
            for (const auto& g : groups) {
                const long g0 = g.first, gcount = g.second;
                double* pack = dPack + (g0 - lstart) * lay.nb * w;
                packPanelKernel<<<gridFor(gcount * lay.nb * (long)w, 256), 256, 0, sPanel>>>(
                    dA, lay.n, pack, P, panelRow0, w, lay.n, lay.nb, lay.rank, lay.nprocs, g0, gcount, c0);
                CUDA_CHECK(cudaGetLastError());
                if (lay.nprocs > 1) {
                    long grows = 0;
                    for (long l = g0; l < g0 + gcount; ++l) grows += lay.blockRows(l * lay.nprocs + lay.rank);
                    CUDA_CHECK(cudaMemcpyAsync(hPack[k % 2] + (g0 - lstart) * lay.nb * w, pack,
                                               (size_t)grows * w * sizeof(double), cudaMemcpyDeviceToHost, sPanel));
                    CUDA_CHECK(cudaEventRecord(evGroup[g0 + gcount - 1], sPanel));
                }
            }

            // owned diagonal blocks (k+1.., local data only), then factor (k+1,k+1)
            gemmBatched<32>(sPanel, P, w, panelRow0, false, P, w, panelRow0, dA, lay.n, w, lstart, lay.nloc - lstart, 2,
                            0, 0, -1.0);
            if (lay.owner(kb0) == lay.rank) {
                CUDA_CHECK(cudaEventSynchronize(evLkk));
                hMsg[kb0 % 2][(size_t)lay.nb * lay.nb] = -1.0;
                factorDiagAsync(kb0);
            }

            // send the packed groups as they land in host memory
            if (lay.nprocs > 1) {
                PanelComm& p = pc[k % 2];
                for (const auto& g : groups) {
                    const long g0 = g.first, gcount = g.second;
                    waitEvent(evGroup[g0 + gcount - 1]);
                    for (long l = g0; l < g0 + gcount; ++l) {
                        const long i = l * lay.nprocs + lay.rank;
                        for (int r = 0; r < lay.nprocs; ++r) {
                            if (r == lay.rank) continue;
                            MPI_Request q;
                            MPI_CHECK(MPI_Isend(hPack[k % 2] + (l - lstart) * lay.nb * w, (int)(lay.blockRows(i) * w),
                                                MPI_DOUBLE, r, tagOf(k, i), comm, &q));
                            p.send.push_back(q);
                        }
                    }
                }
            }

            // broadcast L(k+1)
            if (lay.owner(kb0) == lay.rank) waitEvent(evDiag);
            CUDA_CHECK(cudaEventSynchronize(evLkk));
            MPI_CHECK(MPI_Ibcast(hMsg[kb0 % 2], (int)(lay.blockRows(kb0) * lay.blockRows(kb0) + 1), MPI_DOUBLE,
                                 lay.owner(kb0), comm, &bcastReq));

            // previous panel: finish its exchange and launch its trailing update
            if (k >= 1) drainAndUpdate(k - 1);

            // head of panel k -> update column k+1 of the owned rows >= k+2
            if (k + 2 < nbk) {
                if (lay.owner(kb0) != lay.rank && lay.nprocs > 1) {
                    PanelComm& p = pc[k % 2];
                    MPI_CHECK(MPI_Wait(&p.headReq, MPI_STATUS_IGNORE));
                    CUDA_CHECK(cudaMemcpyAsync(P, hRecv[k % 2], (size_t)lay.blockRows(kb0) * w * sizeof(double),
                                               cudaMemcpyHostToDevice, sCopy));
                    CUDA_CHECK(cudaEventRecord(evHead, sCopy));
                    CUDA_CHECK(cudaStreamWaitEvent(sPanel, evHead, 0));
                }
                CUDA_CHECK(cudaStreamWaitEvent(sPanel, evColNext, 0));
                const long cc0 = kb0 * lay.nb;
                const long cc1 = std::min(cc0 + lay.nb, lay.n);
                const long ls2 = lay.firstLocalAtOrAfter(k + 2);
                gemmBatched<32>(sPanel, P, w, panelRow0, false, P, w, panelRow0, dA, lay.n, w, ls2, lay.nloc - ls2, 0,
                                cc0, cc1, -1.0);
            }

            if (!receiveDiag(kb0)) return false;
        }
        if (nbk >= 2) drainAndUpdate(nbk - 2);

        CUDA_CHECK(cudaStreamSynchronize(sPanel));
        CUDA_CHECK(cudaStreamSynchronize(sCopy));
        CUDA_CHECK(cudaStreamSynchronize(sUpdate));
        if (lay.nloc > 0) {
            zeroUpperKernel<<<gridFor(localRows() * lay.n, 256), 256, 0, sPanel>>>(dA, lay.n, lay.n, lay.nb, lay.rank,
                                                                                   lay.nprocs, lay.nloc);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaStreamSynchronize(sPanel));
        return true;
    }

    // ---- matrix generation / validation (untimed) -------------------------------------
    // Synchronous exchange of column block kc over all block rows into dPanel[0].
    void exchangeColumnBlock(long kc) {
        const int w = (int)lay.blockRows(kc);
        const long c0 = kc * lay.nb;
        if (lay.nloc > 0) {
            packPanelKernel<<<gridFor(lay.nloc * lay.nb * (long)w, 256), 256, 0, sPanel>>>(
                dA, lay.n, dPack, dPanel[0], 0, w, lay.n, lay.nb, lay.rank, lay.nprocs, 0, lay.nloc, c0);
            CUDA_CHECK(cudaGetLastError());
        }
        if (lay.nprocs == 1) {
            CUDA_CHECK(cudaStreamSynchronize(sPanel));
            return;
        }
        const long myRows = rowsFrom(0);
        if (myRows > 0)
            CUDA_CHECK(cudaMemcpyAsync(hPack[0], dPack, (size_t)myRows * w * sizeof(double), cudaMemcpyDeviceToHost,
                                       sPanel));
        CUDA_CHECK(cudaStreamSynchronize(sPanel));
        std::vector<MPI_Request> rq;
        for (long i = 0; i < lay.nblocks; ++i) {
            const int r = lay.owner(i);
            if (r == lay.rank) continue;
            MPI_Request q;
            MPI_CHECK(MPI_Irecv(hRecv[0] + i * lay.nb * w, (int)(lay.blockRows(i) * w), MPI_DOUBLE, r, (int)i, comm, &q));
            rq.push_back(q);
        }
        for (long l = 0; l < lay.nloc; ++l) {
            const long i = l * lay.nprocs + lay.rank;
            for (int r = 0; r < lay.nprocs; ++r) {
                if (r == lay.rank) continue;
                MPI_Request q;
                MPI_CHECK(MPI_Isend(hPack[0] + l * lay.nb * w, (int)(lay.blockRows(i) * w), MPI_DOUBLE, r, (int)i, comm,
                                    &q));
                rq.push_back(q);
            }
        }
        if (!rq.empty()) MPI_CHECK(MPI_Waitall((int)rq.size(), rq.data(), MPI_STATUSES_IGNORE));
        for (long i = 0; i < lay.nblocks; ++i) {
            if (lay.owner(i) == lay.rank) continue;
            CUDA_CHECK(cudaMemcpyAsync(dPanel[0] + i * lay.nb * w, hRecv[0] + i * lay.nb * w,
                                       (size_t)lay.blockRows(i) * w * sizeof(double), cudaMemcpyHostToDevice, sPanel));
        }
        CUDA_CHECK(cudaStreamSynchronize(sPanel));
    }

    // Generate the SPD matrix A = B*B^T + n*I into the local block rows.
    // If hB is non-empty it holds the full matrix B on the host (fallback path),
    // otherwise B is generated on the device.
    void generate(const std::vector<double>& hB) {
        const long n = lay.n;
        if (lay.nloc == 0) return;
        const double randMax = (double)RAND_MAX;

        // my rows of B (local layout)
        double* dBmine = nullptr;
        CUDA_CHECK(cudaMalloc(&dBmine, (size_t)localRows() * n * sizeof(double)));
        for (long l = 0; l < lay.nloc; ++l) {
            const long i = l * lay.nprocs + lay.rank;
            const long rows = lay.blockRows(i);
            double* dst = dBmine + l * lay.nb * n;
            if (hB.empty()) {
                genBRowsKernel<<<gridFor(rows * n, 256), 256, 0, sPanel>>>(dst, i * lay.nb, rows, n, randMax);
                CUDA_CHECK(cudaGetLastError());
            } else {
                CUDA_CHECK(cudaMemcpyAsync(dst, hB.data() + i * lay.nb * n, (size_t)rows * n * sizeof(double),
                                           cudaMemcpyHostToDevice, sPanel));
            }
        }

        // row chunks of B (used as the transposed operand), streamed through a device buffer
        const long chunk = std::max<long>(1, std::min<long>(n, (256L << 20) / (n * 8)));
        double* dBchunk = nullptr;
        CUDA_CHECK(cudaMalloc(&dBchunk, (size_t)chunk * n * sizeof(double)));
        for (long c0 = 0; c0 < n; c0 += chunk) {
            const long rows = std::min(chunk, n - c0);
            if (hB.empty()) {
                genBRowsKernel<<<gridFor(rows * n, 256), 256, 0, sPanel>>>(dBchunk, c0, rows, n, randMax);
                CUDA_CHECK(cudaGetLastError());
            } else {
                CUDA_CHECK(cudaMemcpyAsync(dBchunk, hB.data() + c0 * n, (size_t)rows * n * sizeof(double),
                                           cudaMemcpyHostToDevice, sPanel));
            }
            gemmBatched<64>(sPanel, dBmine, n, 0, true, dBchunk, n, c0, dA, n, n, 0, lay.nloc, 0, c0, c0 + rows, 1.0);
            CUDA_CHECK(cudaStreamSynchronize(sPanel));  // dBchunk is reused
        }
        addDiagonalKernel<<<gridFor(localRows(), 256), 256, 0, sPanel>>>(dA, n, n, lay.nb, lay.rank, lay.nprocs,
                                                                         lay.nloc, (double)n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(sPanel));
        CUDA_CHECK(cudaFree(dBmine));
        CUDA_CHECK(cudaFree(dBchunk));
    }

    // Compute R = L*L^T for the local block rows into dR (device, local layout).
    void reconstruct(double* dR) {
        const long n = lay.n;
        CUDA_CHECK(cudaMemsetAsync(dR, 0, (size_t)std::max<long>(1, localRows() * n) * sizeof(double), sPanel));
        for (long kc = 0; kc < lay.nblocks; ++kc) {
            const int w = (int)lay.blockRows(kc);
            exchangeColumnBlock(kc);  // column block kc over all rows (upper triangle is zero)
            gemmBatched<64>(sPanel, dPanel[0], w, 0, false, dPanel[0], w, 0, dR, n, w, 0, lay.nloc, 0, 0, n, 1.0);
            CUDA_CHECK(cudaStreamSynchronize(sPanel));
        }
    }

    // Download the local block rows to host memory (local layout).
    void download(double* h) const {
        if (localRows() > 0)
            CUDA_CHECK(cudaMemcpy(h, dA, (size_t)localRows() * lay.n * sizeof(double), cudaMemcpyDeviceToHost));
    }
};

// Gather the distributed matrix into a full row-major matrix on rank 0.
static void gatherToRoot(const DistCholesky& dc, const std::vector<double>& hLocal, std::vector<double>& full) {
    const Layout& lay = dc.lay;
    if (lay.rank == 0) full.assign((size_t)lay.n * lay.n, 0.0);
    for (long i = 0; i < lay.nblocks; ++i) {
        const int r = lay.owner(i);
        const long rows = lay.blockRows(i);
        const long cnt = rows * lay.n;
        if (r == 0 && lay.rank == 0) {
            const double* src = hLocal.data() + lay.localIndex(i) * lay.nb * lay.n;
            double* dst = full.data() + i * lay.nb * lay.n;
#pragma omp parallel for schedule(static)
            for (long e = 0; e < cnt; ++e) dst[e] = src[e];
        } else if (lay.rank == 0) {
            MPI_CHECK(MPI_Recv(full.data() + i * lay.nb * lay.n, (int)cnt, MPI_DOUBLE, r, (int)i, dc.comm,
                               MPI_STATUS_IGNORE));
        } else if (lay.rank == r) {
            MPI_CHECK(MPI_Send(hLocal.data() + lay.localIndex(i) * lay.nb * lay.n, (int)cnt, MPI_DOUBLE, 0, (int)i,
                               dc.comm));
        }
    }
}

// Sequential fallback generation of B (exactly the original loop).
static void generateBHost(std::vector<double>& B, long n) {
    B.resize((size_t)n * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < (size_t)n * n; ++i) B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
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
    int provided = 0;
    MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));
    int rank = 0, nprocs = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &nprocs));

    size_t n = 512;
    bool validate = false;
    bool printResults = false;
    int rc = 0;
    bool exitEarly = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            exitEarly = true;
            rc = 0;
            break;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            exitEarly = true;
            rc = 1;
            break;
        }
    }
    if (exitEarly) {
        MPI_Finalize();
        return rc;
    }

    // GPU selection: one GPU per rank on each node.
    MPI_Comm nodeComm;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm));
    int localRank = 0, localSize = 1;
    MPI_CHECK(MPI_Comm_rank(nodeComm, &localRank));
    MPI_CHECK(MPI_Comm_size(nodeComm, &localSize));
    int devCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devCount));
    if (devCount == 0) {
        if (rank == 0) fprintf(stderr, "No CUDA devices found\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % devCount));
    CUDA_CHECK(cudaFree(0));

    // Share host cores between the ranks on a node unless the user decided.
    if (getenv("OMP_NUM_THREADS") == nullptr) {
        int avail = omp_get_num_procs();
        cpu_set_t set;
        if (sched_getaffinity(0, sizeof(set), &set) == 0) avail = std::max(1, CPU_COUNT(&set));
        int threads = std::max(1, avail / std::max(1, localSize));
        omp_set_num_threads(threads);
    }

    int nb = 256;
    if (const char* e = getenv("CHOLESKY_NB")) {
        int v = atoi(e);
        if (v >= 64 && v <= 512 && v % 64 == 0) nb = v;
    }

    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    DistCholesky dc;
    dc.init((long)n, nb, MPI_COMM_WORLD);

    // Generate positive definite matrix
    if (rank == 0) printf("Generating positive definite matrix...\n");
    std::vector<double> hB;  // only used by the fallback path
    const bool fastRand = verifyRandImplementation();
    if (!fastRand) generateBHost(hB, (long)n);
    dc.generate(hB);
    hB.clear();
    hB.shrink_to_fit();

    std::vector<double> hAorig;
    if (validate) {
        hAorig.resize((size_t)std::max<long>(1, dc.localRows() * (long)n));
        dc.download(hAorig.data());  // Save original for validation
    }

    // Perform Cholesky decomposition
    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    fflush(stdout);
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto start = std::chrono::high_resolution_clock::now();

    bool success = dc.factorize();

    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (!success) {
        if (rank == 0) printf("Cholesky decomposition failed\n");
        dc.destroy();
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("Computation time: %ld ms\n", (long)duration.count());

        // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Print results for external validation
    if (printResults) {
        std::vector<double> hL((size_t)std::max<long>(1, dc.localRows() * (long)n));
        dc.download(hL.data());
        std::vector<double> full;
        gatherToRoot(dc, hL, full);
        if (rank == 0) {
            full.resize(n * n);
            print_results(full, "CholeskyL");
        }
    }

    // Validation
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        double* dR = nullptr;
        const size_t elems = (size_t)std::max<long>(1, dc.localRows() * (long)n);
        CUDA_CHECK(cudaMalloc(&dR, elems * sizeof(double)));
        dc.reconstruct(dR);
        std::vector<double> hR(elems);
        if (dc.localRows() > 0)
            CUDA_CHECK(cudaMemcpy(hR.data(), dR, (size_t)dc.localRows() * n * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(dR));

        // Compare with original (only the rows that exist; padding rows of a ragged
        // last block are skipped)
        double maxError = 0.0;
        double relError = 0.0;
        const Layout& lay = dc.lay;
        for (long l = 0; l < lay.nloc; ++l) {
            const long i = l * lay.nprocs + lay.rank;
            const long cnt = lay.blockRows(i) * (long)n;
            const double* r = hR.data() + l * lay.nb * n;
            const double* a = hAorig.data() + l * lay.nb * n;
#pragma omp parallel for schedule(static) reduction(max : maxError, relError)
            for (long e = 0; e < cnt; ++e) {
                const double error = fabs(r[e] - a[e]);
                maxError = std::max(maxError, error);
                const double rel = error / (fabs(a[e]) + 1e-10);
                relError = std::max(relError, rel);
            }
        }
        double g[2] = {maxError, relError}, gmax[2];
        MPI_CHECK(MPI_Allreduce(g, gmax, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));

        bool valid = true;
        if (rank == 0) {
            printf("Max absolute error: %.10e\n", gmax[0]);
            printf("Max relative error: %.10e\n", gmax[1]);
        }
        // Check if error is within tolerance
        if (gmax[1] > 1e-6) {
            if (rank == 0) printf("Validation failed: relative error too large\n");
            valid = false;
        }

        if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        rc = valid ? 0 : 1;
    }

    dc.destroy();
    MPI_Comm_free(&nodeComm);
    MPI_Finalize();
    return rc;
}
