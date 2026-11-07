// Hybrid MPI + OpenMP + CUDA Cholesky decomposition benchmark.
//
// Parallelization strategy
// ------------------------
//  * MPI    : the matrix is distributed in a 1-D block-cyclic fashion over
//             block columns of width NB.  Every rank drives one GPU.  The
//             factorization is a right-looking blocked algorithm: the owner of
//             block column k factors its panel on the GPU and broadcasts it,
//             every rank then applies the rank-NB update to the block columns
//             it owns.  A one-step lookahead overlaps the panel factorization
//             and the (non-blocking) panel broadcast of step k+1 with the
//             trailing update of step k; packing, broadcast and upload of a
//             panel are pipelined in column chunks (device buffers are used
//             directly when MPI is CUDA-aware).
//  * CUDA   : all floating point work (matrix generation, panel factorization,
//             trailing update, validation) runs on the GPU with custom kernels
//             (tiled DGEMM/SYRK, triangular solve, small Cholesky tiles).
//  * OpenMP : host side data movement (transposes, assembling the row-major
//             result, reductions) is multi-threaded.
//
// Semantics are those of the original sequential program: same matrix
// generation (A = B*B^T + n*I with rand_r seed 42), the same lower triangular
// factor L (upper part zeroed) and the same validation / output.

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
#if defined(OPEN_MPI)
#include <mpi-ext.h>
#endif

#include "../common/results_output.hpp"

// ---------------------------------------------------------------------------
// Tunables
// ---------------------------------------------------------------------------
namespace {

constexpr int NB_MAX = 1024;  // upper bound of the runtime panel width (multiple of 64)
constexpr int IB = 32;   // inner block size of the panel factorization
constexpr int TM = 64;   // DGEMM tile rows
constexpr int TN = 64;   // DGEMM tile cols
constexpr int TK = 16;   // DGEMM tile depth
constexpr int GEMM_THREADS = 256;
constexpr int NO_ERROR = 0x7f7f7f7f;  // flag sentinel (cudaMemset 0x7f pattern)

#define CUDA_CHECK(call)                                                            \
    do {                                                                            \
        cudaError_t err__ = (call);                                                 \
        if (err__ != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__,        \
                    __LINE__, cudaGetErrorString(err__));                           \
            MPI_Abort(MPI_COMM_WORLD, 1);                                           \
        }                                                                           \
    } while (0)

#define MPI_CHECK(call)                                                             \
    do {                                                                            \
        int err__ = (call);                                                         \
        if (err__ != MPI_SUCCESS) {                                                 \
            fprintf(stderr, "MPI error %d in %s at %s:%d\n", err__, #call,          \
                    __FILE__, __LINE__);                                            \
            MPI_Abort(MPI_COMM_WORLD, 1);                                           \
        }                                                                           \
    } while (0)

// ---------------------------------------------------------------------------
// CUDA kernels.  All matrices are column-major with explicit leading dimension.
// ---------------------------------------------------------------------------

// C(M x N) = (accumulate ? C : 0) + alpha * A(M x K) * B(N x K)^T for the
// TM x TN tile at (row0, col0).  256 threads, each computes a 4 x 4 block.
__device__ __forceinline__ void gemm_nt_tile(const double* __restrict__ A, int lda,
                                             const double* __restrict__ B, int ldb,
                                             double* __restrict__ C, int ldc, int M,
                                             int N, int K, int row0, int col0,
                                             double alpha, bool accumulate) {
    __shared__ double As[TK][TM];
    __shared__ double Bs[TK][TN];

    const int tid = threadIdx.x;
    const int tx = tid & 15;
    const int ty = tid >> 4;

    double acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = 0.0;

    double ra[4], rb[4];

    auto load_regs = [&](int k0) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int idx = tid + GEMM_THREADS * i;
            const int r = idx & (TM - 1);
            const int kk = idx / TM;
            const int gk = k0 + kk;
            const int gr = row0 + r;
            const int gc = col0 + r;
            ra[i] = (gr < M && gk < K) ? A[(size_t)gk * lda + gr] : 0.0;
            rb[i] = (gc < N && gk < K) ? B[(size_t)gk * ldb + gc] : 0.0;
        }
    };
    auto store_smem = [&]() {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int idx = tid + GEMM_THREADS * i;
            const int r = idx & (TM - 1);
            const int kk = idx / TM;
            As[kk][r] = ra[i];
            Bs[kk][r] = rb[i];
        }
    };

    load_regs(0);
    for (int k0 = 0; k0 < K; k0 += TK) {
        store_smem();
        __syncthreads();
        if (k0 + TK < K) load_regs(k0 + TK);  // prefetch next tile
#pragma unroll
        for (int kk = 0; kk < TK; ++kk) {
            double a[4], b[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                a[i] = As[kk][tx + 16 * i];
                b[i] = Bs[kk][ty + 16 * i];
            }
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }

#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int gc = col0 + ty + 16 * j;
        if (gc >= N) continue;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int gr = row0 + tx + 16 * i;
            if (gr < M) {
                const size_t o = (size_t)gc * ldc + gr;
                const double v = alpha * acc[i][j];
                C[o] = accumulate ? C[o] + v : v;
            }
        }
    }
}

__global__ void __launch_bounds__(GEMM_THREADS)
gemm_nt_kernel(const double* __restrict__ A, int lda, const double* __restrict__ B,
               int ldb, double* __restrict__ C, int ldc, int M, int N, int K,
               double alpha, int accumulate) {
    gemm_nt_tile(A, lda, B, ldb, C, ldc, M, N, K, blockIdx.y * TM, blockIdx.x * TN,
                 alpha, accumulate != 0);
}

// Trailing update of all locally owned block columns p = pStart + blockIdx.z
// with the broadcast panel of step k (packed, rows relative to k*NB, leading
// dimension rowsK):   A(jb) -= Lpanel(rows >= jb*NB) * Lpanel(rows of jb)^T
__global__ void __launch_bounds__(GEMM_THREADS)
trailing_update_kernel(double* __restrict__ panels, const double* __restrict__ dpanel,
                       int n, int NB, int P, int rank, int k, int pStart, int rowsK, int wk) {
    const int p = pStart + blockIdx.z;
    const int jb = p * P + rank;
    const int rowStart = jb * NB;
    const int M = n - rowStart;
    if ((int)(blockIdx.y * TM) >= M) return;  // uniform per block
    const int N = min(NB, n - jb * NB);
    double* C = panels + (size_t)p * n * NB + rowStart;
    const double* A = dpanel + (rowStart - k * NB);
    gemm_nt_tile(A, rowsK, A, rowsK, C, n, M, N, wk, blockIdx.y * TM, blockIdx.x * TN,
                 -1.0, true);
}

constexpr int TRSM_THREADS = 128;  // threads per block of the row solve kernel

// Cholesky of a tw x tw (tw <= IB) diagonal tile (lower part) by a single
// warp (lane t owns row t).  A non positive-definite pivot is recorded in
// *flag (global index) and the factorization stops.
__global__ void potf2_kernel(double* __restrict__ T, int ldt, int tw, int gidx0,
                             int* __restrict__ flag) {
    __shared__ double s[IB][IB + 1];
    const int t = threadIdx.x;
    if (t < tw)
        for (int c = 0; c <= t; ++c) s[t][c] = T[(size_t)c * ldt + t];
    __syncwarp();
    for (int j = 0; j < tw; ++j) {
        double d = s[j][j];
        for (int kk = 0; kk < j; ++kk) d -= s[j][kk] * s[j][kk];
        if (d <= 0.0) {  // uniform: every lane computes the same d
            if (t == 0) atomicMin(flag, gidx0 + j);
            return;
        }
        d = sqrt(d);
        if (t == j) s[j][j] = d;
        if (t > j && t < tw) {
            double v = s[t][j];
            for (int kk = 0; kk < j; ++kk) v -= s[t][kk] * s[j][kk];
            s[t][j] = v / d;
        }
        __syncwarp();
    }
    if (t < tw)
        for (int c = 0; c <= t; ++c) T[(size_t)c * ldt + t] = s[t][c];
}

// R(M x tw) := R * L^{-T}, L lower triangular tw x tw.  One thread per row.
__global__ void __launch_bounds__(TRSM_THREADS)
trsm_rnt_kernel(const double* __restrict__ L, int ldl, double* __restrict__ R, int ldr, int M,
                int tw) {
    __shared__ double s[IB][IB + 1];
    for (int idx = threadIdx.x; idx < IB * IB; idx += blockDim.x) {
        const int r = idx / IB, c = idx % IB;
        s[r][c] = (r < tw && c <= r) ? L[(size_t)c * ldl + r] : 0.0;
    }
    __syncthreads();
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;
    double x[IB];
#pragma unroll
    for (int j = 0; j < IB; ++j) x[j] = (j < tw) ? R[(size_t)j * ldr + row] : 0.0;
#pragma unroll
    for (int j = 0; j < IB; ++j) {
        if (j < tw) {
            double v = x[j];
#pragma unroll
            for (int kk = 0; kk < j; ++kk) v -= x[kk] * s[j][kk];
            x[j] = v / s[j][j];
        }
    }
#pragma unroll
    for (int j = 0; j < IB; ++j)
        if (j < tw) R[(size_t)j * ldr + row] = x[j];
}

// Zero the strictly upper triangular part of the owned block columns.
__global__ void zero_upper_kernel(double* __restrict__ panels, int n, int NB, int P, int rank) {
    const int p = blockIdx.z;
    const int jb = p * P + rank;
    const int c = blockIdx.y;
    if (c >= min(NB, n - jb * NB)) return;
    const int diag = jb * NB + c;
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < diag) panels[(size_t)p * n * NB + (size_t)c * n + row] = 0.0;
}

// A[j][j] += n for owned columns.
__global__ void add_diag_kernel(double* __restrict__ panels, int n, int NB, int P, int rank,
                                int count) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count * NB) return;
    const int p = idx / NB, c = idx % NB;
    const int jb = p * P + rank;
    if (c >= min(NB, n - jb * NB)) return;
    panels[(size_t)p * n * NB + (size_t)c * n + jb * NB + c] += (double)n;
}

// Max absolute / relative error between reconstructed and original panels.
__global__ void error_kernel(const double* __restrict__ recon,
                             const double* __restrict__ orig, int n, int NB, int P, int rank,
                             int count, double* __restrict__ outAbs,
                             double* __restrict__ outRel) {
    __shared__ double sAbs[256], sRel[256];
    double mAbs = 0.0, mRel = 0.0;
    const size_t total = (size_t)count * NB * n;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += (size_t)gridDim.x * blockDim.x) {
        const int p = (int)(idx / ((size_t)NB * n));
        const size_t rem = idx - (size_t)p * NB * n;
        const int c = (int)(rem / n);
        const int jb = p * P + rank;
        if (c >= min(NB, n - jb * NB)) continue;
        const double o = orig[idx];
        const double error = fabs(recon[idx] - o);
        mAbs = fmax(mAbs, error);
        mRel = fmax(mRel, error / (fabs(o) + 1e-10));
    }
    sAbs[threadIdx.x] = mAbs;
    sRel[threadIdx.x] = mRel;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) {
            sAbs[threadIdx.x] = fmax(sAbs[threadIdx.x], sAbs[threadIdx.x + s]);
            sRel[threadIdx.x] = fmax(sRel[threadIdx.x], sRel[threadIdx.x + s]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        outAbs[blockIdx.x] = sAbs[0];
        outRel[blockIdx.x] = sRel[0];
    }
}

// ---------------------------------------------------------------------------
// Host side helpers
// ---------------------------------------------------------------------------

inline int ceilDiv(long a, long b) { return (int)((a + b - 1) / b); }

// Can MPI send/receive directly from device memory?  Overridable with
// CHOLESKY_CUDA_AWARE=0/1; otherwise detected (Open MPI) or assumed absent.
bool mpiIsCudaAware() {
    if (const char* env = getenv("CHOLESKY_CUDA_AWARE")) return atoi(env) != 0;
#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
    return MPIX_Query_cuda_support() != 0;
#else
    return false;
#endif
}

void launchGemmNT(const double* A, int lda, const double* B, int ldb, double* C, int ldc,
                  int M, int N, int K, double alpha, bool accumulate, cudaStream_t cs = 0) {
    if (M <= 0 || N <= 0) return;
    dim3 grid(ceilDiv(N, TN), ceilDiv(M, TM));
    gemm_nt_kernel<<<grid, GEMM_THREADS, 0, cs>>>(A, lda, B, ldb, C, ldc, M, N, K, alpha,
                                                  accumulate ? 1 : 0);
}

// Distributed lower triangular matrix: block column jb (width <= NB) lives on
// rank jb % P at local index jb / P; each local panel is n x NB column-major.
struct DistMatrix {
    int n = 0, NB = 256, P = 1, rank = 0, nblk = 0, count = 0;
    double* d_panels = nullptr;

    int width(int jb) const { return std::min(NB, n - jb * NB); }
    int owner(int jb) const { return jb % P; }
    int local(int jb) const { return jb / P; }
    int global(int p) const { return p * P + rank; }
    double* panel(int p) const { return d_panels + (size_t)p * n * NB; }
    size_t localElems() const { return (size_t)count * n * NB; }
    // first local index with global block > k (or count if none)
    int firstAfter(int k) const {
        if (rank > k) return 0;
        return (k - rank) / P + 1;
    }
};

// Factor the panel of global block column k (owned by this rank) in place:
// L11 = chol(A11), L21 = A21 * L11^{-T}.  Right-looking with inner block IB.
void factorPanel(const DistMatrix& dm, int k, int* d_flag, cudaStream_t cs) {
    const int n = dm.n;
    const int NB = dm.NB;
    const int w = dm.width(k);
    double* base = dm.panel(dm.local(k)) + (size_t)k * NB;  // row k*NB, col 0
    const int rows = n - k * NB;
    for (int jj = 0; jj < w; jj += IB) {
        const int iw = std::min(IB, w - jj);
        double* tile = base + (size_t)jj * n + jj;
        const int Mrows = rows - jj - iw;
        potf2_kernel<<<1, 32, 0, cs>>>(tile, n, iw, k * NB + jj, d_flag);
        if (Mrows > 0)
            trsm_rnt_kernel<<<ceilDiv(Mrows, TRSM_THREADS), TRSM_THREADS, 0, cs>>>(
                tile, n, tile + iw, n, Mrows, iw);
        const int Ncols = w - jj - iw;
        if (Mrows > 0 && Ncols > 0) {
            launchGemmNT(tile + iw, n, tile + iw, n,
                         base + (size_t)(jj + iw) * n + (jj + iw), n, Mrows, Ncols, iw, -1.0,
                         true, cs);
        }
    }
}

// Generate A = B * B^T + n * I (owned block columns only) on the GPU.
// B is generated exactly as in the sequential program.
void generatePositiveDefiniteMatrix(const DistMatrix& dm) {
    const size_t n = (size_t)dm.n;
    const int NB = dm.NB;
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }
    // Column-major copy of B (== transpose of the row-major array)
    std::vector<double> Bcm(n * n);
#pragma omp parallel for schedule(static)
    for (size_t jb = 0; jb < n; jb += 64) {
        for (size_t ib = 0; ib < n; ib += 64) {
            const size_t ie = std::min(n, ib + 64), je = std::min(n, jb + 64);
            for (size_t i = ib; i < ie; ++i)
                for (size_t j = jb; j < je; ++j) Bcm[j * n + i] = B[i * n + j];
        }
    }
    B.clear();
    B.shrink_to_fit();

    if (dm.count == 0) return;

    double* d_B = nullptr;
    CUDA_CHECK(cudaMalloc(&d_B, n * n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_B, Bcm.data(), n * n * sizeof(double), cudaMemcpyHostToDevice));
    for (int p = 0; p < dm.count; ++p) {
        const int jb = dm.global(p);
        // A(:, cols of jb) = B * B(rows of jb, :)^T
        launchGemmNT(d_B, dm.n, d_B + (size_t)jb * NB, dm.n, dm.panel(p), dm.n, dm.n,
                     dm.width(jb), dm.n, 1.0, false);
    }
    add_diag_kernel<<<ceilDiv((long)dm.count * NB, 256), 256>>>(dm.d_panels, dm.n, NB, dm.P,
                                                               dm.rank, dm.count);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_B));
}

// Distributed right-looking Cholesky with one-step lookahead.
// Returns true on success; on failure *badIndex holds the offending diagonal.
//
// Step k (panel k already resident in d_buf[k%2]):
//   1. update the first owned block column after k (lookahead column)
//   2. owner of k+1: factor panel k+1, pack it and start its MPI_Ibcast
//   3. all ranks: trailing update of the remaining owned block columns (GPU,
//      asynchronous) while the host waits for the broadcast of panel k+1 and
//      (if MPI is not CUDA-aware) uploads it on the copy stream.
// Pack, broadcast and upload are pipelined in column chunks.
bool choleskyDecomposition(const DistMatrix& dm, size_t* badIndex, bool cudaAware) {
    const int n = dm.n, NB = dm.NB, P = dm.P, rank = dm.rank, nblk = dm.nblk;
    if (nblk == 0) return true;

    constexpr int CH = 64;                  // columns per communication chunk
    constexpr int MAX_CHUNKS = NB_MAX / CH;

    cudaStream_t cs = nullptr, xs = nullptr;  // compute / copy streams
    CUDA_CHECK(cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&xs, cudaStreamNonBlocking));
    cudaEvent_t evH2D[2], evDone[2], evPack[MAX_CHUNKS];
    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaEventCreateWithFlags(&evH2D[b], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&evDone[b], cudaEventDisableTiming));
    }
    for (int c = 0; c < MAX_CHUNKS; ++c)
        CUDA_CHECK(cudaEventCreateWithFlags(&evPack[c], cudaEventDisableTiming));

    int* d_flag = nullptr;
    int* h_flag = nullptr;
    CUDA_CHECK(cudaMalloc(&d_flag, sizeof(int)));
    CUDA_CHECK(cudaMallocHost(&h_flag, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_flag, 0x7f, sizeof(int)));

    const size_t panelElems = (size_t)n * NB;
    double* h_buf[2] = {nullptr, nullptr};
    double* d_buf[2] = {nullptr, nullptr};
    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaMalloc(&d_buf[b], panelElems * sizeof(double)));
        if (!cudaAware) CUDA_CHECK(cudaMallocHost(&h_buf[b], panelElems * sizeof(double)));
    }
    auto commBuf = [&](int k) { return cudaAware ? d_buf[k % 2] : h_buf[k % 2]; };
    auto rowsOf = [&](int k) { return n - k * NB; };
    auto nChunks = [&](int k) { return (dm.width(k) + CH - 1) / CH; };
    auto chunkCols = [&](int k, int c) { return std::min(CH, dm.width(k) - c * CH); };

    auto packPanel = [&](int k) {
        // Owner: factor panel k and pack rows [k*NB, n) chunk by chunk into
        // the broadcast buffer.  evPack[c] marks completion of chunk c.
        factorPanel(dm, k, d_flag, cs);
        CUDA_CHECK(cudaMemcpyAsync(h_flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost, cs));
        const size_t rows = (size_t)rowsOf(k);
        const double* src = dm.panel(dm.local(k)) + (size_t)k * NB;
        if (cudaAware) CUDA_CHECK(cudaStreamWaitEvent(cs, evDone[k % 2], 0));
        for (int c = 0; c < nChunks(k); ++c) {
            CUDA_CHECK(cudaMemcpy2DAsync(
                commBuf(k) + (size_t)c * CH * rows, rows * sizeof(double),
                src + (size_t)c * CH * n, (size_t)n * sizeof(double), rows * sizeof(double),
                chunkCols(k, c), cudaAware ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost,
                cs));
            CUDA_CHECK(cudaEventRecord(evPack[c], cs));
        }
    };

    auto launchUpdate = [&](int k, int pStart, int zcount) {
        if (zcount <= 0) return;
        const int maxM = n - dm.global(pStart) * NB;
        dim3 grid(ceilDiv(NB, TN), ceilDiv(maxM, TM), zcount);
        trailing_update_kernel<<<grid, GEMM_THREADS, 0, cs>>>(
            dm.d_panels, d_buf[k % 2], n, NB, P, rank, k, pStart, rowsOf(k), dm.width(k));
    };

    auto finish = [&](bool ok) {
        CUDA_CHECK(cudaDeviceSynchronize());
        for (int b = 0; b < 2; ++b) {
            if (h_buf[b]) CUDA_CHECK(cudaFreeHost(h_buf[b]));
            CUDA_CHECK(cudaFree(d_buf[b]));
            CUDA_CHECK(cudaEventDestroy(evH2D[b]));
            CUDA_CHECK(cudaEventDestroy(evDone[b]));
        }
        for (int c = 0; c < MAX_CHUNKS; ++c) CUDA_CHECK(cudaEventDestroy(evPack[c]));
        CUDA_CHECK(cudaFree(d_flag));
        CUDA_CHECK(cudaFreeHost(h_flag));
        CUDA_CHECK(cudaStreamDestroy(cs));
        CUDA_CHECK(cudaStreamDestroy(xs));
        return ok;
    };

    auto checkError = [&](int localFlag) {
        int globalFlag = NO_ERROR;
        MPI_CHECK(MPI_Allreduce(&localFlag, &globalFlag, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD));
        if (globalFlag != NO_ERROR) {
            *badIndex = (size_t)globalFlag;
            return true;
        }
        return false;
    };

    // Post the broadcast of panel k (owner: as soon as each chunk is packed).
    auto postBcast = [&](int k, bool isOwner, MPI_Request* req) {
        const size_t rows = (size_t)rowsOf(k);
        for (int c = 0; c < nChunks(k); ++c) {
            if (isOwner) CUDA_CHECK(cudaEventSynchronize(evPack[c]));
            MPI_CHECK(MPI_Ibcast(commBuf(k) + (size_t)c * CH * rows,
                                 (int)(rows * chunkCols(k, c)), MPI_DOUBLE, dm.owner(k),
                                 MPI_COMM_WORLD, &req[c]));
        }
    };
    // Wait for the broadcast of panel k and upload it (copy stream).
    auto receivePanel = [&](int k, MPI_Request* req) {
        const size_t rows = (size_t)rowsOf(k);
        if (!cudaAware) CUDA_CHECK(cudaStreamWaitEvent(xs, evDone[k % 2], 0));
        for (int c = 0; c < nChunks(k); ++c) {
            MPI_CHECK(MPI_Wait(&req[c], MPI_STATUS_IGNORE));
            if (!cudaAware) {
                CUDA_CHECK(cudaMemcpyAsync(d_buf[k % 2] + (size_t)c * CH * rows,
                                           h_buf[k % 2] + (size_t)c * CH * rows,
                                           rows * chunkCols(k, c) * sizeof(double),
                                           cudaMemcpyHostToDevice, xs));
            }
        }
        CUDA_CHECK(cudaEventRecord(evH2D[k % 2], xs));
    };

    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaEventRecord(evDone[b], cs));
        CUDA_CHECK(cudaEventRecord(evH2D[b], xs));
    }
    MPI_Request req[MAX_CHUNKS];

    // Prologue: panel 0
    {
        int flag = NO_ERROR;
        const bool isOwner = (dm.owner(0) == rank);
        if (isOwner) {
            packPanel(0);
            CUDA_CHECK(cudaEventSynchronize(evPack[0]));
            flag = *h_flag;
        }
        if (checkError(flag)) return finish(false);
        postBcast(0, isOwner, req);
        receivePanel(0, req);
    }

    for (int k = 0; k < nblk; ++k) {
        const int cur = k % 2, nxt = (k + 1) % 2;
        const bool hasNext = (k + 1 < nblk);

        // panel k must be on the device before the compute stream uses it
        CUDA_CHECK(cudaStreamWaitEvent(cs, evH2D[cur], 0));

        const int p1 = dm.firstAfter(k);  // first owned block column after k
        launchUpdate(k, p1, (p1 < dm.count) ? 1 : 0);

        int flag = NO_ERROR;
        const bool isOwner = hasNext && (dm.owner(k + 1) == rank);
        if (isOwner) packPanel(k + 1);
        // remaining trailing update: enqueued now, runs during the communication
        launchUpdate(k, p1 + 1, dm.count - (p1 + 1));
        CUDA_CHECK(cudaEventRecord(evDone[cur], cs));
        CUDA_CHECK(cudaGetLastError());

        if (hasNext) {
            if (isOwner) {
                CUDA_CHECK(cudaEventSynchronize(evPack[0]));  // flag copy precedes chunk 0
                flag = *h_flag;
            } else {
                // the receive buffer of panel k+1 must no longer be in use
                CUDA_CHECK(cudaEventSynchronize(cudaAware ? evDone[nxt] : evH2D[nxt]));
            }
            postBcast(k + 1, isOwner, req);
        }

        const bool failed = checkError(flag);
        if (hasNext) receivePanel(k + 1, req);
        if (failed) return finish(false);
    }
    return finish(true);
}

// Gather all block columns to every rank (host memory, panels padded to NB).
std::vector<double> allgatherPanels(const DistMatrix& dm) {
    const int NB = dm.NB;
    std::vector<int> counts(dm.P), displs(dm.P);
    for (int r = 0; r < dm.P; ++r) {
        const int c = (dm.nblk > r) ? (dm.nblk - r + dm.P - 1) / dm.P : 0;
        counts[r] = (int)((size_t)c * dm.n * NB);
        displs[r] = (r == 0) ? 0 : displs[r - 1] + counts[r - 1];
    }
    std::vector<double> local(dm.localElems());
    if (!local.empty())
        CUDA_CHECK(cudaMemcpy(local.data(), dm.d_panels, local.size() * sizeof(double),
                              cudaMemcpyDeviceToHost));
    std::vector<double> all((size_t)dm.nblk * dm.n * NB);
    MPI_CHECK(MPI_Allgatherv(local.data(), (int)local.size(), MPI_DOUBLE, all.data(),
                             counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD));
    return all;
}

// Pointer to the (gathered, host) block column jb.
const double* gatheredBlock(const DistMatrix& dm, const std::vector<double>& all, int jb) {
    const int NB = dm.NB;
    size_t off = 0;
    for (int r = 0; r < dm.owner(jb); ++r) {
        const int c = (dm.nblk > r) ? (dm.nblk - r + dm.P - 1) / dm.P : 0;
        off += (size_t)c * dm.n * NB;
    }
    return all.data() + off + (size_t)dm.local(jb) * dm.n * NB;
}

bool validateCholesky(const DistMatrix& dm, const std::vector<double>& allL,
                      const double* d_orig) {
    // Validate by computing L * L^T (owned columns) and comparing with the
    // original matrix; the maximum errors are reduced over all ranks.
    const int n = dm.n;
    const int NB = dm.NB;
    double localAbs = 0.0, localRel = 0.0;
    if (dm.count > 0 && n > 0) {
        double* d_L = nullptr;
        double* d_recon = nullptr;
        CUDA_CHECK(cudaMalloc(&d_L, (size_t)n * n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_recon, dm.localElems() * sizeof(double)));
        for (int jb = 0; jb < dm.nblk; ++jb) {
            CUDA_CHECK(cudaMemcpy(d_L + (size_t)jb * NB * n, gatheredBlock(dm, allL, jb),
                                  (size_t)n * dm.width(jb) * sizeof(double),
                                  cudaMemcpyHostToDevice));
        }
        for (int p = 0; p < dm.count; ++p) {
            const int jb = dm.global(p);
            launchGemmNT(d_L, n, d_L + (size_t)jb * NB, n, d_recon + (size_t)p * n * NB, n, n,
                         dm.width(jb), n, 1.0, false);
        }
        const int nblocks = 1024;
        double *d_abs = nullptr, *d_rel = nullptr;
        CUDA_CHECK(cudaMalloc(&d_abs, nblocks * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_rel, nblocks * sizeof(double)));
        error_kernel<<<nblocks, 256>>>(d_recon, d_orig, n, NB, dm.P, dm.rank, dm.count, d_abs, d_rel);
        CUDA_CHECK(cudaGetLastError());
        std::vector<double> hAbs(nblocks), hRel(nblocks);
        CUDA_CHECK(cudaMemcpy(hAbs.data(), d_abs, nblocks * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hRel.data(), d_rel, nblocks * sizeof(double), cudaMemcpyDeviceToHost));
#pragma omp parallel for reduction(max : localAbs, localRel)
        for (int i = 0; i < nblocks; ++i) {
            localAbs = std::max(localAbs, hAbs[i]);
            localRel = std::max(localRel, hRel[i]);
        }
        CUDA_CHECK(cudaFree(d_abs));
        CUDA_CHECK(cudaFree(d_rel));
        CUDA_CHECK(cudaFree(d_recon));
        CUDA_CHECK(cudaFree(d_L));
    }
    double maxError = 0.0, relError = 0.0;
    MPI_CHECK(MPI_Allreduce(&localAbs, &maxError, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Allreduce(&localRel, &relError, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));

    if (dm.rank == 0) {
        printf("Max absolute error: %.10e\n", maxError);
        printf("Max relative error: %.10e\n", relError);
    }

    // Check if error is within tolerance
    if (relError > 1e-6) {
        if (dm.rank == 0) printf("Validation failed: relative error too large\n");
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

}  // namespace

int main(int argc, char** argv) {
    MPI_CHECK(MPI_Init(&argc, &argv));
    int rank = 0, P = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &P));

    size_t n = 512;
    bool validate = false;
    bool printResults = false;
    int exitCode = 0;

    // Parse command line arguments (identical on all ranks)
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    // One GPU per rank: pick by node-local rank
    {
        MPI_Comm nodeComm;
        MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
                                      &nodeComm));
        int localRank = 0;
        MPI_CHECK(MPI_Comm_rank(nodeComm, &localRank));
        MPI_CHECK(MPI_Comm_free(&nodeComm));
        int ndev = 0;
        CUDA_CHECK(cudaGetDeviceCount(&ndev));
        if (ndev < 1) {
            fprintf(stderr, "No CUDA device available\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        CUDA_CHECK(cudaSetDevice(localRank % ndev));
        CUDA_CHECK(cudaFree(0));
    }

    DistMatrix dm;
    dm.n = (int)n;
    dm.P = P;
    dm.rank = rank;
    {
        // Panel width: large enough for an efficient rank-NB update, small
        // enough to keep the (serial) panel factorization off the critical path.
        const char* env = getenv("CHOLESKY_NB");
        int nb = env ? atoi(env) : 256;
        nb = std::max(64, std::min(NB_MAX, (nb / 64) * 64));
        dm.NB = nb;
    }
    const int NB = dm.NB;
    const bool cudaAware = mpiIsCudaAware();
    dm.nblk = (int)((n + NB - 1) / NB);
    dm.count = (dm.nblk > rank) ? (dm.nblk - rank + P - 1) / P : 0;

    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, OpenMP threads: %d, panel width: %d, CUDA-aware MPI: %s\n", P,
               omp_get_max_threads(), dm.NB, cudaAware ? "yes" : "no");
    }
    if (dm.count > 0) CUDA_CHECK(cudaMalloc(&dm.d_panels, dm.localElems() * sizeof(double)));

    // Generate positive definite matrix
    if (rank == 0) printf("Generating positive definite matrix...\n");
    generatePositiveDefiniteMatrix(dm);

    double* d_orig = nullptr;
    if (validate && dm.count > 0) {
        CUDA_CHECK(cudaMalloc(&d_orig, dm.localElems() * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_orig, dm.d_panels, dm.localElems() * sizeof(double),
                              cudaMemcpyDeviceToDevice));
    }

    // Perform Cholesky decomposition
    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto start = std::chrono::high_resolution_clock::now();

    size_t badIndex = 0;
    bool success = choleskyDecomposition(dm, &badIndex, cudaAware);

    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (!success) {
        if (rank == 0) {
            printf("Error: Matrix is not positive definite at diagonal element %zu\n", badIndex);
            printf("Cholesky decomposition failed\n");
        }
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

    if (printResults || validate) {
        if (dm.count > 0) {
            dim3 grid(ceilDiv(n, 256), NB, dm.count);
            zero_upper_kernel<<<grid, 256>>>(dm.d_panels, dm.n, NB, P, rank);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        std::vector<double> allL = allgatherPanels(dm);

        // Print results for external validation
        if (printResults && rank == 0) {
            std::vector<double> A(n * n);
            std::vector<const double*> blocks(dm.nblk);
            for (int jb = 0; jb < dm.nblk; ++jb) blocks[jb] = gatheredBlock(dm, allL, jb);
#pragma omp parallel for schedule(static)
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    const int jb = (int)(j / NB);
                    const size_t c = j % NB;
                    A[i * n + j] = blocks[jb][c * n + i];
                }
            }
            print_results(A, "CholeskyL");
        }

        // Validation
        if (validate) {
            if (rank == 0) printf("Validating result...\n");
            bool valid = validateCholesky(dm, allL, d_orig);
            if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            exitCode = valid ? 0 : 1;
        }
    }

    if (d_orig) CUDA_CHECK(cudaFree(d_orig));
    if (dm.d_panels) CUDA_CHECK(cudaFree(dm.d_panels));
    MPI_Finalize();
    return exitCode;
}
