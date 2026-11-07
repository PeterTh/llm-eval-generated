// Hybrid MPI + OpenMP + CUDA Cholesky decomposition benchmark.
//
// Decomposes a symmetric positive definite matrix A (row-major) in place into
// L * L^T, leaving the lower triangular factor L in A and zeroing the strictly
// upper triangular part - exactly like the original sequential program.
//
// Parallel design
//   * MPI   : block rows of A (block size NB) are distributed block-cyclically
//             over the ranks.  Each rank keeps its block rows as a contiguous
//             local matrix (nloc x n, row-major) in GPU memory.
//   * CUDA  : the heavy numerical work (triangular solves of the panel, the
//             trailing symmetric rank-NB update, matrix generation, validation
//             reconstruction) runs in custom CUDA kernels on the GPU assigned to
//             the rank (round-robin over the devices visible on the node).
//   * OpenMP: the latency-critical factorization of the NB x NB diagonal block
//             runs multithreaded on the host (it would otherwise be starved on
//             a GPU busy with the trailing update), as does host-side data
//             reordering and reductions.
//
// Right-looking blocked algorithm, for every block column k:
//   1. owner rank factorizes the diagonal block L_kk on the host (OpenMP)
//   2. L_kk is broadcast to all ranks
//   3. every rank computes L_ik = A_ik L_kk^{-T} for its block rows i > k
//   4. the block column L_{*k} is exchanged (MPI shared memory window when
//      all ranks share a node, host-staged MPI_Allgatherv otherwise)
//   5. every rank updates its rows of the trailing matrix on the GPU:
//      A_ij -= L_ik L_jk^T   (k < j <= i, lower triangle only)

#define OMPI_SKIP_MPICXX 1
#define MPICH_SKIP_MPICXX 1
#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
static constexpr int NB = 64;   // distribution / panel block size (tuned: small keeps the
                                // lookahead panel chain short, see choleskyDecomposition)
static constexpr int IB = 64;   // inner block size of the panel factorization
static constexpr int TM = 64;   // GEMM tile (rows)
static constexpr int TN = 64;   // GEMM tile (cols)
static constexpr int TK = 16;   // GEMM tile (depth)
static constexpr int GEMM_THREADS = 256;

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err__ = (call);                                               \
        if (err__ != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err__), \
                    __FILE__, __LINE__);                                          \
            MPI_Abort(MPI_COMM_WORLD, 1);                                         \
        }                                                                         \
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
// Block-cyclic row distribution helpers
// ---------------------------------------------------------------------------
struct Dist {
    int P;          // number of ranks
    int r;          // this rank
    size_t n;       // matrix size
    int nblk;       // number of global block rows
    int nlocblk;    // number of local block rows
    size_t nloc;    // number of local rows
};

static inline size_t blockRows(const Dist& d, int I) {
    return std::min<size_t>(NB, d.n - (size_t)I * NB);
}
// number of local block rows of rank q
static inline int localBlocks(const Dist& d, int q) {
    return (d.nblk > q) ? (d.nblk - q + d.P - 1) / d.P : 0;
}
static inline size_t localRows(const Dist& d, int q) {
    size_t rows = 0;
    for (int lb = 0; lb < localBlocks(d, q); ++lb) rows += blockRows(d, lb * d.P + q);
    return rows;
}
// first local row of rank q whose global block index is > k
static inline size_t firstLocalRowAfter(const Dist& d, int q, int k) {
    int lb = k / d.P + (q > k % d.P ? 0 : 1);
    return (size_t)lb * NB;
}

// Reorder a buffer that holds the local matrices of all ranks back to back
// (rank 0's rows, rank 1's rows, ...) into global row order.  Rows have
// `ncols` entries.  Used for the final gather and for validation.
static void blockCyclicToGlobal(const Dist& d, const double* staged, const std::vector<size_t>& rankRowOffset,
                                double* global, size_t ncols) {
#pragma omp parallel for schedule(dynamic)
    for (int I = 0; I < d.nblk; ++I) {
        const int q = I % d.P;
        const int lb = I / d.P;
        const size_t rows = blockRows(d, I);
        const double* src = staged + (rankRowOffset[q] + (size_t)lb * NB) * ncols;
        double* dst = global + (size_t)I * NB * ncols;
        memcpy(dst, src, rows * ncols * sizeof(double));
    }
}

// ---------------------------------------------------------------------------
// CUDA kernels
// ---------------------------------------------------------------------------

// C[M x N] (op)= A[M x K] * B[N x K]^T      (op is "=" or "-=")
// All matrices row-major with the given leading dimensions.
// If `skip` is set, tiles that lie entirely in the strictly upper triangle
// (global column > global row for all elements) are not computed; the global
// row of C's local row i is derived from the block-cyclic distribution.
template <bool SUBTRACT>
__global__ void __launch_bounds__(GEMM_THREADS)
gemm_nt_kernel(int M, int N, int K, const double* __restrict__ A, size_t lda, const double* __restrict__ B,
               size_t ldb, double* __restrict__ C, size_t ldc, int skip, size_t rowbase, int P, int r,
               size_t colbase) {
    const int m0 = blockIdx.y * TM;
    const int n0 = blockIdx.x * TN;

    if (skip) {
        const size_t i = rowbase + (size_t)min(m0 + TM - 1, M - 1);
        const size_t g = ((i / NB) * P + r) * (size_t)NB + i % NB;
        if (colbase + (size_t)n0 > g) return;
    }

    __shared__ double As[TK][TM + 1];
    __shared__ double Bs[TK][TN + 1];

    const int tid = threadIdx.x;
    const int ty = tid >> 4;   // 0..15
    const int tx = tid & 15;   // 0..15

    double acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = 0.0;

    for (int k0 = 0; k0 < K; k0 += TK) {
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int idx = tid + l * GEMM_THREADS;
            const int mm = idx >> 4;
            const int kk = idx & 15;
            const int gm = m0 + mm;
            const int gk = k0 + kk;
            As[kk][mm] = (gm < M && gk < K) ? A[(size_t)gm * lda + gk] : 0.0;
            const int gn = n0 + mm;
            Bs[kk][mm] = (gn < N && gk < K) ? B[(size_t)gn * ldb + gk] : 0.0;
        }
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < TK; ++kk) {
            double a[4], b[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) a[i] = As[kk][ty + 16 * i];
#pragma unroll
            for (int j = 0; j < 4; ++j) b[j] = Bs[kk][tx + 16 * j];
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] = fma(a[i], b[j], acc[i][j]);
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int row = m0 + ty + 16 * i;
        if (row >= M) continue;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int col = n0 + tx + 16 * j;
            if (col >= N) continue;
            double* c = C + (size_t)row * ldc + col;
            if (SUBTRACT)
                *c -= acc[i][j];
            else
                *c = acc[i][j];
        }
    }
}

// Solve X * L^T = B in place for M rows (X overwrites B), where L is a small
// m x m (m <= IB) lower triangular block.  One thread per row.
__global__ void trsm_small_kernel(const double* __restrict__ L, size_t ldl, int m, double* __restrict__ X,
                                  size_t ldx, int M) {
    __shared__ double s[IB][IB + 1];
    for (int idx = threadIdx.x; idx < m * m; idx += blockDim.x) {
        const int i = idx / m, c = idx % m;
        if (c <= i) s[i][c] = L[(size_t)i * ldl + c];
    }
    __syncthreads();

    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;
    double* x = X + (size_t)row * ldx;

    double v[IB];
#pragma unroll
    for (int t = 0; t < IB; ++t) v[t] = (t < m) ? x[t] : 0.0;

#pragma unroll
    for (int c = 0; c < IB; ++c) {
        if (c < m) {
            double acc = v[c];
#pragma unroll
            for (int t = 0; t < c; ++t) acc -= v[t] * s[c][t];
            v[c] = acc / s[c][c];
        }
    }
#pragma unroll
    for (int t = 0; t < IB; ++t)
        if (t < m) x[t] = v[t];
}

// Zero the strictly upper triangular part of the local block-cyclic matrix.
__global__ void zero_upper_kernel(double* __restrict__ A, size_t nloc, size_t n, int P, int r) {
    const size_t col = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t i = blockIdx.y;
    if (i >= nloc || col >= n) return;
    const size_t g = ((i / NB) * P + r) * (size_t)NB + i % NB;
    if (col > g) A[i * n + col] = 0.0;
}

// Add n to the diagonal of the local rows (matrix generation).
__global__ void add_diag_kernel(double* __restrict__ A, size_t nloc, size_t n, int P, int r) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nloc) return;
    const size_t g = ((i / NB) * P + r) * (size_t)NB + i % NB;
    A[i * n + g] += (double)n;
}

// Per-block maximum absolute / relative error between two local matrices.
__global__ void error_kernel(const double* __restrict__ R, const double* __restrict__ A, size_t count,
                             double* __restrict__ partialAbs, double* __restrict__ partialRel) {
    __shared__ double sAbs[256];
    __shared__ double sRel[256];
    double mAbs = 0.0, mRel = 0.0;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < count; i += (size_t)gridDim.x * blockDim.x) {
        const double err = fabs(R[i] - A[i]);
        mAbs = fmax(mAbs, err);
        mRel = fmax(mRel, err / (fabs(A[i]) + 1e-10));
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
        partialAbs[blockIdx.x] = sAbs[0];
        partialRel[blockIdx.x] = sRel[0];
    }
}

// ---------------------------------------------------------------------------
// Kernel launch helpers
// ---------------------------------------------------------------------------
template <bool SUBTRACT>
static void gemm_nt(int M, int N, int K, const double* A, size_t lda, const double* B, size_t ldb, double* C,
                    size_t ldc, cudaStream_t stream, bool skip = false, size_t rowbase = 0, int P = 1, int r = 0,
                    size_t colbase = 0) {
    if (M <= 0 || N <= 0 || K <= 0) return;
    dim3 grid((N + TN - 1) / TN, (M + TM - 1) / TM);
    gemm_nt_kernel<SUBTRACT><<<grid, GEMM_THREADS, 0, stream>>>(M, N, K, A, lda, B, ldb, C, ldc, skip ? 1 : 0,
                                                                rowbase, P, r, colbase);
    CUDA_CHECK(cudaGetLastError());
}

// Blocked triangular solve X * L^T = X for M rows, where L is the bk x bk lower
// triangular block (leading dimension ldl) and X has leading dimension ldx.
static void trsm_panel(const double* L, size_t ldl, int bk, double* X, size_t ldx, int M, cudaStream_t stream) {
    if (M <= 0) return;
    for (int j = 0; j < bk; j += IB) {
        const int jb = std::min(IB, bk - j);
        const int blocks = (M + 63) / 64;
        trsm_small_kernel<<<blocks, 64, 0, stream>>>(L + (size_t)j * ldl + j, ldl, jb, X + j, ldx, M);
        CUDA_CHECK(cudaGetLastError());
        if (j + jb < bk) {
            // X[:, j+jb..bk] -= X[:, j..j+jb] * L[j+jb..bk, j..j+jb]^T
            gemm_nt<true>(M, bk - j - jb, jb, X + j, ldx, L + (size_t)(j + jb) * ldl + j, ldl, X + j + jb, ldx,
                          stream);
        }
    }
}

// In-place Cholesky factorization of the bk x bk diagonal block D (leading
// dimension ld) on the host, multithreaded with OpenMP.  Only the lower
// triangle is read and written.  Returns false on a non-positive pivot and
// stores its global diagonal index in failIndex.
static bool potrf_host(double* D, int bk, size_t ld, size_t gdiag, size_t& failIndex) {
    std::vector<double> yt((size_t)IB * bk);
    const int nthreads = std::max(1, std::min(omp_get_max_threads(), 8));
    for (int j0 = 0; j0 < bk; j0 += IB) {
        const int jb = std::min(IB, bk - j0);
        // unblocked (row oriented) factorization of the jb x jb diagonal sub-block
        for (int i = j0; i < j0 + jb; ++i) {
            double* rowi = D + (size_t)i * ld;
            for (int c = j0; c <= i; ++c) {
                const double* rowc = D + (size_t)c * ld;
                double sum = 0.0;
                for (int t = j0; t < c; ++t) sum += rowi[t] * rowc[t];
                if (c == i) {
                    const double v = rowi[i] - sum;
                    if (v <= 0.0) {
                        failIndex = gdiag + i;
                        return false;
                    }
                    rowi[i] = sqrt(v);
                } else {
                    rowi[c] = (rowi[c] - sum) / rowc[c];
                }
            }
        }
        const int rem = bk - j0 - jb;
        if (rem <= 0) continue;
        const double* Ljj = D + (size_t)j0 * ld + j0;
        // triangular solves of the rows below: x * Ljj^T = x
#pragma omp parallel for schedule(static) num_threads(nthreads)
        for (int i = j0 + jb; i < bk; ++i) {
            double* x = D + (size_t)i * ld + j0;
            for (int c = 0; c < jb; ++c) {
                double acc = x[c];
                const double* lc = Ljj + (size_t)c * ld;
                for (int t = 0; t < c; ++t) acc -= x[t] * lc[t];
                x[c] = acc / lc[c];
            }
        }
        // symmetric rank-jb update of the trailing lower triangle:
        // D[i][c] -= sum_t D[i][j0+t] * D[c][j0+t].  The solved rows are
        // packed transposed (Yt[t][c]) so the inner loop is contiguous.
        double* Yt = yt.data();
#pragma omp parallel for schedule(static) num_threads(nthreads)
        for (int c = 0; c < rem; ++c) {
            const double* y = D + (size_t)(j0 + jb + c) * ld + j0;
            for (int t = 0; t < jb; ++t) Yt[(size_t)t * rem + c] = y[t];
        }
        // 4 x 4 register tiles: acc[q][cc] = sum_t x_q[t] * Yt[t][c0 + cc].
        // Columns past the diagonal of a row lie in the (unused) upper
        // triangle and may be computed freely.
        const int ntiles = (rem + 3) / 4;
#pragma omp parallel for schedule(dynamic, 1) num_threads(nthreads)
        for (int tile = 0; tile < ntiles; ++tile) {
            const int i0 = j0 + jb + tile * 4;
            const int nr = std::min(4, bk - i0);
            const int cmax = i0 + nr - (j0 + jb);  // columns 0 .. cmax-1 (relative to j0 + jb)
            for (int c0 = 0; c0 < cmax; c0 += 4) {
                const int nc = std::min(4, rem - c0);
                if (nr == 4 && nc == 4) {
                    double acc[4][4] = {};
                    const double* x0 = D + (size_t)(i0 + 0) * ld + j0;
                    const double* x1 = D + (size_t)(i0 + 1) * ld + j0;
                    const double* x2 = D + (size_t)(i0 + 2) * ld + j0;
                    const double* x3 = D + (size_t)(i0 + 3) * ld + j0;
                    for (int t = 0; t < jb; ++t) {
                        const double* y = Yt + (size_t)t * rem + c0;
                        for (int cc = 0; cc < 4; ++cc) {
                            acc[0][cc] += x0[t] * y[cc];
                            acc[1][cc] += x1[t] * y[cc];
                            acc[2][cc] += x2[t] * y[cc];
                            acc[3][cc] += x3[t] * y[cc];
                        }
                    }
                    for (int q = 0; q < 4; ++q) {
                        double* row = D + (size_t)(i0 + q) * ld + j0 + jb + c0;
                        for (int cc = 0; cc < 4; ++cc) row[cc] -= acc[q][cc];
                    }
                } else {
                    for (int q = 0; q < nr; ++q) {
                        const double* x = D + (size_t)(i0 + q) * ld + j0;
                        double* row = D + (size_t)(i0 + q) * ld + j0 + jb + c0;
                        for (int cc = 0; cc < nc; ++cc) {
                            double acc = 0.0;
                            for (int t = 0; t < jb; ++t) acc += x[t] * Yt[(size_t)t * rem + c0 + cc];
                            row[cc] -= acc;
                        }
                    }
                }
            }
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Distributed Cholesky decomposition
// ---------------------------------------------------------------------------
// dA : local block-cyclic rows on the device (nloc x n, row-major)
// Returns true on success; on failure `failIndex` holds the global diagonal
// index of the first non-positive pivot (consistent on all ranks).
//
// Lookahead: the trailing update of step k is split into the next block
// column (needed by panel k+1) and the rest.  The next block column is updated
// on the high priority panel stream (the diagonal block first, so its owner
// can factorize and broadcast it immediately), and panel k+1 (factorization,
// broadcast, triangular solves, exchange of the block column) overlaps with
// the bulk update of step k that runs on the second stream.
//
// Block column exchange: when all ranks share a node the packed block column
// is exchanged through an MPI shared memory window (one device-to-host copy by
// the producer, direct host-to-device copies by every consumer).  Otherwise a
// host-staged MPI_Allgatherv is used.
namespace {

struct PanelState {
    PanelState(const Dist& dist, double* a, cudaStream_t s)
        : d(dist), dA(a), sPanel(s), counts(dist.P, 0), displs(dist.P, 0) {}
    const Dist& d;
    double* dA;
    cudaStream_t sPanel;
    double* dLkk = nullptr;        // broadcast diagonal block, NB x NB (ld NB)
    double* dW[2] = {nullptr, nullptr};  // gathered block column, global row order, n x NB (ld bk)
    double* hLkk = nullptr;        // NB*NB + 1 (last entry carries the pivot flag)
    double* hPanelLoc = nullptr;   // local rows of the block column (packed) - allgather path
    double* hPanelAll = nullptr;   // gathered block column (rank order)     - allgather path
    cudaEvent_t evDiag = nullptr;  // diagonal block of the current panel copied to the host
    std::vector<int> counts, displs;
    // shared memory exchange path
    bool useShm = false;
    MPI_Comm nodeComm = MPI_COMM_NULL;
    MPI_Win win = MPI_WIN_NULL;
    std::vector<double*> seg;   // seg[q]: rank q's segment, 2 x nloc(q) x NB doubles
    std::vector<size_t> segRows;
};

// local row index of the diagonal block of block column k on its owner
inline size_t diagLocalRow(int k, int P) { return (size_t)(k / P) * NB; }

// Owner: copy the diagonal block of block column k to the host
// (asynchronously on sPanel, evDiag is recorded afterwards).
void panelDiag(PanelState& st, int k) {
    const Dist& d = st.d;
    if (d.r != k % d.P) return;
    const size_t n = d.n;
    const size_t kb = (size_t)k * NB;
    const int bk = (int)blockRows(d, k);
    const double* D = st.dA + diagLocalRow(k, d.P) * n + kb;
    CUDA_CHECK(cudaMemcpy2DAsync(st.hLkk, (size_t)bk * sizeof(double), D, n * sizeof(double),
                                 (size_t)bk * sizeof(double), bk, cudaMemcpyDeviceToHost, st.sPanel));
    CUDA_CHECK(cudaEventRecord(st.evDiag, st.sPanel));
}

// Broadcast the diagonal block, triangular solves of the local rows below it,
// exchange of the block column into dW[k % 2].  Returns false on a
// non-positive pivot.
bool panelRest(PanelState& st, int k, size_t& failIndex) {
    const Dist& d = st.d;
    const size_t n = d.n;
    const int P = d.P, r = d.r;
    const size_t kb = (size_t)k * NB;
    const int bk = (int)blockRows(d, k);
    const int owner = k % P;
    const size_t ldW = (size_t)bk;
    cudaStream_t s = st.sPanel;

    // 1. owner: factorize the diagonal block on the host (OpenMP)
    long long flag = -1;
    if (r == owner) {
        CUDA_CHECK(cudaEventSynchronize(st.evDiag));
        size_t fi = 0;
        if (!potrf_host(st.hLkk, bk, (size_t)bk, kb, fi)) flag = (long long)fi;
        memcpy(st.hLkk + (size_t)bk * bk, &flag, sizeof(long long));
    }

    // 2. broadcast L_kk (+ pivot flag stored as long long in the last slot)
    MPI_CHECK(MPI_Bcast(st.hLkk, bk * bk + 1, MPI_DOUBLE, owner, MPI_COMM_WORLD));
    memcpy(&flag, st.hLkk + (size_t)bk * bk, sizeof(long long));
    if (flag >= 0) {
        failIndex = (size_t)flag;
        return false;
    }
    if (r == owner) {
        // store the factorized diagonal block back into the local matrix
        double* D = st.dA + diagLocalRow(k, P) * n + kb;
        CUDA_CHECK(cudaMemcpy2DAsync(D, n * sizeof(double), st.hLkk, (size_t)bk * sizeof(double),
                                     (size_t)bk * sizeof(double), bk, cudaMemcpyHostToDevice, s));
    }
    if (k + 1 == d.nblk) return true;  // nothing below the last diagonal block
    CUDA_CHECK(cudaMemcpyAsync(st.dLkk, st.hLkk, (size_t)bk * bk * sizeof(double), cudaMemcpyHostToDevice, s));

    // 3. triangular solves for the local rows below the diagonal block
    const size_t lstart = firstLocalRowAfter(d, r, k);
    const int Mloc = (lstart < d.nloc) ? (int)(d.nloc - lstart) : 0;
    double* Xloc = st.dA + lstart * n + kb;
    trsm_panel(st.dLkk, bk, bk, Xloc, n, Mloc, s);

    // 4. exchange the block column
    const size_t colbase = kb + bk;  // global index of the first trailing row/col
    double* dW = st.dW[k % 2];

    if (st.useShm) {
        // producer: pack the local rows into the own shared segment
        double* mySeg = st.seg[r] + (size_t)(k % 2) * st.segRows[r] * NB;
        if (Mloc > 0) {
            CUDA_CHECK(cudaMemcpy2DAsync(mySeg + lstart * ldW, (size_t)bk * sizeof(double), Xloc,
                                         n * sizeof(double), (size_t)bk * sizeof(double), Mloc,
                                         cudaMemcpyDeviceToHost, s));
        }
        CUDA_CHECK(cudaStreamSynchronize(s));
        MPI_CHECK(MPI_Win_sync(st.win));
        MPI_CHECK(MPI_Barrier(st.nodeComm));
        MPI_CHECK(MPI_Win_sync(st.win));
        // consumers: copy every block row straight from the producer's segment
        for (int I = k + 1; I < d.nblk; ++I) {
            const int q = I % P;
            const size_t srcRow = (size_t)(I / P) * NB;
            const double* src = st.seg[q] + (size_t)(k % 2) * st.segRows[q] * NB + srcRow * ldW;
            const size_t dstRow = (size_t)I * NB - colbase;
            CUDA_CHECK(cudaMemcpyAsync(dW + dstRow * ldW, src, blockRows(d, I) * ldW * sizeof(double),
                                       cudaMemcpyHostToDevice, s));
        }
        return true;
    }

    if (Mloc > 0) {
        CUDA_CHECK(cudaMemcpy2DAsync(st.hPanelLoc, (size_t)bk * sizeof(double), Xloc, n * sizeof(double),
                                     (size_t)bk * sizeof(double), Mloc, cudaMemcpyDeviceToHost, s));
    }
    CUDA_CHECK(cudaStreamSynchronize(s));

    MPI_Datatype rowType;
    MPI_CHECK(MPI_Type_contiguous(bk, MPI_DOUBLE, &rowType));
    MPI_CHECK(MPI_Type_commit(&rowType));
    int total = 0;
    for (int q = 0; q < P; ++q) {
        const size_t ls = firstLocalRowAfter(d, q, k);
        const size_t nl = localRows(d, q);
        st.counts[q] = (ls < nl) ? (int)(nl - ls) : 0;
        st.displs[q] = total;
        total += st.counts[q];
    }
    MPI_CHECK(MPI_Allgatherv(st.hPanelLoc, Mloc, rowType, st.hPanelAll, st.counts.data(), st.displs.data(),
                             rowType, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Type_free(&rowType));

    // unpack into global row order on the device: one copy per block row
    for (int I = k + 1; I < d.nblk; ++I) {
        const int q = I % P;
        const int lb = I / P;
        const size_t srcRow = (size_t)st.displs[q] + ((size_t)lb * NB - firstLocalRowAfter(d, q, k));
        const size_t dstRow = (size_t)I * NB - colbase;
        CUDA_CHECK(cudaMemcpyAsync(dW + dstRow * ldW, st.hPanelAll + srcRow * ldW,
                                   blockRows(d, I) * ldW * sizeof(double), cudaMemcpyHostToDevice, s));
    }
    return true;
}

void setupShm(PanelState& st) {
    const Dist& d = st.d;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, d.r, MPI_INFO_NULL, &st.nodeComm));
    int nodeSize = 0, nodeRank = 0;
    MPI_CHECK(MPI_Comm_size(st.nodeComm, &nodeSize));
    MPI_CHECK(MPI_Comm_rank(st.nodeComm, &nodeRank));
    int sameNode = (nodeSize == d.P && nodeRank == d.r) ? 1 : 0;
    int allSame = 0;
    MPI_CHECK(MPI_Allreduce(&sameNode, &allSame, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD));
    if (!allSame) {
        MPI_CHECK(MPI_Comm_free(&st.nodeComm));
        return;
    }
    st.useShm = true;
    st.seg.resize(d.P);
    st.segRows.resize(d.P);
    const MPI_Aint mySize = (MPI_Aint)(2 * std::max<size_t>(1, d.nloc) * NB * sizeof(double));
    double* base = nullptr;
    MPI_CHECK(MPI_Win_allocate_shared(mySize, sizeof(double), MPI_INFO_NULL, st.nodeComm, &base, &st.win));
    for (int q = 0; q < d.P; ++q) {
        MPI_Aint sz = 0;
        int disp = 0;
        MPI_CHECK(MPI_Win_shared_query(st.win, q, &sz, &disp, &st.seg[q]));
        st.segRows[q] = std::max<size_t>(1, localRows(d, q));
    }
    // Pin the shared segments for fast DMA; fall back silently to pageable copies.
    for (int q = 0; q < d.P; ++q) {
        if (cudaHostRegister(st.seg[q], 2 * st.segRows[q] * NB * sizeof(double), cudaHostRegisterDefault) !=
            cudaSuccess) {
            (void)cudaGetLastError();
        }
    }
}

void teardownShm(PanelState& st) {
    if (!st.useShm) return;
    for (int q = 0; q < st.d.P; ++q) {
        if (cudaHostUnregister(st.seg[q]) != cudaSuccess) (void)cudaGetLastError();
    }
    MPI_CHECK(MPI_Win_free(&st.win));
    MPI_CHECK(MPI_Comm_free(&st.nodeComm));
}

}  // namespace

bool choleskyDecomposition(const Dist& d, double* dA, cudaStream_t sPanel, cudaStream_t sUpdate,
                           size_t& failIndex) {
    const size_t n = d.n;
    const int P = d.P, r = d.r;

    PanelState st(d, dA, sPanel);
    CUDA_CHECK(cudaMalloc(&st.dLkk, (size_t)NB * NB * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&st.dW[0], (size_t)n * NB * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&st.dW[1], (size_t)n * NB * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&st.hLkk, ((size_t)NB * NB + 1) * sizeof(double)));
    CUDA_CHECK(cudaEventCreateWithFlags(&st.evDiag, cudaEventDisableTiming));
    setupShm(st);
    if (!st.useShm) {
        CUDA_CHECK(cudaMallocHost(&st.hPanelLoc, std::max<size_t>(1, d.nloc) * NB * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&st.hPanelAll, std::max<size_t>(1, n) * NB * sizeof(double)));
    }

    cudaEvent_t evPanel, evBulk;
    CUDA_CHECK(cudaEventCreateWithFlags(&evPanel, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&evBulk, cudaEventDisableTiming));

    failIndex = 0;
    panelDiag(st, 0);
    bool ok = panelRest(st, 0, failIndex);

    for (int k = 0; ok && k + 1 < d.nblk; ++k) {
        const size_t kb = (size_t)k * NB;
        const int bk = (int)blockRows(d, k);
        const size_t colbase = kb + bk;
        const int bk1 = (int)blockRows(d, k + 1);
        const size_t ldW = (size_t)bk;
        const double* dW = st.dW[k % 2];
        const bool ownerNext = ((k + 1) % P == r);

        const size_t lstart = firstLocalRowAfter(d, r, k);
        const int Mloc = (lstart < d.nloc) ? (int)(d.nloc - lstart) : 0;
        const double* Xloc = dA + lstart * n + kb;

        // panel k is complete on sPanel (W_k uploaded): the bulk update may start
        CUDA_CHECK(cudaEventRecord(evPanel, sPanel));

        // (a) update the next block column [colbase, colbase + bk1) on the
        //     panel stream; it must see the bulk update of the previous step
        CUDA_CHECK(cudaStreamWaitEvent(sPanel, evBulk, 0));
        if (ownerNext) {
            // diagonal block first (its local rows are lstart .. lstart + bk1),
            // factorize it, then the remaining local rows of the block column
            gemm_nt<true>(bk1, bk1, bk, Xloc, n, dW, ldW, dA + lstart * n + colbase, n, sPanel);
            panelDiag(st, k + 1);
            gemm_nt<true>(Mloc - bk1, bk1, bk, Xloc + (size_t)bk1 * n, n, dW, ldW,
                          dA + (lstart + bk1) * n + colbase, n, sPanel, true, lstart + bk1, P, r, colbase);
        } else {
            gemm_nt<true>(Mloc, bk1, bk, Xloc, n, dW, ldW, dA + lstart * n + colbase, n, sPanel, true, lstart, P,
                          r, colbase);
        }

        // (b) bulk update of the remaining trailing columns on the update
        //     stream.  The block column k+2 (needed by the lookahead of the
        //     next step) is updated first and signalled through evBulk, so the
        //     next step's (a) only waits for that small part.
        CUDA_CHECK(cudaStreamWaitEvent(sUpdate, evPanel, 0));
        const int bk2 = (k + 2 < d.nblk) ? (int)blockRows(d, k + 2) : 0;
        const size_t col2 = colbase + bk1;
        gemm_nt<true>(Mloc, bk2, bk, Xloc, n, dW + (size_t)bk1 * ldW, ldW, dA + lstart * n + col2, n, sUpdate, true,
                      lstart, P, r, col2);
        CUDA_CHECK(cudaEventRecord(evBulk, sUpdate));
        const int Ntrail = (int)(n - col2) - bk2;
        gemm_nt<true>(Mloc, Ntrail, bk, Xloc, n, dW + (size_t)(bk1 + bk2) * ldW, ldW,
                      dA + lstart * n + col2 + bk2, n, sUpdate, true, lstart, P, r, col2 + bk2);

        // panel k + 1 overlaps with (b)
        ok = panelRest(st, k + 1, failIndex);
    }

    CUDA_CHECK(cudaStreamSynchronize(sUpdate));
    CUDA_CHECK(cudaStreamSynchronize(sPanel));
    if (ok && d.nloc > 0) {
        // zero the strictly upper triangular part
        dim3 grid((unsigned)((n + 255) / 256), (unsigned)d.nloc);
        zero_upper_kernel<<<grid, 256, 0, sPanel>>>(dA, d.nloc, n, P, r);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(sPanel));
    }

    teardownShm(st);
    CUDA_CHECK(cudaEventDestroy(evPanel));
    CUDA_CHECK(cudaEventDestroy(evBulk));
    CUDA_CHECK(cudaEventDestroy(st.evDiag));
    CUDA_CHECK(cudaFreeHost(st.hLkk));
    if (st.hPanelLoc) CUDA_CHECK(cudaFreeHost(st.hPanelLoc));
    if (st.hPanelAll) CUDA_CHECK(cudaFreeHost(st.hPanelAll));
    CUDA_CHECK(cudaFree(st.dLkk));
    CUDA_CHECK(cudaFree(st.dW[0]));
    CUDA_CHECK(cudaFree(st.dW[1]));
    return ok;
}
// ---------------------------------------------------------------------------
// Generate a symmetric positive definite matrix (local rows only)
// ---------------------------------------------------------------------------
// Method: A = B * B^T + n * I with B random (identical to the original code;
// each element of B * B^T is accumulated sequentially over k so the generated
// matrix matches the sequential program).
void generatePositiveDefiniteMatrix(const Dist& d, double* dA, cudaStream_t stream) {
    const size_t n = d.n;
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

    double* dB = nullptr;
    CUDA_CHECK(cudaMalloc(&dB, n * n * sizeof(double)));
    CUDA_CHECK(cudaMemcpyAsync(dB, B.data(), n * n * sizeof(double), cudaMemcpyHostToDevice, stream));

    // local rows of B (block-cyclic): copy block by block to a compact buffer
    double* dBloc = nullptr;
    CUDA_CHECK(cudaMalloc(&dBloc, std::max<size_t>(1, d.nloc) * n * sizeof(double)));
    for (int lb = 0; lb < d.nlocblk; ++lb) {
        const int I = lb * d.P + d.r;
        CUDA_CHECK(cudaMemcpyAsync(dBloc + (size_t)lb * NB * n, dB + (size_t)I * NB * n,
                                   blockRows(d, I) * n * sizeof(double), cudaMemcpyDeviceToDevice, stream));
    }

    gemm_nt<false>((int)d.nloc, (int)n, (int)n, dBloc, n, dB, n, dA, n, stream);
    if (d.nloc > 0) {
        add_diag_kernel<<<(unsigned)((d.nloc + 255) / 256), 256, 0, stream>>>(dA, d.nloc, n, d.P, d.r);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(dBloc));
    CUDA_CHECK(cudaFree(dB));
}

// ---------------------------------------------------------------------------
// Gather the distributed matrix in global row order on the host.
// If allRanks is true every rank receives the full matrix, otherwise rank 0.
// ---------------------------------------------------------------------------
static void gatherMatrix(const Dist& d, const double* dA, std::vector<double>& full, bool allRanks,
                         cudaStream_t stream) {
    const size_t n = d.n;
    std::vector<double> local(std::max<size_t>(1, d.nloc) * n);
    CUDA_CHECK(cudaMemcpyAsync(local.data(), dA, d.nloc * n * sizeof(double), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    MPI_Datatype rowType;
    MPI_CHECK(MPI_Type_contiguous((int)n, MPI_DOUBLE, &rowType));
    MPI_CHECK(MPI_Type_commit(&rowType));

    std::vector<int> counts(d.P), displs(d.P);
    std::vector<size_t> offsets(d.P);
    int total = 0;
    for (int q = 0; q < d.P; ++q) {
        counts[q] = (int)localRows(d, q);
        displs[q] = total;
        offsets[q] = (size_t)total;
        total += counts[q];
    }

    const bool receiver = allRanks || d.r == 0;
    std::vector<double> staged;
    if (receiver) staged.resize(n * n);
    if (allRanks) {
        MPI_CHECK(MPI_Allgatherv(local.data(), (int)d.nloc, rowType, staged.data(), counts.data(), displs.data(),
                                 rowType, MPI_COMM_WORLD));
    } else {
        MPI_CHECK(MPI_Gatherv(local.data(), (int)d.nloc, rowType, staged.data(), counts.data(), displs.data(),
                              rowType, 0, MPI_COMM_WORLD));
    }
    MPI_CHECK(MPI_Type_free(&rowType));

    if (receiver) {
        full.resize(n * n);
        blockCyclicToGlobal(d, staged.data(), offsets, full.data(), n);
    }
}

// ---------------------------------------------------------------------------
// Validation: reconstruct L * L^T (distributed, on the GPUs) and compare with
// the original matrix.
// ---------------------------------------------------------------------------
bool validateCholesky(const Dist& d, const double* dL, const double* dAorig, cudaStream_t stream) {
    const size_t n = d.n;

    std::vector<double> Lfull;
    gatherMatrix(d, dL, Lfull, true, stream);

    double* dLfull = nullptr;
    double* dRecon = nullptr;
    CUDA_CHECK(cudaMalloc(&dLfull, n * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dRecon, std::max<size_t>(1, d.nloc) * n * sizeof(double)));
    CUDA_CHECK(cudaMemcpyAsync(dLfull, Lfull.data(), n * n * sizeof(double), cudaMemcpyHostToDevice, stream));

    // reconstructed[i][j] = sum_k L[i][k] * L[j][k]  (local rows i)
    gemm_nt<false>((int)d.nloc, (int)n, (int)n, dL, n, dLfull, n, dRecon, n, stream);

    const int nblocks = 1024;
    double* dPartAbs = nullptr;
    double* dPartRel = nullptr;
    CUDA_CHECK(cudaMalloc(&dPartAbs, nblocks * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dPartRel, nblocks * sizeof(double)));
    error_kernel<<<nblocks, 256, 0, stream>>>(dRecon, dAorig, d.nloc * n, dPartAbs, dPartRel);
    CUDA_CHECK(cudaGetLastError());
    std::vector<double> pAbs(nblocks), pRel(nblocks);
    CUDA_CHECK(cudaMemcpyAsync(pAbs.data(), dPartAbs, nblocks * sizeof(double), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(pRel.data(), dPartRel, nblocks * sizeof(double), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    double maxError = 0.0, relError = 0.0;
#pragma omp parallel for reduction(max : maxError, relError)
    for (int i = 0; i < nblocks; ++i) {
        maxError = std::max(maxError, pAbs[i]);
        relError = std::max(relError, pRel[i]);
    }
    double g[2] = {maxError, relError}, gmax[2];
    MPI_CHECK(MPI_Allreduce(g, gmax, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));
    maxError = gmax[0];
    relError = gmax[1];

    CUDA_CHECK(cudaFree(dPartAbs));
    CUDA_CHECK(cudaFree(dPartRel));
    CUDA_CHECK(cudaFree(dRecon));
    CUDA_CHECK(cudaFree(dLfull));

    if (d.r == 0) {
        printf("Max absolute error: %.10e\n", maxError);
        printf("Max relative error: %.10e\n", relError);
    }
    if (relError > 1e-6) {
        if (d.r == 0) printf("Validation failed: relative error too large\n");
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
    int provided = 0;
    MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));
    int rank = 0, nprocs = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &nprocs));

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

    // Select the GPU: round-robin over the devices visible on this node.
    {
        MPI_Comm nodeComm;
        MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm));
        int localRank = 0, localSize = 1;
        MPI_CHECK(MPI_Comm_rank(nodeComm, &localRank));
        MPI_CHECK(MPI_Comm_size(nodeComm, &localSize));
        MPI_CHECK(MPI_Comm_free(&nodeComm));
        int ndev = 0;
        CUDA_CHECK(cudaGetDeviceCount(&ndev));
        if (ndev <= 0) {
            if (rank == 0) fprintf(stderr, "No CUDA devices found\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        CUDA_CHECK(cudaSetDevice(localRank % ndev));
        // Share the host cores between the ranks on this node unless the user
        // pinned the OpenMP thread count explicitly.
        if (getenv("OMP_NUM_THREADS") == nullptr) {
            const int threads = std::max(1, omp_get_num_procs() / localSize);
            omp_set_num_threads(threads);
        }
    }

    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, OpenMP threads/rank: %d, CUDA device per rank\n", nprocs, omp_get_max_threads());
    }

    cudaStream_t stream;   // panel / general stream (high priority)
    cudaStream_t sUpdate;  // bulk trailing update stream
    {
        int lo = 0, hi = 0;
        CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
        CUDA_CHECK(cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, hi));
        CUDA_CHECK(cudaStreamCreateWithPriority(&sUpdate, cudaStreamNonBlocking, lo));
    }

    Dist d;
    d.P = nprocs;
    d.r = rank;
    d.n = n;
    d.nblk = (int)((n + NB - 1) / NB);
    d.nlocblk = localBlocks(d, rank);
    d.nloc = localRows(d, rank);

    // Allocate local matrix on the device
    double* dA = nullptr;
    double* dAorig = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, std::max<size_t>(1, d.nloc) * n * sizeof(double)));

    // Generate positive definite matrix
    if (rank == 0) printf("Generating positive definite matrix...\n");
    generatePositiveDefiniteMatrix(d, dA, stream);

    if (validate) {
        CUDA_CHECK(cudaMalloc(&dAorig, std::max<size_t>(1, d.nloc) * n * sizeof(double)));
        CUDA_CHECK(cudaMemcpyAsync(dAorig, dA, d.nloc * n * sizeof(double), cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    // Perform Cholesky decomposition
    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto start = std::chrono::high_resolution_clock::now();

    size_t failIndex = 0;
    bool success = choleskyDecomposition(d, dA, stream, sUpdate, failIndex);

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (!success) {
        if (rank == 0) {
            printf("Error: Matrix is not positive definite at diagonal element %zu\n", failIndex);
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

    // Print results for external validation
    if (printResults) {
        std::vector<double> A;
        gatherMatrix(d, dA, A, false, stream);
        if (rank == 0) print_results(A, "CholeskyL");
    }

    int ret = 0;
    // Validation
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        bool valid = validateCholesky(d, dA, dAorig, stream);
        if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        ret = valid ? 0 : 1;
        CUDA_CHECK(cudaFree(dAorig));
    }

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaStreamDestroy(sUpdate));
    MPI_Finalize();
    return ret;
}
