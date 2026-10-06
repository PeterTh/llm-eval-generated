#include <algorithm>
#include <atomic>
#include <climits>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

// Hybrid MPI + OpenMP + CUDA Cholesky decomposition
// Decomposes positive definite matrix A into L * L^T where L is lower triangular.
//
// Distribution: the matrix is split into block columns of width NB which are dealt
// out block-cyclically to the MPI ranks (one GPU per rank). Each rank keeps its block
// columns on the GPU in column-major layout (ld = n). Because A is symmetric, block
// column j of A equals block row j of the row-major input.
//
// Algorithm: right-looking blocked Cholesky with one step of lookahead.
//   for each block column k:
//     owner: factor the NB x NB diagonal block and invert the factor on the CPU (far
//            too little work for the GPU's FP64 units), then solve the panel
//            L_ik = A_ik * L_kk^-T on the GPU as a GEMM with the inverse. The finished
//            block column (rows k0..n) is stored row-major, i.e. in the layout of the
//            final result.
//     the block column is distributed in pipelined chunks: ranks on the same node copy
//            it from the owner's MPI shared-memory window straight to their GPU, ranks
//            on other nodes receive it by message passing.
//     all:   trailing update A_ij -= L_ik * L_jk^T of the owned block columns (GEMM).
// The owner of block column k+1 updates and factors it first on a high-priority stream,
// so its distribution overlaps the remaining trailing update of step k. OpenMP helper
// threads on rank 0 assemble the row-major factor while the factorization proceeds.

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err__ = (call);                                               \
        if (err__ != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err__), \
                    __FILE__, __LINE__);                                          \
            MPI_Abort(MPI_COMM_WORLD, 1);                                         \
        }                                                                         \
    } while (0)

static constexpr int NB = 128;           // block column width
static constexpr int TM = 64;            // GEMM tile rows
static constexpr int TN = 64;            // GEMM tile cols
static constexpr int TK = 16;            // GEMM tile depth
static constexpr int GEMM_THREADS = 256;
static constexpr int CHUNK_ROWS = 512;   // rows per transfer chunk of a block column
static_assert(CHUNK_ROWS >= 2 * NB && CHUNK_ROWS % TM == 0, "first chunk must hold the next diagonal block");

// ---------------------------------------------------------------------------
// GEMM tile: C[0:M,0:N] = alpha * sum_k opA(i,k) * opB(j,k) + beta * C
//   TA == false: opA(i,k) = A[i + k*lda]   TA == true: opA(i,k) = A[k + i*lda]
//   TB == false: opB(j,k) = B[j + k*ldb]   TB == true: opB(j,k) = B[k + j*ldb]
//   C is column-major: C[i + j*ldc]. Pointers are already offset to the tile.
// ---------------------------------------------------------------------------
template <bool TA, bool TB>
__device__ __forceinline__ void gemmTile(int M, int N, int K, double alpha,
                                         const double* __restrict__ A, size_t lda,
                                         const double* __restrict__ B, size_t ldb,
                                         double beta, double* __restrict__ C, size_t ldc) {
    __shared__ double As[TK][TM + 1];
    __shared__ double Bs[TK][TN + 1];

    const int tid = threadIdx.x;
    const int tx = tid % 16;
    const int ty = tid / 16;

    double acc[4][4];
#pragma unroll
    for (int a = 0; a < 4; ++a)
#pragma unroll
        for (int b = 0; b < 4; ++b) acc[a][b] = 0.0;

    for (int k0 = 0; k0 < K; k0 += TK) {
#pragma unroll
        for (int q = 0; q < (TM * TK) / GEMM_THREADS; ++q) {
            const int e = tid + q * GEMM_THREADS;
            int i, k;
            if (TA) { i = e / TK; k = e % TK; } else { i = e % TM; k = e / TM; }
            double v = 0.0;
            if (i < M && k0 + k < K)
                v = TA ? A[(size_t)(k0 + k) + (size_t)i * lda] : A[(size_t)i + (size_t)(k0 + k) * lda];
            As[k][i] = v;
        }
#pragma unroll
        for (int q = 0; q < (TN * TK) / GEMM_THREADS; ++q) {
            const int e = tid + q * GEMM_THREADS;
            int j, k;
            if (TB) { j = e / TK; k = e % TK; } else { j = e % TN; k = e / TN; }
            double v = 0.0;
            if (j < N && k0 + k < K)
                v = TB ? B[(size_t)(k0 + k) + (size_t)j * ldb] : B[(size_t)j + (size_t)(k0 + k) * ldb];
            Bs[k][j] = v;
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < TK; ++kk) {
            double ra[4], rb[4];
#pragma unroll
            for (int a = 0; a < 4; ++a) ra[a] = As[kk][tx + 16 * a];
#pragma unroll
            for (int b = 0; b < 4; ++b) rb[b] = Bs[kk][ty + 16 * b];
#pragma unroll
            for (int a = 0; a < 4; ++a)
#pragma unroll
                for (int b = 0; b < 4; ++b) acc[a][b] = fma(ra[a], rb[b], acc[a][b]);
        }
        __syncthreads();
    }

#pragma unroll
    for (int b = 0; b < 4; ++b) {
        const int j = ty + 16 * b;
        if (j >= N) continue;
#pragma unroll
        for (int a = 0; a < 4; ++a) {
            const int i = tx + 16 * a;
            if (i >= M) continue;
            double* c = &C[(size_t)i + (size_t)j * ldc];
            *c = (beta == 0.0) ? alpha * acc[a][b] : alpha * acc[a][b] + beta * (*c);
        }
    }
}

template <bool TA, bool TB>
__global__ void __launch_bounds__(GEMM_THREADS)
gemmKernel(int M, int N, int K, double alpha, const double* __restrict__ A, size_t lda,
           const double* __restrict__ B, size_t ldb, double beta, double* __restrict__ C,
           size_t ldc) {
    const int i0 = blockIdx.x * TM;
    const int j0 = blockIdx.y * TN;
    if (i0 >= M || j0 >= N) return;
    const double* At = TA ? A + (size_t)i0 * lda : A + i0;
    const double* Bt = TB ? B + (size_t)j0 * ldb : B + j0;
    gemmTile<TA, TB>(min(TM, M - i0), min(TN, N - j0), K, alpha, At, lda, Bt, ldb, beta,
                     C + i0 + (size_t)j0 * ldc, ldc);
}

template <bool TA, bool TB>
static void gemm(cudaStream_t s, int M, int N, int K, double alpha, const double* A, size_t lda,
                 const double* B, size_t ldb, double beta, double* C, size_t ldc) {
    if (M <= 0 || N <= 0) return;
    dim3 grid((M + TM - 1) / TM, (N + TN - 1) / TN);
    gemmKernel<TA, TB><<<grid, GEMM_THREADS, 0, s>>>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    CUDA_CHECK(cudaGetLastError());
}

// Trailing update of local block columns lj in [ljs, ljs + gridDim.z):
//   A(rows, j0:j0+nbj) -= L(rows, k-block) * L(j0:j0+nbj, k-block)^T
// for global rows in [j0 + rowBeg, min(n, j0 + rowEnd)).
// P holds block column k (global rows k0..n) row-major with width nbk.
// Only tiles touching the lower triangle are computed.
__global__ void __launch_bounds__(GEMM_THREADS)
trailingUpdateKernel(double* __restrict__ Aloc, int n, int nprocs, int rank, int ljs,
                     const double* __restrict__ P, int k0, int nbk, int rowBeg, int rowEnd) {
    const int lj = ljs + blockIdx.z;
    const int j0 = (rank + nprocs * lj) * NB;
    const int nbj = min(NB, n - j0);
    const int r0 = j0 + rowBeg + blockIdx.x * TM;   // first global row of this tile
    const int rEnd = (int)min((long)n, (long)j0 + rowEnd);
    const int c0 = blockIdx.y * TN;                 // first local column of this tile
    if (r0 >= rEnd || c0 >= nbj) return;
    if (r0 + TM - 1 < j0 + c0) return;              // tile strictly above the diagonal
    double* C = Aloc + (size_t)lj * NB * n + (size_t)c0 * n + r0;
    gemmTile<true, true>(min(TM, rEnd - r0), min(TN, nbj - c0), nbk, -1.0,
                         P + (size_t)(r0 - k0) * nbk, (size_t)nbk,
                         P + (size_t)(j0 + c0 - k0) * nbk, (size_t)nbk, 1.0, C, (size_t)n);
}

// Add diagonal dominance on a local block column (global column offset j0).
__global__ void addDiagKernel(double* __restrict__ panel, size_t n, int j0, int nbj, double v) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < nbj) panel[(size_t)c * n + j0 + c] += v;
}

// Error statistics between reconstructed and original block column.
__global__ void errorKernel(const double* __restrict__ R, const double* __restrict__ Ao,
                            size_t count, unsigned long long* __restrict__ out) {
    double maxAbs = 0.0, maxRel = 0.0;
    for (size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x; t < count;
         t += (size_t)gridDim.x * blockDim.x) {
        const double error = fabs(R[t] - Ao[t]);
        maxAbs = fmax(maxAbs, error);
        maxRel = fmax(maxRel, error / (fabs(Ao[t]) + 1e-10));
    }
    for (int o = 16; o > 0; o /= 2) {
        maxAbs = fmax(maxAbs, __shfl_down_sync(0xffffffffu, maxAbs, o));
        maxRel = fmax(maxRel, __shfl_down_sync(0xffffffffu, maxRel, o));
    }
    if ((threadIdx.x & 31) == 0) {
        // Non-negative doubles order like their bit patterns
        atomicMax(&out[0], (unsigned long long)__double_as_longlong(maxAbs));
        atomicMax(&out[1], (unsigned long long)__double_as_longlong(maxRel));
    }
}

// ---------------------------------------------------------------------------
// Diagonal block on the host CPU (a 128 x 128 block is far too small for the GPU's
// FP64 units, while a single CPU core factors it in a few tens of microseconds).
// ---------------------------------------------------------------------------

// Factor the m x m block M (column-major, ld = m, lower part used) in place
// (right-looking, rank-4 updates). Returns the first failing diagonal index, or -1.
static int factorDiagonalBlock(double* __restrict__ M, int m) {
    int j = 0;
    for (; j < m; j += 4) {
        const int jb = std::min(4, m - j);
        // Factor columns j..j+jb-1 (already updated by all previous columns)
        for (int jj = j; jj < j + jb; ++jj) {
            double* colj = M + (size_t)jj * m;
            const double val = colj[jj];
            if (val <= 0.0) return jj;   // Matrix is not positive definite
            const double d = sqrt(val);
            colj[jj] = d;
            for (int i = jj + 1; i < m; ++i) colj[i] /= d;
            for (int c = jj + 1; c < j + jb; ++c) {
                double* colc = M + (size_t)c * m;
                const double f = colj[c];
                for (int i = c; i < m; ++i) colc[i] -= colj[i] * f;
            }
        }
        if (jb < 4) break;
        // Rank-4 update of the trailing columns
        const double* l0 = M + (size_t)j * m;
        const double* l1 = l0 + m;
        const double* l2 = l1 + m;
        const double* l3 = l2 + m;
        for (int c = j + 4; c < m; ++c) {
            double* __restrict__ colc = M + (size_t)c * m;
            const double f0 = l0[c], f1 = l1[c], f2 = l2[c], f3 = l3[c];
            for (int i = c; i < m; ++i)
                colc[i] -= (l0[i] * f0 + l1[i] * f1) + (l2[i] * f2 + l3[i] * f3);
        }
    }
    return -1;
}

// Inverse of the lower-triangular factor L (column-major, ld = m) into Inv
// (NB x NB, column-major, upper part and padding zero) by forward substitution.
static void invertLowerTriangular(const double* __restrict__ L, int m, double* __restrict__ Inv) {
    for (size_t t = 0; t < (size_t)NB * NB; ++t) Inv[t] = 0.0;
    // Forward substitution for four right-hand sides (unit vectors) at a time
    for (int j = 0; j < m; j += 4) {
        const int jb = std::min(4, m - j);
        double* x0 = Inv + (size_t)j * NB;
        double* x1 = x0 + NB;
        double* x2 = x1 + NB;
        double* x3 = x2 + NB;
        for (int q = 0; q < jb; ++q) Inv[(size_t)(j + q) * NB + j + q] = 1.0;
        for (int k = j; k < m; ++k) {
            const double* __restrict__ lk = L + (size_t)k * m;
            const double dk = lk[k];
            const double v0 = x0[k] / dk, v1 = (jb > 1) ? x1[k] / dk : 0.0;
            const double v2 = (jb > 2) ? x2[k] / dk : 0.0, v3 = (jb > 3) ? x3[k] / dk : 0.0;
            x0[k] = v0;
            if (jb > 1) x1[k] = v1;
            if (jb > 2) x2[k] = v2;
            if (jb > 3) x3[k] = v3;
            if (jb == 4) {
                for (int i = k + 1; i < m; ++i) {
                    const double l = lk[i];
                    x0[i] -= l * v0;
                    x1[i] -= l * v1;
                    x2[i] -= l * v2;
                    x3[i] -= l * v3;
                }
            } else {
                for (int i = k + 1; i < m; ++i) {
                    const double l = lk[i];
                    x0[i] -= l * v0;
                    if (jb > 1) x1[i] -= l * v1;
                    if (jb > 2) x2[i] -= l * v2;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Distributed context and workspace
// ---------------------------------------------------------------------------
static constexpr int NBUF = 3;   // device / staging buffers for block columns in flight

struct Dist {
    int rank = 0, nprocs = 1;
    size_t n = 0;
    int nblk = 0;              // number of block columns
    int nloc = 0;              // number of local block columns
    double* dA = nullptr;      // local block columns, column-major, ld = n

    // Rank 0: finished block columns k from other nodes (rows k*NB..n, row-major) at
    // hRes + resOff[k]
    double* hRes = nullptr;
    std::vector<size_t> resOff;

    // Workspace of the factorization
    cudaStream_t sHi = nullptr;     // high priority: critical path (next block column)
    cudaStream_t sComp = nullptr;   // trailing updates
    cudaStream_t sUp = nullptr, sDown = nullptr;   // host -> device, device -> host
    double* dP[NBUF] = {};          // block column k on the device in dP[k % NBUF]
    double* hP[NBUF] = {};          // pinned staging for block columns from other nodes
    double* dInv = nullptr;
    double* hInv = nullptr;
    double* hDiag = nullptr;
    cudaEvent_t evReady[NBUF];      // block column k present in dP[k % NBUF]
    cudaEvent_t evUsedComp[NBUF], evUsedHi[NBUF];   // all reads of dP[b] issued so far done
    cudaEvent_t evH2D[NBUF];        // staging buffer hP[b] consumed
    cudaEvent_t evNext, evFirst, evDiag, evDiagH;
    std::vector<cudaEvent_t> evChunk;

    // Node-local shared memory: every rank publishes its finished block columns in
    // NSLOT slots of an MPI shared window (page-locked), so ranks on the same node copy
    // them to their GPU directly; ranks on other nodes receive them by message passing.
    static constexpr int NSLOT = 3;
    MPI_Comm nodeComm = MPI_COMM_NULL;
    MPI_Win win = MPI_WIN_NULL;
    std::vector<int> nodeRankOf;     // world rank -> node-local rank, or -1 if remote
    std::vector<double*> slotBase;   // node-local rank -> its shared slots
    bool anyRemote = false;

    int owner(int j) const { return j % nprocs; }
    bool sameNode(int r) const { return nodeRankOf[r] >= 0; }
    double* slot(int k) const {
        return slotBase[nodeRankOf[owner(k)]] + (size_t)(localIndex(k) % NSLOT) * n * NB;
    }
    int localIndex(int j) const { return j / nprocs; }
    int blockWidth(int j) const { return (int)std::min<size_t>(NB, n - (size_t)j * NB); }
    double* panel(int lj) const { return dA + (size_t)lj * NB * n; }
    // first local block index whose global index is > k
    int firstLocalAfter(int k) const {
        const int t = k + 1 - rank;
        return t <= 0 ? 0 : (t + nprocs - 1) / nprocs;
    }
    // host buffer holding block column k on this rank
    double* hostPanel(int k) const {
        if (sameNode(owner(k))) return slot(k);
        return rank == 0 ? hRes + resOff[k] : hP[k % NBUF];
    }

    void allocWork() {
        int loPrio, hiPrio;
        CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&loPrio, &hiPrio));
        CUDA_CHECK(cudaStreamCreateWithPriority(&sHi, cudaStreamNonBlocking, hiPrio));
        CUDA_CHECK(cudaStreamCreateWithPriority(&sComp, cudaStreamNonBlocking, loPrio));
        CUDA_CHECK(cudaStreamCreateWithPriority(&sUp, cudaStreamNonBlocking, hiPrio));
        CUDA_CHECK(cudaStreamCreateWithPriority(&sDown, cudaStreamNonBlocking, hiPrio));
        const size_t cap = std::max<size_t>(1, n * NB);

        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
        int nodeSize;
        MPI_Comm_size(nodeComm, &nodeSize);
        {
            MPI_Group worldGroup, nodeGroup;
            MPI_Comm_group(MPI_COMM_WORLD, &worldGroup);
            MPI_Comm_group(nodeComm, &nodeGroup);
            std::vector<int> ranks(nprocs);
            for (int r = 0; r < nprocs; ++r) ranks[r] = r;
            nodeRankOf.resize(nprocs);
            MPI_Group_translate_ranks(worldGroup, nprocs, ranks.data(), nodeGroup, nodeRankOf.data());
            for (int& q : nodeRankOf) if (q == MPI_UNDEFINED) q = -1;
            MPI_Group_free(&worldGroup);
            MPI_Group_free(&nodeGroup);
        }
        anyRemote = nodeSize < nprocs;
        MPI_Info info;
        MPI_Info_create(&info);
        MPI_Info_set(info, "alloc_shared_noncontig", "true");
        double* base;
        MPI_Win_allocate_shared((MPI_Aint)(NSLOT * cap * sizeof(double)), sizeof(double), info,
                                nodeComm, &base, &win);
        MPI_Info_free(&info);
        slotBase.resize(nodeSize);
        for (int q = 0; q < nodeSize; ++q) {
            MPI_Aint sz;
            int du;
            MPI_Win_shared_query(win, q, &sz, &du, &slotBase[q]);
            // Page-lock for fast asynchronous copies (plain memory still works otherwise)
            if (cudaHostRegister(slotBase[q], (size_t)sz, cudaHostRegisterPortable) != cudaSuccess)
                cudaGetLastError();
        }
        if (rank == 0 && anyRemote) {
            // Block columns from other nodes are received here
            resOff.assign(nblk, 0);
            size_t total = 0;
            for (int j = 0; j < nblk; ++j) {
                resOff[j] = total;
                total += (n - (size_t)j * NB) * blockWidth(j);
            }
            CUDA_CHECK(cudaMallocHost(&hRes, std::max<size_t>(1, total) * sizeof(double)));
        }

        auto mkEvent = [](cudaEvent_t& e) {
            CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
        };
        for (int b = 0; b < NBUF; ++b) {
            CUDA_CHECK(cudaMalloc(&dP[b], cap * sizeof(double)));
            if (rank != 0 && anyRemote) CUDA_CHECK(cudaMallocHost(&hP[b], cap * sizeof(double)));
            mkEvent(evReady[b]);
            mkEvent(evUsedComp[b]);
            mkEvent(evUsedHi[b]);
            mkEvent(evH2D[b]);
            CUDA_CHECK(cudaEventRecord(evUsedComp[b], sComp));
            CUDA_CHECK(cudaEventRecord(evUsedHi[b], sHi));
            CUDA_CHECK(cudaEventRecord(evH2D[b], sUp));
        }
        CUDA_CHECK(cudaMalloc(&dInv, (size_t)NB * NB * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&hInv, (size_t)NB * NB * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&hDiag, (size_t)NB * NB * sizeof(double)));
        mkEvent(evNext);
        mkEvent(evFirst);
        mkEvent(evDiag);
        mkEvent(evDiagH);
        CUDA_CHECK(cudaEventRecord(evNext, sComp));
        evChunk.resize((n + CHUNK_ROWS - 1) / CHUNK_ROWS + 1);
        for (auto& e : evChunk) mkEvent(e);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void freeWork() {
        for (int b = 0; b < NBUF; ++b) {
            CUDA_CHECK(cudaFree(dP[b]));
            if (hP[b]) CUDA_CHECK(cudaFreeHost(hP[b]));
            CUDA_CHECK(cudaEventDestroy(evReady[b]));
            CUDA_CHECK(cudaEventDestroy(evUsedComp[b]));
            CUDA_CHECK(cudaEventDestroy(evUsedHi[b]));
            CUDA_CHECK(cudaEventDestroy(evH2D[b]));
        }
        CUDA_CHECK(cudaFree(dInv));
        CUDA_CHECK(cudaFreeHost(hInv));
        CUDA_CHECK(cudaFreeHost(hDiag));
        CUDA_CHECK(cudaEventDestroy(evNext));
        CUDA_CHECK(cudaEventDestroy(evFirst));
        CUDA_CHECK(cudaEventDestroy(evDiag));
        CUDA_CHECK(cudaEventDestroy(evDiagH));
        for (auto& e : evChunk) CUDA_CHECK(cudaEventDestroy(e));
        CUDA_CHECK(cudaStreamDestroy(sHi));
        CUDA_CHECK(cudaStreamDestroy(sComp));
        CUDA_CHECK(cudaStreamDestroy(sUp));
        CUDA_CHECK(cudaStreamDestroy(sDown));
        for (double* p : slotBase)
            if (cudaHostUnregister(p) != cudaSuccess) cudaGetLastError();
        MPI_Win_free(&win);
        MPI_Comm_free(&nodeComm);
        if (hRes) CUDA_CHECK(cudaFreeHost(hRes));
        hRes = nullptr;
    }
};

// Copy finished block column k (rows k0..n, row-major on the host) into the row-major
// result; rows are split among nworkers workers.
static void assembleBlock(const Dist& D, std::vector<double>& A, int k, int worker, int nworkers) {
    const size_t n = D.n;
    const size_t k0 = (size_t)k * NB;
    const int nbk = D.blockWidth(k);
    const double* src = D.hostPanel(k);
    for (size_t i = k0 + worker; i < n; i += nworkers)
        memcpy(&A[i * n + k0], &src[(i - k0) * nbk], nbk * sizeof(double));
}

// Returns true on success. On rank 0, A receives the row-major factor (upper part zero).
bool choleskyDecomposition(Dist& D, std::vector<double>& A) {
    const int n = (int)D.n;
    const int P = D.nprocs;
    const int me = D.rank;
    cudaStream_t sHi = D.sHi, sComp = D.sComp, sUp = D.sUp, sDown = D.sDown;

    auto panelRows = [&](int k) { return n - k * NB; };
    auto numChunks = [&](int k) { return (panelRows(k) + CHUNK_ROWS - 1) / CHUNK_ROWS; };

    // Trailing update with block column k of local columns [ljs, lje), rows [rowBeg, rowEnd)
    // relative to each column's diagonal.
    auto update = [&](cudaStream_t s, int k, int ljs, int lje, int rowBeg, int rowEnd) {
        const int nz = lje - ljs;
        const int k0 = k * NB, nbk = D.blockWidth(k);
        const int rows = std::min(n - k0 - nbk, rowEnd) - rowBeg;
        if (nz <= 0 || rows <= 0) return;
        dim3 grid((rows + TM - 1) / TM, (NB + TN - 1) / TN, nz);
        trailingUpdateKernel<<<grid, GEMM_THREADS, 0, s>>>(D.dA, n, P, me, ljs, D.dP[k % NBUF], k0,
                                                           nbk, rowBeg, rowEnd);
        CUDA_CHECK(cudaGetLastError());
    };

    // Owner, step 1: copy the (fully updated) diagonal block of column k to the host.
    auto startDiag = [&](int k) {
        const int k0 = k * NB, nbk = D.blockWidth(k);
        const double* col = D.panel(D.localIndex(k));
        CUDA_CHECK(cudaEventRecord(D.evDiag, sHi));
        CUDA_CHECK(cudaStreamWaitEvent(sDown, D.evDiag, 0));
        CUDA_CHECK(cudaMemcpy2DAsync(D.hDiag, (size_t)nbk * sizeof(double), col + k0,
                                     (size_t)n * sizeof(double), (size_t)nbk * sizeof(double), nbk,
                                     cudaMemcpyDeviceToHost, sDown));
        CUDA_CHECK(cudaEventRecord(D.evDiagH, sDown));
    };

    // Rank 0: helper threads assemble finished block columns into the row-major result
    std::atomic<int> arrived{0}, assembled{0};
    std::atomic<bool> stop{false};

    // Slot reuse protocol: a rank that copied block column k out of the shared slot of a
    // rank on the same node acknowledges it (rank 0 only after assembling it as well).
    constexpr int TAG_ACK = 32767;
    struct PendingAck { int k, root, b; };
    std::vector<PendingAck> pendingAcks;
    size_t ackHead = 0;
    auto waitAssembled = [&](int k) {
        while (assembled.load(std::memory_order_acquire) <= k) {
        }
    };
    auto flushAcks = [&](int forceUpTo) {
        while (ackHead < pendingAcks.size()) {
            PendingAck& a = pendingAcks[ackHead];
            if (a.k <= forceUpTo) {
                CUDA_CHECK(cudaEventSynchronize(D.evH2D[a.b]));
                if (me == 0) waitAssembled(a.k);
            } else {
                if (cudaEventQuery(D.evH2D[a.b]) != cudaSuccess) break;
                if (me == 0 && assembled.load(std::memory_order_acquire) <= a.k) break;
            }
            MPI_Send(&a.k, 1, MPI_INT, a.root, TAG_ACK, MPI_COMM_WORLD);
            ++ackHead;
        }
    };
    // Owner: wait until every rank on the node is done with own block column k
    std::vector<int> ownShared;   // own block columns distributed so far
    size_t ownCollected = 0;
    auto collectAcks = [&](int k) {
        for (int r = 0; r < P; ++r) {
            if (r == me || !D.sameNode(r)) continue;
            int v;
            MPI_Recv(&v, 1, MPI_INT, r, TAG_ACK, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (me == 0) waitAssembled(k);
        ++ownCollected;
    };

    // Owner, step 2: factor the diagonal block on the CPU, solve the panel on the GPU
    // (L_ik = A_ik * Linv^T, row-major into dP) and start copying it to the host.
    // Returns the failing local diagonal index or -1.
    auto finishFactor = [&](int k) -> int {
        const int b = k % NBUF;
        const int k0 = k * NB, nbk = D.blockWidth(k);
        const int rows = panelRows(k), mp = rows - nbk;
        const double* col = D.panel(D.localIndex(k));
        CUDA_CHECK(cudaEventSynchronize(D.evDiagH));
        const int f = factorDiagonalBlock(D.hDiag, nbk);
        if (f >= 0) return f;
        invertLowerTriangular(D.hDiag, nbk, D.hInv);
        // Diagonal block rows of the finished block column (row-major, upper part zero)
        if (k - D.NSLOT * P >= 0) collectAcks(k - D.NSLOT * P);    // slot free again
        double* h = D.hostPanel(k);
        for (int i = 0; i < nbk; ++i)
            for (int j = 0; j < nbk; ++j)
                h[(size_t)i * nbk + j] = (j <= i) ? D.hDiag[i + (size_t)j * nbk] : 0.0;

        CUDA_CHECK(cudaMemcpyAsync(D.dInv, D.hInv, (size_t)NB * NB * sizeof(double),
                                   cudaMemcpyHostToDevice, sHi));
        CUDA_CHECK(cudaStreamWaitEvent(sHi, D.evUsedComp[b], 0));   // dP[b] free
        gemm<false, false>(sHi, nbk, mp, nbk, 1.0, D.dInv, (size_t)NB, col + k0 + nbk, (size_t)n,
                           0.0, D.dP[b] + (size_t)nbk * nbk, (size_t)nbk);
        CUDA_CHECK(cudaEventRecord(D.evReady[b], sHi));
        CUDA_CHECK(cudaStreamWaitEvent(sDown, D.evReady[b], 0));
        for (int c = 0; c < numChunks(k); ++c) {
            const size_t r0 = std::max<size_t>((size_t)c * CHUNK_ROWS, nbk);
            const size_t r1 = std::min<size_t>((size_t)(c + 1) * CHUNK_ROWS, rows);
            if (r1 > r0)
                CUDA_CHECK(cudaMemcpyAsync(h + r0 * nbk, D.dP[b] + r0 * nbk,
                                           (r1 - r0) * nbk * sizeof(double),
                                           cudaMemcpyDeviceToHost, sDown));
            CUDA_CHECK(cudaEventRecord(D.evChunk[c], sDown));
        }
        return -1;
    };

    // Distribute block column k from its owner to all ranks (pipelined chunks).
    // Returns the failing global diagonal index or -1.
    std::vector<MPI_Request> reqs;
    int earlyDiag = -1;   // block column whose diagonal block was already started
    std::vector<int> notes;
    auto sharePanel = [&](int k, int ownerFlag) -> long {
        const int b = k % NBUF;
        const int root = D.owner(k);
        const int nbk = D.blockWidth(k), rows = panelRows(k), nch = numChunks(k);
        double* h = D.hostPanel(k);
        flushAcks(k + 1 - D.NSLOT * P);
        int f = ownerFlag;
        if (P > 1) MPI_Bcast(&f, 1, MPI_INT, root, MPI_COMM_WORLD);
        if (f >= 0) return (long)k * NB + f;

        const int tag = k % TAG_ACK;
        auto chunkPtr = [&](int c) { return h + (size_t)c * CHUNK_ROWS * nbk; };
        auto chunkCount = [&](int c) {
            return (int)(std::min<size_t>(CHUNK_ROWS, rows - (size_t)c * CHUNK_ROWS) * nbk);
        };
        reqs.clear();
        if (root == me) {
            // Ranks on this node only get a short "chunk ready" note and copy the data out
            // of the shared slot themselves; ranks on other nodes get the data. The owner of
            // the next block column is on the critical path and is served first.
            const int next = (k + 1 < D.nblk) ? D.owner(k + 1) : me;
            notes.resize(nch);
            auto post = [&](int d, int c) {
                reqs.emplace_back();
                if (D.sameNode(d))
                    MPI_Isend(&notes[c], 1, MPI_INT, d, tag, MPI_COMM_WORLD, &reqs.back());
                else
                    MPI_Isend(chunkPtr(c), chunkCount(c), MPI_DOUBLE, d, tag, MPI_COMM_WORLD,
                              &reqs.back());
            };
            for (int c = 0; c < nch; ++c) {
                notes[c] = c;
                CUDA_CHECK(cudaEventSynchronize(D.evChunk[c]));
                if (next != me) post(next, c);
                for (int d = 0; d < P; ++d)
                    if (d != me && d != next && D.sameNode(d)) post(d, c);
            }
            MPI_Waitall((int)reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);
            reqs.clear();
            for (int d = 0; d < P; ++d)
                if (d != me && d != next && !D.sameNode(d))
                    for (int c = 0; c < nch; ++c) post(d, c);
            MPI_Waitall((int)reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);
            ownShared.push_back(k);
        } else {
            const bool local = D.sameNode(root);
            if (!local && me != 0) CUDA_CHECK(cudaEventSynchronize(D.evH2D[b]));   // staging free
            reqs.resize(nch);
            notes.resize(nch);
            for (int c = 0; c < nch; ++c) {
                if (local)
                    MPI_Irecv(&notes[c], 1, MPI_INT, root, tag, MPI_COMM_WORLD, &reqs[c]);
                else
                    MPI_Irecv(chunkPtr(c), chunkCount(c), MPI_DOUBLE, root, tag, MPI_COMM_WORLD,
                              &reqs[c]);
            }
            // device buffer free once all earlier readers are done
            CUDA_CHECK(cudaStreamWaitEvent(sUp, D.evUsedComp[b], 0));
            CUDA_CHECK(cudaStreamWaitEvent(sUp, D.evUsedHi[b], 0));
            const bool nextIsMine = k + 1 < D.nblk && D.owner(k + 1) == me;
            for (int c = 0; c < nch; ++c) {
                MPI_Wait(&reqs[c], MPI_STATUS_IGNORE);
                const size_t r0 = (size_t)c * CHUNK_ROWS;
                CUDA_CHECK(cudaMemcpyAsync(D.dP[b] + r0 * nbk, chunkPtr(c),
                                           chunkCount(c) * sizeof(double), cudaMemcpyHostToDevice,
                                           sUp));
                if (c == 0 && nextIsMine) {
                    // The first chunk holds all rows needed for the diagonal block of
                    // block column k+1: update it and ship it to the CPU right away
                    CUDA_CHECK(cudaEventRecord(D.evFirst, sUp));
                    CUDA_CHECK(cudaStreamWaitEvent(sHi, D.evFirst, 0));
                    CUDA_CHECK(cudaStreamWaitEvent(sHi, D.evNext, 0));
                    const int lj = D.localIndex(k + 1);
                    update(sHi, k, lj, lj + 1, 0, NB);
                    startDiag(k + 1);
                    earlyDiag = k + 1;
                }
            }
            CUDA_CHECK(cudaEventRecord(D.evH2D[b], sUp));
            CUDA_CHECK(cudaEventRecord(D.evReady[b], sUp));
            if (local) pendingAcks.push_back({k, root, b});
        }
        return -1;
    };

    const int nthreads = (me == 0) ? std::max(2, std::min(omp_get_max_threads(), 9)) : 1;
    std::vector<std::atomic<int>> assembledBy(nthreads);
    for (auto& v : assembledBy) v.store(0);
    long failed = -1;

#pragma omp parallel num_threads(nthreads)
    {
        const int tid = omp_get_thread_num();
        if (tid == 0) {
            // Main thread: drives the GPU, the diagonal blocks and all MPI communication
            if (D.nblk > 0) {
                int f = -1;
                if (D.owner(0) == me) {
                    startDiag(0);
                    f = finishFactor(0);
                }
                failed = sharePanel(0, f);
                if (failed < 0) arrived.store(1, std::memory_order_release);
            }
            for (int k = 0; k < D.nblk && failed < 0; ++k) {
                const int b = k % NBUF;
                const int nbk = D.blockWidth(k);
                CUDA_CHECK(cudaStreamWaitEvent(sHi, D.evReady[b], 0));
                CUDA_CHECK(cudaStreamWaitEvent(sComp, D.evReady[b], 0));
                int ljs = D.firstLocalAfter(k);
                const bool look = k + 1 < D.nblk && D.owner(k + 1) == me;
                if (look) {
                    // Lookahead on the high-priority stream: diagonal block of column k+1
                    // first, then the rest of the column while the CPU factors the block
                    if (earlyDiag != k + 1) {
                        CUDA_CHECK(cudaStreamWaitEvent(sHi, D.evNext, 0));
                        update(sHi, k, ljs, ljs + 1, 0, NB);
                        startDiag(k + 1);
                    }
                    update(sHi, k, ljs, ljs + 1, NB, n);
                    ++ljs;
                }
                CUDA_CHECK(cudaEventRecord(D.evUsedHi[b], sHi));
                // Remaining trailing update; the next local column separately so that
                // its lookahead in the following step does not wait for the whole update
                update(sComp, k, ljs, std::min(ljs + 1, D.nloc), 0, n);
                CUDA_CHECK(cudaEventRecord(D.evNext, sComp));
                update(sComp, k, ljs + 1, D.nloc, 0, n);
                CUDA_CHECK(cudaEventRecord(D.evUsedComp[b], sComp));
                if (k + 1 < D.nblk) {
                    const int f = look ? finishFactor(k + 1) : -1;
                    failed = sharePanel(k + 1, f);
                    if (failed < 0) arrived.store(k + 2, std::memory_order_release);
                }
            }
            CUDA_CHECK(cudaDeviceSynchronize());
            stop.store(true, std::memory_order_release);
            // Settle the slot protocol
            flushAcks(INT_MAX);
            while (ownCollected < ownShared.size()) collectAcks(ownShared[ownCollected]);
        } else {
            // Helper threads (rank 0 only)
            const int w = tid - 1, nw = nthreads - 1;
            for (int k = 0; k < D.nblk; ++k) {
                while (arrived.load(std::memory_order_acquire) <= k &&
                       !stop.load(std::memory_order_acquire)) {
                }
                if (arrived.load(std::memory_order_acquire) <= k) break;
                assembleBlock(D, A, k, w, nw);
                // block column k is complete once every helper is done with it
                assembledBy[w].store(k + 1, std::memory_order_release);
                if (w == 0) {
                    for (int q = 0; q < nw; ++q)
                        while (assembledBy[q].load(std::memory_order_acquire) <= k) {
                        }
                    assembled.store(k + 1, std::memory_order_release);
                }
            }
        }
    }

    if (failed >= 0 && me == 0)
        printf("Error: Matrix is not positive definite at diagonal element %ld\n", failed);
    return failed < 0;
}

// Generate a symmetric positive definite matrix (local block columns only)
void generatePositiveDefiniteMatrix(Dist& D) {
    // Method: Create A = B * B^T where B is random
    // This guarantees A is positive semi-definite
    // Then add identity to make it strictly positive definite
    const size_t n = D.n;
    std::vector<double> B(n * n);
    unsigned int seed = 42;

    // Generate random matrix B (sequential random stream, identical on every rank)
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

    double* dB;
    CUDA_CHECK(cudaMalloc(&dB, std::max<size_t>(1, n * n) * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(dB, B.data(), n * n * sizeof(double), cudaMemcpyHostToDevice));

    // Compute A = B * B^T for the owned block columns: A(i, j) = sum_k B[i][k] * B[j][k]
    for (int lj = 0; lj < D.nloc; ++lj) {
        const int j = D.rank + D.nprocs * lj;
        const int j0 = j * NB, nbj = D.blockWidth(j);
        gemm<true, true>(0, (int)n, nbj, (int)n, 1.0, dB, n, dB + (size_t)j0 * n, n, 0.0,
                         D.panel(lj), n);
        // Add diagonal dominance to ensure positive definiteness
        addDiagKernel<<<(nbj + 127) / 128, 128>>>(D.panel(lj), n, j0, nbj, (double)n);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(dB));
}

bool validateCholesky(const std::vector<double>& L, const Dist& D, const double* dAorig) {
    // Validate by computing L * L^T and comparing with original matrix
    // (each rank checks its own block columns on its GPU)
    const size_t n = D.n;
    std::vector<double> Lb;
    const double* Lp = L.data();
    if (D.rank != 0) {
        Lb.resize(n * n);
        Lp = Lb.data();
    }
    // Broadcast L in chunks to stay within int counts
    const size_t chunk = (size_t)1 << 28;
    for (size_t o = 0; o < n * n; o += chunk)
        MPI_Bcast((void*)(Lp + o), (int)std::min(chunk, n * n - o), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double *dL, *dR;
    unsigned long long* dErr;
    CUDA_CHECK(cudaMalloc(&dL, std::max<size_t>(1, n * n) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dR, std::max<size_t>(1, n * NB) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dErr, 2 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(dErr, 0, 2 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemcpy(dL, Lp, n * n * sizeof(double), cudaMemcpyHostToDevice));

    for (int lj = 0; lj < D.nloc; ++lj) {
        const int j = D.rank + D.nprocs * lj;
        const int j0 = j * NB, nbj = D.blockWidth(j);
        // reconstructed(i, j) = sum_k L[i][k] * L[j][k]
        gemm<true, true>(0, (int)n, nbj, (int)n, 1.0, dL, n, dL + (size_t)j0 * n, n, 0.0, dR, n);
        errorKernel<<<512, 256>>>(dR, dAorig + (size_t)lj * NB * n, n * nbj, dErr);
        CUDA_CHECK(cudaGetLastError());
    }
    unsigned long long hErr[2];
    CUDA_CHECK(cudaMemcpy(hErr, dErr, sizeof(hErr), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dL));
    CUDA_CHECK(cudaFree(dR));
    CUDA_CHECK(cudaFree(dErr));

    double loc[2], glob[2];
    memcpy(&loc[0], &hErr[0], sizeof(double));
    memcpy(&loc[1], &hErr[1], sizeof(double));
    MPI_Allreduce(loc, glob, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    const double maxError = glob[0];
    const double relError = glob[1];

    if (D.rank == 0) {
        printf("Max absolute error: %.10e\n", maxError);
        printf("Max relative error: %.10e\n", relError);
    }

    // Check if error is within tolerance
    if (relError > 1e-6) {
        if (D.rank == 0) printf("Validation failed: relative error too large\n");
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
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    Dist D;
    MPI_Comm_rank(MPI_COMM_WORLD, &D.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &D.nprocs);
    const bool root = D.rank == 0;

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
            if (root) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    // One GPU per rank (round-robin over the node-local ranks)
    {
        MPI_Comm local;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, D.rank, MPI_INFO_NULL, &local);
        int localRank;
        MPI_Comm_rank(local, &localRank);
        MPI_Comm_free(&local);
        int ndev = 0;
        CUDA_CHECK(cudaGetDeviceCount(&ndev));
        if (ndev == 0) {
            fprintf(stderr, "No CUDA device found\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        CUDA_CHECK(cudaSetDevice(localRank % ndev));
    }

    if (root) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    D.n = n;
    D.nblk = (int)((n + NB - 1) / NB);
    D.nloc = D.rank < D.nblk ? (D.nblk - 1 - D.rank) / D.nprocs + 1 : 0;

    // Allocate matrix (local block columns on the GPU, full result on rank 0)
    CUDA_CHECK(cudaMalloc(&D.dA, std::max<size_t>(1, n * (size_t)D.nloc * NB) * sizeof(double)));
    std::vector<double> A;
    double* dAorig = nullptr;
    if (root) A.resize(n * n);
    D.allocWork();

    // Generate positive definite matrix
    if (root) printf("Generating positive definite matrix...\n");
    generatePositiveDefiniteMatrix(D);

    if (validate) {
        // Save original for validation
        const size_t bytes = std::max<size_t>(1, n * (size_t)D.nloc * NB) * sizeof(double);
        CUDA_CHECK(cudaMalloc(&dAorig, bytes));
        CUDA_CHECK(cudaMemcpy(dAorig, D.dA, bytes, cudaMemcpyDeviceToDevice));
    }

    // Perform Cholesky decomposition
    if (root) printf("Computing Cholesky decomposition...\n");
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    bool success = choleskyDecomposition(D, A);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    D.freeWork();
    CUDA_CHECK(cudaFree(D.dA));

    if (!success) {
        if (root) printf("Cholesky decomposition failed\n");
        if (dAorig) CUDA_CHECK(cudaFree(dAorig));
        MPI_Finalize();
        return 1;
    }

    if (root) {
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        // Print results for external validation
        if (printResults) {
            print_results(A, "CholeskyL");
        }
    }

    int rc = 0;
    // Validation
    if (validate) {
        if (root) printf("Validating result...\n");
        bool valid = validateCholesky(A, D, dAorig);
        CUDA_CHECK(cudaFree(dAorig));

        if (root) {
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
        rc = valid ? 0 : 1;
    }

    MPI_Finalize();
    return rc;
}
