#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

// Hybrid MPI + OpenMP + CUDA Cholesky decomposition.
//
// Block columns of the matrix are distributed over the MPI ranks (load
// balanced by block weight) and each rank keeps its block columns on its GPU.
// The owner of a block column factors its diagonal block on the CPU (a small
// panel is not efficient on the GPU's FP64 units), solves the panel on the GPU
// and publishes the finished block column.  On a single node the published
// block columns travel through an MPI shared-memory window; across nodes they
// are broadcast.  A one-step lookahead on a high-priority CUDA stream overlaps
// the next panel factorization with the current trailing updates.  Every rank
// updates its own trailing block columns with a tuned CUDA GEMM.  OpenMP is
// used for the host-side matrix assembly, the diagonal block
// factorizations/inversions and the final validation.

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err__ = (call);                                               \
        if (err__ != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err__), \
                    __FILE__, __LINE__);                                          \
            MPI_Abort(MPI_COMM_WORLD, 1);                                         \
        }                                                                         \
    } while (0)

static constexpr int NB = 128;        // block column width
static constexpr int TM = 64;         // GEMM tile rows
static constexpr int TN = 64;         // GEMM tile columns
static constexpr int TK = 16;         // GEMM tile depth
static constexpr int GTHREADS = 256;  // GEMM threads per block

// ---------------------------------------------------------------------------
// C(M,N) = alpha * sum_k A(i,k) * B(j,k) + beta * C
// All matrices are column-major.  A is M x K (lda), B is N x K (ldb) and
// C is M x N (ldc).  This is the only GEMM shape needed by the algorithm:
// both the panel solve and the trailing update are products of two matrices
// stored with the reduction index as column.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(GTHREADS)
gemmNTKernel(int M, int N, int K, double alpha,
             const double* __restrict__ A, int lda,
             const double* __restrict__ B, int ldb,
             double beta, double* __restrict__ C, int ldc) {
    __shared__ double As[TK][TM];
    __shared__ double Bs[TK][TN];

    const int tid = threadIdx.x;
    const int tx = tid & 15;
    const int ty = tid >> 4;
    const int i0 = blockIdx.x * TM;
    const int j0 = blockIdx.y * TN;

    double acc[4][4];
#pragma unroll
    for (int a = 0; a < 4; ++a)
#pragma unroll
        for (int b = 0; b < 4; ++b) acc[a][b] = 0.0;

    for (int k0 = 0; k0 < K; k0 += TK) {
        // A tile: 4 coalesced double loads per thread
#pragma unroll
        for (int q = 0; q < (TM * TK) / GTHREADS; ++q) {
            const int e = tid + q * GTHREADS;
            const int i = e % TM;
            const int k = e / TM;
            const int gi = i0 + i;
            const int gk = k0 + k;
            As[k][i] = (gi < M && gk < K) ? A[(size_t)gi + (size_t)gk * lda] : 0.0;
        }
        // B tile
#pragma unroll
        for (int q = 0; q < (TN * TK) / GTHREADS; ++q) {
            const int e = tid + q * GTHREADS;
            const int j = e % TN;
            const int k = e / TN;
            const int gj = j0 + j;
            const int gk = k0 + k;
            Bs[k][j] = (gj < N && gk < K) ? B[(size_t)gj + (size_t)gk * ldb] : 0.0;
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
        const int j = j0 + ty + 16 * b;
        if (j >= N) continue;
#pragma unroll
        for (int a = 0; a < 4; ++a) {
            const int i = i0 + tx + 16 * a;
            if (i >= M) continue;
            double* c = &C[(size_t)i + (size_t)j * ldc];
            *c = (beta == 0.0) ? alpha * acc[a][b] : fma(alpha, acc[a][b], beta * (*c));
        }
    }
}

static void gemmNT(cudaStream_t s, int M, int N, int K, double alpha,
                   const double* A, int lda, const double* B, int ldb,
                   double beta, double* C, int ldc) {
    if (M <= 0 || N <= 0 || K <= 0) return;
    dim3 grid((M + TM - 1) / TM, (N + TN - 1) / TN);
    gemmNTKernel<<<grid, GTHREADS, 0, s>>>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void addDiagonalKernel(double* __restrict__ panel, size_t n,
                                  int j0, int nbj, double value) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < nbj) panel[(size_t)c * n + j0 + c] += value;
}

// ---------------------------------------------------------------------------
// Host-side 128 x 128 diagonal block factorization and inversion
// ---------------------------------------------------------------------------

// Factor the lower triangular part of the m x m column-major block in place.
// Returns the failing diagonal index or -1.
static int factorDiagonalBlock(double* M, int m) {
    int j = 0;
    for (; j + 4 <= m; j += 4) {
        for (int jj = j; jj < j + 4; ++jj) {
            double* colj = M + (size_t)jj * m;
            const double val = colj[jj];
            if (val <= 0.0) return jj;
            const double d = sqrt(val);
            colj[jj] = d;
            for (int i = jj + 1; i < m; ++i) colj[i] /= d;
            for (int c = jj + 1; c < j + 4; ++c) {
                double* colc = M + (size_t)c * m;
                const double f = colj[c];
                for (int i = c; i < m; ++i) colc[i] -= colj[i] * f;
            }
        }
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
    for (; j < m; ++j) {
        double* colj = M + (size_t)j * m;
        const double val = colj[j];
        if (val <= 0.0) return j;
        const double d = sqrt(val);
        colj[j] = d;
        for (int i = j + 1; i < m; ++i) colj[i] /= d;
        for (int c = j + 1; c < m; ++c) {
            double* colc = M + (size_t)c * m;
            const double f = colj[c];
            for (int i = c; i < m; ++i) colc[i] -= colj[i] * f;
        }
    }
    return -1;
}

// Inverse of the lower triangular m x m block L (column-major) into Inv,
// which is an NB x NB column-major buffer (upper part and padding zero).
static void invertLowerTriangular(const double* L, int m, double* Inv) {
    std::fill(Inv, Inv + (size_t)NB * NB, 0.0);
#pragma omp parallel
    {
        double x[NB];
#pragma omp for schedule(static)
        for (int j = 0; j < m; ++j) {
            for (int i = 0; i < j; ++i) x[i] = 0.0;
            x[j] = 1.0 / L[(size_t)j * m + j];
            for (int i = j + 1; i < m; ++i) {
                double s = 0.0;
                for (int k = j; k < i; ++k)
                    s += L[(size_t)k * m + i] * x[k];
                x[i] = -s / L[(size_t)i * m + i];
            }
            for (int i = j; i < m; ++i) Inv[(size_t)j * NB + i] = x[i];
        }
    }
}

// ---------------------------------------------------------------------------
// Distributed context
// ---------------------------------------------------------------------------
struct Dist {
    int rank = 0;
    int nprocs = 1;
    size_t n = 0;
    int nblk = 0;
    int nloc = 0;

    std::vector<int> owner;      // block -> rank
    std::vector<int> localIndex; // block -> local block index on its owner
    std::vector<int> myBlocks;   // local block index -> global block index

    double* dA = nullptr;       // local block columns, column-major, ld = n
    double* dPanel = nullptr;   // two block-column buffers, ld = rows
    double* dInv = nullptr;     // NB x NB inverse of the diagonal factor
    double* hPanel = nullptr;   // pinned host staging for the block column
    double* hInv = nullptr;     // pinned host staging for the inverse
    double* hDiag = nullptr;    // pinned host staging for the diagonal block

    cudaStream_t stream = nullptr;   // trailing updates and panel uploads
    cudaStream_t streamHi = nullptr; // critical-path lookahead work
    cudaEvent_t evH2D[2];            // panel buffer ready on the device

    // When all ranks live on one node, finished block columns are published in
    // an MPI shared-memory window.  Every rank then copies the owner's data
    // straight to its GPU instead of paying for a large MPI_Bcast.
    bool useShm = false;
    MPI_Comm nodeComm = MPI_COMM_NULL;
    int nodeSize = 1;
    int nodeRank = 0;
    MPI_Win win = MPI_WIN_NULL;
    std::vector<double*> shmSlots; // world rank -> published block column

    double* panel(int lj) const { return dA + (size_t)lj * NB * n; }
    double* panelBuf(int b) const { return dPanel + (size_t)b * n * NB; }

    void setupOwnership() {
        nblk = (int)((n + NB - 1) / NB);
        owner.assign(nblk, 0);
        localIndex.assign(nblk, 0);
        myBlocks.clear();

        // Greedy load-balanced assignment: the trailing update weight of block
        // column k is roughly (n - k*NB)^2.  Assign the largest blocks first to
        // the least loaded rank.  This keeps ranks balanced although the lower
        // triangle has non-uniform block weights.
        std::vector<std::pair<long long, int>> order;
        order.reserve(nblk);
        for (int k = 0; k < nblk; ++k) {
            const long long rows = (long long)n - (long long)k * NB;
            order.push_back({rows * rows, k});
        }
        std::sort(order.begin(), order.end(),
                  [](const std::pair<long long, int>& a,
                     const std::pair<long long, int>& b) {
                      if (a.first != b.first) return a.first > b.first;
                      return a.second < b.second;
                  });
        std::vector<long long> load(nprocs, 0);
        for (const auto& w : order) {
            int best = 0;
            for (int r = 1; r < nprocs; ++r)
                if (load[r] < load[best]) best = r;
            owner[w.second] = best;
            load[best] += w.first;
        }
        std::vector<int> counter(nprocs, 0);
        for (int k = 0; k < nblk; ++k) {
            const int r = owner[k];
            localIndex[k] = counter[r]++;
        }
        nloc = counter[rank];
        for (int k = 0; k < nblk; ++k)
            if (owner[k] == rank) myBlocks.push_back(k);
    }

    void allocateWork() {
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        int loPrio = 0, hiPrio = 0;
        CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&loPrio, &hiPrio));
        CUDA_CHECK(cudaStreamCreateWithPriority(&streamHi, cudaStreamNonBlocking,
                                                hiPrio));
        CUDA_CHECK(cudaEventCreateWithFlags(&evH2D[0], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&evH2D[1], cudaEventDisableTiming));

        const size_t panelCap = std::max<size_t>(1, 2 * n * (size_t)NB);
        CUDA_CHECK(cudaMalloc(&dPanel, panelCap * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&dInv, (size_t)NB * NB * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&hPanel, panelCap * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&hInv, (size_t)NB * NB * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&hDiag, (size_t)NB * NB * sizeof(double)));

        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                            MPI_INFO_NULL, &nodeComm);
        MPI_Comm_size(nodeComm, &nodeSize);
        MPI_Comm_rank(nodeComm, &nodeRank);
        if (nodeSize == nprocs) {
            useShm = true;
            double* base = nullptr;
            const size_t slotBytes =
                std::max<size_t>(1, n * (size_t)NB) * sizeof(double);
            MPI_Win_allocate_shared((MPI_Aint)slotBytes, sizeof(double),
                                    MPI_INFO_NULL, MPI_COMM_WORLD, &base, &win);
            shmSlots.resize(nprocs);
            for (int r = 0; r < nprocs; ++r) {
                MPI_Aint sz = 0;
                int disp = 0;
                MPI_Win_shared_query(win, r, &sz, &disp, &shmSlots[r]);
                if (cudaHostRegister(shmSlots[r], (size_t)sz,
                                     cudaHostRegisterPortable) != cudaSuccess)
                    cudaGetLastError();
            }
        }
    }

    void freeWork() {
        if (useShm) {
            for (double* p : shmSlots)
                if (cudaHostUnregister(p) != cudaSuccess) cudaGetLastError();
            MPI_Win_free(&win);
        }
        if (nodeComm != MPI_COMM_NULL) MPI_Comm_free(&nodeComm);
        CUDA_CHECK(cudaFree(dPanel));
        CUDA_CHECK(cudaFree(dInv));
        CUDA_CHECK(cudaFreeHost(hPanel));
        CUDA_CHECK(cudaFreeHost(hInv));
        CUDA_CHECK(cudaFreeHost(hDiag));
        CUDA_CHECK(cudaEventDestroy(evH2D[0]));
        CUDA_CHECK(cudaEventDestroy(evH2D[1]));
        CUDA_CHECK(cudaStreamDestroy(streamHi));
        CUDA_CHECK(cudaStreamDestroy(stream));
    }
};

// Generate the same symmetric positive definite matrix as the reference code.
// Every rank generates the complete B (the random stream has to be identical),
// transposes it into the column-major layout, and then computes only its own
// block columns of A = B * B^T on the GPU.  OpenMP performs the transpose.
static void generatePositiveDefiniteMatrix(Dist& D) {
    const size_t n = D.n;
    const size_t nn = n * n;

    std::vector<double> B(nn);
    unsigned int seed = 42;
    for (size_t i = 0; i < nn; ++i)
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;

    std::vector<double> Bcm(nn);
    const int BS = 32;
    const int ni = (int)n;
#pragma omp parallel for collapse(2) schedule(static)
    for (int ii = 0; ii < ni; ii += BS) {
        for (int kk = 0; kk < ni; kk += BS) {
            const int iEnd = std::min(ii + BS, ni);
            const int kEnd = std::min(kk + BS, ni);
            for (int i = ii; i < iEnd; ++i)
                for (int k = kk; k < kEnd; ++k)
                    Bcm[(size_t)k * n + i] = B[(size_t)i * n + k];
        }
    }
    B.clear();
    B.shrink_to_fit();

    double* dB = nullptr;
    CUDA_CHECK(cudaMalloc(&dB, std::max<size_t>(1, nn) * sizeof(double)));
    CUDA_CHECK(cudaMemcpyAsync(dB, Bcm.data(), nn * sizeof(double),
                               cudaMemcpyHostToDevice, D.stream));
    Bcm.clear();
    Bcm.shrink_to_fit();

    for (int lj = 0; lj < D.nloc; ++lj) {
        const int k = D.myBlocks[lj];
        const int j0 = k * NB;
        const int nbj = (int)std::min<size_t>(NB, n - (size_t)j0);
        gemmNT(D.stream, (int)n, nbj, (int)n, 1.0, dB, (int)n, dB + j0, (int)n,
               0.0, D.panel(lj), (int)n);
        addDiagonalKernel<<<(nbj + 127) / 128, 128, 0, D.stream>>>(
            D.panel(lj), n, j0, nbj, (double)n);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaStreamSynchronize(D.stream));
    CUDA_CHECK(cudaFree(dB));
}

// Save one finished block column (column-major, ld = rows) into the column-major
// result buffer on rank 0.  Each column is a single contiguous copy, which is
// much cheaper than transposing every panel on the critical path.
static void storeBlock(const Dist& D, std::vector<double>& resultCol, int k,
                       const double* src) {
    const size_t n = D.n;
    const size_t k0 = (size_t)k * NB;
    const int rows = (int)(n - k0);
    const int nbk = std::min<int>(NB, rows);
#pragma omp parallel for schedule(static)
    for (int c = 0; c < nbk; ++c) {
        const size_t j = k0 + (size_t)c;
        std::memcpy(&resultCol[j * n + k0], src + (size_t)c * rows,
                    (size_t)rows * sizeof(double));
    }
}

// Convert the column-major lower factor into the row-major result layout and
// zero the upper triangle.  Tiled to keep both source and destination accesses
// cache friendly.
static void transposeResult(const Dist& D, const std::vector<double>& resultCol,
                            std::vector<double>& A) {
    const int n = (int)D.n;
    const int B = 32;
#pragma omp parallel for collapse(2) schedule(static)
    for (int ii = 0; ii < n; ii += B) {
        for (int jj = 0; jj < n; jj += B) {
            const int iEnd = std::min(ii + B, n);
            const int jEnd = std::min(jj + B, n);
            for (int i = ii; i < iEnd; ++i) {
                double* dst = &A[(size_t)i * n];
                for (int j = jj; j < jEnd; ++j)
                    dst[j] = (i >= j) ? resultCol[(size_t)j * n + i] : 0.0;
            }
        }
    }
}

// Returns true on success.  On rank 0, A receives the complete row-major factor
// (upper triangular part zero) exactly like the sequential implementation.
static bool choleskyDecomposition(Dist& D, std::vector<double>& A,
                                  std::vector<double>& resultCol,
                                  bool assembleResult) {
    const int n = (int)D.n;
    const int me = D.rank;
    cudaStream_t s = D.stream;
    cudaStream_t hi = D.streamHi;
    int globalFailure = -1;

    if (D.nblk == 0) return true;

    // Owner-only: factor and publish block column k.  Runs on the high-priority
    // stream so that it can overlap with the trailing updates of the previous
    // block column.  The output goes to the device buffer with parity k & 1.
    auto preparePanel = [&](int k) -> int {
        const int k0 = k * NB;
        const int rows = n - k0;
        const int nbk = std::min(NB, rows);
        const int b = k & 1;
        double* base = D.panel(D.localIndex[k]);

        CUDA_CHECK(cudaMemcpy2DAsync(D.hDiag, (size_t)nbk * sizeof(double),
                                     base + k0, (size_t)n * sizeof(double),
                                     (size_t)nbk * sizeof(double), nbk,
                                     cudaMemcpyDeviceToHost, hi));
        CUDA_CHECK(cudaStreamSynchronize(hi));

        const int f = factorDiagonalBlock(D.hDiag, nbk);
        if (f >= 0) return k0 + f;

        invertLowerTriangular(D.hDiag, nbk, D.hInv);
        CUDA_CHECK(cudaMemcpyAsync(D.dInv, D.hInv,
                                   (size_t)NB * NB * sizeof(double),
                                   cudaMemcpyHostToDevice, hi));
        // L_ik = A_ik * Inv^T for the rows below the diagonal block.
        gemmNT(hi, rows - nbk, nbk, nbk, 1.0, base + k0 + nbk, n, D.dInv,
               NB, 0.0, D.panelBuf(b) + nbk, rows);
        // The diagonal block of the finished block column.
        CUDA_CHECK(cudaMemcpy2DAsync(D.panelBuf(b), (size_t)rows * sizeof(double),
                                     D.hDiag, (size_t)nbk * sizeof(double),
                                     (size_t)nbk * sizeof(double), nbk,
                                     cudaMemcpyHostToDevice, hi));
        // Publish the complete block column for the other ranks.
        double* publish = D.useShm ? D.shmSlots[me] : D.hPanel;
        CUDA_CHECK(cudaMemcpyAsync(publish, D.panelBuf(b),
                                   (size_t)rows * nbk * sizeof(double),
                                   cudaMemcpyDeviceToHost, hi));
        CUDA_CHECK(cudaStreamSynchronize(hi));
        if (D.useShm) MPI_Win_sync(D.win);
        return -1;
    };

    // Publish block column 0 before entering the pipeline.
    int localFailure = -1;
    if (me == D.owner[0]) localFailure = preparePanel(0);
    MPI_Allreduce(&localFailure, &globalFailure, 1, MPI_INT, MPI_MAX,
                  MPI_COMM_WORLD);
    if (globalFailure >= 0) {
        if (me == 0)
            printf("Error: Matrix is not positive definite at diagonal element %d\n",
                   globalFailure);
        return false;
    }
    if (D.useShm) {
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Win_sync(D.win);
    } else {
        MPI_Bcast(D.hPanel, n * std::min<int>(NB, n), MPI_DOUBLE, D.owner[0],
                  MPI_COMM_WORLD);
    }
    const double* published = D.useShm ? D.shmSlots[D.owner[0]] : D.hPanel;

    for (int k = 0; k < D.nblk; ++k) {
        const int k0 = k * NB;
        const int rows = n - k0;
        const int nbk = std::min(NB, rows);
        const int b = k & 1;
        const int next = k + 1;
        const bool haveNext = next < D.nblk;
        const int nextRoot = haveNext ? D.owner[next] : -1;

        // Upload the current block column on the normal stream.  The previous
        // parity buffer is written by the lookahead on the high-priority stream.
        CUDA_CHECK(cudaMemcpyAsync(D.panelBuf(b), published,
                                   (size_t)rows * nbk * sizeof(double),
                                   cudaMemcpyHostToDevice, s));
        CUDA_CHECK(cudaEventRecord(D.evH2D[b], s));

        if (me == 0 && assembleResult) storeBlock(D, resultCol, k, published);

        // Lookahead owner updates and factors the next diagonal block while the
        // other ranks (and its own remaining block columns) process panel k.
        if (haveNext && me == nextRoot) {
            const int nk0 = next * NB;
            const int nrows = n - nk0;
            const int nnbk = std::min(NB, nrows);
            const int lj = D.localIndex[next];
            const double* p = D.panelBuf(b) + (nk0 - k0);
            CUDA_CHECK(cudaStreamWaitEvent(hi, D.evH2D[b], 0));
            gemmNT(hi, n - nk0, nnbk, nbk, -1.0, p, rows, p, rows, 1.0,
                   D.panel(lj) + nk0, n);
        }

        // Trailing updates for block columns j > k.  The block column needed by
        // the lookahead owner is handled on the high-priority stream instead.
        for (int lj = 0; lj < D.nloc; ++lj) {
            const int jblk = D.myBlocks[lj];
            if (jblk <= k) continue;
            if (haveNext && jblk == next && me == nextRoot) continue;
            const int j0 = jblk * NB;
            const int nbj = std::min<int>(NB, n - j0);
            const double* p = D.panelBuf(b) + (j0 - k0);
            gemmNT(s, n - j0, nbj, nbk, -1.0, p, rows, p, rows, 1.0,
                   D.panel(lj) + j0, n);
        }

        // The shared slot (or host staging buffer) may be reused only after the
        // upload has consumed it.
        CUDA_CHECK(cudaEventSynchronize(D.evH2D[b]));
        if (D.useShm) MPI_Barrier(MPI_COMM_WORLD);

        if (!haveNext) break;

        localFailure = -1;
        if (me == nextRoot) localFailure = preparePanel(next);
        MPI_Allreduce(&localFailure, &globalFailure, 1, MPI_INT, MPI_MAX,
                      MPI_COMM_WORLD);
        if (globalFailure >= 0) break;

        if (D.useShm) {
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Win_sync(D.win);
            published = D.shmSlots[nextRoot];
        } else {
            const int nrows = n - next * NB;
            const int nnbk = std::min(NB, nrows);
            MPI_Bcast(D.hPanel, nrows * nnbk, MPI_DOUBLE, nextRoot,
                      MPI_COMM_WORLD);
            published = D.hPanel;
        }
    }

    CUDA_CHECK(cudaStreamSynchronize(s));
    CUDA_CHECK(cudaStreamSynchronize(hi));

    if (me == 0 && assembleResult && globalFailure < 0)
        transposeResult(D, resultCol, A);

    if (globalFailure >= 0 && me == 0)
        printf("Error: Matrix is not positive definite at diagonal element %d\n",
               globalFailure);
    return globalFailure < 0;
}

// ---------------------------------------------------------------------------
// Validation (rank 0, OpenMP).  The original matrix is gathered block column by
// block column from the distributed GPU copies.
// ---------------------------------------------------------------------------
static void assembleOriginal(Dist& D, const double* dAorig,
                             std::vector<double>& Aorig) {
    const size_t n = D.n;
    const size_t perBlock = (size_t)NB * n;
    const size_t count = (size_t)D.nloc * perBlock;
    std::vector<double> buf(std::max<size_t>(1, count));
    if (count)
        CUDA_CHECK(cudaMemcpy(buf.data(), dAorig, count * sizeof(double),
                              cudaMemcpyDeviceToHost));

    auto addBlocks = [&](int r, const double* data) {
        for (int k = 0; k < D.nblk; ++k) {
            if (D.owner[k] != r) continue;
            const int j0 = k * NB;
            const int nbj = (int)std::min<size_t>(NB, n - (size_t)j0);
            const double* blk = data + (size_t)D.localIndex[k] * perBlock;
            for (int c = 0; c < nbj; ++c) {
                const size_t j = (size_t)j0 + c;
                const double* col = blk + (size_t)c * n;
                for (size_t i = 0; i < n; ++i) Aorig[i * n + j] = col[i];
            }
        }
    };

    if (D.rank == 0) {
        addBlocks(0, buf.data());
        for (int r = 1; r < D.nprocs; ++r) {
            int nlocR = 0;
            for (int k = 0; k < D.nblk; ++k)
                if (D.owner[k] == r) ++nlocR;
            const size_t cnt = (size_t)nlocR * perBlock;
            std::vector<double> rbuf(cnt);
            if (cnt)
                MPI_Recv(rbuf.data(), (int)cnt, MPI_DOUBLE, r, 7781,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (cnt) addBlocks(r, rbuf.data());
        }
    } else if (count) {
        MPI_Send(buf.data(), (int)count, MPI_DOUBLE, 0, 7781, MPI_COMM_WORLD);
    }
}

static bool validateCholesky(const std::vector<double>& L,
                             const std::vector<double>& A_orig,
                             const size_t n) {
    std::vector<double> reconstructed(n * n);

#pragma omp parallel for schedule(static)
    for (long long ii = 0; ii < (long long)n; ++ii) {
        const size_t i = (size_t)ii;
        const double* Li = &L[i * n];
        double* Ri = &reconstructed[i * n];
        for (size_t j = 0; j < n; ++j) {
            const double* Lj = &L[j * n];
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) sum += Li[k] * Lj[k];
            Ri[j] = sum;
        }
    }

    double maxError = 0.0;
    double relError = 0.0;
#pragma omp parallel for schedule(static) reduction(max : maxError) \
    reduction(max : relError)
    for (long long ii = 0; ii < (long long)(n * n); ++ii) {
        const double error = fabs(reconstructed[(size_t)ii] - A_orig[(size_t)ii]);
        maxError = std::max(maxError, error);
        const double rel = error / (fabs(A_orig[(size_t)ii]) + 1e-10);
        relError = std::max(relError, rel);
    }

    printf("Max absolute error: %.10e\n", maxError);
    printf("Max relative error: %.10e\n", relError);

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
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    Dist D;
    MPI_Comm_rank(MPI_COMM_WORLD, &D.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &D.nprocs);
    const bool root = D.rank == 0;

    size_t n = 512;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = (size_t)atoi(argv[++i]);
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

    // One GPU per rank (round-robin over the node-local ranks).
    {
        MPI_Comm local = MPI_COMM_NULL;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, D.rank,
                            MPI_INFO_NULL, &local);
        int localRank = 0;
        MPI_Comm_rank(local, &localRank);
        MPI_Comm_free(&local);
        int ndev = 0;
        CUDA_CHECK(cudaGetDeviceCount(&ndev));
        if (ndev == 0) {
            if (root) fprintf(stderr, "No CUDA device found\n");
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
    D.setupOwnership();

    CUDA_CHECK(cudaMalloc(&D.dA,
                          std::max<size_t>(1, n * (size_t)D.nloc * NB) *
                              sizeof(double)));
    std::vector<double> A;
    if (root) A.resize(n * n);
    const bool assembleResult = printResults || validate;
    std::vector<double> resultCol;
    if (root && assembleResult) resultCol.resize(n * n);
    D.allocateWork();

    if (root) printf("Generating positive definite matrix...\n");
    generatePositiveDefiniteMatrix(D);

    double* dAorig = nullptr;
    if (validate) {
        const size_t bytes = std::max<size_t>(1, n * (size_t)D.nloc * NB) *
                             sizeof(double);
        CUDA_CHECK(cudaMalloc(&dAorig, bytes));
        CUDA_CHECK(cudaMemcpyAsync(dAorig, D.dA, bytes,
                                   cudaMemcpyDeviceToDevice, D.stream));
        CUDA_CHECK(cudaStreamSynchronize(D.stream));
    }

    if (root) printf("Computing Cholesky decomposition...\n");
    CUDA_CHECK(cudaStreamSynchronize(D.stream));
    MPI_Barrier(MPI_COMM_WORLD);

    auto start = std::chrono::high_resolution_clock::now();
    bool success = choleskyDecomposition(D, A, resultCol, assembleResult);
    auto end = std::chrono::high_resolution_clock::now();
    const double localMs =
        std::chrono::duration<double, std::milli>(end - start).count();
    double maxMs = 0.0;
    MPI_Allreduce(&localMs, &maxMs, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    D.freeWork();
    CUDA_CHECK(cudaFree(D.dA));
    D.dA = nullptr;

    if (!success) {
        if (root) printf("Cholesky decomposition failed\n");
        if (dAorig) CUDA_CHECK(cudaFree(dAorig));
        MPI_Finalize();
        return 1;
    }

    int rc = 0;
    if (root) {
        const long ms = (long)llround(maxMs);
        printf("Computation time: %ld ms\n", ms);

        const double ops = (double)n * n * n / 3.0;
        const double gflops = ops / (maxMs / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) print_results(A, "CholeskyL");
    }

    if (validate) {
        std::vector<double> A_orig;
        if (root) A_orig.resize(n * n);
        assembleOriginal(D, dAorig, A_orig);
        CUDA_CHECK(cudaFree(dAorig));
        dAorig = nullptr;

        if (root) {
            printf("Validating result...\n");
            const bool valid = validateCholesky(A, A_orig, n);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                rc = 1;
            }
        }
    } else if (dAorig) {
        CUDA_CHECK(cudaFree(dAorig));
    }

    MPI_Finalize();
    return rc;
}
