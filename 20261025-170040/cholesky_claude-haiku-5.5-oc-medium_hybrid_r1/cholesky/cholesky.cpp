#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// Hybrid MPI + OpenMP + CUDA Cholesky decomposition.
//
// Work decomposition:
//   MPI    - rows are distributed in blocks of kBlock rows, block-cyclically over ranks.
//            Each panel (kBlock columns) has its diagonal block factored by the owning rank;
//            the factored diagonal block and the panel's L columns are shared with all ranks.
//   CUDA   - each rank drives one GPU. The trailing update (the bulk of the O(n^3) work) is a
//            tiled kernel over the local lower triangle; the intra-panel solve is a per-row kernel.
//   OpenMP - host-side parallel loops: reordering of gathered results, the validation
//            comparison, and the per-rank thread count.
//
// Numerical semantics. Each element is computed as in the original algorithm:
//   L[i][j] = (A[i][j] - sum_{k<j} L[i][k]*L[j][k]) / L[j][j],   L[j][j] = sqrt(A[j][j] - sum_{k<j} L[j][k]^2)
// and the sum is accumulated in ascending k order. A right-looking, panel-by-panel evaluation
// keeps that order: a running sum per element is extended by each panel's terms in ascending
// k, so every element sees the same sequence of operations as the sequential code. Results are
// independent of the number of ranks. Bitwise agreement with the original binary additionally
// depends on how the compiler contracts multiply-adds (the original's generator loop is
// auto-vectorized, which mixes fused and unfused operations), so agreement is to rounding.

namespace {

constexpr size_t kBlock = 64;        // row/column block (panel width) == GEMM tile edge
constexpr int kGemmTile = 64;        // output tile edge; 16x16 threads, each computes 4x4 outputs
constexpr int kGemmK = 16;           // K-tile staged through shared memory
constexpr size_t kGenChunk = size_t(1) << 22;

static_assert(kBlock == static_cast<size_t>(kGemmTile), "one trailing-update tile per block");

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err_ = (call);                                             \
        if (err_ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err_));                                 \
            MPI_Abort(MPI_COMM_WORLD, 2);                                      \
        }                                                                      \
    } while (0)

// Block-cyclic row layout. Block q (rows q*b .. q*b+b-1) belongs to rank q % nprocs and is
// stored locally as local block q / nprocs.
struct Layout {
    int rank = 0;
    int nprocs = 1;
    size_t n = 0;
    size_t b = kBlock;
    size_t nb = 0;           // total number of blocks
    size_t localBlocks = 0;  // blocks owned by this rank
    size_t localRows = 0;    // localBlocks * b (includes padding rows beyond n)

    size_t blocksOf(int r) const {
        size_t c = 0;
        for (size_t q = static_cast<size_t>(r); q < nb; q += nprocs) ++c;
        return c;
    }
    // Global row index of local row lr on rank `r`.
    __host__ __device__ size_t globalRowOf(int r, size_t lr) const {
        return ((lr / b) * static_cast<size_t>(nprocs) + static_cast<size_t>(r)) * b + lr % b;
    }
    __host__ __device__ size_t globalRow(size_t lr) const { return globalRowOf(rank, lr); }
};

// First local row (on rank r) belonging to a block below panel p (i.e. block index > p).
size_t belowRowStart(int r, size_t p, size_t P, size_t b) {
    const size_t lb0 = (p >= static_cast<size_t>(r)) ? (p - r) / P + 1 : 0;
    return lb0 * b;
}

// C[m][nn] = sum_k X[m][k] * Y[nn][k], k accumulated in ascending order (no reassociation).
__global__ void gemmNTKernel(const double* __restrict__ X, size_t ldx,
                             const double* __restrict__ Y, size_t ldy,
                             int M, int N, int K,
                             double* __restrict__ C, size_t ldc) {
    __shared__ double xs[kGemmTile][kGemmK + 1];
    __shared__ double ys[kGemmTile][kGemmK + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * 16 + tx;
    const int m0 = blockIdx.y * kGemmTile;
    const int n0 = blockIdx.x * kGemmTile;

    double acc[4][4];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) acc[i][j] = 0.0;

    for (int k0 = 0; k0 < K; k0 += kGemmK) {
        for (int e = tid; e < kGemmTile * kGemmK; e += 256) {
            const int r = e / kGemmK;
            const int c = e % kGemmK;
            const int gk = k0 + c;
            const int gm = m0 + r;
            const int gn = n0 + r;
            xs[r][c] = (gm < M && gk < K) ? X[static_cast<size_t>(gm) * ldx + gk] : 0.0;
            ys[r][c] = (gn < N && gk < K) ? Y[static_cast<size_t>(gn) * ldy + gk] : 0.0;
        }
        __syncthreads();

        const int kmax = min(kGemmK, K - k0);
        for (int kk = 0; kk < kmax; ++kk) {
            double a[4], bv[4];
            for (int i = 0; i < 4; ++i) {
                a[i] = xs[ty + 16 * i][kk];
                bv[i] = ys[tx + 16 * i][kk];
            }
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j) acc[i][j] += a[i] * bv[j];
        }
        __syncthreads();
    }

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            const int gm = m0 + ty + 16 * i;
            const int gn = n0 + tx + 16 * j;
            if (gm < M && gn < N) C[static_cast<size_t>(gm) * ldc + gn] = acc[i][j];
        }
    }
}

// Trailing update of the running sums for the lower triangle of a set of local rows:
//   C[m][nn] (global column colStart + nn) += sum_k X[m][k] * Y[nn][k], k ascending,
// continuing from the value already stored in C. Row m is the local row rowStart + m.
// Y is the gathered panel: the rows of column tile t (one block of kGemmTile global rows)
// start at Y + yOff[t] and have leading dimension ldy.
// Only entries with global column <= global row are written; tiles lying entirely above
// the diagonal are skipped. Each tile covers exactly one block of rows (kBlock == kGemmTile).
__global__ void trailingUpdateKernel(const double* __restrict__ X, size_t ldx,
                                     const double* __restrict__ Y, const size_t* __restrict__ yOff,
                                     size_t ldy, int M, int N, int K,
                                     double* __restrict__ C, size_t ldc,
                                     Layout lay, size_t rowStart, size_t colStart) {
    __shared__ double xs[kGemmTile][kGemmK + 1];
    __shared__ double ys[kGemmTile][kGemmK + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * 16 + tx;
    const int m0 = blockIdx.y * kGemmTile;
    const int n0 = blockIdx.x * kGemmTile;

    const int mLast = min(m0 + kGemmTile, M) - 1;
    if (colStart + static_cast<size_t>(n0) > lay.globalRow(rowStart + mLast)) return;

    const double* Yt = Y + yOff[blockIdx.x];

    double acc[4][4];
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            const int gm = m0 + ty + 16 * i;
            const int gn = n0 + tx + 16 * j;
            acc[i][j] = (gm < M && gn < N) ? C[static_cast<size_t>(gm) * ldc + gn] : 0.0;
        }
    }

    for (int k0 = 0; k0 < K; k0 += kGemmK) {
        for (int e = tid; e < kGemmTile * kGemmK; e += 256) {
            const int r = e / kGemmK;
            const int c = e % kGemmK;
            const int gk = k0 + c;
            const int gm = m0 + r;
            const int gn = n0 + r;
            xs[r][c] = (gm < M && gk < K) ? X[static_cast<size_t>(gm) * ldx + gk] : 0.0;
            ys[r][c] = (gn < N && gk < K) ? Yt[static_cast<size_t>(r) * ldy + gk] : 0.0;
        }
        __syncthreads();

        const int kmax = min(kGemmK, K - k0);
        for (int kk = 0; kk < kmax; ++kk) {
            double a[4], bv[4];
            for (int i = 0; i < 4; ++i) {
                a[i] = xs[ty + 16 * i][kk];
                bv[i] = ys[tx + 16 * i][kk];
            }
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j) acc[i][j] += a[i] * bv[j];
        }
        __syncthreads();
    }

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            const int gm = m0 + ty + 16 * i;
            const int gn = n0 + tx + 16 * j;
            if (gm < M && gn < N &&
                colStart + static_cast<size_t>(gn) <= lay.globalRow(rowStart + gm)) {
                C[static_cast<size_t>(gm) * ldc + gn] = acc[i][j];
            }
        }
    }
}

// Intra-panel column solve for `nrows` local rows. Each thread owns one row and processes the
// w panel columns in order. Sb[r][jj] holds the running sum over k < J0 for (row r, column
// J0 + jj). Lh is the factored w x w diagonal block (lower triangle, ld w).
constexpr int kPanelMax = static_cast<int>(kBlock);

__global__ void panelSolveKernel(double* rows, size_t ld,
                                 const double* __restrict__ Sb, size_t ldsb,
                                 const double* __restrict__ Lh,
                                 int J0, int w, int nrows) {
    __shared__ double Ps[kPanelMax][kPanelMax];
    for (int e = threadIdx.x; e < w * w; e += blockDim.x) {
        const int jj = e / w;
        const int k = e % w;
        if (k <= jj) Ps[jj][k] = Lh[jj * w + k];
    }
    __syncthreads();

    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= nrows) return;
    double* row = rows + static_cast<size_t>(r) * ld;
    for (int jj = 0; jj < w; ++jj) {
        double sum = Sb[static_cast<size_t>(r) * ldsb + jj];
        for (int k = 0; k < jj; ++k) sum += row[J0 + k] * Ps[jj][k];
        row[J0 + jj] = (row[J0 + jj] - sum) / Ps[jj][jj];
    }
}

__global__ void addDiagKernel(double* L, Layout lay, double value) {
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t lr = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         lr < lay.localRows; lr += stride) {
        const size_t gi = lay.globalRow(lr);
        if (gi < lay.n) L[lr * lay.n + gi] += value;
    }
}

__global__ void zeroUpperKernel(double* L, Layout lay) {
    const size_t total = lay.localRows * lay.n;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t e = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; e < total; e += stride) {
        const size_t lr = e / lay.n;
        const size_t c = e % lay.n;
        if (c > lay.globalRow(lr)) L[e] = 0.0;
    }
}

unsigned int gridFor(size_t count, unsigned int threads) {
    return static_cast<unsigned int>(std::max<size_t>(1, (count + threads - 1) / threads));
}

void gemmNT(const double* X, size_t ldx, const double* Y, size_t ldy,
            size_t M, size_t N, size_t K, double* C, size_t ldc) {
    if (M == 0 || N == 0) return;
    const dim3 block(16, 16);
    const dim3 grid(static_cast<unsigned int>((N + kGemmTile - 1) / kGemmTile),
                    static_cast<unsigned int>((M + kGemmTile - 1) / kGemmTile));
    gemmNTKernel<<<grid, block>>>(X, ldx, Y, ldy, static_cast<int>(M), static_cast<int>(N),
                                  static_cast<int>(K), C, ldc);
    CUDA_CHECK(cudaGetLastError());
}

// Factors the w x w diagonal block (w <= kBlock). Sb: running sums over k < J0 (lower
// triangle), Ab: original A values, Lh: output (lower triangle). Returns the index inside the
// block of the first non-positive pivot, or -1. The block is tiny, so this runs on the host
// sequentially; the device-side work is the trailing update, not this step.
long factorDiagonalBlock(double* Lh, const double* Sb, const double* Ab, size_t w) {
    for (size_t jj = 0; jj < w; ++jj) {
        double sum = Sb[jj * w + jj];
        for (size_t k = 0; k < jj; ++k) sum += Lh[jj * w + k] * Lh[jj * w + k];
        const double val = Ab[jj * w + jj] - sum;
        if (val <= 0.0) return static_cast<long>(jj);
        Lh[jj * w + jj] = sqrt(val);
        const double diag = Lh[jj * w + jj];

        for (size_t i = jj + 1; i < w; ++i) {
            double s = Sb[i * w + jj];
            for (size_t k = 0; k < jj; ++k) s += Lh[i * w + k] * Lh[jj * w + k];
            Lh[i * w + jj] = (Ab[i * w + jj] - s) / diag;
        }
    }
    return -1;
}

// Fills the local rows with A = B * B^T + n*I, where B is generated exactly as in the original
// code (single rand_r stream, seed 42). Only the lower triangle is generated unless keepUpper
// is set (needed for validation).
void generateMatrix(double* dL, const Layout& lay, bool keepUpper) {
    const size_t n = lay.n;
    const size_t total = n * n;
    const size_t localCount = lay.localRows * n;

    if (localCount) CUDA_CHECK(cudaMemset(dL, 0, localCount * sizeof(double)));
    if (total == 0) return;

    double* dB = nullptr;
    CUDA_CHECK(cudaMalloc(&dB, total * sizeof(double)));

    std::vector<double> chunk(std::min(total, kGenChunk));
    unsigned int seed = 42;
    for (size_t t0 = 0; t0 < total; t0 += chunk.size()) {
        const size_t cnt = std::min(chunk.size(), total - t0);
        for (size_t i = 0; i < cnt; ++i) {
            chunk[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
        }
        CUDA_CHECK(cudaMemcpy(dB + t0, chunk.data(), cnt * sizeof(double), cudaMemcpyHostToDevice));
    }

    for (size_t lb = 0; lb < lay.localBlocks; ++lb) {
        const size_t q = lb * static_cast<size_t>(lay.nprocs) + static_cast<size_t>(lay.rank);
        const size_t gstart = q * lay.b;
        const size_t M = std::min(lay.b, n - gstart);
        const size_t N = keepUpper ? n : gstart + M;  // lower triangle only is used later
        gemmNT(dB + gstart * n, n, dB, n, M, N, n, dL + lb * lay.b * n, n);
    }

    addDiagKernel<<<gridFor(lay.localRows, 128), 128>>>(dL, lay, static_cast<double>(n));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(dB));
}

// Distributed right-looking blocked Cholesky. dL holds the local rows of A on entry and L on
// exit (upper triangle zeroed). Returns false if the matrix is not positive definite; in that
// case failIndex receives the global diagonal index.
bool choleskyDistributed(double* dL, const Layout& lay, size_t& failIndex) {
    const size_t n = lay.n;
    const size_t b = lay.b;
    const size_t nb = lay.nb;
    const int P = lay.nprocs;
    const int rank = lay.rank;
    const size_t localCount = lay.localRows * n;

    // dAcc[i][j]: running sum of L[i][k]*L[j][k] over the panels processed so far (k ascending).
    double* dAcc = nullptr;
    double* dLh = nullptr;
    double* dRecv = nullptr;     // gathered panel, rank-major (see displs)
    size_t* dYoff = nullptr;     // per column tile: element offset of its rows inside dRecv
    double* hSend = nullptr;
    double* hRecv = nullptr;
    CUDA_CHECK(cudaMalloc(&dAcc, std::max<size_t>(1, localCount) * sizeof(double)));
    if (localCount) CUDA_CHECK(cudaMemset(dAcc, 0, localCount * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dLh, b * b * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dRecv, std::max<size_t>(1, nb * b * b) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dYoff, std::max<size_t>(1, nb) * sizeof(size_t)));
    CUDA_CHECK(cudaMallocHost(&hSend, std::max<size_t>(1, lay.localRows * b) * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&hRecv, std::max<size_t>(1, nb * b * b) * sizeof(double)));

    std::vector<double> hSb(b * b), hA(b * b), hLh(b * b);
    std::vector<int> counts(P), displs(P);
    std::vector<size_t> hYoff(std::max<size_t>(1, nb));
    bool success = true;

    for (size_t p = 0; p < nb; ++p) {
        const size_t J0 = p * b;
        const size_t w = std::min(b, n - J0);
        const size_t J1 = J0 + w;
        const int owner = static_cast<int>(p % P);
        const size_t diagLr = (p / P) * b;  // local row of the diagonal block (owner only)

        // (1) Factor the diagonal block on its owner.
        long status = -1;
        if (rank == owner) {
            CUDA_CHECK(cudaMemcpy2D(hSb.data(), w * sizeof(double), dAcc + diagLr * n + J0,
                                    n * sizeof(double), w * sizeof(double), w, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy2D(hA.data(), w * sizeof(double), dL + diagLr * n + J0,
                                    n * sizeof(double), w * sizeof(double), w, cudaMemcpyDeviceToHost));
            std::fill(hLh.begin(), hLh.end(), 0.0);
            const long fail = factorDiagonalBlock(hLh.data(), hSb.data(), hA.data(), w);
            if (fail >= 0) {
                status = static_cast<long>(J0) + fail;
            } else {
                CUDA_CHECK(cudaMemcpy2D(dL + diagLr * n + J0, n * sizeof(double), hLh.data(),
                                        w * sizeof(double), w * sizeof(double), w,
                                        cudaMemcpyHostToDevice));
            }
        }
        MPI_Bcast(&status, 1, MPI_LONG, owner, MPI_COMM_WORLD);
        if (status >= 0) {
            failIndex = static_cast<size_t>(status);
            success = false;
            break;
        }
        MPI_Bcast(hLh.data(), static_cast<int>(w * w), MPI_DOUBLE, owner, MPI_COMM_WORLD);
        CUDA_CHECK(cudaMemcpy(dLh, hLh.data(), w * w * sizeof(double), cudaMemcpyHostToDevice));

        // (2) Intra-panel solve for the locally owned rows below the panel.
        const size_t rowStart = belowRowStart(rank, p, P, b);
        const size_t nrows = lay.localRows > rowStart ? lay.localRows - rowStart : 0;
        if (nrows) {
            panelSolveKernel<<<gridFor(nrows, 128), 128>>>(dL + rowStart * n, n,
                                                           dAcc + rowStart * n + J0, n, dLh,
                                                           static_cast<int>(J0),
                                                           static_cast<int>(w),
                                                           static_cast<int>(nrows));
            CUDA_CHECK(cudaGetLastError());
        }

        // No later columns need this panel after the last block.
        if (J1 >= n) continue;

        // (3) Share this panel's L columns (rows at or below J1) with all ranks. Ranks' segments
        //     are concatenated in rank order and copied to the device as they are.
        size_t total = 0;
        for (int r = 0; r < P; ++r) {
            const size_t rs = belowRowStart(r, p, P, b);
            const size_t rowsR = lay.blocksOf(r) * b;
            const size_t nr = rowsR > rs ? rowsR - rs : 0;
            counts[r] = static_cast<int>(nr * w);
            displs[r] = static_cast<int>(total);
            total += nr * w;
        }
        if (nrows) {
            CUDA_CHECK(cudaMemcpy2D(hSend, w * sizeof(double), dL + rowStart * n + J0,
                                    n * sizeof(double), w * sizeof(double), nrows,
                                    cudaMemcpyDeviceToHost));
        }
        MPI_Allgatherv(hSend, static_cast<int>(nrows * w), MPI_DOUBLE, hRecv, counts.data(),
                       displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        const size_t gathered = static_cast<size_t>(displs[P - 1]) + static_cast<size_t>(counts[P - 1]);
        CUDA_CHECK(cudaMemcpy(dRecv, hRecv, std::max<size_t>(1, gathered) * sizeof(double),
                              cudaMemcpyHostToDevice));

        // (4) Trailing update of the running sums for columns J1.. on the local rows below.
        // Column tile t is global block J1/b + t; its rows sit contiguously in its owner's segment.
        if (nrows) {
            const size_t numTiles = (n - J1 + kGemmTile - 1) / kGemmTile;
            for (size_t t = 0; t < numTiles; ++t) {
                const size_t cb = J1 / b + t;
                const int rc = static_cast<int>(cb % P);
                const size_t lr = (cb / P) * b;
                const size_t rs = belowRowStart(rc, p, P, b);
                hYoff[t] = static_cast<size_t>(displs[rc]) + (lr - rs) * w;
            }
            CUDA_CHECK(cudaMemcpy(dYoff, hYoff.data(), numTiles * sizeof(size_t), cudaMemcpyHostToDevice));

            const dim3 block(16, 16);
            const dim3 grid(static_cast<unsigned int>(numTiles),
                            static_cast<unsigned int>((nrows + kGemmTile - 1) / kGemmTile));
            trailingUpdateKernel<<<grid, block>>>(dL + rowStart * n + J0, n, dRecv, dYoff, w,
                                                  static_cast<int>(nrows), static_cast<int>(n - J1),
                                                  static_cast<int>(w),
                                                  dAcc + rowStart * n + J1, n, lay, rowStart, J1);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    if (success && lay.localRows) {
        zeroUpperKernel<<<gridFor(localCount, 256), 256>>>(dL, lay);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    CUDA_CHECK(cudaFreeHost(hRecv));
    CUDA_CHECK(cudaFreeHost(hSend));
    CUDA_CHECK(cudaFree(dYoff));
    CUDA_CHECK(cudaFree(dRecv));
    CUDA_CHECK(cudaFree(dLh));
    CUDA_CHECK(cudaFree(dAcc));
    return success;
}

// Gathers the full n x n row-major matrix to every rank (global row order).
void gatherGlobal(const double* dL, const Layout& lay, std::vector<double>& full) {
    const size_t n = lay.n;
    const size_t b = lay.b;
    const size_t P = static_cast<size_t>(lay.nprocs);

    std::vector<int> counts(lay.nprocs), displs(lay.nprocs);
    size_t total = 0;
    for (int rr = 0; rr < lay.nprocs; ++rr) {
        const size_t c = lay.blocksOf(rr) * b * n;
        if (total + c > static_cast<size_t>(INT_MAX)) {
            fprintf(stderr, "Matrix too large for MPI gather with this many ranks\n");
            MPI_Abort(MPI_COMM_WORLD, 3);
        }
        counts[rr] = static_cast<int>(c);
        displs[rr] = static_cast<int>(total);
        total += c;
    }

    std::vector<double> recv(total);
    const size_t localCount = lay.localRows * n;
    if (localCount) {
        CUDA_CHECK(cudaMemcpy(recv.data() + displs[lay.rank], dL, localCount * sizeof(double),
                              cudaMemcpyDeviceToHost));
    }
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, recv.data(), counts.data(), displs.data(),
                   MPI_DOUBLE, MPI_COMM_WORLD);

    full.assign(n * n, 0.0);
    for (int rr = 0; rr < lay.nprocs; ++rr) {
        const long nbl = static_cast<long>(lay.blocksOf(rr));
        #pragma omp parallel for schedule(static)
        for (long lbi = 0; lbi < nbl; ++lbi) {
            const size_t lb = static_cast<size_t>(lbi);
            for (size_t rib = 0; rib < b; ++rib) {
                const size_t lr = lb * b + rib;
                const size_t gi = (lb * P + static_cast<size_t>(rr)) * b + rib;
                if (gi < n) {
                    std::copy_n(recv.data() + displs[rr] + lr * n, n, full.data() + gi * n);
                }
            }
        }
    }
}

// Distributed validation: R = L * L^T on the local rows (L gathered on every GPU), compared
// with the original A. Returns false if the relative error exceeds the tolerance.
bool validateDistributed(const double* dL, const double* dA, const Layout& lay,
                         const std::vector<double>& Lfull) {
    const size_t n = lay.n;
    const size_t localCount = lay.localRows * n;

    double* dLfull = nullptr;
    double* dR = nullptr;
    CUDA_CHECK(cudaMalloc(&dLfull, std::max<size_t>(1, n * n) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dR, std::max<size_t>(1, localCount) * sizeof(double)));
    if (n * n != 0) {
        CUDA_CHECK(cudaMemcpy(dLfull, Lfull.data(), n * n * sizeof(double), cudaMemcpyHostToDevice));
    }
    gemmNT(dL, n, dLfull, n, lay.localRows, n, n, dR, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> R(localCount), A(localCount);
    if (localCount) {
        CUDA_CHECK(cudaMemcpy(R.data(), dR, localCount * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(A.data(), dA, localCount * sizeof(double), cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaFree(dR));
    CUDA_CHECK(cudaFree(dLfull));

    double maxError = 0.0;
    double relError = 0.0;
    #pragma omp parallel for schedule(static) reduction(max : maxError, relError)
    for (long lr = 0; lr < static_cast<long>(lay.localRows); ++lr) {
        if (lay.globalRow(static_cast<size_t>(lr)) >= n) continue;
        for (size_t j = 0; j < n; ++j) {
            const size_t idx = static_cast<size_t>(lr) * n + j;
            const double error = fabs(R[idx] - A[idx]);
            maxError = std::max(maxError, error);
            const double rel = error / (fabs(A[idx]) + 1e-10);
            relError = std::max(relError, rel);
        }
    }

    double globalMax = 0.0, globalRel = 0.0;
    MPI_Allreduce(&maxError, &globalMax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&relError, &globalRel, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    if (lay.rank == 0) {
        printf("Max absolute error: %.10e\n", globalMax);
        printf("Max relative error: %.10e\n", globalRel);
        if (globalRel > 1e-6) {
            printf("Validation failed: relative error too large\n");
            return false;
        }
    }
    return globalRel <= 1e-6;
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
    MPI_Init(&argc, &argv);

    int rank = 0;
    int nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    size_t n = 512;
    bool validate = false;
    bool printResults = false;
    int rc = -1;  // -1: continue running

    // Parse command line arguments (identical on every rank)
    for (int i = 1; i < argc && rc < 0; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            rc = 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            rc = 1;
        }
    }
    if (rc >= 0) {
        MPI_Finalize();
        return rc;
    }

    // Bind each rank to a GPU on its node and size the OpenMP team per rank.
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0;
    int localSize = 1;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_size(localComm, &localSize);

    int numDevices = 0;
    cudaGetDeviceCount(&numDevices);
    if (numDevices == 0) {
        if (rank == 0) fprintf(stderr, "Error: no CUDA device available\n");
        MPI_Abort(MPI_COMM_WORLD, 4);
    }
    CUDA_CHECK(cudaSetDevice(localRank % numDevices));

    if (getenv("OMP_NUM_THREADS") == nullptr) {
        omp_set_num_threads(std::max(1, omp_get_num_procs() / localSize));
    }

    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    Layout lay;
    lay.rank = rank;
    lay.nprocs = nprocs;
    lay.n = n;
    lay.nb = (n + kBlock - 1) / kBlock;
    lay.localBlocks = lay.blocksOf(rank);
    lay.localRows = lay.localBlocks * kBlock;

    double* dL = nullptr;
    double* dA = nullptr;
    CUDA_CHECK(cudaMalloc(&dL, std::max<size_t>(1, lay.localRows * n) * sizeof(double)));

    if (rank == 0) printf("Generating positive definite matrix...\n");
    generateMatrix(dL, lay, validate);

    if (validate) {
        // Keep the original matrix for validation
        CUDA_CHECK(cudaMalloc(&dA, std::max<size_t>(1, lay.localRows * n) * sizeof(double)));
        if (lay.localRows * n != 0) {
            CUDA_CHECK(cudaMemcpy(dA, dL, lay.localRows * n * sizeof(double), cudaMemcpyDeviceToDevice));
        }
    }

    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    size_t failIndex = 0;
    const bool success = choleskyDistributed(dL, lay, failIndex);

    auto end = std::chrono::high_resolution_clock::now();
    const double localSeconds = std::chrono::duration<double>(end - start).count();
    double maxSeconds = 0.0;
    MPI_Allreduce(&localSeconds, &maxSeconds, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::duration<double>(maxSeconds));

    if (!success) {
        if (rank == 0) {
            printf("Error: Matrix is not positive definite at diagonal element %zu\n", failIndex);
            printf("Cholesky decomposition failed\n");
        }
        rc = 1;
    }

    if (rc < 0) {
        if (rank == 0) printf("Computation time: %ld ms\n", duration.count());

        // Calculate GFLOPS (approximately n^3/3 operations for Cholesky)
        const double ops = (double)n * n * n / 3.0;
        const double gflops = ops / (duration.count() / 1000.0) / 1e9;
        if (rank == 0) printf("Performance: %.3f GFLOPS\n", gflops);

        std::vector<double> Lfull;
        if (printResults || validate) gatherGlobal(dL, lay, Lfull);

        if (printResults && rank == 0) {
            print_results(Lfull, "CholeskyL");
        }

        rc = 0;
        if (validate) {
            if (rank == 0) printf("Validating result...\n");
            const bool valid = validateDistributed(dL, dA, lay, Lfull);
            if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            rc = valid ? 0 : 1;
        }
    }

    if (dA) CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dL));
    MPI_Comm_free(&localComm);
    MPI_Finalize();
    return rc;
}
