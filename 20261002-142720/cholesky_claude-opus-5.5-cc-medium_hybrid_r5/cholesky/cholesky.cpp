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
#include <unistd.h>

#include "../common/results_output.hpp"

// Hybrid MPI + OpenMP + CUDA Cholesky decomposition (blocked, right-looking).
// Decomposes positive definite matrix A into L * L^T where L is lower triangular.
//
// Distribution: the (padded) matrix is split into block rows of NB rows that are
// assigned round-robin to MPI ranks (1D block-cyclic). Each rank drives one GPU.
// Step k: the owner of block row k factors (and inverts) the small diagonal block on
// the CPU, broadcasts L_kk^{-1}; every rank solves its part of panel column k on the
// GPU, the panel is all-gathered and every rank applies the symmetric trailing update
// to its rows on the GPU. A depth-1 lookahead overlaps panel factorization and
// communication with the trailing update on a second, high-priority CUDA stream.
// OpenMP parallelizes the host-side matrix generation and validation.

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        cudaError_t err_ = (call);                                                    \
        if (err_ != cudaSuccess) {                                                    \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_),     \
                    __FILE__, __LINE__);                                              \
            MPI_Abort(MPI_COMM_WORLD, 1);                                             \
        }                                                                             \
    } while (0)

static constexpr int NB = 128;                 // block size (multiple of GEMM tile)

// ---------------------------------------------------------------------------
// GPU kernels
// ---------------------------------------------------------------------------

// C[M x N] = beta * C + alpha * A[M x K] * B[N x K]^T  (all row-major)
// M, N multiples of T, K multiple of TK. Each thread computes a 4x4 sub-tile.
// If LOWER, tiles lying strictly above the global diagonal are skipped. C's local
// row r belongs to local block row (lb0 + r / NB) whose global block index is
// (lb0 + r / NB) * nprocs + rank; C's column c is global column colBase + c.
static constexpr int TK = 16;

template <int T, bool LOWER>
__global__ void __launch_bounds__((T / 4) * (T / 4)) gemm_nt_kernel(int K, const double* __restrict__ A, size_t lda,
                                                                    const double* __restrict__ B, size_t ldb,
                                                                    double* __restrict__ C, size_t ldc, double alpha,
                                                                    double beta, int lb0, int nprocs, int rank,
                                                                    int colBase) {
    constexpr int TD = T / 4;              // threads per tile dimension
    constexpr int THREADS = TD * TD;
    constexpr int LD = T * TK / THREADS;   // elements loaded per thread per operand
    const int m0 = blockIdx.y * T;
    const int n0 = blockIdx.x * T;
    if (LOWER) {
        const int lb = lb0 + m0 / NB;
        const long gRowMax = (long)(lb * nprocs + rank) * NB + (m0 % NB) + T - 1;
        if ((long)colBase + n0 > gRowMax) return;
    }

    __shared__ double As[2][TK][T + 1];
    __shared__ double Bs[2][TK][T + 1];

    const int tid = threadIdx.x;
    const int tx = tid % TD, ty = tid / TD;

    const double* Ab = A + (size_t)m0 * lda;
    const double* Bb = B + (size_t)n0 * ldb;

    double ra[LD], rb[LD];
    auto loadGlobal = [&](int k0) {
#pragma unroll
        for (int l = 0; l < LD; ++l) {
            const int idx = tid + THREADS * l;
            const int r = idx / TK, kk = idx % TK;
            ra[l] = Ab[(size_t)r * lda + k0 + kk];
            rb[l] = Bb[(size_t)r * ldb + k0 + kk];
        }
    };
    auto storeShared = [&](int buf) {
#pragma unroll
        for (int l = 0; l < LD; ++l) {
            const int idx = tid + THREADS * l;
            const int r = idx / TK, kk = idx % TK;
            As[buf][kk][r] = ra[l];
            Bs[buf][kk][r] = rb[l];
        }
    };

    double acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = 0.0;

    loadGlobal(0);
    storeShared(0);
    __syncthreads();

    int buf = 0;
    for (int k0 = 0; k0 < K; k0 += TK) {
        const bool more = (k0 + TK) < K;
        if (more) loadGlobal(k0 + TK);
#pragma unroll
        for (int kk = 0; kk < TK; ++kk) {
            double a[4], b[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) a[i] = As[buf][kk][ty + TD * i];
#pragma unroll
            for (int j = 0; j < 4; ++j) b[j] = Bs[buf][kk][tx + TD * j];
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] = fma(a[i], b[j], acc[i][j]);
        }
        if (more) {
            storeShared(buf ^ 1);
            __syncthreads();
            buf ^= 1;
        }
    }

#pragma unroll
    for (int i = 0; i < 4; ++i) {
        double* crow = C + (size_t)(m0 + ty + TD * i) * ldc + n0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int c = tx + TD * j;
            if (beta == 0.0)
                crow[c] = alpha * acc[i][j];
            else
                crow[c] = beta * crow[c] + alpha * acc[i][j];
        }
    }
}

static int g_numSMs = 1;

template <int T>
static void launchGemm(bool lower, int M, int N, int K, const double* A, size_t lda, const double* B, size_t ldb,
                       double* C, size_t ldc, double alpha, double beta, int lb0, int nprocs, int rank, int colBase,
                       cudaStream_t s) {
    dim3 grid(N / T, M / T);
    constexpr int threads = (T / 4) * (T / 4);
    if (lower)
        gemm_nt_kernel<T, true><<<grid, threads, 0, s>>>(K, A, lda, B, ldb, C, ldc, alpha, beta, lb0, nprocs, rank,
                                                         colBase);
    else
        gemm_nt_kernel<T, false><<<grid, threads, 0, s>>>(K, A, lda, B, ldb, C, ldc, alpha, beta, lb0, nprocs, rank,
                                                          colBase);
    CUDA_CHECK(cudaGetLastError());
}

static void gemm_nt(bool lower, int M, int N, int K, const double* A, size_t lda, const double* B,
                    size_t ldb, double* C, size_t ldc, double alpha, double beta, int lb0, int nprocs,
                    int rank, int colBase, cudaStream_t s) {
    if (M <= 0 || N <= 0) return;
    // FP64 work per tile is large; use smaller tiles when there are too few to fill the GPU
    long tiles = (long)(M / 64) * (N / 64);
    if (lower) tiles = (tiles + std::min(M, N) / 64) / 2;
    if (tiles < 2L * g_numSMs)
        launchGemm<32>(lower, M, N, K, A, lda, B, ldb, C, ldc, alpha, beta, lb0, nprocs, rank, colBase, s);
    else
        launchGemm<64>(lower, M, N, K, A, lda, B, ldb, C, ldc, alpha, beta, lb0, nprocs, rank, colBase, s);
}

// ---------------------------------------------------------------------------
// Diagonal block factorization on the host (latency-critical, small)
// ---------------------------------------------------------------------------

// Factor the NB x NB block a (row-major, ld NB) in place: lower part = L, upper part = 0.
// On success writes linv = L^{-1} (zero upper part) and returns -1; otherwise returns the
// global index (gBase + j) of the first non-positive diagonal element.
static long potrfInvHost(double* __restrict__ a, double* __restrict__ linv, long gBase) {
    double col[NB];
    for (int j = 0; j < NB; ++j) {
        const double val = a[j * NB + j];
        if (val <= 0.0) return gBase + j;
        const double d = sqrt(val);
        a[j * NB + j] = d;
        for (int i = j + 1; i < NB; ++i) {
            a[i * NB + j] /= d;
            col[i] = a[i * NB + j];
        }
        for (int i = j + 1; i < NB; ++i) {
            const double lij = col[i];
            double* __restrict__ ri = a + i * NB;
            for (int c = j + 1; c <= i; ++c) ri[c] -= lij * col[c];
        }
    }
    for (int i = 0; i < NB; ++i)
        for (int c = i + 1; c < NB; ++c) a[i * NB + c] = 0.0;

    // Row i of X = L^{-1}: X[i][:] = (e_i - sum_{m<i} L[i][m] X[m][:]) / L[i][i]
    for (int i = 0; i < NB; ++i) {
        double* __restrict__ x = linv + i * NB;
        std::fill(x, x + NB, 0.0);
        x[i] = 1.0;
        for (int m = 0; m < i; ++m) {
            const double lim = a[i * NB + m];
            const double* __restrict__ xm = linv + m * NB;
            for (int c = 0; c <= m; ++c) x[c] -= lim * xm[c];
        }
        const double d = a[i * NB + i];
        for (int c = 0; c <= i; ++c) x[c] /= d;
    }
    return -1;
}

// ---------------------------------------------------------------------------
// Distributed matrix layout helpers
// ---------------------------------------------------------------------------

struct Dist {
    size_t n = 0;    // matrix size
    int nblk = 0;    // number of block rows (padded)
    size_t np = 0;   // padded size = nblk * NB (also leading dimension)
    int rank = 0, nprocs = 1;
    int nloc = 0;    // local block rows on this rank

    int nlocOf(int r) const { return nblk > r ? (nblk - r + nprocs - 1) / nprocs : 0; }
    // First local block index of rank r whose global block index >= g
    int firstOf(int r, int g) const {
        const int f = g <= r ? 0 : (g - r + nprocs - 1) / nprocs;
        return std::min(f, nlocOf(r));
    }
    int first(int g) const { return firstOf(rank, g); }
    long globalRow(int lb, int r) const { return (long)(lb * nprocs + rank) * NB + r; }
};

// C[i][j] = sum_{k} X[ri*ldx + k] * X[j*ldx + k] for j <= ri (ri = rows[i]), k ascending.
// Results written to out[i*ldo + j]. If triK, X[j][k] == 0 for k > j is exploited.
// Summation order per element matches the straightforward sequential dot product.
static void syrkLowerRows(const double* X, size_t ldx, size_t klen, const std::vector<long>& rows,
                          double* out, size_t ldo, bool triK) {
    constexpr size_t BI = 32, BJ = 64, BK = 256;
    const size_t nr = rows.size();
    if (nr == 0) return;
    const size_t nti = (nr + BI - 1) / BI;
    // Pairs (i-tile, j-tile) with j-tile start <= max row in i-tile
    std::vector<std::pair<size_t, size_t>> tasks;
    for (size_t ti = 0; ti < nti; ++ti) {
        long maxRow = 0;
        for (size_t i = ti * BI; i < std::min(nr, (ti + 1) * BI); ++i) maxRow = std::max(maxRow, rows[i]);
        for (size_t ja = 0; ja <= (size_t)maxRow; ja += BJ) tasks.push_back({ti, ja});
    }
#pragma omp parallel
    {
        std::vector<double> acc(BI * BJ);
#pragma omp for schedule(dynamic, 1)
        for (size_t t = 0; t < tasks.size(); ++t) {
            const size_t i0 = tasks[t].first * BI, i1 = std::min(nr, i0 + BI);
            const size_t ja = tasks[t].second, jb = std::min<size_t>(ja + BJ, klen);
            std::fill(acc.begin(), acc.end(), 0.0);
            const size_t klim = triK ? std::min(klen, jb) : klen;
            for (size_t k0 = 0; k0 < klim; k0 += BK) {
                const size_t k1 = std::min(klim, k0 + BK);
                for (size_t i = i0; i < i1; ++i) {
                    const size_t ri = rows[i];
                    const double* xi = X + ri * ldx;
                    double* ai = acc.data() + (i - i0) * BJ;
                    const size_t jend = std::min(jb, ri + 1);
                    size_t j = ja;
                    for (; j + 4 <= jend; j += 4) {
                        const double* x0 = X + j * ldx;
                        const double* x1 = x0 + ldx;
                        const double* x2 = x1 + ldx;
                        const double* x3 = x2 + ldx;
                        double s0 = ai[j - ja], s1 = ai[j - ja + 1], s2 = ai[j - ja + 2], s3 = ai[j - ja + 3];
                        for (size_t k = k0; k < k1; ++k) {
                            const double v = xi[k];
                            s0 += v * x0[k];
                            s1 += v * x1[k];
                            s2 += v * x2[k];
                            s3 += v * x3[k];
                        }
                        ai[j - ja] = s0; ai[j - ja + 1] = s1; ai[j - ja + 2] = s2; ai[j - ja + 3] = s3;
                    }
                    for (; j < jend; ++j) {
                        const double* xj = X + j * ldx;
                        double s = ai[j - ja];
                        for (size_t k = k0; k < k1; ++k) s += xi[k] * xj[k];
                        ai[j - ja] = s;
                    }
                }
            }
            for (size_t i = i0; i < i1; ++i) {
                const size_t ri = rows[i];
                const size_t jend = std::min(jb, ri + 1);
                for (size_t j = ja; j < jend; ++j) out[i * ldo + j] = acc[(i - i0) * BJ + j - ja];
            }
        }
    }
}

// Generate this rank's block rows of the symmetric positive definite matrix
// (lower triangle; padding rows get an identity diagonal).
// Method: A = B * B^T where B is random, plus n on the diagonal.
void generatePositiveDefiniteMatrixLocal(double* hA, const Dist& d) {
    const size_t n = d.n;
    std::memset(hA, 0, (size_t)d.nloc * NB * d.np * sizeof(double));

    std::vector<double> B(n * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

    std::vector<long> rows;
    for (int lb = 0; lb < d.nloc; ++lb)
        for (int r = 0; r < NB; ++r) {
            const long g = d.globalRow(lb, r);
            if (g < (long)n) rows.push_back(g);
        }
    // Local real rows are a prefix of local storage rows (only the last global block is padded)
    syrkLowerRows(B.data(), n, n, rows, hA, d.np, false);

#pragma omp parallel for schedule(static)
    for (int lr = 0; lr < d.nloc * NB; ++lr) {
        const long g = d.globalRow(lr / NB, lr % NB);
        if (g < (long)n)
            hA[(size_t)lr * d.np + g] += n;
        else
            hA[(size_t)lr * d.np + g] = 1.0;
    }
}

// Gather distributed block rows into rank 0's dense n x n matrix (first n columns).
static void gatherToRoot(const double* hA, const Dist& d, std::vector<double>& full) {
    const size_t rowLen = d.np;
    std::vector<double> tmp(d.rank == 0 ? (size_t)NB * rowLen : 0);
    for (int I = 0; I < d.nblk; ++I) {
        const int owner = I % d.nprocs;
        const int lb = I / d.nprocs;
        if (d.rank == 0) {
            const double* src;
            if (owner == 0) {
                src = hA + (size_t)lb * NB * rowLen;
            } else {
                MPI_Recv(tmp.data(), (int)(NB * rowLen), MPI_DOUBLE, owner, I, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
                src = tmp.data();
            }
#pragma omp parallel for schedule(static)
            for (int r = 0; r < NB; ++r) {
                const size_t g = (size_t)I * NB + r;
                if (g < d.n) std::memcpy(&full[g * d.n], src + (size_t)r * rowLen, d.n * sizeof(double));
            }
        } else if (owner == d.rank) {
            MPI_Send(hA + (size_t)lb * NB * rowLen, (int)(NB * rowLen), MPI_DOUBLE, 0, I, MPI_COMM_WORLD);
        }
    }
}

// ---------------------------------------------------------------------------
// Distributed blocked Cholesky
// ---------------------------------------------------------------------------

// Device/pinned buffers and streams, allocated once before the timed region.
struct Workspace {
    cudaStream_t s1 = nullptr, s2 = nullptr, s3 = nullptr;  // trailing update, panel, result download
    cudaEvent_t evU = nullptr, evPanel = nullptr;
    double* dA = nullptr;                  // local block rows (nloc*NB x np)
    double* dP[2] = {nullptr, nullptr};    // local panel rows (double buffered)
    double* dG[2] = {nullptr, nullptr};    // gathered panel in global order (P > 1)
    double* dLinv = nullptr;
    double* hLinv[2] = {nullptr, nullptr};
    double* hDiag = nullptr;
    double* hPloc = nullptr;
    double* hRecv = nullptr;

    explicit Workspace(const Dist& d) {
        const size_t locRows = (size_t)d.nloc * NB;
        const size_t panelLoc = std::max<size_t>(locRows, 1) * NB;
        const size_t panelGlob = (size_t)d.nblk * NB * NB;
        // Panel stream (critical path) gets high priority over the trailing-update stream
        int prLeast, prGreatest;
        CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&prLeast, &prGreatest));
        CUDA_CHECK(cudaStreamCreateWithPriority(&s1, cudaStreamNonBlocking, prLeast));
        CUDA_CHECK(cudaStreamCreateWithPriority(&s2, cudaStreamNonBlocking, prGreatest));
        CUDA_CHECK(cudaStreamCreateWithPriority(&s3, cudaStreamNonBlocking, prLeast));
        CUDA_CHECK(cudaEventCreateWithFlags(&evU, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&evPanel, cudaEventDisableTiming));
        CUDA_CHECK(cudaMalloc(&dA, std::max<size_t>(locRows, 1) * d.np * sizeof(double)));
        CUDA_CHECK(cudaMemset(dA, 0, std::max<size_t>(locRows, 1) * d.np * sizeof(double)));
        for (int b = 0; b < 2; ++b) {
            CUDA_CHECK(cudaMalloc(&dP[b], panelLoc * sizeof(double)));
            if (d.nprocs > 1) CUDA_CHECK(cudaMalloc(&dG[b], panelGlob * sizeof(double)));
            CUDA_CHECK(cudaMallocHost(&hLinv[b], (NB * NB + 1) * sizeof(double)));
        }
        CUDA_CHECK(cudaMalloc(&dLinv, NB * NB * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&hDiag, NB * NB * sizeof(double)));
        if (d.nprocs > 1) {
            CUDA_CHECK(cudaMallocHost(&hPloc, panelLoc * sizeof(double)));
            CUDA_CHECK(cudaMallocHost(&hRecv, panelGlob * sizeof(double)));
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    ~Workspace() {
        for (int b = 0; b < 2; ++b) {
            cudaFree(dP[b]);
            if (dG[b]) cudaFree(dG[b]);
            cudaFreeHost(hLinv[b]);
        }
        if (hPloc) cudaFreeHost(hPloc);
        if (hRecv) cudaFreeHost(hRecv);
        cudaFree(dA);
        cudaFree(dLinv);
        cudaFreeHost(hDiag);
        cudaEventDestroy(evU);
        cudaEventDestroy(evPanel);
        cudaStreamDestroy(s1);
        cudaStreamDestroy(s2);
        cudaStreamDestroy(s3);
    }
};

// Factors the local block rows hA (lower triangle, upper triangle zero) in place.
// Returns -1 on success, otherwise the global index of the failing diagonal element.
long choleskyDecomposition(double* hA, const Dist& d, Workspace& w) {
    const int P = d.nprocs, rank = d.rank, nblk = d.nblk;
    const size_t ld = d.np;
    cudaStream_t s1 = w.s1, s2 = w.s2, s3 = w.s3;
    double* dA = w.dA;
    std::vector<int> counts(P), displs(P);

    // Upload the lower triangle of the local block rows
    for (int lb = 0; lb < d.nloc; ++lb) {
        const size_t width = (size_t)(lb * P + rank + 1) * NB;
        CUDA_CHECK(cudaMemcpy2DAsync(dA + (size_t)lb * NB * ld, ld * sizeof(double), hA + (size_t)lb * NB * ld,
                                     ld * sizeof(double), width * sizeof(double), NB, cudaMemcpyHostToDevice, s2));
    }

    const double* G[2] = {nullptr, nullptr};

    // Factor diagonal block j, solve local panel rows i > j, gather panel into G[b].
    auto computePanel = [&](int j, int b) -> long {
        const int owner = j % P;
        double* hl = w.hLinv[b];
        if (rank == owner) {
            double* Djj = dA + (size_t)(j / P) * NB * ld + (size_t)j * NB;
            CUDA_CHECK(cudaMemcpy2DAsync(w.hDiag, NB * sizeof(double), Djj, ld * sizeof(double),
                                         NB * sizeof(double), NB, cudaMemcpyDeviceToHost, s2));
            CUDA_CHECK(cudaStreamSynchronize(s2));
            const long info = potrfInvHost(w.hDiag, hl, (long)j * NB);
            hl[NB * NB] = (double)info;
            if (info < 0) {
                CUDA_CHECK(cudaMemcpy2DAsync(Djj, ld * sizeof(double), w.hDiag, NB * sizeof(double),
                                             NB * sizeof(double), NB, cudaMemcpyHostToDevice, s2));
                CUDA_CHECK(cudaMemcpyAsync(w.dLinv, hl, NB * NB * sizeof(double), cudaMemcpyHostToDevice, s2));
            }
        }
        if (P > 1) MPI_Bcast(hl, NB * NB + 1, MPI_DOUBLE, owner, MPI_COMM_WORLD);
        const long info = (long)hl[NB * NB];
        if (info >= 0) return info;
        if (rank != owner)
            CUDA_CHECK(cudaMemcpyAsync(w.dLinv, hl, NB * NB * sizeof(double), cudaMemcpyHostToDevice, s2));

        // L_ij = A_ij * L_jj^{-T} for local block rows i > j
        const int f = d.first(j + 1);
        const int m = (d.nloc - f) * NB;
        if (m > 0) {
            double* Acol = dA + (size_t)f * NB * ld + (size_t)j * NB;
            gemm_nt(false, m, NB, NB, Acol, ld, w.dLinv, NB, w.dP[b], NB, 1.0, 0.0, 0, P, rank, 0, s2);
            CUDA_CHECK(cudaMemcpy2DAsync(Acol, ld * sizeof(double), w.dP[b], NB * sizeof(double),
                                         NB * sizeof(double), m, cudaMemcpyDeviceToDevice, s2));
        }
        if (j + 1 < nblk) {
            if (P == 1) {
                G[b] = w.dP[b];
            } else {
                if (m > 0)
                    CUDA_CHECK(cudaMemcpyAsync(w.hPloc, w.dP[b], (size_t)m * NB * sizeof(double),
                                               cudaMemcpyDeviceToHost, s2));
                CUDA_CHECK(cudaStreamSynchronize(s2));
                int off = 0;
                for (int r = 0; r < P; ++r) {
                    counts[r] = (d.nlocOf(r) - d.firstOf(r, j + 1)) * NB * NB;
                    displs[r] = off;
                    off += counts[r];
                }
                MPI_Allgatherv(w.hPloc, counts[rank], MPI_DOUBLE, w.hRecv, counts.data(), displs.data(),
                               MPI_DOUBLE, MPI_COMM_WORLD);
                // Place block rows in global order: block g lands at (g - (j+1)) * NB*NB
                for (int r = 0; r < P; ++r) {
                    const int fr = d.firstOf(r, j + 1);
                    for (int lb = fr; lb < d.nlocOf(r); ++lb) {
                        const int g = lb * P + r;
                        CUDA_CHECK(cudaMemcpyAsync(w.dG[b] + (size_t)(g - j - 1) * NB * NB,
                                                   w.hRecv + displs[r] + (size_t)(lb - fr) * NB * NB,
                                                   (size_t)NB * NB * sizeof(double), cudaMemcpyHostToDevice,
                                                   s2));
                    }
                }
                G[b] = w.dG[b];
            }
        }
        CUDA_CHECK(cudaEventRecord(w.evPanel, s2));
        // Block row j is now final: download its lower part in the background
        if (rank == owner) {
            const size_t lb = j / P;
            CUDA_CHECK(cudaStreamWaitEvent(s3, w.evPanel, 0));
            CUDA_CHECK(cudaMemcpy2DAsync(hA + lb * NB * ld, ld * sizeof(double), dA + lb * NB * ld,
                                         ld * sizeof(double), (size_t)(j + 1) * NB * sizeof(double), NB,
                                         cudaMemcpyDeviceToHost, s3));
        }
        return -1;
    };

    long fail = computePanel(0, 0);
    for (int k = 0; k < nblk && fail < 0; ++k) {
        const int b = k & 1;
        const int f1 = d.first(k + 1);
        if (k + 1 < nblk) {
            // Lookahead: update block column k+1 of local rows i >= k+1
            if (k > 0) CUDA_CHECK(cudaStreamWaitEvent(s2, w.evU, 0));
            const int m = (d.nloc - f1) * NB;
            gemm_nt(false, m, NB, NB, w.dP[b], NB, G[b], NB, dA + (size_t)f1 * NB * ld + (size_t)(k + 1) * NB, ld,
                    -1.0, 1.0, 0, P, rank, 0, s2);
        }
        if (k + 2 < nblk) {
            // Trailing update of columns >= k+2 for local rows i >= k+2 (lower tiles only)
            CUDA_CHECK(cudaStreamWaitEvent(s1, w.evPanel, 0));
            const int f2 = d.first(k + 2);
            const int m = (d.nloc - f2) * NB;
            const int colBase = (k + 2) * NB;
            gemm_nt(true, m, (int)(ld - colBase), NB, w.dP[b] + (size_t)(f2 - f1) * NB * NB, NB,
                    G[b] + (size_t)NB * NB, NB, dA + (size_t)f2 * NB * ld + colBase, ld, -1.0, 1.0, f2, P, rank,
                    colBase, s1);
            CUDA_CHECK(cudaEventRecord(w.evU, s1));
        }
        if (k + 1 < nblk) fail = computePanel(k + 1, b ^ 1);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    return fail;
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix

    std::vector<double> reconstructed(n * n);

    // Compute L * L^T (lower part; L is lower triangular and L * L^T is symmetric)
    std::vector<long> rows(n);
    for (size_t i = 0; i < n; ++i) rows[i] = (long)i;
    syrkLowerRows(L.data(), n, n, rows, reconstructed.data(), n, true);

    // Compare with original
    double maxError = 0.0;
    double relError = 0.0;

#pragma omp parallel for schedule(dynamic, 16) reduction(max : maxError, relError)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            const double rec = j <= i ? reconstructed[i * n + j] : reconstructed[j * n + i];
            const double error = fabs(rec - A_orig[i * n + j]);
            maxError = std::max(maxError, error);

            const double rel = error / (fabs(A_orig[i * n + j]) + 1e-10);
            relError = std::max(relError, rel);
        }
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
    MPI_Init(&argc, &argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

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

    // Node-local rank -> GPU and OpenMP thread share
    MPI_Comm local;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
    int lrank, lsize;
    MPI_Comm_rank(local, &lrank);
    MPI_Comm_size(local, &lsize);
    MPI_Comm_free(&local);
    int ndev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    if (ndev == 0) {
        fprintf(stderr, "No CUDA device found\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(lrank % ndev));
    CUDA_CHECK(cudaFree(0));
    CUDA_CHECK(cudaDeviceGetAttribute(&g_numSMs, cudaDevAttrMultiProcessorCount, lrank % ndev));
    if (!getenv("OMP_NUM_THREADS")) {
        const int procs = omp_get_num_procs();
        const long hw = sysconf(_SC_NPROCESSORS_ONLN);
        // Unbound ranks sharing a node split the cores between them
        omp_set_num_threads(std::max(1, procs >= hw ? procs / lsize : procs));
    }

    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    Dist d;
    d.n = n;
    d.rank = rank;
    d.nprocs = nprocs;
    d.nblk = (int)((n + NB - 1) / NB);
    d.np = (size_t)d.nblk * NB;
    d.nloc = d.nlocOf(rank);

    // Allocate local block rows (pinned for fast transfers)
    const size_t locElems = (size_t)d.nloc * NB * d.np;
    double* hA = nullptr;
    CUDA_CHECK(cudaMallocHost(&hA, std::max<size_t>(locElems, 1) * sizeof(double)));
    std::vector<double> A;
    std::vector<double> A_orig;

    // Generate positive definite matrix
    if (rank == 0) printf("Generating positive definite matrix...\n");
    generatePositiveDefiniteMatrixLocal(hA, d);

    if (validate) {
        // Save original for validation (full symmetric matrix on rank 0)
        if (rank == 0) A_orig.resize(n * n);
        gatherToRoot(hA, d, A_orig);
        if (rank == 0) {
#pragma omp parallel for schedule(dynamic, 16)
            for (size_t i = 0; i < n; ++i)
                for (size_t j = i + 1; j < n; ++j) A_orig[i * n + j] = A_orig[j * n + i];
        }
    }

    // Device buffers and streams
    Workspace* ws = n > 0 ? new Workspace(d) : nullptr;

    // Perform Cholesky decomposition
    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    const long fail = n > 0 ? choleskyDecomposition(hA, d, *ws) : -1;

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    delete ws;

    if (fail >= 0) {
        if (rank == 0) {
            printf("Error: Matrix is not positive definite at diagonal element %zu\n", (size_t)fail);
            printf("Cholesky decomposition failed\n");
        }
        cudaFreeHost(hA);
        MPI_Finalize();
        return 1;
    }

    // Collect L on rank 0 when it is needed for output
    if (printResults || validate) {
        if (rank == 0) A.resize(n * n);
        gatherToRoot(hA, d, A);
    }
    cudaFreeHost(hA);

    int ret = 0;
    if (rank == 0) {
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
            } else {
                printf("Validation: FAILED\n");
                ret = 1;
            }
        }
    }

    MPI_Bcast(&ret, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return ret;
}
