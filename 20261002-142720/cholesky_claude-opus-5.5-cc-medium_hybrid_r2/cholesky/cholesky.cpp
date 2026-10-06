#include <algorithm>
#include <chrono>
#include <climits>
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

// Hybrid MPI + OpenMP + CUDA Cholesky decomposition.
// Decomposes positive definite matrix A into L * L^T where L is lower triangular.
//
// Parallelization scheme:
//  - The (padded) matrix is split into block columns of width nb, distributed
//    block-cyclically over MPI ranks (one GPU per rank).
//  - Right-looking blocked algorithm: at step k the owner of block column k
//    factors the panel on its GPU and broadcasts it (via pinned host memory);
//    every rank then applies the rank-nb update to its own block columns.
//  - Look-ahead: the owner of column k+1 updates and factors it on a
//    high-priority stream while the bulk trailing update runs concurrently,
//    so panel factorization and communication are hidden behind compute.
//  - Rank 0 assembles the final L from the broadcast panels (no extra gather).
//  - OpenMP parallelizes the host work (matrix generation, assembly, validation).

#define CUDA_CHECK(call)                                                                  \
    do {                                                                                  \
        cudaError_t err_ = (call);                                                        \
        if (err_ != cudaSuccess) {                                                        \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__, \
                    __LINE__);                                                            \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                 \
        }                                                                                 \
    } while (0)

// ---------------------------------------------------------------------------
// CUDA kernels
// ---------------------------------------------------------------------------

constexpr int IB = 32;        // inner block size of panel factorization
constexpr int GM = 64;        // GEMM tile rows
constexpr int GN = 64;        // GEMM tile cols
constexpr int GK = 16;        // GEMM tile depth
constexpr int TRSM_ROWS = 64;

// C[m x n] -= A[m x k] * B[n x k]^T for one 64x64 tile (row-major, k % GK == 0)
__device__ __forceinline__ void gemmNTSubTile(int m, int n, int k, const double* __restrict__ A,
                                              int lda, const double* __restrict__ B, int ldb,
                                              double* __restrict__ C, int ldc, int row0, int col0) {
    __shared__ double As[GK][GM + 1];
    __shared__ double Bs[GK][GN + 1];

    const int t = threadIdx.x;
    const int tx = t & 15;
    const int ty = t >> 4;

    double acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = 0.0;

    for (int k0 = 0; k0 < k; k0 += GK) {
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int idx = t + l * 256;
            const int r = idx >> 4;
            const int kk = idx & 15;
            const int ar = row0 + r;
            const int br = col0 + r;
            As[kk][r] = (ar < m) ? A[(size_t)ar * lda + k0 + kk] : 0.0;
            Bs[kk][r] = (br < n) ? B[(size_t)br * ldb + k0 + kk] : 0.0;
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < GK; ++kk) {
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
        const int r = row0 + ty + 16 * i;
        if (r >= m) continue;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int c = col0 + tx + 16 * j;
            if (c < n) C[(size_t)r * ldc + c] -= acc[i][j];
        }
    }
}

__global__ void __launch_bounds__(256) gemmNTSub(int m, int n, int k, const double* A, int lda,
                                                 const double* B, int ldb, double* C, int ldc) {
    gemmNTSubTile(m, n, k, A, lda, B, ldb, C, ldc, blockIdx.y * GM, blockIdx.x * GN);
}

// Batched trailing update: for local block columns jl = jl0 + blockIdx.z,
// colbuf_j -= panel_rows(j) * panel_rows(j)[0:nb]^T
__global__ void __launch_bounds__(256)
    trailingUpdate(double* cols, const size_t* offs, int jl0, int nprocs, int rank, int nb,
                   int N, const double* panel, int g) {
    const int jl = jl0 + blockIdx.z;
    const int j = jl * nprocs + rank;
    const int rowStart = j * nb;
    const int Mj = N - rowStart;
    const int row0 = blockIdx.y * GM;
    const int col0 = blockIdx.x * GN;
    if (row0 >= Mj) return;
    if (row0 + GM <= col0) return;  // tile strictly in the (unused) upper triangle
    const double* A = panel + (size_t)(rowStart - g) * nb;
    gemmNTSubTile(Mj, nb, nb, A, nb, A, nb, cols + offs[jl], nb, row0, col0);
}

// Unblocked Cholesky of a 32x32 diagonal tile: one warp, lane i holds row i
__global__ void potrfTile(double* T, int ld, int* info, int gidx) {
    const int i = threadIdx.x;
    double a[IB];
#pragma unroll
    for (int q = 0; q < IB; ++q) a[q] = (q <= i) ? T[(size_t)i * ld + q] : 0.0;
#pragma unroll
    for (int j = 0; j < IB; ++j) {
        const double val = __shfl_sync(0xffffffffu, a[j], j);
        if (i == 0 && val <= 0.0) atomicMin(info, gidx + j);
        const double ljj = sqrt(val);
        if (i == j) a[j] = ljj;
        else if (i > j) a[j] = a[j] / ljj;
#pragma unroll
        for (int q = j + 1; q < IB; ++q) {
            const double lqj = __shfl_sync(0xffffffffu, a[j], q);
            if (q <= i) a[q] -= a[j] * lqj;
        }
    }
#pragma unroll
    for (int q = 0; q < IB; ++q)
        if (q <= i) T[(size_t)i * ld + q] = a[q];
}

// Solve X * L^T = Bm for m rows of width 32 (L is a 32x32 lower triangular tile)
__global__ void __launch_bounds__(TRSM_ROWS) trsmRows(const double* L, double* Bm, int m, int ld) {
    __shared__ double sL[IB][IB + 1];
    __shared__ double sB[TRSM_ROWS][IB + 1];
    const int t = threadIdx.x;
    for (int idx = t; idx < IB * IB; idx += TRSM_ROWS) {
        const int r = idx / IB, c = idx % IB;
        sL[r][c] = (c <= r) ? L[(size_t)r * ld + c] : 0.0;
    }
    const int row0 = blockIdx.x * TRSM_ROWS;
    for (int idx = t; idx < TRSM_ROWS * IB; idx += TRSM_ROWS) {
        const int r = idx / IB, c = idx % IB;
        if (row0 + r < m) sB[r][c] = Bm[(size_t)(row0 + r) * ld + c];
    }
    __syncthreads();
    if (row0 + t < m) {
        double x[IB];
#pragma unroll
        for (int j = 0; j < IB; ++j) x[j] = sB[t][j];
#pragma unroll
        for (int j = 0; j < IB; ++j) {
            double sum = 0.0;
#pragma unroll
            for (int p = 0; p < j; ++p) sum += x[p] * sL[j][p];
            x[j] = (x[j] - sum) / sL[j][j];
        }
#pragma unroll
        for (int j = 0; j < IB; ++j) sB[t][j] = x[j];
    }
    __syncthreads();
    for (int idx = t; idx < TRSM_ROWS * IB; idx += TRSM_ROWS) {
        const int r = idx / IB, c = idx % IB;
        if (row0 + r < m) Bm[(size_t)(row0 + r) * ld + c] = sB[r][c];
    }
}

__global__ void fillInt(int* p, int n, int v) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = v;
}

// Factor a panel of M rows x nb columns (row-major, ld = nb) whose top nb x nb
// block is the diagonal block with global index offset g.
static void factorPanel(double* Pp, int M, int nb, int* info, int g, cudaStream_t s) {
    for (int c = 0; c < nb; c += IB) {
        double* diag = Pp + (size_t)c * nb + c;
        potrfTile<<<1, IB, 0, s>>>(diag, nb, info, g + c);
        const int mr = M - c - IB;
        if (mr <= 0) continue;
        double* below = Pp + (size_t)(c + IB) * nb + c;
        trsmRows<<<(mr + TRSM_ROWS - 1) / TRSM_ROWS, TRSM_ROWS, 0, s>>>(diag, below, mr, nb);
        const int nc = nb - c - IB;
        if (nc > 0) {
            dim3 grid((nc + GN - 1) / GN, (mr + GM - 1) / GM);
            gemmNTSub<<<grid, 256, 0, s>>>(mr, nc, IB, below, nb, below, nb, below + IB, nb);
        }
    }
}

// ---------------------------------------------------------------------------
// Host helpers (OpenMP)
// ---------------------------------------------------------------------------

// out[c] = sum_k x[k] * ys[c][k] for c < cnt, each sum accumulated in sequential k order
static inline void dotMulti(const double* x, const double* const* ys, int cnt, size_t len,
                            double* out) {
    if (cnt == 8) {
        double s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0, s7 = 0;
        const double *y0 = ys[0], *y1 = ys[1], *y2 = ys[2], *y3 = ys[3];
        const double *y4 = ys[4], *y5 = ys[5], *y6 = ys[6], *y7 = ys[7];
        for (size_t k = 0; k < len; ++k) {
            const double a = x[k];
            s0 += a * y0[k];
            s1 += a * y1[k];
            s2 += a * y2[k];
            s3 += a * y3[k];
            s4 += a * y4[k];
            s5 += a * y5[k];
            s6 += a * y6[k];
            s7 += a * y7[k];
        }
        out[0] = s0; out[1] = s1; out[2] = s2; out[3] = s3;
        out[4] = s4; out[5] = s5; out[6] = s6; out[7] = s7;
    } else {
        for (int c = 0; c < cnt; ++c) {
            double sum = 0.0;
            for (size_t k = 0; k < len; ++k) sum += x[k] * ys[c][k];
            out[c] = sum;
        }
    }
}

// Generate random matrix B (same sequence as the original generator)
static void generateB(std::vector<double>& B, const size_t n) {
    unsigned int seed = 42;
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }
}

// Generate a symmetric positive definite matrix A = B * B^T + n * I
// (bit-identical to the sequential generator; uses symmetry and OpenMP)
void generatePositiveDefiniteMatrix(std::vector<double>& A, const std::vector<double>& B,
                                    const size_t n) {
#pragma omp parallel for schedule(dynamic, 4)
    for (size_t i = 0; i < n; ++i) {
        const double* bi = &B[i * n];
        for (size_t j0 = 0; j0 <= i; j0 += 8) {
            const int cnt = (int)std::min<size_t>(8, i + 1 - j0);
            const double* ys[8];
            double out[8];
            for (int c = 0; c < cnt; ++c) ys[c] = &B[(j0 + c) * n];
            dotMulti(bi, ys, cnt, n, out);
            for (int c = 0; c < cnt; ++c) {
                A[i * n + j0 + c] = out[c];
                A[(j0 + c) * n + i] = out[c];
            }
        }
    }
    // Add diagonal dominance to ensure positive definiteness
    for (size_t i = 0; i < n; ++i) {
        A[i * n + i] += n;
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix
    // (L is lower triangular, so only k <= min(i, j) contributes)
    double maxError = 0.0;
    double relError = 0.0;

#pragma omp parallel for schedule(dynamic, 4) reduction(max : maxError, relError)
    for (size_t i = 0; i < n; ++i) {
        const double* li = &L[i * n];
        for (size_t j0 = 0; j0 <= i; j0 += 8) {
            const int cnt = (int)std::min<size_t>(8, i + 1 - j0);
            const double* ys[8];
            double out[8];
            for (int c = 0; c < cnt; ++c) ys[c] = &L[(j0 + c) * n];
            dotMulti(li, ys, cnt, j0 + cnt, out);
            for (int c = 0; c < cnt; ++c) {
                const size_t j = j0 + c;
                const double e1 = fabs(out[c] - A_orig[i * n + j]);
                maxError = std::max(maxError, e1);
                relError = std::max(relError, e1 / (fabs(A_orig[i * n + j]) + 1e-10));
                const double e2 = fabs(out[c] - A_orig[j * n + i]);
                maxError = std::max(maxError, e2);
                relError = std::max(relError, e2 / (fabs(A_orig[j * n + i]) + 1e-10));
            }
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

// ---------------------------------------------------------------------------
// Distributed decomposition
// ---------------------------------------------------------------------------

struct Dist {
    int rank, nprocs;
    size_t n;       // original size
    int N;          // padded size (multiple of nb)
    int nb;         // block column width
    int K;          // number of block columns
    int nloc;       // number of local block columns
    std::vector<size_t> offs;  // element offset of each local column buffer

    // Hierarchical communication: ranks of a node share host panel buffers,
    // one leader per node takes part in inter-node broadcasts.
    MPI_Comm localComm, leaderComm;
    int localRank, localSize;
    int myNode, numNodes;
    std::vector<int> nodeOf;  // node index of every rank

    int global(int jl) const { return jl * nprocs + rank; }
    int owner(int j) const { return j % nprocs; }
    // first local index whose global column is > k
    int firstLocalAfter(int k) const {
        const int j = k + 1;
        if (j <= rank) return 0;
        return (j - rank + nprocs - 1) / nprocs;
    }
};

static void setupComm(Dist& d) {
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, d.rank, MPI_INFO_NULL, &d.localComm);
    MPI_Comm_rank(d.localComm, &d.localRank);
    MPI_Comm_size(d.localComm, &d.localSize);
    MPI_Comm_split(MPI_COMM_WORLD, d.localRank == 0 ? 0 : MPI_UNDEFINED, d.rank, &d.leaderComm);
    int node = 0, nodes = 0;
    if (d.localRank == 0) {
        MPI_Comm_rank(d.leaderComm, &node);
        MPI_Comm_size(d.leaderComm, &nodes);
    }
    MPI_Bcast(&node, 1, MPI_INT, 0, d.localComm);
    MPI_Bcast(&nodes, 1, MPI_INT, 0, d.localComm);
    d.myNode = node;
    d.numNodes = nodes;
    d.nodeOf.resize(d.nprocs);
    MPI_Allgather(&node, 1, MPI_INT, d.nodeOf.data(), 1, MPI_INT, MPI_COMM_WORLD);
}

// Copy broadcast panel k (rows g..N-1, nb columns) into the full n x n matrix
// on rank 0, zeroing the upper triangle of the rows whose diagonal is in it.
static void storePanel(std::vector<double>& A, const double* panel, const Dist& d, int k) {
    const size_t n = d.n;
    const size_t g = (size_t)k * d.nb;
    if (g >= n) return;
    const size_t c1 = std::min(g + d.nb, n);
    // Small, bandwidth-bound copy on the critical path: use only a few threads
    const int nt = (int)std::clamp<size_t>((n - g) * d.nb / (1 << 17), 1, 16);
#pragma omp parallel for schedule(static) num_threads(nt)
    for (size_t i = g; i < n; ++i) {
        const double* src = panel + (i - g) * d.nb;
        double* dst = &A[i * n];
        if (i < g + d.nb) {
            std::memcpy(dst + g, src, (i - g + 1) * sizeof(double));
            std::fill(dst + i + 1, dst + n, 0.0);
        } else {
            std::memcpy(dst + g, src, (c1 - g) * sizeof(double));
        }
    }
}

constexpr int NSLOT = 3;  // node-shared host panel slots

// Device buffers, streams, events and shared host buffers (allocated outside the timed region)
struct GpuContext {
    double* dcols = nullptr;
    size_t* doffs = nullptr;
    double* dpanel[2];
    double* hslot[NSLOT];  // node-shared, page-locked panel buffers (+1 info entry)
    double* shmBase = nullptr;
    size_t shmBytes = 0;
    MPI_Win win;
    int* dinfo = nullptr;
    int* hinfo = nullptr;
    cudaStream_t sc, su, sx;  // critical path (look-ahead), bulk update, copies
    cudaEvent_t evFact, evD2H, evPanel[2], evLA[2], evBulk[2], evFirst[2], evH2D[NSLOT];
};

static void setupGpu(GpuContext& ctx, const Dist& d) {
    const size_t totalLocal = d.offs.back();
    const size_t panelMax = (size_t)d.N * d.nb;
    const size_t slotLen = panelMax + 1;

    CUDA_CHECK(cudaMalloc(&ctx.dcols, std::max<size_t>(1, totalLocal) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&ctx.doffs, d.offs.size() * sizeof(size_t)));
    for (int b = 0; b < 2; ++b)
        CUDA_CHECK(cudaMalloc(&ctx.dpanel[b], std::max<size_t>(1, panelMax) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&ctx.dinfo, std::max(1, d.K) * sizeof(int)));
    CUDA_CHECK(cudaMallocHost(&ctx.hinfo, sizeof(int)));

    // Node-shared panel slots, registered with CUDA in every process
    const long page = sysconf(_SC_PAGESIZE);
    const size_t bytes = ((NSLOT * slotLen * sizeof(double) + page - 1) / page) * page;
    MPI_Aint winBytes = (d.localRank == 0) ? (MPI_Aint)bytes : 0;
    double* mine = nullptr;
    MPI_Win_allocate_shared(winBytes, sizeof(double), MPI_INFO_NULL, d.localComm, &mine, &ctx.win);
    MPI_Aint qsize;
    int qdisp;
    MPI_Win_shared_query(ctx.win, 0, &qsize, &qdisp, &ctx.shmBase);
    ctx.shmBytes = bytes;
    CUDA_CHECK(cudaHostRegister(ctx.shmBase, bytes, cudaHostRegisterPortable));
    for (int s = 0; s < NSLOT; ++s) ctx.hslot[s] = ctx.shmBase + s * slotLen;
    MPI_Win_lock_all(MPI_MODE_NOCHECK, ctx.win);

    int prioLow, prioHigh;
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&prioLow, &prioHigh));
    CUDA_CHECK(cudaStreamCreateWithPriority(&ctx.sc, cudaStreamNonBlocking, prioHigh));
    CUDA_CHECK(cudaStreamCreateWithPriority(&ctx.su, cudaStreamNonBlocking, prioLow));
    CUDA_CHECK(cudaStreamCreateWithPriority(&ctx.sx, cudaStreamNonBlocking, prioHigh));

    auto mk = [](cudaEvent_t* e) { CUDA_CHECK(cudaEventCreateWithFlags(e, cudaEventDisableTiming)); };
    mk(&ctx.evFact);
    mk(&ctx.evD2H);
    for (int b = 0; b < 2; ++b) {
        mk(&ctx.evPanel[b]);
        mk(&ctx.evLA[b]);
        mk(&ctx.evBulk[b]);
        mk(&ctx.evFirst[b]);
    }
    for (int s = 0; s < NSLOT; ++s) mk(&ctx.evH2D[s]);

    CUDA_CHECK(cudaMemcpy(ctx.doffs, d.offs.data(), d.offs.size() * sizeof(size_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Returns -1 on success, otherwise the index of the failing diagonal element.
static long long choleskyDistributed(GpuContext& ctx, const Dist& d, const double* hcols,
                                     std::vector<double>& A) {
    const int nb = d.nb, N = d.N, K = d.K;
    const size_t totalLocal = d.offs.back();
    double* dcols = ctx.dcols;
    const size_t* doffs = ctx.doffs;
    int* dinfo = ctx.dinfo;
    cudaStream_t sc = ctx.sc, su = ctx.su, sx = ctx.sx;

    // Upload local columns
    CUDA_CHECK(cudaMemcpyAsync(dcols, hcols, totalLocal * sizeof(double), cudaMemcpyHostToDevice, sc));
    if (K > 0) fillInt<<<(K + 255) / 256, 256, 0, sc>>>(dinfo, K, INT_MAX);
    CUDA_CHECK(cudaEventRecord(ctx.evFirst[1], sc));  // "step -1" done

    auto colPtr = [&](int j) { return dcols + d.offs[j / d.nprocs]; };

    // Factor panel j (local) on sc, then download it into the node-shared slot on sx
    auto factorAndSend = [&](int j) {
        factorPanel(colPtr(j), N - j * nb, nb, dinfo + j, j * nb, sc);
        CUDA_CHECK(cudaEventRecord(ctx.evFact, sc));
        CUDA_CHECK(cudaEventRecord(ctx.evPanel[j % 2], sc));
        CUDA_CHECK(cudaStreamWaitEvent(sx, ctx.evFact, 0));
        CUDA_CHECK(cudaMemcpyAsync(ctx.hslot[j % NSLOT], colPtr(j),
                                   (size_t)(N - j * nb) * nb * sizeof(double),
                                   cudaMemcpyDeviceToHost, sx));
        CUDA_CHECK(cudaMemcpyAsync(ctx.hinfo, dinfo + j, sizeof(int), cudaMemcpyDeviceToHost, sx));
        CUDA_CHECK(cudaEventRecord(ctx.evD2H, sx));
    };

    // Launch C -= panel * panel^T for local columns [jlA, jlB)
    auto update = [&](int jlA, int jlB, const double* panel, int g, cudaStream_t s) {
        if (jlA >= jlB) return;
        const int Mmax = N - d.global(jlA) * nb;
        dim3 grid(nb / GN, (Mmax + GM - 1) / GM, jlB - jlA);
        trailingUpdate<<<grid, 256, 0, s>>>(dcols, doffs, jlA, d.nprocs, d.rank, nb, N, panel, g);
    };

    if (K > 0 && d.owner(0) == d.rank) factorAndSend(0);

    long long failIdx = -1;
    for (int k = 0; k < K; ++k) {
        const int b = k % 2;
        const int slot = k % NSLOT;
        const int g = k * nb;
        const size_t cnt = (size_t)(N - g) * nb;
        const int root = d.owner(k);
        const bool isRoot = (root == d.rank);
        double* hpanel = ctx.hslot[slot];

        // Own upload of panel k-2 must be finished before its slot is reused
        // (the node-level allreduce below makes this hold for all local ranks).
        CUDA_CHECK(cudaEventSynchronize(ctx.evH2D[(k + 1) % NSLOT]));
        int info = INT_MAX;
        if (isRoot) {
            CUDA_CHECK(cudaEventSynchronize(ctx.evD2H));
            info = *ctx.hinfo;
        }
        MPI_Win_sync(ctx.win);
        MPI_Allreduce(MPI_IN_PLACE, &info, 1, MPI_INT, MPI_MIN, d.localComm);
        if (d.numNodes > 1) {
            if (d.localRank == 0) {
                if (d.nodeOf[root] == d.myNode) hpanel[cnt] = (double)info;
                MPI_Bcast(hpanel, (int)(cnt + 1), MPI_DOUBLE, d.nodeOf[root], d.leaderComm);
                info = (int)hpanel[cnt];
            }
            MPI_Bcast(&info, 1, MPI_INT, 0, d.localComm);
        }
        MPI_Win_sync(ctx.win);
        if (info != INT_MAX) {
            failIdx = info;
            break;
        }

        const double* panel;
        if (isRoot) {
            panel = colPtr(k);
        } else {
            CUDA_CHECK(cudaStreamWaitEvent(sx, ctx.evLA[b], 0));
            CUDA_CHECK(cudaStreamWaitEvent(sx, ctx.evBulk[b], 0));
            CUDA_CHECK(cudaMemcpyAsync(ctx.dpanel[b], hpanel, cnt * sizeof(double),
                                       cudaMemcpyHostToDevice, sx));
            CUDA_CHECK(cudaEventRecord(ctx.evH2D[slot], sx));
            CUDA_CHECK(cudaEventRecord(ctx.evPanel[b], sx));
            panel = ctx.dpanel[b];
        }

        const bool nextOwner = (k + 1 < K) && d.owner(k + 1) == d.rank;
        int jlStart = d.firstLocalAfter(k);
        if (nextOwner) ++jlStart;

        // Bulk trailing update (low priority stream); the first column is
        // launched separately since it is the look-ahead column of step k+1.
        CUDA_CHECK(cudaStreamWaitEvent(su, ctx.evPanel[b], 0));
        update(jlStart, std::min(jlStart + 1, d.nloc), panel, g, su);
        CUDA_CHECK(cudaEventRecord(ctx.evFirst[b], su));
        update(jlStart + 1, d.nloc, panel, g, su);
        CUDA_CHECK(cudaEventRecord(ctx.evBulk[b], su));

        // Look-ahead: update and factor the next panel (high priority stream)
        CUDA_CHECK(cudaStreamWaitEvent(sc, ctx.evPanel[b], 0));
        CUDA_CHECK(cudaStreamWaitEvent(sc, ctx.evFirst[1 - b], 0));
        if (nextOwner) {
            const int jl = (k + 1) / d.nprocs;
            update(jl, jl + 1, panel, g, sc);
        }
        CUDA_CHECK(cudaEventRecord(ctx.evLA[b], sc));
        if (nextOwner) factorAndSend(k + 1);

        // Rank 0 assembles L while the GPU works
        if (d.rank == 0) storePanel(A, hpanel, d, k);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    return failIdx;
}

static void teardownGpu(GpuContext& ctx) {
    for (int b = 0; b < 2; ++b) {
        cudaEventDestroy(ctx.evPanel[b]);
        cudaEventDestroy(ctx.evLA[b]);
        cudaEventDestroy(ctx.evBulk[b]);
        cudaEventDestroy(ctx.evFirst[b]);
        cudaFree(ctx.dpanel[b]);
    }
    for (int s = 0; s < NSLOT; ++s) cudaEventDestroy(ctx.evH2D[s]);
    cudaEventDestroy(ctx.evFact);
    cudaEventDestroy(ctx.evD2H);
    cudaStreamDestroy(ctx.sc);
    cudaStreamDestroy(ctx.su);
    cudaStreamDestroy(ctx.sx);
    cudaFree(ctx.dcols);
    cudaFree(ctx.doffs);
    cudaFree(ctx.dinfo);
    cudaFreeHost(ctx.hinfo);
    cudaHostUnregister(ctx.shmBase);
    MPI_Win_unlock_all(ctx.win);
    MPI_Win_free(&ctx.win);
}

int main(int argc, char** argv) {
    // Load all kernels at context creation (not lazily inside the timed region)
    setenv("CUDA_MODULE_LOADING", "EAGER", 0);
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

    Dist d;
    d.rank = rank;
    d.nprocs = nprocs;
    setupComm(d);
    const int localRank = d.localRank, localSize = d.localSize;

    // Node-local rank -> GPU, and share host cores among node-local ranks
    int ndev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    if (ndev < 1) {
        fprintf(stderr, "No CUDA device found\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % ndev));
    if (!getenv("OMP_NUM_THREADS")) {
        const long online = sysconf(_SC_NPROCESSORS_ONLN);
        int procs = omp_get_num_procs();
        if (procs >= online) procs = std::max(1, (int)(online / localSize));
        omp_set_num_threads(procs);
    }

    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Distribution setup
    d.n = n;
    // Block width: as wide as possible for GEMM efficiency while keeping enough
    // block columns per rank to keep the factorization pipeline short
    d.nb = 256;
    while (d.nb > 64 && n / d.nb < (size_t)64 * nprocs) d.nb /= 2;
    d.K = (int)((n + d.nb - 1) / d.nb);
    d.N = d.K * d.nb;
    d.nloc = (d.K > rank) ? (d.K - rank + nprocs - 1) / nprocs : 0;
    d.offs.resize(d.nloc + 1);
    d.offs[0] = 0;
    for (int jl = 0; jl < d.nloc; ++jl)
        d.offs[jl + 1] = d.offs[jl] + (size_t)(d.N - d.global(jl) * d.nb) * d.nb;

    // Allocate matrix (rank 0 holds the full result)
    std::vector<double> A;
    std::vector<double> A_orig;

    // Generate positive definite matrix: every rank builds its own block columns
    if (rank == 0) printf("Generating positive definite matrix...\n");
    std::vector<double> B(n * n);
    generateB(B, n);
    if (rank == 0) {
        A.resize(n * n);
        if (validate) {
            A_orig.resize(n * n);
            generatePositiveDefiniteMatrix(A_orig, B, n);  // Save original for validation
        }
    }
    // Local block columns in page-locked memory for fast upload
    double* hcols = nullptr;
    CUDA_CHECK(cudaMallocHost(&hcols, std::max<size_t>(1, d.offs[d.nloc]) * sizeof(double)));
    for (int jl = 0; jl < d.nloc; ++jl) {
        const size_t c0 = (size_t)d.global(jl) * d.nb;
        double* col = &hcols[d.offs[jl]];
#pragma omp parallel for schedule(dynamic, 4)
        for (size_t i = c0; i < (size_t)d.N; ++i) {
            double* row = col + (i - c0) * d.nb;
            for (int c = 0; c < d.nb; ++c) row[c] = 0.0;
            if (i >= n) {
                if (i < c0 + d.nb) row[i - c0] = 1.0;  // identity padding
                continue;
            }
            const size_t cEnd = std::min({c0 + d.nb, n, i + 1});
            if (!A_orig.empty()) {
                for (size_t gc = c0; gc < cEnd; ++gc) row[gc - c0] = A_orig[i * n + gc];
                continue;
            }
            for (size_t j0 = c0; j0 < cEnd; j0 += 8) {
                const int cnt = (int)std::min<size_t>(8, cEnd - j0);
                const double* ys[8];
                double out[8];
                for (int c = 0; c < cnt; ++c) ys[c] = &B[(j0 + c) * n];
                dotMulti(&B[i * n], ys, cnt, n, out);
                for (int c = 0; c < cnt; ++c) row[j0 + c - c0] = out[c] + (i == j0 + c ? (double)n : 0.0);
            }
        }
    }
    std::vector<double>().swap(B);

    // Allocate GPU resources outside the timed region
    GpuContext ctx;
    setupGpu(ctx, d);
    MPI_Barrier(MPI_COMM_WORLD);

    // Perform Cholesky decomposition
    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    auto start = std::chrono::high_resolution_clock::now();

    const long long failIdx = choleskyDistributed(ctx, d, hcols, A);
    MPI_Barrier(MPI_COMM_WORLD);

    auto end = std::chrono::high_resolution_clock::now();
    teardownGpu(ctx);
    cudaFreeHost(hcols);
    MPI_Comm_free(&d.localComm);
    if (d.leaderComm != MPI_COMM_NULL) MPI_Comm_free(&d.leaderComm);
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (failIdx >= 0) {
        if (rank == 0) {
            printf("Error: Matrix is not positive definite at diagonal element %zu\n", (size_t)failIdx);
            printf("Cholesky decomposition failed\n");
        }
        MPI_Finalize();
        return 1;
    }

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
