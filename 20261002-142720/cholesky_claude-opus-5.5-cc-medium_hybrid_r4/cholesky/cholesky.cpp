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

#include "../common/results_output.hpp"

// Hybrid MPI + OpenMP + CUDA Cholesky decomposition.
//
// Decomposes positive definite matrix A into L * L^T where L is lower triangular.
// The matrix is split into NB-wide block columns, distributed 1D block-cyclically
// over MPI ranks (one GPU per rank). Each rank keeps its block columns resident on
// its GPU (column-major, full height). A right-looking blocked algorithm with
// one-step lookahead is used:
//   - owner of block column k factors the diagonal tile (POTRF) and solves the
//     panel below it (TRSM) on its GPU,
//   - the panel is broadcast to all ranks (MPI, staged via pinned host memory),
//   - every rank applies the rank-NB update (SYRK/GEMM) to its own trailing block
//     columns on its GPU.
// Rank 0 collects all panels (they are broadcast anyway) into the row-major result.
// OpenMP is used for host-side work (matrix generation, result assembly, validation).

#define CUDA_CHECK(call)                                                                  \
    do {                                                                                  \
        cudaError_t err_ = (call);                                                        \
        if (err_ != cudaSuccess) {                                                        \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__, \
                    __LINE__);                                                            \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                 \
        }                                                                                 \
    } while (0)

constexpr int NB = 64;             // block size (tile edge)
constexpr int TRSM_THREADS = 128;  // rows per TRSM thread block
constexpr int UPD_KC = 32;         // K-chunk in the update kernel

// ---------------------------------------------------------------------------------
// CUDA kernels
// ---------------------------------------------------------------------------------

// Factor the bs x bs diagonal tile starting at global row/col k0, stored in local
// column-major storage at local column lcol. Writes L into storage and into the
// row-major panel buffer W (row stride NB, global row index).
__global__ void potrfKernel(double* __restrict__ loc, int n, size_t lcol, int k0, int bs,
                            double* __restrict__ W, int* __restrict__ info) {
    __shared__ double s[NB][NB + 1];
    const int tid = threadIdx.x;
    const int nth = blockDim.x;

    for (int idx = tid; idx < bs * bs; idx += nth) {
        const int c = idx / bs, r = idx % bs;
        s[r][c] = loc[(lcol + c) * (size_t)n + k0 + r];
    }
    __syncthreads();

    for (int j = 0; j < bs; ++j) {
        if (tid == 0) {
            const double val = s[j][j];
            if (val <= 0.0) atomicMin(info, k0 + j);
            s[j][j] = sqrt(val);
        }
        __syncthreads();
        const double d = s[j][j];
        for (int i = j + 1 + tid; i < bs; i += nth) s[i][j] = s[i][j] / d;
        __syncthreads();
        const int m = bs - j - 1;
        for (int idx = tid; idx < m * m; idx += nth) {
            const int i = j + 1 + idx / m;
            const int c = j + 1 + idx % m;
            if (c <= i) s[i][c] -= s[i][j] * s[c][j];
        }
        __syncthreads();
    }

    for (int idx = tid; idx < bs * bs; idx += nth) {
        const int c = idx / bs, r = idx % bs;
        loc[(lcol + c) * (size_t)n + k0 + r] = (c <= r) ? s[r][c] : 0.0;
    }
    for (int idx = tid; idx < bs * NB; idx += nth) {
        const int r = idx / NB, c = idx % NB;
        W[(size_t)(k0 + r) * NB + c] = (c <= r && c < bs) ? s[r][c] : 0.0;
    }
}

// Solve X * L_kk^T = A_ik for all rows below the diagonal tile (full NB width).
__global__ void __launch_bounds__(TRSM_THREADS)
trsmKernel(double* __restrict__ loc, int n, size_t lcol, int k0, double* __restrict__ W) {
    __shared__ double Ls[NB][NB + 1];
    for (int idx = threadIdx.x; idx < NB * NB; idx += blockDim.x) {
        const int r = idx / NB, c = idx % NB;
        Ls[r][c] = W[(size_t)(k0 + r) * NB + c];
    }
    __syncthreads();

    const int i = k0 + NB + blockIdx.x * TRSM_THREADS + threadIdx.x;
    if (i >= n) return;

    double x[NB];
#pragma unroll
    for (int c = 0; c < NB; ++c) x[c] = loc[(lcol + c) * (size_t)n + i];
#pragma unroll
    for (int j = 0; j < NB; ++j) {
        double s = x[j];
#pragma unroll
        for (int m = 0; m < j; ++m) s -= x[m] * Ls[j][m];
        x[j] = s / Ls[j][j];
    }
#pragma unroll
    for (int c = 0; c < NB; ++c) loc[(lcol + c) * (size_t)n + i] = x[c];
#pragma unroll
    for (int c = 0; c < NB; ++c) W[(size_t)i * NB + c] = x[c];
}

// Trailing update with panel k: for every local block column lb in [lb0, lb0+gridDim.y)
// (global block column jb = lb*P + rank) and row tile ti >= jb:
//   C(ti, jb) -= W(ti) * W(jb)^T
__global__ void __launch_bounds__(256)
updateKernel(double* __restrict__ loc, int n, const double* __restrict__ W, int k, int lb0,
             int P, int rank) {
    const int lb = lb0 + blockIdx.y;
    const int jb = lb * P + rank;
    const int ti = k + 1 + blockIdx.x;
    if (ti < jb) return;

    __shared__ double As[UPD_KC][NB + 1];
    __shared__ double Bs[UPD_KC][NB + 1];

    const int tid = threadIdx.x;
    const int tx = tid % 16, ty = tid / 16;
    const int r0 = ti * NB;
    const int c0 = jb * NB;

    double acc[4][4];
#pragma unroll
    for (int a = 0; a < 4; ++a)
#pragma unroll
        for (int b = 0; b < 4; ++b) acc[a][b] = 0.0;

    for (int q0 = 0; q0 < NB; q0 += UPD_KC) {
        for (int idx = tid; idx < NB * UPD_KC; idx += 256) {
            const int r = idx / UPD_KC, q = idx % UPD_KC;
            As[q][r] = (r0 + r < n) ? W[(size_t)(r0 + r) * NB + q0 + q] : 0.0;
            Bs[q][r] = (c0 + r < n) ? W[(size_t)(c0 + r) * NB + q0 + q] : 0.0;
        }
        __syncthreads();
#pragma unroll 8
        for (int q = 0; q < UPD_KC; ++q) {
            double av[4], bv[4];
#pragma unroll
            for (int a = 0; a < 4; ++a) av[a] = As[q][tx + 16 * a];
#pragma unroll
            for (int b = 0; b < 4; ++b) bv[b] = Bs[q][ty + 16 * b];
#pragma unroll
            for (int a = 0; a < 4; ++a)
#pragma unroll
                for (int b = 0; b < 4; ++b) acc[a][b] = fma(av[a], bv[b], acc[a][b]);
        }
        __syncthreads();
    }

    const size_t lc0 = (size_t)lb * NB;
#pragma unroll
    for (int b = 0; b < 4; ++b) {
        const int c = ty + 16 * b;
        if (c0 + c >= n) continue;
        double* col = loc + (lc0 + c) * (size_t)n + r0;
#pragma unroll
        for (int a = 0; a < 4; ++a) {
            const int r = tx + 16 * a;
            if (r0 + r < n) col[r] -= acc[a][b];
        }
    }
}

// ---------------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------------

// Generate random matrix B exactly as the original benchmark does.
static std::vector<double> generateB(const size_t n) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }
    return B;
}

// Compute A(i, j) = dot(B_i, B_j) (+ n on the diagonal) for 4 rows i0..i0+3 and column
// j; each dot product is accumulated in the same order as the original code.
static inline void dot4(const double* __restrict__ B, size_t n, size_t i0, size_t cnt, size_t j,
                        double out[4]) {
    const double* bj = B + j * n;
    if (cnt == 4) {
        const double* b0 = B + i0 * n;
        const double* b1 = b0 + n;
        const double* b2 = b1 + n;
        const double* b3 = b2 + n;
        double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
        for (size_t k = 0; k < n; ++k) {
            const double v = bj[k];
            s0 += b0[k] * v;
            s1 += b1[k] * v;
            s2 += b2[k] * v;
            s3 += b3[k] * v;
        }
        out[0] = s0; out[1] = s1; out[2] = s2; out[3] = s3;
    } else {
        for (size_t t = 0; t < cnt; ++t) {
            const double* bi = B + (i0 + t) * n;
            double s = 0.0;
            for (size_t k = 0; k < n; ++k) s += bi[k] * bj[k];
            out[t] = s;
        }
    }
    for (size_t t = 0; t < cnt; ++t)
        if (i0 + t == j) out[t] += n;
}

// Generate the full symmetric positive definite matrix (row-major), as the original.
void generatePositiveDefiniteMatrix(std::vector<double>& A, const std::vector<double>& B,
                                    const size_t n) {
#pragma omp parallel for schedule(dynamic, 1)
    for (size_t j = 0; j < n; ++j) {
        double out[4];
        for (size_t i = j; i < n; i += 4) {
            const size_t cnt = std::min<size_t>(4, n - i);
            dot4(B.data(), n, i, cnt, j, out);
            for (size_t t = 0; t < cnt; ++t) {
                A[(i + t) * n + j] = out[t];
                A[j * n + i + t] = out[t];
            }
        }
    }
}

// Generate this rank's block columns (column-major, full height, lower part only).
void generateLocalColumns(double* loc, const std::vector<double>& B, size_t n,
                          int nt, int P, int rank) {
    const int nlb = (nt - rank + P - 1) / P;
    const size_t total = (size_t)std::max(nlb, 0) * NB;
#pragma omp parallel for schedule(dynamic, 1)
    for (size_t lc = 0; lc < total; ++lc) {
        const size_t lb = lc / NB;
        const size_t jb = lb * P + rank;
        const size_t j = jb * NB + lc % NB;
        if (j >= n) continue;
        double out[4];
        for (size_t i = jb * NB; i < n; i += 4) {
            const size_t cnt = std::min<size_t>(4, n - i);
            dot4(B.data(), n, i, cnt, j, out);
            for (size_t t = 0; t < cnt; ++t) loc[lc * n + i + t] = out[t];
        }
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix
    double maxError = 0.0;
    double relError = 0.0;

#pragma omp parallel for schedule(dynamic, 1) reduction(max : maxError, relError)
    for (size_t i = 0; i < n; ++i) {
        const double* li = L.data() + i * n;
        for (size_t j = 0; j <= i; ++j) {
            // L is lower triangular: terms with k > j vanish
            const double* lj = L.data() + j * n;
            double sum = 0.0;
            for (size_t k = 0; k <= j; ++k) sum += li[k] * lj[k];

            const double e1 = fabs(sum - A_orig[i * n + j]);
            const double e2 = fabs(sum - A_orig[j * n + i]);
            maxError = std::max(maxError, std::max(e1, e2));
            relError = std::max(relError, e1 / (fabs(A_orig[i * n + j]) + 1e-10));
            relError = std::max(relError, e2 / (fabs(A_orig[j * n + i]) + 1e-10));
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

using Clock = std::chrono::high_resolution_clock;

// Distributed GPU Cholesky. The local block columns are in `locHost` on every rank;
// on return, rank 0 has the full lower-triangular factor in row-major `A` (upper part
// zero). Returns the first failing diagonal index, or n on success.
// `start`/`end` bracket the decomposition itself (one-time setup such as device/shared
// memory allocation and cleanup is excluded, like allocating A in the original code).
size_t choleskyDecomposition(std::vector<double>& A, const double* locHost,
                             const size_t nsz, const int P, const int rank,
                             Clock::time_point& start, Clock::time_point& end) {
    const int n = (int)nsz;
    const int nt = (n + NB - 1) / NB;
    const int nlb = (nt - rank + P - 1) / P;
    const size_t nlocCols = (size_t)std::max(nlb, 0) * NB;
    auto owner = [&](int kb) { return kb % P; };
    auto lcolOf = [&](int kb) { return (size_t)(kb / P) * NB; };

    // Node topology: panels are exchanged through an MPI shared-memory window within a
    // node (GPU -> shared pinned host memory -> GPUs) and via MPI_Bcast between nodes.
    MPI_Comm nodeComm, leaderComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int nodeRank;
    MPI_Comm_rank(nodeComm, &nodeRank);
    MPI_Comm_split(MPI_COMM_WORLD, nodeRank == 0 ? 0 : MPI_UNDEFINED, rank, &leaderComm);
    int myNode = 0, numNodes = 1;
    if (nodeRank == 0) {
        MPI_Comm_rank(leaderComm, &myNode);
        MPI_Comm_size(leaderComm, &numNodes);
    }
    MPI_Bcast(&myNode, 1, MPI_INT, 0, nodeComm);
    MPI_Bcast(&numNodes, 1, MPI_INT, 0, nodeComm);
    std::vector<int> nodeOf(P);
    MPI_Allgather(&myNode, 1, MPI_INT, nodeOf.data(), 1, MPI_INT, MPI_COMM_WORLD);

    constexpr int NBUF = 3;
    const size_t panelElems = (size_t)nt * NB * NB;
    const size_t panelBytes = std::max<size_t>(panelElems, 1) * sizeof(double);
    MPI_Win win;
    double* shBase = nullptr;
    MPI_Win_allocate_shared(nodeRank == 0 ? (MPI_Aint)(NBUF * panelBytes) : 0, sizeof(double),
                            MPI_INFO_NULL, nodeComm, &shBase, &win);
    {
        MPI_Aint sz;
        int du;
        MPI_Win_shared_query(win, 0, &sz, &du, &shBase);
    }
    MPI_Win_lock_all(MPI_MODE_NOCHECK, win);
    const bool registered =
        cudaHostRegister(shBase, NBUF * panelBytes, cudaHostRegisterPortable) == cudaSuccess;
    if (!registered) cudaGetLastError();  // fall back to pageable transfers

    double* dLoc = nullptr;
    double* dW[NBUF];
    double* hW[NBUF];
    int* dInfo;
    CUDA_CHECK(cudaMalloc(&dLoc, std::max<size_t>(nlocCols * n, 1) * sizeof(double)));
    for (int b = 0; b < NBUF; ++b) {
        CUDA_CHECK(cudaMalloc(&dW[b], panelBytes));
        hW[b] = shBase + b * (panelBytes / sizeof(double));
    }
    CUDA_CHECK(cudaMalloc(&dInfo, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(dInfo, &n, sizeof(int), cudaMemcpyHostToDevice));

    // sMain: bulk trailing updates; sLook: high-priority lookahead (next column update,
    // POTRF, TRSM) running concurrently; sCopy: host <-> device panel transfers.
    cudaStream_t sMain, sLook, sCopy;
    int prioLeast, prioGreatest;
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&prioLeast, &prioGreatest));
    CUDA_CHECK(cudaStreamCreateWithPriority(&sMain, cudaStreamNonBlocking, prioLeast));
    CUDA_CHECK(cudaStreamCreateWithPriority(&sLook, cudaStreamNonBlocking, prioGreatest));
    CUDA_CHECK(cudaStreamCreateWithFlags(&sCopy, cudaStreamNonBlocking));
    cudaEvent_t evFact[NBUF], evD2H[NBUF], evH2D[NBUF], evFree[NBUF], evColReady;
    CUDA_CHECK(cudaEventCreateWithFlags(&evColReady, cudaEventDisableTiming));
    for (int b = 0; b < NBUF; ++b) {
        CUDA_CHECK(cudaEventCreateWithFlags(&evFact[b], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&evD2H[b], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&evH2D[b], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&evFree[b], cudaEventDisableTiming));
    }

    // Warm up (forces lazy module loading); all kernels exit without side effects
    if (nt > 0) {
        potrfKernel<<<1, 256, 0, sMain>>>(dLoc, n, 0, 0, 0, dW[0], dInfo);
        trsmKernel<<<1, TRSM_THREADS, 0, sMain>>>(dLoc, 0, 0, 0, dW[0]);
        updateKernel<<<1, 256, 0, sMain>>>(dLoc, n, dW[0], -1, 1, P, rank);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    start = Clock::now();

    // Upload local block columns (only rows at/below each block's diagonal are used)
    for (int lb = 0; lb < nlb; ++lb) {
        const size_t r0 = (size_t)(lb * P + rank) * NB;
        const size_t w = std::min<size_t>(NB, n - r0);
        CUDA_CHECK(cudaMemcpy2DAsync(dLoc + (size_t)lb * NB * n + r0, n * sizeof(double),
                                     locHost + (size_t)lb * NB * n + r0, n * sizeof(double),
                                     (n - r0) * sizeof(double), w, cudaMemcpyHostToDevice,
                                     sMain));
    }
    CUDA_CHECK(cudaEventRecord(evColReady, sMain));

    // Factor panel kb (owner only): POTRF + TRSM, then stage panel to shared host memory.
    auto factorPanel = [&](int kb) {
        const int buf = kb % NBUF;
        const int k0 = kb * NB;
        const int bs = std::min(NB, n - k0);
        const size_t lcol = lcolOf(kb);
        potrfKernel<<<1, 256, 0, sLook>>>(dLoc, n, lcol, k0, bs, dW[buf], dInfo);
        const int below = n - k0 - NB;
        if (below > 0) {
            trsmKernel<<<(below + TRSM_THREADS - 1) / TRSM_THREADS, TRSM_THREADS, 0, sLook>>>(
                dLoc, n, lcol, k0, dW[buf]);
        }
        CUDA_CHECK(cudaEventRecord(evFact[buf], sLook));
        CUDA_CHECK(cudaStreamWaitEvent(sCopy, evFact[buf], 0));
        CUDA_CHECK(cudaMemcpyAsync(hW[buf] + (size_t)k0 * NB, dW[buf] + (size_t)k0 * NB,
                                   (size_t)(n - k0) * NB * sizeof(double),
                                   cudaMemcpyDeviceToHost, sCopy));
        CUDA_CHECK(cudaEventRecord(evD2H[buf], sCopy));
    };

    // Apply panel kb's update to local block columns lb in [lb0, lb1).
    auto update = [&](int kb, int lb0, int lb1, cudaStream_t st) {
        if (lb1 <= lb0) return;
        const int rowTiles = nt - (kb + 1);
        if (rowTiles <= 0) return;
        dim3 grid(rowTiles, lb1 - lb0);
        updateKernel<<<grid, 256, 0, st>>>(dLoc, n, dW[kb % NBUF], kb, lb0, P, rank);
    };
    // First local block index whose global block index is >= g.
    auto firstLocalAtLeast = [&](int g) {
        if (g <= rank) return 0;
        return (g - rank + P - 1) / P;
    };
    auto nodeBarrier = [&]() {
        MPI_Win_sync(win);
        MPI_Barrier(nodeComm);
        MPI_Win_sync(win);
    };

    // A handful of threads saturate the copy bandwidth for one panel; more threads only
    // add fork/join overhead on the critical path
    const int asmThreads = std::max(1, std::min(omp_get_max_threads(), 8));

    for (int k = 0; k < nt; ++k) {
        const int cur = k % NBUF;
        const int k0 = k * NB;
        const int own = owner(k);
        const bool isOwner = own == rank;
        if (k == 0 && isOwner) {
            CUDA_CHECK(cudaStreamWaitEvent(sLook, evColReady, 0));
            factorPanel(0);
        }

        // Distribute panel k (rows k0..n-1). Before the node barrier every rank makes
        // sure its reads of the buffer used by panel k-2 are complete, so that the
        // owner of panel k+1 (factored right after this barrier) may overwrite it.
        const size_t off = (size_t)k0 * NB;
        const size_t count = (size_t)(n - k0) * NB;
        CUDA_CHECK(cudaEventSynchronize(evH2D[(k + 1) % NBUF]));
        if (isOwner) CUDA_CHECK(cudaEventSynchronize(evD2H[cur]));
        const int ownNode = nodeOf[own];
        if (myNode == ownNode) nodeBarrier();
        if (numNodes > 1 && nodeRank == 0)
            MPI_Bcast(hW[cur] + off, (int)count, MPI_DOUBLE, ownNode, leaderComm);
        if (myNode != ownNode) nodeBarrier();

        if (!isOwner && k + 1 < nt) {
            CUDA_CHECK(cudaStreamWaitEvent(sCopy, evFree[cur], 0));
            CUDA_CHECK(cudaMemcpyAsync(dW[cur] + off, hW[cur] + off, count * sizeof(double),
                                       cudaMemcpyHostToDevice, sCopy));
            CUDA_CHECK(cudaEventRecord(evH2D[cur], sCopy));
            CUDA_CHECK(cudaStreamWaitEvent(sMain, evH2D[cur], 0));
            CUDA_CHECK(cudaStreamWaitEvent(sLook, evH2D[cur], 0));
        } else if (isOwner) {
            CUDA_CHECK(cudaStreamWaitEvent(sMain, evFact[cur], 0));
        }

        // Lookahead (high priority): update and factor the next panel. Column k+1 has
        // already received all earlier updates (evColReady); its panel buffer must no
        // longer be in use by the updates of panel k-2.
        if (k + 1 < nt && owner(k + 1) == rank) {
            const int lbn = (k + 1) / P;
            CUDA_CHECK(cudaStreamWaitEvent(sLook, evColReady, 0));
            CUDA_CHECK(cudaStreamWaitEvent(sLook, evFree[(k + 1) % NBUF], 0));
            update(k, lbn, lbn + 1, sLook);
            factorPanel(k + 1);
        }
        // Remaining trailing update: column k+2 first, so the next lookahead can start
        const int lbStart = firstLocalAtLeast(k + 2);
        if (k + 2 < nt && owner(k + 2) == rank) {
            update(k, lbStart, lbStart + 1, sMain);
            CUDA_CHECK(cudaEventRecord(evColReady, sMain));
            update(k, lbStart + 1, nlb, sMain);
        } else {
            update(k, lbStart, nlb, sMain);
        }
        CUDA_CHECK(cudaEventRecord(evFree[cur], sMain));

        // Rank 0 assembles the row-major result while the GPU works
        if (rank == 0) {
            const int w = std::min(NB, n - k0);
            const double* src = hW[cur] + off;
#pragma omp parallel for schedule(static) num_threads(asmThreads) if (n - k0 > 512)
            for (int i = k0; i < n; ++i) {
                const double* s = src + (size_t)(i - k0) * NB;
                double* d = A.data() + (size_t)i * n + k0;
                const int lim = std::min(w, i - k0 + 1);
                for (int c = 0; c < lim; ++c) d[c] = s[c];
                for (int c = lim; c < w; ++c) d[c] = 0.0;
            }
        }
    }

    CUDA_CHECK(cudaStreamSynchronize(sMain));
    CUDA_CHECK(cudaStreamSynchronize(sLook));
    CUDA_CHECK(cudaStreamSynchronize(sCopy));
    int info = n;
    CUDA_CHECK(cudaMemcpy(&info, dInfo, sizeof(int), cudaMemcpyDeviceToHost));
    int ginfo = n;
    MPI_Allreduce(&info, &ginfo, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    end = Clock::now();

    for (int b = 0; b < NBUF; ++b) {
        cudaEventDestroy(evFact[b]);
        cudaEventDestroy(evD2H[b]);
        cudaEventDestroy(evH2D[b]);
        cudaEventDestroy(evFree[b]);
        cudaFree(dW[b]);
    }
    if (registered) cudaHostUnregister(shBase);
    MPI_Win_unlock_all(win);
    MPI_Win_free(&win);
    if (leaderComm != MPI_COMM_NULL) MPI_Comm_free(&leaderComm);
    MPI_Comm_free(&nodeComm);
    cudaEventDestroy(evColReady);
    cudaStreamDestroy(sMain);
    cudaStreamDestroy(sLook);
    cudaStreamDestroy(sCopy);
    cudaFree(dLoc);
    cudaFree(dInfo);
    return (size_t)ginfo;
}

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank, P;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);
    const bool root = rank == 0;

    // One GPU per rank (round-robin over node-local ranks)
    {
        MPI_Comm local;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
        int localRank;
        MPI_Comm_rank(local, &localRank);
        MPI_Comm_free(&local);
        int ndev = 0;
        CUDA_CHECK(cudaGetDeviceCount(&ndev));
        if (ndev <= 0) {
            fprintf(stderr, "No CUDA device found\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        CUDA_CHECK(cudaSetDevice(localRank % ndev));
    }

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

    if (root) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const int nt = (int)((n + NB - 1) / NB);
    const int nlb = std::max((nt - rank + P - 1) / P, 0);

    // Allocate matrix (result lives on rank 0)
    std::vector<double> A(root ? n * n : 0);
    std::vector<double> A_orig;
    // Local block columns are kept in pinned memory for fast transfer to the GPU
    double* locHost = nullptr;
    CUDA_CHECK(cudaMallocHost(&locHost, std::max<size_t>((size_t)nlb * NB * n, 1) * sizeof(double)));

    // Generate positive definite matrix: every rank generates its own block columns
    if (root) printf("Generating positive definite matrix...\n");
    {
        std::vector<double> B = generateB(n);
        generateLocalColumns(locHost, B, n, nt, P, rank);
        if (root && validate) {
            A_orig.resize(n * n);
            generatePositiveDefiniteMatrix(A_orig, B, n);  // Save original for validation
        }
    }

    // Perform Cholesky decomposition
    if (root) printf("Computing Cholesky decomposition...\n");
    fflush(stdout);
    Clock::time_point start, end;

    const size_t fail = choleskyDecomposition(A, locHost, n, P, rank, start, end);
    const bool success = fail >= n;

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    int ret = 0;
    if (!success) {
        if (root) {
            printf("Error: Matrix is not positive definite at diagonal element %zu\n", fail);
            printf("Cholesky decomposition failed\n");
        }
        ret = 1;
    } else if (root) {
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

    cudaFreeHost(locHost);
    MPI_Bcast(&ret, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return ret;
}
