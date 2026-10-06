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
#include <unistd.h>

#include "../common/results_output.hpp"

// Hybrid MPI + OpenMP + CUDA Cholesky decomposition
// Decomposes positive definite matrix A into L * L^T where L is lower triangular
//
// Blocked right-looking algorithm, 1D block-cyclic distribution of column panels
// (width NB) over MPI ranks, one GPU per rank:
//   - panel factorization (diagonal block + triangular solve) on the CPU (OpenMP)
//   - trailing-matrix updates (SYRK/GEMM) on the GPU (custom CUDA kernel)
//   - factored panels are broadcast with MPI; lookahead overlaps the factorization
//     of panel k+1 with the trailing update of step k.

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), \
                    __FILE__, __LINE__);                                          \
            MPI_Abort(MPI_COMM_WORLD, 1);                                         \
        }                                                                         \
    } while (0)

static int g_rank = 0;
static int g_size = 1;

constexpr int NB = 128;  // panel width
constexpr int TM = 64;   // GPU tile rows
constexpr int TN = 64;   // GPU tile cols
constexpr int KC = 16;   // GPU k-chunk

// Thread count for the latency-bound CPU work on the critical path (panel solves,
// result collection): more threads only add fork/join and wake-up overhead.
inline int critThreads() { return std::min(omp_get_max_threads(), 16); }

// Trailing update: for each local panel j (>= first), rows i >= j*NB:
//   P_j[i][c] -= sum_p Lk[i][p] * Lk[j*NB + c][p]
// Panels and Lk are row-major with leading dimension NB; row r at offset r*NB.
__global__ void __launch_bounds__(256)
trailingUpdateKernel(const double* __restrict__ Lk, double* __restrict__ panels,
                     int lfirst, int nprocs, int rank, int Np, int kw) {
    const int colTiles = NB / TN;
    const int l = lfirst + blockIdx.y / colTiles;
    const int ct = blockIdx.y % colTiles;
    const int j = l * nprocs + rank;
    const int row0 = j * NB + blockIdx.x * TM;
    if (row0 >= Np) return;
    const int col0 = ct * TN;
    const int brow0 = j * NB + col0;  // rows of Lk providing the "B" operand

    double* __restrict__ C = panels + (size_t)l * Np * NB;

    __shared__ double As[KC][TM];
    __shared__ double Bs[KC][TN];

    const int tid = threadIdx.x;
    const int tx = tid % 16;
    const int ty = tid / 16;
    const int lr = tid / 4;        // 0..63 load row
    const int lk = (tid % 4) * 4;  // 0,4,8,12 load k

    double acc[4][4];
#pragma unroll
    for (int a = 0; a < 4; ++a)
#pragma unroll
        for (int b = 0; b < 4; ++b) acc[a][b] = 0.0;

    for (int k0 = 0; k0 < kw; k0 += KC) {
        const double2* pa = reinterpret_cast<const double2*>(Lk + (size_t)(row0 + lr) * NB + k0 + lk);
        const double2* pb = reinterpret_cast<const double2*>(Lk + (size_t)(brow0 + lr) * NB + k0 + lk);
        double2 a0 = pa[0], a1 = pa[1];
        double2 b0 = pb[0], b1 = pb[1];
        __syncthreads();
        As[lk + 0][lr] = a0.x; As[lk + 1][lr] = a0.y;
        As[lk + 2][lr] = a1.x; As[lk + 3][lr] = a1.y;
        Bs[lk + 0][lr] = b0.x; Bs[lk + 1][lr] = b0.y;
        Bs[lk + 2][lr] = b1.x; Bs[lk + 3][lr] = b1.y;
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < KC; ++kk) {
            double av[4], bv[4];
#pragma unroll
            for (int a = 0; a < 4; ++a) av[a] = As[kk][ty + 16 * a];
#pragma unroll
            for (int b = 0; b < 4; ++b) bv[b] = Bs[kk][tx + 16 * b];
#pragma unroll
            for (int a = 0; a < 4; ++a)
#pragma unroll
                for (int b = 0; b < 4; ++b) acc[a][b] = fma(av[a], bv[b], acc[a][b]);
        }
    }

#pragma unroll
    for (int a = 0; a < 4; ++a) {
        double* crow = C + (size_t)(row0 + ty + 16 * a) * NB + col0;
#pragma unroll
        for (int b = 0; b < 4; ++b) crow[tx + 16 * b] -= acc[a][b];
    }
}

// Factor the w x w diagonal block of a panel (ld NB) on the CPU with a right-looking
// unblocked Cholesky; LT receives the transposed factor (LT[j*NB+i] = L[i][j]).
// Returns -1 on success or the global index of the failing diagonal element.
static long factorDiagBlock(double* H, double* LT, int w, size_t r0) {
    for (int j = 0; j < w; ++j) {
        const double val = H[j * NB + j];
        if (val <= 0.0) return (long)(r0 + j);
        const double d = sqrt(val);
        H[j * NB + j] = d;
        double* col = LT + j * NB;
        col[j] = d;
        for (int i = j + 1; i < w; ++i) col[i] = (H[i * NB + j] /= d);
        for (int i = j + 1; i < w; ++i) {
            const double li = col[i];
            double* hi = H + i * NB;
#pragma omp simd
            for (int k = j + 1; k <= i; ++k) hi[k] -= li * col[k];
        }
    }
    // Zero out upper triangular part of the diagonal block
    for (int i = 0; i < w; ++i)
        for (int j = i + 1; j < NB; ++j) H[i * NB + j] = 0.0;
    return -1;
}

// Triangular solve X * L^T = B for panel rows [i0, i1) (rows are independent)
static void solvePanelRows(double* H, const double* LT, int w, size_t i0, size_t i1) {
#pragma omp parallel for schedule(static) num_threads(critThreads())
    for (size_t i = i0; i < i1; ++i) {
        double* x = H + i * NB;
        for (int j = 0; j < w; ++j) {
            const double* col = LT + j * NB;
            const double xj = (x[j] /= col[j]);
#pragma omp simd
            for (int k = j + 1; k < w; ++k) x[k] -= xj * col[k];
        }
    }
}

// Distributed state prepared before the timed section
struct DistMatrix {
    size_t n = 0;
    int NT = 0;          // number of panels
    size_t Np = 0;       // padded size NT*NB
    int nloc = 0;        // number of local panels
    double* hLocal = nullptr;  // pinned host copy of local panels (nloc x Np x NB)
};

static inline int ownerOf(int j) { return j % g_size; }

// Distribute panels of A (rank 0) to the owners' host memory (setup, not timed)
static void distributeMatrix(const std::vector<double>& A, DistMatrix& D) {
    const size_t n = D.n;
    const size_t panelElems = D.Np * NB;
    CUDA_CHECK(cudaMallocHost(&D.hLocal, std::max<size_t>(1, D.nloc * panelElems) * sizeof(double)));
    std::vector<double> tmp;
    if (g_rank == 0 && g_size > 1) tmp.resize(panelElems);
    for (int j = 0; j < D.NT; ++j) {
        const int own = ownerOf(j);
        const size_t r0 = (size_t)j * NB;
        const size_t m = D.Np - r0;
        const int w = (int)std::min<size_t>(NB, n - r0);
        if (g_rank == 0) {
            double* dst = (own == 0) ? D.hLocal + (size_t)(j / g_size) * panelElems + r0 * NB : tmp.data();
#pragma omp parallel for schedule(static)
            for (size_t i = 0; i < m; ++i) {
                const size_t gi = r0 + i;
                for (int c = 0; c < NB; ++c)
                    dst[i * NB + c] = (gi < n && c < w) ? A[gi * n + r0 + c] : 0.0;
            }
            if (own != 0) MPI_Send(dst, (int)(m * NB), MPI_DOUBLE, own, j, MPI_COMM_WORLD);
        } else if (own == g_rank) {
            MPI_Recv(D.hLocal + (size_t)(j / g_size) * panelElems + r0 * NB, (int)(m * NB), MPI_DOUBLE, 0, j,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
}

// Launch the trailing update by panel k of cnt local panels starting at local index lfirst
static void launchUpdate(const double* dLk, double* dPanels, int lfirst, int cnt, int k, const DistMatrix& D,
                         cudaStream_t s) {
    if (cnt <= 0) return;
    const int jfirst = lfirst * g_size + g_rank;
    const size_t rows = D.Np - (size_t)jfirst * NB;
    dim3 grid((unsigned)(rows / TM), (unsigned)(cnt * (NB / TN)));
    const int kw = (int)std::min<size_t>(NB, D.n - (size_t)k * NB);
    const int kwPad = (kw + KC - 1) / KC * KC;
    trailingUpdateKernel<<<grid, 256, 0, s>>>(dLk, dPanels, lfirst, g_size, g_rank, (int)D.Np, kwPad);
    CUDA_CHECK(cudaGetLastError());
}

// Copy finished panel rows [i0, i1) (global row indices; H holds rows from r0 with
// ld NB) into the row-major result A on rank 0, zeroing the upper triangular part
static void unpackRows(std::vector<double>& A, const double* H, size_t r0, int w, size_t i0, size_t i1, size_t n) {
#pragma omp parallel for schedule(static) num_threads(critThreads())
    for (size_t i = i0; i < i1; ++i) {
        const double* src = H + (i - r0) * NB;
        double* dst = &A[i * n + r0];
        for (int c = 0; c < w; ++c) dst[c] = src[c];
        if (i < r0 + w) {
            // last panel touching row i: zero out upper triangular part
            for (size_t j = i + 1; j < n; ++j) A[i * n + j] = 0.0;
        }
    }
}

constexpr size_t CHUNK_ROWS = 512;  // pipelining granularity of panel transfers

struct GpuCtx {
    double* dPanels = nullptr;  // local panels (nloc x Np x NB)
    double* dL[2] = {nullptr, nullptr};  // received remote panels (double-buffered)
    double* hbuf[2] = {nullptr, nullptr};  // pinned host panel buffers (+1 status element)
    cudaStream_t sMain, sLook, sCopy;
    cudaEvent_t evCopy;
    std::vector<cudaEvent_t> evChunk;
    alignas(64) double LT[NB * NB];
};

// Factor (owner) or receive (others) panel j and pass it on along a pipelined ring.
// The panel travels in row chunks: the owner overlaps device->host copies, CPU
// factorization, sending and the host->device write-back; the other ranks forward
// each chunk and upload it to the GPU while the rest is still in flight.
static void panelPipeline(int j, const DistMatrix& D, GpuCtx& G, std::vector<double>& A) {
    const size_t Np = D.Np, n = D.n;
    const size_t r0 = (size_t)j * NB;
    const size_t m = Np - r0;
    const size_t mValid = n - r0;
    const int w = (int)std::min<size_t>(NB, mValid);
    const size_t nch = (m + CHUNK_ROWS - 1) / CHUNK_ROWS;
    double* hb = G.hbuf[j % 2];
    const int root = ownerOf(j);
    // Ring order: root, root+1, ... with rank 0 (which also collects the result) last
    std::vector<int> ring{root};
    for (int q = 1; q < g_size; ++q)
        if ((root + q) % g_size != 0) ring.push_back((root + q) % g_size);
    if (root != 0) ring.push_back(0);
    const int pos = (int)(std::find(ring.begin(), ring.end(), g_rank) - ring.begin());
    const int prev = pos > 0 ? ring[pos - 1] : -1;
    const int next = pos < g_size - 1 ? ring[pos + 1] : -1;
    const bool forward = next >= 0;
    // rank 0 collects the final factor rows as they arrive
    auto collect = [&](size_t c) {
        if (g_rank != 0) return;
        const size_t i0 = r0 + c * CHUNK_ROWS;
        const size_t i1 = std::min(n, i0 + CHUNK_ROWS);
        if (i0 < i1) unpackRows(A, hb, r0, w, i0, i1, n);
    };
    std::vector<MPI_Request> reqs;
    reqs.reserve(nch);

    auto chunkRows = [&](size_t c) { return std::min(CHUNK_ROWS, m - c * CHUNK_ROWS); };
    // the status element travels with the last chunk
    auto chunkCount = [&](size_t c) { return (int)(chunkRows(c) * NB + (c == nch - 1 ? 1 : 0)); };

    if (root == g_rank) {
        double* dp = G.dPanels + (size_t)(j / g_size) * Np * NB + r0 * NB;
        for (size_t c = 0; c < nch; ++c) {
            CUDA_CHECK(cudaMemcpyAsync(hb + c * CHUNK_ROWS * NB, dp + c * CHUNK_ROWS * NB,
                                       chunkRows(c) * NB * sizeof(double), cudaMemcpyDeviceToHost, G.sLook));
            CUDA_CHECK(cudaEventRecord(G.evChunk[c], G.sLook));
        }
        long st = -1;
        for (size_t c = 0; c < nch; ++c) {
            double* hc = hb + c * CHUNK_ROWS * NB;
            CUDA_CHECK(cudaEventSynchronize(G.evChunk[c]));
            if (c == 0) st = factorDiagBlock(hb, G.LT, w, r0);
            if (st < 0) {
                const size_t i0 = std::max<size_t>(w, c * CHUNK_ROWS);
                const size_t i1 = std::min(mValid, c * CHUNK_ROWS + chunkRows(c));
                if (i0 < i1) solvePanelRows(hb, G.LT, w, i0, i1);
                CUDA_CHECK(cudaMemcpyAsync(dp + c * CHUNK_ROWS * NB, hc, chunkRows(c) * NB * sizeof(double),
                                           cudaMemcpyHostToDevice, G.sLook));
            }
            if (c == nch - 1) hb[m * NB] = (double)st;
            if (forward) {
                reqs.emplace_back();
                MPI_Isend(hc, chunkCount(c), MPI_DOUBLE, next, (int)c, MPI_COMM_WORLD, &reqs.back());
            }
            collect(c);
        }
    } else {
        double* dl = G.dL[j % 2] + r0 * NB;
        for (size_t c = 0; c < nch; ++c) {
            double* hc = hb + c * CHUNK_ROWS * NB;
            MPI_Recv(hc, chunkCount(c), MPI_DOUBLE, prev, (int)c, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (forward) {
                reqs.emplace_back();
                MPI_Isend(hc, chunkCount(c), MPI_DOUBLE, next, (int)c, MPI_COMM_WORLD, &reqs.back());
            }
            CUDA_CHECK(cudaMemcpyAsync(dl + c * CHUNK_ROWS * NB, hc, chunkRows(c) * NB * sizeof(double),
                                       cudaMemcpyHostToDevice, G.sCopy));
            collect(c);
        }
        CUDA_CHECK(cudaEventRecord(G.evCopy, G.sCopy));
    }
    MPI_Waitall((int)reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);
}

bool choleskyDecomposition(std::vector<double>& A, DistMatrix& D, GpuCtx& G) {
    const int NT = D.NT;
    const size_t Np = D.Np;
    const size_t panelElems = Np * NB;
    if (NT == 0) return true;

    // Upload local panels
    if (D.nloc > 0)
        CUDA_CHECK(cudaMemcpyAsync(G.dPanels, D.hLocal, D.nloc * panelElems * sizeof(double),
                                   cudaMemcpyHostToDevice, G.sLook));

    panelPipeline(0, D, G, A);

    bool ok = true;
    for (int k = 0; k < NT; ++k) {
        double* hk = G.hbuf[k % 2];
        const size_t r0 = (size_t)k * NB;
        const size_t mk = Np - r0;
        const long st = (long)hk[mk * NB];
        if (st >= 0) {
            if (g_rank == 0) printf("Error: Matrix is not positive definite at diagonal element %ld\n", st);
            ok = false;
            break;
        }

        // Device copy of panel k
        const double* dLk;
        if (ownerOf(k) == g_rank) {
            dLk = G.dPanels + (size_t)(k / g_size) * panelElems;
        } else {
            dLk = G.dL[k % 2];
            CUDA_CHECK(cudaStreamWaitEvent(G.sMain, G.evCopy, 0));
            CUDA_CHECK(cudaStreamWaitEvent(G.sLook, G.evCopy, 0));
        }

        // Local panels with global index > k
        const int lfirst = (k + 1 - g_rank + g_size - 1) / g_size;  // first l with l*P+rank > k
        const int cnt = D.nloc - lfirst;
        if (k + 1 < NT) {
            if (ownerOf(k + 1) == g_rank) {
                // lookahead: update panel k+1 first (high-priority stream), then factor it
                // on the CPU while the GPU updates the remaining trailing matrix
                launchUpdate(dLk, G.dPanels, lfirst, 1, k, D, G.sLook);
                launchUpdate(dLk, G.dPanels, lfirst + 1, cnt - 1, k, D, G.sMain);
            } else {
                launchUpdate(dLk, G.dPanels, lfirst, cnt, k, D, G.sMain);
            }
            panelPipeline(k + 1, D, G, A);
        }
        CUDA_CHECK(cudaStreamSynchronize(G.sLook));
        CUDA_CHECK(cudaStreamSynchronize(G.sMain));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    return ok;
}

static void createGpuCtx(GpuCtx& G, const DistMatrix& D) {
    const size_t panelElems = D.Np * NB;
    CUDA_CHECK(cudaMalloc(&G.dPanels, std::max<size_t>(1, D.nloc * panelElems) * sizeof(double)));
    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaMalloc(&G.dL[b], std::max<size_t>(1, panelElems) * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&G.hbuf[b], (panelElems + 1) * sizeof(double)));
    }
    int prLow = 0, prHigh = 0;
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&prLow, &prHigh));
    CUDA_CHECK(cudaStreamCreateWithPriority(&G.sMain, cudaStreamNonBlocking, prLow));
    // lookahead (critical path) work gets the highest priority
    CUDA_CHECK(cudaStreamCreateWithPriority(&G.sLook, cudaStreamNonBlocking, prHigh));
    CUDA_CHECK(cudaStreamCreateWithPriority(&G.sCopy, cudaStreamNonBlocking, prHigh));
    CUDA_CHECK(cudaEventCreateWithFlags(&G.evCopy, cudaEventDisableTiming));
    G.evChunk.resize(D.Np / CHUNK_ROWS + 2);
    for (auto& e : G.evChunk) CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming | cudaEventBlockingSync));
}

static void destroyGpuCtx(GpuCtx& G) {
    for (auto& e : G.evChunk) cudaEventDestroy(e);
    cudaEventDestroy(G.evCopy);
    cudaStreamDestroy(G.sMain);
    cudaStreamDestroy(G.sLook);
    cudaStreamDestroy(G.sCopy);
    for (int b = 0; b < 2; ++b) {
        cudaFree(G.dL[b]);
        cudaFreeHost(G.hbuf[b]);
    }
    cudaFree(G.dPanels);
}

// out[i*n+j] = sum_k X[i*n+k] * X[j*n+k] for all i, j (symmetric), GPU version.
// Each element is accumulated sequentially in k (same order as the reference loop).
// With lowerTriangularX, terms with k > min(i, j) (exact zeros) are skipped.
__global__ void __launch_bounds__(256)
symmetricProductKernel(const double* __restrict__ X, double* __restrict__ out, int n, bool lowerTriangularX) {
    const int ti = blockIdx.y, tj = blockIdx.x;
    if (tj > ti) return;
    const int i0 = ti * TM, j0 = tj * TN;
    const int kmax = lowerTriangularX ? min(n, j0 + TN) : n;

    __shared__ double As[KC][TM + 1];
    __shared__ double Bs[KC][TN + 1];
    const int tid = threadIdx.x;
    const int tx = tid % 16, ty = tid / 16;
    double acc[4][4];
#pragma unroll
    for (int a = 0; a < 4; ++a)
#pragma unroll
        for (int b = 0; b < 4; ++b) acc[a][b] = 0.0;

    for (int k0 = 0; k0 < kmax; k0 += KC) {
        __syncthreads();
        for (int e = tid; e < KC * TM; e += 256) {
            const int r = e / KC, kk = e % KC;
            const int k = k0 + kk;
            As[kk][r] = (i0 + r < n && k < kmax) ? X[(size_t)(i0 + r) * n + k] : 0.0;
            Bs[kk][r] = (j0 + r < n && k < kmax) ? X[(size_t)(j0 + r) * n + k] : 0.0;
        }
        __syncthreads();
        const int kc = min(KC, kmax - k0);
        for (int kk = 0; kk < kc; ++kk) {
            double av[4], bv[4];
#pragma unroll
            for (int a = 0; a < 4; ++a) av[a] = As[kk][ty + 16 * a];
#pragma unroll
            for (int b = 0; b < 4; ++b) bv[b] = Bs[kk][tx + 16 * b];
#pragma unroll
            for (int a = 0; a < 4; ++a)
#pragma unroll
                for (int b = 0; b < 4; ++b) acc[a][b] = __dadd_rn(acc[a][b], __dmul_rn(av[a], bv[b]));  // unfused, as the reference
        }
    }
#pragma unroll
    for (int a = 0; a < 4; ++a)
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            const int i = i0 + ty + 16 * a, j = j0 + tx + 16 * b;
            if (i < n && j < n) {
                out[(size_t)i * n + j] = acc[a][b];
                out[(size_t)j * n + i] = acc[a][b];
            }
        }
}

// CPU fallback of symmetricProductKernel (used if the GPU lacks memory)
static void symmetricProductCPU(const double* X, double* out, size_t n, bool lowerTriangularX) {
    const size_t nb4 = (n + 3) / 4;
#pragma omp parallel for schedule(dynamic, 1)
    for (size_t bi = 0; bi < nb4; ++bi) {
        const size_t i0 = bi * 4;
        const size_t ni = std::min<size_t>(4, n - i0);
        for (size_t j0 = 0; j0 <= i0; j0 += 4) {
            const size_t nj = std::min<size_t>(4, n - j0);
            double acc[4][4] = {};
            size_t kmax = n;
            if (lowerTriangularX) kmax = std::min(n, j0 + nj);  // L[j][k] == 0 for k > j
            const double* xi[4];
            const double* xj[4];
            for (size_t a = 0; a < 4; ++a) xi[a] = X + std::min(i0 + a, n - 1) * n;
            for (size_t b = 0; b < 4; ++b) xj[b] = X + std::min(j0 + b, n - 1) * n;
            for (size_t k = 0; k < kmax; ++k) {
                for (int a = 0; a < 4; ++a)
                    for (int b = 0; b < 4; ++b) acc[a][b] += xi[a][k] * xj[b][k];
            }
            for (size_t a = 0; a < ni; ++a)
                for (size_t b = 0; b < nj; ++b) {
                    out[(i0 + a) * n + j0 + b] = acc[a][b];
                    out[(j0 + b) * n + i0 + a] = acc[a][b];
                }
        }
    }
}

static void symmetricProduct(const double* X, double* out, size_t n, bool lowerTriangularX) {
    if (n == 0) return;
    const size_t bytes = n * n * sizeof(double);
    double *dX = nullptr, *dOut = nullptr;
    if (cudaMalloc(&dX, bytes) != cudaSuccess || cudaMalloc(&dOut, bytes) != cudaSuccess) {
        cudaGetLastError();
        if (dX) cudaFree(dX);
        symmetricProductCPU(X, out, n, lowerTriangularX);
        return;
    }
    CUDA_CHECK(cudaMemcpy(dX, X, bytes, cudaMemcpyHostToDevice));
    const unsigned tiles = (unsigned)((n + TM - 1) / TM);
    symmetricProductKernel<<<dim3(tiles, tiles), 256>>>(dX, dOut, (int)n, lowerTriangularX);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(out, dOut, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dX));
    CUDA_CHECK(cudaFree(dOut));
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

    // Compute A = B * B^T
    symmetricProduct(B.data(), A.data(), n, false);

    // Add diagonal dominance to ensure positive definiteness
    for (size_t i = 0; i < n; ++i) {
        A[i * n + i] += n;
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix

    std::vector<double> reconstructed(n * n);

    // Compute L * L^T
    symmetricProduct(L.data(), reconstructed.data(), n, true);

    // Compare with original
    double maxError = 0.0;
    double relError = 0.0;

#pragma omp parallel for reduction(max : maxError, relError)
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

// If the launcher did not bind this rank, restrict it to the CPUs local to its GPU
// (avoids cross-socket traffic for the pinned panel transfers)
static void bindToGpuLocalCpus(int dev) {
    cpu_set_t cur;
    CPU_ZERO(&cur);
    if (sched_getaffinity(0, sizeof(cur), &cur) != 0) return;
    if (CPU_COUNT(&cur) < sysconf(_SC_NPROCESSORS_ONLN)) return;  // already bound
    char bus[32] = {};
    if (cudaDeviceGetPCIBusId(bus, sizeof(bus), dev) != cudaSuccess) return;
    for (char* c = bus; *c; ++c) *c = (char)tolower(*c);
    char path[128];
    snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/local_cpulist", bus);
    FILE* f = fopen(path, "r");
    if (!f) return;
    char list[1024] = {};
    const bool okRead = fgets(list, sizeof(list), f) != nullptr;
    fclose(f);
    if (!okRead) return;
    cpu_set_t set;
    CPU_ZERO(&set);
    for (char* tok = strtok(list, ",\n"); tok; tok = strtok(nullptr, ",\n")) {
        int lo = 0, hi = 0;
        const int cnt = sscanf(tok, "%d-%d", &lo, &hi);
        if (cnt < 1) continue;
        if (cnt == 1) hi = lo;
        for (int c = lo; c <= hi && c < CPU_SETSIZE; ++c) CPU_SET(c, &set);
    }
    if (CPU_COUNT(&set) > 0) sched_setaffinity(0, sizeof(set), &set);
}

static int finish(int code) {
    MPI_Finalize();
    return code;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_size);

    // Node-local rank: select GPU and share CPU cores among co-located ranks
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, g_rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0, localSize = 1;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_size(nodeComm, &localSize);
    MPI_Comm_free(&nodeComm);

    int ngpu = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ngpu));
    if (ngpu < 1) {
        fprintf(stderr, "No CUDA device found\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int dev = localRank % ngpu;
    bindToGpuLocalCpus(dev);
    CUDA_CHECK(cudaSetDevice(dev));
    CUDA_CHECK(cudaFree(0));  // initialize context outside the timed region

    if (!getenv("OMP_NUM_THREADS")) {
        const long hw = sysconf(_SC_NPROCESSORS_ONLN);
        // share the node's cores among co-located ranks
        const int np = (int)std::max<long>(1, std::min<long>(omp_get_num_procs(), hw / localSize));
        omp_set_num_threads(std::max(1, np));
    }

#pragma omp parallel
    { (void)omp_get_thread_num(); }  // start the thread pool outside the timed region

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
            if (g_rank == 0) printUsage(argv[0]);
            return finish(0);
        } else {
            if (g_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return finish(1);
        }
    }

    if (g_rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Allocate matrix (full matrix lives on rank 0)
    std::vector<double> A;
    std::vector<double> A_orig;

    // Generate positive definite matrix
    if (g_rank == 0) {
        A.resize(n * n);
        printf("Generating positive definite matrix...\n");
        fflush(stdout);
        generatePositiveDefiniteMatrix(A, n);
        if (validate) {
            A_orig = A; // Save original for validation
        }
    }

    // Distribute column panels to their owners
    DistMatrix D;
    D.n = n;
    D.NT = (int)((n + NB - 1) / NB);
    D.Np = (size_t)D.NT * NB;
    D.nloc = (D.NT > g_rank) ? (D.NT - g_rank + g_size - 1) / g_size : 0;
    distributeMatrix(A, D);
    static GpuCtx G;  // device buffers, streams and pinned buffers (setup, not timed)
    createGpuCtx(G, D);

    // Perform Cholesky decomposition
    if (g_rank == 0) {
        printf("Computing Cholesky decomposition...\n");
        fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    bool success = choleskyDecomposition(A, D, G);
    MPI_Barrier(MPI_COMM_WORLD);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    destroyGpuCtx(G);
    cudaFreeHost(D.hLocal);

    if (!success) {
        if (g_rank == 0) printf("Cholesky decomposition failed\n");
        return finish(1);
    }

    int rc = 0;
    if (g_rank == 0) {
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
                rc = 1;
            }
        }
    }
    MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return finish(rc);
}
