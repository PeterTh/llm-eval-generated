#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

// Hybrid MPI + OpenMP + CUDA Cholesky decomposition (A = L * L^T, L lower triangular).
//
// Parallel layout
//   * MPI: the (zero padded) matrix is distributed in a 1D block-cyclic fashion over
//     block columns of width `nb`. Rank r owns global block columns r, r+P, r+2P, ...
//     and keeps them on its own GPU in row-major local storage (np rows x nloc
//     columns). Right-looking blocked algorithm with look-ahead: the owner of the next
//     panel updates and factors it first and publishes it while everybody else is busy
//     with the bulk trailing update. Panels are published in row chunks through a
//     page-locked MPI-3 shared memory window per node (node mates copy them straight
//     into their GPUs); node leaders forward the chunks to other nodes with MPI_Ibcast.
//   * CUDA: all O(n^3) work (trailing updates, look-ahead updates, triangular solves,
//     matrix generation and validation products) runs in hand-written double precision
//     kernels (no external BLAS dependency) on three streams of different priority so
//     the look-ahead chain preempts the bulk update.
//   * OpenMP: the nb x nb diagonal block of each panel is factored on the host
//     (together with the inverse of its triangular factor, applied on the GPU as one
//     wide GEMM) while the GPU keeps computing; host-side packing / unpacking and the
//     validation error reduction are multithreaded as well.
//
// Every arithmetic step corresponds to the mathematical operations of the sequential
// reference algorithm; only the order of floating point accumulation differs.

#define CUDA_CHECK(call)                                                                   \
    do {                                                                                   \
        cudaError_t err__ = (call);                                                        \
        if (err__ != cudaSuccess) {                                                        \
            fprintf(stderr, "CUDA error '%s' at %s:%d\n", cudaGetErrorString(err__),       \
                    __FILE__, __LINE__);                                                   \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                  \
        }                                                                                  \
    } while (0)

namespace {

constexpr int TILE = 64;          // GEMM tile edge and inner blocking of the host factorization
constexpr int BK = 16;            // GEMM k-step
constexpr int GEMM_THREADS = 256; // 16 x 16 threads, 4 x 4 micro tile each

// ------------------------------------------------------------------------------------
// Distribution helpers
// ------------------------------------------------------------------------------------

// Global column index of local column c (1D block-cyclic distribution over block columns).
__host__ __device__ __forceinline__ int globalCol(int c, int nb, int P, int rank) {
    return ((c / nb) * P + rank) * nb + (c % nb);
}

struct Layout {
    size_t n;    // logical matrix size
    int np;      // padded matrix size (multiple of nb)
    int nb;      // block (panel) width
    int NB;      // number of block columns
    int P;       // number of MPI ranks
    int rank;    // this rank
    int nloc;    // number of local columns on this rank

    int ownerOf(int k) const { return k % P; }
    // Local column offset of global block column k (only valid if this rank owns it).
    int localColOf(int k) const { return (k / P) * nb; }
    // First local column whose global block column index is > k.
    int firstLocalColAfter(int k) const {
        const int cnt = (k >= rank) ? (k - rank) / P + 1 : 0;
        return cnt * nb;
    }
    static int localColsOf(int r, int NB, int P, int nb) {
        const int cnt = (NB > r) ? (NB - 1 - r) / P + 1 : 0;
        return cnt * nb;
    }
};

// ------------------------------------------------------------------------------------
// CUDA kernels
// ------------------------------------------------------------------------------------

// C[M x N] (row-major, ldc)  (-)=  A[M x K] (row-major, lda) * B[N x K]^T (row-major, ldb)
// M, N multiples of TILE, K multiple of BK. Every thread accumulates its output
// elements sequentially in k order (fma), matching the reference accumulation order.
// If SKIP is set, tiles that lie entirely above the diagonal of the global matrix are
// skipped (rowOffset: global row of C row 0; cLocal0: local column of C column 0).
// If TRI is set, B is lower triangular (N == K): zero tiles of B are not visited.
template <bool SUB, bool SKIP, bool TRI>
__global__ void __launch_bounds__(GEMM_THREADS)
gemm_nt_kernel(const double* __restrict__ A, size_t lda,
               const double* __restrict__ B, size_t ldb,
               double* __restrict__ C, size_t ldc, int K,
               int rowOffset, int cLocal0, int nb, int P, int rank) {
    const int tn = blockIdx.x;
    const int tm = blockIdx.y;
    if (SKIP) {
        const int gcol = globalCol(cLocal0 + tn * TILE, nb, P, rank);
        if (rowOffset + tm * TILE + TILE - 1 < gcol) return;
    }

    __shared__ double As[BK][TILE + 1];
    __shared__ double Bs[BK][TILE + 1];

    const int tid = threadIdx.x;
    const int tx = tid & 15;
    const int ty = tid >> 4;

    const double* Ab = A + (size_t)tm * TILE * lda;
    const double* Bb = B + (size_t)tn * TILE * ldb;

    double acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = 0.0;

    const int Kend = TRI ? min(K, (tn + 1) * TILE) : K;
    for (int k0 = 0; k0 < Kend; k0 += BK) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int idx = tid + i * GEMM_THREADS;
            const int r = idx >> 4;
            const int kk = idx & 15;
            As[kk][r] = Ab[(size_t)r * lda + k0 + kk];
            Bs[kk][r] = Bb[(size_t)r * ldb + k0 + kk];
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            double a[4], b[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                a[i] = As[kk][ty + 16 * i];
                b[i] = Bs[kk][tx + 16 * i];
            }
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] = fma(a[i], b[j], acc[i][j]);
        }
        __syncthreads();
    }

    double* Cb = C + (size_t)tm * TILE * ldc + (size_t)tn * TILE;
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const size_t idx = (size_t)(ty + 16 * i) * ldc + tx + 16 * j;
            if (SUB) Cb[idx] -= acc[i][j];
            else     Cb[idx] = acc[i][j];
        }
}

// dst[r][0..W) = src[globalCol(cStart + r) - rowOff][0..W)   for r < R
__global__ void gather_rows_kernel(const double* __restrict__ src, size_t lds,
                                   double* __restrict__ dst, size_t ldd,
                                   int R, int W, int cStart, int nb, int P, int rank, int rowOff) {
    const int r = blockIdx.y;
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= R || p >= W) return;
    const int srcRow = globalCol(cStart + r, nb, P, rank) - rowOff;
    dst[(size_t)r * ldd + p] = src[(size_t)srcRow * lds + p];
}

// Add the diagonal shift: A[i][i] += n for i < n, padded diagonal entries become 1.
__global__ void add_diagonal_kernel(double* __restrict__ Aloc, int nloc, int nb, int P, int rank,
                                    int n) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nloc) return;
    const int i = globalCol(c, nb, P, rank);
    Aloc[(size_t)i * nloc + c] += (i < n) ? (double)n : 1.0;
}

// ------------------------------------------------------------------------------------
// Host side wrappers
// ------------------------------------------------------------------------------------

template <bool SUB, bool SKIP, bool TRI = false>
void gemmNT(cudaStream_t stream, const double* A, size_t lda, const double* B, size_t ldb,
            double* C, size_t ldc, int M, int N, int K, int rowOffset, int cLocal0,
            const Layout& L) {
    if (M <= 0 || N <= 0 || K <= 0) return;
    dim3 grid(N / TILE, M / TILE);
    gemm_nt_kernel<SUB, SKIP, TRI><<<grid, GEMM_THREADS, 0, stream>>>(
        A, lda, B, ldb, C, ldc, K, rowOffset, cLocal0, L.nb, L.P, L.rank);
    CUDA_CHECK(cudaGetLastError());
}

void gatherRows(cudaStream_t stream, const double* src, size_t lds, double* dst, size_t ldd,
                int R, int W, int cStart, int rowOff, const Layout& L) {
    if (R <= 0 || W <= 0) return;
    dim3 grid((W + 255) / 256, R);
    gather_rows_kernel<<<grid, 256, 0, stream>>>(src, lds, dst, ldd, R, W, cStart, L.nb, L.P,
                                                 L.rank, rowOff);
    CUDA_CHECK(cudaGetLastError());
}

constexpr int PANEL_CHUNKS = 4;  // row chunks per panel message (pipelines copies and sends)

struct DeviceState {
    double* Aloc = nullptr;          // np x nloc, row-major (ld = nloc)
    double* panelRecv[2] = {nullptr, nullptr};  // received panel rows, (np - nb) x nb, ld = nb
    double* sendBuf = nullptr;       // packed panel rows of this rank's own panel
    double* Bg[2] = {nullptr, nullptr};         // gathered panel rows for the local columns
    double* Tinv = nullptr;          // inverse of the current diagonal block factor, nb x nb
    double* hostDiag = nullptr;      // pinned nb x nb diagonal block (host factorization)
    double* hostTinv = nullptr;      // pinned nb x nb inverse of its factor

    // hi: look-ahead chain (panel transfers, next panel update + factorization)
    // mid: update of the second pending local block column
    // lo: bulk trailing update
    cudaStream_t hi = nullptr, mid = nullptr, lo = nullptr;
    cudaEvent_t evLow[2] = {}, evMid[2] = {}, evGather[2] = {}, evH2D[2] = {};
    cudaEvent_t evSend[PANEL_CHUNKS] = {};
    cudaEvent_t evDiag = nullptr;
};

void createDeviceState(const Layout& L, DeviceState& d) {
    const size_t locElems = (size_t)L.np * L.nloc;
    const size_t panelElems = std::max<size_t>((size_t)std::max(L.np - L.nb, 0) * L.nb, 1);
    const size_t bgElems = std::max<size_t>((size_t)L.nloc * L.nb, 1);
    CUDA_CHECK(cudaMalloc(&d.Aloc, std::max<size_t>(locElems, 1) * sizeof(double)));
    for (int i = 0; i < 2; ++i) {
        CUDA_CHECK(cudaMalloc(&d.panelRecv[i], panelElems * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d.Bg[i], bgElems * sizeof(double)));
    }
    CUDA_CHECK(cudaMalloc(&d.sendBuf, panelElems * sizeof(double)));
    const size_t diagElems = (size_t)L.nb * L.nb;
    CUDA_CHECK(cudaMalloc(&d.Tinv, diagElems * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&d.hostDiag, diagElems * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&d.hostTinv, diagElems * sizeof(double)));

    int least = 0, greatest = 0;
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&least, &greatest));
    const int midPrio = (greatest < least) ? std::min(greatest + 1, least) : greatest;
    CUDA_CHECK(cudaStreamCreateWithPriority(&d.hi, cudaStreamNonBlocking, greatest));
    CUDA_CHECK(cudaStreamCreateWithPriority(&d.mid, cudaStreamNonBlocking, midPrio));
    CUDA_CHECK(cudaStreamCreateWithPriority(&d.lo, cudaStreamNonBlocking, least));
    for (int i = 0; i < 2; ++i) {
        CUDA_CHECK(cudaEventCreateWithFlags(&d.evLow[i], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&d.evMid[i], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&d.evGather[i], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&d.evH2D[i], cudaEventDisableTiming));
        // Record once so that waiting on a never-used event is a no-op.
        CUDA_CHECK(cudaEventRecord(d.evLow[i], d.lo));
        CUDA_CHECK(cudaEventRecord(d.evMid[i], d.mid));
        CUDA_CHECK(cudaEventRecord(d.evGather[i], d.hi));
        CUDA_CHECK(cudaEventRecord(d.evH2D[i], d.hi));
    }
    for (int i = 0; i < PANEL_CHUNKS; ++i) {
        CUDA_CHECK(cudaEventCreateWithFlags(&d.evSend[i], cudaEventDisableTiming));
    }
    CUDA_CHECK(cudaEventCreateWithFlags(&d.evDiag, cudaEventDisableTiming));
}

void destroyDeviceState(DeviceState& d) {
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d.Aloc));
    for (int i = 0; i < 2; ++i) {
        CUDA_CHECK(cudaFree(d.panelRecv[i]));
        CUDA_CHECK(cudaFree(d.Bg[i]));
        CUDA_CHECK(cudaEventDestroy(d.evLow[i]));
        CUDA_CHECK(cudaEventDestroy(d.evMid[i]));
        CUDA_CHECK(cudaEventDestroy(d.evGather[i]));
        CUDA_CHECK(cudaEventDestroy(d.evH2D[i]));
    }
    for (int i = 0; i < PANEL_CHUNKS; ++i) CUDA_CHECK(cudaEventDestroy(d.evSend[i]));
    CUDA_CHECK(cudaEventDestroy(d.evDiag));
    CUDA_CHECK(cudaFree(d.sendBuf));
    CUDA_CHECK(cudaFree(d.Tinv));
    CUDA_CHECK(cudaFreeHost(d.hostDiag));
    CUDA_CHECK(cudaFreeHost(d.hostTinv));
    CUDA_CHECK(cudaStreamDestroy(d.hi));
    CUDA_CHECK(cudaStreamDestroy(d.mid));
    CUDA_CHECK(cudaStreamDestroy(d.lo));
    d = DeviceState();
}

// Wait for a CUDA event on the host while keeping the MPI progress engine running
// (this rank may still have transfers in flight that other ranks depend on).
void waitEventWithProgress(cudaEvent_t ev, MPI_Comm comm) {
    for (;;) {
        const cudaError_t st = cudaEventQuery(ev);
        if (st == cudaSuccess) return;
        if (st != cudaErrorNotReady) CUDA_CHECK(st);
        int flag = 0;
        MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, comm, &flag, MPI_STATUS_IGNORE);
    }
}

// Host Cholesky factorization (OpenMP, blocked right-looking) of an nb x nb row-major
// block. On success the lower triangle holds L and the strict upper triangle is zeroed.
// Returns the index of the first non-positive pivot, or -1.
constexpr int HOST_FACTOR_THREADS = 16;  // enough for an nb x nb block, keeps overhead low

int hostCholesky(double* A, int nb) {
    constexpr int IB = TILE;
    const int nt = std::min(omp_get_max_threads(), HOST_FACTOR_THREADS);
    std::vector<double> Pt((size_t)IB * nb);  // transposed current column slab
    for (int jb = 0; jb < nb; jb += IB) {
        const int w = std::min(IB, nb - jb);
        // Unblocked factorization of the diagonal block (same recurrences as the
        // reference algorithm).
        for (int i = jb; i < jb + w; ++i) {
            for (int j = jb; j <= i; ++j) {
                const double* ai = A + (size_t)i * nb;
                const double* aj = A + (size_t)j * nb;
                double sum = 0.0;
#pragma omp simd reduction(+ : sum)
                for (int k = jb; k < j; ++k) sum += ai[k] * aj[k];
                if (i == j) {
                    const double val = A[(size_t)j * nb + j] - sum;
                    if (val <= 0.0) return j;
                    A[(size_t)j * nb + j] = sqrt(val);
                } else {
                    A[(size_t)i * nb + j] = (A[(size_t)i * nb + j] - sum) / A[(size_t)j * nb + j];
                }
            }
        }
        const int rem = nb - jb - w;
        if (rem == 0) break;
        // Rows below the diagonal block: X := X * L^{-T}, then transpose the slab.
#pragma omp parallel for schedule(static) num_threads(nt)
        for (int i = jb + w; i < nb; ++i) {
            double* x = A + (size_t)i * nb + jb;
            for (int p = 0; p < w; ++p) {
                const double* lp = A + (size_t)(jb + p) * nb + jb;
                double v = 0.0;
#pragma omp simd reduction(+ : v)
                for (int q = 0; q < p; ++q) v += x[q] * lp[q];
                x[p] = (x[p] - v) / lp[p];
            }
            for (int p = 0; p < w; ++p) Pt[(size_t)p * nb + i] = x[p];
        }
        // Trailing update of the lower triangle: A[i][m] -= sum_p X[i][p] * X[m][p].
#pragma omp parallel for schedule(dynamic, 8) num_threads(nt)
        for (int i = jb + w; i < nb; ++i) {
            double* row = A + (size_t)i * nb;
            for (int p = 0; p < w; ++p) {
                const double a = Pt[(size_t)p * nb + i];
                const double* pr = Pt.data() + (size_t)p * nb;
#pragma omp simd
                for (int m = jb + w; m <= i; ++m) row[m] -= a * pr[m];
            }
        }
    }
#pragma omp parallel for schedule(static) num_threads(nt)
    for (int i = 0; i < nb; ++i) {
        for (int m = i + 1; m < nb; ++m) A[(size_t)i * nb + m] = 0.0;
    }
    return -1;
}

// T := L^{-1} for a lower triangular nb x nb row-major L (T row-major, upper part zero).
void hostTriangularInverse(const double* Lm, double* T, int nb) {
    const int nt = std::min(omp_get_max_threads(), HOST_FACTOR_THREADS);
#pragma omp parallel num_threads(nt)
    {
        std::vector<double> col((size_t)nb);
#pragma omp for schedule(dynamic, 4)
        for (int j = 0; j < nb; ++j) {
            for (int i = 0; i < j; ++i) T[(size_t)i * nb + j] = 0.0;
            col[j] = 1.0 / Lm[(size_t)j * nb + j];
            for (int i = j + 1; i < nb; ++i) {
                const double* li = Lm + (size_t)i * nb;
                double sum = 0.0;
#pragma omp simd reduction(+ : sum)
                for (int k = j; k < i; ++k) sum += li[k] * col[k];
                col[i] = -sum / li[i];
            }
            for (int i = j; i < nb; ++i) T[(size_t)i * nb + j] = col[i];
        }
    }
}

// Panel factorization of global block column k (owned by this rank), in three parts:
//  1. enqueueDiagCopy: copy the (updated) diagonal block to the host (stream order).
//  2. hostFactorDiag: factor it on the host (OpenMP) together with the inverse of its
//     factor and upload both. Returns the global index of the first non-positive pivot
//     or -1 on success.
//  3. the rows below are formed chunk by chunk as X * L^{-T} with a GEMM while being
//     published (see sendPanel).
void enqueueDiagCopy(const Layout& L, DeviceState& d, int k, cudaStream_t s) {
    const size_t ldl = (size_t)L.nloc;
    const double* diag = d.Aloc + (size_t)k * L.nb * ldl + L.localColOf(k);
    const size_t rowBytes = (size_t)L.nb * sizeof(double);
    CUDA_CHECK(cudaMemcpy2DAsync(d.hostDiag, rowBytes, diag, ldl * sizeof(double), rowBytes,
                                 (size_t)L.nb, cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaEventRecord(d.evDiag, s));
}

int hostFactorDiag(const Layout& L, DeviceState& d, int k, cudaStream_t s, MPI_Comm progressComm) {
    const size_t ldl = (size_t)L.nloc;
    double* diag = d.Aloc + (size_t)k * L.nb * ldl + L.localColOf(k);
    const size_t rowBytes = (size_t)L.nb * sizeof(double);

    waitEventWithProgress(d.evDiag, progressComm);
    const int fail = hostCholesky(d.hostDiag, L.nb);
    if (fail >= 0) return k * L.nb + fail;
    hostTriangularInverse(d.hostDiag, d.hostTinv, L.nb);

    CUDA_CHECK(cudaMemcpy2DAsync(diag, ldl * sizeof(double), d.hostDiag, rowBytes, rowBytes,
                                 (size_t)L.nb, cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(d.Tinv, d.hostTinv, (size_t)L.nb * L.nb * sizeof(double),
                               cudaMemcpyHostToDevice, s));
    return -1;
}

// Row range [begin, end) of chunk i of a panel with Mrem rows below the diagonal block.
// Boundaries are multiples of TILE (Mrem itself is a multiple of nb, hence of TILE).
inline void chunkRange(int Mrem, int i, int& begin, int& end) {
    const long long tiles = Mrem / TILE;
    begin = (int)(tiles * i / PANEL_CHUNKS) * TILE;
    end = (int)(tiles * (i + 1) / PANEL_CHUNKS) * TILE;
}

// Panel distribution. Every node keeps two (double buffered by panel parity) panel
// buffers in an MPI-3 shared memory window that is page-locked for fast GPU copies.
// The owner of a panel copies it straight from its GPU into its node's buffer; ranks
// on the same node copy it straight into their GPUs (no MPI data copies at all).
// Between nodes the chunks are forwarded by the node leaders with MPI_Ibcast and then
// published to their node mates in the same way. Tiny point-to-point messages signal
// chunk availability inside a node; a non-blocking node barrier guards buffer reuse.
struct PanelComm {
    MPI_Comm world = MPI_COMM_NULL;
    MPI_Comm node = MPI_COMM_NULL;
    MPI_Comm leaders = MPI_COMM_NULL;  // only valid on node leaders (node rank 0)
    int nodeRank = 0, nodeSize = 1;
    int numNodes = 1;
    std::vector<int> nodeOfRank;        // node index of every world rank
    std::vector<int> nodeRankOfRank;    // node-local rank of every world rank
    int myNode = 0;

    MPI_Win win = MPI_WIN_NULL;
    double* shm[2] = {nullptr, nullptr};
    size_t panelMax = 0;
    bool registered = false;

    std::vector<MPI_Request> readyRecv[2];   // intra-node "chunk ready" notifications
    std::vector<MPI_Request> readySend[2];
    std::vector<MPI_Request> bcastReq[2];    // inter-node chunk broadcasts (leaders)
    MPI_Request barrier[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};
};

void createPanelComm(const Layout& L, MPI_Comm comm, size_t panelMax, PanelComm& pc) {
    pc.world = comm;
    pc.panelMax = panelMax;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, L.rank, MPI_INFO_NULL, &pc.node);
    MPI_Comm_rank(pc.node, &pc.nodeRank);
    MPI_Comm_size(pc.node, &pc.nodeSize);

    // Node id = world rank of the node leader; collect the map on all ranks.
    int leaderRank = L.rank;
    MPI_Bcast(&leaderRank, 1, MPI_INT, 0, pc.node);
    std::vector<int> leaderOf(L.P), nodeRankOf(L.P);
    MPI_Allgather(&leaderRank, 1, MPI_INT, leaderOf.data(), 1, MPI_INT, comm);
    MPI_Allgather(&pc.nodeRank, 1, MPI_INT, nodeRankOf.data(), 1, MPI_INT, comm);
    std::vector<int> leaders;
    for (int r = 0; r < L.P; ++r) {
        if (std::find(leaders.begin(), leaders.end(), leaderOf[r]) == leaders.end()) {
            leaders.push_back(leaderOf[r]);
        }
    }
    pc.numNodes = (int)leaders.size();
    pc.nodeOfRank.resize(L.P);
    pc.nodeRankOfRank = nodeRankOf;
    for (int r = 0; r < L.P; ++r) {
        pc.nodeOfRank[r] = (int)(std::find(leaders.begin(), leaders.end(), leaderOf[r]) - leaders.begin());
    }
    pc.myNode = pc.nodeOfRank[L.rank];
    MPI_Comm_split(comm, pc.nodeRank == 0 ? 0 : MPI_UNDEFINED, pc.myNode, &pc.leaders);
    // The leaders communicator is ordered by node index, so node index == leader rank.

    const MPI_Aint bytes = (MPI_Aint)(2 * panelMax * sizeof(double));
    double* base = nullptr;
    MPI_Win_allocate_shared(pc.nodeRank == 0 ? bytes : 0, sizeof(double), MPI_INFO_NULL, pc.node,
                            &base, &pc.win);
    MPI_Aint qsize = 0;
    int qdisp = 0;
    MPI_Win_shared_query(pc.win, 0, &qsize, &qdisp, &base);
    pc.shm[0] = base;
    pc.shm[1] = base + panelMax;
    // Page-lock the shared buffers for asynchronous, full speed GPU copies. If the
    // driver refuses (unusual mapping), plain pageable copies are used instead.
    cudaError_t err = cudaHostRegister(base, (size_t)bytes, cudaHostRegisterDefault);
    pc.registered = (err == cudaSuccess);
    if (!pc.registered) (void)cudaGetLastError();
    MPI_Barrier(pc.node);
}

void destroyPanelComm(PanelComm& pc) {
    for (int p = 0; p < 2; ++p) {
        if (pc.barrier[p] != MPI_REQUEST_NULL) MPI_Wait(&pc.barrier[p], MPI_STATUS_IGNORE);
        for (auto* v : {&pc.readySend[p], &pc.bcastReq[p]}) {
            if (!v->empty()) MPI_Waitall((int)v->size(), v->data(), MPI_STATUSES_IGNORE);
            v->clear();
        }
        for (MPI_Request& r : pc.readyRecv[p]) {
            if (r != MPI_REQUEST_NULL) {
                MPI_Cancel(&r);
                MPI_Request_free(&r);
            }
        }
        pc.readyRecv[p].clear();
    }
    MPI_Barrier(pc.node);
    if (pc.registered) CUDA_CHECK(cudaHostUnregister(pc.shm[0]));
    MPI_Win_free(&pc.win);
    if (pc.leaders != MPI_COMM_NULL) MPI_Comm_free(&pc.leaders);
    MPI_Comm_free(&pc.node);
}

// Distributed right-looking blocked Cholesky with look-ahead. On return the lower
// triangle of the distributed matrix holds L. Returns false if the matrix is not
// positive definite (the owning rank prints the diagnostic, like the reference code).
//
// The owner of panel k+1 factors it as soon as panel k has been applied to it and
// publishes the rows below the diagonal block in PANEL_CHUNKS row chunks (device->host
// copies, inter-node transfers and host->device copies are pipelined). The last chunk
// carries the failure status as an extra trailing element. The bulk trailing update
// runs concurrently on lower priority CUDA streams.
bool choleskyDecomposition(const Layout& L, DeviceState& d, MPI_Comm comm) {
    if (L.NB == 0) return true;

    const size_t ldl = (size_t)L.nloc;
    const size_t panelMax = (size_t)std::max(L.np - L.nb, 0) * L.nb + 1;
    PanelComm pc;
    createPanelComm(L, comm, panelMax, pc);
    const bool isLeader = (pc.nodeRank == 0);
    const bool multiNode = (pc.numNodes > 1);

    int localStatus[2] = {-1, -1};
    bool ownedPrev[2] = {false, false};  // this rank published the panel of that parity

    auto tagOf = [](int k, int chunk) { return (k & 1) * PANEL_CHUNKS + chunk; };
    auto chunkCount = [&](int Mrem, int i) {
        int b, e;
        chunkRange(Mrem, i, b, e);
        return (int)((size_t)(e - b) * L.nb + (i == PANEL_CHUNKS - 1 ? 1 : 0));
    };
    auto waitAllClear = [](std::vector<MPI_Request>& v) {
        if (!v.empty()) MPI_Waitall((int)v.size(), v.data(), MPI_STATUSES_IGNORE);
        v.clear();
    };

    // The node buffer of parity p is free once the node barrier that every node rank
    // posted after finishing with the previous panel of that parity has completed.
    auto acquireNodeBuffer = [&](int p) {
        if (pc.barrier[p] != MPI_REQUEST_NULL) MPI_Wait(&pc.barrier[p], MPI_STATUS_IGNORE);
    };

    // Everyone: this rank is done with the panel of parity p (host->device copies and
    // all transfers finished). Posts the non-blocking node barrier guarding reuse; the
    // buffer may be overwritten once every node rank has posted it.
    auto releaseNodeBuffer = [&](int p) {
        CUDA_CHECK(cudaEventSynchronize(d.evH2D[p]));
        waitAllClear(pc.bcastReq[p]);
        waitAllClear(pc.readySend[p]);
        if (pc.barrier[p] != MPI_REQUEST_NULL) MPI_Wait(&pc.barrier[p], MPI_STATUS_IGNORE);
        MPI_Ibarrier(pc.node, &pc.barrier[p]);
    };

    // Notify the node mates that chunk i of panel k is available in shm[p].
    auto notifyNode = [&](int k, int i) {
        const int p = k & 1;
        for (int r = 0; r < pc.nodeSize; ++r) {
            if (r == pc.nodeRank) continue;
            MPI_Request rq;
            MPI_Isend(nullptr, 0, MPI_BYTE, r, tagOf(k, i), pc.node, &rq);
            pc.readySend[p].push_back(rq);
        }
    };

    // Receiver side: post the notification receives / inter-node broadcasts for panel k.
    auto postReceive = [&](int k) {
        const int p = k & 1;
        const int owner = L.ownerOf(k);
        const int Mrem = L.np - (k + 1) * L.nb;
        ownedPrev[p] = false;
        if (pc.nodeOfRank[owner] == pc.myNode) {
            // Same node: the owner publishes directly into the node buffer.
            pc.readyRecv[p].assign(PANEL_CHUNKS, MPI_REQUEST_NULL);
            for (int i = 0; i < PANEL_CHUNKS; ++i) {
                MPI_Irecv(nullptr, 0, MPI_BYTE, pc.nodeRankOfRank[owner], tagOf(k, i), pc.node,
                          &pc.readyRecv[p][i]);
            }
        } else if (isLeader) {
            // Remote owner: the leader receives the chunks into the node buffer.
            acquireNodeBuffer(p);
            pc.bcastReq[p].assign(PANEL_CHUNKS, MPI_REQUEST_NULL);
            for (int i = 0; i < PANEL_CHUNKS; ++i) {
                int b, e;
                chunkRange(Mrem, i, b, e);
                MPI_Ibcast(pc.shm[p] + (size_t)b * L.nb, chunkCount(Mrem, i), MPI_DOUBLE,
                           pc.nodeOfRank[owner], pc.leaders, &pc.bcastReq[p][i]);
            }
        } else {
            pc.readyRecv[p].assign(PANEL_CHUNKS, MPI_REQUEST_NULL);
            for (int i = 0; i < PANEL_CHUNKS; ++i) {
                MPI_Irecv(nullptr, 0, MPI_BYTE, 0, tagOf(k, i), pc.node, &pc.readyRecv[p][i]);
            }
        }
    };

    // Owner side: the diagonal block of panel k has been factored (status = failure index
    // or -1) and L^{-1} is on the device. Form the rows below the diagonal block chunk by
    // chunk (X * L^{-T}), copy each chunk into the node buffer and publish it as soon as it
    // has arrived.
    auto sendPanel = [&](int k, int status) {
        const int p = k & 1;
        const int Mrem = L.np - (k + 1) * L.nb;
        localStatus[p] = status;
        if (status >= 0) {
            printf("Error: Matrix is not positive definite at diagonal element %d\n", status);
            fflush(stdout);
        }
        acquireNodeBuffer(p);
        double* below = d.Aloc + (size_t)(k + 1) * L.nb * ldl + L.localColOf(k);
        const size_t rowBytes = (size_t)L.nb * sizeof(double);
        for (int i = 0; i < PANEL_CHUNKS; ++i) {
            int b, e;
            chunkRange(Mrem, i, b, e);
            if (e > b && status < 0) {
                double* packed = d.sendBuf + (size_t)b * L.nb;
                gemmNT<false, false, true>(d.hi, below + (size_t)b * ldl, ldl, d.Tinv, (size_t)L.nb,
                                           packed, (size_t)L.nb, e - b, L.nb, L.nb, 0, 0, L);
                CUDA_CHECK(cudaMemcpyAsync(pc.shm[p] + (size_t)b * L.nb, packed,
                                           (size_t)(e - b) * rowBytes, cudaMemcpyDeviceToHost, d.hi));
                CUDA_CHECK(cudaEventRecord(d.evSend[i], d.hi));
                CUDA_CHECK(cudaMemcpy2DAsync(below + (size_t)b * ldl, ldl * sizeof(double), packed,
                                             rowBytes, rowBytes, (size_t)(e - b),
                                             cudaMemcpyDeviceToDevice, d.hi));
            } else {
                CUDA_CHECK(cudaEventRecord(d.evSend[i], d.hi));
            }
        }
        pc.shm[p][(size_t)Mrem * L.nb] = (double)status;
        for (int i = 0; i < PANEL_CHUNKS; ++i) {
            waitEventWithProgress(d.evSend[i], pc.node);
            if (multiNode && isLeader) {
                int b, e;
                chunkRange(Mrem, i, b, e);
                MPI_Request rq;
                MPI_Ibcast(pc.shm[p] + (size_t)b * L.nb, chunkCount(Mrem, i), MPI_DOUBLE, pc.myNode,
                           pc.leaders, &rq);
                pc.bcastReq[p].push_back(rq);
            }
            notifyNode(k, i);
        }
        ownedPrev[p] = true;
    };

    // Wait for panel k and copy the chunks to the device as they arrive; onChunk(b, e) is
    // invoked after the copy of rows [b, e) has been enqueued on stream hi. Returns the
    // failure status carried with the panel.
    auto waitPanel = [&](int k, const std::function<void(int, int)>& onChunk) -> int {
        const int p = k & 1;
        const int Mrem = L.np - (k + 1) * L.nb;
        if (ownedPrev[p]) return localStatus[p];
        const int owner = L.ownerOf(k);
        const bool ownerOnNode = (pc.nodeOfRank[owner] == pc.myNode);
        const bool forwardLeader = multiNode && isLeader && ownerOnNode;  // owner is a node mate
        const bool remoteLeader = isLeader && !ownerOnNode;
        // panelRecv[p] / Bg[p] were last read by the streams two steps ago.
        CUDA_CHECK(cudaStreamWaitEvent(d.hi, d.evLow[p], 0));
        CUDA_CHECK(cudaStreamWaitEvent(d.hi, d.evMid[p], 0));
        for (int i = 0; i < PANEL_CHUNKS; ++i) {
            int b, e;
            chunkRange(Mrem, i, b, e);
            if (remoteLeader) {
                MPI_Wait(&pc.bcastReq[p][i], MPI_STATUS_IGNORE);
                notifyNode(k, i);
            } else {
                MPI_Wait(&pc.readyRecv[p][i], MPI_STATUS_IGNORE);
                if (forwardLeader) {
                    MPI_Request rq;
                    MPI_Ibcast(pc.shm[p] + (size_t)b * L.nb, chunkCount(Mrem, i), MPI_DOUBLE,
                               pc.myNode, pc.leaders, &rq);
                    pc.bcastReq[p].push_back(rq);
                }
            }
            if (e > b) {
                CUDA_CHECK(cudaMemcpyAsync(d.panelRecv[p] + (size_t)b * L.nb,
                                           pc.shm[p] + (size_t)b * L.nb,
                                           (size_t)(e - b) * L.nb * sizeof(double),
                                           cudaMemcpyHostToDevice, d.hi));
            }
            onChunk(b, e);
        }
        CUDA_CHECK(cudaEventRecord(d.evH2D[p], d.hi));
        pc.readyRecv[p].clear();
        return (int)pc.shm[p][(size_t)Mrem * L.nb];
    };

    // Panel 0 is factored up front by its owner.
    if (L.ownerOf(0) == L.rank) {
        enqueueDiagCopy(L, d, 0, d.hi);
        const int st = hostFactorDiag(L, d, 0, d.hi, pc.node);
        sendPanel(0, st);
    } else {
        postReceive(0);
    }

    bool ok = true;
    for (int k = 0; k < L.NB; ++k) {
        const int p = k & 1;
        // Panel k-1 is no longer needed in the node buffer; this must be posted before the
        // owner of panel k+1 (same parity) publishes into that buffer during this step.
        if (k >= 1) releaseNodeBuffer((k - 1) & 1);

        const bool ownsK = (L.ownerOf(k) == L.rank);
        const bool ownsNext = (k + 1 < L.NB) && (L.ownerOf(k + 1) == L.rank);
        const int Mrem = L.np - (k + 1) * L.nb;
        const int rowOff = (k + 1) * L.nb;
        const int cStart = L.firstLocalColAfter(k);
        const int Ncols = L.nloc - cStart;
        const int n1 = std::min(Ncols, L.nb);          // first pending local block column
        const int n2 = std::min(Ncols - n1, L.nb);     // second pending local block column
        const int n3 = Ncols - n1 - n2;                // remaining columns
        double* Bg = d.Bg[p];
        double* Ctrail = d.Aloc + (size_t)rowOff * ldl + cStart;

        // Panel k rows below the diagonal block, on the device.
        const double* Pd = ownsK ? d.Aloc + (size_t)rowOff * ldl + L.localColOf(k) : d.panelRecv[p];
        const size_t ldp = ownsK ? ldl : (size_t)L.nb;

        // Look-ahead for a received panel: the next panel (first pending local block
        // column, cStart) is updated chunk by chunk as the rows arrive, the diagonal block
        // first so that its host factorization overlaps the rest.
        int c1Done = 0;
        bool diagEnqueued = false;
        auto lookaheadChunk = [&](int b, int e) {
            (void)b;
            if (!ownsNext) return;
            if (!diagEnqueued) {
                if (e < L.nb) return;
                // Panel rows of block column k+1 are the first nb rows of the panel.
                CUDA_CHECK(cudaMemcpyAsync(Bg, Pd, (size_t)L.nb * L.nb * sizeof(double),
                                           cudaMemcpyDeviceToDevice, d.hi));
                gemmNT<true, true>(d.hi, Pd, ldp, Bg, (size_t)L.nb, Ctrail, ldl, L.nb, n1, L.nb,
                                   rowOff, cStart, L);
                enqueueDiagCopy(L, d, k + 1, d.hi);
                diagEnqueued = true;
                c1Done = L.nb;
            }
            if (e > c1Done) {
                gemmNT<true, true>(d.hi, Pd + (size_t)c1Done * ldp, ldp, Bg, (size_t)L.nb,
                                   Ctrail + (size_t)c1Done * ldl, ldl, e - c1Done, n1, L.nb,
                                   rowOff + c1Done, cStart, L);
                c1Done = e;
            }
        };

        const int status = waitPanel(k, lookaheadChunk);
        if (status >= 0) { ok = false; break; }
        if (k == L.NB - 1) break;

        if (!ownsNext) postReceive(k + 1);
        if (ownsK) {
            // Bg[p] / the look-ahead column were last touched two steps ago / on mid.
            CUDA_CHECK(cudaStreamWaitEvent(d.hi, d.evLow[p], 0));
            CUDA_CHECK(cudaStreamWaitEvent(d.hi, d.evMid[p], 0));
            CUDA_CHECK(cudaStreamWaitEvent(d.hi, d.evMid[(k + 1) & 1], 0));
        }

        if (ownsNext && !diagEnqueued) {
            // Own panel (the rows are already on the device): update the next block column
            // directly, diagonal block first.
            gatherRows(d.hi, Pd, ldp, Bg, (size_t)L.nb, n1, L.nb, cStart, rowOff, L);
            gemmNT<true, true>(d.hi, Pd, ldp, Bg, (size_t)L.nb, Ctrail, ldl, L.nb, n1, L.nb,
                               rowOff, cStart, L);
            enqueueDiagCopy(L, d, k + 1, d.hi);
            gemmNT<true, true>(d.hi, Pd + (size_t)L.nb * ldp, ldp, Bg, (size_t)L.nb,
                               Ctrail + (size_t)L.nb * ldl, ldl, Mrem - L.nb, n1, L.nb,
                               rowOff + L.nb, cStart, L);
            diagEnqueued = true;
        }

        // Gather the panel rows of the (remaining) local columns.
        const int gFrom = ownsNext ? n1 : 0;
        gatherRows(d.hi, Pd, ldp, Bg + (size_t)gFrom * L.nb, (size_t)L.nb, Ncols - gFrom, L.nb,
                   cStart + gFrom, rowOff, L);
        CUDA_CHECK(cudaEventRecord(d.evGather[p], d.hi));

        if (!ownsNext) {
            // First block column on the look-ahead stream (it becomes the next panel of
            // this rank eventually).
            gemmNT<true, true>(d.hi, Pd, ldp, Bg, (size_t)L.nb, Ctrail, ldl, Mrem, n1, L.nb,
                               rowOff, cStart, L);
        }

        // Second block column: mid stream. If this rank owned panel k, this column was part
        // of the bulk update during the previous step.
        CUDA_CHECK(cudaStreamWaitEvent(d.mid, d.evGather[p], 0));
        if (ownsK) CUDA_CHECK(cudaStreamWaitEvent(d.mid, d.evLow[(k + 1) & 1], 0));
        gemmNT<true, true>(d.mid, Pd, ldp, Bg + (size_t)n1 * L.nb, (size_t)L.nb, Ctrail + n1,
                           ldl, Mrem, n2, L.nb, rowOff, cStart + n1, L);
        CUDA_CHECK(cudaEventRecord(d.evMid[p], d.mid));

        // Remaining columns: bulk trailing update on the low priority stream.
        CUDA_CHECK(cudaStreamWaitEvent(d.lo, d.evGather[p], 0));
        gemmNT<true, true>(d.lo, Pd, ldp, Bg + (size_t)(n1 + n2) * L.nb, (size_t)L.nb,
                           Ctrail + n1 + n2, ldl, Mrem, n3, L.nb, rowOff, cStart + n1 + n2, L);
        CUDA_CHECK(cudaEventRecord(d.evLow[p], d.lo));

        // Finally (host blocking parts last, so the GPU queues are full): factor the next
        // panel's diagonal block on the host and publish the panel.
        if (ownsNext) {
            const int st = hostFactorDiag(L, d, k + 1, d.hi, pc.node);
            sendPanel(k + 1, st);
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    destroyPanelComm(pc);
    return ok;
}

// Generate the symmetric positive definite test matrix A = B * B^T + n * I directly in
// the distributed layout. B is the same deterministic random matrix as in the
// reference code; the product is evaluated on the GPU with the same per-element
// accumulation order.
void generatePositiveDefiniteMatrix(const Layout& L, DeviceState& d) {
    const size_t np = (size_t)L.np;
    const size_t n = L.n;
    std::vector<double> Bpad(np * np, 0.0);
    unsigned int seed = 42;
    for (size_t i = 0; i < n; ++i) {
        double* row = Bpad.data() + i * np;
        for (size_t j = 0; j < n; ++j) {
            row[j] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
        }
    }

    // With a single rank every row of B is a local column: no gathered copy is needed.
    const bool gather = (L.P > 1);
    double* dB = nullptr;
    double* dBg = nullptr;
    CUDA_CHECK(cudaMalloc(&dB, std::max<size_t>(np * np, 1) * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(dB, Bpad.data(), np * np * sizeof(double), cudaMemcpyHostToDevice));
    if (gather) {
        CUDA_CHECK(cudaMalloc(&dBg, std::max<size_t>(np * L.nloc, 1) * sizeof(double)));
        gatherRows(d.hi, dB, np, dBg, np, L.nloc, L.np, 0, 0, L);
    }
    gemmNT<false, false>(d.hi, dB, np, gather ? dBg : dB, np, d.Aloc, (size_t)L.nloc, L.np, L.nloc,
                         L.np, 0, 0, L);
    if (L.nloc > 0) {
        add_diagonal_kernel<<<(L.nloc + 255) / 256, 256, 0, d.hi>>>(d.Aloc, L.nloc, L.nb, L.P, L.rank,
                                                           (int)L.n);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(dB));
    if (gather) CUDA_CHECK(cudaFree(dBg));
}

// Gather the distributed factor into a full (padded) row-major matrix on every rank,
// with the strict upper triangle zeroed.
void gatherFactor(const Layout& L, const DeviceState& d, MPI_Comm comm, std::vector<double>& Lfull) {
    const size_t np = (size_t)L.np;
    std::vector<double> local(np * (size_t)L.nloc);
    if (!local.empty()) {
        CUDA_CHECK(cudaMemcpy(local.data(), d.Aloc, local.size() * sizeof(double),
                              cudaMemcpyDeviceToHost));
    }

    MPI_Datatype chunk;  // nb doubles, keeps the counts small
    MPI_Type_contiguous(L.nb, MPI_DOUBLE, &chunk);
    MPI_Type_commit(&chunk);

    std::vector<int> counts(L.P), displs(L.P);
    size_t total = 0;
    for (int r = 0; r < L.P; ++r) {
        const int nl = Layout::localColsOf(r, L.NB, L.P, L.nb);
        counts[r] = (int)(np * (size_t)nl / L.nb);
        displs[r] = (int)total;
        total += (size_t)counts[r];
    }
    std::vector<double> recv(total * (size_t)L.nb);
    MPI_Allgatherv(local.data(), counts[L.rank], chunk, recv.data(), counts.data(),
                   displs.data(), chunk, comm);
    MPI_Type_free(&chunk);

    Lfull.assign(np * np, 0.0);
    for (int r = 0; r < L.P; ++r) {
        const int nl = Layout::localColsOf(r, L.NB, L.P, L.nb);
        if (nl == 0) continue;
        const double* src = recv.data() + (size_t)displs[r] * L.nb;
#pragma omp parallel for schedule(static)
        for (int i = 0; i < L.np; ++i) {
            const double* srow = src + (size_t)i * nl;
            double* drow = Lfull.data() + (size_t)i * np;
            for (int bl = 0; bl * L.P + r < L.NB; ++bl) {
                const int gc0 = (bl * L.P + r) * L.nb;
                const int w = std::min(L.nb, i + 1 - gc0);  // columns <= i only
                for (int p = 0; p < w; ++p) drow[gc0 + p] = srow[bl * L.nb + p];
            }
        }
    }
}

// Validate by computing L * L^T (distributed over ranks and GPUs) and comparing with the
// original matrix; semantics as in the reference implementation.
bool validateCholesky(const Layout& L, const std::vector<double>& Lfull,
                      const std::vector<double>& Aorig, MPI_Comm comm) {
    const size_t np = (size_t)L.np;
    const bool gather = (L.P > 1);  // single rank: every row of L is a local column
    double* dL = nullptr;
    double* dLg = nullptr;
    double* dR = nullptr;
    CUDA_CHECK(cudaMalloc(&dL, std::max<size_t>(np * np, 1) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dR, std::max<size_t>(np * L.nloc, 1) * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(dL, Lfull.data(), np * np * sizeof(double), cudaMemcpyHostToDevice));
    if (gather) {
        CUDA_CHECK(cudaMalloc(&dLg, std::max<size_t>(np * L.nloc, 1) * sizeof(double)));
        gatherRows(0, dL, np, dLg, np, L.nloc, L.np, 0, 0, L);
    }
    gemmNT<false, false>(0, dL, np, gather ? dLg : dL, np, dR, (size_t)L.nloc, L.np, L.nloc, L.np,
                         0, 0, L);

    std::vector<double> recon(np * (size_t)L.nloc);
    if (!recon.empty()) {
        CUDA_CHECK(cudaMemcpy(recon.data(), dR, recon.size() * sizeof(double),
                              cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaFree(dL));
    if (gather) CUDA_CHECK(cudaFree(dLg));
    CUDA_CHECK(cudaFree(dR));

    double maxError = 0.0;
    double relError = 0.0;
    const int n = (int)L.n;
#pragma omp parallel for schedule(static) reduction(max : maxError, relError)
    for (int i = 0; i < n; ++i) {
        for (int c = 0; c < L.nloc; ++c) {
            if (globalCol(c, L.nb, L.P, L.rank) >= n) continue;
            const size_t idx = (size_t)i * L.nloc + c;
            const double error = fabs(recon[idx] - Aorig[idx]);
            maxError = std::max(maxError, error);
            const double rel = error / (fabs(Aorig[idx]) + 1e-10);
            relError = std::max(relError, rel);
        }
    }
    double g[2] = {maxError, relError};
    MPI_Allreduce(MPI_IN_PLACE, g, 2, MPI_DOUBLE, MPI_MAX, comm);
    maxError = g[0];
    relError = g[1];

    if (L.rank == 0) {
        printf("Max absolute error: %.10e\n", maxError);
        printf("Max relative error: %.10e\n", relError);
    }

    if (relError > 1e-6) {
        if (L.rank == 0) printf("Validation failed: relative error too large\n");
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

int finish(int code) {
    MPI_Finalize();
    return code;
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank = 0, P = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &P);

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
            return finish(0);
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return finish(1);
        }
    }

    // One GPU per rank: ranks sharing a node pick distinct devices.
    {
        MPI_Comm nodeComm;
        MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
        int localRank = 0;
        MPI_Comm_rank(nodeComm, &localRank);
        MPI_Comm_free(&nodeComm);
        int devCount = 0;
        CUDA_CHECK(cudaGetDeviceCount(&devCount));
        if (devCount < 1) {
            fprintf(stderr, "No CUDA device available\n");
            MPI_Abort(comm, 1);
        }
        CUDA_CHECK(cudaSetDevice(localRank % devCount));
    }

    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // OpenMP: unless the user configured it, share the cores of the node between the
    // ranks running on it (oversubscribed spinning threads slow the host-side work).
    if (getenv("OMP_NUM_THREADS") == nullptr) {
        MPI_Comm nodeComm;
        MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
        int nodeSize = 1;
        MPI_Comm_size(nodeComm, &nodeSize);
        MPI_Comm_free(&nodeComm);
        omp_set_num_threads(std::max(1, omp_get_num_procs() / nodeSize));
    }
    if (rank == 0) printf("MPI ranks: %d, OpenMP threads per rank: %d\n", P, omp_get_max_threads());

    // Block (panel) size. Narrow panels keep the per-step critical path (look-ahead
    // update, host factorization, triangular solve and panel transfer, all proportional
    // to nb^2 per step) short; 128 measured best on 1-4 GPUs for n = 4k..32k.
    // CHOLESKY_NB (multiple of 64) overrides the default.
    int nb = 128;
    if (const char* e = getenv("CHOLESKY_NB")) {
        const int v = atoi(e);
        if (v >= TILE) nb = (v / TILE) * TILE;
    }

    Layout L;
    L.n = n;
    L.nb = nb;
    L.NB = (int)((n + L.nb - 1) / L.nb);
    L.np = L.NB * L.nb;
    L.P = P;
    L.rank = rank;
    L.nloc = Layout::localColsOf(rank, L.NB, P, L.nb);

    DeviceState d;
    const size_t locElems = (size_t)L.np * L.nloc;
    createDeviceState(L, d);

    // Generate positive definite matrix
    if (rank == 0) printf("Generating positive definite matrix...\n");
    generatePositiveDefiniteMatrix(L, d);

    std::vector<double> A_orig;
    if (validate) {
        A_orig.resize(locElems);  // Save original (local part) for validation
        if (locElems > 0) {
            CUDA_CHECK(cudaMemcpy(A_orig.data(), d.Aloc, locElems * sizeof(double),
                                  cudaMemcpyDeviceToHost));
        }
    }

    // Perform Cholesky decomposition
    if (rank == 0) {
        printf("Computing Cholesky decomposition...\n");
        fflush(stdout);
    }
    MPI_Barrier(comm);
    auto start = std::chrono::high_resolution_clock::now();

    bool success = choleskyDecomposition(L, d, comm);

    MPI_Barrier(comm);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (!success) {
        if (rank == 0) printf("Cholesky decomposition failed\n");
        return finish(1);
    }

    if (rank == 0) {
        printf("Computation time: %ld ms\n", (long)duration.count());

        // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    std::vector<double> Lfull;
    if (printResults || validate) {
        gatherFactor(L, d, comm, Lfull);
    }
    destroyDeviceState(d);

    // Print results for external validation
    if (printResults && rank == 0) {
        std::vector<double> A(n * n);
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            std::copy(Lfull.begin() + i * L.np, Lfull.begin() + i * L.np + n, A.begin() + i * n);
        }
        print_results(A, "CholeskyL");
    }

    // Validation
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        bool valid = validateCholesky(L, Lfull, A_orig, comm);

        if (valid) {
            if (rank == 0) printf("Validation: PASSED\n");
            return finish(0);
        } else {
            if (rank == 0) printf("Validation: FAILED\n");
            return finish(1);
        }
    }

    return finish(0);
}
