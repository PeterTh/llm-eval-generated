// Hybrid MPI + OpenMP + CUDA Cholesky decomposition benchmark.
//
// Decomposes a symmetric positive definite matrix A (row-major) into L * L^T
// with L lower triangular, using a right-looking blocked algorithm with
// two-step lookahead:
//
//   * MPI    : block rows of the matrix are distributed block-cyclically over
//              the ranks (block row I lives on rank I mod P). Each rank drives
//              one GPU. Per step only the current panel (one block column of L)
//              and the factored diagonal block are exchanged. Ranks on the same
//              node exchange through an MPI shared-memory window that the GPUs
//              read and write directly; across nodes each chunk is sent once to
//              the node leader.
//   * CUDA   : all O(n^3) work (matrix generation, trailing updates, panel
//              solves, diagonal block factorisation, validation) runs in
//              hand-written kernels on the rank's GPU. A high-priority stream
//              runs the panel path of step k+1 while the bulk trailing update of
//              step k runs on a second stream; uploads use a third stream.
//   * OpenMP : host-side gather/reordering, the validation comparison and
//              reductions.
//
// The executable name, command line options and printed output are identical
// to the original sequential benchmark; only rank 0 prints.

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <dirent.h>
#include <mpi.h>
#include <omp.h>
#include <sched.h>
#include <unistd.h>

#include "../common/results_output.hpp"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                                     \
    do {                                                                                     \
        cudaError_t err__ = (call);                                                          \
        if (err__ != cudaSuccess) {                                                          \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err__), __FILE__, \
                    __LINE__);                                                               \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                    \
        }                                                                                    \
    } while (0)

#define MPI_CHECK(call)                                                            \
    do {                                                                           \
        int err__ = (call);                                                        \
        if (err__ != MPI_SUCCESS) {                                                \
            fprintf(stderr, "MPI error %d at %s:%d\n", err__, __FILE__, __LINE__); \
            MPI_Abort(MPI_COMM_WORLD, 1);                                          \
        }                                                                          \
    } while (0)

// GEMM tile configuration (double precision). 256 threads compute a
// (16*TM) x 64 tile with TM x 4 outputs per thread; TM = 4 for large GEMMs,
// TM = 1 for narrow ones (more thread blocks, better SM utilisation).
constexpr int BN = 64;
constexpr int BK = 16;
constexpr int GEMM_THREADS = 256;
constexpr int SB = 64;        // sub-block size for diagonal block factorisation / panel solves
constexpr int LOOKAHEAD = 2;  // block columns updated ahead of the bulk trailing update

// Describes the block-cyclic distribution and the panel layout for the
// mapped GEMM variants. The exchanged panel of step kb holds, for every rank in
// rank order, that rank's block rows I > kb in increasing order.
struct PanelMap {
    int nb;       // block size
    int P;        // number of ranks
    int rank;     // this rank
    int nBlocks;  // number of block rows/columns (padded)
    int kb;       // panel index (block column) of the current step
    int lstart;   // local block-row index corresponding to C row 0
    int jbase;    // global block index of column 0 of C
};

__host__ __device__ inline int panelFirstLocal(const PanelMap& pm, int r) {
    // first local block row of rank r with global index > kb
    return (pm.kb + 1 > r) ? (pm.kb + 1 - r + pm.P - 1) / pm.P : 0;
}
__host__ __device__ inline int panelTiles(const PanelMap& pm, int r) {
    const int nl = (pm.nBlocks > r) ? (pm.nBlocks - r + pm.P - 1) / pm.P : 0;
    const int t = nl - panelFirstLocal(pm, r);
    return t > 0 ? t : 0;
}
__host__ __device__ inline int panelPos(const PanelMap& pm, int I) {
    // tile index of global block row I inside the exchanged panel
    const int r = I % pm.P;
    int off = 0;
    for (int q = 0; q < r; ++q) off += panelTiles(pm, q);
    return off + I / pm.P - panelFirstLocal(pm, r);
}

// ---------------------------------------------------------------------------
// CUDA kernels
// ---------------------------------------------------------------------------

// C[M x N] = alpha * A[M x K] * B[N x K]^T + beta * C   (all row-major)
// Requirements: M multiple of 16*TM, N multiple of 64, K multiple of 16.
// AMODE 0: A rows are C rows.
// AMODE 1: A is stored in global block-row order (full matrix); the tile at
//          C row row0 (local block row l) is A row I*nb + row0%nb, I = l*P + rank.
// AMODE 2: A is the exchanged panel; tile row = panelPos(I)*nb + row0%nb.
// BMAP   : B is the exchanged panel; the tile for global block column J is at
//          panelPos(J)*nb + col0%nb.
// LOWER  : tiles with global block column J > global block row I are skipped.
template <int TM, int AMODE, bool BMAP, bool LOWER>
__global__ void __launch_bounds__(GEMM_THREADS)
gemm_nt_kernel(int K, double alpha, const double* __restrict__ A, int lda,
               const double* __restrict__ B, int ldb, double beta, double* __restrict__ C,
               int ldc, PanelMap pm) {
    constexpr int BM = 16 * TM;
    const int tid = threadIdx.x;
    const int tx = tid & 15;
    const int ty = tid >> 4;
    const int row0 = blockIdx.y * BM;
    const int col0 = blockIdx.x * BN;

    int arow0 = row0;
    int brow0 = col0;
    if (AMODE != 0 || BMAP || LOWER) {
        const int I = (pm.lstart + row0 / pm.nb) * pm.P + pm.rank;
        const int J = pm.jbase + col0 / pm.nb;
        if (LOWER && J > I) return;
        if (AMODE == 1) arow0 = I * pm.nb + (row0 % pm.nb);
        if (AMODE == 2) arow0 = panelPos(pm, I) * pm.nb + (row0 % pm.nb);
        if (BMAP) brow0 = panelPos(pm, J) * pm.nb + (col0 % pm.nb);
    }

    const double* Ab = A + (size_t)arow0 * lda;
    const double* Bb = B + (size_t)brow0 * ldb;

    __shared__ double As[BK][BM + 1];
    __shared__ double Bs[BK][BN + 1];

    double acc[TM][4];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = 0.0;

    for (int k0 = 0; k0 < K; k0 += BK) {
#pragma unroll
        for (int q = 0; q < TM; ++q) {
            const int idx = tid + q * GEMM_THREADS;
            As[idx & 15][idx >> 4] = Ab[(size_t)(idx >> 4) * lda + k0 + (idx & 15)];
        }
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const int idx = tid + q * GEMM_THREADS;
            Bs[idx & 15][idx >> 4] = Bb[(size_t)(idx >> 4) * ldb + k0 + (idx & 15)];
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            double a[TM], b[4];
#pragma unroll
            for (int i = 0; i < TM; ++i) a[i] = As[kk][ty * TM + i];
#pragma unroll
            for (int j = 0; j < 4; ++j) b[j] = Bs[kk][tx * 4 + j];
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM; ++i) {
        double* crow = C + (size_t)(row0 + ty * TM + i) * ldc + col0 + tx * 4;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            if (beta == 0.0)
                crow[j] = alpha * acc[i][j];
            else
                crow[j] = alpha * acc[i][j] + beta * crow[j];
        }
    }
}

// Unblocked Cholesky of a 64x64 diagonal sub-block held in shared memory,
// followed by its explicit lower-triangular inverse. Single thread block of
// 256 threads: 4 threads per row for the rank-1 updates, 4 threads per column
// for the inverse. On a non-positive pivot the global index (idxBase + j) is
// recorded in *err (only the first failure is recorded) and the kernel exits.
constexpr int POTRF_THREADS = 256;
constexpr size_t POTRF_SMEM = 2 * SB * (SB + 1) * sizeof(double);
__global__ void __launch_bounds__(POTRF_THREADS)
potrf64_kernel(double* __restrict__ T, int ld, double* __restrict__ inv, int* __restrict__ err,
               int idxBase) {
    extern __shared__ double smem[];
    double (*s)[SB + 1] = reinterpret_cast<double (*)[SB + 1]>(smem);
    double (*w)[SB + 1] = s + SB;
    __shared__ int bad;
    const int tid = threadIdx.x;
    const int i = tid >> 2;  // row (or column for the inverse)
    const int q = tid & 3;   // lane within the group of 4

    if (tid == 0) bad = (*err >= 0) ? 1 : 0;
    for (int idx = tid; idx < SB * SB; idx += POTRF_THREADS) {
        const int r = idx / SB, m = idx % SB;
        s[r][m] = T[(size_t)r * ld + m];
    }
    __syncthreads();
    if (bad) return;  // an earlier sub-block already failed

    for (int j = 0; j < SB; ++j) {
        if (tid == 0) {
            const double d = s[j][j];
            if (d <= 0.0) {
                *err = idxBase + j;
                bad = 1;
            } else {
                s[j][j] = sqrt(d);
            }
        }
        __syncthreads();
        if (bad) return;
        if (i > j && q == 0) s[i][j] /= s[j][j];
        __syncthreads();
        if (i > j) {
            const double lij = s[i][j];
            for (int m = j + 1 + q; m <= i; m += 4) s[i][m] -= lij * s[m][j];
        }
        __syncthreads();
    }

    // zero the strict upper triangle of the factor
    for (int idx = tid; idx < SB * SB; idx += POTRF_THREADS) {
        const int r = idx / SB, m = idx % SB;
        if (m > r) s[r][m] = 0.0;
    }
    __syncthreads();

    // inverse by forward substitution: the 4 threads of column j = i split each
    // inner product and combine it with warp shuffles (uniform loop over r for
    // the whole warp, as the shuffles need all lanes)
    {
        const int j = i;
        for (int r = 0; r < SB; ++r) {
            double v = 0.0;
            if (r >= j)
                for (int m = j + q; m < r; m += 4) v += s[r][m] * w[m][j];
            v += __shfl_xor_sync(0xffffffffu, v, 1);
            v += __shfl_xor_sync(0xffffffffu, v, 2);
            if (q == 0) w[r][j] = (r < j) ? 0.0 : (((r == j) ? 1.0 : 0.0) - v) / s[r][r];
            __syncwarp();
        }
    }
    __syncthreads();

    for (int idx = tid; idx < SB * SB; idx += POTRF_THREADS) {
        const int r = idx / SB, m = idx % SB;
        T[(size_t)r * ld + m] = s[r][m];
        inv[r * SB + m] = w[r][m];
    }
}

// A_local[i][i] += n for real rows, = 1 for padding rows (identity padding
// leaves the Cholesky factor of the real part unchanged).
__global__ void fix_diagonal_kernel(double* __restrict__ A, int ldA, int rowsLocal, int nb, int P,
                                    int rank, long n) {
    const int lr = blockIdx.x * blockDim.x + threadIdx.x;
    if (lr >= rowsLocal) return;
    const long gi = (long)((lr / nb) * P + rank) * nb + (lr % nb);
    double* d = A + (size_t)lr * ldA + gi;
    if (gi < n)
        *d += (double)n;
    else
        *d = 1.0;
}

// Zero the strict upper triangle of the local rows.
__global__ void zero_upper_kernel(double* __restrict__ A, int ldA, int rowsLocal, int cols,
                                  int nb, int P, int rank) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int lr = blockIdx.y;
    if (lr >= rowsLocal || c >= cols) return;
    const long gi = (long)((lr / nb) * P + rank) * nb + (lr % nb);
    if (c > gi) A[(size_t)lr * ldA + c] = 0.0;
}

// Set unit diagonal for padding rows of a full (Npad x Npad) matrix.
__global__ void pad_identity_kernel(double* __restrict__ M, int ld, long n, int Npad) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= Npad || i < n) return;
    M[(size_t)i * ld + i] = 1.0;
}

// ---------------------------------------------------------------------------
// Distributed context
// ---------------------------------------------------------------------------

struct Dist {
    int rank = 0, P = 1;
    size_t n = 0;              // real matrix size
    int nb = 256;              // block size
    int Npad = 0;              // padded size (multiple of nb)
    int nBlocks = 0;           // Npad / nb
    int nLocal = 0;            // local block rows
    int rowsLocal = 0;         // nLocal * nb
    size_t realRowsLocal = 0;  // local rows that are < n
    std::vector<size_t> realRowOff;  // host offset (in rows) of each local block row
    std::vector<int> realRowCnt;     // real rows of each local block row
    cudaStream_t stream = nullptr;   // bulk trailing-update stream
    cudaStream_t pstream = nullptr;  // high-priority panel (lookahead) stream
    cudaStream_t cstream = nullptr;  // panel upload stream
    MPI_Datatype rowType = MPI_DATATYPE_NULL;  // n contiguous doubles

    // node topology
    MPI_Comm nodeComm = MPI_COMM_NULL;
    int nodeRank = 0, nodeSize = 1;
    std::vector<int> nodeOf;    // node id (world rank of the node leader) per world rank
    std::vector<int> leaders;   // world ranks of all node leaders
    int myNode = 0;

    int globalBlock(int l) const { return l * P + rank; }
    int ownerOf(int I) const { return I % P; }
    int localOf(int I) const { return I / P; }
};

static int chooseBlockSize(size_t n) {
    if (n <= 1024) return 64;
    if (n <= 4096) return 128;
    return 256;
}

static void setupDist(Dist& d, size_t n) {
    d.n = n;
    d.nb = chooseBlockSize(n);
    d.nBlocks = (int)((n + d.nb - 1) / d.nb);
    d.Npad = d.nBlocks * d.nb;
    d.nLocal = (d.nBlocks > d.rank) ? (d.nBlocks - d.rank + d.P - 1) / d.P : 0;
    d.rowsLocal = d.nLocal * d.nb;
    d.realRowOff.assign(d.nLocal + 1, 0);
    d.realRowCnt.assign(d.nLocal, 0);
    size_t off = 0;
    for (int l = 0; l < d.nLocal; ++l) {
        const long first = (long)d.globalBlock(l) * d.nb;
        long cnt = std::min<long>(d.nb, (long)n - first);
        if (cnt < 0) cnt = 0;
        d.realRowOff[l] = off;
        d.realRowCnt[l] = (int)cnt;
        off += cnt;
    }
    d.realRowOff[d.nLocal] = off;
    d.realRowsLocal = off;
    MPI_CHECK(MPI_Type_contiguous((int)n, MPI_DOUBLE, &d.rowType));
    MPI_CHECK(MPI_Type_commit(&d.rowType));
}

// Node topology: ranks sharing memory form a node; the node leader is the
// lowest world rank on it.
static void setupTopology(Dist& d) {
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                                  &d.nodeComm));
    MPI_CHECK(MPI_Comm_rank(d.nodeComm, &d.nodeRank));
    MPI_CHECK(MPI_Comm_size(d.nodeComm, &d.nodeSize));
    int leader = d.rank;
    MPI_CHECK(MPI_Bcast(&leader, 1, MPI_INT, 0, d.nodeComm));
    d.myNode = leader;
    d.nodeOf.assign(d.P, 0);
    MPI_CHECK(MPI_Allgather(&leader, 1, MPI_INT, d.nodeOf.data(), 1, MPI_INT, MPI_COMM_WORLD));
    for (int r = 0; r < d.P; ++r)
        if (d.nodeOf[r] == r) d.leaders.push_back(r);
}

// NUMA node of the CPU this thread currently runs on (-1 if unknown)
static int currentNumaNode() {
    const int cpu = sched_getcpu();
    if (cpu < 0) return -1;
    char path[128];
    snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d", cpu);
    DIR* dir = opendir(path);
    if (!dir) return -1;
    int node = -1;
    while (dirent* e = readdir(dir)) {
        if (strncmp(e->d_name, "node", 4) == 0 && e->d_name[4] >= '0' && e->d_name[4] <= '9') {
            node = atoi(e->d_name + 4);
            break;
        }
    }
    closedir(dir);
    return node;
}

// NUMA node of a CUDA device (-1 if unknown)
static int deviceNumaNode(int dev) {
    char bus[64];
    if (cudaDeviceGetPCIBusId(bus, sizeof(bus), dev) != cudaSuccess) return -1;
    for (char* p = bus; *p; ++p) *p = (char)tolower(*p);
    char path[160];
    snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/numa_node", bus);
    FILE* f = fopen(path, "r");
    if (!f) return -1;
    int node = -1;
    if (fscanf(f, "%d", &node) != 1) node = -1;
    fclose(f);
    return node;
}

// Pick one GPU per rank on the node, preferring GPUs attached to the rank's
// NUMA node. All ranks on the node compute the same assignment.
static int selectDevice(const Dist& d) {
    int devCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devCount));
    if (devCount <= 0) {
        if (d.rank == 0) fprintf(stderr, "No CUDA devices found\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    std::vector<int> devNuma(devCount);
    for (int i = 0; i < devCount; ++i) devNuma[i] = deviceNumaNode(i);
    const int myNuma = currentNumaNode();
    std::vector<int> prefs(d.nodeSize);
    MPI_CHECK(MPI_Allgather(&myNuma, 1, MPI_INT, prefs.data(), 1, MPI_INT, d.nodeComm));
    std::vector<int> assigned(d.nodeSize, -1);
    std::vector<int> used(devCount, 0);
    for (int r = 0; r < d.nodeSize; ++r) {
        if (r % devCount == 0) std::fill(used.begin(), used.end(), 0);  // wrap around
        int pick = -1;
        for (int i = 0; i < devCount && pick < 0; ++i)
            if (!used[i] && prefs[r] >= 0 && devNuma[i] == prefs[r]) pick = i;
        for (int i = 0; i < devCount && pick < 0; ++i)
            if (!used[i]) pick = i;
        used[pick] = 1;
        assigned[r] = pick;
    }
    return assigned[d.nodeRank];
}

// GEMM launch helper ----------------------------------------------------------

template <int TM, int AMODE, bool BMAP, bool LOWER>
static void gemm_nt(cudaStream_t st, int M, int N, int K, double alpha, const double* A, int lda,
                    const double* B, int ldb, double beta, double* C, int ldc, const PanelMap& pm) {
    if (M <= 0 || N <= 0) return;
    dim3 grid(N / BN, M / (16 * TM));
    gemm_nt_kernel<TM, AMODE, BMAP, LOWER><<<grid, GEMM_THREADS, 0, st>>>(K, alpha, A, lda, B,
                                                                            ldb, beta, C, ldc, pm);
    CUDA_CHECK(cudaGetLastError());
}

static PanelMap plainMap(const Dist& d) { return PanelMap{d.nb, d.P, d.rank, d.nBlocks, 0, 0, 0}; }

// Copy the local (real) rows of a device matrix (ld = Npad) to a host buffer
// with leading dimension n, in local block-row order.
static void localRowsToHost(const Dist& d, const double* dM, double* hM) {
    for (int l = 0; l < d.nLocal; ++l) {
        if (d.realRowCnt[l] == 0) continue;
        CUDA_CHECK(cudaMemcpy2DAsync(hM + d.realRowOff[l] * d.n, d.n * sizeof(double),
                                     dM + (size_t)l * d.nb * d.Npad, (size_t)d.Npad * sizeof(double),
                                     d.n * sizeof(double), d.realRowCnt[l], cudaMemcpyDeviceToHost,
                                     d.stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(d.stream));
}

// Gather the local real rows of all ranks (local block-row order per rank) onto
// rank 0 and reorder into a full n x n matrix.
static void gatherToRoot(const Dist& d, const double* hLocal, std::vector<double>& full) {
    std::vector<int> counts(d.P), displs(d.P);
    const int myCount = (int)d.realRowsLocal;
    MPI_CHECK(MPI_Gather(&myCount, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD));
    std::vector<double> packed;
    if (d.rank == 0) {
        size_t total = 0;
        for (int r = 0; r < d.P; ++r) {
            displs[r] = (int)total;
            total += counts[r];
        }
        packed.resize(total * d.n);
    }
    MPI_CHECK(MPI_Gatherv(hLocal, myCount, d.rowType, packed.data(), counts.data(), displs.data(),
                          d.rowType, 0, MPI_COMM_WORLD));
    if (d.rank == 0) {
        full.resize(d.n * d.n);
        // rank r's chunk holds its block rows in order: block t of rank r is global block t*P + r
#pragma omp parallel for schedule(dynamic)
        for (int I = 0; I < d.nBlocks; ++I) {
            const int r = I % d.P;
            const int t = I / d.P;
            size_t off = 0;  // row offset of block t within rank r's chunk
            for (int tt = 0; tt < t; ++tt) {
                const long first = (long)(tt * d.P + r) * d.nb;
                long cnt = std::min<long>(d.nb, (long)d.n - first);
                if (cnt < 0) cnt = 0;
                off += cnt;
            }
            const long first = (long)I * d.nb;
            long cnt = std::min<long>(d.nb, (long)d.n - first);
            if (cnt <= 0) continue;
            std::memcpy(full.data() + (size_t)first * d.n,
                        packed.data() + ((size_t)displs[r] + off) * d.n,
                        (size_t)cnt * d.n * sizeof(double));
        }
    }
}

// ---------------------------------------------------------------------------
// Matrix generation (same random sequence as the original: rand_r, seed 42)
// A = B * B^T + n * I, computed on the GPU for the local block rows.
// ---------------------------------------------------------------------------

static void generatePositiveDefiniteMatrix(const Dist& d, double* dA) {
    const size_t n = d.n;
    // Random matrix B (identical on every rank; the rand_r sequence is inherently sequential)
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

    double* dB = nullptr;
    CUDA_CHECK(cudaMalloc(&dB, (size_t)d.Npad * d.Npad * sizeof(double)));
    CUDA_CHECK(cudaMemsetAsync(dB, 0, (size_t)d.Npad * d.Npad * sizeof(double), d.stream));
    CUDA_CHECK(cudaMemcpy2DAsync(dB, (size_t)d.Npad * sizeof(double), B.data(), n * sizeof(double),
                                 n * sizeof(double), n, cudaMemcpyHostToDevice, d.stream));

    // A_local = B_local * B^T  (A operand mapped to the global block rows of B)
    gemm_nt<4, 1, false, false>(d.stream, d.rowsLocal, d.Npad, d.Npad, 1.0, dB, d.Npad, dB,
                                d.Npad, 0.0, dA, d.Npad, plainMap(d));

    if (d.rowsLocal > 0) {
        fix_diagonal_kernel<<<(d.rowsLocal + 255) / 256, 256, 0, d.stream>>>(
            dA, d.Npad, d.rowsLocal, d.nb, d.P, d.rank, (long)n);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaStreamSynchronize(d.stream));
    CUDA_CHECK(cudaFree(dB));
}

// ---------------------------------------------------------------------------
// Distributed blocked Cholesky
// ---------------------------------------------------------------------------

struct Workspace {
    double* dPanel[LOOKAHEAD + 1] = {};  // exchanged panel (rank-concatenated tile order),
                                         // multi-buffered for the lookahead
    double* dPanelLoc = nullptr;         // (rowsLocal x nb) local panel tiles
    double* dLkk = nullptr;              // nb x nb factored diagonal block (non-owners)
    double* dInv64 = nullptr;            // (nb/64) inverses of the 64x64 diagonal sub-blocks
    double* dTmp = nullptr;              // max(rowsLocal, nb) x 64 scratch for solves
    int* dErr = nullptr;
    // node-shared window: [panel 0 | panel 1 | diag 0 | diag 1]
    MPI_Win win = MPI_WIN_NULL;
    double* winBase = nullptr;
    size_t winBytes = 0;
    bool winRegistered = false;
    double* shPanel[2] = {nullptr, nullptr};
    double* shDiag[2] = {nullptr, nullptr};
    std::vector<cudaEvent_t> evPanelReady;  // per step: panel fully on the device
    std::vector<cudaEvent_t> evPrio;        // per step: own tiles + lookahead tiles on the device
    std::vector<cudaEvent_t> evUpdateDone;  // per step: bulk trailing update finished
};

static bool choleskyDecomposition(const Dist& d, double* dA, Workspace& w, int& failIndex) {
    const int nb = d.nb;
    const int P = d.P;
    const size_t tile = (size_t)nb * nb;
    const int nsb = nb / SB;
    const size_t invCount = (size_t)nsb * SB * SB;
    const size_t diagCount = tile + invCount;
    cudaStream_t ps = d.pstream;  // panel path (lookahead), high priority
    cudaStream_t us = d.stream;   // bulk trailing updates
    cudaStream_t cs = d.cstream;  // panel uploads
    const PanelMap plain = plainMap(d);

    std::vector<int> counts(P), displs(P);
    std::vector<MPI_Request> reqs;
    failIndex = -1;

    auto mapFor = [&](int kb, int jbase) {
        PanelMap pm{nb, P, d.rank, d.nBlocks, kb, 0, jbase};
        pm.lstart = panelFirstLocal(pm, d.rank);
        return pm;
    };

    // Factorise diagonal block kb in place (owner only), produce the inverses of
    // its 64x64 diagonal sub-blocks and publish both in the shared window.
    // Returns the failing global index or -1.
    auto factorDiagonal = [&](int kb) -> int {
        double* T = dA + (size_t)d.localOf(kb) * nb * d.Npad + (size_t)kb * nb;
        int err = -1;
        CUDA_CHECK(cudaMemsetAsync(w.dErr, 0xFF, sizeof(int), ps));  // -1
        for (int c = 0; c < nsb; ++c) {
            double* Tc = T + (size_t)c * SB * d.Npad + (size_t)c * SB;
            double* invc = w.dInv64 + (size_t)c * SB * SB;
            potrf64_kernel<<<1, POTRF_THREADS, POTRF_SMEM, ps>>>(Tc, d.Npad, invc, w.dErr,
                                                                 kb * nb + c * SB);
            CUDA_CHECK(cudaGetLastError());
            const int R = (nsb - 1 - c) * SB;
            if (R > 0) {
                double* below = T + (size_t)(c + 1) * SB * d.Npad + (size_t)c * SB;
                // L_below = A_below * L_cc^{-T}
                gemm_nt<1, 0, false, false>(ps, R, SB, SB, 1.0, below, d.Npad, invc, SB, 0.0,
                                            w.dTmp, SB, plain);
                CUDA_CHECK(cudaMemcpy2DAsync(below, (size_t)d.Npad * sizeof(double), w.dTmp,
                                             SB * sizeof(double), SB * sizeof(double), R,
                                             cudaMemcpyDeviceToDevice, ps));
                // trailing sub-matrix of the block
                double* trail = T + (size_t)(c + 1) * SB * d.Npad + (size_t)(c + 1) * SB;
                gemm_nt<1, 0, false, false>(ps, R, R, SB, -1.0, below, d.Npad, below, d.Npad, 1.0,
                                            trail, d.Npad, plain);
            }
        }
        double* sh = w.shDiag[kb % 2];
        CUDA_CHECK(cudaMemcpyAsync(&err, w.dErr, sizeof(int), cudaMemcpyDeviceToHost, ps));
        CUDA_CHECK(cudaMemcpy2DAsync(sh, nb * sizeof(double), T, (size_t)d.Npad * sizeof(double),
                                     nb * sizeof(double), nb, cudaMemcpyDeviceToHost, ps));
        CUDA_CHECK(cudaMemcpyAsync(sh + tile, w.dInv64, invCount * sizeof(double),
                                   cudaMemcpyDeviceToHost, ps));
        CUDA_CHECK(cudaStreamSynchronize(ps));
        return err;
    };

    // Panel step for block column kb: factorise the diagonal block, distribute
    // it, solve the local panel tiles, exchange the panel between all ranks and
    // upload it into dPanel[kb % (LOOKAHEAD+1)]: first the tiles needed by the
    // lookahead column updates (stream ps), then the rest (stream cs).
    // Returns false if the matrix is not positive definite.
    auto panelStep = [&](int kb) -> bool {
        const int owner = d.ownerOf(kb);
        int err = -1;
        if (d.rank == owner) err = factorDiagonal(kb);
        MPI_CHECK(MPI_Bcast(&err, 1, MPI_INT, owner, MPI_COMM_WORLD));
        if (err >= 0) {
            failIndex = err;
            return false;
        }
        if (kb + 1 >= d.nBlocks) return true;  // last block: nothing below

        // --- distribute the diagonal block: shared window on the owner's node,
        //     one message per other node (to its leader)
        double* shD = w.shDiag[kb % 2];
        reqs.clear();
        if (d.rank == owner) {
            for (int L : d.leaders) {
                if (L == d.myNode) continue;
                reqs.emplace_back();
                MPI_CHECK(MPI_Isend(shD, (int)diagCount, MPI_DOUBLE, L, 2 * kb + 1, MPI_COMM_WORLD,
                                    &reqs.back()));
            }
        } else if (d.nodeRank == 0 && d.nodeOf[owner] != d.myNode) {
            reqs.emplace_back();
            MPI_CHECK(MPI_Irecv(shD, (int)diagCount, MPI_DOUBLE, owner, 2 * kb + 1, MPI_COMM_WORLD,
                                &reqs.back()));
        }
        if (!reqs.empty()) MPI_CHECK(MPI_Waitall((int)reqs.size(), reqs.data(), MPI_STATUSES_IGNORE));
        MPI_CHECK(MPI_Win_sync(w.win));
        MPI_CHECK(MPI_Barrier(d.nodeComm));
        MPI_CHECK(MPI_Win_sync(w.win));

        const double* Lkk;
        int ldL;
        if (d.rank == owner) {
            Lkk = dA + (size_t)d.localOf(kb) * nb * d.Npad + (size_t)kb * nb;
            ldL = d.Npad;
        } else {
            CUDA_CHECK(cudaMemcpyAsync(w.dLkk, shD, tile * sizeof(double), cudaMemcpyHostToDevice,
                                       ps));
            CUDA_CHECK(cudaMemcpyAsync(w.dInv64, shD + tile, invCount * sizeof(double),
                                       cudaMemcpyHostToDevice, ps));
            Lkk = w.dLkk;
            ldL = nb;
        }

        // --- panel solve X = A_Ik * L_kk^{-T} for the local block rows I > kb,
        //     blocked over 64-column chunks: X_c = (A_c - X_{<c} * L_{c,<c}^T) * L_cc^{-T}
        const PanelMap pm = mapFor(kb, kb + 1);
        const int lstart = pm.lstart;
        const int Mloc = panelTiles(pm, d.rank) * nb;
        int total = 0;
        for (int r = 0; r < P; ++r) {
            counts[r] = (int)(panelTiles(pm, r) * tile);
            displs[r] = total;
            total += counts[r];
        }
        double* shP = w.shPanel[kb % 2];
        if (Mloc > 0) {
            const double* Acol = dA + (size_t)lstart * nb * d.Npad + (size_t)kb * nb;
            for (int c = 0; c < nsb; ++c) {
                CUDA_CHECK(cudaMemcpy2DAsync(w.dTmp, SB * sizeof(double), Acol + c * SB,
                                             (size_t)d.Npad * sizeof(double), SB * sizeof(double),
                                             Mloc, cudaMemcpyDeviceToDevice, ps));
                if (c > 0) {
                    gemm_nt<1, 0, false, false>(ps, Mloc, SB, c * SB, -1.0, w.dPanelLoc, nb,
                                                Lkk + (size_t)c * SB * ldL, ldL, 1.0, w.dTmp, SB,
                                                plain);
                }
                gemm_nt<1, 0, false, false>(ps, Mloc, SB, SB, 1.0, w.dTmp, SB,
                                            w.dInv64 + (size_t)c * SB * SB, SB, 0.0,
                                            w.dPanelLoc + c * SB, nb, plain);
            }
            // publish our tiles in the shared window; final L tiles back into the matrix
            CUDA_CHECK(cudaMemcpyAsync(shP + displs[d.rank], w.dPanelLoc,
                                       (size_t)Mloc * nb * sizeof(double), cudaMemcpyDeviceToHost,
                                       ps));
            CUDA_CHECK(cudaMemcpy2DAsync(dA + (size_t)lstart * nb * d.Npad + (size_t)kb * nb,
                                         (size_t)d.Npad * sizeof(double), w.dPanelLoc,
                                         nb * sizeof(double), nb * sizeof(double), Mloc,
                                         cudaMemcpyDeviceToDevice, ps));
            CUDA_CHECK(cudaStreamSynchronize(ps));
        }

        // --- exchange: send our chunk to the leaders of the other nodes; leaders
        //     receive the chunks of remote ranks into the window
        reqs.clear();
        if (counts[d.rank] > 0) {
            for (int L : d.leaders) {
                if (L == d.myNode) continue;
                reqs.emplace_back();
                MPI_CHECK(MPI_Isend(shP + displs[d.rank], counts[d.rank], MPI_DOUBLE, L, 2 * kb,
                                    MPI_COMM_WORLD, &reqs.back()));
            }
        }
        if (d.nodeRank == 0) {
            for (int r = 0; r < P; ++r) {
                if (d.nodeOf[r] == d.myNode || counts[r] == 0) continue;
                reqs.emplace_back();
                MPI_CHECK(MPI_Irecv(shP + displs[r], counts[r], MPI_DOUBLE, r, 2 * kb,
                                    MPI_COMM_WORLD, &reqs.back()));
            }
        }
        if (!reqs.empty()) MPI_CHECK(MPI_Waitall((int)reqs.size(), reqs.data(), MPI_STATUSES_IGNORE));
        // our reads of this window buffer from step kb-2 must be finished before
        // anyone overwrites it (which happens after this barrier)
        if (kb >= 2) CUDA_CHECK(cudaEventSynchronize(w.evPanelReady[kb - 2]));
        MPI_CHECK(MPI_Win_sync(w.win));
        MPI_CHECK(MPI_Barrier(d.nodeComm));
        MPI_CHECK(MPI_Win_sync(w.win));

        // --- upload: own chunk (device copy) and the tiles needed by the lookahead
        //     column updates first, on the panel stream
        double* dP = w.dPanel[kb % (LOOKAHEAD + 1)];
        if (Mloc > 0) {
            CUDA_CHECK(cudaMemcpyAsync(dP + displs[d.rank], w.dPanelLoc,
                                       (size_t)Mloc * nb * sizeof(double), cudaMemcpyDeviceToDevice,
                                       ps));
        }
        std::vector<int> prio;  // tile positions already uploaded via ps
        for (int t = kb + 1; t <= kb + LOOKAHEAD && t < d.nBlocks; ++t) {
            if (d.ownerOf(t) == d.rank) continue;
            const int pos = panelPos(pm, t);
            prio.push_back(pos);
            CUDA_CHECK(cudaMemcpyAsync(dP + (size_t)pos * tile, shP + (size_t)pos * tile,
                                       tile * sizeof(double), cudaMemcpyHostToDevice, ps));
        }
        CUDA_CHECK(cudaEventRecord(w.evPrio[kb], ps));
        // the rest on the upload stream
        CUDA_CHECK(cudaStreamWaitEvent(cs, w.evPrio[kb], 0));
        for (int r = 0; r < P; ++r) {
            if (r == d.rank || counts[r] == 0) continue;
            const int first = displs[r] / (int)tile;
            const int last = first + counts[r] / (int)tile;  // exclusive
            int from = first;
            for (int t = first; t <= last; ++t) {
                const bool skip = (t == last) || std::find(prio.begin(), prio.end(), t) != prio.end();
                if (!skip) continue;
                if (t > from) {
                    CUDA_CHECK(cudaMemcpyAsync(dP + (size_t)from * tile, shP + (size_t)from * tile,
                                               (size_t)(t - from) * tile * sizeof(double),
                                               cudaMemcpyHostToDevice, cs));
                }
                from = t + 1;
            }
        }
        CUDA_CHECK(cudaEventRecord(w.evPanelReady[kb], cs));
        return true;
    };

    // Trailing update with panel kb for the local rows I > kb and block columns
    // jfirst <= J <= I (jcount block columns), on stream st.
    auto trailingUpdate = [&](cudaStream_t st, int kb, int jfirst, int jcount) {
        const PanelMap pm = mapFor(kb, jfirst);
        const int Mloc = panelTiles(pm, d.rank) * nb;
        if (Mloc <= 0 || jcount <= 0) return;
        double* C = dA + (size_t)pm.lstart * nb * d.Npad + (size_t)jfirst * nb;
        const double* dP = w.dPanel[kb % (LOOKAHEAD + 1)];
        gemm_nt<4, 2, true, true>(st, Mloc, jcount * nb, nb, -1.0, dP, nb, dP, nb, 1.0, C, d.Npad,
                                  pm);
    };

    // Prologue: panel 0
    if (!panelStep(0)) {
        CUDA_CHECK(cudaDeviceSynchronize());
        return false;
    }

    for (int k = 0; k + 1 < d.nBlocks; ++k) {
        // Bulk trailing update of step k (columns J >= k + 1 + LOOKAHEAD) on the
        // update stream; it overlaps with the lookahead panel step k + 1 below.
        CUDA_CHECK(cudaStreamWaitEvent(us, w.evPanelReady[k], 0));
        trailingUpdate(us, k, k + 1 + LOOKAHEAD, d.nBlocks - (k + 1 + LOOKAHEAD));
        CUDA_CHECK(cudaEventRecord(w.evUpdateDone[k], us));

        // Lookahead: update block columns k+1 .. k+LOOKAHEAD with panel k on the
        // panel stream (they only need the bulk update of step k - LOOKAHEAD),
        // then run the panel step k + 1.
        if (k >= LOOKAHEAD) CUDA_CHECK(cudaStreamWaitEvent(ps, w.evUpdateDone[k - LOOKAHEAD], 0));
        trailingUpdate(ps, k, k + 1, std::min(LOOKAHEAD, d.nBlocks - (k + 1)));
        if (!panelStep(k + 1)) {
            CUDA_CHECK(cudaDeviceSynchronize());
            return false;
        }
    }

    // Zero the strict upper triangle of the local rows
    CUDA_CHECK(cudaDeviceSynchronize());
    if (d.rowsLocal > 0) {
        dim3 grid((d.Npad + 255) / 256, d.rowsLocal);
        zero_upper_kernel<<<grid, 256, 0, us>>>(dA, d.Npad, d.rowsLocal, d.Npad, nb, P, d.rank);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaStreamSynchronize(us));
    return true;
}

// ---------------------------------------------------------------------------
// Validation: reconstruct L * L^T on the GPU (local rows) and compare with the
// original matrix rows held on the host.
// ---------------------------------------------------------------------------

static bool validateCholesky(const Dist& d, const double* dL, const std::vector<double>& fullL,
                             const std::vector<double>& hOrigLocal, double* hRecon) {
    const size_t n = d.n;
    // full L (padded with identity) on the device
    double* dLfull = nullptr;
    CUDA_CHECK(cudaMalloc(&dLfull, (size_t)d.Npad * d.Npad * sizeof(double)));
    CUDA_CHECK(cudaMemsetAsync(dLfull, 0, (size_t)d.Npad * d.Npad * sizeof(double), d.stream));
    CUDA_CHECK(cudaMemcpy2DAsync(dLfull, (size_t)d.Npad * sizeof(double), fullL.data(),
                                 n * sizeof(double), n * sizeof(double), n, cudaMemcpyHostToDevice,
                                 d.stream));
    pad_identity_kernel<<<(d.Npad + 255) / 256, 256, 0, d.stream>>>(dLfull, d.Npad, (long)n,
                                                                    d.Npad);
    CUDA_CHECK(cudaGetLastError());

    double* dRecon = nullptr;
    CUDA_CHECK(cudaMalloc(&dRecon, (size_t)std::max(d.rowsLocal, 1) * d.Npad * sizeof(double)));
    gemm_nt<4, 0, false, false>(d.stream, d.rowsLocal, d.Npad, d.Npad, 1.0, dL, d.Npad, dLfull,
                                d.Npad, 0.0, dRecon, d.Npad, plainMap(d));
    localRowsToHost(d, dRecon, hRecon);
    CUDA_CHECK(cudaFree(dRecon));
    CUDA_CHECK(cudaFree(dLfull));

    double maxError = 0.0;
    double relError = 0.0;
    const size_t cnt = d.realRowsLocal * n;
#pragma omp parallel for reduction(max : maxError, relError) schedule(static)
    for (size_t i = 0; i < cnt; ++i) {
        const double error = fabs(hRecon[i] - hOrigLocal[i]);
        maxError = std::max(maxError, error);
        const double rel = error / (fabs(hOrigLocal[i]) + 1e-10);
        relError = std::max(relError, rel);
    }
    double g[2] = {maxError, relError};
    MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, g, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));
    maxError = g[0];
    relError = g[1];

    if (d.rank == 0) {
        printf("Max absolute error: %.10e\n", maxError);
        printf("Max relative error: %.10e\n", relError);
    }
    if (relError > 1e-6) {
        if (d.rank == 0) printf("Validation failed: relative error too large\n");
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
    MPI_CHECK(MPI_Init(&argc, &argv));
    Dist d;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &d.rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &d.P));
    const bool root = (d.rank == 0);

    size_t n = 512;
    bool validate = false;
    bool printResults = false;
    int exitCode = 0;
    bool earlyExit = false;

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
            earlyExit = true;
            break;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            exitCode = 1;
            earlyExit = true;
            break;
        }
    }
    if (earlyExit) {
        MPI_Finalize();
        return exitCode;
    }

    setupTopology(d);

    // Size the OpenMP pool: share the node's cores between the ranks on it (unless
    // the user chose explicitly). The host-side OpenMP work is memory-bound
    // packing/comparison, so a moderate pool is enough; oversubscribing spinning
    // threads would stall the launch/communication thread and the parallel-region
    // overhead of very large pools dominates small runs.
    if (getenv("OMP_NUM_THREADS") == nullptr) {
        const long online = sysconf(_SC_NPROCESSORS_ONLN);
        int t = (int)std::max<long>(1, online / std::max(d.nodeSize, 1));
        t = std::min(t, omp_get_num_procs());
        t = std::min(t, 16);
        omp_set_num_threads(std::max(t, 1));
    }

    // Select the GPU (one per rank, NUMA-aware) and create the streams
    {
        CUDA_CHECK(cudaSetDevice(selectDevice(d)));
        int prLow = 0, prHigh = 0;
        CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&prLow, &prHigh));
        CUDA_CHECK(cudaStreamCreateWithPriority(&d.stream, cudaStreamNonBlocking, prLow));
        CUDA_CHECK(cudaStreamCreateWithPriority(&d.pstream, cudaStreamNonBlocking, prHigh));
        CUDA_CHECK(cudaStreamCreateWithPriority(&d.cstream, cudaStreamNonBlocking, prHigh));
        CUDA_CHECK(cudaFuncSetAttribute(potrf64_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        (int)POTRF_SMEM));
        // Force module loading of all kernels now (CUDA lazy loading would otherwise
        // put the first-launch cost inside the timed region).
        cudaFuncAttributes fa;
        CUDA_CHECK(cudaFuncGetAttributes(&fa, gemm_nt_kernel<4, 0, false, false>));
        CUDA_CHECK(cudaFuncGetAttributes(&fa, gemm_nt_kernel<4, 1, false, false>));
        CUDA_CHECK(cudaFuncGetAttributes(&fa, gemm_nt_kernel<4, 2, true, true>));
        CUDA_CHECK(cudaFuncGetAttributes(&fa, gemm_nt_kernel<1, 0, false, false>));
        CUDA_CHECK(cudaFuncGetAttributes(&fa, potrf64_kernel));
        CUDA_CHECK(cudaFuncGetAttributes(&fa, fix_diagonal_kernel));
        CUDA_CHECK(cudaFuncGetAttributes(&fa, zero_upper_kernel));
        CUDA_CHECK(cudaFuncGetAttributes(&fa, pad_identity_kernel));
    }

    if (root) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    setupDist(d, n);

    // Device matrix: local block rows, padded leading dimension
    double* dA = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, (size_t)std::max(d.rowsLocal, 1) * d.Npad * sizeof(double)));
    CUDA_CHECK(cudaMemset(dA, 0, (size_t)std::max(d.rowsLocal, 1) * d.Npad * sizeof(double)));

    // Host buffer for the local real rows (gather / validation)
    double* hLocal = nullptr;
    CUDA_CHECK(cudaHostAlloc(&hLocal, std::max<size_t>(d.realRowsLocal * n, 1) * sizeof(double),
                             cudaHostAllocDefault));
    std::vector<double> hOrigLocal;

    // Generate positive definite matrix
    if (root) printf("Generating positive definite matrix...\n");
    generatePositiveDefiniteMatrix(d, dA);

    if (validate) {
        localRowsToHost(d, dA, hLocal);
        hOrigLocal.assign(hLocal, hLocal + d.realRowsLocal * n);  // Save original for validation
    }

    // Workspace
    Workspace w;
    const size_t tile = (size_t)d.nb * d.nb;
    const int nsb = d.nb / SB;
    const size_t panelCount = (size_t)d.Npad * d.nb;
    const size_t diagCount = tile + (size_t)nsb * SB * SB;
    for (int b = 0; b <= LOOKAHEAD; ++b) CUDA_CHECK(cudaMalloc(&w.dPanel[b], panelCount * sizeof(double)));
    w.evPanelReady.resize(d.nBlocks);
    w.evPrio.resize(d.nBlocks);
    w.evUpdateDone.resize(d.nBlocks);
    for (int k = 0; k < d.nBlocks; ++k) {
        CUDA_CHECK(cudaEventCreateWithFlags(&w.evPanelReady[k], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&w.evPrio[k], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&w.evUpdateDone[k], cudaEventDisableTiming));
    }
    CUDA_CHECK(cudaMalloc(&w.dPanelLoc, (size_t)std::max(d.rowsLocal, 1) * d.nb * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&w.dLkk, tile * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&w.dInv64, (size_t)nsb * SB * SB * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&w.dTmp, (size_t)std::max(d.rowsLocal, d.nb) * SB * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&w.dErr, sizeof(int)));

    // Node-shared window (allocated by the node leader, mapped by all ranks on the
    // node) holding two panel buffers and two diagonal-block buffers.
    {
        const MPI_Aint mySize =
            (d.nodeRank == 0) ? (MPI_Aint)((2 * panelCount + 2 * diagCount) * sizeof(double)) : 0;
        double* base = nullptr;
        MPI_CHECK(MPI_Win_allocate_shared(mySize, sizeof(double), MPI_INFO_NULL, d.nodeComm, &base,
                                          &w.win));
        MPI_Aint sz = 0;
        int disp = 0;
        MPI_CHECK(MPI_Win_shared_query(w.win, 0, &sz, &disp, &w.winBase));
        w.winBytes = (size_t)sz;
        // pin the mapping so the GPU can DMA to/from it (fall back to pageable copies)
        if (cudaHostRegister(w.winBase, w.winBytes, cudaHostRegisterPortable) == cudaSuccess) {
            w.winRegistered = true;
        } else {
            (void)cudaGetLastError();
        }
        w.shPanel[0] = w.winBase;
        w.shPanel[1] = w.winBase + panelCount;
        w.shDiag[0] = w.winBase + 2 * panelCount;
        w.shDiag[1] = w.shDiag[0] + diagCount;
        MPI_CHECK(MPI_Win_lock_all(MPI_MODE_NOCHECK, w.win));
    }

    // Perform Cholesky decomposition
    if (root) printf("Computing Cholesky decomposition...\n");
    // Create the OpenMP worker pool now so that thread start-up is not timed
#pragma omp parallel
    {
        volatile int sink = omp_get_thread_num();
        (void)sink;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto start = std::chrono::high_resolution_clock::now();

    int failIndex = -1;
    bool success = choleskyDecomposition(d, dA, w, failIndex);

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (!success) {
        if (root) {
            printf("Error: Matrix is not positive definite at diagonal element %d\n", failIndex);
            printf("Cholesky decomposition failed\n");
        }
        MPI_Finalize();
        return 1;
    }

    if (root) {
        printf("Computation time: %ld ms\n", (long)duration.count());
        // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Collect the full factor on rank 0 (needed for result printing / validation)
    std::vector<double> fullL;
    if (printResults || validate) {
        localRowsToHost(d, dA, hLocal);
        gatherToRoot(d, hLocal, fullL);
    }

    // Print results for external validation
    if (printResults && root) {
        print_results(fullL, "CholeskyL");
    }

    // Validation
    if (validate) {
        if (root) printf("Validating result...\n");
        if (!root) fullL.resize(n * n);
        MPI_CHECK(MPI_Bcast(fullL.data(), (int)n, d.rowType, 0, MPI_COMM_WORLD));
        bool valid = validateCholesky(d, dA, fullL, hOrigLocal, hLocal);
        if (root) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        exitCode = valid ? 0 : 1;
    }

    MPI_CHECK(MPI_Win_unlock_all(w.win));
    if (w.winRegistered) CUDA_CHECK(cudaHostUnregister(w.winBase));
    MPI_CHECK(MPI_Win_free(&w.win));
    CUDA_CHECK(cudaFree(w.dErr));
    CUDA_CHECK(cudaFree(w.dTmp));
    CUDA_CHECK(cudaFree(w.dInv64));
    CUDA_CHECK(cudaFree(w.dLkk));
    CUDA_CHECK(cudaFree(w.dPanelLoc));
    for (int k = 0; k < d.nBlocks; ++k) {
        CUDA_CHECK(cudaEventDestroy(w.evUpdateDone[k]));
        CUDA_CHECK(cudaEventDestroy(w.evPrio[k]));
        CUDA_CHECK(cudaEventDestroy(w.evPanelReady[k]));
    }
    for (int b = 0; b <= LOOKAHEAD; ++b) CUDA_CHECK(cudaFree(w.dPanel[b]));
    CUDA_CHECK(cudaFreeHost(hLocal));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaStreamDestroy(d.cstream));
    CUDA_CHECK(cudaStreamDestroy(d.pstream));
    CUDA_CHECK(cudaStreamDestroy(d.stream));
    MPI_Type_free(&d.rowType);
    MPI_Comm_free(&d.nodeComm);
    MPI_Finalize();
    return exitCode;
}
