// Floyd-Warshall All-Pairs Shortest Path Benchmark
//
// Hybrid MPI + OpenMP + CUDA implementation:
//   * MPI    : the distance/path matrices are distributed by contiguous block-rows
//              over the ranks, one GPU per rank. Each blocked step exchanges only the
//              pivot block-row: intra-node through an MPI-3 shared memory window that
//              is registered with CUDA, between nodes through a broadcast among the
//              node leaders. The owner of the next pivot block-row computes it first
//              (look-ahead), so the exchange overlaps with the bulk of the step, and
//              the payload is losslessly packed into the narrowest integer type that
//              fits its current maximum.
//   * CUDA   : the blocked Floyd-Warshall algorithm (3 phases per step) runs entirely
//              on the GPU with shared-memory tiles and register blocking; two streams
//              keep the look-ahead and the bulk phases concurrent.
//   * OpenMP : parallel matrix initialization (via analytic jump-ahead of the rand_r
//              LCG stream so that every rank/thread reproduces the exact serial
//              sequence) and parallel host-side packing of the results.
//
// The computed distance matrix is bit-identical to the serial reference version.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

// ---------------------------------------------------------------------------
// GPU blocking parameters
// ---------------------------------------------------------------------------
// BF    : blocking factor (tile edge length) of the blocked Floyd-Warshall
// TX/TY : CUDA thread block shape; every thread handles RPT x CPT elements
#define BF 64
#define TX 32
#define TY 16
#define CPT (BF / TX)  // columns per thread (2)
#define RPT (BF / TY)  // rows per thread    (4)

#define CUDA_CHECK(call)                                                                       \
    do {                                                                                       \
        const cudaError_t err__ = (call);                                                      \
        if (err__ != cudaSuccess) {                                                            \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__,         \
                    cudaGetErrorString(err__));                                                \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                      \
        }                                                                                      \
    } while (0)

// Index calculation for flattened 2D array
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

// ---------------------------------------------------------------------------
// Exact (bit-identical) reproduction of glibc's rand_r including jump-ahead
// ---------------------------------------------------------------------------
namespace rng {
constexpr unsigned int LCG_A = 1103515245u;
constexpr unsigned int LCG_C = 12345u;

// One rand_r() call advances the state by three LCG steps.
inline unsigned int next_value(unsigned int& state) noexcept {
    unsigned int result;
    state = state * LCG_A + LCG_C;
    result = (state / 65536u) % 2048u;
    state = state * LCG_A + LCG_C;
    result <<= 10;
    result ^= (state / 65536u) % 1024u;
    state = state * LCG_A + LCG_C;
    result <<= 10;
    result ^= (state / 65536u) % 1024u;
    return result;
}

// Affine map x -> a*x + c; composition allows O(log k) jump-ahead.
struct Affine {
    unsigned int a;
    unsigned int c;
};

inline Affine compose(const Affine& f, const Affine& g) noexcept {
    // apply g first, then f
    return Affine{f.a * g.a, f.a * g.c + f.c};
}

inline Affine power(Affine f, size_t k) noexcept {
    Affine r{1u, 0u};
    while (k > 0) {
        if (k & 1) r = compose(r, f);
        f = compose(f, f);
        k >>= 1;
    }
    return r;
}

// State after `calls` invocations of rand_r starting from `seed`.
inline unsigned int state_after(const unsigned int seed, const size_t calls) noexcept {
    const Affine step{LCG_A, LCG_C};
    const Affine per_call = power(step, 3);       // one rand_r call
    const Affine jump = power(per_call, calls);
    return jump.a * seed + jump.c;
}
}  // namespace rng

// ---------------------------------------------------------------------------
// CUDA kernels: blocked Floyd-Warshall
// ---------------------------------------------------------------------------
// Every tile is BF x BF and is processed by a (TX, TY) thread block where each
// thread owns RPT x CPT elements, kept in registers during the k-loop.
//
// Within one tile-local k-step the pivot row k and pivot column k of the tile
// never change (their diagonal element is zero), therefore a single barrier per
// k-step is sufficient - exactly as in the serial algorithm.

__global__ __launch_bounds__(TX* TY) void fwPhase1(unsigned int* __restrict__ dist,
                                                   unsigned int* __restrict__ path,
                                                   unsigned int* __restrict__ piv, const int N,
                                                   const int kb, const int lrow0) {
    __shared__ unsigned int s[BF][BF];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int col0 = kb * BF;

    unsigned int p[RPT][CPT];

#pragma unroll
    for (int r = 0; r < RPT; ++r) {
        const int rr = ty + r * TY;
        const size_t base = (size_t)(lrow0 + rr) * N + col0;
#pragma unroll
        for (int c = 0; c < CPT; ++c) {
            const int cc = tx + c * TX;
            s[rr][cc] = dist[base + cc];
            p[r][c] = path[base + cc];
        }
    }
    __syncthreads();

    for (int kk = 0; kk < BF; ++kk) {
#pragma unroll
        for (int r = 0; r < RPT; ++r) {
            const int rr = ty + r * TY;
            const unsigned int dik = s[rr][kk];
#pragma unroll
            for (int c = 0; c < CPT; ++c) {
                const int cc = tx + c * TX;
                const unsigned int nd = dik + s[kk][cc];
                if (nd < s[rr][cc]) {
                    s[rr][cc] = nd;
                    p[r][c] = col0 + kk;
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int r = 0; r < RPT; ++r) {
        const int rr = ty + r * TY;
        const size_t base = (size_t)(lrow0 + rr) * N + col0;
#pragma unroll
        for (int c = 0; c < CPT; ++c) {
            const int cc = tx + c * TX;
            dist[base + cc] = s[rr][cc];
            path[base + cc] = p[r][c];
            piv[rr * BF + cc] = s[rr][cc];
        }
    }
}

// Pivot block-row: blocks (kb, jb), jb != kb. Only run by the owner of block-row kb.
__global__ __launch_bounds__(TX* TY) void fwPhase2Row(unsigned int* __restrict__ dist,
                                                      unsigned int* __restrict__ path,
                                                      const unsigned int* __restrict__ piv,
                                                      const int N, const int kb, const int lrow0) {
    const int jb = blockIdx.x;
    if (jb == kb) return;

    __shared__ unsigned int sp[BF][BF];
    __shared__ unsigned int sb[BF][BF];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int col0 = jb * BF;

    unsigned int p[RPT][CPT];

#pragma unroll
    for (int r = 0; r < RPT; ++r) {
        const int rr = ty + r * TY;
        const size_t base = (size_t)(lrow0 + rr) * N + col0;
#pragma unroll
        for (int c = 0; c < CPT; ++c) {
            const int cc = tx + c * TX;
            sp[rr][cc] = piv[rr * BF + cc];
            sb[rr][cc] = dist[base + cc];
            p[r][c] = path[base + cc];
        }
    }
    __syncthreads();

    for (int kk = 0; kk < BF; ++kk) {
#pragma unroll
        for (int r = 0; r < RPT; ++r) {
            const int rr = ty + r * TY;
            const unsigned int dik = sp[rr][kk];
#pragma unroll
            for (int c = 0; c < CPT; ++c) {
                const int cc = tx + c * TX;
                const unsigned int nd = dik + sb[kk][cc];
                if (nd < sb[rr][cc]) {
                    sb[rr][cc] = nd;
                    p[r][c] = kb * BF + kk;
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int r = 0; r < RPT; ++r) {
        const int rr = ty + r * TY;
        const size_t base = (size_t)(lrow0 + rr) * N + col0;
#pragma unroll
        for (int c = 0; c < CPT; ++c) {
            const int cc = tx + c * TX;
            dist[base + cc] = sb[rr][cc];
            path[base + cc] = p[r][c];
        }
    }
}

// Pivot block-column: blocks (ib, kb) for all locally owned block-rows ib != kb.
__global__ __launch_bounds__(TX* TY) void fwPhase2Col(unsigned int* __restrict__ dist,
                                                      unsigned int* __restrict__ path,
                                                      const unsigned int* __restrict__ prow,
                                                      const int N, const int kb, const int ibStart,
                                                      const int skipLocalBlock) {
    const int ib = ibStart + (int)blockIdx.x;
    if (ib == skipLocalBlock) return;

    __shared__ unsigned int sp[BF][BF];
    __shared__ unsigned int sb[BF][BF];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int col0 = kb * BF;

    unsigned int p[RPT][CPT];

#pragma unroll
    for (int r = 0; r < RPT; ++r) {
        const int rr = ty + r * TY;
        const size_t base = (size_t)(ib * BF + rr) * N + col0;
#pragma unroll
        for (int c = 0; c < CPT; ++c) {
            const int cc = tx + c * TX;
            sp[rr][cc] = prow[(size_t)rr * N + col0 + cc];
            sb[rr][cc] = dist[base + cc];
            p[r][c] = path[base + cc];
        }
    }
    __syncthreads();

    for (int kk = 0; kk < BF; ++kk) {
#pragma unroll
        for (int r = 0; r < RPT; ++r) {
            const int rr = ty + r * TY;
            const unsigned int dik = sb[rr][kk];
#pragma unroll
            for (int c = 0; c < CPT; ++c) {
                const int cc = tx + c * TX;
                const unsigned int nd = dik + sp[kk][cc];
                if (nd < sb[rr][cc]) {
                    sb[rr][cc] = nd;
                    p[r][c] = kb * BF + kk;
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int r = 0; r < RPT; ++r) {
        const int rr = ty + r * TY;
        const size_t base = (size_t)(ib * BF + rr) * N + col0;
#pragma unroll
        for (int c = 0; c < CPT; ++c) {
            const int cc = tx + c * TX;
            dist[base + cc] = sb[rr][cc];
            path[base + cc] = p[r][c];
        }
    }
}

// Remaining blocks (ib, jb), ib and jb != kb. This is the compute-dominant phase:
// both operand tiles stay constant during the k-loop, so no barriers are needed and
// the updated block is kept entirely in registers. A 4x4 register tile per thread
// (with 128 bit accesses) minimises the shared memory traffic per updated element.
#define P3TX 16
#define P3TY 16
#define P3R 4
#define P3C 4

__global__ __launch_bounds__(P3TX* P3TY) void fwPhase3(unsigned int* __restrict__ dist,
                                                       unsigned int* __restrict__ path,
                                                       const unsigned int* __restrict__ prow,
                                                       const int N, const int kb, const int ibStart,
                                                       const int skipLocalBlock) {
    const int jb = blockIdx.x;
    if (jb == kb) return;
    const int ib = ibStart + (int)blockIdx.y;
    if (ib == skipLocalBlock) return;

    __shared__ unsigned int srow[BF][BF];      // pivot row tile (kb, jb)
    __shared__ unsigned int scol[BF][BF + 4];  // pivot col tile (ib, kb), padded

    const int tid = threadIdx.y * P3TX + threadIdx.x;
    const size_t rowBase = (size_t)(ib * BF) * N;

#pragma unroll
    for (int i = 0; i < (BF * BF / 4) / (P3TX * P3TY); ++i) {
        const int u = tid + i * (P3TX * P3TY);
        const int r = u >> 4;
        const int c = (u & 15) * 4;
        *(uint4*)&srow[r][c] = *(const uint4*)&prow[(size_t)r * N + jb * BF + c];
        *(uint4*)&scol[r][c] = *(const uint4*)&dist[rowBase + (size_t)r * N + kb * BF + c];
    }
    __syncthreads();

    const int row0 = threadIdx.y * P3R;
    const int col0 = threadIdx.x * P3C;

    uint4 d[P3R], p[P3R];
#pragma unroll
    for (int r = 0; r < P3R; ++r) {
        const size_t off = rowBase + (size_t)(row0 + r) * N + jb * BF + col0;
        d[r] = *(const uint4*)&dist[off];
        p[r] = *(const uint4*)&path[off];
    }

    for (int kk = 0; kk < BF; ++kk) {
        const uint4 rw = *(const uint4*)&srow[kk][col0];
        const unsigned int pk = kb * BF + kk;
#pragma unroll
        for (int r = 0; r < P3R; ++r) {
            const unsigned int dik = scol[row0 + r][kk];
            unsigned int nd = dik + rw.x;
            if (nd < d[r].x) { d[r].x = nd; p[r].x = pk; }
            nd = dik + rw.y;
            if (nd < d[r].y) { d[r].y = nd; p[r].y = pk; }
            nd = dik + rw.z;
            if (nd < d[r].z) { d[r].z = nd; p[r].z = pk; }
            nd = dik + rw.w;
            if (nd < d[r].w) { d[r].w = nd; p[r].w = pk; }
        }
    }

#pragma unroll
    for (int r = 0; r < P3R; ++r) {
        const size_t off = rowBase + (size_t)(row0 + r) * N + jb * BF + col0;
        *(uint4*)&dist[off] = d[r];
        *(uint4*)&path[off] = p[r];
    }
}

// ---------------------------------------------------------------------------
// Pivot block-row transport helpers
// ---------------------------------------------------------------------------
// Only the non-padded part of the pivot block-row is exchanged, and it is packed
// into the narrowest integer type that can represent its current maximum (the
// entries of a shortest path matrix only ever decrease). The transform is
// lossless, the reconstructed pivot row is bit-identical on every rank.

__global__ void reduceMaxKernel(const unsigned int* __restrict__ prow, const int N, const int n,
                                unsigned int* __restrict__ out) {
    const int r = blockIdx.y;
    unsigned int m = 0;
    for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < n; c += blockDim.x * gridDim.x) {
        m = max(m, prow[(size_t)r * N + c]);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) m = max(m, __shfl_down_sync(0xffffffffu, m, off));
    if ((threadIdx.x & 31) == 0) atomicMax(out, m);
}

template <typename T>
__global__ void packKernel(const unsigned int* __restrict__ prow, const int N, const int n,
                           T* __restrict__ out) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int r = blockIdx.y;
    if (c < n) out[(size_t)r * n + c] = (T)prow[(size_t)r * N + c];
}

template <typename T>
__global__ void unpackKernel(unsigned int* __restrict__ prow, const int N, const int n,
                             const int rows, const int rowBase, const T* __restrict__ in) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int r = blockIdx.y;
    if (c >= N) return;
    unsigned int v;
    if (r < rows) {
        // real row: padded columns are always INF
        v = (c < n) ? (unsigned int)in[(size_t)r * n + c] : INF;
    } else {
        // padded row: INF everywhere except its own (zero) diagonal element
        v = ((rowBase + r) == c) ? 0u : INF;
    }
    prow[(size_t)r * N + c] = v;
}

// ---------------------------------------------------------------------------
// Distributed driver
// ---------------------------------------------------------------------------
namespace {

int g_rank = 0;
int g_size = 1;
int g_parts = 1;  // ranks that actually own part of the matrix

// A rank needs a certain number of block-rows before distributing pays off: every
// pivot step costs one pivot row exchange, so tiny problems stay on fewer GPUs.
inline int participatingRanks(const int numBlocks) {
    return std::max(1, std::min(g_size, numBlocks / 16));
}

// Row-block ownership: rank r owns global block-rows [blockStart(r), blockStart(r+1))
inline int blockStart(const int r, const int numBlocks) {
    if (r >= g_parts) return numBlocks;
    const long long q = (long long)numBlocks / g_parts;
    const long long rem = (long long)numBlocks % g_parts;
    return (int)(r * q + std::min<long long>(r, rem));
}

inline int blockOwner(const int kb, const int numBlocks) {
    // inverse of blockStart: the last rank whose first block is <= kb
    int lo = 0, hi = g_size - 1;
    while (lo < hi) {
        const int mid = (lo + hi + 1) / 2;
        if (blockStart(mid, numBlocks) <= kb)
            lo = mid;
        else
            hi = mid - 1;
    }
    return lo;
}

}  // namespace

// Initialize the locally owned rows of the distance matrix so that the values are
// identical to the serial reference implementation (rand_r stream jump-ahead).
static void initializeLocalDistance(unsigned int* dist, const size_t n, const size_t N,
                                    const size_t row0, const size_t numRows,
                                    const unsigned int rangeMin, const unsigned int rangeMax) {
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;

#pragma omp parallel for schedule(static)
    for (size_t lr = 0; lr < numRows; ++lr) {
        const size_t gr = row0 + lr;
        unsigned int* row = dist + lr * N;
        if (gr < n) {
            // The serial code draws the values in linear order, row gr consumes
            // the calls [gr*n, gr*n + n).
            unsigned int state = rng::state_after(42u, gr * n);
            for (size_t j = 0; j < n; ++j) {
                row[j] = rangeMin + (unsigned int)(range * rng::next_value(state) / (double)RAND_MAX);
            }
            for (size_t j = n; j < N; ++j) row[j] = INF;  // padding
            row[gr] = 0;                                   // diagonal
        } else {
            for (size_t j = 0; j < N; ++j) row[j] = INF;  // padded row
            row[gr] = 0;                                  // keep pivot steps neutral
        }
    }
}

// path[i][j] = i (matches the serial initializePathMatrix)
static void initializeLocalPath(unsigned int* path, const size_t N, const size_t row0,
                                const size_t numRows) {
#pragma omp parallel for schedule(static)
    for (size_t lr = 0; lr < numRows; ++lr) {
        unsigned int* row = path + lr * N;
        const unsigned int v = (unsigned int)(row0 + lr);
        for (size_t j = 0; j < N; ++j) row[j] = v;
    }
}

bool validateResult(const std::vector<unsigned int>& dist, const size_t numNodes) {
    // Basic sanity checks

    // 1. Diagonal should be zero
    for (size_t i = 0; i < numNodes; ++i) {
        if (dist[idx2(i, i, numNodes)] != 0) {
            printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", i, i);
            return false;
        }
    }

    // 2. Triangle inequality: dist[i][k] + dist[k][j] >= dist[i][j]
    // Check a sample of paths to avoid O(n^3) validation time
    for (size_t i = 0; i < std::min(numNodes, static_cast<size_t>(10)); ++i) {
        for (size_t j = 0; j < std::min(numNodes, static_cast<size_t>(10)); ++j) {
            for (size_t k = 0; k < numNodes; ++k) {
                const unsigned int distIJ = dist[idx2(j, i, numNodes)];
                const unsigned int distIK = dist[idx2(k, i, numNodes)];
                const unsigned int distKJ = dist[idx2(j, k, numNodes)];

                // Check for overflow before addition
                if (distIK < INF && distKJ < INF) {
                    if (distIK + distKJ < distIJ) {
                        printf("Validation failed: triangle inequality violated at [%zu,%zu,%zu]\n",
                               i, j, k);
                        return false;
                    }
                }
            }
        }
    }

    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Number of nodes in the graph (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_size);

    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numNodes = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (g_rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (g_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (g_rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, OpenMP threads: %d\n", g_size, omp_get_max_threads());
    }

    if (numNodes == 0) {
        if (g_rank == 0) printf("Computation time: 0 ms\n");
        MPI_Finalize();
        return 0;
    }

    // -- Padded problem geometry ---------------------------------------------
    const size_t N = ((numNodes + BF - 1) / BF) * BF;  // padded matrix edge
    const int numBlocks = (int)(N / BF);

    g_parts = participatingRanks(numBlocks);
    const bool distributed = (g_parts > 1);
    const bool active = (g_rank < g_parts);  // ranks that own block-rows

    // -- Node topology: one GPU per rank, round-robin within the node --------
    // The pivot row exchange only involves the ranks that own part of the matrix.
    MPI_Comm workComm = MPI_COMM_NULL, nodeComm = MPI_COMM_NULL, leaderComm = MPI_COMM_NULL;
    int localRank = 0, nodeSize = 1;
    std::vector<int> leaderIdxOf(g_size, 0);
    MPI_Comm_split(MPI_COMM_WORLD, active ? 0 : MPI_UNDEFINED, g_rank, &workComm);
    if (active) {
        MPI_Comm_split_type(workComm, MPI_COMM_TYPE_SHARED, g_rank, MPI_INFO_NULL, &nodeComm);
        MPI_Comm_rank(nodeComm, &localRank);
        MPI_Comm_size(nodeComm, &nodeSize);
        // one leader per node performs the inter-node part of the pivot row exchange
        MPI_Comm_split(workComm, localRank == 0 ? 0 : MPI_UNDEFINED, g_rank, &leaderComm);

        // Every rank needs to know, for any other rank, the index of its node leader
        // within the leader communicator (the root of the inter-node broadcast).
        int myLeaderIdx = 0;
        if (leaderComm != MPI_COMM_NULL) MPI_Comm_rank(leaderComm, &myLeaderIdx);
        MPI_Bcast(&myLeaderIdx, 1, MPI_INT, 0, nodeComm);
        MPI_Allgather(&myLeaderIdx, 1, MPI_INT, leaderIdxOf.data(), 1, MPI_INT, workComm);
    }

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        if (g_rank == 0) fprintf(stderr, "No CUDA device available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice((active ? localRank : g_rank) % deviceCount));

    const int myB0 = blockStart(g_rank, numBlocks);
    const int myB1 = blockStart(g_rank + 1, numBlocks);
    const int myNB = myB1 - myB0;
    const size_t myRow0 = (size_t)myB0 * BF;
    const size_t myRows = (size_t)myNB * BF;

    // -- Allocation ----------------------------------------------------------
    unsigned int* d_dist = nullptr;
    unsigned int* d_path = nullptr;
    unsigned int* d_piv = nullptr;       // BF x BF pivot block
    unsigned int* d_prow = nullptr;      // BF x N pivot block-row (current step)
    unsigned int* d_prowNext = nullptr;  // BF x N pivot block-row (look-ahead)
    unsigned int* d_max = nullptr;       // maximum of the pivot block-row
    void* d_pack = nullptr;              // packed pivot block-row (device)
    void* h_slot[2] = {nullptr, nullptr};  // packed pivot block-row (shared window, 2 slots)

    const size_t localElems = myRows * N;
    if (localElems > 0) {
        CUDA_CHECK(cudaMalloc(&d_dist, localElems * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&d_path, localElems * sizeof(unsigned int)));
    }
    const size_t packBytes = (size_t)BF * numNodes * sizeof(unsigned int);
    CUDA_CHECK(cudaMalloc(&d_piv, (size_t)BF * BF * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_prow, (size_t)BF * N * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_prowNext, (size_t)BF * N * sizeof(unsigned int)));
    if (distributed && active) {
        CUDA_CHECK(cudaMalloc(&d_max, sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&d_pack, packBytes));
    }

    // Intra-node exchange buffer: an MPI-3 shared memory window that is mapped into
    // every rank of the node and registered with CUDA, so that the owner's D2H copy
    // is immediately visible to its node neighbours without any further copy.
    // Layout: two slots of [16 byte header | packed pivot block-row].
    MPI_Win win = MPI_WIN_NULL;
    char* winBase = nullptr;
    bool winRegistered = false;
    const size_t slotBytes = ((packBytes + 16 + 4095) / 4096) * 4096;
    if (distributed && active) {
        MPI_Aint winSize = (localRank == 0) ? (MPI_Aint)(2 * slotBytes + 4096) : 0;
        void* ptr = nullptr;
        MPI_Win_allocate_shared(winSize, 1, MPI_INFO_NULL, nodeComm, &ptr, &win);
        MPI_Aint qsize = 0;
        int qdisp = 0;
        MPI_Win_shared_query(win, 0, &qsize, &qdisp, &ptr);
        winBase = (char*)(((uintptr_t)ptr + 4095) & ~(uintptr_t)4095);
        h_slot[0] = winBase;
        h_slot[1] = winBase + slotBytes;
        winRegistered = (cudaHostRegister(winBase, 2 * slotBytes, cudaHostRegisterDefault) ==
                         cudaSuccess);
        if (!winRegistered) cudaGetLastError();  // pageable fallback, still correct
        MPI_Win_lock_all(MPI_MODE_NOCHECK, win);
    }

    // Two streams: the look-ahead stream produces and consumes the pivot block-row
    // (it is on the critical path and gets the higher priority), the bulk stream
    // runs the remaining block updates concurrently with the MPI broadcast.
    int prioLow = 0, prioHigh = 0;
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&prioLow, &prioHigh));
    cudaStream_t streamC, streamB;
    CUDA_CHECK(cudaStreamCreateWithPriority(&streamC, cudaStreamNonBlocking, prioHigh));
    CUDA_CHECK(cudaStreamCreateWithPriority(&streamB, cudaStreamNonBlocking, prioLow));
    // evBulk is indexed by step parity: the pivot row buffer written during step kb
    // is the one that was read by the bulk kernels of step kb-1.
    cudaEvent_t evBulk[2], evProw, h2dDone[2];
    CUDA_CHECK(cudaEventCreateWithFlags(&evBulk[0], cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&evBulk[1], cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&evProw, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&h2dDone[0], cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&h2dDone[1], cudaEventDisableTiming));

    // -- Initialization (OpenMP, distributed) --------------------------------
    if (g_rank == 0) printf("Initializing graph...\n");
    {
        std::vector<unsigned int> h_init(localElems);
        if (localElems > 0) {
            initializeLocalDistance(h_init.data(), numNodes, N, myRow0, myRows, 1, MAX_DISTANCE);
            CUDA_CHECK(cudaMemcpy(d_dist, h_init.data(), localElems * sizeof(unsigned int),
                                  cudaMemcpyHostToDevice));
            initializeLocalPath(h_init.data(), N, myRow0, myRows);
            CUDA_CHECK(cudaMemcpy(d_path, h_init.data(), localElems * sizeof(unsigned int),
                                  cudaMemcpyHostToDevice));
        }
    }

    // -- Blocked Floyd-Warshall ----------------------------------------------
    if (g_rank == 0) printf("Computing shortest paths...\n");

    const dim3 threads(TX, TY);
    const size_t prowElems = (size_t)BF * N;
    const int packThreads = 256;
    const dim3 packGrid((unsigned)((numNodes + packThreads - 1) / packThreads), BF);
    const dim3 unpackGrid((unsigned)((N + packThreads - 1) / packThreads), BF);

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    // Number of non-padded rows of pivot block-row kb
    auto pivotRows = [&](const int kb) {
        return (int)std::min<size_t>(BF, numNodes - (size_t)kb * BF);
    };

    // Updates the locally owned block-rows [lb, ub) for pivot step kb
    // (pivot column blocks + all remaining blocks) on the bulk stream.
    auto updateRowRange = [&](const int kb, const int lb, const int ub, const int skipLocal,
                              cudaStream_t st) {
        if (ub <= lb) return;
        fwPhase2Col<<<dim3(ub - lb), threads, 0, st>>>(d_dist, d_path, d_prow, (int)N, kb, lb,
                                                       skipLocal);
        fwPhase3<<<dim3(numBlocks, ub - lb), dim3(P3TX, P3TY), 0, st>>>(d_dist, d_path, d_prow,
                                                                        (int)N, kb, lb, skipLocal);
    };

    // Owner side: compute pivot block-row kb (phases 1 and 2) into d_prowNext and,
    // for multi-rank runs, pack it (losslessly, see above) into d_pack.
    int g_width = 4;          // byte width used for the last pivot row exchange
    cudaEvent_t evPrevBulk = nullptr;  // completion of the previous step's bulk kernels
    auto payloadOf = [&](const int buf) { return (void*)((char*)h_slot[buf] + 16); };

    auto producePivotRow = [&](const int kb) -> void {
        const int lrow0 = (int)((size_t)kb * BF - myRow0);
        fwPhase1<<<1, threads, 0, streamC>>>(d_dist, d_path, d_piv, (int)N, kb, lrow0);
        fwPhase2Row<<<dim3(numBlocks), threads, 0, streamC>>>(d_dist, d_path, d_piv, (int)N, kb,
                                                              lrow0);
        CUDA_CHECK(cudaMemcpyAsync(d_prowNext, d_dist + (size_t)lrow0 * N,
                                   prowElems * sizeof(unsigned int), cudaMemcpyDeviceToDevice,
                                   streamC));
        CUDA_CHECK(cudaEventRecord(evProw, streamC));
        if (!distributed) return;

        const int rows = pivotRows(kb);
        unsigned int hostMax = 0;
        CUDA_CHECK(cudaMemsetAsync(d_max, 0, sizeof(unsigned int), streamC));
        reduceMaxKernel<<<dim3(packGrid.x, rows), packThreads, 0, streamC>>>(
            d_prowNext, (int)N, (int)numNodes, d_max);
        CUDA_CHECK(cudaMemcpyAsync(&hostMax, d_max, sizeof(unsigned int), cudaMemcpyDeviceToHost,
                                   streamC));
        CUDA_CHECK(cudaStreamSynchronize(streamC));

        const int width = (hostMax <= 0xFFu) ? 1 : ((hostMax <= 0xFFFFu) ? 2 : 4);
        const dim3 grid(packGrid.x, rows);
        if (width == 1) {
            packKernel<unsigned char><<<grid, packThreads, 0, streamC>>>(
                d_prowNext, (int)N, (int)numNodes, (unsigned char*)d_pack);
        } else if (width == 2) {
            packKernel<unsigned short><<<grid, packThreads, 0, streamC>>>(
                d_prowNext, (int)N, (int)numNodes, (unsigned short*)d_pack);
        } else {
            packKernel<unsigned int><<<grid, packThreads, 0, streamC>>>(
                d_prowNext, (int)N, (int)numNodes, (unsigned int*)d_pack);
        }
        g_width = width;
    };

    // Receiver side: rebuild the pivot block-row (including its padded part) in
    // d_prowNext from the shared window slot.
    auto consumePivotRow = [&](const int kb, const int buf, const int width) {
        const int rows = pivotRows(kb);
        CUDA_CHECK(cudaMemcpyAsync(d_pack, payloadOf(buf), (size_t)rows * numNodes * width,
                                   cudaMemcpyHostToDevice, streamC));
        CUDA_CHECK(cudaEventRecord(h2dDone[buf], streamC));
        // d_prowNext is the buffer the previous step's bulk kernels were reading
        if (evPrevBulk) CUDA_CHECK(cudaStreamWaitEvent(streamC, evPrevBulk, 0));
        if (width == 1) {
            unpackKernel<unsigned char><<<unpackGrid, packThreads, 0, streamC>>>(
                d_prowNext, (int)N, (int)numNodes, rows, kb * BF, (const unsigned char*)d_pack);
        } else if (width == 2) {
            unpackKernel<unsigned short><<<unpackGrid, packThreads, 0, streamC>>>(
                d_prowNext, (int)N, (int)numNodes, rows, kb * BF, (const unsigned short*)d_pack);
        } else {
            unpackKernel<unsigned int><<<unpackGrid, packThreads, 0, streamC>>>(
                d_prowNext, (int)N, (int)numNodes, rows, kb * BF, (const unsigned int*)d_pack);
        }
        CUDA_CHECK(cudaEventRecord(evProw, streamC));
    };

    // Broadcast pivot block-row kb from its owner to all ranks: the owner stages it
    // in the shared window slot of its node, the node leaders forward it to the other
    // nodes, and every rank then uploads it from its own node's window.
    auto exchangePivotRow = [&](const int kb, const int owner, const int buf) {

        // (1) nobody must still be reading the slot that is about to be overwritten
        CUDA_CHECK(cudaEventSynchronize(h2dDone[buf]));
        MPI_Barrier(nodeComm);

        // (2) the owner publishes header + payload in its node's window
        if (g_rank == owner) {
            CUDA_CHECK(cudaMemcpyAsync(payloadOf(buf), d_pack,
                                       (size_t)pivotRows(kb) * numNodes * g_width,
                                       cudaMemcpyDeviceToHost, streamC));
            CUDA_CHECK(cudaStreamSynchronize(streamC));
            *(int*)h_slot[buf] = g_width;
        }
        MPI_Win_sync(win);
        MPI_Barrier(nodeComm);

        // (3) inter-node forwarding between the node leaders
        if (leaderComm != MPI_COMM_NULL && g_parts > nodeSize) {
            const int root = leaderIdxOf[owner];
            MPI_Bcast(h_slot[buf], 4, MPI_BYTE, root, leaderComm);
            MPI_Bcast(payloadOf(buf), (int)((size_t)pivotRows(kb) * numNodes * *(int*)h_slot[buf]),
                      MPI_BYTE, root, leaderComm);
        }
        MPI_Win_sync(win);
        MPI_Barrier(nodeComm);

        // (4) everybody but the owner rebuilds the pivot row on its GPU
        g_width = *(int*)h_slot[buf];
        if (g_rank != owner) consumePivotRow(kb, buf, g_width);
    };

    // Prologue: pivot block-row 0 (nothing can overlap with it yet).
    {
        const int owner = blockOwner(0, numBlocks);
        if (g_rank == owner) producePivotRow(0);
        if (distributed && active) exchangePivotRow(0, owner, 0);
        std::swap(d_prow, d_prowNext);
    }

    for (int kb = 0; kb < numBlocks; ++kb) {
        const int nb = kb + 1;
        const int nextOwner = (nb < numBlocks) ? blockOwner(nb, numBlocks) : -1;
        const bool iOwnNext = (nextOwner == g_rank);
        const int pivLocal = (kb >= myB0 && kb < myB1) ? (kb - myB0) : -1;
        const int nextLocal = iOwnNext ? (nb - myB0) : -1;
        const int buf = nb & 1;

        // Look-ahead: the owner of the next pivot block-row updates that row first
        // (it only depends on the previous step) so that the broadcast of the next
        // pivot row can overlap with the bulk of the current step.
        evPrevBulk = evBulk[(kb + 1) & 1];  // recorded for step kb-1
        if (iOwnNext) {
            CUDA_CHECK(cudaStreamWaitEvent(streamC, evPrevBulk, 0));
            updateRowRange(kb, nextLocal, nextLocal + 1, pivLocal, streamC);
        }

        // Bulk of the step on the second stream: all remaining local block-rows.
        CUDA_CHECK(cudaStreamWaitEvent(streamB, evProw, 0));
        if (iOwnNext) {
            updateRowRange(kb, 0, nextLocal, pivLocal, streamB);
            updateRowRange(kb, nextLocal + 1, myNB, pivLocal, streamB);
        } else {
            updateRowRange(kb, 0, myNB, pivLocal, streamB);
        }
        CUDA_CHECK(cudaEventRecord(evBulk[kb & 1], streamB));

        if (nb < numBlocks) {
            // Produce / exchange the next pivot block-row while the bulk stream runs.
            if (iOwnNext) producePivotRow(nb);
            if (distributed && active) exchangePivotRow(nb, nextOwner, buf);
            std::swap(d_prow, d_prowNext);
        }
    }

    CUDA_CHECK(cudaStreamSynchronize(streamC));
    CUDA_CHECK(cudaStreamSynchronize(streamB));
    MPI_Barrier(MPI_COMM_WORLD);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (g_rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate operations per second
        // Floyd-Warshall has O(n³) complexity
        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);
    }

    // -- Collect the result on rank 0 (only when it is actually needed) -------
    if (printResults || validate) {
        // rows of the unpadded matrix owned by this rank
        const size_t r0 = std::min(myRow0, numNodes);
        const size_t r1 = std::min(myRow0 + myRows, numNodes);
        const size_t nrows = (r1 > r0) ? (r1 - r0) : 0;

        std::vector<unsigned int> sendBuf(nrows * numNodes);
        if (nrows > 0) {
            std::vector<unsigned int> h_local(nrows * N);
            CUDA_CHECK(cudaMemcpy(h_local.data(), d_dist + (r0 - myRow0) * N,
                                  nrows * N * sizeof(unsigned int), cudaMemcpyDeviceToHost));
#pragma omp parallel for schedule(static)
            for (size_t r = 0; r < nrows; ++r) {
                memcpy(sendBuf.data() + r * numNodes, h_local.data() + r * N,
                       numNodes * sizeof(unsigned int));
            }
        }

        std::vector<unsigned int> full;
        std::vector<int> counts(g_size), displs(g_size);
        for (int r = 0; r < g_size; ++r) {
            const size_t s0 = std::min((size_t)blockStart(r, numBlocks) * BF, numNodes);
            const size_t s1 = std::min((size_t)blockStart(r + 1, numBlocks) * BF, numNodes);
            counts[r] = (int)((s1 > s0 ? s1 - s0 : 0) * numNodes);
            displs[r] = (int)(s0 * numNodes);
        }
        if (g_rank == 0) full.resize(numNodes * numNodes);
        MPI_Gatherv(sendBuf.data(), (int)(nrows * numNodes), MPI_UNSIGNED,
                    g_rank == 0 ? full.data() : nullptr, counts.data(), displs.data(), MPI_UNSIGNED,
                    0, MPI_COMM_WORLD);

        if (g_rank == 0) {
            // Print results for external validation (integer hash-based)
            if (printResults) {
                print_results_int(full, "DistanceMatrix");
            }

            if (validate) {
                printf("Validating result...\n");
                const bool valid = validateResult(full, numNodes);
                printf(valid ? "Validation: PASSED\n" : "Validation: FAILED\n");
                if (!valid) {
                    MPI_Abort(MPI_COMM_WORLD, 1);
                    return 1;
                }
            }
        }
    }

    CUDA_CHECK(cudaEventDestroy(h2dDone[0]));
    CUDA_CHECK(cudaEventDestroy(h2dDone[1]));
    CUDA_CHECK(cudaEventDestroy(evBulk[0]));
    CUDA_CHECK(cudaEventDestroy(evBulk[1]));
    CUDA_CHECK(cudaEventDestroy(evProw));
    CUDA_CHECK(cudaStreamDestroy(streamC));
    CUDA_CHECK(cudaStreamDestroy(streamB));
    if (d_dist) CUDA_CHECK(cudaFree(d_dist));
    if (d_path) CUDA_CHECK(cudaFree(d_path));
    CUDA_CHECK(cudaFree(d_piv));
    CUDA_CHECK(cudaFree(d_prow));
    CUDA_CHECK(cudaFree(d_prowNext));
    if (d_max) CUDA_CHECK(cudaFree(d_max));
    if (d_pack) CUDA_CHECK(cudaFree(d_pack));
    if (winRegistered) CUDA_CHECK(cudaHostUnregister(winBase));
    if (win != MPI_WIN_NULL) {
        MPI_Win_unlock_all(win);
        MPI_Win_free(&win);
    }
    if (leaderComm != MPI_COMM_NULL) MPI_Comm_free(&leaderComm);
    if (nodeComm != MPI_COMM_NULL) MPI_Comm_free(&nodeComm);
    if (workComm != MPI_COMM_NULL) MPI_Comm_free(&workComm);

    MPI_Finalize();
    return 0;
}
