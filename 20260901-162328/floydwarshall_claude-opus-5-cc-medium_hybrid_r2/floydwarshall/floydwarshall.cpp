// Hybrid MPI + OpenMP + CUDA implementation of the Floyd-Warshall benchmark.
//
// Parallelization strategy
//   * MPI     : the (padded) distance matrix is distributed by row-blocks over the
//               ranks, one GPU per rank.  Every block-step the "pivot" row panel is
//               broadcast from its owner to all other ranks.
//   * CUDA    : the blocked Floyd-Warshall algorithm (dependent / partially
//               dependent / independent phases) runs entirely on the GPU using
//               shared-memory tiles.
//   * OpenMP  : host side work (random initialization, packing, validation).
//
// The algorithm performs exactly the same sequence of relaxations as the classic
// triple loop, therefore both the distance and the predecessor matrices are
// bit-identical to the serial reference implementation.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

// Every finite distance in this benchmark is bounded by MAX_DISTANCE (the graph
// is complete, so the direct edge is always an upper bound), which lets the
// pivot panel travel over the network in the narrowest integer type that can
// still hold every reachable distance.  Padding entries saturate to the largest
// representable value; since that value already exceeds MAX_DISTANCE it can
// never win a relaxation of a real entry, so the result stays bit-identical.
using PanelT = std::conditional_t<
    (MAX_DISTANCE < 255u), unsigned char,
    std::conditional_t<(MAX_DISTANCE < 65535u), unsigned short, unsigned int>>;
constexpr unsigned int PANEL_MAX = static_cast<unsigned int>(std::numeric_limits<PanelT>::max());
static_assert(MAX_DISTANCE < PANEL_MAX, "panel type must be able to hold every finite distance");

inline MPI_Datatype panelMpiType() {
    if (sizeof(PanelT) == 1) return MPI_UNSIGNED_CHAR;
    if (sizeof(PanelT) == 2) return MPI_UNSIGNED_SHORT;
    return MPI_UNSIGNED;
}

__device__ inline PanelT toPanel(const unsigned int v) {
    return static_cast<PanelT>(v < PANEL_MAX ? v : PANEL_MAX);
}

// Tile size used by the blocked GPU algorithm.  Small problems use a smaller
// tile so that enough thread blocks are available to fill the device.
constexpr int TILE_SMALL = 32;
constexpr int TILE_LARGE = 64;

// Number of matrix rows whose pivots are grouped into a single panel exchange.
constexpr int SUPER_ROWS = 256;

// Index calculation for flattened 2D array
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

#define CUDA_CHECK(call)                                                                       \
    do {                                                                                       \
        const cudaError_t err_ = (call);                                                       \
        if (err_ != cudaSuccess) {                                                             \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__,    \
                    __LINE__);                                                                 \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                      \
        }                                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// Random number generation
//
// The reference implementation fills the matrix with a single sequential
// rand_r() stream.  rand_r() is a plain 32 bit LCG advanced three times per
// call, so the stream can be split for parallel generation by jumping the LCG
// ahead.  Correctness of the replication is verified against the libc rand_r()
// at run time; if it ever differs we fall back to the sequential stream.
// ---------------------------------------------------------------------------

namespace lcg {
constexpr unsigned int A = 1103515245u;
constexpr unsigned int C = 12345u;

// Affine transform  s -> a*s + c  (mod 2^32)
struct Step {
    unsigned int a;
    unsigned int c;
};

// Apply f first, then g.
inline Step combine(const Step f, const Step g) noexcept {
    return Step{f.a * g.a, g.a * f.c + g.c};
}

// The transform that advances the state by n single LCG steps.
inline Step advance(unsigned long long n) noexcept {
    Step result{1u, 0u};
    Step base{A, C};
    while (n > 0) {
        if (n & 1ull) {
            result = combine(result, base);
        }
        base = combine(base, base);
        n >>= 1;
    }
    return result;
}

// Bit-exact replica of glibc's rand_r().
inline int next(unsigned int& seed) noexcept {
    unsigned int s = seed;
    s = s * A + C;
    int result = static_cast<int>((s / 65536u) % 2048u);
    s = s * A + C;
    result <<= 10;
    result ^= static_cast<int>((s / 65536u) % 1024u);
    s = s * A + C;
    result <<= 10;
    result ^= static_cast<int>((s / 65536u) % 1024u);
    seed = s;
    return result;
}

// True if the replica matches the libc implementation.
inline bool matchesLibc() noexcept {
    unsigned int a = 42, b = 42;
    for (int i = 0; i < 16; ++i) {
        if (next(a) != rand_r(&b) || a != b) {
            return false;
        }
    }
    return true;
}
} // namespace lcg

// Fill the local row slice [rowStart, rowStart+localRows) of the padded matrix.
// Rows / columns outside of the original problem are padding: they are set to a
// value large enough to never take part in a shortest path but small enough to
// never overflow when two of them are added.
void initializeDistanceMatrix(unsigned int* dist, const size_t numNodes, const size_t paddedNodes,
                              const size_t rowStart, const size_t localRows,
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;
    const size_t realRows = (rowStart < numNodes) ? std::min(localRows, numNodes - rowStart) : 0;

    if (lcg::matchesLibc()) {
        // Parallel generation: every row starts from a jumped-ahead LCG state.
#pragma omp parallel for schedule(static)
        for (size_t r = 0; r < realRows; ++r) {
            const size_t globalRow = rowStart + r;
            const lcg::Step jump = lcg::advance(3ull * static_cast<unsigned long long>(globalRow) *
                                                static_cast<unsigned long long>(numNodes));
            unsigned int seed = jump.a * 42u + jump.c;
            unsigned int* row = dist + r * paddedNodes;
            for (size_t c = 0; c < numNodes; ++c) {
                row[c] = rangeMin + (unsigned int)(range * lcg::next(seed) / (double)RAND_MAX);
            }
        }
    } else {
        // Fallback: reproduce the sequential stream and keep the local slice.
        unsigned int seed = 42;
        for (size_t r = 0; r < numNodes; ++r) {
            const bool mine = (r >= rowStart) && (r < rowStart + localRows);
            unsigned int* row = mine ? dist + (r - rowStart) * paddedNodes : nullptr;
            for (size_t c = 0; c < numNodes; ++c) {
                const unsigned int v =
                    rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
                if (mine) {
                    row[c] = v;
                }
            }
        }
    }

    // Padding columns of the real rows.
#pragma omp parallel for schedule(static)
    for (size_t r = 0; r < realRows; ++r) {
        unsigned int* row = dist + r * paddedNodes;
        for (size_t c = numNodes; c < paddedNodes; ++c) {
            row[c] = INF;
        }
    }

    // Padding rows.
#pragma omp parallel for schedule(static)
    for (size_t r = realRows; r < localRows; ++r) {
        unsigned int* row = dist + r * paddedNodes;
        for (size_t c = 0; c < paddedNodes; ++c) {
            row[c] = INF;
        }
    }

    // Set diagonal to zero (distance from node to itself is 0)
#pragma omp parallel for schedule(static)
    for (size_t r = 0; r < localRows; ++r) {
        const size_t globalRow = rowStart + r;
        if (globalRow < paddedNodes) {
            dist[r * paddedNodes + globalRow] = 0;
        }
    }
}

// The reference initializePathMatrix() assigns path[row][col] = row.
__global__ void initializePathMatrixKernel(unsigned int* __restrict__ path, const int paddedNodes,
                                           const int rowStart, const int localRows) {
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    const size_t total = static_cast<size_t>(localRows) * paddedNodes;
    for (size_t p = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; p < total;
         p += stride) {
        path[p] = static_cast<unsigned int>(rowStart) + static_cast<unsigned int>(p / paddedNodes);
    }
}

// ---------------------------------------------------------------------------
// Blocked Floyd-Warshall GPU kernels.  Every thread block works on a B x B tile
// and every thread owns a 2x2 sub-tile.
//
// To stay bit-identical to the serial triple loop the pivot data has to be used
// exactly as it looked when the corresponding k iteration was executed: a naive
// blocked implementation would relax against pivot entries that the serial code
// only improves at a later k.  Therefore phase 1 and phase 2 snapshot the pivot
// row / column at the very step where it is consumed, and the later phases work
// on those snapshots.  The distance matrix stays identical either way, the
// predecessor matrix does not.
// ---------------------------------------------------------------------------

// Phase 1: the pivot block (K,K) is relaxed against itself.
// Writes the row snapshot into the broadcast panel (columns of the pivot block)
// and the column snapshot into the small pivot scratch buffer.
template <int B>
__global__ void fwPhase1(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
                         PanelT* __restrict__ panel, unsigned int* __restrict__ pivCol,
                         const int pitch, const int kb) {
    constexpr int H = B / 2;
    __shared__ unsigned int sd[B][B];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    unsigned int cur[2][2];
    unsigned int cp[2][2];

#pragma unroll
    for (int a = 0; a < 2; ++a) {
#pragma unroll
        for (int b = 0; b < 2; ++b) {
            const int r = ty + a * H, c = tx + b * H;
            const size_t o = static_cast<size_t>(r) * pitch + kb + c;
            cur[a][b] = dist[o];
            cp[a][b] = path[o];
            sd[r][c] = cur[a][b];
        }
    }
    __syncthreads();

    for (int k = 0; k < B; ++k) {
        // Snapshot row k / column k as seen by serial iteration kb + k.
#pragma unroll
        for (int a = 0; a < 2; ++a) {
            if (ty + a * H == k) {
#pragma unroll
                for (int b = 0; b < 2; ++b) {
                    const int c = tx + b * H;
                    panel[static_cast<size_t>(k) * pitch + kb + c] = toPanel(sd[k][c]);
                }
            }
        }
#pragma unroll
        for (int b = 0; b < 2; ++b) {
            if (tx + b * H == k) {
#pragma unroll
                for (int a = 0; a < 2; ++a) {
                    const int r = ty + a * H;
                    pivCol[r * B + k] = sd[r][k];
                }
            }
        }

#pragma unroll
        for (int a = 0; a < 2; ++a) {
#pragma unroll
            for (int b = 0; b < 2; ++b) {
                const int r = ty + a * H, c = tx + b * H;
                const unsigned int nd = sd[r][k] + sd[k][c];
                if (nd < cur[a][b]) {
                    cur[a][b] = nd;
                    cp[a][b] = kb + k;
                }
            }
        }
        __syncthreads();
#pragma unroll
        for (int a = 0; a < 2; ++a) {
#pragma unroll
            for (int b = 0; b < 2; ++b) {
                sd[ty + a * H][tx + b * H] = cur[a][b];
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int a = 0; a < 2; ++a) {
#pragma unroll
        for (int b = 0; b < 2; ++b) {
            const int r = ty + a * H, c = tx + b * H;
            const size_t o = static_cast<size_t>(r) * pitch + kb + c;
            dist[o] = cur[a][b];
            path[o] = cp[a][b];
        }
    }
}

// Phase 2a: the pivot row panel, blocks (K,j) for all j != K.
// dist/path point at the first row of the pivot row block.  The row snapshots
// are written into the panel that is broadcast to all ranks.
template <int B>
__global__ void fwPhase2Row(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
                            PanelT* __restrict__ panel, const unsigned int* __restrict__ pivCol,
                            const int pitch, const int kb,
                            const int kBlock) {
    constexpr int H = B / 2;
    const int j = blockIdx.x;
    if (j == kBlock) {
        return;
    }

    __shared__ unsigned int sPiv[B][B];
    __shared__ unsigned int sCur[B][B];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int colBase = j * B;

    unsigned int cur[2][2];
    unsigned int cp[2][2];

#pragma unroll
    for (int a = 0; a < 2; ++a) {
#pragma unroll
        for (int b = 0; b < 2; ++b) {
            const int r = ty + a * H, c = tx + b * H;
            const size_t base = static_cast<size_t>(r) * pitch;
            sPiv[r][c] = pivCol[r * B + c];
            cur[a][b] = dist[base + colBase + c];
            cp[a][b] = path[base + colBase + c];
            sCur[r][c] = cur[a][b];
        }
    }
    __syncthreads();

    for (int k = 0; k < B; ++k) {
#pragma unroll
        for (int a = 0; a < 2; ++a) {
            if (ty + a * H == k) {
#pragma unroll
                for (int b = 0; b < 2; ++b) {
                    const int c = tx + b * H;
                    panel[static_cast<size_t>(k) * pitch + colBase + c] = toPanel(sCur[k][c]);
                }
            }
        }

#pragma unroll
        for (int a = 0; a < 2; ++a) {
#pragma unroll
            for (int b = 0; b < 2; ++b) {
                const int r = ty + a * H, c = tx + b * H;
                const unsigned int nd = sPiv[r][k] + sCur[k][c];
                if (nd < cur[a][b]) {
                    cur[a][b] = nd;
                    cp[a][b] = kb + k;
                }
            }
        }
        __syncthreads();
#pragma unroll
        for (int a = 0; a < 2; ++a) {
#pragma unroll
            for (int b = 0; b < 2; ++b) {
                sCur[ty + a * H][tx + b * H] = cur[a][b];
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int a = 0; a < 2; ++a) {
#pragma unroll
        for (int b = 0; b < 2; ++b) {
            const int r = ty + a * H, c = tx + b * H;
            const size_t o = static_cast<size_t>(r) * pitch + colBase + c;
            dist[o] = cur[a][b];
            path[o] = cp[a][b];
        }
    }
}

// Phase 2b: the pivot column panel, blocks (i,K) for all locally owned i.
// The column snapshots are stored per row block for use by phase 3.
template <int B>
__global__ void fwPhase2Col(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
                            const PanelT* __restrict__ panel, unsigned int* __restrict__ colSnap, const int pitch, const int kb,
                            const int rowBlockOffset, const int skipLoA, const int skipHiA,
                            const int skipLoB, const int skipHiB) {
    constexpr int H = B / 2;
    const int i = blockIdx.x + rowBlockOffset;
    if ((i >= skipLoA && i < skipHiA) || (i >= skipLoB && i < skipHiB)) {
        return;
    }

    __shared__ unsigned int sPiv[B][B];
    __shared__ unsigned int sCur[B][B];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int rowBase = i * B;
    unsigned int* snap = colSnap + static_cast<size_t>(i) * B * B;

    unsigned int cur[2][2];
    unsigned int cp[2][2];

#pragma unroll
    for (int a = 0; a < 2; ++a) {
#pragma unroll
        for (int b = 0; b < 2; ++b) {
            const int r = ty + a * H, c = tx + b * H;
            sPiv[r][c] = panel[static_cast<size_t>(r) * pitch + kb + c];
            const size_t o = static_cast<size_t>(rowBase + r) * pitch + kb + c;
            cur[a][b] = dist[o];
            cp[a][b] = path[o];
            sCur[r][c] = cur[a][b];
        }
    }
    __syncthreads();

    for (int k = 0; k < B; ++k) {
#pragma unroll
        for (int b = 0; b < 2; ++b) {
            if (tx + b * H == k) {
#pragma unroll
                for (int a = 0; a < 2; ++a) {
                    const int r = ty + a * H;
                    snap[r * B + k] = sCur[r][k];
                }
            }
        }

#pragma unroll
        for (int a = 0; a < 2; ++a) {
#pragma unroll
            for (int b = 0; b < 2; ++b) {
                const int r = ty + a * H, c = tx + b * H;
                const unsigned int nd = sCur[r][k] + sPiv[k][c];
                if (nd < cur[a][b]) {
                    cur[a][b] = nd;
                    cp[a][b] = kb + k;
                }
            }
        }
        __syncthreads();
#pragma unroll
        for (int a = 0; a < 2; ++a) {
#pragma unroll
            for (int b = 0; b < 2; ++b) {
                sCur[ty + a * H][tx + b * H] = cur[a][b];
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int a = 0; a < 2; ++a) {
#pragma unroll
        for (int b = 0; b < 2; ++b) {
            const int r = ty + a * H, c = tx + b * H;
            const size_t o = static_cast<size_t>(rowBase + r) * pitch + kb + c;
            dist[o] = cur[a][b];
            path[o] = cp[a][b];
        }
    }
}

// Phase 3: all remaining blocks (i,j) - an independent min-plus product of the
// pivot column snapshot (i,K) and the pivot row snapshot (K,j).
template <int B>
__global__ __launch_bounds__((B / 2) * (B / 2)) void fwPhase3(
    unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
    const PanelT* __restrict__ panel, const unsigned int* __restrict__ colSnap,
    const int pitch, const int kb, const int kBlock, const int rowBlockOffset, const int skipLoA,
    const int skipHiA, const int skipLoB, const int skipHiB) {
    constexpr int H = B / 2;
    const int j = blockIdx.x;
    const int i = blockIdx.y + rowBlockOffset;
    if (j == kBlock || (i >= skipLoA && i < skipHiA) || (i >= skipLoB && i < skipHiB)) {
        return;
    }

    __shared__ unsigned int sA[B][B]; // snapshot of block (i,K)
    __shared__ unsigned int sB[B][B]; // snapshot of block (K,j)

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int rowBase = i * B;
    const int colBase = j * B;
    const unsigned int* snap = colSnap + static_cast<size_t>(i) * B * B;

    unsigned int cur[2][2];
    unsigned int cp[2][2];

#pragma unroll
    for (int a = 0; a < 2; ++a) {
#pragma unroll
        for (int b = 0; b < 2; ++b) {
            const int r = ty + a * H, c = tx + b * H;
            sA[r][c] = snap[r * B + c];
            sB[r][c] = panel[static_cast<size_t>(r) * pitch + colBase + c];
            const size_t o = static_cast<size_t>(rowBase + r) * pitch + colBase + c;
            cur[a][b] = dist[o];
            cp[a][b] = path[o];
        }
    }
    __syncthreads();

#pragma unroll 8
    for (int k = 0; k < B; ++k) {
        const unsigned int a0 = sA[ty][k];
        const unsigned int a1 = sA[ty + H][k];
        const unsigned int b0 = sB[k][tx];
        const unsigned int b1 = sB[k][tx + H];
        unsigned int nd;
        nd = a0 + b0;
        if (nd < cur[0][0]) { cur[0][0] = nd; cp[0][0] = kb + k; }
        nd = a0 + b1;
        if (nd < cur[0][1]) { cur[0][1] = nd; cp[0][1] = kb + k; }
        nd = a1 + b0;
        if (nd < cur[1][0]) { cur[1][0] = nd; cp[1][0] = kb + k; }
        nd = a1 + b1;
        if (nd < cur[1][1]) { cur[1][1] = nd; cp[1][1] = kb + k; }
    }

#pragma unroll
    for (int a = 0; a < 2; ++a) {
#pragma unroll
        for (int b = 0; b < 2; ++b) {
            const int r = ty + a * H, c = tx + b * H;
            const size_t o = static_cast<size_t>(rowBase + r) * pitch + colBase + c;
            dist[o] = cur[a][b];
            path[o] = cp[a][b];
        }
    }
}

// Owner of a given pivot block.
static inline int blockOwner(const int block, const int numBlocks, const int size) {
    const int base = numBlocks / size;
    const int rem = numBlocks % size;
    const int big = (base + 1) * rem; // blocks held by the first "rem" ranks
    if (block < big) {
        return block / (base + 1);
    }
    return rem + (block - big) / base;
}

// Distributed blocked Floyd-Warshall.  All matrices live on the device; the
// pivot row panels are exchanged through pinned host memory.
//
// Row blocks are handed out in groups ("super blocks") of up to lookahead
// consecutive pivot blocks.  Updating the pivot rows of a super block only ever
// reads data the owner already holds, so the owner can run all pivots of the
// group ahead of everybody else and ship the resulting panels in a single
// broadcast.  That amortizes the per-exchange latency over lookahead pivots.
//
// The exchange for super block O+1 is additionally overlapped with the phase 3
// work of super block O: its owner processes its own rows first, produces the
// next panels and starts the transfer while the remaining (much larger) part of
// the update is still queued on the GPU.  Panels are double buffered so that the
// incoming panel does not clobber the one still in use.
template <int B>
void floydWarshallDistributed(unsigned int* d_dist, unsigned int* d_path, PanelT* const* d_panel,
                              unsigned int* d_pivCol, unsigned int* d_colSnap,
                              PanelT* const* h_panel, const int paddedNodes, const int numBlocks,
                              const int lookahead, const int numSuper, const int firstBlock,
                              const int myBlocks, const int rank, const int size) {
    const dim3 threads(B / 2, B / 2);
    const size_t panelStride = static_cast<size_t>(B) * paddedNodes;

    cudaStream_t sCompute, sComm;
    CUDA_CHECK(cudaStreamCreate(&sCompute));
    CUDA_CHECK(cudaStreamCreate(&sComm));
    cudaEvent_t evD2H[2], evH2D[2];
    for (int i = 0; i < 2; ++i) {
        CUDA_CHECK(cudaEventCreateWithFlags(&evD2H[i], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&evH2D[i], cudaEventDisableTiming));
    }

    // Number of pivot blocks in super block "sup" (the last one may be short).
    auto superSize = [&](const int sup) {
        return std::min(lookahead, numBlocks - sup * lookahead);
    };

    // Owner-side production of all pivot panels of a super block: for every
    // pivot of the group run phases 1 and 2a and apply the pivot to the other
    // pivot rows of the same group.  Everything needed is owned locally.
    auto producePanels = [&](const int sup, const int buf) {
        const int first = sup * lookahead;
        const int count = superSize(sup);
        const int localFirst = first - firstBlock;
        for (int l = 0; l < count; ++l) {
            const int kBlk = first + l;
            const int kb = kBlk * B;
            const int localK = kBlk - firstBlock;
            const size_t off = static_cast<size_t>(localK) * B * paddedNodes;
            PanelT* panel = d_panel[buf] + static_cast<size_t>(l) * panelStride;
            fwPhase1<B><<<1, threads, 0, sCompute>>>(d_dist + off, d_path + off, panel, d_pivCol,
                                                     paddedNodes, kb);
            fwPhase2Row<B><<<numBlocks, threads, 0, sCompute>>>(d_dist + off, d_path + off, panel,
                                                                d_pivCol, paddedNodes, kb, kBlk);
            if (count > 1) {
                fwPhase2Col<B><<<count, threads, 0, sCompute>>>(d_dist, d_path, panel, d_colSnap,
                                                                paddedNodes, kb, localFirst, localK,
                                                                localK + 1, 0, 0);
                fwPhase3<B><<<dim3(numBlocks, count), threads, 0, sCompute>>>(
                    d_dist, d_path, panel, d_colSnap, paddedNodes, kb, kBlk, localFirst, localK,
                    localK + 1, 0, 0);
            }
        }
        if (size > 1) {
            CUDA_CHECK(cudaMemcpyAsync(h_panel[buf], d_panel[buf],
                                       count * panelStride * sizeof(PanelT),
                                       cudaMemcpyDeviceToHost, sCompute));
            CUDA_CHECK(cudaEventRecord(evD2H[buf], sCompute));
        }
    };

    // Broadcast the panels of a super block and make them visible to sCompute.
    auto distributePanels = [&](const int sup, const int buf) {
        if (size == 1) {
            return; // the panels are produced in place, nothing to exchange
        }
        const int owner = blockOwner(sup, numSuper, size);
        const size_t elems = static_cast<size_t>(superSize(sup)) * panelStride;
        if (rank == owner) {
            CUDA_CHECK(cudaEventSynchronize(evD2H[buf]));
        }
        MPI_Bcast(h_panel[buf], static_cast<int>(elems), panelMpiType(), owner, MPI_COMM_WORLD);
        if (rank != owner) {
            CUDA_CHECK(cudaMemcpyAsync(d_panel[buf], h_panel[buf], elems * sizeof(PanelT),
                                       cudaMemcpyHostToDevice, sComm));
            CUDA_CHECK(cudaEventRecord(evH2D[buf], sComm));
            CUDA_CHECK(cudaStreamWaitEvent(sCompute, evH2D[buf], 0));
        }
    };

    // Apply every pivot of super block "sup" to the row blocks
    // [rowOffset, rowOffset+rowCount) minus the two excluded ranges.
    auto applyPanels = [&](const int sup, const int buf, const int rowOffset, const int rowCount,
                           const int skipLoA, const int skipHiA, const int skipLoB,
                           const int skipHiB) {
        const int first = sup * lookahead;
        const int count = superSize(sup);
        for (int l = 0; l < count; ++l) {
            const int kBlk = first + l;
            const int kb = kBlk * B;
            const PanelT* panel = d_panel[buf] + static_cast<size_t>(l) * panelStride;
            fwPhase2Col<B><<<rowCount, threads, 0, sCompute>>>(d_dist, d_path, panel, d_colSnap,
                                                               paddedNodes, kb, rowOffset, skipLoA,
                                                               skipHiA, skipLoB, skipHiB);
            fwPhase3<B><<<dim3(numBlocks, rowCount), threads, 0, sCompute>>>(
                d_dist, d_path, panel, d_colSnap, paddedNodes, kb, kBlk, rowOffset, skipLoA, skipHiA,
                skipLoB, skipHiB);
        }
    };

    int buf = 0;
    if (rank == blockOwner(0, numSuper, size)) {
        producePanels(0, buf);
    }
    distributePanels(0, buf);

    for (int sup = 0; sup < numSuper; ++sup) {
        const int nextBuf = buf ^ 1;

        // Own pivot rows of this super block: already updated by producePanels.
        int skipLoA = 0, skipHiA = 0;
        if (rank == blockOwner(sup, numSuper, size)) {
            skipLoA = sup * lookahead - firstBlock;
            skipHiA = skipLoA + superSize(sup);
        }

        // Rows that have to be finished first so that the next panels can be
        // produced right away (empty if this rank does not own the next group).
        int prioLo = 0, prioHi = 0;
        if (sup + 1 < numSuper && rank == blockOwner(sup + 1, numSuper, size)) {
            prioLo = (sup + 1) * lookahead - firstBlock;
            prioHi = prioLo + superSize(sup + 1);
        }

        if (prioHi > prioLo) {
            applyPanels(sup, buf, prioLo, prioHi - prioLo, skipLoA, skipHiA, 0, 0);
            producePanels(sup + 1, nextBuf);
        }

        if (myBlocks > 0) {
            applyPanels(sup, buf, 0, myBlocks, skipLoA, skipHiA, prioLo, prioHi);
        }
        CUDA_CHECK(cudaGetLastError());

        if (sup + 1 < numSuper) {
            distributePanels(sup + 1, nextBuf);
        }
        buf = nextBuf;
    }
    CUDA_CHECK(cudaStreamSynchronize(sCompute));

    for (int i = 0; i < 2; ++i) {
        CUDA_CHECK(cudaEventDestroy(evD2H[i]));
        CUDA_CHECK(cudaEventDestroy(evH2D[i]));
    }
    CUDA_CHECK(cudaStreamDestroy(sCompute));
    CUDA_CHECK(cudaStreamDestroy(sComm));
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
    const size_t limit = std::min(numNodes, static_cast<size_t>(10));
    bool valid = true;
#pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = 0; i < limit; ++i) {
        for (size_t j = 0; j < limit; ++j) {
            for (size_t k = 0; k < numNodes; ++k) {
                const unsigned int distIJ = dist[idx2(j, i, numNodes)];
                const unsigned int distIK = dist[idx2(k, i, numNodes)];
                const unsigned int distKJ = dist[idx2(j, k, numNodes)];

                // Check for overflow before addition
                if (distIK < INF && distKJ < INF) {
                    if (distIK + distKJ < distIJ) {
                        printf("Validation failed: triangle inequality violated at [%zu,%zu,%zu]\n",
                               i, j, k);
                        valid = false;
                    }
                }
            }
        }
    }

    return valid;
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

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

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
            if (rank == 0) {
                printUsage(argv[0]);
            }
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

    if (numNodes == 0) {
        if (rank == 0) {
            printf("Number of nodes must be positive\n");
        }
        MPI_Finalize();
        return 1;
    }

    // One GPU per rank, assigned by the rank's position on its node.
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_free(&nodeComm);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA device available on rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));

    // Tile size: keep enough thread blocks in flight for small problems.
    const int tile = (numNodes <= 1024) ? TILE_SMALL : TILE_LARGE;
    const int paddedNodes = static_cast<int>((numNodes + tile - 1) / tile) * tile;
    const int numBlocks = paddedNodes / tile;

    // Consecutive pivot blocks are grouped so that one exchange ships several
    // panels; the groups are the unit of the row distribution.
    int lookahead = std::max(1, SUPER_ROWS / tile);
    lookahead = std::min(lookahead, std::max(1, numBlocks / size));
    const int numSuper = (numBlocks + lookahead - 1) / lookahead;

    const int base = numSuper / size;
    const int rem = numSuper % size;
    const int mySuper = base + (rank < rem ? 1 : 0);
    const int firstSuper = rank * base + std::min(rank, rem);
    const int firstBlock = std::min(firstSuper * lookahead, numBlocks);
    const int myBlocks =
        std::max(0, std::min((firstSuper + mySuper) * lookahead, numBlocks) - firstBlock);
    const size_t rowStart = static_cast<size_t>(firstBlock) * tile;
    const size_t localRows = static_cast<size_t>(myBlocks) * tile;
    const size_t localElems = localRows * paddedNodes;

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, OpenMP threads: %d, GPUs per node: %d\n", size,
               omp_get_max_threads(), deviceCount);
        printf("Padded size: %d, tile: %d, lookahead: %d\n", paddedNodes, tile, lookahead);
    }

    // Allocate matrices
    std::vector<unsigned int> hostDist(localElems);
    unsigned int* d_dist = nullptr;
    unsigned int* d_path = nullptr;
    PanelT* d_panel[2] = {nullptr, nullptr};
    unsigned int* d_pivCol = nullptr;
    unsigned int* d_colSnap = nullptr;
    PanelT* h_panel[2] = {nullptr, nullptr};
    const size_t panelElems = static_cast<size_t>(lookahead) * tile * paddedNodes;
    if (localElems > 0) {
        CUDA_CHECK(cudaMalloc(&d_dist, localElems * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&d_path, localElems * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&d_colSnap,
                              static_cast<size_t>(myBlocks) * tile * tile * sizeof(unsigned int)));
    }
    CUDA_CHECK(cudaMalloc(&d_pivCol, static_cast<size_t>(tile) * tile * sizeof(unsigned int)));
    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaMalloc(&d_panel[b], panelElems * sizeof(PanelT)));
        CUDA_CHECK(cudaHostAlloc(&h_panel[b], panelElems * sizeof(PanelT), cudaHostAllocDefault));
    }

    // Initialize
    if (rank == 0) {
        printf("Initializing graph...\n");
    }
    initializeDistanceMatrix(hostDist.data(), numNodes, paddedNodes, rowStart, localRows, 1,
                             MAX_DISTANCE);
    if (localElems > 0) {
        initializePathMatrixKernel<<<1024, 256>>>(d_path, paddedNodes, static_cast<int>(rowStart),
                                                  static_cast<int>(localRows));
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Run Floyd-Warshall
    if (rank == 0) {
        printf("Computing shortest paths...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    if (localElems > 0) {
        CUDA_CHECK(cudaMemcpy(d_dist, hostDist.data(), localElems * sizeof(unsigned int),
                              cudaMemcpyHostToDevice));
    }
    if (tile == TILE_SMALL) {
        floydWarshallDistributed<TILE_SMALL>(d_dist, d_path, d_panel, d_pivCol, d_colSnap, h_panel,
                                             paddedNodes, numBlocks, lookahead, numSuper,
                                             firstBlock, myBlocks, rank, size);
    } else {
        floydWarshallDistributed<TILE_LARGE>(d_dist, d_path, d_panel, d_pivCol, d_colSnap, h_panel,
                                             paddedNodes, numBlocks, lookahead, numSuper,
                                             firstBlock, myBlocks, rank, size);
    }
    if (localElems > 0) {
        CUDA_CHECK(cudaMemcpy(hostDist.data(), d_dist, localElems * sizeof(unsigned int),
                              cudaMemcpyDeviceToHost));
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate operations per second
        // Floyd-Warshall has O(n³) complexity
        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);
    }

    // Collect the full distance matrix on rank 0 if it is needed.
    std::vector<unsigned int> dist;
    if (validate || printResults) {
        const size_t realRows =
            (rowStart < numNodes) ? std::min(localRows, numNodes - rowStart) : 0;
        std::vector<unsigned int> sendBuf(realRows * numNodes);
#pragma omp parallel for schedule(static)
        for (size_t r = 0; r < realRows; ++r) {
            memcpy(sendBuf.data() + r * numNodes, hostDist.data() + r * paddedNodes,
                   numNodes * sizeof(unsigned int));
        }

        std::vector<int> counts(size), displs(size);
        int myCount = static_cast<int>(realRows * numNodes);
        MPI_Gather(&myCount, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            int off = 0;
            for (int r = 0; r < size; ++r) {
                displs[r] = off;
                off += counts[r];
            }
            dist.resize(numNodes * numNodes);
        }
        MPI_Gatherv(sendBuf.data(), myCount, MPI_UNSIGNED, dist.data(), counts.data(),
                    displs.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    }

    int exitCode = 0;

    if (rank == 0) {
        // Print results for external validation (integer hash-based)
        if (printResults) {
            print_results_int(dist, "DistanceMatrix");
        }

        // Validation
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(dist, numNodes);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (d_dist) {
        CUDA_CHECK(cudaFree(d_dist));
        CUDA_CHECK(cudaFree(d_path));
        CUDA_CHECK(cudaFree(d_colSnap));
    }
    CUDA_CHECK(cudaFree(d_pivCol));
    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaFree(d_panel[b]));
        CUDA_CHECK(cudaFreeHost(h_panel[b]));
    }

    MPI_Finalize();
    return exitCode;
}
