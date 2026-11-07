// Floyd-Warshall All-Pairs Shortest Path Benchmark
//
// Hybrid MPI + OpenMP + CUDA implementation.
//
//  * MPI   : the distance matrix is partitioned by tile-rows across ranks (one
//            GPU per rank). For each pivot tile-row the owner computes the
//            pivot tile and its tile-row and broadcasts it; every rank then
//            updates its own tile-rows. The pivot tile-row is sent in column
//            chunks, so receivers start computing on the first chunk while
//            later chunks are still in flight, and the broadcast of pivot
//            kb+1 is overlapped with the phase-3 work of pivot kb (lookahead).
//  * CUDA  : blocked Floyd-Warshall (phase 1 / 2 / 3 kernels, 64x64 tiles,
//            shared-memory staged, 2x2 elements per thread).
//  * OpenMP: host-side work (parallel graph generation with LCG skip-ahead,
//            padding, validation).
//
// The computed distance matrix is bit-identical to the sequential algorithm.

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

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

// Tile edge length for the blocked algorithm; each thread block is 32x32
// threads and every thread handles a 2x2 sub-block of a tile.
constexpr int TILE = 64;
constexpr int HALF = TILE / 2;
constexpr size_t TILE_ELEMS = (size_t)TILE * TILE;

// The pivot tile-row is broadcast in column chunks of roughly this many bytes
// (at most MAX_CHUNKS chunks) so that transfer and compute are pipelined.
constexpr size_t CHUNK_TARGET_BYTES = 512 * 1024;
constexpr int MAX_CHUNKS = 8;

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__,    \
                    __LINE__, cudaGetErrorString(err__));                       \
            MPI_Abort(MPI_COMM_WORLD, 1);                                       \
        }                                                                       \
    } while (0)

// Index calculation for flattened 2D array
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

// ---------------------------------------------------------------------------
// Random number generation
//
// The original benchmark fills the matrix with a sequential rand_r() stream
// seeded with 42. To generate it in parallel (and to let each MPI rank
// generate only its own rows) we emulate glibc's rand_r, whose state is a
// 32-bit LCG advanced three times per call, and skip ahead in the LCG.
// The emulation is verified at runtime against the real rand_r; if it does
// not match, we fall back to the sequential stream.
// ---------------------------------------------------------------------------
namespace rng {

constexpr unsigned int LCG_A = 1103515245u;
constexpr unsigned int LCG_C = 12345u;

inline unsigned int glibc_rand_r(unsigned int& seed) {
    unsigned int next = seed;
    unsigned int result;
    next = next * LCG_A + LCG_C;
    result = (next / 65536u) % 2048u;
    next = next * LCG_A + LCG_C;
    result <<= 10;
    result ^= (next / 65536u) % 1024u;
    next = next * LCG_A + LCG_C;
    result <<= 10;
    result ^= (next / 65536u) % 1024u;
    seed = next;
    return result;
}

// Advance the LCG state by `steps` steps in O(log steps).
inline unsigned int lcg_skip(unsigned int x, unsigned long long steps) {
    unsigned int A = 1u, C = 0u;
    unsigned int a = LCG_A, c = LCG_C;
    while (steps) {
        if (steps & 1ull) {
            A = A * a;
            C = C * a + c;
        }
        c = a * c + c;
        a = a * a;
        steps >>= 1;
    }
    return A * x + C;
}

inline bool emulation_matches() {
    unsigned int s1 = 42, s2 = 42;
    for (int i = 0; i < 4096; ++i) {
        if ((unsigned int)rand_r(&s1) != glibc_rand_r(s2) || s1 != s2) return false;
    }
    // Also verify the skip-ahead.
    unsigned int s3 = lcg_skip(42, 3ull * 4096ull);
    return s3 == s2;
}

}  // namespace rng

// ---------------------------------------------------------------------------
// Domain decomposition helpers (identical on all ranks)
// ---------------------------------------------------------------------------
struct Decomp {
    size_t n;         // number of nodes
    size_t nPad;      // padded matrix dimension (multiple of TILE)
    size_t T;         // number of tiles per dimension
    int P;            // number of ranks

    size_t tileRowBegin(int r) const { return T * (size_t)r / (size_t)P; }
    size_t tileRowEnd(int r) const { return T * (size_t)(r + 1) / (size_t)P; }
    size_t rowBegin(int r) const { return tileRowBegin(r) * TILE; }
    size_t realRows(int r) const {
        const size_t rb = rowBegin(r);
        const size_t re = std::min(tileRowEnd(r) * TILE, n);
        return re > rb ? re - rb : 0;
    }
};

// ---------------------------------------------------------------------------
// Host initialisation (OpenMP)
// ---------------------------------------------------------------------------

// Fills the local, padded block of rows [rowBegin, rowBegin + localRowsPad)
// of the distance matrix exactly as the sequential benchmark would.
void initializeDistanceMatrix(std::vector<unsigned int>& local, const Decomp& dc,
                              const size_t rowBegin, const size_t realRows,
                              const size_t localRowsPad,
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    const size_t n = dc.n;
    const size_t nPad = dc.nPad;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;
    const unsigned int seed0 = 42;

    const unsigned long long firstElem = (unsigned long long)rowBegin * n;
    const unsigned long long numElems = (unsigned long long)realRows * n;

    if (numElems > 0) {
        if (rng::emulation_matches()) {
            #pragma omp parallel
            {
                const int nt = omp_get_num_threads();
                const int tid = omp_get_thread_num();
                const unsigned long long e0 = numElems * tid / nt;
                const unsigned long long e1 = numElems * (tid + 1) / nt;
                unsigned int seed = rng::lcg_skip(seed0, 3ull * (firstElem + e0));
                size_t row = (size_t)(e0 / n);
                size_t col = (size_t)(e0 % n);
                unsigned int* rowPtr = local.data() + row * nPad;
                for (unsigned long long e = e0; e < e1; ++e) {
                    rowPtr[col] = rangeMin +
                        (unsigned int)(range * rng::glibc_rand_r(seed) / (double)RAND_MAX);
                    if (++col == n) {
                        col = 0;
                        ++row;
                        rowPtr = local.data() + row * nPad;
                    }
                }
            }
        } else {
            // Fallback: walk the sequential stream, storing only our rows.
            unsigned int seed = seed0;
            for (unsigned long long e = 0; e < firstElem; ++e) rand_r(&seed);
            for (size_t r = 0; r < realRows; ++r) {
                unsigned int* rowPtr = local.data() + r * nPad;
                for (size_t c = 0; c < n; ++c) {
                    rowPtr[c] = rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
                }
            }
        }
    }

    // Padding: padded nodes are isolated (INF to/from everything, 0 on the
    // diagonal) so they never influence real shortest paths.
    #pragma omp parallel for schedule(static)
    for (size_t r = 0; r < localRowsPad; ++r) {
        unsigned int* rowPtr = local.data() + r * nPad;
        const size_t gRow = rowBegin + r;
        if (r < realRows) {
            for (size_t c = n; c < nPad; ++c) rowPtr[c] = INF;
            // Set diagonal to zero (distance from node to itself is 0)
            rowPtr[gRow] = 0;
        } else {
            for (size_t c = 0; c < nPad; ++c) rowPtr[c] = INF;
            if (gRow < nPad) rowPtr[gRow] = 0;
        }
    }
}

// ---------------------------------------------------------------------------
// CUDA kernels: blocked Floyd-Warshall
//
// Matrices are stored row-major with pitch `pitch` (= nPad) elements; the
// local matrix of a rank holds its contiguous block of tile-rows.
// The pivot tile-row that is exchanged between ranks is kept tile-major:
// tile j occupies elements [j*TILE*TILE, (j+1)*TILE*TILE), row-major inside.
// Every thread handles elements (i0,j0) (i0,j1) (i1,j0) (i1,j1) of a tile
// with i0 = ty, i1 = ty + 32, j0 = tx, j1 = tx + 32.
// ---------------------------------------------------------------------------

__global__ void initPathKernel(unsigned int* __restrict__ path, const size_t pitch,
                               const size_t rows, const size_t rowBegin) {
    const size_t r = blockIdx.y;
    const size_t c = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (r < rows && c < pitch) {
        // Original: path[i][j] == i for all i, j
        path[r * pitch + c] = (unsigned int)(rowBegin + r);
    }
}

__device__ __forceinline__ void relax(unsigned int& d, unsigned int& p, bool& changed,
                                      const unsigned int a, const unsigned int b,
                                      const unsigned int k) {
    const unsigned int nd = a + b;
    if (nd < d) {
        d = nd;
        p = k;
        changed = true;
    }
}

// Phase 1: the pivot tile (kb, kb). One thread block.
__global__ void __launch_bounds__(1024)
phase1Kernel(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
             const size_t pitch, const size_t kb, const size_t localTileRow,
             unsigned int* __restrict__ stage) {
    __shared__ unsigned int s[TILE][TILE];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const size_t base = localTileRow * TILE * pitch + kb * TILE;
    unsigned int* d = dist + base;
    unsigned int* p = path + base;

    unsigned int d00 = d[ty * pitch + tx];
    unsigned int d01 = d[ty * pitch + tx + HALF];
    unsigned int d10 = d[(ty + HALF) * pitch + tx];
    unsigned int d11 = d[(ty + HALF) * pitch + tx + HALF];
    unsigned int p00 = 0, p01 = 0, p10 = 0, p11 = 0;
    bool c00 = false, c01 = false, c10 = false, c11 = false;
    s[ty][tx] = d00;
    s[ty][tx + HALF] = d01;
    s[ty + HALF][tx] = d10;
    s[ty + HALF][tx + HALF] = d11;
    __syncthreads();

    const unsigned int kBase = (unsigned int)(kb * TILE);
    for (int k = 0; k < TILE; ++k) {
        // Row k and column k of the tile do not change in step k (diagonal is
        // zero), so one barrier per step suffices.
        const unsigned int a0 = s[ty][k];
        const unsigned int a1 = s[ty + HALF][k];
        const unsigned int b0 = s[k][tx];
        const unsigned int b1 = s[k][tx + HALF];
        const unsigned int gk = kBase + k;
        bool w;
        w = false; relax(d00, p00, w, a0, b0, gk); if (w) { s[ty][tx] = d00; c00 = true; }
        w = false; relax(d01, p01, w, a0, b1, gk); if (w) { s[ty][tx + HALF] = d01; c01 = true; }
        w = false; relax(d10, p10, w, a1, b0, gk); if (w) { s[ty + HALF][tx] = d10; c10 = true; }
        w = false; relax(d11, p11, w, a1, b1, gk); if (w) { s[ty + HALF][tx + HALF] = d11; c11 = true; }
        __syncthreads();
    }

    if (c00) { d[ty * pitch + tx] = d00; p[ty * pitch + tx] = p00; }
    if (c01) { d[ty * pitch + tx + HALF] = d01; p[ty * pitch + tx + HALF] = p01; }
    if (c10) { d[(ty + HALF) * pitch + tx] = d10; p[(ty + HALF) * pitch + tx] = p10; }
    if (c11) { d[(ty + HALF) * pitch + tx + HALF] = d11; p[(ty + HALF) * pitch + tx + HALF] = p11; }
    unsigned int* sg = stage + kb * TILE_ELEMS;
    sg[ty * TILE + tx] = d00;
    sg[ty * TILE + tx + HALF] = d01;
    sg[(ty + HALF) * TILE + tx] = d10;
    sg[(ty + HALF) * TILE + tx + HALF] = d11;
}

// Phase 2 (row): tiles (kb, j) for j != kb in [j0, j0 + gridDim.x), using
// the pivot tile from the same (owner-local) matrix. The finished tiles are
// also written tile-major into `stage` for broadcasting. One block per tile.
__global__ void __launch_bounds__(1024)
phase2RowKernel(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
                const size_t pitch, const size_t kb, const size_t localTileRow,
                unsigned int* __restrict__ stage, const size_t j0) {
    const size_t j = j0 + blockIdx.x;
    if (j == kb) return;
    __shared__ unsigned int diag[TILE][TILE];
    __shared__ unsigned int cur[TILE][TILE];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const size_t rowBase = localTileRow * TILE * pitch;
    const unsigned int* dg = dist + rowBase + kb * TILE;
    unsigned int* d = dist + rowBase + j * TILE;
    unsigned int* p = path + rowBase + j * TILE;

    diag[ty][tx] = dg[ty * pitch + tx];
    diag[ty][tx + HALF] = dg[ty * pitch + tx + HALF];
    diag[ty + HALF][tx] = dg[(ty + HALF) * pitch + tx];
    diag[ty + HALF][tx + HALF] = dg[(ty + HALF) * pitch + tx + HALF];
    unsigned int d00 = d[ty * pitch + tx];
    unsigned int d01 = d[ty * pitch + tx + HALF];
    unsigned int d10 = d[(ty + HALF) * pitch + tx];
    unsigned int d11 = d[(ty + HALF) * pitch + tx + HALF];
    unsigned int p00 = 0, p01 = 0, p10 = 0, p11 = 0;
    bool c00 = false, c01 = false, c10 = false, c11 = false;
    cur[ty][tx] = d00;
    cur[ty][tx + HALF] = d01;
    cur[ty + HALF][tx] = d10;
    cur[ty + HALF][tx + HALF] = d11;
    __syncthreads();

    const unsigned int kBase = (unsigned int)(kb * TILE);
    for (int k = 0; k < TILE; ++k) {
        const unsigned int a0 = diag[ty][k];
        const unsigned int a1 = diag[ty + HALF][k];
        const unsigned int b0 = cur[k][tx];
        const unsigned int b1 = cur[k][tx + HALF];
        const unsigned int gk = kBase + k;
        bool w;
        w = false; relax(d00, p00, w, a0, b0, gk); if (w) { cur[ty][tx] = d00; c00 = true; }
        w = false; relax(d01, p01, w, a0, b1, gk); if (w) { cur[ty][tx + HALF] = d01; c01 = true; }
        w = false; relax(d10, p10, w, a1, b0, gk); if (w) { cur[ty + HALF][tx] = d10; c10 = true; }
        w = false; relax(d11, p11, w, a1, b1, gk); if (w) { cur[ty + HALF][tx + HALF] = d11; c11 = true; }
        __syncthreads();
    }

    if (c00) { d[ty * pitch + tx] = d00; p[ty * pitch + tx] = p00; }
    if (c01) { d[ty * pitch + tx + HALF] = d01; p[ty * pitch + tx + HALF] = p01; }
    if (c10) { d[(ty + HALF) * pitch + tx] = d10; p[(ty + HALF) * pitch + tx] = p10; }
    if (c11) { d[(ty + HALF) * pitch + tx + HALF] = d11; p[(ty + HALF) * pitch + tx + HALF] = p11; }
    unsigned int* sg = stage + j * TILE_ELEMS;
    sg[ty * TILE + tx] = d00;
    sg[ty * TILE + tx + HALF] = d01;
    sg[(ty + HALF) * TILE + tx] = d10;
    sg[(ty + HALF) * TILE + tx + HALF] = d11;
}

// Phase 2 (column): tiles (i, kb) for local tile-rows i != kb, using the
// pivot tile taken from the tile-major pivot tile-row `rowBuf`.
// One block per local tile-row, starting at localTileRowStart.
__global__ void __launch_bounds__(1024)
phase2ColKernel(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
                const size_t pitch, const size_t kb,
                const unsigned int* __restrict__ rowBuf,
                const size_t firstTileRow, const size_t localTileRowStart) {
    const size_t lt = localTileRowStart + blockIdx.x;
    if (firstTileRow + lt == kb) return;
    __shared__ unsigned int diag[TILE][TILE];
    __shared__ unsigned int cur[TILE][TILE];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const unsigned int* dg = rowBuf + kb * TILE_ELEMS;
    const size_t base = lt * TILE * pitch + kb * TILE;
    unsigned int* d = dist + base;
    unsigned int* p = path + base;

    diag[ty][tx] = dg[ty * TILE + tx];
    diag[ty][tx + HALF] = dg[ty * TILE + tx + HALF];
    diag[ty + HALF][tx] = dg[(ty + HALF) * TILE + tx];
    diag[ty + HALF][tx + HALF] = dg[(ty + HALF) * TILE + tx + HALF];
    unsigned int d00 = d[ty * pitch + tx];
    unsigned int d01 = d[ty * pitch + tx + HALF];
    unsigned int d10 = d[(ty + HALF) * pitch + tx];
    unsigned int d11 = d[(ty + HALF) * pitch + tx + HALF];
    unsigned int p00 = 0, p01 = 0, p10 = 0, p11 = 0;
    bool c00 = false, c01 = false, c10 = false, c11 = false;
    cur[ty][tx] = d00;
    cur[ty][tx + HALF] = d01;
    cur[ty + HALF][tx] = d10;
    cur[ty + HALF][tx + HALF] = d11;
    __syncthreads();

    const unsigned int kBase = (unsigned int)(kb * TILE);
    for (int k = 0; k < TILE; ++k) {
        const unsigned int a0 = cur[ty][k];
        const unsigned int a1 = cur[ty + HALF][k];
        const unsigned int b0 = diag[k][tx];
        const unsigned int b1 = diag[k][tx + HALF];
        const unsigned int gk = kBase + k;
        bool w;
        w = false; relax(d00, p00, w, a0, b0, gk); if (w) { cur[ty][tx] = d00; c00 = true; }
        w = false; relax(d01, p01, w, a0, b1, gk); if (w) { cur[ty][tx + HALF] = d01; c01 = true; }
        w = false; relax(d10, p10, w, a1, b0, gk); if (w) { cur[ty + HALF][tx] = d10; c10 = true; }
        w = false; relax(d11, p11, w, a1, b1, gk); if (w) { cur[ty + HALF][tx + HALF] = d11; c11 = true; }
        __syncthreads();
    }

    if (c00) { d[ty * pitch + tx] = d00; p[ty * pitch + tx] = p00; }
    if (c01) { d[ty * pitch + tx + HALF] = d01; p[ty * pitch + tx + HALF] = p01; }
    if (c10) { d[(ty + HALF) * pitch + tx] = d10; p[(ty + HALF) * pitch + tx] = p10; }
    if (c11) { d[(ty + HALF) * pitch + tx + HALF] = d11; p[(ty + HALF) * pitch + tx + HALF] = p11; }
}

// Phase 3: tiles (i, j) for local tile-rows i != kb and tile columns
// j != kb in [j0, j0 + gridDim.x), using the column tile (i, kb) from the
// local matrix and the row tile (kb, j) from the tile-major pivot tile-row.
// Grid: (tiles in j, local tile-rows).
__global__ void __launch_bounds__(1024)
phase3Kernel(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
             const size_t pitch, const size_t kb,
             const unsigned int* __restrict__ rowBuf,
             const size_t firstTileRow, const size_t localTileRowStart,
             const size_t j0) {
    const size_t j = j0 + blockIdx.x;
    const size_t lt = localTileRowStart + blockIdx.y;
    if (j == kb || firstTileRow + lt == kb) return;
    __shared__ unsigned int col[TILE][TILE];
    __shared__ unsigned int row[TILE][TILE];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const unsigned int* cs = dist + lt * TILE * pitch + kb * TILE;
    const unsigned int* rs = rowBuf + j * TILE_ELEMS;
    const size_t base = lt * TILE * pitch + j * TILE;
    unsigned int* d = dist + base;
    unsigned int* p = path + base;

    col[ty][tx] = cs[ty * pitch + tx];
    col[ty][tx + HALF] = cs[ty * pitch + tx + HALF];
    col[ty + HALF][tx] = cs[(ty + HALF) * pitch + tx];
    col[ty + HALF][tx + HALF] = cs[(ty + HALF) * pitch + tx + HALF];
    row[ty][tx] = rs[ty * TILE + tx];
    row[ty][tx + HALF] = rs[ty * TILE + tx + HALF];
    row[ty + HALF][tx] = rs[(ty + HALF) * TILE + tx];
    row[ty + HALF][tx + HALF] = rs[(ty + HALF) * TILE + tx + HALF];
    unsigned int d00 = d[ty * pitch + tx];
    unsigned int d01 = d[ty * pitch + tx + HALF];
    unsigned int d10 = d[(ty + HALF) * pitch + tx];
    unsigned int d11 = d[(ty + HALF) * pitch + tx + HALF];
    unsigned int p00 = 0, p01 = 0, p10 = 0, p11 = 0;
    bool c00 = false, c01 = false, c10 = false, c11 = false;
    __syncthreads();

    const unsigned int kBase = (unsigned int)(kb * TILE);
    #pragma unroll 16
    for (int k = 0; k < TILE; ++k) {
        const unsigned int a0 = col[ty][k];
        const unsigned int a1 = col[ty + HALF][k];
        const unsigned int b0 = row[k][tx];
        const unsigned int b1 = row[k][tx + HALF];
        const unsigned int gk = kBase + k;
        relax(d00, p00, c00, a0, b0, gk);
        relax(d01, p01, c01, a0, b1, gk);
        relax(d10, p10, c10, a1, b0, gk);
        relax(d11, p11, c11, a1, b1, gk);
    }

    if (c00) { d[ty * pitch + tx] = d00; p[ty * pitch + tx] = p00; }
    if (c01) { d[ty * pitch + tx + HALF] = d01; p[ty * pitch + tx + HALF] = p01; }
    if (c10) { d[(ty + HALF) * pitch + tx] = d10; p[(ty + HALF) * pitch + tx] = p10; }
    if (c11) { d[(ty + HALF) * pitch + tx + HALF] = d11; p[(ty + HALF) * pitch + tx + HALF] = p11; }
}

// ---------------------------------------------------------------------------
// Distributed blocked Floyd-Warshall driver
// ---------------------------------------------------------------------------
struct LocalState {
    const Decomp* dc;
    int rank;
    size_t firstTileRow;
    size_t numTileRows;
    unsigned int* dDist;
    unsigned int* dPath;
    unsigned int* dRow[2];     // tile-major pivot tile-row buffers (device), by pivot parity
    unsigned int* hRow[2];     // pinned host copies used for MPI, by pivot parity
    cudaStream_t stream;       // compute stream
    cudaStream_t copyStream;   // host<->device copies
    cudaEvent_t evAll;
    std::vector<cudaEvent_t> evIn;    // per chunk: H2D copy done
    std::vector<cudaEvent_t> evComp;  // per chunk: phase 2 (row) done
    std::vector<cudaEvent_t> evOut;   // per chunk: D2H copy done
    std::vector<size_t> chunkStart;   // chunk boundaries in tile columns (size C+1)
    std::vector<MPI_Request> req[2];  // per chunk broadcast requests, by pivot parity
    std::vector<int> owner;           // owner rank of each tile-row
};

static inline size_t tileOff(const size_t j) { return j * TILE_ELEMS; }

static int numChunks(const LocalState& st) { return (int)st.chunkStart.size() - 1; }

// Chunk processing order for pivot kb: the chunk containing the pivot tile
// (needed first, for phase 2) comes first, then the rest in order.
static void chunkOrder(const LocalState& st, const size_t kb, std::vector<int>& order) {
    const int C = numChunks(st);
    int pc = 0;
    while (pc + 1 < C && st.chunkStart[pc + 1] <= kb) ++pc;
    order.clear();
    order.push_back(pc);
    for (int c = 0; c < C; ++c) if (c != pc) order.push_back(c);
}

// Spin until the CUDA event has completed, keeping MPI progressing.
static void waitEventWithProgress(const cudaEvent_t ev, std::vector<MPI_Request>& reqs) {
    while (true) {
        const cudaError_t q = cudaEventQuery(ev);
        if (q == cudaSuccess) break;
        if (q != cudaErrorNotReady) CUDA_CHECK(q);
        if (!reqs.empty()) {
            int flag = 0;
            MPI_Testall((int)reqs.size(), reqs.data(), &flag, MPI_STATUSES_IGNORE);
        }
    }
}

// Launches phase 2 (column) [if withPhase2] and phase 3 for local tile-rows
// [lt0, lt1) and tile columns [j0, j1).
static void computeTiles(LocalState& st, const size_t kb, const unsigned int* rowBuf,
                         const size_t lt0, const size_t lt1, const size_t j0, const size_t j1,
                         const bool withPhase2) {
    if (lt1 <= lt0) return;
    const size_t pitch = st.dc->nPad;
    const dim3 block(HALF, HALF);
    const unsigned int rows = (unsigned int)(lt1 - lt0);
    if (withPhase2) {
        phase2ColKernel<<<rows, block, 0, st.stream>>>(st.dDist, st.dPath, pitch, kb, rowBuf,
                                                       st.firstTileRow, lt0);
    }
    if (j1 > j0) {
        phase3Kernel<<<dim3((unsigned int)(j1 - j0), rows), block, 0, st.stream>>>(
            st.dDist, st.dPath, pitch, kb, rowBuf, st.firstTileRow, lt0, j0);
    }
}

// Owner-side pivot work for tile-row kb: phase 1, then phase 2 (row) chunk by
// chunk, each chunk being copied to the pinned host buffer as soon as it is
// done (on the copy stream, so the compute stream is not blocked).
static void computePivotRow(LocalState& st, const size_t kb, const std::vector<int>& order) {
    const size_t pitch = st.dc->nPad;
    const size_t lt = kb - st.firstTileRow;
    unsigned int* stage = st.dRow[kb & 1];
    unsigned int* host = st.hRow[kb & 1];
    const dim3 block(HALF, HALF);
    phase1Kernel<<<1, block, 0, st.stream>>>(st.dDist, st.dPath, pitch, kb, lt, stage);
    for (const int c : order) {
        const size_t j0 = st.chunkStart[c], j1 = st.chunkStart[c + 1];
        phase2RowKernel<<<(unsigned int)(j1 - j0), block, 0, st.stream>>>(
            st.dDist, st.dPath, pitch, kb, lt, stage, j0);
        CUDA_CHECK(cudaEventRecord(st.evComp[c], st.stream));
        CUDA_CHECK(cudaStreamWaitEvent(st.copyStream, st.evComp[c], 0));
        CUDA_CHECK(cudaMemcpyAsync(host + tileOff(j0), stage + tileOff(j0),
                                   (j1 - j0) * TILE_ELEMS * sizeof(unsigned int),
                                   cudaMemcpyDeviceToHost, st.copyStream));
        CUDA_CHECK(cudaEventRecord(st.evOut[c], st.copyStream));
    }
}

// Owner side: start the broadcast of each chunk of pivot kb once its host copy
// has landed.
static void postPivotSends(LocalState& st, const size_t kb, const std::vector<int>& order) {
    std::vector<MPI_Request>& reqs = st.req[kb & 1];
    for (const int c : order) {
        waitEventWithProgress(st.evOut[c], reqs);
        const size_t j0 = st.chunkStart[c], j1 = st.chunkStart[c + 1];
        MPI_Ibcast(st.hRow[kb & 1] + tileOff(j0), (int)((j1 - j0) * TILE_ELEMS), MPI_UNSIGNED,
                   st.rank, MPI_COMM_WORLD, &reqs[c]);
    }
}

// Receiver side: post the receives for all chunks of pivot kb.
static void postPivotRecvs(LocalState& st, const size_t kb, const std::vector<int>& order) {
    std::vector<MPI_Request>& reqs = st.req[kb & 1];
    for (const int c : order) {
        const size_t j0 = st.chunkStart[c], j1 = st.chunkStart[c + 1];
        MPI_Ibcast(st.hRow[kb & 1] + tileOff(j0), (int)((j1 - j0) * TILE_ELEMS), MPI_UNSIGNED,
                   st.owner[kb], MPI_COMM_WORLD, &reqs[c]);
    }
}

void floydWarshall(LocalState& st) {
    const Decomp& dc = *st.dc;
    const size_t T = dc.T;
    if (T == 0) return;
    const size_t L = st.numTileRows;
    std::vector<int> order, nextOrder;

    // Prologue: pivot 0.
    chunkOrder(st, 0, order);
    if (st.rank == st.owner[0]) {
        computePivotRow(st, 0, order);
        postPivotSends(st, 0, order);
    } else {
        postPivotRecvs(st, 0, order);
    }

    for (size_t kb = 0; kb < T; ++kb) {
        const bool isOwner = (st.rank == st.owner[kb]);
        const bool hasNext = (kb + 1 < T);
        const bool isNextOwner = hasNext && (st.rank == st.owner[kb + 1]);
        unsigned int* rowBuf = st.dRow[kb & 1];
        chunkOrder(st, kb, order);

        if (hasNext) {
            // The buffers of parity (kb+1)&1 were last used for pivot kb-1.
            // Receives into them finished in step kb-1, but if this rank was
            // the sender of pivot kb-1 its broadcasts may still be in flight
            // (receivers may read directly from our buffer), so complete
            // them before the buffers are reused.
            std::vector<MPI_Request>& prevReqs = st.req[(kb + 1) & 1];
            MPI_Waitall((int)prevReqs.size(), prevReqs.data(), MPI_STATUSES_IGNORE);

            // Post the receives for the next pivot early so they can progress
            // while this step runs (collective order is identical on all ranks).
            chunkOrder(st, kb + 1, nextOrder);
            if (!isNextOwner) postPivotRecvs(st, kb + 1, nextOrder);
        }

        // Rows handled in the first (pipelined) pass: either only the next
        // pivot tile-row (lookahead) or all local tile-rows.
        const size_t ltNext = isNextOwner ? (kb + 1 - st.firstTileRow) : 0;
        const size_t r0 = isNextOwner ? ltNext : 0;
        const size_t r1 = isNextOwner ? ltNext + 1 : L;

        if (isOwner) {
            // The pivot tile-row is already complete on the device.
            computeTiles(st, kb, rowBuf, r0, r1, 0, T, true);
        } else {
            // Consume the chunks as they arrive.
            bool first = true;
            for (const int c : order) {
                MPI_Wait(&st.req[kb & 1][c], MPI_STATUS_IGNORE);
                const size_t j0 = st.chunkStart[c], j1 = st.chunkStart[c + 1];
                CUDA_CHECK(cudaMemcpyAsync(rowBuf + tileOff(j0), st.hRow[kb & 1] + tileOff(j0),
                                           (j1 - j0) * TILE_ELEMS * sizeof(unsigned int),
                                           cudaMemcpyHostToDevice, st.copyStream));
                CUDA_CHECK(cudaEventRecord(st.evIn[c], st.copyStream));
                CUDA_CHECK(cudaStreamWaitEvent(st.stream, st.evIn[c], 0));
                computeTiles(st, kb, rowBuf, r0, r1, j0, j1, first);
                first = false;
            }
        }

        if (isNextOwner) {
            // Lookahead: tile-row kb+1 is now up to date for pivot kb, so
            // compute pivot kb+1 and start sending it, then finish the
            // remaining tile-rows of this step while the broadcast runs.
            computePivotRow(st, kb + 1, nextOrder);
            computeTiles(st, kb, rowBuf, 0, ltNext, 0, T, true);
            computeTiles(st, kb, rowBuf, ltNext + 1, L, 0, T, true);
            CUDA_CHECK(cudaEventRecord(st.evAll, st.stream));
            postPivotSends(st, kb + 1, nextOrder);
        } else {
            CUDA_CHECK(cudaEventRecord(st.evAll, st.stream));
        }

        // Wait for this step's GPU work; keep the next pivot's transfers
        // progressing meanwhile.
        if (hasNext) {
            waitEventWithProgress(st.evAll, st.req[(kb + 1) & 1]);
        } else {
            CUDA_CHECK(cudaEventSynchronize(st.evAll));
        }
    }
    for (int b = 0; b < 2; ++b) {
        MPI_Waitall((int)st.req[b].size(), st.req[b].data(), MPI_STATUSES_IGNORE);
    }
    CUDA_CHECK(cudaStreamSynchronize(st.stream));
    CUDA_CHECK(cudaStreamSynchronize(st.copyStream));
}

// ---------------------------------------------------------------------------
// Validation (rank 0, OpenMP)
// ---------------------------------------------------------------------------
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
    const size_t lim = std::min(numNodes, static_cast<size_t>(10));
    bool ok = true;
    #pragma omp parallel for collapse(2) schedule(static) reduction(&& : ok)
    for (size_t i = 0; i < lim; ++i) {
        for (size_t j = 0; j < lim; ++j) {
            for (size_t k = 0; k < numNodes; ++k) {
                const unsigned int distIJ = dist[idx2(j, i, numNodes)];
                const unsigned int distIK = dist[idx2(k, i, numNodes)];
                const unsigned int distKJ = dist[idx2(j, k, numNodes)];
                if (distIK < INF && distKJ < INF) {
                    if (distIK + distKJ < distIJ) ok = false;
                }
            }
        }
    }
    if (!ok) {
        // Report the first violation in sequential order.
        for (size_t i = 0; i < lim; ++i) {
            for (size_t j = 0; j < lim; ++j) {
                for (size_t k = 0; k < numNodes; ++k) {
                    const unsigned int distIJ = dist[idx2(j, i, numNodes)];
                    const unsigned int distIK = dist[idx2(k, i, numNodes)];
                    const unsigned int distKJ = dist[idx2(j, k, numNodes)];
                    if (distIK < INF && distKJ < INF && distIK + distKJ < distIJ) {
                        printf("Validation failed: triangle inequality violated at [%zu,%zu,%zu]\n",
                               i, j, k);
                        return false;
                    }
                }
            }
        }
        return false;
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

// Gather the (unpadded) rows of all ranks into rank 0's full matrix, in
// chunks that stay within the int count limits of MPI.
static void gatherResult(const Decomp& dc, const LocalState& st, std::vector<unsigned int>& dist) {
    const size_t n = dc.n;
    const size_t myRows = dc.realRows(st.rank);
    const size_t myRowBegin = dc.rowBegin(st.rank);
    std::vector<unsigned int> tmp;
    unsigned int* dst = nullptr;
    if (st.rank == 0) {
        dst = dist.data() + myRowBegin * n;
    } else if (myRows > 0) {
        tmp.resize(myRows * n);
        dst = tmp.data();
    }
    if (myRows > 0) {
        CUDA_CHECK(cudaMemcpy2D(dst, n * sizeof(unsigned int), st.dDist,
                                dc.nPad * sizeof(unsigned int), n * sizeof(unsigned int),
                                myRows, cudaMemcpyDeviceToHost));
    }

    const size_t maxRowsPerMsg = std::max<size_t>(1, (size_t)INT_MAX / std::max<size_t>(n, 1));
    if (st.rank == 0) {
        for (int r = 1; r < dc.P; ++r) {
            const size_t rows = dc.realRows(r);
            unsigned int* base = dist.data() + dc.rowBegin(r) * n;
            for (size_t r0 = 0; r0 < rows; r0 += maxRowsPerMsg) {
                const size_t cnt = std::min(maxRowsPerMsg, rows - r0);
                MPI_Recv(base + r0 * n, (int)(cnt * n), MPI_UNSIGNED, r, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    } else {
        for (size_t r0 = 0; r0 < myRows; r0 += maxRowsPerMsg) {
            const size_t cnt = std::min(maxRowsPerMsg, myRows - r0);
            MPI_Send(tmp.data() + r0 * n, (int)(cnt * n), MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD);
        }
    }
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, nRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

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

    // Select the GPU: one per rank on each node, round-robin.
    {
        MPI_Comm nodeComm;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
        int localRank = 0;
        MPI_Comm_rank(nodeComm, &localRank);
        MPI_Comm_free(&nodeComm);
        int devCount = 0;
        CUDA_CHECK(cudaGetDeviceCount(&devCount));
        if (devCount <= 0) {
            fprintf(stderr, "Rank %d: no CUDA device available\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        CUDA_CHECK(cudaSetDevice(localRank % devCount));
    }

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, OpenMP threads per rank: %d\n", nRanks, omp_get_max_threads());
    }

    Decomp dc;
    dc.n = numNodes;
    dc.nPad = (numNodes + TILE - 1) / TILE * TILE;
    dc.T = dc.nPad / TILE;
    dc.P = nRanks;

    LocalState st;
    st.dc = &dc;
    st.rank = rank;
    st.firstTileRow = dc.tileRowBegin(rank);
    st.numTileRows = dc.tileRowEnd(rank) - dc.tileRowBegin(rank);
    st.owner.resize(dc.T);
    for (int r = 0; r < nRanks; ++r) {
        for (size_t t = dc.tileRowBegin(r); t < dc.tileRowEnd(r); ++t) st.owner[t] = r;
    }

    const size_t localRowsPad = st.numTileRows * TILE;
    const size_t localElems = localRowsPad * dc.nPad;
    const size_t rowElems = (size_t)TILE * dc.nPad;

    // Allocate matrices (rank 0 keeps the full result for output / validation)
    std::vector<unsigned int> dist;
    if (rank == 0) dist.resize(numNodes * numNodes);
    std::vector<unsigned int> hLocal(localElems);

    // Initialize
    if (rank == 0) printf("Initializing graph...\n");
    initializeDistanceMatrix(hLocal, dc, dc.rowBegin(rank), dc.realRows(rank), localRowsPad,
                             1, MAX_DISTANCE);

    // Chunking of the pivot tile-row for pipelined broadcasts.
    {
        const size_t rowBytes = rowElems * sizeof(unsigned int);
        size_t C = std::max<size_t>(1, rowBytes / CHUNK_TARGET_BYTES);
        C = std::min<size_t>(C, (size_t)MAX_CHUNKS);
        C = std::min<size_t>(C, std::max<size_t>(dc.T, 1));
        st.chunkStart.resize(C + 1);
        for (size_t c = 0; c <= C; ++c) st.chunkStart[c] = dc.T * c / C;
        st.evIn.resize(C);
        st.evComp.resize(C);
        st.evOut.resize(C);
        for (size_t c = 0; c < C; ++c) {
            CUDA_CHECK(cudaEventCreateWithFlags(&st.evIn[c], cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&st.evComp[c], cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&st.evOut[c], cudaEventDisableTiming));
        }
        st.req[0].assign(C, MPI_REQUEST_NULL);
        st.req[1].assign(C, MPI_REQUEST_NULL);
    }

    st.dDist = st.dPath = nullptr;
    st.dRow[0] = st.dRow[1] = nullptr;
    st.hRow[0] = st.hRow[1] = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&st.stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&st.copyStream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&st.evAll, cudaEventDisableTiming));
    if (localElems > 0) {
        CUDA_CHECK(cudaMalloc(&st.dDist, localElems * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&st.dPath, localElems * sizeof(unsigned int)));
        CUDA_CHECK(cudaMemcpyAsync(st.dDist, hLocal.data(), localElems * sizeof(unsigned int),
                                   cudaMemcpyHostToDevice, st.stream));
        const dim3 pb(256);
        const dim3 pg((unsigned int)((dc.nPad + 255) / 256), (unsigned int)localRowsPad);
        initPathKernel<<<pg, pb, 0, st.stream>>>(st.dPath, dc.nPad, localRowsPad, dc.rowBegin(rank));
    }
    if (rowElems > 0) {
        for (int b = 0; b < 2; ++b) {
            CUDA_CHECK(cudaMalloc(&st.dRow[b], rowElems * sizeof(unsigned int)));
            CUDA_CHECK(cudaHostAlloc(&st.hRow[b], rowElems * sizeof(unsigned int), cudaHostAllocDefault));
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(st.stream));
    CUDA_CHECK(cudaGetLastError());
    std::vector<unsigned int>().swap(hLocal);

    // Run Floyd-Warshall
    if (rank == 0) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    floydWarshall(st);
    CUDA_CHECK(cudaGetLastError());
    MPI_Barrier(MPI_COMM_WORLD);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Collect the result on rank 0
    gatherResult(dc, st, dist);

    int exitCode = 0;
    if (rank == 0) {
        printf("Computation time: %ld ms\n", (long)duration.count());
        
        // Calculate operations per second
        // Floyd-Warshall has O(n³) complexity
        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);
        
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
        fflush(stdout);
    }
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (st.dDist) cudaFree(st.dDist);
    if (st.dPath) cudaFree(st.dPath);
    for (int b = 0; b < 2; ++b) {
        if (st.dRow[b]) cudaFree(st.dRow[b]);
        if (st.hRow[b]) cudaFreeHost(st.hRow[b]);
    }
    for (size_t c = 0; c < st.evIn.size(); ++c) {
        cudaEventDestroy(st.evIn[c]);
        cudaEventDestroy(st.evComp[c]);
        cudaEventDestroy(st.evOut[c]);
    }
    cudaEventDestroy(st.evAll);
    cudaStreamDestroy(st.stream);
    cudaStreamDestroy(st.copyStream);

    MPI_Finalize();
    return exitCode;
}
