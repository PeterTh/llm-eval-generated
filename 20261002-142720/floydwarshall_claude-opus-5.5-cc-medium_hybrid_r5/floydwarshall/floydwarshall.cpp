// Hybrid MPI + OpenMP + CUDA blocked Floyd-Warshall.
//
// Decomposition: the (padded) distance matrix is split into 64x64 tiles. Each
// MPI rank owns a contiguous band of tile rows and keeps it resident on its GPU.
// For every pivot block kb the owning rank computes the pivot tile (phase 1) and
// the pivot tile row (phase 2, row part) and broadcasts that 64 x N strip to all
// ranks. Every rank then updates the pivot tile column of its band (phase 2,
// column part) and all remaining tiles of its band (phase 3). The owner of the
// next pivot strip computes it first (look-ahead), so the broadcast of strip
// kb+1 overlaps with the bulk phase-3 work of iteration kb on all GPUs.
// OpenMP is used for host-side initialization (with an exact jump-ahead of the
// rand_r sequence) and validation.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

// Tile size and thread-block geometry (each thread handles 4x4 elements)
constexpr int TILE = 64;
constexpr int TDIM = 16;
constexpr int PER = TILE / TDIM;
constexpr unsigned int NO_UPDATE = 0xFFFFFFFFu;

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), \
                    __FILE__, __LINE__);                                          \
            MPI_Abort(MPI_COMM_WORLD, 1);                                         \
        }                                                                         \
    } while (0)

// Index calculation for flattened 2D array
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------

// Seed of rand_r after `calls` invocations, assuming the glibc LCG
// (3 steps of next = next * 1103515245 + 12345 per call).
static unsigned int randrJump(unsigned int seed, uint64_t calls) {
    uint64_t steps = calls * 3;
    unsigned int accMul = 1, accAdd = 0;
    unsigned int mul = 1103515245u, add = 12345u;
    while (steps) {
        if (steps & 1) {
            accMul *= mul;
            accAdd = accAdd * mul + add;
        }
        add = add * mul + add;
        mul *= mul;
        steps >>= 1;
    }
    return seed * accMul + accAdd;
}

static bool randrJumpSupported() {
    unsigned int s = 42;
    for (int c = 1; c <= 64; ++c) {
        rand_r(&s);
        if (s != randrJump(42, c)) return false;
    }
    return true;
}

// Fill rows [rowBegin, rowEnd) of the distance matrix (row-major, leading
// dimension ld, padding = INF). Produces exactly the same values as the
// sequential rand_r-based initialization over the whole numNodes^2 matrix.
void initializeDistanceRows(unsigned int* local, const size_t numNodes, const size_t ld,
                            const size_t rowBegin, const size_t rowEnd,
                            const unsigned int rangeMin, const unsigned int rangeMax) {
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;
    const size_t realEnd = std::min(rowEnd, numNodes);
    const size_t nLocalRows = rowEnd - rowBegin;

#pragma omp parallel for schedule(static)
    for (size_t r = 0; r < nLocalRows; ++r) {
        unsigned int* row = local + r * ld;
        for (size_t c = 0; c < ld; ++c) row[c] = INF;
    }
    if (realEnd <= rowBegin) return;

    const size_t first = rowBegin * numNodes;
    const size_t last = realEnd * numNodes;

    if (randrJumpSupported()) {
#pragma omp parallel
        {
            const size_t nt = omp_get_num_threads(), t = omp_get_thread_num();
            const size_t total = last - first;
            const size_t b = first + total * t / nt, e = first + total * (t + 1) / nt;
            unsigned int seed = randrJump(42, b);
            for (size_t i = b; i < e; ++i) {
                const unsigned int v =
                    rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
                local[(i / numNodes - rowBegin) * ld + i % numNodes] = v;
            }
        }
    } else {
        unsigned int seed = 42;
        for (size_t i = 0; i < first; ++i) rand_r(&seed);
        for (size_t i = first; i < last; ++i) {
            const unsigned int v =
                rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
            local[(i / numNodes - rowBegin) * ld + i % numNodes] = v;
        }
    }

    // Set diagonal to zero (distance from node to itself is 0)
    for (size_t i = rowBegin; i < realEnd; ++i) local[(i - rowBegin) * ld + i] = 0;
}

// Path matrix initial state: path[i][j] = i (row index).
__global__ void initPathKernel(unsigned int* __restrict__ path, size_t ld, size_t rows,
                               unsigned int rowBegin) {
    const size_t total = rows * ld;
    for (size_t e = blockIdx.x * (size_t)blockDim.x + threadIdx.x; e < total;
         e += (size_t)gridDim.x * blockDim.x)
        path[e] = rowBegin + (unsigned int)(e / ld);
}

// ---------------------------------------------------------------------------
// Blocked Floyd-Warshall kernels
// ---------------------------------------------------------------------------

// Phases 1 and 2 (tiles depending on themselves).
//   MODE 0: pivot tile          C = C (x) C
//   MODE 1: pivot-row tile      C = D (x) C   (D = pivot tile)
//   MODE 2: pivot-column tile   C = C (x) D
// For MODE 1 tiles are indexed by blockIdx.x as column tiles of the pivot strip;
// for MODE 2 by blockIdx.x as local row tiles. Tiles `skip`/`skip2` are not processed.
template <int MODE>
__global__ void __launch_bounds__(TDIM* TDIM)
    fwDependentKernel(unsigned int* dist, unsigned int* __restrict__ path,
                      const unsigned int* diag, size_t ld, int kb, int skip,
                      int skip2, int tileOffset) {
    __shared__ unsigned int sS[TILE][TILE + 1];
    __shared__ unsigned int sD[MODE == 0 ? 1 : TILE][TILE + 1];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * TDIM + tx;
    const int t = blockIdx.x;
    if (MODE != 0 && (t + tileOffset == skip || t + tileOffset == skip2)) return;

    size_t rowOff, colOff;
    if (MODE == 0) {
        rowOff = 0;
        colOff = (size_t)kb * TILE;
    } else if (MODE == 1) {
        rowOff = 0;
        colOff = (size_t)t * TILE;
    } else {
        rowOff = (size_t)t * TILE;
        colOff = (size_t)kb * TILE;
    }
    unsigned int* C = dist + rowOff * ld + colOff;
    unsigned int* P = path + rowOff * ld + colOff;

#pragma unroll
    for (int q = 0; q < TILE * TILE / (TDIM * TDIM); ++q) {
        const int e = tid + q * TDIM * TDIM;
        const int r = e / TILE, c = e % TILE;
        sS[r][c] = C[r * ld + c];
        if (MODE != 0) sD[r][c] = diag[r * ld + (size_t)kb * TILE + c];
    }
    __syncthreads();

    unsigned int cur[PER][PER], pth[PER][PER];
#pragma unroll
    for (int i = 0; i < PER; ++i)
#pragma unroll
        for (int j = 0; j < PER; ++j) {
            cur[i][j] = sS[ty + TDIM * i][tx + TDIM * j];
            pth[i][j] = NO_UPDATE;
        }

    const unsigned int kBase = (unsigned int)kb * TILE;
    for (int k = 0; k < TILE; ++k) {
        unsigned int a[PER], b[PER];
#pragma unroll
        for (int i = 0; i < PER; ++i) a[i] = (MODE == 1) ? sD[ty + TDIM * i][k] : sS[ty + TDIM * i][k];
#pragma unroll
        for (int j = 0; j < PER; ++j) b[j] = (MODE == 2) ? sD[k][tx + TDIM * j] : sS[k][tx + TDIM * j];
        // Row k / column k of the self tile never change during step k, so
        // in-place updates here do not race with the reads above.
#pragma unroll
        for (int i = 0; i < PER; ++i)
#pragma unroll
            for (int j = 0; j < PER; ++j) {
                const unsigned int nd = a[i] + b[j];
                if (nd < cur[i][j]) {
                    cur[i][j] = nd;
                    pth[i][j] = kBase + k;
                    sS[ty + TDIM * i][tx + TDIM * j] = nd;
                }
            }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < PER; ++i)
#pragma unroll
        for (int j = 0; j < PER; ++j) {
            const size_t o = (size_t)(ty + TDIM * i) * ld + tx + TDIM * j;
            C[o] = cur[i][j];
            if (pth[i][j] != NO_UPDATE) P[o] = pth[i][j];
        }
}

// Phase 3: all tiles not in the pivot row/column of the local band.
// blockIdx.x = column tile, blockIdx.y = local row tile.
__global__ void __launch_bounds__(TDIM* TDIM)
    fwPhase3Kernel(unsigned int* dist, unsigned int* __restrict__ path,
                   const unsigned int* strip, size_t ld, int kb,
                   int tileOffset, int skipRow) {
    const int bx = blockIdx.x, by = blockIdx.y;
    const int gRow = by + tileOffset;
    if (bx == kb || gRow == kb || gRow == skipRow) return;

    __shared__ unsigned int sA[TILE][TILE + 1];  // dist[i][k]
    __shared__ unsigned int sB[TILE][TILE];      // dist[k][j]

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * TDIM + tx;

    const unsigned int* A = dist + (size_t)by * TILE * ld + (size_t)kb * TILE;
    const unsigned int* Bm = strip + (size_t)bx * TILE;
#pragma unroll
    for (int q = 0; q < TILE * TILE / (TDIM * TDIM); ++q) {
        const int e = tid + q * TDIM * TDIM;
        const int r = e / TILE, c = e % TILE;
        sA[r][c] = A[r * ld + c];
        sB[r][c] = Bm[r * ld + c];
    }

    unsigned int* C = dist + (size_t)by * TILE * ld + (size_t)bx * TILE;
    unsigned int* P = path + (size_t)by * TILE * ld + (size_t)bx * TILE;
    unsigned int cur[PER][PER], pth[PER][PER];
#pragma unroll
    for (int i = 0; i < PER; ++i)
#pragma unroll
        for (int j = 0; j < PER; ++j) {
            cur[i][j] = C[(size_t)(ty + TDIM * i) * ld + tx + TDIM * j];
            pth[i][j] = NO_UPDATE;
        }
    __syncthreads();

    const unsigned int kBase = (unsigned int)kb * TILE;
#pragma unroll 8
    for (int k = 0; k < TILE; ++k) {
        unsigned int a[PER], b[PER];
#pragma unroll
        for (int i = 0; i < PER; ++i) a[i] = sA[ty + TDIM * i][k];
#pragma unroll
        for (int j = 0; j < PER; ++j) b[j] = sB[k][tx + TDIM * j];
#pragma unroll
        for (int i = 0; i < PER; ++i)
#pragma unroll
            for (int j = 0; j < PER; ++j) {
                const unsigned int nd = a[i] + b[j];
                if (nd < cur[i][j]) {
                    cur[i][j] = nd;
                    pth[i][j] = kBase + k;
                }
            }
    }

#pragma unroll
    for (int i = 0; i < PER; ++i)
#pragma unroll
        for (int j = 0; j < PER; ++j) {
            if (pth[i][j] != NO_UPDATE) {
                const size_t o = (size_t)(ty + TDIM * i) * ld + tx + TDIM * j;
                C[o] = cur[i][j];
                P[o] = pth[i][j];
            }
        }
}

// ---------------------------------------------------------------------------
// Distributed driver
// ---------------------------------------------------------------------------

struct Band {
    int rank, size;
    int numTiles;         // tiles per dimension (padded)
    size_t ld;            // padded dimension
    int tileBegin;        // first global tile row owned
    int tileCount;        // number of tile rows owned
    std::vector<int> tileStarts;  // per rank
};

// Two-level communication topology: ranks on the same node exchange pivot
// strips GPU-to-GPU through CUDA IPC; ranks with the same node-local index on
// different nodes exchange them through MPI (host staged). If nodes host
// different numbers of ranks, every rank is treated as its own node.
struct Topology {
    MPI_Comm node = MPI_COMM_SELF, cross = MPI_COMM_WORLD, flat = MPI_COMM_WORLD;
    int nodeRank = 0, nodeSize = 1, crossSize = 1;
    std::vector<int> nodeRankOf, crossRankOf;  // indexed by world rank
};

static Topology makeTopology(int rank, int size) {
    Topology t;
    MPI_Comm node;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    int nr, ns, mn, mx;
    MPI_Comm_rank(node, &nr);
    MPI_Comm_size(node, &ns);
    MPI_Allreduce(&ns, &mn, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&ns, &mx, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (mn == mx && ns > 1) {
        t.node = node;
        t.nodeRank = nr;
        t.nodeSize = ns;
        MPI_Comm_split(MPI_COMM_WORLD, nr, rank, &t.cross);
    } else {
        MPI_Comm_free(&node);
        MPI_Comm_dup(MPI_COMM_WORLD, &t.cross);
        t.node = MPI_COMM_SELF;
    }
    MPI_Comm_dup(MPI_COMM_WORLD, &t.flat);
    MPI_Comm_size(t.cross, &t.crossSize);
    int cr;
    MPI_Comm_rank(t.cross, &cr);
    t.nodeRankOf.resize(size);
    t.crossRankOf.resize(size);
    MPI_Allgather(&t.nodeRank, 1, MPI_INT, t.nodeRankOf.data(), 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Allgather(&cr, 1, MPI_INT, t.crossRankOf.data(), 1, MPI_INT, MPI_COMM_WORLD);
    return t;
}

static void freeTopology(Topology& t) {
    if (t.node != MPI_COMM_SELF) MPI_Comm_free(&t.node);
    MPI_Comm_free(&t.cross);
    MPI_Comm_free(&t.flat);
}

static int ownerOf(const Band& b, int tile) {
    // last rank whose range starts at or before `tile` (never an empty range)
    return (int)(std::upper_bound(b.tileStarts.begin(), b.tileStarts.end(), tile) -
                 b.tileStarts.begin()) - 1;
}

// Communication resources for distributing pivot strips (double-buffered):
//  outStrip: strip published to node peers (IPC exported) / filled from MPI
//  inStrip:  local copy of a strip pulled from a node peer
//  hStrip:   pinned host staging for inter-node MPI
struct Exchange {
    Topology topo;
    bool intra = false, inter = false, ipcAlloc = false;
    size_t stripElems = 0, stripBytes = 0;
    int myDev = 0;
    unsigned int* outStrip[2] = {nullptr, nullptr};
    unsigned int* inStrip[2] = {nullptr, nullptr};
    unsigned int* hStrip[2] = {nullptr, nullptr};
    cudaEvent_t readyEv[2] = {}, copiedEv[2] = {}, h2dDone[2] = {}, d2hDone = {};
    std::vector<unsigned int*> peerOut;
    std::vector<cudaEvent_t> peerReady, peerCopied;
    std::vector<int> peerDev;
};

Exchange setupExchange(const Band& b, const Topology& topoIn) {
    Exchange x;
    x.topo = topoIn;
    Topology& topo = x.topo;
    const int ns = topo.nodeSize;
    x.stripElems = (size_t)TILE * b.ld;
    x.stripBytes = x.stripElems * sizeof(unsigned int);
    x.intra = x.ipcAlloc = ns > 1;
    x.peerOut.assign(2 * ns, nullptr);
    x.peerReady.assign(2 * ns, nullptr);
    x.peerCopied.assign(2 * ns, nullptr);
    x.peerDev.assign(ns, -1);
    CUDA_CHECK(cudaGetDevice(&x.myDev));

    if (b.size > 1)
        for (int i = 0; i < 2; ++i) CUDA_CHECK(cudaMalloc(&x.outStrip[i], x.stripBytes));

    if (x.ipcAlloc) {
        for (int i = 0; i < 2; ++i) {
            CUDA_CHECK(cudaMalloc(&x.inStrip[i], x.stripBytes));
            CUDA_CHECK(cudaEventCreateWithFlags(&x.readyEv[i],
                                                cudaEventDisableTiming | cudaEventInterprocess));
            CUDA_CHECK(cudaEventCreateWithFlags(&x.copiedEv[i],
                                                cudaEventDisableTiming | cudaEventInterprocess));
        }
        // Exchange IPC handles of the published strips and events within the node.
        // Peer allocations are mapped in a context on the peer's own GPU, so this
        // works without peer-to-peer support (the driver stages such copies).
        struct Handles {
            cudaIpcMemHandle_t mem[2];
            cudaIpcEventHandle_t ready[2], copied[2];
            char busId[32];
        } mine;
        std::vector<Handles> all(ns);
        CUDA_CHECK(cudaDeviceGetPCIBusId(mine.busId, sizeof(mine.busId), x.myDev));
        for (int i = 0; i < 2; ++i) {
            CUDA_CHECK(cudaIpcGetMemHandle(&mine.mem[i], x.outStrip[i]));
            CUDA_CHECK(cudaIpcGetEventHandle(&mine.ready[i], x.readyEv[i]));
            CUDA_CHECK(cudaIpcGetEventHandle(&mine.copied[i], x.copiedEv[i]));
        }
        MPI_Allgather(&mine, sizeof(Handles), MPI_BYTE, all.data(), sizeof(Handles), MPI_BYTE,
                      topo.node);
        int ok = 1;
        for (int p = 0; p < ns && ok; ++p) {
            if (p == topo.nodeRank) {
                x.peerDev[p] = x.myDev;
                for (int i = 0; i < 2; ++i) {
                    x.peerOut[2 * p + i] = x.outStrip[i];
                    x.peerReady[2 * p + i] = x.readyEv[i];
                    x.peerCopied[2 * p + i] = x.copiedEv[i];
                }
                continue;
            }
            if (cudaDeviceGetByPCIBusId(&x.peerDev[p], all[p].busId) != cudaSuccess) {
                x.peerDev[p] = -1;
                ok = 0;
                break;
            }
            for (int i = 0; i < 2 && ok; ++i) {
                void* ptr = nullptr;
                CUDA_CHECK(cudaSetDevice(x.peerDev[p]));
                if (cudaIpcOpenMemHandle(&ptr, all[p].mem[i], cudaIpcMemLazyEnablePeerAccess) !=
                    cudaSuccess)
                    ok = 0;
                x.peerOut[2 * p + i] = static_cast<unsigned int*>(ptr);
                CUDA_CHECK(cudaSetDevice(x.myDev));
                if (ok && (cudaIpcOpenEventHandle(&x.peerReady[2 * p + i], all[p].ready[i]) !=
                               cudaSuccess ||
                           cudaIpcOpenEventHandle(&x.peerCopied[2 * p + i],
                                                  all[p].copied[i]) != cudaSuccess))
                    ok = 0;
            }
        }
        cudaGetLastError();  // clear any error from a failed IPC attempt
        MPI_Allreduce(MPI_IN_PLACE, &ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (!ok) {
            // Fall back to host-staged MPI between all ranks
            x.intra = false;
            topo.cross = topo.flat;
            topo.crossSize = b.size;
            topo.nodeRank = 0;
            for (int r = 0; r < b.size; ++r) {
                topo.crossRankOf[r] = r;
                topo.nodeRankOf[r] = 0;
            }
        }
    }
    x.inter = topo.crossSize > 1;

    if (x.inter) {
        for (int i = 0; i < 2; ++i) {
            CUDA_CHECK(cudaMallocHost(&x.hStrip[i], x.stripBytes));
            CUDA_CHECK(cudaEventCreateWithFlags(&x.h2dDone[i], cudaEventDisableTiming));
        }
        CUDA_CHECK(cudaEventCreateWithFlags(&x.d2hDone, cudaEventDisableTiming));
    }
    return x;
}

void teardownExchange(Exchange& x) {
    const int ns = (int)x.peerDev.size();
    if (x.ipcAlloc) {
        for (int p = 0; p < ns; ++p) {
            if (p == x.topo.nodeRank || x.peerDev[p] < 0) continue;
            for (int i = 0; i < 2; ++i) {
                if (x.peerReady[2 * p + i]) cudaEventDestroy(x.peerReady[2 * p + i]);
                if (x.peerCopied[2 * p + i]) cudaEventDestroy(x.peerCopied[2 * p + i]);
            }
            CUDA_CHECK(cudaSetDevice(x.peerDev[p]));
            for (int i = 0; i < 2; ++i)
                if (x.peerOut[2 * p + i]) cudaIpcCloseMemHandle(x.peerOut[2 * p + i]);
            CUDA_CHECK(cudaSetDevice(x.myDev));
        }
        MPI_Barrier(MPI_COMM_WORLD);  // peers no longer reference our buffers
        for (int i = 0; i < 2; ++i) {
            CUDA_CHECK(cudaFree(x.inStrip[i]));
            CUDA_CHECK(cudaEventDestroy(x.readyEv[i]));
            CUDA_CHECK(cudaEventDestroy(x.copiedEv[i]));
        }
    }
    for (int i = 0; i < 2; ++i) {
        if (x.outStrip[i]) CUDA_CHECK(cudaFree(x.outStrip[i]));
        if (x.inter) {
            CUDA_CHECK(cudaFreeHost(x.hStrip[i]));
            CUDA_CHECK(cudaEventDestroy(x.h2dDone[i]));
        }
    }
    if (x.inter) CUDA_CHECK(cudaEventDestroy(x.d2hDone));
}

// Runs the blocked algorithm on device-resident band (dDist/dPath).
void floydWarshall(unsigned int* dDist, unsigned int* dPath, const Band& b, Exchange& x,
                   cudaStream_t stream) {
    const int nt = b.numTiles;
    const size_t ld = b.ld;
    const size_t stripElems = x.stripElems;
    const size_t stripBytes = x.stripBytes;
    const dim3 block(TDIM, TDIM);
    const Topology& topo = x.topo;
    const bool intra = x.intra, inter = x.inter;
    const int ns = topo.nodeSize;
    auto& outStrip = x.outStrip;
    auto& inStrip = x.inStrip;
    auto& hStrip = x.hStrip;
    auto& readyEv = x.readyEv;
    auto& copiedEv = x.copiedEv;
    auto& h2dDone = x.h2dDone;
    auto& d2hDone = x.d2hDone;
    auto& peerOut = x.peerOut;
    auto& peerReady = x.peerReady;
    auto& peerCopied = x.peerCopied;

    auto ownsTile = [&](int t) { return t >= b.tileBegin && t < b.tileBegin + b.tileCount; };
    auto localRow = [&](int t) { return dDist + (size_t)(t - b.tileBegin) * stripElems; };
    auto localPath = [&](int t) { return dPath + (size_t)(t - b.tileBegin) * stripElems; };

    // Before overwriting outStrip[buf], wait until all node peers pulled its
    // previous contents. (Their copies were enqueued before the last node
    // barrier, which this rank has already passed.)
    auto waitPeersCopied = [&](int buf) {
        for (int p = 0; p < ns; ++p)
            if (p != topo.nodeRank)
                CUDA_CHECK(cudaStreamWaitEvent(stream, peerCopied[2 * p + buf], 0));
    };

    // Phase 1 + phase 2 (row part) for pivot block kb on its owner, then
    // publish the strip to node peers and stage it for inter-node transfer.
    auto computePivotStrip = [&](int kb) {
        unsigned int* d = localRow(kb);
        unsigned int* p = localPath(kb);
        const int buf = kb & 1;
        fwDependentKernel<0><<<1, block, 0, stream>>>(d, p, d, ld, kb, -1, -1, 0);
        fwDependentKernel<1><<<nt, block, 0, stream>>>(d, p, d, ld, kb, kb, -1, 0);
        if (inter) {
            CUDA_CHECK(cudaMemcpyAsync(hStrip[buf], d, stripBytes, cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaEventRecord(d2hDone, stream));
        }
        if (intra) {
            waitPeersCopied(buf);
            CUDA_CHECK(cudaMemcpyAsync(outStrip[buf], d, stripBytes, cudaMemcpyDeviceToDevice,
                                       stream));
            CUDA_CHECK(cudaEventRecord(readyEv[buf], stream));
        }
    };

    // Distribute strip kb; returns the device pointer holding it on this rank.
    auto shareStrip = [&](int kb) -> const unsigned int* {
        const int owner = ownerOf(b, kb);
        if (!intra && !inter) return localRow(kb);
        const int buf = kb & 1;
        const int srcNodeRank = intra ? topo.nodeRankOf[owner] : 0;
        const bool isSource = topo.nodeRank == srcNodeRank;  // holds strip on this node

        // Inter-node: owner -> ranks with the same node-local index
        if (inter && isSource) {
            if (b.rank == owner) {
                CUDA_CHECK(cudaEventSynchronize(d2hDone));
                MPI_Bcast(hStrip[buf], (int)stripElems, MPI_UNSIGNED, topo.crossRankOf[owner],
                          topo.cross);
            } else {
                CUDA_CHECK(cudaEventSynchronize(h2dDone[buf]));
                MPI_Bcast(hStrip[buf], (int)stripElems, MPI_UNSIGNED, topo.crossRankOf[owner],
                          topo.cross);
                if (intra) waitPeersCopied(buf);
                CUDA_CHECK(cudaMemcpyAsync(outStrip[buf], hStrip[buf], stripBytes,
                                           cudaMemcpyHostToDevice, stream));
                CUDA_CHECK(cudaEventRecord(h2dDone[buf], stream));
                if (intra) CUDA_CHECK(cudaEventRecord(readyEv[buf], stream));
            }
        }

        // Intra-node: peers pull the strip GPU-to-GPU from the node's source
        if (intra) {
            MPI_Barrier(topo.node);  // source has published (event recorded)
            if (!isSource) {
                CUDA_CHECK(cudaStreamWaitEvent(stream, peerReady[2 * srcNodeRank + buf], 0));
                CUDA_CHECK(cudaMemcpyAsync(inStrip[buf], peerOut[2 * srcNodeRank + buf],
                                           stripBytes, cudaMemcpyDefault, stream));
                CUDA_CHECK(cudaEventRecord(copiedEv[buf], stream));
                return inStrip[buf];
            }
        }
        return b.rank == owner ? localRow(kb) : outStrip[buf];
    };

    if (ownsTile(0)) computePivotStrip(0);
    const unsigned int* strip = shareStrip(0);

    for (int kb = 0; kb < nt; ++kb) {
        if (b.tileCount > 0) {
            const int next = kb + 1;
            const bool lookahead = next < nt && ownsTile(next);
            if (lookahead) {
                // Critical path first: finish the next pivot strip and publish it
                fwDependentKernel<2><<<1, block, 0, stream>>>(
                    localRow(next), localPath(next), strip, ld, kb, kb, -1, next);
                fwPhase3Kernel<<<dim3(nt, 1), block, 0, stream>>>(
                    localRow(next), localPath(next), strip, ld, kb, next, -1);
                computePivotStrip(next);
            }
            const int skipRow = lookahead ? next : -1;
            // Phase 2, column part, and phase 3 on the rest of the local band
            fwDependentKernel<2><<<b.tileCount, block, 0, stream>>>(
                dDist, dPath, strip, ld, kb, kb, skipRow, b.tileBegin);
            fwPhase3Kernel<<<dim3(nt, b.tileCount), block, 0, stream>>>(
                dDist, dPath, strip, ld, kb, b.tileBegin, skipRow);
        }
        if (kb + 1 < nt) strip = shareStrip(kb + 1);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (intra) MPI_Barrier(topo.node);  // peers done reading our buffers
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
    const size_t m = std::min(numNodes, static_cast<size_t>(10));
    // Find the first violation (in sequential order) in parallel
    size_t firstBad = SIZE_MAX;
#pragma omp parallel for collapse(2) reduction(min : firstBad) schedule(static)
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < m; ++j) {
            for (size_t k = 0; k < numNodes; ++k) {
                const unsigned int distIJ = dist[idx2(j, i, numNodes)];
                const unsigned int distIK = dist[idx2(k, i, numNodes)];
                const unsigned int distKJ = dist[idx2(j, k, numNodes)];

                // Check for overflow before addition
                if (distIK < INF && distKJ < INF && distIK + distKJ < distIJ) {
                    firstBad = std::min(firstBad, (i * m + j) * numNodes + k);
                    break;
                }
            }
        }
    }
    if (firstBad != SIZE_MAX) {
        const size_t k = firstBad % numNodes, ij = firstBad / numNodes;
        printf("Validation failed: triangle inequality violated at [%zu,%zu,%zu]\n",
               ij / m, ij % m, k);
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

    // Select a GPU per node-local rank
    Topology topo = makeTopology(rank, size);
    {
        int localRank = 0;
        {
            MPI_Comm local;
            MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
                                &local);
            MPI_Comm_rank(local, &localRank);
            MPI_Comm_free(&local);
        }
        int devCount = 0;
        CUDA_CHECK(cudaGetDeviceCount(&devCount));
        if (devCount < 1) {
            fprintf(stderr, "No CUDA device available\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        CUDA_CHECK(cudaSetDevice(localRank % devCount));
    }

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Decompose padded tile rows across ranks
    Band band;
    band.rank = rank;
    band.size = size;
    band.numTiles = (int)((numNodes + TILE - 1) / TILE);
    band.ld = (size_t)band.numTiles * TILE;
    band.tileStarts.resize(size + 1);
    for (int r = 0; r <= size; ++r)
        band.tileStarts[r] = (int)((long long)band.numTiles * r / size);
    band.tileBegin = band.tileStarts[rank];
    band.tileCount = band.tileStarts[rank + 1] - band.tileBegin;

    const size_t rowBegin = (size_t)band.tileBegin * TILE;
    const size_t localRows = (size_t)band.tileCount * TILE;
    const size_t localElems = localRows * band.ld;

    // Initialize
    if (rank == 0) printf("Initializing graph...\n");
    unsigned int* hLocal = nullptr;
    CUDA_CHECK(cudaMallocHost(&hLocal, std::max<size_t>(localElems, 1) * sizeof(unsigned int)));
    initializeDistanceRows(hLocal, numNodes, band.ld, rowBegin, rowBegin + localRows, 1,
                           MAX_DISTANCE);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    unsigned int *dDist = nullptr, *dPath = nullptr;
    CUDA_CHECK(cudaMalloc(&dDist, std::max<size_t>(localElems, 1) * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&dPath, std::max<size_t>(localElems, 1) * sizeof(unsigned int)));
    if (localElems > 0)
        initPathKernel<<<1024, 256, 0, stream>>>(dPath, band.ld, localRows, (unsigned int)rowBegin);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<unsigned int> dist;
    std::vector<unsigned int> path;
    if (rank == 0) {
        dist.resize(numNodes * numNodes);
        path.resize(numNodes * numNodes);
    }

    // Row counts (real rows only) for the gather
    std::vector<int> rowCounts(size), rowDispls(size);
    for (int r = 0; r < size; ++r) {
        const size_t b0 = std::min((size_t)band.tileStarts[r] * TILE, numNodes);
        const size_t b1 = std::min((size_t)band.tileStarts[r + 1] * TILE, numNodes);
        rowCounts[r] = (int)(b1 - b0);
        rowDispls[r] = (int)b0;
    }
    const size_t myRealRows = rowCounts[rank];
    MPI_Datatype rowType;
    MPI_Type_contiguous((int)std::max<size_t>(numNodes, 1), MPI_UNSIGNED, &rowType);
    MPI_Type_commit(&rowType);
    std::vector<unsigned int> sendBuf(rank == 0 ? 0 : myRealRows * numNodes);

    Exchange exchange = setupExchange(band, topo);

    // Page-lock the result buffers so the device-to-host copies run at full speed
    auto pinVector = [](std::vector<unsigned int>& v) {
        if (!v.empty())
            CUDA_CHECK(cudaHostRegister(v.data(), v.size() * sizeof(unsigned int),
                                        cudaHostRegisterDefault));
    };
    pinVector(dist);
    pinVector(path);
    pinVector(sendBuf);

    // Ranks on rank 0's node hand their result bands to rank 0 through CUDA IPC
    struct PeerBand {
        int worldRank, dev;
        unsigned int *dist, *path;
        cudaStream_t stream;
    };
    std::vector<PeerBand> ipcBands;  // rank 0 only
    std::vector<int> viaIpc(size, 0);
    if (exchange.intra) {
        int onRootNode = rank == 0;
        MPI_Allreduce(MPI_IN_PLACE, &onRootNode, 1, MPI_INT, MPI_MAX, topo.node);
        MPI_Allgather(&onRootNode, 1, MPI_INT, viaIpc.data(), 1, MPI_INT, MPI_COMM_WORLD);
        struct BandHandles {
            cudaIpcMemHandle_t dist, path;
            int worldRank;
        } mine;
        CUDA_CHECK(cudaIpcGetMemHandle(&mine.dist, dDist));
        CUDA_CHECK(cudaIpcGetMemHandle(&mine.path, dPath));
        mine.worldRank = rank;
        std::vector<BandHandles> all(topo.nodeSize);
        MPI_Allgather(&mine, sizeof(BandHandles), MPI_BYTE, all.data(), sizeof(BandHandles),
                      MPI_BYTE, topo.node);
        if (rank == 0) {
            for (int p = 0; p < topo.nodeSize; ++p) {
                const int wr = all[p].worldRank;
                if (wr == rank || rowCounts[wr] == 0) continue;
                PeerBand pb{wr, exchange.peerDev[p], nullptr, nullptr, nullptr};
                void *pd = nullptr, *pp = nullptr;
                CUDA_CHECK(cudaSetDevice(exchange.peerDev[p]));
                CUDA_CHECK(cudaIpcOpenMemHandle(&pd, all[p].dist, cudaIpcMemLazyEnablePeerAccess));
                CUDA_CHECK(cudaIpcOpenMemHandle(&pp, all[p].path, cudaIpcMemLazyEnablePeerAccess));
                CUDA_CHECK(cudaSetDevice(exchange.myDev));
                pb.dist = static_cast<unsigned int*>(pd);
                pb.path = static_cast<unsigned int*>(pp);
                CUDA_CHECK(cudaStreamCreateWithFlags(&pb.stream, cudaStreamNonBlocking));
                ipcBands.push_back(pb);
            }
        }
    }
    std::vector<int> mpiRowCounts(rowCounts);
    for (int r = 0; r < size; ++r)
        if (viaIpc[r] || r == 0) mpiRowCounts[r] = 0;

    // Run Floyd-Warshall
    if (rank == 0) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    if (localElems > 0)
        CUDA_CHECK(cudaMemcpyAsync(dDist, hLocal, localElems * sizeof(unsigned int),
                                   cudaMemcpyHostToDevice, stream));

    floydWarshall(dDist, dPath, band, exchange, stream);

    // Collect results on rank 0: node peers through CUDA IPC, other ranks via MPI
    auto gather = [&](const unsigned int* dSrc, std::vector<unsigned int>& dst, bool isPath) {
        const size_t pitch = numNodes * sizeof(unsigned int);
        if (rank == 0) {
            for (const PeerBand& pb : ipcBands)
                CUDA_CHECK(cudaMemcpy2DAsync(dst.data() + (size_t)rowDispls[pb.worldRank] * numNodes,
                                             pitch, isPath ? pb.path : pb.dist,
                                             band.ld * sizeof(unsigned int), pitch,
                                             rowCounts[pb.worldRank], cudaMemcpyDefault,
                                             pb.stream));
        }
        unsigned int* target = rank == 0 ? dst.data() : sendBuf.data();
        if (myRealRows > 0 && (rank == 0 || !viaIpc[rank]))
            CUDA_CHECK(cudaMemcpy2D(target, pitch, dSrc, band.ld * sizeof(unsigned int), pitch,
                                    myRealRows, cudaMemcpyDeviceToHost));
        if (size > 1) {
            if (rank == 0)
                MPI_Gatherv(MPI_IN_PLACE, 0, rowType, dst.data(), mpiRowCounts.data(),
                            rowDispls.data(), rowType, 0, MPI_COMM_WORLD);
            else
                MPI_Gatherv(sendBuf.data(), mpiRowCounts[rank], rowType, nullptr, nullptr,
                            nullptr, rowType, 0, MPI_COMM_WORLD);
        }
    };
    gather(dDist, dist, false);
    gather(dPath, path, true);
    for (const PeerBand& pb : ipcBands) CUDA_CHECK(cudaStreamSynchronize(pb.stream));

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (rank == 0) {
        for (const PeerBand& pb : ipcBands) {
            CUDA_CHECK(cudaStreamDestroy(pb.stream));
            CUDA_CHECK(cudaSetDevice(pb.dev));
            CUDA_CHECK(cudaIpcCloseMemHandle(pb.dist));
            CUDA_CHECK(cudaIpcCloseMemHandle(pb.path));
            CUDA_CHECK(cudaSetDevice(exchange.myDev));
        }
    }
    for (auto* v : {&dist, &path, &sendBuf})
        if (!v->empty()) CUDA_CHECK(cudaHostUnregister(v->data()));
    MPI_Type_free(&rowType);
    teardownExchange(exchange);
    freeTopology(topo);
    CUDA_CHECK(cudaFree(dDist));
    CUDA_CHECK(cudaFree(dPath));
    CUDA_CHECK(cudaFreeHost(hLocal));
    CUDA_CHECK(cudaStreamDestroy(stream));

    int exitCode = 0;
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());

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
    }

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
