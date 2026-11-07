// Floyd-Warshall All-Pairs Shortest Path Benchmark
//
// Hybrid MPI + OpenMP + CUDA implementation.
//
//  * MPI   : the (padded) distance matrix is partitioned by tile-rows across
//            ranks; every rank drives one GPU.  For each pivot block K the
//            owner rank computes the diagonal tile and the row panel; a staged
//            copy of the panel is distributed to all ranks in column-tile
//            chunks: inside a node through an MPI-3 shared-memory window (the
//            readers upload straight from the owner's pinned staging buffer),
//            between nodes through a binomial tree of point-to-point messages
//            over the node leaders.  Everything is pipelined per chunk.
//  * CUDA  : blocked Floyd-Warshall (Katz/Kider style, B = 64, 4x4 register
//            blocking per thread).  The panels used in the off-panel update are
//            recorded in their *pre-pivot* state ("staged"), which makes the
//            result -- distance matrix AND path matrix -- bit-identical to the
//            classic sequential k-i-j algorithm, including tie-breaking.
//            Two streams: a high priority stream computes the next pivot panel
//            (lookahead) while the bulk phase-3 update of the current block
//            runs on the main stream.
//  * OpenMP: host-side work (matrix initialisation, padding, packing for the
//            final gather and the validation).

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

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

// Blocking parameters for the GPU kernels
constexpr int B = 64;        // tile edge length
constexpr int T = 16;        // thread block edge length (T x T threads)
constexpr int E = B / T;     // elements per thread per dimension
constexpr int CHUNK_TILES = 64;  // column tiles per transfer chunk (1 MiB)

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        cudaError_t err__ = (call);                                                   \
        if (err__ != cudaSuccess) {                                                   \
            fprintf(stderr, "CUDA error '%s' at %s:%d\n", cudaGetErrorString(err__),  \
                    __FILE__, __LINE__);                                              \
            MPI_Abort(MPI_COMM_WORLD, 1);                                             \
        }                                                                             \
    } while (0)

// Index calculation for flattened 2D array
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

// ---------------------------------------------------------------------------
// CUDA kernels
//
// Local layout on each rank: dist/path hold nLocalRows x N values, row-major
// (element (i,j) at i*N + j).  N is the padded node count (multiple of B).
//
// Staged buffers (k0 = K*B is the first pivot of block K):
//   Rs  [nBlocks][B][B] : tile-major row panel, Rs[J][k][j] = dist[k0+k][J*B+j]
//                         right before pivot k0+k
//   Dcol[B][B]          : Dcol[i][k] = dist[k0+i][k0+k] right before pivot k0+k
//   Cs  [nLocalRows][B] : Cs[i][k]  = dist[i][k0+k] right before pivot k0+k
// ---------------------------------------------------------------------------

// Phase 1: diagonal tile (K,K).  Single thread block, run by the owner rank.
__global__ void __launch_bounds__(T * T)
phase1Kernel(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
             const size_t N, const int K, const size_t localRow0,
             unsigned int* __restrict__ Rs, unsigned int* __restrict__ Dcol) {
    __shared__ unsigned int s[B][B + 1];

    const int tx = threadIdx.x, ty = threadIdx.y, tid = ty * T + tx;
    const size_t colBase = static_cast<size_t>(K) * B;
    unsigned int p[E][E];

#pragma unroll
    for (int a = 0; a < E; ++a) {
#pragma unroll
        for (int b = 0; b < E; ++b) {
            const int r = ty + a * T, c = tx + b * T;
            const size_t g = (localRow0 + r) * N + colBase + c;
            s[r][c] = dist[g];
            p[a][b] = path[g];
        }
    }
    __syncthreads();

    for (int k = 0; k < B; ++k) {
        // Record the pre-pivot state of pivot row / column k
        if (tid < B) {
            Rs[(colBase + k) * B + tid] = s[k][tid];
        } else if (tid < 2 * B) {
            Dcol[(tid - B) * B + k] = s[tid - B][k];
        }
        unsigned int sik[E], skj[E];
#pragma unroll
        for (int a = 0; a < E; ++a) sik[a] = s[ty + a * T][k];
#pragma unroll
        for (int b = 0; b < E; ++b) skj[b] = s[k][tx + b * T];
        // Row k and column k cannot change during step k (dist[k][k] == 0 and
        // all distances are non-negative), so the only hazard is between the
        // writes of this step and the pivot reads of the next one: a single
        // barrier at the end of the step is sufficient.

        const unsigned int kg = static_cast<unsigned int>(colBase) + k;
#pragma unroll
        for (int a = 0; a < E; ++a) {
#pragma unroll
            for (int b = 0; b < E; ++b) {
                const unsigned int nd = sik[a] + skj[b];
                if (nd < s[ty + a * T][tx + b * T]) {
                    s[ty + a * T][tx + b * T] = nd;
                    p[a][b] = kg;
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int a = 0; a < E; ++a) {
#pragma unroll
        for (int b = 0; b < E; ++b) {
            const int r = ty + a * T, c = tx + b * T;
            const size_t g = (localRow0 + r) * N + colBase + c;
            dist[g] = s[r][c];
            path[g] = p[a][b];
        }
    }
}

// Phase 2 (row panel): tiles (K,J) for J in [tileStart, tileStart + gridDim.x),
// J != K.  Run by the owner rank.
__global__ void __launch_bounds__(T * T)
phase2RowKernel(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
                const size_t N, const int K, const size_t localRow0, const int tileStart,
                unsigned int* __restrict__ Rs, const unsigned int* __restrict__ Dcol) {
    __shared__ unsigned int s[B][B + 1];
    __shared__ unsigned int dc[B][B + 1];

    const int J = tileStart + static_cast<int>(blockIdx.x);
    if (J == K) return;
    const int tx = threadIdx.x, ty = threadIdx.y, tid = ty * T + tx;
    const size_t colBase = static_cast<size_t>(J) * B;
    unsigned int p[E][E];

#pragma unroll
    for (int a = 0; a < E; ++a) {
#pragma unroll
        for (int b = 0; b < E; ++b) {
            const int r = ty + a * T, c = tx + b * T;
            const size_t g = (localRow0 + r) * N + colBase + c;
            s[r][c] = dist[g];
            p[a][b] = path[g];
            dc[r][c] = Dcol[r * B + c];
        }
    }
    __syncthreads();

    for (int k = 0; k < B; ++k) {
        if (tid < B) Rs[(colBase + k) * B + tid] = s[k][tid];
        unsigned int sik[E], skj[E];
#pragma unroll
        for (int a = 0; a < E; ++a) sik[a] = dc[ty + a * T][k];
#pragma unroll
        for (int b = 0; b < E; ++b) skj[b] = s[k][tx + b * T];
        // Row k and column k cannot change during step k (dist[k][k] == 0 and
        // all distances are non-negative), so the only hazard is between the
        // writes of this step and the pivot reads of the next one: a single
        // barrier at the end of the step is sufficient.

        const unsigned int kg = static_cast<unsigned int>(K) * B + k;
#pragma unroll
        for (int a = 0; a < E; ++a) {
#pragma unroll
            for (int b = 0; b < E; ++b) {
                const unsigned int nd = sik[a] + skj[b];
                if (nd < s[ty + a * T][tx + b * T]) {
                    s[ty + a * T][tx + b * T] = nd;
                    p[a][b] = kg;
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int a = 0; a < E; ++a) {
#pragma unroll
        for (int b = 0; b < E; ++b) {
            const int r = ty + a * T, c = tx + b * T;
            const size_t g = (localRow0 + r) * N + colBase + c;
            dist[g] = s[r][c];
            path[g] = p[a][b];
        }
    }
}

// Phase 2 (column panel): tiles (I,K) for the local tile-rows
// [rowStart, rowStart + gridDim.x), I != K.  Tile-rows with global index skipA
// or skipB are left out.
__global__ void __launch_bounds__(T * T)
phase2ColKernel(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
                const size_t N, const int K, const int firstBlock, const int rowStart,
                const int skipA, const int skipB,
                const unsigned int* __restrict__ Rs, unsigned int* __restrict__ Cs) {
    __shared__ unsigned int s[B][B + 1];
    __shared__ unsigned int dr[B][B + 1];

    const int Il = rowStart + static_cast<int>(blockIdx.x);
    const int Ig = firstBlock + Il;
    if (Ig == K || Ig == skipA || Ig == skipB) return;
    const int tx = threadIdx.x, ty = threadIdx.y, tid = ty * T + tx;
    const size_t rowBase = static_cast<size_t>(Il) * B;
    const size_t colBase = static_cast<size_t>(K) * B;
    unsigned int p[E][E];

#pragma unroll
    for (int a = 0; a < E; ++a) {
#pragma unroll
        for (int b = 0; b < E; ++b) {
            const int r = ty + a * T, c = tx + b * T;
            const size_t g = (rowBase + r) * N + colBase + c;
            s[r][c] = dist[g];
            p[a][b] = path[g];
            dr[r][c] = Rs[(colBase + r) * B + c];
        }
    }
    __syncthreads();

    for (int k = 0; k < B; ++k) {
        if (tid < B) Cs[(rowBase + tid) * B + k] = s[tid][k];
        unsigned int sik[E], skj[E];
#pragma unroll
        for (int a = 0; a < E; ++a) sik[a] = s[ty + a * T][k];
#pragma unroll
        for (int b = 0; b < E; ++b) skj[b] = dr[k][tx + b * T];
        // Row k and column k cannot change during step k (dist[k][k] == 0 and
        // all distances are non-negative), so the only hazard is between the
        // writes of this step and the pivot reads of the next one: a single
        // barrier at the end of the step is sufficient.

        const unsigned int kg = static_cast<unsigned int>(colBase) + k;
#pragma unroll
        for (int a = 0; a < E; ++a) {
#pragma unroll
            for (int b = 0; b < E; ++b) {
                const unsigned int nd = sik[a] + skj[b];
                if (nd < s[ty + a * T][tx + b * T]) {
                    s[ty + a * T][tx + b * T] = nd;
                    p[a][b] = kg;
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int a = 0; a < E; ++a) {
#pragma unroll
        for (int b = 0; b < E; ++b) {
            const int r = ty + a * T, c = tx + b * T;
            const size_t g = (rowBase + r) * N + colBase + c;
            dist[g] = s[r][c];
            path[g] = p[a][b];
        }
    }
}

// Phase 3: tiles (I,J) for the local tile-rows [rowStart, rowStart + gridDim.y)
// and J in [tileStart, tileStart + gridDim.x), J != K.  Tile-rows with global
// index K, skipA or skipB are left out.
__global__ void __launch_bounds__(T * T)
phase3Kernel(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
             const size_t N, const int K, const int firstBlock, const int rowStart,
             const int skipA, const int skipB, const int tileStart,
             const unsigned int* __restrict__ Rs, const unsigned int* __restrict__ Cs) {
    __shared__ unsigned int sc[B][B + 1];
    __shared__ unsigned int sr[B][B + 1];

    const int Il = rowStart + static_cast<int>(blockIdx.y);
    const int Ig = firstBlock + Il;
    if (Ig == K || Ig == skipA || Ig == skipB) return;
    const int J = tileStart + static_cast<int>(blockIdx.x);
    if (J == K) return;
    const int tx = threadIdx.x, ty = threadIdx.y;
    const size_t rowBase = static_cast<size_t>(Il) * B;
    const size_t colBase = static_cast<size_t>(J) * B;
    unsigned int d[E][E], p[E][E];

#pragma unroll
    for (int a = 0; a < E; ++a) {
#pragma unroll
        for (int b = 0; b < E; ++b) {
            const int r = ty + a * T, c = tx + b * T;
            const size_t g = (rowBase + r) * N + colBase + c;
            d[a][b] = dist[g];
            p[a][b] = path[g];
            sc[r][c] = Cs[(rowBase + r) * B + c];
            sr[r][c] = Rs[(colBase + r) * B + c];
        }
    }
    __syncthreads();

    const unsigned int kBase = static_cast<unsigned int>(K) * B;
#pragma unroll 8
    for (int k = 0; k < B; ++k) {
        unsigned int sik[E], skj[E];
#pragma unroll
        for (int a = 0; a < E; ++a) sik[a] = sc[ty + a * T][k];
#pragma unroll
        for (int b = 0; b < E; ++b) skj[b] = sr[k][tx + b * T];
        const unsigned int kg = kBase + k;
#pragma unroll
        for (int a = 0; a < E; ++a) {
#pragma unroll
            for (int b = 0; b < E; ++b) {
                const unsigned int nd = sik[a] + skj[b];
                if (nd < d[a][b]) {
                    d[a][b] = nd;
                    p[a][b] = kg;
                }
            }
        }
    }

#pragma unroll
    for (int a = 0; a < E; ++a) {
#pragma unroll
        for (int b = 0; b < E; ++b) {
            const int r = ty + a * T, c = tx + b * T;
            const size_t g = (rowBase + r) * N + colBase + c;
            dist[g] = d[a][b];
            path[g] = p[a][b];
        }
    }
}

// ---------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------

// Initialise the locally owned rows [row0, row0 + nLocalRows) of the padded
// distance matrix.  The rand_r stream is consumed in exactly the same order as
// the original sequential code so the generated graph is identical.
void initializeDistanceMatrix(std::vector<unsigned int>& local, const size_t numNodes,
                              const size_t N, const size_t row0, const size_t nLocalRows,
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;

    const size_t rowEnd = std::min(row0 + nLocalRows, numNodes);
    for (size_t i = 0; i < rowEnd; ++i) {
        const bool own = (i >= row0);
        unsigned int* rowPtr = own ? &local[(i - row0) * N] : nullptr;
        for (size_t j = 0; j < numNodes; ++j) {
            const unsigned int v = rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
            if (own) rowPtr[j] = v;
        }
    }

    // Padding (nodes >= numNodes are unreachable) and zero diagonal
#pragma omp parallel for schedule(static)
    for (size_t r = 0; r < nLocalRows; ++r) {
        unsigned int* rowPtr = &local[r * N];
        const size_t i = row0 + r;
        if (i >= numNodes) {
            for (size_t j = 0; j < N; ++j) rowPtr[j] = INF;
        } else {
            for (size_t j = numNodes; j < N; ++j) rowPtr[j] = INF;
        }
        rowPtr[i] = 0;
    }
}

// The original double loop assigns path[j][i] = j and path[i][j] = i for all
// (i,j), i.e. the final value of every entry (r,c) is its row index r.
void initializePathMatrix(std::vector<unsigned int>& local, const size_t N,
                          const size_t row0, const size_t nLocalRows) {
#pragma omp parallel for schedule(static)
    for (size_t r = 0; r < nLocalRows; ++r) {
        unsigned int* rowPtr = &local[r * N];
        const unsigned int v = static_cast<unsigned int>(row0 + r);
        for (size_t j = 0; j < N; ++j) rowPtr[j] = v;
    }
}

struct Partition {
    size_t N = 0;            // padded node count
    int nBlocks = 0;         // N / B
    int firstBlock = 0;      // first tile-row owned by this rank
    int nLocalBlocks = 0;    // number of tile-rows owned by this rank
    std::vector<int> owner;  // owner rank of every tile-row
};

Partition makePartition(const size_t numNodes, const int rank, const int size) {
    Partition p;
    p.nBlocks = static_cast<int>((numNodes + B - 1) / B);
    p.N = static_cast<size_t>(p.nBlocks) * B;
    p.owner.resize(p.nBlocks);
    for (int r = 0; r < size; ++r) {
        const int b0 = static_cast<int>((static_cast<long long>(p.nBlocks) * r) / size);
        const int b1 = static_cast<int>((static_cast<long long>(p.nBlocks) * (r + 1)) / size);
        for (int b = b0; b < b1; ++b) p.owner[b] = r;
        if (r == rank) {
            p.firstBlock = b0;
            p.nLocalBlocks = b1 - b0;
        }
    }
    return p;
}

// Binomial broadcast tree over `count` participants, rooted at `root`
struct Tree {
    int parent = -1;             // participant index of the parent (-1 for the root)
    std::vector<int> children;   // participant indices of the children
};

static Tree makeTree(const int root, const int me, const int count) {
    Tree t;
    const int rel = (me - root + count) % count;
    const int lowBit = rel ? (rel & -rel) : count;
    if (rel) t.parent = ((rel ^ lowBit) + root) % count;
    for (int m = 1; m < lowBit; m <<= 1) {
        const int c = rel | m;
        if (c < count) t.children.push_back((c + root) % count);
    }
    return t;
}

// Distributed blocked Floyd-Warshall solver state
struct Solver {
    static constexpr int NBUF = 3;   // staged panels in flight (K-1, K, K+1)

    const Partition& part;
    int rank, size;

    // Device data
    unsigned int* dDist = nullptr;   // nLocalRows x N
    unsigned int* dPath = nullptr;   // nLocalRows x N
    unsigned int* dDcol = nullptr;   // staged pivot columns of the diagonal tile
    unsigned int* dCs = nullptr;     // staged pivot columns of the local column panel
    unsigned int* dRs[NBUF] = {};    // staged row panels, [nBlocks][B][B], slot = P % NBUF
    std::vector<cudaEvent_t> evChunk[NBUF];   // chunk c of the panel in a slot is on the device
    cudaStream_t sMain = nullptr;    // bulk phase-3 work
    cudaStream_t sPivot = nullptr;   // high priority: lookahead, pivot panel, transfers
    cudaEvent_t evCol[2] = {};       // phase2col(K) finished on sMain (parity of K)
    cudaEvent_t evRowNext[2] = {};   // phase3(K) on tile-row K+2 finished on sMain (parity of K)

    // Chunking of a panel along column tiles
    int nChunks = 0;
    std::vector<int> chunkStart;     // first column tile of every chunk (+ sentinel)
    size_t panelElems = 0;

    // Node topology
    MPI_Comm nodeComm = MPI_COMM_NULL;
    int nodeRank = 0, nodeSize = 1, nNodes = 1, myNode = 0;
    std::vector<int> nodeOf;         // node index of every world rank
    std::vector<int> leaderOf;       // world rank of the leader (node rank 0) of every node
    std::vector<int> readerRank;     // world ranks of the other ranks on my node

    // Shared-memory window: every node rank owns a segment of NBUF panel slots
    MPI_Win win = MPI_WIN_NULL;
    std::vector<unsigned int*> segOf;        // segment base of every node rank
    std::vector<bool> segRegistered;
    unsigned int* hFallback = nullptr;       // pinned staging buffer for a single rank

    // Outstanding communication per slot
    std::vector<MPI_Request> sendReq[NBUF];  // data sends, notifications
    std::vector<MPI_Request> ackReq[NBUF];   // acknowledgements expected from readers
    std::vector<MPI_Request> ackSendReq;     // acknowledgements sent by this rank
    int lastReadPanel = -1;                  // panel read from a provider's segment, not acked yet

    // State of the chunk-wise production of the next panel (ownership boundary)
    std::vector<char> rowReady;
    bool phase1Done = false;
    int produced = 0;

    Solver(const Partition& p, int r, int s) : part(p), rank(r), size(s) {}

    int chunkOf(const int tile) const { return tile / CHUNK_TILES; }
    int chunkTiles(const int c) const { return chunkStart[c + 1] - chunkStart[c]; }
    size_t chunkOffset(const int c) const { return static_cast<size_t>(chunkStart[c]) * B * B; }
    size_t chunkElems(const int c) const { return static_cast<size_t>(chunkTiles(c)) * B * B; }
    // Chunks of panel P are produced / transferred starting with the chunk that
    // holds the diagonal tile P (needed first by every consumer).
    int chunkInOrder(const int P, const int i) const { return (chunkOf(P) + i) % nChunks; }
    int lastChunk(const int P) const { return chunkInOrder(P, nChunks - 1); }
    static int slotOf(const int P) { return P % NBUF; }
    int dataTag(const int P, const int c) const { return slotOf(P) * nChunks + c; }
    int ackTag(const int P) const { return NBUF * nChunks + slotOf(P); }
    bool owns(const int P) const { return P >= 0 && P < part.nBlocks && part.owner[P] == rank; }
    // World rank that holds the panel of P in its segment on node n
    int providerOn(const int P, const int n) const {
        const int o = part.owner[P];
        return nodeOf[o] == n ? o : leaderOf[n];
    }
    int nodeRankOf(const int worldRank) const {
        // node ranks are assigned in increasing world rank order
        int nr = 0;
        for (int r = 0; r < worldRank; ++r) nr += (nodeOf[r] == myNode) ? 1 : 0;
        return nr;
    }
    unsigned int* hostSlot(const int worldRank, const int P) const {
        unsigned int* base = segOf.empty() ? hFallback : segOf[nodeRankOf(worldRank)];
        return base + static_cast<size_t>(slotOf(P)) * panelElems;
    }

    void init();
    void destroy();
    void waitSlotFree(int slot);
    void sendPendingAck();
    void finishChunk(int P, int c);
    void producePanelLocal(int P);
    void beginProduceChunked(int P);
    void onChunkReady(int K, int c);
    void consumeLocal(int K);
    void distribute(int K);
    void run();
};

void Solver::init() {
    const size_t N = part.N;
    const size_t nLocalRows = static_cast<size_t>(part.nLocalBlocks) * B;
    const size_t localBytes = std::max<size_t>(nLocalRows * N, 1) * sizeof(unsigned int);
    const size_t csBytes = std::max<size_t>(nLocalRows * B, 1) * sizeof(unsigned int);

    nChunks = (part.nBlocks + CHUNK_TILES - 1) / CHUNK_TILES;
    chunkStart.resize(nChunks + 1);
    for (int c = 0; c <= nChunks; ++c) chunkStart[c] = std::min(c * CHUNK_TILES, part.nBlocks);
    panelElems = static_cast<size_t>(part.nBlocks) * B * B;
    const size_t panelBytes = std::max<size_t>(panelElems, 1) * sizeof(unsigned int);
    rowReady.assign(std::max(nChunks, 1), 0);

    // Node topology
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    MPI_Comm_rank(nodeComm, &nodeRank);
    MPI_Comm_size(nodeComm, &nodeSize);
    {
        // Number the nodes by the world rank of their leader
        std::vector<int> leaderMark(size, 0);
        const int mark = (nodeRank == 0) ? 1 : 0;
        MPI_Allgather(&mark, 1, MPI_INT, leaderMark.data(), 1, MPI_INT, MPI_COMM_WORLD);
        nNodes = 0;
        for (int r = 0; r < size; ++r) {
            if (leaderMark[r]) {
                leaderOf.push_back(r);
                ++nNodes;
            }
        }
        int myLeader = rank;
        MPI_Bcast(&myLeader, 1, MPI_INT, 0, nodeComm);
        myNode = static_cast<int>(std::find(leaderOf.begin(), leaderOf.end(), myLeader) -
                                  leaderOf.begin());
        nodeOf.resize(size);
        MPI_Allgather(&myNode, 1, MPI_INT, nodeOf.data(), 1, MPI_INT, MPI_COMM_WORLD);
        for (int r = 0; r < size; ++r) {
            if (nodeOf[r] == myNode && r != rank) readerRank.push_back(r);
        }
    }

    // Shared-memory staging window (NBUF panel slots per node rank)
    if (size > 1) {
        const MPI_Aint segBytes = static_cast<MPI_Aint>(NBUF) * panelBytes;
        unsigned int* base = nullptr;
        MPI_Win_allocate_shared(segBytes, 1, MPI_INFO_NULL, nodeComm, &base, &win);
        segOf.resize(nodeSize);
        segRegistered.assign(nodeSize, false);
        for (int nr = 0; nr < nodeSize; ++nr) {
            MPI_Aint sz = 0;
            int disp = 0;
            void* ptr = nullptr;
            MPI_Win_shared_query(win, nr, &sz, &disp, &ptr);
            segOf[nr] = static_cast<unsigned int*>(ptr);
            // Pin the segment so the GPU can DMA directly from/to it
            if (cudaHostRegister(ptr, segBytes, cudaHostRegisterPortable) == cudaSuccess) {
                segRegistered[nr] = true;
            } else {
                (void)cudaGetLastError();
            }
        }
        MPI_Barrier(nodeComm);
    } else {
        CUDA_CHECK(cudaMallocHost(&hFallback, static_cast<size_t>(NBUF) * panelBytes));
    }

    // Device buffers, streams, events
    CUDA_CHECK(cudaMalloc(&dDist, localBytes));
    CUDA_CHECK(cudaMalloc(&dPath, localBytes));
    CUDA_CHECK(cudaMalloc(&dDcol, static_cast<size_t>(B) * B * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&dCs, csBytes));
    int prioLow = 0, prioHigh = 0;
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&prioLow, &prioHigh));
    CUDA_CHECK(cudaStreamCreateWithPriority(&sMain, cudaStreamNonBlocking, prioLow));
    CUDA_CHECK(cudaStreamCreateWithPriority(&sPivot, cudaStreamNonBlocking, prioHigh));
    for (int i = 0; i < 2; ++i) {
        CUDA_CHECK(cudaEventCreateWithFlags(&evCol[i], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventRecord(evCol[i], sMain));
        CUDA_CHECK(cudaEventCreateWithFlags(&evRowNext[i], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventRecord(evRowNext[i], sMain));
    }
    for (int b = 0; b < NBUF; ++b) {
        CUDA_CHECK(cudaMalloc(&dRs[b], panelBytes));
        evChunk[b].resize(std::max(nChunks, 1));
        for (cudaEvent_t& ev : evChunk[b]) {
            CUDA_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
            CUDA_CHECK(cudaEventRecord(ev, sPivot));
        }
    }
}

void Solver::destroy() {
    for (int b = 0; b < NBUF; ++b) waitSlotFree(b);
    if (!ackSendReq.empty()) {
        MPI_Waitall(static_cast<int>(ackSendReq.size()), ackSendReq.data(), MPI_STATUSES_IGNORE);
        ackSendReq.clear();
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    for (int b = 0; b < NBUF; ++b) {
        for (cudaEvent_t ev : evChunk[b]) CUDA_CHECK(cudaEventDestroy(ev));
        CUDA_CHECK(cudaFree(dRs[b]));
    }
    for (int i = 0; i < 2; ++i) {
        CUDA_CHECK(cudaEventDestroy(evCol[i]));
        CUDA_CHECK(cudaEventDestroy(evRowNext[i]));
    }
    CUDA_CHECK(cudaStreamDestroy(sMain));
    CUDA_CHECK(cudaStreamDestroy(sPivot));
    CUDA_CHECK(cudaFree(dDist));
    CUDA_CHECK(cudaFree(dPath));
    CUDA_CHECK(cudaFree(dDcol));
    CUDA_CHECK(cudaFree(dCs));
    if (win != MPI_WIN_NULL) {
        MPI_Barrier(nodeComm);
        for (int nr = 0; nr < nodeSize; ++nr) {
            if (segRegistered[nr]) CUDA_CHECK(cudaHostUnregister(segOf[nr]));
        }
        MPI_Win_free(&win);
    }
    if (hFallback) CUDA_CHECK(cudaFreeHost(hFallback));
    MPI_Comm_free(&nodeComm);
}

// Before a host slot is overwritten: all sends from it must have completed and
// all readers must have finished uploading from it.
void Solver::waitSlotFree(const int slot) {
    if (!sendReq[slot].empty()) {
        MPI_Waitall(static_cast<int>(sendReq[slot].size()), sendReq[slot].data(),
                    MPI_STATUSES_IGNORE);
        sendReq[slot].clear();
    }
    if (!ackReq[slot].empty()) {
        MPI_Waitall(static_cast<int>(ackReq[slot].size()), ackReq[slot].data(),
                    MPI_STATUSES_IGNORE);
        ackReq[slot].clear();
    }
}

// A reader acknowledges the last panel it uploaded from a provider's segment
// once the upload has finished (the provider may then reuse the slot).
void Solver::sendPendingAck() {
    if (lastReadPanel < 0) return;
    const int P = lastReadPanel;
    lastReadPanel = -1;
    CUDA_CHECK(cudaEventSynchronize(evChunk[slotOf(P)][lastChunk(P)]));
    ackSendReq.emplace_back();
    MPI_Isend(nullptr, 0, MPI_BYTE, providerOn(P, myNode), ackTag(P), MPI_COMM_WORLD,
              &ackSendReq.back());
}

// Owner of P, pivot stream: chunk c of the panel of P is complete on the
// device -> stage it on the host (if needed) and record its event.
void Solver::finishChunk(const int P, const int c) {
    unsigned int* out = dRs[slotOf(P)];
    if (size > 1) {
        CUDA_CHECK(cudaMemcpyAsync(hostSlot(rank, P) + chunkOffset(c), out + chunkOffset(c),
                                   chunkElems(c) * sizeof(unsigned int), cudaMemcpyDeviceToHost,
                                   sPivot));
    }
    CUDA_CHECK(cudaEventRecord(evChunk[slotOf(P)][c], sPivot));
}

// Owner of P whose panel of P-1 was produced locally (or P == 0): bring the
// tile-row P up to date for block P-1 and produce the panel of P with
// full-width kernels on the pivot stream.
void Solver::producePanelLocal(const int P) {
    const size_t N = part.N;
    const dim3 threads(T, T);
    const int Kprev = P - 1;
    unsigned int* out = dRs[slotOf(P)];
    const int localRow = P - part.firstBlock;
    const size_t localRow0 = static_cast<size_t>(localRow) * B;

    waitSlotFree(slotOf(P));   // host slot: sends / reader uploads of panel P-NBUF done
    if (P > 0) {
        const unsigned int* in = dRs[slotOf(Kprev)];
        // Tile-row P must be up to date for block P-1 on the main stream.  That
        // event also implies phase3(P-NBUF) has finished reading the device slot.
        CUDA_CHECK(cudaStreamWaitEvent(sPivot, evRowNext[P & 1], 0));
        CUDA_CHECK(cudaStreamWaitEvent(sPivot, evChunk[slotOf(Kprev)][lastChunk(Kprev)], 0));
        phase2ColKernel<<<1, threads, 0, sPivot>>>(dDist, dPath, N, Kprev, part.firstBlock,
                                                   localRow, -1, -1, in, dCs);
        phase3Kernel<<<dim3(part.nBlocks, 1), threads, 0, sPivot>>>(
            dDist, dPath, N, Kprev, part.firstBlock, localRow, -1, -1, 0, in, dCs);
    }
    phase1Kernel<<<1, threads, 0, sPivot>>>(dDist, dPath, N, P, localRow0, out, dDcol);
    phase2RowKernel<<<part.nBlocks, threads, 0, sPivot>>>(dDist, dPath, N, P, localRow0, 0, out,
                                                          dDcol);
    for (int i = 0; i < nChunks; ++i) finishChunk(P, chunkInOrder(P, i));
}

// Owner of P whose panel of P-1 arrives from another rank: the production is
// driven chunk by chunk from onChunkReady(P-1, c).
void Solver::beginProduceChunked(const int P) {
    waitSlotFree(slotOf(P));
    std::fill(rowReady.begin(), rowReady.end(), 0);
    phase1Done = false;
    produced = 0;
    CUDA_CHECK(cudaStreamWaitEvent(sPivot, evRowNext[P & 1], 0));
}

// Chunk c of the panel of K has just been uploaded to the device (event
// recorded on the pivot stream): issue the dependent work.
void Solver::onChunkReady(const int K, const int c) {
    const size_t N = part.N;
    const dim3 threads(T, T);
    const int slot = slotOf(K);
    const unsigned int* in = dRs[slot];
    const bool first = (c == chunkOf(K));
    const bool last = (c == lastChunk(K));
    const int lookaheadRow = owns(K + 1) ? K + 1 : -1;
    const int priorityRow = owns(K + 2) ? K + 2 : -1;

    // Main stream: phase 2 (column panel) with the first chunk, phase 3 on the
    // columns of this chunk (next-but-one pivot row first).
    if (part.nLocalBlocks > 0) {
        CUDA_CHECK(cudaStreamWaitEvent(sMain, evChunk[slot][c], 0));
        if (first) {
            phase2ColKernel<<<part.nLocalBlocks, threads, 0, sMain>>>(
                dDist, dPath, N, K, part.firstBlock, 0, lookaheadRow, -1, in, dCs);
            CUDA_CHECK(cudaEventRecord(evCol[K & 1], sMain));
        }
        if (priorityRow >= 0) {
            phase3Kernel<<<dim3(chunkTiles(c), 1), threads, 0, sMain>>>(
                dDist, dPath, N, K, part.firstBlock, priorityRow - part.firstBlock, -1, -1,
                chunkStart[c], in, dCs);
            if (last) CUDA_CHECK(cudaEventRecord(evRowNext[K & 1], sMain));
        }
        phase3Kernel<<<dim3(chunkTiles(c), part.nLocalBlocks), threads, 0, sMain>>>(
            dDist, dPath, N, K, part.firstBlock, 0, lookaheadRow, priorityRow, chunkStart[c], in,
            dCs);
    }

    // Pivot stream (owner of K+1): lookahead on tile-row K+1 and production of
    // the panel of K+1 for every chunk whose inputs are complete.
    if (lookaheadRow >= 0) {
        const int P = K + 1;
        const int localRow = P - part.firstBlock;
        const size_t localRow0 = static_cast<size_t>(localRow) * B;
        unsigned int* out = dRs[slotOf(P)];
        if (first) {
            phase2ColKernel<<<1, threads, 0, sPivot>>>(dDist, dPath, N, K, part.firstBlock,
                                                       localRow, -1, -1, in, dCs);
        }
        phase3Kernel<<<dim3(chunkTiles(c), 1), threads, 0, sPivot>>>(
            dDist, dPath, N, K, part.firstBlock, localRow, -1, -1, chunkStart[c], in, dCs);
        rowReady[c] = 1;
        if (c == chunkOf(P)) {
            phase1Kernel<<<1, threads, 0, sPivot>>>(dDist, dPath, N, P, localRow0, out, dDcol);
            phase1Done = true;
        }
        while (phase1Done && produced < nChunks && rowReady[chunkInOrder(P, produced)]) {
            const int pc = chunkInOrder(P, produced);
            phase2RowKernel<<<chunkTiles(pc), threads, 0, sPivot>>>(
                dDist, dPath, N, P, localRow0, chunkStart[pc], out, dDcol);
            finishChunk(P, pc);
            ++produced;
        }
    }
}

// Owner of K: the whole panel of K is (being) produced on the pivot stream ->
// consume it with full-width kernels on the main stream.
void Solver::consumeLocal(const int K) {
    const size_t N = part.N;
    const dim3 threads(T, T);
    const int slot = slotOf(K);
    const unsigned int* in = dRs[slot];
    const int lookaheadRow = owns(K + 1) ? K + 1 : -1;
    const int priorityRow = owns(K + 2) ? K + 2 : -1;
    if (part.nLocalBlocks == 0) return;

    CUDA_CHECK(cudaStreamWaitEvent(sMain, evChunk[slot][lastChunk(K)], 0));
    phase2ColKernel<<<part.nLocalBlocks, threads, 0, sMain>>>(
        dDist, dPath, N, K, part.firstBlock, 0, lookaheadRow, -1, in, dCs);
    CUDA_CHECK(cudaEventRecord(evCol[K & 1], sMain));
    if (priorityRow >= 0) {
        phase3Kernel<<<dim3(part.nBlocks, 1), threads, 0, sMain>>>(
            dDist, dPath, N, K, part.firstBlock, priorityRow - part.firstBlock, -1, -1, 0, in,
            dCs);
        CUDA_CHECK(cudaEventRecord(evRowNext[K & 1], sMain));
    }
    phase3Kernel<<<dim3(part.nBlocks, part.nLocalBlocks), threads, 0, sMain>>>(
        dDist, dPath, N, K, part.firstBlock, 0, lookaheadRow, priorityRow, 0, in, dCs);
}

// Distribute the staged panel of block K from its owner to every rank and
// issue the dependent GPU work as the chunks become available:
//   * between nodes: binomial tree over the node providers (the owner on its
//     node, the node leader elsewhere), chunked point-to-point messages
//     received directly into the provider's shared-memory slot;
//   * inside a node: readers are notified per chunk (0-byte message) and
//     upload the chunk from the provider's slot to their own device.
void Solver::distribute(const int K) {
    const int slot = slotOf(K);
    const int owner = part.owner[K];
    if (size <= 1) return;
    const int provider = providerOn(K, myNode);
    const int rootNode = nodeOf[owner];

    // Acknowledge the previously read panel (its upload is finished by now)
    sendPendingAck();

    if (rank == provider) {
        unsigned int* hSlot = hostSlot(rank, K);
        const Tree tree = makeTree(rootNode, myNode, nNodes);
        if (rank != owner) {
            waitSlotFree(slot);
            // Readers may still be uploading panel K-NBUF from this slot;
            // my own upload from it is covered by the chunk events.
            for (const cudaEvent_t ev : evChunk[slot]) CUDA_CHECK(cudaEventSynchronize(ev));
        }
        // Expect one acknowledgement per reader for this slot
        for (const int r : readerRank) {
            ackReq[slot].emplace_back();
            MPI_Irecv(nullptr, 0, MPI_BYTE, r, ackTag(K), MPI_COMM_WORLD, &ackReq[slot].back());
        }
        std::vector<MPI_Request> recv;
        if (rank != owner) {
            recv.resize(nChunks);
            const int src = providerOn(K, tree.parent);
            for (int c = 0; c < nChunks; ++c) {
                MPI_Irecv(hSlot + chunkOffset(c), static_cast<int>(chunkElems(c)), MPI_UNSIGNED,
                          src, dataTag(K, c), MPI_COMM_WORLD, &recv[c]);
            }
        }
        for (int i = 0; i < nChunks; ++i) {
            const int c = chunkInOrder(K, i);
            if (rank == owner) {
                CUDA_CHECK(cudaEventSynchronize(evChunk[slot][c]));   // chunk is in the slot
            } else {
                MPI_Wait(&recv[c], MPI_STATUS_IGNORE);
            }
            for (const int childNode : tree.children) {
                sendReq[slot].emplace_back();
                MPI_Isend(hSlot + chunkOffset(c), static_cast<int>(chunkElems(c)), MPI_UNSIGNED,
                          providerOn(K, childNode), dataTag(K, c), MPI_COMM_WORLD,
                          &sendReq[slot].back());
            }
            for (const int r : readerRank) {
                sendReq[slot].emplace_back();
                MPI_Isend(nullptr, 0, MPI_BYTE, r, dataTag(K, c), MPI_COMM_WORLD,
                          &sendReq[slot].back());
            }
            if (rank != owner) {
                CUDA_CHECK(cudaMemcpyAsync(dRs[slot] + chunkOffset(c), hSlot + chunkOffset(c),
                                           chunkElems(c) * sizeof(unsigned int),
                                           cudaMemcpyHostToDevice, sPivot));
                CUDA_CHECK(cudaEventRecord(evChunk[slot][c], sPivot));
                onChunkReady(K, c);
            }
        }
    } else {
        // Reader: upload every chunk from the provider's slot as it is announced
        const unsigned int* hSlot = hostSlot(provider, K);
        for (int i = 0; i < nChunks; ++i) {
            const int c = chunkInOrder(K, i);
            MPI_Recv(nullptr, 0, MPI_BYTE, provider, dataTag(K, c), MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
            CUDA_CHECK(cudaMemcpyAsync(dRs[slot] + chunkOffset(c), hSlot + chunkOffset(c),
                                       chunkElems(c) * sizeof(unsigned int),
                                       cudaMemcpyHostToDevice, sPivot));
            CUDA_CHECK(cudaEventRecord(evChunk[slot][c], sPivot));
            onChunkReady(K, c);
        }
        lastReadPanel = K;
    }
}

// Pipeline per pivot block K:
//   main stream : phase2col(K) -> phase3(K) first on tile-row K+2 (if owned,
//                 signalled by evRowNext) -> phase3(K) on the remaining local
//                 tile-rows; chunk by chunk as the panel of K arrives, or with
//                 full-width kernels when it was produced locally
//   pivot stream: (owner of K+1) phase2col(K) + phase3(K) on tile-row K+1,
//                 phase1(K+1), phase2row(K+1) + D2H; chunk by chunk or full
//                 (other ranks) H2D of the panel of K, chunk by chunk
//   host        : chunked distribution of the panel of K
void Solver::run() {
    const int nBlocks = part.nBlocks;
    if (nBlocks == 0) return;

    if (owns(0)) producePanelLocal(0);

    for (int K = 0; K < nBlocks; ++K) {
        // The device slot of K is uploaded to on the pivot stream: it was last
        // read by phase3(K-NBUF), which precedes phase2col(K-2).
        CUDA_CHECK(cudaStreamWaitEvent(sPivot, evCol[K & 1], 0));

        if (owns(K)) {
            if (owns(K + 1)) producePanelLocal(K + 1);
            consumeLocal(K);
        } else if (owns(K + 1)) {
            beginProduceChunked(K + 1);
        }
        distribute(K);
    }
    sendPendingAck();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

bool validateResult(const std::vector<unsigned int>& dist, const size_t numNodes) {
    // Basic sanity checks

    // 1. Diagonal should be zero
    {
        bool ok = true;
#pragma omp parallel for schedule(static) reduction(&& : ok)
        for (size_t i = 0; i < numNodes; ++i) {
            if (dist[idx2(i, i, numNodes)] != 0) ok = false;
        }
        if (!ok) {
            for (size_t i = 0; i < numNodes; ++i) {
                if (dist[idx2(i, i, numNodes)] != 0) {
                    printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", i, i);
                    return false;
                }
            }
        }
    }

    // 2. Triangle inequality: dist[i][k] + dist[k][j] >= dist[i][j]
    // Check a sample of paths to avoid O(n^3) validation time
    const size_t sample = std::min(numNodes, static_cast<size_t>(10));
    bool ok = true;
#pragma omp parallel for collapse(2) schedule(static) reduction(&& : ok)
    for (size_t i = 0; i < sample; ++i) {
        for (size_t j = 0; j < sample; ++j) {
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
        // Report the first violation in the original scan order
        for (size_t i = 0; i < sample; ++i) {
            for (size_t j = 0; j < sample; ++j) {
                for (size_t k = 0; k < numNodes; ++k) {
                    const unsigned int distIJ = dist[idx2(j, i, numNodes)];
                    const unsigned int distIK = dist[idx2(k, i, numNodes)];
                    const unsigned int distKJ = dist[idx2(j, k, numNodes)];
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
    MPI_Init(&argc, &argv);
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

    // One GPU per rank (round-robin over the GPUs visible on this node)
    {
        MPI_Comm nodeComm;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
        int localRank = 0;
        MPI_Comm_rank(nodeComm, &localRank);
        MPI_Comm_free(&nodeComm);
        int deviceCount = 0;
        CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
        if (deviceCount < 1) {
            fprintf(stderr, "No CUDA device found on rank %d\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        CUDA_CHECK(cudaSetDevice(localRank % deviceCount));
        CUDA_CHECK(cudaFree(nullptr));  // establish the context now
    }

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, OpenMP threads per rank: %d, tile size: %d\n", size,
               omp_get_max_threads(), B);
    }

    // Partition the padded matrix by tile-rows
    const Partition part = makePartition(numNodes, rank, size);
    const size_t N = part.N;
    const size_t row0 = static_cast<size_t>(part.firstBlock) * B;
    const size_t nLocalRows = static_cast<size_t>(part.nLocalBlocks) * B;
    const size_t localElems = nLocalRows * N;

    // Allocate matrices (locally owned rows only)
    std::vector<unsigned int> dist(localElems);
    std::vector<unsigned int> path(localElems);

    // Initialize
    if (rank == 0) printf("Initializing graph...\n");
    initializeDistanceMatrix(dist, numNodes, N, row0, nLocalRows, 1, MAX_DISTANCE);
    initializePathMatrix(path, N, row0, nLocalRows);

    // Pin the host matrices for fast DMA transfers
    bool hostPinned = false;
    if (localElems > 0) {
        hostPinned = cudaHostRegister(dist.data(), localElems * sizeof(unsigned int),
                                      cudaHostRegisterDefault) == cudaSuccess &&
                     cudaHostRegister(path.data(), localElems * sizeof(unsigned int),
                                      cudaHostRegisterDefault) == cudaSuccess;
        if (!hostPinned) {
            (void)cudaGetLastError();
            (void)cudaHostUnregister(dist.data());
            (void)cudaGetLastError();
        }
    }

    // Device buffers, streams and events
    Solver solver(part, rank, size);
    solver.init();
    if (localElems > 0) {
        CUDA_CHECK(cudaMemcpy(solver.dDist, dist.data(), localElems * sizeof(unsigned int),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(solver.dPath, path.data(), localElems * sizeof(unsigned int),
                              cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Run Floyd-Warshall
    if (rank == 0) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    solver.run();

    // Bring the locally owned result rows back to host memory
    if (localElems > 0) {
        CUDA_CHECK(cudaMemcpy(dist.data(), solver.dDist, localElems * sizeof(unsigned int),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(path.data(), solver.dPath, localElems * sizeof(unsigned int),
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

    solver.destroy();
    if (hostPinned) {
        CUDA_CHECK(cudaHostUnregister(dist.data()));
        CUDA_CHECK(cudaHostUnregister(path.data()));
    }

    int exitCode = 0;
    if (printResults || validate) {
        // Gather the unpadded numNodes x numNodes distance matrix on rank 0.
        // Rows are sent in pieces so that no message exceeds the int range.
        const size_t realRow0 = std::min(row0, numNodes);
        const size_t realRow1 = std::min(row0 + nLocalRows, numNodes);
        const size_t nRealRows = realRow1 - realRow0;
        const size_t maxElems = static_cast<size_t>(1) << 30;
        const size_t rowsPerMsg = std::max<size_t>(1, numNodes ? maxElems / numNodes : 1);

        std::vector<unsigned int> full;
        if (rank == 0) full.resize(numNodes * numNodes);
        unsigned int* dst = (rank == 0) ? full.data() : nullptr;
        if (rank == 0) {
#pragma omp parallel for schedule(static)
            for (size_t r = 0; r < nRealRows; ++r) {
                std::memcpy(dst + (realRow0 + r) * numNodes, &dist[r * N],
                            numNodes * sizeof(unsigned int));
            }
            for (int src = 1; src < size; ++src) {
                const int b0 = static_cast<int>((static_cast<long long>(part.nBlocks) * src) / size);
                const int b1 = static_cast<int>((static_cast<long long>(part.nBlocks) * (src + 1)) / size);
                const size_t rr0 = std::min(static_cast<size_t>(b0) * B, numNodes);
                const size_t rr1 = std::min(static_cast<size_t>(b1) * B, numNodes);
                for (size_t r = rr0; r < rr1; r += rowsPerMsg) {
                    const size_t rows = std::min(rowsPerMsg, rr1 - r);
                    MPI_Recv(dst + r * numNodes, static_cast<int>(rows * numNodes), MPI_UNSIGNED,
                             src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }
        } else {
            std::vector<unsigned int> packed(nRealRows * numNodes);
#pragma omp parallel for schedule(static)
            for (size_t r = 0; r < nRealRows; ++r) {
                std::memcpy(&packed[r * numNodes], &dist[r * N], numNodes * sizeof(unsigned int));
            }
            for (size_t r = 0; r < nRealRows; r += rowsPerMsg) {
                const size_t rows = std::min(rowsPerMsg, nRealRows - r);
                MPI_Send(packed.data() + r * numNodes, static_cast<int>(rows * numNodes),
                         MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD);
            }
        }

        if (rank == 0) {
            // Print results for external validation (integer hash-based)
            if (printResults) {
                print_results_int(full, "DistanceMatrix");
            }

            // Validation
            if (validate) {
                printf("Validating result...\n");
                bool valid = validateResult(full, numNodes);

                if (valid) {
                    printf("Validation: PASSED\n");
                } else {
                    printf("Validation: FAILED\n");
                    exitCode = 1;
                }
            }
        }
        MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return exitCode;
}
