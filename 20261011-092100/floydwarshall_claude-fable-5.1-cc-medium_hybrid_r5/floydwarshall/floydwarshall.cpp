// Floyd-Warshall All-Pairs Shortest Path Benchmark
//
// Hybrid MPI + OpenMP + CUDA implementation.
//
//  * MPI    : the distance matrix is partitioned by contiguous blocks of rows
//             (aligned to the GPU tile size) across ranks; each rank drives one
//             GPU. For every block of TILE pivot nodes the owning rank closes
//             the pivot row panel on its GPU and broadcasts it (non-blocking,
//             with one-block lookahead so communication overlaps compute).
//  * CUDA   : blocked Floyd-Warshall (Venkataraman et al.) with 64x64 tiles.
//             Phase 1 (diagonal tile) and phase 2 (pivot row panel) run on the
//             owner; phase 2 (pivot column panel) and phase 3 (min-plus
//             update of all remaining tiles) run on every rank for its rows.
//  * OpenMP : host-side matrix initialisation, result staging and validation.
//
// The algorithm produces exactly the same distance matrix as the classic
// triple loop: after processing pivot block [k0, k0+TILE) every entry holds the
// shortest path using only intermediate nodes < k0+TILE, independent of the
// evaluation order.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cctype>
#include <sched.h>
#include <sys/stat.h>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

// GPU tiling parameters
constexpr int TILE = 64;           // pivot block / tile edge length
constexpr int TDIM = 16;           // thread block is TDIM x TDIM threads
constexpr int SUB = TILE / TDIM;   // each thread owns SUB x SUB tile elements
constexpr int TPAD = TILE + 1;     // shared-memory row padding (bank conflicts)
constexpr unsigned int NO_UPDATE = 0xFFFFFFFFu;

#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        cudaError_t err__ = (call);                                                      \
        if (err__ != cudaSuccess) {                                                      \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__,   \
                    cudaGetErrorString(err__));                                          \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                \
        }                                                                                \
    } while (0)

#define MPI_CHECK(call)                                                                  \
    do {                                                                                 \
        int err__ = (call);                                                              \
        if (err__ != MPI_SUCCESS) {                                                      \
            fprintf(stderr, "MPI error %s at %s:%d (code %d)\n", #call, __FILE__,        \
                    __LINE__, err__);                                                    \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                \
        }                                                                                \
    } while (0)

// Index calculation for flattened 2D array (dist[i][j] lives at idx2(j, i, n) = i*n + j)
__host__ __device__ inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

// ---------------------------------------------------------------------------
// CUDA kernels
// ---------------------------------------------------------------------------

// Phase 1: close the diagonal tile [k0,k0+kc)^2. Single thread block.
// dist/path point at the rank-local row block; rowOff is the local row of k0.
__global__ void fwDiagKernel(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
                             const size_t n, const size_t rowOff, const size_t k0, const int kc) {
    __shared__ unsigned int D[TILE][TPAD];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * TDIM + tx;
    unsigned int* base = dist + rowOff * n + k0;
    unsigned int* pbase = path + rowOff * n + k0;

    for (int e = tid; e < TILE * TILE; e += TDIM * TDIM) {
        const int r = e / TILE, c = e % TILE;
        D[r][c] = (r < kc && c < kc) ? base[(size_t)r * n + c] : 0u;
    }
    __syncthreads();

    unsigned int pk[SUB][SUB];
#pragma unroll
    for (int a = 0; a < SUB; ++a)
#pragma unroll
        for (int b = 0; b < SUB; ++b) pk[a][b] = NO_UPDATE;

    for (int k = 0; k < kc; ++k) {
#pragma unroll
        for (int a = 0; a < SUB; ++a) {
            const int r = ty + a * TDIM;
            const unsigned int drk = D[r][k];
#pragma unroll
            for (int b = 0; b < SUB; ++b) {
                const int c = tx + b * TDIM;
                const unsigned int v = drk + D[k][c];
                if (v < D[r][c]) {
                    D[r][c] = v;
                    pk[a][b] = (unsigned int)(k0 + k);
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int a = 0; a < SUB; ++a) {
        const int r = ty + a * TDIM;
#pragma unroll
        for (int b = 0; b < SUB; ++b) {
            const int c = tx + b * TDIM;
            if (r < kc && c < kc) {
                base[(size_t)r * n + c] = D[r][c];
                if (pk[a][b] != NO_UPDATE) pbase[(size_t)r * n + c] = pk[a][b];
            }
        }
    }
}

// Phase 2 (row panel, owner only): update tile (kb, jb) for all jb != kb using the
// closed diagonal tile. Grid: one block per column tile.
__global__ void fwRowPanelKernel(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
                                 const size_t n, const size_t rowOff, const size_t k0, const int kc,
                                 const int kb) {
    const int jb = blockIdx.x;
    if (jb == kb) return;
    const size_t j0 = (size_t)jb * TILE;
    const int jc = (int)min((size_t)TILE, n - j0);

    __shared__ unsigned int D[TILE][TPAD];
    __shared__ unsigned int T[TILE][TPAD];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * TDIM + tx;
    const unsigned int* dbase = dist + rowOff * n + k0;
    unsigned int* tbase = dist + rowOff * n + j0;
    unsigned int* pbase = path + rowOff * n + j0;

    for (int e = tid; e < TILE * TILE; e += TDIM * TDIM) {
        const int r = e / TILE, c = e % TILE;
        D[r][c] = (r < kc && c < kc) ? dbase[(size_t)r * n + c] : 0u;
        T[r][c] = (r < kc && c < jc) ? tbase[(size_t)r * n + c] : 0u;
    }
    __syncthreads();

    unsigned int pk[SUB][SUB];
#pragma unroll
    for (int a = 0; a < SUB; ++a)
#pragma unroll
        for (int b = 0; b < SUB; ++b) pk[a][b] = NO_UPDATE;

    for (int k = 0; k < kc; ++k) {
#pragma unroll
        for (int a = 0; a < SUB; ++a) {
            const int r = ty + a * TDIM;
            const unsigned int drk = D[r][k];
#pragma unroll
            for (int b = 0; b < SUB; ++b) {
                const int c = tx + b * TDIM;
                const unsigned int v = drk + T[k][c];
                if (v < T[r][c]) {
                    T[r][c] = v;
                    pk[a][b] = (unsigned int)(k0 + k);
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int a = 0; a < SUB; ++a) {
        const int r = ty + a * TDIM;
#pragma unroll
        for (int b = 0; b < SUB; ++b) {
            const int c = tx + b * TDIM;
            if (r < kc && c < jc) {
                tbase[(size_t)r * n + c] = T[r][c];
                if (pk[a][b] != NO_UPDATE) pbase[(size_t)r * n + c] = pk[a][b];
            }
        }
    }
}

// Phase 2 (column panel, all ranks): update local tile (ib, kb) for ib != kb using
// the closed diagonal tile taken from the broadcast pivot panel (TILE x n, stride n).
// Grid: one block per local row tile starting at local tile ltBegin.
__global__ void fwColPanelKernel(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
                                 const size_t n, const unsigned int* __restrict__ panel,
                                 const int tile0, const int ltBegin, const size_t k0, const int kc,
                                 const int kb) {
    const int lt = ltBegin + blockIdx.x;
    const int ib = tile0 + lt;
    if (ib == kb) return;
    const size_t i0 = (size_t)ib * TILE;
    const int ic = (int)min((size_t)TILE, n - i0);

    __shared__ unsigned int D[TILE][TPAD];
    __shared__ unsigned int T[TILE][TPAD];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * TDIM + tx;
    const unsigned int* dbase = panel + k0;
    unsigned int* tbase = dist + (size_t)lt * TILE * n + k0;
    unsigned int* pbase = path + (size_t)lt * TILE * n + k0;

    for (int e = tid; e < TILE * TILE; e += TDIM * TDIM) {
        const int r = e / TILE, c = e % TILE;
        D[r][c] = (r < kc && c < kc) ? dbase[(size_t)r * n + c] : 0u;
        T[r][c] = (r < ic && c < kc) ? tbase[(size_t)r * n + c] : 0u;
    }
    __syncthreads();

    unsigned int pk[SUB][SUB];
#pragma unroll
    for (int a = 0; a < SUB; ++a)
#pragma unroll
        for (int b = 0; b < SUB; ++b) pk[a][b] = NO_UPDATE;

    for (int k = 0; k < kc; ++k) {
#pragma unroll
        for (int a = 0; a < SUB; ++a) {
            const int r = ty + a * TDIM;
            const unsigned int trk = T[r][k];
#pragma unroll
            for (int b = 0; b < SUB; ++b) {
                const int c = tx + b * TDIM;
                const unsigned int v = trk + D[k][c];
                if (v < T[r][c]) {
                    T[r][c] = v;
                    pk[a][b] = (unsigned int)(k0 + k);
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int a = 0; a < SUB; ++a) {
        const int r = ty + a * TDIM;
#pragma unroll
        for (int b = 0; b < SUB; ++b) {
            const int c = tx + b * TDIM;
            if (r < ic && c < kc) {
                tbase[(size_t)r * n + c] = T[r][c];
                if (pk[a][b] != NO_UPDATE) pbase[(size_t)r * n + c] = pk[a][b];
            }
        }
    }
}

// Phase 3 (all ranks): for every local tile (ib, jb) with ib != kb and jb != kb
//   dist[i][j] = min(dist[i][j], min_k dist[i][k] + panel[k][j])   (k in pivot block)
// Register-tiled min-plus product: each thread computes SUB x SUB outputs.
// Grid: (column tiles, local row tiles starting at ltBegin).
__global__ void fwUpdateKernel(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
                               const size_t n, const unsigned int* __restrict__ panel,
                               const int tile0, const int ltBegin, const size_t k0, const int kc,
                               const int kb) {
    const int jb = blockIdx.x;
    const int lt = ltBegin + blockIdx.y;
    const int ib = tile0 + lt;
    if (jb == kb || ib == kb) return;
    const size_t i0 = (size_t)ib * TILE;
    const size_t j0 = (size_t)jb * TILE;
    const int ic = (int)min((size_t)TILE, n - i0);
    const int jc = (int)min((size_t)TILE, n - j0);

    __shared__ unsigned int C[TILE][TPAD];  // local column panel tile (rows i, cols k)
    __shared__ unsigned int R[TILE][TPAD];  // pivot row panel tile  (rows k, cols j)
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * TDIM + tx;
    const unsigned int* cbase = dist + (size_t)lt * TILE * n + k0;
    const unsigned int* rbase = panel + j0;
    unsigned int* obase = dist + (size_t)lt * TILE * n + j0;
    unsigned int* pbase = path + (size_t)lt * TILE * n + j0;

    for (int e = tid; e < TILE * TILE; e += TDIM * TDIM) {
        const int r = e / TILE, c = e % TILE;
        C[r][c] = (r < ic && c < kc) ? cbase[(size_t)r * n + c] : 0u;
        R[r][c] = (r < kc && c < jc) ? rbase[(size_t)r * n + c] : 0u;
    }

    unsigned int best[SUB][SUB];
    unsigned int pk[SUB][SUB];
#pragma unroll
    for (int a = 0; a < SUB; ++a) {
        const int r = ty + a * TDIM;
#pragma unroll
        for (int b = 0; b < SUB; ++b) {
            const int c = tx + b * TDIM;
            best[a][b] = (r < ic && c < jc) ? obase[(size_t)r * n + c] : 0u;
            pk[a][b] = NO_UPDATE;
        }
    }
    __syncthreads();

    for (int k = 0; k < kc; ++k) {
        unsigned int rk[SUB];
#pragma unroll
        for (int b = 0; b < SUB; ++b) rk[b] = R[k][tx + b * TDIM];
#pragma unroll
        for (int a = 0; a < SUB; ++a) {
            const unsigned int cik = C[ty + a * TDIM][k];
#pragma unroll
            for (int b = 0; b < SUB; ++b) {
                const unsigned int v = cik + rk[b];
                if (v < best[a][b]) {
                    best[a][b] = v;
                    pk[a][b] = (unsigned int)(k0 + k);
                }
            }
        }
    }

#pragma unroll
    for (int a = 0; a < SUB; ++a) {
        const int r = ty + a * TDIM;
#pragma unroll
        for (int b = 0; b < SUB; ++b) {
            const int c = tx + b * TDIM;
            if (pk[a][b] != NO_UPDATE && r < ic && c < jc) {
                obase[(size_t)r * n + c] = best[a][b];
                pbase[(size_t)r * n + c] = pk[a][b];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------

// Each rank replays the identical sequential rand_r stream of the reference
// implementation and keeps the entries of its own row block, so the input graph
// is bit-identical to the original program regardless of the rank count.
void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes,
                              const size_t row0, const size_t row1,
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;

    for (size_t i = 0; i < numNodes; ++i) {
        const bool local = (i >= row0 && i < row1);
        unsigned int* row = local ? dist.data() + (i - row0) * numNodes : nullptr;
        for (size_t j = 0; j < numNodes; ++j) {
            const unsigned int v = rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
            if (local) row[j] = v;
        }
    }

    // Set diagonal to zero (distance from node to itself is 0)
#pragma omp parallel for schedule(static)
    for (size_t i = row0; i < row1; ++i) {
        dist[idx2(i, i - row0, numNodes)] = 0;
    }
}

// Reference semantics: path[i][j] = i for every (i, j).
void initializePathMatrix(std::vector<unsigned int>& path, const size_t numNodes,
                          const size_t row0, const size_t row1) {
#pragma omp parallel for schedule(static)
    for (size_t i = row0; i < row1; ++i) {
        unsigned int* row = path.data() + (i - row0) * numNodes;
        for (size_t j = 0; j < numNodes; ++j) row[j] = (unsigned int)i;
    }
}

// ---------------------------------------------------------------------------
// Topology helpers (NUMA-aware GPU selection)
// ---------------------------------------------------------------------------

static int cpuNumaNode() {
    const int cpu = sched_getcpu();
    if (cpu < 0) return -1;
    char p[128];
    struct stat st;
    for (int node = 0; node < 1024; ++node) {
        snprintf(p, sizeof(p), "/sys/devices/system/cpu/cpu%d/node%d", cpu, node);
        if (stat(p, &st) == 0) return node;
    }
    return -1;
}

static int gpuNumaNode(const int dev) {
    char bus[32];
    if (cudaDeviceGetPCIBusId(bus, sizeof(bus), dev) != cudaSuccess) return -1;
    for (char* c = bus; *c; ++c) *c = (char)tolower((unsigned char)*c);
    char p[160];
    snprintf(p, sizeof(p), "/sys/bus/pci/devices/%s/numa_node", bus);
    FILE* f = fopen(p, "r");
    if (!f) return -1;
    int node = -1;
    if (fscanf(f, "%d", &node) != 1) node = -1;
    fclose(f);
    return node;
}

// Deterministic device assignment for the ranks of one node: spread ranks over
// devices, preferring a GPU attached to the rank's own NUMA node.
static int selectDevice(MPI_Comm nodeComm, const int nodeRank, const int nodeSize, const int deviceCount) {
    const int myNuma = cpuNumaNode();
    std::vector<int> numaOfRank(nodeSize);
    MPI_CHECK(MPI_Allgather(&myNuma, 1, MPI_INT, numaOfRank.data(), 1, MPI_INT, nodeComm));
    std::vector<int> numaOfDev(deviceCount);
    for (int d = 0; d < deviceCount; ++d) numaOfDev[d] = gpuNumaNode(d);

    std::vector<int> load(deviceCount, 0);
    int chosen = nodeRank % deviceCount;
    for (int r = 0; r < nodeSize; ++r) {
        int best = -1;
        for (int d = 0; d < deviceCount; ++d) {
            if (best < 0) { best = d; continue; }
            const bool dMatch = (numaOfRank[r] >= 0 && numaOfDev[d] == numaOfRank[r]);
            const bool bMatch = (numaOfRank[r] >= 0 && numaOfDev[best] == numaOfRank[r]);
            if (load[d] < load[best] || (load[d] == load[best] && dMatch && !bMatch)) best = d;
        }
        load[best]++;
        if (r == nodeRank) chosen = best;
    }
    return chosen;
}

// ---------------------------------------------------------------------------
// Pivot panel exchange
// ---------------------------------------------------------------------------
//
// Ranks on the same node form a group sharing an MPI-3 shared-memory window:
// every rank owns NSLOT page-locked panel slots in it. The owner of a pivot
// block stages the closed panel (device -> its own slot) and
//   * signals node-local peers with an empty token; they DMA the panel from the
//     shared slot straight into their GPU and acknowledge when done,
//   * sends the panel with MPI to the leader of every other node, which stores
//     it in its own slot and tokens its local peers in turn.
// The ring of NSLOT slots keeps the acknowledgement off the critical path.

constexpr int NSLOT = 3;
constexpr int TAG_MASK = 0x3FFF;

struct Exchange {
    int rank = 0, size = 1;
    MPI_Comm data = MPI_COMM_NULL;    // tokens and panel messages
    MPI_Comm ack = MPI_COMM_NULL;     // acknowledgements
    MPI_Comm node = MPI_COMM_NULL;    // ranks sharing this node
    int nodeRank = 0, nodeSize = 1;
    std::vector<int> groupOf;         // world rank -> group id (leader's world rank)
    std::vector<int> nodeRankOf;      // world rank -> rank in node comm (or -1)
    int myGroup = 0;
    std::vector<int> localPeers;      // world ranks of my group (excluding me)
    std::vector<int> remoteLeaders;   // leaders of the other groups
    MPI_Win win = MPI_WIN_NULL;
    size_t slotElems = 0;             // elements per slot
    std::vector<unsigned int*> segOf; // node rank -> base of its NSLOT slots
    std::vector<bool> registered;     // segment page-locked via cudaHostRegister
    unsigned int* mySeg = nullptr;

    unsigned int* slot(int worldRank, int s) const {
        return segOf[nodeRankOf[worldRank]] + (size_t)s * slotElems;
    }
};

static void exchangeInit(Exchange& x, const size_t numNodes) {
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &x.rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &x.size));
    MPI_CHECK(MPI_Comm_dup(MPI_COMM_WORLD, &x.data));
    MPI_CHECK(MPI_Comm_dup(MPI_COMM_WORLD, &x.ack));
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, x.rank, MPI_INFO_NULL, &x.node));
    MPI_CHECK(MPI_Comm_rank(x.node, &x.nodeRank));
    MPI_CHECK(MPI_Comm_size(x.node, &x.nodeSize));

    // Group id = smallest world rank on the node (its leader).
    int leader = x.rank;
    MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &leader, 1, MPI_INT, MPI_MIN, x.node));
    x.myGroup = leader;
    x.groupOf.resize(x.size);
    MPI_CHECK(MPI_Allgather(&x.myGroup, 1, MPI_INT, x.groupOf.data(), 1, MPI_INT, MPI_COMM_WORLD));
    x.nodeRankOf.assign(x.size, -1);
    {
        std::vector<int> worldOfNode(x.nodeSize);
        MPI_CHECK(MPI_Allgather(&x.rank, 1, MPI_INT, worldOfNode.data(), 1, MPI_INT, x.node));
        for (int i = 0; i < x.nodeSize; ++i) x.nodeRankOf[worldOfNode[i]] = i;
    }
    for (int r = 0; r < x.size; ++r) {
        if (r == x.rank) continue;
        if (x.groupOf[r] == x.myGroup) x.localPeers.push_back(r);
        else if (x.groupOf[r] == r) x.remoteLeaders.push_back(r);
    }

    // Shared-memory window: NSLOT panel slots per rank.
    x.slotElems = std::max<size_t>((size_t)TILE * numNodes, 1);
    const MPI_Aint bytes = (MPI_Aint)(x.slotElems * NSLOT * sizeof(unsigned int));
    MPI_Info info;
    MPI_CHECK(MPI_Info_create(&info));
    MPI_CHECK(MPI_Info_set(info, "alloc_shared_noncontig", "true"));
    void* base = nullptr;
    MPI_CHECK(MPI_Win_allocate_shared(bytes, sizeof(unsigned int), info, x.node, &base, &x.win));
    MPI_CHECK(MPI_Info_free(&info));
    x.mySeg = static_cast<unsigned int*>(base);
    x.segOf.resize(x.nodeSize);
    x.registered.assign(x.nodeSize, false);
    for (int i = 0; i < x.nodeSize; ++i) {
        MPI_Aint sz;
        int disp;
        void* p = nullptr;
        MPI_CHECK(MPI_Win_shared_query(x.win, i, &sz, &disp, &p));
        x.segOf[i] = static_cast<unsigned int*>(p);
        // Page-lock the mapping so panel DMA to/from the GPU is asynchronous.
        cudaError_t e = cudaHostRegister(p, (size_t)bytes, cudaHostRegisterPortable);
        if (e == cudaSuccess) {
            x.registered[i] = true;
        } else {
            (void)cudaGetLastError();  // clear; fall back to pageable copies
        }
    }
    MPI_CHECK(MPI_Barrier(x.node));
}

static void exchangeFinalize(Exchange& x) {
    for (int i = 0; i < x.nodeSize; ++i)
        if (x.registered[i]) CUDA_CHECK(cudaHostUnregister(x.segOf[i]));
    MPI_CHECK(MPI_Win_free(&x.win));
    MPI_CHECK(MPI_Comm_free(&x.node));
    MPI_CHECK(MPI_Comm_free(&x.ack));
    MPI_CHECK(MPI_Comm_free(&x.data));
}

struct GpuContext {
    cudaStream_t compute = nullptr;   // bulk phase 2/3 work (low priority)
    cudaStream_t urgent = nullptr;    // pivot-panel critical path (high priority)
    cudaStream_t transfer = nullptr;  // H2D of received pivot panels
    cudaEvent_t panelFree[NSLOT] = {};    // compute stream done reading d_panel[s]
    cudaEvent_t panelFreeU[NSLOT] = {};   // urgent stream done reading d_panel[s]
    cudaEvent_t panelCopied[NSLOT] = {};  // H2D into d_panel[s] complete
    cudaEvent_t nextTile = nullptr;   // tile kb+2 updated with block kb (compute stream)
    cudaEvent_t d2hDone = nullptr;
    unsigned int* d_panel[NSLOT] = {};    // device pivot panels (TILE x n)
};

// Distributed blocked Floyd-Warshall. This rank owns global tiles [t0, t1).
//
// Pipeline per pivot block kb (one-block lookahead):
//   urgent stream : tile kb+1 <- block kb, close tile kb+1 (phase 1 + row panel),
//                   stage to shared memory, token/send   (owner of kb+1 only)
//   compute stream: tile kb+2 <- block kb (so the next lookahead never waits for
//                   the bulk), then all remaining local tiles <- block kb
// Receivers DMA the next panel on the transfer stream while computing.
void floydWarshall(unsigned int* d_dist, unsigned int* d_path, const size_t numNodes,
                   const int numTiles, const int t0, const int t1, const std::vector<int>& owner,
                   Exchange& x, GpuContext& g) {
    const size_t n = numNodes;
    const int rank = x.rank;
    const size_t row0 = (size_t)t0 * TILE;
    const int ltiles = t1 - t0;
    const dim3 block(TDIM, TDIM);

    // Per-slot bookkeeping
    MPI_Request recvReq[NSLOT];                      // incoming token / panel
    std::vector<MPI_Request> sendReqs[NSLOT];        // my tokens / panel sends (as source)
    std::vector<MPI_Request> ackRecvReqs[NSLOT];     // acks expected from local peers (as source)
    MPI_Request ackSendReq[NSLOT];                   // my ack to the source
    int ackTarget[NSLOT];                            // source to acknowledge (-1: none)
    int ackBlock[NSLOT];                             // block the pending ack refers to
    for (int s = 0; s < NSLOT; ++s) {
        recvReq[s] = MPI_REQUEST_NULL;
        ackSendReq[s] = MPI_REQUEST_NULL;
        ackTarget[s] = -1;
        ackBlock[s] = -1;
    }

    auto kcount = [&](int kb) { return (int)std::min<size_t>(TILE, n - (size_t)kb * TILE); };
    auto isMine = [&](int t) { return t < numTiles && owner[t] == rank; };
    auto tag = [](int kb) { return kb & TAG_MASK; };
    // Who provides the panel of block kb to me (rank itself if I am the owner).
    auto sourceOf = [&](int kb) {
        const int o = owner[kb];
        if (o == rank) return rank;
        if (x.groupOf[o] == x.myGroup) return o;
        if (rank == x.myGroup) return o;  // I am my node's leader: receive by MPI
        return x.myGroup;                 // otherwise: from my leader's slot
    };
    auto iAmLeaderReceiver = [&](int kb) {
        return owner[kb] != rank && x.groupOf[owner[kb]] != x.myGroup && rank == x.myGroup;
    };

    // Make slot s of my segment reusable: previous consumers must be done.
    auto releaseSlot = [&](int s) {
        if (!ackRecvReqs[s].empty()) {
            MPI_CHECK(MPI_Waitall((int)ackRecvReqs[s].size(), ackRecvReqs[s].data(), MPI_STATUSES_IGNORE));
            ackRecvReqs[s].clear();
        }
        if (!sendReqs[s].empty()) {
            MPI_CHECK(MPI_Waitall((int)sendReqs[s].size(), sendReqs[s].data(), MPI_STATUSES_IGNORE));
            sendReqs[s].clear();
        }
    };

    // Source role: token the local peers and expect their acks.
    auto tokenPeers = [&](int kb) {
        const int s = kb % NSLOT;
        for (int p : x.localPeers) {
            sendReqs[s].emplace_back();
            MPI_CHECK(MPI_Isend(nullptr, 0, MPI_BYTE, p, tag(kb), x.data, &sendReqs[s].back()));
            ackRecvReqs[s].emplace_back();
            MPI_CHECK(MPI_Irecv(nullptr, 0, MPI_BYTE, p, tag(kb), x.ack, &ackRecvReqs[s].back()));
        }
    };

    // Owner: phase 1 + phase 2 (row panel) on the local rows of tile kb.
    auto phaseA = [&](int kb, cudaStream_t st) {
        const size_t k0 = (size_t)kb * TILE;
        const int kc = kcount(kb);
        const size_t rowOff = k0 - row0;
        fwDiagKernel<<<1, block, 0, st>>>(d_dist, d_path, n, rowOff, k0, kc);
        if (numTiles > 1)
            fwRowPanelKernel<<<numTiles, block, 0, st>>>(d_dist, d_path, n, rowOff, k0, kc, kb);
    };

    // All ranks: phase 2 (column panel) + phase 3 on local tiles [ltBegin, ltEnd).
    auto phaseBC = [&](int kb, const unsigned int* panel, int ltBegin, int ltEnd, cudaStream_t st) {
        if (ltEnd <= ltBegin) return;
        const size_t k0 = (size_t)kb * TILE;
        const int kc = kcount(kb);
        const int cnt = ltEnd - ltBegin;
        fwColPanelKernel<<<cnt, block, 0, st>>>(d_dist, d_path, n, panel, t0, ltBegin, k0, kc, kb);
        if (numTiles > 1)
            fwUpdateKernel<<<dim3(numTiles, cnt), block, 0, st>>>(d_dist, d_path, n, panel, t0, ltBegin,
                                                                  k0, kc, kb);
    };

    // Owner: stage the closed pivot panel into my shared slot (urgent stream).
    auto stagePanel = [&](int kb) {
        const int s = kb % NSLOT;
        const size_t k0 = (size_t)kb * TILE;
        const int kc = kcount(kb);
        releaseSlot(s);
        CUDA_CHECK(cudaMemcpyAsync(x.slot(rank, s), d_dist + (k0 - row0) * n, (size_t)kc * n * sizeof(unsigned int),
                                   cudaMemcpyDeviceToHost, g.urgent));
        CUDA_CHECK(cudaEventRecord(g.d2hDone, g.urgent));
    };

    // Owner: once staged, hand the panel to local peers (token) and remote leaders (MPI).
    auto publish = [&](int kb) {
        const int s = kb % NSLOT;
        const int kc = kcount(kb);
        CUDA_CHECK(cudaEventSynchronize(g.d2hDone));
        tokenPeers(kb);
        for (int l : x.remoteLeaders) {
            sendReqs[s].emplace_back();
            MPI_CHECK(MPI_Isend(x.slot(rank, s), kc * (int)n, MPI_UNSIGNED, l, tag(kb), x.data, &sendReqs[s].back()));
        }
    };

    // Non-owner: post the receive for block kb (token, or the panel if I lead a remote node).
    auto subscribe = [&](int kb) {
        const int s = kb % NSLOT;
        const int kc = kcount(kb);
        if (iAmLeaderReceiver(kb)) {
            releaseSlot(s);
            MPI_CHECK(MPI_Irecv(x.slot(rank, s), kc * (int)n, MPI_UNSIGNED, owner[kb], tag(kb), x.data, &recvReq[s]));
        } else {
            MPI_CHECK(MPI_Irecv(nullptr, 0, MPI_BYTE, sourceOf(kb), tag(kb), x.data, &recvReq[s]));
        }
    };

    // Acknowledge panel copies that have landed (slot q only, or all; force waits).
    auto flushAcks = [&](int onlySlot, bool force) {
        for (int q = 0; q < NSLOT; ++q) {
            if (ackTarget[q] < 0 || (onlySlot >= 0 && q != onlySlot)) continue;
            if (force) {
                CUDA_CHECK(cudaEventSynchronize(g.panelCopied[q]));
            } else if (cudaEventQuery(g.panelCopied[q]) != cudaSuccess) {
                continue;
            }
            MPI_CHECK(MPI_Wait(&ackSendReq[q], MPI_STATUS_IGNORE));
            MPI_CHECK(MPI_Isend(nullptr, 0, MPI_BYTE, ackTarget[q], tag(ackBlock[q]), x.ack, &ackSendReq[q]));
            ackTarget[q] = -1;
        }
    };

    // Wait for block kb's panel to arrive; meanwhile acknowledge finished copies.
    auto waitIncoming = [&](int kb) {
        const int s = kb % NSLOT;
        int done = 0;
        for (;;) {
            flushAcks(-1, false);
            MPI_CHECK(MPI_Test(&recvReq[s], &done, MPI_STATUS_IGNORE));
            if (done) break;
        }
        if (iAmLeaderReceiver(kb)) tokenPeers(kb);
    };

    if (numTiles > 0) {
        if (isMine(0)) {
            phaseA(0, g.urgent);
            stagePanel(0);
            publish(0);
        } else {
            subscribe(0);
        }
    }

    for (int kb = 0; kb < numTiles; ++kb) {
        const int s = kb % NSLOT;
        const size_t k0 = (size_t)kb * TILE;
        const int kc = kcount(kb);
        const bool mine = isMine(kb);

        const unsigned int* panel;
        if (mine) {
            panel = d_dist + (k0 - row0) * n;  // closed panel already lives in local rows
        } else {
            waitIncoming(kb);
            // DMA the panel from the shared slot of its source on the transfer stream,
            // overlapping with compute which may still be busy with the previous block.
            const int src = sourceOf(kb);
            CUDA_CHECK(cudaStreamWaitEvent(g.transfer, g.panelFree[s], 0));
            CUDA_CHECK(cudaStreamWaitEvent(g.transfer, g.panelFreeU[s], 0));
            CUDA_CHECK(cudaMemcpyAsync(g.d_panel[s], x.slot(src, s), (size_t)kc * n * sizeof(unsigned int),
                                       cudaMemcpyHostToDevice, g.transfer));
            CUDA_CHECK(cudaEventRecord(g.panelCopied[s], g.transfer));
            CUDA_CHECK(cudaStreamWaitEvent(g.compute, g.panelCopied[s], 0));
            CUDA_CHECK(cudaStreamWaitEvent(g.urgent, g.panelCopied[s], 0));
            panel = g.d_panel[s];
            if (src != rank) {  // acknowledge once the copy has landed
                flushAcks(s, true);
                ackTarget[s] = src;
                ackBlock[s] = kb;
            }
        }

        const int next = kb + 1, next2 = kb + 2;
        const bool nextMine = isMine(next), next2Mine = isMine(next2);
        int exLo = ltiles, exHi = ltiles;  // local tiles handled outside the bulk range

        if (!nextMine && next < numTiles) subscribe(next);

        if (nextMine) {
            // Critical path: close the next pivot tile as early as possible.
            const int lt = next - t0;
            CUDA_CHECK(cudaStreamWaitEvent(g.urgent, g.nextTile, 0));  // tile kb+1 <- block kb-1 done
            phaseBC(kb, panel, lt, lt + 1, g.urgent);
            phaseA(next, g.urgent);
            stagePanel(next);
            CUDA_CHECK(cudaEventRecord(g.panelFreeU[s], g.urgent));
            exLo = lt;
            exHi = lt + 1;
        }
        if (next2Mine) {
            // Tile kb+2 first so the next iteration's lookahead does not wait on the bulk.
            const int lt = next2 - t0;
            phaseBC(kb, panel, lt, lt + 1, g.compute);
            CUDA_CHECK(cudaEventRecord(g.nextTile, g.compute));
            if (!nextMine) exLo = lt;
            exHi = lt + 1;
        }
        phaseBC(kb, panel, 0, exLo, g.compute);
        phaseBC(kb, panel, exHi, ltiles, g.compute);
        CUDA_CHECK(cudaEventRecord(g.panelFree[s], g.compute));

        if (nextMine) publish(next);
    }

    // Drain: send outstanding acks, then wait for everything we sent.
    flushAcks(-1, true);
    for (int q = 0; q < NSLOT; ++q) {
        releaseSlot(q);
        MPI_CHECK(MPI_Wait(&ackSendReq[q], MPI_STATUS_IGNORE));
    }

    CUDA_CHECK(cudaStreamSynchronize(g.urgent));
    CUDA_CHECK(cudaStreamSynchronize(g.compute));
    CUDA_CHECK(cudaStreamSynchronize(g.transfer));
    CUDA_CHECK(cudaGetLastError());
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
    const size_t lim = std::min(numNodes, static_cast<size_t>(10));
    bool failed = false;
#pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = 0; i < lim; ++i) {
        for (size_t j = 0; j < lim; ++j) {
            bool stop;
#pragma omp atomic read
            stop = failed;
            if (stop) continue;
            for (size_t k = 0; k < numNodes; ++k) {
                const unsigned int distIJ = dist[idx2(j, i, numNodes)];
                const unsigned int distIK = dist[idx2(k, i, numNodes)];
                const unsigned int distKJ = dist[idx2(j, k, numNodes)];

                // Check for overflow before addition
                if (distIK < INF && distKJ < INF) {
                    if (distIK + distKJ < distIJ) {
#pragma omp critical
                        {
                            if (!failed) {
                                printf("Validation failed: triangle inequality violated at [%zu,%zu,%zu]\n",
                                       i, j, k);
                                failed = true;
                            }
                        }
                        break;
                    }
                }
            }
        }
    }

    return !failed;
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
    MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));
    int rank = 0, size = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &size));

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

    // Select the GPU: one device per rank, NUMA-aware among node-local ranks.
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA device found\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    {
        MPI_Comm nodeComm;
        int nodeRank, nodeSize;
        MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm));
        MPI_CHECK(MPI_Comm_rank(nodeComm, &nodeRank));
        MPI_CHECK(MPI_Comm_size(nodeComm, &nodeSize));
        const int device = selectDevice(nodeComm, nodeRank, nodeSize, deviceCount);
        MPI_CHECK(MPI_Comm_free(&nodeComm));
        CUDA_CHECK(cudaSetDevice(device));
        CUDA_CHECK(cudaFree(nullptr));  // establish the context now, outside the timed region
    }

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Parallel config: %d MPI rank(s), %d OpenMP thread(s)/rank, CUDA tile %d\n", size,
               omp_get_max_threads(), TILE);
    }

    // Row-block decomposition aligned to the GPU tile size.
    const int numTiles = (int)((numNodes + TILE - 1) / TILE);
    std::vector<int> tileBegin(size + 1);
    for (int r = 0; r <= size; ++r) tileBegin[r] = (int)(((long long)numTiles * r) / size);
    std::vector<int> owner(numTiles);
    for (int r = 0; r < size; ++r)
        for (int t = tileBegin[r]; t < tileBegin[r + 1]; ++t) owner[t] = r;
    const int t0 = tileBegin[rank], t1 = tileBegin[rank + 1];
    const size_t row0 = std::min((size_t)t0 * TILE, numNodes);
    const size_t row1 = std::min((size_t)t1 * TILE, numNodes);
    const size_t localRows = row1 - row0;
    const size_t localElems = localRows * numNodes;

    // Allocate matrices (this rank's row block)
    std::vector<unsigned int> dist(localElems);
    std::vector<unsigned int> path(localElems);

    // Initialize
    if (rank == 0) printf("Initializing graph...\n");
    initializeDistanceMatrix(dist, numNodes, row0, row1, 1, MAX_DISTANCE);
    initializePathMatrix(path, numNodes, row0, row1);

    // Device buffers
    const size_t localBytes = std::max<size_t>(localElems, 1) * sizeof(unsigned int);
    const size_t panelBytes = std::max<size_t>((size_t)TILE * numNodes, 1) * sizeof(unsigned int);
    unsigned int *d_dist = nullptr, *d_path = nullptr;
    CUDA_CHECK(cudaMalloc(&d_dist, localBytes));
    CUDA_CHECK(cudaMalloc(&d_path, localBytes));
    if (localElems > 0) {
        CUDA_CHECK(cudaMemcpy(d_dist, dist.data(), localElems * sizeof(unsigned int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_path, path.data(), localElems * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }

    Exchange x;
    exchangeInit(x, numNodes);

    GpuContext g;
    int prioLeast = 0, prioGreatest = 0;
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&prioLeast, &prioGreatest));
    CUDA_CHECK(cudaStreamCreateWithPriority(&g.compute, cudaStreamNonBlocking, prioLeast));
    CUDA_CHECK(cudaStreamCreateWithPriority(&g.urgent, cudaStreamNonBlocking, prioGreatest));
    CUDA_CHECK(cudaStreamCreateWithFlags(&g.transfer, cudaStreamNonBlocking));
    for (int s = 0; s < NSLOT; ++s) {
        CUDA_CHECK(cudaMalloc(&g.d_panel[s], panelBytes));
        CUDA_CHECK(cudaEventCreateWithFlags(&g.panelFree[s], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&g.panelFreeU[s], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&g.panelCopied[s], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventRecord(g.panelFree[s], g.compute));
        CUDA_CHECK(cudaEventRecord(g.panelFreeU[s], g.urgent));
        CUDA_CHECK(cudaEventRecord(g.panelCopied[s], g.transfer));
    }
    CUDA_CHECK(cudaEventCreateWithFlags(&g.nextTile, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventRecord(g.nextTile, g.compute));
    CUDA_CHECK(cudaEventCreateWithFlags(&g.d2hDone, cudaEventDisableTiming));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Run Floyd-Warshall
    if (rank == 0) printf("Computing shortest paths...\n");
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto start = std::chrono::high_resolution_clock::now();

    floydWarshall(d_dist, d_path, numNodes, numTiles, t0, t1, owner, x, g);

    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", (long)duration.count());

        // Calculate operations per second
        // Floyd-Warshall has O(n³) complexity
        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);
    }

    int valid = 1;
    if (printResults || validate) {
        // Bring the row blocks back to the host and gather the full matrix on rank 0.
        if (localElems > 0)
            CUDA_CHECK(cudaMemcpy(dist.data(), d_dist, localElems * sizeof(unsigned int), cudaMemcpyDeviceToHost));

        MPI_Datatype rowType;
        MPI_CHECK(MPI_Type_contiguous((int)numNodes, MPI_UNSIGNED, &rowType));
        MPI_CHECK(MPI_Type_commit(&rowType));
        std::vector<int> counts(size), displs(size);
        for (int r = 0; r < size; ++r) {
            const size_t rb = std::min((size_t)tileBegin[r] * TILE, numNodes);
            const size_t re = std::min((size_t)tileBegin[r + 1] * TILE, numNodes);
            counts[r] = (int)(re - rb);
            displs[r] = (int)rb;
        }
        std::vector<unsigned int> fullDist;
        if (rank == 0) fullDist.resize(numNodes * numNodes);
        MPI_CHECK(MPI_Gatherv(dist.data(), (int)localRows, rowType, rank == 0 ? fullDist.data() : nullptr,
                              counts.data(), displs.data(), rowType, 0, MPI_COMM_WORLD));
        MPI_CHECK(MPI_Type_free(&rowType));

        if (rank == 0) {
            // Print results for external validation (integer hash-based)
            if (printResults) {
                print_results_int(fullDist, "DistanceMatrix");
            }

            // Validation
            if (validate) {
                printf("Validating result...\n");
                valid = validateResult(fullDist, numNodes) ? 1 : 0;
                printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            }
        }
        MPI_CHECK(MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD));
    }

    // Cleanup
    for (int s = 0; s < NSLOT; ++s) {
        CUDA_CHECK(cudaEventDestroy(g.panelFree[s]));
        CUDA_CHECK(cudaEventDestroy(g.panelFreeU[s]));
        CUDA_CHECK(cudaEventDestroy(g.panelCopied[s]));
        CUDA_CHECK(cudaFree(g.d_panel[s]));
    }
    CUDA_CHECK(cudaEventDestroy(g.d2hDone));
    CUDA_CHECK(cudaEventDestroy(g.nextTile));
    CUDA_CHECK(cudaStreamDestroy(g.compute));
    CUDA_CHECK(cudaStreamDestroy(g.urgent));
    CUDA_CHECK(cudaStreamDestroy(g.transfer));
    CUDA_CHECK(cudaFree(d_dist));
    CUDA_CHECK(cudaFree(d_path));
    exchangeFinalize(x);

    MPI_Finalize();
    return valid ? 0 : 1;
}
