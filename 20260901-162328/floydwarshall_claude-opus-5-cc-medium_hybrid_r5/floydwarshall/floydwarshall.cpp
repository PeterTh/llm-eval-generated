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

// ---------------------------------------------------------------------------
// Hybrid MPI + OpenMP + CUDA blocked Floyd-Warshall
//
//  * MPI   : the distance matrix is distributed by contiguous block-rows of
//            TILE rows; one rank per GPU.  Every k-step only requires the
//            pivot block-row panel (TILE x N) to be distributed, i.e. O(n^2)
//            total communication for an O(n^3) computation.  Inside a node the
//            panel travels through a shared-memory window (one download by the
//            producer, one upload per consumer), between nodes it is broadcast
//            among the node leaders.  The exchange overlaps with the bulk of
//            the local phase 3 work, which is queued on the GPU beforehand.
//  * CUDA  : the classical three-phase blocked Floyd-Warshall, TILE = 64,
//            256 threads per block, 16 matrix elements per thread, all inner
//            k-steps served out of shared memory / registers.  Panel uploads
//            run on a second stream concurrently with the compute kernels.
//  * OpenMP: host side work (graph generation with a jump-ahead RNG, packing
//            of the distributed result, validation) is threaded.
// ---------------------------------------------------------------------------

#define CUDA_CHECK(expr)                                                                                               \
    do {                                                                                                               \
        const cudaError_t err__ = (expr);                                                                              \
        if (err__ != cudaSuccess) {                                                                                    \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err__), __FILE__, __LINE__);                 \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                                              \
        }                                                                                                              \
    } while (0)

// Tile (block) size of the blocked algorithm and the thread geometry used by
// all kernels: 32x8 threads, each thread owns 8 rows x 2 columns of a tile.
constexpr int TILE = 64;
constexpr int TX = 32;
constexpr int TY = 8;
constexpr int RPT = TILE / TY; // rows per thread

// Index calculation for flattened 2D array
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

// ---------------------------------------------------------------------------
// Device kernels
// ---------------------------------------------------------------------------

// Loads / stores the RPT x 2 elements of a tile owned by this thread.
#define FW_LOAD_REGS(tile, ptile, v, p, ld)                                                                            \
    _Pragma("unroll") for (int r = 0; r < RPT; ++r) {                                                                   \
        const size_t o__ = (size_t)(ty + r * TY) * (ld);                                                                \
        v[r][0] = (tile)[o__ + tx];                                                                                     \
        v[r][1] = (tile)[o__ + tx + TX];                                                                                \
        p[r][0] = (ptile)[o__ + tx];                                                                                    \
        p[r][1] = (ptile)[o__ + tx + TX];                                                                               \
    }

#define FW_STORE_REGS(tile, ptile, v, p, ld)                                                                           \
    _Pragma("unroll") for (int r = 0; r < RPT; ++r) {                                                                   \
        const size_t o__ = (size_t)(ty + r * TY) * (ld);                                                                \
        (tile)[o__ + tx] = v[r][0];                                                                                     \
        (tile)[o__ + tx + TX] = v[r][1];                                                                                \
        (ptile)[o__ + tx] = p[r][0];                                                                                    \
        (ptile)[o__ + tx + TX] = p[r][1];                                                                               \
    }

// Phase 1: the pivot tile (kb,kb) depends on itself, so the k-loop has to be
// carried out in shared memory with the updated values visible immediately.
__global__ __launch_bounds__(TX* TY) void fwPhase1(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
                                                   const int ld, const int kb) {
    __shared__ unsigned int sd[TILE][TILE];
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    unsigned int v[RPT][2];
    unsigned int p[RPT][2];
    FW_LOAD_REGS(dist, path, v, p, ld);
#pragma unroll
    for (int r = 0; r < RPT; ++r) {
        sd[ty + r * TY][tx] = v[r][0];
        sd[ty + r * TY][tx + TX] = v[r][1];
    }
    __syncthreads();

    for (int kk = 0; kk < TILE; ++kk) {
        const unsigned int kg = (unsigned int)(kb * TILE + kk);
        const unsigned int b0 = sd[kk][tx];
        const unsigned int b1 = sd[kk][tx + TX];
#pragma unroll
        for (int r = 0; r < RPT; ++r) {
            const unsigned int a = sd[ty + r * TY][kk];
            const unsigned int c0 = a + b0;
            const unsigned int c1 = a + b1;
            if (c0 < v[r][0]) {
                v[r][0] = c0;
                p[r][0] = kg;
            }
            if (c1 < v[r][1]) {
                v[r][1] = c1;
                p[r][1] = kg;
            }
        }
        __syncthreads();
#pragma unroll
        for (int r = 0; r < RPT; ++r) {
            sd[ty + r * TY][tx] = v[r][0];
            sd[ty + r * TY][tx + TX] = v[r][1];
        }
        __syncthreads();
    }

    FW_STORE_REGS(dist, path, v, p, ld);
}

// Phase 2a: the tiles of the pivot block-row, (kb,bj).  dist/path point at the
// start of the pivot panel, the pivot tile lives inside that panel.
__global__ __launch_bounds__(TX* TY) void fwPhase2Row(unsigned int* __restrict__ panel,
                                                      unsigned int* __restrict__ panelPath, const int ld,
                                                      const int kb) {
    const int bj = blockIdx.x;
    if (bj == kb) return;

    __shared__ unsigned int sT[TILE][TILE];
    __shared__ unsigned int sP[TILE][TILE];
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    unsigned int* tile = panel + (size_t)bj * TILE;
    unsigned int* ptile = panelPath + (size_t)bj * TILE;
    const unsigned int* pivot = panel + (size_t)kb * TILE;

    unsigned int v[RPT][2];
    unsigned int p[RPT][2];
    FW_LOAD_REGS(tile, ptile, v, p, ld);
#pragma unroll
    for (int r = 0; r < RPT; ++r) {
        const int i = ty + r * TY;
        const size_t o = (size_t)i * ld;
        sT[i][tx] = v[r][0];
        sT[i][tx + TX] = v[r][1];
        sP[i][tx] = pivot[o + tx];
        sP[i][tx + TX] = pivot[o + tx + TX];
    }
    __syncthreads();

    for (int kk = 0; kk < TILE; ++kk) {
        const unsigned int kg = (unsigned int)(kb * TILE + kk);
        const unsigned int b0 = sT[kk][tx];
        const unsigned int b1 = sT[kk][tx + TX];
#pragma unroll
        for (int r = 0; r < RPT; ++r) {
            const unsigned int a = sP[ty + r * TY][kk];
            const unsigned int c0 = a + b0;
            const unsigned int c1 = a + b1;
            if (c0 < v[r][0]) {
                v[r][0] = c0;
                p[r][0] = kg;
            }
            if (c1 < v[r][1]) {
                v[r][1] = c1;
                p[r][1] = kg;
            }
        }
        __syncthreads();
#pragma unroll
        for (int r = 0; r < RPT; ++r) {
            sT[ty + r * TY][tx] = v[r][0];
            sT[ty + r * TY][tx + TX] = v[r][1];
        }
        __syncthreads();
    }

    FW_STORE_REGS(tile, ptile, v, p, ld);
}

// Phase 2b: the tiles of the pivot block-column owned by this rank, (gi,kb).
__global__ __launch_bounds__(TX* TY) void fwPhase2Col(unsigned int* __restrict__ dist,
                                                      unsigned int* __restrict__ path,
                                                      const unsigned int* __restrict__ rowk, const int ld, const int kb,
                                                      const int blockRowBase) {
    const int gi = blockRowBase + blockIdx.x;
    if (gi == kb) return;

    __shared__ unsigned int sT[TILE][TILE];
    __shared__ unsigned int sP[TILE][TILE];
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const size_t tileOff = (size_t)blockIdx.x * TILE * ld + (size_t)kb * TILE;
    unsigned int* tile = dist + tileOff;
    unsigned int* ptile = path + tileOff;
    const unsigned int* pivot = rowk + (size_t)kb * TILE;

    unsigned int v[RPT][2];
    unsigned int p[RPT][2];
    FW_LOAD_REGS(tile, ptile, v, p, ld);
#pragma unroll
    for (int r = 0; r < RPT; ++r) {
        const int i = ty + r * TY;
        const size_t o = (size_t)i * ld;
        sT[i][tx] = v[r][0];
        sT[i][tx + TX] = v[r][1];
        sP[i][tx] = pivot[o + tx];
        sP[i][tx + TX] = pivot[o + tx + TX];
    }
    __syncthreads();

    for (int kk = 0; kk < TILE; ++kk) {
        const unsigned int kg = (unsigned int)(kb * TILE + kk);
        const unsigned int b0 = sP[kk][tx];
        const unsigned int b1 = sP[kk][tx + TX];
#pragma unroll
        for (int r = 0; r < RPT; ++r) {
            const unsigned int a = sT[ty + r * TY][kk];
            const unsigned int c0 = a + b0;
            const unsigned int c1 = a + b1;
            if (c0 < v[r][0]) {
                v[r][0] = c0;
                p[r][0] = kg;
            }
            if (c1 < v[r][1]) {
                v[r][1] = c1;
                p[r][1] = kg;
            }
        }
        __syncthreads();
#pragma unroll
        for (int r = 0; r < RPT; ++r) {
            sT[ty + r * TY][tx] = v[r][0];
            sT[ty + r * TY][tx + TX] = v[r][1];
        }
        __syncthreads();
    }

    FW_STORE_REGS(tile, ptile, v, p, ld);
}

// Phase 3: all remaining tiles (gi,bj) are independent; the whole k-loop runs
// out of shared memory without any synchronisation.
// `onlyRow >= 0` restricts the launch to a single global block-row, `skipRow`
// excludes one (used to split the work around the panel exchange).
__global__ __launch_bounds__(TX* TY) void fwPhase3(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
                                                   const unsigned int* __restrict__ rowk, const int ld, const int kb,
                                                   const int blockRowBase, const int onlyRow, const int skipRow) {
    const int bj = blockIdx.x;
    const int gi = blockRowBase + blockIdx.y;
    if (bj == kb || gi == kb || gi == skipRow) return;
    if (onlyRow >= 0 && gi != onlyRow) return;

    __shared__ unsigned int sA[TILE][TILE]; // tile (gi,kb)
    __shared__ unsigned int sB[TILE][TILE]; // tile (kb,bj)
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const size_t rowOff = (size_t)blockIdx.y * TILE * ld;
    unsigned int* tile = dist + rowOff + (size_t)bj * TILE;
    unsigned int* ptile = path + rowOff + (size_t)bj * TILE;
    const unsigned int* aTile = dist + rowOff + (size_t)kb * TILE;
    const unsigned int* bTile = rowk + (size_t)bj * TILE;

    unsigned int v[RPT][2];
    unsigned int p[RPT][2];
    FW_LOAD_REGS(tile, ptile, v, p, ld);
#pragma unroll
    for (int r = 0; r < RPT; ++r) {
        const int i = ty + r * TY;
        const size_t o = (size_t)i * ld;
        sA[i][tx] = aTile[o + tx];
        sA[i][tx + TX] = aTile[o + tx + TX];
        sB[i][tx] = bTile[o + tx];
        sB[i][tx + TX] = bTile[o + tx + TX];
    }
    __syncthreads();

#pragma unroll 4
    for (int kk = 0; kk < TILE; ++kk) {
        const unsigned int kg = (unsigned int)(kb * TILE + kk);
        const unsigned int b0 = sB[kk][tx];
        const unsigned int b1 = sB[kk][tx + TX];
#pragma unroll
        for (int r = 0; r < RPT; ++r) {
            const unsigned int a = sA[ty + r * TY][kk];
            const unsigned int c0 = a + b0;
            const unsigned int c1 = a + b1;
            if (c0 < v[r][0]) {
                v[r][0] = c0;
                p[r][0] = kg;
            }
            if (c1 < v[r][1]) {
                v[r][1] = c1;
                p[r][1] = kg;
            }
        }
    }

    FW_STORE_REGS(tile, ptile, v, p, ld);
}

// path[i][j] is initialised to the row index i (see initializePathMatrix).
__global__ void fwInitPath(unsigned int* __restrict__ path, const int ld, const int rowBase, const int rows) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < rows && j < ld) path[(size_t)i * ld + j] = (unsigned int)(rowBase + i);
}

// ---------------------------------------------------------------------------
// Host side helpers
// ---------------------------------------------------------------------------

// glibc's rand_r: three steps of the LCG x -> 1103515245*x + 12345 per call.
inline int randR(unsigned int& state) noexcept {
    unsigned int next = state;
    int result;
    next = next * 1103515245u + 12345u;
    result = (int)((next / 65536u) % 2048u);
    next = next * 1103515245u + 12345u;
    result <<= 10;
    result ^= (int)((next / 65536u) % 1024u);
    next = next * 1103515245u + 12345u;
    result <<= 10;
    result ^= (int)((next / 65536u) % 1024u);
    state = next;
    return result;
}

// State of the generator right before the `idx`-th rand_r() call, obtained by
// jumping 3*idx LCG steps ahead - this makes the sequential initialisation
// loop of the original program perfectly parallelisable.
inline unsigned int seedAt(const size_t idx) noexcept {
    unsigned long long steps = 3ull * idx;
    unsigned int A = 1u, C = 0u, a = 1103515245u, c = 12345u;
    while (steps) {
        if (steps & 1ull) {
            A = a * A;
            C = a * C + c;
        }
        c = a * c + c;
        a = a * a;
        steps >>= 1;
    }
    return A * 42u + C;
}

// Fills the block-rows owned by this rank of the (padded) distance matrix with
// exactly the values the original sequential initialisation produces.  Padding
// rows/columns are INF off-diagonal and 0 on the diagonal so that they can
// never improve any real distance.
void initializeDistanceMatrix(unsigned int* dist, const size_t numNodes, const size_t ld, const size_t rowBase,
                              const size_t rows, const unsigned int rangeMin, const unsigned int rangeMax) {
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;

#pragma omp parallel for schedule(static)
    for (size_t lr = 0; lr < rows; ++lr) {
        const size_t g = rowBase + lr;
        unsigned int* row = dist + lr * ld;
        if (g < numNodes) {
            unsigned int seed = seedAt(g * numNodes);
            for (size_t c = 0; c < numNodes; ++c) {
                row[c] = rangeMin + (unsigned int)(range * randR(seed) / (double)RAND_MAX);
            }
            row[g] = 0; // distance from node to itself is 0
            for (size_t c = numNodes; c < ld; ++c) row[c] = INF;
        } else {
            for (size_t c = 0; c < ld; ++c) row[c] = (c == g) ? 0u : INF;
        }
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
    bool ok = true;
    const size_t lim = std::min(numNodes, static_cast<size_t>(10));
#pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = 0; i < lim; ++i) {
        for (size_t j = 0; j < lim; ++j) {
            for (size_t k = 0; k < numNodes; ++k) {
                const unsigned int distIJ = dist[idx2(j, i, numNodes)];
                const unsigned int distIK = dist[idx2(k, i, numNodes)];
                const unsigned int distKJ = dist[idx2(j, k, numNodes)];

                // Check for overflow before addition
                if (distIK < INF && distKJ < INF) {
                    if (distIK + distKJ < distIJ) {
                        printf("Validation failed: triangle inequality violated at [%zu,%zu,%zu]\n", i, j, k);
                        ok = false;
                    }
                }
            }
        }
    }

    return ok;
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

    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

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

    if (numNodes == 0) {
        if (rank == 0) printf("Number of nodes must be positive\n");
        MPI_Finalize();
        return 1;
    }

    // One GPU per rank, assigned by node-local rank.
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0, nodeSize = 1;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_size(nodeComm, &nodeSize);

    // One "leader" per node carries the inter-node traffic; inside a node the
    // pivot panel is exchanged through a shared-memory window, which is far
    // cheaper than a multi-copy MPI broadcast.
    MPI_Comm leaderComm = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, (localRank == 0) ? 0 : MPI_UNDEFINED, rank, &leaderComm);
    int nodeId = 0, numNodesInJob = 1;
    if (leaderComm != MPI_COMM_NULL) {
        MPI_Comm_rank(leaderComm, &nodeId);
        MPI_Comm_size(leaderComm, &numNodesInJob);
    }
    MPI_Bcast(&nodeId, 1, MPI_INT, 0, nodeComm);
    MPI_Bcast(&numNodesInJob, 1, MPI_INT, 0, nodeComm);
    std::vector<int> rankNodeId(nranks);
    MPI_Allgather(&nodeId, 1, MPI_INT, rankNodeId.data(), 1, MPI_INT, MPI_COMM_WORLD);
    const bool isLeader = (localRank == 0);
    const bool multiNode = (numNodesInJob > 1);

    int numDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));
    if (numDevices == 0) {
        if (rank == 0) fprintf(stderr, "No CUDA device available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % numDevices));

    // Padded problem size: a whole number of tiles.
    const int numBlocks = (int)((numNodes + TILE - 1) / TILE);
    const size_t N = (size_t)numBlocks * TILE;

    // Block-row decomposition across the ranks.  Ranks without any block-row
    // still take part in all collectives.
    auto blockRowBegin = [&](const int r) { return (int)((long long)numBlocks * r / nranks); };
    const int myBlockBegin = blockRowBegin(rank);
    const int myBlockEnd = blockRowBegin(rank + 1);
    const int myBlocks = myBlockEnd - myBlockBegin;
    const size_t myRowBase = (size_t)myBlockBegin * TILE;
    const size_t myRows = (size_t)myBlocks * TILE;
    auto ownerOf = [&](const int kb) {
        int lo = 0, hi = nranks - 1;
        while (lo < hi) {
            const int mid = (lo + hi) / 2;
            if (blockRowBegin(mid + 1) > kb) hi = mid; else lo = mid + 1;
        }
        return lo;
    };

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, OpenMP threads: %d, GPUs per node: %d\n", nranks, omp_get_max_threads(), numDevices);
    }

    // Host (pinned) storage for the local block-rows.
    unsigned int* h_dist = nullptr;
    unsigned int* h_path = nullptr;
    const size_t localElems = myRows * N;
    if (localElems > 0) {
        CUDA_CHECK(cudaHostAlloc(&h_dist, localElems * sizeof(unsigned int), cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&h_path, localElems * sizeof(unsigned int), cudaHostAllocDefault));
    }

    // Two node-wide shared slots for the pivot panel (double buffered).
    const size_t panelElems = (size_t)TILE * N;
    const size_t panelBytes = panelElems * sizeof(unsigned int);
    MPI_Win panelWin = MPI_WIN_NULL;
    unsigned int* h_panelBase = nullptr;
    if (nranks > 1) {
        void* base = nullptr;
        MPI_Win_allocate_shared(isLeader ? (MPI_Aint)(2 * panelBytes) : (MPI_Aint)0, (int)sizeof(unsigned int),
                                MPI_INFO_NULL, nodeComm, &base, &panelWin);
        MPI_Aint qsize = 0;
        int qdisp = 0;
        MPI_Win_shared_query(panelWin, 0, &qsize, &qdisp, &base);
        h_panelBase = static_cast<unsigned int*>(base);
        MPI_Win_lock_all(MPI_MODE_NOCHECK, panelWin);
        // Page-lock the window so that the D2H/H2D transfers run at full speed.
        if (cudaHostRegister(h_panelBase, 2 * panelBytes, cudaHostRegisterPortable) != cudaSuccess) {
            cudaGetLastError();
        }
    }
    unsigned int* h_panel[2] = {h_panelBase, h_panelBase ? h_panelBase + panelElems : nullptr};

    unsigned int* d_dist = nullptr;
    unsigned int* d_path = nullptr;
    unsigned int* d_rowk[2] = {nullptr, nullptr};
    if (localElems > 0) {
        CUDA_CHECK(cudaMalloc(&d_dist, localElems * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&d_path, localElems * sizeof(unsigned int)));
    }
    if (nranks > 1) {
        CUDA_CHECK(cudaMalloc(&d_rowk[0], panelBytes));
        CUDA_CHECK(cudaMalloc(&d_rowk[1], panelBytes));
    }

    cudaStream_t stream, copyStream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaStreamCreate(&copyStream));
    cudaEvent_t evUpload[2], evPivot, evPanelReady;
    CUDA_CHECK(cudaEventCreateWithFlags(&evUpload[0], cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&evUpload[1], cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&evPivot, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&evPanelReady, cudaEventDisableTiming));
    bool uploadPending[2] = {false, false};

    // Initialize
    if (rank == 0) printf("Initializing graph...\n");
    initializeDistanceMatrix(h_dist, numNodes, N, myRowBase, myRows, 1, MAX_DISTANCE);
    if (localElems > 0) {
        const dim3 pb(64, 4);
        const dim3 pg((unsigned)((N + pb.x - 1) / pb.x), (unsigned)((myRows + pb.y - 1) / pb.y));
        fwInitPath<<<pg, pb, 0, stream>>>(d_path, (int)N, (int)myRowBase, (int)myRows);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Run Floyd-Warshall
    if (rank == 0) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    if (localElems > 0) {
        CUDA_CHECK(cudaMemcpyAsync(d_dist, h_dist, localElems * sizeof(unsigned int), cudaMemcpyHostToDevice, stream));
    }

    const dim3 threads(TX, TY);

    // Produces the final pivot panel `kb`: its owner runs phase 1 + phase 2a on
    // it and makes it available to the other ranks.  Inside a node the panel is
    // handed over through the shared-memory window (a single device-to-host
    // copy plus one host-to-device copy per consumer), between nodes it is
    // broadcast among the node leaders.  All ranks have their phase 3 work of
    // the current step queued on the GPU before they enter the exchange, so
    // computation and communication run concurrently.
    auto preparePanel = [&](const int kb) {
        const int owner = ownerOf(kb);
        const size_t panelOff = (owner == rank) ? (size_t)(kb - myBlockBegin) * TILE * N : 0;
        if (owner == rank) {
            fwPhase1<<<1, threads, 0, stream>>>(d_dist + panelOff + (size_t)kb * TILE,
                                                d_path + panelOff + (size_t)kb * TILE, (int)N, kb);
            if (numBlocks > 1) {
                fwPhase2Row<<<numBlocks, threads, 0, stream>>>(d_dist + panelOff, d_path + panelOff, (int)N, kb);
            }
        }
        if (nranks == 1) return;

        // Everybody releases the slot that the *next* panel will land in; the
        // owner of that panel only writes it after the barrier below, so this
        // is enough to keep the double buffering race free.
        if (uploadPending[(kb + 1) & 1]) {
            CUDA_CHECK(cudaEventSynchronize(evUpload[(kb + 1) & 1]));
            uploadPending[(kb + 1) & 1] = false;
        }
        if (owner == rank) {
            // Hand the download off to the copy engine so that it proceeds in
            // parallel with the phase 3 work queued right after this call.
            CUDA_CHECK(cudaEventRecord(evPivot, stream));
            CUDA_CHECK(cudaStreamWaitEvent(copyStream, evPivot, 0));
            CUDA_CHECK(
                cudaMemcpyAsync(h_panel[kb & 1], d_dist + panelOff, panelBytes, cudaMemcpyDeviceToHost, copyStream));
            CUDA_CHECK(cudaEventRecord(evPanelReady, copyStream));
        }
    };

    // Completes the exchange started by preparePanel() and uploads the panel.
    auto finishPanel = [&](const int kb) {
        if (nranks == 1) return;
        const int owner = ownerOf(kb);
        if (owner == rank) CUDA_CHECK(cudaEventSynchronize(evPanelReady));
        MPI_Win_sync(panelWin);
        MPI_Barrier(nodeComm);
        MPI_Win_sync(panelWin);
        if (multiNode) {
            if (isLeader) {
                MPI_Bcast(h_panel[kb & 1], (int)panelElems, MPI_UNSIGNED, rankNodeId[owner], leaderComm);
            }
            MPI_Win_sync(panelWin);
            MPI_Barrier(nodeComm);
            MPI_Win_sync(panelWin);
        }
        if (owner != rank && myBlocks > 0) {
            CUDA_CHECK(cudaMemcpyAsync(d_rowk[kb & 1], h_panel[kb & 1], panelBytes, cudaMemcpyHostToDevice, copyStream));
            CUDA_CHECK(cudaEventRecord(evUpload[kb & 1], copyStream));
            uploadPending[kb & 1] = true;
        }
    };

    // The owner keeps reading its panel in place, everybody else the upload.
    auto panelPointer = [&](const int kb) -> const unsigned int* {
        if (ownerOf(kb) == rank) return d_dist + (size_t)(kb - myBlockBegin) * TILE * N;
        return d_rowk[kb & 1];
    };

    preparePanel(0);
    finishPanel(0);

    for (int kb = 0; kb < numBlocks; ++kb) {
        const unsigned int* rowk = panelPointer(kb);
        if (uploadPending[kb & 1]) CUDA_CHECK(cudaStreamWaitEvent(stream, evUpload[kb & 1], 0));

        if (myBlocks > 0) {
            fwPhase2Col<<<myBlocks, threads, 0, stream>>>(d_dist, d_path, rowk, (int)N, kb, myBlockBegin);
        }

        const int next = kb + 1;
        if (next < numBlocks) {
            const bool nextIsMine = (ownerOf(next) == rank);
            const dim3 grid3((unsigned)numBlocks, (unsigned)std::max(myBlocks, 0));
            if (myBlocks > 0) {
                // The owner of the next panel first finishes the block-row that
                // panel lives in, prepares it and only then runs the bulk of
                // phase 3 - so the exchange overlaps with that bulk.  All other
                // ranks queue their whole phase 3 before entering the exchange.
                fwPhase3<<<grid3, threads, 0, stream>>>(d_dist, d_path, rowk, (int)N, kb, myBlockBegin,
                                                        nextIsMine ? next : -1, -1);
            }
            preparePanel(next);
            if (nextIsMine) {
                fwPhase3<<<grid3, threads, 0, stream>>>(d_dist, d_path, rowk, (int)N, kb, myBlockBegin, -1, next);
            }
            finishPanel(next);
        } else if (myBlocks > 0) {
            const dim3 grid3((unsigned)numBlocks, (unsigned)myBlocks);
            fwPhase3<<<grid3, threads, 0, stream>>>(d_dist, d_path, rowk, (int)N, kb, myBlockBegin, -1, -1);
        }
    }

    if (localElems > 0) {
        CUDA_CHECK(cudaMemcpyAsync(h_dist, d_dist, localElems * sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(h_path, d_path, localElems * sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamSynchronize(copyStream));
    CUDA_CHECK(cudaGetLastError());
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

    // Collect the distributed result on rank 0 for output / validation.
    std::vector<unsigned int> dist;
    if (printResults || validate) {
        const size_t myRealRows = (myRowBase >= numNodes) ? 0 : std::min(myRows, numNodes - myRowBase);
        std::vector<unsigned int> sendbuf(myRealRows * numNodes);
#pragma omp parallel for schedule(static)
        for (size_t lr = 0; lr < myRealRows; ++lr) {
            memcpy(&sendbuf[lr * numNodes], h_dist + lr * N, numNodes * sizeof(unsigned int));
        }

        std::vector<int> counts(nranks), displs(nranks);
        if (rank == 0) {
            dist.resize(numNodes * numNodes);
            size_t off = 0;
            for (int r = 0; r < nranks; ++r) {
                const size_t rb = (size_t)blockRowBegin(r) * TILE;
                const size_t re = std::min((size_t)blockRowBegin(r + 1) * TILE, numNodes);
                const size_t rows = (rb >= numNodes) ? 0 : re - rb;
                counts[r] = (int)(rows * numNodes);
                displs[r] = (int)off;
                off += rows * numNodes;
            }
        }
        MPI_Gatherv(sendbuf.data(), (int)(myRealRows * numNodes), MPI_UNSIGNED, dist.data(), counts.data(),
                    displs.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    }

    // Print results for external validation (integer hash-based)
    if (printResults && rank == 0) {
        print_results_int(dist, "DistanceMatrix");
    }

    int status = 0;
    // Validation
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
            bool valid = validateResult(dist, numNodes);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                status = 1;
            }
        }
        MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    if (h_dist) CUDA_CHECK(cudaFreeHost(h_dist));
    if (h_path) CUDA_CHECK(cudaFreeHost(h_path));
    if (d_dist) CUDA_CHECK(cudaFree(d_dist));
    if (d_path) CUDA_CHECK(cudaFree(d_path));
    if (d_rowk[0]) CUDA_CHECK(cudaFree(d_rowk[0]));
    if (d_rowk[1]) CUDA_CHECK(cudaFree(d_rowk[1]));
    CUDA_CHECK(cudaEventDestroy(evUpload[0]));
    CUDA_CHECK(cudaEventDestroy(evUpload[1]));
    CUDA_CHECK(cudaEventDestroy(evPivot));
    CUDA_CHECK(cudaEventDestroy(evPanelReady));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaStreamDestroy(copyStream));
    if (panelWin != MPI_WIN_NULL) {
        cudaHostUnregister(h_panelBase);
        cudaGetLastError();
        MPI_Win_unlock_all(panelWin);
        MPI_Win_free(&panelWin);
    }
    if (leaderComm != MPI_COMM_NULL) MPI_Comm_free(&leaderComm);
    MPI_Comm_free(&nodeComm);
    MPI_Finalize();
    return status;
}
