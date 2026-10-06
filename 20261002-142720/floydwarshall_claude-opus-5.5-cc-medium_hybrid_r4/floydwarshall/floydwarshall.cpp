// Hybrid MPI + OpenMP + CUDA Floyd-Warshall.
//
// Parallelization scheme
//  * The (padded) N x N matrix is cut into TILE x TILE tiles. Each MPI rank owns a
//    contiguous range of tile rows, resident on its own GPU.
//  * Blocked Floyd-Warshall: for every pivot tile index kb
//       phase 1: diagonal tile (kb,kb)            -- owner of tile row kb
//       phase 2: row panel (kb,*)                 -- owner of tile row kb
//                -> copied into a node-shared MPI-3 window (pinned for DMA), node
//                   leaders forward it to other nodes with MPI_Bcast
//       phase 2: column tiles (*,kb)              -- every rank, its own rows
//       phase 3: all remaining tiles              -- every rank, its own rows
//    The owner of tile row kb+1 computes the next row panel first (lookahead), so
//    its broadcast overlaps with the bulk phase-3 work of step kb.
//  * OpenMP parallelizes host-side work (graph generation with an exact rand_r
//    jump-ahead, result packing for the gather).
//
// Path semantics: the original algorithm stores in path[i][j] the last pivot k that
// strictly improved dist[i][j] (or i if never improved). That value is exactly the
// minimal "largest intermediate vertex" among all shortest i->j walks. The blocked
// algorithm therefore runs over the lexicographic (distance, maxIntermediate+1)
// semiring, which is closed under reordering and yields bit-identical dist and path.

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

// Tile edge length and thread-block shape (16x16 threads, 4x4 elements each)
constexpr int TILE = 64;
constexpr int TPB_X = 16;
constexpr int TPB_Y = 16;
constexpr int PER_T = TILE / TPB_X;  // 4
// Distance used for padding nodes: never improves anything, sums never overflow
constexpr unsigned int PAD_DIST = 0x3fffffffu;

#define CUDA_CHECK(call)                                                            \
    do {                                                                            \
        cudaError_t err__ = (call);                                                 \
        if (err__ != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err__),  \
                    __FILE__, __LINE__);                                            \
            MPI_Abort(MPI_COMM_WORLD, 1);                                           \
        }                                                                           \
    } while (0)

// Index calculation for flattened 2D array
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

// ---------------------------------------------------------------------------------
// Device code. Each matrix entry is a pair (d, m): d = distance, m = 1 + largest
// intermediate vertex of the represented walk (0 = direct edge).
// ---------------------------------------------------------------------------------

// Relax (cd,cm) with candidate (nd,nm) in lexicographic order
__device__ __forceinline__ void relax(unsigned int& cd, unsigned int& cm,
                                      const unsigned int nd, const unsigned int nm) {
    // (d, m) compared as one 64-bit key: d in the high word, m in the low word
    const unsigned long long cand = ((unsigned long long)nd << 32) | nm;
    const unsigned long long cur = ((unsigned long long)cd << 32) | cm;
    const bool better = cand < cur;
    cd = better ? nd : cd;
    cm = better ? nm : cm;
}

// Dependent phases (1 and 2): tile T is updated in place using the pivot block,
// sequentially over the 64 pivots kk of block kb. Entries (i,kk) and (kk,j) never
// change during step kk (the diagonal is (0,0) and any candidate through kk has
// m >= kk+1), so their current values can be shared without further ordering.
//
// Thread layout (256 threads): warp w, lane l -> line = w*8 + l/4, group g = l%4.
// Each thread holds 16 consecutive entries of one line (row or column) in registers;
// the pivot entry of its own line is obtained from a lane of the same warp by shuffle.
constexpr int DEP_THREADS = 256;
constexpr int DEP_LD = TILE + 4;

__device__ __forceinline__ void load16(const unsigned int* __restrict__ p, unsigned int (&v)[16]) {
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const uint4 t = *reinterpret_cast<const uint4*>(p + 4 * q);
        v[4 * q] = t.x; v[4 * q + 1] = t.y; v[4 * q + 2] = t.z; v[4 * q + 3] = t.w;
    }
}
__device__ __forceinline__ void store16(unsigned int* __restrict__ p, const unsigned int (&v)[16]) {
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        *reinterpret_cast<uint4*>(p + 4 * q) = make_uint4(v[4 * q], v[4 * q + 1], v[4 * q + 2], v[4 * q + 3]);
    }
}

// Phase 1: diagonal tile (kb,kb) of local tile row 'lt'. T = T (x) T.
// Thread owns row i, columns g*16..g*16+15. Row kk is published through shared memory.
__global__ void __launch_bounds__(DEP_THREADS)
diagKernel(unsigned int* __restrict__ D, unsigned int* __restrict__ M, const size_t ld,
           const int lt, const int kb) {
    __shared__ __align__(16) unsigned int sRowD[2][TILE];
    __shared__ __align__(16) unsigned int sRowM[2][TILE];
    const int lane = threadIdx.x & 31;
    const int i = (threadIdx.x >> 5) * 8 + (lane >> 2);
    const int g = lane & 3;
    const size_t base = ((size_t)lt * TILE + i) * ld + (size_t)kb * TILE + g * 16;

    unsigned int td[16], tm[16];
    load16(D + base, td);
    load16(M + base, tm);
    if (i == 0) { store16(&sRowD[0][g * 16], td); store16(&sRowM[0][g * 16], tm); }
    __syncthreads();

    const unsigned int kBase = (unsigned int)kb * TILE + 1;
    const int srcBase = lane & ~3;
#pragma unroll 1
    for (int kq = 0; kq < 4; ++kq) {
#pragma unroll
        for (int k16 = 0; k16 < 16; ++k16) {
            const int kk = kq * 16 + k16;
            const int b = kk & 1;
            const unsigned int ad = __shfl_sync(0xffffffffu, td[k16], srcBase | kq);
            const unsigned int am = max(__shfl_sync(0xffffffffu, tm[k16], srcBase | kq), kBase + kk);
            unsigned int bd[16], bm[16];
            load16(&sRowD[b][g * 16], bd);
            load16(&sRowM[b][g * 16], bm);
#pragma unroll
            for (int c = 0; c < 16; ++c) relax(td[c], tm[c], ad + bd[c], max(am, bm[c]));
            if (i == kk + 1) { store16(&sRowD[b ^ 1][g * 16], td); store16(&sRowM[b ^ 1][g * 16], tm); }
            __syncthreads();
        }
    }
    store16(D + base, td);
    store16(M + base, tm);
}

// Phase 2, row panel: tiles (kb, J), J != kb, of local tile row 'lt'. T = P (x) T with
// P the finished diagonal tile. Columns are independent: thread owns column j, rows
// g*16..g*16+15. P is kept transposed in shared memory with m pre-maxed by k+1.
__global__ void __launch_bounds__(DEP_THREADS)
rowPanelKernel(unsigned int* __restrict__ D, unsigned int* __restrict__ M, const size_t ld,
               const int lt, const int kb) {
    const int J = blockIdx.x;
    if (J == kb) return;
    __shared__ __align__(16) unsigned int sPd[TILE][DEP_LD];  // [kk][i]
    __shared__ __align__(16) unsigned int sPm[TILE][DEP_LD];
    const size_t row0 = (size_t)lt * TILE;
    const unsigned int kBase = (unsigned int)kb * TILE + 1;
    for (int e = threadIdx.x; e < TILE * TILE; e += DEP_THREADS) {
        const int ii = e / TILE, kk = e % TILE;
        const size_t gidx = (row0 + ii) * ld + (size_t)kb * TILE + kk;
        sPd[kk][ii] = D[gidx];
        sPm[kk][ii] = max(M[gidx], kBase + kk);
    }

    const int lane = threadIdx.x & 31;
    const int j = (threadIdx.x >> 5) * 8 + (lane >> 2);
    const int g = lane & 3;
    const size_t col = (size_t)J * TILE + j;
    unsigned int td[16], tm[16];
#pragma unroll
    for (int r = 0; r < 16; ++r) {
        td[r] = D[(row0 + g * 16 + r) * ld + col];
        tm[r] = M[(row0 + g * 16 + r) * ld + col];
    }
    __syncthreads();

    const int srcBase = lane & ~3;
#pragma unroll 1
    for (int kq = 0; kq < 4; ++kq) {
#pragma unroll
        for (int k16 = 0; k16 < 16; ++k16) {
            const int kk = kq * 16 + k16;
            const unsigned int bd = __shfl_sync(0xffffffffu, td[k16], srcBase | kq);
            const unsigned int bm = __shfl_sync(0xffffffffu, tm[k16], srcBase | kq);
            unsigned int ad[16], am[16];
            load16(&sPd[kk][g * 16], ad);
            load16(&sPm[kk][g * 16], am);
#pragma unroll
            for (int r = 0; r < 16; ++r) relax(td[r], tm[r], ad[r] + bd, max(am[r], bm));
        }
    }
#pragma unroll
    for (int r = 0; r < 16; ++r) {
        D[(row0 + g * 16 + r) * ld + col] = td[r];
        M[(row0 + g * 16 + r) * ld + col] = tm[r];
    }
}

// Phase 2, column: tiles (I, kb) for local tile rows I in [lr0, lr0 + gridDim.x),
// I != skip. T = T (x) P with P the finished diagonal tile (row-major, stride pld).
// Rows are independent: thread owns row i, columns g*16..g*16+15.
__global__ void __launch_bounds__(DEP_THREADS)
colPanelKernel(unsigned int* __restrict__ D, unsigned int* __restrict__ M, const size_t ld,
               const unsigned int* __restrict__ PD, const unsigned int* __restrict__ PM,
               const size_t pld, const int lr0, const int skip, const int kb) {
    const int I = lr0 + blockIdx.x;
    if (I == skip) return;
    __shared__ __align__(16) unsigned int sPd[TILE][DEP_LD];  // [kk][j]
    __shared__ __align__(16) unsigned int sPm[TILE][DEP_LD];
    const unsigned int kBase = (unsigned int)kb * TILE + 1;
    for (int e = threadIdx.x; e < TILE * TILE; e += DEP_THREADS) {
        const int kk = e / TILE, jj = e % TILE;
        const size_t gidx = (size_t)kk * pld + (size_t)kb * TILE + jj;
        sPd[kk][jj] = PD[gidx];
        sPm[kk][jj] = max(PM[gidx], kBase + kk);
    }

    const int lane = threadIdx.x & 31;
    const int i = (threadIdx.x >> 5) * 8 + (lane >> 2);
    const int g = lane & 3;
    const size_t base = ((size_t)I * TILE + i) * ld + (size_t)kb * TILE + g * 16;
    unsigned int td[16], tm[16];
    load16(D + base, td);
    load16(M + base, tm);
    __syncthreads();

    const int srcBase = lane & ~3;
#pragma unroll 1
    for (int kq = 0; kq < 4; ++kq) {
#pragma unroll
        for (int k16 = 0; k16 < 16; ++k16) {
            const int kk = kq * 16 + k16;
            const unsigned int ad = __shfl_sync(0xffffffffu, td[k16], srcBase | kq);
            const unsigned int am = __shfl_sync(0xffffffffu, tm[k16], srcBase | kq);
            unsigned int bd[16], bm[16];
            load16(&sPd[kk][g * 16], bd);
            load16(&sPm[kk][g * 16], bm);
#pragma unroll
            for (int c = 0; c < 16; ++c) relax(td[c], tm[c], ad + bd[c], max(am, bm[c]));
        }
    }
    store16(D + base, td);
    store16(M + base, tm);
}

// Shared memory layout of the independent phase (A stored transposed)
constexpr int SA_LD = TILE + 4;  // keeps 16-byte alignment, reduces bank conflicts
constexpr size_t PHASE3_SMEM = (size_t)(2 * TILE * SA_LD + 2 * TILE * TILE) * sizeof(unsigned int);

// Independent phase 3: C(I,J) = C (x) A(I,kb) (x) P(kb,J) for all local tile rows in
// [tRow0, tRow0 + gridDim.y) and all tile columns except kb. A and P are final.
__global__ void __launch_bounds__(TPB_X * TPB_Y, 2)
independentKernel(unsigned int* __restrict__ D, unsigned int* __restrict__ M, const size_t ld,
                  const unsigned int* __restrict__ PD, const unsigned int* __restrict__ PM,
                  const int tRow0, const int skipRow, const int kb) {
    const int J = blockIdx.x;
    const int I = tRow0 + blockIdx.y;
    if (J == kb || I == skipRow) return;

    extern __shared__ __align__(16) unsigned int smem[];
    unsigned int* sAd = smem;                     // [TILE][SA_LD], transposed: [kk][i]
    unsigned int* sAm = sAd + TILE * SA_LD;
    unsigned int* sBd = sAm + TILE * SA_LD;       // [TILE][TILE]: [kk][j]
    unsigned int* sBm = sBd + TILE * TILE;

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * TPB_X + tx;
    const size_t row0 = (size_t)I * TILE;
    const size_t col0 = (size_t)J * TILE;
    const size_t kcol0 = (size_t)kb * TILE;
    const unsigned int kBase = (unsigned int)kb * TILE + 1;

    // Load A (column tile) transposed and B (row-panel tile) with m pre-maxed by k+1
    for (int e = tid; e < TILE * TILE / 4; e += TPB_X * TPB_Y) {
        const int i = e / (TILE / 4);
        const int q = (e % (TILE / 4)) * 4;
        const uint4 ad = *reinterpret_cast<const uint4*>(&D[(row0 + i) * ld + kcol0 + q]);
        const uint4 am = *reinterpret_cast<const uint4*>(&M[(row0 + i) * ld + kcol0 + q]);
        sAd[(q + 0) * SA_LD + i] = ad.x; sAd[(q + 1) * SA_LD + i] = ad.y;
        sAd[(q + 2) * SA_LD + i] = ad.z; sAd[(q + 3) * SA_LD + i] = ad.w;
        sAm[(q + 0) * SA_LD + i] = am.x; sAm[(q + 1) * SA_LD + i] = am.y;
        sAm[(q + 2) * SA_LD + i] = am.z; sAm[(q + 3) * SA_LD + i] = am.w;

        const uint4 bd = *reinterpret_cast<const uint4*>(&PD[(size_t)i * ld + col0 + q]);
        uint4 bm = *reinterpret_cast<const uint4*>(&PM[(size_t)i * ld + col0 + q]);
        const unsigned int kv = kBase + i;
        bm.x = max(bm.x, kv); bm.y = max(bm.y, kv); bm.z = max(bm.z, kv); bm.w = max(bm.w, kv);
        *reinterpret_cast<uint4*>(&sBd[i * TILE + q]) = bd;
        *reinterpret_cast<uint4*>(&sBm[i * TILE + q]) = bm;
    }

    // Load own 4x4 block of C into registers: rows ty*4+r, cols tx*4+c
    unsigned int cd[PER_T][PER_T], cm[PER_T][PER_T];
#pragma unroll
    for (int r = 0; r < PER_T; ++r) {
        const size_t g = (row0 + ty * PER_T + r) * ld + col0 + tx * PER_T;
        const uint4 vd = *reinterpret_cast<const uint4*>(&D[g]);
        const uint4 vm = *reinterpret_cast<const uint4*>(&M[g]);
        cd[r][0] = vd.x; cd[r][1] = vd.y; cd[r][2] = vd.z; cd[r][3] = vd.w;
        cm[r][0] = vm.x; cm[r][1] = vm.y; cm[r][2] = vm.z; cm[r][3] = vm.w;
    }
    __syncthreads();

#pragma unroll 8
    for (int kk = 0; kk < TILE; ++kk) {
        const uint4 a4d = *reinterpret_cast<const uint4*>(&sAd[kk * SA_LD + ty * PER_T]);
        const uint4 a4m = *reinterpret_cast<const uint4*>(&sAm[kk * SA_LD + ty * PER_T]);
        const uint4 b4d = *reinterpret_cast<const uint4*>(&sBd[kk * TILE + tx * PER_T]);
        const uint4 b4m = *reinterpret_cast<const uint4*>(&sBm[kk * TILE + tx * PER_T]);
        const unsigned int ad[4] = {a4d.x, a4d.y, a4d.z, a4d.w};
        const unsigned int am[4] = {a4m.x, a4m.y, a4m.z, a4m.w};
        const unsigned int bd[4] = {b4d.x, b4d.y, b4d.z, b4d.w};
        const unsigned int bm[4] = {b4m.x, b4m.y, b4m.z, b4m.w};
#pragma unroll
        for (int r = 0; r < PER_T; ++r) {
#pragma unroll
            for (int c = 0; c < PER_T; ++c) {
                relax(cd[r][c], cm[r][c], ad[r] + bd[c], max(am[r], bm[c]));
            }
        }
    }

#pragma unroll
    for (int r = 0; r < PER_T; ++r) {
        const size_t g = (row0 + ty * PER_T + r) * ld + col0 + tx * PER_T;
        *reinterpret_cast<uint4*>(&D[g]) = make_uint4(cd[r][0], cd[r][1], cd[r][2], cd[r][3]);
        *reinterpret_cast<uint4*>(&M[g]) = make_uint4(cm[r][0], cm[r][1], cm[r][2], cm[r][3]);
    }
}

// Convert m-encoding to the original path convention: path[i][j] = m-1, or i if m==0
__global__ void decodePathKernel(unsigned int* __restrict__ M, const size_t ld,
                                 const size_t rows, const size_t globalRow0) {
    const size_t total = rows * ld;
    for (size_t e = blockIdx.x * (size_t)blockDim.x + threadIdx.x; e < total;
         e += (size_t)gridDim.x * blockDim.x) {
        const unsigned int m = M[e];
        M[e] = m ? m - 1 : (unsigned int)(globalRow0 + e / ld);
    }
}

// ---------------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------------

// Advance an LCG state x -> a*x + c (mod 2^32) by 'steps' steps
static unsigned int lcgJump(unsigned int x, uint64_t steps) {
    unsigned int a = 1103515245u, c = 12345u;  // glibc rand_r LCG
    unsigned int accA = 1u, accC = 0u;
    while (steps) {
        if (steps & 1) {
            accA = accA * a;
            accC = accC * a + c;
        }
        c = c * a + c;
        a = a * a;
        steps >>= 1;
    }
    return accA * x + accC;
}

// Check that rand_r advances its state by three LCG steps per call (glibc)
static bool randJumpSupported() {
    unsigned int s = 42;
    for (int t = 1; t <= 1000; ++t) {
        rand_r(&s);
        if (s != lcgJump(42u, 3ull * t)) return false;
    }
    return true;
}

// Generate rows [gRow0, gRow1) of the original distance matrix into a padded
// row-major local buffer (row stride ld). Matches the sequential rand_r stream exactly.
void initializeDistanceMatrix(unsigned int* dist, const size_t numNodes, const size_t ld,
                              const size_t gRow0, const size_t gRow1, const size_t padRows,
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;
    const bool jump = randJumpSupported();

    auto genRow = [&](const size_t gi, unsigned int& seed) {
        unsigned int* row = dist + (gi - gRow0) * ld;
        for (size_t j = 0; j < numNodes; ++j) {
            row[j] = rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
        }
        for (size_t j = numNodes; j < ld; ++j) row[j] = PAD_DIST;
        // Set diagonal to zero (distance from node to itself is 0)
        row[gi] = 0;
    };

    if (jump) {
#pragma omp parallel for schedule(static)
        for (size_t gi = gRow0; gi < gRow1; ++gi) {
            unsigned int seed = lcgJump(42u, 3ull * gi * numNodes);
            genRow(gi, seed);
        }
    } else {
        unsigned int seed = 42;
        for (size_t s = 0; s < gRow0 * numNodes; ++s) rand_r(&seed);
        for (size_t gi = gRow0; gi < gRow1; ++gi) genRow(gi, seed);
    }

    // Padding rows (beyond numNodes): isolated nodes
#pragma omp parallel for schedule(static)
    for (size_t p = 0; p < padRows; ++p) {
        const size_t gi = gRow1 + p;
        unsigned int* row = dist + (gi - gRow0) * ld;
        for (size_t j = 0; j < ld; ++j) row[j] = PAD_DIST;
        row[gi] = 0;
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

// Distributed blocked Floyd-Warshall state for one rank
struct DistFW {
    int rank = 0, nranks = 1;
    size_t n = 0;      // real number of nodes
    size_t N = 0;      // padded size (multiple of TILE)
    int nb = 0;        // number of tile rows/cols
    int t0 = 0, t1 = 0;  // owned tile rows [t0, t1)
    size_t localRows = 0;

    unsigned int *dD = nullptr, *dM = nullptr;           // local rows, row-major, ld = N
    unsigned int *dPanel[2] = {nullptr, nullptr};        // received row panel: dist | m
    cudaStream_t sComp = nullptr, sCopy = nullptr;
    cudaEvent_t evPanelReady[2], evPanelFree[2], evH2DDone[2], evD2HDone[2], evPanelComputed;

    // Panel exchange: one node-shared host buffer pair per node (MPI-3 shared window);
    // node leaders forward panels between nodes.
    MPI_Comm nodeComm = MPI_COMM_NULL, leaderComm = MPI_COMM_NULL;
    int nNodes = 1;
    std::vector<int> nodeOfRank;
    MPI_Win win = MPI_WIN_NULL;
    unsigned int* shPanel[2] = {nullptr, nullptr};
    void* shBase = nullptr;
    bool shRegistered = false;

    int tileStart(int r) const {
        const int base = nb / nranks, extra = nb % nranks;
        return r * base + std::min(r, extra);
    }
    int owner(int kb) const {
        int lo = 0, hi = nranks - 1;
        while (lo < hi) {
            const int mid = (lo + hi + 1) / 2;
            if (tileStart(mid) <= kb) lo = mid; else hi = mid - 1;
        }
        return lo;
    }
    size_t panelElems() const { return (size_t)TILE * N; }
    int localSkip(int kb) const { return (kb >= t0 && kb < t1) ? kb - t0 : -1; }

    // Pointers to the row panel of pivot kb (local storage for the owner)
    const unsigned int* panelD(int kb) const {
        return owner(kb) == rank ? dD + (size_t)(kb - t0) * TILE * N : dPanel[kb & 1];
    }
    const unsigned int* panelM(int kb) const {
        return owner(kb) == rank ? dM + (size_t)(kb - t0) * TILE * N
                                 : dPanel[kb & 1] + panelElems();
    }

    void setupComm(MPI_Comm node) {
        nodeComm = node;
        int nodeRank = 0;
        MPI_Comm_rank(nodeComm, &nodeRank);
        MPI_Comm_split(MPI_COMM_WORLD, nodeRank == 0 ? 0 : MPI_UNDEFINED, rank, &leaderComm);
        int nodeId = 0;
        if (leaderComm != MPI_COMM_NULL) {
            MPI_Comm_rank(leaderComm, &nodeId);
            MPI_Comm_size(leaderComm, &nNodes);
        }
        MPI_Bcast(&nodeId, 1, MPI_INT, 0, nodeComm);
        MPI_Bcast(&nNodes, 1, MPI_INT, 0, nodeComm);
        nodeOfRank.resize(nranks);
        MPI_Allgather(&nodeId, 1, MPI_INT, nodeOfRank.data(), 1, MPI_INT, MPI_COMM_WORLD);

        const size_t bytesPerPanel = 2 * panelElems() * sizeof(unsigned int);
        const MPI_Aint mySize = nodeRank == 0 ? (MPI_Aint)std::max<size_t>(2 * bytesPerPanel, 64) : 0;
        void* myBase = nullptr;
        MPI_Win_allocate_shared(mySize, 1, MPI_INFO_NULL, nodeComm, &myBase, &win);
        MPI_Aint size = 0;
        int dispUnit = 1;
        MPI_Win_shared_query(win, 0, &size, &dispUnit, &shBase);
        shPanel[0] = static_cast<unsigned int*>(shBase);
        shPanel[1] = shPanel[0] + 2 * panelElems();
        // Pin the shared buffer for fast asynchronous DMA (falls back to pageable copies)
        shRegistered = cudaHostRegister(shBase, (size_t)size, cudaHostRegisterPortable) == cudaSuccess;
        if (!shRegistered) (void)cudaGetLastError();
        MPI_Win_lock_all(MPI_MODE_NOCHECK, win);
    }

    void freeComm() {
        MPI_Win_unlock_all(win);
        if (shRegistered) cudaHostUnregister(shBase);
        MPI_Win_free(&win);
        if (leaderComm != MPI_COMM_NULL) MPI_Comm_free(&leaderComm);
    }

    // Phase 1 + phase 2 (row) for pivot kb on the owning rank
    void computePanel(int kb) {
        const int lt = kb - t0;
        diagKernel<<<1, DEP_THREADS, 0, sComp>>>(dD, dM, N, lt, kb);
        if (nb > 1) rowPanelKernel<<<nb, DEP_THREADS, 0, sComp>>>(dD, dM, N, lt, kb);
    }

    // Column tiles (I, kb) for local tile rows [lr0, lr1)
    void computeColumn(int kb, int lr0, int lr1) {
        if (lr1 <= lr0) return;
        colPanelKernel<<<lr1 - lr0, DEP_THREADS, 0, sComp>>>(
            dD, dM, N, panelD(kb), panelM(kb), N, lr0, localSkip(kb), kb);
    }

    // Phase 3 over local tile rows [lr0, lr1)
    void computeRest(int kb, int lr0, int lr1) {
        if (lr1 <= lr0) return;
        dim3 grid(nb, lr1 - lr0);
        independentKernel<<<grid, dim3(TPB_X, TPB_Y), PHASE3_SMEM, sComp>>>(
            dD, dM, N, panelD(kb), panelM(kb), lr0, localSkip(kb), kb);
    }

    // Owner: copy panel kb (already queued on sComp) into the node-shared buffer
    void sendPanel(int kb) {
        const int b = kb & 1;
        const size_t lt = (size_t)(kb - t0);
        CUDA_CHECK(cudaEventRecord(evPanelComputed, sComp));
        CUDA_CHECK(cudaStreamWaitEvent(sCopy, evPanelComputed, 0));
        CUDA_CHECK(cudaMemcpyAsync(shPanel[b], dD + lt * TILE * N, panelElems() * sizeof(unsigned int),
                                   cudaMemcpyDeviceToHost, sCopy));
        CUDA_CHECK(cudaMemcpyAsync(shPanel[b] + panelElems(), dM + lt * TILE * N,
                                   panelElems() * sizeof(unsigned int), cudaMemcpyDeviceToHost, sCopy));
        CUDA_CHECK(cudaEventRecord(evD2HDone[b], sCopy));
    }

    // Distribute panel kb; non-owners upload it to the device and order sComp after it.
    void broadcastPanel(int kb) {
        const int b = kb & 1;
        const int root = owner(kb);
        if (root == rank) CUDA_CHECK(cudaEventSynchronize(evD2HDone[b]));
        // Our upload of panel kb-1 must be finished before its buffer is overwritten
        // (by the owner of kb+1, after the barrier below).
        CUDA_CHECK(cudaEventSynchronize(evH2DDone[b ^ 1]));
        MPI_Win_sync(win);
        MPI_Barrier(nodeComm);
        if (nNodes > 1) {
            if (leaderComm != MPI_COMM_NULL) {
                const size_t count = 2 * panelElems();
                const size_t chunk = (size_t)1 << 28;
                for (size_t off = 0; off < count; off += chunk) {
                    MPI_Bcast(shPanel[b] + off, (int)std::min(chunk, count - off), MPI_UNSIGNED,
                              nodeOfRank[root], leaderComm);
                }
            }
            MPI_Barrier(nodeComm);
        }
        MPI_Win_sync(win);
        if (root != rank) {
            // Device panel buffer must no longer be read by step kb-2
            CUDA_CHECK(cudaStreamWaitEvent(sCopy, evPanelFree[b], 0));
            CUDA_CHECK(cudaMemcpyAsync(dPanel[b], shPanel[b], 2 * panelElems() * sizeof(unsigned int),
                                       cudaMemcpyHostToDevice, sCopy));
            CUDA_CHECK(cudaEventRecord(evH2DDone[b], sCopy));
            CUDA_CHECK(cudaEventRecord(evPanelReady[b], sCopy));
            CUDA_CHECK(cudaStreamWaitEvent(sComp, evPanelReady[b], 0));
        }
    }

    void run() {
        if (nb == 0) return;
        const int nLocal = t1 - t0;

        if (owner(0) == rank) { computePanel(0); sendPanel(0); }
        broadcastPanel(0);

        for (int kb = 0; kb < nb; ++kb) {
            const int next = kb + 1;
            const bool haveNext = next < nb;
            if (haveNext && owner(next) == rank) {
                // Lookahead: finish step kb on tile row 'next' and build its panel first,
                // so the panel exchange overlaps with the bulk of phase 3.
                const int ln = next - t0;
                computeColumn(kb, ln, ln + 1);
                computeRest(kb, ln, ln + 1);
                computePanel(next);
                sendPanel(next);
                computeColumn(kb, 0, ln);
                computeColumn(kb, ln + 1, nLocal);
                computeRest(kb, 0, ln);
                computeRest(kb, ln + 1, nLocal);
            } else {
                computeColumn(kb, 0, nLocal);
                computeRest(kb, 0, nLocal);
            }
            CUDA_CHECK(cudaEventRecord(evPanelFree[kb & 1], sComp));
            if (haveNext) broadcastPanel(next);
        }
        CUDA_CHECK(cudaStreamSynchronize(sComp));
        CUDA_CHECK(cudaGetLastError());
    }
};

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
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

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // One GPU per rank (round-robin over node-local ranks)
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    {
        int localRank = 0;
        MPI_Comm_rank(nodeComm, &localRank);
        int ndev = 0;
        CUDA_CHECK(cudaGetDeviceCount(&ndev));
        if (ndev == 0) {
            fprintf(stderr, "No CUDA device available\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        CUDA_CHECK(cudaSetDevice(localRank % ndev));
    }

    DistFW fw;
    fw.rank = rank;
    fw.nranks = nranks;
    fw.n = numNodes;
    fw.nb = (int)((numNodes + TILE - 1) / TILE);
    fw.N = (size_t)fw.nb * TILE;
    fw.t0 = fw.tileStart(rank);
    fw.t1 = fw.tileStart(rank + 1);
    fw.localRows = (size_t)(fw.t1 - fw.t0) * TILE;
    const size_t N = fw.N;
    const size_t gRow0 = (size_t)fw.t0 * TILE;
    const size_t gRowEndPad = gRow0 + fw.localRows;
    const size_t gRow1 = std::min(gRowEndPad, numNodes);           // real rows [gRow0, gRow1)
    const size_t realRows = gRow1 > gRow0 ? gRow1 - gRow0 : 0;
    const size_t localElems = fw.localRows * N;

    // Host buffers (pinned) for the local block of rows
    unsigned int *hDist = nullptr, *hPath = nullptr;
    CUDA_CHECK(cudaMallocHost(&hDist, std::max<size_t>(localElems, 1) * sizeof(unsigned int)));
    CUDA_CHECK(cudaMallocHost(&hPath, std::max<size_t>(localElems, 1) * sizeof(unsigned int)));

    // Initialize
    if (rank == 0) printf("Initializing graph...\n");
    initializeDistanceMatrix(hDist, numNodes, N, gRow0, std::max(gRow0, gRow1),
                             gRowEndPad - std::max(gRow0, gRow1), 1, MAX_DISTANCE);

    // Device buffers
    CUDA_CHECK(cudaMalloc(&fw.dD, std::max<size_t>(localElems, 1) * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&fw.dM, std::max<size_t>(localElems, 1) * sizeof(unsigned int)));
    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaMalloc(&fw.dPanel[b], std::max<size_t>(2 * fw.panelElems(), 1) * sizeof(unsigned int)));
        CUDA_CHECK(cudaEventCreateWithFlags(&fw.evPanelReady[b], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&fw.evPanelFree[b], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&fw.evH2DDone[b], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&fw.evD2HDone[b], cudaEventDisableTiming));
    }
    fw.setupComm(nodeComm);
    CUDA_CHECK(cudaEventCreateWithFlags(&fw.evPanelComputed, cudaEventDisableTiming));
    CUDA_CHECK(cudaStreamCreateWithFlags(&fw.sComp, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&fw.sCopy, cudaStreamNonBlocking));
    CUDA_CHECK(cudaFuncSetAttribute(independentKernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    (int)PHASE3_SMEM));
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    // Run Floyd-Warshall
    if (rank == 0) printf("Computing shortest paths...\n");
    auto start = std::chrono::high_resolution_clock::now();

    if (localElems > 0) {
        CUDA_CHECK(cudaMemcpyAsync(fw.dD, hDist, localElems * sizeof(unsigned int),
                                   cudaMemcpyHostToDevice, fw.sComp));
        // Path initialization: m = 0 encodes "no intermediate" (path[i][j] = i)
        CUDA_CHECK(cudaMemsetAsync(fw.dM, 0, localElems * sizeof(unsigned int), fw.sComp));
    }

    fw.run();

    if (localElems > 0) {
        decodePathKernel<<<1024, 256, 0, fw.sComp>>>(fw.dM, N, fw.localRows, gRow0);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(hDist, fw.dD, localElems * sizeof(unsigned int),
                                   cudaMemcpyDeviceToHost, fw.sComp));
        CUDA_CHECK(cudaMemcpyAsync(hPath, fw.dM, localElems * sizeof(unsigned int),
                                   cudaMemcpyDeviceToHost, fw.sComp));
    }
    CUDA_CHECK(cudaStreamSynchronize(fw.sComp));
    MPI_Barrier(MPI_COMM_WORLD);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Assemble the full distance matrix on rank 0 (compact rows, padding dropped).
    // The path matrix stays distributed in hPath (rows [gRow0, gRow1)).
    std::vector<unsigned int> localDist(realRows * numNodes);
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < realRows; ++i) {
        memcpy(&localDist[i * numNodes], hDist + i * N, numNodes * sizeof(unsigned int));
    }
    std::vector<unsigned int> dist;
    {
        MPI_Datatype rowType;
        MPI_Type_contiguous((int)std::max<size_t>(numNodes, 1), MPI_UNSIGNED, &rowType);
        MPI_Type_commit(&rowType);
        std::vector<int> counts(nranks), displs(nranks);
        const int myRows = (int)realRows;
        MPI_Gather(&myRows, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            for (int r = 0, acc = 0; r < nranks; ++r) { displs[r] = acc; acc += counts[r]; }
            dist.resize(numNodes * numNodes);
        }
        MPI_Gatherv(localDist.data(), myRows, rowType, dist.data(), counts.data(), displs.data(),
                    rowType, 0, MPI_COMM_WORLD);
        MPI_Type_free(&rowType);
    }

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
        fflush(stdout);
    }
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);

    for (int b = 0; b < 2; ++b) {
        cudaFree(fw.dPanel[b]);
        cudaEventDestroy(fw.evPanelReady[b]);
        cudaEventDestroy(fw.evPanelFree[b]);
        cudaEventDestroy(fw.evH2DDone[b]);
        cudaEventDestroy(fw.evD2HDone[b]);
    }
    fw.freeComm();
    MPI_Comm_free(&nodeComm);
    cudaEventDestroy(fw.evPanelComputed);
    cudaStreamDestroy(fw.sComp);
    cudaStreamDestroy(fw.sCopy);
    cudaFree(fw.dD);
    cudaFree(fw.dM);
    cudaFreeHost(hDist);
    cudaFreeHost(hPath);

    MPI_Finalize();
    return exitCode;
}
