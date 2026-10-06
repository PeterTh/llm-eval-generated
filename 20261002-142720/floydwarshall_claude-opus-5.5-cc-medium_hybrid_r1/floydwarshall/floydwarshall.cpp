// Hybrid MPI + OpenMP + CUDA blocked Floyd-Warshall.
//
// Decomposition: the (padded) distance/path matrices are split into
// block-rows of TILE rows; every MPI rank owns a contiguous range of
// block-rows and keeps them resident on its own GPU.  For each pivot block k:
//   - the owner of block-row k has already finished the diagonal tile
//     (phase 1) and the pivot row-panel (phase 2, row part) and broadcast the
//     panel (TILE x N) to all ranks,
//   - every rank updates its pivot-column tiles (phase 2, column part),
//   - the owner of block-row k+1 first updates that block-row (phase 3),
//     computes phase 1/2 for pivot k+1 and broadcasts it (lookahead) while
//     its GPU continues with the remaining phase-3 tiles,
//   - every rank updates all remaining tiles (phase 3).
// OpenMP is used on the host for distributed graph initialization and
// validation.

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

// Index calculation for flattened 2D array
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), \
                    __FILE__, __LINE__);                                          \
            MPI_Abort(MPI_COMM_WORLD, 1);                                         \
        }                                                                         \
    } while (0)

// ---------------------------------------------------------------------------
// Graph initialization
// ---------------------------------------------------------------------------

// glibc rand_r(): three steps of a 32-bit LCG per call.
constexpr unsigned int LCG_A = 1103515245u;
constexpr unsigned int LCG_C = 12345u;

static inline int randStep(unsigned int& next) {
    int result;
    next = next * LCG_A + LCG_C;
    result = (unsigned int)(next / 65536) % 2048;
    next = next * LCG_A + LCG_C;
    result <<= 10;
    result ^= (unsigned int)(next / 65536) % 1024;
    next = next * LCG_A + LCG_C;
    result <<= 10;
    result ^= (unsigned int)(next / 65536) % 1024;
    return result;
}

// Advance the LCG state by 'steps' single steps in O(log steps).
static unsigned int lcgSkip(unsigned int state, unsigned long long steps) {
    unsigned int accA = 1, accC = 0;
    unsigned int curA = LCG_A, curC = LCG_C;
    while (steps) {
        if (steps & 1) {
            accA = accA * curA;
            accC = accC * curA + curC;
        }
        curC = curC * curA + curC;
        curA = curA * curA;
        steps >>= 1;
    }
    return accA * state + accC;
}

// Check that the jump-ahead generator reproduces this platform's rand_r.
static bool jumpAheadMatchesRandR() {
    unsigned int s1 = 42, s2 = 42;
    for (int i = 0; i < 1000; ++i) {
        if (rand_r(&s1) != randStep(s2)) return false;
    }
    unsigned int s3 = lcgSkip(42, 3ull * 1000);
    return s3 == s2;
}

static inline unsigned int distValue(int r, const unsigned int rangeMin, const double range) {
    return rangeMin + (unsigned int)(range * r / (double)RAND_MAX);
}

// Generate global rows [row0, row1) of the n x n distance matrix exactly as
// the sequential rand_r stream would (row-major), in parallel with OpenMP.
void initializeDistanceRows(unsigned int* out, const size_t numNodes, const size_t row0,
                            const size_t row1, const unsigned int rangeMin,
                            const unsigned int rangeMax) {
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;
    const size_t begin = row0 * numNodes;
    const size_t count = (row1 - row0) * numNodes;

#pragma omp parallel
    {
        const size_t nt = omp_get_num_threads();
        const size_t t = omp_get_thread_num();
        const size_t lo = count * t / nt;
        const size_t hi = count * (t + 1) / nt;
        unsigned int seed = lcgSkip(42u, 3ull * (begin + lo));
        for (size_t e = lo; e < hi; ++e) {
            out[e] = distValue(randStep(seed), rangeMin, range);
        }
    }

    // Set diagonal to zero (distance from node to itself is 0)
    for (size_t i = row0; i < row1; ++i) {
        out[(i - row0) * numNodes + i] = 0;
    }
}

// Sequential reference initialization (fallback if rand_r is not glibc's).
void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes,
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;

    for (size_t i = 0; i < numNodes * numNodes; ++i) {
        dist[i] = rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
    }

    for (size_t i = 0; i < numNodes; ++i) {
        dist[idx2(i, i, numNodes)] = 0;
    }
}

// ---------------------------------------------------------------------------
// CUDA kernels (row-major local storage, leading dimension ld = padded N)
// ---------------------------------------------------------------------------

constexpr int TILE = 64;   // tile edge
constexpr int TDIM = 16;   // threads per tile edge (16x16 threads, 4x4 elements each)
constexpr int SPAD = TILE + 1;

// Initialize padded local block-rows: real entries are uploaded later; padding
// entries get INF (0 on the diagonal). Path matrix: path[i][j] = i (matches
// the original initializePathMatrix).
__global__ void initLocalKernel(unsigned int* __restrict__ d, unsigned int* __restrict__ p,
                                size_t ld, size_t localRows, size_t globalRow0) {
    const size_t total = localRows * ld;
    for (size_t e = blockIdx.x * (size_t)blockDim.x + threadIdx.x; e < total;
         e += (size_t)gridDim.x * blockDim.x) {
        const size_t lr = e / ld;
        const size_t c = e - lr * ld;
        const size_t gr = globalRow0 + lr;
        d[e] = (gr == c) ? 0u : INF;
        p[e] = (unsigned int)gr;
    }
}

// Load a TILE x TILE tile into padded shared memory (thread owns rows 4ty..,
// cols 4tx..).
__device__ __forceinline__ void loadTileShared(unsigned int (*S)[SPAD],
                                               const unsigned int* __restrict__ src, size_t ld) {
    const int tx = threadIdx.x, ty = threadIdx.y;
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int row = ty * 4 + r;
        const uint4 v = *reinterpret_cast<const uint4*>(src + row * ld + tx * 4);
        S[row][tx * 4 + 0] = v.x;
        S[row][tx * 4 + 1] = v.y;
        S[row][tx * 4 + 2] = v.z;
        S[row][tx * 4 + 3] = v.w;
    }
}

__device__ __forceinline__ void storeTile(unsigned int* __restrict__ dst, size_t ld,
                                          const unsigned int (&v)[4][4]) {
    const int tx = threadIdx.x, ty = threadIdx.y;
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        *reinterpret_cast<uint4*>(dst + (ty * 4 + r) * ld + tx * 4) =
            make_uint4(v[r][0], v[r][1], v[r][2], v[r][3]);
    }
}

__device__ __forceinline__ void loadTileRegs(unsigned int (&v)[4][4],
                                             const unsigned int* __restrict__ src, size_t ld) {
    const int tx = threadIdx.x, ty = threadIdx.y;
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const uint4 q = *reinterpret_cast<const uint4*>(src + (ty * 4 + r) * ld + tx * 4);
        v[r][0] = q.x; v[r][1] = q.y; v[r][2] = q.z; v[r][3] = q.w;
    }
}

// Phase 1: diagonal tile, fully dependent within the tile.
__global__ void __launch_bounds__(TDIM * TDIM)
phase1Kernel(unsigned int* __restrict__ d, unsigned int* __restrict__ p, size_t ld,
             size_t localRow0, size_t col0) {
    __shared__ unsigned int S[TILE][SPAD];
    const int tx = threadIdx.x, ty = threadIdx.y;
    unsigned int* dt = d + localRow0 * ld + col0;
    unsigned int* pt = p + localRow0 * ld + col0;

    loadTileShared(S, dt, ld);
    unsigned int pv[4][4];
    loadTileRegs(pv, pt, ld);
    __syncthreads();

    for (int kk = 0; kk < TILE; ++kk) {
        unsigned int a[4], b[4];
#pragma unroll
        for (int r = 0; r < 4; ++r) a[r] = S[ty * 4 + r][kk];
#pragma unroll
        for (int c = 0; c < 4; ++c) b[c] = S[kk][tx * 4 + c];
#pragma unroll
        for (int r = 0; r < 4; ++r)
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                const unsigned int s = a[r] + b[c];
                if (s < S[ty * 4 + r][tx * 4 + c]) {
                    S[ty * 4 + r][tx * 4 + c] = s;
                    pv[r][c] = (unsigned int)col0 + kk;
                }
            }
        __syncthreads();
    }

    unsigned int dv[4][4];
#pragma unroll
    for (int r = 0; r < 4; ++r)
#pragma unroll
        for (int c = 0; c < 4; ++c) dv[r][c] = S[ty * 4 + r][tx * 4 + c];
    storeTile(dt, ld, dv);
    storeTile(pt, ld, pv);
}

// Phase 2 (row part): tiles (k, j), j != k, in the pivot block-row owned locally.
// new[i][j] = diag[i][kk] + own[kk][j]
__global__ void __launch_bounds__(TDIM * TDIM)
phase2RowKernel(unsigned int* __restrict__ d, unsigned int* __restrict__ p, size_t ld,
                size_t localRow0, int kb) {
    const int jb = blockIdx.x;
    if (jb == kb) return;
    __shared__ unsigned int D[TILE][SPAD];
    __shared__ unsigned int O[TILE][SPAD];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const size_t col0 = (size_t)kb * TILE;
    unsigned int* ot = d + localRow0 * ld + (size_t)jb * TILE;
    unsigned int* pt = p + localRow0 * ld + (size_t)jb * TILE;

    loadTileShared(D, d + localRow0 * ld + col0, ld);
    loadTileShared(O, ot, ld);
    unsigned int pv[4][4];
    loadTileRegs(pv, pt, ld);
    __syncthreads();

    for (int kk = 0; kk < TILE; ++kk) {
        unsigned int a[4], b[4];
#pragma unroll
        for (int r = 0; r < 4; ++r) a[r] = D[ty * 4 + r][kk];
#pragma unroll
        for (int c = 0; c < 4; ++c) b[c] = O[kk][tx * 4 + c];
#pragma unroll
        for (int r = 0; r < 4; ++r)
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                const unsigned int s = a[r] + b[c];
                if (s < O[ty * 4 + r][tx * 4 + c]) {
                    O[ty * 4 + r][tx * 4 + c] = s;
                    pv[r][c] = (unsigned int)col0 + kk;
                }
            }
        __syncthreads();
    }

    unsigned int dv[4][4];
#pragma unroll
    for (int r = 0; r < 4; ++r)
#pragma unroll
        for (int c = 0; c < 4; ++c) dv[r][c] = O[ty * 4 + r][tx * 4 + c];
    storeTile(ot, ld, dv);
    storeTile(pt, ld, pv);
}

// Phase 2 (column part): tiles (i, k) for all local block-rows except the
// pivot one. new[i][j] = own[i][kk] + diag[kk][j], diag taken from the panel.
__global__ void __launch_bounds__(TDIM * TDIM)
phase2ColKernel(unsigned int* __restrict__ d, unsigned int* __restrict__ p, size_t ld,
                const unsigned int* __restrict__ panel, int kb, int skipLocal) {
    const int ib = blockIdx.x;
    if (ib == skipLocal) return;
    __shared__ unsigned int D[TILE][SPAD];
    __shared__ unsigned int O[TILE][SPAD];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const size_t col0 = (size_t)kb * TILE;
    unsigned int* ot = d + (size_t)ib * TILE * ld + col0;
    unsigned int* pt = p + (size_t)ib * TILE * ld + col0;

    loadTileShared(D, panel + col0, ld);
    loadTileShared(O, ot, ld);
    unsigned int pv[4][4];
    loadTileRegs(pv, pt, ld);
    __syncthreads();

    for (int kk = 0; kk < TILE; ++kk) {
        unsigned int a[4], b[4];
#pragma unroll
        for (int r = 0; r < 4; ++r) a[r] = O[ty * 4 + r][kk];
#pragma unroll
        for (int c = 0; c < 4; ++c) b[c] = D[kk][tx * 4 + c];
#pragma unroll
        for (int r = 0; r < 4; ++r)
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                const unsigned int s = a[r] + b[c];
                if (s < O[ty * 4 + r][tx * 4 + c]) {
                    O[ty * 4 + r][tx * 4 + c] = s;
                    pv[r][c] = (unsigned int)col0 + kk;
                }
            }
        __syncthreads();
    }

    unsigned int dv[4][4];
#pragma unroll
    for (int r = 0; r < 4; ++r)
#pragma unroll
        for (int c = 0; c < 4; ++c) dv[r][c] = O[ty * 4 + r][tx * 4 + c];
    storeTile(ot, ld, dv);
    storeTile(pt, ld, pv);
}

// Phase 3: independent tiles (i, j), i != k, j != k.
// new[i][j] = col[i][kk] + row[kk][j]; col tile is local, row tile from panel.
__global__ void __launch_bounds__(TDIM * TDIM)
phase3Kernel(unsigned int* __restrict__ d, unsigned int* __restrict__ p, size_t ld,
             const unsigned int* __restrict__ panel, int kb, int localTileBase,
             int skipA, int skipB) {
    const int jb = blockIdx.x;
    const int ib = localTileBase + blockIdx.y;
    if (jb == kb || ib == skipA || ib == skipB) return;

    __shared__ __align__(16) unsigned int Ct[TILE][TILE];  // transposed: Ct[kk][i]
    __shared__ __align__(16) unsigned int R[TILE][TILE];   // R[kk][j]
    const int tx = threadIdx.x, ty = threadIdx.y;
    const size_t col0 = (size_t)kb * TILE;
    const unsigned int* ct = d + (size_t)ib * TILE * ld + col0;
    const unsigned int* rt = panel + (size_t)jb * TILE;
    unsigned int* ot = d + (size_t)ib * TILE * ld + (size_t)jb * TILE;
    unsigned int* pt = p + (size_t)ib * TILE * ld + (size_t)jb * TILE;

#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int row = ty * 4 + r;
        const uint4 cv = *reinterpret_cast<const uint4*>(ct + row * ld + tx * 4);
        Ct[tx * 4 + 0][row] = cv.x;
        Ct[tx * 4 + 1][row] = cv.y;
        Ct[tx * 4 + 2][row] = cv.z;
        Ct[tx * 4 + 3][row] = cv.w;
        *reinterpret_cast<uint4*>(&R[row][tx * 4]) =
            *reinterpret_cast<const uint4*>(rt + row * ld + tx * 4);
    }
    unsigned int dv[4][4], pv[4][4];
    loadTileRegs(dv, ot, ld);
    loadTileRegs(pv, pt, ld);
    __syncthreads();

#pragma unroll 16
    for (int kk = 0; kk < TILE; ++kk) {
        const uint4 a4 = *reinterpret_cast<const uint4*>(&Ct[kk][ty * 4]);
        const uint4 b4 = *reinterpret_cast<const uint4*>(&R[kk][tx * 4]);
        const unsigned int a[4] = {a4.x, a4.y, a4.z, a4.w};
        const unsigned int b[4] = {b4.x, b4.y, b4.z, b4.w};
        const unsigned int kidx = (unsigned int)col0 + kk;
#pragma unroll
        for (int r = 0; r < 4; ++r)
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                const unsigned int s = a[r] + b[c];
                if (s < dv[r][c]) {
                    dv[r][c] = s;
                    pv[r][c] = kidx;
                }
            }
    }

    storeTile(ot, ld, dv);
    storeTile(pt, ld, pv);
}

// ---------------------------------------------------------------------------
// Distributed driver
// ---------------------------------------------------------------------------

struct Layout {
    size_t n = 0;         // real nodes
    size_t N = 0;         // padded nodes (multiple of TILE)
    int numTiles = 0;     // N / TILE
    std::vector<int> tileStart;  // per-rank first block-row (size nprocs + 1)
    std::vector<int> owner;      // block-row -> rank
};

static Layout makeLayout(size_t n, int nprocs) {
    Layout L;
    L.n = n;
    L.N = ((n + TILE - 1) / TILE) * TILE;
    L.numTiles = (int)(L.N / TILE);
    L.tileStart.resize(nprocs + 1);
    for (int r = 0; r <= nprocs; ++r) {
        L.tileStart[r] = (int)((long long)L.numTiles * r / nprocs);
    }
    L.owner.resize(L.numTiles);
    for (int r = 0; r < nprocs; ++r) {
        for (int t = L.tileStart[r]; t < L.tileStart[r + 1]; ++t) L.owner[t] = r;
    }
    return L;
}

// Runs blocked Floyd-Warshall on the locally owned block-rows held on the GPU.
static void floydWarshallDistributed(unsigned int* d, unsigned int* p, unsigned int* panel,
                                     unsigned int* hostPanel[2], cudaEvent_t panelEvent[2],
                                     const Layout& L, int rank, int nprocs,
                                     cudaStream_t stream) {
    const size_t ld = L.N;
    const int t0 = L.tileStart[rank];
    const int localTiles = L.tileStart[rank + 1] - t0;
    const size_t panelElems = (size_t)TILE * ld;
    const dim3 block(TDIM, TDIM);
    const bool multi = nprocs > 1;
    cudaEvent_t d2hDone;
    CUDA_CHECK(cudaEventCreateWithFlags(&d2hDone, cudaEventDisableTiming));

    auto localRow0 = [&](int tile) { return (size_t)(tile - t0) * TILE; };

    // Compute phase 1 and phase-2-row for pivot kb (caller owns kb).
    auto pivotPanel = [&](int kb) {
        phase1Kernel<<<1, block, 0, stream>>>(d, p, ld, localRow0(kb), (size_t)kb * TILE);
        if (L.numTiles > 1) {
            phase2RowKernel<<<L.numTiles, block, 0, stream>>>(d, p, ld, localRow0(kb), kb);
        }
    };

    // Distribute pivot panel kb into 'panel' on every rank. If 'afterLaunch'
    // is given the owner enqueues it between the D2H copy and the broadcast,
    // so it overlaps with communication.
    int buf = 0;
    auto sharePanel = [&](int kb, auto&& afterLaunch) {
        const int root = L.owner[kb];
        unsigned int* hp = hostPanel[buf];
        // The host buffer may still be read by an earlier asynchronous H2D copy.
        CUDA_CHECK(cudaEventSynchronize(panelEvent[buf]));
        if (root == rank) {
            const unsigned int* src = d + localRow0(kb) * ld;
            if (multi) {
                CUDA_CHECK(cudaMemcpyAsync(hp, src, panelElems * sizeof(unsigned int),
                                           cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaEventRecord(d2hDone, stream));
            }
            afterLaunch();
            CUDA_CHECK(cudaMemcpyAsync(panel, src, panelElems * sizeof(unsigned int),
                                       cudaMemcpyDeviceToDevice, stream));
            if (multi) {
                CUDA_CHECK(cudaEventSynchronize(d2hDone));
                MPI_Bcast(hp, (int)panelElems, MPI_UNSIGNED, root, MPI_COMM_WORLD);
            }
        } else {
            afterLaunch();
            MPI_Bcast(hp, (int)panelElems, MPI_UNSIGNED, root, MPI_COMM_WORLD);
            CUDA_CHECK(cudaMemcpyAsync(panel, hp, panelElems * sizeof(unsigned int),
                                       cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaEventRecord(panelEvent[buf], stream));
        }
        buf ^= 1;
    };

    auto launchPhase3 = [&](int kb, int tileBase, int tileCount, int skipA, int skipB) {
        if (tileCount <= 0) return;
        const dim3 grid(L.numTiles, tileCount);
        phase3Kernel<<<grid, block, 0, stream>>>(d, p, ld, panel, kb, tileBase, skipA, skipB);
    };

    // Prologue: pivot 0.
    if (L.owner[0] == rank) pivotPanel(0);
    sharePanel(0, [] {});

    for (int kb = 0; kb < L.numTiles; ++kb) {
        const int skipK = (L.owner[kb] == rank) ? kb - t0 : -1;
        if (localTiles > 0) {
            phase2ColKernel<<<localTiles, block, 0, stream>>>(d, p, ld, panel, kb, skipK);
        }

        const int next = kb + 1;
        if (next < L.numTiles && L.owner[next] == rank) {
            const int nl = next - t0;
            // Lookahead: finish block-row next, then build pivot panel next.
            launchPhase3(kb, nl, 1, skipK, -1);
            pivotPanel(next);
            sharePanel(next, [&] { launchPhase3(kb, 0, localTiles, skipK, nl); });
        } else {
            launchPhase3(kb, 0, localTiles, skipK, -1);
            if (next < L.numTiles) sharePanel(next, [] {});
        }
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaEventDestroy(d2hDone));
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
    for (size_t i = 0; i < lim; ++i) {
        for (size_t j = 0; j < lim; ++j) {
            const unsigned int distIJ = dist[idx2(j, i, numNodes)];
            size_t firstBad = numNodes;
#pragma omp parallel for reduction(min : firstBad)
            for (size_t k = 0; k < numNodes; ++k) {
                const unsigned int distIK = dist[idx2(k, i, numNodes)];
                const unsigned int distKJ = dist[idx2(j, k, numNodes)];

                // Check for overflow before addition
                if (distIK < INF && distKJ < INF && distIK + distKJ < distIJ) {
                    firstBad = std::min(firstBad, k);
                }
            }
            if (firstBad < numNodes) {
                printf("Validation failed: triangle inequality violated at [%zu,%zu,%zu]\n",
                       i, j, firstBad);
                return false;
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
    int rank = 0, nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

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

    // Bind one GPU per rank on each node.
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_free(&nodeComm);
    int numDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));
    if (numDevices < 1) {
        fprintf(stderr, "No CUDA device available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % numDevices));

    if (numNodes == 0) {
        if (rank == 0) printf("Initializing graph...\nComputing shortest paths...\n");
        MPI_Finalize();
        return 0;
    }

    const Layout L = makeLayout(numNodes, nprocs);
    const size_t n = L.n, ld = L.N;
    const int t0 = L.tileStart[rank];
    const size_t localRows = (size_t)(L.tileStart[rank + 1] - t0) * TILE;
    const size_t gRow0 = (size_t)t0 * TILE;
    const size_t realRow0 = std::min(gRow0, n);
    const size_t realRow1 = std::min(gRow0 + localRows, n);
    const size_t realRows = realRow1 - realRow0;

    // Per-rank row counts/displacements for the real n x n matrix (in rows).
    std::vector<int> rowCounts(nprocs), rowDispls(nprocs);
    for (int r = 0; r < nprocs; ++r) {
        const size_t a = std::min((size_t)L.tileStart[r] * TILE, n);
        const size_t b = std::min((size_t)L.tileStart[r + 1] * TILE, n);
        rowCounts[r] = (int)(b - a);
        rowDispls[r] = (int)a;
    }
    MPI_Datatype rowType;
    MPI_Type_contiguous((int)n, MPI_UNSIGNED, &rowType);
    MPI_Type_commit(&rowType);

    // Initialize: every rank generates its own rows of the matrix in parallel.
    if (rank == 0) printf("Initializing graph...\n");
    unsigned int* hostRows = nullptr;
    CUDA_CHECK(cudaMallocHost(&hostRows, std::max<size_t>(realRows * n, 1) * sizeof(unsigned int)));
    std::vector<unsigned int> dist;
    if (rank == 0) dist.resize(n * n);

    if (jumpAheadMatchesRandR()) {
        initializeDistanceRows(hostRows, n, realRow0, realRow1, 1, MAX_DISTANCE);
    } else {
        if (rank == 0) initializeDistanceMatrix(dist, n, 1, MAX_DISTANCE);
        MPI_Scatterv(rank == 0 ? dist.data() : nullptr, rowCounts.data(), rowDispls.data(),
                     rowType, hostRows, (int)realRows, rowType, 0, MPI_COMM_WORLD);
    }

    // Device buffers.
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    unsigned int *dDist = nullptr, *dPath = nullptr, *dPanel = nullptr;
    const size_t localElems = std::max<size_t>(localRows * ld, 1);
    CUDA_CHECK(cudaMalloc(&dDist, localElems * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&dPath, localElems * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&dPanel, (size_t)TILE * ld * sizeof(unsigned int)));
    unsigned int* hostPanel[2];
    cudaEvent_t panelEvent[2];
    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaMallocHost(&hostPanel[b], (size_t)TILE * ld * sizeof(unsigned int)));
        CUDA_CHECK(cudaEventCreateWithFlags(&panelEvent[b], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventRecord(panelEvent[b], stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Run Floyd-Warshall
    if (rank == 0) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    if (localRows > 0) {
        initLocalKernel<<<1024, 256, 0, stream>>>(dDist, dPath, ld, localRows, gRow0);
        if (realRows > 0) {
            CUDA_CHECK(cudaMemcpy2DAsync(dDist, ld * sizeof(unsigned int), hostRows,
                                         n * sizeof(unsigned int), n * sizeof(unsigned int),
                                         realRows, cudaMemcpyHostToDevice, stream));
        }
    }

    floydWarshallDistributed(dDist, dPath, dPanel, hostPanel, panelEvent, L, rank, nprocs,
                             stream);

    if (realRows > 0) {
        CUDA_CHECK(cudaMemcpy2DAsync(hostRows, n * sizeof(unsigned int), dDist,
                                     ld * sizeof(unsigned int), n * sizeof(unsigned int),
                                     realRows, cudaMemcpyDeviceToHost, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    MPI_Barrier(MPI_COMM_WORLD);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Collect the distance matrix on rank 0 when it is needed.
    if (printResults || validate) {
        MPI_Gatherv(hostRows, (int)realRows, rowType, rank == 0 ? dist.data() : nullptr,
                    rowCounts.data(), rowDispls.data(), rowType, 0, MPI_COMM_WORLD);
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
    }
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);

    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaFreeHost(hostPanel[b]));
        CUDA_CHECK(cudaEventDestroy(panelEvent[b]));
    }
    CUDA_CHECK(cudaFree(dDist));
    CUDA_CHECK(cudaFree(dPath));
    CUDA_CHECK(cudaFree(dPanel));
    CUDA_CHECK(cudaFreeHost(hostRows));
    CUDA_CHECK(cudaStreamDestroy(stream));
    MPI_Type_free(&rowType);
    MPI_Finalize();
    return exitCode;
}
