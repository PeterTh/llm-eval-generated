// Hybrid MPI + OpenMP + CUDA Floyd-Warshall all-pairs shortest path benchmark.
//
// Decomposition
//   * MPI:    the NB x NB grid of TILE x TILE tiles is distributed block-cyclically
//             over a P x Q process grid. Tile (I,J) belongs to rank (I%P, J%Q).
//             Each rank drives one GPU (local rank modulo device count).
//   * CUDA:   blocked Floyd-Warshall. For each block index K the pivot tile (K,K),
//             the row/column panels (K,J)/(I,K) are processed first, then the tiles
//             of row/column K+1 (needed by the next block) and finally all remaining
//             tiles are relaxed with the TILE pivots of block K fused in registers.
//             Work is pipelined: the bulk update of block K runs on a second stream
//             while the host exchanges block K+1's pivot data over MPI.
//   * OpenMP: host-side data-parallel loops (tile initialization, result gathering).
//
// Equivalence with the sequential k-outer algorithm
//   Every tile receives the same sequence of relaxations as the original code, in
//   ascending k order: the pivot phases inside a block replay k in order, and each
//   tile is updated by block K only after block K-1 has been applied to it (ordered
//   by stream/event dependencies). For every phase the pivot column and row are
//   captured exactly as they are at the start of that phase (snapshots), so
//   distances and path entries (strict '<' tie-breaking) are identical. Padding
//   (n rounded up to a multiple of TILE) only holds INF rows/columns whose pivots
//   are skipped, so they never influence real entries.

#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

constexpr int TILE = 64;           // tile edge length (pivots fused per block)
constexpr int TT = TILE * TILE;    // elements per tile
constexpr int THREADS = 256;       // 16 x 16 threads per CUDA block
constexpr int RPT = TILE / 16;     // elements per thread per dimension
static_assert(RPT * 16 == TILE, "TILE must be a multiple of 16");

#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        cudaError_t err_ = (call);                                                       \
        if (err_ != cudaSuccess) {                                                       \
            fprintf(stderr, "CUDA error '%s' at %s:%d: %s\n", cudaGetErrorString(err_),  \
                    __FILE__, __LINE__, #call);                                          \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                \
        }                                                                                \
    } while (0)

// Index calculation for flattened 2D array
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

// ---------------------------------------------------------------------------
// Device kernels
// ---------------------------------------------------------------------------

enum PanelMode { PANEL_DIAG = 0, PANEL_COL = 1, PANEL_ROW = 2 };

// Processes the pivot phases of block K inside one tile per CUDA block.
//   PANEL_DIAG: tile (K,K), in place. snapOut receives
//               diagRow[kp][y] = D(K.kp, K.y) and diagCol[kp][x] = D(K.x, K.kp)
//               as they are at the start of phase kp (2*TT entries).
//   PANEL_COL:  tile (I,K), I != K, local tile row blockIdx.x. piv is the diag
//               snapshot; snapOut receives colSnap[kp][x] = D(I.x, K.kp).
//   PANEL_ROW:  tile (K,J), J != K, local tile column blockIdx.x. piv is the diag
//               snapshot; snapOut receives rowSnap[kp][y] = D(K.kp, J.y).
// gStart/gStep map the block index to the global tile index (used to skip the
// pivot tile itself).
template <int MODE>
__global__ void __launch_bounds__(THREADS)
panelKernel(unsigned int* __restrict__ gDist, unsigned int* __restrict__ gPath,
            const size_t blockStride, const int gStart, const int gStep, const int K,
            const int kBase, const int kmax, const unsigned int* __restrict__ piv,
            unsigned int* __restrict__ snapOut) {
    if (MODE != PANEL_DIAG && gStart + static_cast<int>(blockIdx.x) * gStep == K) {
        return;  // the pivot tile itself is handled by PANEL_DIAG
    }

    unsigned int* D = gDist + static_cast<size_t>(blockIdx.x) * blockStride;
    unsigned int* P = gPath + static_cast<size_t>(blockIdx.x) * blockStride;
    // PANEL_DIAG runs as a single block and writes both diagonal snapshots (2*TT).
    unsigned int* S = snapOut + static_cast<size_t>(blockIdx.x) * TT;

    __shared__ unsigned int sD[TT];   // distances, storage index yl*TILE + xl
    __shared__ unsigned int sP[TT];   // path
    __shared__ unsigned int pc[TILE]; // pivot column for the current phase
    __shared__ unsigned int pr[TILE]; // pivot row for the current phase

    const int tid = threadIdx.x;
    const int tx = tid & 15;
    const int ty = tid >> 4;

    for (int t = tid; t < TT; t += THREADS) {
        sD[t] = D[t];
        sP[t] = P[t];
    }
    __syncthreads();

    for (int kp = 0; kp < kmax; ++kp) {
        __syncthreads();  // previous phase fully applied
        if (tid < TILE) {
            if (MODE == PANEL_DIAG) {
                pc[tid] = sD[kp * TILE + tid];   // D(x=tid, kp)
                pr[tid] = sD[tid * TILE + kp];   // D(kp, y=tid)
            } else if (MODE == PANEL_COL) {
                pc[tid] = sD[kp * TILE + tid];   // D(x=tid, kp), local column kp
                pr[tid] = piv[kp * TILE + tid];  // diagRow[kp][y]
            } else {
                pr[tid] = sD[tid * TILE + kp];   // D(kp, y=tid), local row kp
                pc[tid] = piv[TT + kp * TILE + tid];  // diagCol[kp][x]
            }
        }
        __syncthreads();

        if (tid < TILE) {
            if (MODE == PANEL_DIAG) {
                S[kp * TILE + tid] = pr[tid];
                S[TT + kp * TILE + tid] = pc[tid];
            } else if (MODE == PANEL_COL) {
                S[kp * TILE + tid] = pc[tid];
            } else {
                S[kp * TILE + tid] = pr[tid];
            }
        }

        const unsigned int kg = static_cast<unsigned int>(kBase + kp);
#pragma unroll
        for (int a = 0; a < RPT; ++a) {
#pragma unroll
            for (int b = 0; b < RPT; ++b) {
                const int x = tx + 16 * a;
                const int y = ty + 16 * b;
                const int idx = y * TILE + x;
                const unsigned int nd = pc[x] + pr[y];
                if (nd < sD[idx]) {
                    sD[idx] = nd;
                    sP[idx] = kg;
                }
            }
        }
    }
    __syncthreads();

    for (int t = tid; t < TT; t += THREADS) {
        D[t] = sD[t];
        P[t] = sP[t];
    }
}

// Relaxes local tiles (I,J) with I != K and J != K using the kmax pivots of block K.
// part == UPDATE_CRITICAL: only tiles of row K+1 or column K+1 (needed by block K+1).
// part == UPDATE_BULK:     all remaining tiles of block K.
// colSnap[li*TT + kp*TILE + x] = D(I.x, K.kp), rowSnap[lj*TT + kp*TILE + y] = D(K.kp, J.y).
// Grid: (NLJ, NLI) local tiles.
enum UpdatePart { UPDATE_CRITICAL = 0, UPDATE_BULK = 1 };

__global__ void __launch_bounds__(THREADS)
updateKernel(unsigned int* __restrict__ gDist, unsigned int* __restrict__ gPath,
             const int NLJ, const int r, const int c, const int P, const int Q, const int K,
             const int kBase, const int kmax, const int part,
             const unsigned int* __restrict__ colSnap,
             const unsigned int* __restrict__ rowSnap) {
    const int lj = blockIdx.x;
    const int li = blockIdx.y;
    const int I = r + li * P;
    const int J = c + lj * Q;
    if (I == K || J == K) return;
    const bool critical = (I == K + 1 || J == K + 1);
    if (critical != (part == UPDATE_CRITICAL)) return;

    __shared__ unsigned int sCol[TT];  // sCol[kp*TILE + x] = D(I.x, K.kp)
    __shared__ unsigned int sRow[TT];  // sRow[kp*TILE + y] = D(K.kp, J.y)

    const int tid = threadIdx.x;
    const int tx = tid & 15;
    const int ty = tid >> 4;

    const unsigned int* cs = colSnap + static_cast<size_t>(li) * TT;
    const unsigned int* rs = rowSnap + static_cast<size_t>(lj) * TT;
    for (int t = tid; t < kmax * TILE; t += THREADS) {
        sCol[t] = cs[t];
        sRow[t] = rs[t];
    }

    const size_t base = (static_cast<size_t>(li) * NLJ + lj) * TT;
    unsigned int* D = gDist + base;
    unsigned int* Pa = gPath + base;

    unsigned int d[RPT][RPT];
    unsigned int p[RPT][RPT];
#pragma unroll
    for (int a = 0; a < RPT; ++a) {
#pragma unroll
        for (int b = 0; b < RPT; ++b) {
            const int idx = (ty + 16 * b) * TILE + (tx + 16 * a);
            d[a][b] = D[idx];
            p[a][b] = Pa[idx];
        }
    }
    __syncthreads();

    for (int kp = 0; kp < kmax; ++kp) {
        unsigned int cv[RPT];
        unsigned int rv[RPT];
#pragma unroll
        for (int a = 0; a < RPT; ++a) cv[a] = sCol[kp * TILE + tx + 16 * a];
#pragma unroll
        for (int b = 0; b < RPT; ++b) rv[b] = sRow[kp * TILE + ty + 16 * b];

        const unsigned int kg = static_cast<unsigned int>(kBase + kp);
#pragma unroll
        for (int a = 0; a < RPT; ++a) {
#pragma unroll
            for (int b = 0; b < RPT; ++b) {
                const unsigned int nd = cv[a] + rv[b];
                if (nd < d[a][b]) {
                    d[a][b] = nd;
                    p[a][b] = kg;
                }
            }
        }
    }

#pragma unroll
    for (int a = 0; a < RPT; ++a) {
#pragma unroll
        for (int b = 0; b < RPT; ++b) {
            const int idx = (ty + 16 * b) * TILE + (tx + 16 * a);
            D[idx] = d[a][b];
            Pa[idx] = p[a][b];
        }
    }
}

// ---------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------

// Number of indices in [0, numTiles) congruent to `index` modulo `dim`.
static inline int localCount(const int numTiles, const int index, const int dim) {
    return numTiles > index ? (numTiles - index - 1) / dim + 1 : 0;
}

struct Decomposition {
    int P = 1, Q = 1;       // process grid: P process rows, Q process columns
    int r = 0, c = 0;       // this rank's grid coordinates (rank = r*Q + c)
    int NB = 0;             // tiles per matrix dimension
    int NLI = 0, NLJ = 0;   // locally owned tile rows / tile columns
    MPI_Comm rowComm = MPI_COMM_NULL;  // ranks sharing r; rank index equals c
    MPI_Comm colComm = MPI_COMM_NULL;  // ranks sharing c; rank index equals r
};

static Decomposition makeDecomposition(const int rank, const int size, const size_t n) {
    Decomposition d;
    int dims[2] = {0, 0};
    MPI_Dims_create(size, 2, dims);
    d.P = dims[0];
    d.Q = dims[1];
    d.r = rank / d.Q;
    d.c = rank % d.Q;
    d.NB = static_cast<int>((n + TILE - 1) / TILE);
    d.NLI = localCount(d.NB, d.r, d.P);
    d.NLJ = localCount(d.NB, d.c, d.Q);
    MPI_Comm_split(MPI_COMM_WORLD, d.r, d.c, &d.rowComm);
    MPI_Comm_split(MPI_COMM_WORLD, d.c, d.r, &d.colComm);
    return d;
}

static void selectDevice() {
    int numDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));
    if (numDevices == 0) {
        fprintf(stderr, "No CUDA devices available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Comm shm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shm);
    int localRank = 0;
    MPI_Comm_rank(shm, &localRank);
    MPI_Comm_free(&shm);
    CUDA_CHECK(cudaSetDevice(localRank % numDevices));
}

// Fills this rank's tiles. The random values are identical to the original
// initializeDistanceMatrix(): rand_r (seed 42) consumed in storage order
// (element (x,y) at index y*n + x). The path matrix equals the original
// initializePathMatrix(), which yields path(x,y) = y.
static void initializeLocalTiles(const Decomposition& d, const size_t n,
                                 unsigned int* dist, unsigned int* path) {
    const long long numLocal = static_cast<long long>(d.NLI) * d.NLJ;

#pragma omp parallel for schedule(static)
    for (long long t = 0; t < numLocal; ++t) {
        const int li = static_cast<int>(t / d.NLJ);
        const int lj = static_cast<int>(t % d.NLJ);
        const size_t I = static_cast<size_t>(d.r + li * d.P);
        const size_t J = static_cast<size_t>(d.c + lj * d.Q);
        unsigned int* D = dist + static_cast<size_t>(t) * TT;
        unsigned int* Pa = path + static_cast<size_t>(t) * TT;
        for (int yl = 0; yl < TILE; ++yl) {
            for (int xl = 0; xl < TILE; ++xl) {
                const size_t x = I * TILE + xl;
                const size_t y = J * TILE + yl;
                D[yl * TILE + xl] = (x == y) ? 0u : INF;  // padding stays INF
                Pa[yl * TILE + xl] = static_cast<unsigned int>(y);
            }
        }
    }

    unsigned int seed = 42;
    const unsigned int rangeMin = 1;
    const unsigned int rangeMax = MAX_DISTANCE;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;

    // Serial by necessity: the rand_r stream must be consumed in storage order.
    for (size_t y = 0; y < n; ++y) {
        const int J = static_cast<int>(y / TILE);
        const int yl = static_cast<int>(y % TILE);
        const bool colOwned = (J % d.Q) == d.c;
        const int lj = J / d.Q;
        for (size_t x = 0; x < n; ++x) {
            const unsigned int value =
                rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
            const int I = static_cast<int>(x / TILE);
            if (colOwned && (I % d.P) == d.r) {
                const int li = I / d.P;
                const size_t t = static_cast<size_t>(li) * d.NLJ + lj;
                dist[t * TT + yl * TILE + static_cast<int>(x % TILE)] = (x == y) ? 0u : value;
            }
        }
    }
}

// Device and host resources for the pipelined block loop.
struct Pipeline {
    cudaStream_t sMain = nullptr;  // pivot tile, panels, snapshot exchange, tiles of row/col K+1
    cudaStream_t sBulk = nullptr;  // all remaining tiles
    cudaEvent_t evBulk[2] = {nullptr, nullptr};      // bulk of block K finished
    cudaEvent_t evSnap[2] = {nullptr, nullptr};      // device snapshots of block K ready
    cudaEvent_t evHostFree[2] = {nullptr, nullptr};  // host staging of block K no longer in use

    unsigned int* dDiagSnap = nullptr;   // 2*TT, pivot tile snapshot (main stream only)
    unsigned int* dColSnap[2] = {nullptr, nullptr};  // NLI*TT, per block parity
    unsigned int* dRowSnap[2] = {nullptr, nullptr};  // NLJ*TT, per block parity

    unsigned int* hDiagSnap[2] = {nullptr, nullptr};
    unsigned int* hColSnap[2] = {nullptr, nullptr};
    unsigned int* hRowSnap[2] = {nullptr, nullptr};
};

// Runs the blocked Floyd-Warshall over all tiles. The rank's tiles stay on the device;
// only pivot snapshots travel through pinned host staging buffers.
static void runBlockedFloydWarshall(const Decomposition& d, const size_t n,
                                    unsigned int* dDist, unsigned int* dPath, Pipeline& pl) {
    const size_t colElems = static_cast<size_t>(d.NLI) * TT;
    const size_t rowElems = static_cast<size_t>(d.NLJ) * TT;
    const size_t diagBytes = 2 * TT * sizeof(unsigned int);

    for (int K = 0; K < d.NB; ++K) {
        const int b = K & 1;
        const int kBase = K * TILE;
        const int kmax = static_cast<int>(std::min<size_t>(TILE, n - static_cast<size_t>(kBase)));
        const bool ownsRow = (K % d.P) == d.r;  // this rank holds tile row K
        const bool ownsCol = (K % d.Q) == d.c;  // this rank holds tile column K
        const bool ownsDiag = ownsRow && ownsCol;
        const int diagRoot = (K % d.P) * d.Q + (K % d.Q);

        // Host staging of parity b was last used by block K-2; wait until its copies finished.
        if (K >= 2) CUDA_CHECK(cudaEventSynchronize(pl.evHostFree[b]));

        // 1. Pivot tile (K,K) on the owner; broadcast its row/column snapshots.
        if (ownsDiag) {
            const size_t off = (static_cast<size_t>(K / d.P) * d.NLJ + (K / d.Q)) * TT;
            panelKernel<PANEL_DIAG><<<1, THREADS, 0, pl.sMain>>>(
                dDist + off, dPath + off, 0, 0, 1, K, kBase, kmax, nullptr, pl.dDiagSnap);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpyAsync(pl.hDiagSnap[b], pl.dDiagSnap, diagBytes,
                                       cudaMemcpyDeviceToHost, pl.sMain));
            CUDA_CHECK(cudaStreamSynchronize(pl.sMain));
        }
        MPI_Bcast(pl.hDiagSnap[b], 2 * TT, MPI_UNSIGNED, diagRoot, MPI_COMM_WORLD);
        if (!ownsDiag) {
            CUDA_CHECK(cudaMemcpyAsync(pl.dDiagSnap, pl.hDiagSnap[b], diagBytes,
                                       cudaMemcpyHostToDevice, pl.sMain));
        }

        // 2. Column panel (I,K) and row panel (K,J) on their owners.
        if (ownsCol && d.NLI > 0) {
            panelKernel<PANEL_COL><<<d.NLI, THREADS, 0, pl.sMain>>>(
                dDist + static_cast<size_t>(K / d.Q) * TT, dPath + static_cast<size_t>(K / d.Q) * TT,
                static_cast<size_t>(d.NLJ) * TT, d.r, d.P, K, kBase, kmax, pl.dDiagSnap,
                pl.dColSnap[b]);
            CUDA_CHECK(cudaGetLastError());
        }
        if (ownsRow && d.NLJ > 0) {
            const size_t off = static_cast<size_t>(K / d.P) * d.NLJ * TT;
            panelKernel<PANEL_ROW><<<d.NLJ, THREADS, 0, pl.sMain>>>(
                dDist + off, dPath + off, TT, d.c, d.Q, K, kBase, kmax, pl.dDiagSnap,
                pl.dRowSnap[b]);
            CUDA_CHECK(cudaGetLastError());
        }

        // 3. Exchange panel snapshots: column panels along process rows,
        //    row panels along process columns.
        if (ownsCol && colElems > 0) {
            CUDA_CHECK(cudaMemcpyAsync(pl.hColSnap[b], pl.dColSnap[b],
                                       colElems * sizeof(unsigned int), cudaMemcpyDeviceToHost,
                                       pl.sMain));
        }
        if (ownsRow && rowElems > 0) {
            CUDA_CHECK(cudaMemcpyAsync(pl.hRowSnap[b], pl.dRowSnap[b],
                                       rowElems * sizeof(unsigned int), cudaMemcpyDeviceToHost,
                                       pl.sMain));
        }
        if ((ownsCol && colElems > 0) || (ownsRow && rowElems > 0)) {
            CUDA_CHECK(cudaStreamSynchronize(pl.sMain));
        }
        if (colElems > 0) {
            MPI_Bcast(pl.hColSnap[b], static_cast<int>(colElems), MPI_UNSIGNED, K % d.Q, d.rowComm);
        }
        if (rowElems > 0) {
            MPI_Bcast(pl.hRowSnap[b], static_cast<int>(rowElems), MPI_UNSIGNED, K % d.P, d.colComm);
        }
        if (!ownsCol && colElems > 0) {
            CUDA_CHECK(cudaMemcpyAsync(pl.dColSnap[b], pl.hColSnap[b],
                                       colElems * sizeof(unsigned int), cudaMemcpyHostToDevice,
                                       pl.sMain));
        }
        if (!ownsRow && rowElems > 0) {
            CUDA_CHECK(cudaMemcpyAsync(pl.dRowSnap[b], pl.hRowSnap[b],
                                       rowElems * sizeof(unsigned int), cudaMemcpyHostToDevice,
                                       pl.sMain));
        }
        // Host staging of parity b can be reused once these copies have completed.
        CUDA_CHECK(cudaEventRecord(pl.evHostFree[b], pl.sMain));

        // Snapshots of block K are ready on the device; bulk work of block K may start
        // once the bulk work of block K-1 has finished (same stream, in order).
        CUDA_CHECK(cudaEventRecord(pl.evSnap[b], pl.sMain));
        CUDA_CHECK(cudaStreamWaitEvent(pl.sBulk, pl.evSnap[b], 0));

        // 4. Bulk tiles of block K, concurrent with the host-side work of later blocks.
        if (d.NLI > 0 && d.NLJ > 0) {
            updateKernel<<<dim3(d.NLJ, d.NLI), THREADS, 0, pl.sBulk>>>(
                dDist, dPath, d.NLJ, d.r, d.c, d.P, d.Q, K, kBase, kmax, UPDATE_BULK,
                pl.dColSnap[b], pl.dRowSnap[b]);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaEventRecord(pl.evBulk[b], pl.sBulk));

        // 5. Tiles of row/column K+1 (the next pivot block) on the main stream. They
        //    depend on block K-1 having been applied to them, which the bulk stream
        //    provides through evBulk[K-1].
        if (K >= 1) CUDA_CHECK(cudaStreamWaitEvent(pl.sMain, pl.evBulk[b ^ 1], 0));
        if (d.NLI > 0 && d.NLJ > 0) {
            updateKernel<<<dim3(d.NLJ, d.NLI), THREADS, 0, pl.sMain>>>(
                dDist, dPath, d.NLJ, d.r, d.c, d.P, d.Q, K, kBase, kmax, UPDATE_CRITICAL,
                pl.dColSnap[b], pl.dRowSnap[b]);
            CUDA_CHECK(cudaGetLastError());
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Places the tiles of rank (rr,cc) (packed as [tile][yl][xl]) into the full
// matrix stored in the original layout (element (x,y) at y*n + x).
static void scatterTiles(const unsigned int* src, const int rr, const int cc, const Decomposition& d,
                         const size_t n, std::vector<unsigned int>& out) {
    const int NLI = localCount(d.NB, rr, d.P);
    const int NLJ = localCount(d.NB, cc, d.Q);
    const long long numLocal = static_cast<long long>(NLI) * NLJ;

#pragma omp parallel for schedule(static)
    for (long long t = 0; t < numLocal; ++t) {
        const int li = static_cast<int>(t / NLJ);
        const int lj = static_cast<int>(t % NLJ);
        const size_t I = static_cast<size_t>(rr + li * d.P);
        const size_t J = static_cast<size_t>(cc + lj * d.Q);
        const unsigned int* S = src + static_cast<size_t>(t) * TT;
        for (int yl = 0; yl < TILE; ++yl) {
            const size_t y = J * TILE + yl;
            if (y >= n) break;
            for (int xl = 0; xl < TILE; ++xl) {
                const size_t x = I * TILE + xl;
                if (x >= n) break;
                out[y * n + x] = S[yl * TILE + xl];
            }
        }
    }
}

// Gathers the distance matrix onto rank 0 (only needed for -r / -v).
static void gatherDistances(const Decomposition& d, const int rank, const int size, const size_t n,
                            const unsigned int* hLocal, std::vector<unsigned int>& full) {
    const size_t localElems = static_cast<size_t>(d.NLI) * d.NLJ * TT;
    if (rank == 0) {
        full.assign(n * n, 0u);
        scatterTiles(hLocal, 0, 0, d, n, full);
        std::vector<unsigned int> buf;
        for (int rr = 1; rr < size; ++rr) {
            const int r = rr / d.Q;
            const int c = rr % d.Q;
            const size_t count = static_cast<size_t>(localCount(d.NB, r, d.P)) *
                                 localCount(d.NB, c, d.Q) * TT;
            buf.resize(count);
            MPI_Recv(buf.data(), static_cast<int>(count), MPI_UNSIGNED, rr, 0, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
            scatterTiles(buf.data(), r, c, d, n, full);
        }
    } else {
        MPI_Send(hLocal, static_cast<int>(localElems), MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD);
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
    MPI_Init(&argc, &argv);
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const bool root = (rank == 0);

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
            if (root) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (root) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    Decomposition d = makeDecomposition(rank, size, numNodes);
    selectDevice();

    const size_t localTiles = static_cast<size_t>(d.NLI) * d.NLJ;
    const size_t localElems = std::max<size_t>(localTiles * TT, 1);
    const size_t colElems = std::max<size_t>(static_cast<size_t>(d.NLI) * TT, 1);
    const size_t rowElems = std::max<size_t>(static_cast<size_t>(d.NLJ) * TT, 1);

    unsigned int *hDist = nullptr, *hPath = nullptr;
    CUDA_CHECK(cudaMallocHost(&hDist, localElems * sizeof(unsigned int)));
    CUDA_CHECK(cudaMallocHost(&hPath, localElems * sizeof(unsigned int)));

    unsigned int *dDist = nullptr, *dPath = nullptr;
    CUDA_CHECK(cudaMalloc(&dDist, localElems * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&dPath, localElems * sizeof(unsigned int)));

    Pipeline pl;
    CUDA_CHECK(cudaStreamCreateWithFlags(&pl.sMain, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&pl.sBulk, cudaStreamNonBlocking));
    for (int i = 0; i < 2; ++i) {
        CUDA_CHECK(cudaEventCreateWithFlags(&pl.evBulk[i], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&pl.evSnap[i], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&pl.evHostFree[i], cudaEventDisableTiming));
        CUDA_CHECK(cudaMallocHost(&pl.hDiagSnap[i], 2 * TT * sizeof(unsigned int)));
        CUDA_CHECK(cudaMallocHost(&pl.hColSnap[i], colElems * sizeof(unsigned int)));
        CUDA_CHECK(cudaMallocHost(&pl.hRowSnap[i], rowElems * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&pl.dColSnap[i], colElems * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&pl.dRowSnap[i], rowElems * sizeof(unsigned int)));
    }
    CUDA_CHECK(cudaMalloc(&pl.dDiagSnap, 2 * TT * sizeof(unsigned int)));

    // Initialize
    if (root) printf("Initializing graph...\n");
    initializeLocalTiles(d, numNodes, hDist, hPath);
    CUDA_CHECK(cudaMemcpy(dDist, hDist, localElems * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dPath, hPath, localElems * sizeof(unsigned int), cudaMemcpyHostToDevice));

    // Run Floyd-Warshall
    if (root) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    runBlockedFloydWarshall(d, numNodes, dDist, dPath, pl);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (root) {
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate operations per second
        // Floyd-Warshall has O(n³) complexity
        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);
    }

    int exitCode = 0;
    if (printResults || validate) {
        CUDA_CHECK(cudaMemcpy(hDist, dDist, localElems * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        std::vector<unsigned int> full;
        gatherDistances(d, rank, size, numNodes, hDist, full);

        // Print results for external validation (integer hash-based)
        if (root && printResults) {
            print_results_int(full, "DistanceMatrix");
        }

        // Validation
        if (validate && root) {
            printf("Validating result...\n");
            const bool valid = validateResult(full, numNodes);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }

    for (int i = 0; i < 2; ++i) {
        CUDA_CHECK(cudaEventDestroy(pl.evBulk[i]));
        CUDA_CHECK(cudaEventDestroy(pl.evSnap[i]));
        CUDA_CHECK(cudaEventDestroy(pl.evHostFree[i]));
        CUDA_CHECK(cudaFreeHost(pl.hDiagSnap[i]));
        CUDA_CHECK(cudaFreeHost(pl.hColSnap[i]));
        CUDA_CHECK(cudaFreeHost(pl.hRowSnap[i]));
        CUDA_CHECK(cudaFree(pl.dColSnap[i]));
        CUDA_CHECK(cudaFree(pl.dRowSnap[i]));
    }
    CUDA_CHECK(cudaFree(pl.dDiagSnap));
    CUDA_CHECK(cudaStreamDestroy(pl.sMain));
    CUDA_CHECK(cudaStreamDestroy(pl.sBulk));
    CUDA_CHECK(cudaFree(dDist));
    CUDA_CHECK(cudaFree(dPath));
    CUDA_CHECK(cudaFreeHost(hDist));
    CUDA_CHECK(cudaFreeHost(hPath));
    MPI_Comm_free(&d.rowComm);
    MPI_Comm_free(&d.colComm);

    MPI_Finalize();
    return exitCode;
}
