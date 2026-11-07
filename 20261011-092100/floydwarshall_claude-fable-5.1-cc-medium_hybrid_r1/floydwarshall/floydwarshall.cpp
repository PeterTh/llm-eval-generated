// Hybrid MPI + OpenMP + CUDA Floyd-Warshall all-pairs shortest path benchmark.
//
// Parallelization strategy
// ------------------------
//  * MPI   : the distance matrix is partitioned by contiguous block-rows (tiles of
//            TILE rows) across ranks; every rank drives one GPU.  For every pivot
//            block kb the owning rank finalizes the pivot block-row (phase 1 + 2)
//            and broadcasts it; everyone else only needs that panel to update its
//            own rows (phase 2 column tiles + phase 3).  The owner of the *next*
//            pivot uses a lookahead: it updates its pivot block-row first, computes
//            the next panel and posts a non-blocking broadcast before finishing its
//            remaining rows, so communication overlaps computation.
//  * CUDA  : blocked Floyd-Warshall (TILE x TILE tiles, shared memory staging,
//            4x4 register tiling per thread) on the local rows of each rank.
//  * OpenMP: host side work – path matrix initialization, pad/unpad packing of
//            the local rows, the row-gather, and the validation.
//
// Semantics are those of the original sequential code: same deterministic
// graph (rand_r sequence), same relaxation rule (strict improvement), same
// outputs.  Blocked Floyd-Warshall produces exactly the same shortest-path
// distances as the classic triple loop.

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

// Padding value for rows/columns beyond numNodes.  Twice this value still fits
// in 32 bit, so padded entries never produce a wrap-around "improvement".
constexpr unsigned int PAD_DIST = 0x3fffffffu;
constexpr unsigned int NO_UPDATE = 0xffffffffu;

// Blocking parameters
constexpr int TILE = 64;                 // tile edge length
constexpr int BDX = 16;                  // threads in x
constexpr int BDY = 16;                  // threads in y
constexpr int RPT = TILE / BDY;          // rows per thread (4)
constexpr int CPT = TILE / BDX;          // cols per thread (4)

// Index calculation for flattened 2D array
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        cudaError_t err__ = (call);                                                   \
        if (err__ != cudaSuccess) {                                                   \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__, \
                    cudaGetErrorString(err__));                                       \
            MPI_Abort(MPI_COMM_WORLD, 1);                                             \
        }                                                                             \
    } while (0)

// ---------------------------------------------------------------------------
// CUDA kernels.  All matrices are row-major with pitch `np` (padded size).
// `dist`/`path` point to the first *local* row of the rank; `panel` points to
// the TILE x np pivot block-row (either inside the local matrix of the owner or
// to a separate receive buffer).
// ---------------------------------------------------------------------------

// Phase 1: the diagonal pivot tile (kb,kb).  Single thread block.
__global__ void __launch_bounds__(BDX * BDY)
fwPhase1(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
         const size_t np, const int localRow0, const int kb) {
    __shared__ unsigned int s[TILE][TILE + 1];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const size_t base = (size_t)localRow0 * np + (size_t)kb * TILE;

    for (int a = 0; a < RPT; ++a)
        for (int b = 0; b < CPT; ++b)
            s[ty + a * BDY][tx + b * BDX] = dist[base + (size_t)(ty + a * BDY) * np + tx + b * BDX];
    __syncthreads();

    unsigned int bk[RPT][CPT];
    for (int a = 0; a < RPT; ++a)
        for (int b = 0; b < CPT; ++b) bk[a][b] = NO_UPDATE;

    for (int k = 0; k < TILE; ++k) {
        unsigned int rk[RPT], kc[CPT];
        for (int a = 0; a < RPT; ++a) rk[a] = s[ty + a * BDY][k];
        for (int b = 0; b < CPT; ++b) kc[b] = s[k][tx + b * BDX];
        for (int a = 0; a < RPT; ++a)
            for (int b = 0; b < CPT; ++b) {
                const unsigned int v = rk[a] + kc[b];
                if (v < s[ty + a * BDY][tx + b * BDX]) {
                    s[ty + a * BDY][tx + b * BDX] = v;
                    bk[a][b] = k;
                }
            }
        __syncthreads();
    }

    const unsigned int kg = (unsigned int)kb * TILE;
    for (int a = 0; a < RPT; ++a)
        for (int b = 0; b < CPT; ++b) {
            const size_t o = base + (size_t)(ty + a * BDY) * np + tx + b * BDX;
            dist[o] = s[ty + a * BDY][tx + b * BDX];
            if (bk[a][b] != NO_UPDATE) path[o] = kg + bk[a][b];
        }
}

// Phase 2 (row): tiles (kb, j), j != kb, of the pivot block-row.  Owner only.
// blockIdx.x enumerates column tiles.
__global__ void __launch_bounds__(BDX * BDY)
fwPhase2Row(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
            const size_t np, const int localRow0, const int kb) {
    __shared__ unsigned int sd[TILE][TILE + 1];   // diagonal tile
    __shared__ unsigned int so[TILE][TILE + 1];   // own tile
    const int j = blockIdx.x;
    if (j == kb) return;
    const int tx = threadIdx.x, ty = threadIdx.y;
    const size_t rowBase = (size_t)localRow0 * np;
    const size_t dBase = rowBase + (size_t)kb * TILE;
    const size_t oBase = rowBase + (size_t)j * TILE;

    for (int a = 0; a < RPT; ++a)
        for (int b = 0; b < CPT; ++b) {
            const size_t off = (size_t)(ty + a * BDY) * np + tx + b * BDX;
            sd[ty + a * BDY][tx + b * BDX] = dist[dBase + off];
            so[ty + a * BDY][tx + b * BDX] = dist[oBase + off];
        }
    __syncthreads();

    unsigned int bk[RPT][CPT];
    for (int a = 0; a < RPT; ++a)
        for (int b = 0; b < CPT; ++b) bk[a][b] = NO_UPDATE;

    for (int k = 0; k < TILE; ++k) {
        unsigned int rk[RPT], kc[CPT];
        for (int a = 0; a < RPT; ++a) rk[a] = sd[ty + a * BDY][k];
        for (int b = 0; b < CPT; ++b) kc[b] = so[k][tx + b * BDX];
        for (int a = 0; a < RPT; ++a)
            for (int b = 0; b < CPT; ++b) {
                const unsigned int v = rk[a] + kc[b];
                if (v < so[ty + a * BDY][tx + b * BDX]) {
                    so[ty + a * BDY][tx + b * BDX] = v;
                    bk[a][b] = k;
                }
            }
        __syncthreads();
    }

    const unsigned int kg = (unsigned int)kb * TILE;
    for (int a = 0; a < RPT; ++a)
        for (int b = 0; b < CPT; ++b) {
            const size_t o = oBase + (size_t)(ty + a * BDY) * np + tx + b * BDX;
            dist[o] = so[ty + a * BDY][tx + b * BDX];
            if (bk[a][b] != NO_UPDATE) path[o] = kg + bk[a][b];
        }
}

// Phase 2 (column): tiles (i, kb) for local block-rows i (i != kb).
// blockIdx.x enumerates local block-rows.
__global__ void __launch_bounds__(BDX * BDY)
fwPhase2Col(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
            const unsigned int* __restrict__ panel, const size_t np,
            const int blockRow0, const int kb) {
    __shared__ unsigned int sd[TILE][TILE + 1];   // diagonal tile (from panel)
    __shared__ unsigned int so[TILE][TILE + 1];   // own tile
    const int li = blockIdx.x;
    if (blockRow0 + li == kb) return;
    const int tx = threadIdx.x, ty = threadIdx.y;
    const size_t oBase = (size_t)li * TILE * np + (size_t)kb * TILE;
    const size_t dBase = (size_t)kb * TILE;

    for (int a = 0; a < RPT; ++a)
        for (int b = 0; b < CPT; ++b) {
            const size_t off = (size_t)(ty + a * BDY) * np + tx + b * BDX;
            sd[ty + a * BDY][tx + b * BDX] = panel[dBase + off];
            so[ty + a * BDY][tx + b * BDX] = dist[oBase + off];
        }
    __syncthreads();

    unsigned int bk[RPT][CPT];
    for (int a = 0; a < RPT; ++a)
        for (int b = 0; b < CPT; ++b) bk[a][b] = NO_UPDATE;

    for (int k = 0; k < TILE; ++k) {
        unsigned int rk[RPT], kc[CPT];
        for (int a = 0; a < RPT; ++a) rk[a] = so[ty + a * BDY][k];
        for (int b = 0; b < CPT; ++b) kc[b] = sd[k][tx + b * BDX];
        for (int a = 0; a < RPT; ++a)
            for (int b = 0; b < CPT; ++b) {
                const unsigned int v = rk[a] + kc[b];
                if (v < so[ty + a * BDY][tx + b * BDX]) {
                    so[ty + a * BDY][tx + b * BDX] = v;
                    bk[a][b] = k;
                }
            }
        __syncthreads();
    }

    const unsigned int kg = (unsigned int)kb * TILE;
    for (int a = 0; a < RPT; ++a)
        for (int b = 0; b < CPT; ++b) {
            const size_t o = oBase + (size_t)(ty + a * BDY) * np + tx + b * BDX;
            dist[o] = so[ty + a * BDY][tx + b * BDX];
            if (bk[a][b] != NO_UPDATE) path[o] = kg + bk[a][b];
        }
}

// Phase 3: tiles (i, j) for local block-rows i in [rowTileBegin, rowTileEnd),
// i != kb, j != kb.  grid = (numColTiles, rowTileEnd - rowTileBegin).
__global__ void __launch_bounds__(BDX * BDY)
fwPhase3(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
         const unsigned int* __restrict__ panel, const size_t np,
         const int blockRow0, const int rowTileBegin, const int kb) {
    __shared__ unsigned int sc[TILE][TILE + 1];   // column tile (i, kb)
    __shared__ unsigned int sr[TILE][TILE + 1];   // row tile (kb, j) from panel
    const int j = blockIdx.x;
    const int li = rowTileBegin + blockIdx.y;
    if (j == kb || blockRow0 + li == kb) return;
    const int tx = threadIdx.x, ty = threadIdx.y;
    const size_t cBase = (size_t)li * TILE * np + (size_t)kb * TILE;
    const size_t rBase = (size_t)j * TILE;
    const size_t oBase = (size_t)li * TILE * np + (size_t)j * TILE;

    unsigned int cur[RPT][CPT];
    unsigned int bk[RPT][CPT];
    for (int a = 0; a < RPT; ++a)
        for (int b = 0; b < CPT; ++b) {
            const size_t off = (size_t)(ty + a * BDY) * np + tx + b * BDX;
            sc[ty + a * BDY][tx + b * BDX] = dist[cBase + off];
            sr[ty + a * BDY][tx + b * BDX] = panel[rBase + off];
            cur[a][b] = dist[oBase + off];
            bk[a][b] = NO_UPDATE;
        }
    __syncthreads();

#pragma unroll 8
    for (int k = 0; k < TILE; ++k) {
        unsigned int rk[RPT], kc[CPT];
        for (int a = 0; a < RPT; ++a) rk[a] = sc[ty + a * BDY][k];
        for (int b = 0; b < CPT; ++b) kc[b] = sr[k][tx + b * BDX];
        for (int a = 0; a < RPT; ++a)
            for (int b = 0; b < CPT; ++b) {
                const unsigned int v = rk[a] + kc[b];
                if (v < cur[a][b]) {
                    cur[a][b] = v;
                    bk[a][b] = k;
                }
            }
    }

    const unsigned int kg = (unsigned int)kb * TILE;
    for (int a = 0; a < RPT; ++a)
        for (int b = 0; b < CPT; ++b) {
            if (bk[a][b] != NO_UPDATE) {
                const size_t o = oBase + (size_t)(ty + a * BDY) * np + tx + b * BDX;
                dist[o] = cur[a][b];
                path[o] = kg + bk[a][b];
            }
        }
}

// ---------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------

// Each rank reproduces the full deterministic rand_r stream but only stores its
// own rows [row0, row0+rows).  The sequence is inherently serial (cost O(n^2),
// negligible against the O(n^3) solve).
void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes,
                              const size_t row0, const size_t rows,
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;

    const size_t begin = row0 * numNodes;
    const size_t end = (row0 + rows) * numNodes;
    for (size_t i = 0; i < begin; ++i) (void)rand_r(&seed);
    for (size_t i = begin; i < end; ++i) {
        dist[i - begin] = rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
    }

    // Set diagonal to zero (distance from node to itself is 0)
    for (size_t i = row0; i < row0 + rows; ++i) {
        dist[idx2(i, i - row0, numNodes)] = 0;
    }
}

// Local rows of the path matrix: path[i][j] = j (same as the original after its
// symmetric initialization loop).
void initializePathMatrix(std::vector<unsigned int>& path, const size_t numNodes,
                          const size_t rows) {
#pragma omp parallel for schedule(static)
    for (size_t r = 0; r < rows; ++r) {
        for (size_t j = 0; j < numNodes; ++j) {
            path[r * numNodes + j] = (unsigned int)j;
        }
    }
}

struct Partition {
    int tileBegin = 0;   // first block-row owned
    int tileEnd = 0;     // one past last block-row owned
    size_t rowBegin = 0; // first (unpadded) row owned
    size_t rowEnd = 0;   // one past last (unpadded) row owned
};

Partition makePartition(const int rank, const int size, const int numTiles, const size_t numNodes) {
    Partition p;
    p.tileBegin = (int)(((long long)numTiles * rank) / size);
    p.tileEnd = (int)(((long long)numTiles * (rank + 1)) / size);
    p.rowBegin = std::min(numNodes, (size_t)p.tileBegin * TILE);
    p.rowEnd = std::min(numNodes, (size_t)p.tileEnd * TILE);
    return p;
}

inline int ownerOfTile(const int kb, const int size, const int numTiles) {
    // inverse of makePartition: smallest rank r with tileEnd(r) > kb
    int lo = 0, hi = size - 1;
    while (lo < hi) {
        const int mid = (lo + hi) / 2;
        const int end = (int)(((long long)numTiles * (mid + 1)) / size);
        if (end > kb) hi = mid; else lo = mid + 1;
    }
    return lo;
}

// Distributed hybrid Floyd-Warshall on the local row block of this rank.
// dist/path hold the local rows (rows x numNodes, row-major) on entry and exit.
void floydWarshall(std::vector<unsigned int>& dist, std::vector<unsigned int>& path,
                   const size_t numNodes, const Partition& part,
                   const int rank, const int size, const int numTiles) {
    const size_t np = (size_t)numTiles * TILE;           // padded dimension
    const int myTiles = part.tileEnd - part.tileBegin;
    const size_t myRows = part.rowEnd - part.rowBegin;   // real rows
    const size_t myPadRows = (size_t)myTiles * TILE;
    const size_t panelElems = (size_t)TILE * np;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    // Device buffers
    unsigned int *dDist = nullptr, *dPath = nullptr, *dPanel = nullptr;
    const size_t localElems = std::max<size_t>(1, myPadRows * np);
    CUDA_CHECK(cudaMalloc(&dDist, localElems * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&dPath, localElems * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&dPanel, std::max<size_t>(1, panelElems) * sizeof(unsigned int)));

    // Pinned host staging: padded local block and two panel buffers
    unsigned int* hLocal = nullptr;
    unsigned int* hPanel[2] = {nullptr, nullptr};
    CUDA_CHECK(cudaHostAlloc(&hLocal, localElems * sizeof(unsigned int), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&hPanel[0], std::max<size_t>(1, panelElems) * sizeof(unsigned int), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&hPanel[1], std::max<size_t>(1, panelElems) * sizeof(unsigned int), cudaHostAllocDefault));

    // Pack local rows into the padded layout (OpenMP), upload
#pragma omp parallel for schedule(static)
    for (size_t r = 0; r < myPadRows; ++r) {
        unsigned int* dst = hLocal + r * np;
        if (r < myRows) {
            memcpy(dst, dist.data() + r * numNodes, numNodes * sizeof(unsigned int));
            for (size_t c = numNodes; c < np; ++c) dst[c] = PAD_DIST;
        } else {
            for (size_t c = 0; c < np; ++c) dst[c] = PAD_DIST;
        }
    }
    CUDA_CHECK(cudaMemcpyAsync(dDist, hLocal, myPadRows * np * sizeof(unsigned int),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));  // hLocal is reused below
#pragma omp parallel for schedule(static)
    for (size_t r = 0; r < myPadRows; ++r) {
        unsigned int* dst = hLocal + r * np;
        if (r < myRows) {
            memcpy(dst, path.data() + r * numNodes, numNodes * sizeof(unsigned int));
            for (size_t c = numNodes; c < np; ++c) dst[c] = (unsigned int)c;
        } else {
            for (size_t c = 0; c < np; ++c) dst[c] = (unsigned int)c;
        }
    }
    CUDA_CHECK(cudaMemcpyAsync(dPath, hLocal, myPadRows * np * sizeof(unsigned int),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const dim3 block(BDX, BDY);

    // Owner computes the pivot block-row kb (phase 1 + phase 2 row), downloads it
    // into hPanel[kb & 1] and posts the broadcast.
    auto producePanel = [&](const int kb, MPI_Request& req) {
        const int lt = kb - part.tileBegin;
        fwPhase1<<<1, block, 0, stream>>>(dDist, dPath, np, lt * TILE, kb);
        fwPhase2Row<<<numTiles, block, 0, stream>>>(dDist, dPath, np, lt * TILE, kb);
        CUDA_CHECK(cudaMemcpyAsync(hPanel[kb & 1], dDist + (size_t)lt * TILE * np,
                                   panelElems * sizeof(unsigned int),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        MPI_Ibcast(hPanel[kb & 1], (int)panelElems, MPI_UNSIGNED, rank, MPI_COMM_WORLD, &req);
    };

    MPI_Request req = MPI_REQUEST_NULL;
    if (numTiles > 0 && ownerOfTile(0, size, numTiles) == rank) producePanel(0, req);

    for (int kb = 0; kb < numTiles; ++kb) {
        const int owner = ownerOfTile(kb, size, numTiles);
        const bool isOwner = (owner == rank);
        const unsigned int* panel;

        if (isOwner) {
            // Panel kb was produced earlier (prologue or lookahead); the pivot
            // rows live in the local matrix.
            MPI_Wait(&req, MPI_STATUS_IGNORE);
            panel = dDist + (size_t)(kb - part.tileBegin) * TILE * np;
        } else {
            MPI_Ibcast(hPanel[kb & 1], (int)panelElems, MPI_UNSIGNED, owner, MPI_COMM_WORLD, &req);
            MPI_Wait(&req, MPI_STATUS_IGNORE);
            CUDA_CHECK(cudaMemcpyAsync(dPanel, hPanel[kb & 1], panelElems * sizeof(unsigned int),
                                       cudaMemcpyHostToDevice, stream));
            panel = dPanel;
        }
        if (myTiles == 0) continue;

        // Phase 2 column tiles for all local block-rows
        fwPhase2Col<<<myTiles, block, 0, stream>>>(dDist, dPath, panel, np, part.tileBegin, kb);

        const int nextKb = kb + 1;
        const bool nextIsMine = (nextKb < numTiles) &&
                                (nextKb >= part.tileBegin) && (nextKb < part.tileEnd);
        if (nextIsMine) {
            // Lookahead: finish the next pivot block-row first, produce + post
            // its panel, then do the remaining local rows.
            const int lt = nextKb - part.tileBegin;
            fwPhase3<<<dim3(numTiles, 1), block, 0, stream>>>(dDist, dPath, panel, np,
                                                              part.tileBegin, lt, kb);
            producePanel(nextKb, req);
            if (lt > 0)
                fwPhase3<<<dim3(numTiles, lt), block, 0, stream>>>(dDist, dPath, panel, np,
                                                                   part.tileBegin, 0, kb);
            if (myTiles - lt - 1 > 0)
                fwPhase3<<<dim3(numTiles, myTiles - lt - 1), block, 0, stream>>>(
                    dDist, dPath, panel, np, part.tileBegin, lt + 1, kb);
        } else {
            fwPhase3<<<dim3(numTiles, myTiles), block, 0, stream>>>(dDist, dPath, panel, np,
                                                                    part.tileBegin, 0, kb);
        }
        // Make sure the panel receive buffer is free before it is reused two
        // iterations later; the stream order guarantees that once the next
        // iteration's H2D copy is enqueued the previous kernels are done.
    }
    CUDA_CHECK(cudaGetLastError());

    // Download local rows and strip the padding
    CUDA_CHECK(cudaMemcpyAsync(hLocal, dDist, myPadRows * np * sizeof(unsigned int),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
#pragma omp parallel for schedule(static)
    for (size_t r = 0; r < myRows; ++r)
        memcpy(dist.data() + r * numNodes, hLocal + r * np, numNodes * sizeof(unsigned int));
    CUDA_CHECK(cudaMemcpyAsync(hLocal, dPath, myPadRows * np * sizeof(unsigned int),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
#pragma omp parallel for schedule(static)
    for (size_t r = 0; r < myRows; ++r)
        memcpy(path.data() + r * numNodes, hLocal + r * np, numNodes * sizeof(unsigned int));

    CUDA_CHECK(cudaFreeHost(hLocal));
    CUDA_CHECK(cudaFreeHost(hPanel[0]));
    CUDA_CHECK(cudaFreeHost(hPanel[1]));
    CUDA_CHECK(cudaFree(dDist));
    CUDA_CHECK(cudaFree(dPath));
    CUDA_CHECK(cudaFree(dPanel));
    CUDA_CHECK(cudaStreamDestroy(stream));
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
    bool ok = true;
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
#pragma omp critical
                        {
                            if (ok) {
                                printf("Validation failed: triangle inequality violated at [%zu,%zu,%zu]\n",
                                       i, j, k);
                                ok = false;
                            }
                        }
                        break;
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

    // Bind each rank to a GPU (round-robin over the GPUs of the node)
    {
        MPI_Comm nodeComm;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
        int localRank = 0;
        MPI_Comm_rank(nodeComm, &localRank);
        MPI_Comm_free(&nodeComm);
        int devCount = 0;
        CUDA_CHECK(cudaGetDeviceCount(&devCount));
        if (devCount < 1) {
            fprintf(stderr, "No CUDA device found\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        CUDA_CHECK(cudaSetDevice(localRank % devCount));
        CUDA_CHECK(cudaFree(0));  // establish the context before timing
    }

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, OpenMP threads/rank: %d, tile size: %d\n",
               size, omp_get_max_threads(), TILE);
    }

    const int numTiles = (int)((numNodes + TILE - 1) / TILE);
    const Partition part = makePartition(rank, size, numTiles, numNodes);
    const size_t myRows = part.rowEnd - part.rowBegin;

    // Allocate local row blocks
    std::vector<unsigned int> dist(myRows * numNodes);
    std::vector<unsigned int> path(myRows * numNodes);

    // Initialize
    if (rank == 0) printf("Initializing graph...\n");
    initializeDistanceMatrix(dist, numNodes, part.rowBegin, myRows, 1, MAX_DISTANCE);
    initializePathMatrix(path, numNodes, myRows);

    // Run Floyd-Warshall
    if (rank == 0) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    floydWarshall(dist, path, numNodes, part, rank, size, numTiles);

    MPI_Barrier(MPI_COMM_WORLD);
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

    int exitCode = 0;
    if (printResults || validate) {
        // Gather the full distance matrix on rank 0
        std::vector<unsigned int> full;
        std::vector<int> counts(size), displs(size);
        for (int r = 0; r < size; ++r) {
            const Partition pr = makePartition(r, size, numTiles, numNodes);
            counts[r] = (int)((pr.rowEnd - pr.rowBegin) * numNodes);
            displs[r] = (int)(pr.rowBegin * numNodes);
        }
        if (rank == 0) full.resize(numNodes * numNodes);
        MPI_Gatherv(dist.data(), (int)(myRows * numNodes), MPI_UNSIGNED,
                    full.data(), counts.data(), displs.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD);

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
