// Floyd-Warshall All-Pairs Shortest Path Benchmark
//
// Hybrid MPI + OpenMP + CUDA implementation.
//
//  * MPI    : the distance matrix is distributed by contiguous blocks of rows
//             (64-row block granularity) across ranks; every rank drives one
//             GPU. Each block-round of the blocked Floyd-Warshall algorithm
//             requires one broadcast of the pivot row panel (64 x n) from the
//             rank that owns it.
//  * CUDA   : all O(n^3) work is done on the GPU with the classic three-phase
//             blocked Floyd-Warshall algorithm (diagonal tile, pivot row/column
//             tiles, remaining tiles) using shared-memory tiling and register
//             blocking.
//  * OpenMP : host-side O(n^2) work (matrix initialisation / packing, result
//             gathering, validation) runs multithreaded on each rank.
//
// The computed distance matrix is bit-identical to the sequential algorithm
// (the blocked formulation computes exactly the same shortest-path values).

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
constexpr int TILE = 64;             // tile edge length (nodes)
constexpr int TX = 64;               // threads in x (one per tile column)
constexpr int TY = 16;               // threads in y
constexpr int RPT = TILE / TY;       // rows per thread (register blocking)
constexpr unsigned int NOT_CHANGED = 0xFFFFFFFFu;  // path sentinel: entry not improved

// Index calculation for flattened 2D array
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        cudaError_t err__ = (call);                                                   \
        if (err__ != cudaSuccess) {                                                   \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err__),    \
                    __FILE__, __LINE__);                                              \
            MPI_Abort(MPI_COMM_WORLD, 1);                                             \
        }                                                                             \
    } while (0)

#define MPI_CHECK(call)                                                               \
    do {                                                                              \
        int err__ = (call);                                                           \
        if (err__ != MPI_SUCCESS) {                                                   \
            fprintf(stderr, "MPI error %d at %s:%d\n", err__, __FILE__, __LINE__);    \
            MPI_Abort(MPI_COMM_WORLD, 1);                                             \
        }                                                                             \
    } while (0)

// ---------------------------------------------------------------------------
// CUDA kernels
//
// All matrices on the device are row-major with leading dimension `ld`
// (= padded number of nodes). Tile (bi, bj) starts at (bi*TILE)*ld + bj*TILE.
// ---------------------------------------------------------------------------

// Phase 1: the pivot diagonal tile, fully dependent -> one thread block.
__global__ void __launch_bounds__(TX * TY)
fwPhase1Kernel(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
               const size_t ld, const int kb, const size_t tileRowOffset) {
    __shared__ unsigned int s[TILE][TILE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    unsigned int* d = dist + (tileRowOffset) * ld + (size_t)kb * TILE;
    unsigned int* p = path + (tileRowOffset) * ld + (size_t)kb * TILE;

#pragma unroll
    for (int r = 0; r < RPT; ++r) {
        s[ty + r * TY][tx] = d[(size_t)(ty + r * TY) * ld + tx];
    }
    __syncthreads();

    unsigned int pk[RPT];
#pragma unroll
    for (int r = 0; r < RPT; ++r) pk[r] = NOT_CHANGED;

    for (int k = 0; k < TILE; ++k) {
        const unsigned int skj = s[k][tx];
#pragma unroll
        for (int r = 0; r < RPT; ++r) {
            const int i = ty + r * TY;
            const unsigned int v = s[i][k] + skj;
            if (v < s[i][tx]) {
                s[i][tx] = v;
                pk[r] = kb * TILE + k;
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int r = 0; r < RPT; ++r) {
        const int i = ty + r * TY;
        d[(size_t)i * ld + tx] = s[i][tx];
        if (pk[r] != NOT_CHANGED) p[(size_t)i * ld + tx] = pk[r];
    }
}

// Phase 2 (row): tiles (kb, j) for all j != kb, using the finished diagonal tile.
__global__ void __launch_bounds__(TX * TY)
fwPhase2RowKernel(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
                  const size_t ld, const int kb, const size_t tileRowOffset) {
    __shared__ unsigned int diag[TILE][TILE];
    __shared__ unsigned int cur[TILE][TILE];

    const int bj = blockIdx.x;
    if (bj == kb) return;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const unsigned int* dg = dist + tileRowOffset * ld + (size_t)kb * TILE;
    unsigned int* d = dist + tileRowOffset * ld + (size_t)bj * TILE;
    unsigned int* p = path + tileRowOffset * ld + (size_t)bj * TILE;

#pragma unroll
    for (int r = 0; r < RPT; ++r) {
        const int i = ty + r * TY;
        diag[i][tx] = dg[(size_t)i * ld + tx];
        cur[i][tx] = d[(size_t)i * ld + tx];
    }
    __syncthreads();

    unsigned int pk[RPT];
#pragma unroll
    for (int r = 0; r < RPT; ++r) pk[r] = NOT_CHANGED;

    for (int k = 0; k < TILE; ++k) {
        const unsigned int ckj = cur[k][tx];
#pragma unroll
        for (int r = 0; r < RPT; ++r) {
            const int i = ty + r * TY;
            const unsigned int v = diag[i][k] + ckj;
            if (v < cur[i][tx]) {
                cur[i][tx] = v;
                pk[r] = kb * TILE + k;
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int r = 0; r < RPT; ++r) {
        const int i = ty + r * TY;
        d[(size_t)i * ld + tx] = cur[i][tx];
        if (pk[r] != NOT_CHANGED) p[(size_t)i * ld + tx] = pk[r];
    }
}

// Phase 2 (column): local tiles (i, kb) for all local i != kb-tile, using the
// finished diagonal tile (taken from the broadcast pivot panel).
__global__ void __launch_bounds__(TX * TY)
fwPhase2ColKernel(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
                  const unsigned int* __restrict__ panel, const size_t ld,
                  const int kb, const int biOffset, const int skipA, const int skipB) {
    __shared__ unsigned int diag[TILE][TILE];
    __shared__ unsigned int cur[TILE][TILE];

    const int bi = (int)blockIdx.x + biOffset;
    if (bi == skipA || bi == skipB) return;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const unsigned int* dg = panel + (size_t)kb * TILE;
    unsigned int* d = dist + (size_t)bi * TILE * ld + (size_t)kb * TILE;
    unsigned int* p = path + (size_t)bi * TILE * ld + (size_t)kb * TILE;

#pragma unroll
    for (int r = 0; r < RPT; ++r) {
        const int i = ty + r * TY;
        diag[i][tx] = dg[(size_t)i * ld + tx];
        cur[i][tx] = d[(size_t)i * ld + tx];
    }
    __syncthreads();

    unsigned int pk[RPT];
#pragma unroll
    for (int r = 0; r < RPT; ++r) pk[r] = NOT_CHANGED;

    for (int k = 0; k < TILE; ++k) {
        const unsigned int dkj = diag[k][tx];
#pragma unroll
        for (int r = 0; r < RPT; ++r) {
            const int i = ty + r * TY;
            const unsigned int v = cur[i][k] + dkj;
            if (v < cur[i][tx]) {
                cur[i][tx] = v;
                pk[r] = kb * TILE + k;
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int r = 0; r < RPT; ++r) {
        const int i = ty + r * TY;
        d[(size_t)i * ld + tx] = cur[i][tx];
        if (pk[r] != NOT_CHANGED) p[(size_t)i * ld + tx] = pk[r];
    }
}

// Phase 3: all remaining local tiles (i, j), i != kb-tile, j != kb, using the
// local column tile (i, kb) and the pivot panel tile (kb, j). Independent.
// Each thread computes a 4-row x 2-column register block.
constexpr int P3TX = TILE / 2;       // threads in x for phase 3 (2 columns each)
constexpr int P3TY = TY;
constexpr int P3CPT = TILE / P3TX;   // columns per thread
__global__ void __launch_bounds__(P3TX * P3TY)
fwPhase3Kernel(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
               const unsigned int* __restrict__ panel, const size_t ld,
               const int kb, const int biOffset, const int skipA, const int skipB) {
    __shared__ unsigned int a[TILE][TILE];   // tile (i, kb)
    __shared__ unsigned int b[TILE][TILE];   // tile (kb, j)

    const int bj = blockIdx.x;
    const int bi = (int)blockIdx.y + biOffset;
    if (bi == skipA || bi == skipB || bj == kb) return;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const size_t rowBase = (size_t)bi * TILE * ld;
    const unsigned int* da = dist + rowBase + (size_t)kb * TILE;
    const unsigned int* db = panel + (size_t)bj * TILE;
    unsigned int* d = dist + rowBase + (size_t)bj * TILE;
    unsigned int* p = path + rowBase + (size_t)bj * TILE;

    unsigned int c[RPT][P3CPT];
    unsigned int pk[RPT][P3CPT];
#pragma unroll
    for (int r = 0; r < RPT; ++r) {
        const int i = ty + r * TY;
#pragma unroll
        for (int q = 0; q < P3CPT; ++q) {
            const int j = tx + q * P3TX;
            a[i][j] = da[(size_t)i * ld + j];
            b[i][j] = db[(size_t)i * ld + j];
            c[r][q] = d[(size_t)i * ld + j];
            pk[r][q] = NOT_CHANGED;
        }
    }
    __syncthreads();

#pragma unroll 8
    for (int k = 0; k < TILE; ++k) {
        unsigned int bk[P3CPT];
#pragma unroll
        for (int q = 0; q < P3CPT; ++q) bk[q] = b[k][tx + q * P3TX];
#pragma unroll
        for (int r = 0; r < RPT; ++r) {
            const unsigned int aik = a[ty + r * TY][k];
#pragma unroll
            for (int q = 0; q < P3CPT; ++q) {
                const unsigned int v = aik + bk[q];
                if (v < c[r][q]) {
                    c[r][q] = v;
                    pk[r][q] = kb * TILE + k;
                }
            }
        }
    }

#pragma unroll
    for (int r = 0; r < RPT; ++r) {
        const int i = ty + r * TY;
#pragma unroll
        for (int q = 0; q < P3CPT; ++q) {
            const int j = tx + q * P3TX;
            d[(size_t)i * ld + j] = c[r][q];
            if (pk[r][q] != NOT_CHANGED) p[(size_t)i * ld + j] = pk[r][q];
        }
    }
}

// ---------------------------------------------------------------------------
// Row-block distribution helpers
// ---------------------------------------------------------------------------
struct Partition {
    int nBlocks;        // total number of row blocks (padded n / TILE)
    size_t nPad;        // padded number of nodes
    int firstBlock;     // first row block owned by this rank
    int numBlocks;      // number of row blocks owned by this rank
};

static int blockFirst(const int nBlocks, const int rank, const int size) {
    return (int)(((long long)nBlocks * rank) / size);
}

static int ownerOfBlock(const int kb, const int nBlocks, const int size) {
    // Inverse of blockFirst: smallest rank r such that blockFirst(r+1) > kb.
    int lo = 0, hi = size - 1;
    while (lo < hi) {
        const int mid = (lo + hi) / 2;
        if (blockFirst(nBlocks, mid + 1, size) > kb) hi = mid; else lo = mid + 1;
    }
    return lo;
}

// Generate the local row band of the distance matrix. The random sequence is
// identical to the sequential reference (rand_r with seed 42 over the full
// n*n matrix); only the rows owned by this rank are stored (padded to nPad).
void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes,
                              const unsigned int rangeMin, const unsigned int rangeMax,
                              const Partition& part) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;
    const size_t nPad = part.nPad;
    const size_t rowBegin = (size_t)part.firstBlock * TILE;
    const size_t rowEnd = std::min(numNodes, rowBegin + (size_t)part.numBlocks * TILE);
    const size_t localRows = (size_t)part.numBlocks * TILE;

    // Padding entries (rows/cols >= numNodes) are unreachable (INF).
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < localRows * nPad; ++i) dist[i] = INF;

    // Sequential stream; skip the entries that belong to other ranks.
    for (size_t i = 0; i < rowBegin * numNodes; ++i) rand_r(&seed);
    for (size_t r = rowBegin; r < rowEnd; ++r) {
        unsigned int* row = dist.data() + (r - rowBegin) * nPad;
        for (size_t c = 0; c < numNodes; ++c) {
            row[c] = rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
        }
    }

    // Set diagonal to zero (distance from node to itself is 0)
#pragma omp parallel for schedule(static)
    for (size_t r = 0; r < localRows; ++r) {
        dist[r * nPad + (rowBegin + r)] = 0;
    }
}

// Reference semantics: after initialisation path[i][j] == i for all i, j.
void initializePathMatrix(std::vector<unsigned int>& path, const Partition& part) {
    const size_t nPad = part.nPad;
    const size_t rowBegin = (size_t)part.firstBlock * TILE;
    const size_t localRows = (size_t)part.numBlocks * TILE;
#pragma omp parallel for schedule(static)
    for (size_t r = 0; r < localRows; ++r) {
        unsigned int* row = path.data() + r * nPad;
        const unsigned int v = (unsigned int)(rowBegin + r);
        for (size_t c = 0; c < nPad; ++c) row[c] = v;
    }
}

// Distributed blocked Floyd-Warshall. dist/path hold the local row band on the
// host (localRows x nPad); on return they contain the result for this band.
//
// Lookahead pipeline: in round kb the owner of pivot block kb+1 first brings
// its row block kb+1 up to date, computes the next pivot panel (phases 1 and 2
// row), copies it to the host on a side stream and sends it (non-blocking) to
// every rank that owns rows; the transfer overlaps with phase 3 of round kb
// on all ranks. Panels are double buffered on host and device.
void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const Partition& part, const int rank, const int size) {
    const size_t nPad = part.nPad;
    const int nBlocks = part.nBlocks;
    const size_t localRows = (size_t)part.numBlocks * TILE;
    const size_t localElems = localRows * nPad;
    const size_t panelElems = (size_t)TILE * nPad;
    const size_t panelBytes = panelElems * sizeof(unsigned int);
    const bool haveRows = localElems > 0;
    const bool multi = size > 1;

    // Ranks that own at least one row block (only they need the pivot panels)
    std::vector<int> receivers;
    for (int r = 0; r < size; ++r) {
        if (blockFirst(nBlocks, r + 1, size) > blockFirst(nBlocks, r, size)) receivers.push_back(r);
    }
    if (!haveRows) return;  // nothing to compute and no panel traffic to take part in

    unsigned int* dDist = nullptr;
    unsigned int* dPath = nullptr;
    unsigned int* dPanel[2] = {nullptr, nullptr};
    unsigned int* hPanel[2] = {nullptr, nullptr};
    cudaStream_t compute, copy;
    cudaEvent_t evPivot[2], evD2H[2], evH2D[2], evRound[2];
    CUDA_CHECK(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&copy, cudaStreamNonBlocking));
    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaEventCreateWithFlags(&evPivot[b], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&evD2H[b], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&evH2D[b], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&evRound[b], cudaEventDisableTiming));
    }

    CUDA_CHECK(cudaMalloc(&dDist, localElems * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&dPath, localElems * sizeof(unsigned int)));
    if (multi) {
        CUDA_CHECK(cudaMalloc(&dPanel[0], panelBytes));
        CUDA_CHECK(cudaMalloc(&dPanel[1], panelBytes));
        CUDA_CHECK(cudaMallocHost(&hPanel[0], panelBytes));
        CUDA_CHECK(cudaMallocHost(&hPanel[1], panelBytes));
    }
    CUDA_CHECK(cudaMemcpyAsync(dDist, dist.data(), localElems * sizeof(unsigned int),
                               cudaMemcpyHostToDevice, compute));
    CUDA_CHECK(cudaMemcpyAsync(dPath, path.data(), localElems * sizeof(unsigned int),
                               cudaMemcpyHostToDevice, compute));

    const dim3 threads(TX, TY);
    const dim3 threads3(P3TX, P3TY);

    // Outstanding sends per host buffer (owner side) / receive per buffer
    std::vector<MPI_Request> sendReq[2];
    MPI_Request recvReq[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};
    bool h2dPending[2] = {false, false};

    auto localIndex = [&](const int kb) {
        return (kb >= part.firstBlock && kb < part.firstBlock + part.numBlocks)
                   ? kb - part.firstBlock : -1;
    };

    // Block until a CUDA event completes while keeping MPI progressing, so
    // that outstanding sends advance even without an asynchronous progress
    // thread in the MPI library.
    auto waitEventWithProgress = [&](cudaEvent_t ev) {
        for (;;) {
            const cudaError_t st = cudaEventQuery(ev);
            if (st == cudaSuccess) break;
            if (st != cudaErrorNotReady) CUDA_CHECK(st);
            for (int b = 0; b < 2; ++b) {
                if (!sendReq[b].empty()) {
                    int flag = 0;
                    MPI_CHECK(MPI_Testall((int)sendReq[b].size(), sendReq[b].data(), &flag,
                                          MPI_STATUSES_IGNORE));
                    if (flag) sendReq[b].clear();
                }
            }
        }
    };

    // Owner: stage the pivot panel `kb` (already computed on the device, event
    // evPivot[buf] recorded on the compute stream) to the host and send it.
    auto stageAndSend = [&](const int kb, const int buf) {
        if (!multi) return;
        const size_t rowOffset = (size_t)localIndex(kb) * TILE;
        // Buffer reuse: previous sends from this buffer must have completed
        if (!sendReq[buf].empty()) {
            MPI_CHECK(MPI_Waitall((int)sendReq[buf].size(), sendReq[buf].data(), MPI_STATUSES_IGNORE));
            sendReq[buf].clear();
        }
        CUDA_CHECK(cudaStreamWaitEvent(copy, evPivot[buf], 0));
        CUDA_CHECK(cudaMemcpyAsync(hPanel[buf], dDist + rowOffset * nPad, panelBytes,
                                   cudaMemcpyDeviceToHost, copy));
        CUDA_CHECK(cudaEventRecord(evD2H[buf], copy));
        waitEventWithProgress(evD2H[buf]);
        for (int r : receivers) {
            if (r == rank) continue;
            sendReq[buf].emplace_back();
            MPI_CHECK(MPI_Isend(hPanel[buf], (int)panelElems, MPI_UNSIGNED, r, kb,
                                MPI_COMM_WORLD, &sendReq[buf].back()));
        }
    };

    // Non-owner: post the receive for pivot panel `kb` into host buffer `buf`.
    auto postRecv = [&](const int kb, const int buf, const int owner) {
        if (!sendReq[buf].empty()) {
            // This rank sent from the buffer earlier (ownership boundary)
            MPI_CHECK(MPI_Waitall((int)sendReq[buf].size(), sendReq[buf].data(), MPI_STATUSES_IGNORE));
            sendReq[buf].clear();
        }
        if (h2dPending[buf]) {
            // The previous upload from this buffer must be finished before reuse
            CUDA_CHECK(cudaEventSynchronize(evH2D[buf]));
            h2dPending[buf] = false;
        }
        MPI_CHECK(MPI_Irecv(hPanel[buf], (int)panelElems, MPI_UNSIGNED, owner, kb,
                            MPI_COMM_WORLD, &recvReq[buf]));
    };

    // Pivot panel for round 0
    {
        const int owner = ownerOfBlock(0, nBlocks, size);
        if (owner == rank) {
            fwPhase1Kernel<<<1, threads, 0, compute>>>(dDist, dPath, nPad, 0, 0);
            fwPhase2RowKernel<<<nBlocks, threads, 0, compute>>>(dDist, dPath, nPad, 0, 0);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(evPivot[0], compute));
            stageAndSend(0, 0);
        } else {
            postRecv(0, 0, owner);
        }
    }

    for (int kb = 0; kb < nBlocks; ++kb) {
        const int buf = kb & 1;
        const int owner = ownerOfBlock(kb, nBlocks, size);
        const bool isOwner = (owner == rank);
        const int kbLocal = localIndex(kb);

        // Make the pivot panel for this round available on the device
        const unsigned int* panel = nullptr;
        if (isOwner) {
            panel = dDist + (size_t)kbLocal * TILE * nPad;
        } else {
            MPI_CHECK(MPI_Wait(&recvReq[buf], MPI_STATUS_IGNORE));
            // dPanel[buf] was last read by the kernels of round kb-2
            CUDA_CHECK(cudaStreamWaitEvent(copy, evRound[buf], 0));
            CUDA_CHECK(cudaMemcpyAsync(dPanel[buf], hPanel[buf], panelBytes,
                                       cudaMemcpyHostToDevice, copy));
            CUDA_CHECK(cudaEventRecord(evH2D[buf], copy));
            h2dPending[buf] = true;
            CUDA_CHECK(cudaStreamWaitEvent(compute, evH2D[buf], 0));
            panel = dPanel[buf];
        }

        const int nextOwner = (kb + 1 < nBlocks) ? ownerOfBlock(kb + 1, nBlocks, size) : -1;
        const bool ownNext = (nextOwner == rank);

        // Lookahead: produce the pivot panel of round kb+1 as early as possible
        int nextLocal = -1;
        if (ownNext) {
            nextLocal = localIndex(kb + 1);
            const size_t nextRowOffset = (size_t)nextLocal * TILE;
            // Finish round kb for row block kb+1 only
            fwPhase2ColKernel<<<1, threads, 0, compute>>>(dDist, dPath, panel, nPad, kb,
                                                          nextLocal, kbLocal, -1);
            fwPhase3Kernel<<<dim3(nBlocks, 1), threads3, 0, compute>>>(
                dDist, dPath, panel, nPad, kb, nextLocal, kbLocal, -1);
            // Next pivot: phase 1 and phase 2 (row)
            fwPhase1Kernel<<<1, threads, 0, compute>>>(dDist, dPath, nPad, kb + 1, nextRowOffset);
            fwPhase2RowKernel<<<nBlocks, threads, 0, compute>>>(dDist, dPath, nPad, kb + 1,
                                                                nextRowOffset);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(evPivot[buf ^ 1], compute));
        } else if (nextOwner >= 0) {
            postRecv(kb + 1, buf ^ 1, nextOwner);
        }

        // Remaining work of round kb for all other local row blocks
        fwPhase2ColKernel<<<part.numBlocks, threads, 0, compute>>>(dDist, dPath, panel, nPad,
                                                                   kb, 0, kbLocal, nextLocal);
        const dim3 grid3(nBlocks, part.numBlocks);
        fwPhase3Kernel<<<grid3, threads3, 0, compute>>>(dDist, dPath, panel, nPad, kb, 0,
                                                       kbLocal, nextLocal);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(evRound[buf], compute));

        // Ship the next pivot panel while phase 3 runs on the GPU
        if (ownNext) stageAndSend(kb + 1, buf ^ 1);
    }

    CUDA_CHECK(cudaMemcpyAsync(dist.data(), dDist, localElems * sizeof(unsigned int),
                               cudaMemcpyDeviceToHost, compute));
    CUDA_CHECK(cudaMemcpyAsync(path.data(), dPath, localElems * sizeof(unsigned int),
                               cudaMemcpyDeviceToHost, compute));
    for (int b = 0; b < 2; ++b) {
        if (!sendReq[b].empty()) {
            MPI_CHECK(MPI_Waitall((int)sendReq[b].size(), sendReq[b].data(), MPI_STATUSES_IGNORE));
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(compute));
    CUDA_CHECK(cudaStreamSynchronize(copy));

    CUDA_CHECK(cudaFree(dDist));
    CUDA_CHECK(cudaFree(dPath));
    for (int b = 0; b < 2; ++b) {
        if (dPanel[b]) CUDA_CHECK(cudaFree(dPanel[b]));
        if (hPanel[b]) CUDA_CHECK(cudaFreeHost(hPanel[b]));
        CUDA_CHECK(cudaEventDestroy(evPivot[b]));
        CUDA_CHECK(cudaEventDestroy(evD2H[b]));
        CUDA_CHECK(cudaEventDestroy(evH2D[b]));
        CUDA_CHECK(cudaEventDestroy(evRound[b]));
    }
    CUDA_CHECK(cudaStreamDestroy(compute));
    CUDA_CHECK(cudaStreamDestroy(copy));
}

// Gather the distributed (padded) bands into the full n x n matrix on rank 0.
void gatherDistanceMatrix(const std::vector<unsigned int>& localDist, const Partition& part,
                          const size_t numNodes, const int rank, const int size,
                          std::vector<unsigned int>& full) {
    const size_t nPad = part.nPad;
    const size_t rowBegin = (size_t)part.firstBlock * TILE;
    const size_t rowEnd = std::min(numNodes, rowBegin + (size_t)part.numBlocks * TILE);
    const size_t myRows = rowEnd > rowBegin ? rowEnd - rowBegin : 0;

    // Pack local rows without padding
    std::vector<unsigned int> packed(myRows * numNodes);
#pragma omp parallel for schedule(static)
    for (size_t r = 0; r < myRows; ++r) {
        std::memcpy(packed.data() + r * numNodes, localDist.data() + r * nPad,
                    numNodes * sizeof(unsigned int));
    }

    // Transfer in chunks whose element count fits in an int.
    const size_t rowsPerChunk = std::max<size_t>(1, (size_t)(1u << 30) / std::max<size_t>(1, numNodes));

    if (rank == 0) {
        full.resize(numNodes * numNodes);
        std::memcpy(full.data(), packed.data(), packed.size() * sizeof(unsigned int));
        for (int src = 1; src < size; ++src) {
            const int fb = blockFirst(part.nBlocks, src, size);
            const int nb = blockFirst(part.nBlocks, src + 1, size) - fb;
            const size_t rb = (size_t)fb * TILE;
            const size_t re = std::min(numNodes, rb + (size_t)nb * TILE);
            for (size_t r = rb; r < re; r += rowsPerChunk) {
                const size_t rows = std::min(rowsPerChunk, re - r);
                MPI_CHECK(MPI_Recv(full.data() + r * numNodes, (int)(rows * numNodes),
                                   MPI_UNSIGNED, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
            }
        }
    } else {
        for (size_t r = 0; r < myRows; r += rowsPerChunk) {
            const size_t rows = std::min(rowsPerChunk, myRows - r);
            MPI_CHECK(MPI_Send(packed.data() + r * numNodes, (int)(rows * numNodes),
                               MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD));
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
                            printf("Validation failed: triangle inequality violated at [%zu,%zu,%zu]\n",
                                   i, j, k);
                            ok = false;
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
    MPI_CHECK(MPI_Init(&argc, &argv));
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

    // Select one GPU per rank (round-robin over the GPUs visible on this node)
    {
        MPI_Comm nodeComm;
        MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                                      MPI_INFO_NULL, &nodeComm));
        int localRank = 0;
        MPI_CHECK(MPI_Comm_rank(nodeComm, &localRank));
        MPI_CHECK(MPI_Comm_free(&nodeComm));
        int deviceCount = 0;
        CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
        if (deviceCount < 1) {
            fprintf(stderr, "No CUDA devices found\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        CUDA_CHECK(cudaSetDevice(localRank % deviceCount));
        CUDA_CHECK(cudaFree(0));  // establish context now, outside the timed region
        // Force kernel module loading now so it does not land in the timed region
        cudaFuncAttributes attr;
        CUDA_CHECK(cudaFuncGetAttributes(&attr, fwPhase1Kernel));
        CUDA_CHECK(cudaFuncGetAttributes(&attr, fwPhase2RowKernel));
        CUDA_CHECK(cudaFuncGetAttributes(&attr, fwPhase2ColKernel));
        CUDA_CHECK(cudaFuncGetAttributes(&attr, fwPhase3Kernel));
    }

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, OpenMP threads/rank: %d\n", size, omp_get_max_threads());
    }

    // Row-block partition
    Partition part;
    part.nBlocks = (int)((numNodes + TILE - 1) / TILE);
    part.nPad = (size_t)part.nBlocks * TILE;
    part.firstBlock = blockFirst(part.nBlocks, rank, size);
    part.numBlocks = blockFirst(part.nBlocks, rank + 1, size) - part.firstBlock;
    const size_t localElems = (size_t)part.numBlocks * TILE * part.nPad;

    // Allocate matrices (local row band)
    std::vector<unsigned int> dist(localElems);
    std::vector<unsigned int> path(localElems);

    // Initialize
    if (rank == 0) printf("Initializing graph...\n");
    initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE, part);
    initializePathMatrix(path, part);

    // Run Floyd-Warshall
    if (rank == 0) printf("Computing shortest paths...\n");
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto start = std::chrono::high_resolution_clock::now();

    floydWarshall(dist, path, part, rank, size);

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
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

    int exitCode = 0;
    if (printResults || validate) {
        std::vector<unsigned int> full;
        gatherDistanceMatrix(dist, part, numNodes, rank, size, full);

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
    }

    MPI_CHECK(MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD));
    MPI_Finalize();
    return exitCode;
}
