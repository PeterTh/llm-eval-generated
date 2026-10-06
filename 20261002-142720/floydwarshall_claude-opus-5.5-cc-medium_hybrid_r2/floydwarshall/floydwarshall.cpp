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

// Tile size for blocked Floyd-Warshall (each CUDA block handles a TILE x TILE tile,
// THREADS x THREADS threads, each thread owns PER_T x PER_T elements).
constexpr int TILE = 64;
constexpr int THREADS = 16;
constexpr int PER_T = TILE / THREADS;

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        cudaError_t err_ = (call);                                                    \
        if (err_ != cudaSuccess) {                                                    \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_),     \
                    __FILE__, __LINE__);                                              \
            MPI_Abort(MPI_COMM_WORLD, 1);                                             \
        }                                                                             \
    } while (0)

// Index calculation for flattened 2D array
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes, 
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;
    
    // Sequential RNG stream (must stay serial to reproduce the exact sequence)
    for (size_t i = 0; i < numNodes * numNodes; ++i) {
        dist[i] = rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
    }
    
    // Set diagonal to zero (distance from node to itself is 0)
    for (size_t i = 0; i < numNodes; ++i) {
        dist[idx2(i, i, numNodes)] = 0;
    }
}

// ---------------------------------------------------------------------------
// CUDA kernels. Matrices are row-major: element (row i, col j) at i * ld + j,
// which matches dist[idx2(j, i, n)] of the original code.
// ---------------------------------------------------------------------------

// Path matrix initialization: path[i][j] = i (equivalent to the original init loop).
__global__ void initPathKernel(unsigned int* __restrict__ path, size_t ld, size_t rows,
                               unsigned int rowBase) {
    const size_t total = rows * ld;
    for (size_t e = blockIdx.x * (size_t)blockDim.x + threadIdx.x; e < total;
         e += (size_t)gridDim.x * blockDim.x) {
        path[e] = rowBase + (unsigned int)(e / ld);
    }
}

// Phase 1: pivot tile (kb, kb), dependent on itself.
__global__ void __launch_bounds__(THREADS * THREADS)
phase1Kernel(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
             size_t ld, unsigned int kBase) {
    __shared__ unsigned int t[TILE][TILE + 1];
    const int tx = threadIdx.x, ty = threadIdx.y;
    unsigned int p[PER_T][PER_T];
    bool changed[PER_T][PER_T];
#pragma unroll
    for (int r = 0; r < PER_T; ++r)
#pragma unroll
        for (int c = 0; c < PER_T; ++c) {
            t[ty + r * THREADS][tx + c * THREADS] =
                dist[(size_t)(ty + r * THREADS) * ld + tx + c * THREADS];
            changed[r][c] = false;
            p[r][c] = 0;
        }
    __syncthreads();
    for (int m = 0; m < TILE; ++m) {
#pragma unroll
        for (int r = 0; r < PER_T; ++r) {
            const unsigned int a = t[ty + r * THREADS][m];
#pragma unroll
            for (int c = 0; c < PER_T; ++c) {
                const unsigned int s = a + t[m][tx + c * THREADS];
                if (s < t[ty + r * THREADS][tx + c * THREADS]) {
                    t[ty + r * THREADS][tx + c * THREADS] = s;
                    p[r][c] = kBase + m;
                    changed[r][c] = true;
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < PER_T; ++r)
#pragma unroll
        for (int c = 0; c < PER_T; ++c) {
            const size_t o = (size_t)(ty + r * THREADS) * ld + tx + c * THREADS;
            dist[o] = t[ty + r * THREADS][tx + c * THREADS];
            if (changed[r][c]) path[o] = p[r][c];
        }
}

// Phase 2 (row): tiles (kb, jb), jb != kb. rowBlk points to the start of pivot block-row.
__global__ void __launch_bounds__(THREADS * THREADS)
phase2RowKernel(unsigned int* __restrict__ rowBlk, unsigned int* __restrict__ pathBlk,
                size_t ld, int kb, unsigned int kBase) {
    __shared__ unsigned int piv[TILE][TILE + 1];
    __shared__ unsigned int t[TILE][TILE + 1];
    const int jb = blockIdx.x < (unsigned)kb ? blockIdx.x : blockIdx.x + 1;
    const int tx = threadIdx.x, ty = threadIdx.y;
    unsigned int p[PER_T][PER_T];
    bool changed[PER_T][PER_T];
#pragma unroll
    for (int r = 0; r < PER_T; ++r)
#pragma unroll
        for (int c = 0; c < PER_T; ++c) {
            const size_t row = (size_t)(ty + r * THREADS) * ld;
            piv[ty + r * THREADS][tx + c * THREADS] = rowBlk[row + (size_t)kb * TILE + tx + c * THREADS];
            t[ty + r * THREADS][tx + c * THREADS] = rowBlk[row + (size_t)jb * TILE + tx + c * THREADS];
            changed[r][c] = false;
            p[r][c] = 0;
        }
    __syncthreads();
    for (int m = 0; m < TILE; ++m) {
#pragma unroll
        for (int r = 0; r < PER_T; ++r) {
            const unsigned int a = piv[ty + r * THREADS][m];
#pragma unroll
            for (int c = 0; c < PER_T; ++c) {
                const unsigned int s = a + t[m][tx + c * THREADS];
                if (s < t[ty + r * THREADS][tx + c * THREADS]) {
                    t[ty + r * THREADS][tx + c * THREADS] = s;
                    p[r][c] = kBase + m;
                    changed[r][c] = true;
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < PER_T; ++r)
#pragma unroll
        for (int c = 0; c < PER_T; ++c) {
            const size_t o = (size_t)(ty + r * THREADS) * ld + (size_t)jb * TILE + tx + c * THREADS;
            rowBlk[o] = t[ty + r * THREADS][tx + c * THREADS];
            if (changed[r][c]) pathBlk[o] = p[r][c];
        }
}

// Phase 2 (column): tiles (ib, kb) for local block-rows. pivRow is the pivot block-row
// (TILE rows x ld columns, already final for this round).
__global__ void __launch_bounds__(THREADS * THREADS)
phase2ColKernel(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
                const unsigned int* __restrict__ pivRow, size_t ld, int kb, unsigned int kBase) {
    __shared__ unsigned int piv[TILE][TILE + 1];
    __shared__ unsigned int t[TILE][TILE + 1];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const size_t blkOff = (size_t)blockIdx.x * TILE * ld + (size_t)kb * TILE;
    unsigned int p[PER_T][PER_T];
    bool changed[PER_T][PER_T];
#pragma unroll
    for (int r = 0; r < PER_T; ++r)
#pragma unroll
        for (int c = 0; c < PER_T; ++c) {
            const size_t row = (size_t)(ty + r * THREADS) * ld;
            piv[ty + r * THREADS][tx + c * THREADS] = pivRow[row + (size_t)kb * TILE + tx + c * THREADS];
            t[ty + r * THREADS][tx + c * THREADS] = dist[blkOff + row + tx + c * THREADS];
            changed[r][c] = false;
            p[r][c] = 0;
        }
    __syncthreads();
    for (int m = 0; m < TILE; ++m) {
#pragma unroll
        for (int r = 0; r < PER_T; ++r) {
            const unsigned int a = t[ty + r * THREADS][m];
#pragma unroll
            for (int c = 0; c < PER_T; ++c) {
                const unsigned int s = a + piv[m][tx + c * THREADS];
                if (s < t[ty + r * THREADS][tx + c * THREADS]) {
                    t[ty + r * THREADS][tx + c * THREADS] = s;
                    p[r][c] = kBase + m;
                    changed[r][c] = true;
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < PER_T; ++r)
#pragma unroll
        for (int c = 0; c < PER_T; ++c) {
            const size_t o = blkOff + (size_t)(ty + r * THREADS) * ld + tx + c * THREADS;
            dist[o] = t[ty + r * THREADS][tx + c * THREADS];
            if (changed[r][c]) path[o] = p[r][c];
        }
}

// Phase 3: all remaining tiles (ib, jb), ib != kb, jb != kb. Independent tiles.
__global__ void __launch_bounds__(THREADS * THREADS)
phase3Kernel(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
             const unsigned int* __restrict__ pivRow, size_t ld, int kb, unsigned int kBase) {
    __shared__ unsigned int colT[TILE][TILE + 1];
    __shared__ unsigned int rowT[TILE][TILE];
    const int jb = blockIdx.x < (unsigned)kb ? blockIdx.x : blockIdx.x + 1;
    const int tx = threadIdx.x, ty = threadIdx.y;
    const size_t rowBase = (size_t)blockIdx.y * TILE * ld;
    unsigned int d[PER_T][PER_T];
    unsigned int p[PER_T][PER_T];
#pragma unroll
    for (int r = 0; r < PER_T; ++r)
#pragma unroll
        for (int c = 0; c < PER_T; ++c) {
            const size_t row = (size_t)(ty + r * THREADS) * ld;
            colT[ty + r * THREADS][tx + c * THREADS] = dist[rowBase + row + (size_t)kb * TILE + tx + c * THREADS];
            rowT[ty + r * THREADS][tx + c * THREADS] = pivRow[row + (size_t)jb * TILE + tx + c * THREADS];
            d[r][c] = dist[rowBase + row + (size_t)jb * TILE + tx + c * THREADS];
            p[r][c] = 0xFFFFFFFFu;
        }
    __syncthreads();
#pragma unroll 8
    for (int m = 0; m < TILE; ++m) {
        unsigned int a[PER_T], b[PER_T];
#pragma unroll
        for (int r = 0; r < PER_T; ++r) a[r] = colT[ty + r * THREADS][m];
#pragma unroll
        for (int c = 0; c < PER_T; ++c) b[c] = rowT[m][tx + c * THREADS];
#pragma unroll
        for (int r = 0; r < PER_T; ++r)
#pragma unroll
            for (int c = 0; c < PER_T; ++c) {
                const unsigned int s = a[r] + b[c];
                if (s < d[r][c]) {
                    d[r][c] = s;
                    p[r][c] = kBase + m;
                }
            }
    }
#pragma unroll
    for (int r = 0; r < PER_T; ++r)
#pragma unroll
        for (int c = 0; c < PER_T; ++c) {
            const size_t o = rowBase + (size_t)(ty + r * THREADS) * ld + (size_t)jb * TILE + tx + c * THREADS;
            if (p[r][c] != 0xFFFFFFFFu) {
                dist[o] = d[r][c];
                path[o] = p[r][c];
            }
        }
}

// ---------------------------------------------------------------------------
// Distributed blocked Floyd-Warshall (MPI block-row decomposition, one GPU per rank).
// On entry, rank 0 holds the full dist matrix; on exit, rank 0 holds the result.
// Path matrix is kept distributed on the GPUs (it is not needed by the host).
// ---------------------------------------------------------------------------
void floydWarshall(std::vector<unsigned int>& dist, const size_t numNodes,
                   int rank, int nranks, long& elapsedMs) {
    const size_t n = numNodes;
    const int nb = (int)((n + TILE - 1) / TILE);
    const size_t ld = (size_t)nb * TILE;

    // Block-row ownership
    std::vector<int> blkStart(nranks + 1);
    for (int r = 0; r <= nranks; ++r) blkStart[r] = (int)((long long)nb * r / nranks);
    std::vector<int> ownerOf(nb);
    for (int r = 0; r < nranks; ++r)
        for (int b = blkStart[r]; b < blkStart[r + 1]; ++b) ownerOf[b] = r;
    const int myB0 = blkStart[rank], myB1 = blkStart[rank + 1];
    const int myNB = myB1 - myB0;
    const size_t myRowsPad = (size_t)myNB * TILE;

    // Real (unpadded) rows owned by each rank
    auto realRows = [&](int r) -> int {
        const size_t a = std::min(n, (size_t)blkStart[r] * TILE);
        const size_t b = std::min(n, (size_t)blkStart[r + 1] * TILE);
        return (int)(b - a);
    };
    std::vector<int> rowCounts(nranks), rowDispls(nranks);
    for (int r = 0; r < nranks; ++r) {
        rowCounts[r] = realRows(r);
        rowDispls[r] = (int)std::min(n, (size_t)blkStart[r] * TILE);
    }
    const int myRealRows = rowCounts[rank];

    MPI_Datatype rowType;
    MPI_Type_contiguous((int)n, MPI_UNSIGNED, &rowType);
    MPI_Type_commit(&rowType);

    // Host staging buffer for local rows (pinned for fast transfers)
    unsigned int* hLocal = nullptr;
    CUDA_CHECK(cudaMallocHost(&hLocal, std::max<size_t>(1, (size_t)myRealRows * n) * sizeof(unsigned int)));

    MPI_Scatterv(rank == 0 ? dist.data() : nullptr, rowCounts.data(), rowDispls.data(), rowType,
                 hLocal, myRealRows, rowType, 0, MPI_COMM_WORLD);

    // Device allocations
    unsigned int *dDist = nullptr, *dPath = nullptr, *dRowBuf[2] = {nullptr, nullptr};
    const size_t localElems = std::max<size_t>(1, myRowsPad * ld);
    CUDA_CHECK(cudaMalloc(&dDist, localElems * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&dPath, localElems * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&dRowBuf[0], (size_t)TILE * ld * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&dRowBuf[1], (size_t)TILE * ld * sizeof(unsigned int)));
    const size_t rowBytes = (size_t)TILE * ld * sizeof(unsigned int);

    // Pivot-row exchange. Ranks on the same node share a double-buffered host window
    // (MPI-3 shared memory): the owner copies its pivot row from its GPU straight into
    // the window, other local ranks copy it to their GPUs. Across nodes, the row is
    // broadcast once per node among ranks with the same node-local rank.
    MPI_Comm nodeComm, crossComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int nodeRank = 0, nodeSize = 1;
    MPI_Comm_rank(nodeComm, &nodeRank);
    MPI_Comm_size(nodeComm, &nodeSize);
    MPI_Comm_split(MPI_COMM_WORLD, nodeRank, rank, &crossComm);
    int crossRank = 0, crossSize = 1;
    MPI_Comm_rank(crossComm, &crossRank);
    MPI_Comm_size(crossComm, &crossSize);
    int sizeMinMax[2] = {nodeSize, -nodeSize};
    MPI_Allreduce(MPI_IN_PLACE, sizeMinMax, 2, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    // Shared-window path requires every node to host the same number of ranks
    const bool useShm = nranks > 1 && sizeMinMax[0] == -sizeMinMax[1];
    std::vector<int> nodeRankOf(nranks), crossRankOf(nranks);
    MPI_Allgather(&nodeRank, 1, MPI_INT, nodeRankOf.data(), 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Allgather(&crossRank, 1, MPI_INT, crossRankOf.data(), 1, MPI_INT, MPI_COMM_WORLD);

    MPI_Win shmWin = MPI_WIN_NULL;
    unsigned int* hShm[2] = {nullptr, nullptr};
    unsigned int* hRow = nullptr;
    bool shmPinned = false;
    if (useShm) {
        void* base = nullptr;
        MPI_Win_allocate_shared(nodeRank == 0 ? (MPI_Aint)(2 * rowBytes) : 0, 1, MPI_INFO_NULL,
                                nodeComm, &base, &shmWin);
        MPI_Aint sz;
        int du;
        MPI_Win_shared_query(shmWin, 0, &sz, &du, &base);
        hShm[0] = static_cast<unsigned int*>(base);
        hShm[1] = hShm[0] + (size_t)TILE * ld;
        shmPinned = cudaHostRegister(base, 2 * rowBytes, cudaHostRegisterPortable) == cudaSuccess;
        if (!shmPinned) cudaGetLastError();
    } else {
        CUDA_CHECK(cudaMallocHost(&hRow, rowBytes));
    }
    auto hostRow = [&](int kb) { return useShm ? hShm[kb & 1] : hRow; };

    cudaStream_t sComp, sCopy;
    CUDA_CHECK(cudaStreamCreateWithFlags(&sComp, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&sCopy, cudaStreamNonBlocking));
    cudaEvent_t evRound[2], evCopy;
    for (auto& e : evRound) CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&evCopy, cudaEventDisableTiming));

    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::high_resolution_clock::now();

    // Upload: padding entries are INF (never improve any real distance, no overflow)
    if (myNB > 0) {
        if (ld != n || myRowsPad != (size_t)myRealRows) {
            // Build padded block on host in parallel, then upload in one go
            unsigned int* hPad = nullptr;
            CUDA_CHECK(cudaMallocHost(&hPad, myRowsPad * ld * sizeof(unsigned int)));
#pragma omp parallel for schedule(static)
            for (size_t i = 0; i < myRowsPad; ++i) {
                unsigned int* dst = hPad + i * ld;
                if (i < (size_t)myRealRows) {
                    memcpy(dst, hLocal + i * n, n * sizeof(unsigned int));
                    for (size_t j = n; j < ld; ++j) dst[j] = INF;
                } else {
                    for (size_t j = 0; j < ld; ++j) dst[j] = INF;
                }
            }
            CUDA_CHECK(cudaMemcpyAsync(dDist, hPad, myRowsPad * ld * sizeof(unsigned int),
                                       cudaMemcpyHostToDevice, sComp));
            CUDA_CHECK(cudaStreamSynchronize(sComp));
            CUDA_CHECK(cudaFreeHost(hPad));
        } else {
            CUDA_CHECK(cudaMemcpyAsync(dDist, hLocal, myRowsPad * ld * sizeof(unsigned int),
                                       cudaMemcpyHostToDevice, sComp));
        }
        initPathKernel<<<1024, 256, 0, sComp>>>(dPath, ld, myRowsPad, (unsigned int)(myB0 * TILE));
        CUDA_CHECK(cudaGetLastError());
    }

    const dim3 tpb(THREADS, THREADS);

    // Pivot step (owner only): phase 1 + phase 2 row on local block-row kb, then stage to host
    auto doPivot = [&](int kb) {
        const int lb = kb - myB0;
        unsigned int* blk = dDist + (size_t)lb * TILE * ld;
        unsigned int* pblk = dPath + (size_t)lb * TILE * ld;
        const unsigned int kBase = (unsigned int)(kb * TILE);
        phase1Kernel<<<1, tpb, 0, sComp>>>(blk + (size_t)kb * TILE, pblk + (size_t)kb * TILE, ld, kBase);
        if (nb > 1) phase2RowKernel<<<nb - 1, tpb, 0, sComp>>>(blk, pblk, ld, kb, kBase);
        CUDA_CHECK(cudaGetLastError());
        if (nranks > 1) {
            CUDA_CHECK(cudaMemcpyAsync(hostRow(kb), blk, rowBytes, cudaMemcpyDeviceToHost, sComp));
        }
        CUDA_CHECK(cudaStreamSynchronize(sComp));
    };

    // Update local block-rows [l0, l1) for round kb (phase 2 column + phase 3)
    auto updateRange = [&](int l0, int l1, int kb, const unsigned int* pivRow) {
        if (l1 <= l0) return;
        unsigned int* d = dDist + (size_t)l0 * TILE * ld;
        unsigned int* p = dPath + (size_t)l0 * TILE * ld;
        const unsigned int kBase = (unsigned int)(kb * TILE);
        phase2ColKernel<<<l1 - l0, tpb, 0, sComp>>>(d, p, pivRow, ld, kb, kBase);
        if (nb > 1) phase3Kernel<<<dim3(nb - 1, l1 - l0), tpb, 0, sComp>>>(d, p, pivRow, ld, kb, kBase);
        CUDA_CHECK(cudaGetLastError());
    };

    // Receive a broadcast pivot row into device buffer
    auto recvPivot = [&](int kb) {
        const int owner = ownerOf[kb];
        unsigned int* src = hostRow(kb);
        if (useShm) {
            if (crossSize > 1 && nodeRankOf[owner] == nodeRank)
                MPI_Bcast(src, TILE * (int)ld, MPI_UNSIGNED, crossRankOf[owner], crossComm);
            // Row is in the node window; also guarantees the buffer from two steps
            // ago has been consumed by all local ranks before it is overwritten.
            MPI_Barrier(nodeComm);
        } else {
            MPI_Bcast(src, TILE * (int)ld, MPI_UNSIGNED, owner, MPI_COMM_WORLD);
        }
        if (owner != rank) {
            // Buffer kb&1 was last read by round kb-2 kernels
            CUDA_CHECK(cudaStreamWaitEvent(sCopy, evRound[kb & 1], 0));
            CUDA_CHECK(cudaMemcpyAsync(dRowBuf[kb & 1], src, rowBytes, cudaMemcpyHostToDevice, sCopy));
            CUDA_CHECK(cudaEventRecord(evCopy, sCopy));
            CUDA_CHECK(cudaStreamWaitEvent(sComp, evCopy, 0));
            CUDA_CHECK(cudaStreamSynchronize(sCopy));  // host buffer may be reused afterwards
        }
    };

    if (nb > 0) {
        if (ownerOf[0] == rank) doPivot(0);
        if (nranks > 1) recvPivot(0);
    }

    for (int kb = 0; kb < nb; ++kb) {
        const bool ownsK = ownerOf[kb] == rank;
        const unsigned int* pivRow = ownsK ? dDist + (size_t)(kb - myB0) * TILE * ld : dRowBuf[kb & 1];
        const bool hasNext = kb + 1 < nb;
        const bool ownsNext = hasNext && ownerOf[kb + 1] == rank;

        // Look-ahead: finish block-row kb+1 first, compute next pivot early
        if (ownsNext) {
            const int ln = kb + 1 - myB0;
            updateRange(ln, ln + 1, kb, pivRow);
            doPivot(kb + 1);
        }

        // Remaining local block-rows (exclude pivot row kb and look-ahead row kb+1)
        int excl[2];
        int ne = 0;
        if (ownsK) excl[ne++] = kb - myB0;
        if (ownsNext) excl[ne++] = kb + 1 - myB0;
        int cur = 0;
        for (int e = 0; e < ne; ++e) {
            updateRange(cur, excl[e], kb, pivRow);
            cur = excl[e] + 1;
        }
        updateRange(cur, myNB, kb, pivRow);
        CUDA_CHECK(cudaEventRecord(evRound[kb & 1], sComp));

        // Broadcast next pivot row while GPU works on this round
        if (hasNext && nranks > 1) recvPivot(kb + 1);
    }
    CUDA_CHECK(cudaStreamSynchronize(sComp));

    // Download real part of local rows and gather on rank 0
    if (myRealRows > 0) {
        CUDA_CHECK(cudaMemcpy2D(hLocal, n * sizeof(unsigned int), dDist, ld * sizeof(unsigned int),
                                n * sizeof(unsigned int), myRealRows, cudaMemcpyDeviceToHost));
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const auto end = std::chrono::high_resolution_clock::now();
    elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // Collect result on rank 0 (rank 0 copies its own rows in parallel)
    if (rank == 0) {
#pragma omp parallel for schedule(static)
        for (int i = 0; i < myRealRows; ++i)
            memcpy(dist.data() + (size_t)i * n, hLocal + (size_t)i * n, n * sizeof(unsigned int));
        MPI_Gatherv(MPI_IN_PLACE, 0, rowType, dist.data(), rowCounts.data(), rowDispls.data(),
                    rowType, 0, MPI_COMM_WORLD);
    } else {
        MPI_Gatherv(hLocal, myRealRows, rowType, nullptr, rowCounts.data(), rowDispls.data(),
                    rowType, 0, MPI_COMM_WORLD);
    }

    for (auto& e : evRound) cudaEventDestroy(e);
    cudaEventDestroy(evCopy);
    cudaStreamDestroy(sComp);
    cudaStreamDestroy(sCopy);
    cudaFree(dDist);
    cudaFree(dPath);
    cudaFree(dRowBuf[0]);
    cudaFree(dRowBuf[1]);
    if (useShm) {
        if (shmPinned) cudaHostUnregister(hShm[0]);
        MPI_Win_free(&shmWin);
    } else {
        cudaFreeHost(hRow);
    }
    MPI_Comm_free(&crossComm);
    MPI_Comm_free(&nodeComm);
    cudaFreeHost(hLocal);
    MPI_Type_free(&rowType);
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
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    // One GPU per rank, assigned round-robin by node-local rank
    {
        MPI_Comm local;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
        int localRank = 0;
        MPI_Comm_rank(local, &localRank);
        MPI_Comm_free(&local);
        int ndev = 0;
        CUDA_CHECK(cudaGetDeviceCount(&ndev));
        if (ndev < 1) {
            fprintf(stderr, "No CUDA device available\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        CUDA_CHECK(cudaSetDevice(localRank % ndev));
    }

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
    
    // Full matrix only lives on rank 0; ranks hold their block-rows on the GPU
    std::vector<unsigned int> dist(rank == 0 ? numNodes * numNodes : 0);
    
    // Initialize (path matrix is initialized on the GPUs)
    if (rank == 0) {
        printf("Initializing graph...\n");
        initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
        printf("Computing shortest paths...\n");
        fflush(stdout);
    }
    
    // Run Floyd-Warshall
    long elapsedMs = 0;
    if (numNodes > 0) floydWarshall(dist, numNodes, rank, nranks, elapsedMs);
    
    int exitCode = 0;
    if (rank == 0) {
        printf("Computation time: %ld ms\n", elapsedMs);
        
        // Calculate operations per second
        // Floyd-Warshall has O(n³) complexity
        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / (elapsedMs / 1000.0) / 1e9;
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
