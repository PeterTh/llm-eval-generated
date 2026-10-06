#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
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

void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes, 
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;
    
    for (size_t i = 0; i < numNodes * numNodes; ++i) {
        dist[i] = rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
    }
    
    // Set diagonal to zero (distance from node to itself is 0)
    for (size_t i = 0; i < numNodes; ++i) {
        dist[idx2(i, i, numNodes)] = 0;
    }
}

// ---------------------------------------------------------------------------
// Hybrid MPI + OpenMP + CUDA blocked Floyd-Warshall
//
// Data layout: row-major dist[i][j] == dist[idx2(j, i, n)], identical to the
// original code. Rows are distributed across MPI ranks in contiguous groups
// of BK-row blocks; each rank drives one GPU. Columns are padded to a multiple
// of TC with a large "unreachable" value so that padded nodes never improve
// any real path.
//
// For every k-block (BK intermediate nodes) the owning rank finalizes the
// pivot row panel (phase 1 + phase 2 row), broadcasts it to all ranks, and
// every rank then updates its own rows (phase 2 column + phase 3). The next
// pivot panel is computed on a high-priority stream and broadcast while the
// bulk phase-3 update of the current k-block is still running (lookahead).
// ---------------------------------------------------------------------------

constexpr int BK = 32;                  // k-block size / row-block size
constexpr int TC = 128;                 // phase-3 tile width (columns)
constexpr unsigned int PADV = 1u << 20; // distance used for padding entries
constexpr int PSHIFT = 6;               // packing: (dist << 6) | (kk + 1)
constexpr int NSLOT = 4;                // node-shared panel staging slots
constexpr int TAG_NOTIFY = 101;         // "panel is in the shared slot"
constexpr int TAG_ACK = 102;            // "finished reading the shared slot"

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), \
                    __FILE__, __LINE__);                                          \
            MPI_Abort(MPI_COMM_WORLD, 1);                                         \
        }                                                                         \
    } while (0)

// Phase 1: diagonal tile (rows/cols k0..k0+BK) relaxed in place.
// Row kk and column kk never change during step kk (diagonal is zero), so a
// single barrier per step suffices.
__global__ void __launch_bounds__(BK * BK)
fwPhase1(unsigned int* __restrict__ d, unsigned int* __restrict__ p, size_t ld,
         int lr0, int k0) {
    __shared__ unsigned int t[BK][BK + 1];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const size_t g = (size_t)(lr0 + ty) * ld + k0 + tx;
    unsigned int v = d[g];
    unsigned int pv = 0;
    bool changed = false;
    t[ty][tx] = v;
    __syncthreads();
#pragma unroll 8
    for (int kk = 0; kk < BK; ++kk) {
        const unsigned int nd = t[ty][kk] + t[kk][tx];
        if (nd < v) {
            v = nd;
            pv = k0 + kk;
            changed = true;
            t[ty][tx] = v;
        }
        __syncthreads();
    }
    if (changed) {
        d[g] = v;
        p[g] = pv;
    }
}

// Phase 2 (row): tiles of the pivot row panel, using the finished diagonal tile.
__global__ void __launch_bounds__(BK * BK)
fwPhase2Row(unsigned int* __restrict__ d, unsigned int* __restrict__ p, size_t ld,
            int lr0, int k0) {
    const int jb = blockIdx.x;
    if (jb * BK == k0) return;
    __shared__ unsigned int sd[BK][BK + 1];
    __shared__ unsigned int t[BK][BK + 1];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const size_t rowOff = (size_t)(lr0 + ty) * ld;
    const size_t g = rowOff + (size_t)jb * BK + tx;
    sd[ty][tx] = d[rowOff + k0 + tx];
    unsigned int v = d[g];
    unsigned int pv = 0;
    bool changed = false;
    t[ty][tx] = v;
    __syncthreads();
#pragma unroll 8
    for (int kk = 0; kk < BK; ++kk) {
        const unsigned int nd = sd[ty][kk] + t[kk][tx];
        if (nd < v) {
            v = nd;
            pv = k0 + kk;
            changed = true;
            t[ty][tx] = v;
        }
        __syncthreads();
    }
    if (changed) {
        d[g] = v;
        p[g] = pv;
    }
}

// Phase 2 (column): tile (row block, pivot columns) of every local row block,
// using the diagonal tile from the pivot panel. Final values are also stored
// contiguously in colbuf for phase 3.
__global__ void __launch_bounds__(BK * BK)
fwPhase2Col(unsigned int* __restrict__ d, unsigned int* __restrict__ p, size_t ld,
            const unsigned int* __restrict__ panel, unsigned int* __restrict__ colbuf,
            int k0, int rb0, int skip0, int skip1) {
    const int rb = rb0 + blockIdx.x;
    if (rb == skip0 || rb == skip1) return;
    __shared__ unsigned int sd[BK][BK + 1];
    __shared__ unsigned int t[BK][BK + 1];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const size_t g = (size_t)(rb * BK + ty) * ld + k0 + tx;
    sd[ty][tx] = panel[(size_t)ty * ld + k0 + tx];
    unsigned int v = d[g];
    unsigned int pv = 0;
    bool changed = false;
    t[ty][tx] = v;
    __syncthreads();
#pragma unroll 8
    for (int kk = 0; kk < BK; ++kk) {
        const unsigned int nd = t[ty][kk] + sd[kk][tx];
        if (nd < v) {
            v = nd;
            pv = k0 + kk;
            changed = true;
            t[ty][tx] = v;
        }
        __syncthreads();
    }
    if (changed) {
        d[g] = v;
        p[g] = pv;
    }
    colbuf[((size_t)blockIdx.x * BK + ty) * BK + tx] = v;
}

// Phase 3: all remaining tiles of the local row blocks. Both operands are
// final for this k-block, so the sequential strict-less scan over kk equals
// "minimum, earliest kk on ties", computed with packed integers:
// candidate = ((dIK + dKJ) << 6) | (kk + 1), current = dIJ << 6 (wins ties).
__global__ void __launch_bounds__(256)
fwPhase3(unsigned int* __restrict__ d, unsigned int* __restrict__ p, size_t ld,
         const unsigned int* __restrict__ panel, const unsigned int* __restrict__ colbuf,
         int k0, int rb0, int skip0, int skip1) {
    const int rb = rb0 + blockIdx.y;
    if (rb == skip0 || rb == skip1) return;
    __shared__ __align__(16) unsigned int As[BK][BK + 4];
    __shared__ __align__(16) unsigned int Bs[BK][TC];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * 32 + tx;
    const int c0 = blockIdx.x * TC;

    {
        const int r = tid >> 3, kq = (tid & 7) * 4;
        uint4 a = *reinterpret_cast<const uint4*>(colbuf + ((size_t)blockIdx.y * BK + r) * BK + kq);
        a.x <<= PSHIFT; a.y <<= PSHIFT; a.z <<= PSHIFT; a.w <<= PSHIFT;
        *reinterpret_cast<uint4*>(&As[r][kq]) = a;
    }
#pragma unroll
    for (int t = 0; t < (BK * TC / 4) / 256; ++t) {
        const int idx = tid + 256 * t;
        const int kk = idx / (TC / 4), cq = (idx % (TC / 4)) * 4;
        uint4 b = *reinterpret_cast<const uint4*>(panel + (size_t)kk * ld + c0 + cq);
        const unsigned int tag = kk + 1;
        b.x = (b.x << PSHIFT) | tag; b.y = (b.y << PSHIFT) | tag;
        b.z = (b.z << PSHIFT) | tag; b.w = (b.w << PSHIFT) | tag;
        *reinterpret_cast<uint4*>(&Bs[kk][cq]) = b;
    }
    __syncthreads();

    const int col = c0 + tx * 4;
    if (col >= k0 && col < k0 + BK) return;  // pivot columns are done in phase 2

    unsigned int m[4][4];
    size_t rowOff[4];
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        rowOff[r] = (size_t)(rb * BK + ty + 8 * r) * ld + col;
        const uint4 v = *reinterpret_cast<const uint4*>(d + rowOff[r]);
        m[r][0] = v.x << PSHIFT; m[r][1] = v.y << PSHIFT;
        m[r][2] = v.z << PSHIFT; m[r][3] = v.w << PSHIFT;
    }

#pragma unroll
    for (int kk = 0; kk < BK; kk += 4) {
        uint4 a[4];
#pragma unroll
        for (int r = 0; r < 4; ++r) a[r] = *reinterpret_cast<const uint4*>(&As[ty + 8 * r][kk]);
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            const uint4 b = *reinterpret_cast<const uint4*>(&Bs[kk + s][tx * 4]);
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                const unsigned int av = s == 0 ? a[r].x : s == 1 ? a[r].y : s == 2 ? a[r].z : a[r].w;
                m[r][0] = min(m[r][0], av + b.x);
                m[r][1] = min(m[r][1], av + b.y);
                m[r][2] = min(m[r][2], av + b.z);
                m[r][3] = min(m[r][3], av + b.w);
            }
        }
    }

    constexpr unsigned int TAGMASK = (1u << PSHIFT) - 1;
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const unsigned int any = (m[r][0] | m[r][1] | m[r][2] | m[r][3]) & TAGMASK;
        if (any) {
            *reinterpret_cast<uint4*>(d + rowOff[r]) =
                make_uint4(m[r][0] >> PSHIFT, m[r][1] >> PSHIFT, m[r][2] >> PSHIFT, m[r][3] >> PSHIFT);
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const unsigned int tag = m[r][q] & TAGMASK;
                if (tag) p[rowOff[r] + q] = k0 + tag - 1;
            }
        }
    }
}

// Fill the local padded distance matrix with PADV (real entries are copied in
// afterwards) and initialize path[i][j] = i (as the original initializePathMatrix).
__global__ void fwInitLocal(unsigned int* __restrict__ d, unsigned int* __restrict__ p,
                            size_t ld, size_t rows, size_t rowStart) {
    const size_t total = rows * ld;
    for (size_t e = blockIdx.x * (size_t)blockDim.x + threadIdx.x; e < total;
         e += (size_t)gridDim.x * blockDim.x) {
        d[e] = PADV;
        p[e] = (unsigned int)(rowStart + e / ld);
    }
}

// Zero the diagonal of padding rows.
__global__ void fwPadDiag(unsigned int* __restrict__ d, size_t ld, size_t rows,
                          size_t rowStart, size_t numNodes) {
    const size_t r = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (r < rows && rowStart + r >= numNodes) d[r * ld + rowStart + r] = 0;
}

// Lossless transfer encoding. Every shortest distance is bounded by the largest
// edge weight (the graph is complete), so distances fit into 1, 2 or 4 bytes;
// with PAD the all-ones code represents padding entries (PADV).
template <typename T, bool PAD>
__global__ void fwEncode(const unsigned int* __restrict__ src, size_t ld, T* __restrict__ dst,
                         size_t rows, size_t cols) {
    for (size_t r = blockIdx.y; r < rows; r += gridDim.y)
        for (size_t c = blockIdx.x * (size_t)blockDim.x + threadIdx.x; c < cols; c += (size_t)gridDim.x * blockDim.x) {
            const unsigned int v = src[r * ld + c];
            dst[r * cols + c] = (PAD && v >= PADV) ? (T)~(T)0 : (T)v;
        }
}

template <typename T, bool PAD>
__global__ void fwDecode(const T* __restrict__ src, unsigned int* __restrict__ dst, size_t ld,
                         size_t rows, size_t cols) {
    for (size_t r = blockIdx.y; r < rows; r += gridDim.y)
        for (size_t c = blockIdx.x * (size_t)blockDim.x + threadIdx.x; c < cols; c += (size_t)gridDim.x * blockDim.x) {
            const T v = src[r * cols + c];
            dst[r * ld + c] = (PAD && v == (T)~(T)0) ? PADV : (unsigned int)v;
        }
}

static dim3 codecGrid(size_t rows, size_t cols) {
    return dim3((unsigned)std::min<size_t>((cols + 255) / 256, 64), (unsigned)std::min<size_t>(rows, 65535));
}

template <bool PAD>
void encodeRows(int w, const unsigned int* src, size_t ld, void* dst, size_t rows, size_t cols, cudaStream_t s) {
    const dim3 g = codecGrid(rows, cols);
    if (w == 1) fwEncode<uint8_t, PAD><<<g, 256, 0, s>>>(src, ld, (uint8_t*)dst, rows, cols);
    else if (w == 2) fwEncode<uint16_t, PAD><<<g, 256, 0, s>>>(src, ld, (uint16_t*)dst, rows, cols);
    else fwEncode<uint32_t, PAD><<<g, 256, 0, s>>>(src, ld, (uint32_t*)dst, rows, cols);
    CUDA_CHECK(cudaGetLastError());
}

template <bool PAD>
void decodeRows(int w, const void* src, unsigned int* dst, size_t ld, size_t rows, size_t cols, cudaStream_t s) {
    const dim3 g = codecGrid(rows, cols);
    if (w == 1) fwDecode<uint8_t, PAD><<<g, 256, 0, s>>>((const uint8_t*)src, dst, ld, rows, cols);
    else if (w == 2) fwDecode<uint16_t, PAD><<<g, 256, 0, s>>>((const uint16_t*)src, dst, ld, rows, cols);
    else fwDecode<uint32_t, PAD><<<g, 256, 0, s>>>((const uint32_t*)src, dst, ld, rows, cols);
    CUDA_CHECK(cudaGetLastError());
}

// Host-side (OpenMP) packing/unpacking of full matrices on rank 0.
template <typename T>
void packHostT(const unsigned int* src, T* dst, size_t count) {
#pragma omp parallel for schedule(static) if (count > (1u << 18))
    for (size_t e = 0; e < count; ++e) dst[e] = (T)src[e];
}
template <typename T>
void unpackHostT(const T* src, unsigned int* dst, size_t count) {
#pragma omp parallel for schedule(static) if (count > (1u << 18))
    for (size_t e = 0; e < count; ++e) dst[e] = src[e];
}
void packHost(int w, const unsigned int* src, void* dst, size_t count) {
    if (w == 1) packHostT(src, (uint8_t*)dst, count);
    else if (w == 2) packHostT(src, (uint16_t*)dst, count);
    else packHostT(src, (uint32_t*)dst, count);
}
void unpackHost(int w, const void* src, unsigned int* dst, size_t count) {
    if (w == 1) unpackHostT((const uint8_t*)src, dst, count);
    else if (w == 2) unpackHostT((const uint16_t*)src, dst, count);
    else unpackHostT((const uint32_t*)src, dst, count);
}

class FloydWarshallHybrid {
public:
    // codeBytes: bytes per distance used for transfers (see fwEncode).
    FloydWarshallHybrid(size_t numNodes, int rank, int size, int codeBytes)
        : n_(numNodes), rank_(rank), size_(size), w_(codeBytes) {
        nkb_ = (int)((n_ + BK - 1) / BK);
        ld_ = std::max<size_t>(((n_ + TC - 1) / TC) * TC, TC);
        start_.resize(size_ + 1);
        const int base = nkb_ / size_, rem = nkb_ % size_;
        start_[0] = 0;
        for (int r = 0; r < size_; ++r) start_[r + 1] = start_[r] + base + (r < rem ? 1 : 0);
        owner_.resize(nkb_);
        for (int r = 0; r < size_; ++r)
            for (int b = start_[r]; b < start_[r + 1]; ++b) owner_[b] = r;
        myBlocks_ = start_[rank_ + 1] - start_[rank_];
        myRowStart_ = (size_t)start_[rank_] * BK;
        myRows_ = (size_t)myBlocks_ * BK;

        int lo, hi;
        CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
        CUDA_CHECK(cudaStreamCreateWithPriority(&crit_, cudaStreamNonBlocking, hi));
        CUDA_CHECK(cudaStreamCreateWithPriority(&bulk_, cudaStreamNonBlocking, lo));

        const size_t elems = std::max<size_t>(myRows_ * ld_, 1);
        CUDA_CHECK(cudaMalloc(&d_, elems * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&p_, elems * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&colBulk_, std::max<size_t>(myRows_, 1) * BK * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&colCrit_, (size_t)BK * BK * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&panelCode_, (size_t)BK * ld_ * w_));
        CUDA_CHECK(cudaMalloc(&staging_, std::max<size_t>(numRealRows(rank_) * n_ * std::max(w_, pathCodeBytes(n_)), 1)));
        for (int s = 0; s < 2; ++s)
            CUDA_CHECK(cudaMalloc(&panelBuf_[s], (size_t)BK * ld_ * sizeof(unsigned int)));
        for (int s = 0; s < NSLOT; ++s)
            CUDA_CHECK(cudaEventCreateWithFlags(&slotDone_[s], cudaEventDisableTiming));
        panelReady_.resize(nkb_);
        bulkDone_.resize(nkb_);
        for (int k = 0; k < nkb_; ++k) {
            CUDA_CHECK(cudaEventCreateWithFlags(&panelReady_[k], cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&bulkDone_[k], cudaEventDisableTiming));
        }
        setupCommunication();
    }

    ~FloydWarshallHybrid() {
        for (auto e : panelReady_) cudaEventDestroy(e);
        for (auto e : bulkDone_) cudaEventDestroy(e);
        for (int s = 0; s < 2; ++s) cudaFree(panelBuf_[s]);
        for (int s = 0; s < NSLOT; ++s) cudaEventDestroy(slotDone_[s]);
        cudaFree(d_);
        cudaFree(p_);
        cudaFree(colBulk_);
        cudaFree(colCrit_);
        cudaFree(panelCode_);
        cudaFree(staging_);
        cudaStreamDestroy(crit_);
        cudaStreamDestroy(bulk_);
    }

    // Bytes per entry used to transfer the path matrix.
    static int pathCodeBytes(size_t numNodes) { return numNodes <= 65536 ? 2 : 4; }

    // Node-local communicator and the global ranks of its members.
    MPI_Comm nodeComm() const { return nodeComm_; }
    const std::vector<int>& nodeRanks() const { return nodeGlobal_; }

    // Release MPI resources (must be called before MPI_Finalize).
    void release() {
        if (slots_ == nullptr) return;
        cudaHostUnregister(slots_);
        MPI_Win_free(&win_);
        MPI_Comm_free(&nodeComm_);
        MPI_Comm_free(&crossComm_);
        slots_ = nullptr;
    }

    // Number of real (non-padding) rows owned by rank r and its first global row.
    size_t numRealRows(int r) const {
        const size_t s = (size_t)start_[r] * BK, e = std::min((size_t)start_[r + 1] * BK, n_);
        return e > s ? e - s : 0;
    }
    size_t rowStart(int r) const { return (size_t)start_[r] * BK; }

    // Upload this rank's real rows (row-major, numNodes columns, packed with
    // codeBytes per entry, host memory).
    void upload(const void* packedRows) {
        if (myRows_ == 0) return;
        fwInitLocal<<<1024, 256, 0, bulk_>>>(d_, p_, ld_, myRows_, myRowStart_);
        CUDA_CHECK(cudaGetLastError());
        const size_t real = numRealRows(rank_);
        if (real > 0) {
            CUDA_CHECK(cudaMemcpyAsync(staging_, packedRows, real * n_ * w_, cudaMemcpyHostToDevice, bulk_));
            decodeRows<false>(w_, staging_, d_, ld_, real, n_, bulk_);
        }
        fwPadDiag<<<(unsigned)((myRows_ + 255) / 256), 256, 0, bulk_>>>(d_, ld_, myRows_, myRowStart_, n_);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(bulk_));
    }

    // Download this rank's real rows of the distance (codeBytes per entry) or
    // path (bytes per entry given) matrix into packed host memory.
    void downloadDist(void* packedRows) { downloadPacked(d_, w_, packedRows); }
    void downloadPath(void* packedRows, int bytes) { downloadPacked(p_, bytes, packedRows); }

    void run() {
        if (nkb_ == 0) return;
        // Prologue: pivot panel 0.
        if (owner_[0] == rank_) computePivotPanel(0, -1);
        distributePanel(0);
        for (int kb = 0; kb < nkb_; ++kb) {
            const bool hasNext = kb + 1 < nkb_;
            // Bulk update of local rows with panel kb, excluding pivot block kb
            // and (if owned) the next pivot block, which goes on the crit stream.
            const int skip0 = owner_[kb] == rank_ ? kb - start_[rank_] : -1;
            const int skip1 = (hasNext && owner_[kb + 1] == rank_) ? kb + 1 - start_[rank_] : -1;
            CUDA_CHECK(cudaStreamWaitEvent(bulk_, panelReady_[kb], 0));
            updateRows(bulk_, kb, 0, myBlocks_, skip0, skip1, colBulk_);
            CUDA_CHECK(cudaEventRecord(bulkDone_[kb], bulk_));
            if (hasNext) {
                // Lookahead: next pivot panel, overlapping with the bulk update.
                if (owner_[kb + 1] == rank_) computePivotPanel(kb + 1, kb);
                else if (kb >= 1) CUDA_CHECK(cudaStreamWaitEvent(crit_, bulkDone_[kb - 1], 0));
                distributePanel(kb + 1);
            }
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        if (!sendReqs_.empty()) MPI_Waitall((int)sendReqs_.size(), sendReqs_.data(), MPI_STATUSES_IGNORE);
        sendReqs_.clear();
    }

private:
    void downloadPacked(const unsigned int* src, int bytes, void* packedRows) {
        const size_t real = numRealRows(rank_);
        if (real == 0) return;
        encodeRows<false>(bytes, src, ld_, staging_, real, n_, bulk_);
        CUDA_CHECK(cudaMemcpyAsync(packedRows, staging_, real * n_ * bytes, cudaMemcpyDeviceToHost, bulk_));
        CUDA_CHECK(cudaStreamSynchronize(bulk_));
    }

    const unsigned int* panelPtr(int kb) const {
        if (owner_[kb] == rank_) return d_ + (size_t)(kb - start_[rank_]) * BK * ld_;
        return panelBuf_[kb & 1];
    }

    // Phase 2 column + phase 3 for local row blocks [rb0, rb0 + nb) using panel kb.
    void updateRows(cudaStream_t s, int kb, int rb0, int nb, int skip0, int skip1, unsigned int* colbuf) {
        if (nb <= 0) return;
        const int k0 = kb * BK;
        const unsigned int* panel = panelPtr(kb);
        fwPhase2Col<<<nb, dim3(BK, BK), 0, s>>>(d_, p_, ld_, panel, colbuf, k0, rb0, skip0, skip1);
        CUDA_CHECK(cudaGetLastError());
        fwPhase3<<<dim3((unsigned)(ld_ / TC), nb), dim3(32, 8), 0, s>>>(d_, p_, ld_, panel, colbuf, k0, rb0,
                                                                         skip0, skip1);
        CUDA_CHECK(cudaGetLastError());
    }

    // On the owner: bring pivot block kb up to date w.r.t. panel prevKb (if any),
    // then run phases 1 and 2 (row) for k-block kb on the critical stream.
    void computePivotPanel(int kb, int prevKb) {
        const int lb = kb - start_[rank_];
        if (prevKb >= 0) {
            if (prevKb >= 1) CUDA_CHECK(cudaStreamWaitEvent(crit_, bulkDone_[prevKb - 1], 0));
            CUDA_CHECK(cudaStreamWaitEvent(crit_, panelReady_[prevKb], 0));
            updateRows(crit_, prevKb, lb, 1, -1, -1, colCrit_);
        }
        const int k0 = kb * BK;
        const int lr0 = lb * BK;
        fwPhase1<<<1, dim3(BK, BK), 0, crit_>>>(d_, p_, ld_, lr0, k0);
        CUDA_CHECK(cudaGetLastError());
        fwPhase2Row<<<(unsigned)(ld_ / BK), dim3(BK, BK), 0, crit_>>>(d_, p_, ld_, lr0, k0);
        CUDA_CHECK(cudaGetLastError());
    }

    // Panel distribution topology. If all nodes run the same number of ranks,
    // ranks on a node share NSLOT staging slots in an MPI-3 shared-memory
    // window (registered with CUDA for fast D2H/H2D), and the panel crosses
    // nodes once per node via MPI_Bcast among ranks with equal node-local
    // rank. Otherwise every rank is its own "node" and MPI_Bcast is used.
    void setupCommunication() {
        MPI_Comm shm;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank_, MPI_INFO_NULL, &shm);
        int shmSize;
        MPI_Comm_size(shm, &shmSize);
        int minSize, maxSize;
        MPI_Allreduce(&shmSize, &minSize, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&shmSize, &maxSize, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        if (minSize == maxSize) {
            nodeComm_ = shm;
        } else {
            MPI_Comm_free(&shm);
            MPI_Comm_dup(MPI_COMM_SELF, &nodeComm_);
        }
        MPI_Comm_rank(nodeComm_, &localRank_);
        MPI_Comm_size(nodeComm_, &localSize_);
        MPI_Comm_split(MPI_COMM_WORLD, localRank_, rank_, &crossComm_);
        MPI_Comm_size(crossComm_, &crossSize_);
        int crossRank;
        MPI_Comm_rank(crossComm_, &crossRank);
        localRankOf_.resize(size_);
        crossRankOf_.resize(size_);
        nodeGlobal_.resize(localSize_);
        MPI_Allgather(&localRank_, 1, MPI_INT, localRankOf_.data(), 1, MPI_INT, MPI_COMM_WORLD);
        MPI_Allgather(&crossRank, 1, MPI_INT, crossRankOf_.data(), 1, MPI_INT, MPI_COMM_WORLD);
        MPI_Allgather(&rank_, 1, MPI_INT, nodeGlobal_.data(), 1, MPI_INT, nodeComm_);

        slotBytes_ = (size_t)BK * ld_ * w_;
        const size_t total = NSLOT * slotBytes_;
        void* base;
        MPI_Win_allocate_shared(localRank_ == 0 ? (MPI_Aint)total : 0, 1, MPI_INFO_NULL, nodeComm_, &base, &win_);
        MPI_Aint sz;
        int disp;
        MPI_Win_shared_query(win_, 0, &sz, &disp, &base);
        slots_ = static_cast<unsigned char*>(base);
        CUDA_CHECK(cudaHostRegister(slots_, total, cudaHostRegisterDefault));
    }

    // Local rank that writes panel kb into this node's shared slot.
    int writerOf(int kb) const { return localRankOf_[owner_[kb]]; }

    // Acknowledge consumed slots: mandatory for panels whose slot is needed by
    // panel <= kb, opportunistic for already finished copies.
    void flushAcks(int kb) {
        while (!pending_.empty()) {
            const int j = pending_.front();
            if (j + NSLOT >= nkb_) {  // slot never reused
                pending_.pop_front();
                continue;
            }
            if (j + NSLOT <= kb) {
                CUDA_CHECK(cudaEventSynchronize(slotDone_[j % NSLOT]));
            } else {
                const cudaError_t q = cudaEventQuery(slotDone_[j % NSLOT]);
                if (q == cudaErrorNotReady) break;
                CUDA_CHECK(q);
            }
            const int w = writerOf(j + NSLOT);
            if (w != localRank_) {
                sendReqs_.emplace_back();
                MPI_Isend(nullptr, 0, MPI_BYTE, w, TAG_ACK, nodeComm_, &sendReqs_.back());
            }
            pending_.pop_front();
        }
    }

    // Deliver panel kb from its owner to all ranks and record panelReady_[kb]
    // on the critical stream.
    void distributePanel(int kb) {
        const int owner = owner_[kb];
        if (size_ > 1) {
            const int slot = kb % NSLOT;
            unsigned char* sb = slots_ + slot * slotBytes_;
            flushAcks(kb);
            if (localRank_ == writerOf(kb)) {
                // Slot reuse: every other consumer of panel kb - NSLOT must be done.
                if (kb >= NSLOT) {
                    for (int lr = 0; lr < localSize_; ++lr)
                        if (lr != localRank_ && nodeGlobal_[lr] != owner_[kb - NSLOT])
                            MPI_Recv(nullptr, 0, MPI_BYTE, lr, TAG_ACK, nodeComm_, MPI_STATUS_IGNORE);
                }
                if (owner == rank_) {
                    encodeRows<true>(w_, panelPtr(kb), ld_, panelCode_, BK, ld_, crit_);
                    CUDA_CHECK(cudaMemcpyAsync(sb, panelCode_, slotBytes_, cudaMemcpyDeviceToHost, crit_));
                    CUDA_CHECK(cudaStreamSynchronize(crit_));
                }
                if (crossSize_ > 1) MPI_Bcast(sb, (int)slotBytes_, MPI_BYTE, crossRankOf_[owner], crossComm_);
                for (int lr = 0; lr < localSize_; ++lr) {
                    if (lr == localRank_) continue;
                    sendReqs_.emplace_back();
                    MPI_Isend(nullptr, 0, MPI_BYTE, lr, TAG_NOTIFY, nodeComm_, &sendReqs_.back());
                }
            } else {
                MPI_Recv(nullptr, 0, MPI_BYTE, writerOf(kb), TAG_NOTIFY, nodeComm_, MPI_STATUS_IGNORE);
            }
            if (owner != rank_) {
                CUDA_CHECK(cudaMemcpyAsync(panelCode_, sb, slotBytes_, cudaMemcpyHostToDevice, crit_));
                CUDA_CHECK(cudaEventRecord(slotDone_[slot], crit_));
                pending_.push_back(kb);
                decodeRows<true>(w_, panelCode_, panelBuf_[kb & 1], ld_, BK, ld_, crit_);
            }
        }
        CUDA_CHECK(cudaEventRecord(panelReady_[kb], crit_));
    }

    size_t n_;
    int rank_, size_;
    int w_;
    int nkb_;
    size_t ld_;
    std::vector<int> start_, owner_;
    int myBlocks_;
    size_t myRowStart_, myRows_;
    cudaStream_t crit_, bulk_;
    unsigned int *d_ = nullptr, *p_ = nullptr, *colBulk_ = nullptr, *colCrit_ = nullptr;
    unsigned int* panelBuf_[2];
    void* panelCode_ = nullptr;
    void* staging_ = nullptr;  // packed upload/download buffer
    cudaEvent_t slotDone_[NSLOT];
    MPI_Comm nodeComm_ = MPI_COMM_NULL, crossComm_ = MPI_COMM_NULL;
    int localRank_ = 0, localSize_ = 1, crossSize_ = 1;
    std::vector<int> localRankOf_, crossRankOf_, nodeGlobal_;
    MPI_Win win_ = MPI_WIN_NULL;
    unsigned char* slots_ = nullptr;
    size_t slotBytes_ = 0;
    std::deque<int> pending_;
    std::vector<MPI_Request> sendReqs_;
    std::vector<cudaEvent_t> panelReady_, bulkDone_;
};

bool validateResult(const std::vector<unsigned int>& dist, const size_t numNodes) {
    // Basic sanity checks (checked in parallel with OpenMP; on failure the
    // first offending element is reported in the original sequential order)
    
    // 1. Diagonal should be zero
    bool diagOk = true;
#pragma omp parallel for reduction(&& : diagOk) schedule(static)
    for (size_t i = 0; i < numNodes; ++i) {
        diagOk = diagOk && dist[idx2(i, i, numNodes)] == 0;
    }
    if (!diagOk) {
        for (size_t i = 0; i < numNodes; ++i) {
            if (dist[idx2(i, i, numNodes)] != 0) {
                printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", i, i);
                return false;
            }
        }
    }
    
    // 2. Triangle inequality: dist[i][k] + dist[k][j] >= dist[i][j]
    // Check a sample of paths to avoid O(n^3) validation time
    const size_t m = std::min(numNodes, static_cast<size_t>(10));
    auto violated = [&](size_t i, size_t j, size_t k) {
        const unsigned int distIJ = dist[idx2(j, i, numNodes)];
        const unsigned int distIK = dist[idx2(k, i, numNodes)];
        const unsigned int distKJ = dist[idx2(j, k, numNodes)];
        // Check for overflow before addition
        return distIK < INF && distKJ < INF && distIK + distKJ < distIJ;
    };
    bool triOk = true;
#pragma omp parallel for collapse(2) reduction(&& : triOk) schedule(static)
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < m; ++j) {
            for (size_t k = 0; k < numNodes; ++k) {
                if (violated(i, j, k)) {
                    triOk = false;
                    break;
                }
            }
        }
    }
    if (!triOk) {
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < m; ++j) {
                for (size_t k = 0; k < numNodes; ++k) {
                    if (violated(i, j, k)) {
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
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // One GPU per rank, chosen by node-local rank.
    {
        MPI_Comm local;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
        int localRank = 0;
        MPI_Comm_rank(local, &localRank);
        MPI_Comm_free(&local);
        int numDevices = 0;
        CUDA_CHECK(cudaGetDeviceCount(&numDevices));
        if (numDevices == 0) {
            fprintf(stderr, "No CUDA device available\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        CUDA_CHECK(cudaSetDevice(localRank % numDevices));
        CUDA_CHECK(cudaFree(nullptr));
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
    
    // Full matrices live on rank 0 only
    std::vector<unsigned int> dist(rank == 0 ? numNodes * numNodes : 0);
    std::vector<unsigned int> path(rank == 0 ? numNodes * numNodes : 0);
    
    // Initialize (sequential RNG stream, identical to the original)
    int codeBytes = 4;
    if (rank == 0) {
        printf("Initializing graph...\n");
        initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
        // All shortest distances are bounded by the largest edge weight; this
        // selects the lossless transfer encoding width and guarantees the
        // packed-min kernels stay far below PADV.
        unsigned int maxVal = 0;
        const size_t total = numNodes * numNodes;
#pragma omp parallel for reduction(max : maxVal) schedule(static)
        for (size_t e = 0; e < total; ++e) maxVal = std::max(maxVal, dist[e]);
        if (maxVal >= PADV / 2) {
            fprintf(stderr, "Edge weights too large for this implementation\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        codeBytes = maxVal < 0xFFu ? 1 : maxVal < 0xFFFFu ? 2 : 4;
    }
    MPI_Bcast(&codeBytes, 1, MPI_INT, 0, MPI_COMM_WORLD);
    const int pathBytes = FloydWarshallHybrid::pathCodeBytes(numNodes);

    FloydWarshallHybrid fw(numNodes, rank, size, codeBytes);

    // Row datatypes (packed) and per-rank row counts for scatter/gather
    MPI_Datatype distRowType, pathRowType;
    MPI_Type_contiguous((int)(std::max<size_t>(numNodes, 1) * codeBytes), MPI_BYTE, &distRowType);
    MPI_Type_contiguous((int)(std::max<size_t>(numNodes, 1) * pathBytes), MPI_BYTE, &pathRowType);
    MPI_Type_commit(&distRowType);
    MPI_Type_commit(&pathRowType);
    const size_t myReal = fw.numRealRows(rank);

    // Rank 0's packed full matrix lives in a node-shared MPI window: ranks on
    // rank 0's node copy their rows to/from it directly, ranks on other nodes
    // use point-to-point MPI.
    MPI_Comm nodeComm = fw.nodeComm();
    int onRootNode = rank == 0;
    MPI_Allreduce(MPI_IN_PLACE, &onRootNode, 1, MPI_INT, MPI_MAX, nodeComm);
    std::vector<int> remoteRanks;  // ranks not on rank 0's node (used by rank 0)
    if (rank == 0) {
        const auto& local = fw.nodeRanks();
        for (int r = 0; r < size; ++r)
            if (std::find(local.begin(), local.end(), r) == local.end()) remoteRanks.push_back(r);
    }
    const size_t fullBytes = numNodes * numNodes * std::max(codeBytes, pathBytes);
    MPI_Win fullWin;
    unsigned char* full = nullptr;
    {
        void* base;
        MPI_Win_allocate_shared(rank == 0 ? (MPI_Aint)std::max<size_t>(fullBytes, 1) : 0, 1, MPI_INFO_NULL,
                                nodeComm, &base, &fullWin);
        MPI_Aint sz;
        int disp;
        MPI_Win_shared_query(fullWin, 0, &sz, &disp, &base);
        full = static_cast<unsigned char*>(base);
        if (rank == 0) {  // fault in the pages before timing
#pragma omp parallel for schedule(static)
            for (size_t off = 0; off < fullBytes; off += 4096) full[off] = 0;
        }
    }
    std::vector<unsigned char> localBuf(onRootNode ? 0 : myReal * numNodes * std::max(codeBytes, pathBytes));

    // Page-lock the host range this rank copies to/from for fast DMA transfers.
    void* pinnedPtr = nullptr;
    if (myReal > 0) {
        size_t pinnedBytes;
        if (onRootNode) {
            const size_t lo = fw.rowStart(rank) * numNodes * std::min(codeBytes, pathBytes);
            const size_t hi = (fw.rowStart(rank) + myReal) * numNodes * std::max(codeBytes, pathBytes);
            pinnedPtr = full + lo;
            pinnedBytes = hi - lo;
        } else {
            pinnedPtr = localBuf.data();
            pinnedBytes = localBuf.size();
        }
        CUDA_CHECK(cudaHostRegister(pinnedPtr, pinnedBytes, cudaHostRegisterDefault));
    }

    // Collect every rank's real rows into rank 0's shared buffer (packed with
    // `bytes` per entry); `download` fills the given row buffer of this rank.
    auto gatherRows = [&](int bytes, MPI_Datatype rowType, auto&& download) {
        if (onRootNode) {
            download(full + fw.rowStart(rank) * numNodes * bytes);
        } else {
            download(localBuf.data());
            MPI_Send(localBuf.data(), (int)myReal, rowType, 0, 0, MPI_COMM_WORLD);
        }
        if (rank == 0) {
            for (int r : remoteRanks)
                MPI_Recv(full + fw.rowStart(r) * numNodes * bytes, (int)fw.numRealRows(r), rowType, r, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (onRootNode) MPI_Barrier(nodeComm);
    };

    // Run Floyd-Warshall
    if (rank == 0) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    // Distribute the (packed) input rows
    if (rank == 0) {
        packHost(codeBytes, dist.data(), full, numNodes * numNodes);
        std::vector<MPI_Request> reqs(remoteRanks.size());
        for (size_t i = 0; i < remoteRanks.size(); ++i) {
            const int r = remoteRanks[i];
            MPI_Isend(full + fw.rowStart(r) * numNodes * codeBytes, (int)fw.numRealRows(r), distRowType, r, 0,
                      MPI_COMM_WORLD, &reqs[i]);
        }
        MPI_Barrier(nodeComm);
        fw.upload(full);
        MPI_Waitall((int)reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);
    } else if (onRootNode) {
        MPI_Barrier(nodeComm);
        fw.upload(full + fw.rowStart(rank) * numNodes * codeBytes);
    } else {
        MPI_Recv(localBuf.data(), (int)myReal, distRowType, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        fw.upload(localBuf.data());
    }

    fw.run();

    // Collect the distance matrix on rank 0
    if (onRootNode) MPI_Barrier(nodeComm);  // everybody finished reading the input
    gatherRows(codeBytes, distRowType, [&](void* buf) { fw.downloadDist(buf); });
    if (rank == 0) unpackHost(codeBytes, full, dist.data(), numNodes * numNodes);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // The path matrix is not part of the reported results; collect it on rank 0
    // after timing (as in the original, where it stays in host memory).
    if (onRootNode) MPI_Barrier(nodeComm);  // rank 0 finished unpacking distances
    gatherRows(pathBytes, pathRowType, [&](void* buf) { fw.downloadPath(buf, pathBytes); });
    if (rank == 0) unpackHost(pathBytes, full, path.data(), numNodes * numNodes);

    if (pinnedPtr) CUDA_CHECK(cudaHostUnregister(pinnedPtr));
    MPI_Win_free(&fullWin);
    fw.release();
    MPI_Type_free(&distRowType);
    MPI_Type_free(&pathRowType);
    
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
