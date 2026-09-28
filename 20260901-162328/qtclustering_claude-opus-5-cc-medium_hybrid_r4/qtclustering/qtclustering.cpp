// QT Clustering Benchmark - Hybrid MPI + OpenMP + CUDA Version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// The result is identical to the sequential reference: every round the same
// seed wins (maximum cardinality, smallest point index on ties) and the same
// members are added in the same order.  Three observations make that possible
// at a fraction of the cost:
//
//   1. A point can only ever join the cluster of seed s if it lies within
//      `threshold` of s, because s itself is a member.  Candidate sets are
//      therefore the (static) neighbour lists of the seeds.
//   2. "Maximum distance to all current members" is a running maximum, so it
//      can be updated incrementally instead of rescanning the members, and a
//      candidate that once exceeded the threshold is dead forever.
//   3. The growth of a seed only reads the clustered flags of its own
//      neighbourhood.  After a cluster is accepted only the seeds that have one
//      of its members in their neighbourhood have to be re-evaluated; all other
//      cardinalities stay exactly valid and are cached.
//
// Parallelization:
//   * CUDA  : one thread block grows the candidate cluster of one seed.  The
//             block keeps the running maximum per candidate in shared memory,
//             fuses the incremental update with the block wide arg-min that
//             selects the next member, and compacts the candidate array once
//             half of it has died.  The neighbour lists themselves are built on
//             the GPU with a uniform spatial grid, and the per round
//             bookkeeping (clustered flags, invalidation) runs there as well.
//   * MPI   : the seeds that need re-evaluation in a round are split evenly
//             over the ranks, each rank drives its own GPU, and the resulting
//             cardinalities are replicated with a single all-gather, so every
//             rank picks the same winner without further communication.
//   * OpenMP: host side loops (cardinality bookkeeping, the scan for the best
//             seed, cluster validation).
//
#include <algorithm>
#include <chrono>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

static const double MAX_WIDTH = 20.0;
static const double MAX_HEIGHT = 20.0;

// Structure to represent a point in 2D space
struct Point {
    double x, y;
};

// Structure to represent a cluster
struct Cluster {
    std::vector<int> members;
    int seed_point;
};

// ---------------------------------------------------------------------------
// CUDA helpers
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                                  \
    do {                                                                                  \
        const cudaError_t err_ = (call);                                                  \
        if (err_ != cudaSuccess) {                                                        \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_),          \
                    __FILE__, __LINE__);                                                  \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                 \
        }                                                                                 \
    } while (0)

// One block grows the candidate cluster of one seed point.
//
//   md[] holds, per surviving candidate, the maximum squared distance to any
//   current cluster member; QT_DEAD marks a candidate that can never be added
//   any more (already clustered, already a member, or maximum distance >=
//   threshold - that value only ever grows).  Because sqrt() is monotonic,
//   max(sqrt(d2)) == sqrt(max(d2)), so both the selected point and the
//   accept/reject decisions are those of the sequential formulation.
//
//   Each thread permanently owns the strided slice i = tid, tid+BS, ... of the
//   active array, so the incremental update and the arg-min scan fuse into a
//   single pass and a single barrier per added member is enough.  Once half of
//   the candidates have died the array is compacted in place, which makes the
//   cost of an iteration proportional to the surviving set instead of the
//   initial one.
#define QT_DEAD DBL_MAX

// Admission test.  The sequential code tests sqrt(m) < threshold; comparing the
// squared values is equivalent and avoids the (on many GPUs very expensive)
// double precision square root.  Only inside the vanishingly narrow band where
// the rounding of the two formulations could disagree is the exact test used.
__device__ __forceinline__ bool qtAdmissible(const double m, const double t2lo, const double t2hi,
                                             const double threshold) {
    if (m < t2lo) return true;
    if (m > t2hi) return false;
    return sqrt(m) < threshold;
}

// Reduce (best distance, best point index) and the number of survivors over the
// whole block.  Tie breaking is by smallest point index, exactly as in the
// sequential scan over ascending candidate indices.  For a single warp block no
// barrier at all is needed; otherwise the per-warp results are combined
// redundantly by every thread, which costs one barrier instead of three.
template <int BS>
__device__ __forceinline__ void blockReduce(double& bd, int& bi, int& nalive, double* sd, int* si,
                                            int* sn, const int buf) {
    const unsigned mask = 0xffffffffu;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const double od = __shfl_down_sync(mask, bd, off);
        const int oi = __shfl_down_sync(mask, bi, off);
        nalive += __shfl_down_sync(mask, nalive, off);
        if (od < bd || (od == bd && oi < bi)) {
            bd = od;
            bi = oi;
        }
    }

    if (BS == 32) {
        bd = __shfl_sync(mask, bd, 0);
        bi = __shfl_sync(mask, bi, 0);
        nalive = __shfl_sync(mask, nalive, 0);
        return;
    }

    constexpr int NW = BS / 32;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) {
        sd[buf * NW + warp] = bd;
        si[buf * NW + warp] = bi;
        sn[buf * NW + warp] = nalive;
    }
    __syncthreads();

    bd = sd[buf * NW];
    bi = si[buf * NW];
    nalive = sn[buf * NW];
#pragma unroll
    for (int w = 1; w < NW; ++w) {
        const double od = sd[buf * NW + w];
        const int oi = si[buf * NW + w];
        nalive += sn[buf * NW + w];
        if (od < bd || (od == bd && oi < bi)) {
            bd = od;
            bi = oi;
        }
    }
}

template <int BS>
__global__ __launch_bounds__(BS) void growClusterKernel(
    const Point* __restrict__ pts, const long long* __restrict__ nbrOff,
    const int* __restrict__ nbrList, const unsigned char* __restrict__ clustered,
    const int* __restrict__ seeds, const int numSeeds, const double threshold, const double t2lo,
    const double t2hi, int* __restrict__ card, int* __restrict__ membersOut, const int memberStride,
    double* __restrict__ md2Global, int* __restrict__ idxGlobal, const long long mdStride,
    const int useShared) {
    const int b = blockIdx.x;
    if (b >= numSeeds) return;

    const int seed = seeds[b];
    const long long off = nbrOff[seed];
    const int cnt = static_cast<int>(nbrOff[seed + 1] - off);

    constexpr int NW = (BS > 32) ? BS / 32 : 1;
    extern __shared__ char smem[];
    __shared__ double sd[2 * NW];
    __shared__ int si[2 * NW];
    __shared__ int sn[2 * NW];
    __shared__ int scur;

    double* md;
    int* cl;  // candidate point indices, compacted in place
    if (useShared) {
        md = reinterpret_cast<double*>(smem);
        cl = reinterpret_cast<int*>(md + cnt);
    } else {
        md = md2Global + static_cast<long long>(b) * mdStride;
        cl = idxGlobal + static_cast<long long>(b) * mdStride;
    }

    const Point sp = pts[seed];
    const int tid = threadIdx.x;

    int* mem = (membersOut != nullptr) ? membersOut + static_cast<long long>(b) * memberStride
                                       : nullptr;
    if (mem != nullptr && tid == 0) mem[0] = seed;
    if (tid == 0) scur = 0;
    int nmem = 1;

    // Initial pass: distance to the seed, plus the first arg-min candidate.
    double bd = DBL_MAX;
    int bi = INT_MAX;
    int nalive = 0;
    for (int i = tid; i < cnt; i += BS) {
        const int p = nbrList[off + i];
        cl[i] = p;
        if (clustered[p]) {
            md[i] = QT_DEAD;
            continue;
        }
        const Point q = pts[p];
        const double dx = q.x - sp.x;
        const double dy = q.y - sp.y;
        const double m = dx * dx + dy * dy;
        if (qtAdmissible(m, t2lo, t2hi, threshold)) {
            md[i] = m;
            ++nalive;
            if (m < bd || (m == bd && p < bi)) {
                bd = m;
                bi = p;
            }
        } else {
            md[i] = QT_DEAD;
        }
    }
    int act = cnt;
    int buf = 0;

    while (true) {
        blockReduce<BS>(bd, bi, nalive, sd, si, sn, buf);
        buf ^= 1;
        if (bi == INT_MAX) break;  // no admissible point left

        const int winner = bi;
        if (mem != nullptr && tid == 0) mem[nmem] = winner;
        ++nmem;

        // Compact once the array is at most half alive (amortized O(cnt)).
        if (nalive * 2 <= act) {
            if (BS == 32) {
                const int lane = threadIdx.x;
                int cursor = 0;
                for (int base = 0; base < act; base += 32) {
                    const int i = base + lane;
                    const bool alive = (i < act) && (md[i] != QT_DEAD);
                    double m = 0.0;
                    int p = 0;
                    if (alive) {
                        m = md[i];
                        p = cl[i];
                    }
                    const unsigned bal = __ballot_sync(0xffffffffu, alive);
                    const int pos = cursor + __popc(bal & ((1u << lane) - 1u));
                    __syncwarp();
                    if (alive) {
                        md[pos] = m;
                        cl[pos] = p;
                    }
                    cursor += __popc(bal);
                    __syncwarp();
                }
            } else {
                for (int base = 0; base < act; base += BS) {
                    const int i = base + tid;
                    const bool alive = (i < act) && (md[i] != QT_DEAD);
                    double m = 0.0;
                    int p = 0;
                    if (alive) {
                        m = md[i];
                        p = cl[i];
                    }
                    __syncthreads();  // all reads of this tile are done
                    if (alive) {
                        const int pos = atomicAdd(&scur, 1);
                        md[pos] = m;
                        cl[pos] = p;
                    }
                    __syncthreads();  // writes done before the next tile is read
                }
                if (tid == 0) scur = 0;
            }
            act = nalive;
        }

        const Point w = pts[winner];
        bd = DBL_MAX;
        bi = INT_MAX;
        nalive = 0;
        for (int i = tid; i < act; i += BS) {
            double m = md[i];
            if (m == QT_DEAD) continue;
            const int p = cl[i];
            if (p == winner) {  // now a member of this cluster
                md[i] = QT_DEAD;
                continue;
            }
            const Point q = pts[p];
            const double dx = q.x - w.x;
            const double dy = q.y - w.y;
            const double d2 = dx * dx + dy * dy;
            if (d2 > m) m = d2;
            if (qtAdmissible(m, t2lo, t2hi, threshold)) {
                md[i] = m;
                ++nalive;
                if (m < bd || (m == bd && p < bi)) {
                    bd = m;
                    bi = p;
                }
            } else {
                md[i] = QT_DEAD;
            }
        }
    }

    if (tid == 0) card[b] = nmem;
}

// Dispatch to the block size instantiation chosen by the host.
#define QT_LAUNCH(blockSize, nBlocks, shBytes, ...)                          \
    do {                                                                     \
        switch (blockSize) {                                                 \
            case 32:                                                         \
                growClusterKernel<32>                                        \
                    <<<(nBlocks), 32, (shBytes)>>>(__VA_ARGS__);             \
                break;                                                       \
            case 64:                                                         \
                growClusterKernel<64>                                        \
                    <<<(nBlocks), 64, (shBytes)>>>(__VA_ARGS__);             \
                break;                                                       \
            default:                                                         \
                growClusterKernel<128>                                       \
                    <<<(nBlocks), 128, (shBytes)>>>(__VA_ARGS__);            \
                break;                                                       \
        }                                                                    \
    } while (0)

// ---------------------------------------------------------------------------
// Data generation (bit-identical to the sequential reference)
// ---------------------------------------------------------------------------

void generateSyntheticData(std::vector<Point>& points, const int N, unsigned int seed = 42) {
    auto frand = [&seed]() mutable { return rand_r(&seed) / static_cast<double>(RAND_MAX); };

    const double min_dim = std::min(MAX_WIDTH, MAX_HEIGHT);
    int count = 0;

    while (count < N) {
        // Create group_cnt points within a circle of radius R
        // around center point (cntr_x, cntr_y)
        const double cntr_x = frand() * MAX_WIDTH;
        const double cntr_y = frand() * MAX_HEIGHT;
        const double R = frand() * min_dim / 2.0;
        int group_cnt = static_cast<int>(frand() * (N / 30.0));

        // Make sure we don't make more points than we need
        if (group_cnt > (N - count)) {
            group_cnt = N - count;
        }

        while (group_cnt > 0) {
            const double sign = (frand() < 0.5) ? -1.0 : 1.0;
            const double r = frand() * R;
            const double dx = (2.0 * frand() - 1.0) * r;
            const double dy = std::sqrt(r * r - dx * dx) * sign;
            const double x = cntr_x + dx;
            const double y = cntr_y + dy;

            if (x < 0 || x > MAX_WIDTH || y < 0 || y > MAX_HEIGHT) {
                continue;
            }

            points[count] = {x, y};
            count++;
            group_cnt--;
        }
    }
}

// Calculate Euclidean distance between two points
inline double distance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

// ---------------------------------------------------------------------------
// Neighbour lists: for every point all points strictly closer than `threshold`.
// They are static for the whole run and are built on the GPU with a uniform
// spatial grid of cell size `threshold`, so that only 3x3 cells have to be
// searched per point.
// ---------------------------------------------------------------------------

__global__ void cellAssignKernel(const Point* __restrict__ pts, const int N, const double minx,
                                 const double miny, const double invCell, const int nx,
                                 const int ny, int* __restrict__ cellOf,
                                 int* __restrict__ cellCount) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int cx = static_cast<int>((pts[i].x - minx) * invCell);
    int cy = static_cast<int>((pts[i].y - miny) * invCell);
    cx = min(max(cx, 0), nx - 1);
    cy = min(max(cy, 0), ny - 1);
    const int c = cy * nx + cx;
    cellOf[i] = c;
    atomicAdd(&cellCount[c], 1);
}

__global__ void cellFillKernel(const int* __restrict__ cellOf, const int N,
                               int* __restrict__ cursor, int* __restrict__ cellPts) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    cellPts[atomicAdd(&cursor[cellOf[i]], 1)] = i;
}

// One warp per point.  `list` is null in the counting pass.
__global__ __launch_bounds__(128) void neighborScanKernel(
    const Point* __restrict__ pts, const int N, const int* __restrict__ cellOf,
    const int* __restrict__ cellStart, const int* __restrict__ cellPts, const int nx, const int ny,
    const double threshold, const double t2lo, const double t2hi, int* __restrict__ counts,
    const long long* __restrict__ offsets, int* __restrict__ list) {
    const int warpId = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (warpId >= N) return;
    const int lane = threadIdx.x & 31;
    const unsigned full = 0xffffffffu;
    const unsigned lanemask_lt = (1u << lane) - 1u;

    const int i = warpId;
    const Point p = pts[i];
    const int cx = cellOf[i] % nx;
    const int cy = cellOf[i] / nx;

    int total = 0;
    const long long base = (list != nullptr) ? offsets[i] : 0;

    for (int gy = max(cy - 1, 0); gy <= min(cy + 1, ny - 1); ++gy) {
        for (int gx = max(cx - 1, 0); gx <= min(cx + 1, nx - 1); ++gx) {
            const int c = gy * nx + gx;
            const int b = cellStart[c];
            const int e = cellStart[c + 1];
            for (int k = b + lane; __any_sync(full, k < e); k += 32) {
                int j = -1;
                bool keep = false;
                if (k < e) {
                    j = cellPts[k];
                    if (j != i) {
                        const double dx = pts[j].x - p.x;
                        const double dy = pts[j].y - p.y;
                        keep = qtAdmissible(dx * dx + dy * dy, t2lo, t2hi, threshold);
                    }
                }
                const unsigned bal = __ballot_sync(full, keep);
                if (list != nullptr && keep) {
                    list[base + total + __popc(bal & lanemask_lt)] = j;
                }
                total += __popc(bal);
            }
        }
    }

    if (lane == 0 && counts != nullptr) counts[i] = total;
}

// Mark the members of the accepted cluster as clustered.
__global__ void markClusteredKernel(const int* __restrict__ members, const int card,
                                    unsigned char* __restrict__ clustered) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < card) clustered[members[i]] = 1;
}

// Flag every still unclustered point that has one of the new members in its
// neighbourhood: those are exactly the seeds whose cached cardinality became
// invalid (the neighbour relation is symmetric).
__global__ void markDirtyKernel(const int* __restrict__ members,
                                const long long* __restrict__ off, const int* __restrict__ list,
                                const unsigned char* __restrict__ clustered,
                                unsigned char* __restrict__ mark) {
    const int m = members[blockIdx.x];
    const long long b = off[m];
    const int cnt = static_cast<int>(off[m + 1] - b);
    for (int k = threadIdx.x; k < cnt; k += blockDim.x) {
        const int q = list[b + k];
        if (!clustered[q]) mark[q] = 1;
    }
}

// ---------------------------------------------------------------------------
// Main QT clustering algorithm (hybrid MPI + OpenMP + CUDA)
// ---------------------------------------------------------------------------

struct GpuState {
    Point* d_pts = nullptr;
    long long* d_off = nullptr;
    int* d_list = nullptr;
    unsigned char* d_clustered = nullptr;
    unsigned char* d_dirtyMark = nullptr;
    int* d_seeds = nullptr;
    int* d_card = nullptr;
    int* d_members = nullptr;
    double* d_md2 = nullptr;
    int* d_idx = nullptr;
    long long mdStride = 0;
    int maxGlobalBlocks = 0;
};

std::vector<Cluster> qtClustering(const std::vector<Point>& points, const double threshold,
                                  const int mpi_rank, const int mpi_size) {
    const int N = static_cast<int>(points.size());

    // Safety band around threshold^2 for the squared-distance admission test
    // (see qtAdmissible).
    const double t2 = threshold * threshold;
    const double t2lo = t2 * (1.0 - 1e-14);
    const double t2hi = t2 * (1.0 + 1e-14);

    GpuState g;
    CUDA_CHECK(cudaMalloc(&g.d_pts, sizeof(Point) * N));
    CUDA_CHECK(cudaMemcpy(g.d_pts, points.data(), sizeof(Point) * N, cudaMemcpyHostToDevice));

    // ---- neighbour lists (GPU, uniform grid of cell size `threshold`) ----
    std::vector<long long> offsets(N + 1);
    {
        double minx = points[0].x, maxx = points[0].x;
        double miny = points[0].y, maxy = points[0].y;
        for (int i = 1; i < N; ++i) {
            minx = std::min(minx, points[i].x);
            maxx = std::max(maxx, points[i].x);
            miny = std::min(miny, points[i].y);
            maxy = std::max(maxy, points[i].y);
        }
        const int nx = std::max(1, static_cast<int>((maxx - minx) / threshold) + 1);
        const int ny = std::max(1, static_cast<int>((maxy - miny) / threshold) + 1);
        const int ncells = nx * ny;

        int *d_cellOf = nullptr, *d_cellCount = nullptr, *d_cellPts = nullptr, *d_counts = nullptr;
        CUDA_CHECK(cudaMalloc(&d_cellOf, sizeof(int) * N));
        CUDA_CHECK(cudaMalloc(&d_cellCount, sizeof(int) * (ncells + 1)));
        CUDA_CHECK(cudaMalloc(&d_cellPts, sizeof(int) * N));
        CUDA_CHECK(cudaMalloc(&d_counts, sizeof(int) * N));
        CUDA_CHECK(cudaMemset(d_cellCount, 0, sizeof(int) * (ncells + 1)));

        const int tpb = 256;
        cellAssignKernel<<<(N + tpb - 1) / tpb, tpb>>>(g.d_pts, N, minx, miny, 1.0 / threshold, nx,
                                                       ny, d_cellOf, d_cellCount + 1);

        std::vector<int> cellStart(ncells + 1);
        CUDA_CHECK(cudaMemcpy(cellStart.data(), d_cellCount, sizeof(int) * (ncells + 1),
                              cudaMemcpyDeviceToHost));
        for (int c = 0; c < ncells; ++c) cellStart[c + 1] += cellStart[c];
        CUDA_CHECK(cudaMemcpy(d_cellCount, cellStart.data(), sizeof(int) * (ncells + 1),
                              cudaMemcpyHostToDevice));

        int* d_cursor = nullptr;
        CUDA_CHECK(cudaMalloc(&d_cursor, sizeof(int) * ncells));
        CUDA_CHECK(cudaMemcpy(d_cursor, cellStart.data(), sizeof(int) * ncells,
                              cudaMemcpyHostToDevice));
        cellFillKernel<<<(N + tpb - 1) / tpb, tpb>>>(d_cellOf, N, d_cursor, d_cellPts);
        CUDA_CHECK(cudaFree(d_cursor));

        const int warpsPerBlock = 128 / 32;
        const int blocks = (N + warpsPerBlock - 1) / warpsPerBlock;
        neighborScanKernel<<<blocks, 128>>>(g.d_pts, N, d_cellOf, d_cellCount, d_cellPts, nx, ny,
                                            threshold, t2lo, t2hi, d_counts, nullptr, nullptr);

        std::vector<int> counts(N);
        CUDA_CHECK(cudaMemcpy(counts.data(), d_counts, sizeof(int) * N, cudaMemcpyDeviceToHost));
        offsets[0] = 0;
        for (int i = 0; i < N; ++i) offsets[i + 1] = offsets[i] + counts[i];

        CUDA_CHECK(cudaMalloc(&g.d_off, sizeof(long long) * (N + 1)));
        CUDA_CHECK(cudaMemcpy(g.d_off, offsets.data(), sizeof(long long) * (N + 1),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&g.d_list, sizeof(int) * std::max<long long>(offsets[N], 1)));
        neighborScanKernel<<<blocks, 128>>>(g.d_pts, N, d_cellOf, d_cellCount, d_cellPts, nx, ny,
                                            threshold, t2lo, t2hi, nullptr, g.d_off, g.d_list);

        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaFree(d_cellOf));
        CUDA_CHECK(cudaFree(d_cellCount));
        CUDA_CHECK(cudaFree(d_cellPts));
        CUDA_CHECK(cudaFree(d_counts));
    }

    // Per point neighbour counts drive the block size / shared memory decisions.
    int maxCnt = 0;
    long long sumCnt = 0;
    for (int i = 0; i < N; ++i) {
        const int c = static_cast<int>(offsets[i + 1] - offsets[i]);
        maxCnt = std::max(maxCnt, c);
        sumCnt += c;
    }
    const int avgCnt = static_cast<int>(sumCnt / std::max(N, 1));

    // Block size: with the active-set compaction most iterations only touch a
    // small fraction of the initial candidate set, so oversized blocks would
    // just idle in the reduction.  Empirically ~16 candidates per thread is the
    // sweet spot.
    int blockSize = 32;
    while (blockSize < 128 && blockSize * 16 < avgCnt) blockSize <<= 1;

    // Shared memory budget per block (12 bytes per candidate: md + index).
    const int sharedCapBytes = 64 * 1024;
    const int sharedCapCnt = sharedCapBytes / 12;

    CUDA_CHECK(cudaMalloc(&g.d_clustered, N));
    CUDA_CHECK(cudaMalloc(&g.d_dirtyMark, N));
    CUDA_CHECK(cudaMemset(g.d_clustered, 0, N));
    CUDA_CHECK(cudaMalloc(&g.d_seeds, sizeof(int) * N));
    CUDA_CHECK(cudaMalloc(&g.d_card, sizeof(int) * N));
    CUDA_CHECK(cudaMalloc(&g.d_members, sizeof(int) * (maxCnt + 1)));

    // Scratch for seeds whose candidate set does not fit into shared memory.
    if (maxCnt > sharedCapCnt) {
        const size_t budget = 512ull * 1024ull * 1024ull;
        g.mdStride = maxCnt;
        g.maxGlobalBlocks =
            static_cast<int>(std::max<size_t>(1, budget / (sizeof(double) * g.mdStride)));
        g.maxGlobalBlocks = std::min(g.maxGlobalBlocks, N);
        CUDA_CHECK(cudaMalloc(&g.d_md2, sizeof(double) * g.mdStride * g.maxGlobalBlocks));
        CUDA_CHECK(cudaMalloc(&g.d_idx, sizeof(int) * g.mdStride * g.maxGlobalBlocks));
    }

    CUDA_CHECK(cudaFuncSetAttribute(growClusterKernel<32>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, sharedCapBytes));
    CUDA_CHECK(cudaFuncSetAttribute(growClusterKernel<64>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, sharedCapBytes));
    CUDA_CHECK(cudaFuncSetAttribute(growClusterKernel<128>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, sharedCapBytes));

    std::vector<unsigned char> clustered(N, 0);
    std::vector<int> unclustered(N);
    for (int i = 0; i < N; ++i) unclustered[i] = i;
    std::vector<Cluster> clusters;

    // Incremental re-evaluation: the candidate cluster grown from a seed only
    // depends on the clustered flags of the points within `threshold` of that
    // seed.  After a cluster has been accepted only the seeds that have one of
    // its members in their neighbourhood can change, so every other cached
    // cardinality stays exactly valid.  This is what turns the O(rounds * N)
    // seed evaluations of the reference into O(rounds * |affected|).
    std::vector<int> cardCache(N, 0);
    std::vector<unsigned char> dirtyMark(N, 0);
    std::vector<int> dirty(N);
    for (int i = 0; i < N; ++i) dirty[i] = i;  // first round: everything

    std::vector<int> smallPos, bigPos, seedBuf, cards, localCards, dirtyCards;
    smallPos.reserve(N);
    bigPos.reserve(N);
    seedBuf.resize(N);
    cards.resize(N);
    localCards.resize(N);
    dirtyCards.resize(N);
    std::vector<int> counts(mpi_size), displs(mpi_size);
    std::vector<int> winnerMembers(maxCnt + 1);

    while (!unclustered.empty()) {
        const int nd = static_cast<int>(dirty.size());

        // Round-robin deal of the dirty seeds: neighbouring seeds have very
        // similar cost, so interleaving balances the ranks much better than
        // handing out contiguous ranges.
        {
            int acc = 0;
            for (int r = 0; r < mpi_size; ++r) {
                counts[r] = nd / mpi_size + (r < nd % mpi_size ? 1 : 0);
                displs[r] = acc;
                acc += counts[r];
            }
        }
        const int myCount = counts[mpi_rank];

        // Seeds whose candidate set fits into shared memory are run by a
        // separate launch from the (rare) oversized ones.
        smallPos.clear();
        bigPos.clear();
        int maxSmallCnt = 1;
        for (int i = 0; i < myCount; ++i) {
            const int s = dirty[mpi_rank + i * mpi_size];
            const int c = static_cast<int>(offsets[s + 1] - offsets[s]);
            if (c <= sharedCapCnt) {
                smallPos.push_back(i);
                maxSmallCnt = std::max(maxSmallCnt, c);
            } else {
                bigPos.push_back(i);
            }
        }
        const int nSmall = static_cast<int>(smallPos.size());
        const int nBig = static_cast<int>(bigPos.size());
        // Only request as much shared memory as the largest candidate set of
        // this launch actually needs - this directly drives GPU occupancy.
        const int smallShBytes = 12 * maxSmallCnt;

        for (int i = 0; i < nSmall; ++i) seedBuf[i] = dirty[mpi_rank + smallPos[i] * mpi_size];
        for (int i = 0; i < nBig; ++i) seedBuf[nSmall + i] = dirty[mpi_rank + bigPos[i] * mpi_size];

        if (myCount > 0) {
            CUDA_CHECK(cudaMemcpy(g.d_seeds, seedBuf.data(), sizeof(int) * myCount,
                                  cudaMemcpyHostToDevice));
        }
        if (nSmall > 0) {
            QT_LAUNCH(blockSize, nSmall, smallShBytes, g.d_pts, g.d_off, g.d_list, g.d_clustered,
                      g.d_seeds, nSmall, threshold, t2lo, t2hi, g.d_card, nullptr, 0, nullptr,
                      nullptr, 0, 1);
        }
        for (int base = 0; base < nBig; base += g.maxGlobalBlocks) {
            const int nb = std::min(g.maxGlobalBlocks, nBig - base);
            QT_LAUNCH(blockSize, nb, 0, g.d_pts, g.d_off, g.d_list, g.d_clustered,
                      g.d_seeds + nSmall + base, nb, threshold, t2lo, t2hi,
                      g.d_card + nSmall + base, nullptr, 0, g.d_md2, g.d_idx, g.mdStride, 0);
        }
        if (myCount > 0) {
            CUDA_CHECK(cudaMemcpy(cards.data(), g.d_card, sizeof(int) * myCount,
                                  cudaMemcpyDeviceToHost));
        }
        CUDA_CHECK(cudaGetLastError());

        // Undo the small/big reordering, then replicate the results.
#pragma omp parallel for schedule(static) if (myCount > 65536)
        for (int i = 0; i < nSmall; ++i) localCards[smallPos[i]] = cards[i];
#pragma omp parallel for schedule(static) if (nBig > 65536)
        for (int i = 0; i < nBig; ++i) localCards[bigPos[i]] = cards[nSmall + i];

        if (mpi_size > 1) {
            MPI_Allgatherv(localCards.data(), myCount, MPI_INT, dirtyCards.data(), counts.data(),
                           displs.data(), MPI_INT, MPI_COMM_WORLD);
        } else {
            std::copy(localCards.begin(), localCards.begin() + myCount, dirtyCards.begin());
        }

#pragma omp parallel for schedule(static) if (nd > 65536)
        for (int r = 0; r < mpi_size; ++r) {
            const int d0 = displs[r];
            for (int j = 0; j < counts[r]; ++j) {
                cardCache[dirty[r + j * mpi_size]] = dirtyCards[d0 + j];
            }
        }

        // Best seed: maximum cardinality, smallest index on ties - exactly the
        // selection of the sequential scan over the ascending unclustered list.
        const int nunc = static_cast<int>(unclustered.size());
        int bestCard = -1;
        int bestSeed = INT_MAX;
#pragma omp parallel if (nunc > 65536)
        {
            int lc = -1, ls = INT_MAX;
#pragma omp for schedule(static) nowait
            for (int i = 0; i < nunc; ++i) {
                const int s = unclustered[i];
                const int c = cardCache[s];
                if (c > lc || (c == lc && s < ls)) {
                    lc = c;
                    ls = s;
                }
            }
#pragma omp critical
            {
                if (lc > bestCard || (lc == bestCard && ls < bestSeed)) {
                    bestCard = lc;
                    bestSeed = ls;
                }
            }
        }

        if (bestCard <= 0) break;  // no more clusters can be formed

        // Re-grow the winning cluster (single block) to obtain its members.
        CUDA_CHECK(cudaMemcpy(g.d_seeds, &bestSeed, sizeof(int), cudaMemcpyHostToDevice));
        const int wcnt = static_cast<int>(offsets[bestSeed + 1] - offsets[bestSeed]);
        if (wcnt <= sharedCapCnt) {
            QT_LAUNCH(blockSize, 1, 12 * std::max(wcnt, 1), g.d_pts, g.d_off, g.d_list,
                      g.d_clustered, g.d_seeds, 1, threshold, t2lo, t2hi, g.d_card, g.d_members,
                      maxCnt + 1, nullptr, nullptr, 0, 1);
        } else {
            QT_LAUNCH(blockSize, 1, 0, g.d_pts, g.d_off, g.d_list, g.d_clustered, g.d_seeds, 1,
                      threshold, t2lo, t2hi, g.d_card, g.d_members, maxCnt + 1, g.d_md2, g.d_idx,
                      g.mdStride, 0);
        }
        CUDA_CHECK(cudaMemcpy(winnerMembers.data(), g.d_members, sizeof(int) * bestCard,
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaGetLastError());

        Cluster cluster;
        cluster.seed_point = bestSeed;
        cluster.members.assign(winnerMembers.begin(), winnerMembers.begin() + bestCard);
        clusters.push_back(cluster);

        for (int i = 0; i < bestCard; ++i) clustered[cluster.members[i]] = 1;

        unclustered.erase(std::remove_if(unclustered.begin(), unclustered.end(),
                                         [&clustered](int idx) { return clustered[idx] != 0; }),
                          unclustered.end());

        // Seeds that see one of the new members in their neighbourhood have to
        // be re-evaluated.  The members are already in device memory, so both
        // the clustered flags and the invalidation are updated there.
        markClusteredKernel<<<(bestCard + 127) / 128, 128>>>(g.d_members, bestCard, g.d_clustered);
        CUDA_CHECK(cudaMemset(g.d_dirtyMark, 0, N));
        markDirtyKernel<<<bestCard, 128>>>(g.d_members, g.d_off, g.d_list, g.d_clustered,
                                           g.d_dirtyMark);
        CUDA_CHECK(cudaMemcpy(dirtyMark.data(), g.d_dirtyMark, N, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaGetLastError());

        dirty.clear();
        for (int i = 0; i < N; ++i) {
            if (dirtyMark[i]) dirty.push_back(i);
        }
    }

    cudaFree(g.d_pts);
    cudaFree(g.d_off);
    cudaFree(g.d_list);
    cudaFree(g.d_clustered);
    cudaFree(g.d_dirtyMark);
    cudaFree(g.d_seeds);
    cudaFree(g.d_card);
    cudaFree(g.d_members);
    if (g.d_md2) cudaFree(g.d_md2);
    if (g.d_idx) cudaFree(g.d_idx);

    return clusters;
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

bool validateClusters(const std::vector<Cluster>& clusters, const std::vector<Point>& points,
                      const double threshold) {
    bool valid = true;

    printf("Validating clusters:\n");

    const int nc = static_cast<int>(clusters.size());
    std::vector<double> diameters(nc, 0.0);

#pragma omp parallel for schedule(dynamic, 1)
    for (int c = 0; c < nc; ++c) {
        const auto& cluster = clusters[c];
        double max_diameter = 0.0;
        for (size_t i = 0; i < cluster.members.size(); ++i) {
            for (size_t j = i + 1; j < cluster.members.size(); ++j) {
                const double dist =
                    distance(points[cluster.members[i]], points[cluster.members[j]]);
                max_diameter = std::max(max_diameter, dist);
            }
        }
        diameters[c] = max_diameter;
    }

    for (int c = 0; c < nc; ++c) {
        if (c < 10) {  // Print first 10 clusters
            printf("  Cluster %d: size=%zu, seed=%d, diameter=%.4f\n", c,
                   clusters[c].members.size(), clusters[c].seed_point, diameters[c]);
        }

        // Validate diameter is within threshold
        if (diameters[c] > threshold * 1.001) {  // Allow small numerical error
            printf("ERROR: Cluster %d has diameter %.4f > threshold %.4f\n", c, diameters[c],
                   threshold);
            valid = false;
        }
    }

    // Check for duplicate memberships
    std::vector<int> membership(points.size(), -1);
    for (int c = 0; c < nc; ++c) {
        for (size_t i = 0; i < clusters[c].members.size(); ++i) {
            const int member = clusters[c].members[i];
            if (membership[member] >= 0) {
                printf("ERROR: Point %d appears in multiple clusters (%d and %d)\n", member,
                       membership[member], c);
                valid = false;
            }
            membership[member] = c;
        }
    }

    // Count clustered points
    int clustered_count = 0;
    for (size_t i = 0; i < membership.size(); ++i) {
        if (membership[i] >= 0) clustered_count++;
    }

    printf("Total points: %zu, Clustered: %d, Unclustered: %zu\n", points.size(), clustered_count,
           points.size() - clustered_count);

    return valid;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Number of points (default: 1000)\n");
    printf("  -t <float>   Distance threshold for clustering (default: 2.0)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int mpi_rank = 0, mpi_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    // One GPU per rank, assigned by node-local rank.
    {
        MPI_Comm local_comm;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, mpi_rank, MPI_INFO_NULL,
                            &local_comm);
        int local_rank = 0, local_size = 1;
        MPI_Comm_rank(local_comm, &local_rank);
        MPI_Comm_size(local_comm, &local_size);
        MPI_Comm_free(&local_comm);

        // Share the node's cores between the node-local ranks (unless the user
        // pinned the thread count explicitly).
        if (getenv("OMP_NUM_THREADS") == nullptr) {
            const int cores = omp_get_num_procs();
            omp_set_num_threads(std::max(1, cores / std::max(1, local_size)));
        }

        int devCount = 0;
        if (cudaGetDeviceCount(&devCount) != cudaSuccess || devCount == 0) {
            if (mpi_rank == 0) fprintf(stderr, "Error: no CUDA device available\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        CUDA_CHECK(cudaSetDevice(local_rank % devCount));
        CUDA_CHECK(cudaFree(nullptr));  // establish the context up front
        // Force the kernel module to be loaded now (CUDA defaults to lazy
        // loading) so that it does not show up in the measured region.
        // Spawn the OpenMP thread pool once, outside of the measured region.
        volatile int ompWarm = 0;
#pragma omp parallel reduction(+ : ompWarm)
        { ompWarm += 1; }
        (void)ompWarm;

        QT_LAUNCH(32, 1, 0, nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0.0, 0.0, 0.0, nullptr,
                  nullptr, 0, nullptr, nullptr, 0, 0);
        QT_LAUNCH(64, 1, 0, nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0.0, 0.0, 0.0, nullptr,
                  nullptr, 0, nullptr, nullptr, 0, 0);
        QT_LAUNCH(128, 1, 0, nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0.0, 0.0, 0.0, nullptr,
                  nullptr, 0, nullptr, nullptr, 0, 0);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    int num_points = 1000;
    double threshold = 2.0;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            num_points = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            threshold = atof(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (mpi_rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (mpi_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (num_points <= 0 || threshold <= 0.0) {
        if (mpi_rank == 0) {
            printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n", num_points,
                   threshold);
        }
        MPI_Finalize();
        return 1;
    }

    if (mpi_rank == 0) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Generate synthetic data (deterministic, identical on every rank)
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);

    // Perform QT clustering
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_start = std::chrono::high_resolution_clock::now();

    const std::vector<Cluster> clusters = qtClustering(points, threshold, mpi_rank, mpi_size);

    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(cluster_end - cluster_start);

    if (mpi_rank != 0) {
        MPI_Finalize();
        return 0;
    }

    printf("Clustering time: %ld ms\n", cluster_time.count());
    printf("Clusters found: %zu\n", clusters.size());

    // Calculate statistics and performance metrics
    int total_clustered = 0;
    int max_cluster_size = 0;

    for (size_t i = 0; i < clusters.size(); ++i) {
        const int size = static_cast<int>(clusters[i].members.size());
        total_clustered += size;
        max_cluster_size = std::max(max_cluster_size, size);
    }

    const double avg_cluster_size =
        clusters.empty() ? 0.0 : static_cast<double>(total_clustered) / clusters.size();

    printf("Points clustered: %d / %d (%.1f%%)\n", total_clustered, num_points,
           100.0 * total_clustered / num_points);
    printf("Average cluster size: %.2f\n", avg_cluster_size);
    printf("Maximum cluster size: %d\n", max_cluster_size);

    // Performance metrics
    const double time_sec = cluster_time.count() / 1000.0;
    const double clusters_per_sec = clusters.size() / time_sec;
    const double points_per_sec = num_points / time_sec;
    printf("Performance: %.1f clusters/s, %.1f points/s\n", clusters_per_sec, points_per_sec);

    // Print results for external validation
    if (printResults) {
        // Serialize cluster membership for hashing
        std::vector<double> membershipData;
        membershipData.reserve(num_points);
        std::vector<int> membership(num_points, -1);
        for (size_t c = 0; c < clusters.size(); ++c) {
            for (size_t i = 0; i < clusters[c].members.size(); ++i) {
                membership[clusters[c].members[i]] = static_cast<int>(c);
            }
        }
        for (int m : membership) {
            membershipData.push_back(static_cast<double>(m));
        }
        print_results(membershipData, "ClusterMembership");
    }

    // Validation
    int rc = 0;
    if (validate) {
        const bool valid = validateClusters(clusters, points, threshold);

        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            rc = 1;
        }
    }

    MPI_Finalize();
    return rc;
}
