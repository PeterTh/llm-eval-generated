// QT Clustering Benchmark - CUDA Parallel Version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// Parallelization strategy (GPU / CUDA).  Every result is bit-identical to the
// sequential reference; all transformations below preserve its semantics,
// including its tie-breaking rules (lowest index wins).
//
//   * A round of the outer loop tries every remaining unclustered point as a
//     seed.  Those trials are independent, so one CUDA block grows the
//     candidate cluster of one seed and all seeds run concurrently; within a
//     block the candidate scan of a growth step is spread over the threads and
//     finished with a block-wide min-reduction.
//   * Distances are compared in squared form.  x -> x*x is monotone on
//     non-negative values, so every comparison the algorithm performs is
//     unchanged, but the (very slow) double-precision square root disappears.
//   * Instead of recomputing "max distance from candidate to every member" from
//     scratch in each growth step, the running maximum per candidate is updated
//     with the distance to the newly added member only.  max over the same set
//     of values is exact and order independent, which turns a growth step from
//     O(K*|members|) into O(K).
//   * A point can only join a cluster seeded at s if it is within the threshold
//     of s (s is always a member).  The threshold neighbourhood of every point
//     is therefore precomputed once, and a seed only ever scans its own
//     neighbour list instead of all N points.  Candidates whose running maximum
//     has reached the threshold are compacted out of the working list as the
//     cluster grows, so the scanned range shrinks towards the cluster size.
//   * A seed's result only depends on the unclustered points around it, so
//     accepting a cluster only invalidates the cached cardinalities of seeds in
//     the neighbourhood of one of its members.  Every other seed keeps its
//     result and is skipped, which makes all rounds after the first cheap.
//   * The complete round - growth, winner selection, member extraction and the
//     rebuilding of the work lists - runs on the device.  Rounds are captured
//     into a CUDA graph and replayed in batches, so the host synchronises a
//     handful of times for the whole run instead of once per round.

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

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

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        const cudaError_t err__ = (call);                                       \
        if (err__ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__,    \
                    __LINE__, cudaGetErrorString(err__));                       \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// Threads per block used by the per-seed cluster-growth kernel.  Candidate
// neighbourhoods are small, so a narrow block keeps the block-wide reduction
// short and lets many seeds run concurrently per SM.
static const int BLK = 64;
static const int NWARPS = BLK / 32;
// Wider block for the (single-block) bookkeeping kernels, which stream over
// all N points.
static const int BLK2 = 1024;
// Candidate lists up to this length are staged in shared memory (index, running
// max squared distance and coordinates), keeping the hot growth loop off DRAM.
static const int SMEM_CAP = 512;

// Squared Euclidean distance.  The algorithm only ever compares distances
// against each other or against the threshold, and x -> x*x is monotone on
// non-negative values, so working in squared space is equivalent while
// avoiding the (very slow) double-precision square root.
__device__ __forceinline__ double devDist2(const double2 a, const double2 b) {
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    return dx * dx + dy * dy;
}

// Block-wide ordered stream-compaction scratch (at most 32 warps per block).
struct ScanState {
    int warp[32];
    int base;
};

// Returns the output slot for this thread's element and advances sc.base by the
// number of set predicates in the block.  Callers must __syncthreads() after
// consuming the result.
__device__ __forceinline__ int orderedSlot(const bool pred, ScanState& sc,
                                           const int tid, const int lane,
                                           const int warp, const int nwarps) {
    const unsigned int m = __ballot_sync(0xffffffffu, pred);
    if (lane == 0) sc.warp[warp] = __popc(m);
    __syncthreads();
    if (tid == 0) {
        int s = sc.base;
        for (int w = 0; w < nwarps; ++w) {
            const int c = sc.warp[w];
            sc.warp[w] = s;
            s += c;
        }
        sc.base = s;
    }
    __syncthreads();
    return sc.warp[warp] + __popc(m & ((1u << lane) - 1u));
}

// ---------------------------------------------------------------------------
// Neighbour-list construction (done once).
//
// A point can only ever join a cluster seeded at s if its distance to s is
// below the threshold (s is always a member).  The set of admissible candidates
// per seed is therefore fixed up front, which turns the per-round O(N) scan per
// seed into an O(deg(seed)) scan.
// ---------------------------------------------------------------------------

__global__ __launch_bounds__(BLK2) void countNeighborsKernel(
    const double2* __restrict__ pts, const int N, const double thr2,
    int* __restrict__ counts) {
    __shared__ int s_red[BLK2 / 32];

    const int i = blockIdx.x;
    const int tid = threadIdx.x;
    const double2 p = pts[i];

    int c = 0;
    for (int j = tid; j < N; j += BLK2) {
        if (devDist2(pts[j], p) < thr2) ++c;
    }
    for (int off = 16; off > 0; off >>= 1) c += __shfl_down_sync(0xffffffffu, c, off);
    if ((tid & 31) == 0) s_red[tid >> 5] = c;
    __syncthreads();
    if (tid == 0) {
        int t = 0;
        for (int w = 0; w < BLK2 / 32; ++w) t += s_red[w];
        counts[i] = t;
    }
}

__global__ __launch_bounds__(BLK2) void fillNeighborsKernel(
    const double2* __restrict__ pts, const int N, const double thr2,
    const int* __restrict__ offsets, int* __restrict__ nbrIdx) {
    __shared__ ScanState sc;

    const int i = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const double2 p = pts[i];
    int* out = nbrIdx + offsets[i];

    if (tid == 0) sc.base = 0;
    __syncthreads();

    for (int base = 0; base < N; base += BLK2) {
        const int j = base + tid;
        const bool pred = (j < N) && (devDist2(pts[j], p) < thr2);
        const int pos = orderedSlot(pred, sc, tid, lane, warp, BLK2 / 32);
        if (pred) out[pos] = j;
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// Shared device state.
//
// meta[] keeps the whole loop state on the device so that the host never has
// to synchronise in the middle of a round.
// ---------------------------------------------------------------------------
enum MetaSlot {
    META_BEST_SEED = 0,   // seed of the cluster accepted in the current round
    META_MEMBER_CNT = 1,  // size of that cluster
    META_NUM_SEEDS = 2,   // number of still unclustered points
    META_NUM_DIRTY = 3,   // number of seeds that must be (re)grown this round
    META_NUM_CLUST = 4,   // clusters accepted so far
    META_MEMBER_TOT = 5,  // total members written to the output buffer
    META_ONE = 6,         // constant 1 (grid-stride bound of the record pass)
    META_COUNT = 8
};

__global__ void initStateKernel(int* __restrict__ meta, int* __restrict__ seeds,
                                int* __restrict__ dirtyList,
                                unsigned char* __restrict__ clustered,
                                unsigned char* __restrict__ dirty, const int N) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        seeds[i] = i;
        dirtyList[i] = i;
        clustered[i] = 0;
        dirty[i] = 0;
    }
    if (i == 0) {
        meta[META_BEST_SEED] = -1;
        meta[META_MEMBER_CNT] = 0;
        meta[META_NUM_SEEDS] = N;
        meta[META_NUM_DIRTY] = N;
        meta[META_NUM_CLUST] = 0;
        meta[META_MEMBER_TOT] = 0;
        meta[META_ONE] = 1;
    }
}

// ---------------------------------------------------------------------------
// Per-seed candidate cluster growth: one block per seed (grid-stride).
// ---------------------------------------------------------------------------

// Grows the candidate cluster of every seed in `seedList[0 .. *countPtr)` and
// stores its cardinality in `card[seed]`.
//
// `nbrOff`/`nbrIdx` hold the precomputed neighbour lists; if they are null the
// block falls back to scanning all N points (used when the neighbour lists
// would not fit into device memory).
//
// In record mode (`record != 0`, launched with a single block for the winning
// seed) the members are appended to `allMembers` and the cluster bookkeeping in
// meta[] is advanced.
__global__ __launch_bounds__(BLK) void growKernel(
    const double2* __restrict__ pts, const unsigned char* __restrict__ clustered,
    const int N, const double thr2, const int* seedList, const int* countPtr,
    const int* __restrict__ nbrOff, const int* __restrict__ nbrIdx,
    int* __restrict__ card, int* __restrict__ nbrBuf, double* __restrict__ mdBuf,
    const size_t slotStride, int* meta, int* __restrict__ allMembers,
    int* __restrict__ clusterSeed, int* __restrict__ clusterOff, const int record) {
    __shared__ double s_md[SMEM_CAP];
    __shared__ int s_nbr[SMEM_CAP];
    __shared__ double2 s_pt[SMEM_CAP];
    __shared__ double r_val[NWARPS];
    __shared__ int r_pos[NWARPS];
    __shared__ ScanState sc;
    __shared__ int s_seedPos;

    if (meta[META_NUM_SEEDS] == 0) return;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int total = *countPtr;

    for (int si = blockIdx.x; si < total; si += gridDim.x) {
        const int seed = seedList[si];
        if (seed < 0) continue;
        const double2 sp = pts[seed];

        // Upper bound on the candidate count decides where the working set
        // lives; the compacted list is never longer than that.
        const int rs = (nbrOff != nullptr) ? nbrOff[seed] : 0;
        const int span = (nbrOff != nullptr) ? (nbrOff[seed + 1] - rs) : N;
        const bool useShared = (span <= SMEM_CAP);

        double* md = useShared ? s_md
                               : (mdBuf + static_cast<size_t>(blockIdx.x) * slotStride);
        int* nbr = useShared ? s_nbr
                             : (nbrBuf + static_cast<size_t>(blockIdx.x) * slotStride);

        __syncthreads();
        if (tid == 0) {
            sc.base = 0;
            s_seedPos = 0;
        }
        __syncthreads();

        // ---- Compact the still-unclustered candidates in ascending index
        // order.  md[] starts out as the squared distance to the seed, i.e. the
        // running maximum distance to the (single element) member set.
        for (int base = 0; base < span; base += BLK) {
            const int t = base + tid;
            int idx = -1;
            bool pred = false;
            double2 q = sp;
            double d2 = 0.0;
            if (t < span) {
                idx = (nbrIdx != nullptr) ? nbrIdx[rs + t] : t;
                if (!clustered[idx]) {
                    q = pts[idx];
                    d2 = devDist2(q, sp);
                    // Neighbour-list entries already satisfy the threshold.
                    pred = (nbrIdx != nullptr) || (d2 < thr2);
                }
            }
            const int pos = orderedSlot(pred, sc, tid, lane, warp, NWARPS);
            if (pred) {
                nbr[pos] = idx;
                md[pos] = d2;
                if (useShared) s_pt[pos] = q;
                if (idx == seed) s_seedPos = pos;
            }
            __syncthreads();
        }

        // The seed itself is the first member; a negative md marks membership.
        if (tid == 0) md[s_seedPos] = -1.0;
        __syncthreads();

        int cnt = 1;
        int live = sc.base;       // current length of the working list
        double2 lp = sp;          // coordinates of the most recently added member

        // ---- Iteratively add the closest admissible candidate.
        // Each pass refreshes the running maximum distances, picks the winner
        // and at the same time compacts candidates that can never join again
        // (their running maximum already reached the threshold) out of the
        // list, so the scanned range shrinks towards the final cluster size.
        while (cnt < live) {
            double bv = DBL_MAX;
            int bp = INT_MAX;
            __syncthreads();  // every thread has read the previous sc.base
            if (tid == 0) sc.base = 0;
            __syncthreads();

            for (int base = 0; base < live; base += BLK) {
                const int j = base + tid;
                bool keep = false;
                double v = -1.0;
                int idx = -1;
                double2 q = sp;
                if (j < live) {
                    v = md[j];
                    idx = nbr[j];
                    q = useShared ? s_pt[j] : pts[idx];
                    if (v < 0.0) {
                        keep = true;  // already a member, stays in the list
                    } else {
                        const double d2 = devDist2(q, lp);
                        if (d2 > v) v = d2;
                        keep = (v < thr2);
                    }
                }
                const int pos = orderedSlot(keep, sc, tid, lane, warp, NWARPS);
                if (keep) {
                    md[pos] = v;
                    nbr[pos] = idx;
                    if (useShared) s_pt[pos] = q;
                    // Compaction preserves the relative order, so a smaller
                    // slot still means a smaller original point index.
                    if (v >= 0.0 && v < bv) {
                        bv = v;
                        bp = pos;
                    }
                }
                __syncthreads();
            }
            live = sc.base;

            // Warp reduction, then a short pass over the per-warp results.
            for (int off = 16; off > 0; off >>= 1) {
                const double v2 = __shfl_down_sync(0xffffffffu, bv, off);
                const int p2 = __shfl_down_sync(0xffffffffu, bp, off);
                if (v2 < bv || (v2 == bv && p2 < bp)) {
                    bv = v2;
                    bp = p2;
                }
            }
            if (NWARPS > 1) {
                if (lane == 0) {
                    r_val[warp] = bv;
                    r_pos[warp] = bp;
                }
                __syncthreads();
                bv = r_val[0];
                bp = r_pos[0];
                for (int w = 1; w < NWARPS; ++w) {
                    const double v2 = r_val[w];
                    const int p2 = r_pos[w];
                    if (v2 < bv || (v2 == bv && p2 < bp)) {
                        bv = v2;
                        bp = p2;
                    }
                }
            } else {
                bv = __shfl_sync(0xffffffffu, bv, 0);
                bp = __shfl_sync(0xffffffffu, bp, 0);
            }
            if (bp == INT_MAX) break;  // no more points can be added
            lp = useShared ? s_pt[bp] : pts[nbr[bp]];
            if (tid == 0) md[bp] = -1.0;
            ++cnt;
            __syncthreads();
        }

        if (tid == 0) card[seed] = cnt;

        if (record) {
            const int base = meta[META_MEMBER_TOT];
            __syncthreads();  // every thread has read the previous sc.base
            if (tid == 0) sc.base = 0;
            __syncthreads();
            for (int j = tid; j < live; j += BLK) {
                if (md[j] < 0.0) allMembers[base + atomicAdd(&sc.base, 1)] = nbr[j];
            }
            __syncthreads();
            if (tid == 0) {
                const int c = meta[META_NUM_CLUST];
                clusterSeed[c] = seed;
                clusterOff[c] = base;
                meta[META_NUM_CLUST] = c + 1;
                meta[META_MEMBER_TOT] = base + cnt;
                meta[META_MEMBER_CNT] = cnt;
            }
        }
    }
}

// Pick the seed with the largest cardinality (lowest point index on ties).
__global__ __launch_bounds__(BLK2) void pickBestKernel(
    const int* __restrict__ card, const int* __restrict__ seeds,
    int* __restrict__ meta) {
    __shared__ int r_card[BLK2 / 32];
    __shared__ int r_idx[BLK2 / 32];

    const int numSeeds = meta[META_NUM_SEEDS];
    if (numSeeds == 0) return;

    const int tid = threadIdx.x;
    int bc = -1;
    int bi = INT_MAX;
    for (int i = tid; i < numSeeds; i += BLK2) {
        const int c = card[seeds[i]];
        if (c > bc) {
            bc = c;
            bi = i;
        }
    }
    for (int off = 16; off > 0; off >>= 1) {
        const int c2 = __shfl_down_sync(0xffffffffu, bc, off);
        const int i2 = __shfl_down_sync(0xffffffffu, bi, off);
        if (c2 > bc || (c2 == bc && i2 < bi)) {
            bc = c2;
            bi = i2;
        }
    }
    if ((tid & 31) == 0) {
        r_card[tid >> 5] = bc;
        r_idx[tid >> 5] = bi;
    }
    __syncthreads();
    if (tid == 0) {
        bc = r_card[0];
        bi = r_idx[0];
        for (int w = 1; w < BLK2 / 32; ++w) {
            if (r_card[w] > bc || (r_card[w] == bc && r_idx[w] < bi)) {
                bc = r_card[w];
                bi = r_idx[w];
            }
        }
        meta[META_BEST_SEED] = (bi == INT_MAX || bc <= 0) ? -1 : seeds[bi];
    }
}

// Mark the accepted cluster's members, then rebuild the ascending lists of
// still-unclustered points and of the seeds that need recomputation.
//
// A seed's candidate cluster only depends on the unclustered points within the
// threshold around it.  Removing a cluster therefore only invalidates the
// cached cardinalities of seeds in the neighbourhood of one of its members;
// every other seed keeps its result and is skipped in the next round.
__global__ __launch_bounds__(BLK2) void markCompactKernel(
    unsigned char* __restrict__ clustered, unsigned char* __restrict__ dirty,
    const int* __restrict__ allMembers, int* __restrict__ meta,
    int* __restrict__ seeds, int* __restrict__ dirtyList,
    const int* __restrict__ nbrOff, const int* __restrict__ nbrIdx, const int N,
    const int haveNbrLists) {
    __shared__ ScanState scSeed;
    __shared__ ScanState scDirty;

    if (meta[META_NUM_SEEDS] == 0) return;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int cnt = meta[META_MEMBER_CNT];
    const int base = meta[META_MEMBER_TOT] - cnt;

    for (int t = tid; t < cnt; t += BLK2) clustered[allMembers[base + t]] = 1;
    if (haveNbrLists) {
        for (int m = 0; m < cnt; ++m) {
            const int p = allMembers[base + m];
            for (int t = nbrOff[p] + tid; t < nbrOff[p + 1]; t += BLK2) dirty[nbrIdx[t]] = 1;
        }
    }
    if (tid == 0) {
        scSeed.base = 0;
        scDirty.base = 0;
    }
    __syncthreads();

    for (int b = 0; b < N; b += BLK2) {
        const int i = b + tid;
        const bool alive = (i < N) && !clustered[i];
        const bool stale = alive && (!haveNbrLists || dirty[i]);
        const int p1 = orderedSlot(alive, scSeed, tid, lane, warp, BLK2 / 32);
        if (alive) seeds[p1] = i;
        __syncthreads();
        const int p2 = orderedSlot(stale, scDirty, tid, lane, warp, BLK2 / 32);
        if (stale) dirtyList[p2] = i;
        if (i < N) dirty[i] = 0;
        __syncthreads();
    }

    if (tid == 0) {
        meta[META_NUM_SEEDS] = scSeed.base;
        meta[META_NUM_DIRTY] = scDirty.base;
    }
}

// Generate synthetic 2D point data in clusters
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

// Main QT clustering algorithm (GPU driven)
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<Cluster> clusters;
    if (N <= 0) return clusters;

    const double thr2 = threshold * threshold;

    // ---- Device allocations -------------------------------------------------
    // All fixed-size buffers live in a single arena; individual cudaMalloc
    // calls are comparatively expensive and would dominate small problems.
    const size_t align = 256;
    auto roundUp = [align](size_t v) { return (v + align - 1) / align * align; };
    const size_t szPts = roundUp(sizeof(double2) * N);
    const size_t szFlag = roundUp(sizeof(unsigned char) * N);
    const size_t szIdx = roundUp(sizeof(int) * N);
    const size_t szMeta = roundUp(sizeof(int) * META_COUNT);
    char* arena = nullptr;
    CUDA_CHECK(cudaMalloc(&arena, szPts + 2 * szFlag + 6 * szIdx + szMeta));

    size_t at = 0;
    auto take = [&arena, &at](size_t bytes) {
        char* p = arena + at;
        at += bytes;
        return p;
    };
    double2* d_pts = reinterpret_cast<double2*>(take(szPts));
    unsigned char* d_clustered = reinterpret_cast<unsigned char*>(take(szFlag));
    unsigned char* d_dirty = reinterpret_cast<unsigned char*>(take(szFlag));
    int* d_seeds = reinterpret_cast<int*>(take(szIdx));
    int* d_dirtyList = reinterpret_cast<int*>(take(szIdx));
    int* d_card = reinterpret_cast<int*>(take(szIdx));
    int* d_allMembers = reinterpret_cast<int*>(take(szIdx));
    int* d_clusterSeed = reinterpret_cast<int*>(take(szIdx));
    int* d_clusterOff = reinterpret_cast<int*>(take(szIdx));
    int* d_meta = reinterpret_cast<int*>(take(szMeta));

    int* d_nbrOff = nullptr;
    int* d_nbrIdx = nullptr;
    int* d_nbrBuf = nullptr;
    double* d_mdBuf = nullptr;

    int h_meta[META_COUNT] = {0};

    std::vector<double2> hpts(N);
    for (int i = 0; i < N; ++i) hpts[i] = make_double2(points[i].x, points[i].y);
    CUDA_CHECK(cudaMemcpy(d_pts, hpts.data(), sizeof(double2) * N, cudaMemcpyHostToDevice));

    size_t freeMem = 0, totalMem = 0;
    CUDA_CHECK(cudaMemGetInfo(&freeMem, &totalMem));

    // ---- Precompute the threshold neighbourhood of every point -------------
    // Bounds the per-seed candidate set for the whole run and drives the
    // invalidation of cached cardinalities.  Skipped (falling back to a full
    // scan per seed) if the lists would not fit into device memory.
    std::vector<int> scratch(static_cast<size_t>(N) + 1);
    size_t maxSpan = static_cast<size_t>(N);
    {
        countNeighborsKernel<<<N, BLK2>>>(d_pts, N, thr2, d_card);
        CUDA_CHECK(cudaMemcpy(scratch.data(), d_card, sizeof(int) * N,
                              cudaMemcpyDeviceToHost));
        size_t total = 0;
        size_t maxDeg = 0;
        for (int i = 0; i < N; ++i) {
            total += static_cast<size_t>(scratch[i]);
            if (static_cast<size_t>(scratch[i]) > maxDeg) maxDeg = scratch[i];
        }
        if (total * sizeof(int) + (static_cast<size_t>(N) + 1) * sizeof(int) < freeMem / 3 &&
            total <= static_cast<size_t>(INT_MAX)) {
            size_t run = 0;
            for (int i = 0; i < N; ++i) {
                const size_t deg = static_cast<size_t>(scratch[i]);
                scratch[i] = static_cast<int>(run);
                run += deg;
            }
            scratch[N] = static_cast<int>(run);
            CUDA_CHECK(cudaMalloc(&d_nbrOff, sizeof(int) * (static_cast<size_t>(N) + 1)));
            CUDA_CHECK(cudaMalloc(&d_nbrIdx, sizeof(int) * (total > 0 ? total : 1)));
            CUDA_CHECK(cudaMemcpy(d_nbrOff, scratch.data(),
                                  sizeof(int) * (static_cast<size_t>(N) + 1),
                                  cudaMemcpyHostToDevice));
            fillNeighborsKernel<<<N, BLK2>>>(d_pts, N, thr2, d_nbrOff, d_nbrIdx);
            maxSpan = maxDeg;
        }
    }
    const int haveNbrLists = (d_nbrOff != nullptr) ? 1 : 0;

    // ---- Grid size and per-block scratch for oversized candidate lists ------
    int gridBlocks = std::min(N, 8192);
    if (maxSpan > static_cast<size_t>(SMEM_CAP)) {
        CUDA_CHECK(cudaMemGetInfo(&freeMem, &totalMem));
        const size_t perSlot = maxSpan * (sizeof(int) + sizeof(double));
        size_t maxSlots = (freeMem / 2) / perSlot;
        if (maxSlots < 1) maxSlots = 1;
        if (maxSlots > static_cast<size_t>(gridBlocks)) maxSlots = static_cast<size_t>(gridBlocks);
        while (maxSlots > 1 &&
               (cudaMalloc(&d_nbrBuf, sizeof(int) * maxSlots * maxSpan) != cudaSuccess ||
                cudaMalloc(&d_mdBuf, sizeof(double) * maxSlots * maxSpan) != cudaSuccess)) {
            if (d_nbrBuf) cudaFree(d_nbrBuf);
            if (d_mdBuf) cudaFree(d_mdBuf);
            d_nbrBuf = nullptr;
            d_mdBuf = nullptr;
            cudaGetLastError();
            maxSlots /= 2;
        }
        if (d_nbrBuf == nullptr) {
            CUDA_CHECK(cudaMalloc(&d_nbrBuf, sizeof(int) * maxSlots * maxSpan));
            CUDA_CHECK(cudaMalloc(&d_mdBuf, sizeof(double) * maxSlots * maxSpan));
        }
        gridBlocks = static_cast<int>(maxSlots);
    }
    const size_t slotStride = maxSpan;

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));

    initStateKernel<<<(N + 255) / 256, 256, 0, stream>>>(d_meta, d_seeds, d_dirtyList,
                                                         d_clustered, d_dirty, N);

    // ---- Main clustering loop ----------------------------------------------
    // A whole round runs on the device without host interaction, so a batch of
    // rounds is captured into a CUDA graph and replayed; rounds queued after
    // the last cluster return immediately.  The host only synchronises once per
    // batch to test for termination.
    const int kBatch = 8;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graphExec = nullptr;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    for (int b = 0; b < kBatch; ++b) {
        // Re-grow every seed whose neighbourhood changed in the last round.
        growKernel<<<gridBlocks, BLK, 0, stream>>>(
            d_pts, d_clustered, N, thr2, d_dirtyList, d_meta + META_NUM_DIRTY, d_nbrOff,
            d_nbrIdx, d_card, d_nbrBuf, d_mdBuf, slotStride, d_meta, nullptr, nullptr,
            nullptr, 0);
        // Largest candidate cluster wins (lowest seed index on ties).
        pickBestKernel<<<1, BLK2, 0, stream>>>(d_card, d_seeds, d_meta);
        // Re-grow the winner to materialise and record its member list.
        growKernel<<<1, BLK, 0, stream>>>(
            d_pts, d_clustered, N, thr2, d_meta + META_BEST_SEED, d_meta + META_ONE,
            d_nbrOff, d_nbrIdx, d_card, d_nbrBuf, d_mdBuf, slotStride, d_meta,
            d_allMembers, d_clusterSeed, d_clusterOff, 1);
        // Retire the accepted cluster and prepare the next round's work lists.
        markCompactKernel<<<1, BLK2, 0, stream>>>(d_clustered, d_dirty, d_allMembers,
                                                  d_meta, d_seeds, d_dirtyList, d_nbrOff,
                                                  d_nbrIdx, N, haveNbrLists);
    }
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, 0));

    int lastClusterCount = -1;
    while (true) {
        CUDA_CHECK(cudaGraphLaunch(graphExec, stream));
        CUDA_CHECK(cudaMemcpyAsync(h_meta, d_meta, sizeof(int) * META_COUNT,
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if (h_meta[META_NUM_SEEDS] == 0) break;
        // No cluster could be formed from the remaining points (mirrors the
        // reference's "no more clusters" bail-out).
        if (h_meta[META_NUM_CLUST] == lastClusterCount) break;
        lastClusterCount = h_meta[META_NUM_CLUST];
    }

    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
    CUDA_CHECK(cudaStreamDestroy(stream));

    // ---- Read back the clusters --------------------------------------------
    const int numClusters = h_meta[META_NUM_CLUST];
    const int memberTotal = h_meta[META_MEMBER_TOT];
    std::vector<int> hSeed(numClusters), hOff(numClusters), hMembers(memberTotal);
    if (numClusters > 0) {
        CUDA_CHECK(cudaMemcpy(hSeed.data(), d_clusterSeed, sizeof(int) * numClusters,
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hOff.data(), d_clusterOff, sizeof(int) * numClusters,
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hMembers.data(), d_allMembers, sizeof(int) * memberTotal,
                              cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaGetLastError());

    clusters.reserve(numClusters);
    for (int c = 0; c < numClusters; ++c) {
        const int end = (c + 1 < numClusters) ? hOff[c + 1] : memberTotal;
        Cluster cluster;
        cluster.seed_point = hSeed[c];
        cluster.members.assign(hMembers.begin() + hOff[c], hMembers.begin() + end);
        clusters.push_back(cluster);
    }

    cudaFree(arena);
    cudaFree(d_nbrOff);
    cudaFree(d_nbrIdx);
    cudaFree(d_nbrBuf);
    cudaFree(d_mdBuf);

    return clusters;
}

// Validation: check that clusters satisfy the QT clustering properties
bool validateClusters(const std::vector<Cluster>& clusters,
                     const std::vector<Point>& points,
                     const double threshold) {
    bool valid = true;

    printf("Validating clusters:\n");

    // Check each cluster
    for (size_t c = 0; c < clusters.size(); ++c) {
        const auto& cluster = clusters[c];
        double max_diameter = 0.0;

        // Check diameter (max distance between any two points)
        for (size_t i = 0; i < cluster.members.size(); ++i) {
            for (size_t j = i + 1; j < cluster.members.size(); ++j) {
                const double dist = distance(points[cluster.members[i]],
                                           points[cluster.members[j]]);
                max_diameter = std::max(max_diameter, dist);
            }
        }

        if (c < 10) { // Print first 10 clusters
            printf("  Cluster %zu: size=%zu, seed=%d, diameter=%.4f\n",
                   c, cluster.members.size(), cluster.seed_point, max_diameter);
        }

        // Validate diameter is within threshold
        if (max_diameter > threshold * 1.001) { // Allow small numerical error
            printf("ERROR: Cluster %zu has diameter %.4f > threshold %.4f\n",
                   c, max_diameter, threshold);
            valid = false;
        }
    }

    // Check for duplicate memberships
    std::vector<int> membership(points.size(), -1);
    for (size_t c = 0; c < clusters.size(); ++c) {
        for (size_t i = 0; i < clusters[c].members.size(); ++i) {
            const int member = clusters[c].members[i];
            if (membership[member] >= 0) {
                printf("ERROR: Point %d appears in multiple clusters (%d and %zu)\n",
                       member, membership[member], c);
                valid = false;
            }
            membership[member] = static_cast<int>(c);
        }
    }

    // Count clustered points
    int clustered_count = 0;
    for (size_t i = 0; i < membership.size(); ++i) {
        if (membership[i] >= 0) clustered_count++;
    }

    printf("Total points: %zu, Clustered: %d, Unclustered: %zu\n",
           points.size(), clustered_count, points.size() - clustered_count);

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
            printUsage(argv[0]);
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }

    if (num_points <= 0 || threshold <= 0.0) {
        printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
               num_points, threshold);
        return 1;
    }

    printf("QT Clustering Benchmark\n");
    printf("Number of points: %d\n", num_points);
    printf("Distance threshold: %.2f\n", threshold);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");

    // Generate synthetic data
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);

    // Initialise the CUDA context (and eagerly load the device code) up front
    // so that neither is attributed to the measured clustering phase.
    setenv("CUDA_MODULE_LOADING", "EAGER", 1);
    CUDA_CHECK(cudaFree(0));

    // Perform QT clustering
    auto cluster_start = std::chrono::high_resolution_clock::now();

    const std::vector<Cluster> clusters = qtClustering(points, threshold);

    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);

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

    const double avg_cluster_size = clusters.empty() ? 0.0 :
        static_cast<double>(total_clustered) / clusters.size();

    printf("Points clustered: %d / %d (%.1f%%)\n",
           total_clustered, num_points,
           100.0 * total_clustered / num_points);
    printf("Average cluster size: %.2f\n", avg_cluster_size);
    printf("Maximum cluster size: %d\n", max_cluster_size);

    // Performance metrics
    const double time_sec = cluster_time.count() / 1000.0;
    const double clusters_per_sec = clusters.size() / time_sec;
    const double points_per_sec = num_points / time_sec;
    printf("Performance: %.1f clusters/s, %.1f points/s\n",
           clusters_per_sec, points_per_sec);

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
    if (validate) {
        const bool valid = validateClusters(clusters, points, threshold);

        if (valid) {
            printf("Validation: PASSED\n");
            return 0;
        } else {
            printf("Validation: FAILED\n");
            return 1;
        }
    }

    return 0;
}
