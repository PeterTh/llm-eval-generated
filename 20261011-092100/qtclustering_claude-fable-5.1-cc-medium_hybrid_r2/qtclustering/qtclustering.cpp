// QT Clustering Benchmark - Hybrid MPI + OpenMP + CUDA Version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// Parallelization strategy (semantics identical to the sequential version):
//
//  * Every outer iteration evaluates one candidate cluster per unclustered
//    seed point. These evaluations are independent, so the seeds are split
//    across MPI ranks (one GPU per rank, seed s belongs to rank s % nranks)
//    and, inside a rank, across CUDA thread blocks: one block grows one
//    candidate cluster at a time, pulling seeds from a shared work list.
//
//  * A candidate cluster grown from seed s can only ever contain points whose
//    distance to s is below the threshold. Each block therefore first compacts
//    those points into a per-block candidate list and then grows the cluster
//    incrementally, keeping for every candidate the maximum distance to the
//    current members (the "diameter if added"). Each growth step is one
//    block-wide min-reduction over (diameter, index), which reproduces the
//    "smallest max-distance, lowest index on ties" selection of the original.
//    Distances are handled in single precision with rigorous error bounds and
//    fall back to the exact double arithmetic of the original only when a
//    comparison is too close to call (see the kernel section).
//
//  * Pruning (never changes the result): the number of unclustered neighbours
//    within the threshold is an exact upper bound on the cardinality a seed
//    can reach; a Jung-disc bound and "current size + still viable
//    candidates" tighten it during growth. Seeds are processed in decreasing
//    bound order and dropped as soon as their bound falls below the best
//    cardinality already found. Since the winner is the seed with the maximum
//    cardinality (lowest index on ties), dropped seeds can never be it.
//
//  * Memoization across iterations: a seed's cluster depends only on the
//    unclustered points within the threshold, whose count (the bound) is
//    maintained exactly on the GPU. While that count is unchanged, the
//    cardinality / upper bound found earlier remains valid and the seed is
//    not re-evaluated.
//
//  * The winning seed is chosen with one MPI_Allreduce; its member list was
//    recorded on the GPU by the block that evaluated it and is broadcast by
//    the owning rank. OpenMP handles the host-side O(N) bookkeeping (bound
//    refresh, counting sort of seeds, memoization tables), validation, and
//    the (fallback) host re-growth of a cluster, all with the same arithmetic
//    as the GPU (both compiled without FMA contraction).

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <climits>
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

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                         \
                    cudaGetErrorString(err__), __FILE__, __LINE__);             \
            MPI_Abort(MPI_COMM_WORLD, 1);                                       \
        }                                                                       \
    } while (0)

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
__host__ __device__ inline double distance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return sqrt(dx * dx + dy * dy);
}

// ---------------------------------------------------------------------------
// CUDA kernels
// ---------------------------------------------------------------------------
//
// Exactness scheme. The sequential code compares distances d = fl(sqrt(d2)).
// Here we work with exact squared distances d2 (double) and decide every
// comparison the original makes without evaluating sqrt unless the two
// operands are within a few ulps of each other. Outside the margin,
// monotonicity of the correctly rounded sqrt makes the outcome certain;
// inside it, sqrt is evaluated and the original comparison is performed
// literally. On top of that, a single precision estimate with a
// conservative absolute error bound (EPS) is used to skip candidates whose
// running maximum provably does not change, and to reject points that are
// provably outside the threshold.

#define QT_BLOCK 256
#define QT_WARPS (QT_BLOCK / 32)
#define QT_JG 8              // Jung-bound centre grid points per axis
#define QT_JC (QT_JG * QT_JG)
#ifndef QT_MINBLOCKS
#define QT_MINBLOCKS 3
#endif
#define QT_CAND_BYTES (sizeof(float2) + sizeof(float))   // shared memory per candidate
#define QT_SHARED_MIN 1024   // minimum candidate capacity of shared memory per block
#define QT_SHARED_MAX 4096   // maximum candidate capacity of shared memory per block

struct ThrParams {
    double thr;         // threshold
    long long t2bits;   // bit pattern of thr^2
    float eps;          // error bound of float squared distances from seed-relative coordinates
    float tlo_f;        // estimate below this: certainly within threshold
    float thi_f;        // estimate above this: certainly outside threshold
    float sel_margin;   // estimates within this of the minimum need exact treatment
    float abs_hi;       // squared distance from absolute float coordinates above
                        // this: certainly outside threshold
    float jr;           // Jung radius thr / sqrt(3): any set of diameter < thr
                        // fits in a disc of this radius
    float jh;           // spacing of the grid of candidate disc centres
    float jr2;          // (jr + jh / sqrt(2))^2 with float error margins
};

__host__ __device__ inline double dist2(const Point& a, const Point& b) {
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    return dx * dx + dy * dy;
}

// Bit pattern of a non-negative double: monotonic in the value, so integer
// comparisons replace (slow) FP64 comparisons. A pattern difference of more
// than QT_ULP_MARGIN implies a relative difference above 2^-51, which is
// enough for the correctly rounded square roots to differ in the same order.
#define QT_ULP_MARGIN 24LL

__host__ __device__ __forceinline__ long long dbits(const double x) {
#ifdef __CUDA_ARCH__
    return __double_as_longlong(x);
#else
    long long r;
    memcpy(&r, &x, sizeof(r));
    return r;
#endif
}

// true iff fl(sqrt(m)) < thr  (exactly as the sequential code evaluates it)
__host__ __device__ __forceinline__ bool withinThr(const double m, const ThrParams& tp) {
    const long long a = dbits(m);
    if (a + QT_ULP_MARGIN < tp.t2bits) return true;
    if (a > tp.t2bits + QT_ULP_MARGIN) return false;
    return sqrt(m) < tp.thr;
}

// true iff (fl(sqrt(m)), c) is lexicographically smaller than (fl(sqrt(B)), bi)
__host__ __device__ __forceinline__ bool exactLess(const double m, const int c,
                                                   const double B, const int bi) {
    const long long a = dbits(m);
    const long long b = dbits(B);
    if (a + QT_ULP_MARGIN < b) return true;
    if (a > b + QT_ULP_MARGIN) return false;
    const double v = sqrt(m);
    const double bv = sqrt(B);
    return v < bv || (v == bv && c < bi);
}

__device__ __forceinline__ float dist2f(const float2 a, const float2 b) {
    const float dx = a.x - b.x;
    const float dy = a.y - b.y;
    return dx * dx + dy * dy;
}

// Lexicographic (value, index) minimum across a warp; slot < 0 marks an
// empty entry.
__device__ __forceinline__ void warpMin(double& v, int& idx, int& slot) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const double ov = __shfl_down_sync(0xffffffffu, v, off);
        const int oi = __shfl_down_sync(0xffffffffu, idx, off);
        const int os = __shfl_down_sync(0xffffffffu, slot, off);
        if (os >= 0 && (slot < 0 || exactLess(ov, oi, v, idx))) {
            v = ov; idx = oi; slot = os;
        }
    }
}

// Block-wide lexicographic minimum (slot < 0 = empty). Valid in all threads.
// The caller must place a __syncthreads() between two consecutive calls.
__device__ __forceinline__ void blockMinLex(double& v, int& idx, int& slot,
                                            double* s_v, int* s_i, int* s_s) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    warpMin(v, idx, slot);
    if (lane == 0) { s_v[warp] = v; s_i[warp] = idx; s_s[warp] = slot; }
    __syncthreads();
    if (warp == 0) {
        v = (lane < QT_WARPS) ? s_v[lane] : INFINITY;
        idx = (lane < QT_WARPS) ? s_i[lane] : INT_MAX;
        slot = (lane < QT_WARPS) ? s_s[lane] : -1;
        warpMin(v, idx, slot);
        if (lane == 0) { s_v[0] = v; s_i[0] = idx; s_s[0] = slot; }
    }
    __syncthreads();
    v = s_v[0]; idx = s_i[0]; slot = s_s[0];
}

// Merge (minimum, second minimum, slot of minimum) pairs.
__device__ __forceinline__ void merge2(float& v1, float& v2, int& s1,
                                       const float o1, const float o2, const int os) {
    if (o1 < v1) {
        v2 = fminf(v1, o2);
        v1 = o1;
        s1 = os;
    } else {
        v2 = fminf(v2, o1);
    }
}

// Block-wide (minimum, second minimum, slot of minimum) and integer sum.
// Valid in all threads. The caller must place a __syncthreads() between two
// consecutive calls.
__device__ __forceinline__ void blockMin2Sum(float& v1, float& v2, int& s1, int& sum,
                                             float* s_v1, float* s_v2, int* s_s, int* s_n) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const float o1 = __shfl_down_sync(0xffffffffu, v1, off);
        const float o2 = __shfl_down_sync(0xffffffffu, v2, off);
        const int os = __shfl_down_sync(0xffffffffu, s1, off);
        sum += __shfl_down_sync(0xffffffffu, sum, off);
        merge2(v1, v2, s1, o1, o2, os);
    }
    if (lane == 0) { s_v1[warp] = v1; s_v2[warp] = v2; s_s[warp] = s1; s_n[warp] = sum; }
    __syncthreads();
    if (warp == 0) {
        v1 = (lane < QT_WARPS) ? s_v1[lane] : INFINITY;
        v2 = (lane < QT_WARPS) ? s_v2[lane] : INFINITY;
        s1 = (lane < QT_WARPS) ? s_s[lane] : -1;
        sum = (lane < QT_WARPS) ? s_n[lane] : 0;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const float o1 = __shfl_down_sync(0xffffffffu, v1, off);
            const float o2 = __shfl_down_sync(0xffffffffu, v2, off);
            const int os = __shfl_down_sync(0xffffffffu, s1, off);
            sum += __shfl_down_sync(0xffffffffu, sum, off);
            merge2(v1, v2, s1, o1, o2, os);
        }
        if (lane == 0) { s_v1[0] = v1; s_v2[0] = v2; s_s[0] = s1; s_n[0] = sum; }
    }
    __syncthreads();
    v1 = s_v1[0]; v2 = s_v2[0]; s1 = s_s[0]; sum = s_n[0];
}

// Initial exact upper bound for every point: number of points (including
// itself) within the threshold distance.
__global__ void initBoundsKernel(const Point* __restrict__ pts, const float2* __restrict__ ptsf,
                                 const int N, const ThrParams tp, int* __restrict__ bounds) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= N) return;
    const Point me = pts[p];
    const float2 mef = ptsf[p];
    int cnt = 0;
    for (int q = 0; q < N; ++q) {
        if (dist2f(mef, ptsf[q]) > tp.thi_f) continue;
        if (withinThr(dist2(me, pts[q]), tp)) cnt++;
    }
    bounds[p] = cnt;
}

// Subtract the newly clustered members from the bounds of unclustered points.
__global__ void updateBoundsKernel(const Point* __restrict__ pts, const float2* __restrict__ ptsf,
                                   const int N, const unsigned char* __restrict__ clustered,
                                   const int* __restrict__ members, const int m,
                                   const ThrParams tp, int* __restrict__ bounds) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= N || clustered[p]) return;
    const Point me = pts[p];
    const float2 mef = ptsf[p];
    int cnt = 0;
    for (int i = 0; i < m; ++i) {
        const int q = members[i];
        if (dist2f(mef, ptsf[q]) > tp.thi_f) continue;
        if (withinThr(dist2(me, pts[q]), tp)) cnt++;
    }
    bounds[p] -= cnt;
}

// Grow one candidate cluster per thread block; blocks pull seeds dynamically
// from a sorted work list. Writes the cardinality of every seed, or minus an
// upper bound on it if the seed was pruned.
//
// Per candidate i the block keeps:
//   cidx[i]    point index,
//   cxy[i]     single precision coordinates relative to the seed (small
//              magnitudes, hence a tight error bound tp.eps on squared
//              distances computed from them),
//   carg[i]    ordinal (position in the member list) of the current member
//              farthest from the candidate, so the exact (double) max squared
//              distance is dist2(pts[cidx[i]], point of that member),
//   cmf[i]     single precision estimate of that max (within tp.eps), +inf once
//              the candidate is dead (max distance >= threshold).
// Exact double arithmetic is only used for candidates whose estimate is within
// the error bound of the threshold, of a new member distance, or of the
// step's minimum. Such cases are rare, so warps almost never diverge into the
// (slow) FP64 paths. Members are recorded by candidate slot and translated to
// point indices lazily (at compactions and on completion), which keeps
// dependent global loads off the per-step critical path.
__global__ void __launch_bounds__(QT_BLOCK, QT_MINBLOCKS)
seedCardinalityKernel(const Point* __restrict__ pts, const float2* __restrict__ ptsf,
                      const int N,
                      const unsigned char* __restrict__ clustered,
                      const int* __restrict__ seeds, const int nseeds,
                      const int* __restrict__ bounds, const ThrParams tp,
                      int* __restrict__ g_idx, int* __restrict__ g_arg,
                      float* __restrict__ g_mf, float2* __restrict__ g_xy,
                      const int scratch_stride, const int shared_cap,
                      int* __restrict__ g_members,        // [grid][2][member_stride]
                      const int member_stride,
                      int* __restrict__ blk_best,         // [grid][3]: card, seed, buffer
                      int* __restrict__ out_card,         // [nseeds]: card, or -(upper bound) if pruned
                      int* __restrict__ work_counter,
                      int* __restrict__ best_hint) {
    __shared__ double s_rv[QT_WARPS];
    __shared__ float s_rf1[QT_WARPS];
    __shared__ float s_rf2[QT_WARPS];
    __shared__ int s_ri[QT_WARPS];
    __shared__ int s_rs[QT_WARPS];
    __shared__ int s_rn[QT_WARPS];
    __shared__ int s_pos;
    __shared__ int s_count;
    __shared__ int s_hint;
    __shared__ int s_ub;
    __shared__ float2 s_ctr[QT_JC];              // Jung disc centres (seed-relative)
    __shared__ int s_feas[QT_JC];                // centre still within range of all members
    __shared__ int s_cnt[QT_WARPS][QT_JC];
    extern __shared__ float2 s_dyn[];            // [shared_cap] float2 xy, then [shared_cap] mf
    float2* const s_xy = s_dyn;
    float* const s_mf = reinterpret_cast<float*>(s_dyn + shared_cap);

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const unsigned lanemask_lt = (1u << lane) - 1u;
    const float eps2 = 2.0f * tp.eps;
    const float jr2 = tp.jr2;
    const float tlo_f = tp.tlo_f;
    const float thi_f = tp.thi_f;
    const float abs_hi = tp.abs_hi;

    // Per-block scratch: two buffers of each array (ping-pong for compaction).
    int* const idx_buf[2] = {g_idx + static_cast<size_t>(blockIdx.x) * 2 * scratch_stride,
                             g_idx + (static_cast<size_t>(blockIdx.x) * 2 + 1) * scratch_stride};
    int* const arg_buf[2] = {g_arg + static_cast<size_t>(blockIdx.x) * 2 * scratch_stride,
                             g_arg + (static_cast<size_t>(blockIdx.x) * 2 + 1) * scratch_stride};
    float* const mf_buf[2] = {g_mf + static_cast<size_t>(blockIdx.x) * 2 * scratch_stride,
                              g_mf + (static_cast<size_t>(blockIdx.x) * 2 + 1) * scratch_stride};
    float2* const xy_buf[2] = {g_xy + static_cast<size_t>(blockIdx.x) * 2 * scratch_stride,
                               g_xy + (static_cast<size_t>(blockIdx.x) * 2 + 1) * scratch_stride};
    int* const mem_buf = g_members + static_cast<size_t>(blockIdx.x) * 2 * member_stride;

    // Best completed seed of this block (block-uniform values).
    int best_card = 0;
    int best_seed = INT_MAX;
    int cur_buf = 0;                 // member buffer written for the current seed

    while (true) {
        if (tid == 0) {
            s_pos = atomicAdd(work_counter, 1);
            s_count = 0;
            s_hint = *reinterpret_cast<volatile int*>(best_hint);
        }
        __syncthreads();
        const int pos = s_pos;
        if (pos >= nseeds) break;
        const int seed = seeds[pos];
        const Point sp = pts[seed];
        const float2 spf = ptsf[seed];

        // Prune: this seed cannot beat the best cardinality found so far.
        if (bounds[seed] < s_hint) {
            if (tid == 0) out_card[pos] = -(s_hint - 1);
            __syncthreads();
            continue;
        }

        int* cidx = idx_buf[0];
        int* carg = arg_buf[0];
        float* cmf = mf_buf[0];
        const float2* cxy = xy_buf[0];
        int gbuf = 0;                 // global buffer currently holding cidx/carg

        // Phase 1: compact unclustered points within the threshold of the seed
        // (warp-aggregated atomics on the shared counter). Absolute float
        // coordinates reject far points; the rest are classified exactly.
        for (int base = 0; base < N; base += QT_BLOCK) {
            const int c = base + tid;
            bool ok = false;
            double d2 = 0.0;
            Point cp;
            if (c < N && c != seed && !clustered[c] && !(dist2f(ptsf[c], spf) > abs_hi)) {
                cp = pts[c];
                d2 = dist2(cp, sp);
                ok = withinThr(d2, tp);
            }
            const unsigned mask = __ballot_sync(0xffffffffu, ok);
            if (mask) {
                const int leader = __ffs(mask) - 1;
                int slotbase = 0;
                if (lane == leader) slotbase = atomicAdd(&s_count, __popc(mask));
                slotbase = __shfl_sync(0xffffffffu, slotbase, leader);
                if (ok) {
                    const int slot = slotbase + __popc(mask & lanemask_lt);
                    cidx[slot] = c;
                    mf_buf[0][slot] = static_cast<float>(d2);
                    xy_buf[0][slot] = make_float2(static_cast<float>(cp.x - sp.x),
                                                  static_cast<float>(cp.y - sp.y));
                    carg[slot] = 0;
                }
            }
        }
        if (tid < QT_JC) {
            const float gx = -tp.jr + (static_cast<float>(tid % QT_JG) + 0.5f) * tp.jh;
            const float gy = -tp.jr + (static_cast<float>(tid / QT_JG) + 0.5f) * tp.jh;
            s_ctr[tid] = make_float2(gx, gy);
            s_feas[tid] = (gx * gx + gy * gy <= jr2) ? 1 : 0;   // seed is a member
        }
        __syncthreads();
        int n = s_count;

        // Exact bound check after compaction.
        if (n + 1 < s_hint) {
            if (tid == 0) out_card[pos] = -(n + 1);
            __syncthreads();
            continue;
        }

        bool in_shared = false;
        if (n <= shared_cap) {
            for (int i = tid; i < n; i += QT_BLOCK) {
                s_mf[i] = mf_buf[0][i];
                s_xy[i] = xy_buf[0][i];
            }
            cmf = s_mf;
            cxy = s_xy;
            in_shared = true;
        }
        __syncthreads();

        // Phase 2: greedy growth.
        int* const members = mem_buf + cur_buf * member_stride;
        if (tid == 0) members[0] = seed;
        int card = 1;
        int translated = 1;                     // members[0..translated) hold point indices
        float2 npf = make_float2(0.0f, 0.0f);   // newest member, relative to the seed
        // Point of member ordinal k (rare, exact paths only).
        auto memberPoint = [&](const int k) -> Point {
            if (k == 0) return sp;
            const int v = members[k];
            return pts[k < translated ? v : cidx[v]];
        };
        while (true) {
            int hint_now = 0;
            if (tid == 0) hint_now = *reinterpret_cast<volatile int*>(best_hint);
            const int newest = card - 1;

            // Pass 1: update the running maxima with the newest member, kill
            // candidates that reached the threshold, find the two smallest
            // float estimates and count the live candidates.
            float fmin = INFINITY, fmin2 = INFINITY;
            int fslot = -1;
            int alive = 0;
            for (int i = tid; i < n; i += QT_BLOCK) {
                float mf = cmf[i];
                if (mf == INFINITY) continue;
                const float d2f = dist2f(cxy[i], npf);
                bool changed = false;
                if (d2f > mf + eps2) {
                    mf = d2f;
                    changed = true;
                } else if (!(d2f < mf - eps2)) {
                    // ambiguous: exact comparison
                    const Point cp = pts[cidx[i]];
                    const double m_old = dist2(cp, memberPoint(carg[i]));
                    const double d2 = dist2(cp, memberPoint(newest));
                    if (d2 > m_old) {
                        mf = d2f;
                        changed = true;
                    }
                }
                if (changed) {
                    carg[i] = newest;
                    bool ok;
                    if (mf < tlo_f) ok = true;
                    else if (mf > thi_f) ok = false;
                    else ok = withinThr(dist2(pts[cidx[i]], memberPoint(newest)), tp);
                    if (!ok) mf = INFINITY;
                    cmf[i] = mf;
                    if (!ok) continue;
                }
                alive++;
                if (mf < fmin) { fmin2 = fmin; fmin = mf; fslot = i; }
                else fmin2 = fminf(fmin2, mf);
            }
            if (tid == 0) s_hint = hint_now;
            blockMin2Sum(fmin, fmin2, fslot, alive, s_rf1, s_rf2, s_rs, s_rn);
            if (alive == 0) break;             // no candidate can be added
            if (card + alive < s_hint) {       // cannot beat the best: give up
                card = -(card + alive);
                break;
            }

            // Jung bound: the final cluster lies in a disc of radius jr whose
            // centre is within jr of every member; some grid centre is within
            // jr + jh/sqrt(2) of that centre and of every member. Count the
            // alive candidates in the enlarged disc around each feasible
            // grid centre; the maximum bounds the number of future members.
            if (card == 1 || (card & 63) == 0) {
                const float2 g0 = s_ctr[lane];
                const float2 g1 = s_ctr[lane + 32];
                int c0 = 0, c1 = 0;
                for (int i = warp; i < n; i += QT_WARPS) {
                    if (cmf[i] == INFINITY) continue;
                    const float2 p = cxy[i];
                    c0 += (dist2f(p, g0) <= jr2) ? 1 : 0;
                    c1 += (dist2f(p, g1) <= jr2) ? 1 : 0;
                }
                s_cnt[warp][lane] = c0;
                s_cnt[warp][lane + 32] = c1;
                __syncthreads();
                if (warp == 0) {
                    int t0 = 0, t1 = 0;
#pragma unroll
                    for (int w = 0; w < QT_WARPS; ++w) { t0 += s_cnt[w][lane]; t1 += s_cnt[w][lane + 32]; }
                    int m = max(s_feas[lane] ? t0 : 0, s_feas[lane + 32] ? t1 : 0);
#pragma unroll
                    for (int off = 16; off > 0; off >>= 1) m = max(m, __shfl_down_sync(0xffffffffu, m, off));
                    if (lane == 0) s_ub = m;
                }
                __syncthreads();
                const int ub = card + min(alive, s_ub);
                if (ub < s_hint) {
                    card = -ub;
                    break;
                }
            }

            // Estimates above `cut` belong to candidates that can neither be
            // the exact minimum nor tie with it.
            const float cut = fmaxf(fmin + tp.sel_margin, fmin * (1.0f + 1.0f / 1048576.0f));
            int bs = fslot;
            if (!(fmin2 > cut)) {
                // Pass 2: exact lexicographic minimum among the near-minimum ones.
                double B = INFINITY;
                int bi = INT_MAX;
                bs = -1;
                __syncthreads();
                for (int i = tid; i < n; i += QT_BLOCK) {
                    const float mf = cmf[i];
                    if (mf > cut) continue;    // also skips dead (+inf)
                    const int c = cidx[i];
                    const double m = dist2(pts[c], memberPoint(carg[i]));
                    if (exactLess(m, c, B, bi)) { B = m; bi = c; bs = i; }
                }
                blockMinLex(B, bi, bs, s_rv, s_ri, s_rs);
            }
            if (tid == 0) { cmf[bs] = INFINITY; members[card] = bs; }
            card++;
            npf = cxy[bs];
            if (tid < QT_JC && s_feas[tid] && dist2f(s_ctr[tid], npf) > jr2) s_feas[tid] = 0;
            __syncthreads();

            // Compaction: drop dead candidates once they are the majority.
            if (2 * (alive - 1) < n && n >= 2 * QT_BLOCK) {
                // Translate recorded member slots before the slots change.
                for (int k = translated + tid; k < card; k += QT_BLOCK) members[k] = cidx[members[k]];
                if (tid == 0) s_count = 0;
                __syncthreads();
                translated = card;
                int* const nidx = idx_buf[gbuf ^ 1];
                int* const narg = arg_buf[gbuf ^ 1];
                float* const nmf = in_shared ? mf_buf[0] : mf_buf[gbuf ^ 1];
                float2* const nxy = in_shared ? xy_buf[0] : xy_buf[gbuf ^ 1];
                for (int base = 0; base < n; base += QT_BLOCK) {
                    const int i = base + tid;
                    const bool ok = (i < n) && (cmf[i] != INFINITY);
                    const unsigned mask = __ballot_sync(0xffffffffu, ok);
                    if (mask) {
                        const int leader = __ffs(mask) - 1;
                        int slotbase = 0;
                        if (lane == leader) slotbase = atomicAdd(&s_count, __popc(mask));
                        slotbase = __shfl_sync(0xffffffffu, slotbase, leader);
                        if (ok) {
                            const int slot = slotbase + __popc(mask & lanemask_lt);
                            nidx[slot] = cidx[i];
                            narg[slot] = carg[i];
                            nmf[slot] = cmf[i];
                            nxy[slot] = cxy[i];
                        }
                    }
                }
                __syncthreads();
                n = s_count;
                gbuf ^= 1;
                cidx = nidx;
                carg = narg;
                if (in_shared) {
                    for (int i = tid; i < n; i += QT_BLOCK) {
                        s_mf[i] = nmf[i];
                        s_xy[i] = nxy[i];
                    }
                } else {
                    cmf = nmf;
                    cxy = nxy;
                }
                __syncthreads();
            }
        }

        if (card > best_card || (card == best_card && seed < best_seed)) {
            best_card = card;
            best_seed = seed;
            __syncthreads();         // members[] written by thread 0 visible to all
            for (int k = translated + tid; k < card; k += QT_BLOCK) members[k] = cidx[members[k]];
            cur_buf ^= 1;            // keep this member list, write the next seed elsewhere
        }
        if (tid == 0) {
            out_card[pos] = card;
            if (card > 0) atomicMax(best_hint, card);
        }
        __syncthreads();
    }
    if (tid == 0) {
        blk_best[3 * blockIdx.x + 0] = best_card;
        blk_best[3 * blockIdx.x + 1] = best_seed;
        blk_best[3 * blockIdx.x + 2] = cur_buf ^ 1;
    }
}

// ---------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------

// Re-grow the candidate cluster of one seed on the host (OpenMP). Uses the
// same incremental max-distance scheme as the GPU kernel, so the result is
// identical to the kernel's cardinality and to the sequential algorithm.
static void growClusterHost(const int seed,
                            const std::vector<unsigned char>& clustered,
                            const std::vector<Point>& points,
                            const ThrParams& tp,
                            std::vector<int>& members) {
    const int N = static_cast<int>(points.size());
    const Point sp = points[seed];

    // Candidate compaction (ascending index order).
    std::vector<int> cidx;
    std::vector<double> cmd;
    cidx.reserve(1024);
    cmd.reserve(1024);
    for (int c = 0; c < N; ++c) {
        if (c == seed || clustered[c]) continue;
        const double d2 = dist2(points[c], sp);
        if (withinThr(d2, tp)) {
            cidx.push_back(c);
            cmd.push_back(d2);
        }
    }
    const int n = static_cast<int>(cidx.size());

    members.clear();
    members.push_back(seed);
    if (n == 0) return;

    const int nthreads = std::max(1, std::min(omp_get_max_threads(), (n + 255) / 256));
    std::vector<double> t_v(nthreads);
    std::vector<int> t_i(nthreads), t_s(nthreads);
    int newest = seed;
    bool done = false;

#pragma omp parallel num_threads(nthreads)
    {
        const int t = omp_get_thread_num();
        const int chunk = (n + nthreads - 1) / nthreads;
        const int lo = t * chunk;
        const int hi = std::min(n, lo + chunk);
        while (true) {
            const Point np = points[newest];
            double B = std::numeric_limits<double>::infinity();
            int bi = std::numeric_limits<int>::max();
            int bs = -1;
            for (int i = lo; i < hi; ++i) {
                double m = cmd[i];
                if (m == std::numeric_limits<double>::infinity()) continue;
                const int c = cidx[i];
                const double d2 = dist2(points[c], np);
                if (d2 > m) {
                    m = d2;
                    if (!withinThr(m, tp)) {
                        cmd[i] = std::numeric_limits<double>::infinity();
                        continue;
                    }
                    cmd[i] = m;
                }
                if (exactLess(m, c, B, bi)) {
                    B = m; bi = c; bs = i;
                }
            }
            t_v[t] = B; t_i[t] = bi; t_s[t] = bs;
#pragma omp barrier
#pragma omp single
            {
                double gv = std::numeric_limits<double>::infinity();
                int gi = std::numeric_limits<int>::max();
                int gs = -1;
                for (int k = 0; k < nthreads; ++k) {
                    if (t_s[k] >= 0 && exactLess(t_v[k], t_i[k], gv, gi)) {
                        gv = t_v[k]; gi = t_i[k]; gs = t_s[k];
                    }
                }
                if (gs >= 0) {
                    members.push_back(gi);
                    cmd[gs] = std::numeric_limits<double>::infinity();
                    newest = gi;
                } else {
                    done = true;
                }
            } // implicit barrier
            if (done) break;
        }
    }
}

static ThrParams makeThrParams(const double threshold, const std::vector<Point>& points) {
    ThrParams tp;
    tp.thr = threshold;
    const double t2 = threshold * threshold;
    tp.t2bits = dbits(t2);
    double E = 1.0e-30;
    for (const Point& p : points) E = std::max(E, std::max(std::fabs(p.x), std::fabs(p.y)));
    // Seed-relative coordinates of candidates and members have magnitude below
    // R = min(threshold, 2E). The float squared distance computed from them has
    // absolute error below 48 R^2 2^-24; use R^2 2^-16.
    const double R = std::min(threshold, 2.0 * E);
    tp.eps = static_cast<float>(R * R * (1.0 / 65536.0));
    const float t2f = static_cast<float>(t2);
    const float rel = 1.0f / 1048576.0f;   // 2^-20
    tp.tlo_f = t2f * (1.0f - rel) - tp.eps;
    tp.thi_f = t2f * (1.0f + rel) + tp.eps;
    // Candidates with exact max m' > m* (1 + 2^-50) can neither be the minimum
    // nor tie with it after rounding of sqrt; estimates are within eps of the
    // exact values. (A relative 2^-20 margin is applied at the use site.)
    tp.sel_margin = 4.0f * tp.eps;
    // Absolute float coordinates (magnitude <= E): error below E^2 2^-18.
    const float abs_eps = static_cast<float>(E * E * (1.0 / 16384.0));
    tp.abs_hi = t2f * (1.0f + rel) + abs_eps;
    // Jung bound geometry (seed-relative float coordinates). Grid centres are
    // within jr + jh/sqrt(2) < thr of the seed, so the same error bound as for
    // candidates applies when thr <= 2E; otherwise widen it.
    const double jr = threshold / std::sqrt(3.0);
    const double jh = 2.0 * jr / QT_JG;
    const double jrp = jr + jh / std::sqrt(2.0);
    const double Rj = std::max(R, jrp);
    const float eps_j = static_cast<float>(Rj * Rj * (1.0 / 65536.0));
    tp.jr = static_cast<float>(jr);
    tp.jh = static_cast<float>(jh);
    tp.jr2 = static_cast<float>(jrp * jrp) * (1.0f + rel) + 2.0f * eps_j;
    return tp;
}

struct GpuContext {
    int device = 0;
    int num_sms = 1;
    int grid = 1;
    Point* d_pts = nullptr;
    float2* d_ptsf = nullptr;
    float* d_mf = nullptr;
    ThrParams tp;
    unsigned char* d_clustered = nullptr;
    int* d_bounds = nullptr;
    int* d_seeds = nullptr;
    int* d_blk_best = nullptr;
    int* d_card = nullptr;
    int* d_members_rec = nullptr;
    int* d_members = nullptr;
    int* d_idx = nullptr;
    int* d_arg = nullptr;
    float2* d_xy = nullptr;
    int* d_counters = nullptr;   // [0] work counter, [1] best hint
    int scratch_stride = 0;
    int shared_cap = 0;
    int member_stride = 0;
};

static void setupGpuScratch(GpuContext& gpu, int max_bound);

// Main QT clustering algorithm (hybrid MPI + OpenMP + CUDA)
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  GpuContext& gpu, const int rank, const int nranks) {
    const int N = static_cast<int>(points.size());
    std::vector<unsigned char> clustered(N, 0);
    std::vector<int> bounds(N);
    std::vector<Cluster> clusters;

    // Exact cardinality upper bounds for every point.
    initBoundsKernel<<<(N + 255) / 256, 256>>>(gpu.d_pts, gpu.d_ptsf, N, gpu.tp, gpu.d_bounds);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(bounds.data(), gpu.d_bounds, N * sizeof(int), cudaMemcpyDeviceToHost));
    setupGpuScratch(gpu, *std::max_element(bounds.begin(), bounds.end()));

    std::vector<int> sorted;      // this rank's seeds to evaluate, bound desc, index asc
    std::vector<int> my_card;
    std::vector<int> blk_best;
    std::vector<int> hist;
    std::vector<unsigned char> eligible(N, 0);
    std::vector<int> best_members;
    int remaining = N;
    // Host threads for the O(N) bookkeeping: enough to matter, few enough
    // that fork/join overhead stays negligible.
    const int host_threads = std::max(1, std::min(omp_get_max_threads(), std::min(32, N / 8192)));

    // Memoized knowledge about the seeds owned by this rank (seed % nranks ==
    // rank). A seed's candidate cluster depends only on the unclustered points
    // within the threshold of it, whose count is bounds[seed]; as long as that
    // count is unchanged the knowledge below stays exact.
    //   kcard[s]  exact cardinality (0 = unknown)
    //   kub[s]    upper bound on the cardinality
    //   kbound[s] bounds[s] at the time the knowledge was recorded
    std::vector<int> kcard(N, 0), kub(N, INT_MAX), kbound(N, -1);

    while (remaining > 0) {
        // Largest bound among unclustered points.
        int max_bound = 0;
#pragma omp parallel for num_threads(host_threads) reduction(max : max_bound)
        for (int i = 0; i < N; ++i)
            if (!clustered[i] && bounds[i] > max_bound) max_bound = bounds[i];

        // Every remaining point is isolated: in the sequential algorithm each
        // of them becomes a singleton cluster in ascending index order.
        if (max_bound <= 1) {
            for (int i = 0; i < N; ++i) {
                if (clustered[i]) continue;
                Cluster c;
                c.seed_point = i;
                c.members.push_back(i);
                clusters.push_back(c);
                clustered[i] = 1;
            }
            remaining = 0;
            break;
        }

        // Refresh the memoized knowledge of this rank's seeds and collect the
        // best exactly known cardinality (a valid pruning hint).
        int known_card = 0, known_seed = INT_MAX;
#pragma omp parallel num_threads(host_threads)
        {
            int lc = 0, ls = INT_MAX;
#pragma omp for schedule(static)
            for (int s = rank; s < N; s += nranks) {
                if (clustered[s]) continue;
                if (kbound[s] != bounds[s]) {
                    kbound[s] = bounds[s];
                    kcard[s] = 0;
                    kub[s] = bounds[s];
                }
                if (kcard[s] > 0 && (kcard[s] > lc || (kcard[s] == lc && s < ls))) {
                    lc = kcard[s]; ls = s;
                }
            }
#pragma omp critical
            {
                if (lc > known_card || (lc == known_card && ls < known_seed)) {
                    known_card = lc; known_seed = ls;
                }
            }
        }
        int hint0 = 0;
        MPI_Allreduce(&known_card, &hint0, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

        // Seeds that still need evaluation: owned, unclustered, cardinality
        // unknown and upper bound not below the hint. The best known seed is
        // evaluated again so that its member list gets recorded on the GPU.
        int neval = 0;
#pragma omp parallel for num_threads(host_threads) reduction(+ : neval)
        for (int s = 0; s < N; ++s) {
            const bool e = (s % nranks == rank) && !clustered[s] &&
                           (kcard[s] == 0 || s == known_seed) && kub[s] >= hint0;
            eligible[s] = e ? 1 : 0;
            neval += e ? 1 : 0;
        }

        // Counting sort of the eligible seeds by bound (descending), ascending
        // index within equal bounds (deterministic). Parallel over contiguous
        // index ranges with per-thread histograms (OpenMP).
        sorted.resize(neval);
        if (neval > 0) {
            const int nb = max_bound + 1;        // buckets: key = max_bound - bound
            const int nt = host_threads;
            hist.assign(static_cast<size_t>(nt) * nb, 0);
#pragma omp parallel num_threads(nt)
            {
                const int t = omp_get_thread_num();
                const int lo = static_cast<int>(static_cast<long long>(N) * t / nt);
                const int hi = static_cast<int>(static_cast<long long>(N) * (t + 1) / nt);
                int* h = hist.data() + static_cast<size_t>(t) * nb;
                for (int i = lo; i < hi; ++i)
                    if (eligible[i]) h[max_bound - bounds[i]]++;
#pragma omp barrier
#pragma omp single
                {
                    // exclusive scan in (bucket, thread) order
                    int run = 0;
                    for (int b = 0; b < nb; ++b)
                        for (int k = 0; k < nt; ++k) {
                            int& v = hist[static_cast<size_t>(k) * nb + b];
                            const int c = v;
                            v = run;
                            run += c;
                        }
                }
                for (int i = lo; i < hi; ++i)
                    if (eligible[i]) sorted[h[max_bound - bounds[i]]++] = i;
            }
        }

        int local_card = 0, local_seed = INT_MAX;
        int local_block = -1;           // block holding the member list of the local best
        if (neval > 0) {
            CUDA_CHECK(cudaMemcpyAsync(gpu.d_seeds, sorted.data(), neval * sizeof(int),
                                       cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemsetAsync(gpu.d_counters, 0, 2 * sizeof(int)));
            CUDA_CHECK(cudaMemcpyAsync(gpu.d_counters + 1, &hint0, sizeof(int), cudaMemcpyHostToDevice));
            // Shared memory capacity adapted to the largest possible candidate
            // list of this iteration; occupancy follows from it.
            const int cap = std::min(QT_SHARED_MAX, std::max(QT_SHARED_MIN, max_bound));
            const size_t shmem = static_cast<size_t>(cap) * QT_CAND_BYTES;
            int blocks_per_sm = 1;
            CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &blocks_per_sm, seedCardinalityKernel, QT_BLOCK, shmem));
            const int grid = std::min(std::max(1, gpu.num_sms * std::max(1, blocks_per_sm)),
                                      std::min(gpu.grid, neval));
            seedCardinalityKernel<<<grid, QT_BLOCK, shmem>>>(
                gpu.d_pts, gpu.d_ptsf, N, gpu.d_clustered, gpu.d_seeds, neval, gpu.d_bounds, gpu.tp,
                gpu.d_idx, gpu.d_arg, gpu.d_mf, gpu.d_xy, gpu.scratch_stride, cap,
                gpu.d_members_rec, gpu.member_stride, gpu.d_blk_best, gpu.d_card,
                gpu.d_counters, gpu.d_counters + 1);
            CUDA_CHECK(cudaGetLastError());
            blk_best.resize(3 * grid);
            my_card.resize(neval);
            CUDA_CHECK(cudaMemcpyAsync(blk_best.data(), gpu.d_blk_best, 3 * grid * sizeof(int),
                                       cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(my_card.data(), gpu.d_card, neval * sizeof(int),
                                  cudaMemcpyDeviceToHost));

            // Record the new knowledge.
            for (int i = 0; i < neval; ++i) {
                const int sd = sorted[i];
                const int c = my_card[i];
                if (c > 0) { kcard[sd] = c; kub[sd] = c; }
                else kub[sd] = std::min(kub[sd], -c);
            }
            for (int b = 0; b < grid; ++b) {
                const int c = blk_best[3 * b];
                const int sd = blk_best[3 * b + 1];
                if (c > local_card || (c == local_card && sd < local_seed)) {
                    local_card = c; local_seed = sd; local_block = b;
                }
            }
        }
        long long local_best = -1;
        if (local_card > 0) local_best = static_cast<long long>(local_card) * (N + 1) + (N - local_seed);

        // Global winner: max cardinality, lowest seed index on ties.
        long long global_best = -1;
        MPI_Allreduce(&local_best, &global_best, 1, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
        const int best_card = static_cast<int>(global_best / (N + 1));
        const int best_seed = N - static_cast<int>(global_best % (N + 1));

        // The owner of the winning seed provides its member list, recorded by
        // the block that evaluated it (host re-growth is a safety fallback).
        const int owner = best_seed % nranks;
        best_members.resize(best_card);
        if (rank == owner) {
            if (local_block >= 0 && local_seed == best_seed) {
                const int b = local_block;
                const int buf = blk_best[3 * b + 2];
                CUDA_CHECK(cudaMemcpy(best_members.data(),
                                      gpu.d_members_rec + (static_cast<size_t>(b) * 2 + buf) * gpu.member_stride,
                                      best_card * sizeof(int), cudaMemcpyDeviceToHost));
            } else {
                growClusterHost(best_seed, clustered, points, gpu.tp, best_members);
                if (static_cast<int>(best_members.size()) != best_card) {
                    fprintf(stderr, "Internal error: host/device cardinality mismatch (%zu vs %d)\n",
                            best_members.size(), best_card);
                    MPI_Abort(MPI_COMM_WORLD, 2);
                }
            }
        }
        if (nranks > 1) MPI_Bcast(best_members.data(), best_card, MPI_INT, owner, MPI_COMM_WORLD);

        Cluster cluster;
        cluster.seed_point = best_seed;
        cluster.members = best_members;
        clusters.push_back(cluster);

        for (int m : best_members) clustered[m] = 1;
        remaining -= static_cast<int>(best_members.size());

        // Update device state: clustered flags and exact bounds.
        const int m = static_cast<int>(best_members.size());
        CUDA_CHECK(cudaMemcpyAsync(gpu.d_members, best_members.data(), m * sizeof(int),
                                   cudaMemcpyHostToDevice));
        updateBoundsKernel<<<(N + 255) / 256, 256>>>(gpu.d_pts, gpu.d_ptsf, N, gpu.d_clustered,
                                                     gpu.d_members, m, gpu.tp, gpu.d_bounds);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(gpu.d_clustered, clustered.data(), N, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(bounds.data(), gpu.d_bounds, N * sizeof(int), cudaMemcpyDeviceToHost));
    }

    return clusters;
}

// Validation: check that clusters satisfy the QT clustering properties
bool validateClusters(const std::vector<Cluster>& clusters,
                     const std::vector<Point>& points,
                     const double threshold) {
    bool valid = true;

    printf("Validating clusters:\n");

    // Diameters of all clusters (OpenMP over clusters, dynamic for imbalance)
    std::vector<double> diameters(clusters.size(), 0.0);
#pragma omp parallel for schedule(dynamic, 1)
    for (size_t c = 0; c < clusters.size(); ++c) {
        const auto& cluster = clusters[c];
        double max_diameter = 0.0;
        for (size_t i = 0; i < cluster.members.size(); ++i) {
            for (size_t j = i + 1; j < cluster.members.size(); ++j) {
                const double dist = distance(points[cluster.members[i]],
                                           points[cluster.members[j]]);
                max_diameter = std::max(max_diameter, dist);
            }
        }
        diameters[c] = max_diameter;
    }

    // Check each cluster
    for (size_t c = 0; c < clusters.size(); ++c) {
        const auto& cluster = clusters[c];
        const double max_diameter = diameters[c];

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
#pragma omp parallel for reduction(+ : clustered_count)
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

static void setupGpu(GpuContext& gpu, const std::vector<Point>& points, const double threshold) {
    const int N = static_cast<int>(points.size());
    gpu.tp = makeThrParams(threshold, points);
    CUDA_CHECK(cudaSetDevice(gpu.device));
    CUDA_CHECK(cudaDeviceGetAttribute(&gpu.num_sms, cudaDevAttrMultiProcessorCount, gpu.device));
    CUDA_CHECK(cudaFuncSetAttribute(seedCardinalityKernel,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    QT_SHARED_MAX * QT_CAND_BYTES));
    CUDA_CHECK(cudaFuncSetAttribute(seedCardinalityKernel,
                                    cudaFuncAttributePreferredSharedMemoryCarveout,
                                    cudaSharedmemCarveoutMaxShared));
    int blocks_per_sm = 1;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, seedCardinalityKernel, QT_BLOCK,
        QT_SHARED_MIN * QT_CAND_BYTES));
    gpu.grid = std::max(1, gpu.num_sms * std::max(1, blocks_per_sm));   // upper bound

    CUDA_CHECK(cudaMalloc(&gpu.d_pts, N * sizeof(Point)));
    CUDA_CHECK(cudaMalloc(&gpu.d_ptsf, N * sizeof(float2)));
    CUDA_CHECK(cudaMalloc(&gpu.d_clustered, N));
    CUDA_CHECK(cudaMalloc(&gpu.d_bounds, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&gpu.d_seeds, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&gpu.d_members, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&gpu.d_blk_best, 3 * gpu.grid * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&gpu.d_card, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&gpu.d_counters, 2 * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(gpu.d_pts, points.data(), N * sizeof(Point), cudaMemcpyHostToDevice));
    {
        std::vector<float2> ptsf(N);
        for (int i = 0; i < N; ++i)
            ptsf[i] = make_float2(static_cast<float>(points[i].x), static_cast<float>(points[i].y));
        CUDA_CHECK(cudaMemcpy(gpu.d_ptsf, ptsf.data(), N * sizeof(float2), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemset(gpu.d_clustered, 0, N));
}

// Per-block scratch sized by the largest neighbourhood: no seed ever has more
// candidates than max_bound - 1, and bounds only decrease.
static void setupGpuScratch(GpuContext& gpu, const int max_bound) {
    gpu.scratch_stride = std::max(32, ((max_bound + 31) / 32) * 32);
    gpu.member_stride = gpu.scratch_stride;
    const size_t per = static_cast<size_t>(gpu.grid) * 2 * gpu.scratch_stride;
    CUDA_CHECK(cudaMalloc(&gpu.d_idx, per * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&gpu.d_arg, per * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&gpu.d_mf, per * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gpu.d_xy, per * sizeof(float2)));
    CUDA_CHECK(cudaMalloc(&gpu.d_members_rec, per * sizeof(int)));
}

static void teardownGpu(GpuContext& gpu) {
    cudaFree(gpu.d_pts);
    cudaFree(gpu.d_clustered);
    cudaFree(gpu.d_bounds);
    cudaFree(gpu.d_seeds);
    cudaFree(gpu.d_blk_best);
    cudaFree(gpu.d_card);
    cudaFree(gpu.d_members_rec);
    cudaFree(gpu.d_members);
    cudaFree(gpu.d_idx);
    cudaFree(gpu.d_arg);
    cudaFree(gpu.d_xy);
    cudaFree(gpu.d_mf);
    cudaFree(gpu.d_ptsf);
    cudaFree(gpu.d_counters);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

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

    if (num_points <= 0 || threshold <= 0.0) {
        if (rank == 0)
            printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
                   num_points, threshold);
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // One GPU per rank, selected by the rank's position on its node.
    GpuContext gpu;
    {
        MPI_Comm node_comm;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node_comm);
        int node_rank = 0;
        MPI_Comm_rank(node_comm, &node_rank);
        MPI_Comm_free(&node_comm);
        int ndev = 0;
        CUDA_CHECK(cudaGetDeviceCount(&ndev));
        if (ndev <= 0) {
            fprintf(stderr, "No CUDA device available\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        gpu.device = node_rank % ndev;
    }

    // Generate synthetic data (deterministic, identical on every rank)
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);

    setupGpu(gpu, points, threshold);

    // Perform QT clustering
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_start = std::chrono::high_resolution_clock::now();

    const std::vector<Cluster> clusters = qtClustering(points, gpu, rank, nranks);

    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);

    teardownGpu(gpu);

    if (rank != 0) {
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
    int ret = 0;
    if (validate) {
        const bool valid = validateClusters(clusters, points, threshold);

        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            ret = 1;
        }
    }

    MPI_Finalize();
    return ret;
}
