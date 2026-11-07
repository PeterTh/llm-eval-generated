// QT Clustering Benchmark - Hybrid MPI + OpenMP + CUDA Version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// Parallelization strategy
// ------------------------
// The outer loop (one cluster extracted per iteration) is inherently
// sequential, but every iteration evaluates a candidate cluster for every
// still-unclustered seed point, and these candidate evaluations are fully
// independent.  They are the dominant cost.
//
//   * MPI     : the set of seeds is distributed cyclically over the ranks.
//               Each rank evaluates its seeds and the globally best seed is
//               found with a single MPI_MAXLOC all-reduce; the owning rank
//               broadcasts the member list of the winning cluster.
//   * CUDA    : on each rank, candidate clusters are grown on the GPU, one
//               warp per candidate cluster (dynamic work distribution with an
//               atomic counter).  The growth uses an exact lazy-evaluation
//               scheme with convex-hull distance bounds (see the kernel
//               comments) and branch and bound over the seeds: a seed is
//               skipped or its growth aborted as soon as a provable upper
//               bound on its cardinality cannot beat the best cluster found
//               so far on the device (with the original tie rule).
//   * OpenMP  : every rank drives all GPUs assigned to it with one OpenMP
//               thread per GPU (so a single rank per node can still use all
//               accelerators), and host-side work (statistics, validation)
//               is parallelized with OpenMP as well.
//
// The results are identical to the sequential version: among the seeds the
// one with the largest cardinality and, on ties, the smallest index wins; the
// greedy growth adds the candidate with the smallest resulting diameter and,
// on ties, the smallest index.  All floating-point comparisons of the
// sequential code are reproduced exactly (see "Exactness notes" below).

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <vector>

#include <cuda_runtime.h>
#define OMPI_SKIP_MPICXX 1
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
// Error handling helpers
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                         \
                    cudaGetErrorString(err__), __FILE__, __LINE__);             \
            MPI_Abort(MPI_COMM_WORLD, 1);                                       \
        }                                                                       \
    } while (0)

#define MPI_CHECK(call)                                                         \
    do {                                                                        \
        int err__ = (call);                                                     \
        if (err__ != MPI_SUCCESS) {                                             \
            fprintf(stderr, "MPI error %d at %s:%d\n", err__, __FILE__,         \
                    __LINE__);                                                  \
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
inline double distance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

// ---------------------------------------------------------------------------
// Exactness notes for the GPU implementation
// ---------------------------------------------------------------------------
// The sequential code compares distances d = sqrt(dx*dx + dy*dy).  The GPU
// works with the squared distance s = dx*dx + dy*dy (correctly rounded, no
// FMA contraction) instead and reconstructs the exact sequential semantics:
//   * max over members:  sqrt(max s) == max sqrt(s)   (sqrt is monotonic)
//   * d < threshold  <=>  s < S_crit, where S_crit is the smallest double
//     with sqrt(S_crit) >= threshold (computed once on the host).
//   * arg-min:  the sequential code picks the candidate with the smallest d
//     and, on ties, the smallest index.  Two *different* s values can round
//     to the same sqrt only if they are within a few ulps of each other.  The
//     fast path therefore does the arg-min on (s, index) and raises a flag if
//     any comparison involved two distinct values within NEAR_ULPS ulps; the
//     (practically never taken) slow path then redoes the arg-min on
//     (sqrt(s), index) exactly like the sequential code.
// All ordering comparisons of squared distances are done on the IEEE-754 bit
// patterns reinterpreted as signed 64-bit integers.  For non-negative doubles
// (including +inf, used as the "empty" sentinel) this order is identical to
// the floating-point order, and integer compares run at full rate while FP64
// compares run at 1/64 rate on consumer GPUs.
// Additionally an FP32 pre-filter (with a generous error margin) skips the
// expensive FP64 evaluation whenever the result provably cannot change a
// candidate's current maximum, or provably exceeds the threshold.
//
// Lazy evaluation (exact)
// -----------------------
// The diameter d_k of the greedy cluster after k additions is non-decreasing
// in k.  A candidate c whose max squared distance s_c to the current members
// exceeds a bound D can therefore not be chosen before the diameter reaches D.
// Each warp keeps
//   * an ACTIVE set:  candidates with a value v_c <= D that is updated with
//     every new member (arg-min + update at every growth step), and
//   * a DORMANT set:  candidates with a lower bound L_c > D on s_c.  They are
//     only touched when the arg-min over the active set gets within the
//     tolerance of D; then D is raised and dormant candidates with L_c <= D
//     are re-evaluated using the convex hull of the members.
// Convex hull bound: the maximum of the (convex) squared distance over the
// member set is attained at a vertex of the members' convex hull, so
//   hb_c = max over hull vertices of f(c, v)   (f = computed sqdist)
// satisfies hb_c <= s_c <= hb_c * (1 + rho) with rho ~ 2^-35, where the
// tiny excess comes from (i) rounding of f and (ii) the hull being built on
// coordinates truncated to a fixed-point grid of 2^-S (exact integer
// orientation predicates, so no vertex is ever dropped unless it provably
// lies inside the hull of the kept vertices).  Every value v_c used by the
// GPU is therefore a lower bound on s_c that is exact to within rho.  The
// fast path treats values within 2*TOL_BITS (~2^-25 relative, >> rho) as a
// potential tie; in that rare case the exact s_c of all active candidates is
// computed from the full member list and the arg-min is redone on
// (sqrt(s_c), index) exactly like the sequential code.  The same applies to
// the threshold test: values within TOL_BITS of S_crit are resolved exactly.
// Dormant candidates always satisfy L_c > D >= a + 2*TOL_BITS (a being the
// active arg-min), so they can be involved in neither kind of tie.

#ifndef QT_ACTIVE_TARGET
#define QT_ACTIVE_TARGET 64
#endif
#ifndef QT_MIN_BLOCKS
#define QT_MIN_BLOCKS 4
#endif
static const int QT_BLOCK = 128;               // 4 warps per block
static const int QT_WARPS = QT_BLOCK / 32;
static const int H_MAX = 48;                   // hull vertices kept per warp (shared)
static const int A_CAP = 64;                   // active candidates in shared memory per warp
static const int N_CAP = 160;                  // near-dormant candidates in shared memory per warp
static const int SLAB_BUCKETS = 2048;          // x-bucket table for candidate slabs
static const int GRID_G = 24;                  // occupancy grid (cells per axis) for the Jung bound
static const int ACTIVE_TARGET = QT_ACTIVE_TARGET;   // desired active candidates per warp
static const int FAR_TARGET = 3 * QT_ACTIVE_TARGET;  // desired near-dormant candidates per warp
// FP32 pre-filter safety margins (relative and absolute, in squared units).
// The FP32 squared distance error for coordinates in [0, 20] is < 1e-3
// and well below (1 - F32_REL) * s + F32_ABS for every relevant s.
static const float F32_REL = 0.999f;
static const float F32_ABS = 2e-3f;

typedef long long sbits_t;
#define SBITS_INF 0x7FF0000000000000LL
#define TOL_BITS (1LL << 26)                   // tolerance in bit-pattern units
#define NEAR_BITS (2 * TOL_BITS)
#define DELTA_MIN (1LL << 40)
#define DELTA_INIT (1LL << 49)
#define DELTA_MAX (1LL << 54)
#define FULL_MASK 0xffffffffu
#ifndef PRUNE_PERIOD
#define PRUNE_PERIOD 32                        // steps between dormant prune passes
#endif

// Smallest double s such that sqrt(s) >= threshold (sqrt monotonic + correctly rounded).
static double computeSquaredCritical(const double threshold) {
    double s = threshold * threshold;
    while (std::sqrt(s) >= threshold) s = std::nextafter(s, -std::numeric_limits<double>::infinity());
    while (std::sqrt(s) < threshold) s = std::nextafter(s, std::numeric_limits<double>::infinity());
    return s;
}

static sbits_t hostDoubleBits(double d) {
    sbits_t b;
    memcpy(&b, &d, sizeof(b));
    return b;
}

__device__ __forceinline__ double dev_sqdist(double x1, double y1, double x2, double y2) {
    const double dx = x1 - x2;
    const double dy = y1 - y2;
    return __dadd_rn(__dmul_rn(dx, dx), __dmul_rn(dy, dy));
}

__device__ __forceinline__ float dev_sqdist_f(float x1, float y1, float x2, float y2) {
    const float dx = x1 - x2;
    const float dy = y1 - y2;
    return fmaf(dx, dx, dy * dy);
}

// Float value <= the double with bit pattern sb (truncation, integer ops only).
__device__ __forceinline__ float bits_to_float_trunc(sbits_t sb) {
    const int e = static_cast<int>(sb >> 32) - 0x38000000;   // rebias exponent 1023 -> 127
    return (e <= 0) ? 0.0f : __int_as_float(e << 3);
}

// True if a != b and the two values are within NEAR_BITS, or the smaller one
// is tiny (where the relative bound of the hull estimate does not hold).
__device__ __forceinline__ bool dev_near(sbits_t a, sbits_t b, sbits_t tiny) {
    const sbits_t d = a - b;
    const sbits_t mn = (a < b) ? a : b;
    return ((d != 0) && (d <= NEAR_BITS) && (d >= -NEAR_BITS)) || (mn < tiny);
}

// Warp-wide arg-min over (value, index) pairs; ties resolved by smaller index.
// Butterfly pattern: every lane ends up with the result.  `flag` collects
// whether any pair of distinct near-equal values was compared.
__device__ __forceinline__ void warp_argmin(sbits_t& d, int& i, bool& flag, sbits_t tiny) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const sbits_t od = __shfl_xor_sync(FULL_MASK, d, off);
        const int oi = __shfl_xor_sync(FULL_MASK, i, off);
        const int of = __shfl_xor_sync(FULL_MASK, static_cast<int>(flag), off);
        flag = flag || (of != 0) || dev_near(od, d, tiny);
        if (od < d || (od == d && oi < i)) {
            d = od;
            i = oi;
        }
    }
}

__device__ __forceinline__ void warp_argmin_exact(sbits_t& d, int& i) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const sbits_t od = __shfl_xor_sync(FULL_MASK, d, off);
        const int oi = __shfl_xor_sync(FULL_MASK, i, off);
        if (od < d || (od == d && oi < i)) {
            d = od;
            i = oi;
        }
    }
}

__device__ __forceinline__ sbits_t warp_min_ll(sbits_t v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const sbits_t o = __shfl_xor_sync(FULL_MASK, v, off);
        v = (o < v) ? o : v;
    }
    return v;
}

// Track the running local (s, index) minimum plus the near-tie flag.
__device__ __forceinline__ void track_min(sbits_t s, int i, sbits_t& ls, int& li, bool& flag,
                                          sbits_t tiny) {
    if (s < ls) {
        flag = flag || dev_near(s, ls, tiny);
        ls = s;
        li = i;
    } else if (s == ls) {
        li = (i < li) ? i : li;
    } else {
        flag = flag || dev_near(s, ls, tiny);
    }
}

// Conservative slab [lo, hi) of the x-sorted candidate list covering all x in
// [xq - r, xq + r], via the bucket table (bucket granularity, superset).
__device__ __forceinline__ void dev_slab(const int* __restrict__ bstart, const float xmin,
                                         const float xscale, const int nb,
                                         const float xq, const float r, int& lo, int& hi) {
    int b0 = static_cast<int>(floorf((xq - r - xmin) * xscale));
    int b1 = static_cast<int>(floorf((xq + r - xmin) * xscale)) + 1;
    b0 = (b0 < 0) ? 0 : ((b0 > nb) ? nb : b0);
    b1 = (b1 < 0) ? 0 : ((b1 > nb) ? nb : b1);
    lo = bstart[b0];
    hi = bstart[b1];
}

// Exact orientation predicate on fixed-point coordinates (|coord| < 2^62):
// sign of (b - a) x (c - a), evaluated with 128-bit products.
__device__ __forceinline__ int orient_exact(long long ax, long long ay, long long bx, long long by,
                                            long long cx, long long cy) {
    const long long d1x = bx - ax, d1y = cy - ay, d2x = by - ay, d2y = cx - ax;
    const long long h1 = __mul64hi(d1x, d1y);
    const unsigned long long l1 = static_cast<unsigned long long>(d1x) * static_cast<unsigned long long>(d1y);
    const long long h2 = __mul64hi(d2x, d2y);
    const unsigned long long l2 = static_cast<unsigned long long>(d2x) * static_cast<unsigned long long>(d2y);
    if (h1 != h2) return (h1 > h2) ? 1 : -1;
    return (l1 > l2) ? 1 : ((l1 < l2) ? -1 : 0);
}

// Per-warp global scratch (warp-level lists, uniform counts).
struct WarpScratch {
    int* a_idx;      // active candidates (overflow beyond A_CAP)
    sbits_t* a_s;
    int* n_idx;      // near-dormant candidates (overflow beyond N_CAP)
    sbits_t* n_s;
    int* n_k0;
    int* n_w;        // witness member (point index) attaining the bound
    int* f_idx;      // far-dormant candidates (bound > D_far)
    sbits_t* f_s;
    int* f_k0;
    int* f_w;
    int* e_pos;      // evaluation list (positions in the near list)
    int* m_idx;      // members (point indices) of the cluster being grown
    float2* m_xy;    // members (float coordinates)
};

// Per-warp shared memory: convex hull of the members, active set and the
// near-dormant tier (hot data), histogram for the bound selection.
struct WarpShared {
    double hx[H_MAX], hy[H_MAX];
    long long hix[H_MAX], hiy[H_MAX];
    float hfx[H_MAX], hfy[H_MAX];
    int hid[H_MAX];                 // point index of each hull vertex
    int a_idx[A_CAP];
    sbits_t a_s[A_CAP];
    float a_fx[A_CAP], a_fy[A_CAP];
    int n_idx[N_CAP];
    sbits_t n_s[N_CAP];
    int n_k0[N_CAP];
    int n_w[N_CAP];
    int hist[32];
    unsigned grid[GRID_G * GRID_G / 2];   // packed 16-bit cell counts (members + remaining)
};

// Insert member point p into the hull (warp-collective).  `valid` becomes
// false if the hull overflows; then bounds fall back to the full member list.
__device__ void hull_insert(WarpShared& H, int& h, bool& valid, const int lane,
                            const double x, const double y, const long long ix, const long long iy,
                            const float fx, const float fy, const int pid) {
    if (!valid) return;
    if (h <= 2) {
        if (h == 0) {
            if (lane == 0) { H.hx[0] = x; H.hy[0] = y; H.hix[0] = ix; H.hiy[0] = iy; H.hfx[0] = fx; H.hfy[0] = fy; H.hid[0] = pid; }
            h = 1;
        } else if (h == 1) {
            if (ix == H.hix[0] && iy == H.hiy[0]) { __syncwarp(); return; }   // duplicate
            if (lane == 0) { H.hx[1] = x; H.hy[1] = y; H.hix[1] = ix; H.hiy[1] = iy; H.hfx[1] = fx; H.hfy[1] = fy; H.hid[1] = pid; }
            h = 2;
        } else {
            const int o = orient_exact(H.hix[0], H.hiy[0], H.hix[1], H.hiy[1], ix, iy);
            if (o > 0) {
                if (lane == 0) { H.hx[2] = x; H.hy[2] = y; H.hix[2] = ix; H.hiy[2] = iy; H.hfx[2] = fx; H.hfy[2] = fy; H.hid[2] = pid; }
                h = 3;
            } else if (o < 0) {
                if (lane == 0) {
                    H.hx[2] = H.hx[1]; H.hy[2] = H.hy[1]; H.hix[2] = H.hix[1]; H.hiy[2] = H.hiy[1]; H.hfx[2] = H.hfx[1]; H.hfy[2] = H.hfy[1]; H.hid[2] = H.hid[1];
                    H.hx[1] = x; H.hy[1] = y; H.hix[1] = ix; H.hiy[1] = iy; H.hfx[1] = fx; H.hfy[1] = fy; H.hid[1] = pid;
                }
                h = 3;
            } else {
                // Collinear: keep the two extreme points of the segment.
                const bool usex = (H.hix[0] != H.hix[1]);
                const long long a = usex ? H.hix[0] : H.hiy[0];
                const long long b = usex ? H.hix[1] : H.hiy[1];
                const long long p = usex ? ix : iy;
                int rep = -1;
                if (a < b) { if (p < a) rep = 0; else if (p > b) rep = 1; }
                else       { if (p > a) rep = 0; else if (p < b) rep = 1; }
                if (rep >= 0 && lane == 0) { H.hx[rep] = x; H.hy[rep] = y; H.hix[rep] = ix; H.hiy[rep] = iy; H.hfx[rep] = fx; H.hfy[rep] = fy; H.hid[rep] = pid; }
            }
        }
        __syncwarp();
        return;
    }

    // h >= 3: non-degenerate CCW polygon.  Edge i = (K[i], K[i+1 mod h]).
    unsigned long long mask = 0;
    for (int base = 0; base < h; base += 32) {
        const int i = base + lane;
        bool vis = false;
        if (i < h) {
            const int n = (i + 1 == h) ? 0 : i + 1;
            vis = orient_exact(H.hix[i], H.hiy[i], H.hix[n], H.hiy[n], ix, iy) < 0;
        }
        const unsigned m = __ballot_sync(FULL_MASK, vis);
        mask |= static_cast<unsigned long long>(m) << base;
    }
    if (mask == 0ull) return;   // inside or on the boundary
    const unsigned long long hmask = (h == 64) ? ~0ull : ((1ull << h) - 1ull);
    const unsigned long long prevm = ((mask << 1) | (mask >> (h - 1))) & hmask;
    const unsigned long long nextm = ((mask >> 1) | (mask << (h - 1))) & hmask;
    const unsigned long long sm = mask & ~prevm;
    const unsigned long long em = mask & ~nextm;
    if (sm == 0ull || em == 0ull) { valid = false; return; }   // cannot happen for a convex polygon
    const int start = __ffsll(static_cast<long long>(sm)) - 1;
    const int end = __ffsll(static_cast<long long>(em)) - 1;
    const int removed = (end - start + h) % h;      // vertices start+1 .. end
    const int newh = h - removed + 1;
    if (newh > H_MAX) { valid = false; return; }

    // New sequence: K[start], p, K[end+1], ..., K[start-1].
    double nx[2], ny[2];
    long long nix[2], niy[2];
    float nfx[2], nfy[2];
    int nid[2];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        const int j = lane + r * 32;
        if (j < newh) {
            if (j == 1) {
                nx[r] = x; ny[r] = y; nix[r] = ix; niy[r] = iy; nfx[r] = fx; nfy[r] = fy; nid[r] = pid;
            } else {
                const int src = (j == 0) ? start : ((end + j - 1) % h);
                nx[r] = H.hx[src]; ny[r] = H.hy[src]; nix[r] = H.hix[src]; niy[r] = H.hiy[src];
                nfx[r] = H.hfx[src]; nfy[r] = H.hfy[src]; nid[r] = H.hid[src];
            }
        }
    }
    __syncwarp();
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        const int j = lane + r * 32;
        if (j < newh) {
            H.hx[j] = nx[r]; H.hy[j] = ny[r]; H.hix[j] = nix[r]; H.hiy[j] = niy[r];
            H.hfx[j] = nfx[r]; H.hfy[j] = nfy[r]; H.hid[j] = nid[r];
        }
    }
    __syncwarp();
    h = newh;
}

// Lower bound (within rho) on s_c from the hull: max over hull vertices.
__device__ __forceinline__ sbits_t hull_bound(const WarpShared& H, const int h,
                                              const double cx, const double cy,
                                              const float cfx, const float cfy, sbits_t sb,
                                              int& witness) {
    float mf = -INFINITY;
#pragma unroll 4
    for (int i = 0; i < h; ++i) {
        const float f = dev_sqdist_f(H.hfx[i], H.hfy[i], cfx, cfy);
        mf = (f > mf) ? f : mf;
    }
    const float thr = fmaf(mf, F32_REL, -F32_ABS);
    for (int i = 0; i < h; ++i) {
        if (dev_sqdist_f(H.hfx[i], H.hfy[i], cfx, cfy) >= thr) {
            const sbits_t sm = __double_as_longlong(dev_sqdist(H.hx[i], H.hy[i], cx, cy));
            if (sm > sb) {
                sb = sm;
                witness = H.hid[i];
            }
        }
    }
    return sb;
}

// True if candidate c certainly cannot be added any more: some hull vertex
// (hence some member) is at squared distance >= s_crit.  FP32 scan with the
// conservative margin, confirmed in FP64 (exact).
__device__ __forceinline__ bool hull_excludes(const WarpShared& H, const int h,
                                              const double cx, const double cy,
                                              const float cfx, const float cfy,
                                              const float s_crit_hi_f, const sbits_t s_crit_bits) {
    for (int i = 0; i < h; ++i) {
        if (dev_sqdist_f(H.hfx[i], H.hfy[i], cfx, cfy) > s_crit_hi_f) {
            const sbits_t sm = __double_as_longlong(dev_sqdist(H.hx[i], H.hy[i], cx, cy));
            if (!(sm < s_crit_bits)) return true;
        }
    }
    return false;
}

// Exact s_c = max over all k members (starting from the lower bound sb).
__device__ __forceinline__ sbits_t exact_s(const double* __restrict__ px, const double* __restrict__ py,
                                           const int* __restrict__ m_idx, const float2* __restrict__ m_xy,
                                           const int k, const double cx, const double cy,
                                           const float cfx, const float cfy, sbits_t sb) {
    float sf = bits_to_float_trunc(sb);
    for (int m = 0; m < k; ++m) {
        const float2 mc = m_xy[m];
        if (dev_sqdist_f(mc.x, mc.y, cfx, cfy) >= fmaf(sf, F32_REL, -F32_ABS)) {
            const int mi = m_idx[m];
            const sbits_t sm = __double_as_longlong(dev_sqdist(px[mi], py[mi], cx, cy));
            if (sm > sb) {
                sb = sm;
                sf = bits_to_float_trunc(sb);
            }
        }
    }
    return sb;
}

// Warp-level inclusive prefix sum (int).
__device__ __forceinline__ int warp_scan_int(int v, const int lane) {
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
        const int o = __shfl_up_sync(FULL_MASK, v, off);
        if (lane >= off) v += o;
    }
    return v;
}

// Pick the bucket boundary from the 32-bucket histogram so that the
// cumulative count stays <= target (at least bucket 0).  Returns the new
// bound and the band total.
__device__ __forceinline__ sbits_t pick_bound(WarpShared& S, const int lane, const sbits_t lo,
                                              const sbits_t width, const sbits_t hi_cap,
                                              const int target, int& total) {
    __syncwarp();
    const int cum = warp_scan_int(S.hist[lane], lane);
    total = __shfl_sync(FULL_MASK, cum, 31);
    const unsigned ok = __ballot_sync(FULL_MASK, cum <= target);
    const int B = (ok == 0u) ? 0 : (31 - __clz(ok));
    sbits_t bound = (B == 31) ? hi_cap : (lo + ((static_cast<sbits_t>(B + 1) * width) >> 5));
    if (bound > hi_cap) bound = hi_cap;
    __syncwarp();
    return bound;
}

// Bucket of v in (lo, lo + width]; width >= 2^40 so the high 32 bits suffice
// (bucket boundaries only steer the batch size, not correctness).
__device__ __forceinline__ void hist_add(WarpShared& S, const sbits_t v, const sbits_t lo,
                                         const float inv_width_hi) {
    const float f = static_cast<float>(static_cast<unsigned>((v - lo) >> 32)) * inv_width_hi;
    int b = static_cast<int>(f);
    b = (b > 31) ? 31 : ((b < 0) ? 0 : b);
    atomicAdd(&S.hist[b], 1);
}
__device__ __forceinline__ float hist_scale(const sbits_t width) {
    return 32.0f / static_cast<float>(static_cast<unsigned>(width >> 32));
}

// ---- occupancy grid over the seed's square [sx - r, sx + r]^2 ----------
// Cell counts of members + remaining candidates.  By Jung's theorem a set of
// diameter < threshold lies in a disc of radius threshold/sqrt(3), hence in a
// window of grid_w x grid_w cells: the max window count bounds the final
// cardinality of the cluster.
__device__ __forceinline__ int grid_cell(const float fx, const float fy, const float gx0,
                                         const float gy0, const float ginv) {
    int cx = static_cast<int>((fx - gx0) * ginv);
    int cy = static_cast<int>((fy - gy0) * ginv);
    cx = (cx < 0) ? 0 : ((cx >= GRID_G) ? GRID_G - 1 : cx);
    cy = (cy < 0) ? 0 : ((cy >= GRID_G) ? GRID_G - 1 : cy);
    return cy * GRID_G + cx;
}
__device__ __forceinline__ void grid_add(WarpShared& S, const int cell, const int delta) {
    const unsigned v = static_cast<unsigned>(delta) << ((cell & 1) << 4);
    atomicAdd(&S.grid[cell >> 1], v);
}
__device__ __forceinline__ int grid_get(const WarpShared& S, const int cell) {
    return static_cast<int>((S.grid[cell >> 1] >> ((cell & 1) << 4)) & 0xffffu);
}
// Max window count (warp-collective): lane r holds row r; column prefix sums
// via warp scans, row-band differences via shuffles, all in registers.
__device__ int grid_window_max(const WarpShared& S, const int W, const int lane) {
    int P[GRID_G];
    const bool has_row = lane < GRID_G;
    // inclusive prefix along the row (registers)
    {
        int acc = 0;
#pragma unroll
        for (int c = 0; c < GRID_G; ++c) {
            acc += has_row ? grid_get(S, lane * GRID_G + c) : 0;
            P[c] = acc;
        }
    }
    // inclusive prefix down the columns (scan over lanes = rows)
#pragma unroll
    for (int c = 0; c < GRID_G; ++c) {
        int v = P[c];
#pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
            const int o = __shfl_up_sync(FULL_MASK, v, off);
            if (lane >= off) v += o;
        }
        P[c] = v;
    }
    // band sums: rows (lane - W + 1 .. lane) -> B[c] = P[lane][c] - P[lane - W][c]
    int best = 0;
    if (W <= GRID_G) {
        int B[GRID_G];
#pragma unroll
        for (int c = 0; c < GRID_G; ++c) {
            const int up = __shfl_up_sync(FULL_MASK, P[c], W);
            B[c] = P[c] - ((lane >= W) ? up : 0);
        }
        if (has_row && lane >= W - 1) {
#pragma unroll
            for (int c0 = 0; c0 + W - 1 < GRID_G; ++c0) {
                const int s = B[c0 + W - 1] - ((c0 > 0) ? B[c0 - 1] : 0);
                best = (s > best) ? s : best;
            }
        }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const int o = __shfl_xor_sync(FULL_MASK, best, off);
        best = (o > best) ? o : best;
    }
    return best;
}

// Branch-and-bound key: higher cardinality wins, then the smaller seed index.
__host__ __device__ __forceinline__ unsigned long long bb_key(const int card, const int seed) {
    return (static_cast<unsigned long long>(card) << 32) |
           static_cast<unsigned>(INT_MAX - seed);
}

// Upper bound on the cardinality of each seed's cluster: the number of
// unclustered points within the threshold (FP32 test, conservative).
__global__ void __launch_bounds__(QT_BLOCK)
qt_count_kernel(const float* __restrict__ pxf, const float* __restrict__ pyf,
                const float* __restrict__ ux_f, const float* __restrict__ ux_fy,
                const int* __restrict__ bstart, const float xmin, const float xscale,
                const int* __restrict__ seeds, const int nseeds,
                const float s_crit_hi_f, const float thr_slab,
                int* __restrict__ counts) {
    const int lane = threadIdx.x & 31;
    const int w = (blockIdx.x * QT_BLOCK + threadIdx.x) >> 5;
    if (w >= nseeds) return;
    const int seed = seeds[w];
    const float sxf = pxf[seed];
    const float syf = pyf[seed];
    int lo, hi;
    dev_slab(bstart, xmin, xscale, SLAB_BUCKETS, sxf, thr_slab, lo, hi);
    int cnt = 0;
#pragma unroll 4
    for (int j = lo + lane; j < hi; j += 32) {
        if (!(dev_sqdist_f(sxf, syf, ux_f[j], ux_fy[j]) > s_crit_hi_f)) ++cnt;
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) cnt += __shfl_xor_sync(FULL_MASK, cnt, off);
    if (lane == 0) counts[w] = cnt;   // includes the seed itself
}

__global__ void __launch_bounds__(QT_BLOCK, QT_MIN_BLOCKS)
qt_candidate_kernel(const double* __restrict__ px,
                    const double* __restrict__ py,
                    const float* __restrict__ pxf,
                    const float* __restrict__ pyf,
                    const long long* __restrict__ pix,     // fixed-point coordinates
                    const long long* __restrict__ piy,
                    const int* __restrict__ ux_idx,        // unclustered indices sorted by x
                    const float* __restrict__ ux_f,        // their float x coordinates (ascending)
                    const float* __restrict__ ux_fy,       // their float y coordinates
                    const int* __restrict__ bstart,        // slab bucket table
                    const float xmin, const float xscale,
                    const int U,
                    const int* __restrict__ seeds,         // seeds to evaluate (bound-descending)
                    const int* __restrict__ ub_counts,     // upper bound per seed (or null)
                    const int nseeds,
                    const double s_crit,                   // keep iff s < s_crit
                    const sbits_t tiny_bits,               // values below this are resolved exactly
                    const float thr_slab,                  // threshold + float margin
                    const int grid_w,                      // Jung window size in cells
                    int* __restrict__ work_counter,        // dynamic seed scheduling
                    unsigned long long* __restrict__ best_key,   // branch-and-bound state
                    int* __restrict__ best_lock,           // serialises best-member capture
                    int* __restrict__ best_members,        // members of the best cluster (device)
                    int* __restrict__ cardinality_out,     // per seed
                    WarpScratch ws,
                    const int cap_warp,                    // scratch entries per warp
                    const int cap_members)                 // member slots per warp
{
    __shared__ WarpShared s_warp[QT_WARPS];

    const int lane = threadIdx.x & 31;
    const int wib = threadIdx.x >> 5;
    const int gw = blockIdx.x * QT_WARPS + wib;
    WarpShared& S = s_warp[wib];
    const size_t base = static_cast<size_t>(gw) * cap_warp;
    int* __restrict__ ga_idx = ws.a_idx + base;
    sbits_t* __restrict__ ga_s = ws.a_s + base;
    int* __restrict__ gn_idx = ws.n_idx + base;
    sbits_t* __restrict__ gn_s = ws.n_s + base;
    int* __restrict__ gn_k0 = ws.n_k0 + base;
    int* __restrict__ gn_w = ws.n_w + base;
    int* __restrict__ f_idx = ws.f_idx + base;
    sbits_t* __restrict__ f_s = ws.f_s + base;
    int* __restrict__ f_k0 = ws.f_k0 + base;
    int* __restrict__ f_w = ws.f_w + base;
    int* __restrict__ e_pos = ws.e_pos + base;
    int* __restrict__ m_idx = ws.m_idx + static_cast<size_t>(gw) * cap_members;
    float2* __restrict__ m_xy = ws.m_xy + static_cast<size_t>(gw) * cap_members;
    const unsigned lanemask_lt = (1u << lane) - 1u;

    const sbits_t s_crit_bits = __double_as_longlong(s_crit);
    // FP32 bound above which a squared distance certainly exceeds s_crit.
    const float s_crit_hi_f = fmaf(static_cast<float>(s_crit), 1.0f + (1.0f - F32_REL), F32_ABS);

    // ---- list accessors (shared part first, global overflow after) -------
#define A_IDX(e) ((e) < A_CAP ? S.a_idx[e] : ga_idx[(e) - A_CAP])
#define A_S(e)   ((e) < A_CAP ? S.a_s[e] : ga_s[(e) - A_CAP])
#define N_IDX(e) ((e) < N_CAP ? S.n_idx[e] : gn_idx[(e) - N_CAP])
#define N_S(e)   ((e) < N_CAP ? S.n_s[e] : gn_s[(e) - N_CAP])
#define N_K0(e)  ((e) < N_CAP ? S.n_k0[e] : gn_k0[(e) - N_CAP])
#define N_W(e)   ((e) < N_CAP ? S.n_w[e] : gn_w[(e) - N_CAP])
#define A_PUT(e, c, v, fx, fy) do { if ((e) < A_CAP) { S.a_idx[e] = (c); S.a_s[e] = (v); S.a_fx[e] = (fx); S.a_fy[e] = (fy); } \
                                    else { ga_idx[(e) - A_CAP] = (c); ga_s[(e) - A_CAP] = (v); } } while (0)
#define N_PUT(e, c, v, k0, w) do { if ((e) < N_CAP) { S.n_idx[e] = (c); S.n_s[e] = (v); S.n_k0[e] = (k0); S.n_w[e] = (w); } \
                                   else { gn_idx[(e) - N_CAP] = (c); gn_s[(e) - N_CAP] = (v); gn_k0[(e) - N_CAP] = (k0); gn_w[(e) - N_CAP] = (w); } } while (0)

    for (;;) {
        int task = 0;
        if (lane == 0) task = atomicAdd(work_counter, 1);
        task = __shfl_sync(FULL_MASK, task, 0);
        if (task >= nseeds) break;

        const int seed = seeds[task];
        // Branch and bound: skip seeds that cannot beat the best cluster found
        // so far on this device (larger cardinality, or equal and smaller index).
        if (ub_counts != nullptr) {
            const unsigned long long cur = *reinterpret_cast<volatile unsigned long long*>(best_key);
            if (bb_key(ub_counts[task], seed) <= cur) {
                if (lane == 0) cardinality_out[task] = 0;
                continue;
            }
        }
        const double sx = px[seed];
        const double sy = py[seed];
        const float sxf = pxf[seed];
        const float syf = pyf[seed];

        // Candidates: unclustered points in the x-slab around the seed with
        // dist(seed, c) < threshold (points farther away can never be added
        // since the diameter only grows).  All start far-dormant with the
        // exact distance to the seed as bound.
        int lo, hi;
        dev_slab(bstart, xmin, xscale, SLAB_BUCKETS, sxf, thr_slab, lo, hi);
        const float gx0 = sxf - thr_slab;
        const float gy0 = syf - thr_slab;
        const float ginv = static_cast<float>(GRID_G) / (2.0f * thr_slab);
        for (int i = lane; i < GRID_G * GRID_G / 2; i += 32) S.grid[i] = 0u;
        __syncwarp();
        if (lane == 0) grid_add(S, grid_cell(sxf, syf, gx0, gy0, ginv), 1);
        int cnt_a = 0, cnt_n = 0, cnt_f = 0;    // warp-uniform list sizes
        sbits_t fmin = SBITS_INF;               // min bound in the far tier (uniform)
        sbits_t nmin = SBITS_INF;               // min bound in the near tier (uniform)
        for (int j = lo; j < hi; j += 32) {
            const int jj = j + lane;
            bool keep = false;
            int c = 0;
            sbits_t sb = SBITS_INF;
            float fx = 0.f, fy = 0.f;
            if (jj < hi) {
                fx = ux_f[jj];
                fy = ux_fy[jj];
                if (!(dev_sqdist_f(sxf, syf, fx, fy) > s_crit_hi_f)) {
                    c = ux_idx[jj];
                    if (c != seed) {
                        sb = __double_as_longlong(dev_sqdist(sx, sy, px[c], py[c]));
                        keep = sb < s_crit_bits;
                    }
                }
            }
            const unsigned m = __ballot_sync(FULL_MASK, keep);
            if (keep) {
                const int pos = cnt_f + __popc(m & lanemask_lt);
                f_idx[pos] = c;
                f_s[pos] = sb;
                f_k0[pos] = 1;     // exact w.r.t. the seed (member 0)
                f_w[pos] = seed;
                grid_add(S, grid_cell(fx, fy, gx0, gy0, ginv), 1);
            }
            cnt_f += __popc(m);
            fmin = (keep && sb < fmin) ? sb : fmin;
        }
        fmin = warp_min_ll(fmin);
        __syncwarp();
        {
            int ub = cnt_f + 1;
            const int ubg = grid_window_max(S, grid_w, lane);
            ub = (ubg < ub) ? ubg : ub;
            if (bb_key(ub, seed) <= *reinterpret_cast<volatile unsigned long long*>(best_key)) {
                if (lane == 0) cardinality_out[task] = 0;
                continue;
            }
        }

        int k = 0;                     // number of members
        bool aborted = false;
        int last_prune = 0;
        int h = 0;                     // hull size
        bool hull_valid = true;
        sbits_t D = -1;                // active set == candidates with value <= D
        sbits_t D_far = -1;            // far-dormant candidates have bound > D_far
        sbits_t delta = DELTA_INIT;
        sbits_t delta_far = DELTA_INIT << 2;
        sbits_t ls = SBITS_INF;        // running local arg-min over active candidates
        int li = INT_MAX;
        bool flag = false;
        int best = seed;               // member to add

        for (;;) {
            // ---- add member `best` ----------------------------------------
            const double mx = px[best];
            const double my = py[best];
            const float mxf = pxf[best];
            const float myf = pyf[best];
            const long long mix = pix[best];
            const long long miy = piy[best];
            if (lane == 0) {
                m_idx[k] = best;
                m_xy[k] = make_float2(mxf, myf);
            }
            ++k;
            __syncwarp();   // member list visible to all lanes

            // ---- update the active candidates with the new member --------
            if (k > 1) {
                ls = SBITS_INF;
                li = INT_MAX;
                flag = false;
                int w = 0;
                sbits_t l_nmin = SBITS_INF, l_fmin = SBITS_INF;   // per-lane, reduced after the loop
                for (int b0 = 0; b0 < cnt_a; b0 += 32) {
                    const int e = b0 + lane;
                    const bool have = e < cnt_a;
                    int c = INT_MAX;
                    sbits_t sb = SBITS_INF;
                    float cfx = 0.f, cfy = 0.f;
                    double cx = 0.0, cy = 0.0;
                    if (have) {
                        c = A_IDX(e);
                        sb = A_S(e);
                        if (e < A_CAP) { cfx = S.a_fx[e]; cfy = S.a_fy[e]; }
                        else { cfx = pxf[c]; cfy = pyf[c]; }
                        cx = px[c];   // issued early so the latency overlaps the FP32 test
                        cy = py[c];
                    }
                    __syncwarp();   // reads complete before compaction writes
                    bool keep = have && (c != best);
                    bool demote = false;
                    if (keep && dev_sqdist_f(mxf, myf, cfx, cfy) >= fmaf(bits_to_float_trunc(sb), F32_REL, -F32_ABS)) {
                        const sbits_t sm = __double_as_longlong(dev_sqdist(mx, my, cx, cy));
                        if (sm > sb) {
                            sb = sm;
                            if (!(sb < s_crit_bits)) {
                                keep = false;
                            } else if (sb + TOL_BITS >= s_crit_bits) {   // too close to call: exact
                                sb = exact_s(px, py, m_idx, m_xy, k, cx, cy, cfx, cfy, sb);
                                keep = sb < s_crit_bits;
                            }
                            if (!keep) grid_add(S, grid_cell(cfx, cfy, gx0, gy0, ginv), -1);
                            if (keep && sb > D) { keep = false; demote = true; }
                        }
                    }
                    // compact the survivors
                    unsigned m = __ballot_sync(FULL_MASK, keep);
                    if (keep) {
                        const int pos = w + __popc(m & lanemask_lt);
                        A_PUT(pos, c, sb, cfx, cfy);
                        track_min(sb, c, ls, li, flag, tiny_bits);
                    }
                    w += __popc(m);
                    // demote to the near or far tier
                    const bool dn = demote && (sb <= D_far);
                    m = __ballot_sync(FULL_MASK, dn);
                    if (dn) {
                        const int pos = cnt_n + __popc(m & lanemask_lt);
                        N_PUT(pos, c, sb, k, best);
                    }
                    cnt_n += __popc(m);
                    const bool df = demote && !dn;
                    m = __ballot_sync(FULL_MASK, df);
                    if (df) {
                        const int pos = cnt_f + __popc(m & lanemask_lt);
                        f_idx[pos] = c;
                        f_s[pos] = sb;
                        f_k0[pos] = k;
                        f_w[pos] = best;
                    }
                    cnt_f += __popc(m);
                    if (dn) l_nmin = (sb < l_nmin) ? sb : l_nmin;
                    if (df) l_fmin = (sb < l_fmin) ? sb : l_fmin;
                }
                cnt_a = w;
                {
                    const sbits_t nm = warp_min_ll(l_nmin);
                    nmin = (nm < nmin) ? nm : nmin;
                    const sbits_t fm = warp_min_ll(l_fmin);
                    fmin = (fm < fmin) ? fm : fmin;
                }
                __syncwarp();
            }
            // The hull must contain every member before bounds are evaluated.
            hull_insert(S, h, hull_valid, lane, mx, my, mix, miy, mxf, myf, best);
            __syncwarp();

            // ---- choose the next member -----------------------------------
            sbits_t a = ls;
            best = li;
            bool near = flag;
            warp_argmin(a, best, near, tiny_bits);

            const bool need = (a == SBITS_INF) ? ((cnt_n | cnt_f) > 0) : (a + NEAR_BITS > D);
            if (need) {
                // Periodically drop dormant candidates that provably cannot
                // join any more, so that the branch-and-bound test below is
                // tight (the lazy bounds alone would not notice them).
                if (k - last_prune >= PRUNE_PERIOD && hull_valid) {
                    last_prune = k;
                    // Hull bounding box (float): a candidate whose max distance
                    // to the 4 corners is certainly below the threshold cannot
                    // be excluded, skipping the per-vertex scan.
                    float bx0 = INFINITY, bx1 = -INFINITY, by0 = INFINITY, by1 = -INFINITY;
                    for (int i = lane; i < h; i += 32) {
                        const float vx = S.hfx[i], vy = S.hfy[i];
                        bx0 = (vx < bx0) ? vx : bx0; bx1 = (vx > bx1) ? vx : bx1;
                        by0 = (vy < by0) ? vy : by0; by1 = (vy > by1) ? vy : by1;
                    }
#pragma unroll
                    for (int off = 16; off > 0; off >>= 1) {
                        const float o0 = __shfl_xor_sync(FULL_MASK, bx0, off); bx0 = (o0 < bx0) ? o0 : bx0;
                        const float o1 = __shfl_xor_sync(FULL_MASK, bx1, off); bx1 = (o1 > bx1) ? o1 : bx1;
                        const float o2 = __shfl_xor_sync(FULL_MASK, by0, off); by0 = (o2 < by0) ? o2 : by0;
                        const float o3 = __shfl_xor_sync(FULL_MASK, by1, off); by1 = (o3 > by1) ? o3 : by1;
                    }
                    // certainly-below bound in FP32 (mirror of s_crit_hi_f)
                    const float s_crit_lo_f = fmaf(static_cast<float>(s_crit), F32_REL, -F32_ABS);
                    sbits_t new_nmin = SBITS_INF;
                    int wn = 0;
                    for (int b0 = 0; b0 < cnt_n; b0 += 32) {
                        const int e = b0 + lane;
                        const bool have = e < cnt_n;
                        int c = 0, k0 = 0, wit = 0;
                        sbits_t sb = SBITS_INF;
                        if (have) { c = N_IDX(e); sb = N_S(e); k0 = N_K0(e); wit = N_W(e); }
                        __syncwarp();
                        float cfx = 0.f, cfy = 0.f;
                        if (have) { cfx = pxf[c]; cfy = pyf[c]; }
                        bool st = have;
                        if (have) {
                            const float ddx = fmaxf(fabsf(cfx - bx0), fabsf(cfx - bx1));
                            const float ddy = fmaxf(fabsf(cfy - by0), fabsf(cfy - by1));
                            if (fmaf(ddx, ddx, ddy * ddy) >= s_crit_lo_f) {
                                st = !hull_excludes(S, h, px[c], py[c], cfx, cfy, s_crit_hi_f, s_crit_bits);
                            }
                        }
                        const unsigned m = __ballot_sync(FULL_MASK, st);
                        if (st) {
                            const int pos = wn + __popc(m & lanemask_lt);
                            N_PUT(pos, c, sb, k0, wit);
                            new_nmin = (sb < new_nmin) ? sb : new_nmin;
                        } else if (have) {
                            grid_add(S, grid_cell(cfx, cfy, gx0, gy0, ginv), -1);
                        }
                        wn += __popc(m);
                    }
                    cnt_n = wn;
                    nmin = warp_min_ll(new_nmin);
                    sbits_t new_fmin = SBITS_INF;
                    int wf = 0;
                    for (int b0 = 0; b0 < cnt_f; b0 += 32) {
                        const int e = b0 + lane;
                        const bool have = e < cnt_f;
                        int c = 0, k0 = 0, wit = 0;
                        sbits_t sb = SBITS_INF;
                        if (have) { c = f_idx[e]; sb = f_s[e]; k0 = f_k0[e]; wit = f_w[e]; }
                        __syncwarp();
                        float cfx = 0.f, cfy = 0.f;
                        if (have) { cfx = pxf[c]; cfy = pyf[c]; }
                        bool st = have;
                        if (have) {
                            const float ddx = fmaxf(fabsf(cfx - bx0), fabsf(cfx - bx1));
                            const float ddy = fmaxf(fabsf(cfy - by0), fabsf(cfy - by1));
                            if (fmaf(ddx, ddx, ddy * ddy) >= s_crit_lo_f) {
                                st = !hull_excludes(S, h, px[c], py[c], cfx, cfy, s_crit_hi_f, s_crit_bits);
                            }
                        }
                        const unsigned m = __ballot_sync(FULL_MASK, st);
                        if (st) {
                            const int pos = wf + __popc(m & lanemask_lt);
                            f_idx[pos] = c;
                            f_s[pos] = sb;
                            f_k0[pos] = k0;
                            f_w[pos] = wit;
                            new_fmin = (sb < new_fmin) ? sb : new_fmin;
                        } else if (have) {
                            grid_add(S, grid_cell(cfx, cfy, gx0, gy0, ginv), -1);
                        }
                        wf += __popc(m);
                    }
                    cnt_f = wf;
                    fmin = warp_min_ll(new_fmin);
                    __syncwarp();
                    // Jung window bound on the final cardinality.
                    const int ubg = grid_window_max(S, grid_w, lane);
                    if (bb_key(ubg, seed) <= *reinterpret_cast<volatile unsigned long long*>(best_key)) {
                        aborted = true;
                        break;
                    }
                }
                // Branch and bound: members + remaining candidates is an upper
                // bound on the final cardinality.
                if (bb_key(k + cnt_a + cnt_n + cnt_f, seed) <=
                    *reinterpret_cast<volatile unsigned long long*>(best_key)) {
                    aborted = true;
                    break;
                }
                for (;;) {
                    if ((cnt_n | cnt_f) == 0) break;
                    // Trial band (D, Dn].
                    sbits_t Dn = D + delta;
                    const sbits_t lmin_w = (nmin < fmin) ? nmin : fmin;
                    Dn = (lmin_w > Dn) ? lmin_w : Dn;
                    if (a != SBITS_INF && a + NEAR_BITS > Dn) Dn = a + NEAR_BITS;

                    // Refill the near tier from the far tier if the band
                    // reaches beyond D_far (choose the new D_far so that
                    // about FAR_TARGET candidates move).
                    if (Dn > D_far && cnt_f > 0) {
                        sbits_t Fn = D_far + delta_far;
                        Fn = (fmin > Fn) ? fmin : Fn;
                        if (Dn > Fn) Fn = Dn;
                        const sbits_t fwidth = Fn - D_far;
                        const float finv = hist_scale(fwidth);
                        S.hist[lane] = 0;
                        __syncwarp();
                        for (int e = lane; e < cnt_f; e += 32) {
                            const sbits_t sb = f_s[e];
                            if (sb <= Fn) hist_add(S, sb, D_far, finv);
                        }
                        int total = 0;
                        sbits_t Fnew = pick_bound(S, lane, D_far, fwidth, Fn, FAR_TARGET, total);
                        if (Fnew < Dn) Fnew = Dn;
                        // Move far entries with bound <= Fnew to the near tier.
                        sbits_t new_fmin = SBITS_INF, l_nmin = SBITS_INF;
                        int wf = 0;
                        for (int b0 = 0; b0 < cnt_f; b0 += 32) {
                            const int e = b0 + lane;
                            const bool have = e < cnt_f;
                            int c = 0, k0 = 0, wit = 0;
                            sbits_t sb = SBITS_INF;
                            if (have) { c = f_idx[e]; sb = f_s[e]; k0 = f_k0[e]; wit = f_w[e]; }
                            __syncwarp();
                            const bool mv = have && (sb <= Fnew);
                            unsigned m = __ballot_sync(FULL_MASK, mv);
                            if (mv) {
                                const int pos = cnt_n + __popc(m & lanemask_lt);
                                N_PUT(pos, c, sb, k0, wit);
                            }
                            cnt_n += __popc(m);
                            if (mv) l_nmin = (sb < l_nmin) ? sb : l_nmin;
                            const bool st = have && !mv;
                            m = __ballot_sync(FULL_MASK, st);
                            if (st) {
                                const int pos = wf + __popc(m & lanemask_lt);
                                f_idx[pos] = c;
                                f_s[pos] = sb;
                                f_k0[pos] = k0;
                                f_w[pos] = wit;
                                new_fmin = (sb < new_fmin) ? sb : new_fmin;
                            }
                            wf += __popc(m);
                        }
                        cnt_f = wf;
                        fmin = warp_min_ll(new_fmin);
                        {
                            const sbits_t nm = warp_min_ll(l_nmin);
                            nmin = (nm < nmin) ? nm : nmin;
                        }
                        D_far = Fnew;
                        if (total < FAR_TARGET) {
                            delta_far = (delta_far < DELTA_MAX) ? (delta_far << 1) : DELTA_MAX;
                        } else if (total > 3 * FAR_TARGET) {
                            delta_far = (delta_far > DELTA_MIN) ? (delta_far >> 1) : DELTA_MIN;
                        }
                        __syncwarp();
                    } else if (Dn > D_far) {
                        D_far = Dn;
                    }

                    // Pass 1 over the near tier: collect the stale bounds in
                    // the band for (converged) re-evaluation; histogram the
                    // fresh ones directly.
                    const sbits_t width = Dn - D;
                    const float winv = hist_scale(width);
                    S.hist[lane] = 0;
                    __syncwarp();
                    sbits_t a_min = SBITS_INF;
                    int n_eval = 0;
                    for (int b0 = 0; b0 < cnt_n; b0 += 32) {
                        const int e = b0 + lane;
                        bool want = false;
                        if (e < cnt_n) {
                            const sbits_t sb = N_S(e);
                            if (sb <= Dn) {
                                if (N_K0(e) != k) {
                                    want = true;
                                } else {
                                    hist_add(S, sb, D, winv);
                                    a_min = (sb < a_min) ? sb : a_min;
                                }
                            }
                        }
                        const unsigned m = __ballot_sync(FULL_MASK, want);
                        if (want) e_pos[n_eval + __popc(m & lanemask_lt)] = e;
                        n_eval += __popc(m);
                    }
                    __syncwarp();
                    for (int b0 = 0; b0 < n_eval; b0 += 32) {
                        const int ee = b0 + lane;
                        if (ee < n_eval) {
                            const int e = e_pos[ee];
                            const int c = N_IDX(e);
                            sbits_t sb = N_S(e);
                            int wit = N_W(e);
                            const double cx = px[c];
                            const double cy = py[c];
                            const float cfx = pxf[c];
                            const float cfy = pyf[c];
                            // Cheap test first: the previous witness alone is a
                            // valid lower bound; if it already exceeds the band
                            // the full hull evaluation is unnecessary.
                            const sbits_t sw = __double_as_longlong(dev_sqdist(px[wit], py[wit], cx, cy));
                            sb = (sw > sb) ? sw : sb;
                            if (sb <= Dn) {
                                sb = hull_valid ? hull_bound(S, h, cx, cy, cfx, cfy, sb, wit)
                                                : exact_s(px, py, m_idx, m_xy, k, cx, cy, cfx, cfy, sb);
                            }
                            if (sb < s_crit_bits && sb + TOL_BITS >= s_crit_bits) {
                                sb = exact_s(px, py, m_idx, m_xy, k, cx, cy, cfx, cfy, sb);
                            }
                            if (!(sb < s_crit_bits)) sb = SBITS_INF;   // dropped in pass 2
                            N_PUT(e, c, sb, k, wit);
                            if (sb <= Dn) {
                                hist_add(S, sb, D, winv);
                                a_min = (sb < a_min) ? sb : a_min;
                            }
                        }
                    }
                    __syncwarp();

                    // Choose the new D so that about ACTIVE_TARGET candidates
                    // get activated (at least the lowest bucket).
                    int total = 0;
                    sbits_t Dnew = pick_bound(S, lane, D, width, Dn, ACTIVE_TARGET, total);
                    a_min = warp_min_ll(a_min);
                    if (a_min != SBITS_INF && a_min + NEAR_BITS > Dnew) Dnew = a_min + NEAR_BITS;
                    if (Dnew > Dn) Dnew = Dn;

                    // Pass 2: activate everything with a (fresh) bound <= Dnew,
                    // drop the entries marked out of range, push entries whose
                    // fresh bound is beyond D_far back to the far tier, compact.
                    sbits_t new_nmin = SBITS_INF, l_fmin = SBITS_INF;
                    int wn = 0;
                    for (int b0 = 0; b0 < cnt_n; b0 += 32) {
                        const int e = b0 + lane;
                        const bool have = e < cnt_n;
                        int c = 0, k0 = 0, wit = 0;
                        sbits_t sb = SBITS_INF;
                        if (have) { c = N_IDX(e); sb = N_S(e); k0 = N_K0(e); wit = N_W(e); }
                        __syncwarp();
                        if (have && sb == SBITS_INF) grid_add(S, grid_cell(pxf[c], pyf[c], gx0, gy0, ginv), -1);
                        const bool act = have && (sb <= Dnew);
                        unsigned m = __ballot_sync(FULL_MASK, act);
                        if (act) {
                            const int pos = cnt_a + __popc(m & lanemask_lt);
                            A_PUT(pos, c, sb, pxf[c], pyf[c]);
                            track_min(sb, c, ls, li, flag, tiny_bits);
                        }
                        cnt_a += __popc(m);
                        const bool tofar = have && !act && (sb != SBITS_INF) && (sb > D_far + delta_far);
                        m = __ballot_sync(FULL_MASK, tofar);
                        if (tofar) {
                            const int pos = cnt_f + __popc(m & lanemask_lt);
                            f_idx[pos] = c;
                            f_s[pos] = sb;
                            f_k0[pos] = k0;
                            f_w[pos] = wit;
                        }
                        cnt_f += __popc(m);
                        if (tofar) l_fmin = (sb < l_fmin) ? sb : l_fmin;
                        const bool st = have && !act && !tofar && (sb != SBITS_INF);
                        m = __ballot_sync(FULL_MASK, st);
                        if (st) {
                            const int pos = wn + __popc(m & lanemask_lt);
                            N_PUT(pos, c, sb, k0, wit);
                            new_nmin = (sb < new_nmin) ? sb : new_nmin;
                        }
                        wn += __popc(m);
                    }
                    cnt_n = wn;
                    nmin = warp_min_ll(new_nmin);
                    {
                        const sbits_t fm = warp_min_ll(l_fmin);
                        fmin = (fm < fmin) ? fm : fmin;
                    }
                    D = Dnew;
                    __syncwarp();

                    // Adapt the trial band width to the candidate density so
                    // that the band holds roughly the activation batch.
                    if (total < ACTIVE_TARGET) {
                        delta = (delta < DELTA_MAX) ? (delta << 1) : DELTA_MAX;
                    } else if (total > 2 * ACTIVE_TARGET) {
                        delta = (delta > DELTA_MIN) ? (delta >> 1) : DELTA_MIN;
                    }

                    a = ls;
                    best = li;
                    near = flag;
                    warp_argmin(a, best, near, tiny_bits);
                    if (a != SBITS_INF && a + NEAR_BITS <= D) break;
                }
            }
            if (best == INT_MAX) break;   // no candidate keeps diameter < threshold

            if (near) {
                // Slow exact path: exact s for every active candidate, then
                // arg-min on (sqrt(s), index) as in the sequential code (sqrt
                // is monotonic, so comparing bit patterns of the roots is exact).
                sbits_t es = SBITS_INF;
                int ei = INT_MAX;
                for (int e = lane; e < cnt_a; e += 32) {
                    const int c = A_IDX(e);
                    const sbits_t sb = exact_s(px, py, m_idx, m_xy, k, px[c], py[c], pxf[c], pyf[c], A_S(e));
                    const sbits_t v = __double_as_longlong(sqrt(__longlong_as_double(sb)));
                    if (v < es || (v == es && c < ei)) {
                        es = v;
                        ei = c;
                    }
                }
                warp_argmin_exact(es, ei);
                best = ei;
            }
        }

        if (aborted) {
            if (lane == 0) cardinality_out[task] = 0;
        } else {
            const unsigned long long my_key = bb_key(k, seed);
            int won = 0;
            if (lane == 0) {
                cardinality_out[task] = k;
                won = (atomicMax(best_key, my_key) < my_key) ? 1 : 0;
            }
            won = __shfl_sync(FULL_MASK, won, 0);
            if (won) {
                // Publish the member list of the (currently) best cluster.
                // The lock serialises writers; the re-check guarantees that
                // the final buffer holds the members of the final best key.
                if (lane == 0) while (atomicCAS(best_lock, 0, 1) != 0) { }
                __syncwarp();
                __threadfence();
                if (*reinterpret_cast<volatile unsigned long long*>(best_key) == my_key) {
                    for (int i = lane; i < k; i += 32) best_members[i] = m_idx[i];
                }
                __threadfence();
                __syncwarp();
                if (lane == 0) atomicExch(best_lock, 0);
            }
        }
        __syncwarp();
    }
#undef A_IDX
#undef A_S
#undef N_IDX
#undef N_S
#undef N_K0
#undef N_W
#undef A_PUT
#undef N_PUT
}

// ---------------------------------------------------------------------------
// Per-GPU context (one per OpenMP thread of a rank)
// ---------------------------------------------------------------------------
struct GpuContext {
    int device = -1;
    int max_blocks = 0;
    int cap_warp = 0;
    int cap_members = 0;
    double* d_px = nullptr;
    double* d_py = nullptr;
    float* d_pxf = nullptr;
    float* d_pyf = nullptr;
    long long* d_pix = nullptr;
    long long* d_piy = nullptr;
    int* d_ux_idx = nullptr;
    float* d_ux_f = nullptr;
    float* d_ux_fy = nullptr;
    int* d_bstart = nullptr;
    int* d_seeds = nullptr;
    int* d_counts = nullptr;
    int* d_card = nullptr;
    int* d_counter = nullptr;
    unsigned long long* d_best = nullptr;
    int* d_best_lock = nullptr;
    int* d_best_members = nullptr;
    WarpScratch ws = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    cudaStream_t stream = nullptr;
};

static void initGpuContext(GpuContext& ctx, const int device, const std::vector<Point>& points,
                           const int fixed_shift) {
    const int N = static_cast<int>(points.size());
    ctx.device = device;
    CUDA_CHECK(cudaSetDevice(device));
    // Several host threads (one per GPU) may share few cores (e.g. under MPI
    // core binding): yield while waiting for the GPU instead of spinning.
    (void)cudaSetDeviceFlags(cudaDeviceScheduleYield);
    (void)cudaGetLastError();
    CUDA_CHECK(cudaStreamCreateWithFlags(&ctx.stream, cudaStreamNonBlocking));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    int blocks_per_sm = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, qt_candidate_kernel,
                                                             QT_BLOCK, 0));
    if (blocks_per_sm < 1) blocks_per_sm = 1;
    int max_blocks = prop.multiProcessorCount * blocks_per_sm;

    ctx.cap_warp = ((N + 31) / 32) * 32 + 32;
    ctx.cap_members = N;
    const size_t per_warp = static_cast<size_t>(ctx.cap_warp) * (8 * sizeof(int) + 3 * sizeof(sbits_t))
                          + static_cast<size_t>(ctx.cap_members) * (sizeof(int) + sizeof(float2));
    const size_t per_block = per_warp * QT_WARPS;

    // Bound the scratch memory.
    size_t free_mem = 0, total_mem = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    const size_t budget = free_mem / 2;
    const size_t by_mem = std::max<size_t>(1, budget / std::max<size_t>(per_block, 1));
    if (static_cast<size_t>(max_blocks) > by_mem) max_blocks = static_cast<int>(by_mem);
    ctx.max_blocks = std::max(1, max_blocks);

    std::vector<double> hx(N), hy(N);
    std::vector<float> hxf(N), hyf(N);
    std::vector<long long> hix(N), hiy(N);
    for (int i = 0; i < N; ++i) {
        hx[i] = points[i].x;
        hy[i] = points[i].y;
        hxf[i] = static_cast<float>(points[i].x);
        hyf[i] = static_cast<float>(points[i].y);
        hix[i] = static_cast<long long>(std::floor(std::ldexp(points[i].x, fixed_shift)));
        hiy[i] = static_cast<long long>(std::floor(std::ldexp(points[i].y, fixed_shift)));
    }
    CUDA_CHECK(cudaMalloc(&ctx.d_px, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&ctx.d_py, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&ctx.d_pxf, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ctx.d_pyf, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ctx.d_pix, N * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&ctx.d_piy, N * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&ctx.d_ux_idx, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ctx.d_ux_f, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ctx.d_ux_fy, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ctx.d_bstart, (SLAB_BUCKETS + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ctx.d_seeds, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ctx.d_counts, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ctx.d_best, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&ctx.d_best_lock, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ctx.d_best_members, N * sizeof(int)));
    CUDA_CHECK(cudaMemset(ctx.d_best_lock, 0, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ctx.d_card, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ctx.d_counter, sizeof(int)));
    const size_t nwarps = static_cast<size_t>(ctx.max_blocks) * QT_WARPS;
    const size_t entries = nwarps * ctx.cap_warp;
    const size_t mentries = nwarps * ctx.cap_members;
    CUDA_CHECK(cudaMalloc(&ctx.ws.a_idx, entries * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ctx.ws.a_s, entries * sizeof(sbits_t)));
    CUDA_CHECK(cudaMalloc(&ctx.ws.n_idx, entries * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ctx.ws.n_s, entries * sizeof(sbits_t)));
    CUDA_CHECK(cudaMalloc(&ctx.ws.n_k0, entries * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ctx.ws.n_w, entries * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ctx.ws.f_idx, entries * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ctx.ws.f_s, entries * sizeof(sbits_t)));
    CUDA_CHECK(cudaMalloc(&ctx.ws.f_k0, entries * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ctx.ws.f_w, entries * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ctx.ws.e_pos, entries * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ctx.ws.m_idx, mentries * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ctx.ws.m_xy, mentries * sizeof(float2)));
    CUDA_CHECK(cudaMemcpy(ctx.d_px, hx.data(), N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx.d_py, hy.data(), N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx.d_pxf, hxf.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx.d_pyf, hyf.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx.d_pix, hix.data(), N * sizeof(long long), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx.d_piy, hiy.data(), N * sizeof(long long), cudaMemcpyHostToDevice));
}

static void destroyGpuContext(GpuContext& ctx) {
    if (ctx.device < 0) return;
    cudaSetDevice(ctx.device);
    cudaFree(ctx.d_px);
    cudaFree(ctx.d_py);
    cudaFree(ctx.d_pxf);
    cudaFree(ctx.d_pyf);
    cudaFree(ctx.d_pix);
    cudaFree(ctx.d_piy);
    cudaFree(ctx.d_ux_idx);
    cudaFree(ctx.d_ux_f);
    cudaFree(ctx.d_ux_fy);
    cudaFree(ctx.d_bstart);
    cudaFree(ctx.d_seeds);
    cudaFree(ctx.d_counts);
    cudaFree(ctx.d_best);
    cudaFree(ctx.d_best_lock);
    cudaFree(ctx.d_best_members);
    cudaFree(ctx.d_card);
    cudaFree(ctx.d_counter);
    cudaFree(ctx.ws.a_idx);
    cudaFree(ctx.ws.a_s);
    cudaFree(ctx.ws.n_idx);
    cudaFree(ctx.ws.n_s);
    cudaFree(ctx.ws.n_k0);
    cudaFree(ctx.ws.n_w);
    cudaFree(ctx.ws.f_idx);
    cudaFree(ctx.ws.f_s);
    cudaFree(ctx.ws.f_k0);
    cudaFree(ctx.ws.f_w);
    cudaFree(ctx.ws.e_pos);
    cudaFree(ctx.ws.m_idx);
    cudaFree(ctx.ws.m_xy);
    cudaStreamDestroy(ctx.stream);
    ctx.device = -1;
}

// Candidate set for one outer iteration: the unclustered points sorted by x.
struct CandidateSet {
    std::vector<int> idx;     // point indices, ascending x
    std::vector<float> xf;    // float x coordinate of idx[i]
    std::vector<float> yf;    // float y coordinate of idx[i]
    std::vector<int> bstart;  // bstart[b] = first position with bucket(x) >= b (SLAB_BUCKETS + 1 entries)
    float xmin = 0.f, xscale = 0.f;   // bucket(x) = (x - xmin) * xscale
};

// Evaluate the candidate cluster of every seed in `seeds` on one GPU with
// branch and bound.  Returns the best (cardinality, seed) of this GPU and its
// member list (empty if nseeds == 0).
struct GpuBest {
    int card = 0;
    int seed = INT_MAX;
    std::vector<int> members;
};

static void evaluateSeedsOnGpu(GpuContext& ctx,
                               const CandidateSet& cand,
                               const int* seeds_in, const int nseeds,
                               const double s_crit, const sbits_t tiny_bits, const float thr_slab,
                               const int grid_w,
                               GpuBest& out,
                               std::vector<int>& seed_order,     // scratch
                               std::vector<int>& count_buf,      // scratch
                               std::vector<unsigned long long>& sort_keys) {   // scratch
    out.card = 0;
    out.seed = INT_MAX;
    out.members.clear();
    if (nseeds <= 0) return;
    const int U = cand.bstart[SLAB_BUCKETS];
    const float s_crit_hi_f = fmaf(static_cast<float>(s_crit), 1.0f + (1.0f - F32_REL), F32_ABS);
    CUDA_CHECK(cudaSetDevice(ctx.device));
    CUDA_CHECK(cudaMemcpyAsync(ctx.d_ux_idx, cand.idx.data(), U * sizeof(int),
                               cudaMemcpyHostToDevice, ctx.stream));
    CUDA_CHECK(cudaMemcpyAsync(ctx.d_ux_f, cand.xf.data(), U * sizeof(float),
                               cudaMemcpyHostToDevice, ctx.stream));
    CUDA_CHECK(cudaMemcpyAsync(ctx.d_ux_fy, cand.yf.data(), U * sizeof(float),
                               cudaMemcpyHostToDevice, ctx.stream));
    CUDA_CHECK(cudaMemcpyAsync(ctx.d_bstart, cand.bstart.data(), (SLAB_BUCKETS + 1) * sizeof(int),
                               cudaMemcpyHostToDevice, ctx.stream));
    CUDA_CHECK(cudaMemcpyAsync(ctx.d_seeds, seeds_in, nseeds * sizeof(int),
                               cudaMemcpyHostToDevice, ctx.stream));

    // Upper bounds, then process the seeds in descending bound order so that
    // the branch-and-bound cut-off becomes effective early.
    const int cblocks = (nseeds * 32 + QT_BLOCK - 1) / QT_BLOCK;
    qt_count_kernel<<<cblocks, QT_BLOCK, 0, ctx.stream>>>(
        ctx.d_pxf, ctx.d_pyf, ctx.d_ux_f, ctx.d_ux_fy, ctx.d_bstart, cand.xmin, cand.xscale,
        ctx.d_seeds, nseeds, s_crit_hi_f, thr_slab, ctx.d_counts);
    CUDA_CHECK(cudaGetLastError());
    count_buf.resize(nseeds);
    CUDA_CHECK(cudaMemcpyAsync(count_buf.data(), ctx.d_counts, nseeds * sizeof(int),
                               cudaMemcpyDeviceToHost, ctx.stream));
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
    // Sort by (bound descending, seed ascending) using packed 64-bit keys.
    std::vector<unsigned long long>& keys = sort_keys;
    keys.resize(nseeds);
    for (int i = 0; i < nseeds; ++i) {
        keys[i] = (static_cast<unsigned long long>(static_cast<unsigned>(INT_MAX - count_buf[i])) << 32) |
                  static_cast<unsigned>(seeds_in[i]);
    }
    std::sort(keys.begin(), keys.end());
    seed_order.resize(nseeds);
    for (int i = 0; i < nseeds; ++i) {
        seed_order[i] = static_cast<int>(keys[i] & 0xffffffffu);
        count_buf[i] = INT_MAX - static_cast<int>(keys[i] >> 32);
    }
    CUDA_CHECK(cudaMemcpyAsync(ctx.d_seeds, seed_order.data(), nseeds * sizeof(int),
                               cudaMemcpyHostToDevice, ctx.stream));
    CUDA_CHECK(cudaMemcpyAsync(ctx.d_counts, count_buf.data(), nseeds * sizeof(int),
                               cudaMemcpyHostToDevice, ctx.stream));
    CUDA_CHECK(cudaMemsetAsync(ctx.d_counter, 0, sizeof(int), ctx.stream));
    CUDA_CHECK(cudaMemsetAsync(ctx.d_best, 0, sizeof(unsigned long long), ctx.stream));

    const int blocks = std::min(ctx.max_blocks, (nseeds + QT_WARPS - 1) / QT_WARPS);
    qt_candidate_kernel<<<blocks, QT_BLOCK, 0, ctx.stream>>>(
        ctx.d_px, ctx.d_py, ctx.d_pxf, ctx.d_pyf, ctx.d_pix, ctx.d_piy, ctx.d_ux_idx, ctx.d_ux_f,
        ctx.d_ux_fy, ctx.d_bstart, cand.xmin, cand.xscale, U,
        ctx.d_seeds, ctx.d_counts, nseeds, s_crit, tiny_bits, thr_slab, grid_w,
        ctx.d_counter, ctx.d_best, ctx.d_best_lock, ctx.d_best_members, ctx.d_card,
        ctx.ws, ctx.cap_warp, ctx.cap_members);
    CUDA_CHECK(cudaGetLastError());

    unsigned long long key = 0;
    CUDA_CHECK(cudaMemcpyAsync(&key, ctx.d_best, sizeof(key), cudaMemcpyDeviceToHost, ctx.stream));
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
    out.card = static_cast<int>(key >> 32);
    out.seed = INT_MAX - static_cast<int>(key & 0xffffffffu);
    if (out.card > 0) {
        out.members.resize(out.card);
        CUDA_CHECK(cudaMemcpy(out.members.data(), ctx.d_best_members, out.card * sizeof(int),
                              cudaMemcpyDeviceToHost));
    }
}

// ---------------------------------------------------------------------------
// Main QT clustering algorithm (MPI + OpenMP + CUDA)
// ---------------------------------------------------------------------------
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold,
                                  std::vector<GpuContext>& gpus,
                                  const int fixed_shift,
                                  const int mpi_rank, const int mpi_size) {
    const int N = static_cast<int>(points.size());
    const int ngpu = static_cast<int>(gpus.size());
    const double s_crit = computeSquaredCritical(threshold);
    // Below this squared distance the relative bound of the hull estimate no
    // longer holds (fixed-point grid 2^-fixed_shift); such values are
    // resolved exactly.  2^(2*(36 - S)) <=> distance >= 2^(36 - S).
    const sbits_t tiny_bits = hostDoubleBits(std::ldexp(1.0, 2 * (36 - fixed_shift)));
    // x-slab half width in float with margin for the double->float rounding.
    const float thr_slab = static_cast<float>(threshold) * (1.0f + 1e-5f) + 1e-4f;
    // Jung window: a disc of radius threshold/sqrt(3) intersects at most
    // floor(2R/g) + 2 cells per axis (g = cell size), conservatively rounded.
    int grid_w;
    {
        const double g = 2.0 * static_cast<double>(thr_slab) / GRID_G;
        const double R = threshold / std::sqrt(3.0) * (1.0 + 1e-6);
        grid_w = static_cast<int>(std::floor(2.0 * R / g)) + 2;
        grid_w = std::min(std::max(grid_w, 1), GRID_G);
    }
    std::vector<char> clustered(N, 0);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;

    // Initialize unclustered indices
    unclustered_indices.reserve(N);
    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }

    // Points ordered by x (ties by index) for the candidate slabs, with their
    // float coordinates in that order.
    std::vector<int> x_order(N);
    std::iota(x_order.begin(), x_order.end(), 0);
    std::stable_sort(x_order.begin(), x_order.end(),
                     [&points](int a, int b) { return points[a].x < points[b].x; });
    std::vector<float> xo_xf(N), xo_yf(N);
    std::vector<int> xo_bucket(N);
    for (int i = 0; i < N; ++i) {
        xo_xf[i] = static_cast<float>(points[x_order[i]].x);
        xo_yf[i] = static_cast<float>(points[x_order[i]].y);
    }
    CandidateSet cand;
    cand.idx.resize(N);
    cand.xf.resize(N);
    cand.yf.resize(N);
    cand.bstart.resize(SLAB_BUCKETS + 1);
    {
        const float x_lo = xo_xf.front();
        const float x_hi = xo_xf.back();
        cand.xmin = x_lo;
        cand.xscale = (x_hi > x_lo) ? (static_cast<float>(SLAB_BUCKETS) / (x_hi - x_lo)) * (1.0f - 1e-6f)
                                    : 0.f;
        for (int i = 0; i < N; ++i) {
            int b = static_cast<int>(std::floor((xo_xf[i] - cand.xmin) * cand.xscale));
            xo_bucket[i] = std::min(std::max(b, 0), SLAB_BUCKETS - 1);
        }
    }
    std::vector<int> bucket_cnt(SLAB_BUCKETS);

    // Seeds handled by this rank (cyclic distribution for load balance),
    // split again cyclically over the GPUs of this rank.
    std::vector<std::vector<int>> gpu_seeds(ngpu);
    std::vector<GpuBest> gpu_best(ngpu);
    std::vector<std::vector<int>> gpu_order(ngpu), gpu_counts(ngpu);
    std::vector<std::vector<unsigned long long>> gpu_keys(ngpu);
    for (int g = 0; g < ngpu; ++g) {
        gpu_seeds[g].reserve(N / (mpi_size * ngpu) + 1);
    }
    std::vector<int> best_members;

    // Main clustering loop
    while (!unclustered_indices.empty()) {
        const int U = static_cast<int>(unclustered_indices.size());

        // Candidate set: unclustered points in x order, plus the bucket table.
        {
            int n = 0;
            int* __restrict__ cidx = cand.idx.data();
            float* __restrict__ cxf = cand.xf.data();
            float* __restrict__ cyf = cand.yf.data();
            std::fill(bucket_cnt.begin(), bucket_cnt.end(), 0);
            for (int i = 0; i < N; ++i) {
                const int p = x_order[i];
                const int keep = (clustered[p] == 0);
                cidx[n] = p;
                cxf[n] = xo_xf[i];
                cyf[n] = xo_yf[i];
                bucket_cnt[xo_bucket[i]] += keep;
                n += keep;
            }
            // bstart[b] = first position whose bucket >= b (exclusive prefix sum)
            int pos = 0;
            for (int b = 0; b < SLAB_BUCKETS; ++b) {
                cand.bstart[b] = pos;
                pos += bucket_cnt[b];
            }
            cand.bstart[SLAB_BUCKETS] = n;
        }

        // Assign this rank's seeds to its GPUs.
        for (int g = 0; g < ngpu; ++g) gpu_seeds[g].clear();
        int local_count = 0;
        for (int i = mpi_rank; i < U; i += mpi_size, ++local_count) {
            gpu_seeds[local_count % ngpu].push_back(unclustered_indices[i]);
        }

        // Try each seed of this rank: one OpenMP thread per GPU.
#pragma omp parallel for num_threads(ngpu) schedule(static, 1)
        for (int g = 0; g < ngpu; ++g) {
            evaluateSeedsOnGpu(gpus[g], cand, gpu_seeds[g].data(),
                               static_cast<int>(gpu_seeds[g].size()), s_crit, tiny_bits, thr_slab,
                               grid_w, gpu_best[g], gpu_order[g], gpu_counts[g], gpu_keys[g]);
        }

        // Local best: maximal cardinality, smallest seed index on ties.
        int local_best[2] = {-1, INT_MAX};   // {cardinality, seed}
        int local_gpu = -1;
        for (int g = 0; g < ngpu; ++g) {
            const int card = gpu_best[g].card;
            const int seed = gpu_best[g].seed;
            if (card > 0 && (card > local_best[0] || (card == local_best[0] && seed < local_best[1]))) {
                local_best[0] = card;
                local_best[1] = seed;
                local_gpu = g;
            }
        }

        // Global best via MAXLOC (max value, smallest location on ties).
        int global_best[2];
        MPI_CHECK(MPI_Allreduce(local_best, global_best, 1, MPI_2INT, MPI_MAXLOC,
                                MPI_COMM_WORLD));
        const int max_cardinality = global_best[0];
        const int best_seed = global_best[1];

        // If we found a cluster, add it
        if (best_seed >= 0 && max_cardinality > 0) {
            // The owning rank broadcasts the member list (ranks are
            // identified by the seed assignment: seed index position mod size).
            int owner = -1;
            for (int i = 0; i < U; ++i) {
                if (unclustered_indices[i] == best_seed) { owner = i % mpi_size; break; }
            }
            best_members.resize(max_cardinality);
            if (mpi_rank == owner) {
                best_members = gpu_best[local_gpu].members;
            }
            if (mpi_size > 1) {
                MPI_CHECK(MPI_Bcast(best_members.data(), max_cardinality, MPI_INT, owner,
                                    MPI_COMM_WORLD));
            }

            if (mpi_rank == 0) {
                Cluster cluster;
                cluster.seed_point = best_seed;
                cluster.members = best_members;
                clusters.push_back(cluster);
            }

            // Mark all members as clustered
            for (size_t i = 0; i < best_members.size(); ++i) {
                clustered[best_members[i]] = 1;
            }

            // Remove clustered points from unclustered list
            unclustered_indices.erase(
                std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                              [&clustered](int idx) { return clustered[idx] != 0; }),
                unclustered_indices.end()
            );
        } else {
            // No more clusters can be formed
            break;
        }
    }

    return clusters;
}

// Validation: check that clusters satisfy the QT clustering properties
bool validateClusters(const std::vector<Cluster>& clusters,
                     const std::vector<Point>& points,
                     const double threshold) {
    bool valid = true;

    printf("Validating clusters:\n");

    // Compute the diameter of every cluster (max distance between any two
    // points) in parallel; the pair loops are flattened over all clusters.
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

int main(int argc, char** argv) {
    int provided = 0;
    MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));
    int mpi_rank = 0, mpi_size = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &mpi_size));

    int num_points = 1000;
    double threshold = 2.0;
    bool validate = false;
    bool printResults = false;
    int exit_code = 0;
    bool early_exit = false;

    // Parse command line arguments
    for (int i = 1; i < argc && !early_exit; ++i) {
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
            early_exit = true;
            exit_code = 0;
        } else {
            if (mpi_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            early_exit = true;
            exit_code = 1;
        }
    }
    if (early_exit) {
        MPI_Finalize();
        return exit_code;
    }

    if (num_points <= 0 || threshold <= 0.0) {
        if (mpi_rank == 0) {
            printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
                   num_points, threshold);
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

    // ---- GPU assignment: ranks sharing a node share its GPUs -------------
    MPI_Comm node_comm;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, mpi_rank,
                                  MPI_INFO_NULL, &node_comm));
    int node_rank = 0, node_size = 1;
    MPI_CHECK(MPI_Comm_rank(node_comm, &node_rank));
    MPI_CHECK(MPI_Comm_size(node_comm, &node_size));
    MPI_Comm_free(&node_comm);

    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count < 1) {
        fprintf(stderr, "Rank %d: no CUDA device available\n", mpi_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    std::vector<int> my_devices;
    if (node_size >= device_count) {
        my_devices.push_back(node_rank % device_count);
    } else {
        // Fewer ranks than GPUs on this node: spread the GPUs over the ranks.
        for (int d = node_rank; d < device_count; d += node_size) my_devices.push_back(d);
    }

    // Fixed-point scale for the exact hull predicates: |coord| * 2^S < 2^60.
    double max_abs = 1.0;
    for (const auto& p : points) max_abs = std::max(max_abs, std::max(std::fabs(p.x), std::fabs(p.y)));
    int exp2 = 0;
    std::frexp(max_abs, &exp2);          // max_abs < 2^exp2
    const int fixed_shift = 60 - exp2;

    std::vector<GpuContext> gpus(my_devices.size());
#pragma omp parallel for num_threads(static_cast<int>(my_devices.size())) schedule(static, 1)
    for (int g = 0; g < static_cast<int>(my_devices.size()); ++g) {
        initGpuContext(gpus[g], my_devices[g], points, fixed_shift);
    }
    if (mpi_rank == 0) {
        printf("MPI ranks: %d, GPUs on rank 0: %d, OpenMP threads: %d\n",
               mpi_size, static_cast<int>(my_devices.size()), omp_get_max_threads());
    }

    // Perform QT clustering
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto cluster_start = std::chrono::high_resolution_clock::now();

    const std::vector<Cluster> clusters = qtClustering(points, threshold, gpus, fixed_shift, mpi_rank, mpi_size);

    auto cluster_end = std::chrono::high_resolution_clock::now();
    const long local_cluster_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start).count();
    long max_cluster_ms = 0;
    MPI_CHECK(MPI_Reduce(&local_cluster_ms, &max_cluster_ms, 1, MPI_LONG, MPI_MAX,
                         0, MPI_COMM_WORLD));
    auto cluster_time = std::chrono::milliseconds(max_cluster_ms);

    for (auto& g : gpus) destroyGpuContext(g);

    if (mpi_rank == 0) {
        printf("Clustering time: %ld ms\n", static_cast<long>(cluster_time.count()));
        printf("Clusters found: %zu\n", clusters.size());

        // Calculate statistics and performance metrics
        int total_clustered = 0;
        int max_cluster_size = 0;

#pragma omp parallel for reduction(+ : total_clustered) reduction(max : max_cluster_size)
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
                exit_code = 0;
            } else {
                printf("Validation: FAILED\n");
                exit_code = 1;
            }
        }
    }

    // All ranks return the same exit code.
    MPI_CHECK(MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD));
    MPI_Finalize();
    return exit_code;
}
