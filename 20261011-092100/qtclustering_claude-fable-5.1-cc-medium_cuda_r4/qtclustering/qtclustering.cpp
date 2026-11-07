// QT Clustering Benchmark - CUDA Parallel Version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// Parallelization strategy (results are bit-identical to the sequential
// version, including tie-breaking):
//  * Every unclustered point is tried as a seed. Candidate-cluster generation
//    for different seeds is independent, so one thread block is assigned per
//    seed and batches of seeds are spread over all available GPUs.
//  * Only points within `threshold` of the seed can ever join its cluster (the
//    seed is a member), so a block first compacts the unclustered points inside
//    that radius into a working list that lives in registers (variants for up
//    to 32 entries per thread) or in global scratch for huge neighbourhoods.
//  * Each greedy growth step selects argmin over candidates of the maximum
//    distance to the current members. A float running maximum (seed-relative
//    coordinates, rigorous rounding-error bound) is maintained per candidate;
//    only candidates whose error interval can reach the smallest upper bound
//    are brought up to date in exact double arithmetic and compared with the
//    original double criteria, so the selection is exact.
//  * The result of a seed depends only on the unclustered points within
//    `threshold` of it. After a cluster is committed, only seeds whose
//    neighbourhood lost a point are re-evaluated; the neighbourhood size is
//    an upper bound on the cardinality, so seeds that cannot beat the running
//    best are not evaluated at all. Member lists of evaluated seeds are kept
//    on the device so the winner's cluster rarely needs regeneration.
//  * Floating point: exact distances use fma(dx, dx, dy*dy) followed by an
//    IEEE sqrt, which is exactly what the optimized host build computes.

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
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

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                         \
                    cudaGetErrorString(err__), __FILE__, __LINE__);             \
            exit(1);                                                            \
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

// Calculate Euclidean distance between two points (host, used for validation)
inline double distance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

// ---------------------------------------------------------------------------
// Device code
// ---------------------------------------------------------------------------

namespace {

constexpr int BLOCK_THREADS = 128;
constexpr int WARPS_PER_BLOCK = BLOCK_THREADS / 32;
// Largest register-resident working list (candidates per thread = 32)
constexpr int MAX_REG_K = 32;
// Variants whose float member coordinates fit into shared memory (8 B per entry)
constexpr int MAX_SMEM_K = 32;
template <int K>
__host__ __device__ constexpr bool membersInSmem() { return K > 0 && K <= MAX_SMEM_K; }
template <int K>
__host__ __device__ constexpr size_t memberSmemBytes() {
    return membersInSmem<K>() ? 2 * sizeof(float) * K * BLOCK_THREADS : 0;
}
// Bytes of global scratch per working-list entry (see WorkList)
constexpr int BYTES_PER_CANDIDATE = 88;
// Relative slack that makes squared-distance prefilters a strict superset of
// the exact sqrt-based tests (sqrt rounding changes values by < 2^-52 rel.)
constexpr double SQ_SLACK = 1.0 + 1.0 / 1125899906842624.0;   // 1 + 2^-50
constexpr float SQ_SLACK_F = 1.0f + 1.0f / 1048576.0f;         // 1 + 2^-20

// Screening parameters derived on the host (see makeScreen)
struct Screen {
    float err;       // |float squared distance - exact squared distance| <= err
                     // for seed-relative float coordinates
    float thr2_hi;   // >= threshold^2 * SQ_SLACK
    float pass_abs;  // float squared distances (absolute coordinates) below
                     // this may be < threshold; above it they certainly are
                     // not. INF disables the prefilter.
};

// Squared distance, bit-identical to the host build: fma(dx, dx, dy*dy)
__device__ __forceinline__ double dist2(double ax, double ay, double bx, double by) {
    const double dx = __dsub_rn(ax, bx);
    const double dy = __dsub_rn(ay, by);
    return __fma_rn(dx, dx, __dmul_rn(dy, dy));
}

// Float approximation of the squared distance (no fma, so the error analysis
// in makeScreen applies)
__device__ __forceinline__ float dist2f(float ax, float ay, float bx, float by) {
    const float dx = __fsub_rn(ax, bx);
    const float dy = __fsub_rn(ay, by);
    return __fadd_rn(__fmul_rn(dx, dx), __fmul_rn(dy, dy));
}

struct Best {
    double val;   // max distance of the candidate to the cluster
    int orig;     // original point index (tie breaker: lowest wins)
    int pos;      // position in the compact working list
};


__device__ __forceinline__ bool better(double v, int o, const Best& b) {
    return (v < b.val) || (v == b.val && o < b.orig);
}

template <int WIDTH = 32>
__device__ __forceinline__ void warpReduceBest(Best& b) {
#pragma unroll
    for (int off = WIDTH / 2; off > 0; off >>= 1) {
        const double v = __shfl_down_sync(0xffffffffu, b.val, off);
        const int o = __shfl_down_sync(0xffffffffu, b.orig, off);
        const int p = __shfl_down_sync(0xffffffffu, b.pos, off);
        if (better(v, o, b)) { b.val = v; b.orig = o; b.pos = p; }
    }
}

__device__ __forceinline__ float warpMinF(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v = fminf(v, __shfl_down_sync(0xffffffffu, v, off));
    return v;
}

__device__ __forceinline__ double warpMaxD(double v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v = fmax(v, __shfl_xor_sync(0xffffffffu, v, off));
    return v;
}

// Per-block working list in global scratch (cap entries each)
struct WorkList {
    double* cmd;   // exact running max of squared distances over members [0, kc)
    double* cx;    // exact candidate coordinates
    double* cy;
    double* mx;    // member coordinates in insertion order
    double* my;
    float* fmx;    // the same, as seed-relative floats (see growCluster)
    float* fmy;
    float* fx;     // float coordinates / float running max (INF = member);
    float* fy;     // only used as staging when the list lives in registers
    float* mf;
    int* kc;       // number of members folded into cmd
    int* cidx;     // original point index
    int* mpos;     // working-list position of member j
    int* amb;      // overflow of the per-step ambiguity list
    float* ambv;   // float running max of the listed candidates (overflow)
    double* pmax;  // overflow of the per-step partial maxima
};

__device__ __forceinline__ WorkList carveWorkList(unsigned char* base, int cap) {
    WorkList w;
    w.cmd = reinterpret_cast<double*>(base);
    w.pmax = w.cmd + cap;
    w.cx = w.pmax + cap;
    w.cy = w.cx + cap;
    w.mx = w.cy + cap;
    w.my = w.mx + cap;
    w.fx = reinterpret_cast<float*>(w.my + cap);
    w.fy = w.fx + cap;
    w.mf = w.fy + cap;
    w.fmx = w.mf + cap;
    w.fmy = w.fmx + cap;
    w.kc = reinterpret_cast<int*>(w.fmy + cap);
    w.cidx = w.kc + cap;
    w.mpos = w.cidx + cap;
    w.amb = w.mpos + cap;
    w.ambv = reinterpret_cast<float*>(w.amb + cap);
    return w;
}

// Shared-memory capacity of the per-step ambiguity list; longer lists spill
// into global scratch (w.amb / w.pmax)
constexpr int AMB_CAP = 128;

// Exact maximum of squared distances between candidate (x, y) and the members
// j in {j0, j0 + stride, ...} below `members`. `cut` is (float running max of
// the candidate over all current members) - 2 * err: a member whose float
// value is below it is provably not the exact maximum, so only the few
// members at or above the cut are evaluated in double.
__device__ __forceinline__ double exactPartialMax(const double x, const double y,
                                                  const float fx, const float fy,
                                                  const float cut,
                                                  const double* mx, const double* my,
                                                  const float* fmx, const float* fmy,
                                                  int j0, const int stride, const int members)
{
    double part = 0.0;
    for (int j = j0; j < members; j += stride) {
        const float v = dist2f(fx, fy, fmx[j], fmy[j]);
        if (v >= cut) {
            const double d2 = dist2(x, y, mx[j], my[j]);
            part = (part < d2) ? d2 : part;
        }
    }
    return part;
}

// Grows the candidate cluster of one seed exactly like the sequential greedy
// procedure and returns its cardinality (and optionally its members).
//
// Each growth step selects argmin over candidates of max_{m in members}
// dist(c, m) subject to that value being < threshold, ties broken by the
// lowest point index. A float running max with a rigorous error bound is
// maintained for every candidate; only the candidates whose interval can
// reach the smallest upper bound are brought up to date in exact double
// arithmetic and compared with the original criteria.
//
// K > 0: the float working list is held in registers (K entries per thread,
//        list length <= K * BLOCK_THREADS). K == 0: generic, list in scratch.
template <int K>
__device__ void growCluster(const int seed,
                            const double* __restrict__ px,
                            const double* __restrict__ py,
                            const float* __restrict__ pxf,
                            const float* __restrict__ pyf,
                            const unsigned char* __restrict__ clustered,
                            const double threshold,
                            const Screen sc,
                            const int N,
                            const WorkList w,
                            float* smem_members,    // dynamic shared memory (may be unused)
                            int* out_card,
                            int* out_members)
{
    __shared__ int s_count;
    __shared__ int s_seed_pos;
    __shared__ int s_namb;
    __shared__ float s_min[WARPS_PER_BLOCK];
    __shared__ float s_nx, s_ny;
    __shared__ int s_amb[AMB_CAP];            // ambiguous candidates of the step
    __shared__ float s_ambv[AMB_CAP];         // their float running max
    __shared__ double s_part[AMB_CAP];        // partial exact maxima per work item
    __shared__ double s_cm[AMB_CAP];          // exact state of the listed candidates
    __shared__ double s_x[AMB_CAP];
    __shared__ double s_y[AMB_CAP];
    __shared__ int s_ci[AMB_CAP];
    __shared__ Best s_sel;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    const double sx = px[seed];
    const double sy = py[seed];
    const double thr2_hi = threshold * threshold * SQ_SLACK;
    const float err2 = __fadd_ru(sc.err, sc.err);

    if (tid == 0) { s_count = 0; s_namb = 0; }
    __syncthreads();

    // 1. Compact the unclustered points within `threshold` of the seed.
    //    (A point with dist(seed, p) >= threshold can never join: the seed is
    //    a member, so its max distance would be >= threshold.)
    const float sxf = pxf[seed], syf = pyf[seed];
    for (int i = tid; i < N; i += BLOCK_THREADS) {
        if (clustered[i]) continue;
        // Float prefilter (rigorous, see makeScreen), then the exact test
        // used by the original
        if (!(dist2f(pxf[i], pyf[i], sxf, syf) < sc.pass_abs)) continue;
        const double x = px[i], y = py[i];
        const double d2 = dist2(x, y, sx, sy);
        if (d2 >= thr2_hi) continue;
        if (!(sqrt(d2) < threshold)) continue;
        const int pos = atomicAdd(&s_count, 1);
        // Float coordinates relative to the seed (|value| < threshold), which
        // keeps the float rounding error small (see makeScreen)
        const float xf = static_cast<float>(__dsub_rn(x, sx));
        const float yf = static_cast<float>(__dsub_rn(y, sy));
        w.fx[pos] = xf;
        w.fy[pos] = yf;
        w.cx[pos] = x;
        w.cy[pos] = y;
        w.cmd[pos] = d2;
        w.kc[pos] = 1;
        w.cidx[pos] = i;
        if (i == seed) {
            w.mf[pos] = INFINITY;          // the seed is a member already
            s_seed_pos = pos;
        } else {
            w.mf[pos] = dist2f(xf, yf, 0.0f, 0.0f);
        }
    }
    __syncthreads();
    const int M = s_count;

    // Member coordinates in insertion order: shared memory when the variant
    // allows it, global scratch otherwise.
    // Exact member coordinates stay in global scratch (rarely touched); the
    // float ones are scanned at every refresh and live in shared memory when
    // the variant allows it.
    double* mx = w.mx;
    double* my = w.my;
    float* fmx = membersInSmem<K>() ? smem_members : w.fmx;
    float* fmy = membersInSmem<K>() ? fmx + K * BLOCK_THREADS : w.fmy;
    if (tid == 0) { w.mpos[0] = s_seed_pos; mx[0] = sx; my[0] = sy; fmx[0] = 0.0f; fmy[0] = 0.0f; }

    // Register copy of the float working list
    float rfx[K > 0 ? K : 1], rfy[K > 0 ? K : 1], rmf[K > 0 ? K : 1];
    if constexpr (K > 0) {
#pragma unroll
        for (int k = 0; k < K; ++k) {
            const int i = tid + k * BLOCK_THREADS;
            rfx[k] = (i < M) ? w.fx[i] : 0.0f;
            rfy[k] = (i < M) ? w.fy[i] : 0.0f;
            rmf[k] = (i < M) ? w.mf[i] : INFINITY;
        }
    }

    int members = 1;
    if (out_members && tid == 0) out_members[0] = seed;
    float nx = 0.0f, ny = 0.0f;   // most recently added member (relative to seed)

    // 2. Greedy growth
    while (members < M) {
        // 2a. Fold the newest member into the float running max and find the
        //     smallest upper bound among candidates that may still qualify.
        float lmin = INFINITY;
        if constexpr (K > 0) {
#pragma unroll
            for (int k = 0; k < K; ++k) {
                float v = rmf[k];
                if (v != INFINITY) {           // not a member (and i < M)
                    v = fmaxf(v, dist2f(rfx[k], rfy[k], nx, ny));
                    rmf[k] = v;
                    if (__fsub_rd(v, sc.err) < sc.thr2_hi) lmin = fminf(lmin, __fadd_ru(v, sc.err));
                }
            }
        } else {
            for (int i = tid; i < M; i += BLOCK_THREADS) {
                float v = w.mf[i];
                if (v == INFINITY) continue;
                v = fmaxf(v, dist2f(w.fx[i], w.fy[i], nx, ny));
                w.mf[i] = v;
                if (__fsub_rd(v, sc.err) < sc.thr2_hi) lmin = fminf(lmin, __fadd_ru(v, sc.err));
            }
        }
        lmin = warpMinF(lmin);
        if (lane == 0) s_min[warp] = lmin;
        __syncthreads();
        float m = s_min[0];
#pragma unroll
        for (int k = 1; k < WARPS_PER_BLOCK; ++k) m = fminf(m, s_min[k]);
        if (m == INFINITY) break;                  // nothing fits any more
        const float m_slack = __fmul_ru(m, SQ_SLACK_F);

        // 2b. Every candidate whose exact value could be <= the winner's
        //     (interval test, including possible exact ties) goes to the
        //     ambiguity list.
        if constexpr (K > 0) {
#pragma unroll
            for (int k = 0; k < K; ++k) {
                const float v = rmf[k];
                const float lo = __fsub_rd(v, sc.err);
                if ((v != INFINITY) && lo < sc.thr2_hi && lo <= m_slack) {
                    const int a = atomicAdd(&s_namb, 1);
                    const int c = tid + k * BLOCK_THREADS;
                    if (a < AMB_CAP) { s_amb[a] = c; s_ambv[a] = v; }
                    else { w.amb[a] = c; w.ambv[a] = v; }
                }
            }
        } else {
            for (int i = tid; i < M; i += BLOCK_THREADS) {
                const float v = w.mf[i];
                if (v == INFINITY) continue;
                const float lo = __fsub_rd(v, sc.err);
                if (lo < sc.thr2_hi && lo <= m_slack) {
                    const int a = atomicAdd(&s_namb, 1);
                    if (a < AMB_CAP) { s_amb[a] = i; s_ambv[a] = v; }
                    else { w.amb[a] = i; w.ambv[a] = v; }
                }
            }
        }
        __syncthreads();
        const int namb = s_namb;

        // 2c. Exact update of the ambiguous candidates, spread over all warps:
        //     work item = (candidate, slice of its unseen members).
        const int g = (namb >= WARPS_PER_BLOCK) ? 1 : (WARPS_PER_BLOCK / namb);
        for (int item = warp; item < namb * g; item += WARPS_PER_BLOCK) {
            const int a = item / g;
            const int sub = item - a * g;
            const int c = (a < AMB_CAP) ? s_amb[a] : w.amb[a];
            const float v = (a < AMB_CAP) ? s_ambv[a] : w.ambv[a];
            const int k = w.kc[c];
            const double cx = w.cx[c], cy = w.cy[c];
            double part = 0.0;
            if (k < members) {
                part = exactPartialMax(cx, cy, w.fx[c], w.fy[c], __fsub_rd(v, err2),
                                       mx, my, fmx, fmy, k + sub * 32 + lane, g * 32, members);
                part = warpMaxD(part);
            }
            if (lane == 0) {
                if (item < AMB_CAP) s_part[item] = part; else w.pmax[item] = part;
                if (sub == 0 && a < AMB_CAP) {
                    s_cm[a] = w.cmd[c]; s_ci[a] = w.cidx[c]; s_x[a] = cx; s_y[a] = cy;
                }
            }
        }
        __syncthreads();

        // 2d. Original selection criteria over the ambiguous set (warp 0).
        if (warp == 0) {
            Best b{INFINITY, INT_MAX, -1};
            double bx = 0.0, by = 0.0;
            for (int a = lane; a < namb; a += 32) {
                const bool sh = a < AMB_CAP;
                const int c = sh ? s_amb[a] : w.amb[a];
                double cm = sh ? s_cm[a] : w.cmd[c];
                for (int sub = 0; sub < g; ++sub) {
                    const int item = a * g + sub;
                    const double part = (item < AMB_CAP) ? s_part[item] : w.pmax[item];
                    cm = (cm < part) ? part : cm;  // std::max(max_dist, dist)
                }
                w.cmd[c] = cm;
                w.kc[c] = members;
                const double d = sqrt(cm);         // sqrt(max(a,b)) == max(sqrt(a), sqrt(b))
                const int ci = sh ? s_ci[a] : w.cidx[c];
                if (d < threshold && better(d, ci, b)) {
                    b.val = d; b.orig = ci; b.pos = c;
                    bx = sh ? s_x[a] : w.cx[c];
                    by = sh ? s_y[a] : w.cy[c];
                }
            }
            const int local_pos = b.pos;
            warpReduceBest<32>(b);
            b.val = __shfl_sync(0xffffffffu, b.val, 0);
            b.orig = __shfl_sync(0xffffffffu, b.orig, 0);
            b.pos = __shfl_sync(0xffffffffu, b.pos, 0);
            // Coordinates of the winner come from the lane that found it
            const unsigned owner = __ballot_sync(0xffffffffu, b.pos >= 0 && local_pos == b.pos);
            const int ow = (owner != 0u) ? (__ffs(owner) - 1) : 0;
            const double x = __shfl_sync(0xffffffffu, bx, ow);
            const double y = __shfl_sync(0xffffffffu, by, ow);
            if (lane == 0) {
                s_sel = b;
                s_namb = 0;
                if (b.pos >= 0) {
                    const float xf = static_cast<float>(__dsub_rn(x, sx));
                    const float yf = static_cast<float>(__dsub_rn(y, sy));
                    w.mpos[members] = b.pos;
                    mx[members] = x;
                    my[members] = y;
                    fmx[members] = xf;
                    fmy[members] = yf;
                    s_nx = xf;
                    s_ny = yf;
                    if constexpr (K == 0) w.mf[b.pos] = INFINITY;
                    if (out_members) out_members[members] = b.orig;
                }
            }
        }
        __syncthreads();
        const Best sel = s_sel;
        if (sel.pos < 0) break;                    // nothing fits any more
        nx = s_nx;
        ny = s_ny;
        if constexpr (K > 0) {
#pragma unroll
            for (int k = 0; k < K; ++k) {
                if (tid + k * BLOCK_THREADS == sel.pos) rmf[k] = INFINITY;
            }
        }
        ++members;
        // No extra barrier needed: s_min, the ambiguity list and s_sel are
        // each rewritten only after a barrier that follows their last read.
    }

    if (tid == 0) *out_card = members;
}

// One block per seed listed in `seeds` (grid-stride over the list).
// `bounds[seed]` is an upper bound on the working-list length. If
// `slot_off[seed] >= 0` the member list is recorded at `slots + slot_off[seed]`.
template <int K>
__global__ void __launch_bounds__(BLOCK_THREADS, (K >= 32) ? 3 : 4)
candidateClustersKernel(const int* __restrict__ seeds,
                        const int num_seeds,
                        const double* __restrict__ px,
                        const double* __restrict__ py,
                        const float* __restrict__ pxf,
                        const float* __restrict__ pyf,
                        const unsigned char* __restrict__ clustered,
                        const double threshold,
                        const Screen sc,
                        const int N,
                        unsigned char* __restrict__ gscratch,   // per-block scratch
                        const int scratch_cap,                  // entries per block
                        int* __restrict__ cardinality,          // indexed by list position
                        const long long* __restrict__ slot_off,
                        int* __restrict__ slots,
                        int* __restrict__ members_out)          // single-seed regeneration
{
    extern __shared__ float smem_members[];
    const WorkList w = carveWorkList(
        gscratch + static_cast<size_t>(blockIdx.x) * scratch_cap * BYTES_PER_CANDIDATE, scratch_cap);

    for (int s = blockIdx.x; s < num_seeds; s += gridDim.x) {
        const int seed = seeds[s];
        int* mem = members_out;
        if (slot_off && slot_off[seed] >= 0) mem = slots + slot_off[seed];
        growCluster<K>(seed, px, py, pxf, pyf, clustered, threshold, sc, N, w, smem_members,
                       cardinality + s, mem);
        __syncthreads();
    }
}

// Launches the register-resident variant matching `max_bound` (the largest
// working list of the batch) or the generic one, on the given stream.
static void launchCandidateClusters(cudaStream_t stream, int grid, int max_bound,
                                    const int* seeds, int num_seeds,
                                    const double* px, const double* py,
                                    const float* pxf, const float* pyf,
                                    const unsigned char* clustered,
                                    double threshold, Screen sc, int N,
                                    unsigned char* scratch, int scratch_cap,
                                    int* cardinality, const long long* slot_off,
                                    int* slots, int* members_out)
{
#define QT_LAUNCH(KK)                                                                   \
    candidateClustersKernel<KK><<<grid, BLOCK_THREADS, memberSmemBytes<KK>(), stream>>>(  \
        seeds, num_seeds, px, py, pxf, pyf, clustered, threshold, sc, N, scratch,         \
        scratch_cap, cardinality, slot_off, slots, members_out)
    if (max_bound <= 1 * BLOCK_THREADS) QT_LAUNCH(1);
    else if (max_bound <= 2 * BLOCK_THREADS) QT_LAUNCH(2);
    else if (max_bound <= 4 * BLOCK_THREADS) QT_LAUNCH(4);
    else if (max_bound <= 8 * BLOCK_THREADS) QT_LAUNCH(8);
    else if (max_bound <= 16 * BLOCK_THREADS) QT_LAUNCH(16);
    else if (max_bound <= 32 * BLOCK_THREADS) QT_LAUNCH(32);
    else QT_LAUNCH(0);
#undef QT_LAUNCH
    CUDA_CHECK(cudaGetLastError());
}

// Register class of a working-list bound (batches are formed per class)
static int regClass(int bound) {
    int k = 1;
    while (k <= MAX_REG_K && bound > k * BLOCK_THREADS) k *= 2;
    return (k <= MAX_REG_K) ? k : 0;
}

// For every unclustered point counts how many of `new_members` lie strictly
// within `threshold` (exactly the test used when building the working list).
// Used both for the initial neighbourhood sizes and to find the seeds whose
// neighbourhood changed after a cluster was committed.
__global__ void neighbourCountKernel(const double* __restrict__ px,
                                     const double* __restrict__ py,
                                     const float* __restrict__ pxf,
                                     const float* __restrict__ pyf,
                                     const unsigned char* __restrict__ clustered,
                                     const int* __restrict__ new_members,
                                     const int num_new,
                                     const double threshold,
                                     const Screen sc,
                                     const int N,
                                     int* __restrict__ counts)
{
    __shared__ double mx[256];
    __shared__ double my[256];
    __shared__ float mxf[256];
    __shared__ float myf[256];
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = (i < N) && !clustered[i];
    const double sx = active ? px[i] : 0.0;
    const double sy = active ? py[i] : 0.0;
    const float sxf = active ? pxf[i] : 0.0f;
    const float syf = active ? pyf[i] : 0.0f;
    const double thr2_hi = threshold * threshold * SQ_SLACK;
    int cnt = 0;
    for (int base = 0; base < num_new; base += 256) {
        const int n = min(256, num_new - base);
        if (threadIdx.x < n) {
            const int m = new_members[base + threadIdx.x];
            mx[threadIdx.x] = px[m];
            my[threadIdx.x] = py[m];
            mxf[threadIdx.x] = pxf[m];
            myf[threadIdx.x] = pyf[m];
        }
        __syncthreads();
        if (active) {
            for (int j = 0; j < n; ++j) {
                // Float prefilter (rigorous), exact test for the survivors
                if (!(dist2f(sxf, syf, mxf[j], myf[j]) < sc.pass_abs)) continue;
                const double d2 = dist2(sx, sy, mx[j], my[j]);
                if (d2 < thr2_hi && sqrt(d2) < threshold) ++cnt;
            }
        }
        __syncthreads();
    }
    if (i < N) counts[i] = active ? cnt : 0;
}

} // namespace

// Rigorous bound on |dist2f - dist2| for float coordinates with |value| <= X
// and pairs with |dx|, |dy| <= R. Float round-to-nearest has relative error
// u = 2^-24 per operation; the chain is x -> fl(x) -> fl(dx) -> fl(dx^2) ->
// fl(dx^2 + dy^2).
static double floatDist2Error(double X, double R) {
    const double u = 1.0 / 16777216.0;
    const double a = 2.0 * u * X + u * (R + 2.0 * u * X);    // |fl(dx) - dx|
    const double Rp = R + a;                                  // |fl(dx)| bound
    const double b = a * (2.0 * R + a) + u * Rp * Rp;         // |fl(dx^2) - dx^2|
    return 2.0 * (2.0 * b + 2.0 * u * Rp * Rp * (1.0 + u));   // 2x safety
}

static float roundUpF(double v) {
    float f = static_cast<float>(v);
    if (static_cast<double>(f) < v) f = std::nextafter(f, std::numeric_limits<float>::infinity());
    return f;
}

// Screening parameters for a threshold and inputs with |coordinate| <= X_in.
static Screen makeScreen(double threshold, double X_in) {
    const double u = 1.0 / 16777216.0;
    Screen sc;
    // Seed-relative float coordinates: |value| < threshold (and <= 2 X_in),
    // including the rounding of the double subtraction; pairs within 2 X.
    {
        const double X = std::min(threshold, 2.0 * X_in) * (1.0 + 4.0 * u) + 1e-300;
        sc.err = roundUpF(floatDist2Error(X, 2.0 * X));
    }
    const double t2 = threshold * threshold * SQ_SLACK;
    sc.thr2_hi = roundUpF(t2);
    // Absolute float coordinates (prefilters): a pair with exact squared
    // distance < t2 has |dx|, |dy| <= threshold <= R = 2 threshold, so its
    // float value is < t2 + E. If E < threshold^2 a pair with |dx| > R has a
    // float value far above t2 + E, so the filter is rigorous; otherwise it
    // is disabled.
    {
        const double E = floatDist2Error(X_in + 1e-300, 2.0 * threshold);
        sc.pass_abs = (E < threshold * threshold) ? roundUpF(t2 + E)
                                                  : std::numeric_limits<float>::infinity();
    }
    return sc;
}

// Number of usable GPUs (all of them are used)
static int gpuCount() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n < 1) {
        fprintf(stderr, "No CUDA device available\n");
        exit(1);
    }
    return n;
}

// Creates the CUDA contexts and forces (lazy) module loading of all kernels
static void gpuWarmup() {
    const int ndev = gpuCount();
    for (int d = 0; d < ndev; ++d) {
        CUDA_CHECK(cudaSetDevice(d));
        CUDA_CHECK(cudaFree(0));
        cudaFuncAttributes attr;
        CUDA_CHECK(cudaFuncGetAttributes(&attr, neighbourCountKernel));
#define QT_PREPARE(KK)                                                                    \
        CUDA_CHECK(cudaFuncGetAttributes(&attr, candidateClustersKernel<KK>));            \
        if (memberSmemBytes<KK>() > 0)                                                    \
            CUDA_CHECK(cudaFuncSetAttribute(candidateClustersKernel<KK>,                  \
                                            cudaFuncAttributeMaxDynamicSharedMemorySize,  \
                                            static_cast<int>(memberSmemBytes<KK>())))
        QT_PREPARE(0); QT_PREPARE(1); QT_PREPARE(2); QT_PREPARE(4); QT_PREPARE(8); QT_PREPARE(16); QT_PREPARE(32);
#undef QT_PREPARE
    }
    CUDA_CHECK(cudaSetDevice(0));
}

// Per-device state
struct DeviceState {
    int id = 0;
    cudaStream_t stream = nullptr;
    int max_grid = 0;            // resident blocks
    double *px = nullptr, *py = nullptr;
    float *pxf = nullptr, *pyf = nullptr;
    unsigned char* clustered = nullptr;
    int *seeds = nullptr, *card = nullptr, *members = nullptr;
    long long* slot_off = nullptr;
    int* slots = nullptr;
    unsigned char* scratch = nullptr;
    int scratch_cap = 0;
    // neighbour counting (device 0 only)
    int *bound = nullptr, *counts = nullptr;
    // host-side batch bookkeeping
    std::vector<int> batch;
};

// Main QT clustering algorithm (GPU accelerated, same result as sequential)
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<Cluster> clusters;
    if (N == 0) return clusters;

    double max_abs_coord = 0.0;
    for (const Point& p : points) {
        max_abs_coord = std::max(max_abs_coord, std::max(std::fabs(p.x), std::fabs(p.y)));
    }
    const Screen sc = makeScreen(threshold, max_abs_coord);

    // Host-side split of coordinates (structure of arrays for coalescing)
    std::vector<double> hx(N), hy(N);
    std::vector<float> hxf(N), hyf(N);
    for (int i = 0; i < N; ++i) {
        hx[i] = points[i].x; hy[i] = points[i].y;
        hxf[i] = static_cast<float>(hx[i]); hyf[i] = static_cast<float>(hy[i]);
    }

    // Small inputs are dominated by per-iteration latency, so extra devices
    // only pay off once there is enough work per committed cluster.
    const int ndev = std::max(1, std::min(gpuCount(), N / 4000));
    std::vector<DeviceState> devs(ndev);
    int total_grid = 0;
    for (int d = 0; d < ndev; ++d) {
        DeviceState& dv = devs[d];
        dv.id = d;
        CUDA_CHECK(cudaSetDevice(d));
        CUDA_CHECK(cudaStreamCreateWithFlags(&dv.stream, cudaStreamNonBlocking));
        int num_sms = 1, blocks_per_sm = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, d));
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks_per_sm, candidateClustersKernel<8>, BLOCK_THREADS, memberSmemBytes<8>()));
        dv.max_grid = std::max(1, num_sms * std::max(1, blocks_per_sm));
        total_grid += dv.max_grid;

        CUDA_CHECK(cudaMalloc(&dv.px, N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&dv.py, N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&dv.pxf, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dv.pyf, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dv.clustered, N));
        CUDA_CHECK(cudaMalloc(&dv.seeds, N * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&dv.card, N * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&dv.members, N * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&dv.slot_off, N * sizeof(long long)));
        CUDA_CHECK(cudaMemcpy(dv.px, hx.data(), N * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dv.py, hy.data(), N * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dv.pxf, hxf.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dv.pyf, hyf.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(dv.clustered, 0, N));
        if (d == 0) {
            CUDA_CHECK(cudaMalloc(&dv.bound, N * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&dv.counts, N * sizeof(int)));
        }
    }
    // Seeds evaluated per batch before the running best is refreshed
    const int chunk = total_grid * 4;

    std::vector<unsigned char> clustered(N, 0), exact(N, 0);
    std::vector<int> cardinality(N, 0), bound(N), counts(N), members(N), todo, owner(N, -1);
    todo.reserve(N);

    // Initial neighbourhood sizes: |{p : dist(p, s) < threshold}| (includes s).
    // The cardinality of a seed's candidate cluster can never exceed this.
    {
        DeviceState& dv = devs[0];
        CUDA_CHECK(cudaSetDevice(0));
        std::vector<int> all(N);
        for (int i = 0; i < N; ++i) all[i] = i;
        CUDA_CHECK(cudaMemcpy(dv.seeds, all.data(), N * sizeof(int), cudaMemcpyHostToDevice));
        neighbourCountKernel<<<(N + 255) / 256, 256, 0, dv.stream>>>(
            dv.px, dv.py, dv.pxf, dv.pyf, dv.clustered, dv.seeds, N, threshold, sc, N, dv.bound);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(bound.data(), dv.bound, N * sizeof(int),
                                   cudaMemcpyDeviceToHost, dv.stream));
        CUDA_CHECK(cudaStreamSynchronize(dv.stream));
    }

    // Per-block scratch sized by the largest neighbourhood (bounds only
    // shrink), rounded up to a multiple of 32 so that every block's slice
    // stays 8-byte aligned. Member-list slots: each seed that fits into the
    // memory budget gets a permanent slot of its initial neighbourhood size on
    // every device, so the winning cluster's members are usually available
    // without regeneration.
    const int scratch_cap =
        (std::max(WARPS_PER_BLOCK, *std::max_element(bound.begin(), bound.end())) + 31) / 32 * 32;
    std::vector<long long> slot_off(N, -1);
    for (int d = 0; d < ndev; ++d) {
        DeviceState& dv = devs[d];
        CUDA_CHECK(cudaSetDevice(d));
        dv.scratch_cap = scratch_cap;
        CUDA_CHECK(cudaMalloc(&dv.scratch,
                              static_cast<size_t>(dv.max_grid) * scratch_cap * BYTES_PER_CANDIDATE));
        size_t free_mem = 0, total_mem = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
        const long long budget = static_cast<long long>(free_mem / 2) / static_cast<long long>(sizeof(int));
        long long off = 0;
        std::vector<long long> dev_slot_off(N, -1);
        for (int i = 0; i < N; ++i) {
            if (off + bound[i] <= budget) { dev_slot_off[i] = off; off += bound[i]; }
        }
        if (d == 0) slot_off = dev_slot_off;
        else if (dev_slot_off != slot_off) {
            // Different memory budgets: use the common prefix of slots
            for (int i = 0; i < N; ++i) if (dev_slot_off[i] < 0) slot_off[i] = -1;
        }
    }
    for (int d = 0; d < ndev; ++d) {
        DeviceState& dv = devs[d];
        CUDA_CHECK(cudaSetDevice(d));
        long long total = 0;
        for (int i = 0; i < N; ++i) if (slot_off[i] >= 0) total = std::max(total, slot_off[i] + bound[i]);
        if (total > 0) CUDA_CHECK(cudaMalloc(&dv.slots, total * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(dv.slot_off, slot_off.data(), N * sizeof(long long),
                              cudaMemcpyHostToDevice));
    }

    int unclustered_count = N;

    // Main clustering loop: one committed cluster per iteration
    while (unclustered_count > 0) {
        // Running best among the seeds with a valid cached cardinality;
        // ties -> lowest seed index (the sequential scan visits ascending
        // indices and only replaces on a strictly larger cardinality).
        int max_cardinality = -1;
        int best_seed = -1;
        auto consider = [&](int s) {
            if (cardinality[s] > max_cardinality ||
                (cardinality[s] == max_cardinality && s < best_seed)) {
                max_cardinality = cardinality[s];
                best_seed = s;
            }
        };
        for (int i = 0; i < N; ++i) {
            if (!clustered[i] && exact[i]) consider(i);
        }

        // Seeds without a valid cardinality that could still win, largest
        // upper bound first so the running best rises quickly (this also
        // keeps batches homogeneous in register class).
        auto may_win = [&](int s) {
            return bound[s] > max_cardinality ||
                   (bound[s] == max_cardinality && s < best_seed);
        };
        todo.clear();
        for (int i = 0; i < N; ++i) {
            if (!clustered[i] && !exact[i] && may_win(i)) todo.push_back(i);
        }
        std::stable_sort(todo.begin(), todo.end(),
                         [&](int a, int b) { return bound[a] > bound[b]; });

        size_t pos = 0;
        while (pos < todo.size()) {
            // Batch of one register class, split over the devices in
            // round-robin order (the list is sorted, so the load is balanced)
            for (auto& dv : devs) dv.batch.clear();
            int max_bound = 0, nb = 0;
            const int cls = regClass(bound[todo[pos]]);
            for (; pos < todo.size() && nb < chunk; ++pos) {
                const int sd = todo[pos];
                if (regClass(bound[sd]) != cls) break;
                if (!may_win(sd)) continue;
                devs[nb % ndev].batch.push_back(sd);
                max_bound = std::max(max_bound, bound[sd]);
                ++nb;
            }
            if (nb == 0) continue;
            for (auto& dv : devs) {
                if (dv.batch.empty()) continue;
                CUDA_CHECK(cudaSetDevice(dv.id));
                const int n = static_cast<int>(dv.batch.size());
                CUDA_CHECK(cudaMemcpyAsync(dv.seeds, dv.batch.data(), n * sizeof(int),
                                           cudaMemcpyHostToDevice, dv.stream));
                launchCandidateClusters(dv.stream, std::min(n, dv.max_grid), max_bound, dv.seeds, n,
                                        dv.px, dv.py, dv.pxf, dv.pyf, dv.clustered, threshold, sc,
                                        N, dv.scratch, dv.scratch_cap, dv.card, dv.slot_off,
                                        dv.slots, nullptr);
            }
            std::vector<int> sorted;
            sorted.reserve(nb);
            for (auto& dv : devs) {
                if (dv.batch.empty()) continue;
                CUDA_CHECK(cudaSetDevice(dv.id));
                const int n = static_cast<int>(dv.batch.size());
                CUDA_CHECK(cudaMemcpyAsync(members.data(), dv.card, n * sizeof(int),
                                           cudaMemcpyDeviceToHost, dv.stream));
                CUDA_CHECK(cudaStreamSynchronize(dv.stream));
                for (int i = 0; i < n; ++i) {
                    const int sd = dv.batch[i];
                    cardinality[sd] = members[i];
                    exact[sd] = 1;
                    owner[sd] = dv.id;
                    sorted.push_back(sd);
                }
            }
            // Merge in ascending index order to keep the tie-break exact
            std::sort(sorted.begin(), sorted.end());
            for (int sd : sorted) consider(sd);
        }

        if (best_seed < 0 || max_cardinality <= 0) break;

        // Fetch the member list of the winning seed
        if (slot_off[best_seed] >= 0) {
            DeviceState& dv = devs[owner[best_seed]];
            CUDA_CHECK(cudaSetDevice(dv.id));
            CUDA_CHECK(cudaMemcpy(members.data(), dv.slots + slot_off[best_seed],
                                  max_cardinality * sizeof(int), cudaMemcpyDeviceToHost));
        } else {
            DeviceState& dv = devs[0];
            CUDA_CHECK(cudaSetDevice(0));
            CUDA_CHECK(cudaMemcpy(dv.seeds, &best_seed, sizeof(int), cudaMemcpyHostToDevice));
            launchCandidateClusters(dv.stream, 1, bound[best_seed], dv.seeds, 1, dv.px, dv.py,
                                    dv.pxf, dv.pyf, dv.clustered, threshold, sc, N, dv.scratch,
                                    dv.scratch_cap, dv.card, nullptr, nullptr, dv.members);
            CUDA_CHECK(cudaMemcpyAsync(members.data(), dv.members, max_cardinality * sizeof(int),
                                       cudaMemcpyDeviceToHost, dv.stream));
            CUDA_CHECK(cudaStreamSynchronize(dv.stream));
        }

        Cluster cluster;
        cluster.seed_point = best_seed;
        cluster.members.assign(members.begin(), members.begin() + max_cardinality);
        clusters.push_back(cluster);

        // Mark all members as clustered
        for (int m : cluster.members) clustered[m] = 1;
        unclustered_count -= max_cardinality;
        if (unclustered_count <= 0) break;
        for (auto& dv : devs) {
            CUDA_CHECK(cudaSetDevice(dv.id));
            CUDA_CHECK(cudaMemcpyAsync(dv.clustered, clustered.data(), N,
                                       cudaMemcpyHostToDevice, dv.stream));
        }

        // A seed's result depends only on the unclustered points within
        // `threshold` of it: seeds that lost neighbours are re-evaluated,
        // everything else keeps its cached cardinality.
        {
            DeviceState& dv = devs[0];
            CUDA_CHECK(cudaSetDevice(0));
            CUDA_CHECK(cudaMemcpyAsync(dv.members, members.data(), max_cardinality * sizeof(int),
                                       cudaMemcpyHostToDevice, dv.stream));
            neighbourCountKernel<<<(N + 255) / 256, 256, 0, dv.stream>>>(
                dv.px, dv.py, dv.pxf, dv.pyf, dv.clustered, dv.members, max_cardinality,
                threshold, sc, N, dv.counts);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpyAsync(counts.data(), dv.counts, N * sizeof(int),
                                       cudaMemcpyDeviceToHost, dv.stream));
            CUDA_CHECK(cudaStreamSynchronize(dv.stream));
        }
        for (int i = 0; i < N; ++i) {
            if (!clustered[i] && counts[i] > 0) {
                bound[i] -= counts[i];
                exact[i] = 0;
            }
        }
        // The other devices' clustered uploads must land before their next use
        for (auto& dv : devs) {
            if (dv.id == 0) continue;
            CUDA_CHECK(cudaSetDevice(dv.id));
            CUDA_CHECK(cudaStreamSynchronize(dv.stream));
        }
    }

    for (auto& dv : devs) {
        CUDA_CHECK(cudaSetDevice(dv.id));
        CUDA_CHECK(cudaStreamSynchronize(dv.stream));
        CUDA_CHECK(cudaFree(dv.px));
        CUDA_CHECK(cudaFree(dv.py));
        CUDA_CHECK(cudaFree(dv.pxf));
        CUDA_CHECK(cudaFree(dv.pyf));
        CUDA_CHECK(cudaFree(dv.clustered));
        CUDA_CHECK(cudaFree(dv.seeds));
        CUDA_CHECK(cudaFree(dv.card));
        CUDA_CHECK(cudaFree(dv.members));
        CUDA_CHECK(cudaFree(dv.slot_off));
        if (dv.slots) CUDA_CHECK(cudaFree(dv.slots));
        CUDA_CHECK(cudaFree(dv.scratch));
        if (dv.bound) CUDA_CHECK(cudaFree(dv.bound));
        if (dv.counts) CUDA_CHECK(cudaFree(dv.counts));
        CUDA_CHECK(cudaStreamDestroy(dv.stream));
    }
    CUDA_CHECK(cudaSetDevice(0));

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

    // Initialize the CUDA context and load the kernels up front so that
    // one-time driver work is not timed
    gpuWarmup();

    // Generate synthetic data
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);

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
