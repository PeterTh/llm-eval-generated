// QT Clustering Benchmark - CUDA Version
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <math_constants.h>
#include <cub/cub.cuh>

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
// CUDA implementation
//
// Exact reformulation of the sequential algorithm:
//  * For a candidate c, max_dist(c) over the cluster members only grows as
//    members are added, so it is maintained incrementally instead of being
//    recomputed from scratch for every step.
//  * sqrt is monotone, so max over sqrt(d2) == sqrt(max over d2). We keep the
//    max squared distance and test "max_dist < threshold" as "maxd2 <= T"
//    where T is the largest double with sqrt(T) < threshold (exact).
//  * Squared distances are tracked in float (from float coordinates
//    relative to the seed) together with the member that attains the
//    maximum. A rigorous error bound E turns the float values into exact
//    intervals; whenever an interval cannot decide a comparison (rare) the
//    exact double value is computed, evaluated exactly like the host version
//    of distance(). All decisions are therefore exact.
//  * Only points with dist(seed, c) < threshold can ever join the cluster of
//    `seed`, so each seed only works on its neighbour list.
//  * A seed's candidate cluster depends only on the unclustered points within
//    threshold of it; cardinalities are cached across rounds and only seeds
//    with a newly clustered neighbour become invalid.
//  * Invalid seeds are only evaluated if an upper bound on their cluster size
//    could beat the best valid candidate cluster.
// Ties are resolved exactly as in the sequential code (smallest max_dist,
// then lowest index for the closest point; largest cardinality, then lowest
// seed index for the best cluster).
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error '%s' at %s:%d\n",                       \
                    cudaGetErrorString(err_), __FILE__, __LINE__);              \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

static constexpr unsigned FULL_MASK = 0xffffffffu;
static constexpr int NO_POINT = 0x7fffffff;
static constexpr int SEED_WARPS = 4;  // warps (= concurrent seeds) per block

// Exact squared distance, evaluated exactly like the host build of
// distance() (dy*dy rounded, then fused multiply-add of dx*dx).
__device__ __forceinline__ double dist2(const double2 a, const double2 b) {
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    return __fma_rn(dx, dx, __dmul_rn(dy, dy));
}

__device__ __forceinline__ float dist2f(const float ax, const float ay, const float bx,
                                        const float by) {
    const float dx = ax - bx;
    const float dy = ay - by;
    return dx * dx + dy * dy;
}

// Sort key of a seed: larger cardinality first, then lower index.
__host__ __device__ __forceinline__ unsigned long long seedKey(const int card, const int idx) {
    return (static_cast<unsigned long long>(static_cast<unsigned>(card)) << 32) |
           static_cast<unsigned long long>(0xffffffffu - static_cast<unsigned>(idx));
}

__host__ __device__ __forceinline__ int keySeed(const unsigned long long key) {
    return static_cast<int>(0xffffffffu - static_cast<unsigned>(key & 0xffffffffull));
}

struct DeviceData {
    const double2* pts;
    const float2* ptsf;   // float copies of the coordinates
    const unsigned char* clustered;
    const int* nbr_off;   // CSR neighbour lists (nullptr -> scan all points)
    const int* nbr_list;
    int N;
    double T;             // largest d2 with sqrt(d2) < threshold
    float T_lo, T_hi;     // float bounds of T
    float E;              // bound on |float d2 - exact d2| for candidate pairs
    float Ef;             // same for plain float coordinates (any pair)
    int max_m;            // max neighbour count (scratch capacity per warp)
    int smem_cap;         // candidate entries kept in shared memory per warp
    float4* g_hot;        // global scratch beyond smem_cap, max_m per warp
    int* g_idx;
};

// Order of exact keys (sqrt(d2), position) on the bit patterns of the
// non-negative doubles d2 (ordered like unsigned integers). Square roots of
// values more than 8 ulps apart differ, so sqrt is only needed for near ties.
__device__ __forceinline__ bool keyLess(const unsigned long long a, const int ja,
                                        const unsigned long long b, const int jb) {
    if (a != b) {
        if (a < b ? b - a > 8ull : a - b > 8ull) return a < b;
        const double sa = sqrt(__longlong_as_double(static_cast<long long>(a)));
        const double sb = sqrt(__longlong_as_double(static_cast<long long>(b)));
        if (sa != sb) return sa < sb;
    }
    return ja < jb;
}

// Build the candidate cluster of `seed` with one warp. Returns its
// cardinality; if `out` is given, members are written in insertion order.
//
// Live candidates are kept in ascending index order (so position order
// equals index order for tie-breaking) as {x, y, v, m} plus the point index:
// float coordinates relative to the seed, the member m attaining the max
// distance and the float squared distance v to it; the exact max d2 lies in
// [v - E, v + E]. The first `smem_cap` entries live in shared memory, the
// rest in global scratch. Dead candidates are marked (v = NaN); the list is
// compacted (order preserving) once less than half of it is live.
__device__ int warpCluster(const DeviceData& D, const int seed, const size_t slot, int* out,
                           float4* sh_hot, int* sh_idx) {
    const int lane = threadIdx.x & 31;
    const unsigned lt_mask = (1u << lane) - 1u;
    const int C = D.smem_cap;
    const size_t go = slot * D.max_m;
    float4* const gH = D.g_hot ? D.g_hot + go : nullptr;
    int* const gI = D.g_idx ? D.g_idx + go : nullptr;
    auto H = [&](const int j) -> float4* { return j < C ? sh_hot + j : gH + j; };
    auto I = [&](const int j) -> int* { return j < C ? sh_idx + j : gI + j; };
    const float E = D.E;

    // Per-lane two smallest v (v1 at position j1) among owned live candidates.
    float v1 = CUDART_INF_F, v2 = CUDART_INF_F;
    int j1 = NO_POINT;
    int nlive = 0;
    auto track = [&](const float v, const int j) {
        if (v < v1) {
            v2 = v1;
            v1 = v;
            j1 = j;
        } else if (v < v2) {
            v2 = v;
        }
    };
    // Entries appended in ballot order are not at positions owned by the
    // writing lane: (re)build the tracking over the owned positions.
    auto trackOwned = [&](const int L) {
        for (int j = lane; j < L; j += 32) {
            track(H(j)->z, j);
            ++nlive;
        }
    };

    // Gather the live candidates (unclustered, within threshold of the seed).
    const double2 ps = D.pts[seed];
    const bool use_list = D.nbr_off != nullptr;
    const int begin = use_list ? D.nbr_off[seed] : 0;
    const int end = use_list ? D.nbr_off[seed + 1] : D.N;
    int L = 0;
    for (int base = begin; base < end; base += 32) {
        const int j = base + lane;
        bool keep = false;
        int c = -1;
        if (j < end) {
            c = use_list ? D.nbr_list[j] : j;
            // Neighbour lists hold exactly the points within threshold.
            keep = c != seed && !D.clustered[c] &&
                   (use_list || dist2(D.pts[c], ps) <= D.T);
        }
        const unsigned m = __ballot_sync(FULL_MASK, keep);
        if (keep) {
            const double2 pc = D.pts[c];
            const float x = static_cast<float>(pc.x - ps.x);
            const float y = static_cast<float>(pc.y - ps.y);
            const int pos = L + __popc(m & lt_mask);
            *H(pos) = make_float4(x, y, x * x + y * y, __int_as_float(seed));
            *I(pos) = c;
        }
        L += __popc(m);
    }
    __syncwarp();
    trackOwned(L);

    if (out && lane == 0) out[0] = seed;
    int count = 1;

    while (true) {
        // Float pre-selection: the exact minimum is <= hmin = min(v + E), so
        // only candidates with v - E <= hmin can be the closest point.
        // (Non-negative floats order like their bit patterns.)
        const float hmin =
            __uint_as_float(__reduce_min_sync(FULL_MASK, __float_as_uint(v1 + E)));
        const int total = __reduce_add_sync(FULL_MASK, nlive);
        if (total == 0) break;  // no point can be added
        const float cut = hmin + E;  // extra slack, keeps ties inside

        // Exact evaluation of the contenders: key (max_dist, position).
        unsigned long long best_k = ~0ull;  // bits of the exact max d2
        int best_j = NO_POINT;
        auto evalExact = [&](const int j) {
            const unsigned long long k = static_cast<unsigned long long>(__double_as_longlong(
                dist2(D.pts[*I(j)], D.pts[__float_as_int(H(j)->w)])));
            if (keyLess(k, j, best_k, best_j)) {
                best_k = k;
                best_j = j;
            }
        };
        if (v1 - E <= cut) {
            if (v2 - E <= cut) {
                for (int j = lane; j < L; j += 32)
                    if (H(j)->z - E <= cut) evalExact(j);  // false for NaN
            } else {
                evalExact(j1);
            }
        }
        const unsigned has = __ballot_sync(FULL_MASK, best_j != NO_POINT);
        if ((has & (has - 1)) == 0) {
            best_j = __shfl_sync(FULL_MASK, best_j, __ffs(has) - 1);  // single contender lane
        } else {
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                const unsigned long long ok = __shfl_xor_sync(FULL_MASK, best_k, off);
                const int oj = __shfl_xor_sync(FULL_MASK, best_j, off);
                if (keyLess(ok, oj, best_k, best_j)) {
                    best_k = ok;
                    best_j = oj;
                }
            }
        }

        const int pj = best_j;
        const float4 ph = *H(pj);
        const int p = *I(pj);
        if (out && lane == 0) out[count] = p;
        ++count;

        const bool compact = 2 * total < L && L > 64;
        v1 = v2 = CUDART_INF_F;
        j1 = NO_POINT;
        nlive = 0;

        // Add p to candidate j with entry h: update its max distance and
        // member. The point index is only read when the float intervals
        // cannot decide. Returns false if the candidate died.
        auto update = [&](const int j, float4& h, bool& changed) -> bool {
            const float dx = h.x - ph.x;
            const float dy = h.y - ph.y;
            const float d2f = dx * dx + dy * dy;
            if (d2f + E < h.z - E) {
                changed = false;
            } else if (d2f - E > h.z + E) {
                changed = true;
            } else {
                const double2 pc = D.pts[*I(j)];
                changed = dist2(pc, D.pts[p]) > dist2(pc, D.pts[__float_as_int(h.w)]);
            }
            if (!changed) return true;  // max unchanged: still alive
            h.z = d2f;
            h.w = __int_as_float(p);
            if (d2f - E > D.T_hi) return false;
            if (d2f + E <= D.T_lo) return true;
            return dist2(D.pts[*I(j)], D.pts[p]) <= D.T;
        };

        if (!compact) {
            // Update in place; every lane owns positions lane + 32k.
            for (int j = lane; j < L; j += 32) {
                float4 h = *H(j);
                if (h.z != h.z) continue;  // dead
                float* hz = reinterpret_cast<float*>(H(j)) + 2;
                if (j == pj) {
                    *hz = CUDART_NAN_F;
                    continue;
                }
                bool changed;
                if (!update(j, h, changed)) {
                    *hz = CUDART_NAN_F;
                    continue;
                }
                if (changed) *H(j) = h;
                track(h.z, j);
                ++nlive;
            }
        } else {
            // Update and compact (order preserving, in place).
            int newL = 0;
            for (int base = 0; base < L; base += 32) {
                const int j = base + lane;
                bool keep = false;
                float4 h = make_float4(0.f, 0.f, 0.f, 0.f);
                int c = -1;
                if (j < L) {
                    h = *H(j);
                    if (h.z == h.z && j != pj) {
                        bool changed;
                        keep = update(j, h, changed);
                        if (keep) c = *I(j);
                    }
                }
                const unsigned bm = __ballot_sync(FULL_MASK, keep);
                if (keep) {
                    const int pos = newL + __popc(bm & lt_mask);
                    *H(pos) = h;
                    *I(pos) = c;
                }
                newL += __popc(bm);
            }
            L = newL;
            __syncwarp();
            trackOwned(L);
        }
        __syncwarp();
    }
    return count;
}

// Exact "dist2(i, j) <= T", decided in float when the error bound allows.
// Pairs farther than 2*thr in a coordinate have d2f far above T, so the
// float rejection test is valid for all pairs.
__device__ __forceinline__ bool withinThreshold(const double2* __restrict__ pts,
                                                const float2* __restrict__ ptsf,
                                                const int i, const float2 pif,
                                                const double2 pi, const int j,
                                                const double T, const float T_lo,
                                                const float T_hi, const float Ef) {
    const float2 pjf = ptsf[j];
    const float d2f = dist2f(pjf.x, pjf.y, pif.x, pif.y);
    if (d2f - Ef > T_hi) return false;
    if (d2f + Ef <= T_lo) return j != i;
    return j != i && dist2(pts[j], pi) <= T;
}

// Neighbours (other points within threshold) of each point, one warp per
// point. Counts only if `list` is null, else fills the CSR list in
// ascending index order.
__global__ void neighborsKernel(const double2* __restrict__ pts,
                                const float2* __restrict__ ptsf, const int N, const double T,
                                const float T_lo, const float T_hi, const float Ef,
                                int* __restrict__ counts, const int* __restrict__ off,
                                int* __restrict__ list) {
    const int i = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (i >= N) return;
    const double2 pi = pts[i];
    const float2 pif = ptsf[i];
    int w = list ? off[i] : 0;
    for (int base = 0; base < N; base += 32) {
        const int j = base + lane;
        const bool in = j < N && withinThreshold(pts, ptsf, i, pif, pi, j, T, T_lo, T_hi, Ef);
        const unsigned m = __ballot_sync(FULL_MASK, in);
        if (list && in) list[w + __popc(m & ((1u << lane) - 1u))] = j;
        w += __popc(m);
    }
    if (!list && lane == 0) counts[i] = w;
}

// Clustering state shared by the per-round kernels (all rounds run on the
// device without host synchronisation; kernels do nothing once done).
struct RoundState {
    unsigned long long best;  // key (cardinality, seed) of the best cluster
    int offset;               // number of clustered points
    int round;                // number of clusters formed
    int ncand;                // candidates of the current round
    int work;                 // work counter of the candidate evaluation
    unsigned int blocks_done; // completion counter of the cluster kernel
};

// Best seed among those whose cached cardinality is valid.
__global__ void selectBestValidKernel(const unsigned char* __restrict__ clustered,
                                      const unsigned char* __restrict__ valid,
                                      const int* __restrict__ card, const int N,
                                      RoundState* __restrict__ st) {
    if (st->offset >= N) return;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N && !clustered[i] && valid[i]) atomicMax(&st->best, seedKey(card[i], i));
}

// Sort keys of all seeds: the upper bound for seeds without a valid
// cardinality that could still beat the best known cluster, else 0.
__global__ void candidateKeysKernel(const unsigned char* __restrict__ clustered,
                                    const unsigned char* __restrict__ valid,
                                    const int* __restrict__ ub, const int N,
                                    RoundState* __restrict__ st, int* __restrict__ keys,
                                    int* __restrict__ vals) {
    if (st->offset >= N) return;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int key = 0;
    if (!clustered[i] && !valid[i]) {
        const int u = ub[i];
        if (seedKey(u, i) > st->best) key = u;
    }
    keys[i] = key;
    vals[i] = i;
    const unsigned m = __ballot_sync(__activemask(), key != 0);
    if ((threadIdx.x & 31) == __ffs(__activemask()) - 1 && m) atomicAdd(&st->ncand, __popc(m));
}

// Small problems: both of the above in one block, appending the candidates
// in arbitrary order (no sorting needed, nearly all run concurrently).
__global__ void __launch_bounds__(1024)
roundStartSmallKernel(const unsigned char* __restrict__ clustered,
                      const unsigned char* __restrict__ valid, const int* __restrict__ card,
                      const int* __restrict__ ub, const int N, RoundState* __restrict__ st,
                      int* __restrict__ seeds) {
    __shared__ unsigned long long s_best;
    __shared__ int s_n;
    if (st->offset >= N) return;
    if (threadIdx.x == 0) {
        s_best = 0ull;
        s_n = 0;
    }
    __syncthreads();
    unsigned long long b = 0ull;
    for (int i = threadIdx.x; i < N; i += blockDim.x)
        if (!clustered[i] && valid[i]) b = max(b, seedKey(card[i], i));
    atomicMax(&s_best, b);
    __syncthreads();
    b = s_best;
    for (int i = threadIdx.x; i < N; i += blockDim.x)
        if (!clustered[i] && !valid[i] && seedKey(ub[i], i) > b) seeds[atomicAdd(&s_n, 1)] = i;
    __syncthreads();
    if (threadIdx.x == 0) {
        st->best = b;
        st->ncand = s_n;
    }
}

// Exact candidate cluster cardinalities for the candidate seeds (one warp
// per seed, sorted by decreasing upper bound). Seeds whose upper bound
// cannot beat the best cluster found so far are skipped (stay invalid).
__global__ void __launch_bounds__(SEED_WARPS * 32)
computeCardinalityKernel(const DeviceData D, const int* __restrict__ seeds,
                         RoundState* __restrict__ st, int* __restrict__ card,
                         const int* __restrict__ ub, unsigned char* __restrict__ valid,
                         int* __restrict__ member_store) {
    if (st->offset >= D.N) return;
    extern __shared__ float4 dyn_smem[];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const size_t cap = D.smem_cap;
    float4* sh_hot = dyn_smem + warp * cap;
    int* sh_idx = reinterpret_cast<int*>(dyn_smem + SEED_WARPS * cap) + warp * cap;
    const size_t slot = static_cast<size_t>(blockIdx.x) * SEED_WARPS + warp;
    const int n = st->ncand;
    volatile unsigned long long* best = &st->best;
    while (true) {
        int i = 0, skip = 0;
        if (lane == 0) {
            i = atomicAdd(&st->work, 1);
            if (i < n) {
                const int s = seeds[i];
                skip = seedKey(ub[s], s) < *best;
            }
        }
        i = __shfl_sync(FULL_MASK, i, 0);
        skip = __shfl_sync(FULL_MASK, skip, 0);
        if (i >= n) break;
        if (skip) continue;
        const int s = seeds[i];
        int* out = member_store ? member_store + D.nbr_off[s] + s : nullptr;
        const int c = warpCluster(D, s, slot, out, sh_hot, sh_idx);
        if (lane == 0) {
            card[s] = c;
            valid[s] = 1;
            atomicMax(&st->best, seedKey(c, s));
        }
    }
}

// Record the cluster of this round and reset the per-round state.
__device__ __forceinline__ void finishRound(RoundState* st,
                                            unsigned long long* cluster_keys) {
    const unsigned long long key = st->best;
    cluster_keys[st->round] = key;
    st->round += 1;
    st->offset += static_cast<int>(key >> 32);
    st->best = 0ull;
    st->ncand = 0;
    st->work = 0;
}

// Copy the stored members of the winning cluster (one warp per member),
// mark them clustered and invalidate the cached results of their
// unclustered neighbours. (A neighbour clustered in this same round may be
// invalidated too, which is harmless.) The last block finishes the round.
__global__ void formClusterKernel(const int N, const int* __restrict__ nbr_off,
                                  const int* __restrict__ nbr_list,
                                  const int* __restrict__ member_store,
                                  RoundState* __restrict__ st,
                                  unsigned char* __restrict__ clustered,
                                  int* __restrict__ members, unsigned char* __restrict__ valid,
                                  unsigned char* __restrict__ dirty,
                                  unsigned long long* __restrict__ cluster_keys) {
    if (st->offset >= N) return;
    const unsigned long long key = st->best;
    const int seed = keySeed(key);
    const int cnt = static_cast<int>(key >> 32);
    const int* src = member_store + nbr_off[seed] + seed;
    int* out = members + st->offset;
    const int lane = threadIdx.x & 31;
    const int nwarps = (gridDim.x * blockDim.x) >> 5;
    for (int k = (blockIdx.x * blockDim.x + threadIdx.x) >> 5; k < cnt; k += nwarps) {
        const int m = src[k];
        if (lane == 0) {
            out[k] = m;
            clustered[m] = 1;
        }
        for (int j = nbr_off[m] + lane; j < nbr_off[m + 1]; j += 32) {
            const int s = nbr_list[j];
            if (!clustered[s]) {
                valid[s] = 0;
                dirty[s] = 1;
            }
        }
    }
    // Last block to finish records the round.
    __shared__ bool s_last;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0)
        s_last = atomicAdd(&st->blocks_done, 1u) == gridDim.x - 1;
    __syncthreads();
    if (s_last && threadIdx.x == 0) {
        __threadfence();
        st->blocks_done = 0u;
        finishRound(st, cluster_keys);
    }
}

// Without member storage: rebuild the winning cluster (one warp), record
// its members and mark them clustered.
__global__ void recordClusterKernel(const DeviceData D, const RoundState* __restrict__ st,
                                    unsigned char* __restrict__ clustered,
                                    int* __restrict__ members) {
    if (st->offset >= D.N) return;
    extern __shared__ float4 dyn_smem[];
    int* sh_idx = reinterpret_cast<int*>(dyn_smem + D.smem_cap);
    int* out = members + st->offset;
    const int cnt = warpCluster(D, keySeed(st->best), 0, out, dyn_smem, sh_idx);
    __syncwarp();
    for (int k = threadIdx.x; k < cnt; k += 32) clustered[out[k]] = 1;
}

// Fallback without neighbour lists: scan the new members for every seed.
__global__ void markDirtyScanKernel(const double2* __restrict__ pts,
                                    const unsigned char* __restrict__ clustered, const int N,
                                    const double T, const int* __restrict__ members,
                                    const RoundState* __restrict__ st,
                                    unsigned char* __restrict__ valid) {
    if (st->offset >= N) return;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N || clustered[i] || !valid[i]) return;
    const int cnt = static_cast<int>(st->best >> 32);
    const int* new_members = members + st->offset;
    const double2 pi = pts[i];
    for (int k = 0; k < cnt; ++k) {
        if (dist2(pi, pts[new_members[k]]) <= T) {
            valid[i] = 0;
            return;
        }
    }
}

// Upper bound grid: a candidate cluster has diameter < threshold, so in any
// rotated frame it fits into a square of side threshold containing the seed.
// Live neighbours are binned into cells of side threshold/UB_G around the
// seed; any such square is covered by UB_B x UB_B cells (one cell of slack
// for rounding), so the largest such cell block count bounds the cluster.
static constexpr int UB_G = 14;
static constexpr int UB_B = UB_G + 2;
static constexpr int UB_NC = 2 * UB_B;     // cells per dimension
static constexpr int UB_OFF = UB_B;        // cell of the seed
static constexpr int UB_WARPS = 4;         // warps per block

__device__ __forceinline__ int ubCell(const double t) {
    const double f = floor(t) + UB_OFF;
    return f < 0.0 ? 0 : (f > UB_NC - 1 ? UB_NC - 1 : static_cast<int>(f));
}

// Inclusive 2D prefix sums of a UB_NC x UB_NC histogram, then the maximum
// count over cell blocks of UB_B x UB_B cells containing the seed cell.
__device__ int ubMaxBlock(int* h, const int lane) {
    __syncwarp();
    for (int r = lane; r < UB_NC; r += 32)
        for (int c = 1; c < UB_NC; ++c) h[r * UB_NC + c] += h[r * UB_NC + c - 1];
    __syncwarp();
    for (int c = lane; c < UB_NC; c += 32)
        for (int r = 1; r < UB_NC; ++r) h[r * UB_NC + c] += h[(r - 1) * UB_NC + c];
    __syncwarp();
    int best = 0;
    for (int k = lane; k < UB_B * UB_B; k += 32) {
        const int r0 = UB_OFF - UB_B + 1 + k / UB_B;  // >= 1
        const int c0 = UB_OFF - UB_B + 1 + k % UB_B;
        const int r1 = r0 + UB_B - 1;                 // <= UB_NC - 1
        const int c1 = c0 + UB_B - 1;
        const int v = h[r1 * UB_NC + c1] - h[(r0 - 1) * UB_NC + c1] -
                      h[r1 * UB_NC + c0 - 1] + h[(r0 - 1) * UB_NC + c0 - 1];
        best = max(best, v);
    }
    for (int off = 16; off > 0; off >>= 1) best = max(best, __shfl_xor_sync(FULL_MASK, best, off));
    return best;
}

// Recompute the cardinality upper bound of dirty seeds (one warp per seed):
// 1 + live neighbours, tightened by the grid bound when enabled.
__global__ void __launch_bounds__(UB_WARPS * 32)
updateUpperBoundKernel(const int* __restrict__ nbr_off, const int* __restrict__ nbr_list,
                       const double2* __restrict__ pts,
                       const unsigned char* __restrict__ clustered, const int N,
                       const double inv_cell, const bool use_grid,
                       const RoundState* __restrict__ st,
                       unsigned char* __restrict__ dirty, int* __restrict__ ub) {
    __shared__ int s_hist[UB_WARPS][2][UB_NC * UB_NC];
    if (st && st->offset >= N) return;
    const int i = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (i >= N || !dirty[i]) return;
    int* h0 = s_hist[threadIdx.x >> 5][0];
    int* h1 = s_hist[threadIdx.x >> 5][1];
    if (use_grid)
        for (int k = lane; k < UB_NC * UB_NC; k += 32) h0[k] = h1[k] = 0;
    __syncwarp();
    const double2 ps = pts[i];
    const double rs = 0.70710678118654752440 * inv_cell;
    int cnt = 0;
    for (int j = nbr_off[i] + lane; j < nbr_off[i + 1]; j += 32) {
        const int c = nbr_list[j];
        if (clustered[c]) continue;
        ++cnt;
        if (use_grid) {
            const double2 pc = pts[c];
            const double dx = pc.x - ps.x;
            const double dy = pc.y - ps.y;
            atomicAdd(&h0[ubCell(dy * inv_cell) * UB_NC + ubCell(dx * inv_cell)], 1);
            atomicAdd(&h1[ubCell((dy - dx) * rs) * UB_NC + ubCell((dx + dy) * rs)], 1);
        }
    }
    for (int off = 16; off > 0; off >>= 1) cnt += __shfl_xor_sync(FULL_MASK, cnt, off);
    if (use_grid) cnt = min(cnt, min(ubMaxBlock(h0, lane), ubMaxBlock(h1, lane)));
    if (lane == 0) {
        ub[i] = cnt + 1;
        dirty[i] = 0;
    }
}

__global__ void finishRoundKernel(const int N, RoundState* __restrict__ st,
                                  unsigned long long* __restrict__ cluster_keys) {
    if (st->offset >= N) return;
    finishRound(st, cluster_keys);
}

// Largest double T with sqrt(T) < threshold (sqrt is correctly rounded on
// both host and device, so this is exact).
static double squaredThresholdBound(const double threshold) {
    double t = threshold * threshold;
    while (t > 0.0 && !(std::sqrt(t) < threshold)) t = std::nextafter(t, 0.0);
    while (true) {
        const double nt = std::nextafter(t, std::numeric_limits<double>::infinity());
        if (nt == t || !(std::sqrt(nt) < threshold)) break;
        t = nt;
    }
    return t;
}

// Main QT clustering algorithm
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<Cluster> clusters;
    if (N == 0) return clusters;

    const double T = squaredThresholdBound(threshold);
    const int TPB = 256;
    const int nblk = (N + TPB - 1) / TPB;
    const int nblk_warp = (N + TPB / 32 - 1) / (TPB / 32);  // one warp per point

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    // Points (double for exact arithmetic, float for the fast paths)
    std::vector<double2> hpts(N);
    std::vector<float2> hptsf(N);
    double max_coord = 0.0;
    for (int i = 0; i < N; ++i) {
        hpts[i] = make_double2(points[i].x, points[i].y);
        hptsf[i] = make_float2(static_cast<float>(points[i].x), static_cast<float>(points[i].y));
        max_coord = std::max({max_coord, std::fabs(points[i].x), std::fabs(points[i].y)});
    }
    double2* d_pts;
    float2* d_ptsf;
    CUDA_CHECK(cudaMalloc(&d_pts, sizeof(double2) * N));
    CUDA_CHECK(cudaMalloc(&d_ptsf, sizeof(float2) * N));
    CUDA_CHECK(cudaMemcpyAsync(d_pts, hpts.data(), sizeof(double2) * N, cudaMemcpyHostToDevice,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(d_ptsf, hptsf.data(), sizeof(float2) * N, cudaMemcpyHostToDevice,
                               stream));

    // Error bounds (inflated) of float squared distances between points
    // within threshold of a common seed, so |dx|, |dy| < dmax = 2*thr:
    //  * Ef: from the float coordinates,
    //  * E: from float coordinates relative to the seed (|x - sx| < thr).
    // If a bound is not finite the exact path is always taken.
    const double thr_s = std::sqrt(T);
    float Ef_bound = std::numeric_limits<float>::infinity();
    float E_bound = std::numeric_limits<float>::infinity();
    {
        const double u = std::ldexp(1.0, -24);
        const double dmax = 2.0 * thr_s;
        auto bound = [&](const double delta) {
            const double e = 2.0 * delta * (2.0 * dmax + delta) +
                             8.0 * u * (dmax + delta) * (dmax + delta);
            return 2.0 * e + 1e-30;
        };
        // |dxf - dx|: rounding of both coordinates and of the difference
        const double ef = bound((2.0 * max_coord + dmax) * u * 1.01);
        const double er = bound((2.0 * thr_s + dmax) * u * 1.01);
        if (std::isfinite(ef) && ef < 1e30) Ef_bound = static_cast<float>(ef);
        if (std::isfinite(er) && er < 1e30) E_bound = static_cast<float>(er);
    }
    float T_lo = static_cast<float>(T), T_hi = T_lo;
    if (static_cast<double>(T_lo) > T) T_lo = std::nextafter(T_lo, 0.0f);
    if (static_cast<double>(T_hi) < T)
        T_hi = std::nextafter(T_hi, std::numeric_limits<float>::infinity());

    // Neighbour counts
    int* d_counts;
    CUDA_CHECK(cudaMalloc(&d_counts, sizeof(int) * N));
    neighborsKernel<<<nblk_warp, TPB, 0, stream>>>(d_pts, d_ptsf, N, T, T_lo, T_hi, Ef_bound,
                                                   d_counts, nullptr, nullptr);
    CUDA_CHECK(cudaGetLastError());
    std::vector<int> hcounts(N);
    CUDA_CHECK(cudaMemcpyAsync(hcounts.data(), d_counts, sizeof(int) * N,
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const int max_m = std::max(1, *std::max_element(hcounts.begin(), hcounts.end()));

    size_t free_mem = 0, total_mem = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));

    // CSR neighbour lists if they fit comfortably, else scan all points.
    // Member storage per seed uses the same layout (offset nbr_off[s] + s).
    long long total_nbrs = 0;
    for (int i = 0; i < N; ++i) total_nbrs += hcounts[i];
    int *d_off = nullptr, *d_list = nullptr, *d_store = nullptr;
    std::vector<int> hoff(N + 1, 0);
    if (total_nbrs + N < std::numeric_limits<int>::max() &&
        static_cast<size_t>(2 * total_nbrs + N) * sizeof(int) < free_mem / 3) {
        for (int i = 0; i < N; ++i) hoff[i + 1] = hoff[i] + hcounts[i];
        CUDA_CHECK(cudaMalloc(&d_off, sizeof(int) * (N + 1)));
        CUDA_CHECK(cudaMalloc(&d_list, sizeof(int) * std::max<long long>(total_nbrs, 1)));
        CUDA_CHECK(cudaMemcpyAsync(d_off, hoff.data(), sizeof(int) * (N + 1),
                                   cudaMemcpyHostToDevice, stream));
        neighborsKernel<<<nblk_warp, TPB, 0, stream>>>(d_pts, d_ptsf, N, T, T_lo, T_hi,
                                                       Ef_bound, nullptr, d_off, d_list);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMalloc(&d_store, sizeof(int) * (total_nbrs + N)));
    }

    // Candidate entries per warp kept in shared memory (rest in global).
    const size_t entry_bytes = sizeof(float4) + sizeof(int);
    const int smem_cap = std::min(max_m, 256);
    const size_t smem = SEED_WARPS * smem_cap * entry_bytes;
    CUDA_CHECK(cudaFuncSetAttribute(computeCardinalityKernel,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    static_cast<int>(smem)));
    int per_sm = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &per_sm, computeCardinalityKernel, SEED_WARPS * 32, smem));
    long long cgrid = static_cast<long long>(std::max(per_sm, 1)) * prop.multiProcessorCount;
    cgrid = std::min<long long>(cgrid, (N + SEED_WARPS - 1) / SEED_WARPS);

    // Global scratch per warp for candidates beyond the shared memory part
    const bool overflow = smem_cap < max_m;
    float4* d_ghot = nullptr;
    int* d_gidx = nullptr;
    if (overflow) {
        CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
        const long long mem_blocks = static_cast<long long>(
            (free_mem / 2) / (entry_bytes * max_m * SEED_WARPS));
        cgrid = std::max<long long>(1, std::min(cgrid, mem_blocks));
        const size_t nslots = static_cast<size_t>(cgrid) * SEED_WARPS * max_m;
        CUDA_CHECK(cudaMalloc(&d_ghot, sizeof(float4) * nslots));
        CUDA_CHECK(cudaMalloc(&d_gidx, sizeof(int) * nslots));
    }

    // State
    unsigned char *d_clustered, *d_valid, *d_dirty;
    int *d_card, *d_ub, *d_members;
    int *d_keys, *d_vals, *d_keys_sorted, *d_vals_sorted;
    unsigned long long* d_cluster_keys;
    RoundState* d_state;
    CUDA_CHECK(cudaMalloc(&d_clustered, N));
    CUDA_CHECK(cudaMalloc(&d_valid, N));
    CUDA_CHECK(cudaMalloc(&d_dirty, N));
    CUDA_CHECK(cudaMalloc(&d_card, sizeof(int) * N));
    CUDA_CHECK(cudaMalloc(&d_ub, sizeof(int) * N));
    CUDA_CHECK(cudaMalloc(&d_members, sizeof(int) * N));
    CUDA_CHECK(cudaMalloc(&d_keys, sizeof(int) * N));
    CUDA_CHECK(cudaMalloc(&d_vals, sizeof(int) * N));
    CUDA_CHECK(cudaMalloc(&d_keys_sorted, sizeof(int) * N));
    CUDA_CHECK(cudaMalloc(&d_vals_sorted, sizeof(int) * N));
    CUDA_CHECK(cudaMalloc(&d_cluster_keys, sizeof(unsigned long long) * N));
    CUDA_CHECK(cudaMalloc(&d_state, sizeof(RoundState)));
    CUDA_CHECK(cudaMemsetAsync(d_clustered, 0, N, stream));
    CUDA_CHECK(cudaMemsetAsync(d_valid, 0, N, stream));
    CUDA_CHECK(cudaMemsetAsync(d_state, 0, sizeof(RoundState), stream));

    // Upper bound on each seed's cluster size: 1 + neighbour count, tightened
    // by the grid bound (needs neighbour lists, and cells that are large
    // compared to the coordinate rounding errors).
    for (int i = 0; i < N; ++i) hcounts[i] += 1;
    CUDA_CHECK(cudaMemcpyAsync(d_ub, hcounts.data(), sizeof(int) * N, cudaMemcpyHostToDevice,
                               stream));
    const double cell = std::sqrt(T) / UB_G;
    const bool use_grid = std::isfinite(cell) && cell > 1e-9 * std::max(1.0, max_coord);
    const double inv_cell = use_grid ? 1.0 / cell : 0.0;
    const int ub_blocks = (N + UB_WARPS - 1) / UB_WARPS;
    CUDA_CHECK(cudaMemsetAsync(d_dirty, d_off ? 1 : 0, N, stream));
    if (d_off) {
        updateUpperBoundKernel<<<ub_blocks, UB_WARPS * 32, 0, stream>>>(
            d_off, d_list, d_pts, d_clustered, N, inv_cell, use_grid, nullptr, d_dirty, d_ub);
        CUDA_CHECK(cudaGetLastError());
    }

    size_t sort_bytes = 0;
    CUDA_CHECK(cub::DeviceRadixSort::SortPairsDescending(
        nullptr, sort_bytes, d_keys, d_keys_sorted, d_vals, d_vals_sorted, N));
    void* d_sort_tmp;
    CUDA_CHECK(cudaMalloc(&d_sort_tmp, std::max<size_t>(sort_bytes, 1)));
    int key_bits = 1;
    while (key_bits < 32 && (1ll << key_bits) <= max_m + 1) ++key_bits;

    DeviceData D;
    D.pts = d_pts;
    D.ptsf = d_ptsf;
    D.clustered = d_clustered;
    D.nbr_off = d_off;
    D.nbr_list = d_list;
    D.N = N;
    D.T = T;
    D.T_lo = T_lo;
    D.T_hi = T_hi;
    D.E = E_bound;
    D.max_m = max_m;
    D.smem_cap = smem_cap;
    D.g_hot = d_ghot;
    D.g_idx = d_gidx;

    // One clustering round, captured as a CUDA graph.
    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    {
        // Best exact candidate cluster among the cached cardinalities, then
        // the seeds that might beat it (largest upper bound first).
        const int* seeds = d_vals;
        if (N <= 2 * cgrid * SEED_WARPS) {
            roundStartSmallKernel<<<1, 1024, 0, stream>>>(d_clustered, d_valid, d_card, d_ub, N,
                                                          d_state, d_vals);
        } else {
            selectBestValidKernel<<<nblk, TPB, 0, stream>>>(d_clustered, d_valid, d_card, N,
                                                            d_state);
            candidateKeysKernel<<<nblk, TPB, 0, stream>>>(d_clustered, d_valid, d_ub, N,
                                                          d_state, d_keys, d_vals);
            CUDA_CHECK(cub::DeviceRadixSort::SortPairsDescending(
                d_sort_tmp, sort_bytes, d_keys, d_keys_sorted, d_vals, d_vals_sorted, N, 0,
                key_bits, stream));
            seeds = d_vals_sorted;
        }
        computeCardinalityKernel<<<static_cast<int>(cgrid), SEED_WARPS * 32, smem, stream>>>(
            D, seeds, d_state, d_card, d_ub, d_valid, d_store);
        // Form the cluster, invalidate cached results near it, finish the
        // round, and refresh the upper bounds of the invalidated seeds.
        if (d_store) {
            formClusterKernel<<<std::max(1, std::min(nblk_warp, 2048)), TPB, 0, stream>>>(
                N, d_off, d_list, d_store, d_state, d_clustered, d_members, d_valid, d_dirty,
                d_cluster_keys);
            updateUpperBoundKernel<<<ub_blocks, UB_WARPS * 32, 0, stream>>>(
                d_off, d_list, d_pts, d_clustered, N, inv_cell, use_grid, nullptr, d_dirty,
                d_ub);
        } else {
            recordClusterKernel<<<1, 32, smem_cap * entry_bytes, stream>>>(D, d_state,
                                                                          d_clustered, d_members);
            markDirtyScanKernel<<<nblk, TPB, 0, stream>>>(d_pts, d_clustered, N, T, d_members,
                                                          d_state, d_valid);
            finishRoundKernel<<<1, 1, 0, stream>>>(N, d_state, d_cluster_keys);
        }
    }
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, 0));

    // Run rounds in batches until all points are clustered.
    RoundState hstate{};
    const int batch = 16;
    while (hstate.offset < N) {
        for (int r = 0; r < batch; ++r) CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
        CUDA_CHECK(cudaMemcpyAsync(&hstate, d_state, sizeof(RoundState), cudaMemcpyDeviceToHost,
                                   stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    const int nclusters = hstate.round;
    std::vector<unsigned long long> hkeys(nclusters);
    std::vector<int> hmembers(N);
    CUDA_CHECK(cudaMemcpyAsync(hkeys.data(), d_cluster_keys,
                               sizeof(unsigned long long) * nclusters, cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(hmembers.data(), d_members, sizeof(int) * N,
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    int pos = 0;
    for (int c = 0; c < nclusters; ++c) {
        const int card = static_cast<int>(hkeys[c] >> 32);
        Cluster cluster;
        cluster.seed_point = keySeed(hkeys[c]);
        cluster.members.assign(hmembers.begin() + pos, hmembers.begin() + pos + card);
        pos += card;
        clusters.push_back(std::move(cluster));
    }

    CUDA_CHECK(cudaGraphExecDestroy(graph_exec));
    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaStreamDestroy(stream));
    void* bufs[] = {d_pts, d_ptsf, d_counts, d_off, d_list, d_store, d_ghot, d_gidx,
                    d_clustered, d_valid, d_dirty, d_card, d_ub, d_members, d_keys, d_vals,
                    d_keys_sorted, d_vals_sorted, d_cluster_keys, d_state, d_sort_tmp};
    for (void* b : bufs) cudaFree(b);

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
    
    // Initialize the CUDA context (loading all kernels eagerly) outside of
    // the timed region
    setenv("CUDA_MODULE_LOADING", "EAGER", 0);
    CUDA_CHECK(cudaFree(nullptr));

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
