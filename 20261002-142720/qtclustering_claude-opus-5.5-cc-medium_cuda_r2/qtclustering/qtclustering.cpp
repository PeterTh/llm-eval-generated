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
// Exactness notes:
//  * Squared distance is computed as fma(dx, dx, dy*dy), matching the
//    reference host build (GCC contracts dx*dx + dy*dy into an FMA).
//  * sqrt is monotone, so max over sqrt(d2) == sqrt(max d2); the running
//    "max distance to cluster" is kept in squared form.
//  * dist < threshold  <=>  d2 < S, where S is the smallest double with
//    sqrt(S) >= threshold (computed exactly on the host).
//  * argmin ties are resolved exactly on sqrt values with lowest index, as
//    in the sequential scan.
//  * A candidate cluster grown from a seed only depends on the clustered
//    state of the seed's neighbours (points within threshold), so cached
//    cardinalities are recomputed only for seeds that lost a neighbour.
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err_ = (call);                                            \
        if (err_ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                       \
                    cudaGetErrorString(err_), __FILE__, __LINE__);            \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

static const unsigned FULL_MASK = 0xffffffffu;
static const int NB_ROWS = 8;     // rows (warps) per block
static const int NB_TILE = 256;   // points per shared-memory tile

__device__ __forceinline__ double dist2(double ax, double ay, double bx, double by) {
    const double dx = __dsub_rn(ax, bx);
    const double dy = __dsub_rn(ay, by);
    return __fma_rn(dx, dx, __dmul_rn(dy, dy));
}

// Count (FILL=false) or fill (FILL=true) neighbour lists in ascending index
// order. One warp per row, rows of a block share shared-memory point tiles.
template <bool FILL>
__global__ void __launch_bounds__(32 * NB_ROWS)
neighborKernel(const double2* __restrict__ pts, const int N, const double S,
               int* __restrict__ counts, const long long* __restrict__ offs,
               int* __restrict__ nb) {
    __shared__ double2 tile[NB_TILE];
    const int lane = threadIdx.x & 31;
    const unsigned ltmask = (1u << lane) - 1u;
    const int i = blockIdx.x * NB_ROWS + (threadIdx.x >> 5);
    const bool rowValid = i < N;
    const double2 p = rowValid ? pts[i] : make_double2(0.0, 0.0);
    long long base = 0;
    if (FILL && rowValid) base = offs[i];
    int c = 0;
    for (int t = 0; t < N; t += NB_TILE) {
        for (int k = threadIdx.x; k < NB_TILE && t + k < N; k += blockDim.x) tile[k] = pts[t + k];
        __syncthreads();
        if (rowValid) {
            const int lim = min(NB_TILE, N - t);
            for (int k0 = 0; k0 < lim; k0 += 32) {
                const int k = k0 + lane;
                const int j = t + k;
                bool ok = false;
                if (k < lim && j != i) {
                    const double2 q = tile[k];
                    ok = dist2(p.x, p.y, q.x, q.y) < S;
                }
                const unsigned m = __ballot_sync(FULL_MASK, ok);
                if (FILL && ok) nb[base + c + __popc(m & ltmask)] = j;
                c += __popc(m);
            }
        }
        __syncthreads();
    }
    if (!FILL && rowValid && lane == 0) counts[i] = c;
}

// Per-team working set (cap entries each). For every candidate c it holds
// the cluster member m attaining c's max distance (so the exact max squared
// distance is dist2(c, m)), a float estimate sf of it with an error bound ef
// (|sf - exact| <= ef), and c's float coordinates.
struct TeamBuf {
    int* idx;
    int* mem;
    float* xf;
    float* yf;
    float* sf;
    float* ef;
};

static const int ENTRY_BYTES = 2 * sizeof(int) + 4 * sizeof(float);

struct GrowArgs {
    const double2* pts;
    const float2* ptsf;     // coordinates rounded to float
    const long long* offs;
    const int* nb;
    const int* clustered;
    double S;               // exact squared threshold
    float SfLo, SfHi;       // float bounds with SfLo < S < SfHi
    float errLin;           // float distance error terms (see approxDist2)
    float errConst;
    long long cap;          // max neighbour count (entries per team buffer)
    TeamBuf global;         // global scratch, `cap` entries per slot (SoA, slot-major)
    int* cache;             // per-seed member lists (stride cap + 1), or nullptr
};

// Team buffer in dynamic shared memory or global scratch
__device__ __forceinline__ TeamBuf teamBuffer(const GrowArgs& a, const bool useSmem,
                                              char* smem, const int teamInBlock,
                                              const long long slot) {
    int* base;
    if (useSmem) {
        base = reinterpret_cast<int*>(smem + static_cast<size_t>(teamInBlock) * a.cap * ENTRY_BYTES);
    } else {
        base = reinterpret_cast<int*>(a.global.idx) + static_cast<size_t>(slot) * a.cap * 6;
    }
    TeamBuf t;
    t.idx = base;
    t.mem = base + a.cap;
    t.xf = reinterpret_cast<float*>(base + 2 * a.cap);
    t.yf = t.xf + a.cap;
    t.sf = t.yf + a.cap;
    t.ef = t.sf + a.cap;
    return t;
}

// Relative margin beyond which two different squared distances are
// guaranteed to have different (correctly rounded) square roots
#define NEAR_DOWN (1.0 - 0x1p-46)
// Slack factors absorbing float rounding in interval comparisons
#define SL_UP (1.0f + 0x1p-21f)
#define SL_DN (1.0f - 0x1p-21f)

// True if candidate (sa, ia) precedes (sb, ib) in the sequential scan order:
// smaller distance sqrt(s), then lower index. sqrt is only evaluated when the
// squared values are within a few ulps of each other, the only case where
// rounding could make the distances of different squared values equal.
__device__ __forceinline__ bool precedes(const double sa, const int ia,
                                         const double sb, const int ib) {
    if (sa == sb) return ia < ib;
    const double lo = fmin(sa, sb), hi = fmax(sa, sb);
    if (lo < hi * NEAR_DOWN) return sa < sb;
    const double ra = sqrt(sa), rb = sqrt(sb);
    return ra < rb || (ra == rb && ia < ib);
}

__device__ __forceinline__ double exactDist2(const GrowArgs& a, const int i, const int j) {
    const double2 p = a.pts[i], q = a.pts[j];
    return dist2(p.x, p.y, q.x, q.y);
}

// Float estimate fd of the exact squared distance with error bound ed.
// With M = max |coordinate|, each float coordinate difference is within
// e = 2^-22 M (1 + eps) of the exact one; hence
// |fd - d2| <= 2^-22 fd + 2e (|dx| + |dy|) + 2e^2. errLin = 4e and
// errConst = 4e^2 (+ tiny) as well as 2^-20 fd double these terms for slack.
__device__ __forceinline__ void approxDist2(const float ax, const float ay, const float bx,
                                            const float by, const GrowArgs& a,
                                            float& fd, float& ed) {
    const float dx = ax - bx, dy = ay - by;
    fd = fmaf(dx, dx, dy * dy);
    ed = fd * 0x1p-20f + a.errLin * (fabsf(dx) + fabsf(dy)) + a.errConst;
}

// True if the value within [fa +- ea] is certainly below (1 - 2^-21) times
// the value within [fb +- eb]. False (never wrong) when undecidable,
// including NaN/inf bounds.
__device__ __forceinline__ bool surelyBelow(const float fa, const float ea,
                                            const float fb, const float eb) {
    return (fa + ea) * SL_UP < (fb - eb) * SL_DN;
}

// Order of two candidates (c, m, sf, ef) as in the sequential scan; exact
// distances are only computed when the float intervals cannot decide.
__device__ __forceinline__ bool candPrecedes(const GrowArgs& a,
                                             const float fa, const float ea, const int ca, const int ma,
                                             const float fb, const float eb, const int cb, const int mb) {
    if (cb == INT_MAX) return ca != INT_MAX;
    if (ca == INT_MAX) return false;
    if (surelyBelow(fa, ea, fb, eb)) return true;
    if (surelyBelow(fb, eb, fa, ea)) return false;
    return precedes(exactDist2(a, ca, ma), ca, exactDist2(a, cb, mb), cb);
}

// Grow the candidate cluster of `seed` with a team of W warps.
// For W > 1 the team must be the whole thread block.
// Returns the cardinality (uniform across the team). If out != nullptr,
// the member list is written to out in insertion order.
template <int W>
__device__ int growCluster(const int seed, const TeamBuf tb, const GrowArgs& a, int* out) {
    const int lane = threadIdx.x & 31;
    const int w = (W == 1) ? 0 : (threadIdx.x >> 5);
    const unsigned ltmask = (1u << lane) - 1u;

    const long long b = a.offs[seed];
    const int deg = static_cast<int>(a.offs[seed + 1] - b);
    const int segLen = (deg + W - 1) / W;
    const int segStart = min(deg, w * segLen);
    const int segEnd = min(deg, segStart + segLen);
    int* __restrict__ I = tb.idx + segStart;
    int* __restrict__ MM = tb.mem + segStart;
    float* __restrict__ XF = tb.xf + segStart;
    float* __restrict__ YF = tb.yf + segStart;
    float* __restrict__ SF = tb.sf + segStart;
    float* __restrict__ EF = tb.ef + segStart;
    const bool writer = out && w == 0 && lane == 0;

    const float2 spf = a.ptsf[seed];
    if (writer) out[0] = seed;

    // Thread-local best candidate
    float fb = INFINITY, eb = 0.0f, xb = 0.0f, yb = 0.0f;
    int ib = INT_MAX, mb = 0;
    auto consider = [&](float sf, float ef, int c, int m, float xf, float yf) {
        if (candPrecedes(a, sf, ef, c, m, fb, eb, ib, mb)) {
            fb = sf; eb = ef; ib = c; mb = m; xb = xf; yb = yf;
        }
    };

    // Initial pass: gather unclustered neighbours of the seed (all of them
    // are within the threshold of the seed by construction)
    int cnt = 0;
    for (int p0 = segStart; p0 < segEnd; p0 += 32) {
        const int p = p0 + lane;
        bool keep = false;
        int j = 0;
        if (p < segEnd) {
            j = a.nb[b + p];
            keep = !a.clustered[j];
        }
        const unsigned m = __ballot_sync(FULL_MASK, keep);
        if (keep) {
            const int pos = cnt + __popc(m & ltmask);
            const float2 qf = a.ptsf[j];
            float fd, ed;
            approxDist2(qf.x, qf.y, spf.x, spf.y, a, fd, ed);
            I[pos] = j; MM[pos] = seed; XF[pos] = qf.x; YF[pos] = qf.y; SF[pos] = fd; EF[pos] = ed;
            consider(fd, ed, j, seed, qf.x, qf.y);
        }
        cnt += __popc(m);
    }

    __shared__ float shF[2][W > 1 ? W : 1], shE[2][W > 1 ? W : 1];
    __shared__ float shX[2][W > 1 ? W : 1], shY[2][W > 1 ? W : 1];
    __shared__ int shI[2][W > 1 ? W : 1], shM[2][W > 1 ? W : 1];

    int count = 1;
    int parity = 0;
    while (true) {
        // Team-wide argmin on (distance, index)
        {
            float f = fb, e = eb;
            int i = ib, mm = mb;
            for (int o = 16; o > 0; o >>= 1) {
                const float f2 = __shfl_xor_sync(FULL_MASK, f, o);
                const float e2 = __shfl_xor_sync(FULL_MASK, e, o);
                const int i2 = __shfl_xor_sync(FULL_MASK, i, o);
                const int m2 = __shfl_xor_sync(FULL_MASK, mm, o);
                if (candPrecedes(a, f2, e2, i2, m2, f, e, i, mm)) { f = f2; e = e2; i = i2; mm = m2; }
            }
            // Fetch the float coordinates from the lane holding the winner
            const unsigned src = __ballot_sync(FULL_MASK, ib == i && i != INT_MAX);
            const int srcLane = src ? __ffs(src) - 1 : 0;
            xb = __shfl_sync(FULL_MASK, xb, srcLane);
            yb = __shfl_sync(FULL_MASK, yb, srcLane);
            fb = f; eb = e; ib = i; mb = mm;
        }
        if (W > 1) {
            if (lane == 0) {
                shF[parity][w] = fb; shE[parity][w] = eb; shI[parity][w] = ib; shM[parity][w] = mb;
                shX[parity][w] = xb; shY[parity][w] = yb;
            }
            __syncthreads();
            fb = shF[parity][0]; eb = shE[parity][0]; ib = shI[parity][0]; mb = shM[parity][0];
            xb = shX[parity][0]; yb = shY[parity][0];
#pragma unroll
            for (int k = 1; k < W; ++k) {
                const float f2 = shF[parity][k], e2 = shE[parity][k];
                const int i2 = shI[parity][k], m2 = shM[parity][k];
                if (candPrecedes(a, f2, e2, i2, m2, fb, eb, ib, mb)) {
                    fb = f2; eb = e2; ib = i2; mb = m2; xb = shX[parity][k]; yb = shY[parity][k];
                }
            }
            parity ^= 1;
        }
        if (ib == INT_MAX) break;  // no more points can be added

        const int newIdx = ib;
        const float nxf = xb, nyf = yb;
        if (writer) out[count] = newIdx;
        ++count;

        fb = INFINITY; eb = 0.0f; ib = INT_MAX; mb = 0;
        // Update max distances with the new member, drop infeasible ones,
        // compact in place (stable) and find the next best candidate.
        int ncnt = 0;
        for (int p0 = 0; p0 < cnt; p0 += 32) {
            const int p = p0 + lane;
            bool keep = false, changed = false;
            float sf = 0.0f, ef = 0.0f, xf = 0.0f, yf = 0.0f;
            int c = 0, mm = 0;
            if (p < cnt) {
                c = I[p];
                if (c != newIdx) {
                    mm = MM[p]; xf = XF[p]; yf = YF[p]; sf = SF[p]; ef = EF[p];
                    float fd, ed;
                    approxDist2(xf, yf, nxf, nyf, a, fd, ed);
                    keep = true;
                    if (surelyBelow(fd, ed, sf, ef)) {
                        // new member is closer than the current max: unchanged
                    } else {
                        if (surelyBelow(sf, ef, fd, ed)) {
                            mm = newIdx; sf = fd; ef = ed;
                        } else {
                            // undecidable in float: exact comparison
                            const double s = exactDist2(a, c, mm);
                            const double d = exactDist2(a, c, newIdx);
                            const double smax = d > s ? d : s;
                            if (d > s) mm = newIdx;
                            sf = __double2float_rn(smax);
                            ef = sf * 0x1p-23f + 1e-37f;
                        }
                        changed = true;
                        // threshold check: exact max squared distance < S
                        if (!((sf + ef) * SL_UP < a.SfLo)) {
                            if ((sf - ef) * SL_DN > a.SfHi) keep = false;
                            else keep = exactDist2(a, c, mm) < a.S;
                        }
                    }
                }
            }
            const unsigned m = __ballot_sync(FULL_MASK, keep);
            if (keep) {
                const int pos = ncnt + __popc(m & ltmask);
                if (changed || pos != p) { MM[pos] = mm; SF[pos] = sf; EF[pos] = ef; }
                if (pos != p) { I[pos] = c; XF[pos] = xf; YF[pos] = yf; }
                consider(sf, ef, c, mm, xf, yf);
            }
            ncnt += __popc(m);
        }
        cnt = ncnt;
    }
    return count;
}

// Device-resident iteration state; lets the host enqueue many clustering
// iterations without synchronizing.
struct DevState {
    unsigned long long key;     // argmax accumulator: (cardinality << 32) | ~seed
    unsigned long long curKey;  // winner of the current iteration (0 = none)
    int dirtyCount;             // number of seeds in the dirty list
    int work;                   // dynamic work counter of the batch kernels
    int blocksDone;             // argmax completion counter
    int total;                  // points clustered so far
    int curOff;                 // offset of the current cluster in the member array
    int numClusters;
    int done;
};

static const int MAX_TEAM_WARPS = 16;

// Team size (warps per seed) used for a batch of D dirty seeds: enough teams
// to fill the GPU, but no more warps per seed than useful for its neighbours
__host__ __device__ inline int selectTeamWarps(const int D, const int maxDeg, const int warpCap,
                                               const int minW) {
    int w = minW;
    while (w < MAX_TEAM_WARPS && static_cast<long long>(D) * w < warpCap / 2 &&
           64 * w < maxDeg) {
        w *= 2;
    }
    return w;
}

// Recompute cached cardinalities (and member lists) for all dirty seeds.
// Every team size is launched; only the one selected for D does work.
template <int W>
__global__ void __launch_bounds__(W == 1 ? 256 : 32 * W)
batchKernel(const GrowArgs a, const bool useSmem, const int* __restrict__ list,
            int* __restrict__ card, int* __restrict__ dirtyFlag, DevState* st,
            const int maxDeg, const int warpCap, const int minW) {
    if (st->done) return;
    const int D = st->dirtyCount;
    if (selectTeamWarps(D, maxDeg, warpCap, minW) != W) return;

    extern __shared__ __align__(16) char dynSmem[];
    const int teamsPerBlock = blockDim.x / (32 * W);
    const int teamInBlock = threadIdx.x / (32 * W);
    const long long slot = static_cast<long long>(blockIdx.x) * teamsPerBlock + teamInBlock;
    const TeamBuf tb = teamBuffer(a, useSmem, dynSmem, teamInBlock, slot);
    const bool leader = (threadIdx.x % (32 * W)) == 0;
    // The first item is assigned statically so that the first teams are
    // spread across SMs (consecutive blocks land on different SMs); the rest
    // is distributed dynamically.
    const int totalTeams = gridDim.x * teamsPerBlock;
    int t = teamInBlock * gridDim.x + blockIdx.x;
    __shared__ int shT;
    while (true) {
        if (t >= D) break;
        const int seed = list[t];
        int* out = a.cache ? a.cache + static_cast<size_t>(seed) * (a.cap + 1) : nullptr;
        const int c = growCluster<W>(seed, tb, a, out);
        if (leader) {
            card[seed] = c;
            dirtyFlag[seed] = 0;
        }
        if (W == 1) {
            if (leader) t = totalTeams + atomicAdd(&st->work, 1);
            t = __shfl_sync(FULL_MASK, t, 0);
        } else {
            if (leader) shT = totalTeams + atomicAdd(&st->work, 1);
            __syncthreads();
            t = shT;
            __syncthreads();
        }
    }
}

// Find the unclustered seed with maximum cardinality (lowest index on ties).
// The last block to finish records the cluster, emits its members from the
// cache (if available), marks them clustered and resets per-iteration state.
__global__ void __launch_bounds__(256)
argmaxKernel(const GrowArgs a, const int* __restrict__ card, int* __restrict__ clustered,
             const int N, DevState* st, int* __restrict__ members,
             int* __restrict__ clusterSeed, int* __restrict__ clusterSize) {
    if (st->done) {
        if (blockIdx.x == 0 && threadIdx.x == 0) st->curKey = 0;
        return;
    }
    unsigned long long best = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x) {
        if (!clustered[i]) {
            const unsigned long long k =
                (static_cast<unsigned long long>(card[i]) << 32) | (0xFFFFFFFFull - i);
            best = k > best ? k : best;
        }
    }
    for (int o = 16; o > 0; o >>= 1) {
        const unsigned long long v = __shfl_xor_sync(FULL_MASK, best, o);
        best = v > best ? v : best;
    }
    __shared__ unsigned long long sh[32];
    __shared__ bool isLast;
    __shared__ int shOff, shCard, shSeed;
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) sh[wid] = best;
    __syncthreads();
    if (threadIdx.x == 0) {
        best = 0;
        for (int k = 0; k < static_cast<int>(blockDim.x >> 5); ++k) best = sh[k] > best ? sh[k] : best;
        if (best) atomicMax(&st->key, best);
        __threadfence();
        isLast = atomicAdd(&st->blocksDone, 1) == static_cast<int>(gridDim.x) - 1;
    }
    __syncthreads();
    if (!isLast) return;

    if (threadIdx.x == 0) {
        __threadfence();
        const unsigned long long k = atomicAdd(&st->key, 0ull);
        st->key = 0;
        st->blocksDone = 0;
        st->work = 0;
        st->dirtyCount = 0;
        st->curKey = k;
        shCard = 0;
        if (k == 0) {
            st->done = 1;  // no more clusters can be formed
        } else {
            shCard = static_cast<int>(k >> 32);
            shSeed = static_cast<int>(0xFFFFFFFFull - (k & 0xFFFFFFFFull));
            shOff = st->total;
            clusterSeed[st->numClusters] = shSeed;
            clusterSize[st->numClusters] = shCard;
            st->numClusters += 1;
            st->curOff = shOff;
            st->total = shOff + shCard;
            if (st->total >= N) st->done = 1;
        }
    }
    __syncthreads();
    if (a.cache && shCard > 0) {
        const int* src = a.cache + static_cast<size_t>(shSeed) * (a.cap + 1);
        for (int i = threadIdx.x; i < shCard; i += blockDim.x) {
            const int m = src[i];
            members[shOff + i] = m;
            clustered[m] = 1;
        }
    }
}

static const int WINNER_W = 8;

// Without a member cache: regrow the winning cluster, emit its members and
// mark them clustered
__global__ void __launch_bounds__(32 * WINNER_W)
winnerKernel(const GrowArgs a, const bool useSmem, int* __restrict__ clustered,
             int* __restrict__ members, const DevState* st) {
    extern __shared__ __align__(16) char dynSmem[];
    const unsigned long long key = st->curKey;
    if (key == 0) return;
    const int seed = static_cast<int>(0xFFFFFFFFull - (key & 0xFFFFFFFFull));
    int* out = members + st->curOff;
    const int c = growCluster<WINNER_W>(seed, teamBuffer(a, useSmem, dynSmem, 0, 0), a, out);
    __syncthreads();
    for (int i = threadIdx.x; i < c; i += blockDim.x) clustered[out[i]] = 1;
}

// Collect unclustered neighbours of the newly clustered members
__global__ void markDirtyKernel(const long long* __restrict__ offs, const int* __restrict__ nb,
                                const int* __restrict__ clustered, const int* __restrict__ members,
                                int* __restrict__ dirtyFlag, int* __restrict__ list,
                                DevState* st) {
    const unsigned long long key = st->curKey;
    if (key == 0) return;
    const int c = static_cast<int>(key >> 32);
    const int* mem = members + st->curOff;
    const int lane = threadIdx.x & 31;
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int nwarps = (gridDim.x * blockDim.x) >> 5;
    for (int k = warp; k < c; k += nwarps) {
        const int m = mem[k];
        const long long b = offs[m], e = offs[m + 1];
        for (long long p = b + lane; p < e; p += 32) {
            const int j = nb[p];
            if (!clustered[j] && atomicExch(&dirtyFlag[j], 1) == 0) {
                list[atomicAdd(&st->dirtyCount, 1)] = j;
            }
        }
    }
}

// Smallest double S such that sqrt(S) >= threshold, so that
// sqrt(d2) < threshold  <=>  d2 < S  (sqrt is correctly rounded and monotone)
static double squaredThreshold(const double threshold) {
    double S = threshold * threshold;
    if (std::sqrt(S) >= threshold) {
        while (S > 0.0) {
            const double prev = std::nextafter(S, 0.0);
            if (std::sqrt(prev) >= threshold) S = prev; else break;
        }
    } else {
        while (std::sqrt(S) < threshold) S = std::nextafter(S, INFINITY);
    }
    return S;
}

// Launch configuration for a batch kernel instantiation
struct BatchConfig {
    int threads;
    size_t smem;
    bool useSmem;
    int grid;
    int warpsPerSM;  // resident warps per SM
};

template <int W>
static BatchConfig makeConfig(const int numSMs, const long long teamBytes, const int smemLimit,
                              const long long globalSlots) {
    BatchConfig c;
    int teams = (W == 1) ? 8 : 1;
    if (W == 1) {
        while (teams > 1 && teams * teamBytes > smemLimit) teams /= 2;
        if (teams < 4) teams = 8;  // too few warps per block: use global scratch
    }
    c.useSmem = teams * teamBytes <= smemLimit;
    c.threads = 32 * W * teams;
    c.smem = c.useSmem ? static_cast<size_t>(teams * teamBytes) : 0;
    CUDA_CHECK(cudaFuncSetAttribute(batchKernel<W>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    static_cast<int>(c.smem)));
    int perSM = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&perSM, batchKernel<W>,
                                                             c.threads, c.smem));
    c.warpsPerSM = perSM * c.threads / 32;
    long long g = static_cast<long long>(std::max(perSM, 1)) * numSMs;
    if (!c.useSmem) g = std::min(g, globalSlots / teams);
    c.grid = static_cast<int>(std::max(g, 1LL));
    return c;
}

template <int W>
static void launchBatch(const BatchConfig& c, cudaStream_t stream, const GrowArgs& ga,
                        const int* list, int* card, int* dirtyFlag, DevState* st,
                        const int maxDeg, const int warpCap, const int minW) {
    // Only team sizes that selectTeamWarps can return are launched
    if (W < minW || (W > minW && 32 * W >= maxDeg)) return;
    batchKernel<W><<<c.grid, c.threads, c.smem, stream>>>(ga, c.useSmem, list, card, dirtyFlag,
                                                           st, maxDeg, warpCap, minW);
}

// Main QT clustering algorithm
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<Cluster> clusters;
    if (N == 0) return clusters;

    const double S = squaredThreshold(threshold);
    // Float bounds SfLo < S < SfHi (with relative margin 2^-20)
    float SfLo = static_cast<float>(S * (1.0 - 0x1p-20));
    if (static_cast<double>(SfLo) > S * (1.0 - 0x1p-20)) SfLo = std::nextafter(SfLo, 0.0f);
    float SfHi = static_cast<float>(S * (1.0 + 0x1p-20));
    if (static_cast<double>(SfHi) < S * (1.0 + 0x1p-20)) SfHi = std::nextafter(SfHi, INFINITY);

    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    int numSMs = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, dev));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Points (exact, and rounded to float for the filter)
    double2* d_pts = nullptr;
    float2* d_ptsf = nullptr;
    CUDA_CHECK(cudaMalloc(&d_pts, sizeof(double2) * N));
    CUDA_CHECK(cudaMalloc(&d_ptsf, sizeof(float2) * N));
    double maxAbs = 0.0;
    {
        std::vector<double2> hp(N);
        std::vector<float2> hpf(N);
        for (int i = 0; i < N; ++i) {
            hp[i] = make_double2(points[i].x, points[i].y);
            hpf[i] = make_float2(static_cast<float>(points[i].x), static_cast<float>(points[i].y));
            maxAbs = std::max({maxAbs, std::fabs(points[i].x), std::fabs(points[i].y)});
        }
        CUDA_CHECK(cudaMemcpy(d_pts, hp.data(), sizeof(double2) * N, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ptsf, hpf.data(), sizeof(float2) * N, cudaMemcpyHostToDevice));
    }
    // Float distance error terms (see approxDist2), with e = 2^-21 M which
    // includes a factor 2 of slack. For huge or non-finite coordinates the
    // float filter is disabled (NaN/inf bounds make every test undecidable).
    float errLin = INFINITY, errConst = INFINITY;
    if (maxAbs < 1e15) {
        const double e = std::ldexp(std::max(maxAbs, 1e-30), -21);
        errLin = static_cast<float>(4.0 * e) * (1.0f + 0x1p-20f);
        errConst = static_cast<float>(4.0 * e * e) * (1.0f + 0x1p-20f) + 1e-37f;
    }

    // Neighbour lists (CSR)
    int* d_counts = nullptr;
    long long* d_offs = nullptr;
    CUDA_CHECK(cudaMalloc(&d_counts, sizeof(int) * N));
    CUDA_CHECK(cudaMalloc(&d_offs, sizeof(long long) * (N + 1)));
    const int nbBlocks = (N + NB_ROWS - 1) / NB_ROWS;
    neighborKernel<false><<<nbBlocks, 32 * NB_ROWS, 0, stream>>>(d_pts, N, S, d_counts, nullptr, nullptr);
    CUDA_CHECK(cudaGetLastError());
    std::vector<int> hcounts(N);
    CUDA_CHECK(cudaMemcpyAsync(hcounts.data(), d_counts, sizeof(int) * N, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<long long> hoffs(N + 1);
    hoffs[0] = 0;
    int maxDeg = 0;
    for (int i = 0; i < N; ++i) {
        hoffs[i + 1] = hoffs[i] + hcounts[i];
        maxDeg = std::max(maxDeg, hcounts[i]);
    }
    const long long nnz = hoffs[N];
    int* d_nb = nullptr;
    CUDA_CHECK(cudaMalloc(&d_nb, sizeof(int) * std::max(nnz, 1LL)));
    CUDA_CHECK(cudaMemcpyAsync(d_offs, hoffs.data(), sizeof(long long) * (N + 1), cudaMemcpyHostToDevice, stream));
    neighborKernel<true><<<nbBlocks, 32 * NB_ROWS, 0, stream>>>(d_pts, N, S, nullptr, d_offs, d_nb);
    CUDA_CHECK(cudaGetLastError());

    // State
    int *d_clustered, *d_card, *d_dirtyFlag, *d_list, *d_members, *d_clusterSeed, *d_clusterSize;
    DevState* d_st;
    CUDA_CHECK(cudaMalloc(&d_clustered, sizeof(int) * N));
    CUDA_CHECK(cudaMalloc(&d_card, sizeof(int) * N));
    CUDA_CHECK(cudaMalloc(&d_dirtyFlag, sizeof(int) * N));
    CUDA_CHECK(cudaMalloc(&d_list, sizeof(int) * N));
    CUDA_CHECK(cudaMalloc(&d_members, sizeof(int) * N));
    CUDA_CHECK(cudaMalloc(&d_clusterSeed, sizeof(int) * N));
    CUDA_CHECK(cudaMalloc(&d_clusterSize, sizeof(int) * N));
    CUDA_CHECK(cudaMalloc(&d_st, sizeof(DevState)));
    CUDA_CHECK(cudaMemsetAsync(d_clustered, 0, sizeof(int) * N, stream));
    CUDA_CHECK(cudaMemsetAsync(d_dirtyFlag, 0, sizeof(int) * N, stream));
    DevState hst;
    DevState* h_st = &hst;
    memset(h_st, 0, sizeof(DevState));
    h_st->dirtyCount = N;  // initially every point is a seed to evaluate
    CUDA_CHECK(cudaMemcpyAsync(d_st, h_st, sizeof(DevState), cudaMemcpyHostToDevice, stream));
    {
        std::vector<int> all(N);
        for (int i = 0; i < N; ++i) all[i] = i;
        CUDA_CHECK(cudaMemcpyAsync(d_list, all.data(), sizeof(int) * N, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    // Per-team scratch: shared memory when it fits, otherwise global memory
    // (cap rounded so that every per-team array stays 16-byte aligned)
    const long long cap = (std::max(maxDeg, 1) + 3LL) & ~3LL;
    const long long teamBytes = cap * ENTRY_BYTES;
    int smemOptin = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&smemOptin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
    const int smemLimit = smemOptin - 2048;  // leave room for static shared memory
    size_t freeMem = 0, totalMem = 0;
    CUDA_CHECK(cudaMemGetInfo(&freeMem, &totalMem));

    GrowArgs ga{d_pts, d_ptsf, d_offs, d_nb, d_clustered, S, SfLo, SfHi, errLin, errConst, cap,
                {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr}, nullptr};

    // Member-list cache (lets the winner be emitted without regrowing it)
    const long long cacheBytes = static_cast<long long>(N) * (cap + 1) * sizeof(int);
    if (cacheBytes <= static_cast<long long>(freeMem * 0.4)) {
        CUDA_CHECK(cudaMalloc(&ga.cache, cacheBytes));
    }

    long long globalSlots = 0;
    if (4 * teamBytes > smemLimit) {
        const long long wantSlots = static_cast<long long>(numSMs) * 64;
        CUDA_CHECK(cudaMemGetInfo(&freeMem, &totalMem));
        globalSlots = std::min(wantSlots, static_cast<long long>(freeMem * 0.6) / teamBytes);
        globalSlots = std::max(globalSlots, 8LL);
        // one allocation, `ENTRY_BYTES * cap` per slot (see teamBuffer)
        CUDA_CHECK(cudaMalloc(&ga.global.idx, static_cast<size_t>(ENTRY_BYTES) * globalSlots * cap));
    }

    const int warpCap = numSMs * 32;
    const BatchConfig cfg1 = makeConfig<1>(numSMs, teamBytes, smemLimit, globalSlots);
    const BatchConfig cfg2 = makeConfig<2>(numSMs, teamBytes, smemLimit, globalSlots);
    const BatchConfig cfg4 = makeConfig<4>(numSMs, teamBytes, smemLimit, globalSlots);
    const BatchConfig cfg8 = makeConfig<8>(numSMs, teamBytes, smemLimit, globalSlots);
    const BatchConfig cfg16 = makeConfig<16>(numSMs, teamBytes, smemLimit, globalSlots);
    // Smallest team size whose shared-memory configuration keeps enough warps
    // resident per SM; otherwise 8-warp teams on global scratch
    int minW = 8;
    {
        const BatchConfig* cfgs[] = {&cfg1, &cfg2, &cfg4, &cfg8};
        const int ws[] = {1, 2, 4, 8};
        for (int k = 0; k < 4; ++k) {
            if (cfgs[k]->useSmem && cfgs[k]->warpsPerSM >= 8) { minW = ws[k]; break; }
        }
    }
    const size_t winnerSmemBytes = teamBytes <= smemLimit ? static_cast<size_t>(teamBytes) : 0;
    if (!ga.cache) {
        CUDA_CHECK(cudaFuncSetAttribute(winnerKernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(winnerSmemBytes)));
    }
    const int argmaxGrid = std::max(1, std::min((N + 255) / 256, numSMs * 4));
    const int markGrid = std::max(1, numSMs * 4);

    // Each iteration extracts one cluster; all its decisions are made on the
    // device. A group of iterations is captured once into a CUDA graph and
    // replayed until done (iterations after completion are no-ops).
    auto enqueueIteration = [&]() {
        launchBatch<1>(cfg1, stream, ga, d_list, d_card, d_dirtyFlag, d_st, maxDeg, warpCap, minW);
        launchBatch<2>(cfg2, stream, ga, d_list, d_card, d_dirtyFlag, d_st, maxDeg, warpCap, minW);
        launchBatch<4>(cfg4, stream, ga, d_list, d_card, d_dirtyFlag, d_st, maxDeg, warpCap, minW);
        launchBatch<8>(cfg8, stream, ga, d_list, d_card, d_dirtyFlag, d_st, maxDeg, warpCap, minW);
        launchBatch<16>(cfg16, stream, ga, d_list, d_card, d_dirtyFlag, d_st, maxDeg, warpCap, minW);
        argmaxKernel<<<argmaxGrid, 256, 0, stream>>>(ga, d_card, d_clustered, N, d_st, d_members,
                                                     d_clusterSeed, d_clusterSize);
        if (!ga.cache) {
            winnerKernel<<<1, 32 * WINNER_W, winnerSmemBytes, stream>>>(
                ga, winnerSmemBytes > 0, d_clustered, d_members, d_st);
        }
        markDirtyKernel<<<markGrid, 256, 0, stream>>>(d_offs, d_nb, d_clustered, d_members,
                                                      d_dirtyFlag, d_list, d_st);
    };
    const int GROUP = 16;
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    for (int it = 0; it < GROUP; ++it) enqueueIteration();
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, 0));

    // The first iteration (full evaluation of all seeds) runs directly
    enqueueIteration();
    CUDA_CHECK(cudaGetLastError());
    while (true) {
        CUDA_CHECK(cudaGraphLaunch(graphExec, stream));
        CUDA_CHECK(cudaMemcpyAsync(h_st, d_st, sizeof(DevState), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if (h_st->done) break;
    }
    CUDA_CHECK(cudaGraphExecDestroy(graphExec));
    CUDA_CHECK(cudaGraphDestroy(graph));

    const int numClusters = h_st->numClusters;
    const int total = h_st->total;
    std::vector<int> members(total), seeds(numClusters), sizes(numClusters);
    if (numClusters > 0) {
        CUDA_CHECK(cudaMemcpy(members.data(), d_members, sizeof(int) * total, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(seeds.data(), d_clusterSeed, sizeof(int) * numClusters, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(sizes.data(), d_clusterSize, sizeof(int) * numClusters, cudaMemcpyDeviceToHost));
    }
    clusters.reserve(numClusters);
    int off = 0;
    for (int c = 0; c < numClusters; ++c) {
        Cluster cluster;
        cluster.seed_point = seeds[c];
        cluster.members.assign(members.begin() + off, members.begin() + off + sizes[c]);
        off += sizes[c];
        clusters.push_back(std::move(cluster));
    }

    cudaFree(ga.global.idx);
    cudaFree(ga.cache);
    cudaFree(d_pts); cudaFree(d_ptsf); cudaFree(d_counts); cudaFree(d_offs); cudaFree(d_nb);
    cudaFree(d_clustered); cudaFree(d_card); cudaFree(d_dirtyFlag); cudaFree(d_list);
    cudaFree(d_members); cudaFree(d_clusterSeed); cudaFree(d_clusterSize); cudaFree(d_st);
    cudaStreamDestroy(stream);

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
    
    // Initialize the CUDA context (loading all kernels eagerly) outside the
    // timed region
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
