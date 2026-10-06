// QT Clustering Benchmark - Hybrid MPI + OpenMP + CUDA Version
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

#include <cub/cub.cuh>
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

// Squared Euclidean distance between two points.
// Written with an explicit fused multiply-add so host and device produce
// bit-identical results (this matches the contraction of the reference build).
__host__ __device__ inline double pointDistance2(const double ax, const double ay,
                                                 const double bx, const double by) {
    const double dx = ax - bx;
    const double dy = ay - by;
#ifdef __CUDA_ARCH__
    return __fma_rn(dx, dx, __dmul_rn(dy, dy));
#else
    return std::fma(dx, dx, dy * dy);
#endif
}

// Calculate Euclidean distance between two points
inline double distance(const Point& p1, const Point& p2) {
    return std::sqrt(pointDistance2(p1.x, p1.y, p2.x, p2.y));
}

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), \
                    __FILE__, __LINE__);                                        \
            MPI_Abort(MPI_COMM_WORLD, 1);                                       \
        }                                                                       \
    } while (0)

static constexpr int WARP = 32;
static constexpr int BLOCK = 256;
static constexpr int WARPS_PER_BLOCK = BLOCK / WARP;
static constexpr int NO_POINT = 0x7fffffff;

// Candidate clusters are ranked by a packed 64-bit key:
// (cardinality << 32) | (0xffffffff - seed), i.e. larger cardinality first and
// the lowest seed index on ties -- the selection rule of the sequential code.
__host__ __device__ inline unsigned long long packKey(const int card, const int seed) {
    return (static_cast<unsigned long long>(card) << 32) |
           static_cast<unsigned long long>(0xffffffffu - static_cast<unsigned>(seed));
}

struct WorkState {
    unsigned long long best;       // best key found so far (starts from the exact cached ones)
    int counter[2];                // dynamic work distribution counters (warp queue stages)
    int counterBig;                // dynamic work distribution counter (block queue)
    int boundCount;                // number of seeds that need a new upper bound
    int firstRound;                // 1 until the first cluster has been selected
    float2 lastSeed;               // coordinates of the seed of the last selected cluster
};

// Host staging area in mapped pinned memory
struct HostStaging {
    unsigned long long localKey;   // written by publishBestKernel
    unsigned long long globalKey;  // selected cluster, written by the host
    int members[1];                // members of the cluster (N entries allocated)
};

// Per-warp scratch (structure of arrays) for the live candidate set of a seed
struct Scratch {
    float2* xy;   // candidate coordinates relative to the seed (single precision)
    float* mdf;   // single precision copy of md (+inf marks a dropped candidate)
    double* md;   // exact squared max distance from candidate to cluster members
    int* id;      // point index of candidate
};

// Parameters of the exact squared-distance tests.
// Distances are handled squared: max() commutes with the monotone sqrt, and
// "sqrt(m2) < threshold" is equivalent to "m2 < thr2" where thr2 is the smallest
// double whose sqrt reaches the threshold.
// Single precision values are only used as filters with a rigorous error margin
// ("eps"); every decision that could go either way is taken in double precision.
struct DistParams {
    double thr2;
    float thr2f;
    float eps;    // bound on |float estimate - exact| for squared distances (+ slack)
};

static constexpr long long ULP_BAND = 1ll << 16;

// Candidate ordering identical to the reference: sqrt(m2) ascending, then the
// lowest point index. Non-negative doubles are ordered like their bit patterns;
// only values within ULP_BAND units may have square roots rounding to the same
// double, in which case the square roots are compared exactly.
__device__ __forceinline__ bool better(const double a, const int ia, const double b, const int ib) {
    const long long A = __double_as_longlong(a);
    const long long B = __double_as_longlong(b);
    if (A + ULP_BAND < B) return true;
    if (A > B + ULP_BAND) return false;
    if (A == B) return ia < ib;
    const double sa = sqrt(a), sb = sqrt(b);
    return sa < sb || (sa == sb && ia < ib);
}

// Warp-wide arg-min of (md, id); pos (scratch position of the entry) travels along
__device__ __forceinline__ void warpArgMin(double& md, int& id, int& pos) {
    #pragma unroll
    for (int off = WARP / 2; off > 0; off >>= 1) {
        const double omd = __shfl_xor_sync(0xffffffffu, md, off);
        const int oid = __shfl_xor_sync(0xffffffffu, id, off);
        const int opos = __shfl_xor_sync(0xffffffffu, pos, off);
        if (better(omd, oid, md, id)) { md = omd; id = oid; pos = opos; }
    }
}

__device__ __forceinline__ bool lessThr(const double m, const double thr2) {
    return __double_as_longlong(m) < __double_as_longlong(thr2);
}

// Single precision copy of a squared distance (finite, monotone in md)
__device__ __forceinline__ float toFloat(const double md) {
    return fminf((float)md, 3.4028234663852886e38f);
}

// Build the QT candidate cluster of "seed" with a single warp and return its
// cardinality; if "out" is given, the members are written in insertion order.
// nb: unclustered neighbours of seed (dist < threshold), K entries.
// Each greedy step picks the candidate with the smallest max-distance to the
// current members (lowest index on ties), exactly as the sequential algorithm.
// Candidates whose max-distance reaches the threshold are dropped for good
// (the max-distance can only grow); dropped entries are compacted lazily.
__device__ int greedyCluster(const int seed, const int* __restrict__ nb, int K,
                             const double2* __restrict__ pts, const DistParams dp,
                             Scratch s, int* __restrict__ out) {
    const int lane = threadIdx.x & (WARP - 1);
    const unsigned ltMask = (1u << lane) - 1u;
    const double2 sp = pts[seed];
    const float SLACK = 1.0f + 0x1p-20f;

    double bmd = HUGE_VAL;
    int bid = NO_POINT, bpos = -1;
    float bmf = HUGE_VALF;
    int w = 0;
    for (int base = 0; base < K; base += WARP) {
        const int j = base + lane;
        bool keep = false;
        double md = 0.0;
        double2 p = make_double2(0.0, 0.0);
        int id = 0;
        if (j < K) {
            id = nb[j];
            p = pts[id];
            md = pointDistance2(p.x, p.y, sp.x, sp.y);
            keep = lessThr(md, dp.thr2);
        }
        const unsigned m = __ballot_sync(0xffffffffu, keep);
        if (keep) {
            const int pos = w + __popc(m & ltMask);
            const float mf = toFloat(md);
            s.xy[pos] = make_float2((float)(p.x - sp.x), (float)(p.y - sp.y));
            s.mdf[pos] = mf; s.md[pos] = md; s.id[pos] = id;
            if (better(md, id, bmd, bid)) { bmd = md; bid = id; bpos = pos; bmf = mf; }
        }
        w += __popc(m);
    }
    K = w;

    int size = 1;
    if (out && lane == 0) out[0] = seed;
    int dead = 0;

    while (true) {
        warpArgMin(bmd, bid, bpos);
        if (bid == NO_POINT) break;
        const int newId = bid;
        const int newPos = bpos;
        if (out && lane == 0) out[size] = newId;
        ++size;
        const double2 np = pts[newId];
        const float nxf = (float)(np.x - sp.x);
        const float nyf = (float)(np.y - sp.y);

        bmd = HUGE_VAL;
        bid = NO_POINT;
        bpos = -1;
        bmf = HUGE_VALF;
        const bool compact = dead * 4 > K;
        w = 0;
        int died = 0;
        for (int base = 0; base < K; base += WARP) {
            const int j = base + lane;
            bool alive = false, changed = false;
            float f = HUGE_VALF;
            if (j < K) {
                f = s.mdf[j];
                alive = f < HUGE_VALF && j != newPos;
                changed = f < HUGE_VALF && !alive;
            }
            float2 xy = make_float2(0.0f, 0.0f);
            double md = 0.0;
            int id = 0;
            bool mdLoaded = false;
            if (alive) {
                xy = s.xy[j];
                const float dxf = xy.x - nxf, dyf = xy.y - nyf;
                const float d2f = fmaf(dxf, dxf, dyf * dyf);
                if (d2f + dp.eps < f) {
                    // max distance certainly unchanged
                } else if (d2f - dp.eps > dp.thr2f) {
                    alive = false;  // certainly beyond the threshold
                    changed = true;
                } else {
                    id = s.id[j];
                    md = s.md[j];
                    mdLoaded = true;
                    const double2 p = pts[id];
                    const double d2 = pointDistance2(p.x, p.y, np.x, np.y);
                    if (__double_as_longlong(d2) > __double_as_longlong(md)) {
                        md = d2;
                        changed = true;
                        if (lessThr(md, dp.thr2)) f = toFloat(md);
                        else alive = false;
                    }
                }
            }
            int pos = j;
            if (compact) {
                const unsigned m = __ballot_sync(0xffffffffu, alive);
                if (alive) {
                    if (!mdLoaded) { id = s.id[j]; md = s.md[j]; mdLoaded = true; }
                    pos = w + __popc(m & ltMask);
                    s.xy[pos] = xy; s.mdf[pos] = f; s.md[pos] = md; s.id[pos] = id;
                }
                w += __popc(m);
            } else if (changed) {
                if (alive) { s.mdf[j] = f; s.md[j] = md; }
                else { s.mdf[j] = HUGE_VALF; ++died; }
            }
            if (alive && f <= bmf * SLACK) {
                if (!mdLoaded) { id = s.id[j]; md = s.md[j]; }
                if (better(md, id, bmd, bid)) { bmd = md; bid = id; bpos = pos; bmf = f; }
            }
        }
        if (compact) {
            K = w;
            dead = 0;
        } else {
            #pragma unroll
            for (int off = WARP / 2; off > 0; off >>= 1) died += __shfl_xor_sync(0xffffffffu, died, off);
            dead += died;
        }
    }
    return size;
}

// Block-wide variant of greedyCluster for seeds with many neighbours: the live
// candidate set (coordinates and float max-distance) is held in shared memory,
// the exact max-distances and indices in global memory. Entries are never moved,
// so every entry is always handled by the same thread. Same selection rule and
// same result as greedyCluster.
static constexpr int BIG_BLOCK = 512;
static constexpr int BIG_WARPS = BIG_BLOCK / WARP;
static constexpr int KWARP = 4096;   // seeds with more neighbours use the block variant

struct BlockReduce {
    double md[2][BIG_WARPS];
    int id[2][BIG_WARPS];
    int pos[2][BIG_WARPS];
};

__device__ int blockGreedyCluster(const int seed, const int* __restrict__ nb, const int K,
                                  const double2* __restrict__ pts, const DistParams dp,
                                  float2* xy, float* mdf, double* __restrict__ md,
                                  int* __restrict__ idv, int* __restrict__ out, BlockReduce& red) {
    const int tid = threadIdx.x;
    const int lane = tid & (WARP - 1);
    const int warp = tid / WARP;
    const double2 sp = pts[seed];
    const float SLACK = 1.0f + 0x1p-20f;

    double bmd = HUGE_VAL;
    int bid = NO_POINT, bpos = -1;
    float bmf = HUGE_VALF;
    for (int j = tid; j < K; j += BIG_BLOCK) {
        const int id = nb[j];
        const double2 p = pts[id];
        const double m = pointDistance2(p.x, p.y, sp.x, sp.y);
        const float f = lessThr(m, dp.thr2) ? toFloat(m) : HUGE_VALF;
        xy[j] = make_float2((float)(p.x - sp.x), (float)(p.y - sp.y));
        mdf[j] = f; md[j] = m; idv[j] = id;
        if (f < HUGE_VALF && better(m, id, bmd, bid)) { bmd = m; bid = id; bpos = j; bmf = f; }
    }
    int size = 1;
    if (out && tid == 0) out[0] = seed;
    int parity = 0;
    while (true) {
        warpArgMin(bmd, bid, bpos);
        if (lane == 0) { red.md[parity][warp] = bmd; red.id[parity][warp] = bid; red.pos[parity][warp] = bpos; }
        __syncthreads();
        // every warp reduces the per-warp results itself (double buffered, one barrier)
        double rmd = lane < BIG_WARPS ? red.md[parity][lane] : HUGE_VAL;
        int rid = lane < BIG_WARPS ? red.id[parity][lane] : NO_POINT;
        int rpos = lane < BIG_WARPS ? red.pos[parity][lane] : -1;
        warpArgMin(rmd, rid, rpos);
        parity ^= 1;
        if (rid == NO_POINT) break;
        const int newId = rid;
        const int newPos = rpos;
        if (out && tid == 0) out[size] = newId;
        ++size;
        const double2 np = pts[newId];
        const float nxf = (float)(np.x - sp.x);
        const float nyf = (float)(np.y - sp.y);

        bmd = HUGE_VAL;
        bid = NO_POINT;
        bpos = -1;
        bmf = HUGE_VALF;
        for (int j = tid; j < K; j += BIG_BLOCK) {
            float f = mdf[j];
            if (!(f < HUGE_VALF)) continue;
            if (j == newPos) { mdf[j] = HUGE_VALF; continue; }
            const float2 c = xy[j];
            const float dxf = c.x - nxf, dyf = c.y - nyf;
            const float d2f = fmaf(dxf, dxf, dyf * dyf);
            double m = 0.0;
            int id = 0;
            bool loaded = false;
            if (d2f + dp.eps < f) {
                // max distance certainly unchanged
            } else if (d2f - dp.eps > dp.thr2f) {
                mdf[j] = HUGE_VALF;  // certainly beyond the threshold
                continue;
            } else {
                id = idv[j];
                m = md[j];
                loaded = true;
                const double2 p = pts[id];
                const double d2 = pointDistance2(p.x, p.y, np.x, np.y);
                if (__double_as_longlong(d2) > __double_as_longlong(m)) {
                    m = d2;
                    if (!lessThr(m, dp.thr2)) { mdf[j] = HUGE_VALF; continue; }
                    f = toFloat(m);
                    mdf[j] = f;
                    md[j] = m;
                }
            }
            if (f <= bmf * SLACK) {
                if (!loaded) { id = idv[j]; m = md[j]; }
                if (better(m, id, bmd, bid)) { bmd = m; bid = id; bpos = j; bmf = f; }
            }
        }
    }
    return size;
}

// Count neighbours (dist < threshold, excluding self) of each local seed
__global__ void countNeighborsKernel(const double2* __restrict__ pts,
                                     const float2* __restrict__ ptsf, const int N,
                                     const int* __restrict__ seeds, const int M,
                                     const double thr2, const float boxf, int* __restrict__ cnt) {
    const int lane = threadIdx.x & (WARP - 1);
    const int warpGlobal = (blockIdx.x * blockDim.x + threadIdx.x) / WARP;
    const int numWarps = (gridDim.x * blockDim.x) / WARP;
    for (int i = warpGlobal; i < M; i += numWarps) {
        const int s = seeds[i];
        const double2 sp = pts[s];
        const float2 spf = ptsf[s];
        int c = 0;
        for (int base = 0; base < N; base += WARP) {
            const int j = base + lane;
            bool keep = false;
            if (j < N && j != s) {
                // single precision bounding box rejection (conservative margin)
                const float2 pf = ptsf[j];
                if (fabsf(pf.x - spf.x) <= boxf && fabsf(pf.y - spf.y) <= boxf) {
                    const double2 p = pts[j];
                    keep = lessThr(pointDistance2(p.x, p.y, sp.x, sp.y), thr2);
                }
            }
            c += __popc(__ballot_sync(0xffffffffu, keep));
        }
        if (lane == 0) cnt[i] = c;
    }
}

// Fill neighbour lists (sorted by point index) of each local seed
__global__ void fillNeighborsKernel(const double2* __restrict__ pts,
                                    const float2* __restrict__ ptsf, const int N,
                                    const int* __restrict__ seeds, const int M,
                                    const double thr2, const float boxf,
                                    const long long* __restrict__ off, int* __restrict__ nbr) {
    const int lane = threadIdx.x & (WARP - 1);
    const unsigned ltMask = (1u << lane) - 1u;
    const int warpGlobal = (blockIdx.x * blockDim.x + threadIdx.x) / WARP;
    const int numWarps = (gridDim.x * blockDim.x) / WARP;
    for (int i = warpGlobal; i < M; i += numWarps) {
        const int s = seeds[i];
        const double2 sp = pts[s];
        const float2 spf = ptsf[s];
        int* list = nbr + off[i];
        int w = 0;
        for (int base = 0; base < N; base += WARP) {
            const int j = base + lane;
            bool keep = false;
            if (j < N && j != s) {
                const float2 pf = ptsf[j];
                if (fabsf(pf.x - spf.x) <= boxf && fabsf(pf.y - spf.y) <= boxf) {
                    const double2 p = pts[j];
                    keep = lessThr(pointDistance2(p.x, p.y, sp.x, sp.y), thr2);
                }
            }
            const unsigned m = __ballot_sync(0xffffffffu, keep);
            if (keep) list[w + __popc(m & ltMask)] = j;
            w += __popc(m);
        }
    }
}

// Start of a QT round on the local seeds: seeds near the last selected cluster
// drop their newly clustered neighbours. Seeds whose neighbourhood changed (all
// seeds on the first round) lose their exact cardinality and are listed for a
// new upper bound.
__global__ void roundRefreshKernel(const float2* __restrict__ ptsf,
                                   const unsigned char* __restrict__ clustered,
                                   const int* __restrict__ seeds, const int M,
                                   const long long* __restrict__ off,
                                   int* __restrict__ nbrCnt, int* __restrict__ nbr,
                                   unsigned char* __restrict__ exact,
                                   const float box2, int* __restrict__ boundList, WorkState* ws) {
    const int firstRound = ws->firstRound;
    const float2 lastSeed = ws->lastSeed;
    const int lane = threadIdx.x & (WARP - 1);
    const unsigned ltMask = (1u << lane) - 1u;
    const int warpGlobal = (blockIdx.x * blockDim.x + threadIdx.x) / WARP;
    const int numWarps = (gridDim.x * blockDim.x) / WARP;
    for (int i0 = warpGlobal * WARP; i0 < M; i0 += numWarps * WARP) {
        // one seed per lane for the cheap checks
        const int iMine = i0 + lane;
        bool touched = false;
        if (iMine < M) {
            const int seed = seeds[iMine];
            if (!clustered[seed]) {
                // Members of the last cluster lie within threshold of its seed, so
                // only seeds within 2 * threshold (box test, with margin) can change.
                const float2 pf = ptsf[seed];
                touched = firstRound ||
                          (fabsf(pf.x - lastSeed.x) <= box2 && fabsf(pf.y - lastSeed.y) <= box2);
            }
        }
        // compact the neighbour lists of touched seeds, one warp-wide pass per seed
        unsigned todo = __ballot_sync(0xffffffffu, touched);
        while (todo) {
            const int src = __ffs(todo) - 1;
            todo &= todo - 1;
            const int i = i0 + src;
            int* list = nbr + off[i];
            const int K = nbrCnt[i];
            int w = 0;
            for (int base = 0; base < K; base += WARP) {
                const int j = base + lane;
                bool keep = false;
                int id = 0;
                if (j < K) {
                    id = list[j];
                    keep = !clustered[id];
                }
                const unsigned m = __ballot_sync(0xffffffffu, keep);
                if (keep) list[w + __popc(m & ltMask)] = id;
                w += __popc(m);
            }
            if (lane == 0 && (firstRound || w != K)) {
                nbrCnt[i] = w;
                exact[i] = 0;
                boundList[atomicAdd(&ws->boundCount, 1)] = i;
            }
        }
    }
}

// Upper bound on the QT candidate cardinality of a seed.
// All members are pairwise closer than the threshold, so they fit in an
// axis-aligned square of side ~threshold that contains the seed. Neighbours are
// binned into cells of size threshold / BOUND_G around the seed; any such square
// covers at most BOUND_G + 3 consecutive cells per axis (including rounding
// slack), so the largest cell-window count containing the seed's cell bounds
// the cardinality (together with 1 + #neighbours).
static constexpr int BOUND_G = 40;
static constexpr int BOUND_W = BOUND_G + 3;
static constexpr int BOUND_DIM = 2 * BOUND_G + 3;     // cells per axis
static constexpr int BOUND_CENTER = BOUND_G + 1;      // cell of the seed

__global__ void __launch_bounds__(BLOCK)
roundBoundKernel(const double2* __restrict__ pts, const int* __restrict__ seeds,
                 const long long* __restrict__ off, const int* __restrict__ nbrCnt,
                 const int* __restrict__ nbr, int* __restrict__ card,
                 const int* __restrict__ boundList, const float invCell,
                 const int useGrid, const WorkState* ws) {
    __shared__ int grid[BOUND_DIM][BOUND_DIM + 1];
    __shared__ int warpMax[BLOCK / WARP];
    const int count = ws->boundCount;
    for (int t = blockIdx.x; t < count; t += gridDim.x) {
        const int i = boundList[t];
        const int K = nbrCnt[i];
        if (!useGrid) {
            if (threadIdx.x == 0) card[i] = 1 + K;
            continue;
        }
        for (int c = threadIdx.x; c < BOUND_DIM * (BOUND_DIM + 1); c += blockDim.x) {
            (&grid[0][0])[c] = 0;
        }
        __syncthreads();
        const double2 sp = pts[seeds[i]];
        const int* list = nbr + off[i];
        for (int j = threadIdx.x; j < K; j += blockDim.x) {
            const double2 p = pts[list[j]];
            const float u = (float)(p.x - sp.x) * invCell;
            const float v = (float)(p.y - sp.y) * invCell;
            const int cx = min(max((int)floorf(u) + BOUND_CENTER, 0), BOUND_DIM - 1);
            const int cy = min(max((int)floorf(v) + BOUND_CENTER, 0), BOUND_DIM - 1);
            atomicAdd(&grid[cy][cx], 1);
        }
        __syncthreads();
        // inclusive prefix sums: rows, then columns
        for (int r = threadIdx.x; r < BOUND_DIM; r += blockDim.x) {
            int acc = 0;
            for (int c = 0; c < BOUND_DIM; ++c) { acc += grid[r][c]; grid[r][c] = acc; }
        }
        __syncthreads();
        for (int c = threadIdx.x; c < BOUND_DIM; c += blockDim.x) {
            int acc = 0;
            for (int r = 0; r < BOUND_DIM; ++r) { acc += grid[r][c]; grid[r][c] = acc; }
        }
        __syncthreads();
        // windows of BOUND_W cells per axis containing the seed's cell
        const int loMin = BOUND_CENTER - BOUND_W + 1;
        int best = 0;
        for (int wdx = threadIdx.x; wdx < BOUND_W * BOUND_W; wdx += blockDim.x) {
            const int x0 = max(loMin + wdx % BOUND_W, 0);
            const int y0 = max(loMin + wdx / BOUND_W, 0);
            const int x1 = min(x0 + BOUND_W - 1, BOUND_DIM - 1);
            const int y1 = min(y0 + BOUND_W - 1, BOUND_DIM - 1);
            int sum = grid[y1][x1];
            if (x0 > 0) sum -= grid[y1][x0 - 1];
            if (y0 > 0) sum -= grid[y0 - 1][x1];
            if (x0 > 0 && y0 > 0) sum += grid[y0 - 1][x0 - 1];
            best = max(best, sum);
        }
        #pragma unroll
        for (int o = WARP / 2; o > 0; o >>= 1) best = max(best, __shfl_xor_sync(0xffffffffu, best, o));
        if ((threadIdx.x & (WARP - 1)) == 0) warpMax[threadIdx.x / WARP] = best;
        __syncthreads();
        if (threadIdx.x == 0) {
            for (int w = 1; w < BLOCK / WARP; ++w) best = max(best, warpMax[w]);
            card[i] = min(1 + K, 1 + best);
        }
        __syncthreads();
    }
}

// Queue keys of the seeds without exact cardinality (0 otherwise), separately
// for seeds handled by a warp and by a block; exact ones feed the best key
__global__ void roundQueueKernel(const unsigned char* __restrict__ clustered,
                                 const int* __restrict__ seeds, const int M,
                                 const int* __restrict__ nbrCnt, const int* __restrict__ card,
                                 const unsigned char* __restrict__ exact,
                                 unsigned long long* __restrict__ keysSmall,
                                 int* __restrict__ idxSmall,
                                 unsigned long long* __restrict__ keysBig,
                                 int* __restrict__ idxBig, WorkState* ws) {
    const int lane = threadIdx.x & (WARP - 1);
    unsigned long long localExact = 0ull;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < M; i += gridDim.x * blockDim.x) {
        const int seed = seeds[i];
        unsigned long long qkey = 0ull;
        if (!clustered[seed]) {
            const unsigned long long key = packKey(card[i], seed);
            if (exact[i]) localExact = key > localExact ? key : localExact;
            else qkey = key;
        }
        const bool big = nbrCnt[i] > KWARP;
        keysSmall[i] = big ? 0ull : qkey;
        keysBig[i] = big ? qkey : 0ull;
        idxSmall[i] = i;
        idxBig[i] = i;
    }
    #pragma unroll
    for (int o = WARP / 2; o > 0; o >>= 1) {
        const unsigned long long other = __shfl_xor_sync(0xffffffffu, localExact, o);
        localExact = other > localExact ? other : localExact;
    }
    if (lane == 0 && localExact) atomicMax(&ws->best, localExact);
}

// Evaluate queued seeds (exact cardinality) in decreasing order of their upper
// bound, as long as the bound can still beat the best key. Entries [start, count)
// are processed; a first small launch establishes a good best key for the second.
__global__ void roundEvaluateKernel(const double2* __restrict__ pts,
                                    const int* __restrict__ seeds,
                                    const long long* __restrict__ off,
                                    const int* __restrict__ nbrCnt, const int* __restrict__ nbr,
                                    int* __restrict__ card, unsigned char* __restrict__ exact,
                                    const unsigned long long* __restrict__ pendKeys,
                                    const int* __restrict__ pendIdx, const int start,
                                    const int count, const int stage,
                                    const DistParams dp, Scratch scratch,
                                    const long long scratchStride, int* __restrict__ memb,
                                    WorkState* ws) {
    const int lane = threadIdx.x & (WARP - 1);
    const long long warpGlobal = (blockIdx.x * (long long)blockDim.x + threadIdx.x) / WARP;
    Scratch s;
    s.xy = scratch.xy + warpGlobal * scratchStride;
    s.mdf = scratch.mdf + warpGlobal * scratchStride;
    s.md = scratch.md + warpGlobal * scratchStride;
    s.id = scratch.id + warpGlobal * scratchStride;

    while (true) {
        int t = 0;
        unsigned long long cur = 0ull;
        if (lane == 0) {
            t = start + atomicAdd(&ws->counter[stage], 1);
            cur = *((volatile unsigned long long*)&ws->best);
        }
        t = __shfl_sync(0xffffffffu, t, 0);
        cur = __shfl_sync(0xffffffffu, cur, 0);
        if (t >= count) break;
        const unsigned long long key = pendKeys[t];
        if (key <= cur) {
            // the queue is sorted by bound only: stop once no bound can win
            if ((key | 0xffffffffull) <= cur) break;
            continue;
        }
        const int i = pendIdx[t];
        const int seed = seeds[i];
        const int c = greedyCluster(seed, nbr + off[i], nbrCnt[i], pts, dp, s, memb + off[i] + i);
        if (lane == 0) {
            card[i] = c;
            exact[i] = 1;
            atomicMax(&ws->best, packKey(c, seed));
        }
    }
}

// Block-per-seed evaluation of the queue of seeds with many neighbours
// (same protocol as roundEvaluateKernel). Shared memory holds up to smemCap entries;
// larger candidate sets use the block's global scratch slot.
struct BigScratch {
    float2* xy;
    float* mdf;
    double* md;
    int* id;
};

__device__ __forceinline__ void bigSlot(const BigScratch& g, const long long slot, const long long stride,
                                        const int K, const int smemCap, unsigned char* smem,
                                        float2*& xy, float*& mdf, double*& md, int*& idv) {
    if (K <= smemCap) {
        xy = reinterpret_cast<float2*>(smem);
        mdf = reinterpret_cast<float*>(smem + sizeof(float2) * smemCap);
    } else {
        xy = g.xy + slot * stride;
        mdf = g.mdf + slot * stride;
    }
    md = g.md + slot * stride;
    idv = g.id + slot * stride;
}

__global__ void __launch_bounds__(BIG_BLOCK)
roundEvaluateBigKernel(const double2* __restrict__ pts, const int* __restrict__ seeds,
                       const long long* __restrict__ off, const int* __restrict__ nbrCnt,
                       const int* __restrict__ nbr, int* __restrict__ card,
                       unsigned char* __restrict__ exact,
                       const unsigned long long* __restrict__ pendKeys,
                       const int* __restrict__ pendIdx, const int count, const DistParams dp,
                       BigScratch g, const long long stride, const int smemCap,
                       int* __restrict__ memb, WorkState* ws) {
    extern __shared__ __align__(16) unsigned char smem[];
    __shared__ BlockReduce red;
    __shared__ int sT;
    __shared__ unsigned long long sCur;
    while (true) {
        if (threadIdx.x == 0) {
            sT = atomicAdd(&ws->counterBig, 1);
            sCur = *((volatile unsigned long long*)&ws->best);
        }
        __syncthreads();
        const int t = sT;
        const unsigned long long cur = sCur;
        __syncthreads();
        if (t >= count) break;
        const unsigned long long key = pendKeys[t];
        if (key <= cur) {
            if ((key | 0xffffffffull) <= cur) break;
            continue;
        }
        const int i = pendIdx[t];
        const int seed = seeds[i];
        const int K = nbrCnt[i];
        float2* xy; float* mdf; double* md; int* idv;
        bigSlot(g, blockIdx.x, stride, K, smemCap, smem, xy, mdf, md, idv);
        const int c = blockGreedyCluster(seed, nbr + off[i], K, pts, dp, xy, mdf, md, idv,
                                         memb + off[i] + i, red);
        if (threadIdx.x == 0) {
            card[i] = c;
            exact[i] = 1;
            atomicMax(&ws->best, packKey(c, seed));
        }
        __syncthreads();
    }
}

// Publish the local best key and the members of its cluster (stored in insertion
// order by the evaluation) to mapped host memory. The local best seed always
// belongs to this rank, and the rank owning the global best publishes it.
__global__ void publishBestKernel(const int* __restrict__ memb, const long long* __restrict__ off,
                                  const int nranks, const WorkState* ws, HostStaging* h) {
    const unsigned long long key = ws->best;
    if (key != 0ull) {
        const int c = static_cast<int>(key >> 32);
        const int seed = static_cast<int>(0xffffffffu - static_cast<unsigned>(key & 0xffffffffull));
        const int li = seed / nranks;
        const int* src = memb + off[li] + li;
        for (int t = threadIdx.x; t < c; t += blockDim.x) h->members[t] = src[t];
    }
    if (threadIdx.x == 0) h->localKey = key;
}

// Start of a round: mark the members of the cluster selected in the previous
// round (key and members read from mapped host memory) and reset the round state
__global__ void markClusteredKernel(const HostStaging* h, const float2* __restrict__ ptsf,
                                    unsigned char* __restrict__ clustered, WorkState* ws) {
    const unsigned long long key = h->globalKey;
    const int n = static_cast<int>(key >> 32);
    for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < n; t += gridDim.x * blockDim.x) {
        clustered[h->members[t]] = 1;
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        ws->best = 0ull;
        ws->counter[0] = 0;
        ws->counter[1] = 0;
        ws->counterBig = 0;
        ws->boundCount = 0;
        if (n > 0) {
            ws->firstRound = 0;
            ws->lastSeed = ptsf[h->members[0]];  // members start with the seed
        }
    }
}

// Main QT clustering algorithm (distributed over MPI ranks, one GPU per rank).
// Every rank returns the full, identical cluster list.
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold, HostStaging* h) {
    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    const int N = static_cast<int>(points.size());
    std::vector<Cluster> clusters;
    if (N == 0) return clusters;

    // Smallest squared distance whose square root reaches the threshold
    double thr2 = threshold * threshold;
    while (thr2 > 0.0 && std::sqrt(thr2) >= threshold) thr2 = std::nextafter(thr2, 0.0);
    while (std::sqrt(thr2) < threshold) thr2 = std::nextafter(thr2, HUGE_VAL);

    // Single precision filter parameters. Candidate coordinates are taken relative
    // to the seed (|.| < threshold), which bounds the float error of squared
    // distances by ~thr2 * 2^-17; eps = thr2 * 2^-12 leaves a wide safety margin.
    // Filtering is disabled (eps = inf) outside a safe float range.
    DistParams dp;
    dp.thr2 = thr2;
    const bool floatOk = thr2 > 1e-20 && thr2 < 1e20;
    dp.thr2f = floatOk ? static_cast<float>(thr2) : HUGE_VALF;
    dp.eps = floatOk ? static_cast<float>(thr2 * 0x1p-12) : HUGE_VALF;

    // Bounding box margins for single precision coordinate tests:
    // conversion errors are bounded by maxCoord * 2^-24 per coordinate.
    double maxCoord = 0.0;
    #pragma omp parallel for reduction(max:maxCoord) if(N > 65536)
    for (int i = 0; i < N; ++i) {
        maxCoord = std::max(maxCoord, std::max(std::fabs(points[i].x), std::fabs(points[i].y)));
    }
    const double box = threshold * (1.0 + 0x1p-16) + maxCoord * 0x1p-16;
    const bool boxOk = box < 1e30 && std::isfinite(maxCoord);
    const float boxf = boxOk ? static_cast<float>(box) : HUGE_VALF;
    const float box2f = boxOk ? static_cast<float>(2.0 * box) : HUGE_VALF;
    std::vector<float2> pointsf(N);
    #pragma omp parallel for if(N > 65536)
    for (int i = 0; i < N; ++i) {
        pointsf[i] = make_float2(static_cast<float>(points[i].x), static_cast<float>(points[i].y));
    }

    // Seeds are distributed cyclically over ranks (generated groups are contiguous)
    std::vector<int> seeds;
    for (int i = rank; i < N; i += nranks) seeds.push_back(i);
    const int M = static_cast<int>(seeds.size());
    const int Mm = std::max(M, 1);

    int device = 0, numSMs = 1, maxThreadsPerSM = 2048;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, device));
    CUDA_CHECK(cudaDeviceGetAttribute(&maxThreadsPerSM, cudaDevAttrMaxThreadsPerMultiProcessor, device));
    const int fullGridWarps = numSMs * (maxThreadsPerSM / WARP);

    static_assert(sizeof(Point) == sizeof(double2), "Point layout");
    double2* d_pts = nullptr;
    float2* d_ptsf = nullptr;
    unsigned char* d_clustered = nullptr;
    int* d_seeds = nullptr;
    int* d_cnt = nullptr;
    int* d_card = nullptr;
    unsigned char* d_exact = nullptr;
    long long* d_off = nullptr;
    int* d_nbr = nullptr;
    int* d_memb = nullptr;
    WorkState* d_ws = nullptr;
    // queues: [0]/[1] warp-sized seeds (unsorted/sorted), [2]/[3] block-sized seeds
    unsigned long long* d_pendKeys[4] = {nullptr, nullptr, nullptr, nullptr};
    int* d_pendIdx[4] = {nullptr, nullptr, nullptr, nullptr};

    CUDA_CHECK(cudaMalloc(&d_pts, sizeof(double2) * N));
    CUDA_CHECK(cudaMemcpy(d_pts, points.data(), sizeof(double2) * N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_ptsf, sizeof(float2) * N));
    CUDA_CHECK(cudaMemcpy(d_ptsf, pointsf.data(), sizeof(float2) * N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_clustered, N));
    CUDA_CHECK(cudaMemset(d_clustered, 0, N));
    CUDA_CHECK(cudaMalloc(&d_ws, sizeof(WorkState)));
    CUDA_CHECK(cudaMalloc(&d_seeds, sizeof(int) * Mm));
    CUDA_CHECK(cudaMalloc(&d_cnt, sizeof(int) * Mm));
    CUDA_CHECK(cudaMalloc(&d_card, sizeof(int) * Mm));
    CUDA_CHECK(cudaMalloc(&d_exact, Mm));
    CUDA_CHECK(cudaMemset(d_exact, 0, Mm));
    CUDA_CHECK(cudaMalloc(&d_off, sizeof(long long) * (Mm + 1)));
    for (int b = 0; b < 4; ++b) {
        CUDA_CHECK(cudaMalloc(&d_pendKeys[b], sizeof(unsigned long long) * Mm));
        CUDA_CHECK(cudaMalloc(&d_pendIdx[b], sizeof(int) * Mm));
    }

    // Build the neighbour lists of the local seeds
    std::vector<int> cnt(M, 0);
    std::vector<long long> off(M + 1, 0);
    int maxK = 1;
    const int seedBlocks = std::max(1, std::min((M + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK,
                                                fullGridWarps / WARPS_PER_BLOCK));
    if (M > 0) {
        CUDA_CHECK(cudaMemcpy(d_seeds, seeds.data(), sizeof(int) * M, cudaMemcpyHostToDevice));
        countNeighborsKernel<<<seedBlocks, BLOCK>>>(d_pts, d_ptsf, N, d_seeds, M, thr2, boxf, d_cnt);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(cnt.data(), d_cnt, sizeof(int) * M, cudaMemcpyDeviceToHost));
        for (int i = 0; i < M; ++i) {
            off[i + 1] = off[i] + cnt[i];
            maxK = std::max(maxK, cnt[i]);
        }
        CUDA_CHECK(cudaMemcpy(d_off, off.data(), sizeof(long long) * (M + 1), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMalloc(&d_nbr, sizeof(int) * std::max<long long>(off[M], 1)));
    if (M > 0) {
        fillNeighborsKernel<<<seedBlocks, BLOCK>>>(d_pts, d_ptsf, N, d_seeds, M, thr2, boxf, d_off, d_nbr);
        CUDA_CHECK(cudaGetLastError());
    }

    // Per-warp scratch space for seeds with at most KWARP neighbours
    const long long stride = std::min(maxK, KWARP);
    const long long slots = std::min<long long>(fullGridWarps, Mm) / WARPS_PER_BLOCK * WARPS_PER_BLOCK +
                            WARPS_PER_BLOCK;
    const int evalBlocks = static_cast<int>(slots / WARPS_PER_BLOCK);
    Scratch scratch;
    CUDA_CHECK(cudaMalloc(&scratch.xy, sizeof(float2) * slots * stride));
    CUDA_CHECK(cudaMalloc(&scratch.mdf, sizeof(float) * slots * stride));
    CUDA_CHECK(cudaMalloc(&scratch.md, sizeof(double) * slots * stride));
    CUDA_CHECK(cudaMalloc(&scratch.id, sizeof(int) * slots * stride));

    // Per-block scratch for seeds with more neighbours: candidate coordinates and
    // float distances in shared memory (global memory beyond smemCap entries)
    const bool hasBig = maxK > KWARP;
    int maxSmemOptin = 48 * 1024;
    CUDA_CHECK(cudaDeviceGetAttribute(&maxSmemOptin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    const int smemCap = std::max(1, std::min(maxK, (maxSmemOptin - 2048) / static_cast<int>(sizeof(float2) + sizeof(float))));
    const size_t bigSmem = static_cast<size_t>(smemCap) * (sizeof(float2) + sizeof(float));
    int bigBlocks = 1;
    BigScratch bigScratch = {nullptr, nullptr, nullptr, nullptr};
    const long long bigStride = maxK;
    if (hasBig) {
        CUDA_CHECK(cudaFuncSetAttribute(roundEvaluateBigKernel,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)bigSmem));
        int perSM = 1;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&perSM, roundEvaluateBigKernel,
                                                                 BIG_BLOCK, bigSmem));
        bigBlocks = std::max(1, std::min(perSM * numSMs, Mm));
        const long long n = static_cast<long long>(bigBlocks) * bigStride;
        CUDA_CHECK(cudaMalloc(&bigScratch.md, sizeof(double) * n));
        CUDA_CHECK(cudaMalloc(&bigScratch.id, sizeof(int) * n));
        if (maxK > smemCap) {
            CUDA_CHECK(cudaMalloc(&bigScratch.xy, sizeof(float2) * n));
            CUDA_CHECK(cudaMalloc(&bigScratch.mdf, sizeof(float) * n));
        }
    }

    // Temporary storage for sorting the queues by bound (descending); only the
    // bits holding the cardinality bound need to be sorted
    int sortEndBit = 33;
    while (sortEndBit < 64 && (1ll << (sortEndBit - 32)) <= N + 1) ++sortEndBit;
    void* d_sortTemp = nullptr;
    size_t sortTempBytes = 0;
    CUDA_CHECK(cub::DeviceRadixSort::SortPairsDescending(
        nullptr, sortTempBytes, d_pendKeys[0], d_pendKeys[1], d_pendIdx[0], d_pendIdx[1], Mm,
        32, sortEndBit));
    CUDA_CHECK(cudaMalloc(&d_sortTemp, std::max<size_t>(sortTempBytes, 1)));

    // Member lists of evaluated seeds (insertion order), parallel to the neighbour
    // lists: seed i stores up to 1 + #neighbours entries at off[i] + i
    CUDA_CHECK(cudaMalloc(&d_memb, sizeof(int) * (off[M] + M + 1)));

    HostStaging* hd = nullptr;  // device view of the staging area
    CUDA_CHECK(cudaHostGetDevicePointer(&hd, h, 0));
    h->localKey = 0ull;
    h->globalKey = 0ull;

    const int prepBlocks = std::max(1, std::min((M + BLOCK - 1) / BLOCK, fullGridWarps / WARPS_PER_BLOCK));
    const int boundBlocks = std::max(1, std::min(M, numSMs * 8));
    const int stage1Warps = static_cast<int>(std::min<long long>(slots, numSMs));
    const double invCellD = BOUND_G / threshold;
    const bool useGrid = floatOk && std::isfinite(invCellD) && invCellD < 1e30;
    const float invCell = useGrid ? static_cast<float>(invCellD) : 0.0f;
    int* d_boundList = nullptr;
    CUDA_CHECK(cudaMalloc(&d_boundList, sizeof(int) * Mm));

    std::vector<unsigned char> clustered(N, 0);
    {
        WorkState init = {};
        init.firstRound = 1;
        init.lastSeed = make_float2(0.0f, 0.0f);
        CUDA_CHECK(cudaMemcpy(d_ws, &init, sizeof(WorkState), cudaMemcpyHostToDevice));
    }

    // One round as a CUDA graph (all per-round parameters live in device or mapped memory)
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cudaGraphExec_t roundGraph = nullptr;
    if (M > 0) {
        cudaGraph_t graph;
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        markClusteredKernel<<<32, BLOCK, 0, stream>>>(hd, d_ptsf, d_clustered, d_ws);
        // refresh changed neighbourhoods and their bounds, queue seeds without exact cardinality
        roundRefreshKernel<<<prepBlocks, BLOCK, 0, stream>>>(d_ptsf, d_clustered, d_seeds, M, d_off,
                                                            d_cnt, d_nbr, d_exact, box2f, d_boundList,
                                                            d_ws);
        roundBoundKernel<<<boundBlocks, BLOCK, 0, stream>>>(d_pts, d_seeds, d_off, d_cnt, d_nbr, d_card,
                                                           d_boundList, invCell, useGrid ? 1 : 0, d_ws);
        roundQueueKernel<<<prepBlocks, BLOCK, 0, stream>>>(d_clustered, d_seeds, M, d_cnt, d_card,
                                                          d_exact, d_pendKeys[0], d_pendIdx[0],
                                                          d_pendKeys[2], d_pendIdx[2], d_ws);
        // process the queues by decreasing bound: block-sized seeds first, then
        // the most promising warp-sized seeds, then the rest (pruned by the best key)
        if (hasBig) {
            CUDA_CHECK(cub::DeviceRadixSort::SortPairsDescending(
                d_sortTemp, sortTempBytes, d_pendKeys[2], d_pendKeys[3], d_pendIdx[2], d_pendIdx[3],
                M, 32, sortEndBit, stream));
            roundEvaluateBigKernel<<<bigBlocks, BIG_BLOCK, bigSmem, stream>>>(
                d_pts, d_seeds, d_off, d_cnt, d_nbr, d_card, d_exact, d_pendKeys[3], d_pendIdx[3],
                M, dp, bigScratch, bigStride, smemCap, d_memb, d_ws);
        }
        CUDA_CHECK(cub::DeviceRadixSort::SortPairsDescending(
            d_sortTemp, sortTempBytes, d_pendKeys[0], d_pendKeys[1], d_pendIdx[0], d_pendIdx[1],
            M, 32, sortEndBit, stream));
        const int stage1 = std::min(M, stage1Warps);
        roundEvaluateKernel<<<(stage1 + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, BLOCK, 0, stream>>>(
            d_pts, d_seeds, d_off, d_cnt, d_nbr, d_card, d_exact, d_pendKeys[1], d_pendIdx[1],
            0, stage1, 0, dp, scratch, stride, d_memb, d_ws);
        roundEvaluateKernel<<<evalBlocks, BLOCK, 0, stream>>>(
            d_pts, d_seeds, d_off, d_cnt, d_nbr, d_card, d_exact, d_pendKeys[1], d_pendIdx[1],
            stage1, M, 1, dp, scratch, stride, d_memb, d_ws);
        // local best key and its members go straight to host memory
        publishBestKernel<<<1, BLOCK, 0, stream>>>(d_memb, d_off, nranks, d_ws, hd);
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&roundGraph, graph, 0));
        CUDA_CHECK(cudaGraphDestroy(graph));
    }

    while (true) {
        unsigned long long localBest = 0ull;
        if (M > 0) {
            CUDA_CHECK(cudaGraphLaunch(roundGraph, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            localBest = *reinterpret_cast<volatile unsigned long long*>(&h->localKey);
        }

        unsigned long long globalBest = 0ull;
        MPI_Allreduce(&localBest, &globalBest, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
        if (globalBest == 0ull) break;  // every point is clustered
        const int bestCard = static_cast<int>(globalBest >> 32);
        const int bestSeed = static_cast<int>(0xffffffffu - static_cast<unsigned>(globalBest & 0xffffffffull));

        if (bestCard <= 1) {
            // No remaining point has an unclustered neighbour within the threshold:
            // every remaining point becomes a singleton cluster, in index order.
            for (int p = 0; p < N; ++p) {
                if (clustered[p]) continue;
                Cluster cluster;
                cluster.seed_point = p;
                cluster.members.push_back(p);
                clusters.push_back(std::move(cluster));
                clustered[p] = 1;
            }
            break;
        }

        // The owner of the global best already holds its members (its local best);
        // the next round starts by marking them
        if (nranks > 1) MPI_Bcast(h->members, bestCard, MPI_INT, bestSeed % nranks, MPI_COMM_WORLD);
        h->globalKey = globalBest;

        Cluster cluster;
        cluster.seed_point = bestSeed;
        cluster.members.assign(h->members, h->members + bestCard);
        for (int k = 0; k < bestCard; ++k) clustered[h->members[k]] = 1;
        clusters.push_back(std::move(cluster));
    }
    if (roundGraph) CUDA_CHECK(cudaGraphExecDestroy(roundGraph));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaFree(d_sortTemp);
    cudaFree(d_boundList);
    cudaFree(scratch.xy); cudaFree(scratch.mdf); cudaFree(scratch.md); cudaFree(scratch.id);
    cudaFree(bigScratch.xy); cudaFree(bigScratch.mdf); cudaFree(bigScratch.md); cudaFree(bigScratch.id);
    for (int b = 0; b < 4; ++b) { cudaFree(d_pendKeys[b]); cudaFree(d_pendIdx[b]); }
    cudaFree(d_pts); cudaFree(d_ptsf); cudaFree(d_clustered); cudaFree(d_seeds);
    cudaFree(d_cnt); cudaFree(d_card); cudaFree(d_exact); cudaFree(d_off);
    cudaFree(d_nbr); cudaFree(d_memb); cudaFree(d_ws);
    return clusters;
}

// Validation: check that clusters satisfy the QT clustering properties
bool validateClusters(const std::vector<Cluster>& clusters,
                     const std::vector<Point>& points,
                     const double threshold) {
    bool valid = true;
    
    printf("Validating clusters:\n");
    
    // Compute cluster diameters (max distance between any two points) in parallel
    const long long numClusters = static_cast<long long>(clusters.size());
    std::vector<double> diameters(clusters.size(), 0.0);
    #pragma omp parallel for schedule(dynamic, 1)
    for (long long c = 0; c < numClusters; ++c) {
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
    const long long numMembership = static_cast<long long>(membership.size());
    #pragma omp parallel for reduction(+:clustered_count)
    for (long long i = 0; i < numMembership; ++i) {
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

static int runBenchmark(int argc, char** argv, const int rank);

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Bind each rank on a node to one GPU (round robin)
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0, localSize = 1;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_size(localComm, &localSize);
    MPI_Comm_free(&localComm);

    // Share the node's cores among the local ranks (unless set explicitly) and
    // start the OpenMP thread pool outside the timed region
    if (!getenv("OMP_NUM_THREADS")) {
        omp_set_num_threads(std::max(1, omp_get_num_procs() / localSize));
    }
    #pragma omp parallel
    { /* warm-up */ }
    int numDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));
    if (numDevices <= 0) {
        fprintf(stderr, "No CUDA device available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % numDevices));
    CUDA_CHECK(cudaFree(nullptr)); // initialize the context outside the timed region

    // Only rank 0 produces output
    if (rank != 0) {
        if (!freopen("/dev/null", "w", stdout)) { /* ignore */ }
    }

    int ret = runBenchmark(argc, argv, rank);
    MPI_Bcast(&ret, 1, MPI_INT, 0, MPI_COMM_WORLD);
    fflush(stdout);
    MPI_Finalize();
    return ret;
}

static int runBenchmark(int argc, char** argv, const int rank) {
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
    
    // Host staging area (mapped pinned memory) for the per-round GPU <-> MPI exchange
    HostStaging* staging = nullptr;
    CUDA_CHECK(cudaHostAlloc(&staging, sizeof(HostStaging) + sizeof(int) * num_points,
                             cudaHostAllocMapped));

    // Perform QT clustering
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_start = std::chrono::high_resolution_clock::now();
    
    const std::vector<Cluster> clusters = qtClustering(points, threshold, staging);
    
    auto cluster_end = std::chrono::high_resolution_clock::now();
    long cluster_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start).count();
    MPI_Allreduce(MPI_IN_PLACE, &cluster_time_ms, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);
    auto cluster_time = std::chrono::milliseconds(cluster_time_ms);
    
    CUDA_CHECK(cudaFreeHost(staging));

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
    if (printResults && rank == 0) {
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
    if (validate && rank == 0) {
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
