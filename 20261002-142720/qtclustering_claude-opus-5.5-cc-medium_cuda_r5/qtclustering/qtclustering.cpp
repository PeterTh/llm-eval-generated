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
#include <functional>
#include <type_traits>
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
// Semantics are identical to the sequential greedy QT algorithm:
//  * A candidate cluster grown from a seed can only contain points whose
//    distance to the seed is < threshold, so each seed (one thread block)
//    only works on its unclustered neighbourhood. Ties are broken by the
//    smallest point index, exactly like the original linear scan.
//  * The max distance of each candidate to the cluster members is maintained
//    incrementally (max is order independent, so values are bit-identical).
//  * Distances are compared via squared distances; sqrt is correctly rounded
//    and monotone, so max(sqrt) == sqrt(max) and sqrt(a) < t <=> a < S*
//    (S* computed exactly on the host). Ties of the sqrt values are resolved
//    exactly by evaluating sqrt when two squared distances are ulps apart.
//  * The greedy growth of a seed only depends on the points it selects:
//    removing any other point never changes an argmin. Therefore after a
//    cluster is removed, only seeds whose candidate cluster contains a
//    removed point need to be recomputed; all other results are cached
//    (every seed keeps its member list in a device memory slot).
//  * The iterations run entirely on the GPU (a CUDA graph per iteration);
//    the host only polls for completion.
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                       \
                    cudaGetErrorString(err__), __FILE__, __LINE__);            \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

static const int NO_POS = 0x7fffffff;
static const unsigned long long INF_BITS = 0x7ff0000000000000ULL;  // +inf

// Lexicographic minimum of (sqrt(value), point index) for squared distances
// given as IEEE bits; returns true if (b, q) was taken. Different squared
// distances can only have the same sqrt if they are a few ulps apart, so
// sqrt is evaluated only in that rare case.
__device__ __forceinline__ bool combineBest(unsigned long long& a, int& p,
                                            const unsigned long long b, const int q) {
    const unsigned long long lo = a < b ? a : b;
    const unsigned long long diff = (a < b ? b : a) - lo;
    bool tie = diff == 0;
    if (!tie && diff <= 16 && lo != INF_BITS) {
        tie = __dsqrt_rn(__longlong_as_double(static_cast<long long>(a))) ==
              __dsqrt_rn(__longlong_as_double(static_cast<long long>(b)));
    }
    if (tie) {
        a = lo;
        if (q < p) { p = q; return true; }
    } else if (b < a) {
        a = b;
        p = q;
        return true;
    }
    return false;
}

// Warp reductions (hardware redux on sm_80+, shuffles otherwise)
__device__ __forceinline__ unsigned warpMinU32(unsigned v) {
#if __CUDA_ARCH__ >= 800
    return __reduce_min_sync(0xffffffffu, v);
#else
    for (int o = 16; o > 0; o >>= 1) v = min(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
#endif
}

__device__ __forceinline__ int warpSumI32(int v) {
#if __CUDA_ARCH__ >= 800
    return __reduce_add_sync(0xffffffffu, v);
#else
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
#endif
}

// Warp-wide lexicographic minimum of (sqrt(value), index); all lanes receive
// the result. Uses 32-bit hardware reductions: min of the high words, min of
// the low words among those, then min index among sqrt-equal values.
__device__ __forceinline__ void warpArgmin(unsigned long long& v, int& i) {
    const unsigned hi = static_cast<unsigned>(v >> 32);
    const unsigned lo = static_cast<unsigned>(v);
    const unsigned mhi = warpMinU32(hi);
    const unsigned mlo = warpMinU32(hi == mhi ? lo : 0xffffffffu);
    const unsigned long long m = (static_cast<unsigned long long>(mhi) << 32) | mlo;
    bool eq = v == m;
    if (!eq && v - m <= 16 && m != INF_BITS) {
        eq = __dsqrt_rn(__longlong_as_double(static_cast<long long>(v))) ==
             __dsqrt_rn(__longlong_as_double(static_cast<long long>(m)));
    }
    i = static_cast<int>(warpMinU32(eq ? static_cast<unsigned>(i) : 0x7fffffffu));
    v = m;
}

// Float approximation (rounded towards zero, rel. error < 2^-20) of a
// non-negative double given by its bits, using integer instructions only.
__device__ __forceinline__ float bitsToFloatLower(const unsigned long long b) {
    const unsigned hi = static_cast<unsigned>(b >> 32);  // sign bit is zero
    // re-bias the exponent (1023 -> 127) and keep 20 mantissa bits
    const float f = __uint_as_float((hi - ((1023u - 127u) << 20)) << 3);
    // out of range: 0 (= no filtering)
    return (hi < ((1023u - 126u) << 20) || hi >= ((1023u + 61u) << 20)) ? 0.0f : f;
}

// Squared distance, evaluated exactly like the host build of distance()
// (contracted: fma(dx, dx, dy*dy)).
__device__ __forceinline__ double sqDist(const double2 a, const double2 b) {
    const double dx = __dsub_rn(a.x, b.x);
    const double dy = __dsub_rn(a.y, b.y);
    return __fma_rn(dx, dx, __dmul_rn(dy, dy));
}

__device__ __forceinline__ float sqDistF(const float ax, const float ay,
                                         const float bx, const float by) {
    const float dx = ax - bx;
    const float dy = ay - by;
    return dx * dx + dy * dy;
}

struct Ctrl {
    int remaining;   // unclustered points
    int nClusters;   // clusters found so far
    int resPos;      // members written to the result buffer
    int done;
    int needHost;    // the best seed has no stored member list
    int bestSeed;
};

struct KernelArgs {
    const double2* pts;            // exact coordinates
    const float2* ptsf;            // single precision copy (pre-filtering)
    unsigned char* clustered;
    int N;
    double Sstar;                  // dist < threshold <=> sqDist < Sstar
    float SstarF;                  // conservative float bound of Sstar
    int* card;                     // candidate cluster cardinality per seed
    int* pool;                     // member list storage
    long long* slotOff;            // per seed: member list slot (-1: none)
    Ctrl* ctrl;
    // results
    int* resMembers;
    int* resSeed;
    int* resLen;
    // per size class work lists
    int* clsSeeds;                 // nClasses x N
    int* clsCnt;                   // nClasses
    int* clsWork;                  // nClasses
    const unsigned char* clsOf;    // class of each seed
    int nClasses;
};

struct ClassArgs {
    int c;                         // class id
    int cap;                       // max. candidates (even)
    int useShared;
    unsigned char* gscratch;       // per-block candidate lists if not in smem
};

// One block per seed (persistent, dynamically scheduled): grows the
// candidate cluster of each seed of a size class and stores its members.
template <int B>
__global__ void __launch_bounds__(B)
clusterKernel(const KernelArgs args, const ClassArgs ca) {
    constexpr int NW = B / 32;
    extern __shared__ __align__(16) unsigned char smem[];
    __shared__ int warpSums[NW];
    __shared__ unsigned long long redA[2][NW];  // double buffered per step
    __shared__ int redP[2][NW];
    __shared__ int sW;

    if (args.ctrl->done || args.ctrl->needHost) return;
    const int nSeeds = args.clsCnt[ca.c];
    if (static_cast<int>(blockIdx.x) >= nSeeds) return;
    const int* __restrict__ seeds = args.clsSeeds + static_cast<size_t>(ca.c) * args.N;

    const double2* __restrict__ pts = args.pts;
    const float2* __restrict__ ptsf = args.ptsf;
    const unsigned char* __restrict__ clustered = args.clustered;
    int* __restrict__ pool = args.pool;
    const int N = args.N;
    const int cap = ca.cap;

    unsigned char* base = ca.useShared
                              ? smem
                              : ca.gscratch + static_cast<size_t>(blockIdx.x) * cap * 20;
    unsigned long long* maxs = reinterpret_cast<unsigned long long*>(base);
    float2* xy = reinterpret_cast<float2*>(maxs + cap);
    int* idx = reinterpret_cast<int*>(xy + cap);

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wid = tid >> 5;
    const unsigned long long sBits = static_cast<unsigned long long>(__double_as_longlong(args.Sstar));
    const float SstarF = args.SstarF;

    while (true) {
        if (tid == 0) sW = atomicAdd(args.clsWork + ca.c, 1);
        __syncthreads();
        const int w = sW;
        __syncthreads();
        if (w >= nSeeds) break;
        const int seed = seeds[w];
        const double2 sp = pts[seed];
        const float2 spf = ptsf[seed];
        const long long slot = args.slotOff[seed];

        // ---- gather unclustered neighbours of the seed, in index order ----
        int cnt = 0;
        for (int b0 = 0; b0 < N; b0 += B) {
            const int i = b0 + tid;
            bool f = false;
            double s = 0.0;
            float2 q = make_float2(0.f, 0.f);
            if (i < N && i != seed && !clustered[i]) {
                q = ptsf[i];
                if (sqDistF(q.x, q.y, spf.x, spf.y) <= SstarF) {
                    s = sqDist(pts[i], sp);
                    f = s < args.Sstar;
                }
            }
            const unsigned bal = __ballot_sync(0xffffffffu, f);
            if (lane == 0) warpSums[wid] = __popc(bal);
            __syncthreads();
            int off = cnt, total = 0;
            #pragma unroll
            for (int k = 0; k < NW; ++k) {
                const int v = warpSums[k];
                if (k < wid) off += v;
                total += v;
            }
            if (f) {
                const int pos = off + __popc(bal & ((1u << lane) - 1u));
                maxs[pos] = static_cast<unsigned long long>(__double_as_longlong(s));
                xy[pos] = q;
                idx[pos] = i;
            }
            cnt += total;
            __syncthreads();
        }

        // ---- greedy growth ----
        // maxs[p] holds the max over members of the squared distance from
        // candidate p (stored as raw IEEE bits: for non-negative doubles the
        // integer order equals the floating point order). Since sqrt is
        // monotone, sqrt(maxs[p]) is exactly the original max distance; the
        // argmin compares (sqrt(maxs), point index) lexicographically, which
        // is exactly the original "first minimal candidate" rule.
        // Thread t owns the slots t, t+B, t+2B, ... and removes dead
        // candidates by swapping in its last live entry (order is irrelevant).
        int myN = cnt > tid ? (cnt - tid + B - 1) / B : 0;
        int size = 1;
        int par = 0;
        if (tid == 0 && slot >= 0) pool[slot] = seed;
        bool first = true;
        double2 cp = sp;
        float cxf = 0.f, cyf = 0.f;
        while (true) {
            unsigned long long bb = INF_BITS;
            int bi = NO_POS;   // point index of the local best
            int bk = -1;       // its slot
            unsigned long long bbLim = INF_BITS;  // values above cannot win or tie
            for (int k = 0; k < myN;) {
                const int p = k * B + tid;
                unsigned long long ab = maxs[p];
                if (!first) {
                    const float2 q = xy[p];
                    const float sf = sqDistF(q.x, q.y, cxf, cyf);
                    // float pre-filter with a safe error margin
                    if (sf > bitsToFloatLower(ab) * (1.0f - 1e-5f) - 1e-4f) {
                        const unsigned long long db = static_cast<unsigned long long>(
                            __double_as_longlong(sqDist(pts[idx[p]], cp)));
                        if (db > ab) {
                            ab = db;
                            maxs[p] = ab;
                        }
                    }
                }
                if (ab >= sBits) {  // dead for good: remove
                    --myN;
                    const int lp = myN * B + tid;
                    if (k != myN) {
                        maxs[p] = maxs[lp];
                        xy[p] = xy[lp];
                        idx[p] = idx[lp];
                    }
                    continue;  // process the swapped-in entry
                }
                if (ab <= bbLim) {
                    const int pi = idx[p];
                    if (combineBest(bb, bi, ab, pi)) bk = k;
                    bbLim = bb + 16;
                }
                ++k;
            }
            // block-wide argmin: one barrier per step; every warp reduces
            // the per-warp partials redundantly
            warpArgmin(bb, bi);
            if (lane == 0) { redA[par][wid] = bb; redP[par][wid] = bi; }
            __syncthreads();
            unsigned long long rb = lane < NW ? redA[par][lane] : INF_BITS;
            int ri = lane < NW ? redP[par][lane] : NO_POS;
            warpArgmin(rb, ri);
            par ^= 1;
            const int win = ri;
            if (win == NO_POS) break;
            // the owner removes the new member from its candidate slots
            if (bk >= 0 && idx[bk * B + tid] == win) {
                --myN;
                const int p = bk * B + tid, lp = myN * B + tid;
                if (bk != myN) {
                    maxs[p] = maxs[lp];
                    xy[p] = xy[lp];
                    idx[p] = idx[lp];
                }
            }
            if (tid == 0 && slot >= 0) pool[slot + size] = win;
            ++size;
            first = false;
            cp = pts[win];
            const float2 cq = ptsf[win];
            cxf = cq.x;
            cyf = cq.y;
        }

        if (tid == 0) args.card[seed] = size;
        __syncthreads();
    }
}

// Selects the best candidate cluster (largest, then smallest seed index),
// appends it to the results and marks its members as clustered. Also resets
// the work lists for the next iteration. Single block.
__global__ void __launch_bounds__(1024) selectKernel(const KernelArgs args) {
    __shared__ unsigned long long red[32];
    Ctrl* ctrl = args.ctrl;
    if (ctrl->done || ctrl->needHost) return;
    const int tid = threadIdx.x;
    if (tid < args.nClasses) {
        args.clsCnt[tid] = 0;
        args.clsWork[tid] = 0;
    }
    unsigned long long key = 0;
    for (int i = tid; i < args.N; i += blockDim.x) {
        if (!args.clustered[i]) {
            const unsigned long long k =
                (static_cast<unsigned long long>(static_cast<unsigned>(args.card[i])) << 32) |
                (0xffffffffu - static_cast<unsigned>(i));
            key = k > key ? k : key;
        }
    }
    for (int o = 16; o > 0; o >>= 1) {
        const unsigned long long v = __shfl_xor_sync(0xffffffffu, key, o);
        key = v > key ? v : key;
    }
    if ((tid & 31) == 0) red[tid >> 5] = key;
    __syncthreads();
    key = (tid & 31) < static_cast<int>(blockDim.x >> 5) ? red[tid & 31] : 0;
    for (int o = 16; o > 0; o >>= 1) {
        const unsigned long long v = __shfl_xor_sync(0xffffffffu, key, o);
        key = v > key ? v : key;
    }
    if (key == 0) {  // nothing left (cannot happen while remaining > 0)
        if (tid == 0) ctrl->done = 1;
        return;
    }
    const int best = static_cast<int>(0xffffffffu - static_cast<unsigned>(key & 0xffffffffu));
    const int k = static_cast<int>(key >> 32);
    const long long slot = args.slotOff[best];
    if (slot < 0) {
        if (tid == 0) { ctrl->needHost = 1; ctrl->bestSeed = best; }
        return;
    }
    const int pos = ctrl->resPos;
    for (int t = tid; t < k; t += blockDim.x) {
        const int m = args.pool[slot + t];
        args.resMembers[pos + t] = m;
        args.clustered[m] = 1;
    }
    __syncthreads();
    if (tid == 0) {
        const int c = ctrl->nClusters;
        args.resSeed[c] = best;
        args.resLen[c] = k;
        ctrl->nClusters = c + 1;
        ctrl->resPos = pos + k;
        ctrl->remaining -= k;
        if (ctrl->remaining <= 0) ctrl->done = 1;
    }
}

// One warp per unclustered seed: a stored candidate cluster becomes stale iff
// it contains a newly clustered point (seeds without a stored list are always
// recomputed). Stale seeds are appended to the work list of their class.
__global__ void staleKernel(const KernelArgs args) {
    if (args.ctrl->done || args.ctrl->needHost) return;
    const int lane = threadIdx.x & 31;
    const int warpsTotal = gridDim.x * (blockDim.x >> 5);
    for (int i = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5); i < args.N; i += warpsTotal) {
        if (args.clustered[i]) continue;
        const long long slot = args.slotOff[i];
        bool hit = slot < 0;
        if (!hit) {
            const int len = args.card[i];
            for (int t = lane; t < len && !hit; t += 32) hit = args.clustered[args.pool[slot + t]] != 0;
            hit = __any_sync(0xffffffffu, hit);
        }
        if (hit && lane == 0) {
            const int c = args.clsOf[i];
            const int p = atomicAdd(args.clsCnt + c, 1);
            args.clsSeeds[static_cast<size_t>(c) * args.N + p] = i;
        }
    }
}

// Number of neighbours (dist < threshold) of each point (excluding itself);
// one warp per point.
__global__ void degreeKernel(const double2* __restrict__ pts, const float2* __restrict__ ptsf,
                             const int N, const double Sstar, const float SstarF,
                             int* __restrict__ deg) {
    const int lane = threadIdx.x & 31;
    const int warpsTotal = gridDim.x * (blockDim.x >> 5);
    for (int i = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5); i < N; i += warpsTotal) {
        const float2 pf = ptsf[i];
        const double2 pd = pts[i];
        int d = 0;
        for (int j = lane; j < N; j += 32) {
            const float2 q = ptsf[j];
            if (sqDistF(q.x, q.y, pf.x, pf.y) <= SstarF && j != i && sqDist(pts[j], pd) < Sstar) ++d;
        }
        d = warpSumI32(d);
        if (lane == 0) deg[i] = d;
    }
}

// Seeds are processed in size classes by neighbourhood size, so that the
// shared memory per block matches the work and many blocks fit on an SM.
struct SizeClass {
    int cap = 2;          // max. candidates (even)
    int block = 64;       // threads per seed
    int grid = 1;         // resident blocks
    size_t smem = 0;
    bool useShared = true;
    unsigned char* scratch = nullptr;   // global candidate lists if !useShared
    cudaStream_t stream = nullptr;
    cudaEvent_t joined = nullptr;
};

// Call f with the compile-time block size
template <typename F>
static void dispatchBlock(int bs, F&& f) {
    switch (bs) {
        case 64:   f(std::integral_constant<int, 64>{}); break;
        case 128:  f(std::integral_constant<int, 128>{}); break;
        case 256:  f(std::integral_constant<int, 256>{}); break;
        default:   f(std::integral_constant<int, 512>{}); break;
    }
}

struct DeviceInfo {
    int multiProcessorCount = 1;
    int sharedMemPerBlockOptin = 48 * 1024;
};

static std::vector<SizeClass> makeClasses(const int maxdeg, const DeviceInfo& prop) {
    const size_t maxOptin = static_cast<size_t>(prop.sharedMemPerBlockOptin) - 1024;
    const int top = std::max(2, (maxdeg + 1) & ~1);
    std::vector<int> caps;
    for (int c = 256; c < top; c *= 2) caps.push_back(c);
    caps.push_back(top);
    for (int B : {64, 128, 256, 512}) {
        dispatchBlock(B, [&](auto bs) {
            CUDA_CHECK(cudaFuncSetAttribute(clusterKernel<decltype(bs)::value>,
                                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            static_cast<int>(maxOptin)));
        });
    }
    std::vector<SizeClass> classes;
    for (const int cap : caps) {
        SizeClass sc;
        sc.cap = cap;
        sc.block = cap <= 512 ? 64 : (cap <= 1024 ? 128 : (cap <= 2048 ? 256 : 512));
        const size_t need = static_cast<size_t>(cap) * 20;
        sc.useShared = need <= maxOptin;
        sc.smem = sc.useShared ? need : 0;
        int perSM = 0;
        dispatchBlock(sc.block, [&](auto bs) {
            constexpr int B = decltype(bs)::value;
            CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&perSM, clusterKernel<B>, B, sc.smem));
        });
        sc.grid = std::max(1, perSM) * prop.multiProcessorCount;
        if (!sc.useShared) {
            size_t freeMem = 0, totalMem = 0;
            CUDA_CHECK(cudaMemGetInfo(&freeMem, &totalMem));
            sc.grid = static_cast<int>(std::max<size_t>(1, std::min<size_t>(sc.grid, freeMem / 8 / need)));
            CUDA_CHECK(cudaMalloc(&sc.scratch, need * sc.grid));
        }
        CUDA_CHECK(cudaStreamCreateWithFlags(&sc.stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&sc.joined, cudaEventDisableTiming));
        classes.push_back(sc);
    }
    return classes;
}

// Main QT clustering algorithm (GPU)
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<Cluster> clusters;
    if (N == 0) return clusters;

    // sqrt(s) < threshold  <=>  s < Sstar (sqrt is correctly rounded & monotone)
    double Sstar = threshold * threshold;
    while (Sstar > 0.0 && std::sqrt(std::nextafter(Sstar, 0.0)) >= threshold)
        Sstar = std::nextafter(Sstar, 0.0);
    while (std::sqrt(Sstar) < threshold)
        Sstar = std::nextafter(Sstar, std::numeric_limits<double>::infinity());
    // Conservative single precision pre-filter bound
    const float SstarF = std::isinf(Sstar) ? std::numeric_limits<float>::infinity()
                                           : static_cast<float>(Sstar * (1.0 + 1e-5) + 1e-4);

    std::vector<double2> hp(N);
    std::vector<float2> hpf(N);
    for (int i = 0; i < N; ++i) {
        hp[i] = make_double2(points[i].x, points[i].y);
        hpf[i] = make_float2(static_cast<float>(points[i].x), static_cast<float>(points[i].y));
    }

    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    DeviceInfo prop;
    CUDA_CHECK(cudaDeviceGetAttribute(&prop.multiProcessorCount, cudaDevAttrMultiProcessorCount, dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&prop.sharedMemPerBlockOptin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    double2* d_pts;
    float2* d_ptsf;
    unsigned char *d_clustered, *d_clsOf;
    int *d_deg, *d_card, *d_resMembers, *d_resSeed, *d_resLen;
    long long* d_slotOff;
    Ctrl* d_ctrl;
    CUDA_CHECK(cudaMalloc(&d_pts, N * sizeof(double2)));
    CUDA_CHECK(cudaMalloc(&d_ptsf, N * sizeof(float2)));
    CUDA_CHECK(cudaMalloc(&d_clustered, N));
    CUDA_CHECK(cudaMalloc(&d_clsOf, N));
    CUDA_CHECK(cudaMalloc(&d_deg, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_card, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_resMembers, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_resSeed, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_resLen, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_slotOff, N * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_ctrl, sizeof(Ctrl)));
    CUDA_CHECK(cudaMemcpyAsync(d_pts, hp.data(), N * sizeof(double2), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_ptsf, hpf.data(), N * sizeof(float2), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemsetAsync(d_clustered, 0, N, stream));

    // Neighbourhood sizes -> size classes and member list slots
    degreeKernel<<<std::min((N + 7) / 8, prop.multiProcessorCount * 16), 256, 0, stream>>>(
        d_pts, d_ptsf, N, Sstar, SstarF, d_deg);
    CUDA_CHECK(cudaGetLastError());
    std::vector<int> hdeg(N);
    CUDA_CHECK(cudaMemcpyAsync(hdeg.data(), d_deg, N * sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const int maxdeg = *std::max_element(hdeg.begin(), hdeg.end());

    std::vector<SizeClass> classes = makeClasses(maxdeg, prop);
    const int nClasses = static_cast<int>(classes.size());
    auto classOf = [&](int deg) {
        for (int c = 0; c < nClasses; ++c)
            if (deg <= classes[c].cap) return c;
        return nClasses - 1;
    };

    // Member list slots: seed i owns deg[i] + 1 entries (its maximal cluster
    // size); if memory is short, the remaining seeds get no slot and are
    // recomputed whenever needed. One extra slot serves as host fallback.
    size_t freeMem = 0, totalMem = 0;
    CUDA_CHECK(cudaMemGetInfo(&freeMem, &totalMem));
    const unsigned long long budget = freeMem / 2 / sizeof(int);
    std::vector<long long> hslot(N, -1);
    unsigned long long used = static_cast<unsigned long long>(maxdeg) + 1;  // fallback slot first
    for (int i = 0; i < N; ++i) {
        const unsigned long long need = static_cast<unsigned long long>(hdeg[i]) + 1;
        if (used + need <= budget) {
            hslot[i] = static_cast<long long>(used);
            used += need;
        }
    }
    int* d_pool;
    CUDA_CHECK(cudaMalloc(&d_pool, used * sizeof(int)));
    CUDA_CHECK(cudaMemcpyAsync(d_slotOff, hslot.data(), N * sizeof(long long), cudaMemcpyHostToDevice, stream));

    // Initial work lists: every seed, most expensive first
    int *d_clsSeeds, *d_clsCnt, *d_clsWork;
    CUDA_CHECK(cudaMalloc(&d_clsSeeds, static_cast<size_t>(nClasses) * N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_clsCnt, nClasses * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_clsWork, nClasses * sizeof(int)));
    std::vector<unsigned char> hcls(N);
    std::vector<std::vector<int>> lists(nClasses);
    {
        std::vector<int> order(N);
        for (int i = 0; i < N; ++i) order[i] = i;
        std::stable_sort(order.begin(), order.end(), [&](int x, int y) { return hdeg[x] > hdeg[y]; });
        for (const int i : order) {
            hcls[i] = static_cast<unsigned char>(classOf(hdeg[i]));
            lists[hcls[i]].push_back(i);
        }
    }
    std::vector<int> hcnt(nClasses);
    for (int c = 0; c < nClasses; ++c) {
        hcnt[c] = static_cast<int>(lists[c].size());
        if (hcnt[c] > 0)
            CUDA_CHECK(cudaMemcpyAsync(d_clsSeeds + static_cast<size_t>(c) * N, lists[c].data(),
                                       hcnt[c] * sizeof(int), cudaMemcpyHostToDevice, stream));
    }
    CUDA_CHECK(cudaMemcpyAsync(d_clsOf, hcls.data(), N, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_clsCnt, hcnt.data(), nClasses * sizeof(int), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemsetAsync(d_clsWork, 0, nClasses * sizeof(int), stream));

    Ctrl hctrlStorage{N, 0, 0, 0, 0, -1};
    Ctrl* hctrl = &hctrlStorage;
    CUDA_CHECK(cudaMemcpyAsync(d_ctrl, hctrl, sizeof(Ctrl), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    KernelArgs args;
    args.pts = d_pts;
    args.ptsf = d_ptsf;
    args.clustered = d_clustered;
    args.N = N;
    args.Sstar = Sstar;
    args.SstarF = SstarF;
    args.card = d_card;
    args.pool = d_pool;
    args.slotOff = d_slotOff;
    args.ctrl = d_ctrl;
    args.resMembers = d_resMembers;
    args.resSeed = d_resSeed;
    args.resLen = d_resLen;
    args.clsSeeds = d_clsSeeds;
    args.clsCnt = d_clsCnt;
    args.clsWork = d_clsWork;
    args.clsOf = d_clsOf;
    args.nClasses = nClasses;

    // One QT iteration as a CUDA graph: evaluate the stale seeds of all size
    // classes concurrently, select the best cluster, find stale seeds.
    // All kernels become no-ops once the clustering is complete.
    const int staleBlocks = std::max(1, std::min((N + 7) / 8, prop.multiProcessorCount * 8));
    cudaEvent_t fork;
    CUDA_CHECK(cudaEventCreateWithFlags(&fork, cudaEventDisableTiming));
    cudaGraph_t graph;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    CUDA_CHECK(cudaEventRecord(fork, stream));
    for (int c = 0; c < nClasses; ++c) {
        const SizeClass& sc = classes[c];
        CUDA_CHECK(cudaStreamWaitEvent(sc.stream, fork, 0));
        ClassArgs ca;
        ca.c = c;
        ca.cap = sc.cap;
        ca.useShared = sc.useShared ? 1 : 0;
        ca.gscratch = sc.scratch;
        dispatchBlock(sc.block, [&](auto bs) {
            constexpr int B = decltype(bs)::value;
            clusterKernel<B><<<sc.grid, B, sc.smem, sc.stream>>>(args, ca);
        });
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(sc.joined, sc.stream));
        CUDA_CHECK(cudaStreamWaitEvent(stream, sc.joined, 0));
    }
    selectKernel<<<1, 1024, 0, stream>>>(args);
    CUDA_CHECK(cudaGetLastError());
    staleKernel<<<staleBlocks, 256, 0, stream>>>(args);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    cudaGraphExec_t graphExec;
    CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, 0));

    int batch = 4;
    while (true) {
        for (int t = 0; t < batch; ++t) CUDA_CHECK(cudaGraphLaunch(graphExec, stream));
        CUDA_CHECK(cudaMemcpyAsync(hctrl, d_ctrl, sizeof(Ctrl), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if (hctrl->done) break;
        if (hctrl->needHost) {
            // The best seed has no stored member list (memory exhausted):
            // recompute it into the fallback slot, then continue.
            const int s = hctrl->bestSeed;
            const long long fallback = 0;
            const int c = hcls[s];
            CUDA_CHECK(cudaMemcpyAsync(d_slotOff + s, &fallback, sizeof(long long), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemsetAsync(d_clsCnt, 0, nClasses * sizeof(int), stream));
            CUDA_CHECK(cudaMemsetAsync(d_clsWork, 0, nClasses * sizeof(int), stream));
            const int one = 1;
            CUDA_CHECK(cudaMemcpyAsync(d_clsSeeds + static_cast<size_t>(c) * N, &s, sizeof(int), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(d_clsCnt + c, &one, sizeof(int), cudaMemcpyHostToDevice, stream));
            hctrl->needHost = 0;
            CUDA_CHECK(cudaMemcpyAsync(d_ctrl, hctrl, sizeof(Ctrl), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            batch = 1;
            continue;
        }
        batch = std::min(batch * 2, 64);
    }

    // Collect the results
    const int nClusters = hctrl->nClusters;
    std::vector<int> resMembers(hctrl->resPos), resSeed(nClusters), resLen(nClusters);
    if (hctrl->resPos > 0)
        CUDA_CHECK(cudaMemcpy(resMembers.data(), d_resMembers, hctrl->resPos * sizeof(int), cudaMemcpyDeviceToHost));
    if (nClusters > 0) {
        CUDA_CHECK(cudaMemcpy(resSeed.data(), d_resSeed, nClusters * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(resLen.data(), d_resLen, nClusters * sizeof(int), cudaMemcpyDeviceToHost));
    }
    clusters.resize(nClusters);
    for (int c = 0, pos = 0; c < nClusters; ++c) {
        clusters[c].seed_point = resSeed[c];
        clusters[c].members.assign(resMembers.begin() + pos, resMembers.begin() + pos + resLen[c]);
        pos += resLen[c];
    }

    CUDA_CHECK(cudaGraphExecDestroy(graphExec));
    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaEventDestroy(fork));
    for (SizeClass& sc : classes) {
        if (sc.scratch) CUDA_CHECK(cudaFree(sc.scratch));
        CUDA_CHECK(cudaStreamDestroy(sc.stream));
        CUDA_CHECK(cudaEventDestroy(sc.joined));
    }
    CUDA_CHECK(cudaFree(d_pool));
    CUDA_CHECK(cudaFree(d_clsSeeds));
    CUDA_CHECK(cudaFree(d_clsCnt));
    CUDA_CHECK(cudaFree(d_clsWork));
    CUDA_CHECK(cudaFree(d_pts));
    CUDA_CHECK(cudaFree(d_ptsf));
    CUDA_CHECK(cudaFree(d_clustered));
    CUDA_CHECK(cudaFree(d_clsOf));
    CUDA_CHECK(cudaFree(d_deg));
    CUDA_CHECK(cudaFree(d_card));
    CUDA_CHECK(cudaFree(d_resMembers));
    CUDA_CHECK(cudaFree(d_resSeed));
    CUDA_CHECK(cudaFree(d_resLen));
    CUDA_CHECK(cudaFree(d_slotOff));
    CUDA_CHECK(cudaFree(d_ctrl));
    CUDA_CHECK(cudaStreamDestroy(stream));

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
    
    // Initialize the CUDA context (and load all kernels) outside of the timed region
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
