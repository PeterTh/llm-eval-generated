// QT Clustering Benchmark - Hybrid MPI + OpenMP + CUDA Version
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.

#include <algorithm>
#include <cfloat>
#include <climits>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <set>
#include <type_traits>
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
// Hybrid MPI + OpenMP + CUDA QT clustering
//
// Key observations used (all preserve the exact semantics of the original):
//  * A candidate c can only join seed s's cluster if dist(s,c) < threshold, so
//    each seed only needs to look at its threshold-neighbourhood (found with a
//    uniform grid of cell size >= threshold).
//  * The max distance of a candidate to the current cluster is maintained
//    incrementally (max with the distance to the newly added member); the
//    selection rule (smallest max distance, ties -> smallest index) is kept.
//  * The greedy candidate cluster of seed s only depends on the unclustered
//    points within threshold of s.  After a cluster is removed, only seeds
//    within threshold of one of its members need to be recomputed; all other
//    cached cardinalities stay valid.
//  * If the best cardinality is 1, all remaining points are singletons and are
//    emitted in ascending index order (exactly what the original does).
//  * Distances are compared via their exact squared values (the rounded sqrt
//    is monotonic), with an exact treatment of ties after rounding; a float
//    pre-filter with a rigorous error margin skips double-precision work
//    whose outcome is already certain.
//
// Seeds are distributed round-robin over MPI ranks (one GPU per rank); the
// global best seed (max cardinality, lowest index) is found with an
// MPI_Allreduce and its member list (kept on the owner's GPU) is broadcast.
// Each candidate cluster is grown by a cooperative thread group (warp or
// block, sized by the neighbourhood) on the GPU.  Host-side loops use OpenMP.
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err_ = (call);                                             \
        if (err_ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                        \
                    cudaGetErrorString(err_), __FILE__, __LINE__);             \
            MPI_Abort(MPI_COMM_WORLD, 1);                                      \
        }                                                                      \
    } while (0)

// Squared distance exactly as the host distance() computes it before the
// square root.  Depending on the host compiler's floating-point contraction,
// distance() evaluates dx*dx + dy*dy as fma(dx, dx, dy*dy) (GCC with
// -march=native on FMA hardware), fma(dy, dy, dx*dx), or unfused; the variant
// is detected at runtime (detectDistanceMode) and reproduced bit-exactly.
// Since a correctly rounded sqrt is monotonic, comparisons of distances (max,
// < threshold) can be done on these squared values; only exact ties after
// rounding of the square root need special care (see growClusters).
__constant__ int cDistMode;

__device__ __forceinline__ double dist2(double ax, double ay, double bx, double by) {
    const double dx = __dsub_rn(ax, bx);
    const double dy = __dsub_rn(ay, by);
    if (cDistMode == 0) return __fma_rn(dx, dx, __dmul_rn(dy, dy));
    if (cDistMode == 1) return __fma_rn(dy, dy, __dmul_rn(dx, dx));
    return __dadd_rn(__dmul_rn(dx, dx), __dmul_rn(dy, dy));
}

// Which evaluation of dx*dx + dy*dy the host distance() uses (see dist2).
static int detectDistanceMode() {
    unsigned st = 12345u;
    for (int k = 0; k < 100000; ++k) {
        const Point a{rand_r(&st) / static_cast<double>(RAND_MAX) * 20.0,
                      rand_r(&st) / static_cast<double>(RAND_MAX) * 20.0};
        const Point b{rand_r(&st) / static_cast<double>(RAND_MAX) * 20.0,
                      rand_r(&st) / static_cast<double>(RAND_MAX) * 20.0};
        const double ref = distance(a, b);
        volatile double dx = a.x - b.x, dy = a.y - b.y;
        volatile double xx = dx * dx, yy = dy * dy;
        const double v0 = std::sqrt(std::fma(dx, dx, yy));
        const double v1 = std::sqrt(std::fma(dy, dy, xx));
        volatile double sum = xx + yy;
        const double v2 = std::sqrt(sum);
        if (v0 == v1 && v1 == v2) continue;   // not discriminating
        if (ref == v0 && ref != v1 && ref != v2) return 0;
        if (ref == v1 && ref != v0 && ref != v2) return 1;
        if (ref == v2 && ref != v0 && ref != v1) return 2;
    }
    return 0;
}

// Non-negative doubles order like their bit patterns -> cheap integer keys.
__device__ __forceinline__ long long dkey(double v) { return __double_as_longlong(v); }
static constexpr long long kDead = LLONG_MAX;

struct DevData {
    const double* px;        // coordinates by original index
    const double* py;
    const double* sx;        // coordinates sorted by grid cell
    const double* sy;
    const int* sidx;         // original index of sorted entry
    const int* cellStart;    // G*G+1 offsets
    const int* pcell;        // cell (cx | cy << 16) of original index
    const unsigned char* clustered;
    long long s2Key;         // key of S2: dist < threshold  <=>  dist2 < S2
    float s2Hi;              // float >= S2
    float delta;             // error bound of float squared distances (inf: no filter)
    int G;
};

// Where growClusters writes its results.
struct SeedOutput {
    int* cards;               // cards[seed]
    int* members;             // member storage (nullptr: not needed)
    const long long* memOff;  // per-seed offset into members (nullptr: single list)
};

__device__ __forceinline__ void argminMerge(long long& k, int& i, long long ok, int oi) {
    if (ok < k || (ok == k && oi < i)) { k = ok; i = oi; }
}

__device__ __forceinline__ void warpArgmin(long long& k, int& i) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const long long ok = __shfl_xor_sync(0xffffffffu, k, off);
        const int oi = __shfl_xor_sync(0xffffffffu, i, off);
        argminMerge(k, i, ok, oi);
    }
}

struct GroupShared {
    long long k[33];
    int i[33];
    float fx[33], fy[33];
    int cnt;
};

// Group-wide lexicographic argmin of (k, i); also broadcasts the (float)
// coordinates of the winning candidate.  Candidate indices are unique, so the
// owner of the winner is identified by its index.  TPS == 32 -> one warp,
// otherwise the whole block.
template <int TPS>
__device__ __forceinline__ void groupArgmin(long long& k, int& i, float& fx, float& fy, GroupShared& sh) {
    const unsigned full = 0xffffffffu;
    int mine = i;
    warpArgmin(k, i);
    unsigned own = __ballot_sync(full, mine == i);
    int src = own ? __ffs(own) - 1 : 0;
    fx = __shfl_sync(full, fx, src);
    fy = __shfl_sync(full, fy, src);
    if constexpr (TPS > 32) {
        const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
        if (lane == 0) { sh.k[w] = k; sh.i[w] = i; sh.fx[w] = fx; sh.fy[w] = fy; }
        __syncthreads();
        if (w == 0) {
            k = (lane < TPS / 32) ? sh.k[lane] : kDead;
            i = (lane < TPS / 32) ? sh.i[lane] : INT_MAX;
            mine = i;
            warpArgmin(k, i);
            own = __ballot_sync(full, mine == i);
            src = own ? __ffs(own) - 1 : 0;
            if (lane == 0) { sh.k[32] = k; sh.i[32] = i; sh.fx[32] = sh.fx[src]; sh.fy[32] = sh.fy[src]; }
        }
        __syncthreads();
        k = sh.k[32]; i = sh.i[32]; fx = sh.fx[32]; fy = sh.fy[32];
    }
}

template <int TPS>
__device__ __forceinline__ bool groupAny(bool p) {
    if constexpr (TPS > 32) return __syncthreads_or(p) != 0;
    else return __any_sync(0xffffffffu, p);
}

template <int TPS>
__device__ __forceinline__ void groupSync() {
    if constexpr (TPS > 32) __syncthreads(); else __syncwarp();
}

// Gather all unclustered points within threshold of the seed (excluding the
// seed).  Calls emit(sortedPos, slot) for each; returns the count.
template <int TPS, typename Emit>
__device__ __forceinline__ int gatherNeighbours(const DevData& D, int seed, double seedX, double seedY,
                                                int lane, int* sharedCnt, Emit emit) {
    const int pc = D.pcell[seed];
    const int cx = pc & 0xffff, cy = pc >> 16;
    int cnt = 0;
    if constexpr (TPS > 32) {
        if (threadIdx.x == 0) *sharedCnt = 0;
        __syncthreads();
    }
    for (int yy = max(cy - 1, 0); yy <= min(cy + 1, D.G - 1); ++yy) {
        const int rowBase = yy * D.G;
        const int s = D.cellStart[rowBase + max(cx - 1, 0)];
        const int e = D.cellStart[rowBase + min(cx + 1, D.G - 1) + 1];
        for (int base = s; base < e; base += TPS) {
            const int p = base + lane;
            bool ok = false;
            if (p < e) {
                const int j = D.sidx[p];
                if (j != seed && !D.clustered[j]) {
                    ok = dkey(dist2(D.sx[p], D.sy[p], seedX, seedY)) < D.s2Key;
                }
            }
            if constexpr (TPS == 32) {
                const unsigned mask = __ballot_sync(0xffffffffu, ok);
                if (ok) emit(p, cnt + __popc(mask & ((1u << lane) - 1u)));
                cnt += __popc(mask);
            } else {
                if (ok) emit(p, atomicAdd(sharedCnt, 1));
            }
        }
    }
    if constexpr (TPS > 32) {
        __syncthreads();
        cnt = *sharedCnt;
    }
    return cnt;
}

// Candidate state: float coordinates relative to the seed (for the cheap
// filter), exact key of the squared max distance to the cluster, a float lower
// bound of it, and the point index.
struct Cand {
    float fx, fy, lo;
    long long k;
    int i;
};

__device__ __forceinline__ void loadCand(Cand& c, const DevData& D, int p, double seedX, double seedY) {
    const double x = D.sx[p], y = D.sy[p];
    const double d2 = dist2(x, y, seedX, seedY);
    c.fx = static_cast<float>(x - seedX);
    c.fy = static_cast<float>(y - seedY);
    c.k = dkey(d2);
    c.lo = __double2float_rd(d2);
    c.i = D.sidx[p];
}

// Add member w (float coords relative to seed wfx/wfy, exact wx/wy) to the
// cluster: update the candidate's max distance (exactly; the float filter
// only skips work whose outcome is certain).
__device__ __forceinline__ void updateCand(Cand& c, const DevData& D, int w, float wfx, float wfy,
                                           double wx, double wy) {
    if (c.k == kDead) return;
    if (c.i == w) { c.k = kDead; return; }
    const float dx = c.fx - wfx, dy = c.fy - wfy;
    const float fd2 = fmaf(dx, dx, dy * dy);
    if (fd2 + D.delta <= c.lo) return;                    // new distance <= current max
    if (fd2 - D.delta >= D.s2Hi) { c.k = kDead; return; } // new distance >= threshold
    const double d2 = dist2(D.px[c.i], D.py[c.i], wx, wy);
    const long long k2 = dkey(d2);
    if (k2 > c.k) { c.k = k2; c.lo = __double2float_rd(d2); }
    if (c.k >= D.s2Key) c.k = kDead;
}

__device__ __forceinline__ void bestOf(const Cand& c, long long& bk, int& bi, float& bfx, float& bfy) {
    if (c.k != kDead && (c.k < bk || (c.k == bk && c.i < bi))) { bk = c.k; bi = c.i; bfx = c.fx; bfy = c.fy; }
}

// The original compares rounded distances sqrt(.): different squared values
// can round to the same distance, and then the lower index wins.  Such
// candidates have squared values within a few ulps of the minimum.
__device__ __forceinline__ bool tieSuspect(const Cand& c, long long bk, int bi) {
    return c.i < bi && c.k != kDead && c.k <= bk + 64;
}
__device__ __forceinline__ bool tieExact(const Cand& c, long long bk, int bi) {
    return tieSuspect(c, bk, bi) &&
           sqrt(__longlong_as_double(c.k)) == sqrt(__longlong_as_double(bk));
}

// Grow the candidate clusters of a list of seeds (exactly the greedy process
// of the original generateCandidateCluster).  One group of TPS threads per
// seed.  Candidates are held in registers (K per thread) when !GLOBAL, or in
// a per-block global scratch area otherwise.  The number of seeds is read
// from countPtr if given (device-side work lists), else countConst.  Seeds
// that are already clustered are skipped.
template <int TPS, int K, bool GLOBAL>
__global__ void __launch_bounds__(TPS == 32 ? 128 : TPS)
growClusters(const DevData D, const int* __restrict__ seeds, const int* countPtr, int countConst,
             SeedOutput out, Cand* gScratch, int scratchCap) {
    constexpr int BLOCK = TPS == 32 ? 128 : TPS;   // TPS > 32: one group per block
    constexpr int GPB = BLOCK / TPS;
    constexpr int STAGE = GLOBAL ? 1 : TPS * K;
    __shared__ int stage[GPB][STAGE];
    __shared__ GroupShared sh;

    const int gid = threadIdx.x / TPS;
    const int lane = threadIdx.x % TPS;
    const int nSeeds = countPtr ? *countPtr : countConst;
    Cand* gc = GLOBAL ? gScratch + static_cast<size_t>(blockIdx.x) * scratchCap : nullptr;

    for (int g = blockIdx.x * GPB + gid; g < nSeeds; g += gridDim.x * GPB) {
        const int seed = seeds[g];
        if (D.clustered[seed]) continue;   // uniform within the group
        const double seedX = D.px[seed], seedY = D.py[seed];
        int* mem = out.members ? (out.memOff ? out.members + out.memOff[seed] : out.members) : nullptr;
        int* myStage = stage[gid];

        int cnt;
        if constexpr (GLOBAL) {
            cnt = gatherNeighbours<TPS>(D, seed, seedX, seedY, lane, &sh.cnt,
                [&](int p, int slot) { loadCand(gc[slot], D, p, seedX, seedY); });
            __syncthreads();
        } else {
            cnt = gatherNeighbours<TPS>(D, seed, seedX, seedY, lane, &sh.cnt,
                [&](int p, int slot) { myStage[slot] = p; });
            groupSync<TPS>();
        }

        Cand cr[GLOBAL ? 1 : K];
        long long bk = kDead;
        int bi = INT_MAX;
        float bfx = 0.f, bfy = 0.f;
        if constexpr (!GLOBAL) {
#pragma unroll
            for (int t = 0; t < K; ++t) {
                const int e = lane + t * TPS;
                if (e < cnt) {
                    loadCand(cr[t], D, myStage[e], seedX, seedY);
                    bestOf(cr[t], bk, bi, bfx, bfy);
                } else {
                    cr[t].k = kDead; cr[t].i = INT_MAX; cr[t].fx = cr[t].fy = cr[t].lo = 0.f;
                }
            }
        } else {
            for (int e = lane; e < cnt; e += TPS) bestOf(gc[e], bk, bi, bfx, bfy);
        }

        int count = 1;
        if (mem && lane == 0) mem[0] = seed;
        while (true) {
            groupArgmin<TPS>(bk, bi, bfx, bfy, sh);
            if (bi == INT_MAX) break;   // no candidate keeps the diameter below threshold
            // Exact tie handling on rounded distances (rare)
            bool suspect = false;
            if constexpr (!GLOBAL) {
#pragma unroll
                for (int t = 0; t < K; ++t) suspect |= tieSuspect(cr[t], bk, bi);
            } else {
                for (int e = lane; e < cnt; e += TPS) suspect |= tieSuspect(gc[e], bk, bi);
            }
            if (groupAny<TPS>(suspect)) {
                long long tk = kDead;
                int ti = INT_MAX;
                float tfx = 0.f, tfy = 0.f;
                if constexpr (!GLOBAL) {
#pragma unroll
                    for (int t = 0; t < K; ++t) {
                        if (tieExact(cr[t], bk, bi) && cr[t].i < ti) { ti = cr[t].i; tfx = cr[t].fx; tfy = cr[t].fy; tk = 0; }
                    }
                } else {
                    for (int e = lane; e < cnt; e += TPS) {
                        if (tieExact(gc[e], bk, bi) && gc[e].i < ti) { ti = gc[e].i; tfx = gc[e].fx; tfy = gc[e].fy; tk = 0; }
                    }
                }
                groupArgmin<TPS>(tk, ti, tfx, tfy, sh);
                if (ti != INT_MAX) { bi = ti; bfx = tfx; bfy = tfy; }
            }

            const int w = bi;
            const float wfx = bfx, wfy = bfy;
            const double wx = D.px[w], wy = D.py[w];
            if (mem && lane == 0) mem[count] = w;
            ++count;
            bk = kDead; bi = INT_MAX;
            if constexpr (!GLOBAL) {
#pragma unroll
                for (int t = 0; t < K; ++t) {
                    updateCand(cr[t], D, w, wfx, wfy, wx, wy);
                    bestOf(cr[t], bk, bi, bfx, bfy);
                }
            } else {
                for (int e = lane; e < cnt; e += TPS) {
                    Cand c = gc[e];
                    if (c.k == kDead) continue;
                    updateCand(c, D, w, wfx, wfy, wx, wy);
                    gc[e].k = c.k; gc[e].lo = c.lo;
                    bestOf(c, bk, bi, bfx, bfy);
                }
            }
        }
        if (lane == 0) out.cards[seed] = count;
        groupSync<TPS>();
    }
}

// Number of points within threshold of each point (initial upper bound on the
// candidate count of a seed).
__global__ void countNeighbours(const DevData D, int N, int* __restrict__ out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const double x = D.px[i], y = D.py[i];
    const int pc = D.pcell[i];
    const int cx = pc & 0xffff, cy = pc >> 16;
    int cnt = 0;
    for (int yy = max(cy - 1, 0); yy <= min(cy + 1, D.G - 1); ++yy) {
        const int s = D.cellStart[yy * D.G + max(cx - 1, 0)];
        const int e = D.cellStart[yy * D.G + min(cx + 1, D.G - 1) + 1];
        for (int p = s; p < e; ++p) {
            if (D.sidx[p] != i && dkey(dist2(D.sx[p], D.sy[p], x, y)) < D.s2Key) ++cnt;
        }
    }
    out[i] = cnt;
}

// One block per member of the newly formed cluster: mark it clustered and
// collect (deduplicated) seeds owned by this rank within threshold of it.
// Those are appended to the host-visible dirty list and to the device work
// list of their kernel configuration (or get cardinality 1 directly if they
// have no neighbours at all).  Members of the cluster itself may be appended
// as well (marking races with scanning); they are skipped later.
__global__ void commitCluster(const DevData D, const int* __restrict__ members, unsigned char* clustered,
                              int stamp, int rank, int nranks, int* stamps,
                              const signed char* __restrict__ cfgOf, int* buckets, int bucketStride,
                              int* counts, int* nextCounts, int* dirtyOut, int* cardsOut) {
    const int m = members[blockIdx.x];
    if (threadIdx.x == 0) clustered[m] = 1;
    if (blockIdx.x == 0 && threadIdx.x < 16) nextCounts[threadIdx.x] = 0;   // for the next commit
    const double x = D.px[m], y = D.py[m];
    const int pc = D.pcell[m];
    const int cx = pc & 0xffff, cy = pc >> 16;
    for (int yy = max(cy - 1, 0); yy <= min(cy + 1, D.G - 1); ++yy) {
        const int s = D.cellStart[yy * D.G + max(cx - 1, 0)];
        const int e = D.cellStart[yy * D.G + min(cx + 1, D.G - 1) + 1];
        for (int p = s + threadIdx.x; p < e; p += blockDim.x) {
            const int j = D.sidx[p];
            if (j % nranks != rank || D.clustered[j]) continue;
            if (dkey(dist2(D.sx[p], D.sy[p], x, y)) < D.s2Key) {
                if (atomicExch(&stamps[j], stamp) != stamp) {
                    dirtyOut[atomicAdd(&counts[0], 1)] = j;
                    const int c = cfgOf[j];
                    if (c < 0) cardsOut[j] = 1;
                    else buckets[static_cast<size_t>(c) * bucketStride + atomicAdd(&counts[1 + c], 1)] = j;
                }
            }
        }
    }
}

class GpuQT {
public:
    GpuQT(const std::vector<Point>& points, double threshold, int rank, int nranks)
        : N_(static_cast<int>(points.size())), rank_(rank), nranks_(nranks), thr_(threshold) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
        for (int c = 0; c < kNumCfg; ++c) {
            CUDA_CHECK(cudaStreamCreateWithFlags(&cfgStream_[c], cudaStreamNonBlocking));
            CUDA_CHECK(cudaEventCreateWithFlags(&cfgEvent_[c], cudaEventDisableTiming));
        }
        CUDA_CHECK(cudaEventCreateWithFlags(&forkEvent_, cudaEventDisableTiming));
        int dev;
        CUDA_CHECK(cudaGetDevice(&dev));
        CUDA_CHECK(cudaDeviceGetAttribute(&numSM_, cudaDevAttrMultiProcessorCount, dev));
        const int distMode = detectDistanceMode();
        CUDA_CHECK(cudaMemcpyToSymbolAsync(cDistMode, &distMode, sizeof(int), 0, cudaMemcpyHostToDevice, stream_));
        buildGrid(points);

        // One device pool and one pinned (mapped) host pool for the work arrays
        const size_t nI = static_cast<size_t>(N_) + 64;
        CUDA_CHECK(cudaMalloc(&dPool_, sizeof(int) * nI * (5 + kNumCfg)));
        dStamps_ = dPool_;
        dSeeds_ = dPool_ + nI;
        dMembers_ = dPool_ + 2 * nI;
        dCounts_ = dPool_ + 3 * nI;   // 2 x 16: [0] dirty count, [1 + c] bucket counts
        dCfg_ = reinterpret_cast<signed char*>(dPool_ + 3 * nI + 64);
        dClustered_ = reinterpret_cast<unsigned char*>(dPool_ + 3 * nI + 64 + nI / 4 + 1);
        dBuckets_ = dPool_ + 5 * nI;
        CUDA_CHECK(cudaMemsetAsync(dPool_ + 3 * nI, 0, 2 * nI * sizeof(int), stream_));
        CUDA_CHECK(cudaMemsetAsync(dStamps_, 0xff, sizeof(int) * nI, stream_));
        D_.clustered = dClustered_;

        // Pinned host buffers; cardinalities and the dirty list are written
        // by the GPU directly (mapped memory).
        // (registering ordinary memory is much cheaper than cudaHostAlloc)
        pinnedBytes_ = (sizeof(int) * (5 * nI + 64) + 4095) / 4096 * 4096;
        hPinned_ = static_cast<int*>(std::aligned_alloc(4096, pinnedBytes_));
        if (!hPinned_) {
            fprintf(stderr, "Host allocation failed\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        CUDA_CHECK(cudaHostRegister(hPinned_, pinnedBytes_, cudaHostRegisterMapped));
        int* dMapped;
        CUDA_CHECK(cudaHostGetDevicePointer(&dMapped, hPinned_, 0));
        hSeeds_ = hPinned_;
        hMembers_ = hPinned_ + nI;
        hCounts_ = hPinned_ + 2 * nI;
        hCards_ = hPinned_ + 3 * nI;           dCardsMapped_ = dMapped + 3 * nI;
        hDirty_ = hPinned_ + 4 * nI + 64;      dDirtyMapped_ = dMapped + 4 * nI + 64;

        // Upper bound of the number of candidates per seed (counts only
        // decrease while points get clustered).
        countNeighbours<<<(N_ + 255) / 256, 256, 0, stream_>>>(D_, N_, dSeeds_);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(hSeeds_, dSeeds_, sizeof(int) * N_, cudaMemcpyDeviceToHost, stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        nbr_.assign(hSeeds_, hSeeds_ + N_);

        maxNbr_ = 0;
        std::vector<signed char> cfg(N_);
        long long storeTotal = 0;
        memOffHost_.assign(N_, -1);
        for (int i = 0; i < N_; ++i) {
            maxNbr_ = std::max(maxNbr_, nbr_[i]);
            cfg[i] = (nbr_[i] == 0) ? -1 : static_cast<signed char>(configFor(nbr_[i]));
            if (i % nranks_ == rank_) {
                memOffHost_[i] = storeTotal;
                storeTotal += nbr_[i] + 1;
            }
        }
        maxCfg_ = configFor(std::max(1, maxNbr_));
        CUDA_CHECK(cudaMemcpyAsync(dCfg_, cfg.data(), N_, cudaMemcpyHostToDevice, stream_));

        // If affordable, keep the member lists of all owned seeds on the
        // device, so the winning cluster never has to be recomputed.
        size_t freeMem = 0, totalMem = 0;
        CUDA_CHECK(cudaMemGetInfo(&freeMem, &totalMem));
        // Scratch of the global-memory configuration: at most 1/4 of memory
        maxScratchBlocks_ = static_cast<int>(std::min<size_t>(
            numSM_ * 2, std::max<size_t>(1, freeMem / 4 / (sizeof(Cand) * std::max(1, maxNbr_)))));
        if (static_cast<size_t>(storeTotal) * sizeof(int) < freeMem / 2) {
            CUDA_CHECK(cudaMalloc(&dStore_, sizeof(int) * storeTotal));
            CUDA_CHECK(cudaMalloc(&dMemOff_, sizeof(long long) * N_));
            CUDA_CHECK(cudaMemcpyAsync(dMemOff_, memOffHost_.data(), sizeof(long long) * N_,
                                       cudaMemcpyHostToDevice, stream_));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

    ~GpuQT() {
        cudaFree(dGrid_);
        cudaFree(dPool_);
        if (dMemOff_) cudaFree(dMemOff_);
        if (dStore_) cudaFree(dStore_);
        if (dScratch_) cudaFree(dScratch_);
        cudaHostUnregister(hPinned_);
        std::free(hPinned_);
        for (int c = 0; c < kNumCfg; ++c) {
            cudaStreamDestroy(cfgStream_[c]);
            cudaEventDestroy(cfgEvent_[c]);
        }
        cudaEventDestroy(forkEvent_);
        cudaStreamDestroy(stream_);
    }

    // Force loading of all kernels (lazy module loading) outside timed regions
    static void preloadKernels() {
        cudaFuncAttributes a;
        CUDA_CHECK(cudaFuncGetAttributes(&a, growClusters<32, kK, false>));
        CUDA_CHECK(cudaFuncGetAttributes(&a, growClusters<64, kK, false>));
        CUDA_CHECK(cudaFuncGetAttributes(&a, growClusters<128, kK, false>));
        CUDA_CHECK(cudaFuncGetAttributes(&a, growClusters<256, kK, false>));
        CUDA_CHECK(cudaFuncGetAttributes(&a, growClusters<512, kK, false>));
        CUDA_CHECK(cudaFuncGetAttributes(&a, growClusters<1024, kK, false>));
        CUDA_CHECK(cudaFuncGetAttributes(&a, growClusters<1024, 1, true>));
        CUDA_CHECK(cudaFuncGetAttributes(&a, countNeighbours));
        CUDA_CHECK(cudaFuncGetAttributes(&a, commitCluster));
    }

    // Cardinality of the candidate cluster of every given (unclustered) seed.
    void computeAll(const std::vector<int>& seeds, std::vector<int>& cards) {
        const int n = static_cast<int>(seeds.size());
        cards.assign(n, 1);
        int bucketCount[kNumCfg] = {0};
        for (int s : seeds) if (nbr_[s] > 0) bucketCount[configFor(nbr_[s])]++;
        int bucketOff[kNumCfg + 1];
        bucketOff[0] = 0;
        for (int c = 0; c < kNumCfg; ++c) bucketOff[c + 1] = bucketOff[c] + bucketCount[c];
        const int total = bucketOff[kNumCfg];
        if (total > 0) {
            int fill[kNumCfg];
            for (int c = 0; c < kNumCfg; ++c) fill[c] = bucketOff[c];
            for (int s : seeds) if (nbr_[s] > 0) hSeeds_[fill[configFor(nbr_[s])]++] = s;
            CUDA_CHECK(cudaMemcpyAsync(dSeeds_, hSeeds_, sizeof(int) * total, cudaMemcpyHostToDevice, stream_));
            const SeedOutput out{dCardsMapped_, dStore_, dMemOff_};
            fork(kNumCfg - 1);
            for (int c = kNumCfg - 1; c >= 0; --c) {   // big ones first, concurrently
                if (bucketCount[c] > 0) {
                    launch(c, dSeeds_ + bucketOff[c], nullptr, bucketCount[c], out, gridFor(c, bucketCount[c]),
                           cfgStream_[c]);
                }
            }
            join(kNumCfg - 1);
            CUDA_CHECK(cudaStreamSynchronize(stream_));
        }
        for (int k = 0; k < n; ++k) {
            if (nbr_[seeds[k]] > 0) cards[k] = hCards_[seeds[k]];
        }
    }

    // Owner side: member list of an owned seed whose cached cardinality is
    // 'card' (from the device store, or recomputed on the GPU) -> host.
    void ownedMembers(int seed, int card, int* dst) {
        if (dStore_) {
            CUDA_CHECK(cudaMemcpyAsync(dst, dStore_ + memOffHost_[seed], sizeof(int) * card,
                                       cudaMemcpyDeviceToHost, stream_));
        } else {
            hSeeds_[0] = seed;
            CUDA_CHECK(cudaMemcpyAsync(dSeeds_, hSeeds_, sizeof(int), cudaMemcpyHostToDevice, stream_));
            const int c = std::max(nbr_[seed] > 0 ? configFor(nbr_[seed]) : 0, 2);   // wide group: low latency
            const SeedOutput out{dCardsMapped_, dMembers_, nullptr};
            launch(c, dSeeds_, nullptr, 1, out, 1, stream_);
            CUDA_CHECK(cudaMemcpyAsync(dst, dMembers_, sizeof(int) * card, cudaMemcpyDeviceToHost, stream_));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

    int* memberBuffer() { return hMembers_; }
    bool hasStore() const { return dStore_ != nullptr; }

    // Remove the cluster of 'seed' (cardinality 'card') and recompute all
    // owned seeds within threshold of it.  The members are taken from the
    // device store if 'fromStore' (and then also copied to memberBuffer()),
    // otherwise from memberBuffer().  Returns the affected owned seeds in
    // 'dirty' (may contain cluster members; those must be ignored); their new
    // cardinalities are available through card().
    void commit(int seed, int card, bool fromStore, std::vector<int>& dirty) {
        ++stamp_;
        int* counts = dCounts_ + 16 * (stamp_ & 1);
        int* nextCounts = dCounts_ + 16 * ((stamp_ + 1) & 1);
        const int* dMem;
        if (fromStore) {
            dMem = dStore_ + memOffHost_[seed];
            CUDA_CHECK(cudaMemcpyAsync(hMembers_, dMem, sizeof(int) * card, cudaMemcpyDeviceToHost, stream_));
        } else {
            CUDA_CHECK(cudaMemcpyAsync(dMembers_, hMembers_, sizeof(int) * card, cudaMemcpyHostToDevice, stream_));
            dMem = dMembers_;
        }
        commitCluster<<<card, 128, 0, stream_>>>(D_, dMem, dClustered_, stamp_, rank_, nranks_, dStamps_,
                                                 dCfg_, dBuckets_, N_, counts, nextCounts,
                                                 dDirtyMapped_, dCardsMapped_);
        CUDA_CHECK(cudaGetLastError());
        const SeedOutput out{dCardsMapped_, dStore_, dMemOff_};
        if (maxCfg_ == 0) {
            launch(0, dBuckets_, counts + 1, 0, out, gridFor(0, -1), stream_);
        } else {
            fork(maxCfg_);
            for (int c = maxCfg_; c >= 0; --c) {
                launch(c, dBuckets_ + static_cast<size_t>(c) * N_, counts + 1 + c, 0, out, gridFor(c, -1),
                       cfgStream_[c]);
            }
            join(maxCfg_);
        }
        CUDA_CHECK(cudaMemcpyAsync(hCounts_, counts, sizeof(int), cudaMemcpyDeviceToHost, stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        dirty.assign(hDirty_, hDirty_ + hCounts_[0]);
    }

    int card(int seed) const { return hCards_[seed]; }

private:
    // Configurations: register-resident groups of TPS threads * K candidates,
    // followed by a global-memory fallback for very dense neighbourhoods.
    static constexpr int kK = 4;
    static constexpr int kNumCfg = 7;
    static constexpr int kTPS[kNumCfg] = {32, 64, 128, 256, 512, 1024, 1024};

    static int configFor(int bound) {
        for (int c = 0; c < kNumCfg - 1; ++c) {
            if (bound <= kTPS[c] * kK) return c;
        }
        return kNumCfg - 1;
    }

    // n < 0: device-side count (unknown), use one wave of blocks.
    int gridFor(int cfg, int n) const {
        const int tps = kTPS[cfg];
        const int block = (tps == 32 ? 128 : tps);
        const int gpb = block / tps;
        if (cfg == kNumCfg - 1) return std::max(1, std::min(n < 0 ? numSM_ : n, numSM_ * 2));
        if (n < 0) return numSM_ * std::max(1, 1024 / block);
        return std::max(1, std::min((n + gpb - 1) / gpb, numSM_ * std::max(1, 2048 / block) * 4));
    }

    // Configuration streams branch off / merge into the main stream
    void fork(int maxCfg) {
        CUDA_CHECK(cudaEventRecord(forkEvent_, stream_));
        for (int c = 0; c <= maxCfg; ++c) CUDA_CHECK(cudaStreamWaitEvent(cfgStream_[c], forkEvent_, 0));
    }
    void join(int maxCfg) {
        for (int c = 0; c <= maxCfg; ++c) {
            CUDA_CHECK(cudaEventRecord(cfgEvent_[c], cfgStream_[c]));
            CUDA_CHECK(cudaStreamWaitEvent(stream_, cfgEvent_[c], 0));
        }
    }

    void launch(int cfg, const int* seeds, const int* countPtr, int n, const SeedOutput& out, int blocks,
                cudaStream_t st) {
        const int block = (kTPS[cfg] == 32 ? 128 : kTPS[cfg]);
        switch (cfg) {
            case 0: growClusters<32, kK, false><<<blocks, block, 0, st>>>(D_, seeds, countPtr, n, out, nullptr, 0); break;
            case 1: growClusters<64, kK, false><<<blocks, block, 0, st>>>(D_, seeds, countPtr, n, out, nullptr, 0); break;
            case 2: growClusters<128, kK, false><<<blocks, block, 0, st>>>(D_, seeds, countPtr, n, out, nullptr, 0); break;
            case 3: growClusters<256, kK, false><<<blocks, block, 0, st>>>(D_, seeds, countPtr, n, out, nullptr, 0); break;
            case 4: growClusters<512, kK, false><<<blocks, block, 0, st>>>(D_, seeds, countPtr, n, out, nullptr, 0); break;
            case 5: growClusters<1024, kK, false><<<blocks, block, 0, st>>>(D_, seeds, countPtr, n, out, nullptr, 0); break;
            default: {
                blocks = std::min(blocks, maxScratchBlocks_);
                const size_t need = static_cast<size_t>(blocks) * maxNbr_;
                if (need > scratchSize_) {
                    if (dScratch_) CUDA_CHECK(cudaFree(dScratch_));
                    CUDA_CHECK(cudaMalloc(&dScratch_, need * sizeof(Cand)));
                    scratchSize_ = need;
                }
                growClusters<1024, 1, true><<<blocks, 1024, 0, st>>>(D_, seeds, countPtr, n, out, dScratch_, maxNbr_);
                break;
            }
        }
        CUDA_CHECK(cudaGetLastError());
    }

    void buildGrid(const std::vector<Point>& points) {
        double minx = DBL_MAX, miny = DBL_MAX, maxx = -DBL_MAX, maxy = -DBL_MAX;
#pragma omp parallel for reduction(min : minx, miny) reduction(max : maxx, maxy) if (N_ > 100000)
        for (int i = 0; i < N_; ++i) {
            minx = std::min(minx, points[i].x); maxx = std::max(maxx, points[i].x);
            miny = std::min(miny, points[i].y); maxy = std::max(maxy, points[i].y);
        }
        const double extent = std::max(maxx - minx, maxy - miny);
        const int kMaxG = 2048;
        // Cell size strictly larger than threshold (with margin for rounding)
        double h = thr_ * (1.0 + 1e-6);
        if (std::isnan(h)) h = extent + 1.0;
        if (extent / h > kMaxG - 1) h = extent / (kMaxG - 1);
        int G = static_cast<int>(extent / h) + 1;
        G = std::max(1, std::min(G, kMaxG));
        D_.G = G;
        // Squared-distance boundary: sqrt(d2) < thr  <=>  d2 < S2 (exact, as
        // the correctly rounded sqrt is monotonic)
        double S2 = thr_ * thr_;
        if (std::isnan(S2)) {
            S2 = 0.0;   // nothing is within a NaN threshold
        } else if (std::sqrt(S2) < thr_) {
            while (std::sqrt(S2) < thr_) S2 = std::nextafter(S2, INFINITY);
        } else {
            while (S2 > 0.0 && !(std::sqrt(std::nextafter(S2, 0.0)) < thr_)) S2 = std::nextafter(S2, 0.0);
        }
        D_.s2Key = 0;
        memcpy(&D_.s2Key, &S2, sizeof(S2));
        float s2Hi = static_cast<float>(S2);
        if (static_cast<double>(s2Hi) < S2) s2Hi = std::nextafter(s2Hi, INFINITY);
        D_.s2Hi = s2Hi;
        // Bound on the error of the float squared distances of seed-relative
        // coordinates (all within thr of the seed): < 2^-18 thr^2.
        D_.delta = (thr_ > 1e-15 && thr_ < 1e15) ? static_cast<float>(thr_ * thr_ * 0x1p-12) : INFINITY;

        std::vector<int> pcell(N_), cellOf(N_);
#pragma omp parallel for if (N_ > 100000)
        for (int i = 0; i < N_; ++i) {
            int cx = static_cast<int>((points[i].x - minx) / h);
            int cy = static_cast<int>((points[i].y - miny) / h);
            cx = std::max(0, std::min(cx, G - 1));
            cy = std::max(0, std::min(cy, G - 1));
            pcell[i] = cx | (cy << 16);
            cellOf[i] = cy * G + cx;
        }
        std::vector<int> cellStart(static_cast<size_t>(G) * G + 1, 0);
        for (int i = 0; i < N_; ++i) cellStart[cellOf[i] + 1]++;
        for (size_t c = 0; c < static_cast<size_t>(G) * G; ++c) cellStart[c + 1] += cellStart[c];
        std::vector<int> fill(cellStart.begin(), cellStart.end() - 1);
        std::vector<int> sidx(N_);
        std::vector<double> sx(N_), sy(N_), px(N_), py(N_);
        for (int i = 0; i < N_; ++i) sidx[fill[cellOf[i]]++] = i;
#pragma omp parallel for if (N_ > 100000)
        for (int k = 0; k < N_; ++k) {
            sx[k] = points[sidx[k]].x; sy[k] = points[sidx[k]].y;
            px[k] = points[k].x; py[k] = points[k].y;
        }
        // Single device allocation for all grid/point arrays
        const size_t nD = static_cast<size_t>(N_), nC = cellStart.size();
        const size_t bytes = sizeof(double) * 4 * nD + sizeof(int) * (2 * nD + nC);
        CUDA_CHECK(cudaMalloc(&dGrid_, bytes));
        std::vector<char> staging(bytes);
        char* hp = staging.data();
        char* dp = static_cast<char*>(dGrid_);
        auto place = [&](auto*& dptr, const auto& v) {
            using T = typename std::remove_reference_t<decltype(v)>::value_type;
            memcpy(hp, v.data(), sizeof(T) * v.size());
            dptr = reinterpret_cast<std::remove_reference_t<decltype(dptr)>>(dp);
            hp += sizeof(T) * v.size();
            dp += sizeof(T) * v.size();
        };
        place(dPx_, px); place(dPy_, py); place(dSx_, sx); place(dSy_, sy);
        place(dSidx_, sidx); place(dCellStart_, cellStart); place(dPcell_, pcell);
        CUDA_CHECK(cudaMemcpy(dGrid_, staging.data(), bytes, cudaMemcpyHostToDevice));
        D_.px = dPx_; D_.py = dPy_; D_.sx = dSx_; D_.sy = dSy_;
        D_.sidx = dSidx_; D_.cellStart = dCellStart_; D_.pcell = dPcell_;
    }

    int N_, rank_, nranks_;
    double thr_;
    int numSM_ = 1;
    int stamp_ = 0;
    int maxNbr_ = 0;
    int maxCfg_ = 0;
    int maxScratchBlocks_ = 1;
    cudaStream_t stream_;
    cudaStream_t cfgStream_[kNumCfg];
    cudaEvent_t cfgEvent_[kNumCfg];
    cudaEvent_t forkEvent_;
    DevData D_{};
    std::vector<int> nbr_;
    std::vector<long long> memOffHost_;
    void* dGrid_ = nullptr;
    double *dPx_ = nullptr, *dPy_ = nullptr, *dSx_ = nullptr, *dSy_ = nullptr;
    int *dSidx_ = nullptr, *dCellStart_ = nullptr, *dPcell_ = nullptr;
    int* dPool_ = nullptr;
    int *dStamps_ = nullptr, *dSeeds_ = nullptr, *dMembers_ = nullptr, *dCounts_ = nullptr, *dBuckets_ = nullptr;
    signed char* dCfg_ = nullptr;
    unsigned char* dClustered_ = nullptr;
    long long* dMemOff_ = nullptr;
    Cand* dScratch_ = nullptr;
    size_t scratchSize_ = 0;
    int* dStore_ = nullptr;
    size_t pinnedBytes_ = 0;
    int *hPinned_ = nullptr, *hSeeds_ = nullptr, *hMembers_ = nullptr, *hCounts_ = nullptr;
    int *hCards_ = nullptr, *hDirty_ = nullptr;
    int *dCardsMapped_ = nullptr, *dDirtyMapped_ = nullptr;
};

// Main QT clustering algorithm (executed collectively by all MPI ranks; every
// rank returns the full, identical list of clusters).
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    const int N = static_cast<int>(points.size());
    std::vector<Cluster> clusters;
    if (N == 0) return clusters;

    GpuQT gpu(points, threshold, rank, nranks);
    std::vector<unsigned char> clustered(N, 0);
    std::vector<int> card(N, 0);

    // Owned seeds: round-robin for spatial load balance
    std::vector<int> owned;
    for (int i = rank; i < N; i += nranks) owned.push_back(i);
    std::vector<int> cards;
    gpu.computeAll(owned, cards);

    // Ordered by (max cardinality, min index)
    std::set<std::pair<int, int>> order;
    for (size_t k = 0; k < owned.size(); ++k) {
        card[owned[k]] = cards[k];
        order.emplace(-cards[k], owned[k]);
    }

    int remaining = N;
    std::vector<int> dirty;
    int* members = gpu.memberBuffer();
    while (remaining > 0) {
        long long key = -1;
        if (!order.empty()) {
            const auto& top = *order.begin();
            key = (static_cast<long long>(-top.first) << 32) |
                  static_cast<long long>(static_cast<unsigned>(INT_MAX - top.second));
        }
        long long gkey = key;
        if (nranks > 1) MPI_Allreduce(&key, &gkey, 1, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
        if (gkey < 0) break;
        const int bestCard = static_cast<int>(gkey >> 32);
        const int bestSeed = INT_MAX - static_cast<int>(gkey & 0xffffffffLL);

        if (bestCard <= 1) {
            // Every remaining point is a singleton; the original picks them
            // in ascending index order.
            for (int i = 0; i < N; ++i) {
                if (!clustered[i]) {
                    Cluster c;
                    c.seed_point = i;
                    c.members.push_back(i);
                    clusters.push_back(std::move(c));
                    clustered[i] = 1;
                }
            }
            break;
        }

        // The owner holds the member list (device store); distribute it.
        const int owner = bestSeed % nranks;
        bool fromStore = false;
        if (nranks == 1 && gpu.hasStore()) {
            fromStore = true;   // fetched inside commit, no extra round trip
        } else {
            if (rank == owner) gpu.ownedMembers(bestSeed, bestCard, members);
            if (nranks > 1) MPI_Bcast(members, bestCard, MPI_INT, owner, MPI_COMM_WORLD);
            fromStore = (rank == owner && gpu.hasStore());
        }
        gpu.commit(bestSeed, bestCard, fromStore, dirty);

        Cluster cluster;
        cluster.seed_point = bestSeed;
        cluster.members.assign(members, members + bestCard);
        for (int m : cluster.members) {
            clustered[m] = 1;
            if (m % nranks == rank) order.erase({-card[m], m});
        }
        remaining -= bestCard;

        for (int s : dirty) {
            if (clustered[s]) continue;
            order.erase({-card[s], s});
            card[s] = gpu.card(s);
            order.emplace(-card[s], s);
        }
        clusters.push_back(std::move(cluster));
    }

    return clusters;
}

// Validation: check that clusters satisfy the QT clustering properties
bool validateClusters(const std::vector<Cluster>& clusters,
                     const std::vector<Point>& points,
                     const double threshold) {
    bool valid = true;
    
    printf("Validating clusters:\n");
    
    // Compute cluster diameters in parallel
    std::vector<double> diameters(clusters.size(), 0.0);
#pragma omp parallel for schedule(dynamic)
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
    for (size_t i = 0; i < membership.size(); ++i) {
        if (membership[i] >= 0) clustered_count++;
    }
    
    printf("Total points: %zu, Clustered: %d, Unclustered: %zu\n",
           points.size(), clustered_count, points.size() - clustered_count);
    
    return valid;
}

int run(int argc, char** argv);

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
    const int ret = run(argc, argv);
    MPI_Finalize();
    return ret;
}

int run(int argc, char** argv) {
    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    const bool root = (rank == 0);

    // One GPU per rank (round-robin over the GPUs of a node)
    {
        MPI_Comm local;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
        int localRank;
        int localSize;
        MPI_Comm_rank(local, &localRank);
        MPI_Comm_size(local, &localSize);
        MPI_Comm_free(&local);
        int ndev = 0;
        CUDA_CHECK(cudaGetDeviceCount(&ndev));
        if (ndev <= 0) {
            fprintf(stderr, "No CUDA device available\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        CUDA_CHECK(cudaSetDevice(localRank % ndev));
        // Create the context, load kernels and initialise pinned memory
        // support outside the timed region
        CUDA_CHECK(cudaFree(nullptr));
        GpuQT::preloadKernels();
        void* warm = std::aligned_alloc(4096, 4096);
        CUDA_CHECK(cudaHostRegister(warm, 4096, cudaHostRegisterMapped));
        CUDA_CHECK(cudaHostUnregister(warm));
        std::free(warm);

        // Share the node's cores among the ranks on it (unless the user set
        // OMP_NUM_THREADS explicitly)
        if (!getenv("OMP_NUM_THREADS")) {
            omp_set_num_threads(std::max(1, omp_get_num_procs() / localSize));
        }
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
            if (root) printUsage(argv[0]);
            return 0;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }
    
    if (num_points <= 0 || threshold <= 0.0) {
        if (root) printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n", 
               num_points, threshold);
        return 1;
    }
    
    if (root) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Generate synthetic data (identically on every rank)
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);
    
    // Perform QT clustering
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_start = std::chrono::high_resolution_clock::now();
    
    const std::vector<Cluster> clusters = qtClustering(points, threshold);
    
    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);
    
    if (!root) return 0;   // all output and validation on rank 0
    
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
