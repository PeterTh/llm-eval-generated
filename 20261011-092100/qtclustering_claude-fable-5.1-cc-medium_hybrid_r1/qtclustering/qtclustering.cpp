// QT Clustering Benchmark - Hybrid MPI + OpenMP + CUDA Version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// Parallelization strategy (semantics identical to the sequential version):
//
//  * Every outer iteration evaluates a candidate cluster for every seed
//    that may have changed. Seeds are distributed cyclically over the MPI
//    ranks, and inside a rank over all GPUs assigned to it (one OpenMP
//    thread drives each GPU). Each candidate cluster is grown by one CUDA
//    thread block: the block keeps, for every candidate point, its maximum
//    distance to the current cluster members (this is exactly the quantity
//    the sequential code recomputes from scratch each step) and performs a
//    block-wide arg-min with the same tie-breaking (smallest index wins).
//    Candidates whose diameter already exceeds the threshold are compacted
//    away, so a step costs O(neighbourhood) instead of O(N * cluster size).
//
//  * Candidate clusters are cached between outer iterations. Removing the
//    winning cluster B only changes the candidate cluster of a seed s if a
//    member of B could have been part of s's candidate, which requires that
//    some member of B lies within the threshold distance of s. Seeds for
//    which this is impossible keep their cached cardinality; the others are
//    recomputed. This is an exact (conservative) invalidation.
//
//  * The best seed (max cardinality, lowest index on ties - exactly the
//    order of the sequential loop) is selected redundantly on every rank from
//    the all-gathered cardinalities. Member lists of the top-ranked cached
//    candidates are produced by extra blocks of the regular batch launches
//    (off the critical path) on every rank; a freshly computed winner is
//    broadcast by the rank that computed it. Per iteration the communication
//    is one MPI_Allgatherv plus, rarely, one small MPI_Bcast.
//
//  * Numerics: all decisions are taken on squared distances (sqrt is
//    monotonic, so max/threshold comparisons are exact on squared values;
//    exact rounded-sqrt ties are detected and resolved separately). A
//    single-precision prefilter with a rigorous error bound skips the double
//    precision work for candidates that provably cannot change, and
//    comparisons of non-negative doubles are done as 64-bit integer
//    comparisons. Host and device use -ffp-contract=off / -fmad=false so the
//    distances are bit-identical to the sequential program.

#include <algorithm>
#include <chrono>
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

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err__ = (call);                                               \
        if (err__ != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err__), \
                    __FILE__, __LINE__);                                          \
            MPI_Abort(MPI_COMM_WORLD, 1);                                         \
        }                                                                         \
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
// CUDA kernels
// ---------------------------------------------------------------------------

static const unsigned FULL_MASK = 0xffffffffu;

// Squared Euclidean distance: the sequential code uses sqrt(dx*dx + dy*dy).
// sqrt is monotonic, so max/threshold decisions can be taken on the squared
// values; ties are resolved with the actual rounded sqrt (see below).
__host__ __device__ inline double distanceSq(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return dx * dx + dy * dy;
}

// Smallest double q such that sqrt(q) >= threshold. Then for any squared
// distance q: sqrt(q) < threshold  <=>  q < sqCutoff(threshold).
static double sqCutoff(const double threshold) {
    double q = threshold * threshold;
    while (sqrt(q) >= threshold) q = nextafter(q, 0.0);
    while (sqrt(q) < threshold) q = nextafter(q, INFINITY);
    return q;
}

// Non-negative doubles (and +inf) order exactly like their bit patterns, so
// all comparisons of squared distances are done on 64-bit integers (fast on
// GPUs with reduced FP64 throughput).
__device__ __forceinline__ long long d2ll(const double v) { return __double_as_longlong(v); }
__device__ __forceinline__ double ll2d(const long long v) { return __longlong_as_double(v); }

// Rigorous float upper bound on |exact squared distance - df| where df is the
// single-precision estimate computed from single-precision coordinates:
//   |fl(x) - x| <= |x| 2^-24 <= A 2^-24, so |dxf - dx| <= (|dxf| + 2A) 2^-24
//   |dxf^2 - dx^2| <= ex (2|dxf| + ex), plus at most three roundings of the
//   squares and the sum (<= 2^-23 df). Everything is doubled for the
//   roundings committed while evaluating the bound itself.
__device__ __forceinline__ float sqErrBound(const float dxf, const float dyf,
                                            const float df, const float twoA) {
    const float ax = fabsf(dxf), ay = fabsf(dyf);
    const float ex = (ax + twoA) * 5.96046448e-08f;  // 2^-24
    const float ey = (ay + twoA) * 5.96046448e-08f;
    return 2.0f * (df * 1.19209290e-07f + 2.0f * (ex * ax + ey * ay) + ex * ex + ey * ey)
           + 1e-30f;
}

// Grow one candidate cluster per thread block.
//   val[j] : squared maximum distance of candidate idx[j] to all current
//            members (+inf once the candidate became a member)
//   pf[j]  : single-precision coordinates of the candidate (prefilter only)
// A candidate is selectable while val[j] < T2 (i.e. its rounded diameter is
// below the threshold); since val only grows, candidates at or above T2 are
// dead and are compacted away. The list is kept in ascending point-index
// order, so the position in the list is a valid tie-breaker (lowest index
// wins, exactly as in the sequential scan).
//
// Candidate selection: the sequential code takes the first candidate (index
// order) whose rounded diameter sqrt(val) is minimal. We find the minimal
// squared value v1 (and the next larger value v2); only if sqrt(v2) could
// round to sqrt(v1) is an exact pass with sqrt needed to find the lowest
// index among all candidates with the same rounded diameter.
template <int BS, int MINB>
__global__ void __launch_bounds__(BS, MINB)
candidateClusterKernel(const Point* __restrict__ pts,
                       const float2* __restrict__ ptsf,
                       const unsigned char* __restrict__ clustered,
                       const int* __restrict__ seeds,
                       const int N,
                       const double T2,
                       const float T2f_hi,
                       const float twoA,
                       double* __restrict__ gval,
                       float2* __restrict__ gpf,
                       int* __restrict__ gidx,
                       int* __restrict__ out_card,
                       int* __restrict__ out_members,
                       const int mem_cap) {
    constexpr int NW = BS / 32;
    __shared__ long long s_v1[NW];
    __shared__ long long s_v2[NW];
    __shared__ int s_j1[NW];
    __shared__ int s_cnt[NW];
    __shared__ int s_bc;
    __shared__ int s_alive;
    __shared__ int s_tie;
    __shared__ double s_s1;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const unsigned lanemask_lt = (1u << lane) - 1u;
    const int seed = seeds[blockIdx.x];
    const long long INFI = d2ll(INFINITY);
    const long long T2i = d2ll(T2);
    const Point ps = pts[seed];
    const float2 psf = ptsf[seed];

    double* val = gval + static_cast<size_t>(blockIdx.x) * N;
    float2* pf = gpf + static_cast<size_t>(blockIdx.x) * N;
    int* idx = gidx + static_cast<size_t>(blockIdx.x) * N;

    // Phase 0: build the compact candidate list = unclustered points within
    // the threshold of the seed (squared distance to the seed as initial
    // val). Each thread handles 4 consecutive points per chunk so that the
    // list stays in ascending index order.
    int len = 0;
    for (int base = 0; base < N; base += 4 * BS) {
        const int j0 = base + 4 * tid;
        double v[4];
        float2 pj[4];
        unsigned keep = 0;  // bit i: keep item i
        if (j0 + 3 < N) {
            const float4 a = reinterpret_cast<const float4*>(ptsf + j0)[0];
            const float4 b = reinterpret_cast<const float4*>(ptsf + j0)[1];
            pj[0] = make_float2(a.x, a.y);
            pj[1] = make_float2(a.z, a.w);
            pj[2] = make_float2(b.x, b.y);
            pj[3] = make_float2(b.z, b.w);
        } else {
            for (int i = 0; i < 4; ++i) pj[i] = (j0 + i < N) ? ptsf[j0 + i] : make_float2(0.f, 0.f);
        }
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int j = j0 + i;
            v[i] = 0.0;
            if (j < N && j != seed && !clustered[j]) {
                const float dxf = pj[i].x - psf.x, dyf = pj[i].y - psf.y;
                const float df = dxf * dxf + dyf * dyf;
                const float err = sqErrBound(dxf, dyf, df, twoA);
                if (!(__fsub_rd(df, err) >= T2f_hi)) {  // not provably outside
                    v[i] = distanceSq(pts[j], ps);
                    if (d2ll(v[i]) < T2i) keep |= 1u << i;
                }
            }
        }
        const int cnt = __popc(keep);
        // Warp-level exclusive scan of the counts
        int incl = cnt;
        #pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int t = __shfl_up_sync(FULL_MASK, incl, o);
            if (lane >= o) incl += t;
        }
        if (lane == 31) s_cnt[warp] = incl;
        __syncthreads();
        int off = 0, total = 0;
        for (int w = 0; w < NW; ++w) {
            if (w < warp) off += s_cnt[w];
            total += s_cnt[w];
        }
        int pos = len + off + incl - cnt;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            if (keep & (1u << i)) {
                val[pos] = v[i];
                pf[pos] = pj[i];
                idx[pos] = j0 + i;
                pos++;
            }
        }
        len += total;
        __syncthreads();
    }

    int* members = out_members ? out_members + static_cast<size_t>(blockIdx.x) * mem_cap : nullptr;
    if (members && tid == 0) members[0] = seed;

    int count = 1;
    int last = -1;  // newest member; -1: val already holds distances to the seed

    while (true) {
        Point pl = ps;
        float2 plf = psf;
        if (last >= 0) {
            pl = pts[last];
            plf = make_float2(__double2float_rn(pl.x), __double2float_rn(pl.y));
        }
        long long v1 = INFI, v2 = INFI;  // smallest and second smallest val
        int j1 = -1;                     // first position holding v1
        int alive = 0;

        for (int j = tid; j < len; j += BS) {
            long long vi = d2ll(val[j]);
            if (vi >= T2i) continue;  // dead or member
            if (last >= 0) {
                const float2 pj = pf[j];
                const float dxf = pj.x - plf.x, dyf = pj.y - plf.y;
                const float df = dxf * dxf + dyf * dyf;
                const float err = sqErrBound(dxf, dyf, df, twoA);
                const float vlo = __double2float_rd(ll2d(vi));
                if (!(__fadd_ru(df, err) < vlo)) {  // distance may exceed val
                    const double d = distanceSq(pts[idx[j]], pl);
                    const long long di = d2ll(d);
                    if (di > vi) {
                        vi = di;
                        val[j] = d;
                        if (vi >= T2i) continue;
                    }
                }
            }
            alive++;
            if (vi < v1) {
                v2 = v1;
                v1 = vi;
                j1 = j;
            } else if (vi > v1 && vi < v2) {
                v2 = vi;
            }
        }
        for (int off = 16; off > 0; off >>= 1) {
            const long long ov1 = __shfl_down_sync(FULL_MASK, v1, off);
            const long long ov2 = __shfl_down_sync(FULL_MASK, v2, off);
            const int oj1 = __shfl_down_sync(FULL_MASK, j1, off);
            const int oa = __shfl_down_sync(FULL_MASK, alive, off);
            alive += oa;
            if (oj1 >= 0) {
                if (j1 < 0 || ov1 < v1 || (ov1 == v1 && oj1 < j1)) {
                    long long nv2 = min(ov2, v2);
                    if (j1 >= 0 && v1 > ov1) nv2 = min(nv2, v1);
                    v1 = ov1;
                    j1 = oj1;
                    v2 = nv2;
                } else {
                    long long nv2 = min(v2, ov2);
                    if (ov1 > v1) nv2 = min(nv2, ov1);
                    v2 = nv2;
                }
            }
        }
        if (lane == 0) {
            s_v1[warp] = v1;
            s_v2[warp] = v2;
            s_j1[warp] = j1;
            s_cnt[warp] = alive;
        }
        __syncthreads();
        if (tid == 0) {
            v1 = s_v1[0];
            v2 = s_v2[0];
            j1 = s_j1[0];
            alive = s_cnt[0];
            for (int w = 1; w < NW; ++w) {
                alive += s_cnt[w];
                const long long ov1 = s_v1[w];
                const long long ov2 = s_v2[w];
                const int oj1 = s_j1[w];
                if (oj1 >= 0) {
                    if (j1 < 0 || ov1 < v1 || (ov1 == v1 && oj1 < j1)) {
                        long long nv2 = min(ov2, v2);
                        if (j1 >= 0 && v1 > ov1) nv2 = min(nv2, v1);
                        v1 = ov1;
                        j1 = oj1;
                        v2 = nv2;
                    } else {
                        long long nv2 = min(v2, ov2);
                        if (ov1 > v1) nv2 = min(nv2, ov1);
                        v2 = nv2;
                    }
                }
            }
            int tie = 0;
            if (j1 >= 0 && v2 < INFI) {
                const double v1d = ll2d(v1), v2d = ll2d(v2);
                // sqrt(v2) can only round to sqrt(v1) if v2 is within ~2^-51
                // relative of v1
                if (v2d - v1d <= v1d * 0x1p-50) {
                    const double s1 = sqrt(v1d);
                    if (sqrt(v2d) == s1) {
                        tie = 1;
                        s_s1 = s1;
                    }
                }
            }
            s_tie = tie;
            s_j1[0] = j1;
            s_alive = alive;
        }
        __syncthreads();
        j1 = s_j1[0];
        if (s_tie) {
            // Exact pass: lowest position whose rounded diameter equals s1
            const double s1 = s_s1;
            int jb = j1;
            for (int j = tid; j < j1; j += BS) {
                const double v = val[j];
                if (d2ll(v) < T2i && sqrt(v) == s1) {
                    jb = j;
                    break;
                }
            }
            for (int off = 16; off > 0; off >>= 1) {
                const int oj = __shfl_down_sync(FULL_MASK, jb, off);
                jb = min(jb, oj);
            }
            if (lane == 0) s_j1[warp] = jb;
            __syncthreads();
            if (tid == 0) {
                for (int w = 1; w < NW; ++w) jb = min(jb, s_j1[w]);
                s_j1[0] = jb;
            }
            __syncthreads();
            j1 = s_j1[0];
        }
        if (tid == 0) {
            int bc = -1;
            if (j1 >= 0) {
                bc = idx[j1];
                val[j1] = INFINITY;  // the chosen point is now a member
                if (members && count < mem_cap) members[count] = bc;
            }
            s_bc = bc;
        }
        __syncthreads();
        const int bc = s_bc;
        if (bc < 0) break;
        count++;
        last = bc;
        alive = s_alive - 1;

        // Compact the candidate list (order preserving) when a good part of
        // it is dead
        if (len > BS && alive < (len / 4) * 3) {
            int newlen = 0;
            for (int base = 0; base < len; base += BS) {
                const int j = base + tid;
                const bool keep = (j < len) && (d2ll(val[j]) < T2i);
                double v = 0.0;
                float2 p = make_float2(0.f, 0.f);
                int c = 0;
                if (keep) {
                    v = val[j];
                    p = pf[j];
                    c = idx[j];
                }
                const unsigned m = __ballot_sync(FULL_MASK, keep);
                if (lane == 0) s_cnt[warp] = __popc(m);
                __syncthreads();
                int off = 0, total = 0;
                for (int w = 0; w < NW; ++w) {
                    if (w < warp) off += s_cnt[w];
                    total += s_cnt[w];
                }
                if (keep) {
                    const int pos = newlen + off + __popc(m & lanemask_lt);
                    val[pos] = v;
                    pf[pos] = p;
                    idx[pos] = c;
                }
                newlen += total;
                __syncthreads();
            }
            len = newlen;
        }
    }

    if (tid == 0) out_card[blockIdx.x] = count;
}

// Flag every unclustered point that lies within the threshold of any member
// of the newly formed cluster B; only those seeds need to be recomputed.
__global__ void markAffectedKernel(const Point* __restrict__ pts,
                                   const float2* __restrict__ ptsf,
                                   const unsigned char* __restrict__ clustered,
                                   const int* __restrict__ B,
                                   const int nB,
                                   const int N,
                                   const double T2,
                                   const float T2f_hi,
                                   const float twoA,
                                   unsigned char* __restrict__ flags) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= N) return;
    if (clustered[s]) {
        flags[s] = 0;
        return;
    }
    const Point ps = pts[s];
    const float2 psf = ptsf[s];
    const long long T2i = d2ll(T2);
    unsigned char f = 0;
    for (int b = 0; b < nB; ++b) {
        const int m = B[b];
        const float2 pm = ptsf[m];
        const float dxf = pm.x - psf.x, dyf = pm.y - psf.y;
        const float df = dxf * dxf + dyf * dyf;
        const float err = sqErrBound(dxf, dyf, df, twoA);
        if (__fsub_rd(df, err) >= T2f_hi) continue;  // provably outside
        if (d2ll(distanceSq(ps, pts[m])) < T2i) {
            f = 1;
            break;
        }
    }
    flags[s] = f;
}

// ---------------------------------------------------------------------------
// Per-GPU state
// ---------------------------------------------------------------------------

struct DeviceCtx {
    int dev = 0;
    int sm_count = 1;
    int max_batch = 0;
    float T2f_hi = 0.f;
    float twoA = 0.f;
    Point* d_pts = nullptr;
    float2* d_ptsf = nullptr;
    unsigned char* d_clustered = nullptr;
    int* d_seeds = nullptr;
    int* d_card = nullptr;
    double* d_val = nullptr;
    float2* d_pf = nullptr;
    int* d_idx = nullptr;
    int* d_members = nullptr;   // one slot of mem_cap ints per block of a batch
    int mem_cap = 0;
    unsigned char* d_flags = nullptr;
};

static const int SINGLE_BS = 1024;  // block size for the single-seed kernel
static const size_t ENTRY_BYTES = sizeof(double) + sizeof(float2) + sizeof(int);

static void initDevice(DeviceCtx& ctx, const int dev, const std::vector<Point>& points,
                       const double T2) {
    const int N = static_cast<int>(points.size());
    ctx.dev = dev;
    CUDA_CHECK(cudaSetDevice(dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&ctx.sm_count, cudaDevAttrMultiProcessorCount, dev));

    // Single-precision copy of the points and prefilter constants
    std::vector<float2> ptsf(N);
    double A = 0.0;
    for (int i = 0; i < N; ++i) {
        ptsf[i] = make_float2(static_cast<float>(points[i].x), static_cast<float>(points[i].y));
        A = std::max(A, std::max(std::fabs(points[i].x), std::fabs(points[i].y)));
    }
    ctx.twoA = nextafterf(static_cast<float>(2.0 * A), INFINITY);
    ctx.T2f_hi = nextafterf(static_cast<float>(T2), INFINITY);

    CUDA_CHECK(cudaMalloc(&ctx.d_pts, N * sizeof(Point)));
    CUDA_CHECK(cudaMemcpy(ctx.d_pts, points.data(), N * sizeof(Point), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&ctx.d_ptsf, N * sizeof(float2)));
    CUDA_CHECK(cudaMemcpy(ctx.d_ptsf, ptsf.data(), N * sizeof(float2), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&ctx.d_clustered, N));
    CUDA_CHECK(cudaMemset(ctx.d_clustered, 0, N));
    CUDA_CHECK(cudaMalloc(&ctx.d_seeds, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ctx.d_card, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ctx.d_flags, N));

    // Global scratch for the (pre-compaction) candidate lists: one N-entry
    // slot per concurrently processed seed, bounded by available memory.
    size_t free_b = 0, total_b = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
    const size_t per_seed = static_cast<size_t>(N) * ENTRY_BYTES;
    size_t batch = std::max<size_t>(1, (free_b / 2) / per_seed);
    batch = std::min<size_t>(batch, static_cast<size_t>(N));
    ctx.max_batch = static_cast<int>(batch);
    CUDA_CHECK(cudaMalloc(&ctx.d_val, batch * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&ctx.d_pf, batch * N * sizeof(float2)));
    CUDA_CHECK(cudaMalloc(&ctx.d_idx, batch * N * sizeof(int)));

    // Member slots: every block of a batch records its cluster members (if
    // they fit), so member lists come for free with the cardinalities.
    const size_t mem_budget = std::min<size_t>(free_b / 8, size_t(1) << 30);
    size_t mem_cap = std::max<size_t>(1024, mem_budget / (batch * sizeof(int)));
    mem_cap = std::min<size_t>(mem_cap, static_cast<size_t>(N));
    ctx.mem_cap = static_cast<int>(mem_cap);
    CUDA_CHECK(cudaMalloc(&ctx.d_members,
                          std::max(batch * mem_cap, static_cast<size_t>(N)) * sizeof(int)));
}

static void freeDevice(DeviceCtx& ctx) {
    cudaSetDevice(ctx.dev);
    cudaFree(ctx.d_pts);
    cudaFree(ctx.d_ptsf);
    cudaFree(ctx.d_clustered);
    cudaFree(ctx.d_seeds);
    cudaFree(ctx.d_card);
    cudaFree(ctx.d_members);
    cudaFree(ctx.d_flags);
    cudaFree(ctx.d_val);
    cudaFree(ctx.d_pf);
    cudaFree(ctx.d_idx);
}

// Compute the cardinality of the candidate cluster for each seed in
// seeds[0..n) on the given device, writing them to cards[0..n). For the
// seeds whose position is listed in want[0..nwant) the member list is also
// copied back into store[seed] (left empty if it did not fit in a slot).
static void computeCardinalities(DeviceCtx& ctx, const int N, const double T2,
                                 const int* seeds, int* cards, const int n,
                                 const int* want, const int nwant,
                                 std::vector<std::vector<int>>& store,
                                 const int nfresh, int& best_fresh,
                                 std::vector<int>& best_fresh_members) {
    CUDA_CHECK(cudaSetDevice(ctx.dev));
    int w = 0;
    best_fresh = -1;
    int best_fresh_card = -1;
    for (int start = 0; start < n; start += ctx.max_batch) {
        const int cnt = std::min(ctx.max_batch, n - start);
        CUDA_CHECK(cudaMemcpy(ctx.d_seeds, seeds + start, cnt * sizeof(int),
                              cudaMemcpyHostToDevice));
        // Few blocks: use wide blocks to shorten each (sequential) growth step;
        // many blocks: narrow blocks for throughput.
        if (cnt <= ctx.sm_count) {
            candidateClusterKernel<1024, 1><<<cnt, 1024>>>(
                ctx.d_pts, ctx.d_ptsf, ctx.d_clustered, ctx.d_seeds, N, T2, ctx.T2f_hi, ctx.twoA,
                ctx.d_val, ctx.d_pf, ctx.d_idx, ctx.d_card, ctx.d_members, ctx.mem_cap);
        } else if (cnt <= 2 * ctx.sm_count) {
            candidateClusterKernel<512, 2><<<cnt, 512>>>(
                ctx.d_pts, ctx.d_ptsf, ctx.d_clustered, ctx.d_seeds, N, T2, ctx.T2f_hi, ctx.twoA,
                ctx.d_val, ctx.d_pf, ctx.d_idx, ctx.d_card, ctx.d_members, ctx.mem_cap);
        } else if (cnt <= 4 * ctx.sm_count) {
            candidateClusterKernel<256, 4><<<cnt, 256>>>(
                ctx.d_pts, ctx.d_ptsf, ctx.d_clustered, ctx.d_seeds, N, T2, ctx.T2f_hi, ctx.twoA,
                ctx.d_val, ctx.d_pf, ctx.d_idx, ctx.d_card, ctx.d_members, ctx.mem_cap);
        } else {
            candidateClusterKernel<128, 8><<<cnt, 128>>>(
                ctx.d_pts, ctx.d_ptsf, ctx.d_clustered, ctx.d_seeds, N, T2, ctx.T2f_hi, ctx.twoA,
                ctx.d_val, ctx.d_pf, ctx.d_idx, ctx.d_card, ctx.d_members, ctx.mem_cap);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(cards + start, ctx.d_card, cnt * sizeof(int),
                              cudaMemcpyDeviceToHost));
        // Best freshly computed seed of this batch (max cardinality, lowest
        // index): keep its member list, it may be the next winner
        int bp = -1;
        for (int p = start; p < std::min(start + cnt, nfresh); ++p) {
            if (cards[p] > best_fresh_card || (cards[p] == best_fresh_card && seeds[p] < best_fresh)) {
                best_fresh_card = cards[p];
                best_fresh = seeds[p];
                bp = p;
            }
        }
        if (bp >= 0) {
            if (best_fresh_card <= ctx.mem_cap) {
                best_fresh_members.resize(best_fresh_card);
                CUDA_CHECK(cudaMemcpy(best_fresh_members.data(),
                                      ctx.d_members + static_cast<size_t>(bp - start) * ctx.mem_cap,
                                      best_fresh_card * sizeof(int), cudaMemcpyDeviceToHost));
            } else {
                best_fresh_members.clear();
            }
        }
        for (; w < nwant && want[w] < start + cnt; ++w) {
            const int pos = want[w];
            const int seed = seeds[pos];
            const int card = cards[pos];
            std::vector<int>& m = store[seed];
            if (card <= ctx.mem_cap) {
                m.resize(card);
                CUDA_CHECK(cudaMemcpy(m.data(),
                                      ctx.d_members + static_cast<size_t>(pos - start) * ctx.mem_cap,
                                      card * sizeof(int), cudaMemcpyDeviceToHost));
            } else {
                m.clear();
            }
        }
    }
}

// Rebuild the member list of the candidate cluster of one seed.
static void computeMembers(DeviceCtx& ctx, const int N, const double T2,
                           const int seed, std::vector<int>& members) {
    CUDA_CHECK(cudaSetDevice(ctx.dev));
    CUDA_CHECK(cudaMemcpy(ctx.d_seeds, &seed, sizeof(int), cudaMemcpyHostToDevice));
    candidateClusterKernel<SINGLE_BS, 1><<<1, SINGLE_BS>>>(
        ctx.d_pts, ctx.d_ptsf, ctx.d_clustered, ctx.d_seeds, N, T2, ctx.T2f_hi, ctx.twoA,
        ctx.d_val, ctx.d_pf, ctx.d_idx, ctx.d_card, ctx.d_members, N);
    CUDA_CHECK(cudaGetLastError());
    int card = 0;
    CUDA_CHECK(cudaMemcpy(&card, ctx.d_card, sizeof(int), cudaMemcpyDeviceToHost));
    members.resize(card);
    CUDA_CHECK(cudaMemcpy(members.data(), ctx.d_members, card * sizeof(int),
                          cudaMemcpyDeviceToHost));
}

// ---------------------------------------------------------------------------
// Main QT clustering algorithm (hybrid MPI + OpenMP + CUDA)
// ---------------------------------------------------------------------------
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold,
                                  std::vector<DeviceCtx>& devices,
                                  const int mpi_rank,
                                  const int mpi_size) {
    const int N = static_cast<int>(points.size());
    const int ndev = static_cast<int>(devices.size());
    const double T2 = sqCutoff(threshold);
    std::vector<unsigned char> clustered(N, 0);
    std::vector<int> card(N, -1);          // cached candidate cardinality per seed
    std::vector<unsigned char> flags(N, 1);
    std::vector<Cluster> clusters;

    // Seeds whose candidate cluster must be (re)computed this iteration
    std::vector<int> invalid(N);
    for (int i = 0; i < N; ++i) invalid[i] = i;

    std::vector<int> my_seeds, my_cards, all_cards;
    std::vector<int> counts(mpi_size), displs(mpi_size);
    std::vector<std::vector<int>> dev_seeds(ndev), dev_cards(ndev), dev_want(ndev);
    std::vector<std::vector<int>> dev_fresh_members(ndev);
    std::vector<int> dev_fresh(ndev);
    std::vector<int> fresh_members;  // members of this rank's best fresh seed

    // Member lists of the highest-ranked candidate clusters (the upcoming
    // winners) are kept on every rank; they are produced by extra blocks of
    // the regular batch launches, off the critical path. store_seeds lists
    // the seeds whose members are being computed in the current launch.
    const int TOPK = 4;
    std::vector<std::vector<int>> stored(N);
    std::vector<unsigned char> has(N, 0);
    std::vector<int> store_seeds, topk;

    // Upload the (initially empty) clustered mask to all GPUs of this rank
    for (int d = 0; d < ndev; ++d) {
        CUDA_CHECK(cudaSetDevice(devices[d].dev));
        CUDA_CHECK(cudaMemset(devices[d].d_clustered, 0, N));
    }

    // Main clustering loop
    while (true) {
        // --- 1. Cyclic distribution of the invalid seeds over the MPI ranks
        const int ninv = static_cast<int>(invalid.size());
        int total = 0;
        for (int r = 0; r < mpi_size; ++r) {
            counts[r] = (ninv > r) ? (ninv - r - 1) / mpi_size + 1 : 0;
            displs[r] = total;
            total += counts[r];
        }
        const int mine = counts[mpi_rank];
        my_seeds.resize(mine);
        my_cards.resize(mine);
        for (int m = 0; m < mine; ++m) my_seeds[m] = invalid[mpi_rank + m * mpi_size];

        // --- 2. Evaluate the candidate clusters: one OpenMP thread per GPU,
        //        seeds (my share of the invalid ones, then the member-store
        //        seeds) dealt cyclically over the GPUs of this rank
        const int nstore = static_cast<int>(store_seeds.size());
        const int nloc = mine + nstore;
        if (nloc > 0) {
            #pragma omp parallel num_threads(ndev)
            {
                const int d = omp_get_thread_num();
                std::vector<int>& dseeds = dev_seeds[d];
                std::vector<int>& dcards = dev_cards[d];
                std::vector<int>& dwant = dev_want[d];
                const int nd = (nloc > d) ? (nloc - d - 1) / ndev + 1 : 0;
                dseeds.resize(nd);
                dcards.resize(nd);
                dwant.clear();
                for (int m = 0; m < nd; ++m) {
                    const int g = d + m * ndev;
                    if (g < mine) {
                        dseeds[m] = my_seeds[g];
                    } else {
                        dseeds[m] = store_seeds[g - mine];
                        dwant.push_back(m);
                    }
                }
                // seeds at positions < nfresh_d are fresh (invalid) ones
                const int nfresh_d = (mine > d) ? (mine - d - 1) / ndev + 1 : 0;
                dev_fresh[d] = -1;
                if (nd > 0) {
                    computeCardinalities(devices[d], N, T2, dseeds.data(), dcards.data(), nd,
                                         dwant.data(), static_cast<int>(dwant.size()), stored,
                                         nfresh_d, dev_fresh[d], dev_fresh_members[d]);
                }
                for (int m = 0; m < nd; ++m) {
                    const int g = d + m * ndev;
                    if (g < mine) my_cards[g] = dcards[m];
                }
            }
        }
        for (int i = 0; i < nstore; ++i) {
            const int sd = store_seeds[i];
            has[sd] = !stored[sd].empty();
        }
        // This rank's best fresh seed (max cardinality, lowest index)
        int fresh_best = -1;
        {
            int fresh_card = -1, fd = -1;
            for (int m = 0; m < mine; ++m) {
                if (my_cards[m] > fresh_card || (my_cards[m] == fresh_card && my_seeds[m] < fresh_best)) {
                    fresh_card = my_cards[m];
                    fresh_best = my_seeds[m];
                }
            }
            for (int d = 0; d < ndev; ++d) {
                if (dev_fresh[d] == fresh_best) fd = d;
            }
            if (fd >= 0) {
                fresh_members.swap(dev_fresh_members[fd]);
            } else {
                fresh_members.clear();
            }
        }

        // --- 3. Exchange the new cardinalities
        all_cards.resize(total);
        MPI_Allgatherv(my_cards.data(), mine, MPI_INT, all_cards.data(), counts.data(),
                       displs.data(), MPI_INT, MPI_COMM_WORLD);
        for (int r = 0; r < mpi_size; ++r) {
            for (int m = 0; m < counts[r]; ++m) {
                card[invalid[r + m * mpi_size]] = all_cards[displs[r] + m];
            }
        }

        // --- 4. Select the best seed: max cardinality, lowest index on ties
        //        (same order as the sequential loop over unclustered seeds)
        int best_seed = -1;
        int max_cardinality = -1;
        for (int s = 0; s < N; ++s) {
            if (!clustered[s] && card[s] > max_cardinality) {
                max_cardinality = card[s];
                best_seed = s;
            }
        }
        if (best_seed < 0 || max_cardinality <= 0) break;

        // --- 5. The winning cluster's member list: usually already stored,
        //        otherwise rebuild it
        Cluster cluster;
        cluster.seed_point = best_seed;
        if (has[best_seed]) {
            cluster.members.swap(stored[best_seed]);
        } else {
            // A freshly computed winner: the rank that computed it has the
            // member list (unless it overflowed its slot) and broadcasts it
            int owner = -1;
            for (int r = 0; r < mpi_size && owner < 0; ++r) {
                for (int m = 0; m < counts[r]; ++m) {
                    if (invalid[r + m * mpi_size] == best_seed) {
                        owner = r;
                        break;
                    }
                }
            }
            int nmem = -1;
            if (owner == mpi_rank && fresh_best == best_seed && !fresh_members.empty()) {
                nmem = static_cast<int>(fresh_members.size());
            }
            if (owner >= 0 && mpi_size > 1) {
                MPI_Bcast(&nmem, 1, MPI_INT, owner, MPI_COMM_WORLD);
            }
            if (nmem > 0) {
                if (owner == mpi_rank) {
                    cluster.members.swap(fresh_members);
                } else {
                    cluster.members.resize(nmem);
                }
                if (mpi_size > 1) {
                    MPI_Bcast(cluster.members.data(), nmem, MPI_INT, owner, MPI_COMM_WORLD);
                }
            } else {
                computeMembers(devices[0], N, T2, best_seed, cluster.members);
            }
        }
        const int nB = static_cast<int>(cluster.members.size());

        // Mark all members as clustered
        for (int i = 0; i < nB; ++i) clustered[cluster.members[i]] = 1;
        for (int d = 0; d < ndev; ++d) {
            CUDA_CHECK(cudaSetDevice(devices[d].dev));
            CUDA_CHECK(cudaMemcpy(devices[d].d_clustered, clustered.data(), N,
                                  cudaMemcpyHostToDevice));
        }

        // --- 6. Determine which cached candidate clusters may have changed
        {
            DeviceCtx& ctx = devices[0];
            CUDA_CHECK(cudaSetDevice(ctx.dev));
            CUDA_CHECK(cudaMemcpy(ctx.d_members, cluster.members.data(), nB * sizeof(int),
                                  cudaMemcpyHostToDevice));
            const int blocks = (N + 255) / 256;
            markAffectedKernel<<<blocks, 256>>>(ctx.d_pts, ctx.d_ptsf, ctx.d_clustered,
                                                ctx.d_members, nB, N, T2, ctx.T2f_hi,
                                                ctx.twoA, ctx.d_flags);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(flags.data(), ctx.d_flags, N, cudaMemcpyDeviceToHost));
        }
        invalid.clear();
        for (int s = 0; s < N; ++s) {
            if (flags[s]) invalid.push_back(s);
        }

        // --- 7. Drop stored member lists that are no longer valid and pick
        //        the top-ranked unaffected seeds whose members are missing
        for (int i = 0; i < nB; ++i) {
            const int m = cluster.members[i];
            if (has[m]) {
                has[m] = 0;
                std::vector<int>().swap(stored[m]);
            }
        }
        for (int s : invalid) {
            if (has[s]) {
                has[s] = 0;
                std::vector<int>().swap(stored[s]);
            }
        }
        topk.clear();
        for (int s = 0; s < N; ++s) {
            if (clustered[s] || flags[s]) continue;
            // insert s into the (small) sorted top list: card desc, index asc
            if (static_cast<int>(topk.size()) == TOPK && card[topk.back()] >= card[s]) continue;
            int p = static_cast<int>(topk.size());
            if (p < TOPK) topk.push_back(s); else p = TOPK - 1;
            while (p > 0 && card[topk[p - 1]] < card[s]) {
                topk[p] = topk[p - 1];
                --p;
            }
            topk[p] = s;
        }
        store_seeds.clear();
        for (int s : topk) {
            if (!has[s]) store_seeds.push_back(s);
        }
        std::sort(store_seeds.begin(), store_seeds.end());

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
    
    // Check each cluster
    for (size_t c = 0; c < clusters.size(); ++c) {
        const auto& cluster = clusters[c];
        double max_diameter = 0.0;
        
        // Check diameter (max distance between any two points)
        const int n = static_cast<int>(cluster.members.size());
        #pragma omp parallel for schedule(dynamic, 16) reduction(max : max_diameter)
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
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
    MPI_Init(&argc, &argv);
    int mpi_rank = 0, mpi_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    int num_points = 1000;
    double threshold = 2.0;
    bool validate = false;
    bool printResults = false;
    int exit_code = 0;
    
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

    // Assign GPUs: the ranks sharing a node split the node's GPUs among them
    int local_rank = 0, local_size = 1;
    {
        MPI_Comm local_comm;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, mpi_rank, MPI_INFO_NULL,
                            &local_comm);
        MPI_Comm_rank(local_comm, &local_rank);
        MPI_Comm_size(local_comm, &local_size);
        MPI_Comm_free(&local_comm);
    }
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
        fprintf(stderr, "Rank %d: no CUDA device available\n", mpi_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    std::vector<int> my_devices;
    if (local_size >= device_count) {
        my_devices.push_back(local_rank % device_count);
    } else {
        for (int g = local_rank; g < device_count; g += local_size) my_devices.push_back(g);
    }
    std::vector<DeviceCtx> devices(my_devices.size());
    for (size_t d = 0; d < my_devices.size(); ++d) {
        initDevice(devices[d], my_devices[d], points, sqCutoff(threshold));
    }
    if (mpi_rank == 0) {
        printf("MPI ranks: %d, GPUs on rank 0: %zu, OpenMP threads: %d\n",
               mpi_size, my_devices.size(), omp_get_max_threads());
    }
    
    // Perform QT clustering
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_start = std::chrono::high_resolution_clock::now();
    
    const std::vector<Cluster> clusters = qtClustering(points, threshold, devices,
                                                       mpi_rank, mpi_size);
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);

    for (auto& ctx : devices) freeDevice(ctx);

    if (mpi_rank == 0) {
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
            } else {
                printf("Validation: FAILED\n");
                exit_code = 1;
            }
        }
    }

    MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exit_code;
}
