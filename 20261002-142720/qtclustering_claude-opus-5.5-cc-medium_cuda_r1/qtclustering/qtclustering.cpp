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
#include <climits>

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
// Semantics are identical to the sequential algorithm:
//  * For a seed, the candidate cluster grows by repeatedly adding the
//    unclustered point with the smallest max-distance to all current members
//    (ties -> lowest index), as long as that max-distance is < threshold.
//  * The per-candidate max-distance is maintained incrementally as the max of
//    squared distances. Since sqrt is correctly rounded and monotone,
//    max_i sqrt(sq_i) == sqrt(max_i sq_i) exactly, and sqrt(sq) < threshold
//    <=> sq <= sq_cut for a precomputed exact cutoff. Updates are first
//    decided with a conservative FP32 filter (with a proven error margin);
//    only undecidable cases are evaluated exactly in FP64. The argmin is
//    selected via float keys and proven unique, else resolved with exact
//    sqrt values (including ties after rounding).
//  * Candidates whose max-distance reached the threshold are dropped
//    permanently (max-distance is monotone).
//  * Only points within threshold of the seed can ever join, so candidates
//    are gathered from a uniform grid with cell size >= threshold.
//  * Each round the seed with the largest cardinality (ties -> lowest index)
//    wins. A seed's cached candidate cluster only changes if one of its
//    members gets clustered (removing never-chosen candidates changes nothing);
//    all its members lie within threshold of the seed, so only seeds within
//    threshold of a newly clustered point are checked/recomputed.
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err__ = (call);                                               \
        if (err__ != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err__), \
                    __FILE__, __LINE__);                                          \
            exit(1);                                                              \
        }                                                                         \
    } while (0)

struct GridInfo {
    double inv_cs;
    int gw, gh;
};

struct DevData {
    const double* px;       // original order
    const double* py;
    const double* sx;       // sorted by grid cell
    const double* sy;
    const int* sidx;        // original index of sorted entry
    const int* cell_start;  // gw*gh + 1
    unsigned char* clustered;
    int* card;              // cached cardinality per seed (-1 = clustered)
    int* dirty;
    int* dirty_list;
    int* dirty_count;
    int* work_counter;
    int* offset;            // number of points clustered so far
    int* num_clusters;
    int* cl_seed;           // per cluster: seed
    int* cl_size;           // per cluster: cardinality
    int* members;           // all cluster members, cluster after cluster
    int* store;             // cached members per seed (nullptr if disabled)
    double* gws;            // global workspaces (nullptr if unused)
    unsigned long long* best;
    long long cap;          // max candidates per seed
    GridInfo g;
    const float2* pf;       // float copies of the coordinates (original order)
    double sq_cut;          // sqrt(sq) < threshold  <=>  sq <= sq_cut
    float sq_cut_f;         // float(sq_cut)
    float margin;           // bound on |float sq - exact sq| (incl. conversions)
    int N;
};

// Candidate workspace: v = {float x, float y, float(max sq. distance), index
// bits}, d = exact max squared distance, qu = queue of slots needing FP64 work
struct Work {
    float4* v;
    double* d;
    int* qu;
    __device__ Work(double* base, long long cap)
        : v(reinterpret_cast<float4*>(base)), d(base + 2 * cap),
          qu(reinterpret_cast<int*>(base + 3 * cap)) {}
};

// Workspace stride in doubles (keeps every workspace 16-byte aligned)
__host__ __device__ inline long long workStride(long long cap) {
    long long s = 3 * cap + (cap + 1) / 2;
    return s + (s & 1);
}

// Squared distance, computed exactly like the reference distance()
__device__ __forceinline__ double devSq(double ax, double ay, double bx, double by) {
    const double dx = ax - bx;
    const double dy = ay - by;
    return dx * dx + dy * dy;
}

__device__ __forceinline__ int cellCoord(double v, double inv_cs, int g) {
    int c = static_cast<int>(floor(v * inv_cs));
    return c < 0 ? 0 : (c >= g ? g - 1 : c);
}

__device__ __forceinline__ bool better(double d, int i, double bd, int bi) {
    return d < bd || (d == bd && i < bi);
}

__device__ __forceinline__ int seedOf(unsigned long long key) {
    return static_cast<int>(0xFFFFFFFFu - static_cast<unsigned>(key & 0xFFFFFFFFu));
}

// Per-lane tracking of the smallest float key (z = float(max sq), index) and
// of the second smallest z, used to prove uniqueness of the exact argmin.
struct LaneMin {
    float z1, z2;
    int q1;
    __device__ void reset() { z1 = INFINITY; z2 = INFINITY; q1 = INT_MAX; }
    __device__ void add(float z, int q) {
        if (z < z1 || (z == z1 && q < q1)) { z2 = z1; z1 = z; q1 = q; }
        else if (z < z2) z2 = z;
    }
};

// Select the next member: min over candidates of (sqrt(exact max sq), index).
// float(.) is monotone, so the exact minimum has z == zmin; any candidate with
// z > next_float(zmin) has a strictly larger exact sqrt. If only one candidate
// has z <= next_float(zmin) it is the winner; otherwise resolve exactly.
// Returns INT_MAX if there are no candidates.
__device__ __forceinline__ int selectBest(const LaneMin& lm, const Work& w, int n) {
    float zmin = lm.z1;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) zmin = fminf(zmin, __shfl_xor_sync(0xffffffffu, zmin, off));
    if (zmin == INFINITY) return INT_MAX;
    const float T = nextafterf(zmin, INFINITY);
    int cnt = (lm.z1 <= T) + (lm.z2 <= T);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) cnt += __shfl_xor_sync(0xffffffffu, cnt, off);
    if (cnt == 1) {
        const unsigned own = __ballot_sync(0xffffffffu, lm.z1 <= T);
        return __shfl_sync(0xffffffffu, lm.q1, __ffs(own) - 1);
    }
    // Rare: near-ties in float, resolve with exact sqrt values
    const int lane = threadIdx.x & 31;
    double br = INFINITY;
    int bi = INT_MAX;
    for (int j = lane; j < n; j += 32) {
        const float4 v = w.v[j];
        const int q = __float_as_int(v.w);
        if (q >= 0 && v.z <= T) {
            const double r = sqrt(w.d[j]);
            if (better(r, q, br, bi)) { br = r; bi = q; }
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const double orr = __shfl_xor_sync(0xffffffffu, br, off);
        const int oi = __shfl_xor_sync(0xffffffffu, bi, off);
        if (better(orr, oi, br, bi)) { br = orr; bi = oi; }
    }
    return bi;
}

// Grow the candidate cluster of seed s with one warp. Returns the cardinality.
// If out != nullptr, members are written in insertion order.
__device__ int growCluster(const DevData& D, const int s, Work w, int* out) {
    const int lane = threadIdx.x & 31;
    const unsigned lt = (1u << lane) - 1u;
    const double cut = D.sq_cut;
    const float cut_f = D.sq_cut_f;
    const float margin = D.margin;
    const double s_x = D.px[s], s_y = D.py[s];
    const int cx = cellCoord(s_x, D.g.inv_cs, D.g.gw);
    const int cy = cellCoord(s_y, D.g.inv_cs, D.g.gh);

    LaneMin lm;
    lm.reset();
    int n = 0;

    // Gather candidates: unclustered points within threshold of the seed
    for (int yy = max(cy - 1, 0); yy <= min(cy + 1, D.g.gh - 1); ++yy) {
        const int row = yy * D.g.gw;
        const int lo = D.cell_start[row + max(cx - 1, 0)];
        const int hi = D.cell_start[row + min(cx + 1, D.g.gw - 1) + 1];
        for (int base = lo; base < hi; base += 32) {
            const int j = base + lane;
            bool keep = false;
            double sq = 0;
            int q = -1;
            if (j < hi) {
                q = D.sidx[j];
                if (q != s && !D.clustered[q]) {
                    sq = devSq(D.sx[j], D.sy[j], s_x, s_y);
                    keep = sq <= cut;
                }
            }
            const unsigned m = __ballot_sync(0xffffffffu, keep);
            if (keep) {
                const int pos = n + __popc(m & lt);
                const float2 pf = D.pf[q];
                const float z = static_cast<float>(sq);
                w.v[pos] = make_float4(pf.x, pf.y, z, __int_as_float(q));
                w.d[pos] = sq;
                lm.add(z, q);
            }
            n += __popc(m);
        }
    }
    __syncwarp();

    int K = 1;
    if (out && lane == 0) out[0] = s;
    int alive = n;

    while (true) {
        const int mi = selectBest(lm, w, n);
        if (mi == INT_MAX) break;  // No more points can be added
        if (out && lane == 0) out[K] = mi;
        ++K;
        const double m_x = D.px[mi], m_y = D.py[mi];
        const float2 mf = D.pf[mi];

        // Phase A (FP32): kill the new member and certainly-dead candidates
        // in place; candidates whose max is certainly unchanged stay as they
        // are; uncertain ones are queued for exact evaluation.
        lm.reset();
        int nq = 0, dead = 0;
        for (int base = 0; base < n; base += 32) {
            const int p = base + lane;
            bool kill = false, unsure = false;
            if (p < n) {
                const float4 v = w.v[p];
                const int q = __float_as_int(v.w);
                if (q >= 0) {
                    if (q == mi) {
                        kill = true;
                    } else {
                        // Conservative single-precision filter
                        const float fdx = v.x - mf.x, fdy = v.y - mf.y;
                        const float sqf = fmaf(fdx, fdx, fdy * fdy);
                        if (sqf - margin > cut_f) kill = true;
                        else if (!(sqf + margin < v.z)) unsure = true;
                        else lm.add(v.z, q);
                    }
                }
            }
            if (kill) reinterpret_cast<int*>(w.v + p)[3] = -1;
            const unsigned mu = __ballot_sync(0xffffffffu, unsure);
            if (unsure) w.qu[nq + __popc(mu & lt)] = p;
            nq += __popc(mu);
            dead += __popc(__ballot_sync(0xffffffffu, kill));
        }
        __syncwarp();

        // Phase B (FP64): exact update of queued candidates, all lanes busy
        for (int base = 0; base < nq; base += 32) {
            const int t = base + lane;
            bool kill = false;
            if (t < nq) {
                const int p = w.qu[t];
                const int q = __float_as_int(w.v[p].w);
                const double sq = fmax(w.d[p], devSq(D.px[q], D.py[q], m_x, m_y));
                if (sq <= cut) {
                    const float z = static_cast<float>(sq);
                    reinterpret_cast<float*>(w.v + p)[2] = z;
                    w.d[p] = sq;
                    lm.add(z, q);
                } else {
                    reinterpret_cast<int*>(w.v + p)[3] = -1;
                    kill = true;
                }
            }
            dead += __popc(__ballot_sync(0xffffffffu, kill));
        }
        alive -= dead;
        __syncwarp();

        // Occasional warp-wide compaction of dead entries
        if (2 * alive < n) {
            int nc = 0;
            for (int base = 0; base < n; base += 32) {
                const int p = base + lane;
                float4 v = make_float4(0.f, 0.f, 0.f, __int_as_float(-1));
                double d = 0;
                if (p < n) v = w.v[p];
                const bool keep = __float_as_int(v.w) >= 0;
                if (keep) d = w.d[p];
                const unsigned mk = __ballot_sync(0xffffffffu, keep);
                if (keep) {
                    const int np = nc + __popc(mk & lt);
                    if (np != p) { w.v[np] = v; w.d[np] = d; }
                }
                nc += __popc(mk);
                __syncwarp();
            }
            n = nc;
        }
    }
    return K;
}

// Warp-per-seed computation of dirty seeds (dynamic scheduling)
template <bool SMEM>
__global__ void computeKernel(DevData D) {
    extern __shared__ double smem_ws[];
    const int count = *D.dirty_count;
    const int lane = threadIdx.x & 31;
    const long long stride = workStride(D.cap);
    const long long warp = (static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
    Work w(SMEM ? smem_ws + (threadIdx.x >> 5) * stride : D.gws + warp * stride, D.cap);
    while (true) {
        int t = 0;
        if (lane == 0) t = atomicAdd(D.work_counter, 1);
        t = __shfl_sync(0xffffffffu, t, 0);
        if (t >= count) break;
        const int s = D.dirty_list[t];
        int* out = D.store ? D.store + static_cast<long long>(s) * (D.cap + 1) : nullptr;
        if (out) {
            // The cached cluster stays valid unless one of its members got clustered
            const int prev = D.card[s];
            bool hit = prev <= 0;
            for (int k = lane; k < prev && !hit; k += 32) hit = D.clustered[out[k]];
            if (!__any_sync(0xffffffffu, hit)) {
                if (lane == 0) D.dirty[s] = 0;
                continue;
            }
        }
        const int K = growCluster(D, s, w, out);
        if (lane == 0) {
            D.card[s] = K;
            D.dirty[s] = 0;
        }
    }
}

// Argmax of cardinality, ties -> lowest index
__global__ void argMaxKernel(DevData D) {
    unsigned long long key = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < D.N; i += gridDim.x * blockDim.x) {
        const int c = D.card[i];
        if (c > 0) {
            const unsigned long long k = (static_cast<unsigned long long>(c) << 32) |
                                         (0xFFFFFFFFu - static_cast<unsigned>(i));
            if (k > key) key = k;
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const unsigned long long o = __shfl_xor_sync(0xffffffffu, key, off);
        if (o > key) key = o;
    }
    __shared__ unsigned long long sm[32];
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) sm[wid] = key;
    __syncthreads();
    if (wid == 0) {
        key = lane < (blockDim.x >> 5) ? sm[lane] : 0ull;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const unsigned long long o = __shfl_xor_sync(0xffffffffu, key, off);
            if (o > key) key = o;
        }
        if (lane == 0 && key) atomicMax(D.best, key);
    }
}

// Regenerate the winning cluster's members (single warp, no member store)
template <bool SMEM>
__global__ void extractKernel(DevData D) {
    extern __shared__ double smem_ws[];
    const unsigned long long key = *D.best;
    if (key == 0) return;
    Work w(SMEM ? smem_ws : D.gws, D.cap);
    growCluster(D, seedOf(key), w, D.members + *D.offset);
}

// Record the winning cluster's members and mark them clustered
__global__ void markClusteredKernel(DevData D) {
    const unsigned long long key = *D.best;
    if (key == 0) return;
    const int K = static_cast<int>(key >> 32);
    const int seed = seedOf(key);
    int* dst = D.members + *D.offset;
    const int* src = D.store ? D.store + static_cast<long long>(seed) * (D.cap + 1) : dst;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < K; i += gridDim.x * blockDim.x) {
        const int c = src[i];
        if (D.store) dst[i] = c;
        D.clustered[c] = 1;
        D.card[c] = -1;
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *D.dirty_count = 0;
        *D.work_counter = 0;
    }
}

// Mark unclustered seeds within threshold of newly clustered points as dirty
__global__ void markDirtyKernel(DevData D) {
    const unsigned long long key = *D.best;
    if (key == 0) return;
    const int K = static_cast<int>(key >> 32);
    const int* mem = D.members + *D.offset;
    const int lane = threadIdx.x & 31;
    const int nwarps = (gridDim.x * blockDim.x) >> 5;
    for (int wi = (blockIdx.x * blockDim.x + threadIdx.x) >> 5; wi < K; wi += nwarps) {
        const int c = mem[wi];
        const double c_x = D.px[c], c_y = D.py[c];
        const int cx = cellCoord(c_x, D.g.inv_cs, D.g.gw);
        const int cy = cellCoord(c_y, D.g.inv_cs, D.g.gh);
        for (int yy = max(cy - 1, 0); yy <= min(cy + 1, D.g.gh - 1); ++yy) {
            const int row = yy * D.g.gw;
            const int lo = D.cell_start[row + max(cx - 1, 0)];
            const int hi = D.cell_start[row + min(cx + 1, D.g.gw - 1) + 1];
            for (int j = lo + lane; j < hi; j += 32) {
                const int q = D.sidx[j];
                // c joined q's cluster only if dist(c, q) < threshold
                if (!D.clustered[q] && devSq(c_x, c_y, D.sx[j], D.sy[j]) <= D.sq_cut) {
                    if (atomicExch(&D.dirty[q], 1) == 0) {
                        D.dirty_list[atomicAdd(D.dirty_count, 1)] = q;
                    }
                }
            }
        }
    }
}

__global__ void finishRoundKernel(DevData D) {
    const unsigned long long key = *D.best;
    if (key == 0) return;
    const int K = static_cast<int>(key >> 32);
    const int c = *D.num_clusters;
    D.cl_seed[c] = seedOf(key);
    D.cl_size[c] = K;
    *D.num_clusters = c + 1;
    *D.offset += K;
    *D.best = 0;
}

// Upper bound of candidate count for any seed
__global__ void maxDegreeKernel(DevData D, int* maxdeg) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= D.N) return;
    const double s_x = D.px[warp], s_y = D.py[warp];
    const int cx = cellCoord(s_x, D.g.inv_cs, D.g.gw);
    const int cy = cellCoord(s_y, D.g.inv_cs, D.g.gh);
    int cnt = 0;
    for (int yy = max(cy - 1, 0); yy <= min(cy + 1, D.g.gh - 1); ++yy) {
        const int row = yy * D.g.gw;
        const int lo = D.cell_start[row + max(cx - 1, 0)];
        const int hi = D.cell_start[row + min(cx + 1, D.g.gw - 1) + 1];
        for (int j = lo + lane; j < hi; j += 32) {
            if (D.sidx[j] != warp && devSq(D.sx[j], D.sy[j], s_x, s_y) <= D.sq_cut) ++cnt;
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) cnt += __shfl_xor_sync(0xffffffffu, cnt, off);
    if (lane == 0) atomicMax(maxdeg, cnt);
}

// Largest squared distance whose (correctly rounded) sqrt is < threshold
static double squaredCutoff(const double threshold) {
    double x = threshold * threshold;
    while (x > 0 && !(std::sqrt(x) < threshold)) x = std::nextafter(x, 0.0);
    while (std::sqrt(std::nextafter(x, INFINITY)) < threshold) x = std::nextafter(x, INFINITY);
    return std::sqrt(x) < threshold ? x : -1.0;
}

// Main QT clustering algorithm
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<Cluster> clusters;
    if (N == 0) return clusters;

    // Uniform grid with cell size >= threshold
    const double cs = std::max(threshold, std::max(MAX_WIDTH, MAX_HEIGHT) / 1024.0) * (1.0 + 1e-9);
    GridInfo g;
    g.inv_cs = 1.0 / cs;
    g.gw = std::max(1, static_cast<int>(std::ceil(MAX_WIDTH / cs)));
    g.gh = std::max(1, static_cast<int>(std::ceil(MAX_HEIGHT / cs)));
    const int numCells = g.gw * g.gh;
    auto hostCell = [&](double v, int gdim) {
        int c = static_cast<int>(std::floor(v * g.inv_cs));
        return c < 0 ? 0 : (c >= gdim ? gdim - 1 : c);
    };

    std::vector<double> hpx(N), hpy(N), hsx(N), hsy(N);
    std::vector<int> hsidx(N), hcell(N), hstart(numCells + 1, 0);
    for (int i = 0; i < N; ++i) {
        hpx[i] = points[i].x;
        hpy[i] = points[i].y;
        hcell[i] = hostCell(points[i].y, g.gh) * g.gw + hostCell(points[i].x, g.gw);
        hstart[hcell[i] + 1]++;
    }
    for (int c = 0; c < numCells; ++c) hstart[c + 1] += hstart[c];
    {
        std::vector<int> fill(hstart.begin(), hstart.end() - 1);
        for (int i = 0; i < N; ++i) {
            const int p = fill[hcell[i]]++;
            hsx[p] = hpx[i];
            hsy[p] = hpy[i];
            hsidx[p] = i;
        }
    }

    // Device arrays
    double *d_px, *d_py, *d_sx, *d_sy;
    float2* d_pf;
    int *d_sidx, *d_start, *d_card, *d_dirty, *d_dlist, *d_counters, *d_members, *d_clseed, *d_clsize;
    unsigned char* d_clustered;
    unsigned long long* d_best;
    CUDA_CHECK(cudaMalloc(&d_px, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_py, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_sx, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_pf, N * sizeof(float2)));
    CUDA_CHECK(cudaMalloc(&d_sy, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_sidx, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_start, (numCells + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_card, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_dirty, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_dlist, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_counters, 5 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_members, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_clseed, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_clsize, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_clustered, N));
    CUDA_CHECK(cudaMalloc(&d_best, sizeof(unsigned long long)));

    CUDA_CHECK(cudaMemcpy(d_px, hpx.data(), N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_py, hpy.data(), N * sizeof(double), cudaMemcpyHostToDevice));
    {
        std::vector<float2> hpf(N);
        for (int i = 0; i < N; ++i) hpf[i] = make_float2(static_cast<float>(hpx[i]), static_cast<float>(hpy[i]));
        CUDA_CHECK(cudaMemcpy(d_pf, hpf.data(), N * sizeof(float2), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(d_sx, hsx.data(), N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sy, hsy.data(), N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sidx, hsidx.data(), N * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_start, hstart.data(), (numCells + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_clustered, 0, N));
    CUDA_CHECK(cudaMemset(d_card, 0, N * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_dirty, 0, N * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_best, 0, sizeof(unsigned long long)));
    {
        // Initially every point is a dirty seed
        std::vector<int> all(N);
        for (int i = 0; i < N; ++i) all[i] = i;
        CUDA_CHECK(cudaMemcpy(d_dlist, all.data(), N * sizeof(int), cudaMemcpyHostToDevice));
        // dirty_count, work_counter, maxdeg, offset, num_clusters
        const int init[5] = {N, 0, 0, 0, 0};
        CUDA_CHECK(cudaMemcpy(d_counters, init, 5 * sizeof(int), cudaMemcpyHostToDevice));
    }

    DevData D{};
    D.px = d_px; D.py = d_py; D.sx = d_sx; D.sy = d_sy; D.sidx = d_sidx;
    D.cell_start = d_start; D.clustered = d_clustered; D.card = d_card;
    D.dirty = d_dirty; D.dirty_list = d_dlist;
    D.dirty_count = d_counters; D.work_counter = d_counters + 1;
    D.offset = d_counters + 3; D.num_clusters = d_counters + 4;
    D.cl_seed = d_clseed; D.cl_size = d_clsize; D.members = d_members;
    D.best = d_best;
    D.g = g; D.sq_cut = squaredCutoff(threshold); D.N = N;
    D.pf = d_pf;
    D.sq_cut_f = static_cast<float>(D.sq_cut);
    {
        // Error bound of the FP32 squared distance (and of the float
        // conversions of od / sq_cut) relative to the exact FP64 values, for
        // coordinates bounded by M in magnitude: well below 128 * M^2 * 2^-24.
        double M = 1.0;
        for (int i = 0; i < N; ++i) M = std::max(M, std::max(std::fabs(hpx[i]), std::fabs(hpy[i])));
        D.margin = static_cast<float>(128.0 * M * M * std::ldexp(1.0, -24));
    }

    constexpr int TPB = 256;             // generic threads per block
    constexpr int WTPB = 128;            // threads per block, compute kernel
    constexpr int WPB = WTPB / 32;
    maxDegreeKernel<<<(N + TPB / 32 - 1) / (TPB / 32), TPB>>>(D, d_counters + 2);
    CUDA_CHECK(cudaGetLastError());
    int maxdeg = 0;
    CUDA_CHECK(cudaMemcpy(&maxdeg, d_counters + 2, sizeof(int), cudaMemcpyDeviceToHost));

    // Resource sizing
    int device = 0, numSM = 1, maxThrPerSM = 2048, maxSmemOptin = 48 * 1024;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(&numSM, cudaDevAttrMultiProcessorCount, device));
    CUDA_CHECK(cudaDeviceGetAttribute(&maxThrPerSM, cudaDevAttrMaxThreadsPerMultiProcessor, device));
    CUDA_CHECK(cudaDeviceGetAttribute(&maxSmemOptin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    const long long cap = std::max(1, maxdeg);
    D.cap = cap;
    const size_t perWorker = workStride(cap) * sizeof(double);

    // Shared-memory workspaces when they fit without hurting occupancy much
    const size_t warpSmem = WPB * perWorker;
    const bool useSmem = warpSmem <= static_cast<size_t>(maxSmemOptin) / 2;
    int computeBlocks = numSM * (maxThrPerSM / WTPB);
    if (useSmem) {
        CUDA_CHECK(cudaFuncSetAttribute(computeKernel<true>,
                   cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(warpSmem)));
        CUDA_CHECK(cudaFuncSetAttribute(extractKernel<true>,
                   cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(perWorker)));
        int per = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, computeKernel<true>,
                   WTPB, warpSmem));
        computeBlocks = numSM * std::max(1, per);
    }
    computeBlocks = std::min(computeBlocks, (N + WPB - 1) / WPB);

    size_t freeMem = 0, totalMem = 0;
    CUDA_CHECK(cudaMemGetInfo(&freeMem, &totalMem));

    // Per-seed member store avoids regenerating the winning cluster and allows
    // exact cache-invalidation checks
    const size_t storeBytes = static_cast<size_t>(N) * (cap + 1) * sizeof(int);
    int* d_store = nullptr;
    if (storeBytes <= freeMem / 4) {
        CUDA_CHECK(cudaMalloc(&d_store, storeBytes));
        freeMem -= storeBytes;
    }
    D.store = d_store;

    // Global workspaces (only when shared memory is not used)
    double* d_gws = nullptr;
    if (!useSmem) {
        const long long maxWarps = std::max<long long>(WPB, static_cast<long long>((freeMem / 2) / perWorker));
        computeBlocks = static_cast<int>(std::max<long long>(1, std::min<long long>(computeBlocks, maxWarps / WPB)));
        CUDA_CHECK(cudaMalloc(&d_gws, static_cast<size_t>(computeBlocks) * WPB * perWorker));
    }
    D.gws = d_gws;

    const int amBlocks = std::min((N + TPB - 1) / TPB, numSM * 4);
    const int markBlocks = std::min((N + TPB - 1) / TPB, numSM * 2);

    // Main clustering loop: each round forms one cluster. Rounds run entirely
    // on the device; the host only checks for completion in batches (rounds
    // after completion are no-ops).
    auto enqueueRound = [&](cudaStream_t st) {
        if (useSmem) computeKernel<true><<<computeBlocks, WTPB, warpSmem, st>>>(D);
        else computeKernel<false><<<computeBlocks, WTPB, 0, st>>>(D);
        argMaxKernel<<<amBlocks, TPB, 0, st>>>(D);
        if (!d_store) {
            if (useSmem) extractKernel<true><<<1, 32, perWorker, st>>>(D);
            else extractKernel<false><<<1, 32, 0, st>>>(D);
        }
        markClusteredKernel<<<markBlocks, TPB, 0, st>>>(D);
        markDirtyKernel<<<markBlocks, TPB, 0, st>>>(D);
        finishRoundKernel<<<1, 1, 0, st>>>(D);
    };

    int batch = 4;  // rounds between completion checks
    int offset = 0, numClusters = 0, prevClusters = -1;
    while (true) {
        for (int r = 0; r < batch; ++r) enqueueRound(0);
        CUDA_CHECK(cudaGetLastError());
        int st[2];
        CUDA_CHECK(cudaMemcpy(st, d_counters + 3, 2 * sizeof(int), cudaMemcpyDeviceToHost));
        offset = st[0];
        numClusters = st[1];
        if (offset >= N || numClusters == prevClusters) break;  // done / no progress
        prevClusters = numClusters;
        batch = std::min(batch * 2, 64);
    }

    std::vector<int> hmembers(N), hseed(numClusters), hsize(numClusters);
    CUDA_CHECK(cudaMemcpy(hmembers.data(), d_members, offset * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hseed.data(), d_clseed, numClusters * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hsize.data(), d_clsize, numClusters * sizeof(int), cudaMemcpyDeviceToHost));
    clusters.reserve(numClusters);
    int pos = 0;
    for (int c = 0; c < numClusters; ++c) {
        Cluster cluster;
        cluster.seed_point = hseed[c];
        cluster.members.assign(hmembers.begin() + pos, hmembers.begin() + pos + hsize[c]);
        clusters.push_back(std::move(cluster));
        pos += hsize[c];
    }

    if (d_gws) cudaFree(d_gws);
    if (d_store) cudaFree(d_store);
    cudaFree(d_pf);
    cudaFree(d_px); cudaFree(d_py); cudaFree(d_sx); cudaFree(d_sy);
    cudaFree(d_sidx); cudaFree(d_start); cudaFree(d_card); cudaFree(d_dirty);
    cudaFree(d_dlist); cudaFree(d_counters); cudaFree(d_members);
    cudaFree(d_clseed); cudaFree(d_clsize);
    cudaFree(d_clustered); cudaFree(d_best);

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
    
    // Initialize the CUDA context (and load all kernels) outside the timed region
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
