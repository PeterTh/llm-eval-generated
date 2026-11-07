// QT Clustering Benchmark - CUDA Parallel Version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// Parallelization strategy (CUDA):
//   * Every unclustered point is tried as a seed by its own warp (or, for seeds
//     with very many neighbours, by its own thread block). Only points within
//     the threshold of the seed can ever join, so these live candidates are
//     compacted (index + running "max distance to the cluster") and each growth
//     step costs O(live / warp or block size) plus one (min value, min index)
//     reduction.
//   * Member lists of all candidate clusters of a batch are kept on the
//     device; a small selection kernel picks the largest one (ties -> earliest
//     seed, exactly like the sequential code) and keeps it as the running best.
//   * The host only receives the winning cluster per outer iteration, marks
//     its members as clustered and compacts the remaining points.
//   * A seed's candidate cluster depends only on the unclustered points within
//     the threshold of the seed, so cardinalities are cached and only seeds
//     whose neighbourhood lost a point are recomputed in later iterations.

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

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__,    \
                    __LINE__, cudaGetErrorString(err__));                       \
            exit(EXIT_FAILURE);                                                 \
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
// Device kernels
// ---------------------------------------------------------------------------

constexpr int GROW_THREADS = 512;                 // threads per seed block
constexpr int GROW_WARPS = GROW_THREADS / 32;
constexpr double DEV_INF = 1.0e300;               // "excluded candidate" marker

// Combine two (value, index) candidates: smaller value wins, ties -> smaller index.
__device__ __forceinline__ void betterOf(double& v, int& i, const double ov, const int oi) {
    if (ov < v || (ov == v && oi < i)) {
        v = ov;
        i = oi;
    }
}

// Block-wide argmin of (value, index). All threads receive the result.
__device__ __forceinline__ void blockArgMin(double& v, int& i, double* s_val, int* s_idx) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const double ov = __shfl_down_sync(0xffffffffu, v, off);
        const int oi = __shfl_down_sync(0xffffffffu, i, off);
        betterOf(v, i, ov, oi);
    }
    if (lane == 0) {
        s_val[warp] = v;
        s_idx[warp] = i;
    }
    __syncthreads();
    if (warp == 0) {
        v = (lane < GROW_WARPS) ? s_val[lane] : DEV_INF;
        i = (lane < GROW_WARPS) ? s_idx[lane] : -1;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const double ov = __shfl_down_sync(0xffffffffu, v, off);
            const int oi = __shfl_down_sync(0xffffffffu, i, off);
            betterOf(v, i, ov, oi);
        }
        if (lane == 0) {
            s_val[0] = v;
            s_idx[0] = i;
        }
    }
    __syncthreads();
    v = s_val[0];
    i = s_idx[0];
    __syncthreads();   // protect s_val/s_idx before the next reduction reuses them
}

// Block-per-seed kernel (fallback for seeds with many neighbours).
//
// After placing the seed, the live candidates (index + running max distance)
// are compacted once into a per-block global list. Every growth step then
// touches only that list: O(live / blockDim) work per thread plus one
// block-wide (min value, min index) reduction.
//
// members[b * M + k] receives the k-th member (local index) of the cluster,
// cards[b] its cardinality. The kernel only processes slots whose cards entry
// was set to -1 by the warp kernel.
__global__ void __launch_bounds__(GROW_THREADS)
growClustersBlockKernel(const Point* __restrict__ pts,
                        const int M,
                        const int* __restrict__ seeds,
                        const int* __restrict__ unclIdx,
                        const double threshold,
                        int* __restrict__ members,
                        const int persistent,
                        const int rowStride,
                        int* __restrict__ cards,
                        double* __restrict__ wsD,
                        int* __restrict__ wsI) {
    __shared__ double s_val[GROW_WARPS];
    __shared__ int s_idx[GROW_WARPS];
    __shared__ int s_cnt[GROW_WARPS];

    const int slot = blockIdx.x;
    if (cards[slot] != -1) return;   // already handled by the warp kernel
    const int seed = seeds[slot];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const unsigned lanemask_lt = (1u << lane) - 1u;
    const int row = persistent ? unclIdx[seed] : slot;
    int* __restrict__ myMembers = members + static_cast<size_t>(row) * rowStride;
    double* __restrict__ d = wsD + static_cast<size_t>(slot) * M;
    int* __restrict__ ix = wsI + static_cast<size_t>(slot) * M;

    Point mp;
    {
        const double2 sp = __ldg(reinterpret_cast<const double2*>(pts) + seed);
        mp.x = sp.x;
        mp.y = sp.y;
    }

    // Scan all candidates; compact those within threshold of the seed.
    int n = 0;
    double bestV = DEV_INF;
    int bestI = -1;
    for (int base = 0; base < M; base += GROW_THREADS) {
        const int c = base + tid;
        bool alive = false;
        double dist = 0.0;
        if (c < M && c != seed) {
            const double2 cp = __ldg(reinterpret_cast<const double2*>(pts) + c);
            dist = distance(Point{cp.x, cp.y}, mp);
            alive = dist < threshold;
        }
        const unsigned mask = __ballot_sync(0xffffffffu, alive);
        if (lane == 0) s_cnt[warp] = __popc(mask);
        __syncthreads();
        int prefix = 0, total = 0;
#pragma unroll
        for (int w = 0; w < GROW_WARPS; ++w) {
            const int cw = s_cnt[w];
            prefix += (w < warp) ? cw : 0;
            total += cw;
        }
        if (alive) {
            const int pos = n + prefix + __popc(mask & lanemask_lt);
            d[pos] = dist;
            ix[pos] = c;
            if (dist < bestV) {
                bestV = dist;
                bestI = c;
            }
        }
        n += total;
        __syncthreads();
    }

    if (tid == 0) myMembers[0] = unclIdx[seed];
    int count = 1;

    // Iteratively add the closest admissible point.
    while (true) {
        blockArgMin(bestV, bestI, s_val, s_idx);
        if (bestI < 0) break;   // no candidate keeps the diameter below threshold

        const int newMember = bestI;
        if (tid == 0) myMembers[count] = unclIdx[newMember];
        ++count;

        {
            const double2 np = __ldg(reinterpret_cast<const double2*>(pts) + newMember);
            mp.x = np.x;
            mp.y = np.y;
        }

        bestV = DEV_INF;
        bestI = -1;
        for (int j = tid; j < n; j += GROW_THREADS) {
            double v = d[j];
            if (v < threshold) {   // excluded / too-far candidates stay excluded
                const int c = ix[j];
                if (c == newMember) {
                    v = DEV_INF;
                } else {
                    const double2 cp = __ldg(reinterpret_cast<const double2*>(pts) + c);
                    const double nd = distance(Point{cp.x, cp.y}, mp);
                    v = (nd > v) ? nd : v;
                    if (v < threshold && v < bestV) {
                        bestV = v;
                        bestI = c;
                    }
                }
                d[j] = v;
            }
        }
    }

    if (tid == 0) cards[slot] = count;
}

// ---------------------------------------------------------------------------
// Warp-per-seed kernel.
//
// After the seed is placed, only points closer than `threshold` to it can
// ever join the cluster, and once a candidate's running max distance reaches
// the threshold it is excluded for good. Each warp therefore compacts its live
// candidates (index + running max distance) into shared memory and shrinks the
// list every step, so a growth step costs O(live / 32) plus a warp shuffle
// reduction and no block-wide synchronisation at all.
// Seeds with more than WK_CAP live candidates are left to the block kernel.
// ---------------------------------------------------------------------------

constexpr int WK_WARPS = 4;
constexpr int WK_THREADS = WK_WARPS * 32;
constexpr int WK_CAP = 1024;

__device__ __forceinline__ void warpArgMin(double& v, int& i) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const double ov = __shfl_xor_sync(0xffffffffu, v, off);
        const int oi = __shfl_xor_sync(0xffffffffu, i, off);
        betterOf(v, i, ov, oi);
    }
}

__global__ void __launch_bounds__(WK_THREADS)
growClustersWarpKernel(const Point* __restrict__ pts,
                       const int M,
                       const int* __restrict__ seeds,
                       const int* __restrict__ unclIdx,
                       const int batch,
                       const double threshold,
                       int* __restrict__ members,
                       const int persistent,
                       const int rowStride,
                       int* __restrict__ cards) {
    __shared__ double s_d[WK_WARPS][WK_CAP];
    __shared__ int s_i[WK_WARPS][WK_CAP];

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int slot = blockIdx.x * WK_WARPS + warp;
    if (slot >= batch) return;
    const int seed = seeds[slot];
    const unsigned lanemask_lt = (1u << lane) - 1u;

    double* d = s_d[warp];
    int* ix = s_i[warp];
    const int row = persistent ? unclIdx[seed] : slot;
    int* __restrict__ myMembers = members + static_cast<size_t>(row) * rowStride;

    Point mp;
    {
        const double2 sp = __ldg(reinterpret_cast<const double2*>(pts) + seed);
        mp.x = sp.x;
        mp.y = sp.y;
    }

    // Scan all candidates, keep those within threshold of the seed (in index order).
    int n = 0;
    for (int base = 0; base < M; base += 32) {
        const int c = base + lane;
        bool alive = false;
        double dist = 0.0;
        if (c < M && c != seed) {
            const double2 cp = __ldg(reinterpret_cast<const double2*>(pts) + c);
            dist = distance(Point{cp.x, cp.y}, mp);
            alive = dist < threshold;
        }
        const unsigned mask = __ballot_sync(0xffffffffu, alive);
        if (alive) {
            const int pos = n + __popc(mask & lanemask_lt);
            if (pos < WK_CAP) {
                d[pos] = dist;
                ix[pos] = c;
            }
        }
        n += __popc(mask);
    }
    if (n > WK_CAP) {
        if (lane == 0) cards[slot] = -1;   // too many neighbours: block kernel takes over
        return;
    }

    if (lane == 0) myMembers[0] = unclIdx[seed];
    int count = 1;
    int chosen = -1;      // position of the member added in the previous step
    bool hasNew = false;

    while (n > 0) {
        double bestV = DEV_INF;
        int bestJ = -1;
        int nn = 0;
        __syncwarp();
        for (int base = 0; base < n; base += 32) {
            const int j = base + lane;
            bool alive = (j < n) && (j != chosen);
            double v = 0.0;
            int c = 0;
            if (alive) {
                v = d[j];
                c = ix[j];
                if (hasNew) {
                    const double2 cp = __ldg(reinterpret_cast<const double2*>(pts) + c);
                    const double nd = distance(Point{cp.x, cp.y}, mp);
                    v = (nd > v) ? nd : v;
                    alive = v < threshold;
                }
            }
            __syncwarp();
            const unsigned mask = __ballot_sync(0xffffffffu, alive);
            if (alive) {
                const int pos = nn + __popc(mask & lanemask_lt);
                d[pos] = v;
                ix[pos] = c;
                if (v < bestV) {   // positions ascend with index: first minimum wins
                    bestV = v;
                    bestJ = pos;
                }
            }
            nn += __popc(mask);
        }
        n = nn;
        warpArgMin(bestV, bestJ);
        if (bestJ < 0) break;

        __syncwarp();
        chosen = bestJ;
        const int newMember = ix[chosen];
        if (lane == 0) myMembers[count] = unclIdx[newMember];
        ++count;
        {
            const double2 np = __ldg(reinterpret_cast<const double2*>(pts) + newMember);
            mp.x = np.x;
            mp.y = np.y;
        }
        hasNew = true;
    }

    if (lane == 0) cards[slot] = count;
}

// Pick the largest cluster of the batch (ties -> earliest seed) and, if it is
// strictly larger than the running best, make it the running best.
// best[0] = cardinality, best[1] = seed (local index), bestMembers = members.
constexpr int SELECT_THREADS = 1024;

__global__ void __launch_bounds__(SELECT_THREADS)
selectBestKernel(const int* __restrict__ cards,
                 const int* __restrict__ members,
                 const int batch,
                 const int M,
                 const int* __restrict__ seeds,
                 int* __restrict__ best,
                 int* __restrict__ bestMembers) {
    __shared__ int s_card[SELECT_THREADS / 32];
    __shared__ int s_slot[SELECT_THREADS / 32];
    __shared__ int s_winner;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    int bc = -1, bs = 0x7fffffff;
    for (int s = tid; s < batch; s += SELECT_THREADS) {
        const int c = cards[s];
        if (c > bc || (c == bc && s < bs)) {
            bc = c;
            bs = s;
        }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const int oc = __shfl_down_sync(0xffffffffu, bc, off);
        const int os = __shfl_down_sync(0xffffffffu, bs, off);
        if (oc > bc || (oc == bc && os < bs)) {
            bc = oc;
            bs = os;
        }
    }
    if (lane == 0) {
        s_card[warp] = bc;
        s_slot[warp] = bs;
    }
    __syncthreads();
    if (warp == 0) {
        bc = (lane < SELECT_THREADS / 32) ? s_card[lane] : -1;
        bs = (lane < SELECT_THREADS / 32) ? s_slot[lane] : 0x7fffffff;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const int oc = __shfl_down_sync(0xffffffffu, bc, off);
            const int os = __shfl_down_sync(0xffffffffu, bs, off);
            if (oc > bc || (oc == bc && os < bs)) {
                bc = oc;
                bs = os;
            }
        }
        if (lane == 0) {
            // Strictly greater, like the sequential code: earlier seeds win ties.
            if (bc > best[0]) {
                best[0] = bc;
                best[1] = seeds[bs];
                s_winner = bs;
            } else {
                s_winner = -1;
            }
        }
    }
    __syncthreads();
    const int w = s_winner;
    if (w >= 0) {
        const int n = best[0];
        const int* src = members + static_cast<size_t>(w) * M;
        for (int k = tid; k < n; k += SELECT_THREADS) {
            bestMembers[k] = src[k];
        }
    }
}


// Mark unclustered points whose neighbourhood (points within the threshold)
// lost a member: only those seeds need their candidate cluster recomputed.
// The check is deliberately slightly conservative.
constexpr int DIRTY_THREADS = 256;

__global__ void __launch_bounds__(DIRTY_THREADS)
markDirtyKernel(const Point* __restrict__ allPts,
                const int* __restrict__ unclIdx,
                const int M,
                const int* __restrict__ removed,
                const int k,
                const double threshold,
                unsigned char* __restrict__ dirty) {
    __shared__ double2 s_rem[DIRTY_THREADS];
    const int i = blockIdx.x * DIRTY_THREADS + threadIdx.x;
    Point p{0.0, 0.0};
    if (i < M) {
        const double2 pp = __ldg(reinterpret_cast<const double2*>(allPts) + unclIdx[i]);
        p.x = pp.x;
        p.y = pp.y;
    }
    bool isDirty = false;
    const double limit = threshold * (1.0 + 1e-9);
    for (int base = 0; base < k; base += DIRTY_THREADS) {
        const int j = base + threadIdx.x;
        if (j < k) {
            s_rem[threadIdx.x] = __ldg(reinterpret_cast<const double2*>(allPts) + removed[j]);
        }
        __syncthreads();
        const int cnt = min(DIRTY_THREADS, k - base);
        if (i < M && !isDirty) {
            for (int t = 0; t < cnt; ++t) {
                const double2 r = s_rem[t];
                if (distance(p, Point{r.x, r.y}) < limit) {
                    isDirty = true;
                    break;
                }
            }
        }
        __syncthreads();
    }
    if (i < M) dirty[unclIdx[i]] = isDirty ? 1 : 0;
}

// ---------------------------------------------------------------------------
// Host driver
// ---------------------------------------------------------------------------

// Device buffers reused across outer iterations.
struct DeviceState {
    Point* allPts = nullptr;      // all N points (original indices)
    Point* pts = nullptr;         // compacted unclustered points
    int* unclIdx = nullptr;       // M original indices of the compacted points
    int* seeds = nullptr;         // seeds (local indices) to (re)compute
    int* removed = nullptr;       // original indices of the last cluster
    unsigned char* dirty = nullptr;
    int* members = nullptr;       // member lists (original indices)
    int* cards = nullptr;         // batch cardinalities
    double* wsD = nullptr;        // batch * M running max distances (block kernel)
    int* wsI = nullptr;           // batch * M candidate indices (block kernel)
    int* best = nullptr;          // {cardinality, seed}
    int* bestMembers = nullptr;   // N
    size_t membersCap = 0;
    size_t wsCap = 0;
};

// Pinned host staging buffers (fast asynchronous transfers).
struct HostState {
    Point* pts = nullptr;
    int* unclIdx = nullptr;
    int* seeds = nullptr;
    int* cards = nullptr;
    int* removed = nullptr;
    int* members = nullptr;
    int* best = nullptr;
    unsigned char* dirty = nullptr;
};

static void launchGrow(const Point* pts, const int M, const int* seeds, const int* unclIdx,
                       const double threshold, const int batch, int* members,
                       const int persistent, const int rowStride, int* cards,
                       double* wsD, int* wsI) {
    growClustersWarpKernel<<<(batch + WK_WARPS - 1) / WK_WARPS, WK_THREADS>>>(
        pts, M, seeds, unclIdx, batch, threshold, members, persistent, rowStride, cards);
    CUDA_CHECK(cudaGetLastError());
    growClustersBlockKernel<<<batch, GROW_THREADS>>>(pts, M, seeds, unclIdx, threshold, members,
                                                     persistent, rowStride, cards, wsD, wsI);
    CUDA_CHECK(cudaGetLastError());
}

// Main QT clustering algorithm (CUDA)
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<bool> clustered(N, false);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;
    
    // Initialize unclustered indices
    unclustered_indices.reserve(N);
    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }

    DeviceState dev;
    HostState host;
    const size_t Nsz = static_cast<size_t>(N);
    CUDA_CHECK(cudaMalloc(&dev.allPts, Nsz * sizeof(Point)));
    CUDA_CHECK(cudaMalloc(&dev.pts, Nsz * sizeof(Point)));
    CUDA_CHECK(cudaMalloc(&dev.unclIdx, Nsz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dev.seeds, Nsz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dev.removed, Nsz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dev.dirty, Nsz));
    CUDA_CHECK(cudaMalloc(&dev.cards, Nsz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dev.best, 2 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dev.bestMembers, Nsz * sizeof(int)));
    CUDA_CHECK(cudaHostAlloc(&host.pts, Nsz * sizeof(Point), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&host.unclIdx, Nsz * sizeof(int), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&host.seeds, Nsz * sizeof(int), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&host.cards, Nsz * sizeof(int), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&host.removed, Nsz * sizeof(int), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&host.members, Nsz * sizeof(int), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&host.best, 2 * sizeof(int), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&host.dirty, Nsz, cudaHostAllocDefault));
    CUDA_CHECK(cudaMemcpy(dev.allPts, points.data(), Nsz * sizeof(Point), cudaMemcpyHostToDevice));

    // Device memory budget for member lists / workspaces.
    size_t freeMem = 0, totalMem = 0;
    CUDA_CHECK(cudaMemGetInfo(&freeMem, &totalMem));
    const size_t budget = std::max<size_t>(freeMem / 2, static_cast<size_t>(64) << 20);

    // If an N x N member table fits, every seed's member list is kept on the
    // device (row = original index) and stays valid together with its cached
    // cardinality. Otherwise member lists live in a per-batch buffer and the
    // winning cluster is regrown when it comes from the cache.
    const bool persistent = (Nsz * Nsz * sizeof(int)) <= budget / 2;
    if (persistent) {
        dev.membersCap = Nsz * Nsz;
        CUDA_CHECK(cudaMalloc(&dev.members, dev.membersCap * sizeof(int)));
    }

    // Cached cardinality of the candidate cluster of every seed (original
    // index). A seed's cluster only depends on the unclustered points within
    // the threshold of the seed, so it stays valid until one of those is
    // removed; `dirty` tracks which seeds need recomputation.
    std::vector<int> cardCache(N, 0);
    std::vector<unsigned char> dirty(N, 1);
    std::vector<int> dirtySeeds;      // local indices
    dirtySeeds.reserve(N);

    const cudaStream_t stream = 0;

    // Main clustering loop
    while (!unclustered_indices.empty()) {
        const int M = static_cast<int>(unclustered_indices.size());
        const size_t Msz = static_cast<size_t>(M);

        // Upload the compacted point set (local index i <-> unclustered_indices[i]).
        dirtySeeds.clear();
        for (int i = 0; i < M; ++i) {
            const int orig = unclustered_indices[i];
            host.pts[i] = points[orig];
            host.unclIdx[i] = orig;
            if (dirty[orig]) dirtySeeds.push_back(i);
        }
        CUDA_CHECK(cudaMemcpyAsync(dev.pts, host.pts, Msz * sizeof(Point), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(dev.unclIdx, host.unclIdx, Msz * sizeof(int), cudaMemcpyHostToDevice, stream));

        const int D = static_cast<int>(dirtySeeds.size());
        const size_t wsPerSeed = Msz * (sizeof(int) + sizeof(double));
        const size_t memPerSeed = persistent ? 0 : Msz * sizeof(int);
        const int batch = static_cast<int>(std::min<size_t>(std::max(M, 1),
                                                            std::max<size_t>(1, budget / (wsPerSeed + memPerSeed))));
        const int rowStride = persistent ? N : M;

        if (!persistent) {
            const size_t need = static_cast<size_t>(batch) * Msz;
            if (need > dev.membersCap) {
                if (dev.members) CUDA_CHECK(cudaFree(dev.members));
                CUDA_CHECK(cudaMalloc(&dev.members, need * sizeof(int)));
                dev.membersCap = need;
            }
        }
        {
            const size_t need = static_cast<size_t>(batch) * Msz;
            if (need > dev.wsCap) {
                if (dev.wsD) CUDA_CHECK(cudaFree(dev.wsD));
                if (dev.wsI) CUDA_CHECK(cudaFree(dev.wsI));
                CUDA_CHECK(cudaMalloc(&dev.wsD, need * sizeof(double)));
                CUDA_CHECK(cudaMalloc(&dev.wsI, need * sizeof(int)));
                dev.wsCap = need;
            }
        }

        // Recompute the candidate clusters of all dirty seeds, in batches.
        host.best[0] = -1;
        host.best[1] = -1;
        CUDA_CHECK(cudaMemcpyAsync(dev.best, host.best, 2 * sizeof(int), cudaMemcpyHostToDevice, stream));
        if (D > 0) {
            std::copy(dirtySeeds.begin(), dirtySeeds.end(), host.seeds);
            CUDA_CHECK(cudaMemcpyAsync(dev.seeds, host.seeds, static_cast<size_t>(D) * sizeof(int),
                                       cudaMemcpyHostToDevice, stream));
            for (int off = 0; off < D; off += batch) {
                const int b = std::min(batch, D - off);
                launchGrow(dev.pts, M, dev.seeds + off, dev.unclIdx, threshold, b, dev.members,
                           persistent ? 1 : 0, rowStride, dev.cards + off, dev.wsD, dev.wsI);
                if (!persistent) {
                    selectBestKernel<<<1, SELECT_THREADS, 0, stream>>>(
                        dev.cards + off, dev.members, b, M, dev.seeds + off, dev.best, dev.bestMembers);
                    CUDA_CHECK(cudaGetLastError());
                }
            }
            CUDA_CHECK(cudaMemcpyAsync(host.cards, dev.cards, static_cast<size_t>(D) * sizeof(int),
                                       cudaMemcpyDeviceToHost, stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for (int j = 0; j < D; ++j) {
            cardCache[unclustered_indices[dirtySeeds[j]]] = host.cards[j];
        }

        // Pick the seed with the largest cluster (ties -> earliest unclustered index).
        int max_cardinality = -1;
        int best_seed_local = -1;
        for (int i = 0; i < M; ++i) {
            const int c = cardCache[unclustered_indices[i]];
            if (c > max_cardinality) {
                max_cardinality = c;
                best_seed_local = i;
            }
        }

        if (best_seed_local < 0 || max_cardinality <= 0) {
            break;  // No more clusters can be formed
        }

        if (max_cardinality == 1) {
            // No remaining point has any neighbour within the threshold, so every
            // remaining point becomes a singleton cluster, in index order.
            for (int i = 0; i < M; ++i) {
                Cluster cluster;
                cluster.seed_point = unclustered_indices[i];
                cluster.members.push_back(unclustered_indices[i]);
                clusters.push_back(cluster);
                clustered[unclustered_indices[i]] = true;
            }
            unclustered_indices.clear();
            break;
        }

        // Fetch the member list (original indices) of the winning cluster.
        const int best_seed = unclustered_indices[best_seed_local];
        if (persistent) {
            CUDA_CHECK(cudaMemcpyAsync(host.members, dev.members + static_cast<size_t>(best_seed) * N,
                                       static_cast<size_t>(max_cardinality) * sizeof(int),
                                       cudaMemcpyDeviceToHost, stream));
        } else {
            int bestInfo[2];
            CUDA_CHECK(cudaMemcpy(bestInfo, dev.best, sizeof(bestInfo), cudaMemcpyDeviceToHost));
            if (bestInfo[1] != best_seed_local) {
                // Winner is a cached (non-dirty) seed: regrow this single cluster.
                host.seeds[0] = best_seed_local;
                host.best[0] = -1;
                host.best[1] = -1;
                CUDA_CHECK(cudaMemcpyAsync(dev.seeds, host.seeds, sizeof(int), cudaMemcpyHostToDevice, stream));
                CUDA_CHECK(cudaMemcpyAsync(dev.best, host.best, 2 * sizeof(int), cudaMemcpyHostToDevice, stream));
                launchGrow(dev.pts, M, dev.seeds, dev.unclIdx, threshold, 1, dev.members, 0, rowStride,
                           dev.cards, dev.wsD, dev.wsI);
                selectBestKernel<<<1, SELECT_THREADS, 0, stream>>>(
                    dev.cards, dev.members, 1, M, dev.seeds, dev.best, dev.bestMembers);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaMemcpy(bestInfo, dev.best, sizeof(bestInfo), cudaMemcpyDeviceToHost));
                if (bestInfo[0] != max_cardinality || bestInfo[1] != best_seed_local) {
                    fprintf(stderr, "Internal error: cached cardinality mismatch (%d vs %d)\n",
                            bestInfo[0], max_cardinality);
                    exit(EXIT_FAILURE);
                }
            }
            CUDA_CHECK(cudaMemcpyAsync(host.members, dev.bestMembers,
                                       static_cast<size_t>(max_cardinality) * sizeof(int),
                                       cudaMemcpyDeviceToHost, stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));

        Cluster cluster;
        cluster.seed_point = best_seed;
        cluster.members.assign(host.members, host.members + max_cardinality);
        clusters.push_back(cluster);
        
        // Mark all members as clustered
        for (size_t i = 0; i < cluster.members.size(); ++i) {
            clustered[cluster.members[i]] = true;
        }

        // Flag the seeds whose neighbourhood lost a point.
        std::copy(cluster.members.begin(), cluster.members.end(), host.removed);
        CUDA_CHECK(cudaMemcpyAsync(dev.removed, host.removed,
                                   static_cast<size_t>(max_cardinality) * sizeof(int),
                                   cudaMemcpyHostToDevice, stream));
        markDirtyKernel<<<(M + DIRTY_THREADS - 1) / DIRTY_THREADS, DIRTY_THREADS, 0, stream>>>(
            dev.allPts, dev.unclIdx, M, dev.removed, max_cardinality, threshold, dev.dirty);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(host.dirty, dev.dirty, Nsz, cudaMemcpyDeviceToHost, stream));
        
        // Remove clustered points from unclustered list
        unclustered_indices.erase(
            std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                          [&clustered](int idx) { return clustered[idx]; }),
            unclustered_indices.end()
        );
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::copy(host.dirty, host.dirty + N, dirty.begin());
    }

    CUDA_CHECK(cudaFree(dev.allPts));
    CUDA_CHECK(cudaFree(dev.pts));
    CUDA_CHECK(cudaFree(dev.unclIdx));
    CUDA_CHECK(cudaFree(dev.seeds));
    CUDA_CHECK(cudaFree(dev.removed));
    CUDA_CHECK(cudaFree(dev.dirty));
    CUDA_CHECK(cudaFree(dev.cards));
    CUDA_CHECK(cudaFree(dev.best));
    CUDA_CHECK(cudaFree(dev.bestMembers));
    if (dev.members) CUDA_CHECK(cudaFree(dev.members));
    if (dev.wsD) CUDA_CHECK(cudaFree(dev.wsD));
    if (dev.wsI) CUDA_CHECK(cudaFree(dev.wsI));
    CUDA_CHECK(cudaFreeHost(host.pts));
    CUDA_CHECK(cudaFreeHost(host.unclIdx));
    CUDA_CHECK(cudaFreeHost(host.seeds));
    CUDA_CHECK(cudaFreeHost(host.cards));
    CUDA_CHECK(cudaFreeHost(host.removed));
    CUDA_CHECK(cudaFreeHost(host.members));
    CUDA_CHECK(cudaFreeHost(host.best));
    CUDA_CHECK(cudaFreeHost(host.dirty));
    
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

    // Initialise the CUDA context up front so it is not part of the timing.
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaFree(0));
    {
        // Force module loading of all kernel instantiations now (lazy loading
        // would otherwise happen inside the timed region on first launch).
        cudaFuncAttributes attr;
        CUDA_CHECK(cudaFuncGetAttributes(&attr, growClustersBlockKernel));
        CUDA_CHECK(cudaFuncGetAttributes(&attr, growClustersWarpKernel));
        CUDA_CHECK(cudaFuncGetAttributes(&attr, selectBestKernel));
        CUDA_CHECK(cudaFuncGetAttributes(&attr, markDirtyKernel));
    }
    
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
