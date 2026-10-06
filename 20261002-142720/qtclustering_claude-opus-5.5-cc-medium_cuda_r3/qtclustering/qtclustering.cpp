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
#include <math_constants.h>

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
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                         \
                    cudaGetErrorString(err_), __FILE__, __LINE__);              \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

static const unsigned FULL_MASK = 0xffffffffu;
static const int WARP = 32;
static const int CARD_BLOCK = 256;
static const int MEMBERS_BLOCK = 128;

// Squared distance evaluated exactly as the reference host build does (the
// distance is contracted to sqrt(fma(dx, dx, dy * dy))), so that ties and
// threshold tests are bit-identical. Since sqrt is monotone, max/threshold
// operations are performed on squared values and sqrt is only taken when needed.
__device__ __forceinline__ double devDist2(const double2 a, const double2 b) {
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    return __fma_rn(dx, dx, __dmul_rn(dy, dy));
}

struct LaneBest {
    double lim;   // conservative upper bound on squared values whose sqrt can be <= s
    double s;     // sqrt(best squared distance) == reference max_dist
    int idx;
};

__device__ __forceinline__ void laneBestInit(LaneBest& b) {
    b.lim = CUDART_INF; b.s = CUDART_INF; b.idx = INT_MAX;
}

// Lexicographic (max_dist, index) minimum, with sqrt only evaluated for
// squared values that can possibly tie with or beat the current best.
__device__ __forceinline__ void laneBestUpdate(LaneBest& b, const double q, const int id) {
    if (q <= b.lim) {
        const double s = sqrt(q);
        if (s < b.s || (s == b.s && id < b.idx)) {
            b.s = s;
            b.idx = id;
            const double nx = nextafter(s, CUDART_INF);
            b.lim = nx * nx;
        }
    }
}

__device__ __forceinline__ void warpArgmin(double& v, int& idx) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const double ov = __shfl_xor_sync(FULL_MASK, v, off);
        const int oi = __shfl_xor_sync(FULL_MASK, idx, off);
        if (ov < v || (ov == v && oi < idx)) { v = ov; idx = oi; }
    }
}

struct DevData {
    const double2* p;         // point coordinates by index
    const double2* sp;        // point coordinates sorted by grid cell
    const int* sidx;          // original index of sorted point
    const int* cellStart;     // [W*H + 1]
    const int* cellX;         // grid cell of each point
    const int* cellY;
    unsigned char* clustered;
    int W, H;
    int N;
    double qcut;              // sqrt(q) < threshold  <=>  q < qcut
};

struct CandBuf {
    double* q;                // squared max distance to current cluster members
    int* idx;
};

// Shared state for block-cooperative cluster growth
struct GrowShared {
    int count[2];
    int task;
    double s[32];
    int idx[32];
};

// Append kept candidates of the current warp-chunk; returns the output slot.
// In warp mode `n` is the running count; in block mode slots are reserved
// (in arbitrary order, which is fine as selection is order independent).
template <bool BLOCK>
__device__ __forceinline__ int appendSlot(const bool keep, int& n, int* counter) {
    const int lane = threadIdx.x & (WARP - 1);
    const unsigned mask = __ballot_sync(FULL_MASK, keep);
    const int rank = __popc(mask & ((1u << lane) - 1u));
    if (BLOCK) {
        int wbase = 0;
        if (lane == 0 && mask) wbase = atomicAdd(counter, __popc(mask));
        wbase = __shfl_sync(FULL_MASK, wbase, 0);
        return wbase + rank;
    } else {
        const int pos = n + rank;
        n += __popc(mask);
        return pos;
    }
}

// Group-wide argmin of (max_dist, index). In block mode the partials array of the
// previous step may only be overwritten after the barrier at the end of a pass.
template <bool BLOCK>
__device__ __forceinline__ int groupArgmin(const LaneBest& best, GrowShared* sh) {
    double v = best.s;
    int r = best.idx;
    warpArgmin(v, r);
    if (BLOCK) {
        const int lane = threadIdx.x & (WARP - 1);
        const int warp = threadIdx.x / WARP;
        const int nw = blockDim.x / WARP;
        if (lane == 0) { sh->s[warp] = v; sh->idx[warp] = r; }
        __syncthreads();
        v = lane < nw ? sh->s[lane] : CUDART_INF;
        r = lane < nw ? sh->idx[lane] : INT_MAX;
        warpArgmin(v, r);
    }
    return r;
}

// Grow the QT candidate cluster for `seed` using one warp (BLOCK=false) or the
// whole thread block (BLOCK=true).
// Every candidate keeps its running max distance to the cluster members; this is
// exactly the reference's max over members (max is order independent). Candidates
// whose max distance reaches the threshold can never be added again and are dropped.
// The next member is the candidate with smallest (max_dist, index), matching the
// reference's ascending scan with strict '<'.
template <bool BLOCK>
__device__ int growCluster(const DevData& d, const int seed, CandBuf a, CandBuf b,
                           int* members, GrowShared* sh) {
    const int tid = BLOCK ? threadIdx.x : (threadIdx.x & (WARP - 1));
    const int stride = BLOCK ? blockDim.x : WARP;
    const double qcut = d.qcut;
    const double2 sp = d.p[seed];
    const int cx = d.cellX[seed];
    const int cy = d.cellY[seed];
    const int c0 = max(cx - 1, 0);
    const int c1 = min(cx + 1, d.W - 1);
    int parity = 0;

    int n = 0;
    LaneBest best;
    laneBestInit(best);

    // Gather unclustered points within threshold of the seed
    for (int r = max(cy - 1, 0); r <= min(cy + 1, d.H - 1); ++r) {
        const int s = d.cellStart[r * d.W + c0];
        const int e = d.cellStart[r * d.W + c1 + 1];
        for (int base = s; base < e; base += stride) {
            const int j = base + tid;
            bool keep = false;
            double q = 0.0;
            int id = -1;
            if (j < e) {
                id = d.sidx[j];
                if (id != seed && !d.clustered[id]) {
                    q = devDist2(d.sp[j], sp);
                    keep = q < qcut;
                }
            }
            const int pos = appendSlot<BLOCK>(keep, n, &sh->count[parity]);
            if (keep) {
                a.q[pos] = q; a.idx[pos] = id;
                laneBestUpdate(best, q, id);
            }
        }
    }
    if (BLOCK) {
        __syncthreads();
        n = sh->count[parity];
        if (threadIdx.x == 0) sh->count[parity ^ 1] = 0;
    } else {
        __syncwarp();
    }
    int added = groupArgmin<BLOCK>(best, sh);
    parity ^= 1;

    int S = 1;
    if (members && tid == 0) members[0] = seed;

    while (added != INT_MAX) {
        if (members && tid == 0) members[S] = added;
        ++S;
        const double2 ap = d.p[added];

        int newn = 0;
        laneBestInit(best);
        for (int base = 0; base < n; base += stride) {
            const int i = base + tid;
            bool keep = false;
            double q = 0.0;
            int id = -1;
            if (i < n) {
                id = a.idx[i];
                if (id != added) {
                    q = fmax(a.q[i], devDist2(d.p[id], ap));
                    keep = q < qcut;
                }
            }
            const int pos = appendSlot<BLOCK>(keep, newn, &sh->count[parity]);
            if (keep) {
                b.q[pos] = q; b.idx[pos] = id;
                laneBestUpdate(best, q, id);
            }
        }
        if (BLOCK) {
            __syncthreads();
            newn = sh->count[parity];
            if (threadIdx.x == 0) sh->count[parity ^ 1] = 0;
        } else {
            __syncwarp();
        }
        added = groupArgmin<BLOCK>(best, sh);
        parity ^= 1;
        n = newn;
        const CandBuf t = a; a = b; b = t;
    }
    if (BLOCK) __syncthreads();
    return S;
}

__device__ __forceinline__ void slotBuffers(double* dbuf, int* ibuf, size_t slot,
                                            size_t kmax, CandBuf& a, CandBuf& b) {
    a.q = dbuf + slot * kmax * 2; a.idx = ibuf + slot * kmax * 2;
    b.q = a.q + kmax;             b.idx = a.idx + kmax;
}

// Compute cardinality of the candidate cluster for each seed in the dirty list;
// each warp dynamically grabs seeds for load balancing.
__global__ void __launch_bounds__(CARD_BLOCK, 4)
cardKernel(DevData d, const int* __restrict__ dirty, const int* dirtyCount,
           int* workCounter, int* card, double* dbuf, int* ibuf, size_t kmax) {
    const int lane = threadIdx.x & (WARP - 1);
    const int total = *dirtyCount;
    const size_t warpId = (static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x) / WARP;
    CandBuf a, b;
    slotBuffers(dbuf, ibuf, warpId, kmax, a, b);
    while (true) {
        int t = 0;
        if (lane == 0) t = atomicAdd(workCounter, 1);
        t = __shfl_sync(FULL_MASK, t, 0);
        if (t >= total) break;
        const int seed = dirty[t];
        const int S = growCluster<false>(d, seed, a, b, nullptr, nullptr);
        if (lane == 0) card[seed] = S;
    }
}

// Regenerate the member list of the selected seed with a single block (this is on
// the critical path of every iteration)
__global__ void __launch_bounds__(MEMBERS_BLOCK)
membersKernel(DevData d, const unsigned long long* best, int* members,
              double* dbuf, int* ibuf, size_t kmax) {
    __shared__ GrowShared sh;
    CandBuf a, b;
    slotBuffers(dbuf, ibuf, 0, kmax, a, b);
    if (threadIdx.x == 0) sh.count[0] = 0;
    __syncthreads();
    const int seed = static_cast<int>(0xffffffffu - static_cast<unsigned>(*best & 0xffffffffull));
    growCluster<true>(d, seed, a, b, members, &sh);
}

// Select the unclustered seed with maximal cardinality (lowest index on ties)
__global__ void selectKernel(const int* __restrict__ card, const unsigned char* __restrict__ clustered,
                             int N, unsigned long long* best) {
    unsigned long long key = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x) {
        if (!clustered[i]) {
            const unsigned long long k = (static_cast<unsigned long long>(card[i]) << 32) |
                                         (0xffffffffu - static_cast<unsigned>(i));
            key = k > key ? k : key;
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const unsigned long long o = __shfl_xor_sync(FULL_MASK, key, off);
        key = o > key ? o : key;
    }
    if ((threadIdx.x & (WARP - 1)) == 0 && key) atomicMax(best, key);
}

__global__ void markKernel(const int* __restrict__ members, int S, unsigned char* clustered) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < S) clustered[members[i]] = 1;
}

// A seed's candidate cluster only depends on unclustered points within threshold of
// it, so only seeds near a newly clustered point need to be recomputed.
// One warp per point; lanes check the new members in parallel.
__global__ void dirtyKernel(DevData d, const int* __restrict__ members, int S, int seed,
                            int* dirty, int* dirtyCount) {
    const int i = (blockIdx.x * blockDim.x + threadIdx.x) / WARP;
    const int lane = threadIdx.x & (WARP - 1);
    if (i >= d.N || d.clustered[i]) return;
    // Members are within threshold of the seed, so affected points lie within
    // 2 cells (cell size > threshold) of the seed's cell.
    if (abs(d.cellX[i] - d.cellX[seed]) > 2 || abs(d.cellY[i] - d.cellY[seed]) > 2) return;
    const double2 pi = d.p[i];
    bool hit = false;
    for (int k = lane; k < S && !hit; k += WARP)
        hit = devDist2(pi, d.p[members[k]]) < d.qcut;
    if (__any_sync(FULL_MASK, hit) && lane == 0) dirty[atomicAdd(dirtyCount, 1)] = i;
}

__global__ void iotaKernel(int* a, int N) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) a[i] = i;
}

// Main QT clustering algorithm
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<Cluster> clusters;
    if (N == 0) return clusters;

    // Uniform grid with cell size slightly larger than the threshold
    double minx = points[0].x, maxx = points[0].x, miny = points[0].y, maxy = points[0].y;
    for (const Point& p : points) {
        minx = std::min(minx, p.x); maxx = std::max(maxx, p.x);
        miny = std::min(miny, p.y); maxy = std::max(maxy, p.y);
    }
    const double extent = std::max(maxx - minx, maxy - miny);
    const double cell = std::max(threshold * 1.0001, extent / 1024.0 + 1e-12);
    const int W = std::max(1, std::min(1025, static_cast<int>((maxx - minx) / cell) + 1));
    const int H = std::max(1, std::min(1025, static_cast<int>((maxy - miny) / cell) + 1));
    std::vector<int> cellX(N), cellY(N), cellCount(static_cast<size_t>(W) * H + 1, 0);
    for (int i = 0; i < N; ++i) {
        cellX[i] = std::min(W - 1, std::max(0, static_cast<int>((points[i].x - minx) / cell)));
        cellY[i] = std::min(H - 1, std::max(0, static_cast<int>((points[i].y - miny) / cell)));
        cellCount[static_cast<size_t>(cellY[i]) * W + cellX[i]]++;
    }
    std::vector<int> cellStart(static_cast<size_t>(W) * H + 1, 0);
    for (size_t c = 0; c < static_cast<size_t>(W) * H; ++c) cellStart[c + 1] = cellStart[c] + cellCount[c];
    std::vector<int> fill(cellStart.begin(), cellStart.end() - 1);
    std::vector<int> sidx(N);
    std::vector<double2> hp(N), hsp(N);
    for (int i = 0; i < N; ++i) {
        hp[i] = make_double2(points[i].x, points[i].y);
        const int pos = fill[static_cast<size_t>(cellY[i]) * W + cellX[i]]++;
        sidx[pos] = i;
        hsp[pos] = hp[i];
    }

    // Smallest squared distance q with sqrt(q) >= threshold (sqrt is correctly
    // rounded on host and device), so that  sqrt(q) < threshold  <=>  q < qcut.
    double qcut = threshold * threshold;
    while (qcut < std::numeric_limits<double>::infinity() && std::sqrt(qcut) < threshold)
        qcut = std::nextafter(qcut, std::numeric_limits<double>::infinity());
    while (qcut > 0.0 && std::sqrt(std::nextafter(qcut, 0.0)) >= threshold)
        qcut = std::nextafter(qcut, 0.0);
    // Upper bound on candidate list length: points in a 3x3 cell neighbourhood
    size_t kmax = 1;
    for (int cy = 0; cy < H; ++cy) {
        for (int cx = 0; cx < W; ++cx) {
            size_t k = 0;
            const int c0 = std::max(cx - 1, 0), c1 = std::min(cx + 1, W - 1);
            for (int r = std::max(cy - 1, 0); r <= std::min(cy + 1, H - 1); ++r)
                k += cellStart[static_cast<size_t>(r) * W + c1 + 1] - cellStart[static_cast<size_t>(r) * W + c0];
            kmax = std::max(kmax, k);
        }
    }

    // Device allocations
    double2 *d_p, *d_sp;
    double* d_dbuf;
    int *d_sidx, *d_cellStart, *d_cellX, *d_cellY, *d_card, *d_dirty, *d_counters, *d_members, *d_ibuf;
    unsigned char* d_clustered;
    unsigned long long* d_best;
    const size_t nb = sizeof(double2) * N, ni = sizeof(int) * N;
    CUDA_CHECK(cudaMalloc(&d_p, nb));
    CUDA_CHECK(cudaMalloc(&d_sp, nb));
    CUDA_CHECK(cudaMalloc(&d_sidx, ni));
    CUDA_CHECK(cudaMalloc(&d_cellStart, sizeof(int) * cellStart.size()));
    CUDA_CHECK(cudaMalloc(&d_cellX, ni));
    CUDA_CHECK(cudaMalloc(&d_cellY, ni));
    CUDA_CHECK(cudaMalloc(&d_card, ni));
    CUDA_CHECK(cudaMalloc(&d_dirty, ni));
    CUDA_CHECK(cudaMalloc(&d_members, ni));
    CUDA_CHECK(cudaMalloc(&d_counters, sizeof(int) * 2));
    CUDA_CHECK(cudaMalloc(&d_clustered, N));
    CUDA_CHECK(cudaMalloc(&d_best, sizeof(unsigned long long)));

    int dev = 0, numSMs = 1, blocksPerSM = 1;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, dev));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, cardKernel, CARD_BLOCK, 0));
    size_t freeMem = 0, totalMem = 0;
    CUDA_CHECK(cudaMemGetInfo(&freeMem, &totalMem));
    const size_t perWarp = kmax * 2 * (sizeof(double) + sizeof(int));
    const size_t warpsPerBlock = CARD_BLOCK / WARP;
    const size_t budget = std::max<size_t>(perWarp * warpsPerBlock, freeMem / 2);
    const size_t maxBlocksMem = std::max<size_t>(1, budget / (perWarp * warpsPerBlock));
    const int needBlocks = static_cast<int>((static_cast<size_t>(N) + warpsPerBlock - 1) / warpsPerBlock);
    const int cardBlocks = static_cast<int>(std::min<size_t>(
        std::min<size_t>(static_cast<size_t>(numSMs) * std::max(blocksPerSM, 1), needBlocks), maxBlocksMem));
    const size_t totalWarps = static_cast<size_t>(cardBlocks) * warpsPerBlock;
    CUDA_CHECK(cudaMalloc(&d_dbuf, totalWarps * kmax * 2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_ibuf, totalWarps * kmax * 2 * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_p, hp.data(), nb, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sp, hsp.data(), nb, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sidx, sidx.data(), ni, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cellStart, cellStart.data(), sizeof(int) * cellStart.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cellX, cellX.data(), ni, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cellY, cellY.data(), ni, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_clustered, 0, N));

    DevData dd;
    dd.p = d_p; dd.sp = d_sp; dd.sidx = d_sidx;
    dd.cellStart = d_cellStart; dd.cellX = d_cellX; dd.cellY = d_cellY;
    dd.clustered = d_clustered; dd.W = W; dd.H = H; dd.N = N; dd.qcut = qcut;

    const int TB = 256;
    const int nBlocks = (N + TB - 1) / TB;
    const int selBlocks = std::min(nBlocks, numSMs * 4);

    // Initially every point is a (dirty) seed
    const int hInit[2] = {N, 0};
    iotaKernel<<<nBlocks, TB>>>(d_dirty, N);
    CUDA_CHECK(cudaMemcpy(d_counters, hInit, sizeof(hInit), cudaMemcpyHostToDevice));
    cardKernel<<<cardBlocks, CARD_BLOCK>>>(dd, d_dirty, d_counters, d_counters + 1, d_card,
                                           d_dbuf, d_ibuf, kmax);

    std::vector<int> seeds;
    std::vector<int> sizes;
    int offset = 0;
    unsigned long long hBest = 0;
    bool singletonTail = false;

    while (offset < N) {
        CUDA_CHECK(cudaMemsetAsync(d_best, 0, sizeof(unsigned long long)));
        selectKernel<<<selBlocks, TB>>>(d_card, d_clustered, N, d_best);
        CUDA_CHECK(cudaMemcpy(&hBest, d_best, sizeof(hBest), cudaMemcpyDeviceToHost));
        if (hBest == 0) break;
        const int S = static_cast<int>(hBest >> 32);
        const int seed = static_cast<int>(0xffffffffu - static_cast<unsigned>(hBest & 0xffffffffull));
        if (S == 1) {
            // Every remaining point forms a singleton; the reference takes them
            // in ascending index order.
            singletonTail = true;
            break;
        }
        int* mem = d_members + offset;
        membersKernel<<<1, MEMBERS_BLOCK>>>(dd, d_best, mem, d_dbuf, d_ibuf, kmax);
        markKernel<<<(S + TB - 1) / TB, TB>>>(mem, S, d_clustered);
        seeds.push_back(seed);
        sizes.push_back(S);
        offset += S;
        if (offset >= N) break;
        CUDA_CHECK(cudaMemsetAsync(d_counters, 0, sizeof(int) * 2));
        dirtyKernel<<<(N + TB / WARP - 1) / (TB / WARP), TB>>>(dd, mem, S, seed, d_dirty, d_counters);
        cardKernel<<<cardBlocks, CARD_BLOCK>>>(dd, d_dirty, d_counters, d_counters + 1, d_card,
                                               d_dbuf, d_ibuf, kmax);
    }
    CUDA_CHECK(cudaGetLastError());

    std::vector<int> hMembers(offset);
    if (offset > 0)
        CUDA_CHECK(cudaMemcpy(hMembers.data(), d_members, sizeof(int) * offset, cudaMemcpyDeviceToHost));
    std::vector<unsigned char> hClustered(N);
    if (singletonTail)
        CUDA_CHECK(cudaMemcpy(hClustered.data(), d_clustered, N, cudaMemcpyDeviceToHost));

    clusters.reserve(seeds.size() + (singletonTail ? N - offset : 0));
    int pos = 0;
    for (size_t c = 0; c < seeds.size(); ++c) {
        Cluster cl;
        cl.seed_point = seeds[c];
        cl.members.assign(hMembers.begin() + pos, hMembers.begin() + pos + sizes[c]);
        pos += sizes[c];
        clusters.push_back(std::move(cl));
    }
    if (singletonTail) {
        for (int i = 0; i < N; ++i) {
            if (!hClustered[i]) {
                Cluster cl;
                cl.seed_point = i;
                cl.members.push_back(i);
                clusters.push_back(std::move(cl));
            }
        }
    }

    cudaFree(d_p); cudaFree(d_sp); cudaFree(d_sidx);
    cudaFree(d_cellStart); cudaFree(d_cellX); cudaFree(d_cellY); cudaFree(d_card);
    cudaFree(d_dirty); cudaFree(d_members); cudaFree(d_counters); cudaFree(d_clustered);
    cudaFree(d_best); cudaFree(d_dbuf); cudaFree(d_ibuf);

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
    
    // Initialize the CUDA context outside of the timed region
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
