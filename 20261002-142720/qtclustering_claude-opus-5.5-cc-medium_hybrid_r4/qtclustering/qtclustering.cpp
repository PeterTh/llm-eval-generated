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
#include <thread>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

static const double MAX_WIDTH = 20.0;
static const double MAX_HEIGHT = 20.0;
// Minimum loop length for which host-side OpenMP parallelization pays off
#define OMP_MIN_WORK (1 << 17)

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
            const double dx = std::fma(2.0, frand(), -1.0) * r;
            const double dy = std::sqrt(std::fma(r, r, -(dx * dx))) * sign;
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

// Calculate Euclidean distance between two points.
// The FMA is explicit so that host and device produce bit-identical results
// (and match the contracted form emitted for the reference build).
inline double distance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(std::fma(dx, dx, dy * dy));
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

// Seeds are processed in two tiers: small neighbourhoods by small blocks
// (many per SM), large ones by big blocks. The winning cluster is regenerated
// by a single large block.
static const int SMALL_BS = 128;
static const int SMALL_K = 1024;     // max candidates handled by the small tier
static const int BIG_BS = 512;
static const int WIN_BS = 1024;
// Per-candidate "hot" storage: normalized relative x, y and squared max distance (floats)
static const int HOT_BYTES = 3 * sizeof(float);
static const long long BIG_SMEM_LIMIT = 48 * 1024;

// Float filter error budget (normalized squared-distance units). Coordinates
// are seed-relative and scaled by 1/threshold, so |coord| < 1 and squared
// distances are <= 4. Rounding the coordinates (2^-24 relative), their
// difference and the squared sum bounds the error of a candidate's float
// squared max distance by 2^-22 * 2(|dx|+|dy|) + 2^-24 * 8 < 1.9e-6;
// decisions use a > 2.5x larger margin.
static const float FERR = 5e-6f;

// Device-side view of the spatially sorted point set.
// Points are bucketed into a uniform grid with cell size >= threshold, so all
// points within `threshold` of a point lie in its 3x3 cell neighbourhood.
struct DeviceGrid {
    const double* sx;        // x coordinate by sorted position
    const double* sy;        // y coordinate by sorted position
    const int* sidx;         // original index by sorted position
    const int* cellOf;       // cell id by sorted position
    const int* pos;          // sorted position by original index
    const int* cellStart;    // cell -> first sorted position (size cells+1)
    unsigned char* clustered;// clustered flag by sorted position
    int gx, gy;
    double threshold;
    double thrSq;   // smallest double T with sqrt(T) >= threshold: d < thr <=> d2 < T
    double invThr;
};

__device__ __forceinline__ double devDist2(double ax, double ay, double bx, double by) {
    const double dx = __dsub_rn(ax, bx);
    const double dy = __dsub_rn(ay, by);
    return __fma_rn(dx, dx, __dmul_rn(dy, dy));
}

__device__ __forceinline__ void rowRanges(const DeviceGrid& g, int cell, int* rb, int* re, int& nrows) {
    const int cx = cell % g.gx;
    const int cy = cell / g.gx;
    const int x0 = max(cx - 1, 0);
    const int x1 = min(cx + 1, g.gx - 1);
    nrows = 0;
    for (int r = max(cy - 1, 0); r <= min(cy + 1, g.gy - 1); ++r) {
        rb[nrows] = g.cellStart[r * g.gx + x0];
        re[nrows] = g.cellStart[r * g.gx + x1 + 1];
        ++nrows;
    }
}

// Per-block global scratch. Candidates are gathered here; their hot data
// (xy, m2f) is then moved to shared memory whenever the neighbourhood fits.
struct Scratch {
    float2* xy;    // normalized seed-relative coordinates
    float* m2f;    // float squared max distance (+inf: dead / taken)
    double* ex;    // cached exact squared max distance of each candidate ...
    int* exu;      // ... valid over the first exu[k] members
    int* idx;      // original point index of each candidate
    int* pos;      // sorted position of each candidate
    int* memb;     // sorted positions of the current members
    int* cont;     // slow-path contender slots
    long long cap;
};

struct MinPair {
    float v1;   // smallest value
    int s1;     // its slot
    float v2;   // second smallest value
};

// Branch-free update of (smallest, slot, second smallest) with value v.
// Relies on v2 >= v1; +inf values leave the state unchanged.
__device__ __forceinline__ void mergeMin(MinPair& a, float v, int slot) {
    a.v2 = fminf(a.v2, fmaxf(a.v1, v));
    a.s1 = (v < a.v1) ? slot : a.s1;
    a.v1 = fminf(a.v1, v);
}

__device__ __forceinline__ unsigned warpMinU(unsigned v) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    return __reduce_min_sync(0xffffffffu, v);
#else
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = min(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
#endif
}

__device__ __forceinline__ int warpSum(int v) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    return __reduce_add_sync(0xffffffffu, v);
#else
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
#endif
}

// Warp-wide (smallest, its slot, second smallest), result in all lanes;
// non-negative floats order like their unsigned bit patterns.
__device__ __forceinline__ void warpMinPair(MinPair& a) {
    const unsigned b1 = __float_as_uint(a.v1);
    const unsigned m1 = warpMinU(b1);
    const unsigned slot = warpMinU((b1 == m1) ? static_cast<unsigned>(a.s1) : 0xffffffffu);
    const bool owner = (b1 == m1) && (static_cast<unsigned>(a.s1) == slot);
    const unsigned m2 = warpMinU(owner ? __float_as_uint(a.v2) : b1);
    a.v1 = __uint_as_float(m1);
    a.s1 = static_cast<int>(slot);
    a.v2 = __uint_as_float(m2);
}

struct ExactBest {
    double m;  // exact max distance (sqrt of exact squared distance)
    int idx;
    int slot;
};

__device__ __forceinline__ void warpExactBest(ExactBest& b) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const double om = __shfl_down_sync(0xffffffffu, b.m, o);
        const int oi = __shfl_down_sync(0xffffffffu, b.idx, o);
        const int os = __shfl_down_sync(0xffffffffu, b.slot, o);
        if (os >= 0 && (b.slot < 0 || om < b.m || (om == b.m && oi < b.idx))) {
            b.m = om;
            b.idx = oi;
            b.slot = os;
        }
    }
}

// Each block (persistently) takes seeds from the work list and grows the
// candidate QT cluster for that seed exactly as the sequential algorithm does:
// repeatedly add the unclustered point with the smallest maximum distance to
// the current members (ties -> smallest index) while that distance < threshold.
// Only points within threshold of the seed can ever qualify. Each candidate's
// maximum distance is tracked incrementally in float with a rigorous error
// bound; whenever the bound cannot decide the next member (near-ties or the
// threshold boundary), the contenders' exact double-precision maxima are
// recomputed over all members and compared with the original semantics.
template <int BS>
__global__ void __launch_bounds__(BS)
candidateClusterKernel(DeviceGrid g, const int* __restrict__ seeds, const int* nseedsPtr,
                       int singleSeed, int* workCounter, int* card, int* membersOut,
                       Scratch sc, int smemCap, int* overflowList, int* overflowCount) {
    constexpr int NW = BS / 32;
    extern __shared__ float4 smem4[];
    __shared__ float wV1[2][NW], wV2[2][NW];
    __shared__ int wS1[2][NW], wK[2][NW];
    __shared__ double wM[NW];
    __shared__ int wI[NW], wES[NW], wCnt[NW];
    __shared__ int sSeedIdx, sCnt, sNumCont;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const float FINF = __int_as_float(0x7f800000);
    const float DEAD = 1.0f + FERR;  // float value surely beyond the threshold

    const long long off = static_cast<long long>(blockIdx.x) * sc.cap;
    int* I = sc.idx + off;
    int* P = sc.pos + off;
    double* EX = sc.ex + off;
    int* EXU = sc.exu + off;
    int* MEMB = sc.memb + blockIdx.x * (sc.cap + 1);
    int* CONT = sc.cont + off;

    if (tid == 0) sNumCont = 0;
    // Either a device-side work list or a single seed (the winning cluster)
    const int nseeds = seeds ? *nseedsPtr : 1;

    for (;;) {
        if (tid == 0) {
            sSeedIdx = atomicAdd(workCounter, 1);
            sCnt = 0;
        }
        __syncthreads();
        const int si = sSeedIdx;
        if (si >= nseeds) break;

        const int seed = seeds ? seeds[si] : singleSeed;
        const int p = g.pos[seed];
        const double px = g.sx[p];
        const double py = g.sy[p];
        int rb[3], re[3], nrows;
        rowRanges(g, g.cellOf[p], rb, re, nrows);
        // Gather eligible candidates (unclustered, within threshold of seed)
        float2* XY = sc.xy + off;
        float* MF = sc.m2f + off;
        for (int r = 0; r < nrows; ++r) {
            for (int j = rb[r] + tid; j < re[r]; j += BS) {
                if (j == p || g.clustered[j]) continue;
                const double cx = g.sx[j], cy = g.sy[j];
                const double d2 = devDist2(cx, cy, px, py);
                if (d2 < g.thrSq) {
                    const int k = atomicAdd(&sCnt, 1);
                    if (k >= sc.cap) continue;  // overflow: seed is handed to the big tier
                    EX[k] = d2;
                    EXU[k] = 1;
                    const double nx = (cx - px) * g.invThr;
                    const double ny = (cy - py) * g.invThr;
                    XY[k] = make_float2(__double2float_rn(nx), __double2float_rn(ny));
                    MF[k] = __double2float_rn(nx * nx + ny * ny);
                    I[k] = g.sidx[j];
                    P[k] = j;
                }
            }
        }
        __syncthreads();
        int K = sCnt;
        if (K > sc.cap) {
            if (tid == 0) overflowList[atomicAdd(overflowCount, 1)] = seed;
            __syncthreads();
            continue;
        }
        // Move the hot data to shared memory when the neighbourhood fits
        if (K <= smemCap) {
            float2* sXY = reinterpret_cast<float2*>(smem4);
            float* sMF = reinterpret_cast<float*>(sXY + smemCap);
            for (int k = tid; k < K; k += BS) {
                sXY[k] = XY[k];
                sMF[k] = MF[k];
            }
            XY = sXY;
            MF = sMF;
        }
        if (tid == 0) {
            MEMB[0] = p;
            if (membersOut) membersOut[0] = seed;
        }
        __syncthreads();

        int size = 1;
        int removed = 0;
        int parity = 0;
        bool update = false;
        float qx = 0.0f, qy = 0.0f;
        for (;;) {
            // ---- Fast pass: fold in newest member, find two smallest values ----
            MinPair mp;
            mp.v1 = FINF;
            mp.s1 = -1;
            mp.v2 = FINF;
            int killed = 0;
            if (update) {
#pragma unroll 4
                for (int k = tid; k < K; k += BS) {
                    float v = MF[k];
                    const float2 c = XY[k];
                    const float dx = c.x - qx;
                    const float dy = c.y - qy;
                    const float d = fmaf(dx, dx, dy * dy);
                    if (d > v) {  // never true for dead entries (v = +inf)
                        v = d;
                        if (v >= DEAD) {
                            v = FINF;
                            ++killed;
                        }
                        MF[k] = v;
                    }
                    mergeMin(mp, v, k);
                }
            } else {
                for (int k = tid; k < K; k += BS) {
                    mergeMin(mp, MF[k], k);
                }
            }
            warpMinPair(mp);
            killed = warpSum(killed);
            if (lane == 0) {
                wV1[parity][warp] = mp.v1;
                wS1[parity][warp] = mp.s1;
                wV2[parity][warp] = mp.v2;
                wK[parity][warp] = killed;
            }
            __syncthreads();
            // Every warp reduces the partials redundantly -> uniform decision
            mp.v1 = (lane < NW) ? wV1[parity][lane] : FINF;
            mp.s1 = (lane < NW) ? wS1[parity][lane] : -1;
            mp.v2 = (lane < NW) ? wV2[parity][lane] : FINF;
            killed = (lane < NW) ? wK[parity][lane] : 0;
            parity ^= 1;
            warpMinPair(mp);
            const float v1 = mp.v1;
            const float v2 = mp.v2;
            const int s1 = mp.s1;
            removed += warpSum(killed);
            if (s1 < 0) break;  // no candidates left

            int w = s1;
            if (!(v2 > v1 + 2.0f * FERR && v1 + FERR < 1.0f)) {
                // ---- Slow path: exact evaluation of all near-minimal candidates ----
                const float lim = v1 + 2.0f * FERR;
                for (int k = tid; k < K; k += BS) {
                    const float v = MF[k];
                    if (v != FINF && v <= lim) CONT[atomicAdd(&sNumCont, 1)] = k;
                }
                __syncthreads();
                const int nc = sNumCont;
                for (int c = warp; c < nc; c += NW) {
                    // Bring the cached exact maximum up to date with new members
                    const int k = CONT[c];
                    const int pc = P[k];
                    const double cx = g.sx[pc], cy = g.sy[pc];
                    double m2 = EX[k];
                    for (int j = EXU[k] + lane; j < size; j += 32) {
                        const int pm = MEMB[j];
                        m2 = fmax(m2, devDist2(cx, cy, g.sx[pm], g.sy[pm]));
                    }
#pragma unroll
                    for (int o = 16; o > 0; o >>= 1) m2 = fmax(m2, __shfl_down_sync(0xffffffffu, m2, o));
                    if (lane == 0) {
                        EX[k] = m2;
                        EXU[k] = size;
                    }
                }
                __syncthreads();
                ExactBest eb;
                eb.m = 0.0;
                eb.idx = 0x7fffffff;
                eb.slot = -1;
                int dead = 0;
                for (int c = tid; c < nc; c += BS) {
                    const int k = CONT[c];
                    const double m2 = EX[k];
                    if (!(m2 < g.thrSq)) {
                        MF[k] = FINF;
                        ++dead;
                        continue;
                    }
                    const double m = __dsqrt_rn(m2);
                    const int ik = I[k];
                    if (eb.slot < 0 || m < eb.m || (m == eb.m && ik < eb.idx)) {
                        eb.m = m;
                        eb.idx = ik;
                        eb.slot = k;
                    }
                }
                warpExactBest(eb);
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) dead += __shfl_down_sync(0xffffffffu, dead, o);
                if (lane == 0) {
                    wM[warp] = eb.m;
                    wI[warp] = eb.idx;
                    wES[warp] = eb.slot;
                    wCnt[warp] = dead;
                }
                __syncthreads();
                eb.m = (lane < NW) ? wM[lane] : 0.0;
                eb.idx = (lane < NW) ? wI[lane] : 0x7fffffff;
                eb.slot = (lane < NW) ? wES[lane] : -1;
                dead = (lane < NW) ? wCnt[lane] : 0;
                warpExactBest(eb);
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) dead += __shfl_down_sync(0xffffffffu, dead, o);
                removed += __shfl_sync(0xffffffffu, dead, 0);
                const int es = __shfl_sync(0xffffffffu, eb.slot, 0);
                const double em = __shfl_sync(0xffffffffu, eb.m, 0);
                if (tid == 0) sNumCont = 0;
                // Every non-contender's exact value exceeds v1 + 2*FERR - 1.9e-6;
                // the exact best is only conclusive if it is clearly below that.
                const double nrm = em * g.invThr;
                if (es < 0 || nrm * nrm > static_cast<double>(v1) + FERR) {
                    update = false;  // inconclusive: rescan without a new member
                    continue;
                }
                w = es;
            }

            // ---- Accept winner w (all threads agree on it) ----
            if (tid == (w % BS)) MF[w] = FINF;
            const float2 q = XY[w];
            qx = q.x;
            qy = q.y;
            if (tid == 0) {
                MEMB[size] = P[w];
                if (membersOut) membersOut[size] = I[w];
            }
            ++size;
            ++removed;
            update = true;

            // Compact once at least half of the list is dead
            if (2 * removed > K && K > 2 * BS) {
                removed = 0;
                __syncthreads();
                // In-place compaction, one block-wide chunk per round:
                // writes of a round never reach entries of later rounds.
                int newK = 0;
                for (int base = 0; base < K; base += BS) {
                    const int k = base + tid;
                    float2 vxy = make_float2(0.f, 0.f);
                    float vm = FINF;
                    int vi = 0, vp = 0, vu = 0;
                    double vex = 0.0;
                    if (k < K) {
                        vm = MF[k];
                        if (vm != FINF) {
                            vxy = XY[k];
                            vi = I[k];
                            vp = P[k];
                            vex = EX[k];
                            vu = EXU[k];
                        }
                    }
                    const bool alive = vm != FINF;
                    const unsigned bal = __ballot_sync(0xffffffffu, alive);
                    if (lane == 0) wCnt[warp] = __popc(bal);
                    __syncthreads();
                    int pre = 0, tot = 0;
                    for (int ww = 0; ww < NW; ++ww) {
                        const int c = wCnt[ww];
                        pre += (ww < warp) ? c : 0;
                        tot += c;
                    }
                    if (alive) {
                        const int dst = newK + pre + __popc(bal & ((1u << lane) - 1u));
                        XY[dst] = vxy;
                        MF[dst] = vm;
                        I[dst] = vi;
                        P[dst] = vp;
                        EX[dst] = vex;
                        EXU[dst] = vu;
                    }
                    newK += tot;
                    __syncthreads();
                }
                K = newK;
            }
        }
        if (tid == 0) card[seed] = size;
        __syncthreads();
        if (membersOut) {
            // Winning cluster: mark its members clustered
            for (int j = tid; j < size; j += BS) g.clustered[MEMB[j]] = 1;
        }
    }
}

// Best (max cardinality, then smallest index) unclustered seed owned by this rank
__global__ void bestSeedKernel(DeviceGrid g, int N, int rank, int nranks,
                               const int* __restrict__ card, unsigned long long* key,
                               int* resetCounters) {
    // Work counters and the dirty count are idle here: reset them for later kernels
    if (blockIdx.x == 0 && threadIdx.x < 5) resetCounters[threadIdx.x] = 0;
    unsigned long long best = 0;
    const long long nOwned = (N - rank + nranks - 1) / nranks;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < nOwned;
         i += (long long)gridDim.x * blockDim.x) {
        const int s = rank + static_cast<int>(i) * nranks;
        if (g.clustered[g.pos[s]]) continue;
        const unsigned long long k =
            (static_cast<unsigned long long>(card[s]) << 32) | (0xffffffffu - static_cast<unsigned>(s));
        best = (k > best) ? k : best;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const unsigned long long ok = __shfl_down_sync(0xffffffffu, best, o);
        best = (ok > best) ? ok : best;
    }
    if ((threadIdx.x & 31) == 0 && best) atomicMax(key, best);
}

// Unclustered points within threshold of any new member must recompute their
// candidate cluster; all other cached cardinalities remain exact.
__global__ void markDirtyKernel(DeviceGrid g, const int* members, unsigned char* dirty) {
    const int p = g.pos[members[blockIdx.x]];
    const double mx = g.sx[p], my = g.sy[p];
    int rb[3], re[3], nrows;
    rowRanges(g, g.cellOf[p], rb, re, nrows);
    for (int r = 0; r < nrows; ++r) {
        for (int j = rb[r] + threadIdx.x; j < re[r]; j += blockDim.x) {
            if (g.clustered[j] || dirty[j]) continue;
            if (devDist2(g.sx[j], g.sy[j], mx, my) < g.thrSq) dirty[j] = 1;
        }
    }
}

__global__ void collectDirtyKernel(DeviceGrid g, int N, int rank, int nranks,
                                   unsigned char* dirty, int* list, int* count,
                                   unsigned long long* key) {
    // The best-seed key has been consumed by the host: reset for the next round
    if (blockIdx.x == 0 && threadIdx.x == 0) *key = 0;
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < N; j += gridDim.x * blockDim.x) {
        if (!dirty[j]) continue;
        dirty[j] = 0;
        const int s = g.sidx[j];
        if (s % nranks == rank) list[atomicAdd(count, 1)] = s;
    }
}

// Main QT clustering algorithm (distributed over MPI ranks, one GPU per rank)
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    const int N = static_cast<int>(points.size());
    std::vector<Cluster> clusters;

    // ---- Build uniform grid (cell size >= threshold) on the host ----
    // (slightly enlarged so that rounding in the cell computation cannot break
    //  the 3x3 neighbourhood guarantee)
    const double cell_size = std::max(threshold * (1.0 + 1e-9), std::max(MAX_WIDTH, MAX_HEIGHT) / 1024.0);
    const int gx = static_cast<int>(MAX_WIDTH / cell_size) + 1;
    const int gy = static_cast<int>(MAX_HEIGHT / cell_size) + 1;
    const int ncells = gx * gy;

    std::vector<int> cellOfPoint(N);
#pragma omp parallel for schedule(static) if (N >= OMP_MIN_WORK)
    for (int i = 0; i < N; ++i) {
        int cx = static_cast<int>(points[i].x / cell_size);
        int cy = static_cast<int>(points[i].y / cell_size);
        cx = std::min(std::max(cx, 0), gx - 1);
        cy = std::min(std::max(cy, 0), gy - 1);
        cellOfPoint[i] = cy * gx + cx;
    }
    std::vector<int> cellStart(ncells + 1, 0);
    for (int i = 0; i < N; ++i) cellStart[cellOfPoint[i] + 1]++;
    for (int c = 0; c < ncells; ++c) cellStart[c + 1] += cellStart[c];
    std::vector<int> fill(cellStart.begin(), cellStart.end() - 1);
    std::vector<double> hsx(N), hsy(N);
    std::vector<int> hsidx(N), hcell(N), hpos(N);
    for (int i = 0; i < N; ++i) {
        const int p = fill[cellOfPoint[i]]++;
        hsx[p] = points[i].x;
        hsy[p] = points[i].y;
        hsidx[p] = i;
        hcell[p] = cellOfPoint[i];
        hpos[i] = p;
    }
    // Upper bound of points in any 3x3 neighbourhood
    long long maxNb = 0;
#pragma omp parallel for reduction(max : maxNb) schedule(static) if (ncells >= OMP_MIN_WORK)
    for (int c = 0; c < ncells; ++c) {
        const int cx = c % gx, cy = c / gx;
        const int x0 = std::max(cx - 1, 0), x1 = std::min(cx + 1, gx - 1);
        long long n = 0;
        for (int r = std::max(cy - 1, 0); r <= std::min(cy + 1, gy - 1); ++r) {
            n += cellStart[r * gx + x1 + 1] - cellStart[r * gx + x0];
        }
        maxNb = std::max(maxNb, n);
    }

    // ---- Device setup ----
    int numSMs = 0, dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, dev));
    // Shared memory holds the hot candidate data of a neighbourhood when it fits
    int maxOptin = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&maxOptin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
    const int staticSmem = 4096;  // reserve for the kernel's static shared variables
    const long long nbBytes = std::max(1LL, maxNb) * HOT_BYTES;
    auto smemFor = [&](long long limit) {
        return static_cast<int>(std::min<long long>(std::min<long long>(nbBytes, limit),
                                                    maxOptin - staticSmem));
    };
    const long long smallCapK = std::min<long long>(SMALL_K, std::max(1LL, maxNb));
    const int smallSmem = smemFor(smallCapK * HOT_BYTES);
    const int bigSmem = smemFor(BIG_SMEM_LIMIT);
    const int winSmem = smemFor(maxOptin);
    const int smallCap = smallSmem / HOT_BYTES;
    const int bigCap = bigSmem / HOT_BYTES;
    const int winCap = winSmem / HOT_BYTES;
    CUDA_CHECK(cudaFuncSetAttribute(candidateClusterKernel<SMALL_BS>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, smallSmem));
    CUDA_CHECK(cudaFuncSetAttribute(candidateClusterKernel<BIG_BS>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, bigSmem));
    CUDA_CHECK(cudaFuncSetAttribute(candidateClusterKernel<WIN_BS>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, winSmem));
    int smallPerSM = 0, bigPerSM = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &smallPerSM, candidateClusterKernel<SMALL_BS>, SMALL_BS, smallSmem));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &bigPerSM, candidateClusterKernel<BIG_BS>, BIG_BS, bigSmem));
    const int smallBlocks = std::max(1, numSMs * std::max(1, smallPerSM));
    int bigBlocks = std::max(1, numSMs * std::max(1, bigPerSM));

    // Global per-block scratch: small tier up to SMALL_K candidates, big tier
    // (and the winner kernel, which reuses slot 0) the largest neighbourhood
    const long long bigCapK = std::max(1LL, maxNb);
    const size_t perCand = 3 * sizeof(float) + sizeof(double) + 5 * sizeof(int);
    {
        size_t freeMem = 0, totalMem = 0;
        CUDA_CHECK(cudaMemGetInfo(&freeMem, &totalMem));
        const size_t smallBytes = static_cast<size_t>(smallCapK + 1) * perCand * smallBlocks;
        const size_t budget = static_cast<size_t>(freeMem * 0.6) - std::min(smallBytes, static_cast<size_t>(freeMem * 0.3));
        const long long fitBlocks = static_cast<long long>(budget / (static_cast<size_t>(bigCapK + 1) * perCand));
        bigBlocks = static_cast<int>(std::max(1LL, std::min<long long>(bigBlocks, fitBlocks)));
    }

    double *d_sx, *d_sy;
    int *d_sidx, *d_cell, *d_pos, *d_cellStart, *d_card, *d_list, *d_members, *d_counters;
    unsigned char *d_clustered, *d_dirty;
    unsigned long long* d_key;
    CUDA_CHECK(cudaMalloc(&d_sx, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_sy, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_sidx, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cell, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_pos, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cellStart, (ncells + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_card, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_list, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_members, (N + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_counters, 8 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_clustered, N));
    CUDA_CHECK(cudaMalloc(&d_dirty, N));
    CUDA_CHECK(cudaMalloc(&d_key, sizeof(unsigned long long)));
    auto allocScratch = [](Scratch& s, long long cap, int blocks) {
        const size_t n = static_cast<size_t>(cap) * blocks;
        CUDA_CHECK(cudaMalloc(&s.xy, n * sizeof(float2)));
        CUDA_CHECK(cudaMalloc(&s.m2f, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&s.ex, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&s.exu, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&s.idx, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&s.pos, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&s.cont, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&s.memb, static_cast<size_t>(cap + 1) * blocks * sizeof(int)));
        s.cap = cap;
    };
    auto freeScratch = [](Scratch& s) {
        cudaFree(s.xy);
        cudaFree(s.m2f);
        cudaFree(s.ex);
        cudaFree(s.exu);
        cudaFree(s.idx);
        cudaFree(s.pos);
        cudaFree(s.cont);
        cudaFree(s.memb);
    };
    Scratch scSmall, scBig;
    allocScratch(scSmall, smallCapK, smallBlocks);
    allocScratch(scBig, bigCapK, bigBlocks);
    int* d_bigList;
    CUDA_CHECK(cudaMalloc(&d_bigList, std::max(1, N) * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_sx, hsx.data(), N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sy, hsy.data(), N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sidx, hsidx.data(), N * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cell, hcell.data(), N * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pos, hpos.data(), N * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cellStart, cellStart.data(), (ncells + 1) * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_clustered, 0, N));
    CUDA_CHECK(cudaMemset(d_dirty, 0, N));
    CUDA_CHECK(cudaMemset(d_card, 0, N * sizeof(int)));

    DeviceGrid g;
    g.sx = d_sx;
    g.sy = d_sy;
    g.sidx = d_sidx;
    g.cellOf = d_cell;
    g.pos = d_pos;
    g.cellStart = d_cellStart;
    g.clustered = d_clustered;
    g.gx = gx;
    g.gy = gy;
    g.threshold = threshold;
    // Exact boundary for "sqrt(d2) < threshold" expressed on d2
    double T = threshold * threshold;
    while (T > 0.0 && std::sqrt(T) >= threshold) T = std::nextafter(T, 0.0);
    while (std::sqrt(T) < threshold) T = std::nextafter(T, std::numeric_limits<double>::infinity());
    g.thrSq = T;
    g.invThr = 1.0 / threshold;

    // Initially every seed owned by this rank needs its candidate cluster
    std::vector<int> owned;
    for (int s = rank; s < N; s += nranks) owned.push_back(s);
    const int nOwned = static_cast<int>(owned.size());
    if (nOwned > 0) {
        CUDA_CHECK(cudaMemcpy(d_list, owned.data(), nOwned * sizeof(int), cudaMemcpyHostToDevice));
    }
    // counters: [0] work small tier, [1] work big tier, [2] work winner,
    //           [3] big-tier list length, [4] dirty count
    int* d_work = d_counters;
    int* d_workBig = d_counters + 1;
    int* d_work1 = d_counters + 2;
    int* d_bigCount = d_counters + 3;
    int* d_dcount = d_counters + 4;
    CUDA_CHECK(cudaMemset(d_counters, 0, 8 * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_dcount, &nOwned, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_key, 0, sizeof(unsigned long long)));

    const int bestBlocks = std::max(1, std::min((nOwned + 255) / 256, numSMs * 8));
    const int collectBlocks = std::max(1, std::min((N + 255) / 256, numSMs * 8));
    std::vector<std::pair<int, int>> found;  // (seed, cardinality) per cluster
    int remaining = N;
    int offset = 0;  // members of all clusters are stored contiguously on the device

    while (remaining > 0) {
        // Recompute candidate clusters of affected seeds owned by this rank
        candidateClusterKernel<SMALL_BS><<<smallBlocks, SMALL_BS, smallSmem>>>(
            g, d_list, d_dcount, 0, d_work, d_card, nullptr, scSmall, smallCap, d_bigList, d_bigCount);
        candidateClusterKernel<BIG_BS><<<bigBlocks, BIG_BS, bigSmem>>>(
            g, d_bigList, d_bigCount, 0, d_workBig, d_card, nullptr, scBig, bigCap, nullptr, nullptr);
        bestSeedKernel<<<bestBlocks, 256>>>(g, N, rank, nranks, d_card, d_key, d_counters);
        unsigned long long localKey = 0, globalKey = 0;
        CUDA_CHECK(cudaMemcpy(&localKey, d_key, sizeof(localKey), cudaMemcpyDeviceToHost));
        MPI_Allreduce(&localKey, &globalKey, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
        if (globalKey == 0) break;

        const int best_seed = static_cast<int>(0xffffffffu - static_cast<unsigned>(globalKey & 0xffffffffu));
        const int best_card = static_cast<int>(globalKey >> 32);

        // Every rank regenerates the winning cluster (deterministic) - avoids a broadcast
        candidateClusterKernel<WIN_BS><<<1, WIN_BS, winSmem>>>(
            g, nullptr, nullptr, best_seed, d_work1, d_card, d_members + offset, scBig, winCap,
            nullptr, nullptr);
        markDirtyKernel<<<best_card, 256>>>(g, d_members + offset, d_dirty);
        collectDirtyKernel<<<collectBlocks, 256>>>(g, N, rank, nranks, d_dirty, d_list, d_dcount, d_key);

        found.emplace_back(best_seed, best_card);
        offset += best_card;
        remaining -= best_card;
    }
    CUDA_CHECK(cudaGetLastError());
    std::vector<int> allMembers(offset);
    if (offset > 0) {
        CUDA_CHECK(cudaMemcpy(allMembers.data(), d_members, offset * sizeof(int), cudaMemcpyDeviceToHost));
    }
    clusters.resize(found.size());
    for (size_t c = 0, o = 0; c < found.size(); ++c) {
        clusters[c].seed_point = found[c].first;
        clusters[c].members.assign(allMembers.begin() + o, allMembers.begin() + o + found[c].second);
        o += found[c].second;
    }

    cudaFree(d_sx);
    cudaFree(d_sy);
    cudaFree(d_sidx);
    cudaFree(d_cell);
    cudaFree(d_pos);
    cudaFree(d_cellStart);
    cudaFree(d_card);
    cudaFree(d_list);
    cudaFree(d_members);
    cudaFree(d_counters);
    cudaFree(d_clustered);
    cudaFree(d_dirty);
    cudaFree(d_key);
    freeScratch(scSmall);
    freeScratch(scBig);
    cudaFree(d_bigList);
    return clusters;
}

// Validation: check that clusters satisfy the QT clustering properties
bool validateClusters(const std::vector<Cluster>& clusters,
                     const std::vector<Point>& points,
                     const double threshold) {
    bool valid = true;
    
    printf("Validating clusters:\n");
    
    // Compute cluster diameters in parallel (max distance between any two points)
    std::vector<double> diameters(clusters.size(), 0.0);
#pragma omp parallel for schedule(dynamic, 1)
    for (size_t c = 0; c < clusters.size(); ++c) {
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

// Start the OpenMP thread pool and load all CUDA kernels before any timing
static void warmUp() {
    volatile int sink = 0;
#pragma omp parallel
    {
        if (omp_get_thread_num() == 0) sink = omp_get_num_threads();
    }
    (void)sink;
    cudaFuncAttributes attr;
    CUDA_CHECK(cudaFuncGetAttributes(&attr, candidateClusterKernel<SMALL_BS>));
    CUDA_CHECK(cudaFuncGetAttributes(&attr, candidateClusterKernel<BIG_BS>));
    CUDA_CHECK(cudaFuncGetAttributes(&attr, candidateClusterKernel<WIN_BS>));
    CUDA_CHECK(cudaFuncGetAttributes(&attr, bestSeedKernel));
    CUDA_CHECK(cudaFuncGetAttributes(&attr, markDirtyKernel));
    CUDA_CHECK(cudaFuncGetAttributes(&attr, collectDirtyKernel));
}

static int runMain(int argc, char** argv, const int rank) {
    int num_points = 1000;
    double threshold = 2.0;
    bool validate = false;
    bool printResults = false;
    const bool root = (rank == 0);
    
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
        if (root) {
            printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n", 
                   num_points, threshold);
        }
        return 1;
    }
    
    if (root) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Generate synthetic data (deterministic, identical on every rank)
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);
    
    // Perform QT clustering
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_start = std::chrono::high_resolution_clock::now();
    
    const std::vector<Cluster> clusters = qtClustering(points, threshold);
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);
    
    // All ranks hold the identical result; only rank 0 reports
    const long local_time_ms = cluster_time.count();
    long max_time_ms = 0;
    MPI_Reduce(&local_time_ms, &max_time_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!root) return 0;
    cluster_time = std::chrono::milliseconds(max_time_ms);

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
        std::vector<double> membershipData(num_points);
        std::vector<int> membership(num_points, -1);
#pragma omp parallel for schedule(dynamic, 1) if (num_points >= OMP_MIN_WORK)
        for (size_t c = 0; c < clusters.size(); ++c) {
            for (size_t i = 0; i < clusters[c].members.size(); ++i) {
                membership[clusters[c].members[i]] = static_cast<int>(c);
            }
        }
#pragma omp parallel for schedule(static) if (num_points >= OMP_MIN_WORK)
        for (int i = 0; i < num_points; ++i) {
            membershipData[i] = static_cast<double>(membership[i]);
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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // One GPU per rank: map node-local rank onto the node's devices
    MPI_Comm local;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
    int local_rank = 0, local_size = 1;
    MPI_Comm_rank(local, &local_rank);
    MPI_Comm_size(local, &local_size);
    MPI_Comm_free(&local);
    // Share the node's hardware threads among the node-local ranks
    const int hw = static_cast<int>(std::thread::hardware_concurrency());
    if (hw > 0) {
        omp_set_num_threads(std::max(1, std::min(omp_get_max_threads(), hw / local_size)));
    }
    int ndev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    if (ndev <= 0) {
        fprintf(stderr, "Error: no CUDA device available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(local_rank % ndev));
    CUDA_CHECK(cudaFree(nullptr)); // create context outside the timed region
    warmUp();

    int rc = runMain(argc, argv, rank);
    MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD);
    fflush(stdout);
    MPI_Finalize();
    return rc;
}
