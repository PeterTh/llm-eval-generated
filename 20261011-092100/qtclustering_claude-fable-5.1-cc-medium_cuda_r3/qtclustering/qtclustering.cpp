// QT Clustering Benchmark - CUDA Parallel Version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// Parallelization strategy (CUDA):
//   Every round of the algorithm has to grow one candidate cluster from each
//   still-unclustered seed point. These candidate clusters are independent,
//   so each one is grown by a thread block (a persistent grid pulls seeds
//   from a work queue). Inside a block, the threads cooperatively own the
//   candidate points: for each candidate the block keeps the running maximum
//   squared distance to the members already in the cluster (updated
//   incrementally whenever a member joins, which equals the maximum the
//   original code recomputes from scratch), and a block-wide argmin reduction
//   selects the next member with the same tie-breaking as the sequential
//   scan (smallest distance, then smallest index). Member lists are recorded
//   while growing, so the round's winner (and runner-up) are available
//   immediately.
//
//   Exact work reduction on top of that:
//     * A seed whose remaining potential (members so far + still admissible
//       candidates) cannot beat the best cluster already finished in the
//       round is abandoned early. Seeds are scheduled largest-estimate first
//       so that this bound becomes effective quickly.
//     * A removed cluster can only change the candidate cluster of a seed if
//       one of the removed points lies within the threshold of that seed;
//       results (and pruning bounds) of unaffected seeds are carried over to
//       the following rounds without recomputation.
//
//   The unclustered points are compacted into a dense array (in ascending
//   index order) before every round so that no work is spent on points that
//   are already clustered.

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
inline double distance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

// ---------------------------------------------------------------------------
// CUDA kernels
// ---------------------------------------------------------------------------

namespace {

constexpr int kBlockSize = 256;
constexpr int kWarpSize = 32;
constexpr int kNumWarps = kBlockSize / kWarpSize;
constexpr int kMaxBlocksPerSM = 2048 / kBlockSize;
constexpr int kNumBufs = 3;  // member buffers per block: current, best, second
constexpr int kInfoInts = 4; // output header: winner (card, seed), runner-up (card, seed)
// Largest dynamic shared memory allocation we are willing to request for the
// per-candidate running-maximum array (96 KiB, available on sm_70+).
constexpr size_t kMaxSmemBytes = 96 * 1024;

// Squared distance. The original algorithm compares Euclidean distances
// sqrt(d2); since sqrt is monotonic and correctly rounded, max over members
// commutes with sqrt, "sqrt(d2) < threshold" is equivalent to "d2 < T" for a
// suitably chosen bound T (see squaredThresholdBound), and the argmin is
// resolved on squared values with an exact fallback for the (practically
// non-existent) case where distinct squared values round to the same sqrt.
__host__ __device__ __forceinline__ double distance2(const double2 a, const double2 b) {
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    return dx * dx + dy * dy;
}

// (value, index) pair ordering identical to the sequential scan: a candidate
// replaces the current best only if it is strictly closer, so among equal
// distances the smallest index wins.
__device__ __forceinline__ bool betterThan(double v, int i, double bv, int bi) {
    return (v < bv) || (v == bv && i < bi);
}

__device__ __forceinline__ void warpArgMin(double& v, int& i) {
#pragma unroll
    for (int off = kWarpSize / 2; off > 0; off >>= 1) {
        const double ov = __shfl_down_sync(0xffffffffu, v, off);
        const int oi = __shfl_down_sync(0xffffffffu, i, off);
        if (betterThan(ov, oi, v, i)) {
            v = ov;
            i = oi;
        }
    }
}

// Selection state: smallest value (with smallest index among equals), the
// smallest value strictly greater than it, and the number of valid candidates.
struct Sel {
    double min1;
    int idx;
    double min2;
    int cnt;
};

__device__ __forceinline__ void mergeSel(Sel& a, const Sel& b) {
    a.cnt += b.cnt;
    if (b.min1 < a.min1) {
        a.min2 = fmin(a.min1, b.min2);
        a.min1 = b.min1;
        a.idx = b.idx;
    } else if (b.min1 == a.min1) {
        a.idx = min(a.idx, b.idx);
        a.min2 = fmin(a.min2, b.min2);
    } else {
        a.min2 = fmin(a.min2, b.min1);
    }
}

__device__ __forceinline__ void warpMergeSel(Sel& s) {
#pragma unroll
    for (int off = kWarpSize / 2; off > 0; off >>= 1) {
        Sel o;
        o.min1 = __shfl_down_sync(0xffffffffu, s.min1, off);
        o.idx = __shfl_down_sync(0xffffffffu, s.idx, off);
        o.min2 = __shfl_down_sync(0xffffffffu, s.min2, off);
        o.cnt = __shfl_down_sync(0xffffffffu, s.cnt, off);
        mergeSel(s, o);
    }
}

// Packed (cardinality, seed) with the ordering used to pick the round's
// cluster: larger cardinality first, then the smaller seed index.
__host__ __device__ __forceinline__ unsigned long long packBest(int card, int seed, int U) {
    return (static_cast<unsigned long long>(static_cast<unsigned int>(card)) << 32) |
           static_cast<unsigned int>(U - seed);
}
__host__ __device__ __forceinline__ int packedCard(unsigned long long p) {
    return static_cast<int>(p >> 32);
}
__host__ __device__ __forceinline__ int packedSeed(unsigned long long p, int U) {
    return U - static_cast<int>(p & 0xffffffffu);
}

// Number of points within the threshold of each point; used only as a
// scheduling heuristic (ordering of seeds) for the first round.
__global__ void __launch_bounds__(kBlockSize)
neighborCountKernel(const double2* __restrict__ pts, int N, double T, int* __restrict__ out) {
    __shared__ int red[kNumWarps];
    const int i = blockIdx.x;
    const double2 p = pts[i];
    int cnt = 0;
    for (int j = threadIdx.x; j < N; j += kBlockSize) {
        if (distance2(p, pts[j]) < T) ++cnt;
    }
#pragma unroll
    for (int off = kWarpSize / 2; off > 0; off >>= 1) cnt += __shfl_down_sync(0xffffffffu, cnt, off);
    if ((threadIdx.x & (kWarpSize - 1)) == 0) red[threadIdx.x >> 5] = cnt;
    __syncthreads();
    if (threadIdx.x == 0) {
        int total = 0;
        for (int w = 0; w < kNumWarps; ++w) total += red[w];
        out[i] = total;
    }
}

// Marks every point that lies within the threshold of some member of the
// cluster chosen in this round (header/member list taken from `out`). Only
// such seeds can have their candidate cluster changed by the removal.
__global__ void affectedKernel(const double2* __restrict__ pts, int U, double T,
                               const int* __restrict__ out, int N, int* __restrict__ flags) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= U) return;
    const int card = out[0];
    const int* members = out + kInfoInts + N;
    const double2 p = pts[i];
    int f = 0;
    for (int j = 0; j < card; ++j) {
        if (distance2(p, pts[members[j]]) < T) { f = 1; break; }
    }
    flags[i] = f;
}

enum : int { kStepContinue = 0, kStepFinished = 1, kStepPruned = 2, kStepFallback = 3 };

// Persistent kernel: blocks pull seeds from a work queue (in the host-chosen
// order) and grow one candidate cluster per seed, exactly like the
// sequential generateCandidateCluster but with an incrementally maintained
// per-candidate running maximum.
//
//   pts         : compacted coordinates of the U unclustered points
//   U           : number of unclustered points
//   T           : squared-distance bound equivalent to "distance < threshold"
//   seedOrder   : seeds to process, in processing order (slot -> seed)
//   numSeeds    : number of seeds to process
//   out         : [0,2): winner (card, seed), card negated if the winner's
//                 members are not available (host-known winner);
//                 [2,4): runner-up (card, seed) or (0, -1);
//                 [4, 4+numSeeds): per-slot result (cardinality > 0, or
//                 -upperBound if the seed was pruned);
//                 [4+N, ...): winner members, [4+2N, ...): runner-up members
//   gBest       : packed best known (cardinality, seed) (host-initialised)
//   workCounter : next slot (init 0), doneCounter: finished blocks (init 0)
//   blockBest.. : per-block bookkeeping for the member lists
//   memberBufs  : kNumBufs * U ints per block
//   scratch     : U doubles per block if !SMEM
template <bool SMEM>
__global__ void __launch_bounds__(kBlockSize)
growClustersKernel(const double2* __restrict__ pts,
                   const int U,
                   const double T,
                   const int* __restrict__ seedOrder,
                   const int numSeeds,
                   int* __restrict__ out,
                   const int N,
                   unsigned long long* __restrict__ gBest,
                   int* __restrict__ workCounter,
                   int* __restrict__ doneCounter,
                   unsigned long long* __restrict__ blockBest,
                   unsigned long long* __restrict__ blockSecond,
                   int* __restrict__ blockBestBuf,
                   int* __restrict__ blockSecondBuf,
                   int* __restrict__ memberBufs,
                   double* __restrict__ scratch) {
    extern __shared__ double dynSmem[];
    __shared__ double redMin1[kNumWarps];
    __shared__ double redMin2[kNumWarps];
    __shared__ int redIdx[kNumWarps];
    __shared__ int redCnt[kNumWarps];
    __shared__ unsigned long long redU64[kNumWarps];
    __shared__ int sNext;
    __shared__ int sFlag;
    __shared__ int sSlot;
    __shared__ int sBuf;
    __shared__ int sBlockA;
    __shared__ int sBlockB;
    __shared__ int sBufB;
    __shared__ int sIsLast;

    double* maxd = SMEM ? dynSmem : scratch + static_cast<size_t>(blockIdx.x) * U;
    int* myBufs = memberBufs + static_cast<size_t>(blockIdx.x) * kNumBufs * U;

    const int tid = threadIdx.x;
    const int lane = tid & (kWarpSize - 1);
    const int warp = tid >> 5;
    const double INF = INFINITY;

    int* cards = out + kInfoInts;
    int* winnerMembers = out + kInfoInts + N;
    int* secondMembers = out + kInfoInts + 2 * N;

    // Thread 0 bookkeeping: best / second best finished seed of this block.
    unsigned long long myBest = 0ull, mySecond = 0ull;
    int bufBest = 0, bufSecond = 1, bufCur = 2;

    while (true) {
        if (tid == 0) {
            sSlot = atomicAdd(workCounter, 1);
            sBuf = bufCur;
        }
        __syncthreads();
        const int slot = sSlot;
        if (slot >= numSeeds) break;
        const int seed = seedOrder[slot];
        int* members = myBufs + sBuf * U;

        // Initialise: nothing in the cluster but the seed. A running maximum
        // of 0 makes the first update produce exactly dist2(candidate, seed);
        // +inf marks points that are members of this cluster.
        for (int c = tid; c < U; c += kBlockSize) {
            maxd[c] = (c == seed) ? INF : 0.0;
        }
        if (tid == 0) members[0] = seed;
        __syncthreads();  // maxd initialised

        int member = seed;
        int count = 1;
        int flag = kStepContinue;
        int upperBound = 0;

        // Each step ends with a barrier after thread 0 has published the new
        // member (and marked it in maxd), so no barrier is needed here.
        while (true) {
            // Prefetch the global best (only consumed by thread 0 later).
            unsigned long long g = 0ull;
            if (tid == 0) g = *reinterpret_cast<volatile unsigned long long*>(gBest);
            const double2 mp = pts[member];

            Sel s;
            s.min1 = INF;
            s.idx = INT_MAX;
            s.min2 = INF;
            s.cnt = 0;
            for (int c = tid; c < U; c += kBlockSize) {
                double v = maxd[c];
                if (v != INF) {
                    v = fmax(v, distance2(pts[c], mp));
                    maxd[c] = v;
                    if (v < T) {
                        ++s.cnt;
                        if (v < s.min1) {
                            s.min2 = s.min1;
                            s.min1 = v;
                            s.idx = c;
                        } else if (v != s.min1 && v < s.min2) {
                            s.min2 = v;
                        }
                    }
                }
            }

            warpMergeSel(s);
            if (lane == 0) {
                redMin1[warp] = s.min1;
                redIdx[warp] = s.idx;
                redMin2[warp] = s.min2;
                redCnt[warp] = s.cnt;
            }
            __syncthreads();
            if (warp == 0) {
                Sel r;
                r.min1 = (lane < kNumWarps) ? redMin1[lane] : INF;
                r.idx = (lane < kNumWarps) ? redIdx[lane] : INT_MAX;
                r.min2 = (lane < kNumWarps) ? redMin2[lane] : INF;
                r.cnt = (lane < kNumWarps) ? redCnt[lane] : 0;
                warpMergeSel(r);
                if (lane == 0) {
                    const int ub = count + r.cnt;
                    int f;
                    if (packBest(ub, seed, U) < g) {
                        // Cannot become the (first) largest cluster of this round.
                        f = kStepPruned;
                        upperBound = ub;
                    } else if (r.idx == INT_MAX) {
                        f = kStepFinished;  // No more points can be added
                    } else if (r.min2 - r.min1 <= r.min1 * 0x1p-50 &&
                               sqrt(r.min2) == sqrt(r.min1)) {
                        f = kStepFallback;  // distinct squares, same distance
                    } else {
                        f = kStepContinue;
                        maxd[r.idx] = INF;
                        members[count] = r.idx;
                    }
                    sFlag = f;
                    sNext = r.idx;
                }
            }
            __syncthreads();
            flag = sFlag;
            if (flag == kStepFallback) {
                // Exact selection on actual distances (same tie-breaking as
                // the sequential scan).
                double bv = INF;
                int bi = INT_MAX;
                for (int c = tid; c < U; c += kBlockSize) {
                    const double v = maxd[c];
                    if (v != INF && v < T) {
                        const double d = sqrt(v);
                        if (betterThan(d, c, bv, bi)) {
                            bv = d;
                            bi = c;
                        }
                    }
                }
                warpArgMin(bv, bi);
                if (lane == 0) {
                    redMin1[warp] = bv;
                    redIdx[warp] = bi;
                }
                __syncthreads();
                if (warp == 0) {
                    double v = (lane < kNumWarps) ? redMin1[lane] : INF;
                    int i = (lane < kNumWarps) ? redIdx[lane] : INT_MAX;
                    warpArgMin(v, i);
                    if (lane == 0) {
                        maxd[i] = INF;
                        members[count] = i;
                        sNext = i;
                    }
                }
                __syncthreads();
                flag = kStepContinue;
            }
            if (flag != kStepContinue) break;
            member = sNext;
            ++count;
        }

        if (tid == 0) {
            if (flag == kStepPruned) {
                cards[slot] = -upperBound;
            } else {
                cards[slot] = count;
                const unsigned long long packed = packBest(count, seed, U);
                atomicMax(gBest, packed);
                if (packed > myBest) {
                    mySecond = myBest;
                    myBest = packed;
                    const int freed = bufSecond;
                    bufSecond = bufBest;
                    bufBest = bufCur;
                    bufCur = freed;
                } else if (packed > mySecond) {
                    mySecond = packed;
                    const int freed = bufSecond;
                    bufSecond = bufCur;
                    bufCur = freed;
                }
            }
        }
    }

    // Publish this block's results; the last block to finish copies the
    // member lists of the winner and of the runner-up to the output.
    if (tid == 0) {
        blockBest[blockIdx.x] = myBest;
        blockSecond[blockIdx.x] = mySecond;
        blockBestBuf[blockIdx.x] = bufBest;
        blockSecondBuf[blockIdx.x] = bufSecond;
        __threadfence();
        const int d = atomicAdd(doneCounter, 1);
        sIsLast = (d == static_cast<int>(gridDim.x) - 1) ? 1 : 0;
        sBlockA = -1;
        sBlockB = -1;
        sBufB = 0;
    }
    __syncthreads();
    if (!sIsLast) return;
    __threadfence();

    const unsigned long long best = *reinterpret_cast<volatile unsigned long long*>(gBest);
    const int grid = static_cast<int>(gridDim.x);
    unsigned long long sec = 0ull;
    for (int e = tid; e < 2 * grid; e += kBlockSize) {
        const int b = e >> 1;
        const unsigned long long v = (e & 1)
            ? *reinterpret_cast<volatile unsigned long long*>(blockSecond + b)
            : *reinterpret_cast<volatile unsigned long long*>(blockBest + b);
        if (v == best) {
            if (!(e & 1)) sBlockA = b;
        } else if (v > sec) {
            sec = v;
        }
    }
#pragma unroll
    for (int off = kWarpSize / 2; off > 0; off >>= 1) {
        const unsigned long long o = __shfl_down_sync(0xffffffffu, sec, off);
        sec = (o > sec) ? o : sec;
    }
    if (lane == 0) redU64[warp] = sec;
    __syncthreads();
    if (warp == 0) {
        unsigned long long v = (lane < kNumWarps) ? redU64[lane] : 0ull;
#pragma unroll
        for (int off = kWarpSize / 2; off > 0; off >>= 1) {
            const unsigned long long o = __shfl_down_sync(0xffffffffu, v, off);
            v = (o > v) ? o : v;
        }
        if (lane == 0) redU64[0] = v;
    }
    __syncthreads();
    const unsigned long long second = redU64[0];
    if (second != 0ull) {
        for (int e = tid; e < 2 * grid; e += kBlockSize) {
            const int b = e >> 1;
            const unsigned long long v = (e & 1)
                ? *reinterpret_cast<volatile unsigned long long*>(blockSecond + b)
                : *reinterpret_cast<volatile unsigned long long*>(blockBest + b);
            if (v == second) {
                sBlockB = b;
                sBufB = (e & 1) ? *reinterpret_cast<volatile int*>(blockSecondBuf + b)
                                : *reinterpret_cast<volatile int*>(blockBestBuf + b);
            }
        }
    }
    __syncthreads();

    const int wb = sBlockA;
    if (wb >= 0) {
        const int wbuf = *reinterpret_cast<volatile int*>(blockBestBuf + wb);
        const int card = packedCard(best);
        const int* src = memberBufs + (static_cast<size_t>(wb) * kNumBufs + wbuf) * U;
        for (int i = tid; i < card; i += kBlockSize) {
            winnerMembers[i] = *reinterpret_cast<const volatile int*>(src + i);
        }
    }
    const int sb = sBlockB;
    if (sb >= 0) {
        const int card = packedCard(second);
        const int* src = memberBufs + (static_cast<size_t>(sb) * kNumBufs + sBufB) * U;
        for (int i = tid; i < card; i += kBlockSize) {
            secondMembers[i] = *reinterpret_cast<const volatile int*>(src + i);
        }
    }
    if (tid == 0) {
        if (best == 0ull) {
            out[0] = 0;
            out[1] = -1;
        } else {
            out[0] = (wb >= 0) ? packedCard(best) : -packedCard(best);
            out[1] = packedSeed(best, U);
        }
        if (second != 0ull && sb >= 0) {
            out[2] = packedCard(second);
            out[3] = packedSeed(second, U);
        } else {
            out[2] = 0;
            out[3] = -1;
        }
    }
}

// Smallest double T such that sqrt(T) >= threshold, so that for any
// non-negative double d2:  sqrt(d2) < threshold  <=>  d2 < T.
// (sqrt is monotonic and correctly rounded on both host and device.)
double squaredThresholdBound(const double threshold) {
    double x = threshold * threshold;
    if (std::sqrt(x) >= threshold) {
        while (true) {
            const double y = std::nextafter(x, 0.0);
            if (std::sqrt(y) >= threshold) x = y; else break;
        }
    } else {
        while (std::sqrt(x) < threshold) x = std::nextafter(x, INFINITY);
    }
    return x;
}

}  // namespace

// GPU resources for clustering up to `capacity` points: stream, device
// buffers and pinned host buffers. Created once, before the timed region.
struct GpuContext {
    int capacity = 0;
    int numSMs = 0;
    bool useSmem = false;
    int maxGrid = 0;
    size_t outInts = 0;
    cudaStream_t stream = nullptr;
    char* dBase = nullptr;
    char* hBase = nullptr;
    char* dUpload = nullptr;
    char* hUpload = nullptr;
    int* dCounters = nullptr;  // [0]: work, [1]: done, [2..3]: gBest
    int* dSeedOrder = nullptr;
    unsigned long long* dGBest = nullptr;
    int* dOut = nullptr;
    unsigned long long* dBlockBest = nullptr;
    unsigned long long* dBlockSecond = nullptr;
    int* dBlockBestBuf = nullptr;
    int* dBlockSecondBuf = nullptr;
    int* dMemberBufs = nullptr;
    double* dScratch = nullptr;
    int* hCounters = nullptr;
    int* hSeedOrder = nullptr;
    int* hOut = nullptr;
    int* hMembers = nullptr;
};

GpuContext createGpuContext(const int N) {
    GpuContext ctx;
    ctx.capacity = N;

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    ctx.numSMs = prop.multiProcessorCount;

    ctx.useSmem = static_cast<size_t>(N) * sizeof(double) <= kMaxSmemBytes;
    if (ctx.useSmem) {
        CUDA_CHECK(cudaFuncSetAttribute(growClustersKernel<true>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(kMaxSmemBytes)));
    }

    // Persistent grid: at most the number of co-resident blocks.
    int maxGrid = std::min(N, kMaxBlocksPerSM * ctx.numSMs);
    if (!ctx.useSmem) {
        // Global-scratch fallback for very large point counts: bound the
        // scratch to ~1 GiB.
        const size_t budget = static_cast<size_t>(1) << 30;
        const size_t perBlock = static_cast<size_t>(N) * sizeof(double);
        maxGrid = static_cast<int>(std::min<size_t>(maxGrid, std::max<size_t>(1, budget / perBlock)));
    }
    ctx.maxGrid = maxGrid;

    // Output layout (ints): header, per-slot cards, winner members,
    // runner-up members, affected flags.
    const size_t outInts = static_cast<size_t>(kInfoInts) + 4 * static_cast<size_t>(N);
    ctx.outInts = outInts;

    // Per-round upload: [counters (4 ints)] [seedOrder (numSeeds ints, padded)]
    // [pts (U double2)], contiguous so that a single copy suffices.
    const size_t uploadBytes = 4 * sizeof(int) + static_cast<size_t>(N + 4) * sizeof(int) +
                               static_cast<size_t>(N) * sizeof(double2) + 16;

    // Single device allocation carved into the individual buffers.
    auto alignUp = [](size_t x) { return (x + 255) & ~static_cast<size_t>(255); };
    const size_t offUpload = 0;
    const size_t offOut = alignUp(offUpload + uploadBytes);
    const size_t offBlockBest = alignUp(offOut + outInts * sizeof(int));
    const size_t offBlockSecond = alignUp(offBlockBest + static_cast<size_t>(maxGrid) * sizeof(unsigned long long));
    const size_t offBlockBestBuf = alignUp(offBlockSecond + static_cast<size_t>(maxGrid) * sizeof(unsigned long long));
    const size_t offBlockSecondBuf = alignUp(offBlockBestBuf + static_cast<size_t>(maxGrid) * sizeof(int));
    const size_t offMemberBufs = alignUp(offBlockSecondBuf + static_cast<size_t>(maxGrid) * sizeof(int));
    const size_t offScratch = alignUp(offMemberBufs + static_cast<size_t>(maxGrid) * kNumBufs * N * sizeof(int));
    const size_t scratchBytes = ctx.useSmem ? 0 : static_cast<size_t>(maxGrid) * N * sizeof(double);
    const size_t deviceBytes = offScratch + scratchBytes;

    CUDA_CHECK(cudaMalloc(&ctx.dBase, deviceBytes));
    ctx.dUpload = ctx.dBase + offUpload;
    ctx.dCounters = reinterpret_cast<int*>(ctx.dUpload);
    ctx.dSeedOrder = ctx.dCounters + 4;
    ctx.dGBest = reinterpret_cast<unsigned long long*>(ctx.dCounters + 2);
    ctx.dOut = reinterpret_cast<int*>(ctx.dBase + offOut);
    ctx.dBlockBest = reinterpret_cast<unsigned long long*>(ctx.dBase + offBlockBest);
    ctx.dBlockSecond = reinterpret_cast<unsigned long long*>(ctx.dBase + offBlockSecond);
    ctx.dBlockBestBuf = reinterpret_cast<int*>(ctx.dBase + offBlockBestBuf);
    ctx.dBlockSecondBuf = reinterpret_cast<int*>(ctx.dBase + offBlockSecondBuf);
    ctx.dMemberBufs = reinterpret_cast<int*>(ctx.dBase + offMemberBufs);
    ctx.dScratch = ctx.useSmem ? nullptr : reinterpret_cast<double*>(ctx.dBase + offScratch);

    // Single pinned host allocation.
    const size_t hOffUpload = 0;
    const size_t hOffOut = alignUp(hOffUpload + uploadBytes);
    const size_t hOffMembers = alignUp(hOffOut + outInts * sizeof(int));
    const size_t hostBytes = hOffMembers + static_cast<size_t>(kInfoInts + N) * sizeof(int);
    CUDA_CHECK(cudaMallocHost(&ctx.hBase, hostBytes));
    ctx.hUpload = ctx.hBase + hOffUpload;
    ctx.hCounters = reinterpret_cast<int*>(ctx.hUpload);
    ctx.hSeedOrder = ctx.hCounters + 4;
    ctx.hOut = reinterpret_cast<int*>(ctx.hBase + hOffOut);
    ctx.hMembers = reinterpret_cast<int*>(ctx.hBase + hOffMembers);

    CUDA_CHECK(cudaStreamCreate(&ctx.stream));
    return ctx;
}

void destroyGpuContext(GpuContext& ctx) {
    if (ctx.stream) CUDA_CHECK(cudaStreamDestroy(ctx.stream));
    if (ctx.hBase) CUDA_CHECK(cudaFreeHost(ctx.hBase));
    if (ctx.dBase) CUDA_CHECK(cudaFree(ctx.dBase));
    ctx = GpuContext();
}

// Main QT clustering algorithm (CUDA)
//
// Round structure (identical decisions to the sequential algorithm):
//   1. every unclustered seed has a candidate cluster; the first seed (in
//      index order) with the largest cardinality wins and is removed.
//   2. A candidate cluster can only change through the removal if one of the
//      removed points lies within the threshold of the seed (otherwise the
//      removed points were never admissible for that seed). Results and
//      pruning bounds of unaffected seeds are therefore carried over, and
//      only affected / undecided seeds are recomputed on the GPU.
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold,
                                  const GpuContext& ctx) {
    const int N = static_cast<int>(points.size());
    std::vector<Cluster> clusters;
    if (N == 0) return clusters;
    if (N > ctx.capacity) {
        fprintf(stderr, "GPU context too small (%d < %d)\n", ctx.capacity, N);
        exit(EXIT_FAILURE);
    }

    const double T = squaredThresholdBound(threshold);

    const int numSMs = ctx.numSMs;
    const bool useSmem = ctx.useSmem;
    const int maxGrid = ctx.maxGrid;
    const size_t outInts = ctx.outInts;
    const int flagsOffset = kInfoInts + 3 * N;
    cudaStream_t stream = ctx.stream;
    char* const dUpload = ctx.dUpload;
    char* const hUpload = ctx.hUpload;
    int* const dCounters = ctx.dCounters;
    int* const dSeedOrder = ctx.dSeedOrder;
    unsigned long long* const dGBest = ctx.dGBest;
    int* const dOut = ctx.dOut;
    unsigned long long* const dBlockBest = ctx.dBlockBest;
    unsigned long long* const dBlockSecond = ctx.dBlockSecond;
    int* const dBlockBestBuf = ctx.dBlockBestBuf;
    int* const dBlockSecondBuf = ctx.dBlockSecondBuf;
    int* const dMemberBufs = ctx.dMemberBufs;
    double* const dScratch = ctx.dScratch;
    int* const hCounters = ctx.hCounters;
    int* const hSeedOrder = ctx.hSeedOrder;
    int* const hOut = ctx.hOut;
    int* const hMembers = ctx.hMembers;

    // Unclustered point indices, always kept in ascending order so that the
    // compacted index order equals the original index order (tie-breaking).
    std::vector<int> active(N);
    for (int i = 0; i < N; ++i) active[i] = i;
    std::vector<char> clustered(N, 0);
    // Per original point: > 0 exact cardinality of its candidate cluster,
    // < 0 negated valid upper bound (pruned), 0 unknown.
    std::vector<int> result(N, 0);
    // Scheduling key per point: estimated cluster size (larger first).
    std::vector<int> key(N, 0);
    // Cached member lists (original indices) of some seeds with exact results.
    std::vector<std::vector<int>> cache(N);
    std::vector<int> pos(N, -1);  // original index -> compacted index
    std::vector<int> work;        // compacted indices of the seeds to compute
    work.reserve(N);

    bool firstRound = true;

    // Main clustering loop
    while (!active.empty()) {
        const int U = static_cast<int>(active.size());

        if (firstRound) {
            // Neighbour counts as the initial scheduling heuristic.
            double2* hPts0 = reinterpret_cast<double2*>(hSeedOrder + 4);
            double2* dPts0 = reinterpret_cast<double2*>(dSeedOrder + 4);
            for (int i = 0; i < U; ++i) {
                hPts0[i] = make_double2(points[active[i]].x, points[active[i]].y);
            }
            CUDA_CHECK(cudaMemcpyAsync(dPts0, hPts0, static_cast<size_t>(U) * sizeof(double2),
                                       cudaMemcpyHostToDevice, stream));
            neighborCountKernel<<<U, kBlockSize, 0, stream>>>(dPts0, U, T, dOut + kInfoInts);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpyAsync(hOut + kInfoInts, dOut + kInfoInts,
                                       static_cast<size_t>(U) * sizeof(int),
                                       cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            for (int i = 0; i < U; ++i) key[active[i]] = hOut[kInfoInts + i];
            firstRound = false;
        }

        // Best already-known exact result (first seed in index order among
        // the largest cardinalities).
        int knownSeed = -1;
        int knownCard = 0;
        for (int i = 0; i < U; ++i) {
            const int r = result[active[i]];
            if (r > knownCard) {
                knownCard = r;
                knownSeed = i;
            }
        }
        const unsigned long long knownPacked =
            (knownSeed >= 0) ? packBest(knownCard, knownSeed, U) : 0ull;
        const bool knownCached = (knownSeed >= 0) && !cache[active[knownSeed]].empty();

        // Seeds that have to be (re)computed this round.
        work.clear();
        for (int i = 0; i < U; ++i) {
            const int r = result[active[i]];
            if (r == 0) {
                work.push_back(i);
            } else if (r < 0) {
                if (!(packBest(-r, i, U) < knownPacked)) work.push_back(i);
            } else if (i == knownSeed && !knownCached) {
                work.push_back(i);
            }
        }
        const int numSeeds = static_cast<int>(work.size());
        // pts follow the (padded) seed list in the upload buffer.
        const int seedInts = (numSeeds + 3) & ~3;  // keep double2 16-byte aligned
        double2* hPts = reinterpret_cast<double2*>(hSeedOrder + seedInts);
        double2* dPts = reinterpret_cast<double2*>(dSeedOrder + seedInts);
        for (int i = 0; i < U; ++i) {
            hPts[i] = make_double2(points[active[i]].x, points[active[i]].y);
            pos[active[i]] = i;
        }

        // Process seeds in order of decreasing estimated size (stable, so
        // equal keys keep ascending index order). Correctness does not
        // depend on this order; it only makes the pruning effective.
        std::stable_sort(work.begin(), work.end(),
                         [&](int a, int b) { return key[active[a]] > key[active[b]]; });

        int winnerCard = 0;
        int winnerSeed = -1;
        const int* winnerMembers = nullptr;  // compacted indices
        bool membersOnDevice = false;

        if (numSeeds > 0) {
            for (int i = 0; i < numSeeds; ++i) hSeedOrder[i] = work[i];
            hCounters[0] = 0;
            hCounters[1] = 0;
            std::memcpy(hCounters + 2, &knownPacked, sizeof(knownPacked));
            CUDA_CHECK(cudaMemcpyAsync(dUpload, hUpload,
                                       reinterpret_cast<char*>(hPts + U) - hUpload,
                                       cudaMemcpyHostToDevice, stream));

            const size_t smemBytes = useSmem ? static_cast<size_t>(U) * sizeof(double) : 0;
            int grid;
            if (useSmem) {
                int blocksPerSM = 0;
                CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                    &blocksPerSM, growClustersKernel<true>, kBlockSize, smemBytes));
                grid = std::min({numSeeds, maxGrid, std::max(1, blocksPerSM) * numSMs});
                growClustersKernel<true><<<grid, kBlockSize, smemBytes, stream>>>(
                    dPts, U, T, dSeedOrder, numSeeds, dOut, N, dGBest, dCounters, dCounters + 1,
                    dBlockBest, dBlockSecond, dBlockBestBuf, dBlockSecondBuf, dMemberBufs, nullptr);
            } else {
                grid = std::min(numSeeds, maxGrid);
                growClustersKernel<false><<<grid, kBlockSize, 0, stream>>>(
                    dPts, U, T, dSeedOrder, numSeeds, dOut, N, dGBest, dCounters, dCounters + 1,
                    dBlockBest, dBlockSecond, dBlockBestBuf, dBlockSecondBuf, dMemberBufs, dScratch);
            }
            CUDA_CHECK(cudaGetLastError());
            // Affected flags for the kernel-determined winner (no-op if the
            // winner's members are not on the device).
            affectedKernel<<<(U + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
                dPts, U, T, dOut, N, dOut + flagsOffset);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpyAsync(hOut, dOut, outInts * sizeof(int),
                                       cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            for (int j = 0; j < numSeeds; ++j) {
                const int r = hOut[kInfoInts + j];
                const int orig = active[work[j]];
                result[orig] = r;
                key[orig] = std::abs(r);
            }
            winnerCard = std::abs(hOut[0]);
            winnerSeed = hOut[1];
            if (hOut[0] > 0) {
                winnerMembers = hOut + kInfoInts + N;
                membersOnDevice = true;
            }
            // Cache the runner-up's members for later rounds.
            if (hOut[2] > 0) {
                const int orig = active[hOut[3]];
                std::vector<int>& m = cache[orig];
                m.resize(hOut[2]);
                for (int i = 0; i < hOut[2]; ++i) m[i] = active[hOut[kInfoInts + 2 * N + i]];
            }
        } else {
            winnerCard = knownCard;
            winnerSeed = knownSeed;
        }

        if (winnerSeed < 0 || winnerCard <= 0) break;  // cannot happen (seed alone forms a cluster)

        if (!membersOnDevice) {
            // Winner is the host-known seed: use its cached member list.
            if (numSeeds == 0) {
                CUDA_CHECK(cudaMemcpyAsync(dPts, hPts, static_cast<size_t>(U) * sizeof(double2),
                                           cudaMemcpyHostToDevice, stream));
            }
            const std::vector<int>& m = cache[active[winnerSeed]];
            if (static_cast<int>(m.size()) != winnerCard) {
                fprintf(stderr, "Internal error: missing member list for seed %d\n",
                        active[winnerSeed]);
                exit(EXIT_FAILURE);
            }
            hMembers[0] = winnerCard;
            hMembers[1] = winnerSeed;
            hMembers[2] = 0;
            hMembers[3] = -1;
            for (int i = 0; i < winnerCard; ++i) hMembers[kInfoInts + i] = pos[m[i]];
            CUDA_CHECK(cudaMemcpyAsync(dOut, hMembers, kInfoInts * sizeof(int),
                                       cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(dOut + kInfoInts + N, hMembers + kInfoInts,
                                       static_cast<size_t>(winnerCard) * sizeof(int),
                                       cudaMemcpyHostToDevice, stream));
            affectedKernel<<<(U + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
                dPts, U, T, dOut, N, dOut + flagsOffset);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpyAsync(hOut + flagsOffset, dOut + flagsOffset,
                                       static_cast<size_t>(U) * sizeof(int),
                                       cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            winnerMembers = hMembers + kInfoInts;
        }

        Cluster cluster;
        cluster.seed_point = active[winnerSeed];
        cluster.members.resize(winnerCard);
        for (int i = 0; i < winnerCard; ++i) {
            const int idx = active[winnerMembers[i]];
            cluster.members[i] = idx;
            clustered[idx] = 1;
        }
        clusters.push_back(std::move(cluster));

        // Invalidate results of seeds affected by the removal; remove the
        // clustered points from the active list (order preserved).
        const int* flags = hOut + flagsOffset;
        for (int i = 0; i < U; ++i) {
            const int orig = active[i];
            if (clustered[orig]) {
                cache[orig].clear();
                cache[orig].shrink_to_fit();
            } else if (flags[i]) {
                result[orig] = 0;
                cache[orig].clear();
            }
        }
        active.erase(std::remove_if(active.begin(), active.end(),
                                    [&clustered](int idx) { return clustered[idx] != 0; }),
                     active.end());
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
    
    // Initialise the CUDA context up front so that one-time driver setup is
    // not attributed to the clustering time.
    {
        int device = 0;
        CUDA_CHECK(cudaSetDevice(device));
        CUDA_CHECK(cudaFree(nullptr));
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
        printf("CUDA device: %s (%d SMs)\n", prop.name, prop.multiProcessorCount);
        // Force the kernels to be loaded now (CUDA 12 loads modules lazily).
        cudaFuncAttributes attr;
        CUDA_CHECK(cudaFuncGetAttributes(&attr, growClustersKernel<true>));
        CUDA_CHECK(cudaFuncGetAttributes(&attr, growClustersKernel<false>));
        CUDA_CHECK(cudaFuncGetAttributes(&attr, neighborCountKernel));
        CUDA_CHECK(cudaFuncGetAttributes(&attr, affectedKernel));
    }
    GpuContext gpu = createGpuContext(num_points);
    
    // Perform QT clustering
    auto cluster_start = std::chrono::high_resolution_clock::now();
    
    const std::vector<Cluster> clusters = qtClustering(points, threshold, gpu);
    
    auto cluster_end = std::chrono::high_resolution_clock::now();
    destroyGpuContext(gpu);
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
