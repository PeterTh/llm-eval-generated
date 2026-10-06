// QT Clustering Benchmark - Hybrid MPI + OpenMP + CUDA Version
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.

#include <algorithm>
#include <climits>
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
// Parallel implementation (MPI + OpenMP + CUDA)
//
// The candidate-cluster construction of the reference code is reproduced
// exactly, but evaluated far more efficiently:
//  * For every candidate the maximum distance to the current members is kept
//    and updated incrementally when a new member joins (identical max values).
//  * Only points closer than the threshold to the seed can ever join, so the
//    candidate set is restricted to the seed's neighbourhood (uniform grid).
//    Candidates whose max distance reaches the threshold are dropped for good.
//  * Removing points that are not members of a seed's candidate cluster does
//    not change it (they never win a greedy step), so cardinalities are cached
//    and a seed is re-evaluated only when one of its members was removed.
//    Upper bounds of seeds that were not fully evaluated stay valid as well
//    (candidate sets only shrink), so the best seed is found by evaluating
//    only seeds whose bound could beat the best known cardinality.
//  * Once the best cardinality is 1 every remaining point is isolated and the
//    remaining clusters are singletons taken in index order.
// Seeds are evaluated on the GPUs (one thread block per seed). Large batches
// are distributed cyclically over the MPI ranks (one GPU per rank) and the
// results combined with MPI_Allgatherv; small batches are evaluated by every
// rank to avoid communication latency. Member lists stay on the GPU of the
// rank that evaluated the seed. Host-side bookkeeping uses OpenMP.
// Selection order/tie-breaking matches the sequential code exactly.
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err_ = (call);                                               \
        if (err_ != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), \
                    __FILE__, __LINE__);                                         \
            MPI_Abort(MPI_COMM_WORLD, 1);                                        \
        }                                                                        \
    } while (0)

// Squared distance evaluated exactly like the reference build computes
// dx * dx + dy * dy (contracted to fma(dx, dx, dy * dy)).
__host__ __device__ inline double dist2Exact(double ax, double ay, double bx, double by) {
    const double dx = ax - bx;
    const double dy = ay - by;
#ifdef __CUDA_ARCH__
    return __fma_rn(dx, dx, __dmul_rn(dy, dy));
#else
    return std::fma(dx, dx, dy * dy);
#endif
}

static constexpr int QT_BLOCK = 256;         // threads per seed (large candidate lists)
static constexpr int QT_BLOCK_SMALL = 128;   // threads per seed (small candidate lists)
static constexpr int QT_SMALL_CAND = 1024;   // candidate-count limit of the small variant
static constexpr int QT_BLOCK_BIG = 512;     // threads per seed (lists exceeding shared memory)
static constexpr int QT_BLOCK_SINGLE = 1024; // threads for a single seed (fallback path)
// Per-candidate storage: max dist^2 (double bits), x, y (double),
// prefilter x/y/limit (float), original index (int)
static constexpr size_t QT_CAND_BYTES = 3 * sizeof(double) + 3 * sizeof(float) + sizeof(int);
// Safety margin of the single-precision prefilter (in units of threshold^2);
// the prefilter error is below 1e-5, so no exact update can be missed.
static constexpr float QT_FMARGIN = 1e-4f;
// Square-window bound: candidates are binned into QT_HB x QT_HB bins over
// [-L, L)^2 around the seed (L = threshold * (1 + 1e-9)).
static constexpr int QT_HB = 32;
// Shared memory reserved for the kernels' static shared variables
static constexpr size_t QT_STATIC_SMEM = 8192;

struct DevPoints {
    const double* px;        // x by sorted position
    const double* py;        // y by sorted position
    const int* order;        // sorted position -> original index
    const int* cellOf;       // sorted position -> cell id
    const int* cellStart;    // cell id -> first sorted position (ncell + 1)
    const unsigned char* clustered; // by sorted position
    int ncx;                 // cells per dimension
    double threshold;
    double invT;             // 1 / threshold
    double t2crit;           // smallest d2 with sqrt(d2) >= threshold
};

// (key, id) lexicographic minimum that also tracks the smallest key strictly
// greater than the minimum (needed to detect distinct d2 with equal sqrt).
// Keys are bit patterns of non-negative doubles (integer order == fp order).
struct ArgMin {
    long long k, k2;
    int id, p;
};

__device__ __forceinline__ void argMinCombine(ArgMin& a, const long long bk, const long long bk2,
                                              const int bid, const int bp) {
    if (bk < a.k) {
        a.k2 = min(a.k, bk2);
        a.k = bk; a.id = bid; a.p = bp;
    } else if (bk > a.k) {
        a.k2 = min(a.k2, bk);
    } else {
        a.k2 = min(a.k2, bk2);
        if (bid < a.id) { a.id = bid; a.p = bp; }
    }
}

// Structure-of-arrays candidate storage carved from a byte buffer
struct CandArrays {
    long long* md2; // bits of the max squared distance to the members (+inf: gone)
    double* cx;     // exact coordinates
    double* cy;
    float* fx;      // seed-relative coordinates / threshold (prefilter)
    float* fy;
    float* flim;    // prefilter limit: max dist^2 / threshold^2 - margin
    int* cid;       // original index

    __device__ __forceinline__ static CandArrays carve(unsigned char* base, const int n) {
        CandArrays a;
        a.md2 = reinterpret_cast<long long*>(base);
        a.cx = reinterpret_cast<double*>(a.md2 + n);
        a.cy = a.cx + n;
        a.fx = reinterpret_cast<float*>(a.cy + n);
        a.fy = a.fx + n;
        a.flim = a.fy + n;
        a.cid = reinterpret_cast<int*>(a.flim + n);
        return a;
    }
};

// One thread block builds the candidate cluster of one seed (persistent loop
// over a dynamically fetched seed list), reproducing the sequential greedy
// growth exactly: in every step the candidate with the smallest maximum
// distance to the members (lowest index on ties) joins, candidates whose
// maximum distance reaches the threshold are dropped. Each warp owns a
// contiguous segment of the candidate list which it compacts as candidates
// die. Maximum distances are tracked as exact squared distances (sqrt is
// monotone); a single-precision prefilter skips updates that provably cannot
// raise a candidate's maximum.
//
// Storage: without SPILL all candidates live in shared memory ('cap' slots).
// With SPILL the first 'quota' candidates of every warp segment live in shared
// memory and the rest in this block's global scratch slot ('cap' slots);
// compaction moves the surviving candidates into shared memory over time.
// Candidate locations: loc < capS -> shared index, otherwise global loc - capS.
//
// If 'bound' is given, seeds whose candidate count cannot reach the best
// cardinality found so far (max of bound0 and *bound) are skipped and
// an upper bound (1 + candidates, or the square-window bound below) is stored
// negated instead of the cardinality.
// Members (original indices, insertion order) are written to
// memStore + memOff[seed] if memStore is given, otherwise to 'members'.
template <bool SPILL, int BS>
__global__ void __launch_bounds__(BS)
qtSeedKernel(const DevPoints d, const int* __restrict__ seeds, const int nseeds,
             int* __restrict__ counter, int* __restrict__ bound, const int bound0,
             int* __restrict__ cards,
             int* __restrict__ memStore, const long long* __restrict__ memOff,
             int* __restrict__ members, const int cap, const int quota,
             unsigned char* __restrict__ gscratch) {
    constexpr int NW = BS / 32;
    constexpr long long KINF = 0x7ff0000000000000LL; // +inf
    extern __shared__ __align__(16) unsigned char smem[];
    const int capS = SPILL ? NW * quota : cap;
    const CandArrays S = CandArrays::carve(smem, capS);
    const CandArrays G = SPILL ? CandArrays::carve(gscratch + static_cast<size_t>(blockIdx.x) * cap * QT_CAND_BYTES, cap)
                               : S;
    // Gather target: global scratch with SPILL, shared memory otherwise
    const CandArrays& A = SPILL ? G : S;

    __shared__ int s_seed, s_m, s_best, s_tie;
    __shared__ double s_wx, s_wy, s_tieS;
    __shared__ float s_wfx, s_wfy;
    __shared__ long long s_wk[NW], s_wk2[NW], s_tieK;
    __shared__ unsigned long long s_tiePack;
    __shared__ int s_wi[NW], s_wp[NW];
    __shared__ int s_hist[(QT_HB + 1) * (QT_HB + 1)];
    __shared__ int s_ub, s_bound;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const unsigned lanesBelow = (1u << lane) - 1u;
    const long long t2crit = __double_as_longlong(d.t2crit);
    const double invT = d.invT;
    const double invT2 = invT * invT;

    // Field access by candidate location
#define QT_AT(field, loc) (SPILL && (loc) >= capS ? G.field + ((loc) - capS) : S.field + (loc))

    for (;;) {
        if (tid == 0) {
            s_seed = atomicAdd(counter, 1);
            s_m = 0;
            // Block-uniform pruning bound
            s_bound = bound ? max(bound0, *reinterpret_cast<volatile int*>(bound)) : 0;
        }
        __syncthreads();
        const int w = s_seed;
        if (w >= nseeds) break;
        const int sp = seeds[w];
        const double sx = d.px[sp];
        const double sy = d.py[sp];
        const int c = d.cellOf[sp];
        const int ccx = c % d.ncx;
        const int ccy = c / d.ncx;

        // Gather candidates: unclustered points with distance < threshold
        for (int oy = -1; oy <= 1; ++oy) {
            const int yy = ccy + oy;
            if (yy < 0 || yy >= d.ncx) continue;
            for (int ox = -1; ox <= 1; ++ox) {
                const int xx = ccx + ox;
                if (xx < 0 || xx >= d.ncx) continue;
                const int cell = yy * d.ncx + xx;
                const int end = d.cellStart[cell + 1];
                for (int j = d.cellStart[cell] + tid; j < end; j += BS) {
                    if (j == sp || d.clustered[j]) continue;
                    const double qx = d.px[j], qy = d.py[j];
                    const double d2 = dist2Exact(qx, qy, sx, sy);
                    if (__double_as_longlong(d2) < t2crit) {
                        const int k = atomicAdd(&s_m, 1);
                        A.md2[k] = __double_as_longlong(d2);
                        A.cx[k] = qx;
                        A.cy[k] = qy;
                        A.fx[k] = static_cast<float>((qx - sx) * invT);
                        A.fy[k] = static_cast<float>((qy - sy) * invT);
                        A.flim[k] = static_cast<float>(d2 * invT2) - QT_FMARGIN;
                        A.cid[k] = d.order[j];
                    }
                }
            }
        }
        __syncthreads();
        const int m = s_m;
        int ub = 1 + m; // upper bound of the cardinality
        bool pruned = false;
        if (bound != nullptr) {
            const int best = s_bound;
            pruned = (ub < best);
            if (!pruned && m >= 64) {
                // All pairwise distances of a cluster are below the threshold,
                // so the cluster lies in an axis-aligned square of side L that
                // contains the seed. Bin the candidates (bin width L / (HB/2))
                // and take the largest count of a window of HB/2 + 1 bins
                // (+1 bin on each side against rounding) around the seed.
                constexpr int H = QT_HB / 2;
                constexpr int P = QT_HB + 1; // prefix-sum row length
                const double L = d.threshold * (1.0 + 1e-9);
                const double invW = H / L;
                for (int i = tid; i < P * P; i += BS) s_hist[i] = 0;
                if (tid == 0) s_ub = 0;
                __syncthreads();
                for (int k = tid; k < m; k += BS) {
                    const int bx = min(QT_HB - 1, max(0, static_cast<int>(floor((A.cx[k] - sx + L) * invW))));
                    const int by = min(QT_HB - 1, max(0, static_cast<int>(floor((A.cy[k] - sy + L) * invW))));
                    atomicAdd(&s_hist[(by + 1) * P + bx + 1], 1);
                }
                __syncthreads();
                // 2D inclusive prefix sums (row 0 / column 0 are zero)
                if (tid < QT_HB) {
                    int acc = 0;
                    for (int x = 1; x < P; ++x) { acc += s_hist[(tid + 1) * P + x]; s_hist[(tid + 1) * P + x] = acc; }
                }
                __syncthreads();
                if (tid < QT_HB) {
                    int acc = 0;
                    for (int y = 1; y < P; ++y) { acc += s_hist[y * P + tid + 1]; s_hist[y * P + tid + 1] = acc; }
                }
                __syncthreads();
                // Windows [i0 - 1, i0 + H + 1) with i0 in [0, H]
                int localMax = 0;
                for (int wdx = tid; wdx < (H + 1) * (H + 1); wdx += BS) {
                    const int x0 = max(0, wdx % (H + 1) - 1), x1 = min(QT_HB, wdx % (H + 1) + H + 1);
                    const int y0 = max(0, wdx / (H + 1) - 1), y1 = min(QT_HB, wdx / (H + 1) + H + 1);
                    const int cnt = s_hist[y1 * P + x1] - s_hist[y0 * P + x1] - s_hist[y1 * P + x0] +
                                    s_hist[y0 * P + x0];
                    localMax = max(localMax, cnt);
                }
                atomicMax(&s_ub, localMax);
                __syncthreads();
                ub = min(ub, 1 + s_ub);
                pruned = (ub < best);
            }
        }
        int card = 1;
        if (!pruned) {
            int* mout = memStore ? memStore + memOff[sp] : members;
            if (mout != nullptr && tid == 0) mout[0] = d.order[sp];

            // Warp segment: warp-local indices t in [0, segLen) of the global
            // range starting at segBeg
            const int segBeg = static_cast<int>((static_cast<long long>(warp) * m) / NW);
            int segLen = static_cast<int>((static_cast<long long>(warp + 1) * m) / NW) - segBeg;
            int segAlive = segLen;
            auto locOf = [&](const int t) {
                if constexpr (SPILL) return t < quota ? warp * quota + t : capS + segBeg + t;
                else return segBeg + t;
            };
            if constexpr (SPILL) {
                // Move the head of the segment into shared memory
                for (int t = lane; t < min(segLen, quota); t += 32) {
                    const int gi = segBeg + t, si = warp * quota + t;
                    S.md2[si] = G.md2[gi]; S.cx[si] = G.cx[gi]; S.cy[si] = G.cy[gi];
                    S.fx[si] = G.fx[gi]; S.fy[si] = G.fy[gi]; S.flim[si] = G.flim[gi];
                    S.cid[si] = G.cid[gi];
                }
                __syncwarp();
            }

            ArgMin am{KINF, KINF, INT_MAX, -1};
            for (int t = lane; t < segLen; t += 32) {
                const int loc = locOf(t);
                argMinCombine(am, *QT_AT(md2, loc), KINF, *QT_AT(cid, loc), loc);
            }

            for (;;) {
                // Block-wide argmin of (max dist^2, index)
#pragma unroll
                for (int off = 16; off > 0; off >>= 1) {
                    const long long ok = __shfl_down_sync(0xffffffffu, am.k, off);
                    const long long ok2 = __shfl_down_sync(0xffffffffu, am.k2, off);
                    const int oi = __shfl_down_sync(0xffffffffu, am.id, off);
                    const int op = __shfl_down_sync(0xffffffffu, am.p, off);
                    argMinCombine(am, ok, ok2, oi, op);
                }
                if (lane == 0) { s_wk[warp] = am.k; s_wk2[warp] = am.k2; s_wi[warp] = am.id; s_wp[warp] = am.p; }
                __syncthreads();
                if (tid == 0) {
                    ArgMin r{s_wk[0], s_wk2[0], s_wi[0], s_wp[0]};
#pragma unroll
                    for (int k = 1; k < NW; ++k) argMinCombine(r, s_wk[k], s_wk2[k], s_wi[k], s_wp[k]);
                    // Distinct squared distances may round to the same distance:
                    // the sequential code then prefers the lowest index. Such
                    // values are at most a few ulps apart.
                    int tie = 0;
                    if (r.p >= 0 && r.k2 - r.k <= 16) {
                        const double s = sqrt(__longlong_as_double(r.k));
                        if (sqrt(__longlong_as_double(r.k2)) == s) {
                            tie = 1;
                            s_tieS = s;
                            s_tieK = r.k;
                            s_tiePack = (static_cast<unsigned long long>(r.id) << 32) |
                                        static_cast<unsigned>(r.p);
                        }
                    }
                    s_tie = tie;
                    s_best = r.p;
                    if (!tie && r.p >= 0) {
                        s_wx = *QT_AT(cx, r.p);
                        s_wy = *QT_AT(cy, r.p);
                        s_wfx = *QT_AT(fx, r.p);
                        s_wfy = *QT_AT(fy, r.p);
                        if (mout != nullptr) mout[card] = r.id;
                    }
                }
                __syncthreads();
                if (s_tie) {
                    // Rare exact tie resolution over all equal-distance candidates
                    const long long kmin = s_tieK;
                    const double s = s_tieS;
                    for (int t = lane; t < segLen; t += 32) {
                        const int loc = locOf(t);
                        const long long k = *QT_AT(md2, loc);
                        if (k > kmin && k < KINF && sqrt(__longlong_as_double(k)) == s)
                            atomicMin(&s_tiePack, (static_cast<unsigned long long>(*QT_AT(cid, loc)) << 32) |
                                                      static_cast<unsigned>(loc));
                    }
                    __syncthreads();
                    if (tid == 0) {
                        const int id = static_cast<int>(s_tiePack >> 32);
                        const int p = static_cast<int>(s_tiePack & 0xffffffffu);
                        s_best = p;
                        s_wx = *QT_AT(cx, p);
                        s_wy = *QT_AT(cy, p);
                        s_wfx = *QT_AT(fx, p);
                        s_wfy = *QT_AT(fy, p);
                        if (mout != nullptr) mout[card] = id;
                    }
                    __syncthreads();
                }
                const int p = s_best;
                if (p < 0) break;
                ++card;
                const double wx = s_wx, wy = s_wy;
                const float wfx = s_wfx, wfy = s_wfy;

                // Update max distances with the new member, drop candidates that
                // can no longer join, compute the local argmin
                const bool compact = (segAlive * 4 < segLen * 3);
                am = ArgMin{KINF, KINF, INT_MAX, -1};
                int written = 0;
                for (int b = 0; b < segLen; b += 32) {
                    const int t = b + lane;
                    const int loc = locOf(t);
                    bool alive = false, changed = false;
                    long long k = KINF;
                    float x = 0.f, y = 0.f, lim = 0.f;
                    int id = 0;
                    if (t < segLen) {
                        k = *QT_AT(md2, loc);
                        if (k < KINF && loc != p) {
                            x = *QT_AT(fx, loc); y = *QT_AT(fy, loc); lim = *QT_AT(flim, loc);
                            id = *QT_AT(cid, loc);
                            alive = true;
                            const float dx = x - wfx, dy = y - wfy;
                            if (fmaf(dx, dx, dy * dy) >= lim) {
                                // Possibly a new maximum: evaluate exactly
                                const double d2 = dist2Exact(*QT_AT(cx, loc), *QT_AT(cy, loc), wx, wy);
                                const long long kd = __double_as_longlong(d2);
                                if (kd > k) {
                                    if (kd >= t2crit) {
                                        alive = false;
                                    } else {
                                        k = kd;
                                        lim = static_cast<float>(d2 * invT2) - QT_FMARGIN;
                                        changed = true;
                                    }
                                }
                            }
                        }
                    }
                    const unsigned mask = __ballot_sync(0xffffffffu, alive);
                    int dloc = loc;
                    if (compact) {
                        dloc = locOf(written + __popc(mask & lanesBelow));
                        double ex = 0.0, ey = 0.0;
                        const bool move = alive && dloc != loc;
                        if (move) { ex = *QT_AT(cx, loc); ey = *QT_AT(cy, loc); }
                        __syncwarp();
                        if (move) {
                            *QT_AT(cx, dloc) = ex; *QT_AT(cy, dloc) = ey;
                            *QT_AT(fx, dloc) = x; *QT_AT(fy, dloc) = y; *QT_AT(cid, dloc) = id;
                            *QT_AT(md2, dloc) = k; *QT_AT(flim, dloc) = lim;
                        } else if (changed) {
                            *QT_AT(md2, dloc) = k; *QT_AT(flim, dloc) = lim;
                        }
                    } else if (t < segLen) {
                        if (changed) { *QT_AT(md2, loc) = k; *QT_AT(flim, loc) = lim; }
                        else if (!alive && k < KINF) *QT_AT(md2, loc) = KINF;
                    }
                    written += __popc(mask);
                    if (alive) argMinCombine(am, k, KINF, id, dloc);
                }
                segAlive = written;
                if (compact) segLen = written;
                __syncwarp();
            }
        }
        if (tid == 0) {
            if (pruned) {
                cards[w] = -ub;
            } else {
                cards[w] = card;
                if (bound != nullptr) atomicMax(bound, card);
            }
        }
        __syncthreads();
    }
#undef QT_AT
}

// For each (seed, cardinality) pair: does the stored member list of the seed
// contain a clustered point? One warp per seed.
__global__ void memberCheckKernel(const int* __restrict__ pairs, const int n,
                                  const int* __restrict__ memStore,
                                  const long long* __restrict__ memOff,
                                  const int* __restrict__ pos,
                                  const unsigned char* __restrict__ clustered,
                                  int* __restrict__ flags) {
    const int idx = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (idx >= n) return;
    const int sp = pairs[2 * idx], card = pairs[2 * idx + 1];
    const int* mem = memStore + memOff[sp];
    bool hit = false;
    for (int i = lane; i < card && !hit; i += 32) hit = clustered[pos[mem[i]]] != 0;
    hit = __any_sync(0xffffffffu, hit);
    if (lane == 0) flags[idx] = hit ? 1 : 0;
}

__global__ void markClusteredKernel(const int* __restrict__ members, const int n,
                                    const int* __restrict__ pos,
                                    unsigned char* __restrict__ clustered) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) clustered[pos[members[i]]] = 1;
}

// OpenMP team size scaled to the amount of work (avoids fork/join overhead of
// large teams on the short per-iteration host loops)
[[maybe_unused]] static inline int ompThreads(const long long work, const long long grain = 16384) {
    const long long t = std::max(1LL, work / grain);
    return static_cast<int>(std::min<long long>(t, omp_get_max_threads()));
}

struct MpiCtx {
    int rank = 0;
    int size = 1;
};

// Main QT clustering algorithm
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold,
                                  const MpiCtx& mpi) {
    const int N = static_cast<int>(points.size());
    std::vector<Cluster> clusters;
    if (N == 0) return clusters;

    // ---- Uniform grid with cell size >= threshold (3x3 neighbourhood) ----
    int ncx = static_cast<int>(std::floor(MAX_WIDTH / (threshold * (1.0 + 1e-9))));
    // (larger cells are always valid; no point in more cells than ~points)
    ncx = std::max(1, std::min({ncx, 2048, 2 * static_cast<int>(std::sqrt(static_cast<double>(N))) + 1}));
    const double cs = MAX_WIDTH / ncx;
    const int ncell = ncx * ncx;
    auto cellIdx = [&](const Point& p) {
        int ix = static_cast<int>(p.x / cs);
        int iy = static_cast<int>(p.y / cs);
        ix = std::max(0, std::min(ix, ncx - 1));
        iy = std::max(0, std::min(iy, ncx - 1));
        return iy * ncx + ix;
    };
    std::vector<int> cellOfOrig(N);
    std::vector<int> cellStart(ncell + 1, 0);
#pragma omp parallel for schedule(static) num_threads(ompThreads(N))
    for (int i = 0; i < N; ++i) cellOfOrig[i] = cellIdx(points[i]);
    for (int i = 0; i < N; ++i) cellStart[cellOfOrig[i] + 1]++;
    for (int c = 0; c < ncell; ++c) cellStart[c + 1] += cellStart[c];
    std::vector<int> order(N), pos(N), cellOf(N);
    {
        std::vector<int> fill(cellStart.begin(), cellStart.end() - 1);
        for (int i = 0; i < N; ++i) {
            const int p = fill[cellOfOrig[i]]++;
            order[p] = i;
            pos[i] = p;
        }
    }
    std::vector<double> px(N), py(N);
#pragma omp parallel for schedule(static) num_threads(ompThreads(N))
    for (int p = 0; p < N; ++p) {
        px[p] = points[order[p]].x;
        py[p] = points[order[p]].y;
        cellOf[p] = cellOfOrig[order[p]];
    }

    // Visit all sorted positions q within the 3x3 cells around position p
    auto forNeighbourCells = [&](const int p, auto&& fn) {
        const int c = cellOf[p];
        const int ccx = c % ncx, ccy = c / ncx;
        for (int yy = std::max(0, ccy - 1); yy <= std::min(ncx - 1, ccy + 1); ++yy)
            for (int xx = std::max(0, ccx - 1); xx <= std::min(ncx - 1, ccx + 1); ++xx) {
                const int cell = yy * ncx + xx;
                for (int q = cellStart[cell]; q < cellStart[cell + 1]; ++q) fn(q);
            }
    };

    // Smallest squared distance whose square root reaches the threshold:
    // sqrt(d2) < threshold  <=>  d2 < t2crit  (sqrt is correctly rounded and monotone)
    double t2crit = threshold * threshold;
    while (t2crit > 0.0 && std::sqrt(t2crit) >= threshold) t2crit = std::nextafter(t2crit, 0.0);
    while (std::sqrt(t2crit) < threshold) t2crit = std::nextafter(t2crit, HUGE_VAL);

    // Exact neighbour predicate used by the kernels (distance < threshold)
    auto isNeighbour = [&](const int a, const int b) {
        return dist2Exact(px[a], py[a], px[b], py[b]) < t2crit;
    };

    // Unclustered neighbours per seed = its exact candidate count (card <= nbr + 1)
    std::vector<int> nbr(N);
    int maxNbr = 0;
#pragma omp parallel for schedule(dynamic, 64) reduction(max : maxNbr) num_threads(ompThreads(N, 256))
    for (int p = 0; p < N; ++p) {
        int cnt = 0;
        forNeighbourCells(p, [&](int q) {
            if (q != p && isNeighbour(q, p)) ++cnt;
        });
        nbr[p] = cnt;
        maxNbr = std::max(maxNbr, cnt);
    }
    const int cap = (std::max(1, maxNbr) + 1) & ~1;
    // Member-list slot of every seed (candidate counts only shrink over time)
    std::vector<long long> memOff(N + 1, 0);
    for (int p = 0; p < N; ++p) memOff[p + 1] = memOff[p] + nbr[p] + 1;

    // ---- Device setup ----
    double *d_px, *d_py;
    int *d_order, *d_pos, *d_cellOf, *d_cellStart, *d_seeds, *d_cards, *d_members, *d_counter;
    int *d_check, *d_flags;
    int* d_memStore = nullptr;
    long long* d_memOff = nullptr;
    unsigned char *d_clustered, *d_scratch = nullptr;
    CUDA_CHECK(cudaMalloc(&d_px, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_py, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_order, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_pos, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cellOf, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cellStart, (ncell + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_seeds, (N + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cards, (N + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_members, (N + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_check, 2 * static_cast<size_t>(N + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_flags, (N + 1) * sizeof(int)));
    // Per-iteration slots (zeroed once): 3 work counters, pruning bound, spare
    constexpr int CSLOT = 5;
    CUDA_CHECK(cudaMalloc(&d_counter, static_cast<size_t>(CSLOT) * (N + 1) * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_counter, 0, static_cast<size_t>(CSLOT) * (N + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_clustered, N));
    CUDA_CHECK(cudaMemcpy(d_px, px.data(), N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_py, py.data(), N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_order, order.data(), N * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pos, pos.data(), N * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cellOf, cellOf.data(), N * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cellStart, cellStart.data(), (ncell + 1) * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_clustered, 0, N));

    int dev = 0, numSMs = 1, maxOptinSmem = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&maxOptinSmem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
    // Dynamic shared memory available for candidates (rest: static shared memory)
    const size_t shmLimit = static_cast<size_t>(maxOptinSmem) - QT_STATIC_SMEM;
    const int shmMaxCand = static_cast<int>(shmLimit / QT_CAND_BYTES) & ~1;
    for (const void* fn : {reinterpret_cast<const void*>(qtSeedKernel<false, QT_BLOCK>),
                           reinterpret_cast<const void*>(qtSeedKernel<false, QT_BLOCK_SMALL>),
                           reinterpret_cast<const void*>(qtSeedKernel<false, QT_BLOCK_SINGLE>),
                           reinterpret_cast<const void*>(qtSeedKernel<true, QT_BLOCK_BIG>),
                           reinterpret_cast<const void*>(qtSeedKernel<true, QT_BLOCK_SINGLE>)}) {
        CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(shmLimit)));
    }

    // Seeds with too many candidates for shared memory spill to global scratch
    const int quotaBig = static_cast<int>(shmLimit / ((QT_BLOCK_BIG / 32) * QT_CAND_BYTES)) & ~1;
    const int quotaSingle = static_cast<int>(shmLimit / ((QT_BLOCK_SINGLE / 32) * QT_CAND_BYTES)) & ~1;
    const size_t smemBig = static_cast<size_t>(quotaBig) * (QT_BLOCK_BIG / 32) * QT_CAND_BYTES;
    const size_t smemSingle = static_cast<size_t>(quotaSingle) * (QT_BLOCK_SINGLE / 32) * QT_CAND_BYTES;
    int gridGlobal = 0;
    if (cap > shmMaxCand) {
        int bps = 1;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, qtSeedKernel<true, QT_BLOCK_BIG>,
                                                                 QT_BLOCK_BIG, smemBig));
        gridGlobal = std::max(1, bps * numSMs);
        CUDA_CHECK(cudaMalloc(&d_scratch, static_cast<size_t>(gridGlobal) * cap * QT_CAND_BYTES));
    }

    // Persistent member lists of evaluated seeds (avoids recomputing the winner);
    // falls back to recomputation if the device memory is insufficient.
    {
        size_t freeMem = 0, totalMem = 0;
        CUDA_CHECK(cudaMemGetInfo(&freeMem, &totalMem));
        const size_t need = static_cast<size_t>(memOff[N]) * sizeof(int) + (N + 1) * sizeof(long long);
        if (need < freeMem / 2 && cudaMalloc(&d_memStore, memOff[N] * sizeof(int)) == cudaSuccess) {
            CUDA_CHECK(cudaMalloc(&d_memOff, (N + 1) * sizeof(long long)));
            CUDA_CHECK(cudaMemcpy(d_memOff, memOff.data(), (N + 1) * sizeof(long long),
                                  cudaMemcpyHostToDevice));
        } else {
            d_memStore = nullptr;
            cudaGetLastError();
        }
    }

    const DevPoints dp{d_px, d_py, d_order, d_cellOf, d_cellStart, d_clustered,
                       ncx, threshold, 1.0 / threshold, t2crit};

    cudaStream_t stream, stream2, stream3;
    cudaEvent_t forkEv, joinEv2, joinEv3;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream3, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&forkEv, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&joinEv2, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&joinEv3, cudaEventDisableTiming));

    // Launch the shared-memory kernel variant for seeds [s0, s1) of the batch
    auto launchShm = [&](auto kernel, const int bs, cudaStream_t st, const int s0, const int s1,
                         const int mMaxRaw, int* ctr, int* bnd, const int bound0) {
        const int mMax = (std::max(1, mMaxRaw) + 1) & ~1;
        const size_t smem = static_cast<size_t>(mMax) * QT_CAND_BYTES;
        int bps = 1;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, kernel, bs, smem));
        const int grid = std::min(s1 - s0, std::max(1, bps * numSMs));
        kernel<<<grid, bs, smem, st>>>(dp, d_seeds + s0, s1 - s0, ctr, bnd, bound0, d_cards + s0,
                                       d_memStore, d_memOff, nullptr, mMax, 0, nullptr);
        CUDA_CHECK(cudaGetLastError());
    };

    // Evaluate the seeds in d_seeds[0, n) with pruning against 'bound0'. The
    // batch is grouped by candidate count: [0, nBig) too large for shared
    // memory (spill to global scratch), [nBig, nMid) large (at most maxMid
    // candidates), [nMid, n) small (at most maxSmall); each class is launched
    // on its own stream with shared memory sized to the class.
    auto launchBatch = [&](const int n, const int nBig, const int nMid, const int maxMid,
                           const int maxSmall, const int bound0, int* slot) {
        int* bnd = slot + 3;
        CUDA_CHECK(cudaEventRecord(forkEv, stream));
        if (nBig > 0) {
            CUDA_CHECK(cudaStreamWaitEvent(stream2, forkEv, 0));
            qtSeedKernel<true, QT_BLOCK_BIG><<<std::min(nBig, gridGlobal), QT_BLOCK_BIG, smemBig, stream2>>>(
                dp, d_seeds, nBig, slot + 0, bnd, bound0, d_cards, d_memStore, d_memOff,
                nullptr, cap, quotaBig, d_scratch);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(joinEv2, stream2));
        }
        if (nMid > nBig) {
            CUDA_CHECK(cudaStreamWaitEvent(stream3, forkEv, 0));
            launchShm(qtSeedKernel<false, QT_BLOCK>, QT_BLOCK, stream3, nBig, nMid, maxMid,
                      slot + 1, bnd, bound0);
            CUDA_CHECK(cudaEventRecord(joinEv3, stream3));
        }
        if (n > nMid) {
            launchShm(qtSeedKernel<false, QT_BLOCK_SMALL>, QT_BLOCK_SMALL, stream, nMid, n,
                      maxSmall, slot + 2, bnd, bound0);
        }
        if (nBig > 0) CUDA_CHECK(cudaStreamWaitEvent(stream, joinEv2, 0));
        if (nMid > nBig) CUDA_CHECK(cudaStreamWaitEvent(stream, joinEv3, 0));
    };
    // Build the full candidate cluster of the seed in d_seeds[N] into d_members
    auto launchSingle = [&](const int m, int* ctr) {
        if (m <= shmMaxCand) {
            const int mMax = (std::max(1, m) + 1) & ~1;
            qtSeedKernel<false, QT_BLOCK_SINGLE><<<1, QT_BLOCK_SINGLE, mMax * QT_CAND_BYTES, stream>>>(
                dp, d_seeds + N, 1, ctr, nullptr, 0, d_cards + N, nullptr, nullptr, d_members,
                mMax, 0, nullptr);
        } else {
            qtSeedKernel<true, QT_BLOCK_SINGLE><<<1, QT_BLOCK_SINGLE, smemSingle, stream>>>(
                dp, d_seeds + N, 1, ctr, nullptr, 0, d_cards + N, nullptr, nullptr, d_members,
                cap, quotaSingle, d_scratch);
        }
        CUDA_CHECK(cudaGetLastError());
    };

    // Batches with at least one GPU wave of seeds are distributed over the
    // ranks (agreed on by all ranks)
    int distributeMin = 0;
    {
        int bps = 1;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, qtSeedKernel<false, QT_BLOCK_SMALL>,
                                                                 QT_BLOCK_SMALL, 0));
        distributeMin = std::max(1, bps * numSMs);
        if (mpi.size > 1)
            MPI_Allreduce(MPI_IN_PLACE, &distributeMin, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    }

    // ---- Host state (indexed by sorted position) ----
    // val[p]: exact cardinality if exact[p], otherwise the upper bound nbr[p] + 1
    std::vector<unsigned char> clustered(N, 0), exact(N, 0);
    std::vector<int> val(N), owner(N, -1); // owner: rank holding the member list (-1: all)
    for (int p = 0; p < N; ++p) {
        val[p] = nbr[p] + 1;
        exact[p] = (nbr[p] == 0); // isolated point: cardinality is exactly 1
    }
    // Host staging buffers: seeds | cards | members (small transfers; pageable
    // memory avoids the high cost of pinned allocations)
    std::vector<int> h_buf(3 * static_cast<size_t>(N + 1));
    int* h_seeds = h_buf.data();
    int* h_cards = h_seeds + (N + 1);
    int* h_members = h_seeds + 2 * (N + 1);
    std::vector<int> cand, mcount, allCards, dirtyList, results, checkList, checkFlags, checkPairs;
    std::vector<unsigned char> needCheck;
    std::vector<std::vector<int>> buckets(9);
    std::vector<int> recvCounts(mpi.size), displs(mpi.size);
    cand.reserve(N);
    mcount.reserve(N);

    // (cardinality desc, original index asc) ordering of the sequential code
    auto better = [](const int c1, const int i1, const int c2, const int i2) {
        return c1 > c2 || (c1 == c2 && i1 < i2);
    };
    // Best exact seed among the unclustered points
    auto bestExact = [&](int& bc, int& bi, int& bp) {
        bc = 0; bi = INT_MAX; bp = -1;
#pragma omp parallel num_threads(ompThreads(N, 65536))
        {
            int lc = 0, li = INT_MAX, lp = -1;
#pragma omp for schedule(static) nowait
            for (int p = 0; p < N; ++p) {
                if (clustered[p] || !exact[p]) continue;
                if (better(val[p], order[p], lc, li)) { lc = val[p]; li = order[p]; lp = p; }
            }
#pragma omp critical
            if (lp >= 0 && better(lc, li, bc, bi)) { bc = lc; bi = li; bp = lp; }
        }
    };

    int remaining = N;
    for (int iter = 0; remaining > 0; ++iter) {
        int* slot = d_counter + static_cast<size_t>(CSLOT) * iter;
        int Ec, Eidx, Epos;
        bestExact(Ec, Eidx, Epos);

        // Seeds that might beat the best exact one: evaluate them on the GPUs
        cand.clear();
        for (int p = 0; p < N; ++p)
            if (!clustered[p] && !exact[p] && better(val[p], order[p], Ec, Eidx)) cand.push_back(p);
        const int D = static_cast<int>(cand.size());

        if (D > 0) {
            // Most promising seeds first: raises the pruning bound early
            std::sort(cand.begin(), cand.end(), [&](int a, int b) {
                return better(val[a], order[a], val[b], order[b]);
            });
            // Large batches: cyclic distribution over ranks (balances the sorted
            // work). Small batches: every rank evaluates all seeds itself (no
            // communication). Then grouped by size class keeping the bound
            // order within each class.
            const bool distribute = (mpi.size > 1) && (D >= distributeMin);
            const int rStart = distribute ? mpi.rank : 0, rStep = distribute ? mpi.size : 1;
            int myD = 0, nBig = 0, nMid = 0, maxMid = 0, maxSmall = 0;
            mcount.clear();
            for (int j = rStart; j < D; j += rStep) mcount.push_back(cand[j]);
            for (const int p : mcount)
                if (nbr[p] > shmMaxCand) h_seeds[myD++] = p;
            nBig = myD;
            for (const int p : mcount)
                if (nbr[p] <= shmMaxCand && nbr[p] > QT_SMALL_CAND) {
                    h_seeds[myD++] = p;
                    maxMid = std::max(maxMid, nbr[p]);
                }
            nMid = myD;
            for (const int p : mcount)
                if (nbr[p] <= QT_SMALL_CAND) {
                    h_seeds[myD++] = p;
                    maxSmall = std::max(maxSmall, nbr[p]);
                }
            if (myD > 0) {
                CUDA_CHECK(cudaMemcpyAsync(d_seeds, h_seeds, myD * sizeof(int),
                                           cudaMemcpyHostToDevice, stream));
                launchBatch(myD, nBig, nMid, maxMid, maxSmall, Ec, slot);
                CUDA_CHECK(cudaMemcpyAsync(h_cards, d_cards, myD * sizeof(int),
                                           cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
            }
            // Exchange (seed, result) pairs
            results.resize(2 * static_cast<size_t>(myD));
            for (int j = 0; j < myD; ++j) {
                results[2 * j] = h_seeds[j];
                results[2 * j + 1] = h_cards[j];
            }
            const int* res = results.data();
            int nres = myD;
            if (distribute) {
                for (int r = 0, off = 0; r < mpi.size; ++r) {
                    recvCounts[r] = 2 * ((D > r) ? (D - r + mpi.size - 1) / mpi.size : 0);
                    displs[r] = off;
                    off += recvCounts[r];
                }
                allCards.resize(2 * static_cast<size_t>(D));
                MPI_Allgatherv(results.data(), 2 * myD, MPI_INT, allCards.data(), recvCounts.data(),
                               displs.data(), MPI_INT, MPI_COMM_WORLD);
                res = allCards.data();
                nres = D;
            }
            for (int r = 0, j = 0; j < nres; ++j) {
                if (distribute) {
                    while (2 * j >= displs[r] + recvCounts[r]) ++r;
                } else {
                    r = -1; // available on every rank
                }
                const int p = res[2 * j];
                const int v = res[2 * j + 1];
                exact[p] = (v > 0);
                val[p] = (v > 0) ? v : -v;
                owner[p] = r;
            }
            // Every evaluated seed is now exact or bounded below the best exact one
            bestExact(Ec, Eidx, Epos);
        }
        if (Epos < 0) break;

        if (Ec == 1) {
            // Every remaining point is isolated: singletons in index order
            for (int i = 0; i < N; ++i) {
                if (clustered[pos[i]]) continue;
                Cluster cl;
                cl.seed_point = i;
                cl.members.push_back(i);
                clusters.push_back(std::move(cl));
                clustered[pos[i]] = 1;
            }
            remaining = 0;
            break;
        }

        // Members of the winning candidate cluster
        CUDA_CHECK(cudaStreamSynchronize(stream)); // h_members / d_members are free again
        if (d_memStore != nullptr) {
            // Stored by the rank that evaluated the winner
            const int* src = d_memStore + memOff[Epos];
            const bool local = (owner[Epos] < 0 || owner[Epos] == mpi.rank);
            if (local) {
                markClusteredKernel<<<(Ec + 255) / 256, 256, 0, stream>>>(src, Ec, d_pos, d_clustered);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaMemcpyAsync(h_members, src, Ec * sizeof(int), cudaMemcpyDeviceToHost,
                                           stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
            }
            if (mpi.size > 1 && owner[Epos] >= 0) {
                MPI_Bcast(h_members, Ec, MPI_INT, owner[Epos], MPI_COMM_WORLD);
                if (!local) {
                    CUDA_CHECK(cudaMemcpyAsync(d_members, h_members, Ec * sizeof(int),
                                               cudaMemcpyHostToDevice, stream));
                    markClusteredKernel<<<(Ec + 255) / 256, 256, 0, stream>>>(d_members, Ec, d_pos,
                                                                             d_clustered);
                    CUDA_CHECK(cudaGetLastError());
                }
            }
        } else {
            h_seeds[N] = Epos;
            CUDA_CHECK(cudaMemcpyAsync(d_seeds + N, h_seeds + N, sizeof(int), cudaMemcpyHostToDevice, stream));
            launchSingle(nbr[Epos], slot + 4);
            markClusteredKernel<<<(Ec + 255) / 256, 256, 0, stream>>>(d_members, Ec, d_pos, d_clustered);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpyAsync(h_members, d_members, Ec * sizeof(int),
                                       cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        for (int i = 0; i < Ec; ++i) clustered[pos[h_members[i]]] = 1;
        remaining -= Ec;
        Cluster cl;
        cl.seed_point = Eidx;
        cl.members.assign(h_members, h_members + Ec);
        clusters.push_back(std::move(cl));

        // Seeds that lost a neighbour: update candidate counts / bounds. An
        // exact cardinality stays valid unless a member of the seed's cluster
        // was removed (checked against the stored member lists on the owning
        // GPUs). All removed points lie in the 3x3 cells around the winning
        // seed, so the affected seeds lie in its 5x5 cells.
        {
            const int scx = cellOf[Epos] % ncx, scy = cellOf[Epos] / ncx;
            for (auto& bk : buckets) bk.clear();
            for (int i = 0; i < Ec; ++i) {
                const int p = pos[h_members[i]];
                const int b = (cellOf[p] / ncx - scy + 1) * 3 + (cellOf[p] % ncx - scx + 1);
                buckets[b].push_back(p);
            }
            dirtyList.clear();
            for (int yy = std::max(0, scy - 2); yy <= std::min(ncx - 1, scy + 2); ++yy)
                for (int xx = std::max(0, scx - 2); xx <= std::min(ncx - 1, scx + 2); ++xx) {
                    const int cell = yy * ncx + xx;
                    for (int q = cellStart[cell]; q < cellStart[cell + 1]; ++q)
                        if (!clustered[q]) dirtyList.push_back(q);
                }
            const int nq = static_cast<int>(dirtyList.size());
            const bool stored = (d_memStore != nullptr);
            needCheck.assign(nq, 0);
#pragma omp parallel for schedule(dynamic, 16) num_threads(ompThreads(static_cast<long long>(nq) * Ec / 4, 65536))
            for (int j = 0; j < nq; ++j) {
                const int q = dirtyList[j];
                const int qcx = cellOf[q] % ncx - scx, qcy = cellOf[q] / ncx - scy;
                int lost = 0;
                for (int by = std::max(-1, qcy - 1); by <= std::min(1, qcy + 1); ++by)
                    for (int bx = std::max(-1, qcx - 1); bx <= std::min(1, qcx + 1); ++bx)
                        for (const int r : buckets[(by + 1) * 3 + (bx + 1)])
                            lost += isNeighbour(q, r);
                if (lost > 0) {
                    nbr[q] -= lost;
                    if (exact[q] && stored) {
                        needCheck[j] = 1;
                    } else {
                        // Bounds stay valid; candidate count may be tighter
                        val[q] = exact[q] ? nbr[q] + 1 : std::min(val[q], nbr[q] + 1);
                        exact[q] = (nbr[q] == 0);
                    }
                }
            }
            if (stored) {
                checkList.clear();
                for (int j = 0; j < nq; ++j)
                    if (needCheck[j]) checkList.push_back(dirtyList[j]);
                const int nc = static_cast<int>(checkList.size());
                checkFlags.assign(nc, 0);
                checkPairs.resize(2 * static_cast<size_t>(nc));
                int myN = 0;
                bool remote = false; // any member list held by a single rank
                std::vector<int>& idxMine = results; // reuse as scratch
                idxMine.clear();
                for (int j = 0; j < nc; ++j) {
                    const int q = checkList[j];
                    remote |= (owner[q] >= 0);
                    if (owner[q] < 0 || owner[q] == mpi.rank) {
                        checkPairs[2 * myN] = q;
                        checkPairs[2 * myN + 1] = val[q];
                        idxMine.push_back(j);
                        ++myN;
                    }
                }
                if (myN > 0) {
                    CUDA_CHECK(cudaMemcpyAsync(d_check, checkPairs.data(), 2 * myN * sizeof(int),
                                               cudaMemcpyHostToDevice, stream));
                    memberCheckKernel<<<(myN * 32 + 255) / 256, 256, 0, stream>>>(
                        d_check, myN, d_memStore, d_memOff, d_pos, d_clustered, d_flags);
                    CUDA_CHECK(cudaGetLastError());
                    CUDA_CHECK(cudaMemcpyAsync(h_cards, d_flags, myN * sizeof(int),
                                               cudaMemcpyDeviceToHost, stream));
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                    for (int j = 0; j < myN; ++j) checkFlags[idxMine[j]] = h_cards[j];
                }
                if (mpi.size > 1 && remote)
                    MPI_Allreduce(MPI_IN_PLACE, checkFlags.data(), nc, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
                for (int j = 0; j < nc; ++j) {
                    if (!checkFlags[j]) continue;
                    const int q = checkList[j];
                    val[q] = nbr[q] + 1;
                    exact[q] = (nbr[q] == 0);
                }
            }
        }
    }
    // Device work is complete (the last iteration synchronised or only queued
    // the clustered-flag update, which is irrelevant from here on)
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaStreamDestroy(stream2));
    CUDA_CHECK(cudaStreamDestroy(stream3));
    CUDA_CHECK(cudaEventDestroy(forkEv));
    CUDA_CHECK(cudaEventDestroy(joinEv2));
    CUDA_CHECK(cudaEventDestroy(joinEv3));
    cudaFree(d_px); cudaFree(d_py); cudaFree(d_order); cudaFree(d_pos); cudaFree(d_cellOf);
    cudaFree(d_cellStart); cudaFree(d_seeds); cudaFree(d_cards); cudaFree(d_members);
    cudaFree(d_counter); cudaFree(d_clustered); cudaFree(d_check); cudaFree(d_flags);
    if (d_scratch) cudaFree(d_scratch);
    if (d_memStore) cudaFree(d_memStore);
    if (d_memOff) cudaFree(d_memOff);

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
    MPI_Init(&argc, &argv);
    MpiCtx mpi;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi.size);
    const bool root = (mpi.rank == 0);

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
            MPI_Finalize();
            return 0;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    
    if (num_points <= 0 || threshold <= 0.0) {
        if (root) {
            printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n", 
                   num_points, threshold);
        }
        MPI_Finalize();
        return 1;
    }

    // Bind each rank to a GPU of its node (round-robin over local ranks)
    {
        MPI_Comm local;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, mpi.rank, MPI_INFO_NULL, &local);
        int localRank = 0;
        MPI_Comm_rank(local, &localRank);
        MPI_Comm_free(&local);
        int ndev = 0;
        CUDA_CHECK(cudaGetDeviceCount(&ndev));
        if (ndev <= 0) {
            fprintf(stderr, "Error: no CUDA device available\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        CUDA_CHECK(cudaSetDevice(localRank % ndev));
        // Create the context and load the kernels outside the timed region
        CUDA_CHECK(cudaFree(nullptr));
        cudaFuncAttributes fa;
        CUDA_CHECK(cudaFuncGetAttributes(&fa, qtSeedKernel<false, QT_BLOCK>));
    }
    
    if (root) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Generate synthetic data (identical on every rank)
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);
    
    // Perform QT clustering
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_start = std::chrono::high_resolution_clock::now();
    
    const std::vector<Cluster> clusters = qtClustering(points, threshold, mpi);
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);

    if (!root) {
        MPI_Finalize();
        return 0;
    }
    
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
            MPI_Finalize();
            return 0;
        } else {
            printf("Validation: FAILED\n");
            MPI_Finalize();
            return 1;
        }
    }
    
    MPI_Finalize();
    return 0;
}
