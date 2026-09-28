// QT Clustering Benchmark - CUDA Parallel Version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// Parallelization strategy (CUDA):
//   * The expensive part of the algorithm is that, in every round, *every*
//     still-unclustered point is tried as a cluster seed and grown greedily.
//     Those candidate clusters are completely independent of each other, so
//     one CUDA warp is assigned to each seed and the warps are spread over
//     the whole GPU (grid-stride over the seed list).
//   * Inside a warp, the candidate points are processed cooperatively: every
//     lane keeps a slice of the candidate list, the "current maximum distance
//     to any cluster member" is maintained incrementally (mathematically
//     identical to the O(size) rescan of the sequential code, since a maximum
//     is exact and associative) and the closest admissible candidate is found
//     with a warp shuffle reduction.
//   * Because that running maximum is monotonically non-decreasing, a
//     candidate whose value reaches the threshold can never become admissible
//     again.  The candidate list is therefore compacted (warp-aggregated) in
//     every step, which removes the vast majority of the sequential version's
//     work without changing the result.
//   * For the same reason a seed only ever draws from the points that are
//     within the threshold of it, so those neighbourhoods are precomputed once
//     into CSR lists and the per-seed scan is proportional to the
//     neighbourhood rather than to N.
//   * A candidate cluster can only change when one of the seed's neighbours
//     gets clustered.  Cardinalities and member lists are therefore cached per
//     seed and every round only re-grows the seeds that the previous round
//     actually touched.
//   * Selecting the winner, committing it and rebuilding the seed and work
//     lists all happen in device kernels, so whole batches of rounds are
//     enqueued without a single host synchronisation.
//   * The pairwise distance matrix is precomputed once on the GPU when it fits
//     into device memory; otherwise distances are evaluated on the fly with
//     the exact same expression.  Every one of these precomputations has a
//     plain fallback for the case where device memory is too small.
//
// The selection rules of the sequential code are reproduced bit-exactly:
// the admissible candidate with the smallest diameter wins, ties are broken
// towards the smaller point index, and among all seeds the one with the
// largest cardinality wins, ties broken towards the earlier seed.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <climits>
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

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        const cudaError_t err_ = (call);                                      \
        if (err_ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__,  \
                    __LINE__, cudaGetErrorString(err_));                      \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

static const int WARP_SIZE = 32;
static const int BLOCK_THREADS = 256;                          // 8 warps / block
static const int WARPS_PER_BLOCK = BLOCK_THREADS / WARP_SIZE;
static const unsigned FULL_MASK = 0xffffffffu;

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
// Device code
// ---------------------------------------------------------------------------

__device__ __forceinline__ double devDistance(const Point a, const Point b) {
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    return sqrt(dx * dx + dy * dy);
}

// CSR-style adjacency listing, for every point, all other points that are
// strictly closer than the threshold.  Only those can ever share a cluster
// with it, so a candidate cluster never has to look at anything else.
// `offsets == nullptr` means the lists were too large to precompute.
struct NeighborList {
    const int* offsets;    // N + 1 entries
    const int* idx;        // neighbour point ids
    const double* dist;    // matching distances (all < threshold)
};

// Precompute the full (symmetric) pairwise distance matrix.
__global__ void buildDistanceMatrix(const Point* __restrict__ points, int N,
                                    double* __restrict__ dmat) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y;
    if (j >= N) return;
    dmat[static_cast<size_t>(i) * N + j] = devDistance(points[i], points[j]);
}

template <bool USE_MATRIX>
__device__ __forceinline__ double lookupDistance(int a, int b,
                                                 const Point* __restrict__ points,
                                                 const double* __restrict__ dmat,
                                                 int N) {
    if (USE_MATRIX) {
        return __ldg(&dmat[static_cast<size_t>(a) * N + b]);
    }
    return devDistance(points[a], points[b]);
}

// Pass 1 of the neighbour-list construction: how many points are in range.
template <bool USE_MATRIX>
__global__ void countNeighbors(int N, double threshold,
                               const Point* __restrict__ points,
                               const double* __restrict__ dmat,
                               int* __restrict__ counts) {
    const int i = blockIdx.x;
    int local = 0;
    for (int j = threadIdx.x; j < N; j += blockDim.x) {
        if (j == i) continue;
        if (lookupDistance<USE_MATRIX>(i, j, points, dmat, N) < threshold) ++local;
    }
    __shared__ int sTotal;
    if (threadIdx.x == 0) sTotal = 0;
    __syncthreads();
    atomicAdd(&sTotal, local);
    __syncthreads();
    if (threadIdx.x == 0) counts[i] = sTotal;
}

// Pass 2: fill in the neighbour ids together with their distances.
template <bool USE_MATRIX>
__global__ void fillNeighbors(int N, double threshold,
                              const Point* __restrict__ points,
                              const double* __restrict__ dmat,
                              const int* __restrict__ offsets,
                              int* __restrict__ nbrIdx, double* __restrict__ nbrDist) {
    const int i = blockIdx.x;
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const unsigned lanesBelow = (1u << lane) - 1u;
    const int base = offsets[i];

    __shared__ int sCursor;
    if (threadIdx.x == 0) sCursor = 0;
    __syncthreads();

    const int lim = (N + blockDim.x - 1) / blockDim.x * blockDim.x;
    for (int j = threadIdx.x; j < lim; j += blockDim.x) {
        double d = 0.0;
        bool keep = false;
        if (j < N && j != i) {
            d = lookupDistance<USE_MATRIX>(i, j, points, dmat, N);
            keep = d < threshold;
        }
        const unsigned mask = __ballot_sync(FULL_MASK, keep);
        int warpBase = 0;
        if (lane == 0) warpBase = atomicAdd(&sCursor, __popc(mask));
        warpBase = __shfl_sync(FULL_MASK, warpBase, 0);
        if (keep) {
            const int pos = base + warpBase + __popc(mask & lanesBelow);
            nbrIdx[pos] = j;
            nbrDist[pos] = d;
        }
    }
}

// Warp-wide reduction picking the smallest diameter; ties go to the smaller
// point index (matching the sequential `max_dist < min_diameter` scan order).
__device__ __forceinline__ void warpArgMin(double& bestDiam, int& bestIdx) {
#pragma unroll
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
        const double od = __shfl_down_sync(FULL_MASK, bestDiam, off);
        const int oi = __shfl_down_sync(FULL_MASK, bestIdx, off);
        if (od < bestDiam || (od == bestDiam && oi < bestIdx)) {
            bestDiam = od;
            bestIdx = oi;
        }
    }
    bestDiam = __shfl_sync(FULL_MASK, bestDiam, 0);
    bestIdx = __shfl_sync(FULL_MASK, bestIdx, 0);
}

// Grow the candidate cluster of one seed with a single warp.
// Returns the cardinality; when `members` is non-null the member list is
// written to it (in the order in which the sequential code appends them).
template <bool USE_MATRIX>
__device__ int growCluster(int seed, int N, double threshold,
                           const unsigned char* __restrict__ clustered,
                           const Point* __restrict__ points,
                           const double* __restrict__ dmat,
                           const NeighborList nbr,
                           int* __restrict__ actA, double* __restrict__ diamA,
                           int* __restrict__ actB, double* __restrict__ diamB,
                           int* __restrict__ members) {
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const unsigned lanesBelow = (1u << lane) - 1u;

    int* act = actA;
    double* diam = diamA;
    int* actNext = actB;
    double* diamNext = diamB;

    // Seed the candidate list: every unclustered point that is close enough to
    // the seed itself.  Everything else can never join this cluster.
    int count = 0;
    if (nbr.offsets != nullptr) {
        // Walk only the precomputed in-range neighbours of the seed.
        const int begin = nbr.offsets[seed];
        const int end = nbr.offsets[seed + 1];
        const int lim = begin + (((end - begin) + WARP_SIZE - 1) & ~(WARP_SIZE - 1));
        for (int k = begin + lane; k < lim; k += WARP_SIZE) {
            int c = -1;
            double d = 0.0;
            bool keep = false;
            if (k < end) {
                c = nbr.idx[k];
                d = nbr.dist[k];
                keep = !clustered[c];
            }
            const unsigned mask = __ballot_sync(FULL_MASK, keep);
            if (keep) {
                const int pos = count + __popc(mask & lanesBelow);
                act[pos] = c;
                diam[pos] = d;
            }
            count += __popc(mask);
        }
    } else {
        const int limN = (N + WARP_SIZE - 1) & ~(WARP_SIZE - 1);
        for (int c = lane; c < limN; c += WARP_SIZE) {
            double d = 0.0;
            bool keep = (c < N) && (c != seed) && !clustered[c];
            if (keep) {
                d = lookupDistance<USE_MATRIX>(c, seed, points, dmat, N);
                keep = d < threshold;
            }
            const unsigned mask = __ballot_sync(FULL_MASK, keep);
            if (keep) {
                const int pos = count + __popc(mask & lanesBelow);
                act[pos] = c;
                diam[pos] = d;
            }
            count += __popc(mask);
        }
    }

    int cardinality = 1;
    if (members != nullptr && lane == 0) members[0] = seed;

    while (count > 0) {
        // Closest admissible candidate.
        double bestDiam = INFINITY;
        int bestIdx = -1;
        const int limC = (count + WARP_SIZE - 1) & ~(WARP_SIZE - 1);
        for (int i = lane; i < limC; i += WARP_SIZE) {
            if (i < count) {
                const double d = diam[i];
                const int idx = act[i];
                if (d < bestDiam || (d == bestDiam && idx < bestIdx)) {
                    bestDiam = d;
                    bestIdx = idx;
                }
            }
        }
        warpArgMin(bestDiam, bestIdx);

        if (members != nullptr && lane == 0) members[cardinality] = bestIdx;
        ++cardinality;

        // Update the running diameters and drop candidates that became
        // inadmissible (their diameter can only grow from here on).
        int nextCount = 0;
        for (int i = lane; i < limC; i += WARP_SIZE) {
            double nd = 0.0;
            int c = -1;
            bool keep = false;
            if (i < count) {
                c = act[i];
                keep = (c != bestIdx);
                if (keep) {
                    const double d = lookupDistance<USE_MATRIX>(c, bestIdx, points, dmat, N);
                    nd = fmax(diam[i], d);
                    keep = nd < threshold;
                }
            }
            const unsigned mask = __ballot_sync(FULL_MASK, keep);
            if (keep) {
                const int pos = nextCount + __popc(mask & lanesBelow);
                actNext[pos] = c;
                diamNext[pos] = nd;
            }
            nextCount += __popc(mask);
        }

        int* ti = act;   act = actNext;       actNext = ti;
        double* td = diam; diam = diamNext;   diamNext = td;
        count = nextCount;
    }

    return cardinality;
}

// One warp per seed, grid-stride over the seed list.
template <bool USE_MATRIX>
__global__ void candidateCardinalityKernel(const int* __restrict__ workCountPtr,
                                           const int* __restrict__ workList,
                                           int N, double threshold,
                                           const unsigned char* __restrict__ clustered,
                                           const Point* __restrict__ points,
                                           const double* __restrict__ dmat,
                                           const NeighborList nbr, int stride,
                                           int* __restrict__ actA, double* __restrict__ diamA,
                                           int* __restrict__ actB, double* __restrict__ diamB,
                                           int* __restrict__ memberScratch,
                                           int* __restrict__ cardinality) {
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int warpId = blockIdx.x * WARPS_PER_BLOCK + (threadIdx.x >> 5);
    const int totalWarps = gridDim.x * WARPS_PER_BLOCK;
    const size_t base = static_cast<size_t>(warpId) * stride;
    const int work = *workCountPtr;

    for (int w = warpId; w < work; w += totalWarps) {
        const int seed = workList[w];
        // The member list is cached per seed as well, so the winner of the
        // round never has to be regrown.
        int* members = (memberScratch != nullptr)
                           ? memberScratch + static_cast<size_t>(seed) * stride
                           : nullptr;
        const int card = growCluster<USE_MATRIX>(seed, N, threshold, clustered,
                                                 points, dmat, nbr, actA + base,
                                                 diamA + base, actB + base,
                                                 diamB + base, members);
        if (lane == 0) cardinality[seed] = card;
    }
}

// Largest cardinality wins; ties go to the earliest seed (the sequential code
// only replaces the best on a strict `>`), single block.  When the member
// lists were recorded, the winning one is also emitted here.
__global__ void selectBestSeedKernel(const int* __restrict__ nSeedsPtr,
                                     const int* __restrict__ seeds,
                                     const int* __restrict__ cardinality,
                                     const int* __restrict__ memberScratch, int stride,
                                     int* __restrict__ out,
                                     int* __restrict__ bestSeed) {
    const int nSeeds = *nSeedsPtr;
    if (nSeeds == 0) return;

    __shared__ int sCard[BLOCK_THREADS];
    __shared__ int sSeed[BLOCK_THREADS];

    int bc = -1;
    int bs = INT_MAX;
    for (int i = threadIdx.x; i < nSeeds; i += blockDim.x) {
        const int sd = seeds[i];
        const int c = cardinality[sd];
        if (c > bc || (c == bc && sd < bs)) { bc = c; bs = sd; }
    }
    sCard[threadIdx.x] = bc;
    sSeed[threadIdx.x] = bs;
    __syncthreads();

    for (int half = blockDim.x >> 1; half > 0; half >>= 1) {
        if (threadIdx.x < half) {
            const int oc = sCard[threadIdx.x + half];
            const int os = sSeed[threadIdx.x + half];
            if (oc > sCard[threadIdx.x] ||
                (oc == sCard[threadIdx.x] && os < sSeed[threadIdx.x])) {
                sCard[threadIdx.x] = oc;
                sSeed[threadIdx.x] = os;
            }
        }
        __syncthreads();
    }

    const int winner = sSeed[0];
    if (threadIdx.x == 0) bestSeed[0] = winner;

    if (memberScratch != nullptr) {
        const int card = cardinality[winner];
        if (threadIdx.x == 0) out[0] = card;
        const int* src = memberScratch + static_cast<size_t>(winner) * stride;
        for (int i = threadIdx.x; i < card; i += blockDim.x) out[1 + i] = src[i];
    }
}

// Commit the winning cluster and rebuild the seed list, all on the device so
// that the rounds can be enqueued back to back without host synchronisation.
// counters[0] = number of clusters, counters[1] = members written so far.
__global__ void commitClusterKernel(const int* __restrict__ nSeedsPtr,
                                    const int* __restrict__ seedsIn,
                                    int* __restrict__ seedsOut,
                                    int* __restrict__ nSeedsOut,
                                    const int* __restrict__ out,
                                    unsigned char* __restrict__ clustered,
                                    int* __restrict__ clusterSizes,
                                    int* __restrict__ clusterMembers,
                                    int* __restrict__ counters,
                                    const NeighborList nbr, int N,
                                    int* __restrict__ stamp,
                                    int* __restrict__ workList,
                                    int* __restrict__ workCount) {
    const int nSeeds = *nSeedsPtr;
    if (nSeeds == 0) {
        // No-op round (the batch overshot): keep the empty state propagating.
        if (threadIdx.x == 0) { nSeedsOut[0] = 0; workCount[0] = 0; }
        return;
    }

    __shared__ int sOffset;
    __shared__ int sRemaining;
    __shared__ int sWork;
    __shared__ int sRound;
    const int card = out[0];
    if (threadIdx.x == 0) {
        const int cid = counters[0];
        sOffset = counters[1];
        counters[0] = cid + 1;
        counters[1] = sOffset + card;
        clusterSizes[cid] = card;
        sRemaining = 0;
        sWork = 0;
        sRound = cid;
    }
    __syncthreads();

    for (int i = threadIdx.x; i < card; i += blockDim.x) {
        const int m = out[1 + i];
        clusterMembers[sOffset + i] = m;
        clustered[m] = 1;
    }
    __syncthreads();

    // Stream-compact the still unclustered seeds into the other buffer.
    for (int i = threadIdx.x; i < nSeeds; i += blockDim.x) {
        const int sd = seedsIn[i];
        if (!clustered[sd]) seedsOut[atomicAdd(&sRemaining, 1)] = sd;
    }

    // A candidate cluster only changes if one of the seed's in-range
    // neighbours was just clustered, so only those seeds are re-grown next
    // round.  The neighbour relation is symmetric, hence the walk over the
    // neighbours of the removed members.  `stamp` deduplicates the list.
    if (nbr.offsets != nullptr) {
        for (int i = 0; i < card; ++i) {
            const int m = out[1 + i];
            const int begin = nbr.offsets[m];
            const int end = nbr.offsets[m + 1];
            for (int k = begin + threadIdx.x; k < end; k += blockDim.x) {
                const int j = nbr.idx[k];
                if (!clustered[j] && atomicExch(&stamp[j], sRound) != sRound) {
                    workList[atomicAdd(&sWork, 1)] = j;
                }
            }
        }
    } else {
        // Without neighbour lists every remaining seed has to be redone.
        for (int i = threadIdx.x; i < N; i += blockDim.x) {
            if (!clustered[i]) workList[atomicAdd(&sWork, 1)] = i;
        }
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        nSeedsOut[0] = sRemaining;
        workCount[0] = sWork;
    }
}

// Regenerate the winning cluster and record its members.
// out[0] = cardinality, out[1 .. cardinality] = members.
template <bool USE_MATRIX>
__global__ void recordBestClusterKernel(const int* __restrict__ nSeedsPtr,
                                        const int* __restrict__ bestSeed,
                                        int N, double threshold,
                                        const unsigned char* __restrict__ clustered,
                                        const Point* __restrict__ points,
                                        const double* __restrict__ dmat,
                                        const NeighborList nbr,
                                        int* __restrict__ actA, double* __restrict__ diamA,
                                        int* __restrict__ actB, double* __restrict__ diamB,
                                        int* __restrict__ out) {
    if (*nSeedsPtr == 0) return;
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int seed = bestSeed[0];
    const int card = growCluster<USE_MATRIX>(seed, N, threshold, clustered, points, dmat,
                                             nbr, actA, diamA, actB, diamB, out + 1);
    if (lane == 0) out[0] = card;
}

// ---------------------------------------------------------------------------
// Host driver
// ---------------------------------------------------------------------------

namespace {

struct DeviceState {
    Point* points = nullptr;
    double* dmat = nullptr;          // optional precomputed distance matrix
    unsigned char* clustered = nullptr;
    int* seeds = nullptr;
    int* cardinality = nullptr;
    int* bestSeed = nullptr;
    int* out = nullptr;
    int* actA = nullptr;
    int* actB = nullptr;
    double* diamA = nullptr;
    double* diamB = nullptr;
    int* memberScratch = nullptr;
    int* seedsAlt = nullptr;
    int* nSeeds = nullptr;
    int* clusterSizes = nullptr;
    int* clusterMembers = nullptr;
    int* counters = nullptr;
    int* workList = nullptr;
    int* workCount = nullptr;
    int* stamp = nullptr;
    int* nbrOffsets = nullptr;
    int* nbrIdx = nullptr;
    double* nbrDist = nullptr;
    int warps = 0;
    int blocks = 0;
    bool useMatrix = false;
};

// Bump allocator over a single device allocation: cudaMalloc costs tens of
// microseconds, which is significant next to the kernels themselves here.
struct DeviceArena {
    char* base = nullptr;
    size_t used = 0;

    static size_t align(size_t n) { return (n + 255) & ~static_cast<size_t>(255); }

    template <typename T>
    T* take(size_t count) {
        T* p = reinterpret_cast<T*>(base + used);
        used += align(count * sizeof(T));
        return p;
    }
};

// Pick the number of concurrently resident warps: one per seed if the
// per-warp scratch space fits, otherwise as many as memory allows.
int chooseWarpCount(int stride, size_t freeBytes, int nSeedsMax) {
    const size_t perWarp = static_cast<size_t>(stride) * (2 * sizeof(int) + 2 * sizeof(double));
    // Leave head room for the allocator.
    const size_t budget =
        (freeBytes > (256u << 20)) ? freeBytes - (256u << 20) : freeBytes / 2;
    size_t maxWarps = budget / perWarp;
    if (maxWarps < static_cast<size_t>(WARPS_PER_BLOCK)) maxWarps = WARPS_PER_BLOCK;
    int warps = static_cast<int>(std::min<size_t>(maxWarps, static_cast<size_t>(nSeedsMax)));
    // Round up to whole blocks.
    warps = ((warps + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK) * WARPS_PER_BLOCK;
    return warps;
}

}  // namespace

// Main QT clustering algorithm
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<Cluster> clusters;
    if (N == 0) return clusters;

    size_t freeBytes = 0, totalBytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&freeBytes, &totalBytes));

    DeviceState dev;
    const size_t matrixBytes = static_cast<size_t>(N) * static_cast<size_t>(N) * sizeof(double);
    // Use the precomputed matrix whenever it comfortably fits; FP64 sqrt is
    // expensive on the GPU, so this is a large win when it is affordable.
    dev.useMatrix = (matrixBytes + matrixBytes / 4 + (256u << 20) < freeBytes);

    // All the O(N) bookkeeping arrays share one device allocation.
    const size_t nu = static_cast<size_t>(N);
    DeviceArena arena;
    const size_t arenaBytes =
        DeviceArena::align(nu * sizeof(Point)) + DeviceArena::align(nu) +
        DeviceArena::align((nu + 1) * sizeof(int)) * 9 +   // the O(N) int arrays
        DeviceArena::align(sizeof(int)) * 4;               // the scalars
    CUDA_CHECK(cudaMalloc(&arena.base, arenaBytes));
    CUDA_CHECK(cudaMemset(arena.base, 0, arenaBytes));
    dev.points = arena.take<Point>(nu);
    dev.clustered = arena.take<unsigned char>(nu);
    dev.seeds = arena.take<int>(nu);
    dev.seedsAlt = arena.take<int>(nu);
    dev.cardinality = arena.take<int>(nu);
    dev.out = arena.take<int>(nu + 1);
    dev.clusterSizes = arena.take<int>(nu);
    dev.clusterMembers = arena.take<int>(nu);
    dev.workList = arena.take<int>(nu);
    dev.stamp = arena.take<int>(nu);
    dev.nSeeds = arena.take<int>(2);
    dev.counters = arena.take<int>(2);
    dev.bestSeed = arena.take<int>(1);
    dev.workCount = arena.take<int>(1);
    dev.nbrOffsets = arena.take<int>(nu + 1);

    CUDA_CHECK(cudaMemcpy(dev.points, points.data(), N * sizeof(Point),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dev.stamp, 0xff, N * sizeof(int)));   // round stamps: -1

    if (dev.useMatrix) {
        if (cudaMalloc(&dev.dmat, matrixBytes) != cudaSuccess) {
            dev.dmat = nullptr;
            dev.useMatrix = false;
        } else {
            const int tpb = 256;
            const dim3 grid((N + tpb - 1) / tpb, N);
            buildDistanceMatrix<<<grid, tpb>>>(dev.points, N, dev.dmat);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    int maxNeighbors = N;
    // Build the in-range neighbour lists.  They make the initial candidate
    // scan of every seed proportional to its neighbourhood instead of N.
    {
        int* dCounts = dev.cardinality;  // reused as scratch before the main loop
        if (dev.useMatrix) {
            countNeighbors<true><<<N, BLOCK_THREADS>>>(N, threshold, dev.points,
                                                       dev.dmat, dCounts);
        } else {
            countNeighbors<false><<<N, BLOCK_THREADS>>>(N, threshold, dev.points,
                                                        dev.dmat, dCounts);
        }
        CUDA_CHECK(cudaGetLastError());

        std::vector<int> counts(N);
        CUDA_CHECK(cudaMemcpy(counts.data(), dCounts, N * sizeof(int),
                              cudaMemcpyDeviceToHost));

        std::vector<int> offsets(N + 1);
        size_t total = 0;
        int maxCount = 0;
        for (int i = 0; i < N; ++i) {
            offsets[i] = static_cast<int>(total);
            total += static_cast<size_t>(counts[i]);
            maxCount = std::max(maxCount, counts[i]);
        }
        offsets[N] = static_cast<int>(total);
        maxNeighbors = maxCount;

        CUDA_CHECK(cudaMemGetInfo(&freeBytes, &totalBytes));
        const size_t csrBytes = total * (sizeof(int) + sizeof(double));
        if (total < static_cast<size_t>(INT_MAX) && csrBytes < freeBytes / 2 &&
            cudaMalloc(&dev.nbrIdx, std::max<size_t>(total, 1) * sizeof(int)) == cudaSuccess &&
            cudaMalloc(&dev.nbrDist, std::max<size_t>(total, 1) * sizeof(double)) == cudaSuccess) {
            CUDA_CHECK(cudaMemcpy(dev.nbrOffsets, offsets.data(), (N + 1) * sizeof(int),
                                  cudaMemcpyHostToDevice));
            if (dev.useMatrix) {
                fillNeighbors<true><<<N, BLOCK_THREADS>>>(N, threshold, dev.points, dev.dmat,
                                                          dev.nbrOffsets, dev.nbrIdx,
                                                          dev.nbrDist);
            } else {
                fillNeighbors<false><<<N, BLOCK_THREADS>>>(N, threshold, dev.points, dev.dmat,
                                                           dev.nbrOffsets, dev.nbrIdx,
                                                           dev.nbrDist);
            }
            CUDA_CHECK(cudaGetLastError());
        } else {
            // Too large: fall back to scanning all points for every seed.
            if (dev.nbrIdx) { cudaFree(dev.nbrIdx); dev.nbrIdx = nullptr; }
            dev.nbrOffsets = nullptr;
            cudaGetLastError();
            maxNeighbors = N;
        }
    }

    NeighborList nbr;
    nbr.offsets = dev.nbrOffsets;
    nbr.idx = dev.nbrIdx;
    nbr.dist = dev.nbrDist;

    // A candidate cluster never holds more points than the seed has
    // neighbours, so the per-warp scratch rows only need that many slots.
    const int stride = std::min(N, maxNeighbors + 1);

    // Cache of the member list of every seed's candidate cluster.  Together
    // with the work list this makes the winner of a round directly available
    // and lets unaffected seeds keep their result across rounds.
    CUDA_CHECK(cudaMemGetInfo(&freeBytes, &totalBytes));
    const size_t memberBytes = static_cast<size_t>(N) * stride * sizeof(int);
    if (memberBytes + (256u << 20) < freeBytes) {
        if (cudaMalloc(&dev.memberScratch, memberBytes) != cudaSuccess) {
            dev.memberScratch = nullptr;
            cudaGetLastError();
        }
    }

    CUDA_CHECK(cudaMemGetInfo(&freeBytes, &totalBytes));
    dev.warps = chooseWarpCount(stride, freeBytes, N);
    dev.blocks = dev.warps / WARPS_PER_BLOCK;
    const size_t scratch = static_cast<size_t>(dev.warps) * stride;

    DeviceArena work;
    const size_t workBytes = 2 * DeviceArena::align(scratch * sizeof(int)) +
                             2 * DeviceArena::align(scratch * sizeof(double));
    CUDA_CHECK(cudaMalloc(&work.base, workBytes));
    dev.actA = work.take<int>(scratch);
    dev.actB = work.take<int>(scratch);
    dev.diamA = work.take<double>(scratch);
    dev.diamB = work.take<double>(scratch);

    // Initially every point is a seed and every candidate cluster is unknown.
    std::vector<int> hInit(N + 1);
    for (int i = 0; i < N; ++i) hInit[i] = i;
    hInit[N] = N;
    CUDA_CHECK(cudaMemcpy(dev.seeds, hInit.data(), N * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev.workList, hInit.data(), N * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev.nSeeds, &hInit[N], sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev.workCount, &hInit[N], sizeof(int), cudaMemcpyHostToDevice));

    // Every round removes at least its seed, so at most N rounds are needed.
    // Rounds that run out of seeds turn into no-ops, which makes it safe to
    // enqueue them in batches and only check for completion once per batch.
    int* seedBuf[2] = {dev.seeds, dev.seedsAlt};
    int parity = 0;
    const int BATCH = 32;
    int roundsLeft = N;
    while (roundsLeft > 0) {
        const int batch = std::min(BATCH, roundsLeft);
        for (int r = 0; r < batch; ++r) {
            int* seedsIn = seedBuf[parity];
            int* seedsOut = seedBuf[parity ^ 1];
            int* nIn = dev.nSeeds + parity;
            int* nOut = dev.nSeeds + (parity ^ 1);

            if (dev.useMatrix) {
                candidateCardinalityKernel<true><<<dev.blocks, BLOCK_THREADS>>>(
                    dev.workCount, dev.workList, N, threshold, dev.clustered, dev.points, dev.dmat,
                    nbr, stride, dev.actA, dev.diamA, dev.actB, dev.diamB,
                    dev.memberScratch, dev.cardinality);
            } else {
                candidateCardinalityKernel<false><<<dev.blocks, BLOCK_THREADS>>>(
                    dev.workCount, dev.workList, N, threshold, dev.clustered, dev.points, dev.dmat,
                    nbr, stride, dev.actA, dev.diamA, dev.actB, dev.diamB,
                    dev.memberScratch, dev.cardinality);
            }

            selectBestSeedKernel<<<1, BLOCK_THREADS>>>(nIn, seedsIn, dev.cardinality,
                                                       dev.memberScratch, stride,
                                                       dev.out, dev.bestSeed);

            if (dev.memberScratch == nullptr) {
                // Not enough memory to keep every member list: regrow the winner.
                if (dev.useMatrix) {
                    recordBestClusterKernel<true><<<1, WARP_SIZE>>>(
                        nIn, dev.bestSeed, N, threshold, dev.clustered,
                        dev.points, dev.dmat, nbr, dev.actA, dev.diamA, dev.actB,
                        dev.diamB, dev.out);
                } else {
                    recordBestClusterKernel<false><<<1, WARP_SIZE>>>(
                        nIn, dev.bestSeed, N, threshold, dev.clustered,
                        dev.points, dev.dmat, nbr, dev.actA, dev.diamA, dev.actB,
                        dev.diamB, dev.out);
                }
            }

            commitClusterKernel<<<1, BLOCK_THREADS>>>(
                nIn, seedsIn, seedsOut, nOut, dev.out, dev.clustered,
                dev.clusterSizes, dev.clusterMembers, dev.counters, nbr, N,
                dev.stamp, dev.workList, dev.workCount);

            parity ^= 1;
        }
        CUDA_CHECK(cudaGetLastError());
        roundsLeft -= batch;

        int remaining = 0;
        CUDA_CHECK(cudaMemcpy(&remaining, dev.nSeeds + parity, sizeof(int),
                              cudaMemcpyDeviceToHost));
        if (remaining == 0) break;
    }

    // Fetch the concatenated member lists and rebuild the cluster objects.
    int hCounters[2] = {0, 0};
    CUDA_CHECK(cudaMemcpy(hCounters, dev.counters, 2 * sizeof(int),
                          cudaMemcpyDeviceToHost));
    const int numClusters = hCounters[0];
    std::vector<int> sizes(numClusters);
    std::vector<int> members(hCounters[1]);
    CUDA_CHECK(cudaMemcpy(sizes.data(), dev.clusterSizes, numClusters * sizeof(int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(members.data(), dev.clusterMembers,
                          hCounters[1] * sizeof(int), cudaMemcpyDeviceToHost));

    clusters.resize(numClusters);
    int offset = 0;
    for (int c = 0; c < numClusters; ++c) {
        const int size = sizes[c];
        clusters[c].seed_point = members[offset];
        clusters[c].members.assign(members.begin() + offset,
                                   members.begin() + offset + size);
        offset += size;
    }

    CUDA_CHECK(cudaFree(arena.base));
    CUDA_CHECK(cudaFree(work.base));
    if (dev.memberScratch) CUDA_CHECK(cudaFree(dev.memberScratch));
    if (dev.nbrIdx) CUDA_CHECK(cudaFree(dev.nbrIdx));
    if (dev.nbrDist) CUDA_CHECK(cudaFree(dev.nbrDist));
    if (dev.dmat) CUDA_CHECK(cudaFree(dev.dmat));

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

    // Bring up the CUDA context (and load all kernels) before timing so that
    // the measured region contains algorithm work only.
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaFree(nullptr));
    {
        std::vector<Point> warmup(64);
        generateSyntheticData(warmup, 64, 7);
        qtClustering(warmup, 1.0);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Generate synthetic data
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);

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
