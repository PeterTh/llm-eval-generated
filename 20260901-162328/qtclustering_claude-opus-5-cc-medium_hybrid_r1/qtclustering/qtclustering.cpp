// QT Clustering Benchmark - Hybrid MPI + OpenMP + CUDA Version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// Parallelization strategy (unconditional MPI + OpenMP + CUDA):
//   * MPI   : the candidate seeds of every outer iteration are distributed
//             round robin over the ranks; the globally best candidate cluster
//             is picked with one MPI_MAXLOC all-reduce and its membership
//             bitmask is broadcast. One rank drives one GPU.
//   * CUDA  : one warp grows one candidate cluster. Each warp keeps a
//             compacted list of the still admissible candidates together with
//             their running "maximum distance to the cluster", so a growth
//             step costs O(active) instead of O(N * |members|) and shrinks as
//             candidates drop out. A step is a single fused, barrier free
//             pass: update, ballot based order preserving compaction and the
//             reduction of the closest candidate all happen at once.
//   * OpenMP: host side reduction over the per-seed cardinalities, the gather
//             of the compacted point set and the cluster validation.
//
// The greedy decisions (largest candidate cluster, lowest index on ties;
// closest admissible point, lowest index on ties) are identical to the
// sequential reference implementation and so is the arithmetic, hence the
// resulting clustering is bit-for-bit equivalent. Distances are compared in
// squared form (sqrt is monotonic, and FP64 square roots are expensive on
// GPUs); the only case in which that could differ - two distinct squared
// distances whose square roots round to the same double - is detected and
// resolved with real square roots.

#include <algorithm>
#include <chrono>
#include <climits>
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

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        const cudaError_t err_ = (call);                                       \
        if (err_ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                        \
                    cudaGetErrorString(err_), __FILE__, __LINE__);             \
            MPI_Abort(MPI_COMM_WORLD, 1);                                      \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// Data generation (identical to the reference, deterministic on every rank)
// ---------------------------------------------------------------------------

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
// CUDA kernel: grow candidate clusters, one warp per seed
// ---------------------------------------------------------------------------

#define WPB 1               // warps (i.e. concurrent seeds) per block
#define TPB (32 * WPB)
#define MAX_SLOTS 8192      // upper bound on concurrently grown clusters per GPU

// Minimum number of host side elements per thread before spawning an OpenMP
// team pays off. The team size is scaled with the work so that neither tiny
// loops nor several ranks sharing a node oversubscribe the cores.
#define OMP_MIN_WORK 16384

static inline int ompTeam(const long long work) {
    const int t = static_cast<int>(work / OMP_MIN_WORK);
    return std::max(1, std::min(omp_get_max_threads(), t));
}

// Squared distance. Matches the host expression `sqrt(dx*dx + dy*dy)` as
// contracted by the host compiler (fma(dx, dx, dy*dy)) up to the final square
// root, which is never needed: sqrt() is monotonic, so comparing squared
// distances - and taking their maximum - yields exactly the same decisions as
// the reference implementation. The only exception are two *different*
// squared distances whose square roots round to the same double; those are
// flagged as near ties (see minCombine) and resolved explicitly.
__device__ __forceinline__ double gpuDist2(const double2 a, const double2 b) {
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    return fma(dx, dx, dy * dy);
}

// Non-negative doubles compare, and take their maximum, exactly like their
// bit patterns do as unsigned integers. Since FP64 throughput is scarce on
// GPUs while integer throughput is not, the candidate values are kept as bit
// patterns and only the squared distance itself is computed in floating point.
// ULLONG_MAX is used as an "excluded" sentinel; it orders above every double.
typedef unsigned long long u64;
#define VAL_INF (~0ull)

// Two squared values can only share a square root if their bit patterns are
// within four units of each other (a sqrt bucket spans at most 2^-51
// relative, one unit is at least 2^-53); 16 units is a safe margin.
#define TIE_UNITS 16ull

// Keep the smaller squared value, lowest point index on ties. Two distinct
// values that are close enough to possibly share a square root set `tie`.
__device__ __forceinline__ void minCombine(u64& bv, int& bi, const u64 v, const int i, bool& tie) {
    const u64 d = (v > bv) ? (v - bv) : (bv - v);
    if (d != 0 && d <= TIE_UNITS) tie = true;
    if (v < bv || (v == bv && i < bi)) {
        bv = v;
        bi = i;
    }
}

// For the nseeds seeds seedBase, seedBase + seedStep, ... grow the greedy
// candidate cluster and store its cardinality in cardOut[i].
//
// Every warp also keeps the membership bitmask of the largest cluster it has
// grown so far (lowest seed index on ties). The globally winning seed is by
// construction the best seed of its own warp slot, so its member set is
// available afterwards without re-growing it.
//
// One warp owns one seed and keeps a compacted, index ordered list of the
// still admissible candidates (aIn) together with the running maximum
// distance of each candidate to the current cluster (mIn). A growth step is a
// single fused pass over that list which updates the maxima, drops the
// candidates that can no longer stay below the threshold (ballot based, order
// preserving stream compaction into the second buffer) and reduces the
// closest remaining candidate - all without any barrier or shared memory.
__global__ __launch_bounds__(TPB) void qtGrowKernel(
    const double2* __restrict__ pts, const int nu, const u64 thrSq,
    const int seedBase, const int seedStep, const int nseeds, const long long stride,
    const int nslots,
    int* __restrict__ actBuf, u64* __restrict__ mdBuf, int* __restrict__ cardOut,
    unsigned* __restrict__ maskCurBuf, unsigned* __restrict__ maskBestBuf,
    const int maskStride) {
    const int lane = threadIdx.x & 31;
    const int slot = blockIdx.x * WPB + (threadIdx.x >> 5);
    const int nWarpsTotal = gridDim.x * WPB;

    int* const actA = actBuf + static_cast<long long>(slot) * stride;
    int* const actB = actA + static_cast<long long>(nslots) * stride;
    u64* const mdA = mdBuf + static_cast<long long>(slot) * stride;
    u64* const mdB = mdA + static_cast<long long>(nslots) * stride;
    unsigned* const mCur = maskCurBuf + static_cast<long long>(slot) * maskStride;
    unsigned* const mBest = maskBestBuf + static_cast<long long>(slot) * maskStride;
    const int nwords = (nu + 31) >> 5;
    int slotBest = -1;

    for (int s = slot; s < nseeds; s += nWarpsTotal) {
        int* aIn = actA;
        int* aOut = actB;
        u64* mIn = mdA;
        u64* mOut = mdB;

        for (int i = lane; i < nwords; i += 32) mCur[i] = 0u;
        __syncwarp();

        const int seed = seedBase + s * seedStep;
        int member = seed;
        int card = 1;
        if (lane == 0) mCur[seed >> 5] = 1u << (seed & 31);

        int cnt = nu;   // number of admissible candidates
        bool first = true;

        while (card < nu) {
            const double2 pm = pts[member];
            u64 bv = VAL_INF;
            int bi = INT_MAX;
            bool tie = false;
            int o = 0;

            if (first) {
                // The initial candidate list is implicitly 0..nu-1 with all
                // maxima at 0, so it never has to be materialized.
                for (int base = 0; base < nu; base += 32) {
                    const int i = base + lane;
                    u64 v = VAL_INF;
                    if (i < nu && i != member) v = __double_as_longlong(gpuDist2(pts[i], pm));
                    const bool keep = v < thrSq;
                    const unsigned mask = __ballot_sync(0xffffffffu, keep);
                    const int pos = o + __popc(mask & ((1u << lane) - 1u));
                    if (keep) {
                        aOut[pos] = i;
                        mOut[pos] = v;
                        minCombine(bv, bi, v, i, tie);
                    }
                    o += __popc(mask);
                }
            } else {
                for (int base = 0; base < cnt; base += 32) {
                    const int i = base + lane;
                    int j = -1;
                    u64 v = VAL_INF;
                    if (i < cnt) {
                        j = aIn[i];
                        if (j != member) {
                            v = max(mIn[i], __double_as_longlong(gpuDist2(pts[j], pm)));
                        }
                    }
                    const bool keep = v < thrSq;
                    const unsigned mask = __ballot_sync(0xffffffffu, keep);
                    const int pos = o + __popc(mask & ((1u << lane) - 1u));
                    if (keep) {
                        aOut[pos] = j;
                        mOut[pos] = v;
                        minCombine(bv, bi, v, j, tie);
                    }
                    o += __popc(mask);
                }
            }
            first = false;
            cnt = o;  // identical in every lane

            // Warp wide all-reduce of the closest admissible candidate.
            for (int off = 16; off > 0; off >>= 1) {
                const u64 ov = __shfl_xor_sync(0xffffffffu, bv, off);
                const int oi = __shfl_xor_sync(0xffffffffu, bi, off);
                minCombine(bv, bi, ov, oi, tie);
            }

            if (cnt == 0) break;  // no more points can be added

            // Exactness fixup for the (astronomically unlikely) case that the
            // minimum shares its square root with a larger squared value: the
            // reference implementation would then pick the lowest index.
            if (__any_sync(0xffffffffu, tie) && bv > 0ull) {
                const double t = sqrt(__longlong_as_double(bv));
                const u64 bound = bv + TIE_UNITS;
                int cand = INT_MAX;
                for (int base = 0; base < cnt; base += 32) {
                    const int i = base + lane;
                    if (i < cnt) {
                        const int j = aOut[i];
                        const u64 v = mOut[i];
                        if (j < bi && v <= bound && sqrt(__longlong_as_double(v)) == t) {
                            cand = min(cand, j);
                        }
                    }
                }
                for (int off = 16; off > 0; off >>= 1) {
                    cand = min(cand, __shfl_xor_sync(0xffffffffu, cand, off));
                }
                if (cand != INT_MAX) bi = cand;
            }

            member = bi;
            ++card;
            if (lane == 0) mCur[member >> 5] |= 1u << (member & 31);

            int* const ta = aIn; aIn = aOut; aOut = ta;
            u64* const tm = mIn; mIn = mOut; mOut = tm;
        }

        if (lane == 0) cardOut[s] = card;

        // Retain the membership of this slot's best candidate cluster.
        if (card > slotBest) {
            slotBest = card;
            __syncwarp();
            for (int i = lane; i < nwords; i += 32) mBest[i] = mCur[i];
        }
        __syncwarp();
    }
}

// ---------------------------------------------------------------------------
// GPU resources
// ---------------------------------------------------------------------------

struct GpuState {
    double2* d_pts = nullptr;
    int* d_act = nullptr;
    u64* d_md = nullptr;
    int* d_card = nullptr;
    unsigned* d_maskCur = nullptr;
    unsigned* d_maskBest = nullptr;
    double2* h_pts = nullptr;  // pinned staging buffers
    int* h_card = nullptr;
    unsigned* h_mask = nullptr;
    int maskStride = 0;
    int nblocks = 0;  // blocks the scratch memory has been sized for
    int nslots = 0;   // = nblocks * WPB concurrently growable clusters
};

static void gpuInit(GpuState& g, const int N, const int ranksPerDevice) {
    size_t freeB = 0, totalB = 0;
    CUDA_CHECK(cudaMemGetInfo(&freeB, &totalB));

    // Per warp slot scratch: 2 * N ints (candidate lists), 2 * N 64 bit words
    // (running maxima) and 2 membership bitmasks.
    const size_t maskWords = (static_cast<size_t>(N) + 31) / 32;
    const size_t perSlot =
        static_cast<size_t>(N) * (2 * sizeof(int) + 2 * sizeof(u64)) + 2 * maskWords * sizeof(unsigned);
    const size_t perBlock = perSlot * WPB;
    size_t budget = static_cast<size_t>(freeB * 0.7) / static_cast<size_t>(ranksPerDevice);
    if (budget < perBlock) budget = perBlock;  // try anyway with a single block

    size_t nb = budget / perBlock;
    if (nb > MAX_SLOTS / WPB) nb = MAX_SLOTS / WPB;
    if (nb > (static_cast<size_t>(N) + WPB - 1) / WPB) nb = (static_cast<size_t>(N) + WPB - 1) / WPB;
    if (nb < 1) nb = 1;
    g.nblocks = static_cast<int>(nb);
    g.nslots = g.nblocks * WPB;

    CUDA_CHECK(cudaMalloc(&g.d_pts, static_cast<size_t>(N) * sizeof(double2)));
    CUDA_CHECK(cudaMalloc(&g.d_act, 2 * static_cast<size_t>(g.nslots) * N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&g.d_md, 2 * static_cast<size_t>(g.nslots) * N * sizeof(u64)));
    CUDA_CHECK(cudaMalloc(&g.d_card, static_cast<size_t>(N) * sizeof(int)));
    g.maskStride = static_cast<int>(maskWords);
    CUDA_CHECK(cudaMalloc(&g.d_maskCur, static_cast<size_t>(g.nslots) * maskWords * sizeof(unsigned)));
    CUDA_CHECK(cudaMalloc(&g.d_maskBest, static_cast<size_t>(g.nslots) * maskWords * sizeof(unsigned)));

    // Pinned staging memory: faster transfers and truly asynchronous uploads.
    CUDA_CHECK(cudaHostAlloc(&g.h_pts, static_cast<size_t>(N) * sizeof(double2), 0));
    CUDA_CHECK(cudaHostAlloc(&g.h_card, static_cast<size_t>(N) * sizeof(int), 0));
    CUDA_CHECK(cudaHostAlloc(&g.h_mask, maskWords * sizeof(unsigned), 0));

    // Empty launch: forces the (lazily loaded) kernel module in before the
    // measured region starts.
    qtGrowKernel<<<1, TPB>>>(g.d_pts, 0, 0ull, 0, 1, 0, N, g.nslots, g.d_act, g.d_md, g.d_card,
                             g.d_maskCur, g.d_maskBest, g.maskStride);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

static void gpuFree(GpuState& g) {
    cudaFreeHost(g.h_pts);
    cudaFreeHost(g.h_card);
    cudaFreeHost(g.h_mask);
    cudaFree(g.d_pts);
    cudaFree(g.d_act);
    cudaFree(g.d_md);
    cudaFree(g.d_card);
    cudaFree(g.d_maskCur);
    cudaFree(g.d_maskBest);
}

// ---------------------------------------------------------------------------
// Main QT clustering algorithm (hybrid)
// ---------------------------------------------------------------------------

std::vector<Cluster> qtClustering(const std::vector<Point>& points, const double threshold,
                                  const int rank, const int nranks, GpuState& g) {
    const int N = static_cast<int>(points.size());
    std::vector<Cluster> clusters;

    // Indices of the still unclustered points, in ascending order.
    std::vector<int> unclustered(N);
    for (int i = 0; i < N; ++i) unclustered[i] = i;

    // Exact squared threshold: the smallest double whose square root is not
    // below `threshold`, so that sqrt(d2) < threshold <=> d2 < thrSq.
    double thrSq = threshold * threshold;
    if (std::sqrt(thrSq) >= threshold) {
        while (std::sqrt(std::nextafter(thrSq, 0.0)) >= threshold) {
            thrSq = std::nextafter(thrSq, 0.0);
        }
    } else {
        do {
            thrSq = std::nextafter(thrSq, std::numeric_limits<double>::infinity());
        } while (std::sqrt(thrSq) < threshold);
    }

    u64 thrSqBits;
    memcpy(&thrSqBits, &thrSq, sizeof(u64));

    double2* const hpts = g.h_pts;
    int* const hcard = g.h_card;
    unsigned* const hmask = g.h_mask;

    int nu = N;
    while (nu > 0) {
        // Every rank evaluates every nranks-th candidate seed. The round
        // robin distribution balances the load much better than contiguous
        // blocks would, because the point indices are spatially correlated.
        const int nseeds = (nu - rank + nranks - 1) / nranks;

        struct { int card; int seed; } local = {-1, INT_MAX}, best;
        int nWarpsTotal = 0;

        if (nseeds > 0) {
            // Compacted coordinates of the unclustered points. The host side
            // O(nu) work is only worth threading for large problems -
            // otherwise the parallel region costs more than the loop itself.
            const int gatherThreads = ompTeam(nu);
#pragma omp parallel for schedule(static) num_threads(gatherThreads) if (nu >= OMP_MIN_WORK)
            for (int i = 0; i < nu; ++i) {
                const Point& p = points[unclustered[i]];
                hpts[i] = make_double2(p.x, p.y);
            }
            // Asynchronous: the blocking result copy below orders the upload
            // against the next modification of the staging buffer.
            CUDA_CHECK(cudaMemcpyAsync(g.d_pts, hpts, static_cast<size_t>(nu) * sizeof(double2),
                                       cudaMemcpyHostToDevice));

            const int blocks = std::min((nseeds + WPB - 1) / WPB, g.nblocks);
            nWarpsTotal = blocks * WPB;
            qtGrowKernel<<<blocks, TPB>>>(g.d_pts, nu, thrSqBits, rank, nranks, nseeds, N,
                                          g.nslots, g.d_act, g.d_md, g.d_card, g.d_maskCur,
                                          g.d_maskBest, g.maskStride);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(hcard, g.d_card, static_cast<size_t>(nseeds) * sizeof(int),
                                  cudaMemcpyDeviceToHost));

            int bc = -1, bs = INT_MAX;
            const int reduceThreads = ompTeam(nseeds);
#pragma omp parallel num_threads(reduceThreads) if (nseeds >= OMP_MIN_WORK)
            {
                int lc = -1, ls = INT_MAX;
#pragma omp for schedule(static) nowait
                for (int i = 0; i < nseeds; ++i) {
                    if (hcard[i] > lc) {  // strict: keeps the lowest index on ties
                        lc = hcard[i];
                        ls = i;
                    }
                }
#pragma omp critical
                {
                    if (lc > bc || (lc == bc && ls < bs)) {
                        bc = lc;
                        bs = ls;
                    }
                }
            }
            local.card = bc;
            local.seed = rank + bs * nranks;
        }

        // Largest cardinality wins, lowest seed index on ties.
        MPI_Allreduce(&local, &best, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);
        if (best.card <= 0) break;  // no more clusters can be formed

        // The winner is the best seed of the warp slot that processed it, so
        // its membership bitmask is still on the owning rank's GPU. Fetch it
        // there and share it, all ranks track the clustering state.
        const int nwords = (nu + 31) / 32;
        const int root = best.seed % nranks;
        if (rank == root) {
            const int slot = (best.seed / nranks) % nWarpsTotal;
            CUDA_CHECK(cudaMemcpy(hmask, g.d_maskBest + static_cast<size_t>(slot) * g.maskStride,
                                  static_cast<size_t>(nwords) * sizeof(unsigned),
                                  cudaMemcpyDeviceToHost));
        }
        if (nranks > 1) {
            MPI_Bcast(hmask, nwords, MPI_UNSIGNED, root, MPI_COMM_WORLD);
        }

        // Expand the bitmask: the members and the compacted list of the still
        // unclustered points are both extracted in one pass.
        Cluster cluster;
        cluster.seed_point = unclustered[best.seed];
        cluster.members.resize(best.card);
        int nmem = 0, w = 0;
        for (int b = 0; b < nwords; ++b) {
            unsigned bits = hmask[b];
            const int lim = std::min(32, nu - 32 * b);
            for (int k = 0; k < lim; ++k) {
                const int i = 32 * b + k;
                if (bits & 1u) {
                    cluster.members[nmem++] = unclustered[i];
                } else {
                    unclustered[w++] = unclustered[i];
                }
                bits >>= 1;
            }
        }
        if (nmem != best.card) {
            fprintf(stderr, "internal error: member count mismatch (%d != %d)\n", nmem, best.card);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        nu = w;
        clusters.push_back(std::move(cluster));
    }

    return clusters;
}

// ---------------------------------------------------------------------------
// Validation: check that clusters satisfy the QT clustering properties
// ---------------------------------------------------------------------------

bool validateClusters(const std::vector<Cluster>& clusters, const std::vector<Point>& points,
                      const double threshold, const int rank, const int nranks) {
    bool valid = true;
    const int nc = static_cast<int>(clusters.size());

    // The O(k^2) diameter computations are spread over ranks and threads.
    std::vector<double> diameters(nc, 0.0);
#pragma omp parallel for schedule(dynamic)
    for (int c = rank; c < nc; c += nranks) {
        const auto& members = clusters[c].members;
        const int n = static_cast<int>(members.size());
        double max_diameter = 0.0;
        for (int i = 0; i < n; ++i) {
            const Point pi = points[members[i]];
            for (int j = i + 1; j < n; ++j) {
                const double dist = distance(pi, points[members[j]]);
                max_diameter = std::max(max_diameter, dist);
            }
        }
        diameters[c] = max_diameter;
    }
    if (nc > 0) {
        MPI_Allreduce(MPI_IN_PLACE, diameters.data(), nc, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        printf("Validating clusters:\n");

        for (int c = 0; c < nc; ++c) {
            const double max_diameter = diameters[c];

            if (c < 10) {  // Print first 10 clusters
                printf("  Cluster %d: size=%zu, seed=%d, diameter=%.4f\n", c,
                       clusters[c].members.size(), clusters[c].seed_point, max_diameter);
            }

            // Validate diameter is within threshold
            if (max_diameter > threshold * 1.001) {  // Allow small numerical error
                printf("ERROR: Cluster %d has diameter %.4f > threshold %.4f\n", c, max_diameter,
                       threshold);
                valid = false;
            }
        }

        // Check for duplicate memberships
        std::vector<int> membership(points.size(), -1);
        for (int c = 0; c < nc; ++c) {
            for (size_t i = 0; i < clusters[c].members.size(); ++i) {
                const int member = clusters[c].members[i];
                if (membership[member] >= 0) {
                    printf("ERROR: Point %d appears in multiple clusters (%d and %d)\n", member,
                           membership[member], c);
                    valid = false;
                }
                membership[member] = c;
            }
        }

        // Count clustered points
        int clustered_count = 0;
        for (size_t i = 0; i < membership.size(); ++i) {
            if (membership[i] >= 0) clustered_count++;
        }

        printf("Total points: %zu, Clustered: %d, Unclustered: %zu\n", points.size(),
               clustered_count, points.size() - clustered_count);
    }

    // All ranks must agree on the exit status.
    int flag = valid ? 1 : 0;
    MPI_Bcast(&flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return flag != 0;
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

    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

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
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (num_points <= 0 || threshold <= 0.0) {
        if (rank == 0) {
            printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n", num_points,
                   threshold);
        }
        MPI_Finalize();
        return 1;
    }

    // One GPU per rank; ranks sharing a node are distributed round robin.
    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm);
    int local_rank = 0, local_size = 1;
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_size(local_comm, &local_size);
    MPI_Comm_free(&local_comm);

    int ndev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    if (ndev == 0) {
        if (rank == 0) fprintf(stderr, "Error: no CUDA device available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(local_rank % ndev));
    const int ranksPerDevice = (local_size + ndev - 1) / ndev;

    if (rank == 0) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, OpenMP threads/rank: %d, GPUs/node: %d\n", nranks,
               omp_get_max_threads(), ndev);
    }

    // Generate synthetic data (identical on every rank)
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);

    GpuState gpu;
    gpuInit(gpu, num_points, ranksPerDevice);
    CUDA_CHECK(cudaFree(0));  // make sure the context is fully up before timing

    // Perform QT clustering
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_start = std::chrono::high_resolution_clock::now();

    const std::vector<Cluster> clusters = qtClustering(points, threshold, rank, nranks, gpu);

    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(cluster_end - cluster_start);

    if (rank == 0) {
        printf("Clustering time: %ld ms\n", cluster_time.count());
        printf("Clusters found: %zu\n", clusters.size());
    }

    // Calculate statistics and performance metrics
    int total_clustered = 0;
    int max_cluster_size = 0;

    for (size_t i = 0; i < clusters.size(); ++i) {
        const int size = static_cast<int>(clusters[i].members.size());
        total_clustered += size;
        max_cluster_size = std::max(max_cluster_size, size);
    }

    const double avg_cluster_size =
        clusters.empty() ? 0.0 : static_cast<double>(total_clustered) / clusters.size();

    if (rank == 0) {
        printf("Points clustered: %d / %d (%.1f%%)\n", total_clustered, num_points,
               100.0 * total_clustered / num_points);
        printf("Average cluster size: %.2f\n", avg_cluster_size);
        printf("Maximum cluster size: %d\n", max_cluster_size);

        // Performance metrics
        const double time_sec = cluster_time.count() / 1000.0;
        const double clusters_per_sec = clusters.size() / time_sec;
        const double points_per_sec = num_points / time_sec;
        printf("Performance: %.1f clusters/s, %.1f points/s\n", clusters_per_sec, points_per_sec);
    }

    // Print results for external validation
    if (printResults && rank == 0) {
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

    gpuFree(gpu);

    // Validation
    if (validate) {
        const bool valid = validateClusters(clusters, points, threshold, rank, nranks);

        if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        MPI_Finalize();
        return valid ? 0 : 1;
    }

    MPI_Finalize();
    return 0;
}
