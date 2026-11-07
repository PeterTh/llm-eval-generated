// QT Clustering Benchmark - Hybrid MPI + OpenMP + CUDA Version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// Parallelization strategy (semantics identical to the sequential version):
//   * Every outer iteration evaluates one candidate cluster per unclustered
//     seed. Seeds are distributed round-robin across MPI ranks (one GPU per
//     rank); the globally best seed (max cardinality, smallest index on ties)
//     is selected with an MPI_MAXLOC all-reduce.
//   * On each rank, a CUDA kernel grows candidate clusters: one thread block
//     per seed, the block cooperatively maintains the running "max distance to
//     the current members" for every unclustered point and performs a block
//     wide arg-min each growth step (same tie-breaking as the scalar code:
//     smallest index among the minimal values).
//   * OpenMP is used for the host-side work (compaction of the active point
//     set, bookkeeping, statistics and validation).
//
// The squared distance is evaluated as fma(dx, dx, dy * dy) on the device,
// which is the exact operation sequence GCC emits for the scalar reference
// (-O3 -march=native) before its sqrt; the kernel works on squared distances
// and only evaluates the (correctly rounded) sqrt where it is needed to
// reproduce the reference ordering bit for bit (see the kernel section).

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

// ---------------------------------------------------------------------------
// Error handling helpers
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err__ = (call);                                               \
        if (err__ != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err__),\
                    __FILE__, __LINE__);                                          \
            MPI_Abort(MPI_COMM_WORLD, 1);                                         \
        }                                                                         \
    } while (0)

#define MPI_CHECK(call)                                                           \
    do {                                                                          \
        int err__ = (call);                                                       \
        if (err__ != MPI_SUCCESS) {                                               \
            fprintf(stderr, "MPI error %d at %s:%d\n", err__, __FILE__, __LINE__);\
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

// Calculate Euclidean distance between two points (host, validation only)
inline double distance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

// ---------------------------------------------------------------------------
// CUDA kernels
// ---------------------------------------------------------------------------
//
// All comparisons of the reference implementation are done on the rounded
// Euclidean distance d = sqrt_rn(q) with q = fma(dx, dx, dy*dy). Because
// sqrt_rn is monotone non-decreasing,
//   * max_i sqrt(q_i)     == sqrt(max_i q_i)                      (exact)
//   * sqrt(q) < threshold <=> q < Qt, Qt = min{ q : sqrt(q) >= t } (exact)
// so the running "max distance to the members" can be kept squared, and the
// expensive FP64 sqrt is only needed to order candidates exactly like the
// reference, i.e. by the pair (sqrt(q), index). The sqrt is evaluated only
// when a thread's running minimum improves (candidates are visited in
// ascending index order, so a larger q can never beat the current best).

constexpr int QT_BLOCK = 256;
constexpr int QT_WARPS = QT_BLOCK / 32;
constexpr int QT_COMPACT_EVERY = 32;   // live-set re-compaction period (steps)

__device__ __forceinline__ double dev_dist2(double ax, double ay, double bx, double by) {
    const double dx = ax - bx;
    const double dy = ay - by;
    // Same contraction as the host compiler: fma(dx, dx, dy*dy)
    return fma(dx, dx, dy * dy);
}

// Ordered (value, index) min: smaller value wins, ties -> smaller index
__device__ __forceinline__ void take_min(double& v, int& i, double ov, int oi) {
    if (ov < v || (ov == v && oi < i)) {
        v = ov;
        i = oi;
    }
}

__device__ __forceinline__ void warp_argmin(double& v, int& i) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const double ov = __shfl_xor_sync(0xffffffffu, v, off);
        const int oi = __shfl_xor_sync(0xffffffffu, i, off);
        take_min(v, i, ov, oi);
    }
}

// Block-wide arg-min; result broadcast to all threads.
__device__ __forceinline__ void block_argmin(double& v, int& i, double* s_val, int* s_idx) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    warp_argmin(v, i);
    if (lane == 0) {
        s_val[warp] = v;
        s_idx[warp] = i;
    }
    __syncthreads();
    if (warp == 0) {
        double wv = (lane < QT_WARPS) ? s_val[lane] : INFINITY;
        int wi = (lane < QT_WARPS) ? s_idx[lane] : INT_MAX;
        warp_argmin(wv, wi);
        if (lane == 0) {
            s_val[0] = wv;
            s_idx[0] = wi;
        }
    }
    __syncthreads();
    v = s_val[0];
    i = s_idx[0];
    __syncthreads();
}

// Candidate bookkeeping for one thread: (sqrt(q), q, index) of its best entry.
struct LocalBest {
    double s = INFINITY;
    double q = INFINITY;
    int j = INT_MAX;

    // Entries must be offered in ascending index order.
    __device__ __forceinline__ void offer(double qv, int jj, double Qt) {
        if (qv < Qt && qv < q) {
            const double sv = sqrt(qv);
            if (sv < s) {
                s = sv;
                j = jj;
            }
            q = qv;   // sv == s otherwise: keep the smaller index, tighten q
        }
    }
};

// Order-preserving in-place compaction of the live entries (q < Qt).
__device__ __forceinline__ void compact_live(double* __restrict__ q, int* __restrict__ idx,
                                             int& n, const double Qt, int* s_cnt) {
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const unsigned lanemask_lt = (1u << lane) - 1u;
    __syncthreads();
    int total = 0;
    for (int base = 0; base < n; base += QT_BLOCK) {
        const int e = base + tid;
        const bool live = (e < n) && (q[e] < Qt);
        const double qv = live ? q[e] : 0.0;
        const int jv = live ? idx[e] : 0;
        const unsigned m = __ballot_sync(0xffffffffu, live);
        if (lane == 0) s_cnt[warp] = __popc(m);
        __syncthreads();
        int woff = 0, chunk_total = 0;
#pragma unroll
        for (int w = 0; w < QT_WARPS; ++w) {
            const int c = s_cnt[w];
            if (w < warp) woff += c;
            chunk_total += c;
        }
        __syncthreads();
        if (live) {
            const int p = total + woff + __popc(m & lanemask_lt);
            q[p] = qv;
            idx[p] = jv;
        }
        total += chunk_total;
    }
    __syncthreads();
    n = total;
}

// Grow candidate clusters. One block per seed (grid-stride over seeds).
//   pts      : coordinates of the U currently unclustered points (compact)
//   seeds    : S compact indices to use as seeds
//   Qt       : squared-distance threshold boundary (see above)
//   gq/gidx  : gridDim.x * U fallback scratch when the live set exceeds cap
//   cap      : live-set capacity of the dynamic shared memory
//   card     : output cardinality per seed
//   members  : if RECORD, the member list (compact indices) of seeds[0]
template <bool RECORD>
__global__ void __launch_bounds__(QT_BLOCK)
qt_candidate_kernel(const double2* __restrict__ pts,
                    const int U,
                    const int* __restrict__ seeds,
                    const int S,
                    const double Qt,
                    double* __restrict__ gq,
                    int* __restrict__ gidx,
                    const int cap,
                    int* __restrict__ card,
                    int* __restrict__ members)
{
    extern __shared__ double dyn_smem[];
    double* sq = dyn_smem;
    int* sidx = reinterpret_cast<int*>(sq + cap);
    __shared__ double s_val[QT_WARPS];
    __shared__ int s_idx[QT_WARPS];
    __shared__ int s_cnt[QT_WARPS];
    __shared__ int s_total;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const unsigned lanemask_lt = (1u << lane) - 1u;

    // Warp w scans the contiguous index range [jbeg, jend) so that the
    // compacted live list is globally sorted by index.
    const int chunk = (((U + QT_WARPS - 1) / QT_WARPS) + 31) & ~31;
    const int jbeg = min(U, warp * chunk);
    const int jend = min(U, jbeg + chunk);

    for (int si = blockIdx.x; si < S; si += gridDim.x) {
        const int seed = seeds[si];
        const double2 sp = pts[seed];

        // Phase 1: count points within the threshold of the seed (only those
        // can ever join the cluster).
        int wcount = 0;
        for (int base = jbeg; base < jend; base += 32) {
            const int j = base + lane;
            bool live = false;
            if (j < jend && j != seed) {
                const double2 p = pts[j];
                live = dev_dist2(p.x, p.y, sp.x, sp.y) < Qt;
            }
            wcount += __popc(__ballot_sync(0xffffffffu, live));
        }
        if (lane == 0) s_cnt[warp] = wcount;
        __syncthreads();
        if (tid == 0) {
            int acc = 0;
            for (int w = 0; w < QT_WARPS; ++w) {
                const int c = s_cnt[w];
                s_cnt[w] = acc;
                acc += c;
            }
            s_total = acc;
        }
        __syncthreads();
        int live_n = s_total;
        const bool use_smem = live_n <= cap;
        double* __restrict__ q = use_smem ? sq : gq + static_cast<size_t>(blockIdx.x) * U;
        int* __restrict__ idx = use_smem ? sidx : gidx + static_cast<size_t>(blockIdx.x) * U;

        // Phase 2: write the compacted (q, index) live list, sorted by index.
        {
            int pos = s_cnt[warp];
            for (int base = jbeg; base < jend; base += 32) {
                const int j = base + lane;
                bool live = false;
                double qv = 0.0;
                if (j < jend && j != seed) {
                    const double2 p = pts[j];
                    qv = dev_dist2(p.x, p.y, sp.x, sp.y);
                    live = qv < Qt;
                }
                const unsigned m = __ballot_sync(0xffffffffu, live);
                if (live) {
                    const int pp = pos + __popc(m & lanemask_lt);
                    q[pp] = qv;
                    idx[pp] = j;
                }
                pos += __popc(m);
            }
        }
        __syncthreads();

        // Phase 3: greedy growth.
        LocalBest lb;
        for (int e = tid; e < live_n; e += QT_BLOCK) lb.offer(q[e], idx[e], Qt);

        int count = 1;
        if (RECORD && tid == 0) members[0] = seed;
        int since_compact = 0;

        while (true) {
            double v = lb.s;
            int bi = lb.j;
            block_argmin(v, bi, s_val, s_idx);
            if (bi == INT_MAX) break;  // no admissible point left
            const int best = bi;
            if (RECORD && tid == 0) members[count] = best;
            ++count;

            const double2 bp = pts[best];
            lb = LocalBest();
            for (int e = tid; e < live_n; e += QT_BLOCK) {
                double qv = q[e];
                const int j = idx[e];
                if (j == best) {
                    qv = INFINITY;
                } else if (qv != INFINITY) {
                    const double2 p = pts[j];
                    const double nq = dev_dist2(p.x, p.y, bp.x, bp.y);
                    qv = (qv < nq) ? nq : qv;  // std::max semantics
                }
                q[e] = qv;
                lb.offer(qv, j, Qt);
            }

            if (++since_compact == QT_COMPACT_EVERY) {
                since_compact = 0;
                compact_live(q, idx, live_n, Qt, s_cnt);
            }
        }

        if (tid == 0) card[si] = count;
        __syncthreads();
    }
}

// Smallest q such that sqrt_rn(q) >= t  (so sqrt(q) < t  <=>  q < Qt).
static double squared_threshold_boundary(const double t) {
    volatile double Qt = t * t;
    while (std::sqrt(Qt) >= t) Qt = std::nextafter(Qt, 0.0);
    while (std::sqrt(Qt) < t) Qt = std::nextafter(Qt, std::numeric_limits<double>::infinity());
    return Qt;
}

// ---------------------------------------------------------------------------
// Device resources held across the outer iterations
// ---------------------------------------------------------------------------
struct DeviceContext {
    int device = 0;
    int max_blocks = 1;        // resident blocks for the candidate kernel
    int cap = 0;               // live-set entries that fit in shared memory
    int smem_bytes = 0;
    double2* d_pts = nullptr;  // N
    int* d_seeds = nullptr;    // N
    int* d_card = nullptr;     // N
    int* d_members = nullptr;  // N
    double* d_gq = nullptr;    // fallback scratch
    int* d_gidx = nullptr;
    size_t scratch_elems = 0;
    cudaStream_t stream = nullptr;

    void init(int dev, int N) {
        device = dev;
        CUDA_CHECK(cudaSetDevice(device));
        CUDA_CHECK(cudaFree(0));
        CUDA_CHECK(cudaStreamCreate(&stream));

        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

        // Shared-memory budget: aim for ~6 resident blocks per SM.
        cudaFuncAttributes fa;
        CUDA_CHECK(cudaFuncGetAttributes(&fa, qt_candidate_kernel<false>));
        const int target_blocks = 6;
        long budget = static_cast<long>(prop.sharedMemPerMultiprocessor) / target_blocks
                      - static_cast<long>(prop.reservedSharedMemPerBlock)
                      - static_cast<long>(fa.sharedSizeBytes);
        budget = std::min(budget, static_cast<long>(prop.sharedMemPerBlockOptin)
                                  - static_cast<long>(fa.sharedSizeBytes));
        cap = static_cast<int>(std::max(0L, budget) / (sizeof(double) + sizeof(int)));
        cap = std::max(cap, 32);
        cap &= ~1;   // keep the int array 8-byte aligned
        smem_bytes = cap * static_cast<int>(sizeof(double) + sizeof(int));
        CUDA_CHECK(cudaFuncSetAttribute(qt_candidate_kernel<false>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
        CUDA_CHECK(cudaFuncSetAttribute(qt_candidate_kernel<true>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));

        int per_sm = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &per_sm, qt_candidate_kernel<false>, QT_BLOCK, smem_bytes));
        if (per_sm < 1) per_sm = 1;
        max_blocks = std::max(1, per_sm * prop.multiProcessorCount);

        CUDA_CHECK(cudaMalloc(&d_pts, sizeof(double2) * N));
        CUDA_CHECK(cudaMalloc(&d_seeds, sizeof(int) * N));
        CUDA_CHECK(cudaMalloc(&d_card, sizeof(int) * N));
        CUDA_CHECK(cudaMalloc(&d_members, sizeof(int) * N));
    }

    void ensure_scratch(size_t elems) {
        if (elems > scratch_elems) {
            if (d_gq) CUDA_CHECK(cudaFree(d_gq));
            if (d_gidx) CUDA_CHECK(cudaFree(d_gidx));
            CUDA_CHECK(cudaMalloc(&d_gq, sizeof(double) * elems));
            CUDA_CHECK(cudaMalloc(&d_gidx, sizeof(int) * elems));
            scratch_elems = elems;
        }
    }

    void destroy() {
        if (d_pts) cudaFree(d_pts);
        if (d_seeds) cudaFree(d_seeds);
        if (d_card) cudaFree(d_card);
        if (d_members) cudaFree(d_members);
        if (d_gq) cudaFree(d_gq);
        if (d_gidx) cudaFree(d_gidx);
        if (stream) cudaStreamDestroy(stream);
    }
};

// Main QT clustering algorithm (hybrid MPI + OpenMP + CUDA).
// Every rank returns the identical cluster list.
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold,
                                  DeviceContext& ctx,
                                  MPI_Comm comm) {
    int rank = 0, nranks = 1;
    MPI_CHECK(MPI_Comm_rank(comm, &rank));
    MPI_CHECK(MPI_Comm_size(comm, &nranks));

    const int N = static_cast<int>(points.size());
    std::vector<char> clustered(N, 0);
    std::vector<int> unclustered_indices(N);   // always ascending
    std::vector<Cluster> clusters;

    for (int i = 0; i < N; ++i) unclustered_indices[i] = i;

    std::vector<double2> h_pts(N);
    std::vector<int> h_seeds(N);
    std::vector<int> h_card(N);
    std::vector<int> h_members(N);

    CUDA_CHECK(cudaSetDevice(ctx.device));
    const double Qt = squared_threshold_boundary(threshold);

    // Main clustering loop
    while (!unclustered_indices.empty()) {
        const int U = static_cast<int>(unclustered_indices.size());

        // Compact coordinates of the active points (OpenMP)
        #pragma omp parallel for schedule(static) if (U >= 8192)
        for (int j = 0; j < U; ++j) {
            const Point& p = points[unclustered_indices[j]];
            h_pts[j] = make_double2(p.x, p.y);
        }

        // Round-robin seed distribution over ranks (balances uneven seed cost)
        int S = 0;
        for (int j = rank; j < U; j += nranks) h_seeds[S++] = j;

        CUDA_CHECK(cudaMemcpyAsync(ctx.d_pts, h_pts.data(), sizeof(double2) * U,
                                   cudaMemcpyHostToDevice, ctx.stream));

        int local_best[2] = {-1, INT_MAX};   // (cardinality, compact index)
        if (S > 0) {
            const int blocks = std::min(S, ctx.max_blocks);
            ctx.ensure_scratch(static_cast<size_t>(blocks) * U);
            CUDA_CHECK(cudaMemcpyAsync(ctx.d_seeds, h_seeds.data(), sizeof(int) * S,
                                       cudaMemcpyHostToDevice, ctx.stream));
            qt_candidate_kernel<false><<<blocks, QT_BLOCK, ctx.smem_bytes, ctx.stream>>>(
                ctx.d_pts, U, ctx.d_seeds, S, Qt, ctx.d_gq, ctx.d_gidx, ctx.cap,
                ctx.d_card, nullptr);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpyAsync(h_card.data(), ctx.d_card, sizeof(int) * S,
                                       cudaMemcpyDeviceToHost, ctx.stream));
            CUDA_CHECK(cudaStreamSynchronize(ctx.stream));

            // Local winner: max cardinality, smallest compact index on ties
            // (h_seeds is ascending, so first strict improvement wins).
            for (int s = 0; s < S; ++s) {
                if (h_card[s] > local_best[0]) {
                    local_best[0] = h_card[s];
                    local_best[1] = h_seeds[s];
                }
            }
        } else {
            CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
        }

        // Global winner: MAXLOC picks the max cardinality and, on ties, the
        // smallest compact index -> identical choice to the sequential scan.
        int global_best[2];
        MPI_CHECK(MPI_Allreduce(local_best, global_best, 1, MPI_2INT, MPI_MAXLOC, comm));

        const int best_card = global_best[0];
        const int best_compact = global_best[1];
        if (best_card <= 0 || best_compact == INT_MAX) break;  // cannot happen, safety

        // Re-grow the winning cluster (redundantly on every rank) to obtain
        // its member list in insertion order. Cost: one seed.
        {
            ctx.ensure_scratch(static_cast<size_t>(U));
            CUDA_CHECK(cudaMemcpyAsync(ctx.d_seeds, &best_compact, sizeof(int),
                                       cudaMemcpyHostToDevice, ctx.stream));
            qt_candidate_kernel<true><<<1, QT_BLOCK, ctx.smem_bytes, ctx.stream>>>(
                ctx.d_pts, U, ctx.d_seeds, 1, Qt, ctx.d_gq, ctx.d_gidx, ctx.cap,
                ctx.d_card, ctx.d_members);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpyAsync(h_members.data(), ctx.d_members,
                                       sizeof(int) * best_card,
                                       cudaMemcpyDeviceToHost, ctx.stream));
            CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
        }

        Cluster cluster;
        cluster.seed_point = unclustered_indices[best_compact];
        cluster.members.resize(best_card);
        for (int i = 0; i < best_card; ++i) {
            const int g = unclustered_indices[h_members[i]];
            cluster.members[i] = g;
            clustered[g] = 1;
        }
        clusters.push_back(std::move(cluster));

        // Remove clustered points from unclustered list (keeps ascending order)
        unclustered_indices.erase(
            std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                          [&clustered](int idx) { return clustered[idx] != 0; }),
            unclustered_indices.end()
        );
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
    #pragma omp parallel for reduction(+:clustered_count)
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
    int provided = 0;
    MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));
    int rank = 0, nranks = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &nranks));
    const bool is_root = (rank == 0);

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
            if (is_root) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (is_root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    
    if (num_points <= 0 || threshold <= 0.0) {
        if (is_root) {
            printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n", 
                   num_points, threshold);
        }
        MPI_Finalize();
        return 1;
    }
    
    // Select the GPU: ranks sharing a node are spread over the local devices
    MPI_Comm node_comm;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                                  MPI_INFO_NULL, &node_comm));
    int local_rank = 0, local_size = 1;
    MPI_CHECK(MPI_Comm_rank(node_comm, &local_rank));
    MPI_CHECK(MPI_Comm_size(node_comm, &local_size));
    MPI_CHECK(MPI_Comm_free(&node_comm));

    // Unless the user pinned the OpenMP team size, share the node's cores
    // between the ranks on that node (avoids oversubscription when several
    // ranks each spawn a full-size team).
    if (getenv("OMP_NUM_THREADS") == nullptr) {
        const int hw = omp_get_num_procs();
        omp_set_num_threads(std::max(1, hw / std::max(1, local_size)));
    }
    int num_devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&num_devices));
    if (num_devices <= 0) {
        fprintf(stderr, "Error: no CUDA device available on rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (is_root) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, OpenMP threads/rank: %d, CUDA devices/node: %d\n",
               nranks, omp_get_max_threads(), num_devices);
    }
    
    // Generate synthetic data (deterministic, replicated on every rank)
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);

    DeviceContext ctx;
    ctx.init(local_rank % num_devices, num_points);
    
    // Perform QT clustering
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto cluster_start = std::chrono::high_resolution_clock::now();
    
    const std::vector<Cluster> clusters = qtClustering(points, threshold, ctx, MPI_COMM_WORLD);
    
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);

    ctx.destroy();

    int ret = 0;
    if (is_root) {
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
                ret = 1;
            }
        }
        fflush(stdout);
    }

    MPI_CHECK(MPI_Bcast(&ret, 1, MPI_INT, 0, MPI_COMM_WORLD));
    MPI_Finalize();
    return ret;
}
