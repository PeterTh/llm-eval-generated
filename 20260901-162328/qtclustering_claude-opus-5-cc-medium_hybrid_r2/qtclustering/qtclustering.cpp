// QT Clustering Benchmark - Hybrid MPI + OpenMP + CUDA Version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// Parallelization strategy:
//   * MPI   - the candidate seeds of every outer iteration are statically
//             partitioned (cyclically) across the ranks; each rank owns the
//             cached candidate clusters of its seeds.  Only a MAXLOC
//             all-reduce plus the broadcast of the winning cluster is needed
//             per outer iteration.
//   * CUDA  - growing the candidate cluster of a seed is executed by one CUDA
//             block per seed, so thousands of seeds are grown concurrently.
//   * OpenMP- the host side data parallel phases (point marshalling, unpacking
//             of the member lists, cache invalidation, membership assembly and
//             the validation pass) run on the cores of the node.  They are only
//             handed to a team when the amount of work justifies it, because
//             idle OpenMP threads measurably lengthen the CUDA synchronisation
//             latency of the tightly coupled outer iteration.
//
// Algorithmic notes (semantics preserving):
//   * The maximum distance of a candidate to the members of the growing
//     cluster is maintained incrementally instead of being recomputed from
//     scratch (max over the same set of values, hence identical).
//   * All distance work is carried out on *squared* distances.  sqrt() is
//     monotone, so the arg-min is unchanged; the threshold predicate
//     sqrt(d2) < threshold is turned into d2 <= t2max where t2max is the
//     largest double whose square root is still below the threshold, which
//     makes the predicate bit-exactly equivalent.
//   * The candidate cluster of a seed is cached and only recomputed when the
//     cluster that was just committed intersects it.  Removing points that
//     were never selected cannot change any arg-min decision, so the cached
//     result stays exact.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

static const double MAX_WIDTH = 20.0;
static const double MAX_HEIGHT = 20.0;

// Maximum CUDA block size used by the growth/collect kernels
#define QTC_MAX_BLOCK 1024
// Smallest amount of host work (in visited elements) that is worth the cost of
// starting an OpenMP team; below it the sequential path is faster and, more
// importantly, does not add latency to the CUDA synchronisations.
#define QTC_OMP_MIN_WORK (1 << 22)
// Upper bound on the number of concurrently grown seeds (CUDA blocks)
#define QTC_MAX_BLOCKS 8192

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                         \
                    cudaGetErrorString(err_), __FILE__, __LINE__);              \
            MPI_Abort(MPI_COMM_WORLD, 1);                                       \
        }                                                                       \
    } while (0)

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

// Largest double t2 for which (sqrt(d2) < threshold) <=> (d2 <= t2).
// sqrt() is correctly rounded and therefore monotone, so such a value exists.
static double squaredThreshold(const double threshold) {
    double t2 = threshold * threshold;
    while (t2 > 0.0 && std::sqrt(t2) >= threshold) {
        t2 = std::nextafter(t2, 0.0);
    }
    while (std::sqrt(std::nextafter(t2, std::numeric_limits<double>::infinity())) < threshold) {
        t2 = std::nextafter(t2, std::numeric_limits<double>::infinity());
    }
    return t2;
}

// ---------------------------------------------------------------------------
// CUDA kernels
// ---------------------------------------------------------------------------

// Ordering used by the arg-min reduction: smallest value wins, ties are broken
// towards the smaller point index (identical to the sequential scan order).
__device__ __forceinline__ bool qtBetter(double av, int ai, double bv, int bi) {
    if (ai < 0) return false;
    if (bi < 0) return true;
    return (av < bv) || (av == bv && ai < bi);
}

// Warp aggregated stream compaction: every lane with keep==true appends its
// value to `out` using a single atomic per warp.
__device__ __forceinline__ void qtAppend(int* __restrict__ out, int* counter,
                                         const bool keep, const int value,
                                         const int lane) {
    const unsigned mask = __ballot_sync(0xffffffffu, keep);
    if (mask == 0u) return;
    const int leader = __ffs(mask) - 1;
    int base = 0;
    if (lane == leader) base = atomicAdd(counter, __popc(mask));
    base = __shfl_sync(0xffffffffu, base, leader);
    if (keep) out[base + __popc(mask & ((1u << lane) - 1u))] = value;
}

// One block grows the candidate cluster of one seed.
//
// dwork holds a row of N running squared maximum distances per block; a value
// of HUGE_VAL marks a point that is a member of the growing cluster or that was
// already clustered globally.  actwork holds two index lists of N entries per
// block that carry the candidates which can still join the cluster.  Because
// the running maximum only ever grows, a candidate that once exceeded the
// threshold can never become feasible again and is dropped from the list, which
// shrinks the per-step scan from N to the local neighbourhood of the cluster.
__global__ __launch_bounds__(QTC_MAX_BLOCK)
void qtGrowKernel(const double* __restrict__ px,
                  const double* __restrict__ py,
                  const unsigned char* __restrict__ clustered,
                  const int* __restrict__ seeds,
                  const int numSeeds, const int N, const double t2max,
                  double* __restrict__ dwork, int* __restrict__ actwork,
                  int* __restrict__ card) {
    const int b = blockIdx.x;
    if (b >= numSeeds) return;

    const int tid = threadIdx.x;
    const int nt = blockDim.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nwarps = nt >> 5;

    double* __restrict__ d = dwork + static_cast<size_t>(b) * N;
    int* cur_list = actwork + static_cast<size_t>(b) * 2 * N;
    int* nxt_list = cur_list + N;

    __shared__ double sv[QTC_MAX_BLOCK / 32];
    __shared__ int si[QTC_MAX_BLOCK / 32];
    __shared__ int s_next;
    __shared__ int s_keep;

    const int seed = seeds[b];

    // Initialise the distance row and the active candidate list
    if (tid == 0) s_keep = 0;
    __syncthreads();
    {
        const int nround = ((N + nt - 1) / nt) * nt;
        for (int j = tid; j < nround; j += nt) {
            bool keep = false;
            if (j < N) {
                const bool cl = clustered[j] != 0;
                d[j] = cl ? HUGE_VAL : 0.0;
                keep = !cl && j != seed;
            }
            qtAppend(cur_list, &s_keep, keep, j, lane);
        }
    }
    __syncthreads();
    int nact = s_keep;
    if (tid == 0) d[seed] = HUGE_VAL;
    __syncthreads();

    int cnt = 1;
    int cur = seed;

    while (cnt < N && nact > 0) {
        if (tid == 0) s_keep = 0;
        __syncthreads();

        const double mx = px[cur];
        const double my = py[cur];

        double bv = HUGE_VAL;
        int bi = -1;
        const int nround = ((nact + nt - 1) / nt) * nt;
        for (int j = tid; j < nround; j += nt) {
            bool keep = false;
            int i = -1;
            double v = HUGE_VAL;
            if (j < nact) {
                i = cur_list[j];
                v = d[i];
                const double dx = px[i] - mx;
                const double dy = py[i] - my;
                const double s = dx * dx + dy * dy;
                if (s > v) { v = s; d[i] = v; }
                keep = (v <= t2max);
            }
            qtAppend(nxt_list, &s_keep, keep, i, lane);
            if (keep && qtBetter(v, i, bv, bi)) { bv = v; bi = i; }
        }

        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const double ov = __shfl_down_sync(0xffffffffu, bv, off);
            const int oi = __shfl_down_sync(0xffffffffu, bi, off);
            if (qtBetter(ov, oi, bv, bi)) { bv = ov; bi = oi; }
        }
        if (lane == 0) { sv[warp] = bv; si[warp] = bi; }
        __syncthreads();
        if (warp == 0) {
            bv = (lane < nwarps) ? sv[lane] : HUGE_VAL;
            bi = (lane < nwarps) ? si[lane] : -1;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                const double ov = __shfl_down_sync(0xffffffffu, bv, off);
                const int oi = __shfl_down_sync(0xffffffffu, bi, off);
                if (qtBetter(ov, oi, bv, bi)) { bv = ov; bi = oi; }
            }
            if (lane == 0) s_next = bi;
        }
        __syncthreads();

        const int next = s_next;
        nact = s_keep;
        __syncthreads();  // everybody consumed s_next / s_keep

        if (next < 0) break;
        cur = next;
        ++cnt;
        if (tid == 0) d[cur] = HUGE_VAL;
        int* tmp = cur_list; cur_list = nxt_list; nxt_list = tmp;
    }

    if (tid == 0) card[b] = cnt;
}

// Compact the member lists (points marked HUGE_VAL that are not globally
// clustered) of every block into one contiguous output buffer.
__global__ __launch_bounds__(QTC_MAX_BLOCK)
void qtCollectKernel(const unsigned char* __restrict__ clustered,
                     const double* __restrict__ dwork,
                     const int numSeeds, const int N,
                     const int* __restrict__ offsets,
                     int* __restrict__ out) {
    const int b = blockIdx.x;
    if (b >= numSeeds) return;

    __shared__ int pos;
    if (threadIdx.x == 0) pos = offsets[b];
    __syncthreads();

    const double* __restrict__ d = dwork + static_cast<size_t>(b) * N;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        if (d[i] == HUGE_VAL && !clustered[i]) {
            out[atomicAdd(&pos, 1)] = i;
        }
    }
}

// ---------------------------------------------------------------------------
// GPU resource management
// ---------------------------------------------------------------------------

struct GpuContext {
    double* px = nullptr;
    double* py = nullptr;
    unsigned char* clustered = nullptr;
    int* seeds = nullptr;
    double* dwork = nullptr;
    int* actwork = nullptr;
    int* card = nullptr;
    int* offsets = nullptr;
    int* members = nullptr;
    size_t membersCap = 0;

    int* h_card = nullptr;
    int* h_offsets = nullptr;
    int* h_members = nullptr;
    size_t h_membersCap = 0;

    int maxBlocks = 0;
};

// Few concurrent seeds -> wide blocks, so that a single cluster is grown by as
// many threads as possible; many seeds -> narrow blocks for better occupancy.
static int gpuBlockSize(const int numSeeds, const int N) {
    int blk = 256;
    if (numSeeds <= 256) blk = 1024;
    else if (numSeeds <= 1024) blk = 512;
    while (blk > 32 && blk >= 2 * N) blk >>= 1;
    return blk;
}

static void gpuInit(GpuContext& g, const std::vector<double>& hx,
                    const std::vector<double>& hy, const int N,
                    const int ranksPerDevice) {
    size_t freeMem = 0, totalMem = 0;
    CUDA_CHECK(cudaMemGetInfo(&freeMem, &totalMem));

    // The growth kernel needs N doubles of scratch per concurrently grown seed.
    // Ranks that share a device initialise concurrently and all see the same
    // amount of free memory, so the budget has to be split between them.
    size_t budget = static_cast<size_t>(freeMem * 0.5) / (ranksPerDevice > 0 ? ranksPerDevice : 1);
    size_t perBlock = static_cast<size_t>(N) * (sizeof(double) + 2 * sizeof(int));
    int maxBlocks = static_cast<int>(std::min<size_t>(budget / perBlock, QTC_MAX_BLOCKS));
    if (maxBlocks < 1) maxBlocks = 1;
    if (maxBlocks > N) maxBlocks = N;
    // The member offsets of one batch are 32 bit, so keep the worst case
    // member count of a batch inside the int range.
    while (maxBlocks > 1 &&
           static_cast<long long>(maxBlocks) * N > 2147483647LL) {
        maxBlocks >>= 1;
    }
    g.maxBlocks = maxBlocks;

    CUDA_CHECK(cudaMalloc(&g.px, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&g.py, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&g.clustered, N * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&g.seeds, maxBlocks * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&g.dwork, static_cast<size_t>(maxBlocks) * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&g.actwork, static_cast<size_t>(maxBlocks) * 2 * N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&g.card, maxBlocks * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&g.offsets, maxBlocks * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(g.px, hx.data(), N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g.py, hy.data(), N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(g.clustered, 0, N * sizeof(unsigned char)));

    CUDA_CHECK(cudaMallocHost(&g.h_card, maxBlocks * sizeof(int)));
    CUDA_CHECK(cudaMallocHost(&g.h_offsets, maxBlocks * sizeof(int)));
}

static void gpuFree(GpuContext& g) {
    cudaFree(g.px);
    cudaFree(g.py);
    cudaFree(g.clustered);
    cudaFree(g.seeds);
    cudaFree(g.dwork);
    cudaFree(g.actwork);
    cudaFree(g.card);
    cudaFree(g.offsets);
    cudaFree(g.members);
    cudaFreeHost(g.h_card);
    cudaFreeHost(g.h_offsets);
    cudaFreeHost(g.h_members);
}

static void ensureMemberCapacity(GpuContext& g, const size_t need) {
    if (need <= g.membersCap) return;
    size_t cap = std::max<size_t>(need, g.membersCap * 2);
    cudaFree(g.members);
    g.members = nullptr;
    CUDA_CHECK(cudaMalloc(&g.members, cap * sizeof(int)));
    g.membersCap = cap;

    cudaFreeHost(g.h_members);
    g.h_members = nullptr;
    CUDA_CHECK(cudaMallocHost(&g.h_members, cap * sizeof(int)));
    g.h_membersCap = cap;
}

// ---------------------------------------------------------------------------
// Main QT clustering algorithm (hybrid MPI + OpenMP + CUDA)
// ---------------------------------------------------------------------------

std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold,
                                  const int rank, const int nranks,
                                  const int ranksPerDevice) {
    const int N = static_cast<int>(points.size());
    const double t2max = squaredThreshold(threshold);

    std::vector<double> hx(N), hy(N);
    #pragma omp parallel for schedule(static) if (N >= QTC_OMP_MIN_WORK)
    for (int i = 0; i < N; ++i) { hx[i] = points[i].x; hy[i] = points[i].y; }

    GpuContext gpu;
    gpuInit(gpu, hx, hy, N, ranksPerDevice);

    std::vector<unsigned char> clustered(N, 0);
    std::vector<unsigned char> removed(N, 0);
    std::vector<unsigned char> dirty(N, 0);
    std::vector<int> card(N, 0);
    std::vector<std::vector<int>> cache(N);

    // Cyclic ownership of the seeds keeps the (spatially clustered) recompute
    // sets evenly spread over the ranks.
    std::vector<int> owned;
    owned.reserve(N / nranks + 1);
    for (int i = rank; i < N; i += nranks) {
        owned.push_back(i);
        dirty[i] = 1;
    }

    const int nthreads = omp_get_max_threads();

    std::vector<Cluster> clusters;
    std::vector<int> todo;
    bool devMaskValid = true;   // device copy of `clustered` up to date
    int remaining = N;
    // Number of cached member entries the invalidation sweep has to look at.
    // Only when that is substantial does an OpenMP team pay for itself.
    long long cachedTotal = 0;

    while (remaining > 0) {
        // ---- 1. (Re)compute the candidate clusters of the dirty owned seeds
        todo.clear();
        for (size_t j = 0; j < owned.size(); ++j) {
            const int s = owned[j];
            if (dirty[s]) todo.push_back(s);
        }

        const int T = static_cast<int>(todo.size());
        if (T > 0) {
            if (!devMaskValid) {
                CUDA_CHECK(cudaMemcpy(gpu.clustered, clustered.data(),
                                      N * sizeof(unsigned char), cudaMemcpyHostToDevice));
                devMaskValid = true;
            }
            for (int s0 = 0; s0 < T; s0 += gpu.maxBlocks) {
                const int B = std::min(gpu.maxBlocks, T - s0);
                CUDA_CHECK(cudaMemcpy(gpu.seeds, todo.data() + s0, B * sizeof(int),
                                      cudaMemcpyHostToDevice));
                const int blk = gpuBlockSize(B, N);
                qtGrowKernel<<<B, blk>>>(gpu.px, gpu.py, gpu.clustered, gpu.seeds,
                                         B, N, t2max, gpu.dwork, gpu.actwork, gpu.card);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaMemcpy(gpu.h_card, gpu.card, B * sizeof(int),
                                      cudaMemcpyDeviceToHost));

                size_t total = 0;
                for (int b = 0; b < B; ++b) {
                    gpu.h_offsets[b] = static_cast<int>(total);
                    total += static_cast<size_t>(gpu.h_card[b]);
                }
                ensureMemberCapacity(gpu, total);
                CUDA_CHECK(cudaMemcpy(gpu.offsets, gpu.h_offsets, B * sizeof(int),
                                      cudaMemcpyHostToDevice));
                qtCollectKernel<<<B, blk>>>(gpu.clustered, gpu.dwork, B, N,
                                            gpu.offsets, gpu.members);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaMemcpy(gpu.h_members, gpu.members, total * sizeof(int),
                                      cudaMemcpyDeviceToHost));

                // Unpacking the member lists is pure host data movement
                #pragma omp parallel for schedule(static) num_threads(nthreads) \
                        if (total >= QTC_OMP_MIN_WORK)
                for (int b = 0; b < B; ++b) {
                    const int s = todo[s0 + b];
                    const int c = gpu.h_card[b];
                    card[s] = c;
                    cache[s].assign(gpu.h_members + gpu.h_offsets[b],
                                    gpu.h_members + gpu.h_offsets[b] + c);
                    dirty[s] = 0;
                }
            }
        }

        // ---- 2. Globally pick the seed with the largest cardinality.
        // MPI_MAXLOC returns the largest cardinality and, among equal ones, the
        // smallest point index - exactly the seed the sequential scan picks.
        int local[2] = {-1, N};
        cachedTotal = 0;
        for (size_t j = 0; j < owned.size(); ++j) {
            const int s = owned[j];
            cachedTotal += card[s];
            if (card[s] > local[0]) { local[0] = card[s]; local[1] = s; }
        }
        int best[2];
        MPI_Allreduce(local, best, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);
        const int best_card = best[0];
        const int best_seed = best[1];
        if (best_seed >= N || best_card <= 0) break;

        // ---- 3. Broadcast the winning cluster and commit it
        std::vector<int> mem;
        const int root = best_seed % nranks;
        if (rank == root) {
            mem = cache[best_seed];
        } else {
            mem.resize(best_card);
        }
        MPI_Bcast(mem.data(), best_card, MPI_INT, root, MPI_COMM_WORLD);

        for (int j = 0; j < best_card; ++j) {
            clustered[mem[j]] = 1;
            removed[mem[j]] = 1;
        }
        remaining -= best_card;
        devMaskValid = false;

        if (rank == 0) {
            Cluster c;
            c.seed_point = best_seed;
            c.members = mem;
            clusters.push_back(std::move(c));
        }

        // ---- 4. Invalidate the cached candidate clusters that intersect the
        //         cluster we just removed; everything else stays exact.
        const int M = static_cast<int>(owned.size());
        #pragma omp parallel for schedule(static) num_threads(nthreads) \
                if (cachedTotal >= QTC_OMP_MIN_WORK)
        for (int j = 0; j < M; ++j) {
            const int s = owned[j];
            if (clustered[s]) { cache[s].clear(); cache[s].shrink_to_fit(); card[s] = -1; continue; }
            if (dirty[s]) continue;
            const std::vector<int>& cm = cache[s];
            for (size_t k = 0; k < cm.size(); ++k) {
                if (removed[cm[k]]) { dirty[s] = 1; card[s] = -1; break; }
            }
        }
        for (int j = 0; j < best_card; ++j) removed[mem[j]] = 0;

        owned.erase(std::remove_if(owned.begin(), owned.end(),
                                   [&clustered](int idx) { return clustered[idx] != 0; }),
                    owned.end());
    }

    gpuFree(gpu);
    return clusters;
}

// Validation: check that clusters satisfy the QT clustering properties
bool validateClusters(const std::vector<Cluster>& clusters,
                     const std::vector<Point>& points,
                     const double threshold) {
    bool valid = true;

    printf("Validating clusters:\n");

    const int C = static_cast<int>(clusters.size());
    std::vector<double> diameters(C, 0.0);

    #pragma omp parallel for schedule(dynamic, 1)
    for (int c = 0; c < C; ++c) {
        const std::vector<int>& members = clusters[c].members;
        double max_diameter = 0.0;
        for (size_t i = 0; i < members.size(); ++i) {
            for (size_t j = i + 1; j < members.size(); ++j) {
                const double dist = distance(points[members[i]], points[members[j]]);
                max_diameter = std::max(max_diameter, dist);
            }
        }
        diameters[c] = max_diameter;
    }

    // Check each cluster
    for (int c = 0; c < C; ++c) {
        const double max_diameter = diameters[c];

        if (c < 10) { // Print first 10 clusters
            printf("  Cluster %d: size=%zu, seed=%d, diameter=%.4f\n",
                   c, clusters[c].members.size(), clusters[c].seed_point, max_diameter);
        }

        // Validate diameter is within threshold
        if (max_diameter > threshold * 1.001) { // Allow small numerical error
            printf("ERROR: Cluster %d has diameter %.4f > threshold %.4f\n",
                   c, max_diameter, threshold);
            valid = false;
        }
    }

    // Check for duplicate memberships
    std::vector<int> membership(points.size(), -1);
    for (int c = 0; c < C; ++c) {
        for (size_t i = 0; i < clusters[c].members.size(); ++i) {
            const int member = clusters[c].members[i];
            if (membership[member] >= 0) {
                printf("ERROR: Point %d appears in multiple clusters (%d and %d)\n",
                       member, membership[member], c);
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
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    // Bind one GPU per rank and share the cores of a node between the ranks
    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm);
    int local_rank = 0, local_size = 1;
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_size(local_comm, &local_size);

    int devCount = 0;
    if (cudaGetDeviceCount(&devCount) != cudaSuccess || devCount == 0) {
        if (rank == 0) fprintf(stderr, "Error: no CUDA device available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(local_rank % devCount));
    const int ranksPerDevice = (local_size + devCount - 1) / devCount;
    CUDA_CHECK(cudaFree(0));  // establish the context up front

    if (omp_get_max_threads() > 1) {
        const int cores = omp_get_num_procs();
        int t = cores / local_size;
        if (t < 1) t = 1;
        omp_set_num_threads(t);
    }

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
            printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
                   num_points, threshold);
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Generate synthetic data (deterministic, replicated on every rank)
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);

    // Perform QT clustering
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_start = std::chrono::high_resolution_clock::now();

    const std::vector<Cluster> clusters = qtClustering(points, threshold, rank, nranks, ranksPerDevice);

    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_end = std::chrono::high_resolution_clock::now();
    const long local_cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start).count();
    long global_cluster_time = 0;
    MPI_Reduce(&local_cluster_time, &global_cluster_time, 1, MPI_LONG, MPI_MAX, 0,
               MPI_COMM_WORLD);

    int exit_code = 0;

    if (rank == 0) {
        printf("Clustering time: %ld ms\n", global_cluster_time);
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
        const double time_sec = global_cluster_time / 1000.0;
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
    MPI_Comm_free(&local_comm);
    MPI_Finalize();
    return exit_code;
}
