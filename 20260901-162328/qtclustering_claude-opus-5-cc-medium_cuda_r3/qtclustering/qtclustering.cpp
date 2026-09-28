// QT Clustering Benchmark - CUDA Version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// GPU parallelization strategy:
//   * One warp grows the candidate cluster of one seed point, and the warps
//     pull seeds from a dynamic queue, so a single kernel launch evaluates all
//     candidate seeds of a clustering round concurrently.
//   * A warp maintains for every candidate the running maximum distance to the
//     cluster members, which makes a growth step O(candidates) instead of
//     O(candidates * cluster_size), and picks the minimizing point with a warp
//     argmin reduction that reproduces the sequential tie-breaking rules.
//   * Points whose maximum distance exceeded the threshold can never become
//     usable again (the maximum only grows), so the warp keeps a compacted
//     list of the still-viable candidates, which shrinks quickly.
//   * The points are bucketed into a uniform grid of threshold-sized cells, so
//     a cluster only ever has to look at the 3x3 cells around its seed.
//   * A cluster cannot outgrow its current members plus its surviving
//     candidates; seeds that cannot reach the best cardinality of the round
//     are abandoned, which does not change the result.
//   * Selecting the winning seed of a round and marking its members happens on
//     the GPU as well, leaving one small transfer per round.
//
// All comparisons are performed on squared distances, which is bit-exact with
// respect to the original: sqrt is monotonic, so the running maximum and the
// threshold test carry over unchanged, and the only place where the two domains
// can differ - two distances that collapse onto the same double - is detected
// and resolved on true distances.

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

// Error checking helper for CUDA API calls
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        const cudaError_t err_ = (call);                                        \
        if (err_ != cudaSuccess) {                                              \
            printf("CUDA error %s at %s:%d: %s\n", cudaGetErrorName(err_),      \
                   __FILE__, __LINE__, cudaGetErrorString(err_));               \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

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
// CUDA candidate cluster generation
// ---------------------------------------------------------------------------

static const int BLOCK_SIZE = 256;
static const int WARPS_PER_BLOCK = BLOCK_SIZE / 32;

// Keep the "smallest maximum distance, lowest index on ties" rule of the
// sequential code: a candidate replaces the incumbent only on a strictly
// smaller distance, and the scan order of the original loop means the lowest
// original point index wins whenever two candidates are exactly equal.
// `slot` is the position in the cell ordering, `id` the original point index.
__device__ __forceinline__ void argMinUpdate(double& v, int& slot, int& id,
                                             const double v2, const int slot2,
                                             const int id2) {
    if (v2 < v || (v2 == v && id2 < id && id2 >= 0)) {
        v = v2;
        slot = slot2;
        id = id2;
    }
}

// Butterfly reduction: every lane ends up with the warp-wide argmin
__device__ __forceinline__ void warpArgMin(double& v, int& slot, int& id) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        const double v2 = __shfl_xor_sync(0xFFFFFFFFu, v, offset);
        const int slot2 = __shfl_xor_sync(0xFFFFFFFFu, slot, offset);
        const int id2 = __shfl_xor_sync(0xFFFFFFFFu, id, offset);
        argMinUpdate(v, slot, id, v2, slot2, id2);
    }
}

// Squared Euclidean distance. All bookkeeping happens on squared distances,
// which is exact with respect to the original code: sqrt is monotonic, so
// max(sqrt(a), sqrt(b)) == sqrt(max(a, b)) bit for bit, and the threshold test
// uses the exact squared boundary computed on the host.
__device__ __forceinline__ double gpuDistance2(const double2& p1, const double2& p2) {
    const double dx = p1.x - p2.x;
    const double dy = p1.y - p2.y;
    return dx * dx + dy * dy;
}

// Two squared distances can round to the same double under sqrt only if they
// are within about two units in the last place of each other. Non-negative
// doubles are ordered like their bit patterns, so the (generous) test is a
// cheap integer comparison of the representations.
static const long long SQRT_TIE_ULPS = 8;

// Exactly equal squared distances are already handled by the index tie-break,
// so only distinct-but-adjacent representations need the exact fallback.
__device__ __forceinline__ bool nearTie(const double a, const double b) {
    const long long diff = llabs(__double_as_longlong(a) - __double_as_longlong(b));
    return diff != 0 && diff <= SQRT_TIE_ULPS;
}

// Exact tie resolution, only entered when a near-tie was detected: redo the
// argmin of the surviving candidates on true distances.
__device__ __forceinline__ void resolveTies(const int* __restrict__ list,
                                            const int count,
                                            const double* __restrict__ max_dist,
                                            const int* __restrict__ ids,
                                            int& best_slot, int& best_id) {
    const double INF = __longlong_as_double(0x7FF0000000000000LL);
    double best_r = INF;
    int slot_r = -1, id_r = -1;
    for (int t = (threadIdx.x & 31); t < count; t += 32) {
        const int slot = list[t];
        argMinUpdate(best_r, slot_r, id_r, sqrt(max_dist[slot]), slot, ids[slot]);
    }
    warpArgMin(best_r, slot_r, id_r);
    best_slot = slot_r;
    best_id = id_r;
}

// Grows the candidate cluster of one seed per warp. Warps loop over the seed
// list, so a single kernel launch evaluates every candidate seed of a
// clustering round. Each warp reports the largest cluster it produced together
// with the corresponding member list, which is all the host needs.
__global__ __launch_bounds__(BLOCK_SIZE) void qtCandidateClusterKernel(
    const double2* __restrict__ points,        // reordered by grid cell
    const int* __restrict__ ids,               // original index of a slot
    const unsigned char* __restrict__ clustered,
    const int* __restrict__ cell_start,        // prefix sums of cell sizes
    const int* __restrict__ seed_slots,
    const int num_seeds,
    int* __restrict__ work_counter,   // dynamic seed queue
    volatile int* global_max,         // largest completed cardinality so far
    const double threshold2,   // exact squared threshold boundary
    const int point_count,
    const int grid_w,
    const int grid_h,
    const double cell_size,
    const double min_x,
    const double min_y,
    double* __restrict__ scratch_dist,
    int* __restrict__ scratch_list,
    int* __restrict__ scratch_members,
    int* __restrict__ warp_best) {
    const double INF = __longlong_as_double(0x7FF0000000000000LL);
    const int lane = threadIdx.x & 31;
    const unsigned lane_lt = (1u << lane) - 1u;
    const int gwarp = (blockIdx.x * BLOCK_SIZE + threadIdx.x) >> 5;

    // Per-warp scratch, all indexed by slot (position in the cell ordering):
    // running maximum distance to the cluster, two candidate list buffers and
    // two member list buffers (current seed / best seed so far).
    double* max_dist = scratch_dist + static_cast<size_t>(gwarp) * point_count;
    int* list_cur = scratch_list + static_cast<size_t>(gwarp) * 2 * point_count;
    int* list_next = list_cur + point_count;
    int* members[2] = {scratch_members + static_cast<size_t>(gwarp) * 2 * point_count,
                       scratch_members + static_cast<size_t>(gwarp) * 2 * point_count +
                           point_count};
    int cur_buf = 0;
    int best_buf = 1;

    int best_card = 0;
    int best_seed = -1;

    for (;;) {
        // Grab the next seed; a dynamic queue keeps the warps balanced when the
        // cost per seed varies strongly with the local point density.
        int s;
        if (lane == 0) s = atomicAdd(work_counter, 1);
        s = __shfl_sync(0xFFFFFFFFu, s, 0);
        if (s >= num_seeds) break;

        const int seed_slot = seed_slots[s];
        const int seed_id = ids[seed_slot];
        const double2 sp = points[seed_slot];

        // First growth step: only points in the 3x3 cell neighbourhood of the
        // seed can be closer than the threshold. Cells of one row are
        // contiguous, so this is three contiguous slot ranges.
        int scx = static_cast<int>((sp.x - min_x) / cell_size);
        int scy = static_cast<int>((sp.y - min_y) / cell_size);
        scx = min(max(scx, 0), grid_w - 1);
        scy = min(max(scy, 0), grid_h - 1);
        const int cx0 = max(scx - 1, 0);
        const int cx1 = min(scx + 1, grid_w - 1);

        int count = 0;
        double best_m2 = INF;   // smallest squared distance found so far
        int best_slot = -1;
        int best_id = -1;
        bool tie_hint = false;  // set when two candidates are nearly equal

        for (int r = -1; r <= 1; ++r) {
            const int cy = scy + r;
            if (cy < 0 || cy >= grid_h) continue;
            const int begin = cell_start[cy * grid_w + cx0];
            const int span = cell_start[cy * grid_w + cx1 + 1] - begin;
            for (int t = lane; t < ((span + 31) & ~31); t += 32) {
                bool keep = false;
                int slot = 0, id = -1;
                double d2 = 0.0;
                if (t < span) {
                    slot = begin + t;
                    if (!clustered[slot] && slot != seed_slot) {
                        d2 = gpuDistance2(points[slot], sp);
                        if (d2 < threshold2) {
                            keep = true;
                            id = ids[slot];
                        }
                    }
                }
                if (keep) {
                    max_dist[slot] = d2;
                    tie_hint |= nearTie(d2, best_m2);
                    argMinUpdate(best_m2, best_slot, best_id, d2, slot, id);
                }
                const unsigned mask = __ballot_sync(0xFFFFFFFFu, keep);
                if (keep) list_cur[count + __popc(mask & lane_lt)] = slot;
                count += __popc(mask);
            }
        }
        {
            double reduced = best_m2;
            warpArgMin(reduced, best_slot, best_id);
            if (__any_sync(0xFFFFFFFFu, tie_hint || nearTie(best_m2, reduced))) {
                resolveTies(list_cur, count, max_dist, ids, best_slot, best_id);
            }
        }

        int cardinality = 1;
        if (lane == 0) members[cur_buf][0] = seed_id;

        // The cluster can never grow beyond the current members plus the
        // surviving candidates, so a seed that cannot reach the largest
        // cardinality found so far cannot win this round and is dropped.
        bool pruned = cardinality + count < *global_max;

        // Iteratively add the closest point that keeps the diameter below the
        // threshold. A point whose maximum distance once exceeded the threshold
        // can never qualify again, so it is dropped from the candidate list.
        while (best_slot >= 0 && !pruned) {
            const int last = best_slot;
            if (lane == 0) members[cur_buf][cardinality] = best_id;
            cardinality++;

            const double2 lp = points[last];
            const int active = count;
            count = 0;
            best_m2 = INF;
            best_slot = -1;
            best_id = -1;
            tie_hint = false;
            for (int t = lane; t < ((active + 31) & ~31); t += 32) {
                bool keep = false;
                int slot = 0, id = -1;
                double v2 = 0.0;
                if (t < active) {
                    slot = list_cur[t];
                    if (slot != last) {
                        v2 = max_dist[slot];
                        const double d2 = gpuDistance2(points[slot], lp);
                        if (d2 > v2) v2 = d2;
                        if (v2 < threshold2) {
                            keep = true;
                            id = ids[slot];
                        }
                    }
                }
                if (keep) {
                    max_dist[slot] = v2;
                    tie_hint |= nearTie(v2, best_m2);
                    argMinUpdate(best_m2, best_slot, best_id, v2, slot, id);
                }
                const unsigned mask = __ballot_sync(0xFFFFFFFFu, keep);
                if (keep) list_next[count + __popc(mask & lane_lt)] = slot;
                count += __popc(mask);
            }
            {
                double reduced = best_m2;
                warpArgMin(reduced, best_slot, best_id);
                if (__any_sync(0xFFFFFFFFu, tie_hint || nearTie(best_m2, reduced))) {
                    resolveTies(list_next, count, max_dist, ids, best_slot, best_id);
                }
            }

            int* tmp = list_cur;
            list_cur = list_next;
            list_next = tmp;

            pruned = cardinality + count < *global_max;
        }

        if (pruned) continue;

        // Publish the cardinality so that other warps can prune, and keep the
        // largest cluster of this warp (ties keep the lower seed index).
        if (lane == 0) atomicMax(const_cast<int*>(global_max), cardinality);
        if (cardinality > best_card || (cardinality == best_card && seed_id < best_seed)) {
            best_card = cardinality;
            best_seed = seed_id;
            const int tmp = best_buf;
            best_buf = cur_buf;
            cur_buf = tmp;
        }
    }

    if (lane == 0) {
        warp_best[3 * gwarp + 0] = best_card;
        warp_best[3 * gwarp + 1] = best_seed;
        warp_best[3 * gwarp + 2] = best_buf;
    }
}

// Picks the winning seed of a round exactly like the sequential loop (largest
// cardinality, lowest seed index on ties), marks its members as clustered and
// publishes the result, so that the host needs a single small transfer.
__global__ __launch_bounds__(BLOCK_SIZE) void qtSelectWinnerKernel(
    const int* __restrict__ warp_best,
    const int num_warps,
    const int* __restrict__ scratch_members,
    const int point_count,
    const int* __restrict__ slot_of,
    int* __restrict__ work_counter,
    int* __restrict__ global_max,
    int* __restrict__ result) {
    __shared__ int s_card[BLOCK_SIZE];
    __shared__ int s_seed[BLOCK_SIZE];
    __shared__ int s_warp[BLOCK_SIZE];
    const int tid = threadIdx.x;

    int card = 0, seed = -1, warp = -1;
    for (int w = tid; w < num_warps; w += BLOCK_SIZE) {
        const int c = warp_best[3 * w];
        const int sd = warp_best[3 * w + 1];
        if (sd < 0) continue;
        if (c > card || (c == card && sd < seed)) {
            card = c;
            seed = sd;
            warp = w;
        }
    }
    s_card[tid] = card;
    s_seed[tid] = seed;
    s_warp[tid] = warp;
    __syncthreads();
    for (int offset = BLOCK_SIZE / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            const int c = s_card[tid + offset];
            const int sd = s_seed[tid + offset];
            if (sd >= 0 && (c > s_card[tid] || (c == s_card[tid] && sd < s_seed[tid]))) {
                s_card[tid] = c;
                s_seed[tid] = sd;
                s_warp[tid] = s_warp[tid + offset];
            }
        }
        __syncthreads();
    }

    card = s_card[0];
    warp = s_warp[0];
    if (tid == 0) {
        result[0] = card;
        result[1] = s_seed[0];
        *work_counter = 0;   // reset for the next round
        *global_max = 0;
    }
    if (warp < 0) return;

    // Mark the members of the winning cluster (result[2...] holds the flags)
    const int buf = warp_best[3 * warp + 2];
    const int* __restrict__ members =
        scratch_members + (static_cast<size_t>(warp) * 2 + buf) * point_count;
    unsigned char* clustered = reinterpret_cast<unsigned char*>(result + 2);
    for (int i = tid; i < card; i += BLOCK_SIZE) {
        clustered[slot_of[members[i]]] = 1;
    }
}

// Uniform grid over the points with a cell size of at least the threshold, so
// that every point closer than the threshold lies in the 3x3 cell neighbourhood
static const int MAX_GRID_CELLS = 1 << 22;

// Device-side state of the clustering run, allocated once per invocation
struct GpuContext {
    void* memory = nullptr;      // single allocation backing everything below
    double2* points = nullptr;
    int* ids = nullptr;
    unsigned char* clustered = nullptr;
    int* cell_start = nullptr;
    int* seed_slots = nullptr;
    int* slot_of = nullptr;
    int* warp_best = nullptr;
    int* work_counter = nullptr;
    int* global_max = nullptr;
    int* result = nullptr;
    double* scratch_dist = nullptr;
    int* scratch_list = nullptr;
    int* scratch_members = nullptr;
    int max_blocks = 0;
};

// Main QT clustering algorithm (GPU accelerated)
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<Cluster> clusters;
    if (N == 0) return clusters;

    // Smallest squared distance whose square root reaches the threshold, so
    // that "squared distance < threshold2" is exactly "distance < threshold"
    double threshold2 = threshold * threshold;
    if (std::sqrt(threshold2) >= threshold) {
        while (std::sqrt(std::nextafter(threshold2, 0.0)) >= threshold) {
            threshold2 = std::nextafter(threshold2, 0.0);
        }
    } else {
        do {
            threshold2 = std::nextafter(threshold2, std::numeric_limits<double>::max());
        } while (std::sqrt(threshold2) < threshold);
    }

    // ---- Build the uniform grid and reorder the points by cell ----
    double min_x = points[0].x, max_x = points[0].x;
    double min_y = points[0].y, max_y = points[0].y;
    for (int i = 1; i < N; ++i) {
        min_x = std::min(min_x, points[i].x);
        max_x = std::max(max_x, points[i].x);
        min_y = std::min(min_y, points[i].y);
        max_y = std::max(max_y, points[i].y);
    }
    const double extent_x = std::max(max_x - min_x, 1e-12);
    const double extent_y = std::max(max_y - min_y, 1e-12);
    double cell_size = threshold;
    // Keep the number of cells bounded for very small thresholds
    while (static_cast<double>(std::floor(extent_x / cell_size) + 1) *
               (std::floor(extent_y / cell_size) + 1) > MAX_GRID_CELLS) {
        cell_size *= 2.0;
    }
    const int grid_w = static_cast<int>(extent_x / cell_size) + 1;
    const int grid_h = static_cast<int>(extent_y / cell_size) + 1;
    const int num_cells = grid_w * grid_h;

    std::vector<int> cell_of(N);
    std::vector<int> cell_start(num_cells + 1, 0);
    for (int i = 0; i < N; ++i) {
        int cx = static_cast<int>((points[i].x - min_x) / cell_size);
        int cy = static_cast<int>((points[i].y - min_y) / cell_size);
        cx = std::min(std::max(cx, 0), grid_w - 1);
        cy = std::min(std::max(cy, 0), grid_h - 1);
        cell_of[i] = cy * grid_w + cx;
        cell_start[cell_of[i] + 1]++;
    }
    for (int c = 0; c < num_cells; ++c) cell_start[c + 1] += cell_start[c];

    // Counting sort of the points by cell (stable, so slots of a cell keep
    // ascending original indices)
    std::vector<double2> host_points(N);
    std::vector<int> host_ids(N);
    std::vector<int> slot_of(N);
    {
        std::vector<int> fill(cell_start.begin(), cell_start.end() - 1);
        for (int i = 0; i < N; ++i) {
            const int slot = fill[cell_of[i]]++;
            host_points[slot] = make_double2(points[i].x, points[i].y);
            host_ids[slot] = i;
            slot_of[i] = slot;
        }
    }

    // ---- Device setup ----
    int sm_count = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0));
    size_t free_mem = 0, total_mem = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    // Scratch per warp: distances, two candidate lists, two member lists
    const size_t bytes_per_warp =
        static_cast<size_t>(N) * (sizeof(double) + 4 * sizeof(int));
    const size_t budget = (free_mem > (128u << 20)) ? (free_mem - (128u << 20)) : 0;
    size_t blocks_by_mem = budget / (bytes_per_warp * WARPS_PER_BLOCK);
    if (blocks_by_mem < 1) blocks_by_mem = 1;

    GpuContext ctx;
    const int blocks_for_all_seeds = (N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    ctx.max_blocks = static_cast<int>(std::min<size_t>(
        {static_cast<size_t>(blocks_for_all_seeds),
         static_cast<size_t>(sm_count) * 8, blocks_by_mem}));
    const size_t max_warps = static_cast<size_t>(ctx.max_blocks) * WARPS_PER_BLOCK;

    // One allocation for everything, carved up in 256 byte aligned chunks
    size_t offset = 0;
    auto reserve = [&offset](size_t bytes) {
        const size_t at = offset;
        offset += (bytes + 255) & ~static_cast<size_t>(255);
        return at;
    };
    const size_t off_scratch_dist = reserve(sizeof(double) * N * max_warps);
    const size_t off_scratch_list = reserve(sizeof(int) * 2 * N * max_warps);
    const size_t off_scratch_members = reserve(sizeof(int) * 2 * N * max_warps);
    const size_t off_points = reserve(sizeof(double2) * N);
    const size_t off_ids = reserve(sizeof(int) * N);
    const size_t off_slot_of = reserve(sizeof(int) * N);
    const size_t off_cell_start = reserve(sizeof(int) * (num_cells + 1));
    const size_t off_seed_slots = reserve(sizeof(int) * N);
    const size_t off_warp_best = reserve(sizeof(int) * 3 * max_warps);
    const size_t off_counters = reserve(2 * sizeof(int));
    // Round result: cardinality, seed and the clustered flags, transferred in one go
    const size_t result_bytes = 2 * sizeof(int) + N;
    const size_t off_result = reserve(result_bytes);

    CUDA_CHECK(cudaMalloc(&ctx.memory, offset));
    char* base = static_cast<char*>(ctx.memory);
    ctx.scratch_dist = reinterpret_cast<double*>(base + off_scratch_dist);
    ctx.scratch_list = reinterpret_cast<int*>(base + off_scratch_list);
    ctx.scratch_members = reinterpret_cast<int*>(base + off_scratch_members);
    ctx.points = reinterpret_cast<double2*>(base + off_points);
    ctx.ids = reinterpret_cast<int*>(base + off_ids);
    ctx.slot_of = reinterpret_cast<int*>(base + off_slot_of);
    ctx.cell_start = reinterpret_cast<int*>(base + off_cell_start);
    ctx.seed_slots = reinterpret_cast<int*>(base + off_seed_slots);
    ctx.warp_best = reinterpret_cast<int*>(base + off_warp_best);
    ctx.work_counter = reinterpret_cast<int*>(base + off_counters);
    ctx.global_max = ctx.work_counter + 1;
    ctx.result = reinterpret_cast<int*>(base + off_result);
    ctx.clustered = reinterpret_cast<unsigned char*>(ctx.result + 2);

    CUDA_CHECK(cudaMemcpy(ctx.points, host_points.data(), sizeof(double2) * N,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx.ids, host_ids.data(), sizeof(int) * N,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx.slot_of, slot_of.data(), sizeof(int) * N,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx.cell_start, cell_start.data(),
                          sizeof(int) * (num_cells + 1), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(ctx.result, 0, result_bytes));
    CUDA_CHECK(cudaMemset(ctx.work_counter, 0, 2 * sizeof(int)));

    // Unclustered points in ascending order; the seed order does not influence
    // the result, which is decided by cardinality and point index alone.
    std::vector<int> unclustered_indices(N);
    for (int i = 0; i < N; ++i) unclustered_indices[i] = i;

    // Host mirror of the round result: [cardinality, seed, clustered flags]
    std::vector<int> seed_slots(N);
    std::vector<int> result_buffer(2 + (N + sizeof(int) - 1) / sizeof(int));
    int* result = result_buffer.data();
    const unsigned char* host_clustered =
        reinterpret_cast<const unsigned char*>(result + 2);
    std::vector<int> remaining;
    remaining.reserve(N);

    // Main clustering loop
    while (!unclustered_indices.empty()) {
        const int num_seeds = static_cast<int>(unclustered_indices.size());
        for (int i = 0; i < num_seeds; ++i) {
            seed_slots[i] = slot_of[unclustered_indices[i]];
        }
        CUDA_CHECK(cudaMemcpy(ctx.seed_slots, seed_slots.data(), sizeof(int) * num_seeds,
                              cudaMemcpyHostToDevice));

        // Try each unclustered point as a seed, one warp per seed
        const int blocks = std::min((num_seeds + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK,
                                    ctx.max_blocks);
        const int warps = blocks * WARPS_PER_BLOCK;
        qtCandidateClusterKernel<<<blocks, BLOCK_SIZE>>>(
            ctx.points, ctx.ids, ctx.clustered, ctx.cell_start, ctx.seed_slots,
            num_seeds, ctx.work_counter, ctx.global_max, threshold2, N, grid_w, grid_h,
            cell_size, min_x, min_y,
            ctx.scratch_dist, ctx.scratch_list, ctx.scratch_members, ctx.warp_best);
        // Keep the largest candidate cluster and update the clustered flags
        qtSelectWinnerKernel<<<1, BLOCK_SIZE>>>(ctx.warp_best, warps, ctx.scratch_members,
                                                N, ctx.slot_of, ctx.work_counter,
                                                ctx.global_max, ctx.result);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(result, ctx.result, result_bytes,
                              cudaMemcpyDeviceToHost));

        const int max_cardinality = result[0];
        const int best_seed = result[1];
        if (best_seed < 0 || max_cardinality <= 0) {
            // No more clusters can be formed
            break;
        }

        // The newly flagged points form the cluster; the rest stay as seeds
        Cluster cluster;
        cluster.seed_point = best_seed;
        cluster.members.reserve(max_cardinality);
        remaining.clear();
        for (int i = 0; i < num_seeds; ++i) {
            const int idx = unclustered_indices[i];
            if (host_clustered[seed_slots[i]]) {
                cluster.members.push_back(idx);
            } else {
                remaining.push_back(idx);
            }
        }
        clusters.push_back(std::move(cluster));
        unclustered_indices.swap(remaining);
    }

    cudaFree(ctx.memory);

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

    // Create the CUDA context up front so that one-time driver initialization
    // is not attributed to the clustering time
    CUDA_CHECK(cudaSetDevice(0));
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
