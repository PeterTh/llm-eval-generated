// QT Clustering Benchmark - CUDA implementation
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
#include <climits>
#include <cfloat>
#include <cstdint>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>

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

namespace {

constexpr int BLOCK_SIZE = 128;
constexpr int WARP_SIZE = 32;
constexpr unsigned FULL_WARP = 0xffffffffu;

void cudaCheck(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", operation, cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}

// All allocations live for the entire clustering run; there are no allocations
// or host-side candidate searches inside the clustering loop.
template<class T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count = 0) : data_(nullptr) {
        if (count > std::numeric_limits<size_t>::max() / sizeof(T)) {
            fprintf(stderr, "CUDA allocation size overflow\n");
            std::exit(EXIT_FAILURE);
        }
        if (count) cudaCheck(cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)),
                             "cudaMalloc");
    }
    ~DeviceBuffer() { if (data_) cudaFree(data_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* get() const { return data_; }
    void release() {
        if (data_) cudaCheck(cudaFree(data_), "cudaFree");
        data_ = nullptr;
    }
private:
    T* data_;
};

__device__ __forceinline__ double gpuDistance(Point a, Point b) {
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    // Match the native host compiler's multiply-add, using a correctly rounded
    // double square root. Squared distances alone would change rounded ties.
    // The original std::max(0.0, distance) also ignores NaN distances.
    return fmax(0.0, __dsqrt_rn(__fma_rn(dx, dx, __dmul_rn(dy, dy))));
}

template<bool DENSE>
__device__ __forceinline__ double pairDistance(const Point* points,
                                              const double* distances,
                                              int n, int a, int b) {
    if constexpr (DENSE) return distances[static_cast<size_t>(a) * n + b];
    else return gpuDistance(points[a], points[b]);
}

// Compute each off-diagonal tile only once, and transpose it through shared
// memory so that both halves of the symmetric matrix have coalesced stores.
__global__ void buildDistances(const Point* points, double* distances, int n) {
    if (blockIdx.x < blockIdx.y) return;
    __shared__ Point row_points[32], column_points[32];
    __shared__ double tile[32][33];
    const int x = threadIdx.x;
    const int y = threadIdx.y;
    const int row_base = blockIdx.y * 32;
    const int column_base = blockIdx.x * 32;
    if (y == 0) {
        row_points[x] = row_base + x < n ? points[row_base + x] : Point{0.0, 0.0};
        column_points[x] = column_base + x < n ? points[column_base + x] : Point{0.0, 0.0};
    }
    __syncthreads();
    for (int r = y; r < 32; r += 8) {
        const double value = gpuDistance(row_points[r], column_points[x]);
        tile[r][x] = value;
        if (row_base + r < n && column_base + x < n)
            distances[static_cast<size_t>(row_base + r) * n + column_base + x] = value;
    }
    __syncthreads();
    if (blockIdx.x != blockIdx.y) {
        for (int r = y; r < 32; r += 8) {
            if (column_base + r < n && row_base + x < n)
                distances[static_cast<size_t>(column_base + r) * n + row_base + x] = tile[x][r];
        }
    }
}

template<bool DENSE>
__global__ void countNeighbors(const Point* points, const double* distances,
                               size_t* counts, int n, double threshold) {
    using Reduction = cub::BlockReduce<int, BLOCK_SIZE>;
    __shared__ typename Reduction::TempStorage storage;
    const int seed = blockIdx.x;
    int count = 0;
    for (int point = threadIdx.x; point < n; point += BLOCK_SIZE)
        count += point == seed || pairDistance<DENSE>(points, distances, n, seed, point) < threshold;
    const int total = Reduction(storage).Sum(count);
    if (threadIdx.x == 0) counts[seed] = total;
}

template<bool DENSE>
__global__ void buildNeighbors(const Point* points, const double* distances,
                               const size_t* offsets, int* neighbors,
                               int n, double threshold) {
    using Scan = cub::BlockScan<int, BLOCK_SIZE>;
    __shared__ typename Scan::TempStorage storage;
    __shared__ int cursor;
    const int seed = blockIdx.x;
    if (threadIdx.x == 0) cursor = 0;
    __syncthreads();
    for (int base = 0; base < n; base += BLOCK_SIZE) {
        const int point = base + threadIdx.x;
        const int keep = point < n && (point == seed ||
            pairDistance<DENSE>(points, distances, n, seed, point) < threshold);
        int prefix, total;
        Scan(storage).ExclusiveSum(keep, prefix, total);
        if (keep) neighbors[offsets[seed] + cursor + prefix] = point;
        __syncthreads();
        if (threadIdx.x == 0) cursor += total;
        __syncthreads();
    }
}

struct Choice {
    double diameter;
    int point;
};

struct MinimumChoice {
    __device__ __forceinline__ Choice operator()(Choice a, Choice b) const {
        return b.diameter < a.diameter ||
            (b.diameter == a.diameter && b.point < a.point) ? b : a;
    }
};

__device__ __forceinline__ unsigned long long clusterKey(int cardinality, int seed) {
    return (static_cast<unsigned long long>(cardinality) << 32) |
           static_cast<unsigned>(INT_MAX - seed);
}

// One warp builds one candidate. Its running maximum distances and point IDs
// stay in registers. The recurrence max(old_max, distance(new_member, point))
// is exactly the original scan over every member, without repeated work.
template<int ITEMS, bool DENSE>
__device__ int warpCandidate(int seed, const Point* points, const double* distances,
                             const size_t* offsets, const int* neighbors,
                             const unsigned char* clustered, int* cardinalities,
                             int* members, int n, double threshold) {
    const int lane = threadIdx.x % WARP_SIZE;
    if (clustered[seed]) return 0;
    const size_t begin = offsets[seed];
    const int degree = static_cast<int>(offsets[seed + 1] - begin);
    const int old_count = cardinalities[seed];
    if (old_count > 0) {
        bool invalid = false;
        for (int i = lane; i < old_count; i += WARP_SIZE)
            invalid |= clustered[members[begin + i]] != 0;
        // Removing points outside this candidate cannot change any greedy
        // choice. Only candidates that lost a member need to be rebuilt.
        if (!__any_sync(FULL_WARP, invalid)) return old_count;
    }

    int ids[ITEMS];
    double maxima[ITEMS];
    #pragma unroll
    for (int i = 0; i < ITEMS; ++i) {
        const int position = lane + i * WARP_SIZE;
        const int point = position < degree ? neighbors[begin + position] : -1;
        ids[i] = point;
        maxima[i] = point >= 0 && point != seed && !clustered[point]
            ? pairDistance<DENSE>(points, distances, n, seed, point) : DBL_MAX;
    }
    if (lane == 0) members[begin] = seed;
    int count = 1;
    while (true) {
        Choice best{DBL_MAX, INT_MAX};
        #pragma unroll
        for (int i = 0; i < ITEMS; ++i) {
            if (maxima[i] < DBL_MAX) best = MinimumChoice{}(best, Choice{maxima[i], ids[i]});
        }
        #pragma unroll
        for (int delta = 16; delta > 0; delta /= 2) {
            const Choice other{__shfl_down_sync(FULL_WARP, best.diameter, delta),
                               __shfl_down_sync(FULL_WARP, best.point, delta)};
            best = MinimumChoice{}(best, other);
        }
        const int closest = __shfl_sync(FULL_WARP, best.point, 0);
        if (closest == INT_MAX) break;
        if (lane == 0) members[begin + count] = closest;
        ++count;
        #pragma unroll
        for (int i = 0; i < ITEMS; ++i) {
            if (maxima[i] < DBL_MAX) {
                if (ids[i] == closest) maxima[i] = DBL_MAX;
                else {
                    const double d = pairDistance<DENSE>(points, distances, n, closest, ids[i]);
                    const double value = fmax(maxima[i], d);
                    maxima[i] = value < threshold ? value : DBL_MAX;
                }
            }
        }
    }
    if (lane == 0) cardinalities[seed] = count;
    return count;
}

template<int ITEMS, bool DENSE>
__global__ void generateWarpCandidates(const Point* points, const double* distances,
                                      const size_t* offsets, const int* neighbors,
                                      const unsigned char* clustered, int* cardinalities,
                                      int* members, unsigned long long* partial_best,
                                      int n, int seed_count, double threshold) {
    __shared__ unsigned long long warp_best[BLOCK_SIZE / WARP_SIZE];
    const int warp = threadIdx.x / WARP_SIZE;
    const int seed = blockIdx.x * (BLOCK_SIZE / WARP_SIZE) + warp;
    const int count = seed < seed_count
        ? warpCandidate<ITEMS, DENSE>(seed, points, distances, offsets, neighbors,
                                     clustered, cardinalities, members, n, threshold) : 0;
    if (threadIdx.x % WARP_SIZE == 0) warp_best[warp] = count ? clusterKey(count, seed) : 0;
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned long long best = 0;
        for (int i = 0; i < BLOCK_SIZE / WARP_SIZE; ++i) best = max(best, warp_best[i]);
        partial_best[blockIdx.x] = best;
    }
}

// Large neighborhoods use a whole block and shared memory, or global scratch
// when a row does not fit. This keeps register use bounded for arbitrary N.
template<bool DENSE, bool SHARED>
__device__ int blockCandidate(const Point* points, const double* distances,
                             const size_t* offsets, const int* neighbors,
                             const unsigned char* clustered, int* cardinalities,
                             int* members, double* scratch,
                             int n, int seed, double threshold) {
    using Reduction = cub::BlockReduce<Choice, BLOCK_SIZE>;
    __shared__ typename Reduction::TempStorage storage;
    __shared__ int closest;
    extern __shared__ double shared_maxima[];
    if (clustered[seed]) return 0;
    const size_t begin = offsets[seed];
    const int degree = static_cast<int>(offsets[seed + 1] - begin);
    const int old_count = cardinalities[seed];
    if (old_count > 0) {
        bool invalid = false;
        for (int i = threadIdx.x; i < old_count; i += BLOCK_SIZE)
            invalid |= clustered[members[begin + i]] != 0;
        if (!__syncthreads_or(invalid)) return old_count;
    }
    double* maxima = SHARED ? shared_maxima : scratch + begin;
    for (int i = threadIdx.x; i < degree; i += BLOCK_SIZE) {
        const int point = neighbors[begin + i];
        maxima[i] = point != seed && !clustered[point]
            ? pairDistance<DENSE>(points, distances, n, seed, point) : DBL_MAX;
    }
    if (threadIdx.x == 0) members[begin] = seed;
    int count = 1;
    while (true) {
        Choice best{DBL_MAX, INT_MAX};
        for (int i = threadIdx.x; i < degree; i += BLOCK_SIZE) {
            if (maxima[i] < DBL_MAX)
                best = MinimumChoice{}(best, Choice{maxima[i], neighbors[begin + i]});
        }
        best = Reduction(storage).Reduce(best, MinimumChoice{});
        if (threadIdx.x == 0) closest = best.point;
        __syncthreads();
        if (closest == INT_MAX) break;
        if (threadIdx.x == 0) members[begin + count] = closest;
        ++count;
        for (int i = threadIdx.x; i < degree; i += BLOCK_SIZE) {
            if (maxima[i] < DBL_MAX) {
                const int point = neighbors[begin + i];
                if (point == closest) maxima[i] = DBL_MAX;
                else {
                    const double d = pairDistance<DENSE>(points, distances, n, closest, point);
                    const double value = fmax(maxima[i], d);
                    maxima[i] = value < threshold ? value : DBL_MAX;
                }
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) cardinalities[seed] = count;
    return count;
}

template<bool DENSE, bool SHARED>
__global__ void generateBlockCandidates(const Point* points, const double* distances,
                                       const size_t* offsets, const int* neighbors,
                                       const unsigned char* clustered, int* cardinalities,
                                       int* members, double* scratch,
                                       unsigned long long* partial_best,
                                       int n, double threshold) {
    const int seed = blockIdx.x;
    const int count = blockCandidate<DENSE, SHARED>(points, distances, offsets, neighbors,
        clustered, cardinalities, members, scratch, n, seed, threshold);
    if (threadIdx.x == 0) partial_best[seed] = count ? clusterKey(count, seed) : 0;
}

__global__ void selectBest(const unsigned long long* partial_best,
                           unsigned long long* best, int count) {
    using Reduction = cub::BlockReduce<unsigned long long, 256>;
    __shared__ typename Reduction::TempStorage storage;
    unsigned long long value = 0;
    for (int i = threadIdx.x; i < count; i += blockDim.x) value = max(value, partial_best[i]);
    value = Reduction(storage).Reduce(value, cub::Max{});
    if (threadIdx.x == 0) *best = value;
}

__global__ void markClustered(unsigned char* clustered, const int* members, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += gridDim.x * blockDim.x)
        clustered[members[i]] = 1;
}

template<bool DENSE>
void launchCandidates(const Point* points, const double* distances, const size_t* offsets,
                      const int* neighbors, const unsigned char* clustered, int* cardinalities,
                      int* members, double* scratch, unsigned long long* partial_best,
                      int n, int seed_count, size_t maximum_degree, double threshold) {
    const int blocks = (seed_count + BLOCK_SIZE / WARP_SIZE - 1) / (BLOCK_SIZE / WARP_SIZE);
    #define LAUNCH_WARPS(ITEMS) \
        generateWarpCandidates<ITEMS, DENSE><<<blocks, BLOCK_SIZE>>>(points, distances, offsets, \
            neighbors, clustered, cardinalities, members, partial_best, n, seed_count, threshold)
    if (maximum_degree <= 32) { LAUNCH_WARPS(1); }
    else if (maximum_degree <= 64) { LAUNCH_WARPS(2); }
    else if (maximum_degree <= 128) { LAUNCH_WARPS(4); }
    else if (maximum_degree <= 256) { LAUNCH_WARPS(8); }
    else if (maximum_degree <= 512) { LAUNCH_WARPS(16); }
    else if (maximum_degree <= 2048)
        generateBlockCandidates<DENSE, true><<<seed_count, BLOCK_SIZE, maximum_degree * sizeof(double)>>>(
            points, distances, offsets, neighbors, clustered, cardinalities, members,
            scratch, partial_best, n, threshold);
    else
        generateBlockCandidates<DENSE, false><<<seed_count, BLOCK_SIZE>>>(
            points, distances, offsets, neighbors, clustered, cardinalities, members,
            scratch, partial_best, n, threshold);
    #undef LAUNCH_WARPS
    cudaCheck(cudaGetLastError(), "generate candidates");
}

struct RunState {
    int remaining;
    int cluster_count;
    int member_count;
    int stop;
};

// A resident cooperative grid performs all clustering rounds without launch
// overhead or host/device round trips. Its size is bounded by actual kernel
// occupancy so every block can reach the grid-wide barriers safely.
template<int ITEMS, bool DENSE, bool SHARED>
__global__ void persistentClustering(const Point* points, const double* distances,
                                    const size_t* offsets, const int* neighbors,
                                    unsigned char* clustered, int* cardinalities,
                                    int* members, double* scratch,
                                    unsigned long long* partial_best, RunState* state,
                                    int* output_members, int* output_sizes, int* output_seeds,
                                    int n, int seed_count, double threshold) {
    using Reduction = cub::BlockReduce<unsigned long long, BLOCK_SIZE>;
    __shared__ typename Reduction::TempStorage storage;
    __shared__ unsigned long long winner;
    const auto grid = cooperative_groups::this_grid();
    while (true) {
        unsigned long long local_best = 0;
        if constexpr (ITEMS > 0) {
            const int lane = threadIdx.x % WARP_SIZE;
            const int warp = blockIdx.x * (BLOCK_SIZE / WARP_SIZE) + threadIdx.x / WARP_SIZE;
            for (int seed = warp; seed < seed_count; seed += gridDim.x * (BLOCK_SIZE / WARP_SIZE)) {
                const int count = warpCandidate<ITEMS, DENSE>(seed, points, distances, offsets,
                    neighbors, clustered, cardinalities, members, n, threshold);
                if (lane == 0 && count) local_best = max(local_best, clusterKey(count, seed));
            }
        } else {
            for (int seed = blockIdx.x; seed < seed_count; seed += gridDim.x) {
                const int count = blockCandidate<DENSE, SHARED>(points, distances, offsets,
                    neighbors, clustered, cardinalities, members, scratch, n, seed, threshold);
                if (threadIdx.x == 0 && count) local_best = max(local_best, clusterKey(count, seed));
            }
        }
        const auto best = Reduction(storage).Reduce(local_best, cub::Max{});
        if (threadIdx.x == 0) partial_best[blockIdx.x] = best;
        grid.sync();
        if (blockIdx.x == 0) {
            unsigned long long value = 0;
            for (int i = threadIdx.x; i < gridDim.x; i += BLOCK_SIZE)
                value = max(value, partial_best[i]);
            value = Reduction(storage).Reduce(value, cub::Max{});
            if (threadIdx.x == 0) winner = value;
            __syncthreads();
            const int count = static_cast<int>(winner >> 32);
            const int seed = INT_MAX - static_cast<int>(winner & 0xffffffffu);
            if (count <= 1) {
                if (threadIdx.x == 0) state->stop = 1;
            } else {
                const int start = state->member_count;
                for (int i = threadIdx.x; i < count; i += BLOCK_SIZE) {
                    const int point = members[offsets[seed] + i];
                    output_members[start + i] = point;
                    clustered[point] = 1;
                }
                __syncthreads();
                if (threadIdx.x == 0) {
                    output_sizes[state->cluster_count] = count;
                    output_seeds[state->cluster_count] = seed;
                    ++state->cluster_count;
                    state->member_count += count;
                    state->remaining -= count;
                    if (state->remaining == 0) state->stop = 1;
                }
            }
        }
        grid.sync();
        if (state->stop) return;
    }
}

template<int ITEMS, bool DENSE, bool SHARED>
bool launchPersistentKernel(const Point* points, const double* distances,
                            const size_t* offsets, const int* neighbors,
                            unsigned char* clustered, int* cardinalities, int* members,
                            double* scratch, unsigned long long* partial_best, RunState* state,
                            int* output_members, int* output_sizes, int* output_seeds,
                            int n, int seed_count, size_t maximum_degree, double threshold,
                            int multiprocessors) {
    const size_t shared_bytes = SHARED ? maximum_degree * sizeof(double) : 0;
    int resident_blocks;
    cudaCheck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident_blocks,
        persistentClustering<ITEMS, DENSE, SHARED>, BLOCK_SIZE, shared_bytes), "query kernel occupancy");
    if (!resident_blocks) return false;
    const int tasks = ITEMS ? (seed_count + 3) / 4 : seed_count;
    const int blocks = std::min(tasks, multiprocessors * resident_blocks);
    void* arguments[] = {&points, &distances, &offsets, &neighbors, &clustered, &cardinalities,
        &members, &scratch, &partial_best, &state, &output_members, &output_sizes, &output_seeds,
        &n, &seed_count, &threshold};
    cudaCheck(cudaLaunchCooperativeKernel(
        reinterpret_cast<void*>(persistentClustering<ITEMS, DENSE, SHARED>),
        dim3(blocks), dim3(BLOCK_SIZE), arguments, shared_bytes), "launch persistent clustering");
    return true;
}

template<bool DENSE>
bool launchPersistent(const Point* points, const double* distances, const size_t* offsets,
                      const int* neighbors, unsigned char* clustered, int* cardinalities,
                      int* members, double* scratch, unsigned long long* partial_best, RunState* state,
                      int* output_members, int* output_sizes, int* output_seeds,
                      int n, int seed_count, size_t maximum_degree, double threshold,
                      int multiprocessors) {
    #define LAUNCH_PERSISTENT(ITEMS, SHARED) \
        launchPersistentKernel<ITEMS, DENSE, SHARED>(points, distances, offsets, neighbors, \
            clustered, cardinalities, members, scratch, partial_best, state, output_members, \
            output_sizes, output_seeds, n, seed_count, maximum_degree, threshold, multiprocessors)
    if (maximum_degree <= 32) return LAUNCH_PERSISTENT(1, false);
    if (maximum_degree <= 64) return LAUNCH_PERSISTENT(2, false);
    if (maximum_degree <= 128) return LAUNCH_PERSISTENT(4, false);
    if (maximum_degree <= 256) return LAUNCH_PERSISTENT(8, false);
    if (maximum_degree <= 512) return LAUNCH_PERSISTENT(16, false);
    if (maximum_degree <= 2048) return LAUNCH_PERSISTENT(0, true);
    return LAUNCH_PERSISTENT(0, false);
    #undef LAUNCH_PERSISTENT
}

} // namespace

// Candidate construction, winner selection and membership updates all execute
// on the GPU. The host only assembles the winning clusters for the public API.
std::vector<Cluster> qtClustering(const std::vector<Point>& points, const double threshold) {
    const int n = static_cast<int>(points.size());
    std::vector<Cluster> clusters;
    if (n == 0) return clusters;
    cudaCheck(cudaFree(nullptr), "initialize CUDA");
    DeviceBuffer<Point> device_points(n);
    cudaCheck(cudaMemcpy(device_points.get(), points.data(), n * sizeof(Point), cudaMemcpyHostToDevice),
              "copy points");
    size_t free_bytes, total_bytes;
    cudaCheck(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
    const size_t matrix_elements = static_cast<size_t>(n) * n;
    const bool cache_distances = matrix_elements <= free_bytes / (2 * sizeof(double));
    DeviceBuffer<double> distances(cache_distances ? matrix_elements : 0);
    if (cache_distances) {
        buildDistances<<<dim3((n + 31) / 32, (n + 31) / 32), dim3(32, 8)>>>(
            device_points.get(), distances.get(), n);
        cudaCheck(cudaGetLastError(), "build distances");
    }
    DeviceBuffer<size_t> counts(n);
    if (distances.get())
        countNeighbors<true><<<n, BLOCK_SIZE>>>(device_points.get(), distances.get(), counts.get(), n, threshold);
    else
        countNeighbors<false><<<n, BLOCK_SIZE>>>(device_points.get(), nullptr, counts.get(), n, threshold);
    cudaCheck(cudaGetLastError(), "count neighbors");
    std::vector<size_t> host_counts(n);
    cudaCheck(cudaMemcpy(host_counts.data(), counts.get(), n * sizeof(size_t), cudaMemcpyDeviceToHost),
              "copy neighbor counts");
    const size_t maximum_degree = *std::max_element(host_counts.begin(), host_counts.end());
    // A complete graph has a single winning cluster with seed zero. Evaluating
    // every other seed would only reproduce the same cardinality.
    const bool complete = *std::min_element(host_counts.begin(), host_counts.end()) == static_cast<size_t>(n);
    const int seed_count = complete ? 1 : n;
    std::vector<size_t> host_offsets(seed_count + 1, 0);
    for (int i = 0; i < seed_count; ++i) host_offsets[i + 1] = host_offsets[i] + host_counts[i];
    const size_t entries = host_offsets.back();
    const bool global_scratch = maximum_degree > 2048;
    const size_t bytes_per_entry = 2 * sizeof(int) + (global_scratch ? sizeof(double) : 0);
    cudaCheck(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
    if (distances.get() && entries > free_bytes / bytes_per_entry) distances.release();
    counts.release();
    DeviceBuffer<size_t> offsets(seed_count + 1);
    DeviceBuffer<int> neighbors(entries), candidate_members(entries), cardinalities(seed_count);
    DeviceBuffer<double> scratch(global_scratch ? entries : 0);
    DeviceBuffer<unsigned char> clustered(n);
    const int partial_count = maximum_degree <= 512 ? (seed_count + 3) / 4 : seed_count;
    DeviceBuffer<unsigned long long> partial_best(partial_count), device_best(1);
    cudaCheck(cudaMemcpy(offsets.get(), host_offsets.data(), (seed_count + 1) * sizeof(size_t),
                         cudaMemcpyHostToDevice), "copy offsets");
    cudaCheck(cudaMemset(clustered.get(), 0, n), "initialize clustered flags");
    cudaCheck(cudaMemset(cardinalities.get(), 0, seed_count * sizeof(int)), "initialize candidates");
    if (distances.get())
        buildNeighbors<true><<<seed_count, BLOCK_SIZE>>>(device_points.get(), distances.get(),
            offsets.get(), neighbors.get(), n, threshold);
    else
        buildNeighbors<false><<<seed_count, BLOCK_SIZE>>>(device_points.get(), nullptr,
            offsets.get(), neighbors.get(), n, threshold);
    cudaCheck(cudaGetLastError(), "build neighbors");

    int device;
    cudaDeviceProp properties;
    cudaCheck(cudaGetDevice(&device), "cudaGetDevice");
    cudaCheck(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
    if (properties.cooperativeLaunch) {
        DeviceBuffer<RunState> state(1);
        DeviceBuffer<int> output_members(n), output_sizes(n), output_seeds(n);
        RunState host_state{n, 0, 0, 0};
        cudaCheck(cudaMemcpy(state.get(), &host_state, sizeof(host_state), cudaMemcpyHostToDevice),
                  "initialize clustering state");
        const bool launched = distances.get()
            ? launchPersistent<true>(device_points.get(), distances.get(), offsets.get(), neighbors.get(),
                clustered.get(), cardinalities.get(), candidate_members.get(), scratch.get(), partial_best.get(),
                state.get(), output_members.get(), output_sizes.get(), output_seeds.get(),
                n, seed_count, maximum_degree, threshold, properties.multiProcessorCount)
            : launchPersistent<false>(device_points.get(), nullptr, offsets.get(), neighbors.get(),
                clustered.get(), cardinalities.get(), candidate_members.get(), scratch.get(), partial_best.get(),
                state.get(), output_members.get(), output_sizes.get(), output_seeds.get(),
                n, seed_count, maximum_degree, threshold, properties.multiProcessorCount);
        if (launched) {
            cudaCheck(cudaMemcpy(&host_state, state.get(), sizeof(host_state), cudaMemcpyDeviceToHost),
                      "copy clustering state");
            std::vector<int> host_members(host_state.member_count), host_sizes(host_state.cluster_count),
                             host_seeds(host_state.cluster_count);
            if (host_state.cluster_count) {
                cudaCheck(cudaMemcpy(host_members.data(), output_members.get(), host_members.size() * sizeof(int),
                                     cudaMemcpyDeviceToHost), "copy output members");
                cudaCheck(cudaMemcpy(host_sizes.data(), output_sizes.get(), host_sizes.size() * sizeof(int),
                                     cudaMemcpyDeviceToHost), "copy output sizes");
                cudaCheck(cudaMemcpy(host_seeds.data(), output_seeds.get(), host_seeds.size() * sizeof(int),
                                     cudaMemcpyDeviceToHost), "copy output seeds");
            }
            clusters.reserve(host_state.cluster_count + host_state.remaining);
            size_t start = 0;
            for (int i = 0; i < host_state.cluster_count; ++i) {
                Cluster cluster;
                cluster.seed_point = host_seeds[i];
                cluster.members.assign(host_members.begin() + start, host_members.begin() + start + host_sizes[i]);
                start += host_sizes[i];
                clusters.push_back(std::move(cluster));
            }
            if (host_state.remaining) {
                std::vector<unsigned char> host_clustered(n);
                cudaCheck(cudaMemcpy(host_clustered.data(), clustered.get(), n, cudaMemcpyDeviceToHost),
                          "copy singleton flags");
                for (int i = 0; i < n; ++i)
                    if (!host_clustered[i]) clusters.push_back(Cluster{{i}, i});
            }
            return clusters;
        }
    }

    // Devices without cooperative launch support use the same CUDA candidate
    // kernels, with a host-controlled sequence of GPU reductions and updates.
    int remaining = n;
    while (remaining > 0) {
        if (distances.get())
            launchCandidates<true>(device_points.get(), distances.get(), offsets.get(), neighbors.get(),
                clustered.get(), cardinalities.get(), candidate_members.get(), scratch.get(),
                partial_best.get(), n, seed_count, maximum_degree, threshold);
        else
            launchCandidates<false>(device_points.get(), nullptr, offsets.get(), neighbors.get(),
                clustered.get(), cardinalities.get(), candidate_members.get(), scratch.get(),
                partial_best.get(), n, seed_count, maximum_degree, threshold);
        selectBest<<<1, 256>>>(partial_best.get(), device_best.get(), partial_count);
        cudaCheck(cudaGetLastError(), "select best cluster");
        unsigned long long best;
        cudaCheck(cudaMemcpy(&best, device_best.get(), sizeof(best), cudaMemcpyDeviceToHost), "copy best cluster");
        const int count = static_cast<int>(best >> 32);
        const int seed = INT_MAX - static_cast<int>(best & 0xffffffffu);
        if (count == 1) {
            // Once the best cardinality is one, all remaining seeds are
            // singletons, emitted in the original increasing-index order.
            std::vector<unsigned char> host_clustered(n);
            cudaCheck(cudaMemcpy(host_clustered.data(), clustered.get(), n, cudaMemcpyDeviceToHost),
                      "copy singleton flags");
            for (int i = 0; i < n; ++i)
                if (!host_clustered[i]) clusters.push_back(Cluster{{i}, i});
            break;
        }
        if (count <= 0 || count > remaining || seed < 0 || seed >= seed_count) {
            fprintf(stderr, "CUDA clustering produced an invalid winner\n");
            std::exit(EXIT_FAILURE);
        }
        Cluster cluster;
        cluster.seed_point = seed;
        cluster.members.resize(count);
        const int* winning_members = candidate_members.get() + host_offsets[seed];
        cudaCheck(cudaMemcpy(cluster.members.data(), winning_members, count * sizeof(int),
                             cudaMemcpyDeviceToHost), "copy cluster members");
        clusters.push_back(std::move(cluster));
        markClustered<<<(count + 255) / 256, 256>>>(clustered.get(), winning_members, count);
        cudaCheck(cudaGetLastError(), "mark cluster members");
        remaining -= count;
    }
    cudaCheck(cudaDeviceSynchronize(), "finish clustering");
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
    
    cudaCheck(cudaFree(nullptr), "initialize CUDA");

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
