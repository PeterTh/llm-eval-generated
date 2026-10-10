// QT Clustering Benchmark - CUDA Version
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cub/cub.cuh>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

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

// Preserve the original host's native double arithmetic, including contraction.
__host__ __device__ inline double squaredDistance(const Point& p1, const Point& p2) {
    const double dx = p1.x - p2.x;
    const double dy = p1.y - p2.y;
#ifdef __CUDA_ARCH__
#ifdef __FMA__
    return __fma_rn(dx, dx, __dmul_rn(dy, dy));
#else
    return __dadd_rn(__dmul_rn(dx, dx), __dmul_rn(dy, dy));
#endif
#else
    return dx * dx + dy * dy;
#endif
}

__host__ __device__ inline double distance(const Point& p1, const Point& p2) {
#ifdef __CUDA_ARCH__
    return __dsqrt_rn(squaredDistance(p1, p2));
#else
    return std::sqrt(squaredDistance(p1, p2));
#endif
}

// Every clustering path below runs on CUDA; failure to initialize or allocate
// device resources is reported rather than falling back to sequential clustering.
static void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(operation) checkCuda((operation), #operation)

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count = 0) { allocate(count); }
    ~DeviceBuffer() { if (data_) cudaFree(data_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    void allocate(size_t count) {
        if (count > std::numeric_limits<size_t>::max() / sizeof(T)) {
            fprintf(stderr, "CUDA allocation size overflow\n");
            std::exit(EXIT_FAILURE);
        }
        if (count) CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)));
    }
    void reset() {
        if (data_) CUDA_CHECK(cudaFree(data_));
        data_ = nullptr;
    }
    T* get() const { return data_; }
private:
    T* data_ = nullptr;
};

constexpr unsigned WARP_MASK = 0xffffffffu;
constexpr int GPU_THREADS = 128;
constexpr double NO_DISTANCE = std::numeric_limits<double>::max();

__device__ __forceinline__ int warpSum(int value) {
    for (int delta = 16; delta; delta >>= 1)
        value += __shfl_down_sync(WARP_MASK, value, delta);
    return __shfl_sync(WARP_MASK, value, 0);
}

// Cardinality wins first, then the lowest seed, exactly as in the sequential
// scan. Maxima are kept as squared distances; selection explicitly
// resolves ties in the rounded square roots to preserve point-index ordering.
__device__ __forceinline__ unsigned long long clusterKey(int size, int seed) {
    return (static_cast<unsigned long long>(size) << 32) |
           (static_cast<unsigned>(INT_MAX) - static_cast<unsigned>(seed));
}

struct Closest {
    double diameter;
    int point;
};

__device__ __forceinline__ Closest warpClosest(Closest best) {
    for (int delta = 16; delta; delta >>= 1) {
        const double diameter = __shfl_down_sync(WARP_MASK, best.diameter, delta);
        const int point = __shfl_down_sync(WARP_MASK, best.point, delta);
        if (diameter < best.diameter ||
            (diameter == best.diameter && point < best.point))
            best = {diameter, point};
    }
    best.diameter = __shfl_sync(WARP_MASK, best.diameter, 0);
    best.point = __shfl_sync(WARP_MASK, best.point, 0);
    return best;
}

__device__ __forceinline__ int warpMinPoint(int point) {
    for (int delta = 16; delta; delta >>= 1)
        point = min(point, __shfl_down_sync(WARP_MASK, point, delta));
    return __shfl_sync(WARP_MASK, point, 0);
}

// Avoid sqrt for threshold tests far from the boundary. Values between the
// squared predecessor of the threshold and its squared value use the exact
// rounded sqrt, preserving the strict original comparison.
__device__ __forceinline__ bool withinThreshold(double squared, double threshold,
                                                double lower, double upper) {
    if (squared < lower) return true;
    if (squared > upper) return false;
    return __dsqrt_rn(squared) < threshold;
}

// Build a CSR neighborhood graph. The seed itself is always included, even
// when no distance passes the threshold. Warps read consecutive columns.
__global__ void countNeighbors(const Point* points, int n, double threshold,
                               double* distances, size_t* counts, int* max_degree) {
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    if (row >= n) return;
    const int lane = threadIdx.x & 31;
    const Point seed = points[row];
    const double previous = __longlong_as_double(__double_as_longlong(threshold) - 1);
    const double lower = __dmul_rn(previous, previous);
    const double upper = __dmul_rn(threshold, threshold);
    int count = 0;
    for (int column = lane; column < n; column += 32) {
        const double d = squaredDistance(seed, points[column]);
        if (distances) distances[static_cast<size_t>(row) * n + column] = d;
        count += column == row || withinThreshold(d, threshold, lower, upper);
    }
    count = warpSum(count);
    if (!lane) {
        counts[row] = static_cast<size_t>(count);
        atomicMax(max_degree, count);
    }
}

__global__ void fillNeighbors(const Point* points, int n, double threshold,
                              const double* distances, const size_t* offsets,
                              int* neighbors) {
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    if (row >= n) return;
    const int lane = threadIdx.x & 31;
    const Point seed = points[row];
    const double previous = __longlong_as_double(__double_as_longlong(threshold) - 1);
    const double lower = __dmul_rn(previous, previous);
    const double upper = __dmul_rn(threshold, threshold);
    size_t cursor = offsets[row];
    for (int base = 0; base < n; base += 32) {
        const int column = base + lane;
        bool include = false;
        if (column < n) {
            const double d = distances ? distances[static_cast<size_t>(row) * n + column]
                                       : squaredDistance(seed, points[column]);
            include = column == row || withinThreshold(d, threshold, lower, upper);
        }
        const unsigned mask = __ballot_sync(WARP_MASK, include);
        if (include) neighbors[cursor + __popc(mask & ((1u << lane) - 1u))] = column;
        cursor += __popc(mask);
    }
}

struct ClusteringState {
    unsigned long long best;
    int next_seed;
    int remaining;
    int cluster_count;
    int winner_seed;
    int winner_size;
    int output_offset;
    int singletons;
};

struct ClusteringData {
    const Point* points;
    const double* distances;
    const size_t* offsets;
    const int* neighbors;
    int* labels;
    int* cached_sizes;
    int* cached_members;
    double* scratch;
    int scratch_stride;
    int* output_members;
    int* output_seeds;
    int* output_sizes;
    ClusteringState* state;
    int n;
    double threshold;
};

__device__ __forceinline__ double pairSquaredDistance(const ClusteringData& data,
                                               int member, int point) {
    return data.distances ? data.distances[static_cast<size_t>(member) * data.n + point]
                          : squaredDistance(data.points[member], data.points[point]);
}

__device__ __forceinline__ bool cannotWin(const ClusteringData& data,
                                         int upper_bound, int seed) {
    unsigned long long best = 0;
    if (!(threadIdx.x & 31)) best = atomicAdd(&data.state->best, 0ULL);
    best = __shfl_sync(WARP_MASK, best, 0);
    return clusterKey(upper_bound, seed) <= best;
}

// One warp constructs one candidate. Small neighborhoods stay in registers;
// larger ones use reusable device scratch. Each new member updates running
// maximum distances once, replacing repeated scans over all previous members.
template <int Slots>
__device__ void evaluateSeed(const ClusteringData& data, int seed) {
    const int lane = threadIdx.x & 31;
    if (data.labels[seed] >= 0) return;
    const size_t begin = data.offsets[seed];
    const int degree = static_cast<int>(data.offsets[seed + 1] - begin);
    int* members = data.cached_members + begin;
    int cardinality = data.cached_sizes[seed];
    // Removing points outside a cached candidate cannot change its greedy
    // sequence. Rebuild only if the previous winning cluster intersects it.
    if (cardinality) {
        bool invalid = false;
        for (int i = lane; i < cardinality; i += 32)
            invalid |= data.labels[members[i]] >= 0;
        if (!__any_sync(WARP_MASK, invalid)) {
            if (!lane) atomicMax(&data.state->best, clusterKey(cardinality, seed));
            return;
        }
    }
    if (!lane) data.cached_sizes[seed] = 0;
    cardinality = 1;
    const double squared_threshold = __dmul_rn(data.threshold, data.threshold);
    if (!lane) members[0] = seed;

    if (degree <= Slots * 32) {
        int ids[Slots];
        double maxima[Slots];
        int available = 0;
#pragma unroll
        for (int slot = 0; slot < Slots; ++slot) {
            const int position = lane + slot * 32;
            const int point = position < degree ? data.neighbors[begin + position] : INT_MAX;
            ids[slot] = point;
            const bool active = point != INT_MAX && point != seed && data.labels[point] < 0;
            maxima[slot] = active ? pairSquaredDistance(data, seed, point) : NO_DISTANCE;
            available += active;
        }
        if (cannotWin(data, 1 + warpSum(available), seed)) return;
        while (true) {
            Closest closest{NO_DISTANCE, INT_MAX};
#pragma unroll
            for (int slot = 0; slot < Slots; ++slot) {
                if (maxima[slot] < closest.diameter ||
                    (maxima[slot] == closest.diameter && ids[slot] < closest.point))
                    closest = {maxima[slot], ids[slot]};
            }
            const double local_minimum = closest.diameter;
            closest = warpClosest(closest);
            if (closest.diameter == NO_DISTANCE) break;
            const double rounded = __dsqrt_rn(closest.diameter);
            if (!(rounded < data.threshold)) break;
            // Only scores immediately above the squared minimum can round to
            // the same sqrt. The square of the next double is a conservative
            // bound; the exact sqrt comparison then restores ascending-ID ties.
            const double next = __longlong_as_double(__double_as_longlong(rounded) + 1);
            const double tie_bound = __dmul_rn(next, next);
            int tied_point = closest.point;
            if (local_minimum <= tie_bound) {
#pragma unroll
                for (int slot = 0; slot < Slots; ++slot) {
                    if (ids[slot] < tied_point && maxima[slot] <= tie_bound &&
                        __dsqrt_rn(maxima[slot]) == rounded)
                        tied_point = ids[slot];
                }
            }
            closest.point = warpMinPoint(tied_point);
            if (!lane) members[cardinality] = closest.point;
            ++cardinality;
            available = 0;
#pragma unroll
            for (int slot = 0; slot < Slots; ++slot) {
                double value = maxima[slot];
                if (value != NO_DISTANCE) {
                    if (ids[slot] == closest.point) value = NO_DISTANCE;
                    else {
                        const double d = pairSquaredDistance(data, closest.point, ids[slot]);
                        value = value < d ? d : value;
                        if (value > squared_threshold) value = NO_DISTANCE;
                    }
                    maxima[slot] = value;
                }
                available += value != NO_DISTANCE;
            }
            // A competing seed can make further construction unnecessary.
            if ((cardinality & 7) == 0 &&
                cannotWin(data, cardinality + warpSum(available), seed)) return;
        }
    } else {
        // Resident workers reuse scratch between seeds, so large-neighborhood
        // workspace scales with GPU concurrency rather than all graph edges.
        const size_t worker = (static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
        double* maxima = data.scratch + (data.scratch_stride ? worker * data.scratch_stride : begin);
        int available = 0;
        for (int position = lane; position < degree; position += 32) {
            const int point = data.neighbors[begin + position];
            const bool active = point != seed && data.labels[point] < 0;
            maxima[position] = active ? pairSquaredDistance(data, seed, point) : NO_DISTANCE;
            available += active;
        }
        if (cannotWin(data, 1 + warpSum(available), seed)) return;
        while (true) {
            Closest closest{NO_DISTANCE, INT_MAX};
            for (int position = lane; position < degree; position += 32) {
                const double value = maxima[position];
                const int point = data.neighbors[begin + position];
                if (value < closest.diameter ||
                    (value == closest.diameter && point < closest.point))
                    closest = {value, point};
            }
            const double local_minimum = closest.diameter;
            closest = warpClosest(closest);
            if (closest.diameter == NO_DISTANCE) break;
            const double rounded = __dsqrt_rn(closest.diameter);
            if (!(rounded < data.threshold)) break;
            const double next = __longlong_as_double(__double_as_longlong(rounded) + 1);
            const double tie_bound = __dmul_rn(next, next);
            int tied_point = closest.point;
            if (local_minimum <= tie_bound) {
                for (int position = lane; position < degree; position += 32) {
                    const int point = data.neighbors[begin + position];
                    if (point < tied_point && maxima[position] <= tie_bound &&
                        __dsqrt_rn(maxima[position]) == rounded)
                        tied_point = point;
                }
            }
            closest.point = warpMinPoint(tied_point);
            if (!lane) members[cardinality] = closest.point;
            ++cardinality;
            available = 0;
            for (int position = lane; position < degree; position += 32) {
                double value = maxima[position];
                if (value != NO_DISTANCE) {
                    const int point = data.neighbors[begin + position];
                    if (point == closest.point) value = NO_DISTANCE;
                    else {
                        const double d = pairSquaredDistance(data, closest.point, point);
                        value = value < d ? d : value;
                        if (value > squared_threshold) value = NO_DISTANCE;
                    }
                    maxima[position] = value;
                }
                available += value != NO_DISTANCE;
            }
            if ((cardinality & 7) == 0 &&
                cannotWin(data, cardinality + warpSum(available), seed)) return;
        }
    }
    if (!lane) {
        data.cached_sizes[seed] = cardinality;
        atomicMax(&data.state->best, clusterKey(cardinality, seed));
    }
}

__device__ void chooseWinner(const ClusteringData& data) {
    ClusteringState& state = *data.state;
    state.winner_size = static_cast<int>(state.best >> 32);
    state.winner_seed = INT_MAX - static_cast<int>(state.best & 0xffffffffULL);
    // If the largest candidate is a singleton, every remaining point must be
    // a singleton. Finish all of them together with a stable device selection.
    state.singletons = state.winner_size == 1;
    if (!state.singletons) {
        data.output_seeds[state.cluster_count] = state.winner_seed;
        data.output_sizes[state.cluster_count] = state.winner_size;
        state.output_offset = data.n - state.remaining;
        state.remaining -= state.winner_size;
        ++state.cluster_count;
    }
    state.best = 0;
    state.next_seed = 0;
}

__device__ void commitMembers(const ClusteringData& data, int thread, int stride) {
    const ClusteringState& state = *data.state;
    const int* members = data.cached_members + data.offsets[state.winner_seed];
    for (int i = thread; i < state.winner_size; i += stride) {
        const int point = members[i];
        data.labels[point] = state.cluster_count - 1;
        data.output_members[state.output_offset + i] = point;
    }
}

// Resident warps pull seeds from a queue, balancing candidates with different
// sizes. Grid barriers preserve the sequential ordering between QT rounds.
template <int Slots>
__global__ void clusteringKernel(ClusteringData data) {
    const auto grid = cooperative_groups::this_grid();
    const int thread = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane = threadIdx.x & 31;
    while (data.state->remaining) {
        while (true) {
            int seed = 0;
            if (!lane) seed = atomicAdd(&data.state->next_seed, 1);
            seed = __shfl_sync(WARP_MASK, seed, 0);
            if (seed >= data.n) break;
            evaluateSeed<Slots>(data, seed);
        }
        grid.sync();
        if (!thread) chooseWinner(data);
        grid.sync();
        if (data.state->singletons) break;
        commitMembers(data, thread, gridDim.x * blockDim.x);
        grid.sync();
    }
}

// Older devices without cooperative launch still execute all candidate
// construction, selection and cluster removal on CUDA.
template <int Slots>
__global__ void evaluateRound(ClusteringData data) {
    const int seed = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    if (seed < data.n) evaluateSeed<Slots>(data, seed);
}

__global__ void commitRound(ClusteringData data) {
    if (!threadIdx.x) chooseWinner(data);
    __syncthreads();
    if (!data.state->singletons) commitMembers(data, threadIdx.x, blockDim.x);
}

template <int Slots>
static void runClustering(ClusteringData data, const cudaDeviceProp& properties) {
    if (properties.cooperativeLaunch) {
        int blocks_per_sm = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks_per_sm, clusteringKernel<Slots>, GPU_THREADS, 0));
        const int blocks = std::min((data.n + 3) / 4,
                                   blocks_per_sm * properties.multiProcessorCount);
        void* arguments[] = {&data};
        CUDA_CHECK(cudaLaunchCooperativeKernel(
            reinterpret_cast<void*>(clusteringKernel<Slots>), dim3(blocks),
            dim3(GPU_THREADS), arguments));
    } else {
        const int blocks = (data.n + 3) / 4;
        while (true) {
            evaluateRound<Slots><<<blocks, GPU_THREADS>>>(data);
            CUDA_CHECK(cudaGetLastError());
            unsigned long long best = 0;
            CUDA_CHECK(cudaMemcpy(&best, &data.state->best, sizeof(best), cudaMemcpyDeviceToHost));
            if (static_cast<int>(best >> 32) == 1) break;
            commitRound<<<1, GPU_THREADS>>>(data);
            CUDA_CHECK(cudaGetLastError());
            ClusteringState state;
            CUDA_CHECK(cudaMemcpy(&state, data.state, sizeof(state), cudaMemcpyDeviceToHost));
            if (!state.remaining) break;
        }
    }
}

struct IsUnclustered {
    const int* labels;
    __device__ bool operator()(int point) const { return labels[point] < 0; }
};

__global__ void finishSingletons(ClusteringData data, int cluster_count, int remaining) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < remaining) {
        const int point = data.output_members[data.n - remaining + i];
        data.output_seeds[cluster_count + i] = point;
        data.output_sizes[cluster_count + i] = 1;
        data.labels[point] = cluster_count + i;
    }
    if (!i) {
        data.state->remaining = 0;
        data.state->cluster_count = cluster_count + remaining;
    }
}

std::vector<Cluster> qtClustering(const std::vector<Point>& points, double threshold) {
    const int n = static_cast<int>(points.size());
    if (!n) return {};
    int device = 0;
    cudaDeviceProp properties;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));

    DeviceBuffer<Point> gpu_points(n);
    DeviceBuffer<size_t> counts(static_cast<size_t>(n) + 1);
    DeviceBuffer<size_t> offsets(static_cast<size_t>(n) + 1);
    DeviceBuffer<int> max_degree(1);
    CUDA_CHECK(cudaMemcpy(gpu_points.get(), points.data(), n * sizeof(Point), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(counts.get(), 0, (static_cast<size_t>(n) + 1) * sizeof(size_t)));
    CUDA_CHECK(cudaMemset(max_degree.get(), 0, sizeof(int)));

    // Reuse pair distances when a dense table fits; release it below if the
    // measured sparse neighborhoods need the memory. Otherwise,
    // the same CUDA algorithm computes distances from points on demand.
    size_t free_bytes = 0, total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    const size_t pairs = static_cast<size_t>(n) * n;
    DeviceBuffer<double> distances(pairs <= (free_bytes / sizeof(double)) * 9 / 10 ? pairs : 0);
    const int neighbor_blocks = (n + 3) / 4;
    countNeighbors<<<neighbor_blocks, GPU_THREADS>>>(gpu_points.get(), n, threshold,
                                                    distances.get(), counts.get(), max_degree.get());
    CUDA_CHECK(cudaGetLastError());
    size_t scan_bytes = 0;
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(nullptr, scan_bytes, counts.get(), offsets.get(), n + 1));
    DeviceBuffer<unsigned char> scan_storage(scan_bytes);
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(scan_storage.get(), scan_bytes,
                                            counts.get(), offsets.get(), n + 1));
    size_t entries = 0;
    int degree = 0;
    CUDA_CHECK(cudaMemcpy(&entries, offsets.get() + n, sizeof(entries), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&degree, max_degree.get(), sizeof(degree), cudaMemcpyDeviceToHost));
    scan_storage.reset();
    counts.reset();
    max_degree.reset();
    // Leave room for sparse memberships, large-neighborhood scratch and output.
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    size_t scratch_entries = degree > 256 ? entries : 0;
    int scratch_stride = 0;
    if (degree > 256 && properties.cooperativeLaunch) {
        int blocks_per_sm = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks_per_sm, clusteringKernel<8>, GPU_THREADS, 0));
        const int workers = std::min(neighbor_blocks,
            blocks_per_sm * properties.multiProcessorCount) * (GPU_THREADS / 32);
        scratch_stride = degree;
        scratch_entries = static_cast<size_t>(workers) * degree;
    }
    const size_t sparse_bytes = entries * (2 * sizeof(int)) + scratch_entries * sizeof(double);
    if (distances.get() && sparse_bytes > free_bytes * 4 / 5)
        distances.reset();

    DeviceBuffer<int> neighbors(entries);
    DeviceBuffer<int> cached_members(entries);
    DeviceBuffer<double> scratch(scratch_entries);
    DeviceBuffer<int> labels(n), cached_sizes(n);
    DeviceBuffer<int> output_members(n), output_seeds(n), output_sizes(n);
    DeviceBuffer<ClusteringState> gpu_state(1);
    ClusteringState state{};
    state.remaining = n;
    CUDA_CHECK(cudaMemcpy(gpu_state.get(), &state, sizeof(state), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(labels.get(), 0xff, n * sizeof(int)));
    CUDA_CHECK(cudaMemset(cached_sizes.get(), 0, n * sizeof(int)));
    fillNeighbors<<<neighbor_blocks, GPU_THREADS>>>(gpu_points.get(), n, threshold,
                                                    distances.get(), offsets.get(), neighbors.get());
    CUDA_CHECK(cudaGetLastError());

    ClusteringData data{gpu_points.get(), distances.get(), offsets.get(), neighbors.get(),
                        labels.get(), cached_sizes.get(), cached_members.get(), scratch.get(), scratch_stride,
                        output_members.get(), output_seeds.get(), output_sizes.get(),
                        gpu_state.get(), n, threshold};
    if (degree <= 32) runClustering<1>(data, properties);
    else if (degree <= 64) runClustering<2>(data, properties);
    else if (degree <= 128) runClustering<4>(data, properties);
    else runClustering<8>(data, properties);
    CUDA_CHECK(cudaMemcpy(&state, gpu_state.get(), sizeof(state), cudaMemcpyDeviceToHost));
    if (state.remaining) {
        size_t select_bytes = 0;
        const cub::CountingInputIterator<int> indices(0);
        CUDA_CHECK(cub::DeviceSelect::If(nullptr, select_bytes, indices,
            output_members.get() + n - state.remaining, &gpu_state.get()->winner_size,
            n, IsUnclustered{labels.get()}));
        DeviceBuffer<unsigned char> select_storage(select_bytes);
        CUDA_CHECK(cub::DeviceSelect::If(select_storage.get(), select_bytes, indices,
            output_members.get() + n - state.remaining, &gpu_state.get()->winner_size,
            n, IsUnclustered{labels.get()}));
        finishSingletons<<<(state.remaining + GPU_THREADS - 1) / GPU_THREADS, GPU_THREADS>>>(
            data, state.cluster_count, state.remaining);
        CUDA_CHECK(cudaGetLastError());
        state.cluster_count += state.remaining;
    }

    std::vector<int> members(n), seeds(state.cluster_count), sizes(state.cluster_count);
    CUDA_CHECK(cudaMemcpy(members.data(), output_members.get(), n * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(seeds.data(), output_seeds.get(), seeds.size() * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sizes.data(), output_sizes.get(), sizes.size() * sizeof(int), cudaMemcpyDeviceToHost));
    std::vector<Cluster> clusters;
    clusters.reserve(state.cluster_count);
    size_t cursor = 0;
    for (int i = 0; i < state.cluster_count; ++i) {
        Cluster cluster;
        cluster.seed_point = seeds[i];
        cluster.members.assign(members.begin() + cursor, members.begin() + cursor + sizes[i]);
        cursor += sizes[i];
        clusters.push_back(std::move(cluster));
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
    
    // Initialize CUDA before measuring clustering, like other runtime setup.
    CUDA_CHECK(cudaFree(nullptr));

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
