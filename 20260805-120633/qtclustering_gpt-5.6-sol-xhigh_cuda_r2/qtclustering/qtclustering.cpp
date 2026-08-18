// QT Clustering Benchmark - CUDA implementation
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.

#include <algorithm>
#include <chrono>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <limits>
#include <utility>
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

constexpr int CUDA_BLOCK_SIZE = 256;
constexpr int CUDA_WARP_SIZE = 32;

[[noreturn]] void cudaFailure(cudaError_t error, const char* expression,
                              const char* file, int line) {
    std::fprintf(stderr, "CUDA failure at %s:%d: %s failed: %s\n", file, line,
                 expression, cudaGetErrorString(error));
    std::exit(EXIT_FAILURE);
}

#define CUDA_CHECK(expression)                                                   \
    do {                                                                         \
        const cudaError_t cuda_check_error = (expression);                       \
        if (cuda_check_error != cudaSuccess) {                                   \
            cudaFailure(cuda_check_error, #expression, __FILE__, __LINE__);      \
        }                                                                        \
    } while (false)

template <typename T>
class DeviceBuffer {
  public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t count) { allocate(count); }
    ~DeviceBuffer() { release(); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : data_(std::exchange(other.data_, nullptr)),
          count_(std::exchange(other.count_, 0)) {}

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            release();
            data_ = std::exchange(other.data_, nullptr);
            count_ = std::exchange(other.count_, 0);
        }
        return *this;
    }

    void allocate(size_t count) {
        release();
        data_ = nullptr;
        count_ = 0;
        if (count != 0) {
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)));
            count_ = count;
        }
    }

    bool tryAllocate(size_t count) {
        release();
        data_ = nullptr;
        count_ = 0;
        if (count == 0) return true;
        const cudaError_t error =
            cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T));
        if (error != cudaSuccess) {
            data_ = nullptr;
            cudaGetLastError();
            return false;
        }
        count_ = count;
        return true;
    }

    T* get() { return data_; }
    const T* get() const { return data_; }
    size_t size() const { return count_; }

  private:
    void release() noexcept {
        if (data_ != nullptr) {
            // Deallocation is stream ordered, allowing it to overlap the host's
            // result processing instead of extending the measured hot path.
            const cudaError_t error = cudaFreeAsync(data_, nullptr);
            if (error != cudaSuccess) {
                cudaGetLastError();
                cudaFree(data_);
            }
        }
    }

    T* data_ = nullptr;
    size_t count_ = 0;
};

struct CandidateMinimum {
    double diameter;
    int index;
    int state_index;
};

__device__ __forceinline__ CandidateMinimum chooseMinimum(CandidateMinimum a,
                                                           CandidateMinimum b) {
    if (b.index >= 0 &&
        (a.index < 0 || b.diameter < a.diameter ||
         (b.diameter == a.diameter && b.index < a.index))) {
        return b;
    }
    return a;
}

__device__ __forceinline__ CandidateMinimum warpMinimum(CandidateMinimum value) {
    constexpr unsigned int full_warp = 0xffffffffu;
    for (int offset = CUDA_WARP_SIZE / 2; offset > 0; offset /= 2) {
        CandidateMinimum other;
        other.diameter = __shfl_down_sync(full_warp, value.diameter, offset);
        other.index = __shfl_down_sync(full_warp, value.index, offset);
        other.state_index = __shfl_down_sync(full_warp, value.state_index, offset);
        value = chooseMinimum(value, other);
    }
    return value;
}

template <int BlockSize>
__device__ __forceinline__ CandidateMinimum blockMinimum(CandidateMinimum value) {
    static_assert(BlockSize % CUDA_WARP_SIZE == 0, "whole warps are required");
    constexpr int warp_count = BlockSize / CUDA_WARP_SIZE;
    __shared__ double warp_diameters[warp_count];
    __shared__ int warp_indices[warp_count];
    __shared__ int warp_state_indices[warp_count];

    const int lane = threadIdx.x & (CUDA_WARP_SIZE - 1);
    const int warp = threadIdx.x / CUDA_WARP_SIZE;
    value = warpMinimum(value);
    if (lane == 0) {
        warp_diameters[warp] = value.diameter;
        warp_indices[warp] = value.index;
        warp_state_indices[warp] = value.state_index;
    }
    __syncthreads();

    CandidateMinimum result{DBL_MAX, -1, -1};
    if (warp == 0) {
        if (lane < warp_count) {
            result = {warp_diameters[lane], warp_indices[lane],
                      warp_state_indices[lane]};
        }
        result = warpMinimum(result);
    }
    return result;
}

__global__ void initializeActiveSeeds(int* seeds, int count) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < count; index += blockDim.x * gridDim.x) {
        seeds[index] = index;
    }
}

__global__ void buildDistanceMatrix(const Point* __restrict__ points,
                                    double* __restrict__ distances, int count,
                                    size_t element_count) {
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < element_count;
         index += static_cast<size_t>(blockDim.x) * gridDim.x) {
        const int first = static_cast<int>(index / count);
        const int second = static_cast<int>(index - static_cast<size_t>(first) * count);
        const double dx = points[first].x - points[second].x;
        const double dy = points[first].y - points[second].y;
        distances[index] = sqrt(dx * dx + dy * dy);
    }
}

// Compact each seed's threshold-neighborhood once. Every point that can ever
// join a seed's cluster must be in this list, so subsequent greedy rounds can
// avoid scanning the overwhelmingly irrelevant portion of the data set.
__global__ void buildNeighborLists(const double* __restrict__ distances,
                                   int point_count, double threshold,
                                   int* __restrict__ neighbors,
                                   int* __restrict__ neighbor_counts,
                                   int* __restrict__ maximum_count) {
    const int seed = blockIdx.x;
    __shared__ int count;
    if (threadIdx.x == 0) count = 0;
    __syncthreads();

    const size_t row = static_cast<size_t>(seed) * point_count;
    for (int candidate = threadIdx.x; candidate < point_count;
         candidate += blockDim.x) {
        if (candidate != seed && distances[row + candidate] < threshold) {
            const int position = atomicAdd(&count, 1);
            neighbors[row + position] = candidate;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        neighbor_counts[seed] = count;
        atomicMax(maximum_count, count);
    }
}

template <bool PrecomputedDistances>
__device__ __forceinline__ double pointDistance(const Point* __restrict__ points,
                                                 const double* __restrict__ distances,
                                                 int count, int first, int second) {
    if constexpr (PrecomputedDistances) {
        return distances[static_cast<size_t>(first) * count + second];
    } else {
        const double dx = points[first].x - points[second].x;
        const double dy = points[first].y - points[second].y;
        return sqrt(dx * dx + dy * dy);
    }
}

// Each block independently reproduces the greedy candidate cluster for one
// seed. Candidate diameters are updated incrementally, turning the serial
// O(N*k^2) construction into O(N*k) work for a k-member cluster.
template <bool PrecomputedDistances, bool CompactCandidates, bool SharedState>
__global__ void evaluateSeeds(const Point* __restrict__ points,
                              const double* __restrict__ distances,
                              const int* __restrict__ neighbors,
                              const int* __restrict__ neighbor_counts,
                              const unsigned char* __restrict__ clustered,
                              const int* __restrict__ active_seeds,
                              int seed_offset, int point_count, int state_count,
                              double threshold,
                              double* __restrict__ global_state,
                              unsigned long long* __restrict__ best_key) {
    extern __shared__ double shared_state[];
    double* max_diameters = SharedState
        ? shared_state
        : global_state + static_cast<size_t>(blockIdx.x) * state_count;
    const int seed = active_seeds[seed_offset + blockIdx.x];
    const int candidate_count =
        CompactCandidates ? neighbor_counts[seed] : point_count;
    const int* seed_neighbors = CompactCandidates
        ? neighbors + static_cast<size_t>(seed) * point_count
        : nullptr;

    for (int position = threadIdx.x; position < candidate_count;
         position += blockDim.x) {
        const int candidate = CompactCandidates ? seed_neighbors[position] : position;
        max_diameters[position] =
            (clustered[candidate] || candidate == seed) ? -1.0 : 0.0;
    }
    __syncthreads();

    int current_member = seed;
    int cardinality = 1;
    while (true) {
        CandidateMinimum local_best{DBL_MAX, -1, -1};
        for (int position = threadIdx.x; position < candidate_count;
             position += blockDim.x) {
            const int candidate = CompactCandidates ? seed_neighbors[position] : position;
            double max_diameter = max_diameters[position];
            if (max_diameter < 0.0) continue;

            const double candidate_distance =
                pointDistance<PrecomputedDistances>(points, distances, point_count,
                                                     current_member, candidate);
            if (candidate_distance > max_diameter) {
                max_diameter = candidate_distance;
            }

            if (max_diameter < threshold) {
                max_diameters[position] = max_diameter;
                local_best = chooseMinimum(local_best,
                                           {max_diameter, candidate, position});
            } else {
                // Diameters only grow; this candidate can never become valid.
                max_diameters[position] = -1.0;
            }
        }

        const CandidateMinimum block_best = blockMinimum<CUDA_BLOCK_SIZE>(local_best);
        __shared__ int next_member;
        if (threadIdx.x == 0) next_member = block_best.index;
        __syncthreads();
        if (next_member < 0) break;

        if (threadIdx.x == 0) {
            max_diameters[block_best.state_index] = -1.0;
        }
        current_member = next_member;
        ++cardinality;
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        // Higher cardinality wins; equal cardinality chooses the lowest seed,
        // exactly matching the original ascending host loop.
        const unsigned long long key =
            (static_cast<unsigned long long>(static_cast<unsigned int>(cardinality)) << 32) |
            (0xffffffffull - static_cast<unsigned int>(seed));
        atomicMax(best_key, key);
    }
}

template <bool PrecomputedDistances, bool CompactCandidates, bool SharedState>
__global__ void replayWinningSeed(const Point* __restrict__ points,
                                  const double* __restrict__ distances,
                                  const int* __restrict__ neighbors,
                                  const int* __restrict__ neighbor_counts,
                                  const unsigned char* __restrict__ clustered,
                                  const unsigned long long* __restrict__ best_key,
                                  int point_count, int state_count, double threshold,
                                  double* __restrict__ global_state,
                                  int* __restrict__ members,
                                  int* __restrict__ member_count,
                                  int* __restrict__ winning_seed,
                                  int* __restrict__ invariant_error) {
    extern __shared__ double shared_state[];
    double* max_diameters = SharedState ? shared_state : global_state;
    (void)state_count;
    const unsigned long long winner = *best_key;
    const int expected_cardinality = static_cast<int>(winner >> 32);
    const int seed = static_cast<int>(
        0xffffffffu - static_cast<unsigned int>(winner));
    const int candidate_count =
        CompactCandidates ? neighbor_counts[seed] : point_count;
    const int* seed_neighbors = CompactCandidates
        ? neighbors + static_cast<size_t>(seed) * point_count
        : nullptr;

    for (int position = threadIdx.x; position < candidate_count;
         position += blockDim.x) {
        const int candidate = CompactCandidates ? seed_neighbors[position] : position;
        max_diameters[position] =
            (clustered[candidate] || candidate == seed) ? -1.0 : 0.0;
    }
    if (threadIdx.x == 0) {
        members[0] = seed;
        *winning_seed = seed;
    }
    __syncthreads();

    int current_member = seed;
    int cardinality = 1;
    while (true) {
        CandidateMinimum local_best{DBL_MAX, -1, -1};
        for (int position = threadIdx.x; position < candidate_count;
             position += blockDim.x) {
            const int candidate = CompactCandidates ? seed_neighbors[position] : position;
            double max_diameter = max_diameters[position];
            if (max_diameter < 0.0) continue;

            const double candidate_distance =
                pointDistance<PrecomputedDistances>(points, distances, point_count,
                                                     current_member, candidate);
            if (candidate_distance > max_diameter) {
                max_diameter = candidate_distance;
            }
            if (max_diameter < threshold) {
                max_diameters[position] = max_diameter;
                local_best = chooseMinimum(local_best,
                                           {max_diameter, candidate, position});
            } else {
                max_diameters[position] = -1.0;
            }
        }

        const CandidateMinimum block_best = blockMinimum<CUDA_BLOCK_SIZE>(local_best);
        __shared__ int next_member;
        if (threadIdx.x == 0) next_member = block_best.index;
        __syncthreads();
        if (next_member < 0) break;

        if (threadIdx.x == 0) {
            max_diameters[block_best.state_index] = -1.0;
            members[cardinality] = next_member;
        }
        current_member = next_member;
        ++cardinality;
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        *member_count = cardinality;
        if (cardinality != expected_cardinality) *invariant_error = 1;
    }
}

__global__ void markClustered(const int* __restrict__ members,
                              const int* __restrict__ member_count,
                              unsigned char* __restrict__ clustered) {
    const int count = *member_count;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < count; index += blockDim.x * gridDim.x) {
        clustered[members[index]] = 1;
    }
}

__global__ void compactActiveSeeds(const int* __restrict__ input, int input_count,
                                   const unsigned char* __restrict__ clustered,
                                   int* __restrict__ output,
                                   int* __restrict__ output_count) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < input_count; index += blockDim.x * gridDim.x) {
        const int seed = input[index];
        if (!clustered[seed]) {
            const int position = atomicAdd(output_count, 1);
            output[position] = seed;
        }
    }
}

int boundedGridSize(size_t work_items, int multiprocessors, int blocks_per_sm = 16) {
    const size_t needed =
        (work_items + static_cast<size_t>(CUDA_BLOCK_SIZE) - 1) / CUDA_BLOCK_SIZE;
    return static_cast<int>(std::max<size_t>(1, std::min<size_t>(
        needed, static_cast<size_t>(multiprocessors) * blocks_per_sm)));
}

template <bool PrecomputedDistances, bool CompactCandidates>
void launchSeedEvaluation(bool shared_state, int active_count, int batch_rows,
                          size_t shared_bytes, const Point* points,
                          const double* distances, const int* neighbors,
                          const int* neighbor_counts,
                          const unsigned char* clustered, const int* active_seeds,
                          int point_count, int state_count, double threshold,
                          double* global_state, unsigned long long* best_key) {
    for (int offset = 0; offset < active_count; offset += batch_rows) {
        const int rows = std::min(batch_rows, active_count - offset);
        if (shared_state) {
            evaluateSeeds<PrecomputedDistances, CompactCandidates, true>
                <<<rows, CUDA_BLOCK_SIZE, shared_bytes>>>(
                    points, distances, neighbors, neighbor_counts, clustered,
                    active_seeds, offset, point_count, state_count, threshold,
                    nullptr, best_key);
        } else {
            evaluateSeeds<PrecomputedDistances, CompactCandidates, false>
                <<<rows, CUDA_BLOCK_SIZE>>>(
                    points, distances, neighbors, neighbor_counts, clustered,
                    active_seeds, offset, point_count, state_count, threshold,
                    global_state, best_key);
        }
        CUDA_CHECK(cudaGetLastError());
    }
}

template <bool PrecomputedDistances, bool CompactCandidates>
void launchReplay(bool shared_state, size_t shared_bytes, const Point* points,
                  const double* distances, const int* neighbors,
                  const int* neighbor_counts, const unsigned char* clustered,
                  const unsigned long long* best_key, int point_count,
                  int state_count, double threshold, double* global_state,
                  int* members, int* member_count, int* winning_seed,
                  int* invariant_error) {
    if (shared_state) {
        replayWinningSeed<PrecomputedDistances, CompactCandidates, true>
            <<<1, CUDA_BLOCK_SIZE, shared_bytes>>>(
                points, distances, neighbors, neighbor_counts, clustered, best_key,
                point_count, state_count, threshold, nullptr, members, member_count,
                winning_seed, invariant_error);
    } else {
        replayWinningSeed<PrecomputedDistances, CompactCandidates, false>
            <<<1, CUDA_BLOCK_SIZE>>>(
                points, distances, neighbors, neighbor_counts, clustered, best_key,
                point_count, state_count, threshold, global_state, members,
                member_count, winning_seed, invariant_error);
    }
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace

// Main QT clustering algorithm. All candidate construction, winner selection,
// membership updates, and active-set maintenance execute on the GPU.
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));

    DeviceBuffer<Point> device_points(N);
    DeviceBuffer<unsigned char> clustered(N);
    DeviceBuffer<int> active_a(N);
    DeviceBuffer<int> active_b(N);
    DeviceBuffer<int> active_count_device(1);
    DeviceBuffer<int> clustered_members(N);
    DeviceBuffer<int> cluster_sizes(N);
    DeviceBuffer<int> cluster_seeds(N);
    DeviceBuffer<int> invariant_error(1);
    DeviceBuffer<unsigned long long> best_key(1);

    CUDA_CHECK(cudaMemcpy(device_points.get(), points.data(),
                          static_cast<size_t>(N) * sizeof(Point),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(clustered.get(), 0, static_cast<size_t>(N)));
    CUDA_CHECK(cudaMemset(invariant_error.get(), 0, sizeof(int)));

    const int utility_grid = boundedGridSize(N, properties.multiProcessorCount);
    initializeActiveSeeds<<<utility_grid, CUDA_BLOCK_SIZE>>>(active_a.get(), N);
    CUDA_CHECK(cudaGetLastError());

    // A precomputed row-major matrix makes every block's distance loads
    // coalesced and avoids recalculating square roots over many outer rounds.
    DeviceBuffer<double> distance_matrix;
    bool have_distance_matrix = false;
    const size_t n_size = static_cast<size_t>(N);
    size_t matrix_elements = 0;
    if (n_size <= std::numeric_limits<size_t>::max() / n_size &&
        n_size * n_size <= std::numeric_limits<size_t>::max() / sizeof(double)) {
        matrix_elements = n_size * n_size;
        const size_t matrix_bytes = matrix_elements * sizeof(double);
        size_t free_memory = 0;
        size_t total_memory = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_memory, &total_memory));
        (void)total_memory;
        if (matrix_bytes <= free_memory / 2 &&
            distance_matrix.tryAllocate(matrix_elements)) {
            const int matrix_grid =
                boundedGridSize(matrix_elements, properties.multiProcessorCount, 32);
            buildDistanceMatrix<<<matrix_grid, CUDA_BLOCK_SIZE>>>(
                device_points.get(), distance_matrix.get(), N, matrix_elements);
            CUDA_CHECK(cudaGetLastError());
            have_distance_matrix = true;
        }
    }

    DeviceBuffer<int> neighbor_matrix;
    DeviceBuffer<int> neighbor_counts;
    DeviceBuffer<int> maximum_neighbor_count(1);
    bool have_neighbor_lists = false;
    int state_count = N;
    if (have_distance_matrix && neighbor_matrix.tryAllocate(matrix_elements)) {
        neighbor_counts.allocate(n_size);
        CUDA_CHECK(cudaMemset(maximum_neighbor_count.get(), 0, sizeof(int)));
        buildNeighborLists<<<N, CUDA_BLOCK_SIZE>>>(
            distance_matrix.get(), N, threshold, neighbor_matrix.get(),
            neighbor_counts.get(), maximum_neighbor_count.get());
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(&state_count, maximum_neighbor_count.get(), sizeof(int),
                              cudaMemcpyDeviceToHost));
        have_neighbor_lists = true;
    }

    const size_t shared_bytes = static_cast<size_t>(state_count) * sizeof(double);
    const size_t shared_limit =
        static_cast<size_t>(properties.sharedMemPerBlockOptin);
    // Leave room for the statically allocated reduction and control values.
    const bool use_shared_state = shared_bytes + 4096 <= shared_limit;

    DeviceBuffer<double> global_state;
    int batch_rows = N;
    if (use_shared_state) {
        const int dynamic_bytes = static_cast<int>(shared_bytes);
        if (have_neighbor_lists) {
            CUDA_CHECK(cudaFuncSetAttribute(
                evaluateSeeds<true, true, true>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_bytes));
            CUDA_CHECK(cudaFuncSetAttribute(
                replayWinningSeed<true, true, true>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_bytes));
        } else if (have_distance_matrix) {
            CUDA_CHECK(cudaFuncSetAttribute(
                evaluateSeeds<true, false, true>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_bytes));
            CUDA_CHECK(cudaFuncSetAttribute(
                replayWinningSeed<true, false, true>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_bytes));
        } else {
            CUDA_CHECK(cudaFuncSetAttribute(
                evaluateSeeds<false, false, true>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_bytes));
            CUDA_CHECK(cudaFuncSetAttribute(
                replayWinningSeed<false, false, true>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_bytes));
        }
    } else {
        size_t free_memory = 0;
        size_t total_memory = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_memory, &total_memory));
        (void)total_memory;
        const size_t row_bytes = static_cast<size_t>(state_count) * sizeof(double);
        const size_t affordable_rows = std::max<size_t>(1, (free_memory / 2) / row_bytes);
        batch_rows = static_cast<int>(std::min<size_t>(n_size, affordable_rows));
        while (batch_rows > 0 &&
               !global_state.tryAllocate(static_cast<size_t>(batch_rows) * state_count)) {
            batch_rows /= 2;
        }
        if (batch_rows == 0) {
            cudaFailure(cudaErrorMemoryAllocation, "candidate state allocation",
                        __FILE__, __LINE__);
        }
    }

    int active_count = N;
    int cluster_count = 0;
    int* current_active = active_a.get();
    int* next_active = active_b.get();

    while (active_count > 0) {
        CUDA_CHECK(cudaMemset(best_key.get(), 0, sizeof(unsigned long long)));
        if (have_neighbor_lists) {
            launchSeedEvaluation<true, true>(
                use_shared_state, active_count, batch_rows, shared_bytes,
                device_points.get(), distance_matrix.get(), neighbor_matrix.get(),
                neighbor_counts.get(), clustered.get(), current_active, N,
                state_count, threshold, global_state.get(), best_key.get());
        } else if (have_distance_matrix) {
            launchSeedEvaluation<true, false>(
                use_shared_state, active_count, batch_rows, shared_bytes,
                device_points.get(), distance_matrix.get(), nullptr, nullptr,
                clustered.get(), current_active, N, state_count, threshold,
                global_state.get(), best_key.get());
        } else {
            launchSeedEvaluation<false, false>(
                use_shared_state, active_count, batch_rows, shared_bytes,
                device_points.get(), nullptr, nullptr, nullptr, clustered.get(),
                current_active, N, state_count, threshold, global_state.get(),
                best_key.get());
        }

        const int output_offset = N - active_count;

        if (have_neighbor_lists) {
            launchReplay<true, true>(
                use_shared_state, shared_bytes, device_points.get(),
                distance_matrix.get(), neighbor_matrix.get(), neighbor_counts.get(),
                clustered.get(), best_key.get(), N, state_count, threshold,
                global_state.get(), clustered_members.get() + output_offset,
                cluster_sizes.get() + cluster_count,
                cluster_seeds.get() + cluster_count, invariant_error.get());
        } else if (have_distance_matrix) {
            launchReplay<true, false>(
                use_shared_state, shared_bytes, device_points.get(),
                distance_matrix.get(), nullptr, nullptr, clustered.get(),
                best_key.get(), N, state_count, threshold, global_state.get(),
                clustered_members.get() + output_offset,
                cluster_sizes.get() + cluster_count,
                cluster_seeds.get() + cluster_count, invariant_error.get());
        } else {
            launchReplay<false, false>(
                use_shared_state, shared_bytes, device_points.get(), nullptr, nullptr,
                nullptr, clustered.get(), best_key.get(), N, state_count, threshold,
                global_state.get(), clustered_members.get() + output_offset,
                cluster_sizes.get() + cluster_count,
                cluster_seeds.get() + cluster_count, invariant_error.get());
        }

        const int member_grid =
            boundedGridSize(active_count, properties.multiProcessorCount);
        markClustered<<<member_grid, CUDA_BLOCK_SIZE>>>(
            clustered_members.get() + output_offset,
            cluster_sizes.get() + cluster_count, clustered.get());
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemset(active_count_device.get(), 0, sizeof(int)));
        const int compact_grid =
            boundedGridSize(active_count, properties.multiProcessorCount);
        compactActiveSeeds<<<compact_grid, CUDA_BLOCK_SIZE>>>(
            current_active, active_count, clustered.get(), next_active,
            active_count_device.get());
        CUDA_CHECK(cudaGetLastError());
        int next_active_count = 0;
        CUDA_CHECK(cudaMemcpy(&next_active_count, active_count_device.get(), sizeof(int),
                              cudaMemcpyDeviceToHost));
        if (next_active_count >= active_count) {
            std::fprintf(stderr, "CUDA clustering failed to make progress\n");
            std::exit(EXIT_FAILURE);
        }
        active_count = next_active_count;
        ++cluster_count;
        std::swap(current_active, next_active);
    }

    int host_invariant_error = 0;
    CUDA_CHECK(cudaMemcpy(&host_invariant_error, invariant_error.get(), sizeof(int),
                          cudaMemcpyDeviceToHost));
    if (host_invariant_error != 0) {
        std::fprintf(stderr, "CUDA clustering replay invariant failed\n");
        std::exit(EXIT_FAILURE);
    }

    std::vector<int> host_members(N);
    std::vector<int> host_sizes(cluster_count);
    std::vector<int> host_seeds(cluster_count);
    CUDA_CHECK(cudaMemcpy(host_members.data(), clustered_members.get(),
                          n_size * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_sizes.data(), cluster_sizes.get(),
                          static_cast<size_t>(cluster_count) * sizeof(int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_seeds.data(), cluster_seeds.get(),
                          static_cast<size_t>(cluster_count) * sizeof(int),
                          cudaMemcpyDeviceToHost));

    std::vector<Cluster> clusters;
    clusters.reserve(cluster_count);
    int member_offset = 0;
    for (int cluster_index = 0; cluster_index < cluster_count; ++cluster_index) {
        Cluster cluster;
        cluster.seed_point = host_seeds[cluster_index];
        const int member_count = host_sizes[cluster_index];
        cluster.members.assign(host_members.begin() + member_offset,
                               host_members.begin() + member_offset + member_count);
        member_offset += member_count;
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
