// QT Clustering Benchmark - CUDA Parallel Version
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.

#include <algorithm>
#include <chrono>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
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

// CUDA owns the candidate construction, winner selection, and membership state.
// The serial dependency between winning clusters is retained, but independent
// seeds and the distance updates within each seed are evaluated in parallel.
namespace {

constexpr int WARP_SIZE = 32;
constexpr unsigned FULL_WARP = 0xffffffffu;
constexpr int SMALL_ITEMS = 4;
constexpr int SMALL_CAPACITY = WARP_SIZE * SMALL_ITEMS;
constexpr int BLOCK_SIZE = 128;

void cudaCheck(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
    }
}

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) {
        if (count > std::numeric_limits<size_t>::max() / sizeof(T)) {
            throw std::runtime_error("CUDA allocation size overflow");
        }
        if (count) cudaCheck(cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)),
                             "cudaMalloc");
    }
    ~DeviceBuffer() { if (data_) cudaFree(data_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* get() const { return data_; }
private:
    T* data_ = nullptr;
};

struct ClusteringState {
    int remaining;
    int cluster_count;
    int member_count;
};

struct ExecutionResources {
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    ClusteringState* status = nullptr;
    ExecutionResources() {
        cudaCheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");
        const cudaError_t result = cudaMallocHost(reinterpret_cast<void**>(&status), sizeof(*status));
        if (result != cudaSuccess) {
            cudaStreamDestroy(stream);
            cudaCheck(result, "cudaMallocHost");
        }
    }
    ~ExecutionResources() {
        if (executable) cudaGraphExecDestroy(executable);
        if (graph) cudaGraphDestroy(graph);
        if (status) cudaFreeHost(status);
        if (stream) cudaStreamDestroy(stream);
    }
};

__device__ __forceinline__ double gpuSquaredDistance(Point a, Point b) {
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    // Match the native host compiler's contraction with correctly rounded
    // double precision operations (no approximate/fast math).
#if defined(__FMA__)
    return __fma_rn(dx, dx, __dmul_rn(dy, dy));
#else
    return __dadd_rn(__dmul_rn(dx, dx), __dmul_rn(dy, dy));
#endif
}

// sqrt is monotone, so cache squared maxima instead of taking a sqrt for
// every distance update. Equal rounded roots can span a few adjacent double
// values: resolve those comparisons with correctly rounded sqrt, preserving
// the original point-index tie rule. Eight ulps conservatively covers both
// root rounding and the rounding of the squared threshold, including zero.
__device__ __forceinline__ bool nearbySquares(double a, double b) {
    const unsigned long long ai = static_cast<unsigned long long>(__double_as_longlong(a));
    const unsigned long long bi = static_cast<unsigned long long>(__double_as_longlong(b));
    return (ai > bi ? ai - bi : bi - ai) <= 8;
}

__device__ __forceinline__ bool withinThreshold(double square, double threshold,
                                               double threshold_square) {
    if (!(threshold > 0.0)) return false;
    if (nearbySquares(square, threshold_square)) return __dsqrt_rn(square) < threshold;
    return square < threshold_square;
}

__device__ __forceinline__ bool closer(double a, int ai, double b, int bi) {
    if (a == b) return ai < bi;
    if (nearbySquares(a, b)) {
        a = __dsqrt_rn(a);
        b = __dsqrt_rn(b);
        return a < b || (a == b && ai < bi);
    }
    return a < b;
}

__device__ __forceinline__ void warpClosest(double& value, int& index) {
    for (int delta = 16; delta; delta >>= 1) {
        const double other_value = __shfl_down_sync(FULL_WARP, value, delta);
        const int other_index = __shfl_down_sync(FULL_WARP, index, delta);
        if (closer(other_value, other_index, value, index)) {
            value = other_value;
            index = other_index;
        }
    }
}

// Cell width is at least the threshold, so eligible neighbors must lie in
// the same cell or one of its eight neighbors. Directed rounding preserves
// the bin boundary rule even for points immediately next to a boundary.
__device__ __forceinline__ int spatialCell(Point point, int axis, double width) {
    if (axis == 1) return 0;
    const int x = __double2int_rd(fmin(fmax(__ddiv_rd(point.x, width), 0.0), double(axis - 1)));
    const int y = __double2int_rd(fmin(fmax(__ddiv_rd(point.y, width), 0.0), double(axis - 1)));
    return y * axis + x;
}

__global__ void countCells(const Point* points, int n, int axis, double width, int* counts) {
    const size_t point = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (point < static_cast<size_t>(n)) atomicAdd(counts + spatialCell(points[point], axis, width), 1);
}

__global__ void scatterCells(const Point* points, int n, int axis, double width,
                             const int* offsets, int* cursors, int* indices) {
    const size_t point = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (point < static_cast<size_t>(n)) {
        const int cell = spatialCell(points[point], axis, width);
        indices[offsets[cell] + atomicAdd(cursors + cell, 1)] = static_cast<int>(point);
    }
}

// A row contains only points strictly within the threshold of its seed.
// No other point can ever enter this seed's candidate cluster. The extra
// entry for the seed also reserves enough space to cache the whole cluster.
__global__ void countNeighbors(const Point* points, int n, double threshold, int* counts,
                               const int* cell_offsets, const int* cell_points,
                               int axis, double width) {
    const double threshold_square = __dmul_rn(threshold, threshold);
    const int seed = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    if (seed >= n) return;
    const Point origin = points[seed];
    const int cell = spatialCell(origin, axis, width);
    const int x = cell % axis;
    const int y = cell / axis;
    int count = 0;
    for (int cy = max(0, y - 1); cy <= min(axis - 1, y + 1); ++cy) {
        for (int cx = max(0, x - 1); cx <= min(axis - 1, x + 1); ++cx) {
            const int bin = cy * axis + cx;
            const int begin = cell_offsets ? cell_offsets[bin] : 0;
            const int end = cell_offsets ? cell_offsets[bin + 1] : n;
            for (size_t pos = static_cast<size_t>(begin) + lane; pos < static_cast<size_t>(end);
                 pos += WARP_SIZE) {
                const int j = cell_points ? cell_points[pos] : static_cast<int>(pos);
                count += j != seed && withinThreshold(gpuSquaredDistance(points[j], origin),
                                                      threshold, threshold_square);
            }
        }
    }
    for (int delta = 16; delta; delta >>= 1)
        count += __shfl_down_sync(FULL_WARP, count, delta);
    if (lane == 0) counts[seed] = count + 1;
}

__global__ void buildNeighbors(const Point* points, int n, int seed_count, double threshold,
                               const size_t* offsets, int* neighbors, double* seed_squares,
                               const int* cell_offsets, const int* cell_points,
                               int axis, double width) {
    const double threshold_square = __dmul_rn(threshold, threshold);
    const int seed = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    if (seed >= seed_count) return;
    const size_t base = offsets[seed];
    const Point origin = points[seed];
    const int cell = spatialCell(origin, axis, width);
    const int x = cell % axis;
    const int y = cell / axis;
    if (lane == 0) {
        neighbors[base] = seed;
        seed_squares[base] = DBL_MAX;
    }
    size_t written = 1;
    for (int cy = max(0, y - 1); cy <= min(axis - 1, y + 1); ++cy) {
        for (int cx = max(0, x - 1); cx <= min(axis - 1, x + 1); ++cx) {
            const int bin = cy * axis + cx;
            const int begin = cell_offsets ? cell_offsets[bin] : 0;
            const int end = cell_offsets ? cell_offsets[bin + 1] : n;
            for (size_t tile = begin; tile < static_cast<size_t>(end); tile += WARP_SIZE) {
                const size_t pos = tile + lane;
                const int j = pos < static_cast<size_t>(end)
                              ? (cell_points ? cell_points[pos] : static_cast<int>(pos)) : seed;
                const double d = j != seed ? gpuSquaredDistance(points[j], origin) : DBL_MAX;
                const bool eligible = j != seed && withinThreshold(d, threshold, threshold_square);
                const unsigned mask = __ballot_sync(FULL_WARP, eligible);
                if (eligible) {
                    const int rank = __popc(mask & ((1u << lane) - 1u));
                    neighbors[base + written + rank] = j;
                    seed_squares[base + written + rank] = d;
                }
                written += __popc(mask);
            }
        }
    }
}

// One warp per small neighborhood. Candidate distances and coordinates stay
// in registers throughout the greedy construction, with no block barriers.
__global__ void evaluateSmall(const Point* points, int seed_count, double threshold,
                              const size_t* offsets, const int* neighbors,
                              const double* seed_squares, const int* assignments,
                              int* members, int* cardinalities, const ClusteringState* state) {
    const double threshold_square = __dmul_rn(threshold, threshold);
    const int seed = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    if (seed >= seed_count || state->remaining == 0) return;
    if (assignments[seed] >= 0) {
        if (lane == 0) cardinalities[seed] = 0;
        return;
    }
    const size_t base = offsets[seed];
    const int degree = static_cast<int>(offsets[seed + 1] - base);
    if (degree > SMALL_CAPACITY) return;

    const int previous_size = cardinalities[seed];
    bool invalid = previous_size == 0;
    for (int j = lane; j < previous_size; j += WARP_SIZE)
        invalid |= assignments[members[base + j]] >= 0;
    // Removing points outside an existing greedy cluster cannot change any
    // of its choices. Recompute precisely those clusters that lost a member.
    if (!__any_sync(FULL_WARP, invalid)) return;

    int ids[SMALL_ITEMS];
    Point coordinates[SMALL_ITEMS];
    double maxima[SMALL_ITEMS];
#pragma unroll
    for (int item = 0; item < SMALL_ITEMS; ++item) {
        const int j = lane + item * WARP_SIZE;
        ids[item] = j < degree ? neighbors[base + j] : seed;
        coordinates[item] = points[ids[item]];
        maxima[item] = j < degree && ids[item] != seed && assignments[ids[item]] < 0
                       ? seed_squares[base + j] : DBL_MAX;
    }
    if (lane == 0) members[base] = seed;
    int size = 1;
    while (true) {
        double best_distance = DBL_MAX;
        int best = INT_MAX;
#pragma unroll
        for (int item = 0; item < SMALL_ITEMS; ++item) {
            if (maxima[item] < DBL_MAX &&
                closer(maxima[item], ids[item], best_distance, best)) {
                best_distance = maxima[item];
                best = ids[item];
            }
        }
        warpClosest(best_distance, best);
        best = __shfl_sync(FULL_WARP, best, 0);
        if (best == INT_MAX) break;
        if (lane == 0) members[base + size] = best;
        ++size;
        const Point latest = points[best];
#pragma unroll
        for (int item = 0; item < SMALL_ITEMS; ++item) {
            if (ids[item] == best) maxima[item] = DBL_MAX;
            if (maxima[item] < DBL_MAX) {
                maxima[item] = fmax(maxima[item], gpuSquaredDistance(coordinates[item], latest));
                if (!withinThreshold(maxima[item], threshold, threshold_square)) maxima[item] = DBL_MAX;
            }
        }
    }
    if (lane == 0) cardinalities[seed] = size;
}

// Larger rows cooperate across a block. Use shared memory where possible,
// and device scratch for rows that exceed the hardware shared memory limit.
__global__ void evaluateLarge(const Point* points, int seed_count, double threshold,
                              const size_t* offsets, const int* neighbors,
                              const double* seed_squares, const int* assignments,
                              int* members, int* cardinalities, const ClusteringState* state,
                              double* scratch) {
    const double threshold_square = __dmul_rn(threshold, threshold);
    const int seed = blockIdx.x;
    const int tid = threadIdx.x;
    if (seed >= seed_count || state->remaining == 0) return;
    if (assignments[seed] >= 0) {
        if (tid == 0) cardinalities[seed] = 0;
        return;
    }
    const size_t base = offsets[seed];
    const int degree = static_cast<int>(offsets[seed + 1] - base);
    if (degree <= SMALL_CAPACITY) return;
    const int previous_size = cardinalities[seed];
    bool invalid = previous_size == 0;
    for (int j = tid; j < previous_size; j += BLOCK_SIZE)
        invalid |= assignments[members[base + j]] >= 0;
    if (!__syncthreads_or(invalid)) return;

    extern __shared__ double shared_maxima[];
    double* maxima = scratch ? scratch + base : shared_maxima;
    __shared__ double warp_values[BLOCK_SIZE / WARP_SIZE];
    __shared__ int warp_indices[BLOCK_SIZE / WARP_SIZE];
    __shared__ int selected;
    for (int j = tid; j < degree; j += BLOCK_SIZE) {
        const int id = neighbors[base + j];
        maxima[j] = id != seed && assignments[id] < 0 ? seed_squares[base + j] : DBL_MAX;
    }
    if (tid == 0) members[base] = seed;
    int size = 1;
    int latest = seed;
    while (true) {
        const Point latest_point = points[latest];
        double best_distance = DBL_MAX;
        int best = INT_MAX;
        for (int j = tid; j < degree; j += BLOCK_SIZE) {
            double d = maxima[j];
            if (d == DBL_MAX) continue;
            const int id = neighbors[base + j];
            if (size > 1) {
                d = id == latest ? DBL_MAX : fmax(d, gpuSquaredDistance(points[id], latest_point));
                if (d == DBL_MAX || !withinThreshold(d, threshold, threshold_square)) d = DBL_MAX;
                maxima[j] = d;
            }
            if (d < DBL_MAX && closer(d, id, best_distance, best)) {
                best_distance = d;
                best = id;
            }
        }
        warpClosest(best_distance, best);
        if (tid % WARP_SIZE == 0) {
            warp_values[tid / WARP_SIZE] = best_distance;
            warp_indices[tid / WARP_SIZE] = best;
        }
        __syncthreads();
        if (tid < WARP_SIZE) {
            best_distance = tid < BLOCK_SIZE / WARP_SIZE ? warp_values[tid] : DBL_MAX;
            best = tid < BLOCK_SIZE / WARP_SIZE ? warp_indices[tid] : INT_MAX;
            warpClosest(best_distance, best);
            if (tid == 0) {
                selected = best;
                if (best != INT_MAX) members[base + size] = best;
            }
        }
        __syncthreads();
        latest = selected;
        if (latest == INT_MAX) break;
        ++size;
    }
    if (tid == 0) cardinalities[seed] = size;
}

__device__ __forceinline__ void warpLargest(int& size, int& seed) {
    for (int delta = 16; delta; delta >>= 1) {
        const int other_size = __shfl_down_sync(FULL_WARP, size, delta);
        const int other_seed = __shfl_down_sync(FULL_WARP, seed, delta);
        if (other_size > size || (other_size == size && other_seed < seed)) {
            size = other_size;
            seed = other_seed;
        }
    }
}

// The winner reduction uses the same first-seed tie rule as the serial loop.
// When all remaining clusters are singletons, finish them in index order in
// one launch rather than performing a separate round for every point.
__global__ void selectAndCommit(int n, int seed_count, const size_t* offsets, const int* members,
                                const int* cardinalities, int* assignments,
                                int* output_members, int* output_offsets, int* output_seeds,
                                ClusteringState* state) {
    if (state->remaining == 0) return;
    const int tid = threadIdx.x;
    __shared__ int warp_sizes[8];
    __shared__ int warp_seeds[8];
    __shared__ int best_seed;
    __shared__ int best_size;
    __shared__ int emitted;
    int size = 0;
    int seed = INT_MAX;
    for (int i = tid; i < seed_count; i += blockDim.x) {
        const int s = cardinalities[i];
        if (s > size || (s == size && i < seed)) { size = s; seed = i; }
    }
    warpLargest(size, seed);
    if (tid % WARP_SIZE == 0) {
        warp_sizes[tid / WARP_SIZE] = size;
        warp_seeds[tid / WARP_SIZE] = seed;
    }
    __syncthreads();
    if (tid < WARP_SIZE) {
        size = tid < 8 ? warp_sizes[tid] : 0;
        seed = tid < 8 ? warp_seeds[tid] : INT_MAX;
        warpLargest(size, seed);
        if (tid == 0) { best_size = size; best_seed = seed; emitted = 0; }
    }
    __syncthreads();
    const int cluster = state->cluster_count;
    const int start = state->member_count;
    if (best_size == 1) {
        for (size_t tile = 0; tile < static_cast<size_t>(n); tile += blockDim.x) {
            const size_t point = tile + tid;
            int prefix = point < static_cast<size_t>(n) && assignments[point] < 0;
            const int eligible = prefix;
            const int lane = tid % WARP_SIZE;
            for (int delta = 1; delta < WARP_SIZE; delta <<= 1) {
                const int other = __shfl_up_sync(FULL_WARP, prefix, delta);
                if (lane >= delta) prefix += other;
            }
            if (lane == WARP_SIZE - 1) warp_sizes[tid / WARP_SIZE] = prefix;
            __syncthreads();
            for (int w = 0; w < tid / WARP_SIZE; ++w) prefix += warp_sizes[w];
            if (eligible) {
                const int rank = emitted + prefix - 1;
                assignments[point] = cluster + rank;
                output_seeds[cluster + rank] = static_cast<int>(point);
                output_offsets[cluster + rank] = start + rank;
                output_members[start + rank] = static_cast<int>(point);
            }
            __syncthreads();
            if (tid == 0) {
                for (int w = 0; w < 8; ++w) emitted += warp_sizes[w];
            }
            __syncthreads();
        }
        if (tid == 0) {
            state->cluster_count = cluster + emitted;
            state->member_count = start + emitted;
            state->remaining = 0;
            output_offsets[cluster + emitted] = start + emitted;
        }
    } else {
        const size_t base = offsets[best_seed];
        for (int j = tid; j < best_size; j += blockDim.x) {
            const int point = members[base + j];
            output_members[start + j] = point;
            assignments[point] = cluster;
        }
        __syncthreads();
        if (tid == 0) {
            output_seeds[cluster] = best_seed;
            output_offsets[cluster] = start;
            output_offsets[cluster + 1] = start + best_size;
            state->cluster_count = cluster + 1;
            state->member_count = start + best_size;
            state->remaining -= best_size;
        }
    }
}

} // namespace

// All clustering decisions run on CUDA, including degenerate thresholds.
std::vector<Cluster> qtClustering(const std::vector<Point>& points, const double threshold) {
    const int n = static_cast<int>(points.size());
    if (n == 0) return {};
    int device = 0;
    cudaCheck(cudaGetDevice(&device), "cudaGetDevice");
    cudaDeviceProp properties;
    cudaCheck(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
    DeviceBuffer<Point> device_points(n);
    DeviceBuffer<int> counts(n);
    cudaCheck(cudaMemcpy(device_points.get(), points.data(), n * sizeof(Point), cudaMemcpyHostToDevice),
              "upload points");
    // For small inputs a direct GPU scan avoids index construction overhead.
    // Larger inputs use O(n) spatial indexing instead of scanning all pairs.
    const int axis = n >= 2048 && std::isfinite(threshold) && threshold > 0.0
                     ? std::max(1, static_cast<int>(std::min(std::sqrt(double(n)),
                                                             std::floor(MAX_WIDTH / threshold)))) : 1;
    const int cell_count = axis * axis;
    const double cell_width = axis > 1 ? std::max(threshold, MAX_WIDTH / axis) : MAX_WIDTH;
    DeviceBuffer<int> cell_offsets(axis > 1 ? static_cast<size_t>(cell_count) + 1 : 0);
    DeviceBuffer<int> cell_points(axis > 1 ? n : 0);
    if (axis > 1) {
        cudaCheck(cudaMemset(counts.get(), 0, cell_count * sizeof(int)), "initialize spatial cells");
        countCells<<<(static_cast<size_t>(n) + 255) / 256, 256>>>(
            device_points.get(), n, axis, cell_width, counts.get());
        cudaCheck(cudaGetLastError(), "countCells");
        std::vector<int> cell_sizes(cell_count), cell_starts(static_cast<size_t>(cell_count) + 1, 0);
        cudaCheck(cudaMemcpy(cell_sizes.data(), counts.get(), cell_count * sizeof(int),
                             cudaMemcpyDeviceToHost), "download spatial cell sizes");
        for (int cell = 0; cell < cell_count; ++cell)
            cell_starts[cell + 1] = cell_starts[cell] + cell_sizes[cell];
        cudaCheck(cudaMemcpy(cell_offsets.get(), cell_starts.data(), cell_starts.size() * sizeof(int),
                             cudaMemcpyHostToDevice), "upload spatial cell offsets");
        cudaCheck(cudaMemset(counts.get(), 0, cell_count * sizeof(int)), "initialize spatial cursors");
        scatterCells<<<(static_cast<size_t>(n) + 255) / 256, 256>>>(
            device_points.get(), n, axis, cell_width, cell_offsets.get(), counts.get(), cell_points.get());
        cudaCheck(cudaGetLastError(), "scatterCells");
    }
    const unsigned warp_blocks = static_cast<unsigned>((static_cast<size_t>(n) + 7) / 8);
    countNeighbors<<<warp_blocks, 256>>>(device_points.get(), n, threshold, counts.get(),
                                       cell_offsets.get(), cell_points.get(), axis, cell_width);
    cudaCheck(cudaGetLastError(), "countNeighbors");
    std::vector<int> host_counts(n);
    cudaCheck(cudaMemcpy(host_counts.data(), counts.get(), n * sizeof(int), cudaMemcpyDeviceToHost),
              "download neighborhood sizes");
    int max_degree = 0;
    bool complete = true;
    for (int count : host_counts) {
        max_degree = std::max(max_degree, count);
        complete &= count == n;
    }
    // A complete threshold graph guarantees cardinality n for every seed,
    // so the first seed wins. Construct its exact greedy member order only.
    const int seed_count = complete ? 1 : n;
    std::vector<size_t> host_offsets(static_cast<size_t>(seed_count) + 1, 0);
    for (int i = 0; i < seed_count; ++i) {
        if (host_offsets[i] > std::numeric_limits<size_t>::max() - host_counts[i])
            throw std::runtime_error("Neighborhood size overflow");
        host_offsets[i + 1] = host_offsets[i] + host_counts[i];
    }
    const size_t entries = host_offsets.back();
    DeviceBuffer<size_t> offsets(host_offsets.size());
    DeviceBuffer<int> neighbors(entries);
    DeviceBuffer<double> seed_squares(entries);
    DeviceBuffer<int> members(entries);
    DeviceBuffer<int> cardinalities(seed_count);
    DeviceBuffer<int> assignments(n);
    DeviceBuffer<int> output_members(n);
    DeviceBuffer<int> output_offsets(static_cast<size_t>(n) + 1);
    DeviceBuffer<int> output_seeds(n);
    DeviceBuffer<ClusteringState> state(1);
    const size_t shared_bytes = static_cast<size_t>(max_degree) * sizeof(double);
    const bool use_scratch = shared_bytes + 1024 > properties.sharedMemPerBlock;
    DeviceBuffer<double> scratch(max_degree > SMALL_CAPACITY && use_scratch ? entries : 0);
    cudaCheck(cudaMemcpy(offsets.get(), host_offsets.data(), host_offsets.size() * sizeof(size_t),
                         cudaMemcpyHostToDevice), "upload neighborhood offsets");
    cudaCheck(cudaMemset(cardinalities.get(), 0, seed_count * sizeof(int)), "initialize cardinalities");
    cudaCheck(cudaMemset(assignments.get(), 0xff, n * sizeof(int)), "initialize assignments");
    const ClusteringState initial{n, 0, 0};
    cudaCheck(cudaMemcpy(state.get(), &initial, sizeof(initial), cudaMemcpyHostToDevice), "initialize state");
    buildNeighbors<<<(static_cast<size_t>(seed_count) + 7) / 8, 256>>>(
        device_points.get(), n, seed_count, threshold, offsets.get(), neighbors.get(), seed_squares.get(),
        cell_offsets.get(), cell_points.get(), axis, cell_width);
    cudaCheck(cudaGetLastError(), "buildNeighbors");
    // The graph uses a nonblocking stream; explicitly finish preprocessing on
    // the default stream before that stream consumes its neighborhoods.
    cudaCheck(cudaDeviceSynchronize(), "prepare neighborhoods");
    ExecutionResources execution;
    auto launch_round = [&]() {
        evaluateSmall<<<(static_cast<size_t>(seed_count) + 7) / 8, 256, 0, execution.stream>>>(
            device_points.get(), seed_count, threshold, offsets.get(), neighbors.get(),
            seed_squares.get(), assignments.get(), members.get(), cardinalities.get(), state.get());
        if (max_degree > SMALL_CAPACITY) {
            evaluateLarge<<<seed_count, BLOCK_SIZE, use_scratch ? 0 : shared_bytes, execution.stream>>>(
                device_points.get(), seed_count, threshold, offsets.get(), neighbors.get(),
                seed_squares.get(), assignments.get(), members.get(), cardinalities.get(),
                state.get(), scratch.get());
        }
        selectAndCommit<<<1, 256, 0, execution.stream>>>(
            n, seed_count, offsets.get(), members.get(), cardinalities.get(), assignments.get(),
            output_members.get(), output_offsets.get(), output_seeds.get(), state.get());
    };
    if (complete || max_degree == 1) {
        launch_round();
        cudaCheck(cudaGetLastError(), "CUDA clustering kernels");
        cudaCheck(cudaMemcpyAsync(execution.status, state.get(), sizeof(initial), cudaMemcpyDeviceToHost,
                                  execution.stream), "download clustering state");
        cudaCheck(cudaStreamSynchronize(execution.stream), "finish clustering");
    } else {
        // Batch dependent rounds into a reusable graph. Kernel boundaries
        // synchronize the GPU work, with a single host check per eight rounds.
        cudaCheck(cudaStreamBeginCapture(execution.stream, cudaStreamCaptureModeThreadLocal),
                  "cudaStreamBeginCapture");
        for (int round = 0; round < 8; ++round) launch_round();
        cudaCheck(cudaStreamEndCapture(execution.stream, &execution.graph), "cudaStreamEndCapture");
        cudaCheck(cudaGraphInstantiate(&execution.executable, execution.graph, nullptr, nullptr, 0),
                  "cudaGraphInstantiate");
        do {
            cudaCheck(cudaGraphLaunch(execution.executable, execution.stream), "cudaGraphLaunch");
            cudaCheck(cudaMemcpyAsync(execution.status, state.get(), sizeof(initial), cudaMemcpyDeviceToHost,
                                      execution.stream), "download clustering state");
            cudaCheck(cudaStreamSynchronize(execution.stream), "finish clustering batch");
        } while (execution.status->remaining > 0);
    }
    const int cluster_count = execution.status->cluster_count;
    std::vector<int> result_members(n), result_offsets(static_cast<size_t>(cluster_count) + 1),
                     result_seeds(cluster_count);
    cudaCheck(cudaMemcpy(result_members.data(), output_members.get(), n * sizeof(int), cudaMemcpyDeviceToHost),
              "download members");
    cudaCheck(cudaMemcpy(result_offsets.data(), output_offsets.get(), result_offsets.size() * sizeof(int),
                         cudaMemcpyDeviceToHost), "download cluster offsets");
    cudaCheck(cudaMemcpy(result_seeds.data(), output_seeds.get(), cluster_count * sizeof(int),
                         cudaMemcpyDeviceToHost), "download seeds");
    std::vector<Cluster> clusters;
    clusters.reserve(cluster_count);
    for (int c = 0; c < cluster_count; ++c) {
        clusters.push_back({std::vector<int>(result_members.begin() + result_offsets[c],
                                            result_members.begin() + result_offsets[c + 1]),
                            result_seeds[c]});
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
    
    std::vector<Cluster> clusters;
    try {
        clusters = qtClustering(points, threshold);
    } catch (const std::exception& error) {
        fprintf(stderr, "CUDA clustering failed: %s\n", error.what());
        return 1;
    }
    
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
