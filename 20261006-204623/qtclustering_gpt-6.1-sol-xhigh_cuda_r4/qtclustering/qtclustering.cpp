// QT Clustering Benchmark - CUDA Version
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
#include <vector>

#include <cuda_runtime.h>
#include <math_constants.h>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/device/device_scan.cuh>

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

void checkCuda(cudaError_t error) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}

// Allocations are reused across all clustering rounds.
template <typename T>
class DeviceBuffer {
public:
    T* data = nullptr;

    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t count) { allocate(count); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer() { if (data) cudaFree(data); }

    void allocate(size_t count) {
        if (count > std::numeric_limits<size_t>::max() / sizeof(T)) {
            fprintf(stderr, "CUDA allocation size overflow\n");
            std::exit(EXIT_FAILURE);
        }
        if (count) checkCuda(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)));
    }
};

struct Control {
    int total_members = 0;
    int cluster_count = 0;
    int selected_seed = -1;
    int selected_size = 1;
};

struct GpuProblem {
    const Point* points;
    const double* distances;
    const size_t* offsets;
    const int* neighbors;
    int* clustered;
    int* dirty;
    int* cardinalities;
    int* members;
    int* seeds;
    int* starts;
    Control* control;
    int n;
    double threshold;
    double squared_cutoff;

    __device__ int degree(int seed) const {
        return offsets ? static_cast<int>(offsets[seed + 1] - offsets[seed]) : n - 1;
    }

    __device__ int neighbor(int seed, int position) const {
        // A null adjacency list represents the complete threshold graph.
        return neighbors ? neighbors[offsets[seed] + position]
                         : position + (position >= seed);
    }

    __device__ double separation(int a, int b) const {
        if (distances) return distances[static_cast<size_t>(a) * n + b];
        const double dx = points[a].x - points[b].x;
        const double dy = points[a].y - points[b].y;
        // Match the native host compiler's distance expression, including
        // correctly rounded double precision square root and FMA contraction.
#ifdef __FMA__
        const double squared = __fma_rn(dx, dx, __dmul_rn(dy, dy));
#else
        const double squared = __dadd_rn(__dmul_rn(dx, dx), __dmul_rn(dy, dy));
#endif
        // The cutoff is rounded upward past threshold^2. Values above it
        // cannot pass the original sqrt(distance) < threshold comparison.
        // Keep the square root near the boundary and for every eligible pair
        // so that strict comparisons and rounded-distance ties stay exact.
        if (squared >= squared_cutoff) return CUDART_INF;
        return __dsqrt_rn(squared);
    }
};

constexpr int BLOCK_SIZE = 128;
constexpr int NO_POINT = 0x7fffffff;

struct Closest {
    double diameter;
    int point;
};

struct Closer {
    __device__ Closest operator()(Closest a, Closest b) const {
        return (b.diameter < a.diameter ||
                (b.diameter == a.diameter && b.point < a.point)) ? b : a;
    }
};

struct BestCluster {
    int size;
    int seed;
};

struct BetterCluster {
    __device__ BestCluster operator()(BestCluster a, BestCluster b) const {
        return (b.size > a.size || (b.size == a.size && b.seed < a.seed)) ? b : a;
    }
};

// Count the threshold graph and optionally cache the entire distance matrix.
// No squared-distance or single precision approximation is used: comparisons
// remain strict and rounding ties are resolved on the original distances.
__global__ void countNeighbors(GpuProblem p, size_t* degrees, int* max_degree,
                               double* distance_matrix) {
    const int seed = blockIdx.x;
    int count = 0;
    for (int candidate = threadIdx.x; candidate < p.n; candidate += blockDim.x) {
        const double d = p.separation(seed, candidate);
        if (distance_matrix)
            distance_matrix[static_cast<size_t>(seed) * p.n + candidate] = d;
        count += candidate != seed && d < p.threshold;
    }
    using Reduction = cub::BlockReduce<int, BLOCK_SIZE>;
    __shared__ typename Reduction::TempStorage temporary;
    count = Reduction(temporary).Sum(count);
    if (threadIdx.x == 0) {
        degrees[seed] = count;
        atomicMax(max_degree, count);
    }
}

// Compact neighbors in point-index order. Only these points can ever join a
// cluster grown from this seed; the other points need no per-candidate state.
__global__ void fillNeighbors(GpuProblem p, int* neighbors) {
    const int seed = blockIdx.x;
    using Scan = cub::BlockScan<int, BLOCK_SIZE>;
    __shared__ typename Scan::TempStorage temporary;
    __shared__ size_t written;
    if (threadIdx.x == 0) written = p.offsets[seed];
    __syncthreads();
    for (int base = 0; base < p.n; base += BLOCK_SIZE) {
        const int candidate = base + threadIdx.x;
        const int eligible = candidate < p.n && candidate != seed &&
                             p.separation(seed, candidate) < p.threshold;
        int position, count;
        Scan(temporary).ExclusiveSum(eligible, position, count);
        if (eligible) neighbors[written + position] = candidate;
        __syncthreads();
        if (threadIdx.x == 0) written += count;
        __syncthreads();
    }
}

template <bool RECORD>
__device__ void startCluster(GpuProblem p, int seed) {
    if constexpr (RECORD) {
        if (threadIdx.x == 0) {
            const int base = p.control->total_members;
            p.starts[p.control->cluster_count] = base;
            p.seeds[p.control->cluster_count] = seed;
            p.members[base] = seed;
            p.clustered[seed] = 1;
        }
    }
}

template <bool RECORD>
__device__ void appendMember(GpuProblem p, int member, int position) {
    if constexpr (RECORD) {
        if (threadIdx.x == 0) {
            p.members[p.control->total_members + position] = member;
            p.clustered[member] = 1;
        }
    }
}

template <bool RECORD>
__device__ void finishCluster(GpuProblem p, int size) {
    if constexpr (RECORD) {
        if (threadIdx.x == 0) {
            p.control->total_members += size;
            ++p.control->cluster_count;
            p.starts[p.control->cluster_count] = p.control->total_members;
        }
    }
}

// Small neighborhoods live entirely in registers. One warp grows each seed,
// with no block-wide barriers and no global writes inside the growth loop.
template <int ITEMS, bool RECORD>
__device__ int growWarp(GpuProblem p, int seed) {
    const int lane = threadIdx.x & 31;
    const int degree = p.degree(seed);
    int ids[ITEMS];
    double maxima[ITEMS];
#pragma unroll
    for (int item = 0; item < ITEMS; ++item) {
        const int position = lane + item * 32;
        ids[item] = position < degree ? p.neighbor(seed, position) : NO_POINT;
        maxima[item] = ids[item] != NO_POINT && !p.clustered[ids[item]]
                    ? p.separation(seed, ids[item]) : CUDART_INF;
    }

    startCluster<RECORD>(p, seed);
    int size = 1;
    while (true) {
        Closest closest{CUDART_INF, NO_POINT};
#pragma unroll
        for (int item = 0; item < ITEMS; ++item) {
            if (maxima[item] < CUDART_INF)
                closest = Closer{}(closest, {maxima[item], ids[item]});
        }
        for (int shift = 16; shift; shift >>= 1) {
            const Closest other{
                __shfl_down_sync(0xffffffff, closest.diameter, shift),
                __shfl_down_sync(0xffffffff, closest.point, shift)};
            closest = Closer{}(closest, other);
        }
        const int member = __shfl_sync(0xffffffff, closest.point, 0);
        if (member == NO_POINT) break;
        appendMember<RECORD>(p, member, size);
        ++size;
#pragma unroll
        for (int item = 0; item < ITEMS; ++item) {
            if (maxima[item] < CUDART_INF) {
                if (ids[item] == member) {
                    maxima[item] = CUDART_INF;
                } else {
                    const double d = p.separation(member, ids[item]);
                    maxima[item] = fmax(maxima[item], d);
                    if (!(maxima[item] < p.threshold)) maxima[item] = CUDART_INF;
                }
            }
        }
    }
    finishCluster<RECORD>(p, size);
    return size;
}

template <int ITEMS, bool RECORD>
__global__ void warpClusters(GpuProblem p) {
    const int seed = RECORD ? p.control->selected_seed
                            : (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    if (seed < 0 || seed >= p.n) return;
    if constexpr (RECORD) {
        if (p.control->selected_size <= 1) return;
    } else {
        if (p.clustered[seed] || !p.dirty[seed]) return;
    }
    const int size = growWarp<ITEMS, RECORD>(p, seed);
    if constexpr (!RECORD) {
        if ((threadIdx.x & 31) == 0) {
            p.cardinalities[seed] = size;
            p.dirty[seed] = 0;
        }
    }
}

// Larger neighborhoods use shared memory. If a neighborhood exceeds the
// hardware shared-memory limit, the same algorithm uses bounded batches of
// global scratch space instead of requiring quadratic scratch allocations.
template <bool RECORD, bool SHARED>
__global__ void blockClusters(GpuProblem p, int first_seed, int seed_count,
                              double* scratch, int scratch_stride) {
    const int seed = RECORD ? p.control->selected_seed : first_seed + blockIdx.x;
    if constexpr (RECORD) {
        if (p.control->selected_size <= 1) return;
    } else {
        if (blockIdx.x >= static_cast<unsigned int>(seed_count) ||
            p.clustered[seed] || !p.dirty[seed]) return;
    }
    extern __shared__ double shared_maxima[];
    double* maxima = SHARED ? shared_maxima
                           : scratch + static_cast<size_t>(blockIdx.x) * scratch_stride;
    const int degree = p.degree(seed);
    for (int i = threadIdx.x; i < degree; i += blockDim.x) {
        const int candidate = p.neighbor(seed, i);
        maxima[i] = !p.clustered[candidate] ? p.separation(seed, candidate) : CUDART_INF;
    }
    startCluster<RECORD>(p, seed);
    __syncthreads();

    using Reduction = cub::BlockReduce<Closest, BLOCK_SIZE>;
    __shared__ typename Reduction::TempStorage temporary;
    __shared__ int member;
    int size = 1;
    while (true) {
        Closest closest{CUDART_INF, NO_POINT};
        for (int i = threadIdx.x; i < degree; i += blockDim.x)
            if (maxima[i] < CUDART_INF)
                closest = Closer{}(closest, {maxima[i], p.neighbor(seed, i)});
        closest = Reduction(temporary).Reduce(closest, Closer{});
        if (threadIdx.x == 0) member = closest.point;
        __syncthreads();
        if (member == NO_POINT) break;
        appendMember<RECORD>(p, member, size);
        ++size;
        for (int i = threadIdx.x; i < degree; i += blockDim.x) {
            if (maxima[i] < CUDART_INF) {
                const int candidate = p.neighbor(seed, i);
                if (candidate == member) {
                    maxima[i] = CUDART_INF;
                } else {
                    maxima[i] = fmax(maxima[i], p.separation(member, candidate));
                    if (!(maxima[i] < p.threshold)) maxima[i] = CUDART_INF;
                }
            }
        }
        __syncthreads();
    }
    finishCluster<RECORD>(p, size);
    if constexpr (!RECORD) {
        if (threadIdx.x == 0) {
            p.cardinalities[seed] = size;
            p.dirty[seed] = 0;
        }
    }
}

__global__ void selectCluster(GpuProblem p) {
    BestCluster best{-1, NO_POINT};
    for (int seed = threadIdx.x; seed < p.n; seed += blockDim.x)
        if (!p.clustered[seed])
            best = BetterCluster{}(best, {p.cardinalities[seed], seed});
    using Reduction = cub::BlockReduce<BestCluster, BLOCK_SIZE>;
    __shared__ typename Reduction::TempStorage temporary;
    best = Reduction(temporary).Reduce(best, BetterCluster{});
    if (threadIdx.x == 0) {
        p.control->selected_seed = best.seed;
        p.control->selected_size = best.size;
    }
}

// A seed's greedy result can only change when one of its threshold neighbors
// is removed. Conservatively invalidate just these seeds; all other cached
// cardinalities remain exact, regardless of their growth order.
__global__ void invalidateNeighbors(GpuProblem p) {
    if (p.control->selected_size <= 1) return;
    const int first = p.starts[p.control->cluster_count - 1];
    for (int i = blockIdx.x; i < p.control->selected_size; i += gridDim.x) {
        const int member = p.members[first + i];
        const int degree = p.degree(member);
        for (int j = threadIdx.x; j < degree; j += blockDim.x)
            atomicExch(p.dirty + p.neighbor(member, j), 1);
    }
}

// Once the best size is one, every remaining seed is a singleton. Emit them
// together, in exactly the ascending seed order used by the sequential code.
__global__ void emitSingletons(GpuProblem p) {
    if (p.control->selected_size != 1) return;
    const int first_member = p.control->total_members;
    const int first_cluster = p.control->cluster_count;
    using Scan = cub::BlockScan<int, BLOCK_SIZE>;
    __shared__ typename Scan::TempStorage temporary;
    __shared__ int emitted;
    if (threadIdx.x == 0) emitted = 0;
    __syncthreads();
    for (int base = 0; base < p.n; base += BLOCK_SIZE) {
        const int seed = base + threadIdx.x;
        const int available = seed < p.n && !p.clustered[seed];
        int position, count;
        Scan(temporary).ExclusiveSum(available, position, count);
        if (available) {
            const int output = emitted + position;
            p.members[first_member + output] = seed;
            p.seeds[first_cluster + output] = seed;
            p.starts[first_cluster + output] = first_member + output;
            p.clustered[seed] = 1;
        }
        __syncthreads();
        if (threadIdx.x == 0) emitted += count;
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        p.control->total_members += emitted;
        p.control->cluster_count += emitted;
        p.starts[p.control->cluster_count] = p.control->total_members;
    }
}

} // namespace

// All candidate generation, winner selection and membership updates run on
// CUDA. The host only controls rounds and reconstructs the existing result API.
std::vector<Cluster> qtClustering(const std::vector<Point>& points, double threshold) {
    const int n = static_cast<int>(points.size());
    if (!n) return {};

    int device;
    checkCuda(cudaGetDevice(&device));
    cudaDeviceProp properties;
    checkCuda(cudaGetDeviceProperties(&properties, device));
    size_t free_bytes, total_bytes;
    checkCuda(cudaMemGetInfo(&free_bytes, &total_bytes));

    DeviceBuffer<Point> gpu_points(n);
    DeviceBuffer<double> distances;
    const size_t matrix_elements = static_cast<size_t>(n) * n;
    if (matrix_elements <= free_bytes / (4 * sizeof(double)))
        distances.allocate(matrix_elements);

    DeviceBuffer<size_t> degrees(static_cast<size_t>(n) + 1);
    DeviceBuffer<size_t> offsets(static_cast<size_t>(n) + 1);
    DeviceBuffer<int> max_degree_buffer(1);
    DeviceBuffer<int> neighbors;
    DeviceBuffer<int> clustered(n), dirty(n), cardinalities(n);
    DeviceBuffer<int> members(n), seeds(n), starts(static_cast<size_t>(n) + 1);
    DeviceBuffer<Control> control(1);
    DeviceBuffer<double> scratch;
    DeviceBuffer<unsigned char> scan_storage;

    cudaStream_t stream;
    checkCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    Control* status;
    checkCuda(cudaMallocHost(reinterpret_cast<void**>(&status), sizeof(Control)));
    *status = Control{};
    checkCuda(cudaMemcpyAsync(gpu_points.data, points.data(), n * sizeof(Point),
                              cudaMemcpyHostToDevice, stream));
    checkCuda(cudaMemsetAsync(clustered.data, 0, n * sizeof(int), stream));
    checkCuda(cudaMemsetAsync(dirty.data, 1, n * sizeof(int), stream));
    checkCuda(cudaMemsetAsync(degrees.data, 0, (static_cast<size_t>(n) + 1) * sizeof(size_t), stream));
    checkCuda(cudaMemsetAsync(max_degree_buffer.data, 0, sizeof(int), stream));
    checkCuda(cudaMemcpyAsync(control.data, status, sizeof(Control), cudaMemcpyHostToDevice, stream));

    GpuProblem p{gpu_points.data, nullptr, nullptr, nullptr, clustered.data,
                 dirty.data, cardinalities.data, members.data, seeds.data,
                 starts.data, control.data, n, threshold,
                 std::nextafter(threshold * threshold,
                                std::numeric_limits<double>::infinity())};
    countNeighbors<<<n, BLOCK_SIZE, 0, stream>>>(p, degrees.data, max_degree_buffer.data,
                                                distances.data);
    checkCuda(cudaGetLastError());
    p.distances = distances.data;
    size_t scan_bytes = 0;
    checkCuda(cub::DeviceScan::ExclusiveSum(nullptr, scan_bytes, degrees.data,
                                           offsets.data, n + 1, stream));
    scan_storage.allocate(scan_bytes);
    checkCuda(cub::DeviceScan::ExclusiveSum(scan_storage.data, scan_bytes, degrees.data,
                                           offsets.data, n + 1, stream));
    size_t edge_count;
    int max_degree;
    checkCuda(cudaMemcpyAsync(&edge_count, offsets.data + n, sizeof(size_t),
                              cudaMemcpyDeviceToHost, stream));
    checkCuda(cudaMemcpyAsync(&max_degree, max_degree_buffer.data, sizeof(int),
                              cudaMemcpyDeviceToHost, stream));
    checkCuda(cudaStreamSynchronize(stream));

    const bool complete = edge_count == static_cast<size_t>(n) * (n - 1);
    if (edge_count && !complete) {
        neighbors.allocate(edge_count);
        p.offsets = offsets.data;
        p.neighbors = neighbors.data;
        fillNeighbors<<<n, BLOCK_SIZE, 0, stream>>>(p, neighbors.data);
        checkCuda(cudaGetLastError());
    }

    // Limit shared memory per block to keep several seeds resident on each SM.
    const size_t shared_bytes = static_cast<size_t>(max_degree) * sizeof(double);
    const bool use_shared = shared_bytes <= std::min<size_t>(
        properties.sharedMemPerBlock - 1024, 32 * 1024);
    int batch_size = complete ? 1 : n;
    if (max_degree > 256 && !use_shared) {
        checkCuda(cudaMemGetInfo(&free_bytes, &total_bytes));
        batch_size = static_cast<int>(std::min<size_t>(batch_size,
                                                     free_bytes / (4 * shared_bytes)));
        if (batch_size < 1) {
            fprintf(stderr, "Insufficient GPU memory for a candidate cluster\n");
            std::exit(EXIT_FAILURE);
        }
        scratch.allocate(static_cast<size_t>(batch_size) * max_degree);
    }

    auto grow = [&](bool record) {
        const int blocks = record ? 1 : (n + 3) / 4;
        const int threads = record ? 32 : BLOCK_SIZE;
        if (max_degree <= 32) {
            if (record) warpClusters<1, true><<<blocks, threads, 0, stream>>>(p);
            else warpClusters<1, false><<<blocks, threads, 0, stream>>>(p);
        } else if (max_degree <= 64) {
            if (record) warpClusters<2, true><<<blocks, threads, 0, stream>>>(p);
            else warpClusters<2, false><<<blocks, threads, 0, stream>>>(p);
        } else if (max_degree <= 128) {
            if (record) warpClusters<4, true><<<blocks, threads, 0, stream>>>(p);
            else warpClusters<4, false><<<blocks, threads, 0, stream>>>(p);
        } else if (max_degree <= 256) {
            if (record) warpClusters<8, true><<<blocks, threads, 0, stream>>>(p);
            else warpClusters<8, false><<<blocks, threads, 0, stream>>>(p);
        } else if (use_shared) {
            if (record)
                blockClusters<true, true><<<1, BLOCK_SIZE, shared_bytes, stream>>>(
                    p, 0, 1, nullptr, 0);
            else
                blockClusters<false, true><<<n, BLOCK_SIZE, shared_bytes, stream>>>(
                    p, 0, n, nullptr, 0);
        } else {
            if (record)
                blockClusters<true, false><<<1, BLOCK_SIZE, 0, stream>>>(
                    p, 0, 1, scratch.data, max_degree);
            else
                for (int first = 0; first < n; first += batch_size)
                    blockClusters<false, false><<<std::min(batch_size, n - first),
                        BLOCK_SIZE, 0, stream>>>(p, first, std::min(batch_size, n - first),
                                                scratch.data, max_degree);
        }
        checkCuda(cudaGetLastError());
    };

    if (!edge_count) {
        emitSingletons<<<1, BLOCK_SIZE, 0, stream>>>(p);
    } else if (complete) {
        // Every seed would contain every point, so seed zero wins immediately.
        // Still grow it to preserve the exact member insertion order.
        status->selected_seed = 0;
        status->selected_size = n;
        checkCuda(cudaMemcpyAsync(control.data, status, sizeof(Control),
                                  cudaMemcpyHostToDevice, stream));
        grow(true);
    } else {
        // Batch dependent rounds in a replayable graph to amortize both launch
        // and host synchronization overhead. After all points are clustered,
        // the remaining rounds simply have no eligible seeds and do no work.
        cudaGraph_t graph;
        cudaGraphExec_t executable;
        checkCuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        for (int round = 0; round < 8; ++round) {
            grow(false);
            selectCluster<<<1, BLOCK_SIZE, 0, stream>>>(p);
            grow(true);
            invalidateNeighbors<<<std::min(n, properties.multiProcessorCount * 4),
                                  BLOCK_SIZE, 0, stream>>>(p);
            emitSingletons<<<1, BLOCK_SIZE, 0, stream>>>(p);
        }
        checkCuda(cudaGetLastError());
        checkCuda(cudaStreamEndCapture(stream, &graph));
        checkCuda(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        do {
            checkCuda(cudaGraphLaunch(executable, stream));
            checkCuda(cudaMemcpyAsync(status, control.data, sizeof(Control),
                                      cudaMemcpyDeviceToHost, stream));
            checkCuda(cudaStreamSynchronize(stream));
        } while (status->total_members < n);
        checkCuda(cudaGraphExecDestroy(executable));
        checkCuda(cudaGraphDestroy(graph));
    }

    checkCuda(cudaGetLastError());
    checkCuda(cudaMemcpyAsync(status, control.data, sizeof(Control),
                              cudaMemcpyDeviceToHost, stream));
    checkCuda(cudaStreamSynchronize(stream));
    std::vector<int> host_members(n);
    std::vector<int> host_seeds(status->cluster_count);
    std::vector<int> host_starts(static_cast<size_t>(status->cluster_count) + 1);
    checkCuda(cudaMemcpyAsync(host_members.data(), members.data, n * sizeof(int),
                              cudaMemcpyDeviceToHost, stream));
    checkCuda(cudaMemcpyAsync(host_seeds.data(), seeds.data, host_seeds.size() * sizeof(int),
                              cudaMemcpyDeviceToHost, stream));
    checkCuda(cudaMemcpyAsync(host_starts.data(), starts.data, host_starts.size() * sizeof(int),
                              cudaMemcpyDeviceToHost, stream));
    checkCuda(cudaStreamSynchronize(stream));
    checkCuda(cudaFreeHost(status));
    checkCuda(cudaStreamDestroy(stream));

    std::vector<Cluster> clusters(host_seeds.size());
    for (size_t i = 0; i < clusters.size(); ++i) {
        clusters[i].seed_point = host_seeds[i];
        clusters[i].members.assign(host_members.begin() + host_starts[i],
                                  host_members.begin() + host_starts[i + 1]);
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
