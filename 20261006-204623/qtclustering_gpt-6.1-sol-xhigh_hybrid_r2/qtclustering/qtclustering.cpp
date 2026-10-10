// QT Clustering Benchmark - MPI / OpenMP / CUDA
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.

#include <algorithm>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <exception>
#include <vector>

#include <cuda_runtime.h>
#include <math_constants.h>
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

static int world_rank = 0;

static void mpiCheck(int status, const char* operation) {
    if (status == MPI_SUCCESS) return;
    char message[MPI_MAX_ERROR_STRING];
    int length = 0;
    MPI_Error_string(status, message, &length);
    fprintf(stderr, "Rank %d: %s: %.*s\n", world_rank, operation, length, message);
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}

static void cudaCheck(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return;
    fprintf(stderr, "Rank %d: %s: %s\n", world_rank, operation, cudaGetErrorString(status));
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}

#define CUDA_CHECK(operation) cudaCheck((operation), #operation)
#define MPI_CHECK(operation) mpiCheck((operation), #operation)

class MpiRuntime {
public:
    MpiRuntime(int& argc, char**& argv) {
        int provided = 0;
        if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS)
            std::exit(1);
        MPI_CHECK(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));
        MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));
        if (provided < MPI_THREAD_FUNNELED) {
            fprintf(stderr, "MPI_THREAD_FUNNELED is required.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    ~MpiRuntime() { MPI_Finalize(); }
    MpiRuntime(const MpiRuntime&) = delete;
    MpiRuntime& operator=(const MpiRuntime&) = delete;
};

template<class T> class DeviceBuffer {
public:
    T* data = nullptr;
    explicit DeviceBuffer(size_t count = 0) { allocate(count); }
    ~DeviceBuffer() { if (data) cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    void allocate(size_t count) {
        if (!count) return;
        if (count > std::numeric_limits<size_t>::max() / sizeof(T)) {
            fprintf(stderr, "Rank %d: device allocation size overflow\n", world_rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)));
    }
    void tryCache(size_t count) {
        const cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T));
        if (status == cudaErrorMemoryAllocation) {
            // The dense cache is optional; clustering still runs entirely on CUDA.
            data = nullptr;
            cudaGetLastError();
        } else {
            cudaCheck(status, "distance cache allocation");
        }
    }
};

template<class T> class PinnedBuffer {
public:
    T* data = nullptr;
    explicit PinnedBuffer(size_t count) {
        if (count) CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&data), count * sizeof(T)));
    }
    ~PinnedBuffer() { if (data) cudaFreeHost(data); }
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
};

class CudaStream {
public:
    cudaStream_t stream;
    CudaStream() { CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)); }
    ~CudaStream() { cudaStreamDestroy(stream); }
};

static int hostThreads(size_t work) {
    // Avoid creating large CPU teams for tiny GPU result buffers.
    return static_cast<int>(std::min<size_t>(omp_get_max_threads(), std::max<size_t>(1, (work + 255) / 256)));
}

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
        // The original generator never terminates for N <= 30.
        if (N <= 30) group_cnt = 1;
        
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
    return std::sqrt(std::fma(dx, dx, dy * dy));
}

// Use the same double arithmetic as the FMA-enabled host distance function.
// In particular, do not compare squared distances: sqrt rounding can create
// ties, and the sequential algorithm chooses the lowest point index on ties.
__device__ __forceinline__ double pointDistance(Point a, Point b) {
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    return __dsqrt_rn(__fma_rn(dx, dx, __dmul_rn(dy, dy)));
}

__device__ __forceinline__ double thresholdDistance(Point a, Point b, double squared_limit) {
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    const double squared = __fma_rn(dx, dx, __dmul_rn(dy, dy));
    // squared_limit is threshold^2 rounded UP. Above it, even a correctly
    // rounded sqrt cannot be less than threshold. Values near the boundary
    // still use the actual double sqrt for strict comparisons and tie rules.
    return squared > squared_limit ? CUDART_INF : __dsqrt_rn(squared);
}

__device__ __forceinline__ double cachedDistance(const Point* points,
                                                const double* distances,
                                                int n, int a, int b, double squared_limit) {
    return distances ? distances[static_cast<size_t>(a) * n + b]
                     : thresholdDistance(points[a], points[b], squared_limit);
}

// Compute each unordered pair once, and store both orientations. The dense
// cache is bounded by available device memory; sparse rows never depend on it.
// Infeasible pairs can use infinity because they will always be rejected.
__global__ void buildDistances(const Point* points, double* distances, int n, double threshold) {
    if (blockIdx.y > blockIdx.x) return;
    __shared__ Point column_points[32];
    __shared__ Point row_points[32];
    const int column = static_cast<int>(blockIdx.x) * 32 + threadIdx.x;
    if (threadIdx.y == 0 && column < n) column_points[threadIdx.x] = points[column];
    const int row_base = static_cast<int>(blockIdx.y) * 32;
    if (threadIdx.y == 1 && row_base + static_cast<int>(threadIdx.x) < n)
        row_points[threadIdx.x] = points[row_base + threadIdx.x];
    __syncthreads();
    const double squared_limit = __dmul_ru(threshold, threshold);
    for (int r = threadIdx.y; r < 32; r += 8) {
        const int row = row_base + r;
        if (column < n && row < n && column >= row) {
            const double d = thresholdDistance(column_points[threadIdx.x], row_points[r], squared_limit);
            distances[static_cast<size_t>(row) * n + column] = d;
            distances[static_cast<size_t>(column) * n + row] = d;
        }
    }
}

constexpr int GRAPH_THREADS = 128;
constexpr int GRAPH_WARPS = GRAPH_THREADS / 32;
using Offset = unsigned long long;

// Each rank owns cyclic seed rows. A seed can only accept points within its
// own threshold neighborhood, so this sparse graph is a safe candidate bound.
__global__ void countNeighbors(const Point* points, int n, double threshold,
                               int rank, int ranks, int* counts, int* warp_counts) {
    const int row = blockIdx.x;
    const int seed = static_cast<int>(static_cast<long long>(row) * ranks + rank);
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int begin = static_cast<int>(static_cast<long long>(n) * warp / GRAPH_WARPS);
    const int end = static_cast<int>(static_cast<long long>(n) * (warp + 1) / GRAPH_WARPS);
    int count = 0;
    const Point origin = points[seed];
    const double squared_limit = __dmul_ru(threshold, threshold);
    for (long long i = static_cast<long long>(begin) + lane; i < end; i += 32)
        count += i != seed && thresholdDistance(origin, points[i], squared_limit) < threshold;
    for (int offset = 16; offset; offset /= 2)
        count += __shfl_down_sync(0xffffffffu, count, offset);
    __shared__ int totals[GRAPH_WARPS];
    if (lane == 0) {
        totals[warp] = count;
        warp_counts[static_cast<size_t>(row) * GRAPH_WARPS + warp] = count;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int total = 0;
        for (int w = 0; w < GRAPH_WARPS; ++w) total += totals[w];
        counts[row] = total;
    }
}

// Warp ballots compact consecutive candidates into sorted CSR rows without
// atomics. Sorted point indices also coalesce later distance-cache reads.
__global__ void fillNeighbors(const Point* points, const double* distances,
                              int n, double threshold, int rank, int ranks,
                              const Offset* offsets, const int* warp_counts,
                              int* neighbors) {
    const int row = blockIdx.x;
    if (offsets[row] == offsets[row + 1]) return;
    const int seed = static_cast<int>(static_cast<long long>(row) * ranks + rank);
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int begin = static_cast<int>(static_cast<long long>(n) * warp / GRAPH_WARPS);
    const int end = static_cast<int>(static_cast<long long>(n) * (warp + 1) / GRAPH_WARPS);
    const double squared_limit = __dmul_ru(threshold, threshold);
    Offset position = offsets[row];
    for (int w = 0; w < warp; ++w)
        position += warp_counts[static_cast<size_t>(row) * GRAPH_WARPS + w];
    for (long long base = begin; base < end; base += 32) {
        const long long candidate = base + lane;
        const bool eligible = candidate < end && candidate != seed &&
            cachedDistance(points, distances, n, seed, static_cast<int>(candidate), squared_limit) < threshold;
        const unsigned mask = __ballot_sync(0xffffffffu, eligible);
        if (eligible)
            neighbors[position + __popc(mask & ((1u << lane) - 1u))] = static_cast<int>(candidate);
        position += __popc(mask);
    }
}

__device__ __forceinline__ bool closer(double d, int point, double best, int best_point) {
    return d < best || (d == best && point < best_point);
}

// Each block grows one seed independently. Maintain the maximum distance to
// the current cluster incrementally: adding one member requires one distance
// per remaining candidate, rather than rescanning all previous members.
// A cached cluster remains exact until one of its members is removed: deleting
// an unchosen candidate cannot change any of the previous greedy decisions.
template<int THREADS>
__global__ void buildCandidates(const Point* points, const double* distances,
                                int n, double threshold, int rank, int ranks,
                                bool complete_graph, const unsigned char* active,
                                const Offset* offsets, const int* neighbors,
                                double* maxima, int* members, int* cardinalities) {
    const int row = blockIdx.x;
    const int seed = static_cast<int>(static_cast<long long>(row) * ranks + rank);
    if (!active[seed] || (complete_graph && seed != 0)) {
        if (threadIdx.x == 0) cardinalities[row] = 0;
        return;
    }
    const Offset begin = offsets[row];
    const Offset end = offsets[row + 1];
    const Offset member_begin = begin + row;
    const int previous_size = cardinalities[row];
    int dirty = previous_size == 0;
    for (long long i = threadIdx.x; i < previous_size; i += THREADS)
        dirty |= !active[members[member_begin + i]];
    if (!__syncthreads_or(dirty)) return;

    for (Offset e = begin + threadIdx.x; e < end; e += THREADS)
        maxima[e] = active[neighbors[e]] ? 0.0 : DBL_MAX;
    if (threadIdx.x == 0) members[member_begin] = seed;
    __syncthreads();

    constexpr int WARPS = THREADS / 32;
    __shared__ double warp_distances[WARPS];
    __shared__ int warp_points[WARPS];
    __shared__ int next_point;
    int last = seed;
    int size = 1;
    const double squared_limit = __dmul_ru(threshold, threshold);
    while (true) {
        double best_distance = DBL_MAX;
        int best_point = INT_MAX;
        const Point last_point = points[last];
        for (Offset e = begin + threadIdx.x; e < end; e += THREADS) {
            double maximum = maxima[e];
            if (!(maximum < threshold) || maximum == DBL_MAX) continue;
            const int candidate = neighbors[e];
            if (candidate == last) {
                maxima[e] = DBL_MAX;
                continue;
            }
            const double d = distances ? distances[static_cast<size_t>(last) * n + candidate]
                                       : thresholdDistance(last_point, points[candidate], squared_limit);
            maximum = fmax(maximum, d);
            maxima[e] = maximum;
            if (maximum < threshold && closer(maximum, candidate, best_distance, best_point)) {
                best_distance = maximum;
                best_point = candidate;
            }
        }
        for (int offset = 16; offset; offset /= 2) {
            const double d = __shfl_down_sync(0xffffffffu, best_distance, offset);
            const int point = __shfl_down_sync(0xffffffffu, best_point, offset);
            if (closer(d, point, best_distance, best_point)) {
                best_distance = d;
                best_point = point;
            }
        }
        const int lane = threadIdx.x % 32;
        const int warp = threadIdx.x / 32;
        if (lane == 0) {
            warp_distances[warp] = best_distance;
            warp_points[warp] = best_point;
        }
        __syncthreads();
        if (warp == 0) {
            best_distance = lane < WARPS ? warp_distances[lane] : DBL_MAX;
            best_point = lane < WARPS ? warp_points[lane] : INT_MAX;
            for (int offset = 16; offset; offset /= 2) {
                const double d = __shfl_down_sync(0xffffffffu, best_distance, offset);
                const int point = __shfl_down_sync(0xffffffffu, best_point, offset);
                if (closer(d, point, best_distance, best_point)) {
                    best_distance = d;
                    best_point = point;
                }
            }
            if (lane == 0) {
                next_point = best_point;
                if (best_point != INT_MAX) members[member_begin + size] = best_point;
                else cardinalities[row] = size;
            }
        }
        __syncthreads();
        last = next_point;
        if (last == INT_MAX) break;
        ++size;
    }
}

__global__ void removeMembers(unsigned char* active, const int* members, int count) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < static_cast<size_t>(count)) active[members[i]] = 0;
}

static std::vector<Offset> rowOffsets(const int* counts, int rows) {
    std::vector<Offset> offsets(static_cast<size_t>(rows) + 1);
    const int threads = hostThreads(rows);
    std::vector<Offset> totals(threads);
    #pragma omp parallel num_threads(threads)
    {
        const int tid = omp_get_thread_num();
        const int team = omp_get_num_threads();
        const int begin = static_cast<int>(static_cast<long long>(rows) * tid / team);
        const int end = static_cast<int>(static_cast<long long>(rows) * (tid + 1) / team);
        Offset sum = 0;
        for (int row = begin; row < end; ++row) {
            offsets[row] = sum;
            sum += counts[row];
        }
        totals[tid] = sum;
        #pragma omp barrier
        #pragma omp single
        {
            Offset prefix = 0;
            for (int t = 0; t < team; ++t) {
                const Offset value = totals[t];
                totals[t] = prefix;
                prefix += value;
            }
            offsets[rows] = prefix;
        }
        #pragma omp barrier
        for (int row = begin; row < end; ++row) offsets[row] += totals[tid];
    }
    return offsets;
}

// MPI communicates only a winning (size, seed) pair and its members each round.
// Points, active masks and candidate state stay on each rank's GPU. One rank
// per visible GPU is the preferred accelerator-cluster launch configuration.
std::vector<Cluster> qtClustering(const std::vector<Point>& points, const double threshold) {
    int ranks = 1;
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
    const int n = static_cast<int>(points.size());
    const int rows = n / ranks + (world_rank < n % ranks);
    CudaStream execution;
    const cudaStream_t stream = execution.stream;
    DeviceBuffer<Point> device_points(n);
    DeviceBuffer<unsigned char> active(n);
    DeviceBuffer<int> cardinalities(rows), warp_counts(static_cast<size_t>(rows) * GRAPH_WARPS);
    DeviceBuffer<int> removed(n);
    PinnedBuffer<int> host_counts(rows), host_members(n);
    CUDA_CHECK(cudaMemcpyAsync(device_points.data, points.data(), n * sizeof(Point), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemsetAsync(active.data, 1, n, stream));
    if (rows) {
        countNeighbors<<<rows, GRAPH_THREADS, 0, stream>>>(device_points.data, n, threshold,
            world_rank, ranks, cardinalities.data, warp_counts.data);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(host_counts.data, cardinalities.data, rows * sizeof(int), cudaMemcpyDeviceToHost, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    int minimum_degree = INT_MAX, maximum_degree = 0;
    #pragma omp parallel for num_threads(hostThreads(rows)) reduction(min:minimum_degree) reduction(max:maximum_degree)
    for (int row = 0; row < rows; ++row) {
        minimum_degree = std::min(minimum_degree, host_counts.data[row]);
        maximum_degree = std::max(maximum_degree, host_counts.data[row]);
    }
    const int degree_bounds[2] = {-minimum_degree, maximum_degree};
    int global_bounds[2];
    MPI_CHECK(MPI_Allreduce(degree_bounds, global_bounds, 2, MPI_INT, MPI_MAX, MPI_COMM_WORLD));
    if (global_bounds[1] == 0) {
        // All candidates are singletons. Every round would choose the lowest
        // remaining index, so avoid N identical collective selection rounds.
        std::vector<Cluster> singletons(world_rank == 0 ? n : 0);
        if (world_rank == 0) {
            #pragma omp parallel for num_threads(hostThreads(n))
            for (int seed = 0; seed < n; ++seed) singletons[seed] = {{seed}, seed};
        }
        return singletons;
    }
    const bool complete_graph = -global_bounds[0] == n - 1;
    if (complete_graph) {
        // Every point can join every cluster. Seed zero must win the size tie;
        // grow it once to retain the exact greedy member order.
        #pragma omp parallel for num_threads(hostThreads(rows))
        for (int row = 0; row < rows; ++row)
            if (row != 0 || world_rank != 0) host_counts.data[row] = 0;
    }
    const std::vector<Offset> offsets = rowOffsets(host_counts.data, rows);
    const size_t edges = static_cast<size_t>(offsets.back());
    DeviceBuffer<Offset> device_offsets(offsets.size());
    DeviceBuffer<int> neighbors(edges), members(edges + rows);
    DeviceBuffer<double> maxima(edges), distances;
    CUDA_CHECK(cudaMemcpyAsync(device_offsets.data, offsets.data(), offsets.size() * sizeof(Offset), cudaMemcpyHostToDevice, stream));

    size_t free_bytes = 0, total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    const size_t point_count = static_cast<size_t>(n);
    // Required CSR buffers have already been allocated. Leave headroom for the
    // runtime and other jobs; an absent dense cache changes no decisions.
    if (edges && (!complete_graph || world_rank == 0) &&
        point_count <= std::numeric_limits<size_t>::max() / point_count / sizeof(double)) {
        const size_t cache_entries = point_count * point_count;
        if (cache_entries * sizeof(double) <= free_bytes / 3 * 2) {
            distances.tryCache(cache_entries);
            if (distances.data) {
                const unsigned tiles = static_cast<unsigned>((static_cast<size_t>(n) + 31) / 32);
                buildDistances<<<dim3(tiles, tiles), dim3(32, 8), 0, stream>>>(device_points.data, distances.data, n, threshold);
                CUDA_CHECK(cudaGetLastError());
            }
        }
    }
    if (rows) {
        fillNeighbors<<<rows, GRAPH_THREADS, 0, stream>>>(device_points.data, distances.data,
            n, threshold, world_rank, ranks, device_offsets.data, warp_counts.data, neighbors.data);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemsetAsync(cardinalities.data, 0, rows * sizeof(int), stream));
    }

    std::vector<Cluster> clusters;
    int remaining = n;
    const bool small_rows = rows == 0 || edges / static_cast<size_t>(rows) <= 128;
    while (remaining > 0) {
        if (rows) {
            if (small_rows)
                buildCandidates<64><<<rows, 64, 0, stream>>>(device_points.data, distances.data,
                    n, threshold, world_rank, ranks, complete_graph, active.data, device_offsets.data,
                    neighbors.data, maxima.data, members.data, cardinalities.data);
            else
                buildCandidates<128><<<rows, 128, 0, stream>>>(device_points.data, distances.data,
                    n, threshold, world_rank, ranks, complete_graph, active.data, device_offsets.data,
                    neighbors.data, maxima.data, members.data, cardinalities.data);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpyAsync(host_counts.data, cardinalities.data, rows * sizeof(int), cudaMemcpyDeviceToHost, stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        // A lexicographic integer key implements largest size / smallest seed.
        unsigned long long best_key = 0;
        #pragma omp parallel for num_threads(hostThreads(rows)) reduction(max:best_key)
        for (int row = 0; row < rows; ++row) {
            const int size = host_counts.data[row];
            if (size > 0) {
                const int seed = static_cast<int>(static_cast<long long>(row) * ranks + world_rank);
                const unsigned long long key = (static_cast<unsigned long long>(size) << 32) |
                    static_cast<unsigned>(INT_MAX - seed);
                best_key = std::max(best_key, key);
            }
        }
        const int local_best[2] = {static_cast<int>(best_key >> 32),
                                  INT_MAX - static_cast<int>(best_key & 0xffffffffull)};
        int global_best[2];
        MPI_CHECK(MPI_Allreduce(local_best, global_best, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD));
        const int size = global_best[0];
        const int seed = global_best[1];
        if (size <= 0 || size > remaining || seed < 0 || seed >= n) {
            fprintf(stderr, "Rank %d: invalid winning cluster\n", world_rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (size == 1) {
            // If the largest candidate is a singleton, no eligible pair
            // remains. Removing more points cannot change that; all later
            // winners are precisely the active points in ascending order.
            if (world_rank == 0) {
                std::vector<unsigned char> active_points(n);
                CUDA_CHECK(cudaMemcpyAsync(active_points.data(), active.data, n,
                    cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
                clusters.reserve(clusters.size() + remaining);
                for (int point = 0; point < n; ++point)
                    if (active_points[point]) clusters.push_back({{point}, point});
            }
            break;
        }
        const int owner = seed % ranks;
        if (world_rank == owner) {
            const int row = seed / ranks;
            CUDA_CHECK(cudaMemcpyAsync(host_members.data, members.data + offsets[row] + row,
                size * sizeof(int), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        MPI_CHECK(MPI_Bcast(host_members.data, size, MPI_INT, owner, MPI_COMM_WORLD));
        if (world_rank == 0)
            clusters.push_back({std::vector<int>(host_members.data, host_members.data + size), seed});
        remaining -= size;
        if (remaining) {
            CUDA_CHECK(cudaMemcpyAsync(removed.data, host_members.data, size * sizeof(int), cudaMemcpyHostToDevice, stream));
            removeMembers<<<(static_cast<size_t>(size) + 255) / 256, 256, 0, stream>>>(active.data, removed.data, size);
            CUDA_CHECK(cudaGetLastError());
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return clusters;
}

// Validation: check that clusters satisfy the QT clustering properties
bool validateClusters(const std::vector<Cluster>& clusters,
                     const std::vector<Point>& points,
                     const double threshold) {
    bool valid = true;
    
    printf("Validating clusters:\n");

    // Calculate diameters in parallel, then report in the original order.
    std::vector<double> diameters(clusters.size(), 0.0);
    #pragma omp parallel for schedule(dynamic, 1) num_threads(hostThreads(points.size()))
    for (size_t c = 0; c < clusters.size(); ++c) {
        const auto& cluster = clusters[c];
        double maximum = 0.0;
        for (size_t i = 0; i < cluster.members.size(); ++i)
            for (size_t j = i + 1; j < cluster.members.size(); ++j)
                maximum = std::max(maximum, distance(points[cluster.members[i]], points[cluster.members[j]]));
        diameters[c] = maximum;
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
    for (size_t i = 0; i < membership.size(); ++i) {
        if (membership[i] >= 0) clustered_count++;
    }
    
    printf("Total points: %zu, Clustered: %d, Unclustered: %zu\n",
           points.size(), clustered_count, points.size() - clustered_count);
    if (static_cast<size_t>(clustered_count) != points.size()) valid = false;
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
    MpiRuntime mpi(argc, argv);
    try {
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
                if (world_rank == 0) printUsage(argv[0]);
                return 0;
            } else {
                if (world_rank == 0) {
                    printf("Unknown option: %s\n", argv[i]);
                    printUsage(argv[0]);
                }
                return 1;
            }
        }

        if (num_points <= 0 || !(threshold > 0.0)) {
            if (world_rank == 0)
                printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n", num_points, threshold);
            return 1;
        }

        int ranks = 1;
        MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
        MPI_Comm node_comm;
        MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, world_rank, MPI_INFO_NULL, &node_comm));
        int local_rank = 0;
        MPI_CHECK(MPI_Comm_rank(node_comm, &local_rank));
        int device_count = 0;
        CUDA_CHECK(cudaGetDeviceCount(&device_count));
        if (device_count == 0) {
            fprintf(stderr, "Rank %d: a CUDA accelerator is required\n", world_rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        CUDA_CHECK(cudaSetDevice(local_rank % device_count));
        CUDA_CHECK(cudaFree(nullptr)); // Initialize the context before timing.
        MPI_CHECK(MPI_Comm_free(&node_comm));

        if (world_rank == 0) {
            printf("QT Clustering Benchmark\n");
            printf("Number of points: %d\n", num_points);
            printf("Distance threshold: %.2f\n", threshold);
            printf("Validation: %s\n", validate ? "enabled" : "disabled");
            printf("Hybrid execution: %d MPI rank(s), up to %d OpenMP thread(s) per rank, CUDA\n", ranks, omp_get_max_threads());
        }

        // Generate synthetic data
        std::vector<Point> points(num_points);
        if (world_rank == 0) generateSyntheticData(points, num_points);
        static_assert(sizeof(Point) == 2 * sizeof(double), "Point must contain two adjacent doubles");
        MPI_Datatype point_type;
        MPI_CHECK(MPI_Type_contiguous(2, MPI_DOUBLE, &point_type));
        MPI_CHECK(MPI_Type_commit(&point_type));
        MPI_CHECK(MPI_Bcast(points.data(), num_points, point_type, 0, MPI_COMM_WORLD));
        MPI_CHECK(MPI_Type_free(&point_type));

        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
        const double cluster_start = MPI_Wtime();
        const std::vector<Cluster> clusters = qtClustering(points, threshold);
        const double local_time = MPI_Wtime() - cluster_start;
        double time_sec = 0.0;
        MPI_CHECK(MPI_Reduce(&local_time, &time_sec, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
        if (world_rank != 0) return 0;

        printf("Clustering time: %lld ms\n", static_cast<long long>(time_sec * 1000.0));
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
            #pragma omp parallel for num_threads(hostThreads(num_points))
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
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", world_rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
