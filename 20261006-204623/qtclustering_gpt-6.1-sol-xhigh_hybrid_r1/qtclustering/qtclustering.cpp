// QT Clustering Benchmark - MPI / OpenMP / CUDA
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.

#include <algorithm>
#include <climits>
#include <cstdint>
#include <memory>
#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cub/cub.cuh>
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
        // The original generator cannot make progress for N <= 30.
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
    return std::sqrt(dx * dx + dy * dy);
}

// Candidate growth remains sequential within a seed, but seeds are independent.
// MPI partitions seeds cyclically; OpenMP drives the GPUs assigned to each rank.
// A cached candidate is still exact if none of its members were removed: deleting
// a point never chosen by its greedy path cannot change any of that path's choices.
namespace {
constexpr int BLOCK = 128;
constexpr int WARP_CANDIDATES = 256;
constexpr int SHARED_CANDIDATES = 2048;
int world_rank = 0;

void cudaCheck(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "Rank %d: %s: %s\n", world_rank, operation,
                     cudaGetErrorString(status));
        std::abort();
    }
}
#define CUDA_CHECK(operation) cudaCheck((operation), #operation)

void mpiCheck(int status, const char* operation) {
    if (status != MPI_SUCCESS) {
        char message[MPI_MAX_ERROR_STRING];
        int length = 0;
        MPI_Error_string(status, message, &length);
        std::fprintf(stderr, "Rank %d: %s: %.*s\n", world_rank, operation, length, message);
        MPI_Abort(MPI_COMM_WORLD, status);
    }
}
#define MPI_CHECK(operation) mpiCheck((operation), #operation)

template<class T> void deviceAllocate(T*& pointer, size_t count) {
    if (count > std::numeric_limits<size_t>::max() / sizeof(T)) std::abort();
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&pointer), std::max(size_t(1), count) * sizeof(T)));
}

struct Best {
    int cardinality;
    int seed;
};
struct LargerCluster {
    __host__ __device__ Best operator()(Best a, Best b) const {
        return b.cardinality > a.cardinality ||
               (b.cardinality == a.cardinality && b.seed < a.seed) ? b : a;
    }
};
struct Closest {
    double diameter;
    int point;
};
struct SmallerDistance {
    __device__ Closest operator()(Closest a, Closest b) const {
        return b.diameter < a.diameter ||
               (b.diameter == a.diameter && b.point < a.point) ? b : a;
    }
};
__device__ Closest warpMinimum(Closest value) {
    #pragma unroll
    for (int delta = 16; delta; delta >>= 1) {
        Closest other{__shfl_down_sync(0xffffffffu, value.diameter, delta),
                      __shfl_down_sync(0xffffffffu, value.point, delta)};
        value = SmallerDistance{}(value, other);
    }
    return {__shfl_sync(0xffffffffu, value.diameter, 0),
            __shfl_sync(0xffffffffu, value.point, 0)};
}

__device__ double gpuDistance(Point a, Point b) {
    double dx = a.x - b.x, dy = a.y - b.y;
    // Match the original native host's contraction and rounding, including
    // CPUs without FMA. Never use approximate sqrt or fast-math comparisons.
    #ifdef QT_NATIVE_FMA
    return __dsqrt_rn(__fma_rn(dx, dx, __dmul_rn(dy, dy)));
    #else
    return __dsqrt_rn(__dadd_rn(__dmul_rn(dx, dx), __dmul_rn(dy, dy)));
    #endif
}

// Uniform cells have width >= threshold. Only the surrounding nine cells
// can contain a neighbor. Grid storage is linear even for tiny thresholds.
// Neighbor order may vary; every greedy reduction breaks ties by point index.
template<bool Fill>
__global__ void buildNeighbors(const Point* points, const int* seeds, int rows,
                               int grid_dim, double inverse_cell_width,
                               const int* cell_offsets, const int* cell_points,
                               double threshold, size_t* offsets, int* neighbors) {
    using Sum = cub::BlockReduce<size_t, BLOCK>;
    using Scan = cub::BlockScan<int, BLOCK>;
    __shared__ typename Sum::TempStorage sum_storage;
    __shared__ typename Scan::TempStorage scan_storage;
    for (int row = blockIdx.x; row < rows; row += gridDim.x) {
        int seed = seeds[row];
        Point center = points[seed];
        int cx = min(grid_dim - 1, int(center.x * inverse_cell_width));
        int cy = min(grid_dim - 1, int(center.y * inverse_cell_width));
        int count = 0;
        for (int y = max(0, cy - 1); y <= min(grid_dim - 1, cy + 1); ++y) {
            for (int x = max(0, cx - 1); x <= min(grid_dim - 1, cx + 1); ++x) {
                int cell = y * grid_dim + x;
                for (int i = cell_offsets[cell] + threadIdx.x; i < cell_offsets[cell + 1]; i += BLOCK) {
                    int candidate = cell_points[i];
                    count += candidate == seed || gpuDistance(center, points[candidate]) < threshold;
                }
            }
        }
        if constexpr (!Fill) {
            size_t total = Sum(sum_storage).Sum(size_t(count));
            if (threadIdx.x == 0) offsets[row] = total;
        } else {
            int position;
            Scan(scan_storage).ExclusiveSum(count, position);
            size_t output = offsets[row] + position;
            for (int y = max(0, cy - 1); y <= min(grid_dim - 1, cy + 1); ++y) {
                for (int x = max(0, cx - 1); x <= min(grid_dim - 1, cx + 1); ++x) {
                    int cell = y * grid_dim + x;
                    for (int i = cell_offsets[cell] + threadIdx.x; i < cell_offsets[cell + 1]; i += BLOCK) {
                        int candidate = cell_points[i];
                        if (candidate == seed || gpuDistance(center, points[candidate]) < threshold)
                            neighbors[output++] = candidate;
                    }
                }
            }
        }
        __syncthreads();
    }
}

__global__ void removeMembers(unsigned char* active, const int* members, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x)
        active[members[i]] = 0;
}

// One warp per small candidate: all distances stay in registers across growth
// steps, avoiding both global scratch traffic and block-wide barriers.
__global__ void growWarpCandidates(const Point* points, const int* seeds, int rows,
                                   const size_t* offsets, const int* neighbors,
                                   const unsigned char* active, double threshold,
                                   int* cardinalities, int* members) {
    int lane = threadIdx.x & 31;
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int warp_stride = gridDim.x * (blockDim.x / 32);
    for (int row = warp; row < rows; row += warp_stride) {
        size_t offset = offsets[row];
        int degree = int(offsets[row + 1] - offset);
        if (degree > WARP_CANDIDATES) continue;
        int seed = seeds[row];
        if (!active[seed]) {
            if (lane == 0) cardinalities[row] = 0;
            continue;
        }
        int old_count = cardinalities[row];
        bool dirty = old_count == 0;
        for (int i = lane; i < old_count; i += 32)
            dirty |= !active[members[offset + i]];
        if (!__any_sync(0xffffffffu, dirty)) continue;
        int ids[WARP_CANDIDATES / 32];
        double maxima[WARP_CANDIDATES / 32];
        Point center = points[seed];
        #pragma unroll
        for (int k = 0; k < WARP_CANDIDATES / 32; ++k) {
            int index = lane + 32 * k;
            int id = index < degree ? neighbors[offset + index] : seed;
            ids[k] = id;
            maxima[k] = index < degree && active[id] && id != seed ?
                        gpuDistance(center, points[id]) : CUDART_INF;
        }
        if (lane == 0) members[offset] = seed;
        int count = 1;
        for (;;) {
            Closest best{CUDART_INF, INT_MAX};
            #pragma unroll
            for (int k = 0; k < WARP_CANDIDATES / 32; ++k)
                if (maxima[k] < threshold)
                    best = SmallerDistance{}(best, {maxima[k], ids[k]});
            best = warpMinimum(best);
            if (best.point == INT_MAX) break;
            if (lane == 0) members[offset + count] = best.point;
            ++count;
            Point added = points[best.point];
            #pragma unroll
            for (int k = 0; k < WARP_CANDIDATES / 32; ++k) {
                if (ids[k] == best.point) maxima[k] = CUDART_INF;
                if (maxima[k] < threshold) {
                    double d = gpuDistance(added, points[ids[k]]);
                    maxima[k] = d > maxima[k] ? d : maxima[k];
                }
            }
        }
        if (lane == 0) cardinalities[row] = count;
    }
}

// Larger neighborhoods use a block, with shared distances when they fit and
// a reusable global workspace for dense rows. Distances are updated only for
// the newly added member, reducing growth from O(degree * size^2) to O(degree * size).
template<bool UseShared>
__global__ void growBlockCandidates(const Point* points, const int* seeds, int rows,
                                    const size_t* offsets, const int* neighbors,
                                    const unsigned char* active, double threshold,
                                    int shared_capacity, double* scratch,
                                    int* cardinalities, int* members) {
    using Reduce = cub::BlockReduce<Closest, BLOCK>;
    __shared__ typename Reduce::TempStorage reduction;
    __shared__ Closest chosen;
    __shared__ int dirty;
    extern __shared__ double shared_distances[];
    int* shared_ids = reinterpret_cast<int*>(shared_distances + shared_capacity);
    for (int row = blockIdx.x; row < rows; row += gridDim.x) {
        size_t offset = offsets[row];
        int degree = int(offsets[row + 1] - offset);
        if (degree <= WARP_CANDIDATES) continue;
        if constexpr (UseShared) {
            if (degree > shared_capacity) continue;
        } else {
            if (degree <= SHARED_CANDIDATES) continue;
        }
        int seed = seeds[row];
        if (!active[seed]) {
            if (threadIdx.x == 0) cardinalities[row] = 0;
            continue;
        }
        int old_count = cardinalities[row];
        if (threadIdx.x == 0) dirty = old_count == 0;
        __syncthreads();
        for (int i = threadIdx.x; i < old_count; i += BLOCK) {
            if (!active[members[offset + i]]) {
                atomicExch(&dirty, 1);
                break;
            }
        }
        __syncthreads();
        int rebuild = dirty;
        __syncthreads();
        if (!rebuild) continue;
        double* maxima = UseShared ? shared_distances : scratch + offset;
        const int* ids = UseShared ? shared_ids : neighbors + offset;
        Point center = points[seed];
        for (int i = threadIdx.x; i < degree; i += BLOCK) {
            int id = neighbors[offset + i];
            if constexpr (UseShared) shared_ids[i] = id;
            maxima[i] = active[id] && id != seed ? gpuDistance(center, points[id]) : CUDART_INF;
        }
        if (threadIdx.x == 0) members[offset] = seed;
        __syncthreads();
        int count = 1;
        for (;;) {
            Closest best{CUDART_INF, INT_MAX};
            for (int i = threadIdx.x; i < degree; i += BLOCK)
                if (maxima[i] < threshold)
                    best = SmallerDistance{}(best, {maxima[i], ids[i]});
            Closest result = Reduce(reduction).Reduce(best, SmallerDistance{});
            if (threadIdx.x == 0) chosen = result;
            __syncthreads();
            if (chosen.point == INT_MAX) break;
            if (threadIdx.x == 0) members[offset + count] = chosen.point;
            ++count;
            Point added = points[chosen.point];
            for (int i = threadIdx.x; i < degree; i += BLOCK) {
                if (ids[i] == chosen.point) maxima[i] = CUDART_INF;
                if (maxima[i] < threshold) {
                    double d = gpuDistance(added, points[ids[i]]);
                    maxima[i] = d > maxima[i] ? d : maxima[i];
                }
            }
            // All threads must finish reading chosen before the next reduction.
            __syncthreads();
        }
        if (threadIdx.x == 0) cardinalities[row] = count;
        __syncthreads();
    }
}

__global__ void selectBest(const int* seeds, const int* cardinalities, int rows, Best* result) {
    using Reduce = cub::BlockReduce<Best, 256>;
    __shared__ typename Reduce::TempStorage storage;
    Best best{0, INT_MAX};
    for (int i = threadIdx.x; i < rows; i += blockDim.x)
        best = LargerCluster{}(best, {cardinalities[i], seeds[i]});
    Best winner = Reduce(storage).Reduce(best, LargerCluster{});
    if (threadIdx.x == 0) *result = winner;
}

struct GpuWorker {
    int device = 0, rows = 0, max_degree = 0, shared_capacity = 0;
    cudaStream_t stream = nullptr;
    Point* points = nullptr;
    int *seeds = nullptr, *neighbors = nullptr, *cardinalities = nullptr;
    int *members = nullptr, *removed = nullptr;
    unsigned char* active = nullptr;
    size_t* offsets = nullptr;
    double* scratch = nullptr;
    Best* device_best = nullptr;
    Best* host_best = nullptr;
    std::vector<size_t> host_offsets;

    void initialize(int device_id, const std::vector<Point>& input,
                    const std::vector<int>& local_seeds, int grid_dim,
                    const std::vector<int>& cell_offsets,
                    const std::vector<int>& cell_points, double threshold) {
        device = device_id;
        rows = int(local_seeds.size());
        CUDA_CHECK(cudaSetDevice(device));
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        deviceAllocate(points, input.size());
        deviceAllocate(seeds, rows);
        deviceAllocate(offsets, size_t(rows) + 1);
        deviceAllocate(cardinalities, rows);
        deviceAllocate(active, input.size());
        deviceAllocate(removed, input.size());
        deviceAllocate(device_best, 1);
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&host_best), sizeof(Best)));
        CUDA_CHECK(cudaMemcpyAsync(points, input.data(), input.size() * sizeof(Point), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(seeds, local_seeds.data(), rows * sizeof(int), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemsetAsync(cardinalities, 0, rows * sizeof(int), stream));
        CUDA_CHECK(cudaMemsetAsync(active, 1, input.size(), stream));
        CUDA_CHECK(cudaMemsetAsync(offsets, 0, (size_t(rows) + 1) * sizeof(size_t), stream));
        int *grid_offsets = nullptr, *grid_points = nullptr;
        deviceAllocate(grid_offsets, cell_offsets.size());
        deviceAllocate(grid_points, cell_points.size());
        CUDA_CHECK(cudaMemcpyAsync(grid_offsets, cell_offsets.data(), cell_offsets.size() * sizeof(int), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(grid_points, cell_points.data(), cell_points.size() * sizeof(int), cudaMemcpyHostToDevice, stream));
        int blocks = std::min(rows, 65535);
        if (rows) {
            buildNeighbors<false><<<blocks, BLOCK, 0, stream>>>(points, seeds, rows, grid_dim,
                grid_dim / MAX_WIDTH, grid_offsets, grid_points, threshold, offsets, nullptr);
            CUDA_CHECK(cudaGetLastError());
        }
        void* scan_workspace = nullptr;
        size_t scan_bytes = 0;
        CUDA_CHECK(cub::DeviceScan::ExclusiveSum(scan_workspace, scan_bytes, offsets, offsets, rows + 1, stream));
        CUDA_CHECK(cudaMalloc(&scan_workspace, scan_bytes));
        CUDA_CHECK(cub::DeviceScan::ExclusiveSum(scan_workspace, scan_bytes, offsets, offsets, rows + 1, stream));
        host_offsets.resize(size_t(rows) + 1);
        CUDA_CHECK(cudaMemcpyAsync(host_offsets.data(), offsets, host_offsets.size() * sizeof(size_t), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaFree(scan_workspace));
        for (int i = 0; i < rows; ++i) {
            int degree = int(host_offsets[i + 1] - host_offsets[i]);
            max_degree = std::max(max_degree, degree);
            if (degree <= SHARED_CANDIDATES && degree > WARP_CANDIDATES)
                shared_capacity = std::max(shared_capacity, degree);
        }
        size_t edges = host_offsets.back();
        deviceAllocate(neighbors, edges);
        deviceAllocate(members, edges);
        if (max_degree > SHARED_CANDIDATES) deviceAllocate(scratch, edges);
        if (rows) {
            buildNeighbors<true><<<blocks, BLOCK, 0, stream>>>(points, seeds, rows, grid_dim,
                grid_dim / MAX_WIDTH, grid_offsets, grid_points, threshold, offsets, neighbors);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaFree(grid_offsets));
        CUDA_CHECK(cudaFree(grid_points));
    }

    Best evaluate(const int* removed_members, int removed_count, double threshold) {
        CUDA_CHECK(cudaSetDevice(device));
        if (removed_count) {
            CUDA_CHECK(cudaMemcpyAsync(removed, removed_members, removed_count * sizeof(int), cudaMemcpyHostToDevice, stream));
            removeMembers<<<std::min(65535, (removed_count + 255) / 256), 256, 0, stream>>>(active, removed, removed_count);
        }
        if (rows) {
            growWarpCandidates<<<std::min(65535, (rows + 3) / 4), BLOCK, 0, stream>>>(points, seeds,
                rows, offsets, neighbors, active, threshold, cardinalities, members);
            if (shared_capacity > WARP_CANDIDATES) {
                size_t shared_bytes = size_t(shared_capacity) * (sizeof(double) + sizeof(int));
                growBlockCandidates<true><<<std::min(rows, 65535), BLOCK, shared_bytes, stream>>>(points, seeds,
                    rows, offsets, neighbors, active, threshold, shared_capacity, scratch, cardinalities, members);
            }
            if (max_degree > SHARED_CANDIDATES) {
                // Dense rows reserve no dynamic shared memory, allowing many
                // more blocks to reside on an SM while accessing global scratch.
                growBlockCandidates<false><<<std::min(rows, 65535), BLOCK, 0, stream>>>(points, seeds,
                    rows, offsets, neighbors, active, threshold, 0, scratch, cardinalities, members);
            }
        }
        selectBest<<<1, 256, 0, stream>>>(seeds, cardinalities, rows, device_best);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(host_best, device_best, sizeof(Best), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        return *host_best;
    }

    void copyMembers(int row, int count, int* output) {
        CUDA_CHECK(cudaSetDevice(device));
        CUDA_CHECK(cudaMemcpy(output, members + host_offsets[row], count * sizeof(int), cudaMemcpyDeviceToHost));
    }

    ~GpuWorker() {
        CUDA_CHECK(cudaSetDevice(device));
        CUDA_CHECK(cudaFree(points)); CUDA_CHECK(cudaFree(seeds));
        CUDA_CHECK(cudaFree(offsets)); CUDA_CHECK(cudaFree(neighbors));
        CUDA_CHECK(cudaFree(cardinalities)); CUDA_CHECK(cudaFree(members));
        CUDA_CHECK(cudaFree(removed)); CUDA_CHECK(cudaFree(active));
        CUDA_CHECK(cudaFree(scratch)); CUDA_CHECK(cudaFree(device_best));
        CUDA_CHECK(cudaFreeHost(host_best));
        CUDA_CHECK(cudaStreamDestroy(stream));
    }
};
} // namespace

std::vector<Cluster> qtClustering(const std::vector<Point>& points, double threshold) {
    int rank_count, local_rank, local_size, device_count;
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &rank_count));
    MPI_Comm local_comm;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, world_rank, MPI_INFO_NULL, &local_comm));
    MPI_CHECK(MPI_Comm_rank(local_comm, &local_rank));
    MPI_CHECK(MPI_Comm_size(local_comm, &local_size));
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (!device_count) {
        std::fprintf(stderr, "Rank %d: QT clustering requires a CUDA device.\n", world_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    std::vector<int> devices;
    for (int device = local_rank; device < device_count; device += local_size) devices.push_back(device);
    // CUDA_VISIBLE_DEVICES may expose one private GPU per rank.
    if (devices.empty()) devices.push_back(local_rank % device_count);
    MPI_CHECK(MPI_Comm_free(&local_comm));
    int worker_count = int(devices.size());
    // Balance by GPU count, including ranks with different numbers of GPUs.
    // Each GPU owns seeds global_worker + k * total_workers.
    std::vector<int> rank_workers(rank_count), worker_offsets(size_t(rank_count) + 1, 0);
    MPI_CHECK(MPI_Allgather(&worker_count, 1, MPI_INT, rank_workers.data(), 1, MPI_INT, MPI_COMM_WORLD));
    for (int rank = 0; rank < rank_count; ++rank)
        worker_offsets[rank + 1] = worker_offsets[rank] + rank_workers[rank];
    int total_workers = worker_offsets.back();
    int n = int(points.size());
    int host_threads = std::min(omp_get_max_threads(), std::max(1, std::min(32, n / 4096)));
    int grid_limit = std::max(1, int(std::sqrt(double(n))));
    double requested_dim = MAX_WIDTH / threshold;
    int grid_dim = std::isnan(requested_dim) ? 1 :
                   int(std::max(1.0, std::min(double(grid_limit), std::floor(requested_dim))));
    int cells = grid_dim * grid_dim;
    std::vector<int> cell_counts(cells, 0), cell_offsets(size_t(cells) + 1, 0);
    std::vector<int> cell_points(n), point_cells(n);
    double xmin = points[0].x, xmax = xmin, ymin = points[0].y, ymax = ymin;
    #pragma omp parallel for num_threads(host_threads) schedule(static) if(host_threads > 1) reduction(min:xmin,ymin) reduction(max:xmax,ymax)
    for (int i = 0; i < n; ++i) {
        xmin = std::min(xmin, points[i].x); xmax = std::max(xmax, points[i].x);
        ymin = std::min(ymin, points[i].y); ymax = std::max(ymax, points[i].y);
        int x = std::min(grid_dim - 1, int(points[i].x * (grid_dim / MAX_WIDTH)));
        int y = std::min(grid_dim - 1, int(points[i].y * (grid_dim / MAX_WIDTH)));
        int cell = y * grid_dim + x;
        point_cells[i] = cell;
        #pragma omp atomic update
        cell_counts[cell]++;
    }
    // If even the bounding-box diagonal is strictly below the threshold,
    // every seed grows to N. Only seed zero can win the original tie break.
    // Evaluate its full greedy path, preserving member order without O(N^3)
    // work or O(N^2) neighborhood storage for all the redundant seeds.
    bool single_cluster = distance(Point{xmin, ymin}, Point{xmax, ymax}) < threshold;
    for (int cell = 0; cell < cells; ++cell) cell_offsets[cell + 1] = cell_offsets[cell] + cell_counts[cell];
    cell_counts.assign(cell_offsets.begin(), cell_offsets.end() - 1);
    #pragma omp parallel for num_threads(host_threads) schedule(static) if(host_threads > 1)
    for (int i = 0; i < n; ++i) {
        int position;
        #pragma omp atomic capture
        position = cell_counts[point_cells[i]]++;
        cell_points[position] = i;
    }
    std::vector<std::unique_ptr<GpuWorker>> workers(worker_count);
    #pragma omp parallel for num_threads(worker_count) schedule(static)
    for (int w = 0; w < worker_count; ++w) {
        std::vector<int> seeds;
        for (int64_t seed = worker_offsets[world_rank] + w; seed < n;
             seed += total_workers) {
            if (!single_cluster || seed == 0) seeds.push_back(int(seed));
        }
        workers[w] = std::make_unique<GpuWorker>();
        workers[w]->initialize(devices[w], points, seeds, grid_dim, cell_offsets, cell_points, threshold);
    }
    int* winning_members = nullptr;
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&winning_members), size_t(n) * sizeof(int)));
    std::vector<Best> worker_best(worker_count);
    std::vector<unsigned char> active(n, 1);
    std::vector<Cluster> clusters;
    int remaining = n, previous_count = 0;
    while (remaining) {
        #pragma omp parallel for num_threads(worker_count) schedule(static)
        for (int w = 0; w < worker_count; ++w)
            worker_best[w] = workers[w]->evaluate(winning_members, previous_count, threshold);
        Best local_best{0, INT_MAX}, winner;
        for (const Best& value : worker_best) local_best = LargerCluster{}(local_best, value);
        static_assert(sizeof(Best) == 2 * sizeof(int), "MPI_2INT requires two contiguous ints");
        MPI_CHECK(MPI_Allreduce(&local_best, &winner, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD));
        // Once the largest candidate is a singleton, all remaining clusters
        // are singletons selected in ascending seed order, exactly as before.
        if (winner.cardinality == 1) {
            if (world_rank == 0)
                for (int seed = 0; seed < n; ++seed)
                    if (active[seed]) clusters.push_back(Cluster{{seed}, seed});
            break;
        }
        if (winner.cardinality <= 0 || winner.cardinality > remaining) MPI_Abort(MPI_COMM_WORLD, 1);
        int global_worker = winner.seed % total_workers;
        int owner = int(std::upper_bound(worker_offsets.begin(), worker_offsets.end(), global_worker) -
                        worker_offsets.begin()) - 1;
        if (world_rank == owner) {
            workers[global_worker - worker_offsets[owner]]->copyMembers(winner.seed / total_workers,
                                                                        winner.cardinality, winning_members);
        }
        MPI_CHECK(MPI_Bcast(winning_members, winner.cardinality, MPI_INT, owner, MPI_COMM_WORLD));
        if (world_rank == 0) {
            Cluster cluster;
            cluster.seed_point = winner.seed;
            cluster.members.assign(winning_members, winning_members + winner.cardinality);
            clusters.push_back(std::move(cluster));
        }
        #pragma omp parallel for num_threads(host_threads) schedule(static) if(winner.cardinality >= 4096)
        for (int i = 0; i < winner.cardinality; ++i) active[winning_members[i]] = 0;
        remaining -= winner.cardinality;
        previous_count = winner.cardinality;
    }
    CUDA_CHECK(cudaFreeHost(winning_members));
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
    int provided = MPI_THREAD_SINGLE;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) return 1;
    struct MpiLifetime {
        ~MpiLifetime() { MPI_Finalize(); }
    } mpi_lifetime;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));
    MPI_CHECK(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));
    if (provided < MPI_THREAD_FUNNELED) {
        if (world_rank == 0) std::fprintf(stderr, "MPI_THREAD_FUNNELED is required.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
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
    
    if (num_points <= 0 || threshold <= 0.0) {
        if (world_rank == 0) printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n", 
               num_points, threshold);
        return 1;
    }
    
    if (world_rank == 0) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Generate synthetic data
    std::vector<Point> points(num_points);
    if (world_rank == 0) generateSyntheticData(points, num_points);
    static_assert(sizeof(Point) == 2 * sizeof(double), "Point must contain two packed doubles");
    // Chunk the broadcast so MPI counts remain representable for large inputs.
    size_t input_bytes = points.size() * sizeof(Point);
    auto* point_bytes = reinterpret_cast<unsigned char*>(points.data());
    for (size_t offset = 0; offset < input_bytes;) {
        int count = int(std::min(size_t(INT_MAX), input_bytes - offset));
        MPI_CHECK(MPI_Bcast(point_bytes + offset, count, MPI_BYTE, 0, MPI_COMM_WORLD));
        offset += count;
    }
    
    // Perform QT clustering
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    const double cluster_start = MPI_Wtime();
    
    const std::vector<Cluster> clusters = qtClustering(points, threshold);
    
    const double local_time = MPI_Wtime() - cluster_start;
    double time_sec = 0.0;
    MPI_CHECK(MPI_Reduce(&local_time, &time_sec, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    if (world_rank != 0) return 0;
    const long long cluster_time = static_cast<long long>(time_sec * 1000.0);
    
    printf("Clustering time: %lld ms\n", cluster_time);
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
