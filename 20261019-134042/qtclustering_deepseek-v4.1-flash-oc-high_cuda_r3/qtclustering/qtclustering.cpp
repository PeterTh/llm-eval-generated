// QT Clustering Benchmark - CUDA Parallel Version
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
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

static void checkCuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error (%s): %s\n", what, cudaGetErrorString(err));
        exit(1);
    }
}

template <typename T>
struct DeviceBuffer {
    T* ptr = nullptr;

    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t count) { allocate(count); }
    ~DeviceBuffer() { if (ptr) cudaFree(ptr); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    void allocate(size_t count) {
        if (ptr) {
            checkCuda(cudaFree(ptr), "cudaFree");
            ptr = nullptr;
        }
        if (count > 0) {
            checkCuda(cudaMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(T)), "cudaMalloc");
        }
    }

    bool try_allocate(size_t count) {
        if (ptr) {
            checkCuda(cudaFree(ptr), "cudaFree");
            ptr = nullptr;
        }
        if (count == 0) return true;
        const cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(T));
        if (err == cudaSuccess) return true;
        ptr = nullptr;
        if (err == cudaErrorMemoryAllocation) {
            cudaGetLastError();
            return false;
        }
        checkCuda(err, "cudaMalloc");
        return false;
    }
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

template <bool USE_NEIGHBORS>
__device__ __forceinline__ int find_closest_point_device(
    const int* members,
    const int mcount,
    const unsigned char* __restrict__ clustered,
    const double* __restrict__ px,
    const double* __restrict__ py,
    const int* __restrict__ neighbors,
    const int neighbor_count,
    const int point_count,
    const double threshold,
    const double reject_sq) {
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int num_warps = (blockDim.x + 31) >> 5;

    __shared__ double warp_dist[32];
    __shared__ int warp_index[32];

    double best_dist = DBL_MAX;
    int best_index = -1;
    const int candidate_count = USE_NEIGHBORS ? neighbor_count : point_count;

    for (int index = tid; index < candidate_count; index += blockDim.x) {
        const int candidate = USE_NEIGHBORS ? neighbors[index] : index;
        if (clustered[candidate]) continue;

        const double cx = px[candidate];
        const double cy = py[candidate];
        double max_sq = 0.0;
        bool is_member = false;
        bool rejected = false;

        for (int i = 0; i < mcount; ++i) {
            const int member = members[i];
            if (member == candidate) {
                is_member = true;
                break;
            }
            const double dx = cx - px[member];
            const double dy = cy - py[member];
            const double sq = dx * dx + dy * dy;
            if (sq > max_sq) max_sq = sq;
            if (max_sq >= reject_sq) {
                rejected = true;
                break;
            }
        }

        if (is_member || rejected) continue;

        const double max_dist = sqrt(max_sq);
        if (max_dist < threshold && max_dist < best_dist) {
            best_dist = max_dist;
            best_index = candidate;
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        const double other_dist = __shfl_down_sync(0xffffffffu, best_dist, offset);
        const int other_index = __shfl_down_sync(0xffffffffu, best_index, offset);
        if (other_dist < best_dist ||
            (other_dist == best_dist && other_index >= 0 &&
             (best_index < 0 || other_index < best_index))) {
            best_dist = other_dist;
            best_index = other_index;
        }
    }

    if (lane == 0) {
        warp_dist[warp] = best_dist;
        warp_index[warp] = best_index;
    }
    __syncthreads();

    if (warp == 0) {
        double block_dist = (lane < num_warps) ? warp_dist[lane] : DBL_MAX;
        int block_index = (lane < num_warps) ? warp_index[lane] : -1;

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            const double other_dist = __shfl_down_sync(0xffffffffu, block_dist, offset);
            const int other_index = __shfl_down_sync(0xffffffffu, block_index, offset);
            if (other_dist < block_dist ||
                (other_dist == block_dist && other_index >= 0 &&
                 (block_index < 0 || other_index < block_index))) {
                block_dist = other_dist;
                block_index = other_index;
            }
        }

        if (lane == 0) warp_index[0] = block_index;
    }
    __syncthreads();

    const int result = warp_index[0];
    __syncthreads();
    return result;
}

__device__ __forceinline__ int find_min_candidate_device(
    const double* candidate_max,
    const int count) {
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int num_warps = (blockDim.x + 31) >> 5;

    __shared__ double warp_dist[32];
    __shared__ int warp_index[32];

    double best_dist = DBL_MAX;
    int best_index = -1;

    for (int i = tid; i < count; i += blockDim.x) {
        const double dist = candidate_max[i];
        if (dist < best_dist) {
            best_dist = dist;
            best_index = i;
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        const double other_dist = __shfl_down_sync(0xffffffffu, best_dist, offset);
        const int other_index = __shfl_down_sync(0xffffffffu, best_index, offset);
        if (other_dist < best_dist ||
            (other_dist == best_dist && other_index >= 0 &&
             (best_index < 0 || other_index < best_index))) {
            best_dist = other_dist;
            best_index = other_index;
        }
    }

    if (lane == 0) {
        warp_dist[warp] = best_dist;
        warp_index[warp] = best_index;
    }
    __syncthreads();

    if (warp == 0) {
        double block_dist = (lane < num_warps) ? warp_dist[lane] : DBL_MAX;
        int block_index = (lane < num_warps) ? warp_index[lane] : -1;

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            const double other_dist = __shfl_down_sync(0xffffffffu, block_dist, offset);
            const int other_index = __shfl_down_sync(0xffffffffu, block_index, offset);
            if (other_dist < block_dist ||
                (other_dist == block_dist && other_index >= 0 &&
                 (block_index < 0 || other_index < block_index))) {
                block_dist = other_dist;
                block_index = other_index;
            }
        }

        if (lane == 0) warp_index[0] = block_index;
    }
    __syncthreads();

    const int result = warp_index[0];
    __syncthreads();
    return result;
}

__device__ __forceinline__ int generate_incremental_cluster(
    const int seed,
    const int* __restrict__ seed_neighbors,
    const int neighbor_count,
    const double* __restrict__ px,
    const double* __restrict__ py,
    const unsigned char* __restrict__ clustered,
    const int point_count,
    const double threshold,
    double* __restrict__ candidate_max,
    int* __restrict__ out_members) {
    const int tid = threadIdx.x;

    for (int i = tid; i < neighbor_count; i += blockDim.x) {
        const int candidate = seed_neighbors[i];
        double value = DBL_MAX;
        if (!clustered[candidate]) {
            const double dx = px[candidate] - px[seed];
            const double dy = py[candidate] - py[seed];
            const double dist = sqrt(dx * dx + dy * dy);
            if (dist < threshold) value = dist;
        }
        candidate_max[i] = value;
    }
    __syncthreads();

    if (out_members && tid == 0) out_members[0] = seed;

    int mcount = 1;
    while (mcount < point_count) {
        const int best_index = find_min_candidate_device(candidate_max, neighbor_count);
        if (best_index < 0) break;

        const int closest = seed_neighbors[best_index];
        if (out_members && tid == 0) out_members[mcount] = closest;
        if (tid == 0) candidate_max[best_index] = DBL_MAX;
        __syncthreads();
        ++mcount;

        for (int i = tid; i < neighbor_count; i += blockDim.x) {
            double value = candidate_max[i];
            if (value < DBL_MAX) {
                const int candidate = seed_neighbors[i];
                const double dx = px[candidate] - px[closest];
                const double dy = py[candidate] - py[closest];
                const double dist = sqrt(dx * dx + dy * dy);
                if (dist > value) value = dist;
                candidate_max[i] = (value < threshold) ? value : DBL_MAX;
            }
        }
        __syncthreads();
    }

    return mcount;
}

__global__ void candidate_incremental_kernel(
    const double* __restrict__ px,
    const double* __restrict__ py,
    const int* __restrict__ neighbor_offsets,
    const int* __restrict__ neighbor_list,
    const unsigned char* __restrict__ clustered,
    const int point_count,
    const double threshold,
    int* __restrict__ cardinality) {
    extern __shared__ double candidate_max[];
    const int seed = blockIdx.x;
    if (clustered[seed]) {
        cardinality[seed] = -1;
        return;
    }

    const int neighbor_count = neighbor_offsets[seed + 1] - neighbor_offsets[seed];
    const int* seed_neighbors = neighbor_list + neighbor_offsets[seed];
    cardinality[seed] = generate_incremental_cluster(
        seed, seed_neighbors, neighbor_count, px, py, clustered,
        point_count, threshold, candidate_max, nullptr);
}

__global__ void build_incremental_cluster_kernel(
    const int seed,
    const double* __restrict__ px,
    const double* __restrict__ py,
    const int* __restrict__ neighbor_offsets,
    const int* __restrict__ neighbor_list,
    const unsigned char* __restrict__ clustered,
    const int point_count,
    const double threshold,
    int* __restrict__ members) {
    extern __shared__ double candidate_max[];
    const int neighbor_count = neighbor_offsets[seed + 1] - neighbor_offsets[seed];
    const int* seed_neighbors = neighbor_list + neighbor_offsets[seed];
    generate_incremental_cluster(
        seed, seed_neighbors, neighbor_count, px, py, clustered,
        point_count, threshold, candidate_max, members);
}

__global__ void count_neighbors_kernel(
    const double* __restrict__ px,
    const double* __restrict__ py,
    const int point_count,
    const double threshold,
    int* __restrict__ counts) {
    const int member = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int num_warps = (blockDim.x + 31) >> 5;

    __shared__ int warp_counts[32];
    const double mx = px[member];
    const double my = py[member];
    int count = 0;

    for (int candidate = tid; candidate < point_count; candidate += blockDim.x) {
        if (candidate == member) continue;
        const double dx = px[candidate] - mx;
        const double dy = py[candidate] - my;
        const double dist = sqrt(dx * dx + dy * dy);
        if (dist < threshold) ++count;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        count += __shfl_down_sync(0xffffffffu, count, offset);
    }

    if (lane == 0) warp_counts[warp] = count;
    __syncthreads();

    if (warp == 0) {
        int total = (lane < num_warps) ? warp_counts[lane] : 0;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            total += __shfl_down_sync(0xffffffffu, total, offset);
        }
        if (lane == 0) counts[member] = total;
    }
}

__global__ void build_neighbors_kernel(
    const double* __restrict__ px,
    const double* __restrict__ py,
    const int point_count,
    const double threshold,
    const int* __restrict__ offsets,
    int* __restrict__ neighbors) {
    const int member = blockIdx.x;
    const int tid = threadIdx.x;
    const double mx = px[member];
    const double my = py[member];

    __shared__ int prefix[256];
    int running = 0;

    for (int base = 0; base < point_count; base += blockDim.x) {
        const int candidate = base + tid;
        int flag = 0;
        if (candidate < point_count && candidate != member) {
            const double dx = px[candidate] - mx;
            const double dy = py[candidate] - my;
            const double dist = sqrt(dx * dx + dy * dy);
            flag = (dist < threshold) ? 1 : 0;
        }

        prefix[tid] = flag;
        __syncthreads();

        for (int offset = 1; offset < blockDim.x; offset <<= 1) {
            const int value = (tid >= offset) ? prefix[tid - offset] : 0;
            __syncthreads();
            prefix[tid] += value;
            __syncthreads();
        }

        if (flag) {
            neighbors[offsets[member] + running + prefix[tid] - flag] = candidate;
        }
        running += prefix[blockDim.x - 1];
        __syncthreads();
    }
}

template <bool USE_GLOBAL_MEMBERS, bool USE_NEIGHBORS>
__global__ void candidate_cardinality_kernel(
    const double* __restrict__ px,
    const double* __restrict__ py,
    const int* __restrict__ neighbor_offsets,
    const int* __restrict__ neighbor_list,
    const unsigned char* __restrict__ clustered,
    const int point_count,
    const double threshold,
    const double reject_sq,
    int* __restrict__ cardinality,
    int* __restrict__ global_members) {
    extern __shared__ int shared_members[];
    int* members = USE_GLOBAL_MEMBERS
                       ? global_members + static_cast<size_t>(blockIdx.x) * point_count
                       : shared_members;

    for (int seed = blockIdx.x; seed < point_count; seed += gridDim.x) {
        if (clustered[seed]) {
            cardinality[seed] = -1;
            continue;
        }

        const int neighbor_count = USE_NEIGHBORS
                                       ? neighbor_offsets[seed + 1] - neighbor_offsets[seed]
                                       : 0;
        const int* seed_neighbors = USE_NEIGHBORS
                                        ? neighbor_list + neighbor_offsets[seed]
                                        : nullptr;

        if (threadIdx.x == 0) members[0] = seed;
        __syncthreads();

        int mcount = 1;
        while (mcount < point_count) {
            const int closest = find_closest_point_device<USE_NEIGHBORS>(
                members, mcount, clustered, px, py, seed_neighbors, neighbor_count,
                point_count, threshold, reject_sq);
            if (closest < 0) break;

            if (threadIdx.x == 0) members[mcount] = closest;
            __syncthreads();
            ++mcount;
        }

        cardinality[seed] = mcount;
        __syncthreads();
    }
}

__global__ void select_best_seed_kernel(
    const int* __restrict__ cardinality,
    const int point_count,
    int* __restrict__ best) {
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int num_warps = (blockDim.x + 31) >> 5;

    __shared__ int shared_card[32];
    __shared__ int shared_index[32];

    int best_card = -1;
    int best_index = -1;

    for (int i = tid; i < point_count; i += blockDim.x) {
        const int card = cardinality[i];
        if (card > best_card ||
            (card == best_card && card >= 0 && (best_index < 0 || i < best_index))) {
            best_card = card;
            best_index = i;
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        const int other_card = __shfl_down_sync(0xffffffffu, best_card, offset);
        const int other_index = __shfl_down_sync(0xffffffffu, best_index, offset);
        if (other_card > best_card ||
            (other_card == best_card && other_card >= 0 && other_index >= 0 &&
             (best_index < 0 || other_index < best_index))) {
            best_card = other_card;
            best_index = other_index;
        }
    }

    if (lane == 0) {
        shared_card[warp] = best_card;
        shared_index[warp] = best_index;
    }
    __syncthreads();

    if (warp == 0) {
        int block_card = (lane < num_warps) ? shared_card[lane] : -1;
        int block_index = (lane < num_warps) ? shared_index[lane] : -1;

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            const int other_card = __shfl_down_sync(0xffffffffu, block_card, offset);
            const int other_index = __shfl_down_sync(0xffffffffu, block_index, offset);
            if (other_card > block_card ||
                (other_card == block_card && other_card >= 0 && other_index >= 0 &&
                 (block_index < 0 || other_index < block_index))) {
                block_card = other_card;
                block_index = other_index;
            }
        }

        if (lane == 0) {
            best[0] = block_index;
            best[1] = block_card;
        }
    }
}

template <bool USE_NEIGHBORS>
__global__ void build_best_cluster_kernel(
    const int seed,
    const unsigned char* __restrict__ clustered,
    const double* __restrict__ px,
    const double* __restrict__ py,
    const int* __restrict__ neighbor_offsets,
    const int* __restrict__ neighbor_list,
    const int point_count,
    const double threshold,
    const double reject_sq,
    int* __restrict__ members) {
    const int neighbor_count = USE_NEIGHBORS
                                   ? neighbor_offsets[seed + 1] - neighbor_offsets[seed]
                                   : 0;
    const int* seed_neighbors = USE_NEIGHBORS
                                    ? neighbor_list + neighbor_offsets[seed]
                                    : nullptr;

    if (threadIdx.x == 0) members[0] = seed;
    __syncthreads();

    int mcount = 1;
    while (mcount < point_count) {
        const int closest = find_closest_point_device<USE_NEIGHBORS>(
            members, mcount, clustered, px, py, seed_neighbors, neighbor_count,
            point_count, threshold, reject_sq);
        if (closest < 0) break;

        if (threadIdx.x == 0) members[mcount] = closest;
        __syncthreads();
        ++mcount;
    }
}

__global__ void mark_clustered_kernel(
    const int* __restrict__ members,
    const int count,
    unsigned char* __restrict__ clustered) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) clustered[members[i]] = 1;
}

// Main QT clustering algorithm
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<Cluster> clusters;
    if (N == 0) return clusters;

    const double threshold_sq = threshold * threshold;
    const double reject_sq = std::isfinite(threshold_sq)
                                 ? std::nextafter(threshold_sq,
                                                  std::numeric_limits<double>::infinity())
                                 : threshold_sq;

    std::vector<double> host_x(N);
    std::vector<double> host_y(N);
    for (int i = 0; i < N; ++i) {
        host_x[i] = points[i].x;
        host_y[i] = points[i].y;
    }

    DeviceBuffer<double> device_x(N);
    DeviceBuffer<double> device_y(N);
    DeviceBuffer<unsigned char> device_clustered(N);
    DeviceBuffer<int> device_cardinality(N);
    DeviceBuffer<int> device_best(2);
    DeviceBuffer<int> device_all_members(N);

    checkCuda(cudaMemcpy(device_x.ptr, host_x.data(), N * sizeof(double), cudaMemcpyHostToDevice), "copy x");
    checkCuda(cudaMemcpy(device_y.ptr, host_y.data(), N * sizeof(double), cudaMemcpyHostToDevice), "copy y");
    checkCuda(cudaMemset(device_clustered.ptr, 0, N * sizeof(unsigned char)), "clear clustered");

    int device_id = 0;
    checkCuda(cudaGetDevice(&device_id), "cudaGetDevice");
    int sm_count = 0;
    checkCuda(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_id), "sm count");
    int max_shared = 0;
    checkCuda(cudaDeviceGetAttribute(&max_shared, cudaDevAttrMaxSharedMemoryPerBlockOptin, device_id), "max shared");

    int threads = 256;
    if (N < 256) {
        threads = ((N + 31) / 32) * 32;
        if (threads < 32) threads = 32;
    }

    const size_t shared_bytes = static_cast<size_t>(N) * sizeof(int);
    const size_t shared_limit = (max_shared > 1024) ? static_cast<size_t>(max_shared - 1024) : 0;
    const bool use_shared = shared_bytes <= shared_limit;

    const int neighbor_threads = 256;
    DeviceBuffer<int> device_neighbor_offsets;
    DeviceBuffer<int> device_neighbor_list;
    bool use_neighbors = false;
    int max_neighbor_count = 0;

    if (N >= 128) {
        std::vector<int> host_counts(N);
        {
            DeviceBuffer<int> device_counts(N);
            count_neighbors_kernel<<<N, neighbor_threads>>>(
                device_x.ptr, device_y.ptr, N, threshold, device_counts.ptr);
            checkCuda(cudaGetLastError(), "count neighbors kernel");
            checkCuda(cudaMemcpy(host_counts.data(), device_counts.ptr,
                                 N * sizeof(int), cudaMemcpyDeviceToHost),
                      "copy neighbor counts");
        }

        std::vector<int> host_offsets(N + 1);
        long long total = 0;
        bool fits = true;
        for (int i = 0; i < N; ++i) {
            if (host_counts[i] > max_neighbor_count) max_neighbor_count = host_counts[i];
            host_offsets[i] = static_cast<int>(total);
            total += host_counts[i];
            if (total > std::numeric_limits<int>::max()) {
                fits = false;
                break;
            }
        }
        host_offsets[N] = static_cast<int>(total);

        if (fits) {
            size_t free_memory = 0;
            size_t total_memory = 0;
            checkCuda(cudaMemGetInfo(&free_memory, &total_memory), "memory info");
            const size_t list_bytes = static_cast<size_t>(total) * sizeof(int);
            size_t memory_cap = free_memory / 2;
            const size_t hard_cap = static_cast<size_t>(8) << 30;
            if (memory_cap > hard_cap) memory_cap = hard_cap;

            if (list_bytes <= memory_cap) {
                device_neighbor_offsets.allocate(N + 1);
                device_neighbor_list.try_allocate(
                    static_cast<size_t>(total > 0 ? total : 1));

                if (total == 0 || device_neighbor_list.ptr != nullptr) {
                    checkCuda(cudaMemcpy(device_neighbor_offsets.ptr, host_offsets.data(),
                                         (N + 1) * sizeof(int), cudaMemcpyHostToDevice),
                              "copy neighbor offsets");
                    if (total > 0) {
                        build_neighbors_kernel<<<N, neighbor_threads>>>(
                            device_x.ptr, device_y.ptr, N, threshold,
                            device_neighbor_offsets.ptr, device_neighbor_list.ptr);
                        checkCuda(cudaGetLastError(), "build neighbors kernel");
                    }
                    use_neighbors = true;
                }
            }
        }
    }

    const size_t incremental_shared_bytes =
        static_cast<size_t>(max_neighbor_count) * sizeof(double);
    const bool use_incremental =
        use_neighbors && incremental_shared_bytes <= shared_limit;

    if (use_incremental && incremental_shared_bytes > 48 * 1024) {
        checkCuda(cudaFuncSetAttribute(
                      candidate_incremental_kernel,
                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                      static_cast<int>(incremental_shared_bytes)),
                  "incremental shared attribute");
        checkCuda(cudaFuncSetAttribute(
                      build_incremental_cluster_kernel,
                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                      static_cast<int>(incremental_shared_bytes)),
                  "incremental build shared attribute");
    }

    int global_grid = 0;
    DeviceBuffer<int> device_member_workspace;
    if (!use_shared && !use_incremental) {
        int blocks_per_sm = 1;
        if (use_neighbors) {
            checkCuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                          &blocks_per_sm, candidate_cardinality_kernel<true, true>, threads, 0),
                      "occupancy");
        } else {
            checkCuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                          &blocks_per_sm, candidate_cardinality_kernel<true, false>, threads, 0),
                      "occupancy");
        }
        global_grid = sm_count * blocks_per_sm;
        if (global_grid > 512) global_grid = 512;
        if (global_grid > N) global_grid = N;
        if (global_grid < 1) global_grid = 1;
        device_member_workspace.allocate(static_cast<size_t>(global_grid) * N);
    } else if (!use_incremental && shared_bytes > 48 * 1024) {
        if (use_neighbors) {
            checkCuda(cudaFuncSetAttribute(
                          candidate_cardinality_kernel<false, true>,
                          cudaFuncAttributeMaxDynamicSharedMemorySize,
                          static_cast<int>(shared_bytes)),
                      "shared attribute");
        } else {
            checkCuda(cudaFuncSetAttribute(
                          candidate_cardinality_kernel<false, false>,
                          cudaFuncAttributeMaxDynamicSharedMemorySize,
                          static_cast<int>(shared_bytes)),
                      "shared attribute");
        }
    }

    std::vector<int> cluster_seeds;
    std::vector<int> cluster_sizes;
    int total_members = 0;

    while (true) {
        if (use_incremental) {
            candidate_incremental_kernel<<<N, threads, incremental_shared_bytes>>>(
                device_x.ptr, device_y.ptr,
                device_neighbor_offsets.ptr, device_neighbor_list.ptr,
                device_clustered.ptr, N, threshold, device_cardinality.ptr);
        } else if (use_shared) {
            if (use_neighbors) {
                candidate_cardinality_kernel<false, true><<<N, threads, shared_bytes>>>(
                    device_x.ptr, device_y.ptr,
                    device_neighbor_offsets.ptr, device_neighbor_list.ptr,
                    device_clustered.ptr, N, threshold,
                    reject_sq, device_cardinality.ptr, nullptr);
            } else {
                candidate_cardinality_kernel<false, false><<<N, threads, shared_bytes>>>(
                    device_x.ptr, device_y.ptr, nullptr, nullptr,
                    device_clustered.ptr, N, threshold,
                    reject_sq, device_cardinality.ptr, nullptr);
            }
        } else {
            if (use_neighbors) {
                candidate_cardinality_kernel<true, true><<<global_grid, threads>>>(
                    device_x.ptr, device_y.ptr,
                    device_neighbor_offsets.ptr, device_neighbor_list.ptr,
                    device_clustered.ptr, N, threshold,
                    reject_sq, device_cardinality.ptr, device_member_workspace.ptr);
            } else {
                candidate_cardinality_kernel<true, false><<<global_grid, threads>>>(
                    device_x.ptr, device_y.ptr, nullptr, nullptr,
                    device_clustered.ptr, N, threshold,
                    reject_sq, device_cardinality.ptr, device_member_workspace.ptr);
            }
        }
        checkCuda(cudaGetLastError(), "candidate kernel");

        select_best_seed_kernel<<<1, 1024>>>(
            device_cardinality.ptr, N, device_best.ptr);
        checkCuda(cudaGetLastError(), "select kernel");

        int host_best[2];
        checkCuda(cudaMemcpy(host_best, device_best.ptr, 2 * sizeof(int), cudaMemcpyDeviceToHost), "copy best");

        const int best_seed = host_best[0];
        const int best_card = host_best[1];
        if (best_seed < 0 || best_card <= 0) break;

        int* cluster_members = device_all_members.ptr + total_members;
        if (use_incremental) {
            build_incremental_cluster_kernel<<<1, threads, incremental_shared_bytes>>>(
                best_seed, device_x.ptr, device_y.ptr,
                device_neighbor_offsets.ptr, device_neighbor_list.ptr,
                device_clustered.ptr, N, threshold, cluster_members);
        } else if (use_neighbors) {
            build_best_cluster_kernel<true><<<1, threads>>>(
                best_seed, device_clustered.ptr, device_x.ptr, device_y.ptr,
                device_neighbor_offsets.ptr, device_neighbor_list.ptr,
                N, threshold, reject_sq, cluster_members);
        } else {
            build_best_cluster_kernel<false><<<1, threads>>>(
                best_seed, device_clustered.ptr, device_x.ptr, device_y.ptr,
                nullptr, nullptr, N, threshold, reject_sq, cluster_members);
        }
        checkCuda(cudaGetLastError(), "build kernel");

        const int mark_blocks = (best_card + threads - 1) / threads;
        mark_clustered_kernel<<<mark_blocks, threads>>>(
            cluster_members, best_card, device_clustered.ptr);
        checkCuda(cudaGetLastError(), "mark kernel");

        cluster_seeds.push_back(best_seed);
        cluster_sizes.push_back(best_card);
        total_members += best_card;
    }

    std::vector<int> all_members(total_members);
    if (total_members > 0) {
        checkCuda(cudaMemcpy(all_members.data(), device_all_members.ptr,
                             total_members * sizeof(int), cudaMemcpyDeviceToHost),
                  "copy members");
    }

    clusters.reserve(cluster_seeds.size());
    int offset = 0;
    for (size_t c = 0; c < cluster_seeds.size(); ++c) {
        Cluster cluster;
        cluster.seed_point = cluster_seeds[c];
        cluster.members.assign(all_members.begin() + offset,
                               all_members.begin() + offset + cluster_sizes[c]);
        offset += cluster_sizes[c];
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

    checkCuda(cudaFree(0), "CUDA init");
    void* warmup_allocation = nullptr;
    checkCuda(cudaMalloc(&warmup_allocation, 1), "CUDA warmup allocation");
    checkCuda(cudaFree(warmup_allocation), "CUDA warmup free");

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
