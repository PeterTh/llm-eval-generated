// QT Clustering Benchmark - CUDA
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
#include <climits>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <utility>
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

constexpr int warp_size = 32;
constexpr int block_size = 128;
constexpr int warps_per_block = block_size / warp_size;
constexpr unsigned full_warp = 0xffffffffu;
using Winner = unsigned long long;

void checkCuda(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) {
        if (count != 0) {
            checkCuda(cudaMalloc(&data_, count * sizeof(T)));
        }
    }

    ~DeviceBuffer() { if (data_) cudaFree(data_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* get() const { return data_; }

private:
    T* data_ = nullptr;
};

struct OutputMember {
    int point;
    int cluster_size;  // Used only at the first member of each cluster.
};

struct RoundState {
    Winner best;
    int written;
};

// Find the largest squared distance whose rounded root passes the original
// strict threshold test. Positive binary64 values are ordered by their bits.
double squaredLimit(double threshold) {
    if (!(threshold > 0)) return -1;
    uint64_t low = 0;
    uint64_t high = 0x7fefffffffffffffull;  // DBL_MAX
    while (low < high) {
        const uint64_t mid = low + (high - low + 1) / 2;
        double squared;
        std::memcpy(&squared, &mid, sizeof(squared));
        if (std::sqrt(squared) < threshold) {
            low = mid;
        } else {
            high = mid - 1;
        }
    }
    double limit;
    std::memcpy(&limit, &low, sizeof(limit));
    return limit;
}

__device__ double squaredDistance(Point a, Point b) {
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    return __fma_rn(dx, dx, dy * dy);
}

// Find the horizontal search window in x-sorted point order using the
// squared horizontal distance as a lower bound.
__device__ int windowBoundary(const Point* points, const int* sorted, int count,
                              double x, double limit, bool upper) {
    int low = 0;
    int high = count;
    while (low < high) {
        const int mid = low + (high - low) / 2;
        const double dx = points[sorted[mid]].x - x;
        const bool outside = dx * dx > limit;
        const bool before = upper ? !(dx > 0 && outside) : dx < 0 && outside;
        if (before) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    return low;
}

// Count and then compact threshold neighbors into one contiguous row per seed.
template <bool Fill>
__global__ void buildNeighbors(const Point* points, const int* sorted, int count,
                               double limit, size_t* counts,
                               const size_t* offsets, int* neighbors) {
    const int seed = blockIdx.x * warps_per_block + threadIdx.x / warp_size;
    const int lane = threadIdx.x % warp_size;
    if (seed >= count) return;

    const Point center = points[seed];
    int first = 0;
    int last = 0;
    if (lane == 0 && limit >= 0) {
        first = windowBoundary(points, sorted, count, center.x, limit, false);
        last = windowBoundary(points, sorted, count, center.x, limit, true);
    }
    first = __shfl_sync(full_warp, first, 0);
    last = __shfl_sync(full_warp, last, 0);
    size_t written = Fill ? offsets[seed] : 0;
    for (size_t base = first; base < static_cast<size_t>(last); base += warp_size) {
        const size_t pos = base + lane;
        int point = -1;
        bool eligible = false;
        if (pos < static_cast<size_t>(last)) {
            point = sorted[pos];
            eligible = point != seed && squaredDistance(center, points[point]) <= limit;
        }
        const unsigned mask = __ballot_sync(full_warp, eligible);
        if constexpr (Fill) {
            if (eligible) {
                neighbors[written + __popc(mask & ((1u << lane) - 1))] = point;
            }
        }
        written += __popc(mask);
    }
    if constexpr (!Fill) {
        if (lane == 0) counts[seed] = written;
    }
}

struct Choice {
    double diameter;
    int point;
};

__device__ Choice earlier(Choice a, Choice b) {
    if (a.diameter == b.diameter) return a.point < b.point ? a : b;
    const Choice lower = a.diameter < b.diameter ? a : b;
    const Choice upper = a.diameter < b.diameter ? b : a;
    // A binary64 square-root rounding interval spans fewer than eight
    // adjacent squared values. Resolve possible rounded ties by point ID.
    if (upper.point < lower.point &&
        __double_as_longlong(upper.diameter) - __double_as_longlong(lower.diameter) < 8 &&
        __dsqrt_rn(upper.diameter) == __dsqrt_rn(lower.diameter)) {
        return upper;
    }
    return lower;
}

// All lanes own a subset of the seed's neighbors. Each surviving neighbor's
// diameter is updated against just the newest member of the candidate cluster.
template <int Slots>
__device__ int growCandidate(const Point* points, const int* clustered,
                             const int* neighbors, int degree, Point center,
                             double* scratch, double limit, int* members) {
    const int lane = threadIdx.x % warp_size;
    Choice candidates[Slots == 0 ? 1 : Slots];
    Point positions[Slots == 0 ? 1 : Slots];
    if constexpr (Slots == 0) {
        for (size_t i = lane; i < static_cast<size_t>(degree); i += warp_size) {
            const int point = neighbors[i];
            scratch[i] = clustered[point] ? CUDART_INF : squaredDistance(points[point], center);
        }
    } else {
        #pragma unroll
        for (int slot = 0; slot < Slots; ++slot) {
            const int i = slot * warp_size + lane;
            candidates[slot] = {CUDART_INF, INT_MAX};
            if (i < degree) {
                const int point = neighbors[i];
                if (!clustered[point]) {
                    positions[slot] = points[point];
                    candidates[slot] = {squaredDistance(positions[slot], center), point};
                }
            }
        }
    }

    int size = 1;
    int newest = -1;
    while (true) {
        Choice best{CUDART_INF, INT_MAX};
        const Point added = newest >= 0 ? points[newest] : Point{};
        if constexpr (Slots == 0) {
            for (size_t i = lane; i < static_cast<size_t>(degree); i += warp_size) {
                double diameter = scratch[i];
                if (diameter > limit) continue;
                const int point = neighbors[i];
                if (newest >= 0) {
                    diameter = point == newest ? CUDART_INF
                        : fmax(diameter, squaredDistance(points[point], added));
                    scratch[i] = diameter;
                }
                if (diameter <= limit) best = earlier(best, {diameter, point});
            }
        } else {
            #pragma unroll
            for (int slot = 0; slot < Slots; ++slot) {
                Choice& candidate = candidates[slot];
                if (newest >= 0 && candidate.diameter <= limit) {
                    candidate.diameter = candidate.point == newest ? CUDART_INF
                        : fmax(candidate.diameter, squaredDistance(positions[slot], added));
                }
                if (candidate.diameter <= limit) best = earlier(best, candidate);
            }
        }
        for (int delta = warp_size / 2; delta > 0; delta /= 2) {
            const Choice other{__shfl_down_sync(full_warp, best.diameter, delta),
                               __shfl_down_sync(full_warp, best.point, delta)};
            best = earlier(best, other);
        }
        newest = __shfl_sync(full_warp, best.point, 0);
        if (newest == INT_MAX) return size;
        if (lane == 0) members[size - 1] = newest;
        ++size;
    }
}

__global__ void evaluateSeeds(const Point* points, int count, double limit,
                              const size_t* offsets, const int* neighbors,
                              const int* clustered, int* sizes, int* members,
                              double* scratch, RoundState* state) {
    if (state->written == count) return;
    const int seed = blockIdx.x * warps_per_block + threadIdx.x / warp_size;
    const int lane = threadIdx.x % warp_size;
    Winner key = 0;
    if (seed < count && !clustered[seed]) {
        int size = sizes[seed];
        const size_t start = offsets[seed];

        // Removing points outside the old candidate leaves every previously
        // selected minimum available, so the entire greedy sequence is unchanged.
        bool rebuild = size == 0;
        const size_t cached_members = size > 0 ? size - 1 : 0;
        for (size_t i = lane; i < cached_members && !rebuild; i += warp_size) {
            rebuild = clustered[members[start + i]];
        }
        if (__any_sync(full_warp, rebuild)) {
            const int degree = static_cast<int>(offsets[seed + 1] - start);
            size = 1;
            if (degree != 0) {
                const Point center = points[seed];
                const int* row = neighbors + start;
                int* selected = members + start;
                // Small rows keep all candidate state in registers.
                if (degree <= warp_size) {
                    size = growCandidate<1>(points, clustered, row, degree, center,
                                            nullptr, limit, selected);
                } else if (degree <= 2 * warp_size) {
                    size = growCandidate<2>(points, clustered, row, degree, center,
                                            nullptr, limit, selected);
                } else if (degree <= 4 * warp_size) {
                    size = growCandidate<4>(points, clustered, row, degree, center,
                                            nullptr, limit, selected);
                } else {
                    size = growCandidate<0>(points, clustered, row, degree, center,
                                            scratch + start, limit, selected);
                }
            }
            if (lane == 0) sizes[seed] = size;
        }
        if (lane == 0) {
            key = (static_cast<Winner>(size) << 32) | (UINT_MAX - static_cast<unsigned>(seed));
        }
    }
    using BlockReduce = cub::BlockReduce<Winner, block_size>;
    __shared__ BlockReduce::TempStorage storage;
    const Winner best = BlockReduce(storage).Reduce(key, cub::Max());
    if (threadIdx.x == 0 && best != 0) atomicMax(&state->best, best);
}

__global__ void commitCluster(int count, const size_t* offsets, const int* members,
                              int* clustered, OutputMember* output, RoundState* state) {
    const int written = state->written;
    if (written == count) return;
    const Winner best = state->best;
    const int size = static_cast<int>(best >> 32);
    const int seed = static_cast<int>(UINT_MAX - static_cast<unsigned>(best));
    const int lane = threadIdx.x;
    int end = written + size;
    if (size == 1) {
        // With no remaining pair below threshold, all remaining seeds are
        // singleton clusters and can be emitted together in original order.
        using BlockScan = cub::BlockScan<int, block_size>;
        __shared__ BlockScan::TempStorage storage;
        end = written;
        for (size_t base = 0; base < static_cast<size_t>(count); base += block_size) {
            const size_t point = base + lane;
            const int active = point < static_cast<size_t>(count) && !clustered[point];
            int rank;
            int total;
            BlockScan(storage).ExclusiveSum(active, rank, total);
            if (active) output[end + rank] = {static_cast<int>(point), 1};
            end += total;
            __syncthreads();
        }
    } else {
        const size_t start = offsets[seed];
        for (size_t i = lane; i < static_cast<size_t>(size); i += block_size) {
            const int point = i == 0 ? seed : members[start + i - 1];
            clustered[point] = 1;
            output[written + i] = {point, i == 0 ? size : 0};
        }
    }
    __syncthreads();
    if (lane == 0) {
        state->best = 0;
        state->written = end;
    }
}

}  // namespace

std::vector<Cluster> qtClustering(const std::vector<Point>& points, double threshold) {
    const int count = static_cast<int>(points.size());
    if (count == 0) return {};

    const double limit = squaredLimit(threshold);
    std::vector<int> order(count);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&points](int a, int b) {
        return points[a].x < points[b].x;
    });

    DeviceBuffer<Point> device_points(count);
    DeviceBuffer<int> sorted(count);
    DeviceBuffer<size_t> counts(points.size() + 1);
    DeviceBuffer<size_t> offsets(points.size() + 1);
    checkCuda(cudaMemcpy(device_points.get(), points.data(), count * sizeof(Point),
                         cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(sorted.get(), order.data(), count * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemset(counts.get() + count, 0, sizeof(size_t)));

    const int blocks = (count - 1) / warps_per_block + 1;
    buildNeighbors<false><<<blocks, block_size>>>(device_points.get(), sorted.get(), count,
                                                limit, counts.get(), nullptr, nullptr);
    checkCuda(cudaGetLastError());
    size_t scan_bytes = 0;
    checkCuda(cub::DeviceScan::ExclusiveSum(nullptr, scan_bytes, counts.get(), offsets.get(),
                                            points.size() + 1));
    DeviceBuffer<unsigned char> scan_storage(scan_bytes);
    checkCuda(cub::DeviceScan::ExclusiveSum(scan_storage.get(), scan_bytes, counts.get(),
                                            offsets.get(), points.size() + 1));

    size_t edges;
    checkCuda(cudaMemcpy(&edges, offsets.get() + count, sizeof(edges), cudaMemcpyDeviceToHost));
    DeviceBuffer<int> neighbors(edges);
    if (edges > 0) {
        buildNeighbors<true><<<blocks, block_size>>>(device_points.get(), sorted.get(), count,
                                                   limit, nullptr, offsets.get(), neighbors.get());
        checkCuda(cudaGetLastError());
    }

    DeviceBuffer<int> clustered(count);
    DeviceBuffer<int> sizes(count);
    DeviceBuffer<int> members(edges);
    DeviceBuffer<double> scratch(edges);
    DeviceBuffer<OutputMember> result(count);
    DeviceBuffer<RoundState> state(1);
    checkCuda(cudaMemset(clustered.get(), 0, count * sizeof(int)));
    checkCuda(cudaMemset(sizes.get(), 0, count * sizeof(int)));
    checkCuda(cudaMemset(state.get(), 0, sizeof(RoundState)));

    // Keep the greedy rounds on the device and poll completion in batches.
    RoundState progress{};
    constexpr int rounds_per_batch = 16;
    do {
        for (int round = 0; round < rounds_per_batch; ++round) {
            evaluateSeeds<<<blocks, block_size>>>(device_points.get(), count, limit,
                offsets.get(), neighbors.get(), clustered.get(), sizes.get(), members.get(),
                scratch.get(), state.get());
            commitCluster<<<1, block_size>>>(count, offsets.get(), members.get(), clustered.get(),
                                            result.get(), state.get());
        }
        checkCuda(cudaGetLastError());
        checkCuda(cudaMemcpy(&progress, state.get(), sizeof(progress), cudaMemcpyDeviceToHost));
    } while (progress.written < count);

    std::vector<OutputMember> output(count);
    checkCuda(cudaMemcpy(output.data(), result.get(), count * sizeof(OutputMember),
                         cudaMemcpyDeviceToHost));
    std::vector<Cluster> clusters;
    for (int i = 0; i < count; ) {
        const OutputMember first = output[i];
        Cluster cluster;
        cluster.seed_point = first.point;
        cluster.members.reserve(first.cluster_size);
        for (int j = 0; j < first.cluster_size; ++j) {
            cluster.members.push_back(output[i + j].point);
        }
        i += first.cluster_size;
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
    
    std::vector<Cluster> clusters;
    try {
        clusters = qtClustering(points, threshold);
    } catch (const std::exception& error) {
        fprintf(stderr, "QT clustering failed: %s\n", error.what());
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
