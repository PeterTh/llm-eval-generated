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
#include <climits>
#include <cfloat>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

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
__host__ __device__ inline double distance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

static void cudaCheck(cudaError_t status) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(status));
}

template <class T> class DeviceBuffer {
public:
    T* data = nullptr;
    explicit DeviceBuffer(size_t count) {
        if (count > std::numeric_limits<size_t>::max() / sizeof(T))
            throw std::runtime_error("CUDA allocation size overflow");
        if (count) cudaCheck(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)));
    }
    ~DeviceBuffer() { if (data) cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

// A warp builds one sorted sparse neighborhood. Every possible member must be
// strictly within threshold of the seed. Include the seed even for NaN threshold.
template <bool Fill>
__global__ void neighborhoods(const Point* points, int n, double threshold,
                              size_t* offsets, int* neighbors) {
    const int seed = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x % 32;
    if (seed >= n) return;
    size_t count = 0;
    const size_t start = Fill ? offsets[seed] : 0;
    for (size_t base = 0; base < static_cast<size_t>(n); base += 32) {
        const size_t j = base + lane;
        const bool keep = j < static_cast<size_t>(n) &&
            (j == static_cast<size_t>(seed) || distance(points[seed], points[j]) < threshold);
        const unsigned mask = __ballot_sync(0xffffffffu, keep);
        if (Fill && keep)
            neighbors[start + count + __popc(mask & ((1u << lane) - 1))] = static_cast<int>(j);
        count += __popc(mask);
    }
    if (!Fill && lane == 0) offsets[seed] = count;
}

struct Choice { double value; int index; };
struct MinimumChoice {
    __device__ Choice operator()(Choice a, Choice b) const {
        return b.value < a.value || (b.value == a.value && b.index < a.index) ? b : a;
    }
};

// One block per seed, lanes parallelize candidate checks and deterministic argmin.
// Maxima only need updating against the most recently appended member, reducing
// cluster growth from cubic to quadratic work. Scratch is sparse, not N by N.
__global__ void growCandidates(const Point* points, int n, double threshold,
                               const size_t* offsets, const int* neighbors,
                               double* maxima, int* members, int* sizes,
                               const unsigned char* clustered, const unsigned char* dirty) {
    const int seed = blockIdx.x;
    if (seed >= n) return;
    if (clustered[seed]) { if (threadIdx.x == 0) sizes[seed] = 0; return; }
    if (!dirty[seed]) return;
    using Reduce = cub::BlockReduce<Choice, 128>;
    __shared__ typename Reduce::TempStorage storage;
    __shared__ int selected;
    const size_t begin = offsets[seed], end = offsets[seed + 1];
    for (size_t j = begin + threadIdx.x; j < end; j += blockDim.x)
        maxima[j] = (clustered[neighbors[j]] || neighbors[j] == seed) ?
            DBL_MAX : 0.0;
    int last = seed, count = 1;
    if (threadIdx.x == 0) members[begin] = seed;
    __syncthreads();
    while (true) {
        Choice local{DBL_MAX, INT_MAX};
        for (size_t j = begin + threadIdx.x; j < end; j += blockDim.x) {
            double value = maxima[j];
            if (value < threshold && value < DBL_MAX) {
                const int candidate = neighbors[j];
                if (candidate == last) value = DBL_MAX;
                else value = fmax(value, distance(points[candidate], points[last]));
                maxima[j] = value;
                if (value < threshold && value < DBL_MAX)
                    local = MinimumChoice{}(local, Choice{value, candidate});
            }
        }
        const Choice best = Reduce(storage).Reduce(local, MinimumChoice{});
        if (threadIdx.x == 0) {
            selected = best.index;
            if (selected != INT_MAX) members[begin + count] = selected;
        }
        __syncthreads();
        if (selected == INT_MAX) break;
        last = selected;
        ++count;
        // The barrier also protects reduction storage before its next use.
    }
    if (threadIdx.x == 0) sizes[seed] = count;
}

struct Winner { int size; int seed; };
struct MaximumWinner {
    __device__ Winner operator()(Winner a, Winner b) const {
        return b.size > a.size || (b.size == a.size && b.seed < a.seed) ? b : a;
    }
};

__global__ void selectCluster(int n, int remaining, const size_t* offsets, const int* members,
                             const int* sizes, unsigned char* clustered, int* output) {
    using Reduce = cub::BlockReduce<Winner, 256>;
    using Scan = cub::BlockScan<int, 256>;
    __shared__ typename Reduce::TempStorage storage;
    __shared__ typename Scan::TempStorage scanStorage;
    __shared__ Winner winner;
    Winner local{0, INT_MAX};
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        local = MaximumWinner{}(local, Winner{sizes[i], i});
    const Winner best = Reduce(storage).Reduce(local, MaximumWinner{});
    if (threadIdx.x == 0) {
        winner = best;
        output[0] = best.size;
        output[1] = best.seed;
    }
    __syncthreads();
    // If the maximum size is one, all remaining clusters are singletons in
    // ascending seed order. Compact them on the GPU in one final round trip.
    if (winner.size == 1) {
        int written = 0;
        for (size_t base = 0; base < static_cast<size_t>(n); base += blockDim.x) {
            const size_t i = base + threadIdx.x;
            const int keep = i < static_cast<size_t>(n) && sizes[i] != 0;
            int prefix, total;
            Scan(scanStorage).ExclusiveSum(keep, prefix, total);
            if (keep) {
                output[2 + written + prefix] = static_cast<int>(i);
                clustered[i] = 1;
            }
            written += total;
            __syncthreads();
        }
        if (threadIdx.x == 0) output[0] = -remaining;
        return;
    }
    for (int i = threadIdx.x; i < winner.size; i += blockDim.x) {
        const int member = members[offsets[winner.seed] + i];
        output[2 + i] = member;
        clustered[member] = 1;
    }
}

// Removing points outside a cached greedy candidate cannot change that candidate.
__global__ void invalidateCandidates(int n, const size_t* offsets, const int* members,
                                    const int* sizes, const unsigned char* clustered,
                                    unsigned char* dirty) {
    const int seed = blockIdx.x;
    if (seed >= n || clustered[seed]) return;
    bool invalid = false;
    for (int i = threadIdx.x; i < sizes[seed]; i += blockDim.x)
        invalid |= clustered[members[offsets[seed] + i]] != 0;
    const int any = __syncthreads_or(invalid);
    if (threadIdx.x == 0) dirty[seed] = any;
}

std::vector<Cluster> qtClustering(const std::vector<Point>& points, const double threshold) {
    const int n = static_cast<int>(points.size());
    std::vector<Cluster> clusters;
    if (!n) return clusters;
    DeviceBuffer<Point> devicePoints(n);
    DeviceBuffer<size_t> offsets(static_cast<size_t>(n) + 1);
    DeviceBuffer<int> sizes(n), output(static_cast<size_t>(n) + 2);
    DeviceBuffer<unsigned char> clustered(n), dirty(n);
    cudaCheck(cudaMemcpy(devicePoints.data, points.data(), n * sizeof(Point), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemset(clustered.data, 0, n));
    cudaCheck(cudaMemset(dirty.data, 1, n));
    cudaCheck(cudaMemset(offsets.data + n, 0, sizeof(size_t)));
    neighborhoods<false><<<(static_cast<size_t>(n) + 3) / 4, 128>>>(
        devicePoints.data, n, threshold, offsets.data, nullptr);
    cudaCheck(cudaGetLastError());
    auto ptr = thrust::device_pointer_cast(offsets.data);
    thrust::exclusive_scan(ptr, ptr + static_cast<size_t>(n) + 1, ptr);
    size_t edges = 0;
    cudaCheck(cudaMemcpy(&edges, offsets.data + n, sizeof(size_t), cudaMemcpyDeviceToHost));
    DeviceBuffer<int> neighbors(edges), members(edges);
    DeviceBuffer<double> maxima(edges);
    neighborhoods<true><<<(static_cast<size_t>(n) + 3) / 4, 128>>>(
        devicePoints.data, n, threshold, offsets.data, neighbors.data);
    cudaCheck(cudaGetLastError());
    int remaining = n;
    while (remaining) {
        growCandidates<<<n, 128>>>(devicePoints.data, n, threshold, offsets.data,
            neighbors.data, maxima.data, members.data, sizes.data, clustered.data, dirty.data);
        cudaCheck(cudaGetLastError());
        selectCluster<<<1, 256>>>(n, remaining, offsets.data, members.data, sizes.data, clustered.data, output.data);
        cudaCheck(cudaGetLastError());
        int header[2];
        cudaCheck(cudaMemcpy(header, output.data, sizeof(header), cudaMemcpyDeviceToHost));
        if (header[0] < 0) {
            std::vector<int> singletons(-header[0]);
            cudaCheck(cudaMemcpy(singletons.data(), output.data + 2,
                                 singletons.size() * sizeof(int), cudaMemcpyDeviceToHost));
            for (int seed : singletons) clusters.push_back(Cluster{{seed}, seed});
            break;
        }
        Cluster cluster;
        cluster.seed_point = header[1];
        cluster.members.resize(header[0]);
        cudaCheck(cudaMemcpy(cluster.members.data(), output.data + 2,
                             header[0] * sizeof(int), cudaMemcpyDeviceToHost));
        remaining -= header[0];
        clusters.push_back(std::move(cluster));
        if (remaining) {
            invalidateCandidates<<<n, 128>>>(n, offsets.data, members.data, sizes.data,
                                             clustered.data, dirty.data);
            cudaCheck(cudaGetLastError());
        }
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
        fprintf(stderr, "Clustering failed: %s\n", error.what());
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
