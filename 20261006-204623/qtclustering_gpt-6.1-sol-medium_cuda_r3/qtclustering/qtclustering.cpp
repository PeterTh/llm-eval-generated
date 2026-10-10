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
#include <stdexcept>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <climits>
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
inline double distance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

// CUDA errors are fatal: clustering never falls back to a CPU implementation.
static void cudaCheck(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

template <typename T> class DeviceBuffer {
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

static constexpr int BLOCK_SIZE = 128;
static constexpr int SHARED_CANDIDATES = 4096;

__device__ double gpuDistance(Point a, Point b) {
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    // Match the CPU's double precision distance and strict threshold comparison.
    return sqrt(dx * dx + dy * dy);
}

// Lexicographic minimum (distance, point index), reproducing sequential ties.
__device__ int blockMinimum(double value, int index) {
    __shared__ double warp_values[BLOCK_SIZE / 32];
    __shared__ int warp_indices[BLOCK_SIZE / 32];
    __shared__ int result;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x / 32;
    for (int offset = 16; offset; offset /= 2) {
        const double other = __shfl_down_sync(0xffffffff, value, offset);
        const int other_index = __shfl_down_sync(0xffffffff, index, offset);
        if (other < value || (other == value && other_index < index)) {
            value = other;
            index = other_index;
        }
    }
    if (lane == 0) {
        warp_values[warp] = value;
        warp_indices[warp] = index;
    }
    __syncthreads();
    if (warp == 0) {
        value = lane < BLOCK_SIZE / 32 ? warp_values[lane] : CUDART_INF;
        index = lane < BLOCK_SIZE / 32 ? warp_indices[lane] : INT_MAX;
        for (int offset = 16; offset; offset /= 2) {
            const double other = __shfl_down_sync(0xffffffff, value, offset);
            const int other_index = __shfl_down_sync(0xffffffff, index, offset);
            if (other < value || (other == value && other_index < index)) {
                value = other;
                index = other_index;
            }
        }
        if (lane == 0) result = index;
    }
    __syncthreads();
    return result;
}

// Only points within the threshold of a seed can ever join its cluster.
// CSR storage avoids allocating a full N-by-N distance matrix.
__global__ void countNeighbors(const Point* points, int n, double threshold,
                               size_t* offsets) {
    const int seed = blockIdx.x;
    int count = 0;
    const Point origin = points[seed];
    for (int i = threadIdx.x; i < n; i += BLOCK_SIZE)
        count += (i == seed || gpuDistance(origin, points[i]) < threshold);
    __shared__ int counts[BLOCK_SIZE];
    counts[threadIdx.x] = count;
    __syncthreads();
    for (int stride = BLOCK_SIZE / 2; stride; stride /= 2) {
        if (threadIdx.x < stride) counts[threadIdx.x] += counts[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) offsets[seed + 1] = counts[0];
}

__global__ void fillNeighbors(const Point* points, int n, double threshold,
                              const size_t* offsets, int* neighbors) {
    const int seed = blockIdx.x;
    const Point origin = points[seed];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x / 32;
    __shared__ int warp_offsets[BLOCK_SIZE / 32];
    __shared__ int total;
    size_t out = offsets[seed];
    for (size_t tile = 0; tile < static_cast<size_t>(n); tile += BLOCK_SIZE) {
        const size_t i = tile + threadIdx.x;
        const bool include = i < static_cast<size_t>(n) &&
            (i == static_cast<size_t>(seed) || gpuDistance(origin, points[i]) < threshold);
        const unsigned mask = __ballot_sync(0xffffffff, include);
        if (lane == 0) warp_offsets[warp] = __popc(mask);
        __syncthreads();
        if (threadIdx.x == 0) {
            int sum = 0;
            for (int w = 0; w < BLOCK_SIZE / 32; ++w) {
                const int count = warp_offsets[w];
                warp_offsets[w] = sum;
                sum += count;
            }
            total = sum;
        }
        __syncthreads();
        if (include) neighbors[out + warp_offsets[warp] +
            __popc(mask & ((1u << lane) - 1u))] = static_cast<int>(i);
        out += total;
        __syncthreads();
    }
}

// A block grows one seed's cluster; threads update candidates and reduce the
// next member cooperatively. Maximum distances need only the newest member.
__global__ void candidateClusters(const Point* points, int n, double threshold,
                                  const size_t* offsets, const int* neighbors,
                                  const unsigned char* clustered, int* cardinalities,
                                  int* members, double* scratch) {
    const int seed = blockIdx.x;
    if (seed >= n) return;
    if (clustered[seed]) {
        if (threadIdx.x == 0) cardinalities[seed] = 0;
        return;
    }
    const size_t begin = offsets[seed];
    const int degree = static_cast<int>(offsets[seed + 1] - begin);
    const int previous = cardinalities[seed];
    if (previous > 0) {
        bool invalid = false;
        for (int i = threadIdx.x; i < previous; i += BLOCK_SIZE)
            invalid |= clustered[members[begin + i]] != 0;
        // Removing points outside this cluster cannot change its greedy path.
        if (!__syncthreads_or(invalid)) return;
    }
    extern __shared__ double shared_distances[];
    double* max_distances = degree <= SHARED_CANDIDATES ? shared_distances : scratch + begin;
    for (int i = threadIdx.x; i < degree; i += BLOCK_SIZE) {
        const int point = neighbors[begin + i];
        max_distances[i] = point == seed || clustered[point] ? CUDART_INF : 0.0;
    }
    if (threadIdx.x == 0) members[begin] = seed;
    __syncthreads();
    int last = seed;
    int count = 1;
    while (count < degree) {
        double best_distance = CUDART_INF;
        int best_index = INT_MAX;
        const Point newest = points[last];
        for (int i = threadIdx.x; i < degree; i += BLOCK_SIZE) {
            const int point = neighbors[begin + i];
            double maximum = max_distances[i];
            if (point == last) maximum = CUDART_INF;
            if (maximum < threshold) {
                maximum = fmax(maximum, gpuDistance(points[point], newest));
                if (!(maximum < threshold)) maximum = CUDART_INF;
            }
            max_distances[i] = maximum;
            if (maximum < best_distance ||
                (maximum == best_distance && maximum < threshold && point < best_index)) {
                best_distance = maximum;
                best_index = point;
            }
        }
        last = blockMinimum(best_distance, best_index);
        if (last == INT_MAX) break;
        if (threadIdx.x == 0) members[begin + count] = last;
        ++count;
    }
    if (threadIdx.x == 0) cardinalities[seed] = count;
}

// Choose the largest cluster, with the smallest seed index winning ties.
// Keep membership, output ordering, and progress on the device between rounds.
__global__ void commitCluster(int n, const size_t* offsets, const int* cardinalities,
                              const int* members, unsigned char* clustered,
                              int* output, int* ends, int* seeds, int* progress) {
    double best = CUDART_INF;
    int seed = INT_MAX;
    for (int i = threadIdx.x; i < n; i += BLOCK_SIZE) {
        const int count = cardinalities[i];
        if (count > 0 && (-static_cast<double>(count) < best ||
            (-static_cast<double>(count) == best && i < seed))) {
            best = -static_cast<double>(count);
            seed = i;
        }
    }
    seed = blockMinimum(best, seed);
    if (seed == INT_MAX) return;
    const int count = cardinalities[seed];
    if (count == 1) {
        // All remaining clusters are singletons. Their sequential winner order
        // is simply ascending point index, so finish them in one device pass.
        if (threadIdx.x == 0) {
            int start = progress[0];
            int cluster = progress[1];
            for (int i = 0; i < n; ++i) {
                if (cardinalities[i] > 0) {
                    output[start++] = i;
                    clustered[i] = 1;
                    ends[cluster] = start;
                    seeds[cluster++] = i;
                }
            }
            progress[0] = start;
            progress[1] = cluster;
        }
        return;
    }
    const int start = progress[0];
    const int cluster = progress[1];
    for (int i = threadIdx.x; i < count; i += BLOCK_SIZE) {
        const int member = members[offsets[seed] + i];
        output[start + i] = member;
        clustered[member] = 1;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        ends[cluster] = start + count;
        seeds[cluster] = seed;
        progress[0] = start + count;
        progress[1] = cluster + 1;
    }
}

std::vector<Cluster> qtClustering(const std::vector<Point>& points, double threshold) {
    const int n = static_cast<int>(points.size());
    if (!n) return {};
    // If a conservative bound proves all points fit together, seed zero
    // necessarily wins. Grow only that candidate, retaining its greedy order.
    Point low = points[0], high = points[0];
    for (const Point& p : points) {
        low.x = std::min(low.x, p.x);
        low.y = std::min(low.y, p.y);
        high.x = std::max(high.x, p.x);
        high.y = std::max(high.y, p.y);
    }
    const double bound = std::nextafter((high.x - low.x) + (high.y - low.y),
                                       std::numeric_limits<double>::infinity());
    const int seed_count = bound < threshold ? 1 : n;
    DeviceBuffer<Point> device_points(n);
    DeviceBuffer<size_t> offsets(static_cast<size_t>(n) + 1);
    DeviceBuffer<unsigned char> clustered(n);
    DeviceBuffer<int> cardinalities(n), output(n), ends(n), seeds(n), progress(2);
    cudaCheck(cudaMemcpy(device_points.data, points.data(), n * sizeof(Point), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemset(offsets.data, 0, (static_cast<size_t>(n) + 1) * sizeof(size_t)));
    cudaCheck(cudaMemset(clustered.data, 0, n));
    cudaCheck(cudaMemset(cardinalities.data, 0, n * sizeof(int)));
    cudaCheck(cudaMemset(progress.data, 0, 2 * sizeof(int)));
    countNeighbors<<<seed_count, BLOCK_SIZE>>>(device_points.data, n, threshold, offsets.data);
    cudaCheck(cudaGetLastError());
    auto offset_ptr = thrust::device_pointer_cast(offsets.data);
    thrust::inclusive_scan(offset_ptr, offset_ptr + static_cast<size_t>(n) + 1, offset_ptr);
    std::vector<size_t> host_offsets(static_cast<size_t>(n) + 1);
    cudaCheck(cudaMemcpy(host_offsets.data(), offsets.data, host_offsets.size() * sizeof(size_t), cudaMemcpyDeviceToHost));
    size_t max_degree = 0;
    for (int i = 0; i < n; ++i)
        max_degree = std::max(max_degree, host_offsets[i + 1] - host_offsets[i]);
    const size_t edges = host_offsets[n];
    DeviceBuffer<int> neighbors(edges), members(edges);
    // Large neighborhoods use global scratch; ordinary neighborhoods stay in shared memory.
    DeviceBuffer<double> scratch(max_degree > SHARED_CANDIDATES ? edges : 0);
    const size_t shared_bytes = std::min(max_degree, static_cast<size_t>(SHARED_CANDIDATES)) * sizeof(double);
    fillNeighbors<<<seed_count, BLOCK_SIZE>>>(device_points.data, n, threshold, offsets.data, neighbors.data);
    cudaCheck(cudaGetLastError());
    int host_progress[2] = {0, 0};
    while (host_progress[0] < n) {
        candidateClusters<<<seed_count, BLOCK_SIZE, shared_bytes>>>(device_points.data, n, threshold,
            offsets.data, neighbors.data, clustered.data, cardinalities.data, members.data, scratch.data);
        cudaCheck(cudaGetLastError());
        commitCluster<<<1, BLOCK_SIZE>>>(n, offsets.data, cardinalities.data, members.data,
            clustered.data, output.data, ends.data, seeds.data, progress.data);
        cudaCheck(cudaGetLastError());
        cudaCheck(cudaMemcpy(host_progress, progress.data, sizeof(host_progress), cudaMemcpyDeviceToHost));
    }
    std::vector<int> host_output(n), host_ends(host_progress[1]), host_seeds(host_progress[1]);
    cudaCheck(cudaMemcpy(host_output.data(), output.data, n * sizeof(int), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(host_ends.data(), ends.data, host_ends.size() * sizeof(int), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(host_seeds.data(), seeds.data, host_seeds.size() * sizeof(int), cudaMemcpyDeviceToHost));
    std::vector<Cluster> clusters;
    clusters.reserve(host_ends.size());
    int start = 0;
    for (size_t c = 0; c < host_ends.size(); ++c) {
        clusters.push_back({std::vector<int>(host_output.begin() + start,
                                           host_output.begin() + host_ends[c]), host_seeds[c]});
        start = host_ends[c];
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
    // Initialize the required CUDA runtime before measuring the algorithm.
    try {
        cudaCheck(cudaFree(nullptr));
    } catch (const std::exception& error) {
        fprintf(stderr, "CUDA initialization failed: %s\n", error.what());
        return 1;
    }
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
