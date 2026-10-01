// QT Clustering Benchmark - CUDA implementation
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
#include <stdexcept>
#include <cuda_runtime.h>
#include <math_constants.h>

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

// All CUDA failures are fatal: clustering never falls back to the CPU.
static void cudaCheck(cudaError_t error) {
    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }
}

template<class T> class DeviceBuffer {
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

constexpr int BLOCK_SIZE = 128;

__device__ double gpuDistance(Point a, Point b) {
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    // Match the original double-precision sqrt and separate multiply/adds.
    return sqrt(dx * dx + dy * dy);
}

struct Choice {
    double distance;
    int point;
};

__device__ Choice better(Choice a, Choice b) {
    return b.distance < a.distance ||
           (b.distance == a.distance && b.point < a.point) ? b : a;
}

__device__ Choice warpMinimum(Choice value) {
    for (int delta = 16; delta; delta /= 2) {
        Choice other{__shfl_down_sync(0xffffffff, value.distance, delta),
                     __shfl_down_sync(0xffffffff, value.point, delta)};
        value = better(value, other);
    }
    return value;
}

__device__ Choice blockMinimum(Choice value, Choice* shared) {
    value = warpMinimum(value);
    if ((threadIdx.x & 31) == 0) shared[threadIdx.x / 32] = value;
    __syncthreads();
    if (threadIdx.x < 32) {
        value = threadIdx.x < BLOCK_SIZE / 32 ? shared[threadIdx.x] :
                Choice{CUDART_INF, INT_MAX};
        value = warpMinimum(value);
        if (threadIdx.x == 0) shared[0] = value;
    }
    __syncthreads();
    return shared[0];
}

// A seed can only ever admit points within threshold of itself. Store these
// neighborhoods in CSR form, avoiding a dense N-by-N distance matrix.
__global__ void countNeighbors(const Point* points, int n, double threshold, int* counts) {
    const int seed = blockIdx.x;
    const Point p = points[seed];
    int count = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        count += i == seed || gpuDistance(p, points[i]) < threshold;
    __shared__ int sums[BLOCK_SIZE];
    sums[threadIdx.x] = count;
    __syncthreads();
    for (int stride = BLOCK_SIZE / 2; stride; stride /= 2) {
        if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) counts[seed] = sums[0];
}

__global__ void fillNeighbors(const Point* points, int n, double threshold,
                             const size_t* offsets, int* neighbors) {
    const int seed = blockIdx.x;
    const Point p = points[seed];
    const unsigned lane = threadIdx.x & 31;
    __shared__ int used;
    if (threadIdx.x == 0) used = 0;
    __syncthreads();
    for (size_t base = 0; base < static_cast<size_t>(n); base += BLOCK_SIZE) {
        const size_t i = base + threadIdx.x;
        const bool include = i < static_cast<size_t>(n) &&
            (i == static_cast<size_t>(seed) || gpuDistance(p, points[i]) < threshold);
        const unsigned mask = __ballot_sync(0xffffffff, include);
        int start = 0;
        if (lane == 0) start = atomicAdd(&used, __popc(mask));
        start = __shfl_sync(0xffffffff, start, 0);
        if (include) {
            const unsigned preceding = mask & ((1u << lane) - 1u);
            neighbors[offsets[seed] + start + __popc(preceding)] = static_cast<int>(i);
        }
    }
}

// One cooperative block per seed. Updating the maximum distance with only the
// newest member is equivalent to rescanning every member on every iteration.
// Cached clusters remain valid when none of their members have been removed:
// deleting other candidates cannot change any of their greedy choices.
__global__ void buildCandidates(const Point* points, const size_t* offsets,
                                const int* neighbors, const unsigned char* clustered,
                                double threshold, double* maxima, int* members,
                                int* sizes) {
    const int seed = blockIdx.x;
    const int tid = threadIdx.x;
    if (clustered[seed]) {
        if (tid == 0) sizes[seed] = 0;
        return;
    }
    const size_t begin = offsets[seed];
    const size_t end = offsets[seed + 1];
    const int oldSize = sizes[seed];
    int invalid = oldSize == 0;
    for (int i = tid; i < oldSize; i += BLOCK_SIZE)
        invalid |= clustered[members[begin + i]];
    if (!__syncthreads_or(invalid)) return;

    for (size_t i = begin + tid; i < end; i += BLOCK_SIZE)
        maxima[i] = clustered[neighbors[i]] ? CUDART_INF : 0.0;
    if (tid == 0) members[begin] = seed;
    __syncthreads();
    __shared__ Choice reduction[BLOCK_SIZE / 32];
    int newest = seed;
    int size = 1;
    for (;;) {
        const Point p = points[newest];
        Choice best{CUDART_INF, INT_MAX};
        for (size_t i = begin + tid; i < end; i += BLOCK_SIZE) {
            double d = maxima[i];
            const int candidate = neighbors[i];
            if (candidate == newest) d = CUDART_INF;
            if (d < threshold) {
                d = fmax(d, gpuDistance(points[candidate], p));
                if (!(d < threshold)) d = CUDART_INF;
            }
            maxima[i] = d;
            if (d < threshold) best = better(best, Choice{d, candidate});
        }
        const Choice chosen = blockMinimum(best, reduction);
        if (chosen.point == INT_MAX) break;
        newest = chosen.point;
        if (tid == 0) members[begin + size] = newest;
        ++size;
        // Protect the shared reduction storage before its next use.
        __syncthreads();
    }
    if (tid == 0) sizes[seed] = size;
}

__global__ void chooseCluster(const int* sizes, int n, const size_t* offsets,
                              const int* members, unsigned char* clustered, int* winner) {
    Choice best{CUDART_INF, INT_MAX};
    for (int i = threadIdx.x; i < n; i += BLOCK_SIZE)
        if (sizes[i]) best = better(best, Choice{-static_cast<double>(sizes[i]), i});
    __shared__ Choice reduction[BLOCK_SIZE / 32];
    const int seed = blockMinimum(best, reduction).point;
    const int size = sizes[seed];
    if (threadIdx.x == 0) {
        winner[0] = seed;
        winner[1] = size;
    }
    for (int i = threadIdx.x; i < size; i += BLOCK_SIZE)
        clustered[members[offsets[seed] + i]] = 1;
}

std::vector<Cluster> qtClustering(const std::vector<Point>& points, double threshold) {
    const int n = static_cast<int>(points.size());
    // Initialize CUDA even for empty input; there is no sequential execution path.
    cudaCheck(cudaFree(nullptr));
    if (n == 0) return {};
    DeviceBuffer<Point> devicePoints(n);
    DeviceBuffer<int> sizes(n), winner(2);
    DeviceBuffer<size_t> deviceOffsets(static_cast<size_t>(n) + 1);
    DeviceBuffer<unsigned char> clustered(n);
    cudaCheck(cudaMemcpy(devicePoints.data, points.data(), n * sizeof(Point), cudaMemcpyHostToDevice));
    countNeighbors<<<n, BLOCK_SIZE>>>(devicePoints.data, n, threshold, sizes.data);
    cudaCheck(cudaGetLastError());
    std::vector<int> counts(n);
    cudaCheck(cudaMemcpy(counts.data(), sizes.data, n * sizeof(int), cudaMemcpyDeviceToHost));
    std::vector<size_t> offsets(static_cast<size_t>(n) + 1, 0);
    for (int i = 0; i < n; ++i) offsets[i + 1] = offsets[i] + counts[i];
    DeviceBuffer<int> neighbors(offsets[n]), members(offsets[n]);
    DeviceBuffer<double> maxima(offsets[n]);
    cudaCheck(cudaMemcpy(deviceOffsets.data, offsets.data(), offsets.size() * sizeof(size_t), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemset(clustered.data, 0, n));
    cudaCheck(cudaMemset(sizes.data, 0, n * sizeof(int)));
    fillNeighbors<<<n, BLOCK_SIZE>>>(devicePoints.data, n, threshold, deviceOffsets.data, neighbors.data);
    cudaCheck(cudaGetLastError());
    std::vector<Cluster> clusters;
    int remaining = n;
    while (remaining) {
        buildCandidates<<<n, BLOCK_SIZE>>>(devicePoints.data, deviceOffsets.data, neighbors.data,
            clustered.data, threshold, maxima.data, members.data, sizes.data);
        cudaCheck(cudaGetLastError());
        chooseCluster<<<1, BLOCK_SIZE>>>(sizes.data, n, deviceOffsets.data, members.data,
                                        clustered.data, winner.data);
        cudaCheck(cudaGetLastError());
        int selected[2];
        cudaCheck(cudaMemcpy(selected, winner.data, sizeof(selected), cudaMemcpyDeviceToHost));
        Cluster cluster;
        cluster.seed_point = selected[0];
        cluster.members.resize(selected[1]);
        cudaCheck(cudaMemcpy(cluster.members.data(), members.data + offsets[selected[0]],
                            selected[1] * sizeof(int), cudaMemcpyDeviceToHost));
        remaining -= selected[1];
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
