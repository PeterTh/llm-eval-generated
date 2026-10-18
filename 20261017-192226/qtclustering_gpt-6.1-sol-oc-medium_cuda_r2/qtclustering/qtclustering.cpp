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
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cub/block/block_reduce.cuh>

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

static void cudaCheck(cudaError_t error) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}

template<class T> struct DeviceBuffer {
    T* data = nullptr;
    explicit DeviceBuffer(size_t count) {
        if (count > std::numeric_limits<size_t>::max() / sizeof(T)) {
            fprintf(stderr, "CUDA allocation size overflow\n");
            std::exit(EXIT_FAILURE);
        }
        cudaCheck(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)));
    }
    ~DeviceBuffer() { cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

__device__ double pointDistance(Point a, Point b) {
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    return sqrt(dx * dx + dy * dy);
}

// One warp per row builds a sorted sparse threshold-neighborhood. Every
// candidate cluster is a subset of its seed's neighborhood, including the seed.
template<bool Fill>
__global__ void neighborhoods(const Point* points, int n, double threshold,
                              int* degrees, const size_t* offsets, int* neighbors) {
    const int lane = threadIdx.x % 32;
    const int seed = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    if (seed >= n) return;
    int count = 0;
    const Point p = points[seed];
    for (long long base = 0; base < n; base += 32) {
        const long long j = base + lane;
        const bool keep = j < n && (j == seed || pointDistance(p, points[j]) < threshold);
        const unsigned mask = __ballot_sync(0xffffffffu, keep);
        if (Fill && keep) {
            const int rank = __popc(mask & ((1u << lane) - 1));
            neighbors[offsets[seed] + count + rank] = static_cast<int>(j);
        }
        count += __popc(mask);
    }
    if (!Fill && lane == 0) degrees[seed] = count;
}

struct Choice { double diameter; int slot; };
struct MinimumChoice {
    __device__ Choice operator()(Choice a, Choice b) const {
        return b.diameter < a.diameter ||
               (b.diameter == a.diameter && b.slot < a.slot) ? b : a;
    }
};

constexpr int BLOCK_SIZE = 128;

// Independent seeds run in separate blocks. A running maximum per neighbor
// avoids rescanning all previous members at every greedy insertion.
__global__ void candidateClusters(const Point* points, double threshold,
                                  const size_t* offsets, const int* neighbors,
                                  const unsigned char* clustered, double* maxima,
                                  int* members, int* sizes) {
    const int seed = blockIdx.x;
    if (clustered[seed]) { if (threadIdx.x == 0) sizes[seed] = 0; return; }
    const size_t start = offsets[seed];
    const int degree = static_cast<int>(offsets[seed + 1] - start);
    using Reduction = cub::BlockReduce<Choice, BLOCK_SIZE>;
    __shared__ typename Reduction::TempStorage reduction;
    __shared__ int dirty;
    __shared__ int selected;
    __shared__ int last;
    if (threadIdx.x == 0) dirty = sizes[seed] == 0;
    __syncthreads();
    // Removing points outside a cached greedy cluster cannot change it.
    for (int i = threadIdx.x; i < sizes[seed]; i += BLOCK_SIZE)
        if (clustered[members[start + i]]) atomicExch(&dirty, 1);
    __syncthreads();
    if (!dirty) return;

    for (int i = threadIdx.x; i < degree; i += BLOCK_SIZE) {
        const int p = neighbors[start + i];
        maxima[start + i] = (clustered[p] || p == seed) ? CUDART_INF : 0.0;
    }
    if (threadIdx.x == 0) { members[start] = seed; last = seed; }
    __syncthreads();
    int count = 1;
    while (count < degree) {
        Choice best = {CUDART_INF, INT_MAX};
        const Point latest = points[last];
        for (int i = threadIdx.x; i < degree; i += BLOCK_SIZE) {
            double diameter = maxima[start + i];
            if (diameter < threshold) {
                diameter = fmax(diameter, pointDistance(latest, points[neighbors[start + i]]));
                maxima[start + i] = diameter;
                if (diameter < threshold)
                    best = MinimumChoice{}(best, Choice{diameter, i});
            }
        }
        const Choice winner = Reduction(reduction).Reduce(best, MinimumChoice{});
        if (threadIdx.x == 0) {
            selected = winner.slot;
            if (selected != INT_MAX) {
                last = neighbors[start + selected];
                members[start + count] = last;
                maxima[start + selected] = CUDART_INF;
            }
        }
        __syncthreads();
        if (selected == INT_MAX) break;
        ++count;
    }
    if (threadIdx.x == 0) sizes[seed] = count;
}

__global__ void chooseCluster(const int* sizes, int n, unsigned long long* best) {
    using Reduction = cub::BlockReduce<unsigned long long, BLOCK_SIZE>;
    __shared__ typename Reduction::TempStorage temp;
    unsigned long long value = 0;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < static_cast<size_t>(n);
         i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        // Higher cardinality wins, then the lowest original seed index.
        const auto key = (static_cast<unsigned long long>(sizes[i]) << 32) |
                         static_cast<unsigned>(n - i);
        value = value > key ? value : key;
    }
    const auto maximum = Reduction(temp).Reduce(value, cub::Max());
    if (threadIdx.x == 0) atomicMax(best, maximum);
}

__global__ void commitCluster(const size_t* offsets, const int* members,
                              unsigned char* clustered, int seed, int count,
                              int* output) {
    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        const int p = members[offsets[seed] + i];
        clustered[p] = 1;
        output[i] = p;
    }
}

std::vector<Cluster> qtClustering(const std::vector<Point>& points, double threshold) {
    const int n = static_cast<int>(points.size());
    // A CUDA device is required; there is deliberately no sequential fallback.
    cudaCheck(cudaSetDevice(0));
    DeviceBuffer<Point> devicePoints(n);
    DeviceBuffer<int> degrees(n), sizes(n), output(n);
    DeviceBuffer<size_t> offsets(n + 1);
    DeviceBuffer<unsigned char> clustered(n);
    DeviceBuffer<unsigned long long> best(1);
    cudaCheck(cudaMemcpy(devicePoints.data, points.data(), n * sizeof(Point), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemset(clustered.data, 0, n));
    cudaCheck(cudaMemset(sizes.data, 0, n * sizeof(int)));
    const int rowBlocks = (n + 3LL) / 4;
    neighborhoods<false><<<rowBlocks, 128>>>(devicePoints.data, n, threshold,
                                            degrees.data, nullptr, nullptr);
    cudaCheck(cudaGetLastError());
    std::vector<int> hostDegrees(n);
    cudaCheck(cudaMemcpy(hostDegrees.data(), degrees.data, n * sizeof(int), cudaMemcpyDeviceToHost));
    std::vector<size_t> hostOffsets(n + 1, 0);
    for (int i = 0; i < n; ++i) hostOffsets[i + 1] = hostOffsets[i] + hostDegrees[i];
    cudaCheck(cudaMemcpy(offsets.data, hostOffsets.data(), (n + 1) * sizeof(size_t), cudaMemcpyHostToDevice));
    DeviceBuffer<int> neighbors(hostOffsets.back()), members(hostOffsets.back());
    DeviceBuffer<double> maxima(hostOffsets.back());
    neighborhoods<true><<<rowBlocks, 128>>>(devicePoints.data, n, threshold,
                                           degrees.data, offsets.data, neighbors.data);
    cudaCheck(cudaGetLastError());

    std::vector<Cluster> clusters;
    int assigned = 0;
    while (assigned < n) {
        candidateClusters<<<n, BLOCK_SIZE>>>(devicePoints.data, threshold, offsets.data,
            neighbors.data, clustered.data, maxima.data, members.data, sizes.data);
        cudaCheck(cudaGetLastError());
        cudaCheck(cudaMemset(best.data, 0, sizeof(unsigned long long)));
        chooseCluster<<<std::min(1024LL, (n + BLOCK_SIZE - 1LL) / BLOCK_SIZE), BLOCK_SIZE>>>(sizes.data, n, best.data);
        cudaCheck(cudaGetLastError());
        unsigned long long winner;
        cudaCheck(cudaMemcpy(&winner, best.data, sizeof(winner), cudaMemcpyDeviceToHost));
        const int count = static_cast<int>(winner >> 32);
        const int seed = n - static_cast<unsigned>(winner);
        commitCluster<<<1, BLOCK_SIZE>>>(offsets.data, members.data, clustered.data,
                                         seed, count, output.data + assigned);
        cudaCheck(cudaGetLastError());
        clusters.push_back(Cluster{std::vector<int>(count), seed});
        assigned += count;
    }
    std::vector<int> hostOutput(n);
    cudaCheck(cudaMemcpy(hostOutput.data(), output.data, n * sizeof(int), cudaMemcpyDeviceToHost));
    size_t pos = 0;
    for (auto& cluster : clusters) {
        std::copy_n(hostOutput.begin() + pos, cluster.members.size(), cluster.members.begin());
        pos += cluster.members.size();
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
