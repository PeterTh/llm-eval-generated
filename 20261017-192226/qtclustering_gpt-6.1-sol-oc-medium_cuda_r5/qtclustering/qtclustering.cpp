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
#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>

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

static void cudaCheck(cudaError_t status) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

template<class T> struct DeviceBuffer {
    T* data = nullptr;
    explicit DeviceBuffer(size_t count) {
        if (count > SIZE_MAX / sizeof(T)) {
            fprintf(stderr, "CUDA allocation size overflow\n");
            std::exit(EXIT_FAILURE);
        }
        cudaCheck(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)));
    }
    ~DeviceBuffer() { cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

constexpr int BLOCK_SIZE = 128;

__device__ double gpuDistance(Point a, Point b) {
    const double dx = a.x - b.x, dy = a.y - b.y;
    return sqrt(dx * dx + dy * dy);
}

// Only seed-neighbors can ever enter a candidate. Store these in CSR format,
// including the seed itself, to keep storage proportional to local density.
__global__ void countNeighbors(const Point* points, int n, double threshold, int* counts) {
    using Reduce = cub::BlockReduce<int, BLOCK_SIZE>;
    __shared__ typename Reduce::TempStorage storage;
    const int seed = blockIdx.x;
    int count = 0;
    for (int j = threadIdx.x; j < n; j += blockDim.x)
        count += (j == seed || gpuDistance(points[seed], points[j]) < threshold);
    count = Reduce(storage).Sum(count);
    if (threadIdx.x == 0) counts[seed] = count;
}

__global__ void fillNeighbors(const Point* points, int n, double threshold,
                              const size_t* offsets, int* neighbors) {
    using Scan = cub::BlockScan<int, BLOCK_SIZE>;
    __shared__ typename Scan::TempStorage storage;
    __shared__ int written;
    const int seed = blockIdx.x;
    if (threadIdx.x == 0) written = 0;
    __syncthreads();
    for (size_t base = 0; base < static_cast<size_t>(n); base += blockDim.x) {
        const size_t j = base + threadIdx.x;
        const int keep = j < static_cast<size_t>(n) &&
            (j == static_cast<size_t>(seed) || gpuDistance(points[seed], points[j]) < threshold);
        int position, total;
        Scan(storage).ExclusiveSum(keep, position, total);
        if (keep) neighbors[offsets[seed] + written + position] = static_cast<int>(j);
        __syncthreads();
        if (threadIdx.x == 0) written += total;
        __syncthreads();
    }
}

struct Choice { double diameter; int point; };
struct ChooseClosest {
    __device__ Choice operator()(Choice a, Choice b) const {
        return (b.diameter < a.diameter ||
                (b.diameter == a.diameter && b.point < a.point)) ? b : a;
    }
};

// One cooperative block per seed; all seeds grow independently in parallel.
// Maintain each candidate's maximum distance incrementally instead of rescanning
// all previous cluster members. Reductions explicitly break ties by point ID.
__global__ void growCandidates(const Point* points, double threshold,
                               const size_t* offsets, const int* neighbors,
                               const unsigned char* removed, double* maxima,
                               int* members, int* sizes, unsigned long long* best) {
    const int seed = blockIdx.x;
    if (removed[seed]) return;
    const size_t start = offsets[seed], end = offsets[seed + 1];
    using IntReduce = cub::BlockReduce<int, BLOCK_SIZE>;
    using MinReduce = cub::BlockReduce<Choice, BLOCK_SIZE>;
    __shared__ typename IntReduce::TempStorage intStorage;
    __shared__ typename MinReduce::TempStorage minStorage;
    __shared__ int rebuild, last, size;
    const int oldSize = sizes[seed];
    int invalid = oldSize == 0;
    // Removing points that were never selected cannot alter the greedy path.
    for (int i = threadIdx.x; i < oldSize; i += blockDim.x)
        invalid |= removed[members[start + i]];
    invalid = IntReduce(intStorage).Sum(invalid);
    if (threadIdx.x == 0) rebuild = invalid;
    __syncthreads();
    if (rebuild) {
        for (size_t i = start + threadIdx.x; i < end; i += blockDim.x) {
            const int p = neighbors[i];
            maxima[i] = (removed[p] || p == seed) ? DBL_MAX : 0.0;
        }
        if (threadIdx.x == 0) {
            last = seed;
            size = 1;
            members[start] = seed;
        }
        __syncthreads();
        while (true) {
            const int newest = last;
            Choice closest = {DBL_MAX, INT_MAX};
            for (size_t i = start + threadIdx.x; i < end; i += blockDim.x) {
                double d = maxima[i];
                const int p = neighbors[i];
                if (p == newest) d = DBL_MAX;
                if (d != DBL_MAX) {
                    d = fmax(d, gpuDistance(points[p], points[newest]));
                    // A point exceeding the threshold can never become valid again.
                    if (!(d < threshold)) d = DBL_MAX;
                }
                maxima[i] = d;
                if (d < DBL_MAX) closest = ChooseClosest()(closest, Choice{d, p});
            }
            closest = MinReduce(minStorage).Reduce(closest, ChooseClosest());
            if (threadIdx.x == 0) {
                last = closest.point;
                if (last != INT_MAX) members[start + size++] = last;
            }
            __syncthreads();
            if (last == INT_MAX) break;
        }
        if (threadIdx.x == 0) sizes[seed] = size;
    }
    if (threadIdx.x == 0) {
        // Largest cardinality, then smallest seed, exactly as in the serial loop.
        const unsigned long long key =
            (static_cast<unsigned long long>(sizes[seed]) << 32) |
            (UINT_MAX - static_cast<unsigned int>(seed));
        atomicMax(best, key);
    }
}

__global__ void removeCluster(const int* members, size_t start, int count,
                              unsigned char* removed) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += blockDim.x * gridDim.x) removed[members[start + i]] = 1;
}

std::vector<Cluster> qtClustering(const std::vector<Point>& points, double threshold) {
    const int n = static_cast<int>(points.size());
    std::vector<Cluster> clusters;
    if (n == 0) return clusters;
    DeviceBuffer<Point> dPoints(n);
    DeviceBuffer<int> dCounts(n), sizes(n);
    DeviceBuffer<size_t> dOffsets(static_cast<size_t>(n) + 1);
    DeviceBuffer<unsigned char> removed(n);
    DeviceBuffer<unsigned long long> best(1);
    cudaCheck(cudaMemcpy(dPoints.data, points.data(), n * sizeof(Point), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemset(removed.data, 0, n));
    cudaCheck(cudaMemset(sizes.data, 0, n * sizeof(int)));
    countNeighbors<<<n, BLOCK_SIZE>>>(dPoints.data, n, threshold, dCounts.data);
    cudaCheck(cudaGetLastError());
    std::vector<int> counts(n);
    cudaCheck(cudaMemcpy(counts.data(), dCounts.data, n * sizeof(int), cudaMemcpyDeviceToHost));
    std::vector<size_t> offsets(static_cast<size_t>(n) + 1, 0);
    for (int i = 0; i < n; ++i) offsets[i + 1] = offsets[i] + counts[i];
    cudaCheck(cudaMemcpy(dOffsets.data, offsets.data(), offsets.size() * sizeof(size_t), cudaMemcpyHostToDevice));
    DeviceBuffer<int> neighbors(offsets.back()), members(offsets.back());
    DeviceBuffer<double> maxima(offsets.back());
    fillNeighbors<<<n, BLOCK_SIZE>>>(dPoints.data, n, threshold, dOffsets.data, neighbors.data);
    cudaCheck(cudaGetLastError());
    int remaining = n;
    while (remaining > 0) {
        cudaCheck(cudaMemset(best.data, 0, sizeof(unsigned long long)));
        growCandidates<<<n, BLOCK_SIZE>>>(dPoints.data, threshold, dOffsets.data,
            neighbors.data, removed.data, maxima.data, members.data, sizes.data, best.data);
        cudaCheck(cudaGetLastError());
        unsigned long long winner;
        cudaCheck(cudaMemcpy(&winner, best.data, sizeof(winner), cudaMemcpyDeviceToHost));
        const int count = static_cast<int>(winner >> 32);
        const int seed = static_cast<int>(UINT_MAX - static_cast<unsigned int>(winner));
        Cluster cluster;
        cluster.seed_point = seed;
        cluster.members.resize(count);
        cudaCheck(cudaMemcpy(cluster.members.data(), members.data + offsets[seed],
                             count * sizeof(int), cudaMemcpyDeviceToHost));
        clusters.push_back(std::move(cluster));
        removeCluster<<<(count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            members.data, offsets[seed], count, removed.data);
        cudaCheck(cudaGetLastError());
        remaining -= count;
    }
    cudaCheck(cudaDeviceSynchronize());
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
