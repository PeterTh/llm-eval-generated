// QT Clustering Benchmark - CUDA Version
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
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

static void cudaCheck(cudaError_t error) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}

template<class T> struct DeviceBuffer {
    T* data = nullptr;
    explicit DeviceBuffer(size_t count) {
        cudaCheck(cudaMalloc(reinterpret_cast<void**>(&data), std::max(size_t(1), count) * sizeof(T)));
    }
    ~DeviceBuffer() { cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

constexpr int BLOCK_SIZE = 128;

__device__ double pointDistance(Point a, Point b) {
    const double dx = a.x - b.x, dy = a.y - b.y;
    return sqrt(dx * dx + dy * dy);
}

// Only threshold-neighbors can ever enter a seed's candidate. Construct a
// compact graph in two passes, avoiding a dense N-by-N distance matrix.
template<bool Fill>
__global__ void buildNeighbors(const Point* points, int n, double threshold,
                               int* counts, const size_t* offsets, int* neighbors) {
    const int seed = blockIdx.x;
    const int lane = threadIdx.x & 31;
    __shared__ int total;
    __shared__ int warpBase[BLOCK_SIZE / 32];
    if (threadIdx.x == 0) total = 0;
    __syncthreads();
    const Point origin = points[seed];
    for (int base = 0; base < n; base += BLOCK_SIZE) {
        const int j = base + threadIdx.x;
        const bool take = j < n && (j == seed || pointDistance(origin, points[j]) < threshold);
        const unsigned mask = __ballot_sync(0xffffffffu, take);
        if (lane == 0) warpBase[threadIdx.x / 32] = atomicAdd(&total, __popc(mask));
        __syncwarp();
        if (Fill && take) {
            const int pos = warpBase[threadIdx.x / 32] + __popc(mask & ((1u << lane) - 1));
            neighbors[offsets[seed] + pos] = j;
        }
        // Do not overwrite warpBase before every lane has consumed it.
        __syncwarp();
    }
    __syncthreads();
    if (threadIdx.x == 0) counts[seed] = total;
}

struct Choice { double value; int index; };

__device__ Choice better(Choice a, Choice b) {
    return b.value < a.value || (b.value == a.value && b.index < a.index) ? b : a;
}

__device__ Choice blockMinimum(Choice value, Choice* shared) {
    const int lane = threadIdx.x & 31;
    for (int d = 16; d; d >>= 1) {
        Choice other{__shfl_down_sync(0xffffffffu, value.value, d),
                     __shfl_down_sync(0xffffffffu, value.index, d)};
        value = better(value, other);
    }
    if (lane == 0) shared[threadIdx.x / 32] = value;
    __syncthreads();
    if (threadIdx.x == 0) {
        Choice result = shared[0];
        for (int i = 1; i < BLOCK_SIZE / 32; ++i) result = better(result, shared[i]);
        shared[0] = result;
    }
    __syncthreads();
    return shared[0];
}

// A block owns a seed; threads update independent candidate diameters. The
// maximum distance is maintained incrementally instead of rescanning members.
__global__ void growCandidates(const Point* points, int n, double threshold,
                                const size_t* offsets, const int* neighbors,
                                const int* clustered, double* diameters,
                                int* members, int* cardinalities) {
    const int seed = blockIdx.x;
    if (seed >= n) return;
    if (clustered[seed]) {
        if (threadIdx.x == 0) cardinalities[seed] = 0;
        return;
    }
    const size_t start = offsets[seed], degree = offsets[seed + 1] - start;
    __shared__ int dirty;
    __shared__ Choice reduction[BLOCK_SIZE / 32];
    if (threadIdx.x == 0) dirty = cardinalities[seed] == 0;
    __syncthreads();
    // Removal cannot change the greedy sequence unless it removes one of the
    // previously chosen members. Reuse all unaffected candidates exactly.
    for (int i = threadIdx.x; i < cardinalities[seed]; i += BLOCK_SIZE)
        if (clustered[members[start + i]]) atomicExch(&dirty, 1);
    __syncthreads();
    if (!dirty) return;
    for (size_t i = threadIdx.x; i < degree; i += BLOCK_SIZE)
        diameters[start + i] = 0.0;
    if (threadIdx.x == 0) members[start] = seed;
    __syncthreads();
    int last = seed, count = 1;
    while (true) {
        Choice local{CUDART_INF, INT_MAX};
        const Point added = points[last];
        for (size_t i = threadIdx.x; i < degree; i += BLOCK_SIZE) {
            const int candidate = neighbors[start + i];
            double maximum = diameters[start + i];
            if (clustered[candidate] || candidate == last) maximum = CUDART_INF;
            if (maximum < threshold) {
                maximum = fmax(maximum, pointDistance(points[candidate], added));
                if (maximum < threshold) local = better(local, Choice{maximum, candidate});
            }
            diameters[start + i] = maximum;
        }
        const Choice best = blockMinimum(local, reduction);
        if (best.index == INT_MAX) break;
        last = best.index;
        if (threadIdx.x == 0) members[start + count] = last;
        ++count;
        // The reduction storage is reused on the next iteration.
        __syncthreads();
    }
    if (threadIdx.x == 0) cardinalities[seed] = count;
}

__global__ void selectCluster(int n, const size_t* offsets, const int* members,
                              const int* cardinalities, int* clustered, int* output) {
    __shared__ Choice reduction[BLOCK_SIZE / 32];
    Choice local{0.0, INT_MAX};
    for (int i = threadIdx.x; i < n; i += BLOCK_SIZE)
        if (cardinalities[i] > 0)
            local = better(local, Choice{-double(cardinalities[i]), i});
    const Choice best = blockMinimum(local, reduction);
    const int seed = best.index, count = -int(best.value);
    if (threadIdx.x == 0) { output[0] = seed; output[1] = count; }
    for (int i = threadIdx.x; i < count; i += BLOCK_SIZE) {
        const int member = members[offsets[seed] + i];
        output[2 + i] = member;
        clustered[member] = 1;
    }
}

std::vector<Cluster> qtClustering(const std::vector<Point>& points, const double threshold) {
    const int n = static_cast<int>(points.size());
    std::vector<Cluster> clusters;
    if (n == 0) return clusters;
    DeviceBuffer<Point> devicePoints(n);
    DeviceBuffer<int> counts(n), clustered(n), cardinalities(n), output(size_t(n) + 2);
    DeviceBuffer<size_t> offsets(size_t(n) + 1);
    cudaCheck(cudaMemcpy(devicePoints.data, points.data(), n * sizeof(Point), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemset(clustered.data, 0, n * sizeof(int)));
    cudaCheck(cudaMemset(cardinalities.data, 0, n * sizeof(int)));
    buildNeighbors<false><<<n, BLOCK_SIZE>>>(devicePoints.data, n, threshold, counts.data, nullptr, nullptr);
    cudaCheck(cudaGetLastError());
    std::vector<int> hostCounts(n);
    cudaCheck(cudaMemcpy(hostCounts.data(), counts.data, n * sizeof(int), cudaMemcpyDeviceToHost));
    std::vector<size_t> hostOffsets(size_t(n) + 1, 0);
    for (int i = 0; i < n; ++i) hostOffsets[i + 1] = hostOffsets[i] + hostCounts[i];
    const size_t entries = hostOffsets[n];
    DeviceBuffer<int> neighbors(entries), members(entries);
    DeviceBuffer<double> diameters(entries);
    cudaCheck(cudaMemcpy(offsets.data, hostOffsets.data(), hostOffsets.size() * sizeof(size_t), cudaMemcpyHostToDevice));
    buildNeighbors<true><<<n, BLOCK_SIZE>>>(devicePoints.data, n, threshold, counts.data, offsets.data, neighbors.data);
    cudaCheck(cudaGetLastError());
    int remaining = n;
    while (remaining > 0) {
        growCandidates<<<n, BLOCK_SIZE>>>(devicePoints.data, n, threshold, offsets.data,
            neighbors.data, clustered.data, diameters.data, members.data, cardinalities.data);
        cudaCheck(cudaGetLastError());
        selectCluster<<<1, BLOCK_SIZE>>>(n, offsets.data, members.data, cardinalities.data, clustered.data, output.data);
        cudaCheck(cudaGetLastError());
        int header[2];
        cudaCheck(cudaMemcpy(header, output.data, sizeof(header), cudaMemcpyDeviceToHost));
        Cluster cluster;
        cluster.seed_point = header[0];
        cluster.members.resize(header[1]);
        cudaCheck(cudaMemcpy(cluster.members.data(), output.data + 2, header[1] * sizeof(int), cudaMemcpyDeviceToHost));
        remaining -= header[1];
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

    // Initialize the required CUDA runtime before timing the clustering work.
    cudaCheck(cudaFree(nullptr));
    
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
