// QT Clustering Benchmark - Simplified Sequential Version
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.

#include <algorithm>
#include <chrono>
#include <climits>
#include <cfloat>
#include <cmath>
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

namespace {

constexpr int CUDA_BLOCK_SIZE = 256;

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error during %s: %s\n", operation,
                     cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count = 0) : count_(count) {
        if (count_ != 0) checkCuda(cudaMalloc(&data_, count_ * sizeof(T)), "device allocation");
    }
    ~DeviceBuffer() { if (data_) cudaFree(data_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* get() const { return data_; }
private:
    T* data_ = nullptr;
    size_t count_ = 0;
};

__global__ void buildDistanceMatrix(const Point* points, double* distances, int n) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(n) * n;
    if (index >= total) return;
    const int row = static_cast<int>(index / n);
    const int col = static_cast<int>(index - static_cast<size_t>(row) * n);
    const double dx = points[row].x - points[col].x;
    const double dy = points[row].y - points[col].y;
    distances[index] = sqrt(dx * dx + dy * dy);
}

// One block builds one candidate cluster.  Candidate tests in each greedy
// expansion are independent, so the block evaluates them in parallel.
__global__ void buildCandidateClusters(const double* distances,
                                       const unsigned char* clustered,
                                       const int* seeds,
                                       int* work_members,
                                       int* cardinalities,
                                       double threshold,
                                       int n) {
    const int seed = seeds[blockIdx.x];
    if (clustered[seed]) {
        if (threadIdx.x == 0) cardinalities[blockIdx.x] = -1;
        return;
    }

    __shared__ double reduction_dist[CUDA_BLOCK_SIZE];
    __shared__ int reduction_index[CUDA_BLOCK_SIZE];
    __shared__ int member_count;
    __shared__ int selected;

    int* const members = work_members + static_cast<size_t>(seed) * n;
    if (threadIdx.x == 0) {
        members[0] = seed;
        member_count = 1;
    }
    __syncthreads();

    while (member_count < n) {
        double local_best_dist = DBL_MAX;
        int local_best_index = INT_MAX;
        const int current_count = member_count;

        for (int candidate = threadIdx.x; candidate < n; candidate += blockDim.x) {
            if (clustered[candidate]) continue;
            bool already_member = false;
            double max_dist = 0.0;
            const size_t row = static_cast<size_t>(candidate) * n;
            for (int m = 0; m < current_count; ++m) {
                const int member = members[m];
                already_member |= (candidate == member);
                const double d = distances[row + member];
                max_dist = d > max_dist ? d : max_dist;
            }
            // The explicit index comparison preserves the sequential scan's
            // first-candidate tie break exactly.
            if (!already_member && max_dist < threshold &&
                (max_dist < local_best_dist ||
                 (max_dist == local_best_dist && candidate < local_best_index))) {
                local_best_dist = max_dist;
                local_best_index = candidate;
            }
        }

        reduction_dist[threadIdx.x] = local_best_dist;
        reduction_index[threadIdx.x] = local_best_index;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                const double other_dist = reduction_dist[threadIdx.x + stride];
                const int other_index = reduction_index[threadIdx.x + stride];
                if (other_dist < reduction_dist[threadIdx.x] ||
                    (other_dist == reduction_dist[threadIdx.x] &&
                     other_index < reduction_index[threadIdx.x])) {
                    reduction_dist[threadIdx.x] = other_dist;
                    reduction_index[threadIdx.x] = other_index;
                }
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) selected = reduction_index[0];
        __syncthreads();
        if (selected == INT_MAX) break;
        if (threadIdx.x == 0) members[member_count++] = selected;
        __syncthreads();
    }
    if (threadIdx.x == 0) cardinalities[blockIdx.x] = member_count;
}

__global__ void markClustered(unsigned char* clustered, const int* members, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) clustered[members[index]] = 1;
}

}  // namespace

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

// Find the closest unclustered point to the current cluster that maintains diameter < threshold
// Returns -1 if no such point exists
int findClosestPoint(const std::vector<int>& cluster_members,
                     const std::vector<bool>& clustered,
                     const std::vector<bool>& in_cluster,
                     const std::vector<Point>& points,
                     const double threshold,
                     const int point_count) {
    int closest_point = -1;
    double min_diameter = std::numeric_limits<double>::max();
    
    // Try each unclustered point as a candidate
    for (int candidate = 0; candidate < point_count; ++candidate) {
        // Skip if already clustered or already in this cluster
        if (clustered[candidate] || in_cluster[candidate]) continue;
        
        // Calculate the maximum distance from candidate to all cluster members
        double max_dist = 0.0;
        for (size_t i = 0; i < cluster_members.size(); ++i) {
            const int member = cluster_members[i];
            const double dist = distance(points[candidate], points[member]);
            max_dist = std::max(max_dist, dist);
        }
        
        // If adding this point keeps diameter below threshold and is better than current best
        if (max_dist < threshold && max_dist < min_diameter) {
            min_diameter = max_dist;
            closest_point = candidate;
        }
    }
    
    return closest_point;
}

// Generate a candidate cluster starting from a seed point
// Returns the cardinality (size) of the cluster
int generateCandidateCluster(const int seed_point,
                              const std::vector<bool>& clustered,
                              const std::vector<Point>& points,
                              const double threshold,
                              const int point_count,
                              std::vector<int>* cluster_members = nullptr) {
    std::vector<bool> in_cluster(point_count, false);
    std::vector<int> members;
    
    // Add seed point
    in_cluster[seed_point] = true;
    members.push_back(seed_point);
    
    // Iteratively add closest points
    while (static_cast<int>(members.size()) < point_count) {
        // Find closest point to current cluster that maintains diameter < threshold
        const int closest = findClosestPoint(members, clustered, in_cluster, points, 
                                             threshold, point_count);
        
        if (closest < 0) break; // No more points can be added
        
        in_cluster[closest] = true;
        members.push_back(closest);
    }
    
    // Copy members if requested
    if (cluster_members) {
        *cluster_members = members;
    }
    
    return static_cast<int>(members.size());
}

// Main QT clustering algorithm
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<Cluster> clusters;
    if (N == 0) return clusters;

    // Keep the state resident on the GPU.  Each outer QT round launches all
    // still-eligible seeds concurrently; only choosing the winning seed is
    // inherently sequential and remains on the host for exact tie semantics.
    DeviceBuffer<Point> device_points(N);
    DeviceBuffer<double> device_distances(static_cast<size_t>(N) * N);
    DeviceBuffer<unsigned char> device_clustered(N);
    DeviceBuffer<int> device_seeds(N);
    DeviceBuffer<int> device_work_members(static_cast<size_t>(N) * N);
    DeviceBuffer<int> device_cardinalities(N);
    checkCuda(cudaMemcpy(device_points.get(), points.data(), N * sizeof(Point),
                         cudaMemcpyHostToDevice), "copying points to device");
    checkCuda(cudaMemset(device_clustered.get(), 0, N * sizeof(unsigned char)),
              "initializing device cluster state");

    const size_t distance_count = static_cast<size_t>(N) * N;
    const int distance_blocks = static_cast<int>((distance_count + CUDA_BLOCK_SIZE - 1) /
                                                  CUDA_BLOCK_SIZE);
    buildDistanceMatrix<<<distance_blocks, CUDA_BLOCK_SIZE>>>(device_points.get(),
                                                                device_distances.get(), N);
    checkCuda(cudaGetLastError(), "launching distance matrix kernel");
    checkCuda(cudaDeviceSynchronize(), "building distance matrix");

    std::vector<unsigned char> clustered(N, 0);
    std::vector<int> cardinalities(N);
    std::vector<int> best_cluster_members(N);
    std::vector<int> unclustered_indices(N);
    for (int i = 0; i < N; ++i) unclustered_indices[i] = i;
    while (!unclustered_indices.empty()) {
        int max_cardinality = -1;
        int best_seed = -1;

        const int active_seeds = static_cast<int>(unclustered_indices.size());
        checkCuda(cudaMemcpy(device_seeds.get(), unclustered_indices.data(),
                             active_seeds * sizeof(int), cudaMemcpyHostToDevice),
                  "copying active seed list");
        buildCandidateClusters<<<active_seeds, CUDA_BLOCK_SIZE>>>(device_distances.get(),
                                                                    device_clustered.get(),
                                                                    device_seeds.get(),
                                                                    device_work_members.get(),
                                                                    device_cardinalities.get(),
                                                                    threshold, N);
        checkCuda(cudaGetLastError(), "launching candidate cluster kernel");
        checkCuda(cudaMemcpy(cardinalities.data(), device_cardinalities.get(),
                             active_seeds * sizeof(int), cudaMemcpyDeviceToHost),
                  "copying candidate cardinalities");

        // Ascending scan is the original unclustered_indices order.  Strict
        // comparison intentionally retains the first seed on equal sizes.
        for (int i = 0; i < active_seeds; ++i) {
            const int seed = unclustered_indices[i];
            if (cardinalities[i] > max_cardinality) {
                max_cardinality = cardinalities[i];
                best_seed = seed;
            }
        }

        if (best_seed >= 0 && max_cardinality > 0) {
            Cluster cluster;
            cluster.seed_point = best_seed;
            checkCuda(cudaMemcpy(best_cluster_members.data(),
                                 device_work_members.get() + static_cast<size_t>(best_seed) * N,
                                 max_cardinality * sizeof(int), cudaMemcpyDeviceToHost),
                      "copying winning cluster members");
            cluster.members.assign(best_cluster_members.begin(),
                                   best_cluster_members.begin() + max_cardinality);
            clusters.push_back(cluster);

            for (int member : cluster.members) {
                clustered[member] = 1;
            }
            unclustered_indices.erase(
                std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                               [&clustered](int index) { return clustered[index] != 0; }),
                unclustered_indices.end());
            const int mark_blocks = (max_cardinality + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
            markClustered<<<mark_blocks, CUDA_BLOCK_SIZE>>>(device_clustered.get(),
                                                              device_work_members.get() +
                                                                  static_cast<size_t>(best_seed) * N,
                                                              max_cardinality);
            checkCuda(cudaGetLastError(), "launching cluster state update kernel");
        } else {
            break;
        }
    }
    checkCuda(cudaDeviceSynchronize(), "completing clustering");
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
