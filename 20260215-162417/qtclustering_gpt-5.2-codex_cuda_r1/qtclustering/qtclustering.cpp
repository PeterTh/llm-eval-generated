// QT Clustering Benchmark - Simplified Sequential Version
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

inline void checkCuda(cudaError_t result, const char* func, const char* file, int line) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d (%s): %s\n",
                file, line, func, cudaGetErrorString(result));
        std::exit(1);
    }
}

#define CUDA_CHECK(val) checkCuda((val), #val, __FILE__, __LINE__)

constexpr int kCudaBlockSize = 256;

__global__ void computeMaxDistances(const Point* points,
                                    const int* cluster_members,
                                    int cluster_size,
                                    const unsigned char* clustered,
                                    const unsigned char* in_cluster,
                                    double* max_dist_sq,
                                    double threshold_sq,
                                    int point_count) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= point_count) return;
    if (clustered[idx] || in_cluster[idx]) {
        max_dist_sq[idx] = DBL_MAX;
        return;
    }
    const Point p = points[idx];
    double max_val = 0.0;
    for (int i = 0; i < cluster_size; ++i) {
        const int member = cluster_members[i];
        const Point m = points[member];
        const double dx = p.x - m.x;
        const double dy = p.y - m.y;
        const double dist_sq = dx * dx + dy * dy;
        if (dist_sq > max_val) {
            max_val = dist_sq;
            if (max_val >= threshold_sq) break;
        }
    }
    max_dist_sq[idx] = max_val;
}

struct GpuWorkspace {
    explicit GpuWorkspace(int count, const std::vector<Point>& points)
        : point_count(count),
          host_max_dist(count),
          host_clustered(count, 0) {
        CUDA_CHECK(cudaMalloc(&device_points, point_count * sizeof(Point)));
        CUDA_CHECK(cudaMalloc(&device_clustered, point_count * sizeof(unsigned char)));
        CUDA_CHECK(cudaMalloc(&device_in_cluster, point_count * sizeof(unsigned char)));
        CUDA_CHECK(cudaMalloc(&device_members, point_count * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&device_max_dist, point_count * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(device_points, points.data(),
                              point_count * sizeof(Point),
                              cudaMemcpyHostToDevice));
    }

    ~GpuWorkspace() {
        if (device_points) cudaFree(device_points);
        if (device_clustered) cudaFree(device_clustered);
        if (device_in_cluster) cudaFree(device_in_cluster);
        if (device_members) cudaFree(device_members);
        if (device_max_dist) cudaFree(device_max_dist);
    }

    GpuWorkspace(const GpuWorkspace&) = delete;
    GpuWorkspace& operator=(const GpuWorkspace&) = delete;

    void updateClustered(const std::vector<bool>& clustered) {
        for (int i = 0; i < point_count; ++i) {
            host_clustered[i] = clustered[i] ? 1 : 0;
        }
        CUDA_CHECK(cudaMemcpy(device_clustered, host_clustered.data(),
                              point_count * sizeof(unsigned char),
                              cudaMemcpyHostToDevice));
    }

    void resetInCluster() {
        CUDA_CHECK(cudaMemset(device_in_cluster, 0, point_count * sizeof(unsigned char)));
    }

    void setInCluster(int index) {
        const unsigned char one = 1;
        CUDA_CHECK(cudaMemcpy(device_in_cluster + index, &one,
                              sizeof(unsigned char), cudaMemcpyHostToDevice));
    }

    void setMember(int position, int value) {
        CUDA_CHECK(cudaMemcpy(device_members + position, &value,
                              sizeof(int), cudaMemcpyHostToDevice));
    }

    int point_count;
    Point* device_points = nullptr;
    unsigned char* device_clustered = nullptr;
    unsigned char* device_in_cluster = nullptr;
    int* device_members = nullptr;
    double* device_max_dist = nullptr;
    std::vector<double> host_max_dist;
    std::vector<unsigned char> host_clustered;
};

int findClosestPointCuda(GpuWorkspace& workspace,
                         int cluster_size,
                         double threshold_sq) {
    const int blocks = (workspace.point_count + kCudaBlockSize - 1) / kCudaBlockSize;
    computeMaxDistances<<<blocks, kCudaBlockSize>>>(
        workspace.device_points,
        workspace.device_members,
        cluster_size,
        workspace.device_clustered,
        workspace.device_in_cluster,
        workspace.device_max_dist,
        threshold_sq,
        workspace.point_count);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(workspace.host_max_dist.data(), workspace.device_max_dist,
                          workspace.point_count * sizeof(double),
                          cudaMemcpyDeviceToHost));

    int closest_point = -1;
    double min_dist_sq = std::numeric_limits<double>::max();
    for (int i = 0; i < workspace.point_count; ++i) {
        const double dist_sq = workspace.host_max_dist[i];
        if (dist_sq < threshold_sq && dist_sq < min_dist_sq) {
            min_dist_sq = dist_sq;
            closest_point = i;
        }
    }
    return closest_point;
}
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

// Generate a candidate cluster starting from a seed point
// Returns the cardinality (size) of the cluster
int generateCandidateCluster(const int seed_point,
                              const double threshold,
                              GpuWorkspace& workspace,
                              std::vector<int>* cluster_members = nullptr) {
    std::vector<int> members;
    members.reserve(workspace.point_count);
    
    // Add seed point
    members.push_back(seed_point);
    workspace.resetInCluster();
    workspace.setInCluster(seed_point);
    workspace.setMember(0, seed_point);
    
    // Iteratively add closest points
    const double threshold_sq = threshold * threshold;
    while (static_cast<int>(members.size()) < workspace.point_count) {
        // Find closest point to current cluster that maintains diameter < threshold
        const int closest = findClosestPointCuda(workspace,
                                                 static_cast<int>(members.size()),
                                                 threshold_sq);
        
        if (closest < 0) break; // No more points can be added
        
        members.push_back(closest);
        workspace.setInCluster(closest);
        workspace.setMember(static_cast<int>(members.size()) - 1, closest);
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
    std::vector<bool> clustered(N, false);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;

    GpuWorkspace workspace(N, points);
    workspace.updateClustered(clustered);
    
    // Initialize unclustered indices
    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }
    
    // Main clustering loop
    while (!unclustered_indices.empty()) {
        int max_cardinality = -1;
        int best_seed = -1;
        std::vector<int> best_cluster_members;
        
        // Try each unclustered point as a seed
        for (size_t i = 0; i < unclustered_indices.size(); ++i) {
            const int seed = unclustered_indices[i];
            if (clustered[seed]) continue;
            
            std::vector<int> candidate_members;
            const int cardinality = generateCandidateCluster(seed, threshold,
                                                             workspace,
                                                             &candidate_members);
            
            if (cardinality > max_cardinality) {
                max_cardinality = cardinality;
                best_seed = seed;
                best_cluster_members = candidate_members;
            }
        }
        
        // If we found a cluster, add it
        if (best_seed >= 0 && max_cardinality > 0) {
            Cluster cluster;
            cluster.seed_point = best_seed;
            cluster.members = best_cluster_members;
            clusters.push_back(cluster);
            
            // Mark all members as clustered
            for (size_t i = 0; i < best_cluster_members.size(); ++i) {
                clustered[best_cluster_members[i]] = true;
            }
            
            // Remove clustered points from unclustered list
            unclustered_indices.erase(
                std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                              [&clustered](int idx) { return clustered[idx]; }),
                unclustered_indices.end()
            );

            workspace.updateClustered(clustered);
        } else {
            // No more clusters can be formed
            break;
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

void initializeCuda() {
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count <= 0) {
        fprintf(stderr, "Error: No CUDA devices available.\n");
        std::exit(1);
    }
    CUDA_CHECK(cudaSetDevice(0));
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

    initializeCuda();
    
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
