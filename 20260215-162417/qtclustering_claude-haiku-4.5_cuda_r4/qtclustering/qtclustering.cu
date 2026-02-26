// QT Clustering Benchmark - CUDA Parallel Version
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

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "../common/results_output.hpp"

// CUDA error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error at line %d: %s\n", __LINE__, cudaGetErrorString(error)); \
        exit(1); \
    } \
} while(0)

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

// GPU device structures
__device__ inline double d_distance(double x1, double y1, double x2, double y2) {
    double dx = x1 - x2;
    double dy = y1 - y2;
    return sqrt(dx * dx + dy * dy);
}

// Kernel: Calculate max distances from each candidate to all cluster members
// Each thread processes one candidate point
__global__ void compute_distances_kernel(
    const double* d_points_x, const double* d_points_y,
    const int* d_cluster_members, const int cluster_size,
    const unsigned char* d_clustered, const unsigned char* d_in_cluster,
    double* d_max_distances, unsigned char* d_valid_candidates,
    int point_count, double threshold)
{
    int candidate = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (candidate >= point_count) return;
    
    // Skip if already clustered or in this cluster
    if (d_clustered[candidate] || d_in_cluster[candidate]) {
        d_valid_candidates[candidate] = 0;
        return;
    }
    
    // Calculate max distance to cluster members
    double max_dist = 0.0;
    for (int i = 0; i < cluster_size; ++i) {
        int member = d_cluster_members[i];
        double dist = d_distance(d_points_x[candidate], d_points_y[candidate],
                                 d_points_x[member], d_points_y[member]);
        max_dist = fmax(max_dist, dist);
    }
    
    // Mark as valid if distance is within threshold
    if (max_dist < threshold) {
        d_max_distances[candidate] = max_dist;
        d_valid_candidates[candidate] = 1;
    } else {
        d_valid_candidates[candidate] = 0;
    }
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

// Find the closest unclustered point using CUDA parallelization
int findClosestPoint_GPU(const std::vector<int>& cluster_members,
                         const std::vector<bool>& clustered,
                         const std::vector<bool>& in_cluster,
                         const std::vector<Point>& points,
                         const double threshold,
                         const int point_count,
                         double* d_points_x, double* d_points_y,
                         int* d_cluster_members, unsigned char* d_clustered, unsigned char* d_in_cluster,
                         double* d_max_distances, unsigned char* d_valid_candidates) {
    int closest_point = -1;
    double min_diameter = std::numeric_limits<double>::max();
    
    // Copy cluster members to GPU
    CUDA_CHECK(cudaMemcpy(d_cluster_members, cluster_members.data(),
                          cluster_members.size() * sizeof(int), cudaMemcpyHostToDevice));
    
    // Convert bool vectors to unsigned char for GPU transfer
    std::vector<unsigned char> clustered_uc(clustered.size());
    std::vector<unsigned char> in_cluster_uc(in_cluster.size());
    for (size_t i = 0; i < clustered.size(); ++i) {
        clustered_uc[i] = clustered[i] ? 1 : 0;
        in_cluster_uc[i] = in_cluster[i] ? 1 : 0;
    }
    
    // Copy flags to GPU
    CUDA_CHECK(cudaMemcpy(d_clustered, clustered_uc.data(),
                          point_count * sizeof(unsigned char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_in_cluster, in_cluster_uc.data(),
                          point_count * sizeof(unsigned char), cudaMemcpyHostToDevice));
    
    // Launch kernel to compute distances
    int threads_per_block = 256;
    int blocks = (point_count + threads_per_block - 1) / threads_per_block;
    
    compute_distances_kernel<<<blocks, threads_per_block>>>(
        d_points_x, d_points_y, d_cluster_members, cluster_members.size(),
        d_clustered, d_in_cluster, d_max_distances, d_valid_candidates,
        point_count, threshold);
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy results back
    std::vector<double> h_max_distances(point_count);
    std::vector<unsigned char> h_valid_candidates(point_count);
    
    CUDA_CHECK(cudaMemcpy(h_max_distances.data(), d_max_distances,
                          point_count * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_valid_candidates.data(), d_valid_candidates,
                          point_count * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    
    // Find best candidate on CPU
    for (int candidate = 0; candidate < point_count; ++candidate) {
        if (h_valid_candidates[candidate] && h_max_distances[candidate] < min_diameter) {
            min_diameter = h_max_distances[candidate];
            closest_point = candidate;
        }
    }
    
    return closest_point;
}

// Generate a candidate cluster starting from a seed point using GPU
// Returns the cardinality (size) of the cluster
int generateCandidateCluster(const int seed_point,
                              const std::vector<bool>& clustered,
                              const std::vector<Point>& points,
                              const double threshold,
                              const int point_count,
                              double* d_points_x, double* d_points_y,
                              int* d_cluster_members, unsigned char* d_clustered, unsigned char* d_in_cluster,
                              double* d_max_distances, unsigned char* d_valid_candidates,
                              std::vector<int>* cluster_members = nullptr) {
    std::vector<bool> in_cluster(point_count, false);
    std::vector<int> members;
    
    // Add seed point
    in_cluster[seed_point] = true;
    members.push_back(seed_point);
    
    // Iteratively add closest points
    while (static_cast<int>(members.size()) < point_count) {
        // Find closest point to current cluster that maintains diameter < threshold
        const int closest = findClosestPoint_GPU(members, clustered, in_cluster, points, 
                                                 threshold, point_count,
                                                 d_points_x, d_points_y, d_cluster_members,
                                                 d_clustered, d_in_cluster,
                                                 d_max_distances, d_valid_candidates);
        
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

// Main QT clustering algorithm using GPU acceleration
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<bool> clustered(N, false);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;
    
    // Prepare point data for GPU transfer
    std::vector<double> points_x(N), points_y(N);
    for (int i = 0; i < N; ++i) {
        points_x[i] = points[i].x;
        points_y[i] = points[i].y;
    }
    
    // Allocate GPU memory
    double *d_points_x, *d_points_y;
    int *d_cluster_members;
    unsigned char *d_clustered, *d_in_cluster;
    double *d_max_distances;
    unsigned char *d_valid_candidates;
    
    CUDA_CHECK(cudaMalloc(&d_points_x, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_points_y, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cluster_members, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_clustered, N * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_in_cluster, N * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_max_distances, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_valid_candidates, N * sizeof(unsigned char)));
    
    // Copy point data to GPU
    CUDA_CHECK(cudaMemcpy(d_points_x, points_x.data(), N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_points_y, points_y.data(), N * sizeof(double), cudaMemcpyHostToDevice));
    
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
            const int cardinality = generateCandidateCluster(seed, clustered, points, 
                                                       threshold, N,
                                                       d_points_x, d_points_y, d_cluster_members,
                                                       d_clustered, d_in_cluster,
                                                       d_max_distances, d_valid_candidates,
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
        } else {
            // No more clusters can be formed
            break;
        }
    }
    
    // Free GPU memory
    CUDA_CHECK(cudaFree(d_points_x));
    CUDA_CHECK(cudaFree(d_points_y));
    CUDA_CHECK(cudaFree(d_cluster_members));
    CUDA_CHECK(cudaFree(d_clustered));
    CUDA_CHECK(cudaFree(d_in_cluster));
    CUDA_CHECK(cudaFree(d_max_distances));
    CUDA_CHECK(cudaFree(d_valid_candidates));
    
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
