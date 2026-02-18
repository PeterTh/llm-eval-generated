// QT Clustering Benchmark - CUDA Parallel Version
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
// Parallelized with CUDA for GPU acceleration.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while (0)

static const double MAX_WIDTH = 20.0;
static const double MAX_HEIGHT = 20.0;
static const int THREADS_PER_BLOCK = 256;

// Structure to represent a point in 2D space
struct Point {
    double x, y;
};

// Structure to represent a cluster
struct Cluster {
    std::vector<int> members;
    int seed_point;
};

// CUDA kernel: Evaluate cardinality of candidate clusters from multiple seeds
// Each seed is evaluated in parallel
__global__ void kernel_evaluateSeeds(const Point* d_points,
                                     const int* d_seeds,
                                     const int num_seeds,
                                     const unsigned char* d_clustered,
                                     const double threshold,
                                     const int point_count,
                                     int* d_cardinalities) {
    int seed_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (seed_idx >= num_seeds) return;
    
    int seed = d_seeds[seed_idx];
    if (d_clustered[seed]) {
        d_cardinalities[seed_idx] = 0;
        return;
    }
    
    int cardinality = 1;
    for (int candidate = 0; candidate < point_count; ++candidate) {
        if (d_clustered[candidate] || candidate == seed) continue;
        
        // Check if this candidate can be in the cluster
        const Point& p_seed = d_points[seed];
        const Point& p_cand = d_points[candidate];
        double dist = sqrt((p_seed.x - p_cand.x) * (p_seed.x - p_cand.x) +
                          (p_seed.y - p_cand.y) * (p_seed.y - p_cand.y));
        
        if (dist < threshold) {
            cardinality++;
        }
    }
    
    d_cardinalities[seed_idx] = cardinality;
}

// CUDA kernel: Find closest point that maintains diameter constraint
// Each candidate is evaluated in parallel
__global__ void kernel_findClosestPoint(const Point* d_points,
                                        const int* d_cluster_members,
                                        const int cluster_size,
                                        const unsigned char* d_clustered,
                                        const unsigned char* d_in_cluster,
                                        const double threshold,
                                        const int point_count,
                                        int* d_closest_point,
                                        double* d_min_diameter) {
    int candidate = blockIdx.x * blockDim.x + threadIdx.x;
    if (candidate >= point_count) return;
    
    // Skip if already clustered or already in this cluster
    if (d_clustered[candidate] || d_in_cluster[candidate]) {
        d_closest_point[candidate] = -1;
        d_min_diameter[candidate] = 1e18;
        return;
    }
    
    // Calculate maximum distance from candidate to all cluster members
    double max_dist = 0.0;
    for (int i = 0; i < cluster_size; ++i) {
        const int member = d_cluster_members[i];
        const Point& p1 = d_points[candidate];
        const Point& p2 = d_points[member];
        double dx = p1.x - p2.x;
        double dy = p1.y - p2.y;
        double dist = sqrt(dx * dx + dy * dy);
        max_dist = fmax(max_dist, dist);
    }
    
    // Check if this point is valid
    if (max_dist < threshold) {
        d_closest_point[candidate] = candidate;
        d_min_diameter[candidate] = max_dist;
    } else {
        d_closest_point[candidate] = -1;
        d_min_diameter[candidate] = 1e18;
    }
}

// GPU state management
struct GPUState {
    Point* d_points;
    unsigned char* d_clustered;
    unsigned char* d_in_cluster;
    int* d_cluster_members;
    int* d_closest_candidates;
    double* d_max_distances;
    int* d_seeds;
    int* d_cardinalities;
    int point_count;
    
    void allocate(int N) {
        point_count = N;
        CUDA_CHECK(cudaMalloc(&d_points, N * sizeof(Point)));
        CUDA_CHECK(cudaMalloc(&d_clustered, N * sizeof(unsigned char)));
        CUDA_CHECK(cudaMalloc(&d_in_cluster, N * sizeof(unsigned char)));
        CUDA_CHECK(cudaMalloc(&d_cluster_members, N * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_closest_candidates, N * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_max_distances, N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_seeds, N * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_cardinalities, N * sizeof(int)));
    }
    
    void free() {
        cudaFree(d_points);
        cudaFree(d_clustered);
        cudaFree(d_in_cluster);
        cudaFree(d_cluster_members);
        cudaFree(d_closest_candidates);
        cudaFree(d_max_distances);
        cudaFree(d_seeds);
        cudaFree(d_cardinalities);
    }
    
    void copyPointsToGPU(const std::vector<Point>& points) {
        CUDA_CHECK(cudaMemcpy(d_points, points.data(), points.size() * sizeof(Point),
                             cudaMemcpyHostToDevice));
    }
    
    void copyClusteredToGPU(const std::vector<bool>& clustered) {
        std::vector<unsigned char> bool_data;
        for (bool b : clustered) {
            bool_data.push_back(b ? 1 : 0);
        }
        CUDA_CHECK(cudaMemcpy(d_clustered, bool_data.data(), bool_data.size(),
                             cudaMemcpyHostToDevice));
    }
} g_gpu_state;

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
// Uses GPU for parallel distance computation
int findClosestPoint(const std::vector<int>& cluster_members,
                     const std::vector<bool>& clustered,
                     const std::vector<bool>& in_cluster,
                     const std::vector<Point>& points,
                     const double threshold,
                     const int point_count) {
    int closest_point = -1;
    double min_diameter = std::numeric_limits<double>::max();
    
    // Copy clustered state to GPU (must be updated each time)
    std::vector<unsigned char> clustered_data;
    for (bool b : clustered) {
        clustered_data.push_back(b ? 1 : 0);
    }
    CUDA_CHECK(cudaMemcpy(g_gpu_state.d_clustered, clustered_data.data(),
                         clustered_data.size(), cudaMemcpyHostToDevice));
    
    // Copy cluster state to GPU
    std::vector<int> cluster_members_padded = cluster_members;
    CUDA_CHECK(cudaMemcpy(g_gpu_state.d_cluster_members, cluster_members_padded.data(),
                         cluster_members.size() * sizeof(int), cudaMemcpyHostToDevice));
    
    std::vector<unsigned char> in_cluster_data;
    for (bool b : in_cluster) {
        in_cluster_data.push_back(b ? 1 : 0);
    }
    CUDA_CHECK(cudaMemcpy(g_gpu_state.d_in_cluster, in_cluster_data.data(),
                         in_cluster.size(), cudaMemcpyHostToDevice));
    
    // Prepare output arrays
    std::vector<int> h_closest_candidates(point_count);
    std::vector<double> h_max_distances(point_count);
    
    // Launch GPU kernel to find closest point for each candidate
    int blocks = (point_count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    kernel_findClosestPoint<<<blocks, THREADS_PER_BLOCK>>>(
        g_gpu_state.d_points,
        g_gpu_state.d_cluster_members,
        static_cast<int>(cluster_members.size()),
        g_gpu_state.d_clustered,
        g_gpu_state.d_in_cluster,
        threshold,
        point_count,
        g_gpu_state.d_closest_candidates,
        g_gpu_state.d_max_distances
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy results back from GPU
    CUDA_CHECK(cudaMemcpy(h_closest_candidates.data(), g_gpu_state.d_closest_candidates,
                         point_count * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_max_distances.data(), g_gpu_state.d_max_distances,
                         point_count * sizeof(double), cudaMemcpyDeviceToHost));
    
    // Find best candidate from GPU results
    for (int i = 0; i < point_count; ++i) {
        if (h_closest_candidates[i] >= 0 && h_max_distances[i] < min_diameter) {
            min_diameter = h_max_distances[i];
            closest_point = h_closest_candidates[i];
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

// Main QT clustering algorithm - GPU accelerated
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<bool> clustered(N, false);
    std::vector<Cluster> clusters;
    
    // Main clustering loop
    while (true) {
        // Get unclustered indices
        std::vector<int> unclustered_seeds;
        for (int i = 0; i < N; ++i) {
            if (!clustered[i]) {
                unclustered_seeds.push_back(i);
            }
        }
        
        if (unclustered_seeds.empty()) break;
        
        // Copy clustered state to GPU
        std::vector<unsigned char> clustered_data;
        for (bool b : clustered) {
            clustered_data.push_back(b ? 1 : 0);
        }
        CUDA_CHECK(cudaMemcpy(g_gpu_state.d_clustered, clustered_data.data(),
                             clustered_data.size(), cudaMemcpyHostToDevice));
        
        // Copy seeds to GPU
        CUDA_CHECK(cudaMemcpy(g_gpu_state.d_seeds, unclustered_seeds.data(),
                             unclustered_seeds.size() * sizeof(int), cudaMemcpyHostToDevice));
        
        // Evaluate all unclustered seeds in parallel
        int num_seeds = static_cast<int>(unclustered_seeds.size());
        int blocks = (num_seeds + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        kernel_evaluateSeeds<<<blocks, THREADS_PER_BLOCK>>>(
            g_gpu_state.d_points,
            g_gpu_state.d_seeds,
            num_seeds,
            g_gpu_state.d_clustered,
            threshold,
            N,
            g_gpu_state.d_cardinalities
        );
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Find best seed
        std::vector<int> h_cardinalities(num_seeds);
        CUDA_CHECK(cudaMemcpy(h_cardinalities.data(), g_gpu_state.d_cardinalities,
                             num_seeds * sizeof(int), cudaMemcpyDeviceToHost));
        
        int best_seed_idx = -1;
        int max_cardinality = 0;
        for (int i = 0; i < num_seeds; ++i) {
            if (h_cardinalities[i] > max_cardinality) {
                max_cardinality = h_cardinalities[i];
                best_seed_idx = i;
            }
        }
        
        if (best_seed_idx < 0) break;
        
        // Build cluster from best seed
        int best_seed = unclustered_seeds[best_seed_idx];
        std::vector<int> cluster_members;
        cluster_members.push_back(best_seed);
        std::vector<bool> in_cluster(N, false);
        in_cluster[best_seed] = true;
        
        // Iteratively add closest points using GPU
        while (static_cast<int>(cluster_members.size()) < N) {
            const int closest = findClosestPoint(cluster_members, clustered, in_cluster,
                                               points, threshold, N);
            
            if (closest < 0) break;
            
            in_cluster[closest] = true;
            cluster_members.push_back(closest);
        }
        
        // Add cluster and mark members as clustered
        Cluster cluster;
        cluster.seed_point = best_seed;
        cluster.members = cluster_members;
        clusters.push_back(cluster);
        
        for (int member : cluster_members) {
            clustered[member] = true;
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
    
    printf("QT Clustering Benchmark (CUDA)\n");
    printf("Number of points: %d\n", num_points);
    printf("Distance threshold: %.2f\n", threshold);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    // Initialize GPU
    g_gpu_state.allocate(num_points);
    
    // Generate synthetic data
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);
    
    // Copy points to GPU
    g_gpu_state.copyPointsToGPU(points);
    
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
    bool valid = true;
    if (validate) {
        valid = validateClusters(clusters, points, threshold);
        
        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
        }
    }
    
    // Cleanup GPU resources
    g_gpu_state.free();
    
    return valid ? 0 : 1;
}
