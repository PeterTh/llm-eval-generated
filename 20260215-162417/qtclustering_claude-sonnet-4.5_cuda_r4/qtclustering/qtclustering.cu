// QT Clustering Benchmark - CUDA Parallelized Version
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

#include "../common/results_output.hpp"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
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
    return sqrt(dx * dx + dy * dy);
}

// CUDA kernel to find the closest point to a cluster on GPU
__global__ void findClosestPointKernel(
    const int* cluster_members,
    const int cluster_size,
    const bool* clustered,
    const bool* in_cluster,
    const Point* points,
    const double threshold,
    const int point_count,
    int* closest_point,
    double* min_diameter
) {
    int candidate = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (candidate >= point_count) return;
    if (clustered[candidate] || in_cluster[candidate]) return;
    
    // Calculate the maximum distance from candidate to all cluster members
    double max_dist = 0.0;
    for (int i = 0; i < cluster_size; ++i) {
        const int member = cluster_members[i];
        const double dist = distance(points[candidate], points[member]);
        max_dist = max(max_dist, dist);
    }
    
    // If this point is valid and better than current best, try to update atomically
    if (max_dist < threshold) {
        // Use atomic compare-and-swap pattern for finding minimum
        unsigned long long* min_as_ull = (unsigned long long*)min_diameter;
        unsigned long long old = *min_as_ull;
        unsigned long long assumed;
        
        do {
            assumed = old;
            double current_min = __longlong_as_double(assumed);
            if (max_dist >= current_min) break;
            
            old = atomicCAS(min_as_ull, assumed, __double_as_longlong(max_dist));
        } while (assumed != old);
        
        // Update closest point if we set the new minimum
        if (__longlong_as_double(old) > max_dist) {
            atomicExch(closest_point, candidate);
        }
    }
}

// CUDA kernel to generate candidate clusters in parallel
__global__ void generateCandidateClustersKernel(
    const int* seed_points,
    const int num_seeds,
    const bool* clustered,
    const Point* points,
    const double threshold,
    const int point_count,
    int* cardinalities,
    int* cluster_members_all,
    const int max_cluster_size,
    bool* workspace_in_cluster,
    int* workspace_members
) {
    int seed_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (seed_idx >= num_seeds) return;
    
    int seed_point = seed_points[seed_idx];
    if (clustered[seed_point]) {
        cardinalities[seed_idx] = 0;
        return;
    }
    
    // Each thread uses its own workspace
    bool* in_cluster = &workspace_in_cluster[seed_idx * point_count];
    int* members = &workspace_members[seed_idx * point_count];
    
    for (int i = 0; i < point_count; ++i) {
        in_cluster[i] = false;
    }
    
    // Add seed point
    in_cluster[seed_point] = true;
    members[0] = seed_point;
    int member_count = 1;
    
    // Iteratively add closest points
    const int max_iterations = min(point_count, max_cluster_size);
    for (int iter = 0; iter < max_iterations; ++iter) {
        if (member_count >= max_cluster_size) break;
        
        int closest = -1;
        double min_diam = 1e100;
        
        // Try each unclustered point as a candidate
        for (int candidate = 0; candidate < point_count; ++candidate) {
            if (clustered[candidate] || in_cluster[candidate]) continue;
            
            // Calculate the maximum distance from candidate to all cluster members
            double max_dist = 0.0;
            for (int i = 0; i < member_count; ++i) {
                const int member = members[i];
                const double dist = distance(points[candidate], points[member]);
                max_dist = max(max_dist, dist);
                
                // Early exit if already exceeds threshold
                if (max_dist >= threshold) break;
            }
            
            // If adding this point keeps diameter below threshold and is better
            if (max_dist < threshold && max_dist < min_diam) {
                min_diam = max_dist;
                closest = candidate;
            }
        }
        
        if (closest < 0) break;
        
        in_cluster[closest] = true;
        members[member_count] = closest;
        member_count++;
    }
    
    // Store results
    cardinalities[seed_idx] = member_count;
    int offset = seed_idx * max_cluster_size;
    for (int i = 0; i < member_count; ++i) {
        cluster_members_all[offset + i] = members[i];
    }
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

// Main QT clustering algorithm with CUDA parallelization
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<unsigned char> clustered_host(N, 0);  // Use unsigned char instead of bool
    std::vector<Cluster> clusters;
    
    // Allocate device memory
    Point* d_points;
    bool* d_clustered;
    
    CUDA_CHECK(cudaMalloc(&d_points, N * sizeof(Point)));
    CUDA_CHECK(cudaMalloc(&d_clustered, N * sizeof(bool)));
    
    // Copy data to device (convert unsigned char to bool)
    CUDA_CHECK(cudaMemcpy(d_points, points.data(), N * sizeof(Point), cudaMemcpyHostToDevice));
    
    bool* clustered_bool = new bool[N];
    for (int i = 0; i < N; ++i) clustered_bool[i] = false;
    CUDA_CHECK(cudaMemcpy(d_clustered, clustered_bool, N * sizeof(bool), cudaMemcpyHostToDevice));
    
    std::vector<int> unclustered_indices;
    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }
    
    // Main clustering loop
    while (!unclustered_indices.empty()) {
        int num_seeds = static_cast<int>(unclustered_indices.size());
        
        // Prepare seed array
        std::vector<int> seeds;
        for (int idx : unclustered_indices) {
            if (!clustered_host[idx]) {
                seeds.push_back(idx);
            }
        }
        
        if (seeds.empty()) break;
        
        num_seeds = static_cast<int>(seeds.size());
        
        // Allocate device memory for candidate generation
        int* d_seeds;
        int* d_cardinalities;
        int* d_cluster_members_all;
        bool* d_workspace_in_cluster;
        int* d_workspace_members;
        int max_cluster_size = N;
        
        CUDA_CHECK(cudaMalloc(&d_seeds, num_seeds * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_cardinalities, num_seeds * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_cluster_members_all, num_seeds * max_cluster_size * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_workspace_in_cluster, num_seeds * N * sizeof(bool)));
        CUDA_CHECK(cudaMalloc(&d_workspace_members, num_seeds * N * sizeof(int)));
        
        // Copy seeds to device
        CUDA_CHECK(cudaMemcpy(d_seeds, seeds.data(), num_seeds * sizeof(int), cudaMemcpyHostToDevice));
        
        // Launch kernel to generate candidate clusters in parallel
        // Use smaller block size to allow more concurrent blocks
        int blockSize = 64;
        int numBlocks = (num_seeds + blockSize - 1) / blockSize;
        
        generateCandidateClustersKernel<<<numBlocks, blockSize>>>(
            d_seeds, num_seeds, d_clustered, d_points, threshold, N,
            d_cardinalities, d_cluster_members_all, max_cluster_size,
            d_workspace_in_cluster, d_workspace_members
        );
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Copy results back
        std::vector<int> cardinalities(num_seeds);
        std::vector<int> cluster_members_all(num_seeds * max_cluster_size);
        
        CUDA_CHECK(cudaMemcpy(cardinalities.data(), d_cardinalities, 
                             num_seeds * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(cluster_members_all.data(), d_cluster_members_all,
                             num_seeds * max_cluster_size * sizeof(int), cudaMemcpyDeviceToHost));
        
        // Find best cluster on CPU
        int max_cardinality = -1;
        int best_seed_idx = -1;
        
        for (int i = 0; i < num_seeds; ++i) {
            if (cardinalities[i] > max_cardinality) {
                max_cardinality = cardinalities[i];
                best_seed_idx = i;
            }
        }
        
        // Free device memory for this iteration
        CUDA_CHECK(cudaFree(d_seeds));
        CUDA_CHECK(cudaFree(d_cardinalities));
        CUDA_CHECK(cudaFree(d_cluster_members_all));
        CUDA_CHECK(cudaFree(d_workspace_in_cluster));
        CUDA_CHECK(cudaFree(d_workspace_members));
        
        // If we found a cluster, add it
        if (best_seed_idx >= 0 && max_cardinality > 0) {
            Cluster cluster;
            cluster.seed_point = seeds[best_seed_idx];
            
            int offset = best_seed_idx * max_cluster_size;
            for (int i = 0; i < max_cardinality; ++i) {
                cluster.members.push_back(cluster_members_all[offset + i]);
            }
            
            clusters.push_back(cluster);
            
            // Mark all members as clustered
            for (int member : cluster.members) {
                clustered_host[member] = 1;
                clustered_bool[member] = true;
            }
            
            // Update device clustered array
            CUDA_CHECK(cudaMemcpy(d_clustered, clustered_bool, 
                                 N * sizeof(bool), cudaMemcpyHostToDevice));
            
            // Remove clustered points from unclustered list
            unclustered_indices.erase(
                std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                              [&clustered_host](int idx) { return clustered_host[idx]; }),
                unclustered_indices.end()
            );
        } else {
            break;
        }
    }
    
    // Free device memory
    CUDA_CHECK(cudaFree(d_points));
    CUDA_CHECK(cudaFree(d_clustered));
    delete[] clustered_bool;
    
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
