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

#include "../common/results_output.hpp"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

static const double MAX_WIDTH = 20.0;
static const double MAX_HEIGHT = 20.0;
static const int BLOCK_SIZE = 256;

// Structure to represent a point in 2D space
struct Point {
    double x, y;
};

// Structure to represent a cluster
struct Cluster {
    std::vector<int> members;
    int seed_point;
};

// CUDA kernel: Calculate Euclidean distance
__device__ inline double cuda_distance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return sqrt(dx * dx + dy * dy);
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

// CUDA kernel: Find closest point to cluster (parallel candidate evaluation)
__global__ void findClosestPointKernel(const int* cluster_members,
                                       const int cluster_size,
                                       const bool* clustered,
                                       const bool* in_cluster,
                                       const Point* points,
                                       const double threshold,
                                       const int point_count,
                                       int* result_point,
                                       double* result_diameter) {
    int candidate = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (candidate >= point_count) return;
    if (clustered[candidate] || in_cluster[candidate]) return;
    
    // Calculate max distance from candidate to all cluster members
    double max_dist = 0.0;
    for (int i = 0; i < cluster_size; ++i) {
        int member = cluster_members[i];
        double dist = cuda_distance(points[candidate], points[member]);
        max_dist = max(max_dist, dist);
    }
    
    // If valid, try to update the result using atomic operations
    if (max_dist < threshold) {
        // Use atomic compare-and-swap to find minimum diameter
        unsigned long long* addr = (unsigned long long*)result_diameter;
        unsigned long long old = *addr;
        unsigned long long assumed;
        
        do {
            assumed = old;
            double current_min = __longlong_as_double(assumed);
            if (max_dist >= current_min) break;
            
            old = atomicCAS(addr, assumed, __double_as_longlong(max_dist));
        } while (assumed != old);
        
        // If we set the new minimum, also update the result point
        if (__longlong_as_double(old) > max_dist) {
            atomicExch(result_point, candidate);
        }
    }
}

// CUDA kernel: Generate candidate clusters for multiple seeds in parallel (optimized)
__global__ void generateCandidateClustersKernel(const int* seeds,
                                                 const int num_seeds,
                                                 const bool* clustered,
                                                 const Point* points,
                                                 const double threshold,
                                                 const int point_count,
                                                 int* cardinalities,
                                                 int* cluster_members_buffer,
                                                 const int max_cluster_size,
                                                 bool* in_cluster_buffer) {
    int seed_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (seed_idx >= num_seeds) return;
    
    int seed_point = seeds[seed_idx];
    if (clustered[seed_point]) {
        cardinalities[seed_idx] = 0;
        return;
    }
    
    // Use pre-allocated buffer for in_cluster flags
    int* local_members = cluster_members_buffer + seed_idx * max_cluster_size;
    bool* local_in_cluster = in_cluster_buffer + seed_idx * point_count;
    
    // Initialize in_cluster flags
    for (int i = 0; i < point_count; ++i) {
        local_in_cluster[i] = false;
    }
    
    // Add seed point
    local_in_cluster[seed_point] = true;
    local_members[0] = seed_point;
    int members_count = 1;
    
    // Iteratively add closest points
    while (members_count < point_count && members_count < max_cluster_size) {
        int closest = -1;
        double min_diameter = 1e100;
        
        // Find closest point
        for (int candidate = 0; candidate < point_count; ++candidate) {
            if (clustered[candidate] || local_in_cluster[candidate]) continue;
            
            double max_dist = 0.0;
            for (int i = 0; i < members_count; ++i) {
                int member = local_members[i];
                double dist = cuda_distance(points[candidate], points[member]);
                max_dist = max(max_dist, dist);
            }
            
            if (max_dist < threshold && max_dist < min_diameter) {
                min_diameter = max_dist;
                closest = candidate;
            }
        }
        
        if (closest < 0) break;
        
        local_in_cluster[closest] = true;
        local_members[members_count] = closest;
        members_count++;
    }
    
    cardinalities[seed_idx] = members_count;
}

// Calculate Euclidean distance between two points (CPU version)
inline double distance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

// Find the closest unclustered point to the current cluster (GPU-accelerated)
int findClosestPoint(const std::vector<int>& cluster_members,
                     const bool* d_clustered,
                     const bool* d_in_cluster,
                     const Point* d_points,
                     const double threshold,
                     const int point_count) {
    // Allocate device memory for cluster members
    int* d_cluster_members;
    int cluster_size = static_cast<int>(cluster_members.size());
    CUDA_CHECK(cudaMalloc(&d_cluster_members, cluster_size * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_cluster_members, cluster_members.data(), 
                         cluster_size * sizeof(int), cudaMemcpyHostToDevice));
    
    // Allocate result on device
    int* d_result_point;
    double* d_result_diameter;
    CUDA_CHECK(cudaMalloc(&d_result_point, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_result_diameter, sizeof(double)));
    
    int init_point = -1;
    double init_diameter = std::numeric_limits<double>::max();
    CUDA_CHECK(cudaMemcpy(d_result_point, &init_point, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_result_diameter, &init_diameter, sizeof(double), cudaMemcpyHostToDevice));
    
    // Launch kernel
    int numBlocks = (point_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    findClosestPointKernel<<<numBlocks, BLOCK_SIZE>>>(
        d_cluster_members, cluster_size, d_clustered, d_in_cluster,
        d_points, threshold, point_count, d_result_point, d_result_diameter);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy result back
    int result_point;
    CUDA_CHECK(cudaMemcpy(&result_point, d_result_point, sizeof(int), cudaMemcpyDeviceToHost));
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_cluster_members));
    CUDA_CHECK(cudaFree(d_result_point));
    CUDA_CHECK(cudaFree(d_result_diameter));
    
    return result_point;
}

// Generate a candidate cluster starting from a seed point (GPU-accelerated)
int generateCandidateCluster(const int seed_point,
                              const std::vector<bool>& clustered,
                              const Point* d_points,
                              const double threshold,
                              const int point_count,
                              std::vector<int>* cluster_members = nullptr) {
    // Convert std::vector<bool> to regular bool array
    bool* h_clustered = new bool[point_count];
    for (int i = 0; i < point_count; ++i) {
        h_clustered[i] = clustered[i];
    }
    
    // Upload clustered array to device
    bool* d_clustered;
    bool* d_in_cluster;
    CUDA_CHECK(cudaMalloc(&d_clustered, point_count * sizeof(bool)));
    CUDA_CHECK(cudaMalloc(&d_in_cluster, point_count * sizeof(bool)));
    CUDA_CHECK(cudaMemcpy(d_clustered, h_clustered, 
                         point_count * sizeof(bool), cudaMemcpyHostToDevice));
    
    std::vector<bool> in_cluster(point_count, false);
    bool* h_in_cluster = new bool[point_count];
    std::vector<int> members;
    
    // Add seed point
    in_cluster[seed_point] = true;
    members.push_back(seed_point);
    
    // Iteratively add closest points
    while (static_cast<int>(members.size()) < point_count) {
        // Convert in_cluster to regular array
        for (int i = 0; i < point_count; ++i) {
            h_in_cluster[i] = in_cluster[i];
        }
        
        // Upload current in_cluster state
        CUDA_CHECK(cudaMemcpy(d_in_cluster, h_in_cluster, 
                             point_count * sizeof(bool), cudaMemcpyHostToDevice));
        
        // Find closest point using GPU
        const int closest = findClosestPoint(members, d_clustered, d_in_cluster,
                                            d_points, threshold, point_count);
        
        if (closest < 0) break;
        
        in_cluster[closest] = true;
        members.push_back(closest);
    }
    
    // Copy members if requested
    if (cluster_members) {
        *cluster_members = members;
    }
    
    // Cleanup
    delete[] h_clustered;
    delete[] h_in_cluster;
    CUDA_CHECK(cudaFree(d_clustered));
    CUDA_CHECK(cudaFree(d_in_cluster));
    
    return static_cast<int>(members.size());
}

// Main QT clustering algorithm with CUDA acceleration
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<bool> clustered(N, false);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;
    
    // Initialize unclustered indices
    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }
    
    // Transfer points to GPU
    Point* d_points;
    CUDA_CHECK(cudaMalloc(&d_points, N * sizeof(Point)));
    CUDA_CHECK(cudaMemcpy(d_points, points.data(), N * sizeof(Point), cudaMemcpyHostToDevice));
    
    // Allocate device memory for batch processing
    bool* h_clustered = new bool[N];
    bool* d_clustered;
    CUDA_CHECK(cudaMalloc(&d_clustered, N * sizeof(bool)));
    
    // Main clustering loop
    while (!unclustered_indices.empty()) {
        // Prepare seeds for parallel processing
        std::vector<int> seeds;
        for (size_t i = 0; i < unclustered_indices.size(); ++i) {
            const int seed = unclustered_indices[i];
            if (!clustered[seed]) {
                seeds.push_back(seed);
            }
        }
        
        if (seeds.empty()) break;
        
        // Process seeds in batches to manage memory
        const int max_batch_size = std::min(512, static_cast<int>(seeds.size()));
        int max_cardinality_global = -1;
        int best_seed_global = -1;
        std::vector<int> best_cluster_members_global;
        
        for (size_t batch_start = 0; batch_start < seeds.size(); batch_start += max_batch_size) {
            size_t batch_end = std::min(batch_start + max_batch_size, seeds.size());
            int num_seeds = static_cast<int>(batch_end - batch_start);
            const int max_cluster_size = N;
            
            // Upload current clustered status
            for (int i = 0; i < N; ++i) {
                h_clustered[i] = clustered[i];
            }
            CUDA_CHECK(cudaMemcpy(d_clustered, h_clustered, N * sizeof(bool), cudaMemcpyHostToDevice));
            
            // Allocate device memory for this batch
            int* d_seeds;
            int* d_cardinalities;
            int* d_cluster_members_buffer;
            bool* d_in_cluster_buffer;
            
            CUDA_CHECK(cudaMalloc(&d_seeds, num_seeds * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_cardinalities, num_seeds * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_cluster_members_buffer, num_seeds * max_cluster_size * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_in_cluster_buffer, (size_t)num_seeds * N * sizeof(bool)));
            
            // Copy seeds for this batch
            std::vector<int> batch_seeds(seeds.begin() + batch_start, seeds.begin() + batch_end);
            CUDA_CHECK(cudaMemcpy(d_seeds, batch_seeds.data(), num_seeds * sizeof(int), cudaMemcpyHostToDevice));
            
            // Launch kernel to generate all candidate clusters in parallel
            int threadsPerBlock = 1;
            int numBlocks = num_seeds;
            
            generateCandidateClustersKernel<<<numBlocks, threadsPerBlock>>>(
                d_seeds, num_seeds, d_clustered, d_points, threshold, N,
                d_cardinalities, d_cluster_members_buffer, max_cluster_size, d_in_cluster_buffer);
            CUDA_CHECK(cudaDeviceSynchronize());
            
            // Copy results back to host
            std::vector<int> cardinalities(num_seeds);
            CUDA_CHECK(cudaMemcpy(cardinalities.data(), d_cardinalities, 
                                 num_seeds * sizeof(int), cudaMemcpyDeviceToHost));
            
            // Find best cluster in this batch
            int max_cardinality = -1;
            int best_seed_idx = -1;
            for (int i = 0; i < num_seeds; ++i) {
                if (cardinalities[i] > max_cardinality) {
                    max_cardinality = cardinalities[i];
                    best_seed_idx = i;
                }
            }
            
            if (best_seed_idx >= 0 && max_cardinality > max_cardinality_global) {
                // Copy back the best cluster members
                std::vector<int> best_cluster_members(max_cardinality);
                CUDA_CHECK(cudaMemcpy(best_cluster_members.data(),
                                     d_cluster_members_buffer + best_seed_idx * max_cluster_size,
                                     max_cardinality * sizeof(int), cudaMemcpyDeviceToHost));
                
                max_cardinality_global = max_cardinality;
                best_seed_global = batch_seeds[best_seed_idx];
                best_cluster_members_global = best_cluster_members;
            }
            
            // Cleanup batch memory
            CUDA_CHECK(cudaFree(d_seeds));
            CUDA_CHECK(cudaFree(d_cardinalities));
            CUDA_CHECK(cudaFree(d_cluster_members_buffer));
            CUDA_CHECK(cudaFree(d_in_cluster_buffer));
        }
        
        if (best_seed_global >= 0 && max_cardinality_global > 0) {
            Cluster cluster;
            cluster.seed_point = best_seed_global;
            cluster.members = best_cluster_members_global;
            clusters.push_back(cluster);
            
            // Mark all members as clustered
            for (size_t i = 0; i < best_cluster_members_global.size(); ++i) {
                clustered[best_cluster_members_global[i]] = true;
            }
            
            // Remove clustered points from unclustered list
            unclustered_indices.erase(
                std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                              [&clustered](int idx) { return clustered[idx]; }),
                unclustered_indices.end()
            );
        } else {
            break;
        }
    }
    
    // Cleanup GPU memory
    delete[] h_clustered;
    CUDA_CHECK(cudaFree(d_clustered));
    CUDA_CHECK(cudaFree(d_points));
    
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
