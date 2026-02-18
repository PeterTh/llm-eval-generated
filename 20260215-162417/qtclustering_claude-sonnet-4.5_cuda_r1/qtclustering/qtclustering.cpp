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

// CUDA kernel to evaluate all candidate seeds in parallel and build their clusters
// Each block handles one seed, threads collaborate to build that seed's cluster
__global__ void evaluateSeedsParallel(
    const Point* points,
    const bool* clustered,
    const int* seeds,
    int num_seeds,
    int point_count,
    double threshold,
    int* cardinalities,
    int* cluster_members_flat,  // Flattened array of all cluster members
    int max_cluster_size) {
    
    int seed_idx = blockIdx.x;
    if (seed_idx >= num_seeds) return;
    
    int seed = seeds[seed_idx];
    if (clustered[seed]) {
        cardinalities[seed_idx] = 0;
        return;
    }
    
    // Use shared memory for this seed's cluster
    extern __shared__ int shared_data[];
    int* members = shared_data;  // Current cluster members
    bool* in_cluster = (bool*)&members[max_cluster_size];  // Cluster membership flags
    
    // Initialize
    for (int i = threadIdx.x; i < point_count; i += blockDim.x) {
        in_cluster[i] = false;
    }
    __syncthreads();
    
    // Add seed point
    if (threadIdx.x == 0) {
        in_cluster[seed] = true;
        members[0] = seed;
    }
    __syncthreads();
    
    int cluster_size = 1;
    
    // Iteratively add closest points
    for (int iter = 0; iter < point_count && cluster_size < point_count; ++iter) {
        int best_candidate = -1;
        
        // Each thread evaluates a subset of candidates
        int thread_best = -1;
        double thread_min_dist = 1e100;
        
        for (int candidate = threadIdx.x; candidate < point_count; candidate += blockDim.x) {
            if (clustered[candidate] || in_cluster[candidate]) continue;
            
            // Calculate max distance from candidate to all current members
            double max_dist = 0.0;
            for (int i = 0; i < cluster_size; ++i) {
                int member = members[i];
                double dist = distance(points[candidate], points[member]);
                if (dist > max_dist) max_dist = dist;
            }
            
            // Check if this is a valid candidate and better than current best
            if (max_dist < threshold && max_dist < thread_min_dist) {
                thread_min_dist = max_dist;
                thread_best = candidate;
            }
        }
        
        // Use shared memory for reduction
        __shared__ int s_candidates[256];
        __shared__ double s_distances[256];
        
        s_candidates[threadIdx.x] = thread_best;
        s_distances[threadIdx.x] = thread_min_dist;
        __syncthreads();
        
        // Reduction to find global best
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                if (s_candidates[threadIdx.x + s] >= 0 &&
                    (s_candidates[threadIdx.x] < 0 || 
                     s_distances[threadIdx.x + s] < s_distances[threadIdx.x])) {
                    s_candidates[threadIdx.x] = s_candidates[threadIdx.x + s];
                    s_distances[threadIdx.x] = s_distances[threadIdx.x + s];
                }
            }
            __syncthreads();
        }
        
        best_candidate = s_candidates[0];
        
        if (best_candidate < 0) break;  // No more valid candidates
        
        // Add best candidate to cluster
        if (threadIdx.x == 0) {
            in_cluster[best_candidate] = true;
            members[cluster_size] = best_candidate;
        }
        __syncthreads();
        
        cluster_size++;
    }
    
    // Write results
    if (threadIdx.x == 0) {
        cardinalities[seed_idx] = cluster_size;
        // Copy members to global memory
        int offset = seed_idx * max_cluster_size;
        for (int i = 0; i < cluster_size; ++i) {
            cluster_members_flat[offset + i] = members[i];
        }
    }
}

// Main QT clustering algorithm with optimized CUDA acceleration
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<bool> clustered(N, false);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;
    
    // Allocate GPU memory
    Point* d_points;
    bool* d_clustered;
    int* d_seeds;
    int* d_cardinalities;
    int* d_cluster_members_flat;
    
    // Use a reasonable max cluster size to limit memory
    const int max_cluster_size = std::min(N, 256);  // Limit for memory efficiency
    
    CUDA_CHECK(cudaMalloc(&d_points, N * sizeof(Point)));
    CUDA_CHECK(cudaMalloc(&d_clustered, N * sizeof(bool)));
    CUDA_CHECK(cudaMalloc(&d_seeds, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cardinalities, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cluster_members_flat, N * max_cluster_size * sizeof(int)));
    
    // Copy points to device
    CUDA_CHECK(cudaMemcpy(d_points, points.data(), N * sizeof(Point), cudaMemcpyHostToDevice));
    
    // Initialize unclustered indices
    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }
    
    // Get device properties to check shared memory limits
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    size_t max_shared_mem = prop.sharedMemPerBlock;
    
    // Main clustering loop
    while (!unclustered_indices.empty()) {
        int num_unclustered = unclustered_indices.size();
        
        // Copy clustered status to device
        bool* h_clustered = new bool[N];
        for (int i = 0; i < N; ++i) {
            h_clustered[i] = clustered[i];
        }
        CUDA_CHECK(cudaMemcpy(d_clustered, h_clustered, N * sizeof(bool), cudaMemcpyHostToDevice));
        delete[] h_clustered;
        
        // Copy unclustered seeds to device
        CUDA_CHECK(cudaMemcpy(d_seeds, unclustered_indices.data(), 
                             num_unclustered * sizeof(int), cudaMemcpyHostToDevice));
        
        // Calculate required shared memory
        size_t required_shared_mem = (max_cluster_size * sizeof(int)) + (N * sizeof(bool));
        
        // Launch kernel to evaluate all seeds in parallel
        int blockSize = 256;
        int gridSize = num_unclustered;
        
        // Check if we exceed shared memory limits
        if (required_shared_mem > max_shared_mem) {
            // Use a smaller block size or reduce max_cluster_size for this iteration
            required_shared_mem = max_shared_mem - 1024;  // Leave some headroom
        }
        
        evaluateSeedsParallel<<<gridSize, blockSize, required_shared_mem>>>(
            d_points, d_clustered, d_seeds, num_unclustered, N, threshold,
            d_cardinalities, d_cluster_members_flat, max_cluster_size);
        
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Copy results back
        std::vector<int> cardinalities(num_unclustered);
        CUDA_CHECK(cudaMemcpy(cardinalities.data(), d_cardinalities, 
                             num_unclustered * sizeof(int), cudaMemcpyDeviceToHost));
        
        // Find best cluster on CPU
        int max_cardinality = -1;
        int best_seed_idx = -1;
        
        for (int i = 0; i < num_unclustered; ++i) {
            if (cardinalities[i] > max_cardinality) {
                max_cardinality = cardinalities[i];
                best_seed_idx = i;
            }
        }
        
        // If we found a cluster, add it
        if (best_seed_idx >= 0 && max_cardinality > 0) {
            // Copy cluster members from device
            std::vector<int> cluster_members(max_cardinality);
            int offset = best_seed_idx * max_cluster_size;
            CUDA_CHECK(cudaMemcpy(cluster_members.data(), 
                                 d_cluster_members_flat + offset,
                                 max_cardinality * sizeof(int), 
                                 cudaMemcpyDeviceToHost));
            
            Cluster cluster;
            cluster.seed_point = unclustered_indices[best_seed_idx];
            cluster.members = cluster_members;
            clusters.push_back(cluster);
            
            // Mark all members as clustered
            for (int member : cluster_members) {
                clustered[member] = true;
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
    
    // Free GPU memory
    CUDA_CHECK(cudaFree(d_points));
    CUDA_CHECK(cudaFree(d_clustered));
    CUDA_CHECK(cudaFree(d_seeds));
    CUDA_CHECK(cudaFree(d_cardinalities));
    CUDA_CHECK(cudaFree(d_cluster_members_flat));
    
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
