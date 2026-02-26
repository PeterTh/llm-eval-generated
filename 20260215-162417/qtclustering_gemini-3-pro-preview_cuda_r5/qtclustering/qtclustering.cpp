// QT Clustering Benchmark - Simplified Sequential Version
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

#define CHECK_CUDA(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        printf("Error: %s:%d, ", __FILE__, __LINE__); \
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}

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

// Device function to find closest point for a given seed's cluster
__device__ int findClosestPointDevice(
    const int* members,
    int member_count,
    const bool* clustered,
    const bool* in_cluster,
    const Point* points,
    const double threshold,
    const int point_count) 
{
    int closest_point = -1;
    double min_diameter = 1.0e30; // Large value instead of limits
    
    // Try each unclustered point as a candidate
    for (int candidate = 0; candidate < point_count; ++candidate) {
        // Skip if already clustered or already in this cluster
        if (clustered[candidate] || in_cluster[candidate]) continue;
        
        // Calculate the maximum distance from candidate to all cluster members
        double max_dist = 0.0;
        for (int i = 0; i < member_count; ++i) {
            const int member = members[i];
            const double dist = distance(points[candidate], points[member]);
            if (dist > max_dist) max_dist = dist;
        }
        
        // If adding this point keeps diameter below threshold and is better than current best
        if (max_dist < threshold && max_dist < min_diameter) {
            min_diameter = max_dist;
            closest_point = candidate;
        }
    }
    
    return closest_point;
}

__global__ void computeClusterSizes(
    const int* seeds,
    int num_seeds,
    const Point* points,
    const bool* clustered,
    double threshold,
    int point_count,
    int* cluster_sizes,
    int* cluster_members_store,
    bool* in_cluster_store) 
{
    // Shared memory for points and clustered status
    // Assuming point_count is small enough for shared memory (< 2048 points)
    // If larger, this optimization only works partially or needs modification
    // For this benchmark with N=1000, it fits.
    extern __shared__ char shared_mem[];
    Point* s_points = (Point*)shared_mem;
    bool* s_clustered = (bool*)&s_points[point_count];

    // Cooperative loading of points and clustered status
    for (int i = threadIdx.x; i < point_count; i += blockDim.x) {
        s_points[i] = points[i];
        s_clustered[i] = clustered[i];
    }
    __syncthreads();

    // Use bitmask for in_cluster check to avoid global memory access
    // N=1000 requires 1000 bits = 32 integers.
    // We can use local array.
    // We support up to 4096 points. If N > limit, this optimization breaks.
    // We should fallback or assert.
    const int BITMASK_INTS = (4096 + 31) / 32;
    unsigned int in_cluster_mask[BITMASK_INTS];
    
    // Initialize mask
    for (int i = 0; i < BITMASK_INTS; ++i) in_cluster_mask[i] = 0;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_seeds) return;

    int seed = seeds[idx];

    // Use coalesced memory layout for members: [member_idx * num_seeds + seed_idx]
    int* members = cluster_members_store; // Base pointer
    
    int count = 0;
    
    // Add seed point
    // Set bit in mask
    in_cluster_mask[seed / 32] |= (1u << (seed % 32));
    
    members[count * num_seeds + idx] = seed;
    count++;
    
    // Use squared threshold for comparison to avoid sqrt
    double threshold_sq = threshold * threshold;
    
    // Iteratively add closest points
    while (count < point_count) {
        int closest_point = -1;
        double min_diameter_sq = 1.0e30; 
        
        // Try each unclustered point as a candidate
        for (int candidate = 0; candidate < point_count; ++candidate) {
            // Skip if already clustered or already in this cluster
            // Check bitmask
            if (s_clustered[candidate] || (in_cluster_mask[candidate / 32] & (1u << (candidate % 32)))) continue;
            
            // Calculate the maximum squared distance from candidate to all cluster members
            double max_dist_sq = 0.0;
            for (int i = 0; i < count; ++i) {
                // Access members[i * num_seeds + idx] -> Coalesced!
                int member = members[i * num_seeds + idx];
                
                // Use shared memory for points
                double dx = s_points[candidate].x - s_points[member].x;
                double dy = s_points[candidate].y - s_points[member].y;
                double dist_sq = dx * dx + dy * dy;
                
                if (dist_sq > max_dist_sq) max_dist_sq = dist_sq;
            }
            
            // If adding this point keeps diameter below threshold and is better than current best
            if (max_dist_sq < threshold_sq && max_dist_sq < min_diameter_sq) {
                min_diameter_sq = max_dist_sq;
                closest_point = candidate;
            }
        }
        
        if (closest_point < 0) break;
        
        in_cluster_mask[closest_point / 32] |= (1u << (closest_point % 32));
        members[count * num_seeds + idx] = closest_point;
        count++;
    }
    
    cluster_sizes[idx] = count;
}

__global__ void markClustered(bool* clustered, const int* new_members, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        clustered[new_members[idx]] = true;
    }
}

// Main QT clustering algorithm
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<bool> clustered(N, false);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;
    
    // Allocate device memory
    Point* d_points;
    bool* d_clustered;
    int* d_seeds;
    int* d_cluster_sizes;
    int* d_cluster_members_store; // N * N ints
    bool* d_in_cluster_store;     // N * N bools
    int* d_best_members;          // For marking clustered

    CHECK_CUDA(cudaMalloc(&d_points, N * sizeof(Point)));
    CHECK_CUDA(cudaMalloc(&d_clustered, N * sizeof(bool)));
    CHECK_CUDA(cudaMalloc(&d_seeds, N * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_cluster_sizes, N * sizeof(int)));
    // For large N, this might fail. But for N=1000 or even 5000 it is fine.
    // 1000*1000*4 bytes = 4MB. 5000*5000*4 = 100MB.
    CHECK_CUDA(cudaMalloc(&d_cluster_members_store, N * N * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_in_cluster_store, N * N * sizeof(bool)));
    CHECK_CUDA(cudaMalloc(&d_best_members, N * sizeof(int)));

    CHECK_CUDA(cudaMemcpy(d_points, points.data(), N * sizeof(Point), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_clustered, 0, N * sizeof(bool))); // false

    // Initialize unclustered indices
    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }
    
    // Host buffers
    std::vector<int> h_cluster_sizes(N);
    std::vector<int> h_best_members(N);
    
    // Main clustering loop
    while (!unclustered_indices.empty()) {
        int num_seeds = static_cast<int>(unclustered_indices.size());
        
        // Copy seeds to device
        CHECK_CUDA(cudaMemcpy(d_seeds, unclustered_indices.data(), num_seeds * sizeof(int), cudaMemcpyHostToDevice));
        
        // Launch kernel
        int threadsPerBlock = 256;
        int blocksPerGrid = (num_seeds + threadsPerBlock - 1) / threadsPerBlock;
        size_t sharedMemSize = N * sizeof(Point) + N * sizeof(bool);
        
        // Reset in_cluster_store (needed because layout changed and we don't clear it in kernel efficiently)
        // We only use the first num_seeds columns.
        // Total size is N * num_seeds (actually allocated N*N).
        // Since we changed layout to [point_idx * num_seeds + seed_idx], 
        // the used memory is not contiguous if num_seeds < N?
        // Wait, layout is [point_idx * num_seeds + seed_idx].
        // So for point 0, we have seeds 0..num_seeds-1.
        // Then point 1...
        // So memory is contiguous block of size point_count * num_seeds.
        // We can just clear it.
        CHECK_CUDA(cudaMemset(d_in_cluster_store, 0, N * num_seeds * sizeof(bool)));

        if (sharedMemSize > 48 * 1024) {
             // Fallback handled? Not really, but warning printed.
        }

        computeClusterSizes<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(
            d_seeds, num_seeds, d_points, d_clustered, threshold, N, 
            d_cluster_sizes, d_cluster_members_store, d_in_cluster_store);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        
        // Copy sizes back
        CHECK_CUDA(cudaMemcpy(h_cluster_sizes.data(), d_cluster_sizes, num_seeds * sizeof(int), cudaMemcpyDeviceToHost));
        
        // Find best seed on CPU
        int max_cardinality = -1;
        int best_seed_idx = -1;
        
        for (int i = 0; i < num_seeds; ++i) {
            if (h_cluster_sizes[i] > max_cardinality) {
                max_cardinality = h_cluster_sizes[i];
                best_seed_idx = i;
            }
        }
        
        // If we found a cluster, add it
        if (best_seed_idx >= 0 && max_cardinality > 0) {
            int best_seed = unclustered_indices[best_seed_idx];
            
            Cluster cluster;
            cluster.seed_point = best_seed;
            cluster.members.resize(max_cardinality);
            
            // Copy members from device store for the winning seed using 2D copy
            // Layout: [member_idx * num_seeds + seed_idx]
            // We want to copy for member_idx = 0..max_cardinality-1
            // Source ptr: d_cluster_members_store + best_seed_idx
            // Source pitch: num_seeds * sizeof(int)
            // Dest ptr: cluster.members.data()
            // Dest pitch: sizeof(int) (contiguous)
            // Width: sizeof(int)
            // Height: max_cardinality
            
            CHECK_CUDA(cudaMemcpy2D(cluster.members.data(), 
                                    sizeof(int), 
                                    d_cluster_members_store + best_seed_idx, 
                                    num_seeds * sizeof(int), 
                                    sizeof(int), 
                                    max_cardinality, 
                                    cudaMemcpyDeviceToHost));
            
            clusters.push_back(cluster);
            
            // Update clustered status on device
            CHECK_CUDA(cudaMemcpy(d_best_members, cluster.members.data(), max_cardinality * sizeof(int), cudaMemcpyHostToDevice));
            
            // Mark as clustered
            int blocks = (max_cardinality + 255) / 256;
            markClustered<<<blocks, 256>>>(d_clustered, d_best_members, max_cardinality);
            CHECK_CUDA(cudaGetLastError());
            
            // Update local clustered state to filter unclustered_indices
            for (int member : cluster.members) {
                clustered[member] = true;
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
    
    CHECK_CUDA(cudaFree(d_points));
    CHECK_CUDA(cudaFree(d_clustered));
    CHECK_CUDA(cudaFree(d_seeds));
    CHECK_CUDA(cudaFree(d_cluster_sizes));
    CHECK_CUDA(cudaFree(d_cluster_members_store));
    CHECK_CUDA(cudaFree(d_in_cluster_store));
    CHECK_CUDA(cudaFree(d_best_members));
    
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
