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

// Calculate Squared Euclidean distance between two points
__host__ __device__ inline double distance_sq(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return dx * dx + dy * dy;
}

// Calculate Euclidean distance between two points
__host__ __device__ inline double distance(const Point& p1, const Point& p2) {
    return sqrt(distance_sq(p1, p2));
}

// CUDA Kernel to calculate cluster size for each seed
__global__ void generate_candidate_clusters_kernel(
    const Point* points,
    const bool* clustered,
    int* cardinalities,
    float* distances_sq,  // Scratchpad: N * N (Stores squared distances)
    bool* in_cluster,  // Scratchpad: N * N
    int N,
    double threshold_sq // Squared threshold
) {
    int seed = blockIdx.x * blockDim.x + threadIdx.x;
    if (seed >= N) return;

    // If seed is already clustered, return 0 cardinality
    if (clustered[seed]) {
        cardinalities[seed] = 0;
        return;
    }

    // Pointers to local scratch memory in global memory
    float* my_distances_sq = distances_sq + seed * N;
    bool* my_in_cluster = in_cluster + seed * N;

    // Initialize state
    for (int i = 0; i < N; ++i) {
        my_distances_sq[i] = 0.0f; // Initial max distance to empty cluster is 0
        my_in_cluster[i] = false;
    }

    // Add seed point
    my_in_cluster[seed] = true;
    int current_cardinality = 1;
    
    // Track the last added point to update distances
    int last_added = seed;

    // Iteratively add closest points
    while (current_cardinality < N) {
        int best_candidate = -1;
        float min_max_dist_sq = 1e30f; // Infinity

        // Update distances based on last added point and find best candidate
        for (int i = 0; i < N; ++i) {
            if (clustered[i] || my_in_cluster[i]) continue;

            // Update max_dist for candidate i
            // new_max_dist = max(old_max_dist, dist(i, last_added))
            // We use squared distances
            float dist_sq = distance_sq(points[i], points[last_added]);
            
            float current_dist = my_distances_sq[i];
            
            if (dist_sq > current_dist) {
                current_dist = dist_sq;
                my_distances_sq[i] = current_dist;
            }

            // Check if valid and best
            if (current_dist < threshold_sq) {
                if (current_dist < min_max_dist_sq) {
                    min_max_dist_sq = current_dist;
                    best_candidate = i;
                }
            }
        }

        if (best_candidate != -1) {
            my_in_cluster[best_candidate] = true;
            current_cardinality++;
            last_added = best_candidate;
        } else {
            break; // No more points can be added
        }
    }

    cardinalities[seed] = current_cardinality;
}

// Helper to regenerate the best cluster on CPU (to avoid storing members on GPU)
// Uses the optimized O(N^2) logic (O(N) per point added) instead of original O(N^3)
void regenerateBestCluster(int seed, const std::vector<bool>& clustered, 
                           const std::vector<Point>& points, 
                           double threshold, 
                           std::vector<int>& members) {
    int N = points.size();
    std::vector<bool> in_cluster(N, false);
    std::vector<double> current_max_dists(N, 0.0);
    
    members.clear();
    members.push_back(seed);
    in_cluster[seed] = true;
    
    int last_added = seed;
    
    while (static_cast<int>(members.size()) < N) {
        int best_candidate = -1;
        double min_diameter = std::numeric_limits<double>::max();
        
        for (int i = 0; i < N; ++i) {
            if (clustered[i] || in_cluster[i]) continue;
            
            double d = distance(points[i], points[last_added]);
            if (d > current_max_dists[i]) {
                current_max_dists[i] = d;
            }
            
            if (current_max_dists[i] < threshold) {
                if (current_max_dists[i] < min_diameter) {
                    min_diameter = current_max_dists[i];
                    best_candidate = i;
                }
            }
        }
        
        if (best_candidate != -1) {
            in_cluster[best_candidate] = true;
            members.push_back(best_candidate);
            last_added = best_candidate;
        } else {
            break;
        }
    }
}

// Main QT clustering algorithm
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<bool> clustered(N, false);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;
    
    // Initialize unclustered indices (on CPU, though we might not strictly need this vector anymore if we iterate N)
    int unclustered_count = N;

    // CUDA Memory Allocation
    Point* d_points;
    bool* d_clustered;
    int* d_cardinalities;
    float* d_distances;
    bool* d_in_cluster;

    CUDA_CHECK(cudaMalloc(&d_points, N * sizeof(Point)));
    CUDA_CHECK(cudaMalloc(&d_clustered, N * sizeof(bool)));
    CUDA_CHECK(cudaMalloc(&d_cardinalities, N * sizeof(int)));
    // Scratchpads
    CUDA_CHECK(cudaMalloc(&d_distances, (size_t)N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_in_cluster, (size_t)N * N * sizeof(bool)));

    // Copy initial data
    CUDA_CHECK(cudaMemcpy(d_points, points.data(), N * sizeof(Point), cudaMemcpyHostToDevice));
    
    // Host side buffers for results
    std::vector<int> h_cardinalities(N);

    // Main clustering loop
    while (unclustered_count > 0) {
        // Copy clustered status to GPU
        // Optimization: We could keep clustered on GPU and only update it, but keeping sync is easier for now
        // or just update d_clustered when we find a cluster.
        // Let's copy for safety and simplicity first.
        std::vector<bool> temp_clustered = clustered; // copy from vector<bool> proxy
        bool* temp_bool_array = new bool[N];
        for(int i=0; i<N; ++i) temp_bool_array[i] = clustered[i];
        
        CUDA_CHECK(cudaMemcpy(d_clustered, temp_bool_array, N * sizeof(bool), cudaMemcpyHostToDevice));
        delete[] temp_bool_array;

        // Launch Kernel
        int blockSize = 256;
        int numBlocks = (N + blockSize - 1) / blockSize;
        generate_candidate_clusters_kernel<<<numBlocks, blockSize>>>(
            d_points, d_clustered, d_cardinalities, d_distances, d_in_cluster, N, threshold * threshold
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Read back cardinalities
        CUDA_CHECK(cudaMemcpy(h_cardinalities.data(), d_cardinalities, N * sizeof(int), cudaMemcpyDeviceToHost));

        // Find best seed on CPU
        int max_cardinality = -1;
        int best_seed = -1;

        for (int i = 0; i < N; ++i) {
            if (!clustered[i]) {
                if (h_cardinalities[i] > max_cardinality) {
                    max_cardinality = h_cardinalities[i];
                    best_seed = i;
                }
            }
        }

        // If we found a cluster, add it
        if (best_seed >= 0 && max_cardinality > 0) {
            Cluster cluster;
            cluster.seed_point = best_seed;
            
            // Regenerate members on CPU
            regenerateBestCluster(best_seed, clustered, points, threshold, cluster.members);
            
            clusters.push_back(cluster);
            
            // Mark all members as clustered
            for (int member : cluster.members) {
                clustered[member] = true;
                unclustered_count--;
            }
        } else {
            // No more clusters can be formed
            break;
        }
    }
    
    CUDA_CHECK(cudaFree(d_points));
    CUDA_CHECK(cudaFree(d_clustered));
    CUDA_CHECK(cudaFree(d_cardinalities));
    CUDA_CHECK(cudaFree(d_distances));
    CUDA_CHECK(cudaFree(d_in_cluster));

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
