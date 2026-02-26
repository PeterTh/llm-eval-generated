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
inline double distance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

// CUDA kernel: Calculate distance between two points
__device__ inline double cuda_distance(double x1, double y1, double x2, double y2) {
    double dx = x1 - x2;
    double dy = y1 - y2;
    return sqrt(dx * dx + dy * dy);
}

// CUDA kernel: Find maximum distance from a candidate point to all cluster members
__global__ void computeMaxDistances(
    const double* __restrict__ points_x,
    const double* __restrict__ points_y,
    const int* __restrict__ cluster_members,
    const int cluster_size,
    const bool* __restrict__ clustered,
    const bool* __restrict__ in_cluster,
    const int point_count,
    double* __restrict__ max_distances,
    const double threshold
) {
    int candidate = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (candidate >= point_count || clustered[candidate] || in_cluster[candidate]) {
        if (candidate < point_count) {
            max_distances[candidate] = INFINITY;
        }
        return;
    }
    
    double max_dist = 0.0;
    double cand_x = points_x[candidate];
    double cand_y = points_y[candidate];
    
    for (int i = 0; i < cluster_size; ++i) {
        int member = cluster_members[i];
        double dist = cuda_distance(cand_x, cand_y, points_x[member], points_y[member]);
        max_dist = fmax(max_dist, dist);
    }
    
    max_distances[candidate] = (max_dist < threshold) ? max_dist : INFINITY;
}

// CUDA kernel: Generate candidate cluster for a single seed
__global__ void generateCandidateClusterKernel(
    const double* __restrict__ points_x,
    const double* __restrict__ points_y,
    const int seed_point,
    const bool* __restrict__ clustered,
    const int point_count,
    const double threshold,
    int* __restrict__ cluster_members,
    int* __restrict__ cluster_size_out
) {
    if (blockIdx.x > 0 || threadIdx.x > 0) return;
    
    extern __shared__ bool in_cluster[];
    
    for (int i = 0; i < point_count; ++i) {
        in_cluster[i] = false;
    }
    
    in_cluster[seed_point] = true;
    cluster_members[0] = seed_point;
    int current_size = 1;
    
    while (current_size < point_count) {
        int best_candidate = -1;
        double min_max_dist = INFINITY;
        
        for (int candidate = 0; candidate < point_count; ++candidate) {
            if (clustered[candidate] || in_cluster[candidate]) continue;
            
            double cand_x = points_x[candidate];
            double cand_y = points_y[candidate];
            double max_dist = 0.0;
            
            for (int i = 0; i < current_size; ++i) {
                int member = cluster_members[i];
                double dist = cuda_distance(cand_x, cand_y, points_x[member], points_y[member]);
                max_dist = fmax(max_dist, dist);
            }
            
            if (max_dist < threshold && max_dist < min_max_dist) {
                min_max_dist = max_dist;
                best_candidate = candidate;
            }
        }
        
        if (best_candidate < 0) break;
        
        in_cluster[best_candidate] = true;
        cluster_members[current_size] = best_candidate;
        current_size++;
    }
    
    *cluster_size_out = current_size;
}

// CUDA kernel: Evaluate all seeds in parallel (using global memory for in_cluster)
__global__ void evaluateAllSeeds(
    const double* __restrict__ points_x,
    const double* __restrict__ points_y,
    const int* __restrict__ unclustered_indices,
    const int unclustered_count,
    const bool* __restrict__ clustered,
    const int point_count,
    const double threshold,
    int* __restrict__ cardinalities,
    int* __restrict__ all_cluster_members,
    bool* __restrict__ all_in_cluster,
    const int max_cluster_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= unclustered_count) return;
    
    int seed = unclustered_indices[idx];
    if (clustered[seed]) {
        cardinalities[idx] = 0;
        return;
    }
    
    // Each thread gets its own section of the in_cluster array
    bool* in_cluster = &all_in_cluster[idx * point_count];
    int* my_members = &all_cluster_members[idx * max_cluster_size];
    
    for (int i = 0; i < point_count; ++i) {
        in_cluster[i] = false;
    }
    
    in_cluster[seed] = true;
    my_members[0] = seed;
    int current_size = 1;
    
    while (current_size < point_count && current_size < max_cluster_size) {
        int best_candidate = -1;
        double min_max_dist = INFINITY;
        
        for (int candidate = 0; candidate < point_count; ++candidate) {
            if (clustered[candidate] || in_cluster[candidate]) continue;
            
            double cand_x = points_x[candidate];
            double cand_y = points_y[candidate];
            double max_dist = 0.0;
            
            for (int i = 0; i < current_size; ++i) {
                int member = my_members[i];
                double dist = cuda_distance(cand_x, cand_y, points_x[member], points_y[member]);
                max_dist = fmax(max_dist, dist);
            }
            
            if (max_dist < threshold && max_dist < min_max_dist) {
                min_max_dist = max_dist;
                best_candidate = candidate;
            }
        }
        
        if (best_candidate < 0) break;
        
        in_cluster[best_candidate] = true;
        my_members[current_size] = best_candidate;
        current_size++;
    }
    
    cardinalities[idx] = current_size;
}

// Main QT clustering algorithm with CUDA
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
    
    // Allocate GPU memory for points (Structure of Arrays)
    double *d_points_x, *d_points_y;
    CUDA_CHECK(cudaMalloc(&d_points_x, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_points_y, N * sizeof(double)));
    
    // Copy points to GPU
    std::vector<double> points_x(N), points_y(N);
    for (int i = 0; i < N; ++i) {
        points_x[i] = points[i].x;
        points_y[i] = points[i].y;
    }
    CUDA_CHECK(cudaMemcpy(d_points_x, points_x.data(), N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_points_y, points_y.data(), N * sizeof(double), cudaMemcpyHostToDevice));
    
    // Allocate GPU memory for clustering state (persistent)
    bool *d_clustered;
    CUDA_CHECK(cudaMalloc(&d_clustered, N * sizeof(bool)));
    
    // Pre-allocate maximum working memory (reuse across iterations)
    // Use adaptive batch size based on problem size
    const int max_batch_size = std::min(N, 1024); // Larger batches for better GPU utilization
    int *d_unclustered_indices, *d_cardinalities, *d_all_cluster_members;
    bool *d_all_in_cluster;
    const int max_cluster_size = N;
    
    CUDA_CHECK(cudaMalloc(&d_unclustered_indices, max_batch_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cardinalities, max_batch_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_all_cluster_members, max_batch_size * max_cluster_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_all_in_cluster, max_batch_size * N * sizeof(bool)));
    
    // Helper buffer for bool vector
    std::vector<unsigned char> clustered_buffer(N);
    
    // Main clustering loop
    while (!unclustered_indices.empty()) {
        const int unclustered_count = static_cast<int>(unclustered_indices.size());
        const int batch_size = std::min(unclustered_count, max_batch_size);
        
        // Copy current clustered state to GPU (convert vector<bool> to buffer)
        for (int i = 0; i < N; ++i) {
            clustered_buffer[i] = clustered[i] ? 1 : 0;
        }
        CUDA_CHECK(cudaMemcpy(d_clustered, clustered_buffer.data(), N * sizeof(bool), cudaMemcpyHostToDevice));
        
        // Copy batch of unclustered indices
        CUDA_CHECK(cudaMemcpy(d_unclustered_indices, unclustered_indices.data(), 
                             batch_size * sizeof(int), cudaMemcpyHostToDevice));
        
        // Launch kernel to evaluate seeds in parallel (batch at a time)
        const int threads_per_block = std::min(batch_size, 256);
        const int blocks = (batch_size + threads_per_block - 1) / threads_per_block;
        
        evaluateAllSeeds<<<blocks, threads_per_block>>>(
            d_points_x, d_points_y,
            d_unclustered_indices, batch_size,
            d_clustered, N, threshold,
            d_cardinalities, d_all_cluster_members, d_all_in_cluster, max_cluster_size
        );
        
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Copy results back
        std::vector<int> cardinalities(batch_size);
        CUDA_CHECK(cudaMemcpy(cardinalities.data(), d_cardinalities, 
                             batch_size * sizeof(int), cudaMemcpyDeviceToHost));
        
        // Find best seed on CPU
        int max_cardinality = -1;
        int best_idx = -1;
        for (int i = 0; i < batch_size; ++i) {
            if (cardinalities[i] > max_cardinality) {
                max_cardinality = cardinalities[i];
                best_idx = i;
            }
        }
        
        // If we found a cluster, add it
        if (best_idx >= 0 && max_cardinality > 0) {
            std::vector<int> best_cluster_members(max_cardinality);
            CUDA_CHECK(cudaMemcpy(best_cluster_members.data(), 
                                 d_all_cluster_members + best_idx * max_cluster_size,
                                 max_cardinality * sizeof(int), cudaMemcpyDeviceToHost));
            
            Cluster cluster;
            cluster.seed_point = unclustered_indices[best_idx];
            cluster.members = best_cluster_members;
            clusters.push_back(cluster);
            
            // Mark all members as clustered
            for (int member : best_cluster_members) {
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
    CUDA_CHECK(cudaFree(d_points_x));
    CUDA_CHECK(cudaFree(d_points_y));
    CUDA_CHECK(cudaFree(d_clustered));
    CUDA_CHECK(cudaFree(d_unclustered_indices));
    CUDA_CHECK(cudaFree(d_cardinalities));
    CUDA_CHECK(cudaFree(d_all_cluster_members));
    CUDA_CHECK(cudaFree(d_all_in_cluster));
    
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
