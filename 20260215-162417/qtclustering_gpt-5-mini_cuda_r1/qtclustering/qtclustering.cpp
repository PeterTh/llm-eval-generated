// QT Clustering Benchmark - CUDA-accelerated version
//
// This file is adapted to perform the candidate distance evaluations on the GPU
// using CUDA kernels while keeping algorithm semantics unchanged.

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

// CUDA error check
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s)\n", __FILE__, __LINE__, (int)err, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// GPU context for reused device buffers
namespace gpu {
    int N = 0;
    double* d_x = nullptr;
    double* d_y = nullptr;
    double* d_out = nullptr; // per-candidate max distance
    unsigned char* d_clustered = nullptr;
    unsigned char* d_incluster = nullptr;
    int* d_members = nullptr;
    size_t members_cap = 0;

    void init_points(const std::vector<Point>& points) {
        N = static_cast<int>(points.size());
        std::vector<double> xs(N), ys(N);
        for (int i = 0; i < N; ++i) { xs[i] = points[i].x; ys[i] = points[i].y; }

        CUDA_CHECK(cudaMalloc((void**)&d_x, sizeof(double) * N));
        CUDA_CHECK(cudaMalloc((void**)&d_y, sizeof(double) * N));
        CUDA_CHECK(cudaMemcpy(d_x, xs.data(), sizeof(double) * N, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_y, ys.data(), sizeof(double) * N, cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc((void**)&d_out, sizeof(double) * N));
        CUDA_CHECK(cudaMalloc((void**)&d_clustered, sizeof(unsigned char) * N));
        CUDA_CHECK(cudaMalloc((void**)&d_incluster, sizeof(unsigned char) * N));
    }

    void ensure_members_capacity(size_t cap) {
        if (cap <= members_cap) return;
        if (d_members) CUDA_CHECK(cudaFree(d_members));
        CUDA_CHECK(cudaMalloc((void**)&d_members, sizeof(int) * cap));
        members_cap = cap;
    }

    void free_all() {
        if (d_x) CUDA_CHECK(cudaFree(d_x));
        if (d_y) CUDA_CHECK(cudaFree(d_y));
        if (d_out) CUDA_CHECK(cudaFree(d_out));
        if (d_clustered) CUDA_CHECK(cudaFree(d_clustered));
        if (d_incluster) CUDA_CHECK(cudaFree(d_incluster));
        if (d_members) CUDA_CHECK(cudaFree(d_members));
        d_x = d_y = d_out = nullptr;
        d_clustered = d_incluster = nullptr;
        d_members = nullptr;
        N = 0; members_cap = 0;
    }
}

// Kernel: for each candidate index, compute the maximum distance to any member
extern "C" __global__ void compute_max_distances(const double* xs, const double* ys,
                                                   int N,
                                                   const int* members, int mcount,
                                                   const unsigned char* clustered,
                                                   const unsigned char* in_cluster,
                                                   double* out_max) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    if (clustered[idx] || in_cluster[idx]) {
        out_max[idx] = 1e300; // mark as invalid
        return;
    }

    double px = xs[idx];
    double py = ys[idx];
    double maxd = 0.0;
    for (int i = 0; i < mcount; ++i) {
        int mem = members[i];
        double dx = px - xs[mem];
        double dy = py - ys[mem];
        double d = sqrt(dx * dx + dy * dy);
        if (d > maxd) maxd = d;
    }
    out_max[idx] = maxd;
}

// Generate synthetic 2D point data in clusters
void generateSyntheticData(std::vector<Point>& points, const int N, unsigned int seed = 42) {
    auto frand = [&seed]() mutable { return rand_r(&seed) / static_cast<double>(RAND_MAX); };
    
    const double min_dim = std::min(MAX_WIDTH, MAX_HEIGHT);
    int count = 0;
    
    while (count < N) {
        const double cntr_x = frand() * MAX_WIDTH;
        const double cntr_y = frand() * MAX_HEIGHT;
        const double R = frand() * min_dim / 2.0;
        int group_cnt = static_cast<int>(frand() * (N / 30.0));
        
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

// Calculate Euclidean distance between two points (host)
inline double distance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

// GPU-accelerated version of findClosestPoint
int findClosestPoint(const std::vector<int>& cluster_members,
                     const std::vector<bool>& clustered,
                     const std::vector<bool>& in_cluster,
                     const std::vector<Point>& points,
                     const double threshold,
                     const int point_count) {
    // Prepare host-side buffers
    const int N = point_count;
    int closest_point = -1;
    double min_diameter = std::numeric_limits<double>::max();

    // Convert bool vectors to unsigned char arrays for device copy
    std::vector<unsigned char> h_clustered(N);
    std::vector<unsigned char> h_incluster(N);
    for (int i = 0; i < N; ++i) { h_clustered[i] = clustered[i] ? 1 : 0; h_incluster[i] = in_cluster[i] ? 1 : 0; }

    // Members
    const int mcount = static_cast<int>(cluster_members.size());
    gpu::ensure_members_capacity(mcount);

    // Copy members and flags to device
    if (mcount > 0) CUDA_CHECK(cudaMemcpy(gpu::d_members, cluster_members.data(), sizeof(int) * mcount, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu::d_clustered, h_clustered.data(), sizeof(unsigned char) * N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu::d_incluster, h_incluster.data(), sizeof(unsigned char) * N, cudaMemcpyHostToDevice));

    // Launch kernel to compute per-candidate max distance
    const int block = 256;
    const int grid = (N + block - 1) / block;
    compute_max_distances<<<grid, block>>>(gpu::d_x, gpu::d_y, N, gpu::d_members, mcount, gpu::d_clustered, gpu::d_incluster, gpu::d_out);
    CUDA_CHECK(cudaGetLastError());

    // Copy results back
    std::vector<double> h_out(N);
    CUDA_CHECK(cudaMemcpy(h_out.data(), gpu::d_out, sizeof(double) * N, cudaMemcpyDeviceToHost));

    for (int candidate = 0; candidate < N; ++candidate) {
        const double max_dist = h_out[candidate];
        if (!(max_dist < 1e200)) continue; // invalid or marked
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
    std::vector<bool> clustered(N, false);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;
    
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

    // Initialize GPU context and copy points
    gpu::init_points(points);
    
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
    
    // Cleanup GPU
    gpu::free_all();

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
