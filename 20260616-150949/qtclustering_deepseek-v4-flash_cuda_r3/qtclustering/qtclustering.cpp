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

#include "../common/results_output.hpp"

#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// Maximum GPU state allocation (512 MB for per-thread in_cluster arrays)
static const size_t MAX_GPU_STATE_BYTES = 512ULL * 1024 * 1024;

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

// CUDA kernel: compute candidate cluster cardinality for all unclustered seed points
// Each thread evaluates one seed point independently, running the full sequential
// cluster-building algorithm. Per-thread in_cluster state is in a global buffer.
// Kernel: reads points directly from global memory (used when N > 2400)
__global__ void computeCardinalitiesGMEM(
    const double* __restrict__ points_x,
    const double* __restrict__ points_y,
    const char* __restrict__ clustered,
    const int* __restrict__ unclustered_indices,
    int num_unclustered,
    int num_points,
    double threshold,
    char* __restrict__ thread_state,
    int* __restrict__ members_buffer,
    int* __restrict__ out_cardinalities
) {
    const double threshold_sq = threshold * threshold;

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= num_unclustered) return;

    int seed = unclustered_indices[idx];
    if (clustered[seed]) {
        out_cardinalities[idx] = 0;
        return;
    }

    char* in_cluster = thread_state + (size_t)idx * num_points;
    int* members = members_buffer + (size_t)idx * num_points;
    in_cluster[seed] = 1;
    members[0] = seed;
    int member_count = 1;

    while (member_count < num_points) {
        int best_candidate = -1;
        double best_max_dist_sq = threshold_sq;

        for (int candidate = 0; candidate < num_points; ++candidate) {
            if (clustered[candidate] || in_cluster[candidate]) continue;

            double max_dist_sq = 0.0;
            for (int m = 0; m < member_count; ++m) {
                int other = members[m];
                double dx = points_x[candidate] - points_x[other];
                double dy = points_y[candidate] - points_y[other];
                double dist_sq = dx * dx + dy * dy;
                if (dist_sq > max_dist_sq) max_dist_sq = dist_sq;
                if (max_dist_sq >= threshold_sq) break;
            }

            if (max_dist_sq < threshold_sq && max_dist_sq < best_max_dist_sq) {
                best_max_dist_sq = max_dist_sq;
                best_candidate = candidate;
            }
        }

        if (best_candidate < 0) break;

        in_cluster[best_candidate] = 1;
        members[member_count] = best_candidate;
        member_count++;
    }

    out_cardinalities[idx] = member_count;
}

// Shared memory kernel (for N <= 2400, points fit in 48KB shared memory)
__global__ void computeCardinalitiesSMEM(
    const double* __restrict__ points_x,
    const double* __restrict__ points_y,
    const char* __restrict__ clustered,
    const int* __restrict__ unclustered_indices,
    int num_unclustered,
    int num_points,
    double threshold,
    char* __restrict__ thread_state,
    int* __restrict__ members_buffer,
    int* __restrict__ out_cardinalities
) {
    extern __shared__ double smem_points[];
    double* sx = smem_points;
    double* sy = smem_points + num_points;

    for (int i = threadIdx.x; i < num_points; i += blockDim.x) {
        sx[i] = points_x[i];
        sy[i] = points_y[i];
    }
    __syncthreads();

    const double threshold_sq = threshold * threshold;

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= num_unclustered) return;

    int seed = unclustered_indices[idx];
    if (clustered[seed]) {
        out_cardinalities[idx] = 0;
        return;
    }

    char* in_cluster = thread_state + (size_t)idx * num_points;
    int* members = members_buffer + (size_t)idx * num_points;
    in_cluster[seed] = 1;
    members[0] = seed;
    int member_count = 1;

    while (member_count < num_points) {
        int best_candidate = -1;
        double best_max_dist_sq = threshold_sq;

        for (int candidate = 0; candidate < num_points; ++candidate) {
            if (clustered[candidate] || in_cluster[candidate]) continue;

            double cand_x = sx[candidate];
            double cand_y = sy[candidate];

            double max_dist_sq = 0.0;
            for (int m = 0; m < member_count; ++m) {
                int other = members[m];
                double dx = cand_x - sx[other];
                double dy = cand_y - sy[other];
                double dist_sq = dx * dx + dy * dy;
                if (dist_sq > max_dist_sq) max_dist_sq = dist_sq;
                if (max_dist_sq >= threshold_sq) break;
            }

            if (max_dist_sq < threshold_sq && max_dist_sq < best_max_dist_sq) {
                best_max_dist_sq = max_dist_sq;
                best_candidate = candidate;
            }
        }

        if (best_candidate < 0) break;

        in_cluster[best_candidate] = 1;
        members[member_count] = best_candidate;
        member_count++;
    }

    out_cardinalities[idx] = member_count;
}


// Main QT clustering algorithm (GPU-accelerated)
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<bool> clustered(N, false);
    std::vector<char> h_clustered(N, 0);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;

    // Initialize unclustered indices
    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }

    // Convert points to SoA layout for coalesced GPU access
    std::vector<double> h_points_x(N), h_points_y(N);
    for (int i = 0; i < N; ++i) {
        h_points_x[i] = points[i].x;
        h_points_y[i] = points[i].y;
    }

    // Allocate GPU memory
    double *d_points_x = nullptr, *d_points_y = nullptr;
    char *d_clustered = nullptr;
    int *d_unclustered = nullptr, *d_cardinalities = nullptr, *d_members = nullptr;
    char *d_state = nullptr;

    CUDA_CHECK(cudaMalloc(&d_points_x, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_points_y, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_clustered, N * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&d_unclustered, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cardinalities, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_members, (size_t)N * N * sizeof(int)));

    // Per-thread state: up to N threads * N points bytes (in_cluster flags)
    size_t state_size = (size_t)N * (size_t)N;
    if (state_size > MAX_GPU_STATE_BYTES) {
        fprintf(stderr, "Error: N=%d requires %zu MB GPU state memory (limit %zu MB)\n",
                N, state_size / (1024 * 1024),
                MAX_GPU_STATE_BYTES / (1024 * 1024));
        exit(1);
    }
    CUDA_CHECK(cudaMalloc(&d_state, state_size));

    // Copy points to GPU (constant throughout the algorithm)
    CUDA_CHECK(cudaMemcpy(d_points_x, h_points_x.data(), N * sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_points_y, h_points_y.data(), N * sizeof(double),
                          cudaMemcpyHostToDevice));

    std::vector<int> h_cardinalities(N);
    const int threads_per_block = 128;

    while (!unclustered_indices.empty()) {
        int num_unclustered = static_cast<int>(unclustered_indices.size());

        // Copy current clustered state to GPU
        CUDA_CHECK(cudaMemcpy(d_clustered, h_clustered.data(), N * sizeof(char),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_unclustered, unclustered_indices.data(),
                              num_unclustered * sizeof(int), cudaMemcpyHostToDevice));

        // Clear per-thread state for this iteration
        CUDA_CHECK(cudaMemset(d_state, 0, (size_t)num_unclustered * N));

        // Launch kernel: one thread per seed point
        int blocks = (num_unclustered + threads_per_block - 1) / threads_per_block;
        if (blocks > 0) {
            // Use shared memory kernel for N <= 2400 (fits in 48KB limit)
            if (N * 2 * (int)sizeof(double) <= 48 * 1024) {
                int smem_bytes = N * 2 * (int)sizeof(double);
                computeCardinalitiesSMEM<<<blocks, threads_per_block, smem_bytes>>>(
                    d_points_x, d_points_y, d_clustered, d_unclustered,
                    num_unclustered, N, threshold, d_state, d_members, d_cardinalities
                );
            } else {
                computeCardinalitiesGMEM<<<blocks, threads_per_block>>>(
                    d_points_x, d_points_y, d_clustered, d_unclustered,
                    num_unclustered, N, threshold, d_state, d_members, d_cardinalities
                );
            }
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // Copy cardinalities back to host
        CUDA_CHECK(cudaMemcpy(h_cardinalities.data(), d_cardinalities,
                              num_unclustered * sizeof(int), cudaMemcpyDeviceToHost));

        // Find the seed with the largest candidate cluster
        int best_idx = -1;
        int max_cardinality = 0;
        for (int i = 0; i < num_unclustered; ++i) {
            if (h_cardinalities[i] > max_cardinality) {
                max_cardinality = h_cardinalities[i];
                best_idx = i;
            }
        }

        if (best_idx < 0 || max_cardinality <= 0) break;

        int best_seed = unclustered_indices[best_idx];

        // Re-generate the best cluster's members on CPU (one seed only)
        std::vector<int> best_members;
        generateCandidateCluster(best_seed, clustered, points, threshold, N, &best_members);

        // Add cluster
        clusters.push_back({best_members, best_seed});

        // Mark all members as clustered on both host arrays
        for (int m : best_members) {
            clustered[m] = true;
            h_clustered[m] = 1;
        }

        // Remove clustered points from unclustered list
        unclustered_indices.erase(
            std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                          [&clustered](int idx) { return clustered[idx]; }),
            unclustered_indices.end()
        );
    }

    // Free GPU memory
    CUDA_CHECK(cudaFree(d_points_x));
    CUDA_CHECK(cudaFree(d_points_y));
    CUDA_CHECK(cudaFree(d_clustered));
    CUDA_CHECK(cudaFree(d_unclustered));
    CUDA_CHECK(cudaFree(d_cardinalities));
    CUDA_CHECK(cudaFree(d_members));
    CUDA_CHECK(cudaFree(d_state));

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
