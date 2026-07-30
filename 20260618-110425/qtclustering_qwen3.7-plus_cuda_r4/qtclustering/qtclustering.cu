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

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
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

// Generate synthetic 2D point data in clusters (same as original, sequential)
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

// ======================== CUDA Kernels ========================

// Kernel: find the closest unclustered point that maintains diameter < threshold
// Each thread handles one candidate point
// Computes max distance from candidate to all current cluster members
// Block-level reduction finds the best candidate per block
// Per-block results are stored for final CPU reduction
__global__ void find_closest_point_kernel(
    const int* candidates,       // candidate point indices (all N points)
    const int* members,          // current cluster member indices
    const double* points_x,
    const double* points_y,
    const int* is_clustered,     // 1 if point is already clustered
    const int* is_in_cluster,    // 1 if point is in current candidate cluster
    int num_candidates,
    int num_members,
    double threshold,
    double* block_best_dist,     // [num_blocks] output
    int* block_best_point)       // [num_blocks] output
{
    extern __shared__ double sdata[];
    double* sdist = sdata;
    int* sidx = (int*)&sdata[blockDim.x];
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    double my_dist = 1e300;
    int my_point = -1;
    
    if (gid < num_candidates) {
        int cand_idx = candidates[gid];
        
        if (!is_clustered[cand_idx] && !is_in_cluster[cand_idx]) {
            double cx = points_x[cand_idx];
            double cy = points_y[cand_idx];
            
            double max_d = 0.0;
            for (int m = 0; m < num_members; ++m) {
                int mem_idx = members[m];
                double dx = cx - points_x[mem_idx];
                double dy = cy - points_y[mem_idx];
                double d = sqrt(dx * dx + dy * dy);
                if (d > max_d) max_d = d;
            }
            
            if (max_d < threshold) {
                my_dist = max_d;
                my_point = cand_idx;
            }
        }
    }
    
    sdist[tid] = my_dist;
    sidx[tid] = my_point;
    __syncthreads();
    
    // Block-level reduction to find minimum
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdist[tid + s] < sdist[tid]) {
                sdist[tid] = sdist[tid + s];
                sidx[tid] = sidx[tid + s];
            }
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        block_best_dist[blockIdx.x] = sdist[0];
        block_best_point[blockIdx.x] = sidx[0];
    }
}

// ======================== Host-side CUDA wrapper ========================

// Find closest point using GPU acceleration
int findClosestPointGPU(
    const std::vector<int>& cluster_members,
    const int* d_is_clustered,
    const int* d_is_in_cluster,
    const double* d_points_x,
    const double* d_points_y,
    int* d_candidates,
    int* d_members,
    double* d_block_best_dist,
    int* d_block_best_point,
    std::vector<double>& h_block_dist,
    std::vector<int>& h_block_point,
    const double threshold,
    const int point_count,
    const int block_size,
    const int max_blocks)
{
    // Build candidate list - all points 0..N-1, kernel checks is_clustered/is_in_cluster
    // Upload sequential indices as candidates
    // Actually, we pre-upload [0,1,2,...,N-1] once and reuse
    
    int num_candidates = point_count;
    int num_members = static_cast<int>(cluster_members.size());
    
    // Copy members to device
    CUDA_CHECK(cudaMemcpy(d_members, cluster_members.data(), num_members * sizeof(int), cudaMemcpyHostToDevice));
    
    // Launch kernel
    const int num_blocks = (num_candidates + block_size - 1) / block_size;
    size_t shared_mem = block_size * (sizeof(double) + sizeof(int));
    
    find_closest_point_kernel<<<num_blocks, block_size, shared_mem>>>(
        d_candidates, d_members,
        d_points_x, d_points_y,
        d_is_clustered, d_is_in_cluster,
        num_candidates, num_members,
        threshold,
        d_block_best_dist, d_block_best_point
    );
    CUDA_CHECK(cudaGetLastError());
    
    // Download block results
    CUDA_CHECK(cudaMemcpy(h_block_dist.data(), d_block_best_dist, num_blocks * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_block_point.data(), d_block_best_point, num_blocks * sizeof(int), cudaMemcpyDeviceToHost));
    
    // CPU-side final reduction
    int best_point = -1;
    double best_diameter = std::numeric_limits<double>::max();
    
    for (int b = 0; b < num_blocks; ++b) {
        if (h_block_dist[b] < best_diameter) {
            best_diameter = h_block_dist[b];
            best_point = h_block_point[b];
        }
    }
    
    return best_point;
}

// ======================== Main clustering algorithm ========================

// Generate a candidate cluster starting from a seed point (uses GPU for findClosestPoint)
int generateCandidateClusterGPU(
    const int seed_point,
    const int* d_is_clustered,
    int* d_is_in_cluster,
    const double* d_points_x,
    const double* d_points_y,
    int* d_candidates,
    int* d_members,
    double* d_block_best_dist,
    int* d_block_best_point,
    std::vector<double>& h_block_dist,
    std::vector<int>& h_block_point,
    const double threshold,
    const int point_count,
    const int block_size,
    const int max_blocks,
    std::vector<int>* cluster_members_out)
{
    std::vector<bool> in_cluster(point_count, false);
    std::vector<int> members;
    
    // Add seed point
    in_cluster[seed_point] = true;
    members.push_back(seed_point);
    
    // Upload is_in_cluster for the seed
    int one = 1;
    CUDA_CHECK(cudaMemcpy(&d_is_in_cluster[seed_point], &one, sizeof(int), cudaMemcpyHostToDevice));
    
    // Iteratively add closest points
    while (static_cast<int>(members.size()) < point_count) {
        const int closest = findClosestPointGPU(
            members,
            d_is_clustered, d_is_in_cluster,
            d_points_x, d_points_y,
            d_candidates, d_members,
            d_block_best_dist, d_block_best_point,
            h_block_dist, h_block_point,
            threshold, point_count,
            block_size, max_blocks);
        
        if (closest < 0) break;
        
        in_cluster[closest] = true;
        members.push_back(closest);
        
        // Update is_in_cluster on device
        CUDA_CHECK(cudaMemcpy(&d_is_in_cluster[closest], &one, sizeof(int), cudaMemcpyHostToDevice));
    }
    
    if (cluster_members_out) {
        *cluster_members_out = members;
    }
    
    return static_cast<int>(members.size());
}

// Main QT clustering algorithm with GPU acceleration
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
    
    // Allocate device memory for points
    std::vector<double> h_points_x(N), h_points_y(N);
    for (int i = 0; i < N; ++i) {
        h_points_x[i] = points[i].x;
        h_points_y[i] = points[i].y;
    }
    
    double* d_points_x;
    double* d_points_y;
    int* d_is_clustered;
    int* d_is_in_cluster;
    int* d_candidates;
    int* d_members;
    double* d_block_best_dist;
    int* d_block_best_point;
    
    CUDA_CHECK(cudaMalloc(&d_points_x, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_points_y, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_is_clustered, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_is_in_cluster, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_candidates, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_members, N * sizeof(int)));
    
    const int block_size = 256;
    const int max_blocks = (N + block_size - 1) / block_size;
    CUDA_CHECK(cudaMalloc(&d_block_best_dist, max_blocks * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_block_best_point, max_blocks * sizeof(int)));
    
    // Upload point coordinates
    CUDA_CHECK(cudaMemcpy(d_points_x, h_points_x.data(), N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_points_y, h_points_y.data(), N * sizeof(double), cudaMemcpyHostToDevice));
    
    // Initialize is_clustered to 0
    CUDA_CHECK(cudaMemset(d_is_clustered, 0, N * sizeof(int)));
    
    // Pre-upload candidates array [0, 1, 2, ..., N-1]
    std::vector<int> h_candidates(N);
    for (int i = 0; i < N; ++i) h_candidates[i] = i;
    CUDA_CHECK(cudaMemcpy(d_candidates, h_candidates.data(), N * sizeof(int), cudaMemcpyHostToDevice));
    
    // Pre-allocate host block result buffers
    std::vector<double> h_block_dist(max_blocks);
    std::vector<int> h_block_point(max_blocks);
    
    // Main clustering loop
    while (!unclustered_indices.empty()) {
        int max_cardinality = -1;
        int best_seed = -1;
        std::vector<int> best_cluster_members;
        
        // Try each unclustered point as a seed
        for (size_t i = 0; i < unclustered_indices.size(); ++i) {
            const int seed = unclustered_indices[i];
            if (clustered[seed]) continue;
            
            // Reset is_in_cluster on device
            CUDA_CHECK(cudaMemset(d_is_in_cluster, 0, N * sizeof(int)));
            
            std::vector<int> candidate_members;
            const int cardinality = generateCandidateClusterGPU(
                seed,
                d_is_clustered, d_is_in_cluster,
                d_points_x, d_points_y,
                d_candidates, d_members,
                d_block_best_dist, d_block_best_point,
                h_block_dist, h_block_point,
                threshold, N,
                block_size, max_blocks,
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
            
            // Mark all members as clustered (on host and device)
            // Batch upload for efficiency
            std::vector<int> h_ones(best_cluster_members.size(), 1);
            for (size_t i = 0; i < best_cluster_members.size(); ++i) {
                clustered[best_cluster_members[i]] = true;
            }
            // Upload is_clustered for each member
            for (size_t i = 0; i < best_cluster_members.size(); ++i) {
                int idx = best_cluster_members[i];
                CUDA_CHECK(cudaMemcpy(&d_is_clustered[idx], &h_ones[i], sizeof(int), cudaMemcpyHostToDevice));
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
    
    // Free device memory
    CUDA_CHECK(cudaFree(d_points_x));
    CUDA_CHECK(cudaFree(d_points_y));
    CUDA_CHECK(cudaFree(d_is_clustered));
    CUDA_CHECK(cudaFree(d_is_in_cluster));
    CUDA_CHECK(cudaFree(d_candidates));
    CUDA_CHECK(cudaFree(d_members));
    CUDA_CHECK(cudaFree(d_block_best_dist));
    CUDA_CHECK(cudaFree(d_block_best_point));
    
    return clusters;
}

// Calculate Euclidean distance between two points (CPU version for validation)
inline double distance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
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
