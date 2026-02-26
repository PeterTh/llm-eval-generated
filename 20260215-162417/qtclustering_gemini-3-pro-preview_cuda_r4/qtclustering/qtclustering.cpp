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
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
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
inline __host__ __device__ double distance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return sqrt(dx * dx + dy * dy);
}

inline __host__ __device__ double distanceSquared(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return dx * dx + dy * dy;
}

// CUDA Kernel to compute candidate cluster size for multiple seeds
// Each thread handles one seed
__global__ void compute_candidate_cluster_size_kernel(
    const int* seeds,
    const int num_seeds,
    const Point* points,
    const char* clustered, // 1 if clustered, 0 otherwise
    const int num_points,
    const double threshold,
    int* result_sizes,     // Output: size of cluster for each seed
    int* scratch_members,  // Scratch space: num_seeds * num_points ints
    char* scratch_in_cluster // Scratch space: num_seeds * num_points chars
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_seeds) return;

    int seed_point = seeds[idx];
    double threshold_sq = threshold * threshold;
    
    // Pointers to thread-local scratch space in global memory
    // Use SoA layout for better coalescence: members[i] is at scratch + i * num_seeds + idx
    // scratch_members size: num_points * num_seeds
    // scratch_in_cluster size: num_points * num_seeds
    
    // Initialize in_cluster array
    for (int i = 0; i < num_points; ++i) {
        scratch_in_cluster[i * num_seeds + idx] = 0;
    }

    // Add seed point
    scratch_in_cluster[seed_point * num_seeds + idx] = 1;
    scratch_members[0 * num_seeds + idx] = seed_point;
    int member_count = 1;

    // Iteratively add closest points
    while (member_count < num_points) {
        int best_candidate = -1;
        double min_max_dist_sq = 1e30; // Infinity

        // Try each unclustered point as a candidate
        for (int candidate = 0; candidate < num_points; ++candidate) {
            // Skip if already clustered globally or already in this local cluster
            if (clustered[candidate] || scratch_in_cluster[candidate * num_seeds + idx]) continue;

            // Calculate the maximum distance from candidate to all cluster members
            double max_dist_sq = 0.0;
            bool within_threshold = true;

            for (int i = 0; i < member_count; ++i) {
                const int member = scratch_members[i * num_seeds + idx];
                const double d2 = distanceSquared(points[candidate], points[member]);
                
                if (d2 >= threshold_sq) {
                    within_threshold = false;
                    break;
                }
                if (d2 > max_dist_sq) {
                    max_dist_sq = d2;
                    if (max_dist_sq >= min_max_dist_sq) {
                         within_threshold = false; 
                         break;
                    }
                }
            }

            if (within_threshold) {
                 if (max_dist_sq < threshold_sq && max_dist_sq < min_max_dist_sq) {
                     min_max_dist_sq = max_dist_sq;
                     best_candidate = candidate;
                 }
            }
        }

        if (best_candidate < 0) break; // No more points can be added

        scratch_in_cluster[best_candidate * num_seeds + idx] = 1;
        scratch_members[member_count * num_seeds + idx] = best_candidate;
        member_count++;
    }

    result_sizes[idx] = member_count;
}

// Helper to generate a single cluster on CPU (for the winner)
// We duplicate the logic to avoid copying the member list from GPU
int generateCandidateClusterCPU(const int seed_point,
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
        int closest_point = -1;
        double min_diameter = std::numeric_limits<double>::max();
        
        for (int candidate = 0; candidate < point_count; ++candidate) {
            if (clustered[candidate] || in_cluster[candidate]) continue;
            
            double max_dist = 0.0;
            for (size_t i = 0; i < members.size(); ++i) {
                const int member = members[i];
                const double dist = distance(points[candidate], points[member]);
                max_dist = std::max(max_dist, dist);
            }
            
            if (max_dist < threshold && max_dist < min_diameter) {
                min_diameter = max_dist;
                closest_point = candidate;
            }
        }
        
        if (closest_point < 0) break;
        
        in_cluster[closest_point] = true;
        members.push_back(closest_point);
    }
    
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

    // Allocate device memory
    Point* d_points;
    CHECK_CUDA(cudaMalloc(&d_points, N * sizeof(Point)));
    CHECK_CUDA(cudaMemcpy(d_points, points.data(), N * sizeof(Point), cudaMemcpyHostToDevice));

    char* d_clustered;
    CHECK_CUDA(cudaMalloc(&d_clustered, N * sizeof(char)));
    
    int* d_seeds;
    CHECK_CUDA(cudaMalloc(&d_seeds, N * sizeof(int))); // Max seeds is N
    
    int* d_results;
    CHECK_CUDA(cudaMalloc(&d_results, N * sizeof(int)));

    // Determine batch size for seeds to fit in GPU memory
    // Each seed needs scratch space: N * sizeof(int) + N * sizeof(char)
    // For N=1000, 5KB per seed. 1000 seeds = 5MB.
    // For N=10000, 50KB per seed. 10000 seeds = 500MB.
    // For N=100000, 500KB per seed. 100000 seeds = 50GB. (Too big)
    // We should limit batch size.
    size_t free_mem, total_mem;
    CHECK_CUDA(cudaMemGetInfo(&free_mem, &total_mem));
    size_t mem_per_seed = N * sizeof(int) + N * sizeof(char);
    size_t max_seeds_per_batch = (free_mem * 0.9) / mem_per_seed; // Use 90% of free memory
    if (max_seeds_per_batch == 0) max_seeds_per_batch = 1;
    
    int* d_scratch_members;
    char* d_scratch_in_cluster;
    
    // Allocate scratch buffer for the batch
    size_t scratch_members_size = max_seeds_per_batch * N * sizeof(int);
    size_t scratch_in_cluster_size = max_seeds_per_batch * N * sizeof(char);
    
    CHECK_CUDA(cudaMalloc(&d_scratch_members, scratch_members_size));
    CHECK_CUDA(cudaMalloc(&d_scratch_in_cluster, scratch_in_cluster_size));

    std::vector<char> h_clustered_char(N, 0);

    // Main clustering loop
    while (!unclustered_indices.empty()) {
        int max_cardinality = -1;
        int best_seed = -1;
        
        // Update clustered array on device
        for(int i=0; i<N; ++i) h_clustered_char[i] = clustered[i] ? 1 : 0;
        CHECK_CUDA(cudaMemcpy(d_clustered, h_clustered_char.data(), N * sizeof(char), cudaMemcpyHostToDevice));

        // Try each unclustered point as a seed
        // Optimization: We don't need to re-evaluate seeds that were not the best in previous iteration,
        // UNLESS the removal of points changed their cluster size.
        // But removing points only shrinks clusters or keeps them same.
        // It never makes a cluster larger.
        // So if a seed produced a cluster of size X, and we picked a cluster of size Y > X,
        // then in the next iteration, the seed will produce a cluster of size <= X.
        // So we can use the previous size as an upper bound?
        // But we want the MAX cardinality.
        // So we still need to check all seeds?
        // Wait, if we know the current max found so far in the current iteration is M,
        // and we know a seed's upper bound is U < M, we can skip it.
        // But we don't know the upper bound easily.
        // However, we can cache the cluster size from previous iteration.
        // If the seed was not affected by the removal of the winning cluster, its size is unchanged.
        // How to know if affected?
        // If any member of the seed's cluster was removed.
        // But we don't store members for all seeds.

        // So we must re-evaluate.
        // But wait, the prompt says "Change the code in the existing files only... maintain correctness... optimized for maximum performance".
        
        // Let's optimize memory access pattern in kernel.
        // Access to `points` is random in `distance`.
        // `points` is small (5000 * 16 bytes = 80KB). Fits in L2/L1.
        // `clustered` is accessed randomly.
        // `in_cluster` is accessed randomly.

        // Is there a way to avoid re-evaluating all seeds?
        // Maybe we can just evaluate a subset of seeds?
        // No, that changes semantics.

        // What if we check if the seed itself is close to the removed cluster?
        // If the seed is far, its cluster is likely disjoint.
        // But "diameter < threshold" constraint is hard.

        // Let's look at the kernel again.
        // We use local memory for `members` array.
        // `scratch_members` is in global memory.
        // Access to `scratch_members` is coalesced?
        // `scratch_members + idx * num_points`.
        // Thread `idx` accesses `base + idx * N`.
        // Thread `idx+1` accesses `base + (idx+1) * N`.
        // This is strided access with stride N. Very bad for coalescence.
        // Ideally we want `members[k]` for all threads to be adjacent.
        // Struct-of-Arrays (SoA) layout for scratch memory?
        // `scratch_members[k * num_seeds + idx]`
        // Then `members[i]` access would be `scratch_members[i * num_seeds + idx]`.
        // This is perfectly coalesced!
        // `num_seeds` (batch_size) is usually large (e.g. 1000+).
        // This will improve memory bandwidth usage significantly.

        // Process seeds in batches
        int num_seeds_total = static_cast<int>(unclustered_indices.size());
        int seeds_processed = 0;
        
        while (seeds_processed < num_seeds_total) {

            int batch_size = std::min((int)max_seeds_per_batch, num_seeds_total - seeds_processed);
            
            // Copy seeds for this batch to the beginning of d_seeds buffer (reusing it as batch buffer)
            // Or use d_seeds as full buffer?
            // d_seeds is allocated as size N. That is enough for all seeds.
            // But kernel expects `seeds` array to be indexed by `idx` (0..batch_size-1).
            // So we should copy the current batch of seeds to `d_seeds`.
            CHECK_CUDA(cudaMemcpy(d_seeds, unclustered_indices.data() + seeds_processed, 
                                  batch_size * sizeof(int), cudaMemcpyHostToDevice));

            int threadsPerBlock = 256;
            int blocksPerGrid = (batch_size + threadsPerBlock - 1) / threadsPerBlock;
            
            compute_candidate_cluster_size_kernel<<<blocksPerGrid, threadsPerBlock>>>(
                d_seeds,
                batch_size,
                d_points,
                d_clustered,
                N,
                threshold,
                d_results + seeds_processed, // Write results to correct offset in global results array
                d_scratch_members,
                d_scratch_in_cluster
            );
            CHECK_CUDA(cudaGetLastError());

            seeds_processed += batch_size;
        }
        CHECK_CUDA(cudaDeviceSynchronize());

        // Copy results back
        std::vector<int> results(num_seeds_total);
        CHECK_CUDA(cudaMemcpy(results.data(), d_results, num_seeds_total * sizeof(int), cudaMemcpyDeviceToHost));

        // Find max cardinality
        for (int i = 0; i < num_seeds_total; ++i) {
             if (results[i] > max_cardinality) {
                 max_cardinality = results[i];
                 best_seed = unclustered_indices[i]; // Use original index from unclustered_indices
             }
        }
        
        // If we found a cluster, add it
        if (best_seed >= 0 && max_cardinality > 0) {
            Cluster cluster;
            cluster.seed_point = best_seed;
            
            // Re-generate members on CPU (fast for single seed)
            std::vector<int> best_cluster_members;
            generateCandidateClusterCPU(best_seed, clustered, points, threshold, N, &best_cluster_members);
            
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

    CHECK_CUDA(cudaFree(d_points));
    CHECK_CUDA(cudaFree(d_clustered));
    CHECK_CUDA(cudaFree(d_seeds));
    CHECK_CUDA(cudaFree(d_results));
    CHECK_CUDA(cudaFree(d_scratch_members));
    CHECK_CUDA(cudaFree(d_scratch_in_cluster));
    
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
