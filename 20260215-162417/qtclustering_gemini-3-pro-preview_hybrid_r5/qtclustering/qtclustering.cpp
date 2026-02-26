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
#include <mpi.h>
#include <omp.h>
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

// CUDA Kernel to compute cluster sizes for a batch of seeds
__global__ void compute_cluster_sizes_kernel(
    const Point* points,
    const char* clustered,
    const int* seeds,
    int* sizes,
    int* scratch_members,
    char* scratch_in_cluster,
    int num_seeds,
    int num_points,
    double threshold
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_seeds) return;

    int seed_point = seeds[idx];
    
    // Calculate offsets for this thread's scratch space
    int* members = &scratch_members[idx * num_points];
    char* in_cluster = &scratch_in_cluster[idx * num_points];
    
    // Initialize
    for (int i = 0; i < num_points; ++i) {
        in_cluster[i] = 0;
    }
    
    int member_count = 0;
    
    // Add seed point
    in_cluster[seed_point] = 1;
    members[member_count++] = seed_point;
    
    // Iteratively add closest points
    while (member_count < num_points) {
        int closest_point = -1;
        double min_diameter = 1e30; // Large value
        
        // Try each unclustered point as a candidate
        for (int candidate = 0; candidate < num_points; ++candidate) {
            // Skip if already clustered or already in this cluster
            if (clustered[candidate] || in_cluster[candidate]) continue;
            
            // Calculate the maximum distance from candidate to all cluster members
            double max_dist = 0.0;
            for (int i = 0; i < member_count; ++i) {
                const int member = members[i];
                double dx = points[candidate].x - points[member].x;
                double dy = points[candidate].y - points[member].y;
                double dist = sqrt(dx * dx + dy * dy);
                if (dist > max_dist) max_dist = dist;
            }
            
            // If adding this point keeps diameter below threshold and is better than current best
            if (max_dist < threshold && max_dist < min_diameter) {
                min_diameter = max_dist;
                closest_point = candidate;
            }
        }
        
        if (closest_point < 0) break; // No more points can be added
        
        in_cluster[closest_point] = 1;
        members[member_count++] = closest_point;
    }
    
    sizes[idx] = member_count;
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

// Calculate Euclidean distance between two points
inline double distance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

// Find the closest unclustered point to the current cluster that maintains diameter < threshold
// Returns -1 if no such point exists
int findClosestPoint(const std::vector<int>& cluster_members,
                     const std::vector<char>& clustered,
                     const std::vector<char>& in_cluster,
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
            const double dx = points[candidate].x - points[member].x;
            const double dy = points[candidate].y - points[member].y;
            const double dist = std::sqrt(dx * dx + dy * dy);
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
                              const std::vector<char>& clustered,
                              const std::vector<Point>& points,
                              const double threshold,
                              const int point_count,
                              std::vector<int>* cluster_members = nullptr) {
    std::vector<char> in_cluster(point_count, 0);
    std::vector<int> members;
    
    // Add seed point
    in_cluster[seed_point] = 1;
    members.push_back(seed_point);
    
    // Iteratively add closest points
    while (static_cast<int>(members.size()) < point_count) {
        // Find closest point to current cluster that maintains diameter < threshold
        const int closest = findClosestPoint(members, clustered, in_cluster, points, 
                                             threshold, point_count);
        
        if (closest < 0) break; // No more points can be added
        
        in_cluster[closest] = 1;
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
    std::vector<char> clustered(N, 0);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;
    
    // Initialize unclustered indices
    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }

    // MPI Setup
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // CUDA Setup
    Point* d_points = nullptr;
    char* d_clustered = nullptr;
    int* d_seeds = nullptr;
    int* d_sizes = nullptr;
    int* d_scratch_members = nullptr;
    char* d_scratch_in_cluster = nullptr;

    // Determine max seeds per batch to limit memory usage
    // For N=2000, 2000*2000*4 bytes = 16MB per batch per rank. Fits easily.
    int max_batch_size = 2048; 
    
    CUDA_CHECK(cudaMalloc(&d_points, N * sizeof(Point)));
    CUDA_CHECK(cudaMemcpy(d_points, points.data(), N * sizeof(Point), cudaMemcpyHostToDevice));
    
    CUDA_CHECK(cudaMalloc(&d_clustered, N * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&d_seeds, max_batch_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sizes, max_batch_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_scratch_members, max_batch_size * N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_scratch_in_cluster, max_batch_size * N * sizeof(char)));

    // Main clustering loop
    while (!unclustered_indices.empty()) {
        // Distribute work
        int total_unclustered = unclustered_indices.size();
        
        // Simple partitioning
        int seeds_per_rank = (total_unclustered + size - 1) / size;
        int start_idx = rank * seeds_per_rank;
        int end_idx = std::min(start_idx + seeds_per_rank, total_unclustered);
        
        int local_max_cardinality = -1;
        int local_best_seed = -1;

        if (start_idx < end_idx) {
            std::vector<int> local_seeds;
            for (int i = start_idx; i < end_idx; ++i) {
                local_seeds.push_back(unclustered_indices[i]);
            }

            // Update clustered array on device
            CUDA_CHECK(cudaMemcpy(d_clustered, clustered.data(), N * sizeof(char), cudaMemcpyHostToDevice));

            // Process in batches
            for (size_t batch_start = 0; batch_start < local_seeds.size(); batch_start += max_batch_size) {
                int batch_count = std::min((int)(local_seeds.size() - batch_start), max_batch_size);
                
                CUDA_CHECK(cudaMemcpy(d_seeds, &local_seeds[batch_start], batch_count * sizeof(int), cudaMemcpyHostToDevice));
                
                int threadsPerBlock = 256;
                int blocksPerGrid = (batch_count + threadsPerBlock - 1) / threadsPerBlock;
                
                compute_cluster_sizes_kernel<<<blocksPerGrid, threadsPerBlock>>>(
                    d_points, d_clustered, d_seeds, d_sizes, 
                    d_scratch_members, d_scratch_in_cluster, 
                    batch_count, N, threshold
                );
                CUDA_CHECK(cudaGetLastError());
                
                std::vector<int> host_sizes(batch_count);
                CUDA_CHECK(cudaMemcpy(host_sizes.data(), d_sizes, batch_count * sizeof(int), cudaMemcpyDeviceToHost));
                
                // Find local best in this batch
                for (int i = 0; i < batch_count; ++i) {
                    if (host_sizes[i] > local_max_cardinality) {
                        local_max_cardinality = host_sizes[i];
                        local_best_seed = local_seeds[batch_start + i];
                    }
                }
            }
        }
        
        // Global reduction to find the best seed
        struct {
            int val;
            int rank;
        } local_res, global_res;
        
        local_res.val = local_max_cardinality;
        local_res.rank = rank;
        
        MPI_Allreduce(&local_res, &global_res, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);
        
        int best_seed = -1;
        int max_cardinality = global_res.val;
        
        if (max_cardinality > 0) {
            // The rank that found the max cardinality broadcasts the seed
            if (rank == global_res.rank) {
                best_seed = local_best_seed;
            }
            MPI_Bcast(&best_seed, 1, MPI_INT, global_res.rank, MPI_COMM_WORLD);
            
            // Reconstruct the best cluster on all ranks
            std::vector<int> best_cluster_members;
            generateCandidateCluster(best_seed, clustered, points, threshold, N, &best_cluster_members);
            
            Cluster cluster;
            cluster.seed_point = best_seed;
            cluster.members = best_cluster_members;
            clusters.push_back(cluster);
            
            // Mark all members as clustered
            for (int member : best_cluster_members) {
                clustered[member] = 1;
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

    CUDA_CHECK(cudaFree(d_points));
    CUDA_CHECK(cudaFree(d_clustered));
    CUDA_CHECK(cudaFree(d_seeds));
    CUDA_CHECK(cudaFree(d_sizes));
    CUDA_CHECK(cudaFree(d_scratch_members));
    CUDA_CHECK(cudaFree(d_scratch_in_cluster));
    
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
    
    // Initialize MPI
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
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
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) printf("Unknown option: %s\n", argv[i]);
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 1;
        }
    }
    
    if (num_points <= 0 || threshold <= 0.0) {
        if (rank == 0) {
            printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n", 
                   num_points, threshold);
        }
        MPI_Finalize();
        return 1;
    }
    
    if (rank == 0) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI Size: %d\n", size);
    }
    
    // Generate synthetic data
    // Ensure all ranks generate same data by using same seed
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points, 42); // Explicit seed 42
    
    // Perform QT clustering
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_start = std::chrono::high_resolution_clock::now();
    
    const std::vector<Cluster> clusters = qtClustering(points, threshold);
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);
    
    if (rank == 0) {
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
            } else {
                printf("Validation: FAILED\n");
                MPI_Finalize();
                return 1;
            }
        }
    }
    
    MPI_Finalize();
    return 0;
}
