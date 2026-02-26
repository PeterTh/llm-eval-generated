// QT Clustering Benchmark - Hybrid Parallel Version
// 
// Parallelized with MPI (process-level), OpenMP (thread-level), and CUDA (GPU-level)
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

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#endif

#include "../common/results_output.hpp"

static const double MAX_WIDTH = 20.0;
static const double MAX_HEIGHT = 20.0;

// Global MPI rank and size
static int mpi_rank = 0;
static int mpi_size = 1;

#ifdef ENABLE_CUDA
// CUDA kernel for batch distance calculations
__global__ void cuda_compute_distances(const double* candidates_x, const double* candidates_y,
                                       const double* members_x, const double* members_y,
                                       int num_candidates, int num_members,
                                       double* max_distances) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_candidates) return;
    
    double max_dist = 0.0;
    double cx = candidates_x[idx];
    double cy = candidates_y[idx];
    
    for (int j = 0; j < num_members; ++j) {
        double dx = cx - members_x[j];
        double dy = cy - members_y[j];
        double dist = sqrt(dx * dx + dy * dy);
        max_dist = fmax(max_dist, dist);
    }
    
    max_distances[idx] = max_dist;
}
#endif

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
    
    // Collect candidates (unclustered points not in this cluster)
    std::vector<int> candidates;
    for (int candidate = 0; candidate < point_count; ++candidate) {
        if (!clustered[candidate] && !in_cluster[candidate]) {
            candidates.push_back(candidate);
        }
    }
    
    if (candidates.empty()) {
        return -1;
    }

#ifdef ENABLE_CUDA
    // Use CUDA for GPU acceleration
    int num_candidates = static_cast<int>(candidates.size());
    int num_members = static_cast<int>(cluster_members.size());
    
    // Prepare data on host
    std::vector<double> candidates_x(num_candidates), candidates_y(num_candidates);
    std::vector<double> members_x(num_members), members_y(num_members);
    
    for (int i = 0; i < num_candidates; ++i) {
        candidates_x[i] = points[candidates[i]].x;
        candidates_y[i] = points[candidates[i]].y;
    }
    for (int i = 0; i < num_members; ++i) {
        members_x[i] = points[cluster_members[i]].x;
        members_y[i] = points[cluster_members[i]].y;
    }
    
    // Allocate GPU memory
    double *d_candidates_x, *d_candidates_y, *d_members_x, *d_members_y, *d_max_distances;
    cudaMalloc(&d_candidates_x, num_candidates * sizeof(double));
    cudaMalloc(&d_candidates_y, num_candidates * sizeof(double));
    cudaMalloc(&d_members_x, num_members * sizeof(double));
    cudaMalloc(&d_members_y, num_members * sizeof(double));
    cudaMalloc(&d_max_distances, num_candidates * sizeof(double));
    
    // Copy data to GPU
    cudaMemcpy(d_candidates_x, candidates_x.data(), num_candidates * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_candidates_y, candidates_y.data(), num_candidates * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_members_x, members_x.data(), num_members * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_members_y, members_y.data(), num_members * sizeof(double), cudaMemcpyHostToDevice);
    
    // Launch kernel
    int block_size = 256;
    int grid_size = (num_candidates + block_size - 1) / block_size;
    cuda_compute_distances<<<grid_size, block_size>>>(
        d_candidates_x, d_candidates_y,
        d_members_x, d_members_y,
        num_candidates, num_members,
        d_max_distances
    );
    
    // Copy results back
    std::vector<double> max_distances(num_candidates);
    cudaMemcpy(max_distances.data(), d_max_distances, num_candidates * sizeof(double), cudaMemcpyDeviceToHost);
    
    // Free GPU memory
    cudaFree(d_candidates_x);
    cudaFree(d_candidates_y);
    cudaFree(d_members_x);
    cudaFree(d_members_y);
    cudaFree(d_max_distances);
    
    // Find best candidate
    for (int i = 0; i < num_candidates; ++i) {
        if (max_distances[i] < threshold && max_distances[i] < min_diameter) {
            min_diameter = max_distances[i];
            closest_point = candidates[i];
        }
    }
    
#else
    // CPU version with OpenMP parallelization
    #pragma omp parallel for reduction(min:min_diameter) reduction(max:closest_point)
    for (size_t i = 0; i < candidates.size(); ++i) {
        const int candidate = candidates[i];
        
        // Calculate the maximum distance from candidate to all cluster members
        double max_dist = 0.0;
        #pragma omp parallel for reduction(max:max_dist)
        for (int j = 0; j < static_cast<int>(cluster_members.size()); ++j) {
            const int member = cluster_members[j];
            const double dist = distance(points[candidate], points[member]);
            max_dist = std::max(max_dist, dist);
        }
        
        // If adding this point keeps diameter below threshold and is better than current best
        if (max_dist < threshold && max_dist < min_diameter) {
            #pragma omp critical
            {
                if (max_dist < min_diameter) {
                    min_diameter = max_dist;
                    closest_point = candidate;
                }
            }
        }
    }
#endif
    
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

// Main QT clustering algorithm with MPI parallelization
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
        
        // MPI: Distribute seed evaluation across ranks
        int local_max_cardinality = -1;
        int local_best_seed = -1;
        std::vector<int> local_best_cluster_members;
        
        // Each rank evaluates a subset of seeds
        int seeds_per_rank = (unclustered_indices.size() + mpi_size - 1) / mpi_size;
        int start_idx = mpi_rank * seeds_per_rank;
        int end_idx = std::min(start_idx + seeds_per_rank, static_cast<int>(unclustered_indices.size()));
        
        // Try assigned seeds with OpenMP parallelization
        #pragma omp parallel for reduction(max:local_max_cardinality)
        for (int i = start_idx; i < end_idx; ++i) {
            const int seed = unclustered_indices[i];
            if (clustered[seed]) continue;
            
            std::vector<int> candidate_members;
            const int cardinality = generateCandidateCluster(seed, clustered, points, 
                                                       threshold, N, 
                                                       &candidate_members);
            
            if (cardinality > local_max_cardinality) {
                #pragma omp critical
                {
                    if (cardinality > local_max_cardinality) {
                        local_max_cardinality = cardinality;
                        local_best_seed = seed;
                        local_best_cluster_members = candidate_members;
                    }
                }
            }
        }
        
        // MPI_Allreduce to find global best cluster
        struct {
            int cardinality;
            int rank;
        } local_result, global_result;
        
        local_result.cardinality = local_max_cardinality;
        local_result.rank = mpi_rank;
        
        MPI_Allreduce(&local_result, &global_result, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);
        
        // Broadcast the best cluster from the winning rank
        int broadcast_seed = (global_result.rank == mpi_rank) ? local_best_seed : -1;
        std::vector<int> broadcast_members;
        if (global_result.rank == mpi_rank) {
            broadcast_members = local_best_cluster_members;
        }
        
        MPI_Bcast(&broadcast_seed, 1, MPI_INT, global_result.rank, MPI_COMM_WORLD);
        
        int members_count = broadcast_members.size();
        MPI_Bcast(&members_count, 1, MPI_INT, global_result.rank, MPI_COMM_WORLD);
        
        if (members_count > 0) {
            broadcast_members.resize(members_count);
            if (global_result.rank == mpi_rank) {
                broadcast_members = local_best_cluster_members;
            }
            MPI_Bcast(broadcast_members.data(), members_count, MPI_INT, 
                     global_result.rank, MPI_COMM_WORLD);
        }
        
        max_cardinality = global_result.cardinality;
        best_seed = broadcast_seed;
        best_cluster_members = broadcast_members;
        
        // If we found a cluster, add it
        if (best_seed >= 0 && max_cardinality > 0) {
            Cluster cluster;
            cluster.seed_point = best_seed;
            cluster.members = best_cluster_members;
            clusters.push_back(cluster);
            
            // Mark all members as clustered (synchronized across MPI ranks)
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
    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    
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
            if (mpi_rank == 0) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            if (mpi_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    
    if (num_points <= 0 || threshold <= 0.0) {
        if (mpi_rank == 0) {
            printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n", 
                   num_points, threshold);
        }
        MPI_Finalize();
        return 1;
    }
    
    if (mpi_rank == 0) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Generate synthetic data
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);
    
    // Perform QT clustering
    auto cluster_start = std::chrono::high_resolution_clock::now();
    
    const std::vector<Cluster> clusters = qtClustering(points, threshold);
    
    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);
    
    if (mpi_rank == 0) {
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
    }
    
    // Validation (run on all ranks for consistency check)
    int validation_result = 0;
    if (validate) {
        const bool valid = validateClusters(clusters, points, threshold);
        validation_result = valid ? 0 : 1;
    }
    
    MPI_Finalize();
    
    return validation_result;
}
