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

// CUDA Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            MPI_Abort(MPI_COMM_WORLD, 1); \
        } \
    } while (0)

static const double MAX_WIDTH = 20.0;
static const double MAX_HEIGHT = 20.0;

// Structure to represent a point in 2D space
struct Point {
    double x, y;
};

// Device function for distance
__device__ inline double device_distance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return sqrt(dx * dx + dy * dy);
}

// Kernel to compute candidate cluster sizes for a batch of seeds
__global__ void compute_cluster_sizes_kernel(
    const Point* points,
    const int* clustered,
    const int* seeds,
    int* sizes,
    int num_points,
    int num_seeds,
    double threshold,
    int* workspace_members,
    int* workspace_in_cluster
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_seeds) return;

    int seed = seeds[idx];
    
    // Each thread needs its own workspace area
    int* my_members = &workspace_members[idx * num_points];
    int* my_in_cluster = &workspace_in_cluster[idx * num_points];
    
    // Initialize my_in_cluster
    for (int i = 0; i < num_points; ++i) {
        my_in_cluster[i] = false;
    }
    
    // Add seed
    my_in_cluster[seed] = true;
    my_members[0] = seed;
    int current_size = 1;
    
    while (current_size < num_points) {
        int best_candidate = -1;
        double min_max_dist = 1e20; // Large value
        
        // Try each unclustered point
        for (int p = 0; p < num_points; ++p) {
            // Optimization: clustered check is fast
            if (clustered[p] || my_in_cluster[p]) continue;
            
            // Optimization: Early exit if p is too far from seed?
            // Distance to seed is always checked first (m=0).
            // If dist(p, seed) > threshold, then max_dist > threshold.
            // So we can skip.
            // This prunes search space significantly.
            double d_seed = device_distance(points[p], points[seed]);
            if (d_seed > threshold) continue;

            double max_dist = d_seed;
            // Check other members only if necessary
            for (int m = 1; m < current_size; ++m) {
                double d = device_distance(points[p], points[my_members[m]]);
                if (d > max_dist) max_dist = d;
                if (max_dist > threshold) break; // Early exit inner loop
            }
            
            if (max_dist < threshold && max_dist < min_max_dist) {
                min_max_dist = max_dist;
                best_candidate = p;
            }
        }
        
        if (best_candidate != -1) {
            my_in_cluster[best_candidate] = true;
            my_members[current_size] = best_candidate;
            current_size++;
        } else {
            break;
        }
    }
    
    sizes[idx] = current_size;
}

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
                     const std::vector<int>& clustered,
                     const std::vector<int>& in_cluster,
                     const std::vector<Point>& points,
                     const double threshold,
                     const int point_count) {
    int closest_point = -1;
    double min_diameter = std::numeric_limits<double>::max();
    
    // Try each unclustered point as a candidate
    for (int candidate = 0; candidate < point_count; ++candidate) {
        // Skip if already clustered or already in this cluster
        if (clustered[candidate] || in_cluster[candidate]) continue;
        
        // Optimization: Early exit if candidate is too far from seed (first member)
        // seed point is members[0]
        double d_seed = distance(points[candidate], points[cluster_members[0]]);
        if (d_seed > threshold) continue;
        
        // Calculate the maximum distance from candidate to all cluster members
        double max_dist = d_seed;
        // Start from 1 since we checked 0
        for (size_t i = 1; i < cluster_members.size(); ++i) {
            const int member = cluster_members[i];
            const double dist = distance(points[candidate], points[member]);
            max_dist = std::max(max_dist, dist);
            if (max_dist > threshold) break; // Optimization: early exit
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
                              const std::vector<int>& clustered,
                              const std::vector<Point>& points,
                              const double threshold,
                              const int point_count,
                              std::vector<int>* cluster_members = nullptr) {
    std::vector<int> in_cluster(point_count, false);
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
    std::vector<int> clustered(N, false);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;
    
    // MPI Info
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Initialize unclustered indices
    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }

    // CUDA Setup
    bool use_cuda = false;
    Point* d_points = nullptr;
    int* d_clustered = nullptr;
    int* d_seeds = nullptr;
    int* d_sizes = nullptr;
    int* d_workspace_members = nullptr;
    int* d_workspace_in_cluster = nullptr;
    int max_local_seeds = 0;

    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count > 0) {
        cudaSetDevice(rank % device_count); 
        use_cuda = true;
        if (rank == 0) printf("Rank 0 using CUDA acceleration.\n");

        CUDA_CHECK(cudaMalloc(&d_points, N * sizeof(Point)));
        CUDA_CHECK(cudaMemcpy(d_points, points.data(), N * sizeof(Point), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_clustered, N * sizeof(int)));
        
        max_local_seeds = (N + size - 1) / size + 16; 
        
        CUDA_CHECK(cudaMalloc(&d_seeds, max_local_seeds * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_sizes, max_local_seeds * sizeof(int)));
        
        CUDA_CHECK(cudaMalloc(&d_workspace_members, (size_t)max_local_seeds * N * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_workspace_in_cluster, (size_t)max_local_seeds * N * sizeof(int)));
    } else {
        if (rank == 0) printf("Rank 0 using CPU only (OpenMP).\n");
    }
    
    // Main clustering loop
    while (true) {
        // Rebuild unclustered list locally to ensure consistency
        unclustered_indices.clear();
        for(int i=0; i<N; ++i) {
            if (!clustered[i]) unclustered_indices.push_back(i);
        }

        if (unclustered_indices.empty()) break;

        // Distribute seeds
        int total_unclustered = static_cast<int>(unclustered_indices.size());
        int seeds_per_rank = total_unclustered / size;
        int remainder = total_unclustered % size;
        
        int my_start = rank * seeds_per_rank + std::min(rank, remainder);
        int my_count = seeds_per_rank + (rank < remainder ? 1 : 0);
        
        // Local computation
        int local_max_cardinality = -1;
        int local_best_seed = -1;
        
        if (my_count > 0) {
            if (use_cuda) {
                // Prepare data
                std::vector<int> my_seeds(my_count);
                for(int i=0; i<my_count; ++i) my_seeds[i] = unclustered_indices[my_start + i];
                
                CUDA_CHECK(cudaMemcpy(d_clustered, clustered.data(), N * sizeof(int), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_seeds, my_seeds.data(), my_count * sizeof(int), cudaMemcpyHostToDevice));
                
                // Launch kernel
                int threads = 128;
                int blocks = (my_count + threads - 1) / threads;
                compute_cluster_sizes_kernel<<<blocks, threads>>>(
                    d_points, d_clustered, d_seeds, d_sizes, 
                    N, my_count, threshold, 
                    d_workspace_members, d_workspace_in_cluster
                );
                CUDA_CHECK(cudaGetLastError());
                
                // Copy results back
                std::vector<int> h_sizes(my_count);
                CUDA_CHECK(cudaMemcpy(h_sizes.data(), d_sizes, my_count * sizeof(int), cudaMemcpyDeviceToHost));
                
                // Find local best
                for(int i=0; i<my_count; ++i) {
                    if (h_sizes[i] > local_max_cardinality) {
                        local_max_cardinality = h_sizes[i];
                        local_best_seed = my_seeds[i];
                    }
                }

            } else {
                // CPU OpenMP
                #pragma omp parallel
                {
                    int thread_max_card = -1;
                    int thread_best_seed = -1;

                    #pragma omp for nowait
                    for (int i = 0; i < my_count; ++i) {
                        int seed = unclustered_indices[my_start + i];
                        int card = generateCandidateCluster(seed, clustered, points, threshold, N);
                        if (card > thread_max_card) {
                            thread_max_card = card;
                            thread_best_seed = seed;
                        }
                    }
                    
                    #pragma omp critical
                    {
                        if (thread_max_card > local_max_cardinality) {
                            local_max_cardinality = thread_max_card;
                            local_best_seed = thread_best_seed;
                        }
                    }
                }
            }
        }
        
        // Global Reduction
        struct { int val; int rank; } local_res, global_res;
        local_res.val = local_max_cardinality;
        local_res.rank = rank;
        
        MPI_Allreduce(&local_res, &global_res, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);
        
        if (global_res.val <= 0) break;

        // Broadcast best seed
        int best_seed = -1;
        if (rank == global_res.rank) {
            best_seed = local_best_seed;
        }
        MPI_Bcast(&best_seed, 1, MPI_INT, global_res.rank, MPI_COMM_WORLD);
        
        // Reconstruct best cluster
        std::vector<int> best_members;
        if (rank == global_res.rank) {
            generateCandidateCluster(best_seed, clustered, points, threshold, N, &best_members);
        }
        
        // Broadcast members
        int member_count = global_res.val; 
        best_members.resize(member_count);
        MPI_Bcast(best_members.data(), member_count, MPI_INT, global_res.rank, MPI_COMM_WORLD);
        
        // Update clusters and clustered array
        Cluster cluster;
        cluster.seed_point = best_seed;
        cluster.members = best_members;
        clusters.push_back(cluster);
        
        for (int m : best_members) {
            clustered[m] = true;
        }
    }
    
    // Cleanup CUDA
    if (use_cuda) {
        cudaFree(d_points);
        cudaFree(d_clustered);
        cudaFree(d_seeds);
        cudaFree(d_sizes);
        cudaFree(d_workspace_members);
        cudaFree(d_workspace_in_cluster);
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
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

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
        if (rank == 0) printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n", 
               num_points, threshold);
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
    std::vector<Point> points(num_points);
    // All ranks generate same data because seed is constant (42) in generateSyntheticData
    generateSyntheticData(points, num_points);
    
    // Perform QT clustering
    auto cluster_start = std::chrono::high_resolution_clock::now();
    
    const std::vector<Cluster> clusters = qtClustering(points, threshold);
    
    auto cluster_end = std::chrono::high_resolution_clock::now();
    const long long local_cluster_time_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(cluster_end - cluster_start).count();
    long long global_cluster_time_ns = 0;
    MPI_Reduce(&local_cluster_time_ns, &global_cluster_time_ns, 1,
               MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::nanoseconds(global_cluster_time_ns));
        
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
            }
        }
    }
    
    MPI_Finalize();
    return 0;
}
