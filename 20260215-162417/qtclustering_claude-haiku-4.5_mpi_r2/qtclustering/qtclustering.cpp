// QT Clustering Benchmark - MPI Distributed Version
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

#include "../common/results_output.hpp"

static const double MAX_WIDTH = 20.0;
static const double MAX_HEIGHT = 20.0;

// MPI utility structures
struct CandidateResult {
    int seed_point;
    int cardinality;
    // Fixed-size array for cluster members (max points per rank)
    int members[10000];  // Will be resized based on needs
};

// Global MPI datatype for candidate results
static MPI_Datatype MPI_CANDIDATE_RESULT = MPI_DATATYPE_NULL;

void create_mpi_datatype() {
    if (MPI_CANDIDATE_RESULT != MPI_DATATYPE_NULL) return;
    
    MPI_Datatype types[3] = {MPI_INT, MPI_INT, MPI_INT};
    int lengths[3] = {1, 1, 10000};
    MPI_Aint offsets[3];
    
    CandidateResult dummy{};
    MPI_Get_address(&dummy.seed_point, &offsets[0]);
    MPI_Get_address(&dummy.cardinality, &offsets[1]);
    MPI_Get_address(&dummy.members, &offsets[2]);
    
    offsets[1] -= offsets[0];
    offsets[2] -= offsets[0];
    offsets[0] = 0;
    
    MPI_Type_create_struct(3, lengths, offsets, types, &MPI_CANDIDATE_RESULT);
    MPI_Type_commit(&MPI_CANDIDATE_RESULT);
}

// Custom reduction function for finding best candidate
void reduce_candidates(void* in, void* inout, int* len, MPI_Datatype* /* dtype */) {
    CandidateResult* in_res = static_cast<CandidateResult*>(in);
    CandidateResult* inout_res = static_cast<CandidateResult*>(inout);
    
    for (int i = 0; i < *len; ++i) {
        if (in_res[i].cardinality > inout_res[i].cardinality) {
            inout_res[i] = in_res[i];
        } else if (in_res[i].cardinality == inout_res[i].cardinality && 
                  in_res[i].seed_point >= 0 && in_res[i].seed_point < inout_res[i].seed_point) {
            inout_res[i].seed_point = in_res[i].seed_point;
        }
    }
}

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

// Main QT clustering algorithm
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    
    const int N = static_cast<int>(points.size());
    std::vector<int> clustered_int(N, 0);  // Use int instead of bool for MPI compatibility
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;
    
    // Initialize unclustered indices
    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }
    
    // Main clustering loop
    while (!unclustered_indices.empty()) {
        // Synchronize clustered status across all processes
        MPI_Bcast(clustered_int.data(), N, MPI_INT, 0, MPI_COMM_WORLD);
        
        // Update unclustered indices on all processes (for consistent seed selection)
        int num_unclustered = static_cast<int>(unclustered_indices.size());
        MPI_Bcast(&num_unclustered, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        if (num_unclustered == 0) break;
        
        unclustered_indices.clear();
        unclustered_indices.resize(num_unclustered);
        
        if (mpi_rank == 0) {
            int idx = 0;
            for (int i = 0; i < N; ++i) {
                if (!clustered_int[i]) {
                    unclustered_indices[idx++] = i;
                }
            }
        }
        
        MPI_Bcast(unclustered_indices.data(), num_unclustered, MPI_INT, 0, MPI_COMM_WORLD);
        
        // Parallel seed evaluation: each process tries a subset of seeds
        int max_cardinality_local = -1;
        int best_seed_local = -1;
        std::vector<int> best_cluster_members_local;
        
        // Distribute seeds across processes
        int seeds_per_process = (num_unclustered + mpi_size - 1) / mpi_size;
        int start_idx = mpi_rank * seeds_per_process;
        int end_idx = std::min((mpi_rank + 1) * seeds_per_process, num_unclustered);
        
        // Convert clustered_int back to bool for function calls
        std::vector<bool> clustered_bool(clustered_int.begin(), clustered_int.end());
        
        // Each process evaluates its subset of seeds
        for (int i = start_idx; i < end_idx; ++i) {
            if (i >= num_unclustered) break;
            
            const int seed = unclustered_indices[i];
            if (clustered_bool[seed]) continue;
            
            std::vector<int> candidate_members;
            const int cardinality = generateCandidateCluster(seed, clustered_bool, points, 
                                                       threshold, N, 
                                                       &candidate_members);
            
            if (cardinality > max_cardinality_local) {
                max_cardinality_local = cardinality;
                best_seed_local = seed;
                best_cluster_members_local = candidate_members;
            }
        }
        
        // Gather best results from all processes using custom reduction
        CandidateResult local_result;
        local_result.seed_point = best_seed_local;
        local_result.cardinality = max_cardinality_local;
        
        CandidateResult global_result;
        
        MPI_Op reduce_op;
        MPI_Op_create((MPI_User_function*)reduce_candidates, 1, &reduce_op);
        
        MPI_Allreduce(&local_result, &global_result, 1, MPI_CANDIDATE_RESULT, 
                      reduce_op, MPI_COMM_WORLD);
        
        MPI_Op_free(&reduce_op);
        
        // If we found a cluster on rank 0, add it
        if (mpi_rank == 0) {
            if (global_result.seed_point >= 0 && global_result.cardinality > 0) {
                // Reconstruct members for best cluster
                std::vector<int> candidate_members;
                generateCandidateCluster(global_result.seed_point, clustered_bool, points,
                                       threshold, N, &candidate_members);
                
                Cluster cluster;
                cluster.seed_point = global_result.seed_point;
                cluster.members = candidate_members;
                clusters.push_back(cluster);
                
                // Mark all members as clustered
                for (size_t i = 0; i < candidate_members.size(); ++i) {
                    clustered_int[candidate_members[i]] = 1;
                }
                
                // Remove clustered points from unclustered list
                unclustered_indices.erase(
                    std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                                  [&clustered_int](int idx) { return clustered_int[idx]; }),
                    unclustered_indices.end()
                );
            } else {
                // No more clusters can be formed
                break;
            }
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
    // Initialize MPI unconditionally
    MPI_Init(&argc, &argv);
    
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    
    // Create MPI datatype for candidate results
    create_mpi_datatype();
    
    int num_points = 1000;
    double threshold = 2.0;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments (on all ranks for consistency)
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
        printf("QT Clustering Benchmark (MPI with %d processes)\n", mpi_size);
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Generate synthetic data (all processes generate the same data independently)
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
    }
    
    // Calculate statistics and performance metrics
    int total_clustered = 0;
    int max_cluster_size = 0;
    
    if (mpi_rank == 0) {
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
    }
    
    // Print results for external validation (only on rank 0)
    if (mpi_rank == 0 && printResults) {
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
        
        if (mpi_rank == 0) {
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
        
        MPI_Finalize();
        return valid ? 0 : 1;
    }
    
    MPI_Finalize();
    return 0;
}
