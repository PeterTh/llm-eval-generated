// QT Clustering Benchmark - MPI Parallel Version
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// MPI Parallelization:
// - Seed evaluation is distributed across ranks (round-robin assignment)
// - MPI_Allreduce finds the global best seed each iteration
// - Distance matrix precomputed and replicated on all ranks
// - Incremental max-distance tracking for O(k*N) candidate evaluation

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

// Calculate Euclidean distance between two points
inline double distance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

// Main QT clustering algorithm - MPI parallelized
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    const int N = static_cast<int>(points.size());
    const double threshold_sq = threshold * threshold;

    // Precompute squared distance matrix for all point pairs
    std::vector<double> dist_sq(N * N);
    for (int i = 0; i < N; ++i) {
        dist_sq[i * N + i] = 0.0;
        for (int j = i + 1; j < N; ++j) {
            double dx = points[i].x - points[j].x;
            double dy = points[i].y - points[j].y;
            double d2 = dx * dx + dy * dy;
            dist_sq[i * N + j] = d2;
            dist_sq[j * N + i] = d2;
        }
    }

    std::vector<char> clustered(N, 0);
    std::vector<int> unclustered_indices;
    unclustered_indices.reserve(N);
    std::vector<Cluster> clusters;

    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }

    // Pre-allocated working buffers to avoid repeated allocation
    std::vector<char> in_cluster(N, 0);
    std::vector<int> members;
    members.reserve(N);
    std::vector<double> max_dist_sq(N);

    // Main clustering loop
    while (!unclustered_indices.empty()) {
        int local_max_cardinality = 0;
        int local_best_seed = -1;
        int local_best_pos = static_cast<int>(unclustered_indices.size());
        std::vector<int> local_best_members;

        // Each rank evaluates a subset of seeds (round-robin)
        for (size_t i = 0; i < unclustered_indices.size(); ++i) {
            const int seed = unclustered_indices[i];
            if (clustered[seed]) continue;

            // Round-robin: only process seeds assigned to this rank
            if (static_cast<int>(i % nprocs) != rank) continue;

            // Reset working buffers
            std::fill(in_cluster.begin(), in_cluster.end(), 0);
            std::fill(max_dist_sq.begin(), max_dist_sq.end(), 0.0);
            members.clear();

            // Add seed point
            in_cluster[seed] = 1;
            members.push_back(seed);

            // Initialize max_dist_sq for all candidates based on seed
            const double* seed_dists = &dist_sq[seed * N];
            for (int c = 0; c < N; ++c) {
                if (!clustered[c] && c != seed) {
                    max_dist_sq[c] = seed_dists[c];
                }
            }

            // Iteratively add closest points maintaining diameter < threshold
            while (static_cast<int>(members.size()) < N) {
                int best_candidate = -1;
                double best_max_dsq = threshold_sq;

                // Find candidate with minimum max-distance to current cluster members
                // that still satisfies the diameter constraint
                for (int c = 0; c < N; ++c) {
                    if (clustered[c] || in_cluster[c]) continue;
                    if (max_dist_sq[c] >= threshold_sq) continue;
                    if (max_dist_sq[c] < best_max_dsq) {
                        best_max_dsq = max_dist_sq[c];
                        best_candidate = c;
                    }
                }

                if (best_candidate < 0) break;

                in_cluster[best_candidate] = 1;
                members.push_back(best_candidate);

                // Incrementally update max_dist_sq with new member
                const double* new_member_dists = &dist_sq[best_candidate * N];
                for (int c = 0; c < N; ++c) {
                    if (!clustered[c] && !in_cluster[c]) {
                        double d2 = new_member_dists[c];
                        if (d2 > max_dist_sq[c]) {
                            max_dist_sq[c] = d2;
                        }
                    }
                }
            }

            const int cardinality = static_cast<int>(members.size());
            // Tie-break: prefer the seed with lower index in unclustered_indices
            // to match sequential behavior (first encountered wins)
            if (cardinality > local_max_cardinality ||
                (cardinality == local_max_cardinality && static_cast<int>(i) < local_best_pos)) {
                local_max_cardinality = cardinality;
                local_best_seed = seed;
                local_best_pos = static_cast<int>(i);
                local_best_members = members;
            }
        }

        // MPI_Allreduce to find global best seed across all ranks
        int global_max_cardinality = 0;
        MPI_Allreduce(&local_max_cardinality, &global_max_cardinality, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

        if (global_max_cardinality <= 0) break;

        // Find which rank has the best seed with the lowest position
        int has_best = (local_max_cardinality == global_max_cardinality && local_best_seed >= 0) ? 1 : 0;
        int global_has_best = 0;
        MPI_Allreduce(&has_best, &global_has_best, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        if (global_has_best == 0) break;

        // Determine the winning rank (rank with the best cardinality and lowest position)
        int my_pos = has_best ? local_best_pos : std::numeric_limits<int>::max();
        int global_min_pos = 0;
        MPI_Allreduce(&my_pos, &global_min_pos, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        
        int global_best_rank = (has_best && local_best_pos == global_min_pos) ? rank : nprocs;
        int final_best_rank = nprocs;
        MPI_Allreduce(&global_best_rank, &final_best_rank, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

        // Broadcast the best seed and its members from the winning rank
        int best_seed = local_best_seed;
        MPI_Bcast(&best_seed, 1, MPI_INT, final_best_rank, MPI_COMM_WORLD);

        // Broadcast the cluster members
        int members_size = static_cast<int>(local_best_members.size());
        MPI_Bcast(&members_size, 1, MPI_INT, final_best_rank, MPI_COMM_WORLD);

        std::vector<int> best_members(members_size);
        if (rank == final_best_rank) {
            best_members = local_best_members;
        }
        MPI_Bcast(best_members.data(), members_size, MPI_INT, final_best_rank, MPI_COMM_WORLD);

        // All ranks update their clustered state identically
        Cluster cluster;
        cluster.seed_point = best_seed;
        cluster.members = best_members;
        clusters.push_back(cluster);

        for (int m : best_members) {
            clustered[m] = 1;
        }

        // Remove clustered points from unclustered list
        unclustered_indices.erase(
            std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                          [&clustered](int idx) { return clustered[idx]; }),
            unclustered_indices.end()
        );
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
    
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    
    int num_points = 1000;
    double threshold = 2.0;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments (all ranks parse identically)
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
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
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
        printf("QT Clustering Benchmark (MPI, %d ranks)\n", nprocs);
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Generate synthetic data (identical on all ranks)
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);
    
    // Synchronize before timing
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Perform QT clustering
    double cluster_start = MPI_Wtime();
    
    const std::vector<Cluster> clusters = qtClustering(points, threshold);
    
    double cluster_end = MPI_Wtime();
    double cluster_time_sec = cluster_end - cluster_start;
    
    // Reduce timing to rank 0 for reporting
    double max_cluster_time = 0.0;
    MPI_Reduce(&cluster_time_sec, &max_cluster_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        long long cluster_time_ms = static_cast<long long>(max_cluster_time * 1000.0);
        printf("Clustering time: %lld ms\n", cluster_time_ms);
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
        const double clusters_per_sec = (max_cluster_time > 0.0) ? clusters.size() / max_cluster_time : 0.0;
        const double points_per_sec = (max_cluster_time > 0.0) ? num_points / max_cluster_time : 0.0;
        printf("Performance: %.1f clusters/s, %.1f points/s\n", 
               clusters_per_sec, points_per_sec);
    }
    
    // Print results for external validation (rank 0 only)
    if (printResults && rank == 0) {
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
    
    // Validation (rank 0 only)
    int valid_flag = 0;
    if (validate) {
        if (rank == 0) {
            const bool valid = validateClusters(clusters, points, threshold);
            valid_flag = valid ? 1 : 0;
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
        MPI_Bcast(&valid_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    
    MPI_Finalize();
    return validate ? (valid_flag ? 0 : 1) : 0;
}
