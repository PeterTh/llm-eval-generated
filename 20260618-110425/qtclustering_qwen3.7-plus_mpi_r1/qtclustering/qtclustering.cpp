// QT Clustering Benchmark - MPI Parallel Version
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// MPI parallelization: the seed evaluation loop in each clustering iteration
// is distributed across ranks via round-robin. MPI_Allreduce (MAXLOC) finds
// the global best seed, and MPI_Bcast shares the winning cluster members.
//
// Optimizations:
// - Precomputed pairwise distance matrix (avoids redundant sqrt)
// - Incremental cluster building with pruning (O(N*|members|) per seed
//   instead of O(N*|members|^2))

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

// Precompute all pairwise Euclidean distances into a flat N*N matrix.
// dist[i * N + j] = distance(points[i], points[j])
void precomputeDistances(const std::vector<Point>& points, const int N,
                        std::vector<double>& dist) {
    dist.resize(static_cast<size_t>(N) * N);
    for (int i = 0; i < N; ++i) {
        dist[static_cast<size_t>(i) * N + i] = 0.0;
        for (int j = i + 1; j < N; ++j) {
            const double dx = points[i].x - points[j].x;
            const double dy = points[i].y - points[j].y;
            const double d = std::sqrt(dx * dx + dy * dy);
            dist[static_cast<size_t>(i) * N + j] = d;
            dist[static_cast<size_t>(j) * N + i] = d;
        }
    }
}

// Generate a candidate cluster starting from a seed point using an incremental
// approach with pruning. For each candidate we maintain the maximum distance
// to any current cluster member; when a new member is added we update these
// values in O(N) and prune candidates whose max distance reaches the threshold.
// This reduces complexity from O(N * |members|^2) to O(N * |members|) per seed.
// Returns the cardinality (size) of the cluster.
int generateCandidateCluster(const int seed_point,
                            const std::vector<char>& clustered,
                            const std::vector<double>& dist,
                            const double threshold,
                            const int point_count,
                            std::vector<int>& cluster_members) {
    std::vector<double> max_dist(point_count, 0.0);
    std::vector<char> eligible(point_count, 0);

    // Initialize: compute distance from seed to every other point
    const double* seed_dists = &dist[static_cast<size_t>(seed_point) * point_count];
    for (int c = 0; c < point_count; ++c) {
        if (!clustered[c] && c != seed_point) {
            const double d = seed_dists[c];
            max_dist[c] = d;
            eligible[c] = (d < threshold) ? 1 : 0;
        }
    }

    cluster_members.clear();
    cluster_members.push_back(seed_point);

    while (static_cast<int>(cluster_members.size()) < point_count) {
        // Find the eligible candidate with minimum max_dist (< threshold)
        int closest = -1;
        double min_d = threshold;
        for (int c = 0; c < point_count; ++c) {
            if (eligible[c] && max_dist[c] < min_d) {
                min_d = max_dist[c];
                closest = c;
            }
        }

        if (closest < 0) break;

        cluster_members.push_back(closest);
        eligible[closest] = 0;  // Mark as no longer eligible (already in cluster)

        // Update max_dist for remaining eligible candidates and prune
        const double* new_dists = &dist[static_cast<size_t>(closest) * point_count];
        for (int c = 0; c < point_count; ++c) {
            if (eligible[c]) {
                const double d = new_dists[c];
                if (d > max_dist[c]) {
                    max_dist[c] = d;
                }
                if (max_dist[c] >= threshold) {
                    eligible[c] = 0;
                }
            }
        }
    }

    return static_cast<int>(cluster_members.size());
}

// Main QT clustering algorithm - MPI parallelized
// Seed evaluations are distributed round-robin across ranks.
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold,
                                  const int rank, const int size) {
    const int N = static_cast<int>(points.size());

    // Precompute pairwise distance matrix (each rank does this independently)
    std::vector<double> dist;
    precomputeDistances(points, N, dist);

    std::vector<char> clustered(N, 0);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;

    unclustered_indices.reserve(N);
    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }

    while (!unclustered_indices.empty()) {
        int local_max_card = -1;
        int local_best_idx = -1;  // index into unclustered_indices
        int local_best_seed = -1;
        std::vector<int> local_best_members;

        const int num_seeds = static_cast<int>(unclustered_indices.size());

        // Each rank processes seeds in round-robin: rank r handles indices r, r+size, r+2*size, ...
        for (int i = rank; i < num_seeds; i += size) {
            const int seed = unclustered_indices[i];
            if (clustered[seed]) continue;

            std::vector<int> candidate_members;
            const int cardinality = generateCandidateCluster(
                seed, clustered, dist, threshold, N, candidate_members);

            if (cardinality > local_max_card) {
                local_max_card = cardinality;
                local_best_idx = i;
                local_best_seed = seed;
                local_best_members = std::move(candidate_members);
            }
        }

        // Global reduction: find the best seed across all ranks.
        // MPI_MAXLOC on (cardinality, -index_in_unclustered):
        //   - max cardinality wins
        //   - for ties, max(-index) = min(index) wins → matches sequential first-found semantics
        struct { double val; int loc; } local_pair, global_pair;
        local_pair.val = static_cast<double>(local_max_card);
        local_pair.loc = (local_best_idx >= 0) ? -local_best_idx : 0;

        MPI_Allreduce(&local_pair, &global_pair, 1, MPI_DOUBLE_INT,
                      MPI_MAXLOC, MPI_COMM_WORLD);

        const int global_max_card = static_cast<int>(global_pair.val);
        if (global_max_card <= 0) break;

        const int global_best_idx = -global_pair.loc;
        const int owner_rank = global_best_idx % size;

        // Broadcast the winning cluster size, members, and seed from the owning rank
        int members_size = (rank == owner_rank)
                               ? static_cast<int>(local_best_members.size())
                               : 0;
        MPI_Bcast(&members_size, 1, MPI_INT, owner_rank, MPI_COMM_WORLD);

        std::vector<int> best_members(members_size);
        if (rank == owner_rank) {
            best_members = std::move(local_best_members);
        }
        MPI_Bcast(best_members.data(), members_size, MPI_INT,
                  owner_rank, MPI_COMM_WORLD);

        int best_seed = (rank == owner_rank) ? local_best_seed : 0;
        MPI_Bcast(&best_seed, 1, MPI_INT, owner_rank, MPI_COMM_WORLD);

        // Record the cluster
        Cluster cluster;
        cluster.seed_point = best_seed;
        cluster.members = std::move(best_members);
        clusters.push_back(std::move(cluster));

        // All ranks update the clustered array identically
        for (const int m : clusters.back().members) {
            clustered[m] = 1;
        }

        // Remove clustered points from unclustered list (order-preserving)
        unclustered_indices.erase(
            std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                           [&clustered](int idx) { return clustered[idx]; }),
            unclustered_indices.end());
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
                const double dx = points[cluster.members[i]].x - points[cluster.members[j]].x;
                const double dy = points[cluster.members[i]].y - points[cluster.members[j]].y;
                const double dist = std::sqrt(dx * dx + dy * dy);
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
    
    // Parse command line arguments (all ranks)
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
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Generate synthetic data (all ranks generate identical data)
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);
    
    // Synchronize before timing
    MPI_Barrier(MPI_COMM_WORLD);
    const double wall_start = MPI_Wtime();

    // Perform QT clustering (parallel)
    const std::vector<Cluster> clusters = qtClustering(points, threshold, rank, size);

    // Synchronize after clustering
    MPI_Barrier(MPI_COMM_WORLD);
    const double wall_end = MPI_Wtime();
    const long cluster_time_ms = static_cast<long>((wall_end - wall_start) * 1000.0);

    if (rank == 0) {
        printf("Clustering time: %ld ms\n", cluster_time_ms);
        printf("Clusters found: %zu\n", clusters.size());
        
        // Calculate statistics and performance metrics
        int total_clustered = 0;
        int max_cluster_size = 0;
        
        for (size_t i = 0; i < clusters.size(); ++i) {
            const int sz = static_cast<int>(clusters[i].members.size());
            total_clustered += sz;
            max_cluster_size = std::max(max_cluster_size, sz);
        }
        
        const double avg_cluster_size = clusters.empty() ? 0.0 : 
            static_cast<double>(total_clustered) / clusters.size();
        
        printf("Points clustered: %d / %d (%.1f%%)\n", 
               total_clustered, num_points, 
               100.0 * total_clustered / num_points);
        printf("Average cluster size: %.2f\n", avg_cluster_size);
        printf("Maximum cluster size: %d\n", max_cluster_size);
        
        // Performance metrics
        const double time_sec = cluster_time_ms / 1000.0;
        const double clusters_per_sec = (time_sec > 0.0) ? clusters.size() / time_sec : 0.0;
        const double points_per_sec = (time_sec > 0.0) ? num_points / time_sec : 0.0;
        printf("Performance: %.1f clusters/s, %.1f points/s\n", 
               clusters_per_sec, points_per_sec);
        
        // Print results for external validation
        if (printResults) {
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
    
    // Validation
    int ret = 0;
    if (validate) {
        bool valid = false;
        if (rank == 0) {
            valid = validateClusters(clusters, points, threshold);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
        int valid_int = valid ? 1 : 0;
        MPI_Bcast(&valid_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
        ret = valid_int ? 0 : 1;
    }

    MPI_Finalize();
    return ret;
}
