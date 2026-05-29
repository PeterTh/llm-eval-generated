// QT Clustering Benchmark - MPI Parallel Version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// Parallelization strategy:
//   - Distribute seed-point evaluations across MPI ranks each iteration
//   - Each rank independently generates candidate clusters for its seeds
//   - Allreduce to find the global best seed (max cardinality)
//   - Broadcast the winning cluster members to all ranks
//   - Synchronize the clustered[] array via Allreduce (int flags)
//   - Repeat until all points are clustered

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

// Generate synthetic 2D point data in clusters (identical on all ranks)
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

// Main QT clustering algorithm - MPI parallelized
// Parallelizes the seed-evaluation loop across MPI ranks
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold,
                                  const int rank, const int num_ranks) {
    const int N = static_cast<int>(points.size());
    std::vector<bool> clustered(N, false);
    std::vector<Cluster> clusters;

    // Main clustering loop
    while (true) {
        // Build list of unclustered indices (local copy on each rank)
        std::vector<int> unclustered_indices;
        unclustered_indices.reserve(N);
        for (int i = 0; i < N; ++i) {
            if (!clustered[i]) unclustered_indices.push_back(i);
        }

        if (unclustered_indices.empty()) break;

        const int num_seeds = static_cast<int>(unclustered_indices.size());

        // Distribute seeds across ranks: rank r gets seeds [start, start+count)
        const int base_count = num_seeds / num_ranks;
        const int remainder = num_seeds % num_ranks;
        const int my_count = base_count + (rank < remainder ? 1 : 0);
        const int my_start = rank * base_count + std::min(rank, remainder);

        // Evaluate candidate clusters for my seeds
        int local_max_cardinality = -1;
        int local_best_seed = -1;
        std::vector<int> local_best_members;

        for (int i = 0; i < my_count; ++i) {
            const int seed = unclustered_indices[my_start + i];

            std::vector<int> candidate_members;
            const int cardinality = generateCandidateCluster(seed, clustered, points,
                                                             threshold, N,
                                                             &candidate_members);

            if (cardinality > local_max_cardinality) {
                local_max_cardinality = cardinality;
                local_best_seed = seed;
                local_best_members = std::move(candidate_members);
            }
        }

        // Global reduction: find best seed across all ranks
        int global_max_cardinality, global_best_seed;

        // Allreduce for max cardinality
        MPI_Allreduce(&local_max_cardinality, &global_max_cardinality, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

        if (global_max_cardinality <= 0) break; // No more clusters can be formed

        // Ranks matching the global max contribute their seed; others contribute INT_MAX
        int my_seed_for_reduce = (local_max_cardinality == global_max_cardinality)
                                     ? local_best_seed
                                     : std::numeric_limits<int>::max();
        // Among ties, pick lowest seed index for determinism
        MPI_Allreduce(&my_seed_for_reduce, &global_best_seed, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

        // Determine member count on the winning rank
        int local_members_count = 0;
        if (local_best_seed == global_best_seed && local_max_cardinality == global_max_cardinality) {
            local_members_count = static_cast<int>(local_best_members.size());
        }
        int global_members_count;
        MPI_Allreduce(&local_members_count, &global_members_count, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

        // Find the rank of the winner for broadcasting
        int my_rank_if_winner = (local_best_seed == global_best_seed &&
                                  local_max_cardinality == global_max_cardinality)
                                    ? rank
                                    : num_ranks;
        int root_rank;
        MPI_Allreduce(&my_rank_if_winner, &root_rank, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

        // Broadcast the winning cluster members from the winning rank
        std::vector<int> best_cluster_members(global_members_count);
        if (rank == root_rank) {
            best_cluster_members = local_best_members;
        }
        MPI_Bcast(best_cluster_members.data(), global_members_count, MPI_INT, root_rank, MPI_COMM_WORLD);

        // Add the cluster
        Cluster cluster;
        cluster.seed_point = global_best_seed;
        cluster.members = std::move(best_cluster_members);
        clusters.push_back(cluster);

        // Mark members as clustered
        for (int m : clusters.back().members) {
            clustered[m] = true;
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
    printf("Usage: mpirun -np <procs> %s [options]\n", progName);
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
    int rank, num_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    // Parse command line arguments (all ranks parse the same args)
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
        printf("MPI ranks: %d\n", num_ranks);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Generate synthetic data (identical on all ranks)
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);

    // Perform QT clustering
    auto cluster_start = std::chrono::high_resolution_clock::now();

    const std::vector<Cluster> clusters = qtClustering(points, threshold, rank, num_ranks);

    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);

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

    if (rank == 0) {
        printf("Clustering time: %ld ms\n", cluster_time.count());
        printf("Clusters found: %zu\n", clusters.size());
        printf("Points clustered: %d / %d (%.1f%%)\n",
               total_clustered, num_points,
               100.0 * total_clustered / num_points);
        printf("Average cluster size: %.2f\n", avg_cluster_size);
        printf("Maximum cluster size: %d\n", max_cluster_size);

        // Performance metrics
        const double time_sec = cluster_time.count() / 1000.0;
        const double clusters_per_sec = time_sec > 0.0 ? clusters.size() / time_sec : 0.0;
        const double points_per_sec = time_sec > 0.0 ? num_points / time_sec : 0.0;
        printf("Performance: %.1f clusters/s, %.1f points/s\n",
               clusters_per_sec, points_per_sec);
    }

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
        if (rank == 0) {
            print_results(membershipData, "ClusterMembership");
        }
    }

    // Validation
    if (validate) {
        bool valid = true;
        if (rank == 0) {
            valid = validateClusters(clusters, points, threshold);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }

        int valid_int = valid ? 0 : 1;
        int global_valid;
        MPI_Allreduce(&valid_int, &global_valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

        MPI_Finalize();
        return global_valid;
    }

    MPI_Finalize();
    return 0;
}
