// QT Clustering Benchmark - MPI Parallel Version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// Parallelization strategy:
// - All ranks share the full point dataset
// - In each clustering iteration, unclustered seed candidates are distributed
//   across ranks; each rank independently generates candidate clusters for
//   its assigned seeds
// - MPI_Allreduce selects the globally best cluster (max cardinality,
//   lowest seed index for ties)
// - All ranks synchronize on the selected cluster and update state
// - This preserves exact sequential semantics while parallelizing the
//   expensive candidate generation phase

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

// MPI-parallel QT clustering algorithm
// Preserves exact sequential semantics by using MPI collectives to
// coordinate the best-cluster selection across all ranks
std::vector<Cluster> qtClusteringMPI(const std::vector<Point>& points,
                                     const double threshold,
                                     int rank, int num_ranks) {
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
        // Each rank finds its local best cluster from its assigned seeds
        int local_max_cardinality = -1;
        int local_best_seed = -1;
        std::vector<int> local_best_members;

        // Distribute unclustered seed candidates across ranks
        const int total_seeds = static_cast<int>(unclustered_indices.size());
        // Compute range of seeds this rank should evaluate
        int start_idx = (total_seeds * rank) / num_ranks;
        int end_idx = (total_seeds * (rank + 1)) / num_ranks;

        for (int i = start_idx; i < end_idx; ++i) {
            const int seed = unclustered_indices[i];
            if (clustered[seed]) continue;

            std::vector<int> candidate_members;
            const int cardinality = generateCandidateCluster(seed, clustered, points,
                                                             threshold, N,
                                                             &candidate_members);

            // Use the same selection criteria as sequential: max cardinality,
            // lowest seed index for ties
            if (cardinality > local_max_cardinality ||
                (cardinality == local_max_cardinality && (local_best_seed < 0 || seed < local_best_seed))) {
                local_max_cardinality = cardinality;
                local_best_seed = seed;
                local_best_members = std::move(candidate_members);
            }
        }

        // Pack local best result for MPI_Allreduce
        // We need to find the global best: max cardinality, then min seed index for ties
        // Use a comparison key: pack as (cardinality, -seed) so max works correctly
        // For MPI_Allreduce we send (cardinality, seed) and use a custom comparator
        // via MPI_Allreduce with MAX on a packed value

        // Pack: high 32 bits = cardinality, low 32 bits = (~seed) so that
        // MPI_MAX gives us max cardinality, and for ties, max(~seed) = min(seed)
        uint64_t local_packed = 0;
        if (local_max_cardinality > 0) {
            local_packed = (static_cast<uint64_t>(local_max_cardinality) << 32) |
                           (static_cast<uint64_t>(~static_cast<uint32_t>(local_best_seed)));
        }

        uint64_t global_packed = 0;
        MPI_Allreduce(&local_packed, &global_packed, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);

        // Unpack global best
        int global_max_cardinality = static_cast<int>(global_packed >> 32);
        int global_best_seed = -1;
        if (global_max_cardinality > 0) {
            global_best_seed = static_cast<int>(~static_cast<uint32_t>(global_packed & 0xFFFFFFFF));
        }

        // Broadcast the best cluster members from the rank that found it
        // We need to find which rank has the best members to broadcast
        // Use a two-step approach: first find the source rank, then broadcast

        // Determine if this rank has the best result
        bool is_source = (local_max_cardinality == global_max_cardinality &&
                          local_best_seed == global_best_seed &&
                          global_max_cardinality > 0);

        // Find minimum rank with best result using MPI_Allreduce
        int my_rank_if_source = is_source ? rank : num_ranks;
        int source = 0;
        MPI_Allreduce(&my_rank_if_source, &source, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (source >= num_ranks) source = 0; // safety, shouldn't happen

        // Broadcast cluster members from source rank
        // First broadcast the size
        int members_size = 0;
        if (is_source) {
            members_size = static_cast<int>(local_best_members.size());
        }
        MPI_Bcast(&members_size, 1, MPI_INT, source, MPI_COMM_WORLD);

        // Broadcast members
        std::vector<int> global_best_members;
        if (members_size > 0) {
            global_best_members.resize(members_size);
            if (is_source) {
                global_best_members = local_best_members;
            }
            MPI_Bcast(global_best_members.data(), members_size, MPI_INT, source, MPI_COMM_WORLD);
        }

        // All ranks now have the global best cluster; add it
        if (global_max_cardinality > 0 && global_best_seed >= 0) {
            Cluster cluster;
            cluster.seed_point = global_best_seed;
            cluster.members = global_best_members;
            clusters.push_back(cluster);

            // Mark all members as clustered
            for (size_t i = 0; i < global_best_members.size(); ++i) {
                clustered[global_best_members[i]] = true;
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
    int num_ranks, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    int num_points = 1000;
    double threshold = 2.0;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (only on rank 0, then broadcast)
    if (rank == 0) {
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
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                MPI_Finalize();
                return 1;
            }
        }

        if (num_points <= 0 || threshold <= 0.0) {
            printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
                   num_points, threshold);
            MPI_Finalize();
            return 1;
        }
    }

    // Broadcast parameters to all ranks
    MPI_Bcast(&num_points, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&threshold, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Print header only on rank 0
    if (rank == 0) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("MPI ranks: %d\n", num_ranks);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Generate synthetic data on rank 0, then broadcast to all ranks
    // This ensures all ranks have identical data
    std::vector<Point> points(num_points);
    if (rank == 0) {
        generateSyntheticData(points, num_points);
    }
    MPI_Bcast(points.data(), num_points * 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Perform QT clustering (MPI parallel)
    auto cluster_start = std::chrono::high_resolution_clock::now();

    const std::vector<Cluster> clusters = qtClusteringMPI(points, threshold, rank, num_ranks);

    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);
    const long long local_cluster_time_ms = cluster_time.count();
    long long global_cluster_time_ms = 0;
    MPI_Reduce(&local_cluster_time_ms, &global_cluster_time_ms, 1,
               MPI_LONG_LONG_INT, MPI_MAX, 0, MPI_COMM_WORLD);

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

    // Print results only on rank 0
    if (rank == 0) {
        printf("Clustering time: %lld ms\n", global_cluster_time_ms);
        printf("Clusters found: %zu\n", clusters.size());

        printf("Points clustered: %d / %d (%.1f%%)\n",
               total_clustered, num_points,
               100.0 * total_clustered / num_points);
        printf("Average cluster size: %.2f\n", avg_cluster_size);
        printf("Maximum cluster size: %d\n", max_cluster_size);

        // Performance metrics
        const double time_sec = global_cluster_time_ms / 1000.0;
        const double clusters_per_sec = clusters.size() / time_sec;
        const double points_per_sec = num_points / time_sec;
        printf("Performance: %.1f clusters/s, %.1f points/s\n",
               clusters_per_sec, points_per_sec);
    }

    // Print results for external validation (rank 0 only)
    if (rank == 0 && printResults) {
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

    // Validation (rank 0 only)
    if (rank == 0 && validate) {
        const bool valid = validateClusters(clusters, points, threshold);

        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
        }
    }

    MPI_Finalize();
    return 0;
}
