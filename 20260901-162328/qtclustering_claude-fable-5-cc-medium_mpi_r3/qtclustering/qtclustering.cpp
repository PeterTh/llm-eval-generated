// QT Clustering Benchmark - MPI Parallel Version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// Parallelization: in every round of the main loop each unclustered point
// is tried as a cluster seed. These independent candidate-cluster
// generations are distributed cyclically across MPI ranks. The globally
// best candidate (maximum cardinality, ties broken by smallest seed index,
// exactly as in the sequential code) is selected with an
// MPI_Allreduce(MAXLOC) on an encoded key, and its member list is
// broadcast from the owning rank so all ranks keep identical state.

#include <algorithm>
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

// Generate a candidate cluster starting from a seed point
// Returns the cardinality (size) of the cluster
//
// Equivalent to the original (find the unclustered point whose maximum
// distance to all current members is smallest and below the threshold,
// ties broken by smallest index), but instead of rescanning all members
// for every candidate in every step, each candidate's max distance to the
// cluster is maintained incrementally: adding a member can only raise it,
// and max() over the same distances yields the identical value.
int generateCandidateCluster(const int seed_point,
                              const std::vector<char>& clustered,
                              const std::vector<Point>& points,
                              const double threshold,
                              const int point_count,
                              std::vector<int>* cluster_members = nullptr) {
    std::vector<int> members;
    // Max distance from each candidate to the current cluster members
    std::vector<double> max_dist(point_count, 0.0);
    // Candidates that are clustered, already members, or exceed the threshold
    std::vector<char> excluded(clustered);

    // Add seed point
    excluded[seed_point] = 1;
    members.push_back(seed_point);
    int newest = seed_point;

    // Iteratively add closest points
    while (static_cast<int>(members.size()) < point_count) {
        const Point np = points[newest];
        int closest = -1;
        double min_diameter = std::numeric_limits<double>::max();

        for (int candidate = 0; candidate < point_count; ++candidate) {
            if (excluded[candidate]) continue;

            const double dist = distance(points[candidate], np);
            double md = max_dist[candidate];
            if (dist > md) {
                md = dist;
                max_dist[candidate] = dist;
            }

            if (md >= threshold) {
                // Can never rejoin this cluster; skip in future steps
                excluded[candidate] = 1;
                continue;
            }
            if (md < min_diameter) {
                min_diameter = md;
                closest = candidate;
            }
        }

        if (closest < 0) break; // No more points can be added

        excluded[closest] = 1;
        members.push_back(closest);
        newest = closest;
    }

    // Copy members if requested
    if (cluster_members) {
        *cluster_members = members;
    }

    return static_cast<int>(members.size());
}

// Main QT clustering algorithm (MPI parallel)
//
// All ranks hold the full point set and identical clustering state. Each
// round, the candidate seeds are split cyclically across ranks; the global
// winner is selected via MPI_Allreduce(MAXLOC) and its member list is
// broadcast from the owning rank.
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold,
                                  const int rank,
                                  const int nprocs) {
    const int N = static_cast<int>(points.size());
    std::vector<char> clustered(N, 0);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;

    // Initialize unclustered indices
    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }

    // Main clustering loop
    while (!unclustered_indices.empty()) {
        int local_max_cardinality = -1;
        int local_best_seed = -1;
        std::vector<int> local_best_members;

        // Try this rank's share of unclustered points as seeds
        for (size_t i = rank; i < unclustered_indices.size(); i += nprocs) {
            const int seed = unclustered_indices[i];

            std::vector<int> candidate_members;
            const int cardinality = generateCandidateCluster(seed, clustered, points,
                                                       threshold, N,
                                                       &candidate_members);

            // Sequential order visits seeds in ascending index order, so a
            // later seed only wins with strictly greater cardinality; within
            // a rank the visiting order is also ascending, preserving this.
            if (cardinality > local_max_cardinality) {
                local_max_cardinality = cardinality;
                local_best_seed = seed;
                local_best_members = candidate_members;
            }
        }

        // Encode (cardinality, seed) so that the maximum key corresponds to
        // the sequential winner: highest cardinality, ties broken by the
        // smallest seed index. Seeds are unique, so keys never collide.
        long local_key = -1;
        if (local_best_seed >= 0) {
            local_key = static_cast<long>(local_max_cardinality) * (N + 1L)
                        + (N - local_best_seed);
        }
        struct { long key; int rank; } local_kr, global_kr;
        local_kr.key = local_key;
        local_kr.rank = rank;
        MPI_Allreduce(&local_kr, &global_kr, 1, MPI_LONG_INT, MPI_MAXLOC,
                      MPI_COMM_WORLD);

        // If we found a cluster, add it
        if (global_kr.key >= 0) {
            const int best_seed =
                N - static_cast<int>(global_kr.key % (N + 1L));
            int best_size = (rank == global_kr.rank)
                ? static_cast<int>(local_best_members.size()) : 0;
            MPI_Bcast(&best_size, 1, MPI_INT, global_kr.rank, MPI_COMM_WORLD);

            std::vector<int> best_cluster_members;
            if (rank == global_kr.rank) {
                best_cluster_members = std::move(local_best_members);
            } else {
                best_cluster_members.resize(best_size);
            }
            MPI_Bcast(best_cluster_members.data(), best_size, MPI_INT,
                      global_kr.rank, MPI_COMM_WORLD);

            Cluster cluster;
            cluster.seed_point = best_seed;
            cluster.members = best_cluster_members;
            clusters.push_back(cluster);

            // Mark all members as clustered
            for (size_t i = 0; i < best_cluster_members.size(); ++i) {
                clustered[best_cluster_members[i]] = 1;
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
    MPI_Init(&argc, &argv);

    int rank = 0, nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

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
        printf("MPI processes: %d\n", nprocs);
    }

    // Generate synthetic data (deterministic; identical on all ranks)
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);

    // Perform QT clustering
    MPI_Barrier(MPI_COMM_WORLD);
    const double cluster_start = MPI_Wtime();

    const std::vector<Cluster> clusters = qtClustering(points, threshold,
                                                       rank, nprocs);

    const double cluster_end = MPI_Wtime();
    const double local_cluster_time = cluster_end - cluster_start;
    double global_cluster_time = 0.0;
    MPI_Reduce(&local_cluster_time, &global_cluster_time, 1, MPI_DOUBLE,
               MPI_MAX, 0, MPI_COMM_WORLD);
    const long cluster_time_ms =
        static_cast<long>(global_cluster_time * 1000.0);

    if (rank != 0) {
        MPI_Finalize();
        return 0;
    }

    printf("Clustering time: %ld ms\n", cluster_time_ms);
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
    const double time_sec = cluster_time_ms / 1000.0;
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
            MPI_Finalize();
            return 0;
        } else {
            printf("Validation: FAILED\n");
            MPI_Finalize();
            return 1;
        }
    }

    MPI_Finalize();
    return 0;
}
