// QT Clustering Benchmark - MPI Parallel Version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// Parallelization: in each round, the candidate-cluster generation for the
// unclustered seed points is distributed cyclically across MPI ranks. The
// globally best candidate cluster is selected with MPI_MAXLOC (which breaks
// ties on the lowest seed position, matching the sequential iteration order)
// and its member list is broadcast so all ranks keep identical state.

#include <algorithm>
#include <chrono>
#include <climits>
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
// Instead of recomputing the max distance from each candidate to all cluster
// members in every iteration, a running maximum per candidate is maintained
// and updated with the distance to the newly added member. Since max() over
// doubles is exact and order-independent, this yields the identical result
// to the original O(n) rescan per candidate.
int generateCandidateCluster(const int seed_point,
                              const std::vector<char>& clustered,
                              const std::vector<Point>& points,
                              const double threshold,
                              const int point_count,
                              std::vector<int>* cluster_members = nullptr) {
    std::vector<int> members;
    members.push_back(seed_point);

    // max_dist[c]: max distance from candidate c to all current cluster members
    // (initialized with the distance to the seed point)
    std::vector<double> max_dist(point_count);
    const Point seed = points[seed_point];
    for (int c = 0; c < point_count; ++c) {
        max_dist[c] = distance(points[c], seed);
    }

    std::vector<char> in_cluster(point_count, 0);
    in_cluster[seed_point] = 1;

    // Iteratively add closest points
    while (static_cast<int>(members.size()) < point_count) {
        // Find the unclustered point whose max distance to all cluster
        // members (i.e. the resulting diameter) is smallest and < threshold
        int closest = -1;
        double min_diameter = std::numeric_limits<double>::max();

        for (int c = 0; c < point_count; ++c) {
            if (clustered[c] || in_cluster[c]) continue;
            const double d = max_dist[c];
            if (d < threshold && d < min_diameter) {
                min_diameter = d;
                closest = c;
            }
        }

        if (closest < 0) break; // No more points can be added

        in_cluster[closest] = 1;
        members.push_back(closest);

        // Update running max distances with the newly added member
        const Point added = points[closest];
        for (int c = 0; c < point_count; ++c) {
            const double d = distance(points[c], added);
            if (d > max_dist[c]) max_dist[c] = d;
        }
    }

    // Copy members if requested
    if (cluster_members) {
        *cluster_members = members;
    }

    return static_cast<int>(members.size());
}

// Main QT clustering algorithm (MPI parallel)
// All ranks hold the full point set and identical clustering state; the seed
// evaluation loop of each round is split cyclically across ranks.
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold,
                                  const int mpi_rank,
                                  const int mpi_size) {
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
        int local_best_pos = INT_MAX; // position within unclustered_indices
        std::vector<int> local_best_members;

        // Try each unclustered point assigned to this rank as a seed
        for (size_t i = mpi_rank; i < unclustered_indices.size();
             i += static_cast<size_t>(mpi_size)) {
            const int seed = unclustered_indices[i];

            std::vector<int> candidate_members;
            const int cardinality = generateCandidateCluster(seed, clustered, points,
                                                       threshold, N,
                                                       &candidate_members);

            if (cardinality > local_max_cardinality) {
                local_max_cardinality = cardinality;
                local_best_pos = static_cast<int>(i);
                local_best_members = std::move(candidate_members);
            }
        }

        // Select the globally best candidate. MPI_MAXLOC picks the maximum
        // cardinality and, on ties, the minimum position — the same seed the
        // sequential loop (strict '>' in iteration order) would select.
        struct { long card; int pos; } local_sel, global_sel;
        local_sel.card = local_max_cardinality;
        local_sel.pos = local_best_pos;
        MPI_Allreduce(&local_sel, &global_sel, 1, MPI_LONG_INT,
                      MPI_MAXLOC, MPI_COMM_WORLD);

        const int max_cardinality = static_cast<int>(global_sel.card);

        // If we found a cluster, add it
        if (global_sel.pos != INT_MAX && max_cardinality > 0) {
            const int winner_rank = static_cast<int>(global_sel.pos % mpi_size);
            const int best_seed = unclustered_indices[global_sel.pos];

            // Broadcast the winning cluster's member list to all ranks
            std::vector<int> best_cluster_members;
            if (mpi_rank == winner_rank) {
                best_cluster_members = std::move(local_best_members);
            } else {
                best_cluster_members.resize(max_cardinality);
            }
            MPI_Bcast(best_cluster_members.data(), max_cardinality, MPI_INT,
                      winner_rank, MPI_COMM_WORLD);

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

    int mpi_rank = 0, mpi_size = 1;
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
            if (mpi_rank == 0) printUsage(argv[0]);
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
        printf("MPI ranks: %d\n", mpi_size);
    }

    // Generate synthetic data (deterministic, so every rank generates the
    // identical point set locally without communication)
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);

    // Perform QT clustering
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_start = std::chrono::high_resolution_clock::now();

    const std::vector<Cluster> clusters = qtClustering(points, threshold,
                                                       mpi_rank, mpi_size);

    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);
    const long local_ms = cluster_time.count();
    long max_ms = 0;
    MPI_Reduce(&local_ms, &max_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    int exit_code = 0;

    if (mpi_rank == 0) {
        printf("Clustering time: %ld ms\n", max_ms);
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
        const double time_sec = max_ms / 1000.0;
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
                exit_code = 1;
            }
        }
    }

    // All ranks return the same exit code
    MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Finalize();
    return exit_code;
}
