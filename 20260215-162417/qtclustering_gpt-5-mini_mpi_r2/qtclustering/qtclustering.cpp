// QT Clustering Benchmark - MPI-parallelized version
// Parallelization strategy: distribute seed evaluations among MPI ranks. Each rank
// computes candidate clusters for assigned seeds using the global "clustered" state.
// The best cluster per iteration is chosen via reductions and then applied consistently
// on all ranks.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include <climits>

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

// Generate synthetic 2D point data in clusters (uses rand_r for reproducibility)
void generateSyntheticData(std::vector<Point>& points, const int N, unsigned int seed = 42) {
    auto frand = [&seed]() mutable { return rand_r(&seed) / static_cast<double>(RAND_MAX); };
    
    const double min_dim = std::min(MAX_WIDTH, MAX_HEIGHT);
    int count = 0;
    
    while (count < N) {
        const double cntr_x = frand() * MAX_WIDTH;
        const double cntr_y = frand() * MAX_HEIGHT;
        const double R = frand() * min_dim / 2.0;
        int group_cnt = static_cast<int>(frand() * (N / 30.0));
        if (group_cnt > (N - count)) group_cnt = N - count;
        
        while (group_cnt > 0) {
            const double sign = (frand() < 0.5) ? -1.0 : 1.0;
            const double r = frand() * R;
            const double dx = (2.0 * frand() - 1.0) * r;
            const double dy = std::sqrt(std::max(0.0, r * r - dx * dx)) * sign;
            const double x = cntr_x + dx;
            const double y = cntr_y + dy;
            
            if (x < 0 || x > MAX_WIDTH || y < 0 || y > MAX_HEIGHT) continue;
            
            points[count] = {x, y};
            count++;
            group_cnt--;
        }
    }
}

// Calculate Euclidean distance between two points
inline double distance(const Point& p1, const Point& p2) {
    const double dx = p1.x - p2.x;
    const double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

// Find the closest unclustered point to the current cluster that maintains diameter < threshold
int findClosestPoint(const std::vector<int>& cluster_members,
                     const std::vector<bool>& clustered,
                     const std::vector<bool>& in_cluster,
                     const std::vector<Point>& points,
                     const double threshold,
                     const int point_count) {
    int closest_point = -1;
    double min_diameter = std::numeric_limits<double>::max();
    
    for (int candidate = 0; candidate < point_count; ++candidate) {
        if (clustered[candidate] || in_cluster[candidate]) continue;
        double max_dist = 0.0;
        for (size_t i = 0; i < cluster_members.size(); ++i) {
            const int member = cluster_members[i];
            const double dist = distance(points[candidate], points[member]);
            if (dist > max_dist) max_dist = dist;
        }
        if (max_dist < threshold && max_dist < min_diameter) {
            min_diameter = max_dist;
            closest_point = candidate;
        }
    }
    return closest_point;
}

// Generate a candidate cluster starting from a seed point
int generateCandidateCluster(const int seed_point,
                              const std::vector<bool>& clustered,
                              const std::vector<Point>& points,
                              const double threshold,
                              const int point_count,
                              std::vector<int>* cluster_members = nullptr) {
    std::vector<bool> in_cluster(point_count, false);
    std::vector<int> members;
    members.reserve(64);
    in_cluster[seed_point] = true;
    members.push_back(seed_point);
    
    while (static_cast<int>(members.size()) < point_count) {
        const int closest = findClosestPoint(members, clustered, in_cluster, points, threshold, point_count);
        if (closest < 0) break;
        in_cluster[closest] = true;
        members.push_back(closest);
    }
    if (cluster_members) *cluster_members = members;
    return static_cast<int>(members.size());
}

// Validation: check that clusters satisfy the QT clustering properties
bool validateClusters(const std::vector<Cluster>& clusters,
                      const std::vector<Point>& points,
                      const double threshold) {
    bool valid = true;
    printf("Validating clusters:\n");
    for (size_t c = 0; c < clusters.size(); ++c) {
        const auto& cluster = clusters[c];
        double max_diameter = 0.0;
        for (size_t i = 0; i < cluster.members.size(); ++i) {
            for (size_t j = i + 1; j < cluster.members.size(); ++j) {
                const double dist = distance(points[cluster.members[i]], points[cluster.members[j]]);
                if (dist > max_diameter) max_diameter = dist;
            }
        }
        if (c < 10) printf("  Cluster %zu: size=%zu, seed=%d, diameter=%.4f\n", c, cluster.members.size(), cluster.seed_point, max_diameter);
        if (max_diameter > threshold * 1.001) {
            printf("ERROR: Cluster %zu has diameter %.4f > threshold %.4f\n", c, max_diameter, threshold);
            valid = false;
        }
    }
    std::vector<int> membership(points.size(), -1);
    for (size_t c = 0; c < clusters.size(); ++c) {
        for (size_t i = 0; i < clusters[c].members.size(); ++i) {
            const int member = clusters[c].members[i];
            if (membership[member] >= 0) {
                printf("ERROR: Point %d appears in multiple clusters (%d and %zu)\n", member, membership[member], c);
                valid = false;
            }
            membership[member] = static_cast<int>(c);
        }
    }
    int clustered_count = 0;
    for (size_t i = 0; i < membership.size(); ++i) if (membership[i] >= 0) clustered_count++;
    printf("Total points: %zu, Clustered: %d, Unclustered: %zu\n", points.size(), clustered_count, points.size() - clustered_count);
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
    int rank = 0, comm_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    // Parse arguments (only rank 0 prints help/errors)
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
        if (rank == 0) printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n", num_points, threshold);
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("QT Clustering Benchmark (MPI)\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Prepare points: generate on rank 0 and broadcast to all ranks
    std::vector<Point> points(num_points);
    if (rank == 0) {
        generateSyntheticData(points, num_points);
    }
    // Pack coordinates as contiguous array for broadcast
    std::vector<double> coords;
    coords.resize(num_points * 2);
    if (rank == 0) {
        for (int i = 0; i < num_points; ++i) { coords[2*i] = points[i].x; coords[2*i+1] = points[i].y; }
    }
    MPI_Bcast(coords.data(), num_points * 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank != 0) {
        for (int i = 0; i < num_points; ++i) { points[i].x = coords[2*i]; points[i].y = coords[2*i+1]; }
    }

    // All ranks maintain same clustered state and local cluster results
    std::vector<bool> clustered(num_points, false);
    std::vector<Cluster> clusters;
    clusters.reserve(128);

    // Synchronize and start timing on all ranks
    MPI_Barrier(MPI_COMM_WORLD);
    const auto tstart = std::chrono::high_resolution_clock::now();

    // Main distributed clustering loop
    while (true) {
        // Count total unclustered points across all ranks
        int local_unclustered = 0;
        for (int i = 0; i < num_points; ++i) if (!clustered[i]) local_unclustered++;
        int total_unclustered = 0;
        MPI_Allreduce(&local_unclustered, &total_unclustered, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        if (total_unclustered == 0) break;

        // Each rank evaluates seeds assigned by (seed % comm_size == rank)
        int local_best_card = -1;
        int local_best_seed = -1;
        for (int seed = 0; seed < num_points; ++seed) {
            if (clustered[seed]) continue;
            if ((seed % comm_size) != rank) continue;
            int card = generateCandidateCluster(seed, clustered, points, threshold, num_points, nullptr);
            if (card > local_best_card) { local_best_card = card; local_best_seed = seed; }
        }

        // Find global maximum cardinality
        int global_best_card = 0;
        MPI_Allreduce(&local_best_card, &global_best_card, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        if (global_best_card <= 0) break; // no cluster can be formed

        // Among ranks that achieved global_best_card choose smallest seed id (deterministic)
        int local_seed_for_min = (local_best_card == global_best_card) ? local_best_seed : INT_MAX;
        int global_best_seed = INT_MAX;
        MPI_Allreduce(&local_seed_for_min, &global_best_seed, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (global_best_seed == INT_MAX) break; // should not happen

        // All ranks generate the chosen cluster members for global_best_seed
        std::vector<int> chosen_members;
        generateCandidateCluster(global_best_seed, clustered, points, threshold, num_points, &chosen_members);

        // Apply clustering update
        Cluster c;
        c.seed_point = global_best_seed;
        c.members = chosen_members;
        clusters.push_back(c);
        for (int m : chosen_members) clustered[m] = true;

        // Continue loop until no unclustered points left
    }

    // End timing and report the maximum per-rank elapsed time
    const auto tend = std::chrono::high_resolution_clock::now();
    const long long local_cluster_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(tend - tstart).count();
    long long cluster_time_ms = 0;
    MPI_Reduce(&local_cluster_time_ms, &cluster_time_ms, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Clustering time: %lld ms\n", cluster_time_ms);
        printf("Clusters found: %zu\n", clusters.size());

        int total_clustered = 0; int max_cluster_size = 0;
        for (size_t i = 0; i < clusters.size(); ++i) {
            const int s = static_cast<int>(clusters[i].members.size());
            total_clustered += s; if (s > max_cluster_size) max_cluster_size = s;
        }
        const double avg_cluster_size = clusters.empty() ? 0.0 : static_cast<double>(total_clustered) / clusters.size();
        printf("Points clustered: %d / %d (%.1f%%)\n", total_clustered, num_points, 100.0 * total_clustered / num_points);
        printf("Average cluster size: %.2f\n", avg_cluster_size);
        printf("Maximum cluster size: %d\n", max_cluster_size);
        double time_sec = cluster_time_ms / 1000.0;
        double clusters_per_sec = clusters.size() / (time_sec > 0.0 ? time_sec : 1.0);
        double points_per_sec = num_points / (time_sec > 0.0 ? time_sec : 1.0);
        printf("Performance: %.1f clusters/s, %.1f points/s\n", clusters_per_sec, points_per_sec);

        if (printResults) {
            std::vector<double> membershipData; membershipData.reserve(num_points);
            std::vector<int> membership(num_points, -1);
            for (size_t ci = 0; ci < clusters.size(); ++ci) for (int m : clusters[ci].members) membership[m] = static_cast<int>(ci);
            for (int m : membership) membershipData.push_back(static_cast<double>(m));
            print_results(membershipData, "ClusterMembership");
        }

        if (validate) {
            const bool valid = validateClusters(clusters, points, threshold);
            if (valid) { printf("Validation: PASSED\n"); }
            else { printf("Validation: FAILED\n"); }
        }
    }

    MPI_Finalize();
    return 0;
}
