// QT Clustering Benchmark - MPI Distributed Memory Version
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// The candidate clusters for all unclustered seeds are generated in parallel,
// one seed partition per MPI rank.  Each rank only needs its own copy of the
// point data and the (tiny) clustered-state vector; no point data is
// communicated during the clustering loop.

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

// Generate a candidate cluster starting from a seed point.
//
// The maximum distance from every unclustered point to the cluster is
// maintained incrementally: adding a member only requires one distance
// per remaining candidate instead of recomputing the distance to every
// member.  Squared distances are used for the threshold test, which is
// equivalent and avoids a square root in the innermost loop.
//
// Returns the cardinality (size) of the cluster; members are stored in
// `members` (seed first, then points in the order they were added).
int generateCandidateCluster(const int seed_point,
                             const std::vector<char>& clustered,
                             const std::vector<Point>& points,
                             const std::vector<int>& candidates,
                             const double threshold2,
                             std::vector<char>& in_cluster,
                             std::vector<double>& max_dist2,
                             std::vector<int>& members) {
    (void)clustered;
    const int point_count = static_cast<int>(points.size());

    std::fill(in_cluster.begin(), in_cluster.end(), 0);
    members.clear();
    members.push_back(seed_point);
    in_cluster[seed_point] = 1;

    // Initialize each remaining candidate's maximum squared distance to
    // the cluster, which currently consists only of the seed.
    const Point& seed = points[seed_point];
    for (const int candidate : candidates) {
        if (in_cluster[candidate]) continue;
        const double dx = points[candidate].x - seed.x;
        const double dy = points[candidate].y - seed.y;
        max_dist2[candidate] = dx * dx + dy * dy;
    }

    while (static_cast<int>(members.size()) < point_count) {
        // Find the closest candidate that keeps the diameter below the
        // threshold.  Iterating in ascending point order and using a
        // strict comparison reproduces the sequential tie-break rule.
        int best = -1;
        double best_dist2 = threshold2;

        for (const int candidate : candidates) {
            if (in_cluster[candidate]) continue;
            const double d2 = max_dist2[candidate];
            // d2 >= threshold2 can never become valid again because the
            // maximum distance is monotonically non-decreasing.
            if (d2 < best_dist2) {
                best_dist2 = d2;
                best = candidate;
            }
        }

        if (best < 0) break;

        in_cluster[best] = 1;
        members.push_back(best);

        // Update only candidates that are still able to join.
        const Point& added = points[best];
        for (const int candidate : candidates) {
            if (in_cluster[candidate]) continue;
            if (max_dist2[candidate] >= threshold2) continue;
            const double dx = points[candidate].x - added.x;
            const double dy = points[candidate].y - added.y;
            const double d2 = dx * dx + dy * dy;
            if (d2 > max_dist2[candidate]) {
                max_dist2[candidate] = d2;
            }
        }
    }

    return static_cast<int>(members.size());
}

// Main QT clustering algorithm, parallelised across MPI ranks.
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold,
                                  const int rank,
                                  const int num_ranks) {
    const int N = static_cast<int>(points.size());
    const double threshold2 = threshold * threshold;
    std::vector<char> clustered(N, 0);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;

    unclustered_indices.reserve(N);
    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }

    // Per-rank scratch buffers, reused across candidate generations.
    std::vector<char> in_cluster(N, 0);
    std::vector<double> max_dist2(N, 0.0);
    std::vector<int> members;
    std::vector<int> local_best_members;

    // Main clustering loop
    while (!unclustered_indices.empty()) {
        unsigned long long local_key = 0;
        int local_best_seed = -1;

        // Each round all ranks search a disjoint, strided subset of the
        // unclustered seeds.  The strided assignment keeps the (ordered)
        // seed list balanced even as points are removed.
        const size_t unclustered_count = unclustered_indices.size();
        for (size_t pos = static_cast<size_t>(rank); pos < unclustered_count;
             pos += static_cast<size_t>(num_ranks)) {
            const int seed = unclustered_indices[pos];

            const int cardinality = generateCandidateCluster(
                seed, clustered, points, unclustered_indices, threshold2,
                in_cluster, max_dist2, members);

            // Pack (cardinality, ~seed) so that MPI_MAX selects the
            // largest cluster and, on ties, the smallest seed index.
            const unsigned long long key =
                (static_cast<unsigned long long>(cardinality) << 32) |
                static_cast<unsigned int>(~static_cast<unsigned int>(seed));

            if (key > local_key) {
                local_key = key;
                local_best_seed = seed;
                local_best_members = members;
            }
        }

        unsigned long long global_key = 0;
        MPI_Allreduce(&local_key, &global_key, 1, MPI_UNSIGNED_LONG_LONG,
                      MPI_MAX, MPI_COMM_WORLD);

        const unsigned int low =
            static_cast<unsigned int>(global_key & 0xffffffffull);
        const int best_seed = static_cast<int>(~low);

        // The owner of the winning seed holds its member list.
        const size_t best_pos = static_cast<size_t>(
            std::lower_bound(unclustered_indices.begin(),
                             unclustered_indices.end(), best_seed) -
            unclustered_indices.begin());
        const int owner = static_cast<int>(best_pos % num_ranks);

        int member_count = 0;
        if (rank == owner) {
            // Defensive fallback; ordinarily the owner's local best is
            // exactly the global best.
            if (local_best_seed != best_seed) {
                generateCandidateCluster(best_seed, clustered, points,
                                         unclustered_indices, threshold2,
                                         in_cluster, max_dist2,
                                         local_best_members);
            }
            member_count = static_cast<int>(local_best_members.size());
        }

        MPI_Bcast(&member_count, 1, MPI_INT, owner, MPI_COMM_WORLD);

        std::vector<int> best_members(member_count);
        if (rank == owner) {
            std::copy(local_best_members.begin(), local_best_members.end(),
                      best_members.begin());
        }
        if (member_count > 0) {
            MPI_Bcast(best_members.data(), member_count, MPI_INT, owner,
                      MPI_COMM_WORLD);
        }

        // Add the winning cluster and mark its members as clustered.
        Cluster cluster;
        cluster.seed_point = best_seed;
        cluster.members = std::move(best_members);
        clusters.push_back(std::move(cluster));

        for (const int member : clusters.back().members) {
            clustered[member] = 1;
        }

        // Remove clustered points from the unclustered list (order kept).
        unclustered_indices.erase(
            std::remove_if(unclustered_indices.begin(),
                           unclustered_indices.end(),
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

    int rank = 0;
    int num_ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

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
    }
    
    // Generate synthetic data (deterministic, so every rank gets the same
    // points without any communication).
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);
    
    // Perform QT clustering
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_start = std::chrono::high_resolution_clock::now();
    
    const std::vector<Cluster> clusters =
        qtClustering(points, threshold, rank, num_ranks);
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);
    
    if (rank == 0) {
        printf("Clustering time: %ld ms\n", cluster_time.count());
        printf("Clusters found: %zu\n", clusters.size());
    }
    
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
    int valid = 1;
    if (validate) {
        if (rank == 0) {
            valid = validateClusters(clusters, points, threshold) ? 1 : 0;
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();

    return valid ? 0 : 1;
}
