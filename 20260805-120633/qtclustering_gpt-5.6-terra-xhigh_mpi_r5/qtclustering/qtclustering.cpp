// QT Clustering Benchmark - MPI distributed-memory implementation
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.

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
static_assert(sizeof(Point) == 2 * sizeof(double),
              "Point must be representable as two consecutive MPI_DOUBLE values");

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
                     const std::vector<unsigned char>& clustered,
                     const std::vector<int>& in_cluster_stamp,
                     const int current_stamp,
                     const std::vector<Point>& points,
                     const double threshold,
                     const int point_count) {
    int closest_point = -1;
    double min_diameter = std::numeric_limits<double>::max();
    
    // Try each unclustered point as a candidate
    for (int candidate = 0; candidate < point_count; ++candidate) {
        // Skip if already clustered or already in this cluster
        if (clustered[candidate] || in_cluster_stamp[candidate] == current_stamp) continue;
        
        // Calculate the maximum distance from candidate to all cluster members
        double max_dist = 0.0;
        bool can_improve = true;
        for (size_t i = 0; i < cluster_members.size(); ++i) {
            const int member = cluster_members[i];
            const double dist = distance(points[candidate], points[member]);
            max_dist = std::max(max_dist, dist);

            // Diameter can only grow.  Once this candidate cannot satisfy
            // either selection condition, the remaining distances cannot
            // change the result.
            if (max_dist >= threshold || max_dist >= min_diameter) {
                can_improve = false;
                break;
            }
        }
        
        // If adding this point keeps diameter below threshold and is better than current best
        if (can_improve) {
            min_diameter = max_dist;
            closest_point = candidate;
        }
    }
    
    return closest_point;
}

// Generate a candidate cluster starting from a seed point
// Returns the cardinality (size) of the cluster
int generateCandidateCluster(const int seed_point,
                             const std::vector<unsigned char>& clustered,
                             const std::vector<Point>& points,
                             const double threshold,
                             const int point_count,
                             std::vector<int>& in_cluster_stamp,
                             int& next_stamp,
                             std::vector<int>& members) {
    // A generation stamp avoids clearing an O(N) bitmap for every seed.
    if (next_stamp == std::numeric_limits<int>::max()) {
        std::fill(in_cluster_stamp.begin(), in_cluster_stamp.end(), 0);
        next_stamp = 1;
    } else {
        ++next_stamp;
    }
    const int current_stamp = next_stamp;
    members.clear();
    
    // Add seed point
    in_cluster_stamp[seed_point] = current_stamp;
    members.push_back(seed_point);
    
    // Iteratively add closest points
    while (static_cast<int>(members.size()) < point_count) {
        // Find closest point to current cluster that maintains diameter < threshold
        const int closest = findClosestPoint(members, clustered, in_cluster_stamp,
                                             current_stamp, points, threshold, point_count);
        
        if (closest < 0) break; // No more points can be added
        
        in_cluster_stamp[closest] = current_stamp;
        members.push_back(closest);
    }

    return static_cast<int>(members.size());
}

// MPI QT clustering.  Every rank has the point coordinates, but each rank
// evaluates only a cyclic subset of seed points.  The collective choice of a
// winner is deterministic: maximum cardinality, then lowest point index.
std::vector<Cluster> qtClusteringMPI(const std::vector<Point>& points,
                                     const double threshold,
                                     MPI_Comm communicator) {
    int rank = 0;
    int world_size = 1;
    MPI_Comm_rank(communicator, &rank);
    MPI_Comm_size(communicator, &world_size);

    const int N = static_cast<int>(points.size());
    std::vector<unsigned char> clustered(N, 0);
    std::vector<Cluster> clusters;
    if (rank == 0) {
        clusters.reserve(N);
    }

    // Scratch storage is local to a rank and reused for all of its seeds.
    std::vector<int> in_cluster_stamp(N, 0);
    std::vector<int> candidate_members;
    std::vector<int> local_best_members;
    candidate_members.reserve(N);
    local_best_members.reserve(N);
    int next_stamp = 0;
    int remaining_points = N;

    // Main clustering loop
    while (remaining_points > 0) {
        int local_max_cardinality = 0;
        int local_best_seed = N;

        // Cyclic assignment balances the contiguous synthetic data groups
        // much better than contiguous seed ranges as clustering progresses.
        for (int seed = rank; seed < N; seed += world_size) {
            if (clustered[seed]) continue;

            const int cardinality = generateCandidateCluster(
                seed, clustered, points, threshold, N, in_cluster_stamp, next_stamp,
                candidate_members);

            if (cardinality > local_max_cardinality ||
                (cardinality == local_max_cardinality && seed < local_best_seed)) {
                local_max_cardinality = cardinality;
                local_best_seed = seed;
                local_best_members = candidate_members;
            }
        }

        int max_cardinality = 0;
        MPI_Allreduce(&local_max_cardinality, &max_cardinality, 1, MPI_INT, MPI_MAX,
                      communicator);

        // A second reduction resolves cardinality ties exactly as the
        // sequential increasing-index seed traversal does.
        const int proposed_seed = local_max_cardinality == max_cardinality
                                      ? local_best_seed
                                      : N;
        int best_seed = N;
        MPI_Allreduce(&proposed_seed, &best_seed, 1, MPI_INT, MPI_MIN, communicator);

        if (max_cardinality <= 0 || best_seed == N) {
            break;
        }

        const int owner = best_seed % world_size;
        int member_count = rank == owner ? static_cast<int>(local_best_members.size()) : 0;
        MPI_Bcast(&member_count, 1, MPI_INT, owner, communicator);

        // Reuse the local-best buffer as the receive buffer.  Its capacity is
        // retained from round to round, avoiding a per-cluster allocation.
        if (rank != owner) {
            local_best_members.resize(member_count);
        }
        MPI_Bcast(local_best_members.data(), member_count, MPI_INT, owner, communicator);

        if (rank == 0) {
            Cluster cluster;
            cluster.seed_point = best_seed;
            cluster.members = local_best_members;
            clusters.push_back(cluster);
        }

        for (const int member : local_best_members) {
            clustered[member] = 1;
        }
        remaining_points -= member_count;
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
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

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
            if (rank == 0) {
                printUsage(argv[0]);
            }
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
    
    // Generate once, then make the globally indexed point set available to
    // every rank.  Seed-level work is distributed below.
    std::vector<Point> points(num_points);
    if (rank == 0) {
        generateSyntheticData(points, num_points);
    }
    MPI_Bcast(points.data(), 2 * num_points, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Perform QT clustering
    MPI_Barrier(MPI_COMM_WORLD);
    const double cluster_start = MPI_Wtime();
    const std::vector<Cluster> clusters = qtClusteringMPI(points, threshold, MPI_COMM_WORLD);
    const double local_cluster_time = MPI_Wtime() - cluster_start;
    double cluster_time = 0.0;
    MPI_Reduce(&local_cluster_time, &cluster_time, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Clustering time: %.0f ms\n", cluster_time * 1000.0);
        printf("Clusters found: %zu\n", clusters.size());
    }
    
    // Calculate statistics and performance metrics
    int exit_code = 0;
    if (rank == 0) {
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
               total_clustered, num_points, 100.0 * total_clustered / num_points);
        printf("Average cluster size: %.2f\n", avg_cluster_size);
        printf("Maximum cluster size: %d\n", max_cluster_size);

        // Performance metrics use the slowest rank's elapsed time.
        const double clusters_per_sec = clusters.size() / cluster_time;
        const double points_per_sec = num_points / cluster_time;
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

    MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exit_code;
}
