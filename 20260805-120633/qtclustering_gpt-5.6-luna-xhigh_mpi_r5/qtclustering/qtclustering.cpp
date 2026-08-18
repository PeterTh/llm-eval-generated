// QT Clustering Benchmark - MPI distributed-memory version
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.

#include <algorithm>
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
        // For N <= 30 the original expression is always zero; keep the
        // original stream for normal inputs while making small test cases
        // terminate and produce one point per generated group.
        if (N <= 30) {
            group_cnt = std::max(1, group_cnt);
        }
        
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

// Squared Euclidean distance is sufficient for all candidate comparisons:
// sqrt is monotonic for non-negative values, so this preserves both the
// threshold test and the closest-candidate ordering without repeated sqrt.
inline double distanceSquared(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return dx * dx + dy * dy;
}

inline double distance(const Point& p1, const Point& p2) {
    return std::sqrt(distanceSquared(p1, p2));
}

// Generate a candidate cluster starting from a seed point
// Returns the cardinality (size) of the cluster
int generateCandidateCluster(const int seed_point,
                              const std::vector<unsigned char>& clustered,
                              const std::vector<Point>& points,
                              const double threshold_squared,
                              const int point_count,
                              std::vector<int>& members,
                              std::vector<unsigned char>& in_cluster,
                              std::vector<double>& max_distances,
                              const unsigned char cluster_mark) {
    members.clear();
    members.push_back(seed_point);
    in_cluster[seed_point] = cluster_mark;

    // Initialize each candidate's distance from the seed. Subsequent
    // iterations only need one new distance per candidate.
    for (int candidate = 0; candidate < point_count; ++candidate) {
        if (clustered[candidate] || in_cluster[candidate] == cluster_mark) continue;
        max_distances[candidate] = distanceSquared(points[candidate], points[seed_point]);
    }
    
    // Iteratively add closest points
    while (static_cast<int>(members.size()) < point_count) {
        int closest = -1;
        double min_diameter_squared = std::numeric_limits<double>::max();

        // Candidates are scanned in index order, matching the original
        // tie-breaking rule. max_distances contains the exact maximum over
        // the current member list.
        for (int candidate = 0; candidate < point_count; ++candidate) {
            if (clustered[candidate] || in_cluster[candidate] == cluster_mark) continue;

            const double max_dist_squared = max_distances[candidate];
            if (max_dist_squared < threshold_squared &&
                max_dist_squared < min_diameter_squared) {
                min_diameter_squared = max_dist_squared;
                closest = candidate;
            }
        }
        
        if (closest < 0) break; // No more points can be added
        
        in_cluster[closest] = cluster_mark;
        members.push_back(closest);

        // Add the newly selected member to every remaining candidate's
        // running diameter. This changes the candidate construction from a
        // repeated O(N * |cluster|) rescan into an O(N) update per member.
        for (int candidate = 0; candidate < point_count; ++candidate) {
            if (clustered[candidate] || in_cluster[candidate] == cluster_mark) continue;
            const double dist_squared = distanceSquared(points[candidate], points[closest]);
            max_distances[candidate] = std::max(max_distances[candidate], dist_squared);
        }
    }
    
    return static_cast<int>(members.size());
}

// Main QT clustering algorithm
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold,
                                  const int rank,
                                  const int process_count) {
    const int N = static_cast<int>(points.size());
    const double threshold_squared = threshold * threshold;
    std::vector<unsigned char> clustered(N, 0);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;
    
    // Initialize unclustered indices
    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }
    
    // Main clustering loop
    // These scratch buffers are reused for every seed. The mark array avoids
    // clearing an O(N) in-cluster vector for every candidate seed.
    std::vector<unsigned char> in_cluster(N, 0);
    std::vector<double> max_distances(N, 0.0);
    std::vector<int> candidate_members;
    std::vector<int> best_local_members;
    unsigned char cluster_mark = 1;

    while (!unclustered_indices.empty()) {
        int local_best_cardinality = -1;
        int local_best_seed = INT_MAX;
        best_local_members.clear();

        // Cyclic assignment spreads spatially correlated and therefore
        // potentially expensive seeds across ranks.
        for (size_t i = static_cast<size_t>(rank);
             i < unclustered_indices.size();
             i += static_cast<size_t>(process_count)) {
            const int seed = unclustered_indices[i];

            // Each seed needs a distinct marker. Reuse the compact byte
            // array and clear it only after all 254 usable markers expire.
            if (cluster_mark == UCHAR_MAX) {
                std::fill(in_cluster.begin(), in_cluster.end(), 0);
                cluster_mark = 1;
            }
            const int cardinality = generateCandidateCluster(
                seed, clustered, points, threshold_squared, N,
                candidate_members, in_cluster, max_distances, cluster_mark);
            ++cluster_mark;

            // Strict comparison retains the original first-seed tie break.
            if (cardinality > local_best_cardinality) {
                local_best_cardinality = cardinality;
                local_best_seed = seed;
                best_local_members.swap(candidate_members);
            }
        }

        // MPI_MAXLOC selects the largest cardinality and, on a tie, the
        // smallest seed index, exactly matching the sequential scan.
        const int local_best[2] = {local_best_cardinality, local_best_seed};
        int global_best[2] = {-1, INT_MAX};
        MPI_Allreduce(local_best, global_best, 1, MPI_2INT, MPI_MAXLOC,
                      MPI_COMM_WORLD);

        const int best_seed = global_best[1];
        const int max_cardinality = global_best[0];
        if (best_seed < 0 || max_cardinality <= 0) break;

        // The cyclic assignment makes the owner derivable by the seed's
        // position in the shared, sorted active-seed list.
        const auto best_position = std::lower_bound(
            unclustered_indices.begin(), unclustered_indices.end(), best_seed);
        const int winner_rank = static_cast<int>(
            std::distance(unclustered_indices.begin(), best_position)) % process_count;

        int winning_size = (rank == winner_rank)
            ? static_cast<int>(best_local_members.size()) : 0;
        MPI_Bcast(&winning_size, 1, MPI_INT, winner_rank, MPI_COMM_WORLD);

        std::vector<int> winning_members(static_cast<size_t>(winning_size));
        if (rank == winner_rank) {
            winning_members = best_local_members;
        }
        if (winning_size > 0) {
            MPI_Bcast(winning_members.data(), winning_size, MPI_INT,
                      winner_rank, MPI_COMM_WORLD);
        }

        // All ranks update the same state from the one broadcast winner.
        if (rank == 0) {
            Cluster cluster;
            cluster.seed_point = best_seed;
            cluster.members = winning_members;
            clusters.push_back(std::move(cluster));
        }
        for (const int member : winning_members) {
            clustered[member] = 1;
        }

        unclustered_indices.erase(
            std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                           [&clustered](const int index) { return clustered[index] != 0; }),
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
    int process_count = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &process_count);

    int num_points = 1000;
    double threshold = 2.0;
    bool validate = false;
    bool printResults = false;
    bool show_help = false;
    bool parse_error = false;
    
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
            show_help = true;
        } else {
            parse_error = true;
        }
    }

    if (show_help || parse_error) {
        if (rank == 0) {
            if (parse_error) {
                printf("Unknown or incomplete command-line option\n");
            }
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return parse_error ? 1 : 0;
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
    
    // Generate synthetic data
    std::vector<Point> points(num_points);
    if (rank == 0) {
        generateSyntheticData(points, num_points);
    }
    MPI_Bcast(points.data(), 2 * num_points, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Perform QT clustering
    MPI_Barrier(MPI_COMM_WORLD);
    const double cluster_start = MPI_Wtime();
    
    const std::vector<Cluster> clusters = qtClustering(
        points, threshold, rank, process_count);
    
    const double local_cluster_seconds = MPI_Wtime() - cluster_start;
    double cluster_seconds = 0.0;
    MPI_Reduce(&local_cluster_seconds, &cluster_seconds, 1, MPI_DOUBLE,
               MPI_MAX, 0, MPI_COMM_WORLD);
    long cluster_time_ms = 0;
    if (rank == 0) {
        cluster_time_ms = static_cast<long>(cluster_seconds * 1000.0);
    }
    MPI_Bcast(&cluster_time_ms, 1, MPI_LONG, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
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
    }
    
    // Validation
    int valid = 1;
    if (validate) {
        if (rank == 0) {
            valid = validateClusters(clusters, points, threshold) ? 1 : 0;
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (valid != 0) {
            if (rank == 0) {
            printf("Validation: PASSED\n");
            }
        } else {
            if (rank == 0) {
                printf("Validation: FAILED\n");
            }
        }
    }

    MPI_Finalize();
    return valid == 0 ? 1 : 0;
}
