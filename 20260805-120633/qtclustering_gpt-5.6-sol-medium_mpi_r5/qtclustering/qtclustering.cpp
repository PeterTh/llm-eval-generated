// QT Clustering Benchmark - distributed-memory MPI version
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
#include <mpi.h>
#include <vector>

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
        // The original expression can produce only zero when N < 30.
        // Keep its random stream unchanged for all inputs that did terminate.
        if (N < 30) group_cnt = 1;
        
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
// while incrementally maintaining each point's diameter to the cluster.
int generateCandidateCluster(const int seed_point,
                              const std::vector<int>& unclustered,
                              const std::vector<Point>& points,
                              const double threshold_squared,
                              std::vector<int>& members,
                              std::vector<int>& candidates,
                              std::vector<double>& diameters) {
    members.clear();
    candidates.clear();
    diameters.clear();
    members.push_back(seed_point);

    candidates.reserve(unclustered.size());
    diameters.reserve(unclustered.size());
    const Point seed = points[seed_point];
    for (const int candidate : unclustered) {
        if (candidate == seed_point) continue;
        const double dx = points[candidate].x - seed.x;
        const double dy = points[candidate].y - seed.y;
        candidates.push_back(candidate);
        diameters.push_back(dx * dx + dy * dy);
    }

    const double infinity = std::numeric_limits<double>::infinity();
    for (;;) {
        double minimum_diameter = threshold_squared;
        int closest_slot = -1;

        // candidates has the same ascending point order as the original scan.
        // A strict comparison therefore also preserves its tie breaking.
        for (size_t slot = 0; slot < candidates.size(); ++slot) {
            if (diameters[slot] < minimum_diameter) {
                minimum_diameter = diameters[slot];
                closest_slot = static_cast<int>(slot);
            }
        }
        if (closest_slot < 0) break;

        const int closest = candidates[closest_slot];
        members.push_back(closest);
        diameters[closest_slot] = infinity;

        const Point added = points[closest];
        for (size_t slot = 0; slot < candidates.size(); ++slot) {
            if (diameters[slot] == infinity) continue;
            const Point candidate = points[candidates[slot]];
            const double dx = candidate.x - added.x;
            const double dy = candidate.y - added.y;
            diameters[slot] = std::max(diameters[slot], dx * dx + dy * dy);
        }
    }

    return static_cast<int>(members.size());
}

// Main QT clustering algorithm. Every rank holds the small replicated state,
// while expensive candidate seeds are distributed cyclically between ranks.
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold,
                                  const int rank,
                                  const int rank_count,
                                  MPI_Comm communicator) {
    const int N = static_cast<int>(points.size());
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;
    std::vector<unsigned char> selected(N, 0);
    const double threshold_squared = threshold * threshold;
    unclustered_indices.reserve(N);
    
    // Initialize unclustered indices
    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }
    
    // Main clustering loop
    while (!unclustered_indices.empty()) {
        int max_cardinality = -1;
        int best_seed = -1;
        std::vector<int> best_cluster_members;
        std::vector<int> candidate_members;
        std::vector<int> candidate_indices;
        std::vector<double> candidate_diameters;
        candidate_members.reserve(unclustered_indices.size());
        candidate_indices.reserve(unclustered_indices.size());
        candidate_diameters.reserve(unclustered_indices.size());
        
        // Cyclic assignment keeps seed counts balanced and remains deterministic.
        for (size_t i = static_cast<size_t>(rank); i < unclustered_indices.size();
             i += static_cast<size_t>(rank_count)) {
            const int seed = unclustered_indices[i];
            const int cardinality = generateCandidateCluster(
                seed, unclustered_indices, points, threshold_squared,
                candidate_members, candidate_indices, candidate_diameters);
            
            if (cardinality > max_cardinality ||
                (cardinality == max_cardinality && seed < best_seed)) {
                max_cardinality = cardinality;
                best_seed = seed;
                best_cluster_members = candidate_members;
            }
        }

        // MPI_MINLOC on {-cardinality, seed} chooses maximum cardinality and,
        // on ties, the earliest seed exactly as the sequential algorithm does.
        struct { int value; int index; } local_choice, global_choice;
        local_choice.value = -max_cardinality;
        local_choice.index = best_seed >= 0 ? best_seed : INT_MAX;
        MPI_Allreduce(&local_choice, &global_choice, 1, MPI_2INT, MPI_MINLOC,
                      communicator);
        max_cardinality = -global_choice.value;
        best_seed = global_choice.index;
        
        // If we found a cluster, add it
        if (best_seed >= 0 && max_cardinality > 0) {
            const auto seed_position = std::lower_bound(
                unclustered_indices.begin(), unclustered_indices.end(), best_seed);
            const int owner = static_cast<int>(
                std::distance(unclustered_indices.begin(), seed_position) % rank_count);
            if (rank != owner) best_cluster_members.resize(max_cardinality);
            MPI_Bcast(best_cluster_members.data(), max_cardinality, MPI_INT, owner,
                      communicator);

            Cluster cluster;
            cluster.seed_point = best_seed;
            cluster.members = best_cluster_members;
            clusters.push_back(cluster);

            // Remove the winner using a temporary byte mask. All ranks receive
            // the same winner and consequently keep identical replicated state.
            for (const int member : best_cluster_members) selected[member] = 1;
            unclustered_indices.erase(
                std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                              [&selected](int idx) { return selected[idx] != 0; }),
                unclustered_indices.end()
            );
            for (const int member : best_cluster_members) selected[member] = 0;
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

    int rank = 0;
    int rank_count = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &rank_count);

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
        printf("MPI ranks: %d\n", rank_count);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Generate once to retain the original data stream, then replicate the
    // read-only points needed by each distributed candidate task.
    std::vector<Point> points(num_points);
    if (rank == 0) generateSyntheticData(points, num_points);
    static_assert(sizeof(Point) == 2 * sizeof(double));
    MPI_Bcast(points.data(), 2 * num_points, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Perform QT clustering
    MPI_Barrier(MPI_COMM_WORLD);
    const double cluster_start = MPI_Wtime();
    const std::vector<Cluster> clusters = qtClustering(
        points, threshold, rank, rank_count, MPI_COMM_WORLD);
    const double local_cluster_time = MPI_Wtime() - cluster_start;
    double cluster_time_sec = 0.0;
    MPI_Reduce(&local_cluster_time, &cluster_time_sec, 1, MPI_DOUBLE, MPI_MAX,
               0, MPI_COMM_WORLD);

    int exit_code = 0;
    if (rank == 0) {
        const long cluster_time_ms = static_cast<long>(cluster_time_sec * 1000.0);
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
        const double clusters_per_sec = clusters.size() / cluster_time_sec;
        const double points_per_sec = num_points / cluster_time_sec;
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
    
        if (validate) {
            const bool valid = validateClusters(clusters, points, threshold);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            exit_code = valid ? 0 : 1;
        }
    }

    MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exit_code;
}
