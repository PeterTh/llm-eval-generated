// QT Clustering Benchmark - Simplified Sequential Version
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
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
        // For N < 30 the original scale factor would make every group empty
        // and never terminate.  Keep the original random stream for normal
        // benchmark sizes while making the small-input case well-defined.
        if (N < 30 && group_cnt == 0) group_cnt = 1;
        
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
int generateCandidateCluster(const int seed_point,
                              const std::vector<int>& unclustered_indices,
                              const std::vector<Point>& points,
                              const double threshold,
                              std::vector<double>& candidate_max_distances,
                              std::vector<std::uint32_t>& in_cluster_marks,
                              const std::uint32_t cluster_mark,
                              std::vector<int>& members) {
    members.clear();

    // Add seed point
    in_cluster_marks[seed_point] = cluster_mark;
    members.push_back(seed_point);

    // Keep the current maximum distance for each possible candidate.  The
    // original implementation recomputed these maxima from scratch for every
    // iteration.  Updating them when a member is added performs the same
    // ordered comparisons while avoiding a large amount of repeated work and
    // allocation.
    for (const int candidate : unclustered_indices) {
        if (in_cluster_marks[candidate] == cluster_mark) continue;
        candidate_max_distances[candidate] = std::max(
            0.0, distance(points[candidate], points[seed_point]));
    }

    // Iteratively add the closest point.  unclustered_indices is sorted, so
    // the strict comparison preserves the original lowest-index tie break.
    while (members.size() < unclustered_indices.size()) {
        int closest = -1;
        double min_diameter = std::numeric_limits<double>::max();

        for (const int candidate : unclustered_indices) {
            if (in_cluster_marks[candidate] == cluster_mark) continue;

            const double max_dist = candidate_max_distances[candidate];
            if (max_dist < threshold && max_dist < min_diameter) {
                min_diameter = max_dist;
                closest = candidate;
            }
        }

        if (closest < 0) break; // No more points can be added

        in_cluster_marks[closest] = cluster_mark;
        members.push_back(closest);

        // A candidate whose current maximum is already outside the threshold
        // can never become eligible, since adding members only increases its
        // maximum distance.
        for (const int candidate : unclustered_indices) {
            if (in_cluster_marks[candidate] == cluster_mark ||
                candidate_max_distances[candidate] >= threshold) {
                continue;
            }
            candidate_max_distances[candidate] = std::max(
                candidate_max_distances[candidate],
                distance(points[candidate], points[closest]));
        }
    }

    return static_cast<int>(members.size());
}

// Main QT clustering algorithm
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold,
                                  const int rank,
                                  const int process_count,
                                  MPI_Comm communicator) {
    const int N = static_cast<int>(points.size());
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;

    // Initialize unclustered indices
    unclustered_indices.resize(N);
    std::iota(unclustered_indices.begin(), unclustered_indices.end(), 0);

    // These scratch arrays are reused for every seed on this rank.  A mark
    // array avoids constructing and clearing a vector<bool> for every trial.
    std::vector<double> candidate_max_distances(N);
    std::vector<std::uint32_t> in_cluster_marks(N, 0);
    std::uint32_t next_cluster_mark = 0;

    // Main clustering loop
    while (!unclustered_indices.empty()) {
        int local_max_cardinality = -1;
        int local_best_seed = -1;
        std::vector<int> local_best_cluster_members;
        std::vector<int> candidate_members;

        // Try an interleaved subset of the available seeds.  Interleaving
        // spreads spatially correlated and therefore similarly expensive
        // seeds over ranks better than a contiguous block.
        for (size_t i = static_cast<size_t>(rank); i < unclustered_indices.size();
             i += static_cast<size_t>(process_count)) {
            ++next_cluster_mark;
            if (next_cluster_mark == 0) {
                std::fill(in_cluster_marks.begin(), in_cluster_marks.end(), 0);
                next_cluster_mark = 1;
            }

            const int seed = unclustered_indices[i];
            const int cardinality = generateCandidateCluster(
                seed, unclustered_indices, points, threshold,
                candidate_max_distances, in_cluster_marks, next_cluster_mark,
                candidate_members);

            if (cardinality > local_max_cardinality ||
                (cardinality == local_max_cardinality &&
                 (local_best_seed < 0 || seed < local_best_seed))) {
                local_max_cardinality = cardinality;
                local_best_seed = seed;
                local_best_cluster_members = candidate_members;
            }
        }

        // MPI_MAXLOC gives the largest cardinality and, on a tie, the
        // smallest location.  Using the seed as the location therefore
        // preserves the sequential loop's first-wins tie break.
        int local_choice[2] = {
            local_max_cardinality,
            local_best_seed >= 0 ? local_best_seed : std::numeric_limits<int>::max()
        };
        int global_choice[2] = {0, 0};
        MPI_Allreduce(local_choice, global_choice, 1, MPI_2INT, MPI_MAXLOC,
                      communicator);

        const int best_seed = global_choice[1];
        if (global_choice[0] > 0 && best_seed >= 0) {
            // Seeds are assigned by their position in this sorted list, so
            // the owner of the winning seed is known without another
            // collective.  Broadcast only the selected cluster, rather than
            // gathering all local candidates or replicating trial work.
            const auto best_position = static_cast<size_t>(std::lower_bound(
                unclustered_indices.begin(), unclustered_indices.end(), best_seed) -
                unclustered_indices.begin());
            const int owner = static_cast<int>(best_position %
                                               static_cast<size_t>(process_count));
            int best_cluster_size = owner == rank
                ? static_cast<int>(local_best_cluster_members.size()) : 0;
            MPI_Bcast(&best_cluster_size, 1, MPI_INT, owner, communicator);

            std::vector<int> best_cluster_members;
            if (owner == rank) {
                best_cluster_members = local_best_cluster_members;
            } else {
                best_cluster_members.resize(best_cluster_size);
            }
            MPI_Bcast(best_cluster_members.data(), best_cluster_size, MPI_INT,
                      owner, communicator);

            Cluster cluster;
            cluster.seed_point = best_seed;
            cluster.members = best_cluster_members;
            clusters.push_back(cluster);

            // Remove clustered points from the sorted available list.
            std::vector<int> sorted_cluster_members = best_cluster_members;
            std::sort(sorted_cluster_members.begin(), sorted_cluster_members.end());
            unclustered_indices.erase(
                std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                              [&sorted_cluster_members](const int index) {
                                  return std::binary_search(sorted_cluster_members.begin(),
                                                            sorted_cluster_members.end(), index);
                              }),
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

    int rank = 0;
    int process_count = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &process_count);

    int num_points = 1000;
    double threshold = 2.0;
    bool validate = false;
    bool printResults = false;
    bool showHelp = false;
    bool parseError = false;
    
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
            showHelp = true;
        } else {
            parseError = true;
        }
    }

    if (showHelp) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return 0;
    }

    if (parseError || num_points <= 0 || threshold <= 0.0) {
        if (rank == 0) {
            if (parseError) {
                printf("Unknown or malformed option\n");
                printUsage(argv[0]);
            } else {
                printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
                       num_points, threshold);
            }
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
        points, threshold, rank, process_count, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    long cluster_time_ms = static_cast<long>((MPI_Wtime() - cluster_start) * 1000.0);
    MPI_Allreduce(MPI_IN_PLACE, &cluster_time_ms, 1, MPI_LONG, MPI_MAX,
                  MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Clustering time: %ld ms\n", cluster_time_ms);
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
    }
    
    // Performance metrics
    const double time_sec = cluster_time_ms / 1000.0;
    const double clusters_per_sec = clusters.size() / time_sec;
    const double points_per_sec = num_points / time_sec;
    if (rank == 0) {
        printf("Performance: %.1f clusters/s, %.1f points/s\n",
               clusters_per_sec, points_per_sec);
    }
    
    // Print results for external validation
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
    
    // Validation
    if (rank == 0 && validate) {
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
